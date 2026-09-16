import hydra
import pyrootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import List
import lightning as L
from lightning import LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from amelia_tf import utils
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"

log = utils.get_pylogger(__name__)


def load_model_state(model, ckpt_path: str):
    """Load model weights from a Lightning checkpoint (score_head stays random-init)."""
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        for pre in ("net.", "mode_model.net.", "traj_model.net."):
            if k.startswith(pre):
                k = k[len(pre):]
                break
        cleaned_state_dict[k] = v
    result = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"[LOAD] missing keys: {result.missing_keys}")
    print(f"[LOAD] unexpected keys: {result.unexpected_keys}")
    return model


def _build_nets(cfg):
    """Instantiate + load the frozen mode/traj nets onto device."""
    log.info("Instantiating networks...")
    mode_net = hydra.utils.instantiate(cfg.model.mode_net)
    traj_net = hydra.utils.instantiate(cfg.model.traj_net)
    mode_net = load_model_state(mode_net, cfg.mode_ckpt_path)
    traj_net = load_model_state(traj_net, cfg.traj_ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_net.to(device)
    traj_net.to(device)
    return mode_net, traj_net, device


def _build_model_and_trainer(cfg, mode_net, traj_net):
    """Build CombinedTrajPredSystem + trainer."""
    log.info("Building combined model...")
    model = CombinedTrajPredSystem(
        mode_model=mode_net,
        traj_model=traj_net,
        extra_params=cfg.model.extra_params,
    )
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    log.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        default_root_dir=str(Path(cfg.paths.output_dir) / "test"),
    )
    return model, trainer


def _run_test(cfg, model, trainer, datamodule, tag: str):
    """Run trainer.test and save metrics json under output_dir/<tag>."""
    log.info(f"Starting testing ({tag})...")
    results = trainer.test(model=model, datamodule=datamodule)
    out_dir = Path(cfg.paths.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(results[0] if results else {}, f, indent=2)
    log.info(f"Results saved to: {out_dir / 'test_metrics.json'}")
    return results


def _stage_eval(cfg, datamodule):
    """Standard combined-system test (selection_mode from config)."""
    mode_net, traj_net, _ = _build_nets(cfg)
    model, trainer = _build_model_and_trainer(cfg, mode_net, traj_net)
    _run_test(cfg, model, trainer, datamodule, tag="test")


# ----------------------------------------------------------------------
# Score-head stage: train the GMM score head on VAL (frozen backbone),
# save it, then TEST with score-based hypothesis selection.
# ----------------------------------------------------------------------

def _find_gmm(traj_net):
    from amelia_tf.models.components.gmm import GMM  # adjust path if needed
    for m in traj_net.modules():
        if isinstance(m, GMM):
            return m
    raise RuntimeError("GMM head not found in traj_net; fix import in _find_gmm.")


def _to_device(batch, device):
    sd = batch['scene_dict']
    for k, v in sd.items():
        if torch.is_tensor(v):
            sd[k] = v.to(device)
    return batch


_SCORER_ATTRS = ('score_head', 'score_heads', 'q_proj', 'k_proj', 'q_projs', 'k_projs')


def _scorer_modules(gmm):
    """
    Return [(attr_name, module), ...] for the submodules that make up the
    scoring head, covering both head types:
      MLP       -> score_head (shared) or score_heads (per-mode)
      attention -> q_proj/k_proj (shared) or q_projs/k_projs (per-mode)

    Looked up as attributes on the GMM object rather than by matching parameter
    NAMES: transformer blocks in the backbone commonly name their own
    projections q_proj/k_proj too, so a name match would silently unfreeze
    backbone weights.
    """
    mods = []
    for attr in _SCORER_ATTRS:
        m = getattr(gmm, attr, None)
        if m is not None:
            mods.append((attr, m))
    return mods


def _scorer_param_ids(gmm):
    """Set of id()s of every parameter belonging to the scoring head."""
    ids = set()
    for _, m in _scorer_modules(gmm):
        for p in m.parameters():
            ids.add(id(p))
    return ids


def _freeze_bn(net):
    """Keep frozen norm layers in eval mode during train().

    The scoring heads are Linear/GELU stacks with no norm layers of their own,
    so freezing every norm layer in the net is equivalent to excluding the
    scorer explicitly, and works for both head types.
    """
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.eval()


def _encode_true_mode(batch, ego_agent):
    """Return the ego per-sample true mode index (B,) from rule_based_encoding."""
    Y_mode = batch['scene_dict'].get('rule_based_encoding')
    if Y_mode is None:
        return None
    tmi = Y_mode[..., :4].float().argmax(dim=-1).long()          # (B,A)
    tmi_ego = separate_ego_agent(tmi, ego_agent).reshape(-1)      # (B,)
    return tmi_ego


def _score_step(batch, traj_net, mode_net, hist_len, num_modes, train, opt, route='true'):
    """
    Train/eval the score head on one batch; label = winner (min-ADE candidate).

    Masking is ALWAYS applied: the winner (min-ADE candidate) and the score
    aggregation are computed only over VALID future timesteps, and samples whose
    ego future is fully masked are skipped. This is required once runway / z
    masking is active, otherwise padded/invalid points corrupt winner & score.

    `route` selects which samples each mode's head is trained on:
      - 'true' : only samples whose TRUE mode is m (clean, focused signal; B)
      - 'pred' : only samples whose PREDICTED mode is m (from mode_net; C)
      - 'all'  : all samples forced into mode m (original scheme; A)
    """
    Y = batch['scene_dict']['rel_sequences'][..., :4]
    X = torch.zeros_like(Y).float()
    X[:, :, :hist_len] = Y[:, :, :hist_len]
    context = batch['scene_dict']['context']
    adjacency = batch['scene_dict']['adjacency']
    ego_agent = batch['scene_dict']['ego_agent_id']
    Yxyz = Y[..., G.REL_XYZ[:4]]
    ego_fut = separate_ego_agent(Yxyz[:, :, hist_len:, :], ego_agent)

    # ego future mask (valid timesteps) -- ALWAYS used
    masks = batch['scene_dict']['agent_masks']
    ego_mask = separate_ego_agent(masks, ego_agent)              # (B,1,T) or (B,1,T,1)
    fut_mask = ego_mask[:, :, hist_len:]                         # (B,1,Tp[,1])
    fut_mask = fut_mask.reshape(fut_mask.shape[0], 1, -1).float()  # (B,1,Tp)

    true_mode = _encode_true_mode(batch, ego_agent)   # (B,) or None

    # predicted mode from the mode prediction network (route='pred')
    pred_mode = None
    if route == 'pred' and mode_net is not None:
        feasibility = batch['scene_dict'].get('feasibility', None)
        with torch.no_grad():
            mode_logits = mode_net(X[:, :, :, :4], context=context,
                                   adjacency=adjacency, mask=None,
                                   feasibility=feasibility, output_mode_only=True)
            ego_logits = separate_ego_agent(mode_logits, ego_agent)   # (B,1,M)
            pred_mode = ego_logits.reshape(-1, ego_logits.shape[-1]).argmax(-1)  # (B,)

    total_loss = 0.0
    per_mode_match = [0] * num_modes
    per_mode_cnt = [0] * num_modes
    B, A = X.shape[:2]
    _gmm = _find_gmm(traj_net)
    n_used_modes = 0

    # per-sample valid: at least one valid future timestep
    valid_sample = (fut_mask.sum(dim=2).reshape(-1) > 0)         # (B,)

    for m in range(num_modes):
        if _gmm is not None:
            _gmm._active_mode = m
        one_hot = F.one_hot(torch.full((B, A), m, dtype=torch.long, device=X.device),
                            num_classes=num_modes).float()
        out = traj_net(X[:, :, :, :4], context=context, adjacency=adjacency,
                       mask=None, mode_probs=one_hot)
        mu, sigma, score = out  # score head enabled
        ego_mu = separate_ego_agent(mu, ego_agent)
        ego_score = separate_ego_agent(score, ego_agent)
        mu_fut = ego_mu[:, :, hist_len:, :, :2]                  # (B,1,Tp,K,2)

        m_exp = fut_mask[..., None]                             # (B,1,Tp,1)
        denom = m_exp.sum(dim=2).clamp_min(1)                   # (B,1,1)
        with torch.no_grad():
            # mask-weighted ADE per candidate -> winner
            d_pt = ((mu_fut - ego_fut[:, :, :, None, :2]) ** 2).sum(-1)  # (B,1,Tp,K)
            d = (d_pt * m_exp).sum(dim=2) / denom               # (B,1,K)
            winner = d.argmin(-1).squeeze(1)                    # (B,)
        # mask-weighted score aggregation
        s_pt = ego_score[:, :, hist_len:]                       # (B,1,Tp,K)
        s = (s_pt * m_exp).sum(dim=2) / denom                   # (B,1,K)
        s = s.squeeze(1)                                        # (B,K)

        # --- routing: which samples this head is trained on ---
        if route == 'true' and true_mode is not None:
            sel = (true_mode == m)
        elif route == 'pred' and pred_mode is not None:
            sel = (pred_mode == m)
        else:  # 'all' (or fallback if mode info missing)
            sel = torch.ones(B, dtype=torch.bool, device=s.device)
        sel = sel & valid_sample                                # always drop fully-masked
        n_sel = int(sel.sum().item())
        if n_sel == 0:
            continue

        s_m = s[sel]
        winner_m = winner[sel]
        loss = F.cross_entropy(s_m, winner_m)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
        n_used_modes += 1
        per_mode_match[m] += (s_m.argmax(-1) == winner_m).sum().item()
        per_mode_cnt[m] += n_sel
    return total_loss / max(n_used_modes, 1), per_mode_match, per_mode_cnt


def _stage_score(cfg, datamodule):
    """
    Train the score head (frozen backbone, score_detach=True), save it, then
    TEST with score-based hypothesis selection. Strict separation from test.

    Two data sources, selected by cfg.scorer.data_source:
      'val_split'  (default, unchanged behaviour): train and hold out on the
          VAL split, 4/5 vs 1/5 by batch index. The backbone never saw val,
          so the candidate distribution the scorer trains on matches the one
          it will face at test time.
      'train_val': train on the TRAIN split, hold out on the whole VAL split.
          Gives far more data (especially for the rare turning modes), but
          the backbone WAS trained on that split, so the candidates it
          generates there are systematically closer to the ground truth than
          on unseen data -- the scorer may learn a discrimination rule that
          does not transfer. Watch for train_match rising while
          HOLDOUT_match stalls or drops, which is exactly that failure.
    """
    data_source = cfg.scorer.get("data_source", "val_split")

    # The datamodule only builds data_train when task_name == "train" (see
    # DataModule.setup), and its setup() is guarded so it will not rebuild
    # once any split exists -- so this has to be set BEFORE setup() runs.
    if data_source == "train_val" and getattr(datamodule, "task_name", None) != "train":
        log.info("[score] data_source='train_val' requires the train split; "
                 "setting datamodule.task_name='train' before setup()")
        datamodule.task_name = "train"

    datamodule.prepare_data(); datamodule.setup()
    val_loader = datamodule.val_dataloader()
    if datamodule.data_val is None:
        raise ValueError("data_val is None; ensure the datamodule builds the val split.")

    if data_source == "train_val":
        if getattr(datamodule, "data_train", None) is None:
            raise ValueError("data_train is None; ensure the datamodule builds the train split.")
        train_loader = datamodule.train_dataloader()
        ho_loader = val_loader
        keep_tr = lambda bi: True          # use every train batch
        keep_ho = lambda bi: True          # use every val batch as holdout
        log.info(f"[score] data source = 'train_val': training on TRAIN split "
                 f"({len(datamodule.data_train)} samples), holding out on the "
                 f"full VAL split ({len(datamodule.data_val)} samples)")
        log.info("[score] NOTE: the backbone was trained on this split; if "
                 "train_match rises while HOLDOUT_match does not, the scorer "
                 "is fitting candidate patterns specific to the train split.")
    else:
        train_loader = val_loader
        ho_loader = val_loader
        keep_tr = lambda bi: (bi % 5 != 0)
        keep_ho = lambda bi: (bi % 5 == 0)
        log.info("[score] data source = 'val_split': training on 4/5 of VAL, "
                 "holding out on the remaining 1/5")

    mode_net, traj_net, device = _build_nets(cfg)
    mode_net.eval(); traj_net.eval()

    gmm = _find_gmm(traj_net)
    assert getattr(gmm, "enable_score_head", False) and gmm.score_mode != 0, \
        "Set enable_score_head=true and score_mode=4 in the traj_net GMM config."
    log.info(f"[score] score_mode={gmm.score_mode} detach={gmm.score_detach} "
             f"hidden={gmm.score_hidden} head_type={getattr(gmm, 'score_head_type', 'mlp')} "
             f"per_mode={getattr(gmm, 'per_mode_score', False)}")

    for p in mode_net.parameters():
        p.requires_grad = False

    scorer_ids = _scorer_param_ids(gmm)
    if not scorer_ids:
        raise ValueError(
            "No scoring-head parameters found on the GMM. Expected one of "
            f"{_SCORER_ATTRS} to exist -- check enable_score_head and "
            "score_head_type in the traj_net GMM config.")
    for p in traj_net.parameters():
        p.requires_grad = (id(p) in scorer_ids)
    trainable = [p for p in traj_net.parameters() if p.requires_grad]
    log.info(f"[score] training {', '.join(a for a, _ in _scorer_modules(gmm))} only: "
             f"{sum(p.numel() for p in trainable)} params")

    bb_name, bb_ref = next((n, p) for n, p in traj_net.named_parameters()
                           if id(p) not in scorer_ids and p.dim() > 1)
    bb_snapshot = bb_ref.detach().clone()
    log.info(f"[score] watching backbone tensor '{bb_name}' for accidental updates")

    opt = torch.optim.AdamW(trainable, lr=cfg.scorer.get('lr', 1e-3),
                            weight_decay=cfg.scorer.get('weight_decay', 1e-4))

    hist_len = traj_net.hist_len
    num_modes = traj_net.num_modes
    mode_names = getattr(traj_net, 'mode_names',
                         [f'mode{m}' for m in range(num_modes)])

    n_epochs = cfg.scorer.get("epochs", 5)
    route = cfg.scorer.get("score_route", "true")   # 'true' | 'pred' | 'all'
    log.info(f"[score] training route = '{route}' (mask always applied)")
    for ep in range(n_epochs):
        traj_net.train(); _freeze_bn(traj_net)
        trl, nb = 0.0, 0
        tr_match = [0] * num_modes
        tr_cnt = [0] * num_modes
        for bi, batch in enumerate(train_loader):
            if not keep_tr(bi):
                continue
            batch = _to_device(batch, device)
            l, pm_match, pm_cnt = _score_step(batch, traj_net, mode_net, hist_len, num_modes, True, opt, route=route)
            trl += l; nb += 1
            for m in range(num_modes):
                tr_match[m] += pm_match[m]; tr_cnt[m] += pm_cnt[m]
        traj_net.eval()
        ho_match = [0] * num_modes
        ho_cnt = [0] * num_modes
        with torch.no_grad():
            for bi, batch in enumerate(ho_loader):
                if not keep_ho(bi):
                    continue
                batch = _to_device(batch, device)
                _, pm_match, pm_cnt = _score_step(batch, traj_net, mode_net, hist_len, num_modes, False, None, route=route)
                for m in range(num_modes):
                    ho_match[m] += pm_match[m]; ho_cnt[m] += pm_cnt[m]
        tr_tot = sum(tr_match); tr_n = sum(tr_cnt)
        ho_tot = sum(ho_match); ho_n = sum(ho_cnt)
        log.info(f"[score] epoch {ep}: train_loss={trl/max(nb,1):.4f} "
                 f"train_match={tr_tot/max(tr_n,1):.1%} HOLDOUT_match={ho_tot/max(ho_n,1):.1%}")
        # per-mode holdout accuracy (this is what reveals the turn improvement)
        for m in range(num_modes):
            name = mode_names[m] if m < len(mode_names) else f'mode{m}'
            log.info(f"[score]   {name}: train={tr_match[m]/max(tr_cnt[m],1):.1%} "
                     f"HOLDOUT={ho_match[m]/max(ho_cnt[m],1):.1%} (n={ho_cnt[m]})")

        drift = (bb_ref.detach() - bb_snapshot).abs().max().item()
        if drift > 0:
            log.error(f"[score] *** BACKBONE CHANGED *** '{bb_name}' drifted {drift:.3e}")
        else:
            log.info(f"[score]   backbone intact (drift=0)")

    save_path = cfg.scorer.get("score_head_save", None)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Keyed by attribute name so the file is self-describing and works for
        # both head types (MLP: score_head/score_heads; attention: q_proj/k_proj
        # or q_projs/k_projs). Legacy files holding a bare state_dict for the
        # MLP head are still readable -- see the loader in _stage_score_test.
        blob = {attr: m.state_dict() for attr, m in _scorer_modules(gmm)}
        torch.save(blob, save_path)
        log.info(f"[score] saved {', '.join(blob.keys())} -> {save_path}")

    traj_net.eval()
    # test doesn't need deterministic algorithms; disabling avoids the CuBLAS
    # deterministic-mode error on Linear layers (CUDA >= 10.2).
    # torch.use_deterministic_algorithms(False)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.model.extra_params.selection_mode = 'score'
    model, trainer = _build_model_and_trainer(cfg, mode_net, traj_net)
    _run_test(cfg, model, trainer, datamodule, tag="test_score")


def _stage_score_test(cfg, datamodule):
    """
    TEST ONLY with a pre-trained score head, on a CLEAN backbone.

    This avoids any chance that score-head training contaminated the backbone:
    we reload the original traj checkpoint (pristine), then load ONLY the saved
    score_head weights on top, then test. Use this if the all-in-one 'score'
    stage produced broken metrics (a sign the backbone was modified in-place).
    """
    datamodule.prepare_data(); datamodule.setup()
    mode_net, traj_net, device = _build_nets(cfg)   # fresh load of original ckpts
    mode_net.eval(); traj_net.eval()

    gmm = _find_gmm(traj_net)
    assert getattr(gmm, "enable_score_head", False) and gmm.score_mode != 0, \
        "Set enable_score_head=true and score_mode=4 in the traj_net GMM config."

    # load saved score head onto the pristine backbone
    sh_path = cfg.scorer.get("score_head_load", cfg.scorer.get("score_head_save"))
    blob = torch.load(sh_path, map_location=device)

    # New format: {attr_name: state_dict} covering whichever submodules the
    # configured head type uses. Legacy format: a bare state_dict for the MLP
    # head, saved before the attention head existed.
    if isinstance(blob, dict) and set(blob.keys()) <= set(_SCORER_ATTRS):
        for attr, sd in blob.items():
            mod = getattr(gmm, attr, None)
            if mod is None:
                raise ValueError(
                    f"Checkpoint contains '{attr}' but the current GMM config "
                    f"does not build it (head_type="
                    f"{getattr(gmm, 'score_head_type', 'mlp')}, per_mode="
                    f"{getattr(gmm, 'per_mode_score', False)}). The scorer "
                    "config must match the one used at training time.")
            mod.load_state_dict(sd)
        log.info(f"[score_test] loaded {', '.join(blob.keys())} <- {sh_path}")
    elif getattr(gmm, "per_mode_score", False):
        gmm.score_heads.load_state_dict(blob)
        log.info(f"[score_test] loaded per-mode score_heads <- {sh_path}")
    else:
        gmm.score_head.load_state_dict(blob)
        log.info(f"[score_test] loaded score_head <- {sh_path}")

    # sanity: backbone must be pristine -> check a sigma stat on one batch
    # torch.use_deterministic_algorithms(False)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.model.extra_params.selection_mode = 'score'
    model, trainer = _build_model_and_trainer(cfg, mode_net, traj_net)
    _run_test(cfg, model, trainer, datamodule, tag="test_score")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    stage = cfg.get("scorer", {}).get("stage", "eval") if cfg.get("scorer") else "eval"
    log.info(f"Run stage: {stage}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    if stage == "score":
        _stage_score(cfg, datamodule)
    elif stage == "score_test":
        _stage_score_test(cfg, datamodule)
    else:
        _stage_eval(cfg, datamodule)


if __name__ == "__main__":
    main()