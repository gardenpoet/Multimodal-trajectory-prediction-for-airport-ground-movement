"""
PROBE EXPERIMENT — does the model's internal hidden embedding contain more
"which candidate is best" information than handcrafted geometric features?

What it does:
  - Loads the frozen pretrained mode+traj models.
  - Runs the traj GMM head with enable_score_head=True, score_mode=2 (hidden h).
  - FREEZES everything except score_head.
  - Trains score_head ONLY, with label = winner (the candidate with min ADE to GT).
  - Reports the score head's match (how often its argmax == winner).

Compare the resulting match to the closed-book learned tree (~75%):
  - match >> 75%  -> hidden embedding has real extra info; the embedding idea
                     (score_mode=4) is worth pursuing.
  - match ~ 75%   -> hidden is information-equivalent to handcrafted features;
                     the embedding idea won't beat the tree. Stop.

This does NOT touch the trajectory backbone (score_detach=True + frozen params),
so the trajectory predictions are unchanged; we only probe selectability.

USAGE (single GPU is plenty, score head is tiny):
  python probe_score_head.py     # uses the same hydra config as eval_two_stage

Notes:
  - Uses the VAL split (task_name=train builds it), to match the scorer's
    leakage-free protocol. Internal 80/20 holdout for honest match.
  - Trains for a few epochs over val; the head is small so this is fast.
"""
import hydra
import pyrootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from amelia_tf import utils
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G

log = utils.get_pylogger(__name__)


def load_model_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("net."):
            nk = k[len("net."):]
        elif k.startswith("mode_model.net."):
            nk = k[len("mode_model.net."):]
        elif k.startswith("traj_model.net."):
            nk = k[len("traj_model.net."):]
        else:
            nk = k
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)  # score_head stays random-init
    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_two_stage")
def main(cfg):
    utils.extras(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- datamodule (val split) ----
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data(); dm.setup()
    val_loader = dm.val_dataloader()

    # ---- models ----
    mode_net = hydra.utils.instantiate(cfg.model.mode_net)
    traj_net = hydra.utils.instantiate(cfg.model.traj_net)
    mode_net = load_model_state(mode_net, cfg.mode_ckpt_path)
    traj_net = load_model_state(traj_net, cfg.traj_ckpt_path)
    mode_net.to(device).eval()
    traj_net.to(device).eval()

    # sanity: score head must exist & be enabled
    gmm = _find_gmm(traj_net)
    assert getattr(gmm, "enable_score_head", False), \
        "Set enable_score_head=true and score_mode=2 in the traj_net GMM config."
    log.info(f"[probe] score_mode={gmm.score_mode} score_detach={gmm.score_detach} "
             f"score_hidden={gmm.score_hidden}")

    # ---- FREEZE everything except score_head ----
    for p in mode_net.parameters():
        p.requires_grad = False
    for n, p in traj_net.named_parameters():
        p.requires_grad = ('score_head' in n)
    trainable = [p for p in traj_net.parameters() if p.requires_grad]
    log.info(f"[probe] trainable params (score_head only): "
             f"{sum(p.numel() for p in trainable)}")

    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

    hist_len = traj_net.hist_len
    num_modes = traj_net.num_modes

    # ---- train/holdout split: every 5th batch is HELD OUT (never trained on).
    #      This gives an honest generalization match to compare with the
    #      closed-book tree's 75% (which was measured on a held-out split). ----
    def is_holdout(bi):
        return (bi % 5 == 0)

    n_epochs = cfg.get("probe_epochs", 5)
    for ep in range(n_epochs):
        # ---- train pass (skip holdout batches) ----
        tr_loss, tr_corr, tr_tot, nb = 0.0, 0, 0, 0
        for bi, batch in enumerate(val_loader):
            if is_holdout(bi):
                continue
            batch = _to_device(batch, device)
            loss, mn, cnt = _step(batch, mode_net, traj_net, gmm,
                                  hist_len, num_modes, train=True, opt=opt)
            tr_loss += loss; tr_corr += mn; tr_tot += cnt; nb += 1
        # ---- holdout eval (no grad, no update) ----
        ho_corr, ho_tot = 0, 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if not is_holdout(bi):
                    continue
                batch = _to_device(batch, device)
                _, mn, cnt = _step(batch, mode_net, traj_net, gmm,
                                   hist_len, num_modes, train=False, opt=None)
                ho_corr += mn; ho_tot += cnt
        log.info(f"[probe] epoch {ep}: train_loss={tr_loss/max(nb,1):.4f} "
                 f"train_match={tr_corr/max(tr_tot,1):.1%}  "
                 f"HOLDOUT_match={ho_corr/max(ho_tot,1):.1%}")

    log.info("[probe] DONE. Compare HOLDOUT_match (not train) to closed-book tree (~75%).")
    log.info("[probe]  HOLDOUT >>75% -> real generalizable info; pursue full score-head training.")
    log.info("[probe]  HOLDOUT ~75%  -> train_match was overfit; hidden no better than handcrafted.")



def _find_gmm(net):
    """Locate the GMM head module inside traj_net."""
    from amelia_tf.models.components.gmm import GMM  # adjust import if needed
    for m in net.modules():
        if isinstance(m, GMM):
            return m
    raise RuntimeError("GMM head not found in traj_net; fix the import in _find_gmm.")


def _to_device(batch, device):
    sd = batch['scene_dict']
    for k, v in sd.items():
        if torch.is_tensor(v):
            sd[k] = v.to(device)
    return batch


def _step(batch, mode_net, traj_net, gmm, hist_len, num_modes, train, opt):
    Y = batch['scene_dict']['rel_sequences'][..., :4]
    X = torch.zeros_like(Y).float()
    X[:, :, :hist_len] = Y[:, :, :hist_len]
    context = batch['scene_dict']['context']
    adjacency = batch['scene_dict']['adjacency']
    ego_agent = batch['scene_dict']['ego_agent_id']

    Yxyz = Y[..., G.REL_XYZ[:4]]
    ego_fut = separate_ego_agent(Yxyz[:, :, hist_len:, :], ego_agent)  # (B,1,Tp,D)

    # run traj head per mode, collect mu + score (score head is on)
    total_loss = 0.0; match_n = 0; cnt = 0
    B, A = X.shape[:2]
    for m in range(num_modes):
        one_hot = F.one_hot(torch.full((B, A), m, dtype=torch.long, device=X.device),
                            num_classes=num_modes).float()
        out = traj_net(X[:, :, :, :4], context=context, adjacency=adjacency,
                       mask=None, mode_probs=one_hot)
        # out is (mu, sigma, score) when score head enabled
        mu, sigma, score = out
        # ego rows
        ego_mu = separate_ego_agent(mu, ego_agent)          # (B,1,T,K,D)
        ego_score = separate_ego_agent(score, ego_agent)    # (B,1,T,K)
        mu_fut = ego_mu[:, :, hist_len:, :, :2]              # (B,1,Tp,K,2)
        # winner label = candidate with min ADE to GT (only for true-mode samples? use all)
        with torch.no_grad():
            d = ((mu_fut - ego_fut[:, :, :, None, :2]) ** 2).sum(-1).mean(2)  # (B,1,K)
            winner = d.argmin(-1).squeeze(1)                 # (B,)
        # score: aggregate over time (mean) to (B,1,K) -> (B,K)
        s = ego_score[:, :, hist_len:].mean(2).squeeze(1)    # (B,K)
        loss = F.cross_entropy(s, winner)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
        match_n += (s.argmax(-1) == winner).sum().item()
        cnt += B
    return total_loss / num_modes, match_n, cnt


if __name__ == "__main__":
    main()