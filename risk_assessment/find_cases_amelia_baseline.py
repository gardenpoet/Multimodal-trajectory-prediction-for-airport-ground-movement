"""
Contribution 3 (risk-aware downstream application) case-finding + scoring --
AmeliaTF_main (plain, non-manoeuvre-conditioned) baseline adapter.

Unlike the two-stage model (find_cases_two_stage.py), this baseline has no
manoeuvre/mode classifier at all -- amelia_tf.models.components.gmm.GMM
produces H raw multi-hypothesis trajectories per agent with a genuine
softmax pred_scores (B,A,H) directly, no "mode" grouping above it. Every
hypothesis is therefore already a flat (trajectory, probability) pair, fed
straight into common.aggregate_risk with no mode x candidate flattening
step needed (H takes the role two_stage's flattened M*K grid played).

naive = the argmax(pred_scores) hypothesis (what a downstream module that
only consumes the top-1 prediction would see). ambiguous = top1-top2 gap
on pred_scores directly (no "feasible modes" concept to restrict to here,
since there's no map-derived feasibility mask for a raw hypothesis
distribution -- gate_pool is every hypothesis). worst_case/prob_weighted
marginalise over all H hypotheses. See common.py's docstring for the
SAFETY_MARGIN_KM/AMBIGUITY_MARGIN/RELEVANCE_RADIUS_KM rationale.

rule_based_encoding (ground-truth manoeuvre label) DOES exist in this
repo's scene_dict (confirmed: amelia_tf/data/components/amelia_dataset.py
builds it, inherited from the same data pipeline as the two-stage repo),
but the model itself never sees or is conditioned on it -- it's used here
purely as a descriptive `gt_mode` grouping column (e.g. "does the naive-
risk miss rate differ by manoeuvre category even though the model itself
is manoeuvre-blind"), not as anything the model predicted or was gated by.

risk_assessment/ lives at the top level of this repo (see
find_cases_two_stage.py's docstring for why) -- run from the repo ROOT,
not from inside AmeliaTF_main/. This file inserts AmeliaTF_main's path
onto sys.path itself so `import amelia_tf...` resolves to ITS copy.

Usage (AmeliaTF_main only has a working eval config for KLAX confirmed --
configs/eval_klax.yaml, composing data=klax.yaml, model=marginal.yaml,
paths=default.yaml; pass --config-name for a different airport if/when
an equivalent eval_<airport>.yaml exists there):

    python -m risk_assessment.find_cases_amelia_baseline \\
        --config-name=eval_klax \\
        ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/out/.../epoch_062.ckpt \\
        +output_csv=/gpfs/scratch/exy064/ljx/Risk-Assessment/out/risk_assessment/klax_baseline_50_cases.csv

Output columns: airport, batch_idx, sample_idx, scene_file, ego_id,
gt_mode, num_candidates, risk_naive, risk_prob_weighted, risk_worst_case,
risk_gated, strategy_divergence, ambiguous, top1_min_sep_km (PREDICTED:
top-1 candidate vs other agents' real trajectories) + top1_min_sep_
agent_type/_on_road/_edge_dist_m, true_min_sep_km (REALIZED: ego's own
actual trajectory vs reality) + true_min_sep_agent_type/_on_road/
_edge_dist_m -- see find_cases_two_stage.py's docstring for the predicted-
vs-realized distinction and common.check_on_road's docstring for the
on/off-road gate-vs-movement-area proxy.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASELINE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main")
if _BASELINE_REPO not in sys.path:
    sys.path.insert(0, _BASELINE_REPO)

# configs/paths/default.yaml resolves root_dir via ${oc.env:PROJECT_ROOT} --
# normally set by this repo's own eval.py/train_*.py via
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True),
# which walks up from THEIR OWN file location to find AmeliaTF_main/.project-root.
# That walk would find the wrong (or no) root starting from this file's
# location instead, so set it directly to the repo whose config we're
# actually composing.
os.environ.setdefault("PROJECT_ROOT", _BASELINE_REPO)

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODES
from amelia_tf.utils.off_road_evaluator import OffRoadEvaluator
from amelia_scenes.utils.transform_utils import inv_transform

from risk_assessment.common import (
    to_device, min_separation, min_separation_with_type,
    aggregate_risk, seed_for_reproducible_ego_selection, AMBIGUITY_MARGIN,
    load_airport_ref, check_on_road,
    mc_violation_risk, MC_THRESHOLD_KM_200FT, MC_SAMPLES_DEFAULT,
)

GT_MODE_NAMES = TURN_MODES  # descriptive only -- see module docstring
AGENT_TYPE_NAMES = {0: "Aircraft", 1: "Vehicle", 2: "Unknown"}

# Lazily built, one OffRoadEvaluator per airport -- see
# find_cases_two_stage.py's equivalent cache for why.
_OFF_ROAD_EVALUATORS = {}
_AIRPORT_REFS = {}


def _get_off_road_evaluator(assets_dir, airport):
    if airport not in _OFF_ROAD_EVALUATORS:
        _OFF_ROAD_EVALUATORS[airport] = OffRoadEvaluator(assets_dir, airport)
        _AIRPORT_REFS[airport] = load_airport_ref(assets_dir, airport)
    return _OFF_ROAD_EVALUATORS[airport], _AIRPORT_REFS[airport]


def _hypothesis_trajectories_abs(ego_mu, sequences, ego_ids, hist_len):
    """
    ego_mu: (B, T_total, H, D) relative predicted trajectories (ego only).
    sequences: (B, A, T_total, D_seq) absolute per-agent sequences (whole scene).
    ego_ids: list[int] of length B.
    Returns: (H, B, T_pred, 2) absolute XY per hypothesis, and (B,)
    start_heading -- the SAME theta inv_transform used to rotate ego_mu
    into traj_abs, returned so a caller can apply the identical rotation to
    ego_sigma (see common.mc_violation_risk's docstring for why).

    No inv_transform_batch in this repo's vendored amelia_scenes (that
    batched helper is specific to AmeliaTF_main_two_phases_4T) -- loops
    over samples with the per-sample inv_transform instead, same maths.
    """
    B, T_total, H, D = ego_mu.shape
    T_pred = T_total - hist_len

    start_abs = np.stack([
        sequences[b, ego_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
        for b in range(B)
    ], axis=0)  # (B, 2)
    start_heading = np.array([
        float(sequences[b, ego_ids[b], hist_len - 1, G.HD].detach().cpu().item())
        for b in range(B)
    ])  # (B,), degrees -- inv_transform expects degrees (see its docstring)

    traj_abs = np.zeros((H, B, T_pred, 2), dtype=np.float64)
    for h in range(H):
        future_rel = ego_mu[:, hist_len:, h, :2].detach().cpu().numpy()  # (B, T_pred, 2)
        for b in range(B):
            traj_abs[h, b] = inv_transform(future_rel[b], start_abs[b], start_heading[b])
    return traj_abs, start_heading


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main/configs",
            config_name="eval_klax")
def main(cfg: DictConfig) -> None:
    output_csv = cfg.get("output_csv")
    if not output_csv:
        raise ValueError("Pass +output_csv=/path/to/cases.csv on the command line.")
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Pass ckpt_path=/path/to/checkpoint.ckpt on the command line.")

    # See find_cases_two_stage.py's docstring for risk_method="mc"'s meaning
    # and defaults -- identical flag here.
    risk_method = cfg.get("risk_method", "margin")
    mc_threshold_km = (cfg.get("mc_threshold_ft") * 0.3048 / 1000.0) if cfg.get("mc_threshold_ft") else MC_THRESHOLD_KM_200FT
    mc_samples = cfg.get("mc_samples", MC_SAMPLES_DEFAULT)
    mc_rng = np.random.default_rng(cfg.get("seed", 42)) if risk_method == "mc" else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model)

    # Plain state-dict load (mirrors AmeliaTF_main_two_phases_4T/amelia_tf/eval_two_stage.py's
    # load_model_state) rather than TrajPred.load_from_checkpoint, since the
    # latter needs the exact __init__ kwargs reconstructed and this repo has
    # no confirmed helper for that -- state_dict keys already carry the
    # "net." prefix from how the checkpoint was saved (self.net = ... inside
    # TrajPred), so loading onto the freshly-instantiated LightningModule
    # directly (not model.net) is correct.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] missing keys: {result.missing_keys}")
    print(f"[LOAD] unexpected keys: {result.unexpected_keys}")
    model.to(device)
    model.eval()

    hist_len = model.hist_len

    datamodule = hydra.utils.instantiate(cfg.data)
    # See common.py's seed_for_reproducible_ego_selection docstring -- forces
    # num_workers=0 and seeds random/numpy/torch so the per-sample random ego
    # selection in amelia_dataset.py's transform_scene_data is reproducible
    # and IDENTICAL across separate find_cases_*.py runs (this script never
    # goes through a Trainer, so Lightning's own seed_everything(workers=True)
    # per-worker reproducibility never applies here).
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    # AmeliaTF_main's DataModule.test_dataloader() can return a list
    # [original_test_set, balanced_test_set] like the two-stage repo's does --
    # only the first (original, unbalanced) one is what every other eval
    # script in this repo reports its headline numbers against.
    if isinstance(dataloader, (list, tuple)):
        dataloader = dataloader[0]

    limit_batches = cfg.get("limit_batches")

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit_batches is not None and batch_idx >= limit_batches:
                print(f"[find_cases] stopping early: limit_batches={limit_batches}")
                break
            batch = to_device(batch, device)
            scene = batch['scene_dict']

            # Exact X construction from TrajPred.model_step (amelia_tf/models/trajpred.py),
            # replicated directly rather than going through model_step/test_step
            # since those also compute training loss and do plotting/logging
            # this script doesn't need.
            Y = scene['rel_sequences']
            X = torch.zeros_like(Y).type(torch.float)
            X[:, :, :hist_len] = Y[:, :, :hist_len]
            X = X[:, :, :, :4]
            context = scene['context']
            adjacency = scene['adjacency']

            pred_scores, mu, sigma = model.net(X, context=context, adjacency=adjacency, mask=None)

            # ego_agent_id_test, NOT ego_agent_id -- test_step in trajpred.py
            # uses the _test variant specifically; using the train/val one
            # here would silently evaluate the wrong agent's predictions.
            ego_agent = scene['ego_agent_id_test']
            ego_ids = [
                ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                for b in range(mu.shape[0])
            ]
            airport_ids = scene.get('airport_id')
            scene_files = scene.get('scene_file')

            ego_pred_scores = separate_ego_agent(pred_scores, ego_ids).squeeze(1)  # (B, H)
            ego_mu = separate_ego_agent(mu, ego_ids).squeeze(1)                    # (B, T_total, H, D)
            ego_sigma_t = separate_ego_agent(sigma, ego_ids).squeeze(1) if risk_method == "mc" else None  # (B, T_total, H, D)

            rule_based = scene.get('rule_based_encoding')
            gt_mode_np = None
            if rule_based is not None:
                true_mode_idx = rule_based[..., :4].float().argmax(dim=-1).long()  # (B, A)
                ego_true_mode = separate_ego_agent(true_mode_idx, ego_ids).squeeze(1)  # (B,)
                gt_mode_np = ego_true_mode.detach().cpu().numpy()

            sequences = scene['sequences']
            agent_masks = scene['agent_masks']
            agent_types = scene['agent_types']
            B, A = sequences.shape[:2]
            # merge_seq1d_by_padding concatenates (torch.cat, not
            # torch.stack) each padded per-sample vector -- flat
            # (B * per_sample_len,), not (B, per_sample_len). Reshape to
            # recover per-sample indexing.
            agent_types = agent_types.reshape(B, -1)
            H = ego_pred_scores.shape[1]

            traj_abs, start_heading = _hypothesis_trajectories_abs(
                ego_mu, sequences, ego_ids, hist_len)  # (H, B, T_pred, 2), (B,)

            probs_np = ego_pred_scores.detach().cpu().numpy()  # (B, H)

            for b in range(B):
                ego_id = ego_ids[b]
                other_xy, other_valid, other_types = [], [], []
                for a in range(A):
                    if a == ego_id:
                        continue
                    valid_a = agent_masks[b, a, hist_len:].bool().detach().cpu().numpy()
                    if not valid_a.any():
                        continue
                    xy_a = sequences[b, a, hist_len:, G.XY].detach().cpu().numpy()
                    other_xy.append(xy_a)
                    other_valid.append(valid_a)
                    t_a = agent_types[b, a]
                    other_types.append(int(t_a.item()) if torch.is_tensor(t_a) else int(t_a))

                T_pred = traj_abs.shape[2]
                if other_xy:
                    other_xy_arr = np.stack(other_xy, axis=0)
                    other_valid_arr = np.stack(other_valid, axis=0)
                    other_types_arr = np.array(other_types)
                else:
                    other_xy_arr = np.zeros((0, T_pred, 2))
                    other_valid_arr = np.zeros((0, T_pred), dtype=bool)
                    other_types_arr = np.zeros((0,), dtype=int)

                min_sep = np.array([
                    min_separation(traj_abs[h, b], other_xy_arr, other_valid_arr)
                    for h in range(H)
                ])  # (H,)

                # No relevance pre-filter -- see find_cases_two_stage.py's
                # equivalent comment: every sample is kept, with all_zero_risk
                # marking the ones with no nearby agent at all.

                mc_risk = None
                if risk_method == "mc":
                    ego_sigma_b = ego_sigma_t[b, hist_len:].detach().cpu().numpy()  # (T_pred, H, D)
                    mc_risk = np.array([
                        mc_violation_risk(
                            traj_abs[h, b], ego_sigma_b[:, h, :2], start_heading[b],
                            other_xy_arr, other_valid_arr,
                            threshold_km=mc_threshold_km, num_samples=mc_samples, rng=mc_rng)
                        for h in range(H)
                    ])  # (H,)

                probs_b = probs_np[b]  # (H,), already a genuine softmax over H
                naive_idx = int(probs_b.argmax())

                # No feasibility concept for this model, so the gate pool is
                # hypotheses rather than modes, but the same chaining logic
                # applies: only the cluster of hypotheses mutually within
                # AMBIGUITY_MARGIN of each other, not every one of the H
                # hypotheses regardless of how far its own probability is
                # from the contested top ones -- see find_cases_two_stage.py's
                # equivalent comment for why (this used to make gated
                # indistinguishable from plain worst-case whenever ambiguous
                # fired).
                order = np.argsort(probs_b)[::-1]
                sorted_probs = probs_b[order]
                ambiguous = bool((sorted_probs[0] - sorted_probs[1]) < AMBIGUITY_MARGIN) \
                    if H >= 2 else False
                gate_pool_idx = np.array([], dtype=int)
                if ambiguous:
                    group_end = 1
                    while (group_end < H
                           and sorted_probs[group_end - 1] - sorted_probs[group_end] < AMBIGUITY_MARGIN):
                        group_end += 1
                    gate_pool_idx = order[:group_end]

                risk = aggregate_risk(
                    min_sep, probs_b, naive_idx, gate_pool_idx, ambiguous,
                    risk_flat=mc_risk)

                # For case-study screening only -- see find_cases_two_stage.py's
                # equivalent comment (ground service vehicles legitimately
                # operate within sub-metre distance of a gate-adjacent
                # aircraft as routine, non-hazardous ground ops).
                top1_xy = traj_abs[naive_idx, b]
                _, top1_closest_type, top1_t_idx = min_separation_with_type(
                    top1_xy, other_xy_arr, other_valid_arr, other_types_arr)

                # true_min_sep_km: ego's own REALIZED trajectory vs reality
                # (folds in what used to need a separate check_agent_types.py
                # pass) -- see find_cases_two_stage.py's equivalent comment
                # for the predicted-vs-realized distinction.
                real_ego_xy = sequences[b, ego_id, hist_len:, G.XY].detach().cpu().numpy()
                true_min_sep, true_closest_type, true_t_idx = min_separation_with_type(
                    real_ego_xy, other_xy_arr, other_valid_arr, other_types_arr)

                # Case-study legibility screen -- see find_cases_two_stage.py's
                # equivalent comment: net displacement and straightness over
                # ego's REALIZED history+future, so a near-stationary or
                # erratic-reversal agent can be filtered out even when its
                # risk numbers look clean.
                hist_ego_mask = agent_masks[b, ego_id, :hist_len].bool().detach().cpu().numpy()
                hist_ego_xy = sequences[b, ego_id, :hist_len, G.XY].detach().cpu().numpy()[hist_ego_mask]
                full_ego_mask = np.concatenate([
                    hist_ego_mask,
                    agent_masks[b, ego_id, hist_len:].bool().detach().cpu().numpy(),
                ])
                full_ego_xy = sequences[b, ego_id, :, G.XY].detach().cpu().numpy()[full_ego_mask]

                def _displacement_and_straightness(xy):
                    if xy.shape[0] < 2:
                        return 0.0, 1.0
                    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                    path_len = float(seg.sum())
                    net = float(np.linalg.norm(xy[-1] - xy[0]))
                    return net, (net / path_len if path_len > 1e-9 else 1.0)

                ego_hist_disp_km, _ = _displacement_and_straightness(hist_ego_xy)
                ego_full_disp_km, ego_full_straightness = _displacement_and_straightness(full_ego_xy)

                # On/off-road location check (folds in what used to need a
                # separate check_case_location.py pass) -- see
                # common.check_on_road's docstring.
                evaluator, ref = _get_off_road_evaluator(cfg.paths.assets_dir, airport_ids[b])
                top1_on_road, top1_edge_dist = (
                    check_on_road(evaluator, top1_xy[top1_t_idx], ref)
                    if top1_t_idx is not None else (None, None))
                true_on_road, true_edge_dist = (
                    check_on_road(evaluator, real_ego_xy[true_t_idx], ref)
                    if true_t_idx is not None else (None, None))

                gt_mode = None
                if gt_mode_np is not None:
                    idx = int(gt_mode_np[b])
                    gt_mode = GT_MODE_NAMES[idx] if 0 <= idx < len(GT_MODE_NAMES) else idx

                row = {
                    "airport": airport_ids[b] if airport_ids is not None else None,
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    # Stable, repo-independent scene identifier -- see
                    # amelia_dataset.py's __getitem__ comment. batch_idx/
                    # sample_idx alone are NOT comparable across model repos
                    # (independently-populated proc_full_scenes/ copies have
                    # no guaranteed os.listdir order), so use this for
                    # cross-model case alignment instead.
                    "scene_file": scene_files[b] if scene_files is not None else None,
                    # see find_cases_two_stage.py's row dict comment: ego agent
                    # selection can be randomised per-sample and isn't
                    # guaranteed reproducible across reruns, so record the
                    # resolved index for later case-study lookups.
                    "ego_id": ego_id,
                    "gt_mode": gt_mode,
                    "num_candidates": H,
                    "ambiguous": ambiguous,
                    **risk,
                    "all_zero_risk": bool(
                        risk["risk_naive"] == 0.0 and risk["risk_worst_case"] == 0.0
                        and risk["risk_prob_weighted"] == 0.0 and risk["risk_gated"] == 0.0),
                    "top1_min_sep_km": (
                        float(min_sep[naive_idx]) if np.isfinite(min_sep[naive_idx]) else None),
                    "top1_min_sep_agent_type": AGENT_TYPE_NAMES.get(top1_closest_type),
                    "top1_min_sep_on_road": top1_on_road,
                    "top1_min_sep_edge_dist_m": top1_edge_dist,
                    "true_min_sep_km": (
                        float(true_min_sep) if np.isfinite(true_min_sep) else None),
                    "true_min_sep_agent_type": AGENT_TYPE_NAMES.get(true_closest_type),
                    "true_min_sep_on_road": true_on_road,
                    "true_min_sep_edge_dist_m": true_edge_dist,
                    "ego_hist_displacement_m": ego_hist_disp_km * 1000.0,
                    "ego_full_displacement_m": ego_full_disp_km * 1000.0,
                    "ego_track_straightness": ego_full_straightness,
                }
                rows.append(row)

            if batch_idx % 50 == 0:
                print(f"[find_cases] processed batch {batch_idx}, {len(rows)} samples so far")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[find_cases] wrote {len(df)} rows to {output_csv}")
    print(f"[find_cases] all_zero_risk rate (excluded from the stats below): {df['all_zero_risk'].mean():.3f}")
    relevant = df[~df["all_zero_risk"]]
    print(f"[find_cases] ambiguous rate: {relevant['ambiguous'].mean():.3f}")
    print(f"[find_cases] strategy_divergence rate: {relevant['strategy_divergence'].mean():.3f}")


if __name__ == "__main__":
    main()
