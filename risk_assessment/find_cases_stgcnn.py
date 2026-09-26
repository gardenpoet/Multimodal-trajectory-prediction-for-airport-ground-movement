"""
Contribution 3 (risk-aware downstream application) case-finding + scoring --
STGCNN_baseline (Zhang, Zhong & Mahadevan 2022, TRC 144:103873) adapter.

This model is genuinely unimodal: STGCNNPredictor.forward() returns exactly
one (mu, sigma) pair per agent per future timestep -- no candidate/mode
dimension, no mixture weight, num_dec_heads=1 by construction (confirmed:
its own module docstring states "a single bivariate Gaussian per agent per
future timestep, no turn-mode classification"). So there is exactly one
hypothesis per sample here (N=1, probability=1), fed through the same
common.aggregate_risk as every other adapter in this folder for schema
consistency -- naive/prob_weighted/worst_case/gated necessarily collapse to
the SAME number for every row (nothing to be ambiguous about, nothing to
take a worst-case or weighted-average over), which is the honest, expected
result for a model with no multi-hypothesis structure at all, not a bug in
this script. The point of running it through the same pipeline as the
other two adapters is exactly to make that contrast explicit in a later
cross-model comparison (a downstream module built on this predictor has NO
way to represent "I'm not sure which of several futures is right", by
construction).

rule_based_encoding exists in this repo's scene_dict too (same data
pipeline, copied unmodified from AmeliaTF_main) but is never fed to or
supervised by the model -- gt_mode below is a purely descriptive grouping
column, same caveat as find_cases_amelia_baseline.py.

risk_assessment/ lives at the top level of this repo (see
find_cases_two_stage.py's docstring for why) -- run from the repo ROOT,
not from inside STGCNN_baseline/. This file inserts STGCNN_baseline's path
onto sys.path itself so `import amelia_tf...` resolves to ITS copy.

Usage (there is no separate eval.py in this repo -- eval-only mode is
train=false plus an explicit ckpt_path, same pattern as every
train_stgcnn_<airport>[_20].sh script here):

    python -m risk_assessment.find_cases_stgcnn \\
        --config-name=train_stgcnn_klax \\
        train=false \\
        ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_176.ckpt \\
        +output_csv=/gpfs/scratch/exy064/ljx/Risk-Assessment/out/risk_assessment/klax_stgcnn_50_cases.csv

Output columns: airport, batch_idx, sample_idx, gt_mode, num_candidates
(always 1), risk_naive, risk_prob_weighted, risk_worst_case, risk_gated
(all four identical per row -- see module docstring), ambiguous (always
False), top1_min_sep_km.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STGCNN_REPO = os.path.join(_REPO_ROOT, "STGCNN_baseline")
if _STGCNN_REPO not in sys.path:
    sys.path.insert(0, _STGCNN_REPO)

# configs/paths/default.yaml resolves root_dir via ${oc.env:PROJECT_ROOT} --
# normally set by this repo's own train_stgcnn_*.py via
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True),
# which walks up from THEIR OWN file location to find STGCNN_baseline/.project-root.
# That walk would find the wrong (or no) root starting from this file's
# location instead, so set it directly to the repo whose config we're
# actually composing.
os.environ.setdefault("PROJECT_ROOT", _STGCNN_REPO)

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODES
from amelia_scenes.utils.transform_utils import inv_transform

from risk_assessment.common import (
    to_device, min_separation, min_separation_with_type, has_nearby_agent,
    aggregate_risk, seed_for_reproducible_ego_selection,
)

GT_MODE_NAMES = TURN_MODES  # descriptive only -- see module docstring
AGENT_TYPE_NAMES = {0: "Aircraft", 1: "Vehicle", 2: "Unknown"}


def _ego_trajectory_abs(ego_mu, sequences, ego_ids, hist_len):
    """
    ego_mu: (B, T_pred, 2) relative predicted trajectory (ego only, single
        hypothesis -- no H dim to loop over, unlike the other two adapters).
    sequences: (B, A, T_total, D_seq) absolute per-agent sequences (whole scene).
    ego_ids: list[int] of length B.
    Returns: (B, T_pred, 2) absolute XY.
    """
    B = ego_mu.shape[0]
    future_rel = ego_mu[..., :2].detach().cpu().numpy()  # (B, T_pred, 2)
    traj_abs = np.zeros_like(future_rel)
    for b in range(B):
        start_abs = sequences[b, ego_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
        start_heading = float(sequences[b, ego_ids[b], hist_len - 1, G.HD].detach().cpu().item())
        traj_abs[b] = inv_transform(future_rel[b], start_abs, start_heading)
    return traj_abs


@hydra.main(version_base="1.3", config_path="../STGCNN_baseline/configs",
            config_name="train_stgcnn_klax")
def main(cfg: DictConfig) -> None:
    output_csv = cfg.get("output_csv")
    if not output_csv:
        raise ValueError("Pass +output_csv=/path/to/cases.csv on the command line.")
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "Pass train=false ckpt_path=/path/to/checkpoint.ckpt on the command line "
            "(same eval-only convention as this repo's own train_stgcnn_*.sh scripts).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] missing keys: {result.missing_keys}")
    print(f"[LOAD] unexpected keys: {result.unexpected_keys}")
    model.to(device)
    model.eval()

    hist_len = model.hist_len
    max_pred_len = model.max_pred_len

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

            # Exact X/mask construction from STGCNNTrajPred.model_step: only the
            # OBSERVED history window is passed in (no zero-padded future) --
            # different from the Amelia-baseline adapter, which zero-pads and
            # passes the full T_total-length tensor through its net.
            seq = scene['rel_sequences']              # (B, A, T, 7)
            masks = scene['agent_masks'].bool()        # (B, A, T)
            X = seq[:, :, :hist_len].float()
            mask_h = masks[:, :, :hist_len]

            mu, sigma = model.net(X, mask_h)           # (B, A, Tp, 2) each, unimodal

            ego_agent = scene['ego_agent_id_test']
            ego_ids = [
                ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                for b in range(mu.shape[0])
            ]
            airport_ids = scene.get('airport_id')
            scene_files = scene.get('scene_file')

            ego_mu = separate_ego_agent(mu, ego_ids).squeeze(1)  # (B, Tp, 2)

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

            traj_abs = _ego_trajectory_abs(ego_mu, sequences, ego_ids, hist_len)  # (B, Tp, 2)

            for b in range(B):
                ego_id = ego_ids[b]
                other_xy, other_valid, other_types = [], [], []
                for a in range(A):
                    if a == ego_id:
                        continue
                    fut_end = hist_len + max_pred_len
                    valid_a = agent_masks[b, a, hist_len:fut_end].bool().detach().cpu().numpy()
                    if not valid_a.any():
                        continue
                    xy_a = sequences[b, a, hist_len:fut_end, G.XY].detach().cpu().numpy()
                    other_xy.append(xy_a)
                    other_valid.append(valid_a)
                    t_a = agent_types[b, a]
                    other_types.append(int(t_a.item()) if torch.is_tensor(t_a) else int(t_a))

                T_pred = traj_abs.shape[1]
                if other_xy:
                    other_xy_arr = np.stack(other_xy, axis=0)
                    other_valid_arr = np.stack(other_valid, axis=0)
                    other_types_arr = np.array(other_types)
                else:
                    other_xy_arr = np.zeros((0, T_pred, 2))
                    other_valid_arr = np.zeros((0, T_pred), dtype=bool)
                    other_types_arr = np.zeros((0,), dtype=int)

                min_sep = np.array([min_separation(traj_abs[b], other_xy_arr, other_valid_arr)])  # (1,)

                if not has_nearby_agent(min_sep):
                    continue  # Filter 1: no scenario relevance, skip

                probs_b = np.array([1.0])  # single hypothesis, probability 1
                risk = aggregate_risk(min_sep, probs_b, naive_idx=0,
                                       gate_pool_idx=np.array([0]), ambiguous=False)

                # For case-study screening only -- see find_cases_two_stage.py's
                # equivalent comment (ground service vehicles legitimately
                # operate within sub-metre distance of a gate-adjacent
                # aircraft as routine, non-hazardous ground ops).
                _, top1_closest_type = min_separation_with_type(
                    traj_abs[b], other_xy_arr, other_valid_arr, other_types_arr)

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
                    "num_candidates": 1,
                    "ambiguous": False,
                    **risk,
                    "top1_min_sep_km": float(min_sep[0]) if np.isfinite(min_sep[0]) else None,
                    "top1_min_sep_agent_type": AGENT_TYPE_NAMES.get(top1_closest_type),
                }
                rows.append(row)

            if batch_idx % 50 == 0:
                print(f"[find_cases] processed batch {batch_idx}, {len(rows)} relevant samples so far")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[find_cases] wrote {len(df)} rows to {output_csv}")
    print(f"[find_cases] strategy_divergence rate: {df['strategy_divergence'].mean():.3f} "
          f"(expected 0.0 -- N=1 hypothesis, nothing for the strategies to disagree on)")


if __name__ == "__main__":
    main()
