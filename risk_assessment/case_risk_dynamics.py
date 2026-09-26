"""
Case-study deep dive: for ONE confirmed candidate, compute the FULL
per-timestep separation-distance curve (not the collapsed min-over-horizon
scalar the aggregate stats/find_cases_*.py use) against a FIXED reference
agent (the one identified as closest-approach in check_agent_types.py), for
every (mode, candidate) hypothesis the two-stage model produces, plus the
realized ground-truth trajectory -- so the case study can show how risk
evolves over the 50s horizon, across modes, and across candidates within a
mode, not just a single collapsed number.

Uses the SAME reference agent for every curve (rather than "whichever agent
is closest at each timestep" independently per curve) so all curves answer
the same question: "how close does ego's candidate/real trajectory get to
THAT SPECIFIC other agent, second by second" -- comparable across modes/
candidates/reality.

Outputs a single JSON (not a plot -- the plot is built from this data
separately) with:
    meta: airport, batch_idx, sample_idx, ego_id, ref_agent_idx,
          ref_agent_type, mode_names, gt_mode, argmax_mode, mode_probs
    modes: {mode_name: {candidates: [{prob, dist_m: [T_pred floats]}, ...]}}
    real: {dist_m: [T_pred floats]}

Usage (reuses eval_two_stage.yaml's data/paths/model composition, same as
find_cases_two_stage.py):

    python -m risk_assessment.case_risk_dynamics \\
        ckpt=kmsy2 data=kmsy.yaml \\
        mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \\
        traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \\
        model.traj_net.config.num_hypotheses=4 \\
        +case_batch_idx=2708 +case_sample_idx=36 +case_ref_agent_idx=1 \\
        +output_json=/gpfs/scratch/exy064/ljx/Risk-Assessment/out/risk_assessment/kmsy_case_2708_36_dynamics.json
"""
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TWO_STAGE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main_two_phases_4T")
if _TWO_STAGE_REPO not in sys.path:
    sys.path.insert(0, _TWO_STAGE_REPO)

os.environ.setdefault("PROJECT_ROOT", _TWO_STAGE_REPO)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from amelia_tf.eval_two_stage import _build_nets, _find_gmm, _SCORER_ATTRS
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODES
from amelia_scenes.utils.transform_utils import inv_transform_batch

from risk_assessment.common import to_device, seed_for_reproducible_ego_selection

MODE_NAMES = TURN_MODES
AGENT_TYPE_NAMES = {0: "Aircraft", 1: "Vehicle", 2: "Unknown"}


def _mode_candidate_trajectories_abs(ego_mu, sequences, ego_ids, hist_len):
    """Same as find_cases_two_stage.py's helper -- (M,K,B,T_pred,2) abs XY."""
    B, T_total, M, K, D = ego_mu.shape
    T_pred = T_total - hist_len
    start_abs = np.stack([
        sequences[b, ego_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
        for b in range(B)
    ], axis=0)
    start_heading = np.array([
        float(sequences[b, ego_ids[b], hist_len - 1, G.HD].detach().cpu().item())
        for b in range(B)
    ])
    traj_abs = np.zeros((M, K, B, T_pred, 2), dtype=np.float64)
    for m in range(M):
        for k in range(K):
            future_rel = ego_mu[:, hist_len:, m, k, :2].detach().cpu().numpy()
            traj_abs[m, k] = inv_transform_batch(future_rel, start_abs, start_heading)
    return traj_abs


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main_two_phases_4T/configs",
            config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    output_json = cfg.get("output_json")
    if not output_json:
        raise ValueError("Pass +output_json=/path/to/case_dynamics.json")
    target_batch = cfg.get("case_batch_idx")
    target_sample = cfg.get("case_sample_idx")
    ref_agent_idx = cfg.get("case_ref_agent_idx")
    if target_batch is None or target_sample is None or ref_agent_idx is None:
        raise ValueError("Pass +case_batch_idx=, +case_sample_idx=, +case_ref_agent_idx=")
    target_batch, target_sample, ref_agent_idx = int(target_batch), int(target_sample), int(ref_agent_idx)

    mode_net, traj_net, device = _build_nets(cfg)

    sh_path = cfg.get("scorer", {}).get("score_head_load") if cfg.get("scorer") else None
    if sh_path:
        gmm = _find_gmm(traj_net)
        blob = torch.load(sh_path, map_location=device)
        if isinstance(blob, dict) and set(blob.keys()) <= set(_SCORER_ATTRS):
            for attr, sd in blob.items():
                getattr(gmm, attr).load_state_dict(sd)
        elif getattr(gmm, "per_mode_score", False):
            gmm.score_heads.load_state_dict(blob)
        else:
            gmm.score_head.load_state_dict(blob)
        print(f"[case_risk_dynamics] loaded trained score head from {sh_path}")

    model = CombinedTrajPredSystem(
        mode_model=mode_net, traj_model=traj_net, extra_params=cfg.model.extra_params)
    model.to(device)
    model.eval()

    hist_len = model.hist_len

    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > target_batch:
                raise RuntimeError(f"Never reached batch {target_batch} (stopped at {batch_idx})")
            if batch_idx != target_batch:
                continue
            batch = to_device(batch, device)
            scene = batch['scene_dict']

            mode_probs, traj_mu, traj_sigma, traj_score = model.forward(batch)

            ego_agent = scene['ego_agent_id']
            ego_ids = [
                ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                for b in range(mode_probs.shape[0])
            ]
            airport_ids = scene.get('airport_id')

            rule_based = scene.get('rule_based_encoding')
            true_mode_idx = rule_based[..., :4].float().argmax(dim=-1).long()
            ego_true_mode = separate_ego_agent(true_mode_idx, ego_ids).squeeze(1)

            ego_probs = separate_ego_agent(mode_probs, ego_ids).squeeze(1)      # (B, M)
            ego_mu = separate_ego_agent(traj_mu, ego_ids).squeeze(1)            # (B, T_total, M, K, D)

            sequences = scene['sequences']
            agent_masks = scene['agent_masks']

            if traj_score is not None:
                ego_score = separate_ego_agent(traj_score, ego_ids).squeeze(1)
                ego_mask = separate_ego_agent(agent_masks, ego_ids).squeeze(1)
                fut_mask = ego_mask[:, hist_len:].float()
                m_exp = fut_mask[:, :, None, None]
                denom = m_exp.sum(dim=1).clamp_min(1)
                s_pt = ego_score[:, hist_len:]
                score_fut = (s_pt * m_exp).sum(dim=1) / denom
                cand_probs = torch.softmax(score_fut, dim=-1).detach().cpu().numpy()  # (B, M, K)
            else:
                B_, M_, K_ = ego_probs.shape[0], ego_probs.shape[1], ego_mu.shape[3]
                cand_probs = np.ones((B_, M_, K_), dtype=np.float64) / K_

            traj_abs = _mode_candidate_trajectories_abs(
                ego_mu, sequences, ego_ids, hist_len)  # (M,K,B,T_pred,2)

            b = target_sample
            ego_id = ego_ids[b]
            gt_mode = int(ego_true_mode[b].item())
            argmax_mode = int(ego_probs[b].argmax().item())

            ref_xy = sequences[b, ref_agent_idx, hist_len:, G.XY].detach().cpu().numpy()  # (T_pred, 2)
            ref_valid = agent_masks[b, ref_agent_idx, hist_len:].bool().detach().cpu().numpy()
            ref_type = int(scene['agent_types'].reshape(sequences.shape[0], -1)[b, ref_agent_idx].item())

            real_ego_xy = sequences[b, ego_id, hist_len:, G.XY].detach().cpu().numpy()  # (T_pred, 2)
            real_dist = np.linalg.norm(real_ego_xy - ref_xy, axis=-1)
            real_dist = np.where(ref_valid, real_dist, np.nan) * 1000  # km -> m

            M, K = traj_abs.shape[0], traj_abs.shape[1]
            modes_out = {}
            for m in range(M):
                cands = []
                for k in range(K):
                    dist = np.linalg.norm(traj_abs[m, k, b] - ref_xy, axis=-1)
                    dist = np.where(ref_valid, dist, np.nan) * 1000  # km -> m
                    cands.append({
                        "prob": float(cand_probs[b, m, k]),
                        "dist_m": [None if np.isnan(v) else float(v) for v in dist],
                    })
                modes_out[MODE_NAMES[m]] = {
                    "mode_prob": float(ego_probs[b, m].item()),
                    "candidates": cands,
                }

            out = {
                "meta": {
                    "airport": airport_ids[b] if airport_ids is not None else None,
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "ego_id": ego_id,
                    "ref_agent_idx": ref_agent_idx,
                    "ref_agent_type": AGENT_TYPE_NAMES.get(ref_type, str(ref_type)),
                    "mode_names": MODE_NAMES,
                    "gt_mode": MODE_NAMES[gt_mode] if 0 <= gt_mode < len(MODE_NAMES) else gt_mode,
                    "argmax_mode": MODE_NAMES[argmax_mode],
                },
                "modes": modes_out,
                "real": {"dist_m": [None if np.isnan(v) else float(v) for v in real_dist]},
            }
            with open(output_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[case_risk_dynamics] wrote {output_json}")
            return

    raise RuntimeError(f"Batch {target_batch} not found in dataloader")


if __name__ == "__main__":
    main()
