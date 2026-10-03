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

The reference agent is AUTO-DETECTED as whichever agent achieved the
closest approach in the REALIZED (ground-truth) trajectory -- i.e. the
same agent find_cases_two_stage.py's true_min_sep_gt_agent_type column
already identifies the type of, just resolved here to its raw agent index
(not persisted in that CSV, since the index isn't stable/meaningful outside
one script's own run). This is the physically real "other aircraft"
involved in the case, so it's the right fixed reference regardless of
which mode/candidate curve is being examined. Pass +case_ref_agent_idx= to
override (e.g. to inspect a different agent's proximity for the same case).

Outputs a single JSON (not a plot -- the plot is built from this data
separately) with:
    meta: airport, batch_idx, sample_idx, scene_file, ego_id, ref_agent_idx,
          ref_agent_type, mode_names, gt_mode, argmax_mode, hist_len
    gt: {ego, ref_agent, other_agents: [...]} -- each {agent_idx, agent_type,
        latlon_hist, latlon_fut, valid_hist, valid_fut} for a map overlay
        (lat/lon, not the local-XY km frame the risk maths uses -- see
        _xy_to_latlon)
    modes: {mode_name: {mode_prob, feasible (bool -- same turn_feasibility
            mask risk_gated restricts its max to; an infeasible mode's
            candidates have no physical meaning and should be omitted from
            the chart), candidates: [{prob, dist_m: [T_pred floats],
            latlon: [[lat,lon], ...], sigma_xy: [[sigma_x,sigma_y], ...]},
            ...]}}
    real: {dist_m: [T_pred floats]} (kept for the risk-dynamics chart;
          redundant with gt.ego's future half but in the ref-agent-distance
          form that chart wants directly)

    sigma_xy is the GMM decoder's own predicted per-timestep std (x,y), RAW
    -- i.e. in the same egocentric-at-t=hist_len-1, heading-relative frame
    mu is in BEFORE inv_transform_batch's rotation+translation, NOT rotated
    into the same absolute/lat-lon frame dist_m/latlon are in. Rotating a
    covariance correctly needs R(theta) Sigma R(theta)^T, not just
    reinterpreting sigma_x/sigma_y unchanged -- meta.start_heading_deg is
    the theta that rotation would need, for whenever that gets built.
    Currently unused by anything (no consumer reads this field yet) --
    captured now, alongside the model output it was always computing
    anyway, so a future Monte-Carlo/uncertainty pass doesn't need to
    re-run every case from scratch.

The latlon fields are for risk_assessment/case_trajectories_{stgcnn,
amelia_baseline}.py's map-overlay companion outputs -- see those scripts'
docstrings for why cross-model trajectory comparison needs lat/lon (a
shared coordinate frame) rather than each repo's own local-XY frame.

Usage (reuses eval_two_stage.yaml's data/paths/model composition, same as
find_cases_two_stage.py; case_ref_agent_idx is optional, see above):

    python -m risk_assessment.case_risk_dynamics \\
        ckpt=kmsy2 data=kmsy.yaml \\
        mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \\
        traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \\
        model.traj_net.config.num_hypotheses=4 \\
        +case_batch_idx=292 +case_sample_idx=46 \\
        +output_json=/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_292_46_dynamics_4T.json
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

from risk_assessment.common import (
    to_device, seed_for_reproducible_ego_selection, load_airport_ref, xy_array_to_latlon,
)

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
    if target_batch is None or target_sample is None:
        raise ValueError("Pass +case_batch_idx=, +case_sample_idx=")
    target_batch, target_sample = int(target_batch), int(target_sample)
    # None (not set) means auto-detect from the realized trajectory below --
    # see the module docstring for why that's the right default reference.
    ref_agent_idx_override = cfg.get("case_ref_agent_idx")
    ref_agent_idx_override = (
        int(ref_agent_idx_override) if ref_agent_idx_override is not None else None)

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

    # Materialising every batch up to target_batch via enumerate(dataloader)
    # forces __getitem__ on every OTHER scene along the way too. Since
    # +data.dataset.config.ego_agent_id is a dataset-wide override (applied
    # in __getitem__ to every scene, not just the one we actually want), any
    # other scene with fewer real agents than that pinned index crashes the
    # whole run before ever reaching the target batch -- even though the
    # target scene itself may be perfectly fine. Restricting to a
    # single-item Subset containing only the target scene's own dataset
    # index sidesteps every other scene entirely, so a global
    # ego_agent_id override is safe regardless of what the rest of the test
    # split looks like.
    batch_size = dataloader.batch_size or 1
    dataset_idx = target_batch * batch_size + target_sample
    single_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataloader.dataset, [dataset_idx]),
        batch_size=1, shuffle=False, num_workers=0,
        collate_fn=dataloader.collate_fn,
    )
    batch = next(iter(single_loader))
    batch = to_device(batch, device)
    scene = batch['scene_dict']

    with torch.no_grad():
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
        ego_sigma = separate_ego_agent(traj_sigma, ego_ids).squeeze(1)      # (B, T_total, M, K, D)

        # Which modes are geometrically feasible for this scene (same
        # mask risk_gated restricts its max to in find_cases_two_stage.py)
        # -- an infeasible mode's "prediction" is a candidate with no
        # physical meaning and shouldn't be shown alongside real ones.
        feasibility = scene.get('turn_feasibility', None)
        if feasibility is not None:
            ego_feas = separate_ego_agent(feasibility, ego_ids).squeeze(1)  # (B, M)
            ego_feas = ego_feas.bool().cpu().numpy()
        else:
            ego_feas = np.ones(ego_probs.shape, dtype=bool)

        sequences = scene['sequences']
        agent_masks = scene['agent_masks']
        agent_types_flat = scene['agent_types'].reshape(sequences.shape[0], -1)
        scene_files = scene.get('scene_file')

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

        b = 0  # single-item batch now; was `target_sample` into the full batch
        ego_id = ego_ids[b]
        gt_mode = int(ego_true_mode[b].item())
        argmax_mode = int(ego_probs[b].argmax().item())

        # Same heading _mode_candidate_trajectories_abs uses to rotate mu
        # into the absolute frame -- recorded so sigma_xy (left in the
        # RAW, un-rotated frame below) can be correctly rotated later.
        start_heading_deg = float(
            sequences[b, ego_id, hist_len - 1, G.HD].detach().cpu().item())

        real_ego_xy = sequences[b, ego_id, hist_len:, G.XY].detach().cpu().numpy()  # (T_pred, 2)

        if ref_agent_idx_override is not None:
            ref_agent_idx = ref_agent_idx_override
        else:
            # Auto-detect: whichever agent achieved the closest approach
            # in the REALIZED trajectory (same quantity
            # find_cases_two_stage.py's true_min_sep_gt_agent_type
            # identifies the type of, resolved here to its raw index).
            A_total = sequences.shape[1]
            other_idx, other_xy, other_valid = [], [], []
            for a in range(A_total):
                if a == ego_id:
                    continue
                valid_a = agent_masks[b, a, hist_len:].bool().detach().cpu().numpy()
                if not valid_a.any():
                    continue
                other_idx.append(a)
                other_xy.append(sequences[b, a, hist_len:, G.XY].detach().cpu().numpy())
                other_valid.append(valid_a)
            if not other_idx:
                raise RuntimeError(
                    f"No valid other agent in batch={target_batch} sample={target_sample} "
                    "to auto-detect a reference agent from.")
            other_xy_arr = np.stack(other_xy, axis=0)
            other_valid_arr = np.stack(other_valid, axis=0)
            dist = np.linalg.norm(other_xy_arr - real_ego_xy[None, :, :], axis=-1)
            dist = np.where(other_valid_arr, dist, np.inf)
            agent_i, _t_i = np.unravel_index(np.argmin(dist), dist.shape)
            ref_agent_idx = other_idx[agent_i]

        ref_xy = sequences[b, ref_agent_idx, hist_len:, G.XY].detach().cpu().numpy()  # (T_pred, 2)
        ref_valid = agent_masks[b, ref_agent_idx, hist_len:].bool().detach().cpu().numpy()
        ref_type = int(agent_types_flat[b, ref_agent_idx].item())

        real_dist = np.linalg.norm(real_ego_xy - ref_xy, axis=-1)
        real_dist = np.where(ref_valid, real_dist, np.nan) * 1000  # km -> m

        airport = airport_ids[b] if airport_ids is not None else None
        ref = load_airport_ref(cfg.paths.assets_dir, airport)

        def _agent_latlon(a_idx):
            xy = sequences[b, a_idx, :, G.XY].detach().cpu().numpy()          # (T_total, 2)
            valid = agent_masks[b, a_idx, :].bool().detach().cpu().numpy()    # (T_total,)
            xy_masked = np.where(valid[:, None], xy, np.nan)
            return {
                "agent_idx": a_idx,
                "agent_type": AGENT_TYPE_NAMES.get(int(agent_types_flat[b, a_idx].item())),
                "latlon_hist": xy_array_to_latlon(xy_masked[:hist_len], ref),
                "latlon_fut": xy_array_to_latlon(xy_masked[hist_len:], ref),
            }

        A = sequences.shape[1]
        gt_out = {
            "ego": _agent_latlon(ego_id),
            "ref_agent": _agent_latlon(ref_agent_idx),
            "other_agents": [
                _agent_latlon(a) for a in range(A)
                if a not in (ego_id, ref_agent_idx)
                and agent_masks[b, a, :].bool().any().item()
            ],
        }

        M, K = traj_abs.shape[0], traj_abs.shape[1]
        modes_out = {}
        for m in range(M):
            cands = []
            for k in range(K):
                cand_xy = traj_abs[m, k, b]  # (T_pred, 2)
                dist = np.linalg.norm(cand_xy - ref_xy, axis=-1)
                dist = np.where(ref_valid, dist, np.nan) * 1000  # km -> m
                cand_xy_masked = np.where(ref_valid[:, None], cand_xy, np.nan)
                cand_sigma = ego_sigma[b, hist_len:, m, k, :2].detach().cpu().numpy()  # (T_pred, 2)
                cands.append({
                    "prob": float(cand_probs[b, m, k]),
                    "dist_m": [None if np.isnan(v) else float(v) for v in dist],
                    "latlon": xy_array_to_latlon(cand_xy_masked, ref),
                    "sigma_xy": [[float(sx), float(sy)] for sx, sy in cand_sigma],
                })
            modes_out[MODE_NAMES[m]] = {
                "mode_prob": float(ego_probs[b, m].item()),
                "feasible": bool(ego_feas[b, m]),
                "candidates": cands,
            }

        out = {
            "meta": {
                "airport": airport,
                "batch_idx": target_batch,
                "sample_idx": target_sample,
                "scene_file": scene_files[b] if scene_files is not None else None,
                "ego_id": ego_id,
                "ref_agent_idx": ref_agent_idx,
                "ref_agent_type": AGENT_TYPE_NAMES.get(ref_type, str(ref_type)),
                "mode_names": MODE_NAMES,
                "gt_mode": MODE_NAMES[gt_mode] if 0 <= gt_mode < len(MODE_NAMES) else gt_mode,
                "argmax_mode": MODE_NAMES[argmax_mode],
                "hist_len": hist_len,
                "start_heading_deg": start_heading_deg,
            },
            "gt": gt_out,
            "modes": modes_out,
            "real": {"dist_m": [None if np.isnan(v) else float(v) for v in real_dist]},
        }
        with open(output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[case_risk_dynamics] wrote {output_json}")


if __name__ == "__main__":
    main()
