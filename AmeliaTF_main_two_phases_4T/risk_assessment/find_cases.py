"""
Contribution 3 (risk-aware downstream application) case-finding + scoring.

Finds test-set samples where a naive risk assessment (built from only the
top-predicted mode) would disagree with mitigated strategies -- probability-
weighted aggregation, worst-case (max-over-modes), and feasibility-based
confidence gating -- so those samples can be used either as hand-picked
case-study examples or as input to a statistical pass across the whole
test set.

Works with 1T/2T/4T checkpoints alike. Every (mode, candidate) pair is
treated uniformly as one weighted hypothesis: joint_prob[m,k] =
mode_prob[m] * scorer_cand_prob[k|m], where scorer_cand_prob is the same
masked-softmax-over-K quantity the scorer NLL fix uses (see
traj_pred_combined.py's test_step, sel_mode == 'score' branch) -- reused
here rather than re-derived, so this stays consistent with how the model
is actually deployed (hard mode-argmax, hard scorer-argmax-within-mode
for risk_naive; the full joint for risk_prob_weighted/risk_worst_case).
For a 1T checkpoint K=1 and this degenerates to the old mode-only
behaviour (softmax over a single candidate is trivially 1.0).

Risk proxy: minimum separation distance (same local-XY units as
amelia_tf.utils.global_masks.G.XY, which are kilometres per this repo's
range_scale convention) between ego's predicted future trajectory (for a
given mode/candidate) and every other valid agent's ground-truth future
trajectory, over the prediction horizon. A "conflict" is flagged when
this distance drops below SAFETY_MARGIN_KM. Continuous "risk score" per
hypothesis is max(0, SAFETY_MARGIN_KM - min_separation), i.e. zero when
clear of the margin and rising the closer/more-violating the approach is.

SAFETY_MARGIN_KM = 0.05 (50 m) is a single, unified threshold (not split
by agent type/size -- the dataset only exposes a coarse 3-way Aircraft/
Vehicle/Unknown role field, see amelia_tf.utils.global_masks.AGENT_TYPES,
no per-aircraft wingspan/model/ADG, so a size-adaptive radius a la Pang
et al. 2026's "mean-wingspan collision radius" isn't implementable here).
50 m is conservative relative to FAA AC 150/5300-13B taxiway-design
wingtip-clearance minima (~6-11 m across Aircraft Design Groups I-VI) and
the right order of magnitude for Pang et al. 2026's wingspan-based radius
for the aircraft mix in this dataset -- still sanity-check it against
real separation minima for the airports you're using before trusting the
case selection.

AMBIGUITY_MARGIN and RELEVANCE_RADIUS_KM are, likewise, starting points.

Usage (reuses configs/eval_two_stage.yaml's data/paths/model composition,
so all the usual data=/paths=/ckpt= overrides from the eval scripts in
the parent directory apply):

    python -m risk_assessment.find_cases \\
        ckpt=klax2 data=klax.yaml \\
        mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \\
        traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \\
        model.traj_net.config.num_hypotheses=1 \\
        +output_csv=/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/risk_assessment/klax_50_cases.csv

Output: one row per (sample in the test set that has at least one other
valid agent nearby), written to output_csv, with columns:
    airport, batch_idx, sample_idx, gt_mode, argmax_mode, mode_error,
    ambiguous, feasible_modes (comma-joined mode indices),
    min_sep_{mode} / risk_score_{mode} (for the scorer-selected candidate
    within that mode -- i.e. what deployment would actually produce),
    risk_naive, risk_prob_weighted, risk_worst_case, risk_gated,
    strategy_divergence, scene_min_sep_gt (ground-truth mode's selected
    candidate)

Filtering to case-study or statistical subsets is just pandas on this
CSV, e.g.:
    df[df.mode_error & df.ambiguous]                      # gating should matter here
    df[(df.risk_worst_case > 0) & (df.risk_naive == 0)]    # worst-case catches what naive misses
"""
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from amelia_tf.eval_two_stage import _build_nets
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_scenes.utils.transform_utils import inv_transform_batch

MODE_NAMES = ["Hold", "Straight", "TurnLeft", "TurnRight"]  # index order per rule_based_encoding[..., :4]

SAFETY_MARGIN_KM = 0.05     # ~50 m; see module docstring for the FAA/Pang-et-al-2026-informed rationale
AMBIGUITY_MARGIN = 0.10     # top1-top2 probability gap among feasible modes, below which "ambiguous"
RELEVANCE_RADIUS_KM = 1.0   # scene must have another valid agent within this to count as "relevant"


def _to_device(batch, device):
    sd = batch['scene_dict']
    for k, v in sd.items():
        if torch.is_tensor(v):
            sd[k] = v.to(device)
    return batch


def _mode_candidate_trajectories_abs(ego_mu, sequences, ego_ids, hist_len):
    """
    ego_mu: (B, T_total, M, K, D) relative predicted trajectories (ego only).
    sequences: (B, A, T_total, D_seq) absolute per-agent sequences (whole scene).
    ego_ids: list[int] of length B.
    Returns: (M, K, B, T_pred, 2) absolute XY per (mode, candidate).
    """
    B, T_total, M, K, D = ego_mu.shape
    T_pred = T_total - hist_len

    start_abs = np.stack([
        sequences[b, ego_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
        for b in range(B)
    ], axis=0)  # (B, 2)
    start_heading = np.array([
        float(sequences[b, ego_ids[b], hist_len - 1, G.HD].detach().cpu().numpy())
        for b in range(B)
    ])  # (B,)

    traj_abs = np.zeros((M, K, B, T_pred, 2), dtype=np.float64)
    for m in range(M):
        for k in range(K):
            future_rel = ego_mu[:, hist_len:, m, k, :2].detach().cpu().numpy()  # (B, T_pred, 2)
            traj_abs[m, k] = inv_transform_batch(future_rel, start_abs, start_heading)
    return traj_abs


def _min_separation(ego_traj_abs, other_xy, other_valid):
    """
    ego_traj_abs: (T_pred, 2)
    other_xy: (num_others, T_pred, 2)
    other_valid: (num_others, T_pred) bool
    Returns minimum separation distance over all valid (other agent, timestep)
    pairs, or np.inf if there is no valid other agent at all.
    """
    if other_xy.shape[0] == 0:
        return float("inf")
    dist = np.linalg.norm(other_xy - ego_traj_abs[None, :, :], axis=-1)  # (num_others, T_pred)
    dist = np.where(other_valid, dist, np.inf)
    finite = np.isfinite(dist)
    return float(dist[finite].min()) if finite.any() else float("inf")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    output_csv = cfg.get("output_csv")
    if not output_csv:
        raise ValueError("Pass +output_csv=/path/to/cases.csv on the command line.")

    mode_net, traj_net, device = _build_nets(cfg)
    model = CombinedTrajPredSystem(
        mode_model=mode_net, traj_model=traj_net, extra_params=cfg.model.extra_params)
    model.to(device)
    model.eval()

    hist_len = model.hist_len

    datamodule = hydra.utils.instantiate(cfg.data)
    # prepare_data() generates the per-run split-list files that setup() then
    # reads; normally the Trainer calls this automatically before setup(), but
    # this script bypasses the Trainer entirely for custom per-sample control.
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    # For a first sanity-check run before committing to the full test set:
    # +limit_batches=5 processes only the first 5 batches and exits, so you
    # can confirm the script runs end-to-end and inspect the CSV columns
    # (this whole script hasn't been executed anywhere yet -- verify it
    # actually works before trusting a full run).
    limit_batches = cfg.get("limit_batches")

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit_batches is not None and batch_idx >= limit_batches:
                print(f"[find_cases] stopping early: limit_batches={limit_batches}")
                break
            batch = _to_device(batch, device)
            scene = batch['scene_dict']

            mode_probs, traj_mu, traj_sigma, traj_score = model.forward(batch)

            ego_agent = scene['ego_agent_id']
            ego_ids = [
                ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                for b in range(mode_probs.shape[0])
            ]
            airport_ids = scene.get('airport_id')

            rule_based = scene.get('rule_based_encoding')
            true_mode_idx = rule_based[..., :4].float().argmax(dim=-1).long()  # (B, A)
            ego_true_mode = separate_ego_agent(true_mode_idx, ego_ids).squeeze(1)  # (B,)

            ego_probs = separate_ego_agent(mode_probs, ego_ids).squeeze(1)      # (B, M)
            ego_mu = separate_ego_agent(traj_mu, ego_ids).squeeze(1)            # (B, T_total, M, K, D)

            feasibility = scene.get('turn_feasibility', None)
            if feasibility is not None:
                ego_feas = separate_ego_agent(feasibility, ego_ids).squeeze(1)  # (B, M)
                ego_feas = ego_feas.bool().cpu().numpy()
            else:
                ego_feas = np.ones(ego_probs.shape, dtype=bool)

            sequences = scene['sequences']
            agent_masks = scene['agent_masks']
            B, A = sequences.shape[:2]
            M = ego_probs.shape[1]
            K = ego_mu.shape[3]

            # Per-(mode, candidate) probability, mirroring traj_pred_combined.py's
            # test_step (sel_mode == 'score' branch) exactly: mask-weighted mean of
            # the scorer's per-timestep score over the future horizon, then softmax
            # over K within each mode. For a 1T checkpoint (K=1, no score head
            # trained) traj_score is None and there is nothing to select between --
            # every mode's single candidate gets probability 1.
            if traj_score is not None:
                ego_score = separate_ego_agent(traj_score, ego_ids).squeeze(1)  # (B, T_total, M, K)
                ego_mask = separate_ego_agent(agent_masks, ego_ids).squeeze(1)  # (B, T_total)
                fut_mask = ego_mask[:, hist_len:].float()                       # (B, Tp)
                m_exp = fut_mask[:, :, None, None]                              # (B, Tp, 1, 1)
                denom = m_exp.sum(dim=1).clamp_min(1)                           # (B, 1, 1)
                s_pt = ego_score[:, hist_len:]                                  # (B, Tp, M, K)
                score_fut = (s_pt * m_exp).sum(dim=1) / denom                   # (B, M, K)
                cand_probs = torch.softmax(score_fut, dim=-1).detach().cpu().numpy()  # (B, M, K)
            else:
                cand_probs = np.ones((B, M, K), dtype=np.float64)

            traj_abs = _mode_candidate_trajectories_abs(
                ego_mu, sequences, ego_ids, hist_len)  # (M,K,B,T_pred,2)

            probs_np = ego_probs.detach().cpu().numpy()
            gt_mode_np = ego_true_mode.detach().cpu().numpy()

            for b in range(B):
                ego_id = ego_ids[b]
                other_xy, other_valid = [], []
                for a in range(A):
                    if a == ego_id:
                        continue
                    valid_a = agent_masks[b, a, hist_len:].bool().detach().cpu().numpy()
                    if not valid_a.any():
                        continue
                    xy_a = sequences[b, a, hist_len:, G.XY].detach().cpu().numpy()
                    other_xy.append(xy_a)
                    other_valid.append(valid_a)

                T_pred = traj_abs.shape[3]
                if other_xy:
                    other_xy_arr = np.stack(other_xy, axis=0)
                    other_valid_arr = np.stack(other_valid, axis=0)
                else:
                    other_xy_arr = np.zeros((0, T_pred, 2))
                    other_valid_arr = np.zeros((0, T_pred), dtype=bool)

                # (M, K) grid: every (mode, candidate) hypothesis is one weighted
                # trajectory, so risk is computed on all of them uniformly, not
                # just a single k=0 slice per mode.
                min_sep = np.array([
                    [_min_separation(traj_abs[m, k, b], other_xy_arr, other_valid_arr)
                     for k in range(K)]
                    for m in range(M)
                ])  # (M, K)
                risk_score = np.where(
                    np.isfinite(min_sep), np.maximum(0.0, SAFETY_MARGIN_KM - min_sep), 0.0)

                has_nearby_agent = bool(np.isfinite(min_sep).any() and np.nanmin(
                    np.where(np.isfinite(min_sep), min_sep, np.inf)) < RELEVANCE_RADIUS_KM)
                if not has_nearby_agent:
                    continue  # Filter 1: no scenario relevance, skip

                probs_b = probs_np[b]         # (M,)
                cand_probs_b = cand_probs[b]  # (M, K), sums to 1 over K within each mode
                joint_probs = probs_b[:, None] * cand_probs_b  # (M, K), sums to 1 overall
                selected_k = cand_probs_b.argmax(-1)  # (M,) -- as-deployed candidate per mode

                gt_mode = int(gt_mode_np[b])
                argmax_mode = int(probs_b.argmax())
                mode_error = argmax_mode != gt_mode

                feas_b = ego_feas[b]
                feas_idx = np.where(feas_b)[0]
                ambiguous = False
                if feas_idx.size >= 2:
                    feas_probs = np.sort(probs_b[feas_idx])[::-1]
                    ambiguous = bool((feas_probs[0] - feas_probs[1]) < AMBIGUITY_MARGIN)

                # Per-mode risk of the as-deployed (scorer-selected) candidate --
                # what min_sep_{mode}/risk_score_{mode} report, and what naive/
                # gated draw on, since neither strategy second-guesses the
                # scorer's own within-mode choice.
                risk_selected = risk_score[np.arange(M), selected_k]  # (M,)

                risk_naive = float(risk_selected[argmax_mode])
                risk_prob_weighted = float((joint_probs * risk_score).sum())
                risk_worst_case = float(risk_score.max())
                risk_gated = float(risk_selected[feas_idx].max()) if (ambiguous and feas_idx.size) \
                    else risk_naive

                strategy_divergence = bool(
                    (risk_worst_case > 0 and risk_naive == 0)
                    or abs(risk_prob_weighted - risk_naive) > 1e-6
                )

                row = {
                    "airport": airport_ids[b] if airport_ids is not None else None,
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "gt_mode": MODE_NAMES[gt_mode] if 0 <= gt_mode < len(MODE_NAMES) else gt_mode,
                    "argmax_mode": MODE_NAMES[argmax_mode],
                    "mode_error": mode_error,
                    "ambiguous": ambiguous,
                    "feasible_modes": ",".join(MODE_NAMES[i] for i in feas_idx),
                    "num_candidates": K,
                    "risk_naive": risk_naive,
                    "risk_prob_weighted": risk_prob_weighted,
                    "risk_worst_case": risk_worst_case,
                    "risk_gated": risk_gated,
                    "strategy_divergence": strategy_divergence,
                    "scene_min_sep_gt": (
                        float(min_sep[gt_mode, selected_k[gt_mode]])
                        if np.isfinite(min_sep[gt_mode, selected_k[gt_mode]]) else None),
                }
                for m in range(M):
                    row[f"min_sep_{MODE_NAMES[m]}"] = (
                        float(min_sep[m, selected_k[m]])
                        if np.isfinite(min_sep[m, selected_k[m]]) else None)
                    row[f"risk_score_{MODE_NAMES[m]}"] = float(risk_selected[m])
                rows.append(row)

            if batch_idx % 50 == 0:
                print(f"[find_cases] processed batch {batch_idx}, {len(rows)} relevant samples so far")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[find_cases] wrote {len(df)} rows to {output_csv}")
    print(f"[find_cases] mode_error rate: {df['mode_error'].mean():.3f}")
    print(f"[find_cases] ambiguous rate: {df['ambiguous'].mean():.3f}")
    print(f"[find_cases] strategy_divergence rate: {df['strategy_divergence'].mean():.3f}")


if __name__ == "__main__":
    main()
