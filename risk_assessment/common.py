"""
Shared risk-aggregation math for the Contribution 3 case-finding scripts
(find_cases_two_stage.py and, later, one adapter per baseline model). Each
model's own adapter is responsible for turning its particular output format
into a flat list of (trajectory, probability) hypotheses -- this module only
knows about that flat representation, not about "modes" or any other
model-specific structure, so it's reusable across every model unchanged.

SAFETY_MARGIN_KM = 0.05 (50 m) is a single, unified threshold (not split by
agent type/size -- the Amelia dataset only exposes a coarse 3-way Aircraft/
Vehicle/Unknown role field, see amelia_tf.utils.global_masks.AGENT_TYPES, no
per-aircraft wingspan/model/ADG, so a size-adaptive radius a la Pang et al.
2026's "mean-wingspan collision radius" isn't implementable here). 50 m is
conservative relative to FAA AC 150/5300-13B taxiway-design wingtip-clearance
minima (~6-11 m across Aircraft Design Groups I-VI) and the right order of
magnitude for Pang et al. 2026's wingspan-based radius for the aircraft mix
in this dataset -- still sanity-check it against real separation minima for
the airports you're using before trusting the case selection.

AMBIGUITY_MARGIN and RELEVANCE_RADIUS_KM are, likewise, starting points.
"""
import numpy as np
import torch

SAFETY_MARGIN_KM = 0.05     # ~50 m; see module docstring for the FAA/Pang-et-al-2026-informed rationale
AMBIGUITY_MARGIN = 0.10     # top1-top2 probability gap, below which "ambiguous"
RELEVANCE_RADIUS_KM = 1.0   # scene must have another valid agent within this to count as "relevant"


def to_device(batch, device):
    sd = batch['scene_dict']
    for k, v in sd.items():
        if torch.is_tensor(v):
            sd[k] = v.to(device)
    return batch


def min_separation(ego_traj_abs, other_xy, other_valid):
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


def has_nearby_agent(min_sep_flat, relevance_radius_km=RELEVANCE_RADIUS_KM):
    """min_sep_flat: (N,) min separation per hypothesis, for one sample."""
    finite = np.isfinite(min_sep_flat)
    if not finite.any():
        return False
    return bool(np.nanmin(np.where(finite, min_sep_flat, np.inf)) < relevance_radius_km)


def aggregate_risk(min_sep_flat, prob_flat, naive_idx, gate_pool_idx, ambiguous,
                    safety_margin_km=SAFETY_MARGIN_KM):
    """
    The four comparison strategies, computed uniformly over a flat list of N
    weighted hypotheses -- works the same whether N came from a two-stage
    model's mode x candidate grid, a plain multi-hypothesis baseline's raw H
    candidates, or a unimodal baseline's single (N=1) output.

    min_sep_flat: (N,) minimum separation distance per hypothesis.
    prob_flat: (N,) probability per hypothesis; expected to sum to ~1.
    naive_idx: int, the index of the "as-deployed" hypothesis (e.g.
        argmax(prob_flat), or a model-specific hard-selection index).
    gate_pool_idx: array of indices eligible for the gated max when
        ambiguous (e.g. all indices for a model with no feasibility concept,
        or only the feasible-mode indices for the two-stage model).
    ambiguous: bool, whether the top-1 vs top-2 probability gap was below
        AMBIGUITY_MARGIN (computed by the caller, since what counts as
        "top-1 vs top-2" differs: mode-level for two-stage, hypothesis-level
        for a flat multi-hypothesis baseline).

    Returns a dict: risk_naive, risk_prob_weighted, risk_worst_case,
    risk_gated, strategy_divergence.
    """
    risk_flat = np.where(
        np.isfinite(min_sep_flat), np.maximum(0.0, safety_margin_km - min_sep_flat), 0.0)

    risk_naive = float(risk_flat[naive_idx])
    risk_worst_case = float(risk_flat.max())
    risk_prob_weighted = float((prob_flat * risk_flat).sum())
    risk_gated = (
        float(risk_flat[gate_pool_idx].max())
        if (ambiguous and len(gate_pool_idx)) else risk_naive)

    strategy_divergence = bool(
        (risk_worst_case > 0 and risk_naive == 0)
        or abs(risk_prob_weighted - risk_naive) > 1e-6
    )

    return {
        "risk_naive": risk_naive,
        "risk_prob_weighted": risk_prob_weighted,
        "risk_worst_case": risk_worst_case,
        "risk_gated": risk_gated,
        "strategy_divergence": strategy_divergence,
    }
