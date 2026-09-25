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
import random

import numpy as np
import torch

SAFETY_MARGIN_KM = 0.05     # ~50 m; see module docstring for the FAA/Pang-et-al-2026-informed rationale
AMBIGUITY_MARGIN = 0.10     # top1-top2 probability gap, below which "ambiguous"
RELEVANCE_RADIUS_KM = 1.0   # scene must have another valid agent within this to count as "relevant"


def seed_for_reproducible_ego_selection(datamodule, seed=42):
    """
    amelia_dataset.py's transform_scene_data draws the ego agent for each
    sample via plain `random.randint()` (random_ego=True is the default and is
    never overridden for eval/test anywhere in any of the three model repos)
    -- with no seeding at all, that draw is OS-entropy-seeded and different
    every run, AND different across separate model scripts (each is an
    independent process). Two consequences that matter for this framework
    specifically: (1) a flagged case can't be re-located later for a case
    study without ALSO recording ego_id (see the row dict in every adapter
    here) since re-running won't reproduce the same draw; (2) comparing
    naive/prob_weighted/etc. risk ACROSS models for "the same" (batch_idx,
    sample_idx) is only a fair comparison if every model's run resolved that
    index to the same underlying (scene, ego agent) pair, which nothing
    guarantees without an explicit, controlled seed.

    Unlike eval_two_stage.py/train_stgcnn_*.py (which call
    `L.seed_everything(cfg.seed, workers=True)` before handing the DataLoader
    to a Trainer, which is what lets Lightning re-seed each worker
    subprocess deterministically from cfg.seed regardless of how much
    randomness model instantiation consumed beforehand in the main process),
    every adapter in this folder bypasses the Trainer entirely and therefore
    never gets that reproducibility machinery for free. Rather than
    reimplementing Lightning's per-worker seeding here, this forces
    num_workers=0 (data loading happens in THIS process, no fork-inherited or
    independently-OS-seeded worker RNG state to reason about) and seeds
    random/numpy/torch directly, immediately before the dataset is touched --
    slower than parallel loading, but every adapter run with the same seed
    now draws the identical, deterministic sequence of ego-agent choices,
    which is what actually matters for cross-model comparability here (this
    trades that guarantee for data-loading throughput deliberately).

    Call this AFTER building the model/loading checkpoints (so whatever
    randomness those steps consume doesn't matter) and BEFORE
    datamodule.prepare_data()/setup().
    """
    datamodule.eparams.num_workers = 0
    # test_dataloader()/val_dataloader() pass persistent_workers unconditionally
    # (unlike train_dataloader(), which guards it behind `if num_workers > 0`)
    # -- PyTorch's DataLoader raises ValueError("persistent_workers option
    # needs num_workers > 0") if that's left True here.
    datamodule.eparams.persistent_workers = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def min_separation_with_type(ego_traj_abs, other_xy, other_valid, other_types):
    """
    Same as min_separation, but also identifies WHICH other agent achieved
    the minimum and returns its role type (amelia_tf.utils.global_masks.
    AGENT_TYPES: 0=Aircraft, 1=Vehicle, 2=Unknown). Ground service vehicles
    legitimately operate within sub-metre distance of a gate-adjacent
    aircraft as routine, non-hazardous ground ops -- a "near miss" whose
    closest agent turns out to be a Vehicle is very likely not the
    aircraft-aircraft conflict a risk-assessment case study wants to show,
    so this is meant for screening candidate cases, not for the aggregate
    per-hypothesis risk_score computation (which stays type-agnostic, per
    the single-unified-threshold decision -- see this module's docstring).

    other_types: (num_others,) int array, one role-type code per other agent
        (matching other_xy/other_valid's ordering).

    Returns (min_dist, closest_agent_type) -- closest_agent_type is None if
    there's no valid other agent at all (min_dist is inf in that case too).
    """
    if other_xy.shape[0] == 0:
        return float("inf"), None
    dist = np.linalg.norm(other_xy - ego_traj_abs[None, :, :], axis=-1)  # (num_others, T_pred)
    dist = np.where(other_valid, dist, np.inf)
    if not np.isfinite(dist).any():
        return float("inf"), None
    agent_i, _t_i = np.unravel_index(np.argmin(dist), dist.shape)
    return float(dist[agent_i].min()), int(other_types[agent_i])


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
