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
import json
import os
import random

import numpy as np
import torch
from geographiclib.geodesic import Geodesic

SAFETY_MARGIN_KM = 0.05     # ~50 m; see module docstring for the FAA/Pang-et-al-2026-informed rationale
AMBIGUITY_MARGIN = 0.10     # top1-top2 probability gap, below which "ambiguous"
RELEVANCE_RADIUS_KM = 1.0   # scene must have another valid agent within this to count as "relevant"

# Zhang et al. 2022 (zhang2022airport)'s own paired threshold for their
# Monte Carlo separation-violation formulation (FAA ATC Terminal Spacing/
# Sequencing), used only when risk_method="mc" (see mc_violation_risk) --
# kept separate from SAFETY_MARGIN_KM (the deterministic margin proxy's own
# 50 m) rather than replacing it, per the 2026-09-28 decision to pair each
# methodology with its own threshold instead of mixing them.
MC_THRESHOLD_KM_200FT = 200 * 0.3048 / 1000.0  # ~0.06096 km
MC_SAMPLES_DEFAULT = 100    # see mc_violation_risk's docstring for why this is 10x smaller than the case study's S=1000


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

    Returns (min_dist, closest_agent_type, t_idx) -- t_idx is the index into
    ego_traj_abs's own time axis at which the minimum occurred (so the
    caller can look up ego_traj_abs[t_idx] for an on/off-road location
    check, see check_on_road()). closest_agent_type and t_idx are both None
    if there's no valid other agent at all (min_dist is inf in that case too).
    """
    if other_xy.shape[0] == 0:
        return float("inf"), None, None
    dist = np.linalg.norm(other_xy - ego_traj_abs[None, :, :], axis=-1)  # (num_others, T_pred)
    dist = np.where(other_valid, dist, np.inf)
    if not np.isfinite(dist).any():
        return float("inf"), None, None
    agent_i, t_i = np.unravel_index(np.argmin(dist), dist.shape)
    return float(dist[agent_i, t_i]), int(other_types[agent_i]), int(t_i)


def mc_violation_risk(ego_traj_abs, ego_sigma_rel, heading_deg, other_xy, other_valid,
                       threshold_km=MC_THRESHOLD_KM_200FT, num_samples=MC_SAMPLES_DEFAULT, rng=None):
    """
    Monte Carlo separation-violation probability, following Zhang et al.
    2022's Eq. 9 -- the probabilistic analogue of min_separation, returning
    a probability in [0, 1] instead of a raw distance (smaller distance in
    min_separation <=> larger risk here, so where min_separation takes the
    MIN over (other agent, timestep) pairs, this takes the MAX). Only the
    ego side is sampled (unlike Zhang et al.'s two-sided sampling), since
    only ego has a predicted distribution in this pipeline -- the other
    agent's position is its real, deterministic trajectory.

    ego_traj_abs: (T_pred, 2) absolute mu, same frame (this repo's G.XY
        local-XY km convention) as other_xy -- i.e. the SAME array
        min_separation would take.
    ego_sigma_rel: (T_pred, 2) sigma_x, sigma_y -- RAW, in the same
        egocentric/heading-relative frame ego_traj_abs's own mu was in
        BEFORE inv_transform/inv_transform_batch's rotation (confirmed a
        standard deviation, not a variance: traj_pred_combined.py's
        training loss passes sigma**2 into F.gaussian_nll_loss's variance
        argument). Do not pass an already-rotated/absolute-frame sigma.
    heading_deg: scalar, the SAME theta inv_transform/inv_transform_batch
        used to rotate mu into ego_traj_abs -- applying the identical
        rotation here lands the sampled offset in ego_traj_abs's own frame.
    other_xy: (num_others, T_pred, 2).
    other_valid: (num_others, T_pred) bool.
    threshold_km: separation-violation threshold (200 ft by default -- see
        MC_THRESHOLD_KM_200FT's docstring for why this differs from
        SAFETY_MARGIN_KM).
    num_samples: S. Defaults to 10x fewer than the case-study's S=1000
        (MC_SAMPLES_DEFAULT=100): an aggregate statistic over hundreds of
        thousands of rows already averages away each row's own per-sample
        MC noise, so the extra precision a full S=1000 buys per row isn't
        needed here, and this keeps a full-test-set run's compute cost an
        order of magnitude lower. Pass num_samples=1000 to match the case
        study exactly if that's ever needed instead.
    rng: a numpy.random.Generator (caller-owned, e.g. one Generator built
        once per process and reused across every row, for reproducibility
        the same way seed_for_reproducible_ego_selection's seeding is
        reused). Falls back to a fresh, unseeded Generator if omitted.

    Returns 0.0 if there is no valid other agent at all (matches
    min_separation's inf-distance / has_nearby_agent gate).
    """
    if other_xy.shape[0] == 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng()
    T_pred = ego_traj_abs.shape[0]
    theta = np.radians(heading_deg)
    ct, st = np.cos(theta), np.sin(theta)
    sx = ego_sigma_rel[:, 0]  # (T_pred,)
    sy = ego_sigma_rel[:, 1]
    z = rng.standard_normal((num_samples, T_pred, 2))
    # Same R(theta) @ diag(sigma_x, sigma_y) @ z rotation inv_transform
    # applies to mu, applied here to a raw (sigma_x*z0, sigma_y*z1) offset
    # instead -- see this function's own docstring for why ego_sigma_rel
    # must be the pre-rotation sigma.
    off_x = ct * sx[None, :] * z[..., 0] - st * sy[None, :] * z[..., 1]  # (S, T_pred)
    off_y = st * sx[None, :] * z[..., 0] + ct * sy[None, :] * z[..., 1]  # (S, T_pred)
    sample_x = ego_traj_abs[None, :, 0] + off_x  # (S, T_pred)
    sample_y = ego_traj_abs[None, :, 1] + off_y  # (S, T_pred)

    dx = sample_x[None, :, :] - other_xy[:, None, :, 0]  # (num_others, S, T_pred)
    dy = sample_y[None, :, :] - other_xy[:, None, :, 1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    violated = dist < threshold_km
    prob = violated.mean(axis=1)  # (num_others, T_pred) -- fraction of S samples that violate
    if not other_valid.any():
        return 0.0
    return float(prob[other_valid].max())


def select_candidates(df, criterion="near_miss", max_sep_km=SAFETY_MARGIN_KM):
    """
    Filters+sorts an already-computed find_cases_two_stage.py CSV for bulk
    case-study screening (plot_risk_cases.py/plot_risk_cases_pred.py). Both
    criteria require the ego's realized closest-approach agent to be an
    Aircraft on the movement-area network -- the same legibility gate every
    case-study candidate needs, regardless of which story it's telling.

    "near_miss" (default): true_min_sep_gt < max_sep_km -- a genuine close
        REAL approach. Sorted tightest-first.
    "group_b": gt_mode == argmax_mode (mode predicted correctly) AND
        risk_naive == 0 (the top/gt-mode prediction is itself safe) AND
        risk_worst_case > 0 (some OTHER mode/candidate lands inside the
        safety margin) -- the "risk_worst_case looks beyond the single
        most-likely prediction" story. Sorted by risk_worst_case
        descending (the most dramatic worst-case-vs-reality gap first).
    """
    base = (df["true_min_sep_gt_agent_type"] == "Aircraft") & (df["true_min_sep_gt_on_road"] == True)
    if criterion == "near_miss":
        return df[base & (df["true_min_sep_gt"] < max_sep_km)].sort_values("true_min_sep_gt")
    if criterion == "group_b":
        return df[
            base & (df["gt_mode"] == df["argmax_mode"])
            & (df["risk_naive"] == 0) & (df["risk_worst_case"] > 0)
        ].sort_values("risk_worst_case", ascending=False)
    raise ValueError(f"unknown criterion: {criterion!r} (expected 'near_miss' or 'group_b')")


def resolve_case_scene_file(cfg):
    """
    Returns the scene_file for a case_trajectories_*.py cross-model run.
    Accepts EITHER a direct +case_scene_file=..., or +case_batch_idx=X
    +case_sample_idx=Y +cases_csv=/path/to/{airport}_*_cases_ranked.csv
    (the same ranked CSV find_cases_two_stage.py/rank_candidates.py write,
    with batch_idx/sample_idx/scene_file columns) -- lets a case be named by
    the two-stage repo's own (batch_idx, sample_idx), which is what a
    human picks from screenshots, instead of requiring the scene_file to be
    looked up by hand first.
    """
    scene_file = cfg.get("case_scene_file")
    if scene_file:
        return scene_file
    batch_idx = cfg.get("case_batch_idx")
    sample_idx = cfg.get("case_sample_idx")
    cases_csv = cfg.get("cases_csv")
    if batch_idx is None or sample_idx is None or not cases_csv:
        raise ValueError(
            "Pass +case_scene_file=<airport>/<day>/<scenario>.pkl, or "
            "+case_batch_idx=X +case_sample_idx=Y +cases_csv=/path/to/cases_ranked.csv"
        )
    import pandas as pd
    df = pd.read_csv(cases_csv)
    rows = df[(df["batch_idx"] == int(batch_idx)) & (df["sample_idx"] == int(sample_idx))]
    if len(rows) == 0:
        raise ValueError(f"No row with batch_idx={batch_idx} sample_idx={sample_idx} in {cases_csv}")
    return rows.iloc[0]["scene_file"]


def load_airport_ref(assets_dir, airport):
    """
    Returns (ref_lat, ref_lon, range_scale) from assets_dir/{airport}/
    limits.json -- the reference point every repo's local-XY frame (G.XY,
    km, this repo's range_scale convention) is defined relative to.
    """
    limits_path = os.path.join(assets_dir, airport, "limits.json")
    with open(limits_path) as f:
        d = json.load(f)
    return d["ref_lat"], d["ref_lon"], d["range_scale"]


def xy_array_to_latlon(xy, ref):
    """
    For map-overlay case-study plots: converts local-XY points (this repo's
    range_scale-km convention, e.g. from G.XY) to (lat, lon) -- the shared
    coordinate frame needed to plot trajectories from DIFFERENT model repos
    (each with its own local-XY frame) on the same airport map.

    xy: (N, 2) array, possibly containing NaN rows (invalid/padded
        timesteps) -- those become None in the output rather than a bogus
        lat/lon.
    ref: (ref_lat, ref_lon, range_scale) from load_airport_ref().

    Reimplements the same range+bearing geodesic conversion as
    amelia_scenes.utils.transform_utils.xy_to_ll/direct_wrapper directly
    (rather than importing it) so this module stays repo-agnostic -- it's
    imported by adapters for three different repos, each putting a
    different copy of amelia_scenes onto sys.path, and this avoids any
    dependency on which one happens to be active when this is called.
    """
    ref_lat, ref_lon, range_scale = ref
    geod = Geodesic.WGS84
    out = []
    for x, y in xy:
        if not (np.isfinite(x) and np.isfinite(y)):
            out.append(None)
            continue
        r = float(np.sqrt(x ** 2 + y ** 2)) * range_scale
        b = float(np.degrees(np.arctan2(y, x)))
        g = geod.Direct(ref_lat, ref_lon, b, r)
        out.append([g["lat2"], g["lon2"]])
    return out


def check_on_road(evaluator, xy_point, ref):
    """
    For case-screening: is this local-XY point on the airport's mapped
    taxiway/runway movement-area network, or off it (a proxy for "at a
    gate/apron/stand", which are not covered by that network -- same logic
    the Ch.3 turn-mode Hold-gate-proximity check used)? On-road + close to
    an edge corroborates a genuine movement-area event; off-road + far
    corroborates a routine gate/stand encounter (e.g. a Vehicle's legitimate
    close approach to a parked aircraft). Does not resolve Aircraft-vs-
    Vehicle identity, just gate-vs-movement-area location.

    evaluator: an amelia_tf.utils.off_road_evaluator.OffRoadEvaluator
        instance, duck-typed here (not imported -- this module stays
        repo-agnostic; each of the 3 repos has its own copy of that class).
    xy_point: (2,) local-XY point (this repo's range_scale-km convention).
    ref: (ref_lat, ref_lon, range_scale) from load_airport_ref().

    Returns (is_on_road: bool, dist_to_nearest_edge_m: float), or
    (None, None) if xy_point is invalid (NaN/padded) or the evaluator has
    no reference network loaded for this airport.
    """
    if getattr(evaluator, "reference_gdf", None) is None:
        return None, None
    x, y = xy_point
    if not (np.isfinite(x) and np.isfinite(y)):
        return None, None
    latlon = xy_array_to_latlon(np.array([[x, y]]), ref)[0]
    if latlon is None:
        return None, None
    lat, lon = latlon
    result = evaluator._evaluate_trajectory([(lon, lat)], safety_margin=1.0)
    is_off = result["per_point_is_off"][0]
    dist = result["per_point_distances"][0]
    return (not is_off), float(dist)


def has_nearby_agent(min_sep_flat, relevance_radius_km=RELEVANCE_RADIUS_KM):
    """min_sep_flat: (N,) min separation per hypothesis, for one sample."""
    finite = np.isfinite(min_sep_flat)
    if not finite.any():
        return False
    return bool(np.nanmin(np.where(finite, min_sep_flat, np.inf)) < relevance_radius_km)


def aggregate_risk(min_sep_flat, prob_flat, naive_idx, gate_pool_idx, ambiguous,
                    safety_margin_km=SAFETY_MARGIN_KM, risk_flat=None):
    """
    The four comparison strategies, computed uniformly over a flat list of N
    weighted hypotheses -- works the same whether N came from a two-stage
    model's mode x candidate grid, a plain multi-hypothesis baseline's raw H
    candidates, or a unimodal baseline's single (N=1) output.

    min_sep_flat: (N,) minimum separation distance per hypothesis. Only used
        to derive the deterministic margin-violation risk_flat when the
        caller doesn't already have one -- ignored if risk_flat is passed.
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
    risk_flat: (N,) pre-computed per-hypothesis risk, e.g. from
        mc_violation_risk (already a probability in [0, 1], not a distance)
        -- pass this to aggregate over Monte Carlo risk instead of the
        deterministic margin-violation proxy; min_sep_flat/safety_margin_km
        are then unused.

    Returns a dict: risk_naive, risk_prob_weighted, risk_worst_case,
    risk_gated, strategy_divergence.
    """
    if risk_flat is None:
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
