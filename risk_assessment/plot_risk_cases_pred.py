"""
Bulk-renders risk-critical candidates as static PNGs WITH the two-stage
model's per-mode candidate predictions overlaid -- plot_risk_cases.py's
ground-truth-only screening can miss a case whose real trajectory looks
clean but whose predictions are messy/divergent (or vice versa), so this
covers what that one can't without a model.

Needs the model (GPU), unlike plot_risk_cases.py: reuses find_cases_two_
stage.py/case_risk_dynamics.py's own model-loading and per-(mode,
candidate) absolute-trajectory computation, generalized from "one hand-
picked (batch,sample)" to "every (batch,sample) pair a CSV's near-miss
filter selects", drawn on the SAME real background+heading-rotated-icon
infrastructure plot_risk_cases.py already uses (amelia_scenes.
visualization.common), plus the predicted candidate lines this repo's own
scene_viz code has no working equivalent for (its 'marginal_pred'/
'benchmark_pred' paths need a different scene/prediction shape and were
never actually exercised -- see this session's other notes).

Cross-model (STGCNN/Amelia-baseline) predictions are NOT drawn here --
that's what the full case_trajectories_stgcnn.py/case_trajectories_
amelia_baseline.py pipeline (+ the interactive artifact) is for, once a
shortlist is picked from this pass.

Usage (reuses eval_two_stage.yaml's data/paths/model composition, same as
find_cases_two_stage.py/case_risk_dynamics.py):

    python -m risk_assessment.plot_risk_cases_pred \\
        ckpt=kbos2 data=kbos.yaml \\
        mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \\
        traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \\
        model.traj_net.config.num_hypotheses=4 \\
        +model.traj_net.config.decoder.enable_score_head=true \\
        +model.traj_net.config.decoder.score_mode=5 \\
        +model.traj_net.config.decoder.score_head_type=attention \\
        +scorer.score_head_load='/gpfs/.../per_mode_scorer_hard.pt' \\
        +input_csv=/gpfs/.../kbos_50_4T_cases_ranked.csv \\
        +out_dir=/gpfs/.../risk_assessment/out/case_screenshots_pred/kbos \\
        +max_sep_km=0.05 +limit_batches=390
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TWO_STAGE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main_two_phases_4T")
if _TWO_STAGE_REPO not in sys.path:
    sys.path.insert(0, _TWO_STAGE_REPO)
os.environ.setdefault("PROJECT_ROOT", _TWO_STAGE_REPO)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from amelia_tf.eval_two_stage import _build_nets, _find_gmm, _SCORER_ATTRS
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODES
from amelia_scenes.utils.transform_utils import inv_transform_batch
from amelia_scenes.utils.dataset import load_assets
from amelia_scenes.visualization import common as C

from risk_assessment.common import (
    to_device, seed_for_reproducible_ego_selection, load_airport_ref, xy_array_to_latlon,
)

MODE_NAMES = TURN_MODES
# Same validated categorical palette used in the interactive artifact --
# fixed hue per mode, not cycled.
MODE_COLORS = {
    "TurnLeft": "#2a78d6", "TurnRight": "#eb6834",
    "Straight": "#1baf7a", "Hold": "#8a5fbf",
}

_ASSET_CACHE = {}


def _get_assets(base_dir, airport):
    if airport not in _ASSET_CACHE:
        _ASSET_CACHE[airport] = load_assets(base_dir, airport)
    return _ASSET_CACHE[airport]


def _mode_candidate_trajectories_abs(ego_mu, sequences, ego_ids, hist_len):
    """Same as find_cases_two_stage.py/case_risk_dynamics.py's helper."""
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


def _bearing(p1, p2):
    """Compass bearing (deg, cw from north) from lat/lon p1 -> p2, flat-earth
    (fine over a few hundred metres)."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    mid_lat_rad = np.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * np.cos(mid_lat_rad)
    dy = (lat2 - lat1)
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def _plot_case(out_path, airport, assets, ego_hist_ll, ego_fut_ll, other_agents_ll,
               modes_out, gt_mode, argmax_mode, dpi=150):
    bkg, hold_lines, graph_nx, limits, agent_icons = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits

    # Crop bounds: ego's own hist+fut track, +200m padding (same convention
    # as plot_risk_cases.py's _compute_crop).
    all_pts = [p for p in (ego_hist_ll + ego_fut_ll) if p is not None]
    if not all_pts:
        return False
    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mid_lat = (min_lat + max_lat) / 2.0
    pad_lat = 200.0 / 111320.0
    pad_lon = 200.0 / (111320.0 * np.cos(np.radians(mid_lat)))
    crop_west, crop_east = min_lon - pad_lon, max_lon + pad_lon
    crop_south, crop_north = min_lat - pad_lat, max_lat + pad_lat

    img_h, img_w = bkg.shape[0], bkg.shape[1]

    def _lat_to_row(lat):
        return int(round((north - lat) / (north - south) * img_h))

    def _lon_to_col(lon):
        return int(round((lon - west) / (east - west) * img_w))

    r0, r1 = sorted([_lat_to_row(crop_north), _lat_to_row(crop_south)])
    c0, c1 = sorted([_lon_to_col(crop_west), _lon_to_col(crop_east)])
    r0, r1 = max(0, r0), min(img_h, r1)
    c0, c1 = max(0, c0), min(img_w, c1)
    bkg_draw = bkg[r0:r1, c0:c1] if r1 > r0 and c1 > c0 else bkg
    extent_draw = [crop_west, crop_east, crop_south, crop_north] if r1 > r0 and c1 > c0 \
        else [west, east, south, north]

    fig, ax = plt.subplots()
    ax.imshow(bkg_draw, zorder=0, extent=extent_draw, alpha=0.3)

    def _plot_track(hist_ll, fut_ll, color, lw, icon_zoom_scale=None):
        hist_pts = [p for p in hist_ll if p is not None]
        fut_pts = [p for p in fut_ll if p is not None]
        if hist_pts:
            hp = np.array(hist_pts)
            ax.plot(hp[:, 1], hp[:, 0], color=color, lw=lw, ls='dashed', alpha=0.6)
        bridged = ([hist_pts[-1]] + fut_pts) if hist_pts else fut_pts
        if bridged:
            bp = np.array(bridged)
            ax.plot(bp[:, 1], bp[:, 0], color=color, lw=lw, ls='solid', alpha=1.0)
        if icon_zoom_scale and (hist_pts or fut_pts):
            anchor = hist_pts[-1] if hist_pts else fut_pts[0]
            heading_pts = hist_pts if len(hist_pts) >= 2 else (hist_pts + fut_pts)
            heading = _bearing(heading_pts[-2], heading_pts[-1]) if len(heading_pts) >= 2 else 0.0
            icon = agent_icons[C.AIRCRAFT]
            img = C.plot_agent(icon, heading, zoom=C.ZOOM[C.AIRCRAFT] * icon_zoom_scale,
                                native_bearing=C.AIRCRAFT_NOSE_BEARING)
            from matplotlib.offsetbox import AnnotationBbox
            ab = AnnotationBbox(img, (anchor[1], anchor[0]), frameon=False)
            ax.add_artist(ab)

    for hist_ll, fut_ll in other_agents_ll:
        _plot_track(hist_ll, fut_ll, C.MOTION_COLORS['other_agent'][0], 1.0)
    for mode_name, mode_info in modes_out.items():
        if not mode_info["feasible"]:
            continue
        color = MODE_COLORS.get(mode_name, "#888888")
        for cand in sorted(mode_info["candidates"], key=lambda c: c["prob"]):
            pts = [p for p in cand["latlon"] if p is not None]
            if not pts:
                continue
            arr = np.array(pts)
            ax.plot(arr[:, 1], arr[:, 0], color=color, lw=1.2 + cand["prob"] * 1.8,
                    alpha=0.25 + 0.65 * cand["prob"])
    _plot_track(ego_hist_ll, ego_fut_ll, "#000000", 2.2, icon_zoom_scale=15.0)

    title = f"{airport.upper()}  gt={gt_mode}  argmax={argmax_mode}"
    ax.set_title(title, fontsize=9)
    ax.set_xlim(crop_west, crop_east)
    ax.set_ylim(crop_south, crop_north)
    C.save(ax, out_path, dpi, tight=False)
    return True


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main_two_phases_4T/configs",
            config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    input_csv = cfg.get("input_csv")
    out_dir = cfg.get("out_dir")
    if not input_csv or not out_dir:
        raise ValueError("Pass +input_csv=... +out_dir=... on the command line.")
    max_sep_km = float(cfg.get("max_sep_km", 0.05))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[
        (df["true_min_sep_gt_agent_type"] == "Aircraft")
        & (df["true_min_sep_gt_on_road"] == True)
        & (df["true_min_sep_gt"] < max_sep_km)
    ]
    limit = cfg.get("limit")
    if limit:
        df = df.sort_values("true_min_sep_gt").head(int(limit))
    wanted = {}
    for _, row in df.iterrows():
        wanted.setdefault(int(row["batch_idx"]), []).append(int(row["sample_idx"]))
    if not wanted:
        print("[plot_risk_cases_pred] no candidates matched the filter, nothing to do")
        return
    max_wanted_batch = max(wanted)
    print(f"[plot_risk_cases_pred] {sum(len(v) for v in wanted.values())} candidates "
          f"across {len(wanted)} batches (up to batch {max_wanted_batch})")

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
        print(f"[plot_risk_cases_pred] loaded trained score head from {sh_path}")

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

    n_ok, n_fail = 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > max_wanted_batch:
                break
            if batch_idx not in wanted:
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

            ego_probs = separate_ego_agent(mode_probs, ego_ids).squeeze(1)
            ego_mu = separate_ego_agent(traj_mu, ego_ids).squeeze(1)
            rule_based = scene.get('rule_based_encoding')
            true_mode_idx = rule_based[..., :4].float().argmax(dim=-1).long()
            ego_true_mode = separate_ego_agent(true_mode_idx, ego_ids).squeeze(1)

            feasibility = scene.get('turn_feasibility', None)
            if feasibility is not None:
                ego_feas = separate_ego_agent(feasibility, ego_ids).squeeze(1).bool().cpu().numpy()
            else:
                ego_feas = np.ones(ego_probs.shape, dtype=bool)

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
                cand_probs = torch.softmax(score_fut, dim=-1).detach().cpu().numpy()
            else:
                B_, M_, K_ = ego_probs.shape[0], ego_probs.shape[1], ego_mu.shape[3]
                cand_probs = np.ones((B_, M_, K_), dtype=np.float64) / K_

            traj_abs = _mode_candidate_trajectories_abs(ego_mu, sequences, ego_ids, hist_len)
            A_total = sequences.shape[1]

            for b in wanted[batch_idx]:
                try:
                    ego_id = ego_ids[b]
                    airport = airport_ids[b] if airport_ids is not None else None
                    ref = load_airport_ref(cfg.paths.assets_dir, airport)

                    def _agent_latlon(a_idx):
                        xy = sequences[b, a_idx, :, G.XY].detach().cpu().numpy()
                        valid = agent_masks[b, a_idx, :].bool().detach().cpu().numpy()
                        xy_masked = np.where(valid[:, None], xy, np.nan)
                        return (xy_array_to_latlon(xy_masked[:hist_len], ref),
                                xy_array_to_latlon(xy_masked[hist_len:], ref))

                    ego_hist_ll, ego_fut_ll = _agent_latlon(ego_id)
                    other_agents_ll = [
                        _agent_latlon(a) for a in range(A_total)
                        if a != ego_id and agent_masks[b, a, :].bool().any().item()
                    ]

                    gt_mode = MODE_NAMES[int(ego_true_mode[b].item())]
                    argmax_mode = MODE_NAMES[int(ego_probs[b].argmax().item())]

                    modes_out = {}
                    for m in range(len(MODE_NAMES)):
                        cands = []
                        for k in range(traj_abs.shape[1]):
                            cand_xy = traj_abs[m, k, b]
                            cands.append({
                                "prob": float(cand_probs[b, m, k]),
                                "latlon": xy_array_to_latlon(cand_xy, ref),
                            })
                        modes_out[MODE_NAMES[m]] = {
                            "feasible": bool(ego_feas[b, m]),
                            "candidates": cands,
                        }

                    sep_m = float(df[(df.batch_idx == batch_idx) & (df.sample_idx == b)]
                                  ["true_min_sep_gt"].iloc[0]) * 1000.0
                    fname = os.path.join(out_dir, f"{airport}_b{batch_idx}_s{b}_sep{sep_m:.1f}m.png")
                    ok = _plot_case(fname, airport, _get_assets(cfg.paths.base_dir, airport),
                                     ego_hist_ll, ego_fut_ll, other_agents_ll, modes_out,
                                     gt_mode, argmax_mode)
                    if ok:
                        n_ok += 1
                    else:
                        n_fail += 1
                        print(f"[plot_risk_cases_pred] no valid ego points for batch={batch_idx} sample={b}")
                except Exception as e:
                    n_fail += 1
                    print(f"[plot_risk_cases_pred] FAILED batch={batch_idx} sample={b}: {e}")

            if batch_idx % 20 == 0:
                print(f"[plot_risk_cases_pred] scanned batch {batch_idx}, {n_ok} plotted so far")

    print(f"[plot_risk_cases_pred] done: {n_ok} plotted, {n_fail} failed, output in {out_dir}")


if __name__ == "__main__":
    main()
