"""
Measure how far the GROUND-TRUTH trajectory sits from the RUNWAY centerline,
using only points that are actually ON a runway. Runway motion is the cleanest
accuracy yardstick: aircraft on a runway move along the centerline with no path
choice, so deviation there reflects positioning/sensor noise rather than
behavioural variability. Industry practice is to use runway-centerline deviation
as the dataset's positioning-accuracy benchmark.

Requires a dataset that has NOT removed runway trajectories (runway points are
otherwise masked out). Loading is identical to the taxiway version.

Originally measured distance to the taxiway network. This tells us whether map matching is needed and, if so, for
which modes:

  - If GT trajectories hug the taxiways (small distance), the map is well
    aligned and deviation in predictions is a model problem, not a data problem;
    map matching of inputs won't help much.
  - If GT trajectories are far from taxiways (large distance), either the map is
    incomplete (e.g. aircraft move through apron/ramp not covered by taxiway
    polylines) or the coordinates are noisy -> map matching / better map
    coverage could help.

Uses the raw sequence xy (same local frame as the taxiway polylines), so no
rel<->local conversion is needed. Distance is point-to-nearest-taxiway-segment,
averaged over the future horizon, then aggregated per true mode.
"""
import numpy as np
import hydra
import torch
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from amelia_tf import utils
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G

log = utils.get_pylogger(__name__)
NAMES = ["TurnLeft", "TurnRight", "Straight", "Hold"]


def _load_runway_segments(context_dir, airport):
    import os, pickle
    path = os.path.join(context_dir, airport, 'semantic_graph.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        g = pickle.load(f)
    pl = g['map_infos']['all_polylines']
    rw = pl[pl[:, 8] == 3]          # type==3 = runway (thr_id)
    if rw.shape[0] == 0:
        return None
    return rw[:, [2, 3, 6, 7]].astype(np.float64)


def _pts_min_dist_to_segs(pts, segs):
    """pts (P,2), segs (N,4)=[x1,y1,x2,y2] -> (P,) min dist per point."""
    a = segs[:, :2]; b = segs[:, 2:4]
    ab = b - a                                   # (N,2)
    ab2 = (ab * ab).sum(-1) + 1e-9               # (N,)
    # for each point, project onto each segment
    # (P,1,2) - (1,N,2) = (P,N,2)
    ap = pts[:, None, :] - a[None, :, :]         # (P,N,2)
    t = (ap * ab[None, :, :]).sum(-1) / ab2[None, :]   # (P,N)
    t = np.clip(t, 0, 1)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (P,N,2)
    d = np.linalg.norm(proj - pts[:, None, :], axis=-1)    # (P,N)
    return d.min(axis=1)                          # (P,)


@torch.no_grad()
def main_collect(loader, hist_len, max_batches, seq_x, seq_y, runway_segs,
                 on_runway_thresh, use_full_seq):
    # collect PER-POINT distances of on-runway points to the runway centerline.
    # a point counts as "on runway" if its distance to the nearest runway
    # segment is below on_runway_thresh; only those points measure accuracy.
    per_mode_dists = {m: [] for m in range(4)}   # per-trajectory mean (on-runway pts)
    all_point_dists = []                          # every on-runway point distance
    n_on_runway_pts = 0
    n_total_pts = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        sd = batch['scene_dict']
        ego_agent = sd['ego_agent_id']
        raw = sd.get('sequences', None)
        if raw is None:
            continue
        ego_raw = separate_ego_agent(raw, ego_agent)[:, 0]          # (B,T,C)
        masks = sd['agent_masks']
        ego_amask = separate_ego_agent(masks, ego_agent)[:, 0]      # (B,T)

        rb = sd.get('rule_based_encoding', None)
        if rb is None:
            continue
        tmi = rb[..., :4].float().argmax(-1).long()
        true_mode = separate_ego_agent(tmi, ego_agent).reshape(-1)

        # use full sequence (history+future) or just future, per flag
        if use_full_seq:
            xy = ego_raw[:, :, [seq_x, seq_y]].cpu().numpy()        # (B,T,2)
            mk = ego_amask.bool().cpu().numpy()                    # (B,T)
        else:
            xy = ego_raw[:, hist_len:, [seq_x, seq_y]].cpu().numpy()
            mk = ego_amask[:, hist_len:].bool().cpu().numpy()
        B = xy.shape[0]
        for b in range(B):
            valid = mk[b]
            if valid.sum() < 3:
                continue
            pts = xy[b][valid]                     # (P,2) valid points
            d = _pts_min_dist_to_segs(pts, runway_segs)   # (P,) dist to runway
            n_total_pts += len(d)
            # WHOLE-TRAJECTORY filter: keep only trajectories whose EVERY valid
            # point is on a runway. This excludes turn-onto/off-runway and
            # taxiway-edge trajectories, leaving pure runway roll (the clean
            # accuracy yardstick).
            if not (d < on_runway_thresh).all():
                continue
            n_on_runway_pts += len(d)
            all_point_dists.extend(d.tolist())
            per_mode_dists[int(true_mode[b].item())].append(d.mean())
    return per_mode_dists, np.array(all_point_dists), n_on_runway_pts, n_total_pts


@hydra.main(version_base="1.3", config_path="../configs", config_name="diagnose")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    datamodule = hydra.utils.instantiate(cfg.data)
    split = cfg.get("diag_split", "test")
    if split == "train" and getattr(datamodule, "task_name", None) != "train":
        datamodule.task_name = "train"
    datamodule.prepare_data(); datamodule.setup()
    loader = (datamodule.train_dataloader() if split == "train"
              else datamodule.test_dataloader())

    hist_len = cfg.get("hist_len", 10)
    max_batches = cfg.get("diag_max_batches", None)
    seq_x = getattr(G.SEQ_IDX, 'x', 6)
    seq_y = getattr(G.SEQ_IDX, 'y', 7)
    # 1 unit = range_scale metres (KMSY limits.json: 1000). used to report metres.
    range_scale = cfg.get("range_scale", 1000.0)
    # a point is "on a runway" if within this distance (in local-xy units) of a
    # runway segment. runway half-width ~30 m -> 0.03 (=30 m) default.
    on_runway_thresh = cfg.get("on_runway_thresh", 0.03)
    use_full_seq = cfg.get("use_full_seq", True)   # runway motion spans hist+fut

    ctx_dir = None
    for getter in (lambda: cfg.paths.context_dir,
                   lambda: cfg.context_dir,
                   lambda: getattr(datamodule, "context_dir", None)):
        try:
            ctx_dir = getter()
            if ctx_dir:
                break
        except Exception:
            continue
    airport = cfg.get("diag_airport", "kmsy")
    runway_segs = _load_runway_segments(ctx_dir, airport) if ctx_dir else None
    if runway_segs is None:
        log.info("[rw-acc] no runway segments; aborting")
        return
    log.info(f"[rw-acc] loaded {runway_segs.shape[0]} runway segments for {airport}; "
             f"on_runway_thresh={on_runway_thresh} ({on_runway_thresh*range_scale:.0f} m), "
             f"use_full_seq={use_full_seq}")

    per_mode, all_pts, n_on, n_tot = main_collect(
        loader, hist_len, max_batches, seq_x, seq_y,
        runway_segs, on_runway_thresh, use_full_seq)

    sc = range_scale
    log.info(f"\n[rw-acc] ===== runway-centerline deviation (accuracy benchmark, {split}) =====")
    log.info(f"[rw-acc]  points from fully-on-runway trajectories: {n_on} / {n_tot} total "
             f"({100*n_on/max(n_tot,1):.1f}%); only trajectories entirely on a runway are used")
    if len(all_pts) == 0:
        log.info("[rw-acc]  no on-runway points found; check on_runway_thresh / dataset")
        return

    log.info(f"[rw-acc]  --- per-point deviation (metres) ---")
    log.info(f"[rw-acc]    median={np.median(all_pts)*sc:.2f}  mean={all_pts.mean()*sc:.2f}  "
             f"p90={np.percentile(all_pts,90)*sc:.2f}  p99={np.percentile(all_pts,99)*sc:.2f}  "
             f"max={all_pts.max()*sc:.2f}")

    log.info(f"[rw-acc]  --- per-trajectory mean deviation, by mode (metres) ---")
    all_d = []
    for m in range(4):
        ds = np.array(per_mode[m])
        if len(ds) == 0:
            log.info(f"[rw-acc]   {NAMES[m]:>10}: no on-runway samples")
            continue
        all_d.append(ds)
        log.info(f"[rw-acc]   {NAMES[m]:>10}: n={len(ds):>6}  "
                 f"median={np.median(ds)*sc:.2f}  mean={ds.mean()*sc:.2f}  "
                 f"p90={np.percentile(ds,90)*sc:.2f}  p99={np.percentile(ds,99)*sc:.2f}")
    if all_d:
        allc = np.concatenate(all_d)
        log.info(f"[rw-acc]   {'ALL':>10}: n={len(allc):>6}  "
                 f"median={np.median(allc)*sc:.2f}  mean={allc.mean()*sc:.2f}  "
                 f"p90={np.percentile(allc,90)*sc:.2f}  p99={np.percentile(allc,99)*sc:.2f}")

    log.info(f"\n[rw-acc] interpretation:")
    log.info(f"[rw-acc]  runway motion is single-behaviour (along centerline), so this")
    log.info(f"[rw-acc]  deviation is a clean estimate of the dataset's positioning")
    log.info(f"[rw-acc]  accuracy. small median (a few m) = accurate tracking; a large")
    log.info(f"[rw-acc]  mean/tail indicates noisy tracks or map misalignment.")

    # ---- distribution plot: all on-runway per-point deviations (metres) ----
    try:
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        dev_m = all_pts * sc                       # metres
        med = np.median(dev_m); mean = dev_m.mean()
        p90 = np.percentile(dev_m, 90); p99 = np.percentile(dev_m, 99)

        fig, ax = plt.subplots(figsize=(10, 6))
        # clip the long tail for a readable histogram, but report full stats
        hi = np.percentile(dev_m, 99.5)
        ax.hist(dev_m[dev_m <= hi], bins=80, color='#5b8ff9',
                alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(med, color='#f5222d', linestyle='-', linewidth=1.6,
                   label=f'median = {med:.2f} m')
        ax.axvline(mean, color='#fa8c16', linestyle='--', linewidth=1.6,
                   label=f'mean = {mean:.2f} m')
        ax.axvline(p90, color='#722ed1', linestyle=':', linewidth=1.4,
                   label=f'p90 = {p90:.2f} m')
        ax.axvline(p99, color='#8c8c8c', linestyle=':', linewidth=1.4,
                   label=f'p99 = {p99:.2f} m')
        ax.set_xlabel('Deviation from runway centerline (m)')
        ax.set_ylabel('Number of trajectory points')
        ax.set_title(f'Runway-centerline deviation ({airport}, {split})\n'
                     f'all points from fully-on-runway trajectories '
                     f'(n={len(dev_m)})')
        ax.legend()
        out_png = os.path.join(os.getcwd(), f'runway_accuracy_hist_{airport}_{split}.png')
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close(fig)
        log.info(f"[rw-acc] saved deviation histogram -> {out_png}")
    except Exception as e:
        import traceback
        log.info(f"[rw-acc] histogram failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()