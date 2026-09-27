"""
Bulk-renders risk-critical case candidates as static PNGs using the REPO'S
OWN existing visualization code (amelia_scenes.visualization.scene_viz +
amelia_scenes.utils.dataset.load_assets -- the real bkg_map.png background
and the ac.png/vc.png/uk_ac.png agent icons, rotated to heading, already
handled by that code's plot_sequences()) -- instead of building a custom
per-case interactive plot one candidate at a time. Lets the user browse
MANY candidates at once and do their own initial visual screening, rather
than iterating through the risk_assessment pipeline's case_risk_dynamics.py
+ artifact workflow three candidates at a time.

Reads directly from an ALREADY-COMPUTED find_cases_two_stage.py CSV (or its
rank_candidates.py-augmented version -- either works, this only reads
columns find_cases_two_stage.py itself already writes) and the scene .pkl
files it references -- no model, no Hydra, no GPU: this only needs the
ground-truth scene data (agent_sequences/masks/types/ids/valid, all present
directly in the RAW per-scene .pkl file -- see scene_processor.py's dict
construction, the same keys plot_sequences() expects), plus the SAME
sampling_strategy/k_agents subsetting amelia_dataset.py applies at training
time (AGENT_ORDER_STRATEGY/K_AGENTS below, from configs/data/default.yaml),
so a CSV row's ego_id (an index into that subset) maps to the right
physical agent in the full scene.

Usage:
    python -m risk_assessment.plot_risk_cases \\
        --input_csv /gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv \\
        --out_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/case_screenshots/kbos \\
        --airport kbos \\
        --max_sep_km 0.05
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd

# This file lives in the top-level risk_assessment/ folder (sibling to each
# model's own repo folder) -- insert AmeliaTF_main_two_phases_4T onto
# sys.path so `import amelia_scenes...` below resolves to its copy (any of
# the three repos' copies would do here, since this only touches
# ground-truth scene data/plotting, not anything two-stage-specific).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TWO_STAGE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main_two_phases_4T")
if _TWO_STAGE_REPO not in sys.path:
    sys.path.insert(0, _TWO_STAGE_REPO)
os.environ.setdefault("PROJECT_ROOT", _TWO_STAGE_REPO)

from amelia_scenes.visualization import scene_viz
from amelia_scenes.utils.dataset import load_assets

# configs/data/default.yaml's sampling_strategy/k_agents -- the subsetting
# amelia_dataset.py's transform_scene_data applies before assigning ego_id,
# reproduced here so a CSV row's ego_id (an index into agents_in_scene)
# resolves to the correct physical agent index in the full raw scene.
AGENT_ORDER_STRATEGY = "critical"
K_AGENTS = 5

_ASSET_CACHE = {}


def _get_assets(base_dir, airport):
    if airport not in _ASSET_CACHE:
        _ASSET_CACHE[airport] = load_assets(base_dir, airport)
    return _ASSET_CACHE[airport]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--airport", required=True)
    ap.add_argument(
        "--base_dir",
        default="/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia",
        help="Parent of assets/ and graph_data_.../ -- configs/paths/default.yaml's base_dir.")
    ap.add_argument(
        "--in_data_dir", default=None,
        help="Defaults to {base_dir}/traj_data_a10v08/proc_full_scenes/ "
             "(configs/paths/default.yaml's scenes_dir).")
    ap.add_argument(
        "--max_sep_km", type=float, default=0.05,
        help="Only plot rows with true_min_sep_gt below this (default: "
             "common.py's SAFETY_MARGIN_KM, 50m).")
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of scenes plotted -- sanity-check with a small "
             "number (e.g. 3) before doing a full run.")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    in_data_dir = args.in_data_dir or os.path.join(
        args.base_dir, "traj_data_a10v08", "proc_full_scenes")
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df[
        (df["true_min_sep_gt_agent_type"] == "Aircraft")
        & (df["true_min_sep_gt_on_road"] == True)
        & (df["true_min_sep_gt"] < args.max_sep_km)
    ].sort_values("true_min_sep_gt")
    if args.limit:
        df = df.head(args.limit)
    print(f"[plot_risk_cases] {len(df)} candidates to plot "
          f"(true_min_sep_gt < {args.max_sep_km * 1000:.0f}m, Aircraft, on-road)")

    assets = _get_assets(args.base_dir, args.airport)

    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        scene_file = row["scene_file"]
        pkl_path = os.path.join(in_data_dir, scene_file)
        try:
            with open(pkl_path, "rb") as f:
                scene = pickle.load(f)
            agents_in_scene = scene["meta"]["agent_order"][AGENT_ORDER_STRATEGY][:K_AGENTS]
            real_ego_id = int(agents_in_scene[int(row["ego_id"])])

            sep_m = row["true_min_sep_gt"] * 1000.0
            tag = f"b{row['batch_idx']}_s{row['sample_idx']}_sep{sep_m:.1f}m"
            filename = os.path.join(args.out_dir, f"{args.airport}_{tag}.png")
            scene_viz.plot_scene(
                scene, assets, filename, scene_type="simple",
                agents_interest=[real_ego_id], dpi=args.dpi)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[plot_risk_cases] FAILED {scene_file}: {e}")

    print(f"[plot_risk_cases] done: {n_ok} plotted, {n_fail} failed, "
          f"output in {args.out_dir}")


if __name__ == "__main__":
    main()
