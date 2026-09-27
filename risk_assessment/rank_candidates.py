"""
Post-hoc case-study legibility ranking for an ALREADY-COMPUTED
find_cases_two_stage.py CSV -- no model, no checkpoint, no GPU needed.

find_cases_two_stage.py's near-miss columns (true_min_sep_gt, its
agent_type/on_road, scene_min_sep_gt, min_sep_{mode}, ...) already identify
which rows are genuine confirmed near-misses; re-deriving those needs the
trained model. But whether a candidate is a LEGIBLE case-study illustration
-- does the agent actually go somewhere, and does it go there smoothly --
is a pure ground-truth-trajectory question and never touches a prediction,
so it doesn't need the model either. This script adds exactly those two
columns (see find_cases_two_stage.py's identical inline computation for the
rationale) to an existing CSV by looking up each row's scene_file once and
reading its ego ground-truth trajectory directly out of the dataloader --
skipping model instantiation/checkpoint loading/GPU forward entirely, which
is what made the original find_cases_*.py runs slow. Also drop
data.dataset.config.add_context=false (below, NO + prefix -- add_context
already has a default in configs/data/default.yaml, so this OVERRIDES it
rather than adding a new key) since context-map generation is pure
per-scene CPU overhead this script has no use for.

Usage (reuses AmeliaTF_main_two_phases_4T's config exactly like
find_cases_two_stage.py does, so the same data=/paths= overrides apply --
PASS THE SAME +limit_batches= VALUE used to produce input_csv, so every
scene_file already in it is guaranteed to be encountered while scanning):

    python -m risk_assessment.rank_candidates \\
        data=kbos.yaml \\
        +data.dataset.config.random_ego=false \\
        data.dataset.config.add_context=false \\
        +input_csv=/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases.csv \\
        +output_csv=/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv \\
        +limit_batches=390
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AmeliaTF_main_two_phases_4T"))
os.environ.setdefault("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), "..", "AmeliaTF_main_two_phases_4T"))

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from amelia_tf.utils import global_masks as G
from risk_assessment.common import seed_for_reproducible_ego_selection


def _displacement_and_straightness(xy):
    """xy: (T, 2) valid (already-masked) points, this repo's local-XY km
    units. Net displacement (start-to-end) and straightness (net / total
    path length actually walked) -- see find_cases_two_stage.py's identical
    helper for what each catches (near-stationary vs. real reversal)."""
    if xy.shape[0] < 2:
        return 0.0, 1.0
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(seg.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    return net, (net / path_len if path_len > 1e-9 else 1.0)


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main_two_phases_4T/configs",
            config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    input_csv = cfg.get("input_csv")
    output_csv = cfg.get("output_csv")
    if not input_csv or not output_csv:
        raise ValueError("Pass +input_csv=... +output_csv=... on the command line.")

    df = pd.read_csv(input_csv)
    wanted = set(df["scene_file"].dropna().unique())
    print(f"[rank_candidates] {len(wanted)} unique scene_files to look up")

    hist_len = cfg.data.dataset.config.hist_len

    # Dataset/dataloader only -- deliberately never touches cfg.model or
    # cfg.ckpt_dir/mode_ckpt_path/traj_ckpt_path, so no checkpoint is loaded
    # and nothing runs on GPU.
    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    limit_batches = cfg.get("limit_batches")
    stats = {}  # scene_file -> (hist_disp_m, full_disp_m, straightness)
    for batch_idx, batch in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            print(f"[rank_candidates] stopping early: limit_batches={limit_batches}")
            break

        scene = batch['scene_dict']
        scene_files = scene.get('scene_file')
        if scene_files is None:
            continue

        ego_agent = scene['ego_agent_id']
        sequences = scene['sequences']
        agent_masks = scene['agent_masks']
        B = sequences.shape[0]

        for b in range(B):
            sf = scene_files[b]
            if sf not in wanted or sf in stats:
                continue
            ego_id = ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])

            hist_mask = agent_masks[b, ego_id, :hist_len].bool().detach().cpu().numpy()
            hist_xy = sequences[b, ego_id, :hist_len, G.XY].detach().cpu().numpy()[hist_mask]
            fut_mask = agent_masks[b, ego_id, hist_len:].bool().detach().cpu().numpy()
            full_mask = np.concatenate([hist_mask, fut_mask])
            full_xy = sequences[b, ego_id, :, G.XY].detach().cpu().numpy()[full_mask]

            hist_disp_km, _ = _displacement_and_straightness(hist_xy)
            full_disp_km, straightness = _displacement_and_straightness(full_xy)
            stats[sf] = (hist_disp_km * 1000.0, full_disp_km * 1000.0, straightness)

        if batch_idx % 200 == 0:
            print(f"[rank_candidates] scanned batch {batch_idx}, matched {len(stats)}/{len(wanted)}")
        if len(stats) >= len(wanted):
            print(f"[rank_candidates] all {len(wanted)} scene_files found, stopping early")
            break

    missing = wanted - set(stats)
    if missing:
        print(f"[rank_candidates] WARNING: {len(missing)} scene_files never matched "
              f"(different +limit_batches/split than input_csv's run?) -- "
              f"their new columns will be blank")

    df["ego_hist_displacement_m"] = df["scene_file"].map(lambda s: stats.get(s, (None, None, None))[0])
    df["ego_full_displacement_m"] = df["scene_file"].map(lambda s: stats.get(s, (None, None, None))[1])
    df["ego_track_straightness"] = df["scene_file"].map(lambda s: stats.get(s, (None, None, None))[2])
    df.to_csv(output_csv, index=False)
    print(f"[rank_candidates] wrote {output_csv}")


if __name__ == "__main__":
    main()
