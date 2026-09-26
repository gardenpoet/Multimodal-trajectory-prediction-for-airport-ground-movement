"""
One-off diagnostic: fetch the stable `scene_file` identifier (see
amelia_dataset.py's __getitem__ comment / the scene_file propagation commit)
for the 30 case-study candidates already identified from
find_cases_two_stage.py's *_50_4T_cases.csv files (airport, batch_idx,
sample_idx, ego_id) -- so those candidates can later be cross-referenced
against a *_stgcnn_cases.csv (or any other model's CSV) run with the same
fix, WITHOUT needing to rerun the expensive full find_cases_two_stage.py
pass (430-460K samples/airport) just to pick up one new column.

batch_idx/sample_idx alone are NOT valid keys across model repos (see the
scene_file propagation commit message) -- this script only needs to iterate
the two-stage dataloader up to each target's own batch_idx, same cheap
targeted-iteration pattern as check_agent_types.py.

Usage: edit TARGETS below (one entry per candidate row: batch_idx,
sample_idx, expected_ego_id -- same triples as check_agent_types.py's
TARGETS), then per airport:

    python -m risk_assessment.get_scene_files \\
        data=kmsy.yaml \\
        ckpt=kmsy2 \\
        +targets_airport=kmsy
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TWO_STAGE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main_two_phases_4T")
if _TWO_STAGE_REPO not in sys.path:
    sys.path.insert(0, _TWO_STAGE_REPO)

os.environ.setdefault("PROJECT_ROOT", _TWO_STAGE_REPO)

import hydra
import torch
from omegaconf import DictConfig

from risk_assessment.common import seed_for_reproducible_ego_selection

# Same 30 candidates as check_agent_types.py's TARGETS.
TARGETS = {
    "kbos": [
        (3243, 11, 4), (2570, 1, 3), (3265, 125, 1), (2070, 98, 4), (3182, 22, 0),
        (1242, 24, 1), (2510, 57, 0), (552, 60, 2), (1032, 55, 1), (628, 61, 1),
    ],
    "klax": [
        (2304, 60, 3), (501, 20, 0), (3194, 38, 0), (431, 44, 0), (454, 36, 0),
        (487, 81, 2), (4032, 124, 0), (1978, 109, 3), (824, 89, 1), (600, 55, 2),
    ],
    "kmsy": [
        (2708, 36, 0), (1741, 79, 1), (1082, 109, 0), (634, 1, 1), (279, 13, 0),
        (1000, 17, 0), (788, 122, 0), (998, 68, 1), (2680, 97, 1), (382, 24, 0),
    ],
}


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main_two_phases_4T/configs",
            config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    airport = cfg.get("targets_airport")
    if not airport or airport not in TARGETS:
        raise ValueError(f"Pass +targets_airport=<{'|'.join(TARGETS.keys())}>")
    targets = {(b, s): ego for b, s, ego in TARGETS[airport]}
    wanted_batches = sorted({b for b, s in targets})

    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    found = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > max(wanted_batches):
                break
            if batch_idx not in wanted_batches:
                continue
            scene = batch["scene_dict"]
            ego_agent = scene["ego_agent_id"]
            scene_files = scene["scene_file"]
            B = len(scene_files)

            for b in range(B):
                key = (batch_idx, b)
                if key not in targets:
                    continue
                found += 1
                expected_ego = targets[key]
                ego_id = ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                match = "OK" if ego_id == expected_ego else f"MISMATCH (expected {expected_ego})"
                print(f"[{airport}] batch={batch_idx} sample={b} ego_id={ego_id} ({match}): "
                      f"scene_file={scene_files[b]}")

    print(f"\n[get_scene_files] found {found}/{len(targets)} requested (batch, sample) pairs")


if __name__ == "__main__":
    main()
