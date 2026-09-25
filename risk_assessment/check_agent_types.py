"""
One-off diagnostic: for a handful of case-study candidates already identified
from a find_cases_two_stage.py CSV (airport, batch_idx, sample_idx, ego_id),
check whether the agent that achieved the ground-truth minimum-separation
distance (scene_min_sep_gt) is an Aircraft or a Vehicle
(amelia_tf.utils.global_masks.AGENT_TYPES). Ground service vehicles
legitimately operate within sub-metre distance of a parked/gate-adjacent
aircraft as routine, non-hazardous ground ops -- if the "closest agent" in a
supposedly-compelling near-miss case turns out to be a Vehicle rather than
another Aircraft, that case is very likely NOT the aircraft-aircraft
near-miss the case study wants to illustrate and should be dropped from
consideration.

Needs no model/checkpoint at all -- only ground-truth data (sequences,
agent_masks, agent_types), so this is far cheaper than a full
find_cases_two_stage.py run: just the datamodule, iterated only up to each
requested batch_idx.

Usage: edit TARGETS below (one entry per candidate row from the CSV you're
checking), then per airport:

    python -m risk_assessment.check_agent_types \\
        data=kbos.yaml \\
        +targets_airport=kbos
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TWO_STAGE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main_two_phases_4T")
if _TWO_STAGE_REPO not in sys.path:
    sys.path.insert(0, _TWO_STAGE_REPO)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G

from risk_assessment.common import seed_for_reproducible_ego_selection, min_separation

AGENT_TYPE_NAMES = {0: "Aircraft", 1: "Vehicle", 2: "Unknown"}

# (batch_idx, sample_idx, expected_ego_id) triples, one per candidate row
# pulled from each airport's *_50_4T_cases.csv -- see the case-selection
# shortlist from 2026-09-25.
TARGETS = {
    "kbos": [
        (3243, 11, 4),   # top (b) candidate: scene_min_sep_gt=0.000307 km
        (2570, 1, 3),
        (3265, 125, 1),
        (2070, 98, 4),
        (3182, 22, 0),
        (1242, 24, 1),   # top (a) candidate
        (2510, 57, 0),
        (552, 60, 2),
        (1032, 55, 1),
        (628, 61, 1),
    ],
    "klax": [
        (2304, 60, 3),   # top (b) candidate: scene_min_sep_gt=0.000726 km
        (501, 20, 0),
        (3194, 38, 0),
        (431, 44, 0),
        (454, 36, 0),
        (487, 81, 2),    # top (a) candidate
        (4032, 124, 0),
        (1978, 109, 3),
        (824, 89, 1),
        (600, 55, 2),
    ],
    "kmsy": [
        (2708, 36, 0),   # top (b) candidate: scene_min_sep_gt=0.001945 km
        (1741, 79, 1),
        (1082, 109, 0),
        (634, 1, 1),
        (279, 13, 0),
        (1000, 17, 0),   # top (a) candidate
        (788, 122, 0),
        (998, 68, 1),
        (2680, 97, 1),
        (382, 24, 0),
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

    # No model needed here, but the future window starts right after
    # hist_len -- read it straight from the data config.
    hist_len = cfg.data.dataset.config.hist_len

    found = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > max(wanted_batches):
                break
            if batch_idx not in wanted_batches:
                continue
            scene = batch["scene_dict"]
            ego_agent = scene["ego_agent_id"]
            sequences = scene["sequences"]
            agent_masks = scene["agent_masks"]
            agent_types = scene["agent_types"]
            B, A = sequences.shape[:2]

            for b in range(B):
                key = (batch_idx, b)
                if key not in targets:
                    continue
                found += 1
                ego_id = ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                expected_ego = targets[key]
                match = "OK" if ego_id == expected_ego else \
                    f"MISMATCH (expected {expected_ego})"

                ego_xy_fut = sequences[b, ego_id, hist_len:, G.XY].detach().cpu().numpy()  # (T,2)

                other_xy, other_valid, other_types, other_idx = [], [], [], []
                for a in range(A):
                    if a == ego_id:
                        continue
                    valid_a = agent_masks[b, a, hist_len:].bool().detach().cpu().numpy()
                    if not valid_a.any():
                        continue
                    other_xy.append(sequences[b, a, hist_len:, G.XY].detach().cpu().numpy())
                    other_valid.append(valid_a)
                    t = agent_types[b, a]
                    other_types.append(int(t.item()) if torch.is_tensor(t) else int(t))
                    other_idx.append(a)

                if not other_xy:
                    print(f"[{airport}] batch={batch_idx} sample={b} ego_id={ego_id} ({match}): "
                          "no valid other agent (shouldn't happen -- was in the relevant-sample set)")
                    continue

                other_xy_arr = np.stack(other_xy, axis=0)
                other_valid_arr = np.stack(other_valid, axis=0)
                dist = np.linalg.norm(other_xy_arr - ego_xy_fut[None, :, :], axis=-1)
                dist = np.where(other_valid_arr, dist, np.inf)
                flat_idx = np.argmin(dist)
                agent_i, t_i = np.unravel_index(flat_idx, dist.shape)
                min_dist = dist[agent_i, t_i]
                closest_type = AGENT_TYPE_NAMES.get(other_types[agent_i], f"?{other_types[agent_i]}")

                print(f"[{airport}] batch={batch_idx} sample={b} ego_id={ego_id} ({match}): "
                      f"min_sep_gt={min_dist:.6f} km, closest_agent_type={closest_type} "
                      f"(agent idx {other_idx[agent_i]}, t={t_i})")

    print(f"\n[check_agent_types] found {found}/{len(targets)} requested (batch, sample) pairs")


if __name__ == "__main__":
    main()
