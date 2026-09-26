"""
One-off diagnostic: for a case-study candidate whose closest-approach agent
has an ambiguous type label (Unknown -- per the Amelia dataset's own docs,
this means the raw label was too noisy to confidently classify as Aircraft
or Vehicle, not a third physical category), check whether the closest-
approach point sits ON the airport's taxiway/runway movement-area network
or OFF it (i.e. off the semantic_graph.pkl reference network built from
OSM data).

Rationale: gate/apron/stand areas are NOT covered by the movement-area
graph (see the Chapter-3 turn-mode work: 99.9% of stationary Hold samples
sit off this same network, since they're gate-adjacent, not on a mapped
taxiway/runway edge). So "off-road and far from any edge" is corroborating
evidence for "this is a routine gate/stand encounter" (weakens the
near-miss case), while "on-road, close to a taxiway/runway edge" is
corroborating evidence for "this happened out on the movement area" (an
aircraft or vehicle legitimately on the network at that point, closer to
the near-miss story this case study wants) -- it does NOT resolve the
Aircraft-vs-Vehicle ambiguity itself, only the gate-vs-movement-area
question, which is the next best thing available given the dataset's own
noisy-labelling limitation (see AGENT_TYPES's docstring / check_agent_types.py).

Reuses amelia_tf.utils.off_road_evaluator.OffRoadEvaluator's reference
network (loaded once per airport) rather than reimplementing the
point-to-edge distance logic.

Usage: edit TARGETS below (one entry per (airport, batch_idx, sample_idx,
expected_ego_id, closest_agent_idx, closest_t) tuple -- closest_agent_idx/
closest_t come straight from check_agent_types.py's printed output), then:

    python -m risk_assessment.check_case_location \\
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
import numpy as np
import torch
from omegaconf import DictConfig
from geographiclib.geodesic import Geodesic

from amelia_tf.utils import global_masks as G
from amelia_tf.utils.off_road_evaluator import OffRoadEvaluator
from amelia_scenes.utils.transform_utils import xy_to_ll

from risk_assessment.common import seed_for_reproducible_ego_selection

# (batch_idx, sample_idx, expected_ego_id, closest_agent_idx, closest_t) --
# closest_agent_idx/closest_t are check_agent_types.py's printed values for
# the candidates worth a location check (currently: KMSY's only
# predicted-risk-and-real-risk-agreeing candidate, closest_agent_type=Unknown).
TARGETS = {
    "kmsy": [
        (2708, 36, 0, 1, 41),
    ],
}


def _xy_to_latlon(xy: np.ndarray, ref) -> tuple:
    """xy: (2,) absolute XY in the dataset's local km frame. Returns (lat, lon)."""
    traj_rel = torch.tensor(xy, dtype=torch.float64).reshape(1, 1, 2)
    traj_ll = xy_to_ll(
        traj_rel, np.zeros(2), 0.0, ref, Geodesic.WGS84)
    return traj_ll[0, 0, 0].item(), traj_ll[0, 0, 1].item()


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main_two_phases_4T/configs",
            config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    airport = cfg.get("targets_airport")
    if not airport or airport not in TARGETS:
        raise ValueError(f"Pass +targets_airport=<{'|'.join(TARGETS.keys())}>")
    targets = {(b, s): (ego, agent_idx, t) for b, s, ego, agent_idx, t in TARGETS[airport]}
    wanted_batches = sorted({b for b, s in targets})

    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    evaluator = OffRoadEvaluator(cfg.paths.assets_dir, airport)
    ref = evaluator.ref

    # check_agent_types.py's printed `t` is an index into the FUTURE window
    # (sequences[b, agent, hist_len:, ...]), not the full sequence.
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
            B = sequences.shape[0]

            for b in range(B):
                key = (batch_idx, b)
                if key not in targets:
                    continue
                found += 1
                expected_ego, agent_idx, t_fut = targets[key]
                t = hist_len + t_fut
                ego_id = ego_agent[b].item() if torch.is_tensor(ego_agent) else int(ego_agent[b])
                match = "OK" if ego_id == expected_ego else f"MISMATCH (expected {expected_ego})"

                ego_xy = sequences[b, ego_id, t, G.XY].detach().cpu().numpy()
                other_xy = sequences[b, agent_idx, t, G.XY].detach().cpu().numpy()

                for label, xy in [("ego", ego_xy), (f"agent_{agent_idx}", other_xy)]:
                    lat, lon = _xy_to_latlon(xy, ref)
                    result = evaluator._evaluate_trajectory([(lon, lat)], safety_margin=1.0)
                    off = result["per_point_is_off"][0]
                    dist = result["per_point_distances"][0]
                    status = "OFF-road (likely gate/apron/stand)" if off else "ON-road (movement-area network)"
                    print(f"[{airport}] batch={batch_idx} sample={b} ego_id={ego_id} ({match}) t={t}: "
                          f"{label} at ({lat:.6f},{lon:.6f}) -> {status}, "
                          f"dist_to_nearest_edge={dist:.2f}m")

    print(f"\n[check_case_location] found {found}/{len(targets)} requested targets")


if __name__ == "__main__":
    main()
