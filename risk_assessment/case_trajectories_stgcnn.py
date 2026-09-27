"""
Case-study deep dive companion to case_risk_dynamics.py -- STGCNN_baseline
adapter. For ONE case identified by its scene_file (NOT batch_idx/sample_idx,
which are NOT comparable across model repos -- see the scene_file
propagation commit), dumps this model's own predicted trajectory (a single
hypothesis, no mode/candidate structure -- see find_cases_stgcnn.py's module
docstring) plus the realized ground-truth trajectory, in the same schema
case_risk_dynamics.py uses, so a case-study chart can overlay this model's
prediction against the two-stage model's and the realized outcome for the
SAME real event.

Must scan the FULL test set (no limit_batches) since scene_file's position
in THIS repo's own batch ordering isn't known in advance -- this is a
one-off per-case lookup, not a bulk pass, so that's an acceptable cost.

Reference agent is auto-detected the same way as case_risk_dynamics.py
(closest approach in the REALIZED trajectory) -- see that module's
docstring for why. Since agents_in_scene ordering is baked into the scene
file itself (identical across repos for the same scene_file, unlike batch/
sample indices), this should independently resolve to the same physical
agent as the two-stage script's auto-detection for the same case, but is
computed fresh here rather than assumed.

Output JSON schema (see case_risk_dynamics.py's docstring for the general
shape):
    meta: airport, batch_idx, sample_idx, scene_file, ego_id, ref_agent_idx,
          ref_agent_type, hist_len
    gt: {ego, ref_agent, other_agents: [...]} -- same as case_risk_dynamics.py
    prediction: {dist_m: [T_pred floats], latlon: [[lat,lon], ...]} -- this
        model's single hypothesis (no modes/candidates dimension)
    real: {dist_m: [T_pred floats]}

Usage (mirrors find_cases_stgcnn.py's model-loading convention):

    python -m risk_assessment.case_trajectories_stgcnn \\
        --config-name=train_stgcnn_kmsy \\
        train=false \\
        ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-22-39/checkpoints/epoch_110.ckpt \\
        +case_scene_file=kmsy/KMSY_190_1688763600/001683_n-4.pkl \\
        +output_json=/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_292_46_trajectories_stgcnn.json
"""
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STGCNN_REPO = os.path.join(_REPO_ROOT, "STGCNN_baseline")
if _STGCNN_REPO not in sys.path:
    sys.path.insert(0, _STGCNN_REPO)

os.environ.setdefault("PROJECT_ROOT", _STGCNN_REPO)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_scenes.utils.transform_utils import inv_transform

from risk_assessment.common import (
    to_device, seed_for_reproducible_ego_selection, load_airport_ref, xy_array_to_latlon,
)

AGENT_TYPE_NAMES = {0: "Aircraft", 1: "Vehicle", 2: "Unknown"}


def _ego_trajectory_abs(ego_mu, sequences, ego_ids, hist_len):
    """Same as find_cases_stgcnn.py's helper -- (B, T_pred, 2) abs XY."""
    B = ego_mu.shape[0]
    future_rel = ego_mu[..., :2].detach().cpu().numpy()
    traj_abs = np.zeros_like(future_rel)
    for b in range(B):
        start_abs = sequences[b, ego_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
        start_heading = float(sequences[b, ego_ids[b], hist_len - 1, G.HD].detach().cpu().item())
        traj_abs[b] = inv_transform(future_rel[b], start_abs, start_heading)
    return traj_abs


@hydra.main(version_base="1.3", config_path="../STGCNN_baseline/configs",
            config_name="train_stgcnn_klax")
def main(cfg: DictConfig) -> None:
    output_json = cfg.get("output_json")
    if not output_json:
        raise ValueError("Pass +output_json=/path/to/case_trajectories.json")
    target_scene_file = cfg.get("case_scene_file")
    if not target_scene_file:
        raise ValueError("Pass +case_scene_file=<airport>/<day>/<scenario>.pkl")
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Pass train=false ckpt_path=/path/to/checkpoint.ckpt on the command line.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] missing keys: {result.missing_keys}")
    print(f"[LOAD] unexpected keys: {result.unexpected_keys}")
    model.to(device)
    model.eval()

    hist_len = model.hist_len
    max_pred_len = model.max_pred_len

    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    if isinstance(dataloader, (list, tuple)):
        dataloader = dataloader[0]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            scene = batch['scene_dict']
            scene_files = scene.get('scene_file')
            if scene_files is None:
                raise RuntimeError("scene_file not in scene_dict -- rerun with the scene_file fix.")
            if target_scene_file not in list(scene_files):
                if batch_idx % 200 == 0:
                    print(f"[case_trajectories_stgcnn] scanned batch {batch_idx}, not found yet")
                continue

            b = list(scene_files).index(target_scene_file)
            print(f"[case_trajectories_stgcnn] found at batch={batch_idx} sample={b}")

            batch = to_device(batch, device)
            scene = batch['scene_dict']

            seq = scene['rel_sequences']
            masks = scene['agent_masks'].bool()
            X = seq[:, :, :hist_len].float()
            mask_h = masks[:, :, :hist_len]
            mu, sigma = model.net(X, mask_h)  # (B, A, Tp, 2) each

            ego_agent = scene['ego_agent_id_test']
            ego_ids = [
                ego_agent[i].item() if torch.is_tensor(ego_agent) else int(ego_agent[i])
                for i in range(mu.shape[0])
            ]
            airport_ids = scene.get('airport_id')
            ego_id = ego_ids[b]

            ego_mu = separate_ego_agent(mu, ego_ids).squeeze(1)  # (B, Tp, 2)

            sequences = scene['sequences']
            agent_masks = scene['agent_masks']
            agent_types_flat = scene['agent_types'].reshape(sequences.shape[0], -1)

            traj_abs = _ego_trajectory_abs(ego_mu, sequences, ego_ids, hist_len)  # (B, Tp, 2)
            fut_end = hist_len + max_pred_len
            pred_xy = traj_abs[b]  # (Tp, 2)
            real_ego_xy = sequences[b, ego_id, hist_len:fut_end, G.XY].detach().cpu().numpy()

            # Auto-detect reference agent from the REALIZED trajectory --
            # same logic as case_risk_dynamics.py, computed independently.
            A_total = sequences.shape[1]
            other_idx, other_xy, other_valid = [], [], []
            for a in range(A_total):
                if a == ego_id:
                    continue
                valid_a = agent_masks[b, a, hist_len:fut_end].bool().detach().cpu().numpy()
                if not valid_a.any():
                    continue
                other_idx.append(a)
                other_xy.append(sequences[b, a, hist_len:fut_end, G.XY].detach().cpu().numpy())
                other_valid.append(valid_a)
            if not other_idx:
                raise RuntimeError(f"No valid other agent for scene_file={target_scene_file}")
            other_xy_arr = np.stack(other_xy, axis=0)
            other_valid_arr = np.stack(other_valid, axis=0)
            dist = np.linalg.norm(other_xy_arr - real_ego_xy[None, :, :], axis=-1)
            dist = np.where(other_valid_arr, dist, np.inf)
            agent_i, _t_i = np.unravel_index(np.argmin(dist), dist.shape)
            ref_agent_idx = other_idx[agent_i]

            ref_xy = sequences[b, ref_agent_idx, hist_len:fut_end, G.XY].detach().cpu().numpy()
            ref_valid = agent_masks[b, ref_agent_idx, hist_len:fut_end].bool().detach().cpu().numpy()
            ref_type = int(agent_types_flat[b, ref_agent_idx].item())

            real_dist = np.linalg.norm(real_ego_xy - ref_xy, axis=-1)
            real_dist = np.where(ref_valid, real_dist, np.nan) * 1000  # km -> m
            pred_dist = np.linalg.norm(pred_xy - ref_xy, axis=-1)
            pred_dist = np.where(ref_valid, pred_dist, np.nan) * 1000  # km -> m

            airport = airport_ids[b] if airport_ids is not None else None
            ref = load_airport_ref(cfg.paths.assets_dir, airport)

            def _agent_latlon(a_idx):
                xy = sequences[b, a_idx, :, G.XY].detach().cpu().numpy()
                valid = agent_masks[b, a_idx, :].bool().detach().cpu().numpy()
                xy_masked = np.where(valid[:, None], xy, np.nan)
                return {
                    "agent_idx": a_idx,
                    "agent_type": AGENT_TYPE_NAMES.get(int(agent_types_flat[b, a_idx].item())),
                    "latlon_hist": xy_array_to_latlon(xy_masked[:hist_len], ref),
                    "latlon_fut": xy_array_to_latlon(xy_masked[hist_len:], ref),
                }

            A = sequences.shape[1]
            gt_out = {
                "ego": _agent_latlon(ego_id),
                "ref_agent": _agent_latlon(ref_agent_idx),
                "other_agents": [
                    _agent_latlon(a) for a in range(A)
                    if a not in (ego_id, ref_agent_idx)
                    and agent_masks[b, a, :].bool().any().item()
                ],
            }

            pred_xy_masked = np.where(ref_valid[:, None], pred_xy, np.nan)
            out = {
                "meta": {
                    "airport": airport,
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "scene_file": target_scene_file,
                    "ego_id": ego_id,
                    "ref_agent_idx": ref_agent_idx,
                    "ref_agent_type": AGENT_TYPE_NAMES.get(ref_type, str(ref_type)),
                    "hist_len": hist_len,
                },
                "gt": gt_out,
                "prediction": {
                    "dist_m": [None if np.isnan(v) else float(v) for v in pred_dist],
                    "latlon": xy_array_to_latlon(pred_xy_masked, ref),
                },
                "real": {"dist_m": [None if np.isnan(v) else float(v) for v in real_dist]},
            }
            with open(output_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[case_trajectories_stgcnn] wrote {output_json}")
            return

    raise RuntimeError(f"scene_file={target_scene_file} never found in the full test set")


if __name__ == "__main__":
    main()
