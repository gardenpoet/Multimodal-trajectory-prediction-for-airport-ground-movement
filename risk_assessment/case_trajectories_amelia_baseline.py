"""
Case-study deep dive companion to case_risk_dynamics.py -- AmeliaTF_main
(plain, non-manoeuvre-conditioned) baseline adapter. For ONE case identified
by its scene_file (NOT batch_idx/sample_idx, which are NOT comparable
across model repos -- see the scene_file propagation commit), dumps this
model's H raw hypothesis trajectories (a flat GMM head, no mode/candidate
structure -- see find_cases_amelia_baseline.py's module docstring) plus the
realized ground-truth trajectory, in a schema close to
case_risk_dynamics.py's, so a case-study chart can overlay this model's
predictions against the two-stage model's and the realized outcome for the
SAME real event.

Must scan the FULL test set (no limit_batches) since scene_file's position
in THIS repo's own batch ordering isn't known in advance -- this is a
one-off per-case lookup, not a bulk pass, so that's an acceptable cost.

Reference agent is auto-detected the same way as case_risk_dynamics.py
(closest approach in the REALIZED trajectory) -- see that module's
docstring for why. Computed independently here (not assumed to match the
other adapters' auto-detection), though it should resolve to the same
physical agent since agents_in_scene ordering is baked into the scene file
itself.

Output JSON schema (see case_risk_dynamics.py's docstring for the general
shape, including the sigma_xy/start_heading_deg caveat -- RAW, un-rotated
sigma, not yet used by anything):
    meta: airport, batch_idx, sample_idx, scene_file, ego_id, ref_agent_idx,
          ref_agent_type, hist_len, start_heading_deg
    gt: {ego, ref_agent, other_agents: [...]} -- same as case_risk_dynamics.py
    hypotheses: [{prob, dist_m: [T_pred floats], latlon: [[lat,lon], ...],
        sigma_xy: [[sigma_x,sigma_y], ...]}, ...] -- H raw hypotheses, no
        mode grouping (unlike case_risk_dynamics.py's modes dict)
    real: {dist_m: [T_pred floats]}

Usage (mirrors find_cases_amelia_baseline.py's model-loading convention).
scene_file can be given directly, or resolved from a ranked-candidates CSV
by batch_idx/sample_idx (see risk_assessment/common.py's
resolve_case_scene_file):

    python -m risk_assessment.case_trajectories_amelia_baseline \\
        --config-name=eval_kmsy \\
        ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/checkpoints/Single-Airport/kmsy/kmsy_baseline_50.ckpt \\
        +case_batch_idx=292 +case_sample_idx=46 \\
        +cases_csv=/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_50_4T_cases_ranked.csv \\
        +output_json=/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_292_46_trajectories_amelia_baseline.json
"""
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASELINE_REPO = os.path.join(_REPO_ROOT, "AmeliaTF_main")
if _BASELINE_REPO not in sys.path:
    sys.path.insert(0, _BASELINE_REPO)

os.environ.setdefault("PROJECT_ROOT", _BASELINE_REPO)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_scenes.utils.transform_utils import inv_transform

from risk_assessment.common import (
    to_device, seed_for_reproducible_ego_selection, load_airport_ref, xy_array_to_latlon,
    resolve_case_scene_file,
)

AGENT_TYPE_NAMES = {0: "Aircraft", 1: "Vehicle", 2: "Unknown"}


def _hypothesis_trajectories_abs(ego_mu, sequences, ego_ids, hist_len):
    """Same as find_cases_amelia_baseline.py's helper -- (H,B,T_pred,2) abs XY."""
    B, T_total, H, D = ego_mu.shape
    T_pred = T_total - hist_len
    start_abs = np.stack([
        sequences[b, ego_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
        for b in range(B)
    ], axis=0)
    start_heading = np.array([
        float(sequences[b, ego_ids[b], hist_len - 1, G.HD].detach().cpu().item())
        for b in range(B)
    ])
    traj_abs = np.zeros((H, B, T_pred, 2), dtype=np.float64)
    for h in range(H):
        future_rel = ego_mu[:, hist_len:, h, :2].detach().cpu().numpy()
        for b in range(B):
            traj_abs[h, b] = inv_transform(future_rel[b], start_abs[b], start_heading[b])
    return traj_abs


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main/configs",
            config_name="eval_klax")
def main(cfg: DictConfig) -> None:
    output_json = cfg.get("output_json")
    if not output_json:
        raise ValueError("Pass +output_json=/path/to/case_trajectories.json")
    target_scene_file = resolve_case_scene_file(cfg)
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Pass ckpt_path=/path/to/checkpoint.ckpt on the command line.")
    # None (not set) means auto-detect from the realized trajectory below --
    # see case_risk_dynamics.py's module docstring for why that's the right
    # default, and its "auto-detect can pick the wrong agent for a
    # counterfactual (e.g. Hold-mode) candidate" caveat for when to override.
    ref_agent_idx_override = cfg.get("case_ref_agent_idx")
    ref_agent_idx_override = (
        int(ref_agent_idx_override) if ref_agent_idx_override is not None else None)

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

    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    if isinstance(dataloader, (list, tuple)):
        dataloader = dataloader[0]

    # Scanning batch-by-batch via enumerate(dataloader) to find scene_file
    # forces __getitem__ on every OTHER scene along the way too. Since
    # +data.dataset.config.ego_agent_id is a dataset-wide override (applied
    # in __getitem__ to every scene, not just the target one), any other
    # scene with fewer real agents than that pinned index crashes the whole
    # scan before ever reaching the target scene -- even though the target
    # scene itself may be perfectly fine. The scenario_list's own relative
    # paths (set at dataset-construction time, before ego_agent_id is ever
    # consumed) let us find the target's dataset index directly, with no
    # pickle loading or __getitem__ calls on any other scene.
    dataset = dataloader.dataset
    dataset_idx = None
    for i, item in enumerate(dataset.scenario_list):
        if os.path.relpath(str(item), dataset.in_data_dir) == target_scene_file:
            dataset_idx = i
            break
    if dataset_idx is None:
        raise RuntimeError(f"scene_file={target_scene_file} not found in this repo's scenario_list")
    print(f"[case_trajectories_amelia_baseline] found at dataset_idx={dataset_idx}")
    single_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [dataset_idx]),
        batch_size=1, shuffle=False, num_workers=0,
        collate_fn=dataloader.collate_fn,
    )
    batch = next(iter(single_loader))
    scene = batch['scene_dict']
    scene_files = scene.get('scene_file')
    if scene_files is None:
        raise RuntimeError("scene_file not in scene_dict -- rerun with the scene_file fix.")
    b = list(scene_files).index(target_scene_file)

    batch = to_device(batch, device)
    scene = batch['scene_dict']

    with torch.no_grad():
        Y = scene['rel_sequences']
        X = torch.zeros_like(Y).type(torch.float)
        X[:, :, :hist_len] = Y[:, :, :hist_len]
        X = X[:, :, :, :4]
        context = scene['context']
        adjacency = scene['adjacency']
        pred_scores, mu, sigma = model.net(X, context=context, adjacency=adjacency, mask=None)

        ego_agent = scene['ego_agent_id_test']
        ego_ids = [
            ego_agent[i].item() if torch.is_tensor(ego_agent) else int(ego_agent[i])
            for i in range(mu.shape[0])
        ]
        airport_ids = scene.get('airport_id')
        ego_id = ego_ids[b]

        ego_pred_scores = separate_ego_agent(pred_scores, ego_ids).squeeze(1)  # (B, H)
        ego_mu = separate_ego_agent(mu, ego_ids).squeeze(1)                    # (B, T_total, H, D)
        ego_sigma = separate_ego_agent(sigma, ego_ids).squeeze(1)              # (B, T_total, H, D)

        sequences = scene['sequences']
        agent_masks = scene['agent_masks']
        agent_types_flat = scene['agent_types'].reshape(sequences.shape[0], -1)

        traj_abs = _hypothesis_trajectories_abs(
            ego_mu, sequences, ego_ids, hist_len)  # (H, B, T_pred, 2)
        probs_np = ego_pred_scores.detach().cpu().numpy()  # (B, H)
        H = traj_abs.shape[0]

        # RAW (un-rotated) sigma -- see case_risk_dynamics.py's sigma_xy
        # docstring note for why this isn't in the same frame as hyp_xy.
        start_heading_deg = float(
            sequences[b, ego_id, hist_len - 1, G.HD].detach().cpu().item())

        real_ego_xy = sequences[b, ego_id, hist_len:, G.XY].detach().cpu().numpy()

        if ref_agent_idx_override is not None:
            ref_agent_idx = ref_agent_idx_override
        else:
            # Auto-detect reference agent from the REALIZED trajectory --
            # same logic as case_risk_dynamics.py, computed independently.
            A_total = sequences.shape[1]
            other_idx, other_xy, other_valid = [], [], []
            for a in range(A_total):
                if a == ego_id:
                    continue
                valid_a = agent_masks[b, a, hist_len:].bool().detach().cpu().numpy()
                if not valid_a.any():
                    continue
                other_idx.append(a)
                other_xy.append(sequences[b, a, hist_len:, G.XY].detach().cpu().numpy())
                other_valid.append(valid_a)
            if not other_idx:
                raise RuntimeError(f"No valid other agent for scene_file={target_scene_file}")
            other_xy_arr = np.stack(other_xy, axis=0)
            other_valid_arr = np.stack(other_valid, axis=0)
            dist = np.linalg.norm(other_xy_arr - real_ego_xy[None, :, :], axis=-1)
            dist = np.where(other_valid_arr, dist, np.inf)
            agent_i, _t_i = np.unravel_index(np.argmin(dist), dist.shape)
            ref_agent_idx = other_idx[agent_i]

        ref_xy = sequences[b, ref_agent_idx, hist_len:, G.XY].detach().cpu().numpy()
        ref_valid = agent_masks[b, ref_agent_idx, hist_len:].bool().detach().cpu().numpy()
        ref_type = int(agent_types_flat[b, ref_agent_idx].item())

        real_dist = np.linalg.norm(real_ego_xy - ref_xy, axis=-1)
        real_dist = np.where(ref_valid, real_dist, np.nan) * 1000

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

        hyps_out = []
        for h in range(H):
            hyp_xy = traj_abs[h, b]  # (T_pred, 2)
            dist_h = np.linalg.norm(hyp_xy - ref_xy, axis=-1)
            dist_h = np.where(ref_valid, dist_h, np.nan) * 1000
            hyp_xy_masked = np.where(ref_valid[:, None], hyp_xy, np.nan)
            hyp_sigma = ego_sigma[b, hist_len:, h, :2].detach().cpu().numpy()  # (T_pred, 2)
            hyps_out.append({
                "prob": float(probs_np[b, h]),
                "dist_m": [None if np.isnan(v) else float(v) for v in dist_h],
                "latlon": xy_array_to_latlon(hyp_xy_masked, ref),
                "sigma_xy": [[float(sx), float(sy)] for sx, sy in hyp_sigma],
            })

        out = {
            "meta": {
                "airport": airport,
                "batch_idx": None,
                "sample_idx": dataset_idx,
                "scene_file": target_scene_file,
                "ego_id": ego_id,
                "ref_agent_idx": ref_agent_idx,
                "ref_agent_type": AGENT_TYPE_NAMES.get(ref_type, str(ref_type)),
                "hist_len": hist_len,
                "start_heading_deg": start_heading_deg,
            },
            "gt": gt_out,
            "hypotheses": hyps_out,
            "real": {"dist_m": [None if np.isnan(v) else float(v) for v in real_dist]},
        }
        with open(output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[case_trajectories_amelia_baseline] wrote {output_json}")


if __name__ == "__main__":
    main()
