"""
One-off diagnostic: for a single scene, print the mode-classification
softmax BOTH with the turn-feasibility hard mask applied (the normal,
as-deployed output -- should match the mode_prob values already in that
case's case_risk_dynamics.py JSON) AND with the mask disabled, to recover
the RAW pre-mask probability the model would have assigned an
infeasibility-excluded mode.

Mechanism (see AmeliaMode.forward(), amelia_tf/models/components/
amelia_mode.py): mode_logits get masked_fill(-1e9) at infeasible entries
BEFORE softmax, controlled by the AmeliaMode instance's own
self.apply_hard_mask (default True, read once at __init__, a plain
attribute afterward -- not re-read from config per call). Flipping that
attribute on the already-built model instance is the minimal way to get
an unmasked forward pass without touching any other code path (feature
inputs / soft feasibility conditioning are untouched; only the hard
masked_fill is skipped).

Does NOT change any file, retrain anything, or affect any other script's
output -- this is read-only diagnostic, single scene, single forward
pass twice.

Usage (same data/paths/model composition as case_risk_dynamics.py):

    python -m risk_assessment.case_raw_mode_prob \\
        ckpt=kbos2 data=kbos.yaml \\
        mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \\
        traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \\
        model.traj_net.config.num_hypotheses=4 \\
        +data.dataset.config.random_ego=false \\
        +case_batch_idx=7 +case_sample_idx=62
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

from amelia_tf.eval_two_stage import _build_nets
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils.modes import TURN_MODES

from risk_assessment.common import to_device, seed_for_reproducible_ego_selection

MODE_NAMES = TURN_MODES


@hydra.main(version_base="1.3", config_path="../AmeliaTF_main_two_phases_4T/configs",
            config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    target_batch = cfg.get("case_batch_idx")
    target_sample = cfg.get("case_sample_idx")
    if target_batch is None or target_sample is None:
        raise ValueError("Pass +case_batch_idx=, +case_sample_idx=")
    target_batch, target_sample = int(target_batch), int(target_sample)

    mode_net, traj_net, device = _build_nets(cfg)
    model = CombinedTrajPredSystem(
        mode_model=mode_net, traj_model=traj_net, extra_params=cfg.model.extra_params)
    model.to(device)
    model.eval()

    datamodule = hydra.utils.instantiate(cfg.data)
    seed_for_reproducible_ego_selection(datamodule, seed=cfg.get("seed", 42))
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > target_batch:
                raise RuntimeError(f"Never reached batch {target_batch} (stopped at {batch_idx})")
            if batch_idx != target_batch:
                continue
            batch = to_device(batch, device)
            b = target_sample

            # Pass 1: as-deployed, hard mask applied (should reproduce the
            # mode_prob values already in this case's dynamics JSON).
            model.mode_model.apply_hard_mask = True
            mode_probs_masked, _, _, _ = model.forward(batch)
            ego_agent = batch['scene_dict']['ego_agent_id']
            ego_ids = [
                ego_agent[i].item() if torch.is_tensor(ego_agent) else int(ego_agent[i])
                for i in range(mode_probs_masked.shape[0])
            ]
            ego_masked = separate_ego_agent(mode_probs_masked, ego_ids).squeeze(1)[b]

            # Pass 2: hard mask disabled -- raw softmax over all 4 mode
            # logits, i.e. what the model "really thinks" before the
            # feasibility constraint zeroes the infeasible ones out.
            model.mode_model.apply_hard_mask = False
            mode_probs_raw, _, _, _ = model.forward(batch)
            ego_raw = separate_ego_agent(mode_probs_raw, ego_ids).squeeze(1)[b]
            model.mode_model.apply_hard_mask = True  # restore, tidiness only

            print(f"batch={target_batch} sample={target_sample} ego_id={ego_ids[b]}")
            print(f"{'mode':<12}{'masked (as-deployed)':<22}{'raw (mask disabled)':<22}")
            for m, name in enumerate(MODE_NAMES):
                print(f"{name:<12}{ego_masked[m].item():<22.10f}{ego_raw[m].item():<22.10f}")
            return


if __name__ == "__main__":
    main()
