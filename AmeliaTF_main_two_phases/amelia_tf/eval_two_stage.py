import hydra
import pyrootutils
import torch
import json
from pathlib import Path
from typing import Dict, Any, List

import lightning as L
from lightning import LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from amelia_tf import utils
from amelia_tf.models.traj_pred_combined import CombinedTrajPredSystem

log = utils.get_pylogger(__name__)


def load_model_state(model, ckpt_path: str):
    """Load model weights from Lightning checkpoint."""
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # remove Lightning prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_key = k[len("net."):]
        elif k.startswith("mode_model.net."):
            new_key = k[len("mode_model.net."):]
        elif k.startswith("traj_model.net."):
            new_key = k[len("traj_model.net."):]
        else:
            new_key = k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_two_stage")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # ====== seed ======
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # ====== datamodule ======
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # ====== load models ======
    log.info("Instantiating networks...")
    mode_net = hydra.utils.instantiate(cfg.model.mode_net)
    traj_net = hydra.utils.instantiate(cfg.model.traj_net)

    # load checkpoints
    mode_net = load_model_state(mode_net, cfg.mode_ckpt_path)
    traj_net = load_model_state(traj_net, cfg.traj_ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_net.to(device)
    traj_net.to(device)

    # ====== combined model ======
    log.info("Building combined model...")
    model = CombinedTrajPredSystem(
        mode_model=mode_net,
        traj_model=traj_net,
        extra_params=cfg.model.extra_params
    )

    # ====== logger ======
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # ====== trainer ======
    log.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        default_root_dir=str(Path(cfg.paths.output_dir) / "test"),
    )

    # ====== test ======
    log.info("Starting testing...")
    results = trainer.test(model=model, datamodule=datamodule)

    # ====== save results ======
    output_dir = Path(cfg.paths.output_dir) / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "test_metrics.json"
    with open(result_path, "w") as f:
        json.dump(results[0] if results else {}, f, indent=2)

    log.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()