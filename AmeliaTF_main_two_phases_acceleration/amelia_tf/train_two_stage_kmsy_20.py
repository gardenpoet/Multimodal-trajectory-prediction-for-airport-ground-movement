"""
Two-stage training pipeline for aircraft trajectory prediction.

Stage 1: Train mode prediction model (classification)
Stage 2: Train trajectory prediction model with teacher forcing (regression)
Stage 3: Optional end-to-end testing with combined models

Author: Amelia TF Team
"""
import hydra
import lightning as L
import pyrootutils
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from lightning import Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from amelia_tf import utils
from amelia_tf.models.traj_pred_combined import (
    ModePredictionModel,
    TrajectoryPredictionModel,
    CombinedTrajPredSystem
)

log = utils.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class TwoStageTrainer:
    """
    Manages two-stage training pipeline for trajectory prediction.

    Handles checkpointing, model loading, and end-to-end evaluation.
    Follows the same trainer/callback/logger instantiation pattern as train.py.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.paths.output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._create_directories()

        self.mode_ckpt_path: Optional[Path] = None
        self.traj_ckpt_path: Optional[Path] = None

    def _create_directories(self) -> None:
        """Create necessary output directories."""
        dirs = [
            self.output_dir / "mode_model" / "checkpoints",
            self.output_dir / "mode_model" / "logs",
            self.output_dir / "traj_model" / "checkpoints",
            self.output_dir / "traj_model" / "logs",
            self.output_dir / "combined" / "results",
            self.output_dir / "combined" / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _save_config(self, stage: str) -> None:
        """Save configuration for reproducibility."""
        config_path = self.output_dir / stage / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(self.cfg, f)

    def _create_trainer(
        self,
        stage: str,
        monitor: str,
        mode: str,
        max_epochs: int,
    ) -> L.Trainer:
        """
        Instantiate a Lightning Trainer following the same pattern as train.py.

        Callbacks and loggers are built from Hydra config. ModelCheckpoint and
        EarlyStopping are stripped from the yaml-instantiated list and replaced
        with stage-specific versions so that each stage monitors the correct
        metric and saves to its own directory.

        Args:
            stage:      Training stage name, used for directory paths.
            monitor:    Metric to monitor for checkpointing / early stopping.
            mode:       'min' or 'max'.
            max_epochs: Maximum training epochs for this stage.

        Returns:
            Configured Lightning Trainer instance.
        """
        # Instantiate all callbacks from Hydra config
        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = utils.instantiate_callbacks(self.cfg.get("callbacks"))

        # Strip ModelCheckpoint and EarlyStopping from the yaml list —
        # they are stage-specific and must be re-added with the correct
        # monitor metric and output directory for each stage.
        callbacks = [
            cb for cb in callbacks
            if not isinstance(cb, (
                L.pytorch.callbacks.ModelCheckpoint,
                L.pytorch.callbacks.EarlyStopping,
            ))
        ]

        # Stage-specific ModelCheckpoint
        callbacks.append(
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=self.output_dir / stage / "checkpoints",
                filename=f"{stage}-{{epoch:02d}}-{{{monitor}:.4f}}",
                monitor=monitor,
                mode=mode,
                save_top_k=self.cfg.get("save_top_k", 3),
                save_last=True,
            )
        )

        # Stage-specific EarlyStopping
        callbacks.append(
            L.pytorch.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=self.cfg.patience,
                verbose=True,
            )
        )

        # Loggers from Hydra config (WandB, CSV, TensorBoard, etc.)
        log.info("Instantiating loggers...")
        logger: List[Logger] = utils.instantiate_loggers(self.cfg.get("logger"))

        # Trainer from Hydra config, override stage-specific fields
        log.info(f"Instantiating trainer for stage: {stage}")
        trainer: L.Trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            max_epochs=max_epochs,
            default_root_dir=str(self.output_dir / stage),
            callbacks=callbacks,
            logger=logger,
        )

        return trainer

    def _load_model_state(
        self,
        model: torch.nn.Module,
        checkpoint_path: Path,
        model_type: str,
        strict: bool = False
    ) -> None:
        """
        Load model weights from a Lightning checkpoint with flexible key handling.

        Strips common Lightning wrapper prefixes ('net.', 'mode_model.net.',
        'traj_model.net.') so that raw network modules can be loaded from
        checkpoints saved by any of the three Lightning wrappers.

        Args:
            model:           Target nn.Module to load weights into.
            checkpoint_path: Path to the .ckpt file.
            model_type:      Human-readable label for log messages.
            strict:          Whether to enforce strict key matching.
        """
        log.info(f"Loading {model_type} model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        prefix_map = {
            'net.':            '',
            'mode_model.net.': '',
            'traj_model.net.': '',
        }
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix, replacement in prefix_map.items():
                if k.startswith(prefix):
                    new_key = replacement + k[len(prefix):]
                    break
            cleaned_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=strict)
        if missing:
            log.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Successfully loaded {model_type} model")

    def train_mode_model(self, datamodule: LightningDataModule) -> None:
        """
        Stage 1: Train the mode prediction model.

        Monitors validation mode accuracy and saves the best checkpoint.
        """
        log.info("=" * 70)
        log.info("Stage 1: Training Mode Prediction Model")
        log.info("=" * 70)

        self._save_config("mode_model")

        mode_model = ModePredictionModel(
            optimizer=hydra.utils.instantiate(self.cfg.model.mode_optimizer),
            scheduler=hydra.utils.instantiate(self.cfg.model.mode_scheduler)
                if self.cfg.model.get("mode_scheduler") else None,
            net=hydra.utils.instantiate(self.cfg.model.mode_net),
            extra_params=self.cfg.model.extra_params
        )

        trainer = self._create_trainer(
            stage="mode_model",
            monitor="val_mode_acc",
            mode="max",
            max_epochs=self.cfg.mode_epochs
        )

        if getattr(trainer, "logger", None):
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters({
                "cfg": self.cfg,
                "model": mode_model,
                "trainer": trainer,
            })

        trainer.fit(model=mode_model, datamodule=datamodule)

        if self.cfg.get("test", False):
            trainer.test(model=mode_model, datamodule=datamodule)

        self.mode_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
        log.info(f"Best mode model saved to: {self.mode_ckpt_path}")

        if self.cfg.get("save_model_paths", False):
            with open(self.output_dir / "mode_model" / "best_model_path.txt", "w") as f:
                f.write(str(self.mode_ckpt_path))

    def train_trajectory_model(self, datamodule: LightningDataModule) -> None:
        """
        Stage 2: Train the trajectory prediction model with teacher forcing.

        Uses ground-truth turn modes during training. Monitors validation ADE
        at the maximum prediction horizon.
        """
        log.info("=" * 70)
        log.info("Stage 2: Training Trajectory Prediction Model")
        log.info("=" * 70)

        self._save_config("traj_model")

        traj_model = TrajectoryPredictionModel(
            optimizer=hydra.utils.instantiate(self.cfg.model.traj_optimizer),
            scheduler=hydra.utils.instantiate(self.cfg.model.traj_scheduler)
                if self.cfg.model.get("traj_scheduler") else None,
            net=hydra.utils.instantiate(self.cfg.model.traj_net),
            extra_params=self.cfg.model.extra_params
        )

        trainer = self._create_trainer(
            stage="traj_model",
            monitor="val_ade/t=max",
            mode="min",
            max_epochs=self.cfg.traj_epochs
        )

        if getattr(trainer, "logger", None):
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters({
                "cfg": self.cfg,
                "model": traj_model,
                "trainer": trainer,
            })

        trainer.fit(model=traj_model, datamodule=datamodule)

        if self.cfg.get("test", False):
            trainer.test(model=traj_model, datamodule=datamodule)

        self.traj_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
        log.info(f"Best trajectory model saved to: {self.traj_ckpt_path}")

        if self.cfg.get("save_model_paths", False):
            with open(self.output_dir / "traj_model" / "best_model_path.txt", "w") as f:
                f.write(str(self.traj_ckpt_path))

    def run_end_to_end_test(self, datamodule: LightningDataModule) -> Dict[str, Any]:
        """
        Stage 3: End-to-end evaluation with the combined two-stage system.

        Loads the best checkpoints from Stage 1 and 2 into CombinedTrajPredSystem
        and runs the test loop - no teacher forcing.
        """
        log.info("=" * 70)
        log.info("Stage 3: End-to-End Testing with Combined Models")
        log.info("=" * 70)

        if self.mode_ckpt_path is None or self.traj_ckpt_path is None:
            raise ValueError("Both mode and trajectory models must be trained before Stage 3")

        # Instantiate raw networks and load pre-trained weights
        mode_net = hydra.utils.instantiate(self.cfg.model.mode_net)
        traj_net = hydra.utils.instantiate(self.cfg.model.traj_net)

        self._load_model_state(mode_net, self.mode_ckpt_path,  "mode",       strict=False)
        self._load_model_state(traj_net, self.traj_ckpt_path,  "trajectory", strict=False)

        combined_model = CombinedTrajPredSystem(
            mode_model=mode_net,
            traj_model=traj_net,
            extra_params=self.cfg.model.extra_params
        )

        # Reuse logger from config for Stage 3
        logger: List[Logger] = utils.instantiate_loggers(self.cfg.get("logger"))

        trainer: L.Trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            default_root_dir=str(self.output_dir / "combined"),
            logger=logger,
        )

        results = trainer.test(model=combined_model, datamodule=datamodule)

        results_path = self.output_dir / "combined" / "results" / "test_metrics.json"
        with open(results_path, "w") as f:
            json.dump(results[0] if results else {}, f, indent=2)
        log.info(f"End-to-end test results saved to: {results_path}")

        return results[0] if results else {}

    def run(self) -> Dict[str, Any]:
        """Execute the complete two-stage training pipeline."""
        if self.cfg.get("seed"):
            L.seed_everything(self.cfg.seed, workers=True)

        log.info(f"Instantiating datamodule <{self.cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.data)

        self.train_mode_model(datamodule)
        self.train_trajectory_model(datamodule)

        results = {}
        if self.cfg.get("run_end_to_end_test", False):
            results = self.run_end_to_end_test(datamodule)

        log.info("=" * 70)
        log.info("Two-stage training completed successfully!")
        log.info(f"All outputs saved to: {self.output_dir}")
        log.info("=" * 70)

        return {
            "mode_model_path": str(self.mode_ckpt_path) if self.mode_ckpt_path else None,
            "traj_model_path": str(self.traj_ckpt_path) if self.traj_ckpt_path else None,
            "test_results":    results
        }


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_two_stage_kmsy_20")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for two-stage training.

    Args:
        cfg: Hydra configuration object built from train_two_stage.yaml.
    """
    # Apply extra utilities (tag enforcement, config printing, etc.)
    utils.extras(cfg)

    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate required config keys before starting any training
    required_keys = [
        "paths.output_dir",
        "model.mode_optimizer", "model.mode_net",
        "model.traj_optimizer", "model.traj_net",
        "mode_epochs", "traj_epochs"
    ]
    for key in required_keys:
        if not OmegaConf.select(cfg, key):
            raise ValueError(f"Missing required config key: {key}")

    trainer = TwoStageTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()