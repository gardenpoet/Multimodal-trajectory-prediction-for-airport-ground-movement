"""
Two-stage training pipeline for aircraft trajectory prediction.

Stage 1: Train mode prediction model (classification)
Stage 2: Train trajectory prediction model with teacher forcing (regression)
Stage 3: Optional end-to-end testing with combined models

E2E ADDITION (e2e_finetune=true): load pretrained traj_model, enable the GMM
score head, jointly fine-tune backbone + score head. ALL ORIGINAL CODE BELOW
IS UNCHANGED; only two methods (finetune_trajectory_model_e2e, _e2e_test) and
one main() branch are added.

Author: Amelia TF Team
"""
import hydra
import lightning as L
import pyrootutils
import torch
import json
import shutil
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
        config_path.parent.mkdir(parents=True, exist_ok=True)
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
        """
        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = utils.instantiate_callbacks(self.cfg.get("callbacks"))

        callbacks = [
            cb for cb in callbacks
            if not isinstance(cb, (
                L.pytorch.callbacks.ModelCheckpoint,
                L.pytorch.callbacks.EarlyStopping,
            ))
        ]

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

        callbacks.append(
            L.pytorch.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=self.cfg.patience,
                verbose=True,
            )
        )

        log.info("Instantiating loggers...")
        logger: List[Logger] = utils.instantiate_loggers(self.cfg.get("logger"))

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
        """Load model weights from a Lightning checkpoint with flexible key handling."""
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
        backbone_missing = [k for k in missing if 'score_head' not in k]
        if backbone_missing:
            log.error(f"*** BACKBONE KEYS NOT LOADED *** ({len(backbone_missing)}): {backbone_missing[:10]}")
        if missing:
            log.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Successfully loaded {model_type} model")

    def train_mode_model(self, datamodule: LightningDataModule) -> None:
        """Stage 1: Train the mode prediction model."""
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
            monitor="val/modal_accuracy",
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

        # Copy the best checkpoint to the fixed `mode_ckpt_path` location (if
        # configured) so that other runs (e.g. the 2T/4T sweep for the same
        # airport+horizon) can reuse it via skip_mode_training=True, instead of
        # retraining an identical mode classifier from scratch each time.
        fixed_mode_ckpt = self.cfg.get("mode_ckpt_path", None)
        if fixed_mode_ckpt:
            fixed_mode_ckpt = Path(fixed_mode_ckpt)
            fixed_mode_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.mode_ckpt_path, fixed_mode_ckpt)
            log.info(f"Copied best mode checkpoint to fixed reuse path: {fixed_mode_ckpt}")

    def load_pretrained_mode_model(self, checkpoint_path: Path) -> None:
        """
        Skip Stage 1 training and reuse an existing mode-model checkpoint.

        Used when sweeping num_futures for Stage 2: the mode classifier does
        not depend on num_futures, so it only needs to be trained once per
        airport+horizon (see train_mode_model's fixed-path copy above).
        """
        log.info("=" * 70)
        log.info("Skipping Stage 1 (mode) training - loading existing checkpoint")
        log.info("=" * 70)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Mode model checkpoint not found: {checkpoint_path}. "
                "Run a mode-training job first (skip_mode_training=False) "
                "for this airport+horizon before sweeping num_futures."
            )

        self.mode_ckpt_path = checkpoint_path
        log.info(f"Using pretrained mode checkpoint: {self.mode_ckpt_path}")

        if self.cfg.get("save_model_paths", False):
            with open(self.output_dir / "mode_model" / "best_model_path.txt", "w") as f:
                f.write(str(self.mode_ckpt_path))

    def train_trajectory_model(self, datamodule: LightningDataModule) -> None:
        """Stage 2: Train the trajectory prediction model with teacher forcing."""
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
            monitor="val/ade/t=max",
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
                
    def train_traj_scratch_joint(self, datamodule):
        """
        Train traj model FROM SCRATCH with joint loss (WTA + score selection).
        Mode model is NOT retrained: it is loaded from mode_ckpt_path for the
        final test. Only traj is trained here (Stage 2 equivalent, from scratch,
        with score head enabled and score_detach=False).
        """
        log.info("=" * 70)
        log.info("FROM-SCRATCH joint training: traj backbone + score head")
        log.info("=" * 70)
        self._save_config("traj_model_scratch")
    
        # build traj net FRESH (random init) with score head ON
        traj_net = hydra.utils.instantiate(self.cfg.model.traj_net)
    
        from amelia_tf.models.components.gmm import GMM
        gmm = next((m for m in traj_net.modules() if isinstance(m, GMM)), None)
        assert gmm is not None and getattr(gmm, "enable_score_head", False) \
            and gmm.score_mode != 0, \
            "Enable score head: enable_score_head=true, score_mode=4."
        if gmm.score_detach:
            log.warning("[scratch] score_detach=True -> forcing False (joint training).")
            gmm.score_detach = False
    
        traj_model = TrajectoryPredictionModel(
            optimizer=hydra.utils.instantiate(self.cfg.model.traj_optimizer),
            scheduler=hydra.utils.instantiate(self.cfg.model.traj_scheduler)
                if self.cfg.model.get("traj_scheduler") else None,
            net=traj_net,
            extra_params=self.cfg.model.extra_params)
    
        trainer = self._create_trainer(
            stage="traj_model_scratch",
            monitor="val/selected_ade/t=max", mode="min",
            max_epochs=self.cfg.traj_epochs)
    
        if getattr(trainer, "logger", None):
            utils.log_hyperparameters({
                "cfg": self.cfg, "model": traj_model, "trainer": trainer})
    
        trainer.fit(model=traj_model, datamodule=datamodule)
        self.traj_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
        log.info(f"[scratch] best traj checkpoint: {self.traj_ckpt_path}")
    
        # ---- auto-test: load EXISTING mode + trained traj, score selection ----
        self._e2e_test(datamodule)   # reuses the same test path as e2e

    def run_end_to_end_test(self, datamodule: LightningDataModule) -> Dict[str, Any]:
        """Stage 3: End-to-End evaluation with the combined two-stage system."""
        log.info("=" * 70)
        log.info("Stage 3: End-to-End Testing with Combined Models")
        log.info("=" * 70)

        if self.mode_ckpt_path is None or self.traj_ckpt_path is None:
            raise ValueError("Both mode and trajectory models must be trained before Stage 3")

        mode_net = hydra.utils.instantiate(self.cfg.model.mode_net)
        traj_net = hydra.utils.instantiate(self.cfg.model.traj_net)

        self._load_model_state(mode_net, self.mode_ckpt_path,  "mode",       strict=False)
        self._load_model_state(traj_net, self.traj_ckpt_path,  "trajectory", strict=False)

        combined_model = CombinedTrajPredSystem(
            mode_model=mode_net,
            traj_model=traj_net,
            extra_params=self.cfg.model.extra_params
        )

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

    # ==================================================================
    # NEW: end-to-end score-head fine-tuning (added; nothing above changed)
    # ==================================================================
    def finetune_trajectory_model_e2e(self, datamodule: LightningDataModule) -> None:
        """
        Load pretrained traj_model, enable the GMM score head, and JOINTLY
        fine-tune backbone + score head with loss = traj_WTA + lambda*score_CE
        (the score loss is added inside TrajectoryPredictionModel.model_step).

        EarlyStopping on val/ade guards against the backbone degrading
        trajectories to make scoring easy. The fine-tuned checkpoint is saved
        to the SAME directory as the original traj checkpoint but with an
        `_e2e` filename, so the original is never overwritten.
        """
        log.info("=" * 70)
        log.info("E2E: Joint fine-tune of trajectory backbone + score head")
        log.info("=" * 70)
        self._save_config("traj_model_e2e")

        pre_ckpt = self.cfg.get("traj_ckpt_path", None)
        if pre_ckpt is None:
            raise ValueError("Set traj_ckpt_path to the pretrained traj checkpoint.")

        traj_net = hydra.utils.instantiate(self.cfg.model.traj_net)
        self._load_model_state(traj_net, Path(pre_ckpt), "trajectory(pretrained)", strict=False)

        # score head must be enabled and NOT detached for end-to-end
        from amelia_tf.models.components.gmm import GMM
        gmm = next((m for m in traj_net.modules() if isinstance(m, GMM)), None)
        if gmm is None:
            raise RuntimeError("GMM head not found in traj_net.")
        assert getattr(gmm, "enable_score_head", False) and gmm.score_mode != 0, \
            "Enable score head in config: enable_score_head=true, score_mode=4."
        if gmm.score_detach:
            log.warning("[e2e] score_detach=True -> forcing False so gradients reach the backbone.")
            gmm.score_detach = False

        traj_model = TrajectoryPredictionModel(
            optimizer=hydra.utils.instantiate(self.cfg.model.traj_optimizer),
            scheduler=hydra.utils.instantiate(self.cfg.model.traj_scheduler)
                if self.cfg.model.get("traj_scheduler") else None,
            net=traj_net,
            extra_params=self.cfg.model.extra_params
        )

        # checkpoint to the SAME dir as the original, with an _e2e name
        traj_dir = Path(pre_ckpt).parent
        orig_stem = Path(pre_ckpt).stem
        log.info("Instantiating callbacks...")
        callbacks = utils.instantiate_callbacks(self.cfg.get("callbacks"))
        callbacks = [cb for cb in callbacks if not isinstance(
            cb, (L.pytorch.callbacks.ModelCheckpoint, L.pytorch.callbacks.EarlyStopping))]
        ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(traj_dir),
            filename=f"{orig_stem}_e2e-{{epoch:02d}}",
            monitor="val/ade/t=max", mode="min",
            save_top_k=self.cfg.get("save_top_k", 1), save_last=False)
        early_cb = L.pytorch.callbacks.EarlyStopping(
            monitor="val/ade/t=max", mode="min", patience=self.cfg.patience, verbose=True)
        callbacks += [ckpt_cb, early_cb]

        logger = utils.instantiate_loggers(self.cfg.get("logger"))
        trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            max_epochs=self.cfg.get("e2e_epochs", 10),
            default_root_dir=str(self.output_dir / "traj_model_e2e"),
            callbacks=callbacks, logger=logger)

        if getattr(trainer, "logger", None):
            utils.log_hyperparameters({
                "cfg": self.cfg, "model": traj_model, "trainer": trainer})

        trainer.fit(model=traj_model, datamodule=datamodule)
        self.traj_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
        log.info(f"[e2e] fine-tuned traj checkpoint: {self.traj_ckpt_path}")
        log.info(f"[e2e] original untouched at: {pre_ckpt}")

        self._e2e_test(datamodule)

    def _e2e_test(self, datamodule: LightningDataModule) -> None:
        """Auto-test after e2e fine-tune: mode + fine-tuned traj, score selection."""
        from omegaconf import open_dict
        log.info("[e2e] running test with score-based selection...")

        mode_ckpt = self.cfg.get("mode_ckpt_path", None)
        if mode_ckpt is None:
            raise ValueError("Set mode_ckpt_path for the e2e test phase.")
        mode_net = hydra.utils.instantiate(self.cfg.model.mode_net)
        self._load_model_state(mode_net, Path(mode_ckpt), "mode", strict=False)

        best_traj = TrajectoryPredictionModel.load_from_checkpoint(
            str(self.traj_ckpt_path),
            net=hydra.utils.instantiate(self.cfg.model.traj_net),
            optimizer=hydra.utils.instantiate(self.cfg.model.traj_optimizer),
            scheduler=hydra.utils.instantiate(self.cfg.model.traj_scheduler)
                if self.cfg.model.get("traj_scheduler") else None,
            extra_params=self.cfg.model.extra_params).net
        best_traj.eval()

        torch.use_deterministic_algorithms(False)
        with open_dict(self.cfg):
            self.cfg.model.extra_params.selection_mode = 'score'

        combined = CombinedTrajPredSystem(
            mode_model=mode_net, traj_model=best_traj,
            extra_params=self.cfg.model.extra_params)
        logger = utils.instantiate_loggers(self.cfg.get("logger"))
        test_trainer = hydra.utils.instantiate(
            self.cfg.trainer, logger=logger,
            default_root_dir=str(self.output_dir / "combined_e2e"))
        results = test_trainer.test(model=combined, datamodule=datamodule)
        out = self.output_dir / "combined_e2e" / "results"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "test_metrics.json", "w") as f:
            json.dump(results[0] if results else {}, f, indent=2)
        log.info(f"[e2e] test metrics -> {out / 'test_metrics.json'}")

    def run(self) -> Dict[str, Any]:
        """Execute the complete two-stage training pipeline."""
        if self.cfg.get("seed"):
            L.seed_everything(self.cfg.seed, workers=True)

        log.info(f"Instantiating datamodule <{self.cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.data)

        # Stage 1: train mode model, or reuse an existing checkpoint. Mode
        # classification does not depend on num_futures, so it only needs to
        # be trained once per airport+horizon; see load_pretrained_mode_model.
        if self.cfg.get("skip_mode_training", False):
            mode_ckpt_path = self.cfg.get("mode_ckpt_path", None)
            if mode_ckpt_path is None:
                raise ValueError("skip_mode_training=True but mode_ckpt_path is not set")
            self.load_pretrained_mode_model(Path(mode_ckpt_path))
        else:
            self.train_mode_model(datamodule)

        # Mode-only run: stop here so a dedicated "train mode once" job does
        # not also pay for a full Stage 2 trajectory training.
        if self.cfg.get("skip_traj_training", False):
            log.info("skip_traj_training=True: stopping after Stage 1 (mode-only run).")
            return {
                "mode_model_path": str(self.mode_ckpt_path) if self.mode_ckpt_path else None,
                "traj_model_path": None,
                "test_results": {}
            }

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_two_stage_klax")
def main(cfg: DictConfig) -> None:
    """Main entry point for two-stage training (and e2e fine-tune)."""
    utils.extras(cfg)
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ---- E2E fine-tune branch: skip from-scratch training entirely ----
    if cfg.get("e2e_finetune", False):
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)
        trainer = TwoStageTrainer(cfg)
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule = hydra.utils.instantiate(cfg.data)
        trainer.finetune_trajectory_model_e2e(datamodule)
        return
        
    # ---- from-scratch joint training (WTA + score selection) ----
    if cfg.get("scratch_joint", False):
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)
        trainer = TwoStageTrainer(cfg)
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule = hydra.utils.instantiate(cfg.data)
        trainer.train_traj_scratch_joint(datamodule)
        return

    # ---- original from-scratch two-stage pipeline (unchanged) ----
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