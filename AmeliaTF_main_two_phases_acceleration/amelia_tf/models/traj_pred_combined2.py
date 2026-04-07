import itertools
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import date
from easydict import EasyDict
from geographiclib.geodesic import Geodesic
from lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Any, Dict, Tuple, Optional, Union, List

from amelia_tf.models.components.common import LayerNorm
from amelia_tf.utils.utils import plot_scene_batch, separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODE_MAP, MODE_NAMES, VALID_TURN_MODES, TURN_MODES_NAMES
from amelia_tf.utils.metrics import (
    ModalClassificationMetrics, compute_mode_accuracy,
    marginal_ade, marginal_fde, joint_ade, joint_fde,
    compute_mode_rmse, compute_nll
)

np.printoptions(precision=5, suppress=True)


class ModePredictionModel(LightningModule):
    """
    Turn-mode prediction model for aircraft trajectory forecasting.

    Predicts discrete turn modes (left/right/straight/etc.) based on historical
    trajectory and contextual information. Used as a conditioning signal for
    trajectory prediction models.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            net: torch.nn.Module,
            extra_params: EasyDict
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'], logger=False)

        # Core components
        self.net = net
        self.hist_len = self.net.hist_len
        self.num_modes = self.net.num_modes
        self.eparams = extra_params

        # Mode configuration
        self.MODE_MAP = TURN_MODE_MAP
        self.valid_modes = VALID_TURN_MODES
        self.mode_names = TURN_MODES_NAMES

        # Loss metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Accuracy metrics
        self.val_mode_acc = MeanMetric()
        self.test_mode_acc = MeanMetric()

        # Per-mode accuracy tracking
        self.val_per_mode_acc = nn.ModuleDict({
            self.mode_names[mode_idx]: MeanMetric()
            for mode_idx in self.valid_modes
        })
        self.test_per_mode_acc = nn.ModuleDict({
            self.mode_names[mode_idx]: MeanMetric()
            for mode_idx in self.valid_modes
        })

        # Comprehensive classification metrics
        self.val_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_modes,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )
        self.test_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_modes,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )

        self.mode_weights = None

    def setup(self, stage: str) -> None:
        """Inject class weights from datamodule for imbalanced mode classification."""
        if stage == "fit":
            dm = self.trainer.datamodule
            if hasattr(dm, "mode_weights"):
                self.mode_weights = dm.mode_weights.to(self.device)
                print(f"Mode weights injected: {self.mode_weights}")
            else:
                raise RuntimeError("Datamodule missing required 'mode_weights' attribute")

    def _encode_rule_based_to_mode_index(self, rule_based: torch.Tensor) -> torch.Tensor:
        """
        Convert rule-based encoding to turn-only mode index (0-3).

        Args:
            rule_based: Tensor of shape (..., 8) where first 4 dims are turn one-hot

        Returns:
            Long tensor of mode indices
        """
        turn_idx = rule_based[..., :4].float().argmax(dim=-1)
        return turn_idx.long()

    def model_step(
            self,
            batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute one step of mode prediction.

        Args:
            batch: Input batch containing scene data

        Returns:
            loss: Cross-entropy loss for mode classification
            mode_logits: Raw logits from the network
            true_mode_idx: Ground truth mode indices (or None if unavailable)
        """
        # Extract input data
        Y = batch['scene_dict']['rel_sequences']
        Y_mode = batch['scene_dict'].get('rule_based_encoding')

        # Prepare historical input (position only)
        X = torch.zeros_like(Y[..., :4]).float()
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len, :4]

        # Network forward pass
        mode_logits = self.net(
            X,
            context=batch['scene_dict']['context'],
            adjacency=batch['scene_dict']['adjacency'],
            mask=None,
            output_mode_only=True
        )

        # Compute loss if ground truth available
        if Y_mode is not None:
            true_mode_idx = self._encode_rule_based_to_mode_index(Y_mode)
            ego_agent = batch['scene_dict']['ego_agent_id']
            true_mode_idx_ego = separate_ego_agent(true_mode_idx, ego_agent)
            ego_mode_logits = separate_ego_agent(mode_logits, ego_agent)

            loss = F.cross_entropy(
                ego_mode_logits.view(-1, ego_mode_logits.shape[-1]),
                true_mode_idx_ego.view(-1),
                weight=self.mode_weights,
                label_smoothing=0.1,
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=self.device)
            true_mode_idx_ego = None

        return loss, mode_logits, true_mode_idx_ego

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        loss, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "losses/train_mode",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Execute validation step with detailed metrics."""
        loss, mode_logits, true_mode_idx = self.model_step(batch)

        # Extract ego agent data
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']
        ego_mask = separate_ego_agent(masks, ego_agent)
        mask_agent_level = ego_mask.any(dim=-1).float()

        ego_mode_logits = separate_ego_agent(mode_logits, ego_agent)
        ego_true_mode = true_mode_idx
        ego_probs = torch.softmax(ego_mode_logits, dim=-1)

        # Update accumulator metrics
        self.val_modal_accumulator.update(ego_probs, ego_true_mode, mask_agent_level)

        # Compute and log accuracy
        mode_acc = compute_mode_accuracy(ego_probs, ego_true_mode, mask_agent_level)
        self.val_mode_acc(mode_acc)
        self.val_loss(loss)

        self.log("losses/val_mode", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mode_acc", self.val_mode_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Per-mode accuracy
        for mode_idx in self.valid_modes:
            mode_mask = (ego_true_mode == mode_idx).float() * mask_agent_level
            if mode_mask.sum() > 0:
                mode_pred = ego_probs[mode_mask.bool()]
                mode_target = ego_true_mode[mode_mask.bool()]
                mode_acc_per_mode = compute_mode_accuracy(mode_pred, mode_target, None)

                mode_name = self.mode_names[mode_idx]
                self.val_per_mode_acc[mode_name](mode_acc_per_mode)
                self.log(
                    f"val_per_mode_acc/{mode_name}",
                    self.val_per_mode_acc[mode_name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )

    def on_validation_epoch_end(self) -> None:
        """Log comprehensive validation metrics at epoch end."""
        modal_metrics = self.val_modal_accumulator.compute(use_merged=False)

        self.log("val/modal_accuracy", modal_metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log("val/modal_macro_f1", modal_metrics['macro_f1'], on_epoch=True, prog_bar=True)
        self.log("val/modal_weighted_f1", modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)

        for i, mode_idx in enumerate(self.valid_modes):
            mode_name = self.mode_names[mode_idx]
            self.log(f"val/modal_precision/{mode_name}",
                     modal_metrics['precision_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"val/modal_recall/{mode_name}",
                     modal_metrics['recall_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"val/modal_f1/{mode_name}",
                     modal_metrics['f1_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"val/modal_support/{mode_name}",
                     modal_metrics['support_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

        # print the report
        if self.current_epoch % 5 == 0:
            self.val_modal_accumulator.print_report(title=f"Validation Epoch {self.current_epoch}")
            # print the confusion matrix
            if self.current_epoch % 10 == 0:
                self.val_modal_accumulator.print_confusion_matrix(
                    title=f"Validation Confusion Matrix (Epoch {self.current_epoch})"
                )

        self.val_modal_accumulator.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Execute test step."""
        loss, mode_logits, true_mode_idx = self.model_step(batch)

        # Extract ego agent data
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']
        ego_mask = separate_ego_agent(masks, ego_agent)
        mask_agent_level = ego_mask.any(dim=-1).float()

        ego_mode_logits = separate_ego_agent(mode_logits, ego_agent)
        ego_true_mode = true_mode_idx

        self.test_modal_accumulator.update(
            torch.softmax(ego_mode_logits, dim=-1),
            ego_true_mode,
            mask_agent_level
        )

        self.test_loss(loss)
        self.log("losses/test_mode", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Log comprehensive test metrics and generate reports."""
        modal_metrics = self.test_modal_accumulator.compute(use_merged=False)

        self.log("test/modal_accuracy", modal_metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log("test/modal_macro_f1", modal_metrics['macro_f1'], on_epoch=True, prog_bar=True)
        self.log("test/modal_weighted_f1", modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)
        for i, mode_idx in enumerate(self.valid_modes):
            mode_name = self.mode_names[mode_idx]
            self.log(f"test/modal_precision/{mode_name}",
                     modal_metrics['precision_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"test/modal_recall/{mode_name}",
                     modal_metrics['recall_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"test/modal_f1/{mode_name}",
                     modal_metrics['f1_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

        self.test_modal_accumulator.print_report(title="TEST SET MODE CLASSIFICATION")
        self.test_modal_accumulator.print_confusion_matrix(title="TEST SET CONFUSION MATRIX")

        self.test_modal_accumulator.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with parameter-specific weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 1 or name.endswith(".bias") or any(module in name for module in ['ln', 'norm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.optimizer.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2)
        )

        config = {"optimizer": optimizer}

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "losses/val_mode",
                "interval": "epoch",
                "frequency": 1,
            }

        return config


class TrajectoryPredictionModel(LightningModule):
      """
      Multi-modal trajectory prediction model conditioned on turn modes.
  
      At ALL times (train/val/test), mode conditioning comes from a frozen
      Stage 1 mode prediction network. No teacher forcing.
      Loss: GMM NLL over future steps.
      Network output: (B, A, T, num_modes, D)
      """
  
      def __init__(
              self,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler,
              net: torch.nn.Module,
              mode_net: torch.nn.Module,       # frozen Stage 1 network
              extra_params: EasyDict
      ):
          super().__init__()
          self.save_hyperparameters(ignore=['net', 'mode_net'], logger=False)
  
          # Stage 2 (trainable)
          self.net = net
          self.hist_len  = self.net.hist_len
          self.pred_lens = self.net.pred_lens
          self.num_modes = self.net.num_modes
          self.eparams   = extra_params
  
          # Stage 1 (frozen)
          self.mode_net = mode_net
          for p in self.mode_net.parameters():
              p.requires_grad = False
          self.mode_net.eval()
  
          # Airport configuration
          self.seen_airports   = self.eparams.seen_airports
          self.unseen_airports = self.eparams.unseen_airports
  
          # Mode configuration
          self.valid_modes = VALID_TURN_MODES
          self.mode_names  = TURN_MODES_NAMES
  
          # Loss metrics
          self.train_loss = MeanMetric()
          self.val_loss   = MeanMetric()
          self.test_loss  = MeanMetric()
  
          self.max_pred_len = max(self.pred_lens)
  
          # ADE/FDE metrics
          self.val_ade = nn.ModuleDict({
              f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
              for t in self.pred_lens
          })
          self.test_ade = nn.ModuleDict({
              f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
              for t in self.pred_lens
          })
          self.val_fde = nn.ModuleDict({
              f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
              for t in self.pred_lens
          })
          self.test_fde = nn.ModuleDict({
              f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
              for t in self.pred_lens
          })
  
          self._init_airport_metrics()
          self.ade, self.fde = self._get_metric_functions()
          self.geodesic = Geodesic.WGS84
          self._init_plot_dirs()
  
      def _get_mode_probs(self, X: torch.Tensor, batch: dict) -> torch.Tensor:
          """
          Run frozen Stage 1 to get mode probabilities.
  
          Returns:
              mode_probs: (B, A, num_modes)
          """
          with torch.no_grad():
              mode_logits = self.mode_net(
                  X,
                  context=batch['scene_dict']['context'],
                  adjacency=batch['scene_dict']['adjacency'],
                  mask=None,
                  output_mode_only=True
              )
          return torch.softmax(mode_logits, dim=-1)   # (B, A, num_modes)
  
      def model_step(
              self,
              batch: Dict[str, Any],
              plot: bool = False,
              tag: str = 'temp',
              out_dir: str = 'temp'
      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
          """
          Execute one step of trajectory prediction.
  
          Returns:
              loss:       GMM NLL loss (scalar)
              traj_mu:    (B, A, T, num_modes, D)
              traj_sigma: (B, A, T, num_modes, D)
              fut_rel:    (B, A, T_fut, D)
              mode_probs: (B, A, num_modes)
          """
          Y = batch['scene_dict']['rel_sequences'][..., :4]
  
          # Zero out future timesteps
          X = torch.zeros_like(Y).float()
          X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]
  
          B, A, T, D = Y.shape
          Y = Y[..., G.REL_XYZ[:D]]
          ego_agent = batch['scene_dict']['ego_agent_id']
          masks     = batch['scene_dict']['agent_masks']
  
          # -- Stage 1: get mode probs (frozen) ------------------
          mode_probs = self._get_mode_probs(X, batch)  # (B, A, num_modes)
  
          # -- Stage 2: trajectory prediction --------------------
          # traj_mu / traj_sigma: (B, A, T, num_modes, D)
          traj_mu, traj_sigma = self.net(
              X,
              context=batch['scene_dict']['context'],
              adjacency=batch['scene_dict']['adjacency'],
              mask=None,
              mode_probs=mode_probs
          )
  
          # -- Extract ego agent ---------------------------------
          ego_mu        = separate_ego_agent(traj_mu,    ego_agent)  # (B, 1, T, M, D)
          ego_sigma     = separate_ego_agent(traj_sigma, ego_agent)  # (B, 1, T, M, D)
          ego_fut       = separate_ego_agent(
              Y[:, :, self.hist_len:, :], ego_agent)                 # (B, 1, T_fut, D)
          ego_mask      = separate_ego_agent(masks, ego_agent)       # (B, 1, T)
          ego_mode_probs= separate_ego_agent(
              mode_probs, ego_agent)                                  # (B, 1, num_modes)
  
          # -- GMM NLL loss over future steps --------------------
          mu_future    = ego_mu[:, :, self.hist_len:]         # (B, 1, T_fut, M, D)
          sigma_future = ego_sigma[:, :, self.hist_len:].clamp_min(1e-4)
  
          nll_per_agent = compute_nll(
              traj_mu    = mu_future,       # (B, 1, T_fut, M, D)
              traj_sigma = sigma_future,    # (B, 1, T_fut, M, D)
              mode_probs = ego_mode_probs,  # (B, 1, M)
              Y          = ego_fut,         # (B, 1, T_fut, D)
              mask       = ego_mask         # (B, 1, T)
          )                                                    # (B, 1)
  
          mask_agent_level = ego_mask.any(dim=-1).float()      # (B, 1)
          loss = (nll_per_agent * mask_agent_level).sum() / \
                 mask_agent_level.sum().clamp_min(1)
  
          if plot:
              plot_scene_batch(
                  self.eparams.asset_dir, batch,
                  (mode_probs, traj_mu, traj_sigma),
                  self.hist_len, self.geodesic,
                  tag, out_dir, self.eparams.propagation
              )
  
          return loss, traj_mu, traj_sigma, Y[:, :, self.hist_len:, :], mode_probs
  
      def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
          """Training step — Stage 1 frozen, only Stage 2 trains."""
          # Keep mode_net in eval mode throughout training
          self.mode_net.eval()
  
          loss, _, _, _, _ = self.model_step(batch)
          self.train_loss(loss)
          self.log("losses/train_traj", self.train_loss,
                   on_step=False, on_epoch=True, prog_bar=True)
          return loss
  
      def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
          plot = (
              self.eparams.plot_val
              and self.current_epoch >= self.eparams.plot_after_n_epochs
              and (batch_idx + 1) % self.eparams.plot_every_n == 0
          )
          tag = f"epoch-{self.current_epoch}_batch-{batch_idx}"
  
          loss, traj_mu, traj_sigma, fut_rel, mode_probs = self.model_step(
              batch, plot=plot, tag=tag, out_dir=self.val_out_dir
          )
  
          ego_agent = batch['scene_dict']['ego_agent_id']
          ego_mu    = separate_ego_agent(traj_mu,  ego_agent)  # (B, 1, T, M, D)
          ego_fut   = separate_ego_agent(fut_rel,  ego_agent)  # (B, 1, T_fut, D)
          mask      = separate_ego_agent(
              batch['scene_dict']['agent_masks'], ego_agent)
  
          self.val_loss(loss)
          self.log("losses/val_traj", self.val_loss,
                   on_step=False, on_epoch=True, prog_bar=True)
  
          for t in self.pred_lens:
              mu_t   = ego_mu[:, :, :self.hist_len + t]
              mask_t = mask[:, :, :self.hist_len + t]
              fut_t  = ego_fut[:, :, :t]
              key    = 't=max' if t == self.max_pred_len else f"t={t}"
  
              self.val_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
              self.log(f"val_ade/{key}", self.val_ade[key],
                       on_step=False, on_epoch=True, prog_bar=True)
              self.val_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
              self.log(f"val_fde/{key}", self.val_fde[key],
                       on_step=False, on_epoch=True, prog_bar=True)
  
          self._log_airport_metrics(batch, ego_mu, ego_fut, mask, 'val')
  
      def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
          loss, traj_mu, traj_sigma, fut_rel, mode_probs = self.model_step(
              batch,
              plot=self.eparams.plot_test and (batch_idx + 1) % 10 == 0,
              tag=f"epoch-{self.current_epoch}_batch-{batch_idx}",
              out_dir=self.test_out_dir
          )
  
          ego_agent = batch['scene_dict']['ego_agent_id']
          ego_mu    = separate_ego_agent(traj_mu,  ego_agent)
          ego_fut   = separate_ego_agent(fut_rel,  ego_agent)
          mask      = separate_ego_agent(
              batch['scene_dict']['agent_masks'], ego_agent)
  
          self.test_loss(loss)
          self.log("losses/test_traj", self.test_loss,
                   on_step=False, on_epoch=True, prog_bar=True)
  
          for t in self.pred_lens:
              mu_t   = ego_mu[:, :, :self.hist_len + t]
              mask_t = mask[:, :, :self.hist_len + t]
              fut_t  = ego_fut[:, :, :t]
              key    = 't=max' if t == self.max_pred_len else f"t={t}"
  
              self.test_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
              self.log(f"test_ade/{key}", self.test_ade[key],
                       on_step=False, on_epoch=True)
              self.test_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
              self.log(f"test_fde/{key}", self.test_fde[key],
                       on_step=False, on_epoch=True)
  
          self._log_airport_metrics(batch, ego_mu, ego_fut, mask, 'test')
  
      def _log_airport_metrics(
              self,
              batch: Dict[str, Any],
              ego_mu: torch.Tensor,
              ego_fut: torch.Tensor,
              mask: torch.Tensor,
              stage: str
      ) -> None:
          """Log airport-specific trajectory metrics."""
          airport_ids = batch['scene_dict']['airport_id']
  
          for airport in self.seen_airports:
              airport_idx = np.where(airport_ids == airport)[0]
              if len(airport_idx) == 0:
                  continue
  
              airport_mu = ego_mu[airport_idx]
              airport_fut = ego_fut[airport_idx]
              airport_mask = mask[airport_idx]
  
              for t in self.pred_lens:
                  mu_t = airport_mu[:, :, :self.hist_len + t]
                  fut_t = airport_fut[:, :, :t]
                  mask_t = airport_mask[:, :, :self.hist_len + t]
                  key = f"{airport}_t={t}"
  
                  if stage == 'val':
                      self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                      self.log(f"val_seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True)
                      self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                      self.log(f"val_seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True)
                  else:
                      self.test_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                      self.log(f"test_seen_ade/{key}", self.test_seen_ade[key], on_step=False, on_epoch=True)
                      self.test_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                      self.log(f"test_seen_fde/{key}", self.test_seen_fde[key], on_step=False, on_epoch=True)
  
          if stage == 'test' and self.unseen_airports:
              for airport in self.unseen_airports:
                  airport_idx = np.where(airport_ids == airport)[0]
                  if len(airport_idx) == 0:
                      continue
  
                  airport_mu = ego_mu[airport_idx]
                  airport_fut = ego_fut[airport_idx]
                  airport_mask = mask[airport_idx]
  
                  for t in self.pred_lens:
                      mu_t = airport_mu[:, :, :self.hist_len + t]
                      fut_t = airport_fut[:, :, :t]
                      mask_t = airport_mask[:, :, :self.hist_len + t]
                      key = f"{airport}_t={t}"
  
                      self.test_unseen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                      self.log(f"test_unseen_ade/{key}", self.test_unseen_ade[key], on_step=False, on_epoch=True)
                      self.test_unseen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                      self.log(f"test_unseen_fde/{key}", self.test_unseen_fde[key], on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with parameter-specific weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "temporal_decoder" in name:
                no_decay_params.append(param)
            elif len(param.shape) == 1 or name.endswith(".bias") or any(m in name for m in ['ln', 'norm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.optimizer.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2)
        )

        config = {"optimizer": optimizer}

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "losses/val_traj",
                "interval": "epoch",
                "frequency": 1,
            }

        return config


class CombinedTrajPredSystem(LightningModule):
    """
    Two-stage inference system for multi-modal trajectory prediction.

    Intended for Stage 3 evaluation only - not training.
    Both sub-models are loaded from pre-trained checkpoints.

    Pipeline:
        1. mode_model  mode probability distribution  (B, A, num_modes)
        2. traj_model  one trajectory per turn mode   (B, A, T, num_modes, D)

    Metrics computed:
        - Mode top-1 accuracy
        - minADE, minFDE  (best mode over all turn modes)
        - GMM NLL          (proper mixture likelihood)
        - RMSE             (best-mode RMSE)
    """

    def __init__(
            self,
            mode_model: torch.nn.Module,
            traj_model: torch.nn.Module,
            extra_params: EasyDict
    ):
        super().__init__()
        self.mode_model = mode_model  # AmeliaMode network (the raw net, not LightningModule)
        self.traj_model = traj_model  # AmeliaTrajectory network (the raw net)
        self.eparams = extra_params
        self.num_modes = traj_model.num_modes  # 4 turn modes

        # Test metrics
        self.test_min_ade = MeanMetric()
        self.test_min_fde = MeanMetric()
        self.test_nll = MeanMetric()
        self.test_rmse = MeanMetric()
        self.test_mode_acc = MeanMetric()

    def _encode_rule_based_to_mode_index(self, rule_based):
        return rule_based[..., :4].float().argmax(dim=-1).long()

    def forward(self, batch):
        Y = batch['scene_dict']['rel_sequences'][..., :4]
        X = torch.zeros_like(Y).float()
        X[:, :, :self.traj_model.hist_len] = Y[:, :, :self.traj_model.hist_len]
        context   = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
    
        # Stage 1 (frozen)
        with torch.no_grad():
            mode_logits = self.mode_model(
                X, context=context, adjacency=adjacency,
                mask=None, output_mode_only=True
            )
        mode_probs = torch.softmax(mode_logits, dim=-1)  # (B, A, num_modes)
    
        # Stage 2: all modes in one forward pass
        traj_mu, traj_sigma = self.traj_model(
            X, context=context, adjacency=adjacency,
            mask=None, mode_probs=mode_probs
        )  # (B, A, T, num_modes, D)
    
        return mode_probs, traj_mu, traj_sigma

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        End-to-end test step -???,?????batch??????
        """
        print(batch_idx)
        
        if batch_idx == 0:
            print("\n" + "="*60)
            print(f"[TEST STEP - Batch {batch_idx}]")
            print("="*60)
            
            Y = batch['scene_dict']['rel_sequences'][..., :4]
            print(f"\n[INPUT]")
            print(f"  Y.shape: {Y.shape} (B={Y.shape[0]}, A={Y.shape[1]}, T={Y.shape[2]})")
            print(f"  hist_len: {self.traj_model.hist_len}")
            print(f"  pred_len: {Y.shape[2] - self.traj_model.hist_len}")
            print(f"  num_modes: {self.num_modes}")
        
        # Forward pass
        mode_probs, traj_mu, traj_sigma = self.forward(batch)
        
        if batch_idx == 0:
            print(f"\n[OUTPUT]")
            print(f"  mode_probs.shape: {mode_probs.shape}  # (B, A, num_modes)")
            print(f"  traj_mu.shape: {traj_mu.shape}  # (B, A, T_pred, num_modes, D)")
            print(f"  traj_sigma.shape: {traj_sigma.shape}")
        
        # Ground truth
        Y = batch['scene_dict']['rel_sequences'][..., :4]
        Y = Y[..., G.REL_XYZ[:4]]
        fut_rel = Y[:, :, self.traj_model.hist_len:, :]
        Y_mode = batch['scene_dict']['rule_based_encoding']
        true_modes = self._encode_rule_based_to_mode_index(Y_mode)
        
        if batch_idx == 0:
            print(f"\n[GROUND TRUTH]")
            print(f"  fut_rel.shape: {fut_rel.shape}")
            print(f"  true_modes.shape: {true_modes.shape}")
            # mode
            mode_counts = torch.bincount(true_modes.flatten(), minlength=self.num_modes)
            print(f"  Mode distribution: {mode_counts.detach().cpu().numpy()}")
        
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']
        
        # Extract ego agent
        ego_probs = separate_ego_agent(mode_probs, ego_agent)
        ego_mu = separate_ego_agent(traj_mu, ego_agent)
        ego_sigma = separate_ego_agent(traj_sigma, ego_agent)
        ego_fut = separate_ego_agent(fut_rel, ego_agent)
        ego_mask = separate_ego_agent(masks, ego_agent)
        ego_true_modes = separate_ego_agent(true_modes, ego_agent)
        mask_agent_level = ego_mask.any(dim=-1).float()
        
        if batch_idx == 0:
            print(f"\n[EGO AGENT]")
            print(f"  ego_probs.shape: {ego_probs.shape}")
            print(f"  ego_mu.shape: {ego_mu.shape}")
            print(f"  ego_fut.shape: {ego_fut.shape}")
            print(f"  ego_true_modes[0,0]: {ego_true_modes[0,0].item()}")
            print(f"  ego_probs[0,0]: {ego_probs[0,0].detach().cpu().numpy()}")
        
        # -- Mode accuracy ------------------------------------------------------
        mode_acc = compute_mode_accuracy(ego_probs, ego_true_modes, mask_agent_level)
        self.test_mode_acc(mode_acc)
        self.log("test/mode_acc", self.test_mode_acc, on_step=False, on_epoch=True)
        
        # -- minADE / minFDE ----------------------------------------------------
        min_ade = marginal_ade(ego_mu, ego_fut, mask=ego_mask)
        min_fde = marginal_fde(ego_mu, ego_fut, mask=ego_mask)
        
        self.test_min_ade(min_ade)
        self.test_min_fde(min_fde)
        self.log("test/min_ade", self.test_min_ade, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/min_fde", self.test_min_fde, on_step=False, on_epoch=True, prog_bar=True)
        
        # -- GMM NLL ------------------------------------------------------------
        nll = compute_nll(ego_mu, ego_sigma, ego_probs, ego_fut, mask=ego_mask)
        self.test_nll(nll)
        self.log("test/nll", self.test_nll, on_step=False, on_epoch=True)
        
        # -- RMSE (best mode) ---------------------------------------------------
        rmse = compute_mode_rmse(ego_mu, ego_probs, ego_fut, mask=ego_mask)
        self.test_rmse(rmse)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True)
        
        if batch_idx == 0:
            print(f"\n[METRICS - Batch {batch_idx}]")
            print(f"  mode_acc: {mode_acc.item():.4f}")
            print(f"  min_ade: {min_ade.mean().item():.4f}")
            print(f"  min_fde: {min_fde.mean().item():.4f}")
            print(f"  nll: {nll.mean().item():.4f}")
            print(f"  rmse: {rmse.mean().item():.4f}")
            
            # ?????ego agent?????
            print(f"\n[EGO TRAJECTORY EXAMPLE - First batch]")
            for mode in range(self.num_modes):
                traj_first = ego_mu[0, 0, 0, mode, :2].detach().cpu().numpy()
                prob = ego_probs[0, 0, mode].item()
                print(f"  Mode {mode} (p={prob:.3f}): ({traj_first[0]:.3f}, {traj_first[1]:.3f})")
            
            gt_first = ego_fut[0, 0, 0, :2].detach().cpu().numpy()
            print(f"  GT step 0: ({gt_first[0]:.3f}, {gt_first[1]:.3f})")
            print("="*60 + "\n")
        
        # -- Optional plotting --------------------------------------------------
        if self.eparams.plot_test and (batch_idx + 1) % 10 == 0:
            out_dir = os.path.join(
                self.eparams.plot_dir,
                f"{date.today()}_{self.eparams.tag}",
                'test_combined'
            )
            os.makedirs(out_dir, exist_ok=True)
            plot_scene_batch(
                self.eparams.asset_dir,
                batch,
                (mode_probs, traj_mu, traj_sigma),
                self.traj_model.hist_len,
                self.traj_model.geodesic,
                f"test_batch-{batch_idx}",
                out_dir,
                self.eparams.propagation
            )

    def configure_optimizers(self):
        # Inference-only system — no optimizer needed
        return []