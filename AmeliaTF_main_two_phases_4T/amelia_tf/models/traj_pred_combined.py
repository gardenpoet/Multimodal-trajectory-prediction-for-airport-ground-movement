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

from amelia_tf.utils.off_road_selector import OffRoadSelector

from amelia_tf.models.components.common import LayerNorm
from amelia_tf.utils.utils import plot_scene_batch, separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODE_MAP, MODE_NAMES, VALID_TURN_MODES, TURN_MODES_NAMES
from amelia_tf.utils.metrics import (
    ModalClassificationMetrics, compute_mode_accuracy,
    marginal_ade, marginal_fde, joint_ade, joint_fde,
    compute_mode_rmse, compute_nll, mode_ade, mode_fde
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

        self.net = net
        self.hist_len = self.net.hist_len
        self.num_modes = self.net.num_modes
        self.eparams = extra_params

        self.MODE_MAP = TURN_MODE_MAP
        self.valid_modes = VALID_TURN_MODES
        self.mode_names = TURN_MODES_NAMES

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_modes,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )
        self.test_modal_accumulator_original = ModalClassificationMetrics(
            num_modes=self.num_modes,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )
        self.test_modal_accumulator_balanced = ModalClassificationMetrics(
            num_modes=self.num_modes,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )

        self.mode_weights = None

    def setup(self, stage: str) -> None:
        """Inject class weights from datamodule for imbalanced mode classification."""
        if stage == "fit":
            dm = self.trainer.datamodule
            if not hasattr(dm, "mode_weights"):
                raise RuntimeError("Datamodule missing required 'mode_weights' attribute")
            # dm.mode_weights is explicitly None when use_mode_weights=False (see
            # datamodule.py setup()) - hasattr() alone is True either way, since the
            # attribute always exists; must also check it isn't None before calling
            # .to() on it, or this crashes with the mode-weighting grid disabled.
            if dm.mode_weights is not None:
                self.mode_weights = dm.mode_weights.to(self.device)
                print(f"Mode weights injected: {self.mode_weights}")
            else:
                self.mode_weights = None
                print("use_mode_weights=False: mode classification loss is unweighted.")

    def _encode_rule_based_to_mode_index(self, rule_based: torch.Tensor) -> torch.Tensor:
        """Convert rule-based encoding to turn-only mode index (0-3)."""
        turn_idx = rule_based[..., :4].float().argmax(dim=-1)
        return turn_idx.long()

    def model_step(
            self,
            batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute one step of mode prediction.

        Passes turn_feasibility from the batch to the network if available.
        The network uses it as a soft conditioning signal (and optionally as a
        hard logit mask) depending on its own config flags.
        """
        Y      = batch['scene_dict']['rel_sequences']
        Y_mode = batch['scene_dict'].get('rule_based_encoding')

        # turn_feasibility: (B, A, 4) float tensor or None
        feasibility = batch['scene_dict'].get('turn_feasibility', None)
        if feasibility is not None:
            feasibility = feasibility.to(Y.device)

        X = torch.zeros_like(Y[..., :4]).float()
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len, :4]

        mode_logits = self.net(
            X,
            context=batch['scene_dict']['context'],
            adjacency=batch['scene_dict']['adjacency'],
            mask=None,
            feasibility=feasibility,
            output_mode_only=True
        )

        if Y_mode is not None:
            true_mode_idx     = self._encode_rule_based_to_mode_index(Y_mode)
            ego_agent         = batch['scene_dict']['ego_agent_id']
            true_mode_idx_ego = separate_ego_agent(true_mode_idx, ego_agent)
            ego_mode_logits   = separate_ego_agent(mode_logits, ego_agent)

            # Label smoothing is incompatible with hard feasibility masking:
            # smoothing distributes probability to masked (-inf) classes,
            # causing log(~0) * smoothing_weight -> numerical explosion.
            # Disable smoothing when the network applies a hard mask.
            use_smoothing = (
                0.0 if getattr(self.net, 'apply_hard_mask', False)
                else 0.1
            )
            loss = F.cross_entropy(
                ego_mode_logits.view(-1, ego_mode_logits.shape[-1]),
                true_mode_idx_ego.view(-1),
                weight=self.mode_weights,
                label_smoothing=use_smoothing,
                reduction='mean'
            )
        else:
            loss              = torch.tensor(0.0, device=self.device)
            true_mode_idx_ego = None

        return loss, mode_logits, true_mode_idx_ego

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("losses/train_mode", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss, mode_logits, true_mode_idx = self.model_step(batch)

        ego_agent        = batch['scene_dict']['ego_agent_id']
        masks            = batch['scene_dict']['agent_masks']
        ego_mask         = separate_ego_agent(masks, ego_agent)
        mask_agent_level = ego_mask.any(dim=-1).float()

        ego_mode_logits = separate_ego_agent(mode_logits, ego_agent)
        ego_probs       = torch.softmax(ego_mode_logits, dim=-1)

        self.val_modal_accumulator.update(ego_probs, true_mode_idx, mask_agent_level)

        self.val_loss(loss)
        self.log("losses/val_mode", self.val_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        modal_metrics = self.val_modal_accumulator.compute(use_merged=False)

        self.log("val/modal_accuracy",    modal_metrics['accuracy'],    on_epoch=True, prog_bar=True)
        self.log("val/modal_macro_f1",    modal_metrics['macro_f1'],    on_epoch=True, prog_bar=True)
        self.log("val/modal_weighted_f1", modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)

        for i, mode_idx in enumerate(self.valid_modes):
            mode_name = self.mode_names[mode_idx]
            self.log(f"val/modal_precision/{mode_name}",
                     modal_metrics['precision_per_mode'][mode_idx], on_epoch=True)
            self.log(f"val/modal_recall/{mode_name}",
                     modal_metrics['recall_per_mode'][mode_idx], on_epoch=True)
            self.log(f"val/modal_f1/{mode_name}",
                     modal_metrics['f1_per_mode'][mode_idx], on_epoch=True)
            self.log(f"val/modal_support/{mode_name}",
                     modal_metrics['support_per_mode'][mode_idx], on_epoch=True)

        if self.current_epoch % 5 == 0:
            self.val_modal_accumulator.print_report(
                title=f"Validation Epoch {self.current_epoch}")
            if self.current_epoch % 10 == 0:
                self.val_modal_accumulator.print_confusion_matrix(
                    title=f"Validation Confusion Matrix (Epoch {self.current_epoch})")

        self.val_modal_accumulator.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        loss, mode_logits, true_mode_idx = self.model_step(batch)

        ego_agent        = batch['scene_dict']['ego_agent_id']
        masks            = batch['scene_dict']['agent_masks']
        ego_mask         = separate_ego_agent(masks, ego_agent)
        mask_agent_level = ego_mask.any(dim=-1).float()

        ego_mode_logits = separate_ego_agent(mode_logits, ego_agent)
        ego_probs       = torch.softmax(ego_mode_logits, dim=-1)

        if dataloader_idx == 0:
            self.test_modal_accumulator_original.update(
                ego_probs, true_mode_idx, mask_agent_level)
        else:
            self.test_modal_accumulator_balanced.update(
                ego_probs, true_mode_idx, mask_agent_level)

    def on_test_epoch_end(self) -> None:
        self._log_test_epoch_metrics(
            self.test_modal_accumulator_original,
            "test/original", "ORIGINAL TEST SET")
        self._log_test_epoch_metrics(
            self.test_modal_accumulator_balanced,
            "test/balanced", "BALANCED TEST SET")
        self.test_modal_accumulator_original.reset()
        self.test_modal_accumulator_balanced.reset()

    def _log_test_epoch_metrics(self, accumulator, prefix: str, title: str) -> None:
        modal_metrics = accumulator.compute(use_merged=False)

        self.log(f"{prefix}/modal_accuracy",    modal_metrics['accuracy'],    on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/modal_macro_f1",    modal_metrics['macro_f1'],    on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/modal_weighted_f1", modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)

        for mode_idx in self.valid_modes:
            mode_name = self.mode_names[mode_idx]
            self.log(f"{prefix}/modal_precision/{mode_name}",
                     modal_metrics['precision_per_mode'][mode_idx], on_epoch=True)
            self.log(f"{prefix}/modal_recall/{mode_name}",
                     modal_metrics['recall_per_mode'][mode_idx], on_epoch=True)
            self.log(f"{prefix}/modal_f1/{mode_name}",
                     modal_metrics['f1_per_mode'][mode_idx], on_epoch=True)
            self.log(f"{prefix}/modal_support/{mode_name}",
                     modal_metrics['support_per_mode'][mode_idx], on_epoch=True)

        accumulator.print_report(title=f"{title} MODE CLASSIFICATION")
        accumulator.print_confusion_matrix(title=f"{title} CONFUSION MATRIX")

    def configure_optimizers(self) -> Dict[str, Any]:
        decay_params    = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (len(param.shape) == 1 or name.endswith(".bias") or
                    any(m in name for m in ['ln', 'norm'])):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.hparams.optimizer.weight_decay},
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
                "monitor":   "losses/val_mode",
                "interval":  "epoch",
                "frequency": 1,
            }

        return config


class TrajectoryPredictionModel(LightningModule):
    """
    Single-mode trajectory prediction model conditioned on a turn mode.

    At training time uses teacher forcing (ground truth mode index to one-hot).
    At test time the mode index comes from the mode prediction model.
    Network output is (B, A, T, D) one trajectory per sample.
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

        self.net       = net
        self.hist_len  = self.net.hist_len
        self.pred_lens = self.net.pred_lens
        self.num_modes = self.net.num_modes
        self.eparams   = extra_params

        self.seen_airports   = self.eparams.seen_airports
        self.unseen_airports = self.eparams.unseen_airports

        self.valid_modes = VALID_TURN_MODES
        self.mode_names  = TURN_MODES_NAMES

        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()

        self.max_pred_len = max(self.pred_lens)
        
        self.lambda_score = getattr(self.eparams, 'lambda_score', 10)
        
        self.mode_weights = None
        


        self.val_ade = nn.ModuleDict({
            f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for t in self.pred_lens})
        self.val_fde = nn.ModuleDict({
            f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for t in self.pred_lens})

        self.test_ade_original = nn.ModuleDict({
            f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for t in self.pred_lens})
        self.test_fde_original = nn.ModuleDict({
            f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for t in self.pred_lens})

        self.test_ade_balanced = nn.ModuleDict({
            f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for t in self.pred_lens})
        self.test_fde_balanced = nn.ModuleDict({
            f"t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for t in self.pred_lens})

        # ---- PER-MODE VALIDATION METRICS (NEW) ----
        self.val_mode_ade = nn.ModuleDict({
            f"{self.mode_names[m]}_t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for m, t in itertools.product(self.valid_modes, self.pred_lens)
        })
        self.val_mode_fde = nn.ModuleDict({
            f"{self.mode_names[m]}_t={'max' if t == self.max_pred_len else t}": MeanMetric()
            for m, t in itertools.product(self.valid_modes, self.pred_lens)
        })

        self._init_airport_metrics()
        self._init_per_mode_metrics()

        self.ade, self.fde = self._get_metric_functions()
        self.geodesic      = Geodesic.WGS84

        self._init_plot_dirs()

    def _init_per_mode_metrics(self) -> None:
        self.test_mode_ade_original = nn.ModuleDict({
            f"{self.mode_names[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(self.valid_modes, self.pred_lens)})
        self.test_mode_fde_original = nn.ModuleDict({
            f"{self.mode_names[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(self.valid_modes, self.pred_lens)})

    def _init_airport_metrics(self) -> None:
        self.val_seen_ade = nn.ModuleDict({
            f"{a}_t={t}": MeanMetric()
            for t, a in itertools.product(self.pred_lens, self.seen_airports)})
        self.val_seen_fde = nn.ModuleDict({
            f"{a}_t={t}": MeanMetric()
            for t, a in itertools.product(self.pred_lens, self.seen_airports)})
        self.test_seen_ade_original = nn.ModuleDict({
            f"{a}_t={t}": MeanMetric()
            for t, a in itertools.product(self.pred_lens, self.seen_airports)})
        self.test_seen_fde_original = nn.ModuleDict({
            f"{a}_t={t}": MeanMetric()
            for t, a in itertools.product(self.pred_lens, self.seen_airports)})
        self.test_seen_ade_balanced = nn.ModuleDict({
            f"{a}_t={t}": MeanMetric()
            for t, a in itertools.product(self.pred_lens, self.seen_airports)})
        self.test_seen_fde_balanced = nn.ModuleDict({
            f"{a}_t={t}": MeanMetric()
            for t, a in itertools.product(self.pred_lens, self.seen_airports)})

        if self.unseen_airports:
            self.test_unseen_ade_original = nn.ModuleDict({
                f"{a}_t={t}": MeanMetric()
                for t, a in itertools.product(self.pred_lens, self.unseen_airports)})
            self.test_unseen_fde_original = nn.ModuleDict({
                f"{a}_t={t}": MeanMetric()
                for t, a in itertools.product(self.pred_lens, self.unseen_airports)})
            self.test_unseen_ade_balanced = nn.ModuleDict({
                f"{a}_t={t}": MeanMetric()
                for t, a in itertools.product(self.pred_lens, self.unseen_airports)})
            self.test_unseen_fde_balanced = nn.ModuleDict({
                f"{a}_t={t}": MeanMetric()
                for t, a in itertools.product(self.pred_lens, self.unseen_airports)})

    def _get_metric_functions(self) -> Tuple[callable, callable]:
        if self.eparams.propagation == 'marginal':
            return marginal_ade, marginal_fde
        return joint_ade, joint_fde

    def _init_plot_dirs(self) -> None:
        os.makedirs(self.eparams.plot_dir, exist_ok=True)
        out_dir          = os.path.join(self.eparams.plot_dir,
                                        f"{date.today()}_{self.eparams.tag}")
        self.val_out_dir  = os.path.join(out_dir, 'val_traj')
        self.test_out_dir = os.path.join(out_dir, 'test_traj')
        os.makedirs(self.val_out_dir,  exist_ok=True)
        os.makedirs(self.test_out_dir, exist_ok=True)

    def _encode_rule_based_to_mode_index(self, rule_based: torch.Tensor) -> torch.Tensor:
        return rule_based[..., :4].float().argmax(dim=-1).long()

        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            dm = self.trainer.datamodule
            # Same None-vs-missing distinction as ModePredictionModel.setup(): the
            # attribute always exists, but is None when use_mode_weights=False.
            if getattr(dm, "mode_weights", None) is not None:
                self.mode_weights = dm.mode_weights.to(self.device)
                print(f"[traj] Mode weights injected: {self.mode_weights}")
            else:
                print("[traj] mode_weights disabled/unavailable; score loss unweighted.")

    def model_step(
            self,
            batch: Dict[str, Any],
            mode_predictions: Optional[torch.Tensor] = None,
            use_teacher_forcing: bool = True,
            plot: bool = False,
            tag: str = 'temp',
            out_dir: str = 'temp'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Execute one step of trajectory prediction."""
        
        torch.autograd.set_detect_anomaly(True)
        
        Y      = batch['scene_dict']['rel_sequences'][..., :4]
        Y_mode = batch['scene_dict'].get('rule_based_encoding')

        X = torch.zeros_like(Y).float()
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]
        B, A, T, D = Y.shape
        Y       = Y[..., G.REL_XYZ[:D]]
        context   = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks     = batch['scene_dict']['agent_masks']

        if use_teacher_forcing and Y_mode is not None:
            true_modes   = self._encode_rule_based_to_mode_index(Y_mode)
            modes_to_use = true_modes
        elif mode_predictions is not None:
            true_modes   = None
            modes_to_use = mode_predictions.argmax(dim=-1)
        else:
            raise ValueError("Must provide either ground-truth modes or mode_predictions")

        assert modes_to_use.min() >= 0 and modes_to_use.max() < self.num_modes, \
            f"modes_to_use range {modes_to_use.min()}-{modes_to_use.max()} vs num_modes {self.num_modes}"
        
        ego_true = separate_ego_agent(modes_to_use, ego_agent).squeeze(1)
        sample_weights = self.mode_weights[ego_true] if self.mode_weights is not None else None

        mode_probs_input = F.one_hot(
            modes_to_use, num_classes=self.num_modes).float()

        traj_mu, traj_sigma, traj_score = self.net(
        X, context=context, adjacency=adjacency,
        mask=None, mode_probs=mode_probs_input)
      

        ego_mu    = separate_ego_agent(traj_mu,    ego_agent)
        ego_sigma = separate_ego_agent(traj_sigma, ego_agent)
        ego_fut   = separate_ego_agent(Y[:, :, self.hist_len:, :], ego_agent)
        ego_mask  = separate_ego_agent(masks, ego_agent)
        
        K = traj_mu.shape[3]
        
        mu_future    = ego_mu[:, :, self.hist_len:]
        sigma_future = ego_sigma[:, :, self.hist_len:].clamp_min(1e-4)
        score_match = None

        if K == 1:
            # ------------------------------------------------------------------  # CHANGED
            # Legacy path: identical loss to original (single hypothesis).        # CHANGED
            # Squeeze K so shapes match the original (B, A, T_pred, D).          # CHANGED
            # ------------------------------------------------------------------  # CHANGED
            mu_future_sq    = mu_future.squeeze(3)     # (B_ego, 1, T_pred, D)   # CHANGED
            sigma_future_sq = sigma_future.squeeze(3)                             # CHANGED
 
            loss = F.gaussian_nll_loss(
                mu_future_sq, ego_fut, sigma_future_sq ** 2,
                reduction='none')
            mask_future = ego_mask[:, :, self.hist_len:].unsqueeze(-1)
            # Per-sample loss before averaging
            per_sample_loss = (loss * mask_future).sum(dim=(1, 2, 3)) / \
                              mask_future.sum(dim=(1, 2, 3)).clamp_min(1)  # (B,)

            # Apply per-sample weights if available
            if sample_weights is not None:
                loss = (per_sample_loss * sample_weights).mean()
            else:
                loss = per_sample_loss.mean()
            #loss = (loss * mask_future).sum() / mask_future.sum().clamp_min(1)
            
 
        else:
            # ------------------------------------------------------------------  # CHANGED
            # Multi-hypothesis path: Winner-Takes-All (WTA) loss.                # CHANGED
            # For each sample we compute the L2 distance of every hypothesis     # CHANGED
            # to the ground-truth future, pick the closest one (no gradient      # CHANGED
            # through argmin), and compute NLL only for that winner.             # CHANGED
            # ------------------------------------------------------------------  # CHANGED
            with torch.no_grad():
                fut_exp = ego_fut.unsqueeze(3)
                dist_k = (mu_future - fut_exp).pow(2).sum(-1).mean(-2)  # (B,1,K)
                best_k = dist_k.argmin(dim=-1)                          # (B,1) oracle winner
            k_exp  = best_k[:, :, None, None, None].expand(
                best_k.shape[0], 1, mu_future.shape[2], 1, mu_future.shape[4])
            best_mu    = mu_future.gather(3, k_exp).squeeze(3)
            best_sigma = sigma_future.gather(3, k_exp).squeeze(3)
            
            assert (best_sigma > 0).all(), f"best_sigma has non-positive: min={best_sigma.min()}"
            
            mask_future = ego_mask[:, :, self.hist_len:].unsqueeze(-1)   # (B,1,Tp,1)
            # WTA gaussian NLL, kept PER-SAMPLE (no global mean yet)
            wta_elem = F.gaussian_nll_loss(
                best_mu, ego_fut, best_sigma ** 2, reduction='none')     # (B,1,Tp,D)
            wta_per = (wta_elem * mask_future).sum(dim=(1, 2, 3)) / \
                      mask_future.sum(dim=(1, 2, 3)).clamp_min(1)         # (B,)
            wta_loss = wta_per.mean()                                    # scalar (for logging / K==1 parity)

            # ------------------------------------------------------------------
            # Mode-differentiated loss (only wta_loss + selected_gnll):
            #   - Turn modes (Left/Right): wta (anti-collapse anchor) + heavy
            #     selected_gnll  -> the model must LEARN TO SELECT among diverse
            #     candidates (this is where selection matters).
            #   - Other modes (Straight/Hold): mostly wta -> candidates just fit
            #     the GT and naturally contract (quality; selection irrelevant).
            # No score_ce, no diversity regulariser (the latter was unbounded and
            # blew up). Everything here is bounded gaussian NLL -> stable.
            # ------------------------------------------------------------------
            if traj_score is None:
                if sample_weights is not None:
                    loss = (wta_per * sample_weights).mean()
                else:
                    loss = wta_loss
            else:
                warmup      = getattr(self.eparams, 'score_warmup_epochs', 5)
                # per-mode mixing weights
                turn_wta   = getattr(self.eparams, 'turn_wta_weight',   0.0)
                turn_gnll  = getattr(self.eparams, 'turn_gnll_weight',  1.0)
                other_wta  = getattr(self.eparams, 'other_wta_weight',  1.0)
                other_gnll = getattr(self.eparams, 'other_gnll_weight', 0.0)
                # score classification (direct selection supervision):
                # turn modes need it, others don't (selecting is meaningless there).
                # CE magnitude (~1) is small vs gaussian NLL (~17), so scale it up.
                turn_ce    = getattr(self.eparams, 'turn_ce_weight',  1.0)
                other_ce   = getattr(self.eparams, 'other_ce_weight', 0.0)
                alpha_ce   = getattr(self.eparams, 'alpha_score_ce',  10.0)

                ego_agent_sc = batch['scene_dict']['ego_agent_id']
                ego_score = separate_ego_agent(traj_score, ego_agent_sc)   # (B,1,T,K)
                s = ego_score[:, :, self.hist_len:].mean(2).squeeze(1)     # (B,K)
                winner = best_k.squeeze(1)

                # ---- straight-through hard selection (forward: argmax candidate; backward: differentiable) ----
                w_soft = F.softmax(s, dim=-1)                              # (B,K)
                sel_k = s.argmax(dim=-1)                                   # (B,)
                w_hard = F.one_hot(sel_k, num_classes=s.shape[-1]).float() # (B,K)
                w_st = w_hard + (w_soft - w_soft.detach())                # (B,K) straight-through
                w_exp = w_st[:, None, None, :, None]                      # (B,1,1,K,1)
                picked_mu    = (mu_future    * w_exp).sum(dim=3)           # (B,1,Tp,D)
                picked_sigma = (sigma_future * w_exp).sum(dim=3)           # (B,1,Tp,D)

                
                gnll_elem = F.gaussian_nll_loss(
                    picked_mu, ego_fut, picked_sigma ** 2, reduction='none')  # (B,1,Tp,D)
                gnll_per = (gnll_elem * mask_future).sum(dim=(1, 2, 3)) / \
                           mask_future.sum(dim=(1, 2, 3)).clamp_min(1)     # (B,)

                # ---- per-sample mode -> turn vs other mixing ----
                Y_mode = batch['scene_dict'].get('rule_based_encoding')
                if Y_mode is not None:
                    tmi = self._encode_rule_based_to_mode_index(Y_mode)
                    tmi_ego = separate_ego_agent(tmi, ego_agent_sc).view(-1)  # (B,)
                    turn_mask = (tmi_ego == 0) | (tmi_ego == 1)              # Left/Right
                else:
                    turn_mask = torch.zeros_like(winner, dtype=torch.bool)

                w_wta  = torch.where(turn_mask,
                                     wta_per.new_full((), turn_wta),
                                     wta_per.new_full((), other_wta))       # (B,) via broadcast
                w_gnll = torch.where(turn_mask,
                                     gnll_per.new_full((), turn_gnll),
                                     gnll_per.new_full((), other_gnll))
                w_ce   = torch.where(turn_mask,
                                     wta_per.new_full((), turn_ce),
                                     wta_per.new_full((), other_ce))
                                     
                # ---- score classification (direct selection supervision) ----
                # cross-entropy pushing the score toward the oracle winner.
                ce_per = F.cross_entropy(s, winner, reduction='none')      # (B,)
                score_ce = (ce_per * w_ce).mean()

                # during warmup: only WTA everywhere (let candidates diversify)
                if self.current_epoch < warmup:
                    loss = wta_per.mean()
                else:
                    loss = (w_wta * wta_per + w_gnll * gnll_per).mean() + alpha_ce * score_ce

                selected_gnll = gnll_per.mean()
                score_match = (s.argmax(-1) == winner).float().mean()

                # ---- diagnostics ----
                with torch.no_grad():
                    end = mu_future[:, :, -1, :, :2]
                    cand_spread = end.std(dim=2).mean()
                    turn_spread  = end.std(dim=2).mean(-1).squeeze(1)[turn_mask].mean() \
                                   if turn_mask.any() else torch.tensor(0.0, device=s.device)
                    other_spread = end.std(dim=2).mean(-1).squeeze(1)[~turn_mask].mean() \
                                   if (~turn_mask).any() else torch.tensor(0.0, device=s.device)
                self.log("train/wta_loss",       wta_loss,      on_step=False, on_epoch=True)
                self.log("train/selected_gnll",  selected_gnll, on_step=False, on_epoch=True)
                self.log("train/cand_spread",    cand_spread,   on_step=False, on_epoch=True, prog_bar=True)
                self.log("train/turn_spread",    turn_spread,   on_step=False, on_epoch=True, prog_bar=True)
                self.log("train/other_spread",   other_spread,  on_step=False, on_epoch=True)
                self.log("train/score_match",    score_match,   on_step=False, on_epoch=True, prog_bar=True)
            

        if plot:
            # traj_mu passed to plot_scene_batch must be (B, A, T, D).
            # Squeeze K (or take k=0) so the plotting code stays unchanged.     # CHANGED
            traj_mu_for_plot = traj_mu[:, :, :, 0, :]  # take first hypothesis  # CHANGED
            plot_scene_batch(
                self.eparams.asset_dir, batch,
                (None, traj_mu_for_plot, traj_sigma[:, :, :, 0, :]),            # CHANGED
                self.hist_len, self.geodesic, tag, out_dir,
                self.eparams.propagation)

        return loss, traj_mu, traj_sigma, Y[:, :, self.hist_len:, :], true_modes, score_match, traj_score

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _, _, _,score_match,_ = self.model_step(batch, use_teacher_forcing=True)
        if score_match is not None:
            self.log("train/score_match", score_match,
            on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss(loss)
        self.log("losses/train_traj", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        plot = (self.eparams.plot_val
                and self.current_epoch >= self.eparams.plot_after_n_epochs
                and (batch_idx + 1) % self.eparams.plot_every_n == 0)
        tag  = f"epoch-{self.current_epoch}_batch-{batch_idx}"

        loss, traj_mu, traj_sigma, fut_rel, true_modes, score_match, traj_score = self.model_step(
            batch, use_teacher_forcing=True, plot=plot, tag=tag,
            out_dir=self.val_out_dir)
            
        
        if score_match is not None:
            self.log("val/score_match", score_match,
            on_step=False, on_epoch=True, prog_bar=True)

        ego_agent = batch['scene_dict']['ego_agent_id']
        ego_mu  = separate_ego_agent(traj_mu,  ego_agent)#.unsqueeze(-2)
        ego_fut = separate_ego_agent(fut_rel,  ego_agent)
        mask    = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)

        self.val_loss(loss)
        self.log("losses/val_traj", self.val_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
                 
        if traj_score is not None:
            ego_score = separate_ego_agent(traj_score, ego_agent)    # (B,1,T,K)
            s = ego_score[:, :, self.hist_len:].mean(2).squeeze(1)   # (B,K)
            sel_k = s.argmax(-1)                                      # (B,)
    
            # ego_mu is (B,1,T,K,D); gather the score-selected candidate on K (dim=3)
            Bv, _, Tv, Kv, Dv = ego_mu.shape
            k_exp = sel_k[:, None, None, None, None].expand(Bv, 1, Tv, 1, Dv)
            sel_mu = ego_mu.gather(3, k_exp).squeeze(3)              # (B,1,T,D)
    
            # align to future part before comparing with ego_fut (B,1,Tp,D)
            sel_mu_fut = sel_mu[:, :, self.hist_len:]                # (B,1,Tp,D)
            mask_fut   = mask[:, :, self.hist_len:]                  # (B,1,Tp)
            sel_ade = ((sel_mu_fut[..., :2] - ego_fut[..., :2]) ** 2
                       ).sum(-1).sqrt()                              # (B,1,Tp)
            sel_ade = (sel_ade * mask_fut).sum() / mask_fut.sum().clamp_min(1)
            self.log("val/selected_ade/t=max", sel_ade,
                     on_step=False, on_epoch=True, prog_bar=True)
                     
        # ---- per-mode selection accuracy (score-selected k vs oracle min-ADE k) ----
        if traj_score is not None and true_modes is not None:
            with torch.no_grad():
                ego_score = separate_ego_agent(traj_score, ego_agent)    # (B,1,T,K)
                s = ego_score[:, :, self.hist_len:].mean(2).squeeze(1)   # (B,K)
                sel_k = s.argmax(-1)                                      # (B,)
    
                mu_fut = ego_mu[:, :, self.hist_len:]                     # (B,1,Tp,K,D)
                gt = ego_fut[:, :, :, :2]                                 # (B,1,Tp,2)
                m_fut = mask[:, :, self.hist_len:].float()               # (B,1,Tp)
                d = ((mu_fut[..., :2] - gt.unsqueeze(3)) ** 2).sum(-1).sqrt()  # (B,1,Tp,K)
                denom = m_fut.sum(dim=2).clamp_min(1)                     # (B,1)
                ade_k = (d * m_fut.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)  # (B,1,K)
                oracle_k = ade_k.squeeze(1).argmin(-1)                    # (B,)
    
                correct = (sel_k == oracle_k)                            # (B,)
                tm = separate_ego_agent(true_modes, ego_agent).reshape(-1)  # (B,)
    
                # per-mode only (overall is already logged as val/score_match)
                for m, name in enumerate(self.mode_names):
                    mmask = (tm == m)
                    if mmask.any():
                        self.log(f"val/sel_acc/{name}",
                                 correct[mmask].float().mean(),
                                 on_step=False, on_epoch=True)

        for t in self.pred_lens:
            mu_t   = ego_mu[:, :, :self.hist_len + t]
            mask_t = mask[:, :, :self.hist_len + t]
            fut_t  = ego_fut[:, :, :t]
            key    = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"val/ade/{key}", self.val_ade[key], on_step=False, on_epoch=True, prog_bar=True)
            self.val_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"val/fde/{key}", self.val_fde[key], on_step=False, on_epoch=True, prog_bar=True)

        # ---- PER-MODE ADE/FDE metrics (NEW) ----
        if true_modes is not None:
            ego_true_mode = separate_ego_agent(true_modes, ego_agent).squeeze(1).squeeze()  # (B,)

            for mode_idx in self.valid_modes:
                mode_name = self.mode_names[mode_idx]
                sample_mask = (ego_true_mode == mode_idx)

                if sample_mask.sum() == 0:
                    continue

                mu_mode = ego_mu[sample_mask]
                fut_mode = ego_fut[sample_mask]
                mask_mode = mask[sample_mask]

                for t in self.pred_lens:
                    mu_t = mu_mode[:, :, :self.hist_len + t]
                    fut_t = fut_mode[:, :, :t]
                    mask_t = mask_mode[:, :, :self.hist_len + t]

                    t_key = 't=max' if t == self.max_pred_len else f"t={t}"
                    metric_key = f"{mode_name}_{t_key}"

                    # Compute per-sample ADE (returns tensor of shape (B_mode,))
                    mode_ade_per_sample = self.ade(mu_t, fut_t, mask=mask_t)

                    # Take mean to get scalar
                    mode_ade_scalar = mode_ade_per_sample.mean()

                    # Update accumulator with scalar
                    if metric_key in self.val_mode_ade:
                        self.val_mode_ade[metric_key](mode_ade_scalar)
                        self.log(f"val/mode_ade/{metric_key}",
                                 self.val_mode_ade[metric_key],
                                 on_step=False, on_epoch=True, prog_bar=True)

                    # Similarly for FDE
                    mode_fde_per_sample = self.fde(mu_t, fut_t, mask=mask_t)
                    mode_fde_scalar = mode_fde_per_sample.mean()

                    if metric_key in self.val_mode_fde:
                        self.val_mode_fde[metric_key](mode_fde_scalar)
                        self.log(f"val/mode_fde/{metric_key}",
                                 self.val_mode_fde[metric_key],
                                 on_step=False, on_epoch=True, prog_bar=True)

        self._log_airport_metrics(batch, ego_mu, ego_fut, mask, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=0,
                  mode_predictions=None) -> None:
        plot      = self.eparams.plot_test and (batch_idx + 1) % 10 == 0
        tag       = f"epoch-{self.current_epoch}_batch-{batch_idx}"
        use_teach = mode_predictions is None

        loss, traj_mu, traj_sigma, fut_rel, true_modes, _,_ = self.model_step(
            batch, mode_predictions=mode_predictions,
            use_teacher_forcing=use_teach,
            plot=plot, tag=tag, out_dir=self.test_out_dir)

        ego_agent = batch['scene_dict']['ego_agent_id']
        ego_mu  = separate_ego_agent(traj_mu,  ego_agent)#.unsqueeze(-2)
        ego_fut = separate_ego_agent(fut_rel,  ego_agent)
        mask    = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)

        if dataloader_idx == 0:
            prefix        = "original"
            ade_dict      = self.test_ade_original
            fde_dict      = self.test_fde_original
            seen_ade_dict = self.test_seen_ade_original
            seen_fde_dict = self.test_seen_fde_original
            unseen_ade_dict = getattr(self, 'test_unseen_ade_original', None)
            unseen_fde_dict = getattr(self, 'test_unseen_fde_original', None)
            if true_modes is not None:
                ego_true_mode = separate_ego_agent(true_modes, ego_agent).squeeze(1).squeeze()
                for mode_idx in self.valid_modes:
                    mode_name   = self.mode_names[mode_idx]
                    sample_mask = (ego_true_mode == mode_idx)
                    if sample_mask.sum() == 0:
                        continue
                    for t in self.pred_lens:
                        mu_t   = ego_mu[sample_mask][:, :, :self.hist_len + t]
                        fut_t  = ego_fut[sample_mask][:, :, :t]
                        mask_t = mask[sample_mask][:, :, :self.hist_len + t]
                        key    = f"{mode_name}_t={t}"
                        self.test_mode_ade_original[key](self.ade(mu_t, fut_t, mask=mask_t))
                        self.log(f"test/original/mode_ade/{mode_name}/t={t}",
                                 self.test_mode_ade_original[key], on_step=False, on_epoch=True)
                        self.test_mode_fde_original[key](self.fde(mu_t, fut_t, mask=mask_t))
                        self.log(f"test/original/mode_fde/{mode_name}/t={t}",
                                 self.test_mode_fde_original[key], on_step=False, on_epoch=True)
        else:
            prefix        = "balanced"
            ade_dict      = self.test_ade_balanced
            fde_dict      = self.test_fde_balanced
            seen_ade_dict = self.test_seen_ade_balanced
            seen_fde_dict = self.test_seen_fde_balanced
            unseen_ade_dict = getattr(self, 'test_unseen_ade_balanced', None)
            unseen_fde_dict = getattr(self, 'test_unseen_fde_balanced', None)

        for t in self.pred_lens:
            mu_t   = ego_mu[:, :, :self.hist_len + t]
            mask_t = mask[:, :, :self.hist_len + t]
            fut_t  = ego_fut[:, :, :t]
            key    = 't=max' if t == self.max_pred_len else f"t={t}"
            ade_dict[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"test/{prefix}/ade/{key}", ade_dict[key], on_step=False, on_epoch=True)
            fde_dict[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"test/{prefix}/fde/{key}", fde_dict[key], on_step=False, on_epoch=True)

        self._log_airport_metrics_test(
            batch, ego_mu, ego_fut, mask,
            seen_ade_dict, seen_fde_dict,
            unseen_ade_dict, unseen_fde_dict, prefix)

    def _log_airport_metrics(self, batch, ego_mu, ego_fut, mask, stage) -> None:
        airport_ids = batch['scene_dict']['airport_id']
        for airport in self.seen_airports:
            idx = np.where(airport_ids == airport)[0]
            if len(idx) == 0:
                continue
            for t in self.pred_lens:
                mu_t   = ego_mu[idx][:, :, :self.hist_len + t]
                fut_t  = ego_fut[idx][:, :, :t]
                mask_t = mask[idx][:, :, :self.hist_len + t]
                key    = f"{airport}_t={t}"
                if stage == 'val':
                    self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(f"val/seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True)
                    self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(f"val/seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True)

    def _log_airport_metrics_test(
            self, batch, ego_mu, ego_fut, mask,
            seen_ade_dict, seen_fde_dict,
            unseen_ade_dict, unseen_fde_dict, prefix) -> None:
        airport_ids = batch['scene_dict']['airport_id']
        for airport in self.seen_airports:
            idx = np.where(airport_ids == airport)[0]
            if len(idx) == 0:
                continue
            for t in self.pred_lens:
                mu_t   = ego_mu[idx][:, :, :self.hist_len + t]
                fut_t  = ego_fut[idx][:, :, :t]
                mask_t = mask[idx][:, :, :self.hist_len + t]
                key    = f"{airport}_t={t}"
                seen_ade_dict[key](self.ade(mu_t, fut_t, mask=mask_t))
                self.log(f"test/{prefix}/seen_ade/{key}", seen_ade_dict[key], on_step=False, on_epoch=True)
                seen_fde_dict[key](self.fde(mu_t, fut_t, mask=mask_t))
                self.log(f"test/{prefix}/seen_fde/{key}", seen_fde_dict[key], on_step=False, on_epoch=True)

        if self.unseen_airports and unseen_ade_dict is not None:
            for airport in self.unseen_airports:
                idx = np.where(airport_ids == airport)[0]
                if len(idx) == 0:
                    continue
                for t in self.pred_lens:
                    mu_t   = ego_mu[idx][:, :, :self.hist_len + t]
                    fut_t  = ego_fut[idx][:, :, :t]
                    mask_t = mask[idx][:, :, :self.hist_len + t]
                    key    = f"{airport}_t={t}"
                    unseen_ade_dict[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(f"test/{prefix}/unseen_ade/{key}", unseen_ade_dict[key], on_step=False, on_epoch=True)
                    unseen_fde_dict[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(f"test/{prefix}/unseen_fde/{key}", unseen_fde_dict[key], on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        print("\n" + "=" * 70)
        print("TRAJECTORY PREDICTION TEST COMPLETE")
        print("=" * 70)

    def configure_optimizers(self) -> Dict[str, Any]:
        decay_params    = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ("temporal_decoder" in name or len(param.shape) == 1
                    or name.endswith(".bias")
                    or any(m in name for m in ['ln', 'norm'])):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [{"params": decay_params,    "weight_decay": self.hparams.optimizer.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2))

        config = {"optimizer": optimizer}
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            config["lr_scheduler"] = {
                "scheduler": scheduler, "monitor": "losses/val_traj",
                "interval": "epoch", "frequency": 1}
        return config


class CombinedTrajPredSystem(LightningModule):
    """
    Two-stage inference system for multi-modal trajectory prediction.

    Pipeline:
        1. mode_model: predicts mode probability distribution (B, A, num_modes)
        2. traj_model: generates one trajectory per turn mode (B, A, T, num_modes, K, D)
    """

    def __init__(
            self,
            mode_model: torch.nn.Module,
            traj_model: torch.nn.Module,
            extra_params: EasyDict
    ):
        super().__init__()
        self.mode_model = mode_model
        self.traj_model = traj_model
        self.eparams = extra_params
        self.num_modes = traj_model.num_modes
        self.hist_len = traj_model.hist_len
        self.pred_lens = traj_model.pred_lens
        self.max_pred_len = max(self.pred_lens)
        self.mode_names  = TURN_MODES_NAMES
        os.makedirs(self.eparams.plot_dir, exist_ok=True)
        out_dir          = os.path.join(self.eparams.plot_dir,
                                        f"{date.today()}_{self.eparams.tag}")
        self.test_out_dir = os.path.join(out_dir, 'test_traj2')

        self.num_hypotheses = traj_model.num_hypotheses

        self.geodesic = Geodesic.WGS84

        # ---------- TEST SET METRICS ----------
        self.test_min_ade = nn.ModuleDict()
        self.test_min_fde = nn.ModuleDict()
        self.test_prob_ade = nn.ModuleDict()
        self.test_prob_fde = nn.ModuleDict()
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_min_ade[key] = MeanMetric()
            self.test_min_fde[key] = MeanMetric()
            self.test_prob_ade[key] = MeanMetric()
            self.test_prob_fde[key] = MeanMetric()
        self.test_nll = MeanMetric()
        self.test_rmse = MeanMetric()

        # ---------- PER-MODE METRICS ----------
        self.test_mode_ade = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)})
        self.test_mode_fde = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)})
        self.test_mode_prob_ade = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)})
        self.test_mode_prob_fde = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)})
        self.test_mode_nll = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}": MeanMetric() for m in VALID_TURN_MODES})
        self.test_mode_rmse = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}": MeanMetric() for m in VALID_TURN_MODES})

        # ---------- OFF-ROAD DETECTION METRICS ----------
        self.test_off_road_rate = MeanMetric()
        self.test_off_road_max_dist = MeanMetric()
        self.test_off_road_mean_dist = MeanMetric()
        self.test_off_road_critical_rate = MeanMetric()
        self.test_off_road_episodes = MeanMetric()
        self.test_off_road_duration = MeanMetric()

        # ---------- PER-MODE OFF-ROAD METRICS ----------
        self.test_mode_off_road_rate = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}": MeanMetric()
            for m in VALID_TURN_MODES
        })
        self.test_mode_off_road_max_dist = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}": MeanMetric()
            for m in VALID_TURN_MODES
        })
        self.test_mode_off_road_mean_dist = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}": MeanMetric()
            for m in VALID_TURN_MODES
        })
        self.test_mode_off_road_critical_rate = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}": MeanMetric()
            for m in VALID_TURN_MODES
        })

        # ---------- DIAGNOSTIC METRICS ----------
        for attr in [
            'test_sigma_mean', 'test_sigma_min', 'test_sigma_max', 'test_sigma_std',
            'test_max_prob_mean', 'test_entropy_mean',
            'test_nll_max_contrib', 'test_nll_mix_penalty', 'test_multi_mode_contrib_percent',
        ]:
            setattr(self, attr, MeanMetric())

        self.ade, self.fde, self.mode_ade, self.mode_fde = self._get_metric_functions()
        self.compute_nll = compute_nll
        self.compute_mode_rmse = compute_mode_rmse

        self.test_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_modes, valid_modes=VALID_TURN_MODES,
            mode_names=TURN_MODES_NAMES)

        # Offline dump for scorer analysis
        self._dump_dir = getattr(self.eparams, 'dump_dir', None)
        self._dump_enabled = getattr(self.eparams, 'dump_enabled', False)
        if self._dump_enabled and self._dump_dir:
            os.makedirs(self._dump_dir, exist_ok=True)

        # Off-road evaluator (lazy initialization)
        self._off_road_evaluator = None
        self._current_airport = None

        # Background map cache for plotting
        self._plot_bg_cache = {}

    def _get_metric_functions(self):
        if self.eparams.propagation == 'marginal':
            return marginal_ade, marginal_fde, mode_ade, mode_fde
        return joint_ade, joint_fde, mode_ade, mode_fde

    def _encode_rule_based_to_mode_index(self, rule_based):
        return rule_based[..., :4].float().argmax(dim=-1).long()

    def _get_off_road_evaluator(self, airport_code: str):
        """Lazy initialization of off-road evaluator for a specific airport."""
        if self._off_road_evaluator is None or self._current_airport != airport_code:
            from amelia_tf.utils.off_road_evaluator import OffRoadEvaluator
            self._off_road_evaluator = OffRoadEvaluator(
                asset_dir=self.eparams.asset_dir,
                airport_code=airport_code
            )
            self._current_airport = airport_code
        return self._off_road_evaluator

    def _evaluate_off_road(self, batch, ego_mu, ego_probs, hist_len, safety_margin=1.0):
        """Helper function to evaluate off-road metrics."""
        try:
            airport_code = batch['scene_dict']['airport_id'][0]
            evaluator = self._get_off_road_evaluator(airport_code)

            off_road_results_list = evaluator.evaluate_prediction(
                batch=batch,
                ego_mu=ego_mu,
                ego_pred_scores=ego_probs,
                hist_len=hist_len,
                safety_margin=safety_margin
            )
            return off_road_results_list
        except Exception as e:
            print(f"Warning: Off-road evaluation failed: {e}")
            return None

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full two-stage inference forward pass.

        turn_feasibility is forwarded to the mode model so that its soft
        conditioning and optional hard masking take effect at inference time.
        """
        Y = batch['scene_dict']['rel_sequences']
        X = torch.zeros_like(Y).float()
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]

        context = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
        B, A = X.shape[:2]

        feasibility = batch['scene_dict'].get('turn_feasibility', None)
        if feasibility is not None:
            feasibility = feasibility.to(X.device)

        # Stage 1: predict mode probability distribution
        mode_logits = self.mode_model(
            X[:, :, :, :4],
            context=context, adjacency=adjacency,
            mask=None,
            feasibility=feasibility,
            output_mode_only=True
        )
        mode_probs = torch.softmax(mode_logits, dim=-1)

        # Stage 2: one call per turn mode, each returns K hypotheses
        # traj_mu stack shape: (B, A, T, M, K, D)
        all_mu, all_sigma, all_score = [], [], []
        _gmm = None
        for _m in self.traj_model.modules():          # to find gmm
            if hasattr(_m, 'per_mode_score'):
                _gmm = _m
                break
        for k in range(self.num_modes):
            if _gmm is not None:
                _gmm._active_mode = k
            one_hot = F.one_hot(
                torch.full((B, A), k, dtype=torch.long, device=X.device),
                num_classes=self.num_modes).float()
            mu_k, sigma_k, score_k = self.traj_model(
                X[:, :, :, :4], context=context, adjacency=adjacency,
                mask=None, mode_probs=one_hot)
            all_mu.append(mu_k)
            all_sigma.append(sigma_k)
            all_score.append(score_k)

        traj_mu = torch.stack(all_mu, dim=3)  # (B,A,T,M,K,D)
        traj_sigma = torch.stack(all_sigma, dim=3)
        traj_score = (torch.stack(all_score, dim=3)
                      if all_score[0] is not None else None)
        return mode_probs, traj_mu, traj_sigma, traj_score

    # --------------------------------------------------------------------------
    # Diagnostic helpers
    # --------------------------------------------------------------------------

    def _log_sigma_stats(self, sigma, prefix="test") -> None:
        sigma_flat = sigma[~torch.isnan(sigma)]
        if sigma_flat.numel() > 0:
            self.log(f"{prefix}/sigma_mean", sigma_flat.mean(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_min", sigma_flat.min(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_max", sigma_flat.max(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_std", sigma_flat.std(), on_step=False, on_epoch=True)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                self.log(f"{prefix}/sigma_p{p}",
                         torch.quantile(sigma_flat, p / 100.0),
                         on_step=False, on_epoch=True)

    def _log_mode_probs_stats(self, mode_probs, prefix="test") -> None:
        B, A, M = mode_probs.shape
        for k in range(M):
            self.log(f"{prefix}/mode_{k}_prob", mode_probs[..., k].mean(),
                     on_step=False, on_epoch=True)
        max_probs = mode_probs.max(dim=-1)[0]
        self.log(f"{prefix}/max_prob_mean", max_probs.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/max_prob_min", max_probs.min(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/max_prob_std", max_probs.std(), on_step=False, on_epoch=True)
        entropy = -(mode_probs * torch.log(mode_probs.clamp_min(1e-8))).sum(dim=-1)
        self.log(f"{prefix}/entropy_mean", entropy.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/entropy_std", entropy.std(), on_step=False, on_epoch=True)
        mode_counts = mode_probs.argmax(dim=-1).flatten()
        for k in range(M):
            pct = (mode_counts == k).sum().float() / mode_counts.numel() * 100
            self.log(f"{prefix}/mode_{k}_percentage", pct, on_step=False, on_epoch=True)

    def _log_prediction_quality(self, mu, sigma, Y, mode_probs, mask=None, prefix="test") -> None:
        B, A, T_total, M, D = mu.shape
        _, _, T_pred, _ = Y.shape
        mu_f = mu[:, :, -T_pred:, :, :]
        sig_f = sigma[:, :, -T_pred:, :, :].clamp_min(1e-3)
        best_idx = mode_probs.argmax(dim=-1)
        best_mu = mu_f[torch.arange(B)[:, None, None], torch.arange(A)[None, :, None],
                  torch.arange(T_pred)[None, None, :], best_idx[:, :, None], :]
        best_sig = sig_f[torch.arange(B)[:, None, None], torch.arange(A)[None, :, None],
                   torch.arange(T_pred)[None, None, :], best_idx[:, :, None], :]
        log_prob = -0.5 * (((best_mu - Y) / best_sig) ** 2 +
                           torch.log(2 * torch.pi * best_sig ** 2)).sum(dim=-1)
        if mask is not None:
            mp = mask[:, :, -T_pred:]
            lpm = (log_prob * mp).sum(dim=-1) / mp.sum(dim=-1).clamp_min(1)
        else:
            lpm = log_prob.mean(dim=-1)
        self.log(f"{prefix}/best_mode_log_prob_mean", lpm.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_log_prob_min", lpm.min(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_log_prob_max", lpm.max(), on_step=False, on_epoch=True)

    def _analyze_nll_components(self, mu, sigma, mode_probs, Y, mask=None, prefix="test") -> None:
        B, A, T_total, M, D = mu.shape
        _, _, T_pred, _ = Y.shape
        mu_f = mu[:, :, -T_pred:, :, :]
        sig_f = sigma[:, :, -T_pred:, :, :].clamp_min(1e-3)
        ye = Y.unsqueeze(-2)
        logp = -0.5 * (torch.log(2 * torch.pi * sig_f ** 2) + ((ye - mu_f) / sig_f) ** 2)
        logp = logp.sum(dim=-1)
        log_pi = torch.log(mode_probs.clamp_min(1e-8)).unsqueeze(2)
        logp_mix = logp + log_pi
        max_logp = logp_mix.max(dim=-1)[0]
        lse = torch.logsumexp(logp_mix, dim=-1)
        mix_pen = lse - max_logp
        if mask is not None:
            mp = mask[:, :, -T_pred:]

            def _mean(x):
                return (x * mp).sum(dim=-1) / mp.sum(dim=-1).clamp_min(1)
        else:
            def _mean(x):
                return x.mean(dim=-1)
        self.log(f"{prefix}/nll_max_mode_contribution", (-_mean(max_logp)).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_log_sum_exp_contribution", (-_mean(lse)).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty", _mean(mix_pen).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty_max", mix_pen.max(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty_min", mix_pen.min(), on_step=False, on_epoch=True)
        sig_contrib = (mix_pen > 0.1).float()
        if mask is not None:
            mp2 = mask[:, :, -T_pred:]
            pct = sig_contrib.sum() / mp2.sum().clamp_min(1) * 100
        else:
            pct = sig_contrib.mean() * 100
        self.log(f"{prefix}/samples_with_multi_mode_contribution_percent",
                 pct, on_step=False, on_epoch=True)

    def _filter_batch(self, batch: dict, mask: torch.Tensor) -> dict:
        """
        Return a new batch containing only the samples selected by boolean mask.

        Fields are split into three groups:
          - per-sample (dim 0 == B): tensors indexed directly by mask.
          - non-tensor scalars / None: passed through unchanged.
          - everything else (e.g. context / adjacency padded to B*k_agents):
            left unchanged because evaluate_prediction does not read them.
        """
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        B = mask.shape[0]

        per_sample_keys = {
            'sequences', 'rel_sequences', 'agent_masks',
            'rule_based_encoding', 'turn_feasibility',
            'ego_agent_id', 'airport_id', 'scenario_id', 'num_agents',
        }

        filtered = {'scene_dict': {}}
        for key, val in batch['scene_dict'].items():
            if key in per_sample_keys:
                if isinstance(val, torch.Tensor) and val.shape[0] == B:
                    filtered['scene_dict'][key] = val[mask]
                elif isinstance(val, np.ndarray) and val.shape[0] == B:
                    filtered['scene_dict'][key] = val[mask_np]
                elif isinstance(val, list) and len(val) == B:
                    filtered['scene_dict'][key] = [
                        v for v, m in zip(val, mask_np) if m
                    ]
                else:
                    filtered['scene_dict'][key] = val
            else:
                filtered['scene_dict'][key] = val
        return filtered

    # --------------------------------------------------------------------------
    # Selection diagnostics
    # --------------------------------------------------------------------------
    #
    # Selection accuracy (did the score pick the oracle / min-ADE candidate)
    # is evaluated on the candidates of the PREDICTED mode, since that is what
    # a deployed system actually selects from at inference time. Results are
    # split by whether the mode prediction itself was correct or wrong, both
    # overall and per (true) mode. The oracle shown in the high-ADE plots is
    # still computed on the TRUE mode's candidates, as the upper bound if mode
    # classification were perfect.
    #
    # z-variation / valid-ratio diagnostics were removed: the dataset is
    # already filtered (runway trajectories excluded upstream), so these
    # per-sample checks are no longer needed.
    # --------------------------------------------------------------------------

    def _reset_selection_diag(self) -> None:
        """
        Reset accumulators at the start of each test epoch.

        Accumulators are kept as GPU tensors (not Python ints/dicts) so that
        per-batch updates in _update_selection_diag are pure in-place tensor
        adds with NO device sync. The only sync for the whole test epoch
        happens once, in _finalise_selection_diag. Per-batch syncing (even a
        single .item()/.tolist() call per batch) serialises against the GPU's
        async execution queue and was the actual cost driver, not the choice
        of reduction op within a batch.
        """
        M = self.num_modes
        device = self.device
        self._sel_total_mode_ok_pm = torch.zeros(M, dtype=torch.long, device=device)
        self._sel_correct_mode_ok_pm = torch.zeros(M, dtype=torch.long, device=device)
        self._sel_total_mode_wrong_pm = torch.zeros(M, dtype=torch.long, device=device)
        self._sel_correct_mode_wrong_pm = torch.zeros(M, dtype=torch.long, device=device)
        self._ade_mode_correct_sum_pm = torch.zeros(M, dtype=torch.float32, device=device)
        self._ade_mode_wrong_sum_pm = torch.zeros(M, dtype=torch.float32, device=device)

        # Candidates for high-ADE visualisation
        self._highade_candidates = []
        self._highade_max_keep = int(getattr(self.eparams, 'highade_plot_topk', 20))

    def _retrieve_neighbours_for_plot(self, hist_abs_b, node_abs_b, node_head_b,
                                      pred_mode_b, cand_rel_b=None):
        """
        Re-run retrieval for a SINGLE sample, for plotting. Returns the retrieved
        real futures in the rel frame, AND (if cand_rel_b is given) the exact
        per-candidate scoring for this sample so the plot can display numbers
        that correspond to the drawn case.
 
        hist_abs_b  : (hl,2) absolute local-xy history
        node_abs_b  : (2,)   node absolute position
        node_head_b : scalar node heading in RADIANS
        pred_mode_b : int    predicted mode
        cand_rel_b  : (Tp,K,2) this sample's candidates in rel frame, optional
 
        Returns dict: {'nn_rel': (kk,Tp,2) or None, 'score': {...} or None}
        """
        out = {'nn_rel': None, 'score': None}
        db = getattr(self, '_retrieval_db', None)
        if db is None or pred_mode_b not in db:
            return out
        dbm = db[pred_mode_b]
        key_db = dbm['key']
        fut_db = dbm['fut']
        dev = key_db.device
        q = hist_abs_b.reshape(1, -1).to(dev)
        d = torch.cdist(q, key_db)
        kk = min(getattr(self, '_retrieval_topk', 20), key_db.shape[0])
        topv, nn_idx = d.topk(kk, largest=False)
        nn_idx = nn_idx[0]
        nn_local = fut_db[nn_idx].to(dev)                    # (kk,Tp,2) local xy
 
        # local -> rel using THIS sample's node (draw in this sample's frame)
        n_abs = node_abs_b.to(dev)
        shifted = nn_local - n_abs[None, None, :]
        c_h = torch.cos(-node_head_b); s_h = torch.sin(-node_head_b)
        rx = c_h * shifted[..., 0] - s_h * shifted[..., 1]
        ry = s_h * shifted[..., 0] + c_h * shifted[..., 1]
        nn_rel = torch.stack([rx, ry], dim=-1)               # (kk,Tp,2) rel
        out['nn_rel'] = nn_rel.cpu().numpy()
 
        # exact scoring for this sample, mirroring the selection branch. Both
        # candidates and neighbours are put in local xy (the selection frame).
        if cand_rel_b is not None:
            cand = self._rel_to_local_single(cand_rel_b.to(dev), n_abs, node_head_b)  # (Tp,K,2) local
            Tp = min(cand.shape[0], nn_local.shape[1])
            c = cand[:Tp].permute(1, 0, 2)                   # (K,Tp,2)
            nf = nn_local[:, :Tp, :]                         # (kk,Tp,2)
            dist = (c[:, None, :, :] - nf[None, :, :, :]).norm(dim=-1).mean(dim=-1)  # (K,kk)
            tau = float(getattr(self.eparams, 'retrieval_tau', 0.05))
            support = torch.exp(-dist / tau).sum(dim=-1)     # (K,)
            out['score'] = {
                'mean_dist': [round(x, 4) for x in dist.mean(-1).tolist()],
                'support': [round(x, 3) for x in support.tolist()],
                'pick': int(support.argmax().item()),
                'knn_keydist': round(float(topv.mean().item()), 4),
            }
        return out
        
    def _rel_to_local_single(self, rel, node_abs, node_head):
        # rel: (Tp,K,2) -> local xy (Tp,K,2), same transform as the selection branch
        c = torch.cos(node_head); s = torch.sin(node_head)
        rx = rel[..., 0]; ry = rel[..., 1]
        gx = c * rx - s * ry
        gy = s * rx + c * ry
        return torch.stack([gx, gy], dim=-1) + node_abs[None, None, :]
 

    def _update_selection_diag(self, ego_mu_full, ego_fut, ego_mask,
                               ego_probs, ego_true_mode, selected_k_idx,
                               batch, dataloader_idx):
        """
        Accumulate selection accuracy and selected-ADE, split by whether the
        mode prediction was correct or wrong (overall and per true mode), and
        collect high-ADE samples for plotting.
 
        ego_mu_full   : (B,1,T,M,K,D)  all K candidates
        ego_fut       : (B,1,Tp,D)     ground-truth future (ego)
        ego_mask      : (B,1,T)        mask
        ego_probs     : (B,1,M)        mode probabilities
        ego_true_mode : (B,1) or (B,)  true mode index
        selected_k_idx: (B,M) or None  score-selected candidate per mode
        """
        if selected_k_idx is None:
            return
        with torch.no_grad():
            B = ego_mu_full.shape[0]
            M = self.num_modes
            hl = self.hist_len
 
            true_mode = ego_true_mode.reshape(B, -1)[:, 0].long()  # (B,)
            pred_mode = ego_probs.reshape(B, M).argmax(-1)  # (B,)
            valid = (true_mode >= 0) & (true_mode < M) & (pred_mode >= 0) & (pred_mode < M)
 
            fut_cand = ego_mu_full[:, 0, hl:, :, :, :2]  # (B,Tp,M,K,2)
            gt = ego_fut[:, 0, :, :2]  # (B,Tp,2)
            m_fut = ego_mask[:, 0, hl:].float()  # (B,Tp)
            Tp, K = fut_cand.shape[1], fut_cand.shape[3]
 
            # ---- gather the predicted-mode candidate set for every sample
            # at once, and compute all K candidates' ADE in one shot. This is
            # both what is actually selected from at inference, and what the
            # oracle reference in the high-ADE plots is drawn from, so oracle
            # and selected are always directly comparable on the same
            # candidate set (no mode-classification error mixed in). ----
            pm_safe = pred_mode.clamp(0, M - 1)
            pm_exp = pm_safe[:, None, None, None, None].expand(B, Tp, 1, K, 2)
            cand_pred = fut_cand.gather(2, pm_exp).squeeze(2)  # (B,Tp,K,2)
 
            denom = m_fut.sum(dim=1).clamp_min(1)  # (B,)
            d = ((cand_pred - gt[:, :, None, :]) ** 2).sum(-1).sqrt()  # (B,Tp,K)
            ade_k = (d * m_fut[:, :, None]).sum(dim=1) / denom[:, None]  # (B,K)
 
            oracle_k = ade_k.argmin(dim=-1)  # (B,)
            sel_k = selected_k_idx.gather(1, pm_safe[:, None]).squeeze(1)  # (B,)
            sel_ade = ade_k.gather(1, sel_k[:, None]).squeeze(1)  # (B,)
            oracle_ade = ade_k.gather(1, oracle_k[:, None]).squeeze(1)  # (B,)
            selection_correct = (sel_k == oracle_k)
            mode_ok = (pred_mode == true_mode)
 
            # ---- accumulate accuracy / ADE stats with ZERO device syncs per
            # batch. Earlier versions synced once per batch (either many
            # .item() calls or one combined .tolist()) -- but ANY per-batch
            # sync serialises against the GPU's async queue, and that per-
            # batch cost (not the choice of reduction op) is what actually
            # dominates over a full test epoch with many batches. Here the
            # per-mode one-hot sums are added in-place directly onto the GPU
            # accumulator tensors (see _reset_selection_diag) with no
            # .item()/.tolist() at all; the single sync for the whole epoch
            # happens once, in _finalise_selection_diag. ----
            tm_v = true_mode[valid]
            wrong_v = (~mode_ok)[valid]          # False = mode correct, True = mode wrong
            sel_ade_v = sel_ade[valid]
            sel_correct_v = selection_correct[valid]
 
            if tm_v.numel() > 0:
                tm_onehot = F.one_hot(tm_v, num_classes=M)               # (n_valid, M) long
                ok_onehot = tm_onehot * (~wrong_v)[:, None]              # (n_valid, M) long
                wrong_onehot = tm_onehot * wrong_v[:, None]              # (n_valid, M) long
 
                self._sel_total_mode_ok_pm += ok_onehot.sum(dim=0)
                self._sel_total_mode_wrong_pm += wrong_onehot.sum(dim=0)
                self._sel_correct_mode_ok_pm += (
                    ok_onehot * sel_correct_v[:, None]).sum(dim=0)
                self._sel_correct_mode_wrong_pm += (
                    wrong_onehot * sel_correct_v[:, None]).sum(dim=0)
                self._ade_mode_correct_sum_pm += (
                    ok_onehot.float() * sel_ade_v[:, None]).sum(dim=0)
                self._ade_mode_wrong_sum_pm += (
                    wrong_onehot.float() * sel_ade_v[:, None]).sum(dim=0)
 
            # ---- high-ADE plotting: build per-sample metadata only for
            # samples that can actually enter the kept top-K (compare against
            # the current worst kept sel_ade), instead of every sample in the
            # test set. Most batches contribute nothing once the list is full. ----
            if valid.any():
                cur_worst = (min(c['sel_ade'] for c in self._highade_candidates)
                            if len(self._highade_candidates) >= self._highade_max_keep
                            else -float('inf'))
                cand_idx = valid.nonzero(as_tuple=True)[0]
                keep_mask = sel_ade[cand_idx] > cur_worst
                cand_idx = cand_idx[keep_mask]
 
                if cand_idx.numel() > 0:
                    sequences = batch['scene_dict']['sequences']
                    ego_agent_ids = batch['scene_dict']['ego_agent_id']
                    airport_ids = batch['scene_dict']['airport_id']
                    rel_seq = batch['scene_dict']['rel_sequences']  # (B, A, T, 7)
                    turn_feas = batch['scene_dict'].get('turn_feasibility', None)  # (B,A,4) or None
 
                    for b in cand_idx.tolist():
                        ego_id = ego_agent_ids[b]
                        if torch.is_tensor(ego_id):
                            ego_id = ego_id.item()
                        try:
                            start_abs = sequences[b, ego_id, hl - 1, G.XY].detach().cpu().numpy()
                            start_heading = sequences[b, ego_id, hl - 1, G.HD].detach().cpu().numpy().item()
                        except Exception:
                            start_abs = np.array(gt[b, 0, :2].cpu().numpy(), dtype=np.float64)
                            start_heading = 0.0
 
                        rel_xy = rel_seq[b, ego_id, :hl, [0, 1]].detach().cpu().numpy()
 
                        airport = airport_ids[b]
                        if torch.is_tensor(airport):
                            airport = airport.item()
 
                        # feasible_TurnLeft/TurnRight/Straight/Hold (0/1), in
                        # the same order as self.mode_names. This is the
                        # topology-derived hard mask already applied upstream
                        # to mode_logits, so pred_mode should always show
                        # feasible=1 here -- useful as a sanity check, and to
                        # see how many modes were even legally possible at
                        # this position (e.g. a dead-end leaves only one).
                        if turn_feas is not None:
                            try:
                                feas_vec = turn_feas[b, ego_id, :].detach().cpu().numpy().tolist()
                            except Exception:
                                feas_vec = None
                        else:
                            feas_vec = None
 
                        # retrieved neighbours for this sample, re-run fresh here
                        # (gated by a switch). Building the same absolute-history
                        # query and calling the shared retrieval helper avoids any
                        # cross-batch cache/timing/index issues -- what's drawn is
                        # exactly what the same retrieval returns for this sample.
                        nn_rel = None
                        nn_score = None
                        if getattr(self.eparams, 'plot_retrieval_neighbours', False) \
                                and getattr(self, '_retrieval_db', None) is not None:
                            try:
                                sx, sy = G.SEQ_IDX.x, G.SEQ_IDX.y
                                raw_b = sequences[b, ego_id]                       # (T,C)
                                hist_abs_b = raw_b[:hl, [sx, sy]]                  # (hl,2) abs
                                node_abs_b = raw_b[hl - 1, [sx, sy]]              # (2,)
                                node_head_b = raw_b[hl - 1, G.SEQ_IDX.Heading] * (torch.pi / 180.0)
                                # this sample's candidates for the predicted mode, rel
                                cand_rel_b = cand_pred[b]                          # (Tp,K,2) rel
                                res = self._retrieve_neighbours_for_plot(
                                    hist_abs_b, node_abs_b, node_head_b,
                                    int(pred_mode[b].item()),
                                    cand_rel_b=torch.as_tensor(cand_rel_b, device=sequences.device)
                                        if not torch.is_tensor(cand_rel_b) else cand_rel_b)
                                nn_rel = res['nn_rel']
                                nn_score = res['score']
                            except Exception as e:
                                print(f"[plot-nn] retrieval for sample failed: {e}")
                                nn_rel = None; nn_score = None
 
                        self._highade_candidates.append({
                            'sel_ade': float(sel_ade[b].item()),
                            'oracle_ade': float(oracle_ade[b].item()),
                            'sel_k': int(sel_k[b].item()),
                            'oracle_k': int(oracle_k[b].item()),
                            'true_mode': int(true_mode[b].item()),
                            'pred_mode': int(pred_mode[b].item()),
                            'turn_feasibility': feas_vec,  # [TL,TR,Straight,Hold] 0/1 or None
                            'cand': cand_pred[b].detach().cpu().numpy(),  # (Tp,K,2)
                            'gt': gt[b].detach().cpu().numpy(),  # (Tp,2)
                            'dl_idx': dataloader_idx,
                            'airport': airport,
                            'start_abs': start_abs,
                            'start_heading': start_heading,
                            'history': rel_xy,
                            'nn_rel': nn_rel,  # (kk,Tp,2) rel retrieved neighbours, or None
                            'nn_score': nn_score,  # per-candidate dist/support/pick for THIS sample
                        })
 
                    # Keep only the worst (highest sel_ade) samples
                    self._highade_candidates.sort(key=lambda r: r['sel_ade'], reverse=True)
                    if len(self._highade_candidates) > self._highade_max_keep:
                        self._highade_candidates = self._highade_candidates[:self._highade_max_keep]

    def _finalise_selection_diag(self) -> None:
        """Log accumulated selection diagnostics and plot the worst high-ADE samples."""
        # ---- the ONE sync for the whole test epoch. Every batch during
        # _update_selection_diag only did in-place GPU tensor adds (no
        # .item()/.tolist()); here, at the very end, all accumulators are
        # pulled to the CPU in a single combined .tolist() call and
        # re-expanded into the same self._* attribute names/shapes (plain
        # dicts/ints) that the rest of this method already expects, so
        # nothing below this block needs to change. ----
        M = self.num_modes
        stacked = torch.cat([
            self._sel_total_mode_ok_pm.float(),
            self._sel_correct_mode_ok_pm.float(),
            self._sel_total_mode_wrong_pm.float(),
            self._sel_correct_mode_wrong_pm.float(),
            self._ade_mode_correct_sum_pm,
            self._ade_mode_wrong_sum_pm,
        ])
        vals = stacked.tolist()  # <-- single device sync for the entire epoch

        tot_ok_pm = [int(round(v)) for v in vals[0 * M:1 * M]]
        corr_ok_pm = [int(round(v)) for v in vals[1 * M:2 * M]]
        tot_wrong_pm = [int(round(v)) for v in vals[2 * M:3 * M]]
        corr_wrong_pm = [int(round(v)) for v in vals[3 * M:4 * M]]
        ade_correct_sum_pm = vals[4 * M:5 * M]
        ade_wrong_sum_pm = vals[5 * M:6 * M]

        self._sel_total_mode_ok_pm = {m: tot_ok_pm[m] for m in range(M)}
        self._sel_correct_mode_ok_pm = {m: corr_ok_pm[m] for m in range(M)}
        self._sel_total_mode_wrong_pm = {m: tot_wrong_pm[m] for m in range(M)}
        self._sel_correct_mode_wrong_pm = {m: corr_wrong_pm[m] for m in range(M)}
        self._ade_mode_correct_sum_pm = {m: ade_correct_sum_pm[m] for m in range(M)}
        self._ade_mode_correct_n_pm = {m: tot_ok_pm[m] for m in range(M)}
        self._ade_mode_wrong_sum_pm = {m: ade_wrong_sum_pm[m] for m in range(M)}
        self._ade_mode_wrong_n_pm = {m: tot_wrong_pm[m] for m in range(M)}

        self._sel_total_mode_ok = sum(tot_ok_pm)
        self._sel_correct_mode_ok = sum(corr_ok_pm)
        self._sel_total_mode_wrong = sum(tot_wrong_pm)
        self._sel_correct_mode_wrong = sum(corr_wrong_pm)
        self._ade_mode_correct_sum = sum(ade_correct_sum_pm)
        self._ade_mode_correct_n = sum(tot_ok_pm)
        self._ade_mode_wrong_sum = sum(ade_wrong_sum_pm)
        self._ade_mode_wrong_n = sum(tot_wrong_pm)

        # ---- everything below is unchanged: same attribute names, now
        # populated as plain Python ints/dicts instead of GPU tensors. ----
        print("\n" + "-" * 70)
        print("SELECTION ACCURACY (selected == oracle, on the predicted-mode candidates)")
        print("-" * 70)

        if self._sel_total_mode_ok > 0:
            acc_ok = self._sel_correct_mode_ok / self._sel_total_mode_ok
            self.log("test/selection_acc/mode_correct", acc_ok,
                     on_step=False, on_epoch=True)
            print(f"  mode CORRECT: {acc_ok:.4f} "
                  f"({self._sel_correct_mode_ok}/{self._sel_total_mode_ok})")
        if self._sel_total_mode_wrong > 0:
            acc_wrong = self._sel_correct_mode_wrong / self._sel_total_mode_wrong
            self.log("test/selection_acc/mode_wrong", acc_wrong,
                     on_step=False, on_epoch=True)
            print(f"  mode WRONG:   {acc_wrong:.4f} "
                  f"({self._sel_correct_mode_wrong}/{self._sel_total_mode_wrong})")

        print("\n  per-mode breakdown:")
        for m in range(self.num_modes):
            name = self.mode_names[m]
            tot_ok = self._sel_total_mode_ok_pm[m]
            tot_wrong = self._sel_total_mode_wrong_pm[m]

            if tot_ok > 0:
                acc = self._sel_correct_mode_ok_pm[m] / tot_ok
                self.log(f"test/selection_acc/mode_correct/{name}", acc,
                         on_step=False, on_epoch=True)
                print(f"    {name} (mode CORRECT): {acc:.4f} "
                      f"({self._sel_correct_mode_ok_pm[m]}/{tot_ok})")
            else:
                print(f"    {name} (mode CORRECT): N/A (no samples)")

            if tot_wrong > 0:
                acc = self._sel_correct_mode_wrong_pm[m] / tot_wrong
                self.log(f"test/selection_acc/mode_wrong/{name}", acc,
                         on_step=False, on_epoch=True)
                print(f"    {name} (mode WRONG):   {acc:.4f} "
                      f"({self._sel_correct_mode_wrong_pm[m]}/{tot_wrong})")
            else:
                print(f"    {name} (mode WRONG):   N/A (no samples)")
        print("-" * 70)

        if self._ade_mode_correct_n > 0:
            ade_c = self._ade_mode_correct_sum / self._ade_mode_correct_n
            self.log("test/selected_ade/mode_correct", ade_c,
                     on_step=False, on_epoch=True)
            print(f"[mode-split] selected ADE (mode CORRECT): {ade_c:.4f} "
                  f"(n={self._ade_mode_correct_n})")
        if self._ade_mode_wrong_n > 0:
            ade_w = self._ade_mode_wrong_sum / self._ade_mode_wrong_n
            self.log("test/selected_ade/mode_wrong", ade_w,
                     on_step=False, on_epoch=True)
            print(f"[mode-split] selected ADE (mode WRONG):   {ade_w:.4f} "
                  f"(n={self._ade_mode_wrong_n})")

        print("\n" + "-" * 70)
        print("PER-MODE SELECTED ADE (split by mode prediction correctness)")
        print("-" * 70)

        for m in range(self.num_modes):
            name = self.mode_names[m]

            n_correct = self._ade_mode_correct_n_pm.get(m, 0)
            if n_correct > 0:
                ade_correct = self._ade_mode_correct_sum_pm[m] / n_correct
                self.log(f"test/selected_ade/{name}/mode_correct", ade_correct,
                         on_step=False, on_epoch=True)
                print(f"  {name} (mode CORRECT): {ade_correct:.4f} (n={n_correct})")
            else:
                print(f"  {name} (mode CORRECT): N/A (no samples)")

            n_wrong = self._ade_mode_wrong_n_pm.get(m, 0)
            if n_wrong > 0:
                ade_wrong = self._ade_mode_wrong_sum_pm[m] / n_wrong
                self.log(f"test/selected_ade/{name}/mode_wrong", ade_wrong,
                         on_step=False, on_epoch=True)
                print(f"  {name} (mode WRONG):   {ade_wrong:.4f} (n={n_wrong})")
            else:
                print(f"  {name} (mode WRONG):   N/A (no samples)")

            if n_correct > 0 and n_wrong > 0:
                diff = ade_wrong - ade_correct
                self.log(f"test/selected_ade/{name}/wrong_vs_correct_diff", diff,
                         on_step=False, on_epoch=True)
                print(f"  {name} diff (wrong - correct): {diff:.4f}")

        print("-" * 70)

        # ---- plot the worst high-ADE samples ----
        self._plot_highade_samples()

    def _plot_highade_samples(self) -> None:
        """
        Plot high-ADE samples with background map, marking oracle vs
        score-selected candidate. Includes historical trajectory
        visualisation.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cv2
        import json
        import numpy as np
        from easydict import EasyDict
        from amelia_scenes.utils.transform_utils import xy_to_ll
 
        cands = getattr(self, '_highade_candidates', [])
        if not cands:
            return
 
        out_dir = os.path.join(self.test_out_dir, 'high_ade')
        os.makedirs(out_dir, exist_ok=True)
 
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f']
 
        for rank, r in enumerate(cands):
            cand = r['cand']  # (Tp, K, 2)
            gt = r['gt']      # (Tp, 2)
            Tp, K, _ = cand.shape
            airport = r.get('airport', 'kmsy')
 
            # Load background map with caching
            if airport not in self._plot_bg_cache:
                bg_path = os.path.join(self.eparams.asset_dir, airport, 'bkg_map.png')
                if os.path.exists(bg_path):
                    im = cv2.imread(bg_path)
                    if im is not None:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, (im.shape[0] // 2, im.shape[1] // 2))
                    else:
                        im = None
                else:
                    im = None
 
                limits_path = os.path.join(self.eparams.asset_dir, airport, 'limits.json')
                extent = None
                ref = None
                if os.path.exists(limits_path):
                    with open(limits_path, 'r') as f:
                        ref_data = json.load(f)
                    if 'espg_4326' in ref_data:
                        espg = EasyDict(ref_data['espg_4326'])
                        extent = (espg.north, espg.east, espg.south, espg.west)
                    ref = (ref_data['ref_lat'], ref_data['ref_lon'], ref_data['range_scale'])
 
                self._plot_bg_cache[airport] = {'image': im, 'extent': extent, 'ref': ref}
 
            bg_info = self._plot_bg_cache[airport]
            bg_img = bg_info['image']
            extent = bg_info['extent']
            ref = bg_info['ref']
 
            # ---- speed curves, computed from existing history/gt/cand data ----
            # No new fields needed in the collected dict: speed is derived
            # here from position differences. All positions are in the
            # ego-centric rel frame where the last history step is the
            # origin (0,0), so prepending a zero point before the future
            # segments correctly captures the first step's speed too.
            history = r.get('history', None)
 
            def _speed_series(xy):
                """(T,2) positions -> (T-1,) per-step speed, 1 Hz so units/step == units/sec."""
                diffs = xy[1:, :] - xy[:-1, :]
                return np.linalg.norm(diffs, axis=-1)
 
            speed_hist = _speed_series(history) if history is not None and len(history) > 1 else np.array([])
            t_hist = np.arange(-len(speed_hist), 0) if len(speed_hist) > 0 else np.array([])
 
            origin = np.zeros((1, 2), dtype=gt.dtype)
            gt_with_origin = np.concatenate([origin, gt], axis=0)
            speed_gt = _speed_series(gt_with_origin)           # (Tp,)
            t_fut = np.arange(1, len(speed_gt) + 1)
 
            sel_k_i, oracle_k_i = r['sel_k'], r['oracle_k']
            cand_sel_with_origin = np.concatenate([origin, cand[:, sel_k_i, :]], axis=0)
            speed_sel = _speed_series(cand_sel_with_origin)     # (Tp,)
            cand_oracle_with_origin = np.concatenate([origin, cand[:, oracle_k_i, :]], axis=0)
            speed_oracle = _speed_series(cand_oracle_with_origin)  # (Tp,)
 
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(24, 12))
 
            # Plot background map
            if bg_img is not None and extent is not None:
                north, east, south, west = extent
                ax.imshow(bg_img, zorder=0, extent=[west, east, south, north],
                          alpha=0.7, cmap='gray_r')
 
            # Get starting position and heading
            start_abs = r.get('start_abs', np.array(gt[0, :2], dtype=np.float64))
            start_heading = r.get('start_heading', 0.0)
 
            if isinstance(start_abs, (list, tuple)):
                start_abs = np.array(start_abs, dtype=np.float64)
            if isinstance(start_heading, (list, tuple)):
                start_heading = np.array(start_heading, dtype=np.float64)
 
            # ========== Historical trajectory ==========
            history = r.get('history', None)
            if history is not None and len(history) > 0:
                hist_batch = history[np.newaxis, :, :].astype(np.float32)
                hist_ll = xy_to_ll(
                    torch.from_numpy(hist_batch),
                    start_abs.reshape(1, -1) if isinstance(start_abs, np.ndarray) else np.array([start_abs]),
                    np.array([start_heading]) if isinstance(start_heading, (int, float)) else start_heading.reshape(1),
                    ref,
                    self.geodesic
                ).numpy()[0]
                hist_lon = hist_ll[:, 1]
                hist_lat = hist_ll[:, 0]
 
                ax.plot(hist_lon, hist_lat, 'k--', linewidth=2.5, alpha=0.7, label='History')
                ax.scatter(hist_lon[0], hist_lat[0], c='dimgray', s=60, marker='o', zorder=4, alpha=0.8)
                ax.scatter(hist_lon[-1], hist_lat[-1], c='dimgray', s=80, marker='s', zorder=4, alpha=0.8)
            # ==========================================
 
            # ===== retrieved neighbours (analog precedents) =====
            # Faint thin lines showing the real train futures that voted for the
            # selection. If they cluster near GT and the selected (red) line
            # follows them, retrieval is working; if they're scattered or the
            # red line ignores them, it explains a bad pick.
            # Convert ONE neighbour at a time with the exact same xy_to_ll call
            # shape as a single candidate (below), so no batch-dim format issue.
            nn_rel = r.get('nn_rel', None)
            if nn_rel is not None and len(nn_rel) > 0:
                for j in range(nn_rel.shape[0]):
                    colors = plt.cm.tab10(np.linspace(0, 1, self._highade_max_keep))
                    nn_batch = nn_rel[j][np.newaxis, :, :].astype(np.float32)   # (1,Tp,2)
                    nn_ll = xy_to_ll(
                        torch.from_numpy(nn_batch),
                        start_abs.reshape(1, -1) if isinstance(start_abs, np.ndarray) else np.array([start_abs]),
                        np.array([start_heading]) if isinstance(start_heading, (int, float)) else start_heading.reshape(1),
                        ref,
                        self.geodesic
                    ).numpy()[0]   # (Tp,2)
                    color = colors[j % len(colors)]
                    ax.plot(nn_ll[:, 1], nn_ll[:, 0], '-', color=color,
                            linewidth=2.0, alpha=0.8,
                            label='Retrieved neighbours' if j == 0 else None, zorder=1)
 
            # Convert all predictions from relative to lat/lon
            all_pred_lon, all_pred_lat = [], []
            for k in range(K):
                rel_traj = cand[:, k, :]
                rel_traj_batch = rel_traj[np.newaxis, :, :].astype(np.float32)
 
                pred_future_ll = xy_to_ll(
                    torch.from_numpy(rel_traj_batch),
                    start_abs.reshape(1, -1) if isinstance(start_abs, np.ndarray) else np.array([start_abs]),
                    np.array([start_heading]) if isinstance(start_heading, (int, float)) else start_heading.reshape(1),
                    ref,
                    self.geodesic
                ).numpy()[0]
 
                pred_lon = pred_future_ll[:, 1]
                pred_lat = pred_future_ll[:, 0]
                all_pred_lon.extend(pred_lon)
                all_pred_lat.extend(pred_lat)
 
                is_oracle = (k == r['oracle_k'])
                is_selected = (k == r['sel_k'])
 
                if is_oracle and is_selected:
                    color, linewidth, alpha, linestyle, label = 'purple', 3.0, 0, '-', f'Oracle == Selected (k={k})'
                elif is_oracle:
                    color, linewidth, alpha, linestyle, label = 'green', 3.0, 0, '-', f'Oracle (k={k})'
                elif is_selected:
                    color, linewidth, alpha, linestyle, label = 'red', 3.0, 0, '-', f'Selected (k={k})'
                else:
                    color, linewidth, alpha, linestyle, label = colors[k % len(colors)], 1.5, 0, '--', f'Hyp {k}'
 
                ax.plot(pred_lon, pred_lat, linestyle, linewidth=linewidth,
                        color=color, alpha=alpha, label=label)
                ax.scatter(pred_lon[0], pred_lat[0], c=color, s=40 if is_selected or is_oracle else 30,
                           marker='s', alpha=alpha, zorder=5)
                ax.scatter(pred_lon[-1], pred_lat[-1], c=color, s=60 if is_selected or is_oracle else 40,
                           marker='^', alpha=alpha, zorder=5)
 
            # Convert GT from relative to lat/lon
            gt_batch = gt[np.newaxis, :, :].astype(np.float32)
            gt_ll = xy_to_ll(
                torch.from_numpy(gt_batch),
                start_abs.reshape(1, -1) if isinstance(start_abs, np.ndarray) else np.array([start_abs]),
                np.array([start_heading]) if isinstance(start_heading, (int, float)) else start_heading.reshape(1),
                ref,
                self.geodesic
            ).numpy()[0]
 
            gt_lon = gt_ll[:, 1]
            gt_lat = gt_ll[:, 0]
 
            ax.plot(gt_lon, gt_lat, 'k-', linewidth=3, alpha=0.9, label='GT')
            ax.scatter(gt_lon[0], gt_lat[0], c='k', s=60, marker='s', zorder=5)
            ax.scatter(gt_lon[-1], gt_lat[-1], c='k', s=80, marker='*', zorder=5)
 
            # Title (z_var / valid removed -- dataset is already filtered)
            tmn = self.mode_names[r['true_mode']] if r['true_mode'] < len(self.mode_names) else f"mode_{r['true_mode']}"
            pmn = self.mode_names[r['pred_mode']] if r['pred_mode'] < len(self.mode_names) else f"mode_{r['pred_mode']}"
            mode_ok = '✓' if r['true_mode'] == r['pred_mode'] else '✗'
 
            feas_vec = r.get('turn_feasibility', None)
            if feas_vec is not None:
                # order matches self.mode_names: TurnLeft, TurnRight, Straight, Hold
                feas_str = "  ".join(
                    f"{self.mode_names[i]}={'Y' if feas_vec[i] else 'N'}"
                    for i in range(min(len(feas_vec), len(self.mode_names))))
            else:
                feas_str = "N/A"
 
            score_line = ""
            nsc = r.get('nn_score', None)
            if nsc is not None:
                score_line = (f"\nretr: dist={nsc['mean_dist']} support={nsc['support']} "
                              f"pick={nsc['pick']} knnkey={nsc['knn_keydist']}")
 
            ax.set_title(
                f"High-ADE Sample #{rank + 1}  |  selected ADE={r['sel_ade']:.2f}  oracle ADE={r['oracle_ade']:.2f}\n"
                f"true={tmn}  pred={pmn}  ({mode_ok})  "
                f"selected k={r['sel_k']}  oracle k={r['oracle_k']}\n"
                f"feasible: {feas_str}{score_line}",
                fontsize=11
            )
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
 
            # Set axis limits
            all_lon = list(gt_lon) + all_pred_lon
            all_lat = list(gt_lat) + all_pred_lat
            if history is not None and len(history) > 0:
                all_lon.extend(hist_lon)
                all_lat.extend(hist_lat)
 
            if all_lon and all_lat:
                lon_min, lon_max = min(all_lon), max(all_lon)
                lat_min, lat_max = min(all_lat), max(all_lat)
                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                pad_lon = max(lon_range, 0.0005) * 0.15
                pad_lat = max(lat_range, 0.0005) * 0.15
 
                x_lo, x_hi = lon_min - pad_lon, lon_max + pad_lon
                y_lo, y_hi = lat_min - pad_lat, lat_max + pad_lat
 
                if extent is not None:
                    north, east, south, west = extent
                    x_lo = max(x_lo, west)
                    x_hi = min(x_hi, east)
                    y_lo = max(y_lo, south)
                    y_hi = min(y_hi, north)
 
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
 
            # ========== Speed-vs-time panel (right side) ==========
            # History (dashed grey) + GT future (solid black) + selected
            # candidate's future (red) + oracle candidate's future (green),
            # all on the same time axis (t=0 is "now", the last history
            # step). This is for checking whether the historical speed
            # trend actually carries over into the future -- if history and
            # GT-future speeds diverge sharply, a rule that extrapolates
            # from historical speed alone cannot work for that sample.
            if len(speed_hist) > 0:
                ax2.plot(t_hist, speed_hist, color='dimgray', linestyle='--',
                         linewidth=2, alpha=0.8, label='History speed')
            ax2.plot(t_fut, speed_gt, color='black', linewidth=3,
                     alpha=0.9, label='GT future speed')
            ax2.plot(t_fut, speed_sel, color='red', linewidth=2.5,
                     alpha=0.9, label=f'Selected (k={sel_k_i}) speed')
            ax2.plot(t_fut, speed_oracle, color='green', linewidth=2.5,
                     alpha=0.9, linestyle='-.', label=f'Oracle (k={oracle_k_i}) speed')
            ax2.axvline(0, color='blue', linewidth=1.0, alpha=0.6, linestyle=':')
            ax2.set_xlabel('Time step (0 = now / end of history)')
            ax2.set_ylabel('Speed (rel units / step, 1 Hz)')
            ax2.set_title('Speed over time: history vs future (GT / selected / oracle)',
                          fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)
 
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"highade_{rank:02d}.png"), dpi=150)
            plt.close(fig)
 
        print(f"[plot] saved {len(cands)} high-ADE plots to {out_dir}")

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        mode_probs, traj_mu, traj_sigma, traj_score = self.forward(batch)

        Y = batch['scene_dict']['rel_sequences'][..., :4]
        Y = Y[..., G.REL_XYZ[:4]]
        fut_rel = Y[:, :, self.hist_len:, :]
        Y_mode = batch['scene_dict'].get('rule_based_encoding')
        true_mode_idx = self._encode_rule_based_to_mode_index(Y_mode)

        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']

        ego_probs = separate_ego_agent(mode_probs, ego_agent)
        ego_true_mode = separate_ego_agent(true_mode_idx, ego_agent)
        ego_mu = separate_ego_agent(traj_mu, ego_agent)  # (B, 1, T_total, M, K, D)
        ego_sigma = separate_ego_agent(traj_sigma, ego_agent)
        ego_fut = separate_ego_agent(fut_rel, ego_agent)
        ego_mask = separate_ego_agent(masks, ego_agent)
        mask_agent_level = ego_mask.any(dim=-1).float()

        # -----------------------------------------------------------------------
        # Off-road guided hypothesis selection
        # -----------------------------------------------------------------------
        prefix = "test"
        selected_k_idx = None

        sel_mode = getattr(self.eparams, 'selection_mode', 'score')
        if sel_mode == 'score' and traj_score is not None:
            ego_score = separate_ego_agent(traj_score, ego_agent)  # (B,1,T,M,K)
            # Mask-weighted aggregation, matching how the scorer was TRAINED
            # (see _score_step in eval_two_stage.py). A plain .mean() over the
            # horizon averages in the scores of invalid/padded timesteps,
            # which the scorer never saw during training -- the selection at
            # test time would then be made on a quantity that is not the one
            # the head was optimised to produce.
            fut_mask = ego_mask[:, 0, self.hist_len:].float()      # (B,Tp)
            m_exp = fut_mask[:, :, None, None]                     # (B,Tp,1,1)
            denom = m_exp.sum(dim=1).clamp_min(1)                  # (B,1,1)
            s_pt = ego_score[:, 0, self.hist_len:]                 # (B,Tp,M,K)
            score_fut = (s_pt * m_exp).sum(dim=1) / denom          # (B,M,K)
            selected_k_idx = score_fut.argmax(-1)  # (B,M)
            B_, _, T_, M_, K_, D_ = ego_mu.shape
            idx = selected_k_idx[:, None, None, :, None, None].expand(
                B_, 1, T_, M_, 1, D_)
            selected_ego_mu = ego_mu.gather(4, idx).squeeze(4)  # (B,1,T,M,D)

            if self.num_hypotheses > 1:
                for m in range(self.num_modes):
                    self.log(f"{prefix}/selected_hypothesis/{TURN_MODES_NAMES[m]}",
                             selected_k_idx[:, m].float().mean(),
                             on_step=False, on_epoch=True)
        elif sel_mode == 'rule_distance':
            # ------------------------------------------------------------
            # Rule 4: extrapolate how far the aircraft should travel from
            # its recent historical speed, then pick whichever candidate's
            # ARC LENGTH (cumulative path length travelled over the future
            # segment) is closest to that extrapolated distance.
            #
            # Arc length, not endpoint displacement: speed * time is a
            # path-length quantity (how much distance was/will be covered),
            # not a straight-line displacement. For a straight candidate the
            # two coincide, but for a turning or looping candidate they can
            # differ by an order of magnitude -- a large loop can end up
            # close to where it started (small endpoint displacement) while
            # still covering a long path (large arc length). Comparing the
            # extrapolated distance against endpoint displacement would
            # then wrongly reject a loop candidate even when its actual
            # arc length matches the extrapolated travel distance well.
            # This directly targets the large-turn/loop failure cases seen
            # in the high-ADE plots.
            #
            # Speed is estimated from raw position differences over the
            # last few HISTORY frames (not a stored "speed" column, to
            # avoid depending on an unverified feature index), at 1 Hz, so
            # units/frame == units/second.
            # ------------------------------------------------------------
            ego_hist_xy = separate_ego_agent(
                Y[:, :, :self.hist_len, :2], ego_agent)          # (B,1,hist_len,2)
            hist_xy = ego_hist_xy[:, 0, :, :]                    # (B,hist_len,2)
            step_disp = (hist_xy[:, 1:, :] - hist_xy[:, :-1, :]).norm(dim=-1)  # (B,hist_len-1)

            n_recent = min(5, step_disp.shape[1]) if step_disp.shape[1] > 0 else 0
            if n_recent > 0:
                recent_speed = step_disp[:, -n_recent:].mean(dim=1)  # (B,) units/step (1 Hz)
            else:
                recent_speed = torch.zeros(step_disp.shape[0], device=step_disp.device)
            extrapolated_dist = recent_speed * self.max_pred_len    # (B,)

            # Arc length of each candidate's future segment, including the
            # first step from "now" (origin, since positions are rel to the
            # last history step) to the first predicted point.
            fut_all = ego_mu[:, 0, self.hist_len:, :, :, :2]        # (B,Tp,M,K,2)
            origin = torch.zeros_like(fut_all[:, :1, :, :, :])      # (B,1,M,K,2)
            fut_with_origin = torch.cat([origin, fut_all], dim=1)   # (B,Tp+1,M,K,2)
            step_len = (fut_with_origin[:, 1:, :, :, :]
                       - fut_with_origin[:, :-1, :, :, :]).norm(dim=-1)  # (B,Tp,M,K)
            path_length = step_len.sum(dim=1)                        # (B,M,K)

            selected_k_idx = (
                path_length - extrapolated_dist[:, None, None]
            ).abs().argmin(dim=-1)                                  # (B,M)

            B_, _, T_, M_, K_, D_ = ego_mu.shape
            idx = selected_k_idx[:, None, None, :, None, None].expand(
                B_, 1, T_, M_, 1, D_)
            selected_ego_mu = ego_mu.gather(4, idx).squeeze(4)      # (B,1,T,M,D)

            if self.num_hypotheses > 1:
                for m in range(self.num_modes):
                    self.log(f"{prefix}/selected_hypothesis/{TURN_MODES_NAMES[m]}",
                             selected_k_idx[:, m].float().mean(),
                             on_step=False, on_epoch=True)
        elif sel_mode == 'rule_speed_trend':
            # ------------------------------------------------------------
            # Rule 5: match ACCELERATION, not distance and not raw speed
            # level.
            #
            # Rule 4 (rule_distance) compresses the whole future speed
            # profile into one scalar (total arc length) and matches it
            # against one scalar extrapolated from history (avg speed x
            # full horizon). That structurally cannot capture a mid-horizon
            # speed change (e.g. accelerating onto a runway 20-30s in): the
            # historical speed carries no information about an event that
            # hasn't started yet, so extrapolating it over the full horizon
            # just injects noise into the far future.
            #
            # An earlier version of this rule also compared mean SPEED
            # LEVEL (history window average vs future window average), but
            # that is flawed for exactly the case this rule targets: if the
            # aircraft is genuinely accelerating, the future window's mean
            # speed is naturally higher than the history window's mean --
            # comparing levels would wrongly penalise the very candidate
            # that correctly continues the acceleration. Acceleration
            # (net speed change over a window) captures the trend directly
            # and doesn't have this problem: a candidate that continues
            # accelerating at the same rate matches regardless of the
            # absolute speed level.
            #
            # Window: last `hist_window_s` seconds of history vs first
            # `fut_window_s` seconds of the future (confirmed by inspecting
            # the speed-vs-time plots to be the range over which the trend
            # is reliably continuous; matching over the full horizon is not
            # justified, see Rule 4's docstring above).
            # ------------------------------------------------------------
            hist_window_s = int(getattr(self.eparams, 'rule5_hist_window_s', 5))
            fut_window_s = int(getattr(self.eparams, 'rule5_fut_window_s', 10))

            ego_hist_xy = separate_ego_agent(
                Y[:, :, :self.hist_len, :2], ego_agent)          # (B,1,hist_len,2)
            hist_xy = ego_hist_xy[:, 0, :, :]                    # (B,hist_len,2)
            hist_speed = (hist_xy[:, 1:, :] - hist_xy[:, :-1, :]).norm(dim=-1)  # (B,hist_len-1)

            n_hist = min(hist_window_s, hist_speed.shape[1]) if hist_speed.shape[1] > 0 else 0
            if n_hist > 1:
                v_hist_win = hist_speed[:, -n_hist:]                  # (B, n_hist)
                accel_hist = (v_hist_win[:, -1] - v_hist_win[:, 0]) / (n_hist - 1)  # (B,)
            else:
                accel_hist = torch.zeros(hist_xy.shape[0], device=hist_xy.device)

            # candidate future speed, per timestep (kept as a series so we
            # can take a windowed slice near "now")
            fut_all = ego_mu[:, 0, self.hist_len:, :, :, :2]        # (B,Tp,M,K,2)
            origin = torch.zeros_like(fut_all[:, :1, :, :, :])      # (B,1,M,K,2)
            fut_with_origin = torch.cat([origin, fut_all], dim=1)   # (B,Tp+1,M,K,2)
            cand_speed = (fut_with_origin[:, 1:, :, :, :]
                         - fut_with_origin[:, :-1, :, :, :]).norm(dim=-1)  # (B,Tp,M,K)

            Tp = cand_speed.shape[1]
            n_fut = min(fut_window_s, Tp)
            if n_fut > 1:
                cand_speed_win = cand_speed[:, :n_fut, :, :]         # (B, n_fut, M, K)
                accel_cand = (cand_speed_win[:, -1, :, :]
                             - cand_speed_win[:, 0, :, :]) / (n_fut - 1)  # (B,M,K)
            else:
                accel_cand = torch.zeros(
                    cand_speed.shape[0], self.num_modes,
                    cand_speed.shape[3], device=cand_speed.device)

            accel_err = (accel_cand - accel_hist[:, None, None]).abs()   # (B,M,K)
            selected_k_idx = accel_err.argmin(dim=-1)                    # (B,M)

            B_, _, T_, M_, K_, D_ = ego_mu.shape
            idx = selected_k_idx[:, None, None, :, None, None].expand(
                B_, 1, T_, M_, 1, D_)
            selected_ego_mu = ego_mu.gather(4, idx).squeeze(4)          # (B,1,T,M,D)

            if self.num_hypotheses > 1:
                for m in range(self.num_modes):
                    self.log(f"{prefix}/selected_hypothesis/{TURN_MODES_NAMES[m]}",
                             selected_k_idx[:, m].float().mean(),
                             on_step=False, on_epoch=True)
        elif sel_mode == 'tree':
            # ------------------------------------------------------------
            # GBT hypothesis scorer. Loads the per-mode HistGradientBoosting
            # trees trained by eval_two_stage._stage_tree, builds the SAME
            # 50-dim features (via the shared hyp_features_torch module, so
            # train/test features cannot drift), and picks each mode's
            # candidate by that mode's tree. No GT is used: ego_fut is passed
            # only to satisfy the shared function signature; the returned
            # winner is ignored and a zero tensor is used in its place.
            # ------------------------------------------------------------
            import numpy as np
            from amelia_tf.utils.hyp_features_torch import hyp_features_and_winner

            if not hasattr(self, "_tree_scorer"):
                import joblib
                tree_path = getattr(self.eparams, 'tree_scorer_path', None)
                if tree_path is None:
                    raise ValueError("selection_mode='tree' requires eparams.tree_scorer_path")
                blob = joblib.load(tree_path)
                # per-mode dict {mode_idx: tree} or a single shared tree
                self._tree_scorer = blob
                self._tree_per_mode = isinstance(blob, dict)
                print(f"[tree-select] loaded {'per-mode' if self._tree_per_mode else 'shared'} "
                      f"tree scorer from {tree_path}")

            B_, _, T_, M_, K_, D_ = ego_mu.shape
            hl = self.hist_len

            # sample-level history features (match _stage_tree). Y is already
            # the xyz-reduced rel tensor (see top of test_step), so its last
            # dim is xyz; take the first two (xy) directly -- do NOT re-apply
            # REL_XYZ here, that mask is for the original 5-wide layout.
            ego_hist_rel = separate_ego_agent(Y[:, :, :hl, :2], ego_agent)[:, 0]
            hstep_seq = (ego_hist_rel[:, 1:, :] - ego_hist_rel[:, :-1, :]).norm(dim=-1)
            hist_step = hstep_seq.mean(1)
            raw_seq = batch['scene_dict'].get('sequences', None)
            node_pos = None
            use_position = getattr(self.eparams, 'tree_use_position', False)
            if raw_seq is not None:
                ego_raw = separate_ego_agent(raw_seq[:, :, :hl, :], ego_agent)[:, 0]
                hist_mean_speed = ego_raw[:, :, 0].mean(1)
                if use_position:
                    node_pos = ego_raw[:, -1, [G.SEQ_IDX.x, G.SEQ_IDX.y]]   # (B,2) absolute
            else:
                hist_mean_speed = hist_step

            fut_mask_t = ego_mask[:, 0, hl:].float()                 # (B,Tp)
            fut_mask_b = fut_mask_t[:, None, :]                      # (B,1,Tp)
            dummy_fut = torch.zeros(B_, 1, T_ - hl, 2, device=ego_mu.device)

            selected_k_idx = torch.zeros(B_, M_, dtype=torch.long, device=ego_mu.device)
            for m in range(self.num_modes):
                mu_m = ego_mu[:, :, :, m, :, :]                     # (B,1,T,K,D)
                sig_m = ego_sigma[:, :, :, m, :, :]
                feats, _ = hyp_features_and_winner(
                    mu_m, sig_m, dummy_fut, fut_mask_b,
                    hist_mean_speed, hist_step, m, self.num_modes, hl,
                    use_mask=True, node_pos=node_pos)
                feats_np = feats.detach().cpu().numpy()
                tree = self._tree_scorer[m] if self._tree_per_mode else self._tree_scorer
                pred_k = tree.predict(feats_np)                     # (B,)
                selected_k_idx[:, m] = torch.as_tensor(
                    pred_k, dtype=torch.long, device=ego_mu.device)

            idx = selected_k_idx[:, None, None, :, None, None].expand(
                B_, 1, T_, M_, 1, D_)
            selected_ego_mu = ego_mu.gather(4, idx).squeeze(4)      # (B,1,T,M,D)

            if self.num_hypotheses > 1:
                for m in range(self.num_modes):
                    self.log(f"{prefix}/selected_hypothesis/{TURN_MODES_NAMES[m]}",
                             selected_k_idx[:, m].float().mean(),
                             on_step=False, on_epoch=True)
        elif sel_mode == 'retrieval':
            # ------------------------------------------------------------
            # Non-parametric analog-forecasting selection. For each sample,
            # retrieve the k nearest TRAIN segments by (history-xy + node
            # position) within the predicted mode's sub-database, average
            # their real futures into a reference curve, and pick the
            # candidate whose future is closest (mean L2) to that reference.
            #
            # The key includes position because the variance diagnostic showed
            # future shape is determined by WHERE the aircraft is, not by
            # history alone. No GT of the current sample is used -- the
            # reference comes only from OTHER (train) samples' real futures.
            # ------------------------------------------------------------
            if not hasattr(self, "_retrieval_db"):
                db_path = getattr(self.eparams, 'retrieval_db_path', None)
                if db_path is None:
                    raise ValueError("selection_mode='retrieval' requires eparams.retrieval_db_path")
                db = torch.load(db_path, map_location=ego_mu.device, weights_only=False)
                self._retrieval_db = db
                self._retrieval_topk = int(getattr(self.eparams, 'retrieval_topk', 5))
                print(f"[retrieval] loaded db with modes {list(db.keys())}, topk={self._retrieval_topk}")

            B_, _, T_, M_, K_, D_ = ego_mu.shape
            hl = self.hist_len
            topk = self._retrieval_topk

            # ALL-ABSOLUTE (scheme A). Query = absolute-ish local-xy history
            # trajectory (raw 'sequences' x/y), matching the DB key.
            raw_seq = batch['scene_dict'].get('sequences', None)
            if raw_seq is None:
                raise ValueError("retrieval needs raw 'sequences' for local-xy coords")
            ego_raw = separate_ego_agent(raw_seq[:, :, :, :], ego_agent)[:, 0]      # (B,T,C)
            sx, sy = G.SEQ_IDX.x, G.SEQ_IDX.y
            hist_abs = ego_raw[:, :hl, :][:, :, [sx, sy]]                           # (B,hl,2) local xy
            query = hist_abs.reshape(B_, -1)                                        # (B, hl*2)
            node_abs = ego_raw[:, hl - 1, [sx, sy]]                                 # (B,2) node local xy
            node_head_deg = ego_raw[:, hl - 1, G.SEQ_IDX.Heading]                  # (B,) heading in DEGREES
            node_head = node_head_deg * (torch.pi / 180.0)                         # -> radians (must convert!)

            # rel -> local-xy transform, matching the plotting code (xy_to_ll
            # rotates the rel trajectory by start_heading, then translates by
            # start_abs). Candidates (ego_mu) are in the rel frame: origin at
            # the node, x-axis along the heading. A translation-only shift
            # (the previous bug) left them unrotated, collapsing their apparent
            # motion. Build the rotation once from the node heading.
            cos_h = torch.cos(node_head)                                            # (B,)
            sin_h = torch.sin(node_head)

            def rel_to_local(rel):
                # rel: (B, T, K, 2) in node/heading frame -> (B, T, K, 2) local xy
                rx = rel[..., 0]
                ry = rel[..., 1]
                gx = cos_h[:, None, None] * rx - sin_h[:, None, None] * ry
                gy = sin_h[:, None, None] * rx + cos_h[:, None, None] * ry
                out = torch.stack([gx, gy], dim=-1)
                return out + node_abs[:, None, None, :]

            # one-time self-check: transform the GT's rel future and compare to
            # the GT's raw local-xy future; they must match if the rotation
            # convention is right.
            if getattr(self.eparams, 'retrieval_debug', True) and not hasattr(self, '_ret_xform_checked'):
                gt_rel = separate_ego_agent(Y[:, :, hl:, :2], ego_agent)[:, 0]      # (B,Tp,2) rel
                gt_local_raw = ego_raw[:, hl:, :][:, :, [sx, sy]]                   # (B,Tp,2) local xy
                gt_local_xf = rel_to_local(gt_rel[:, :, None, :])[:, :, 0, :]       # via our transform
                err = (gt_local_xf - gt_local_raw).norm(dim=-1).mean().item()
                print(f"[retrieval-xform-check] mean rel->local error vs raw = {err:.6f} "
                      f"(should be ~0 if rotation convention is correct)")
                self._ret_xform_checked = True

            selected_k_idx = torch.zeros(B_, M_, dtype=torch.long, device=ego_mu.device)
            for m in range(self.num_modes):
                if m not in self._retrieval_db:
                    continue
                dbm = self._retrieval_db[m]
                key_db = dbm['key'].to(ego_mu.device)         # (N, hl*2) local-xy history
                fut_db = dbm['fut'].to(ego_mu.device)         # (N, Tp, 2) local-xy future
                q = query                                     # (B, hl*2) local xy, raw

                d = torch.cdist(q, key_db)                    # (B, N)
                kk = min(topk, key_db.shape[0])
                nn_idx = d.topk(kk, largest=False).indices    # (B, kk)
                nn_fut = fut_db[nn_idx]                       # (B, kk, Tp, 2) local-xy real futures

                # candidates: rel -> local xy (rotate by heading, then translate)
                cand_rel = ego_mu[:, 0, hl:, m, :, :2]        # (B, Tp, K, 2) rel
                cand = rel_to_local(cand_rel)                 # (B, Tp, K, 2) local xy
                Tp = min(cand.shape[1], nn_fut.shape[2])

                # NOT averaged. Each retrieved future is a real precedent; each
                # votes for the candidate closest to it (mean xy L2 over the
                # full 50s), preserving multi-modal path structure. Speed is
                # already contained in the xy trajectory -- comparing full
                # trajectories captures it, no separate speed term needed.
                c = cand[:, :Tp, :, :].permute(0, 2, 1, 3)   # (B,K,Tp,2)
                nf = nn_fut[:, :, :Tp, :]                     # (B,kk,Tp,2)
                dist = (c[:, :, None, :, :] - nf[:, None, :, :, :]).norm(dim=-1).mean(dim=-1)  # (B,K,kk)

                if getattr(self.eparams, 'retrieval_debug', True) and not hasattr(self, '_ret_dbg_done'):
                    c_spd_all = (c[:, :, 1:, :] - c[:, :, :-1, :]).norm(dim=-1).mean(-1)  # (B,K)
                    printed = 0
                    for bb in range(cand.shape[0]):
                        # only print samples whose candidates actually MOVE, and
                        # that show real spread among candidates -- those are the
                        # ones where the pick matters. Low-speed/static samples
                        # (any pick is fine) are skipped so they don't mask a
                        # real scoring problem.
                        #if c_spd_all[bb].max().item() < 0.02:
                        #    continue
                        sup_dbg = torch.exp(-dist[bb] / float(getattr(self.eparams, 'retrieval_tau', 0.05))).sum(-1)
                        nbr_spd = (nf[bb, :, 1:, :] - nf[bb, :, :-1, :]).norm(dim=-1).mean().item()
                        cand_spd = c_spd_all[bb].tolist()
                        knn_d = d[bb].topk(kk, largest=False).values.mean().item()
                        print(f"[retrieval-dbg] mode{m} s{bb}: knn_keydist={knn_d:.4f} "
                              f"nbr_spd={nbr_spd:.4f} cand_spd={[round(x,4) for x in cand_spd]} "
                              f"dist={[round(x,3) for x in dist[bb].mean(-1).tolist()]} "
                              f"support={[round(x,2) for x in sup_dbg.tolist()]} pick={sup_dbg.argmax().item()}")
                        printed += 1
                        if printed >= 5:
                            break
                    #if m == self.num_modes - 1:
                    #    self._ret_dbg_done = False

                tau = float(getattr(self.eparams, 'retrieval_tau', 0.05))
                support = torch.exp(-dist / tau).sum(dim=-1)  # (B,K)
                selected_k_idx[:, m] = support.argmax(dim=-1)

            idx = selected_k_idx[:, None, None, :, None, None].expand(
                B_, 1, T_, M_, 1, D_)
            selected_ego_mu = ego_mu.gather(4, idx).squeeze(4)

            if self.num_hypotheses > 1:
                for m in range(self.num_modes):
                    self.log(f"{prefix}/selected_hypothesis/{TURN_MODES_NAMES[m]}",
                             selected_k_idx[:, m].float().mean(),
                             on_step=False, on_epoch=True)
        else:  # case1: K=1, case2: K=4 and post-hoc selection
            try:
                airport_code = batch['scene_dict']['airport_id'][0]
                evaluator = self._get_off_road_evaluator(airport_code)
                _mode = getattr(self.eparams, 'selection_mode', 'ade_oracle')
                if self._dump_enabled:
                    _mode = 'phase'
                selector = OffRoadSelector(
                    evaluator, safety_margin=1.0,
                    selection_mode=_mode,
                    scorer_path=getattr(self.eparams, 'scorer_path', None),
                )

                selected_ego_mu, selected_k_idx = selector.select_best_hypothesis(
                    batch=batch,
                    ego_mu=ego_mu,
                    ego_probs=ego_probs,
                    hist_len=self.hist_len,
                    ego_sigma=ego_sigma,
                )

                if self.num_hypotheses > 1:
                    for m in range(self.num_modes):
                        mode_name = TURN_MODES_NAMES[m]
                        avg_k = selected_k_idx[:, m].float().mean()
                        self.log(f"{prefix}/selected_hypothesis/{mode_name}",
                                 avg_k, on_step=False, on_epoch=True)

            except Exception as e:
                print(f"Warning: hypothesis selection failed ({e}), falling back to k=0.")
                selected_ego_mu = ego_mu[:, :, :, :, 0, :]

        # Store the full K for plotting (before we overwrite ego_mu)
        ego_mu_full = ego_mu.clone()

        # Selection diagnostics
        if batch_idx == 0 and dataloader_idx == 0:
            self._reset_selection_diag()
        self._update_selection_diag(
            ego_mu_full=ego_mu_full, ego_fut=ego_fut, ego_mask=ego_mask,
            ego_probs=ego_probs, ego_true_mode=ego_true_mode,
            selected_k_idx=selected_k_idx, batch=batch, dataloader_idx=dataloader_idx)

        ego_mu = selected_ego_mu

        # For sigma we also take the best hypothesis
        try:
            idx_exp = selected_k_idx[:, None, None, :, None, None].expand(
                *ego_sigma.shape[:4], 1, ego_sigma.shape[-1])
            ego_sigma = ego_sigma.gather(4, idx_exp).squeeze(4)
        except Exception:
            ego_sigma = ego_sigma[:, :, :, :, 0, :]

        # ---- Per-mode metrics ----
        ego_true_flat = ego_true_mode.squeeze(1).squeeze()
        for mode_idx in VALID_TURN_MODES:
            mode_name = TURN_MODES_NAMES[mode_idx]
            sample_mask = (ego_true_flat == mode_idx)
            if sample_mask.sum() == 0:
                continue
            mu_m = ego_mu[sample_mask]
            sig_m = ego_sigma[sample_mask]
            probs_m = ego_probs[sample_mask]
            fut_m = ego_fut[sample_mask]
            mask_m = ego_mask[sample_mask]
            nll_m = self.compute_nll(mu_m, sig_m, probs_m, fut_m, mask=mask_m)
            self.test_mode_nll[mode_name](nll_m)
            self.log(f"test/mode_nll/{mode_name}",
                     self.test_mode_nll[mode_name], on_step=False, on_epoch=True)
            rmse_m = self.compute_mode_rmse(mu_m, probs_m, fut_m, mask=mask_m)
            self.test_mode_rmse[mode_name](rmse_m)
            self.log(f"test/mode_rmse/{mode_name}",
                     self.test_mode_rmse[mode_name], on_step=False, on_epoch=True)
            for t in self.pred_lens:
                mu_t = mu_m[:, :, :self.hist_len + t]
                fut_t = fut_m[:, :, :t]
                mask_t = mask_m[:, :, :self.hist_len + t]
                t_str = f"t={t}"
                self.test_mode_ade[f"{mode_name}_t={t}"](self.ade(mu_t, fut_t, mask=mask_t))
                self.log(f"test/mode_min_ade/{mode_name}/{t_str}",
                         self.test_mode_ade[f"{mode_name}_t={t}"], on_step=False, on_epoch=True)
                self.test_mode_fde[f"{mode_name}_t={t}"](self.fde(mu_t, fut_t, mask=mask_t))
                self.log(f"test/mode_min_fde/{mode_name}/{t_str}",
                         self.test_mode_fde[f"{mode_name}_t={t}"], on_step=False, on_epoch=True)
                self.test_mode_prob_ade[f"{mode_name}_t={t}"](
                    self.mode_ade(mu_t, probs_m, fut_t, mask=mask_t))
                self.log(f"test/mode_prob_ade/{mode_name}/{t_str}",
                         self.test_mode_prob_ade[f"{mode_name}_t={t}"], on_step=False, on_epoch=True)
                self.test_mode_prob_fde[f"{mode_name}_t={t}"](
                    self.mode_fde(mu_t, probs_m, fut_t, mask=mask_t))
                self.log(f"test/mode_prob_fde/{mode_name}/{t_str}",
                         self.test_mode_prob_fde[f"{mode_name}_t={t}"], on_step=False, on_epoch=True)

        # ---- minADE / minFDE; probADE / probFDE ----
        for t in self.pred_lens:
            mu_t = ego_mu[:, :, :self.hist_len + t]
            fut_t = ego_fut[:, :, :t]
            mask_t = ego_mask[:, :, :self.hist_len + t]
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_min_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.test_min_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"{prefix}/min_ade/{key}", self.test_min_ade[key], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{prefix}/min_fde/{key}", self.test_min_fde[key], on_step=False, on_epoch=True, prog_bar=True)
            mode_ade_t = self.mode_ade(mu_t, ego_probs, fut_t, mask=mask_t)
            mode_fde_t = self.mode_fde(mu_t, ego_probs, fut_t, mask=mask_t)
            self.test_prob_ade[key](mode_ade_t)
            self.test_prob_fde[key](mode_fde_t)
            self.log(f"{prefix}/prob_ade/{key}", self.test_prob_ade[key], on_step=False, on_epoch=True)
            self.log(f"{prefix}/prob_fde/{key}", self.test_prob_fde[key], on_step=False, on_epoch=True)

        # ---- NLL and RMSE ----
        nll = self.compute_nll(ego_mu, ego_sigma, ego_probs, ego_fut, mask=ego_mask)
        self.test_nll(nll)
        self.log(f"{prefix}/nll", self.test_nll, on_step=False, on_epoch=True, prog_bar=True)

        rmse = self.compute_mode_rmse(ego_mu, ego_probs, ego_fut, mask=ego_mask)
        self.test_rmse(rmse)
        self.log(f"{prefix}/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # ---- Modal classification ----
        self.test_modal_accumulator.update(ego_probs, ego_true_mode, mask_agent_level)

        # ---- Diagnostics ----
        self._log_sigma_stats(ego_sigma, prefix)
        self._log_mode_probs_stats(ego_probs, prefix)
        self._log_prediction_quality(ego_mu, ego_sigma, ego_fut, ego_probs, ego_mask, prefix)
        self._analyze_nll_components(ego_mu, ego_sigma, ego_probs, ego_fut, ego_mask, prefix)

        # ---- Off-road detection metrics ----
        off_road_results_list = self._evaluate_off_road(
            batch=batch,
            ego_mu=ego_mu,
            ego_probs=ego_probs,
            hist_len=self.hist_len,
            safety_margin=1.0
        )

        if off_road_results_list is not None:
            for off_road_results in off_road_results_list:
                self.test_off_road_rate(off_road_results['off_road_rate'])
                self.test_off_road_max_dist(off_road_results['max_off_road_distance_m'])
                self.test_off_road_mean_dist(off_road_results['mean_off_road_distance_m'])
                self.test_off_road_critical_rate(off_road_results['critical_off_road_rate'])
                self.test_off_road_episodes(off_road_results['num_off_road_episodes'])
                self.test_off_road_duration(off_road_results['mean_episode_duration_steps'])

            self.log(f"{prefix}/off_road_rate", self.test_off_road_rate, on_step=False, on_epoch=True)
            self.log(f"{prefix}/off_road_max_dist_m", self.test_off_road_max_dist, on_step=False, on_epoch=True)
            self.log(f"{prefix}/off_road_mean_dist_m", self.test_off_road_mean_dist, on_step=False, on_epoch=True)
            self.log(f"{prefix}/off_road_critical_rate", self.test_off_road_critical_rate, on_step=False, on_epoch=True)
            self.log(f"{prefix}/off_road_episodes", self.test_off_road_episodes, on_step=False, on_epoch=True)
            self.log(f"{prefix}/off_road_duration_steps", self.test_off_road_duration, on_step=False, on_epoch=True)

        # ---- Per-mode off-road metrics ----
        if Y_mode is not None:
            ego_true_flat = ego_true_mode.squeeze(1).squeeze()
            for mode_idx in VALID_TURN_MODES:
                mode_name = TURN_MODES_NAMES[mode_idx]
                sample_mask = (ego_true_flat == mode_idx)
                if sample_mask.sum() == 0:
                    continue

                mu_mode = ego_mu[sample_mask]
                probs_mode = ego_probs[sample_mask]
                mode_batch = self._filter_batch(batch, sample_mask)

                mode_off_road_list = self._evaluate_off_road(
                    batch=mode_batch,
                    ego_mu=mu_mode,
                    ego_probs=probs_mode,
                    hist_len=self.hist_len,
                    safety_margin=1.0
                )

                if False:
                    mode_rates = [r['off_road_rate'] for r in mode_off_road_list]
                    mode_max_dists = [r['max_off_road_distance_m'] for r in mode_off_road_list]
                    mode_mean_dists = [r['mean_off_road_distance_m'] for r in mode_off_road_list]
                    mode_critical_rates = [r['critical_off_road_rate'] for r in mode_off_road_list]

                    self.test_mode_off_road_rate[mode_name](np.mean(mode_rates))
                    self.test_mode_off_road_max_dist[mode_name](np.mean(mode_max_dists))
                    self.test_mode_off_road_mean_dist[mode_name](np.mean(mode_mean_dists))
                    self.test_mode_off_road_critical_rate[mode_name](np.mean(mode_critical_rates))

                    self.log(f"test/mode_off_road_rate/{mode_name}",
                             self.test_mode_off_road_rate[mode_name], on_step=False, on_epoch=True)
                    self.log(f"test/mode_off_road_max_dist/{mode_name}",
                             self.test_mode_off_road_max_dist[mode_name], on_step=False, on_epoch=True)
                    self.log(f"test/mode_off_road_mean_dist/{mode_name}",
                             self.test_mode_off_road_mean_dist[mode_name], on_step=False, on_epoch=True)
                    self.log(f"test/mode_off_road_critical_rate/{mode_name}",
                             self.test_mode_off_road_critical_rate[mode_name], on_step=False, on_epoch=True)

        # ---- Plotting ----
        if self.eparams.plot_test and (batch_idx) % 10 == 0:
            out_dir = os.path.join(
                self.eparams.plot_dir,
                f"{date.today()}_{self.eparams.tag}",
                'test_combined')
            os.makedirs(out_dir, exist_ok=True)
            plot_scene_batch(
                self.eparams.asset_dir, batch,
                (mode_probs, traj_mu, traj_sigma),
                self.hist_len, Geodesic.WGS84,
                f"test_batch-{batch_idx}", out_dir,
                self.eparams.propagation)

    def on_test_start(self) -> None:
        device = self.device
        for attr in [
            'test_nll', 'test_rmse',
            'test_sigma_mean', 'test_sigma_min', 'test_sigma_max', 'test_sigma_std',
            'test_max_prob_mean', 'test_entropy_mean',
            'test_nll_max_contrib', 'test_nll_mix_penalty', 'test_multi_mode_contrib_percent',
            'test_off_road_rate', 'test_off_road_max_dist', 'test_off_road_mean_dist',
            'test_off_road_critical_rate', 'test_off_road_episodes', 'test_off_road_duration',
        ]:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))

    def on_test_epoch_end(self) -> None:
        print("\n" + "=" * 70)
        print("TEST EPOCH COMPLETE")
        print("=" * 70)

        # Selection diagnostics summary + high-ADE plots
        try:
            self._finalise_selection_diag()
        except Exception as e:
            print(f"[selection] finalise failed: {e}")

        # Modal classification results
        self.test_modal_accumulator.print_report(
            title="TEST SET - MODE CLASSIFICATION")
        self.test_modal_accumulator.print_confusion_matrix(
            title="TEST SET - CONFUSION MATRIX")

        # Off-road detection metrics summary
        print("\n" + "=" * 70)
        print("OFF-ROAD DETECTION METRICS")
        print("=" * 70)
        print(f"  Off-road rate: {self.test_off_road_rate.compute():.4f}")
        print(f"  Mean off-road distance: {self.test_off_road_mean_dist.compute():.2f} m")
        print(f"  Max off-road distance: {self.test_off_road_max_dist.compute():.2f} m")
        print(f"  Critical off-road rate (>15m): {self.test_off_road_critical_rate.compute():.4f}")
        print(f"  Off-road episodes: {self.test_off_road_episodes.compute():.2f}")
        print(f"  Off-road duration (steps): {self.test_off_road_duration.compute():.2f}")

        print("\n" + "=" * 70)
        print("PER-MODE OFF-ROAD METRICS")
        print("=" * 70)
        for mode_idx in VALID_TURN_MODES:
            mode_name = TURN_MODES_NAMES[mode_idx]
            rate = self.test_mode_off_road_rate[mode_name].compute()
            mean_dist = self.test_mode_off_road_mean_dist[mode_name].compute()
            max_dist = self.test_mode_off_road_max_dist[mode_name].compute()
            critical_rate = self.test_mode_off_road_critical_rate[mode_name].compute()
            print(f"  {mode_name}:")
            print(f"    Off-road rate: {rate:.4f}")
            print(f"    Mean distance: {mean_dist:.2f} m")
            print(f"    Max distance: {max_dist:.2f} m")
            print(f"    Critical rate: {critical_rate:.4f}")

        # Reset accumulators
        self.test_modal_accumulator.reset()

    def configure_optimizers(self):
        return []