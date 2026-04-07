import itertools
import numpy as np
import os
import torch
import torch.nn as nn

from datetime import date
from easydict import EasyDict
from geographiclib.geodesic import Geodesic
from lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Any

from amelia_tf.models.components.common import LayerNorm
from amelia_tf.utils.utils import plot_scene_batch
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.utils import separate_ego_agent

np.printoptions(precision=5, suppress=True)


class TrajPred(LightningModule):
    """ Trajectory Prediction module wrapper based on:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
            net: torch.nn.Module, extra_params: EasyDict
    ):
        """ Initializes the trajectory prediction module.

        Inputs
        ------
            optimizer[torch.optim.Optimizer]: optizimer object.
            scheduler[torch.optim.lr_scheduler]: learning rate scheduler.
            net[torch.nn.Module]: model object.
            extra_params[EasyDict]: dictionary containing all other parameters needed by the module.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute also ensures init
        # params will be stored in ckpt
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net
        self.hist_len = self.net.hist_len
        self.pred_lens = self.net.pred_lens
        self.num_dec_heads = self.net.num_dec_heads

        self.eparams = extra_params
        self.seen_airports = self.eparams.seen_airports
        self.unseen_airports = self.eparams.unseen_airports

        # For averaging loss across batches
        self.train_loss, self.val_loss, self.test_loss = MeanMetric(), MeanMetric(), MeanMetric()
        self.val_loss_cls, self.val_loss_reg = MeanMetric(), MeanMetric()
        self.test_loss_cls, self.test_loss_reg = MeanMetric(), MeanMetric()

        # Metrics for validation and test
        self.mode_acc_val, self.rmse_val, self.nll_val = MeanMetric(), MeanMetric(), MeanMetric()
        self.mode_acc_test, self.rmse_test, self.nll_test = MeanMetric(), MeanMetric(), MeanMetric()

        # For tracking best so far validation and testing accuracy
        self.max_pred_len = max(self.pred_lens)
        self.val_ade, self.test_ade, self.val_fde, self.test_fde = {}, {}, {}, {}
        self.val_prob_ade, self.test_prob_ade, self.val_prob_fde, self.test_prob_fde = {}, {}, {}, {}
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key], self.test_ade[key] = MeanMetric(), MeanMetric()
            self.val_fde[key], self.test_fde[key] = MeanMetric(), MeanMetric()
            self.val_prob_ade[key], self.test_prob_ade[key] = MeanMetric(), MeanMetric()
            self.val_prob_fde[key], self.test_prob_fde[key] = MeanMetric(), MeanMetric()
        self.val_ade, self.test_ade = nn.ModuleDict(self.val_ade), nn.ModuleDict(self.test_ade)
        self.val_fde, self.test_fde = nn.ModuleDict(self.val_fde), nn.ModuleDict(self.test_fde)
        self.val_prob_ade, self.test_prob_ade = nn.ModuleDict(self.val_prob_ade), nn.ModuleDict(self.test_prob_ade)
        self.val_prob_fde, self.test_prob_fde = nn.ModuleDict(self.val_prob_fde), nn.ModuleDict(self.test_prob_fde)

        self.val_seen_ade, self.test_seen_ade = {}, {}
        self.val_seen_fde, self.test_seen_fde = {}, {}
        self.val_seen_prob_ade, self.test_seen_prob_ade = {}, {}
        self.val_seen_prob_fde, self.test_seen_prob_fde = {}, {}
        for pred_len, airport in itertools.product(self.pred_lens, self.seen_airports):
            key = f"{airport}_t={pred_len}"
            self.val_seen_ade[key], self.test_seen_ade[key] = MeanMetric(), MeanMetric()
            self.val_seen_fde[key], self.test_seen_fde[key] = MeanMetric(), MeanMetric()
            self.val_seen_prob_ade[key], self.test_seen_prob_ade[key] = MeanMetric(), MeanMetric()
            self.val_seen_prob_fde[key], self.test_seen_prob_fde[key] = MeanMetric(), MeanMetric()
        self.val_seen_ade = nn.ModuleDict(self.val_seen_ade)
        self.val_seen_fde = nn.ModuleDict(self.val_seen_fde)
        self.test_seen_ade = nn.ModuleDict(self.test_seen_ade)
        self.test_seen_fde = nn.ModuleDict(self.test_seen_fde)
        self.val_seen_prob_ade = nn.ModuleDict(self.val_seen_prob_ade)
        self.val_seen_prob_fde = nn.ModuleDict(self.val_seen_prob_fde)
        self.test_seen_prob_ade = nn.ModuleDict(self.test_seen_prob_ade)
        self.test_seen_prob_fde = nn.ModuleDict(self.test_seen_prob_fde)

        # Create metrics for unseen airports
        if len(self.unseen_airports) > 0:
            self.test_unseen_ade, self.test_unseen_fde = {}, {}
            self.test_unseen_prob_ade, self.test_unseen_prob_fde = {}, {}
            for pred_len, airport in itertools.product(self.pred_lens, self.unseen_airports):
                key = f"{airport}_t={pred_len}"
                self.test_unseen_ade[key], self.test_unseen_fde[key] = MeanMetric(), MeanMetric()
                self.test_unseen_prob_ade[key], self.test_unseen_prob_fde[key] = MeanMetric(), MeanMetric()
            self.test_unseen_ade = nn.ModuleDict(self.test_unseen_ade)
            self.test_unseen_fde = nn.ModuleDict(self.test_unseen_fde)
            self.test_unseen_prob_ade = nn.ModuleDict(self.test_unseen_prob_ade)
            self.test_unseen_prob_fde = nn.ModuleDict(self.test_unseen_prob_fde)

        assert self.eparams.propagation in ['joint', 'marginal']
        if self.eparams.propagation == 'marginal':
            from amelia_tf.utils.metrics import compute_mode_accuracy, compute_mode_rmse, compute_nll
            from amelia_tf.utils.metrics import mode_ade, mode_fde
            from amelia_tf.utils.metrics import ModalClassificationMetrics
            from amelia_tf.utils.metrics import marginal_ade as ade
            from amelia_tf.utils.metrics import marginal_fde as fde
            from amelia_tf.utils.metrics import marginal_prob_ade as prob_ade
            from amelia_tf.utils.metrics import marginal_prob_fde as prob_fde
            from amelia_tf.utils.losses import acceleration_marginal_loss as compute_loss
        else:
            from amelia_tf.utils.metrics import joint_ade as ade
            from amelia_tf.utils.metrics import joint_fde as fde
            from amelia_tf.utils.metrics import joint_prob_ade as prob_ade
            from amelia_tf.utils.metrics import joint_prob_fde as prob_fde
            from amelia_tf.utils.losses import lmbd_marginal_joint_loss as compute_loss

        self.val_modal_accumulator = ModalClassificationMetrics(self.num_dec_heads)
        self.test_modal_accumulator = ModalClassificationMetrics(self.num_dec_heads)
        self.compute_mode_accuracy, self.compute_mode_rmse, self.compute_nll = compute_mode_accuracy, compute_mode_rmse, compute_nll
        self.mode_ade, self.mode_fde = mode_ade, mode_fde
        self.ade, self.fde, self.prob_ade, self.prob_fde = ade, fde, prob_ade, prob_fde
        self.compute_loss = compute_loss
        self.geodesic = Geodesic.WGS84

        os.makedirs(self.eparams.plot_dir, exist_ok=True)
        out_dir = os.path.join(self.eparams.plot_dir, f"{date.today()}_{self.eparams.tag}")
        self.val_out_dir = os.path.join(out_dir, 'val')
        os.makedirs(self.val_out_dir, exist_ok=True)
        self.test_out_dir = os.path.join(out_dir, 'test')
        os.makedirs(self.test_out_dir, exist_ok=True)

    def log_sigma_stats(self, sigma: torch.Tensor, prefix: str = "test"):
        """
        Log statistics of predicted sigma values.

        Args:
            sigma: [B, A, T_total, M, D] predicted standard deviations
            prefix: logging prefix (e.g., "test")
        """
        sigma_flat = sigma[~torch.isnan(sigma)]
        if sigma_flat.numel() > 0:
            self.log(f"{prefix}/sigma_mean", sigma_flat.mean(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_min", sigma_flat.min(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_max", sigma_flat.max(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_std", sigma_flat.std(), on_step=False, on_epoch=True)

            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                val = torch.quantile(sigma_flat, p / 100.0)
                self.log(f"{prefix}/sigma_p{p}", val, on_step=False, on_epoch=True)

    def log_mode_probs_stats(self, mode_probs: torch.Tensor, prefix: str = "test"):
        """
        Log statistics of mode probability distributions.

        Args:
            mode_probs: [B, A, M] predicted mode probabilities
            prefix: logging prefix (e.g., "test")
        """
        B, A, M = mode_probs.shape

        for k in range(M):
            self.log(f"{prefix}/mode_{k}_prob", mode_probs[..., k].mean(), on_step=False, on_epoch=True)

        max_probs = mode_probs.max(dim=-1)[0]
        self.log(f"{prefix}/max_prob_mean", max_probs.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/max_prob_min", max_probs.min(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/max_prob_std", max_probs.std(), on_step=False, on_epoch=True)

        entropy = -(mode_probs * torch.log(mode_probs.clamp_min(1e-8))).sum(dim=-1)
        self.log(f"{prefix}/entropy_mean", entropy.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/entropy_std", entropy.std(), on_step=False, on_epoch=True)

        mode_counts = mode_probs.argmax(dim=-1).flatten()
        for k in range(M):
            count = (mode_counts == k).sum().float()
            percentage = count / mode_counts.numel() * 100
            self.log(f"{prefix}/mode_{k}_percentage", percentage, on_step=False, on_epoch=True)

    def log_prediction_quality(
            self,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            Y: torch.Tensor,
            mode_probs: torch.Tensor,
            mask: torch.Tensor = None,
            prefix: str = "test"
    ):
        """
        Log prediction quality metrics for the best mode.

        Args:
            mu: [B, A, T_total, M, D] predicted means
            sigma: [B, A, T_total, M, D] predicted standard deviations
            Y: [B, A, T_pred, D] ground truth future trajectories
            mode_probs: [B, A, M] predicted mode probabilities
            mask: [B, A, T_total] agent validity mask
            prefix: logging prefix (e.g., "test")
        """
        B, A, T_total, M, D = mu.shape
        _, _, T_pred, _ = Y.shape

        mu_future = mu[:, :, -T_pred:, :, :]
        sigma_future = sigma[:, :, -T_pred:, :, :]

        best_mode_idx = mode_probs.argmax(dim=-1)

        best_mu = mu_future[torch.arange(B)[:, None, None],
                  torch.arange(A)[None, :, None],
                  torch.arange(T_pred)[None, None, :],
                  best_mode_idx[:, :, None],
                  :]

        best_sigma = sigma_future[torch.arange(B)[:, None, None],
                     torch.arange(A)[None, :, None],
                     torch.arange(T_pred)[None, None, :],
                     best_mode_idx[:, :, None],
                     :]

        error = (best_mu - Y).norm(dim=-1)

        if mask is not None:
            mask_pred = mask[:, :, -T_pred:]
            error = error * mask_pred
            ade = error.sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
            last_valid_idx = (mask_pred != 0).cumsum(dim=-1).argmax(dim=-1)
            fde = error[torch.arange(B)[:, None], torch.arange(A)[None, :], last_valid_idx]
        else:
            ade = error.mean(dim=-1)
            fde = error[..., -1]

        self.log(f"{prefix}/best_mode_ade", ade.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_fde", fde.mean(), on_step=False, on_epoch=True)

        # Compute log probability with safe sigma
        safe_sigma = best_sigma.clamp_min(1e-3)
        log_prob = -0.5 * (
                ((best_mu - Y) / safe_sigma) ** 2 +
                torch.log(2 * torch.pi * safe_sigma ** 2)
        )
        log_prob = log_prob.sum(dim=-1)

        if mask is not None:
            log_prob = log_prob * mask_pred
            log_prob_mean = log_prob.sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
        else:
            log_prob_mean = log_prob.mean(dim=-1)

        self.log(f"{prefix}/best_mode_log_prob_mean", log_prob_mean.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_log_prob_min", log_prob_mean.min(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_log_prob_max", log_prob_mean.max(), on_step=False, on_epoch=True)

    def analyze_nll_components(
            self,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            mode_probs: torch.Tensor,
            Y: torch.Tensor,
            mask: torch.Tensor = None,
            prefix: str = "test"
    ):
        """
        Decompose NLL into components to understand what drives the value.

        Args:
            mu: [B, A, T_total, M, D] predicted means
            sigma: [B, A, T_total, M, D] predicted standard deviations
            mode_probs: [B, A, M] predicted mode probabilities
            Y: [B, A, T_pred, D] ground truth future trajectories
            mask: [B, A, T_total] agent validity mask
            prefix: logging prefix (e.g., "test")
        """
        B, A, T_total, M, D = mu.shape
        _, _, T_pred, _ = Y.shape

        mu_future = mu[:, :, -T_pred:, :, :]
        sigma_future = sigma[:, :, -T_pred:, :, :]

        y_expanded = Y.unsqueeze(-2)

        logp_mode = -0.5 * (
                torch.log(2 * torch.pi * sigma_future.clamp_min(1e-3) ** 2) +
                ((y_expanded - mu_future) / sigma_future.clamp_min(1e-3)) ** 2
        )
        logp_mode = logp_mode.sum(dim=-1)

        log_pi = torch.log(mode_probs.clamp_min(1e-8)).unsqueeze(2)
        logp_mix = logp_mode + log_pi

        max_logp_mode = logp_mix.max(dim=-1)[0]
        log_sum_exp = torch.logsumexp(logp_mix, dim=-1)

        mixture_penalty = log_sum_exp - max_logp_mode

        if mask is not None:
            mask_pred = mask[:, :, -T_pred:]

            max_logp_mean = (max_logp_mode * mask_pred).sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
            log_sum_exp_mean = (log_sum_exp * mask_pred).sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
            mixture_penalty_mean = (mixture_penalty * mask_pred).sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
        else:
            max_logp_mean = max_logp_mode.mean(dim=-1)
            log_sum_exp_mean = log_sum_exp.mean(dim=-1)
            mixture_penalty_mean = mixture_penalty.mean(dim=-1)

        self.log(f"{prefix}/nll_max_mode_contribution", (-max_logp_mean).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_log_sum_exp_contribution", (-log_sum_exp_mean).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty", mixture_penalty_mean.mean(), on_step=False, on_epoch=True)

        self.log(f"{prefix}/nll_mixture_penalty_max", mixture_penalty.max(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty_min", mixture_penalty.min(), on_step=False, on_epoch=True)

        significant_contribution = (mixture_penalty > 0.1).float()
        if mask is not None:
            significant_contribution = significant_contribution * mask_pred
            contrib_percent = significant_contribution.sum() / mask_pred.sum().clamp_min(1) * 100
        else:
            contrib_percent = significant_contribution.mean() * 100

        self.log(f"{prefix}/samples_with_multi_mode_contribution_percent", contrib_percent, on_step=False,
                 on_epoch=True)

    def on_train_start(self):
        """ by default lightning executes validation step sanity checks before training starts, so
        it's worth to make sure validation metrics don't store results from these checks. """
        self.val_loss.reset()

    def model_step(self, batch, plot: bool = False, tag: str = 'temp', out_dir: str = 'temp'):
        """ Runs the model's forward function and then computes the loss function. If plot is True
        it will run and save scene visualizations.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            plot[bool]: if True, it visualizes the scene.
            out_dir[str]: output directory.
            tag[str]: tag name to save the output file.

        Output
        ------
            loss[torch.tensor]: model's loss value.
            pred_scores[torch.tensor]: predictions scores.
            mu[torch.tensor]: predicted means.
            sigma[torch.tensor]: predicted standard deviations.
            Y_out[torch.tensor]: ground truth futures.
        """
        # TODO: roll up ego-agent. TF not viewpoint invariant.
        # (B, N, T, D)
        Y = batch['scene_dict']['rel_sequences']
        X = torch.zeros_like(Y).type(torch.float)
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]
        init_states = self._extract_initial_states(X)
        Y = Y[:, :, :, :4]
        X = X[:, :, :, :4]

        # -----------------------------------------
        # TODO: incorporate heading prediction
        B, N, T, D = Y.shape
        Y = Y[..., G.REL_XYZ[:D]]
        context = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']

        # TODO: address attention-based masking
        pred_scores, accel_mu, accel_sigma = self.net(
            X, context=context, adjacency=adjacency,
            mask=None,
        )

        traj_mu, traj_sigma = self._acceleration_to_trajectory(init_states, accel_mu, accel_sigma)

        loss, loss_cls, loss_reg = self.compute_loss(
            pred_scores, traj_mu, traj_sigma, Y, ego_agent=ego_agent, epoch=self.current_epoch + 1,
            agent_mask=batch['scene_dict']['agent_masks'],
        )

        if plot:
            print("plot")
            predictions = (pred_scores, traj_mu, traj_sigma)
            plot_scene_batch(
                self.eparams.asset_dir, batch, predictions, self.hist_len, self.geodesic, tag,
                out_dir, self.eparams.propagation
            )

        return loss, loss_cls, loss_reg, pred_scores, traj_mu, traj_sigma, Y[:, :, self.hist_len:, :]

    def _extract_initial_states(self, x: torch.tensor, mask: torch.tensor = None):
        """
        Extract initial states [position, velocity] from input trajectories.

        Args:
            x (torch.Tensor): Input trajectories of shape [B, A, T, D].
                B: batch size
                A: number of agents
                T: time steps
                D: feature dimension (e.g., x, y, ...)
            mask (torch.Tensor, optional): Not used in this version.

        Returns:
            torch.Tensor: Initial states [B, A, 6] = [x, y, z, vx, vy, vz].
        """
        B, A, T, D = x.shape

        # Last observed position
        last_pos = x[:, :, self.hist_len - 1, :D - 4]  # [B, A, 3]

        # Last observed velocity
        last_vel = x[:, :, self.hist_len - 1, D - 3:D]  # [B, A, 3]

        # Combine position and velocity into initial state
        initial_states = torch.cat([last_pos, last_vel], dim=-1)  # [B, A, 6]

        return initial_states

    def _acceleration_to_trajectory(self, initial_states: torch.Tensor,
                                    accel_mu: torch.Tensor,
                                    accel_sigma: torch.Tensor,
                                    dt: float = 1.0):
        """
        Vectorized version that follows your original simplified logic:
          - semi-implicit integration for traj_mu (pos uses v_t + 0.5*a_t)
          - traj_sigma at each timestep = accel_sigma / 2 (no accumulation)

        Args:
            initial_states: [B, A, 6]  -> [x0, y0, vx0, vy0]
            accel_mu:        [B, A, T, H, 3]
            accel_sigma:     [B, A, T, H, 3]
            dt:              scalar time step (kept for potential scaling)

        Returns:
            traj_mu:    [B, A, T, H, 3]
            traj_sigma: [B, A, T, H, 3]
        """
        B, A, T, H, D = accel_mu.shape
        v0 = initial_states[:, :, None, None, 3:].expand(B, A, 1, H, D)  # [B, A, 1, 1, 3]
        x0 = initial_states[:, :, None, None, :3].expand(B, A, 1, H, D)  # [B, A, 1, 1, 3]

        vel_mu = torch.cat([
            v0,
            v0 + torch.cumsum(accel_mu * dt, dim=2)[:, :, :-1]
        ], dim=2)  # [B, A, T, H, 3]

        traj_increment = vel_mu + 0.5 * (accel_mu * dt)

        traj_mu = x0 + torch.cumsum(traj_increment * dt, dim=2)

        # accel_sigma: [B, A, T, H, 3]
        traj_sigma = accel_sigma ** 2 / 2
        # traj_sigma = torch.clamp(traj_sigma, min=0.001, max=1.0)

        return traj_mu, traj_sigma

    def training_step(self, batch: Any, batch_idx: int):
        """ Performs a model step on a training batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.

        Output
        ------
            loss[torch.tensor]: model's loss value.
        """
        loss, loss_cls, loss_reg, _, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("losses/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        """Reset accumulator at the start of validation epoch"""
        self.val_modal_accumulator.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        """ Performs a model step on a validation batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.
        """
        plot = self.eparams.plot_val \
            if self.current_epoch >= self.eparams.plot_after_n_epochs \
               and (batch_idx + 1) % self.eparams.plot_every_n == 0 else False

        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, loss_cls, loss_reg, pred_scores, mu, sigma, fut_rel = self.model_step(batch, plot, tag, self.val_out_dir)

        # Separate ego agent prediction
        if self.eparams.propagation == 'marginal':
            ego_agent = batch['scene_dict']['ego_agent_id']
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            mask = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError

        self.val_loss(loss)
        self.val_loss_cls(loss_cls)
        self.val_loss_reg(loss_reg)
        self.log("losses/val", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_cls/val", self.val_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_reg/val", self.val_loss_reg, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t = ego_mu[:, :, :self.hist_len + t]
            mask_t = mask[:, :, :self.hist_len + t]
            fut_t = ego_fut[:, :, :t]

            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"val_ade/{key}", self.val_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.val_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"val_fde/{key}", self.val_fde[key], on_step=False, on_epoch=True, prog_bar=True)

            # Probability-max ADE and FDE per prediction length
            mode_ade_t = self.mode_ade(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            mode_fde_t = self.mode_fde(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            self.val_prob_ade[key](mode_ade_t)
            self.val_prob_fde[key](mode_fde_t)
            self.log(f"val_prob_ade/{key}", self.val_prob_ade[key], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"val_prob_fde/{key}", self.val_prob_fde[key], on_step=False, on_epoch=True, prog_bar=False)

        self.val_modal_accumulator.update(ego_mu, ego_pred_scores, ego_fut, mask)
        mode_acc = self.compute_mode_accuracy(ego_mu, ego_pred_scores, ego_fut, mask)
        self.mode_acc_val(mode_acc)
        self.log("val/mode_acc", self.mode_acc_val, on_step=False, on_epoch=True, prog_bar=True)

        # Probability-max RMSE
        rmse_val = self.compute_mode_rmse(ego_mu, ego_pred_scores, ego_fut, mask=mask)
        self.rmse_val(rmse_val)
        self.log("val/prob_rmse", self.rmse_val, on_step=False, on_epoch=True, prog_bar=True)

        # NLL
        nll_val = self.compute_nll(ego_mu, ego_sigma, ego_pred_scores, ego_fut, mask=mask)
        self.nll_val(nll_val)
        self.log("val/nll", self.nll_val, on_step=False, on_epoch=True, prog_bar=True)

        if len(self.seen_airports) > 1:
            airport_ids = batch['scene_dict']['airport_id']
            for airport in self.seen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]
                airport_pred_scores = ego_pred_scores[airport_idx]

                for t in self.pred_lens:
                    mu_t = airport_mu[:, :, :self.hist_len + t]
                    fut_t = airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len + t]

                    key = f"{airport}_t={t}"
                    self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"val_seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True,
                        prog_bar=True)

                    self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"val_seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True,
                        prog_bar=True)

                    # Probability-max ADE and FDE per airport
                    mode_ade_t = self.mode_ade(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    mode_fde_t = self.mode_fde(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    self.val_seen_prob_ade[key](mode_ade_t)
                    self.val_seen_prob_fde[key](mode_fde_t)
                    self.log(f"val_seen_prob_ade/{key}", self.val_seen_prob_ade[key], on_step=False, on_epoch=True,
                             prog_bar=False)
                    self.log(f"val_seen_prob_fde/{key}", self.val_seen_prob_fde[key], on_step=False, on_epoch=True,
                             prog_bar=False)

    def on_validation_epoch_end(self):
        """Compute final metrics at the end of validation epoch"""
        modal_metrics = self.val_modal_accumulator.compute()

        self.log("val/modal_accuracy", modal_metrics['accuracy'],
                 on_epoch=True, prog_bar=True)
        self.log("val/modal_macro_f1", modal_metrics['macro_f1'],
                 on_epoch=True, prog_bar=True)
        self.log("val/modal_weighted_f1", modal_metrics['weighted_f1'],
                 on_epoch=True, prog_bar=True)

        num_modes = len(modal_metrics['precision_per_mode'])
        for mode_idx in range(num_modes):
            self.log(f"val/modal_precision/mode_{mode_idx}",
                     modal_metrics['precision_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

            self.log(f"val/modal_recall/mode_{mode_idx}",
                     modal_metrics['recall_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

            self.log(f"val/modal_f1/mode_{mode_idx}",
                     modal_metrics['f1_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

            self.log(f"val/modal_support/mode_{mode_idx}",
                     modal_metrics['support_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

        if self.current_epoch % 5 == 0:
            print("\n" + "=" * 100)
            print(f"VALIDATION EPOCH {self.current_epoch} - MODAL CLASSIFICATION RESULTS")
            print("=" * 100)
            self.val_modal_accumulator.print_report()
            self.val_modal_accumulator.print_confusion_matrix()

        self.val_modal_accumulator.reset()

    def on_test_epoch_start(self):
        """Reset accumulator at the start of test epoch"""
        self.test_modal_accumulator.reset()

    def on_test_epoch_end(self):
        """Compute final metrics at the end of test epoch"""
        modal_metrics = self.test_modal_accumulator.compute()

        self.log("test/modal_accuracy", modal_metrics['accuracy'],
                 on_epoch=True, prog_bar=True)
        self.log("test/modal_macro_f1", modal_metrics['macro_f1'],
                 on_epoch=True, prog_bar=True)
        self.log("test/modal_weighted_f1", modal_metrics['weighted_f1'],
                 on_epoch=True, prog_bar=True)

        num_modes = len(modal_metrics['precision_per_mode'])
        for mode_idx in range(num_modes):
            self.log(f"test/modal_precision/mode_{mode_idx}",
                     modal_metrics['precision_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

            self.log(f"test/modal_recall/mode_{mode_idx}",
                     modal_metrics['recall_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

            self.log(f"test/modal_f1/mode_{mode_idx}",
                     modal_metrics['f1_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

            self.log(f"test/modal_support/mode_{mode_idx}",
                     modal_metrics['support_per_mode'][mode_idx],
                     on_epoch=True, prog_bar=False)

        print("\n" + "=" * 100)
        print("TEST SET MODAL CLASSIFICATION RESULTS")
        print("=" * 100)
        self.test_modal_accumulator.print_report()
        self.test_modal_accumulator.print_confusion_matrix()

        self.test_modal_accumulator.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """ Performs a model step on a test batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.
        """
        plot = self.eparams.plot_test if (batch_idx + 1) % 10 == 0 else False

        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, loss_cls, loss_reg, pred_scores, mu, sigma, fut_rel = self.model_step(batch, plot, tag, self.test_out_dir)
        ego_agent = batch['scene_dict']['ego_agent_id']

        if self.eparams.propagation == 'marginal':
            # Separate ego agent prediction
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            mask = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError

        self.test_loss(loss)
        self.test_loss_cls(loss_cls)
        self.test_loss_reg(loss_reg)
        self.log("losses/test", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_cls/test", self.test_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_reg/test", self.test_loss_reg, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t, fut_t = ego_mu[:, :, :self.hist_len + t], ego_fut[:, :, :t]
            mask_t = mask[:, :, :self.hist_len + t]

            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(
                f"test_ade/{t}", self.test_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.test_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(
                f"test_fde/{t}", self.test_fde[key], on_step=False, on_epoch=True, prog_bar=True)

            # Probability-max ADE and FDE per prediction length
            mode_ade_t = self.mode_ade(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            mode_fde_t = self.mode_fde(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            self.test_prob_ade[key](mode_ade_t)
            self.test_prob_fde[key](mode_fde_t)
            self.log(f"test_prob_ade/{key}", self.test_prob_ade[key], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"test_prob_fde/{key}", self.test_prob_fde[key], on_step=False, on_epoch=True, prog_bar=False)

        self.test_modal_accumulator.update(ego_mu, ego_pred_scores, ego_fut, mask)
        mode_acc = self.compute_mode_accuracy(ego_mu, ego_pred_scores, ego_fut, mask)
        self.mode_acc_test(mode_acc)
        self.log("test/mode_acc", self.mode_acc_test, on_step=False, on_epoch=True, prog_bar=True)

        # Probability-max RMSE
        rmse_test = self.compute_mode_rmse(ego_mu, ego_pred_scores, ego_fut, mask=mask)
        self.rmse_test(rmse_test)
        self.log("test/prob_rmse", self.rmse_test, on_step=False, on_epoch=True, prog_bar=True)

        # NLL
        nll_test = self.compute_nll(ego_mu, ego_sigma, ego_pred_scores, ego_fut, mask=mask)
        self.nll_test(nll_test)
        self.log("test/nll", self.nll_test, on_step=False, on_epoch=True, prog_bar=True)

        # ========== NLL DIAGNOSTICS ==========
        self.log_sigma_stats(ego_sigma, "test")
        self.log_mode_probs_stats(ego_pred_scores, "test")
        self.log_prediction_quality(ego_mu, ego_sigma, ego_fut, ego_pred_scores, mask, "test")
        self.analyze_nll_components(ego_mu, ego_sigma, ego_pred_scores, ego_fut, mask, "test")
        # ========== END DIAGNOSTICS ==========

        airport_ids = batch['scene_dict']['airport_id']
        for airport in self.seen_airports:
            airport_idx = np.where(airport_ids == airport)[0]
            if len(airport_idx) == 0:
                continue
            airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
            airport_mask = mask[airport_idx]
            airport_pred_scores = ego_pred_scores[airport_idx]

            for t in self.pred_lens:
                mu_t, fut_t = airport_mu[:, :, :self.hist_len + t], airport_fut[:, :, :t]
                mask_t = airport_mask[:, :, :self.hist_len + t]

                key = f"{airport}_t={t}"
                self.test_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                self.log(
                    f"test_seen_ade/{key}", self.test_seen_ade[key], on_step=False, on_epoch=True,
                    prog_bar=True)

                self.test_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                self.log(
                    f"test_seen_fde/{key}", self.test_seen_fde[key], on_step=False, on_epoch=True,
                    prog_bar=True)

                # Probability-max ADE and FDE per airport for seen airports
                mode_ade_t = self.mode_ade(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                mode_fde_t = self.mode_fde(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                self.test_seen_prob_ade[key](mode_ade_t)
                self.test_seen_prob_fde[key](mode_fde_t)
                self.log(f"test_seen_prob_ade/{key}", self.test_seen_prob_ade[key], on_step=False, on_epoch=True,
                         prog_bar=False)
                self.log(f"test_seen_prob_fde/{key}", self.test_seen_prob_fde[key], on_step=False, on_epoch=True,
                         prog_bar=False)

        if len(self.unseen_airports) > 0:
            for airport in self.unseen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]
                airport_pred_scores = ego_pred_scores[airport_idx]

                for t in self.pred_lens:
                    mu_t, fut_t = airport_mu[:, :, :self.hist_len + t], airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len + t]

                    key = f"{airport}_t={t}"
                    self.test_unseen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"test_unseen_ade/{key}", self.test_unseen_ade[key], on_step=False,
                        on_epoch=True, prog_bar=True)

                    self.test_unseen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"test_unseen_fde/{key}", self.test_unseen_fde[key], on_step=False,
                        on_epoch=True, prog_bar=True)

                    # Probability-max ADE and FDE per airport for unseen airports
                    mode_ade_t = self.mode_ade(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    mode_fde_t = self.mode_fde(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    self.test_unseen_prob_ade[key](mode_ade_t)
                    self.test_unseen_prob_fde[key](mode_fde_t)
                    self.log(f"test_unseen_prob_ade/{key}", self.test_unseen_prob_ade[key], on_step=False,
                             on_epoch=True, prog_bar=False)
                    self.log(f"test_unseen_prob_fde/{key}", self.test_unseen_prob_fde[key], on_step=False,
                             on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        """ This long function is unfortunately doing something very simple and is being very
        defensive: We are separating out all parameters of the model into two buckets: those that
        will experience weight decay for regularization and those that won't (biases, layernorm,
        embedding weights). We are then returning the PyTorch optimizer object.

        NOTE: For reference as to why this function is needed:
            https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025
            https://discuss.pytorch.org/t/ \
                weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        """
        # separate out all parameters that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.SyncBatchNorm, nn.LayerNorm, LayerNorm, nn.Embedding, nn.BatchNorm1d,
            nn.BatchNorm2d, nn.MultiheadAttention
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.hparams.optimizer.weight_decay},
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2)
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "losses/val",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {
            "optimizer": optimizer
        }


if __name__ == "__main__":
    _ = TrajPred(None, None, None)