import itertools
import numpy as np
import os
import torch
import torch.nn as nn
import traceback

from datetime import date
from easydict import EasyDict
from geographiclib.geodesic import Geodesic
from lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Any

from amelia_tf.models.components.common import LayerNorm
from amelia_tf.utils.utils import plot_scene_batch, separate_ego_agent
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import VALID_TURN_MODES, TURN_MODES_NAMES

np.printoptions(precision=5, suppress=True)


class TrajPred(LightningModule):
    """ Trajectory Prediction module wrapper based on:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
            net: torch.nn.Module, extra_params: EasyDict
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net
        self.hist_len = self.net.hist_len
        self.pred_lens = self.net.pred_lens
        self.num_dec_heads = self.net.num_dec_heads

        self.eparams = extra_params
        self.seen_airports = self.eparams.seen_airports
        self.unseen_airports = self.eparams.unseen_airports

        # Loss metrics
        self.train_loss, self.val_loss = MeanMetric(), MeanMetric()
        self.train_loss_cls, self.val_loss_cls = MeanMetric(), MeanMetric()
        self.train_loss_reg, self.val_loss_reg = MeanMetric(), MeanMetric()

        self.mode_acc_val, self.rmse_val, self.nll_val = MeanMetric(), MeanMetric(), MeanMetric()
        self.mode_acc_test, self.rmse_test, self.nll_test = MeanMetric(), MeanMetric(), MeanMetric()

        self.max_pred_len = max(self.pred_lens)

        # ========== VALIDATION METRICS ==========
        self.val_ade, self.val_fde = {}, {}
        self.val_prob_ade, self.val_prob_fde = {}, {}
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key] = MeanMetric()
            self.val_fde[key] = MeanMetric()
            self.val_prob_ade[key] = MeanMetric()
            self.val_prob_fde[key] = MeanMetric()
        self.val_ade = nn.ModuleDict(self.val_ade)
        self.val_fde = nn.ModuleDict(self.val_fde)
        self.val_prob_ade = nn.ModuleDict(self.val_prob_ade)
        self.val_prob_fde = nn.ModuleDict(self.val_prob_fde)

        # ========== ORIGINAL TEST SET METRICS ==========
        self.test_min_ade, self.test_min_fde = {}, {}
        self.test_prob_ade, self.test_prob_fde = {}, {}
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_min_ade[key] = MeanMetric()
            self.test_min_fde[key] = MeanMetric()
            self.test_prob_ade[key] = MeanMetric()
            self.test_prob_fde[key] = MeanMetric()
        self.test_min_ade = nn.ModuleDict(self.test_min_ade)
        self.test_min_fde = nn.ModuleDict(self.test_min_fde)
        self.test_prob_ade = nn.ModuleDict(self.test_prob_ade)
        self.test_prob_fde = nn.ModuleDict(self.test_prob_fde)

        # ========== BALANCED TEST SET METRICS ==========
        self.test_balanced_min_ade, self.test_balanced_min_fde = {}, {}
        self.test_balanced_prob_ade, self.test_balanced_prob_fde = {}, {}
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_balanced_min_ade[key] = MeanMetric()
            self.test_balanced_min_fde[key] = MeanMetric()
            self.test_balanced_prob_ade[key] = MeanMetric()
            self.test_balanced_prob_fde[key] = MeanMetric()
        self.test_balanced_min_ade = nn.ModuleDict(self.test_balanced_min_ade)
        self.test_balanced_min_fde = nn.ModuleDict(self.test_balanced_min_fde)
        self.test_balanced_prob_ade = nn.ModuleDict(self.test_balanced_prob_ade)
        self.test_balanced_prob_fde = nn.ModuleDict(self.test_balanced_prob_fde)

        # ========== SEEN AIRPORT METRICS ==========
        self.val_seen_ade, self.val_seen_fde = {}, {}
        self.val_seen_prob_ade, self.val_seen_prob_fde = {}, {}
        self.test_seen_ade, self.test_seen_fde = {}, {}
        self.test_seen_prob_ade, self.test_seen_prob_fde = {}, {}
        self.test_balanced_seen_ade, self.test_balanced_seen_fde = {}, {}
        self.test_balanced_seen_prob_ade, self.test_balanced_seen_prob_fde = {}, {}

        for pred_len, airport in itertools.product(self.pred_lens, self.seen_airports):
            key = f"{airport}_t={pred_len}"
            self.val_seen_ade[key] = MeanMetric()
            self.val_seen_fde[key] = MeanMetric()
            self.val_seen_prob_ade[key] = MeanMetric()
            self.val_seen_prob_fde[key] = MeanMetric()
            self.test_seen_ade[key] = MeanMetric()
            self.test_seen_fde[key] = MeanMetric()
            self.test_seen_prob_ade[key] = MeanMetric()
            self.test_seen_prob_fde[key] = MeanMetric()
            self.test_balanced_seen_ade[key] = MeanMetric()
            self.test_balanced_seen_fde[key] = MeanMetric()
            self.test_balanced_seen_prob_ade[key] = MeanMetric()
            self.test_balanced_seen_prob_fde[key] = MeanMetric()

        self.val_seen_ade = nn.ModuleDict(self.val_seen_ade)
        self.val_seen_fde = nn.ModuleDict(self.val_seen_fde)
        self.val_seen_prob_ade = nn.ModuleDict(self.val_seen_prob_ade)
        self.val_seen_prob_fde = nn.ModuleDict(self.val_seen_prob_fde)
        self.test_seen_ade = nn.ModuleDict(self.test_seen_ade)
        self.test_seen_fde = nn.ModuleDict(self.test_seen_fde)
        self.test_seen_prob_ade = nn.ModuleDict(self.test_seen_prob_ade)
        self.test_seen_prob_fde = nn.ModuleDict(self.test_seen_prob_fde)
        self.test_balanced_seen_ade = nn.ModuleDict(self.test_balanced_seen_ade)
        self.test_balanced_seen_fde = nn.ModuleDict(self.test_balanced_seen_fde)
        self.test_balanced_seen_prob_ade = nn.ModuleDict(self.test_balanced_seen_prob_ade)
        self.test_balanced_seen_prob_fde = nn.ModuleDict(self.test_balanced_seen_prob_fde)

        # ========== UNSEEN AIRPORT METRICS ==========
        if len(self.unseen_airports) > 0:
            self.test_unseen_ade, self.test_unseen_fde = {}, {}
            self.test_unseen_prob_ade, self.test_unseen_prob_fde = {}, {}
            self.test_balanced_unseen_ade, self.test_balanced_unseen_fde = {}, {}
            self.test_balanced_unseen_prob_ade, self.test_balanced_unseen_prob_fde = {}, {}

            for pred_len, airport in itertools.product(self.pred_lens, self.unseen_airports):
                key = f"{airport}_t={pred_len}"
                self.test_unseen_ade[key] = MeanMetric()
                self.test_unseen_fde[key] = MeanMetric()
                self.test_unseen_prob_ade[key] = MeanMetric()
                self.test_unseen_prob_fde[key] = MeanMetric()
                self.test_balanced_unseen_ade[key] = MeanMetric()
                self.test_balanced_unseen_fde[key] = MeanMetric()
                self.test_balanced_unseen_prob_ade[key] = MeanMetric()
                self.test_balanced_unseen_prob_fde[key] = MeanMetric()

            self.test_unseen_ade = nn.ModuleDict(self.test_unseen_ade)
            self.test_unseen_fde = nn.ModuleDict(self.test_unseen_fde)
            self.test_unseen_prob_ade = nn.ModuleDict(self.test_unseen_prob_ade)
            self.test_unseen_prob_fde = nn.ModuleDict(self.test_unseen_prob_fde)
            self.test_balanced_unseen_ade = nn.ModuleDict(self.test_balanced_unseen_ade)
            self.test_balanced_unseen_fde = nn.ModuleDict(self.test_balanced_unseen_fde)
            self.test_balanced_unseen_prob_ade = nn.ModuleDict(self.test_balanced_unseen_prob_ade)
            self.test_balanced_unseen_prob_fde = nn.ModuleDict(self.test_balanced_unseen_prob_fde)

        # ========== PER-MODE METRICS (original test set only) ==========
        self.test_mode_min_ade = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)
        })
        self.test_mode_min_fde = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)
        })
        self.test_mode_prob_ade = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)
        })
        self.test_mode_prob_fde = nn.ModuleDict({
            f"{TURN_MODES_NAMES[m]}_t={t}": MeanMetric()
            for m, t in itertools.product(VALID_TURN_MODES, self.pred_lens)
        })
        self.test_mode_nll = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_rmse = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })

        # ========== PER-MODE OFF-ROAD METRICS - PREDICTED (original test set only) ==========
        self.test_mode_off_road_rate = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_off_road_max_dist = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_off_road_mean_dist = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_off_road_critical_rate = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })

        # ========== PER-MODE OFF-ROAD METRICS - GROUND TRUTH (original test set only) ==========
        # Reference: how often GT itself goes off-road per maneuver type.
        # Useful for diagnosing label noise concentrated in specific modes.
        self.test_mode_gt_off_road_rate = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_gt_off_road_max_dist = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_gt_off_road_mean_dist = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })
        self.test_mode_gt_off_road_critical_rate = nn.ModuleDict({
            TURN_MODES_NAMES[m]: MeanMetric() for m in VALID_TURN_MODES
        })

        # ========== OFF-ROAD DETECTION METRICS ==========
        # Predicted trajectory - original test set
        self.test_off_road_rate = MeanMetric()
        self.test_off_road_max_dist = MeanMetric()
        self.test_off_road_mean_dist = MeanMetric()
        self.test_off_road_critical_rate = MeanMetric()
        self.test_off_road_episodes = MeanMetric()
        self.test_off_road_duration = MeanMetric()

        # Predicted trajectory - balanced test set
        self.test_balanced_off_road_rate = MeanMetric()
        self.test_balanced_off_road_max_dist = MeanMetric()
        self.test_balanced_off_road_mean_dist = MeanMetric()
        self.test_balanced_off_road_critical_rate = MeanMetric()
        self.test_balanced_off_road_episodes = MeanMetric()
        self.test_balanced_off_road_duration = MeanMetric()

        # Ground truth trajectory - original test set
        self.test_gt_off_road_rate = MeanMetric()
        self.test_gt_off_road_max_dist = MeanMetric()
        self.test_gt_off_road_mean_dist = MeanMetric()
        self.test_gt_off_road_critical_rate = MeanMetric()

        # Ground truth trajectory - balanced test set
        self.test_balanced_gt_off_road_rate = MeanMetric()
        self.test_balanced_gt_off_road_max_dist = MeanMetric()
        self.test_balanced_gt_off_road_mean_dist = MeanMetric()
        self.test_balanced_gt_off_road_critical_rate = MeanMetric()

        # Off-road evaluator (lazy initialization)
        self._off_road_evaluator = None
        self._current_airport = None

        # ========== METRIC FUNCTIONS ==========
        assert self.eparams.propagation in ['joint', 'marginal']
        if self.eparams.propagation == 'marginal':
            from amelia_tf.utils.metrics import compute_mode_accuracy, compute_mode_rmse, compute_nll
            from amelia_tf.utils.metrics import mode_ade, mode_fde
            from amelia_tf.utils.metrics import ModalClassificationMetrics
            from amelia_tf.utils.metrics import marginal_ade as ade
            from amelia_tf.utils.metrics import marginal_fde as fde
            from amelia_tf.utils.metrics import marginal_prob_ade as prob_ade
            from amelia_tf.utils.metrics import marginal_prob_fde as prob_fde
            from amelia_tf.utils.losses import marginal_loss as compute_loss
        else:
            from amelia_tf.utils.metrics import joint_ade as ade
            from amelia_tf.utils.metrics import joint_fde as fde
            from amelia_tf.utils.metrics import joint_prob_ade as prob_ade
            from amelia_tf.utils.metrics import joint_prob_fde as prob_fde
            from amelia_tf.utils.losses import lmbd_marginal_joint_loss as compute_loss

        self.val_modal_accumulator = ModalClassificationMetrics(self.num_dec_heads)
        self.test_modal_accumulator = ModalClassificationMetrics(self.num_dec_heads)
        self.test_balanced_modal_accumulator = ModalClassificationMetrics(self.num_dec_heads)

        self.compute_mode_accuracy = compute_mode_accuracy
        self.compute_mode_rmse = compute_mode_rmse
        self.compute_nll = compute_nll
        self.mode_ade, self.mode_fde = mode_ade, mode_fde
        self.ade, self.fde = ade, fde
        self.prob_ade, self.prob_fde = prob_ade, prob_fde
        self.compute_loss = compute_loss
        self.geodesic = Geodesic.WGS84

        os.makedirs(self.eparams.plot_dir, exist_ok=True)
        out_dir = os.path.join(self.eparams.plot_dir, f"{date.today()}_{self.eparams.tag}")
        self.val_out_dir = os.path.join(out_dir, 'val')
        os.makedirs(self.val_out_dir, exist_ok=True)
        self.test_out_dir = os.path.join(out_dir, 'test')
        os.makedirs(self.test_out_dir, exist_ok=True)

    def _get_off_road_evaluator(self, airport_code: str):
        """Lazy initialization of off-road evaluator for a specific airport."""
        if not self.eparams.get('enable_off_road_eval', True):
            return None
        if self._off_road_evaluator is None or self._current_airport != airport_code:
            from amelia_tf.utils.off_road_evaluator import OffRoadEvaluator
            self._off_road_evaluator = OffRoadEvaluator(
                asset_dir=self.eparams.asset_dir,
                airport_code=airport_code
            )
            self._current_airport = airport_code
        return self._off_road_evaluator

    def _evaluate_off_road(
            self,
            evaluator,
            ego_mu: torch.Tensor,
            ego_pred_scores: torch.Tensor,
            sequences: torch.Tensor,
            ego_agents,
            airport_codes,
            prefix: str,
            rate_metric: MeanMetric,
            max_dist_metric: MeanMetric,
            mean_dist_metric: MeanMetric,
            critical_rate_metric: MeanMetric,
            episodes_metric: MeanMetric = None,
            duration_metric: MeanMetric = None,
    ):
        """
        Run off-road evaluation for a given trajectory set and log results.

        sequences, ego_agents, airport_codes must already be sliced to match
        the batch size of ego_mu - no index remapping is done here.

        Args:
            evaluator:            OffRoadEvaluator instance for the current airport.
            ego_mu:               (B, 1, T, H, D) predicted or (B, 1, T, 1, D) GT.
            ego_pred_scores:      (B, 1, H) mode scores. Pass uniform ones for GT.
            sequences:            (B, A, T, D) absolute coordinates for these B samples.
            ego_agents:           (B,) ego agent index for each sample.
            airport_codes:        list[B] airport code for each sample.
            prefix:               Wandb/tensorboard logging prefix.
            rate_metric:          MeanMetric for off-road rate.
            max_dist_metric:      MeanMetric for max off-road distance (m).
            mean_dist_metric:     MeanMetric for mean off-road distance (m).
            critical_rate_metric: MeanMetric for critical off-road rate.
            episodes_metric:      Optional MeanMetric for number of off-road episodes.
            duration_metric:      Optional MeanMetric for mean episode duration (steps).
        """
        results_list = evaluator.evaluate_prediction(
            ego_mu=ego_mu,
            ego_pred_scores=ego_pred_scores,
            sequences=sequences,
            ego_agents=ego_agents,
            airport_codes=airport_codes,
            hist_len=self.hist_len,
            safety_margin=1.0,
        )
        for result in results_list:
            rate_metric(result['off_road_rate'])
            max_dist_metric(result['max_off_road_distance_m'])
            mean_dist_metric(result['mean_off_road_distance_m'])
            critical_rate_metric(result['critical_off_road_rate'])
            if episodes_metric is not None:
                episodes_metric(result['num_off_road_episodes'])
            if duration_metric is not None:
                duration_metric(result['mean_episode_duration_steps'])

        self.log(f"{prefix}/off_road_rate",          rate_metric,          on_step=False, on_epoch=True)
        self.log(f"{prefix}/off_road_max_dist_m",    max_dist_metric,      on_step=False, on_epoch=True)
        self.log(f"{prefix}/off_road_mean_dist_m",   mean_dist_metric,     on_step=False, on_epoch=True)
        self.log(f"{prefix}/off_road_critical_rate", critical_rate_metric, on_step=False, on_epoch=True)
        if episodes_metric is not None:
            self.log(f"{prefix}/off_road_episodes",       episodes_metric, on_step=False, on_epoch=True)
        if duration_metric is not None:
            self.log(f"{prefix}/off_road_duration_steps", duration_metric, on_step=False, on_epoch=True)

    def log_sigma_stats(self, sigma: torch.Tensor, prefix: str = "test"):
        sigma_flat = sigma[~torch.isnan(sigma)]
        if sigma_flat.numel() > 0:
            self.log(f"{prefix}/sigma_mean", sigma_flat.mean(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_min", sigma_flat.min(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_max", sigma_flat.max(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/sigma_std", sigma_flat.std(), on_step=False, on_epoch=True)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                self.log(f"{prefix}/sigma_p{p}",
                         torch.quantile(sigma_flat, p / 100.0), on_step=False, on_epoch=True)

    def log_mode_probs_stats(self, mode_probs: torch.Tensor, prefix: str = "test"):
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
            self.log(f"{prefix}/mode_{k}_percentage",
                     count / mode_counts.numel() * 100, on_step=False, on_epoch=True)

    def log_prediction_quality(self, mu, sigma, Y, mode_probs, mask=None, prefix="test"):
        B, A, T_total, M, D = mu.shape
        _, _, T_pred, _ = Y.shape

        mu_future = mu[:, :, -T_pred:, :, :]
        sigma_future = sigma[:, :, -T_pred:, :, :]
        best_mode_idx = mode_probs.argmax(dim=-1)

        best_mu = mu_future[
            torch.arange(B)[:, None, None],
            torch.arange(A)[None, :, None],
            torch.arange(T_pred)[None, None, :],
            best_mode_idx[:, :, None], :]

        best_sigma = sigma_future[
            torch.arange(B)[:, None, None],
            torch.arange(A)[None, :, None],
            torch.arange(T_pred)[None, None, :],
            best_mode_idx[:, :, None], :]

        log_prob = -0.5 * (
            ((best_mu - Y) / best_sigma.clamp_min(1e-6)) ** 2 +
            torch.log(2 * torch.pi * best_sigma.clamp_min(1e-6) ** 2)
        ).sum(dim=-1)

        if mask is not None:
            mask_pred = mask[:, :, -T_pred:]
            log_prob_mean = (log_prob * mask_pred).sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
        else:
            log_prob_mean = log_prob.mean(dim=-1)

        self.log(f"{prefix}/best_mode_log_prob_mean", log_prob_mean.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_log_prob_min", log_prob_mean.min(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/best_mode_log_prob_max", log_prob_mean.max(), on_step=False, on_epoch=True)

    def analyze_nll_components(self, mu, sigma, mode_probs, Y, mask=None, prefix="test"):
        B, A, T_total, M, D = mu.shape
        _, _, T_pred, _ = Y.shape

        mu_future = mu[:, :, -T_pred:, :, :]
        sigma_future = sigma[:, :, -T_pred:, :, :]
        y_expanded = Y.unsqueeze(-2)

        logp_mode = -0.5 * (
            torch.log(2 * torch.pi * sigma_future.clamp_min(1e-6) ** 2) +
            ((y_expanded - mu_future) / sigma_future.clamp_min(1e-6)) ** 2
        ).sum(dim=-1)

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
            mask_pred = None
            max_logp_mean = max_logp_mode.mean(dim=-1)
            log_sum_exp_mean = log_sum_exp.mean(dim=-1)
            mixture_penalty_mean = mixture_penalty.mean(dim=-1)

        self.log(f"{prefix}/nll_max_mode_contribution", (-max_logp_mean).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_log_sum_exp_contribution", (-log_sum_exp_mean).mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty", mixture_penalty_mean.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty_max", mixture_penalty.max(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/nll_mixture_penalty_min", mixture_penalty.min(), on_step=False, on_epoch=True)

        significant = (mixture_penalty > 0.1).float()
        if mask_pred is not None:
            significant = significant * mask_pred
            contrib_percent = significant.sum() / mask_pred.sum().clamp_min(1) * 100
        else:
            contrib_percent = significant.mean() * 100
        self.log(f"{prefix}/samples_with_multi_mode_contribution_percent",
                 contrib_percent, on_step=False, on_epoch=True)

    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch, plot: bool = False, tag: str = 'temp', out_dir: str = 'temp'):
        Y = batch['scene_dict']['rel_sequences']
        X = torch.zeros_like(Y).type(torch.float)
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]
        Y = Y[:, :, :, :4]
        X = X[:, :, :, :4]
        B, N, T, D = Y.shape
        Y = Y[..., G.REL_XYZ[:D]]
        context = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']

        pred_scores, mu, sigma = self.net(
            X, context=context, adjacency=adjacency, mask=None,
        )
        loss, loss_cls, loss_reg = self.compute_loss(
            pred_scores, mu, sigma, Y,
            ego_agent=ego_agent,
            epoch=self.current_epoch + 1,
            agent_mask=masks,
        )

        if plot:
            plot_scene_batch(
                self.eparams.asset_dir, batch,
                (pred_scores, mu, sigma),
                self.hist_len, self.geodesic,
                tag, out_dir, self.eparams.propagation
            )

        return loss, loss_cls, loss_reg, pred_scores, mu, sigma, Y[:, :, self.hist_len:]

    def training_step(self, batch: Any, batch_idx: int):
        loss, loss_cls, loss_reg, _, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.train_loss_cls(loss_cls)
        self.train_loss_reg(loss_reg)
        self.log("losses/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses/train_mode", self.train_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses/train_traj", self.train_loss_reg, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_modal_accumulator.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        plot = (
            self.eparams.plot_val
            and self.current_epoch >= self.eparams.plot_after_n_epochs
            and (batch_idx + 1) % self.eparams.plot_every_n == 0
        )
        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, loss_cls, loss_reg, pred_scores, mu, sigma, fut_rel = self.model_step(
            batch, plot, tag, self.val_out_dir
        )

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
        self.log("losses/val_mode", self.val_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses/val_traj", self.val_loss_reg, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t = ego_mu[:, :, :self.hist_len + t]
            mask_t = mask[:, :, :self.hist_len + t]
            fut_t = ego_fut[:, :, :t]
            key = 't=max' if t == self.max_pred_len else f"t={t}"

            self.val_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"val/ade/{key}", self.val_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.val_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"val/fde/{key}", self.val_fde[key], on_step=False, on_epoch=True, prog_bar=True)

            mode_ade_t = self.mode_ade(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            mode_fde_t = self.mode_fde(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            self.val_prob_ade[key](mode_ade_t)
            self.val_prob_fde[key](mode_fde_t)
            self.log(f"val/prob_ade/{key}", self.val_prob_ade[key], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"val/prob_fde/{key}", self.val_prob_fde[key], on_step=False, on_epoch=True, prog_bar=False)

        self.val_modal_accumulator.update(ego_mu, ego_pred_scores, ego_fut, mask)

        rmse_val = self.compute_mode_rmse(ego_mu, ego_pred_scores, ego_fut, mask=mask)
        self.rmse_val(rmse_val)
        self.log("val/rmse", self.rmse_val, on_step=False, on_epoch=True, prog_bar=True)

        nll_val = self.compute_nll(ego_mu, ego_sigma, ego_pred_scores, ego_fut, mask=mask)
        self.nll_val(nll_val)
        self.log("val/nll", self.nll_val, on_step=False, on_epoch=True, prog_bar=True)

        if len(self.seen_airports) > 1:
            airport_ids = batch['scene_dict']['airport_id']
            for airport in self.seen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu = ego_mu[airport_idx]
                airport_fut = ego_fut[airport_idx]
                airport_mask = mask[airport_idx]
                airport_pred_scores = ego_pred_scores[airport_idx]

                for t in self.pred_lens:
                    mu_t = airport_mu[:, :, :self.hist_len + t]
                    fut_t = airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len + t]
                    key = f"{airport}_t={t}"

                    self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(f"val_seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True)
                    self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(f"val_seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True)

                    mode_ade_t = self.mode_ade(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    mode_fde_t = self.mode_fde(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    self.val_seen_prob_ade[key](mode_ade_t)
                    self.val_seen_prob_fde[key](mode_fde_t)
                    self.log(f"val_seen_prob_ade/{key}", self.val_seen_prob_ade[key], on_step=False, on_epoch=True)
                    self.log(f"val_seen_prob_fde/{key}", self.val_seen_prob_fde[key], on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        modal_metrics = self.val_modal_accumulator.compute()

        self.log("val/modal_accuracy", modal_metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log("val/modal_macro_f1", modal_metrics['macro_f1'], on_epoch=True, prog_bar=True)
        self.log("val/modal_weighted_f1", modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)

        for mode_idx in range(len(modal_metrics['precision_per_mode'])):
            self.log(f"val/modal_precision/mode_{mode_idx}",
                     modal_metrics['precision_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"val/modal_recall/mode_{mode_idx}",
                     modal_metrics['recall_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"val/modal_f1/mode_{mode_idx}",
                     modal_metrics['f1_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"val/modal_support/mode_{mode_idx}",
                     modal_metrics['support_per_mode'][mode_idx], on_epoch=True, prog_bar=False)

        if self.current_epoch % 5 == 0:
            print("\n" + "=" * 100)
            print(f"VALIDATION EPOCH {self.current_epoch} - MODAL CLASSIFICATION RESULTS")
            print("=" * 100)
            self.val_modal_accumulator.print_report()
            self.val_modal_accumulator.print_confusion_matrix()

        self.val_modal_accumulator.reset()

    def on_test_epoch_start(self):
        self.test_modal_accumulator.reset()
        self.test_balanced_modal_accumulator.reset()

    def on_test_epoch_end(self):
        # Original test set modal classification
        print("\n" + "=" * 100)
        print("ORIGINAL TEST SET MODAL CLASSIFICATION RESULTS")
        print("=" * 100)
        modal_metrics = self.test_modal_accumulator.compute()

        self.log("test/original/modal_accuracy", modal_metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log("test/original/modal_macro_f1", modal_metrics['macro_f1'], on_epoch=True, prog_bar=True)
        self.log("test/original/modal_weighted_f1", modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)

        num_modes = len(modal_metrics['precision_per_mode'])
        for mode_idx in range(num_modes):
            self.log(f"test/original/modal_precision/mode_{mode_idx}",
                     modal_metrics['precision_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"test/original/modal_recall/mode_{mode_idx}",
                     modal_metrics['recall_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"test/original/modal_f1/mode_{mode_idx}",
                     modal_metrics['f1_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"test/original/modal_support/mode_{mode_idx}",
                     modal_metrics['support_per_mode'][mode_idx], on_epoch=True, prog_bar=False)

        self.test_modal_accumulator.print_report()
        self.test_modal_accumulator.print_confusion_matrix()

        # Balanced test set modal classification
        print("\n" + "=" * 100)
        print("BALANCED TEST SET MODAL CLASSIFICATION RESULTS")
        print("=" * 100)
        balanced_modal_metrics = self.test_balanced_modal_accumulator.compute()

        self.log("test/balanced/modal_accuracy", balanced_modal_metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log("test/balanced/modal_macro_f1", balanced_modal_metrics['macro_f1'], on_epoch=True, prog_bar=True)
        self.log("test/balanced/modal_weighted_f1", balanced_modal_metrics['weighted_f1'], on_epoch=True, prog_bar=True)

        for mode_idx in range(num_modes):
            self.log(f"test/balanced/modal_precision/mode_{mode_idx}",
                     balanced_modal_metrics['precision_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"test/balanced/modal_recall/mode_{mode_idx}",
                     balanced_modal_metrics['recall_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"test/balanced/modal_f1/mode_{mode_idx}",
                     balanced_modal_metrics['f1_per_mode'][mode_idx], on_epoch=True, prog_bar=False)
            self.log(f"test/balanced/modal_support/mode_{mode_idx}",
                     balanced_modal_metrics['support_per_mode'][mode_idx], on_epoch=True, prog_bar=False)

        self.test_balanced_modal_accumulator.print_report()
        self.test_balanced_modal_accumulator.print_confusion_matrix()

        # Per-mode off-road summary: predicted vs GT side by side
        print("\n" + "=" * 100)
        print("PER-MODE OFF-ROAD METRICS (PREDICTED vs GROUND TRUTH)")
        print("=" * 100)
        print(f"{'Mode':<20} {'Pred Rate':>10} {'GT Rate':>10} {'Pred MeanDist':>14} "
              f"{'GT MeanDist':>12} {'Pred MaxDist':>13} {'GT MaxDist':>11}")
        print("-" * 100)
        for mode_idx in VALID_TURN_MODES:
            mode_name = TURN_MODES_NAMES[mode_idx]
            pred_rate  = self.test_mode_off_road_rate[mode_name].compute()
            gt_rate    = self.test_mode_gt_off_road_rate[mode_name].compute()
            pred_mean  = self.test_mode_off_road_mean_dist[mode_name].compute()
            gt_mean    = self.test_mode_gt_off_road_mean_dist[mode_name].compute()
            pred_max   = self.test_mode_off_road_max_dist[mode_name].compute()
            gt_max     = self.test_mode_gt_off_road_max_dist[mode_name].compute()
            print(f"{mode_name:<20} {pred_rate:>10.4f} {gt_rate:>10.4f} {pred_mean:>14.2f} "
                  f"{gt_mean:>12.2f} {pred_max:>13.2f} {gt_max:>11.2f}")

        self.test_modal_accumulator.reset()
        self.test_balanced_modal_accumulator.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        plot = self.eparams.plot_test if (batch_idx) % 10 == 0 else False
        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"

        loss, loss_cls, loss_reg, pred_scores, mu, sigma, fut_rel = self.model_step(
            batch, plot, tag, self.test_out_dir
        )
        ego_agent = batch['scene_dict']['ego_agent_id_test']

        if self.eparams.propagation == 'marginal':
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            mask = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError

        # Prepare batch-level data needed by the off-road evaluator.
        # These stay as the full-batch reference; per-mode calls will slice them.
        full_sequences   = batch['scene_dict']['sequences']       # (B, A, T, D)
        full_ego_agents  = batch['scene_dict']['ego_agent_id_test']  # (B,)
        full_airport_codes = batch['scene_dict']['airport_id']    # list[B]

        # Select metrics and prefix based on which test set this batch belongs to
        if dataloader_idx == 0:
            prefix = "test/original"
            min_ade_dict = self.test_min_ade
            min_fde_dict = self.test_min_fde
            prob_ade_dict = self.test_prob_ade
            prob_fde_dict = self.test_prob_fde
            seen_ade_dict = self.test_seen_ade
            seen_fde_dict = self.test_seen_fde
            seen_prob_ade_dict = self.test_seen_prob_ade
            seen_prob_fde_dict = self.test_seen_prob_fde
            unseen_ade_dict = self.test_unseen_ade if len(self.unseen_airports) > 0 else None
            unseen_fde_dict = self.test_unseen_fde if len(self.unseen_airports) > 0 else None
            unseen_prob_ade_dict = self.test_unseen_prob_ade if len(self.unseen_airports) > 0 else None
            unseen_prob_fde_dict = self.test_unseen_prob_fde if len(self.unseen_airports) > 0 else None
            self.test_modal_accumulator.update(ego_mu, ego_pred_scores, ego_fut, mask)
            nll_metric = self.nll_test
            rmse_metric = self.rmse_test
            off_road_metrics = dict(
                rate=self.test_off_road_rate,
                max_dist=self.test_off_road_max_dist,
                mean_dist=self.test_off_road_mean_dist,
                critical_rate=self.test_off_road_critical_rate,
                episodes=self.test_off_road_episodes,
                duration=self.test_off_road_duration,
            )
            gt_off_road_metrics = dict(
                rate=self.test_gt_off_road_rate,
                max_dist=self.test_gt_off_road_max_dist,
                mean_dist=self.test_gt_off_road_mean_dist,
                critical_rate=self.test_gt_off_road_critical_rate,
            )
        else:
            prefix = "test/balanced"
            min_ade_dict = self.test_balanced_min_ade
            min_fde_dict = self.test_balanced_min_fde
            prob_ade_dict = self.test_balanced_prob_ade
            prob_fde_dict = self.test_balanced_prob_fde
            seen_ade_dict = self.test_balanced_seen_ade
            seen_fde_dict = self.test_balanced_seen_fde
            seen_prob_ade_dict = self.test_balanced_seen_prob_ade
            seen_prob_fde_dict = self.test_balanced_seen_prob_fde
            unseen_ade_dict = self.test_balanced_unseen_ade if len(self.unseen_airports) > 0 else None
            unseen_fde_dict = self.test_balanced_unseen_fde if len(self.unseen_airports) > 0 else None
            unseen_prob_ade_dict = self.test_balanced_unseen_prob_ade if len(self.unseen_airports) > 0 else None
            unseen_prob_fde_dict = self.test_balanced_unseen_prob_fde if len(self.unseen_airports) > 0 else None
            self.test_balanced_modal_accumulator.update(ego_mu, ego_pred_scores, ego_fut, mask)
            nll_metric = self.nll_test
            rmse_metric = self.rmse_test
            off_road_metrics = dict(
                rate=self.test_balanced_off_road_rate,
                max_dist=self.test_balanced_off_road_max_dist,
                mean_dist=self.test_balanced_off_road_mean_dist,
                critical_rate=self.test_balanced_off_road_critical_rate,
                episodes=self.test_balanced_off_road_episodes,
                duration=self.test_balanced_off_road_duration,
            )
            gt_off_road_metrics = dict(
                rate=self.test_balanced_gt_off_road_rate,
                max_dist=self.test_balanced_gt_off_road_max_dist,
                mean_dist=self.test_balanced_gt_off_road_mean_dist,
                critical_rate=self.test_balanced_gt_off_road_critical_rate,
            )

        # minADE / minFDE; probADE / probFDE
        for t in self.pred_lens:
            mu_t = ego_mu[:, :, :self.hist_len + t]
            fut_t = ego_fut[:, :, :t]
            mask_t = mask[:, :, :self.hist_len + t]
            key = 't=max' if t == self.max_pred_len else f"t={t}"

            min_ade_dict[key](self.ade(mu_t, fut_t, mask=mask_t))
            min_fde_dict[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"{prefix}/min_ade/{key}", min_ade_dict[key], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{prefix}/min_fde/{key}", min_fde_dict[key], on_step=False, on_epoch=True, prog_bar=True)

            mode_ade_t = self.mode_ade(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            mode_fde_t = self.mode_fde(mu_t, ego_pred_scores, fut_t, mask=mask_t)
            prob_ade_dict[key](mode_ade_t)
            prob_fde_dict[key](mode_fde_t)
            self.log(f"{prefix}/prob_ade/{key}", prob_ade_dict[key], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{prefix}/prob_fde/{key}", prob_fde_dict[key], on_step=False, on_epoch=True, prog_bar=False)

        # NLL and RMSE
        nll = self.compute_nll(ego_mu, ego_sigma, ego_pred_scores, ego_fut, mask=mask)
        nll_metric(nll)
        self.log(f"{prefix}/nll", nll_metric, on_step=False, on_epoch=True, prog_bar=True)

        rmse = self.compute_mode_rmse(ego_mu, ego_pred_scores, ego_fut, mask=mask)
        rmse_metric(rmse)
        self.log(f"{prefix}/rmse", rmse_metric, on_step=False, on_epoch=True, prog_bar=True)

        # Diagnostics
        self.log_sigma_stats(ego_sigma, prefix)
        self.log_mode_probs_stats(ego_pred_scores, prefix)
        self.log_prediction_quality(ego_mu, ego_sigma, ego_fut, ego_pred_scores, mask, prefix)
        self.analyze_nll_components(ego_mu, ego_sigma, ego_pred_scores, ego_fut, mask, prefix)

        # ========== PREDICTED TRAJECTORY OFF-ROAD METRICS ==========
        evaluator = None
        try:
            airport_code = full_airport_codes[0]
            evaluator = self._get_off_road_evaluator(airport_code)

            if evaluator is not None:
                self._evaluate_off_road(
                    evaluator=evaluator,
                    ego_mu=ego_mu,
                    ego_pred_scores=ego_pred_scores,
                    sequences=full_sequences,
                    ego_agents=full_ego_agents,
                    airport_codes=full_airport_codes,
                    prefix=prefix,
                    rate_metric=off_road_metrics['rate'],
                    max_dist_metric=off_road_metrics['max_dist'],
                    mean_dist_metric=off_road_metrics['mean_dist'],
                    critical_rate_metric=off_road_metrics['critical_rate'],
                    episodes_metric=off_road_metrics['episodes'],
                    duration_metric=off_road_metrics['duration'],
                )
        except Exception as e:
            print(f"Warning: Off-road evaluation failed: {e}")
            traceback.print_exc()

        # ========== GROUND TRUTH TRAJECTORY OFF-ROAD METRICS ==========
        # ego_fut (B, 1, T, D) is wrapped into (B, 1, T, 1, D) to match the
        # evaluator interface. Uniform scores ensure the single dummy head is
        # always selected. Serves as a lower-bound reference.
        if evaluator is not None:
            try:
                B = ego_fut.shape[0]
                self._evaluate_off_road(
                    evaluator=evaluator,
                    ego_mu=ego_fut.unsqueeze(3),
                    ego_pred_scores=torch.ones(B, 1, 1, device=ego_fut.device),
                    sequences=full_sequences,
                    ego_agents=full_ego_agents,
                    airport_codes=full_airport_codes,
                    prefix=f"{prefix}/gt",
                    rate_metric=gt_off_road_metrics['rate'],
                    max_dist_metric=gt_off_road_metrics['max_dist'],
                    mean_dist_metric=gt_off_road_metrics['mean_dist'],
                    critical_rate_metric=gt_off_road_metrics['critical_rate'],
                )
            except Exception as e:
                print(f"Warning: GT off-road evaluation failed: {e}")
                traceback.print_exc()

        # ========== PER-MODE METRICS (original test set only) ==========
        if dataloader_idx == 0:
            Y_mode = batch['scene_dict'].get('rule_based_encoding')
            if Y_mode is not None:
                true_mode_idx = Y_mode[..., :4].float().argmax(dim=-1).long()
                ego_true_mode = separate_ego_agent(true_mode_idx, ego_agent).squeeze(1)

                for mode_idx in VALID_TURN_MODES:
                    mode_name = TURN_MODES_NAMES[mode_idx]
                    sample_mask = (ego_true_mode == mode_idx)
                    if sample_mask.sum() == 0:
                        continue

                    mu_m     = ego_mu[sample_mask]
                    sigma_m  = ego_sigma[sample_mask]
                    scores_m = ego_pred_scores[sample_mask]
                    fut_m    = ego_fut[sample_mask]
                    mask_m   = mask[sample_mask]

                    # Slice batch-level data to match this mode's samples
                    mode_sequences    = full_sequences[sample_mask]
                    mode_ego_agents   = full_ego_agents[sample_mask.cpu()]
                    mode_airport_codes = [full_airport_codes[i]
                                          for i in sample_mask.nonzero(as_tuple=True)[0].tolist()]

                    # NLL and RMSE
                    nll_m = self.compute_nll(mu_m, sigma_m, scores_m, fut_m, mask=mask_m)
                    self.test_mode_nll[mode_name](nll_m)
                    self.log(f"test/original/mode_nll/{mode_name}",
                             self.test_mode_nll[mode_name], on_step=False, on_epoch=True)

                    rmse_m = self.compute_mode_rmse(mu_m, scores_m, fut_m, mask=mask_m)
                    self.test_mode_rmse[mode_name](rmse_m)
                    self.log(f"test/original/mode_rmse/{mode_name}",
                             self.test_mode_rmse[mode_name], on_step=False, on_epoch=True)

                    # minADE / minFDE and probADE / probFDE per pred_len
                    for t in self.pred_lens:
                        mu_t   = mu_m[:, :, :self.hist_len + t]
                        fut_t  = fut_m[:, :, :t]
                        mask_t = mask_m[:, :, :self.hist_len + t]
                        t_str  = f"t={t}"
                        key    = f"{mode_name}_t={t}"

                        self.test_mode_min_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                        self.log(f"test/original/mode_min_ade/{mode_name}/{t_str}",
                                 self.test_mode_min_ade[key], on_step=False, on_epoch=True)

                        self.test_mode_min_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                        self.log(f"test/original/mode_min_fde/{mode_name}/{t_str}",
                                 self.test_mode_min_fde[key], on_step=False, on_epoch=True)

                        self.test_mode_prob_ade[key](self.mode_ade(mu_t, scores_m, fut_t, mask=mask_t))
                        self.log(f"test/original/mode_prob_ade/{mode_name}/{t_str}",
                                 self.test_mode_prob_ade[key], on_step=False, on_epoch=True)

                        self.test_mode_prob_fde[key](self.mode_fde(mu_t, scores_m, fut_t, mask=mask_t))
                        self.log(f"test/original/mode_prob_fde/{mode_name}/{t_str}",
                                 self.test_mode_prob_fde[key], on_step=False, on_epoch=True)

                    if evaluator is not None:
                        # Per-mode off-road - predicted trajectories
                        try:
                            self._evaluate_off_road(
                                evaluator=evaluator,
                                ego_mu=mu_m,
                                ego_pred_scores=scores_m,
                                sequences=mode_sequences,
                                ego_agents=mode_ego_agents,
                                airport_codes=mode_airport_codes,
                                prefix=f"test/original/mode_{mode_name}",
                                rate_metric=self.test_mode_off_road_rate[mode_name],
                                max_dist_metric=self.test_mode_off_road_max_dist[mode_name],
                                mean_dist_metric=self.test_mode_off_road_mean_dist[mode_name],
                                critical_rate_metric=self.test_mode_off_road_critical_rate[mode_name],
                            )
                        except Exception as e:
                            print(f"Warning: Per-mode pred off-road failed for {mode_name}: {e}")

                        # Per-mode off-road - ground truth trajectories
                        try:
                            B_m = fut_m.shape[0]
                            self._evaluate_off_road(
                                evaluator=evaluator,
                                ego_mu=fut_m.unsqueeze(3),
                                ego_pred_scores=torch.ones(B_m, 1, 1, device=fut_m.device),
                                sequences=mode_sequences,
                                ego_agents=mode_ego_agents,
                                airport_codes=mode_airport_codes,
                                prefix=f"test/original/mode_{mode_name}/gt",
                                rate_metric=self.test_mode_gt_off_road_rate[mode_name],
                                max_dist_metric=self.test_mode_gt_off_road_max_dist[mode_name],
                                mean_dist_metric=self.test_mode_gt_off_road_mean_dist[mode_name],
                                critical_rate_metric=self.test_mode_gt_off_road_critical_rate[mode_name],
                            )
                        except Exception as e:
                            print(f"Warning: Per-mode GT off-road failed for {mode_name}: {e}")

        # Airport metrics - Seen
        airport_ids = batch['scene_dict']['airport_id']
        for airport in self.seen_airports:
            airport_idx = np.where(airport_ids == airport)[0]
            if len(airport_idx) == 0:
                continue
            airport_mu = ego_mu[airport_idx]
            airport_fut = ego_fut[airport_idx]
            airport_mask = mask[airport_idx]
            airport_pred_scores = ego_pred_scores[airport_idx]

            for t in self.pred_lens:
                mu_t = airport_mu[:, :, :self.hist_len + t]
                fut_t = airport_fut[:, :, :t]
                mask_t = airport_mask[:, :, :self.hist_len + t]
                key = f"{airport}_t={t}"

                seen_ade_dict[key](self.ade(mu_t, fut_t, mask=mask_t))
                seen_fde_dict[key](self.fde(mu_t, fut_t, mask=mask_t))
                self.log(f"{prefix}/seen_ade/{key}", seen_ade_dict[key], on_step=False, on_epoch=True)
                self.log(f"{prefix}/seen_fde/{key}", seen_fde_dict[key], on_step=False, on_epoch=True)

                mode_ade_t = self.mode_ade(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                mode_fde_t = self.mode_fde(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                seen_prob_ade_dict[key](mode_ade_t)
                seen_prob_fde_dict[key](mode_fde_t)
                self.log(f"{prefix}/seen_prob_ade/{key}", seen_prob_ade_dict[key], on_step=False, on_epoch=True)
                self.log(f"{prefix}/seen_prob_fde/{key}", seen_prob_fde_dict[key], on_step=False, on_epoch=True)

        # Airport metrics - Unseen
        if len(self.unseen_airports) > 0 and unseen_ade_dict is not None:
            for airport in self.unseen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu = ego_mu[airport_idx]
                airport_fut = ego_fut[airport_idx]
                airport_mask = mask[airport_idx]
                airport_pred_scores = ego_pred_scores[airport_idx]

                for t in self.pred_lens:
                    mu_t = airport_mu[:, :, :self.hist_len + t]
                    fut_t = airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len + t]
                    key = f"{airport}_t={t}"

                    unseen_ade_dict[key](self.ade(mu_t, fut_t, mask=mask_t))
                    unseen_fde_dict[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(f"{prefix}/unseen_ade/{key}", unseen_ade_dict[key], on_step=False, on_epoch=True)
                    self.log(f"{prefix}/unseen_fde/{key}", unseen_fde_dict[key], on_step=False, on_epoch=True)

                    mode_ade_t = self.mode_ade(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    mode_fde_t = self.mode_fde(mu_t, airport_pred_scores, fut_t, mask=mask_t)
                    unseen_prob_ade_dict[key](mode_ade_t)
                    unseen_prob_fde_dict[key](mode_fde_t)
                    self.log(f"{prefix}/unseen_prob_ade/{key}", unseen_prob_ade_dict[key], on_step=False, on_epoch=True)
                    self.log(f"{prefix}/unseen_prob_fde/{key}", unseen_prob_fde_dict[key], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.SyncBatchNorm, nn.LayerNorm, LayerNorm, nn.Embedding,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.MultiheadAttention
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": self.hparams.optimizer.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
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

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = TrajPred(None, None, None)