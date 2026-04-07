import itertools
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
        self.num_dec_heads = self.net.mode_config.num_modes

        self.eparams = extra_params
        self.seen_airports = self.eparams.seen_airports
        self.unseen_airports = self.eparams.unseen_airports
        
        # ========== ????? ==========
        self.stage_configs = {
            'stage1': {
                'epochs': getattr(self.eparams, 'stage1_epochs', 10),
                'lambda_mode': 1.0,      # ???????
                'lambda_marginal': 0.0,
                'freeze_traj': True,     # ???????
                'use_true_modes': False, # ?????
                'lr_mode': 1e-4,
                'lr_traj': 0.0,          # ???
                'lr_other': 1e-4
            },
            'stage2': {
                'epochs': getattr(self.eparams, 'stage2_epochs', 10),
                'lambda_mode': 0.0,      # ???????
                'lambda_marginal': 1.0,
                'freeze_mode': True,     # ???????
                'use_true_modes': True,  # ???????
                'lr_mode': 0.0,          # ???
                'lr_traj': 5e-4,         # ??????????
                'lr_other': 1e-4
            },
            'stage3': {
                'epochs': getattr(self.eparams, 'stage3_epochs', 20),
                'lambda_mode': 0.2,      # ????(??????)
                'lambda_marginal': 1.0,
                'freeze_mode': False,
                'freeze_traj': False,
                'use_true_modes': True,
                'warmup_epochs': getattr(self.eparams, 'warmup_epochs', 5),
                'lr_mode': 1e-4,
                'lr_traj': 2e-4,
                'lr_other': 1e-4
            }
        }
        
        # ??????
        self.three_stages = getattr(self.eparams, 'three_stages', True)  # ??True
        self.current_stage = 'stage1'
        self.epoch_in_stage = 0
        self.total_epochs = sum(cfg['epochs'] for cfg in self.stage_configs.values())
        
        # ????
        self.stage_boundaries = {
            'stage1': self.stage_configs['stage1']['epochs'],
            'stage2': self.stage_configs['stage1']['epochs'] + self.stage_configs['stage2']['epochs'],
            'stage3': self.stage_configs['stage1']['epochs'] + self.stage_configs['stage2']['epochs'] + 
                     self.stage_configs['stage3']['epochs']
        }
        
        # For tracking mode prediction accuracy
        self.val_mode_acc, self.test_mode_acc = MeanMetric(), MeanMetric()
        
        # ??:????????
        self.val_per_mode_acc, self.test_per_mode_acc = {}, {}
        
        # ??????(??????MODE_MAP)
        self.MODE_MAP = {
            "TurnLeft_Accel": 0, "TurnLeft_Decel": 1, "TurnLeft_Normal": 2, "TurnLeft_Hold": 3,
            "TurnRight_Accel": 4, "TurnRight_Decel": 5, "TurnRight_Normal": 6, "TurnRight_Hold": 7,
            "Straight_Accel": 8, "Straight_Decel": 9, "Straight_Normal": 10, "Straight_Hold": 11,
            "Hold_Accel": 12, "Hold_Decel": 13, "Hold_Normal": 14, "Hold_Hold": 15,
        }
        
        # ??????
        self.valid_modes = [0, 1, 2, 4, 5, 6, 8, 9, 10, 15]
        
        # ????????(??logging)
        self.mode_names = {}
        for name, idx in self.MODE_MAP.items():
            self.mode_names[idx] = name
        
        # ????????metrics
        for mode_idx in self.valid_modes:
            mode_name = self.mode_names[mode_idx]
            self.val_per_mode_acc[mode_name] = MeanMetric()
            self.test_per_mode_acc[mode_name] = MeanMetric()
        
        # ???ModuleDict
        self.val_per_mode_acc = nn.ModuleDict(self.val_per_mode_acc)
        self.test_per_mode_acc = nn.ModuleDict(self.test_per_mode_acc)
        
        # For averaging loss across batches
        self.train_loss, self.val_loss, self.test_loss = MeanMetric(), MeanMetric(), MeanMetric()
        self.train_loss_cls, self.val_loss_cls, self.test_loss_cls = MeanMetric(), MeanMetric(), MeanMetric()
        self.train_loss_reg, self.val_loss_reg, self.test_loss_reg = MeanMetric(), MeanMetric(), MeanMetric()

        # For tracking best so far validation and testing accuracy
        self.max_pred_len = max(self.pred_lens)
        self.val_ade, self.test_ade, self.val_fde, self.test_fde = {}, {}, {}, {}
        self.val_nll, self.test_nll, self.val_prob_rmse, self.test_prob_rmse = {}, {}, {}, {}
        
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            
            # Existing ADE/FDE
            self.val_ade[key], self.test_ade[key] = MeanMetric(), MeanMetric()
            self.val_fde[key], self.test_fde[key] = MeanMetric(), MeanMetric()
            
            # NEW NLL and Prob-RMSE
            self.val_nll[key], self.test_nll[key] = MeanMetric(), MeanMetric()
            self.val_prob_rmse[key], self.test_prob_rmse[key] = MeanMetric(), MeanMetric()
        
        # Convert to ModuleDicts
        self.val_ade, self.test_ade = nn.ModuleDict(self.val_ade), nn.ModuleDict(self.test_ade)
        self.val_fde, self.test_fde = nn.ModuleDict(self.val_fde), nn.ModuleDict(self.test_fde)
        self.val_nll, self.test_nll = nn.ModuleDict(self.val_nll), nn.ModuleDict(self.test_nll)
        self.val_prob_rmse, self.test_prob_rmse = nn.ModuleDict(self.val_prob_rmse), nn.ModuleDict(self.test_prob_rmse)

        # self.val_prob_ade, self.test_prob_ade = MeanMetric(), MeanMetric()
        # self.val_prob_fde, self.test_prob_fde = MeanMetric(), MeanMetric()

        self.val_seen_ade, self.test_seen_ade = {}, {}
        self.val_seen_fde, self.test_seen_fde = {}, {}
        for pred_len, airport in itertools.product(self.pred_lens, self.seen_airports):
            key = f"{airport}_t={pred_len}"
            self.val_seen_ade[key], self.test_seen_ade[key] = MeanMetric(), MeanMetric()
            self.val_seen_fde[key], self.test_seen_fde[key] = MeanMetric(), MeanMetric()
        self.val_seen_ade  = nn.ModuleDict(self.val_seen_ade)
        self.val_seen_fde  = nn.ModuleDict(self.val_seen_fde)
        self.test_seen_ade = nn.ModuleDict(self.test_seen_ade)
        self.test_seen_fde = nn.ModuleDict(self.test_seen_fde)

        # Create metrics for unseen airports
        if len(self.unseen_airports) > 0:
            self.test_unseen_ade, self.test_unseen_fde = {}, {}
            for pred_len, airport in itertools.product(self.pred_lens, self.unseen_airports):
                key = f"{airport}_t={pred_len}"
                self.test_unseen_ade[key], self.test_unseen_fde[key] = MeanMetric(), MeanMetric()
            self.test_unseen_ade = nn.ModuleDict(self.test_unseen_ade)
            self.test_unseen_fde = nn.ModuleDict(self.test_unseen_fde)

        assert self.eparams.propagation in ['joint', 'marginal']
        if self.eparams.propagation == 'marginal':
            from amelia_tf.utils.metrics import ModalClassificationMetrics, create_turn_only_mapping
            from amelia_tf.utils.metrics import compute_mode_accuracy
            from amelia_tf.utils.metrics import marginal_ade as ade
            from amelia_tf.utils.metrics import marginal_fde as fde
            from amelia_tf.utils.metrics import compute_mode_rmse
            from amelia_tf.utils.metrics import compute_nll
            from amelia_tf.utils.metrics import marginal_prob_ade as prob_ade
            from amelia_tf.utils.metrics import marginal_prob_fde as prob_fde
            from amelia_tf.utils.losses import acceleration_marginal_loss as compute_loss
        else:
            from amelia_tf.utils.metrics import joint_ade as ade
            from amelia_tf.utils.metrics import joint_fde as fde
            from amelia_tf.utils.metrics import joint_prob_ade as prob_ade
            from amelia_tf.utils.metrics import joint_prob_fde as prob_fde
            from amelia_tf.utils.losses import lmbd_marginal_joint_loss as compute_loss
            
        self.val_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_dec_heads,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )
        self.test_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_dec_heads,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names
        )
        
        turn_mapping, turn_names = create_turn_only_mapping()

        self.val_merged_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_dec_heads,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names,
            merge_mapping=turn_mapping,
            merged_mode_names=turn_names
        )
        
        self.test_merged_modal_accumulator = ModalClassificationMetrics(
            num_modes=self.num_dec_heads,
            valid_modes=self.valid_modes,
            mode_names=self.mode_names,
            merge_mapping=turn_mapping,
            merged_mode_names=turn_names
        )

        self.ade, self.fde, self.prob_ade, self.prob_fde = ade, fde, prob_ade, prob_fde
        self.compute_mode_accuracy = compute_mode_accuracy
        self.compute_nll = compute_nll
        self.compute_prob_rmse = compute_mode_rmse
        self.compute_loss = compute_loss
        self.geodesic = Geodesic.WGS84
        self.mode_weights = None

        os.makedirs(self.eparams.plot_dir, exist_ok=True)
        out_dir = os.path.join(self.eparams.plot_dir, f"{date.today()}_{self.eparams.tag}")
        self.val_out_dir = os.path.join(out_dir, 'val')
        os.makedirs(self.val_out_dir, exist_ok=True)
        self.test_out_dir = os.path.join(out_dir, 'test')
        os.makedirs(self.test_out_dir, exist_ok=True)
        
    def setup(self, stage: str):
        if stage == "fit":
            dm = self.trainer.datamodule

            if hasattr(dm, "mode_weights"):
                self.mode_weights = dm.mode_weights.to(self.device)
                print("? mode_weights injected into model")
            else:
                raise RuntimeError("datamodule has no mode_weights")  
                
            if self.three_stages:
                # ??????
                self._update_parameter_freezing() 

    def on_train_start(self):
        """ by default lightning executes validation step sanity checks before training starts, so
        it's worth to make sure validation metrics don't store results from these checks. """
        self.val_loss.reset()
        if self.three_stages:
            print("\n" + "="*60)
            print(f"Starting {self.current_stage.upper()} Training")
            print(f"Stage 1: {self.stage_configs['stage1']['epochs']} epochs "
                  f"(Modal classification only)")
            print(f"Stage 2: {self.stage_configs['stage2']['epochs']} epochs "
                  f"(Trajectory decoder only)")
            print(f"Stage 3: {self.stage_configs['stage3']['epochs']} epochs "
                  f"(Joint fine-tuning)")
            print("="*60 + "\n")
            
    def on_train_epoch_start(self):
        """??epoch???????"""
        if not self.three_stages:
        # ?????????,????
            return
        
        total_epoch = self.current_epoch
        
        # ??????
        if total_epoch < self.stage_boundaries['stage1']:
            new_stage = 'stage1'
            self.epoch_in_stage = total_epoch
        elif total_epoch < self.stage_boundaries['stage2']:
            new_stage = 'stage2'
            self.epoch_in_stage = total_epoch - self.stage_boundaries['stage1']
        else:
            new_stage = 'stage3'
            self.epoch_in_stage = total_epoch - self.stage_boundaries['stage2']
        
        # ??????,???????????
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            self.epoch_in_stage = 0
            
            print("\n" + "="*60)
            print(f"Switching to {self.current_stage.upper()} Training")
            print(f"Epoch {total_epoch+1}/{self.total_epochs}")
            
            stage_cfg = self.stage_configs[self.current_stage]
            print(f"Mode loss weight: {stage_cfg['lambda_mode']}")
            print(f"Traj loss weight: {stage_cfg['lambda_marginal']}")
            print(f"Use true modes: {stage_cfg['use_true_modes']}")
            print("="*60 + "\n")
            
            self._update_parameter_freezing()
            
    def _update_parameter_freezing(self):
        """
        Update parameter freezing according to the current training stage.
        Backbone remains trainable; only task-specific heads are frozen/unfrozen.
        """
        if not self.three_stages:
            return
    
        stage_cfg = self.stage_configs[self.current_stage]
    
        # ?? Step 1: Unfreeze all parameters first
        # This avoids accidentally keeping parameters frozen from a previous stage
        for p in self.net.parameters():
            p.requires_grad = True
    
        # ?? Step 2: Freeze mode-related components if required
        if stage_cfg.get("freeze_mode", False):
            for p in self.net.mode_classifier.parameters():
                p.requires_grad = False
            for p in self.net.mode_embedding.parameters():
                p.requires_grad = False
    
        # ?? Step 3: Freeze trajectory-related components if required
        if stage_cfg.get("freeze_traj", False):
            for p in self.net.trajectory_fusion.parameters():
                p.requires_grad = False
            for p in self.net.decoder_head.parameters():
                p.requires_grad = False
    
        # ?? Debug: print number of trainable parameters
        trainable = sum(p.requires_grad for p in self.net.parameters())
        total = sum(1 for _ in self.net.parameters())
        print(f"[{self.current_stage}] Trainable params: {trainable}/{total}")


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
        Y_mode = batch['scene_dict']['rule_based_encoding']
        X = torch.zeros_like(Y).type(torch.float)
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]
        # init_states = self._extract_initial_states(X)
        Y = Y[..., :4]
        X = X[..., :4]
        # -----------------------------------------
        # TODO: incorporate heading prediction
        B, N, T, D = Y.shape
        Y = Y[..., G.REL_XYZ[:D]]
        context = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']

        # TODO: address attention-based masking
        # mode_probs, traj_mu, traj_sigma= self.net(
        #     X, context=context, adjacency=adjacency,
        #     mask=None,
        # )

        mode_probs, traj_mu, traj_sigma = self.net(
            X, context=context, adjacency=adjacency,
            mask=None,
        )

        # traj_mu, traj_sigma = self._acceleration_to_trajectory(init_states, accel_mu, accel_sigma)
        
        if self.three_stages:
            # ????????
            stage_cfg = self.stage_configs[self.current_stage]
            
            # ========== ??????? ==========
            if stage_cfg['lambda_mode'] > 0:
                # ??????
                if Y_mode is not None:
                    mode_loss = F.cross_entropy(
                        mode_probs.view(-1, mode_probs.shape[-1]),
                        self._encode_rule_based_to_mode_index(Y_mode).view(-1),
                        weight=self.mode_weights,
                        label_smoothing=0.1,
                        reduction='mean'
                    )
                else:
                    mode_loss = torch.tensor(0.0).to(self.device)
            else:
                mode_loss = torch.tensor(0.0).to(self.device)
            
            if stage_cfg['lambda_marginal'] > 0:
                # ??????
                if stage_cfg.get('use_true_modes', True) and Y_mode is not None:
                    # ??2:??????
                    traj_loss = self._compute_trajectory_loss_with_true_modes(
                        traj_mu, traj_sigma, Y, Y_mode, ego_agent, masks
                    )
                else:
                    # ??1?3:??????
                    traj_loss = self._compute_trajectory_loss_weighted(
                        traj_mu, traj_sigma, mode_probs, Y, ego_agent, masks
                    )
            else:
                traj_loss = torch.tensor(0.0).to(self.device)
            
            loss_cls = mode_loss
            loss_reg = traj_loss
            
            # ???
            loss = (
                stage_cfg['lambda_mode'] * mode_loss + 
                stage_cfg['lambda_marginal'] * traj_loss
            )
            
        else:

            loss, loss_cls, loss_reg = self.compute_loss(
                traj_mu, traj_sigma, mode_probs, self.mode_weights, Y, Y_mode, ego_agent=ego_agent, epoch =self.current_epoch+1,
                agent_mask=batch['scene_dict']['agent_masks'],
            )
        # print("loss:", loss)


        if plot:
            print("plot")
            predictions = (pred_scores, traj_mu, traj_sigma)
            plot_scene_batch(
                self.eparams.asset_dir, batch, predictions, self.hist_len, self.geodesic, tag,
                out_dir, self.eparams.propagation
            )
            
        traj_mu = traj_mu.squeeze(-1)        # (B, A, T, D, M)
        traj_sigma = traj_sigma.squeeze(-1)  # (B, A, T, D, M)
        
        traj_mu = traj_mu.permute(0, 1, 2, 4, 3)        # (B, A, T, M, D)
        traj_sigma = traj_sigma.permute(0, 1, 2, 4, 3)  # (B, A, T, M, D)

        return loss, loss_cls, loss_reg, mode_probs, traj_mu, traj_sigma, Y[:, :, self.hist_len:, :], Y_mode
        
    def _compute_trajectory_loss_with_true_modes(
        self, mu, sigma, Y, Y_mode, ego_agent, masks
    ):
        """
        Compute Gaussian NLL loss using the ground-truth mode.
        Loss is computed ONLY over future timesteps.
        
        mu:    (B, A, T, D, M, H)
        sigma: (B, A, T, D, M, H)
        Y:     (B, A, T, D)   # history + future
        masks: (B, A, T)
        """
    
        # Encode rule-based mode index
        true_mode_idx = self._encode_rule_based_to_mode_index(Y_mode)
    
        # Separate ego agent
        mu_ego = separate_ego_agent(mu, ego_agent)
        sigma_ego = separate_ego_agent(sigma, ego_agent)
        Y_ego = separate_ego_agent(Y, ego_agent)
        true_mode_idx_ego = separate_ego_agent(true_mode_idx, ego_agent)
        mask_ego = separate_ego_agent(masks, ego_agent)
    
        B, A, T_total, D, M, H = mu_ego.shape
        T_hist = self.hist_len
        T_future = T_total - T_hist
    
        # -------------------------
        # 1?? Select true mode
        # -------------------------
        mode_mask = F.one_hot(true_mode_idx_ego, num_classes=M)  # (B, A, M)
        mode_mask = mode_mask.view(B, A, 1, 1, M, 1)
    
        mu_true = (mu_ego * mode_mask).sum(dim=4)      # (B, A, T, D, H)
        sigma_true = (sigma_ego * mode_mask).sum(dim=4)
    
        # Average over multi-head dimension H
        mu_true = mu_true.mean(dim=-1)                 # (B, A, T, D)
        sigma_true = sigma_true.mean(dim=-1)           # (B, A, T, D)
    
        # -------------------------
        # 2?? Slice future only
        # -------------------------
        mu_future = mu_true[:, :, T_hist:]             # (B, A, 20, D)
        sigma_future = sigma_true[:, :, T_hist:]
        Y_future = Y_ego[:, :, T_hist:]
        mask_future = mask_ego[:, :, T_hist:]          # (B, A, 20)
    
        # Safety check
        assert mu_future.shape[2] == T_future, \
            f"Future length mismatch: {mu_future.shape[2]} vs {T_future}"
    
        # -------------------------
        # 3?? Compute Gaussian NLL
        # -------------------------
        loss = F.gaussian_nll_loss(
            mu_future,
            Y_future,
            sigma_future.clamp_min(1e-4) ** 2,
            reduction='none'
        )  # (B, A, 20, D)
    
        # -------------------------
        # 4?? Apply mask
        # -------------------------
        mask_future = mask_future.unsqueeze(-1)  # (B, A, 20, 1)
    
        loss = (loss * mask_future).sum() / mask_future.sum().clamp_min(1)
    
        return loss

        
    def _encode_rule_based_to_mode_index(self, rule_based: torch.Tensor) -> torch.Tensor:
        """???????????(?model_step???)"""
        turn_idx = rule_based[..., :4].float().argmax(dim=-1)
        speed_idx = rule_based[..., 4:].float().argmax(dim=-1)
        mode_idx = turn_idx * 4 + speed_idx
        return mode_idx.long()
    
    def _compute_trajectory_loss_weighted(self, mu, sigma, mode_probs, Y, ego_agent, masks):
        """????????????(????)"""
        # ??????????compute_loss????
        # ???????compute_loss??
        return self.compute_loss(
            mu, sigma, mode_probs, self.mode_weights, Y, 
            target_mode=None, ego_agent=ego_agent, epoch=self.current_epoch+1,
            agent_mask=masks, lambda_mode=0.0, lambda_marginal=1.0
        )

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

        # Compute last velocity using the last two position points
        # if self.hist_len >= 2:
        # last_vel = (x[:, :, self.hist_len-1, :D] - x[:, :, self.hist_len-2, :D])  # [B, A, 3]
        # else:
        # last_vel = torch.zeros_like(last_pos)

        # Combine position and velocity into initial state
        initial_states = torch.cat([last_pos, last_vel], dim=-1)  # [B, A, 6]

        return initial_states

    def _acceleration_to_trajectory(
        self,
        initial_states: torch.Tensor,   # [B, A, 6]
        accel_mu: torch.Tensor,          # [B, A, T, D, M, 1]
        accel_sigma: torch.Tensor,       # [B, A, T, D, M, 1]
        dt: float = 1.0
    ):
        """
        Acceleration -> trajectory (mode-aware, H=1)
    
        Returns:
            traj_mu    : [B, A, T, D, M, 1]
            traj_sigma : [B, A, T, D, M, 1]
        """
    
        B, A, T, D, M, H = accel_mu.shape
        assert H == 1, "This function assumes H=1"
    
        device = accel_mu.device
    
        # ------------------------------------------------
        # Initial velocity & position
        # ------------------------------------------------
        v0 = initial_states[:, :, 3:3 + D]    # [B, A, D]
        x0 = initial_states[:, :, :D]          # [B, A, D]
    
        # Expand to [B, A, 1, D, M, 1]
        v0 = v0[:, :, None, :, None, None].expand(B, A, 1, D, M, 1)
        x0 = x0[:, :, None, :, None, None].expand(B, A, 1, D, M, 1)
    
        # ------------------------------------------------
        # Velocity integration
        # v_t = v0 + sum_{k<t} a_k
        # ------------------------------------------------
        accel_cumsum = torch.cumsum(accel_mu * dt, dim=2)  # [B,A,T,D,M,1]
    
        vel_mu = torch.cat(
            [
                v0,
                v0 + accel_cumsum[:, :, :-1]
            ],
            dim=2
        )  # [B,A,T,D,M,1]
    
        # ------------------------------------------------
        # Position integration
        # x_t = x0 + sum (v_t + 0.5 a_t)
        # ------------------------------------------------
        traj_increment = vel_mu + 0.5 * accel_mu * dt
        traj_mu = x0 + torch.cumsum(traj_increment * dt, dim=2)
    
        # ------------------------------------------------
        # Uncertainty (no accumulation, same as before)
        # ------------------------------------------------
        traj_sigma = accel_sigma ** 2 / 2
    
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
        loss, loss_cls, loss_reg, _, _, _, _,_ = self.model_step(batch)
        self.train_loss(loss)
        self.train_loss_cls(loss_cls)
        self.train_loss_reg(loss_reg)
        
        if self.three_stages:
            # ??????
            stage_cfg = self.stage_configs[self.current_stage]
            self.log(f"stage/{self.current_stage}", float(self.current_stage[-1]), on_step=False, on_epoch=True)
            self.log(f"lambda/mode", stage_cfg['lambda_mode'], on_step=False, on_epoch=True)
            self.log(f"lambda/marginal", stage_cfg['lambda_marginal'], on_step=False, on_epoch=True)
        
        self.log("losses/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_cls/train", self.train_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_reg/train", self.train_loss_reg, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    
    def validation_step(self, batch: Any, batch_idx: int):
        """ Performs a model step on a validation batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.
        """
        plot = self.eparams.plot_val \
            if self.current_epoch >= self.eparams.plot_after_n_epochs \
            and (batch_idx+1) % self.eparams.plot_every_n == 0 else False

        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, loss_cls, loss_reg, pred_scores, mu, sigma, fut_rel, mode_real = self.model_step(batch, plot, tag, self.val_out_dir)

        # Separate ego agent prediction
        if self.eparams.propagation == 'marginal':
            ego_agent = batch['scene_dict']['ego_agent_id']
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            ego_mode_real = separate_ego_agent(mode_real, ego_agent)
            mask = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError
            
        # Mode prediction accuracy
        mode_real_encoded = self._encode_rule_based_to_mode_index(ego_mode_real)  # [B, A]
        mask_agent_level = mask.any(dim=-1).float() if mask is not None else None  # [B, A]
        
        self.val_modal_accumulator.update(
            ego_pred_scores,  # ????????
            mode_real_encoded,  # ??????????
            mask_agent_level  # ??agent???mask
        )
        
        self.val_merged_modal_accumulator.update(
            ego_pred_scores,  # ????????
            mode_real_encoded,  # ??????????
            mask_agent_level  # ??agent???mask
        )
        
        mode_acc = self.compute_mode_accuracy(ego_pred_scores, mode_real_encoded, mask_agent_level)
        self.val_mode_acc(mode_acc)
        self.log("val_mode_acc", self.val_mode_acc, on_step=False, on_epoch=True, prog_bar=True)
        # ??????????
        for mode_idx in self.valid_modes:
            # ??ground truth???????
            mode_mask = (mode_real_encoded == mode_idx).float() * mask_agent_level
            if mode_mask.sum() > 0:
                # ???????????
                mode_pred = ego_pred_scores[mode_mask.bool()]
                mode_target = mode_real_encoded[mode_mask.bool()]
                mode_acc_per_mode = self.compute_mode_accuracy(mode_pred, mode_target, None)
                
                mode_name = self.mode_names[mode_idx]
                self.val_per_mode_acc[mode_name](mode_acc_per_mode)
                self.log(
                    f"val_per_mode_acc/{mode_name}", 
                    self.val_per_mode_acc[mode_name], 
                    on_step=False, on_epoch=True, prog_bar=False
                )

        self.val_loss(loss)
        self.val_loss_cls(loss_cls)
        self.val_loss_reg(loss_reg)
        self.log("losses/val", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_cls/val", self.val_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_reg/val", self.val_loss_reg, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t = ego_mu[:, :, :self.hist_len+t]
            mask_t = mask[:, :, :self.hist_len+t]
            fut_t = ego_fut[:, :, :t]

            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"val_ade/{key}", self.val_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.val_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"val_fde/{key}", self.val_fde[key], on_step=False, on_epoch=True, prog_bar=True)
            
            # NEW: NLL and Prob-RMSE for this horizon
            nll_t = self.compute_nll(mu_t, ego_sigma[:, :, :self.hist_len+t], ego_pred_scores, fut_t, mask_t)
            rmse_prob_t = self.compute_prob_rmse(mu_t, ego_pred_scores, fut_t, mask_t)
            
            self.val_nll[key](nll_t)
            self.val_prob_rmse[key](rmse_prob_t)
            self.log(f"val_nll/{key}", self.val_nll[key], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_prob_rmse/{key}", self.val_prob_rmse[key], on_step=False, on_epoch=True, prog_bar=True)

        # self.val_prob_ade(self.prob_ade(ego_mu, ego_pred_scores, ego_fut, mask=mask))
        # self.log("val/prob_ade", self.val_prob_ade, on_step=False, on_epoch=True, prog_bar=True)

        # self.val_prob_fde(self.prob_fde(ego_mu, ego_pred_scores, ego_fut, mask=mask))
        # self.log("val/prob_fde", self.val_prob_fde, on_step=False, on_epoch=True, prog_bar=True)

        if len(self.seen_airports) > 1:
            airport_ids = batch['scene_dict']['airport_id']
            for airport in self.seen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]

                for t in self.pred_lens:
                    mu_t = airport_mu[:, :, :self.hist_len+t]
                    fut_t = airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len+t]

                    key = f"{airport}_t={t}"
                    self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"val_seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True,
                        prog_bar=True)

                    self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"val_seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True,
                        prog_bar=True)
                        
    def on_validation_epoch_end(self):
        """???validation epoch?????????"""
        modal_metrics = self.val_modal_accumulator.compute(use_merged=False)
        
        # ??????
        self.log("val/modal_accuracy", modal_metrics['accuracy'], 
                 on_epoch=True, prog_bar=True)
        self.log("val/modal_macro_f1", modal_metrics['macro_f1'], 
                 on_epoch=True, prog_bar=True)
        self.log("val/modal_weighted_f1", modal_metrics['weighted_f1'], 
                 on_epoch=True, prog_bar=True)
        
        # ???????????
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
        
        # ??:??????
        if self.current_epoch % 5 == 0:
            self.val_modal_accumulator.print_report(title=f"Validation Epoch {self.current_epoch}")
            # ??????(?10?epoch)
            if self.current_epoch % 10 == 0:
                self.val_modal_accumulator.print_confusion_matrix(
                    title=f"Validation Confusion Matrix (Epoch {self.current_epoch})"
                )
        
        merged_metrics = self.val_merged_modal_accumulator.compute(use_merged=True)
        
        # ??????
        self.log("val/merged_modal_accuracy", merged_metrics['merged_accuracy'],
                 on_epoch=True, prog_bar=True)
        self.log("val/merged_modal_macro_f1", merged_metrics['merged_macro_f1'],
                 on_epoch=True, prog_bar=True)
        self.log("val/merged_modal_weighted_f1", merged_metrics['merged_weighted_f1'],
                 on_epoch=True, prog_bar=True)
        
        # ???????????
        turn_names = {0: "Left", 1: "Right", 2: "Straight", 3: "Hold"}
        for turn_idx in range(4):
            turn_name = turn_names[turn_idx]
            self.log(f"val/merged_modal_precision/{turn_name}",
                     merged_metrics['merged_precision_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"val/merged_modal_recall/{turn_name}",
                     merged_metrics['merged_recall_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"val/merged_modal_f1/{turn_name}",
                     merged_metrics['merged_f1_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"val/merged_modal_support/{turn_name}",
                     merged_metrics['merged_support_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
        
        # ??:??????
        if self.current_epoch % 5 == 0:
            self.val_merged_modal_accumulator.print_report(title=f"Validation Epoch {self.current_epoch}")
            # ??????(?10?epoch)
            if self.current_epoch % 10 == 0:
                self.val_merged_modal_accumulator.print_confusion_matrix(
                    title=f"Validation Confusion Matrix (Epoch {self.current_epoch})"
                )
        
        # ?????
        self.val_modal_accumulator.reset()
        self.val_merged_modal_accumulator.reset()
    
    
    def on_test_epoch_end(self):
        """???test epoch?????????"""
        modal_metrics = self.test_modal_accumulator.compute(use_merged=False)
        
        # ??????
        self.log("test/modal_accuracy", modal_metrics['accuracy'], 
                 on_epoch=True, prog_bar=True)
        self.log("test/modal_macro_f1", modal_metrics['macro_f1'], 
                 on_epoch=True, prog_bar=True)
        self.log("test/modal_weighted_f1", modal_metrics['weighted_f1'], 
                 on_epoch=True, prog_bar=True)
        
        # ???????????
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
        
        # ??????
        print("\n" + "="*100)
        print("TEST SET FINAL RESULTS")
        print("="*100)
        self.test_modal_accumulator.print_report(title="TEST SET MODAL CLASSIFICATION")
        self.test_modal_accumulator.print_confusion_matrix(title="TEST SET CONFUSION MATRIX")
        
        merged_metrics = self.test_merged_modal_accumulator.compute(use_merged=True)
        
        # ??????
        self.log("test/merged_modal_accuracy", merged_metrics['merged_accuracy'],
                 on_epoch=True, prog_bar=True)
        self.log("test/merged_modal_macro_f1", merged_metrics['merged_macro_f1'],
                 on_epoch=True, prog_bar=True)
        self.log("test/merged_modal_weighted_f1", merged_metrics['merged_weighted_f1'],
                 on_epoch=True, prog_bar=True)
        
        # ???????????
        turn_names = {0: "Left", 1: "Right", 2: "Straight", 3: "Hold"}
        for turn_idx in range(4):
            turn_name = self.mode_names[turn_idx]
            self.log(f"test/merged_modal_precision/{turn_name}",
                     merged_metrics['merged_precision_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"test/merged_modal_recall/{turn_name}",
                     merged_metrics['merged_recall_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"test/merged_modal_f1/{turn_name}",
                     merged_metrics['merged_f1_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
            self.log(f"test/merged_modal_support/{turn_name}",
                     merged_metrics['merged_support_per_mode'][turn_idx],
                     on_epoch=True, prog_bar=False)
        
        # ??????
        print("\n" + "="*100)
        print("TEST SET FINAL RESULTS")
        print("="*100)
        self.test_merged_modal_accumulator.print_report(title="TEST SET MERGED MODAL CLASSIFICATION")
        self.test_merged_modal_accumulator.print_confusion_matrix(title="TEST SET CONFUSION MATRIX")
        
        # ?????
        self.test_modal_accumulator.reset()
        self.test_merged_modal_accumulator.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """ Performs a model step on a test batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.
        """
        # plot = self.eparams.plot_test if (batch_idx+1) % self.eparams.plot_every_n == 0 else False
        plot = self.eparams.plot_test if (batch_idx + 1) % 10 == 0 else False

        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, loss_cls, loss_reg, pred_scores, mu, sigma, fut_rel, mode_real = self.model_step(batch, plot, tag, self.test_out_dir)
        ego_agent = batch['scene_dict']['ego_agent_id']

        if self.eparams.propagation == 'marginal':
            # Separate ego agent prediction
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            ego_mode_real = separate_ego_agent(mode_real, ego_agent)
            mask = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError
            
        # Mode prediction accuracy
        mode_real_encoded = self._encode_rule_based_to_mode_index(ego_mode_real)  # [B, A]
        mask_agent_level = mask.any(dim=-1).float() if mask is not None else None  # [B, A]
        
        self.test_modal_accumulator.update(
            ego_pred_scores,
            mode_real_encoded,
            mask_agent_level
        )
        
        self.test_merged_modal_accumulator.update(
          ego_pred_scores,
          mode_real_encoded,
          mask_agent_level
      )
        
        mode_acc = self.compute_mode_accuracy(ego_pred_scores, mode_real_encoded, mask_agent_level)
        self.test_mode_acc(mode_acc)
        self.log("test_mode_acc", self.test_mode_acc, on_step=False, on_epoch=True, prog_bar=True)
        # ??????????
        for mode_idx in self.valid_modes:
            # ??ground truth???????
            mode_mask = (mode_real_encoded == mode_idx).float() * mask_agent_level
            if mode_mask.sum() > 0:
                # ???????????
                mode_pred = ego_pred_scores[mode_mask.bool()]
                mode_target = mode_real_encoded[mode_mask.bool()]
                mode_acc_per_mode = self.compute_mode_accuracy(mode_pred, mode_target, None)
                
                mode_name = self.mode_names[mode_idx]
                self.test_per_mode_acc[mode_name](mode_acc_per_mode)
                self.log(
                    f"test_per_mode_acc/{mode_name}", 
                    self.test_per_mode_acc[mode_name], 
                    on_step=False, on_epoch=True, prog_bar=False
                )

        self.test_loss(loss)
        self.test_loss_cls(loss_cls)
        self.test_loss_reg(loss_reg)
        self.log("losses/test", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_cls/test", self.test_loss_cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("losses_reg/test", self.test_loss_reg, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t, fut_t = ego_mu[:, :, :self.hist_len+t], ego_fut[:, :, :t]
            mask_t = mask[:, :, :self.hist_len+t]

            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(
                f"test_ade/{t}", self.test_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.test_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(
                f"test_fde/{t}", self.test_fde[key], on_step=False, on_epoch=True, prog_bar=True)
                
            # NEW: NLL and Prob-RMSE for this horizon
            nll_t = self.compute_nll(mu_t, ego_sigma[:, :, :self.hist_len+t], ego_pred_scores, fut_t, mask_t)
            rmse_prob_t = self.compute_prob_rmse(mu_t, ego_pred_scores, fut_t, mask_t)
            
            self.test_nll[key](nll_t)
            self.test_prob_rmse[key](rmse_prob_t)
            self.log(f"test_nll/{key}", self.test_nll[key], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test_prob_rmse/{key}", self.test_prob_rmse[key], on_step=False, on_epoch=True, prog_bar=True)

        # self.test_prob_ade(self.prob_ade(ego_mu, ego_pred_scores, ego_fut))
        # self.log("test/prob_ade", self.test_prob_ade, on_step=False, on_epoch=True, prog_bar=True)

        # self.test_prob_fde(self.prob_fde(ego_mu, ego_pred_scores, ego_fut))
        # self.log("test/prob_fde", self.test_prob_fde, on_step=False, on_epoch=True, prog_bar=True)

        airport_ids = batch['scene_dict']['airport_id']
        for airport in self.seen_airports:
            airport_idx = np.where(airport_ids == airport)[0]
            if len(airport_idx) == 0:
                continue
            airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
            airport_mask = mask[airport_idx]

            for t in self.pred_lens:
                mu_t, fut_t = airport_mu[:, :, :self.hist_len+t], airport_fut[:, :, :t]
                mask_t = airport_mask[:, :, :self.hist_len+t]

                key = f"{airport}_t={t}"
                self.test_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                self.log(
                    f"test_seen_ade/{key}", self.test_seen_ade[key], on_step=False, on_epoch=True,
                    prog_bar=True)

                self.test_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                self.log(
                    f"test_seen_fde/{key}", self.test_seen_fde[key], on_step=False, on_epoch=True,
                    prog_bar=True)

        if len(self.unseen_airports) > 0:
            for airport in self.unseen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]

                for t in self.pred_lens:
                    mu_t, fut_t = airport_mu[:, :, :self.hist_len+t], airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len+t]

                    key = f"{airport}_t={t}"
                    self.test_unseen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"test_unseen_ade/{key}", self.test_unseen_ade[key], on_step=False,
                        on_epoch=True, prog_bar=True)

                    self.test_unseen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"test_unseen_fde/{key}", self.test_unseen_fde[key], on_step=False,
                        on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """?????(???????)"""
    
        if not self.three_stages:
            # ????(?????)
            return self._configure_original_optimizer()
        else:
            # ???????
            return self._configure_original_optimizer()

    def _configure_original_optimizer(self):
        """???????????"""
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
                # === NEW: handle LSTM decoder parameters ===
                if "temporal_decoder" in mn:
                    # all LSTM weights and biases go to no weight decay
                    no_decay.add(fpn)
                    continue
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

    # def _configure_three_stage_optimizer(self):
    #     """???????????"""
    #     stage_cfg = self.stage_configs[self.current_stage]
    #
    #     # ?????(???????)
    #     mode_params = []
    #     traj_params = []
    #     other_params = []
    #
    #     # ??????????
    #     for name, param in self.net.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #
    #         if 'mode_classifier' in name or 'mode_embedding' in name:
    #             mode_params.append(param)
    #         elif 'decoder_head' in name or 'trajectory_fusion' in name:
    #             traj_params.append(param)
    #         else:
    #             other_params.append(param)
    #
    #     # ?????(??????????)
    #     optim_groups = []
    #
    #     # ????????
    #     if mode_params and stage_cfg['lr_mode'] > 0:
    #         optim_groups.append({
    #             'params': mode_params,
    #             'lr': stage_cfg['lr_mode'],
    #             'weight_decay': self.hparams.optimizer.weight_decay,
    #             'name': 'mode_params'
    #         })
    #
    #     # ????????
    #     if traj_params and stage_cfg['lr_traj'] > 0:
    #         optim_groups.append({
    #             'params': traj_params,
    #             'lr': stage_cfg['lr_traj'],
    #             'weight_decay': self.hparams.optimizer.weight_decay,
    #             'name': 'traj_params'
    #         })
    #
    #     # ????(?????)
    #     if other_params and stage_cfg['lr_other'] > 0:
    #         optim_groups.append({
    #             'params': other_params,
    #             'lr': stage_cfg['lr_other'],
    #             'weight_decay': self.hparams.optimizer.weight_decay,
    #             'name': 'other_params'
    #         })
    #
    #     # ???????????,??????
    #     if not optim_groups:
    #         print(f"Warning: No trainable parameters in stage {self.current_stage}")
    #         return {"optimizer": torch.optim.Adam([torch.tensor(0.0)], lr=0.0)}
    #
    #     # ?????
    #     optimizer = torch.optim.AdamW(
    #         optim_groups,
    #         lr=self.hparams.optimizer.lr,  # ?????(????????)
    #         betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2),
    #         eps=1e-8
    #     )
    #
    #     # ????????
    #     if self.current_stage == 'stage3' and stage_cfg.get('warmup_epochs', 0) > 0:
    #         # ??3:warmup + cosine??
    #         from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
    #
    #         # Warmup???
    #         warmup_scheduler = LinearLR(
    #             optimizer,
    #             start_factor=0.01,
    #             end_factor=1.0,
    #             total_iters=stage_cfg['warmup_epochs']
    #         )
    #
    #         # Cosine?????
    #         cosine_scheduler = CosineAnnealingLR(
    #             optimizer,
    #             T_max=stage_cfg['epochs'] - stage_cfg['warmup_epochs'],
    #             eta_min=1e-6
    #         )
    #
    #         # ?????:?warmup,?cosine
    #         scheduler = torch.optim.lr_scheduler.SequentialLR(
    #             optimizer,
    #             schedulers=[warmup_scheduler, cosine_scheduler],
    #             milestones=[stage_cfg['warmup_epochs']]
    #         )
    #
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "interval": "epoch",
    #                 "frequency": 1,
    #                 "name": f"stage3_warmup_cosine"
    #             }
    #         }
    #     elif self.hparams.scheduler is not None:
    #         # ??1?2:?????ReduceLROnPlateau
    #         scheduler = self.hparams.scheduler(optimizer=optimizer)
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "monitor": "losses/val",
    #                 "interval": "epoch",
    #                 "frequency": 1,
    #                 "name": f"{self.current_stage}_plateau"
    #             }
    #         }
    #     else:
    #         # ?????
    #         return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = TrajPred(None, None, None)