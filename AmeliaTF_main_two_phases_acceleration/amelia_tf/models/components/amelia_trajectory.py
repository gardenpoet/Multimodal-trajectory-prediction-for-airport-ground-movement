# amelia_tf/models/components/amelia_trajectory.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Any, Tuple, Optional, Dict

from .base_encoder import BaseAmeliaEncoder
from .gmm import GMM


class AmeliaTrajectory(BaseAmeliaEncoder):
    """
    Trajectory prediction network that predicts acceleration and converts to trajectory.
    Supports both fine-grained mode conditioning and turn-level fusion.
    """
    
    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)
        
        self.pred_lens = config.encoder.pred_lens  # List of prediction horizons
        self.hist_len = config.encoder.hist_len
        
        # Decoder configuration
        self.decoder_config = config.decoder
        self.num_modes = self.decoder_config.num_modes  # TurnLeft, TurnRight, Straight, Hold
        
        # Configuration for mode embedding
        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': self.num_modes,
            'mode_embed_dim': 32,
            'dropout': 0.1,
        }))
        
        # Learnable embedding for turn-level modes
        self.turn_embedding = nn.Embedding(
            self.num_modes,
            self.mode_config.mode_embed_dim
        )
        
        # Fusion network to combine encoded features with mode embeddings
        self.trajectory_fusion = nn.Sequential(
            nn.Linear(self.embed_size + self.mode_config.mode_embed_dim, self.embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_size, self.embed_size)
        )
        
        # GMM decoder head for acceleration prediction
        # Output: acceleration means and stds for each future timestep
        self.decoder_config.in_size = self.embed_size
        self.accel_head = GMM(self.decoder_config)
        
        # Time step (1 second per step)
        self.dt = 1.0
        
        print("Trajectory model parameters: %.2fM" % (self.get_num_params()/1e6,))
        print(f"  - Turn-level modes: {self.num_modes}")
        print(f"  - Prediction horizons: {self.pred_lens}")
    
    @property
    def num_dec_heads(self) -> int:
        return self.accel_head.num_futures
    
    def _prepare_decoder_input(self, encoded: torch.Tensor, mode_emb: torch.Tensor) -> torch.Tensor:
        """Prepare input for acceleration decoder."""
        B, A, T, embed_size = encoded.shape
        
        # Expand mode embedding to match time dimension
        mode_emb_expanded = mode_emb.unsqueeze(2).unsqueeze(2)  # (B, A, 1, 1, mode_embed_dim)
        mode_emb_expanded = mode_emb_expanded.expand(-1, -1, T, -1, -1)
        
        # Concatenate and fuse
        encoded_exp = encoded.unsqueeze(-2)  # (B, A, T, 1, embed_size)
        decoder_input = torch.cat([encoded_exp, mode_emb_expanded], dim=-1)
        decoder_input = self.trajectory_fusion(decoder_input)
        decoder_input = encoded_exp + decoder_input  # residual connection
        
        return decoder_input
    
    def _acceleration_to_trajectory(
        self,
        initial_states: torch.Tensor,
        accel_mu: torch.Tensor,
        accel_sigma: torch.Tensor,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert predicted acceleration to trajectory using semi-implicit integration.
        
        Args:
            initial_states: [B, A, 6] = [x, y, z, vx, vy, vz] at last observed time step
            accel_mu:       [B, A, T_pred, H, 3] predicted acceleration means
            accel_sigma:    [B, A, T_pred, H, 3] predicted acceleration standard deviations
            dt:             Time step interval (default: 1.0 second)
            
        Returns:
            traj_mu:    [B, A, T_pred, H, 3] predicted trajectory means
            traj_sigma: [B, A, T_pred, H, 3] predicted trajectory standard deviations
        """
        B, A, T_pred, H, D = accel_mu.shape
        
        # Extract initial velocity and position
        v0 = initial_states[:, :, None, None, 3:].expand(B, A, 1, H, D)  # [B, A, 1, H, 3]
        x0 = initial_states[:, :, None, None, :3].expand(B, A, 1, H, D)   # [B, A, 1, H, 3]
        
        # Compute velocity: v_t = v_0 + S a_i * dt (cumulative sum of accelerations)
        # First step uses v0, subsequent steps use cumulative sum
        vel_mu = torch.cat([
            v0,
            v0 + torch.cumsum(accel_mu * dt, dim=2)[:, :, :-1]
        ], dim=2)  # [B, A, T_pred, H, 3]
        
        # Position update: x_{t+1} = x_t + v_t * dt + 0.5 * a_t * dt^2
        # This is semi-implicit: uses current velocity and acceleration
        traj_increment = vel_mu * dt + 0.5 * (accel_mu * dt ** 2)
        
        # Cumulative sum to get positions
        traj_mu = x0 + torch.cumsum(traj_increment, dim=2)  # [B, A, T_pred, H, 3]
        
        # Trajectory uncertainty propagation
        # Using simplified approximation: s_traj ˜ s_acc ² / 2 (variance accumulates)
        # This is a conservative estimate
        traj_sigma = accel_sigma ** 2 / 2  # [B, A, T_pred, H, 3]
        
        return traj_mu, traj_sigma
    
    def forward(
        self, 
        x: torch.Tensor, 
        context=None, 
        adjacency=None, 
        mask=None, 
        mode_probs: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified mode conditioning, predicting acceleration then trajectory.
    
        Args:
            x: Input trajectory history (B, A, T, D)
            context: Scene context features
            adjacency: Agent interaction graph
            mask: Optional attention masks
            mode_probs: Mode probability distribution (B, A, num_modes).
                        Accepts one-hot (teacher forcing / argmax) or soft probabilities.
            initial_states: Initial states [B, A, 6] = [x, y, z, vx, vy, vz] at last observed step.
                           If None, will compute from input trajectory.
    
        Returns:
            traj_mu:    Predicted trajectory means  (B, A, T_pred, D)
            traj_sigma: Predicted trajectory stds   (B, A, T_pred, D)
        """
        B, A, T, _ = x.shape
        T_pred = max(self.pred_lens)
        H = self.accel_head.num_futures
    
        if mode_probs is None:
            raise ValueError("Must provide mode_probs (one-hot or soft)")
        
        # Compute initial states if not provided
        if initial_states is None:
            # Extract last observed position (x, y, z)
            last_pos = x[:, :, self.hist_len - 1, :3]  # [B, A, 3]
            
            # Compute velocity from last two observed positions
            if self.hist_len >= 2:
                last_vel = x[:, :, self.hist_len - 1, :3] - x[:, :, self.hist_len - 2, :3]  # [B, A, 3]
            else:
                last_vel = torch.zeros_like(last_pos)
            
            initial_states = torch.cat([last_pos, last_vel], dim=-1)  # [B, A, 6]
    
        # Encode trajectory history -> (B, A, T, embed_size)
        encoded = super().forward(x, context, adjacency, mask)
    
        # Compute mode embedding via weighted sum of turn embeddings
        # Works for both one-hot and soft probabilities
        mode_emb = torch.matmul(mode_probs, self.turn_embedding.weight)  # [B, A, mode_embed_dim]
    
        # Fuse encoded features with mode embedding -> (B, A, T, 1, embed_size)
        decoder_input = self._prepare_decoder_input(encoded, mode_emb)
    
        # Predict acceleration means and stds for all future timesteps
        accel_mu, accel_sigma = self.accel_head(decoder_input)  # [B, A, T_pred, H, D_out]
        
    
        # Convert acceleration to trajectory
        traj_mu, traj_sigma = self._acceleration_to_trajectory(
            initial_states, accel_mu, accel_sigma, dt=self.dt
        )
    
        # Squeeze mode dimension (H should be 1 for single-mode prediction)
        # The network is designed for single-mode per forward pass
        traj_mu = traj_mu.squeeze(3)      # [B, A, T_pred, 3]
        traj_sigma = traj_sigma.squeeze(3)  # [B, A, T_pred, 3]
        # traj_sigma = torch.clamp(traj_sigma, min=0.001, max=1.0)
    
        return traj_mu, traj_sigma
    
    def get_initial_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract initial states from input trajectory for external use.
        
        Args:
            x: Input trajectory history (B, A, T, D)
            
        Returns:
            initial_states: [B, A, 6] = [x, y, z, vx, vy, vz]
        """
        B, A, T, _ = x.shape
        
        # Last observed position
        last_pos = x[:, :, self.hist_len - 1, :3]  # [B, A, 3]
        
        # Compute velocity from last two observed positions
        if self.hist_len >= 2:
            last_vel = x[:, :, self.hist_len - 1, :3] - x[:, :, self.hist_len - 2, :3]  # [B, A, 3]
        else:
            last_vel = torch.zeros_like(last_pos)
        
        return torch.cat([last_pos, last_vel], dim=-1)