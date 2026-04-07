# amelia_tf/models/components/amelia_trajectory.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Any, Tuple, Optional, Dict

from .base_encoder import BaseAmeliaEncoder
from .gmm import GMM
from amelia_tf.utils.modes import MODE_MAP, VALID_MODES
from amelia_tf.utils.metrics import create_turn_only_mapping


class AmeliaTrajectory(BaseAmeliaEncoder):
    """
    Trajectory prediction network that fuses fine-grained modes into turn-only modes.
    Supports both fine-grained mode conditioning and turn-level fusion.
    """
    
    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)
        
        # Decoder configuration
        self.decoder_config = config.decoder
        self.num_modes = self.decoder_config.num_modes  # TurnLeft, TurnRight, Straight, Hold
        
        # Configuration for mode embedding
        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': self.num_modes,
            'mode_embed_dim': 32,
            'dropout': 0.1,
        }))
        
        
        # Learnable embedding for turn-level modes (fused representation)
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
        
        # GMM decoder head
        self.decoder_config.in_size = self.embed_size
        self.decoder_head = GMM(self.decoder_config)
        
        
        print("Trajectory model parameters: %.2fM" % (self.get_num_params()/1e6,))
        print(f"  - Turn-level modes: {self.num_modes}")
    
    @property
    def num_dec_heads(self) -> int:
        return self.decoder_head.num_futures
    
    
    def _prepare_decoder_input(self, encoded: torch.Tensor, mode_emb: torch.Tensor) -> torch.Tensor:
        """Prepare input for GMM decoder."""
        B, A, T, embed_size = encoded.shape
        
        # Expand mode embedding
        mode_emb_expanded = mode_emb.unsqueeze(2).unsqueeze(2)  # (B, A, 1, 1, mode_embed_dim)
        mode_emb_expanded = mode_emb_expanded.expand(-1, -1, T, -1, -1)
        
        # Concatenate and fuse
        encoded_exp = encoded.unsqueeze(-2)  # (B, A, T, 1, embed_size)
        decoder_input = torch.cat([encoded_exp, mode_emb_expanded], dim=-1)
        decoder_input = self.trajectory_fusion(decoder_input)
        decoder_input = encoded_exp + decoder_input  # residual
        
        return decoder_input
    
    def forward(
        self, 
        x: torch.Tensor, 
        context=None, 
        adjacency=None, 
        mask=None, 
        mode_probs: Optional[torch.Tensor] = None  # [B, A, num_modes], one-hot or soft probabilities
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified mode conditioning.
    
        Args:
            x: Input trajectory history (B, A, T, D)
            context: Scene context features
            adjacency: Agent interaction graph
            mask: Optional attention masks
            mode_probs: Mode probability distribution (B, A, num_modes).
                        Accepts one-hot (teacher forcing / argmax) or soft probabilities.
    
        Returns:
            traj_mu:    Predicted trajectory means  (B, A, T, D)
            traj_sigma: Predicted trajectory stds   (B, A, T, D)
        """
        B, A, T, _ = x.shape
        H = self.decoder_head.num_futures
    
        if mode_probs is None:
            raise ValueError("Must provide mode_probs (one-hot or soft)")
    
        # Encode trajectory history -> (B, A, T, embed_size)
        encoded = super().forward(x, context, adjacency, mask)
    
        # Compute mode embedding via weighted sum of turn embeddings.
        # Works for both one-hot (selects a single row) and soft probs (weighted average).
        # (B, A, num_modes) x (num_modes, mode_embed_dim) -> (B, A, mode_embed_dim)
        mode_emb = torch.matmul(mode_probs, self.turn_embedding.weight)
    
        # Fuse encoded features with mode embedding -> (B, A, T, 1, embed_size)
        decoder_input = self._prepare_decoder_input(encoded, mode_emb)
    
        # Decode trajectories
        traj_mu, traj_sigma = self.decoder_head(decoder_input)
    
    
        # Slice out future steps only -> (B, A, T, D)
        traj_mu    = traj_mu.squeeze(3)
        traj_sigma = traj_sigma.squeeze(3)
    
        return traj_mu, traj_sigma