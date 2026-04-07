# amelia_tf/models/components/amelia_mode.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Any, Tuple, Optional, Union

from .base_encoder import BaseAmeliaEncoder
from amelia_tf.utils.modes import TURN_MODE_MAP, VALID_TURN_MODES


class AmeliaMode(BaseAmeliaEncoder):
    """
    Turn-level mode prediction model.
    Predicts driving modes (TurnLeft, TurnRight, Straight, Hold) from trajectory history.
    Supports multiple temporal pooling strategies.
    """
    
    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)
        
        self.num_modes = len(TURN_MODE_MAP)
        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': self.num_modes,         
            'hidden_dim': 64,
            'dropout': 0.1,
            'pooling': 'last',  # Options: 'last', 'mean', 'max', 'concat', 'attention'
        }))
        
        # No illegal modes needed for 4-class turn-level
        self.mode_config.illegal_mode_mask = torch.zeros(self.num_modes, dtype=torch.bool)
        
        # Get pooling type from config
        self.pooling_type = self.mode_config.get('pooling', 'last')
        
        # Compute classifier input dimension based on pooling type
        if self.pooling_type == 'concat':
            classifier_input_dim = self.embed_size * 3
        else:
            classifier_input_dim = self.embed_size
        
        # Initialize attention layer if needed
        if self.pooling_type == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size // 4),
                nn.Tanh(),
                nn.Linear(self.embed_size // 4, 1)
            )
        
        # Mode classifier head
        self.mode_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.mode_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.mode_config.dropout),
            nn.Linear(self.mode_config.hidden_dim, self.num_modes)
        )
        
        print(f"Mode model (pooling={self.pooling_type}): {self.get_num_params()/1e6:.2f}M parameters")
    
    def _temporal_pooling(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Temporal pooling to aggregate information across time dimension.
        
        Transforms (B, A, T, E) -> (B, A, output_dim) by collapsing the time dimension.
        
        Args:
            encoded: Encoded agent features (Batch, Agents, Time, EmbedSize)
        
        Returns:
            features: Pooled features (Batch, Agents, output_dim)
            
        Pooling strategies:
            - 'last': Take only the last time step. Fastest, but loses historical info.
            - 'mean': Global average pooling. Good for consistent/continuous actions.
            - 'max': Global max pooling. Captures strongest signals (e.g., peak turning).
            - 'concat': Concatenate mean, max, and last. Preserves multiple perspectives.
            - 'attention': Learnable weighted average. Model learns which timesteps matter.
        """
        if self.pooling_type == 'last':
            # Take only the final timestep
            features = encoded[:, :, -1, :]
            
        elif self.pooling_type == 'mean':
            # Global average pooling across time
            features = encoded.mean(dim=2)
            
        elif self.pooling_type == 'max':
            # Global max pooling across time
            features = encoded.max(dim=2)[0]
            
        elif self.pooling_type == 'concat':
            # Concatenate mean, max, and last step features
            mean_pool = encoded.mean(dim=2)
            max_pool = encoded.max(dim=2)[0]
            last_step = encoded[:, :, -1, :]
            features = torch.cat([mean_pool, max_pool, last_step], dim=-1)
            
        elif self.pooling_type == 'attention':
            # Attention-weighted pooling
            # Compute attention scores for each timestep
            attention_weights = self.temporal_attention(encoded)  # (B, A, T, 1)
            attention_weights = attention_weights.squeeze(-1)  # (B, A, T)
            attention_weights = F.softmax(attention_weights, dim=-1)  # Normalize across time
            
            # Weighted sum across time dimension
            features = (encoded * attention_weights.unsqueeze(-1)).sum(dim=2)  # (B, A, E)
            
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return features
    
    def forward(
        self, 
        x: torch.Tensor, 
        context=None, 
        adjacency=None, 
        mask=None, 
        output_mode_only: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for mode prediction.
        
        Args:
            x: Agent trajectory history (B, A, T, D)
            context: Scene context features
            adjacency: Agent interaction graph
            mask: Optional attention mask
            output_mode_only: If True, return logits; if False, return softmax probabilities
        
        Returns:
            mode_logits or mode_probs: (B, A, num_modes)
        """
        # Encode trajectory with context using the backbone encoder
        encoded = super().forward(x, context, adjacency, mask)  # (B, A, T, embed_size)
        
        # Aggregate temporal information via pooling
        features = self._temporal_pooling(encoded)  # (B, A, classifier_input_dim)
        
        # Predict mode logits
        mode_logits = self.mode_classifier(features)  # (B, A, num_modes)
        
        # Apply illegal mode mask if specified (e.g., no U-turn in certain scenarios)
        if hasattr(self.mode_config, "illegal_mode_mask"):
            illegal_mask = self.mode_config.illegal_mode_mask.to(x.device).view(1, 1, -1)
            mode_logits = mode_logits.masked_fill(illegal_mask, -1e9)
        
        if output_mode_only:
            return mode_logits
        else:
            return F.softmax(mode_logits, dim=-1)