# amelia_tf/models/components/amelia_mode.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Any, Tuple, Optional

from .base_encoder import BaseAmeliaEncoder
from amelia_tf.utils.modes import TURN_MODE_MAP, VALID_TURN_MODES


class AmeliaMode(BaseAmeliaEncoder):
    
    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)
        
        self.num_modes = len(TURN_MODE_MAP)
        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': self.num_modes,         
            'hidden_dim': 64,
            'dropout': 0.1,
        }))
        
        # No illegal modes needed for 4-class turn-level
        self.mode_config.illegal_mode_mask = torch.zeros(self.num_modes, dtype=torch.bool)
        
        self.mode_classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.mode_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.mode_config.dropout),
            nn.Linear(self.mode_config.hidden_dim, self.num_modes)
        )
        
        print("Mode model parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def forward(
        self, 
        x: torch.Tensor, 
        context=None, 
        adjacency=None, 
        mask=None, 
        output_mode_only: bool = True
    ) -> torch.Tensor:
        
    
        encoded = super().forward(x, context, adjacency, mask)  # (B, A, T, embed_size)
        
        last_features = encoded[:, :, -1, :]  # (B, A, embed_size)
        
        mode_logits = self.mode_classifier(last_features)  # (B, A, num_modes)
        
        if hasattr(self.mode_config, "illegal_mode_mask"):
            illegal_mask = self.mode_config.illegal_mode_mask.to(x.device).view(1, 1, -1)
            mode_logits = mode_logits.masked_fill(illegal_mask, -1e9)
        
        if output_mode_only:
            return mode_logits
        else:
            return F.softmax(mode_logits, dim=-1)
    