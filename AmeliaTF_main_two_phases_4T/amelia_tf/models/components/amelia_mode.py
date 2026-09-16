# amelia_tf/models/components/amelia_mode.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Optional

from .base_encoder import BaseAmeliaEncoder
from amelia_tf.utils.modes import TURN_MODE_MAP, VALID_TURN_MODES


class AmeliaMode(BaseAmeliaEncoder):
    """
    Turn-level mode prediction model.
    Predicts driving modes (TurnLeft, TurnRight, Straight, Hold) from trajectory history.

    Supports multiple temporal pooling strategies and optional turn feasibility conditioning.

    Turn feasibility conditioning
    -----------------------------
    turn_feasibility is a (B, A, 4) binary float tensor derived from map topology BFS,
    where each of the 4 values indicates whether TurnLeft / TurnRight / Straight / Hold
    is physically reachable from the agent's current position within the prediction horizon.

    Two complementary mechanisms are supported and independently switchable for ablation:

    1. Soft conditioning  (use_feasibility=True):
       The feasibility vector is embedded and concatenated to the pooled trajectory
       features before the classifier head. The model learns to leverage topological
       constraints as a soft prior while retaining the ability to override them when
       trajectory evidence is strong (e.g. noisy map data).

    2. Hard masking  (apply_hard_mask=True):
       Infeasible classes (feasibility == 0) have their logits set to -inf so that
       softmax assigns them zero probability. Recommended at inference when map
       quality is trusted and strict topological constraints are desired.

    Both mechanisms use the 4-class turn-level feasibility vector
    [TurnLeft, TurnRight, Straight, Hold]. Since AmeliaMode predicts
    4 turn classes (not 16), the feasibility mask applies directly
    without any expansion.

    Ablation configurations (set in config or rely on defaults):
        baseline  : use_feasibility=False, apply_hard_mask=False
        soft only : use_feasibility=True,  apply_hard_mask=False  (default)
        soft+hard : use_feasibility=True,  apply_hard_mask=True
        hard only : use_feasibility=False, apply_hard_mask=True

    Default values ensure the model runs without any config file changes.
    """

    # Default config values — used when the key is absent from the config file.
    _DEFAULTS = EasyDict({
        'num_modes':             4,       # overridden by len(TURN_MODE_MAP) at runtime
        'hidden_dim':            64,
        'dropout':               0.1,
        'pooling':               'last',  # 'last' | 'mean' | 'max' | 'concat' | 'attention'
        'use_feasibility':       True,    # enable soft feasibility conditioning
        'apply_hard_mask':       True,   # enable hard logit masking at forward time
        'feasibility_embed_dim': 16,      # embedding size for the 4-dim feasibility vector
    })

    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)

        self.num_modes = len(TURN_MODE_MAP)

        # Merge user config over defaults so missing keys always have a value
        mode_cfg = EasyDict(dict(self._DEFAULTS))
        mode_cfg.update(getattr(config, 'mode_predictor', EasyDict()))
        mode_cfg.num_modes = self.num_modes
        self.mode_config = mode_cfg

        # Convenience attributes read from merged config
        self.pooling_type           = self.mode_config.pooling
        self.use_feasibility        = self.mode_config.use_feasibility
        self.apply_hard_mask        = self.mode_config.apply_hard_mask
        self.feasibility_embed_dim  = self.mode_config.feasibility_embed_dim

        # Base classifier input dimension determined by pooling strategy
        if self.pooling_type == 'concat':
            base_input_dim = self.embed_size * 3
        else:
            base_input_dim = self.embed_size

        # Soft feasibility conditioning: embed 4-dim binary vector, concat to features
        if self.use_feasibility:
            self.feasibility_embed = nn.Sequential(
                nn.Linear(4, self.feasibility_embed_dim),
                nn.ReLU(),
            )
            classifier_input_dim = base_input_dim + self.feasibility_embed_dim
        else:
            classifier_input_dim = base_input_dim

        # Attention module for 'attention' pooling strategy
        if self.pooling_type == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size // 4),
                nn.Tanh(),
                nn.Linear(self.embed_size // 4, 1),
            )

        # Mode classifier head
        self.mode_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.mode_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.mode_config.dropout),
            nn.Linear(self.mode_config.hidden_dim, self.num_modes),
        )

        print(
            f"Mode model ("
            f"pooling={self.pooling_type}, "
            f"use_feasibility={self.use_feasibility}, "
            f"apply_hard_mask={self.apply_hard_mask}"
            f"): {self.get_num_params()/1e6:.2f}M parameters"
        )

    def _temporal_pooling(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Aggregate encoded features across the time dimension.

        Args:
            encoded: (B, A, T, E)

        Returns:
            features: (B, A, output_dim)

        Strategies:
            last      - final timestep only; fastest, loses history.
            mean      - global average; good for smooth/continuous actions.
            max       - global max; captures peak signals (e.g. turning apex).
            concat    - cat(mean, max, last); richest representation, 3x wider.
            attention - learnable weighted sum; model learns which steps matter.
        """
        if self.pooling_type == 'last':
            return encoded[:, :, -1, :]

        elif self.pooling_type == 'mean':
            return encoded.mean(dim=2)

        elif self.pooling_type == 'max':
            return encoded.max(dim=2)[0]

        elif self.pooling_type == 'concat':
            mean_pool = encoded.mean(dim=2)
            max_pool  = encoded.max(dim=2)[0]
            last_step = encoded[:, :, -1, :]
            return torch.cat([mean_pool, max_pool, last_step], dim=-1)

        elif self.pooling_type == 'attention':
            attn_w = self.temporal_attention(encoded).squeeze(-1)   # (B, A, T)
            attn_w = F.softmax(attn_w, dim=-1)
            return (encoded * attn_w.unsqueeze(-1)).sum(dim=2)      # (B, A, E)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def forward(
        self,
        x:                torch.Tensor,
        context=None,
        adjacency=None,
        mask=None,
        feasibility:      Optional[torch.Tensor] = None,
        output_mode_only: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for mode prediction.

        Args:
            x               : Agent trajectory history (B, A, T, D)
            context         : Scene context features
            adjacency       : Agent interaction graph
            mask            : Optional attention mask
            feasibility     : Turn feasibility tensor (B, A, 4), float {0, 1}.
                              Column order: [TurnLeft, TurnRight, Straight, Hold].
                              If None, both soft conditioning and hard masking are skipped
                              regardless of use_feasibility / apply_hard_mask settings.
            output_mode_only: If True return raw logits; if False return softmax probs.

        Returns:
            mode_logits or mode_probs: (B, A, num_modes)

        Ablation note:
            Conditioning behaviour is fully determined by the config flags set at __init__:
                use_feasibility  - controls soft embedding (default: True)
                apply_hard_mask  - controls hard logit masking (default: False)
            No extra arguments needed at call time for ablation experiments.
        """
        # Encode trajectory with context using the backbone encoder
        encoded  = super().forward(x, context, adjacency, mask)    # (B, A, T, E)

        # Collapse time dimension via pooling
        features = self._temporal_pooling(encoded)                  # (B, A, input_dim)

        # Soft feasibility conditioning
        if self.use_feasibility and feasibility is not None:
            feat_feas = self.feasibility_embed(feasibility)         # (B, A, embed_dim)
            features  = torch.cat([features, feat_feas], dim=-1)   # (B, A, input_dim+embed_dim)

        # Classify
        mode_logits = self.mode_classifier(features)                # (B, A, num_modes)

        # Hard feasibility masking.
        # mode_logits is (B, A, 4) - one logit per turn class (TurnLeft,
        # TurnRight, Straight, Hold). feasibility is also (B, A, 4) with the
        # same column order, so the mask applies directly without expansion.
        if self.apply_hard_mask and feasibility is not None:
            infeasible  = (feasibility == 0)                          # (B, A, 4)
            mode_logits = mode_logits.masked_fill(infeasible, -1e9)

        if output_mode_only:
            return mode_logits
        else:
            return F.softmax(mode_logits, dim=-1)