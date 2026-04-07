# amelia_tf/models/components/base_encoder.py

import torch
import torch.nn as nn
from easydict import EasyDict
from typing import Any, Optional, Tuple

from amelia_tf.models.components.self_attention import SelfAttentionBlock
from amelia_tf.models.components.cross_attention import CrossAttentionBlock
from amelia_tf.models.components.common import MLP, LayerNorm


class BaseAmeliaEncoder(nn.Module):
    """
    Shared encoder backbone for AmeliaTF trajectory and mode prediction models.

    Encodes agent trajectory history together with scene context using a stack of
    self-attention (temporal + social) and cross-attention (agent-context) blocks.
    The resulting feature tensor is consumed by task-specific decoder heads.
    """

    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.encoder_config = config.encoder
        self.in_size    = self.encoder_config.in_size + self.encoder_config.interp_flag
        self.embed_size = self.encoder_config.embed_size

        # --- Agent feature extraction ---
        # Projects raw trajectory features into the model embedding space.
        self.agents_fe = nn.Sequential(
            nn.Linear(self.in_size, self.embed_size),
            LayerNorm(self.embed_size, self.encoder_config.bias),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)
        )

        # --- Context feature extraction ---
        # Selects context encoder variant based on configuration.
        ctx_enc_type = self.encoder_config.context_encoder_type
        if ctx_enc_type == 'v0':
            from amelia_tf.models.components.context import ContextNetv0 as ContextNet
            context_config = self.encoder_config.contextnet_v0
        elif ctx_enc_type == 'v1':
            from amelia_tf.models.components.context import ContextNetv1 as ContextNet
            context_config = self.encoder_config.contextnet_v1
        elif ctx_enc_type == 'v2':
            from amelia_tf.models.components.context import ContextNetv2 as ContextNet
            context_config = self.encoder_config.contextnet_v2
        elif ctx_enc_type == 'v3':
            from amelia_tf.models.components.context import ContextNetv3 as ContextNet
            context_config = self.encoder_config.contextnet_v3
        elif ctx_enc_type == 'v4':
            from amelia_tf.models.components.context import ContextNetv4 as ContextNet
            context_config = self.encoder_config.contextnet_v4
        else:
            raise NotImplementedError(f"Unknown context encoder type: {ctx_enc_type}")
        self.context_fe = ContextNet(context_config)

        # --- Sequence length bookkeeping ---
        self.hist_len  = self.encoder_config.hist_len
        self.pred_lens = self.encoder_config.pred_lens

        # Learnable temporal positional embeddings
        self.time_pe = nn.Embedding(self.encoder_config.T_size, self.embed_size)

        self.drop = nn.Dropout(self.encoder_config.dropout)

        # --- Social-Temporal attention blocks ---
        # Built in four stages:
        #   1. Pre-blocks  (self-attention over time + agents)
        #   2. Pre cross-attention blocks (agent-context)
        #   3. Main blocks (self-attention over time + agents)
        #   4. Main cross-attention blocks (agent-context)
        #   5. Post-blocks (self-attention over time + agents)
        self.encoder_config.in_size = self.embed_size
        att_blocks = []

        for _ in range(self.encoder_config.num_satt_pre_blocks):
            att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        for _ in range(self.encoder_config.num_catt_pre_blocks):
            att_blocks.append(CrossAttentionBlock(self.encoder_config))

        for _ in range(self.encoder_config.num_satt_blocks):
            att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        for _ in range(self.encoder_config.num_catt_blocks):
            att_blocks.append(CrossAttentionBlock(self.encoder_config))

        for _ in range(self.encoder_config.num_satt_post_blocks):
            att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        self.att_blocks = nn.ModuleList(att_blocks)

        # Final MLP to refine encoded features
        self.refine_mlp = MLP(self.encoder_config)

        self.apply(self._init_weights)

    def _init_weights(self, module: Any) -> None:
        """Initialize linear and embedding weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        context=None,
        adjacency=None,
        mask=None
    ) -> torch.Tensor:
        """
        Encode agent trajectory history with scene context.

        Args:
            x:         Agent trajectory history  (B, A, T, D)
            context:   Scene context features
            adjacency: Agent interaction graph
            mask:      Optional attention mask

        Returns:
            encoded: Contextualised agent features  (B, A, T, embed_size)
        """
        device = x.device
        B, A, T, D = x.size()
        assert T <= self.encoder_config.T_size, (
            f"Sequence length {T} exceeds maximum block size {self.encoder_config.T_size}"
        )

        # Project trajectory features into embedding space
        x = self.agents_fe(x)                                          # (B, A, T, embed_size)

        # Add learnable temporal positional embeddings
        time_idx = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
        time_emb = self.time_pe(time_idx).unsqueeze(0)                 # (1, 1, T, embed_size)
        x = self.drop(x + time_emb)

        # Encode scene context (map, graph, etc.)
        cx = self.context_fe(context, adj=adjacency)

        # Apply social-temporal attention blocks
        for block in self.att_blocks:
            x = block(x, cx, mask=mask)

        # Final feature refinement
        x = self.refine_mlp(x)                                         # (B, A, T, embed_size)

        return x

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count the number of trainable parameters.

        Args:
            non_embedding: If True, excludes temporal positional embedding parameters.

        Returns:
            Total parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'time_pe') and hasattr(self.time_pe, 'weight'):
            n_params -= self.time_pe.weight.numel()
        return n_params