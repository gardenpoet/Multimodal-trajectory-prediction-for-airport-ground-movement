# amelia_tf/models/components/amelia_trajectory.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Optional, Tuple

from .base_encoder import BaseAmeliaEncoder
from .gmm import GMM


class AmeliaTrajectory(BaseAmeliaEncoder):
    """
    Multi-modal trajectory prediction network.

    For each of the 4 turn modes (Left/Right/Straight/Hold), an independent
    trajectory hypothesis is decoded. mode_probs from Stage 1 serve as GMM
    mixture weights at loss time — they are NOT averaged inside the decoder,
    so multi-modality is fully preserved.

    Input shape:  x (B, A, T, D),  mode_probs (B, A, num_modes)
    Output shape: traj_mu / traj_sigma  (B, A, T, num_modes, out_dim)
    """

    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)

        self.decoder_config = config.decoder
        self.num_modes = self.decoder_config.num_modes  # 4

        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': self.num_modes,
            'mode_embed_dim': 32,
            'dropout': 0.1,
        }))
        self.mode_embed_dim = self.mode_config.mode_embed_dim

        # One embedding vector per turn mode
        self.turn_embedding = nn.Embedding(self.num_modes, self.mode_embed_dim)

        # Fuses encoder features with mode embedding
        # Input:  embed_size + mode_embed_dim
        # Output: embed_size
        self.trajectory_fusion = nn.Sequential(
            nn.Linear(self.embed_size + self.mode_embed_dim, self.embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_size, self.embed_size)
        )

        # GMM decoder — expects (B, A, T, M, embed_size), outputs (B, A, T, M, out_dim)
        self.decoder_config.in_size = self.embed_size
        self.decoder_head = GMM(self.decoder_config)

        print("AmeliaTrajectory (multi-modal) parameters: %.2fM" % (
            self.get_num_params() / 1e6,))
        print(f"  - Turn-level modes : {self.num_modes}")
        print(f"  - GMM out_dim      : {self.decoder_head.out_dim}")

    @property
    def num_dec_heads(self) -> int:
        return self.decoder_head.num_futures

    def forward(
        self,
        x: torch.Tensor,
        context=None,
        adjacency=None,
        mask=None,
        mode_probs: Optional[torch.Tensor] = None,  # (B, A, num_modes)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, A, T, D)
            context:    scene context features
            adjacency:  agent interaction graph
            mask:       optional attention mask
            mode_probs: (B, A, num_modes) — soft probs or one-hot from Stage 1

        Returns:
            traj_mu:    (B, A, T, num_modes, out_dim)
            traj_sigma: (B, A, T, num_modes, out_dim)
        """
        if mode_probs is None:
            raise ValueError("mode_probs must be provided (from Stage 1).")

        B, A, T, _ = x.shape
        M = self.num_modes

        # -- 1. Encode once, shared across all modes -----------
        # encoded: (B, A, T, embed_size)
        encoded = super().forward(x, context, adjacency, mask)

        # -- 2. Build mode embeddings ---------------------------
        # mode_emb_table: (M, mode_embed_dim)
        mode_emb_table = self.turn_embedding.weight

        # Expand to (B, A, T, M, mode_embed_dim)
        mode_emb = mode_emb_table.view(1, 1, 1, M, self.mode_embed_dim) \
                                  .expand(B, A, T, M, -1)

        # -- 3. Fuse encoded features with mode embeddings ------
        # Expand encoded to (B, A, T, M, embed_size)
        encoded_exp = encoded.unsqueeze(3).expand(-1, -1, -1, M, -1)

        # Concatenate: (B, A, T, M, embed_size + mode_embed_dim)
        decoder_input = torch.cat([encoded_exp, mode_emb], dim=-1)

        # Fuse: (B, A, T, M, embed_size)
        decoder_input = self.trajectory_fusion(decoder_input)

        # Residual connection
        decoder_input = encoded_exp + decoder_input  # (B, A, T, M, embed_size)

        # -- 4. GMM decode --------------------------------------
        # GMM expects (B, A, T, M, C) ? outputs (B, A, T, M, out_dim)
        traj_mu, traj_sigma = self.decoder_head(decoder_input)

        return traj_mu, traj_sigma