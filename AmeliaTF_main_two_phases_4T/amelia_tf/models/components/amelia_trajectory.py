# amelia_tf/models/components/amelia_trajectory.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from typing import Tuple, Optional

from .base_encoder import BaseAmeliaEncoder
from .gmm import GMM


class AmeliaTrajectory(BaseAmeliaEncoder):
    """
    Trajectory prediction network conditioned on turn-level modes.

    Supports three feature fusion strategies controlled by subspace_mode:

        'none':
            Original path. Mode embeddings (turn_embedding) are blended with
            encoded features via weighted sum, then fused through a shared
            linear layer. No feature separation between modes.

        'fusion':
            The fusion layer maps encoded features to M * embed_size. After
            reshape to (B, A, T, M, embed_size), the slice corresponding to
            argmax(mode_probs) is selected. Each mode activates a dedicated
            region of the fused feature space.

        'encoder':
            The encoder output (embed_size) is split directly into M equal
            slices of size embed_size // num_modes. The slice corresponding to
            argmax(mode_probs) is selected. No fusion layer is needed.
            Requires embed_size % num_modes == 0.

    Output shape change vs. original:
        Original : traj_mu / traj_sigma  (B, A, T, D)        [squeezed]
        New      : traj_mu / traj_sigma  (B, A, T, K, D)     [K = num_hypotheses]

        When num_hypotheses == 1 (default), K=1 and the shape is (B, A, T, 1, D).
        Callers that previously relied on the squeezed (B, A, T, D) shape must
        either squeeze themselves or handle the extra dimension - see
        CombinedTrajPredSystem for the updated handling.
    """

    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)

        self.decoder_config = config.decoder
        self.num_modes = self.decoder_config.num_modes  # TurnLeft, TurnRight, Straight, Hold

        # ---------------------------------------------------------------------------
        # num_hypotheses (K): number of trajectory candidates per mode.
        # Defaults to 1, which preserves exact backward compatibility with
        # checkpoints trained before this change was introduced.
        # ---------------------------------------------------------------------------
        self.num_hypotheses = getattr(config, 'num_hypotheses', 2)

        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': self.num_modes,
            'mode_embed_dim': 32,
            'dropout': 0.1,
        }))

        # Subspace strategy: 'none' | 'fusion' | 'encoder'
        self.subspace_mode = getattr(config, 'subspace_mode', 'none')
        assert self.subspace_mode in ('none', 'fusion', 'encoder'), (
            f"subspace_mode must be one of 'none', 'fusion', 'encoder', "
            f"got '{self.subspace_mode}'"
        )

        if self.subspace_mode == 'none':
            self.turn_embedding = nn.Embedding(
                self.num_modes,
                self.mode_config.mode_embed_dim
            )
            self.trajectory_fusion = nn.Sequential(
                nn.Linear(self.embed_size + self.mode_config.mode_embed_dim, self.embed_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_size, self.embed_size)
            )
            decoder_in = self.embed_size

        elif self.subspace_mode == 'fusion':
            self.trajectory_fusion = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size * self.num_modes),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_size * self.num_modes, self.embed_size * self.num_modes)
            )
            decoder_in = self.embed_size

        else:  # 'encoder'
            assert self.embed_size % self.num_modes == 0, (
                f"embed_size ({self.embed_size}) must be divisible by "
                f"num_modes ({self.num_modes}) when subspace_mode='encoder'"
            )
            self.subspace_dim = self.embed_size // self.num_modes
            decoder_in = self.subspace_dim

        # ---------------------------------------------------------------------------
        # Tell the GMM head to output num_hypotheses futures instead of 1.
        # For pre-trained checkpoints with num_hypotheses=1, this is a no-op
        # because the original decoder was already trained with num_futures=1.
        # ---------------------------------------------------------------------------
        self.decoder_config.in_size = decoder_in
        self.decoder_config.num_futures = self.num_hypotheses
        self.decoder_head = GMM(self.decoder_config)

        print("Trajectory model parameters: %.2fM" % (self.get_num_params() / 1e6,))
        print(f"  - Turn-level modes  : {self.num_modes}")
        print(f"  - Hypotheses per mode: {self.num_hypotheses}")
        print(f"  - Subspace mode     : {self.subspace_mode}")

    @property
    def num_dec_heads(self) -> int:
        return self.decoder_head.num_futures

    def _prepare_decoder_input(
        self,
        encoded: torch.Tensor,
        mode_probs: torch.Tensor
    ) -> torch.Tensor:
        # Unchanged from original - see original file for full docstring.
        B, A, T, _ = encoded.shape
        encoded_exp = encoded.unsqueeze(-2)  # (B, A, T, 1, embed_size)

        if self.subspace_mode == 'none':
            mode_emb = torch.matmul(mode_probs, self.turn_embedding.weight)
            mode_emb_exp = mode_emb.unsqueeze(2).unsqueeze(2).expand(-1, -1, T, -1, -1)
            decoder_input = torch.cat([encoded_exp, mode_emb_exp], dim=-1)
            decoder_input = self.trajectory_fusion(decoder_input)
            decoder_input = encoded_exp + decoder_input

        elif self.subspace_mode == 'fusion':
            fused = self.trajectory_fusion(encoded)
            fused = fused.view(B, A, T, self.num_modes, self.embed_size)
            mode_idx = mode_probs.argmax(dim=-1)
            mode_idx_exp = mode_idx[:, :, None, None, None].expand(B, A, T, 1, self.embed_size)
            fused_k = fused.gather(3, mode_idx_exp)
            decoder_input = encoded_exp + fused_k

        else:  # 'encoder'
            encoded_split = encoded.view(B, A, T, self.num_modes, self.subspace_dim)
            mode_idx = mode_probs.argmax(dim=-1)
            mode_idx_exp = mode_idx[:, :, None, None, None].expand(B, A, T, 1, self.subspace_dim)
            decoder_input = encoded_split.gather(3, mode_idx_exp)

        return decoder_input

    def forward(
        self,
        x: torch.Tensor,
        context=None,
        adjacency=None,
        mask=None,
        mode_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with turn-mode conditioning.

        Args:
            x:          (B, A, T, D)          input trajectory
            context:    scene context features
            adjacency:  agent interaction graph
            mask:       optional attention mask
            mode_probs: (B, A, num_modes)      one-hot or soft probabilities

        Returns:
            traj_mu:    (B, A, T, K, D)   predicted trajectory means
            traj_sigma: (B, A, T, K, D)   predicted trajectory std devs

            K = num_hypotheses (1 by default ? backward-compatible with old
            checkpoints; callers must handle the extra K dimension).

        Change vs. original:
            The final .squeeze(3) calls have been removed so that the K
            dimension is always present. When K=1 the shapes are identical to
            the old (B, A, T, 1, D), which CombinedTrajPredSystem now handles
            explicitly rather than relying on an implicit squeeze.
        """
        if mode_probs is None:
            raise ValueError("mode_probs must be provided (one-hot or soft probabilities).")

        encoded = super().forward(x, context, adjacency, mask)  # (B, A, T, embed_size)
        decoder_input = self._prepare_decoder_input(encoded, mode_probs)
        

        # GMM output: (B, A, T, K, D)  where K = num_hypotheses
        traj_mu, traj_sigma, score = self.decoder_head(decoder_input)

        # NOTE: .squeeze(3) intentionally removed here.
        # The K dimension is kept so CombinedTrajPredSystem can run
        # off-road-guided selection across hypotheses before squeezing.
        # For K=1 (default / pre-trained checkpoints) the shape is (B, A, T, 1, D),
        # which is handled in the system-level forward pass.

        return traj_mu, traj_sigma, score