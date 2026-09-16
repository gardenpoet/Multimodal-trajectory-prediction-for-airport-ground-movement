"""
Spatio-Temporal Graph Convolutional Network (STG-CNN) component.

Reproduces Section 3.2.2 (graph construction) and Section 3.2.3 (STG-CNN, Eq. 4-6) of:

    Zhang, Zhong & Mahadevan (2022), "Airport surface movement prediction and safety
    assessment with spatial-temporal graph convolutional neural network",
    Transportation Research Part C 144: 103873.

The paper's STG-CNN graph convolution is functionally identical to the `st_gcn` block
introduced by Mohamed et al. (2020), "Social-STGCNN: A Social Spatio-Temporal Graph
Convolutional Neural Network for Human Trajectory Prediction" (CVPR), which this paper
explicitly builds on. This module mirrors that mechanism (a graph-shift operation via a
fixed, per-timestep weighted adjacency, combined with a temporal convolution) rather than
literally reusing Social-STGCNN's code.
"""
from typing import Optional

import torch
import torch.nn as nn


def build_adjacency(
    pos: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-9
) -> torch.Tensor:
    """ Builds the per-timestep weighted agent graph described in Sec. 3.2.2.

    Edge weight between objects i and j at time t is the inverse squared Euclidean
    distance: a^t_ij = 1 / ((x^t_i - x^t_j)^2 + (y^t_i - y^t_j)^2) if the denominator is
    non-zero, else 0 (this is the standard Social-STGCNN edge weighting). A self-loop
    (a_ii = 1) is then added so every node also attends to itself, and each row is
    normalized to sum to 1 (D^-1 A), matching the normalized graph-shift operator used
    by ST-GCN / Social-STGCNN.

    Inputs
    ------
        pos[torch.Tensor(B, T, K, 2)]: (x, y) position of each of the K objects, at
            each of the T timesteps.
        mask[torch.Tensor(B, T, K)]: optional validity mask (1 = real object,
            0 = padded/invalid). Padded nodes get zero weight on every edge touching
            them, so they cannot influence, or be influenced by, valid nodes.
        eps[float]: numerical floor to avoid division by zero.

    Output
    ------
        A[torch.Tensor(B, T, K, K)]: row-normalized adjacency. A[b, t, i, j] is the
            (normalized) weight with which object j's features contribute to object i.
    """
    diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)           # (B, T, K, K, 2): i - j
    dist_sq = (diff ** 2).sum(-1)                          # (B, T, K, K)

    A = torch.where(
        dist_sq > eps, 1.0 / dist_sq.clamp_min(eps), torch.zeros_like(dist_sq)
    )

    K = pos.shape[-2]
    eye = torch.eye(K, device=pos.device, dtype=pos.dtype).expand_as(A)
    A = A + eye  # self-loop

    if mask is not None:
        m = mask.to(pos.dtype)
        pair_mask = m.unsqueeze(-1) * m.unsqueeze(-2)      # (B, T, K, K)
        A = A * pair_mask

    row_sum = A.sum(-1, keepdim=True).clamp_min(eps)
    return A / row_sum


class SpatialGraphConv(nn.Module):
    """ Single graph-convolution step: a 1x1 conv (feature transform), followed by a
    graph-shift (weighted aggregation of neighbor features via the adjacency matrix). """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Inputs
        ------
            x[torch.Tensor(B, C, T, K)]: node features.
            A[torch.Tensor(B, T, K, K)]: per-timestep weighted adjacency.

        Output
        ------
            out[torch.Tensor(B, C_out, T, K)]: graph-convolved node features.
        """
        x = self.conv(x)
        # out[b, c, t, i] = sum_j x[b, c, t, j] * A[b, t, i, j]
        return torch.einsum('bctj,btij->bcti', x, A)


class STGCNN(nn.Module):
    """ Spatio-temporal graph convolution layer (Sec. 3.2.3, Eq. 4-6): a single graph
    convolution that operates jointly over the temporal neighborhood (via a temporal
    Conv2d, matching Fig. 4's k x k kernel spanning t-1/t/t+1) and the spatial
    neighborhood (via the weighted adjacency from `build_adjacency`) of each node.
    Produces an embedding of shape (B, hidden_channels, T, K). """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        out_channels: int = 64,
        temporal_kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, hidden_channels)
        self.gcn_act = nn.PReLU()

        pad = temporal_kernel // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(pad, 0)),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.out_act = nn.PReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Inputs
        ------
            x[torch.Tensor(B, in_channels, T, K)]: node features
                (x, y, cos(heading), sin(heading)).
            A[torch.Tensor(B, T, K, K)]: per-timestep weighted adjacency.

        Output
        ------
            out[torch.Tensor(B, out_channels, T, K)]: spatio-temporal embedding.
        """
        res = self.residual(x)
        x = self.gcn_act(self.gcn(x, A))
        x = self.tcn(x)
        return self.out_act(x + res)
