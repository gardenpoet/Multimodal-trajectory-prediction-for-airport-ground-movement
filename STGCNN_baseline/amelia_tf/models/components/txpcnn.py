"""
Time-Extrapolator CNN (TXP-CNN) component.

Reproduces Section 3.2.4 of Zhang, Zhong & Mahadevan (2022) (see stgcnn.py for the full
reference), which extrapolates the STG-CNN embedding from `hist_len` observed timesteps
to `pred_len` future timesteps by treating the time axis as the convolutional *channel*
axis: the (hidden_channels, K) feature map at each timestep is treated as one "image",
so a stack of `hist_len` such images is a (hist_len, hidden_channels, K) tensor, and a
Conv2d whose input/output channel counts are timestep counts remaps that stack from
`hist_len` "channels" to `pred_len` "channels". This mirrors the TXP-CNN used in
Social-STGCNN (Mohamed et al., 2020), which this paper's time-extrapolation stage follows.
"""
import torch
import torch.nn as nn


class TXPCNN(nn.Module):
    """ Extrapolates a (B, hist_len, C, K) stack of per-timestep feature maps to a
    (B, pred_len, C, K) stack via `num_layers` Conv2d + PReLU layers. The first layer
    changes the number of "channels" (timesteps) from hist_len to pred_len; subsequent
    layers keep it at pred_len and add a residual connection, following standard
    practice in the Social-STGCNN reference TXP-CNN. """

    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "TXP-CNN needs at least one layer"
        pad = kernel_size // 2

        convs = [nn.Conv2d(hist_len, pred_len, kernel_size=kernel_size, padding=pad)]
        for _ in range(num_layers - 1):
            convs.append(nn.Conv2d(pred_len, pred_len, kernel_size=kernel_size, padding=pad))

        self.convs = nn.ModuleList(convs)
        self.prelus = nn.ModuleList([nn.PReLU() for _ in convs])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        -----
            x[torch.Tensor(B, hist_len, C, K)]: STG-CNN embedding, with the observed
                time axis moved into the channel slot.

        Output
        ------
            out[torch.Tensor(B, pred_len, C, K)]: extrapolated embedding, with the
                predicted time axis now in the channel slot.
        """
        x = self.prelus[0](self.convs[0](x))
        for conv, act in zip(self.convs[1:], self.prelus[1:]):
            x = act(conv(x)) + x  # residual: shapes match from the 2nd layer onward
        return self.dropout(x)
