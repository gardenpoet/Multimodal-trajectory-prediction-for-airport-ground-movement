"""
STG-CNN + TXP-CNN trajectory-prediction baseline.

Reproduces:
    Zhang, Zhong & Mahadevan (2022), "Airport surface movement prediction and safety
    assessment with spatial-temporal graph convolutional neural network",
    Transportation Research Part C 144: 103873.

This is a *unimodal* baseline (single bivariate Gaussian per agent per future timestep,
no turn-mode classification), trained and evaluated on the exact same scenes/splits/
filtering/coordinate-normalization as the project's other AmeliaTF-based methods (see
`amelia_tf/data` and `amelia_scenes`, copied unmodified from `AmeliaTF_main`).

See STGCNN_baseline/README.md for a full list of judgment calls made while reproducing
this paper, including how this differs from `AmeliaTF_main/amelia_tf/models/trajpred.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Any, List

from amelia_tf.models.components.stgcnn import STGCNN, build_adjacency
from amelia_tf.models.components.txpcnn import TXPCNN
from amelia_tf.utils.utils import separate_ego_agent
from amelia_tf.utils.metrics import marginal_ade, marginal_fde


def diagonal_gaussian_nll(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """ Negative log-likelihood of `target` (x, y) under a diagonal (independent-axes)
    bivariate Gaussian parameterized by (mu_x, mu_y, sigma_x, sigma_y) -- no correlation
    term. This intentionally drops the `rho` term from the paper's Eq. 1 so this
    baseline's NLL is directly comparable to `AmeliaTF_main`'s GMM head
    (`amelia_tf/models/components/gmm.py`), which likewise has no correlation term and
    computes its regression loss via `torch.nn.functional.gaussian_nll_loss(mu, target,
    sigma**2)` (see `amelia_tf/utils/losses.py::marginal_loss`). Reused verbatim here,
    per-axis and summed, so this baseline's `val/nll`/`test/nll` are on the same footing
    as every other method's.

    Inputs
    ------
        mu, sigma[torch.Tensor(..., 2)]: predicted mean / std-dev of (x, y).
        target[torch.Tensor(..., 2)]: ground-truth (x, y).

    Output
    ------
        nll[torch.Tensor(...)]: negative log-likelihood, summed over the x/y axes.
    """
    var = sigma ** 2
    nll_per_axis = F.gaussian_nll_loss(mu, target, var, reduction='none')  # (..., 2)
    return nll_per_axis.sum(dim=-1)


class STGCNNPredictor(nn.Module):
    """ Ties the STG-CNN graph-construction + graph-conv layer and the TXP-CNN
    time-extrapolator together, plus a final 1x1-conv head projecting the TXP-CNN's
    hidden embedding down to the 4 diagonal-Gaussian parameters (mu_x, mu_y, sigma_x,
    sigma_y) per (agent, future timestep). No correlation term (rho) -- see
    `diagonal_gaussian_nll` and the README for why. """

    def __init__(
        self,
        hist_len: int,
        pred_lens: List[int],
        in_channels: int = 4,
        hidden_channels: int = 64,
        stgcnn_temporal_kernel: int = 3,
        num_txp_layers: int = 5,
        txp_kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hist_len = hist_len
        self.pred_lens = list(pred_lens)
        self.pred_len = max(self.pred_lens)  # T_hat: longest prediction horizon
        self.num_dec_heads = 1  # unimodal: a single Gaussian, no mode classification

        self.stgcnn = STGCNN(
            in_channels=in_channels, hidden_channels=hidden_channels,
            out_channels=hidden_channels, temporal_kernel=stgcnn_temporal_kernel,
            dropout=dropout,
        )
        self.txpcnn = TXPCNN(
            hist_len=hist_len, pred_len=self.pred_len, num_layers=num_txp_layers,
            kernel_size=txp_kernel_size, dropout=dropout,
        )
        self.head = nn.Conv2d(hidden_channels, 4, kernel_size=1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        """
        Inputs
        ------
            X[torch.Tensor(B, A, hist_len, >=4)]: relative agent sequences, ego-frame.
                Columns 0, 1 are (x, y); column 3 is heading (rad). Only the observed
                `hist_len` timesteps are passed in (no zero-padded future).
            mask[torch.Tensor(B, A, hist_len)]: per-(agent, timestep) validity mask.

        Outputs
        -------
            mu[torch.Tensor(B, A, pred_len, 2)]
            sigma[torch.Tensor(B, A, pred_len, 2)]: positive (softplus + 1e-3).
        """
        pos = X[..., :2]                                     # (B, A, Th, 2)
        heading = X[..., 3]                                  # (B, A, Th)
        node_feats = torch.stack(
            [pos[..., 0], pos[..., 1], torch.cos(heading), torch.sin(heading)], dim=-1
        )                                                     # (B, A, Th, 4)

        node_feats = node_feats.permute(0, 3, 2, 1)           # (B, 4, Th, A)
        pos_t = pos.permute(0, 2, 1, 3)                       # (B, Th, A, 2)
        mask_t = mask.permute(0, 2, 1)                        # (B, Th, A)

        A = build_adjacency(pos_t, mask_t)                   # (B, Th, A, A)
        emb = self.stgcnn(node_feats, A)                      # (B, Chat, Th, A)

        emb = emb.permute(0, 2, 1, 3)                         # (B, Th, Chat, A): T as channel
        out = self.txpcnn(emb)                                # (B, Tp, Chat, A)
        out = out.permute(0, 2, 1, 3)                         # (B, Chat, Tp, A)
        out = self.head(out)                                  # (B, 4, Tp, A)
        out = out.permute(0, 3, 2, 1)                          # (B, A, Tp, 4)

        mu = out[..., :2]
        sigma = F.softplus(out[..., 2:4]) + 1e-3              # same convention as gmm.py
        return mu, sigma


class STGCNNTrajPred(LightningModule):
    """ Lightning wrapper for the STG-CNN + TXP-CNN baseline. Mirrors the metric-naming
    conventions of `AmeliaTF_main/amelia_tf/models/trajpred.py` (`val/ade/t=..`,
    `test/ade/t=..`, `losses/train`, etc.) so W&B logs are structurally comparable
    across methods, but reports only the unimodal "All" ADE/FDE/NLL (no per-mode
    breakdown, no ego/joint propagation choice, no off-road/modal-classification
    metrics -- those are specific to the project's mode-conditioned methods). """

    def __init__(self, optimizer: EasyDict, net: nn.Module, extra_params: EasyDict):
        super().__init__()
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net
        self.hist_len = net.hist_len
        self.pred_lens = net.pred_lens
        self.max_pred_len = net.pred_len

        self.eparams = extra_params

        self.train_loss, self.val_loss, self.test_loss = MeanMetric(), MeanMetric(), MeanMetric()
        self.val_nll, self.test_nll = MeanMetric(), MeanMetric()

        self.val_ade, self.val_fde = {}, {}
        self.test_ade, self.test_fde = {}, {}
        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key] = MeanMetric()
            self.val_fde[key] = MeanMetric()
            self.test_ade[key] = MeanMetric()
            self.test_fde[key] = MeanMetric()
        self.val_ade = nn.ModuleDict(self.val_ade)
        self.val_fde = nn.ModuleDict(self.val_fde)
        self.test_ade = nn.ModuleDict(self.test_ade)
        self.test_fde = nn.ModuleDict(self.test_fde)

    def model_step(self, batch: Any):
        seq = batch['scene_dict']['rel_sequences']            # (B, A, T, 7)
        masks = batch['scene_dict']['agent_masks'].bool()      # (B, A, T)

        X = seq[:, :, :self.hist_len].float()                  # observed window only
        mask_h = masks[:, :, :self.hist_len]

        fut_end = self.hist_len + self.max_pred_len
        Y = seq[:, :, self.hist_len:fut_end, :2].float()       # (B, A, Tp, 2): (x, y)
        mask_f = masks[:, :, self.hist_len:fut_end]            # (B, A, Tp)

        mu, sigma = self.net(X, mask_h)

        nll = diagonal_gaussian_nll(mu, sigma, Y)              # (B, A, Tp)
        nll = nll * mask_f
        # Eq. 1: NLL summed over the prediction horizon, then averaged over agents
        # that have at least one valid future step, then over the batch.
        nll_per_agent = nll.sum(dim=-1)                        # (B, A)
        valid_agent = mask_f.any(dim=-1)                       # (B, A)
        if valid_agent.any():
            loss = nll_per_agent[valid_agent].mean()
        else:
            loss = nll_per_agent.sum() * 0.0

        return loss, mu, sigma, Y, mask_f

    def _ego_slices(self, batch: Any, mu, sigma, Y, mask_f, ego_key: str):
        ego_agent = batch['scene_dict'][ego_key]
        # marginal_ade/marginal_fde expect Y_hat with a mode-head dim (B, A, T, H, D)
        ego_mu = separate_ego_agent(mu.unsqueeze(3), ego_agent)      # (B, 1, Tp, 1, 2)
        ego_sigma = separate_ego_agent(sigma.unsqueeze(3), ego_agent)
        ego_Y = separate_ego_agent(Y, ego_agent)                     # (B, 1, Tp, 2)
        ego_mask = separate_ego_agent(mask_f, ego_agent)             # (B, 1, Tp)
        return ego_mu, ego_sigma, ego_Y, ego_mask

    def training_step(self, batch: Any, batch_idx: int):
        loss, *_ = self.model_step(batch)
        self.train_loss(loss)
        self.log("losses/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, mu, sigma, Y, mask_f = self.model_step(batch)
        ego_mu, ego_sigma, ego_Y, ego_mask = self._ego_slices(batch, mu, sigma, Y, mask_f, 'ego_agent_id')

        self.val_loss(loss)
        self.log("losses/val", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            mu_t = ego_mu[:, :, :t]
            Y_t = ego_Y[:, :, :t]
            mask_t = ego_mask[:, :, :t]

            self.val_ade[key](marginal_ade(mu_t, Y_t, mask=mask_t))
            self.log(f"val/ade/{key}", self.val_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.val_fde[key](marginal_fde(mu_t, Y_t, mask=mask_t))
            self.log(f"val/fde/{key}", self.val_fde[key], on_step=False, on_epoch=True, prog_bar=True)

        nll = diagonal_gaussian_nll(ego_mu.squeeze(3), ego_sigma.squeeze(3), ego_Y)
        nll = (nll * ego_mask).sum(dim=-1) / ego_mask.sum(dim=-1).clamp_min(1)
        self.val_nll(nll.mean())
        self.log("val/nll", self.val_nll, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, mu, sigma, Y, mask_f = self.model_step(batch)
        ego_mu, ego_sigma, ego_Y, ego_mask = self._ego_slices(
            batch, mu, sigma, Y, mask_f, 'ego_agent_id_test'
        )

        self.test_loss(loss)
        self.log("losses/test", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            mu_t = ego_mu[:, :, :t]
            Y_t = ego_Y[:, :, :t]
            mask_t = ego_mask[:, :, :t]

            self.test_ade[key](marginal_ade(mu_t, Y_t, mask=mask_t))
            self.log(f"test/ade/{key}", self.test_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.test_fde[key](marginal_fde(mu_t, Y_t, mask=mask_t))
            self.log(f"test/fde/{key}", self.test_fde[key], on_step=False, on_epoch=True, prog_bar=True)

        nll = diagonal_gaussian_nll(ego_mu.squeeze(3), ego_sigma.squeeze(3), ego_Y)
        nll = (nll * ego_mask).sum(dim=-1) / ego_mask.sum(dim=-1).clamp_min(1)
        self.test_nll(nll.mean())
        self.log("test/nll", self.test_nll, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Paper (Sec. 4.1): plain SGD, lr=0.02. No scheduler is reported, so none is used.
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.hparams.optimizer.lr,
            momentum=self.hparams.optimizer.get('momentum', 0.9),
            weight_decay=self.hparams.optimizer.get('weight_decay', 0.0),
        )
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = STGCNNTrajPred(None, None, None)
