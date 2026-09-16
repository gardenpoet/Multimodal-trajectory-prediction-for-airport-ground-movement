import torch
import torch.nn as nn
from torch.nn import functional as F
from easydict import EasyDict
from typing import Tuple, Optional
import math


class GMM(nn.Module):
    """
    GMM head for multi-future trajectory prediction, with a switchable score head.

    score_head_type (config, default 'mlp'):
      'mlp'       : original score head. An MLP over [hidden (+ flattened traj)]
                    that emits K numbers. Index<->candidate correspondence is by
                    CONVENTION (learned via the winner label). This is the
                    original, always-available fallback.

                    NOTE on what this head can and cannot see: with score_mode=4
                    the per-candidate part of the input is mu at ONE timestep,
                    i.e. K*out_dim numbers describing where the K candidates are
                    at that instant; `hidden` is shared across candidates. The
                    head is applied per timestep and the scores are averaged over
                    the horizon, so the final score has the form
                        score_k = mean_t f(hidden_t, mu_t)[k].
                    A pointwise function of mu_t, averaged over t, cannot recover
                    any difference between consecutive timesteps -- so this head
                    structurally cannot compute speed or acceleration, only
                    position-based quantities. If the discriminative signal lies
                    in the speed profile, use the attention head instead.

      'attention' : query-key attention. query = proj(hidden intent context),
                    key_k = proj(extent features of candidate k). score_k is the
                    dot product. Correspondence is STRUCTURAL (key_k built from
                    candidate k), agnostic to K, and biased toward "which extent
                    matches the intent" -- suited to turning modes where the K
                    candidates share a shape but differ in how far they travel.
                    The key features are explicitly derived from step-to-step
                    differences, so this head does see speed/acceleration.

    per_mode_score (config, default False):
      Build one independent scoring head per turning mode instead of a single
      shared one. Supported by BOTH head types; the caller sets `_active_mode`
      before each per-mode forward to select which head is used.

    Decoder is a single Sequential named `future_heads` so its parameter names
    match pre-trained checkpoints (future_heads.0 = first Linear, .2 = second).
    """

    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        self.config = config
        self.num_futures = config.num_futures
        self.out_dim = int(config.num_dims // 2)
        self.in_size = config.in_size

        self.enable_score_head = getattr(config, "enable_score_head", False)
        self.score_mode = getattr(config, "score_mode", 5)
        self.score_detach = getattr(config, "score_detach", True)
        self.score_hidden = getattr(config, "score_hidden", 256)

        self.key_feature_mode = getattr(config, "key_feature_mode", "full")

        # Switch between the original MLP head and the attention head
        self.score_head_type = getattr(config, "score_head_type", "attention")
        self.score_attn_dim = getattr(config, "score_attn_dim", 128)
        self.hist_len = getattr(config, "hist_len", 10)

        # Future length (Tp) needed to size the 'full' key encoder at init time.
        self.pred_len = getattr(config, "pred_len",
                          getattr(config, "T_pred", 50))

        # Per-mode scoring (shared by both head types)
        self.per_mode_score = getattr(config, "per_mode_score", True)
        self.num_modes_score = getattr(config, "score_num_modes", 4)

        # Prediction-node speed features
        self.use_node_features = getattr(config, "use_node_features", True)
        self.node_window = getattr(config, "node_window", 5)

        # Turning features
        self.use_turning_features = getattr(config, "use_turning_features", True)
        self.turning_window = getattr(config, "turning_window", 10)

        self.use_sigma_features = getattr(config, "use_sigma_features", True)

        # Extent dimension for score_mode=5
        # Base: 7 (endpoint, disp, arclen, mean_spd, final_spd)
        # + 5 node speed features
        # + 6 turning features (3 hist + 3 pred)
        # + 3 sigma features (mag / end / start)
        self.extent_dim = (7
                           + (5 if self.use_node_features else 0)
                           + (6 if self.use_turning_features else 0)
                           + (3 if self.use_sigma_features else 0))

        gmm_embd = self.in_size

        # Trajectory decoder (names match checkpoints: future_heads.0/.2)
        self.future_heads = nn.Sequential(
            nn.Linear(gmm_embd, 4 * gmm_embd),
            nn.GELU(),
            nn.Linear(4 * gmm_embd, config.num_dims * self.num_futures, bias=False),
        )

        if self.enable_score_head:
            if self.score_head_type == "attention":
                self._build_attention_head(gmm_embd)
            else:
                self._build_mlp_head(gmm_embd)

    # ------------------------------------------------------------------
    # Turning features extraction
    # ------------------------------------------------------------------
    def _turning_features(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Extract turning/heading change features from both history and future.

        Returns (B, A, K, 6):
            hist_turn_rate:  Average turning rate in history (radians/step)
            hist_turn_std:    Standard deviation of history turning rate
            hist_total_turn:  Total turning angle in history

            pred_turn_rate:   Average turning rate in prediction (radians/step)
            pred_turn_std:    Standard deviation of prediction turning rate
            pred_total_turn:  Total turning angle in prediction
        """
        xy = mu[..., :2]  # (B, A, T, K, 2)
        B, A, T, K, _ = xy.shape
        hl = self.hist_len
        w = min(self.turning_window, hl - 1, T - hl - 1)

        # Compute heading: heading = atan2(dy, dx)
        vel = xy[:, :, 1:, :, :] - xy[:, :, :-1, :, :]  # (B, A, T-1, K, 2)
        heading = torch.atan2(vel[..., 1], vel[..., 0])  # (B, A, T-1, K)

        # Compute heading change rate (turning rate)
        heading_diff = heading[:, :, 1:, :] - heading[:, :, :-1, :]  # (B, A, T-2, K)
        # Normalize angle to [-pi, pi]
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))

        # ---- History turning features ----
        hist_start = max(0, hl - w - 2)
        hist_end = max(0, hl - 2)
        if hist_end > hist_start:
            hist_turn = heading_diff[:, :, hist_start:hist_end, :]  # (B, A, n_steps, K)
            hist_turn_rate = hist_turn.mean(dim=2)  # (B, A, K)
            hist_turn_std = hist_turn.std(dim=2)    # (B, A, K)
            hist_total_turn = hist_turn.sum(dim=2)  # (B, A, K)
        else:
            hist_turn_rate = torch.zeros(B, A, K, device=mu.device)
            hist_turn_std = torch.zeros(B, A, K, device=mu.device)
            hist_total_turn = torch.zeros(B, A, K, device=mu.device)

        # ---- Prediction turning features ----
        pred_start = max(0, hl - 1)
        pred_end = min(T - 2, hl - 1 + w)
        if pred_end > pred_start:
            pred_turn = heading_diff[:, :, pred_start:pred_end, :]  # (B, A, n_steps, K)
            pred_turn_rate = pred_turn.mean(dim=2)  # (B, A, K)
            pred_turn_std = pred_turn.std(dim=2)    # (B, A, K)
            pred_total_turn = pred_turn.sum(dim=2)  # (B, A, K)
        else:
            pred_turn_rate = torch.zeros(B, A, K, device=mu.device)
            pred_turn_std = torch.zeros(B, A, K, device=mu.device)
            pred_total_turn = torch.zeros(B, A, K, device=mu.device)

        return torch.stack([
            hist_turn_rate, hist_turn_std, hist_total_turn,
            pred_turn_rate, pred_turn_std, pred_total_turn,
        ], dim=-1)  # (B, A, K, 6)

    # ------------------------------------------------------------------
    # Node speed features (speed behavior around prediction node)
    # ------------------------------------------------------------------
    def _node_speed_features(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Speed behaviour around the PREDICTION NODE -> (B,A,K,5).

        The base extent features are computed on the future segment only, and
        their step differences start one step INTO the future -- so the
        transition step itself (from the last history point to the first
        predicted point) is not represented at all, and neither is anything on
        the history side. Those are exactly the quantities that decide whether a
        candidate continues the observed motion or breaks from it.

        Indexing: with spd[j] = ||xy[j+1] - xy[j]||, the node step is spd[hl-1],
        the pre-node window is spd[hl-1-w : hl-1] and the post-node window is
        spd[hl-1 : hl-1+w].

        Returns 5 features:
            node_spd:     Speed at the node
            fut_mean:     Mean speed in future window
            fut_accel:    Acceleration in future window
            speed_jump:   Speed discontinuity across the node (fut_mean - hist_mean)
            accel_delta:  Acceleration discontinuity across the node
        """
        xy = mu[..., :2]                                              # (B,A,T,K,2)
        hl = self.hist_len
        w = max(int(self.node_window), 1)

        spd = (xy[:, :, 1:, :, :] - xy[:, :, :-1, :, :]).norm(dim=-1)  # (B,A,T-1,K)
        n_steps = spd.shape[2]
        node_j = min(max(hl - 1, 0), n_steps - 1)

        node_spd = spd[:, :, node_j, :]                                # (B,A,K)

        fut_win = spd[:, :, node_j: min(node_j + w, n_steps), :]       # (B,A,<=w,K)
        h0 = max(node_j - w, 0)
        hist_win = spd[:, :, h0: node_j, :]                            # (B,A,<=w,K)

        fut_mean = fut_win.mean(dim=2)                                 # (B,A,K)
        if fut_win.shape[2] > 1:
            fut_accel = (fut_win[:, :, -1, :] - fut_win[:, :, 0, :]) / (fut_win.shape[2] - 1)
        else:
            fut_accel = torch.zeros_like(fut_mean)

        if hist_win.shape[2] > 0:
            hist_mean = hist_win.mean(dim=2)
            if hist_win.shape[2] > 1:
                hist_accel = (hist_win[:, :, -1, :] - hist_win[:, :, 0, :]) / (hist_win.shape[2] - 1)
            else:
                hist_accel = torch.zeros_like(hist_mean)
        else:
            hist_mean = torch.zeros_like(fut_mean)
            hist_accel = torch.zeros_like(fut_accel)

        speed_jump = fut_mean - hist_mean          # discontinuity across the node
        accel_delta = fut_accel - hist_accel       # does the trend carry over?

        return torch.stack([
            node_spd, fut_mean, fut_accel, speed_jump, accel_delta
        ], dim=-1)  # (B,A,K,5)

    # ------------------------------------------------------------------
    # Extent features (comprehensive motion features)
    # ------------------------------------------------------------------
    def _extent_features(self, mu: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
        xy = mu[..., :2]
        fut = xy[:, :, self.hist_len:, :, :]
        start = xy[:, :, self.hist_len - 1, :, :]
        end = fut[:, :, -1, :, :]
        disp = end - start
        steps = fut[:, :, 1:, :, :] - fut[:, :, :-1, :, :]
        seg = steps.norm(dim=-1)
        arclen = seg.sum(dim=2)
        mean_spd = seg.mean(dim=2)
        final_spd = seg[:, :, -1, :]

        base = torch.cat([
            end, disp,
            arclen.unsqueeze(-1), mean_spd.unsqueeze(-1), final_spd.unsqueeze(-1),
        ], dim=-1)                                                   # (B,A,K,7)

        if self.use_node_features:
            base = torch.cat([base, self._node_speed_features(mu)], dim=-1)

        if self.use_turning_features:
            base = torch.cat([base, self._turning_features(mu)], dim=-1)

        if self.use_sigma_features:
            base = torch.cat([base, self._sigma_features(sigma)], dim=-1)

        # --- permutation-importance hook (probe only; no effect unless set) ---
        perm_dim = getattr(self, "_perm_dim", None)
        if perm_dim is not None and 0 <= perm_dim < base.shape[-1]:
            # base: (B, A, K, extent_dim). Permute feature `perm_dim` across the
            # K axis independently per (B,A), destroying its ability to tell
            # candidates apart while leaving every other feature untouched.
            Bx, Ax, Kx, _ = base.shape
            perm = torch.argsort(torch.rand(Bx, Ax, Kx, device=base.device), dim=-1)
            col = base[..., perm_dim].gather(-1, perm)
            base = base.clone()
            base[..., perm_dim] = col
        # ----------------------------------------------------------------------

        return base

    def _sigma_features(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Model-uncertainty features per candidate -> (B,A,K,3).

        sigma is the decoder's own predicted std (B,A,T,K,D). A candidate the
        model is confident about (small sigma) tended, in earlier analysis, to
        be the better pick -- min-sigma alone outperformed hand-crafted rules,
        and it directly addresses the positive-NLL failure seen with the
        current scorer (it was selecting poorly-calibrated, large-sigma
        candidates). All three are taken from the FUTURE segment only, so they
        do not depend on the (possibly free-running) history region of mu.
            sig_mag:   mean sigma magnitude over the future horizon
            sig_end:   sigma magnitude at the final step (endpoint confidence)
            sig_start: sigma magnitude at the first predicted step (node)
        """
        if sigma is None:
            raise ValueError("sigma is required when use_sigma_features=True")
        fut_sig = sigma[:, :, self.hist_len:, :, :2]           # (B,A,Tp,K,2)
        mag = fut_sig.norm(dim=-1)                             # (B,A,Tp,K)
        sig_mag = mag.mean(dim=2)                              # (B,A,K)
        sig_end = mag[:, :, -1, :]                             # (B,A,K)
        sig_start = mag[:, :, 0, :]                            # (B,A,K)
        return torch.stack([sig_mag, sig_end, sig_start], dim=-1)   # (B,A,K,3)

    # ------------------------------------------------------------------
    # Full trajectory features (flattened future positions + uncertainty)
    # ------------------------------------------------------------------
    def _full_traj_features(self, mu: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
        """
        mu (B,A,T,K,D), sigma (B,A,T,K,D) -> per-candidate FULL future
        trajectory flattened to (B,A,K, Tp*2) for mu, concatenated with the
        flattened future sigma (B,A,K, Tp*2) when sigma is provided, giving
        (B,A,K, Tp*4) overall. Keeps endpoint AND mid-path shape (no
        hand-crafting), plus the model's own per-step uncertainty.
        """
        xy = mu[..., :2]                              # (B,A,T,K,2)
        fut = xy[:, :, self.hist_len:, :, :]          # (B,A,Tp,K,2)
        # Move K before time, then flatten (Tp,2) -> Tp*2
        fut = fut.permute(0, 1, 3, 2, 4)              # (B,A,K,Tp,2)
        B, A, K, Tp, two = fut.shape
        mu_flat = fut.reshape(B, A, K, Tp * two)      # (B,A,K,Tp*2)

        if sigma is None:
            return mu_flat

        sig_xy = sigma[..., :2]                        # (B,A,T,K,2)
        fut_sig = sig_xy[:, :, self.hist_len:, :, :]   # (B,A,Tp,K,2)
        fut_sig = fut_sig.permute(0, 1, 3, 2, 4)       # (B,A,K,Tp,2)
        sig_flat = fut_sig.reshape(B, A, K, Tp * two)  # (B,A,K,Tp*2)

        return torch.cat([mu_flat, sig_flat], dim=-1)  # (B,A,K,Tp*4)

    # ------------------------------------------------------------------
    # Build MLP score head
    # ------------------------------------------------------------------
    def _build_mlp_head(self, gmm_embd: int):
        """
        Build MLP-based score head with support for multiple score_modes.

        score_mode:
            0: Disabled
            1: Decoder input features
            2: Hidden state
            3: Trajectory positions only
            4: Hidden state + trajectory positions
            5: Hidden state + extent features (speed + turning), or full
               trajectory (mu[+sigma]) when key_feature_mode == "full"
        """
        if self.score_mode == 1:
            in_dim = gmm_embd * 4
        elif self.score_mode == 2:
            in_dim = gmm_embd * 4
        elif self.score_mode == 3:
            in_dim = self.num_futures * self.out_dim
        elif self.score_mode == 4:
            in_dim = gmm_embd * 4 + self.num_futures * self.out_dim
        elif self.score_mode == 5:
            # Hidden + per-candidate features (extent, or full traj [+sigma])
            if self.key_feature_mode == "full":
                per_cand = self.pred_len * 2 * (2 if self.use_sigma_features else 1)
            else:
                per_cand = self.extent_dim
            in_dim = gmm_embd * 4 + per_cand * self.num_futures
        else:
            in_dim = gmm_embd * 4
        self._score_in_dim = in_dim

        def _mk():
            return nn.Sequential(
                nn.Linear(in_dim, self.score_hidden),
                nn.GELU(),
                nn.Linear(self.score_hidden, self.num_futures),
            )

        if self.per_mode_score:
            self.score_heads = nn.ModuleList([_mk() for _ in range(self.num_modes_score)])
        else:
            self.score_head = _mk()

    # ------------------------------------------------------------------
    # Build Attention score head
    # ------------------------------------------------------------------
    def _build_attention_head(self, gmm_embd: int):
        """
        Build attention-based score head using query-key mechanism.

        Query: Projected hidden state (intent context)
        Key: Projected extent features of each candidate
        Score: Dot product of query and key
        """
        d = self.score_attn_dim

        # Key feature source: 'extent' (hand-crafted) or 'full' (whole future traj)
        self.key_feature_mode = getattr(self.config, "key_feature_mode", "full")

        if self.key_feature_mode == "full":
            # Use the entire future trajectory as the key input: mu, plus sigma
            # when use_sigma_features is on (both flattened over Tp*2).
            key_in = self.pred_len * 2 * (2 if self.use_sigma_features else 1)
            self._full_key_in = key_in
        else:
            # Use extent features
            key_in = 7 + (5 if self.use_node_features else 0) + (6 if self.use_turning_features else 0) + (3 if self.use_sigma_features else 0)
            self._extent_dim = key_in

        self._key_in = key_in

        def _mk_q():
            return nn.Sequential(
                nn.Linear(4 * gmm_embd, self.score_hidden), nn.GELU(),
                nn.Linear(self.score_hidden, d))

        def _mk_k():
            return nn.Sequential(
                nn.Linear(key_in, self.score_hidden), nn.GELU(),
                nn.Linear(self.score_hidden, d))

        if self.per_mode_score:
            # One independent query/key projection pair per turning mode
            self.q_projs = nn.ModuleList([_mk_q() for _ in range(self.num_modes_score)])
            self.k_projs = nn.ModuleList([_mk_k() for _ in range(self.num_modes_score)])
        else:
            self.q_proj = _mk_q()
            self.k_proj = _mk_k()

        self._attn_scale = 1.0 / math.sqrt(d)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the GMM head.

        Args:
            x: Input tensor (B, A, T, M, C) where M=1 (single mode)

        Returns:
            mu: Predicted trajectory means (B, A, T, K, D)
            sigma: Predicted trajectory variances (B, A, T, K, D)
            score: Candidate scores (B, A, T, K) or None if disabled
        """
        B, A, T, M, C = x.shape

        h = self.future_heads[1](self.future_heads[0](x))    # (B,A,T,1,4C)
        out = self.future_heads[2](h)                        # (B,A,T,1,K*2D)
        out = out.squeeze(3).view(B, A, T, self.num_futures, 2 * self.out_dim)

        mu = out[..., :self.out_dim]
        raw_sigma = out[..., self.out_dim:]
        sigma = F.softplus(raw_sigma) + 1e-3

        score = None
        if self.enable_score_head and self.score_mode != 0:
            if self.score_head_type == "attention":
                score = self._forward_attention(h, mu, sigma, B, A, T)
            else:
                score = self._forward_mlp(x, h, mu, sigma, B, A, T)

        return mu, sigma, score

    # ------------------------------------------------------------------
    # Helper: Get active mode index for per-mode scoring
    # ------------------------------------------------------------------
    def _active_mode_index(self) -> int:
        """Mode index set by the caller before a per-mode forward (default 0)."""
        m_idx = int(getattr(self, "_active_mode", 0))
        return max(0, min(m_idx, self.num_modes_score - 1))

    # ------------------------------------------------------------------
    # MLP score head forward pass
    # ------------------------------------------------------------------
    def _forward_mlp(self, x, h, mu, sigma, B, A, T):
        """
        Forward pass through MLP score head.

        Supports score_mode:
            1: Decoder input features
            2: Hidden state
            3: Trajectory positions only
            4: Hidden state + trajectory positions
            5: Hidden state + extent features (speed + turning), or full
               trajectory (mu[+sigma]) when key_feature_mode == "full"
        """
        if self.score_mode == 1:
            feat = x.squeeze(3)
        elif self.score_mode == 2:
            feat = h.squeeze(3)
        elif self.score_mode == 3:
            feat = mu.detach().reshape(B, A, T, -1)
        elif self.score_mode == 4:
            hidden = h.squeeze(3)
            traj = mu.detach().reshape(B, A, T, -1)
            feat = torch.cat([hidden, traj], dim=-1)
        elif self.score_mode == 5:
            # Extract speed + turning features, or full trajectory features
            hidden = h.squeeze(3)  # (B, A, T, 4*gmm_embd)

            if self.key_feature_mode == "full":
                extent_feat = self._full_traj_features(mu, sigma)  # (B, A, K, Tp*2[*2])
            else:
                extent_feat = self._extent_features(mu, sigma)  # (B, A, K, extent_dim)

            # Expand extent features to time dimension
            extent_feat = extent_feat.unsqueeze(2)  # (B, A, 1, K, feat_dim)
            extent_feat = extent_feat.expand(B, A, T, self.num_futures, -1)  # (B, A, T, K, feat_dim)
            extent_flat = extent_feat.reshape(B, A, T, -1)  # (B, A, T, K * feat_dim)

            if self.score_detach:
                hidden = hidden.detach()
                extent_flat = extent_flat.detach()

            feat = torch.cat([hidden, extent_flat], dim=-1)  # (B, A, T, 4*gmm_embd + K*feat_dim)
        else:
            raise ValueError(f"Invalid score_mode: {self.score_mode}")

        if self.score_detach and self.score_mode not in [5]:
            feat = feat.detach()

        if not self.per_mode_score:
            return self.score_head(feat)                     # (B,A,T,K) shared head
        return self.score_heads[self._active_mode_index()](feat)   # (B,A,T,K)

    # ------------------------------------------------------------------
    # Attention score head forward pass
    # ------------------------------------------------------------------
    def _forward_attention(self, h, mu, sigma, B, A, T):
        """
        Forward pass through attention score head.

        Query: Projected hidden state (intent context averaged over time)
        Key: Projected extent (or full-trajectory [+sigma]) features of
             each candidate
        Score: Dot product of query and key, scaled by sqrt(d)
        """
        hidden = h.squeeze(3)                                # (B,A,T,4C)
        q_ctx = hidden.mean(dim=2)                           # (B,A,4C) intent summary

        if self.key_feature_mode == "full":
            feat = self._full_traj_features(mu, sigma)              # (B,A,K,Tp*2[*2])
            if feat.shape[-1] != self._full_key_in:
                raise ValueError(
                    f"'full' key encoder built for {self._full_key_in} features "
                    f"but got {feat.shape[-1]}. Set GMM config pred_len to the "
                    f"actual future length, and make sure use_sigma_features "
                    f"matches what the head was built with.")
        else:
            feat = self._extent_features(mu, sigma)                 # (B,A,K,7/12/18)

        if self.score_detach:
            q_ctx = q_ctx.detach()
            feat = feat.detach()

        if self.per_mode_score:
            m = self._active_mode_index()
            q = self.q_projs[m](q_ctx)                       # (B,A,d)
            k = self.k_projs[m](feat)                        # (B,A,K,d)
        else:
            q = self.q_proj(q_ctx)                           # (B,A,d)
            k = self.k_proj(feat)                            # (B,A,K,d)

        s = (q.unsqueeze(2) * k).sum(-1) * self._attn_scale  # (B,A,K)
        return s.unsqueeze(2).expand(B, A, T, self.num_futures)  # (B,A,T,K)