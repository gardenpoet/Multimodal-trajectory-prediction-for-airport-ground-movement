"""
Shared hypothesis-scorer feature extraction.

CRITICAL: this exact function is used at BOTH training time (train_scorer.py)
and inference time (OffRoadSelector.selection_mode='learned'). Any change here
must be made in one place only, so training and inference features never diverge.

The scorer answers: among the K trajectory hypotheses of a given (sample, mode),
which one is closest to the ground truth? Features are purely closed-book
(predicted trajectories, predicted sigma, history speed) -- no ground truth.
"""
import numpy as np


# Number of per-hypothesis feature channels (keep in sync with _per_k_feats below).
N_PERK_FEATS = 11


def _per_k_feats(ego_mu, ego_sigma, hist_len):
    """
    Per-hypothesis geometric/kinematic features.

    Args:
        ego_mu:    (B, T_total, M, K, D) full-K predicted means, model rel frame.
        ego_sigma: (B, T_total, M, K, D) full-K predicted sigmas.
        hist_len:  number of history steps.

    Returns:
        list of (B, M, K) arrays, length N_PERK_FEATS.
    """
    B, T_total, M, K, D = ego_mu.shape
    T_pred = T_total - hist_len

    fut = ego_mu[:, hist_len:, :, :, :2]            # (B, Tp, M, K, 2)
    fsig = ego_sigma[:, hist_len:, :, :, :2]        # (B, Tp, M, K, 2)

    step = np.linalg.norm(np.diff(fut, axis=1), axis=-1)   # (B, Tp-1, M, K)
    arc = step.sum(1)                                       # (B, M, K) path length
    n3 = max((T_pred - 1) // 3, 1)
    vf = step[:, :n3].mean(1)                               # early speed
    vm = step[:, n3:2 * n3].mean(1)                         # mid speed
    vl = step[:, -n3:].mean(1)                              # late speed
    accel = vl / (vf + 1e-9)                                # accel ratio

    end = fut[:, -1]                                        # (B, M, K, 2) endpoint
    along = end[..., 0]                                     # along-track (model x)
    cross = end[..., 1]                                     # cross-track (model y)

    dx = np.diff(fut[..., 0], axis=1)
    dy = np.diff(fut[..., 1], axis=1)
    turn = np.abs(((np.diff(np.arctan2(dy, dx), axis=1) + np.pi)
                   % (2 * np.pi)) - np.pi).sum(1)           # (B, M, K) total |heading change|

    sig_end = np.linalg.norm(fsig[:, -1], axis=-1)          # (B, M, K) end-step sigma
    sig_mean = np.linalg.norm(fsig, axis=-1).mean(1)        # (B, M, K) mean sigma

    return [arc, along, cross, accel, turn, vf, vm, vl, sig_end, sig_mean]


def compute_hyp_features(ego_mu, ego_sigma, seq_ego, hist_len):
    """
    Build the per-(sample, mode) feature matrix for the hypothesis scorer.

    Args:
        ego_mu:    (B, T_total, M, K, D) full-K predicted means (numpy float).
        ego_sigma: (B, T_total, M, K, D) full-K predicted sigmas (numpy float).
        seq_ego:   (B, T_total, 9) ego raw sequence; col 0 = speed (knots),
                   cols 6:8 = local xy (km).
        hist_len:  number of history steps.

    Returns:
        X: (B, M, F) feature tensor, F = K*N_PERK_FEATS_used + 2 + M.
           Each (sample, mode) row holds the K hypotheses' features (z-normalized
           across K, so the model sees *relative* differences) plus the raw
           end-step sigma, plus sample-level features (history speed, history
           step) and a one-hot mode indicator.
    """
    ego_mu = np.asarray(ego_mu, dtype=np.float64)
    ego_sigma = np.asarray(ego_sigma, dtype=np.float64)
    seq_ego = np.asarray(seq_ego, dtype=np.float64)
    B, T_total, M, K, D = ego_mu.shape

    feats = _per_k_feats(ego_mu, ego_sigma, hist_len)
    arc, along, cross, accel, turn, vf, vm, vl, sig_end, sig_mean = feats

    def znorm(a):
        mu = a.mean(2, keepdims=True)
        sd = a.std(2, keepdims=True) + 1e-9
        return (a - mu) / sd

    # z-normalized channels across K + raw sig_end (absolute magnitude matters)
    perK = [znorm(arc), znorm(along), znorm(cross), znorm(accel), znorm(turn),
            znorm(vf), znorm(vm), znorm(vl), znorm(sig_end), sig_end, znorm(sig_mean)]

    Xk = np.stack(perK, axis=-1)             # (B, M, K, F_perK)
    F_perK = Xk.shape[-1]
    Xk = Xk.reshape(B, M, K * F_perK)        # (B, M, K*F_perK)

    # sample-level features
    mhs = seq_ego[:, :hist_len, 0].mean(1)                          # history mean speed (knots)
    hxy = seq_ego[:, :hist_len, 6:8]
    hstep = np.linalg.norm(np.diff(hxy, axis=1), axis=-1).mean(1)   # history mean per-step (km)
    samp = np.stack([mhs, hstep], 1)[:, None, :].repeat(M, 1)       # (B, M, 2)
    modeoh = np.eye(M)[None].repeat(B, 0)                           # (B, M, M)

    X = np.concatenate([Xk, samp, modeoh], axis=-1)                # (B, M, F)
    return X


def oracle_ade_labels(ego_mu, ego_fut, hist_len):
    """
    Training labels: for each (sample, mode), the hypothesis k with minimum ADE
    to the ground-truth future. Requires GT, so used at TRAINING time only.

    Args:
        ego_mu:  (B, T_total, M, K, D) full-K predicted means.
        ego_fut: (B, T_pred, D) GT future in the same model rel frame.
        hist_len: number of history steps.

    Returns:
        y: (B, M) int array of oracle-by-ADE hypothesis indices.
    """
    ego_mu = np.asarray(ego_mu, dtype=np.float64)
    ego_fut = np.asarray(ego_fut, dtype=np.float64)
    fut = ego_mu[:, hist_len:, :, :, :2]            # (B, Tp, M, K, 2)
    gt = ego_fut[:, :, :2]                          # (B, Tp, 2)
    ade = np.sqrt(((fut - gt[:, :, None, None, :]) ** 2).sum(-1)).mean(1)  # (B, M, K)
    return ade.argmin(2)
