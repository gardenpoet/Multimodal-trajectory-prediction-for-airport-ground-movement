"""
Shared hypothesis-feature extraction for the tree scorer.

Both the tree TRAINING (eval_two_stage._stage_tree) and the TEST-time tree
selection (CombinedTrajPredSystem, selection_mode='tree') import this single
function, so the 50-dim feature vector the tree sees is guaranteed identical in
both places. Any drift between a re-implementation here and there would feed the
tree out-of-distribution features and silently wreck the selection, so there
must be exactly one implementation.

Feature layout (must match the original hyp_features.py), per (sample, mode):
  per-hypothesis (11 channels x K, z-normalized across K except raw sig_end):
    arc, along, cross, accel, turn, vf, vm, vl, sig_end_z, sig_end_raw, sig_mean_z
  sample-level (2): hist_mean_speed, hist_step
  mode one-hot (num_modes)
Total = 11*K + 2 + num_modes  (= 50 for K=4, num_modes=4).
"""
import torch


def hyp_features_and_winner(mu, sigma, ego_fut, fut_mask,
                            hist_mean_speed, hist_step,
                            mode_idx, num_modes, hist_len, use_mask=True):
    """
    mu       : (B,1,T,K,2+) candidate means for ONE mode (ego-separated)
    sigma    : (B,1,T,K,2+) candidate stds for the same mode
    ego_fut  : (B,1,Tp,2+)  GT future (only used for the winner label; pass
                             zeros at test time and ignore the returned winner)
    fut_mask : (B,1,Tp)     valid-timestep mask
    hist_mean_speed, hist_step : (B,) sample-level history features
    mode_idx : int          which mode this block is (one-hot source)

    Returns:
      feats  : (B, 11*K + 2 + num_modes)
      winner : (B,)  min-ADE candidate index (meaningful only if ego_fut is GT)
    """
    B = mu.shape[0]
    K = mu.shape[3]
    hl = hist_len
    xy = mu[:, 0, :, :, :2]                     # (B,T,K,2)
    fut = xy[:, hl:, :, :]                      # (B,Tp,K,2)
    end = fut[:, -1, :, :]                      # (B,K,2)
    fsig = sigma[:, 0, hl:, :, :2]              # (B,Tp,K,2)
    m = fut_mask[:, 0, :]                       # (B,Tp)
    m3 = m[:, :, None]                          # (B,Tp,1)
    denom = m.sum(1).clamp_min(1)               # (B,)

    gt = ego_fut[:, 0, :, :2]                   # (B,Tp,2)
    d_pt = ((fut - gt[:, :, None, :]) ** 2).sum(-1).sqrt()   # (B,Tp,K)
    if use_mask:
        ade_k = (d_pt * m3).sum(1) / denom[:, None]
    else:
        ade_k = d_pt.mean(1)
    winner = ade_k.argmin(-1)                   # (B,)

    step = (fut[:, 1:, :, :] - fut[:, :-1, :, :]).norm(dim=-1)   # (B,Tp-1,K)
    Tp = fut.shape[1]
    n3 = max((Tp - 1) // 3, 1)

    arc = step.sum(1)
    vf = step[:, :n3, :].mean(1)
    vm = step[:, n3:2 * n3, :].mean(1)
    vl = step[:, -n3:, :].mean(1)
    accel = vl / (vf + 1e-9)

    along = end[..., 0]
    cross = end[..., 1]

    dxy = fut[:, 1:, :, :] - fut[:, :-1, :, :]
    ang = torch.atan2(dxy[..., 1], dxy[..., 0])
    dang = ang[:, 1:, :] - ang[:, :-1, :]
    dang = ((dang + torch.pi) % (2 * torch.pi)) - torch.pi
    turn = dang.abs().sum(1)

    sig_end = fsig[:, -1, :, :].norm(dim=-1)
    sig_mean = fsig.norm(dim=-1).mean(1)

    def znorm(t):
        mu_ = t.mean(-1, keepdim=True)
        sd_ = t.std(-1, keepdim=True) + 1e-9
        return (t - mu_) / sd_

    per_hyp = torch.stack([
        znorm(arc), znorm(along), znorm(cross), znorm(accel), znorm(turn),
        znorm(vf), znorm(vm), znorm(vl),
        znorm(sig_end), sig_end, znorm(sig_mean),
    ], dim=-1)                                              # (B,K,11)
    per_hyp = per_hyp.reshape(B, K * 11)

    samp = torch.stack([hist_mean_speed, hist_step], dim=-1)
    oh = torch.zeros(B, num_modes, device=mu.device)
    oh[:, mode_idx] = 1.0

    feats = torch.cat([per_hyp, samp, oh], dim=-1)
    return feats, winner
