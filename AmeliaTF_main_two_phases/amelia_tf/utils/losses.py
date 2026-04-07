import torch
from torch.nn import functional as F
from amelia_tf.utils.utils import separate_ego_agent

def encode_rule_based_to_mode_index(rule_based: torch.Tensor) -> torch.Tensor:
    """
    Encode rule-based turn/speed into a single mode index.
    """
    turn_idx = rule_based[..., :4].float().argmax(dim=-1)   # (B, A)
    speed_idx = rule_based[..., 4:].float().argmax(dim=-1) # (B, A)
    mode_idx = turn_idx * 4 + speed_idx
    return mode_idx.long()


def acceleration_marginal_loss(
    mu: torch.Tensor, sigma: torch.Tensor, mode_probs: torch.Tensor, mode_weights: torch.Tensor,
    target: torch.Tensor, target_mode: torch.Tensor = None, ego_agent: torch.Tensor = None,
    agent_mask: torch.Tensor = None, epoch: int = 0, max_epochs: int = 100,
    mask_historical_steps: int = 10, H: int = 1, 
    lambda_mode: float = 1.0, lambda_marginal: float = 1.0,
    topk: int = 3
) -> torch.Tensor:
    """
    Compute acceleration marginal loss:
      - Cross-entropy for mode prediction (top-k optional)
      - Gaussian NLL for regression (only on true mode)
    """
    B, A, T, D, M, H_check = mu.shape
    assert H == H_check, f"H mismatch: {H} vs {H_check}"
    device = mu.device

    # -------- separate ego agent if provided --------
    if ego_agent is not None:
        A = 1
        mu = separate_ego_agent(mu, ego_agent)
        sigma = separate_ego_agent(sigma, ego_agent)
        mode_probs = separate_ego_agent(mode_probs, ego_agent)
        target = separate_ego_agent(target, ego_agent)
        if target_mode is not None:
            target_mode = separate_ego_agent(target_mode, ego_agent)
        if agent_mask is not None:
            agent_mask = separate_ego_agent(agent_mask, ego_agent)

    # -------- Cross-entropy on mode prediction --------
    if target_mode is not None:
        target_mode_idx = encode_rule_based_to_mode_index(target_mode)
        #loss_mode = F.cross_entropy(
        #    mode_probs.view(B * A, M),
        #    target_mode_idx.view(B * A),
        #    label_smoothing=0.1,
        #   reduction='mean'
        #)
        loss_mode = F.cross_entropy(
            mode_probs.view(B * A, M),
            target_mode_idx.view(B * A),
            weight=mode_weights,        # ?? ??
            label_smoothing=0.1,
            reduction='mean'
        )


        # -------- regression only on true mode --------
        idx_expanded = (
            target_mode_idx
            .view(B, A, 1, 1, 1, 1)
            .expand(B, A, T, D, 1, H)
        )
        
        mu_true = torch.gather(mu, dim=4, index=idx_expanded).squeeze(4)
        sigma_true = torch.gather(sigma, dim=4, index=idx_expanded).squeeze(4)

    else:
        loss_mode = 0.0
        # fallback: mean over modes
        mu_true = mu.mean(dim=4)
        sigma_true = sigma.mean(dim=4)

    # -------- mean over H if needed --------
    mu_true = mu_true.mean(dim=-1) if H > 1 else mu_true.squeeze(-1)
    sigma_true = sigma_true.mean(dim=-1) if H > 1 else sigma_true.squeeze(-1)

    # -------- variance --------
    var = (sigma_true ** 2).clamp_min(1e-4)

    # -------- temporal & agent mask --------
    temporal_mask = torch.ones(B, A, T, 1, device=device, dtype=torch.bool)
    if mask_historical_steps > 0:
        temporal_mask[:, :, :min(mask_historical_steps, T)] = False

    combined_mask = temporal_mask
    if agent_mask is not None:
        combined_mask = combined_mask & agent_mask.unsqueeze(-1)

    # -------- Gaussian NLL on true mode --------
    loss_reg = F.gaussian_nll_loss(
        mu_true,
        target,
        var,
        reduction='none'
    )
    loss_reg = (loss_reg * combined_mask.expand_as(loss_reg)).sum() / combined_mask.sum().clamp_min(1)

    # -------- total loss --------
    total_loss = lambda_mode * loss_mode + lambda_marginal * loss_reg
    return total_loss, lambda_mode * loss_mode,  lambda_marginal * loss_reg



def acceleration_marginal_loss2(
        pred_scores: torch.tensor,
    traj_mu: torch.tensor,
    traj_sigma: torch.tensor,
    target_traj: torch.tensor,
    ego_agent: torch.tensor = None,
    agent_mask: torch.tensor = None,
    epoch: int = 0,
    max_epochs: int = 100,
    mask_historical_steps: int = 10
) -> torch.tensor:
    """
    Marginal loss considering only trajectory terms,
    ignoring the first `mask_historical_steps` timesteps (historical horizon).
    """
    B, A, T_traj, N, D = traj_mu.size()

    # ===== 1. Handle ego-agent slicing if provided =====
    if ego_agent is not None:
        A = 1
        assert ego_agent.shape[0] == traj_mu.shape[0]
        traj_mu = separate_ego_agent(traj_mu, ego_agent)
        traj_sigma = separate_ego_agent(traj_sigma, ego_agent)
        pred_scores = separate_ego_agent(pred_scores, ego_agent)
        target_traj = separate_ego_agent(target_traj, ego_agent)
        agent_mask = None if agent_mask is None else separate_ego_agent(agent_mask, ego_agent)

    # ===== 2. Create temporal mask for historical timesteps =====
    traj_temporal_mask = torch.ones(B, A, T_traj, 1, 1, device=traj_mu.device, dtype=torch.bool)
    
    if mask_historical_steps > 0:
        traj_temporal_mask[:, :, :min(mask_historical_steps, T_traj), :, :] = False

    # ===== 3. Compute trajectory distances =====
    traj_distance = (traj_mu - target_traj[..., None, :]).norm(dim=-1)  # [B, A, T_traj, N]

    # ===== 4. Combine agent and temporal masks =====
    if agent_mask is not None:
        agent_mask = agent_mask.view(B, A, T_traj, 1, 1)
        traj_combined_mask = agent_mask & traj_temporal_mask
    else:
        traj_combined_mask = traj_temporal_mask
    
    # ===== 5. Apply mask when aggregating distances =====
    def masked_mean_distance(distance, mask):
        mask = mask.squeeze(-1)
        mask = mask.expand_as(distance)
        masked_dist = distance * mask.squeeze(-1).squeeze(-1)
        valid_steps = mask.sum(dim=2).clamp_min(1)
        return masked_dist.sum(dim=2) / valid_steps.squeeze(-1).squeeze(-1)

    agg_traj_distance = masked_mean_distance(traj_distance, traj_combined_mask)

    # ===== 6. Select best mode =====
    gt_idx = agg_traj_distance.argmin(dim=-1)  # [B, A]

    # ===== 7. Build one-hot mode mask =====
    traj_mask = F.one_hot(gt_idx, num_classes=N)[..., None, :, None].repeat(1, 1, T_traj, 1, D)
    traj_mask = traj_mask & traj_combined_mask

    masked_traj_mu = traj_mu * traj_mask
    masked_traj_sigma = traj_sigma * traj_mask
    masked_target_traj = target_traj[..., None, :].repeat(1, 1, 1, N, 1) * traj_mask

    # ===== 8. Compute final losses =====
    loss_cls = F.cross_entropy(
        input=pred_scores.flatten(0, 1),
        target=gt_idx.flatten(),
        reduction='mean',
        ignore_index=N
    )

    loss_traj_reg = F.gaussian_nll_loss(
        masked_traj_mu, 
        masked_target_traj, 
        masked_traj_sigma
    )

    total_loss = loss_cls + loss_traj_reg

    return total_loss

def joint_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    epoch: int = 0, max_epochs: int = 100, add_diversity: bool = True, ego_agent = None
) -> torch.tensor:
    """ Computes a classification and regression loss for the scene jointly. We treat each future to
    be coherent futures across all agents. Thus, we aggregate the loss across all agents and time
    steps. We only back-propagate the loss through the individual future that most closely matches
    the ground-truth in terms of displacement loss.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the joint loss.

    """
    B, A, T, N, D = mu.size()

    # distance: (B, A, T, N, D) -> (B, A, T, N)
    distance = (mu - target[...,None,:]).norm(dim=-1)
    # agg_distance: (B, A, T, N) -> (B, A, N) -> (B, N)
    agg_distance = distance.mean(dim=(2, 1))

    # index of joint future with smallest error
    # gt_idx: (B)
    gt_idx = agg_distance.argmin(dim=-1)

    # select the correct joint future; mask all else
    mask = F.one_hot(gt_idx, num_classes = N)[:, None, None, :, None].repeat(1, A, T, 1, D)
    mu = mu * mask
    sigma = sigma * mask
    target = target[..., None, :].repeat(1, 1, 1, N, 1) * mask

    loss_cls = F.cross_entropy(
        input=pred_scores.flatten(0, 1), target=gt_idx[:, None].repeat(1, A).flatten(),
        reduction='mean', ignore_index=N)
    loss_reg = F.gaussian_nll_loss(mu, target, sigma)

    return loss_cls + loss_reg

def diversity_loss(pred: torch.tensor, sigma_d: float = 0.001) -> torch.tensor:
    B, A, T, N, D = pred.shape
    # ----------------------
    # TODO: vectorize
    # ----------------------
    diversity_loss = 0.0
    for n1 in range(N):
        for n2 in range(N):
            if n1 == n2:
                continue

            y_n1 = pred[:, :, :, n1]
            y_n2 = pred[:, :, :, n2]
            diversity_loss += torch.exp(-(y_n1 - y_n2).norm(dim=1) / sigma_d).mean(dim=(2, 1))

    return (diversity_loss / (N * (N - 1))).sum()


def lmbd_marginal_joint_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    lmbd: float = 0.5, epoch: int = 0, max_epochs: int = 100, add_diversity: bool = True, ego_agent = None
) -> torch.tensor:
    """ Computes a classification and regression loss for the scene marginally and jointly. We treat
    each future to be coherent futures across all agents. Thus, we aggregate the loss across all agents
    and time steps. However, we also consider each element independently.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the joint loss.

    """
    assert lmbd >= 0 and lmbd < 1.0
    m = marginal_loss(pred_scores, mu, sigma, target)
    j = joint_loss(pred_scores, mu, sigma, target)

    if add_diversity:
        d = diversity_loss(mu)
        return (1.0 - lmbd) * m + lmbd * j + 0.1 * d

    return lmbd * m + (1 - lmbd) * j

def weighted_marginal_joint_loss(
    pred_scores: torch.tensor, mu: torch.tensor, sigma: torch.tensor, target: torch.tensor,
    lmbd: float = 0.5, epoch: int = 0, max_epochs: int = 100, add_diversity: bool = True
) -> torch.tensor:
    """ Computes a classification and regression loss for the scene marginally and jointly. We treat
    each future to be coherent futures across all agents. Thus, we aggregate the loss across all agents
    and time steps. However, we also consider each element independently.
    We follow these references:
        - MTR: https://arxiv.org/pdf/2209.13508.pdf
        - SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf

    Inputs:
    -------
        pred_scores[torch.tensor(B, A, T, N)]: tensor containing the predictions scores.
            B: batch size
            A: number of agents
            T: trajectory length
            N: number of predicted heads
        mu[torch.tensor(B, A, T, N, D)]: tensor containing the prediction means.
            D: number of dimensions
        sigma[torch.tensor(B, A, T, N, D)]: tensor containing the prediction standard deviations.
        target[torch.tensor(B, A, T, D)]: tensor containing the ground truth trajectory.

    Output
    ------
        error[torch.tensor]: scalar value representing the joint loss.

    """
    assert epoch > 0
    w = epoch / max_epochs

    m = marginal_loss(pred_scores, mu, sigma, target)
    j = joint_loss(pred_scores, mu, sigma, target)

    if add_diversity:
        d = diversity_loss(mu)
        return (1.0 - w) * m + w * j + 0.1 * d

    return (1.0 - w) * m + w * j