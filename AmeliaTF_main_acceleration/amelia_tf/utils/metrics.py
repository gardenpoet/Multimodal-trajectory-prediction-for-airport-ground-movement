"""
Modal classification metrics for trajectory prediction.
Provides comprehensive evaluation of modality prediction as a classification task.
"""
import torch
from typing import Dict, Optional, List
from collections.abc import Sequence


class ModalClassificationMetrics:
    """
    Accumulates modal classification metrics across batches and computes final metrics.

    This class accumulates a confusion matrix across all batches and computes
    precision, recall, F1, and other classification metrics for modality prediction.

    Example:
        >>> metrics = ModalClassificationMetrics(num_modes=5)
        >>> for batch in dataloader:
        ...     metrics.update(Y_hat, pred_scores, Y, mask)
        >>> results = metrics.compute()
        >>> metrics.print_report()
    """

    def __init__(self, num_modes: int):
        """
        Args:
            num_modes: Number of prediction modes (H)
        """
        self.num_modes = num_modes
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.confusion_matrix = torch.zeros((self.num_modes, self.num_modes), dtype=torch.float32)
        self.total_samples = 0

    def update(
            self,
            Y_hat: torch.Tensor,
            pred_scores: torch.Tensor,
            Y: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ):
        """
        Update accumulated statistics with a new batch.

        Args:
            Y_hat: [B, A, T_total, H, D] predicted trajectories (only last T steps used)
            pred_scores: [B, A, H] predicted modality probabilities (softmax)
            Y: [B, A, T, D] ground truth future trajectories
            mask: [B, A, T_total] agent validity mask
        """
        B, A, T, D = Y.size()
        H = pred_scores.size(-1)
        assert H == self.num_modes, f"Expected {self.num_modes} modes, got {H}"

        # Compute per-mode errors (L2 distance)
        error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1)  # [B, A, T, H]

        # Average over time with mask handling (matching marginal_ade style)
        if mask is None:
            error = error.mean(dim=2)  # [B, A, H]
        else:
            mask = mask[:, :, -T:]
            error = error.view(B * A, T, -1)
            BA, T_, H_ = error.shape
            mask = mask.view(BA, T_)
            error_masked = torch.zeros(BA, H_, device=error.device)
            for ba in range(BA):
                amask = mask[ba]
                if amask.any():
                    error_masked[ba] = error[ba, amask].mean(dim=0)
                else:
                    error_masked[ba] = error[ba].mean(dim=0)  # fallback
            error = error_masked.view(B, A, H)

        # Get predicted and ground truth modalities
        pred_modes = pred_scores.argmax(dim=-1)  # [B, A] - predicted as max probability
        true_modes = error.argmin(dim=-1)  # [B, A] - ground truth as min error

        # Create valid agent mask
        valid_agents = torch.ones_like(pred_modes, dtype=torch.bool)
        if mask is not None:
            valid_agents = mask.any(dim=-1)

        # Flatten and filter valid agents
        pred_flat = pred_modes[valid_agents].cpu()
        true_flat = true_modes[valid_agents].cpu()

        # Update confusion matrix
        for t, p in zip(true_flat, pred_flat):
            self.confusion_matrix[t.long(), p.long()] += 1

        self.total_samples += len(true_flat)

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics from accumulated confusion matrix.

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - precision_per_mode: [H] precision for each mode
                - recall_per_mode: [H] recall for each mode
                - f1_per_mode: [H] F1 score for each mode
                - support_per_mode: [H] number of true instances per mode
                - macro_precision: Unweighted mean precision
                - macro_recall: Unweighted mean recall
                - macro_f1: Unweighted mean F1
                - weighted_precision: Support-weighted precision
                - weighted_recall: Support-weighted recall
                - weighted_f1: Support-weighted F1
                - confusion_matrix: [H, H] confusion matrix
                - total_samples: Total number of valid samples
        """
        cm = self.confusion_matrix
        eps = 1e-10

        # True positives per mode
        tp = torch.diag(cm)

        # False positives per mode (column sums - TP)
        fp = cm.sum(dim=0) - tp

        # False negatives per mode (row sums - TP)
        fn = cm.sum(dim=1) - tp

        # Support (actual instances per mode)
        support = cm.sum(dim=1)

        # Per-mode metrics
        precision_per_mode = tp / (tp + fp + eps)
        recall_per_mode = tp / (tp + fn + eps)
        f1_per_mode = 2 * precision_per_mode * recall_per_mode / (precision_per_mode + recall_per_mode + eps)

        # Overall accuracy
        accuracy = tp.sum() / (cm.sum() + eps)

        # Macro averages (unweighted)
        macro_precision = precision_per_mode.mean()
        macro_recall = recall_per_mode.mean()
        macro_f1 = f1_per_mode.mean()

        # Weighted averages
        total_samples = support.sum() + eps
        weighted_precision = (precision_per_mode * support).sum() / total_samples
        weighted_recall = (recall_per_mode * support).sum() / total_samples
        weighted_f1 = (f1_per_mode * support).sum() / total_samples

        return {
            'accuracy': accuracy,
            'precision_per_mode': precision_per_mode,
            'recall_per_mode': recall_per_mode,
            'f1_per_mode': f1_per_mode,
            'support_per_mode': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm,
            'total_samples': self.total_samples
        }

    def print_report(self, mode_names: Optional[List[str]] = None):
        """Print a scikit-learn style classification report."""
        metrics = self.compute()

        if mode_names is None:
            mode_names = [f'Mode {i}' for i in range(self.num_modes)]

        print("\n" + "=" * 80)
        print(f"MODALITY CLASSIFICATION REPORT (Total samples: {metrics['total_samples']})")
        print("=" * 80)
        print(f"{'':15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
        print("-" * 80)

        for i in range(self.num_modes):
            print(f"{mode_names[i]:15} "
                  f"{metrics['precision_per_mode'][i]:12.4f} "
                  f"{metrics['recall_per_mode'][i]:12.4f} "
                  f"{metrics['f1_per_mode'][i]:12.4f} "
                  f"{metrics['support_per_mode'][i]:12.0f}")

        print("-" * 80)
        print(f"{'Macro Avg':15} "
              f"{metrics['macro_precision']:12.4f} "
              f"{metrics['macro_recall']:12.4f} "
              f"{metrics['macro_f1']:12.4f} ")
        print(f"{'Weighted Avg':15} "
              f"{metrics['weighted_precision']:12.4f} "
              f"{metrics['weighted_recall']:12.4f} "
              f"{metrics['weighted_f1']:12.4f} ")
        print(f"{'Accuracy':15} {metrics['accuracy']:37.4f}")
        print("=" * 80)

        # Print confusion matrix
        self.print_confusion_matrix(mode_names)

    def print_confusion_matrix(self, mode_names: Optional[List[str]] = None):
        """Print the confusion matrix."""
        metrics = self.compute()
        cm = metrics['confusion_matrix']

        if mode_names is None:
            mode_names = [f'M{i}' for i in range(self.num_modes)]

        print("\nConfusion Matrix:")
        print("-" * (12 + 10 * self.num_modes))
        print("True\\Pred", end=" ")
        for name in mode_names:
            print(f"{name:>8}", end=" ")
        print()

        for i in range(self.num_modes):
            print(f"{mode_names[i]:8}", end=" ")
            for j in range(self.num_modes):
                print(f"{cm[i, j]:8.0f}", end=" ")
            print()
        print("-" * (12 + 10 * self.num_modes))


def compute_mode_accuracy(Y_hat: torch.Tensor, pred_scores: torch.Tensor, Y: torch.Tensor,
                          mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes mode classification accuracy: whether the highest-probability trajectory
    matches the ground-truth closest trajectory.

    Args:
        Y_hat: [B, A, T_total, H, D] predicted trajectories (only last T steps used)
        pred_scores: [B, A, H] predicted modality probabilities
        Y: [B, A, T, D] ground truth future trajectories
        mask: [B, A, T_total] agent validity mask (only last T steps used)

    Returns:
        accuracy: [B, A] binary tensor (1 if predicted best mode == GT best mode)
    """
    B, A, T, D = Y.size()

    # Compute L2 distance for each modality (matches marginal_ade)
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1)  # [B, A, T, H]

    if mask is None:
        # Average over time dimension
        error = error.mean(dim=2)  # [B, A, H]
    else:
        # Exact marginal_ade mask handling: per-agent time averaging
        mask = mask[:, :, -T:]
        error = error.view(B * A, T, -1)
        BA, T_, H = error.shape
        mask = mask.view(BA, T_)
        error_masked = torch.zeros(BA, H, device=error.device)
        for ba in range(BA):
            amask = mask[ba]
            error_masked[ba] = error[ba, amask].mean(dim=0)
        error = error_masked.view(B, A, H)

    # GT best modality (closest to ground truth) vs predicted best modality
    best_modal = error.argmin(dim=-1)  # [B, A]
    prob_max_modal = pred_scores.argmax(dim=-1)  # [B, A]
    acc = (best_modal == prob_max_modal).float()
    return (best_modal == prob_max_modal).float().mean()  # [B, A]


def compute_nll(Y_hat: torch.Tensor, sigma: torch.Tensor, pred_scores: torch.Tensor,
                Y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes multi-modal Gaussian mixture Negative Log-Likelihood (NLL).
    Args:
        Y_hat: [B, A, T_total, H, D] predicted trajectory means (only last T used)
        sigma: [B, A, T_total, H, D] predicted trajectory std devs (only last T used)
        pred_scores: [B, A, H] modality mixing coefficients (softmax probabilities)
        Y: [B, A, T, D] ground truth future trajectories
        mask: [B, A, T_total] agent validity mask (only last T steps used)
    Returns:
        nll: [B, A] negative log-likelihood for each agent
    """
    B, A, T, D = Y.size()

    # Per-modality Gaussian log-likelihoods
    diff = Y[..., None, :] - Y_hat[..., -T:, :, :]  # [B, A, T, H, D]
    var = sigma[..., -T:, :, :].pow(2).clamp_min(1e-6)
    logp_mode = -0.5 * (torch.log(2 * torch.pi * var) + diff.pow(2) / var)
    logp_mode = logp_mode.sum(dim=-1)  # [B, A, T, H]

    # Add log mixture weights BEFORE logsumexp (per timestep)
    log_pi = torch.log(pred_scores.clamp_min(1e-8))  # [B, A, H]
    logp_mix = logp_mode + log_pi[:, :, None, :]  # [B, A, T, H]

    # Mixture over modes: log sum_h exp(logp + log_pi) ? [B, A, T]
    logp_t = torch.logsumexp(logp_mix, dim=-1)  # [B, A, T]
    nll_t = -logp_t  # [B, A, T]

    # Average over time with optional mask
    if mask is None:
        nll = nll_t.mean(dim=-1)  # [B, A]
    else:
        mask_pred = mask[:, :, -T:]  # [B, A, T]
        nll = (nll_t * mask_pred).sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)

    return nll  # [B, A]

def compute_mode_rmse(Y_hat: torch.Tensor, pred_scores: torch.Tensor, Y: torch.Tensor,
                      mask: torch.Tensor = None, scale: float = 1000.0) -> torch.Tensor:
    """
    Computes RMSE of highest-probability trajectory (matches marginal_ade return shape).

    Args:
        Y_hat: [B, A, T_total, H, D] predicted trajectories (only last T steps used)
        pred_scores: [B, A, H] predicted modality probabilities
        Y: [B, A, T, D] ground truth future trajectories
        mask: [B, A, T_total] agent validity mask (only last T steps used)
        scale: scaling factor for final RMSE

    Returns:
        rmse: [B, A] RMSE for each agent using its highest-probability trajectory
    """
    B, A, T, D = Y.size()

    # Select highest-probability trajectory for each agent
    prob_max_idx = pred_scores.argmax(dim=-1, keepdim=True)  # [B, A, 1]
    best_traj = torch.gather(
        Y_hat[..., -T:, :, :], 3,
        prob_max_idx.unsqueeze(2).unsqueeze(-1).expand(-1, -1, T, 1, D)
    )[..., 0, :]  # [B, A, T, D]

    # Squared error with marginal_ade-style mask handling
    error_sq = ((best_traj - Y).norm(dim=-1)).pow(2)
    if mask is None:
        error_sq = error_sq.mean(dim=(2))  # [B, A]
    else:
        mask = mask[:, :, -T:]
        error_sq = error_sq.view(B * A, T)
        BA, T_ = error_sq.shape
        mask = mask.view(BA, T_)
        error_masked = torch.zeros(BA, device=error_sq.device)
        for ba in range(BA):
            amask = mask[ba]
            error_masked[ba] = error_sq[ba, amask].mean(dim=0)
        error_sq = error_masked.view(B, A)

    rmse = torch.sqrt(error_sq)  # [B, A]
    return (scale * rmse)
    
def mode_ade(
    Y_hat: torch.Tensor, 
    pred_scores: torch.Tensor, 
    Y: torch.Tensor, 
    mask: torch.Tensor = None, 
    scale: float = 1000.0
) -> torch.Tensor:
    """
    Computes ADE using the highest-probability trajectory (matches compute_mode_rmse logic).
    
    Args:
        Y_hat: [B, A, T_total, H, D] predicted trajectories (only last T steps used)
        pred_scores: [B, A, H] predicted modality probabilities
        Y: [B, A, T, D] ground truth future trajectories
        mask: [B, A, T_total] agent validity mask (only last T steps used)
        scale: scaling factor for final ADE
    
    Returns:
        ade: [B, A] ADE for each agent using its highest-probability trajectory
    """
    B, A, T, D = Y.size()
    
    # Select highest-probability trajectory for each agent
    prob_max_idx = pred_scores.argmax(dim=-1, keepdim=True)  # [B, A, 1]
    best_traj = torch.gather(
        Y_hat[..., -T:, :, :], 3,
        prob_max_idx.unsqueeze(2).unsqueeze(-1).expand(-1, -1, T, 1, D)
    )[..., 0, :]  # [B, A, T, D]
    
    # Compute displacement error per time step
    error = (best_traj - Y).norm(dim=-1)  # [B, A, T]
    
    # Average over time with optional mask
    if mask is None:
        ade = error.mean(dim=-1)  # [B, A]
    else:
        mask_pred = mask[:, :, -T:]  # [B, A, T]
        ade = (error * mask_pred).sum(dim=-1) / mask_pred.sum(dim=-1).clamp_min(1)
    
    return scale * ade  # [B, A]


def mode_fde(
    Y_hat: torch.Tensor, 
    pred_scores: torch.Tensor, 
    Y: torch.Tensor, 
    mask: torch.Tensor = None, 
    scale: float = 1000.0
) -> torch.Tensor:
    """
    Computes FDE using the highest-probability trajectory (matches compute_mode_rmse logic).
    
    Args:
        Y_hat: [B, A, T_total, H, D] predicted trajectories (only last T steps used)
        pred_scores: [B, A, H] predicted modality probabilities
        Y: [B, A, T, D] ground truth future trajectories
        mask: [B, A, T_total] agent validity mask (only last T steps used)
        scale: scaling factor for final FDE
    
    Returns:
        fde: [B, A] FDE for each agent using its highest-probability trajectory
    """
    B, A, T, D = Y.size()
    
    # Select highest-probability trajectory for each agent
    prob_max_idx = pred_scores.argmax(dim=-1, keepdim=True)  # [B, A, 1]
    
    # Get final time step prediction
    if mask is None:
        # Use last time step
        best_traj_final = torch.gather(
            Y_hat[..., -1:, :, :], 2,  # [B, A, 1, H, D]
            prob_max_idx.unsqueeze(2).unsqueeze(-1).expand(-1, -1, 1, 1, D)
        )[..., 0, :]  # [B, A, 1, D]
        best_traj_final = best_traj_final.squeeze(2)  # [B, A, D]
        
        # Ground truth final position
        Y_final = Y[..., -1, :]  # [B, A, D]
        
        # Compute final displacement error
        error = (best_traj_final - Y_final).norm(dim=-1)  # [B, A]
        
    else:
        # Use last valid time step based on mask
        mask_pred = mask[:, :, -T:]  # [B, A, T]
        
        # Get last valid index for each agent
        last_valid_idx = (mask_pred != 0).cumsum(dim=-1).argmax(dim=-1)  # [B, A]
        
        # Create indices for gathering
        b_idx, a_idx = torch.meshgrid(
            torch.arange(B, device=Y.device), 
            torch.arange(A, device=Y.device), 
            indexing='ij'
        )
        
        # Ground truth at last valid time step
        Y_final = Y[b_idx, a_idx, last_valid_idx]  # [B, A, D]
        
        # Get predictions for all modes at last valid time step
        Y_hat_final_all_modes = Y_hat[..., -T:, :, :][b_idx, a_idx, last_valid_idx, :, :]  # [B, A, H, D]
        
        # Select highest-probability mode
        best_traj_final = torch.gather(
            Y_hat_final_all_modes, 2,
            prob_max_idx.unsqueeze(-1).expand(-1, -1, 1, D)
        )[..., 0, :]  # [B, A, D]
        
        # Compute final displacement error
        error = (best_traj_final - Y_final).norm(dim=-1)  # [B, A]
    
    return scale * error  # [B, A]

def marginal_ade(
    Y_hat: torch.tensor, Y: torch.tensor, mask: torch.tensor = None, scale: int = 1000.0
) -> torch.tensor:
    """ Computes the marginal Average Displacement Error (mADE). It computes the mean error across 
    time steps, and selects the future with the smallest error for each agent independently, and 
    then computes the mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: marginal average displacement error.
    """

    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    if mask is None:
        error = error.mean(dim=2)                                  # B, A, T, H    -> B, A, H
    else:
        mask = mask[:, :, -T:]
        error = error.view(B * A, T, -1)
        BA, T, H = error.shape
        mask = mask.view(BA, T)
        error_masked = torch.zeros(BA, H).to(error.device)
        for ba in range(BA):
            amask = mask[ba]
            error_masked[ba] = error[ba, amask].mean(dim=0)
        error = error_masked.view(B, A, H)
    error = error.min(dim=-1)[0]                                   # B, A, H       -> B, A
    return scale * error#.mean()                                   # B, A          -> 1

def marginal_prob_ade(
    Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor, mask: torch.tensor = None
) -> torch.tensor:
    """ Computes the marginal probability-weighted Average Displacement Error (mADE). It computes the 
    mean error across time steps and weights it by the scores for the predicted trajectories. Then 
    selects the future with the smallest error for each agent independently, and then computes the 
    mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: marginal average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    if mask is None:
        error = error.mean(dim=2) * Y_hat_scores                   # B, A, T, H    -> B, A, H                               
    else:
        mask = mask[:, :, -T:]
        error = error.view(B * A, T, -1)
        BA, T, H = error.shape
        mask = mask.view(BA, T)
        error_masked = torch.zeros(BA, H).to(error.device)
        for ba in range(BA):
            amask = mask[ba]
            error_masked[ba] = error[ba, amask].mean(dim=0)
        error = error_masked.view(B, A, H) * Y_hat_scores
    error = error.sum(dim=-1)                                      # B, A, H       -> B, A
    return error#.mean()                                           # B, A          -> 1

def marginal_fde(
    Y_hat: torch.tensor, Y: torch.tensor, mask: torch.tensor = None, scale: int = 1000.0
) -> torch.tensor:
    """ Computes the marginal Final Displacement Error (mFDE). It computes the mean error for the 
    last time step, and selects the future with the smallest error for each independently, and then
    computes the mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: marginal final displacement error.
    """
    B, A, T, D = Y.size()
    if mask is None:
        error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    else: 
        mask, Y_hat = mask[:, :, -T:], Y_hat[:, :, -T:]
        # Get the last valid index
        t = (mask != 0).cumsum(-1).argmax(-1)
        x, y = torch.meshgrid(torch.arange(0, B), torch.arange(0, A), indexing='ij')
        Y_T = Y[x, y, t]          # B, A, D
        Y_hat_T = Y_hat[x, y, t]  # B, A, H, D
        error = (Y_hat_T - Y_T[..., None, :]).norm(dim=-1)

    error = error.min(dim=-1)[0]                                      # B, A, H       -> B, A   
    return scale * error#.mean()                                      # B, A          -> 1

def marginal_prob_fde(
    Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor, mask: torch.tensor = None
) -> torch.tensor:
    """ Computes the marginal probability-weighted Final Displacement Error (mFDE). It computes the 
    mean error for the last time step and weights it by the scores for the predicted trajectories.
    Then, it selects the future with the smallest error for each independently, and then computes the 
    mean across the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: marginal final displacement error.
    """
    B, A, T, D = Y.size()
    if mask is None:
        error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    else:
        # Get the last valid index
        mask, Y_hat = mask[:, :, -T:], Y_hat[:, :, -T:]
        t = (mask != 0).cumsum(-1).argmax(-1)
        x, y = torch.meshgrid(torch.arange(0, B), torch.arange(0, A), indexing='ij')
        Y_hat_T = Y_hat[x, y, t]  # B, A, H, D
        Y_T = Y[x, y, t]          # B, A, D
        error = (Y_hat_T - Y_T[..., None, :]).norm(dim=-1)
    error = error * Y_hat_scores
    error = error.sum(dim=-1)                                         # B, A, H       -> B, A
    return error#.mean()                                              # B, A          -> 1
    
def joint_ade(Y_hat: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Average Displacement Error (jADE). Take the average error over all agents 
    within a sample before selecting the best one to use in evaluation, then computes the mean across 
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    
    # error across time and agents 
    error = error.mean(dim=(2, 1))                                 # B, A, T, H    -> B, H
    error = error.min(dim=-1)[0]                                   # B, H          -> B
    return error#.mean()                                           # B             -> 1

def joint_prob_ade(Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Average Displacement Error (jADE). Take the average error over all agents 
    within a sample before selecting the best one to use in evaluation, then computes the mean across 
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y_hat_scores[torch.tensor(B, A, H)]: scores for the predicted trajectories
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1) # B, A, T, H, 2 -> B, A, T, H
    error = error * Y_hat_scores[..., None, :]
    # error across time and agents 
    error = error.mean(dim=(2, 1))                                 # B, A, T, H    -> B, H
    error = error.sum(dim=-1)                                      # B, H          -> B
    return error#.mean()  

def joint_fde(Y_hat: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Final Displacement Error (jFDE). Take the final error over all agents 
    within a sample before selecting the best one to use in evaluation, then computes the mean across 
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    
    # error across agents 
    error = error.mean(dim=1)                                         # B, A, H    -> B, H
    error = error.min(dim=-1)[0]                                      # B, H       -> B
    return error#.mean()                                              # B          -> 1

def joint_prob_fde(Y_hat: torch.tensor, Y_hat_scores: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """ Computes the joint Final Displacement Error (jFDE). Take the final error over all agents 
    within a sample before selecting the best one to use in evaluation, then computes the mean across 
    the batch.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
        Y[torch.tensor(B, A, T, D)]: ground truth trajectory.
    
    Output
    ------
        error[torch.tensor]: joint average displacement error.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1) # B, A, H, 2 -> B, A, H
    error = error * Y_hat_scores
    # error across agents 
    error = error.mean(dim=1)                                         # B, A, H    -> B, H
    error = error.sum(dim=-1)                                         # B, H       -> B
    return error#.mean()                                              # B          -> 1