import torch
import torch.nn.functional as F
import numpy as np
import math

import torch
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
from collections import defaultdict

class ModalClassificationMetrics:
    """
    Accumulates modal classification metrics across batches and computes final metrics.
    
    This class accumulates a confusion matrix across all batches and computes
    precision, recall, F1, and other classification metrics for modality prediction.
    
    It supports merging fine-grained modes into coarse-grained categories (e.g., 
    16 turn+speed modes -> 4 turn-only modes).
    """
    def __init__(self, num_modes: int, valid_modes: Optional[List[int]] = None, 
                 mode_names: Optional[Dict[int, str]] = None,
                 merge_mapping: Optional[Dict[int, int]] = None,
                 merged_mode_names: Optional[Dict[int, str]] = None):
        """
        Args:
            num_modes: Total number of prediction modes (H)
            valid_modes: List of valid mode indices (e.g., [0,1,2,4,5,6,8,9,10,15])
            mode_names: Dictionary mapping mode indices to human-readable names
            merge_mapping: Dictionary mapping original mode indices to merged mode indices
            merged_mode_names: Dictionary mapping merged mode indices to names
        """
        self.num_modes = num_modes
        self.valid_modes = valid_modes if valid_modes is not None else list(range(num_modes))
        self.mode_names = mode_names if mode_names is not None else {}
        
        # Merging configuration
        self.merge_mapping = merge_mapping
        self.merged_mode_names = merged_mode_names
        
        if merge_mapping is not None:
            # Get unique merged mode indices
            self.merged_modes = sorted(set(merge_mapping.values()))
            self.num_merged_modes = len(self.merged_modes)
            print(f"Merging {num_modes} modes into {self.num_merged_modes} merged modes")
            print(f"Merged modes: {self.merged_modes}")
        else:
            self.merged_modes = None
            self.num_merged_modes = num_modes
        
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        # Original confusion matrix (fine-grained)
        self.confusion_matrix = torch.zeros((self.num_modes, self.num_modes), dtype=torch.float32)
        
        # Merged confusion matrix (coarse-grained)
        if self.merge_mapping is not None:
            self.merged_confusion_matrix = torch.zeros(
                (self.num_merged_modes, self.num_merged_modes), dtype=torch.float32
            )
        else:
            self.merged_confusion_matrix = None
        
        self.total_samples = 0
        # Track per-mode sample counts
        self.per_mode_counts = torch.zeros(self.num_modes, dtype=torch.float32)
    
    def _merge_modes(self, mode_indices: torch.Tensor) -> torch.Tensor:
        """Merge fine-grained mode indices into coarse-grained categories."""
        if self.merge_mapping is None:
            return mode_indices
        
        # Create mapping tensor
        mapping = torch.zeros(self.num_modes, dtype=torch.long)
        for orig_idx, merged_idx in self.merge_mapping.items():
            mapping[orig_idx] = merged_idx
        
        # Map each index
        return mapping[mode_indices.long()]
    
    def update(
        self, 
        pred_scores: torch.Tensor, 
        true_modes: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ):
        """
        Update accumulated statistics with a new batch.
        
        Args:
            pred_scores: [B, A, H] predicted modality probabilities
            true_modes: [B, A] ground truth mode indices (already encoded)
            mask: [B, A] agent validity mask (optional)
        """
        # Get predicted modes (argmax of probabilities)
        pred_modes = pred_scores.argmax(dim=-1)  # [B, A]
        
        # Create valid agent mask
        if mask is None:
            valid_mask = torch.ones_like(pred_modes, dtype=torch.bool)
        else:
            valid_mask = mask.bool()
        
        # Filter valid agents
        pred_modes_valid = pred_modes[valid_mask].cpu()
        true_modes_valid = true_modes[valid_mask].cpu()
        
        # ===== Update fine-grained confusion matrix =====
        for t, p in zip(true_modes_valid, pred_modes_valid):
            self.confusion_matrix[t.long(), p.long()] += 1
            self.per_mode_counts[t.long()] += 1
        
        # ===== Update merged confusion matrix if needed =====
        if self.merge_mapping is not None:
            # Merge mode indices
            pred_modes_merged = self._merge_modes(pred_modes_valid)
            true_modes_merged = self._merge_modes(true_modes_valid)
            
            # Update merged confusion matrix
            for t, p in zip(true_modes_merged, pred_modes_merged):
                self.merged_confusion_matrix[t.long(), p.long()] += 1
        
        self.total_samples += len(true_modes_valid)
    
    def compute(self, use_merged: bool = True, normalize: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics from accumulated confusion matrix.
        
        Args:
            use_merged: If True and merge_mapping is provided, compute metrics on merged modes
            normalize: If True, normalize confusion matrix rows (recall)
        
        Returns:
            Dictionary containing all metrics
        """
        # Decide which confusion matrix to use
        if use_merged and self.merge_mapping is not None:
            cm = self.merged_confusion_matrix
            mode_list = self.merged_modes
            prefix = "merged_"
            mode_count = self.num_merged_modes
        else:
            cm = self.confusion_matrix
            mode_list = self.valid_modes
            prefix = ""
            mode_count = self.num_modes
        
        eps = 1e-10
        
        # Overall accuracy
        accuracy = torch.diag(cm).sum() / (cm.sum() + eps)
        
        # Per-mode metrics
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        support = cm.sum(dim=1)
        
        precision_per_mode = tp / (tp + fp + eps)
        recall_per_mode = tp / (tp + fn + eps)
        f1_per_mode = 2 * precision_per_mode * recall_per_mode / (precision_per_mode + recall_per_mode + eps)
        per_mode_accuracy = tp / (cm.sum(dim=1) + eps)
        
        # Filter for valid modes (for fine-grained) or all merged modes
        if use_merged and self.merge_mapping is not None:
            # For merged modes, all are considered valid
            valid_indices = torch.arange(mode_count)
            precision_valid = precision_per_mode
            recall_valid = recall_per_mode
            f1_valid = f1_per_mode
            support_valid = support
        else:
            valid_indices = torch.tensor(self.valid_modes, dtype=torch.long)
            precision_valid = precision_per_mode[valid_indices]
            recall_valid = recall_per_mode[valid_indices]
            f1_valid = f1_per_mode[valid_indices]
            support_valid = support[valid_indices]
        
        # Macro averages
        macro_precision = precision_valid.mean()
        macro_recall = recall_valid.mean()
        macro_f1 = f1_valid.mean()
        
        # Weighted averages
        total_valid_samples = support_valid.sum() + eps
        weighted_precision = (precision_valid * support_valid).sum() / total_valid_samples
        weighted_recall = (recall_valid * support_valid).sum() / total_valid_samples
        weighted_f1 = (f1_valid * support_valid).sum() / total_valid_samples
        
        # Normalized confusion matrix
        cm_normalized = cm / (cm.sum(dim=1, keepdim=True) + eps)
        
        results = {
            f'{prefix}accuracy': accuracy,
            f'{prefix}precision_per_mode': precision_per_mode,
            f'{prefix}recall_per_mode': recall_per_mode,
            f'{prefix}f1_per_mode': f1_per_mode,
            f'{prefix}support_per_mode': support,
            f'{prefix}per_mode_accuracy': per_mode_accuracy,
            f'{prefix}confusion_matrix': cm_normalized if normalize else cm,
            f'{prefix}confusion_matrix_raw': cm,
            f'{prefix}total_samples': self.total_samples,
            # Valid modes only
            f'{prefix}precision_valid': precision_valid,
            f'{prefix}recall_valid': recall_valid,
            f'{prefix}f1_valid': f1_valid,
            f'{prefix}support_valid': support_valid,
            f'{prefix}macro_precision': macro_precision,
            f'{prefix}macro_recall': macro_recall,
            f'{prefix}macro_f1': macro_f1,
            f'{prefix}weighted_precision': weighted_precision,
            f'{prefix}weighted_recall': weighted_recall,
            f'{prefix}weighted_f1': weighted_f1,
        }
        
        return results
    
    def print_report(self, use_merged: bool = True, title: str = "Modal Classification Report"):
        """Print a comprehensive classification report."""
        metrics = self.compute(use_merged=use_merged)
        prefix = "merged_" if (use_merged and self.merge_mapping is not None) else ""
        
        print("\n" + "="*100)
        print(f"{title:^100}")
        print("="*100)
        print(f"Total samples: {metrics[f'{prefix}total_samples']}")
        print(f"Overall Accuracy: {metrics[f'{prefix}accuracy']:.4f}")
        print("-"*100)
        
        # Print header
        print(f"{'Mode':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} "
              f"{'Accuracy':>10} {'Support':>10}")
        print("-"*100)
        
        # Determine which modes to print
        if use_merged and self.merge_mapping is not None:
            mode_list = self.merged_modes
            mode_names = self.merged_mode_names if self.merged_mode_names else {}
        else:
            mode_list = self.valid_modes
            mode_names = self.mode_names
        
        # Print each mode
        for mode_idx in mode_list:
            mode_name = mode_names.get(mode_idx, f"Mode {mode_idx}")
            
            # Get metrics for this mode
            if use_merged and self.merge_mapping is not None:
                # For merged modes, indices are 0,1,2,3
                precision = metrics[f'{prefix}precision_valid'][mode_idx].item()
                recall = metrics[f'{prefix}recall_valid'][mode_idx].item()
                f1 = metrics[f'{prefix}f1_valid'][mode_idx].item()
                acc = metrics[f'{prefix}precision_valid'][mode_idx].item()
                support = metrics[f'{prefix}support_valid'][mode_idx].item()
            else:
                # For fine-grained modes, need to map correctly
                pos = list(self.valid_modes).index(mode_idx) if mode_idx in self.valid_modes else -1
                if pos >= 0:
                    precision = metrics[f'{prefix}precision_valid'][pos].item()
                    recall = metrics[f'{prefix}recall_valid'][pos].item()
                    f1 = metrics[f'{prefix}f1_valid'][pos].item()
                    acc = metrics[f'{prefix}precision_valid'][pos].item()
                    support = metrics[f'{prefix}support_valid'][pos].item()
                else:
                    continue
            
            print(f"{mode_name:<25} {precision:10.4f} {recall:10.4f} {f1:10.4f} "
                  f"{acc:10.4f} {support:10.0f}")
        
        print("-"*100)
        print(f"{'Macro Avg':<25} {metrics[f'{prefix}macro_precision']:10.4f} "
              f"{metrics[f'{prefix}macro_recall']:10.4f} {metrics[f'{prefix}macro_f1']:10.4f}")
        print(f"{'Weighted Avg':<25} {metrics[f'{prefix}weighted_precision']:10.4f} "
              f"{metrics[f'{prefix}weighted_recall']:10.4f} {metrics[f'{prefix}weighted_f1']:10.4f}")
        print("="*100)
    
    def print_confusion_matrix(self, use_merged: bool = True, title: str = "Confusion Matrix", max_display: int = 10):
        """Print the confusion matrix."""
        metrics = self.compute(use_merged=use_merged)
        prefix = "merged_" if (use_merged and self.merge_mapping is not None) else ""
        
        cm = metrics[f'{prefix}confusion_matrix_raw']
        
        print(f"\n{title}:")
        
        # Determine which modes to display
        if use_merged and self.merge_mapping is not None:
            display_modes = self.merged_modes
            mode_names = self.merged_mode_names if self.merged_mode_names else {}
        else:
            display_modes = self.valid_modes[:min(max_display, len(self.valid_modes))]
            mode_names = self.mode_names
        
        n_display = len(display_modes)
        print("-" * (12 + 10 * n_display))
        print("True\\Pred", end=" ")
        for mode_idx in display_modes:
            short_name = mode_names.get(mode_idx, f"M{mode_idx}")[:8]
            print(f"{short_name:>8}", end=" ")
        print()
        
        # Print rows
        for i, true_idx in enumerate(display_modes):
            row_name = mode_names.get(true_idx, f"M{true_idx}")[:8]
            print(f"{row_name:8}", end=" ")
            for j, pred_idx in enumerate(display_modes):
                if use_merged and self.merge_mapping is not None:
                    # For merged matrix, indices are 0,1,2,3
                    val = cm[true_idx, pred_idx].item()
                else:
                    # For fine-grained matrix
                    val = cm[true_idx, pred_idx].item()
                print(f"{val:8.0f}", end=" ")
            print()
        print("-" * (12 + 10 * n_display))
        
        if not use_merged and len(self.valid_modes) > max_display:
            print(f"... and {len(self.valid_modes) - max_display} more modes")


def create_turn_only_mapping() -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Create mapping from 16 fine-grained modes to 4 turn-only modes.
    
    Turn modes:
        0: TurnLeft
        1: TurnRight  
        2: Straight
        3: Hold
    
    Original mode indices (turn*4 + speed):
        TurnLeft:   0-3   (TurnLeft_Accel, TurnLeft_Decel, TurnLeft_Normal, TurnLeft_Hold)
        TurnRight:  4-7   (TurnRight_Accel, TurnRight_Decel, TurnRight_Normal, TurnRight_Hold)
        Straight:   8-11  (Straight_Accel, Straight_Decel, Straight_Normal, Straight_Hold)
        Hold:       12-15 (Hold_Accel, Hold_Decel, Hold_Normal, Hold_Hold)
    
    Valid modes (from your code): [0,1,2,4,5,6,8,9,10,15]
    """
    mapping = {}
    
    # Map all 16 modes to turn categories
    for mode_idx in range(16):
        turn_idx = mode_idx // 4  # Integer division gives 0,1,2,3
        mapping[mode_idx] = turn_idx
    
    # Merged mode names
    merged_names = {
        0: "TurnLeft",
        1: "TurnRight",
        2: "Straight",
        3: "Hold"
    }
    
    return mapping, merged_names

def compute_mode_accuracy(
    mode_probs: torch.Tensor, 
    target_mode: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes mode prediction accuracy.

    Args:
        mode_probs: [B, A, M] - predicted mode probabilities
        target_mode: [B, A] - ground truth mode indices (from rule-based encoding)
        mask: [B, A] - agent mask (optional)

    Returns:
        scalar mean accuracy over batch
    """
    # Get predicted mode indices
    pred_mode = torch.argmax(mode_probs, dim=-1)  # [B, A]
    
    # Compute correct predictions
    correct = (pred_mode == target_mode).float()  # [B, A]
    
    # Apply mask if provided
    if mask is not None:
        # If mask is [B, A, T], reduce to [B, A] by checking any valid timestep
        if mask.dim() == 3:
            mask = mask.any(dim=-1).float()  # [B, A]
        correct = correct * mask
        denom = mask.sum().clamp_min(1)
        accuracy = correct.sum() / denom
    else:
        accuracy = correct.mean()
    
    return accuracy

def compute_mode_rmse(
    traj_mu: torch.Tensor, 
    mode_probs: torch.Tensor, 
    Y: torch.Tensor, 
    mask: torch.Tensor = None, 
    scale: float = 1000.0
) -> torch.Tensor:
    B, A, T_total, M, D = traj_mu.size()
    _, _, T_pred, _ = Y.size()

    traj_mu_future = traj_mu[:, :, -T_pred:, :, :]  # [B,A,T,M,D]
    mode_idx = torch.argmax(mode_probs, dim=-1)     # [B,A]

    best_mu = traj_mu_future[torch.arange(B)[:,None,None],
                             torch.arange(A)[None,:,None],
                             torch.arange(T_pred)[None,None,:],
                             mode_idx[:,:,None],
                             :]                     # [B,A,T,D]

    # RMSE
    error_sq = (best_mu - Y).norm(dim=-1).pow(2)
    
    if mask is not None:
        mask_pred = mask[:, :, -T_pred:]         # [B,A,T]
        error_sq = error_sq * mask_pred
        denom = mask_pred.sum(dim=-1).clamp_min(1)
        error_sq = error_sq.sum(dim=-1) / denom  # [B,A]
    else:
        error_sq = error_sq.mean(dim=-1)         # [B,A]
    
    rmse = torch.sqrt(error_sq)                
    return scale * rmse  # [B,A]

def compute_nll(
    traj_mu: torch.Tensor, 
    traj_sigma: torch.Tensor, 
    mode_probs: torch.Tensor, 
    Y: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes proper GMM Negative Log-Likelihood (NLL) as evaluation metric.

    Args:
        traj_mu:     [B, A, T_total, M, D]
        traj_sigma:  [B, A, T_total, M, D]
        mode_probs:  [B, A, M]
        Y:           [B, A, T_pred, D]
        mask:        [B, A, T_total] (optional)

    Returns:
        scalar mean NLL over batch
    """

    B, A, T_total, M, D = traj_mu.size()
    _, _, T_pred, _ = Y.size()

    # --- Extract future ---
    mu = traj_mu[:, :, -T_pred:, :, :]          # [B,A,T,M,D]
    sigma = traj_sigma[:, :, -T_pred:, :, :].clamp_min(1e-6)

    # --- Expand GT ---
    y = Y[:, :, :, None, :]                     # [B,A,T,1,D]

    # --- Gaussian log probability per mode ---
    logp = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(sigma)
        - 0.5 * ((y - mu) / sigma) ** 2
    )                                           # [B,A,T,M,D]

    # Sum over spatial dimension D
    logp = logp.sum(dim=-1)                     # [B,A,T,M]

    # --- Add log mixture weights ---
    log_mode_probs = torch.log(mode_probs.clamp_min(1e-9))  # [B,A,M]
    log_mode_probs = log_mode_probs[:, :, None, :]          # [B,A,1,M]

    logp = logp + log_mode_probs                # broadcast

    # --- Mixture over modes ---
    logp_mix = torch.logsumexp(logp, dim=-1)    # [B,A,T]

    nll = -logp_mix                             # [B,A,T]

    # --- Apply mask if provided ---
    if mask is not None:
        mask_pred = mask[:, :, -T_pred:]        # [B,A,T]
        nll = nll * mask_pred
        denom = mask_pred.sum(dim=-1).clamp_min(1)
        nll = nll.sum(dim=-1) / denom           # [B,A]
    else:
        nll = nll.mean(dim=-1)                  # [B,A]
    return nll


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
    error = error.min(dim=-1)[0]  
    return scale * error#.mean()                                   # B, A          -> 1
    
def mode_ade(
    traj_mu: torch.Tensor,
    mode_probs: torch.Tensor,
    Y: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1000.0
) -> torch.Tensor:
    """
    Computes ADE using the highest-probability trajectory (matches compute_mode_rmse logic).
    
    Args:
        traj_mu:    [B, A, T_total, M, D] predicted trajectory means
        mode_probs: [B, A, M] predicted modality probabilities
        Y:          [B, A, T_pred, D] ground truth future trajectories
        mask:       [B, A, T_total] agent validity mask (optional)
        scale:      scaling factor for final ADE
    
    Returns:
        ade: [B, A] ADE for each agent using its highest-probability trajectory
    """
    B, A, T_total, M, D = traj_mu.size()
    _, _, T_pred, _ = Y.size()
    
    # Extract future predictions
    traj_mu_future = traj_mu[:, :, -T_pred:, :, :]  # [B, A, T_pred, M, D]
    
    # Select highest-probability mode for each agent
    mode_idx = torch.argmax(mode_probs, dim=-1)  # [B, A]
    
    # Gather the best trajectory for each agent
    best_traj = traj_mu_future[torch.arange(B)[:, None, None],
                                torch.arange(A)[None, :, None],
                                torch.arange(T_pred)[None, None, :],
                                mode_idx[:, :, None],
                                :]  # [B, A, T_pred, D]
    
    # Compute displacement error per time step
    error = (best_traj - Y).norm(dim=-1)  # [B, A, T_pred]
    
    # Average over time with optional mask
    if mask is not None:
        mask_pred = mask[:, :, -T_pred:]  # [B, A, T_pred]
        error = error * mask_pred
        denom = mask_pred.sum(dim=-1).clamp_min(1)
        ade = error.sum(dim=-1) / denom  # [B, A]
    else:
        ade = error.mean(dim=-1)  # [B, A]
    
    return scale * ade  # [B, A]


def mode_fde(
    traj_mu: torch.Tensor,
    mode_probs: torch.Tensor,
    Y: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1000.0
) -> torch.Tensor:
    """
    Computes FDE using the highest-probability trajectory (matches compute_mode_rmse logic).
    
    Args:
        traj_mu:    [B, A, T_total, M, D] predicted trajectory means
        mode_probs: [B, A, M] predicted modality probabilities
        Y:          [B, A, T_pred, D] ground truth future trajectories
        mask:       [B, A, T_total] agent validity mask (optional)
        scale:      scaling factor for final FDE
    
    Returns:
        fde: [B, A] FDE for each agent using its highest-probability trajectory
    """
    B, A, T_total, M, D = traj_mu.size()
    _, _, T_pred, _ = Y.size()
    
    # Extract future predictions
    traj_mu_future = traj_mu[:, :, -T_pred:, :, :]  # [B, A, T_pred, M, D]
    
    # Select highest-probability mode for each agent
    mode_idx = torch.argmax(mode_probs, dim=-1)  # [B, A]
    
    if mask is None:
        # Use last time step
        best_traj_final = traj_mu_future[torch.arange(B)[:, None],
                                          torch.arange(A)[None, :],
                                          -1,  # last time step
                                          mode_idx,
                                          :]  # [B, A, D]
        
        # Ground truth final position
        Y_final = Y[..., -1, :]  # [B, A, D]
        
        # Compute final displacement error
        error = (best_traj_final - Y_final).norm(dim=-1)  # [B, A]
        
    else:
        # Use last valid time step based on mask
        mask_pred = mask[:, :, -T_pred:]  # [B, A, T_pred]
        
        # Get last valid index for each agent
        last_valid_idx = (mask_pred != 0).cumsum(dim=-1).argmax(dim=-1)  # [B, A]
        
        # Get ground truth at last valid time step
        Y_final = Y[torch.arange(B)[:, None],
                     torch.arange(A)[None, :],
                     last_valid_idx]  # [B, A, D]
        
        # Get predictions at last valid time step for each agent
        # Need to gather from traj_mu_future at the last_valid_idx time step
        best_traj_final = traj_mu_future[torch.arange(B)[:, None, None],
                                          torch.arange(A)[None, :, None],
                                          last_valid_idx[:, :, None],
                                          mode_idx[:, :, None],
                                          :]  # [B, A, 1, D]
        best_traj_final = best_traj_final.squeeze(2)  # [B, A, D]
        
        # Compute final displacement error
        error = (best_traj_final - Y_final).norm(dim=-1)  # [B, A]
    
    return scale * error  # [B, A]

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