"""
Pairwise Ranking Loss — 成对排序损失

Encourages the GNN to correctly rank design quality comparisons,
not just predict absolute scores. This aligns with real design workflow
where we compare alternatives rather than score in isolation.

Formula:
    L_rank = mean(ReLU(margin - (pred_i - pred_j)))
    for all pairs (i,j) where target_i > target_j

Reference: Burges et al., "Learning to Rank using Gradient Descent" (ICML 2005)
"""

import torch
import torch.nn as nn
import itertools


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.05,
    max_pairs: int = 500,
) -> torch.Tensor:
    """
    Pairwise ranking loss for a batch of predictions.

    Args:
        predictions: [N] tensor of predicted scores.
        targets: [N] tensor of ground-truth scores.
        margin: Minimum desired gap between higher and lower scores.
        max_pairs: Maximum number of pairs to consider (limits O(N²) cost).

    Returns:
        Scalar ranking loss.
    """
    n = predictions.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=predictions.device)

    # Find all pairs where target_i > target_j
    target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # [N, N]
    pairs_mask = target_diff > 1e-6  # strictly greater

    if not pairs_mask.any():
        return torch.tensor(0.0, device=predictions.device)

    # Get pair indices
    idx_i, idx_j = pairs_mask.nonzero(as_tuple=True)

    # Limit pairs
    if idx_i.shape[0] > max_pairs:
        perm = torch.randperm(idx_i.shape[0], device=predictions.device)[:max_pairs]
        idx_i = idx_i[perm]
        idx_j = idx_j[perm]

    pred_diff = predictions[idx_i] - predictions[idx_j]
    loss = torch.relu(margin - pred_diff).mean()

    return loss


def multitask_ranking_loss(
    predictions: dict,
    targets: dict,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Pairwise ranking loss across all tasks.

    Args:
        predictions: Dict[str, Tensor] from MultiTaskScorer.
        targets: Dict[str, Tensor] of ground-truth labels.

    Returns:
        Scalar total ranking loss.
    """
    losses = []
    for task_name in predictions:
        if task_name in targets:
            losses.append(
                pairwise_ranking_loss(predictions[task_name], targets[task_name], margin)
            )
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()
