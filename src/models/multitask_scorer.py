"""
Multi-Task Heterogeneous GNN Scorer — 多任务异构图神经网络评分器

Predicts 7 quality dimensions simultaneously from a shared graph embedding.
Each head focuses on one architectural quality aspect, enabling per-dimension
subgraph explanation (Phase C1).

Architecture:
  HeteroGNNEncoder (shared, 3-layer SAGEConv, 128-dim)
      ↓
  GlobalMeanPool (room nodes only) → [batch, 128]
      ↓
  ┌─────────────────┬──────────────────┬─────────────────┐
  │  daylight_head   │  circulation_head │  fire_safety    │  ...
  │  [128→64→1]      │  [128→64→1]       │  [128→64→1]    │
  └─────────────────┴──────────────────┴─────────────────┘
      ↓                  ↓                   ↓
  daylight_score    circ_score         fire_score        ...

Usage:
    model = MultiTaskScorer(hidden_dim=128, num_layers=3)
    scores = model(hetero_data)
    # scores = {'daylight_quality': 0.68, 'circulation_efficiency': 0.04, ...}
"""

import torch
import torch.nn as nn

from models.encoder import HeteroGNNEncoder


# Task head definitions: name → (weight in overall score, description)
# Note: daylight_quality excluded — daylight metric still under development
TASK_HEADS = [
    ('circulation_efficiency', 1.0, 'Circulation efficiency'),
    ('fire_safety_margin',     1.0, 'Fire safety margin'),
    ('graph_robustness',       1.0, 'Graph robustness (lambda_2)'),
    ('path_redundancy',        1.0, 'Path redundancy (mesh vs tree)'),
    ('zone_cohesion',          1.0, 'Zone cohesion (intra-zone edges)'),
    ('overall_quality',        1.0, 'Overall quality (weighted average)'),
]


class TaskHead(nn.Module):
    """Single-task prediction head: Linear → ReLU → Dropout → Linear → Sigmoid."""

    def __init__(self, hidden_dim: int = 128, head_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiTaskScorer(nn.Module):
    """
    Multi-task GNN quality scorer with 7 prediction heads.

    Args:
        hidden_dim: Hidden dimension for GNN encoder (default 128).
        num_layers: Number of message-passing layers (default 3).
        dropout: Dropout probability (default 0.2).
        tasks: List of (task_name, weight, description) tuples.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        tasks: list = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared encoder
        self.encoder = HeteroGNNEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Task definitions
        self.task_defs = tasks or TASK_HEADS

        # Per-task prediction heads
        self.heads = nn.ModuleDict()
        for task_name, _, _ in self.task_defs:
            self.heads[task_name] = TaskHead(hidden_dim, 64, dropout)

    def forward(self, data) -> dict:
        """
        Predict all quality dimensions for a school building graph.

        Args:
            data: PyG HeteroData object.

        Returns:
            Dict[str, Tensor] mapping task_name → scalar prediction.
        """
        node_embs = self.encoder(data)
        room_embs = node_embs['room']
        graph_emb = room_embs.mean(dim=0)  # [hidden_dim]

        scores = {}
        for task_name, _, _ in self.task_defs:
            scores[task_name] = self.heads[task_name](graph_emb)

        return scores

    @torch.no_grad()
    def predict_batch(self, data_list: list) -> list:
        """Score a list of HeteroData graphs. Returns list of dicts."""
        self.eval()
        results = []
        for data in data_list:
            scores = self.forward(data)
            results.append({k: v.item() for k, v in scores.items()})
        return results

    def get_task_names(self) -> list:
        """Return list of task names in order."""
        return [t[0] for t in self.task_defs]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        weights: dict = None,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss across all tasks.

        Args:
            predictions: Dict[str, Tensor] from forward().
            targets: Dict[str, Tensor] of ground-truth labels.
            weights: Optional per-task weight dict.

        Returns:
            Scalar total loss.
        """
        loss_parts = []
        for task_name, _, default_w in self.task_defs:
            if task_name in predictions and task_name in targets:
                mse = nn.functional.mse_loss(predictions[task_name], targets[task_name])
                loss_parts.append(mse)  # equal weight: all tasks weight=1.0
        if not loss_parts:
            return torch.tensor(0.0)
        return torch.stack(loss_parts).sum()
