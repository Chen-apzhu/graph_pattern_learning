"""
School Graph Scorer — 学校图质量评分器

Wraps the HeteroGNN encoder with a graph-level pooling and MLP head
to predict a single quality score ∈ [0, 1] for a school building graph.

Usage:
    model = SchoolGraphScorer(hidden_dim=128, num_layers=3)
    score = model(hetero_data)  # shape: [1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import HeteroGNNEncoder


class SchoolGraphScorer(nn.Module):
    """
    End-to-end graph quality scoring model.

    Architecture:
      HeteroGNNEncoder → GlobalMeanPool → MLP Head → Sigmoid → score

    Args:
        hidden_dim: Hidden dimension for GNN encoder.
        num_layers: Number of message-passing layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = HeteroGNNEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # MLP scoring head: [hidden_dim] → [64] → [1]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Score a school building graph.

        Args:
            data: PyG HeteroData with room and environment nodes.

        Returns:
            Scalar quality score tensor ∈ [0, 1], shape [] (0-dim) or [1].
            Higher → better quality (more constraints passed).
        """
        # Encode
        node_embs = self.encoder(data)

        # Global mean pooling over room nodes only
        room_embs = node_embs['room']  # [N, hidden_dim]
        graph_emb = room_embs.mean(dim=0)  # [hidden_dim]

        # Score
        score = self.head(graph_emb).squeeze(-1)  # scalar

        return score

    def predict_batch(
        self,
        data_list: list,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Score a list of HeteroData graphs.

        Args:
            data_list: List of HeteroData objects.
            device: Device to run on.

        Returns:
            Tensor of scores [batch_size].
        """
        self.eval()
        scores = []
        with torch.no_grad():
            for data in data_list:
                if device:
                    data = data.to(device)
                score = self.forward(data)
                scores.append(score.cpu())
        return torch.stack(scores)
