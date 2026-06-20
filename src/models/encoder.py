"""
Heterogeneous GNN Encoder — 异构图神经网络编码器

Encodes a school building HeteroData graph into node embeddings using
HeteroConv with SAGEConv layers.

Architecture:
  - 3-layer HeteroConv (SAGEConv)
  - LayerNorm + ReLU after each layer
  - Skip connections (residual)
  - Output: [num_rooms, hidden_dim] node embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.nn.norm import LayerNorm

from utils.constants import ROOM_FEAT_DIM, ENV_FEAT_DIM


class HeteroGNNEncoder(nn.Module):
    """
    3-layer heterogeneous GNN with SAGE convolution.

    Handles the 5 edge types of the school graph:
      - ('room', 'physical_connects', 'room')
      - ('room', 'acoustic_blocks', 'room')
      - ('room', 'sight_lines', 'room')
      - ('room', 'sight_lines', 'environment')
      - ('room', 'physical_connects', 'environment')

    Args:
        hidden_dim: Hidden dimension for node embeddings (default 128).
        num_layers: Number of message-passing layers (default 3).
        dropout: Dropout probability after each layer (default 0.2).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Input projections
        self.room_proj = Linear(ROOM_FEAT_DIM, hidden_dim)
        self.env_proj = Linear(ENV_FEAT_DIM, hidden_dim)

        # HeteroConv layers with skip connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Edge types dict for HeteroConv
        self.edge_types = [
            ('room', 'physical_connects', 'room'),
            ('room', 'acoustic_blocks', 'room'),
            ('room', 'sight_lines', 'room'),
            ('room', 'sight_lines', 'environment'),
            ('room', 'physical_connects', 'environment'),
            # Add reverse edges for undirected message passing
            ('room', 'rev_physical_connects', 'room'),
            ('room', 'rev_acoustic_blocks', 'room'),
            ('room', 'rev_sight_lines', 'room'),
            ('environment', 'rev_sight_lines', 'room'),
            ('environment', 'rev_physical_connects', 'room'),
        ]

        for i in range(num_layers):
            conv_dict = {}
            for et in self.edge_types:
                conv_dict[et] = SAGEConv(
                    (-1, -1), hidden_dim,
                    aggr='mean',
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.norms.append(nn.ModuleDict({
                'room': LayerNorm(hidden_dim),
                'environment': LayerNorm(hidden_dim),
            }))

    def forward(
        self,
        data,
        masks: dict = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            data: PyG HeteroData with 'room' and 'environment' node types
                  and all edge types populated.
            masks: Optional dict of topology masks (unused in this version,
                   masks are handled implicitly by edge existence).

        Returns:
            Dict[str, Tensor] with 'room' and 'environment' node embeddings.
        """
        # Initial projection
        x_dict = {
            'room': self.room_proj(data['room'].x),
            'environment': self.env_proj(data['environment'].x),
        }

        # Build edge_index_dict from data, adding reverse edges
        # Apply masks if provided (for subgraph explainer)
        edge_index_dict = {}
        for et in self.edge_types:
            src, rel, dst = et
            if 'rev_' in rel:
                fwd_rel = rel.replace('rev_', '')
                fwd_et = (src, fwd_rel, dst)
                try:
                    ei = data[fwd_et].edge_index
                    if ei.numel() > 0:
                        ei_rev = torch.stack([ei[1], ei[0]])
                        # Apply mask for this edge type if provided
                        if masks and fwd_et in masks:
                            mask = masks[fwd_et]
                            ei_rev = ei_rev[:, mask]
                        if ei_rev.numel() > 0:
                            edge_index_dict[et] = ei_rev
                except (KeyError, AttributeError):
                    continue
            else:
                try:
                    ei = data[et].edge_index
                    # Apply mask for this edge type if provided
                    if masks and et in masks and ei.numel() > 0:
                        mask = masks[et]
                        ei = ei[:, mask]
                    if ei.numel() > 0:
                        edge_index_dict[et] = ei
                except (KeyError, AttributeError):
                    continue

        # Message passing layers
        for i in range(self.num_layers):
            x_res = {k: v.clone() for k, v in x_dict.items()}

            x_dict = self.convs[i](x_dict, edge_index_dict)

            # Norm + activation + dropout
            for node_type in x_dict:
                if node_type in self.norms[i]:
                    x_dict[node_type] = self.norms[i][node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(
                    x_dict[node_type], p=self.dropout_rate, training=self.training
                )

            # Residual connection
            for node_type in x_dict:
                if node_type in x_res:
                    x_dict[node_type] = x_dict[node_type] + x_res[node_type]

        return x_dict
