"""
Heterogeneous GraphSAGE for provider-level fraud classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, SAGEConv

from hgad_cms.graph.constants import NODE_PROVIDER


class HeteroGraphSAGE(nn.Module):
    """
    Multi-layer heterogeneous GraphSAGE with a provider-level binary head.

    Uses ``SAGEConv`` per edge type inside ``HeteroConv``; supports mean and
    max neighbor aggregation variants.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int,
        num_layers: int,
        *,
        dropout: float = 0.3,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if aggregator not in ("mean", "max"):
            raise ValueError("aggregator must be 'mean' or 'max'")

        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_dim,
                    aggr=aggregator,
                    normalize=True,
                )
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.provider_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        """Return logits per node type (provider head applied only to providers)."""
        for layer_idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if layer_idx < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        if NODE_PROVIDER not in x_dict:
            raise KeyError(f"Missing node type '{NODE_PROVIDER}' in forward pass")

        provider_logits = self.provider_head(x_dict[NODE_PROVIDER]).view(-1)
        return {NODE_PROVIDER: provider_logits}

    def encode(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> Tensor:
        """Return provider hidden representations before the classification head."""
        for layer_idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if layer_idx < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        if NODE_PROVIDER not in x_dict:
            raise KeyError(f"Missing node type '{NODE_PROVIDER}' in encode pass")
        return x_dict[NODE_PROVIDER]

    @staticmethod
    def provider_scores(logits: Tensor) -> Tensor:
        """Convert provider logits to fraud probabilities."""
        return torch.sigmoid(logits)
