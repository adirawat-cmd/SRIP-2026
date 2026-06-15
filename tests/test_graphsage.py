"""Tests for GraphSAGE model and configuration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData

from hgad_cms.graph.constants import NODE_BENEFICIARY, NODE_PHYSICIAN, NODE_PROVIDER
from hgad_cms.graphsage.config import GraphSAGEConfig, iter_search_grid
from hgad_cms.graphsage.model import HeteroGraphSAGE


def _toy_hetero_data() -> HeteroData:
    data = HeteroData()
    data[NODE_PROVIDER].x = torch.randn(4, 8)
    data[NODE_BENEFICIARY].x = torch.randn(6, 5)
    data[NODE_PHYSICIAN].x = torch.randn(3, 4)
    data[NODE_PROVIDER, "treats", NODE_BENEFICIARY].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data[NODE_BENEFICIARY, "treats_rev", NODE_PROVIDER].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data[NODE_PROVIDER, "bills_with", NODE_PHYSICIAN].edge_index = torch.tensor([[0, 1], [0, 1]])
    data[NODE_PHYSICIAN, "bills_with_rev", NODE_PROVIDER].edge_index = torch.tensor([[0, 1], [0, 1]])
    data[NODE_PROVIDER, "collaborates", NODE_PROVIDER].edge_index = torch.tensor([[0], [1]])
    data[NODE_PROVIDER, "collaborates_rev", NODE_PROVIDER].edge_index = torch.tensor([[1], [0]])
    return data


def test_graphsage_forward_mean_and_max():
    data = _toy_hetero_data()
    metadata = data.metadata()
    for aggregator in ("mean", "max"):
        model = HeteroGraphSAGE(metadata, hidden_dim=16, num_layers=2, aggregator=aggregator)
        logits = model(data.x_dict, data.edge_index_dict)[NODE_PROVIDER]
        assert logits.shape == (4,)
        probs = HeteroGraphSAGE.provider_scores(logits)
        assert torch.all((probs >= 0) & (probs <= 1))


def test_config_search_grid_sizes():
    assert len(iter_search_grid("single")) == 1
    assert len(iter_search_grid("quick")) == 6
    assert len(iter_search_grid("eval")) == 12
    assert len(iter_search_grid("full")) == 36


def test_config_id_stable():
    config = GraphSAGEConfig(hidden_dim=64, num_layers=2, dropout=0.3, aggregator="mean")
    assert "h64" in config.config_id
    assert "mean" in config.config_id
