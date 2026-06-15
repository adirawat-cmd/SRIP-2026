"""Tests for R-GCN model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData

from hgad_cms.graph.constants import NODE_BENEFICIARY, NODE_PHYSICIAN, NODE_PROVIDER
from hgad_cms.rgcn.config import iter_search_grid
from hgad_cms.rgcn.model import HeteroRGCN


def _toy_hetero_data() -> HeteroData:
    data = HeteroData()
    data[NODE_PROVIDER].x = torch.randn(4, 8)
    data[NODE_BENEFICIARY].x = torch.randn(6, 5)
    data[NODE_PHYSICIAN].x = torch.randn(3, 4)
    data[NODE_PROVIDER, "treats", NODE_BENEFICIARY].edge_index = torch.tensor([[0, 1], [0, 1]])
    data[NODE_BENEFICIARY, "treats_rev", NODE_PROVIDER].edge_index = torch.tensor([[0, 1], [0, 1]])
    data[NODE_PROVIDER, "bills_with", NODE_PHYSICIAN].edge_index = torch.tensor([[0], [0]])
    data[NODE_PHYSICIAN, "bills_with_rev", NODE_PROVIDER].edge_index = torch.tensor([[0], [0]])
    data[NODE_PROVIDER, "collaborates", NODE_PROVIDER].edge_index = torch.tensor([[0], [1]])
    data[NODE_PROVIDER, "collaborates_rev", NODE_PROVIDER].edge_index = torch.tensor([[1], [0]])
    return data


def test_rgcn_forward():
    data = _toy_hetero_data()
    in_channels = {nt: data[nt].x.size(-1) for nt in data.node_types}
    model = HeteroRGCN(
        data.metadata(),
        hidden_dim=16,
        num_layers=2,
        in_channels=in_channels,
    )
    logits = model(data.x_dict, data.edge_index_dict)[NODE_PROVIDER]
    assert logits.shape == (4,)


def test_rgcn_search_grid():
    assert len(iter_search_grid("eval")) == 6
    assert len(iter_search_grid("full")) == 18
