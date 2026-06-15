"""
R-GCN graph preparation, ablations, and provider-level inference.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from hgad_cms.graph.builder import HeteroGraph
from hgad_cms.graph.constants import NODE_BENEFICIARY, NODE_PHYSICIAN, NODE_PROVIDER
from hgad_cms.graphsage.inference import (
    FoldGraphData,
    NodeScalers,
    build_inference_graph,
    fit_node_scalers,
    hetero_graph_to_hetero_data,
    prepare_fold_graph_data as _prepare_fold_graph_data,
)
from hgad_cms.rgcn.config import FeatureAblation, RelationAblation, active_edge_types
from hgad_cms.rgcn.model import HeteroRGCN

# Re-export shared types
__all__ = [
    "FoldGraphData",
    "NodeScalers",
    "prepare_fold_graph_data",
    "apply_relation_ablation",
    "apply_feature_ablation",
    "predict_provider_scores",
    "extract_provider_embeddings",
]


class ProviderClassifier(Protocol):
    """Protocol for provider-level GNN models."""

    def __call__(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]: ...

    @staticmethod
    def provider_scores(logits: Tensor) -> Tensor: ...


def apply_relation_ablation(
    data: HeteroData,
    relation_ablation: RelationAblation = "full",
) -> HeteroData:
    """Drop edge types not included in the ablation preset."""
    allowed = set(active_edge_types(relation_ablation))
    out = data.clone()
    for edge_type in list(out.edge_types):
        if edge_type not in allowed:
            empty = torch.empty((2, 0), dtype=torch.long)
            out[edge_type].edge_index = empty
    return out


def apply_feature_ablation(
    data: HeteroData,
    feature_ablation: FeatureAblation = "full",
) -> HeteroData:
    """Zero non-provider node features for provider-only ablation."""
    if feature_ablation == "full":
        return data
    out = data.clone()
    for node_type in (NODE_BENEFICIARY, NODE_PHYSICIAN):
        if node_type in out.node_types and hasattr(out[node_type], "x"):
            out[node_type].x = torch.zeros_like(out[node_type].x)
    return out


def prepare_fold_graph_data(
    train_graph: HeteroGraph,
    manifest: dict[str, Any],
    *,
    val_provider_ids: list[str],
    claims,
    beneficiaries,
    providers,
    schema_name: str = "v1.1",
    relation_ablation: RelationAblation = "full",
    feature_ablation: FeatureAblation = "full",
) -> FoldGraphData:
    """Build fold graphs with optional relation and feature ablations."""
    fold_data = _prepare_fold_graph_data(
        train_graph,
        manifest,
        val_provider_ids=val_provider_ids,
        claims=claims,
        beneficiaries=beneficiaries,
        providers=providers,
        schema_name=schema_name,
    )
    train_data = apply_feature_ablation(
        apply_relation_ablation(fold_data.train_data, relation_ablation),
        feature_ablation,
    )
    inference_data = apply_feature_ablation(
        apply_relation_ablation(fold_data.inference_data, relation_ablation),
        feature_ablation,
    )
    metadata = dict(fold_data.metadata)
    metadata["relation_ablation"] = relation_ablation
    metadata["feature_ablation"] = feature_ablation
    return FoldGraphData(
        train_data=train_data,
        inference_data=inference_data,
        train_provider_indices=fold_data.train_provider_indices,
        val_provider_indices=fold_data.val_provider_indices,
        val_provider_ids=fold_data.val_provider_ids,
        val_labels=fold_data.val_labels,
        scalers=fold_data.scalers,
        metadata=metadata,
    )


@torch.no_grad()
def predict_provider_scores(
    model: ProviderClassifier,
    data: HeteroData,
    provider_indices: Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Full-graph forward pass returning fraud probabilities for providers."""
    model.eval()
    data = data.to(device)
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    edge_index_dict = data.edge_index_dict
    logits = model(x_dict, edge_index_dict)[NODE_PROVIDER]
    scores = HeteroRGCN.provider_scores(logits).detach().cpu().numpy()
    idx = provider_indices.detach().cpu().numpy().astype(np.int64)
    return scores[idx]


@torch.no_grad()
def extract_provider_embeddings(
    model: HeteroRGCN,
    data: HeteroData,
    provider_indices: Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Return provider hidden embeddings before the classification head."""
    model.eval()
    data = data.to(device)
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    embeddings = model.encode(x_dict, data.edge_index_dict).detach().cpu().numpy()
    idx = provider_indices.detach().cpu().numpy().astype(np.int64)
    return embeddings[idx]
