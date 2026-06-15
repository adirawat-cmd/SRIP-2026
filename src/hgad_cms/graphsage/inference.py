"""
Graph data conversion and provider-level inference for GraphSAGE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.data import HeteroData

from hgad_cms.constants import COL_BENE_ID, COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.exceptions import GNNError
from hgad_cms.graph.builder import (
    HeteroGraph,
    _eligible_physicians,
    build_bills_with_edges,
    build_collaborates_edges,
    build_treats_edges,
)
from hgad_cms.graph.constants import (
    FORBIDDEN_FEATURE_COLUMNS,
    GraphSchemaSpec,
    LABEL_MASKED,
    NODE_BENEFICIARY,
    NODE_PHYSICIAN,
    NODE_PROVIDER,
    get_schema,
)
from hgad_cms.graph.features import (
    build_beneficiary_features,
    build_physician_features,
    build_provider_features,
    select_feature_matrix,
)
from hgad_cms.graphsage.model import HeteroGraphSAGE

logger = logging.getLogger(__name__)

NODE_ID_COLUMNS: dict[str, str] = {
    NODE_PROVIDER: COL_PROVIDER,
    NODE_BENEFICIARY: COL_BENE_ID,
    NODE_PHYSICIAN: "Physician",
}


@dataclass
class NodeScalers:
    """Per-node-type feature scalers fit on train-fold nodes only."""

    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    feature_columns: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class FoldGraphData:
    """Train and inference-ready heterogeneous graphs for one CV fold."""

    train_data: HeteroData
    inference_data: HeteroData
    train_provider_indices: Tensor
    val_provider_indices: Tensor
    val_provider_ids: list[str]
    val_labels: np.ndarray
    scalers: NodeScalers
    metadata: dict[str, Any] = field(default_factory=dict)


def _id_column(node_type: str) -> str:
    if node_type not in NODE_ID_COLUMNS:
        raise GNNError(f"Unsupported node type: {node_type}")
    return NODE_ID_COLUMNS[node_type]


def _extract_feature_matrix(
    frame: pd.DataFrame,
    node_type: str,
    feature_columns: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    id_col = _id_column(node_type)
    if feature_columns is None:
        cols, matrix = select_feature_matrix(frame, id_col)
    else:
        cols = [c for c in feature_columns if c in frame.columns]
        matrix = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    skip = {id_col, "node_idx", *FORBIDDEN_FEATURE_COLUMNS, COL_FRAUD_LABEL}
    cols = [c for c in cols if c not in skip]
    if not cols:
        raise GNNError(f"No feature columns for node type {node_type}")
    matrix = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    return cols, matrix


def fit_node_scalers(graph: HeteroGraph, manifest: dict[str, Any]) -> NodeScalers:
    """Fit StandardScaler per node type on train-fold graph nodes."""
    scalers: dict[str, StandardScaler] = {}
    feature_columns: dict[str, list[str]] = {}
    for node_type, frame in graph.node_frames.items():
        manifest_cols = manifest.get("feature_columns", {}).get(node_type)
        cols, matrix = _extract_feature_matrix(frame, node_type, manifest_cols)
        scaler = StandardScaler()
        scaler.fit(matrix)
        scalers[node_type] = scaler
        feature_columns[node_type] = cols
    return NodeScalers(scalers=scalers, feature_columns=feature_columns)


def _scale_features(matrix: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(matrix).astype(np.float32)


def _edge_tensor(frame: pd.DataFrame) -> Tensor:
    if frame.empty:
        return torch.empty((2, 0), dtype=torch.long)
    src = frame["src_idx"].astype(np.int64).to_numpy()
    dst = frame["dst_idx"].astype(np.int64).to_numpy()
    return torch.from_numpy(np.vstack([src, dst]))


def hetero_graph_to_hetero_data(
    graph: HeteroGraph,
    manifest: dict[str, Any],
    *,
    node_scalers: NodeScalers | None = None,
) -> HeteroData:
    """Convert a saved ``HeteroGraph`` artifact to PyG ``HeteroData``."""
    if node_scalers is None:
        node_scalers = fit_node_scalers(graph, manifest)

    data = HeteroData()
    id_maps: dict[str, dict[str, int]] = {}

    for node_type, frame in graph.node_frames.items():
        id_col = _id_column(node_type)
        cols, matrix = _extract_feature_matrix(frame, node_type, node_scalers.feature_columns.get(node_type))
        scaled = _scale_features(matrix, node_scalers.scalers[node_type])
        data[node_type].x = torch.from_numpy(scaled)
        data[node_type].num_nodes = len(frame)
        if "node_idx" in frame.columns:
            id_maps[node_type] = frame.set_index(id_col)["node_idx"].astype(int).to_dict()
        else:
            id_maps[node_type] = {str(row[id_col]): idx for idx, row in frame.iterrows()}

        if node_type == NODE_PROVIDER and COL_FRAUD_LABEL in frame.columns:
            labels = frame[COL_FRAUD_LABEL].fillna(LABEL_MASKED).astype(np.float32).to_numpy()
            data[node_type].y = torch.from_numpy(labels)

    for edge_key, frame in graph.edge_frames.items():
        data[edge_key].edge_index = _edge_tensor(frame)

    data["id_maps"] = id_maps  # type: ignore[index]
    data["feature_columns"] = node_scalers.feature_columns  # type: ignore[index]
    return data


def _assign_new_indices(
    ids: list[str],
    existing_map: dict[str, int],
    start_idx: int,
) -> tuple[dict[str, int], list[str]]:
    updated = dict(existing_map)
    new_ids: list[str] = []
    next_idx = start_idx
    for node_id in sorted(set(ids)):
        if node_id not in updated:
            updated[node_id] = next_idx
            new_ids.append(node_id)
            next_idx += 1
    return updated, new_ids


def _append_node_type(
    data: HeteroData,
    node_type: str,
    feature_frame: pd.DataFrame,
    node_scalers: NodeScalers,
) -> None:
    id_col = _id_column(node_type)
    cols = node_scalers.feature_columns[node_type]
    new_matrix = feature_frame.reindex(columns=cols, fill_value=0.0)
    new_matrix = new_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    scaled = _scale_features(new_matrix, node_scalers.scalers[node_type])
    new_x = torch.from_numpy(scaled)
    if hasattr(data[node_type], "x") and data[node_type].x is not None:
        data[node_type].x = torch.cat([data[node_type].x, new_x], dim=0)
    else:
        data[node_type].x = new_x
    data[node_type].num_nodes = int(data[node_type].x.size(0))


def _concat_edges(data: HeteroData, edge_key: tuple[str, str, str], new_edges: Tensor) -> None:
    if new_edges.numel() == 0:
        return
    if hasattr(data[edge_key], "edge_index") and data[edge_key].edge_index is not None:
        data[edge_key].edge_index = torch.cat([data[edge_key].edge_index, new_edges], dim=1)
    else:
        data[edge_key].edge_index = new_edges


def build_inference_graph(
    train_graph: HeteroGraph,
    train_data: HeteroData,
    *,
    val_provider_ids: list[str],
    claims: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    providers: pd.DataFrame,
    schema: GraphSchemaSpec,
    node_scalers: NodeScalers,
) -> tuple[HeteroData, Tensor, list[str]]:
    """
    Extend the train-fold graph with validation providers for inductive inference.

    Validation provider features and edges are built from val-fold claims only.
    New beneficiary / physician nodes are appended when required by val edges.
    """
    inference = train_data.clone()
    id_maps: dict[str, dict[str, int]] = {
        node_type: dict(mapping) for node_type, mapping in train_data["id_maps"].items()
    }

    train_ids = set(train_graph.train_provider_ids)
    val_ids = sorted({str(p) for p in val_provider_ids})
    if train_ids.intersection(val_ids):
        raise GNNError("Train and validation provider overlap in inference graph")

    metadata = train_graph.metadata
    top_dx = metadata.get("top_diagnosis_codes", [])
    top_proc = metadata.get("top_procedure_codes", [])

    val_claims = claims[claims[COL_PROVIDER].astype(str).isin(val_ids)].copy()
    if val_claims.empty:
        raise GNNError("No validation claims available for inference graph")

    train_claims = claims[claims[COL_PROVIDER].astype(str).isin(train_ids)].copy()
    eligible_physicians = _eligible_physicians(train_claims, schema.physician_min_claim_refs)

    val_provider_feat = build_provider_features(
        val_claims,
        beneficiaries,
        val_ids,
        top_dx,
        top_proc,
    )

    provider_map = id_maps[NODE_PROVIDER]
    next_provider_idx = int(inference[NODE_PROVIDER].num_nodes)
    provider_map, new_provider_ids = _assign_new_indices(val_ids, provider_map, next_provider_idx)
    if not new_provider_ids:
        raise GNNError("Validation providers already present in train graph")

    val_provider_frame = val_provider_feat[
        val_provider_feat[COL_PROVIDER].astype(str).isin(new_provider_ids)
    ].copy()
    _append_node_type(inference, NODE_PROVIDER, val_provider_frame, node_scalers)

    treats = build_treats_edges(val_claims)
    bills = build_bills_with_edges(val_claims, eligible_physicians)
    collaborates = (
        build_collaborates_edges(val_claims, schema.pp_min_shared_beneficiaries)
        if schema.include_provider_provider
        else pd.DataFrame()
    )

    bene_map = id_maps.get(NODE_BENEFICIARY, {})
    if not treats.empty:
        bene_ids = sorted(set(treats[COL_BENE_ID].astype(str)))
        next_bene_idx = int(inference[NODE_BENEFICIARY].num_nodes) if NODE_BENEFICIARY in inference.node_types else 0
        bene_map, new_bene_ids = _assign_new_indices(bene_ids, bene_map, next_bene_idx)
        if new_bene_ids:
            bene_feat = build_beneficiary_features(beneficiaries, new_bene_ids)
            _append_node_type(inference, NODE_BENEFICIARY, bene_feat, node_scalers)

    phys_map = id_maps.get(NODE_PHYSICIAN, {})
    if not bills.empty:
        phys_ids = sorted(set(bills["Physician"].astype(str)))
        next_phys_idx = int(inference[NODE_PHYSICIAN].num_nodes) if NODE_PHYSICIAN in inference.node_types else 0
        phys_map, new_phys_ids = _assign_new_indices(phys_ids, phys_map, next_phys_idx)
        if new_phys_ids:
            phys_feat = build_physician_features(val_claims, new_phys_ids)
            _append_node_type(inference, NODE_PHYSICIAN, phys_feat, node_scalers)

    id_maps[NODE_BENEFICIARY] = bene_map
    id_maps[NODE_PHYSICIAN] = phys_map
    id_maps[NODE_PROVIDER] = provider_map

    def _map_edges(
        frame: pd.DataFrame,
        src_col: str,
        dst_col: str,
        src_map: dict[str, int],
        dst_map: dict[str, int],
    ) -> Tensor:
        if frame.empty:
            return torch.empty((2, 0), dtype=torch.long)
        src = frame[src_col].astype(str).map(src_map)
        dst = frame[dst_col].astype(str).map(dst_map)
        valid = src.notna() & dst.notna()
        if not valid.any():
            return torch.empty((2, 0), dtype=torch.long)
        src_idx = src[valid].astype(np.int64).to_numpy()
        dst_idx = dst[valid].astype(np.int64).to_numpy()
        return torch.from_numpy(np.vstack([src_idx, dst_idx]))

    treats_edges = _map_edges(treats, COL_PROVIDER, COL_BENE_ID, provider_map, bene_map)
    _concat_edges(inference, (NODE_PROVIDER, "treats", NODE_BENEFICIARY), treats_edges)
    _concat_edges(
        inference,
        (NODE_BENEFICIARY, "treats_rev", NODE_PROVIDER),
        treats_edges.flip(0),
    )

    bills_edges = _map_edges(bills, COL_PROVIDER, "Physician", provider_map, phys_map)
    _concat_edges(inference, (NODE_PROVIDER, "bills_with", NODE_PHYSICIAN), bills_edges)
    _concat_edges(
        inference,
        (NODE_PHYSICIAN, "bills_with_rev", NODE_PROVIDER),
        bills_edges.flip(0),
    )

    if not collaborates.empty:
        collab_edges = _map_edges(
            collaborates,
            "provider_src",
            "provider_dst",
            provider_map,
            provider_map,
        )
        _concat_edges(inference, (NODE_PROVIDER, "collaborates", NODE_PROVIDER), collab_edges)
        _concat_edges(
            inference,
            (NODE_PROVIDER, "collaborates_rev", NODE_PROVIDER),
            collab_edges.flip(0),
        )

    val_provider_indices = torch.tensor(
        [provider_map[str(p)] for p in val_ids],
        dtype=torch.long,
    )
    inference["id_maps"] = id_maps  # type: ignore[index]
    return inference, val_provider_indices, val_ids


def prepare_fold_graph_data(
    train_graph: HeteroGraph,
    manifest: dict[str, Any],
    *,
    val_provider_ids: list[str],
    claims: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    providers: pd.DataFrame,
    schema_name: str = "v1.1",
) -> FoldGraphData:
    """Build train and inference hetero graphs for a CV fold."""
    schema = get_schema(schema_name)
    node_scalers = fit_node_scalers(train_graph, manifest)
    train_data = hetero_graph_to_hetero_data(train_graph, manifest, node_scalers=node_scalers)

    provider_frame = train_graph.node_frames[NODE_PROVIDER]
    train_provider_indices = torch.tensor(
        provider_frame["node_idx"].astype(np.int64).to_numpy(),
        dtype=torch.long,
    )

    inference_data, val_provider_indices, ordered_val_ids = build_inference_graph(
        train_graph,
        train_data,
        val_provider_ids=val_provider_ids,
        claims=claims,
        beneficiaries=beneficiaries,
        providers=providers,
        schema=schema,
        node_scalers=node_scalers,
    )

    provider_labels = providers.copy()
    provider_labels[COL_PROVIDER] = provider_labels[COL_PROVIDER].astype(str)
    label_map = provider_labels.set_index(COL_PROVIDER)[COL_FRAUD_LABEL]
    val_labels = label_map.loc[ordered_val_ids].astype(np.int64).to_numpy()

    return FoldGraphData(
        train_data=train_data,
        inference_data=inference_data,
        train_provider_indices=train_provider_indices,
        val_provider_indices=val_provider_indices,
        val_provider_ids=ordered_val_ids,
        val_labels=val_labels,
        scalers=node_scalers,
        metadata={"schema_name": schema_name},
    )


@torch.no_grad()
def predict_provider_scores(
    model: HeteroGraphSAGE,
    data: HeteroData,
    provider_indices: Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Run full-graph forward pass and return fraud probabilities for providers."""
    model.eval()
    data = data.to(device)
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    edge_index_dict = data.edge_index_dict
    logits = model(x_dict, edge_index_dict)[NODE_PROVIDER]
    scores = HeteroGraphSAGE.provider_scores(logits).detach().cpu().numpy()
    idx = provider_indices.detach().cpu().numpy().astype(np.int64)
    return scores[idx]


@torch.no_grad()
def extract_provider_embeddings(
    model: HeteroGraphSAGE,
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
