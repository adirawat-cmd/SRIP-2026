"""Graph persistence utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hgad_cms.graph.builder import HeteroGraph
from hgad_cms.graph.constants import GRAPH_MANIFEST_NAME

logger = logging.getLogger(__name__)


def edge_key_to_filename(src: str, rel: str, dst: str) -> str:
    safe_rel = rel.replace("/", "_")
    return f"edges__{src}__{safe_rel}__{dst}.parquet"


def save_hetero_graph(graph: HeteroGraph, output_dir: Path) -> dict[str, Any]:
    """
    Persist heterogeneous graph nodes, edges, and manifest to disk.

    Returns
    -------
    dict
        Manifest dictionary written to ``graph_manifest.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    node_paths: dict[str, str] = {}
    for node_type, frame in graph.node_frames.items():
        path = output_dir / f"nodes_{node_type}.parquet"
        frame.to_parquet(path, index=False)
        node_paths[node_type] = str(path.resolve())

    edge_paths: dict[str, str] = {}
    edge_counts: dict[str, int] = {}
    for (src, rel, dst), frame in graph.edge_frames.items():
        filename = edge_key_to_filename(src, rel, dst)
        path = output_dir / filename
        frame.to_parquet(path, index=False)
        key = f"{src}|{rel}|{dst}"
        edge_paths[key] = str(path.resolve())
        edge_counts[key] = len(frame)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_name": graph.schema_name,
        "fold_id": graph.fold_id,
        "is_reference": graph.is_reference,
        "train_provider_ids": graph.train_provider_ids,
        "val_provider_ids": graph.val_provider_ids,
        "node_counts": {k: len(v) for k, v in graph.node_frames.items()},
        "edge_counts": edge_counts,
        "feature_columns": graph.feature_columns,
        "node_paths": node_paths,
        "edge_paths": edge_paths,
        "metadata": graph.metadata,
        "gate_g2_passed": None,
    }

    manifest_path = output_dir / GRAPH_MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Saved graph manifest: %s", manifest_path)
    return manifest


def load_hetero_graph(input_dir: Path) -> tuple[HeteroGraph, dict[str, Any]]:
    """Load heterogeneous graph from a saved artifact directory."""
    manifest_path = input_dir / GRAPH_MANIFEST_NAME
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    node_frames = {
        node_type: pd.read_parquet(path)
        for node_type, path in manifest["node_paths"].items()
    }
    edge_frames = {}
    for key, path in manifest["edge_paths"].items():
        src, rel, dst = key.split("|", 2)
        edge_frames[(src, rel, dst)] = pd.read_parquet(path)

    graph = HeteroGraph(
        schema_name=manifest["schema_name"],
        node_frames=node_frames,
        edge_frames=edge_frames,
        feature_columns=manifest.get("feature_columns", {}),
        fold_id=manifest.get("fold_id"),
        is_reference=manifest.get("is_reference", False),
        train_provider_ids=manifest.get("train_provider_ids", []),
        val_provider_ids=manifest.get("val_provider_ids", []),
        metadata=manifest.get("metadata", {}),
    )
    return graph, manifest


def update_manifest_g2_status(output_dir: Path, passed: bool) -> None:
    """Update graph manifest with G2 pass/fail flag."""
    manifest_path = output_dir / GRAPH_MANIFEST_NAME
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["gate_g2_passed"] = passed
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
