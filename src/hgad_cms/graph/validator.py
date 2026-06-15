"""
Gate G2 validation for heterogeneous graph artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from hgad_cms.constants import COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.exceptions import GraphValidationError
from hgad_cms.graph.builder import HeteroGraph
from hgad_cms.graph.constants import (
    FORBIDDEN_FEATURE_COLUMNS,
    G2_COUNT_TOLERANCE_FRAC,
    G2_EXPECTED_EDGE_COUNTS_V1_1,
    G2_EXPECTED_NODE_COUNTS,
    G2_MIN_TRAIN_PROVIDER_EDGE_COVERAGE,
    G2_PHYSICIAN_COUNT_TOLERANCE,
    LABEL_MASKED,
    NODE_BENEFICIARY,
    NODE_PHYSICIAN,
    NODE_PROVIDER,
    REL_BILLS_WITH,
    REL_COLLABORATES,
    REL_TREATS,
    GraphSchemaSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphValidationCheck:
    """Single G2 validation outcome."""

    name: str
    passed: bool
    message: str
    expected: str | float | int | None = None
    actual: str | float | int | None = None


@dataclass
class GraphValidationReport:
    """Aggregate Gate G2 report."""

    passed: bool
    checks: list[GraphValidationCheck] = field(default_factory=list)
    schema_name: str = ""
    fold_id: int | None = None

    def raise_if_failed(self) -> None:
        if not self.passed:
            failed = [c for c in self.checks if not c.passed]
            messages = "; ".join(f"{c.name}: {c.message}" for c in failed)
            raise GraphValidationError(f"G2 validation failed: {messages}")


def _within_frac(actual: int, expected: int, frac: float) -> bool:
    if expected == 0:
        return actual == 0
    lower = int(expected * (1.0 - frac))
    upper = int(expected * (1.0 + frac)) + 1
    return lower <= actual <= upper


def _provider_has_edge(graph: HeteroGraph, provider_idx: int) -> bool:
    for (_, _, _), edges in graph.edge_frames.items():
        if edges.empty or "src_idx" not in edges.columns:
            continue
        src_match = (edges["src_idx"].to_numpy() == provider_idx).any()
        dst_match = (edges["dst_idx"].to_numpy() == provider_idx).any()
        if src_match or dst_match:
            return True
    return False


def validate_hetero_graph(
    graph: HeteroGraph,
    schema: GraphSchemaSpec,
    *,
    reference_mode: bool = False,
) -> GraphValidationReport:
    """
    Run Gate G2 checks on a constructed heterogeneous graph.

    Parameters
    ----------
    graph:
        Graph artifact to validate.
    schema:
        Schema specification used to build the graph.
    reference_mode:
        When True, apply full-corpus count thresholds for schema v1.1.
    """
    checks: list[GraphValidationCheck] = []

    if reference_mode and schema.name == "v1.1":
        for node_type, expected in G2_EXPECTED_NODE_COUNTS.items():
            actual = graph.node_count(node_type) if node_type in graph.node_frames else 0
            tol = G2_PHYSICIAN_COUNT_TOLERANCE if node_type == NODE_PHYSICIAN else 0
            passed = abs(actual - expected) <= tol if tol else actual == expected
            checks.append(
                GraphValidationCheck(
                    name=f"node_count_{node_type}",
                    passed=passed,
                    message=f"count={actual} expected={expected} tol={tol}",
                    expected=expected,
                    actual=actual,
                )
            )

        for relation, expected in G2_EXPECTED_EDGE_COUNTS_V1_1.items():
            if relation == REL_COLLABORATES and not schema.include_provider_provider:
                continue
            actual = graph.relation_edge_count(relation)
            passed = _within_frac(actual, expected, G2_COUNT_TOLERANCE_FRAC)
            checks.append(
                GraphValidationCheck(
                    name=f"edge_count_{relation}",
                    passed=passed,
                    message=f"count={actual} expected≈{expected}",
                    expected=expected,
                    actual=actual,
                )
            )

        if schema.include_provider_provider and schema.pp_min_shared_beneficiaries >= 2:
            collab_key = (NODE_PROVIDER, REL_COLLABORATES, NODE_PROVIDER)
            if collab_key in graph.edge_frames:
                min_shared = graph.edge_frames[collab_key].get(
                    "n_shared_beneficiaries", pd.Series(dtype=float)
                )
                single_edges = int((min_shared == 1).sum()) if len(min_shared) else 0
                checks.append(
                    GraphValidationCheck(
                        name="no_single_shared_pp_edges",
                        passed=single_edges == 0,
                        message=f"single_shared={single_edges}",
                        expected=0,
                        actual=single_edges,
                    )
                )

    provider_frame = graph.node_frames[NODE_PROVIDER]
    forbidden_in_features = FORBIDDEN_FEATURE_COLUMNS.intersection(
        set(graph.feature_columns.get(NODE_PROVIDER, []))
    )
    checks.append(
        GraphValidationCheck(
            name="no_leakage_feature_columns",
            passed=len(forbidden_in_features) == 0,
            message=f"forbidden={sorted(forbidden_in_features)}",
        )
    )

    if COL_FRAUD_LABEL in provider_frame.columns:
        masked = (provider_frame[COL_FRAUD_LABEL] == LABEL_MASKED).sum()
        checks.append(
            GraphValidationCheck(
                name="provider_labels_present",
                passed=True,
                message=f"masked_labels={int(masked)}",
                actual=int(masked),
            )
        )

    for edge_key, edges in graph.edge_frames.items():
        if edges.empty:
            continue
        dup = edges.duplicated(subset=["src_idx", "dst_idx"]).sum()
        checks.append(
            GraphValidationCheck(
                name=f"unique_edges_{edge_key[0]}_{edge_key[1]}_{edge_key[2]}",
                passed=int(dup) == 0,
                message=f"duplicates={int(dup)}",
                expected=0,
                actual=int(dup),
            )
        )
        if edge_key[1] == REL_COLLABORATES:
            self_loop = int((edges["src_idx"] == edges["dst_idx"]).sum())
            checks.append(
                GraphValidationCheck(
                    name="no_pp_self_loops",
                    passed=self_loop == 0,
                    message=f"self_loops={self_loop}",
                    expected=0,
                    actual=self_loop,
                )
            )

    train_ids = set(graph.train_provider_ids)
    if train_ids and not graph.is_reference:
        val_in_nodes = set(provider_frame[COL_PROVIDER].astype(str)).intersection(
            set(graph.val_provider_ids)
        )
        checks.append(
            GraphValidationCheck(
                name="val_providers_excluded",
                passed=len(val_in_nodes) == 0,
                message=f"val_in_graph={len(val_in_nodes)}",
                expected=0,
                actual=len(val_in_nodes),
            )
        )

    if NODE_PROVIDER in graph.node_frames and graph.node_frames[NODE_PROVIDER].shape[0]:
        missing_edges = 0
        for provider_id in graph.train_provider_ids:
            idx = provider_frame.loc[
                provider_frame[COL_PROVIDER].astype(str) == str(provider_id), "node_idx"
            ]
            if idx.empty:
                missing_edges += 1
                continue
            if not _provider_has_edge(graph, int(idx.iloc[0])):
                missing_edges += 1
        coverage = 1.0 - missing_edges / max(len(graph.train_provider_ids), 1)
        checks.append(
            GraphValidationCheck(
                name="train_provider_edge_coverage",
                passed=coverage >= G2_MIN_TRAIN_PROVIDER_EDGE_COVERAGE,
                message=f"coverage={coverage:.4f}",
                expected=G2_MIN_TRAIN_PROVIDER_EDGE_COVERAGE,
                actual=round(coverage, 6),
            )
        )

    required_nodes = {NODE_PROVIDER, NODE_BENEFICIARY, NODE_PHYSICIAN}
    for node_type in required_nodes:
        if schema.name.startswith("v1") or schema.name == "v1b" or schema.name == "v4":
            checks.append(
                GraphValidationCheck(
                    name=f"has_node_type_{node_type}",
                    passed=node_type in graph.node_frames and graph.node_count(node_type) > 0,
                    message=f"count={graph.node_count(node_type) if node_type in graph.node_frames else 0}",
                )
            )

    passed = all(c.passed for c in checks)
    report = GraphValidationReport(
        passed=passed,
        checks=checks,
        schema_name=schema.name,
        fold_id=graph.fold_id,
    )

    for check in checks:
        log_fn = logger.info if check.passed else logger.error
        log_fn("G2 check [%s] %s", check.name, check.message)

    return report
