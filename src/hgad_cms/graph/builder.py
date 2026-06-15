"""
Heterogeneous graph construction for schema_v1.1 and ablation variants.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from hgad_cms.constants import (
    COL_BENE_ID,
    COL_FRAUD_LABEL,
    COL_PROVIDER,
    COL_REIMBURSED,
    COL_IS_INPATIENT,
    COL_CLAIM_DURATION_DAYS,
    DIAGNOSIS_COLUMNS,
)
from hgad_cms.exceptions import GraphBuildError
from hgad_cms.graph.constants import (
    GraphSchemaSpec,
    LABEL_MASKED,
    NODE_BENEFICIARY,
    NODE_DIAGNOSIS,
    NODE_PHYSICIAN,
    NODE_PROVIDER,
    PHYSICIAN_ROLE_COLUMNS,
    REL_BILLS_WITH,
    REL_COLLABORATES,
    REL_DIAGNOSED_WITH,
    REL_SEEN_BY,
    REL_TREATS,
)
from hgad_cms.graph.features import (
    build_beneficiary_features,
    build_bills_with_edge_features,
    build_collaborates_edge_features,
    build_diagnosed_with_edge_features,
    build_diagnosis_features,
    build_physician_features,
    build_provider_features,
    build_seen_by_edge_features,
    build_treats_edge_features,
    compute_top_code_lists,
    select_feature_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class HeteroGraph:
    """In-memory heterogeneous graph artifact."""

    schema_name: str
    node_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    edge_frames: dict[tuple[str, str, str], pd.DataFrame] = field(default_factory=dict)
    feature_columns: dict[str, list[str]] = field(default_factory=dict)
    fold_id: int | None = None
    is_reference: bool = False
    train_provider_ids: list[str] = field(default_factory=list)
    val_provider_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_count(self, node_type: str) -> int:
        return len(self.node_frames[node_type])

    def edge_count(self, src: str, rel: str, dst: str) -> int:
        return len(self.edge_frames.get((src, rel, dst), []))

    def relation_edge_count(self, relation: str) -> int:
        for (src, rel, dst), frame in self.edge_frames.items():
            if rel == relation:
                return len(frame)
        return 0


def _count_physician_refs(claims: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for role in PHYSICIAN_ROLE_COLUMNS:
        if role in claims.columns:
            for value in claims[role].dropna().astype(str):
                if value not in {"", "nan", "NA"}:
                    counts[value] += 1
    return counts


def _eligible_physicians(claims: pd.DataFrame, min_refs: int) -> set[str]:
    counts = _count_physician_refs(claims)
    return {phys for phys, count in counts.items() if count >= min_refs}


def build_treats_edges(claims: pd.DataFrame) -> pd.DataFrame:
    """Build provider-beneficiary edges aggregated from train claims."""
    grouped = claims.groupby([COL_PROVIDER, COL_BENE_ID], as_index=False).agg(
        n_claims=("ClaimID", "count"),
        total_reimb=(COL_REIMBURSED, "sum"),
        inpatient_claims=(COL_IS_INPATIENT, "sum"),
        mean_duration=(COL_CLAIM_DURATION_DAYS, "mean")
        if COL_CLAIM_DURATION_DAYS in claims.columns
        else (COL_REIMBURSED, "mean"),
    )
    grouped[COL_PROVIDER] = grouped[COL_PROVIDER].astype(str)
    grouped[COL_BENE_ID] = grouped[COL_BENE_ID].astype(str)
    return build_treats_edge_features(grouped)


def build_bills_with_edges(
    claims: pd.DataFrame,
    eligible_physicians: set[str],
) -> pd.DataFrame:
    """Build provider-physician edges with role counts."""
    role_map = {
        "AttendingPhysician": "n_attending",
        "OperatingPhysician": "n_operating",
        "OtherPhysician": "n_other",
    }
    frames: list[pd.DataFrame] = []
    for role_col, count_col in role_map.items():
        if role_col not in claims.columns:
            continue
        sub = claims.dropna(subset=[role_col])[
            [COL_PROVIDER, role_col, COL_REIMBURSED]
        ].copy()
        sub[COL_PROVIDER] = sub[COL_PROVIDER].astype(str)
        sub[role_col] = sub[role_col].astype(str)
        sub = sub[sub[role_col].isin(eligible_physicians)]
        if sub.empty:
            continue
        grouped = sub.groupby([COL_PROVIDER, role_col], as_index=False).agg(
            role_claims=(COL_REIMBURSED, "count"),
            role_reimb=(COL_REIMBURSED, "sum"),
        )
        grouped = grouped.rename(columns={role_col: "Physician", "role_claims": count_col})
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(
            columns=[
                COL_PROVIDER,
                "Physician",
                "n_claims",
                "total_reimb",
                "n_attending",
                "n_operating",
                "n_other",
            ]
        )

    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on=[COL_PROVIDER, "Physician"], how="outer")
    for col in ("n_attending", "n_operating", "n_other"):
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0)
    reimb_cols = [c for c in merged.columns if c.startswith("role_reimb")]
    merged["total_reimb"] = merged[reimb_cols].sum(axis=1) if reimb_cols else 0.0
    merged["n_claims"] = merged["n_attending"] + merged["n_operating"] + merged["n_other"]
    keep = [COL_PROVIDER, "Physician", "n_claims", "total_reimb", "n_attending", "n_operating", "n_other"]
    keep = [c for c in keep if c in merged.columns]
    return build_bills_with_edge_features(merged[keep].fillna(0))


def build_collaborates_edges(
    claims: pd.DataFrame,
    min_shared_beneficiaries: int,
) -> pd.DataFrame:
    """Build provider-provider edges from shared beneficiaries."""
    provider_panel = claims.groupby(COL_PROVIDER)[COL_BENE_ID].apply(
        lambda s: set(s.astype(str).unique())
    )
    bene_providers = claims.groupby(COL_BENE_ID)[COL_PROVIDER].apply(
        lambda s: sorted(set(s.astype(str).unique()))
    )

    pair_shared: dict[tuple[str, str], int] = defaultdict(int)
    for provs in bene_providers:
        if len(provs) < 2:
            continue
        for a, b in combinations(provs, 2):
            pair_shared[(a, b)] += 1

    rows: list[dict[str, Any]] = []
    for (prov_a, prov_b), shared in pair_shared.items():
        if shared < min_shared_beneficiaries:
            continue
        panel_a = provider_panel.get(prov_a, set())
        panel_b = provider_panel.get(prov_b, set())
        union_size = len(panel_a | panel_b)
        rows.append(
            {
                "provider_src": prov_a,
                "provider_dst": prov_b,
                "n_shared_beneficiaries": shared,
                "panel_union_size": union_size,
                "n_shared_claims": shared,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "provider_src",
                "provider_dst",
                "n_shared_beneficiaries",
                "panel_union_size",
                "n_shared_claims",
            ]
        )
    return build_collaborates_edge_features(pd.DataFrame(rows))


def build_seen_by_edges(
    claims: pd.DataFrame,
    eligible_physicians: set[str],
) -> pd.DataFrame:
    """Build beneficiary-physician edges (schema v1b)."""
    frames: list[pd.DataFrame] = []
    for role in PHYSICIAN_ROLE_COLUMNS:
        if role not in claims.columns:
            continue
        sub = claims.dropna(subset=[role])[[COL_BENE_ID, role]].copy()
        sub[COL_BENE_ID] = sub[COL_BENE_ID].astype(str)
        sub[role] = sub[role].astype(str)
        sub = sub[sub[role].isin(eligible_physicians)]
        if sub.empty:
            continue
        frames.append(sub.rename(columns={role: "Physician"}))

    if not frames:
        return pd.DataFrame(columns=[COL_BENE_ID, "Physician", "n_claims"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby([COL_BENE_ID, "Physician"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "n_claims"})
    return build_seen_by_edge_features(grouped)


def build_diagnosed_with_edges(
    claims: pd.DataFrame,
    top_diagnosis_codes: list[str],
) -> pd.DataFrame:
    """Build provider-diagnosis edges for schema v2."""
    if not top_diagnosis_codes:
        return pd.DataFrame(columns=[COL_PROVIDER, "DiagnosisCode", "n_claims"])

    melted = claims[[COL_PROVIDER, *DIAGNOSIS_COLUMNS]].melt(
        id_vars=COL_PROVIDER,
        value_name="DiagnosisCode",
    )
    melted[COL_PROVIDER] = melted[COL_PROVIDER].astype(str)
    melted = melted.dropna(subset=["DiagnosisCode"])
    melted["DiagnosisCode"] = melted["DiagnosisCode"].astype(str).str.strip()
    melted = melted[melted["DiagnosisCode"].isin(top_diagnosis_codes)]

    grouped = melted.groupby([COL_PROVIDER, "DiagnosisCode"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "n_claims"})
    return build_diagnosed_with_edge_features(grouped)


def _assign_node_indices(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = frame.copy()
    out = out.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    out["node_idx"] = np.arange(len(out), dtype=np.int64)
    return out


def _map_edge_endpoints(
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    src_map: dict[str, int],
    dst_map: dict[str, int],
) -> pd.DataFrame:
    out = edges.copy()
    out["src_idx"] = out[src_col].astype(str).map(src_map)
    out["dst_idx"] = out[dst_col].astype(str).map(dst_map)
    missing_src = int(out["src_idx"].isna().sum())
    missing_dst = int(out["dst_idx"].isna().sum())
    if missing_src or missing_dst:
        raise GraphBuildError(
            f"Edge mapping failed: missing_src={missing_src}, missing_dst={missing_dst}"
        )
    return out


def _edge_frame_from_mapped(mapped: pd.DataFrame) -> pd.DataFrame:
    """Return edge frame with integer endpoints and numeric features."""
    exclude = {
        COL_PROVIDER,
        COL_BENE_ID,
        "Physician",
        "provider_src",
        "provider_dst",
        "DiagnosisCode",
        "src_idx",
        "dst_idx",
    }
    feature_cols = [c for c in mapped.columns if c not in exclude]
    return mapped[["src_idx", "dst_idx", *feature_cols]].copy()


def build_hetero_graph(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    schema: GraphSchemaSpec,
    train_provider_ids: list[str] | None = None,
    val_provider_ids: list[str] | None = None,
    fold_id: int | None = None,
    is_reference: bool = False,
) -> HeteroGraph:
    """
    Build a fold-safe heterogeneous graph from processed tables.

    Parameters
    ----------
    claims:
        Processed claims table (full or already filtered).
    providers:
        Provider label table.
    beneficiaries:
        Beneficiary feature table.
    schema:
        Graph schema specification.
    train_provider_ids:
        Provider IDs used for training in this fold. If None, all providers
        in ``claims`` are treated as train/reference nodes.
    val_provider_ids:
        Held-out provider IDs (must not appear in graph nodes when fold-safe).
    fold_id:
        CV fold index, if applicable.
    is_reference:
        When True, builds full-corpus reference graph for G2 validation.

    Returns
    -------
    HeteroGraph
        Constructed graph artifact.
    """
    if train_provider_ids is None:
        train_ids = sorted(claims[COL_PROVIDER].astype(str).unique())
    else:
        train_ids = sorted({str(p) for p in train_provider_ids})

    val_ids = sorted({str(p) for p in (val_provider_ids or [])})
    if val_ids and not is_reference:
        leaked = set(train_ids).intersection(val_ids)
        if leaked:
            raise GraphBuildError(
                f"Train/val provider overlap in fold graph: {sorted(leaked)[:5]}"
            )

    train_claims = claims[claims[COL_PROVIDER].astype(str).isin(train_ids)].copy()
    if train_claims.empty:
        raise GraphBuildError("No claims remain after filtering to train providers")

    if not is_reference and val_ids:
        val_in_claims = train_claims[COL_PROVIDER].astype(str).isin(val_ids).any()
        if val_in_claims:
            raise GraphBuildError("Validation providers found in train-fold claims")

    eligible_physicians = _eligible_physicians(
        train_claims,
        min_refs=schema.physician_min_claim_refs,
    )
    top_dx, top_proc = compute_top_code_lists(train_claims)
    top_dx_nodes = (
        compute_top_codes_from_list(train_claims, schema.top_diagnosis_node_codes)
        if schema.include_diagnosis_nodes
        else []
    )

    treats = build_treats_edges(train_claims)
    bills = build_bills_with_edges(train_claims, eligible_physicians)
    collaborates = (
        build_collaborates_edges(train_claims, schema.pp_min_shared_beneficiaries)
        if schema.include_provider_provider
        else pd.DataFrame()
    )
    seen_by = (
        build_seen_by_edges(train_claims, eligible_physicians)
        if schema.include_beneficiary_physician
        else pd.DataFrame()
    )
    diagnosed = (
        build_diagnosed_with_edges(train_claims, top_dx_nodes)
        if schema.include_diagnosis_nodes
        else pd.DataFrame()
    )

    provider_ids = sorted(set(train_ids))
    beneficiary_ids = sorted(set(treats[COL_BENE_ID].astype(str))) if len(treats) else []
    physician_ids = sorted(set(bills["Physician"].astype(str))) if len(bills) else []
    if schema.include_beneficiary_physician and len(seen_by):
        physician_ids = sorted(set(physician_ids) | set(seen_by["Physician"].astype(str)))
        beneficiary_ids = sorted(set(beneficiary_ids) | set(seen_by[COL_BENE_ID].astype(str)))
    diagnosis_ids = sorted(set(diagnosed["DiagnosisCode"].astype(str))) if len(diagnosed) else []

    provider_feat = build_provider_features(
        train_claims,
        beneficiaries,
        provider_ids,
        top_dx,
        top_proc,
    )
    bene_feat = build_beneficiary_features(beneficiaries, beneficiary_ids)
    phys_feat = build_physician_features(train_claims, physician_ids) if physician_ids else pd.DataFrame(
        columns=["Physician"]
    )
    dx_feat = build_diagnosis_features(diagnosis_ids) if diagnosis_ids else pd.DataFrame(
        columns=["DiagnosisCode"]
    )

    provider_labels = providers.copy()
    provider_labels[COL_PROVIDER] = provider_labels[COL_PROVIDER].astype(str)
    label_map = provider_labels.set_index(COL_PROVIDER)[COL_FRAUD_LABEL].to_dict()
    provider_feat[COL_FRAUD_LABEL] = provider_feat[COL_PROVIDER].map(label_map).fillna(LABEL_MASKED).astype(
        np.int64
    )

    node_frames: dict[str, pd.DataFrame] = {}
    feature_columns: dict[str, list[str]] = {}

    provider_nodes = _assign_node_indices(provider_feat, COL_PROVIDER)
    node_frames[NODE_PROVIDER] = provider_nodes
    feature_columns[NODE_PROVIDER], _ = select_feature_matrix(provider_nodes, COL_PROVIDER)

    if beneficiary_ids:
        bene_nodes = _assign_node_indices(bene_feat, COL_BENE_ID)
        node_frames[NODE_BENEFICIARY] = bene_nodes
        feature_columns[NODE_BENEFICIARY], _ = select_feature_matrix(bene_nodes, COL_BENE_ID)

    if physician_ids:
        phys_nodes = _assign_node_indices(phys_feat, "Physician")
        node_frames[NODE_PHYSICIAN] = phys_nodes
        feature_columns[NODE_PHYSICIAN], _ = select_feature_matrix(phys_nodes, "Physician")

    if diagnosis_ids:
        dx_nodes = _assign_node_indices(dx_feat, "DiagnosisCode")
        node_frames[NODE_DIAGNOSIS] = dx_nodes
        feature_columns[NODE_DIAGNOSIS], _ = select_feature_matrix(dx_nodes, "DiagnosisCode")

    provider_map = provider_nodes.set_index(COL_PROVIDER)["node_idx"].to_dict()
    bene_map = (
        node_frames[NODE_BENEFICIARY].set_index(COL_BENE_ID)["node_idx"].to_dict()
        if NODE_BENEFICIARY in node_frames
        else {}
    )
    phys_map = (
        node_frames[NODE_PHYSICIAN].set_index("Physician")["node_idx"].to_dict()
        if NODE_PHYSICIAN in node_frames
        else {}
    )
    dx_map = (
        node_frames[NODE_DIAGNOSIS].set_index("DiagnosisCode")["node_idx"].to_dict()
        if NODE_DIAGNOSIS in node_frames
        else {}
    )

    edge_frames: dict[tuple[str, str, str], pd.DataFrame] = {}

    if len(treats):
        treats_mapped = _map_edge_endpoints(
            treats, COL_PROVIDER, COL_BENE_ID, provider_map, bene_map
        )
        edge_frames[(NODE_PROVIDER, REL_TREATS, NODE_BENEFICIARY)] = _edge_frame_from_mapped(
            treats_mapped
        )
        reverse = treats_mapped.copy()
        reverse["src_idx"], reverse["dst_idx"] = (
            treats_mapped["dst_idx"].values,
            treats_mapped["src_idx"].values,
        )
        edge_frames[(NODE_BENEFICIARY, f"{REL_TREATS}_rev", NODE_PROVIDER)] = (
            _edge_frame_from_mapped(reverse)
        )

    if len(bills):
        bills_mapped = _map_edge_endpoints(
            bills, COL_PROVIDER, "Physician", provider_map, phys_map
        )
        edge_frames[(NODE_PROVIDER, REL_BILLS_WITH, NODE_PHYSICIAN)] = _edge_frame_from_mapped(
            bills_mapped
        )
        reverse = bills_mapped.copy()
        reverse["src_idx"], reverse["dst_idx"] = (
            bills_mapped["dst_idx"].values,
            bills_mapped["src_idx"].values,
        )
        edge_frames[(NODE_PHYSICIAN, f"{REL_BILLS_WITH}_rev", NODE_PROVIDER)] = (
            _edge_frame_from_mapped(reverse)
        )

    if schema.include_provider_provider and len(collaborates):
        collab_mapped = _map_edge_endpoints(
            collaborates, "provider_src", "provider_dst", provider_map, provider_map
        )
        edge_frames[(NODE_PROVIDER, REL_COLLABORATES, NODE_PROVIDER)] = _edge_frame_from_mapped(
            collab_mapped
        )
        if schema.symmetrize_edges:
            reverse = collab_mapped.copy()
            reverse["src_idx"], reverse["dst_idx"] = (
                collab_mapped["dst_idx"].values,
                collab_mapped["src_idx"].values,
            )
            edge_frames[(NODE_PROVIDER, f"{REL_COLLABORATES}_rev", NODE_PROVIDER)] = (
                _edge_frame_from_mapped(reverse)
            )

    if schema.include_beneficiary_physician and len(seen_by):
        seen_mapped = _map_edge_endpoints(
            seen_by, COL_BENE_ID, "Physician", bene_map, phys_map
        )
        edge_frames[(NODE_BENEFICIARY, REL_SEEN_BY, NODE_PHYSICIAN)] = _edge_frame_from_mapped(
            seen_mapped
        )
        reverse = seen_mapped.copy()
        reverse["src_idx"], reverse["dst_idx"] = (
            seen_mapped["dst_idx"].values,
            seen_mapped["src_idx"].values,
        )
        edge_frames[(NODE_PHYSICIAN, f"{REL_SEEN_BY}_rev", NODE_BENEFICIARY)] = (
            _edge_frame_from_mapped(reverse)
        )

    if schema.include_diagnosis_nodes and len(diagnosed):
        dx_mapped = _map_edge_endpoints(
            diagnosed, COL_PROVIDER, "DiagnosisCode", provider_map, dx_map
        )
        edge_frames[(NODE_PROVIDER, REL_DIAGNOSED_WITH, NODE_DIAGNOSIS)] = _edge_frame_from_mapped(
            dx_mapped
        )
        reverse = dx_mapped.copy()
        reverse["src_idx"], reverse["dst_idx"] = (
            dx_mapped["dst_idx"].values,
            dx_mapped["src_idx"].values,
        )
        edge_frames[(NODE_DIAGNOSIS, f"{REL_DIAGNOSED_WITH}_rev", NODE_PROVIDER)] = (
            _edge_frame_from_mapped(reverse)
        )

    metadata = {
        "schema": schema.name,
        "fold_id": fold_id,
        "is_reference": is_reference,
        "train_provider_count": len(train_ids),
        "val_provider_count": len(val_ids),
        "eligible_physician_count": len(eligible_physicians),
        "top_diagnosis_codes": top_dx,
        "top_procedure_codes": top_proc,
        "top_diagnosis_node_codes": top_dx_nodes,
    }

    graph = HeteroGraph(
        schema_name=schema.name,
        node_frames=node_frames,
        edge_frames=edge_frames,
        feature_columns=feature_columns,
        fold_id=fold_id,
        is_reference=is_reference,
        train_provider_ids=train_ids,
        val_provider_ids=val_ids,
        metadata=metadata,
    )

    logger.info(
        "Built graph schema=%s fold=%s providers=%s edges=%s",
        schema.name,
        fold_id,
        graph.node_count(NODE_PROVIDER),
        sum(len(e) for e in edge_frames.values()),
    )
    return graph


def compute_top_codes_from_list(claims: pd.DataFrame, top_k: int) -> list[str]:
    """Top diagnosis codes for diagnosis node layer."""
    from hgad_cms.graph.features import compute_top_codes

    return compute_top_codes(claims, DIAGNOSIS_COLUMNS, top_k)
