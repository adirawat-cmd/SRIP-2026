"""Tests for hgad_cms.graph.builder."""

import pandas as pd

from hgad_cms.constants import COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.data.merger import merge_claims, merge_provider_labels
from hgad_cms.graph.builder import build_hetero_graph, build_collaborates_edges
from hgad_cms.graph.constants import SCHEMA_V1_1, SCHEMA_V1_1_NO_PP, SCHEMA_V1B, get_schema


def _processed_mini_tables(sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers):
    claims = merge_claims(sample_inpatient, sample_outpatient)
    claims = merge_provider_labels(claims, sample_providers)
    providers = sample_providers.copy()
    providers[COL_FRAUD_LABEL] = providers["PotentialFraud"].map({"No": 0, "Yes": 1})
    return claims, providers, sample_beneficiaries


def test_build_v1_1_mini_graph(sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers):
    claims, providers, beneficiaries = _processed_mini_tables(
        sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
    )
    train_ids = sorted(providers[COL_PROVIDER].astype(str).tolist())
    graph = build_hetero_graph(
        claims=claims,
        providers=providers,
        beneficiaries=beneficiaries,
        schema=SCHEMA_V1_1,
        train_provider_ids=train_ids,
        val_provider_ids=[],
        fold_id=0,
    )
    assert graph.node_count("provider") == 3
    assert graph.relation_edge_count("treats") == 3
    assert graph.relation_edge_count("bills_with") > 0


def test_build_no_pp_schema_has_no_collaborates(
    sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
):
    claims, providers, beneficiaries = _processed_mini_tables(
        sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
    )
    train_ids = sorted(providers[COL_PROVIDER].astype(str).tolist())
    graph = build_hetero_graph(
        claims=claims,
        providers=providers,
        beneficiaries=beneficiaries,
        schema=SCHEMA_V1_1_NO_PP,
        train_provider_ids=train_ids,
    )
    assert graph.relation_edge_count("collaborates") == 0


def test_collaborates_respects_threshold():
    claims = pd.DataFrame(
        {
            "Provider": ["P1", "P1", "P2", "P3", "P3"],
            "BeneID": ["B1", "B2", "B1", "B1", "B2"],
            "ClaimID": ["C1", "C2", "C3", "C4", "C5"],
            "InscClaimAmtReimbursed": [10, 10, 10, 10, 10],
            "is_inpatient": [False] * 5,
        }
    )
    edges_t1 = build_collaborates_edges(claims, min_shared_beneficiaries=1)
    edges_t2 = build_collaborates_edges(claims, min_shared_beneficiaries=2)
    assert len(edges_t1) > len(edges_t2)
    assert len(edges_t2) == 1  # P1-P3 share B1 and B2


def test_fold_graph_excludes_val_providers(
    sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
):
    claims, providers, beneficiaries = _processed_mini_tables(
        sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
    )
    train_ids = ["PRV1", "PRV2"]
    val_ids = ["PRV3"]
    graph = build_hetero_graph(
        claims=claims,
        providers=providers,
        beneficiaries=beneficiaries,
        schema=SCHEMA_V1_1,
        train_provider_ids=train_ids,
        val_provider_ids=val_ids,
        fold_id=0,
    )
    node_providers = set(graph.node_frames["provider"][COL_PROVIDER].astype(str))
    assert "PRV3" not in node_providers


def test_v1b_includes_seen_by_edges(
    sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
):
    claims, providers, beneficiaries = _processed_mini_tables(
        sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers
    )
    graph = build_hetero_graph(
        claims=claims,
        providers=providers,
        beneficiaries=beneficiaries,
        schema=get_schema("v1b"),
        train_provider_ids=sorted(providers[COL_PROVIDER].astype(str)),
    )
    assert graph.relation_edge_count("seen_by") > 0
