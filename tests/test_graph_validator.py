"""Tests for hgad_cms.graph.validator."""

import pandas as pd
import pytest

from hgad_cms.constants import COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.data.merger import merge_claims, merge_provider_labels
from hgad_cms.exceptions import GraphValidationError
from hgad_cms.graph.builder import build_hetero_graph
from hgad_cms.graph.constants import SCHEMA_V1_1
from hgad_cms.graph.validator import validate_hetero_graph


def _mini_graph(sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers):
    claims = merge_claims(sample_inpatient, sample_outpatient)
    claims = merge_provider_labels(claims, sample_providers)
    providers = sample_providers.copy()
    providers[COL_FRAUD_LABEL] = providers["PotentialFraud"].map({"No": 0, "Yes": 1})
    train_ids = sorted(providers[COL_PROVIDER].astype(str).tolist())
    return build_hetero_graph(
        claims=claims,
        providers=providers,
        beneficiaries=sample_beneficiaries,
        schema=SCHEMA_V1_1,
        train_provider_ids=train_ids,
        val_provider_ids=[],
        fold_id=0,
    )


def test_validate_mini_graph_passes(sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers):
    graph = _mini_graph(sample_inpatient, sample_outpatient, sample_beneficiaries, sample_providers)
    report = validate_hetero_graph(graph, SCHEMA_V1_1, reference_mode=False)
    assert report.passed is True


def test_validate_detects_val_provider_leakage():
    chronic = {
        f"ChronicCond_{c}": [2]
        for c in [
            "Alzheimer",
            "Heartfailure",
            "KidneyDisease",
            "Cancer",
            "ObstrPulmonary",
            "Depression",
            "Diabetes",
            "IschemicHeart",
            "Osteoporasis",
            "rheumatoidarthritis",
            "stroke",
        ]
    }
    graph = build_hetero_graph(
        claims=pd.DataFrame(
            {
                "Provider": ["P1"],
                "BeneID": ["B1"],
                "ClaimID": ["C1"],
                "InscClaimAmtReimbursed": [1.0],
                "is_inpatient": [False],
                "AttendingPhysician": ["PHY1"],
                "OperatingPhysician": [None],
                "OtherPhysician": [None],
                "ClaimStartDt": ["2009-01-01"],
                "ClaimEndDt": ["2009-01-01"],
                "ClmDiagnosisCode_1": ["D1"],
                "ClmProcedureCode_1": ["PR1"],
            }
        ),
        providers=pd.DataFrame({COL_PROVIDER: ["P1", "P2"], COL_FRAUD_LABEL: [0, 1]}),
        beneficiaries=pd.DataFrame(
            {
                "BeneID": ["B1"],
                "Gender": ["F"],
                "gender_norm": [0],
                "Race": [1],
                "RenalDiseaseIndicator": [0],
                "State": [1],
                "County": [1],
                "NoOfMonths_PartACov": [12],
                "NoOfMonths_PartBCov": [12],
                **chronic,
            }
        ),
        schema=SCHEMA_V1_1,
        train_provider_ids=["P1"],
        val_provider_ids=["P2"],
        fold_id=0,
    )
    # Manually inject val provider into node set to simulate leakage
    bad = graph.node_frames["provider"].copy()
    bad.loc[len(bad)] = {COL_PROVIDER: "P2", COL_FRAUD_LABEL: 1, "node_idx": len(bad)}
    graph.node_frames["provider"] = bad
    report = validate_hetero_graph(graph, SCHEMA_V1_1, reference_mode=False)
    assert report.passed is False
    with pytest.raises(GraphValidationError):
        report.raise_if_failed()
