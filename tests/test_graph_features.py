"""Tests for hgad_cms.graph.features."""

import pandas as pd
import pytest

from hgad_cms.constants import COL_BENE_ID, COL_PROVIDER
from hgad_cms.exceptions import GraphBuildError
from hgad_cms.graph.features import (
    assert_no_leakage_columns,
    build_beneficiary_features,
    build_provider_features,
    compute_top_code_lists,
)


def test_assert_no_leakage_columns_raises():
    with pytest.raises(GraphBuildError, match="PotentialFraud"):
        assert_no_leakage_columns(["total_claims", "PotentialFraud"])


def test_compute_top_code_lists(sample_inpatient, sample_outpatient):
    from hgad_cms.data.merger import merge_claims

    claims = merge_claims(sample_inpatient, sample_outpatient)
    dx, proc = compute_top_code_lists(claims)
    assert "D1" in dx
    assert "P1" in proc


def test_build_provider_features_no_leakage(
    sample_inpatient, sample_outpatient, sample_beneficiaries
):
    from hgad_cms.data.merger import merge_claims

    claims = merge_claims(sample_inpatient, sample_outpatient)
    providers = sorted(claims[COL_PROVIDER].astype(str).unique())
    dx, proc = compute_top_code_lists(claims)
    feat = build_provider_features(claims, sample_beneficiaries, providers, dx, proc)
    assert COL_PROVIDER in feat.columns
    assert "PotentialFraud" not in feat.columns
    assert "fraud_label" not in feat.columns
    assert len(feat) == len(providers)


def test_build_beneficiary_features_excludes_annual_reimb(sample_beneficiaries):
    feat = build_beneficiary_features(sample_beneficiaries, ["B1", "B2", "B3"])
    assert "IPAnnualReimbursementAmt" not in feat.columns
    assert len(feat) == 3
