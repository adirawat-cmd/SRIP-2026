"""Tests for hgad_cms.data.merger."""

import pandas as pd
import pytest

from hgad_cms.constants import COL_IS_INPATIENT, COL_POTENTIAL_FRAUD
from hgad_cms.data.merger import (
    build_provider_table,
    merge_claims,
    merge_provider_labels,
)
from hgad_cms.exceptions import DataMergeError


def test_merge_claims_row_count(sample_inpatient, sample_outpatient):
    merged = merge_claims(sample_inpatient, sample_outpatient)
    assert len(merged) == 3
    assert COL_IS_INPATIENT in merged.columns
    assert merged[COL_IS_INPATIENT].tolist() == [True, True, False]


def test_merge_claims_unique_claim_ids(sample_inpatient, sample_outpatient):
    dup = sample_outpatient.copy()
    dup.iloc[0, dup.columns.get_loc("ClaimID")] = "C1"
    with pytest.raises(DataMergeError, match="duplicate ClaimID"):
        merge_claims(sample_inpatient, dup)


def test_merge_provider_labels_success(sample_inpatient, sample_outpatient, sample_providers):
    claims = merge_claims(sample_inpatient, sample_outpatient)
    labeled = merge_provider_labels(claims, sample_providers)
    assert labeled[COL_POTENTIAL_FRAUD].notna().all()


def test_merge_provider_labels_unknown_provider(sample_inpatient, sample_outpatient, sample_providers):
    claims = merge_claims(sample_inpatient, sample_outpatient)
    claims.loc[0, "Provider"] = "PRV_MISSING"
    with pytest.raises(DataMergeError, match="no provider label match"):
        merge_provider_labels(claims, sample_providers)


def test_build_provider_table_fraud_label(sample_providers):
    table = build_provider_table(sample_providers)
    assert "fraud_label" in table.columns
    assert set(table["fraud_label"].unique()) <= {0, 1}
