"""Tests for hgad_cms.data.validator."""

import pandas as pd
import pytest

from hgad_cms.constants import (
    COL_BENE_ID,
    COL_CLAIM_ID,
    COL_FRAUD_LABEL,
    COL_POTENTIAL_FRAUD,
    COL_PROVIDER,
    FRAUD_LABEL_NO,
    FRAUD_LABEL_YES,
)
from hgad_cms.data.validator import validate_cms_data
from hgad_cms.exceptions import DataValidationError


def _mini_valid_tables():
    providers = pd.DataFrame(
        {
            COL_PROVIDER: ["PRV1", "PRV2"],
            COL_POTENTIAL_FRAUD: [FRAUD_LABEL_NO, FRAUD_LABEL_YES],
            COL_FRAUD_LABEL: [0, 1],
        }
    )
    beneficiaries = pd.DataFrame({COL_BENE_ID: ["B1", "B2", "B3"]})
    claims = pd.DataFrame(
        {
            COL_CLAIM_ID: ["C1", "C2", "C3"],
            COL_BENE_ID: ["B1", "B2", "B3"],
            COL_PROVIDER: ["PRV1", "PRV2", "PRV1"],
            COL_POTENTIAL_FRAUD: [FRAUD_LABEL_NO, FRAUD_LABEL_YES, FRAUD_LABEL_NO],
        }
    )
    return claims, providers, beneficiaries


def test_validate_mini_tables_passes_join_checks():
    claims, providers, beneficiaries = _mini_valid_tables()
    report = validate_cms_data(claims, providers, beneficiaries)
    join_checks = {c.name: c.passed for c in report.checks if "join" in c.name}
    assert join_checks["provider_claim_join"] is True
    assert join_checks["beneficiary_claim_join"] is True


def test_validate_raises_on_duplicate_claims():
    claims, providers, beneficiaries = _mini_valid_tables()
    claims = pd.concat([claims, claims.iloc[[0]]], ignore_index=True)
    report = validate_cms_data(claims, providers, beneficiaries)
    assert report.passed is False
    with pytest.raises(DataValidationError):
        report.raise_if_failed()


def test_validate_count_checks_fail_on_mini_sample():
    claims, providers, beneficiaries = _mini_valid_tables()
    report = validate_cms_data(claims, providers, beneficiaries)
    count_check = next(c for c in report.checks if c.name == "provider_count")
    assert count_check.passed is False
