"""Tests for hgad_cms.data.cleaner."""

import pandas as pd

from hgad_cms.constants import COL_GENDER_NORM, COL_REIMBURSED
from hgad_cms.data.cleaner import (
    clean_beneficiaries,
    clean_claims,
    clip_reimbursement_outliers,
    date_parse_success_rate,
    fill_physician_nulls,
    normalize_gender,
    parse_dates,
)


def test_normalize_gender_codes():
    series = pd.Series(["F", "M", "m", "FF", "MF", "NA", None])
    codes = normalize_gender(series)
    assert codes.tolist()[:5] == [0, 1, 1, 0, 1]
    assert codes.iloc[5] == 2


def test_parse_dates_mixed_formats():
    series = pd.Series(["2009-04-12", "13/2/2025", "NA"])
    parsed = parse_dates(series)
    assert parsed.notna().sum() == 2


def test_date_parse_success_rate():
    series = pd.Series(["2009-01-01", "invalid", "NA"])
    rate = date_parse_success_rate(series)
    assert 0.0 <= rate <= 1.0


def test_fill_physician_nulls():
    frame = pd.DataFrame({"AttendingPhysician": ["PHY1", "NA", "nan"]})
    cleaned = fill_physician_nulls(frame)
    assert cleaned["AttendingPhysician"].isna().sum() == 2


def test_clip_reimbursement_outliers():
    frame = pd.DataFrame({COL_REIMBURSED: [0.0, 100.0, 1_000_000.0]})
    clipped = clip_reimbursement_outliers(frame, quantile=0.9)
    assert clipped[COL_REIMBURSED].max() < 1_000_000.0


def test_clean_claims_adds_duration(sample_inpatient, sample_outpatient):
    from hgad_cms.data.merger import merge_claims

    claims = merge_claims(sample_inpatient, sample_outpatient)
    cleaned = clean_claims(claims)
    assert "claim_duration_days" in cleaned.columns
    assert cleaned["ClaimID"].dtype == object


def test_clean_beneficiaries_gender_norm(sample_beneficiaries):
    cleaned = clean_beneficiaries(sample_beneficiaries)
    assert COL_GENDER_NORM in cleaned.columns
