"""Data cleaning and type normalization for CMS tables."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from hgad_cms.constants import (
    COL_ADMISSION,
    COL_ADMIT_DX,
    COL_ATTENDING,
    COL_BENE_ID,
    COL_CLAIM_DURATION_DAYS,
    COL_CLAIM_END,
    COL_CLAIM_ID,
    COL_CLAIM_START,
    COL_DEDUCTIBLE,
    COL_DISCHARGE,
    COL_GENDER,
    COL_GENDER_NORM,
    COL_OPERATING,
    COL_OTHER_PHYSICIAN,
    COL_PROVIDER,
    COL_REIMBURSED,
    GENDER_FEMALE_CODE,
    GENDER_FEMALE_VALUES,
    GENDER_MALE_CODE,
    GENDER_MALE_VALUES,
    GENDER_UNK_CODE,
    GENDER_UNK_VALUE,
    NA_STRINGS,
    REIMBURSEMENT_CLIP_QUANTILE,
)
from hgad_cms.exceptions import DataCleanError

logger = logging.getLogger(__name__)

PHYSICIAN_COLUMNS: tuple[str, ...] = (
    COL_ATTENDING,
    COL_OPERATING,
    COL_OTHER_PHYSICIAN,
)

DATE_COLUMNS_CLAIMS: tuple[str, ...] = (
    COL_CLAIM_START,
    COL_CLAIM_END,
    COL_ADMISSION,
    COL_DISCHARGE,
)

DATE_COLUMNS_BENEFICIARY: tuple[str, ...] = ("DOB", "DOD")


def _replace_na_strings(series: pd.Series) -> pd.Series:
    """Replace string NA tokens with pandas NA."""
    if series.dtype == object:
        stripped = series.astype(str).str.strip()
        mask = stripped.isin(NA_STRINGS) | stripped.eq("nan")
        return series.mask(mask, other=pd.NA)
    return series


def normalize_gender(series: pd.Series) -> pd.Series:
    """
    Map raw gender strings to normalized codes.

    Returns
    -------
    pd.Series
        Integer codes: 0=female, 1=male, 2=unknown.
    """
    cleaned = _replace_na_strings(series.astype(str).str.strip())

    def _map_value(value: object) -> int:
        if pd.isna(value):
            return GENDER_UNK_CODE
        token = str(value).strip()
        if token in GENDER_FEMALE_VALUES:
            return GENDER_FEMALE_CODE
        if token in GENDER_MALE_VALUES:
            return GENDER_MALE_CODE
        return GENDER_UNK_CODE

    mapped = cleaned.map(_map_value)
    return mapped.astype("Int64")


def normalize_gender_labels(series: pd.Series) -> pd.Series:
    """Return string labels F/M/UNK for reporting."""
    codes = normalize_gender(series)
    label_map = {
        GENDER_FEMALE_CODE: "F",
        GENDER_MALE_CODE: "M",
        GENDER_UNK_CODE: GENDER_UNK_VALUE,
    }
    return codes.map(label_map)


def parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse mixed-format date strings to timezone-naive datetime64.

    Tries ISO format first, then day-first for remaining values.
    """
    as_string = _replace_na_strings(series)
    iso_parsed = pd.to_datetime(as_string, errors="coerce", utc=False)
    remaining = iso_parsed.isna() & as_string.notna()
    if remaining.any():
        dayfirst = pd.to_datetime(as_string[remaining], errors="coerce", dayfirst=True)
        iso_parsed.loc[remaining] = dayfirst
    return iso_parsed


def fill_physician_nulls(claims: pd.DataFrame) -> pd.DataFrame:
    """Ensure physician ID columns use pandas NA instead of string NA tokens."""
    cleaned = claims.copy()
    for column in PHYSICIAN_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _replace_na_strings(cleaned[column])
    return cleaned


def clip_reimbursement_outliers(
    claims: pd.DataFrame,
    quantile: float = REIMBURSEMENT_CLIP_QUANTILE,
) -> pd.DataFrame:
    """
    Clip InscClaimAmtReimbursed at the upper quantile to limit extreme outliers.

    Values above the clip threshold are capped; negative values are set to 0.
    """
    if COL_REIMBURSED not in claims.columns:
        raise DataCleanError(f"Missing reimbursement column: {COL_REIMBURSED}")

    cleaned = claims.copy()
    amounts = pd.to_numeric(cleaned[COL_REIMBURSED], errors="coerce")
    if amounts.isna().all():
        raise DataCleanError("All reimbursement values are null after numeric conversion")

    negative_count = int((amounts < 0).sum())
    if negative_count > 0:
        logger.warning("Clamping %s negative reimbursement values to 0", negative_count)
        amounts = amounts.clip(lower=0)

    upper = amounts.quantile(quantile)
    clipped_count = int((amounts > upper).sum())
    if clipped_count > 0:
        logger.info(
            "Clipping %s reimbursement values above quantile %.4f (value=%.2f)",
            clipped_count,
            quantile,
            upper,
        )
    cleaned[COL_REIMBURSED] = amounts.clip(upper=upper)
    return cleaned


def _clean_date_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = parse_dates(cleaned[column])
    return cleaned


def _compute_claim_duration(claims: pd.DataFrame) -> pd.DataFrame:
    cleaned = claims.copy()
    if COL_CLAIM_START in cleaned.columns and COL_CLAIM_END in cleaned.columns:
        delta = cleaned[COL_CLAIM_END] - cleaned[COL_CLAIM_START]
        cleaned[COL_CLAIM_DURATION_DAYS] = (
            delta.dt.total_seconds().div(86400.0).astype("float64")
        )
        negative = cleaned[COL_CLAIM_DURATION_DAYS] < 0
        if negative.any():
            logger.warning(
                "Found %s claims with negative duration; setting to 0",
                int(negative.sum()),
            )
            cleaned.loc[negative, COL_CLAIM_DURATION_DAYS] = 0.0
    return cleaned


def clean_claims(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to unified claims table.

    Parameters
    ----------
    claims:
        Merged inpatient/outpatient claims.

    Returns
    -------
    pd.DataFrame
        Cleaned claims.

    Raises
    ------
    DataCleanError
        On critical cleaning failures.
    """
    cleaned = claims.copy()
    cleaned[COL_CLAIM_ID] = cleaned[COL_CLAIM_ID].astype(str)
    cleaned[COL_BENE_ID] = cleaned[COL_BENE_ID].astype(str)
    cleaned[COL_PROVIDER] = cleaned[COL_PROVIDER].astype(str)

    cleaned = _clean_date_columns(cleaned, DATE_COLUMNS_CLAIMS)
    cleaned = fill_physician_nulls(cleaned)
    cleaned[COL_REIMBURSED] = pd.to_numeric(cleaned[COL_REIMBURSED], errors="coerce")
    if COL_DEDUCTIBLE in cleaned.columns:
        cleaned[COL_DEDUCTIBLE] = pd.to_numeric(cleaned[COL_DEDUCTIBLE], errors="coerce")
    cleaned = clip_reimbursement_outliers(cleaned)
    cleaned = _compute_claim_duration(cleaned)

    if cleaned[COL_CLAIM_ID].isna().any():
        raise DataCleanError("ClaimID contains null values after cleaning")

    logger.info("Cleaned claims frame: %s rows", len(cleaned))
    return cleaned


def clean_beneficiaries(beneficiaries: pd.DataFrame) -> pd.DataFrame:
    """
    Clean beneficiary table: dates, gender normalization, ID typing.

    Parameters
    ----------
    beneficiaries:
        Raw beneficiary records.

    Returns
    -------
    pd.DataFrame
        Cleaned beneficiaries with gender_norm column.
    """
    cleaned = beneficiaries.copy()
    cleaned[COL_BENE_ID] = cleaned[COL_BENE_ID].astype(str)
    cleaned = _clean_date_columns(cleaned, DATE_COLUMNS_BENEFICIARY)
    cleaned[COL_GENDER_NORM] = normalize_gender(cleaned[COL_GENDER])

    for column in ("State", "County"):
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    logger.info("Cleaned beneficiaries frame: %s rows", len(cleaned))
    return cleaned


def clean_providers(providers: pd.DataFrame) -> pd.DataFrame:
    """Ensure provider IDs are string-typed."""
    cleaned = providers.copy()
    cleaned[COL_PROVIDER] = cleaned[COL_PROVIDER].astype(str)
    return cleaned


def date_parse_success_rate(series: pd.Series) -> float:
    """Return fraction of non-null input values successfully parsed as dates."""
    raw = series.copy()
    non_null = raw.notna() & ~raw.astype(str).str.strip().isin(NA_STRINGS)
    if non_null.sum() == 0:
        return 1.0
    parsed = parse_dates(raw)
    success = parsed.notna() & non_null
    return float(success.sum()) / float(non_null.sum())
