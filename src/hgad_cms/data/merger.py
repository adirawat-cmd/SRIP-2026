"""Claim merging and provider label attachment."""

from __future__ import annotations

import logging

import pandas as pd

from hgad_cms.constants import (
    CLAIM_ALL_COLUMNS,
    COL_BENE_ID,
    COL_CLAIM_ID,
    COL_FRAUD_LABEL,
    COL_IS_INPATIENT,
    COL_POTENTIAL_FRAUD,
    COL_PROVIDER,
    FRAUD_LABEL_MAP,
    INPATIENT_EXTRA_COLUMNS,
    OUTPATIENT_EXTRA_COLUMNS,
)
from hgad_cms.exceptions import DataMergeError

logger = logging.getLogger(__name__)


def _align_claim_schema(frame: pd.DataFrame, extra_columns: tuple[str, ...]) -> pd.DataFrame:
    """Ensure claim frame contains all columns in CLAIM_ALL_COLUMNS."""
    aligned = frame.copy()
    for column in CLAIM_ALL_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[list(CLAIM_ALL_COLUMNS)]


def attach_claim_type(
    inpatient: pd.DataFrame,
    outpatient: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tag inpatient and outpatient frames with is_inpatient flag.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Copies of input frames with ``is_inpatient`` column set.
    """
    ip = inpatient.copy()
    op = outpatient.copy()
    ip[COL_IS_INPATIENT] = True
    op[COL_IS_INPATIENT] = False
    return ip, op


def merge_claims(inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate inpatient and outpatient claims into a unified schema.

    Parameters
    ----------
    inpatient:
        Inpatient claim records.
    outpatient:
        Outpatient claim records.

    Returns
    -------
    pd.DataFrame
        Unified claims table with aligned columns and is_inpatient flag.

    Raises
    ------
    DataMergeError
        If duplicate ClaimID values exist or merge yields unexpected row count.
    """
    ip, op = attach_claim_type(inpatient, outpatient)
    ip_aligned = _align_claim_schema(ip, INPATIENT_EXTRA_COLUMNS)
    op_aligned = _align_claim_schema(op, OUTPATIENT_EXTRA_COLUMNS)

    combined = pd.concat([ip_aligned, op_aligned], ignore_index=True, sort=False)

    duplicate_ids = combined[COL_CLAIM_ID].duplicated()
    if duplicate_ids.any():
        dup_count = int(duplicate_ids.sum())
        sample = combined.loc[duplicate_ids, COL_CLAIM_ID].head(5).tolist()
        raise DataMergeError(
            f"Found {dup_count} duplicate ClaimID values after merge. Sample: {sample}"
        )

    expected_rows = len(inpatient) + len(outpatient)
    if len(combined) != expected_rows:
        raise DataMergeError(
            f"Merged claim row count {len(combined)} != "
            f"inpatient ({len(inpatient)}) + outpatient ({len(outpatient)})"
        )

    logger.info(
        "Merged claims: %s total (%s inpatient, %s outpatient)",
        len(combined),
        len(inpatient),
        len(outpatient),
    )
    return combined


def merge_provider_labels(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join provider fraud labels onto claims.

    Parameters
    ----------
    claims:
        Unified claims DataFrame with Provider column.
    providers:
        Provider label table.

    Returns
    -------
    pd.DataFrame
        Claims with PotentialFraud column attached.

    Raises
    ------
    DataMergeError
        If join coverage is incomplete or unknown fraud labels appear.
    """
    label_cols = providers[[COL_PROVIDER, COL_POTENTIAL_FRAUD]].drop_duplicates(
        subset=[COL_PROVIDER]
    )

    dup_providers = label_cols[COL_PROVIDER].duplicated()
    if dup_providers.any():
        raise DataMergeError(
            f"Provider label table has duplicate Provider IDs: "
            f"{int(dup_providers.sum())}"
        )

    unknown_labels = set(label_cols[COL_POTENTIAL_FRAUD].unique()) - set(FRAUD_LABEL_MAP)
    if unknown_labels:
        raise DataMergeError(f"Unknown PotentialFraud labels: {sorted(unknown_labels)}")

    merged = claims.merge(label_cols, on=COL_PROVIDER, how="left", validate="m:1")

    missing_labels = merged[COL_POTENTIAL_FRAUD].isna().sum()
    if missing_labels > 0:
        sample_providers = (
            merged.loc[merged[COL_POTENTIAL_FRAUD].isna(), COL_PROVIDER]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise DataMergeError(
            f"{missing_labels} claims have no provider label match. "
            f"Sample providers: {sample_providers}"
        )

    logger.info("Attached provider labels to %s claims", len(merged))
    return merged


def build_provider_table(providers: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize provider label table with integer fraud_label column.

    Parameters
    ----------
    providers:
        Raw provider CSV content.

    Returns
    -------
    pd.DataFrame
        Provider table with fraud_label int column.

    Raises
    ------
    DataMergeError
        If labels are invalid or provider IDs duplicate.
    """
    table = providers[[COL_PROVIDER, COL_POTENTIAL_FRAUD]].copy()
    if table[COL_PROVIDER].duplicated().any():
        raise DataMergeError("Duplicate provider IDs in provider label file")

    unknown = set(table[COL_POTENTIAL_FRAUD].unique()) - set(FRAUD_LABEL_MAP)
    if unknown:
        raise DataMergeError(f"Unknown PotentialFraud values: {sorted(unknown)}")

    table[COL_FRAUD_LABEL] = table[COL_POTENTIAL_FRAUD].map(FRAUD_LABEL_MAP)
    if table[COL_FRAUD_LABEL].isna().any():
        raise DataMergeError("Failed to map one or more PotentialFraud labels")

    table[COL_PROVIDER] = table[COL_PROVIDER].astype(str)
    return table


def build_beneficiary_table(beneficiaries: pd.DataFrame) -> pd.DataFrame:
    """
    Copy beneficiary table with BeneID as string for consistent joins.

    Parameters
    ----------
    beneficiaries:
        Raw beneficiary CSV content.

    Returns
    -------
    pd.DataFrame
        Beneficiary table with string BeneID.
    """
    table = beneficiaries.copy()
    table[COL_BENE_ID] = table[COL_BENE_ID].astype(str)
    return table


def extract_labeled_claims(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full merge pipeline: claims + labels and normalized provider table.

    Parameters
    ----------
    claims:
        Unified claims (without labels).
    providers:
        Raw provider labels.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        labeled_claims, provider_table
    """
    labeled = merge_provider_labels(claims, providers)
    provider_table = build_provider_table(providers)
    return labeled, provider_table
