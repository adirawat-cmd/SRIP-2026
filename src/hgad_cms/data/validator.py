"""G1 validation checks for processed CMS data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from hgad_cms.constants import (
    COL_BENE_ID,
    COL_CLAIM_ID,
    COL_FRAUD_LABEL,
    COL_POTENTIAL_FRAUD,
    COL_PROVIDER,
    EXPECTED_BENEFICIARY_COUNT,
    EXPECTED_CLAIM_COUNT,
    EXPECTED_FRAUD_RATE,
    EXPECTED_PROVIDER_COUNT,
    FRAUD_RATE_TOLERANCE,
    MIN_BENEFICIARY_CLAIM_JOIN_RATE,
    MIN_PROVIDER_CLAIM_JOIN_RATE,
)
from hgad_cms.exceptions import DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Single named validation outcome."""

    name: str
    passed: bool
    message: str
    expected: str | float | int | None = None
    actual: str | float | int | None = None


@dataclass
class ValidationReport:
    """Aggregate G1 validation report."""

    passed: bool
    checks: list[ValidationCheck] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        """Raise DataValidationError when any check failed."""
        if not self.passed:
            failed = [c for c in self.checks if not c.passed]
            messages = "; ".join(f"{c.name}: {c.message}" for c in failed)
            raise DataValidationError(f"G1 validation failed: {messages}")


def _check_count(
    name: str,
    actual: int,
    expected: int,
    tolerance: int = 0,
) -> ValidationCheck:
    lower = expected - tolerance
    upper = expected + tolerance
    passed = lower <= actual <= upper
    message = (
        f"count={actual} within [{lower}, {upper}]"
        if passed
        else f"count={actual} outside [{lower}, {upper}]"
    )
    return ValidationCheck(
        name=name,
        passed=passed,
        message=message,
        expected=expected,
        actual=actual,
    )


def _check_rate(
    name: str,
    actual: float,
    expected: float,
    tolerance: float,
) -> ValidationCheck:
    passed = abs(actual - expected) <= tolerance
    message = (
        f"rate={actual:.4f} within ±{tolerance} of {expected:.4f}"
        if passed
        else f"rate={actual:.4f} outside ±{tolerance} of {expected:.4f}"
    )
    return ValidationCheck(
        name=name,
        passed=passed,
        message=message,
        expected=expected,
        actual=round(actual, 6),
    )


def validate_cms_data(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
    beneficiaries: pd.DataFrame,
) -> ValidationReport:
    """
    Run Gate G1 validation checks on processed CMS tables.

    Parameters
    ----------
    claims:
        Cleaned labeled claims.
    providers:
        Provider table with fraud_label.
    beneficiaries:
        Cleaned beneficiary table.

    Returns
    -------
    ValidationReport
        Structured pass/fail report.

    Raises
    ------
    DataValidationError
        Only when ``report.raise_if_failed()`` is called and checks fail.
    """
    checks: list[ValidationCheck] = []

    checks.append(
        _check_count("provider_count", len(providers), EXPECTED_PROVIDER_COUNT)
    )
    checks.append(_check_count("claim_count", len(claims), EXPECTED_CLAIM_COUNT))
    checks.append(
        _check_count("beneficiary_count", len(beneficiaries), EXPECTED_BENEFICIARY_COUNT)
    )

    fraud_rate = float(providers[COL_FRAUD_LABEL].mean())
    checks.append(
        _check_rate(
            "fraud_rate",
            fraud_rate,
            EXPECTED_FRAUD_RATE,
            FRAUD_RATE_TOLERANCE,
        )
    )

    duplicate_claims = int(claims[COL_CLAIM_ID].duplicated().sum())
    checks.append(
        ValidationCheck(
            name="unique_claim_ids",
            passed=duplicate_claims == 0,
            message=(
                "all ClaimID unique"
                if duplicate_claims == 0
                else f"{duplicate_claims} duplicate ClaimID values"
            ),
            expected=0,
            actual=duplicate_claims,
        )
    )

    claim_providers = set(claims[COL_PROVIDER].astype(str).unique())
    label_providers = set(providers[COL_PROVIDER].astype(str).unique())
    provider_join_rate = len(claim_providers & label_providers) / max(len(claim_providers), 1)
    checks.append(
        ValidationCheck(
            name="provider_claim_join",
            passed=provider_join_rate >= MIN_PROVIDER_CLAIM_JOIN_RATE,
            message=f"join_rate={provider_join_rate:.4f}",
            expected=MIN_PROVIDER_CLAIM_JOIN_RATE,
            actual=round(provider_join_rate, 6),
        )
    )

    claim_benes = set(claims[COL_BENE_ID].astype(str).unique())
    bene_ids = set(beneficiaries[COL_BENE_ID].astype(str).unique())
    bene_join_rate = len(claim_benes & bene_ids) / max(len(claim_benes), 1)
    checks.append(
        ValidationCheck(
            name="beneficiary_claim_join",
            passed=bene_join_rate >= MIN_BENEFICIARY_CLAIM_JOIN_RATE,
            message=f"join_rate={bene_join_rate:.4f}",
            expected=MIN_BENEFICIARY_CLAIM_JOIN_RATE,
            actual=round(bene_join_rate, 6),
        )
    )

    missing_fraud_labels = int(claims[COL_POTENTIAL_FRAUD].isna().sum())
    checks.append(
        ValidationCheck(
            name="claims_have_fraud_labels",
            passed=missing_fraud_labels == 0,
            message=(
                "all claims labeled"
                if missing_fraud_labels == 0
                else f"{missing_fraud_labels} claims missing labels"
            ),
            expected=0,
            actual=missing_fraud_labels,
        )
    )

    fraud_providers_in_labels = set(
        providers.loc[providers[COL_FRAUD_LABEL] == 1, COL_PROVIDER].astype(str)
    )
    if len(fraud_providers_in_labels) == 0:
        checks.append(
            ValidationCheck(
                name="fraud_class_present",
                passed=False,
                message="no fraud providers in label table",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="fraud_class_present",
                passed=True,
                message=f"{len(fraud_providers_in_labels)} fraud providers",
                actual=len(fraud_providers_in_labels),
            )
        )

    passed = all(c.passed for c in checks)
    report = ValidationReport(passed=passed, checks=checks)

    for check in checks:
        log_fn = logger.info if check.passed else logger.error
        log_fn("G1 check [%s] %s", check.name, check.message)

    return report
