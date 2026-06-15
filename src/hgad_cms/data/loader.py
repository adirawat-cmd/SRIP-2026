"""Raw CMS dataset loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hgad_cms.constants import (
    BENEFICIARY_REQUIRED_COLUMNS,
    COL_POTENTIAL_FRAUD,
    COL_PROVIDER,
    RAW_FILE_PATTERNS,
)
from hgad_cms.exceptions import DataLoadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawCMSPaths:
    """Resolved absolute paths to the four train CSV files."""

    providers: Path
    beneficiaries: Path
    inpatient: Path
    outpatient: Path


@dataclass
class RawCMSData:
    """In-memory representation of raw CMS train tables."""

    providers: pd.DataFrame
    beneficiaries: pd.DataFrame
    inpatient: pd.DataFrame
    outpatient: pd.DataFrame
    paths: RawCMSPaths


def detect_file_names(data_dir: Path) -> RawCMSPaths:
    """
    Resolve train CSV paths using glob patterns defined in constants.

    Parameters
    ----------
    data_dir:
        Directory containing Kaggle CMS CSV files.

    Returns
    -------
    RawCMSPaths
        Resolved paths for providers, beneficiaries, inpatient, outpatient.

    Raises
    ------
    DataLoadError
        If data_dir is missing or any required file pattern is ambiguous / absent.
    """
    if not data_dir.is_dir():
        raise DataLoadError(f"Data directory does not exist: {data_dir}")

    resolved: dict[str, Path] = {}
    for key, pattern in RAW_FILE_PATTERNS.items():
        matches = sorted(data_dir.glob(pattern))
        if len(matches) == 0:
            raise DataLoadError(
                f"No file matching pattern '{pattern}' in {data_dir} (key={key})"
            )
        if len(matches) > 1:
            logger.warning(
                "Multiple files match pattern '%s': %s; using first match",
                pattern,
                [m.name for m in matches],
            )
        resolved[key] = matches[0]
        logger.debug("Resolved %s -> %s", key, matches[0])

    return RawCMSPaths(
        providers=resolved["providers"],
        beneficiaries=resolved["beneficiaries"],
        inpatient=resolved["inpatient"],
        outpatient=resolved["outpatient"],
    )


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    """Read a CSV with consistent settings and basic non-empty check."""
    try:
        frame = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Missing {description} file: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Failed to parse {description} CSV: {path}") from exc
    except OSError as exc:
        raise DataLoadError(f"OS error reading {description} CSV: {path}") from exc

    if frame.empty:
        raise DataLoadError(f"{description} CSV is empty: {path}")

    logger.info("Loaded %s: %s rows, %s columns", description, len(frame), len(frame.columns))
    return frame


def _validate_provider_columns(frame: pd.DataFrame) -> None:
    missing = {COL_PROVIDER, COL_POTENTIAL_FRAUD} - set(frame.columns)
    if missing:
        raise DataLoadError(f"Provider file missing columns: {sorted(missing)}")


def _validate_beneficiary_columns(frame: pd.DataFrame) -> None:
    missing = set(BENEFICIARY_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DataLoadError(f"Beneficiary file missing columns: {sorted(missing)}")


def _validate_claim_columns(frame: pd.DataFrame, claim_type: str) -> None:
    required = {"BeneID", "ClaimID", "Provider", "InscClaimAmtReimbursed"}
    missing = required - set(frame.columns)
    if missing:
        raise DataLoadError(
            f"{claim_type} claims file missing columns: {sorted(missing)}"
        )


def load_raw_cms(data_dir: Path) -> RawCMSData:
    """
    Load all CMS train CSV files into a RawCMSData container.

    Parameters
    ----------
    data_dir:
        Path to directory with raw Kaggle CMS files.

    Returns
    -------
    RawCMSData
        DataFrames for providers, beneficiaries, inpatient, outpatient.

    Raises
    ------
    DataLoadError
        On missing files, parse errors, or schema violations.
    """
    paths = detect_file_names(data_dir)

    providers = _read_csv(paths.providers, "providers")
    beneficiaries = _read_csv(paths.beneficiaries, "beneficiaries")
    inpatient = _read_csv(paths.inpatient, "inpatient")
    outpatient = _read_csv(paths.outpatient, "outpatient")

    _validate_provider_columns(providers)
    _validate_beneficiary_columns(beneficiaries)
    _validate_claim_columns(inpatient, "Inpatient")
    _validate_claim_columns(outpatient, "Outpatient")

    return RawCMSData(
        providers=providers,
        beneficiaries=beneficiaries,
        inpatient=inpatient,
        outpatient=outpatient,
        paths=paths,
    )
