#!/usr/bin/env python3
"""
Phase 1 preprocessing entry point.

Loads raw CMS CSV files, merges and cleans data, validates Gate G1,
and writes processed parquet artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hgad_cms.constants import (
    OUTPUT_BENEFICIARIES,
    OUTPUT_CLAIMS,
    OUTPUT_MANIFEST,
    OUTPUT_PROVIDERS,
)
from hgad_cms.data.cleaner import clean_beneficiaries, clean_claims, clean_providers
from hgad_cms.data.loader import load_raw_cms
from hgad_cms.data.merger import (
    build_beneficiary_table,
    build_provider_table,
    merge_claims,
    merge_provider_labels,
)
from hgad_cms.data.validator import validate_cms_data
from hgad_cms.exceptions import DataValidationError, HGADError
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.preprocess")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CMS Healthcare Provider Fraud Detection — Phase 1 preprocess",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw Kaggle CMS CSV files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed parquet files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip G1 validation (not recommended)",
    )
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("Wrote %s (%s rows, %s cols)", path, len(frame), len(frame.columns))


def _build_manifest(
    raw_dir: Path,
    processed_dir: Path,
    raw_data_paths: dict[str, Path],
    row_counts: dict[str, int],
    validation_passed: bool,
) -> dict[str, object]:
    source_files = {
        name: {"path": str(path), "sha256": _file_sha256(path)}
        for name, path in raw_data_paths.items()
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "source_files": source_files,
        "row_counts": row_counts,
        "outputs": {
            "claims": str((processed_dir / OUTPUT_CLAIMS).resolve()),
            "providers": str((processed_dir / OUTPUT_PROVIDERS).resolve()),
            "beneficiaries": str((processed_dir / OUTPUT_BENEFICIARIES).resolve()),
        },
        "gate_g1_passed": validation_passed,
    }


def run_preprocess(
    raw_dir: Path,
    processed_dir: Path,
    skip_validation: bool = False,
) -> dict[str, object]:
    """
    Execute the full Phase 1 preprocessing pipeline.

    Returns
    -------
    dict
        Manifest dictionary written to disk.
    """
    raw = load_raw_cms(raw_dir)

    unified_claims = merge_claims(raw.inpatient, raw.outpatient)
    labeled_claims = merge_provider_labels(unified_claims, raw.providers)

    claims_clean = clean_claims(labeled_claims)
    providers_clean = clean_providers(build_provider_table(raw.providers))
    beneficiaries_clean = clean_beneficiaries(build_beneficiary_table(raw.beneficiaries))

    validation_passed = False
    if skip_validation:
        logger.warning("G1 validation skipped by flag")
    else:
        report = validate_cms_data(claims_clean, providers_clean, beneficiaries_clean)
        report.raise_if_failed()
        validation_passed = True
        logger.info("Gate G1 PASSED")

    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(claims_clean, processed_dir / OUTPUT_CLAIMS)
    _write_parquet(providers_clean, processed_dir / OUTPUT_PROVIDERS)
    _write_parquet(beneficiaries_clean, processed_dir / OUTPUT_BENEFICIARIES)

    row_counts = {
        "claims": len(claims_clean),
        "providers": len(providers_clean),
        "beneficiaries": len(beneficiaries_clean),
        "fraud_providers": int(providers_clean["fraud_label"].sum()),
    }

    manifest = _build_manifest(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_data_paths={
            "providers": raw.paths.providers,
            "beneficiaries": raw.paths.beneficiaries,
            "inpatient": raw.paths.inpatient,
            "outpatient": raw.paths.outpatient,
        },
        row_counts=row_counts,
        validation_passed=validation_passed,
    )

    manifest_path = processed_dir / OUTPUT_MANIFEST
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)

    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    try:
        manifest = run_preprocess(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            skip_validation=args.skip_validation,
        )
        logger.info(
            "Preprocessing complete: %s claims, %s providers, %s beneficiaries",
            manifest["row_counts"]["claims"],
            manifest["row_counts"]["providers"],
            manifest["row_counts"]["beneficiaries"],
        )
        return 0
    except DataValidationError as exc:
        logger.error("G1 validation failed: %s", exc)
        return 2
    except HGADError as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
