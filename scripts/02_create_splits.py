#!/usr/bin/env python3
"""
Create provider-disjoint stratified 5-fold CV splits (Phase 2).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from hgad_cms.constants import OUTPUT_PROVIDERS
from hgad_cms.exceptions import HGADError, SplitError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, DEFAULT_SPLIT_SEED, SPLITS_DIR_NAME
from hgad_cms.graph.splits import create_provider_disjoint_splits, save_splits
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.create_splits")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create provider-disjoint stratified CV splits",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed providers parquet",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data") / SPLITS_DIR_NAME,
        help="Output directory for fold split JSON files",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of stratified folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def run_create_splits(
    processed_dir: Path,
    splits_dir: Path,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    """Execute split creation pipeline."""
    providers_path = processed_dir / OUTPUT_PROVIDERS
    if not providers_path.is_file():
        raise SplitError(f"Missing providers parquet: {providers_path}")

    providers = pd.read_parquet(providers_path)
    splits = create_provider_disjoint_splits(providers, n_folds=n_folds, seed=seed)
    manifest = save_splits(splits, splits_dir, n_folds=n_folds, seed=seed)
    logger.info("Created %s folds in %s", len(splits), splits_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)
    try:
        run_create_splits(
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        return 0
    except SplitError as exc:
        logger.error("Split creation failed: %s", exc)
        return 2
    except HGADError as exc:
        logger.error("Error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
