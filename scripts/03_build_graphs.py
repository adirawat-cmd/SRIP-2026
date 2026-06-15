#!/usr/bin/env python3
"""
Build heterogeneous graphs per schema and CV fold (Phase 2).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from hgad_cms.constants import OUTPUT_BENEFICIARIES, OUTPUT_CLAIMS, OUTPUT_PROVIDERS
from hgad_cms.exceptions import GraphValidationError, HGADError, SplitError
from hgad_cms.graph.builder import build_hetero_graph
from hgad_cms.graph.constants import (
    DEFAULT_N_FOLDS,
    GRAPHS_DIR_NAME,
    REFERENCE_GRAPH_DIR,
    SCHEMA_REGISTRY,
    SPLITS_DIR_NAME,
    get_schema,
)
from hgad_cms.graph.io import save_hetero_graph, update_manifest_g2_status
from hgad_cms.graph.splits import create_provider_disjoint_splits, load_fold_split
from hgad_cms.graph.validator import validate_hetero_graph
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.build_graphs")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build heterogeneous graphs for CMS fraud")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data") / SPLITS_DIR_NAME,
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("artifacts") / GRAPHS_DIR_NAME,
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="v1.1",
        choices=sorted(SCHEMA_REGISTRY.keys()),
        help="Graph schema name",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Comma-separated fold ids or 'all'",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Build full-corpus reference graph (all providers)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Gate G2 validation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _parse_folds(folds_arg: str, n_folds: int) -> list[int]:
    if folds_arg.strip().lower() == "all":
        return list(range(n_folds))
    return [int(x.strip()) for x in folds_arg.split(",") if x.strip()]


def _load_processed_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_parquet(processed_dir / OUTPUT_CLAIMS)
    providers = pd.read_parquet(processed_dir / OUTPUT_PROVIDERS)
    beneficiaries = pd.read_parquet(processed_dir / OUTPUT_BENEFICIARIES)
    return claims, providers, beneficiaries


def run_build_graphs(
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    schema_name: str = "v1.1",
    folds: Sequence[int] | None = None,
    *,
    build_reference: bool = False,
    skip_validation: bool = False,
    n_folds: int = DEFAULT_N_FOLDS,
) -> dict:
    """
    Build graphs for selected folds and optional reference graph.

    Returns
    -------
    dict
        Summary manifest with G2 status per artifact.
    """
    schema = get_schema(schema_name)
    claims, providers, beneficiaries = _load_processed_tables(processed_dir)
    all_provider_ids = sorted(providers["Provider"].astype(str).unique())

    results: dict[str, dict] = {"schema": schema_name, "artifacts": {}}

    if build_reference:
        logger.info("Building reference graph schema=%s", schema_name)
        ref_graph = build_hetero_graph(
            claims=claims,
            providers=providers,
            beneficiaries=beneficiaries,
            schema=schema,
            train_provider_ids=all_provider_ids,
            val_provider_ids=[],
            fold_id=None,
            is_reference=True,
        )
        ref_dir = graphs_dir / schema_name / REFERENCE_GRAPH_DIR
        ref_manifest = save_hetero_graph(ref_graph, ref_dir)
        g2_passed = False
        if not skip_validation:
            report = validate_hetero_graph(ref_graph, schema, reference_mode=True)
            report.raise_if_failed()
            g2_passed = True
            logger.info("Gate G2 PASSED (reference graph)")
        update_manifest_g2_status(ref_dir, g2_passed)
        ref_manifest["gate_g2_passed"] = g2_passed
        results["artifacts"]["reference"] = ref_manifest

    fold_list = list(folds) if folds is not None else list(range(n_folds))
    for fold_id in fold_list:
        try:
            fold = load_fold_split(splits_dir, fold_id)
        except SplitError:
            logger.warning("Split file missing for fold %s; creating splits on the fly", fold_id)
            splits = create_provider_disjoint_splits(providers)
            fold = splits[fold_id]

        graph = build_hetero_graph(
            claims=claims,
            providers=providers,
            beneficiaries=beneficiaries,
            schema=schema,
            train_provider_ids=fold.train_provider_ids,
            val_provider_ids=fold.val_provider_ids,
            fold_id=fold_id,
            is_reference=False,
        )
        out_dir = graphs_dir / schema_name / f"fold_{fold_id}"
        manifest = save_hetero_graph(graph, out_dir)
        g2_passed = False
        if not skip_validation:
            report = validate_hetero_graph(graph, schema, reference_mode=False)
            report.raise_if_failed()
            g2_passed = True
            logger.info("Gate G2 PASSED fold=%s schema=%s", fold_id, schema_name)
        update_manifest_g2_status(out_dir, g2_passed)
        manifest["gate_g2_passed"] = g2_passed
        results["artifacts"][f"fold_{fold_id}"] = manifest

    return results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)
    try:
        fold_ids = _parse_folds(args.folds, DEFAULT_N_FOLDS)
        run_build_graphs(
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            graphs_dir=args.graphs_dir,
            schema_name=args.schema,
            folds=fold_ids,
            build_reference=args.reference,
            skip_validation=args.skip_validation,
        )
        return 0
    except GraphValidationError as exc:
        logger.error("G2 validation failed: %s", exc)
        return 2
    except (SplitError, HGADError) as exc:
        logger.error("Graph build failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
