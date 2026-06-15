#!/usr/bin/env python3
"""
Run provider-level baseline models under 5-fold provider-disjoint CV (Phase 3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hgad_cms.baselines import BASELINE_REGISTRY
from hgad_cms.baselines.catboost_model import CatBoostBaseline
from hgad_cms.baselines.logistic_regression import LogisticRegressionBaseline
from hgad_cms.baselines.random_forest import RandomForestBaseline
from hgad_cms.baselines.rf_centrality import RFCentralityBaseline
from hgad_cms.evaluation.cross_validation import (
    BASELINES_DIR_NAME,
    RESULTS_DIR_NAME,
    build_comparison_table,
    load_baseline_result,
    run_provider_disjoint_cv,
    save_baseline_result,
    save_comparison_table,
    validate_gate_g3,
)
from hgad_cms.tracking.research_docs import on_gate_result, sync_all
from hgad_cms.evaluation.significance import compare_all_pairs
from hgad_cms.exceptions import BaselineError, EvaluationError, HGADError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME, SPLITS_DIR_NAME
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.run_baselines")

DEFAULT_MODELS = sorted(BASELINE_REGISTRY.keys())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CMS provider fraud baselines")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data") / SPLITS_DIR_NAME)
    parser.add_argument("--graphs-dir", type=Path, default=Path("artifacts") / GRAPHS_DIR_NAME)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts") / RESULTS_DIR_NAME)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated baseline names",
    )
    parser.add_argument("--schema", type=str, default="v1.1")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--skip-g3", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing baseline JSON results and run only missing models",
    )
    parser.add_argument("--journal-path", type=Path, default=Path("docs/experiment_journal.md"))
    parser.add_argument("--run-id", type=str, default="baselines_v1")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _model_factory(name: str):
    if name == RFCentralityBaseline.name:
        return RFCentralityBaseline
    if name == LogisticRegressionBaseline.name:
        return LogisticRegressionBaseline
    if name == RandomForestBaseline.name:
        return RandomForestBaseline
    if name == CatBoostBaseline.name:
        return CatBoostBaseline
    raise BaselineError(f"Unknown baseline: {name}")


def run_baselines(
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    results_dir: Path,
    *,
    model_names: list[str],
    schema_name: str = "v1.1",
    n_folds: int = DEFAULT_N_FOLDS,
    skip_g3: bool = False,
    resume: bool = False,
    journal_path: Path | None = None,
    run_id: str = "baselines_v1",
) -> dict[str, object]:
    """Execute baseline CV pipeline and return summary manifest."""
    output_dir = results_dir / BASELINES_DIR_NAME
    all_results = []

    for name in model_names:
        result_path = output_dir / f"{name}.json"
        if resume and result_path.is_file():
            logger.info("Resuming: loaded existing result for %s", name)
            all_results.append(load_baseline_result(result_path))
            continue

        use_centrality = name == RFCentralityBaseline.name
        result = run_provider_disjoint_cv(
            _model_factory(name),
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            graphs_dir=graphs_dir,
            schema_name=schema_name,
            n_folds=n_folds,
            use_centrality=use_centrality,
        )
        save_baseline_result(result, output_dir)
        all_results.append(result)

    comparison = build_comparison_table(all_results)
    comparison_path = save_comparison_table(comparison, output_dir)

    significance = compare_all_pairs(
        {r.model_name: r.per_fold_scores["auprc"] for r in all_results},
        metric="auprc",
    )
    significance_path = output_dir / "significance_auprc.json"
    with significance_path.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in significance], handle, indent=2)

    g3_report = None
    if not skip_g3:
        g3_report = validate_gate_g3(all_results)
        if g3_report["passed"]:
            logger.info("Gate G3 PASSED (best AUPRC=%.4f)", g3_report["best_auprc_mean"])
        else:
            logger.warning("Gate G3 FAILED")

    if journal_path is not None:
        append_baseline_journal_entry(all_results, journal_path=journal_path, run_id=run_id)

    manifest = {
        "models": model_names,
        "comparison_table": str(comparison_path.resolve()),
        "significance_table": str(significance_path.resolve()),
        "gate_g3_passed": g3_report["passed"] if g3_report else None,
        "results": {r.model_name: r.summary for r in all_results},
    }
    manifest_path = output_dir / "baseline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    try:
        manifest = run_baselines(
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            graphs_dir=args.graphs_dir,
            results_dir=args.results_dir,
            model_names=model_names,
            schema_name=args.schema,
            n_folds=args.n_folds,
            skip_g3=args.skip_g3,
            resume=args.resume,
            journal_path=args.journal_path,
            run_id=args.run_id,
        )
        logger.info("Baseline run complete. G3=%s", manifest.get("gate_g3_passed"))
        on_gate_result(
            "G3",
            passed=bool(manifest.get("gate_g3_passed")),
            experiment_id=args.run_id or "baselines_v1",
            details=f"manifest={args.results_dir}; gate_g3={manifest.get('gate_g3_passed')}",
        )
        sync_all(results_dir=Path(args.results_dir))
        return 0
    except EvaluationError as exc:
        logger.error("Evaluation/G3 failed: %s", exc)
        return 2
    except (BaselineError, HGADError) as exc:
        logger.error("Baseline run failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
