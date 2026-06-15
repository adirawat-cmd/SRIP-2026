#!/usr/bin/env python3
"""
Complete GraphSAGE evaluation: HPO search, 5-fold benchmark, plots, diagnosis report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from hgad_cms.evaluation.significance import paired_test
from hgad_cms.exceptions import GNNError, HGADError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME, SPLITS_DIR_NAME
from hgad_cms.graphsage.config import GraphSAGEConfig, iter_search_grid
from hgad_cms.graphsage.cross_validation import (
    GNN_DIR_NAME,
    RESULTS_DIR_NAME,
    load_graphsage_result,
    run_graphsage_cv,
    save_graphsage_result,
)
from hgad_cms.graphsage.evaluation import (
    build_hpo_leaderboard,
    compare_factor,
    diagnose_performance,
    hpo_leaderboard_dataframe,
    load_epoch_curves,
    plot_factor_comparisons,
    plot_training_curves,
    run_hpo_search,
    write_diagnosis_report,
)
from hgad_cms.tracking.research_docs import on_benchmark_complete, on_gate_result, sync_all
from hgad_cms.tracking.experiment import ExperimentTracker
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.evaluate_graphsage")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphSAGE HPO evaluation and diagnosis")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data") / SPLITS_DIR_NAME)
    parser.add_argument("--graphs-dir", type=Path, default=Path("artifacts") / GRAPHS_DIR_NAME)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts") / RESULTS_DIR_NAME)
    parser.add_argument(
        "--search-mode",
        choices=("eval", "full", "quick"),
        default="eval",
        help="eval=12 configs (h×L×agg), full=36, quick=6",
    )
    parser.add_argument("--schema", type=str, default="v1.1")
    parser.add_argument("--search-fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--run-id", type=str, default="graphsage_eval")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _runtime_configs(
    mode: str,
    *,
    schema: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> list[GraphSAGEConfig]:
    configs = []
    for base in iter_search_grid(mode):  # type: ignore[arg-type]
        configs.append(
            GraphSAGEConfig(
                hidden_dim=base.hidden_dim,
                num_layers=base.num_layers,
                dropout=base.dropout,
                aggregator=base.aggregator,
                fanout=base.fanout,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                schema_name=schema,
            )
        )
    return configs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)

    output_dir = args.results_dir / GNN_DIR_NAME
    plots_dir = output_dir / "plots"
    tracker = ExperimentTracker(output_dir / "experiments.jsonl")

    try:
        configs = _runtime_configs(
            args.search_mode,
            schema=args.schema,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        logger.info("HPO grid: %s configs (%s mode)", len(configs), args.search_mode)

        hpo_results = run_hpo_search(
            configs,
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            graphs_dir=args.graphs_dir,
            output_dir=output_dir,
            search_fold=args.search_fold,
            device=args.device,
            run_id=args.run_id,
            resume=args.resume,
            tracker=tracker,
        )

        leaderboard = build_hpo_leaderboard(hpo_results, top_k=args.top_k)
        leaderboard_df = hpo_leaderboard_dataframe(leaderboard)
        leaderboard_path = output_dir / "hpo_leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        logger.info("Top config: %s AUPRC=%.4f", leaderboard[0].config_id, leaderboard[0].auprc)

        benchmark = None
        if not args.skip_benchmark:
            best_config = leaderboard[0].config
            benchmark_path = output_dir / "graphsage_benchmark.json"
            if args.resume and benchmark_path.is_file():
                benchmark = load_graphsage_result(benchmark_path)
                logger.info("Loaded existing 5-fold benchmark")
            else:
                logger.info("Running 5-fold benchmark: %s", best_config.config_id)
                benchmark = run_graphsage_cv(
                    best_config,
                    processed_dir=args.processed_dir,
                    splits_dir=args.splits_dir,
                    graphs_dir=args.graphs_dir,
                    n_folds=DEFAULT_N_FOLDS,
                    device=args.device,
                    tracker=tracker,
                    run_id=args.run_id,
                )
                payload = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_metric": "auprc",
                    **benchmark.to_dict(),
                }
                with benchmark_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)

        plot_paths: list[Path] = []
        if not args.skip_plots:
            curves = load_epoch_curves(output_dir / "experiments.jsonl")
            top_ids = [row.config_id for row in leaderboard[:5]]
            plot_paths.extend(plot_training_curves(curves, config_ids=top_ids, output_dir=plots_dir))
            plot_paths.extend(plot_factor_comparisons(hpo_results, plots_dir))

        diagnosis = diagnose_performance(
            hpo_results,
            benchmark,
            baseline_path=args.results_dir / "baselines" / "logistic_regression.json",
        )
        significance = None
        if benchmark is not None:
            baseline_path = args.results_dir / "baselines" / "logistic_regression.json"
            if baseline_path.is_file():
                from hgad_cms.evaluation.cross_validation import load_baseline_result

                baseline = load_baseline_result(baseline_path)
                if len(baseline.per_fold_scores["auprc"]) == len(benchmark.per_fold_scores["auprc"]):
                    significance = paired_test(
                        benchmark.per_fold_scores["auprc"],
                        baseline.per_fold_scores["auprc"],
                        metric="auprc",
                        model_a=benchmark.model_name,
                        model_b="logistic_regression",
                    ).to_dict()

        report_path = Path("docs") / "graphsage_diagnosis.md"
        write_diagnosis_report(
            output_path=report_path,
            leaderboard=leaderboard,
            hpo_results=hpo_results,
            benchmark=benchmark,
            diagnosis=diagnosis,
            plot_paths=plot_paths,
            significance=significance,
        )

        summary = {
            "search_mode": args.search_mode,
            "n_configs": len(configs),
            "top_config": leaderboard[0].config_id,
            "top_auprc_fold0": leaderboard[0].auprc,
            "benchmark_auprc_mean": (
                float(benchmark.summary["auprc"]["mean"]) if benchmark else None
            ),
            "beats_lr": diagnosis.get("beats_lr_baseline"),
            "recommend_rgcn": diagnosis.get("recommend_rgcn"),
            "report": str(report_path.resolve()),
        }
        with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        logger.info("Evaluation complete. Report: %s", report_path)
        mean_auprc = summary.get("benchmark_auprc_mean")
        if mean_auprc is not None:
            on_benchmark_complete(
                experiment_id="graphsage_eval",
                phase="Phase 4 — GraphSAGE",
                model_name=str(summary.get("top_config", "graphsage")),
                auprc_mean=float(mean_auprc),
                auprc_std=0.0,
                beats_lr=summary.get("beats_lr"),
            )
        on_gate_result(
            "G4",
            passed=bool(summary.get("beats_lr")),
            experiment_id="graphsage_eval",
            details=str(report_path.resolve()),
        )
        sync_all(results_dir=output_dir)
        return 0
    except (GNNError, HGADError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
