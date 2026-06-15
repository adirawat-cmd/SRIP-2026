#!/usr/bin/env python3
"""
Train GraphSAGE benchmark under provider-disjoint 5-fold CV (Phase 4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hgad_cms.evaluation.significance import paired_test
from hgad_cms.exceptions import EvaluationError, GNNError, HGADError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME, SPLITS_DIR_NAME
from hgad_cms.graphsage.config import GraphSAGEConfig, LR_BASELINE_AUPRC, iter_search_grid
from hgad_cms.graphsage.cross_validation import (
    GNN_DIR_NAME,
    RESULTS_DIR_NAME,
    GraphSAGECVResult,
    load_graphsage_result,
    run_graphsage_cv,
    save_graphsage_result,
    validate_gate_g4,
)
from hgad_cms.tracking.experiment import ExperimentTracker
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.train_graphsage")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE benchmark (schema_v1.1)")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data") / SPLITS_DIR_NAME)
    parser.add_argument("--graphs-dir", type=Path, default=Path("artifacts") / GRAPHS_DIR_NAME)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts") / RESULTS_DIR_NAME)
    parser.add_argument("--schema", type=str, default="v1.1")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--search-mode",
        choices=("full", "quick", "single", "none"),
        default="quick",
        help="Hyperparameter search grid (default: quick on fold 0, then 5-fold benchmark)",
    )
    parser.add_argument("--search-fold", type=int, default=0, help="Fold used for HPO selection")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--skip-g4", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Load saved best-result JSON if present")
    parser.add_argument("--run-id", type=str, default="graphsage_v1")
    parser.add_argument("--journal-path", type=Path, default=Path("docs/experiment_journal.md"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _runtime_config(
    base: GraphSAGEConfig,
    *,
    schema_name: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> GraphSAGEConfig:
    return GraphSAGEConfig(
        hidden_dim=base.hidden_dim,
        num_layers=base.num_layers,
        dropout=base.dropout,
        aggregator=base.aggregator,
        fanout=base.fanout,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=base.learning_rate,
        weight_decay=base.weight_decay,
        seed=base.seed,
        schema_name=schema_name,
    )


def _select_best_config(
    configs: list[GraphSAGEConfig],
    *,
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    search_fold: int,
    device: str,
    tracker: ExperimentTracker,
    run_id: str,
) -> tuple[GraphSAGEConfig, list[GraphSAGECVResult]]:
    """Run HPO on a single fold and return the best configuration."""
    search_results: list[GraphSAGECVResult] = []
    best_config = configs[0]
    best_auprc = -1.0

    for config in configs:
        logger.info("HPO trial config=%s fold=%s", config.config_id, search_fold)
        result = run_graphsage_cv(
            config,
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            graphs_dir=graphs_dir,
            device=device,
            tracker=tracker,
            run_id=f"{run_id}_search",
            fold_ids=[search_fold],
        )
        search_results.append(result)
        auprc = float(result.summary["auprc"]["mean"])
        tracker.log_search_result(
            run_id=run_id,
            config=config,
            mean_auprc=auprc,
            std_auprc=float(result.summary["auprc"]["std"]),
            selected=False,
        )
        if auprc > best_auprc:
            best_auprc = auprc
            best_config = config

    tracker.log_search_result(
        run_id=run_id,
        config=best_config,
        mean_auprc=best_auprc,
        std_auprc=0.0,
        selected=True,
    )
    logger.info("Selected config=%s fold=%s auprc=%.4f", best_config.config_id, search_fold, best_auprc)
    return best_config, search_results


def _append_journal_entry(
    result: GraphSAGECVResult,
    *,
    journal_path: Path,
    run_id: str,
    g4_passed: bool,
) -> None:
    if not journal_path.is_file():
        return
    mean = result.summary["auprc"]["mean"]
    std = result.summary["auprc"]["std"]
    lines = [
        "",
        f"### {run_id}",
        "",
        "```",
        f"Run ID: {run_id}",
        "Experiment Name: Phase 4 — GraphSAGE benchmark (schema_v1.1)",
        f"Config: {result.config.config_id}",
        f"GraphSAGE AUPRC: {mean:.4f} +/- {std:.4f}",
        f"LR baseline AUPRC: {LR_BASELINE_AUPRC:.4f}",
        f"Gate G4 passed: {g4_passed}",
        "Next Action: Phase 5 — advanced GNN / fusion (if G4 passed)",
        "```",
        "",
    ]
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_graphsage_benchmark(
    *,
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    results_dir: Path,
    schema_name: str = "v1.1",
    n_folds: int = DEFAULT_N_FOLDS,
    search_mode: str = "quick",
    search_fold: int = 0,
    device: str = "auto",
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 15,
    skip_g4: bool = False,
    resume: bool = False,
    run_id: str = "graphsage_v1",
    journal_path: Path | None = None,
) -> dict[str, object]:
    """Execute GraphSAGE HPO + 5-fold benchmark."""
    output_dir = results_dir / GNN_DIR_NAME
    tracker = ExperimentTracker(output_dir / "experiments.jsonl")
    best_result_path = output_dir / "graphsage_best.json"

    if resume and best_result_path.is_file():
        logger.info("Resuming from %s", best_result_path)
        final_result = load_graphsage_result(best_result_path)
    else:
        if search_mode == "none":
            selected_config = _runtime_config(
                GraphSAGEConfig(schema_name=schema_name),
                schema_name=schema_name,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
            )
        else:
            configs = [
                _runtime_config(
                    GraphSAGEConfig(
                        hidden_dim=c.hidden_dim,
                        num_layers=c.num_layers,
                        dropout=c.dropout,
                        aggregator=c.aggregator,
                        fanout=c.fanout,
                        schema_name=schema_name,
                    ),
                    schema_name=schema_name,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                for c in iter_search_grid(search_mode)  # type: ignore[arg-type]
            ]
            selected_config, _ = _select_best_config(
                configs,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                graphs_dir=graphs_dir,
                search_fold=search_fold,
                device=device,
                tracker=tracker,
                run_id=run_id,
            )

        logger.info("Benchmarking selected config=%s on %s folds", selected_config.config_id, n_folds)
        final_result = run_graphsage_cv(
            selected_config,
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            graphs_dir=graphs_dir,
            n_folds=n_folds,
            device=device,
            tracker=tracker,
            run_id=run_id,
        )
        save_graphsage_result(final_result, output_dir)
        with best_result_path.open("w", encoding="utf-8") as handle:
            json.dump(final_result.to_dict(), handle, indent=2)

    g4_report = None
    if not skip_g4:
        g4_report = validate_gate_g4(final_result)
        logger.info(
            "Gate G4 PASSED: GraphSAGE AUPRC=%.4f > LR=%.4f",
            g4_report["graphsage_auprc_mean"],
            g4_report["baseline_auprc_mean"],
        )

    baseline_path = results_dir / "baselines" / "logistic_regression.json"
    significance = None
    if baseline_path.is_file() and len(final_result.per_fold_scores["auprc"]) >= 2:
        from hgad_cms.evaluation.cross_validation import load_baseline_result

        baseline_result = load_baseline_result(baseline_path)
        baseline_scores = baseline_result.per_fold_scores["auprc"]
        graphsage_scores = final_result.per_fold_scores["auprc"]
        if len(baseline_scores) == len(graphsage_scores):
            significance = paired_test(
                graphsage_scores,
                baseline_scores,
                metric="auprc",
                model_a=final_result.model_name,
                model_b="logistic_regression",
            ).to_dict()
            sig_path = output_dir / "significance_vs_lr.json"
            with sig_path.open("w", encoding="utf-8") as handle:
                json.dump(significance, handle, indent=2)

    if journal_path is not None:
        _append_journal_entry(
            final_result,
            journal_path=journal_path,
            run_id=run_id,
            g4_passed=g4_report["passed"] if g4_report else False,
        )

    manifest = {
        "run_id": run_id,
        "schema": schema_name,
        "best_config": final_result.config.to_dict(),
        "summary": final_result.summary,
        "gate_g4_passed": g4_report["passed"] if g4_report else None,
        "lr_baseline_auprc": LR_BASELINE_AUPRC,
        "significance_vs_lr": significance,
        "result_path": str((output_dir / f"{final_result.model_name}.json").resolve()),
    }
    manifest_path = output_dir / "graphsage_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)

    try:
        manifest = run_graphsage_benchmark(
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            graphs_dir=args.graphs_dir,
            results_dir=args.results_dir,
            schema_name=args.schema,
            n_folds=args.n_folds,
            search_mode=args.search_mode,
            search_fold=args.search_fold,
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            skip_g4=args.skip_g4,
            resume=args.resume,
            run_id=args.run_id,
            journal_path=args.journal_path,
        )
        logger.info("GraphSAGE benchmark complete. G4=%s", manifest.get("gate_g4_passed"))
        return 0
    except EvaluationError as exc:
        logger.error("Evaluation/G4 failed: %s", exc)
        return 2
    except (GNNError, HGADError) as exc:
        logger.error("GraphSAGE training failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
