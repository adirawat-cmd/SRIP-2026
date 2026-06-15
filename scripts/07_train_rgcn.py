#!/usr/bin/env python3
"""
Train and evaluate R-GCN benchmark (Phase 5) with HPO, ablations, and diagnosis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from hgad_cms.evaluation.cross_validation import load_baseline_result
from hgad_cms.evaluation.significance import compare_all_pairs, paired_test
from hgad_cms.exceptions import EvaluationError, GNNError, HGADError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME, SPLITS_DIR_NAME
from hgad_cms.graphsage.config import LR_BASELINE_AUPRC
from hgad_cms.graphsage.cross_validation import load_graphsage_result
from hgad_cms.rgcn.config import (
    GRAPHSAGE_BENCHMARK_AUPRC,
    FeatureAblation,
    RelationAblation,
    RGCNConfig,
    iter_search_grid,
)
from hgad_cms.graphsage.evaluation import (
    load_epoch_curves,
    plot_training_curves,
)
from hgad_cms.tracking.research_docs import on_benchmark_complete, on_gate_result, sync_all
from hgad_cms.rgcn.cross_validation import (
    RGCN_DIR_NAME,
    RESULTS_DIR_NAME,
    RGCNCVResult,
    load_rgcn_result,
    run_rgcn_cv,
    save_rgcn_result,
    validate_gate_g5,
)
from hgad_cms.tracking.experiment import ExperimentTracker
from hgad_cms.tracking.logger import setup_logging

logger = logging.getLogger("hgad_cms.train_rgcn")

RELATION_ABLATIONS: tuple[RelationAblation, ...] = (
    "full",
    "no_pp",
    "no_treats",
    "no_bills",
)
FEATURE_ABLATIONS: tuple[FeatureAblation, ...] = ("full", "provider_only")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R-GCN benchmark (schema_v1.1)")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data") / SPLITS_DIR_NAME)
    parser.add_argument("--graphs-dir", type=Path, default=Path("artifacts") / GRAPHS_DIR_NAME)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts") / RESULTS_DIR_NAME)
    parser.add_argument("--schema", type=str, default="v1.1")
    parser.add_argument("--search-mode", choices=("eval", "full", "quick", "none"), default="eval")
    parser.add_argument("--search-fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-g5", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--run-id", type=str, default="rgcn_v1")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _runtime_config(base: RGCNConfig, args: argparse.Namespace) -> RGCNConfig:
    return RGCNConfig(
        hidden_dim=base.hidden_dim,
        num_layers=base.num_layers,
        dropout=base.dropout,
        num_bases=base.num_bases,
        fanout=base.fanout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        schema_name=args.schema,
        relation_ablation=base.relation_ablation,
        feature_ablation=base.feature_ablation,
    )


def _run_hpo(
    configs: list[RGCNConfig],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    tracker: ExperimentTracker,
) -> list[RGCNCVResult]:
    hpo_dir = output_dir / "hpo_search"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    results: list[RGCNCVResult] = []
    for config in configs:
        path = hpo_dir / f"rgcn_{config.config_id}.json"
        if args.resume and path.is_file():
            results.append(load_rgcn_result(path))
            continue
        result = run_rgcn_cv(
            config,
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            graphs_dir=args.graphs_dir,
            device=args.device,
            tracker=tracker,
            run_id=f"{args.run_id}_hpo",
            fold_ids=[args.search_fold],
        )
        save_rgcn_result(result, hpo_dir)
        results.append(result)
        tracker.log_search_result(
            run_id=args.run_id,
            config=config,
            mean_auprc=float(result.summary["auprc"]["mean"]),
            std_auprc=float(result.summary["auprc"]["std"]),
        )
    return results


def _write_diagnosis_report(
    *,
    path: Path,
    leaderboard: list[dict],
    benchmark: RGCNCVResult | None,
    ablation_results: dict[str, float],
    significance: list[dict],
    beats_lr: bool,
    beats_graphsage: bool,
) -> None:
    lines = [
        "# R-GCN Evaluation Diagnosis Report",
        "",
        "## Executive summary",
        "",
        f"- **LR baseline AUPRC:** {LR_BASELINE_AUPRC:.4f}",
        f"- **GraphSAGE benchmark AUPRC:** {GRAPHSAGE_BENCHMARK_AUPRC:.4f}",
    ]
    if benchmark:
        s = benchmark.summary
        lines.append(f"- **Best R-GCN (5-fold):** {s['auprc']['mean']:.4f} ± {s['auprc']['std']:.4f}")
        lines.append(f"- **Gate G5 (beat LR):** {'PASSED' if beats_lr else 'FAILED'}")
        lines.append(f"- **Beat GraphSAGE:** {'YES' if beats_graphsage else 'NO'}")
    lines.extend(["", "## Top configurations (HPO fold 0)", ""])
    lines.append("| Rank | Config | AUPRC | ROC-AUC | Prec | Recall | F1 |")
    lines.append("|------|--------|-------|---------|------|--------|-----|")
    for row in leaderboard[:10]:
        lines.append(
            f"| {row['rank']} | `{row['config_id']}` | {row['auprc']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    if ablation_results:
        lines.extend(["", "## Relation & feature ablations (fold 0)", ""])
        lines.append("| Ablation | AUPRC |")
        lines.append("|----------|-------|")
        for name, auprc in sorted(ablation_results.items(), key=lambda x: -x[1]):
            lines.append(f"| {name} | {auprc:.4f} |")
    if significance:
        lines.extend(["", "## Statistical significance (paired Wilcoxon)", ""])
        for item in significance:
            lines.append(
                f"- **{item['model_a']} vs {item['model_b']}**: "
                f"p={item['p_value']:.4f}, mean_diff={item['mean_diff']:.4f}, "
                f"significant={item['significant_at_0_05']}"
            )
    if not beats_lr:
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                "R-GCN did not exceed the LR baseline. Relation-specific convolutions alone "
                "may be insufficient; consider fusion models or richer edge features.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)
    output_dir = args.results_dir / RGCN_DIR_NAME
    tracker = ExperimentTracker(output_dir / "experiments.jsonl")

    try:
        if args.search_mode == "none":
            best_config = RGCNConfig(schema_name=args.schema)
            hpo_results = []
        else:
            configs = [_runtime_config(c, args) for c in iter_search_grid(args.search_mode)]  # type: ignore[arg-type]
            logger.info("HPO grid: %s configs", len(configs))
            hpo_results = _run_hpo(configs, args=args, output_dir=output_dir, tracker=tracker)

        if hpo_results:
            ranked = sorted(
                hpo_results,
                key=lambda r: float(r.summary["auprc"]["mean"]),
                reverse=True,
            )
            best_config = ranked[0].config
            leaderboard = []
            for rank, result in enumerate(ranked[:10], start=1):
                s = result.summary
                leaderboard.append(
                    {
                        "rank": rank,
                        "config_id": result.config.config_id,
                        "auprc": float(s["auprc"]["mean"]),
                        "roc_auc": float(s["roc_auc"]["mean"]),
                        "precision": float(s["precision"]["mean"]),
                        "recall": float(s["recall"]["mean"]),
                        "f1": float(s["f1"]["mean"]),
                    }
                )
        else:
            best_config = _runtime_config(best_config, args)
            leaderboard = []

        benchmark_path = output_dir / "rgcn_benchmark.json"
        if args.resume and benchmark_path.is_file():
            benchmark = load_rgcn_result(benchmark_path)
        else:
            best_config = _runtime_config(best_config, args)
            benchmark = run_rgcn_cv(
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

        ablation_results: dict[str, float] = {}
        if not args.skip_ablations:
            base_kwargs = {
                "hidden_dim": best_config.hidden_dim,
                "num_layers": best_config.num_layers,
                "dropout": best_config.dropout,
                "schema_name": args.schema,
            }
            for rel in RELATION_ABLATIONS:
                cfg = _runtime_config(
                    RGCNConfig(**base_kwargs, relation_ablation=rel),
                    args,
                )
                abl_dir = output_dir / "ablations"
                abl_dir.mkdir(parents=True, exist_ok=True)
                path = abl_dir / f"rgcn_{cfg.config_id}.json"
                if args.resume and path.is_file():
                    ablation_results[f"relation:{rel}"] = float(
                        load_rgcn_result(path).summary["auprc"]["mean"]
                    )
                    continue
                result = run_rgcn_cv(
                    cfg,
                    processed_dir=args.processed_dir,
                    splits_dir=args.splits_dir,
                    graphs_dir=args.graphs_dir,
                    device=args.device,
                    fold_ids=[args.search_fold],
                    run_id=f"{args.run_id}_abl",
                )
                save_rgcn_result(result, abl_dir)
                ablation_results[f"relation:{rel}"] = float(result.summary["auprc"]["mean"])

            for feat in FEATURE_ABLATIONS:
                if feat == "full":
                    continue
                cfg = _runtime_config(
                    RGCNConfig(**base_kwargs, feature_ablation=feat),
                    args,
                )
                result = run_rgcn_cv(
                    cfg,
                    processed_dir=args.processed_dir,
                    splits_dir=args.splits_dir,
                    graphs_dir=args.graphs_dir,
                    device=args.device,
                    fold_ids=[args.search_fold],
                    run_id=f"{args.run_id}_abl",
                )
                ablation_results[f"feature:{feat}"] = float(result.summary["auprc"]["mean"])

        # Significance vs LR, CatBoost, GraphSAGE
        comparisons: dict[str, list[float]] = {benchmark.model_name: benchmark.per_fold_scores["auprc"]}
        baseline_dir = args.results_dir / "baselines"
        if (baseline_dir / "logistic_regression.json").is_file():
            comparisons["logistic_regression"] = load_baseline_result(
                baseline_dir / "logistic_regression.json"
            ).per_fold_scores["auprc"]
        if (baseline_dir / "catboost.json").is_file():
            comparisons["catboost"] = load_baseline_result(
                baseline_dir / "catboost.json"
            ).per_fold_scores["auprc"]
        gs_path = args.results_dir / "gnn" / "graphsage_benchmark.json"
        if gs_path.is_file():
            comparisons["graphsage"] = load_graphsage_result(gs_path).per_fold_scores["auprc"]

        significance = [
            item.to_dict()
            for item in compare_all_pairs(comparisons, metric="auprc")
            if benchmark.model_name in (item.model_a, item.model_b)
        ]
        sig_path = output_dir / "significance_auprc.json"
        with sig_path.open("w", encoding="utf-8") as handle:
            json.dump(significance, handle, indent=2)

        beats_lr = float(benchmark.summary["auprc"]["mean"]) > LR_BASELINE_AUPRC
        beats_graphsage = float(benchmark.summary["auprc"]["mean"]) > GRAPHSAGE_BENCHMARK_AUPRC
        g5 = None
        try:
            if not args.skip_g5:
                g5 = validate_gate_g5(benchmark)
        except EvaluationError:
            g5 = {"passed": False}

        report_path = Path("docs") / "rgcn_diagnosis.md"
        _write_diagnosis_report(
            path=report_path,
            leaderboard=leaderboard,
            benchmark=benchmark,
            ablation_results=ablation_results,
            significance=significance,
            beats_lr=beats_lr,
            beats_graphsage=beats_graphsage,
        )

        summary = {
            "best_config": best_config.config_id,
            "benchmark_auprc": float(benchmark.summary["auprc"]["mean"]),
            "beats_lr": beats_lr,
            "beats_graphsage": beats_graphsage,
            "gate_g5_passed": g5["passed"] if g5 else None,
            "report": str(report_path.resolve()),
        }
        with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        plot_paths: list[Path] = []
        if not args.skip_plots and hpo_results:
            plots_dir = output_dir / "plots"
            experiments_path = output_dir / "experiments.jsonl"
            curves = load_epoch_curves(experiments_path)
            top_ids = [row["config_id"] for row in leaderboard[:5]]
            plot_paths = plot_training_curves(curves, config_ids=top_ids, output_dir=plots_dir)

        logger.info(
            "R-GCN complete. AUPRC=%.4f G5=%s",
            benchmark.summary["auprc"]["mean"],
            summary["gate_g5_passed"],
        )
        on_benchmark_complete(
            experiment_id=args.run_id,
            phase="Phase 5 — R-GCN",
            model_name=best_config.config_id,
            auprc_mean=float(benchmark.summary["auprc"]["mean"]),
            auprc_std=float(benchmark.summary["auprc"]["std"]),
            beats_lr=beats_lr,
        )
        if g5 is not None:
            on_gate_result(
                "G5",
                passed=bool(g5.get("passed")),
                experiment_id=args.run_id,
                details=str(report_path.resolve()),
            )
        sync_all(results_dir=output_dir)
        return 0 if (g5 and g5["passed"]) or args.skip_g5 else 2
    except EvaluationError as exc:
        logger.error("G5/evaluation failed: %s", exc)
        return 2
    except (GNNError, HGADError) as exc:
        logger.error("R-GCN failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
