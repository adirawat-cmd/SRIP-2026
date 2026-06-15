#!/usr/bin/env python3
"""
Phase 6 — Hybrid anomaly fusion benchmark.

Trains Isolation Forest on tabular / GraphSAGE / R-GCN embeddings, fuses scores
with supervised models, and compares against LR, CatBoost, GraphSAGE, and R-GCN.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hgad_cms.evaluation.significance import compare_all_pairs
from hgad_cms.exceptions import FusionError, HGADError
from hgad_cms.fusion.config import FUSION_DIR_NAME, FusionConfig, load_best_configs_from_artifacts
from hgad_cms.fusion.cross_validation import run_fusion_cv, save_fusion_result
from hgad_cms.fusion.evaluation import write_fusion_diagnosis
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME, SPLITS_DIR_NAME
from hgad_cms.graphsage.config import LR_BASELINE_AUPRC
from hgad_cms.tracking.experiment import ExperimentTracker
from hgad_cms.tracking.logger import setup_logging
from hgad_cms.tracking.research_docs import on_benchmark_complete, sync_all

logger = logging.getLogger("hgad_cms.train_fusion")

RESULTS_DIR_NAME = "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid anomaly fusion benchmark (Phase 6)")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data") / SPLITS_DIR_NAME)
    parser.add_argument("--graphs-dir", type=Path, default=Path("artifacts") / GRAPHS_DIR_NAME)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts") / RESULTS_DIR_NAME)
    parser.add_argument("--schema", type=str, default="v1.1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--fold-ids", type=int, nargs="*", default=None)
    parser.add_argument("--run-id", type=str, default="fusion_v1")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_level=args.log_level)
    output_dir = args.results_dir / FUSION_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(output_dir / "experiments.jsonl")

    try:
        gs_cfg, rg_cfg = load_best_configs_from_artifacts(args.results_dir)
        config = FusionConfig(
            schema_name=args.schema,
            graphsage_config=gs_cfg,
            rgcn_config=rg_cfg,
        )
        logger.info(
            "Fusion configs: GraphSAGE=%s R-GCN=%s",
            gs_cfg.config_id,
            rg_cfg.config_id,
        )

        result = run_fusion_cv(
            config,
            processed_dir=args.processed_dir,
            splits_dir=args.splits_dir,
            graphs_dir=args.graphs_dir,
            n_folds=args.n_folds,
            device=args.device,
            tracker=tracker,
            run_id=args.run_id,
            fold_ids=args.fold_ids,
        )

        benchmark_path = output_dir / "fusion_benchmark.json"
        save_fusion_result(result, benchmark_path)

        comparisons = dict(result.per_fold_scores)
        if "fusion_weighted" in comparisons:
            fusion_name = "fusion_weighted"
        else:
            fusion_name = next(iter(comparisons))
        significance = [
            item.to_dict()
            for item in compare_all_pairs(comparisons, metric="auprc")
            if fusion_name in (item.model_a, item.model_b)
        ]
        sig_path = output_dir / "significance_auprc.json"
        with sig_path.open("w", encoding="utf-8") as handle:
            json.dump(significance, handle, indent=2)

        report_path = Path("docs") / "fusion_diagnosis.md"
        write_fusion_diagnosis(path=report_path, result=result, significance=significance)

        best_name = max(
            result.model_scores,
            key=lambda k: float(result.model_scores[k]["auprc"]["mean"]),
        )
        best_auprc = float(result.model_scores[best_name]["auprc"]["mean"])
        best_std = float(result.model_scores[best_name]["auprc"]["std"])
        beats_lr = best_auprc > LR_BASELINE_AUPRC

        summary = {
            "best_model": best_name,
            "benchmark_auprc": best_auprc,
            "benchmark_auprc_std": best_std,
            "beats_lr": beats_lr,
            "report": str(report_path.resolve()),
        }
        with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        logger.info(
            "Fusion complete. Best=%s AUPRC=%.4f beats_lr=%s",
            best_name,
            best_auprc,
            beats_lr,
        )

        on_benchmark_complete(
            experiment_id=args.run_id,
            phase="Phase 6 — Hybrid Fusion",
            model_name=best_name,
            auprc_mean=best_auprc,
            auprc_std=best_std,
            beats_lr=beats_lr,
        )
        sync_all(results_dir=args.results_dir)
        return 0 if beats_lr else 2
    except (FusionError, HGADError) as exc:
        logger.error("Fusion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
