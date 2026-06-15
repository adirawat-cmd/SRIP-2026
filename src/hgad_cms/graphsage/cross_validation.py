"""
Provider-disjoint cross-validation orchestration for GraphSAGE.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hgad_cms.constants import OUTPUT_BENEFICIARIES, OUTPUT_CLAIMS, OUTPUT_PROVIDERS
from hgad_cms.evaluation.metrics import PRIMARY_METRIC, aggregate_metric_values
from hgad_cms.exceptions import EvaluationError, GNNError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME
from hgad_cms.graph.io import load_hetero_graph
from hgad_cms.graph.splits import load_fold_split
from hgad_cms.graphsage.config import GraphSAGEConfig, LR_BASELINE_AUPRC
from hgad_cms.graphsage.inference import prepare_fold_graph_data
from hgad_cms.graphsage.trainer import FoldTrainResult, GraphSAGETrainer
from hgad_cms.tracking.experiment import ExperimentTracker

logger = logging.getLogger(__name__)

GNN_DIR_NAME = "gnn"
RESULTS_DIR_NAME = "results"


@dataclass
class GraphSAGECVResult:
    """Aggregated GraphSAGE CV result."""

    config: GraphSAGEConfig
    fold_results: list[FoldTrainResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    per_fold_scores: dict[str, list[float]] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return f"graphsage_{self.config.config_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config.to_dict(),
            "summary": self.summary,
            "per_fold_scores": self.per_fold_scores,
            "folds": [
                {
                    "fold_id": fr.fold_id,
                    "metrics": fr.metrics.to_dict(),
                    "confusion_matrix": fr.confusion_matrix,
                    "best_epoch": fr.best_epoch,
                    "best_val_auprc": fr.best_val_auprc,
                    "history": fr.history,
                }
                for fr in self.fold_results
            ],
        }


def _load_processed_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_parquet(processed_dir / OUTPUT_CLAIMS)
    providers = pd.read_parquet(processed_dir / OUTPUT_PROVIDERS)
    beneficiaries = pd.read_parquet(processed_dir / OUTPUT_BENEFICIARIES)
    return claims, providers, beneficiaries


def run_graphsage_cv(
    config: GraphSAGEConfig,
    *,
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    n_folds: int = DEFAULT_N_FOLDS,
    device: str | None = "auto",
    tracker: ExperimentTracker | None = None,
    run_id: str = "graphsage_v1",
    fold_ids: list[int] | None = None,
) -> GraphSAGECVResult:
    """Run provider-disjoint GraphSAGE CV for a single configuration."""
    claims, providers, beneficiaries = _load_processed_tables(processed_dir)
    trainer = GraphSAGETrainer(config, device=device, tracker=tracker)
    fold_results: list[FoldTrainResult] = []
    selected_folds = fold_ids if fold_ids is not None else list(range(n_folds))

    for fold_id in selected_folds:
        fold = load_fold_split(splits_dir, fold_id)
        graph_dir = graphs_dir / config.schema_name / f"fold_{fold_id}"
        train_graph, manifest = load_hetero_graph(graph_dir)
        fold_data = prepare_fold_graph_data(
            train_graph,
            manifest,
            val_provider_ids=fold.val_provider_ids,
            claims=claims,
            beneficiaries=beneficiaries,
            providers=providers,
            schema_name=config.schema_name,
        )

        if tracker is not None:
            tracker.log_run_start(run_id=run_id, config=config, fold_id=fold_id)

        logger.info("GraphSAGE fold=%s config=%s", fold_id, config.config_id)
        result = trainer.train_fold(fold_data, fold_id=fold_id, run_id=run_id)
        fold_results.append(result)

        if tracker is not None:
            tracker.log_run_end(
                run_id=run_id,
                fold_id=fold_id,
                config=config,
                metrics={"auprc": result.metrics.auprc},
                best_epoch=result.best_epoch,
            )

    metric_names = [
        "auprc",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "recall_at_k",
        "precision_at_k",
    ]
    per_fold_scores: dict[str, list[float]] = {name: [] for name in metric_names}
    for fold_result in fold_results:
        payload = fold_result.metrics.to_dict()
        for name in metric_names:
            per_fold_scores[name].append(float(payload[name]))

    summary = {name: aggregate_metric_values(values) for name, values in per_fold_scores.items()}
    return GraphSAGECVResult(
        config=config,
        fold_results=fold_results,
        summary=summary,
        per_fold_scores=per_fold_scores,
    )


def save_graphsage_result(result: GraphSAGECVResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.model_name}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        **result.to_dict(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved GraphSAGE result: %s", path)
    return path


def load_graphsage_result(path: Path) -> GraphSAGECVResult:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    config = GraphSAGEConfig.from_dict(payload["config"])
    fold_results: list[FoldTrainResult] = []
    for fold in payload["folds"]:
        from hgad_cms.evaluation.metrics import ClassificationMetrics

        metrics_payload = fold["metrics"]
        fold_results.append(
            FoldTrainResult(
                fold_id=int(fold["fold_id"]),
                metrics=ClassificationMetrics(
                    auprc=float(metrics_payload["auprc"]),
                    f1=float(metrics_payload["f1"]),
                    precision=float(metrics_payload["precision"]),
                    recall=float(metrics_payload["recall"]),
                    roc_auc=float(metrics_payload["roc_auc"]),
                    recall_at_k=float(metrics_payload["recall_at_k"]),
                    precision_at_k=float(metrics_payload["precision_at_k"]),
                    k=int(metrics_payload["k"]),
                    threshold=float(metrics_payload.get("threshold", 0.5)),
                    n_samples=int(metrics_payload.get("n_samples", 0)),
                    n_positive=int(metrics_payload.get("n_positive", 0)),
                ),
                confusion_matrix=fold["confusion_matrix"],
                best_epoch=int(fold["best_epoch"]),
                best_val_auprc=float(fold["best_val_auprc"]),
                history=fold.get("history", []),
            )
        )
    return GraphSAGECVResult(
        config=config,
        fold_results=fold_results,
        summary=payload["summary"],
        per_fold_scores=payload["per_fold_scores"],
    )


def validate_gate_g4(
    result: GraphSAGECVResult,
    *,
    baseline_auprc: float = LR_BASELINE_AUPRC,
) -> dict[str, object]:
    """
    Gate G4: GraphSAGE beats the logistic regression baseline on mean AUPRC.

    Requirements:
    - Valid mean AUPRC (not NaN)
    - Mean AUPRC > LR baseline (0.6810)
    - Sanity: mean AUPRC > 0.05
    """
    import numpy as np

    auprc_mean = float(result.summary["auprc"]["mean"])
    if np.isnan(auprc_mean):
        raise EvaluationError("G4 failed: NaN GraphSAGE AUPRC")

    sanity_pass = auprc_mean > 0.05
    if not sanity_pass:
        raise EvaluationError(f"G4 sanity failed: auprc={auprc_mean:.4f}")

    beats_baseline = auprc_mean > baseline_auprc
    report = {
        "passed": beats_baseline,
        "graphsage_auprc_mean": auprc_mean,
        "graphsage_auprc_std": float(result.summary["auprc"]["std"]),
        "baseline_auprc_mean": baseline_auprc,
        "primary_metric": PRIMARY_METRIC,
        "config_id": result.config.config_id,
        "graph_structure_beats_tabular": beats_baseline,
    }
    if not beats_baseline:
        raise EvaluationError(
            f"G4 failed: GraphSAGE AUPRC {auprc_mean:.4f} <= LR baseline {baseline_auprc:.4f}"
        )
    return report
