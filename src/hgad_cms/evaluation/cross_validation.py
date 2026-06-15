"""
Provider-disjoint cross-validation for baseline models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hgad_cms.baselines.base import ProviderBaseline
from hgad_cms.baselines.centrality_features import (
    centrality_feature_matrix,
    compute_centrality_from_claims,
)
from hgad_cms.constants import (
    COL_FRAUD_LABEL,
    COL_PROVIDER,
    OUTPUT_BENEFICIARIES,
    OUTPUT_CLAIMS,
    OUTPUT_PROVIDERS,
)
from hgad_cms.evaluation.metrics import (
    PRIMARY_METRIC,
    ClassificationMetrics,
    aggregate_metric_values,
    compute_classification_metrics,
    compute_confusion_matrix,
)
from hgad_cms.exceptions import BaselineError, EvaluationError
from hgad_cms.graph.constants import DEFAULT_N_FOLDS, GRAPHS_DIR_NAME
from hgad_cms.graph.features import (
    assert_no_leakage_columns,
    compute_top_code_lists,
    select_feature_matrix,
)
from hgad_cms.graph.features import build_provider_features
from hgad_cms.graph.splits import FoldSplit, load_fold_split

logger = logging.getLogger(__name__)

RESULTS_DIR_NAME = "results"
BASELINES_DIR_NAME = "baselines"


@dataclass
class FoldResult:
    """Metrics and artifacts for a single CV fold."""

    fold_id: int
    metrics: ClassificationMetrics
    confusion_matrix: list[list[int]]
    n_train: int
    n_val: int


@dataclass
class BaselineCVResult:
    """Aggregated cross-validation result for one baseline."""

    model_name: str
    fold_results: list[FoldResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    per_fold_scores: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "summary": self.summary,
            "per_fold_scores": self.per_fold_scores,
            "folds": [
                {
                    "fold_id": fr.fold_id,
                    "metrics": fr.metrics.to_dict(),
                    "confusion_matrix": fr.confusion_matrix,
                    "n_train": fr.n_train,
                    "n_val": fr.n_val,
                }
                for fr in self.fold_results
            ],
        }


@dataclass
class FeatureArtifacts:
    """Fold-specific feature engineering artifacts fit on train only."""

    feature_columns: list[str]
    scaler: StandardScaler
    top_diagnosis_codes: list[str]
    top_procedure_codes: list[str]


def _load_processed_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_parquet(processed_dir / OUTPUT_CLAIMS)
    providers = pd.read_parquet(processed_dir / OUTPUT_PROVIDERS)
    beneficiaries = pd.read_parquet(processed_dir / OUTPUT_BENEFICIARIES)
    return claims, providers, beneficiaries


def build_tabular_features_for_fold(
    claims: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    provider_ids: list[str],
    artifacts: FeatureArtifacts,
) -> tuple[list[str], np.ndarray]:
    """Build tabular provider features using train-fitted vocabulary and scaler."""
    subset_claims = claims[claims[COL_PROVIDER].astype(str).isin(provider_ids)].copy()
    features = build_provider_features(
        subset_claims,
        beneficiaries,
        provider_ids,
        artifacts.top_diagnosis_codes,
        artifacts.top_procedure_codes,
    )
    feature_cols, matrix = select_feature_matrix(features, COL_PROVIDER)
    assert_no_leakage_columns(feature_cols)
    scaled = artifacts.scaler.transform(matrix)
    return feature_cols, scaled.astype(np.float32)


def fit_feature_artifacts(
    claims: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    train_provider_ids: list[str],
) -> FeatureArtifacts:
    """Fit top-code lists, feature columns, and scaler on train-fold data only."""
    train_claims = claims[claims[COL_PROVIDER].astype(str).isin(train_provider_ids)].copy()
    top_dx, top_proc = compute_top_code_lists(train_claims)
    train_features = build_provider_features(
        train_claims,
        beneficiaries,
        train_provider_ids,
        top_dx,
        top_proc,
    )
    feature_cols, train_matrix = select_feature_matrix(train_features, COL_PROVIDER)
    assert_no_leakage_columns(feature_cols)
    scaler = StandardScaler()
    scaler.fit(train_matrix)
    return FeatureArtifacts(
        feature_columns=feature_cols,
        scaler=scaler,
        top_diagnosis_codes=top_dx,
        top_procedure_codes=top_proc,
    )


def _labels_for_providers(providers: pd.DataFrame, provider_ids: list[str]) -> np.ndarray:
    frame = providers.copy()
    frame[COL_PROVIDER] = frame[COL_PROVIDER].astype(str)
    labels = frame.set_index(COL_PROVIDER).loc[provider_ids, COL_FRAUD_LABEL]
    return labels.astype(np.int64).to_numpy()


def run_single_fold(
    model: ProviderBaseline,
    fold: FoldSplit,
    claims: pd.DataFrame,
    providers: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    *,
    graphs_dir: Path,
    schema_name: str = "v1.1",
    use_centrality: bool = False,
) -> FoldResult:
    """Train and evaluate a baseline on one CV fold."""
    artifacts = fit_feature_artifacts(claims, beneficiaries, fold.train_provider_ids)

    _, X_train = build_tabular_features_for_fold(
        claims, beneficiaries, fold.train_provider_ids, artifacts
    )
    _, X_val = build_tabular_features_for_fold(
        claims, beneficiaries, fold.val_provider_ids, artifacts
    )

    if use_centrality:
        train_claims = claims[
            claims[COL_PROVIDER].astype(str).isin(fold.train_provider_ids)
        ]
        val_claims = claims[claims[COL_PROVIDER].astype(str).isin(fold.val_provider_ids)]
        cent_train = compute_centrality_from_claims(train_claims, fold.train_provider_ids)
        cent_val = compute_centrality_from_claims(val_claims, fold.val_provider_ids)
        _, cent_train_mat = centrality_feature_matrix(cent_train)
        _, cent_val_mat = centrality_feature_matrix(cent_val)
        X_train = np.hstack([X_train, cent_train_mat])
        X_val = np.hstack([X_val, cent_val_mat])

    y_train = _labels_for_providers(providers, fold.train_provider_ids)
    y_val = _labels_for_providers(providers, fold.val_provider_ids)

    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_val)
    metrics = compute_classification_metrics(y_val, y_score)
    cm = compute_confusion_matrix(y_val, y_score).tolist()

    return FoldResult(
        fold_id=fold.fold_id,
        metrics=metrics,
        confusion_matrix=cm,
        n_train=len(fold.train_provider_ids),
        n_val=len(fold.val_provider_ids),
    )


def run_provider_disjoint_cv(
    model_factory: Callable[[], ProviderBaseline],
    *,
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    schema_name: str = "v1.1",
    n_folds: int = DEFAULT_N_FOLDS,
    use_centrality: bool = False,
) -> BaselineCVResult:
    """Run provider-disjoint stratified CV for a baseline model."""
    claims, providers, beneficiaries = _load_processed_tables(processed_dir)
    model_name = model_factory().name
    fold_results: list[FoldResult] = []

    for fold_id in range(n_folds):
        fold = load_fold_split(splits_dir, fold_id)
        model = model_factory()
        logger.info("Running fold %s model=%s", fold_id, model_name)
        fold_results.append(
            run_single_fold(
                model,
                fold,
                claims,
                providers,
                beneficiaries,
                graphs_dir=graphs_dir,
                schema_name=schema_name,
                use_centrality=use_centrality,
            )
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
    per_fold_scores: dict[str, list[float]] = {m: [] for m in metric_names}
    for fold_result in fold_results:
        metric_dict = fold_result.metrics.to_dict()
        for metric in metric_names:
            per_fold_scores[metric].append(float(metric_dict[metric]))

    summary = {metric: aggregate_metric_values(scores) for metric, scores in per_fold_scores.items()}

    return BaselineCVResult(
        model_name=model_name,
        fold_results=fold_results,
        summary=summary,
        per_fold_scores=per_fold_scores,
    )


def save_baseline_result(result: BaselineCVResult, output_dir: Path) -> Path:
    """Persist baseline CV result JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.model_name}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        **result.to_dict(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved baseline result: %s", path)
    return path


def load_baseline_result(path: Path) -> BaselineCVResult:
    """Load a persisted baseline CV result JSON."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    fold_results: list[FoldResult] = []
    for fold in payload["folds"]:
        metrics_payload = fold["metrics"]
        fold_results.append(
            FoldResult(
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
                n_train=int(fold["n_train"]),
                n_val=int(fold["n_val"]),
            )
        )

    return BaselineCVResult(
        model_name=payload["model_name"],
        fold_results=fold_results,
        summary=payload["summary"],
        per_fold_scores=payload["per_fold_scores"],
    )


def build_comparison_table(results: list[BaselineCVResult]) -> pd.DataFrame:
    """Build mean +/- std comparison table across baselines."""
    rows: list[dict[str, object]] = []
    metrics = ["auprc", "f1", "precision", "recall", "roc_auc", "recall_at_k", "precision_at_k"]
    for result in results:
        row: dict[str, object] = {"model": result.model_name}
        for metric in metrics:
            stats = result.summary.get(metric, {"mean": 0.0, "std": 0.0})
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_formatted"] = f"{stats['mean']:.4f} +/- {stats['std']:.4f}"
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(by="auprc_mean", ascending=False)
    return table


def save_comparison_table(table: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_comparison.csv"
    json_path = output_dir / "baseline_comparison.json"
    table.to_csv(csv_path, index=False)
    table.to_json(json_path, orient="records", indent=2)
    return csv_path


def validate_gate_g3(results: list[BaselineCVResult]) -> dict[str, object]:
    """
    Gate G3: all baselines completed with valid primary metrics.

    Requirements:
    - At least 4 tabular baselines + rf_centrality present when full run requested
    - No NaN primary metrics
    - Every model mean AUPRC > 0.05 (sanity above chance)
    - Best model mean AUPRC >= dataset prevalence (~0.0935)
    """
    if len(results) < 4:
        raise EvaluationError(f"G3 requires at least 4 baselines, got {len(results)}")

    checks: list[dict[str, object]] = []
    best_auprc = 0.0
    for result in results:
        auprc_mean = float(result.summary["auprc"]["mean"])
        if np.isnan(auprc_mean):
            raise EvaluationError(f"NaN AUPRC for model {result.model_name}")
        sanity_pass = auprc_mean > 0.05
        checks.append(
            {
                "model": result.model_name,
                "auprc_mean": auprc_mean,
                "sanity_passed": sanity_pass,
            }
        )
        if not sanity_pass:
            raise EvaluationError(
                f"G3 sanity failed: {result.model_name} auprc={auprc_mean:.4f}"
            )
        best_auprc = max(best_auprc, auprc_mean)

    prevalence_floor = 0.0935
    benchmark_pass = best_auprc >= prevalence_floor
    report = {
        "passed": benchmark_pass,
        "checks": checks,
        "best_auprc_mean": best_auprc,
        "prevalence_floor": prevalence_floor,
        "primary_metric": PRIMARY_METRIC,
    }
    if not benchmark_pass:
        raise EvaluationError(
            f"G3 benchmark failed: best AUPRC {best_auprc:.4f} < prevalence {prevalence_floor}"
        )
    return report
