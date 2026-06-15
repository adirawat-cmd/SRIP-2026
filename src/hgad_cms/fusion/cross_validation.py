"""
Provider-disjoint cross-validation for hybrid anomaly fusion (Phase 6).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hgad_cms.baselines.logistic_regression import LogisticRegressionBaseline
from hgad_cms.constants import OUTPUT_BENEFICIARIES, OUTPUT_CLAIMS, OUTPUT_PROVIDERS
from hgad_cms.evaluation.cross_validation import (
    build_tabular_features_for_fold,
    fit_feature_artifacts,
    _labels_for_providers,
)
from hgad_cms.evaluation.metrics import (
    PRIMARY_METRIC,
    aggregate_metric_values,
    compute_classification_metrics,
)
from hgad_cms.exceptions import GNNError
from hgad_cms.fusion.anomaly import IsolationForestScorer
from hgad_cms.fusion.config import FUSION_KEYS, FusionConfig, SCORE_KEYS
from hgad_cms.fusion.strategies import (
    logistic_stack_fusion,
    optimize_weights_on_train,
    rank_fusion,
    weighted_average_fusion,
)
from hgad_cms.graph.constants import DEFAULT_N_FOLDS
from hgad_cms.graph.io import load_hetero_graph
from hgad_cms.graph.splits import load_fold_split
from hgad_cms.graphsage.inference import extract_provider_embeddings as gs_embeddings
from hgad_cms.graphsage.inference import prepare_fold_graph_data as prepare_gs_fold
from hgad_cms.graphsage.trainer import GraphSAGETrainer
from hgad_cms.rgcn.inference import extract_provider_embeddings as rgcn_embeddings
from hgad_cms.rgcn.inference import prepare_fold_graph_data as prepare_rgcn_fold
from hgad_cms.rgcn.trainer import RGCNTrainer
from hgad_cms.tracking.experiment import ExperimentTracker

logger = logging.getLogger(__name__)

FUSION_DIR_NAME = "fusion"


@dataclass
class FusionFoldResult:
    fold_id: int
    metrics: dict[str, Any]
    val_provider_ids: list[str]
    val_labels: list[int]
    val_scores: dict[str, list[float]]
    fusion_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FusionCVResult:
    model_scores: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    per_fold_scores: dict[str, list[float]] = field(default_factory=dict)
    fold_results: list[FusionFoldResult] = field(default_factory=list)
    error_overlap: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": PRIMARY_METRIC,
            "model_summaries": self.model_scores,
            "per_fold_scores": self.per_fold_scores,
            "error_overlap": self.error_overlap,
            "folds": [
                {
                    "fold_id": fr.fold_id,
                    "metrics": fr.metrics,
                    "val_provider_ids": fr.val_provider_ids,
                    "val_labels": fr.val_labels,
                    "val_scores": fr.val_scores,
                    "fusion_weights": fr.fusion_weights,
                }
                for fr in self.fold_results
            ],
        }


def _load_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims = pd.read_parquet(processed_dir / OUTPUT_CLAIMS)
    providers = pd.read_parquet(processed_dir / OUTPUT_PROVIDERS)
    beneficiaries = pd.read_parquet(processed_dir / OUTPUT_BENEFICIARIES)
    return claims, providers, beneficiaries


def _try_catboost():
    try:
        from hgad_cms.baselines.catboost_model import CatBoostBaseline

        return CatBoostBaseline()
    except ImportError:
        logger.warning("CatBoost not installed; skipping catboost scores")
        return None


def _run_single_fusion_fold(
    fold_id: int,
    *,
    config: FusionConfig,
    claims: pd.DataFrame,
    providers: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    splits_dir: Path,
    graphs_dir: Path,
    device: str,
    tracker: ExperimentTracker | None,
    run_id: str,
) -> FusionFoldResult:
    fold = load_fold_split(splits_dir, fold_id)
    artifacts = fit_feature_artifacts(claims, beneficiaries, fold.train_provider_ids)
    _, X_train = build_tabular_features_for_fold(
        claims, beneficiaries, fold.train_provider_ids, artifacts
    )
    _, X_val = build_tabular_features_for_fold(
        claims, beneficiaries, fold.val_provider_ids, artifacts
    )
    y_train = _labels_for_providers(providers, fold.train_provider_ids)
    y_val = _labels_for_providers(providers, fold.val_provider_ids)

    val_scores: dict[str, np.ndarray] = {}
    train_scores: dict[str, np.ndarray] = {}

    # Supervised tabular
    lr = LogisticRegressionBaseline(random_state=config.seed)
    lr.fit(X_train, y_train)
    val_scores["logistic_regression"] = lr.predict_proba(X_val)
    train_scores["logistic_regression"] = lr.predict_proba(X_train)

    catboost = _try_catboost()
    if catboost is not None:
        catboost.fit(X_train, y_train)
        val_scores["catboost"] = catboost.predict_proba(X_val)
        train_scores["catboost"] = catboost.predict_proba(X_train)

    # Graph models + embeddings
    graph_dir = graphs_dir / config.schema_name / f"fold_{fold_id}"
    train_graph, manifest = load_hetero_graph(graph_dir)

    gs_fold = prepare_gs_fold(
        train_graph,
        manifest,
        val_provider_ids=fold.val_provider_ids,
        claims=claims,
        beneficiaries=beneficiaries,
        providers=providers,
        schema_name=config.schema_name,
    )
    gs_trainer = GraphSAGETrainer(config.graphsage_config, device=device, tracker=tracker)
    gs_trainer.train_fold(gs_fold, fold_id=fold_id, run_id=f"{run_id}_gs")
    if gs_trainer.last_model is None:
        raise GNNError("GraphSAGE trainer did not retain model")
    gs_model = gs_trainer.last_model
    from hgad_cms.graphsage.inference import predict_provider_scores as gs_predict

    val_scores["graphsage"] = gs_predict(
        gs_model, gs_fold.inference_data, gs_fold.val_provider_indices, device=gs_trainer.device
    )
    train_scores["graphsage"] = gs_predict(
        gs_model, gs_fold.train_data, gs_fold.train_provider_indices, device=gs_trainer.device
    )
    X_train_gs = gs_embeddings(
        gs_model, gs_fold.train_data, gs_fold.train_provider_indices, device=gs_trainer.device
    )
    X_val_gs = gs_embeddings(
        gs_model, gs_fold.inference_data, gs_fold.val_provider_indices, device=gs_trainer.device
    )

    rgcn_fold = prepare_rgcn_fold(
        train_graph,
        manifest,
        val_provider_ids=fold.val_provider_ids,
        claims=claims,
        beneficiaries=beneficiaries,
        providers=providers,
        schema_name=config.schema_name,
    )
    rgcn_trainer = RGCNTrainer(config.rgcn_config, device=device, tracker=tracker)
    rgcn_trainer.train_fold(rgcn_fold, fold_id=fold_id, run_id=f"{run_id}_rgcn")
    if rgcn_trainer.last_model is None:
        raise GNNError("R-GCN trainer did not retain model")
    rgcn_model = rgcn_trainer.last_model
    from hgad_cms.rgcn.inference import predict_provider_scores as rgcn_predict

    val_scores["rgcn"] = rgcn_predict(
        rgcn_model,
        rgcn_fold.inference_data,
        rgcn_fold.val_provider_indices,
        device=rgcn_trainer.device,
    )
    train_scores["rgcn"] = rgcn_predict(
        rgcn_model,
        rgcn_fold.train_data,
        rgcn_fold.train_provider_indices,
        device=rgcn_trainer.device,
    )
    X_train_rgcn = rgcn_embeddings(
        rgcn_model,
        rgcn_fold.train_data,
        rgcn_fold.train_provider_indices,
        device=rgcn_trainer.device,
    )
    X_val_rgcn = rgcn_embeddings(
        rgcn_model,
        rgcn_fold.inference_data,
        rgcn_fold.val_provider_indices,
        device=rgcn_trainer.device,
    )

    # Isolation Forest towers
    if_scorer = IsolationForestScorer(
        contamination=config.if_contamination,
        n_estimators=config.if_n_estimators,
        random_state=config.if_random_state,
    )
    if_scorer.fit(X_train)
    val_scores["if_tabular"] = if_scorer.score(X_val)
    train_scores["if_tabular"] = if_scorer.score(X_train)

    if_gs = IsolationForestScorer(
        contamination=config.if_contamination,
        n_estimators=config.if_n_estimators,
        random_state=config.if_random_state + 1,
    )
    if_gs.fit(X_train_gs)
    val_scores["if_graphsage"] = if_gs.score(X_val_gs)
    train_scores["if_graphsage"] = if_gs.score(X_train_gs)

    if_rgcn = IsolationForestScorer(
        contamination=config.if_contamination,
        n_estimators=config.if_n_estimators,
        random_state=config.if_random_state + 2,
    )
    if_rgcn.fit(X_train_rgcn)
    val_scores["if_rgcn"] = if_rgcn.score(X_val_rgcn)
    train_scores["if_rgcn"] = if_rgcn.score(X_train_rgcn)

    # Fusion strategies (all available score keys)
    fusion_keys = tuple(k for k in SCORE_KEYS if k in val_scores)
    weights = optimize_weights_on_train(train_scores, y_train, keys=fusion_keys)
    val_scores["fusion_weighted"] = weighted_average_fusion(
        val_scores, keys=fusion_keys, weights=weights
    )
    val_scores["fusion_stack_logistic"], _ = logistic_stack_fusion(
        train_scores,
        y_train,
        val_scores,
        keys=fusion_keys,
        holdout_fraction=config.stack_holdout_fraction,
        random_state=config.seed,
    )
    val_scores["fusion_rank"] = rank_fusion(val_scores, keys=fusion_keys)

    # Metrics per model
    all_models = list(val_scores.keys())
    metrics: dict[str, Any] = {}
    for name in all_models:
        m = compute_classification_metrics(y_val, val_scores[name])
        metrics[name] = m.to_dict()

    return FusionFoldResult(
        fold_id=fold_id,
        metrics=metrics,
        val_provider_ids=list(fold.val_provider_ids),
        val_labels=y_val.astype(int).tolist(),
        val_scores={k: v.astype(float).tolist() for k, v in val_scores.items()},
        fusion_weights=weights,
    )


def run_fusion_cv(
    config: FusionConfig,
    *,
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    n_folds: int = DEFAULT_N_FOLDS,
    device: str = "auto",
    tracker: ExperimentTracker | None = None,
    run_id: str = "fusion_v1",
    fold_ids: list[int] | None = None,
) -> FusionCVResult:
    """Run hybrid fusion evaluation across provider-disjoint CV folds."""
    claims, providers, beneficiaries = _load_tables(processed_dir)
    selected = fold_ids if fold_ids is not None else list(range(n_folds))
    fold_results: list[FusionFoldResult] = []

    for fold_id in selected:
        logger.info("Fusion fold=%s", fold_id)
        fold_results.append(
            _run_single_fusion_fold(
                fold_id,
                config=config,
                claims=claims,
                providers=providers,
                beneficiaries=beneficiaries,
                splits_dir=splits_dir,
                graphs_dir=graphs_dir,
                device=device,
                tracker=tracker,
                run_id=run_id,
            )
        )

    per_fold: dict[str, list[float]] = {}
    model_names = sorted({k for fr in fold_results for k in fr.metrics})
    for name in model_names:
        per_fold[name] = [float(fr.metrics[name]["auprc"]) for fr in fold_results]

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for name in model_names:
        values = {metric: [float(fr.metrics[name][metric]) for fr in fold_results] for metric in (
            "auprc",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "recall_at_k",
            "precision_at_k",
        )}
        summaries[name] = {m: aggregate_metric_values(values[m]) for m in values}

    from hgad_cms.fusion.evaluation import compute_error_overlap

    overlap = compute_error_overlap(fold_results)

    return FusionCVResult(
        model_scores=summaries,
        per_fold_scores=per_fold,
        fold_results=fold_results,
        error_overlap=overlap,
    )


def save_fusion_result(result: FusionCVResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **result.to_dict(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_fusion_result(path: Path) -> FusionCVResult:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    fold_results = [
        FusionFoldResult(
            fold_id=int(fr["fold_id"]),
            metrics=fr["metrics"],
            val_provider_ids=list(fr["val_provider_ids"]),
            val_labels=list(fr["val_labels"]),
            val_scores={k: list(v) for k, v in fr["val_scores"].items()},
            fusion_weights=fr.get("fusion_weights", {}),
        )
        for fr in payload.get("folds", [])
    ]
    return FusionCVResult(
        model_scores=payload.get("model_summaries", {}),
        per_fold_scores=payload.get("per_fold_scores", {}),
        fold_results=fold_results,
        error_overlap=payload.get("error_overlap", {}),
    )
