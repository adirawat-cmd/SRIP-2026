"""
Evaluation metrics for provider-level fraud classification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC: str = "auprc"
DEFAULT_RECALL_AT_K: int = 506


@dataclass
class ClassificationMetrics:
    """Container for provider-level classification metrics."""

    auprc: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    recall_at_k: float
    precision_at_k: float
    k: int
    threshold: float = 0.5
    n_samples: int = 0
    n_positive: int = 0
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        payload = asdict(self)
        payload.pop("extras", None)
        payload.update(self.extras)
        return payload


def _safe_metric(fn, *args, default: float = 0.0, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return default


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Recall@K: fraction of positives captured in the top-K scored providers.
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(-y_score)
    top_k = y_true[order[:k]]
    return float(top_k.sum()) / float(positives)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision@K: precision among the top-K scored providers."""
    if len(y_true) == 0 or k <= 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(-y_score)
    top_k = y_true[order[:k]]
    return float(top_k.sum()) / float(k)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    k: int | None = None,
) -> ClassificationMetrics:
    """
    Compute primary and secondary metrics for binary provider fraud classification.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Predicted fraud probabilities or scores.
    threshold:
        Decision threshold for hard-label metrics.
    k:
        Cutoff for Recall@K / Precision@K. Defaults to number of positives.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    n_positive = int(y_true.sum())
    k_value = n_positive if k is None else k
    y_pred = (y_score >= threshold).astype(np.int64)

    return ClassificationMetrics(
        auprc=_safe_metric(average_precision_score, y_true, y_score),
        f1=_safe_metric(f1_score, y_true, y_pred),
        precision=_safe_metric(precision_score, y_true, y_pred, zero_division=0),
        recall=_safe_metric(recall_score, y_true, y_pred, zero_division=0),
        roc_auc=_safe_metric(roc_auc_score, y_true, y_score),
        recall_at_k=recall_at_k(y_true, y_score, k_value),
        precision_at_k=precision_at_k(y_true, y_score, k_value),
        k=k_value,
        threshold=threshold,
        n_samples=int(len(y_true)),
        n_positive=n_positive,
    )


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return sklearn confusion matrix [[tn, fp], [fn, tp]]."""
    y_pred = (np.asarray(y_score) >= threshold).astype(np.int64)
    return confusion_matrix(np.asarray(y_true), y_pred, labels=[0, 1])


def aggregate_metric_values(values: list[float]) -> dict[str, float]:
    """Compute mean and std for a list of per-fold metric values."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_folds": float(len(arr)),
    }
