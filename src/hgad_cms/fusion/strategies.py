"""Score normalization and fusion strategies."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    arr = np.asarray(scores, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


def rank_scores(scores: np.ndarray) -> np.ndarray:
    """Convert scores to average ranks in [0, 1] (higher = better)."""
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    if len(arr) <= 1:
        return ranks
    return ranks / (len(arr) - 1)


def weighted_average_fusion(
    score_map: dict[str, np.ndarray],
    *,
    keys: tuple[str, ...],
    weights: dict[str, float],
) -> np.ndarray:
    combined = np.zeros(len(next(iter(score_map.values()))), dtype=np.float64)
    total_w = 0.0
    for key in keys:
        w = weights.get(key, 0.0)
        if w <= 0 or key not in score_map:
            continue
        combined += w * normalize_scores(score_map[key])
        total_w += w
    if total_w <= 0:
        raise ValueError("weighted_average_fusion requires at least one positive weight")
    return combined / total_w


def optimize_weights_on_train(
    score_map: dict[str, np.ndarray],
    y_train: np.ndarray,
    *,
    keys: tuple[str, ...],
) -> dict[str, float]:
    """Grid-search non-negative weights maximizing train AUPRC."""
    grid = (0.0, 0.25, 0.5, 1.0)
    best_weights: dict[str, float] = {k: 1.0 for k in keys}
    best_auprc = -1.0
    for combo in product(grid, repeat=len(keys)):
        if sum(combo) == 0:
            continue
        weights = {k: float(w) for k, w in zip(keys, combo)}
        fused = weighted_average_fusion(score_map, keys=keys, weights=weights)
        auprc = average_precision_score(y_train, fused)
        if auprc > best_auprc:
            best_auprc = auprc
            best_weights = weights
    return best_weights


def logistic_stack_fusion(
    train_scores: dict[str, np.ndarray],
    y_train: np.ndarray,
    val_scores: dict[str, np.ndarray],
    *,
    keys: tuple[str, ...],
    holdout_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a logistic meta-learner on stratified train holdout predictions.

    Returns fused scores for validation and holdout (for diagnostics).
    """
    n = len(y_train)
    if n < 20 or holdout_fraction <= 0:
        X = np.column_stack([normalize_scores(train_scores[k]) for k in keys])
        meta = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)
        meta.fit(X, y_train)
        X_val = np.column_stack([normalize_scores(val_scores[k]) for k in keys])
        return meta.predict_proba(X_val)[:, 1], meta.predict_proba(X)[:, 1]

    idx = np.arange(n)
    fit_idx, hold_idx = train_test_split(
        idx,
        test_size=holdout_fraction,
        stratify=y_train,
        random_state=random_state,
    )
    hold_scores = {k: train_scores[k][hold_idx] for k in keys}
    y_hold = y_train[hold_idx]
    X_hold = np.column_stack([normalize_scores(hold_scores[k]) for k in keys])
    meta = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)
    meta.fit(X_hold, y_hold)

    X_val = np.column_stack([normalize_scores(val_scores[k]) for k in keys])
    return meta.predict_proba(X_val)[:, 1], meta.predict_proba(X_hold)[:, 1]


def rank_fusion(
    score_map: dict[str, np.ndarray],
    *,
    keys: tuple[str, ...],
) -> np.ndarray:
    """Average rank-normalized scores across models."""
    n = len(next(iter(score_map.values())))
    combined = np.zeros(n, dtype=np.float64)
    used = 0
    for key in keys:
        if key not in score_map:
            continue
        combined += rank_scores(score_map[key])
        used += 1
    if used == 0:
        raise ValueError("rank_fusion requires at least one score vector")
    return combined / used
