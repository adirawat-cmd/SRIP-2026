"""Tests for hybrid fusion strategies and anomaly scoring."""

from __future__ import annotations

import numpy as np
import pytest

from hgad_cms.fusion.anomaly import IsolationForestScorer
from hgad_cms.fusion.strategies import (
    normalize_scores,
    rank_fusion,
    rank_scores,
    weighted_average_fusion,
)


def test_normalize_scores_range():
    raw = np.array([1.0, 2.0, 5.0])
    out = normalize_scores(raw)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_rank_scores_monotonic():
    scores = np.array([0.1, 0.9, 0.5])
    ranks = rank_scores(scores)
    assert ranks[1] > ranks[2] > ranks[0]


def test_weighted_average_fusion():
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([1.0, 0.5, 0.0])
    fused = weighted_average_fusion(
        {"a": a, "b": b},
        keys=("a", "b"),
        weights={"a": 1.0, "b": 1.0},
    )
    assert fused.shape == (3,)
    assert fused[1] == pytest.approx(0.5)


def test_rank_fusion():
    fused = rank_fusion(
        {"a": np.array([0.1, 0.9]), "b": np.array([0.2, 0.8])},
        keys=("a", "b"),
    )
    assert len(fused) == 2


def test_isolation_forest_scorer():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 8))
    X_val = rng.normal(size=(50, 8))
    scorer = IsolationForestScorer(n_estimators=50, random_state=42)
    scorer.fit(X_train)
    scores = scorer.score(X_val)
    assert scores.shape == (50,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
