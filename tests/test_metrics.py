"""Tests for hgad_cms.evaluation.metrics."""

import numpy as np

from hgad_cms.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    recall_at_k,
)


def test_compute_classification_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
    assert 0.0 <= metrics.auprc <= 1.0
    assert metrics.n_samples == 4
    assert metrics.n_positive == 2


def test_recall_at_k_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    assert recall_at_k(y_true, y_score, k=2) == 1.0


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.2, 0.8, 0.7, 0.3])
    cm = compute_confusion_matrix(y_true, y_score)
    assert cm.shape == (2, 2)
