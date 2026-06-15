"""
Baseline model protocol and shared utilities.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ProviderBaseline(Protocol):
    """Common interface for provider-level baseline classifiers."""

    name: str

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ProviderBaseline: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def validate_training_arrays(X: np.ndarray, y: np.ndarray) -> None:
    if len(X) != len(y):
        raise ValueError("Feature matrix and labels must have equal length")
    if len(np.unique(y)) < 2:
        raise ValueError("Training labels must contain both classes")
