"""
Logistic regression baseline for provider fraud detection.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hgad_cms.baselines.base import validate_training_arrays

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """Standardized logistic regression with class balancing."""

    name = "logistic_regression"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = 2000,
        C: float = 1.0,
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self._pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=max_iter,
                        C=C,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegressionBaseline:
        validate_training_arrays(X_train, y_train)
        self._pipeline.fit(X_train, y_train)
        logger.info("Fitted %s on %s samples", self.name, len(y_train))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)
        return proba[:, 1].astype(np.float64)
