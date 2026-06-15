"""
Random forest baseline for provider fraud detection.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from hgad_cms.baselines.base import validate_training_arrays

logger = logging.getLogger(__name__)


class RandomForestBaseline:
    """Balanced random forest classifier."""

    name = "random_forest"

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_estimators: int = 300,
        max_depth: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestBaseline:
        validate_training_arrays(X_train, y_train)
        self._model.fit(X_train, y_train)
        logger.info("Fitted %s on %s samples", self.name, len(y_train))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1].astype(np.float64)
