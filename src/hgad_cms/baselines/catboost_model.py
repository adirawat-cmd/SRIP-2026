"""
CatBoost baseline for provider fraud detection.
"""

from __future__ import annotations

import logging

import numpy as np

from hgad_cms.baselines.base import validate_training_arrays
from hgad_cms.exceptions import BaselineError

logger = logging.getLogger(__name__)


class CatBoostBaseline:
    """CatBoost classifier with balanced class weights."""

    name = "catboost"

    def __init__(
        self,
        *,
        random_state: int = 42,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> None:
        self.random_state = random_state
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> CatBoostBaseline:
        validate_training_arrays(X_train, y_train)
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise BaselineError(
                "CatBoost is not installed. Install with: pip install -e '.[models]'"
            ) from exc

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            auto_class_weights="Balanced",
            verbose=self.verbose,
            allow_writing_files=False,
        )
        self._model.fit(X_train, y_train)
        logger.info("Fitted %s on %s samples", self.name, len(y_train))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise BaselineError("CatBoostBaseline must be fitted before predict_proba")
        return self._model.predict_proba(X)[:, 1].astype(np.float64)
