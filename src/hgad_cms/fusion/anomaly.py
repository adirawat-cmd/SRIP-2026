"""Isolation Forest anomaly scoring for tabular and embedding features."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from hgad_cms.fusion.strategies import normalize_scores


class IsolationForestScorer:
    """
    Unsupervised anomaly scorer where higher scores indicate greater fraud likelihood.

    sklearn ``score_samples`` returns higher values for inliers; we negate and
    min-max normalize so that outliers (potential fraud) receive higher scores.
    """

    def __init__(
        self,
        *,
        contamination: float | str = "auto",
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: IsolationForest | None = None

    def fit(self, X_train: np.ndarray) -> IsolationForestScorer:
        if X_train.ndim != 2 or len(X_train) < 10:
            raise ValueError("IsolationForest requires at least 10 training samples")
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsolationForestScorer must be fit before scoring")
        raw = -self._model.score_samples(X)
        return normalize_scores(raw)
