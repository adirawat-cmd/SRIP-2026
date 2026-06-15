"""
Random forest on tabular + graph centrality features.
"""

from __future__ import annotations

import logging

import numpy as np

from hgad_cms.baselines.random_forest import RandomForestBaseline

logger = logging.getLogger(__name__)


class RFCentralityBaseline(RandomForestBaseline):
    """Random forest trained on concatenated tabular and centrality features."""

    name = "rf_centrality"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> RFCentralityBaseline:
        super().fit(X_train, y_train)
        logger.info("Fitted %s with %s features", self.name, X_train.shape[1])
        return self
