"""Baseline model package."""

from hgad_cms.baselines.catboost_model import CatBoostBaseline
from hgad_cms.baselines.logistic_regression import LogisticRegressionBaseline
from hgad_cms.baselines.random_forest import RandomForestBaseline
from hgad_cms.baselines.rf_centrality import RFCentralityBaseline

BASELINE_REGISTRY: dict[str, type] = {
    LogisticRegressionBaseline.name: LogisticRegressionBaseline,
    RandomForestBaseline.name: RandomForestBaseline,
    CatBoostBaseline.name: CatBoostBaseline,
    RFCentralityBaseline.name: RFCentralityBaseline,
}

__all__ = [
    "BASELINE_REGISTRY",
    "CatBoostBaseline",
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "RFCentralityBaseline",
]
