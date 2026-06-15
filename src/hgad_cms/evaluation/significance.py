"""
Statistical significance utilities for paired model comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired statistical test on per-fold metric scores."""

    metric: str
    model_a: str
    model_b: str
    test: str
    statistic: float
    p_value: float
    mean_diff: float
    n_pairs: int
    significant_at_0_05: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_diff": self.mean_diff,
            "n_pairs": self.n_pairs,
            "significant_at_0_05": self.significant_at_0_05,
        }


def paired_test(
    scores_a: list[float],
    scores_b: list[float],
    *,
    metric: str,
    model_a: str,
    model_b: str,
    test: Literal["wilcoxon", "ttest"] = "wilcoxon",
) -> PairedTestResult:
    """
    Compare two models using paired per-fold metric scores.

    Parameters
    ----------
    scores_a, scores_b:
        Equal-length lists of per-fold scores for the same folds.
    test:
        ``wilcoxon`` (default, non-parametric) or ``ttest`` (paired t-test).
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Paired score vectors must have equal length")
    if len(a) < 2:
        raise ValueError("At least two paired folds are required")

    diff = a - b
    if test == "wilcoxon":
        stat, p_value = stats.wilcoxon(a, b, zero_method="wilcox")
        test_name = "wilcoxon_signed_rank"
    elif test == "ttest":
        stat, p_value = stats.ttest_rel(a, b)
        test_name = "paired_ttest"
    else:
        raise ValueError(f"Unsupported test: {test}")

    return PairedTestResult(
        metric=metric,
        model_a=model_a,
        model_b=model_b,
        test=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        mean_diff=float(diff.mean()),
        n_pairs=int(len(a)),
        significant_at_0_05=bool(p_value < 0.05),
    )


def compare_all_pairs(
    fold_scores: dict[str, list[float]],
    *,
    metric: str,
    test: Literal["wilcoxon", "ttest"] = "wilcoxon",
) -> list[PairedTestResult]:
    """Run paired tests for every unordered model pair."""
    models = sorted(fold_scores.keys())
    results: list[PairedTestResult] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            results.append(
                paired_test(
                    fold_scores[model_a],
                    fold_scores[model_b],
                    metric=metric,
                    model_a=model_a,
                    model_b=model_b,
                    test=test,
                )
            )
    return results
