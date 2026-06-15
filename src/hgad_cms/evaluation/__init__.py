"""Evaluation package."""

from hgad_cms.evaluation.cross_validation import (
    BaselineCVResult,
    build_comparison_table,
    run_provider_disjoint_cv,
    save_baseline_result,
    save_comparison_table,
    validate_gate_g3,
)
from hgad_cms.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from hgad_cms.evaluation.significance import PairedTestResult, paired_test

__all__ = [
    "BaselineCVResult",
    "ClassificationMetrics",
    "PairedTestResult",
    "build_comparison_table",
    "compute_classification_metrics",
    "paired_test",
    "run_provider_disjoint_cv",
    "save_baseline_result",
    "save_comparison_table",
    "validate_gate_g3",
]
