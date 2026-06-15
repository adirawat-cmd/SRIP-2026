"""Fusion evaluation, error overlap, and diagnosis reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hgad_cms.fusion.cross_validation import FusionCVResult, FusionFoldResult
from hgad_cms.graphsage.config import LR_BASELINE_AUPRC
from hgad_cms.rgcn.config import GRAPHSAGE_BENCHMARK_AUPRC


def _top_k_mask(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    k = max(int(y_true.sum()), 1)
    k = min(k, len(y_true))
    order = np.argsort(-scores)
    mask = np.zeros(len(y_true), dtype=bool)
    mask[order[:k]] = True
    return mask


def compute_error_overlap(fold_results: list[FusionFoldResult]) -> dict[str, Any]:
    """
    Analyze whether anomaly scorers catch fraud providers missed by LR or R-GCN.

    A provider is "caught" if it appears in the top-K scored set (K = # fraud in fold).
    """
    lr_missed_if_caught = 0
    lr_missed_rgcn_caught = 0
    lr_missed_if_tab_caught = 0
    lr_missed_if_gs_caught = 0
    lr_missed_if_rgcn_caught = 0
    lr_missed_fusion_caught = 0
    total_fraud = 0
    lr_missed_fraud = 0

    per_fold: list[dict[str, Any]] = []

    for fr in fold_results:
        y = np.asarray(fr.val_labels, dtype=np.int64)
        fraud_mask = y == 1
        n_fraud = int(fraud_mask.sum())
        if n_fraud == 0:
            continue
        total_fraud += n_fraud
        scores = {k: np.asarray(v, dtype=np.float64) for k, v in fr.val_scores.items()}

        lr_top = _top_k_mask(y, scores["logistic_regression"])
        rgcn_top = _top_k_mask(y, scores.get("rgcn", scores["logistic_regression"]))
        if_top = _top_k_mask(y, scores.get("if_tabular", scores["logistic_regression"]))
        if_gs_top = _top_k_mask(y, scores.get("if_graphsage", scores["logistic_regression"]))
        if_rgcn_top = _top_k_mask(y, scores.get("if_rgcn", scores["logistic_regression"]))
        fusion_top = _top_k_mask(y, scores.get("fusion_weighted", scores["logistic_regression"]))

        lr_miss = fraud_mask & ~lr_top
        lr_missed_fraud += int(lr_miss.sum())
        fold_lr_missed = int(lr_miss.sum())

        fold_stats = {
            "fold_id": fr.fold_id,
            "n_fraud": n_fraud,
            "lr_missed_fraud": fold_lr_missed,
            "if_tabular_caught_lr_miss": int((lr_miss & if_top).sum()),
            "if_graphsage_caught_lr_miss": int((lr_miss & if_gs_top).sum()),
            "if_rgcn_caught_lr_miss": int((lr_miss & if_rgcn_top).sum()),
            "rgcn_caught_lr_miss": int((lr_miss & rgcn_top).sum()),
            "fusion_weighted_caught_lr_miss": int((lr_miss & fusion_top).sum()),
        }
        per_fold.append(fold_stats)

        lr_missed_if_tab_caught += fold_stats["if_tabular_caught_lr_miss"]
        lr_missed_if_gs_caught += fold_stats["if_graphsage_caught_lr_miss"]
        lr_missed_if_rgcn_caught += fold_stats["if_rgcn_caught_lr_miss"]
        lr_missed_rgcn_caught += fold_stats["rgcn_caught_lr_miss"]
        lr_missed_fusion_caught += fold_stats["fusion_weighted_caught_lr_miss"]
        lr_missed_if_caught += fold_stats["if_tabular_caught_lr_miss"]

    return {
        "total_fraud_providers": total_fraud,
        "lr_missed_fraud_total": lr_missed_fraud,
        "lr_missed_caught_by_if_tabular": lr_missed_if_tab_caught,
        "lr_missed_caught_by_if_graphsage": lr_missed_if_gs_caught,
        "lr_missed_caught_by_if_rgcn": lr_missed_if_rgcn_caught,
        "lr_missed_caught_by_rgcn": lr_missed_rgcn_caught,
        "lr_missed_caught_by_fusion_weighted": lr_missed_fusion_caught,
        "per_fold": per_fold,
    }


def write_fusion_diagnosis(
    *,
    path: Path,
    result: FusionCVResult,
    significance: list[dict[str, Any]],
    lr_baseline: float = LR_BASELINE_AUPRC,
) -> None:
    """Write fusion diagnosis markdown report."""
    lines = [
        "# Hybrid Anomaly Fusion Diagnosis Report",
        "",
        "## Research question",
        "",
        "Can embedding-space anomaly detection improve provider fraud detection when "
        "supervised models plateau around AUPRC 0.65–0.68?",
        "",
        "## Executive summary",
        "",
        f"- **LR baseline AUPRC:** {lr_baseline:.4f}",
        f"- **GraphSAGE benchmark:** {GRAPHSAGE_BENCHMARK_AUPRC:.4f}",
    ]

    ranked = sorted(
        result.model_scores.items(),
        key=lambda x: float(x[1]["auprc"]["mean"]),
        reverse=True,
    )
    if ranked:
        best_name, best_summary = ranked[0]
        best_auprc = float(best_summary["auprc"]["mean"])
        lines.append(
            f"- **Best model:** `{best_name}` — AUPRC **{best_auprc:.4f} ± "
            f"{best_summary['auprc']['std']:.4f}**"
        )
        beats_lr = best_auprc > lr_baseline
        lines.append(f"- **Beat LR:** {'YES' if beats_lr else 'NO'}")

    lines.extend(["", "## Model comparison (5-fold CV)", ""])
    lines.append("| Model | AUPRC | ROC-AUC | Prec | Recall | F1 | Recall@K |")
    lines.append("|-------|-------|---------|------|--------|-----|----------|")
    for name, summary in ranked:
        s = summary
        lines.append(
            f"| `{name}` | {s['auprc']['mean']:.4f}±{s['auprc']['std']:.4f} | "
            f"{s['roc_auc']['mean']:.4f} | {s['precision']['mean']:.4f} | "
            f"{s['recall']['mean']:.4f} | {s['f1']['mean']:.4f} | "
            f"{s['recall_at_k']['mean']:.4f} |"
        )

    overlap = result.error_overlap
    lines.extend(
        [
            "",
            "## Error overlap (LR misses caught by anomaly / graph)",
            "",
            f"- Fraud providers in val (total): **{overlap.get('total_fraud_providers', 0)}**",
            f"- Missed by LR top-K: **{overlap.get('lr_missed_fraud_total', 0)}**",
            f"- LR misses caught by IF-tabular: **{overlap.get('lr_missed_caught_by_if_tabular', 0)}**",
            f"- LR misses caught by IF-GraphSAGE: **{overlap.get('lr_missed_caught_by_if_graphsage', 0)}**",
            f"- LR misses caught by IF-R-GCN: **{overlap.get('lr_missed_caught_by_if_rgcn', 0)}**",
            f"- LR misses caught by R-GCN: **{overlap.get('lr_missed_caught_by_rgcn', 0)}**",
            f"- LR misses caught by fusion (weighted): **{overlap.get('lr_missed_caught_by_fusion_weighted', 0)}**",
        ]
    )

    if significance:
        lines.extend(["", "## Statistical significance (paired Wilcoxon vs fusion_weighted)", ""])
        for item in significance:
            lines.append(
                f"- **{item['model_a']} vs {item['model_b']}**: p={item['p_value']:.4f}, "
                f"mean_diff={item['mean_diff']:.4f}, significant={item['significant_at_0_05']}"
            )

    best_auprc = float(ranked[0][1]["auprc"]["mean"]) if ranked else 0.0
    lines.extend(["", "## Conclusion", ""])
    if best_auprc <= lr_baseline:
        lines.append(
            "Hybrid fusion and anomaly towers did not improve AUPRC over logistic regression. "
            "This supports the hypothesis that CMS provider fraud detection in this corpus is "
            "**primarily feature-driven** rather than graph-structure-driven: graph embeddings "
            "and unsupervised anomaly scores add little beyond tabular provider features."
        )
    else:
        lines.append(
            "Hybrid fusion exceeded the LR baseline, suggesting complementary signal from "
            "graph embeddings and/or anomaly scoring beyond tabular features alone."
        )

    path.write_text("\n".join(lines), encoding="utf-8")
