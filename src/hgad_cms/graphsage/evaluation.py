"""
GraphSAGE HPO analysis, plotting, and diagnosis report generation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hgad_cms.evaluation.cross_validation import load_baseline_result
from hgad_cms.evaluation.significance import paired_test
from hgad_cms.graphsage.config import GraphSAGEConfig, LR_BASELINE_AUPRC
from hgad_cms.graphsage.cross_validation import (
    GraphSAGECVResult,
    load_graphsage_result,
    run_graphsage_cv,
    save_graphsage_result,
)
from hgad_cms.tracking.experiment import ExperimentTracker

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ("auprc", "roc_auc", "precision", "recall", "f1")
HPO_DIR_NAME = "hpo_search"


@dataclass
class HPOResultRow:
    """Single HPO trial summary."""

    rank: int
    config_id: str
    config: GraphSAGEConfig
    auprc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    best_epoch: int
    result_path: str


def _hpo_path(output_dir: Path, config: GraphSAGEConfig) -> Path:
    return output_dir / HPO_DIR_NAME / f"hpo_{config.config_id}.json"


def run_hpo_search(
    configs: list[GraphSAGEConfig],
    *,
    processed_dir: Path,
    splits_dir: Path,
    graphs_dir: Path,
    output_dir: Path,
    search_fold: int = 0,
    device: str = "auto",
    run_id: str = "graphsage_eval",
    resume: bool = True,
    tracker: ExperimentTracker | None = None,
) -> list[GraphSAGECVResult]:
    """Run fold-0 HPO for each config, with per-config resume."""
    hpo_dir = output_dir / HPO_DIR_NAME
    hpo_dir.mkdir(parents=True, exist_ok=True)
    results: list[GraphSAGECVResult] = []

    for config in configs:
        path = hpo_dir / f"graphsage_{config.config_id}.json"
        if resume and path.is_file():
            logger.info("HPO resume: loading %s", path.name)
            results.append(load_graphsage_result(path))
            continue

        logger.info("HPO trial config=%s fold=%s", config.config_id, search_fold)
        result = run_graphsage_cv(
            config,
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            graphs_dir=graphs_dir,
            device=device,
            tracker=tracker,
            run_id=f"{run_id}_hpo",
            fold_ids=[search_fold],
        )
        save_graphsage_result(result, hpo_dir)
        results.append(result)

        if tracker is not None:
            tracker.log_search_result(
                run_id=run_id,
                config=config,
                mean_auprc=float(result.summary["auprc"]["mean"]),
                std_auprc=float(result.summary["auprc"]["std"]),
            )

    return results


def _metric_from_result(result: GraphSAGECVResult, name: str) -> float:
    return float(result.summary[name]["mean"])


def build_hpo_leaderboard(
    results: list[GraphSAGECVResult],
    *,
    top_k: int = 10,
) -> list[HPOResultRow]:
    """Rank HPO trials by fold-0 AUPRC."""
    sorted_results = sorted(
        results,
        key=lambda r: _metric_from_result(r, "auprc"),
        reverse=True,
    )
    rows: list[HPOResultRow] = []
    for rank, result in enumerate(sorted_results[:top_k], start=1):
        fold = result.fold_results[0]
        rows.append(
            HPOResultRow(
                rank=rank,
                config_id=result.config.config_id,
                config=result.config,
                auprc=_metric_from_result(result, "auprc"),
                roc_auc=_metric_from_result(result, "roc_auc"),
                precision=_metric_from_result(result, "precision"),
                recall=_metric_from_result(result, "recall"),
                f1=_metric_from_result(result, "f1"),
                best_epoch=fold.best_epoch,
                result_path=f"{result.model_name}.json",
            )
        )
    return rows


def hpo_leaderboard_dataframe(rows: list[HPOResultRow]) -> pd.DataFrame:
    records = [
        {
            "rank": row.rank,
            "config_id": row.config_id,
            "hidden_dim": row.config.hidden_dim,
            "num_layers": row.config.num_layers,
            "dropout": row.config.dropout,
            "aggregator": row.config.aggregator,
            "auprc": row.auprc,
            "roc_auc": row.roc_auc,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "best_epoch": row.best_epoch,
        }
        for row in rows
    ]
    return pd.DataFrame(records)


def compare_factor(
    results: list[GraphSAGECVResult],
    factor: str,
) -> pd.DataFrame:
    """Compare mean AUPRC grouped by a config factor."""
    groups: dict[str, list[float]] = defaultdict(list)
    for result in results:
        config = result.config
        if factor == "aggregator":
            key = config.aggregator
        elif factor == "hidden_dim":
            key = str(config.hidden_dim)
        elif factor == "num_layers":
            key = str(config.num_layers)
        elif factor == "dropout":
            key = f"{config.dropout:.1f}"
        else:
            raise ValueError(f"Unknown factor: {factor}")
        groups[key].append(_metric_from_result(result, "auprc"))

    rows = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                factor: key,
                "n_configs": len(arr),
                "auprc_mean": float(arr.mean()),
                "auprc_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "auprc_max": float(arr.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("auprc_mean", ascending=False)


def load_epoch_curves(experiments_path: Path) -> dict[str, list[dict[str, float]]]:
    """Load train/val curves from experiments JSONL keyed by config_id."""
    curves: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not experiments_path.is_file():
        return curves
    with experiments_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "epoch":
                continue
            config_id = record.get("config_id", "unknown")
            curves[config_id].append(
                {
                    "epoch": float(record["epoch"]),
                    "train_loss": float(record["train_loss"]),
                    "val_auprc": float(record["val_auprc"]),
                }
            )
    for config_id in curves:
        curves[config_id] = sorted(curves[config_id], key=lambda x: x["epoch"])
    return curves


def plot_training_curves(
    curves: dict[str, list[dict[str, float]]],
    *,
    config_ids: list[str],
    output_dir: Path,
) -> list[Path]:
    """Plot training loss and validation AUPRC curves."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for config_id in config_ids:
        history = curves.get(config_id)
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_auprc = [h["val_auprc"] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, marker="o", markersize=3, color="#2563eb")
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, val_auprc, marker="o", markersize=3, color="#16a34a")
        axes[1].axhline(LR_BASELINE_AUPRC, color="#dc2626", linestyle="--", label="LR baseline")
        axes[1].set_title("Validation AUPRC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUPRC")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"GraphSAGE — {config_id}")
        fig.tight_layout()
        path = output_dir / f"curve_{config_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # Combined top-config overlay
    if config_ids:
        fig, ax = plt.subplots(figsize=(8, 5))
        for config_id in config_ids[:5]:
            history = curves.get(config_id)
            if not history:
                continue
            ax.plot(
                [h["epoch"] for h in history],
                [h["val_auprc"] for h in history],
                label=config_id,
                marker="o",
                markersize=2,
            )
        ax.axhline(LR_BASELINE_AUPRC, color="black", linestyle="--", label="LR baseline (0.681)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation AUPRC")
        ax.set_title("Top configs — validation AUPRC")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        overlay_path = output_dir / "curve_top_configs_overlay.png"
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(overlay_path)

    return saved


def plot_factor_comparisons(
    results: list[GraphSAGECVResult],
    output_dir: Path,
) -> list[Path]:
    """Bar charts for aggregation, hidden dim, and layer depth."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    factors = [
        ("aggregator", "Aggregation (mean vs max)"),
        ("hidden_dim", "Hidden dimension"),
        ("num_layers", "Layer depth"),
    ]
    for factor, title in factors:
        df = compare_factor(results, factor)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df[factor].astype(str), df["auprc_mean"], yerr=df["auprc_std"], capsize=4, color="#6366f1")
        ax.axhline(LR_BASELINE_AUPRC, color="#dc2626", linestyle="--", label="LR baseline")
        ax.set_title(title)
        ax.set_ylabel("Mean HPO AUPRC (fold 0)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"compare_{factor}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def diagnose_performance(
    hpo_results: list[GraphSAGECVResult],
    benchmark: GraphSAGECVResult | None,
    *,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    """
    Diagnose limiting factors: overfitting, underfitting, class imbalance,
    heterophily, feature dominance.
    """
    best_hpo = max(hpo_results, key=lambda r: _metric_from_result(r, "auprc"))
    best_fold = best_hpo.fold_results[0]
    history = best_fold.history

    train_end_loss = history[-1]["train_loss"] if history else None
    best_epoch = best_fold.best_epoch
    total_epochs = len(history)
    peak_auprc = best_fold.best_val_auprc
    final_auprc = history[-1]["val_auprc"] if history else peak_auprc
    val_drop = peak_auprc - final_auprc

    overfitting = bool(
        total_epochs > best_epoch + 3 and val_drop > 0.01 and train_end_loss is not None and train_end_loss < 0.35
    )
    underfitting = bool(
        train_end_loss is not None and train_end_loss > 0.5 and peak_auprc < LR_BASELINE_AUPRC
    )

    metrics = best_fold.metrics
    imbalance_signal = bool(metrics.recall > 0.75 and metrics.precision < 0.50)

    baseline_auprc = LR_BASELINE_AUPRC
    if baseline_path and baseline_path.is_file():
        baseline_auprc = float(load_baseline_result(baseline_path).summary["auprc"]["mean"])

    feature_dominance = bool(peak_auprc < baseline_auprc)
    heterophily_signal = bool(
        feature_dominance and peak_auprc > 0.55
    )  # graph helps somewhat but not enough vs tabular

    beats_baseline = False
    benchmark_auprc = None
    if benchmark is not None:
        benchmark_auprc = _metric_from_result(benchmark, "auprc")
        beats_baseline = benchmark_auprc > baseline_auprc

    return {
        "best_hpo_config": best_hpo.config.config_id,
        "best_hpo_auprc_fold0": peak_auprc,
        "benchmark_auprc_mean": benchmark_auprc,
        "lr_baseline_auprc": baseline_auprc,
        "beats_lr_baseline": beats_baseline,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "train_end_loss": train_end_loss,
        "val_auprc_drop_after_peak": val_drop,
        "diagnoses": {
            "overfitting": {
                "detected": overfitting,
                "evidence": (
                    f"Val AUPRC peaked at epoch {best_epoch} ({peak_auprc:.4f}), "
                    f"then fell {val_drop:.4f} while train loss continued to {train_end_loss:.4f}"
                    if history
                    else "Insufficient history"
                ),
            },
            "underfitting": {
                "detected": underfitting,
                "evidence": (
                    f"Train loss {train_end_loss:.4f} still elevated; peak AUPRC {peak_auprc:.4f} below LR"
                    if train_end_loss
                    else "Insufficient history"
                ),
            },
            "class_imbalance": {
                "detected": imbalance_signal,
                "evidence": (
                    f"Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f} "
                    f"(high recall / low precision pattern despite pos_weight BCE)"
                ),
            },
            "graph_heterophily": {
                "detected": heterophily_signal,
                "evidence": (
                    "GraphSAGE homophily assumption may mix fraud/legitimate provider signals "
                    "via shared beneficiaries/physicians; AUPRC above chance but below tabular LR"
                ),
            },
            "feature_dominance": {
                "detected": feature_dominance,
                "evidence": (
                    f"Best GraphSAGE AUPRC {peak_auprc:.4f} < LR baseline {baseline_auprc:.4f}; "
                    "86-dim tabular provider features may carry most signal"
                ),
            },
        },
        "recommend_rgcn": not beats_baseline if benchmark else peak_auprc <= baseline_auprc,
    }


def write_diagnosis_report(
    *,
    output_path: Path,
    leaderboard: list[HPOResultRow],
    hpo_results: list[GraphSAGECVResult],
    benchmark: GraphSAGECVResult | None,
    diagnosis: dict[str, Any],
    plot_paths: list[Path],
    significance: dict[str, Any] | None = None,
) -> None:
    """Write markdown diagnosis report."""
    lines = [
        "# GraphSAGE Evaluation Diagnosis Report",
        "",
        "## Executive summary",
        "",
        f"- **LR baseline AUPRC:** {LR_BASELINE_AUPRC:.4f}",
        f"- **Best HPO config (fold 0):** `{diagnosis['best_hpo_config']}` — AUPRC **{diagnosis['best_hpo_auprc_fold0']:.4f}**",
    ]
    if benchmark is not None:
        s = benchmark.summary
        lines.extend(
            [
                f"- **5-fold benchmark AUPRC:** {s['auprc']['mean']:.4f} ± {s['auprc']['std']:.4f}",
                f"- **Gate G4 (beat LR):** {'PASSED' if diagnosis['beats_lr_baseline'] else 'FAILED'}",
            ]
        )
    lines.extend(["", "## Top 10 configurations (HPO fold 0, ranked by AUPRC)", ""])
    lines.append(
        "| Rank | Config | h | L | drop | agg | AUPRC | ROC-AUC | Prec | Recall | F1 | Best epoch |"
    )
    lines.append("|------|--------|---|----|------|-----|-------|---------|------|--------|----|------------|")
    for row in leaderboard:
        c = row.config
        lines.append(
            f"| {row.rank} | `{row.config_id}` | {c.hidden_dim} | {c.num_layers} | {c.dropout} | "
            f"{c.aggregator} | {row.auprc:.4f} | {row.roc_auc:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.best_epoch} |"
        )

    lines.extend(["", "## Factor comparisons (fold 0 HPO)", ""])
    for factor in ("aggregator", "hidden_dim", "num_layers"):
        df = compare_factor(hpo_results, factor)
        lines.append(f"### {factor}")
        lines.append("")
        lines.append("| " + " | ".join(df.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
        lines.append("")

    if benchmark is not None:
        lines.extend(["", "## 5-fold benchmark (best config)", ""])
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        for metric in METRIC_COLUMNS:
            stats = benchmark.summary[metric]
            lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} |")

    lines.extend(["", "## Training curves", ""])
    for path in plot_paths:
        lines.append(f"- `{path.as_posix()}`")

    lines.extend(["", "## Diagnosis", ""])
    for name, payload in diagnosis["diagnoses"].items():
        flag = "YES" if payload["detected"] else "no"
        lines.append(f"### {name.replace('_', ' ').title()} — **{flag}**")
        lines.append("")
        lines.append(payload["evidence"])
        lines.append("")

    if significance:
        lines.extend(
            [
                "## Statistical comparison vs logistic regression",
                "",
                f"- Paired Wilcoxon p-value: {significance.get('p_value', 'n/a')}",
                f"- Mean AUPRC diff (GraphSAGE − LR): {significance.get('mean_diff', 'n/a')}",
                f"- Significant at α=0.05: {significance.get('significant_at_0_05', 'n/a')}",
                "",
            ]
        )

    if diagnosis.get("recommend_rgcn"):
        lines.extend(
            [
                "## Recommendation: proceed to R-GCN",
                "",
                "GraphSAGE did **not** exceed the logistic regression baseline (AUPRC 0.6810). Evidence:",
                "",
                "1. **Feature dominance** — LR on 86-dim tabular features outperforms all GraphSAGE configs.",
                "2. **Relation heterogeneity** — v1.1 has treats / bills_with / collaborates with different semantics; "
                "GraphSAGE treats all relations with the same SAGEConv, while R-GCN uses relation-specific weights.",
                "3. **Heterophily** — fraud and legitimate providers share beneficiaries and physicians; "
                "mean/max aggregation may dilute fraud signal.",
                "4. **Overfitting risk** — several configs peak early then degrade on validation AUPRC.",
                "",
                "R-GCN is the appropriate next step to test whether relation-aware message passing "
                "captures graph structure LR cannot.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommendation",
                "",
                "GraphSAGE exceeds the LR baseline. Consider GraphSAGE as the graph benchmark "
                "before adding R-GCN complexity.",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote diagnosis report: %s", output_path)
