#!/usr/bin/env python3
"""Generate publication-quality figures from frozen artifacts/published/."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "artifacts" / "published"
DIAGNOSTICS = ROOT / "artifacts" / "diagnostics"
FIGURES = ROOT / "figures"

# IEEE / Springer friendly styling
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PALETTE = {
    "lr": "#2166AC",
    "tabular": "#4393C3",
    "graph": "#D6604D",
    "fusion": "#B2182B",
    "best": "#1B7837",
    "muted": "#636363",
    "provider": "#2166AC",
    "beneficiary": "#4DAF4A",
    "physician": "#984EA3",
    "gate": "#E66101",
}


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _ci95(mean: float, std: float, n: int = 5) -> tuple[float, float]:
    """95% confidence interval for the mean (t-distribution, n folds)."""
    if n <= 1 or std == 0:
        return mean, mean
    half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)
    return mean - half, mean + half


def _export(fig: plt.Figure, stem: str) -> None:
    for fmt in ("png", "svg", "pdf"):
        out_dir = FIGURES / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.{fmt}")
    plt.close(fig)


def _metric_from_benchmark(path: Path) -> tuple[float, float]:
    data = _load_json(path)
    s = data["summary"]["auprc"]
    return float(s["mean"]), float(s["std"])


def _metric_from_fusion(name: str) -> tuple[float, float]:
    data = _load_json(PUBLISHED / "fusion" / "fusion_benchmark.json")
    s = data["model_summaries"][name]["auprc"]
    return float(s["mean"]), float(s["std"])


def figure1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    stages = [
        ("CMS Dataset", None),
        ("Data\nProcessing", "G1"),
        ("Graph\nConstruction", "G2"),
        ("Feature\nEngineering", None),
        ("Baseline\nModels", "G3"),
        ("Graph\nModels", "G4–G5"),
        ("Fusion\nModels", "G6"),
        ("Fraud\nPrediction", None),
    ]
    xs = np.linspace(0.6, 9.4, len(stages))
    y_main = 3.2
    box_w, box_h = 1.05, 0.95

    for i, (label, gate) in enumerate(stages):
        x = xs[i]
        color = "#E8F1FA" if gate else "#F5F5F5"
        edge = PALETTE["gate"] if gate else "#333333"
        box = FancyBboxPatch(
            (x - box_w / 2, y_main - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2 if gate else 0.9,
            edgecolor=edge,
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y_main, label, ha="center", va="center", fontsize=9, zorder=3)
        if gate:
            ax.text(
                x,
                y_main + box_h / 2 + 0.18,
                gate,
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["gate"],
                fontweight="bold",
            )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - box_w / 2 - 0.05, y_main),
                xytext=(x + box_w / 2 + 0.05, y_main),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3),
            )

    # Branch annotations
    ax.annotate(
        "Tabular features",
        xy=(xs[4], y_main - box_h / 2 - 0.05),
        xytext=(xs[3], 1.5),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["tabular"], lw=1.0),
        fontsize=8,
        color=PALETTE["tabular"],
        ha="center",
    )
    ax.annotate(
        "Heterogeneous graph",
        xy=(xs[5], y_main - box_h / 2 - 0.05),
        xytext=(xs[2], 1.5),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["graph"], lw=1.0),
        fontsize=8,
        color=PALETTE["graph"],
        ha="center",
    )

    ax.set_title(
        "Hybrid CMS Provider Fraud Detection Pipeline",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        5.0,
        0.35,
        "Provider-disjoint stratified 5-fold CV  ·  Primary metric: AUPRC",
        ha="center",
        fontsize=8.5,
        color=PALETTE["muted"],
    )
    _export(fig, "fig01_system_architecture")


def figure2_graph_schema() -> None:
    manifest = _load_json(PUBLISHED / "graphs" / "v1.1" / "reference" / "graph_manifest.json")
    counts = manifest["node_counts"]
    edges = manifest["edge_counts"]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.2)
    ax.axis("off")

    pos = {
        "provider": (-0.55, 0.0),
        "beneficiary": (0.55, 0.55),
        "physician": (0.55, -0.55),
    }
    radii = {"provider": 0.22, "beneficiary": 0.20, "physician": 0.20}
    colors = {
        "provider": PALETTE["provider"],
        "beneficiary": PALETTE["beneficiary"],
        "physician": PALETTE["physician"],
    }
    labels = {
        "provider": f"Provider\n(n={counts['provider']:,})",
        "beneficiary": f"Beneficiary\n(n={counts['beneficiary']:,})",
        "physician": f"Physician\n(n={counts['physician']:,})",
    }

    def draw_node(key: str, fraud: bool = False) -> None:
        x, y = pos[key]
        r = radii[key]
        circle = plt.Circle((x, y), r, facecolor=colors[key], edgecolor="white", lw=2, alpha=0.92)
        ax.add_patch(circle)
        ax.text(x, y, labels[key], ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if fraud:
            ax.text(x, y - r - 0.08, "Fraud label", ha="center", fontsize=7.5, color="#B2182B")

    draw_node("provider", fraud=True)
    draw_node("beneficiary")
    draw_node("physician")

    def edge_label(xm: float, ym: float, text: str, weight: str) -> None:
        ax.text(xm, ym, f"{text}\n{weight}", ha="center", va="center", fontsize=7.5, color="#222222")

    # treats
    ax.annotate(
        "",
        xy=(pos["beneficiary"][0] - radii["beneficiary"], pos["beneficiary"][1] - 0.05),
        xytext=(pos["provider"][0] + radii["provider"], pos["provider"][1] + 0.05),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4, connectionstyle="arc3,rad=0.1"),
    )
    edge_label(-0.02, 0.42, "treats", f"w = claim count\n({edges['provider|treats|beneficiary']:,} edges)")

    # bills_with
    ax.annotate(
        "",
        xy=(pos["physician"][0] - radii["physician"], pos["physician"][1] + 0.05),
        xytext=(pos["provider"][0] + radii["provider"], pos["provider"][1] - 0.05),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4, connectionstyle="arc3,rad=-0.1"),
    )
    edge_label(-0.02, -0.42, "bills_with", f"w = claim count\n({edges['provider|bills_with|physician']:,} edges)")

    # collaborates (self-loop style between two provider nodes)
    p1 = (-0.82, 0.55)
    p2 = (-0.82, -0.55)
    for p, dy in ((p1, 0.12), (p2, -0.12)):
        c = plt.Circle(p, 0.12, facecolor=PALETTE["provider"], edgecolor="white", lw=1.5, alpha=0.85)
        ax.add_patch(c)
        ax.text(p[0], p[1], "P", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.annotate(
        "",
        xy=(p2[0] + 0.12, p2[1]),
        xytext=(p1[0] + 0.12, p1[1]),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.3, connectionstyle="arc3,rad=0.45"),
    )
    ax.text(
        -1.05,
        0.0,
        "collaborates\nw = shared beneficiaries\n(≥2 required)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#222222",
    )
    ax.text(-1.05, -0.35, f"{edges['provider|collaborates|provider']:,} edges", ha="center", fontsize=7, color=PALETTE["muted"])

    ax.set_title("schema_v1.1 Heterogeneous Graph", fontsize=12, fontweight="bold", pad=10)
    ax.text(
        0.0,
        -0.92,
        "Prediction target: provider-level fraud (506 / 5,410 = 9.35%)",
        ha="center",
        fontsize=8.5,
        color=PALETTE["muted"],
    )
    _export(fig, "fig02_graph_schema")


def figure3_model_comparison() -> None:
    models = [
        ("Logistic Regression", _metric_from_benchmark(PUBLISHED / "baselines" / "logistic_regression.json"), PALETTE["lr"], True),
        ("Random Forest", _metric_from_benchmark(PUBLISHED / "baselines" / "random_forest.json"), PALETTE["tabular"], False),
        ("CatBoost", _metric_from_benchmark(PUBLISHED / "baselines" / "catboost.json"), PALETTE["tabular"], False),
        ("RF + Centrality", _metric_from_benchmark(PUBLISHED / "baselines" / "rf_centrality.json"), PALETTE["tabular"], False),
        ("GraphSAGE", _metric_from_benchmark(PUBLISHED / "gnn" / "graphsage_benchmark.json"), PALETTE["graph"], False),
        ("R-GCN", _metric_from_benchmark(PUBLISHED / "rgcn" / "rgcn_benchmark.json"), PALETTE["graph"], False),
        ("Fusion (stack)", _metric_from_fusion("fusion_stack_logistic"), PALETTE["fusion"], False),
    ]

    names = [m[0] for m in models]
    means = [m[1][0] for m in models]
    stds = [m[1][1] for m in models]
    colors = [m[2] for m in models]
    is_best = [m[3] for m in models]

    cis = [_ci95(m, s) for m, s in zip(means, stds)]
    ci_lo = [m - low for m, (low, _) in zip(means, cis)]
    ci_hi = [high - m for m, (_, high) in zip(means, cis)]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, width=0.72, zorder=2)
    ax.errorbar(
        x,
        means,
        yerr=[ci_lo, ci_hi],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )

    for i, (bar, best) in enumerate(zip(bars, is_best)):
        if best:
            bar.set_edgecolor(PALETTE["best"])
            bar.set_linewidth(2.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                means[i] + ci_hi[i] + 0.012,
                "Best",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["best"],
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylabel("Mean AUPRC (5-fold CV)")
    ax.set_ylim(0.58, max(means) + 0.08)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.text(
        0.02,
        0.02,
        "Error bars: 95% CI of fold means (t-distribution, n=5)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=PALETTE["muted"],
    )
    _export(fig, "fig03_model_comparison")


def figure4_ablations() -> None:
    ablation_dir = DIAGNOSTICS / "rgcn" / "ablations"
    mapping = [
        ("Full R-GCN", ablation_dir / "rgcn_h128_L2_d0.4_b4_f15-10.json"),
        ("No PP", ablation_dir / "rgcn_h128_L2_d0.4_b4_f15-10_no_pp_full.json"),
        ("No Treats", ablation_dir / "rgcn_h128_L2_d0.4_b4_f15-10_no_treats_full.json"),
        ("No Bills", ablation_dir / "rgcn_h128_L2_d0.4_b4_f15-10_no_bills_full.json"),
    ]
    # Provider-only: documented in rgcn_diagnosis.md (fold-0); no separate JSON archived
    provider_only_auprc = 0.6525

    names: list[str] = []
    auprcs: list[float] = []
    for name, path in mapping:
        data = _load_json(path)
        names.append(name)
        auprcs.append(float(data["summary"]["auprc"]["mean"]))
    names.append("Provider Only")
    auprcs.append(provider_only_auprc)

    baseline = auprcs[0]
    rel_loss = [(baseline - v) / baseline * 100 for v in auprcs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})

    x = np.arange(len(names))
    colors = [PALETTE["graph"] if i == 0 else "#F4A582" for i in range(len(names))]
    ax1.bar(x, auprcs, color=colors, edgecolor="white", width=0.68)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("AUPRC (fold 0)")
    ax1.set_ylim(min(auprcs) - 0.02, max(auprcs) + 0.015)
    ax1.set_title("(a) Absolute AUPRC", fontweight="bold", loc="left")
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)

    ax2.barh(x, rel_loss, color=colors, edgecolor="white", height=0.68)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Relative loss vs Full (%)")
    ax2.axvline(0, color="#333333", lw=0.8)
    ax2.set_title("(b) Relative Performance Loss", fontweight="bold", loc="left")
    ax2.xaxis.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("R-GCN Relation & Feature Ablations (h128_L2_d0.4)", fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Provider Only AUPRC from diagnosis report (fold-0); other bars from diagnostics/rgcn/ablations/",
        ha="center",
        fontsize=7.5,
        color=PALETTE["muted"],
    )
    _export(fig, "fig04_rgcn_ablations")


def _mean_pr_curve(fusion_path: Path, model_key: str, n_points: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    """Average per-fold PR curves; returns recall grid, mean precision, mean fold AUPRC."""
    data = _load_json(fusion_path)
    recall_grid = np.linspace(0.0, 1.0, n_points)
    precisions: list[np.ndarray] = []
    fold_auprcs: list[float] = []

    for fold in data["folds"]:
        y_true = np.array(fold["val_labels"], dtype=int)
        scores = np.array(fold["val_scores"][model_key], dtype=float)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        fold_auprcs.append(float(auc(rec, prec)))
        # precision_recall_curve returns rec ascending after flip
        prec_interp = np.interp(recall_grid, rec[::-1], prec[::-1], left=1.0, right=0.0)
        precisions.append(prec_interp)

    return recall_grid, np.mean(precisions, axis=0), float(np.mean(fold_auprcs))


def figure5_pr_curves() -> None:
    fusion_path = PUBLISHED / "fusion" / "fusion_benchmark.json"
    model_map = {
        "logistic_regression": ("Logistic Regression", PALETTE["lr"], 2.5),
        "catboost": ("CatBoost", PALETTE["tabular"], 1.8),
        "graphsage": ("GraphSAGE", PALETTE["graph"], 1.8),
        "rgcn": ("R-GCN", "#F4A582", 1.8),
        "fusion_stack_logistic": ("Fusion (stack)", PALETTE["fusion"], 1.8),
    }

    prevalence = 506 / 5410

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for key, (label, color, lw) in model_map.items():
        rec, prec, mean_auprc = _mean_pr_curve(fusion_path, key)
        is_best = key == "logistic_regression"
        ax.plot(
            rec,
            prec,
            label=f"{label} (AUPRC={mean_auprc:.3f})",
            color=color,
            lw=lw if is_best else 1.6,
            alpha=1.0 if is_best else 0.85,
            zorder=5 if is_best else 3,
        )

    ax.axhline(prevalence, color=PALETTE["muted"], ls="--", lw=1.0, label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=8)
    ax.set_title("Precision–Recall Curves (mean across 5 folds)", fontweight="bold")
    ax.text(
        0.02,
        0.02,
        "Curves averaged via per-fold interpolation; AUPRC = mean fold AP",
        transform=ax.transAxes,
        fontsize=7.5,
        color=PALETTE["muted"],
    )
    _export(fig, "fig05_precision_recall")


def figure6_findings_summary() -> None:
    lr_m, lr_s = _metric_from_benchmark(PUBLISHED / "baselines" / "logistic_regression.json")
    gs_m, gs_s = _metric_from_benchmark(PUBLISHED / "gnn" / "graphsage_benchmark.json")
    rg_m, rg_s = _metric_from_benchmark(PUBLISHED / "rgcn" / "rgcn_benchmark.json")
    fu_m, fu_s = _metric_from_fusion("fusion_stack_logistic")

    entries = [
        ("Logistic Regression", lr_m, lr_s, PALETTE["lr"]),
        ("GraphSAGE", gs_m, gs_s, PALETTE["graph"]),
        ("R-GCN", rg_m, rg_s, "#F4A582"),
        ("Fusion (stack)", fu_m, fu_s, PALETTE["fusion"]),
    ]

    fig = plt.figure(figsize=(7.0, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.35)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")

    names = [e[0] for e in entries]
    means = [e[1] for e in entries]
    stds = [e[2] for e in entries]
    colors = [e[3] for e in entries]

    y = np.arange(len(names))
    ax_bar.barh(y, means, xerr=stds, color=colors, edgecolor="white", height=0.62, capsize=3, zorder=2)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names)
    ax_bar.set_xlabel("Mean AUPRC ± std (5-fold)")
    ax_bar.set_xlim(0.60, 0.72)
    ax_bar.invert_yaxis()
    ax_bar.axvline(lr_m, color=PALETTE["best"], ls="--", lw=1.2, alpha=0.8)
    ax_bar.text(lr_m + 0.002, -0.35, "LR baseline", fontsize=8, color=PALETTE["best"])
    ax_bar.set_title("(a) Feature vs Graph Methods", fontweight="bold", loc="left")
    ax_bar.xaxis.grid(True, linestyle=":", alpha=0.5)

    for i, (name, mean, std, _) in enumerate(entries):
        ax_bar.text(mean + std + 0.004, i, f"{mean:.3f}", va="center", fontsize=8)

    interpretation = (
        "Key finding\n"
        "───────────\n"
        "Tabular logistic regression\n"
        "outperforms all graph-based\n"
        "and fusion models on the\n"
        "CMS-Kaggle benchmark under\n"
        "provider-disjoint CV.\n\n"
        "• LR AUPRC: 0.681 ± 0.039\n"
        "• Best graph model (R-GCN):\n"
        "  0.654 ± 0.046 (−4.0%)\n"
        "• Best fusion: 0.667 ± 0.052\n"
        "  (−2.1% vs LR)\n\n"
        "Rich provider tabular features\n"
        "dominate relational signal\n"
        "under strict leakage control."
    )
    ax_txt.text(
        0.0,
        1.0,
        interpretation,
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )
    ax_txt.set_title("(b) Interpretation", fontweight="bold", loc="left")

    fig.suptitle("Research Findings: Feature-Based Methods Win", fontweight="bold", y=1.02)
    _export(fig, "fig06_findings_summary")


def main() -> int:
    print("Generating publication figures from artifacts/published/ ...")
    figure1_architecture()
    print("  [1/6] System architecture")
    figure2_graph_schema()
    print("  [2/6] Graph schema")
    figure3_model_comparison()
    print("  [3/6] Model comparison")
    figure4_ablations()
    print("  [4/6] R-GCN ablations")
    figure5_pr_curves()
    print("  [5/6] Precision–recall curves")
    figure6_findings_summary()
    print("  [6/6] Findings summary")
    print(f"Done. Outputs: {FIGURES}/{{png,svg,pdf}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
