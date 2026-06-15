"""
Experiment journal helpers for evaluation runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from hgad_cms.evaluation.cross_validation import BaselineCVResult

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_PATH = Path("docs/experiment_journal.md")


def append_baseline_journal_entry(
    results: list[BaselineCVResult],
    *,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    run_id: str = "baselines_v1",
) -> None:
    """Append a baseline CV summary entry to the experiment journal."""
    if not journal_path.is_file():
        logger.warning("Journal file not found: %s", journal_path)
        return

    lines = [
        "",
        f"### {run_id}",
        "",
        "```",
        f"Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"Run ID: {run_id}",
        "Git Commit: (uncommitted)",
        "Experiment Name: Phase 3 — baseline model benchmark",
        "Model: LR / RF / CatBoost / RF+Centrality",
        "Dataset Split: Provider-disjoint stratified 5-fold CV",
        "Configuration:",
        "  - feature_source: train-fold provider aggregations",
        "  - primary_metric: auprc",
        "Metrics:",
    ]
    for result in sorted(results, key=lambda r: r.summary["auprc"]["mean"], reverse=True):
        mean = result.summary["auprc"]["mean"]
        std = result.summary["auprc"]["std"]
        lines.append(f"  - {result.model_name}_auprc: {mean:.4f} +/- {std:.4f}")
    lines.extend(
        [
            "Observations:",
            "  - Per-fold metrics and confusion matrices saved under artifacts/results/baselines/",
            "Issues: None",
            "Next Action: Phase 4 — GNN model training",
            "```",
            "",
        ]
    )

    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logger.info("Appended journal entry: %s", run_id)
