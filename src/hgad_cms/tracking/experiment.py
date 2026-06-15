"""
JSONL experiment tracking for GNN training runs.
"""

from __future__ import annotations

from typing import Any, Protocol

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class TrackableConfig(Protocol):
    @property
    def config_id(self) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


class ExperimentTracker:
    """Append-only JSONL tracker for GNN experiments."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_run_start(
        self,
        *,
        run_id: str,
        config: TrackableConfig | dict[str, object],
        fold_id: int | None = None,
        phase: str = "train",
    ) -> None:
        payload = {
            "event": "run_start",
            "run_id": run_id,
            "phase": phase,
            "fold_id": fold_id,
            "config": config.to_dict() if hasattr(config, "to_dict") else config,
        }
        self._append(payload)
        logger.debug("Logged run start: %s fold=%s", run_id, fold_id)

    def log_epoch(
        self,
        *,
        run_id: str,
        fold_id: int,
        epoch: int,
        train_loss: float,
        val_auprc: float,
        config_id: str | None = None,
        config: TrackableConfig | None = None,
    ) -> None:
        cid = config_id or (config.config_id if config is not None else "unknown")
        self._append(
            {
                "event": "epoch",
                "run_id": run_id,
                "fold_id": fold_id,
                "config_id": cid,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auprc": val_auprc,
            }
        )

    def log_run_end(
        self,
        *,
        run_id: str,
        fold_id: int,
        metrics: dict[str, float],
        best_epoch: int,
        config: TrackableConfig | None = None,
        config_id: str | None = None,
    ) -> None:
        cid = config_id or (config.config_id if config is not None else "unknown")
        self._append(
            {
                "event": "run_end",
                "run_id": run_id,
                "fold_id": fold_id,
                "config_id": cid,
                "best_epoch": best_epoch,
                "metrics": metrics,
            }
        )

    def log_search_result(
        self,
        *,
        run_id: str,
        mean_auprc: float,
        std_auprc: float,
        selected: bool = False,
        config: TrackableConfig | None = None,
        config_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "search_result",
            "run_id": run_id,
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
            "selected": selected,
        }
        if config is not None:
            payload["config"] = config.to_dict()
            payload["config_id"] = config.config_id
        elif config_id is not None:
            payload["config_id"] = config_id
        self._append(payload)

    def read_records(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.log_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
