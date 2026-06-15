"""Experiment tracking utilities."""

from hgad_cms.tracking.logger import setup_logging
from hgad_cms.tracking.research_docs import (
    on_benchmark_complete,
    on_gate_result,
    record_decision,
    record_finding,
    sync_all,
)

__all__ = [
    "setup_logging",
    "on_benchmark_complete",
    "on_gate_result",
    "record_decision",
    "record_finding",
    "sync_all",
]
