"""Fusion and anomaly-detection configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hgad_cms.graphsage.config import GraphSAGEConfig, LR_BASELINE_AUPRC
from hgad_cms.rgcn.config import GRAPHSAGE_BENCHMARK_AUPRC, RGCNConfig

FUSION_DIR_NAME = "fusion"
RESULTS_DIR_NAME = "results"

# Best configs from Phase 4/5 publication benchmarks
DEFAULT_GRAPHSAGE_CONFIG = GraphSAGEConfig(
    hidden_dim=32,
    num_layers=2,
    dropout=0.3,
    aggregator="mean",
    fanout=(15, 10),
    max_epochs=100,
    patience=15,
)

DEFAULT_RGCN_CONFIG = RGCNConfig(
    hidden_dim=128,
    num_layers=2,
    dropout=0.4,
    num_bases=4,
    fanout=(15, 10),
    max_epochs=100,
    patience=10,
)

SCORE_KEYS: tuple[str, ...] = (
    "logistic_regression",
    "catboost",
    "graphsage",
    "rgcn",
    "if_tabular",
    "if_graphsage",
    "if_rgcn",
)

FUSION_KEYS: tuple[str, ...] = (
    "fusion_weighted",
    "fusion_stack_logistic",
    "fusion_rank",
)


@dataclass(frozen=True)
class FusionConfig:
    """Hybrid fusion experiment configuration."""

    schema_name: str = "v1.1"
    graphsage_config: GraphSAGEConfig = DEFAULT_GRAPHSAGE_CONFIG
    rgcn_config: RGCNConfig = DEFAULT_RGCN_CONFIG
    if_contamination: float | str = "auto"
    if_n_estimators: int = 200
    if_random_state: int = 42
    stack_holdout_fraction: float = 0.15
    seed: int = 42


def load_best_configs_from_artifacts(
    results_dir: Path,
) -> tuple[GraphSAGEConfig, RGCNConfig]:
    """Load best GraphSAGE/R-GCN configs from benchmark JSON if present."""
    gs_path = results_dir / "gnn" / "graphsage_benchmark.json"
    rg_path = results_dir / "rgcn" / "rgcn_benchmark.json"
    gs_cfg = DEFAULT_GRAPHSAGE_CONFIG
    rg_cfg = DEFAULT_RGCN_CONFIG
    if gs_path.is_file():
        import json

        payload = json.loads(gs_path.read_text(encoding="utf-8"))
        gs_cfg = GraphSAGEConfig.from_dict(payload["config"])
    if rg_path.is_file():
        import json

        payload = json.loads(rg_path.read_text(encoding="utf-8"))
        rg_cfg = RGCNConfig.from_dict(payload["config"])
    return gs_cfg, rg_cfg
