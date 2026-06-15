"""R-GCN benchmark for provider-level CMS fraud detection."""

from hgad_cms.rgcn.config import (
    GRAPHSAGE_BENCHMARK_AUPRC,
    LR_BASELINE_AUPRC,
    RGCNConfig,
    iter_search_grid,
)
from hgad_cms.rgcn.cross_validation import (
    RGCNCVResult,
    run_rgcn_cv,
    save_rgcn_result,
    validate_gate_g5,
)
from hgad_cms.rgcn.model import HeteroRGCN

__all__ = [
    "GRAPHSAGE_BENCHMARK_AUPRC",
    "HeteroRGCN",
    "LR_BASELINE_AUPRC",
    "RGCNConfig",
    "RGCNCVResult",
    "iter_search_grid",
    "run_rgcn_cv",
    "save_rgcn_result",
    "validate_gate_g5",
]
