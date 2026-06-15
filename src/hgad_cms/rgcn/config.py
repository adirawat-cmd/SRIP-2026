"""
R-GCN hyperparameter configuration and search grids (schema_v1.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

from hgad_cms.graphsage.config import LR_BASELINE_AUPRC

SearchMode = Literal["full", "eval", "quick", "single"]
RelationAblation = Literal["full", "no_pp", "no_treats", "no_bills", "provider_only"]
FeatureAblation = Literal["full", "provider_only"]

HIDDEN_DIMS: tuple[int, ...] = (32, 64, 128)
NUM_LAYERS: tuple[int, ...] = (2, 3)
DROPOUTS: tuple[float, ...] = (0.2, 0.4, 0.5)
GRAPHSAGE_BENCHMARK_AUPRC: float = 0.6530

# Canonical forward relations for ablation (reverse edges removed symmetrically)
REL_TREATS: tuple[str, str, str] = ("provider", "treats", "beneficiary")
REL_BILLS: tuple[str, str, str] = ("provider", "bills_with", "physician")
REL_COLLAB: tuple[str, str, str] = ("provider", "collaborates", "provider")


def _default_fanout(num_layers: int) -> tuple[int, ...]:
    if num_layers == 1:
        return (15,)
    if num_layers == 2:
        return (15, 10)
    return (15, 10, 5)


def active_edge_types(
    relation_ablation: RelationAblation = "full",
) -> tuple[tuple[str, str, str], ...]:
    """Return edge-type keys enabled for a relation ablation preset."""
    treats_fwd = (REL_TREATS, ("beneficiary", "treats_rev", "provider"))
    bills_fwd = (REL_BILLS, ("physician", "bills_with_rev", "provider"))
    collab_fwd = (REL_COLLAB, ("provider", "collaborates_rev", "provider"))
    if relation_ablation == "provider_only":
        return ()
    keys: list[tuple[str, str, str]] = []
    if relation_ablation != "no_treats":
        keys.extend(treats_fwd)
    if relation_ablation != "no_bills":
        keys.extend(bills_fwd)
    if relation_ablation not in ("no_pp", "provider_only"):
        keys.extend(collab_fwd)
    return tuple(keys)


@dataclass(frozen=True)
class RGCNConfig:
    """R-GCN training configuration tuned for RTX 3050 4GB."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    num_bases: int = 4
    fanout: tuple[int, ...] = (15, 10)
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    seed: int = 42
    schema_name: str = "v1.1"
    relation_ablation: RelationAblation = "full"
    feature_ablation: FeatureAblation = "full"

    def __post_init__(self) -> None:
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be >= 8")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if len(self.fanout) != self.num_layers:
            object.__setattr__(self, "fanout", _default_fanout(self.num_layers))

    @property
    def config_id(self) -> str:
        fanout_str = "-".join(str(v) for v in self.fanout)
        rel = self.relation_ablation
        feat = self.feature_ablation
        suffix = ""
        if rel != "full" or feat != "full":
            suffix = f"_{rel}_{feat}"
        return f"h{self.hidden_dim}_L{self.num_layers}_d{self.dropout:.1f}_b{self.num_bases}_f{fanout_str}{suffix}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["fanout"] = list(self.fanout)
        payload["config_id"] = self.config_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RGCNConfig:
        fanout = payload.get("fanout", (15, 10))
        return cls(
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            dropout=float(payload["dropout"]),
            num_bases=int(payload.get("num_bases", 4)),
            fanout=tuple(int(v) for v in fanout),  # type: ignore[arg-type]
            batch_size=int(payload.get("batch_size", 256)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-5)),
            max_epochs=int(payload.get("max_epochs", 100)),
            patience=int(payload.get("patience", 15)),
            min_delta=float(payload.get("min_delta", 1e-4)),
            seed=int(payload.get("seed", 42)),
            schema_name=str(payload.get("schema_name", "v1.1")),
            relation_ablation=str(payload.get("relation_ablation", "full")),  # type: ignore[arg-type]
            feature_ablation=str(payload.get("feature_ablation", "full")),  # type: ignore[arg-type]
        )


def iter_search_grid(mode: SearchMode = "eval") -> list[RGCNConfig]:
    """Yield R-GCN configs for hyperparameter search."""
    if mode == "single":
        return [RGCNConfig()]

    if mode == "full":
        hidden, layers, dropouts = HIDDEN_DIMS, NUM_LAYERS, DROPOUTS
    elif mode == "eval":
        hidden, layers, dropouts = HIDDEN_DIMS, NUM_LAYERS, (0.2, 0.4)
    else:  # quick
        hidden, layers, dropouts = (32, 64), (2,), (0.3,)

    return [
        RGCNConfig(
            hidden_dim=h,
            num_layers=L,
            dropout=d,
            fanout=_default_fanout(L),
        )
        for h, L, d in product(hidden, layers, dropouts)
    ]
