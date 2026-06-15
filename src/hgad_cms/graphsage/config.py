"""
GraphSAGE hyperparameter configuration and search grids (schema_v1.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Literal

AggregationType = Literal["mean", "max"]
SearchMode = Literal["full", "eval", "quick", "single"]


@dataclass(frozen=True)
class GraphSAGEConfig:
    """GraphSAGE training configuration tuned for RTX 3050 4GB."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    aggregator: AggregationType = "mean"
    fanout: tuple[int, ...] = (15, 10)
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    seed: int = 42
    schema_name: str = "v1.1"

    def __post_init__(self) -> None:
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be >= 8")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if len(self.fanout) != self.num_layers:
            object.__setattr__(self, "fanout", _default_fanout(self.num_layers))
        if self.aggregator not in ("mean", "max"):
            raise ValueError("aggregator must be 'mean' or 'max'")

    @property
    def config_id(self) -> str:
        fanout_str = "-".join(str(v) for v in self.fanout)
        return (
            f"h{self.hidden_dim}_L{self.num_layers}_d{self.dropout:.1f}_"
            f"{self.aggregator}_f{fanout_str}"
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["fanout"] = list(self.fanout)
        payload["config_id"] = self.config_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GraphSAGEConfig:
        fanout = payload.get("fanout", (15, 10))
        return cls(
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            dropout=float(payload["dropout"]),
            aggregator=str(payload["aggregator"]),  # type: ignore[arg-type]
            fanout=tuple(int(v) for v in fanout),  # type: ignore[arg-type]
            batch_size=int(payload.get("batch_size", 256)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-5)),
            max_epochs=int(payload.get("max_epochs", 100)),
            patience=int(payload.get("patience", 15)),
            min_delta=float(payload.get("min_delta", 1e-4)),
            seed=int(payload.get("seed", 42)),
            schema_name=str(payload.get("schema_name", "v1.1")),
        )


def _default_fanout(num_layers: int) -> tuple[int, ...]:
    if num_layers == 1:
        return (15,)
    if num_layers == 2:
        return (15, 10)
    base = [15, 10]
    while len(base) < num_layers:
        base.append(max(5, base[-1] // 2))
    return tuple(base[:num_layers])


HIDDEN_DIMS: tuple[int, ...] = (32, 64, 128)
NUM_LAYERS: tuple[int, ...] = (2, 3)
DROPOUTS: tuple[float, ...] = (0.2, 0.4, 0.5)
AGGREGATORS: tuple[AggregationType, ...] = ("mean", "max")
DEFAULT_FANOUT: tuple[int, ...] = (15, 10)

LR_BASELINE_AUPRC: float = 0.6810


def iter_search_grid(mode: SearchMode = "full") -> list[GraphSAGEConfig]:
    """Yield GraphSAGE configs for hyperparameter search."""
    if mode == "single":
        return [GraphSAGEConfig()]

    if mode == "full":
        hidden = HIDDEN_DIMS
        layers = NUM_LAYERS
        dropouts = DROPOUTS
        aggregators = AGGREGATORS
    elif mode == "eval":
        # Quick structural search: hidden × layers × aggregation (fixed dropout)
        hidden = HIDDEN_DIMS
        layers = NUM_LAYERS
        dropouts = (0.3,)
        aggregators = AGGREGATORS
    else:  # quick
        hidden = (32, 64, 128)
        layers = (2,)
        dropouts = (0.3,)
        aggregators = AGGREGATORS

    configs: list[GraphSAGEConfig] = []
    for hidden_dim, num_layers, dropout, aggregator in product(
        hidden, layers, dropouts, aggregators
    ):
        configs.append(
            GraphSAGEConfig(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                aggregator=aggregator,
                fanout=_default_fanout(num_layers),
            )
        )
    return configs
