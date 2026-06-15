"""
Graph schema constants, relation names, and ablation schema definitions (schema_v1.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------
NODE_PROVIDER: Final[str] = "provider"
NODE_BENEFICIARY: Final[str] = "beneficiary"
NODE_PHYSICIAN: Final[str] = "physician"
NODE_DIAGNOSIS: Final[str] = "diagnosis"

CORE_NODE_TYPES: Final[tuple[str, ...]] = (
    NODE_PROVIDER,
    NODE_BENEFICIARY,
    NODE_PHYSICIAN,
)

# ---------------------------------------------------------------------------
# Edge relation names (canonical forward); reverse = f"{forward}_rev"
# ---------------------------------------------------------------------------
REL_TREATS: Final[str] = "treats"
REL_BILLS_WITH: Final[str] = "bills_with"
REL_COLLABORATES: Final[str] = "collaborates"
REL_SEEN_BY: Final[str] = "seen_by"
REL_DIAGNOSED_WITH: Final[str] = "diagnosed_with"

FORBIDDEN_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "PotentialFraud",
        "fraud_label",
    }
)

# ---------------------------------------------------------------------------
# Feature engineering defaults
# ---------------------------------------------------------------------------
TOP_DIAGNOSIS_FEATURES: Final[int] = 50
TOP_PROCEDURE_FEATURES: Final[int] = 20
TOP_DIAGNOSIS_NODES: Final[int] = 200
PHYSICIAN_ROLE_COLUMNS: Final[tuple[str, ...]] = (
    "AttendingPhysician",
    "OperatingPhysician",
    "OtherPhysician",
)

# ---------------------------------------------------------------------------
# CV / splits
# ---------------------------------------------------------------------------
DEFAULT_N_FOLDS: Final[int] = 5
DEFAULT_SPLIT_SEED: Final[int] = 42
LABEL_MASKED: Final[int] = -1

# ---------------------------------------------------------------------------
# G2 validation thresholds (full-corpus reference graph, schema v1.1)
# ---------------------------------------------------------------------------
G2_EXPECTED_NODE_COUNTS: Final[dict[str, int]] = {
    NODE_PROVIDER: 5_410,
    NODE_BENEFICIARY: 138_556,
    NODE_PHYSICIAN: 100_737,
}

G2_EXPECTED_EDGE_COUNTS_V1_1: Final[dict[str, int]] = {
    REL_TREATS: 363_300,
    REL_BILLS_WITH: 109_339,
    REL_COLLABORATES: 75_604,
}

G2_COUNT_TOLERANCE_FRAC: Final[float] = 0.01
G2_PHYSICIAN_COUNT_TOLERANCE: Final[int] = 50
G2_MIN_TRAIN_PROVIDER_EDGE_COVERAGE: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
SPLITS_DIR_NAME: Final[str] = "splits"
GRAPHS_DIR_NAME: Final[str] = "graphs"
SPLIT_MANIFEST_NAME: Final[str] = "split_manifest.json"
GRAPH_MANIFEST_NAME: Final[str] = "graph_manifest.json"
FOLD_SPLIT_TEMPLATE: Final[str] = "fold_{fold}.json"
REFERENCE_GRAPH_DIR: Final[str] = "reference"


@dataclass(frozen=True)
class GraphSchemaSpec:
    """Configuration for a graph construction schema / ablation variant."""

    name: str
    include_provider_provider: bool = True
    pp_min_shared_beneficiaries: int = 2
    include_beneficiary_physician: bool = False
    include_diagnosis_nodes: bool = False
    top_diagnosis_node_codes: int = TOP_DIAGNOSIS_NODES
    physician_min_claim_refs: int = 1
    symmetrize_edges: bool = True

    def __post_init__(self) -> None:
        if self.pp_min_shared_beneficiaries < 1:
            raise ValueError("pp_min_shared_beneficiaries must be >= 1")
        if self.physician_min_claim_refs < 1:
            raise ValueError("physician_min_claim_refs must be >= 1")

    @property
    def node_types(self) -> tuple[str, ...]:
        types: list[str] = list(CORE_NODE_TYPES)
        if self.include_diagnosis_nodes:
            types.append(NODE_DIAGNOSIS)
        return tuple(types)

    def expected_edge_count(self, relation: str) -> int | None:
        """Return G2 reference count when defined for v1.1 primary schema."""
        if self.name != "v1.1" and self.name != "v1.1-no_pp":
            return None
        if relation == REL_COLLABORATES and not self.include_provider_provider:
            return 0
        if relation == REL_COLLABORATES and self.pp_min_shared_beneficiaries != 2:
            return None
        return G2_EXPECTED_EDGE_COUNTS_V1_1.get(relation)

    def relation_names(self) -> tuple[str, ...]:
        """Forward relation names enabled by this schema."""
        relations = [REL_TREATS, REL_BILLS_WITH]
        if self.include_provider_provider:
            relations.append(REL_COLLABORATES)
        if self.include_beneficiary_physician:
            relations.append(REL_SEEN_BY)
        if self.include_diagnosis_nodes:
            relations.append(REL_DIAGNOSED_WITH)
        return tuple(relations)

    def all_edge_keys(self) -> tuple[tuple[str, str, str], ...]:
        """PyG-style (src, rel, dst) keys including reverse edges."""
        keys: list[tuple[str, str, str]] = []
        if REL_TREATS in self.relation_names():
            keys.extend(
                [
                    (NODE_PROVIDER, REL_TREATS, NODE_BENEFICIARY),
                    (NODE_BENEFICIARY, f"{REL_TREATS}_rev", NODE_PROVIDER),
                ]
            )
        if REL_BILLS_WITH in self.relation_names():
            keys.extend(
                [
                    (NODE_PROVIDER, REL_BILLS_WITH, NODE_PHYSICIAN),
                    (NODE_PHYSICIAN, f"{REL_BILLS_WITH}_rev", NODE_PROVIDER),
                ]
            )
        if REL_COLLABORATES in self.relation_names():
            keys.extend(
                [
                    (NODE_PROVIDER, REL_COLLABORATES, NODE_PROVIDER),
                    (NODE_PROVIDER, f"{REL_COLLABORATES}_rev", NODE_PROVIDER),
                ]
            )
        if REL_SEEN_BY in self.relation_names():
            keys.extend(
                [
                    (NODE_BENEFICIARY, REL_SEEN_BY, NODE_PHYSICIAN),
                    (NODE_PHYSICIAN, f"{REL_SEEN_BY}_rev", NODE_BENEFICIARY),
                ]
            )
        if REL_DIAGNOSED_WITH in self.relation_names():
            keys.extend(
                [
                    (NODE_PROVIDER, REL_DIAGNOSED_WITH, NODE_DIAGNOSIS),
                    (NODE_DIAGNOSIS, f"{REL_DIAGNOSED_WITH}_rev", NODE_PROVIDER),
                ]
            )
        return tuple(keys)


SCHEMA_V1_1 = GraphSchemaSpec(name="v1.1")

SCHEMA_V1_1_NO_PP = GraphSchemaSpec(
    name="v1.1-no_pp",
    include_provider_provider=False,
)

SCHEMA_V1_1_PP_T1 = GraphSchemaSpec(
    name="v1.1-pp_t1",
    pp_min_shared_beneficiaries=1,
)

SCHEMA_V1_1_PP_T5 = GraphSchemaSpec(
    name="v1.1-pp_t5",
    pp_min_shared_beneficiaries=5,
)

SCHEMA_V1B = GraphSchemaSpec(
    name="v1b",
    include_beneficiary_physician=True,
)

SCHEMA_V2 = GraphSchemaSpec(
    name="v2",
    include_diagnosis_nodes=True,
)

SCHEMA_V4 = GraphSchemaSpec(
    name="v4",
    physician_min_claim_refs=5,
)

SCHEMA_REGISTRY: Final[dict[str, GraphSchemaSpec]] = {
    spec.name: spec
    for spec in (
        SCHEMA_V1_1,
        SCHEMA_V1_1_NO_PP,
        SCHEMA_V1_1_PP_T1,
        SCHEMA_V1_1_PP_T5,
        SCHEMA_V1B,
        SCHEMA_V2,
        SCHEMA_V4,
    )
}


def get_schema(name: str) -> GraphSchemaSpec:
    """Return a registered schema spec by name."""
    if name not in SCHEMA_REGISTRY:
        known = ", ".join(sorted(SCHEMA_REGISTRY))
        raise KeyError(f"Unknown graph schema '{name}'. Known: {known}")
    return SCHEMA_REGISTRY[name]
