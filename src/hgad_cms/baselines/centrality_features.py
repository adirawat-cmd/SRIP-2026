"""
Provider graph centrality features computed from fold-safe heterogeneous graphs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from hgad_cms.constants import (
    COL_ATTENDING,
    COL_BENE_ID,
    COL_OPERATING,
    COL_OTHER_PHYSICIAN,
    COL_PROVIDER,
)
from hgad_cms.exceptions import BaselineError
from hgad_cms.graph.constants import (
    NODE_BENEFICIARY,
    NODE_PHYSICIAN,
    NODE_PROVIDER,
    REL_BILLS_WITH,
    REL_COLLABORATES,
    REL_TREATS,
)
logger = logging.getLogger(__name__)

CENTRALITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "cent_treats_degree",
    "cent_bills_degree",
    "cent_collaborates_degree",
    "cent_total_degree",
    "cent_pagerank",
    "cent_betweenness",
    "cent_clustering",
    "cent_eigenvector",
)


def _edge_key(graph_dir: Path, src: str, rel: str, dst: str) -> Path:
    return graph_dir / f"edges__{src}__{rel}__{dst}.parquet"


def _load_forward_edges(graph_dir: Path, src: str, rel: str, dst: str) -> pd.DataFrame:
    path = _edge_key(graph_dir, src, rel, dst)
    if not path.is_file():
        return pd.DataFrame(columns=["src_idx", "dst_idx"])
    return pd.read_parquet(path)


def _provider_index_map(graph_dir: Path) -> dict[int, str]:
    nodes = pd.read_parquet(graph_dir / f"nodes_{NODE_PROVIDER}.parquet")
    return nodes.set_index("node_idx")[COL_PROVIDER].astype(str).to_dict()


def build_provider_network(graph_dir: Path) -> nx.Graph:
    """
    Build an undirected provider graph from fold heterogeneous graph edges.

    Includes provider-provider collaborates edges and projected one-hop
    provider connections via beneficiaries and physicians.
    """
    idx_to_provider = _provider_index_map(graph_dir)
    graph = nx.Graph()

    for provider_id in idx_to_provider.values():
        graph.add_node(provider_id)

    def _add_edges(frame: pd.DataFrame, src_type: str, dst_type: str) -> None:
        if frame.empty:
            return
        if src_type == NODE_PROVIDER and dst_type == NODE_PROVIDER:
            providers = pd.read_parquet(graph_dir / f"nodes_{NODE_PROVIDER}.parquet")
            id_map = providers.set_index("node_idx")[COL_PROVIDER].astype(str)
            for _, row in frame.iterrows():
                a = id_map.get(int(row["src_idx"]))
                b = id_map.get(int(row["dst_idx"]))
                if a is not None and b is not None and a != b:
                    graph.add_edge(str(a), str(b))
            return

        if src_type == NODE_PROVIDER:
            src_nodes = pd.read_parquet(graph_dir / f"nodes_{NODE_PROVIDER}.parquet")
            dst_nodes = pd.read_parquet(graph_dir / f"nodes_{dst_type}.parquet")
            src_col = COL_PROVIDER
            dst_col = "Physician" if dst_type == NODE_PHYSICIAN else COL_PROVIDER if dst_type == NODE_PROVIDER else "BeneID"
            if dst_type == NODE_BENEFICIARY:
                dst_col = "BeneID"
            src_map = src_nodes.set_index("node_idx")[src_col].astype(str)
            dst_map = dst_nodes.set_index("node_idx")[dst_col].astype(str)
            for _, row in frame.iterrows():
                provider = src_map.get(int(row["src_idx"]))
                other = dst_map.get(int(row["dst_idx"]))
                if provider is None or other is None:
                    continue
                graph.nodes[provider]["neighbors_" + dst_type] = graph.nodes[provider].get(
                    "neighbors_" + dst_type, set()
                ) | {other}

    collaborates = _load_forward_edges(graph_dir, NODE_PROVIDER, REL_COLLABORATES, NODE_PROVIDER)
    _add_edges(collaborates, NODE_PROVIDER, NODE_PROVIDER)

    treats = _load_forward_edges(graph_dir, NODE_PROVIDER, REL_TREATS, NODE_BENEFICIARY)
    bills = _load_forward_edges(graph_dir, NODE_PROVIDER, REL_BILLS_WITH, NODE_PHYSICIAN)

    bene_to_providers: dict[str, set[str]] = {}
    if not treats.empty:
        providers = pd.read_parquet(graph_dir / f"nodes_{NODE_PROVIDER}.parquet")
        beneficiaries = pd.read_parquet(graph_dir / f"nodes_{NODE_BENEFICIARY}.parquet")
        prov_map = providers.set_index("node_idx")[COL_PROVIDER].astype(str)
        bene_map = beneficiaries.set_index("node_idx")["BeneID"].astype(str)
        for _, row in treats.iterrows():
            provider = prov_map.get(int(row["src_idx"]))
            bene = bene_map.get(int(row["dst_idx"]))
            if provider is None or bene is None:
                continue
            bene_to_providers.setdefault(bene, set()).add(provider)

    for providers_set in bene_to_providers.values():
        provider_list = sorted(providers_set)
        for i, a in enumerate(provider_list):
            for b in provider_list[i + 1 :]:
                graph.add_edge(a, b, relation="shared_beneficiary")

    if not bills.empty:
        providers = pd.read_parquet(graph_dir / f"nodes_{NODE_PROVIDER}.parquet")
        physicians = pd.read_parquet(graph_dir / f"nodes_{NODE_PHYSICIAN}.parquet")
        prov_map = providers.set_index("node_idx")[COL_PROVIDER].astype(str)
        phys_map = physicians.set_index("node_idx")["Physician"].astype(str)
        phys_to_providers: dict[str, set[str]] = {}
        for _, row in bills.iterrows():
            provider = prov_map.get(int(row["src_idx"]))
            physician = phys_map.get(int(row["dst_idx"]))
            if provider is None or physician is None:
                continue
            phys_to_providers.setdefault(physician, set()).add(provider)
        for providers_set in phys_to_providers.values():
            provider_list = sorted(providers_set)
            for i, a in enumerate(provider_list):
                for b in provider_list[i + 1 :]:
                    graph.add_edge(a, b, relation="shared_physician")

    return graph


def _degree_by_relation(graph_dir: Path) -> pd.DataFrame:
    providers = pd.read_parquet(graph_dir / f"nodes_{NODE_PROVIDER}.parquet")
    base = providers[[COL_PROVIDER]].copy()
    base[COL_PROVIDER] = base[COL_PROVIDER].astype(str)

    treats = _load_forward_edges(graph_dir, NODE_PROVIDER, REL_TREATS, NODE_BENEFICIARY)
    bills = _load_forward_edges(graph_dir, NODE_PROVIDER, REL_BILLS_WITH, NODE_PHYSICIAN)
    collab = _load_forward_edges(graph_dir, NODE_PROVIDER, REL_COLLABORATES, NODE_PROVIDER)

    prov_map = providers.set_index("node_idx")[COL_PROVIDER].astype(str)

    def _count_edges(frame: pd.DataFrame, col_name: str) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=np.int64)
        mapped = frame["src_idx"].map(prov_map)
        return mapped.value_counts().rename(col_name)

    counts = base.set_index(COL_PROVIDER)
    counts = counts.join(_count_edges(treats, "cent_treats_degree"), how="left")
    counts = counts.join(_count_edges(bills, "cent_bills_degree"), how="left")
    counts = counts.join(_count_edges(collab, "cent_collaborates_degree"), how="left")
    counts = counts.fillna(0)
    counts["cent_total_degree"] = (
        counts["cent_treats_degree"]
        + counts["cent_bills_degree"]
        + counts["cent_collaborates_degree"]
    )
    return counts.reset_index()


def compute_centrality_features(
    graph_dir: Path,
    provider_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute graph centrality features for providers in a fold graph artifact.

    Parameters
    ----------
    graph_dir:
        Path to saved fold graph directory.
    provider_ids:
        Optional provider subset to return.

    Returns
    -------
    pd.DataFrame
        Provider centrality features.
    """
    if not graph_dir.is_dir():
        raise BaselineError(f"Graph directory not found: {graph_dir}")

    manifest_path = graph_dir / "graph_manifest.json"
    if not manifest_path.is_file():
        raise BaselineError(f"Missing graph manifest: {manifest_path}")

    network = build_provider_network(graph_dir)
    degrees = _degree_by_relation(graph_dir)

    if network.number_of_nodes() == 0:
        raise BaselineError(f"No provider nodes found in graph: {graph_dir}")

    ordered = degrees[COL_PROVIDER].astype(str).tolist()
    centralities = _network_centralities(network, ordered)
    features = degrees.merge(centralities, on=COL_PROVIDER, how="left").fillna(0.0)

    if provider_ids is not None:
        wanted = {str(p) for p in provider_ids}
        features = features[features[COL_PROVIDER].astype(str).isin(wanted)].copy()

    logger.info(
        "Computed centrality features for %s providers from %s",
        len(features),
        graph_dir,
    )
    return features


def centrality_feature_matrix(features: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Extract numeric centrality matrix from feature frame."""
    cols = [c for c in CENTRALITY_FEATURE_COLUMNS if c in features.columns]
    matrix = features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    return cols, matrix.astype(np.float32)


def _ordered_provider_ids(provider_ids: list[str]) -> list[str]:
    """Match row order used by ``build_provider_features``."""
    return sorted({str(p) for p in provider_ids})


def build_provider_network_from_claims(
    claims: pd.DataFrame,
    *,
    min_shared_beneficiaries: int = 2,
) -> nx.Graph:
    """
    Build an undirected provider graph from a fold-safe claims subset.

    Includes explicit collaborates edges (shared beneficiaries above threshold)
    and projected one-hop provider connections via beneficiaries and physicians.
    """
    frame = claims.copy()
    if frame.empty:
        return nx.Graph()

    frame[COL_PROVIDER] = frame[COL_PROVIDER].astype(str)
    graph = nx.Graph()
    for provider in frame[COL_PROVIDER].unique():
        graph.add_node(str(provider))

    bene_providers = frame.groupby(COL_BENE_ID)[COL_PROVIDER].apply(
        lambda series: sorted({str(p) for p in series.unique()})
    )
    pair_shared: dict[tuple[str, str], int] = defaultdict(int)
    for providers in bene_providers:
        if len(providers) < 2:
            continue
        for left, right in combinations(providers, 2):
            pair_shared[(left, right)] += 1

    for (left, right), shared in pair_shared.items():
        if shared >= min_shared_beneficiaries:
            graph.add_edge(left, right, relation=REL_COLLABORATES)

    for providers in bene_providers:
        if len(providers) < 2:
            continue
        for left, right in combinations(providers, 2):
            if not graph.has_edge(left, right):
                graph.add_edge(left, right, relation="shared_beneficiary")

    physician_frames: list[pd.DataFrame] = []
    for column in (COL_ATTENDING, COL_OPERATING, COL_OTHER_PHYSICIAN):
        if column not in frame.columns:
            continue
        subset = frame[[COL_PROVIDER, column]].dropna(subset=[column])
        if subset.empty:
            continue
        physician_frames.append(
            subset.assign(Physician=subset[column].astype(str))[[COL_PROVIDER, "Physician"]]
        )

    if physician_frames:
        physician_claims = pd.concat(physician_frames, ignore_index=True)
        physician_to_providers = physician_claims.groupby("Physician")[COL_PROVIDER].apply(
            lambda series: sorted({str(p) for p in series.unique()})
        )
        for providers in physician_to_providers:
            if len(providers) < 2:
                continue
            for left, right in combinations(providers, 2):
                if not graph.has_edge(left, right):
                    graph.add_edge(left, right, relation="shared_physician")

    return graph


def _degree_from_claims(
    claims: pd.DataFrame,
    ordered_providers: list[str],
    *,
    min_shared_beneficiaries: int = 2,
) -> pd.DataFrame:
    """Compute relation-specific provider degrees from claims."""
    frame = claims.copy()
    frame[COL_PROVIDER] = frame[COL_PROVIDER].astype(str)

    treats_degree = (
        frame.groupby(COL_PROVIDER)[COL_BENE_ID].nunique()
        if not frame.empty and COL_BENE_ID in frame.columns
        else pd.Series(dtype=np.int64)
    )

    physician_frames: list[pd.DataFrame] = []
    for column in (COL_ATTENDING, COL_OPERATING, COL_OTHER_PHYSICIAN):
        if column not in frame.columns:
            continue
        subset = frame[[COL_PROVIDER, column]].dropna(subset=[column])
        if subset.empty:
            continue
        physician_frames.append(
            subset.assign(physician=subset[column].astype(str))[[COL_PROVIDER, "physician"]]
        )

    if physician_frames:
        physician_claims = pd.concat(physician_frames, ignore_index=True)
        bills_degree = physician_claims.groupby(COL_PROVIDER)["physician"].nunique()
    else:
        bills_degree = pd.Series(dtype=np.int64)

    collaborates_degree: dict[str, int] = defaultdict(int)
    if not frame.empty and COL_BENE_ID in frame.columns:
        bene_providers = frame.groupby(COL_BENE_ID)[COL_PROVIDER].apply(
            lambda series: sorted({str(p) for p in series.unique()})
        )
        pair_shared: dict[tuple[str, str], int] = defaultdict(int)
        for providers in bene_providers:
            if len(providers) < 2:
                continue
            for left, right in combinations(providers, 2):
                pair_shared[(left, right)] += 1
        for (left, right), shared in pair_shared.items():
            if shared >= min_shared_beneficiaries:
                collaborates_degree[left] += 1
                collaborates_degree[right] += 1

    rows: list[dict[str, object]] = []
    for provider in ordered_providers:
        treats = int(treats_degree.get(provider, 0))
        bills = int(bills_degree.get(provider, 0))
        collaborates = int(collaborates_degree.get(provider, 0))
        rows.append(
            {
                COL_PROVIDER: provider,
                "cent_treats_degree": treats,
                "cent_bills_degree": bills,
                "cent_collaborates_degree": collaborates,
                "cent_total_degree": treats + bills + collaborates,
            }
        )
    return pd.DataFrame(rows)


def _network_centralities(network: nx.Graph, ordered_providers: list[str]) -> pd.DataFrame:
    """Compute global centrality metrics for providers in ``ordered_providers``."""
    if network.number_of_nodes() == 0:
        zeros = {provider: 0.0 for provider in ordered_providers}
        pagerank = betweenness = clustering = eigenvector = zeros
    else:
        pagerank = nx.pagerank(network, alpha=0.85, max_iter=100)
        betweenness = nx.betweenness_centrality(network, normalized=True)
        clustering = nx.clustering(network)
        try:
            eigenvector = nx.eigenvector_centrality(network, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {node: 0.0 for node in network.nodes}

    rows: list[dict[str, object]] = []
    for provider in ordered_providers:
        rows.append(
            {
                COL_PROVIDER: provider,
                "cent_pagerank": float(pagerank.get(provider, 0.0)),
                "cent_betweenness": float(betweenness.get(provider, 0.0)),
                "cent_clustering": float(clustering.get(provider, 0.0)),
                "cent_eigenvector": float(eigenvector.get(provider, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def compute_centrality_from_claims(
    claims: pd.DataFrame,
    provider_ids: list[str],
    *,
    min_shared_beneficiaries: int = 2,
) -> pd.DataFrame:
    """
    Compute graph centrality features from fold-safe claims for any provider split.

    Train providers use train-fold claims only; validation providers use their
    own val-fold claims so inference does not depend on val nodes in saved graphs.
    """
    ordered = _ordered_provider_ids(provider_ids)
    if not ordered:
        return pd.DataFrame(columns=[COL_PROVIDER, *CENTRALITY_FEATURE_COLUMNS])

    subset = claims[claims[COL_PROVIDER].astype(str).isin(ordered)].copy()
    network = build_provider_network_from_claims(
        subset,
        min_shared_beneficiaries=min_shared_beneficiaries,
    )
    degrees = _degree_from_claims(
        subset,
        ordered,
        min_shared_beneficiaries=min_shared_beneficiaries,
    )
    centralities = _network_centralities(network, ordered)
    features = degrees.merge(centralities, on=COL_PROVIDER, how="left").fillna(0.0)
    logger.info(
        "Computed claims-based centrality for %s providers (%s claims)",
        len(features),
        len(subset),
    )
    return features
