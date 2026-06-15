"""Integration tests for graph construction pipeline."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hgad_cms.graph.constants import SCHEMA_V1_1
from hgad_cms.graph.io import load_hetero_graph
from hgad_cms.graph.validator import validate_hetero_graph


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_graphs_module():
    return _load_script_module("03_build_graphs.py")


@pytest.fixture
def create_splits_module():
    return _load_script_module("02_create_splits.py")


def test_pipeline_mini_dataset(
    tmp_path,
    sample_inpatient,
    sample_outpatient,
    sample_beneficiaries,
    sample_providers,
    build_graphs_module,
):
    processed = tmp_path / "processed"
    processed.mkdir()
    from hgad_cms.data.merger import merge_claims, merge_provider_labels

    claims = merge_claims(sample_inpatient, sample_outpatient)
    claims = merge_provider_labels(claims, sample_providers)
    providers = sample_providers.copy()
    providers["fraud_label"] = providers["PotentialFraud"].map({"No": 0, "Yes": 1})
    claims.to_parquet(processed / "claims.parquet", index=False)
    providers.to_parquet(processed / "providers.parquet", index=False)
    sample_beneficiaries.to_parquet(processed / "beneficiaries.parquet", index=False)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    fold_payload = {
        "fold_id": 0,
        "train_provider_ids": ["PRV1", "PRV2"],
        "val_provider_ids": ["PRV3"],
        "n_train": 2,
        "n_val": 1,
    }
    with (splits_dir / "fold_0.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_payload, handle)

    graphs_dir = tmp_path / "graphs"
    build_graphs_module.run_build_graphs(
        processed_dir=processed,
        splits_dir=splits_dir,
        graphs_dir=graphs_dir,
        schema_name="v1.1-no_pp",
        folds=[0],
        build_reference=False,
        skip_validation=False,
        n_folds=1,
    )

    graph_dir = graphs_dir / "v1.1-no_pp" / "fold_0"
    graph, manifest = load_hetero_graph(graph_dir)
    assert manifest["gate_g2_passed"] is True
    report = validate_hetero_graph(graph, SCHEMA_V1_1, reference_mode=False)
    assert report.passed is True


def test_g2_reference_graph_v1_1_real_corpus(build_graphs_module, create_splits_module, tmp_path):
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (processed / "claims.parquet").is_file():
        pytest.skip("Processed CMS data not available")

    splits_dir = tmp_path / "splits"
    graphs_dir = tmp_path / "graphs"
    create_splits_module.run_create_splits(processed, splits_dir)

    build_graphs_module.run_build_graphs(
        processed_dir=processed,
        splits_dir=splits_dir,
        graphs_dir=graphs_dir,
        schema_name="v1.1",
        folds=[0],
        build_reference=True,
        skip_validation=False,
    )

    ref_dir = graphs_dir / "v1.1" / "reference"
    _, manifest = load_hetero_graph(ref_dir)
    assert manifest["gate_g2_passed"] is True
    assert manifest["node_counts"]["provider"] == 5410
    edge_counts = manifest["edge_counts"]
    treats_key = "provider|treats|beneficiary"
    collab_key = "provider|collaborates|provider"
    assert edge_counts[treats_key] == 363_300
    assert abs(edge_counts[collab_key] - 75_604) <= 757

    with (ref_dir / "graph_manifest.json").open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["gate_g2_passed"] is True
