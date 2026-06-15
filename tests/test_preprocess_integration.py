"""Integration tests for scripts/01_preprocess.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def preprocess_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_module"] = module
    spec.loader.exec_module(module)
    return module


def test_run_preprocess_mini_dataset_skipping_g1(preprocess_module, cms_raw_dir: Path, tmp_path: Path):
    out = tmp_path / "processed"
    manifest = preprocess_module.run_preprocess(
        raw_dir=cms_raw_dir,
        processed_dir=out,
        skip_validation=True,
    )
    assert (out / "claims.parquet").exists()
    assert (out / "providers.parquet").exists()
    assert (out / "beneficiaries.parquet").exists()
    assert manifest["row_counts"]["claims"] == 3


def test_run_preprocess_real_dataset_g1():
    raw_dir = Path(__file__).resolve().parents[2] / "datasets" / "Healthcare Provider Fraud Detection Analysis"
    if not raw_dir.is_dir():
        pytest.skip("CMS dataset not available in workspace")

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess_module_real", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_module_real"] = module
    spec.loader.exec_module(module)

    out = Path(__file__).resolve().parents[1] / "data" / "processed_test_run"
    manifest = module.run_preprocess(raw_dir=raw_dir, processed_dir=out, skip_validation=False)
    assert manifest["gate_g1_passed"] is True
    assert manifest["row_counts"]["claims"] == 558_211
    with (out / "manifest.json").open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["gate_g1_passed"] is True
