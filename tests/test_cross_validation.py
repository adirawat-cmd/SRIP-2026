"""Tests for hgad_cms.evaluation.cross_validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hgad_cms.baselines.logistic_regression import LogisticRegressionBaseline
from hgad_cms.constants import COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.evaluation.cross_validation import (
    build_comparison_table,
    fit_feature_artifacts,
    run_provider_disjoint_cv,
    save_baseline_result,
    validate_gate_g3,
)


class _DummyBaseline:
    name = "dummy"

    def fit(self, X_train, y_train):
        self._mean = float(np.mean(y_train))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self._mean)


def _synthetic_processed(tmp_path: Path) -> tuple[Path, Path, Path]:
    providers = pd.DataFrame(
        {
            COL_PROVIDER: [f"P{i}" for i in range(20)],
            COL_FRAUD_LABEL: [1 if i < 2 else 0 for i in range(20)],
        }
    )
    claims = pd.DataFrame(
        {
            COL_PROVIDER: [f"P{i % 20}" for i in range(100)],
            "BeneID": [f"B{i}" for i in range(100)],
            "ClaimID": [f"C{i}" for i in range(100)],
            "InscClaimAmtReimbursed": np.random.default_rng(0).random(100) * 1000,
            "is_inpatient": [False] * 100,
            "AttendingPhysician": [f"PHY{i % 5}" for i in range(100)],
            "OperatingPhysician": [None] * 100,
            "OtherPhysician": [None] * 100,
            "ClmDiagnosisCode_1": ["D1"] * 100,
            "ClmProcedureCode_1": ["PR1"] * 100,
        }
    )
    beneficiaries = pd.DataFrame(
        {
            "BeneID": [f"B{i}" for i in range(100)],
            "Gender": ["F"] * 100,
            "gender_norm": [0] * 100,
            "Race": [1] * 100,
            "RenalDiseaseIndicator": [0] * 100,
            "State": [1] * 100,
            "County": [1] * 100,
            "NoOfMonths_PartACov": [12] * 100,
            "NoOfMonths_PartBCov": [12] * 100,
            **{
                f"ChronicCond_{c}": [2] * 100
                for c in [
                    "Alzheimer",
                    "Heartfailure",
                    "KidneyDisease",
                    "Cancer",
                    "ObstrPulmonary",
                    "Depression",
                    "Diabetes",
                    "IschemicHeart",
                    "Osteoporasis",
                    "rheumatoidarthritis",
                    "stroke",
                ]
            },
        }
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    claims.to_parquet(processed / "claims.parquet", index=False)
    providers.to_parquet(processed / "providers.parquet", index=False)
    beneficiaries.to_parquet(processed / "beneficiaries.parquet", index=False)

    splits = tmp_path / "splits"
    splits.mkdir()
    for fold in range(2):
        train = [f"P{i}" for i in range(20) if i % 2 == fold]
        val = [f"P{i}" for i in range(20) if i % 2 != fold]
        pd.Series(train).to_json(splits / f"fold_{fold}.json", orient="values")  # wrong format

    # proper fold json
    import json

    for fold in range(2):
        payload = {
            "fold_id": fold,
            "train_provider_ids": [f"P{i}" for i in range(20) if i % 2 == fold],
            "val_provider_ids": [f"P{i}" for i in range(20) if i % 2 != fold],
            "n_train": 10,
            "n_val": 10,
        }
        with (splits / f"fold_{fold}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    graphs = tmp_path / "graphs" / "v1.1"
    for fold in range(2):
        fold_dir = graphs / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        nodes = pd.DataFrame({COL_PROVIDER: [f"P{i}" for i in range(20)], "node_idx": range(20)})
        nodes.to_parquet(fold_dir / "nodes_provider.parquet", index=False)
        edges = pd.DataFrame({"src_idx": [0], "dst_idx": [1], "n_shared_beneficiaries": [2]})
        edges.to_parquet(fold_dir / "edges__provider__collaborates__provider.parquet", index=False)
        edges.to_parquet(fold_dir / "edges__provider__treats__beneficiary.parquet", index=False)
        edges.to_parquet(fold_dir / "edges__provider__bills_with__physician.parquet", index=False)
        with (fold_dir / "graph_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"schema_name": "v1.1", "fold_id": fold}, handle)

    return processed, splits, graphs


def test_fit_feature_artifacts_no_leakage(tmp_path):
    processed, _, _ = _synthetic_processed(tmp_path)
    claims = pd.read_parquet(processed / "claims.parquet")
    beneficiaries = pd.read_parquet(processed / "beneficiaries.parquet")
    artifacts = fit_feature_artifacts(claims, beneficiaries, [f"P{i}" for i in range(10)])
    assert len(artifacts.feature_columns) > 0
    assert "PotentialFraud" not in artifacts.feature_columns


def test_run_provider_disjoint_cv_dummy(tmp_path):
    processed, splits, graphs = _synthetic_processed(tmp_path)
    result = run_provider_disjoint_cv(
        _DummyBaseline,
        processed_dir=processed,
        splits_dir=splits,
        graphs_dir=graphs.parent,
        schema_name="v1.1",
        n_folds=2,
    )
    assert len(result.fold_results) == 2
    assert "auprc" in result.summary


def test_run_provider_disjoint_cv_rf_centrality(tmp_path):
    processed, splits, graphs = _synthetic_processed(tmp_path)
    from hgad_cms.baselines.rf_centrality import RFCentralityBaseline

    result = run_provider_disjoint_cv(
        RFCentralityBaseline,
        processed_dir=processed,
        splits_dir=splits,
        graphs_dir=graphs.parent,
        schema_name="v1.1",
        n_folds=2,
        use_centrality=True,
    )
    assert len(result.fold_results) == 2
    assert result.fold_results[0].metrics.auprc >= 0.0


def test_compute_centrality_from_claims_val_disjoint(tmp_path):
    from hgad_cms.baselines.centrality_features import compute_centrality_from_claims

    claims = pd.DataFrame(
        {
            COL_PROVIDER: ["P0", "P0", "P1", "P2", "P2"],
            "BeneID": ["B0", "B1", "B1", "B2", "B3"],
            "AttendingPhysician": ["PHY0", "PHY0", "PHY1", "PHY2", "PHY2"],
        }
    )
    train = compute_centrality_from_claims(
        claims[claims[COL_PROVIDER].isin(["P0", "P1"])],
        ["P0", "P1"],
    )
    val = compute_centrality_from_claims(
        claims[claims[COL_PROVIDER].isin(["P2"])],
        ["P2"],
    )
    assert len(train) == 2
    assert len(val) == 1
    assert val.iloc[0][COL_PROVIDER] == "P2"


def test_validate_gate_g3():
    class Result:
        def __init__(self, name, auprc):
            self.model_name = name
            self.summary = {"auprc": {"mean": auprc, "std": 0.01}}

    results = [Result("a", 0.15), Result("b", 0.12), Result("c", 0.10), Result("d", 0.094)]
    report = validate_gate_g3(results)
    assert report["passed"] is True
