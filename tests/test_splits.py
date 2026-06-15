"""Tests for hgad_cms.graph.splits."""

import pandas as pd
import pytest

from hgad_cms.constants import COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.exceptions import SplitError
from hgad_cms.graph.splits import create_provider_disjoint_splits, load_fold_split, save_splits


def test_create_splits_stratified(sample_providers):
    providers = sample_providers.copy()
    providers[COL_FRAUD_LABEL] = providers["PotentialFraud"].map({"No": 0, "Yes": 1})
    with pytest.raises(SplitError):
        create_provider_disjoint_splits(providers, n_folds=5)


def test_save_and_load_fold(tmp_path):
    providers = pd.DataFrame(
        {
            COL_PROVIDER: [f"P{i}" for i in range(50)],
            COL_FRAUD_LABEL: [1 if i < 5 else 0 for i in range(50)],
        }
    )
    splits = create_provider_disjoint_splits(providers, n_folds=5, seed=42)
    save_splits(splits, tmp_path, n_folds=5, seed=42)
    loaded = load_fold_split(tmp_path, 0)
    assert loaded.fold_id == 0
    assert loaded.n_train + loaded.n_val == 50
    assert set(loaded.train_provider_ids).isdisjoint(loaded.val_provider_ids)
