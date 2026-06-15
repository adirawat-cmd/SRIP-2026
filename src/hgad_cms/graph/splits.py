"""
Provider-disjoint stratified cross-validation splits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hgad_cms.constants import COL_FRAUD_LABEL, COL_PROVIDER
from hgad_cms.exceptions import SplitError
from hgad_cms.graph.constants import (
    DEFAULT_N_FOLDS,
    DEFAULT_SPLIT_SEED,
    FOLD_SPLIT_TEMPLATE,
    SPLIT_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldSplit:
    """Single CV fold provider partition."""

    fold_id: int
    train_provider_ids: list[str]
    val_provider_ids: list[str]

    @property
    def n_train(self) -> int:
        return len(self.train_provider_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_provider_ids)


def create_provider_disjoint_splits(
    providers: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[FoldSplit]:
    """
    Create stratified provider-disjoint K-fold splits.

    Parameters
    ----------
    providers:
        Provider table with ``Provider`` and ``fraud_label`` columns.
    n_folds:
        Number of CV folds.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[FoldSplit]
        One split per fold.

    Raises
    ------
    SplitError
        If providers table is invalid or folds lack fraud examples.
    """
    required = {COL_PROVIDER, COL_FRAUD_LABEL}
    missing = required - set(providers.columns)
    if missing:
        raise SplitError(f"Providers table missing columns: {sorted(missing)}")

    frame = providers[[COL_PROVIDER, COL_FRAUD_LABEL]].drop_duplicates(subset=[COL_PROVIDER])
    if frame[COL_PROVIDER].duplicated().any():
        raise SplitError("Duplicate provider IDs in providers table")

    provider_ids = frame[COL_PROVIDER].astype(str).values
    labels = frame[COL_FRAUD_LABEL].astype(int).values

    if len(provider_ids) < n_folds:
        raise SplitError(f"Not enough providers ({len(provider_ids)}) for {n_folds} folds")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: list[FoldSplit] = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(provider_ids, labels)):
        train_ids = sorted(provider_ids[train_idx].tolist())
        val_ids = sorted(provider_ids[val_idx].tolist())
        overlap = set(train_ids).intersection(val_ids)
        if overlap:
            raise SplitError(f"Fold {fold_id} has provider overlap between train and val")

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        if train_labels.sum() == 0 or val_labels.sum() == 0:
            raise SplitError(f"Fold {fold_id} missing fraud providers in train or val")

        splits.append(
            FoldSplit(
                fold_id=fold_id,
                train_provider_ids=train_ids,
                val_provider_ids=val_ids,
            )
        )
        logger.info(
            "Fold %s: train=%s val=%s fraud_train=%s fraud_val=%s",
            fold_id,
            len(train_ids),
            len(val_ids),
            int(train_labels.sum()),
            int(val_labels.sum()),
        )

    return splits


def fold_to_dict(fold: FoldSplit) -> dict[str, Any]:
    """Serialize a fold split to JSON-compatible dict."""
    return {
        "fold_id": fold.fold_id,
        "train_provider_ids": fold.train_provider_ids,
        "val_provider_ids": fold.val_provider_ids,
        "n_train": fold.n_train,
        "n_val": fold.n_val,
    }


def save_splits(
    splits: list[FoldSplit],
    output_dir: Path,
    *,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Write per-fold JSON files and split manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_paths: dict[str, str] = {}
    for fold in splits:
        path = output_dir / FOLD_SPLIT_TEMPLATE.format(fold=fold.fold_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(fold_to_dict(fold), handle, indent=2)
        fold_paths[str(fold.fold_id)] = str(path.resolve())

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_folds": n_folds,
        "seed": seed,
        "fold_paths": fold_paths,
        "provider_counts": {
            str(f.fold_id): {"train": f.n_train, "val": f.n_val} for f in splits
        },
    }
    manifest_path = output_dir / SPLIT_MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Wrote split manifest: %s", manifest_path)
    return manifest


def load_fold_split(splits_dir: Path, fold_id: int) -> FoldSplit:
    """Load a single fold split from disk."""
    path = splits_dir / FOLD_SPLIT_TEMPLATE.format(fold=fold_id)
    if not path.is_file():
        raise SplitError(f"Missing fold split file: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return FoldSplit(
        fold_id=int(data["fold_id"]),
        train_provider_ids=[str(p) for p in data["train_provider_ids"]],
        val_provider_ids=[str(p) for p in data["val_provider_ids"]],
    )
