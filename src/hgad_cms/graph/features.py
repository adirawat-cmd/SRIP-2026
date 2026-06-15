"""
Fold-safe node and edge feature engineering for heterogeneous graphs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hgad_cms.constants import (
    CHRONIC_CONDITION_COLUMNS,
    COL_BENE_ID,
    COL_CLAIM_DURATION_DAYS,
    COL_FRAUD_LABEL,
    COL_GENDER_NORM,
    COL_IS_INPATIENT,
    COL_PROVIDER,
    COL_REIMBURSED,
    DIAGNOSIS_COLUMNS,
    PROCEDURE_COLUMNS,
)
from hgad_cms.exceptions import GraphBuildError
from hgad_cms.graph.constants import (
    FORBIDDEN_FEATURE_COLUMNS,
    PHYSICIAN_ROLE_COLUMNS,
    TOP_DIAGNOSIS_FEATURES,
    TOP_PROCEDURE_FEATURES,
)

logger = logging.getLogger(__name__)


def assert_no_leakage_columns(columns: Iterable[str]) -> None:
    """Raise if forbidden label columns appear in feature column list."""
    bad = FORBIDDEN_FEATURE_COLUMNS.intersection(set(columns))
    if bad:
        raise GraphBuildError(f"Leakage columns present in features: {sorted(bad)}")


def _safe_log1p(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.log1p(np.clip(arr, a_min=0.0, a_max=None))


def compute_top_codes(
    claims: pd.DataFrame,
    code_columns: tuple[str, ...],
    top_k: int,
) -> list[str]:
    """Return top-K most frequent diagnosis or procedure codes in train claims."""
    present = [c for c in code_columns if c in claims.columns]
    if not present:
        return []
    melted = claims[present].melt(value_name="code").dropna()
    codes = melted["code"].astype(str).str.strip()
    codes = codes[~codes.isin({"", "nan", "NA", "None"})]
    if codes.empty:
        return []
    return codes.value_counts().head(top_k).index.tolist()


def build_code_rate_features(
    claims: pd.DataFrame,
    entity_col: str,
    code_columns: tuple[str, ...],
    top_codes: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Build per-entity code-rate feature matrix."""
    if not top_codes:
        return pd.DataFrame({entity_col: claims[entity_col].astype(str).unique()})

    present = [c for c in code_columns if c in claims.columns]
    if not present:
        return pd.DataFrame({entity_col: claims[entity_col].astype(str).unique()})

    melted = claims[[entity_col, *present]].melt(
        id_vars=entity_col,
        value_name="code",
    )
    melted[entity_col] = melted[entity_col].astype(str)
    melted = melted.dropna(subset=["code"])
    melted["code"] = melted["code"].astype(str).str.strip()
    melted = melted[melted["code"].isin(top_codes)]

    counts = (
        melted.groupby([entity_col, "code"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_codes, fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    rates = counts.div(totals, axis=0).fillna(0.0)
    rates.columns = [f"{prefix}_{code}" for code in top_codes]
    return rates.reset_index()


def build_provider_features(
    claims: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    provider_ids: Iterable[str],
    top_diagnosis_codes: list[str],
    top_procedure_codes: list[str],
) -> pd.DataFrame:
    """
    Compute provider node features from train-fold claims only.

    Parameters
    ----------
    claims:
        Training-fold claim records.
    beneficiaries:
        Beneficiary table for panel demographic aggregation.
    provider_ids:
        Provider node IDs to include.
    top_diagnosis_codes:
        Diagnosis codes for rate features.
    top_procedure_codes:
        Procedure codes for rate features.

    Returns
    -------
    pd.DataFrame
        One row per provider with numeric feature columns.
    """
    providers = sorted({str(p) for p in provider_ids})
    base = pd.DataFrame({COL_PROVIDER: providers})

    if claims.empty:
        raise GraphBuildError("Cannot build provider features from empty claims")

    grouped = claims.groupby(COL_PROVIDER, as_index=False)
    agg = grouped.agg(
        total_claims=("ClaimID", "count"),
        total_reimb=(COL_REIMBURSED, "sum"),
        mean_reimb=(COL_REIMBURSED, "mean"),
        std_reimb=(COL_REIMBURSED, "std"),
        unique_benes=(COL_BENE_ID, "nunique"),
        inpatient_claims=(COL_IS_INPATIENT, "sum"),
    )
    agg[COL_PROVIDER] = agg[COL_PROVIDER].astype(str)
    agg["std_reimb"] = agg["std_reimb"].fillna(0.0)
    agg["inpatient_ratio"] = agg["inpatient_claims"] / agg["total_claims"].clip(lower=1)

    if COL_CLAIM_DURATION_DAYS in claims.columns:
        duration = grouped[COL_CLAIM_DURATION_DAYS].mean().rename(
            columns={COL_CLAIM_DURATION_DAYS: "mean_duration"}
        )
        duration[COL_PROVIDER] = duration[COL_PROVIDER].astype(str)
        agg = agg.merge(duration, on=COL_PROVIDER, how="left")
        agg["mean_duration"] = agg["mean_duration"].fillna(0.0)
    else:
        agg["mean_duration"] = 0.0

    role_counts = {role: [] for role in PHYSICIAN_ROLE_COLUMNS}
    for role in PHYSICIAN_ROLE_COLUMNS:
        if role in claims.columns:
            rc = (
                claims.dropna(subset=[role])
                .groupby(COL_PROVIDER)[role]
                .nunique()
                .rename("count")
                .reset_index()
            )
            rc[COL_PROVIDER] = rc[COL_PROVIDER].astype(str)
            role_counts[role] = rc

    merged = base.merge(agg, on=COL_PROVIDER, how="left")
    for role in PHYSICIAN_ROLE_COLUMNS:
        col_name = f"unique_{role.lower()}"
        rc = role_counts[role]
        if isinstance(rc, pd.DataFrame) and not rc.empty:
            merged = merged.merge(
                rc.rename(columns={"count": col_name}),
                on=COL_PROVIDER,
                how="left",
            )
        else:
            merged[col_name] = 0.0

    merged["unique_physicians"] = merged[
        [f"unique_{r.lower()}" for r in PHYSICIAN_ROLE_COLUMNS]
    ].sum(axis=1)

    dx_rates = build_code_rate_features(
        claims, COL_PROVIDER, DIAGNOSIS_COLUMNS, top_diagnosis_codes, "dx_rate"
    )
    proc_rates = build_code_rate_features(
        claims, COL_PROVIDER, PROCEDURE_COLUMNS, top_procedure_codes, "proc_rate"
    )
    merged = merged.merge(dx_rates, on=COL_PROVIDER, how="left")
    merged = merged.merge(proc_rates, on=COL_PROVIDER, how="left")

    bene_cols = [COL_BENE_ID, "DOB", *CHRONIC_CONDITION_COLUMNS]
    if COL_GENDER_NORM in beneficiaries.columns:
        bene_cols.insert(1, COL_GENDER_NORM)
    elif "Gender" in beneficiaries.columns:
        bene_cols.insert(1, "Gender")
    bene_subset = beneficiaries[[c for c in bene_cols if c in beneficiaries.columns]].copy()
    bene_subset[COL_BENE_ID] = bene_subset[COL_BENE_ID].astype(str)

    if "DOB" in bene_subset.columns:
        bene_subset["age"] = (
            pd.Timestamp("2009-12-31") - pd.to_datetime(bene_subset["DOB"], errors="coerce")
        ).dt.days / 365.25
    else:
        bene_subset["age"] = np.nan

    claim_benes = claims[[COL_PROVIDER, COL_BENE_ID]].drop_duplicates()
    claim_benes[COL_PROVIDER] = claim_benes[COL_PROVIDER].astype(str)
    claim_benes[COL_BENE_ID] = claim_benes[COL_BENE_ID].astype(str)
    panel = claim_benes.merge(bene_subset, on=COL_BENE_ID, how="left")
    if COL_GENDER_NORM not in panel.columns and "Gender" in panel.columns:
        from hgad_cms.data.cleaner import normalize_gender

        panel[COL_GENDER_NORM] = normalize_gender(panel["Gender"])

    chronic_cols = [c for c in CHRONIC_CONDITION_COLUMNS if c in panel.columns]
    for col in chronic_cols + ([COL_GENDER_NORM] if COL_GENDER_NORM in panel.columns else []):
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    agg_dict: dict[str, tuple[str, str]] = {
        "panel_mean_age": ("age", "mean"),
    }
    if COL_GENDER_NORM in panel.columns:
        agg_dict["panel_mean_gender"] = (COL_GENDER_NORM, "mean")
    if chronic_cols:
        panel["chronic_mean"] = panel[chronic_cols].mean(axis=1)
        agg_dict["panel_mean_chronic"] = ("chronic_mean", "mean")

    panel_agg = panel.groupby(COL_PROVIDER, as_index=False).agg(**agg_dict)
    merged = merged.merge(panel_agg, on=COL_PROVIDER, how="left")
    numeric = merged.drop(columns=[COL_PROVIDER]).fillna(0.0)
    assert_no_leakage_columns(numeric.columns)
    merged = pd.concat([merged[[COL_PROVIDER]], numeric], axis=1)
    logger.info("Built provider features: %s providers, %s dims", len(merged), len(numeric.columns))
    return merged


def build_beneficiary_features(
    beneficiaries: pd.DataFrame,
    beneficiary_ids: Iterable[str],
) -> pd.DataFrame:
    """Build beneficiary node features (no annual reimbursement fields)."""
    ids = sorted({str(b) for b in beneficiary_ids})
    frame = beneficiaries[beneficiaries[COL_BENE_ID].astype(str).isin(ids)].copy()
    frame[COL_BENE_ID] = frame[COL_BENE_ID].astype(str)

    if "DOB" in frame.columns:
        frame["age"] = (
            pd.Timestamp("2009-12-31") - pd.to_datetime(frame["DOB"], errors="coerce")
        ).dt.days / 365.25
    else:
        frame["age"] = np.nan

    keep = [COL_BENE_ID, "age"]
    if COL_GENDER_NORM in frame.columns:
        keep.append(COL_GENDER_NORM)
    elif "Gender" in frame.columns:
        keep.append("Gender")
    keep.extend(
        [
            "Race",
            "RenalDiseaseIndicator",
            "State",
            "County",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            *CHRONIC_CONDITION_COLUMNS,
        ]
    )
    keep = [c for c in keep if c in frame.columns]
    out = frame[keep].copy()
    for col in out.columns:
        if col != COL_BENE_ID:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.fillna(0.0)
    assert_no_leakage_columns(out.columns)
    logger.info("Built beneficiary features: %s nodes, %s dims", len(out), len(out.columns) - 1)
    return out


def build_physician_features(
    claims: pd.DataFrame,
    physician_ids: Iterable[str],
) -> pd.DataFrame:
    """Build physician node features from train-fold claims."""
    physicians = sorted({str(p) for p in physician_ids})
    if not physicians:
        return pd.DataFrame(columns=["Physician"])

    counts: dict[str, dict[str, float]] = {
        phys: {
            "claim_refs": 0.0,
            "unique_providers": 0.0,
            "unique_beneficiaries": 0.0,
            "total_reimb": 0.0,
            "mean_reimb": 0.0,
            "n_attending": 0.0,
            "n_operating": 0.0,
            "n_other": 0.0,
        }
        for phys in physicians
    }
    phys_set = set(physicians)

    role_map = {
        "AttendingPhysician": "n_attending",
        "OperatingPhysician": "n_operating",
        "OtherPhysician": "n_other",
    }
    provider_sets: dict[str, set[str]] = defaultdict(set)
    bene_sets: dict[str, set[str]] = defaultdict(set)
    reimb_sums: dict[str, float] = defaultdict(float)
    reimb_counts: dict[str, int] = defaultdict(int)

    for role_col, count_key in role_map.items():
        if role_col not in claims.columns:
            continue
        sub = claims.dropna(subset=[role_col])[[COL_PROVIDER, COL_BENE_ID, COL_REIMBURSED, role_col]].copy()
        sub[role_col] = sub[role_col].astype(str)
        sub = sub[sub[role_col].isin(phys_set)]
        for phys, group in sub.groupby(role_col):
            counts[phys]["claim_refs"] += len(group)
            counts[phys][count_key] += len(group)
            provider_sets[phys].update(group[COL_PROVIDER].astype(str).tolist())
            bene_sets[phys].update(group[COL_BENE_ID].astype(str).tolist())
            reimb_sums[phys] += float(group[COL_REIMBURSED].sum())
            reimb_counts[phys] += len(group)

    rows: list[dict[str, object]] = []
    for phys in physicians:
        stats = counts[phys]
        total_refs = max(int(stats["claim_refs"]), 1)
        rows.append(
            {
                "Physician": phys,
                "claim_refs": int(stats["claim_refs"]),
                "unique_providers": len(provider_sets.get(phys, set())),
                "unique_beneficiaries": len(bene_sets.get(phys, set())),
                "total_reimb": reimb_sums.get(phys, 0.0),
                "mean_reimb": reimb_sums.get(phys, 0.0) / max(reimb_counts.get(phys, 0), 1),
                "frac_attending": stats.get("n_attending", 0) / total_refs,
                "frac_operating": stats.get("n_operating", 0) / total_refs,
                "frac_other": stats.get("n_other", 0) / total_refs,
            }
        )

    out = pd.DataFrame(rows)
    assert_no_leakage_columns(out.columns)
    logger.info("Built physician features: %s nodes, %s dims", len(out), len(out.columns) - 1)
    return out


def build_diagnosis_features(diagnosis_ids: Iterable[str]) -> pd.DataFrame:
    """Identity index feature for diagnosis nodes."""
    ids = sorted({str(d) for d in diagnosis_ids})
    return pd.DataFrame(
        {
            "DiagnosisCode": ids,
            "code_index": np.arange(len(ids), dtype=np.float64),
        }
    )


def build_treats_edge_features(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric edge features for provider-beneficiary edges."""
    out = edge_df.copy()
    out["weight"] = _safe_log1p(out["n_claims"])
    out["log_n_claims"] = _safe_log1p(out["n_claims"])
    out["log_total_reimb"] = _safe_log1p(out["total_reimb"])
    out["mean_reimb"] = out["total_reimb"] / out["n_claims"].clip(lower=1)
    out["inpatient_ratio"] = out["inpatient_claims"] / out["n_claims"].clip(lower=1)
    if "mean_duration" in out.columns:
        out["log_mean_duration"] = _safe_log1p(out["mean_duration"])
    else:
        out["log_mean_duration"] = 0.0
    return out


def build_bills_with_edge_features(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric edge features for provider-physician edges."""
    out = edge_df.copy()
    for col in ("n_attending", "n_operating", "n_other"):
        if col not in out.columns:
            out[col] = 0.0
    out["weight"] = _safe_log1p(out["n_claims"])
    out["log_n_claims"] = _safe_log1p(out["n_claims"])
    out["log_total_reimb"] = _safe_log1p(out["total_reimb"])
    total_refs = (
        out["n_attending"].fillna(0)
        + out["n_operating"].fillna(0)
        + out["n_other"].fillna(0)
    ).clip(lower=1)
    out["frac_attending"] = out["n_attending"].fillna(0) / total_refs
    out["frac_operating"] = out["n_operating"].fillna(0) / total_refs
    out["frac_other"] = out["n_other"].fillna(0) / total_refs
    return out


def build_collaborates_edge_features(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Add PP edge weight and Jaccard features."""
    out = edge_df.copy()
    out["weight"] = _safe_log1p(out["n_shared_beneficiaries"])
    out["log_n_shared_beneficiaries"] = _safe_log1p(out["n_shared_beneficiaries"])
    out["jaccard_panel_similarity"] = out["n_shared_beneficiaries"] / out["panel_union_size"].clip(
        lower=1
    )
    out["log_n_shared_claims"] = _safe_log1p(out.get("n_shared_claims", 0))
    return out


def build_seen_by_edge_features(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Add beneficiary-physician edge features."""
    out = edge_df.copy()
    out["weight"] = _safe_log1p(out["n_claims"])
    out["log_n_claims"] = _safe_log1p(out["n_claims"])
    return out


def build_diagnosed_with_edge_features(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Add provider-diagnosis edge features."""
    out = edge_df.copy()
    out["weight"] = _safe_log1p(out["n_claims"])
    out["log_n_claims"] = _safe_log1p(out["n_claims"])
    return out


def fit_transform_standard_scaler(
    train_matrix: np.ndarray,
    apply_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit StandardScaler on train and optionally transform another matrix."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    if apply_matrix is None:
        return train_scaled, scaler
    return scaler.transform(apply_matrix), scaler


def select_feature_matrix(
    frame: pd.DataFrame,
    id_column: str,
) -> tuple[list[str], np.ndarray]:
    """Extract numeric feature matrix excluding ID and forbidden columns."""
    feature_cols = [
        c
        for c in frame.columns
        if c != id_column and c not in FORBIDDEN_FEATURE_COLUMNS and c != COL_FRAUD_LABEL
    ]
    matrix = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    assert_no_leakage_columns(feature_cols)
    return feature_cols, matrix.astype(np.float32)


def compute_top_code_lists(
    claims: pd.DataFrame,
    top_diagnosis: int = TOP_DIAGNOSIS_FEATURES,
    top_procedure: int = TOP_PROCEDURE_FEATURES,
) -> tuple[list[str], list[str]]:
    """Compute top diagnosis and procedure code lists from train claims."""
    dx = compute_top_codes(claims, DIAGNOSIS_COLUMNS, top_diagnosis)
    proc = compute_top_codes(claims, PROCEDURE_COLUMNS, top_procedure)
    return dx, proc
