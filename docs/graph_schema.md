# Graph Construction Specification — Phase 2

**Project:** Hybrid Explainable Heterogeneous Graph Anomaly Detection for CMS Provider Fraud  
**Status:** Specification (pre-implementation)  
**Gate:** G2 — Graph Construction Validation  
**Data source:** `data/processed/` (Gate G1 passed)

**Version:** `graph_schema_v1.1` (updated after publication-focused provider–provider edge review)  
**Last updated:** 2026-06-08

---

## Primary schema (`schema_v1.1`)

### Node types
`(provider, beneficiary, physician)` — unchanged from v1.0

### Edge types

| Relation | Direction | Construction |
|---|---|---|
| `treats` / `treated_by` | provider ↔ beneficiary | Unique (Provider, BeneID) from **train-fold claims** |
| `bills_with` / `billed_by` | provider ↔ physician | Unique (Provider, Physician) from train-fold claims |
| `collaborates` / `collaborated_by` | provider ↔ provider | Pairs sharing **≥ 2 beneficiaries** in train-fold claims |

### Provider–provider edge policy (v1.1 change)

- **Inclusion threshold:** `n_shared_beneficiaries ≥ 2` (hard filter)
- **Edge weight (message passing):** `log1p(n_shared_beneficiaries)`
- **Edge features (stored, not hard-filter):** `n_shared_beneficiaries`, `jaccard_panel_similarity`, `log1p(n_shared_claims)`
- **Excluded:** single-shared-beneficiary pairs (55% of unfiltered edges; treated as noise)

### Expected full-corpus counts (reference, pre-split)

| Metric | v1.0 (≥1 shared) | v1.1 (≥2 shared) |
|---|---:|---:|
| Provider–provider pairs | 168,104 | 75,604 |
| Cross-label edge fraction | 37.5% | 40.5% |
| Label homophily | 0.625 | 0.595 |

### Fold-safe construction (mandatory)
All edges and aggregations computed from **train-fold claims only** per CV fold. Validation/test providers excluded from train graph node set.

---

See Phase 2 chat specification for full sections A–E, leakage analysis, and G2 requirements.
