# Experiment Journal

Chronological record of all experiment runs, configuration changes, and observations.

---

## Entry Template

```
Date:
Run ID:
Git Commit:
Experiment Name:
Model:
Dataset Split:
Configuration:
Metrics:
Observations:
Issues:
Next Action:
```

---

### preprocess_v1

```
Date: 2026-06-06
Run ID: preprocess_v1
Git Commit: (uncommitted)
Experiment Name: Phase 1 — CMS data preprocessing
Model: N/A (data pipeline)
Dataset Split: Full train corpus (no CV split yet)
Configuration:
  - raw_dir: datasets/Healthcare Provider Fraud Detection Analysis
  - processed_dir: data/processed
  - reimbursement_clip_quantile: 0.999
  - skip_validation: false
Metrics:
  - providers: 5410
  - claims: 558211
  - beneficiaries: 138556
  - fraud_providers: 506
  - fraud_rate: 0.0935
  - gate_g1_passed: true
Observations:
  - Inpatient/outpatient claims merged with is_inpatient flag
  - Provider labels joined at claim level; all claims labeled
  - Join rates: provider 100%, beneficiary ≥99%
Issues: None
Next Action: Proceed to Phase 2 — graph construction (Gate G2)
```

### graph_construct_v1_1

```
Date: 2026-06-11
Run ID: graph_construct_v1_1
Git Commit: (uncommitted)
Experiment Name: Phase 2 — schema_v1.1 heterogeneous graph construction
Model: N/A (graph pipeline)
Dataset Split: Provider-disjoint stratified 5-fold CV (seed=42)
Configuration:
  - schema: v1.1
  - pp_min_shared_beneficiaries: 2
  - physician_min_claim_refs: 1
  - ablations registered: v1.1-no_pp, v1.1-pp_t1, v1.1-pp_t5, v1b, v2, v4
Metrics (reference graph):
  - providers: 5410
  - beneficiaries: 138556
  - physicians: 100737
  - treats edges: 363300
  - bills_with edges: 109339
  - collaborates edges: 75604
  - gate_g2_passed: true
Observations:
  - Fold-safe construction excludes val providers from train graph nodes
  - No PotentialFraud / fraud_label in node feature columns
  - PP edges use log1p(n_shared) weight and Jaccard edge feature
Issues: None
Next Action: Phase 3 — baseline and model training
```

### baselines_v1

```
Date: 2026-06-11
Run ID: baselines_v1
Git Commit: (uncommitted)
Experiment Name: Phase 3 — baseline model benchmark
Model: LR / RF / CatBoost / RF+Centrality
Dataset Split: Provider-disjoint stratified 5-fold CV
Configuration:
  - feature_source: train-fold provider aggregations
  - primary_metric: auprc
Metrics:
  - logistic_regression_auprc: 0.6810 +/- 0.0389
  - catboost_auprc: 0.6615 +/- 0.0507
  - rf_centrality_auprc: 0.6561 +/- 0.0591
  - random_forest_auprc: 0.6517 +/- 0.0556
Observations:
  - Per-fold metrics and confusion matrices saved under artifacts/results/baselines/
Issues: None
Next Action: Phase 4 — GNN model training
```
