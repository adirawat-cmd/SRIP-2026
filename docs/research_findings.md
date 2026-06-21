# Research Findings

Automated record of scientific findings. Updated after each benchmark, ablation, and gate event.

Last synced: 2026-06-21T15:04:07.875235+00:00

---

## F001

**Date:** 2026-06-06  
**Phase:** Phase 1 — Data Pipeline  
**Experiment ID:** preprocess_v1

**Observation:**
Provider-level fraud rate is 9.35% (506/5410 providers).

**Evidence:**
data/processed/manifest.json; G1 validation passed.

**Impact:**
Establishes prevalence floor for AUPRC benchmark (0.0935).

**Follow-up Action:**
Use stratified provider-disjoint CV.

---

## F002

**Date:** 2026-06-11  
**Phase:** Phase 2 — Graph Construction  
**Experiment ID:** graph_construct_v1_1

**Observation:**
schema_v1.1 reference graph: 363K treats, 109K bills_with, 76K collaborates edges.

**Evidence:**
artifacts/published/graphs/v1.1/reference/graph_manifest.json; Gate G2 passed.

**Impact:**
Confirms fold-safe heterogeneous graph artifacts for GNN training.

**Follow-up Action:**
Proceed to tabular and graph baselines.

---

## F003

**Date:** 2026-06-11  
**Phase:** Phase 3 — Baselines  
**Experiment ID:** baselines_v1

**Observation:**
Logistic Regression outperformed CatBoost, RF, and RF+Centrality on mean AUPRC.

**Evidence:**
LR AUPRC 0.6810±0.0389 vs CatBoost 0.6615±0.0507 (artifacts/published/baselines/).

**Impact:**
LR becomes primary tabular benchmark; Gate G3 passed.

**Follow-up Action:**
Test whether graph structure adds value beyond tabular features.

---

## F004

**Date:** 2026-06-11  
**Phase:** Phase 4 — GraphSAGE  
**Experiment ID:** graphsage_eval

**Observation:**
GraphSAGE failed to beat Logistic Regression (5-fold AUPRC 0.6530 vs 0.6810).

**Evidence:**
docs/graphsage_diagnosis.md; artifacts/published/gnn/graphsage_benchmark.json.

**Impact:**
Gate G4 failed; homophilic GraphSAGE insufficient for fraud heterophily.

**Follow-up Action:**
Proceed to relation-aware R-GCN (Phase 5).

---

## F005

**Date:** 2026-06-11  
**Phase:** Phase 4 — GraphSAGE  
**Experiment ID:** graphsage_eval

**Observation:**
Mean neighbor aggregation outperformed max aggregation on fold-0 HPO.

**Evidence:**
Mean AUPRC 0.631 vs max 0.598 across 12 HPO configs.

**Impact:**
Select mean aggregation for future homophilic GNN baselines.

**Follow-up Action:**
Document in graph model design guidelines.

---

## F006

**Date:** 2026-06-11  
**Phase:** Phase 4 — GraphSAGE  
**Experiment ID:** graphsage_eval

**Observation:**
2-layer GraphSAGE outperformed 3-layer on fold-0 HPO.

**Evidence:**
Layer-2 mean AUPRC 0.622 vs layer-3 mean 0.607.

**Impact:**
Prefer shallow GNNs on 4GB GPU constraint; reduces overfitting risk.

**Follow-up Action:**
Cap default depth at 2 layers for HPO quick mode.

---

## F007

**Date:** 2026-06-11  
**Phase:** Phase 4 — GraphSAGE  
**Experiment ID:** graphsage_eval

**Observation:**
Hidden dimension 32 achieved best fold-0 AUPRC (0.6413).

**Evidence:**
h32_L2_d0.3_mean_f15-10 ranked #1 in HPO leaderboard.

**Impact:**
Smaller models generalize better than h128 on this corpus.

**Follow-up Action:**
Prioritize h32/h64 in R-GCN HPO.

---

## F008

**Date:** 2026-06-11  
**Phase:** Phase 5 — R-GCN  
**Experiment ID:** rgcn_pub_v2

**Observation:**
R-GCN eval benchmark AUPRC 0.6542; does not beat LR.

**Evidence:**
artifacts/published/rgcn/rgcn_benchmark.json; docs/rgcn_diagnosis.md

**Impact:**
Gate G5 failed.

**Follow-up Action:**
Fusion model or richer edge features.

---

## F009

**Date:** 2026-06-11  
**Phase:** Phase 5 — R-GCN  
**Experiment ID:** rgcn_audit

**Observation:**
Prior rgcn_benchmark.json (AUPRC 0.6159) used max_epochs=2 and is invalid for baseline comparison.

**Evidence:**
artifacts/archived/rgcn_invalid_2epoch_20260611/

**Impact:**
Withdrawn from publication narrative until republication-grade benchmark completes.

**Follow-up Action:**
Run rgcn_pub_v2 with max_epochs=100, patience=10, full HPO grid.

---

## F010

**Date:** 2026-06-11  
**Phase:** Phase 5 — R-GCN  
**Experiment ID:** rgcn_pub_v2

**Observation:**
Publication-grade R-GCN (rgcn_pub_v2): h128_L2_d0.4 AUPRC 0.6542±0.0464; beats GraphSAGE (0.6530) but not LR (0.6810).

**Evidence:**
artifacts/published/rgcn/rgcn_benchmark.json; max_epochs=100, patience=10, 12-config HPO.

**Impact:**
Gate G5 still failed vs LR; marginal lift over GraphSAGE not significant (p=1.0).

**Follow-up Action:**
Phase 6 fusion model; do not cite invalid 2-epoch benchmark.

---

## F011

**Date:** 2026-06-11  
**Phase:** Gate G5  
**Experiment ID:** rgcn_pub_v2

**Observation:**
Gate G5 FAILED.

**Evidence:**
D:\docs\hgad-cms-fraud\docs\rgcn_diagnosis.md

**Impact:**
Blocks next phase.

**Follow-up Action:**
See project_status.md next_action.

---

## F012

**Date:** 2026-06-11  
**Phase:** Phase 6 — Hybrid Fusion  
**Experiment ID:** fusion_v1

**Observation:**
logistic_regression 5-fold AUPRC 0.6810±0.0389; does not beat LR baseline.

**Evidence:**
experiment_id=fusion_v1; artifacts/published/fusion/fusion_benchmark.json

**Impact:**
Updates best-model ranking and publication story.

**Follow-up Action:**
Sync research docs; compare significance tests.

---
