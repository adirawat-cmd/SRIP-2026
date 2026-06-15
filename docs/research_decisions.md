# Research Decisions

Automated record of major design decisions.

Last synced: 2026-06-11T14:15:38.185547+00:00

---

## D001

**Date:** 2026-06-06  
**Experiment ID:** graph_construct_v1_1

**Decision:**
Adopt schema_v1.1 as primary heterogeneous graph schema.

**Alternatives Considered:**
v1b (bene-physician edges), v2 (diagnosis nodes), v4 (physician filter).

**Reason:**
Matches literature PP threshold; validated reference counts; fold-safe construction.

**Expected Impact:**
Reproducible graph artifacts for all models.

---

## D002

**Date:** 2026-06-06  
**Experiment ID:** graph_construct_v1_1

**Decision:**
Provider-provider collaborates edges require >= 2 shared beneficiaries.

**Alternatives Considered:**
Threshold 1 (pp_t1), threshold 5 (pp_t5), no PP (v1.1-no_pp).

**Reason:**
Reduces noise from single co-occurrence; G2 reference count 75,604 edges.

**Expected Impact:**
Sparsifies PP graph; improves signal-to-noise for GNN message passing.

---

## D003

**Date:** 2026-06-06  
**Experiment ID:** preprocess_v1

**Decision:**
Claims are not materialized as graph nodes.

**Alternatives Considered:**
Claim-level heterogeneous graph; provider-only projected graph.

**Reason:**
Provider-level prediction target; claim features aggregated to provider nodes.

**Expected Impact:**
Manageable graph size; aligns with CMS fraud detection unit.

---

## D004

**Date:** 2026-06-11  
**Experiment ID:** graphsage_eval

**Decision:**
Proceed to R-GCN after GraphSAGE Gate G4 failure.

**Alternatives Considered:**
HGT, GAT, deeper GraphSAGE, tabular-only ensemble.

**Reason:**
GraphSAGE homophily assumption likely limits performance; relations have distinct semantics.

**Expected Impact:**
Test whether relation-aware modeling beats LR AUPRC 0.6810.

---

## D005

**Date:** 2026-06-06  
**Experiment ID:** preprocess_v1

**Decision:**
Provider-disjoint stratified 5-fold CV (seed=42) for all evaluation.

**Alternatives Considered:**
Random split, temporal split, beneficiary-disjoint CV.

**Reason:**
Prevents provider leakage; stratification preserves fraud prevalence per fold.

**Expected Impact:**
Conservative generalization estimate; comparable across models.

---

## D006

**Date:** 2026-06-11  
**Experiment ID:** baselines_v1

**Decision:**
Primary metric: AUPRC; secondary: ROC-AUC, F1, Precision, Recall, Recall@K.

**Alternatives Considered:**
ROC-AUC primary, F1 primary, provider-level AP only.

**Reason:**
Class imbalance (9.35% fraud); AUPRC sensitive to ranking quality.

**Expected Impact:**
Consistent model selection across phases and gates.

---

## D007

**Date:** 2026-06-11  
**Experiment ID:** rgcn_audit

**Decision:**
Invalidate and archive rgcn_benchmark.json produced with max_epochs=2 (rgcn_v1 smoke-scale).

**Alternatives Considered:**
Keep 0.6159 AUPRC for G5 reporting; partial HPO continuation.

**Reason:**
Audit found 2-epoch cap is not comparable to GraphSAGE (max_epochs=100) or tabular baselines.

**Expected Impact:**
Re-run publication-grade R-GCN HPO and 5-fold benchmark before any G5 or paper claims.

---
