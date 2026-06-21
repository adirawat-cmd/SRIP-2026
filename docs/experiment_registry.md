# Experiment Registry

**Project:** Hybrid Explainable Heterogeneous Graph Anomaly Detection for CMS Provider Fraud  
**Metric:** Mean 5-fold AUPRC (provider-disjoint stratified CV, seed=42)  
**Last updated:** 2026-06-11  
**Best model:** Logistic Regression (E001) — **0.6810**

---

## Primary benchmark models

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| E001 | LR | Complete | **0.6810** |
| E002 | CatBoost | Complete | 0.6615 |
| E003 | GraphSAGE | Complete | 0.6530 |
| E004 | R-GCN | Complete | 0.6542 |
| E005 | Fusion (stack_logistic) | Complete | 0.6671 |

---

## All experiments

### Pipeline & infrastructure

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| P001 | Data Processing (G1) | Complete | — |
| P002 | Graph Construction schema_v1.1 (G2) | Complete | — |

### Supervised baselines (G3)

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| E001 | LR | Complete | **0.6810** |
| E002 | CatBoost | Complete | 0.6615 |
| E006 | Random Forest | Complete | 0.6517 |
| E007 | RF + Centrality | Complete | 0.6561 |

### Graph neural networks (G4–G5)

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| E003 | GraphSAGE (`h32_L2_d0.3_mean_f15-10`) | Complete | 0.6530 |
| E004 | R-GCN (`h128_L2_d0.4_b4_f15-10`) | Complete | 0.6542 |

### Anomaly detection & fusion (G6)

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| E008 | Isolation Forest — Tabular | Complete | 0.3620 |
| E009 | Isolation Forest — GraphSAGE embeddings | Complete | 0.4476 |
| E010 | Isolation Forest — R-GCN embeddings | Complete | 0.4073 |
| E011 | Fusion — Weighted Average | Complete | 0.6615 |
| E005 | Fusion — Logistic Stacking | Complete | 0.6671 |
| E012 | Fusion — Rank Fusion | Complete | 0.6536 |

### Withdrawn / invalid runs

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| E004X | R-GCN smoke (`max_epochs=2`, `rgcn_v1`) | **Withdrawn** | ~~0.6159~~ |

> Invalid 2-epoch R-GCN run archived at `artifacts/archived/rgcn_invalid_2epoch_20260611/`. Do not cite in publications.

---

## Full registry (sorted by AUPRC)

| ID | Model | Status | AUPRC |
|----|-------|--------|-------|
| E001 | LR | Complete | **0.6810** |
| E005 | Fusion (stack_logistic) | Complete | 0.6671 |
| E002 | CatBoost | Complete | 0.6615 |
| E011 | Fusion — Weighted Average | Complete | 0.6615 |
| E007 | RF + Centrality | Complete | 0.6561 |
| E004 | R-GCN | Complete | 0.6542 |
| E012 | Fusion — Rank Fusion | Complete | 0.6536 |
| E003 | GraphSAGE | Complete | 0.6530 |
| E006 | Random Forest | Complete | 0.6517 |
| E009 | IF — GraphSAGE embeddings | Complete | 0.4476 |
| E010 | IF — R-GCN embeddings | Complete | 0.4073 |
| E008 | IF — Tabular | Complete | 0.3620 |
| P001 | Data Processing | Complete | — |
| P002 | Graph Construction | Complete | — |

---

## Experiment metadata

| ID | Run ID | Phase | Gate | Artifact |
|----|--------|-------|------|----------|
| P001 | `preprocess_v1` | G1 | PASS | `data/processed/manifest.json` |
| P002 | `graph_construct_v1_1` | G2 | PASS | `artifacts/published/graphs/v1.1/` |
| E001 | `baselines_v1` | G3 | PASS | `artifacts/published/baselines/logistic_regression.json` |
| E002 | `baselines_v1` | G3 | PASS | `artifacts/published/baselines/catboost.json` |
| E006 | `baselines_v1` | G3 | PASS | `artifacts/published/baselines/random_forest.json` |
| E007 | `baselines_v1` | G3 | PASS | `artifacts/published/baselines/rf_centrality.json` |
| E003 | `graphsage_eval` | G4 | FAIL | `artifacts/published/gnn/graphsage_benchmark.json` |
| E004 | `rgcn_pub_v2` | G5 | FAIL | `artifacts/published/rgcn/rgcn_benchmark.json` |
| E005–E012 | `fusion_v1` | G6 | FAIL | `artifacts/published/fusion/fusion_benchmark.json` |

---

## Notes

- **E005** is the best fusion method (`fusion_stack_logistic`); label **“Fusion”** in summaries refers to this run.
- G4/G5 gate: beat LR AUPRC **0.6810** — both failed.
- G6 gate: beat LR — failed; LR re-evaluated inside fusion CV matches E001.
- AUPRC values rounded to 4 decimal places; full means and std in `fusion_benchmark.json` and phase diagnosis reports.
