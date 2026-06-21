# 4. Methodology

## 4.1 Dataset

We use the **CMS Healthcare Provider Fraud Detection Analysis** training release, comprising:

- Provider labels and attributes (5,410 providers; 506 flagged as potential fraud, **9.35%** prevalence),
- Beneficiary demographics (138,556 beneficiaries),
- Inpatient and outpatient claim records linking providers, beneficiaries, and physicians.

The prediction target is **binary provider-level fraud** (`PotentialFraud`). Claims are aggregated to providers; claim-level nodes are not materialized in schema_v1.1 to keep graph size manageable and align with CMS program integrity units.

## 4.2 Preprocessing (G1)

The preprocessing pipeline (`scripts/01_preprocess.py`):

1. Loads raw CSV files and harmonizes identifiers across tables.
2. Aggregates claim-level fields into **provider-level tabular features** (86 dimensions), including utilization totals, reimbursement statistics, diagnosis-rate features (top 50 ICD codes), procedure-rate features (top 20 procedure codes), and beneficiary panel summaries.
3. Validates data integrity (Gate G1) and writes `data/processed/` with checksum manifest.

All downstream models consume the same processed provider feature matrix unless ablated for R-GCN *provider-only* feature mode.

## 4.3 Cross-validation splits

Provider-disjoint **stratified 5-fold cross-validation** is created with `StratifiedKFold` (seed = 42). Each fold holds out approximately 1,082 providers (~101 fraud cases). Train and validation provider sets are strictly disjoint. Tabular scalers, vocabulary filters, and graph edges are computed from **training-fold claims only**.

## 4.4 Graph construction — schema_v1.1 (G2)

### 4.4.1 Node types

| Node type | Count (reference graph) | Features |
|-----------|-------------------------|----------|
| Provider | 5,410 | 86-dim tabular + graph node index |
| Beneficiary | 138,556 | Demographics, chronic conditions |
| Physician | 100,737 | Claim reference counts, role fractions |

### 4.4.2 Relation types

| Relation | Direction | Semantics | Edge weight |
|----------|-----------|-----------|-------------|
| *treats* | Provider → Beneficiary | Provider submitted claims for beneficiary | Claim count |
| *bills_with* | Provider → Physician | Provider billed with physician on claims | Claim count |
| *collaborates* | Provider ↔ Provider | Shared beneficiary panel | Number of shared beneficiaries |

Reverse edges (*treats_rev*, *bills_with_rev*, *collaborates_rev*) are stored for message passing. Reference edge counts: **363,300** treats; **109,339** bills_with; **75,604** collaborates (undirected pairs counted in both directions in storage).

### 4.4.3 Provider–provider collaboration design

Provider–provider (*collaborates*) edges are constructed when two providers share at least **two beneficiaries** on training-fold claims (`pp_min_shared_beneficiaries = 2`). This threshold reduces noise from single co-occurrences while retaining plausible care-network structure. Alternative thresholds (1 or 5) were considered in schema design but v1.1 with threshold 2 passed Gate G2 validation.

### 4.4.4 Fold-safe construction

For each CV fold, graphs are built from **train-fold claims only** for training message passing. At validation time, an **inference graph** appends validation providers and their edges while reusing beneficiary and physician nodes observed during training (semi-transductive setting; see Section 7 and threats to validity).

## 4.5 Tabular baselines (G3)

Four supervised baselines share the same CV protocol:

1. **Logistic regression** (L2, class-weight balanced),
2. **Random forest**,
3. **CatBoost** gradient boosting,
4. **Random forest + centrality** — RF on tabular features augmented with graph centrality statistics (treats degree, bills degree, collaborates degree) computed per fold from the training graph.

Primary comparison baseline: **logistic regression** (best G3 AUPRC).

## 4.6 GraphSAGE (G4)

We train **inductive GraphSAGE** with mean or max neighbor aggregation, 2–3 layers, hidden dimensions {32, 64, 128}, dropout 0.3, fanout (15, 10), batch size 256, Adam optimizer (lr = 0.001), and early stopping on validation AUPRC (max 100 epochs, patience 15).

**Hyperparameter search:** 12 configurations evaluated on **fold 0**; best config `h32_L2_d0.3_mean_f15-10` confirmed with **5-fold benchmark**.

GraphSAGE uses homogeneous convolution over the heterogeneous graph (relation types are treated as distinct edge types in the PyG heterogeneous data object but share the SAGE aggregation pattern per metapath slice).

## 4.7 R-GCN (G5)

**Relation-aware graph convolutional networks** assign basis-decomposed weights per relation type. Search grid: hidden dim {32, 64, 128}, layers {2, 3}, dropout {0.2, 0.4}, num_bases = 4, fanout (15, 10), max 100 epochs, patience 10.

Best 5-fold configuration: **`h128_L2_d0.4_b4_f15-10`**. Relation ablations (fold 0) remove *collaborates*, *treats*, or *bills_with* edges, or restrict to provider features only.

## 4.8 Hybrid fusion framework (G6)

The fusion module combines **six score towers** per fold:

| Tower | Description |
|-------|-------------|
| `logistic_regression` | Tabular LR probability |
| `catboost` | Tabular CatBoost probability |
| `graphsage` | GraphSAGE provider probability |
| `rgcn` | R-GCN provider probability |
| `if_tabular` | Isolation forest on tabular features |
| `if_graphsage` | Isolation forest on GraphSAGE embeddings |
| `if_rgcn` | Isolation forest on R-GCN embeddings |

**Fusion strategies:**

1. **Weighted average (`fusion_weighted`):** Non-negative weights optimized on training scores to maximize training AUPRC, applied to min–max normalized tower scores.
2. **Stacked logistic (`fusion_stack_logistic`):** Logistic meta-learner on normalized tower scores, fit on a 15% stratified holdout from training providers.
3. **Rank fusion (`fusion_rank`):** Average of per-tower rank scores.

Isolation forests use `contamination='auto'`, `n_estimators=200`, seed = 42.

## 4.9 Evaluation protocol

- **Primary metric:** Mean **AUPRC** across five folds.
- **Secondary metrics:** ROC-AUC, precision, recall, F1, recall@K (K = number of fraud providers in validation fold).
- **Significance:** Paired **Wilcoxon signed-rank** tests on per-fold AUPRC differences (*n* = 5 pairs).
- **Gates G4–G6:** Pass if mean AUPRC strictly exceeds LR baseline (0.6810); all graph and fusion gates **failed** in published benchmarks.

All reported numbers derive from `artifacts/published/` frozen on 2026-06-21.
