# Hybrid Explainable Heterogeneous Graph Anomaly Detection for Healthcare Provider Fraud Detection

**Dataset:** CMS Healthcare Provider Fraud Detection Analysis  
**Evaluation:** Provider-disjoint stratified 5-fold CV (seed=42) · Primary metric: AUPRC  
**Best model:** Logistic Regression (AUPRC = 0.6810 ± 0.0389)  
**Artifact source:** `artifacts/published/` (frozen 2026-06-21)

---
# Abstract

Healthcare provider fraud imposes substantial costs on public insurance programs, yet detecting fraudulent billing patterns remains difficult because fraud manifests through complex relationships among providers, beneficiaries, and physicians. Graph neural networks (GNNs) offer a natural representation for such relational data, and hybrid fusion with anomaly detection has been proposed to combine tabular and structural signals. We present a systematic evaluation of heterogeneous graph methods for Medicare provider fraud detection on the CMS Healthcare Provider Fraud Detection Analysis dataset, comparing tabular baselines, GraphSAGE, relation-aware R-GCN, and hybrid fusion under provider-disjoint stratified 5-fold cross-validation.

Our pipeline constructs schema_v1.1 heterogeneous graphs with provider, beneficiary, and physician nodes and *treats*, *bills_with*, and *collaborates* relations, using fold-safe graph construction to prevent provider leakage. Contrary to the hypothesis that graph structure would improve ranking quality beyond tabular features, **logistic regression achieves the best mean area under the precisionâ€“recall curve (AUPRC = 0.6810 Â± 0.0389)**, outperforming CatBoost (0.6615), the best fusion model (stacked logistic meta-learner, 0.6671), R-GCN (0.6542), and GraphSAGE (0.6530). Paired Wilcoxon tests do not reject equal performance at Î± = 0.05 for most pairwise comparisons, including LR versus R-GCN (*p* = 0.0625), reflecting limited statistical power with five folds and ~101 fraud providers per validation split. Relation ablations indicate that removing billing or treatment edges degrades R-GCN more than removing providerâ€“provider collaboration edges, suggesting that claim-volume relations carry more signal than collaboration structure aloneâ€”yet this signal is already captured by rich provider-level tabular features.

These results do not invalidate graph-based fraud detection in general; they demonstrate that, **under strict provider-disjoint evaluation on this CMS corpus, tabular feature dominance and graph heterophily limit the marginal value of GNN message passing**. We discuss implications for benchmark design, semi-transductive GNN evaluation, and the need for explainability methods that can articulate when relational models add value. All benchmarks and reproduction commands are frozen in the project repository.

**Keywords:** healthcare fraud detection, Medicare, graph neural networks, heterogeneous graphs, R-GCN, GraphSAGE, provider fraud, area under the precisionâ€“recall curve, anomaly detection


---

# 1. Introduction

## 1.1 Healthcare fraud and Medicare program integrity

Healthcare fraud, waste, and abuse account for tens of billions of dollars in annual losses to U.S. public insurance programs. The Centers for Medicare & Medicaid Services (CMS) and its program integrity contractors investigate providers whose billing patterns deviate from peer norms, exhibit upcoding or unbundling, or involve collusion among entities in the care delivery network. Machine learning has been applied to prioritize providers for audit, but operational deployment demands high precision at constrained investigation budgets and auditable decision support.

The **CMS Healthcare Provider Fraud Detection Analysis** dataset released for research and education provides labeled provider-level fraud indicators alongside beneficiary demographics, inpatient and outpatient claims, and physician involvement. Although not a live enforcement feed, it offers a reproducible benchmark for comparing fraud detection methodologies under class imbalance (approximately 9.35% fraud prevalence at the provider level).

## 1.2 Limitations of existing approaches

Much prior work treats fraud detection as a tabular classification problem: aggregate claims into provider-level features (utilization rates, diagnosis and procedure mixtures, panel characteristics) and train gradient-boosted trees, logistic models, or neural networks. These methods excel when predictive signal resides in scalar summaries but may miss **relational structure**â€”for example, fraud rings linked through shared beneficiaries, referral patterns, or physician co-occurrence.

Conversely, graph-based approaches model providers as nodes in heterogeneous networks but face practical challenges: **label leakage** across train and validation splits if providers or their neighborhoods overlap improperly; **heterophily**, when fraudulent and legitimate providers connect to similar beneficiaries; and **feature dominance**, when tabular summaries already encode most of the information that message passing would propagate.

## 1.3 Motivation for graph learning

Medicare claims naturally form a heterogeneous graph: providers treat beneficiaries, bill with physicians, and may collaborate with other providers through shared patient panels. Graph neural networks (GNNs) can propagate information along these relations and learn representations that complement tabular features. Homophilic models such as GraphSAGE aggregate neighbor embeddings under the assumption that connected nodes share similar labels; relation-aware models such as R-GCN assign separate transformation weights per edge type, which may better capture the distinct semantics of treatment, billing, and collaboration relations.

We hypothesized that graph structure would improve provider fraud ranking beyond a strong tabular baseline when evaluated under **provider-disjoint cross-validation**, which holds out entire providers per fold and rebuilds fold-safe graphs from training claims only.

## 1.4 Motivation for explainability

Program integrity investigators require more than a risk score: they need **interpretable evidence** linking predictions to billing patterns, peer groups, or anomalous subgraphs. Explainable AI (XAI) for GNNsâ€”including attention weights, subgraph explanations, and feature attributionâ€”could support audit workflows if graph models demonstrate additive value. Our project title emphasizes hybrid **explainable** heterogeneous graph anomaly detection; however, the experiments reported here focus on predictive benchmarking and ablation analysis. Explainability case studies (e.g., GNNExplainer) are identified as future work and are not claimed as completed contributions in this manuscript.

## 1.5 Contributions and findings

This paper reports a complete G1â€“G6 experimental pipeline with frozen, reproducible artifacts:

1. **Fold-safe heterogeneous graph construction (schema_v1.1)** with providerâ€“provider *collaborates* edges requiring at least two shared beneficiaries.
2. **Tabular baselines** including logistic regression, random forest, CatBoost, and random forest with graph centrality features.
3. **GraphSAGE and R-GCN benchmarks** with hyperparameter search on fold 0 and 5-fold confirmation of the best configuration.
4. **Hybrid fusion** combining supervised scores (tabular and GNN) with isolation-forest anomaly scores via weighted averaging, stacked logistic meta-learning, and rank fusion.
5. **Rigorous evaluation** using mean AUPRC as the primary metric under provider-disjoint stratified 5-fold CV (seed = 42).

**Primary finding:** Logistic regression achieves the highest mean AUPRC (**0.6810**) and none of the graph or fusion methods beat this baseline in the locked evaluation protocol. We analyze why, discuss threats to validity, and outline directions for explainability and richer relational modeling.

## 1.6 Paper organization

Section 2 reviews related work. Section 3 analyzes the research gap. Section 4 describes methodology. Section 5 details experimental setup. Section 6 presents results and ablations. Section 7 discusses implications. Section 8 concludes. Appendices provide hyperparameters, graph schema details, supplementary results, and reproduction instructions.


---

# 2. Literature Review

## 2.1 Healthcare fraud detection

Healthcare fraud detection spans supervised classification on claims or providers, unsupervised anomaly detection on utilization outliers, and rule-based systems aligned with billing codes. Provider-level models aggregate longitudinal claims into feature vectors capturing reimbursement totals, service mix, and patient panel characteristics. Gradient-boosted decision trees and logistic regression remain strong baselines on structured health care data because they handle heterogeneous features, missing values, and class imbalance with relatively low tuning cost.

Public datasets such as the CMS Healthcare Provider Fraud Detection Analysis enable comparative studies but differ materially from operational CMS feeds: labels are binary potential-fraud flags without adjudication detail, and the corpus is static rather than temporally streaming. Evaluations that randomize claims or beneficiaries without provider disjointness can inflate performance through leakage; provider-disjoint protocols are therefore recommended for this prediction unit.

## 2.2 Graph neural networks for relational health data

GNNs generalize convolution to irregular graph structure. GraphSAGE samples and aggregates neighborhood features, supporting inductive inference on unseen nodes when features are available. Heterogeneous extensions and metapath-based methods address multiple node and edge types common in clinical and insurance graphs. For fraud, graphs may link providers to beneficiaries, physicians, diagnoses, or other providers via shared patients.

Performance gains from GNNs are not guaranteed: when node features are rich and neighborhoods are heterophilic (connected nodes differ in label), message passing may **dilute** rather than sharpen fraud signal. Benchmarks must specify whether beneficiaries and physicians are observed transductively at inference, as this affects the inductive claims that can be made.

## 2.3 Graph anomaly detection

Graph anomaly detection identifies nodes, edges, or subgraphs that deviate from learned normality. Unsupervised methods such as isolation forests on embeddings, autoencoders, or contrastive learning have been applied when labeled fraud is scarce. In our pipeline, isolation forests operate on tabular, GraphSAGE, and R-GCN embeddings as **anomaly towers** within a hybrid fusion framework, testing whether unsupervised scores complement supervised classifiers.

Prior work often reports gains on synthetic or homophilic anomaly benchmarks; real Medicare graphs may not exhibit clean embedding-space separability for fraud, particularly when labels are noisy and prevalence is low.

## 2.4 Explainable AI for fraud and graphs

Explainability methods for tree models (feature importance, SHAP) are mature; for GNNs, techniques such as GNNExplainer, PGExplainer, and attention visualization identify influential neighbors and edges. In regulated healthcare settings, explanations must align with investigator workflowsâ€”highlighting specific claims, peers, or beneficiariesâ€”not merely saliency heatmaps.

Our study motivates explainability but **does not yet report quantitative XAI experiments**; predictive benchmarks establish whether graph models warrant investment in explanation tooling.

## 2.5 CMS fraud and Kaggle literature

The CMS Kaggle-style fraud dataset has been used in data science competitions and academic exercises, often with random splits and accuracy-focused metrics. Published results vary widely depending on split protocol and feature engineering. Few studies simultaneously report:

- provider-disjoint cross-validation,
- heterogeneous graph construction with explicit relation types,
- relation-aware GNNs compared to strong tabular baselines on **AUPRC** under class imbalance,
- hybrid fusion with anomaly detection towers, and
- paired significance testing across folds.

Our work fills this protocol gap with a negative but scientifically informative result: graph and fusion methods do not surpass logistic regression under the stated evaluation.

---

# 3. Research Gap Analysis

| Gap | Prior practice | This study |
|-----|----------------|------------|
| **Evaluation unit** | Claim- or row-level splits | Provider-level prediction with provider-disjoint 5-fold CV |
| **Primary metric** | Accuracy, ROC-AUC | AUPRC (prevalence â‰ˆ 9.35%) |
| **Graph schema** | Homogeneous or implicit graphs | Explicit schema_v1.1 with three node types and three semantic relations |
| **PP edge design** | Ad hoc thresholds | *collaborates* edges require â‰¥ 2 shared beneficiaries (75,604 edges) |
| **GNN comparison** | GraphSAGE only or non-relation-aware | GraphSAGE vs R-GCN vs tabular LR/CatBoost/RF |
| **Fusion** | Simple ensembling | Weighted, stacked logistic, rank fusion + isolation-forest towers |
| **Leakage control** | Often unspecified | Fold-safe graphs; scalers fit on train fold only |
| **Statistical testing** | Point estimates | Paired Wilcoxon signed-rank tests across folds |
| **Reproducibility** | Ad hoc scripts | Frozen artifacts (`artifacts/published/`), full G1â€“G6 pipeline |

**Remaining gaps not addressed here:** temporal validation, geographic holdout, full 5-fold HPO (search performed on fold 0 only for GNNs), operational cost-sensitive learning, and post-hoc explainability case studies.

**Central research question:** Under provider-disjoint evaluation on the CMS corpus, does heterogeneous graph learningâ€”or hybrid fusion with graph anomaly scoresâ€”improve provider fraud ranking beyond strong tabular baselines?

**Answer from experiments:** No. Logistic regression (AUPRC 0.6810) remains best; graph models cluster near 0.653â€“0.654; best fusion reaches 0.6671 but still below LR.


---

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

## 4.4 Graph construction â€” schema_v1.1 (G2)

### 4.4.1 Node types

| Node type | Count (reference graph) | Features |
|-----------|-------------------------|----------|
| Provider | 5,410 | 86-dim tabular + graph node index |
| Beneficiary | 138,556 | Demographics, chronic conditions |
| Physician | 100,737 | Claim reference counts, role fractions |

### 4.4.2 Relation types

| Relation | Direction | Semantics | Edge weight |
|----------|-----------|-----------|-------------|
| *treats* | Provider â†’ Beneficiary | Provider submitted claims for beneficiary | Claim count |
| *bills_with* | Provider â†’ Physician | Provider billed with physician on claims | Claim count |
| *collaborates* | Provider â†” Provider | Shared beneficiary panel | Number of shared beneficiaries |

Reverse edges (*treats_rev*, *bills_with_rev*, *collaborates_rev*) are stored for message passing. Reference edge counts: **363,300** treats; **109,339** bills_with; **75,604** collaborates (undirected pairs counted in both directions in storage).

### 4.4.3 Providerâ€“provider collaboration design

Providerâ€“provider (*collaborates*) edges are constructed when two providers share at least **two beneficiaries** on training-fold claims (`pp_min_shared_beneficiaries = 2`). This threshold reduces noise from single co-occurrences while retaining plausible care-network structure. Alternative thresholds (1 or 5) were considered in schema design but v1.1 with threshold 2 passed Gate G2 validation.

### 4.4.4 Fold-safe construction

For each CV fold, graphs are built from **train-fold claims only** for training message passing. At validation time, an **inference graph** appends validation providers and their edges while reusing beneficiary and physician nodes observed during training (semi-transductive setting; see Section 7 and threats to validity).

## 4.5 Tabular baselines (G3)

Four supervised baselines share the same CV protocol:

1. **Logistic regression** (L2, class-weight balanced),
2. **Random forest**,
3. **CatBoost** gradient boosting,
4. **Random forest + centrality** â€” RF on tabular features augmented with graph centrality statistics (treats degree, bills degree, collaborates degree) computed per fold from the training graph.

Primary comparison baseline: **logistic regression** (best G3 AUPRC).

## 4.6 GraphSAGE (G4)

We train **inductive GraphSAGE** with mean or max neighbor aggregation, 2â€“3 layers, hidden dimensions {32, 64, 128}, dropout 0.3, fanout (15, 10), batch size 256, Adam optimizer (lr = 0.001), and early stopping on validation AUPRC (max 100 epochs, patience 15).

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

1. **Weighted average (`fusion_weighted`):** Non-negative weights optimized on training scores to maximize training AUPRC, applied to minâ€“max normalized tower scores.
2. **Stacked logistic (`fusion_stack_logistic`):** Logistic meta-learner on normalized tower scores, fit on a 15% stratified holdout from training providers.
3. **Rank fusion (`fusion_rank`):** Average of per-tower rank scores.

Isolation forests use `contamination='auto'`, `n_estimators=200`, seed = 42.

## 4.9 Evaluation protocol

- **Primary metric:** Mean **AUPRC** across five folds.
- **Secondary metrics:** ROC-AUC, precision, recall, F1, recall@K (K = number of fraud providers in validation fold).
- **Significance:** Paired **Wilcoxon signed-rank** tests on per-fold AUPRC differences (*n* = 5 pairs).
- **Gates G4â€“G6:** Pass if mean AUPRC strictly exceeds LR baseline (0.6810); all graph and fusion gates **failed** in published benchmarks.

All reported numbers derive from `artifacts/published/` frozen on 2026-06-21.


---

# 5. Experimental Setup

## 5.1 Hardware and software

| Component | Specification |
|-----------|---------------|
| Python | â‰¥ 3.10 |
| OS tested | Windows 10/11, Linux |
| GPU | Optional CUDA for G4â€“G6; CPU fallback supported |
| Deep learning | PyTorch â‰¥ 2.0, PyTorch Geometric â‰¥ 2.4 |
| Tabular models | scikit-learn â‰¥ 1.3, CatBoost â‰¥ 1.2 |
| Random seed (CV) | 42 |

Install: `pip install -e ".[models,gnn,dev]"` from repository root.

## 5.2 Data and artifact paths

- Raw data: `datasets/Healthcare Provider Fraud Detection Analysis/` (four Train CSV files)
- Processed features: `data/processed/`
- CV splits: `data/splits/` (seed 42, 5 folds)
- Published benchmarks: `artifacts/published/`

## 5.3 Hyperparameters â€” summary

### GraphSAGE (best 5-fold config)

| Parameter | Value |
|-----------|-------|
| Config ID | `h32_L2_d0.3_mean_f15-10` |
| Hidden dim | 32 |
| Layers | 2 |
| Dropout | 0.3 |
| Aggregator | mean |
| Fanout | (15, 10) |
| Batch size | 256 |
| Learning rate | 0.001 |
| Max epochs | 100 |
| Patience | 15 |
| HPO | 12 configs, fold 0 â†’ 5-fold confirm |

### R-GCN (best 5-fold config)

| Parameter | Value |
|-----------|-------|
| Config ID | `h128_L2_d0.4_b4_f15-10` |
| Hidden dim | 128 |
| Layers | 2 |
| Dropout | 0.4 |
| Num bases | 4 |
| Fanout | (15, 10) |
| Max epochs | 100 |
| Patience | 10 |
| HPO | 12 configs, fold 0 â†’ 5-fold confirm |

### Fusion

| Parameter | Value |
|-----------|-------|
| GraphSAGE config | As G4 best |
| R-GCN config | As G5 best |
| IF contamination | auto |
| IF estimators | 200 |
| Stack holdout | 15% train |

Full grids appear in Appendix A.

## 5.4 Provider-disjoint cross-validation

```
Total providers:     5,410
Fraud providers:       506  (9.35%)
Folds:                   5
Val providers/fold:  ~1,082  (~101 fraud)
```

Within each fold:

1. Train tabular models on train providers only.
2. Build train graph from train-fold claims; fit GNN on train providers.
3. Construct inference graph for validation providers; score held-out providers.
4. Compute metrics on validation provider labels only.

No validation provider appears in training graph nodes or loss computation.

## 5.5 Metrics

**AUPRC** (area under precisionâ€“recall curve) is the primary metric because fraud prevalence is low; ROC-AUC can be high (~0.91â€“0.93) even when precision at operational thresholds is moderate.

Reported values are **mean Â± standard deviation** across five folds unless noted as fold-0 ablation only.

## 5.6 Reproduction commands

Sequential pipeline (abbreviated):

```bash
python scripts/01_preprocess.py --raw-dir "datasets/Healthcare Provider Fraud Detection Analysis"
python scripts/02_create_splits.py --seed 42
python scripts/03_build_graphs.py --schema v1.1 --folds all --reference
python scripts/04_run_baselines.py --run-id baselines_v1
python scripts/06_evaluate_graphsage.py --search-mode eval --run-id graphsage_eval --no-resume
python scripts/07_train_rgcn.py --run-id rgcn_pub_v2 --max-epochs 100 --patience 10 --no-resume
python scripts/09_train_fusion.py --run-id fusion_v1
```

Expected best LR AUPRC: **0.6810**. See Appendix D and `docs/reproducibility.md`.

## 5.7 Figures

Publication figures are generated from published artifacts (`scripts/10_generate_publication_figures.py`):

- Fig. 1: System architecture
- Fig. 2: Graph schema
- Fig. 3: Model comparison (AUPRC with 95% CI)
- Fig. 4: R-GCN ablations
- Fig. 5: Precisionâ€“recall curves
- Fig. 6: Findings summary

See `docs/figure_catalog.md`.


---

# 6. Results

All metrics below are from **5-fold provider-disjoint CV** unless labeled as fold-0 ablation. Source: `artifacts/published/`.

## 6.1 Baseline comparison (G3)

| Model | AUPRC (mean Â± std) | ROC-AUC | Precision | Recall | F1 |
|-------|-------------------|---------|-----------|--------|-----|
| **Logistic regression** | **0.6810 Â± 0.0389** | 0.9274 Â± 0.0093 | 0.3921 Â± 0.0126 | 0.8458 Â± 0.0357 | 0.5355 Â± 0.0116 |
| CatBoost | 0.6615 Â± 0.0507 | 0.9266 Â± 0.0120 | 0.5650 Â± 0.0275 | 0.6523 Â± 0.0615 | 0.6048 Â± 0.0383 |
| RF + centrality | 0.6561 Â± 0.0591 | 0.9312 Â± 0.0102 | 0.7750 Â± 0.0539 | 0.3717 Â± 0.0503 | 0.5008 Â± 0.0515 |
| Random forest | 0.6517 Â± 0.0556 | 0.9295 Â± 0.0088 | 0.7624 Â± 0.0828 | 0.4210 Â± 0.0768 | 0.5408 Â± 0.0799 |

Logistic regression achieves the highest mean AUPRC. Random forest and RF + centrality reach higher precision but lower recall, reflecting different threshold behavior under imbalance.

**Gate G3:** PASSED (LR best).

### Baseline significance (Wilcoxon, AUPRC)

| Comparison | *p*-value | Mean diff | Significant (Î±=0.05) |
|------------|-----------|-----------|----------------------|
| LR vs CatBoost | 0.1250 | +0.0195 (LR higher) | No |
| LR vs Random forest | 0.1250 | +0.0293 | No |
| LR vs RF + centrality | 0.1250 | +0.0248 | No |
| CatBoost vs Random forest | 0.0625 | +0.0098 | No |

No pairwise baseline difference reaches *p* < 0.05 with five folds.

## 6.2 GraphSAGE results (G4)

| Metric | Value |
|--------|-------|
| Best HPO config (fold 0) | `h32_L2_d0.3_mean_f15-10` |
| Fold-0 AUPRC | 0.6413 |
| **5-fold AUPRC** | **0.6530 Â± 0.0429** |
| ROC-AUC | 0.9154 Â± 0.0127 |
| Precision | 0.4561 Â± 0.0289 |
| Recall | 0.7469 Â± 0.0975 |
| F1 | 0.5624 Â± 0.0193 |

**Gate G4 (beat LR):** FAILED (âˆ’0.028 vs LR).

### GraphSAGE HPO factors (fold 0)

| Factor | Best level | Mean AUPRC (fold 0) |
|--------|------------|---------------------|
| Aggregator | mean (0.631) vs max (0.598) | Mean preferred |
| Layers | 2 (0.622) vs 3 (0.607) | Shallow preferred |
| Hidden dim | 32 best single config (0.6413) | Smaller models competitive |

**GraphSAGE vs LR:** Wilcoxon *p* = 0.0625, mean diff = âˆ’0.0279, not significant at Î± = 0.05.

## 6.3 R-GCN results (G5)

| Metric | Value |
|--------|-------|
| Best config | `h128_L2_d0.4_b4_f15-10` |
| **5-fold AUPRC** | **0.6542 Â± 0.0464** |
| ROC-AUC | 0.9141 Â± 0.0120 |
| Precision | 0.4066 Â± 0.0315 |
| Recall | 0.8161 Â± 0.0651 |
| F1 | 0.5405 Â± 0.0360 |

**Gate G5 (beat LR):** FAILED (âˆ’0.0268 vs LR).

R-GCN **mean AUPRC exceeds GraphSAGE** (0.6542 vs 0.6530) but Wilcoxon *p* = 1.0000 (mean diff = âˆ’0.0012), so the difference is not statistically significant.

### R-GCN significance

| Comparison | *p*-value | Mean diff | Significant |
|------------|-----------|-----------|-------------|
| LR vs R-GCN | 0.0625 | +0.0267 (LR higher) | No |
| GraphSAGE vs R-GCN | 1.0000 | âˆ’0.0012 | No |
| CatBoost vs R-GCN | 0.6250 | +0.0073 | No |

## 6.4 Fusion results (G6)

| Model | AUPRC (mean Â± std) | ROC-AUC | Precision | Recall | F1 |
|-------|-------------------|---------|-----------|--------|-----|
| **Logistic regression** | **0.6810 Â± 0.0389** | 0.9274 | 0.3921 | 0.8458 | 0.5355 |
| Fusion stack (logistic) | 0.6671 Â± 0.0521 | 0.9247 | 0.5506 | 0.6621 | 0.5997 |
| CatBoost / fusion weighted | 0.6615 Â± 0.0507 | 0.9266 | ~0.56 | ~0.65 | ~0.60 |
| R-GCN (in fusion pipeline) | 0.6540 Â± 0.0464 | 0.9141 | 0.4066 | 0.8161 | 0.5391 |
| Fusion rank | 0.6536 Â± 0.0395 | 0.9206 | 0.2123 | 0.9565 | 0.3475 |
| GraphSAGE (in fusion pipeline) | 0.6524 Â± 0.0434 | 0.9143 | 0.4620 | 0.7509 | 0.5675 |
| IF tabular | 0.3620 Â± 0.0574 | 0.8276 | 0.3748 | 0.4010 | 0.3685 |
| IF GraphSAGE | 0.4476 Â± 0.0685 | 0.8835 | 0.2895 | 0.8576 | 0.4298 |
| IF R-GCN | 0.4073 Â± 0.1104 | 0.8682 | 0.2898 | 0.8754 | 0.4327 |

**Gate G6 (beat LR):** FAILED. Best fusion method (`fusion_stack_logistic`) remains **0.0139 AUPRC below LR**.

Stacked logistic fusion improves precision (0.55 vs 0.39) but sacrifices recall (0.66 vs 0.85) relative to LR. Rank fusion achieves very high recall (0.96) with low precision (0.21), unsuitable for fixed audit budgets without threshold tuning.

### Error overlap (LR misses)

Across all validation folds (506 fraud providers total):

- LR misses in top-K ranking: **198** fraud providers
- Caught among LR misses by IF-tabular: 39
- IF-GraphSAGE: 34; IF-R-GCN: 36
- R-GCN: 32; fusion weighted: 37

Anomaly towers capture **partial overlap** with LR errors but do not lift aggregate AUPRC above LR.

### Fusion significance (vs fusion_weighted baseline)

| Comparison | *p*-value | Significant |
|------------|-----------|-------------|
| fusion_weighted vs LR | 0.1250 | No |
| fusion_stack vs fusion_weighted | 0.1875 | No |
| fusion_weighted vs GraphSAGE | 0.1875 | No |
| fusion_weighted vs IF tabular | 0.0625 | No |

## 6.5 Summary ranking

| Rank | Model | AUPRC |
|------|-------|-------|
| 1 | Logistic regression | **0.6810** |
| 2 | Fusion stack (logistic) | 0.6671 |
| 3 | CatBoost / fusion weighted | 0.6615 |
| 4 | RF + centrality | 0.6561 |
| 5 | R-GCN | 0.6542 |
| 6 | Fusion rank | 0.6536 |
| 7 | GraphSAGE | 0.6530 |
| 8 | Random forest | 0.6517 |

---

# 7. Ablation Study (R-GCN)

Relation and feature ablations were run on **validation fold 0** with config `h128_L2_d0.4_b4_f15-10`. Source: `artifacts/diagnostics/rgcn/ablations/` and `docs/rgcn_diagnosis.md`.

| Configuration | AUPRC (fold 0) | Î” vs full |
|---------------|----------------|-----------|
| Full R-GCN | 0.6505 | â€” |
| No PP (*collaborates* removed) | 0.6502 | âˆ’0.0003 |
| Provider features only | 0.6525 | +0.0020 |
| No treats | 0.6435 | âˆ’0.0070 |
| No bills | 0.6358 | âˆ’0.0147 |

**Interpretation:**

1. **Removing *bills_with* or *treats*** hurts performance more than removing *collaborates*, indicating claim-volume relations contribute more than providerâ€“provider collaboration alone.
2. **Provider-only features** match or slightly exceed full-graph performance on fold 0, reinforcing that node features dominate relation signal for R-GCN on this corpus.
3. Ablation magnitudes are small (< 2% relative loss for bills removal), consistent with the narrow band between graph models and LR on full 5-fold benchmarks.

*Limitation:* Ablations are fold-0 only; 5-fold ablation benchmarks were not completed.


---

# 8. Discussion

## 8.1 Why logistic regression outperformed graph methods

Three mechanismsâ€”supported by benchmarks and ablationsâ€”explain LRâ€™s advantage:

**Feature dominance.** Provider tabular vectors summarize 86 dimensions of utilization, reimbursement, diagnosis, procedure, and panel statistics. These features already encode much of what message passing would aggregate from neighbors. R-GCN *provider-only* ablation (AUPRC 0.6525 on fold 0) nearly matches full-graph R-GCN (0.6505), confirming that **node features carry most usable signal**.

**Graph heterophily.** Fraudulent and legitimate providers share beneficiaries and physicians; homophilic aggregation (GraphSAGE mean) mixes labels across heterogeneous neighborhoods. GraphSAGE underperforms LR by 2.8 AUPRC points on average. Mean aggregation outperformed max in HPO, but neither closed the gap to tabular models.

**Evaluation strictness.** Provider-disjoint CV prevents the most common leakage but still allows **semi-transductive** beneficiary nodes at inference. Even under this favorable graph setting, GNNs did not surpass LRâ€”suggesting that additional leakage is not the primary reason for graph underperformance on this corpus.

Logistic regression also offers **calibrated linear decision boundaries** on engineered features with minimal tuning, while GNNs require HPO and early stopping with higher variance (GraphSAGE std 0.043; fusion stack std 0.052).

## 8.2 Heterophily and relation semantics

Schema_v1.1 distinguishes *treats*, *bills_with*, and *collaborates*, yet R-GCN gains over GraphSAGE are negligible (*p* = 1.0). Relation ablations show *treats* and *bills_with* matter more than *collaborates*, aligning with billing-centric fraud patterns. Collaboration edges (â‰¥ 2 shared beneficiaries) may capture weak or noisy peer structure relative to claim volumes already summarized in tabular features.

## 8.3 Anomaly detection limitations

Isolation forests on tabular, GraphSAGE, and R-GCN embeddings yield AUPRC **0.36â€“0.45**, well below supervised models. Fraud providers in this dataset are not reliably embedding-space outliers; they may mimic legitimate utilization with subtle distortions better captured by supervised feature weights than by unsupervised density estimates.

Fusion that includes IF towers does not beat LR; stacked logistic fusion **reweights supervised towers** toward higher precision but cannot exceed LR ranking quality on AUPRC.

## 8.4 Implications for healthcare fraud research

1. **Strong tabular baselines are mandatory.** Claims of graph superiority require provider-disjoint protocols and AUPRC under imbalance; beating LR (0.681) on this CMS corpus is non-trivial.
2. **Negative results are informative.** They caution against deploying GNN complexity without evidence of uplift on the target audit metric.
3. **Explainability should be conditional.** XAI investments for graph models are justified when graphs beat tabular baselines or reveal complementary error patterns; LR misses 198/506 fraud providers, leaving room for **targeted** complementary models, but our fusion/anomaly towers did not achieve this at scale.
4. **Benchmark reporting.** Publish fold-wise scores, significance tests, and graph construction details to avoid inflated graph metrics from split leakage.

## 8.5 Threats to validity

### Internal validity

| Threat | Mitigation | Residual risk |
|--------|------------|---------------|
| Provider leakage | Provider-disjoint CV, fold-safe graphs | Beneficiary transductivity may optimistic GNN scores |
| HPO on fold 0 only | 5-fold confirm of best config | Selected configs may be fold-specific |
| Low fold count (*n*=5) | Wilcoxon tests | Underpowered; LR vs R-GCN *p*=0.0625 |
| Class imbalance | AUPRC primary metric | Precision/recall trade-offs at fixed K differ |

### External validity

| Threat | Impact |
|--------|--------|
| Static CMS educational corpus | May not generalize to live CMS feeds |
| Binary fraud label | No fraud typology or severity |
| Single time snapshot | No temporal drift evaluation |
| U.S. Medicare only | Transfer to other payers unknown |

### Construct validity

AUPRC measures ranking quality for rare positives but **does not map directly to investigation ROI** under budget constraints. High-recall models (fusion rank: recall 0.96, precision 0.21) may not suit operational workflows.

Full analysis: `docs/threats_to_validity.md`.

## 8.6 Future work

1. **Explainability case studies:** GNNExplainer or SHAP on LR vs R-GCN errors for the 198 LR-missed fraud providers.
2. **5-fold relation ablations** and richer edge features (temporal decay, amount-weighted edges).
3. **Temporal and geographic holdout** beyond random provider folds.
4. **Cost-sensitive learning** aligned with audit budget (optimize precision@K for fixed K).
5. **Alternative heterogeneous architectures** (HGT, heterogeneous Graph Transformers) if justified by pilot uplift.
6. **External validation** on independent CMS releases or synthetic fraud injections with known plant patterns.

## 8.7 Limitations of this study

- Explainability methods are **motivated but not experimentally evaluated** in the reported benchmarks.
- Graph models use semi-transductive beneficiary nodes; fully inductive evaluation remains open.
- Statistical tests rarely reach Î± = 0.05; confidence intervals on fold means should accompany point estimates (see Fig. 3).
- One invalid R-GCN smoke run (2 epochs) was archived and excluded from all reported results.


---

# 9. Conclusion

We presented a reproducible pipeline for hybrid heterogeneous graph anomaly detection on the CMS Healthcare Provider Fraud Detection Analysis dataset, spanning fold-safe graph construction (schema_v1.1), tabular baselines, GraphSAGE, R-GCN, and hybrid fusion with isolation-forest anomaly towersâ€”all evaluated under provider-disjoint stratified 5-fold cross-validation with AUPRC as the primary metric.

**The central empirical finding is that logistic regression achieves the best performance (AUPRC = 0.6810 Â± 0.0389)** and neither relation-aware graph convolution nor stacked fusion improves this ranking metric on the published benchmarks. GraphSAGE (0.6530) and R-GCN (0.6542) perform similarly to each other; the best fusion variant (stacked logistic, 0.6671) narrows but does not close the gap to LR. Isolation-forest anomaly scores on graph embeddings remain substantially weaker (AUPRC < 0.45). R-GCN ablations indicate that treatment and billing relations contribute more than providerâ€“provider collaboration edges, yet provider tabular features already capture most predictive signal.

These results challenge the assumption that graph structure must improve Medicare provider fraud detection under strict evaluation. They emphasize **feature dominance, heterophily, and protocol rigor** as first-order considerations before adopting GNN complexity. For program integrity research, we recommend treating strong tabular baselines as gates for graph investment, reporting provider-disjoint AUPRC with significance tests, and pursuing explainability where models demonstrate measurable complementarity.

All artifacts, figures, and reproduction commands are frozen in the open repository (`artifacts/published/`, `docs/reproducibility.md`). Rather than overstating graph novelty, this work contributes a **verified negative result** with implications for benchmark design and future explainable relational modeling in healthcare fraud.


---

# Appendices

## Appendix A â€” Hyperparameters

### A.1 GraphSAGE search grid (12 configs, fold 0)

| Dimension | Values |
|-----------|--------|
| hidden_dim | 32, 64, 128 |
| num_layers | 2, 3 |
| dropout | 0.3 |
| aggregator | mean, max |
| fanout | (15, 10) for L=2; (15, 10, 5) for L=3 |
| batch_size | 256 |
| learning_rate | 0.001 |
| weight_decay | 1e-5 |
| max_epochs | 100 |
| patience | 15 |

**Selected for 5-fold benchmark:** `h32_L2_d0.3_mean_f15-10` (AUPRC 0.6413 fold 0; 0.6530 Â± 0.0429 five-fold).

Full leaderboard: `artifacts/published/gnn/hpo_leaderboard.csv`.

### A.2 R-GCN search grid (12 configs, fold 0)

| Dimension | Values |
|-----------|--------|
| hidden_dim | 32, 64, 128 |
| num_layers | 2, 3 |
| dropout | 0.2, 0.4 |
| num_bases | 4 |
| fanout | (15, 10) or (15, 10, 5) |
| max_epochs | 100 |
| patience | 10 |

**Selected for 5-fold benchmark:** `h128_L2_d0.4_b4_f15-10` (AUPRC 0.6542 Â± 0.0464).

### A.3 Tabular baselines

- Logistic regression: L2, `class_weight='balanced'`, max_iter=2000
- Random forest: 500 trees, `class_weight='balanced'`
- CatBoost: default binary classification with early stopping on validation
- RF + centrality: RF on 86 tabular + 4 centrality features per fold

### A.4 Fusion configuration

```yaml
schema: v1.1
graphsage: h32_L2_d0.3_mean_f15-10
rgcn: h128_L2_d0.4_b4_f15-10
if_contamination: auto
if_n_estimators: 200
stack_holdout_fraction: 0.15
seed: 42
```

---

## Appendix B â€” Graph schema (schema_v1.1)

### B.1 Reference graph statistics

| Element | Count |
|---------|-------|
| Providers | 5,410 |
| Beneficiaries | 138,556 |
| Physicians | 100,737 |
| treats edges | 363,300 |
| bills_with edges | 109,339 |
| collaborates edges | 75,604 |

### B.2 Node features (summary)

- **Provider (86 features):** claims, reimbursement stats, inpatient ratio, physician counts, diagnosis rates (50), procedure rates (20), panel age/gender/chronic summaries
- **Beneficiary:** age, gender, race, state, chronic condition indicators
- **Physician:** claim refs, unique providers/beneficiaries, reimbursement stats, attending/operating/other fractions

### B.3 Collaboration threshold

Providerâ€“provider edge if â‰¥ **2** shared beneficiaries on fold claims. Rationale: reduce single co-occurrence noise (Decision D002 in experiment registry).

### B.4 Fold artifacts

Per-fold graphs: `artifacts/published/graphs/v1.1/fold_{0-4}/`  
Reference graph: `artifacts/published/graphs/v1.1/reference/`

---

## Appendix C â€” Additional results

### C.1 Per-fold LR AUPRC

| Fold | AUPRC |
|------|-------|
| 0 | 0.6786 |
| 1 | 0.6521 |
| 2 | 0.7476 |
| 3 | 0.6549 |
| 4 | 0.6716 |

Source: `artifacts/published/baselines/logistic_regression.json`.

### C.2 Gate summary

| Gate | Criterion | Result |
|------|-----------|--------|
| G2 | Valid reference graph | PASS |
| G3 | Baseline quality | PASS (LR best) |
| G4 | GraphSAGE beats LR | **FAIL** |
| G5 | R-GCN beats LR | **FAIL** |
| G6 | Fusion beats LR | **FAIL** |

### C.3 Invalid run (excluded)

R-GCN `rgcn_v1` with `max_epochs=2` (AUPRC 0.6159) archived at `artifacts/archived/rgcn_invalid_2epoch_20260611/`. **Not cited** in this manuscript.

### C.4 Figures

| Figure | File stem |
|--------|-----------|
| 1 | `figures/pdf/fig01_system_architecture.pdf` |
| 2 | `figures/pdf/fig02_graph_schema.pdf` |
| 3 | `figures/pdf/fig03_model_comparison.pdf` |
| 4 | `figures/pdf/fig04_rgcn_ablations.pdf` |
| 5 | `figures/pdf/fig05_precision_recall.pdf` |
| 6 | `figures/pdf/fig06_findings_summary.pdf` |

---

## Appendix D â€” Reproducibility protocol

1. Clone repository; install `pip install -e ".[models,gnn,dev]"`.
2. Place CMS Train CSVs in `datasets/Healthcare Provider Fraud Detection Analysis/`.
3. Run G1â€“G6 scripts in order (see `docs/reproducibility.md`).
4. Compare outputs to `artifacts/published/MANIFEST.json`.
5. Regenerate figures: `python scripts/10_generate_publication_figures.py`.
6. Sync documentation: `python scripts/08_sync_research_docs.py --published-dir artifacts/published`.

**Verification checklist (expected):**

```
LR AUPRC      â‰ˆ 0.6810
GraphSAGE     â‰ˆ 0.6530
R-GCN         â‰ˆ 0.6542
Fusion stack  â‰ˆ 0.6671
```

**Frozen artifact index:** `docs/repository_freeze_report.md`  
**Experiment registry:** `docs/experiment_registry.md`

---

## Appendix E â€” Statistical tests reference

All tests: paired Wilcoxon signed-rank on per-fold AUPRC (*n* = 5).

Key comparisons:

| Test | *p* | Significant |
|------|-----|-------------|
| LR vs GraphSAGE | 0.0625 | No |
| LR vs R-GCN | 0.0625 | No |
| GraphSAGE vs R-GCN | 1.0000 | No |
| LR vs fusion_weighted | 0.1250 | No |

Full tables: `artifacts/published/baselines/significance_auprc.json`, `rgcn/significance_auprc.json`, `fusion/significance_auprc.json`.

