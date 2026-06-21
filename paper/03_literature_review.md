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

Explainability methods for tree models (feature importance, SHAP) are mature; for GNNs, techniques such as GNNExplainer, PGExplainer, and attention visualization identify influential neighbors and edges. In regulated healthcare settings, explanations must align with investigator workflows—highlighting specific claims, peers, or beneficiaries—not merely saliency heatmaps.

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
| **Primary metric** | Accuracy, ROC-AUC | AUPRC (prevalence ≈ 9.35%) |
| **Graph schema** | Homogeneous or implicit graphs | Explicit schema_v1.1 with three node types and three semantic relations |
| **PP edge design** | Ad hoc thresholds | *collaborates* edges require ≥ 2 shared beneficiaries (75,604 edges) |
| **GNN comparison** | GraphSAGE only or non-relation-aware | GraphSAGE vs R-GCN vs tabular LR/CatBoost/RF |
| **Fusion** | Simple ensembling | Weighted, stacked logistic, rank fusion + isolation-forest towers |
| **Leakage control** | Often unspecified | Fold-safe graphs; scalers fit on train fold only |
| **Statistical testing** | Point estimates | Paired Wilcoxon signed-rank tests across folds |
| **Reproducibility** | Ad hoc scripts | Frozen artifacts (`artifacts/published/`), full G1–G6 pipeline |

**Remaining gaps not addressed here:** temporal validation, geographic holdout, full 5-fold HPO (search performed on fold 0 only for GNNs), operational cost-sensitive learning, and post-hoc explainability case studies.

**Central research question:** Under provider-disjoint evaluation on the CMS corpus, does heterogeneous graph learning—or hybrid fusion with graph anomaly scores—improve provider fraud ranking beyond strong tabular baselines?

**Answer from experiments:** No. Logistic regression (AUPRC 0.6810) remains best; graph models cluster near 0.653–0.654; best fusion reaches 0.6671 but still below LR.
