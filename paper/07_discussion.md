# 8. Discussion

## 8.1 Why logistic regression outperformed graph methods

Three mechanisms—supported by benchmarks and ablations—explain LR’s advantage:

**Feature dominance.** Provider tabular vectors summarize 86 dimensions of utilization, reimbursement, diagnosis, procedure, and panel statistics. These features already encode much of what message passing would aggregate from neighbors. R-GCN *provider-only* ablation (AUPRC 0.6525 on fold 0) nearly matches full-graph R-GCN (0.6505), confirming that **node features carry most usable signal**.

**Graph heterophily.** Fraudulent and legitimate providers share beneficiaries and physicians; homophilic aggregation (GraphSAGE mean) mixes labels across heterogeneous neighborhoods. GraphSAGE underperforms LR by 2.8 AUPRC points on average. Mean aggregation outperformed max in HPO, but neither closed the gap to tabular models.

**Evaluation strictness.** Provider-disjoint CV prevents the most common leakage but still allows **semi-transductive** beneficiary nodes at inference. Even under this favorable graph setting, GNNs did not surpass LR—suggesting that additional leakage is not the primary reason for graph underperformance on this corpus.

Logistic regression also offers **calibrated linear decision boundaries** on engineered features with minimal tuning, while GNNs require HPO and early stopping with higher variance (GraphSAGE std 0.043; fusion stack std 0.052).

## 8.2 Heterophily and relation semantics

Schema_v1.1 distinguishes *treats*, *bills_with*, and *collaborates*, yet R-GCN gains over GraphSAGE are negligible (*p* = 1.0). Relation ablations show *treats* and *bills_with* matter more than *collaborates*, aligning with billing-centric fraud patterns. Collaboration edges (≥ 2 shared beneficiaries) may capture weak or noisy peer structure relative to claim volumes already summarized in tabular features.

## 8.3 Anomaly detection limitations

Isolation forests on tabular, GraphSAGE, and R-GCN embeddings yield AUPRC **0.36–0.45**, well below supervised models. Fraud providers in this dataset are not reliably embedding-space outliers; they may mimic legitimate utilization with subtle distortions better captured by supervised feature weights than by unsupervised density estimates.

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
- Statistical tests rarely reach α = 0.05; confidence intervals on fold means should accompany point estimates (see Fig. 3).
- One invalid R-GCN smoke run (2 epochs) was archived and excluded from all reported results.
