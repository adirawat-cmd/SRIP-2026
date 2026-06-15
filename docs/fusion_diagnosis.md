# Hybrid Anomaly Fusion Diagnosis Report

## Research question

Can embedding-space anomaly detection improve provider fraud detection when supervised models plateau around AUPRC 0.65–0.68?

## Executive summary

- **LR baseline AUPRC:** 0.6810
- **GraphSAGE benchmark:** 0.6530
- **Best model:** `logistic_regression` — AUPRC **0.6810 ± 0.0389**
- **Beat LR:** NO

## Model comparison (5-fold CV)

| Model | AUPRC | ROC-AUC | Prec | Recall | F1 | Recall@K |
|-------|-------|---------|------|--------|-----|----------|
| `logistic_regression` | 0.6810±0.0389 | 0.9274 | 0.3921 | 0.8458 | 0.5355 | 0.6087 |
| `fusion_stack_logistic` | 0.6671±0.0521 | 0.9247 | 0.5506 | 0.6621 | 0.5997 | 0.6068 |
| `catboost` | 0.6615±0.0507 | 0.9266 | 0.5650 | 0.6523 | 0.6048 | 0.6048 |
| `fusion_weighted` | 0.6615±0.0507 | 0.9266 | 0.5630 | 0.6542 | 0.6045 | 0.6048 |
| `rgcn` | 0.6540±0.0464 | 0.9141 | 0.4066 | 0.8161 | 0.5391 | 0.6087 |
| `fusion_rank` | 0.6536±0.0395 | 0.9206 | 0.2123 | 0.9565 | 0.3475 | 0.5988 |
| `graphsage` | 0.6524±0.0434 | 0.9143 | 0.4620 | 0.7509 | 0.5675 | 0.5909 |
| `if_graphsage` | 0.4476±0.0685 | 0.8835 | 0.2895 | 0.8576 | 0.4298 | 0.4704 |
| `if_rgcn` | 0.4073±0.1104 | 0.8682 | 0.2898 | 0.8754 | 0.4327 | 0.4114 |
| `if_tabular` | 0.3620±0.0574 | 0.8276 | 0.3748 | 0.4010 | 0.3685 | 0.3874 |

## Error overlap (LR misses caught by anomaly / graph)

- Fraud providers in val (total): **506**
- Missed by LR top-K: **198**
- LR misses caught by IF-tabular: **39**
- LR misses caught by IF-GraphSAGE: **34**
- LR misses caught by IF-R-GCN: **36**
- LR misses caught by R-GCN: **32**
- LR misses caught by fusion (weighted): **37**

## Statistical significance (paired Wilcoxon vs fusion_weighted)

- **catboost vs fusion_weighted**: p=1.0000, mean_diff=0.0000, significant=False
- **fusion_rank vs fusion_weighted**: p=0.4375, mean_diff=-0.0079, significant=False
- **fusion_stack_logistic vs fusion_weighted**: p=0.1875, mean_diff=0.0056, significant=False
- **fusion_weighted vs graphsage**: p=0.1875, mean_diff=0.0091, significant=False
- **fusion_weighted vs if_graphsage**: p=0.0625, mean_diff=0.2139, significant=False
- **fusion_weighted vs if_rgcn**: p=0.0625, mean_diff=0.2542, significant=False
- **fusion_weighted vs if_tabular**: p=0.0625, mean_diff=0.2995, significant=False
- **fusion_weighted vs logistic_regression**: p=0.1250, mean_diff=-0.0195, significant=False
- **fusion_weighted vs rgcn**: p=0.6250, mean_diff=0.0075, significant=False

## Conclusion

Hybrid fusion and anomaly towers did not improve AUPRC over logistic regression. This supports the hypothesis that CMS provider fraud detection in this corpus is **primarily feature-driven** rather than graph-structure-driven: graph embeddings and unsupervised anomaly scores add little beyond tabular provider features.