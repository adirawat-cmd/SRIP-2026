# GraphSAGE Evaluation Diagnosis Report

## Executive summary

- **LR baseline AUPRC:** 0.6810
- **Best HPO config (fold 0):** `h32_L2_d0.3_mean_f15-10` — AUPRC **0.6413**
- **5-fold benchmark AUPRC:** 0.6530 ± 0.0429
- **Gate G4 (beat LR):** FAILED

## Top 10 configurations (HPO fold 0, ranked by AUPRC)

| Rank | Config | h | L | drop | agg | AUPRC | ROC-AUC | Prec | Recall | F1 | Best epoch |
|------|--------|---|----|------|-----|-------|---------|------|--------|----|------------|
| 1 | `h32_L2_d0.3_mean_f15-10` | 32 | 2 | 0.3 | mean | 0.6413 | 0.9142 | 0.4438 | 0.7822 | 0.5663 | 7 |
| 2 | `h128_L2_d0.3_mean_f15-10` | 128 | 2 | 0.3 | mean | 0.6390 | 0.9066 | 0.3689 | 0.8218 | 0.5092 | 2 |
| 3 | `h64_L2_d0.3_mean_f15-10` | 64 | 2 | 0.3 | mean | 0.6369 | 0.9127 | 0.4241 | 0.8020 | 0.5548 | 4 |
| 4 | `h64_L3_d0.3_mean_f15-10-5` | 64 | 3 | 0.3 | mean | 0.6347 | 0.9061 | 0.3915 | 0.8218 | 0.5304 | 3 |
| 5 | `h128_L3_d0.3_mean_f15-10-5` | 128 | 3 | 0.3 | mean | 0.6303 | 0.8949 | 0.4162 | 0.7624 | 0.5385 | 15 |
| 6 | `h32_L2_d0.3_max_f15-10` | 32 | 2 | 0.3 | max | 0.6160 | 0.9057 | 0.4747 | 0.7426 | 0.5792 | 5 |
| 7 | `h32_L3_d0.3_mean_f15-10-5` | 32 | 3 | 0.3 | mean | 0.6044 | 0.8938 | 0.4843 | 0.7624 | 0.5923 | 19 |
| 8 | `h64_L2_d0.3_max_f15-10` | 64 | 2 | 0.3 | max | 0.6010 | 0.9056 | 0.4800 | 0.7129 | 0.5737 | 4 |
| 9 | `h128_L2_d0.3_max_f15-10` | 128 | 2 | 0.3 | max | 0.5986 | 0.8672 | 0.4923 | 0.6337 | 0.5541 | 11 |
| 10 | `h64_L3_d0.3_max_f15-10-5` | 64 | 3 | 0.3 | max | 0.5971 | 0.8957 | 0.5238 | 0.6535 | 0.5815 | 3 |

## Factor comparisons (fold 0 HPO)

### aggregator

| aggregator | n_configs | auprc_mean | auprc_std | auprc_max |
| --- | --- | --- | --- | --- |
| mean | 6 | 0.6311285272007865 | 0.01360671892209556 | 0.6412738366457518 |
| max | 6 | 0.5980238704348463 | 0.011047770622854202 | 0.6159847619796617 |

### hidden_dim

| hidden_dim | n_configs | auprc_mean | auprc_std | auprc_max |
| --- | --- | --- | --- | --- |
| 64 | 4 | 0.6174429110490606 | 0.02132040344465585 | 0.6369480957815228 |
| 32 | 4 | 0.6137972730497421 | 0.020490342228441642 | 0.6412738366457518 |
| 128 | 4 | 0.6124884123546466 | 0.02675667783805963 | 0.6390499030486129 |

### num_layers

| num_layers | n_configs | auprc_mean | auprc_std | auprc_max |
| --- | --- | --- | --- | --- |
| 2 | 6 | 0.6221346895435788 | 0.019556722572612272 | 0.6412738366457518 |
| 3 | 6 | 0.6070177080920539 | 0.021098998568865157 | 0.6347332365959524 |


## 5-fold benchmark (best config)

| Metric | Mean | Std |
|--------|------|-----|
| auprc | 0.6530 | 0.0429 |
| roc_auc | 0.9154 | 0.0127 |
| precision | 0.4561 | 0.0289 |
| recall | 0.7469 | 0.0975 |
| f1 | 0.5624 | 0.0193 |

## Training curves

- `artifacts/diagnostics/gnn/plots/curve_h32_L2_d0.3_mean_f15-10.png`
- `artifacts/diagnostics/gnn/plots/curve_h128_L2_d0.3_mean_f15-10.png`
- `artifacts/diagnostics/gnn/plots/curve_h64_L2_d0.3_mean_f15-10.png`
- `artifacts/diagnostics/gnn/plots/curve_h64_L3_d0.3_mean_f15-10-5.png`
- `artifacts/diagnostics/gnn/plots/curve_h128_L3_d0.3_mean_f15-10-5.png`
- `artifacts/diagnostics/gnn/plots/curve_top_configs_overlay.png`
- `artifacts/diagnostics/gnn/plots/compare_aggregator.png`
- `artifacts/diagnostics/gnn/plots/compare_hidden_dim.png`
- `artifacts/diagnostics/gnn/plots/compare_num_layers.png`

## Diagnosis

### Overfitting — **no**

Val AUPRC peaked at epoch 7 (0.6413), then fell 0.0016 while train loss continued to 0.2583

### Underfitting — **no**

Train loss 0.2583 still elevated; peak AUPRC 0.6413 below LR

### Class Imbalance — **YES**

Precision=0.444, Recall=0.782 (high recall / low precision pattern despite pos_weight BCE)

### Graph Heterophily — **YES**

GraphSAGE homophily assumption may mix fraud/legitimate provider signals via shared beneficiaries/physicians; AUPRC above chance but below tabular LR

### Feature Dominance — **YES**

Best GraphSAGE AUPRC 0.6413 < LR baseline 0.6810; 86-dim tabular provider features may carry most signal

## Statistical comparison vs logistic regression

- Paired Wilcoxon p-value: 0.0625
- Mean AUPRC diff (GraphSAGE − LR): -0.027916083883795982
- Significant at α=0.05: False

## Recommendation: proceed to R-GCN

GraphSAGE did **not** exceed the logistic regression baseline (AUPRC 0.6810). Evidence:

1. **Feature dominance** — LR on 86-dim tabular features outperforms all GraphSAGE configs.
2. **Relation heterogeneity** — v1.1 has treats / bills_with / collaborates with different semantics; GraphSAGE treats all relations with the same SAGEConv, while R-GCN uses relation-specific weights.
3. **Heterophily** — fraud and legitimate providers share beneficiaries and physicians; mean/max aggregation may dilute fraud signal.
4. **Overfitting risk** — several configs peak early then degrade on validation AUPRC.

R-GCN is the appropriate next step to test whether relation-aware message passing captures graph structure LR cannot.
