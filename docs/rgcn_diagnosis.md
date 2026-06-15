# R-GCN Evaluation Diagnosis Report

## Executive summary

- **LR baseline AUPRC:** 0.6810
- **GraphSAGE benchmark AUPRC:** 0.6530
- **Best R-GCN (5-fold):** 0.6542 ± 0.0464
- **Gate G5 (beat LR):** FAILED
- **Beat GraphSAGE:** YES

## Top configurations (HPO fold 0)

| Rank | Config | AUPRC | ROC-AUC | Prec | Recall | F1 |
|------|--------|-------|---------|------|--------|-----|
| 1 | `h128_L2_d0.4_b4_f15-10` | 0.6528 | 0.9187 | 0.4308 | 0.8317 | 0.5676 |
| 2 | `h128_L2_d0.2_b4_f15-10` | 0.6438 | 0.9126 | 0.3373 | 0.8515 | 0.4831 |
| 3 | `h64_L2_d0.4_b4_f15-10` | 0.6430 | 0.9119 | 0.4724 | 0.7624 | 0.5833 |
| 4 | `h128_L3_d0.4_b4_f15-10-5` | 0.6404 | 0.9137 | 0.4293 | 0.8119 | 0.5616 |
| 5 | `h32_L3_d0.2_b4_f15-10-5` | 0.6387 | 0.9155 | 0.5000 | 0.7228 | 0.5911 |
| 6 | `h128_L3_d0.2_b4_f15-10-5` | 0.6316 | 0.9026 | 0.3664 | 0.8416 | 0.5105 |
| 7 | `h64_L3_d0.2_b4_f15-10-5` | 0.6279 | 0.9009 | 0.4966 | 0.7327 | 0.5920 |
| 8 | `h64_L2_d0.2_b4_f15-10` | 0.6258 | 0.9074 | 0.3728 | 0.8416 | 0.5167 |
| 9 | `h32_L2_d0.4_b4_f15-10` | 0.6167 | 0.9073 | 0.4309 | 0.7723 | 0.5532 |
| 10 | `h64_L3_d0.4_b4_f15-10-5` | 0.6135 | 0.8921 | 0.4797 | 0.7030 | 0.5703 |

## Relation & feature ablations (fold 0)

| Ablation | AUPRC |
|----------|-------|
| feature:provider_only | 0.6525 |
| relation:full | 0.6505 |
| relation:no_pp | 0.6502 |
| relation:no_treats | 0.6435 |
| relation:no_bills | 0.6358 |

## Statistical significance (paired Wilcoxon)

- **catboost vs rgcn_h128_L2_d0.4_b4_f15-10**: p=0.6250, mean_diff=0.0073, significant=False
- **graphsage vs rgcn_h128_L2_d0.4_b4_f15-10**: p=1.0000, mean_diff=-0.0012, significant=False
- **logistic_regression vs rgcn_h128_L2_d0.4_b4_f15-10**: p=0.0625, mean_diff=0.0267, significant=False

## Conclusion

R-GCN did not exceed the LR baseline. Relation-specific convolutions alone may be insufficient; consider fusion models or richer edge features.