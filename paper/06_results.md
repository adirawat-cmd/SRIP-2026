# 6. Results

All metrics below are from **5-fold provider-disjoint CV** unless labeled as fold-0 ablation. Source: `artifacts/published/`.

## 6.1 Baseline comparison (G3)

| Model | AUPRC (mean ± std) | ROC-AUC | Precision | Recall | F1 |
|-------|-------------------|---------|-----------|--------|-----|
| **Logistic regression** | **0.6810 ± 0.0389** | 0.9274 ± 0.0093 | 0.3921 ± 0.0126 | 0.8458 ± 0.0357 | 0.5355 ± 0.0116 |
| CatBoost | 0.6615 ± 0.0507 | 0.9266 ± 0.0120 | 0.5650 ± 0.0275 | 0.6523 ± 0.0615 | 0.6048 ± 0.0383 |
| RF + centrality | 0.6561 ± 0.0591 | 0.9312 ± 0.0102 | 0.7750 ± 0.0539 | 0.3717 ± 0.0503 | 0.5008 ± 0.0515 |
| Random forest | 0.6517 ± 0.0556 | 0.9295 ± 0.0088 | 0.7624 ± 0.0828 | 0.4210 ± 0.0768 | 0.5408 ± 0.0799 |

Logistic regression achieves the highest mean AUPRC. Random forest and RF + centrality reach higher precision but lower recall, reflecting different threshold behavior under imbalance.

**Gate G3:** PASSED (LR best).

### Baseline significance (Wilcoxon, AUPRC)

| Comparison | *p*-value | Mean diff | Significant (α=0.05) |
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
| **5-fold AUPRC** | **0.6530 ± 0.0429** |
| ROC-AUC | 0.9154 ± 0.0127 |
| Precision | 0.4561 ± 0.0289 |
| Recall | 0.7469 ± 0.0975 |
| F1 | 0.5624 ± 0.0193 |

**Gate G4 (beat LR):** FAILED (−0.028 vs LR).

### GraphSAGE HPO factors (fold 0)

| Factor | Best level | Mean AUPRC (fold 0) |
|--------|------------|---------------------|
| Aggregator | mean (0.631) vs max (0.598) | Mean preferred |
| Layers | 2 (0.622) vs 3 (0.607) | Shallow preferred |
| Hidden dim | 32 best single config (0.6413) | Smaller models competitive |

**GraphSAGE vs LR:** Wilcoxon *p* = 0.0625, mean diff = −0.0279, not significant at α = 0.05.

## 6.3 R-GCN results (G5)

| Metric | Value |
|--------|-------|
| Best config | `h128_L2_d0.4_b4_f15-10` |
| **5-fold AUPRC** | **0.6542 ± 0.0464** |
| ROC-AUC | 0.9141 ± 0.0120 |
| Precision | 0.4066 ± 0.0315 |
| Recall | 0.8161 ± 0.0651 |
| F1 | 0.5405 ± 0.0360 |

**Gate G5 (beat LR):** FAILED (−0.0268 vs LR).

R-GCN **mean AUPRC exceeds GraphSAGE** (0.6542 vs 0.6530) but Wilcoxon *p* = 1.0000 (mean diff = −0.0012), so the difference is not statistically significant.

### R-GCN significance

| Comparison | *p*-value | Mean diff | Significant |
|------------|-----------|-----------|-------------|
| LR vs R-GCN | 0.0625 | +0.0267 (LR higher) | No |
| GraphSAGE vs R-GCN | 1.0000 | −0.0012 | No |
| CatBoost vs R-GCN | 0.6250 | +0.0073 | No |

## 6.4 Fusion results (G6)

| Model | AUPRC (mean ± std) | ROC-AUC | Precision | Recall | F1 |
|-------|-------------------|---------|-----------|--------|-----|
| **Logistic regression** | **0.6810 ± 0.0389** | 0.9274 | 0.3921 | 0.8458 | 0.5355 |
| Fusion stack (logistic) | 0.6671 ± 0.0521 | 0.9247 | 0.5506 | 0.6621 | 0.5997 |
| CatBoost / fusion weighted | 0.6615 ± 0.0507 | 0.9266 | ~0.56 | ~0.65 | ~0.60 |
| R-GCN (in fusion pipeline) | 0.6540 ± 0.0464 | 0.9141 | 0.4066 | 0.8161 | 0.5391 |
| Fusion rank | 0.6536 ± 0.0395 | 0.9206 | 0.2123 | 0.9565 | 0.3475 |
| GraphSAGE (in fusion pipeline) | 0.6524 ± 0.0434 | 0.9143 | 0.4620 | 0.7509 | 0.5675 |
| IF tabular | 0.3620 ± 0.0574 | 0.8276 | 0.3748 | 0.4010 | 0.3685 |
| IF GraphSAGE | 0.4476 ± 0.0685 | 0.8835 | 0.2895 | 0.8576 | 0.4298 |
| IF R-GCN | 0.4073 ± 0.1104 | 0.8682 | 0.2898 | 0.8754 | 0.4327 |

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

| Configuration | AUPRC (fold 0) | Δ vs full |
|---------------|----------------|-----------|
| Full R-GCN | 0.6505 | — |
| No PP (*collaborates* removed) | 0.6502 | −0.0003 |
| Provider features only | 0.6525 | +0.0020 |
| No treats | 0.6435 | −0.0070 |
| No bills | 0.6358 | −0.0147 |

**Interpretation:**

1. **Removing *bills_with* or *treats*** hurts performance more than removing *collaborates*, indicating claim-volume relations contribute more than provider–provider collaboration alone.
2. **Provider-only features** match or slightly exceed full-graph performance on fold 0, reinforcing that node features dominate relation signal for R-GCN on this corpus.
3. Ablation magnitudes are small (< 2% relative loss for bills removal), consistent with the narrow band between graph models and LR on full 5-fold benchmarks.

*Limitation:* Ablations are fold-0 only; 5-fold ablation benchmarks were not completed.
