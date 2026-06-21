# 5. Experimental Setup

## 5.1 Hardware and software

| Component | Specification |
|-----------|---------------|
| Python | ≥ 3.10 |
| OS tested | Windows 10/11, Linux |
| GPU | Optional CUDA for G4–G6; CPU fallback supported |
| Deep learning | PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4 |
| Tabular models | scikit-learn ≥ 1.3, CatBoost ≥ 1.2 |
| Random seed (CV) | 42 |

Install: `pip install -e ".[models,gnn,dev]"` from repository root.

## 5.2 Data and artifact paths

- Raw data: `datasets/Healthcare Provider Fraud Detection Analysis/` (four Train CSV files)
- Processed features: `data/processed/`
- CV splits: `data/splits/` (seed 42, 5 folds)
- Published benchmarks: `artifacts/published/`

## 5.3 Hyperparameters — summary

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
| HPO | 12 configs, fold 0 → 5-fold confirm |

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
| HPO | 12 configs, fold 0 → 5-fold confirm |

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

**AUPRC** (area under precision–recall curve) is the primary metric because fraud prevalence is low; ROC-AUC can be high (~0.91–0.93) even when precision at operational thresholds is moderate.

Reported values are **mean ± standard deviation** across five folds unless noted as fold-0 ablation only.

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
- Fig. 5: Precision–recall curves
- Fig. 6: Findings summary

See `docs/figure_catalog.md`.
