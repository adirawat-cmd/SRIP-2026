# Appendices

## Appendix A — Hyperparameters

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

**Selected for 5-fold benchmark:** `h32_L2_d0.3_mean_f15-10` (AUPRC 0.6413 fold 0; 0.6530 ± 0.0429 five-fold).

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

**Selected for 5-fold benchmark:** `h128_L2_d0.4_b4_f15-10` (AUPRC 0.6542 ± 0.0464).

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

## Appendix B — Graph schema (schema_v1.1)

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

Provider–provider edge if ≥ **2** shared beneficiaries on fold claims. Rationale: reduce single co-occurrence noise (Decision D002 in experiment registry).

### B.4 Fold artifacts

Per-fold graphs: `artifacts/published/graphs/v1.1/fold_{0-4}/`  
Reference graph: `artifacts/published/graphs/v1.1/reference/`

---

## Appendix C — Additional results

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

## Appendix D — Reproducibility protocol

1. Clone repository; install `pip install -e ".[models,gnn,dev]"`.
2. Place CMS Train CSVs in `datasets/Healthcare Provider Fraud Detection Analysis/`.
3. Run G1–G6 scripts in order (see `docs/reproducibility.md`).
4. Compare outputs to `artifacts/published/MANIFEST.json`.
5. Regenerate figures: `python scripts/10_generate_publication_figures.py`.
6. Sync documentation: `python scripts/08_sync_research_docs.py --published-dir artifacts/published`.

**Verification checklist (expected):**

```
LR AUPRC      ≈ 0.6810
GraphSAGE     ≈ 0.6530
R-GCN         ≈ 0.6542
Fusion stack  ≈ 0.6671
```

**Frozen artifact index:** `docs/repository_freeze_report.md`  
**Experiment registry:** `docs/experiment_registry.md`

---

## Appendix E — Statistical tests reference

All tests: paired Wilcoxon signed-rank on per-fold AUPRC (*n* = 5).

Key comparisons:

| Test | *p* | Significant |
|------|-----|-------------|
| LR vs GraphSAGE | 0.0625 | No |
| LR vs R-GCN | 0.0625 | No |
| GraphSAGE vs R-GCN | 1.0000 | No |
| LR vs fusion_weighted | 0.1250 | No |

Full tables: `artifacts/published/baselines/significance_auprc.json`, `rgcn/significance_auprc.json`, `fusion/significance_auprc.json`.
