# Figure Catalog

Publication figures for the CMS Healthcare Provider Fraud Detection paper.  
**Generated from:** `artifacts/published/` and `artifacts/diagnostics/rgcn/ablations/`  
**Regenerate:** `python scripts/10_generate_publication_figures.py`  
**Formats:** `figures/png/` (300 dpi), `figures/svg/`, `figures/pdf/`

---

## Figure 1 — System Architecture

| Field | Value |
|-------|-------|
| **Files** | `fig01_system_architecture.{png,svg,pdf}` |
| **Caption** | End-to-end hybrid fraud detection pipeline from raw CMS claims to provider-level prediction. Orange gate labels (G1–G6) mark validated experiment milestones under provider-disjoint 5-fold cross-validation. |
| **Purpose** | Orient readers to the full experimental system and evaluation protocol. |
| **Paper section** | §II Methodology / §III Experimental Setup |
| **Data source** | Pipeline design (G1–G6); no numeric artifacts |

---

## Figure 2 — Dataset Graph Schema

| Field | Value |
|-------|-------|
| **Files** | `fig02_graph_schema.{png,svg,pdf}` |
| **Caption** | Heterogeneous graph schema_v1.1 with Provider (fraud label), Beneficiary, and Physician node types. Directed relations *treats* and *bills_with* are weighted by claim count; *collaborates* links providers sharing ≥2 beneficiaries (75,604 edges in reference graph). |
| **Purpose** | Illustrate the relational structure used by GraphSAGE and R-GCN. |
| **Paper section** | §II-B Graph Construction |
| **Data source** | `artifacts/published/graphs/v1.1/reference/graph_manifest.json` |

**Reference counts:** 5,410 providers · 138,556 beneficiaries · 100,737 physicians · 363,300 treats · 109,339 bills_with · 75,604 collaborates

---

## Figure 3 — Model Performance Comparison

| Field | Value |
|-------|-------|
| **Files** | `fig03_model_comparison.{png,svg,pdf}` |
| **Caption** | Mean 5-fold AUPRC for all primary models. Error bars show 95% confidence intervals of the fold mean (t-distribution, *n*=5). Logistic Regression achieves the highest AUPRC (0.681) and is marked as best. |
| **Purpose** | Primary quantitative comparison across tabular, graph, and fusion approaches. |
| **Paper section** | §IV Results — Main Benchmark |
| **Data source** | `artifacts/published/baselines/*.json`, `gnn/graphsage_benchmark.json`, `rgcn/rgcn_benchmark.json`, `fusion/fusion_benchmark.json` |

| Model | AUPRC (mean ± std) |
|-------|-------------------|
| Logistic Regression | 0.6810 ± 0.0389 |
| Random Forest | 0.6517 ± 0.0556 |
| CatBoost | 0.6615 ± 0.0507 |
| RF + Centrality | 0.6561 ± 0.0591 |
| GraphSAGE | 0.6530 ± 0.0429 |
| R-GCN | 0.6542 ± 0.0464 |
| Fusion (stack) | 0.6671 ± 0.0521 |

---

## Figure 4 — Graph Model Ablations

| Field | Value |
|-------|-------|
| **Files** | `fig04_rgcn_ablations.{png,svg,pdf}` |
| **Caption** | R-GCN relation and feature ablations on validation fold 0 (config h128_L2_d0.4_b4_f15-10). Panel (a) shows absolute AUPRC; panel (b) shows relative performance loss versus the full model. Removing billing or treatment edges degrades performance most; provider-only features retain most of the full-model score. |
| **Purpose** | Diagnose which graph relations contribute to R-GCN performance. |
| **Paper section** | §IV Results — Ablation Study |
| **Data source** | `artifacts/diagnostics/rgcn/ablations/*.json`; Provider Only AUPRC (0.6525) from `docs/rgcn_diagnosis.md` (fold-0, artifact not separately archived) |

| Ablation | AUPRC (fold 0) | Rel. loss |
|----------|----------------|-----------|
| Full R-GCN | 0.6505 | 0.0% |
| No PP | 0.6502 | 0.05% |
| Provider Only | 0.6525 | −0.31% |
| No Treats | 0.6435 | 1.08% |
| No Bills | 0.6358 | 2.26% |

---

## Figure 5 — Precision–Recall Curves

| Field | Value |
|-------|-------|
| **Files** | `fig05_precision_recall.{png,svg,pdf}` |
| **Caption** | Mean precision–recall curves averaged across 5 CV folds (per-fold interpolation). Legend reports mean fold AUPRC. Dashed line indicates provider fraud prevalence (9.35%). Logistic Regression (bold) dominates at all operating points. |
| **Purpose** | Show ranking quality beyond a single threshold metric; highlight class imbalance baseline. |
| **Paper section** | §IV Results — Precision–Recall Analysis |
| **Data source** | `artifacts/published/fusion/fusion_benchmark.json` → `folds[].val_labels`, `folds[].val_scores` |

| Model | Mean fold AUPRC |
|-------|-----------------|
| Logistic Regression | 0.6810 |
| CatBoost | 0.6615 |
| Fusion (stack) | 0.6671 |
| R-GCN | 0.6540 |
| GraphSAGE | 0.6524 |

---

## Figure 6 — Research Findings Summary

| Field | Value |
|-------|-------|
| **Files** | `fig06_findings_summary.{png,svg,pdf}` |
| **Caption** | Summary of the central empirical finding: tabular logistic regression outperforms graph-based and fusion models under strict provider-disjoint evaluation. Green dashed line marks the LR baseline; error bars show fold standard deviation. |
| **Purpose** | Visual abstract / key takeaway figure for abstract, conclusion, or slide deck. |
| **Paper section** | §V Discussion / §VI Conclusion |
| **Data source** | Same benchmarks as Figure 3 (subset: LR, GraphSAGE, R-GCN, Fusion stack) |

**Interpretation:** Rich provider-level tabular features (claims, reimbursements, diagnosis rates) set a high bar (AUPRC 0.681). Graph message passing and hybrid fusion fail to exceed this baseline under provider-disjoint CV, suggesting relational signal is subsumed by tabular features on CMS-Kaggle.

---

## Export checklist

- [x] Serif typography (Times-compatible)
- [x] 300 dpi PNG for raster submission
- [x] Vector SVG + PDF for IEEE/Springer/Elsevier
- [x] Colorblind-safe palette
- [x] All numeric values traceable to `artifacts/published/`
- [x] Regenerable via `scripts/10_generate_publication_figures.py`

---

*Last generated: 2026-06-21*
