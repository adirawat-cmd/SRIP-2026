# Research Timeline

**Project:** Hybrid Explainable Heterogeneous Graph Anomaly Detection for CMS Provider Fraud  
**Primary metric:** AUPRC (5-fold provider-disjoint stratified CV, seed=42)  
**Last updated:** 2026-06-11  
**Experiment IDs:** `preprocess_v1`, `graph_construct_v1_1`, `baselines_v1`, `graphsage_eval`, `rgcn_pub_v2`, `fusion_v1`

---

## G1 — Data Processing

**Objective**  
Ingest raw CMS Healthcare Provider Fraud Detection data, clean and merge claims at the provider level, and produce leakage-safe processed tables with validated fraud prevalence for downstream modeling.

**Methods**
- Source: CMS train split (`datasets/Healthcare Provider Fraud Detection Analysis/`)
- Pipeline: `scripts/01_preprocess.py` → `data/processed/` (providers, beneficiaries, claims parquet)
- Validation: Gate G1 checks row counts, label integrity, and schema consistency
- Output: 5,410 providers; provider-level fraud rate **9.35%** (506 positive)
- Splits deferred to G1 companion step: `scripts/02_create_splits.py` → stratified 5-fold provider-disjoint CV

**Result**
- Gate G1 **PASSED** (2026-06-06)
- Processed manifest: `data/processed/manifest.json`
- Fraud prevalence floor for AUPRC benchmarking established at ~0.094

**Pass / Fail**  
**PASS**

**Lessons learned**
- Provider-disjoint stratified CV is mandatory; beneficiary-level leakage would inflate graph and tabular performance.
- A ~9% fraud rate makes AUPRC the right primary metric (ROC-AUC alone overstates separability).
- Raw CMS tables require explicit merge keys and duplicate handling before any feature engineering.

---

## G2 — Graph Construction

**Objective**  
Build fold-safe heterogeneous graphs linking providers, beneficiaries, and physicians under a publication-focused schema suitable for GNN training without validation leakage.

**Methods**
- Schema: **v1.1** (`docs/graph_schema.md`)
- Node types: provider, beneficiary, physician (claims **not** materialized as nodes)
- Edge types: `treats`, `bills_with`, `collaborates` (+ reverse edges)
- Provider–provider policy: shared beneficiaries **≥ 2**; weight `log1p(n_shared)`
- Fold-safe construction: all edges from **train-fold claims only** per CV fold
- Script: `scripts/03_build_graphs.py` → `artifacts/published/graphs/v1.1/fold_{0–4}/`
- Reference graph counts: ~363K `treats`, ~109K `bills_with`, ~76K `collaborates` edges

**Result**
- Gate G2 **PASSED** (2026-06-11)
- Fold-safe graph artifacts validated for all 5 folds + reference graph
- PP threshold ≥2 reduced noise vs threshold-1 (168K → 76K PP pairs)

**Pass / Fail**  
**PASS**

**Lessons learned**
- Single-shared-beneficiary PP edges are mostly noise; threshold ≥2 improves signal-to-noise.
- Fold-safe graph construction is non-negotiable for inductive GNN evaluation on held-out providers.
- Storing edge features separately from hard filters preserves ablation flexibility (G5).

---

## G3 — Baselines

**Objective**  
Establish strong tabular baselines under provider-disjoint 5-fold CV and select the primary benchmark for all subsequent graph and fusion comparisons.

**Methods**
- Models: Logistic Regression, CatBoost, Random Forest, RF + graph centrality
- Features: 85-dim provider tabular matrix (claim aggregates, top-50 dx / top-20 proc rates, panel stats)
- CV: `scripts/04_run_baselines.py`; fold-specific scaler and vocabulary fit on train only
- Gate G3: pipeline completes with reproducible metrics and comparison table

**Result**
- Gate G3 **PASSED** (2026-06-11)
- **Best model: Logistic Regression — AUPRC 0.6810 ± 0.0389**
- CatBoost: 0.6615 ± 0.0507 | RF: 0.6517 ± 0.0556 | RF+Centrality: 0.6561 ± 0.0591
- Artifacts: `artifacts/published/baselines/`

**Pass / Fail**  
**PASS**

**Lessons learned**
- Simple linear model with balanced LR beats tree ensembles on this corpus — tabular signal is strong.
- Centrality features alone do not close the gap to full tabular features.
- LR at **0.6810** becomes the immovable benchmark for G4–G6; all gates defined as “beat LR AUPRC.”

---

## G4 — GraphSAGE

**Objective**  
Test whether homophilic heterogeneous GraphSAGE message passing over schema_v1.1 improves fraud detection beyond tabular LR (Gate G4: AUPRC > 0.6810).

**Methods**
- Model: Hetero GraphSAGE (SAGEConv per edge type, provider binary head)
- HPO: 12 configs (eval mode) — hidden {32,64,128} × layers {2,3} × aggregator {mean,max}; fold-0 search
- Benchmark: best config **5-fold CV** with early stopping (max_epochs=100, patience=15)
- Best config: `h32_L2_d0.3_mean_f15-10` (fanout [15,10])
- Script: `scripts/05_train_graphsage.py`, `scripts/06_evaluate_graphsage.py`
- Inductive inference: val providers appended via `build_inference_graph` without train leakage

**Result**
- Gate G4 **FAILED** (2026-06-11)
- **5-fold AUPRC: 0.6530 ± 0.0429** (below LR 0.6810)
- Fold-0 HPO best: 0.6413; mean aggregation > max; 2 layers > 3 layers; h32 best on fold 0
- Artifacts: `artifacts/published/gnn/`, `docs/graphsage_diagnosis.md`

**Pass / Fail**  
**FAIL** (did not beat LR)

**Lessons learned**
- Homophilic GraphSAGE cannot overcome rich tabular provider features; graph structure adds limited incremental signal.
- Mean aggregation and shallow (2-layer) models preferred under GPU/memory constraints.
- Peak val AUPRC well below LR suggests **feature dominance**, not just overfitting — diagnosis confirmed neither pure over- nor under-fitting as root cause.
- Motivated Phase 5 relation-aware modeling (R-GCN).

---

## G5 — R-GCN

**Objective**  
Test whether relation-specific convolutions (separate weights per edge type) improve over homophilic GraphSAGE and beat LR (Gate G5: AUPRC > 0.6810).

**Methods**
- Model: Hetero R-GCN (relation-specific SAGEConv via HeteroConv, shared hidden projection)
- HPO: 12 configs — hidden {32,64,128} × layers {2,3} × dropout {0.2,0.4}; fold-0 search
- Benchmark: publication-grade **5-fold CV** (`rgcn_pub_v2`) — max_epochs=100, patience=10
- Best config: `h128_L2_d0.4_b4_f15-10`
- Relation ablations (fold 0): full, no_pp, no_treats, no_bills
- Script: `scripts/07_train_rgcn.py`
- Note: initial 2-epoch smoke benchmark (AUPRC 0.6159) **invalidated and archived** before final run

**Result**
- Gate G5 **FAILED** (2026-06-11)
- **5-fold AUPRC: 0.6542 ± 0.0464** (below LR; marginally above GraphSAGE 0.6530, not significant p=1.0)
- Beat GraphSAGE on point estimate: **YES**; beat LR: **NO** (Wilcoxon p=0.0625)
- Ablations: removing `bills_with` hurts most; PP edges contribute modestly
- Artifacts: `artifacts/published/rgcn/rgcn_benchmark.json`, `docs/rgcn_diagnosis.md`

**Pass / Fail**  
**FAIL** (did not beat LR)

**Lessons learned**
- Relation-aware modeling alone is insufficient; R-GCN does not unlock large gains over GraphSAGE.
- Always enforce full training budget (100 epochs + early stopping) before gate decisions — short smoke runs are not comparable to baselines.
- Provider tabular features dominate; graph convolutions mostly re-express information already in node features.
- Motivated Phase 6 hybrid fusion and anomaly detection in embedding space.

---

## G6 — Fusion

**Objective**  
Determine whether embedding-space anomaly detection (Isolation Forest) combined with supervised scores can improve provider fraud detection when individual models plateau at AUPRC ~0.65–0.68 (success = beat LR 0.6810).

**Methods**
- Anomaly towers: Isolation Forest on (1) tabular features, (2) GraphSAGE embeddings, (3) R-GCN embeddings
- Supervised towers: LR, CatBoost, GraphSAGE, R-GCN (retrained per fold with best G4/G5 configs)
- Fusion strategies: weighted average (grid-optimized on train), logistic stacking (15% holdout meta-learner), rank fusion
- CV: 5-fold provider-disjoint; same protocol as prior phases
- Error overlap: fraud providers missed by LR top-K vs caught by IF / graph / fusion
- Script: `scripts/09_train_fusion.py` (`fusion_v1`)

**Result**
- Gate G6 **FAILED** (2026-06-11) — best model remains **LR 0.6810 ± 0.0389**
- Best fusion: `fusion_stack_logistic` **0.6671 ± 0.0521**; `fusion_weighted` **0.6615 ± 0.0507**
- IF-only: tabular 0.362, GraphSAGE emb 0.448, R-GCN emb 0.407 (well below supervised)
- Error overlap: of 198 LR-missed fraud providers (top-K), IF-tabular recovered **39**, fusion weighted **37**
- Artifacts: `artifacts/published/fusion/fusion_benchmark.json`, `docs/fusion_diagnosis.md`

**Pass / Fail**  
**FAIL** (did not beat LR)

**Lessons learned**
- Unsupervised anomaly scores on tabular or GNN embeddings **underperform** supervised LR — fraud is not well separated as unsupervised outliers in this embedding space.
- Fusion cannot compensate when all towers peak below LR; stacking improves precision but sacrifices recall vs LR.
- CMS provider fraud in this corpus appears **primarily feature-driven**, not graph-structure-driven.
- Complementary error overlap exists (IF catches some LR misses) but too few to lift aggregate AUPRC.
- Publication narrative: report negative graph/fusion results as evidence, not failure — strong tabular baseline + rigorous CV is the core finding.

---

## Summary

| Gate | Phase | Best AUPRC | vs LR (0.6810) | Status |
|------|-------|------------|----------------|--------|
| G1 | Data Processing | — | — | **PASS** |
| G2 | Graph Construction | — | — | **PASS** |
| G3 | Baselines | **0.6810** (LR) | — | **PASS** |
| G4 | GraphSAGE | 0.6530 | −0.028 | **FAIL** |
| G5 | R-GCN | 0.6542 | −0.027 | **FAIL** |
| G6 | Fusion | 0.6810 (LR wins) | 0.000 | **FAIL** |

**Overall best model:** Logistic Regression — **AUPRC 0.6810 ± 0.0389**

**Cross-cutting conclusion:** Graph structure, relation-aware convolutions, and embedding-space anomaly detection do not improve over strong tabular baselines under provider-disjoint evaluation. Future work should emphasize explainability (GNNExplainer), case studies on LR misses, and feature-driven fraud typologies rather than deeper GNN architectures alone.
