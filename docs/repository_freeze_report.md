# Repository Freeze Report

**Project:** CMS Healthcare Provider Fraud Detection Analysis  
**Freeze date:** 2026-06-21  
**CV protocol:** Provider-disjoint stratified 5-fold (seed=42)  
**Primary metric:** Mean AUPRC  
**Audit trail:** [`artifact_audit.md`](artifact_audit.md)

---

## Executive summary

All completed G1–G6 experiments have been audited, classified, and reorganized into a publication-ready artifact layout. **Nothing was deleted**; superseded, invalid, and diagnostic outputs were moved to `artifacts/archived/` and `artifacts/diagnostics/`. Authoritative benchmarks live under `artifacts/published/`.

**Best model:** Logistic Regression — **AUPRC 0.6810 ± 0.0389** (Gate G3 passed).  
GraphSAGE, R-GCN, and hybrid fusion **did not beat LR** under the locked evaluation protocol.

---

## 1. Final experiment inventory

| ID | Run ID | Phase | Gate | Status | Primary artifact |
|----|--------|-------|------|--------|------------------|
| P001 | `preprocess_v1` | G1 Data | G1 | PASS | `data/processed/manifest.json` |
| P002 | `graph_construct_v1_1` | G2 Graphs | G2 | PASS | `artifacts/published/graphs/v1.1/` |
| E001 | `baselines_v1` | G3 Baselines | G3 | PASS | `artifacts/published/baselines/logistic_regression.json` |
| E002 | `baselines_v1` | G3 Baselines | G3 | PASS | `artifacts/published/baselines/catboost.json` |
| E006 | `baselines_v1` | G3 Baselines | G3 | PASS | `artifacts/published/baselines/random_forest.json` |
| E007 | `baselines_v1` | G3 Baselines | G3 | PASS | `artifacts/published/baselines/rf_centrality.json` |
| E003 | `graphsage_eval` | G4 GraphSAGE | G4 | FAIL | `artifacts/published/gnn/graphsage_benchmark.json` |
| E004 | `rgcn_pub_v2` | G5 R-GCN | G5 | FAIL | `artifacts/published/rgcn/rgcn_benchmark.json` |
| E005–E012 | `fusion_v1` | G6 Fusion | G6 | FAIL | `artifacts/published/fusion/fusion_benchmark.json` |

### Archived / invalid runs (do not cite)

| Run ID | Reason | Location |
|--------|--------|----------|
| `rgcn_v1` | 2-epoch smoke test (AUPRC 0.6159); invalid baseline | `artifacts/archived/rgcn_invalid_2epoch_20260611/` |
| GraphSAGE HPO fold-0 (12 configs) | Superseded by 5-fold benchmark | `artifacts/archived/gnn_hpo_search/` |
| R-GCN HPO fold-0 (12 configs) | Superseded by 5-fold benchmark | `artifacts/archived/rgcn_hpo_search/` |
| `graphsage_best.json` | Intermediate h64 quick-run | `artifacts/archived/gnn_intermediate/` |

No interrupted partial-fold benchmarks were detected; all published JSON benchmarks report `n_folds: 5`.

---

## 2. Final benchmark inventory

### 2.1 Model ranking (mean 5-fold AUPRC)

| Rank | Model | AUPRC (mean ± std) | Source |
|------|-------|-------------------|--------|
| 1 | **logistic_regression** | **0.6810 ± 0.0389** | `published/baselines/logistic_regression.json` |
| 2 | fusion_stack_logistic | 0.6671 ± 0.0521 | `published/fusion/fusion_benchmark.json` |
| 3 | catboost | 0.6615 ± 0.0507 | `published/baselines/catboost.json` |
| 4 | rf_centrality | 0.6561 ± 0.0412 | `published/baselines/rf_centrality.json` |
| 5 | rgcn (h128_L2_d0.4) | 0.6542 ± 0.0464 | `published/rgcn/rgcn_benchmark.json` |
| 6 | fusion_rank | 0.6536 ± 0.0395 | `published/fusion/fusion_benchmark.json` |
| 7 | graphsage (h32_L2_d0.3_mean) | 0.6530 ± 0.0429 | `published/gnn/graphsage_benchmark.json` |
| 8 | random_forest | 0.6517 ± 0.0386 | `published/baselines/random_forest.json` |

### 2.2 Gate summaries

| Gate | Criterion | Result |
|------|-----------|--------|
| G2 | Fold-safe heterogeneous graphs | **PASS** |
| G3 | Best baseline AUPRC ≥ threshold | **PASS** (LR 0.6810) |
| G4 | GraphSAGE beats LR | **FAIL** (0.6530 vs 0.6810) |
| G5 | R-GCN beats LR | **FAIL** (0.6542 vs 0.6810) |
| G6 | Fusion beats LR | **FAIL** (best fusion 0.6671 vs 0.6810) |

### 2.3 Published artifact tree (79 files)

```
artifacts/published/
├── MANIFEST.json
├── graphs/v1.1/
│   ├── reference/          # 1 manifest + 3 node + 6 edge parquet
│   └── fold_{0-4}/         # 5 manifests + 15 node + 30 edge parquet each
├── baselines/              # 8 JSON/CSV files (4 models + comparison + significance)
├── gnn/                    # graphsage_benchmark.json, evaluation_summary, hpo_leaderboard.csv
├── rgcn/                   # rgcn_benchmark.json, evaluation_summary, significance
└── fusion/                 # fusion_benchmark.json, evaluation_summary, significance
```

### 2.4 Archived inventory (35 files)

- `rgcn_invalid_2epoch_20260611/` — invalid smoke benchmark + README
- `gnn_hpo_search/` — 12 fold-0 GraphSAGE HPO JSONs
- `rgcn_hpo_search/` — 12 fold-0 R-GCN HPO JSONs
- `gnn_intermediate/` — superseded quick-run outputs
- `docs_state/research_state.json` — internal doc-sync state

### 2.5 Diagnostics inventory (23 files)

- `gnn/experiments.jsonl`, `rgcn/experiments.jsonl`, `fusion/experiments.jsonl`
- `rgcn/ablations/` — 4 fold-0 relation ablation JSONs
- README files

---

## 3. Reproducibility status

| Component | Status | Notes |
|-----------|--------|-------|
| Data preprocessing (G1) | ✓ Documented | `docs/reproducibility.md` § G1 |
| Splits (seed=42) | ✓ Locked | `data/splits/split_manifest.json` |
| Graph construction (G2) | ✓ Frozen | Parquet + manifests in `published/graphs/v1.1/` |
| Baselines (G3) | ✓ Frozen | All 4 models + significance |
| GraphSAGE (G4) | ✓ Frozen | 5-fold benchmark + HPO leaderboard |
| R-GCN (G5) | ✓ Frozen | Publication run `rgcn_pub_v2` (max_epochs=100) |
| Fusion (G6) | ✓ Frozen | 9 fusion variants in benchmark JSON |
| Unit tests | ⚠ Partial | Core CV/fusion tests pass; full suite needs venv + `networkx` |
| Doc sync | ✓ Updated | `scripts/08_sync_research_docs.py --published-dir artifacts/published` |
| Pipeline script defaults | ⚠ Legacy | Scripts still write to `artifacts/results/`; see redirect READMEs |

**Reproduction path:** Follow [`reproducibility.md`](reproducibility.md). After a full re-run, compare outputs against [`artifacts/published/MANIFEST.json`](../artifacts/published/MANIFEST.json).

---

## 4. Research documentation verification

All artifact paths cited in primary research documents now resolve to publication-grade or explicitly archived locations:

| Document | Direct artifact paths? | Verified |
|----------|------------------------|----------|
| [`research_findings.md`](research_findings.md) | Yes — `artifacts/published/` and `artifacts/archived/rgcn_invalid_2epoch_20260611/` | ✓ |
| [`publication_story.md`](publication_story.md) | Metrics only (no paths) | ✓ |
| [`project_status.md`](project_status.md) | Metrics only | ✓ |
| [`experiment_registry.md`](experiment_registry.md) | Metadata table → `published/` | ✓ |
| [`reproducibility.md`](reproducibility.md) | Verification checklist → `published/` | ✓ |
| [`paper_assets.md`](paper_assets.md) | Tables → `published/`; figures → `diagnostics/` | ⚠ PNG plots absent |

---

## 5. Publication readiness assessment

### Ready for publication

- Complete G1–G6 experimental pipeline with locked CV protocol
- Authoritative benchmarks frozen under `artifacts/published/` with `MANIFEST.json`
- Full audit trail in [`artifact_audit.md`](artifact_audit.md) (133 files classified)
- Diagnosis reports: `graphsage_diagnosis.md`, `rgcn_diagnosis.md`, `fusion_diagnosis.md`
- Validity analysis: `threats_to_validity.md`
- Experiment registry and reproducibility guide updated for frozen layout
- Clear negative result narrative: tabular LR dominates graph and fusion models

### Outstanding items (non-blocking for artifact freeze)

| Item | Priority | Action |
|------|----------|--------|
| Diagnostic PNG plots missing | Medium | Regenerate from `diagnostics/*/experiments.jsonl` or re-run plotting |
| Script default paths (`artifacts/results/`) | Low | Pass `--graphs-dir artifacts/published/graphs` or update defaults in a follow-up PR |
| GNNExplainer case studies | Future | Phase 7; listed in `project_status.md` |
| `baseline_manifest.json` internal paths | Low | Still reference legacy `artifacts/results/` strings; metrics unaffected |

### Recommended citation paths for paper

| Asset | Path |
|-------|------|
| Best model results | `artifacts/published/baselines/logistic_regression.json` |
| Model comparison table | `artifacts/published/baselines/baseline_comparison.csv` |
| GraphSAGE benchmark | `artifacts/published/gnn/graphsage_benchmark.json` |
| R-GCN benchmark | `artifacts/published/rgcn/rgcn_benchmark.json` |
| Fusion benchmark | `artifacts/published/fusion/fusion_benchmark.json` |
| Significance tests | `artifacts/published/*/significance_auprc.json` |
| Graph artifacts | `artifacts/published/graphs/v1.1/` |

---

## 6. Freeze checklist

- [x] Audit all experiment artifacts (`artifact_audit.md`)
- [x] Create `artifacts/published/`, `archived/`, `diagnostics/` layout
- [x] Move publication-grade outputs to `published/` (79 files)
- [x] Archive invalid/smoke/HPO/intermediate outputs (35 files)
- [x] Move diagnostics (logs, ablations) to `diagnostics/` (23 files)
- [x] No deletions
- [x] Update research doc artifact references
- [x] Verify `research_findings.md`, `publication_story.md`, `project_status.md`
- [x] Generate this freeze report

---

*Repository frozen for publication. For questions about artifact provenance, start with [`artifact_audit.md`](artifact_audit.md) § 10 (post-move verification).*
