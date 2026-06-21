# Artifact Audit

**Audit date:** 2026-06-21  
**Auditor role:** Research engineering lead (publication freeze)  
**Scope:** All files under `artifacts/` (133 files, ~85 MB excluding large parquet)  
**Policy:** No deletions. Archive-only reorganization into `published/`, `archived/`, `diagnostics/`.

---

## Executive summary

| Category | File count | Action |
|----------|------------|--------|
| Publication-grade benchmarks & graphs | 89 | → `artifacts/published/` |
| HPO fold-0 search grids (superseded) | 24 | → `artifacts/archived/` |
| Invalid smoke benchmark (2-epoch R-GCN) | 7 | → `artifacts/archived/` |
| Intermediate / duplicate GNN outputs | 2 | → `artifacts/archived/` |
| Training logs, plots, ablations | 10 | → `artifacts/diagnostics/` |
| Internal doc-sync state | 1 | → `artifacts/archived/` |

**Final publication benchmarks (authoritative):**

| Experiment | Run ID | AUPRC | Artifact |
|------------|--------|-------|----------|
| LR (G3) | `baselines_v1` | **0.6810** | `published/baselines/logistic_regression.json` |
| GraphSAGE (G4) | `graphsage_eval` | 0.6530 | `published/gnn/graphsage_benchmark.json` |
| R-GCN (G5) | `rgcn_pub_v2` | 0.6542 | `published/rgcn/rgcn_benchmark.json` |
| Fusion (G6) | `fusion_v1` | 0.6671 (best fusion) | `published/fusion/fusion_benchmark.json` |

---

## Classification legend

| Decision | Meaning |
|----------|---------|
| **KEEP → published** | Cited in paper/research docs; 5-fold CV; full training budget |
| **KEEP → diagnostics** | Supports diagnosis reports; not primary benchmark |
| **ARCHIVE** | Smoke test, interrupted, superseded HPO, invalid, or duplicate |
| **ARCHIVE (already)** | Previously archived invalid run |

---

## 1. Graph artifacts (schema v1.1)

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/graphs/v1.1/reference/graph_manifest.json` | Full-corpus reference graph metadata | **published** | G2 gate passed; cited in F002 |
| `artifacts/graphs/v1.1/reference/nodes_*.parquet` (3) | Reference node features | **published** | Required for graph reproduction |
| `artifacts/graphs/v1.1/reference/edges_*.parquet` (6) | Reference edge lists | **published** | Required for graph reproduction |
| `artifacts/graphs/v1.1/fold_{0-4}/graph_manifest.json` (5) | Fold-safe train graph metadata | **published** | G2 passed; CV graph inputs |
| `artifacts/graphs/v1.1/fold_{0-4}/nodes_*.parquet` (15) | Fold node features | **published** | GNN/baseline graph features |
| `artifacts/graphs/v1.1/fold_{0-4}/edges_*.parquet` (30) | Fold edge lists | **published** | Fold-safe message passing |

**Total graph files:** 60 → all **published** (no invalid graph builds detected).

---

## 2. Baseline results (G3)

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/results/baselines/baseline_manifest.json` | G3 gate summary | **published** | Primary G3 evidence |
| `artifacts/results/baselines/baseline_comparison.csv` | Model comparison table | **published** | Paper table asset |
| `artifacts/results/baselines/baseline_comparison.json` | JSON comparison table | **published** | Machine-readable duplicate of CSV |
| `artifacts/results/baselines/significance_auprc.json` | Paired significance tests | **published** | Statistical comparison |
| `artifacts/results/baselines/logistic_regression.json` | LR 5-fold CV | **published** | **Best model** AUPRC 0.6810 |
| `artifacts/results/baselines/catboost.json` | CatBoost 5-fold CV | **published** | Primary baseline comparison |
| `artifacts/results/baselines/random_forest.json` | RF 5-fold CV | **published** | Primary baseline comparison |
| `artifacts/results/baselines/rf_centrality.json` | RF+centrality 5-fold CV | **published** | Graph feature ablation baseline |

---

## 3. GraphSAGE results (G4)

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/results/gnn/graphsage_benchmark.json` | **5-fold benchmark** (best config) | **published** | Authoritative G4 result; AUPRC 0.6530 |
| `artifacts/results/gnn/evaluation_summary.json` | Gate G4 summary | **published** | Cited in F004–F007 |
| `artifacts/results/gnn/hpo_leaderboard.csv` | Top-12 HPO ranking (fold 0) | **published** | Paper table; config selection record |
| `artifacts/results/gnn/hpo_search/*.json` (12 files) | Fold-0 HPO per config | **archived** | Superseded by 5-fold `graphsage_benchmark.json` |
| `artifacts/results/gnn/graphsage_best.json` | Early quick-run best (h64) | **archived** | Superseded; config ≠ final h32 benchmark |
| `artifacts/results/gnn/graphsage_h64_L2_d0.3_mean_f15-10.json` | Single-config duplicate | **archived** | Duplicate of hpo_search entry |
| `artifacts/results/gnn/experiments.jsonl` | Training epoch logs | **diagnostics** | Reproducibility / curve regeneration |
| `artifacts/results/gnn/plots/*.png` (9 files) | HPO curves & factor plots | **diagnostics** | Referenced in `graphsage_diagnosis.md` |

**Interrupted runs:** None detected (HPO completed 12/12; benchmark completed).

**Smoke tests:** None in GNN folder (contrast with R-GCN invalid run).

---

## 4. R-GCN results (G5)

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/results/rgcn/rgcn_benchmark.json` | **5-fold benchmark** (`max_epochs=100`) | **published** | Authoritative G5 result; AUPRC 0.6542 |
| `artifacts/results/rgcn/evaluation_summary.json` | Gate G5 summary | **published** | Cited in F010–F011 |
| `artifacts/results/rgcn/significance_auprc.json` | vs LR/CatBoost/GraphSAGE | **published** | Statistical evidence |
| `artifacts/results/rgcn/hpo_search/*.json` (12 files) | Fold-0 HPO grid | **archived** | Superseded by 5-fold benchmark |
| `artifacts/results/rgcn/ablations/*.json` (4 files) | Relation ablations (fold 0) | **diagnostics** | Cited in `rgcn_diagnosis.md`, `paper_assets.md` |
| `artifacts/results/rgcn/experiments.jsonl` | Training epoch logs | **diagnostics** | Curve/plot regeneration |
| `artifacts/results/rgcn/plots/*.png` (6 files) | Training curves | **diagnostics** | Paper figure candidates |

### Invalid / smoke benchmark (ARCHIVE)

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/results/rgcn/archive/invalid_2epoch_20260611_173022/README.txt` | Archive notice | **archived** | Documents invalid run |
| `.../rgcn_benchmark.json` | **2-epoch smoke benchmark** | **archived** | AUPRC 0.6159; `max_epochs=2`; **not publication-grade** (F009) |
| `.../evaluation_summary.json` | Smoke summary | **archived** | Invalid gate decision input |
| `.../significance_auprc.json` | Smoke significance | **archived** | Invalid comparison |
| `.../experiments.jsonl` | Smoke training log | **archived** | Incomplete training |
| `.../hpo_search/*.json` (2 files) | Smoke HPO (partial grid) | **archived** | Interrupted / wrong budget |

**Note:** Final `rgcn_benchmark.json` at results root is the valid `rgcn_pub_v2` run (`max_epochs=100`, `patience=10`).

---

## 5. Fusion results (G6)

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/results/fusion/fusion_benchmark.json` | **5-fold fusion benchmark** | **published** | Authoritative G6 result |
| `artifacts/results/fusion/evaluation_summary.json` | Gate G6 summary | **published** | Best-model ranking |
| `artifacts/results/fusion/significance_auprc.json` | Fusion vs baselines | **published** | Statistical evidence |
| `artifacts/results/fusion/experiments.jsonl` | Fusion training log | **diagnostics** | Audit trail |

---

## 6. Internal / non-experiment artifacts

| Path (pre-move) | Purpose | Decision | Justification |
|-----------------|---------|----------|---------------|
| `artifacts/docs_state/research_state.json` | Auto-sync doc state | **archived** | Internal automation; not a scientific result |

---

## 7. Issues identified

| Issue | Severity | Resolution |
|-------|----------|------------|
| Invalid 2-epoch R-GCN benchmark (AUPRC 0.6159) | **High** | Already isolated; move to `archived/` |
| `graphsage_best.json` uses h64, not final h32 config | Medium | Archive; cite `graphsage_benchmark.json` only |
| 24 fold-0 HPO JSON files duplicate benchmark summaries | Low | Archive; retain `hpo_leaderboard.csv` |
| `research_findings.md` F008 lists experiment `rgcn_v1` but cites valid benchmark path | Low | Narrative fix in freeze report |
| Legacy paths `artifacts/results/` hardcoded in scripts | Medium | Redirect READMEs + update `research_docs.py` |
| No interrupted partial folds detected in JSON artifacts | — | All 5-fold benchmarks have `n_folds: 5` |

---

## 8. Post-reorganization target layout

```
artifacts/
├── published/          # Publication-grade outputs only
│   ├── graphs/v1.1/
│   ├── baselines/
│   ├── gnn/
│   ├── rgcn/
│   └── fusion/
├── archived/           # Invalid, smoke, superseded HPO, internal state
│   ├── rgcn_invalid_2epoch_20260611/
│   ├── gnn_hpo_search/
│   ├── gnn_intermediate/
│   ├── rgcn_hpo_search/
│   └── docs_state/
├── diagnostics/        # Plots, ablations, experiment logs
│   ├── gnn/
│   ├── rgcn/
│   └── fusion/
├── README.md           # Layout guide
└── results/README.md   # Redirect (legacy path)
```

---

## 9. Research doc cross-reference verification (pre-move)

| Document | Artifact references | Valid? |
|----------|---------------------|--------|
| `research_findings.md` | `artifacts/graphs/...`, `artifacts/results/baselines/`, `gnn/graphsage_benchmark.json`, `rgcn/rgcn_benchmark.json`, `rgcn/archive/invalid_2epoch_*`, `artifacts/results/` (fusion) | Valid paths pre-move; invalid archive correctly cited as withdrawn |
| `publication_story.md` | No direct artifact paths; metrics match published benchmarks | ✓ |
| `project_status.md` | No direct artifact paths; AUPRC 0.6810 matches LR benchmark | ✓ |

**Post-move action:** Update paths to `artifacts/published/...` in research docs and `research_docs.py`.

---

*This audit was completed before any file moves. Reorganization executed immediately after audit sign-off.*

---

## 10. Post-reorganization verification (2026-06-21)

### Layout counts

| Tier | Files | Role |
|------|-------|------|
| `artifacts/published/` | 79 | Authoritative benchmarks, graphs, gate summaries |
| `artifacts/archived/` | 35 | Invalid 2-epoch R-GCN, fold-0 HPO grids, intermediate GNN, doc state |
| `artifacts/diagnostics/` | 23 | Experiment logs, R-GCN ablations (plots referenced but not present in repo) |

### Research doc cross-reference (post-move)

| Document | Artifact references | Valid? |
|----------|---------------------|--------|
| `research_findings.md` | All paths under `artifacts/published/` or `artifacts/archived/rgcn_invalid_2epoch_20260611/` | ✓ |
| `publication_story.md` | No direct paths; metrics match `MANIFEST.json` | ✓ |
| `project_status.md` | No direct paths; AUPRC 0.6810 matches LR benchmark | ✓ |
| `reproducibility.md` | Verification checklist uses `artifacts/published/` | ✓ |
| `experiment_registry.md` | Metadata table uses `artifacts/published/` | ✓ |
| `paper_assets.md` | Tables → `published/`; figures → `diagnostics/` (PNG files absent) | ⚠ plots missing |

### Known gaps

1. **Diagnostic plots:** `paper_assets.md` and `graphsage_diagnosis.md` reference 15 PNG training curves under `artifacts/diagnostics/*/plots/`; files are not in the repository (regenerate from `experiments.jsonl` or re-run HPO plotting).
2. **Script defaults:** Pipeline scripts still default to `artifacts/results/` and `artifacts/graphs/` for new runs; frozen outputs live under `artifacts/published/`.
3. **F009 narrative:** Correctly cites archived invalid run; F010/F008 both document valid `rgcn_pub_v2` benchmark (intentional audit trail).

See [`repository_freeze_report.md`](repository_freeze_report.md) for full publication readiness assessment.
