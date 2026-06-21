# Reproducibility Guide

**Project:** Hybrid Explainable Heterogeneous Graph Anomaly Detection for CMS Provider Fraud  
**Repository:** [adirawat-cmd/SRIP-2026](https://github.com/adirawat-cmd/SRIP-2026)  
**Last updated:** 2026-06-21  
**Primary metric:** Mean 5-fold AUPRC (provider-disjoint stratified CV, seed=42)

This guide reproduces the G1–G6 pipeline from raw CMS data through all benchmark artifacts. Commands assume you run from the repository root (`hgad-cms-fraud/`).

> **Publication freeze (2026-06-21):** Final benchmark outputs are frozen under `artifacts/published/`. Scripts still write to `artifacts/results/` by default; after a full re-run, copy or compare against `artifacts/published/` using [`MANIFEST.json`](../artifacts/published/MANIFEST.json). See [`docs/repository_freeze_report.md`](repository_freeze_report.md).

---

## 1. Environment

### Requirements

| Component | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| OS tested | Windows 10/11, Linux |
| GPU | Optional (CUDA for G4–G6; `--device auto` falls back to CPU) |

### Install

```bash
cd hgad-cms-fraud
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e ".[models,gnn,dev]"
```

**Expected output (truncated):**

```
Successfully installed hgad-cms-fraud-0.1.0 catboost-... torch-... torch-geometric-... pyarrow-...
```

Verify imports:

```bash
python -c "import hgad_cms; import torch; import catboost; print('OK')"
```

**Expected output:**

```
OK
```

### Fixed random seeds

| Setting | Value | Location |
|---------|-------|----------|
| CV split seed | `42` | `scripts/02_create_splits.py`, `DEFAULT_SPLIT_SEED` |
| Number of folds | `5` | All CV scripts |
| Graph schema | `v1.1` | `scripts/03_build_graphs.py` |

---

## 2. Data layout

Raw CMS **train** files must be available. The repo includes them under `datasets/`:

```
datasets/Healthcare Provider Fraud Detection Analysis/
  Train-1542865627584.csv
  Train_Beneficiarydata-1542865627584.csv
  Train_Inpatientdata-1542865627584.csv
  Train_Outpatientdata-1542865627584.csv
```

Point preprocessing at this directory (or symlink/copy to `data/raw/`):

```bash
# Optional: symlink for default --raw-dir data/raw
mkdir -p data/raw
# copy or link the four Train*.csv files into data/raw/
```

---

## 3. Full pipeline (G1 → G6)

Run phases in order. Each section lists the **exact command**, **exit code**, **key log lines**, and **artifacts to verify**.

### Quick reference — expected final AUPRC

| Phase | Model / gate | Expected AUPRC | Exit code |
|-------|----------------|----------------|-----------|
| G3 | LR (best) | **0.6810** ± 0.0389 | `0` |
| G4 | GraphSAGE | 0.6530 ± 0.0429 | `0` |
| G5 | R-GCN | 0.6542 ± 0.0464 | `2` (gate fail) |
| G6 | Fusion (best: stack_logistic) | 0.6671 ± 0.0521 | `2` (gate fail) |

> Exit code `2` on G5/G6/Fusion means the script completed but the model did **not** beat LR — this is expected for the published results.

---

### G1 — Data preprocessing

```bash
python scripts/01_preprocess.py \
  --raw-dir "datasets/Healthcare Provider Fraud Detection Analysis" \
  --processed-dir data/processed
```

**Expected log:**

```
INFO | Loaded providers: 5410 rows, ...
INFO | Loaded beneficiaries: 138556 rows, ...
INFO | Loaded Inpatient: ... rows, ...
INFO | Loaded Outpatient: ... rows, ...
INFO | Gate G1 PASSED
INFO | Wrote data/processed/claims.parquet (558211 rows, ...)
INFO | Wrote data/processed/providers.parquet (5410 rows, ...)
INFO | Wrote data/processed/beneficiaries.parquet (138556 rows, ...)
```

**Expected exit code:** `0`

**Verify:**

```bash
python -c "import json; m=json.load(open('data/processed/manifest.json')); print(m['row_counts'], m['gate_g1_passed'])"
```

**Expected output:**

```
{'claims': 558211, 'providers': 5410, 'beneficiaries': 138556, 'fraud_providers': 506} True
```

---

### G1b — Provider-disjoint CV splits

```bash
python scripts/02_create_splits.py \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --n-folds 5 \
  --seed 42
```

**Expected log:**

```
INFO | Fold 0: train=4328 val=1082 fraud_train=405 fraud_val=101
INFO | Fold 1: train=4328 val=1082 fraud_train=405 fraud_val=101
...
INFO | Created 5 folds in data/splits
```

**Expected exit code:** `0`

**Verify:** `data/splits/split_manifest.json` contains `"seed": 42`, `"n_folds": 5`, and per-fold train/val counts of 4328 / 1082.

---

### G2 — Graph construction (schema v1.1)

```bash
python scripts/03_build_graphs.py \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --graphs-dir artifacts/graphs \
  --schema v1.1 \
  --folds all \
  --reference
```

**Expected log:**

```
INFO | Building reference graph schema=v1.1
INFO | Gate G2 PASSED (reference graph)
INFO | Building fold 0 graph ...
...
INFO | Gate G2 PASSED (fold 0)
...
```

**Expected exit code:** `0`

**Verify reference graph** (`artifacts/published/graphs/v1.1/reference/graph_manifest.json`):

| Field | Expected value |
|-------|----------------|
| `gate_g2_passed` | `true` |
| `node_counts.provider` | `5410` |
| `node_counts.beneficiary` | `138556` |
| `node_counts.physician` | `100737` |
| `edge_counts.provider\|treats\|beneficiary` | `363300` |
| `edge_counts.provider\|bills_with\|physician` | `109339` |
| `edge_counts.provider\|collaborates\|provider` | `75604` |

---

### G3 — Tabular baselines

```bash
python scripts/04_run_baselines.py \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --graphs-dir artifacts/graphs \
  --results-dir artifacts/results \
  --schema v1.1 \
  --run-id baselines_v1
```

**Expected log:**

```
INFO | Running fold 0 model=logistic_regression
...
INFO | Gate G3 PASSED (best AUPRC=0.6810)
INFO | Baseline run complete. G3=True
```

**Expected exit code:** `0`

**Verify** (`artifacts/published/baselines/baseline_manifest.json`):

| Model | AUPRC (mean) |
|-------|--------------|
| logistic_regression | **0.6810** |
| catboost | 0.6615 |
| random_forest | 0.6517 |
| rf_centrality | 0.6561 |

Quick check:

```bash
python -c "
import json
m = json.load(open('artifacts/published/baselines/baseline_manifest.json'))
lr = m['results']['logistic_regression']['auprc']['mean']
print(f'LR AUPRC={lr:.4f}  G3={m[\"gate_g3_passed\"]}')
"
```

**Expected output:**

```
LR AUPRC=0.6810  G3=True
```

---

### G4 — GraphSAGE (HPO + 5-fold benchmark)

```bash
python scripts/06_evaluate_graphsage.py \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --graphs-dir artifacts/graphs \
  --results-dir artifacts/results \
  --search-mode eval \
  --search-fold 0 \
  --max-epochs 100 \
  --patience 15 \
  --run-id graphsage_eval \
  --no-resume
```

**Expected log:**

```
INFO | Top config: h32_L2_d0.3_mean_f15-10 AUPRC=0.6413
INFO | Running 5-fold benchmark: h32_L2_d0.3_mean_f15-10
INFO | Evaluation complete. Report: .../docs/graphsage_diagnosis.md
```

**Expected exit code:** `0`

**Verify** (`artifacts/published/gnn/evaluation_summary.json`):

```json
{
  "top_config": "h32_L2_d0.3_mean_f15-10",
  "top_auprc_fold0": 0.6413,
  "benchmark_auprc_mean": 0.6530,
  "beats_lr": false,
  "recommend_rgcn": true
}
```

**Artifacts (published):**
- `artifacts/published/gnn/graphsage_benchmark.json`
- `artifacts/published/gnn/hpo_leaderboard.csv`
- `docs/graphsage_diagnosis.md`
- `artifacts/diagnostics/gnn/plots/*.png` (regenerate from `experiments.jsonl` if missing)

---

### G5 — R-GCN (publication benchmark)

Use the **full training budget**. Do not use 2-epoch smoke settings.

```bash
python scripts/07_train_rgcn.py \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --graphs-dir artifacts/graphs \
  --results-dir artifacts/results \
  --search-mode eval \
  --max-epochs 100 \
  --patience 10 \
  --run-id rgcn_pub_v2 \
  --no-resume
```

**Expected log:**

```
INFO | R-GCN complete. AUPRC=0.6542 G5=False
```

**Expected exit code:** `2` (gate G5 failed — did not beat LR 0.6810)

**Verify** (`artifacts/published/rgcn/evaluation_summary.json`):

```json
{
  "best_config": "h128_L2_d0.4_b4_f15-10",
  "benchmark_auprc": 0.6542,
  "beats_lr": false,
  "beats_graphsage": true,
  "gate_g5_passed": false
}
```

**Artifacts (published + diagnostics):**
- `artifacts/published/rgcn/rgcn_benchmark.json`
- `artifacts/archived/rgcn_hpo_search/` (fold-0 HPO grid; superseded)
- `artifacts/diagnostics/rgcn/ablations/`
- `docs/rgcn_diagnosis.md`

> **Invalid run (do not reproduce):** `rgcn_v1` with `--max-epochs 2` produced AUPRC 0.6159 and was archived under `artifacts/archived/rgcn_invalid_2epoch_20260611/`.

---

### G6 — Hybrid fusion

Requires G4 and G5 benchmarks (loads best configs automatically).

```bash
python scripts/09_train_fusion.py \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --graphs-dir artifacts/graphs \
  --results-dir artifacts/results \
  --schema v1.1 \
  --run-id fusion_v1
```

**Expected log:**

```
INFO | Fusion configs: GraphSAGE=h32_L2_d0.3_mean_f15-10 R-GCN=h128_L2_d0.4_b4_f15-10
INFO | Fusion complete. Best=fusion_stack_logistic AUPRC=0.6671 beats_lr=False
```

**Expected exit code:** `2` (fusion did not beat LR)

**Verify** (`artifacts/published/fusion/evaluation_summary.json`):

```json
{
  "best_model": "logistic_regression",
  "benchmark_auprc": 0.6810,
  "benchmark_auprc_std": 0.0389,
  "beats_lr": false
}
```

Note: `best_model` in the summary reflects the overall winner across all towers; best **fusion** method is `fusion_stack_logistic` at AUPRC **0.6671** (see `fusion_benchmark.json` → `model_summaries`).

**Per-model AUPRC** (`artifacts/published/fusion/fusion_benchmark.json` → `model_summaries`):

| Model | AUPRC |
|-------|-------|
| logistic_regression | 0.6810 |
| fusion_stack_logistic | 0.6671 |
| catboost / fusion_weighted | 0.6615 |
| rgcn | 0.6540 |
| fusion_rank | 0.6536 |
| graphsage | 0.6524 |
| if_graphsage | 0.4476 |
| if_rgcn | 0.4073 |
| if_tabular | 0.3620 |

**Artifacts (published):**
- `artifacts/published/fusion/fusion_benchmark.json`
- `artifacts/published/fusion/significance_auprc.json`
- `docs/fusion_diagnosis.md`

---

### Sync research documentation

After any benchmark script completes, auto-maintained docs can be refreshed manually:

```bash
python scripts/08_sync_research_docs.py --published-dir artifacts/published
```

**Expected log:**

```
INFO | Research documentation synced at ...
INFO | Updated docs/research_findings.md, research_decisions.md, project_status.md, paper_assets.md, publication_story.md
```

**Expected exit code:** `0`

---

## 4. One-shot reproduction script

Save as `reproduce.sh` (Linux/macOS) or run commands sequentially in PowerShell:

```bash
#!/usr/bin/env bash
set -euo pipefail

RAW="datasets/Healthcare Provider Fraud Detection Analysis"

python scripts/01_preprocess.py --raw-dir "$RAW" --processed-dir data/processed
python scripts/02_create_splits.py --seed 42
python scripts/03_build_graphs.py --schema v1.1 --folds all --reference
python scripts/04_run_baselines.py --run-id baselines_v1
python scripts/06_evaluate_graphsage.py --search-mode eval --run-id graphsage_eval --no-resume
python scripts/07_train_rgcn.py --run-id rgcn_pub_v2 --max-epochs 100 --patience 10 --no-resume
python scripts/09_train_fusion.py --run-id fusion_v1
python scripts/08_sync_research_docs.py
```

**PowerShell equivalent (repo root):**

```powershell
$Raw = "datasets/Healthcare Provider Fraud Detection Analysis"
python scripts/01_preprocess.py --raw-dir $Raw --processed-dir data/processed
python scripts/02_create_splits.py --seed 42
python scripts/03_build_graphs.py --schema v1.1 --folds all --reference
python scripts/04_run_baselines.py --run-id baselines_v1
python scripts/06_evaluate_graphsage.py --search-mode eval --run-id graphsage_eval --no-resume
python scripts/07_train_rgcn.py --run-id rgcn_pub_v2 --max-epochs 100 --patience 10 --no-resume
python scripts/09_train_fusion.py --run-id fusion_v1
python scripts/08_sync_research_docs.py
```

---

## 5. Resume / partial re-runs

Most scripts support `--resume` to skip completed work:

```bash
# Re-run only missing baselines
python scripts/04_run_baselines.py --resume

# Re-run GraphSAGE using saved HPO + benchmark JSON
python scripts/06_evaluate_graphsage.py --resume

# Re-run R-GCN HPO/benchmark from saved JSON
python scripts/07_train_rgcn.py --run-id rgcn_pub_v2 --resume
```

If `data/processed/` and published graph manifests already exist in the cloned repo, you can start from **G3** directly (graphs must be rebuilt for full G4–G6).

---

## 6. Unit tests

Core pipeline logic tests (no full GPU training):

```bash
python -m pytest tests/test_splits.py tests/test_fusion.py tests/test_cross_validation.py -q
```

**Expected output (representative):**

```
....................                                                     [100%]
N passed in ...s
```

Run the full suite (some graph integration tests may fail on certain environments):

```bash
python -m pytest tests/ -q
```

---

## 7. Verification checklist

After a full reproduction, confirm against **published** artifacts:

- [ ] `data/processed/manifest.json` → `gate_g1_passed: true`, 506 fraud providers
- [ ] `data/splits/split_manifest.json` → seed 42, 5 folds, 4328/1082 split
- [ ] `artifacts/published/graphs/v1.1/reference/graph_manifest.json` → `gate_g2_passed: true`
- [ ] `artifacts/published/baselines/baseline_manifest.json` → LR AUPRC ≈ **0.6810**, G3 pass
- [ ] `artifacts/published/gnn/evaluation_summary.json` → benchmark ≈ **0.6530**, `beats_lr: false`
- [ ] `artifacts/published/rgcn/evaluation_summary.json` → benchmark ≈ **0.6542**, `gate_g5_passed: false`
- [ ] `artifacts/published/fusion/evaluation_summary.json` → `beats_lr: false`
- [ ] Diagnosis reports exist: `docs/graphsage_diagnosis.md`, `docs/rgcn_diagnosis.md`, `docs/fusion_diagnosis.md`

**Single-command benchmark summary** (after all phases):

```bash
python -c "
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

lr = load('artifacts/published/baselines/baseline_manifest.json')['results']['logistic_regression']['auprc']['mean']
gs = load('artifacts/published/gnn/evaluation_summary.json')['benchmark_auprc_mean']
rg = load('artifacts/published/rgcn/evaluation_summary.json')['benchmark_auprc']
fu = load('artifacts/published/fusion/fusion_benchmark.json')['model_summaries']['fusion_stack_logistic']['auprc']['mean']
print(f'LR={lr:.4f}  GraphSAGE={gs:.4f}  RGCN={rg:.4f}  Fusion={fu:.4f}')
"
```

**Expected output:**

```
LR=0.6810  GraphSAGE=0.6530  RGCN=0.6542  Fusion=0.6671
```

---

## 8. Runtime estimates

| Phase | Approx. time (CPU) | Approx. time (GPU) |
|-------|--------------------|--------------------|
| G1 preprocess | 2–5 min | — |
| G2 graphs | 5–15 min | — |
| G3 baselines | 5–10 min | — |
| G4 GraphSAGE (12 configs + 5-fold) | 2–6 hr | 30–90 min |
| G5 R-GCN (12 configs + ablations + 5-fold) | 3–8 hr | 45–120 min |
| G6 fusion | 1–4 hr | 20–60 min |

Times vary with hardware. Use `--resume` to avoid repeating completed folds.

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: hgad_cms` | Run `pip install -e ".[models,gnn,dev]"` from repo root |
| `ImportError: pyarrow` | `pip install pyarrow>=14` |
| `No file matching pattern 'Train-*.csv'` | Check `--raw-dir` points to CMS train CSV folder |
| R-GCN AUPRC ≈ 0.62 | Likely 2-epoch smoke run; use `--max-epochs 100 --patience 10 --no-resume` |
| Fusion fails on missing configs | Run G4 and G5 first; ensure `graphsage_benchmark.json` and `rgcn_benchmark.json` exist |
| G5/G6 exit code 2 | Normal when model does not beat LR; check JSON artifacts, not exit code alone |

---

## 10. Related documents

- [`experiment_registry.md`](experiment_registry.md) — experiment IDs and AUPRC table  
- [`research_timeline.md`](research_timeline.md) — phase objectives and gate outcomes  
- [`threats_to_validity.md`](threats_to_validity.md) — CV and dataset limitations  
- [`graph_schema.md`](graph_schema.md) — schema v1.1 edge policies
