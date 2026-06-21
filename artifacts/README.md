# Artifacts Layout (Publication Freeze)

**Frozen:** 2026-06-21  
**Audit:** [`docs/artifact_audit.md`](../docs/artifact_audit.md)  
**Freeze report:** [`docs/repository_freeze_report.md`](../docs/repository_freeze_report.md)

## Directories

| Path | Contents |
|------|----------|
| [`published/`](published/) | **Authoritative** publication-grade benchmarks, graphs, and gate summaries |
| [`archived/`](archived/) | Invalid smoke runs, superseded HPO grids, intermediate outputs (nothing deleted) |
| [`diagnostics/`](diagnostics/) | Training logs, plots, fold-0 ablations supporting diagnosis reports |

## Authoritative benchmarks

| Phase | Path |
|-------|------|
| G2 Graphs | `published/graphs/v1.1/` |
| G3 Baselines | `published/baselines/` |
| G4 GraphSAGE | `published/gnn/graphsage_benchmark.json` |
| G5 R-GCN | `published/rgcn/rgcn_benchmark.json` |
| G6 Fusion | `published/fusion/fusion_benchmark.json` |

## Legacy paths

Scripts and docs may still reference `artifacts/results/` or `artifacts/graphs/`.  
See redirect notes in those directories. **Cite `artifacts/published/` for publication.**
