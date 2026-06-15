# Project Status

Last updated: 2026-06-11T14:15:38.185547+00:00

**Current Phase:**
Phase 6 — Fusion / Explainability (planned)

**Last Completed Gate:**
G5 (failed)

**Current Best Model:**
logistic_regression

**Current Best AUPRC:**
0.6810

**Pending Experiments:**
Fusion model (tabular + graph); GNNExplainer case studies; optional R-GCN full HPO if eval-mode results insufficient

**Known Risks:**
Tabular LR dominates GNNs; class imbalance drives high recall/low precision; 4GB GPU limits model depth; val providers require inductive inference graphs

**Next Action:**
Implement fusion or richer edge features; document why GNNs underperform LR
