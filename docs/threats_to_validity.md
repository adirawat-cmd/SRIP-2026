# Threats to Validity

**Project:** Hybrid Explainable Heterogeneous Graph Anomaly Detection for CMS Provider Fraud  
**Last updated:** 2026-06-11  
**Evaluation protocol:** 5-fold provider-disjoint stratified CV (seed=42); primary metric AUPRC

This document records internal and external validity threats for the G1–G6 experiments. Each section states the threat, what we did to mitigate it, and what remains unresolved.

---

## 1. Provider-disjoint cross-validation

### Threat

Random or beneficiary-level splits would leak information across train and validation because beneficiaries, physicians, and claims are shared across providers. A provider seen in training could indirectly expose validation labels through graph neighborhoods or aggregated tabular statistics.

Even with provider-disjoint splits, **generalization is limited to new providers within the same CMS training corpus**. Folds partition the same static snapshot; they do not test temporal drift, new geographies, or new billing schemes.

### Mitigation

- All models evaluated under **provider-disjoint stratified 5-fold CV** (`scripts/02_create_splits.py`, `StratifiedKFold`, seed=42).
- Train and validation provider sets are verified disjoint per fold; stratification preserves fraud prevalence (~9.35%) in every fold.
- Tabular features, scalers, diagnosis/procedure vocabularies, and graph edges are fit from **train-fold data only**.
- Fold-safe graphs built from train-fold claims only (`artifacts/published/graphs/v1.1/fold_{0–4}/`).

### Residual risk

| Issue | Impact |
|-------|--------|
| Small corpus (5,410 providers; ~101 fraud providers per fold) | High fold-to-fold variance (e.g., LR AUPRC std ≈ 0.039) |
| HPO on fold 0 only (GraphSAGE, R-GCN) | Selected configs may be fold-specific; full 5-fold mean is the reported benchmark |
| No temporal or geographic holdout | Deployment performance on future CMS data is unknown |
| Wilcoxon tests often non-significant at α=0.05 | Differences between models (e.g., GraphSAGE vs LR, p=0.0625) may be underpowered |

**Conclusion:** Provider-disjoint CV gives a **conservative, leakage-aware** estimate within this dataset. It does **not** establish external validity on unseen CMS releases or real-world audit populations.

---

## 2. Transductive beneficiaries

### Threat

GNN evaluation is **inductive for providers** (validation providers are held out during training) but **transductive for beneficiaries and physicians**. At inference, validation providers are appended to the train-fold graph via `build_inference_graph`; their `treats` and `bills_with` edges connect to beneficiary and physician nodes that were already present in the training graph.

Message passing therefore flows from train-side structure into validation provider representations. If beneficiary neighborhoods encode provider-label signal (e.g., fraud and legitimate providers share panels), validation scores may benefit from structural information not available for a fully inductive deployment on entirely new patient populations.

### Mitigation

- Validation **provider IDs** never appear in the training graph or training loss.
- Validation provider features and edges are built from **validation-fold claims only** (no validation claims used during train-graph construction).
- Provider–provider `collaborates` edges for validation providers use validation-fold claims only; train–val provider pairs are not linked via PP edges computed on the full corpus.

### Residual risk

| Issue | Impact |
|-------|--------|
| Shared beneficiaries across train/val providers | Semi-transductive setup; GNN AUPRC may be **optimistic** relative to fully inductive beneficiary graphs |
| Beneficiary/physician nodes lack fraud labels | Models may exploit demographic or utilization proxies correlated with provider labels |
| New beneficiaries in val folds are appended but often sparse | Most val provider edges attach to **existing** train beneficiaries |

**Conclusion:** Provider-disjoint CV prevents the most severe leakage, but **beneficiary transductivity** remains a threat to strict inductive claims. Graph model results should be interpreted as performance under semi-transductive inference, not cold-start deployment on new beneficiary universes.

---

## 3. Class imbalance

### Threat

Provider-level fraud prevalence is **9.35%** (506 / 5,410). Rare positives inflate ROC-AUC (all models ≈ 0.91–0.93) while precision remains moderate. Models can achieve high recall by flagging many providers, yielding misleading F1 or accuracy impressions.

### Mitigation

- **AUPRC** chosen as the primary metric (Decision D006); gates G4–G6 defined on mean AUPRC vs LR.
- GNN training uses **positive-class weighting** in binary cross-entropy.
- Report precision, recall, F1, and recall@K alongside AUPRC in all benchmark artifacts.

### Residual risk

| Issue | Impact |
|-------|--------|
| ~101 fraud providers per validation fold | Unstable precision and high variance in rare-class metrics |
| No cost-sensitive or audit-budget calibration | AUPRC ranking does not map to operational investigation capacity |
| Isolation Forest (G6) assumes anomalies are minority | Fraud may not present as embedding-space outliers (IF AUPRC 0.36–0.45) |

**Observed pattern:** LR recall ≈ 0.85 with precision ≈ 0.39; GraphSAGE recall ≈ 0.75 with precision ≈ 0.46. High-recall / lower-precision behavior is consistent with imbalance-driven threshold effects.

**Conclusion:** AUPRC mitigates the worst imbalance bias, but **operational utility** at a fixed investigation budget (top-K providers) should be reported separately and may not align with AUPRC rankings.

---

## 4. CMS dataset limitations

### Threat

The **CMS Healthcare Provider Fraud Detection Analysis** train split is a curated, educational corpus—not a live CMS fraud enforcement feed. Labels (`PotentialFraud`) are binary provider flags without fraud typology, severity, or adjudication timeline. Results may not transfer to real audit workflows, other CMS programs, or other payers.

### Mitigation

- Single reproducible pipeline from documented raw files (`data/processed/manifest.json` with SHA256 checksums).
- Provider as the prediction unit aligns with CMS program integrity framing.
- Negative graph/fusion results reported transparently rather than over-claiming graph superiority.

### Residual risk

| Issue | Impact |
|-------|--------|
| **Train split only** — no held-out CMS test set in benchmarks | No publisher-style locked test evaluation |
| Static time window | No temporal validation; billing rule changes not modeled |
| Label noise and definition unknown | False positives/negatives in `PotentialFraud` propagate to all metrics |
| Limited provider count (5,410) | Under-represents specialty, region, and scheme diversity |
| Claim/beneficiary tables are simplified | Missing external data (appeals, investigations, social network beyond claims) |

**Conclusion:** Findings characterize **this CMS benchmark corpus** under rigorous internal CV. External validity to production CMS fraud detection requires evaluation on independent, temporally separated, and adjudicated data.

---

## 5. Graph construction assumptions

### Threat

The heterogeneous graph is a **model of claimed relationships**, not ground-truth clinical or collusion networks. Several assumptions embed structural bias:

1. **Provider–beneficiary `treats` edges** — one edge per (Provider, BeneID) from claims; assumes co-occurrence implies a stable care relationship.
2. **Provider–physician `bills_with` edges** — derived from billing references; physicians filtered by minimum claim references.
3. **Provider–provider `collaborates` edges** — pairs sharing **≥ 2 beneficiaries** (schema v1.1); shared panels proxy for professional overlap, not proven coordination.
4. **Claims not materialized as nodes** — claim-level sequencing, upcoding patterns, and multi-claim bursts are collapsed into provider aggregates.
5. **Edge weights** — `log1p(n_shared_beneficiaries)` for PP edges assumes more sharing implies stronger relational signal.

If these assumptions misrepresent fraud mechanisms (e.g., fraud is claim-pattern-based rather than network-based), GNN and fusion models are testing the wrong structural hypothesis.

### Mitigation

- Publication-focused schema v1.1 with PP threshold ≥ 2 (reduced 168K → 76K noisy single-shared pairs).
- Fold-safe construction prevents validation claims from influencing train-graph topology.
- R-GCN ablations (no PP, no treats, no bills) quantify edge-type contribution; removing `bills_with` hurt most.
- RF + centrality baseline tests whether simple graph statistics add signal beyond tabular features (AUPRC 0.6561 vs LR 0.6810).

### Residual risk

| Issue | Impact |
|-------|--------|
| Threshold ≥ 2 is heuristic | Alternative thresholds (1, 5) not fully benchmarked in final gates |
| No ground-truth collusion labels | PP edges conflate legitimate group practice with fraud rings |
| Physician/beneficiary features are demographic/utilization summaries | May encode spurious correlations |
| Homogeneous treatment of edge types in GraphSAGE | Motivated R-GCN, which still failed to beat LR |

**Conclusion:** Graph construction choices are **explicit and auditable**, but unverified against real fraud topology. The null result (graph ≈ tabular) may reflect true feature dominance **or** misspecified graph semantics.

---

## 6. Heterophily

### Threat

Homophilic GNNs (GraphSAGE, and to a lesser extent R-GCN) assume connected nodes share similar labels. In this graph, **fraud and legitimate providers are intermixed**:

- Cross-label provider–provider edge fraction ≈ **40.5%** (schema v1.1 reference counts).
- Label homophily on PP edges ≈ **0.595** (below strong homophily regimes where GNNs excel).
- Fraudulent and legitimate providers **share beneficiaries and physicians** through `treats` and `bills_with` edges.

Mean aggregation propagates neighborhood embeddings across opposing labels, diluting fraud-specific signal. This is consistent with GraphSAGE plateauing at AUPRC 0.6530 despite reasonable ROC-AUC (0.915).

### Mitigation

- Diagnosis reports flag heterophily as a root cause (`docs/graphsage_diagnosis.md`).
- Tested relation-aware R-GCN (separate weights per edge type); marginal gain over GraphSAGE (+0.001 AUPRC), still below LR.
- Max aggregator tested in HPO; underperformed mean, suggesting noisy neighborhoods rather than missing salient neighbors.

### Residual risk

| Issue | Impact |
|-------|--------|
| No heterophily-aware architectures (e.g., H2GCN, GPR-GNN) evaluated | Alternative graph hypothesis untested |
| Fraud may be **attribute-driven** within heterogeneous neighborhoods | Graph methods cannot recover label signal that is not structurally local |
| IF on embeddings assumes outliers are fraud | Fraud providers are not embedding anomalies (IF AUPRC < 0.45) |

**Conclusion:** Heterophily is a **plausible and evidence-supported** explanation for GNN underperformance relative to tabular LR. It is not independently proven—alternative explanations (feature dominance, insufficient model capacity) coexist—but cross-label edge statistics and diagnosis reports support it.

---

## Summary matrix

| Threat | Severity | Mitigation strength | Affects most |
|--------|----------|---------------------|--------------|
| Provider-disjoint CV (incomplete external validity) | Medium | Strong (internal) | All phases |
| Transductive beneficiaries | Medium–High | Partial | G4, G5, G6 (GNN/fusion) |
| Class imbalance | Medium | Strong (metric choice) | All supervised models |
| CMS dataset limitations | High | Weak (inherent to data) | External generalization |
| Graph construction assumptions | Medium–High | Partial (ablations) | G2–G6 |
| Heterophily | Medium | Partial (R-GCN, diagnosis) | G4–G6 |

---

## Recommended reporting language

For publication or stakeholder summaries, we recommend:

1. **Internal validity:** Results are under provider-disjoint stratified CV with fold-safe graphs and train-only feature fitting.
2. **Graph models:** GNN scores reflect semi-transductive inference with shared beneficiary nodes; they are not fully inductive on new patient populations.
3. **External validity:** Findings apply to the CMS Healthcare Provider Fraud Detection **train benchmark**; generalization to production CMS data is unverified.
4. **Negative results:** Graph and fusion methods not beating LR (AUPRC 0.6810) are evidence about **this corpus and graph schema**, not a universal statement that graphs never help fraud detection.

---

## Related documents

- [`research_timeline.md`](research_timeline.md) — phase outcomes and lessons learned  
- [`graph_schema.md`](graph_schema.md) — edge policies and homophily statistics  
- [`experiment_registry.md`](experiment_registry.md) — experiment IDs and AUPRC benchmarks  
- [`graphsage_diagnosis.md`](graphsage_diagnosis.md), [`rgcn_diagnosis.md`](rgcn_diagnosis.md), [`fusion_diagnosis.md`](fusion_diagnosis.md) — phase-specific validity notes
