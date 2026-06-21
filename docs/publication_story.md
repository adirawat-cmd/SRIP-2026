# Publication Story

Living narrative for the CMS provider fraud detection paper.

Last updated: 2026-06-21T15:04:07.874831+00:00

## Research Problem

Detect fraudulent Medicare providers using heterogeneous relational data (providers, beneficiaries, physicians) with explainable graph-based models.


## Literature Gap

Prior CMS fraud work relies on tabular features or homogeneous graphs; relation-specific heterogeneous modeling under provider-disjoint CV is under-tested.


## Hypotheses

H1: Graph structure improves AUPRC beyond tabular LR. H2: Relation-aware R-GCN beats homophilic GraphSAGE. H3: PP collaborates edges carry fraud signal beyond treats/bills_with.


## Experimental Evidence

G3: LR AUPRC 0.6810 beats CatBoost/RF. G4 failed: GraphSAGE AUPRC 0.6530. G5 failed: R-GCN AUPRC 0.6542. G6 failed: LR remains best overall; best fusion fusion_stack_logistic AUPRC 0.6671. Mean aggregation and 2-layer depth preferred for GraphSAGE.


## Current Conclusions

Strong tabular baseline (LR) sets high bar. GraphSAGE homophilic message passing does not overcome feature dominance. R-GCN and hybrid fusion also fail to beat LR under provider-disjoint CV; explainability case studies remain the primary path to publication-grade contribution.

