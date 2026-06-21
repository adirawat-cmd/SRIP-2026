# 1. Introduction

## 1.1 Healthcare fraud and Medicare program integrity

Healthcare fraud, waste, and abuse account for tens of billions of dollars in annual losses to U.S. public insurance programs. The Centers for Medicare & Medicaid Services (CMS) and its program integrity contractors investigate providers whose billing patterns deviate from peer norms, exhibit upcoding or unbundling, or involve collusion among entities in the care delivery network. Machine learning has been applied to prioritize providers for audit, but operational deployment demands high precision at constrained investigation budgets and auditable decision support.

The **CMS Healthcare Provider Fraud Detection Analysis** dataset released for research and education provides labeled provider-level fraud indicators alongside beneficiary demographics, inpatient and outpatient claims, and physician involvement. Although not a live enforcement feed, it offers a reproducible benchmark for comparing fraud detection methodologies under class imbalance (approximately 9.35% fraud prevalence at the provider level).

## 1.2 Limitations of existing approaches

Much prior work treats fraud detection as a tabular classification problem: aggregate claims into provider-level features (utilization rates, diagnosis and procedure mixtures, panel characteristics) and train gradient-boosted trees, logistic models, or neural networks. These methods excel when predictive signal resides in scalar summaries but may miss **relational structure**—for example, fraud rings linked through shared beneficiaries, referral patterns, or physician co-occurrence.

Conversely, graph-based approaches model providers as nodes in heterogeneous networks but face practical challenges: **label leakage** across train and validation splits if providers or their neighborhoods overlap improperly; **heterophily**, when fraudulent and legitimate providers connect to similar beneficiaries; and **feature dominance**, when tabular summaries already encode most of the information that message passing would propagate.

## 1.3 Motivation for graph learning

Medicare claims naturally form a heterogeneous graph: providers treat beneficiaries, bill with physicians, and may collaborate with other providers through shared patient panels. Graph neural networks (GNNs) can propagate information along these relations and learn representations that complement tabular features. Homophilic models such as GraphSAGE aggregate neighbor embeddings under the assumption that connected nodes share similar labels; relation-aware models such as R-GCN assign separate transformation weights per edge type, which may better capture the distinct semantics of treatment, billing, and collaboration relations.

We hypothesized that graph structure would improve provider fraud ranking beyond a strong tabular baseline when evaluated under **provider-disjoint cross-validation**, which holds out entire providers per fold and rebuilds fold-safe graphs from training claims only.

## 1.4 Motivation for explainability

Program integrity investigators require more than a risk score: they need **interpretable evidence** linking predictions to billing patterns, peer groups, or anomalous subgraphs. Explainable AI (XAI) for GNNs—including attention weights, subgraph explanations, and feature attribution—could support audit workflows if graph models demonstrate additive value. Our project title emphasizes hybrid **explainable** heterogeneous graph anomaly detection; however, the experiments reported here focus on predictive benchmarking and ablation analysis. Explainability case studies (e.g., GNNExplainer) are identified as future work and are not claimed as completed contributions in this manuscript.

## 1.5 Contributions and findings

This paper reports a complete G1–G6 experimental pipeline with frozen, reproducible artifacts:

1. **Fold-safe heterogeneous graph construction (schema_v1.1)** with provider–provider *collaborates* edges requiring at least two shared beneficiaries.
2. **Tabular baselines** including logistic regression, random forest, CatBoost, and random forest with graph centrality features.
3. **GraphSAGE and R-GCN benchmarks** with hyperparameter search on fold 0 and 5-fold confirmation of the best configuration.
4. **Hybrid fusion** combining supervised scores (tabular and GNN) with isolation-forest anomaly scores via weighted averaging, stacked logistic meta-learning, and rank fusion.
5. **Rigorous evaluation** using mean AUPRC as the primary metric under provider-disjoint stratified 5-fold CV (seed = 42).

**Primary finding:** Logistic regression achieves the highest mean AUPRC (**0.6810**) and none of the graph or fusion methods beat this baseline in the locked evaluation protocol. We analyze why, discuss threats to validity, and outline directions for explainability and richer relational modeling.

## 1.6 Paper organization

Section 2 reviews related work. Section 3 analyzes the research gap. Section 4 describes methodology. Section 5 details experimental setup. Section 6 presents results and ablations. Section 7 discusses implications. Section 8 concludes. Appendices provide hyperparameters, graph schema details, supplementary results, and reproduction instructions.
