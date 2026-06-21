"""
Automated research documentation maintenance.

Updates docs/research_findings.md, research_decisions.md, project_status.md,
paper_assets.md, and publication_story.md from experiment artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
PUBLISHED_DIR = Path("artifacts/published")
DIAGNOSTICS_DIR = Path("artifacts/diagnostics")
ARCHIVED_DIR = Path("artifacts/archived")
STATE_DIR = ARCHIVED_DIR / "docs_state"
STATE_PATH = STATE_DIR / "research_state.json"

FINDINGS_PATH = DOCS_DIR / "research_findings.md"
DECISIONS_PATH = DOCS_DIR / "research_decisions.md"
STATUS_PATH = DOCS_DIR / "project_status.md"
ASSETS_PATH = DOCS_DIR / "paper_assets.md"
STORY_PATH = DOCS_DIR / "publication_story.md"

LR_BASELINE_AUPRC = 0.6810
GRAPHSAGE_BENCHMARK_AUPRC = 0.6530


@dataclass
class Finding:
    finding_id: str
    date: str
    phase: str
    observation: str
    evidence: str
    impact: str
    follow_up_action: str
    experiment_id: str = ""


@dataclass
class Decision:
    decision_id: str
    date: str
    decision: str
    alternatives_considered: str
    reason: str
    expected_impact: str
    experiment_id: str = ""


@dataclass
class ResearchState:
    findings: list[Finding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    project_status: dict[str, Any] = field(default_factory=dict)
    paper_assets: dict[str, list[str]] = field(default_factory=dict)
    publication_story: dict[str, str] = field(default_factory=dict)
    last_synced_utc: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_state() -> ResearchState:
    if not STATE_PATH.is_file():
        return _bootstrap_state()
    with STATE_PATH.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return ResearchState(
        findings=[Finding(**f) for f in raw.get("findings", [])],
        decisions=[Decision(**d) for d in raw.get("decisions", [])],
        project_status=raw.get("project_status", {}),
        paper_assets=raw.get("paper_assets", {}),
        publication_story=raw.get("publication_story", {}),
        last_synced_utc=raw.get("last_synced_utc", ""),
    )


def _save_state(state: ResearchState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "findings": [asdict(f) for f in state.findings],
        "decisions": [asdict(d) for d in state.decisions],
        "project_status": state.project_status,
        "paper_assets": state.paper_assets,
        "publication_story": state.publication_story,
        "last_synced_utc": state.last_synced_utc,
    }
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _upsert_finding(state: ResearchState, finding: Finding) -> None:
    state.findings = [f for f in state.findings if f.finding_id != finding.finding_id]
    state.findings.append(finding)
    state.findings.sort(key=lambda f: f.finding_id)


def _upsert_decision(state: ResearchState, decision: Decision) -> None:
    state.decisions = [d for d in state.decisions if d.decision_id != decision.decision_id]
    state.decisions.append(decision)
    state.decisions.sort(key=lambda d: d.decision_id)


def _bootstrap_state() -> ResearchState:
    """Seed findings and decisions from completed project phases."""
    state = ResearchState()
    today = _utc_date()

    seed_findings = [
        Finding(
            finding_id="F001",
            date="2026-06-06",
            phase="Phase 1 — Data Pipeline",
            observation="Provider-level fraud rate is 9.35% (506/5410 providers).",
            evidence="data/processed/manifest.json; G1 validation passed.",
            impact="Establishes prevalence floor for AUPRC benchmark (0.0935).",
            follow_up_action="Use stratified provider-disjoint CV.",
            experiment_id="preprocess_v1",
        ),
        Finding(
            finding_id="F002",
            date="2026-06-11",
            phase="Phase 2 — Graph Construction",
            observation="schema_v1.1 reference graph: 363K treats, 109K bills_with, 76K collaborates edges.",
            evidence="artifacts/published/graphs/v1.1/reference/graph_manifest.json; Gate G2 passed.",
            impact="Confirms fold-safe heterogeneous graph artifacts for GNN training.",
            follow_up_action="Proceed to tabular and graph baselines.",
            experiment_id="graph_construct_v1_1",
        ),
        Finding(
            finding_id="F003",
            date="2026-06-11",
            phase="Phase 3 — Baselines",
            observation="Logistic Regression outperformed CatBoost, RF, and RF+Centrality on mean AUPRC.",
            evidence="LR AUPRC 0.6810±0.0389 vs CatBoost 0.6615±0.0507 (artifacts/published/baselines/).",
            impact="LR becomes primary tabular benchmark; Gate G3 passed.",
            follow_up_action="Test whether graph structure adds value beyond tabular features.",
            experiment_id="baselines_v1",
        ),
        Finding(
            finding_id="F004",
            date="2026-06-11",
            phase="Phase 4 — GraphSAGE",
            observation="GraphSAGE failed to beat Logistic Regression (5-fold AUPRC 0.6530 vs 0.6810).",
            evidence="docs/graphsage_diagnosis.md; artifacts/published/gnn/graphsage_benchmark.json.",
            impact="Gate G4 failed; homophilic GraphSAGE insufficient for fraud heterophily.",
            follow_up_action="Proceed to relation-aware R-GCN (Phase 5).",
            experiment_id="graphsage_eval",
        ),
        Finding(
            finding_id="F005",
            date="2026-06-11",
            phase="Phase 4 — GraphSAGE",
            observation="Mean neighbor aggregation outperformed max aggregation on fold-0 HPO.",
            evidence="Mean AUPRC 0.631 vs max 0.598 across 12 HPO configs.",
            impact="Select mean aggregation for future homophilic GNN baselines.",
            follow_up_action="Document in graph model design guidelines.",
            experiment_id="graphsage_eval",
        ),
        Finding(
            finding_id="F006",
            date="2026-06-11",
            phase="Phase 4 — GraphSAGE",
            observation="2-layer GraphSAGE outperformed 3-layer on fold-0 HPO.",
            evidence="Layer-2 mean AUPRC 0.622 vs layer-3 mean 0.607.",
            impact="Prefer shallow GNNs on 4GB GPU constraint; reduces overfitting risk.",
            follow_up_action="Cap default depth at 2 layers for HPO quick mode.",
            experiment_id="graphsage_eval",
        ),
        Finding(
            finding_id="F007",
            date="2026-06-11",
            phase="Phase 4 — GraphSAGE",
            observation="Hidden dimension 32 achieved best fold-0 AUPRC (0.6413).",
            evidence="h32_L2_d0.3_mean_f15-10 ranked #1 in HPO leaderboard.",
            impact="Smaller models generalize better than h128 on this corpus.",
            follow_up_action="Prioritize h32/h64 in R-GCN HPO.",
            experiment_id="graphsage_eval",
        ),
        Finding(
            finding_id="F008",
            date="2026-06-11",
            phase="Phase 5 — R-GCN",
            observation="R-GCN eval benchmark AUPRC 0.6542; does not beat LR.",
            evidence="artifacts/published/rgcn/rgcn_benchmark.json; docs/rgcn_diagnosis.md; Gate G5 failed.",
            impact="Gate G5 failed; relation-specific convolutions alone insufficient.",
            follow_up_action="Fusion model or richer edge features.",
            experiment_id="rgcn_pub_v2",
        ),
    ]

    seed_decisions = [
        Decision(
            decision_id="D001",
            date="2026-06-06",
            decision="Adopt schema_v1.1 as primary heterogeneous graph schema.",
            alternatives_considered="v1b (bene-physician edges), v2 (diagnosis nodes), v4 (physician filter).",
            reason="Matches literature PP threshold; validated reference counts; fold-safe construction.",
            expected_impact="Reproducible graph artifacts for all models.",
            experiment_id="graph_construct_v1_1",
        ),
        Decision(
            decision_id="D002",
            date="2026-06-06",
            decision="Provider-provider collaborates edges require >= 2 shared beneficiaries.",
            alternatives_considered="Threshold 1 (pp_t1), threshold 5 (pp_t5), no PP (v1.1-no_pp).",
            reason="Reduces noise from single co-occurrence; G2 reference count 75,604 edges.",
            expected_impact="Sparsifies PP graph; improves signal-to-noise for GNN message passing.",
            experiment_id="graph_construct_v1_1",
        ),
        Decision(
            decision_id="D003",
            date="2026-06-06",
            decision="Claims are not materialized as graph nodes.",
            alternatives_considered="Claim-level heterogeneous graph; provider-only projected graph.",
            reason="Provider-level prediction target; claim features aggregated to provider nodes.",
            expected_impact="Manageable graph size; aligns with CMS fraud detection unit.",
            experiment_id="preprocess_v1",
        ),
        Decision(
            decision_id="D004",
            date="2026-06-11",
            decision="Proceed to R-GCN after GraphSAGE Gate G4 failure.",
            alternatives_considered="HGT, GAT, deeper GraphSAGE, tabular-only ensemble.",
            reason="GraphSAGE homophily assumption likely limits performance; relations have distinct semantics.",
            expected_impact="Test whether relation-aware modeling beats LR AUPRC 0.6810.",
            experiment_id="graphsage_eval",
        ),
        Decision(
            decision_id="D005",
            date="2026-06-06",
            decision="Provider-disjoint stratified 5-fold CV (seed=42) for all evaluation.",
            alternatives_considered="Random split, temporal split, beneficiary-disjoint CV.",
            reason="Prevents provider leakage; stratification preserves fraud prevalence per fold.",
            expected_impact="Conservative generalization estimate; comparable across models.",
            experiment_id="preprocess_v1",
        ),
        Decision(
            decision_id="D006",
            date="2026-06-11",
            decision="Primary metric: AUPRC; secondary: ROC-AUC, F1, Precision, Recall, Recall@K.",
            alternatives_considered="ROC-AUC primary, F1 primary, provider-level AP only.",
            reason="Class imbalance (9.35% fraud); AUPRC sensitive to ranking quality.",
            expected_impact="Consistent model selection across phases and gates.",
            experiment_id="baselines_v1",
        ),
    ]

    for f in seed_findings:
        _upsert_finding(state, f)
    for d in seed_decisions:
        _upsert_decision(state, d)

    state.last_synced_utc = _utc_now()
    return state


def _scan_artifacts(
    state: ResearchState,
    *,
    published_dir: Path = PUBLISHED_DIR,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> None:
    """Refresh project status and paper assets from publication-grade artifact directories."""
    baselines_dir = published_dir / "baselines"
    gnn_dir = published_dir / "gnn"
    rgcn_dir = published_dir / "rgcn"
    fusion_dir = published_dir / "fusion"

    tables: list[str] = []
    figures: list[str] = []
    ablations: list[str] = []
    baselines: list[str] = []
    models: list[str] = []
    case_studies: list[str] = []

    for pattern, bucket in [
        ("**/*.csv", tables),
        ("**/*.json", tables),
    ]:
        for path in published_dir.glob(pattern):
            rel = path.as_posix()
            if "comparison" in path.name or "leaderboard" in path.name:
                if rel not in tables:
                    tables.append(rel)

    plot_dirs = [diagnostics_dir / "gnn" / "plots", diagnostics_dir / "rgcn" / "plots"]
    for plot_dir in plot_dirs:
        if plot_dir.is_dir():
            for png in sorted(plot_dir.glob("*.png")):
                figures.append(png.as_posix())

    for name in ["logistic_regression", "catboost", "random_forest", "rf_centrality"]:
        if (baselines_dir / f"{name}.json").is_file():
            baselines.append(name)
    if (gnn_dir / "graphsage_benchmark.json").is_file():
        models.append("graphsage")
    if (rgcn_dir / "rgcn_benchmark.json").is_file():
        models.append("rgcn")
    if (fusion_dir / "fusion_benchmark.json").is_file():
        models.append("fusion")

    abl_dir = diagnostics_dir / "rgcn" / "ablations"
    if abl_dir.is_dir():
        for p in abl_dir.glob("*.json"):
            ablations.append(p.stem)

    gnn_hpo = ARCHIVED_DIR / "gnn_hpo_search"
    if gnn_hpo.is_dir():
        ablations.extend([f"graphsage_hpo:{p.stem}" for p in gnn_hpo.glob("*.json")])

    # Best model selection
    best_model = "logistic_regression"
    best_auprc = LR_BASELINE_AUPRC
    last_gate = "G3"

    if (gnn_dir / "evaluation_summary.json").is_file():
        with (gnn_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            gs = json.load(handle)
        gs_auprc = float(gs.get("benchmark_auprc_mean", 0))
        if gs_auprc > best_auprc:
            best_auprc = gs_auprc
            best_model = f"graphsage ({gs.get('top_config', 'unknown')})"
        last_gate = "G4 (failed)" if not gs.get("beats_lr", False) else "G4"

    if (rgcn_dir / "evaluation_summary.json").is_file():
        with (rgcn_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            rg = json.load(handle)
        rg_auprc = float(rg.get("benchmark_auprc", 0))
        if rg_auprc > best_auprc:
            best_auprc = rg_auprc
            best_model = f"rgcn ({rg.get('best_config', 'unknown')})"
        g5_passed = rg.get("gate_g5_passed")
        if g5_passed is True:
            last_gate = "G5 (passed)"
        elif g5_passed is False or rg.get("beats_lr") is False:
            last_gate = "G5 (failed)"
        else:
            last_gate = "G5 (eval complete)"

    lr_path = baselines_dir / "logistic_regression.json"
    if lr_path.is_file():
        with lr_path.open(encoding="utf-8") as handle:
            lr = json.load(handle)
        lr_auprc = float(lr["summary"]["auprc"]["mean"])
        if lr_auprc >= best_auprc:
            best_auprc = lr_auprc
            best_model = "logistic_regression"

    if (rgcn_dir / "evaluation_summary.json").is_file():
        with (rgcn_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            rg = json.load(handle)
        rg_auprc = float(rg.get("benchmark_auprc", 0))
        _upsert_finding(
            state,
            Finding(
                finding_id="F008",
                date="2026-06-11",
                phase="Phase 5 — R-GCN",
                observation=(
                    f"R-GCN eval benchmark AUPRC {rg_auprc:.4f}; "
                    f"{'beats' if rg.get('beats_lr') else 'does not beat'} LR."
                ),
                evidence="artifacts/published/rgcn/rgcn_benchmark.json; docs/rgcn_diagnosis.md",
                impact="Gate G5 failed." if not rg.get("beats_lr") else "Gate G5 passed.",
                follow_up_action="Fusion model or richer edge features.",
                experiment_id="rgcn_pub_v2",
            ),
        )

    current_phase = "Phase 5 — R-GCN Benchmark"
    pending = (
        "R-GCN eval HPO (6 configs); relation ablations; fusion model; "
        "GNNExplainer case studies"
    )
    next_action = "Complete R-GCN benchmark and ablations"

    if (rgcn_dir / "evaluation_summary.json").is_file():
        with (rgcn_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            rg = json.load(handle)
        if rg.get("gate_g5_passed") is False or rg.get("beats_lr") is False:
            current_phase = "Phase 6 — Fusion / Explainability (planned)"
            pending = (
                "Fusion model (tabular + graph); GNNExplainer case studies; "
                "optional R-GCN full HPO if eval-mode results insufficient"
            )
            next_action = (
                "Implement fusion or richer edge features; document why GNNs underperform LR"
            )
        else:
            pending = (
                "R-GCN full HPO (18 configs); relation ablations; fusion model; "
                "GNNExplainer case studies"
            )
            next_action = (
                "Complete R-GCN full HPO and ablations; if G5 fails, implement fusion"
            )

    fusion_auprc_str = ""
    if (fusion_dir / "evaluation_summary.json").is_file():
        with (fusion_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            fu = json.load(handle)
        last_gate = "G6 (failed)" if not fu.get("beats_lr", True) else "G6 (passed)"
        current_phase = "Phase 6 — Fusion complete; Explainability (planned)"
        pending = "GNNExplainer case studies; optional richer edge features or ensemble tuning"
        next_action = (
            "Document negative GNN/fusion results for publication; "
            "pursue explainability case studies"
        )
        best_fusion_name = "fusion_stack_logistic"
        best_fusion_auprc = 0.0
        bench_path = fusion_dir / "fusion_benchmark.json"
        if bench_path.is_file():
            with bench_path.open(encoding="utf-8") as handle:
                bench = json.load(handle)
            summaries = bench.get("model_summaries", {})
            if best_fusion_name in summaries:
                best_fusion_auprc = float(summaries[best_fusion_name]["auprc"]["mean"])
            else:
                fusion_models = [
                    (name, float(m["auprc"]["mean"]))
                    for name, m in summaries.items()
                    if name.startswith("fusion_")
                ]
                if fusion_models:
                    best_fusion_name, best_fusion_auprc = max(fusion_models, key=lambda x: x[1])
        fusion_auprc_str = (
            f" G6 failed: LR remains best overall; "
            f"best fusion {best_fusion_name} AUPRC {best_fusion_auprc:.4f}."
        )

    state.project_status = {
        "updated_utc": _utc_now(),
        "current_phase": current_phase,
        "last_completed_gate": last_gate,
        "current_best_model": best_model,
        "current_best_auprc": f"{best_auprc:.4f}",
        "pending_experiments": pending,
        "known_risks": (
            "Tabular LR dominates GNNs; class imbalance drives high recall/low precision; "
            "4GB GPU limits model depth; val providers require inductive inference graphs"
        ),
        "next_action": next_action,
    }

    state.paper_assets = {
        "tables_generated": sorted(set(tables)),
        "figures_generated": sorted(set(figures)),
        "ablations_completed": sorted(set(ablations)),
        "baselines_completed": sorted(set(baselines)),
        "models_completed": sorted(set(models)),
        "case_studies_completed": case_studies,
    }

    rg_auprc_str = "pending"
    if (rgcn_dir / "evaluation_summary.json").is_file():
        with (rgcn_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
            rg_ev = json.load(handle)
        rg_auprc_str = f"{float(rg_ev.get('benchmark_auprc', 0)):.4f}"

    state.publication_story = {
        "research_problem": (
            "Detect fraudulent Medicare providers using heterogeneous relational data "
            "(providers, beneficiaries, physicians) with explainable graph-based models."
        ),
        "literature_gap": (
            "Prior CMS fraud work relies on tabular features or homogeneous graphs; "
            "relation-specific heterogeneous modeling under provider-disjoint CV is under-tested."
        ),
        "hypotheses": (
            "H1: Graph structure improves AUPRC beyond tabular LR. "
            "H2: Relation-aware R-GCN beats homophilic GraphSAGE. "
            "H3: PP collaborates edges carry fraud signal beyond treats/bills_with."
        ),
        "experimental_evidence": (
            f"G3: LR AUPRC {LR_BASELINE_AUPRC:.4f} beats CatBoost/RF. "
            f"G4 failed: GraphSAGE AUPRC {GRAPHSAGE_BENCHMARK_AUPRC:.4f}. "
            f"G5 failed: R-GCN AUPRC {rg_auprc_str}."
            f"{fusion_auprc_str} "
            "Mean aggregation and 2-layer depth preferred for GraphSAGE."
        ),
        "current_conclusions": (
            "Strong tabular baseline (LR) sets high bar. GraphSAGE homophilic message passing "
            "does not overcome feature dominance. R-GCN and hybrid fusion also fail to beat LR "
            "under provider-disjoint CV; explainability case studies remain the primary path to "
            "publication-grade contribution."
        ),
        "updated_utc": _utc_now(),
    }


def record_finding(
    *,
    finding_id: str,
    phase: str,
    observation: str,
    evidence: str,
    impact: str,
    follow_up_action: str,
    experiment_id: str = "",
    date: str | None = None,
) -> None:
    """Record or update a scientific finding and refresh docs."""
    state = _load_state()
    _upsert_finding(
        state,
        Finding(
            finding_id=finding_id,
            date=date or _utc_date(),
            phase=phase,
            observation=observation,
            evidence=evidence,
            impact=impact,
            follow_up_action=follow_up_action,
            experiment_id=experiment_id,
        ),
    )
    _scan_artifacts(state)
    state.last_synced_utc = _utc_now()
    _save_state(state)
    write_all_docs(state)


def record_decision(
    *,
    decision_id: str,
    decision: str,
    alternatives_considered: str,
    reason: str,
    expected_impact: str,
    experiment_id: str = "",
    date: str | None = None,
) -> None:
    """Record or update a design decision and refresh docs."""
    state = _load_state()
    _upsert_decision(
        state,
        Decision(
            decision_id=decision_id,
            date=date or _utc_date(),
            decision=decision,
            alternatives_considered=alternatives_considered,
            reason=reason,
            expected_impact=expected_impact,
            experiment_id=experiment_id,
        ),
    )
    _scan_artifacts(state)
    state.last_synced_utc = _utc_now()
    _save_state(state)
    write_all_docs(state)


def on_gate_result(
    gate: str,
    *,
    passed: bool,
    experiment_id: str,
    details: str,
) -> None:
    """Hook for gate pass/fail events."""
    state = _load_state()
    finding_id = f"F{len(state.findings) + 1:03d}"
    _upsert_finding(
        state,
        Finding(
            finding_id=finding_id,
            date=_utc_date(),
            phase=f"Gate {gate}",
            observation=f"Gate {gate} {'PASSED' if passed else 'FAILED'}.",
            evidence=details,
            impact=f"{'Unlocks' if passed else 'Blocks'} next phase.",
            follow_up_action="See project_status.md next_action.",
            experiment_id=experiment_id,
        ),
    )
    _scan_artifacts(state)
    state.last_synced_utc = _utc_now()
    _save_state(state)
    write_all_docs(state)


def on_benchmark_complete(
    *,
    experiment_id: str,
    phase: str,
    model_name: str,
    auprc_mean: float,
    auprc_std: float,
    beats_lr: bool | None = None,
) -> None:
    """Hook after benchmark runs."""
    state = _load_state()
    finding_id = f"F{len(state.findings) + 1:03d}"
    obs = f"{model_name} 5-fold AUPRC {auprc_mean:.4f}±{auprc_std:.4f}"
    if beats_lr is not None:
        obs += f"; {'beats' if beats_lr else 'does not beat'} LR baseline."
    _upsert_finding(
        state,
        Finding(
            finding_id=finding_id,
            date=_utc_date(),
            phase=phase,
            observation=obs,
            evidence=f"experiment_id={experiment_id}; artifacts/published/",
            impact="Updates best-model ranking and publication story.",
            follow_up_action="Sync research docs; compare significance tests.",
            experiment_id=experiment_id,
        ),
    )
    _scan_artifacts(state)
    state.last_synced_utc = _utc_now()
    _save_state(state)
    write_all_docs(state)


def sync_all(*, published_dir: Path = PUBLISHED_DIR) -> None:
    """Full sync from published artifacts and rewrite all documentation files."""
    state = _load_state()
    _scan_artifacts(state, published_dir=published_dir)
    state.last_synced_utc = _utc_now()
    _save_state(state)
    write_all_docs(state)
    logger.info("Research documentation synced at %s", state.last_synced_utc)


def write_all_docs(state: ResearchState) -> None:
    """Write all five research markdown files."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text(_render_findings(state), encoding="utf-8")
    DECISIONS_PATH.write_text(_render_decisions(state), encoding="utf-8")
    STATUS_PATH.write_text(_render_status(state), encoding="utf-8")
    ASSETS_PATH.write_text(_render_assets(state), encoding="utf-8")
    STORY_PATH.write_text(_render_story(state), encoding="utf-8")


def _render_findings(state: ResearchState) -> str:
    lines = [
        "# Research Findings",
        "",
        "Automated record of scientific findings. Updated after each benchmark, ablation, and gate event.",
        "",
        f"Last synced: {state.last_synced_utc or 'never'}",
        "",
        "---",
        "",
    ]
    for f in state.findings:
        lines.extend(
            [
                f"## {f.finding_id}",
                "",
                f"**Date:** {f.date}  ",
                f"**Phase:** {f.phase}  ",
                f"**Experiment ID:** {f.experiment_id or 'N/A'}",
                "",
                "**Observation:**",
                f"{f.observation}",
                "",
                "**Evidence:**",
                f"{f.evidence}",
                "",
                "**Impact:**",
                f"{f.impact}",
                "",
                "**Follow-up Action:**",
                f"{f.follow_up_action}",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def _render_decisions(state: ResearchState) -> str:
    lines = [
        "# Research Decisions",
        "",
        "Automated record of major design decisions.",
        "",
        f"Last synced: {state.last_synced_utc or 'never'}",
        "",
        "---",
        "",
    ]
    for d in state.decisions:
        lines.extend(
            [
                f"## {d.decision_id}",
                "",
                f"**Date:** {d.date}  ",
                f"**Experiment ID:** {d.experiment_id or 'N/A'}",
                "",
                "**Decision:**",
                f"{d.decision}",
                "",
                "**Alternatives Considered:**",
                f"{d.alternatives_considered}",
                "",
                "**Reason:**",
                f"{d.reason}",
                "",
                "**Expected Impact:**",
                f"{d.expected_impact}",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def _render_status(state: ResearchState) -> str:
    ps = state.project_status
    return "\n".join(
        [
            "# Project Status",
            "",
            f"Last updated: {ps.get('updated_utc', state.last_synced_utc or 'never')}",
            "",
            "**Current Phase:**",
            ps.get("current_phase", "Unknown"),
            "",
            "**Last Completed Gate:**",
            ps.get("last_completed_gate", "Unknown"),
            "",
            "**Current Best Model:**",
            ps.get("current_best_model", "Unknown"),
            "",
            "**Current Best AUPRC:**",
            ps.get("current_best_auprc", "Unknown"),
            "",
            "**Pending Experiments:**",
            ps.get("pending_experiments", "None"),
            "",
            "**Known Risks:**",
            ps.get("known_risks", "None"),
            "",
            "**Next Action:**",
            ps.get("next_action", "None"),
            "",
        ]
    )


def _render_assets(state: ResearchState) -> str:
    pa = state.paper_assets
    lines = [
        "# Paper Assets",
        "",
        "Automatically tracked tables, figures, and completed experiments for publication.",
        "",
        f"Last synced: {state.last_synced_utc or 'never'}",
        "",
    ]
    for section, key in [
        ("Tables Generated", "tables_generated"),
        ("Figures Generated", "figures_generated"),
        ("Ablations Completed", "ablations_completed"),
        ("Baselines Completed", "baselines_completed"),
        ("Models Completed", "models_completed"),
        ("Case Studies Completed", "case_studies_completed"),
    ]:
        items = pa.get(key, [])
        lines.append(f"## {section}")
        lines.append("")
        if items:
            lines.extend(f"- `{item}`" for item in items)
        else:
            lines.append("- (none yet)")
        lines.append("")
    return "\n".join(lines)


def _render_story(state: ResearchState) -> str:
    ps = state.publication_story
    lines = [
        "# Publication Story",
        "",
        "Living narrative for the CMS provider fraud detection paper.",
        "",
        f"Last updated: {ps.get('updated_utc', state.last_synced_utc or 'never')}",
        "",
    ]
    for title, key in [
        ("Research Problem", "research_problem"),
        ("Literature Gap", "literature_gap"),
        ("Hypotheses", "hypotheses"),
        ("Experimental Evidence", "experimental_evidence"),
        ("Current Conclusions", "current_conclusions"),
    ]:
        lines.extend(["## " + title, "", ps.get(key, "TBD."), "", ""])
    return "\n".join(lines)
