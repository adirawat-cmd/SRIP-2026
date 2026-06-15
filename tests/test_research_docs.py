"""Tests for automated research documentation maintenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hgad_cms.tracking.research_docs import (
    STATE_PATH,
    sync_all,
    write_all_docs,
    _load_state,
)


@pytest.fixture
def isolated_docs(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    state = tmp_path / "state"
    results = tmp_path / "artifacts" / "results"
    results.mkdir(parents=True)
    monkeypatch.setattr("hgad_cms.tracking.research_docs.DOCS_DIR", docs)
    monkeypatch.setattr("hgad_cms.tracking.research_docs.STATE_DIR", state)
    monkeypatch.setattr("hgad_cms.tracking.research_docs.STATE_PATH", state / "research_state.json")
    monkeypatch.setattr(
        "hgad_cms.tracking.research_docs.FINDINGS_PATH",
        docs / "research_findings.md",
    )
    monkeypatch.setattr(
        "hgad_cms.tracking.research_docs.DECISIONS_PATH",
        docs / "research_decisions.md",
    )
    monkeypatch.setattr(
        "hgad_cms.tracking.research_docs.STATUS_PATH",
        docs / "project_status.md",
    )
    monkeypatch.setattr(
        "hgad_cms.tracking.research_docs.ASSETS_PATH",
        docs / "paper_assets.md",
    )
    monkeypatch.setattr(
        "hgad_cms.tracking.research_docs.STORY_PATH",
        docs / "publication_story.md",
    )
    return tmp_path


def test_bootstrap_and_write_docs(isolated_docs: Path):
    sync_all(results_dir=isolated_docs / "artifacts" / "results")
    docs = isolated_docs / "docs"
    assert (docs / "research_findings.md").is_file()
    assert (docs / "project_status.md").is_file()
    state = _load_state()
    assert len(state.findings) >= 8
    assert len(state.decisions) >= 6
    text = (docs / "research_findings.md").read_text(encoding="utf-8")
    assert "F001" in text
    assert "Observation:" in text


def test_persist_state_roundtrip(isolated_docs: Path):
    sync_all(results_dir=isolated_docs / "artifacts" / "results")
    assert STATE_PATH.is_file()
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    assert "findings" in payload
    assert payload["findings"][0]["finding_id"] == "F001"
