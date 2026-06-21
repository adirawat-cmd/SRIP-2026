#!/usr/bin/env python3
"""Regenerate all research documentation from artifacts and persisted state."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hgad_cms.tracking.research_docs import sync_all  # noqa: E402
from hgad_cms.tracking.logger import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync research documentation markdown files.")
    parser.add_argument(
        "--published-dir",
        type=Path,
        default=Path("artifacts/published"),
        help="Publication-grade artifact root (default: artifacts/published).",
    )
    args = parser.parse_args()
    setup_logging()
    sync_all(published_dir=args.published_dir)
    logging.getLogger(__name__).info(
        "Updated docs/research_findings.md, research_decisions.md, "
        "project_status.md, paper_assets.md, publication_story.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
