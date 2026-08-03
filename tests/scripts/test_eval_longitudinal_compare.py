"""Tests for the longitudinal artifact comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import eval_longitudinal_compare as compare  # noqa: E402


def test_extract_row_surfaces_validity_and_data_quality(tmp_path: Path) -> None:
    artifact = tmp_path / "1681.HK_20260802_190854_analysis.json"
    artifact.write_text(
        json.dumps(
            {
                "metadata": {"ticker": "1681.HK"},
                "run_summary": {
                    "publishable": False,
                    "required_failures": ["fundamentals_report"],
                    "consultant_verdict": "CLEAN",
                    "auditor_status": "PARTIAL_DATA",
                },
                "final_decision": {"decision": ""},
                "structured_inputs": {
                    "raw_financial_metrics": {
                        "payload": {
                            "_quality": {"coverage_pct": 70.5},
                            "_quarterly_diagnostics": [
                                {"field": "revenueGrowth_TTM", "status": "unavailable"},
                                {
                                    "field": "earningsGrowth_TTM",
                                    "status": "unavailable",
                                },
                            ],
                            "_source_conflicts": {"earningsGrowth": {}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    row = compare.extract_row(artifact)

    assert row.publishable is False
    assert row.required_failures == ["fundamentals_report"]
    assert row.data_coverage_pct == 70.5
    assert row.growth_gap_count == 2
    assert row.source_conflict_count == 1

    rendered = compare.render_timeline_markdown("1681.HK", [row])
    assert "INCOMPLETE: fundamentals_report" in rendered
    assert "coverage 70.5%; growth gaps 2; conflicts 1" in rendered
