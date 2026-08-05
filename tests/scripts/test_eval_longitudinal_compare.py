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


def _artifact(tmp_path, name, *, decision_text, snapshot):
    import json

    payload = {
        "final_decision": {"decision": decision_text},
        "prediction_snapshot": snapshot,
        "run_summary": {},
        "red_flags": [],
    }
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


_PM_TEXT = (
    "PORTFOLIO MANAGER VERDICT: BUY\n"
    "### --- START PM_BLOCK ---\n"
    "VERDICT: BUY\nHEALTH_ADJ: 92\nGROWTH_ADJ: 67\nRISK_TALLY: -0.5\nZONE: LOW\n"
    "### --- END PM_BLOCK ---\n"
)


def test_structured_snapshot_takes_precedence(tmp_path):
    """Persisted structured fields outrank reparsed narrative prose."""
    from scripts.eval_longitudinal_compare import extract_row

    p = _artifact(
        tmp_path,
        "AAA_20260802_161755_analysis.json",
        decision_text=_PM_TEXT,
        snapshot={"risk_tally": -0.5, "health_adj": 92, "growth_adj": 67},
    )
    row = extract_row(p)
    assert row.risk_total == -0.5
    assert row.health_adj == 92


def test_negative_tally_recovered_when_snapshot_is_null(tmp_path):
    """Legacy artifacts lost the value; the canonical parser recovers it."""
    from scripts.eval_longitudinal_compare import extract_row

    p = _artifact(
        tmp_path,
        "AAA_20260802_161755_analysis.json",
        decision_text=_PM_TEXT,
        snapshot={"risk_tally": None, "health_adj": None, "growth_adj": None},
    )
    assert extract_row(p).risk_total == -0.5


def test_absent_or_malformed_snapshot_does_not_crash(tmp_path):
    """`snapshot` is rebound later in extract_row; a non-dict must not reach the picker."""
    from scripts.eval_longitudinal_compare import extract_row

    for snapshot in (None, [], "nope", {}):
        p = _artifact(
            tmp_path,
            "AAA_20260802_161755_analysis.json",
            decision_text=_PM_TEXT,
            snapshot=snapshot,
        )
        assert extract_row(p).risk_total == -0.5
