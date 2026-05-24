"""Tests for the agent_attribution block in saved analysis JSON.

The attribution layer (added Stage 1 of the logging-audit follow-up) maps each
PM-input artifact to its producing agent, validity, char count, and aggregated
token-usage row from the token tracker. Three invariants matter:

1. Every row in `_ARTIFACT_AGENT_MAP` appears in saved JSON.
2. Token-usage join: agent_attribution[...]["token_usage"] matches the
   corresponding token_usage.agents[...] row (or aggregates multiple rows).
3. Map drift: every `tracked_callbacks(...)` display name in
   `src/graph/components.py` either appears in the map or in a small allowlist.
   The AST scan guards against silent renames.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from src.persistence import _ARTIFACT_AGENT_MAP, save_results_to_file

# Token-agent names produced by tracked_callbacks(...) that are intentionally
# NOT mapped to a saved-JSON artifact (e.g., the retry path's secondary LLM,
# which inherits whichever agent it was retrying for).
_UNMAPPED_TOKEN_AGENTS: frozenset[str] = frozenset(
    {
        "Retry Agent (Deep)",
    }
)


def _baseline_result(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "market_report": "market text",
        "sentiment_report": "sentiment text",
        "news_report": "news text",
        "fundamentals_report": "fundamentals text",
        "apac_regional_report": "apac text",
        "raw_fundamentals_data": "junior raw json",
        "foreign_language_report": "foreign text",
        "legal_report": "legal text",
        "value_trap_report": "value trap text",
        "investment_plan": "plan",
        "trader_investment_plan": "trader plan",
        "valuation_params": "valuation params",
        "consultant_review": "consultant text",
        "auditor_report": "auditor text",
        "final_trade_decision": "VERDICT: BUY",
        "investment_debate_state": {
            "bull_history": "bull",
            "bear_history": "bear",
            "count": 1,
        },
        "risk_debate_state": {
            "current_risky_response": "risky view",
            "current_safe_response": "safe view",
            "current_neutral_response": "neutral view",
        },
        "company_of_interest": "TEST",
    }
    if extra:
        result.update(extra)
    return result


def _save(tmp_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    filepath = save_results_to_file(result=result, ticker="TEST", results_dir=tmp_path)
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def test_agent_attribution_block_present_with_all_fields(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    attribution = saved["agent_attribution"]
    expected_fields = {field for field, _, _ in _ARTIFACT_AGENT_MAP}
    assert set(attribution.keys()) == expected_fields


def test_agent_attribution_row_shape(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    row = saved["agent_attribution"]["market_report"]
    assert row["agent"] == "market_analyst"
    assert row["artifact_field"] == "market_report"
    assert row["token_agents"] == ["Market Analyst"]
    assert row["present"] is True
    assert row["char_count"] == len("market text")
    assert row["direct_pm_input"] is True


def test_agent_attribution_uses_canonical_sentiment_slug(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    row = saved["agent_attribution"]["sentiment_report"]
    assert row["agent"] == "sentiment_analyst"


def test_agent_attribution_marks_missing_optional_as_absent(tmp_path: Path) -> None:
    base = _baseline_result()
    # Strip optional artifacts that would normally be empty when a run skips
    # the consultant gate.
    for field in ("auditor_report", "consultant_review", "valuation_params"):
        base.pop(field, None)
    saved = _save(tmp_path, base)
    for field in ("auditor_report", "consultant_review", "valuation_params"):
        row = saved["agent_attribution"][field]
        assert row["present"] is False
        assert row["char_count"] == 0
        assert row["token_usage"] is None


def test_agent_attribution_risk_debate_aggregates_three_siblings(
    tmp_path: Path,
) -> None:
    saved = _save(tmp_path, _baseline_result())
    row = saved["agent_attribution"]["risk_debate_state"]
    expected = len("risky view") + 1 + len("safe view") + 1 + len("neutral view")
    assert row["char_count"] == expected
    assert row["present"] is True
    assert row["token_agents"] == [
        "Risky Analyst",
        "Safe Analyst",
        "Neutral Analyst",
    ]


def test_agent_attribution_risk_debate_absent_when_empty(tmp_path: Path) -> None:
    base = _baseline_result()
    base["risk_debate_state"] = {}
    saved = _save(tmp_path, base)
    row = saved["agent_attribution"]["risk_debate_state"]
    assert row["present"] is False
    assert row["char_count"] == 0


def test_agent_attribution_preserves_legacy_reports_map(tmp_path: Path) -> None:
    """Stage 1 promise: reports stays as raw strings, not wrapped."""
    saved = _save(tmp_path, _baseline_result())
    assert isinstance(saved["reports"], dict)
    assert saved["reports"]["market_report"] == "market text"


def test_agent_attribution_token_usage_joins_to_token_tracker_rows(
    tmp_path: Path,
) -> None:
    """When the token tracker has a row for an agent, attribution copies it."""
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    tracker.reset()
    tracker.record_usage(
        agent_name="Market Analyst",
        model_name="gemini-3-flash",
        prompt_tokens=100,
        completion_tokens=50,
        elapsed_seconds=1.5,
    )
    saved = _save(tmp_path, _baseline_result())
    market = saved["agent_attribution"]["market_report"]["token_usage"]
    assert market is not None
    assert market["calls"] == 1
    assert market["prompt_tokens"] == 100
    assert market["completion_tokens"] == 50
    assert market["total_tokens"] == 150
    assert market["contributors"] == ["Market Analyst"]
    tracker.reset()


def test_agent_attribution_aggregates_multi_agent_token_rows(tmp_path: Path) -> None:
    """Risk debate sums three risk-analyst rows; investment_plan sums RM + bull + bear."""
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    tracker.reset()
    for name in ("Risky Analyst", "Safe Analyst", "Neutral Analyst"):
        tracker.record_usage(
            agent_name=name,
            model_name="gemini-3-flash",
            prompt_tokens=10,
            completion_tokens=5,
            elapsed_seconds=1.0,
        )
    saved = _save(tmp_path, _baseline_result())
    risk = saved["agent_attribution"]["risk_debate_state"]["token_usage"]
    assert risk["calls"] == 3
    assert risk["prompt_tokens"] == 30
    assert risk["completion_tokens"] == 15
    assert risk["total_tokens"] == 45
    assert risk["contributors"] == [
        "Risky Analyst",
        "Safe Analyst",
        "Neutral Analyst",
    ]
    tracker.reset()


def _extract_tracked_callbacks_names() -> set[str]:
    """AST scan: collect every string literal passed to tracked_callbacks(...)."""
    source = Path("src/graph/components.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tracked_callbacks"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def test_agent_attribution_map_matches_live_token_callbacks() -> None:
    """Drift guard: every name in _ARTIFACT_AGENT_MAP must exist as a
    tracked_callbacks(...) literal, and every literal must be either in the
    map or the unmapped allowlist."""
    live_names = _extract_tracked_callbacks_names()
    assert live_names, "AST scan found no tracked_callbacks() call sites"

    mapped_names: set[str] = set()
    for _, _, token_agents in _ARTIFACT_AGENT_MAP:
        mapped_names.update(token_agents)

    missing_from_live = mapped_names - live_names
    assert not missing_from_live, (
        f"_ARTIFACT_AGENT_MAP names not found in src/graph/components.py "
        f"tracked_callbacks(...) calls: {sorted(missing_from_live)}"
    )

    extra_in_live = live_names - mapped_names - _UNMAPPED_TOKEN_AGENTS
    assert not extra_in_live, (
        f"tracked_callbacks(...) names missing from _ARTIFACT_AGENT_MAP "
        f"(add to map or to _UNMAPPED_TOKEN_AGENTS allowlist): "
        f"{sorted(extra_in_live)}"
    )


def test_direct_pm_inputs_match_between_persistence_and_decision_nodes() -> None:
    """Stage 1 + Stage 2 must agree on which artifacts are direct PM inputs."""
    from src.agents.pm_inputs import DIRECT_PM_INPUT_FIELDS, DIRECT_PM_INPUTS

    # decision_nodes lists the 12 non-risk-debate inputs; persistence
    # adds risk_debate_state because it's a synthetic aggregate field.
    assert set(DIRECT_PM_INPUT_FIELDS) | {"risk_debate_state"} == DIRECT_PM_INPUTS
