"""Tests for the agent_attribution block in saved analysis JSON.

The attribution layer (added Stage 1 of the logging-audit follow-up) maps each
PM-input artifact to its producing agent, validity, char count, and aggregated
token-usage row from the token tracker. Three invariants matter:

1. Every row in `_ARTIFACT_AGENT_MAP` appears in saved JSON.
2. Token-usage join: agent_attribution[...]["token_usage"] matches the
   corresponding token_usage.agents[...] row (or aggregates multiple rows).
3. Map drift: every graph seat's registry-owned callback name, plus every
   explicit `tracked_callbacks(...)` name, appears in the map or allowlist.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from src.persistence import _ARTIFACT_AGENT_MAP, save_results_to_file

# Token-agent names produced by tracked_callbacks(...) that are intentionally
# NOT mapped to a saved-JSON artifact. Empty since the deep-retry LLM stopped
# carrying its own "Retry Agent (Deep)" callback (July 2026): retry cost is now
# attributed to the originating agent via a per-call callback in analyst_nodes,
# so there is no unmapped pooled bucket. Kept as the seam for future additions.
_UNMAPPED_TOKEN_AGENTS: frozenset[str] = frozenset()


@pytest.fixture(autouse=True)
def _reset_token_tracker():
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    tracker.reset()
    yield
    tracker.reset()


def _baseline_result(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "market_report": "market text",
        "sentiment_report": "sentiment text",
        "news_report": "news text",
        "fundamentals_report": "fundamentals text",
        "apac_regional_report": "apac text",
        "raw_fundamentals_data": "junior raw json",
        "management_guidance_evidence": "guidance preflight text",
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
        "company_name": "Test Corp",
        "company_name_resolved": True,
        "entity_governance_card": {
            "ticker": "TEST",
            "canonical_name": "Test Corp",
            "entity_role": "STANDALONE",
            "confidence": "clean",
        },
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


def test_saved_token_usage_carries_cost_rollup_dimensions(tmp_path: Path) -> None:
    # The A3/A4 rollups must survive serialization into results/*.json (they
    # ride get_total_stats() wholesale). Record spend on two providers/tiers
    # plus an unpriced model, then round-trip through save_results_to_file.
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    tracker.record_usage(
        agent_name="Consultant",
        model_name="gpt-5.6-terra",
        prompt_tokens=10_000,
        completion_tokens=1_000,
        service_tier="flex",
    )
    tracker.record_usage(
        agent_name="Mystery",
        model_name="totally-unpriced-model-x",
        prompt_tokens=1_000,
        completion_tokens=100,
    )
    tu = _save(tmp_path, _baseline_result())["token_usage"]
    assert {"by_provider", "by_model", "by_tier", "unpriced_models"} <= set(tu)
    assert tu["by_provider"]["openai"]["cost_usd"] > 0
    assert "flex" in tu["by_tier"]
    assert "totally-unpriced-model-x" in tu["unpriced_models"]


def test_agent_attribution_row_shape(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    row = saved["agent_attribution"]["market_report"]
    assert row["agent"] == "market_analyst"
    assert row["artifact_field"] == "market_report"
    assert row["token_agents"] == []
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
    assert row["token_agents"] == []


def test_agent_attribution_risk_debate_absent_when_empty(tmp_path: Path) -> None:
    base = _baseline_result()
    base["risk_debate_state"] = {}
    saved = _save(tmp_path, base)
    row = saved["agent_attribution"]["risk_debate_state"]
    assert row["present"] is False
    assert row["char_count"] == 0


def test_agent_attribution_governance_card_counts_structured_dict(
    tmp_path: Path,
) -> None:
    saved = _save(tmp_path, _baseline_result())
    row = saved["agent_attribution"]["entity_governance_card"]
    assert row["agent"] == "financial_health_validator"
    assert row["present"] is True
    assert row["direct_pm_input"] is True
    assert row["token_usage"] is None
    assert row["char_count"] > 0


def test_saved_json_persists_runtime_identity(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    assert "company_name" not in saved
    assert "company_name_resolved" not in saved
    assert saved["entity_governance_card"]["canonical_name"] == "Test Corp"
    assert saved["metadata"]["company_name"] == "Test Corp"
    assert saved["metadata"]["company_name_resolved"] is True


def test_saved_json_uses_null_for_missing_governance_card(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result({"entity_governance_card": {}}))
    assert saved["entity_governance_card"] is None


def test_saved_json_persists_pm_source_artifacts(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    assert saved["source_artifacts"] == {
        "management_guidance_evidence": "guidance preflight text",
        "raw_fundamentals_data": "junior raw json",
        "foreign_language_report": "foreign text",
        "legal_report": "legal text",
        "value_trap_report": "value trap text",
    }


def test_saved_json_persists_auditor_budget_telemetry(tmp_path: Path) -> None:
    telemetry = {
        "policy": {"max_llm_calls": 4},
        "tool_calls": {"get_official_document": 2},
        "llm_calls": 2,
        "evidence_chars": 18_000,
        "evidence_truncated": True,
        "outcomes": ["EVIDENCE_CHAR_LIMIT"],
    }
    saved = _save(tmp_path, _baseline_result({"auditor_budget": telemetry}))
    assert saved["auditor_budget"] == telemetry


def test_saved_json_uses_null_for_missing_auditor_budget(tmp_path: Path) -> None:
    saved = _save(tmp_path, _baseline_result())
    assert saved["auditor_budget"] is None


def test_saved_json_bounds_large_source_artifacts(tmp_path: Path) -> None:
    saved = _save(
        tmp_path,
        _baseline_result({"raw_fundamentals_data": "x" * 50_100}),
    )
    raw = saved["source_artifacts"]["raw_fundamentals_data"]
    assert len(raw) < 50_050
    assert raw.endswith("\n[...truncated]")


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
    assert saved["agent_attribution"]["market_report"]["token_agents"] == [
        "Market Analyst"
    ]
    tracker.reset()


def test_agent_attribution_omits_unused_fallback_models(tmp_path: Path) -> None:
    """Initialized-but-unused Sol/direct-retry models are not contributors."""
    from src.token_tracker import get_tracker

    tracker = get_tracker()
    tracker.reset()
    tracker.record_usage(
        agent_name="Global Forensic Auditor",
        model_name="gpt-5.6-terra",
        prompt_tokens=10,
        completion_tokens=5,
        elapsed_seconds=1.0,
    )
    tracker.record_usage(
        agent_name="APAC Regional Specialist",
        model_name="glm-5.2",
        prompt_tokens=10,
        completion_tokens=5,
        elapsed_seconds=1.0,
    )
    saved = _save(tmp_path, _baseline_result())

    assert saved["agent_attribution"]["auditor_report"]["token_agents"] == [
        "Global Forensic Auditor"
    ]
    assert saved["agent_attribution"]["apac_regional_report"]["token_agents"] == [
        "APAC Regional Specialist"
    ]
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
    """Collect registry-driven graph seats and explicit callback literals."""
    from src.llm_runtime.seats import SEATS, SeatId

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
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "seat_model"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "SeatId"
        ):
            tracked = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "tracked"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, bool)
                ),
                True,
            )
            if not tracked:
                continue
            seat_id = SeatId[node.args[0].attr]
            names.add(SEATS[seat_id].callback_name)
    return names


def test_agent_attribution_map_matches_live_token_callbacks() -> None:
    """Every graph callback name must be mapped or explicitly exempted."""
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
