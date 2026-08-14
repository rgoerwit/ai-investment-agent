"""Tests for the cost-rollup accuracy work (A3 + A4).

A3: an in-use model missing from ``MODEL_PRICING_PER_1M`` is surfaced in the
persisted stats (``unpriced_models``), not just a once-per-process log.
A4: per-agent ``by_model`` and top-level ``by_provider``/``by_model`` rollups are
derived from the authoritative per-call usages, so they reconcile to
``total_cost_usd`` and give a provider/model cost dimension the flat agent
rollup lacked.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.token_tracker import (
    DEFAULT_PRICING_PER_1M,
    TokenTracker,
    TokenTrackingCallback,
    _provider_for_model,
    canonical_display_name,
)


@pytest.fixture
def tracker() -> TokenTracker:
    t = TokenTracker()
    t.reset()
    yield t
    t.reset()


def _record(t: TokenTracker, agent: str, model: str, prompt=100_000, completion=50_000):
    t.record_usage(
        agent_name=agent,
        model_name=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


class TestByModelAndProvider:
    def test_per_agent_by_model_sums_to_agent_cost(self, tracker):
        # One agent that spanned two models (e.g. a quick->deep retry).
        _record(tracker, "Fundamentals Analyst", "gemini-3.1-flash-lite")
        _record(tracker, "Fundamentals Analyst", "gemini-3.1-pro-preview")
        stats = tracker.get_total_stats()
        agent = stats["agents"]["Fundamentals Analyst"]
        assert set(agent["by_model"]) == {
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
        }
        assert sum(m["cost_usd"] for m in agent["by_model"].values()) == pytest.approx(
            agent["cost_usd"]
        )
        assert sum(m["calls"] for m in agent["by_model"].values()) == agent["calls"]

    def test_by_provider_sums_to_total_cost(self, tracker):
        _record(tracker, "Consultant", "gpt-5.6-terra")
        _record(tracker, "PM", "gemini-3.1-pro-preview")
        _record(tracker, "APAC", "kimi-k3")
        stats = tracker.get_total_stats()
        assert set(stats["by_provider"]) == {"openai", "google", "moonshot"}
        assert sum(
            p["cost_usd"] for p in stats["by_provider"].values()
        ) == pytest.approx(stats["total_cost_usd"])
        # by_model is the same total, sliced the other way.
        assert sum(m["cost_usd"] for m in stats["by_model"].values()) == pytest.approx(
            stats["total_cost_usd"]
        )

    def test_by_tier_exposes_flex_vs_standard_fallback(self, tracker):
        # A flex-configured run where one call got flex and one fell back to
        # standard (full rate) — the rollup must show both so the operator can
        # see how much they paid full-rate when flex was unavailable.
        tracker.record_usage(
            agent_name="PM",
            model_name="gemini-3.1-pro-preview",
            prompt_tokens=100_000,
            completion_tokens=50_000,
            service_tier="flex",
        )
        tracker.record_usage(
            agent_name="PM",
            model_name="gemini-3.1-pro-preview",
            prompt_tokens=100_000,
            completion_tokens=50_000,
            service_tier="standard",
        )
        stats = tracker.get_total_stats()
        assert set(stats["by_tier"]) == {"flex", "standard"}
        # Same tokens, but the standard call costs 2x the flex call.
        assert stats["by_tier"]["standard"]["cost_usd"] == pytest.approx(
            stats["by_tier"]["flex"]["cost_usd"] * 2
        )
        assert sum(t["cost_usd"] for t in stats["by_tier"].values()) == pytest.approx(
            stats["total_cost_usd"]
        )

    def test_provider_mapping(self):
        assert _provider_for_model("gemini-3.6-flash") == "google"
        assert _provider_for_model("gpt-5.6-terra") == "openai"
        assert _provider_for_model("claude-opus-4-6") == "anthropic"
        assert _provider_for_model("deepseek-v4-pro") == "deepseek"
        assert _provider_for_model("kimi-k3") == "moonshot"
        assert _provider_for_model("glm-5.2") == "zhipu"
        assert _provider_for_model("moonshot/kimi-k3") == "moonshot"
        assert _provider_for_model("some-future-model-9") == "unknown"


class TestCanonicalAgentName:
    """A5: call_attempts names reconcile to the agents-rollup display namespace."""

    def test_spelling_map(self):
        assert canonical_display_name("External Consultant") == "Consultant"
        assert (
            canonical_display_name("Global Forensic Accountant")
            == "Global Forensic Auditor"
        )
        assert canonical_display_name("Bull Analyst") == "Bull Researcher"
        assert canonical_display_name("Bear Analyst") == "Bear Researcher"

    def test_suffix_stripping(self):
        assert canonical_display_name("Bull Analyst R1") == "Bull Researcher"
        assert canonical_display_name("Bear Analyst R2") == "Bear Researcher"
        assert (
            canonical_display_name("External Consultant_final_synthesis")
            == "Consultant"
        )
        assert (
            canonical_display_name("Fundamentals Analyst (RETRY-HIGH)")
            == "Fundamentals Analyst"
        )
        assert (
            canonical_display_name("Global Forensic Auditor Escalation")
            == "Global Forensic Auditor"
        )

    def test_display_names_pass_through(self):
        assert canonical_display_name("Portfolio Manager") == "Portfolio Manager"
        assert canonical_display_name("Market Analyst") == "Market Analyst"

    def test_call_attempt_records_canonical(self, tracker):
        tracker.record_call_attempt(
            agent_name="Bull Analyst R2",
            provider="google",
            model_name="gemini-3.6-flash",
            status="failure",
            attempt=1,
            elapsed_seconds=1.0,
            failure_kind="timeout",
        )
        stats = tracker.get_total_stats()
        attempt = stats["call_attempts"][0]
        assert attempt["agent_name"] == "Bull Analyst R2"  # raw preserved
        assert attempt["canonical_agent"] == "Bull Researcher"  # joinable
        # failed_by_agent aggregates on the canonical name.
        assert stats["call_diagnostics"]["failed_by_agent"] == {"Bull Researcher": 1}

    def test_escalation_and_direct_retry_context_strings_join_to_rollup(self, tracker):
        """The exact context= strings used at the real call sites (consultant_nodes
        .py's auditor escalation and apac_specialist_node.py's direct retry) must
        reconcile canonical_agent back to the agents-rollup key set via
        tracked_callbacks() in components.py — not just the isolated
        canonical_display_name() unit checks above."""
        tracker.record_call_attempt(
            agent_name="Global Forensic Auditor Escalation",
            provider="openai",
            model_name="gpt-5.6-terra",
            status="success",
            attempt=1,
            elapsed_seconds=1.0,
        )
        tracker.record_call_attempt(
            agent_name="APAC Regional Specialist Direct Retry",
            provider="google",
            model_name="gemini-3.6-flash",
            status="success",
            attempt=1,
            elapsed_seconds=1.0,
        )
        stats = tracker.get_total_stats()
        canonical_agents = {
            attempt["canonical_agent"] for attempt in stats["call_attempts"]
        }
        assert "Global Forensic Auditor" in canonical_agents
        assert "APAC Regional Specialist" in canonical_agents

    def test_explicit_canonical_overrides_derivation(self, tracker):
        tracker.record_call_attempt(
            agent_name="something odd",
            canonical_agent="Consultant",
            provider="openai",
            model_name="gpt-5.6-terra",
            status="success",
            attempt=1,
            elapsed_seconds=1.0,
        )
        assert tracker.get_total_stats()["call_attempts"][0]["canonical_agent"] == (
            "Consultant"
        )


class TestRetryCostAttribution:
    """A6: deep-model retry cost lands on the originating agent, not a pool."""

    def test_per_call_callback_routes_to_originating_agent(self, tracker):
        # Mirrors the analyst retry path: retry_llm carries no bound callback; a
        # per-call TokenTrackingCallback labeled with the (canonicalized)
        # originating agent is attached at invoke time.
        cb = TokenTrackingCallback(
            canonical_display_name("Fundamentals Analyst"), tracker=tracker
        )
        message = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10_000,
                "output_tokens": 500,
                "total_tokens": 10_500,
            },
        )
        cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        stats = tracker.get_total_stats()
        assert "Fundamentals Analyst" in stats["agents"]
        assert "Retry Agent (Deep)" not in stats["agents"]


class TestUnpricedModelsSurfaced:
    def test_priced_run_reports_empty(self, tracker):
        _record(tracker, "PM", "gemini-3.1-pro-preview")
        assert tracker.get_total_stats()["unpriced_models"] == []

    def test_unpriced_model_is_listed(self, tracker):
        _record(tracker, "PM", "gemini-3.1-pro-preview")
        _record(tracker, "Mystery", "totally-unknown-model-7")
        stats = tracker.get_total_stats()
        assert stats["unpriced_models"] == ["totally-unknown-model-7"]
        # Known usage is still costed; the unknown row itself contributes zero.
        assert stats["total_cost_usd"] > 0

    def test_default_priced_model_flagged(self):
        from src.token_tracker import _is_model_priced

        assert _is_model_priced("gemini-3.1-pro-preview") is True
        assert _is_model_priced("totally-unknown-model-7") is False
        # Identity of the fallback object is what the flag relies on.
        from src.token_tracker import _lookup_model_pricing

        assert (
            _lookup_model_pricing("totally-unknown-model-7") is DEFAULT_PRICING_PER_1M
        )

    def test_resolved_identity_drives_vendor_seat_and_group_rollups(self, tracker):
        tracker.record_usage(
            "Consultant",
            "custom-compatible-model",
            100,
            20,
            seat_id="consultant",
            binding_group="review",
            vendor_id="moonshot",
            model_lineage="kimi",
            adapter_kind="openai_compatible",
            endpoint_host="api.moonshot.cn",
        )
        stats = tracker.get_total_stats()
        assert stats["by_provider"]["moonshot"]["calls"] == 1
        assert stats["by_seat"]["consultant"]["calls"] == 1
        assert stats["by_binding_group"]["review"]["calls"] == 1
        assert stats["binding_usage"][0]["endpoint_host"] == "api.moonshot.cn"
        assert stats["binding_usage"][0]["cost_usd"] == 0.0
