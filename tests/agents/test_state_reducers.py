"""Unit tests for AgentState reducer functions.

These reducers run on every LangGraph parallel-fan-out write, so their
correctness is critical for data integrity under concurrent Bull/Bear/Risk
agent execution.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents import (
    MESSAGE_TAIL_LIMIT,
    InvestDebateState,
    RiskDebateState,
    merge_and_cap_messages,
    merge_dicts,
    merge_flag_lists,
    merge_invest_debate_state,
    merge_risk_state,
)

# ─── merge_dicts ─────────────────────────────────────────────────────────────


class TestMergeDicts:
    def test_both_none_returns_empty(self):
        assert merge_dicts(None, None) == {}

    def test_x_none_returns_y(self):
        assert merge_dicts(None, {"a": 1}) == {"a": 1}

    def test_y_none_returns_x(self):
        assert merge_dicts({"a": 1}, None) == {"a": 1}

    def test_disjoint_keys_both_preserved(self):
        """Parallel agents writing to distinct keys → both survive."""
        x = {"market_analyst": {"get_news"}}
        y = {"news_analyst": {"get_macro"}}
        merged = merge_dicts(x, y)
        assert "market_analyst" in merged
        assert "news_analyst" in merged

    def test_overlapping_key_y_wins(self):
        """Same key written by two agents → last update (y) wins."""
        merged = merge_dicts({"a": 1}, {"a": 2})
        assert merged["a"] == 2

    def test_original_dicts_not_mutated(self):
        x = {"a": 1}
        y = {"b": 2}
        merge_dicts(x, y)
        assert "b" not in x
        assert "a" not in y


class TestMergeFlagLists:
    def test_distinct_parallel_red_flags_are_preserved(self):
        x = [{"type": "GEOPOLITICAL_RISK", "severity": "high"}]
        y = [{"type": "CUSTOMER_CONCENTRATION", "severity": "medium"}]

        assert merge_flag_lists(x, y) == [*x, *y]

    def test_duplicate_red_flags_are_deduped(self):
        flag = {"type": "PFIC_RISK", "severity": "medium"}

        assert merge_flag_lists([flag], [dict(flag)]) == [flag]

    def test_none_inputs_return_empty_list(self):
        assert merge_flag_lists(None, None) == []


# ─── merge_invest_debate_state ───────────────────────────────────────────────


def _debate(
    bull_round1="",
    bear_round1="",
    bull_round2="",
    bear_round2="",
    current_round=1,
    bull_history="",
    bear_history="",
    history="",
    current_response="",
    judge_decision="",
    count=0,
) -> InvestDebateState:
    return InvestDebateState(
        bull_round1=bull_round1,
        bear_round1=bear_round1,
        bull_round2=bull_round2,
        bear_round2=bear_round2,
        current_round=current_round,
        bull_history=bull_history,
        bear_history=bear_history,
        history=history,
        current_response=current_response,
        judge_decision=judge_decision,
        count=count,
    )


class TestMergeInvestDebateState:
    def test_x_none_returns_y(self):
        y = _debate(bull_round1="Bull arg")
        result = merge_invest_debate_state(None, y)
        assert result["bull_round1"] == "Bull arg"

    def test_y_none_returns_x(self):
        x = _debate(bear_round1="Bear arg")
        result = merge_invest_debate_state(x, None)
        assert result["bear_round1"] == "Bear arg"

    def test_both_none_returns_empty_default(self):
        result = merge_invest_debate_state(None, None)
        assert result["bull_round1"] == ""
        assert result["current_round"] == 1

    def test_parallel_bull_bear_round1_both_preserved(self):
        """Core invariant: Bull writes bull_round1, Bear writes bear_round1.

        When LangGraph merges the two concurrent updates, both fields must
        survive — neither should overwrite the other.
        """
        # State after Bull finishes: bull_round1 set, bear_round1 still empty
        x = _debate(bull_round1="Bullish thesis: strong FCF")
        # Bear's update: bear_round1 set, bull_round1 empty (Bear doesn't touch it)
        y = _debate(bear_round1="Bearish: margin compression risk")
        result = merge_invest_debate_state(x, y)
        assert result["bull_round1"] == "Bullish thesis: strong FCF"
        assert result["bear_round1"] == "Bearish: margin compression risk"

    def test_empty_y_does_not_overwrite_nonempty_x(self):
        """If y's update has empty strings, existing x values are preserved.

        This prevents a late-arriving empty-field update from erasing content.
        """
        x = _debate(history="Round 1 complete", bull_history="Bull made case")
        y = _debate()  # all empty strings
        result = merge_invest_debate_state(x, y)
        assert result["history"] == "Round 1 complete"
        assert result["bull_history"] == "Bull made case"

    def test_nonempty_y_overwrites_same_nonempty_x(self):
        """When both have content, y (newer update) wins for string fields."""
        x = _debate(bull_round1="Old bull arg")
        y = _debate(bull_round1="Revised bull arg")
        result = merge_invest_debate_state(x, y)
        assert result["bull_round1"] == "Revised bull arg"

    def test_numeric_current_round_last_write_wins(self):
        """Numeric fields use last-write-wins, not empty preference."""
        x = _debate(current_round=1, count=2)
        y = _debate(current_round=2, count=3)
        result = merge_invest_debate_state(x, y)
        assert result["current_round"] == 2
        assert result["count"] == 3


# ─── merge_risk_state ────────────────────────────────────────────────────────


def _risk(risky="", safe="", neutral="", latest_speaker="") -> RiskDebateState:
    return RiskDebateState(
        latest_speaker=latest_speaker,
        current_risky_response=risky,
        current_safe_response=safe,
        current_neutral_response=neutral,
    )


class TestMergeRiskState:
    def test_x_none_returns_y(self):
        y = _risk(risky="High risk take")
        result = merge_risk_state(None, y)
        assert result["current_risky_response"] == "High risk take"

    def test_y_none_returns_x(self):
        x = _risk(safe="Conservative take")
        result = merge_risk_state(x, None)
        assert result["current_safe_response"] == "Conservative take"

    def test_three_parallel_agents_all_preserved(self):
        """Risky, Safe, Neutral run in parallel; each writes its own field.

        LangGraph calls the reducer with (accumulated_state, agent_partial_update).
        Agents return ONLY the keys they changed — the reducer must preserve the
        keys it doesn't see in y. Using {**x, **y}: partial y merges correctly
        because absent keys from y are kept from x.
        """
        # Each agent returns only its own key (partial dict, not full TypedDict)
        base: RiskDebateState = _risk()
        after_risky = merge_risk_state(
            base,
            {"current_risky_response": "Aggressive buy", "latest_speaker": "risky"},
        )
        after_safe = merge_risk_state(
            after_risky,
            {"current_safe_response": "Wait for pullback", "latest_speaker": "safe"},
        )
        after_neutral = merge_risk_state(
            after_safe,
            {"current_neutral_response": "Balanced view", "latest_speaker": "neutral"},
        )

        assert after_neutral["current_risky_response"] == "Aggressive buy"
        assert after_neutral["current_safe_response"] == "Wait for pullback"
        assert after_neutral["current_neutral_response"] == "Balanced view"

    def test_latest_speaker_last_write_wins(self):
        """latest_speaker is shared; last-write-wins is acceptable."""
        x = _risk(risky="A", latest_speaker="risky_analyst")
        y = _risk(neutral="B", latest_speaker="neutral_analyst")
        result = merge_risk_state(x, y)
        assert result["latest_speaker"] == "neutral_analyst"


class TestMergeAndCapMessages:
    def test_preserves_first_human_and_recent_tail(self):
        initial = HumanMessage(content="analyze AAPL")
        generic_tail = [AIMessage(content=f"msg-{idx}") for idx in range(20)]

        result = merge_and_cap_messages([initial], generic_tail)

        assert result[0] is initial
        assert len(result) == 1 + MESSAGE_TAIL_LIMIT
        assert [msg.content for msg in result[1:]] == [
            f"msg-{idx}" for idx in range(20 - MESSAGE_TAIL_LIMIT, 20)
        ]

    def test_preserves_old_provenance_tool_messages(self):
        initial = HumanMessage(content="analyze AAPL")
        field_sources = ToolMessage(
            content='{"_field_sources": {"marketCap": "yfinance"}}',
            tool_call_id="field-sources",
        )
        source_conflicts = ToolMessage(
            content='{"_source_conflicts": {"pe": {"old": 10, "new": 12}}}',
            tool_call_id="source-conflicts",
        )
        generic_tail = [AIMessage(content=f"tail-{idx}") for idx in range(20)]

        result = merge_and_cap_messages(
            [initial, field_sources, source_conflicts], generic_tail
        )

        assert result[0] is initial
        assert field_sources in result
        assert source_conflicts in result
        assert result[-1].content == "tail-19"

    def test_handles_non_string_tool_content_without_crashing(self):
        initial = HumanMessage(content="analyze AAPL")
        tool_message = ToolMessage(
            content={"structured": "value"},
            tool_call_id="structured-tool",
        )

        result = merge_and_cap_messages([initial], [tool_message])

        assert result[0] is initial
        assert tool_message in result

    def test_duplicate_provenance_message_not_repeated(self):
        initial = HumanMessage(content="analyze AAPL")
        provenance = ToolMessage(
            content='{"_field_sources": {"marketCap": "eodhd"}}',
            tool_call_id="same-tool",
        )

        result = merge_and_cap_messages([initial, provenance], [provenance])

        assert result.count(provenance) == 1

    def test_handles_empty_and_no_human_message(self):
        assert merge_and_cap_messages(None, None) == []

        tool_message = ToolMessage(
            content='{"_field_sources": {"marketCap": "eodhd"}}',
            tool_call_id="sources",
        )
        ai_messages = [AIMessage(content=f"tail-{idx}") for idx in range(3)]

        result = merge_and_cap_messages([tool_message], ai_messages)

        assert result[0] is tool_message
        assert [msg.content for msg in result[1:]] == ["tail-0", "tail-1", "tail-2"]

    def test_malformed_json_marker_irrelevant_to_preservation(self):
        initial = HumanMessage(content="analyze AAPL")
        tool_message = ToolMessage(
            content='{"_field_sources": not-valid-json}',
            tool_call_id="bad-json",
        )
        generic_tail = [AIMessage(content=f"tail-{idx}") for idx in range(15)]

        result = merge_and_cap_messages([initial, tool_message], generic_tail)

        assert tool_message in result
        assert len(result) == 2 + MESSAGE_TAIL_LIMIT
