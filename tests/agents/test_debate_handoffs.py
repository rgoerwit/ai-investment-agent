from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.debate_handoffs import (
    STRUCTURED_RATIONALE_END,
    STRUCTURED_RATIONALE_START,
    DebateReasoningPolicy,
    extract_native_reasoning,
    paired_handoff_telemetry,
    parse_structured_rationale_candidate,
    render_balanced_handoffs,
    render_opponent_handoff,
    split_structured_rationale,
    structured_rationale_addendum,
    structured_rationale_repair_prompt,
)
from src.agents.state import merge_invest_debate_state


def _capsule() -> str:
    return """POSITION: Bullish with valuation discipline.
DECISIVE_PREMISES:
- Margin durability.
EVIDENCE_REFERENCES:
- Fundamentals report: gross margin.
INFERENCE_BRIDGE:
- Stable margin supports cash conversion.
STRONGEST_DISCONFIRMING_EVIDENCE:
- Demand could slow.
MATERIAL_UNCERTAINTIES:
- Next-quarter volume.
WHAT_WOULD_CHANGE_MY_VIEW:
- Two quarters of margin contraction."""


@pytest.mark.parametrize(
    ("enabled", "max_rounds", "round_num", "emits", "consumes"),
    [
        (False, 1, 1, False, False),
        (False, 2, 1, False, False),
        (False, 2, 2, False, False),
        (True, 1, 1, False, False),
        (True, 2, 1, True, False),
        (True, 2, 2, False, True),
    ],
)
def test_policy_round_truth_table(
    enabled: bool,
    max_rounds: int,
    round_num: int,
    emits: bool,
    consumes: bool,
) -> None:
    policy = DebateReasoningPolicy(enabled=enabled, max_rounds=max_rounds)
    assert policy.emits_in(round_num) is emits
    assert policy.consumes_in(round_num) is consumes


def test_policy_rejects_out_of_range_and_unsupported_rounds() -> None:
    with pytest.raises(ValueError, match="one or two"):
        DebateReasoningPolicy(enabled=True, max_rounds=3)
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    with pytest.raises(ValueError, match="outside"):
        policy.emits_in(0)
    with pytest.raises(ValueError, match="outside"):
        policy.consumes_in(3)


def test_budget_bonus_is_scoped_to_active_debate_seats() -> None:
    active = DebateReasoningPolicy(enabled=True, max_rounds=2)
    inactive = DebateReasoningPolicy(enabled=True, max_rounds=1)
    assert active.output_bonus("Bull Researcher") == 1_024
    assert active.output_bonus("Bear Researcher") == 1_024
    assert active.output_bonus("Research Manager") == 1_024
    assert active.output_bonus("Portfolio Manager") == 0
    assert inactive.output_bonus("Bull Researcher") == 0
    assert active.structured_repair_output_tokens == 1_024


def test_policy_rejects_nonpositive_repair_budget() -> None:
    with pytest.raises(ValueError, match="repair output tokens must be positive"):
        DebateReasoningPolicy(
            enabled=True,
            max_rounds=2,
            structured_repair_output_tokens=0,
        )


def test_capsule_is_requested_only_in_non_final_round() -> None:
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    addendum = structured_rationale_addendum(policy, 1)
    assert STRUCTURED_RATIONALE_START in addendum
    # Canonical argument first, capsule appended: the argument is the
    # deliverable, so a response cut at the output cap must lose only the
    # adjunct. The parser reads either order, so this is a prompt contract.
    assert "canonical argument first" in addendum
    assert addendum.index("canonical argument first") < addendum.index(
        STRUCTURED_RATIONALE_START
    )
    assert "incomplete if either marker is absent" in addendum
    assert structured_rationale_addendum(policy, 2) == ""


def test_split_capsule_keeps_canonical_argument_and_valid_rationale() -> None:
    policy = DebateReasoningPolicy(
        enabled=True,
        max_rounds=2,
        structured_char_cap=2_000,
    )
    raw = (
        "Canonical argument.\n\n"
        f"{STRUCTURED_RATIONALE_START}\n{_capsule()}\n{STRUCTURED_RATIONALE_END}"
    )
    canonical, rationale = split_structured_rationale(raw, policy=policy, round_num=1)
    assert canonical == "Canonical argument."
    assert rationale == _capsule()
    assert STRUCTURED_RATIONALE_START not in canonical


def test_split_prefix_capsule_keeps_following_canonical_argument() -> None:
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    raw = (
        f"{STRUCTURED_RATIONALE_START}\n{_capsule()}\n{STRUCTURED_RATIONALE_END}"
        "\n\nCanonical argument."
    )

    canonical, rationale = split_structured_rationale(raw, policy=policy, round_num=1)

    assert canonical == "Canonical argument."
    assert rationale == _capsule()
    assert STRUCTURED_RATIONALE_START not in canonical


def test_repair_prompt_is_extraction_only_and_unmarked_candidate_is_accepted() -> None:
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)

    prompt = structured_rationale_repair_prompt(policy)

    assert "structure-only extraction" in prompt
    assert "Do not add facts" in prompt
    assert "NOT STATED" in prompt
    assert parse_structured_rationale_candidate(_capsule(), policy=policy) == _capsule()


def test_repair_candidate_rejects_partial_markers_instead_of_leaking_them() -> None:
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    malformed = f"{STRUCTURED_RATIONALE_START}\n{_capsule()}"

    assert parse_structured_rationale_candidate(malformed, policy=policy) == ""


def test_repair_candidate_accepts_one_line_markdown_labels() -> None:
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    candidate = " ".join(
        (
            "**POSITION:** Bullish",
            "**DECISIVE_PREMISES:** Durable margins",
            "**EVIDENCE_REFERENCES:** Reported gross margin",
            "**INFERENCE_BRIDGE:** Margin supports cash conversion",
            "**STRONGEST_DISCONFIRMING_EVIDENCE:** Demand could slow",
            "**MATERIAL_UNCERTAINTIES:** Next-quarter volume",
            "**WHAT_WOULD_CHANGE_MY_VIEW:** Margin contraction",
        )
    )

    parsed = parse_structured_rationale_candidate(candidate, policy=policy)

    assert parsed
    assert len(parsed) <= policy.structured_char_cap


def test_oversized_complete_capsule_is_normalized_and_bounded() -> None:
    policy = DebateReasoningPolicy(
        enabled=True,
        max_rounds=2,
        structured_char_cap=len(_capsule()) - 1,
    )
    raw = (
        "Canonical argument.\n\n"
        f"{STRUCTURED_RATIONALE_START}\n{_capsule()}\n{STRUCTURED_RATIONALE_END}"
    )
    canonical, rationale = split_structured_rationale(raw, policy=policy, round_num=1)
    assert canonical == "Canonical argument."
    assert 0 < len(rationale) <= policy.structured_char_cap
    for field in (
        "POSITION",
        "DECISIVE_PREMISES",
        "EVIDENCE_REFERENCES",
        "INFERENCE_BRIDGE",
        "STRONGEST_DISCONFIRMING_EVIDENCE",
        "MATERIAL_UNCERTAINTIES",
        "WHAT_WOULD_CHANGE_MY_VIEW",
    ):
        assert f"{field}:" in rationale


def test_capsule_with_empty_required_field_is_stripped_and_withheld() -> None:
    empty_position = _capsule().replace(
        "POSITION: Bullish with valuation discipline.", "POSITION:"
    )
    raw = (
        "Canonical argument.\n\n"
        f"{STRUCTURED_RATIONALE_START}\n{empty_position}\n{STRUCTURED_RATIONALE_END}"
    )
    canonical, rationale = split_structured_rationale(
        raw,
        policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
        round_num=1,
    )
    assert canonical == "Canonical argument."
    assert rationale == ""


@pytest.mark.parametrize(
    "raw",
    [
        f"Canonical.\n{STRUCTURED_RATIONALE_START}\nPOSITION: Bullish",
        (
            f"Canonical.\n{STRUCTURED_RATIONALE_START}\nPOSITION: Bullish\n"
            f"{STRUCTURED_RATIONALE_END}"
        ),
    ],
)
def test_malformed_capsule_is_removed_but_not_published(raw: str) -> None:
    canonical, rationale = split_structured_rationale(
        raw,
        policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
        round_num=1,
    )
    assert canonical == "Canonical."
    assert rationale == ""


def test_native_reasoning_uses_normalized_readable_blocks_only() -> None:
    response = AIMessage(
        content_blocks=[
            {"type": "reasoning", "reasoning": "Readable summary."},
            {"type": "text", "text": "Canonical answer."},
            {
                "type": "reasoning",
                "reasoning": "Second summary.",
                "extras": {"signature": "opaque-secret-signature"},
            },
        ]
    )
    extracted = extract_native_reasoning(response, char_cap=2_000)
    assert extracted == "Readable summary.\n\nSecond summary."
    assert "opaque-secret-signature" not in extracted


def test_native_reasoning_absent_for_opaque_or_malformed_blocks() -> None:
    response = AIMessage(
        content=[
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "reasoning", "extras": {"signature": "opaque"}},
            {"type": "text", "text": "Answer"},
        ]
    )
    assert extract_native_reasoning(response, char_cap=2_000) == ""


def test_pairing_withholds_each_one_sided_component() -> None:
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    telemetry = paired_handoff_telemetry(
        policy=policy,
        bull={"structured": "bull structured", "native": "bull native"},
        bear={"structured": "bear structured", "native": ""},
    )
    assert telemetry["structured_pair"] is True
    assert telemetry["native_pair"] is False
    rendered = render_balanced_handoffs(
        bull={"structured": "bull structured", "native": "bull native"},
        bear={"structured": "bear structured", "native": ""},
        telemetry=telemetry,
    )
    assert "bull structured" in rendered
    assert "bear structured" in rendered
    assert "bull native" not in rendered


def test_opponent_and_manager_renderers_preserve_role_balance() -> None:
    bull = {"structured": "bull structured", "native": "bull native"}
    bear = {"structured": "bear structured", "native": "bear native"}
    telemetry = paired_handoff_telemetry(
        policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
        bull=bull,
        bear=bear,
    )
    opponent = render_opponent_handoff(
        opponent_role="bear", opponent=bear, telemetry=telemetry
    )
    manager = render_balanced_handoffs(bull=bull, bear=bear, telemetry=telemetry)
    assert "bear structured" in opponent and "bull structured" not in opponent
    assert all(value in manager for value in (*bull.values(), *bear.values()))
    assert "UNTRUSTED" in manager and "NON-EVIDENTIARY" in manager


def test_parallel_state_merge_is_order_independent_for_role_private_handoffs() -> None:
    bull = {
        "bull_round1": "bull argument",
        "bull_round1_handoff": {"structured": "bull", "native": ""},
    }
    bear = {
        "bear_round1": "bear argument",
        "bear_round1_handoff": {"structured": "bear", "native": ""},
    }
    forward = merge_invest_debate_state(bull, bear)  # type: ignore[arg-type]
    reverse = merge_invest_debate_state(bear, bull)  # type: ignore[arg-type]
    for merged in (forward, reverse):
        assert merged["bull_round1_handoff"]["structured"] == "bull"
        assert merged["bear_round1_handoff"]["structured"] == "bear"


@pytest.mark.asyncio
async def test_researcher_strips_capsule_and_captures_native_summary() -> None:
    from src.agents.research_nodes import create_researcher_node

    llm = MagicMock()
    response = AIMessage(
        content_blocks=[
            {"type": "reasoning", "reasoning": "Native summary"},
            {
                "type": "text",
                "text": (
                    "Canonical thesis\n"
                    f"{STRUCTURED_RATIONALE_START}\n{_capsule()}\n"
                    f"{STRUCTURED_RATIONALE_END}"
                ),
            },
        ]
    )

    async def invoke(*args, **kwargs):
        return response

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {},
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            llm,
            None,
            "bull_researcher",
            round_num=1,
            handoff_policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
        )
        result = await node(state, {})

    debate = result["investment_debate_state"]
    assert debate["bull_round1"].endswith("Canonical thesis")
    assert STRUCTURED_RATIONALE_START not in debate["bull_round1"]
    assert debate["bull_round1_handoff"]["native"] == "Native summary"
    assert debate["bull_round1_handoff"]["structured"]


@pytest.mark.asyncio
async def test_bad_request_for_native_summary_retries_once_on_ordinary_model() -> None:
    from src.agents.research_nodes import create_researcher_node

    handoff_llm = MagicMock(model="gpt-test")
    ordinary_llm = MagicMock(model="gpt-test")
    calls: list[object] = []

    async def invoke(runnable, *args, **kwargs):
        calls.append(runnable)
        if runnable is handoff_llm:
            raise RuntimeError("Error code: 400 - invalid_request_error: summary")
        return AIMessage(content="Canonical fallback thesis")

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {},
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            handoff_llm,
            None,
            "bull_researcher",
            round_num=1,
            handoff_policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
            fallback_llm=ordinary_llm,
        )
        result = await node(state, {})

    assert calls == [handoff_llm, ordinary_llm, ordinary_llm]
    assert result["investment_debate_state"]["bull_round1"].endswith(
        "Canonical fallback thesis"
    )


@pytest.mark.asyncio
async def test_missing_inline_capsule_is_repaired_from_canonical_argument() -> None:
    from src.agents.research_nodes import create_researcher_node

    handoff_llm = MagicMock(model="gpt-test")
    ordinary_llm = MagicMock(model="gpt-test")
    repair_llm = MagicMock(model="gpt-fast-test")
    runnables: list[object] = []
    prompts: list[str] = []

    async def invoke(runnable, messages, *args, **kwargs):
        runnables.append(runnable)
        prompts.append(messages[0].content)
        if len(prompts) == 1:
            return AIMessage(
                content_blocks=[
                    {"type": "reasoning", "reasoning": "Native summary"},
                    {"type": "text", "text": "Canonical thesis without capsule"},
                ]
            )
        return AIMessage(content=_capsule())

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {},
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            handoff_llm,
            None,
            "bull_researcher",
            round_num=1,
            handoff_policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
            fallback_llm=ordinary_llm,
            structured_repair_llm=repair_llm,
        )
        result = await node(state, {})

    debate = result["investment_debate_state"]
    assert len(prompts) == 2
    assert runnables == [handoff_llm, repair_llm]
    assert "CANONICAL ROUND-1 ARGUMENT TO STRUCTURE" in prompts[1]
    assert debate["bull_round1"].endswith("Canonical thesis without capsule")
    assert debate["bull_round1_handoff"] == {
        "structured": _capsule(),
        "native": "Native summary",
    }


@pytest.mark.asyncio
async def test_structured_repair_failure_preserves_canonical_and_native_output() -> (
    None
):
    from src.agents.research_nodes import create_researcher_node

    calls = 0

    async def invoke(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content_blocks=[
                    {"type": "reasoning", "reasoning": "Native summary"},
                    {"type": "text", "text": "Canonical thesis"},
                ]
            )
        raise TimeoutError("repair timed out")

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {},
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            MagicMock(model="gpt-test"),
            None,
            "bull_researcher",
            round_num=1,
            handoff_policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
        )
        result = await node(state, {})

    debate = result["investment_debate_state"]
    assert calls == 2
    assert debate["bull_round1"].endswith("Canonical thesis")
    assert debate["bull_round1_handoff"] == {
        "structured": "",
        "native": "Native summary",
    }


@pytest.mark.asyncio
async def test_transient_failure_does_not_bypass_normal_retry_policy() -> None:
    from src.agents.research_nodes import create_researcher_node

    handoff_llm = MagicMock(model="gpt-test")
    ordinary_llm = MagicMock(model="gpt-test")
    calls: list[object] = []

    async def invoke(runnable, *args, **kwargs):
        calls.append(runnable)
        raise TimeoutError("provider timed out")

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {},
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
    ):
        node = create_researcher_node(
            handoff_llm,
            None,
            "bull_researcher",
            round_num=1,
            handoff_policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
            fallback_llm=ordinary_llm,
        )
        result = await node(state, {})

    assert calls == [handoff_llm]
    assert "SYSTEM ERROR" in result["investment_debate_state"]["bull_round1"]


@pytest.mark.asyncio
async def test_partial_reasoning_response_retries_with_canonical_prompt() -> None:
    from src.agents.research_nodes import create_researcher_node

    handoff_llm = MagicMock(model="gpt-test")
    ordinary_llm = MagicMock(model="gpt-test")
    prompts: list[str] = []

    async def invoke(runnable, messages, *args, **kwargs):
        prompts.append(messages[0].content)
        if runnable is handoff_llm:
            return AIMessage(
                content=(
                    "Partial thesis\n"
                    f"{STRUCTURED_RATIONALE_START}\n{_capsule()}\n"
                    f"{STRUCTURED_RATIONALE_END}"
                ),
                response_metadata={"finish_reason": "length"},
            )
        if len(prompts) == 3:
            return AIMessage(content=_capsule())
        return AIMessage(
            content="Complete canonical fallback",
            response_metadata={"finish_reason": "stop"},
        )

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {},
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            handoff_llm,
            None,
            "bull_researcher",
            round_num=1,
            handoff_policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
            fallback_llm=ordinary_llm,
        )
        result = await node(state, {})

    assert len(prompts) == 3
    assert STRUCTURED_RATIONALE_START in prompts[0]
    assert STRUCTURED_RATIONALE_START not in prompts[1]
    assert "CANONICAL ROUND-1 ARGUMENT TO STRUCTURE" in prompts[2]
    debate = result["investment_debate_state"]
    assert debate["bull_round1"].endswith("Complete canonical fallback")
    assert debate["bull_round1_handoff"] == {
        "structured": _capsule(),
        "native": "",
    }


@pytest.mark.asyncio
async def test_round2_receives_only_the_opponents_paired_handoff() -> None:
    from src.agents.research_nodes import create_researcher_node

    captured: dict[str, str] = {}

    async def invoke(runnable, messages, *args, **kwargs):
        captured["prompt"] = messages[0].content
        return AIMessage(content="Round 2 rebuttal")

    prompt = SimpleNamespace(
        system_message="Debate.", agent_name="Bull Researcher", version="test"
    )
    bull = {"structured": "BULL-STRUCTURED-ONLY", "native": "BULL-NATIVE-ONLY"}
    bear = {"structured": "BEAR-STRUCTURED-ONLY", "native": "BEAR-NATIVE-ONLY"}
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    state = {
        "company_of_interest": "TEST",
        "company_name": "Test Co",
        "company_name_resolved": True,
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "investment_debate_state": {
            "bull_round1": "Bull argument",
            "bear_round1": "Bear argument",
            "bull_round1_handoff": bull,
            "bear_round1_handoff": bear,
            "handoff_telemetry": paired_handoff_telemetry(
                policy=policy, bull=bull, bear=bear
            ),
        },
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_researcher_node(
            MagicMock(),
            None,
            "bull_researcher",
            round_num=2,
            handoff_policy=policy,
        )
        await node(state, {})

    assert "BEAR-STRUCTURED-ONLY" in captured["prompt"]
    assert "BEAR-NATIVE-ONLY" in captured["prompt"]
    assert "BULL-STRUCTURED-ONLY" not in captured["prompt"]
    assert "BULL-NATIVE-ONLY" not in captured["prompt"]
    assert STRUCTURED_RATIONALE_START not in captured["prompt"]


@pytest.mark.asyncio
async def test_research_manager_receives_the_balanced_pair() -> None:
    from src.agents.research_nodes import create_research_manager_node

    captured: dict[str, str] = {}

    async def invoke(runnable, messages, *args, **kwargs):
        captured["prompt"] = messages[0].content
        return AIMessage(content="Synthesis")

    prompt = SimpleNamespace(
        system_message="Synthesize.", agent_name="Research Manager", version="test"
    )
    bull = {"structured": "BULL-STRUCTURED-PAIR", "native": "BULL-NATIVE-PAIR"}
    bear = {"structured": "BEAR-STRUCTURED-PAIR", "native": "BEAR-NATIVE-PAIR"}
    policy = DebateReasoningPolicy(enabled=True, max_rounds=2)
    state = {
        "company_of_interest": "TEST",
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "value_trap_report": "V",
        "messages": [],
        "investment_debate_state": {
            "bull_history": "Bull debate",
            "bear_history": "Bear debate",
            "bull_round1_handoff": bull,
            "bear_round1_handoff": bear,
            "handoff_telemetry": paired_handoff_telemetry(
                policy=policy, bull=bull, bear=bear
            ),
        },
    }
    with (
        patch("src.prompts.get_prompt", return_value=prompt),
        patch(
            "src.agents.research_nodes.agent_runtime.invoke_with_rate_limit_handling",
            side_effect=invoke,
        ),
        patch(
            "src.retrospective.format_lessons_for_injection",
            new=AsyncMock(return_value=""),
        ),
        patch("src.retrospective.create_lessons_memory", return_value=None),
        patch("src.agents.research_nodes.log_output_diagnostics"),
        patch("src.agents.research_nodes.log_truncation_diagnostic"),
    ):
        node = create_research_manager_node(MagicMock(), None, handoff_policy=policy)
        await node(state, {})

    assert all(
        value in captured["prompt"] for value in (*bull.values(), *bear.values())
    )
    assert "BALANCED ROUND-1 REASONING ADJUNCTS" in captured["prompt"]
    assert "NON-EVIDENTIARY" in captured["prompt"]


def test_no_handoff_content_is_rendered_when_pair_is_unavailable() -> None:
    telemetry = paired_handoff_telemetry(
        policy=DebateReasoningPolicy(enabled=True, max_rounds=2),
        bull={"structured": "bull", "native": ""},
        bear={"structured": "", "native": "bear"},
    )
    assert not telemetry["published_rounds"]
    assert (
        render_balanced_handoffs(
            bull={"structured": "bull", "native": ""},
            bear={"structured": "", "native": "bear"},
            telemetry=telemetry,
        )
        == ""
    )


class TestNativeReasoningCrossesTheNormalizationLayer:
    """Guard the langchain-core translation the other tests deliberately skip.

    The `content_blocks=[...]` tests elsewhere in this file assert the correct
    consumer seam and should stay. But they bypass the provider-metadata
    dispatch that turns a vendor's raw block into the normalized `reasoning`
    shape — and that layer is exactly what made a 2026-08 review conclude,
    twice and wrongly, that this feature was dead on every provider. A fixture
    built without `response_metadata` normalizes to `non_standard`, a state no
    live response reaches. Mirrors the precedent in
    `tests/test_llms_flex.py::TestGeminiRequestInjection`: if a langchain
    upgrade breaks the passthrough, this fails loudly.
    """

    @staticmethod
    def _policy() -> DebateReasoningPolicy:
        return DebateReasoningPolicy(enabled=True, max_rounds=2)

    def test_google_thinking_block_normalizes_and_extracts(self):
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "The moat is narrowing because…"},
                {"type": "text", "text": "BULL CASE"},
            ],
            response_metadata={"model_provider": "google_genai"},
        )

        assert extract_native_reasoning(message, char_cap=2_000) == (
            "The moat is narrowing because…"
        )

    def test_openai_responses_summary_normalizes_and_extracts(self):
        message = AIMessage(
            content=[
                {
                    "type": "reasoning",
                    "id": "rs_abc",
                    "summary": [
                        {"type": "summary_text", "text": "Weighing the moat evidence."}
                    ],
                },
                {"type": "text", "text": "BULL CASE"},
            ],
            response_metadata={"model_provider": "openai"},
        )

        assert extract_native_reasoning(message, char_cap=2_000) == (
            "Weighing the moat evidence."
        )

    def test_a_fixture_without_provider_metadata_yields_nothing(self):
        """Pinned deliberately: this is the shape that produced a false finding.

        Do not "fix" this by teaching the extractor to read `non_standard`.
        No live response arrives without provider metadata, and widening the
        reader would make it accept blocks langchain declined to interpret.
        """
        message = AIMessage(
            content=[{"type": "thinking", "thinking": "unreachable in production"}]
        )

        assert extract_native_reasoning(message, char_cap=2_000) == ""


class TestNativeSummaryTruncationIsMarked:
    def test_summary_under_the_cap_is_returned_verbatim(self):
        message = AIMessage(
            content=[{"type": "reasoning", "reasoning": "short summary"}],
            response_metadata={"model_provider": "openai"},
        )

        assert extract_native_reasoning(message, char_cap=2_000) == "short summary"

    def test_summary_over_the_cap_is_marked_and_stays_within_the_cap(self):
        message = AIMessage(
            content=[{"type": "reasoning", "reasoning": "word " * 1_000}],
            response_metadata={"model_provider": "openai"},
        )

        summary = extract_native_reasoning(message, char_cap=100)

        assert len(summary) <= 100
        assert summary.endswith("…")

    def test_one_unbroken_token_longer_than_the_cap_still_returns_marked_text(self):
        message = AIMessage(
            content=[{"type": "reasoning", "reasoning": "x" * 500}],
            response_metadata={"model_provider": "openai"},
        )

        summary = extract_native_reasoning(message, char_cap=10)

        assert summary == "x" * 9 + "…"
