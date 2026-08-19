"""The bounded-verification loops must agree on how tool calls are issued.

Three surfaces have to stay aligned for a verification seat: the loop executes a
turn concurrently, the budget allows a whole plan to land in one turn, and the
prompt tells the model to plan then batch. When they drifted apart (Aug 2026) a
batch-planning vendor had its plan truncated at 4 of 7 calls, re-planned, spent
the wall-clock budget on round trips, and on one ticker crossed the partial-tool-
failure ratio — discarding the entire cross-check.

These are cheap static assertions, in the spirit of
``test_consultant_gate.py::TestGateFlagTokensAreLive``.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.consultant_tools import get_consultant_tools
from src.eval.prompt_contracts import prompt_text

# Seats that run a bounded loop whose purpose is *systematic verification*: they
# plan a set of independent lookups up front, so batching is the correct shape.
# Keyed by ``agent_key``, which is not always the filename (auditor.json declares
# ``global_forensic_auditor``) — ``prompt_text`` resolves the former.
_VERIFICATION_LOOP_PROMPTS = ("consultant", "global_forensic_auditor")

# Deliberately excluded. Legal Counsel also runs a tool loop, but it fills gaps
# for a strict-JSON extractor rather than running a planned verification sweep,
# it has no per-turn cap to collide with, and its output contract is already
# guarded by ``test_prompt_schema_keys.py``. Recorded here so the exclusion is a
# decision rather than an oversight.
_EXCLUDED_TOOL_LOOP_PROMPTS = {"legal_counsel": "gap-filling JSON extractor"}

_BATCHING_MARKER = "PLAN-THEN-BATCH"


@pytest.mark.parametrize("agent_key", _VERIFICATION_LOOP_PROMPTS)
def test_verification_prompts_mandate_plan_then_batch(agent_key: str) -> None:
    """Both verification loops must teach the same tool-call discipline."""
    # prompt_text reads the on-disk prompt; get_prompt can be mutated by env.
    text = prompt_text(agent_key)
    assert _BATCHING_MARKER in text, (
        f"{agent_key}.json lost its {_BATCHING_MARKER} directive. Its loop "
        "executes a turn concurrently and caps fan-out per turn, so a model "
        "left to probe sequentially spends the budget on round trips."
    )
    assert "MANDATORY" in text.split(_BATCHING_MARKER, 1)[1][:80], (
        f"{agent_key}.json states {_BATCHING_MARKER} as guidance rather than a "
        "requirement; a batching hint the model may ignore is the drift this "
        "guard exists to catch."
    )


def test_excluded_tool_loops_are_recorded_not_forgotten() -> None:
    """The exclusion list names a real prompt and carries a reason."""
    for agent_key, reason in _EXCLUDED_TOOL_LOOP_PROMPTS.items():
        assert prompt_text(agent_key), f"{agent_key} is not a real prompt"
        assert reason, f"{agent_key} is excluded with no recorded reason"


def test_consultant_fan_out_fits_a_full_verification_round() -> None:
    """The per-turn cap must admit every tool being used at least twice.

    Derived from the live registry rather than a literal, so adding a
    verification tool cannot silently re-create the truncation this guard
    exists to prevent.
    """
    tool_count = len(get_consultant_tools())
    cap = Settings.model_fields["consultant_max_tool_calls_per_turn"].default
    assert cap >= 2 * tool_count, (
        f"consultant_max_tool_calls_per_turn={cap} cannot fit two calls for "
        f"each of {tool_count} consultant tools in one turn; a planned batch "
        "would be truncated and the surplus calls SKIPPED."
    )


def test_consultant_loop_ceiling_is_not_widened_by_accident() -> None:
    """Reshaping the budget must stay budget-neutral on executed tool calls.

    4x3 became 6x2: the executed-tool ceiling is unchanged at 12 while the
    LLM-call ceiling drops. A future edit that raises both numbers would grow
    per-ticker cost silently.
    """
    fields = Settings.model_fields
    iterations = fields["consultant_max_tool_iterations"].default
    per_turn = fields["consultant_max_tool_calls_per_turn"].default
    assert iterations * per_turn <= 12, (
        f"executed-tool ceiling {iterations}x{per_turn}={iterations * per_turn} "
        "exceeds the 12 the loop was budgeted for."
    )
