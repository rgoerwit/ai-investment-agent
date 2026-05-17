"""Tests for the P0-2 Consultant bypass gate in src/graph/routing.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.graph.routing import (
    CONSULTANT_SKIP_SENTINEL,
    consultant_gate_router,
    should_invoke_consultant,
)


def _config(quick_mode: bool) -> dict[str, Any]:
    context = SimpleNamespace(quick_mode=quick_mode)
    return {"configurable": {"context": context}}


def _state(
    *,
    plan: str | None = None,
    auditor: str | None = None,
    red_flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "investment_plan": plan,
        "auditor_report": auditor,
        "red_flags": red_flags or [],
    }


_POSITIVE_PLAN = "FINAL RECOMMENDATION: BUY\nConviction: HIGH"
_NEGATIVE_PLAN = "FINAL RECOMMENDATION: DO NOT INITIATE\nConviction: HIGH"
_AMBIGUOUS_PLAN = "Notes: Mixed signals; awaiting next quarter."
_CLEAN_AUDITOR = "STATUS: CLEAN\nNo material anomalies detected."
_PROBLEM_AUDITOR = "STATUS: PROBLEM\nMaterial restatement risk identified."


# ---------- should_invoke_consultant -----------------------------------------


def test_full_mode_always_invokes():
    state = _state(plan=_POSITIVE_PLAN, auditor=_CLEAN_AUDITOR)
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=False))
    assert invoke is True
    assert reason == "full_mode"


def test_quick_clean_consensus_skips():
    state = _state(plan=_POSITIVE_PLAN, auditor=_CLEAN_AUDITOR, red_flags=[])
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is False
    assert reason == "clean_consensus"


def test_quick_negative_rm_verdict_skips():
    state = _state(plan=_NEGATIVE_PLAN, auditor=_CLEAN_AUDITOR)
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is False
    assert reason == "rm_clear_negative"


def test_quick_red_flag_keeps_consultant_even_with_clean_audit():
    state = _state(
        plan=_POSITIVE_PLAN,
        auditor=_CLEAN_AUDITOR,
        red_flags=["VALUE_TRAP_HIGH_RISK"],
    )
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True
    assert reason == "default_invoke"


def test_quick_problem_auditor_keeps_consultant():
    state = _state(plan=_POSITIVE_PLAN, auditor=_PROBLEM_AUDITOR)
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True


def test_quick_ambiguous_plan_keeps_consultant():
    state = _state(plan=_AMBIGUOUS_PLAN, auditor=_CLEAN_AUDITOR)
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True


def test_quick_conflict_marker_keeps_consultant():
    plan = _POSITIVE_PLAN + "\nNote: CONFLICT between analyst and filing data."
    state = _state(plan=plan, auditor=_CLEAN_AUDITOR)
    invoke, _ = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True


@pytest.mark.parametrize(
    "verdict_line",
    [
        "RECOMMENDATION: STRONG SELL",
        "VERDICT: SELL",
        "FINAL RECOMMENDATION: STRONG_HOLD",
        "DECISION: REJECT",
        "RECOMMENDATION: do not initiate",
    ],
)
def test_quick_negative_verdict_variants_skip(verdict_line):
    state = _state(plan=f"... preamble ...\n{verdict_line}\n... tail ...")
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is False
    assert reason == "rm_clear_negative"


def test_quick_missing_auditor_treated_as_clean():
    """Auditor unset (e.g., ENABLE_CONSULTANT off elsewhere) should not block
    the Fast-Pass — the absence of an auditor report is not evidence of
    problems."""
    state = _state(plan=_POSITIVE_PLAN, auditor=None)
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is False
    assert reason == "clean_consensus"


def test_quick_unparseable_auditor_keeps_consultant():
    """If the auditor returned text but with no STATUS line, prefer to keep
    Consultant — conservative behavior."""
    state = _state(plan=_POSITIVE_PLAN, auditor="Some prose with no status header.")
    invoke, _ = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True


# ---------- consultant_gate_router routes node names -------------------------


def test_router_returns_consultant_when_invoke():
    state = _state(plan=_AMBIGUOUS_PLAN, auditor=_CLEAN_AUDITOR)
    assert consultant_gate_router(state, _config(quick_mode=True)) == "Consultant"


def test_router_returns_skip_when_clean():
    state = _state(plan=_POSITIVE_PLAN, auditor=_CLEAN_AUDITOR)
    assert consultant_gate_router(state, _config(quick_mode=True)) == "Consultant Skip"


def test_router_returns_skip_when_negative_rm():
    state = _state(plan=_NEGATIVE_PLAN)
    assert consultant_gate_router(state, _config(quick_mode=True)) == "Consultant Skip"


def test_router_logs_skip_reason(monkeypatch):
    """Skip path emits the grep-able structured event with the actual reason."""
    captured: list[tuple[str, dict[str, Any]]] = []

    from src.graph import routing as routing_mod

    class _StubLogger:
        def info(self, event, **kwargs):
            captured.append((event, kwargs))

        def debug(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    monkeypatch.setattr(routing_mod, "logger", _StubLogger())
    state = _state(plan=_NEGATIVE_PLAN)
    consultant_gate_router(state, _config(quick_mode=True))
    skip_events = [
        (event, kw)
        for event, kw in captured
        if event == "consultant_skipped_for_screening"
    ]
    assert len(skip_events) == 1
    assert skip_events[0][1].get("reason") == "rm_clear_negative"


def test_sentinel_records_reason():
    text = CONSULTANT_SKIP_SENTINEL.format(reason="clean_consensus")
    assert "clean_consensus" in text
    assert "SKIPPED_BY_GATE" in text
