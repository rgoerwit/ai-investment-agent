"""Tests for the P0-2 Consultant bypass gate in src/graph/routing.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.graph import routing
from src.graph.routing import (
    CONSULTANT_SKIP_SENTINEL,
    _classify_rm_verdict,
    consultant_gate_router,
    should_invoke_consultant,
)


@pytest.mark.parametrize(
    "plan, expected",
    [
        # The Research Manager emits markdown-prefixed headers; the classifier
        # must see through "### " and the FINAL/INVESTMENT qualifiers.
        ("### FINAL RECOMMENDATION: REJECT", "negative"),
        ("### FINAL RECOMMENDATION: DO NOT INITIATE", "negative"),
        ("### INVESTMENT RECOMMENDATION: BUY", "positive"),
        ("### FINAL RECOMMENDATION: STRONG BUY", "positive"),
        ("FINAL RECOMMENDATION: BUY", "positive"),
        ("Notes: mixed signals only", "ambiguous"),
    ],
)
def test_classify_rm_verdict_tolerates_markdown_headers(plan, expected):
    assert _classify_rm_verdict(plan) == expected


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


def test_quick_negative_with_data_discrepancy_keeps_consultant():
    """A reject resting on an unresolved data conflict still gets the
    Consultant's reconciliation (145020.KQ OCF period-mismatch, July 2026)."""
    state = _state(
        plan=_NEGATIVE_PLAN,
        auditor=_CLEAN_AUDITOR,
        red_flags=["OCF_SOURCE_DISCREPANCY"],
    )
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True
    assert reason == "default_invoke"


def test_quick_negative_with_dict_shaped_discrepancy_flag():
    """Production red flags are dicts with a 'type' key — must not raise and
    must still be recognized."""
    state = _state(plan=_NEGATIVE_PLAN, auditor=_CLEAN_AUDITOR)
    state["red_flags"] = [
        {"type": "OCF_FILING_VALUE_UNCORROBORATED", "severity": "WARNING"}
    ]
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True
    assert reason == "default_invoke"


def test_quick_negative_with_non_discrepancy_flag_still_skips():
    """Well-founded value-trap rejects don't need a second opinion."""
    state = _state(
        plan=_NEGATIVE_PLAN,
        auditor=_CLEAN_AUDITOR,
        red_flags=["VALUE_TRAP_HIGH_RISK"],
    )
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is False
    assert reason == "rm_clear_negative"


def test_quick_negative_with_conflict_marker_keeps_consultant():
    plan = _NEGATIVE_PLAN + "\nNote: CONFLICT between analyst and filing data."
    state = _state(plan=plan, auditor=_CLEAN_AUDITOR)
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=True))
    assert invoke is True
    assert reason == "default_invoke"


def test_full_mode_negative_with_discrepancy_unchanged():
    state = _state(plan=_NEGATIVE_PLAN, red_flags=["OCF_SOURCE_DISCREPANCY"])
    invoke, reason = should_invoke_consultant(state, _config(quick_mode=False))
    assert invoke is True
    assert reason == "full_mode"


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


class TestSharedOpenAIPlaneInvariant:
    """Consultant and Auditor both gate on the shared OpenAI cross-check plane
    (``enable_consultant`` + OpenAI key) via ``_is_auditor_enabled`` /
    ``is_openai_consultant_available`` — neither is subordinate to the other's
    LLM object. The auditor has independent PM (OCF corroboration) and report
    consumers, so it must NOT be disabled merely because the consultant object
    failed to build while the plane is up.
    """

    def test_auditor_off_when_consultant_disabled(self, monkeypatch):
        monkeypatch.setattr(routing.config, "enable_consultant", False)
        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: True)
        assert routing._is_auditor_enabled() is False

    def test_auditor_off_when_no_openai_key(self, monkeypatch):
        monkeypatch.setattr(routing.config, "enable_consultant", True)
        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: False)
        assert routing._is_auditor_enabled() is False

    def test_auditor_on_when_plane_up(self, monkeypatch):
        monkeypatch.setattr(routing.config, "enable_consultant", True)
        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: True)
        assert routing._is_auditor_enabled() is True

    def test_auditor_enable_matches_openai_availability_gate(self, monkeypatch):
        # The auditor gate and the consultant availability gate read the SAME two
        # conditions; enabling the plane flips both together.
        monkeypatch.setattr(routing.config, "enable_consultant", True)
        for available in (True, False):
            monkeypatch.setattr(
                routing, "is_openai_consultant_available", lambda a=available: a
            )
            assert routing._is_auditor_enabled() is available


class TestGateFlagTokensAreLive:
    """Every flag name the gate keys on must actually be emitted by src/.

    `CMIC_LISTED` sat in the blocking set for months producing nothing: the real
    emitters are CMIC_FLAGGED / CMIC_UNCERTAIN, so a genuine NS-CMIC hit never
    kept the Consultant active through this gate. A dead string here fails open
    and is invisible, so it gets a structural guard rather than a comment.
    """

    @staticmethod
    def _emitted_flag_types() -> set[str]:
        import ast
        import pathlib

        src_root = pathlib.Path(routing.__file__).resolve().parents[1]
        emitted: set[str] = set()
        for path in src_root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                # A flag is emitted as a dict literal carrying a "type" key.
                if not isinstance(node, ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values, strict=False):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "type"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        emitted.add(value.value)
        return emitted

    @pytest.mark.parametrize(
        "gate_set_name",
        ["_GATE_BLOCKING_RED_FLAGS", "_GATE_DATA_DISCREPANCY_FLAGS"],
    )
    def test_gate_flags_are_emitted_somewhere(self, gate_set_name):
        emitted = self._emitted_flag_types()
        assert emitted, "AST scan found no flag literals — the scan itself is broken"
        gate_flags = getattr(routing, gate_set_name)
        dead = sorted(flag for flag in gate_flags if flag not in emitted)
        assert not dead, f"{gate_set_name} references flags nothing emits: {dead}"

    def test_cmic_tokens_match_the_detector(self):
        from src.validators.red_flag_detector import RedFlagDetector

        for status, expected in (
            ("FLAGGED", "CMIC_FLAGGED"),
            ("UNCERTAIN", "CMIC_UNCERTAIN"),
        ):
            flags = RedFlagDetector.detect_legal_flags({"cmic_status": status}, "T")
            assert flags[0]["type"] == expected
            assert expected in routing._GATE_BLOCKING_RED_FLAGS
