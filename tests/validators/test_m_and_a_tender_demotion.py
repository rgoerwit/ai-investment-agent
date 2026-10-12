"""An ACTIVE_TENDER with no tender price is demoted before anything acts on it.

6670.TW is the acquirer in a tender for 51% of 6904.TW (tender price N/A). The
Senior still wrote ``M_AND_A_STATUS: ACTIVE_TENDER``, which exempts the health and
growth code gates and routes held positions to an M&A exit. The status is only
honored when a positive tender price accompanies it.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest
import structlog

from src import data_block_utils
from src.agents.verdict_policy import assess_health_gate, assess_verdict_policy
from src.data_block_utils import effective_m_and_a_status
from src.decision_inputs import DecisionInputs
from src.retrospective import extract_snapshot
from src.validators.metric_extractor import extract_metrics

ROOT = Path(__file__).resolve().parents[2]


class _RecordingLogger:
    """structlog caches bound loggers on first use, so capture_logs is unreliable."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def info(self, event: str, **fields) -> None:
        self.events.append({"event": event, **fields})


def _report(status: str, price: str) -> str:
    return (
        "### --- START DATA_BLOCK ---\n"
        f"M_AND_A_STATUS: {status}\n"
        f"M_AND_A_TENDER_PRICE: {price}\n"
        "### --- END DATA_BLOCK ---"
    )


class TestEffectiveStatus:
    @pytest.mark.parametrize(
        "price",
        ["TWD 120.5", "JPY 1,200", "$18", "120", "TWD 120.5 per share (cash)"],
    )
    def test_priced_tender_is_kept(self, price):
        assert effective_m_and_a_status("ACTIVE_TENDER", price) == "ACTIVE_TENDER"

    @pytest.mark.parametrize(
        "price",
        [
            "N/A",
            "",
            None,
            "0",
            "0.00",
            "TBD",
            "-",
            "51%",
            "N/A (51% stake)",
            "N/A (period ends 2026-11-30)",
        ],
    )
    def test_unpriced_tender_is_demoted_and_logged(self, price, monkeypatch):
        recorder = _RecordingLogger()
        monkeypatch.setattr(data_block_utils, "logger", recorder)
        assert effective_m_and_a_status("ACTIVE_TENDER", price) == "RUMORED"
        assert len(recorder.events) == 1
        assert recorder.events[0]["event"] == "m_and_a_tender_unpriced_demoted"
        assert recorder.events[0]["status"] == "ACTIVE_TENDER"

    @pytest.mark.parametrize("status", ["RUMORED", "NONE", None])
    def test_other_statuses_pass_through(self, status, monkeypatch):
        recorder = _RecordingLogger()
        monkeypatch.setattr(data_block_utils, "logger", recorder)
        assert effective_m_and_a_status(status, "N/A") == status
        assert effective_m_and_a_status(status, "JPY 100") == status
        assert recorder.events == []


class TestMetricExtractor:
    def test_acquirer_without_price_is_not_active_tender(self):
        metrics = extract_metrics(_report("ACTIVE_TENDER", "N/A"))
        assert metrics["m_and_a_status"] == "RUMORED"

    def test_priced_target_stays_active_tender(self):
        metrics = extract_metrics(_report("ACTIVE_TENDER", "JPY 1,200"))
        assert metrics["m_and_a_status"] == "ACTIVE_TENDER"

    def test_non_tender_status_is_unchanged(self):
        assert extract_metrics(_report("NONE", "N/A"))["m_and_a_status"] == "NONE"


def _gate_inputs(report: str) -> DecisionInputs:
    metrics = extract_metrics(report)
    base = DecisionInputs.from_metrics_and_snapshot(
        {"adjusted_health_score": 30, "adjusted_growth_score": 10}, None, None
    )
    return replace(
        base,
        health_decision_pct=30,
        growth_decision_pct=10,
        health_exception_pb_ratio=None,
        decision_metrics={
            **base.decision_metrics,
            "m_and_a_status": metrics["m_and_a_status"],
        },
    )


def _pm(verdict: str) -> str:
    return (
        f"### PORTFOLIO MANAGER VERDICT: {verdict}\n"
        f"### --- START PM_BLOCK ---\nVERDICT: {verdict}\nPOSITION_SIZE: 0.0\n"
        "### --- END PM_BLOCK ---"
    )


class TestGatesFollowTheEffectiveStatus:
    def test_unpriced_tender_no_longer_exempts_health(self):
        resolved = _gate_inputs(_report("ACTIVE_TENDER", "N/A"))
        gate = assess_health_gate(resolved)
        assert gate.applicable is True
        assert gate.status == "FAIL"
        assert assess_verdict_policy(_pm("BUY"), decision_inputs=resolved) is not None

    def test_priced_tender_remains_exempt(self):
        resolved = _gate_inputs(_report("ACTIVE_TENDER", "TWD 120"))
        assert assess_health_gate(resolved).applicable is False
        assert assess_verdict_policy(_pm("HOLD"), decision_inputs=resolved) is None


class TestSnapshot:
    @staticmethod
    def _snapshot(price: str) -> dict:
        return extract_snapshot(
            {
                "final_trade_decision": "PORTFOLIO MANAGER VERDICT: HOLD",
                "fundamentals_report": _report("ACTIVE_TENDER", price),
            },
            "6670.TW",
        )

    def test_unpriced_tender_is_snapshotted_as_rumored(self):
        assert self._snapshot("N/A")["m_and_a_status"] == "RUMORED"

    def test_priced_tender_is_snapshotted_as_active(self):
        assert self._snapshot("TWD 120")["m_and_a_status"] == "ACTIVE_TENDER"


def test_value_trap_call_sites_use_the_helper():
    tree = ast.parse((ROOT / "src/agents/decision_nodes.py").read_text())
    sites = [
        kw
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "m_and_a_status"
    ]
    assert len(sites) == 2
    for kw in sites:
        assert isinstance(kw.value, ast.Call)
        assert getattr(kw.value.func, "id", None) == "effective_m_and_a_status"


def test_prompts_carry_the_subject_role_contract():
    fla = json.loads((ROOT / "prompts/foreign_language_analyst.json").read_text())
    senior = json.loads((ROOT / "prompts/fundamentals_analyst.json").read_text())
    assert "- Subject Role: [TARGET / ACQUIRER / OTHER]" in fla["system_message"]
    instruction = senior["system_message"].split("9. **M&A Event", 1)[1]
    instruction = instruction.split("**CRITICAL**", 1)[0]
    assert "Subject Role" in instruction and "TARGET" in instruction
    assert "ACQUIRER" in instruction
