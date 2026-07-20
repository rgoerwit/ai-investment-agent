"""Retail wording guard (July 2026 alignment).

The operator is a long-term retail investor: recommendations must not read as
trading-desk instructions. "Stop-loss" never appears in operator-facing output;
a price break renders as a review trigger coupled to fundamentals, and SELLs
carry fundamental-failure framing. These tests pin the surfaced vocabulary so
a refactor cannot quietly reintroduce the old framing.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.portfolio_manager import format_report
from src.ibkr.portfolio_presentation import (
    ACTION_BASIS_LABELS,
    SELL_TYPE_LABELS,
)
from src.ibkr.reconciler import reconcile
from tests.ibkr.reconciler_cases import (
    _make_analysis,
    _make_portfolio,
    _make_position,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Operator-facing prompts that must not teach stop-loss framing. The TRADE_BLOCK
# machine tokens (STOP:/TARGET_*) are exempt — they are internal field names the
# prompt explicitly documents as reference levels, not orders.
_GUARDED_PROMPTS = (
    "trader.json",
    "portfolio_manager.json",
    "market_analyst.json",
    "risky_analyst.json",
    "safe_analyst.json",
    "neutral_analyst.json",
    "fundamentals_analyst.json",
    "news_analyst.json",
    "research_manager.json",
    "bull_researcher.json",
    "bear_researcher.json",
    "writer.json",
    "editor.json",
)


class TestLabelVocabulary:
    def test_no_stop_language_in_display_labels(self):
        for label in (*SELL_TYPE_LABELS.values(), *ACTION_BASIS_LABELS.values()):
            assert "STOP" not in label.upper(), label

    def test_price_break_labels_read_as_reviews(self):
        assert SELL_TYPE_LABELS["STOP_BREACH"] == "PRICE-DROP REVIEW"
        assert ACTION_BASIS_LABELS["STOP_LOSS"] == "PRICE-DROP REVIEW"


class TestRenderedReportWording:
    def _report(self) -> str:
        # One of each surfaced class produced by the CURRENT pipeline: a price
        # break, a profit-take advisory, an overweight advisory, and a hold.
        positions = [
            _make_position(ticker="7203.T", current_price=1700, conid=1),
            _make_position(
                ticker="6758.T",
                current_price=2600,
                avg_cost=2000,
                conid=2,
                tax_term="LONG_TERM",
            ),
            _make_position(
                ticker="9432.T", current_price=2100, market_value_usd=30000, conid=3
            ),
            _make_position(ticker="8035.T", current_price=2150, conid=4),
        ]
        analyses = {
            "7203.T": _make_analysis(ticker="7203.T", stop_price=1900),
            "6758.T": _make_analysis(
                ticker="6758.T",
                target_1=2500,
                capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
            ),
            "9432.T": _make_analysis(ticker="9432.T", verdict="BUY", size_pct=5.0),
            "8035.T": _make_analysis(ticker="8035.T", verdict="BUY"),
        }
        items = reconcile(
            positions,
            analyses,
            _make_portfolio(value=100000),
            overweight_threshold_pct=20.0,
        )
        return format_report(items, _make_portfolio())

    def test_no_stop_loss_or_breach_language_in_report(self):
        report = self._report()
        lowered = report.lower()
        assert "stop-loss" not in lowered
        assert "stop loss" not in lowered
        assert "stop breached" not in lowered
        assert "[stop breach]" not in lowered

    def test_price_break_couples_to_fundamentals(self):
        report = self._report()
        assert "review level" in report
        assert "fundamental failure evidence" in report

    def test_no_executable_sell_or_trim_rows(self):
        report = self._report()
        assert "→ SELL" not in report
        assert "→ TRIM" not in report


class TestPromptWording:
    def test_guarded_prompts_have_no_stop_loss_language(self):
        for name in _GUARDED_PROMPTS:
            sm = json.loads((_PROMPTS_DIR / name).read_text())["system_message"]
            lowered = sm.lower()
            assert "stop-loss" not in lowered, name
            assert "stop loss" not in lowered, name
            assert "profit target" not in lowered, name
            assert "trailing stop" not in lowered, name

    def test_standalone_analysis_prompts_do_not_claim_sale_authority(self):
        forbidden = (
            "sell if held",
            "instant sell",
            "mandatory sell",
            "immediate sell",
            "triggers mandatory sell",
        )
        for name in _GUARDED_PROMPTS:
            sm = json.loads((_PROMPTS_DIR / name).read_text())["system_message"]
            lowered = sm.lower()
            for phrase in forbidden:
                assert phrase not in lowered, f"{name}: {phrase}"

    def test_pm_defers_held_disposition_to_reconciliation(self):
        sm = json.loads((_PROMPTS_DIR / "portfolio_manager.json").read_text())[
            "system_message"
        ]
        assert "Held-position disposition is deferred to portfolio reconciliation" in sm
        assert "VERDICT: [BUY/HOLD/DO_NOT_INITIATE]" in sm
        assert "VERDICT: [BUY/HOLD/DO_NOT_INITIATE/SELL]" not in sm

    def test_trader_prompt_keeps_machine_tokens(self):
        sm = json.loads((_PROMPTS_DIR / "trader.json").read_text())["system_message"]
        for token in ("TRADE_BLOCK:", "STOP:", "TARGET_1:", "TARGET_2:", "HORIZON:"):
            assert token in sm, token
