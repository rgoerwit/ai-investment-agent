"""One canonical debtToEquity ratio from the source boundary to every consumer."""

from __future__ import annotations

import math

import pytest
import structlog

from src.agents.capital_structure import assess_capital_structure_scale
from src.data.fetcher import SmartMarketDataFetcher
from src.data.merge_policy import smart_merge_with_quality
from src.decision_inputs import DecisionInputs
from src.validators.financial_rules import detect_red_flags
from src.validators.metric_extractor import extract_debt_to_equity, extract_metrics
from src.validators.sector_classifier import Sector


def _merge(source_results: dict) -> tuple[dict, dict]:
    return smart_merge_with_quality(source_results, "TEST", lambda results, _s: results)


class TestSourceBoundary:
    def test_tot_to_shape_yahoo_percent_equals_ratio_source(self):
        """yfinance 8.047 (percent) and a statement ratio 0.08047 agree."""
        merged_yf, meta_yf = _merge({"yfinance": {"debtToEquity": 8.047}})
        merged_st, meta_st = _merge(
            {
                "yfinance": {
                    "debtToEquity": 0.08047,
                    "_debtToEquity_source": "calculated_from_statements",
                }
            }
        )
        assert merged_yf["debtToEquity"] == pytest.approx(0.08047)
        assert merged_st["debtToEquity"] == pytest.approx(0.08047)
        assert "debtToEquity" not in meta_yf["source_conflicts"]
        assert "debtToEquity" not in meta_st["source_conflicts"]

    def test_percent_and_ratio_sources_do_not_register_a_conflict(self):
        merged, meta = _merge(
            {
                "yahooquery": {"debtToEquity": 8.047},
                "yfinance": {
                    "debtToEquity": 0.08047,
                    "_debtToEquity_source": "calculated_from_statements",
                },
            }
        )
        assert merged["debtToEquity"] == pytest.approx(0.08047)
        assert "debtToEquity" not in meta["source_conflicts"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(1.2, 0.012), (9.9, 0.099), (10.0, 0.10), (10.1, 0.101), (0.0, 0.0)],
    )
    def test_yahoo_percent_scaled_regardless_of_magnitude(self, raw, expected):
        merged, _ = _merge({"yfinance": {"debtToEquity": raw}})
        assert merged["debtToEquity"] == pytest.approx(expected)

    def test_missing_provenance_on_verified_source_uses_source_unit(self):
        merged, _ = _merge({"yahooquery": {"debtToEquity": 150.0}})
        assert merged["debtToEquity"] == pytest.approx(1.5)
        assert merged["_debtToEquity_unit"] == "ratio"

    def test_double_normalisation_is_idempotent(self):
        from src.data.debt_equity import normalize_source_results

        once = normalize_source_results({"yfinance": {"debtToEquity": 8.047}})
        twice = normalize_source_results(once)
        assert once["yfinance"]["debtToEquity"] == pytest.approx(0.08047)
        assert twice["yfinance"]["debtToEquity"] == once["yfinance"]["debtToEquity"]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "abc", True, -0.5])
    def test_invalid_values_are_dropped(self, bad):
        merged, _ = _merge({"yfinance": {"debtToEquity": bad, "marketCap": 1e9}})
        assert "debtToEquity" not in merged
        assert merged["marketCap"] == 1e9

    def test_negative_equity_statement_ratio_is_dropped(self):
        merged, _ = _merge(
            {
                "yfinance": {
                    "debtToEquity": -2.0,
                    "_debtToEquity_source": "calculated_from_statements",
                }
            }
        )
        assert "debtToEquity" not in merged

    @pytest.mark.parametrize("source", ["fmp", "alpha_vantage", "eodhd", "mystery"])
    def test_unverified_source_contributes_nothing(self, source):
        merged, meta = _merge({source: {"debtToEquity": 0.5, "marketCap": 1e9}})
        assert "debtToEquity" not in merged
        assert "debtToEquity" not in meta["field_sources"]

    def test_unverified_source_does_not_displace_or_conflict(self):
        merged, meta = _merge(
            {"fmp": {"debtToEquity": 1.8}, "yfinance": {"debtToEquity": 8.047}}
        )
        assert merged["debtToEquity"] == pytest.approx(0.08047)
        assert "debtToEquity" not in meta["source_conflicts"]


class TestCrossCheck:
    def _fetcher(self):
        return SmartMarketDataFetcher.__new__(SmartMarketDataFetcher)

    def _info(self, ratio):
        # debt 51.2M over book equity 17.428 * 36.4M = 634M -> 0.0807
        return {
            "debtToEquity": ratio,
            "totalDebt": 51.2e6,
            "bookValue": 17.428,
            "sharesOutstanding": 36.4e6,
            "enterpriseValue": 1.2967e9,
            "marketCap": 1.2955e9,
        }

    def test_x100_disagreement_is_logged_not_corrected(self, monkeypatch):
        from src.data import debt_equity

        logs: list[dict] = []

        class _Recorder:
            def warning(self, event, **fields):
                logs.append({"event": event, **fields})

            info = warning

        monkeypatch.setattr(debt_equity, "logger", _Recorder())
        info = debt_equity.cross_check(self._info(8.047), "TOT.TO")
        # The unit was fixed at the source boundary; book-value arithmetic from
        # another period or share class cannot say which side is wrong.
        assert info["debtToEquity"] == 8.047
        events = [e for e in logs if e["event"] == "debt_equity_unit_conflict"]
        assert len(events) == 1
        assert events[0]["symbol"] == "TOT.TO"

    def test_agreeing_ratio_is_left_alone(self):
        from src.data.debt_equity import cross_check

        info = cross_check(self._info(0.0807), "TOT.TO")
        assert info["debtToEquity"] == 0.0807

    def test_net_cash_is_not_unit_evidence(self):
        """EV below market cap must not rescale a genuine 4x ratio."""
        from src.data.debt_equity import cross_check

        info = self._info(4.0)
        info["enterpriseValue"] = 1.0e9
        assert cross_check(info, "X")["debtToEquity"] == 4.0

    def test_non_positive_equity_skips_cross_check(self):
        from src.data.debt_equity import cross_check

        info = self._info(8.047)
        info["bookValue"] = -3.0
        assert cross_check(info, "X")["debtToEquity"] == 8.047

    def test_normalize_data_integrity_runs_cross_check(self, monkeypatch):
        from src.data import debt_equity

        seen: list[str] = []

        class _Recorder:
            def warning(self, event, **_fields):
                seen.append(event)

            info = warning

        monkeypatch.setattr(debt_equity, "logger", _Recorder())
        fetcher = self._fetcher()
        fetcher.get_currency_rate = lambda *_a, **_k: 1.0  # type: ignore[method-assign]
        info = fetcher._normalize_data_integrity(self._info(8.047), "TOT.TO")
        assert info["debtToEquity"] == 8.047
        assert "debt_equity_unit_conflict" in seen

    def test_statement_ratio_survives_merge_and_cross_check(self):
        """A verified 8x statement ratio must not be rescaled by book-value data."""
        from src.data.debt_equity import cross_check
        from src.data.merge_policy import smart_merge_with_quality

        merged, _ = smart_merge_with_quality(
            {
                "yfinance": {
                    "debtToEquity": 8.0,
                    "_debtToEquity_source": "calculated_from_statements",
                    # Book value from another period: derived ratio is 0.08.
                    "totalDebt": 8.0e6,
                    "bookValue": 10.0,
                    "sharesOutstanding": 1.0e7,
                }
            },
            "LEV.X",
            lambda results, _symbol: results,
        )

        assert merged["_debtToEquity_source"] == "calculated_from_statements"
        assert cross_check(merged, "LEV.X")["debtToEquity"] == 8.0

    @pytest.mark.parametrize(
        ("book", "shares"),
        [(1e-200, 1e-200), (1e200, 1e200), (10**400, 1.0), ("12abc", 1.0)],
    )
    def test_degenerate_equity_inputs_are_skipped(self, book, shares):
        from src.data.debt_equity import cross_check

        info = self._info(0.0807)
        info["bookValue"] = book
        info["sharesOutstanding"] = shares
        assert cross_check(info, "X")["debtToEquity"] == 0.0807

    def test_conflict_bounds_derive_from_the_conversion_factor(self):
        from src.data import debt_equity

        assert debt_equity._UNIT_CONFLICT_LOW < debt_equity.PERCENT_PER_RATIO
        assert debt_equity._UNIT_CONFLICT_HIGH > debt_equity.PERCENT_PER_RATIO


class TestExtractor:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("DE_RATIO: 0.08", 8.0),
            ("DE_RATIO: 12", 1200.0),
            ("DE_RATIO: 45%", 45.0),
            ("DE_RATIO: 0.45", 45.0),
            ("DE_RATIO: 8.05 %", 8.05),
            ("DE_RATIO: 0.45 (financial debt only)", 45.0),
        ],
    )
    def test_de_ratio_units(self, text, expected):
        assert extract_debt_to_equity(_report(text)) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text",
        [
            "DE_RATIO: 8.05e-2",
            "DE_RATIO: 1,234",
            "DE_RATIO: 0.45x",
            "DE_RATIO: N/A",
            "DE_RATIO: 0.45 / 9",
            "DE_RATIO: 0.45 percent",
            "DE_RATIO: 0.45 or 9",
            "DE_RATIO: 0.45 (a) (b)",
        ],
    )
    def test_unsupported_de_ratio_notation_is_rejected_not_truncated(self, text):
        assert extract_debt_to_equity(_report(text)) is None

    def test_de_ratio_reads_the_final_data_block(self):
        report = _report("DE_RATIO: 9") + "\nRevised:\n" + _report("DE_RATIO: 0.08")
        assert extract_debt_to_equity(report) == pytest.approx(8.0)

    @pytest.mark.parametrize(
        "field_line",
        ["DE_RATIO: N/A", "DE_RATIO: 0.45 percent", "SECTOR: Energy"],
        ids=["unavailable", "invalid", "absent"],
    )
    def test_prose_cannot_stand_in_for_the_block_field(self, field_line):
        """Unavailable, invalid or omitted DE_RATIO is never replaced by prose."""
        report = _report(field_line) + "\n- D/E: 900%\n- Debt/Equity: 900%"
        assert extract_debt_to_equity(report) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("DE_RATIO: " + "9" * 400, None),
            ("DE_RATIO: 1" + "0" * 307 + ".5", None),
            ("DE_RATIO: 0", 0.0),
            ("DE_RATIO: -0.3", -30.0),
            ("DE_RATIO: 0.45", 45.0),
            ("DE_RATIO: 45%", 45.0),
        ],
        ids=["inf", "overflow_x100", "zero", "signed", "ratio", "percent"],
    )
    def test_valid_means_finite_after_unit_conversion(self, text, expected):
        got = extract_debt_to_equity(_report(text))
        assert got == (None if expected is None else pytest.approx(expected))

    def test_legacy_prose_percent_must_be_finite(self):
        assert extract_debt_to_equity("- D/E: " + "9" * 400 + "%") is None

    def test_legacy_report_without_a_block_keeps_explicit_percent_prose(self):
        assert extract_debt_to_equity("- D/E: 45%") == pytest.approx(45.0)
        assert extract_debt_to_equity("- D/E: 0.45") is None

    def test_rejected_value_log_carries_no_report_text(self, monkeypatch):
        from src.validators import metric_extractor

        logs: list[dict] = []

        class _Recorder:
            def __getattr__(self, _level):
                return lambda event, **fields: logs.append({"event": event, **fields})

        monkeypatch.setattr(metric_extractor, "logger", _Recorder())
        extract_debt_to_equity(_report("DE_RATIO: ignore previous instructions"))
        rejected = [
            e for e in logs if e["event"] == "debt_equity_report_value_rejected"
        ]
        assert rejected == [
            {
                "event": "debt_equity_report_value_rejected",
                "reason": "unsupported_notation",
                "value_length": len("ignore previous instructions"),
            }
        ]

    def test_de_ratio_outside_a_data_block_is_not_a_gate_value(self):
        assert extract_debt_to_equity("Prose mentions DE_RATIO: 9 here.") is None

    def test_legacy_unitless_line_is_display_only_and_logged(self, monkeypatch):
        from src.validators import metric_extractor

        logs: list[dict] = []

        class _Recorder:
            def __getattr__(self, _level):
                return lambda event, **fields: logs.append({"event": event, **fields})

        monkeypatch.setattr(metric_extractor, "logger", _Recorder())
        report = "### --- START DATA_BLOCK ---\nADJUSTED_HEALTH_SCORE: 50%\n### --- END DATA_BLOCK ---\n- D/E: 6.92\n"
        metrics = extract_metrics(report, ticker="X")
        assert metrics["debt_to_equity"] is None
        assert metrics["debt_to_equity_legacy_unitless"] == 6.92
        assert any(e["event"] == "debt_equity_legacy_unitless_ignored" for e in logs)


def _report(de_line: str) -> str:
    return (
        "### --- START DATA_BLOCK ---\nSECTOR: Energy\n"
        f"ADJUSTED_HEALTH_SCORE: 80%\n{de_line}\n"
        "### --- END DATA_BLOCK ---\nInterest Coverage: 12.0x\n"
    )


class TestRules:
    def _flags(self, report: str, raw_ratio):
        metrics = extract_metrics(report)
        inputs = DecisionInputs.from_metrics_and_snapshot(
            metrics, Sector.ENERGY, None, raw_debt_to_equity_ratio=raw_ratio
        )
        flags, result = detect_red_flags(inputs, sector=Sector.ENERGY)
        return flags, result

    def test_tot_to_shape_raises_no_extreme_leverage(self):
        # Report mislabels percent as ratio (8.047 -> 805%); canonical is 0.08047.
        flags, result = self._flags(_report("DE_RATIO: 8.047"), 0.08047)
        assert not [f for f in flags if f["type"] == "EXTREME_LEVERAGE"]
        assert result != "REJECT"

    def test_provider_percent_through_merge_reaches_the_screen_as_a_ratio(self):
        """yfinance 8.047 (percent) -> merge -> raw ratio -> no 805% reject."""
        merged, _ = smart_merge_with_quality(
            {"yfinance": {"debtToEquity": 8.047, "totalDebt": 51.2e6}},
            "TOT.TO",
            lambda results, _symbol: results,
        )
        flags, result = self._flags(_report("DE_RATIO: 0.08"), merged["debtToEquity"])
        assert merged["debtToEquity"] == pytest.approx(0.08047)
        assert not [
            f for f in flags if f["type"] in {"EXTREME_LEVERAGE", "DE_DISCREPANCY"}
        ]
        assert result != "REJECT"

    def test_genuine_805_percent_is_rejected_in_energy(self):
        flags, result = self._flags(_report("DE_RATIO: 8.05"), 8.05)
        assert [f for f in flags if f["type"] == "EXTREME_LEVERAGE"]
        assert result == "REJECT"

    def test_conflicting_report_does_not_override_canonical(self):
        # Canonical 8.05x is extreme; report prose claims 0.45 (e.g. holdco note).
        flags, result = self._flags(_report("DE_RATIO: 0.45"), 8.05)
        assert [f for f in flags if f["type"] == "EXTREME_LEVERAGE"]
        assert [f for f in flags if f["type"] == "DE_DISCREPANCY"]
        assert result == "REJECT"

    def test_discrepancy_flag_never_blocks(self):
        flags, _ = self._flags(_report("DE_RATIO: 8.05"), 0.08)
        disc = [f for f in flags if f["type"] == "DE_DISCREPANCY"]
        assert disc
        assert disc[0]["action"] == "REVIEW"
        assert not [f for f in flags if f["type"] == "EXTREME_LEVERAGE"]

    def test_without_canonical_value_report_de_ratio_still_gates(self):
        flags, result = self._flags(_report("DE_RATIO: 9.0"), None)
        assert [f for f in flags if f["type"] == "EXTREME_LEVERAGE"]
        assert result == "REJECT"


class TestConsumerAgreement:
    RAW = 12.0  # canonical ratio: 1200%

    def test_pre_screen_and_capital_structure_read_the_same_value(self):
        raw_metrics = {
            "debtToEquity": self.RAW,
            "totalDebt": 1.2e9,
            "financialCurrency": "CAD",
        }
        scale = assess_capital_structure_scale(
            {"amount": "CAD 10M", "amount_basis": "PRESENT_VALUE"},
            raw_metrics,
            leverage_threshold=800,
        )
        inputs = DecisionInputs.from_metrics_and_snapshot(
            {},
            Sector.ENERGY,
            None,
            raw_debt_to_equity_ratio=raw_metrics["debtToEquity"],
        )
        assert scale["reported_de_pct"] == pytest.approx(self.RAW * 100)
        assert inputs.raw_debt_to_equity_ratio == self.RAW
        assert scale["reported_de_pct"] == pytest.approx(
            inputs.raw_debt_to_equity_ratio * 100
        )


class TestSourceToScreenChain:
    """Provider observation -> merge -> canonical claim -> HEALTH scorecard ->
    report projection -> state-derived decision inputs -> pre-screen."""

    _PAYLOAD = {
        "returnOnEquity": 0.2,
        "returnOnAssets": 0.1,
        "operatingMargins": 0.15,
        "totalDebt": 100.0,
        "totalCash": 20.0,
        "ebitda": 50.0,
        "currentRatio": 2.0,
        "operatingCashflow": 30.0,
        "freeCashflow": 20.0,
        "marketCap": 200.0,
        "trailingPE": 12.5,
        "enterpriseToEbitda": 8.0,
        "priceToBook": 1.2,
    }

    @classmethod
    def _run(cls, raw_percent: float, sector: str, report_de: str, de_award: str):
        import re

        from src.agents.decision_nodes import _decision_inputs_from_state
        from src.analysis_snapshot import (
            add_validated_derivations,
            build_pre_senior_snapshot,
            project_analysis_report,
        )
        from src.tooling.structured_ingress import build_structured_ingress_record
        from src.validators.red_flag_detector import RedFlagDetector

        merged, _ = smart_merge_with_quality(
            {"yfinance": {**cls._PAYLOAD, "debtToEquity": raw_percent}},
            "CHAIN.X",
            lambda results, _symbol: results,
        )
        state = {
            "company_of_interest": "CHAIN.X",
            "structured_inputs": {
                "raw_financial_metrics": build_structured_ingress_record(
                    merged,
                    agent_key="junior_fundamentals_analyst",
                    tool_name="get_financial_metrics",
                )
            },
        }
        snapshot = build_pre_senior_snapshot(state)
        available = 11 if de_award == "REMOVED" else 12
        earned = available - (1 if de_award == "0" else 0)
        report = (
            "### --- START DATA_BLOCK ---\n"
            f"SECTOR: {sector}\nDE_RATIO: {report_de}\n"
            "HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=1; OPERATING_MARGIN=1; "
            f"DE_RATIO={de_award}; NET_DEBT_EBITDA=1; CURRENT_RATIO=1; "
            "OCF_POSITIVE=1; FCF_POSITIVE=1; FCF_YIELD=1; PE_OR_PEG=1; "
            "EV_EBITDA=1; PB_OR_PS=1\n"
            f"RAW_HEALTH_SCORE: {earned}/{available}\n"
            f"ADJUSTED_HEALTH_SCORE: {round(100 * earned / available, 1)}% "
            f"(based on {available} available points)\n"
            "### --- END DATA_BLOCK ---"
        )
        derived = add_validated_derivations(snapshot, report)
        projected = project_analysis_report(report, derived)
        inputs = _decision_inputs_from_state(
            {**state, "fundamentals_report": projected, "analysis_snapshot": derived},
            "CHAIN.X",
        )
        flags, result = RedFlagDetector.detect_red_flags(
            inputs, "CHAIN.X", RedFlagDetector.detect_sector(projected)
        )
        fields = {c["id"]: c["field"] for c in derived["claims"].values()}
        health = derived["scorecards"]["HEALTH"]
        de = health["criteria"]["DE_RATIO"]
        projected_score = re.search(r"ADJUSTED_HEALTH_SCORE: ([\d.]+)%", projected)
        return {
            "claim": next(
                c["value"]
                for c in derived["claims"].values()
                if c["field"] == "DE_RATIO_RAW"
            ),
            "health": health,
            "de_award": de["award"],
            "de_lineage": [fields[i] for i in de["derived_from"]],
            "projected_pct": float(projected_score.group(1)),
            "inputs": inputs,
            "types": {f["type"] for f in flags},
            "result": result,
        }

    @staticmethod
    def _assert_scorecard_consumed(out):
        assert out["health"]["decision_eligible"] is True
        assert out["projected_pct"] == pytest.approx(out["health"]["percentage"])
        assert out["inputs"].health_decision_pct == pytest.approx(
            out["health"]["percentage"]
        )

    def test_low_canonical_leverage_with_exaggerated_report(self):
        out = self._run(8.047, "Energy", "8.05", "0")
        assert out["claim"] == "0.08"
        assert out["inputs"].raw_debt_to_equity_ratio == pytest.approx(0.08047)
        self._assert_scorecard_consumed(out)
        # The D/E award is sector-dependent and stays the analyst's; its lineage
        # is the canonical claim, and the screen follows the canonical value.
        assert out["de_award"] == "0"
        assert out["de_lineage"] == ["DE_RATIO_RAW"]
        assert "EXTREME_LEVERAGE" not in out["types"]
        assert "DE_DISCREPANCY" in out["types"]
        assert out["result"] != "REJECT"

    def test_high_canonical_leverage_with_understated_report(self):
        out = self._run(805.0, "Energy", "0.45", "1")
        assert out["claim"] == "8.05"
        assert out["inputs"].raw_debt_to_equity_ratio == pytest.approx(8.05)
        self._assert_scorecard_consumed(out)
        assert out["de_lineage"] == ["DE_RATIO_RAW"]
        assert {"EXTREME_LEVERAGE", "DE_DISCREPANCY"} <= out["types"]
        assert out["result"] == "REJECT"

    def test_financials_remove_the_de_criterion_and_its_screen(self):
        out = self._run(805.0, "Financials", "0.45", "REMOVED")
        self._assert_scorecard_consumed(out)
        assert out["de_award"] == "REMOVED"
        assert out["de_lineage"] == []
        assert out["health"]["available"] == pytest.approx(11.0)
        assert "EXTREME_LEVERAGE" not in out["types"]
