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
        ],
    )
    def test_unsupported_de_ratio_notation_is_rejected_not_truncated(self, text):
        assert extract_debt_to_equity(_report(text)) is None

    def test_de_ratio_reads_the_final_data_block(self):
        report = _report("DE_RATIO: 9") + "\nRevised:\n" + _report("DE_RATIO: 0.08")
        assert extract_debt_to_equity(report) == pytest.approx(8.0)

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
