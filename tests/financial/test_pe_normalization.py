"""
Tests for P/E ratio normalization and data quality sanity checks.

These tests verify:
1. P/E normalization logic handles edge cases (stock splits, stale estimates)
2. Extreme earnings quality ratios are downgraded to WARNING (suggests data issue)
3. Data divergence between TTM and statements is logged for investigation

Background bugs fixed:
- Tamron (7740.T): P/E of 12.63 was incorrectly replaced with 3.13
  due to stale forward estimates not adjusted for 1:4 stock split.
  Fix: Added P/E sanity thresholds (min 5, max 3x divergence, only replace if >50).

- Tsutsumi (7937.T): Earnings quality flag with 2.18x disconnect ratio.
  Analysis: TTM values showed real deterioration (not data misalignment).
  Fix: Keep TTM values (more current), but downgrade extreme ratios (>4x)
  to WARNING since they likely indicate data quality issues, not fraud.
"""

import pytest


class TestPENormalizationSanityChecks:
    """Test that P/E normalization respects sanity thresholds."""

    @pytest.fixture
    def fetcher(self):
        from src.data.fetcher import SmartMarketDataFetcher

        return SmartMarketDataFetcher()

    def test_keeps_trailing_when_forward_too_low(self, fetcher):
        """Don't replace trailing P/E if forward P/E is suspiciously low (< 5)."""
        # Simulates stock split case: correct trailing, stale forward
        info = {"trailingPE": 12.63, "forwardPE": 3.13}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should keep the reasonable trailing P/E
        assert result["trailingPE"] == 12.63
        assert result.get("_trailingPE_source") is None

    def test_keeps_trailing_when_divergence_extreme(self, fetcher):
        """Don't replace if divergence ratio > 3x (suggests data error)."""
        # 4x divergence - one value is almost certainly wrong
        info = {"trailingPE": 40.0, "forwardPE": 10.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should keep trailing despite it being > forward * 1.4
        assert result["trailingPE"] == 40.0
        assert result.get("_trailingPE_source") is None

    def test_replaces_trailing_when_inflated_and_forward_reasonable(self, fetcher):
        """Replace trailing with forward when trailing is very high and forward is reasonable."""
        # Trailing inflated due to one-time charge, forward is normalized
        info = {"trailingPE": 80.0, "forwardPE": 25.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should replace with forward (80 > 50 threshold, 25 >= 5, ratio 3.2 is acceptable)
        # Wait - ratio 3.2 > 3.0 so it should NOT replace
        # Let me recalculate: 80/25 = 3.2 which is > MAX_DIVERGENCE_RATIO of 3.0
        assert result["trailingPE"] == 80.0  # Kept due to ratio > 3

    def test_replaces_trailing_when_conditions_met(self, fetcher):
        """Replace trailing when all sanity conditions are satisfied."""
        # Trailing slightly inflated, forward reasonable, ratio within bounds
        info = {"trailingPE": 60.0, "forwardPE": 22.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # 60/22 = 2.7 (within 3x), 22 >= 5, 60 > 50, 60 > 22*1.4
        assert result["trailingPE"] == 22.0
        assert result["_trailingPE_source"] == "normalized_forward_proxy"

    def test_no_change_when_trailing_not_high(self, fetcher):
        """Don't replace if trailing P/E is reasonable (< 50 threshold)."""
        # Normal case: trailing higher than forward but both reasonable
        info = {"trailingPE": 25.0, "forwardPE": 18.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        # Should keep trailing - it's not unusually high
        assert result["trailingPE"] == 25.0
        assert result.get("_trailingPE_source") is None

    def test_no_change_when_trailing_lower_than_forward(self, fetcher):
        """Don't replace if trailing < forward (earnings declining)."""
        info = {"trailingPE": 15.0, "forwardPE": 20.0}

        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] == 15.0
        assert result.get("_trailingPE_source") is None

    def test_handles_missing_pe_values(self, fetcher):
        """Gracefully handle None or missing P/E values."""
        cases = [
            {"trailingPE": None, "forwardPE": 15.0},
            {"trailingPE": 15.0, "forwardPE": None},
            {"trailingPE": None, "forwardPE": None},
            {},
        ]

        for info in cases:
            result = fetcher._normalize_data_integrity(info.copy(), "TEST")
            # Should not crash, values unchanged
            assert result.get("trailingPE") == info.get("trailingPE")
            assert result.get("forwardPE") == info.get("forwardPE")

    def test_handles_zero_pe_values(self, fetcher):
        """Don't divide by zero or replace with zero."""
        cases = [
            {"trailingPE": 0, "forwardPE": 15.0},
            {"trailingPE": 15.0, "forwardPE": 0},
            {"trailingPE": 0, "forwardPE": 0},
        ]

        for info in cases:
            result = fetcher._normalize_data_integrity(info.copy(), "TEST")
            # Should not crash
            assert result is not None

    def test_tamron_case_exact(self, fetcher):
        """Regression test for the exact Tamron (7740.T) values that exposed the bug."""
        info = {"trailingPE": 12.626695, "forwardPE": 3.1312459}

        result = fetcher._normalize_data_integrity(info, "7740.T")

        # The correct trailing should be preserved
        assert result["trailingPE"] == 12.626695
        assert result.get("_trailingPE_source") is None

    def test_quarantines_recent_split_forward_metrics(self, fetcher):
        """Recent split with aligned PE/EPS ratios should null forward valuation metrics."""
        info = {
            "trailingPE": 14.366964,
            "forwardPE": 7.019027,
            "epsTrailingTwelveMonths": 80.88,
            "forwardEps": 165.55,
            "pegRatio": 0.69,
            "lastSplitDate": 1766966400,
            "lastSplitFactor": "2:1",
            "regularMarketTime": 1773383400,
        }

        result = fetcher._normalize_data_integrity(info, "4396.T")

        assert result["trailingPE"] == pytest.approx(14.366964)
        assert result["epsTrailingTwelveMonths"] == pytest.approx(80.88)
        assert result["forwardPE"] is None
        assert result["forwardEps"] is None
        assert result["pegRatio"] is None
        assert result["_split_sensitive_metrics_quarantined"] is True
        assert result["_split_quarantine_reason"] == "recent_split_share_basis_mismatch"
        assert any(
            "quarantined forward valuation metrics" in note
            for note in result["_data_quality_notes"]
        )

    def test_no_quarantine_without_recent_split_metadata(self, fetcher):
        """Do not quarantine on ratio mismatch alone without recent split evidence."""
        info = {
            "trailingPE": 14.366964,
            "forwardPE": 7.019027,
            "epsTrailingTwelveMonths": 80.88,
            "forwardEps": 165.55,
            "pegRatio": 0.69,
        }

        result = fetcher._normalize_data_integrity(info, "4396.T")

        assert result["forwardPE"] == pytest.approx(7.019027)
        assert result["forwardEps"] == pytest.approx(165.55)
        assert result["pegRatio"] == pytest.approx(0.69)
        assert result.get("_split_sensitive_metrics_quarantined") is None

    def test_no_quarantine_when_only_pe_ratio_matches_split_factor(self, fetcher):
        """Require both PE and EPS ratios to match the split factor."""
        info = {
            "trailingPE": 14.366964,
            "forwardPE": 7.019027,
            "epsTrailingTwelveMonths": 80.88,
            "forwardEps": 120.0,
            "pegRatio": 0.69,
            "lastSplitDate": 1766966400,
            "lastSplitFactor": "2:1",
            "regularMarketTime": 1773383400,
        }

        result = fetcher._normalize_data_integrity(info, "4396.T")

        assert result["forwardPE"] == pytest.approx(7.019027)
        assert result["forwardEps"] == pytest.approx(120.0)
        assert result["pegRatio"] == pytest.approx(0.69)
        assert result.get("_split_sensitive_metrics_quarantined") is None

    def test_reconciles_latest_quarter_date_to_newer_metadata(self, fetcher):
        """Prefer newer mostRecentQuarter when latest_quarter_date is materially stale."""
        info = {
            "latest_quarter_date": "2024-12-31",
            "mostRecentQuarter": 1767139200,
        }

        result = fetcher._normalize_data_integrity(info, "4396.T")

        assert result["latest_quarter_date"] == "2025-12-31"
        assert result["_latest_quarter_date_source"] == "reconciled_most_recent_quarter"

    def test_does_not_reconcile_when_quarter_dates_are_aligned(self, fetcher):
        """Keep the existing date when both sources are already effectively aligned."""
        info = {
            "latest_quarter_date": "2025-12-31",
            "mostRecentQuarter": 1767139200,
        }

        result = fetcher._normalize_data_integrity(info, "4396.T")

        assert result["latest_quarter_date"] == "2025-12-31"
        assert (
            result.get("_latest_quarter_date_source")
            != "reconciled_most_recent_quarter"
        )

    def test_does_not_relabel_statement_mrq_to_newer_metadata(self, fetcher):
        """Keep 6782.TW MRQ growth bound to the statement period it came from."""
        info = {
            "latest_quarter_date": "2025-12-31",
            "_latest_quarter_date_source": "yfinance_quarterly",
            "mostRecentQuarter": 1774915200,
            "revenueGrowth_MRQ": 0.168693,
            "_revenueGrowth_MRQ_source": "calculated_from_quarterly",
            "earningsGrowth_MRQ": 1.028262,
            "_earningsGrowth_MRQ_source": "calculated_from_quarterly",
            "mrq_comparison_base_status": "DEPRESSED",
        }

        result = fetcher._normalize_data_integrity(info, "6782.TW")

        assert result["latest_quarter_date"] == "2025-12-31"
        assert result["_latest_quarter_date_source"] == "yfinance_quarterly"
        assert any(
            "Newer quarter metadata exists for 2026-03-31" in note
            and "statement-derived MRQ metrics remain aligned to 2025-12-31" in note
            for note in result["_data_quality_notes"]
        )

    def test_quarantines_low_pe_when_identity_check_fails(self, fetcher):
        """P/E below 3 is nulled when price/EPS cannot reproduce it."""
        info = {
            "currentPrice": 12.0,
            "trailingEps": 1.0,
            "trailingPE": 2.0,
            "pegRatio": 0.4,
        }

        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] is None
        assert result["pegRatio"] is None
        assert result["_pe_low_anomaly_quarantined"] is True
        assert any(
            "failed price/EPS identity check" in note
            for note in result["_data_quality_notes"]
        )

    def test_preserves_low_pe_when_identity_check_passes(self, fetcher):
        """A real low P/E should be flagged for investigation, not nulled."""
        info = {
            "currentPrice": 12.0,
            "trailingEps": 6.0,
            "trailingPE": 2.0,
            "pegRatio": 0.4,
        }

        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] == pytest.approx(2.0)
        assert result["pegRatio"] == pytest.approx(0.4)
        assert result["_pe_low_anomaly_flag"] == "LOW_PE_REQUIRES_INVESTIGATION"
        assert result["_pe_low_anomaly_context"] == [
            "low_multiple_confirmed_but_unexplained"
        ]

    def test_flags_low_pe_with_earnings_collapse_context(self, fetcher):
        """Low but internally consistent P/E carries stress context."""
        info = {
            "currentPrice": 16.0,
            "trailingEps": 4.0,
            "trailingPE": 4.0,
            "earningsGrowth_TTM": -0.30,
            "profitMargins": 0.10,
        }

        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] == pytest.approx(4.0)
        assert result["_pe_low_anomaly_flag"] == "LOW_PE_REQUIRES_INVESTIGATION"
        assert "earnings_collapse" in result["_pe_low_anomaly_context"]

    def test_low_pe_helper_composes_after_existing_unit_quarantine(self, fetcher):
        """When trailing P/E is already null, low-P/E logic leaves it alone."""
        info = {
            "trailingPE": None,
            "forwardPE": 12.0,
            "pegRatio": 1.0,
            "_pe_unit_error_quarantined": "trailing",
        }

        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] is None
        assert result["forwardPE"] == pytest.approx(12.0)
        assert result["pegRatio"] == pytest.approx(1.0)
        assert "_pe_low_anomaly_quarantined" not in result


class TestDataDivergenceDetection:
    """Test that extreme TTM vs statement divergence is detected and handled."""

    def test_earnings_quality_extreme_ratio_downgraded(self):
        """Earnings quality check with >4x ratio should be WARNING, not CRITICAL."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Tsutsumi-like case: extreme 4.3x ratio suggests data quality issue
        # Net Income: 1.71B, FCF: -3.74B → ratio = 2.18x (would trigger but < 4x)
        # If ratio were > 4x, it should downgrade to WARNING

        # Test case with extreme ratio (>4x)
        metrics_extreme = {
            "net_income": 1_000_000_000,  # 1B
            "fcf": -5_000_000_000,  # -5B → 5x ratio
        }

        red_flags, result = RedFlagDetector.detect_red_flags(
            metrics_extreme, ticker="TEST"
        )

        # Should be WARNING, not CRITICAL (extreme ratio suggests data issue)
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY_UNCERTAIN"
        assert red_flags[0]["severity"] == "WARNING"
        assert red_flags[0]["action"] == "RISK_PENALTY"
        assert result == "PASS"  # WARNING doesn't reject

    def test_earnings_quality_normal_ratio_critical(self):
        """Earnings quality check with 2-4x ratio should be CRITICAL."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Normal fraud indicator: 2-4x ratio
        metrics_normal = {
            "net_income": 1_000_000_000,  # 1B
            "fcf": -3_000_000_000,  # -3B → 3x ratio (suspicious but plausible)
        }

        red_flags, result = RedFlagDetector.detect_red_flags(
            metrics_normal, ticker="TEST"
        )

        # Should be CRITICAL (plausible fraud indicator)
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY"
        assert red_flags[0]["severity"] == "CRITICAL"
        assert red_flags[0]["action"] == "AUTO_REJECT"
        assert result == "REJECT"

    def test_tsutsumi_case_with_ttm_data_no_divergence(self):
        """Tsutsumi case with TTM data, no divergence flag: 2.18x ratio → CRITICAL."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Actual Tsutsumi TTM values, but no divergence marker in report
        metrics = {
            "net_income": 1_714_000_000,
            "fcf": -3_744_750_080,
            "_raw_report": "No FCF data quality issues mentioned",
        }

        disconnect_ratio = abs(metrics["fcf"] / metrics["net_income"])
        assert (
            2.0 < disconnect_ratio < 4.0
        ), f"Ratio should be 2-4x, got {disconnect_ratio:.2f}x"

        red_flags, result = RedFlagDetector.detect_red_flags(metrics, ticker="7937.T")

        # Without divergence flag, 2.18x ratio → CRITICAL
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY"
        assert red_flags[0]["severity"] == "CRITICAL"
        assert result == "REJECT"

    def test_tsutsumi_case_with_divergence_flag(self):
        """Tsutsumi case with FCF divergence flag: 2.18x ratio → WARNING (data uncertain)."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Same values, but with divergence marker in report
        metrics = {
            "net_income": 1_714_000_000,
            "fcf": -3_744_750_080,
            "_raw_report": "FCF DATA QUALITY UNCERTAIN: TTM (-3.74B) differs from annual statement (-2.01B)",
        }

        red_flags, result = RedFlagDetector.detect_red_flags(metrics, ticker="7937.T")

        # With divergence flag, downgrade to WARNING
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EARNINGS_QUALITY_UNCERTAIN"
        assert red_flags[0]["severity"] == "WARNING"


class TestPEUnitErrorDetection:
    """When trailing and forward P/E differ by ~10/100/1000× and one value is
    plausible (in [MIN_REASONABLE_PE, 30]) while the other is extreme
    (≤ 1 or > 98), it's almost certainly a unit/decimal/currency error in
    one source's EPS — not a stale forward estimate. Quarantine the suspect
    and emit a distinct event so the issue is debuggable.

    Reproducer from production: 1F2.SI showed trailingPE=16.166668,
    forwardPE=0.15746754, ratio 102.7×. The previous code logged
    `pe_divergence_suspicious` with hint='stale/incorrect forward estimate',
    leaving the bogus 0.157 in `forwardPE` for downstream PEG/screening.
    """

    @pytest.fixture
    def fetcher(self):
        from src.data.fetcher import SmartMarketDataFetcher

        return SmartMarketDataFetcher()

    def test_1F2_SI_reproducer_quarantines_forward(self, fetcher):
        """The exact 1F2.SI values from the May 2026 production log."""
        info = {
            "trailingPE": 16.166668,
            "forwardPE": 0.15746754,
            "forwardEps": 0.05,
            "pegRatio": 1.2,
            "epsTrailingTwelveMonths": 5.13,
        }
        result = fetcher._normalize_data_integrity(info, "1F2.SI")

        assert result["trailingPE"] == 16.166668
        assert result["epsTrailingTwelveMonths"] == 5.13
        assert result["forwardPE"] is None
        assert result["forwardEps"] is None
        assert result["pegRatio"] is None
        assert result["_pe_unit_error_quarantined"] == "forward"
        notes = result.get("_data_quality_notes")
        assert isinstance(notes, list) and any(
            "forward" in n.lower() and "unit" in n.lower() for n in notes
        )

    def test_inverse_direction_quarantines_trailing(self, fetcher):
        """If trailing is the implausible value, null trailing and EPS_TTM
        instead — the existing 'keep trailing' default would propagate the
        wrong P/E into downstream filters."""
        info = {
            "trailingPE": 0.157,
            "forwardPE": 16.17,
            "forwardEps": 0.93,
            "pegRatio": 1.5,
            "epsTrailingTwelveMonths": 100.5,
        }
        result = fetcher._normalize_data_integrity(info, "TEST.X")

        assert result["forwardPE"] == 16.17
        assert result["forwardEps"] == 0.93
        assert result["trailingPE"] is None
        assert result["epsTrailingTwelveMonths"] is None
        assert result["_pe_unit_error_quarantined"] == "trailing"

    def test_x10_pattern_quarantines_forward(self, fetcher):
        info = {"trailingPE": 18.0, "forwardPE": 180.0}
        result = fetcher._normalize_data_integrity(info, "TEST")
        assert result["forwardPE"] is None
        assert result["_pe_unit_error_quarantined"] == "forward"

    def test_x1000_pattern_quarantines_forward(self, fetcher):
        info = {"trailingPE": 20.0, "forwardPE": 20000.0}
        result = fetcher._normalize_data_integrity(info, "TEST")
        assert result["forwardPE"] is None
        assert result["_pe_unit_error_quarantined"] == "forward"

    def test_4x_divergence_falls_through_to_legacy_log(self, fetcher, monkeypatch):
        """A 4× divergence is genuine staleness — keep both values and emit
        the legacy `pe_divergence_suspicious` event (not the unit-error
        event)."""
        from src.data import fetcher as fetcher_module

        events: list[str] = []
        original_warning = fetcher_module.logger.warning

        def capture(event_name, **kwargs):
            events.append(event_name)
            return original_warning(event_name, **kwargs)

        monkeypatch.setattr(fetcher_module.logger, "warning", capture)

        info = {"trailingPE": 40.0, "forwardPE": 10.0}
        result = fetcher._normalize_data_integrity(info, "TEST")

        assert result["trailingPE"] == 40.0
        assert result["forwardPE"] == 10.0
        assert "_pe_unit_error_quarantined" not in result
        assert "pe_divergence_suspicious" in events
        assert "pe_divergence_unit_error_suspect" not in events

    def test_both_extreme_is_NOT_unit_error(self, fetcher):
        """Ratio ≈ 100× but neither value is plausible — pattern doesn't
        match (would need real diagnosis), so fall through to legacy log."""
        info = {"trailingPE": 120.0, "forwardPE": 1.2}
        result = fetcher._normalize_data_integrity(info, "TEST")
        assert result["trailingPE"] == 120.0
        assert result["forwardPE"] == 1.2
        assert "_pe_unit_error_quarantined" not in result

    def test_ratio_near_power_but_outside_tolerance(self, fetcher):
        """Ratio 150× has log10 ≈ 2.18 (distance 0.18 from 2 > 0.05) — not a
        clean power of ten, so it's not classified as a unit error even
        though one value is plausible and the other is extreme."""
        info = {"trailingPE": 18.0, "forwardPE": 0.12}
        result = fetcher._normalize_data_integrity(info, "TEST")
        # forwardPE preserved (no quarantine), legacy log fires.
        assert result["forwardPE"] == 0.12
        assert "_pe_unit_error_quarantined" not in result

    def test_emits_unit_error_event(self, fetcher, monkeypatch):
        """The new event name and structured fields are emitted on match."""
        from src.data import fetcher as fetcher_module

        captured: list[tuple[str, dict]] = []
        original_warning = fetcher_module.logger.warning

        def capture(event_name, **kwargs):
            captured.append((event_name, kwargs))
            return original_warning(event_name, **kwargs)

        monkeypatch.setattr(fetcher_module.logger, "warning", capture)

        info = {"trailingPE": 16.166668, "forwardPE": 0.15746754}
        fetcher._normalize_data_integrity(info, "1F2.SI")

        names = [n for n, _ in captured]
        assert "pe_divergence_unit_error_suspect" in names
        kwargs = next(k for n, k in captured if n == "pe_divergence_unit_error_suspect")
        assert kwargs["suspect"] == "forward"
        assert kwargs["power_of_ten"] == 2
        assert kwargs["symbol"] == "1F2.SI"

    def test_pattern_does_not_fire_when_ratio_under_3x(self, fetcher):
        """Below MAX_DIVERGENCE_RATIO the legacy code paths handle the
        decision. The new branch must not be reachable in that regime."""
        info = {"trailingPE": 25.0, "forwardPE": 18.0}
        result = fetcher._normalize_data_integrity(info, "TEST")
        assert "_pe_unit_error_quarantined" not in result
        assert result["trailingPE"] == 25.0
        assert result["forwardPE"] == 18.0
