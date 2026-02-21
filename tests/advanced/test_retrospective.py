"""
Tests for the Lessons Learned / Retrospective System.

Tests snapshot extraction, comparison logic, confidence weighting,
lesson generation, storage deduplication, and prompt injection formatting.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrospective import (
    EXCHANGE_BENCHMARK,
    EXCHANGE_CURRENCY,
    FAILURE_MODES,
    FALLBACK_BENCHMARK,
    FALLBACK_CURRENCY,
    LESSON_TYPES,
    LESSONS_COLLECTION_NAME,
    MAX_LESSONS_PER_TICKER,
    MINIMUM_DAYS_ELAPSED,
    _extract_bear_risks,
    _extract_data_block_field,
    _extract_data_block_float,
    _get_ticker_suffix,
    _lesson_already_processed,
    compare_to_reality,
    compute_confidence,
    create_lessons_memory,
    extract_snapshot,
    format_lessons_for_injection,
    generate_lesson,
    load_past_snapshots,
    run_retrospective,
    store_lesson,
)

# ══════════════════════════════════════════════════════════════════════════════
# Test Helpers
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_DATA_BLOCK = """
Some analyst commentary here.

### --- START DATA_BLOCK (INTERNAL SCORING — NOT THIRD-PARTY RATINGS) ---
TICKER: 2767.T
COMPANY: TSUBURAYA FIELDS HOLDINGS
SECTOR: Consumer Cyclical
CURRENT_PRICE: 1774
PE_RATIO_TTM: 8.5
PEG_RATIO: 0.03
PB_RATIO: 1.12
ANALYST_COVERAGE_ENGLISH: 4
ANALYST_COVERAGE_TOTAL_EST: 12
PROFITABILITY_TREND: UNSTABLE
52W_HIGH: 2100
52W_LOW: 1200
ADJUSTED_HEALTH_SCORE: 58%
GROWTH_SCORE: 55%
### --- END DATA_BLOCK ---
"""

SAMPLE_PM_BLOCK = """
Based on my analysis...

### --- START PM_BLOCK ---
VERDICT: BUY
HEALTH_ADJ: 62
GROWTH_ADJ: 58
RISK_TALLY: 1.33
ZONE: MODERATE
POSITION_SIZE: 3.0
SHOW_VALUATION_CHART: YES
VALUATION_DISCOUNT: 0.9
VALUATION_CONTEXT: CONTEXTUAL_PASS
### --- END PM_BLOCK ---
"""


def _make_result(
    fundamentals_report: str = SAMPLE_DATA_BLOCK,
    final_trade_decision: str = SAMPLE_PM_BLOCK,
    bear_history: str = "KEY RISKS: Cyclical peak, single-segment concentration.",
) -> dict:
    """Create a mock analysis result dict."""
    return {
        "fundamentals_report": fundamentals_report,
        "final_trade_decision": final_trade_decision,
        "investment_debate_state": {
            "bear_history": bear_history,
            "bull_history": "Strong value metrics.",
        },
        "market_report": "Market trending up.",
        "sentiment_report": "Positive sentiment.",
        "news_report": "No major news.",
    }


def _make_snapshot(**overrides) -> dict:
    """Create a mock prediction snapshot."""
    base = {
        "verdict": "BUY",
        "health_adj": 62,
        "growth_adj": 58,
        "risk_tally": 1.33,
        "zone": "MODERATE",
        "position_size": 3.0,
        "current_price": 1774.0,
        "sector": "Consumer Cyclical",
        "pe_ratio": 8.5,
        "peg_ratio": 0.03,
        "pb_ratio": 1.12,
        "analyst_coverage": 4.0,
        "profitability_trend": "UNSTABLE",
        "52w_high": 2100.0,
        "52w_low": 1200.0,
        "bear_risks_excerpt": "KEY RISKS: Cyclical peak, single-segment concentration.",
        "exchange": "T",
        "currency": "JPY",
        "benchmark_index": "^N225",
        "fx_rate_to_usd": 0.0067,
        "ticker": "2767.T",
        "analysis_date": (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
        "deep_model": "gemini-3-pro-preview",
        "quick_model": "gemini-2.5-flash",
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Component 1: Snapshot Extraction Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractSnapshot:
    """Test prediction snapshot extraction from analysis results."""

    def test_extract_snapshot_from_real_data(self):
        """Parse actual DATA_BLOCK/PM_BLOCK text."""
        result = _make_result()
        snapshot = extract_snapshot(result, "2767.T")

        assert snapshot["ticker"] == "2767.T"
        assert snapshot["verdict"] == "BUY"
        assert snapshot["health_adj"] == 62
        assert snapshot["growth_adj"] == 58
        assert snapshot["risk_tally"] == 1.33
        assert snapshot["zone"] == "MODERATE"
        assert snapshot["position_size"] == 3.0
        assert snapshot["current_price"] == 1774.0
        assert snapshot["sector"] == "Consumer Cyclical"
        assert snapshot["pe_ratio"] == 8.5
        assert snapshot["peg_ratio"] == 0.03
        assert snapshot["pb_ratio"] == 1.12
        assert snapshot["analyst_coverage"] == 4.0
        assert snapshot["profitability_trend"] == "UNSTABLE"
        assert snapshot["exchange"] == "T"
        assert snapshot["currency"] == "JPY"
        assert snapshot["benchmark_index"] == "^N225"
        assert snapshot["analysis_date"] == datetime.now().strftime("%Y-%m-%d")
        assert "bear_risks_excerpt" in snapshot

    def test_extract_snapshot_missing_blocks(self):
        """Snapshot with None fields when DATA_BLOCK/PM_BLOCK absent."""
        result = {
            "fundamentals_report": "No data block here.",
            "final_trade_decision": "I recommend BUY.",
            "investment_debate_state": {},
        }
        snapshot = extract_snapshot(result, "0005.HK")

        assert snapshot["ticker"] == "0005.HK"
        assert snapshot["exchange"] == "HK"
        assert snapshot["currency"] == "HKD"
        assert snapshot["benchmark_index"] == "^HSI"
        # PM_BLOCK fields should be None
        assert snapshot["health_adj"] is None
        assert snapshot["growth_adj"] is None
        # DATA_BLOCK fields should be None
        assert snapshot["current_price"] is None
        assert snapshot["sector"] is None
        assert snapshot["pe_ratio"] is None

    def test_extract_snapshot_empty_result(self):
        """Snapshot from empty result dict."""
        snapshot = extract_snapshot({}, "AAPL")
        assert snapshot["ticker"] == "AAPL"
        assert snapshot["verdict"] is None
        assert snapshot["exchange"] == "US"
        assert snapshot["currency"] == "USD"
        assert snapshot["benchmark_index"] == "^GSPC"


# ══════════════════════════════════════════════════════════════════════════════
# DATA_BLOCK Field Extraction Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDataBlockExtraction:
    """Test DATA_BLOCK field extraction helpers."""

    def test_extract_field(self):
        assert (
            _extract_data_block_field(SAMPLE_DATA_BLOCK, "SECTOR")
            == "Consumer Cyclical"
        )
        assert _extract_data_block_field(SAMPLE_DATA_BLOCK, "TICKER") == "2767.T"
        assert (
            _extract_data_block_field(SAMPLE_DATA_BLOCK, "PROFITABILITY_TREND")
            == "UNSTABLE"
        )

    def test_extract_float(self):
        assert _extract_data_block_float(SAMPLE_DATA_BLOCK, "CURRENT_PRICE") == 1774.0
        assert _extract_data_block_float(SAMPLE_DATA_BLOCK, "PE_RATIO_TTM") == 8.5
        assert _extract_data_block_float(SAMPLE_DATA_BLOCK, "PEG_RATIO") == 0.03

    def test_extract_field_na(self):
        data_block = """### --- START DATA_BLOCK ---
SECTOR: N/A
### --- END DATA_BLOCK ---"""
        assert _extract_data_block_field(data_block, "SECTOR") is None

    def test_extract_field_missing(self):
        assert _extract_data_block_field(SAMPLE_DATA_BLOCK, "NONEXISTENT") is None
        assert _extract_data_block_float(SAMPLE_DATA_BLOCK, "NONEXISTENT") is None

    def test_extract_from_empty(self):
        assert _extract_data_block_field("", "SECTOR") is None
        assert _extract_data_block_field(None, "SECTOR") is None
        assert _extract_data_block_float("", "PRICE") is None

    def test_extract_adjusted_health_score_with_percent(self):
        """ADJUSTED_HEALTH_SCORE has % sign — float extraction handles it."""
        val = _extract_data_block_float(SAMPLE_DATA_BLOCK, "ADJUSTED_HEALTH_SCORE")
        assert val == 58.0


# ══════════════════════════════════════════════════════════════════════════════
# Bear Risks Extraction Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBearRisksExtraction:
    def test_extract_key_risks(self):
        result = {
            "investment_debate_state": {
                "bear_history": "Intro.\nKEY RISKS: Cyclical peak, FX risk.\n\nConclusion."
            }
        }
        excerpt = _extract_bear_risks(result)
        assert "KEY RISKS" in excerpt
        assert "Cyclical peak" in excerpt

    def test_extract_fallback_round1(self):
        result = {
            "investment_debate_state": {
                "bear_history": "",
                "bear_round1": "Bear argument text about downside risks.",
            }
        }
        excerpt = _extract_bear_risks(result)
        assert "downside risks" in excerpt

    def test_extract_empty(self):
        assert _extract_bear_risks({}) == ""
        assert _extract_bear_risks({"investment_debate_state": {}}) == ""

    def test_truncation_at_500(self):
        result = {
            "investment_debate_state": {
                "bear_history": "KEY RISKS: " + "x" * 600,
            }
        }
        excerpt = _extract_bear_risks(result)
        assert len(excerpt) <= 500


# ══════════════════════════════════════════════════════════════════════════════
# Exchange/Currency Mapping Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExchangeBenchmarkMapping:
    """Verify all ticker suffixes map correctly."""

    def test_all_exchange_benchmarks(self):
        expected = {
            ".T": "^N225",
            ".HK": "^HSI",
            ".TW": "^TWII",
            ".KS": "^KS11",
            ".AS": "^AEX",
            ".DE": "^GDAXI",
            ".L": "^FTSE",
            ".PA": "^FCHI",
            ".TO": "^GSPTSE",
            ".AX": "^AXJO",
            ".SI": "^STI",
        }
        for suffix, benchmark in expected.items():
            assert EXCHANGE_BENCHMARK.get(suffix) == benchmark, f"Failed for {suffix}"

    def test_fallback_benchmark(self):
        assert EXCHANGE_BENCHMARK.get(".XX") is None
        assert FALLBACK_BENCHMARK == "^GSPC"

    def test_all_exchange_currencies(self):
        expected = {
            ".T": "JPY",
            ".HK": "HKD",
            ".TW": "TWD",
            ".KS": "KRW",
            ".AS": "EUR",
            ".DE": "EUR",
            ".L": "GBP",
        }
        for suffix, currency in expected.items():
            assert EXCHANGE_CURRENCY.get(suffix) == currency, f"Failed for {suffix}"

    def test_ticker_suffix_extraction(self):
        assert _get_ticker_suffix("7203.T") == ".T"
        assert _get_ticker_suffix("0005.HK") == ".HK"
        assert _get_ticker_suffix("AAPL") == ""
        assert _get_ticker_suffix("BRK.B") == ".B"


# ══════════════════════════════════════════════════════════════════════════════
# Confidence Weighting Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConfidenceWeighting:
    """Test confidence computation: temporal × model_quality × mode × signal_strength."""

    def test_optimal_case(self):
        """180 days, gemini-3-pro, normal mode, 32% excess."""
        comparison = _make_snapshot(
            days_elapsed=180,
            excess_return_pct=-32.0,
            deep_model="gemini-3-pro-preview",
            quick_model="gemini-2.5-flash",
        )
        conf = compute_confidence(comparison)
        # temporal=1.0, model=1.0, mode=1.0, signal=min(32/30,1)=1.0
        assert conf == 1.0

    def test_too_early(self):
        """15 days elapsed — below minimum, won't be called in practice but test weight."""
        comparison = _make_snapshot(
            days_elapsed=15,
            excess_return_pct=-20.0,
            deep_model="gemini-3-pro-preview",
            quick_model="gemini-2.5-flash",
        )
        conf = compute_confidence(comparison)
        # temporal=0.3, model=1.0, mode=1.0, signal=0.67
        assert conf == pytest.approx(0.3 * 1.0 * 1.0 * (20 / 30), abs=0.01)

    def test_stale_prediction(self):
        """600 days — stale."""
        comparison = _make_snapshot(
            days_elapsed=600,
            excess_return_pct=-25.0,
            deep_model="gemini-3-pro-preview",
            quick_model="gemini-2.5-flash",
        )
        conf = compute_confidence(comparison)
        # temporal=0.3, model=1.0, mode=1.0, signal=0.833
        assert conf == pytest.approx(0.3 * 1.0 * 1.0 * (25 / 30), abs=0.01)

    def test_weak_model_quick_mode(self):
        """Flash model in quick mode (same model for quick and deep)."""
        comparison = _make_snapshot(
            days_elapsed=180,
            excess_return_pct=-18.0,
            deep_model="gemini-2.0-flash",
            quick_model="gemini-2.0-flash",
        )
        conf = compute_confidence(comparison)
        # temporal=1.0, model=0.6, mode=0.7 (quick=deep), signal=0.6
        assert conf == pytest.approx(1.0 * 0.6 * 0.7 * 0.6, abs=0.01)

    def test_unknown_model(self):
        """Unknown model defaults to 0.5."""
        comparison = _make_snapshot(
            days_elapsed=180,
            excess_return_pct=-30.0,
            deep_model="some-new-model-v42",
            quick_model="other-model",
        )
        conf = compute_confidence(comparison)
        # temporal=1.0, model=0.5, mode=1.0, signal=1.0
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_small_signal(self):
        """Small excess return -> proportionally smaller signal component."""
        comparison = _make_snapshot(
            days_elapsed=180,
            excess_return_pct=-16.0,
            deep_model="gemini-3-pro-preview",
            quick_model="gemini-2.5-flash",
        )
        conf = compute_confidence(comparison)
        # signal = 16/30 ≈ 0.533
        assert conf == pytest.approx(16.0 / 30.0, abs=0.01)

    def test_temporal_brackets(self):
        """Verify each temporal bracket."""
        base = _make_snapshot(
            excess_return_pct=-30.0,
            deep_model="gemini-3-pro-preview",
            quick_model="gemini-2.5-flash",
        )
        # 0-30: 0.3
        base["days_elapsed"] = 25
        assert compute_confidence(base) == pytest.approx(0.3, abs=0.01)

        # 31-90: 0.7
        base["days_elapsed"] = 60
        assert compute_confidence(base) == pytest.approx(0.7, abs=0.01)

        # 91-270: 1.0
        base["days_elapsed"] = 200
        assert compute_confidence(base) == pytest.approx(1.0, abs=0.01)

        # 271-540: 0.7
        base["days_elapsed"] = 400
        assert compute_confidence(base) == pytest.approx(0.7, abs=0.01)

        # 541+: 0.3
        base["days_elapsed"] = 700
        assert compute_confidence(base) == pytest.approx(0.3, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison Threshold Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCompareToReality:
    """Test comparison logic and threshold detection."""

    @pytest.mark.asyncio
    async def test_too_recent_skipped(self):
        """Snapshots < 30 days old return None."""
        snapshot = _make_snapshot(
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
        )
        result = await compare_to_reality(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_ticker(self):
        snapshot = _make_snapshot(ticker=None)
        result = await compare_to_reality(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_verdict(self):
        snapshot = _make_snapshot(verdict=None)
        result = await compare_to_reality(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_date(self):
        snapshot = _make_snapshot(analysis_date="not-a-date")
        result = await compare_to_reality(snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_yfinance_failure_returns_none(self):
        """Graceful None return when yfinance raises."""
        import asyncio as real_asyncio

        snapshot = _make_snapshot(
            analysis_date=(datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d"),
        )

        # Patch asyncio.wait_for inside the module
        original_wait_for = real_asyncio.wait_for

        async def failing_wait_for(coro, timeout):
            # Cancel the coroutine to avoid RuntimeWarning
            if hasattr(coro, "close"):
                coro.close()
            raise Exception("yfinance down")

        with patch("asyncio.wait_for", side_effect=failing_wait_for):
            result = await compare_to_reality(snapshot)
            assert result is None

    @pytest.mark.asyncio
    async def test_below_threshold_returns_none(self):
        """Small excess return below trigger threshold returns None."""
        snapshot = _make_snapshot(
            analysis_date=(datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d"),
        )

        # Mock asyncio.to_thread to run synchronously with our mock data
        # This bypasses the thread pool where patches don't propagate
        mock_data = {
            "start_adj_close": 1774.0,
            "end_adj_close": 1800.0,  # +1.5%
            "bench_start": 30000.0,
            "bench_end": 30300.0,  # +1.0%
        }

        async def mock_to_thread(func, *args, **kwargs):
            return mock_data

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            result = await compare_to_reality(snapshot)
            assert result is None  # +0.5% excess is below 15% BUY threshold


# ══════════════════════════════════════════════════════════════════════════════
# Load Snapshots Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadPastSnapshots:
    """Test loading snapshots from analysis JSON files."""

    def test_nonexistent_dir(self, tmp_path):
        result = load_past_snapshots("2767.T", tmp_path / "nonexistent")
        assert result == {}

    def test_no_json_files(self, tmp_path):
        result = load_past_snapshots("2767.T", tmp_path)
        assert result == {}

    def test_json_without_snapshot(self, tmp_path):
        """Old analysis files without prediction_snapshot are skipped."""
        filepath = tmp_path / "2767_T_20251202_184443_analysis.json"
        filepath.write_text(json.dumps({"metadata": {"ticker": "2767.T"}}))
        result = load_past_snapshots("2767.T", tmp_path)
        assert result == {}

    def test_json_with_snapshot(self, tmp_path):
        """Files with prediction_snapshot are loaded."""
        snapshot = _make_snapshot()
        data = {"metadata": {"ticker": "2767.T"}, "prediction_snapshot": snapshot}
        filepath = tmp_path / "2767_T_20260101_120000_analysis.json"
        filepath.write_text(json.dumps(data))

        result = load_past_snapshots("2767.T", tmp_path)
        assert "2767.T" in result
        assert len(result["2767.T"]) == 1
        assert result["2767.T"][0]["verdict"] == "BUY"

    def test_load_all_tickers(self, tmp_path):
        """Load all tickers when ticker=None."""
        for ticker, prefix in [("2767.T", "2767_T"), ("0005.HK", "0005_HK")]:
            snapshot = _make_snapshot(ticker=ticker)
            data = {"prediction_snapshot": snapshot}
            filepath = tmp_path / f"{prefix}_20260101_120000_analysis.json"
            filepath.write_text(json.dumps(data))

        result = load_past_snapshots(None, tmp_path)
        assert "2767.T" in result
        assert "0005.HK" in result

    def test_malformed_json_skipped(self, tmp_path):
        """Malformed JSON files are skipped with warning."""
        filepath = tmp_path / "BAD_20260101_120000_analysis.json"
        filepath.write_text("not valid json {{{")

        result = load_past_snapshots(None, tmp_path)
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# Lesson Deduplication Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLessonDeduplication:
    """Verify same ticker+date is not stored twice."""

    @pytest.mark.asyncio
    async def test_duplicate_skipped(self):
        """If lesson already exists for ticker+date, skip storage."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection = MagicMock()
        # Simulate existing lesson found
        mock_memory.situation_collection.get.return_value = {
            "ids": ["existing_lesson_1"],
            "documents": ["Some old lesson"],
        }

        comparison = _make_snapshot()
        stored = await store_lesson(
            "Test lesson", "missed_risk", "CYCLICAL_PEAK", comparison, 0.9, mock_memory
        )
        assert stored is False

    @pytest.mark.asyncio
    async def test_new_lesson_stored(self):
        """New lesson (no existing match) is stored."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection = MagicMock()
        # No existing lessons
        mock_memory.situation_collection.get.return_value = {"ids": [], "documents": []}
        mock_memory.add_situations = AsyncMock(return_value=True)

        comparison = _make_snapshot()
        stored = await store_lesson(
            "Test lesson", "missed_risk", "CYCLICAL_PEAK", comparison, 0.9, mock_memory
        )
        assert stored is True
        mock_memory.add_situations.assert_called_once()

    @pytest.mark.asyncio
    async def test_unavailable_memory(self):
        """Unavailable memory returns False gracefully."""
        mock_memory = MagicMock()
        mock_memory.available = False
        stored = await store_lesson(
            "Test", "missed_risk", "CYCLICAL_PEAK", {}, 0.5, mock_memory
        )
        assert stored is False


# ══════════════════════════════════════════════════════════════════════════════
# Lesson Injection Format Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLessonInjectionFormat:
    """Verify lessons_text is well-formed and not too long."""

    @pytest.mark.asyncio
    async def test_format_with_lessons(self):
        """Formatted output contains header and lessons."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.query_similar_situations = AsyncMock(
            return_value=[
                {
                    "document": "Low PEG in cyclical entertainment stocks indicates peak earnings.",
                    "metadata": {
                        "confidence_weight": 0.9,
                        "failure_mode": "CYCLICAL_PEAK",
                        "sector": "Consumer Cyclical",
                        "exchange": "T",
                        "currency": "JPY",
                    },
                    "distance": 0.2,
                },
                {
                    "document": "PFIC burden may be worth bearing for biopharma.",
                    "metadata": {
                        "confidence_weight": 0.7,
                        "failure_mode": "REGULATORY_SHIFT",
                        "sector": "Healthcare",
                        "exchange": "HK",
                        "currency": "HKD",
                    },
                    "distance": 0.4,
                },
            ]
        )

        text = await format_lessons_for_injection(
            mock_memory, "7203.T", "Consumer Cyclical"
        )
        assert "LESSONS FROM PAST ANALYSES" in text
        assert "CYCLICAL_PEAK" in text
        assert "Consumer Cyclical" in text
        # Should be reasonably short
        assert len(text) < 1000

    @pytest.mark.asyncio
    async def test_format_empty_when_no_lessons(self):
        """Returns empty string when no lessons available."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.query_similar_situations = AsyncMock(return_value=[])

        text = await format_lessons_for_injection(
            mock_memory, "7203.T", "Consumer Cyclical"
        )
        assert text == ""

    @pytest.mark.asyncio
    async def test_format_empty_when_memory_unavailable(self):
        """Returns empty string when memory is unavailable."""
        mock_memory = MagicMock()
        mock_memory.available = False

        text = await format_lessons_for_injection(
            mock_memory, "7203.T", "Consumer Cyclical"
        )
        assert text == ""

    @pytest.mark.asyncio
    async def test_format_none_memory(self):
        """Returns empty string when memory is None."""
        text = await format_lessons_for_injection(None, "7203.T", "Consumer Cyclical")
        assert text == ""

    @pytest.mark.asyncio
    async def test_geographic_boost_same_exchange(self):
        """Lessons from same exchange get boosted."""
        mock_memory = MagicMock()
        mock_memory.available = True
        # Two lessons: one from same exchange (T), one from different (HK)
        # Same exchange should rank higher even with lower base confidence
        mock_memory.query_similar_situations = AsyncMock(
            return_value=[
                {
                    "document": "Lesson from T exchange",
                    "metadata": {
                        "confidence_weight": 0.5,
                        "failure_mode": "CYCLICAL_PEAK",
                        "sector": "Consumer Cyclical",
                        "exchange": "T",
                        "currency": "JPY",
                    },
                    "distance": 0.3,
                },
                {
                    "document": "Lesson from HK exchange",
                    "metadata": {
                        "confidence_weight": 0.55,
                        "failure_mode": "GOVERNANCE_BLEED",
                        "sector": "Real Estate",
                        "exchange": "HK",
                        "currency": "HKD",
                    },
                    "distance": 0.3,
                },
            ]
        )

        text = await format_lessons_for_injection(
            mock_memory, "7203.T", "Consumer Cyclical"
        )
        # T exchange lesson gets +0.15 (exchange) + 0.10 (currency) = 0.75 effective
        # HK exchange lesson stays at 0.55 effective
        # So T exchange lesson should appear first
        lines = text.strip().split("\n")
        assert "T exchange" in lines[1]  # First lesson after header

    @pytest.mark.asyncio
    async def test_low_confidence_filtered(self):
        """Lessons below 0.4 effective confidence are filtered out."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.query_similar_situations = AsyncMock(
            return_value=[
                {
                    "document": "Low confidence lesson",
                    "metadata": {
                        "confidence_weight": 0.2,
                        "failure_mode": "MACRO_REGIME",
                        "sector": "Technology",
                        "exchange": "US",
                        "currency": "USD",
                    },
                    "distance": 0.5,
                },
            ]
        )

        text = await format_lessons_for_injection(mock_memory, "7203.T", "Technology")
        assert text == ""  # 0.2 < 0.4 threshold


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompat:
    """Old JSONs without prediction_snapshot are handled gracefully."""

    def test_old_json_skipped(self, tmp_path):
        # Write a file without prediction_snapshot
        old_data = {
            "metadata": {"ticker": "7203.T"},
            "reports": {"market_report": "Some old report"},
        }
        filepath = tmp_path / "7203_T_20251202_analysis.json"
        filepath.write_text(json.dumps(old_data))

        snapshots = load_past_snapshots("7203.T", tmp_path)
        assert snapshots == {}


# ══════════════════════════════════════════════════════════════════════════════
# Max Lessons Per Ticker Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMaxLessonsPerTicker:
    """Verify cap of MAX_LESSONS_PER_TICKER per retrospective run."""

    @pytest.mark.asyncio
    async def test_max_lessons_capped(self):
        """Only MAX_LESSONS_PER_TICKER lessons stored per ticker per run."""
        # Create more snapshots than the cap
        snapshots = {}
        for i in range(5):
            days = 180 + i * 30
            snap = _make_snapshot(
                analysis_date=(datetime.now() - timedelta(days=days)).strftime(
                    "%Y-%m-%d"
                ),
            )
            snap["_source_file"] = f"file_{i}.json"
            if "2767.T" not in snapshots:
                snapshots["2767.T"] = []
            snapshots["2767.T"].append(snap)

        # Mock everything
        comparison = _make_snapshot(
            excess_return_pct=-35.0,
            days_elapsed=180,
            start_price=1774.0,
            end_price=1200.0,
        )
        comparison["_confidence"] = 0.9

        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection = MagicMock()
        mock_memory.situation_collection.get.return_value = {"ids": [], "documents": []}
        mock_memory.add_situations = AsyncMock(return_value=True)

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=comparison,
            ),
            patch(
                "src.retrospective.generate_lesson",
                new_callable=AsyncMock,
                return_value=("test lesson", "missed_risk", "CYCLICAL_PEAK"),
            ),
        ):
            lessons = await run_retrospective("2767.T", Path("/fake"), mock_memory)

        # Should cap at MAX_LESSONS_PER_TICKER
        stored_count = sum(1 for rec in lessons if rec.get("stored"))
        assert stored_count <= MAX_LESSONS_PER_TICKER


# ══════════════════════════════════════════════════════════════════════════════
# Constants Validation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Validate constant definitions."""

    def test_failure_modes_complete(self):
        assert len(FAILURE_MODES) == 8
        assert "CYCLICAL_PEAK" in FAILURE_MODES
        assert "FX_DRIVEN" in FAILURE_MODES
        assert "VALUATION_TRAP" in FAILURE_MODES

    def test_lesson_types_complete(self):
        assert len(LESSON_TYPES) == 4
        assert "missed_risk" in LESSON_TYPES
        assert "correct_call" in LESSON_TYPES

    def test_collection_name(self):
        assert LESSONS_COLLECTION_NAME == "lessons_learned"

    def test_minimum_days(self):
        assert MINIMUM_DAYS_ELAPSED == 30

    def test_max_lessons(self):
        assert MAX_LESSONS_PER_TICKER == 3


# ══════════════════════════════════════════════════════════════════════════════
# Lesson Generation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateLesson:
    """Test LLM-based lesson generation."""

    @pytest.mark.asyncio
    async def test_generation_failure_returns_none(self):
        """LLM failure returns None gracefully."""
        comparison = _make_snapshot(
            excess_return_pct=-32.0,
            days_elapsed=180,
        )
        with patch(
            "src.llms.create_quick_thinking_llm",
            side_effect=Exception("LLM unavailable"),
        ):
            result = await generate_lesson(comparison)
            assert result is None

    @pytest.mark.asyncio
    async def test_generation_parses_response(self):
        """Valid LLM response is parsed into (lesson, type, mode)."""
        comparison = _make_snapshot(
            excess_return_pct=-32.0,
            days_elapsed=180,
        )

        mock_response = MagicMock()
        mock_response.content = (
            "LESSON: Low PEG in cyclical stocks signals peak earnings not undervaluation.\n"
            "TYPE: missed_risk\n"
            "FAILURE_MODE: CYCLICAL_PEAK"
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with (
            patch("src.llms.create_quick_thinking_llm", return_value=mock_llm),
            patch(
                "src.agents.extract_string_content", return_value=mock_response.content
            ),
        ):
            result = await generate_lesson(comparison)

        assert result is not None
        lesson, lesson_type, failure_mode = result
        assert "PEG" in lesson
        assert lesson_type == "missed_risk"
        assert failure_mode == "CYCLICAL_PEAK"

    @pytest.mark.asyncio
    async def test_invalid_type_defaults(self):
        """Invalid lesson_type/failure_mode default to safe values."""
        comparison = _make_snapshot(excess_return_pct=-20.0, days_elapsed=180)

        mock_response = MagicMock()
        mock_response.content = (
            "LESSON: Some lesson text.\n"
            "TYPE: invalid_type\n"
            "FAILURE_MODE: INVALID_MODE"
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with (
            patch("src.llms.create_quick_thinking_llm", return_value=mock_llm),
            patch(
                "src.agents.extract_string_content", return_value=mock_response.content
            ),
        ):
            result = await generate_lesson(comparison)

        assert result is not None
        _, lesson_type, failure_mode = result
        assert lesson_type == "missed_risk"  # default
        assert failure_mode == "OPERATIONAL_MISS"  # default


# ══════════════════════════════════════════════════════════════════════════════
# Create Lessons Memory Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateLessonsMemory:
    def test_factory_returns_memory(self):
        """create_lessons_memory returns a FinancialSituationMemory instance."""
        with patch("src.memory.FinancialSituationMemory") as MockFSM:
            mock_instance = MagicMock()
            MockFSM.return_value = mock_instance
            result = create_lessons_memory()
            MockFSM.assert_called_once_with("lessons_learned")
            assert result == mock_instance


# ══════════════════════════════════════════════════════════════════════════════
# Early Dedup Helper Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLessonAlreadyProcessed:
    """Test _lesson_already_processed() ChromaDB metadata query."""

    def test_returns_true_when_lesson_exists(self):
        """Returns True when ChromaDB has a matching (ticker, date) entry."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection.get.return_value = {
            "ids": ["lesson_1"],
            "documents": ["Some lesson"],
        }

        assert _lesson_already_processed(mock_memory, "2767.T", "2025-08-01") is True
        mock_memory.situation_collection.get.assert_called_once()

    def test_returns_false_when_no_lesson(self):
        """Returns False when ChromaDB has no matching entry."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection.get.return_value = {"ids": [], "documents": []}

        assert _lesson_already_processed(mock_memory, "2767.T", "2025-08-01") is False

    def test_returns_false_when_memory_unavailable(self):
        """Returns False when memory is not available."""
        mock_memory = MagicMock()
        mock_memory.available = False

        assert _lesson_already_processed(mock_memory, "2767.T", "2025-08-01") is False

    def test_returns_false_when_memory_none(self):
        """Returns False when memory is None."""
        assert _lesson_already_processed(None, "2767.T", "2025-08-01") is False

    def test_returns_false_on_exception(self):
        """Returns False on ChromaDB query failure (graceful degradation)."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection.get.side_effect = Exception("ChromaDB down")

        assert _lesson_already_processed(mock_memory, "2767.T", "2025-08-01") is False


# ══════════════════════════════════════════════════════════════════════════════
# Count Fast-Path Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCountFastPath:
    """Verify format_lessons_for_injection() returns "" immediately when empty."""

    @pytest.mark.asyncio
    async def test_empty_collection_skips_embedding(self):
        """count() == 0 returns empty string without calling query_similar_situations."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection.count.return_value = 0
        mock_memory.query_similar_situations = AsyncMock(return_value=[])

        text = await format_lessons_for_injection(
            mock_memory, "7203.T", "Consumer Cyclical"
        )
        assert text == ""
        # query_similar_situations should NOT have been called
        mock_memory.query_similar_situations.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonempty_collection_proceeds(self):
        """count() > 0 proceeds to normal query path."""
        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.situation_collection.count.return_value = 3
        mock_memory.query_similar_situations = AsyncMock(return_value=[])

        text = await format_lessons_for_injection(
            mock_memory, "7203.T", "Consumer Cyclical"
        )
        assert text == ""
        # query_similar_situations SHOULD have been called
        mock_memory.query_similar_situations.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Early Dedup in run_retrospective() Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEarlyDedupInRetrospective:
    """Verify already-processed snapshots skip compare_to_reality()."""

    @pytest.mark.asyncio
    async def test_already_processed_snapshots_skipped(self):
        """Snapshots with existing lessons skip the expensive yfinance call."""
        snapshot = _make_snapshot(
            analysis_date=(datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
        )
        snapshot["_source_file"] = "test.json"
        snapshots = {"2767.T": [snapshot]}

        mock_memory = MagicMock()
        mock_memory.available = True
        # Simulate lesson already exists for this snapshot
        mock_memory.situation_collection.get.return_value = {
            "ids": ["existing_1"],
            "documents": ["Existing lesson"],
        }

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
            ) as mock_compare,
        ):
            lessons = await run_retrospective("2767.T", Path("/fake"), mock_memory)

        # compare_to_reality should NOT have been called
        mock_compare.assert_not_called()
        assert lessons == []

    @pytest.mark.asyncio
    async def test_new_snapshots_processed(self):
        """Snapshots without existing lessons proceed to comparison."""
        snapshot = _make_snapshot(
            analysis_date=(datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
        )
        snapshot["_source_file"] = "test.json"
        snapshots = {"2767.T": [snapshot]}

        mock_memory = MagicMock()
        mock_memory.available = True
        # No existing lesson
        mock_memory.situation_collection.get.return_value = {"ids": [], "documents": []}
        mock_memory.add_situations = AsyncMock(return_value=True)

        comparison = _make_snapshot(
            excess_return_pct=-35.0,
            days_elapsed=180,
            start_price=1774.0,
            end_price=1200.0,
        )
        comparison["_confidence"] = 0.9

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=comparison,
            ) as mock_compare,
            patch(
                "src.retrospective.generate_lesson",
                new_callable=AsyncMock,
                return_value=("test lesson", "missed_risk", "CYCLICAL_PEAK"),
            ),
        ):
            lessons = await run_retrospective("2767.T", Path("/fake"), mock_memory)

        # compare_to_reality SHOULD have been called
        mock_compare.assert_called_once()
        assert len(lessons) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Analyst Coverage Field Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAnalystCoverageFields:
    """Tests for ANALYST_COVERAGE_ENGLISH field fix and ANALYST_COVERAGE_TOTAL_EST."""

    def test_snapshot_extracts_analyst_coverage_english(self):
        """Verify field name fix: ANALYST_COVERAGE_ENGLISH is extracted, not ANALYST_COVERAGE."""
        result = _make_result()
        snapshot = extract_snapshot(result, "2767.T")
        # Should extract 4.0 from ANALYST_COVERAGE_ENGLISH field
        assert snapshot["analyst_coverage"] == 4.0

    def test_snapshot_extracts_analyst_coverage_total_est(self):
        """Verify new field ANALYST_COVERAGE_TOTAL_EST is extracted."""
        result = _make_result()
        snapshot = extract_snapshot(result, "2767.T")
        assert snapshot["analyst_coverage_total_est"] == "12"

    def test_snapshot_total_est_missing(self):
        """ANALYST_COVERAGE_TOTAL_EST absent → None."""
        data_block_no_total = """
### --- START DATA_BLOCK ---
ANALYST_COVERAGE_ENGLISH: 5
PROFITABILITY_TREND: STABLE
### --- END DATA_BLOCK ---
"""
        result = _make_result(fundamentals_report=data_block_no_total)
        snapshot = extract_snapshot(result, "0005.HK")
        assert snapshot["analyst_coverage_total_est"] is None

    def test_snapshot_total_est_tier(self):
        """ANALYST_COVERAGE_TOTAL_EST with tier value (HIGH)."""
        data_block = """
### --- START DATA_BLOCK ---
ANALYST_COVERAGE_ENGLISH: 3
ANALYST_COVERAGE_TOTAL_EST: HIGH
### --- END DATA_BLOCK ---
"""
        result = _make_result(fundamentals_report=data_block)
        snapshot = extract_snapshot(result, "7203.T")
        assert snapshot["analyst_coverage_total_est"] == "HIGH"

    def test_old_analyst_coverage_field_returns_none(self):
        """Old DATA_BLOCK with ANALYST_COVERAGE (no _ENGLISH suffix) → None coverage."""
        old_data_block = """
### --- START DATA_BLOCK ---
ANALYST_COVERAGE: 4
### --- END DATA_BLOCK ---
"""
        result = _make_result(fundamentals_report=old_data_block)
        snapshot = extract_snapshot(result, "2767.T")
        # ANALYST_COVERAGE doesn't match ANALYST_COVERAGE_ENGLISH, so should be None
        assert snapshot["analyst_coverage"] is None
