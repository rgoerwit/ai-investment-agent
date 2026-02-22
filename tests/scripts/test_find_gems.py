"""Tests for scripts/find_gems.py — screening pipeline."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add scripts/ to path so we can import find_gems as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import find_gems  # noqa: E402

# Path to real exchange config for integration tests
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "exchanges.json"


# ============================================================
# TestPassesFilters — pure logic, no mocking
# ============================================================
class TestPassesFilters:
    """Unit tests for _passes_filters() with default thresholds."""

    DEFAULT_KWARGS = {"max_pe": 18.0, "min_roe": 13.0, "min_roa": 6.0, "max_de": 150.0}

    @staticmethod
    def _make_row(**overrides):
        base = {
            "YF_Ticker": "TEST.T",
            "P/E": 12.0,
            "Forward_PE": None,
            "ROE": 0.15,
            "ROA": 0.08,
            "Debt_to_Equity": 80.0,
            "Operating_Cash_Flow": 1_000_000,
            "Net_Income": 800_000,
            "OCF_NI_Ratio": 1.25,
            "Net_Debt_to_Equity": 50.0,
            "Total_Debt": 500_000,
            "Total_Cash": 200_000,
            "Revenue_Years_Positive": 4,
            "Analyst_Coverage": 5,
        }
        base.update(overrides)
        return base

    def test_passes_all_criteria(self):
        row = self._make_row()
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_missing_pe_fails(self):
        row = self._make_row(**{"P/E": None})
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_high_pe_fails(self):
        row = self._make_row(**{"P/E": 25.0})
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_low_profitability_fails(self):
        row = self._make_row(ROE=0.05, ROA=0.03)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_roe_or_roa_logic_roe_only(self):
        """ROE above threshold with ROA missing should pass (OR logic)."""
        row = self._make_row(ROE=0.15, ROA=None)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_roe_or_roa_logic_roa_only(self):
        """ROA above threshold with ROE missing should pass (OR logic)."""
        row = self._make_row(ROE=None, ROA=0.08)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_missing_de_fails(self):
        row = self._make_row(Debt_to_Equity=None)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_high_de_fails(self):
        row = self._make_row(Debt_to_Equity=200.0, Net_Debt_to_Equity=180.0)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_high_gross_de_passes_via_net_debt_fallback(self):
        """Gross D/E above threshold but net D/E below should pass."""
        row = self._make_row(Debt_to_Equity=200.0, Net_Debt_to_Equity=100.0)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_high_gross_de_no_net_debt_fails(self):
        """Gross D/E above threshold with no net debt data should fail."""
        row = self._make_row(Debt_to_Equity=200.0, Net_Debt_to_Equity=None)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    # --- OCF / Net Income earnings quality ---

    def test_low_ocf_ni_ratio_fails(self):
        """OCF/NI below 0.8 with positive NI should fail."""
        row = self._make_row(
            Net_Income=1_000_000, OCF_NI_Ratio=0.5, Operating_Cash_Flow=500_000
        )
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_good_ocf_ni_ratio_passes(self):
        """OCF/NI above 0.8 should pass."""
        row = self._make_row(Net_Income=800_000, OCF_NI_Ratio=1.25)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_negative_ni_skips_ocf_ni_check(self):
        """When NI is negative, OCF/NI check should be skipped entirely."""
        row = self._make_row(Net_Income=-500_000, OCF_NI_Ratio=None)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_missing_ni_skips_ocf_ni_check(self):
        """When NI is None, OCF/NI check should be skipped."""
        row = self._make_row(Net_Income=None, OCF_NI_Ratio=None)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    # --- Revenue history ---

    def test_insufficient_revenue_history_fails(self):
        """Fewer than 3 years of positive revenue should fail."""
        row = self._make_row(Revenue_Years_Positive=2)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_sufficient_revenue_history_passes(self):
        """3+ years of positive revenue should pass."""
        row = self._make_row(Revenue_Years_Positive=4)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_missing_revenue_history_passes(self):
        """When revenue data unavailable, check should be skipped (no penalty)."""
        row = self._make_row(Revenue_Years_Positive=None)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_exactly_three_revenue_years_passes(self):
        """Exactly 3 years should pass (boundary check)."""
        row = self._make_row(Revenue_Years_Positive=3)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_negative_ocf_fails(self):
        row = self._make_row(Operating_Cash_Flow=-500)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_zero_ocf_fails(self):
        row = self._make_row(Operating_Cash_Flow=0)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

    def test_custom_thresholds(self):
        """Relaxed thresholds should allow previously-failing rows."""
        row = self._make_row(**{"P/E": 22.0}, ROE=0.06, Debt_to_Equity=140.0)
        # Fails with defaults (P/E 22 > 18, ROE 6% < 13%)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False
        # Passes with relaxed thresholds
        assert (
            find_gems._passes_filters(
                row, max_pe=25.0, min_roe=5.0, min_roa=6.0, max_de=150.0
            )
            is True
        )

    def test_none_row_fails(self):
        assert find_gems._passes_filters(None, **self.DEFAULT_KWARGS) is False

    def test_empty_dict_fails(self):
        assert find_gems._passes_filters({}, **self.DEFAULT_KWARGS) is False

    def test_boundary_pe_exactly_at_threshold(self):
        """P/E exactly at max_pe should fail (> not >=)."""
        # The code uses `pe > max_pe`, so pe == max_pe should pass
        row = self._make_row(**{"P/E": 18.0})
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True

    def test_string_numeric_values_handled(self):
        """Values stored as strings should be converted via _safe_float."""
        row = self._make_row(
            **{"P/E": "12.5"},
            ROE="0.15",
            Debt_to_Equity="80",
            Operating_Cash_Flow="1000000",
        )
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is True


# ============================================================
# TestScrapeExchanges — mocked HTTP
# ============================================================
class TestScrapeExchanges:
    """Integration tests for scrape_exchanges() with mocked network."""

    @staticmethod
    def _make_config(*exchanges):
        return {
            "meta": {"description": "Test config"},
            "exchanges": list(exchanges),
        }

    @staticmethod
    def _make_exchange(country, name, suffix=".T", method="download_csv"):
        return {
            "country": country,
            "exchange_name": name,
            "yahoo_suffix": suffix,
            "source_url": f"https://example.com/{name}.csv",
            "method": method,
            "params": {"ticker_col": "Code", "name_col": "Name"},
        }

    def test_us_excluded_by_default(self):
        us_ex = self._make_exchange("United States", "NYSE")
        ca_ex = self._make_exchange("Canada", "TSX", suffix=".TO")
        config = self._make_config(us_ex, ca_ex)

        ca_df = pd.DataFrame({"Code": ["RY", "TD"], "Name": ["Royal Bank", "TD Bank"]})
        mock_handler = MagicMock(return_value=ca_df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 2
        assert all(result["Country"] == "Canada")
        # Handler should only be called for Canada (US skipped)
        assert mock_handler.call_count == 1

    def test_include_us(self):
        us_ex = self._make_exchange("United States", "NYSE", suffix="")
        config = self._make_config(us_ex)

        us_df = pd.DataFrame({"Code": ["AAPL"], "Name": ["Apple"]})
        mock_handler = MagicMock(return_value=us_df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=False)

        assert len(result) == 1

    def test_empty_exchange_skipped(self):
        ex = self._make_exchange("Japan", "TSE")
        config = self._make_config(ex)

        mock_handler = MagicMock(return_value=pd.DataFrame())

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 0

    def test_failed_exchange_continues(self):
        ex1 = self._make_exchange("Japan", "TSE")
        ex2 = self._make_exchange("Korea", "KRX", suffix=".KS")
        config = self._make_config(ex1, ex2)

        kr_df = pd.DataFrame({"Code": ["005930"], "Name": ["Samsung"]})

        def side_effect(cfg, session):
            if cfg["exchange_name"] == "TSE":
                raise ConnectionError("Network down")
            return kr_df

        mock_handler = MagicMock(side_effect=side_effect)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 1
        assert result.iloc[0]["Country"] == "Korea"

    def test_filter_param_applied(self):
        """Config with filter: {"Type": "Equity"} should keep only matching rows."""
        ex = self._make_exchange("Hong Kong", "HKEX", suffix=".HK")
        ex["params"]["filter"] = {"Type": "Equity"}
        config = self._make_config(ex)

        df = pd.DataFrame(
            {
                "Code": ["0001", "W001", "0005", "C002"],
                "Name": ["CKH", "Warrant A", "HSBC", "CBBC B"],
                "Type": ["Equity", "Warrant", "Equity", "CBBC"],
            }
        )
        mock_handler = MagicMock(return_value=df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 2
        tickers = result["YF_Ticker"].tolist()
        assert "0001.HK" in tickers
        assert "0005.HK" in tickers

    def test_exclude_filter_list(self):
        """exclude_filter with a list should remove rows matching any value."""
        ex = self._make_exchange("Germany", "XETRA", suffix=".DE")
        ex["params"]["exclude_filter"] = {"Type": ["ETF", "Bond"]}
        config = self._make_config(ex)

        df = pd.DataFrame(
            {
                "Code": ["SAP", "ETF1", "SIE", "BND1"],
                "Name": ["SAP SE", "ETF Fund", "Siemens", "Bond Corp"],
                "Type": ["Equity", "ETF", "Equity", "Bond"],
            }
        )
        mock_handler = MagicMock(return_value=df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 2
        tickers = result["YF_Ticker"].tolist()
        assert "SAP.DE" in tickers
        assert "SIE.DE" in tickers

    def test_exclude_filter_string_contains(self):
        """exclude_filter with a string should use case-insensitive contains."""
        ex = self._make_exchange("Europe", "Euronext", suffix=".PA")
        ex["params"]["exclude_filter"] = {"Market": "Growth"}
        config = self._make_config(ex)

        df = pd.DataFrame(
            {
                "Code": ["AI", "ML", "BIG"],
                "Name": ["Air Liquide", "ML Growth Co", "BIG Corp"],
                "Market": ["Euronext Paris", "Euronext Growth Paris", "Euronext Paris"],
            }
        )
        mock_handler = MagicMock(return_value=df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 2
        tickers = result["YF_Ticker"].tolist()
        assert "AI.PA" in tickers
        assert "BIG.PA" in tickers

    def test_filter_fuzzy_column_match(self):
        """Filter column name with different case should still match."""
        ex = self._make_exchange("Hong Kong", "HKEX", suffix=".HK")
        ex["params"]["filter"] = {"category": "Equity"}  # lowercase
        config = self._make_config(ex)

        df = pd.DataFrame(
            {
                "Code": ["0001", "W001"],
                "Name": ["CKH", "Warrant A"],
                "Category": ["Equity", "Warrant"],  # Uppercase in DataFrame
            }
        )
        mock_handler = MagicMock(return_value=df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 1
        assert result.iloc[0]["YF_Ticker"] == "0001.HK"

    def test_disabled_exchange_skipped(self):
        """Exchanges with enabled:false should be skipped entirely."""
        ex_enabled = self._make_exchange("Japan", "TSE")
        ex_disabled = self._make_exchange("South Korea", "KOSPI", suffix=".KS")
        ex_disabled["enabled"] = False
        config = self._make_config(ex_enabled, ex_disabled)

        jp_df = pd.DataFrame({"Code": ["7203"], "Name": ["Toyota"]})
        kr_df = pd.DataFrame({"Code": ["005930"], "Name": ["Samsung"]})

        def side_effect(cfg, session):
            if "KOSPI" in cfg["exchange_name"]:
                return kr_df
            return jp_df

        mock_handler = MagicMock(side_effect=side_effect)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 1
        assert result.iloc[0]["YF_Ticker"] == "7203.T"
        # Handler should only be called once (Korea skipped)
        assert mock_handler.call_count == 1

    def test_filter_empty_result_skipped(self):
        """If filter removes all rows, exchange should be skipped gracefully."""
        ex = self._make_exchange("Japan", "TSE")
        ex["params"]["filter"] = {"Type": "Equity"}
        config = self._make_config(ex)

        df = pd.DataFrame(
            {
                "Code": ["W001"],
                "Name": ["Warrant A"],
                "Type": ["Warrant"],
            }
        )
        mock_handler = MagicMock(return_value=df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 0

    def test_deduplication(self):
        ex1 = self._make_exchange("Japan", "TSE1")
        ex2 = self._make_exchange("Japan", "TSE2")
        config = self._make_config(ex1, ex2)

        df = pd.DataFrame({"Code": ["7203"], "Name": ["Toyota"]})
        mock_handler = MagicMock(return_value=df)

        with patch.dict(find_gems._HANDLERS, {"download_csv": mock_handler}):
            with patch.object(find_gems, "_check_deps"):
                with patch("find_gems.time.sleep"):
                    result = find_gems.scrape_exchanges(config, exclude_us=True)

        assert len(result) == 1


# ============================================================
# TestFetchAndFilter — mocked yfinance
# ============================================================
class TestFetchAndFilter:
    """Filter phase with mocked yfinance."""

    @staticmethod
    def _make_tickers_df(*tickers):
        return pd.DataFrame({"YF_Ticker": list(tickers)})

    @staticmethod
    def _mock_info_good():
        return {
            "regularMarketPrice": 1500,
            "currentPrice": 1500,
            "trailingPE": 12.0,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.08,
            "debtToEquity": 80.0,
            "operatingCashflow": 1_000_000_000,
            "freeCashflow": 500_000_000,
            "marketCap": 100_000_000_000,  # ¥100B (~$670M USD)
            "averageVolume": 1_000_000,
            "sector": "Industrials",
            "industry": "Auto Manufacturers",
            "longName": "Test Corp",
            "currency": "JPY",
            "quoteType": "EQUITY",
        }

    @staticmethod
    def _mock_info_bad_pe():
        info = TestFetchAndFilter._mock_info_good()
        info["trailingPE"] = 25.0
        return info

    def test_passing_tickers_returned(self):
        df = self._make_tickers_df("7203.T")

        mock_ticker = MagicMock()
        mock_ticker.info = self._mock_info_good()

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                passing, enriched = find_gems.fetch_and_filter(
                    df, max_pe=18.0, min_roe=13.0, min_roa=6.0, max_de=150.0, workers=1
                )

        assert len(passing) == 1
        assert passing.iloc[0]["YF_Ticker"] == "7203.T"

    def test_failing_tickers_excluded(self):
        df = self._make_tickers_df("FAIL.T")

        mock_ticker = MagicMock()
        mock_ticker.info = self._mock_info_bad_pe()

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                passing, enriched = find_gems.fetch_and_filter(
                    df, max_pe=18.0, min_roe=13.0, min_roa=6.0, max_de=150.0, workers=1
                )

        assert len(passing) == 0
        assert len(enriched) == 1  # Still in enriched even though filtered out

    def test_yfinance_error_handled(self):
        df = self._make_tickers_df("ERR.T")

        with patch("find_gems.yf.Ticker", side_effect=Exception("404 Not Found")):
            with patch("find_gems.time.sleep"):
                passing, enriched = find_gems.fetch_and_filter(
                    df, max_pe=18.0, min_roe=13.0, min_roa=6.0, max_de=150.0, workers=1
                )

        assert len(passing) == 0
        assert len(enriched) == 0

    def test_empty_info_handled(self):
        df = self._make_tickers_df("EMPTY.T")

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                passing, enriched = find_gems.fetch_and_filter(
                    df, max_pe=18.0, min_roe=13.0, min_roa=6.0, max_de=150.0, workers=1
                )

        assert len(passing) == 0


# ============================================================
# TestWriteOutputs — filesystem with tmp_path
# ============================================================
class TestWriteOutputs:
    """Output file generation."""

    def test_ticker_only_file(self, tmp_path):
        df = pd.DataFrame({"YF_Ticker": ["0005.HK", "7203.T", "2330.TW"]})
        out = tmp_path / "gems.txt"

        find_gems.write_outputs(df, str(out))

        lines = out.read_text().strip().split("\n")
        assert lines == ["0005.HK", "2330.TW", "7203.T"]  # sorted

    def test_details_csv(self, tmp_path):
        df = pd.DataFrame(
            {
                "YF_Ticker": ["7203.T"],
                "Company_YF": ["Toyota"],
                "P/E": [12.0],
                "Debt_to_Equity": [80.0],
            }
        )
        out = tmp_path / "gems.txt"
        details = tmp_path / "details.csv"

        find_gems.write_outputs(df, str(out), details_path=str(details))

        assert details.exists()
        result = pd.read_csv(details)
        assert "YF_Ticker" in result.columns
        assert len(result) == 1

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "gems.txt"
        df = pd.DataFrame({"YF_Ticker": ["TEST.T"]})

        find_gems.write_outputs(df, str(out))

        assert out.exists()
        assert out.read_text().strip() == "TEST.T"

    def test_no_details_file_when_not_requested(self, tmp_path):
        df = pd.DataFrame({"YF_Ticker": ["TEST.T"]})
        out = tmp_path / "gems.txt"

        find_gems.write_outputs(df, str(out), details_path=None)

        assert out.exists()
        # No details CSV should exist
        assert len(list(tmp_path.glob("*.csv"))) == 0


# ============================================================
# TestCLIParsing — argument parsing
# ============================================================
class TestCLIParsing:
    """CLI argument parsing validation."""

    def test_scrape_only_and_filter_only_exclusive(self):
        """--scrape-only and --filter-only are mutually exclusive."""
        with pytest.raises(SystemExit):
            find_gems.parse_args.__wrapped__ if hasattr(
                find_gems.parse_args, "__wrapped__"
            ) else None
            with patch(
                "sys.argv",
                [
                    "find_gems.py",
                    "--scrape-only",
                    "--filter-only",
                    "file.csv",
                    "--output",
                    "out.txt",
                ],
            ):
                find_gems.parse_args()

    def test_defaults(self):
        with patch("sys.argv", ["find_gems.py", "--output", "out.txt"]):
            args = find_gems.parse_args()

        assert args.max_pe == 18.0
        assert args.min_roe == 13.0
        assert args.min_roa == 6.0
        assert args.max_de == 150.0
        assert args.min_mcap == 50_000_000
        assert args.min_volume == 100_000
        assert args.max_coverage == 30
        assert args.no_ocf_waiver is False
        assert args.workers == 4
        assert args.debug is False
        assert args.include_us is False
        assert args.scrape_only is False
        assert args.filter_only is None

    def test_custom_thresholds_parsed(self):
        with patch(
            "sys.argv",
            [
                "find_gems.py",
                "--output",
                "out.txt",
                "--max-pe",
                "25",
                "--min-roe",
                "8",
                "--min-roa",
                "3",
                "--max-de",
                "200",
            ],
        ):
            args = find_gems.parse_args()

        assert args.max_pe == 25.0
        assert args.min_roe == 8.0
        assert args.min_roa == 3.0
        assert args.max_de == 200.0


# ============================================================
# TestNormalizeTicker — ticker format conversion
# ============================================================
class TestNormalizeTicker:
    """Tests for _normalize_ticker() utility."""

    def test_standard_suffix_unchanged(self):
        assert find_gems._normalize_ticker("7203.T") == "7203.T"
        assert find_gems._normalize_ticker("0005.HK") == "0005.HK"

    def test_multi_dot_becomes_dash(self):
        assert find_gems._normalize_ticker("A.B.TO") == "A-B.TO"

    def test_single_char_suffix_becomes_dash(self):
        # e.g., "BRK.B" -> "BRK-B" (single char, not exchange suffix)
        assert find_gems._normalize_ticker("BRK.B") == "BRK-B"

    def test_non_string_converted(self):
        assert find_gems._normalize_ticker(12345) == "12345"


# ============================================================
# TestSafeFloat — type coercion
# ============================================================
class TestSafeFloat:
    """Tests for _safe_float() helper."""

    def test_normal_float(self):
        assert find_gems._safe_float(12.5) == 12.5

    def test_string_float(self):
        assert find_gems._safe_float("12.5") == 12.5

    def test_none_returns_none(self):
        assert find_gems._safe_float(None) is None

    def test_invalid_string_returns_none(self):
        assert find_gems._safe_float("N/A") is None

    def test_int_converted(self):
        assert find_gems._safe_float(42) == 42.0


# ============================================================
# TestApplyFilters — unit tests for extracted helper
# ============================================================
class TestApplyFilters:
    """Unit tests for the _apply_filters() helper."""

    def test_positive_filter(self):
        df = pd.DataFrame(
            {"Type": ["Equity", "Warrant", "Equity"], "Name": ["A", "B", "C"]}
        )
        config = {"params": {"filter": {"Type": "Equity"}}}
        result = find_gems._apply_filters(df, config)
        assert len(result) == 2

    def test_exclude_filter_list(self):
        df = pd.DataFrame({"Type": ["Equity", "ETF", "Bond"], "Name": ["A", "B", "C"]})
        config = {"params": {"exclude_filter": {"Type": ["ETF", "Bond"]}}}
        result = find_gems._apply_filters(df, config)
        assert len(result) == 1
        assert result.iloc[0]["Name"] == "A"

    def test_exclude_filter_string_contains(self):
        df = pd.DataFrame(
            {"Market": ["Paris", "Growth Paris", "Amsterdam"], "Name": ["A", "B", "C"]}
        )
        config = {"params": {"exclude_filter": {"Market": "Growth"}}}
        result = find_gems._apply_filters(df, config)
        assert len(result) == 2

    def test_no_filters_returns_unchanged(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        config = {"params": {}}
        result = find_gems._apply_filters(df, config)
        assert len(result) == 3

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"Name": ["A", "B"]})
        config = {"params": {"filter": {"NonExistent": "X"}}}
        result = find_gems._apply_filters(df, config)
        assert len(result) == 2


# ============================================================
# TestToUsd — FX conversion helper
# ============================================================
class TestToUsd:
    """Unit tests for _to_usd() helper."""

    def test_usd_identity(self):
        rates = {"USD": 1.0, "JPY": 0.0067}
        assert find_gems._to_usd(1000, "USD", rates) == 1000

    def test_jpy_conversion(self):
        rates = {"USD": 1.0, "JPY": 0.0067}
        result = find_gems._to_usd(150_000, "JPY", rates)
        assert result == pytest.approx(150_000 * 0.0067)

    def test_unknown_currency_returns_none(self):
        rates = {"USD": 1.0}
        assert find_gems._to_usd(1000, "XYZ", rates) is None

    def test_gbp_pence_normalization(self):
        """GBp (pence) should be divided by 100 and converted as GBP."""
        rates = {"USD": 1.0, "GBP": 1.27}
        result = find_gems._to_usd(10000, "GBp", rates)
        # 10000 pence = 100 GBP * 1.27 = 127 USD
        assert result == pytest.approx(100 * 1.27)

    def test_none_value_returns_none(self):
        rates = {"USD": 1.0, "JPY": 0.0067}
        assert find_gems._to_usd(None, "JPY", rates) is None

    def test_none_currency_returns_none(self):
        rates = {"USD": 1.0}
        assert find_gems._to_usd(1000, None, rates) is None


# ============================================================
# TestQuoteTypeGuard — Filter A
# ============================================================
class TestQuoteTypeGuard:
    """Filter A: quoteType must be EQUITY."""

    @staticmethod
    def _make_info(**overrides):
        base = {
            "regularMarketPrice": 100,
            "currentPrice": 100,
            "trailingPE": 10,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.08,
            "debtToEquity": 50,
            "operatingCashflow": 1_000_000,
            "marketCap": 1_000_000_000,
            "currency": "USD",
            "quoteType": "EQUITY",
        }
        base.update(overrides)
        return base

    def test_etf_rejected(self):
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(quoteType="ETF")
        row = {"YF_Ticker": "VTI"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(row, debug=True)

        assert result is None

    def test_equity_passes(self):
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(quoteType="EQUITY")
        row = {"YF_Ticker": "7203.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(row)

        assert result is not None

    def test_missing_quotetype_passes(self):
        """If quoteType is absent, don't reject (fail-open)."""
        info = self._make_info()
        del info["quoteType"]
        mock_ticker = MagicMock()
        mock_ticker.info = info
        row = {"YF_Ticker": "UNKNOWN.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(row)

        assert result is not None

    def test_warrant_rejected(self):
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(quoteType="WARRANT")
        row = {"YF_Ticker": "W001.HK"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(row)

        assert result is None


# ============================================================
# TestMarketCapFilter — Filter B
# ============================================================
class TestMarketCapFilter:
    """Filter B: market cap USD must exceed threshold."""

    @staticmethod
    def _make_info(**overrides):
        base = {
            "regularMarketPrice": 100,
            "currentPrice": 100,
            "trailingPE": 10,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.08,
            "debtToEquity": 50,
            "operatingCashflow": 1_000_000,
            "marketCap": 1_000_000_000,
            "currency": "USD",
            "quoteType": "EQUITY",
        }
        base.update(overrides)
        return base

    def test_micro_cap_rejected(self):
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(marketCap=10_000_000, currency="USD")
        row = {"YF_Ticker": "TINY.V"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_mcap=50_000_000
                )

        assert result is None

    def test_mid_cap_passes(self):
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(marketCap=500_000_000, currency="USD")
        row = {"YF_Ticker": "MID.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_mcap=50_000_000
                )

        assert result is not None

    def test_missing_marketcap_passes(self):
        """Missing market cap should not reject (fail-open)."""
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(marketCap=None, currency="USD")
        row = {"YF_Ticker": "NOMC.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_mcap=50_000_000
                )

        assert result is not None

    def test_unknown_currency_passes(self):
        """Unknown currency → can't convert → fail-open."""
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(marketCap=10_000_000, currency="XYZ")
        row = {"YF_Ticker": "WEIRD.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_mcap=50_000_000
                )

        assert result is not None

    def test_jpy_conversion_applied(self):
        """JPY market cap should be converted to USD before threshold check."""
        # 5B JPY * 0.0067 = 33.5M USD → below $50M threshold
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(marketCap=5_000_000_000, currency="JPY")
        row = {"YF_Ticker": "JPSMALL.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0, "JPY": 0.0067}, min_mcap=50_000_000
                )

        assert result is None  # 33.5M < 50M

    def test_disabled_when_zero(self):
        """min_mcap=0 should disable the check."""
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(marketCap=1_000, currency="USD")
        row = {"YF_Ticker": "MICRO.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(row, fx_rates={"USD": 1.0}, min_mcap=0)

        assert result is not None


# ============================================================
# TestVolumeFilter — Filter C
# ============================================================
class TestVolumeFilter:
    """Filter C: daily dollar volume must exceed threshold."""

    @staticmethod
    def _make_info(**overrides):
        base = {
            "regularMarketPrice": 100,
            "currentPrice": 100,
            "trailingPE": 10,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.08,
            "debtToEquity": 50,
            "operatingCashflow": 1_000_000,
            "marketCap": 1_000_000_000,
            "averageVolume": 100_000,
            "currency": "USD",
            "quoteType": "EQUITY",
        }
        base.update(overrides)
        return base

    def test_low_volume_rejected(self):
        # 500 shares * $10 = $5,000/day → below $100K
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(averageVolume=500, currentPrice=10)
        row = {"YF_Ticker": "ILLIQUID.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_volume=100_000
                )

        assert result is None

    def test_adequate_volume_passes(self):
        # 10,000 shares * $50 = $500K/day → above $100K
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(averageVolume=10_000, currentPrice=50)
        row = {"YF_Ticker": "LIQUID.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_volume=100_000
                )

        assert result is not None

    def test_london_pence_correction(self):
        """For .L tickers, price*volume should be divided by 100 (pence→GBP)."""
        # 50,000 shares * 500p = 25,000,000p → /100 = 250,000 GBP * 1.27 = 317,500 USD
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(
            averageVolume=50_000, currentPrice=500, currency="GBP"
        )
        row = {"YF_Ticker": "VOD.L"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row,
                    fx_rates={"USD": 1.0, "GBP": 1.27},
                    min_volume=100_000,
                )

        assert result is not None
        # turnover_usd should be ~317,500 (above threshold)
        assert result["Daily_Turnover_USD"] == pytest.approx(
            50_000 * 500 / 100.0 * 1.27, rel=0.01
        )

    def test_missing_volume_passes(self):
        """Missing averageVolume should not reject (fail-open)."""
        mock_ticker = MagicMock()
        mock_ticker.info = self._make_info(averageVolume=None)
        row = {"YF_Ticker": "NOVOL.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_volume=100_000
                )

        assert result is not None

    def test_missing_price_passes(self):
        """Missing price should not reject (fail-open)."""
        mock_ticker = MagicMock()
        info = self._make_info(averageVolume=100)
        info["currentPrice"] = None
        info["regularMarketPrice"] = None
        mock_ticker.info = info
        row = {"YF_Ticker": "NOPRICE.T"}

        with patch("find_gems.yf.Ticker", return_value=mock_ticker):
            with patch("find_gems.time.sleep"):
                result = find_gems._process_row(
                    row, fx_rates={"USD": 1.0}, min_volume=100_000
                )

        # Missing price means we can't compute turnover → fail-open
        assert result is not None


# ============================================================
# TestCoverageFilter — Filter D
# ============================================================
class TestCoverageFilter:
    """Filter D: analyst coverage must not exceed threshold."""

    KWARGS = {"max_pe": 18.0, "min_roe": 13.0, "min_roa": 6.0, "max_de": 150.0}

    @staticmethod
    def _make_row(**overrides):
        base = {
            "YF_Ticker": "TEST.T",
            "P/E": 12.0,
            "Forward_PE": None,
            "ROE": 0.15,
            "ROA": 0.08,
            "Debt_to_Equity": 80.0,
            "Operating_Cash_Flow": 1_000_000,
            "Net_Income": 800_000,
            "OCF_NI_Ratio": 1.25,
            "Net_Debt_to_Equity": 50.0,
            "Total_Debt": 500_000,
            "Total_Cash": 200_000,
            "Revenue_Years_Positive": 4,
            "Analyst_Coverage": 5,
        }
        base.update(overrides)
        return base

    def test_high_coverage_rejected(self):
        row = self._make_row(Analyst_Coverage=35)
        assert find_gems._passes_filters(row, **self.KWARGS, max_coverage=30) is False

    def test_moderate_coverage_passes(self):
        row = self._make_row(Analyst_Coverage=12)
        assert find_gems._passes_filters(row, **self.KWARGS, max_coverage=30) is True

    def test_none_coverage_passes(self):
        """Missing coverage data should not reject (fail-open)."""
        row = self._make_row(Analyst_Coverage=None)
        assert find_gems._passes_filters(row, **self.KWARGS, max_coverage=30) is True

    def test_boundary_coverage_passes(self):
        """Exactly at threshold (30) should pass (> not >=)."""
        row = self._make_row(Analyst_Coverage=30)
        assert find_gems._passes_filters(row, **self.KWARGS, max_coverage=30) is True

    def test_disabled_when_zero(self):
        """max_coverage=0 should disable the check."""
        row = self._make_row(Analyst_Coverage=100)
        assert find_gems._passes_filters(row, **self.KWARGS, max_coverage=0) is True


# ============================================================
# TestOCFWaiver — Filter E
# ============================================================
class TestOCFWaiver:
    """Filter E: forward-PE recovery waiver for negative OCF."""

    KWARGS = {"max_pe": 18.0, "min_roe": 13.0, "min_roa": 6.0, "max_de": 150.0}

    @staticmethod
    def _make_row(**overrides):
        base = {
            "YF_Ticker": "TEST.T",
            "P/E": 16.0,
            "Forward_PE": None,
            "ROE": 0.15,
            "ROA": 0.08,
            "Debt_to_Equity": 80.0,
            "Operating_Cash_Flow": -500_000,
            "Net_Income": 800_000,
            "OCF_NI_Ratio": None,
            "Net_Debt_to_Equity": 50.0,
            "Total_Debt": 500_000,
            "Total_Cash": 200_000,
            "Revenue_Years_Positive": 4,
            "Analyst_Coverage": 5,
        }
        base.update(overrides)
        return base

    def test_negative_ocf_with_recovery_passes(self):
        """Negative OCF + forward PE < trailing PE and < 15 → waiver fires."""
        row = self._make_row(
            **{"P/E": 16.0},
            Forward_PE=10.0,
            Operating_Cash_Flow=-500_000,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is True

    def test_negative_ocf_no_forward_pe_fails(self):
        """Negative OCF + no forward PE data → strict fail."""
        row = self._make_row(Forward_PE=None, Operating_Cash_Flow=-500_000)
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is False

    def test_negative_ocf_forward_pe_too_high_fails(self):
        """Negative OCF + forward PE >= 15 → waiver doesn't fire."""
        row = self._make_row(
            **{"P/E": 18.0},
            Forward_PE=16.0,
            Operating_Cash_Flow=-500_000,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is False

    def test_negative_ocf_forward_pe_above_trailing_fails(self):
        """Negative OCF + forward PE >= trailing PE → no recovery signal."""
        row = self._make_row(
            **{"P/E": 10.0},
            Forward_PE=12.0,
            Operating_Cash_Flow=-500_000,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is False

    def test_waiver_disabled_via_flag(self):
        """With ocf_waiver=False, even valid recovery signals are rejected."""
        row = self._make_row(
            **{"P/E": 16.0},
            Forward_PE=10.0,
            Operating_Cash_Flow=-500_000,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=False) is False

    def test_positive_ocf_unaffected(self):
        """Positive OCF should pass regardless of waiver setting."""
        row = self._make_row(
            **{"P/E": 12.0},
            Operating_Cash_Flow=1_000_000,
            Net_Income=800_000,
            OCF_NI_Ratio=1.25,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is True
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=False) is True

    def test_zero_ocf_with_recovery_passes(self):
        """OCF == 0 + recovery signal → waiver fires."""
        row = self._make_row(
            **{"P/E": 16.0},
            Forward_PE=10.0,
            Operating_Cash_Flow=0,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is True

    def test_none_ocf_fails_even_with_waiver(self):
        """OCF is None (missing data) → fails; waiver only applies to actual negative/zero."""
        row = self._make_row(
            **{"P/E": 16.0},
            Forward_PE=10.0,
            Operating_Cash_Flow=None,
        )
        assert find_gems._passes_filters(row, **self.KWARGS, ocf_waiver=True) is False


# ============================================================
# TestExchangeScrapeIntegration — live exchange scraping
# ============================================================
def _load_enabled_exchanges():
    """Load enabled exchanges with min_expected_rows from config for parametrize."""
    if not _CONFIG_PATH.exists():
        return []
    with open(_CONFIG_PATH) as f:
        config = json.load(f)
    exchanges = []
    for ex in config["exchanges"]:
        if ex.get("enabled", True) and ex.get("min_expected_rows"):
            exchanges.append(ex)
    return exchanges


def _exchange_ids(exchanges):
    return [ex["exchange_name"] for ex in exchanges]


_ENABLED_EXCHANGES = _load_enabled_exchanges()


@pytest.mark.integration
@pytest.mark.slow
class TestExchangeScrapeIntegration:
    """Live integration tests: hit each exchange source and verify sane row counts.

    Catches broken URLs, changed HTML structure, misconfigured filters,
    and dead sources before they silently produce empty results.

    Run with: pytest tests/scripts/test_find_gems.py::TestExchangeScrapeIntegration -v
    """

    @pytest.fixture(autouse=True)
    def _session(self):
        self.session = find_gems._get_session()

    @pytest.mark.parametrize(
        "exchange", _ENABLED_EXCHANGES, ids=_exchange_ids(_ENABLED_EXCHANGES)
    )
    def test_exchange_returns_sane_count(self, exchange):
        handler = find_gems._HANDLERS.get(exchange["method"])
        assert handler is not None, f"Unknown method: {exchange['method']}"

        df = handler(exchange, self.session)
        assert df is not None, f"Handler returned None for {exchange['exchange_name']}"
        assert (
            not df.empty
        ), f"Handler returned empty DataFrame for {exchange['exchange_name']}"

        # Apply the same filters used by scrape_exchanges()
        df = find_gems._apply_filters(df, exchange)

        min_rows = exchange["min_expected_rows"]
        assert (
            len(df) >= min_rows
        ), f"{exchange['exchange_name']}: got {len(df)} rows, expected >= {min_rows}"

        # Verify the configured ticker column exists (pre-standardization name)
        ticker_col = exchange["params"].get("ticker_col")
        if ticker_col:
            actual = find_gems._find_col_fuzzy(df, ticker_col)
            assert actual is not None, (
                f"{exchange['exchange_name']}: ticker_col '{ticker_col}' not found "
                f"in columns {list(df.columns)}"
            )
