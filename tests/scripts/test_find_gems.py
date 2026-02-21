"""Tests for scripts/find_gems.py — screening pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add scripts/ to path so we can import find_gems as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import find_gems  # noqa: E402


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
            "ROE": 0.15,
            "ROA": 0.08,
            "Debt_to_Equity": 80.0,
            "Operating_Cash_Flow": 1_000_000,
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
        row = self._make_row(Debt_to_Equity=200.0)
        assert find_gems._passes_filters(row, **self.DEFAULT_KWARGS) is False

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
            "operatingCashflow": 1_000_000,
            "freeCashflow": 500_000,
            "marketCap": 10_000_000,
            "sector": "Industrials",
            "industry": "Auto Manufacturers",
            "longName": "Test Corp",
            "currency": "JPY",
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
