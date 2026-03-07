"""Tests for IBKR <-> yfinance ticker mapping."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.ibkr.exceptions import IBKRTickerResolutionError
from src.ibkr.ticker_mapper import (
    _yf_search_ticker,
    ibkr_symbol_to_yf,
    parse_trade_block_price,
    resolve_conid,
    resolve_yf_ticker_from_position,
    yf_to_ibkr_format,
)


class TestIbkrSymbolToYf:
    """Test IBKR symbol+exchange → yfinance ticker conversion."""

    def test_hong_kong_zero_padding(self):
        assert ibkr_symbol_to_yf("5", "SEHK") == "0005.HK"

    def test_hong_kong_already_padded(self):
        assert ibkr_symbol_to_yf("0005", "SEHK") == "0005.HK"

    def test_hong_kong_three_digit(self):
        assert ibkr_symbol_to_yf("388", "SEHK") == "0388.HK"

    def test_hong_kong_four_digit(self):
        assert ibkr_symbol_to_yf("3600", "SEHK") == "3600.HK"

    def test_tokyo(self):
        assert ibkr_symbol_to_yf("7203", "TSE") == "7203.T"

    def test_amsterdam(self):
        assert ibkr_symbol_to_yf("ASML", "AEB") == "ASML.AS"

    def test_london(self):
        assert ibkr_symbol_to_yf("HSBA", "LSE") == "HSBA.L"

    def test_xetra(self):
        assert ibkr_symbol_to_yf("SAP", "IBIS") == "SAP.DE"

    def test_paris(self):
        assert ibkr_symbol_to_yf("MC", "SBF") == "MC.PA"

    def test_korea(self):
        assert ibkr_symbol_to_yf("005930", "KRX") == "005930.KS"

    def test_taiwan(self):
        assert ibkr_symbol_to_yf("2330", "TWSE") == "2330.TW"

    def test_singapore(self):
        assert ibkr_symbol_to_yf("D05", "SGX") == "D05.SI"

    def test_us_smart(self):
        assert ibkr_symbol_to_yf("AAPL", "SMART") == "AAPL"

    def test_us_nasdaq(self):
        assert ibkr_symbol_to_yf("MSFT", "NASDAQ") == "MSFT"

    def test_swiss(self):
        assert ibkr_symbol_to_yf("NOVN", "SWX") == "NOVN.SW"

    def test_australia(self):
        assert ibkr_symbol_to_yf("BHP", "ASX") == "BHP.AX"

    def test_brazil(self):
        assert ibkr_symbol_to_yf("VALE3", "BVMF") == "VALE3.SA"

    def test_unknown_exchange(self):
        # Unknown exchange with no search result → bare symbol returned
        with patch("src.ibkr.ticker_mapper._yf_search_ticker", return_value=""):
            assert ibkr_symbol_to_yf("WEIRD", "UNKNOWN") == "WEIRD"

    # --- Currency fallback tests ---

    def test_currency_fallback_hkd(self):
        # Unknown exchange + HKD → .HK suffix
        assert ibkr_symbol_to_yf("1681", "UNKNOWN", "HKD") == "1681.HK"

    def test_currency_fallback_jpy(self):
        # Empty exchange + JPY → .T suffix
        assert ibkr_symbol_to_yf("7203", "", "JPY") == "7203.T"

    def test_currency_fallback_hk_zero_padding(self):
        # Currency fallback also applies HK zero-padding
        assert ibkr_symbol_to_yf("5", "", "HKD") == "0005.HK"

    def test_currency_fallback_ambiguous_eur_no_change(self):
        # EUR is deliberately omitted from _CURRENCY_TO_SUFFIX (multi-country).
        # When yfinance.Search also finds nothing, bare symbol is returned.
        with patch("src.ibkr.ticker_mapper._yf_search_ticker", return_value=""):
            assert ibkr_symbol_to_yf("ASML", "UNKNOWN", "EUR") == "ASML"

    def test_exchange_takes_precedence_over_currency(self):
        # Known exchange wins even when currency would suggest different suffix
        assert ibkr_symbol_to_yf("5", "SEHK", "USD") == "0005.HK"

    def test_currency_fallback_myr(self):
        # Unknown exchange + MYR → Bursa Malaysia .KL suffix
        assert ibkr_symbol_to_yf("PADINI", "UNKNOWN", "MYR") == "PADINI.KL"

    def test_currency_fallback_myr_scientx(self):
        # Another MYR stock; exchange absent
        assert ibkr_symbol_to_yf("SCIENTX", "", "MYR") == "SCIENTX.KL"

    def test_madrid_sibe_exchange(self):
        # SIBE is IBKR's code for Bolsa Madrid electronic order book
        assert ibkr_symbol_to_yf("VID", "SIBE") == "VID.MC"

    def test_madrid_bm_exchange_unchanged(self):
        # BM mapping must remain intact
        assert ibkr_symbol_to_yf("ITX", "BM") == "ITX.MC"


class TestYfSearchTicker:
    """Tests for the yfinance.Search fallback resolver."""

    def _make_quote(
        self, symbol: str, exchange: str, quote_type: str = "EQUITY"
    ) -> dict:
        return {"symbol": symbol, "exchange": exchange, "quoteType": quote_type}

    def _mock_search(self, quotes: list[dict]):
        mock = MagicMock()
        mock.quotes = quotes
        return mock

    def test_resolves_by_currency_suffix(self):
        """PLN currency → prefers .WA result over other equities."""
        quotes = [
            self._make_quote("KTY.WA", "WSE"),
            self._make_quote("KTYFOO", "GER"),
        ]
        with patch("yfinance.Search", return_value=self._mock_search(quotes)):
            with patch("src.ibkr.ticker_mapper._get_cache", return_value={}):
                with patch("src.ibkr.ticker_mapper._flush_cache"):
                    assert _yf_search_ticker("KTY", "NEWEXCH", "PLN") == "KTY.WA"

    def test_best_guess_for_ambiguous_currency(self):
        """EUR (ambiguous) → returns first non-OTC equity."""
        quotes = [
            self._make_quote("ANDR", "PNK"),  # filtered out (OTC)
            self._make_quote("ANDR.VI", "VIE"),  # first non-OTC equity
            self._make_quote("AZ2.DE", "GER"),
        ]
        with patch("yfinance.Search", return_value=self._mock_search(quotes)):
            with patch("src.ibkr.ticker_mapper._get_cache", return_value={}):
                with patch("src.ibkr.ticker_mapper._flush_cache"):
                    assert _yf_search_ticker("ANDR", "VSE", "EUR") == "ANDR.VI"

    def test_returns_empty_when_only_otc(self):
        """All results are OTC → returns empty string."""
        quotes = [self._make_quote("FOO", "PNK"), self._make_quote("FOO", "OTC")]
        with patch("yfinance.Search", return_value=self._mock_search(quotes)):
            with patch("src.ibkr.ticker_mapper._get_cache", return_value={}):
                with patch("src.ibkr.ticker_mapper._flush_cache"):
                    assert _yf_search_ticker("FOO", "WEIRD", "USD") == ""

    def test_returns_empty_on_exception(self):
        """Exception during Search → returns empty string (graceful degradation)."""
        with patch("yfinance.Search", side_effect=Exception("network error")):
            with patch("src.ibkr.ticker_mapper._get_cache", return_value={}):
                assert _yf_search_ticker("FOO", "WEIRD", "EUR") == ""

    def test_cache_hit_returns_cached(self):
        """Cache hit skips the network call entirely."""
        cache = {"ibkr:KTY:WSE": {"yf_ticker": "KTY.WA", "ts": time.time()}}
        with patch("src.ibkr.ticker_mapper._get_cache", return_value=cache):
            assert _yf_search_ticker("KTY", "WSE", "PLN") == "KTY.WA"

    def test_negative_cache_skips_search(self):
        """Negative cache entry (empty yf_ticker) prevents repeat search."""
        cache = {"ibkr:NOMATCH:WEIRD": {"yf_ticker": "", "ts": time.time()}}
        with patch("src.ibkr.ticker_mapper._get_cache", return_value=cache):
            with patch("yfinance.Search") as mock_search:
                result = _yf_search_ticker("NOMATCH", "WEIRD", "EUR")
        assert result == ""
        mock_search.assert_not_called()

    def test_ibkr_symbol_to_yf_uses_search_fallback(self):
        """ibkr_symbol_to_yf calls _yf_search_ticker for unknown exchanges."""
        with patch(
            "src.ibkr.ticker_mapper._yf_search_ticker", return_value="ANDR.VI"
        ) as mock_fn:
            result = ibkr_symbol_to_yf("ANDR", "NEWEXCH", "EUR")
        assert result == "ANDR.VI"
        mock_fn.assert_called_once_with("ANDR", "NEWEXCH", "EUR")

    def test_filters_non_equity_quote_types(self):
        """FUTURE and MUTUALFUND results are filtered out."""
        quotes = [
            self._make_quote("CL=F", "NYM", quote_type="FUTURE"),
            self._make_quote("APR.WA", "WSE", quote_type="EQUITY"),
        ]
        with patch("yfinance.Search", return_value=self._mock_search(quotes)):
            with patch("src.ibkr.ticker_mapper._get_cache", return_value={}):
                with patch("src.ibkr.ticker_mapper._flush_cache"):
                    assert _yf_search_ticker("APR", "NEWEXCH", "PLN") == "APR.WA"


class TestYfToIbkrFormat:
    """Test yfinance ticker → IBKR symbol+exchange conversion."""

    def test_hong_kong(self):
        symbol, exchange = yf_to_ibkr_format("0005.HK")
        assert exchange == "SEHK"
        assert symbol == "0005"

    def test_tokyo(self):
        symbol, exchange = yf_to_ibkr_format("7203.T")
        assert exchange == "TSE"
        assert symbol == "7203"

    def test_amsterdam(self):
        symbol, exchange = yf_to_ibkr_format("ASML.AS")
        assert exchange == "AEB"

    def test_us_plain(self):
        symbol, exchange = yf_to_ibkr_format("AAPL")
        assert exchange == "SMART"
        assert symbol == "AAPL"

    def test_london(self):
        symbol, exchange = yf_to_ibkr_format("HSBA.L")
        assert exchange == "LSE"


class TestResolveConid:
    """Test conid resolution with cache and API mocking."""

    def setup_method(self):
        """Reset module-level session cache before each test."""
        import src.ibkr.ticker_mapper as tm

        tm._cache = None

    def test_returns_none_without_client(self):
        result = resolve_conid("7203.T", client=None)
        assert result is None

    @patch("src.ibkr.ticker_mapper._load_cache")
    def test_cache_hit(self, mock_cache):
        mock_cache.return_value = {
            "7203.T": {
                "conid": 123456,
                "symbol": "7203",
                "exchange": "TSE",
                "ts": time.time(),
            }
        }
        result = resolve_conid("7203.T", client=None)
        assert result == 123456

    @patch("src.ibkr.ticker_mapper._save_cache")
    @patch("src.ibkr.ticker_mapper._load_cache", return_value={})
    def test_api_resolution(self, mock_load, mock_save):
        mock_client = MagicMock()
        mock_client.stock_conid_by_symbol.return_value = {
            "7203": [{"conid": 123456, "exchange": "TSE"}]
        }
        result = resolve_conid("7203.T", client=mock_client)
        assert result == 123456
        mock_save.assert_called_once()

    @patch("src.ibkr.ticker_mapper._load_cache", return_value={})
    def test_api_no_results_raises(self, mock_load):
        mock_client = MagicMock()
        mock_client.stock_conid_by_symbol.return_value = {}
        with pytest.raises(IBKRTickerResolutionError):
            resolve_conid("FAKE.XX", client=mock_client)

    @patch("src.ibkr.ticker_mapper._save_cache")
    @patch("src.ibkr.ticker_mapper._load_cache", return_value={})
    def test_exchange_match_preferred(self, mock_load, mock_save):
        mock_client = MagicMock()
        mock_client.stock_conid_by_symbol.return_value = {
            "ASML": [
                {"conid": 111, "exchange": "SMART"},
                {"conid": 222, "exchange": "AEB"},
            ]
        }
        result = resolve_conid("ASML.AS", client=mock_client)
        assert result == 222  # AEB match preferred over SMART


class TestResolveYfTickerFromPosition:
    """Test converting IBKR position dicts to yfinance tickers."""

    def test_standard_position(self):
        pos = {"contractDesc": "7203", "listingExchange": "TSE"}
        assert resolve_yf_ticker_from_position(pos) == "7203.T"

    def test_hk_position(self):
        pos = {"contractDesc": "5", "listingExchange": "SEHK"}
        assert resolve_yf_ticker_from_position(pos) == "0005.HK"

    def test_hyphenated_symbol(self):
        pos = {"contractDesc": "7203-TSE", "listingExchange": ""}
        assert resolve_yf_ticker_from_position(pos) == "7203.T"

    def test_ticker_field_fallback(self):
        pos = {"ticker": "AAPL", "exchange": "SMART"}
        assert resolve_yf_ticker_from_position(pos) == "AAPL"

    def test_empty_position(self):
        assert resolve_yf_ticker_from_position({}) == ""

    def test_currency_fallback_for_missing_exchange(self):
        # listingExchange absent but currency present → suffix derived from currency
        pos = {"contractDesc": "1681", "listingExchange": "", "currency": "HKD"}
        assert resolve_yf_ticker_from_position(pos) == "1681.HK"


class TestParseTradeBlockPrice:
    """Test TRADE_BLOCK price parsing."""

    def test_simple_price(self):
        assert parse_trade_block_price("436.00 (Limit)") == 436.0

    def test_comma_price(self):
        assert parse_trade_block_price("2,145 (Scaled Limit)") == 2145.0

    def test_stop_price_with_pct(self):
        assert parse_trade_block_price("1,930 (-10.0%)") == 1930.0

    def test_target_price_with_pct(self):
        assert parse_trade_block_price("2,575 (+20.0%)") == 2575.0

    def test_na(self):
        assert parse_trade_block_price("N/A") is None

    def test_na_with_reason(self):
        assert parse_trade_block_price("N/A (Liquidity Fail)") is None

    def test_empty(self):
        assert parse_trade_block_price("") is None

    def test_none(self):
        assert parse_trade_block_price(None) is None

    def test_integer_price(self):
        assert parse_trade_block_price("58") == 58.0

    def test_large_price(self):
        assert parse_trade_block_price("12,345.67 (GTC)") == 12345.67
