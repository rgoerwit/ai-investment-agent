"""Tests for IBKR <-> yfinance ticker mapping."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.ibkr.exceptions import IBKRTickerResolutionError
from src.ibkr.ticker_mapper import (
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
        # Unknown exchange → symbol only (no suffix)
        assert ibkr_symbol_to_yf("WEIRD", "UNKNOWN") == "WEIRD"


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
