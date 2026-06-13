"""Unit coverage for src/ticker_policy.py — exchange-qualified ticker rules."""

from __future__ import annotations

import pytest

from src.ticker_policy import (
    CHINA_SUFFIXES,
    FRAGILE_EXCHANGE_SUFFIXES,
    KOREA_SUFFIXES,
    allows_search_resolution,
    get_ticker_suffix,
    is_pure_numeric_base,
    is_safe_symbol_crossmatch_base,
    same_exchange,
    sibling_ticker_candidates,
    split_ticker,
    ticker_in_group,
)


class TestSuffixExtraction:
    @pytest.mark.parametrize(
        ("ticker", "suffix"),
        [
            ("7203.T", ".T"),
            ("0005.HK", ".HK"),
            ("2330.TW", ".TW"),
            ("005930.KS", ".KS"),
            ("ASML.AS", ".AS"),
            ("MEGP.L", ".L"),
        ],
    )
    def test_known_exchange_suffixes(self, ticker, suffix):
        assert get_ticker_suffix(ticker) == suffix

    def test_no_suffix_returns_empty(self):
        assert get_ticker_suffix("AAPL") == ""

    def test_lowercase_and_whitespace_normalized(self):
        assert get_ticker_suffix("  7203.t ") == ".T"

    def test_empty_string(self):
        assert get_ticker_suffix("") == ""

    def test_multi_dot_uses_final_segment(self):
        assert get_ticker_suffix("BRK.B.US") == ".US"


class TestSplitTicker:
    def test_split_with_suffix(self):
        assert split_ticker("0005.HK") == ("0005", ".HK")

    def test_split_without_suffix(self):
        assert split_ticker("aapl") == ("AAPL", "")

    def test_split_empty(self):
        assert split_ticker("") == ("", "")


class TestSiblingCandidates:
    def test_tw_sibling_is_two(self):
        assert sibling_ticker_candidates("2330.TW") == ("2330.TWO",)

    def test_two_sibling_is_tw(self):
        assert sibling_ticker_candidates("6488.TWO") == ("6488.TW",)

    def test_no_siblings_for_other_exchanges(self):
        assert sibling_ticker_candidates("7203.T") == ()
        assert sibling_ticker_candidates("AAPL") == ()


class TestBaseSymbolPolicies:
    def test_pure_numeric_base(self):
        assert is_pure_numeric_base("0005") is True
        assert is_pure_numeric_base("ASML") is False
        assert is_pure_numeric_base("") is False

    def test_crossmatch_safety_excludes_numeric_bases(self):
        # Numeric bases clash across exchanges (e.g., 0005.HK vs 0005.T)
        assert is_safe_symbol_crossmatch_base("MEGP") is True
        assert is_safe_symbol_crossmatch_base("0005") is False
        assert is_safe_symbol_crossmatch_base("") is False


class TestGroupingPredicates:
    def test_allows_search_resolution(self):
        assert allows_search_resolution("0005.HK") is True
        assert allows_search_resolution("7203.T") is False
        assert allows_search_resolution("AAPL") is False

    def test_same_exchange(self):
        assert same_exchange("0005.HK", "0700.HK") is True
        assert same_exchange("0005.HK", "7203.T") is False
        assert same_exchange("AAPL", "MSFT") is True  # both suffix-less

    def test_ticker_in_group(self):
        assert ticker_in_group("005930.KS", KOREA_SUFFIXES) is True
        assert ticker_in_group("600519.SS", CHINA_SUFFIXES) is True
        assert ticker_in_group("ASML.AS", FRAGILE_EXCHANGE_SUFFIXES) is False
