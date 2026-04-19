"""Tests for canonical macro- and sentiment-region derivation."""

from __future__ import annotations

import pytest

from src.exchange_metadata import EXCHANGES_BY_SUFFIX
from src.macro_regions import (
    MACRO_REGIONS,
    display_region_for_suffix,
    get_macro_region_info,
    infer_macro_region,
    infer_sentiment_region,
    query_hint_for_macro_region,
)


class TestMacroRegionMapping:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("7203.T", "JAPAN"),
            ("0005.HK", "HONG_KONG"),
            ("600519.SS", "CHINA"),
            ("000001.SZ", "CHINA"),
            ("2330.TW", "TAIWAN"),
            ("6488.TWO", "TAIWAN"),
            ("005930.KS", "KOREA"),
            ("RELIANCE.NS", "INDIA"),
            ("D05.SI", "SEA"),
            ("1155.KL", "SEA"),
            ("PTT.BK", "SEA"),
            ("BBCA.JK", "SEA"),
            ("CBA.AX", "AUSTRALIA"),
            ("FPH.NZ", "AUSTRALIA"),
            ("RY.TO", "CANADA"),
            ("SHOP.V", "CANADA"),
            ("AZN.L", "UK"),
            ("SAP.DE", "EUROPE"),
            ("MC.PA", "EUROPE"),
            ("ASML.AS", "EUROPE"),
            ("NOVN.SW", "EUROPE"),
            ("VOE.VI", "EUROPE"),
            ("VOLV-B.ST", "EUROPE"),
            ("NESTE.HE", "EUROPE"),
            ("NOVO-B.CO", "EUROPE"),
            ("VALE3.SA", "LATAM"),
            ("WALMEX.MX", "LATAM"),
            ("VNM.VN", "SEA"),
            ("SAP.XETRA", "EUROPE"),
            ("NOVN.S", "EUROPE"),
            ("ITX.MA", "EUROPE"),
            ("AAPL", "GLOBAL"),
            ("FOO.XX", "GLOBAL"),
        ],
    )
    def test_specific_tickers(self, ticker: str, expected: str):
        assert infer_macro_region(ticker) == expected

    @pytest.mark.parametrize("suffix", sorted(EXCHANGES_BY_SUFFIX))
    def test_every_known_suffix_resolves_to_a_valid_bucket(self, suffix: str):
        region = infer_macro_region(f"TEST{suffix}")
        assert region in MACRO_REGIONS

    def test_macro_region_info_uses_exchange_metadata(self):
        info = get_macro_region_info("7203.T")
        assert info.suffix == ".T"
        assert info.country == "Japan"
        assert info.macro_region == "JAPAN"
        assert info.display_region == "Japan"
        assert info.query_hint

    def test_display_region_for_suffix(self):
        assert display_region_for_suffix(".T") == "Japan"
        assert display_region_for_suffix(".HK") == "Hong Kong"
        assert display_region_for_suffix(".XX") == ""

    def test_query_hints_exist_for_all_macro_regions(self):
        for region in MACRO_REGIONS:
            assert query_hint_for_macro_region(region)


class TestSentimentRegionMapping:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("7203.T", "japan"),
            ("0005.HK", "hong_kong"),
            ("005930.KS", "south_korea"),
            ("2330.TW", "taiwan"),
            ("600519.SS", "china"),
            ("RELIANCE.NS", "india"),
            ("D05.SI", "singapore"),
            ("1155.KL", "malaysia"),
            ("PTT.BK", "thailand"),
            ("BBCA.JK", "indonesia"),
            ("CBA.AX", "australia"),
            ("RY.TO", "canada"),
            ("AZN.L", "uk"),
            ("SAP.DE", "germany"),
            ("MC.PA", "france"),
            ("VOE.VI", "germany"),
            ("NOVN.SW", "switzerland"),
            ("NOVO-B.CO", "denmark"),
            ("PKO.WA", "poland"),
            ("VALE3.SA", "brazil"),
            ("WALMEX.MX", "mexico"),
            ("VNM.VN", "vietnam"),
            ("2222.SR", "middle_east"),
            ("AAPL", "unknown"),
            ("FOO.XX", "unknown"),
        ],
    )
    def test_specific_tickers(self, ticker: str, expected: str):
        assert infer_sentiment_region(ticker) == expected

    def test_detect_market_region_remains_region_platform_compatible(self):
        from src.enhanced_sentiment_toolkit import (
            REGION_PLATFORMS,
            detect_market_region,
        )

        for ticker, expected in [
            ("7203.T", "japan"),
            ("0005.HK", "hong_kong"),
            ("D05.SI", "singapore"),
            ("1155.KL", "malaysia"),
            ("CBA.AX", "australia"),
            ("RY.TO", "canada"),
            ("AZN.L", "uk"),
        ]:
            region = detect_market_region(ticker)
            assert region == expected
            assert region in REGION_PLATFORMS
