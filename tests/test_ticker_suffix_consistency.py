"""
Ticker suffix consistency tests.

These tests enforce the canonical exchange metadata invariants and the key
cross-module behaviors that have historically drifted out of sync.
"""

import importlib
import re

import pytest

from src.exchange_metadata import (
    EXCHANGES_BY_SUFFIX,
    IBKR_EXCHANGE_ALIASES,
    IBKR_TO_YFINANCE,
)
from src.ticker_policy import FRAGILE_EXCHANGE_SUFFIXES


class TestIbkrMapConsistency:
    def test_all_ibkr_keys_are_uppercase(self):
        bad = [key for key in IBKR_TO_YFINANCE if key != key.upper()]
        assert bad == [], (
            f"IBKR_TO_YFINANCE has non-uppercase keys: {bad}. "
            "Ticker.from_ibkr() uppercases the exchange code before lookup."
        )

    def test_canonical_ibkr_codes_round_trip(self):
        mismatches = []
        for suffix, info in EXCHANGES_BY_SUFFIX.items():
            mapped_suffix = IBKR_TO_YFINANCE.get(info.ibkr_code)
            if mapped_suffix != suffix:
                mismatches.append(
                    f"{info.ibkr_code}: maps to {mapped_suffix!r}, expected {suffix!r}"
                )
        assert mismatches == [], "\n".join(mismatches)

    def test_aliases_reference_valid_suffixes(self):
        bad = {
            alias: suffix
            for alias, suffix in IBKR_EXCHANGE_ALIASES.items()
            if suffix not in EXCHANGES_BY_SUFFIX
        }
        assert bad == {}, f"IBKR aliases point to unknown suffixes: {bad}"


class TestTickerNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3217.TWO", "3217.TWO"),
            ("2330.TW", "2330.TW"),
            ("7203.T", "7203.T"),
            ("0005.HK", "0005.HK"),
            ("005930.KS", "005930.KS"),
            ("005930.KQ", "005930.KQ"),
            ("1234.SS", "1234.SS"),
            ("1234.SZ", "1234.SZ"),
            ("NESN.SWX", "NESN.SW"),
            ("NESN.VX", "NESN.SW"),
        ],
    )
    def test_normalize_preserves_or_resolves_expected_suffix(self, raw, expected):
        from src.ticker_utils import normalize_ticker

        assert normalize_ticker(raw) == expected

    def test_two_and_tw_are_distinct(self):
        from src.ticker_utils import normalize_ticker

        assert normalize_ticker("3217.TWO") != "3217.TW"
        assert normalize_ticker("2330.TW") != "2330.TWO"


class TestIbkrRoundTrip:
    @pytest.mark.parametrize(
        ("symbol", "exchange", "expected_yf"),
        [
            ("3217", "TPEX", "3217.TWO"),
            ("3217", "TPEx", "3217.TWO"),
            ("2330", "TWSE", "2330.TW"),
            ("7203", "TSE", "7203.T"),
            ("5", "SEHK", "0005.HK"),
            ("5934", "KRX", "5934.KS"),
            ("SAP", "IBIS2", "SAP.DE"),
            ("ANDR", "VSE", "ANDR.VI"),
        ],
    )
    def test_from_ibkr_produces_correct_yf(self, symbol, exchange, expected_yf):
        from src.ibkr.ticker import Ticker

        ticker = Ticker.from_ibkr(symbol=symbol, exchange=exchange)
        assert ticker.yf == expected_yf


class TestRetrospectiveMapCoverage:
    def test_benchmark_keys_are_subset_of_currency_keys(self):
        from src.retrospective import EXCHANGE_BENCHMARK, EXCHANGE_CURRENCY

        missing = set(EXCHANGE_BENCHMARK) - set(EXCHANGE_CURRENCY)
        assert not missing, (
            "Every benchmarked exchange must have a matching currency entry: "
            f"{sorted(missing)}"
        )

    def test_taiwan_otc_benchmark_and_currency_exist(self):
        from src.retrospective import EXCHANGE_BENCHMARK, EXCHANGE_CURRENCY

        assert EXCHANGE_BENCHMARK[".TWO"] == "^TWII"
        assert EXCHANGE_CURRENCY[".TWO"] == "TWD"

    def test_korea_and_china_suffixes_covered(self):
        from src.retrospective import EXCHANGE_CURRENCY

        for suffix in (".KS", ".KQ", ".SS", ".SZ"):
            assert suffix in EXCHANGE_CURRENCY


class TestFragileExchangeList:
    def test_tw_and_two_both_fragile(self):
        assert ".TW" in FRAGILE_EXCHANGE_SUFFIXES
        assert ".TWO" in FRAGILE_EXCHANGE_SUFFIXES

    def test_hk_and_japan_fragile(self):
        assert ".HK" in FRAGILE_EXCHANGE_SUFFIXES
        assert ".T" in FRAGILE_EXCHANGE_SUFFIXES

    def test_korea_fragile(self):
        assert ".KS" in FRAGILE_EXCHANGE_SUFFIXES


_TICKER_TOOLS = [
    ("src.tools.market", "get_financial_metrics"),
    ("src.tools.market", "get_yfinance_data"),
    ("src.tools.market", "get_technical_indicators"),
    ("src.tools.market", "get_fundamental_analysis"),
    ("src.tools.news", "get_news"),
]
_SUFFIX_PATTERN = re.compile(r"\.\w+")


class TestToolAnnotations:
    @pytest.mark.parametrize("module_path,tool_name", _TICKER_TOOLS)
    def test_ticker_annotation_mentions_exchange_suffix(self, module_path, tool_name):
        mod = importlib.import_module(module_path)
        fn = getattr(mod, tool_name)
        orig = getattr(fn, "func", fn)
        try:
            import typing

            hints = typing.get_type_hints(orig, include_extras=True)
        except Exception:
            pytest.skip(f"Could not get type hints for {tool_name}")

        ticker_param = next(
            (hint for key, hint in hints.items() if key in ("ticker", "symbol")), None
        )
        assert ticker_param is not None

        metadata = getattr(ticker_param, "__metadata__", ())
        assert metadata
        annotation_text = " ".join(str(item) for item in metadata)
        assert _SUFFIX_PATTERN.search(annotation_text)

    @pytest.mark.parametrize("module_path,tool_name", _TICKER_TOOLS)
    def test_ticker_annotation_mentions_two(self, module_path, tool_name):
        mod = importlib.import_module(module_path)
        fn = getattr(mod, tool_name)
        orig = getattr(fn, "func", fn)
        try:
            import typing

            hints = typing.get_type_hints(orig, include_extras=True)
        except Exception:
            pytest.skip(f"Could not get type hints for {tool_name}")

        ticker_param = next(
            (hint for key, hint in hints.items() if key in ("ticker", "symbol")), None
        )
        if ticker_param is None:
            return
        metadata = getattr(ticker_param, "__metadata__", ())
        annotation_text = " ".join(str(item) for item in metadata)
        assert ".TWO" in annotation_text


class TestTaiwanSuffixRegression:
    def test_canonical_metadata_has_both_tw_and_two(self):
        assert ".TW" in EXCHANGES_BY_SUFFIX
        assert ".TWO" in EXCHANGES_BY_SUFFIX

    def test_tw_and_two_have_distinct_ibkr_codes(self):
        assert (
            EXCHANGES_BY_SUFFIX[".TW"].ibkr_code
            != EXCHANGES_BY_SUFFIX[".TWO"].ibkr_code
        )

    def test_tw_and_two_have_distinct_suffixes(self):
        assert EXCHANGES_BY_SUFFIX[".TW"].yf_suffix == ".TW"
        assert EXCHANGES_BY_SUFFIX[".TWO"].yf_suffix == ".TWO"

    def test_ibkr_to_yfinance_has_both_twse_and_tpex(self):
        assert IBKR_TO_YFINANCE["TWSE"] == ".TW"
        assert IBKR_TO_YFINANCE["TPEX"] == ".TWO"

    def test_tpex_key_is_uppercase_not_mixed(self):
        assert "TPEx" not in IBKR_TO_YFINANCE

    def test_normalize_3217_two_stays_two(self):
        from src.ticker_utils import normalize_ticker

        assert normalize_ticker("3217.TWO") == "3217.TWO"

    def test_ibkr_ticker_tpex_produces_two_suffix(self):
        from src.ibkr.ticker import Ticker

        assert Ticker.from_ibkr(symbol="3217", exchange="TPEX").yf == "3217.TWO"

    def test_ibkr_ticker_twse_does_not_produce_two_suffix(self):
        from src.ibkr.ticker import Ticker

        ticker = Ticker.from_ibkr(symbol="2330", exchange="TWSE")
        assert ticker.yf == "2330.TW"
        assert not ticker.yf.endswith(".TWO")
