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
    ExchangeInfo,
    format_ibkr_symbol,
    format_yahoo_symbol,
    padded_numeric_suffixes,
    validate_exchange_metadata,
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

    # Ground truth: IBKR Client Portal exchange codes, human-verified against live
    # `conid_resolved` logs. Hardcoded on purpose — NOT derived from the metadata —
    # so a silent swap (e.g. Tokyo/Toronto) fails here even though it stays
    # self-consistent and passes test_canonical_ibkr_codes_round_trip above.
    # Tokyo=TSEJ, Toronto=TSE (these were swapped with .T=TSE / .TO=TSX historically).
    GROUND_TRUTH_IBKR_CODES = {
        ".T": "TSEJ",
        ".TO": "TSE",
        ".V": "VENTURE",
        ".HK": "SEHK",
        ".KS": "KRX",
        ".KQ": "KOSDAQ",
        ".TW": "TWSE",
        ".TWO": "TPEX",
        ".L": "LSE",
        ".AX": "ASX",
        ".SI": "SGX",
        ".KL": "KLSE",
        ".MX": "MEXI",
        ".SA": "BVMF",
        ".WA": "WSE",
        ".AS": "AEB",
        ".PA": "SBF",
        ".DE": "IBIS",
        ".BR": "EBR",
        ".SW": "SWX",
    }

    def test_ground_truth_ibkr_codes(self):
        for suffix, expected in self.GROUND_TRUTH_IBKR_CODES.items():
            assert suffix in EXCHANGES_BY_SUFFIX, f"canonical suffix {suffix} removed"
            actual = EXCHANGES_BY_SUFFIX[suffix].ibkr_code
            assert actual == expected, (
                f"{suffix}: ibkr_code is {actual!r}, expected {expected!r} "
                "(IBKR Client Portal ground truth — Tokyo=TSEJ, Toronto=TSE)"
            )


class TestExchangeMetadataValidation:
    def test_rejects_non_positive_numeric_width(self, monkeypatch):
        monkeypatch.setitem(
            EXCHANGES_BY_SUFFIX,
            ".BAD",
            ExchangeInfo(".BAD", "Bad Exchange", "Nowhere", "BADX", "BAD", 0),
        )
        with pytest.raises(ValueError, match="Numeric symbol width"):
            validate_exchange_metadata()

    def test_rejects_strip_mode_without_width(self, monkeypatch):
        monkeypatch.setitem(
            EXCHANGES_BY_SUFFIX,
            ".BAD",
            ExchangeInfo(
                ".BAD",
                "Bad Exchange",
                "Nowhere",
                "BADX",
                "BAD",
                ibkr_numeric_symbol_mode="strip_leading_zeroes",
            ),
        )
        with pytest.raises(ValueError, match="strip mode requires numeric width"):
            validate_exchange_metadata()


class TestTickerNormalization:
    def test_metadata_formats_verified_numeric_symbol_cases(self):
        assert format_yahoo_symbol("5", ".HK") == "0005"
        assert format_ibkr_symbol("0005", ".HK") == "5"
        assert format_yahoo_symbol("5930", ".KS") == "005930"
        assert format_ibkr_symbol("5930", ".KS") == "005930"
        assert format_yahoo_symbol("35420", ".KQ") == "035420"
        assert format_ibkr_symbol("35420", ".KQ") == "035420"

    def test_metadata_leaves_mixed_and_unspecified_symbols_unchanged(self):
        assert format_yahoo_symbol("ABC123", ".KS") == "ABC123"
        assert format_ibkr_symbol("ABC123", ".KS") == "ABC123"
        assert format_yahoo_symbol("ASML", ".AS") == "ASML"
        assert format_ibkr_symbol("ASML", ".AS") == "ASML"

    def test_padded_numeric_suffixes_match_metadata(self):
        # Property test: the helper's output must agree with the underlying
        # metadata. Hard-coding the expected set would force an unrelated edit
        # every time a new padded exchange (e.g. China) is enabled.
        derived = padded_numeric_suffixes()
        from_metadata = {
            suffix
            for suffix, info in EXCHANGES_BY_SUFFIX.items()
            if info.numeric_symbol_width is not None
        }
        assert derived == from_metadata
        assert derived  # at minimum HK/KS/KQ are populated today

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
            ("7203", "TSEJ", "7203.T"),  # Tokyo: IBKR Client Portal code is TSEJ
            (
                "PEY",
                "TSE",
                "PEY.TO",
            ),  # Toronto: IBKR Client Portal code is TSE (not TSEJ)
            ("5", "SEHK", "0005.HK"),
            ("5934", "KRX", "005934.KS"),
            ("005930", "KRX", "005930.KS"),
            ("10130", "KRX", "010130.KS"),
            ("001060", "KRX", "001060.KS"),
            ("5930", "KSE", "005930.KS"),
            ("35420", "KOSDAQ", "035420.KQ"),
            ("SAP", "IBIS2", "SAP.DE"),
            ("ANDR", "VSE", "ANDR.VI"),
        ],
    )
    def test_from_ibkr_produces_correct_yf(self, symbol, exchange, expected_yf):
        from src.ibkr.ticker import Ticker

        ticker = Ticker.from_ibkr(symbol=symbol, exchange=exchange)
        assert ticker.yf == expected_yf

    def test_krw_currency_fallback_defaults_to_ks(self):
        from src.ibkr.ticker import Ticker

        ticker = Ticker.from_ibkr(symbol="5930", exchange="", currency="KRW")
        assert ticker.yf == "005930.KS"


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

    def test_brazil_benchmark_and_currency_exist(self):
        from src.retrospective import EXCHANGE_BENCHMARK, EXCHANGE_CURRENCY

        assert EXCHANGE_BENCHMARK[".SA"] == "^BVSP"
        assert EXCHANGE_CURRENCY[".SA"] == "BRL"

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

    def test_canada_fragile(self):
        # Thin yfinance coverage for Canadian listings (esp. TSXV); warrants the
        # same Panic-Mode/sibling-rescue caution as JP/HK/KR/TW/UK.
        assert ".TO" in FRAGILE_EXCHANGE_SUFFIXES
        assert ".V" in FRAGILE_EXCHANGE_SUFFIXES


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
