"""
Ticker suffix consistency tests.

Enforces that every exchange suffix in EXCHANGE_SUFFIXES is fully handled
across all relevant lookup tables in the codebase.  Any new exchange added to
EXCHANGE_SUFFIXES that is missing from one of the downstream tables will fail
here immediately, preventing the class of .TWO/.TW confusion bugs.

Tests are organised into:
  - TickerNormalizationTests      : normalize_ticker correctness
  - IbkrRoundTripTests            : Ticker.from_ibkr ↔ .yf round-trips
  - IbkrMapConsistencyTests       : IBKR_TO_YFINANCE key/value invariants
  - RetrospectiveMapTests         : EXCHANGE_BENCHMARK / EXCHANGE_CURRENCY coverage
  - FragileExchangeTests          : fetcher fragile-exchange list coverage
  - ToolAnnotationTests           : tool parameter descriptions are non-trivial
  - TaiwanRegressionTests         : explicit .TWO/.TW non-confusion tests
"""

import importlib
import inspect
import re

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _exchange_suffixes():
    """Return the canonical EXCHANGE_SUFFIXES dict from ticker_utils."""
    from src.ticker_utils import TickerFormatter

    return TickerFormatter.EXCHANGE_SUFFIXES


def _ibkr_to_yfinance():
    from src.ticker_utils import TickerFormatter

    return TickerFormatter.IBKR_TO_YFINANCE


# ── IBKR map consistency ──────────────────────────────────────────────────────


class TestIbkrMapConsistency:
    """IBKR_TO_YFINANCE structural invariants."""

    def test_all_ibkr_keys_are_uppercase(self):
        """All IBKR_TO_YFINANCE keys must be uppercase.

        Ticker.from_ibkr() calls .upper() on the exchange code before looking up
        this dict.  A mixed-case key (e.g. 'TPEx') will silently miss the lookup
        and fall back to the currency-based suffix, producing the wrong exchange.
        """
        bad = [k for k in _ibkr_to_yfinance() if k != k.upper()]
        assert bad == [], (
            f"IBKR_TO_YFINANCE has non-uppercase keys: {bad}. "
            "Ticker.from_ibkr() uppercases the exchange code before lookup — "
            "mixed-case keys are never matched."
        )

    def test_exchange_suffixes_ibkr_codes_match_ibkr_to_yfinance(self):
        """Every ibkr_exchange code stored in EXCHANGE_SUFFIXES must have a
        corresponding IBKR_TO_YFINANCE entry that maps back to the same suffix.

        This is the bridge between the two tables.  A mismatch means
        normalize_ticker("X.EXT") → IBKR format produces a code that
        Ticker.from_ibkr() cannot reverse.
        """
        ibkr_map = _ibkr_to_yfinance()
        mismatches = []
        for suffix_key, (
            yf_suffix,
            _exch_name,
            _country,
            ibkr_code,
        ) in _exchange_suffixes().items():
            if not ibkr_code:
                continue
            mapped_suffix = ibkr_map.get(ibkr_code)
            if mapped_suffix is None:
                mismatches.append(
                    f".{suffix_key}: ibkr_code '{ibkr_code}' has no entry in IBKR_TO_YFINANCE"
                )
            elif mapped_suffix != yf_suffix:
                mismatches.append(
                    f".{suffix_key}: ibkr_code '{ibkr_code}' maps to '{mapped_suffix}' "
                    f"in IBKR_TO_YFINANCE, expected '{yf_suffix}'"
                )
        assert mismatches == [], "\n".join(mismatches)

    def test_exchange_suffixes_ibkr_codes_are_uppercase(self):
        """The ibkr_exchange field (4th element) in every EXCHANGE_SUFFIXES tuple
        must be uppercase, because IBKR_TO_YFINANCE keys are all uppercase and
        Ticker.from_ibkr() uppercases before lookup.
        """
        bad = []
        for suffix_key, (_, _, _, ibkr_code) in _exchange_suffixes().items():
            if ibkr_code and ibkr_code != ibkr_code.upper():
                bad.append(
                    f".{suffix_key}: ibkr_exchange='{ibkr_code}' is not uppercase"
                )
        assert bad == [], "\n".join(bad)


# ── Ticker normalization ──────────────────────────────────────────────────────


class TestTickerNormalization:
    """normalize_ticker correctness for ambiguous / easily confused suffixes."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("3217.TWO", "3217.TWO"),  # Taiwan OTC — must NOT become 3217.TW
            ("2330.TW", "2330.TW"),  # Taiwan TWSE — must NOT become 2330.TWO
            ("7203.T", "7203.T"),  # Japan
            ("0005.HK", "0005.HK"),  # Hong Kong
            ("005930.KS", "005930.KS"),  # Korea KOSPI
            ("005930.KQ", "005930.KQ"),  # Korea KOSDAQ
            ("1234.SS", "1234.SS"),  # Shanghai
            ("1234.SZ", "1234.SZ"),  # Shenzhen
        ],
    )
    def test_normalize_preserves_suffix(self, raw, expected):
        from src.ticker_utils import normalize_ticker

        assert (
            normalize_ticker(raw) == expected
        ), f"normalize_ticker({raw!r}) should return {expected!r}"

    def test_two_and_tw_are_distinct(self):
        """Normalizing a .TWO ticker must never produce a .TW ticker and vice versa."""
        from src.ticker_utils import normalize_ticker

        assert normalize_ticker("3217.TWO") != "3217.TW"
        assert normalize_ticker("2330.TW") != "2330.TWO"


# ── IBKR round-trip ───────────────────────────────────────────────────────────


class TestIbkrRoundTrip:
    """Ticker.from_ibkr → .yf must survive the upper() inside from_ibkr."""

    @pytest.mark.parametrize(
        "symbol,exchange,expected_yf",
        [
            ("3217", "TPEX", "3217.TWO"),  # IBKR all-caps — canonical case
            (
                "3217",
                "TPEx",
                "3217.TWO",
            ),  # IBKR mixed-case — must also work via upper()
            ("2330", "TWSE", "2330.TW"),
            ("7203", "TSE", "7203.T"),
            ("5", "SEHK", "0005.HK"),  # HK zero-padding
            ("5934", "KRX", "5934.KS"),
        ],
    )
    def test_from_ibkr_produces_correct_yf(self, symbol, exchange, expected_yf):
        from src.ibkr.ticker import Ticker

        t = Ticker.from_ibkr(symbol=symbol, exchange=exchange)
        assert t.yf == expected_yf, (
            f"Ticker.from_ibkr('{symbol}', '{exchange}').yf == {t.yf!r}, "
            f"expected {expected_yf!r}"
        )

    def test_tpex_and_twse_produce_different_suffixes(self):
        """TPEX and TWSE are different exchanges — their yfinance suffixes must differ."""
        from src.ibkr.ticker import Ticker

        twse = Ticker.from_ibkr(symbol="2330", exchange="TWSE")
        tpex = Ticker.from_ibkr(symbol="3217", exchange="TPEX")
        assert twse.yf.endswith(".TW"), f"TWSE ticker should end .TW, got {twse.yf}"
        assert tpex.yf.endswith(".TWO"), f"TPEX ticker should end .TWO, got {tpex.yf}"
        assert twse.suffix != tpex.suffix


# ── Retrospective map coverage ────────────────────────────────────────────────


class TestRetrospectiveMaps:
    """EXCHANGE_BENCHMARK and EXCHANGE_CURRENCY must cover the same set of
    suffixes.  Also enforces Taiwan-specific entries.
    """

    def test_benchmark_and_currency_have_same_keys(self):
        from src.retrospective import EXCHANGE_BENCHMARK, EXCHANGE_CURRENCY

        b_keys = set(EXCHANGE_BENCHMARK)
        c_keys = set(EXCHANGE_CURRENCY)
        only_in_benchmark = b_keys - c_keys
        only_in_currency = c_keys - b_keys
        assert not only_in_benchmark, f"Suffixes in EXCHANGE_BENCHMARK but not EXCHANGE_CURRENCY: {only_in_benchmark}"
        assert not only_in_currency, f"Suffixes in EXCHANGE_CURRENCY but not EXCHANGE_BENCHMARK: {only_in_currency}"

    def test_taiwan_otc_has_benchmark(self):
        from src.retrospective import EXCHANGE_BENCHMARK

        assert ".TWO" in EXCHANGE_BENCHMARK, (
            ".TWO missing from EXCHANGE_BENCHMARK — retrospective comparisons "
            "silently skip Taiwan OTC stocks"
        )

    def test_taiwan_otc_has_currency(self):
        from src.retrospective import EXCHANGE_CURRENCY

        assert ".TWO" in EXCHANGE_CURRENCY

    def test_taiwan_otc_and_twse_share_benchmark(self):
        """Both Taiwan exchanges use the same broad market index."""
        from src.retrospective import EXCHANGE_BENCHMARK

        assert EXCHANGE_BENCHMARK.get(".TW") == EXCHANGE_BENCHMARK.get(
            ".TWO"
        ), ".TW and .TWO should use the same benchmark index (^TWII)"

    def test_taiwan_otc_and_twse_share_currency(self):
        from src.retrospective import EXCHANGE_CURRENCY

        assert EXCHANGE_CURRENCY.get(".TW") == EXCHANGE_CURRENCY.get(".TWO") == "TWD"

    def test_korea_both_exchanges_covered(self):
        """KOSPI and KOSDAQ are both Korean — both should have entries."""
        from src.retrospective import EXCHANGE_BENCHMARK, EXCHANGE_CURRENCY

        for suffix in (".KS", ".KQ"):
            assert (
                suffix in EXCHANGE_BENCHMARK
            ), f"{suffix} missing from EXCHANGE_BENCHMARK"
            assert (
                suffix in EXCHANGE_CURRENCY
            ), f"{suffix} missing from EXCHANGE_CURRENCY"

    def test_china_both_exchanges_covered(self):
        from src.retrospective import EXCHANGE_CURRENCY

        for suffix in (".SS", ".SZ"):
            assert (
                suffix in EXCHANGE_CURRENCY
            ), f"{suffix} missing from EXCHANGE_CURRENCY"


# ── Fragile exchange list ─────────────────────────────────────────────────────


class TestFragileExchangeList:
    """The is_fragile_exchange check in fetcher._fetch_metrics_inner must include
    all exchanges known to have poor yfinance data coverage.
    """

    def _get_fragile_set(self) -> set[str]:
        """Parse the tuple literal from fetcher source to get the covered suffixes."""
        import src.data.fetcher as fetcher_mod

        source = inspect.getsource(fetcher_mod)
        # Find: ticker.endswith(("...", "...", ...))
        m = re.search(
            r"is_fragile_exchange\s*=\s*ticker\.endswith\(\s*\(([^)]+)\)\s*\)",
            source,
            re.DOTALL,
        )
        assert m, "Could not locate is_fragile_exchange line in fetcher.py"
        raw = m.group(1)
        return set(re.findall(r'"(\.[A-Z]+)"', raw))

    def test_tw_and_two_both_fragile(self):
        fragile = self._get_fragile_set()
        assert ".TW" in fragile, ".TW missing from is_fragile_exchange"
        assert ".TWO" in fragile, ".TWO missing from is_fragile_exchange"

    def test_hk_and_japan_fragile(self):
        fragile = self._get_fragile_set()
        assert ".HK" in fragile
        assert ".T" in fragile

    def test_korea_fragile(self):
        fragile = self._get_fragile_set()
        assert ".KS" in fragile


# ── Tool annotation quality ───────────────────────────────────────────────────

_TICKER_TOOLS = [
    ("src.tools.market", "get_financial_metrics"),
    ("src.tools.market", "get_yfinance_data"),
    ("src.tools.market", "get_technical_indicators"),
    ("src.tools.market", "get_fundamental_analysis"),
    ("src.tools.news", "get_news"),
]

# Minimum bar: annotation must mention at least one concrete example suffix so
# the LLM has format evidence at tool-call time.
_SUFFIX_PATTERN = re.compile(r"\.\w+")


class TestToolAnnotations:
    """Tool parameter annotations for ticker/symbol args must be informative
    enough to prevent the LLM from normalizing unfamiliar suffixes.
    """

    @pytest.mark.parametrize("module_path,tool_name", _TICKER_TOOLS)
    def test_ticker_annotation_mentions_exchange_suffix(self, module_path, tool_name):
        mod = importlib.import_module(module_path)
        fn = getattr(mod, tool_name)
        # LangChain wraps with .func; unwrap to get the original signature
        orig = getattr(fn, "func", fn)
        hints = {}
        try:
            import typing

            hints = typing.get_type_hints(orig, include_extras=True)
        except Exception:
            pytest.skip(f"Could not get type hints for {tool_name}")

        # Find the ticker/symbol param
        ticker_param = next(
            (h for k, h in hints.items() if k in ("ticker", "symbol")), None
        )
        assert (
            ticker_param is not None
        ), f"{tool_name}: no 'ticker' or 'symbol' parameter found in type hints"

        # Extract the annotation string from Annotated metadata
        metadata = getattr(ticker_param, "__metadata__", ())
        assert metadata, (
            f"{tool_name}: ticker/symbol parameter has no Annotated metadata — "
            "the LLM receives no format guidance when constructing tool calls"
        )
        annotation_text = " ".join(str(m) for m in metadata)

        assert _SUFFIX_PATTERN.search(annotation_text), (
            f"{tool_name}: ticker/symbol annotation '{annotation_text}' contains no "
            "exchange suffix example (e.g. '.TWO', '.T', '.HK'). Without a concrete "
            "example the LLM may normalize unfamiliar suffixes."
        )

    @pytest.mark.parametrize("module_path,tool_name", _TICKER_TOOLS)
    def test_tickers_annotation_mentions_twо(self, module_path, tool_name):
        """.TWO must appear in at least one tool annotation so it's visible
        in the schema and the LLM knows it's a valid distinct suffix.
        """
        mod = importlib.import_module(module_path)
        fn = getattr(mod, tool_name)
        orig = getattr(fn, "func", fn)
        try:
            import typing

            hints = typing.get_type_hints(orig, include_extras=True)
        except Exception:
            pytest.skip(f"Could not get type hints for {tool_name}")

        ticker_param = next(
            (h for k, h in hints.items() if k in ("ticker", "symbol")), None
        )
        if ticker_param is None:
            return
        metadata = getattr(ticker_param, "__metadata__", ())
        annotation_text = " ".join(str(m) for m in metadata)
        assert ".TWO" in annotation_text, (
            f"{tool_name}: annotation does not mention '.TWO'. "
            "Without this example the LLM may silently normalize Taiwan OTC "
            "tickers from .TWO to .TW."
        )


# ── Taiwan-specific regression ────────────────────────────────────────────────


class TestTaiwanSuffixRegression:
    """.TWO and .TW are distinct exchanges; any code that treats them as
    interchangeable is a bug.
    """

    def test_exchange_suffixes_has_both_tw_and_two(self):
        sfx = _exchange_suffixes()
        assert "TW" in sfx, "TW  missing from EXCHANGE_SUFFIXES"
        assert "TWO" in sfx, "TWO missing from EXCHANGE_SUFFIXES"

    def test_tw_and_two_map_to_different_ibkr_codes(self):
        sfx = _exchange_suffixes()
        tw_ibkr = sfx["TW"][3]
        two_ibkr = sfx["TWO"][3]
        assert tw_ibkr != two_ibkr, (
            f"TW and TWO share the same IBKR code '{tw_ibkr}' — "
            "they are different exchanges and must map differently"
        )

    def test_tw_and_two_map_to_different_yfinance_suffixes(self):
        sfx = _exchange_suffixes()
        assert sfx["TW"][0] == ".TW"
        assert sfx["TWO"][0] == ".TWO"

    def test_ibkr_to_yfinance_has_both_twse_and_tpex(self):
        m = _ibkr_to_yfinance()
        assert "TWSE" in m, "TWSE missing from IBKR_TO_YFINANCE"
        assert "TPEX" in m, "TPEX missing from IBKR_TO_YFINANCE"
        assert m["TWSE"] == ".TW"
        assert m["TPEX"] == ".TWO"

    def test_tpex_key_is_uppercase_not_mixed(self):
        """Regression: key was 'TPEx' (mixed case) which broke Ticker.from_ibkr()
        because that method uppercases the exchange code before lookup."""
        m = _ibkr_to_yfinance()
        assert "TPEx" not in m, (
            "'TPEx' found in IBKR_TO_YFINANCE — this key is unreachable because "
            "Ticker.from_ibkr() uppercases the exchange code. Use 'TPEX' instead."
        )

    def test_normalize_3217_two_stays_two(self):
        from src.ticker_utils import normalize_ticker

        result = normalize_ticker("3217.TWO")
        assert result == "3217.TWO", (
            f"normalize_ticker('3217.TWO') returned '{result}' — "
            "Taiwan OTC ticker must not be normalized to .TW"
        )

    def test_ibkr_ticker_tpex_produces_two_suffix(self):
        from src.ibkr.ticker import Ticker

        t = Ticker.from_ibkr(symbol="3217", exchange="TPEX")
        assert t.yf == "3217.TWO", f"Expected '3217.TWO', got '{t.yf}'"

    def test_ibkr_ticker_tpex_mixed_case_produces_two_suffix(self):
        """IBKR API may return 'TPEx'; after upper() this becomes 'TPEX'."""
        from src.ibkr.ticker import Ticker

        t = Ticker.from_ibkr(symbol="3217", exchange="TPEx")
        assert t.yf == "3217.TWO", (
            f"Ticker.from_ibkr with exchange='TPEx' returned '{t.yf}'. "
            "After .upper() the code is 'TPEX' which should map to '.TWO'."
        )

    def test_ibkr_ticker_twse_does_not_produce_two_suffix(self):
        from src.ibkr.ticker import Ticker

        t = Ticker.from_ibkr(symbol="2330", exchange="TWSE")
        assert t.yf == "2330.TW", f"Expected '2330.TW', got '{t.yf}'"
        assert not t.yf.endswith(".TWO")
