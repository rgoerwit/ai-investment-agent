"""Unit tests for src.ibkr.ticker.Ticker value object."""

import pytest

from src.ibkr.ticker import _CURRENCY_TO_SUFFIX, Ticker

# ── TestTickerFromIbkr ────────────────────────────────────────────────────────


class TestTickerFromIbkr:
    """Tests for Ticker.from_ibkr() — IBKR raw fields → Ticker."""

    # Known exchange codes (static IBKR_TO_YFINANCE map wins)

    def test_tse_japan(self):
        t = Ticker.from_ibkr("7203", "TSE", "JPY")
        assert t.suffix == ".T"
        assert t.yf == "7203.T"
        assert t.ibkr == "7203"

    def test_sehk_hong_kong(self):
        t = Ticker.from_ibkr("5", "SEHK", "HKD")
        assert t.suffix == ".HK"
        assert t.yf == "0005.HK"
        assert t.ibkr == "5"

    def test_lse_london(self):
        t = Ticker.from_ibkr("MEGP", "LSE", "GBX")
        assert t.suffix == ".L"
        assert t.yf == "MEGP.L"

    def test_aeb_amsterdam(self):
        t = Ticker.from_ibkr("ASML", "AEB", "EUR")
        assert t.suffix == ".AS"
        assert t.yf == "ASML.AS"

    def test_ibis2_xetra(self):
        t = Ticker.from_ibkr("SAP", "IBIS2", "EUR")
        assert t.suffix == ".DE"
        assert t.yf == "SAP.DE"

    # HK zero-padding edge cases

    def test_hk_1_digit_padded_to_4(self):
        t = Ticker.from_ibkr("5", "SEHK")
        assert t.yf == "0005.HK"
        assert t.symbol == "5"  # stored without padding

    def test_hk_3_digit_padded_to_4(self):
        t = Ticker.from_ibkr("700", "SEHK")
        assert t.yf == "0700.HK"
        assert t.symbol == "700"

    def test_hk_4_digit_unchanged(self):
        t = Ticker.from_ibkr("2318", "SEHK")
        assert t.yf == "2318.HK"
        assert t.symbol == "2318"

    def test_hk_pre_padded_input_stripped(self):
        """IBKR sometimes sends pre-padded "0005" — stored as "5", yf = "0005.HK"."""
        t = Ticker.from_ibkr("0005", "SEHK")
        assert t.symbol == "5"
        assert t.yf == "0005.HK"

    # Currency fallback (when exchange is unknown/SMART)

    def test_currency_fallback_hkd(self):
        t = Ticker.from_ibkr("XYZ", "UNKNOWN_EXCH", "HKD")
        assert t.suffix == ".HK"

    def test_currency_fallback_jpy(self):
        t = Ticker.from_ibkr("1234", "UNKNOWN_EXCH", "JPY")
        assert t.suffix == ".T"

    def test_currency_fallback_gbx(self):
        t = Ticker.from_ibkr("GAMA", "UNKNOWN_EXCH", "GBX")
        assert t.suffix == ".L"

    def test_currency_fallback_gbp(self):
        t = Ticker.from_ibkr("GAMA", "UNKNOWN_EXCH", "GBP")
        assert t.suffix == ".L"

    def test_currency_fallback_nok(self):
        t = Ticker.from_ibkr("STB", "UNKNOWN_EXCH", "NOK")
        assert t.suffix == ".OL"

    def test_currency_fallback_pln(self):
        t = Ticker.from_ibkr("PKN", "UNKNOWN_EXCH", "PLN")
        assert t.suffix == ".WA"

    def test_currency_fallback_myr(self):
        t = Ticker.from_ibkr("MEGP", "MESDAQ", "MYR")
        assert t.suffix == ".KL"

    # Exchange priority over conflicting currency

    def test_exchange_wins_over_currency(self):
        """SEHK exchange → .HK even if currency says USD (ADR scenario)."""
        t = Ticker.from_ibkr("5", "SEHK", "USD")
        assert t.suffix == ".HK"
        assert t.yf == "0005.HK"

    # US / bare result

    def test_smart_gives_no_suffix(self):
        t = Ticker.from_ibkr("AAPL", "SMART", "USD")
        assert t.suffix == ""
        assert t.yf == "AAPL"
        assert not t.has_suffix

    def test_nasdaq_gives_no_suffix(self):
        t = Ticker.from_ibkr("MSFT", "NASDAQ", "USD")
        assert t.suffix == ""
        assert not t.has_suffix

    def test_unknown_exchange_ambiguous_currency_bare(self):
        """Unknown exchange + EUR (multi-exchange) → no suffix, bare result."""
        t = Ticker.from_ibkr("CEK", "UNKNOWN", "EUR")
        assert t.suffix == ""
        assert t.yf == "CEK"
        assert not t.has_suffix


# ── TestTickerFromYf ──────────────────────────────────────────────────────────


class TestTickerFromYf:
    """Tests for Ticker.from_yf() — yfinance string → Ticker."""

    def test_japan_suffix(self):
        t = Ticker.from_yf("7203.T")
        assert t.symbol == "7203"
        assert t.suffix == ".T"
        assert t.yf == "7203.T"

    def test_japan_alphanumeric_suffix(self):
        t = Ticker.from_yf("262A.T")
        assert t.symbol == "262A"
        assert t.suffix == ".T"
        assert t.yf == "262A.T"

    def test_hong_kong_suffix(self):
        t = Ticker.from_yf("0005.HK")
        assert t.symbol == "5"  # zero-padding stripped from symbol
        assert t.suffix == ".HK"
        assert t.yf == "0005.HK"  # re-applied on output

    def test_amsterdam_suffix(self):
        t = Ticker.from_yf("ASML.AS")
        assert t.symbol == "ASML"
        assert t.suffix == ".AS"
        assert t.yf == "ASML.AS"

    def test_london_suffix(self):
        t = Ticker.from_yf("MEGP.L")
        assert t.symbol == "MEGP"
        assert t.suffix == ".L"
        assert t.yf == "MEGP.L"

    def test_frankfurt_suffix(self):
        t = Ticker.from_yf("SAP.DE")
        assert t.symbol == "SAP"
        assert t.suffix == ".DE"
        assert t.yf == "SAP.DE"

    def test_us_bare(self):
        t = Ticker.from_yf("AAPL")
        assert t.symbol == "AAPL"
        assert t.exchange == "SMART"
        assert t.suffix == ""
        assert t.yf == "AAPL"
        assert not t.has_suffix

    def test_hk_zero_strip_round_trip(self):
        """from_yf strips leading zeros; .yf re-pads to 4 digits."""
        t = Ticker.from_yf("0005.HK")
        assert t.symbol == "5"
        assert t.yf == "0005.HK"

    def test_hk_1_digit_round_trip(self):
        t = Ticker.from_yf("0001.HK")
        assert t.symbol == "1"
        assert t.yf == "0001.HK"

    def test_currency_kwarg_preserved(self):
        t = Ticker.from_yf("7203.T", currency="JPY")
        assert t.currency == "JPY"

    def test_currency_kwarg_upcased(self):
        t = Ticker.from_yf("ASML.AS", currency="eur")
        assert t.currency == "EUR"

    def test_no_currency_kwarg_empty_string(self):
        t = Ticker.from_yf("7203.T")
        assert t.currency == ""


# ── TestTickerProperties ──────────────────────────────────────────────────────


class TestTickerProperties:
    """Tests for Ticker properties, equality, hashing, and immutability."""

    def test_str_returns_yf(self):
        t = Ticker.from_yf("7203.T")
        assert str(t) == "7203.T"

    def test_str_hk_zero_padded(self):
        t = Ticker.from_ibkr("5", "SEHK")
        assert str(t) == "0005.HK"

    def test_equality_same_fields(self):
        a = Ticker("7203", "TSE", "JPY")
        b = Ticker("7203", "TSE", "JPY")
        assert a == b

    def test_inequality_different_symbol(self):
        a = Ticker("7203", "TSE", "JPY")
        b = Ticker("9201", "TSE", "JPY")
        assert a != b

    def test_inequality_different_exchange(self):
        a = Ticker("5", "SEHK", "HKD")
        b = Ticker("5", "TSE", "HKD")
        assert a != b

    def test_hashable_as_dict_key(self):
        t1 = Ticker.from_yf("7203.T")
        t2 = Ticker.from_yf("7203.T")
        d = {t1: "value"}
        assert d[t2] == "value"

    def test_hashable_in_set(self):
        t1 = Ticker.from_yf("0005.HK")
        t2 = Ticker.from_yf("0005.HK")
        s = {t1, t2}
        assert len(s) == 1

    def test_frozen_raises_on_attribute_assignment(self):
        t = Ticker.from_yf("AAPL")
        with pytest.raises((AttributeError, TypeError)):
            t.symbol = "MSFT"  # type: ignore[misc]

    def test_has_suffix_true_for_international(self):
        assert Ticker.from_yf("7203.T").has_suffix is True
        assert Ticker.from_yf("0005.HK").has_suffix is True
        assert Ticker.from_yf("ASML.AS").has_suffix is True

    def test_has_suffix_false_for_us(self):
        assert Ticker.from_yf("AAPL").has_suffix is False
        assert Ticker.from_ibkr("MSFT", "SMART").has_suffix is False

    def test_ibkr_returns_bare_symbol(self):
        t = Ticker.from_yf("0005.HK")
        assert t.ibkr == "5"

    def test_ibkr_no_suffix_for_us(self):
        t = Ticker.from_yf("AAPL")
        assert t.ibkr == "AAPL"

    def test_suffix_consistency_with_yf(self):
        """For any ticker, yf must end with suffix (or suffix is empty)."""
        cases = ["7203.T", "0005.HK", "ASML.AS", "MEGP.L", "AAPL"]
        for yf_str in cases:
            t = Ticker.from_yf(yf_str)
            if t.has_suffix:
                assert t.yf.endswith(t.suffix)
            else:
                assert t.suffix == ""


# ── TestCurrencyToSuffixExported ──────────────────────────────────────────────


class TestCurrencyToSuffixExported:
    """Tests for the _CURRENCY_TO_SUFFIX module-level dict in ticker.py."""

    def test_importable(self):
        from src.ibkr.ticker import _CURRENCY_TO_SUFFIX as CTS  # noqa: N811

        assert isinstance(CTS, dict)

    def test_hkd_maps_to_hk(self):
        assert _CURRENCY_TO_SUFFIX["HKD"] == ".HK"

    def test_jpy_maps_to_t(self):
        assert _CURRENCY_TO_SUFFIX["JPY"] == ".T"

    def test_gbx_maps_to_l(self):
        assert _CURRENCY_TO_SUFFIX["GBX"] == ".L"

    def test_gbp_maps_to_l(self):
        assert _CURRENCY_TO_SUFFIX["GBP"] == ".L"

    def test_eur_not_present(self):
        """EUR spans multiple exchanges — must NOT be in the dict."""
        assert "EUR" not in _CURRENCY_TO_SUFFIX

    def test_chf_not_present(self):
        """CHF spans multiple exchanges — must NOT be in the dict."""
        assert "CHF" not in _CURRENCY_TO_SUFFIX

    def test_cad_not_present(self):
        """CAD spans multiple exchanges — must NOT be in the dict."""
        assert "CAD" not in _CURRENCY_TO_SUFFIX

    def test_min_entry_count(self):
        """Dict should have at least 14 entries (all unambiguous single-exchange currencies)."""
        assert len(_CURRENCY_TO_SUFFIX) >= 14
