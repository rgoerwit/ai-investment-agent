"""Unit tests for src.ibkr.ticker.Ticker value object."""

import pytest

from src.ibkr.ticker import Ticker

# ── TestTickerFromIbkr ────────────────────────────────────────────────────────


class TestTickerFromIbkr:
    """Tests for Ticker.from_ibkr() — IBKR raw fields → Ticker."""

    # Known exchange codes (static IBKR_TO_YFINANCE map wins)

    def test_tsej_japan(self):
        # IBKR Client Portal reports Tokyo as "TSEJ" (not "TSE").
        t = Ticker.from_ibkr("7203", "TSEJ", "JPY")
        assert t.suffix == ".T"
        assert t.yf == "7203.T"
        assert t.ibkr == "7203"

    def test_tse_toronto(self):
        # IBKR Client Portal reports Toronto as "TSE" → must map to .TO, not .T.
        t = Ticker.from_ibkr("PEY", "TSE", "CAD")
        assert t.suffix == ".TO"
        assert t.yf == "PEY.TO"
        assert t.ibkr == "PEY"

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

    # Korea zero-padding edge cases

    def test_krx_short_numeric_padded_to_6(self):
        t = Ticker.from_ibkr("5930", "KRX", "KRW")
        assert t.symbol == "005930"
        assert t.suffix == ".KS"
        assert t.yf == "005930.KS"

    def test_krx_pre_padded_input_preserved(self):
        t = Ticker.from_ibkr("005930", "KRX", "KRW")
        assert t.symbol == "005930"
        assert t.yf == "005930.KS"

    def test_krx_five_digit_numeric_padded_to_6(self):
        t = Ticker.from_ibkr("10130", "KRX", "KRW")
        assert t.symbol == "010130"
        assert t.yf == "010130.KS"

    def test_krx_padded_five_digit_input_preserved(self):
        t = Ticker.from_ibkr("001060", "KRX", "KRW")
        assert t.symbol == "001060"
        assert t.yf == "001060.KS"

    def test_kosdaq_short_numeric_padded_to_6(self):
        t = Ticker.from_ibkr("35420", "KOSDAQ", "KRW")
        assert t.symbol == "035420"
        assert t.suffix == ".KQ"
        assert t.yf == "035420.KQ"

    def test_korea_mixed_symbol_not_padded(self):
        t = Ticker.from_ibkr("ABC123", "KRX", "KRW")
        assert t.symbol == "ABC123"
        assert t.yf == "ABC123.KS"

    # Brazil B3 — alphanumeric tickers, no padding rule

    def test_brazil_b3_ticker_alphanumeric(self):
        t = Ticker.from_ibkr("PETR4", "BVMF", "BRL")
        assert t.symbol == "PETR4"
        assert t.suffix == ".SA"
        assert t.yf == "PETR4.SA"

    def test_brazil_b3_units_11_suffix(self):
        t = Ticker.from_ibkr("KNRI11", "BVMF", "BRL")
        assert t.yf == "KNRI11.SA"

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

    def test_currency_fallback_chf(self):
        t = Ticker.from_ibkr("NESN", "UNKNOWN_EXCH", "CHF")
        assert t.suffix == ".SW"
        assert t.yf == "NESN.SW"

    def test_currency_fallback_pln(self):
        t = Ticker.from_ibkr("PKN", "UNKNOWN_EXCH", "PLN")
        assert t.suffix == ".WA"

    def test_currency_fallback_myr(self):
        t = Ticker.from_ibkr("MEGP", "MESDAQ", "MYR")
        assert t.suffix == ".KL"

    def test_currency_fallback_krw_defaults_to_ks(self):
        t = Ticker.from_ibkr("5930", "", "KRW")
        assert t.suffix == ".KS"
        assert t.yf == "005930.KS"

    def test_currency_fallback_brl_defaults_to_sa(self):
        t = Ticker.from_ibkr("PETR4", "SMART", "BRL")
        assert t.suffix == ".SA"
        assert t.yf == "PETR4.SA"

    # Exchange priority over conflicting currency

    def test_exchange_wins_over_currency(self):
        """SEHK exchange → .HK even if currency says USD (ADR scenario)."""
        t = Ticker.from_ibkr("5", "SEHK", "USD")
        assert t.suffix == ".HK"
        assert t.yf == "0005.HK"

    def test_kosdaq_exchange_wins_over_krw_fallback(self):
        t = Ticker.from_ibkr("35420", "KOSDAQ", "KRW")
        assert t.suffix == ".KQ"
        assert t.yf == "035420.KQ"

    # US / bare result

    def test_smart_gives_no_suffix(self):
        t = Ticker.from_ibkr("AAPL", "SMART", "USD")
        assert t.suffix == ""
        assert t.yf == "AAPL"
        assert not t.has_suffix

    def test_smart_non_usd_currency_falls_back(self):
        t = Ticker.from_ibkr("5", "SMART", "HKD")
        assert t.suffix == ".HK"
        assert t.yf == "0005.HK"

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

    def test_krx_round_trip(self):
        t = Ticker.from_yf("005930.KS")
        assert t.symbol == "005930"
        assert t.exchange == "KRX"
        assert t.yf == "005930.KS"

    def test_krx_short_yf_normalizes(self):
        t = Ticker.from_yf("5930.KS")
        assert t.symbol == "005930"
        assert t.yf == "005930.KS"

    def test_krx_five_digit_yf_normalizes(self):
        t = Ticker.from_yf("10130.KS")
        assert t.symbol == "010130"
        assert t.yf == "010130.KS"

    def test_kosdaq_round_trip(self):
        t = Ticker.from_yf("035420.KQ")
        assert t.symbol == "035420"
        assert t.exchange == "KOSDAQ"
        assert t.yf == "035420.KQ"

    def test_brazil_b3_round_trip(self):
        for yf_str in ("PETR4.SA", "VALE3.SA", "ITUB4.SA", "KNRI11.SA"):
            t = Ticker.from_yf(yf_str)
            assert t.exchange == "BVMF"
            assert t.yf == yf_str

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


class TestCurrencyFallbackPolicy:
    """Tests for public currency-fallback behavior in Ticker.from_ibkr()."""

    def test_unique_currency_fallback_resolves_switzerland(self):
        ticker = Ticker.from_ibkr("NESN", "UNKNOWN_EXCH", "CHF")
        assert ticker.suffix == ".SW"
        assert ticker.yf == "NESN.SW"

    def test_explicit_override_prefers_tw_over_two(self):
        ticker = Ticker.from_ibkr("2330", "UNKNOWN_EXCH", "TWD")
        assert ticker.suffix == ".TW"
        assert ticker.yf == "2330.TW"

    def test_ambiguous_eur_stays_bare(self):
        ticker = Ticker.from_ibkr("ASML", "UNKNOWN_EXCH", "EUR")
        assert ticker.suffix == ""
        assert ticker.yf == "ASML"

    def test_ambiguous_cad_stays_bare(self):
        ticker = Ticker.from_ibkr("MTL", "UNKNOWN_EXCH", "CAD")
        assert ticker.suffix == ""
        assert ticker.yf == "MTL"


# ── TestBrazilBDRTranslation ──────────────────────────────────────────────────
#
# ARCHITECTURAL INVARIANT: BDR exclusion lives at the SCRAPER layer
# (config/exchanges.json exclude_filter), NOT at the translation layer.
#
# Reason: scripts/portfolio_manager.py reconciles live IBKR positions against
# analysis records via Ticker.from_ibkr() → ticker.yf lookups. If a user holds
# an existing BDR position (e.g. ADBE34 acquired before BDR exclusion shipped,
# or held intentionally outside the screening universe), portfolio_manager.py
# MUST still translate it to ADBE34.SA for the analysis lookup and back to
# ADBE34 for display.
#
# If a future refactor "tidies" the BDR rejection into the Ticker layer
# (e.g. by raising on 3[2-9] codes), portfolio_manager.py silently breaks for
# every held BDR. These tests are the regression guard.


class TestBrazilBDRTranslation:
    """BDRs must round-trip through Ticker — exclusion is a scraper concern only."""

    # ── Sponsored BDRs across all 32-39 suffix levels ────────────────────────

    @pytest.mark.parametrize(
        "symbol",
        ["ADBE34", "GILD34", "INTU34", "CMCS34", "FSLR34", "GDBR34"],
    )
    def test_sponsored_bdr_from_ibkr(self, symbol):
        t = Ticker.from_ibkr(symbol, "BVMF", "BRL")
        assert t.symbol == symbol
        assert t.suffix == ".SA"
        assert t.yf == f"{symbol}.SA"
        assert t.ibkr == symbol  # display value for portfolio_manager.py

    @pytest.mark.parametrize(
        "suffix_digits",
        ["32", "33", "34", "35", "36", "37", "38", "39"],
    )
    def test_bdr_all_sponsorship_levels_translate(self, suffix_digits):
        """All BDR sponsorship classes 32-39 must round-trip."""
        symbol = f"XYZA{suffix_digits}"
        t = Ticker.from_ibkr(symbol, "BVMF", "BRL")
        assert t.yf == f"{symbol}.SA"
        # Round-trip back through yf
        assert Ticker.from_yf(f"{symbol}.SA").yf == f"{symbol}.SA"

    # ── Unsponsored BDRs (alphanumeric prefix with internal digit) ───────────

    @pytest.mark.parametrize(
        "symbol",
        ["A1GI34", "B1RF34", "M2RV34", "C1HK34", "D1VN34", "F1NI34"],
    )
    def test_unsponsored_bdr_translates(self, symbol):
        """Unsponsored BDRs have an internal digit (e.g. A1GI34) — must not break parsing."""
        t = Ticker.from_ibkr(symbol, "BVMF", "BRL")
        assert t.symbol == symbol
        assert t.yf == f"{symbol}.SA"
        assert t.ibkr == symbol

    # ── Real-world edge cases ────────────────────────────────────────────────

    def test_bdr_via_smart_exchange_currency_fallback(self):
        """IBKR sometimes reports SMART for watchlist contracts; BRL fallback must resolve."""
        t = Ticker.from_ibkr("ADBE34", "SMART", "BRL")
        assert t.suffix == ".SA"
        assert t.yf == "ADBE34.SA"

    def test_bdr_via_empty_exchange_currency_fallback(self):
        """Empty exchange string + BRL currency: fallback path must still pick .SA."""
        t = Ticker.from_ibkr("GILD34", "", "BRL")
        assert t.suffix == ".SA"
        assert t.yf == "GILD34.SA"

    def test_bdr_with_whitespace_padding(self):
        """IBKR API occasionally returns padded fields; from_ibkr strips them."""
        t = Ticker.from_ibkr("  ADBE34  ", "  BVMF  ", "  BRL  ")
        assert t.yf == "ADBE34.SA"

    def test_bdr_with_lowercase_exchange_and_currency(self):
        """Exchange code + currency are normalized to upper-case."""
        t = Ticker.from_ibkr("ADBE34", "bvmf", "brl")
        assert t.suffix == ".SA"
        assert t.yf == "ADBE34.SA"

    def test_bdr_yf_round_trip_independent_of_currency(self):
        """yf → ibkr → yf must be lossless even without a currency hint."""
        for yf_str in ("ADBE34.SA", "A1GI34.SA", "M2RV34.SA", "XYZA39.SA"):
            t = Ticker.from_yf(yf_str)
            assert t.yf == yf_str
            assert t.exchange == "BVMF"

    def test_bdr_yf_round_trip_with_currency_kwarg(self):
        """Currency kwarg propagates and does not perturb yf result."""
        t = Ticker.from_yf("ADBE34.SA", currency="BRL")
        assert t.yf == "ADBE34.SA"
        assert t.currency == "BRL"

    # ── Architectural invariant: native + BDR coexist ────────────────────────

    def test_native_and_bdr_coexist_in_translation_layer(self):
        """Both native B3 and BDR codes must translate identically through Ticker.

        The translation layer is exchange-aware (.SA / BVMF / BRL), not
        share-class-aware. If someone adds share-class filtering here, this
        test fails and points them at the comment block above.
        """
        cases = [
            # (label, ibkr_symbol, expected_yf)
            ("native ON", "PETR3", "PETR3.SA"),
            ("native PN", "PETR4", "PETR4.SA"),
            ("native preferred", "ITUB4", "ITUB4.SA"),
            ("native units", "KNRI11", "KNRI11.SA"),
            ("native mixed", "B3SA3", "B3SA3.SA"),
            ("sponsored BDR", "ADBE34", "ADBE34.SA"),
            ("unsponsored BDR", "A1GI34", "A1GI34.SA"),
        ]
        for label, sym, expected_yf in cases:
            t = Ticker.from_ibkr(sym, "BVMF", "BRL")
            assert t.yf == expected_yf, f"{label}: expected {expected_yf}, got {t.yf}"
            assert t.ibkr == sym, f"{label}: ibkr display lost"
            # Round-trip yf → ibkr → yf must be identity
            assert (
                Ticker.from_yf(t.yf).yf == expected_yf
            ), f"{label}: yf round-trip not identity"

    def test_portfolio_manager_display_contract_for_brazilian_positions(self):
        """Documents the contract scripts/portfolio_manager.py:_display_ticker depends on.

        _display_ticker(item) returns item.ticker.ibkr (the bare IBKR symbol),
        while run commands use item.ticker.yf (yfinance format with suffix).
        Both must be available for every Brazilian position type.
        """
        for sym in ("PETR4", "ADBE34", "A1GI34", "KNRI11", "B3SA3"):
            t = Ticker.from_ibkr(sym, "BVMF", "BRL")
            assert t.ibkr == sym, f"{sym}: display value must equal IBKR symbol"
            assert (
                t.yf == f"{sym}.SA"
            ), f"{sym}: run-command value must carry .SA suffix"
            assert t.has_suffix, f"{sym}: must be flagged as international"
            assert (
                t.exchange_resolved
            ), f"{sym}: BVMF must resolve, no ⚠ warning expected"
