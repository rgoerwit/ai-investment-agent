from __future__ import annotations

from src.exchange_metadata import (
    EXCHANGES_BY_SUFFIX,
    canonical_suffix_for_reuters_exchange,
)
from src.ticker_corrections import TickerCorrector
from src.ticker_utils import TickerFormatter, normalize_ticker


def test_canonical_suffix_for_reuters_exchange_uses_exchange_metadata():
    assert canonical_suffix_for_reuters_exchange("N", "CH") == ".SW"
    assert canonical_suffix_for_reuters_exchange("N", "JP") == ".T"
    assert canonical_suffix_for_reuters_exchange("unknown", "xx") is None


def test_normalize_ticker_reuters_metadata_comes_from_canonical_exchange_registry():
    normalized, metadata = TickerFormatter.normalize_ticker("NOVN.N-CH")

    assert normalized == "NOVN.SW"
    assert metadata["format"] == "reuters"
    assert metadata["exchange_suffix"] == ".SW"
    assert metadata["exchange_name"] == EXCHANGES_BY_SUFFIX[".SW"].exchange_name
    assert metadata["country"] == EXCHANGES_BY_SUFFIX[".SW"].country
    assert metadata["ibkr_exchange"] == EXCHANGES_BY_SUFFIX[".SW"].ibkr_code


def test_normalize_ticker_handles_unknown_reuters_exchange_cleanly():
    normalized, metadata = TickerFormatter.normalize_ticker("ABC.ZZ-XY")

    assert normalized == "ABC.ZZ-XY"
    assert metadata["format"] == "invalid"
    assert metadata["exchange_name"] == "Unknown"
    assert metadata["country"] == "Unknown"


def test_ticker_corrector_known_valid_metadata_derives_exchange_facts():
    is_valid, metadata = TickerCorrector.is_known_valid("NOVN.SW")

    assert is_valid is True
    assert metadata == {
        "name": "Novartis AG",
        "exchange": "SIX Swiss Exchange",
        "country": "Switzerland",
    }


def test_ticker_corrector_identity_mappings_are_not_reported_as_corrections():
    for ticker in (
        "SIE.DE",
        "VOW3.DE",
        "BAYN.DE",
        "ULVR.L",
        "7203.T",
        "6758.T",
        "9984.T",
    ):
        corrected, was_corrected, name = TickerCorrector.apply_correction(ticker)

        assert corrected == ticker
        assert was_corrected is False
        assert name


def test_ticker_corrector_real_mapping_is_still_reported_as_correction():
    corrected, was_corrected, name = TickerCorrector.apply_correction("SAPG.DE")

    assert corrected == "SAP.DE"
    assert was_corrected is True
    assert name == "SAP SE"


def test_ticker_corrector_maps_boditech_kospi_typo_to_kosdaq():
    corrected, was_corrected, name = TickerCorrector.apply_correction("206640.KS")

    assert corrected == "206640.KQ"
    assert was_corrected is True
    assert name == "Boditech Med Inc."
    assert normalize_ticker("206640.KS") == "206640.KQ"


def test_ticker_corrector_unknown_ticker_returns_unchanged_without_correction():
    corrected, was_corrected, name = TickerCorrector.apply_correction("UNKNOWN.XX")

    assert corrected == "UNKNOWN.XX"
    assert was_corrected is False
    assert name is None


def test_ticker_corrector_normalizes_case_and_whitespace_before_identity_check():
    corrected, was_corrected, name = TickerCorrector.apply_correction("  sie.de ")

    assert corrected == "SIE.DE"
    assert was_corrected is False
    assert name == "Siemens AG"


def test_ticker_corrector_strips_whitespace_before_identity_check():
    corrected, was_corrected, name = TickerCorrector.apply_correction(" 7203.T ")

    assert corrected == "7203.T"
    assert was_corrected is False
    assert name == "Toyota Motor Corporation"


def test_us_tickers_remain_suffixless_and_normalize_without_international_drift():
    assert normalize_ticker("AAPL") == "AAPL"
    normalized, metadata = TickerFormatter.normalize_ticker("AAPL")
    assert normalized == "AAPL"
    assert metadata["country"] == "United States"
