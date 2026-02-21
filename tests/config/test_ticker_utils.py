# tests/test_ticker_utils.py

import pytest

from src.ticker_utils import (
    TickerFormatter,
    generate_strict_search_query,
    get_ticker_info,
    normalize_company_name,
    to_ibkr,
    to_yfinance,
)


@pytest.mark.parametrize(
    "raw_name, expected",
    [
        # Basic cases
        ("China Resources Beer (Holdings) Company Limited", "China Resources Beer"),
        ("Samsung Electronics Co., Ltd.", "Samsung Electronics"),
        ("Tencent Holdings Limited", "Tencent"),
        ("Apple Inc.", "Apple"),
        ("Nestlé S.A.", "Nestlé"),
        # Stacked suffixes
        ("ABC Group Holdings Ltd", "ABC"),
        ("XYZ Corporation Inc.", "XYZ"),
        # Parentheses removal - FIXED: function removes both parentheses AND suffixes
        ("Tencent Holdings (0700)", "Tencent"),  # Both (0700) and Holdings removed
        ("Samsung Electronics Co., Ltd. (005930)", "Samsung Electronics"),
        # Non-English
        ("腾讯控股有限公司", "腾讯控股有限公司"),  # Should not strip if no match
        ("삼성전자 주식회사", "삼성전자 주식회사"),
        # Edge cases
        ("Holdings", "Holdings"),  # Safety valve prevents over-stripping
        ("A B C", "A B C"),
        ("", ""),
        (None, ""),
        ("Ltd", "Ltd"),
        ("Company Limited", "Company"),
        # Multiple iterations
        ("Test Holdings Company Limited", "Test"),
    ],
)
def test_normalize_company_name(raw_name, expected):
    result = normalize_company_name(raw_name)
    assert result == expected, f"Expected '{expected}', got '{result}'"


@pytest.mark.parametrize(
    "ticker, raw_name, topic, expected",
    [
        # Multi-word name -> quoted
        (
            "0700.HK",
            "China Resources Beer (Holdings) Company Limited",
            "revenue",
            '"China Resources Beer" 0700.HK revenue',
        ),
        (
            "005930.KS",
            "Samsung Electronics Co., Ltd.",
            "earnings",
            '"Samsung Electronics" 005930.KS earnings',
        ),
        # Single word -> unquoted
        ("AAPL", "Apple Inc.", "price", "Apple AAPL price"),
        # Non-English - FIXED: Chinese names without spaces are treated as single words (no quotes)
        ("0700.HK", "腾讯控股", "revenue", "腾讯控股 0700.HK revenue"),
        # Edge cases
        ("", "", "", "  "),
        ("TSLA", "Tesla", "", "Tesla TSLA "),
    ],
)
def test_generate_strict_search_query(ticker, raw_name, topic, expected):
    result = generate_strict_search_query(ticker, raw_name, topic)
    assert result == expected, f"Expected '{expected}', got '{result}'"


@pytest.mark.parametrize(
    "input_ticker, expected_normalized, expected_metadata_keys",
    [
        # Standard format
        (
            "NOVN.SW",
            "NOVN.SW",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        (
            "7203.T",
            "7203.T",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        # IBKR format
        (
            "NOVN:SWX",
            "NOVN.SW",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        # Reuters format (assuming mapping exists)
        (
            "NOVN.N-CH",
            "NOVN.SW",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        # Plain US
        (
            "AAPL",
            "AAPL",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        # Unknown suffix
        (
            "TEST.XX",
            "TEST.XX",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        # Invalid
        (
            "INVALID@",
            "INVALID@",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
        # Lowercase input
        (
            "novn.sw",
            "NOVN.SW",
            [
                "original",
                "symbol",
                "exchange_suffix",
                "exchange_name",
                "country",
                "ibkr_exchange",
                "format",
            ],
        ),
    ],
)
def test_normalize_ticker(input_ticker, expected_normalized, expected_metadata_keys):
    normalized, metadata = TickerFormatter.normalize_ticker(input_ticker)
    assert normalized == expected_normalized
    assert list(metadata.keys()) == expected_metadata_keys
    assert metadata["original"] == input_ticker.upper()
    assert metadata["format"] in [
        "standard",
        "ibkr",
        "reuters",
        "plain",
        "unknown",
        "invalid",
    ]
    assert metadata["country"] != "Unknown" or metadata["format"] in [
        "unknown",
        "invalid",
    ]


@pytest.mark.parametrize(
    "input_ticker, expected",
    [
        ("NOVN.SW", "NOVN.SW"),
        ("NOVN:SWX", "NOVN.SW"),
        ("NOVN.N-CH", "NOVN.SW"),
        ("AAPL", "AAPL"),
        ("7203.T", "7203.T"),
        ("INVALID", "INVALID"),
    ],
)
def test_to_yfinance(input_ticker, expected):
    result = to_yfinance(input_ticker)
    assert result == expected


@pytest.mark.parametrize(
    "input_ticker, expected",
    [
        ("NOVN.SW", "NOVN:SWX"),
        ("NOVN:SWX", "NOVN:SWX"),
        ("NOVN.N-CH", "NOVN:SWX"),
        ("AAPL", "AAPL:SMART"),
        ("7203.T", "7203:TSE"),
        ("INVALID", "INVALID:SMART"),
    ],
)
def test_to_ibkr(input_ticker, expected):
    result = to_ibkr(input_ticker)
    assert result == expected


@pytest.mark.parametrize(
    "input_ticker, expected_country, expected_exchange",
    [
        ("NOVN.SW", "Switzerland", "SIX Swiss Exchange"),
        ("7203.T", "Japan", "Tokyo Stock Exchange"),
        ("AAPL", "United States", "US Exchange (assumed)"),
        ("0700.HK", "Hong Kong", "Hong Kong Stock Exchange"),
        ("UNKNOWN.XX", "Unknown", "Unknown"),
    ],
)
def test_get_ticker_info(input_ticker, expected_country, expected_exchange):
    info = get_ticker_info(input_ticker)
    assert info["country"] == expected_country
    assert info["exchange_name"] == expected_exchange


@pytest.mark.parametrize(
    "ticker, expected",
    [
        ("AAPL", False),  # US
        ("NOVN.SW", True),  # International
        ("7203.T", True),
        ("UNKNOWN.XX", True),  # Defaults to Unknown -> True
        ("MSFT", False),
    ],
)
def test_is_international(ticker, expected):
    result = TickerFormatter.is_international(ticker)
    assert result == expected
