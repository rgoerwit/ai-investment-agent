from datetime import date

from src.currency_resolver import CurrencyResolution, resolve_local_trading_currency


def test_resolve_suffixed_high_confidence():
    # PINFRA.MX -> MXN, exchange_suffix, high
    res = resolve_local_trading_currency(ticker="PINFRA.MX")
    assert res.code == "MXN"
    assert res.source == "exchange_suffix"
    assert res.confidence == "high"
    assert res.conflict_warning is None


def test_resolve_suffixed_conflict():
    # PINFRA.MX with provider_currency="USD"
    res = resolve_local_trading_currency(ticker="PINFRA.MX", provider_currency="USD")
    assert res.code == "MXN"
    assert res.source == "exchange_suffix"
    assert res.confidence == "high"
    assert res.conflict_warning is not None
    assert "MXN" in res.conflict_warning and "USD" in res.conflict_warning


def test_resolve_bare_ticker():
    # APR, no provider hint -> unresolved
    res = resolve_local_trading_currency(ticker="APR")
    assert res.code is None
    assert res.source == "unresolved"
    assert res.confidence == "low"


def test_resolve_bare_ticker_with_provider_currency():
    res = resolve_local_trading_currency(
        ticker="APR", provider_currency="USD", ibkr_exchange="NASDAQ"
    )
    assert res.code == "USD"
    assert res.source == "provider_currency"
    assert res.confidence == "medium"


def test_resolve_bare_ticker_with_ibkr_fallback():
    res = resolve_local_trading_currency(ticker="APR", ibkr_exchange="NASDAQ")
    assert res.code == "USD"
    assert res.source == "fallback"
    assert res.confidence == "low"


def test_resolve_bare_ticker_no_inference():
    # AAPL with no hints should not infer USD
    res = resolve_local_trading_currency(ticker="AAPL")
    assert res.code is None
    assert res.source == "unresolved"


def test_resolve_malformed():
    res = resolve_local_trading_currency(ticker="???")
    assert res.code is None
    assert res.source == "unresolved"


def test_resolve_none():
    res = resolve_local_trading_currency(ticker=None)
    assert res.code is None
    assert res.source == "unresolved"


def test_resolve_as_of_future_proofing():
    res = resolve_local_trading_currency(ticker="PINFRA.MX", as_of=date(2026, 1, 1))
    assert res.code == "MXN"
