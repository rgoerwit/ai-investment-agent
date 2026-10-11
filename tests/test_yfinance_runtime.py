"""Tests for shared yfinance runtime defaults."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yfinance as yf

import src.yfinance_runtime as yfinance_runtime
from tests.helpers.offline_network import install_offline_network


def _fake_yf(retries: int, hide_exceptions: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            network=SimpleNamespace(retries=retries),
            debug=SimpleNamespace(hide_exceptions=hide_exceptions),
        )
    )


def test_configure_yfinance_defaults_tolerates_missing_config(monkeypatch) -> None:
    monkeypatch.setattr(yfinance_runtime, "yf", SimpleNamespace(), raising=False)

    yfinance_runtime.configure_yfinance_defaults()


def test_configure_yfinance_defaults_tolerates_missing_debug_section(
    monkeypatch,
) -> None:
    fake_yf = SimpleNamespace(
        config=SimpleNamespace(network=SimpleNamespace(retries=3))
    )
    monkeypatch.setattr(yfinance_runtime, "yf", fake_yf, raising=False)

    yfinance_runtime.configure_yfinance_defaults()

    assert fake_yf.config.network.retries == 0


@pytest.mark.parametrize("starting_retries", [0, 1, 3, 5, 1000])
def test_configure_yfinance_defaults_pins_zero_retries_and_loud_errors(
    monkeypatch, starting_retries: int
) -> None:
    """The library must not retry: the application owns retries (and deadlines)."""
    fake_yf = _fake_yf(starting_retries)
    monkeypatch.setattr(yfinance_runtime, "yf", fake_yf, raising=False)

    yfinance_runtime.configure_yfinance_defaults()

    assert yfinance_runtime.YFINANCE_NETWORK_RETRIES == 0
    assert fake_yf.config.network.retries == 0
    assert fake_yf.config.debug.hide_exceptions is False


def test_configure_yfinance_defaults_is_idempotent(monkeypatch) -> None:
    fake_yf = _fake_yf(3)
    monkeypatch.setattr(yfinance_runtime, "yf", fake_yf, raising=False)

    yfinance_runtime.configure_yfinance_defaults()
    yfinance_runtime.configure_yfinance_defaults()

    assert fake_yf.config.network.retries == 0


def test_a_refused_yahoo_request_is_sent_once_not_retried(monkeypatch) -> None:
    """Counts the requests that leave the process (refused at libcurl).

    With the library's former setting of 3 retries one refused chart request
    was sent four times, sleeping 1 + 2 + 4 s between them.
    """
    monkeypatch.setattr(yf.config.network, "retries", 3)
    monkeypatch.setattr(yf.config.debug, "hide_exceptions", True)
    yfinance_runtime.configure_yfinance_defaults()
    attempts = install_offline_network(monkeypatch)

    try:
        yf.Ticker("AAPL").history(period="5d")
    except Exception:  # noqa: BLE001 - the refusal's type is not under test
        pass

    chart_requests = [a for a in attempts if "/v8/finance/chart/AAPL" in a]
    assert len(chart_requests) == 1, attempts
