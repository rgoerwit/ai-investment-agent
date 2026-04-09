"""Tests for optional yfinance runtime configuration."""

from __future__ import annotations

from types import SimpleNamespace

import src.yfinance_runtime as yfinance_runtime


def test_configure_yfinance_defaults_tolerates_missing_config(monkeypatch) -> None:
    monkeypatch.setattr(yfinance_runtime, "yf", SimpleNamespace(), raising=False)

    yfinance_runtime.configure_yfinance_defaults()


def test_configure_yfinance_defaults_sets_retries_and_debug(monkeypatch) -> None:
    fake_yf = SimpleNamespace(
        config=SimpleNamespace(
            network=SimpleNamespace(retries=0),
            debug=SimpleNamespace(hide_exceptions=True),
        )
    )
    monkeypatch.setattr(yfinance_runtime, "yf", fake_yf, raising=False)

    yfinance_runtime.configure_yfinance_defaults()

    assert fake_yf.config.network.retries == 3
    assert fake_yf.config.debug.hide_exceptions is False
