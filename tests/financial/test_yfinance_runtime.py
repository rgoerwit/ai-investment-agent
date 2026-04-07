"""Tests for shared yfinance runtime defaults."""

import yfinance as yf

from src.yfinance_runtime import configure_yfinance_defaults


def test_configure_yfinance_defaults_sets_retries_and_debug_flags():
    original_retries = yf.config.network.retries
    original_hide_exceptions = yf.config.debug.hide_exceptions
    try:
        yf.config.network.retries = 0
        yf.config.debug.hide_exceptions = True

        configure_yfinance_defaults()

        assert yf.config.network.retries == 3
        assert yf.config.debug.hide_exceptions is False
    finally:
        yf.config.network.retries = original_retries
        yf.config.debug.hide_exceptions = original_hide_exceptions


def test_configure_yfinance_defaults_does_not_lower_existing_retry_budget():
    original_retries = yf.config.network.retries
    original_hide_exceptions = yf.config.debug.hide_exceptions
    try:
        yf.config.network.retries = 5
        yf.config.debug.hide_exceptions = True

        configure_yfinance_defaults()

        assert yf.config.network.retries == 5
        assert yf.config.debug.hide_exceptions is False
    finally:
        yf.config.network.retries = original_retries
        yf.config.debug.hide_exceptions = original_hide_exceptions
