"""Integration prerequisite detection must never mistake a pytest-env dummy for a key."""

from __future__ import annotations

import os

import pytest

from tests.helpers.integration_prerequisites import (
    missing_credential_reason,
    pytest_env_dummies,
    real_credential,
)

DUMMIES = {"GOOGLE_API_KEY": "dummy-value", "TAVILY_API_KEY": "dummy-value"}


def test_dummies_are_read_from_pyproject_and_match_the_live_environment():
    dummies = pytest_env_dummies()

    assert {"GOOGLE_API_KEY", "TAVILY_API_KEY", "FINNHUB_API_KEY"} <= set(dummies)
    # pytest-env applied exactly these values, so the ambient key is a dummy.
    assert os.environ["GOOGLE_API_KEY"] == dummies["GOOGLE_API_KEY"]
    assert (
        real_credential("GOOGLE_API_KEY", {"GOOGLE_API_KEY": dummies["GOOGLE_API_KEY"]})
        is None
    )


def test_ambient_environment_has_no_real_key_under_pytest_env():
    if os.environ.get("INTEGRATION_GOOGLE_API_KEY"):
        pytest.skip("operator supplied a real integration key")
    assert real_credential("GOOGLE_API_KEY") is None


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"GOOGLE_API_KEY": ""},
        {"GOOGLE_API_KEY": "   "},
        {"GOOGLE_API_KEY": "dummy-value"},
        {"INTEGRATION_GOOGLE_API_KEY": "dummy-value"},
        {"INTEGRATION_GOOGLE_API_KEY": "", "GOOGLE_API_KEY": "dummy-value"},
    ],
)
def test_missing_blank_or_dummy_values_are_not_credentials(environ):
    assert real_credential("GOOGLE_API_KEY", environ, DUMMIES) is None


def test_integration_variable_wins_over_the_ambient_one():
    environ = {"INTEGRATION_GOOGLE_API_KEY": " real-a ", "GOOGLE_API_KEY": "real-b"}

    assert real_credential("GOOGLE_API_KEY", environ, DUMMIES) == "real-a"


def test_real_ambient_key_is_used_when_pytest_env_is_not_active():
    assert (
        real_credential("GOOGLE_API_KEY", {"GOOGLE_API_KEY": "real-b"}, DUMMIES)
        == "real-b"
    )


def test_dummy_integration_variable_falls_through_to_a_real_ambient_key():
    environ = {"INTEGRATION_GOOGLE_API_KEY": "dummy-value", "GOOGLE_API_KEY": "real-b"}

    assert real_credential("GOOGLE_API_KEY", environ, DUMMIES) == "real-b"


def test_name_without_a_configured_dummy_accepts_any_non_blank_value():
    assert real_credential("OTHER_KEY", {"OTHER_KEY": "x"}, DUMMIES) == "x"


def test_skip_reason_names_the_prerequisite_and_no_value(monkeypatch):
    monkeypatch.setenv("INTEGRATION_GOOGLE_API_KEY", "secret-should-not-print")

    reason = missing_credential_reason("GOOGLE_API_KEY")

    assert "INTEGRATION_GOOGLE_API_KEY" in reason
    assert "secret-should-not-print" not in reason
    assert pytest_env_dummies()["GOOGLE_API_KEY"] not in reason
