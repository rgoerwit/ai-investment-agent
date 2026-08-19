"""OPENAI_API_BASE routing for the OpenAI-compatible plane.

A custom base (e.g. Kimi/Moonshot) is injected once at the shared
`_construct_chat_openai` chokepoint, which covers the consultant, auditor,
editor, and writer-fallback seats. A custom base is treated as an OpenAI-
*compatible* (Chat Completions) endpoint, so the OpenAI-only Responses API
fields are dropped. The default (empty base) path must stay byte-identical.
"""

from unittest.mock import MagicMock, patch

import src.llms as llms_mod
from src.config import config


def _set_base(monkeypatch, value):
    """Patch openai_api_base on every live config binding (reload-proof)."""
    import src.config as config_module

    targets = {
        id(config_module.config): config_module.config,
        id(llms_mod.config): llms_mod.config,
        id(config): config,
    }
    for target in targets.values():
        monkeypatch.setattr(target, "openai_api_base", value, raising=False)


def test_apply_base_injects_and_downgrades_responses_api(monkeypatch):
    _set_base(monkeypatch, "https://api.moonshot.cn/v1")
    kwargs = {
        "model": "kimi-k2",
        "use_responses_api": True,
        "output_version": "responses/v1",
    }

    llms_mod._apply_openai_api_base(kwargs)

    assert kwargs["base_url"] == "https://api.moonshot.cn/v1"
    assert "use_responses_api" not in kwargs
    assert "output_version" not in kwargs


def test_apply_base_noop_when_unset(monkeypatch):
    _set_base(monkeypatch, "")
    kwargs = {
        "model": "gpt-5.4",
        "use_responses_api": True,
        "output_version": "responses/v1",
    }

    llms_mod._apply_openai_api_base(kwargs)

    assert "base_url" not in kwargs
    assert kwargs["use_responses_api"] is True
    assert kwargs["output_version"] == "responses/v1"


def test_official_openai_base_preserves_responses_api(monkeypatch):
    _set_base(monkeypatch, "https://api.openai.com/v1")
    kwargs = {
        "model": "gpt-5.4",
        "use_responses_api": True,
        "output_version": "responses/v1",
    }

    llms_mod._apply_openai_api_base(kwargs)

    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["use_responses_api"] is True
    assert kwargs["output_version"] == "responses/v1"


def test_apply_base_noop_for_bare_mock_config():
    """A bare-MagicMock ``config`` (get_openai_api_base returns a MagicMock, not
    a string) must not corrupt the default path — the guard acts only on a real
    URL string. Mirrors the wholesale ``@patch('src.llms.config')`` seat tests.
    """
    with patch("src.llms.config", MagicMock()):
        kwargs = {
            "model": "gpt-5.4",
            "use_responses_api": True,
            "output_version": "responses/v1",
        }
        llms_mod._apply_openai_api_base(kwargs)

    assert "base_url" not in kwargs
    assert kwargs["use_responses_api"] is True
    assert kwargs["output_version"] == "responses/v1"


def test_apply_base_strips_whitespace(monkeypatch):
    _set_base(monkeypatch, "  https://api.moonshot.cn/v1  ")
    kwargs = {"model": "kimi-k2"}

    llms_mod._apply_openai_api_base(kwargs)

    assert kwargs["base_url"] == "https://api.moonshot.cn/v1"


def test_construct_chat_openai_passes_custom_base(monkeypatch):
    _set_base(monkeypatch, "https://api.moonshot.cn/v1")
    fake_cls = MagicMock()

    with patch("langchain_openai.ChatOpenAI", fake_cls):
        llms_mod._construct_chat_openai(
            {
                "model": "kimi-k2",
                "use_responses_api": True,
                "output_version": "responses/v1",
            }
        )

    _, kwargs = fake_cls.call_args
    assert kwargs["base_url"] == "https://api.moonshot.cn/v1"
    assert "use_responses_api" not in kwargs
    assert "output_version" not in kwargs


def test_construct_chat_openai_default_keeps_responses_api(monkeypatch):
    _set_base(monkeypatch, "")
    fake_cls = MagicMock()

    with patch("langchain_openai.ChatOpenAI", fake_cls):
        llms_mod._construct_chat_openai(
            {
                "model": "gpt-5.4",
                "use_responses_api": True,
                "output_version": "responses/v1",
            }
        )

    _, kwargs = fake_cls.call_args
    assert "base_url" not in kwargs
    assert kwargs["use_responses_api"] is True


class TestCompatibleTimeoutIsIndependentOfServiceTier:
    """A pricing flag must not be the only way to buy a timeout.

    Before Aug 2026 ``_apply_openai_service_tier`` owned both the tier *and* the
    client-timeout floor, so an operator pointing OPENAI_API_BASE at Moonshot had
    to set ``OPENAI_SERVICE_TIER=flex`` — a product Moonshot does not sell — to
    stop kimi calls dying at the OpenAI-shaped default. The two concerns are now
    separate: the tier is skipped entirely for a non-OpenAI base, and the timeout
    comes from ``OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS``.
    """

    def test_compatible_base_takes_its_own_timeout(self, monkeypatch):
        _set_base(monkeypatch, "https://api.moonshot.ai/v1")
        monkeypatch.setattr(
            llms_mod.config, "openai_compatible_client_timeout_seconds", 275
        )
        kwargs = {"model": "kimi-k3", "timeout": 120}

        llms_mod._apply_openai_api_base(kwargs)

        assert kwargs["timeout"] == 275

    def test_openai_proper_keeps_its_seat_timeout(self, monkeypatch):
        _set_base(monkeypatch, "")
        monkeypatch.setattr(
            llms_mod.config, "openai_compatible_client_timeout_seconds", 275
        )
        kwargs = {"model": "gpt-5.4", "timeout": 120}

        llms_mod._apply_openai_api_base(kwargs)

        assert kwargs["timeout"] == 120

    def test_service_tier_is_not_sent_to_a_compatible_vendor(self, monkeypatch):
        _set_base(monkeypatch, "https://api.moonshot.ai/v1")
        monkeypatch.setattr(llms_mod.config, "openai_service_tier", "flex")
        kwargs = {"model": "kimi-k3", "timeout": 120}

        llms_mod._apply_openai_service_tier(kwargs, label="test")

        assert "service_tier" not in kwargs
        assert "flex_fallback_to_standard" not in kwargs
        # The flex floor must not be what stretches a compatible client.
        assert kwargs["timeout"] == 120

    def test_openai_proper_flex_still_floors_the_timeout(self, monkeypatch):
        _set_base(monkeypatch, "")
        monkeypatch.setattr(llms_mod.config, "openai_service_tier", "flex")
        kwargs = {"model": "gpt-5.4", "timeout": 120}

        llms_mod._apply_openai_service_tier(kwargs, label="test")

        assert kwargs["service_tier"] == "flex"
        assert kwargs["timeout"] > 120
