"""Tests for the macOS fork-safety env-var mitigation (src.config).

Guards the benign Network.framework atfork SIGSEGV mitigation: it must set the
proxy/ObjC env vars on macOS when safe, never override a proxy user, and stay a
no-op off macOS or when disabled.
"""

from __future__ import annotations

import pytest

from src.config import _apply_macos_fork_safety_env

_MANAGED = ("no_proxy", "NO_PROXY", "OBJC_DISABLE_INITIALIZE_FORK_SAFETY")
_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Clear managed + proxy vars (monkeypatch restores them after the test)."""
    for var in (*_MANAGED, *_PROXY_VARS):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_darwin_no_proxy_sets_all_mitigation_vars(clean_env):
    _apply_macos_fork_safety_env(enabled=True, platform="darwin")

    import os

    assert os.environ["no_proxy"] == "*"
    assert os.environ["NO_PROXY"] == "*"
    assert os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == "YES"


def test_darwin_with_proxy_does_not_force_no_proxy(clean_env):
    # Proxy users must keep proxying: no_proxy is NOT forced, but the ObjC guard
    # (which does not affect connectivity) is still applied.
    clean_env.setenv("HTTPS_PROXY", "http://corp-proxy:8080")

    _apply_macos_fork_safety_env(enabled=True, platform="darwin")

    import os

    assert "no_proxy" not in os.environ
    assert "NO_PROXY" not in os.environ
    assert os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == "YES"


def test_darwin_preserves_user_set_no_proxy(clean_env):
    clean_env.setenv("no_proxy", "example.com")

    _apply_macos_fork_safety_env(enabled=True, platform="darwin")

    import os

    assert os.environ["no_proxy"] == "example.com"  # not overwritten with "*"
    assert "NO_PROXY" not in os.environ  # not added when user already set no_proxy


def test_darwin_preserves_user_set_objc_flag(clean_env):
    clean_env.setenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "NO")

    _apply_macos_fork_safety_env(enabled=True, platform="darwin")

    import os

    assert os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == "NO"  # setdefault


def test_non_darwin_is_noop(clean_env):
    _apply_macos_fork_safety_env(enabled=True, platform="linux")

    import os

    for var in _MANAGED:
        assert var not in os.environ


def test_disabled_is_noop_on_darwin(clean_env):
    _apply_macos_fork_safety_env(enabled=False, platform="darwin")

    import os

    for var in _MANAGED:
        assert var not in os.environ


def test_settings_exposes_consumed_flag():
    # The field must exist (and be consumed by setup_environment) so it is not
    # flagged as dead config by tests/config/test_settings_hygiene.py.
    from src.config import Settings

    assert "macos_fork_safety_mitigation" in Settings.model_fields
    assert Settings.model_fields["macos_fork_safety_mitigation"].default is True
