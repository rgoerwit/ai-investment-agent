"""Shared fixtures for the LLM binding/construction suite."""

import pytest


@pytest.fixture
def restore_runtime_config():
    """Undo any ``bind_runtime_config`` a test performs.

    The binding is a ContextVar, so a leaked one would silently re-point base
    seats for every later test in the session — the exact cross-test leakage the
    config-singleton pin exists to prevent, one layer over.
    """
    from src.runtime_config import _CURRENT_RUNTIME_CONFIG

    token = _CURRENT_RUNTIME_CONFIG.set(_CURRENT_RUNTIME_CONFIG.get())
    try:
        yield
    finally:
        _CURRENT_RUNTIME_CONFIG.reset(token)
