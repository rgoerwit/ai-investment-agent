"""Pytest configuration for Multi-Agent Trading System tests."""

import logging
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# macOS fork-safety: apply the same mitigation the app uses, but at test-session
# import time (before any test forks). Several tests + scripts/find_gems.py use
# multiprocessing "spawn"; spawning a worker does fork()+exec(), and on macOS the
# child can SIGSEGV in Apple's Network.framework atfork handler before exec once
# the gRPC/proxy stack is active — a benign but dialog-popping forked-child crash.
# Guarded: no-op off macOS and when a proxy is configured (so proxied CI keeps
# working); only sets no_proxy='*' on clean machines. See src/config.py.
from src.config import _apply_macos_fork_safety_env  # noqa: E402

_apply_macos_fork_safety_env(enabled=True, platform=sys.platform)


@pytest.fixture(autouse=True)
def _reset_ibkr_session_manager():
    """Reset the process-wide pooled IBKR session between tests.

    The pool is a singleton; without this, a fake client_cls injected by one test
    would stay connected and be reused by the next, bleeding state across tests.
    """
    from src.ibkr.session_manager import reset_ibkr_session_manager

    reset_ibkr_session_manager()
    yield
    reset_ibkr_session_manager()


@pytest.fixture(autouse=True)
def _reset_flex_health():
    """Reset the process-wide flex-health cache between tests.

    Same hazard as the IBKR pool: a degradation recorded by one test silently
    changes the tier another test's model is constructed with, and the failure
    only appears under full-suite ordering. The sibling capability cache
    (``_flex_unsupported_models``) has the same shape; tests that touch it reset
    it explicitly.
    """
    from src.service_tiers import _reset_flex_health_for_tests

    _reset_flex_health_for_tests()
    yield
    _reset_flex_health_for_tests()


# Capture real API key if present (for integration tests)
_REAL_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """
    Set up test environment variables.
    This fixture runs for the entire session and applies default MOCK values.
    Individual tests that need real keys (integration tests) must override this.
    """
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "ERROR",
        "ENABLE_MEMORY": "false",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "GOOGLE_API_KEY": "test-key",  # Default to dummy key
        "TAVILY_API_KEY": "test-key",
        "FINNHUB_API_KEY": "test-key",
        # Service tiers must be pinned to defaults: an operator .env with
        # GEMINI/OPENAI_SERVICE_TIER=flex flips llms.py construction paths
        # (flex subclasses bypass ChatOpenAI/ChatGoogleGenerativeAI mocks) and
        # floors timeout assertions (35 s quick budgets become 1350 s). Flex
        # behavior is tested explicitly via config patching in
        # tests/test_llms_flex.py / tests/test_service_tiers.py.
        "GEMINI_SERVICE_TIER": "standard",
        "OPENAI_SERVICE_TIER": "auto",
        "FLEX_FALLBACK_TO_STANDARD": "true",
        "FLEX_LLM_TIMEOUT_SECONDS": "900",
        # Same rationale, one layer down: an operator .env pointing
        # OPENAI_API_BASE at a compatible vendor makes every OpenAI-plane seat
        # take the compatible path — no service tier, no Responses API, and the
        # OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS client timeout. Tests for the
        # compatible path set the base explicitly (tests/test_llms_openai_base.py).
        "OPENAI_API_BASE": "",
        # NOTE: the LLM_*_PROVIDER schema selectors are deliberately NOT pinned
        # here. A process env var beats `_env_file`, which would break the tests
        # that load a purpose-built new-schema .env fixture. They are pinned on
        # the config singleton instead — see `_pin_legacy_binding_schema` below.
        # Pin the APEX-tier knobs to unset: an operator .env with APEX_MODEL
        # flips the llms.py construction path for the Senior Fundamentals and
        # PM seats (dedicated deep-tier instance instead of the quick/deep
        # tiers) under mock-based unit tests. Apex-path behavior is tested
        # explicitly via config patching in tests/test_llms_apex.py.
        "APEX_MODEL": "",
        "APEX_QUICK_MODEL": "",
        # Same rationale: an operator .env overriding the thinking level
        # breaks default-value assertions.
        "APEX_THINKING_LEVEL": "high",
    }

    # 1. Patch os.environ
    with patch.dict(os.environ, test_env, clear=False):
        # 2. Update the global config singleton IN-PLACE
        # We import here so the import happens AFTER os.environ is patched
        from src.config import Settings, config

        # Create a new Settings object which reads the NOW-PATCHED os.environ
        # (and ignores any local .env file because it's already loaded)
        new_settings = Settings()

        # Update the existing global singleton's state to match the new settings
        # This fixes the "import time" problem because we modify the object everyone is holding
        config.__dict__.update(new_settings.__dict__)

        yield


@pytest.fixture(autouse=True)
def _block_network_for_security_tests(request, monkeypatch):
    """Fail fast if security-marked tests attempt real socket access."""
    if request.node.get_closest_marker("security") is None:
        return

    request.getfixturevalue("socket_disabled")

    def blocked_name_resolution(*args, **kwargs):
        raise RuntimeError("network access blocked for security-marked tests")

    monkeypatch.setattr(socket, "getaddrinfo", blocked_name_resolution)


@pytest.fixture(autouse=True)
def _pin_config_singleton_identity():
    """Pin ``src.config.config`` to its session-baseline object identity.

    Some tests in this suite call ``importlib.reload(src.config)`` —
    typically to test pydantic validation errors or environment overrides.
    A reload swaps the module-level ``src.config.config`` for a brand-new
    Settings instance while every already-imported production module
    (``src.persistence``, ``src.agents.research_nodes``,
    ``src.retrospective``, …) still holds a reference to the OLD object.

    Subsequent tests that ``monkeypatch.setattr(config, …)`` then mutate
    a different instance from the one production reads, producing silent
    cross-test leakage that is invisible in single-file runs but fails
    only under full-suite ordering. The May 2026 cross-test leakage
    incident traced back to exactly this pattern.

    This fixture re-aliases ``src.config.config`` back to the original
    session-baseline object before every test. Cheap (~µs identity
    check), and makes future reload regressions self-healing.
    """
    import src.config

    baseline = getattr(_pin_config_singleton_identity, "_baseline", None)
    if baseline is None:
        _pin_config_singleton_identity._baseline = src.config.config
    else:
        if src.config.config is not baseline:
            src.config.config = baseline
    yield


@pytest.fixture(autouse=True)
def _pin_legacy_binding_schema(monkeypatch):
    """Keep the config singleton on the legacy LLM binding schema.

    Same hazard as the pinned service tiers and APEX models, one layer wider: a
    single ``LLM_*_PROVIDER`` selector in an operator's ``.env`` switches EVERY
    seat from the legacy factories to the provider-scoped adapters. Mock-based
    unit tests that patch a legacy facade (``components.get_consultant_llm``,
    ``create_apex_llm``, …) then observe no calls at all, and fail only on the
    machine whose ``.env`` has been migrated.

    Pinned on the singleton rather than via ``os.environ`` on purpose: a process
    env var outranks ``_env_file``, which would break the tests that load a
    purpose-built new-schema ``.env`` fixture. Multi-provider behavior is tested
    with explicit ``Settings(...)`` objects, which this fixture does not touch.
    """
    import src.config

    for field in (
        "llm_base_provider",
        "llm_review_provider",
        "llm_regional_provider",
        "llm_writer_provider",
        "llm_operational_provider",
        "llm_judge_provider",
    ):
        monkeypatch.setattr(src.config.config, field, None, raising=False)
    yield


@pytest.fixture(autouse=True)
def configure_structlog_for_tests():
    """Configure structlog for test environment."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            # FIX: Removed format_exc_info because ConsoleRenderer handles exceptions prettily
            # structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.root.setLevel(logging.WARNING)
    yield


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="BUY"))
    return mock


@pytest.fixture(autouse=True)
def skip_chart_generation_in_tests(monkeypatch):
    """Prevent chart generation from writing to ./images/ during tests.

    QuietModeReporter generates charts by default, writing to ./images/.
    This fixture patches the chart generation methods to no-op, preventing
    leftover files from accumulating in the repo.

    Tests that specifically need chart generation should use tmp_path
    and explicitly set image_dir.
    """
    from src.report_generator import QuietModeReporter

    monkeypatch.setattr(QuietModeReporter, "_generate_chart", lambda self, r: None)
    monkeypatch.setattr(
        QuietModeReporter, "_generate_radar_chart", lambda self, r: None
    )
