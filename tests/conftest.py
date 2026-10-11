"""Pytest configuration for Multi-Agent Trading System tests."""

import logging
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from scripts.check_macos_cloud_files import (
    collection_input_files,
    find_dataless_files,
    format_pytest_error,
    materialize_files,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# iCloud's Optimize Mac Storage can leave source, tests, and fixtures as
# ``compressed,dataless`` placeholders. Letting collection discover them one by one
# produces long silent stalls inside importlib and can look like a broken native
# library. Materialize public inputs before importing application modules; if macOS
# cannot fetch them, fail with the explicit public-file-only recovery command.
if sys.platform == "darwin":
    _repository_root = Path(__file__).parent.parent
    _collection_inputs = collection_input_files(_repository_root)
    _dataless_files = find_dataless_files(_collection_inputs)
    if _dataless_files:
        print(
            f"pytest preflight: materializing {len(_dataless_files)} "
            "macOS cloud placeholder(s)",
            file=sys.stderr,
        )
        materialize_files(_dataless_files)
        _dataless_files = find_dataless_files(_collection_inputs)
        if _dataless_files:
            raise pytest.UsageError(
                format_pytest_error(_dataless_files, _repository_root)
            )

# macOS fork-safety: apply the same mitigation the app uses, but at test-session
# import time (before any test forks). Several tests + scripts/find_gems.py use
# multiprocessing "spawn"; spawning a worker does fork()+exec(), and on macOS the
# child can SIGSEGV in Apple's Network.framework atfork handler before exec once
# the gRPC/proxy stack is active — a benign but dialog-popping forked-child crash.
# Guarded: no-op off macOS and when a proxy is configured (so proxied CI keeps
# working); only sets no_proxy='*' on clean machines. See src/config.py.
from src.config import _apply_macos_fork_safety_env  # noqa: E402

_apply_macos_fork_safety_env(enabled=True, platform=sys.platform)


# Re-register subdirectory conftest fixtures on a re-collected Directory; see
# tests/helpers/conftest_recollection.py for the pytest 9.1 mechanism.
from tests.helpers.conftest_recollection import (  # noqa: E402, F401
    pytest_make_collect_report,
)


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


@pytest.fixture(scope="session")
def test_chroma_persist_dir(tmp_path_factory) -> Path:
    """Allocate the only ChromaDB directory a pytest session may use.

    Kept as a named fixture so ``test_chroma_isolation.py`` can assert that the
    live config points to this exact temporary path, rather than merely to a path
    that is not the repository default. The latter would still permit an
    operator-supplied external Chroma directory to be destroyed.
    """
    return tmp_path_factory.mktemp("chroma_db_test")


@pytest.fixture(scope="session", autouse=True)
def setup_test_env(test_chroma_persist_dir: Path):
    """
    Set up test environment variables.
    This fixture runs for the entire session and applies default MOCK values.
    Individual tests that need real keys (integration tests) must override this.
    """
    # The test session must never touch the operator's persistent ChromaDB.
    #
    # This is not hygiene, it is data loss: `cleanup_all_memories(days=0)` with no
    # ticker skips its prefix filter and calls `delete_collection` on *every*
    # collection it finds, and `tests/memory/test_contamination_vectors.py` calls
    # it exactly that way (twice) to get a "clean slate". Pointed at the default
    # `./chroma_db`, a single `pytest tests/memory/` destroyed the entire
    # `lessons_learned` corpus along with `macro_events` and every legacy
    # collection — which is why the lessons store kept reappearing empty and
    # same-day, and why a stray `test_memory__emb_*` collection was found living
    # in the real database.
    #
    # Redirecting the whole session is the boundary fix: it protects against any
    # future test that constructs a real memory, not just the two known callers.
    test_env = {
        "CHROMA_PERSIST_DIR": str(test_chroma_persist_dir),
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "ERROR",
        "ENABLE_MEMORY": "false",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "GOOGLE_API_KEY": "test-key",  # Default to dummy key
        "TAVILY_API_KEY": "test-key",
        "FINNHUB_API_KEY": "test-key",
        # EDGAR URLs are refused without a contact address; the suite's sec.gov
        # fixtures need one, and an operator's real address must not leak in.
        "SEC_CONTACT_EMAIL": "tests@example.invalid",
        # Service tiers must be pinned to defaults: an operator .env with
        # GEMINI/OPENAI_SERVICE_TIER=flex flips llms.py construction paths
        # (the flex path goes through _build_openai_chat, and Gemini's flex
        # path bypasses ChatGoogleGenerativeAI mocks) and floors timeout
        # assertions (35 s quick budgets become 1350 s). Flex
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


class EmbeddingProviderUnreachable(ConnectionError):
    """What every embedding-SDK request raises in a unit test."""


_EMBEDDING_SDK_METHODS = (
    "embed_query",
    "embed_documents",
    "aembed_query",
    "aembed_documents",
)


@pytest.fixture(autouse=True)
def _embedding_provider_unreachable(request, monkeypatch):
    """Refuse every embedding-SDK request a unit test makes, before it is sent.

    The session's dummy ``GOOGLE_API_KEY`` is enough for ``FinancialSituationMemory``
    to build a real embeddings client and run its startup probe, so any test that
    constructs a memory without patching the client -- including, until October
    2026, every graph built with memory disabled -- sent an
    ``embed_content`` request to Google (57 requests from 19 tests in one full run;
    offline, each waited out the probe timeout). Refusing at the SDK method keeps
    the client real for construction assertions, fails the probe exactly as an
    unreachable provider would, and lets a test's own patch of the client or its
    transport win. Exempt: ``integration`` tests (they need the provider) and
    ``security`` tests (they install their own network block and wire fakes).
    Guarded by ``tests/memory/test_embedding_probe_offline.py``.
    """
    if request.node.get_closest_marker("integration") is not None:
        return
    if request.node.get_closest_marker("security") is not None:
        return

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import OpenAIEmbeddings

    def refuse(self, *args, **kwargs):
        raise EmbeddingProviderUnreachable(
            "embedding provider unreachable: unit tests run offline"
        )

    async def arefuse(self, *args, **kwargs):
        refuse(self)

    for client_class in (GoogleGenerativeAIEmbeddings, OpenAIEmbeddings):
        for method in _EMBEDDING_SDK_METHODS:
            replacement = arefuse if method.startswith("a") else refuse
            monkeypatch.setattr(client_class, method, replacement)


@pytest.fixture(autouse=True)
def _instant_process_limiters(request, monkeypatch):
    """Replace every process-fallback provider limiter with one that never waits.

    A unit test that sends provider requests through ``limiter_for_binding`` (no
    bound ``ProviderRuntime``) used to wait on the real Google limiter, paced at
    ``RPM * 0.8 / 60`` requests a second -- about 5 s per acquire after the first
    token. Three memory modules spent over 100 s of a full run doing so, and a
    test's wall time depended on which test ran before it. Limiters are built by
    ``create_process_rate_limiter``; replacing that (and the module-level copies
    that bind it) covers the fallback table, ``GLOBAL_RATE_LIMITER`` and the lazily
    built OpenAI limiter. Tests of the real limiter opt out with the
    ``real_rate_limiter`` marker; ``integration`` tests keep the real one.
    Guarded by ``tests/test_isolation_fixtures.py``.
    """
    if (
        request.node.get_closest_marker("integration") is not None
        or request.node.get_closest_marker("real_rate_limiter") is not None
    ):
        yield
        return

    from src import llms, runtime_services
    from src.llm_runtime import rate_limits
    from tests.helpers.instant_limiter import InstantLimiter

    def instant(*_args, **_kwargs) -> InstantLimiter:
        return InstantLimiter()

    rate_limits.reset_fallback_limiters_for_tests()
    monkeypatch.setattr(rate_limits, "create_process_rate_limiter", instant)
    monkeypatch.setattr(runtime_services, "create_process_rate_limiter", instant)
    monkeypatch.setattr(llms.GLOBAL_RATE_LIMITER, "_instance", None)
    monkeypatch.setattr(llms, "_openai_rate_limiter", None)
    monkeypatch.setattr(llms, "_openai_rate_limiter_initialized", False)
    yield
    rate_limits.reset_fallback_limiters_for_tests()


@pytest.fixture
def offline_main_flow(monkeypatch):
    """The advisory lookups ``src.main`` makes around a run, answered "unavailable".

    ``run_analysis`` fetches a benchmark market note, a 52-week price snapshot and
    the macro context (a Tavily search), and publication looks up the company name
    for the report header; each degrades to an empty or failed result when its
    provider is unreachable.
    Request it (``pytest.mark.usefixtures``) in any module that drives ``main`` or
    ``run_analysis`` with the graph mocked. A test asserting on one of them patches
    it again in its own body, which wins.
    """
    from unittest.mock import AsyncMock

    monkeypatch.setattr("src.main._fetch_market_context", AsyncMock(return_value=""))
    monkeypatch.setattr("src.main._fetch_price_snapshot", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "src.output._load_company_name_for_output_async",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "src.macro_context.get_macro_context",
        AsyncMock(side_effect=RuntimeError("macro context unavailable offline")),
    )


@pytest.fixture(autouse=True)
def _reset_shared_memory_state():
    """Drop the process-wide embeddings and Chroma clients between tests.

    ``FinancialSituationMemory`` caches its probed embeddings client and its Chroma
    client on the class. A test that left them populated made the next test's
    memories ``available`` without ever building a client (two
    ``TestMemoryIsolation`` tests skipped when run alone and passed only after an
    earlier test happened to leave state behind).
    """
    from src.memory import FinancialSituationMemory

    FinancialSituationMemory._reset_shared_state_for_tests()
    yield
    FinancialSituationMemory._reset_shared_state_for_tests()


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
