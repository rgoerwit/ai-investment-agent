"""
Configuration module using Pydantic Settings.

Provides validated, type-safe configuration from environment variables.
Uses fail-fast validation - the app crashes immediately if required
environment variables are missing or have invalid types.

Migration Notes (Dec 2025):
- Migrated from dataclass to Pydantic Settings for better validation
- All existing attribute names preserved for backwards compatibility
- SecretStr used for API keys to prevent accidental logging
- Config class alias maintained for backwards compatibility
"""

import functools
import logging
import os
import sys
from pathlib import Path
from typing import Literal

import structlog
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Note: Pydantic Settings handles .env loading natively via env_file in SettingsConfigDict.
# No manual load_dotenv() needed - it's cleaner and avoids double-loading issues.

# --- Logging Setup (must happen before Settings to capture validation errors) ---
logging.basicConfig(
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    stream=sys.stderr,
    level=logging.INFO,
    force=True,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.KeyValueRenderer(
            key_order=["timestamp", "level", "event"]
        ),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = logging.getLogger(__name__)


# --- Helper Functions (preserved for backwards compatibility with tests) ---


def _get_env_var(var: str, required: bool = True, default: str | None = None) -> str:
    """Get environment variable with validation.

    Preserved for backwards compatibility with existing tests.
    New code should use the Settings class directly.
    """
    value = os.environ.get(var, default)
    if required and not value:
        logger.error(f"Missing required environment variable: {var}")
        return ""
    return value or ""


def configure_langsmith_tracing(settings: "Settings") -> None:
    """Log LangSmith tracing configuration status.

    Note (Dec 2025): LangSmith SDK auto-detects configuration from environment
    variables (LANGSMITH_API_KEY, LANGSMITH_PROJECT, etc.) via Pydantic Settings.
    The Settings class provides defaults for LANGSMITH_PROJECT and LANGSMITH_ENDPOINT.
    This function only logs the configuration status for visibility - it does NOT
    set any environment variables (the SDK handles auto-detection).

    Args:
        settings: Settings instance (required - no more os.environ fallback).
    """
    has_api_key = bool(settings.get_langsmith_api_key())
    project_name = settings.langsmith_project
    tracing_enabled = bool(getattr(settings, "langsmith_tracing_enabled", False))

    if has_api_key and tracing_enabled:
        # LangSmith SDK auto-detects from Pydantic Settings - just log for visibility
        logger.info(f"LangSmith tracing enabled for project: {project_name}")


def _parse_env_file() -> dict[str, str]:
    """Parse .env file to get explicitly set values (ignoring comments and blank lines).

    Preserved for backwards compatibility with existing tests.
    """
    env_file = Path(".env")
    env_values: dict[str, str] = {}

    if not env_file.exists():
        return env_values

    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Strip inline comments (everything after #)
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Store non-empty values
                    if value:
                        env_values[key] = value
    except Exception as e:
        logger.warning(f"Could not parse .env file: {e}")

    return env_values


@functools.lru_cache(maxsize=1)
def _cached_env_file_values() -> dict[str, str]:
    """Memoized accessor for `.env` values; cleared by tests via cache_clear()."""
    return _parse_env_file()


def get_env_value(name: str) -> str | None:
    """Resolve an environment variable by name with `.env` fallback.

    The repo loads `.env` via pydantic-settings into the ``Settings`` model;
    raw ``os.getenv`` does not see those values. Code paths that look up env
    vars by *name* (e.g., MCP server auth, where the variable name is data-
    driven from the registry) should use this helper so the `.env` file is
    honored consistently with shell-exported values winning when both are set.
    """
    value = os.getenv(name)
    if value:
        return value
    return _cached_env_file_values().get(name)


def _check_env_overrides() -> None:
    """Check for shell environment variable overrides that conflict with .env file.

    Preserved for backwards compatibility with existing tests.
    """
    env_file_values = _parse_env_file()

    # Critical variables to check (where override could cause performance issues)
    critical_vars = {
        "GEMINI_RPM_LIMIT": {
            "name": "GEMINI_RPM_LIMIT",
            "description": "Gemini API rate limit",
            "comparison": "higher",  # Shell override higher than .env is problematic
        }
    }

    for var_key, var_info in critical_vars.items():
        env_file_value = env_file_values.get(var_key)
        shell_value = os.environ.get(var_key)

        # Skip if not set in .env file
        if not env_file_value:
            continue

        # Skip if shell value matches .env value
        if shell_value == env_file_value:
            continue

        # Shell environment override detected
        if shell_value:
            try:
                env_file_int = int(env_file_value)
                shell_int = int(shell_value)

                # Check if override is problematic
                if var_info["comparison"] == "higher" and shell_int > env_file_int:
                    logger.warning(
                        f"SHELL ENVIRONMENT OVERRIDE DETECTED: {var_key}\n"
                        f"    .env file setting:     {var_key}={env_file_int}\n"
                        f"    Shell environment:     {var_key}={shell_int}\n"
                        f"    USING: {shell_int} (from shell - this may cause rate limit errors!)\n"
                        f"    \n"
                        f"    If you have a free-tier API key ({env_file_int} RPM), using {shell_int} RPM\n"
                        f"    will cause HTTP 429 errors and severe performance degradation.\n"
                        f"    \n"
                        f"    To fix: Run 'unset {var_key}' in your shell, or check ~/.bashrc, ~/.zshrc"
                    )
                elif shell_int != env_file_int:
                    logger.info(
                        f"Shell environment override: {var_key}={shell_int} (overrides .env value of {env_file_int})"
                    )
            except ValueError:
                # Non-integer values
                logger.info(
                    f"Shell environment override: {var_key}={shell_value} (overrides .env value of {env_file_value})"
                )


def _apply_macos_fork_safety_env(*, enabled: bool, platform: str) -> None:
    """Export env vars that prevent benign macOS fork-safety crashes.

    The gRPC/Gemini stack loads Apple Network.framework and makes the process
    multi-threaded; a later ``fork()+exec()`` (subprocess / multiprocessing-spawn
    from a transitive dep) crashes in Network's atfork handler before ``exec`` —
    a SIGSEGV in a short-lived forked child that is harmless but pops a macOS
    crash dialog. Disabling proxy auto-discovery short-circuits the
    SystemConfiguration/Network.framework lookup that trips the handler;
    ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY`` covers the ObjC +initialize variant.

    No-op off macOS, when disabled, when a proxy is configured (proxy users must
    keep proxying), or when the user already set the relevant vars.
    """
    if not enabled or platform != "darwin":
        return
    proxy_vars = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    if (
        not any(os.environ.get(v) for v in proxy_vars)
        and "no_proxy" not in os.environ
        and "NO_PROXY" not in os.environ
    ):
        os.environ["no_proxy"] = "*"
        os.environ["NO_PROXY"] = "*"
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


def validate_environment_variables() -> None:
    """Validate required environment variables.

    This function is preserved for backwards compatibility with main.py.
    With Pydantic Settings, most validation happens at Settings instantiation,
    but this function still handles:
    - Warning about optional EODHD key
    - Checking for problematic shell overrides
    - Configuring LangSmith tracing

    Note: Uses the 'config' singleton (created at module load) to check API keys.
    This ensures .env values are properly loaded via Pydantic Settings.
    """
    # Use the config singleton to check API keys (loaded via Pydantic Settings)
    # This avoids dependency on load_dotenv() polluting os.environ
    required_checks = [
        ("GOOGLE_API_KEY", config.get_google_api_key),
        ("FINNHUB_API_KEY", config.get_finnhub_api_key),
        ("TAVILY_API_KEY", config.get_tavily_api_key),
    ]

    # Check for EODHD key (Optional but recommended)
    if not config.get_eodhd_api_key():
        logger.warning(
            "EODHD_API_KEY missing - High quality international data will be disabled."
        )

    missing_vars = [name for name, getter in required_checks if not getter()]

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Check for problematic shell environment overrides
    _check_env_overrides()

    configure_langsmith_tracing(settings=config)
    logger.info("Environment variables validated")


# --- Pydantic Settings Class ---


class Settings(BaseSettings):
    """
    Configuration class for the Multi-Agent Trading System.

    Uses Pydantic Settings for validated, type-safe configuration from
    environment variables. Provides fail-fast validation at startup.

    API keys use SecretStr to prevent accidental logging. Use the
    get_*_api_key() methods to retrieve the actual values.
    """

    # --- Directory Paths ---
    results_dir: Path = Field(
        default=Path("./results"),
        validation_alias="RESULTS_DIR",
        description="Directory for analysis result files",
    )
    data_cache_dir: Path = Field(
        default=Path("./data_cache"),
        validation_alias="DATA_CACHE_DIR",
        description="Directory for cached data files",
    )
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        validation_alias="CHROMA_PERSIST_DIR",
        description="Directory for ChromaDB vector storage",
    )
    images_dir: Path = Field(
        default=Path("images"),
        validation_alias="IMAGES_DIR",
        description="Directory for generated chart images",
    )

    # --- LLM Configuration ---
    llm_provider: str = Field(
        default="google",
        validation_alias="LLM_PROVIDER",
        description="LLM provider (google, openai, anthropic)",
    )
    deep_think_llm: str = Field(
        default="gemini-3.1-pro-preview",
        validation_alias="DEEP_MODEL",
        description="Model for deep thinking/synthesis agents",
    )
    # Flash models work with langchain-google-genai 4.0.0+
    quick_think_llm: str = Field(
        default="gemini-3-flash-preview",
        validation_alias="QUICK_MODEL",
        description="Model for quick thinking/data gathering agents",
    )

    # --- Debate & Risk Configuration ---
    max_debate_rounds: int = Field(
        default=2,
        ge=1,
        le=2,
        validation_alias="MAX_DEBATE_ROUNDS",
        description="Maximum rounds of bull/bear debate (1 or 2; graph supports R1/R2 only)",
    )

    # --- Feature Flags ---
    online_tools: bool = Field(
        default=True,
        validation_alias="ONLINE_TOOLS",
        description="Enable online data fetching tools",
    )
    enable_memory: bool = Field(
        default=True,
        validation_alias="ENABLE_MEMORY",
        description="Enable ChromaDB memory system",
    )
    enable_consultant: bool = Field(
        default=True,
        validation_alias="ENABLE_CONSULTANT",
        description="Enable OpenAI consultant for cross-validation",
    )
    # --- BUY stability / hysteresis gate (off-watchlist new BUYs) ---
    # Default ON since the gate was decoupled from src.agents (the enabled path is
    # now agents-free — src/ibkr/buy_stability.py + the neutral pm_decision_parser).
    # A fresh off-watchlist BUY contradicted by a recent same-ticker run (or
    # marginal with an unresolved peak/transient flag) is withheld pending
    # stability. Set BUY_STABILITY_ENABLED=false to opt out.
    buy_stability_enabled: bool = Field(
        default=True,
        validation_alias="BUY_STABILITY_ENABLED",
        description="Withhold unstable/marginal off-watchlist BUYs (reproducibility gate)",
    )
    buy_stability_lookback_days: int = Field(
        default=7,
        ge=1,
        validation_alias="BUY_STABILITY_LOOKBACK_DAYS",
        description="Lookback window (days) for same-ticker verdict stability checks",
    )
    buy_stability_margin_tally: float = Field(
        default=0.5,
        ge=0.0,
        validation_alias="BUY_STABILITY_MARGIN_TALLY",
        description="Risk-tally at/above which a BUY is 'marginal' for the stability gate",
    )
    # --- Consultant Configuration ---
    consultant_model: str = Field(
        default="gpt-5.4",
        validation_alias="CONSULTANT_MODEL",
        description="OpenAI model for consultant in normal mode",
    )
    consultant_quick_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias="CONSULTANT_QUICK_MODEL",
        description="OpenAI model for consultant in quick mode",
    )
    consultant_tools_in_quick: bool = Field(
        default=False,
        validation_alias="CONSULTANT_TOOLS_IN_QUICK",
        description="Allow Consultant MCP/tool loop during --quick screening runs",
    )
    # P1-5: lowered 60s → 35s. Saved-artifact data shows Consultant finishes
    # well under 30s when it returns at all; the 60s cap mostly hid hung
    # provider calls. The Consultant gate (P0-2) already short-circuits the
    # cheap "clean consensus" and "RM-negative" cases, so the residual
    # invocations are exactly the adversarial reviews where a tighter deadline
    # is acceptable.
    consultant_quick_total_timeout_seconds: float = Field(
        default=35.0,
        gt=0.0,
        validation_alias="CONSULTANT_QUICK_TOTAL_TIMEOUT_SECONDS",
        description="Total wall-clock Consultant budget in --quick mode",
    )
    consultant_quick_max_completion_tokens: int = Field(
        default=4096,
        ge=1024,
        validation_alias="CONSULTANT_QUICK_MAX_COMPLETION_TOKENS",
        description="Maximum Consultant completion token budget in --quick mode",
    )
    auditor_model: str | None = Field(
        default=None,
        validation_alias="AUDITOR_MODEL",
        description="Model for the auditor agent (optional)",
    )
    auditor_quick_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias="AUDITOR_QUICK_MODEL",
        description="Model for the auditor agent in --quick mode",
    )
    enable_apac_specialist: bool = Field(
        default=False,
        validation_alias="ENABLE_APAC_SPECIALIST",
        description="Enable optional APAC Regional Specialist audit node",
    )
    apac_specialist_model: str = Field(
        default="deepseek-v4-pro",
        validation_alias="APAC_SPECIALIST_MODEL",
        description="Model for the optional APAC Regional Specialist",
    )
    apac_specialist_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias="APAC_SPECIALIST_BASE_URL",
        description="OpenAI-compatible base URL for the APAC specialist",
    )
    apac_specialist_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="APAC_SPECIALIST_API_KEY",
        description="API key for the optional APAC Regional Specialist",
    )
    editor_model: str | None = Field(
        default=None,
        validation_alias="EDITOR_MODEL",
        description="OpenAI model for Editor-in-Chief article revision (optional)",
    )

    # --- Writer Configuration (Anthropic/Claude) ---
    claude_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="CLAUDE_KEY",
        description="Anthropic API key for article writer (optional, falls back to Gemini)",
    )
    writer_model: str = Field(
        default="claude-opus-4-6",
        validation_alias="WRITER_MODEL",
        description="Claude model for article writing",
    )

    # --- Capital Efficiency Thresholds ---
    # Used by fetcher and red_flag_detector for ROIC/leverage quality checks
    roic_hurdle_rate: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        validation_alias="ROIC_HURDLE_RATE",
        description="Minimum acceptable ROIC (proxy for WACC). Default 8%.",
    )
    roic_strong_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        validation_alias="ROIC_STRONG_THRESHOLD",
        description="ROIC threshold for 'strong' capital efficiency. Default 15%.",
    )
    leverage_suspect_ratio: float = Field(
        default=2.0,
        ge=1.0,
        validation_alias="LEVERAGE_SUSPECT_RATIO",
        description="ROE/ROIC ratio above which returns are suspect. Default 2.0.",
    )
    leverage_engineered_ratio: float = Field(
        default=3.0,
        ge=1.0,
        validation_alias="LEVERAGE_ENGINEERED_RATIO",
        description="ROE/ROIC ratio above which returns are likely engineered. Default 3.0.",
    )
    idle_cash_net_cash_to_mc_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
        validation_alias="IDLE_CASH_NET_CASH_TO_MC_THRESHOLD",
        description="Net cash / market cap threshold for idle-cash risk. Default 25%.",
    )
    idle_cash_severe_net_cash_to_mc_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=5.0,
        validation_alias="IDLE_CASH_SEVERE_NET_CASH_TO_MC_THRESHOLD",
        description="Severe idle-cash threshold for net cash / market cap. Default 40%.",
    )
    idle_cash_cash_to_assets_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        validation_alias="IDLE_CASH_CASH_TO_ASSETS_THRESHOLD",
        description="Cash / total assets threshold for idle-cash risk. Default 20%.",
    )
    idle_cash_min_payout_ratio: float = Field(
        default=20.0,
        ge=0.0,
        le=1000.0,
        validation_alias="IDLE_CASH_MIN_PAYOUT_RATIO",
        description="Minimum payout ratio before retained cash is treated as weak shareholder return. Default 20%.",
    )
    capex_to_da_underinvesting_threshold: float = Field(
        default=0.75,
        ge=0.0,
        validation_alias="CAPEX_TO_DA_UNDERINVESTING_THRESHOLD",
        description="Capex / D&A ratio below which reinvestment is treated as underinvesting. Default 0.75.",
    )
    capex_to_da_growth_threshold: float = Field(
        default=1.25,
        ge=0.0,
        validation_alias="CAPEX_TO_DA_GROWTH_THRESHOLD",
        description="Capex / D&A ratio above which reinvestment is treated as growth investing. Default 1.25.",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # --- API Configuration ---
    # Timeout from 120 -> 300 seconds (5 minutes) to handle massive prefill
    api_timeout: int = Field(
        default=300,
        ge=1,
        validation_alias="API_TIMEOUT",
        description="API request timeout in seconds",
    )
    # Quick-mode screening should not inherit a 5-minute SDK socket/read
    # timeout. The outer runtime helper still enforces the hard wall-clock cap;
    # this SDK timeout catches provider stalls earlier and avoids 300s slow-tail
    # failures from quick Gemini calls.
    quick_llm_api_timeout_seconds: int = Field(
        default=120,
        ge=1,
        validation_alias="QUICK_LLM_API_TIMEOUT_SECONDS",
        description="Provider SDK timeout in seconds for quick-mode Gemini LLMs",
    )
    # Lowered from 10 -> 2: at 10 retries × 300s timeout, a single hung Gemini
    # call could legitimately consume 50min before raising. The outer
    # `invoke_with_rate_limit_handling` already retries 3× with backoff, so
    # SDK-level retries are deliberately tight. Bump if you see frequent
    # transient 5xx errors that aren't being absorbed.
    api_retry_attempts: int = Field(
        default=2,
        ge=0,
        validation_alias="API_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed API calls",
    )
    # Wall-clock ceiling for a single `runnable.ainvoke()` call. Enforced via
    # `run_with_hard_timeout` in `invoke_with_rate_limit_handling` so a hung
    # provider SDK can't park a worker for hours. Keep tight for screening; use
    # env override if a specific deep/manual run needs more headroom.
    llm_call_hard_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        validation_alias="LLM_CALL_HARD_TIMEOUT_SECONDS",
        description=(
            "Hard wall-clock cap (seconds) for a single LLM ainvoke; "
            "raised as TimeoutError if exceeded."
        ),
    )
    # Tighter per-call cap for --quick mode. Quick Gemini Flash calls should
    # resolve in 10-30s; 60s surfaces a hung provider ~2x faster than the
    # normal cap, cutting worst-case stage-1 screening time significantly.
    quick_llm_call_hard_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        validation_alias="QUICK_LLM_CALL_HARD_TIMEOUT_SECONDS",
        description=(
            "Hard wall-clock cap (seconds) for a single LLM ainvoke in --quick mode."
        ),
    )

    # --- LLM circuit breaker (P2-7) ----------------------------------------
    # When a provider/model starts serving back-to-back hard-timeouts (e.g.,
    # regional Gemini Flash degradation), the breaker short-circuits the
    # next call with CircuitOpenError instead of waiting another full
    # timeout. In-memory only; reset per process.
    llm_circuit_breaker_enabled: bool = Field(
        default=True,
        validation_alias="LLM_CIRCUIT_BREAKER_ENABLED",
        description="Enable per-(agent, provider, model) circuit breaker on chronic timeouts",
    )
    llm_circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        validation_alias="LLM_CIRCUIT_BREAKER_THRESHOLD",
        description="Consecutive timeout failures within the window before the breaker opens",
    )
    llm_circuit_breaker_window_seconds: float = Field(
        default=300.0,
        gt=0.0,
        validation_alias="LLM_CIRCUIT_BREAKER_WINDOW_SECONDS",
        description="Sliding-window length (s) for counting timeout failures",
    )
    llm_circuit_breaker_cool_off_seconds: float = Field(
        default=60.0,
        gt=0.0,
        validation_alias="LLM_CIRCUIT_BREAKER_COOL_OFF_SECONDS",
        description="Cool-off (s) before the breaker enters half-open to probe recovery",
    )
    # --- Network breaker (process-global, host-level DNS/connect failures) --
    # Distinct from `llm_circuit_breaker_*` (per-(agent,provider,model),
    # timeout-only). This breaker watches dns_resolution + connect_error
    # across ALL contexts so a host network outage doesn't cause every
    # parallel analyst to independently burn its retry budget. Tuned tight:
    # 4 failures in 30s opens; cool_off 45s. See src/agents/network_breaker.py.
    network_breaker_enabled: bool = Field(
        default=True,
        validation_alias="NETWORK_BREAKER_ENABLED",
        description="Enable process-global circuit breaker on DNS/connect failures",
    )
    network_breaker_threshold: int = Field(
        default=4,
        ge=1,
        validation_alias="NETWORK_BREAKER_THRESHOLD",
        description="DNS/connect failures within the window before the breaker opens",
    )
    network_breaker_window_seconds: float = Field(
        default=30.0,
        gt=0.0,
        validation_alias="NETWORK_BREAKER_WINDOW_SECONDS",
        description="Sliding-window length (s) for counting network failures",
    )
    network_breaker_cool_off_seconds: float = Field(
        default=45.0,
        gt=0.0,
        validation_alias="NETWORK_BREAKER_COOL_OFF_SECONDS",
        description="Cool-off (s) before the network breaker probes recovery",
    )
    # Hard ceiling on shutdown cleanup (cleanup_async_resources). Protects
    # against httpx.AsyncClient.aclose() and similar paths that block on
    # dead sockets when the host's DNS / network is down — observed during
    # the May 2026 overnight macOS DNS outage where a "completed" run sat
    # in the finally block for minutes waiting on socket close.
    shutdown_hard_timeout_seconds: float = Field(
        default=15.0,
        gt=0.0,
        validation_alias="SHUTDOWN_HARD_TIMEOUT_SECONDS",
        description=(
            "Hard wall-clock cap (seconds) for cleanup_async_resources at "
            "process shutdown; on timeout the process forces os._exit()."
        ),
    )
    llm_base_output_tokens: int = Field(
        default=32768,
        ge=1024,
        validation_alias="LLM_BASE_OUTPUT_TOKENS",
        description=(
            "Global base output-token budget used to derive per-agent caps. "
            "Increasing this scales fractional agent budgets proportionally."
        ),
    )
    llm_default_reasoning_reserve_tokens: int = Field(
        default=2048,
        ge=0,
        validation_alias="LLM_DEFAULT_REASONING_RESERVE_TOKENS",
        description=(
            "Additional API-side token reserve for models whose hidden reasoning "
            "shares the same output/completion token pool."
        ),
    )
    llm_deep_reasoning_reserve_tokens: int = Field(
        default=8192,
        ge=0,
        validation_alias="LLM_DEEP_REASONING_RESERVE_TOKENS",
        description=(
            "Additional API-side token reserve for deep-model paths when hidden "
            "reasoning shares the same output/completion token pool."
        ),
    )

    # --- Rate Limiting ---
    # Free tier: 15 RPM | Paid tier 1: 360 RPM | Tier 2: 1000+ RPM
    gemini_rpm_limit: int = Field(
        default=15,
        ge=1,
        validation_alias="GEMINI_RPM_LIMIT",
        description="Gemini API rate limit (requests per minute)",
    )
    # OpenAI RPM limit for consultant/auditor/editor LLMs. None (default) means
    # no rate limiter is attached — set OPENAI_RPM_LIMIT in .env to enable.
    openai_rpm_limit: int | None = Field(
        default=None,
        ge=1,
        validation_alias="OPENAI_RPM_LIMIT",
        description="OpenAI API rate limit (requests per minute); unset = no throttle",
    )

    # --- Token Management ---
    # Default: 7000 chars (~1750 tokens) per search result
    tavily_max_chars: int = Field(
        default=7000,
        ge=100,
        validation_alias="TAVILY_MAX_CHARS",
        description="Maximum characters per Tavily search result",
    )

    # --- Environment ---
    environment: str = Field(
        default="dev",
        validation_alias="ENVIRONMENT",
        description="Environment (dev, prod, test)",
    )
    app_release: str = Field(
        default="3.1.0",
        validation_alias="APP_RELEASE",
        description="Application release/version tag for observability",
    )

    # --- Runtime Flags ---
    quiet_mode: bool = Field(
        default=False,
        validation_alias="QUIET_MODE",
        description="Suppress verbose logging output (set via CLI --quiet)",
    )

    # --- Untrusted Content Inspection ---
    untrusted_content_inspection_enabled: bool = Field(
        default=False,
        validation_alias="UNTRUSTED_CONTENT_INSPECTION_ENABLED",
        description=("Enable untrusted-content inspection for external ingress paths"),
    )
    untrusted_content_inspection_mode: Literal["warn", "sanitize", "block"] = Field(
        default="warn",
        validation_alias="UNTRUSTED_CONTENT_INSPECTION_MODE",
        description=(
            "Inspection action mode: warn for observation-first rollout, "
            "sanitize for safe rewrites, block for placeholder substitution"
        ),
    )
    untrusted_content_fail_policy: Literal["fail_open", "fail_closed"] = Field(
        default="fail_open",
        validation_alias="UNTRUSTED_CONTENT_FAIL_POLICY",
        description=(
            "Policy when the backend errors: fail_open preserves availability, "
            "fail_closed blocks on inspector failures"
        ),
    )
    untrusted_content_backend: Literal[
        "null", "http", "python", "subprocess", "composite"
    ] = Field(
        default="null",
        validation_alias="UNTRUSTED_CONTENT_BACKEND",
        description=(
            "Inspection backend. Current in-process options are null, python "
            "(heuristic), and composite (heuristic plus selective judge)."
        ),
    )

    # --- MCP Client Configuration ---
    mcp_enabled: bool = Field(
        default=False,
        validation_alias="MCP_ENABLED",
        description="Enable MCP client integration for cross-checks",
    )
    consultant_mcp_enabled: bool = Field(
        default=False,
        validation_alias="CONSULTANT_MCP_ENABLED",
        description="Enable MCP tools for the Consultant agent (requires mcp_enabled)",
    )
    mcp_servers_path: Path = Field(
        default=Path("./config/mcp_servers.json"),
        validation_alias="MCP_SERVERS_PATH",
        description="Path to the MCP server registry JSON file",
    )
    mcp_usage_db_path: Path = Field(
        default=Path("./runtime/mcp_usage.db"),
        validation_alias="MCP_USAGE_DB_PATH",
        description="Path to the SQLite database for MCP usage tracking",
    )

    # --- Optional IBKR market-data source (analysis pipeline) ---
    ibkr_data_source_enabled: bool = Field(
        default=False,
        validation_alias="IBKR_DATA_SOURCE_ENABLED",
        description=(
            "Opt in to using the IBKR market-data snapshot as an advisory source in the "
            "analysis data merge (requires IBKR creds). Supplies point-in-time ratios + "
            "price only; overrides Yahoo/FMP but defers to EODHD/filings. Default off so "
            "non-IBKR users and latency-sensitive batches are unaffected."
        ),
    )

    # --- Telemetry & System Overrides ---
    # These settings are exported to os.environ for third-party libraries
    # that read directly from environment variables (ChromaDB, gRPC).
    disable_chroma_telemetry: bool = Field(
        default=True,
        validation_alias="DISABLE_CHROMA_TELEMETRY",
        description="Disable ChromaDB anonymous telemetry",
    )
    grpc_enable_fork_support: bool = Field(
        default=True,
        validation_alias="GRPC_ENABLE_FORK_SUPPORT",
        description="Enable gRPC fork support (macOS compatibility)",
    )
    grpc_poll_strategy: str = Field(
        default="poll",
        validation_alias="GRPC_POLL_STRATEGY",
        description="gRPC poll strategy (poll is most compatible)",
    )
    macos_fork_safety_mitigation: bool = Field(
        default=True,
        validation_alias="MACOS_FORK_SAFETY_MITIGATION",
        description=(
            "On macOS, disable proxy auto-discovery + ObjC fork-init to prevent "
            "benign Network.framework atfork SIGSEGV crash dialogs"
        ),
    )

    # --- Prompts ---
    prompts_dir: Path = Field(
        default=Path("./prompts"),
        validation_alias="PROMPTS_DIR",
        description="Directory containing agent prompt JSON files",
    )

    # --- LangSmith ---
    langsmith_tracing_enabled: bool = Field(
        default=True,
        validation_alias="LANGSMITH_TRACING",
        description="Enable LangSmith tracing",
    )
    langsmith_project: str = Field(
        default="Deep-Trading-System-Gemini3",
        validation_alias="LANGSMITH_PROJECT",
        description="LangSmith project name",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias="LANGSMITH_ENDPOINT",
        description="LangSmith API endpoint",
    )

    # --- Langfuse (Alternative open-source observability) ---
    langfuse_enabled: bool = Field(
        default=False,
        validation_alias="LANGFUSE_ENABLED",
        description="Enable Langfuse tracing (alternative to LangSmith)",
    )
    langfuse_public_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGFUSE_PUBLIC_KEY",
        description="Langfuse public API key",
    )
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGFUSE_SECRET_KEY",
        description="Langfuse secret API key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_BASE_URL",
        description="Langfuse base URL (EU default, US: https://us.cloud.langfuse.com)",
    )
    langfuse_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias="LANGFUSE_SAMPLE_RATE",
        description="Trace sample rate (0.0-1.0) for cost control",
    )
    langfuse_debug: bool = Field(
        default=False,
        validation_alias="LANGFUSE_DEBUG",
        description="Enable Langfuse debug logging",
    )
    langfuse_environment: str = Field(
        default="development",
        validation_alias="LANGFUSE_TRACING_ENVIRONMENT",
        description="Environment tag for filtering in Langfuse dashboard",
    )
    langfuse_prompt_fetch_enabled: bool = Field(
        default=False,
        validation_alias="LANGFUSE_PROMPT_FETCH_ENABLED",
        description="Enable fetching prompts from Langfuse at runtime",
    )
    langfuse_prompt_label: str = Field(
        default="production",
        validation_alias="LANGFUSE_PROMPT_LABEL",
        description="Langfuse prompt label to resolve when prompt fetch is enabled",
    )
    langfuse_prompt_cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        validation_alias="LANGFUSE_PROMPT_CACHE_TTL_SECONDS",
        description="Cache TTL in seconds for Langfuse prompt fetches",
    )

    # --- API Keys (SecretStr prevents accidental logging) ---
    # These are optional at Settings instantiation but required for actual use.
    # The validate_environment_variables() function checks for required keys.
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GOOGLE_API_KEY",
        description="Google Gemini API key (required)",
    )
    tavily_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="TAVILY_API_KEY",
        description="Tavily search API key (required)",
    )
    finnhub_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="FINNHUB_API_KEY",
        description="Finnhub market data API key (required)",
    )
    eodhd_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="EODHD_API_KEY",
        description="EODHD international data API key (optional)",
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for consultant agent (optional)",
    )
    langsmith_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGSMITH_API_KEY",
        description="LangSmith tracing API key (optional)",
    )
    fmp_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="FMP_API_KEY",
        description="Financial Modeling Prep API key (optional fallback)",
    )
    alpha_vantage_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="ALPHAVANTAGE_API_KEY",
        description="Alpha Vantage API key (optional fallback)",
    )
    edinet_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="EDINET_API_KEY",
        description="Japan EDINET filing API key (optional, free registration)",
    )

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        # Load from .env file
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra environment variables (don't fail on unknown vars)
        extra="ignore",
        # Case-insensitive env var matching
        case_sensitive=False,
        # Some runtime paths still set non-override operational fields in process.
        frozen=False,
        # Use validation_alias for env var names
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def setup_environment(self) -> "Settings":
        """Post-initialization setup: normalize paths, configure logging, and export SDK settings.

        Note: Third-party SDKs (LangSmith, etc.) expect configuration in os.environ.
        Since we removed load_dotenv(), we must export necessary settings here.
        This is intentional and targeted - we only export what SDKs need.
        """
        # Expand user directories (handling ~)
        self.results_dir = Path(os.path.expanduser(str(self.results_dir)))
        self.data_cache_dir = Path(os.path.expanduser(str(self.data_cache_dir)))
        self.chroma_persist_directory = os.path.expanduser(
            self.chroma_persist_directory
        )
        self.images_dir = Path(os.path.expanduser(str(self.images_dir)))
        self.prompts_dir = Path(os.path.expanduser(str(self.prompts_dir)))
        self.mcp_servers_path = Path(os.path.expanduser(str(self.mcp_servers_path)))
        self.mcp_usage_db_path = Path(os.path.expanduser(str(self.mcp_usage_db_path)))

        # Set logging level on the ROOT logger only. Never force-level every
        # registered logger (the old loggerDict loop): that flattened the
        # deliberate WARNING suppression of noisy third-party loggers (httpx,
        # ibind, google_genai, ddgs, ...) applied by CLI entrypoints — Settings
        # construction can happen after those entrypoints configure logging.
        # Loggers left at NOTSET inherit the root level anyway.
        log_level_value = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level_value)

        # Export LangSmith settings to os.environ for SDK auto-detection.
        # The LangSmith SDK reads directly from os.environ, not from our config.
        # Only export if values are set (don't overwrite existing shell env vars).
        langsmith_api_key = self.langsmith_api_key.get_secret_value()
        if langsmith_api_key and "LANGSMITH_API_KEY" not in os.environ:
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        if self.langsmith_project and "LANGSMITH_PROJECT" not in os.environ:
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        if self.langsmith_endpoint and "LANGSMITH_ENDPOINT" not in os.environ:
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
        # LangSmith tracing is enabled if LANGSMITH_TRACING=true
        if self.langsmith_tracing_enabled and "LANGSMITH_TRACING" not in os.environ:
            os.environ["LANGSMITH_TRACING"] = "true"

        # Export Langfuse settings to os.environ for SDK compatibility.
        # Observability now initializes the client explicitly, but keeping these
        # env vars populated helps third-party callbacks and local debugging.
        langfuse_public = self.langfuse_public_key.get_secret_value()
        langfuse_secret = self.langfuse_secret_key.get_secret_value()
        if langfuse_public and "LANGFUSE_PUBLIC_KEY" not in os.environ:
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public
        if langfuse_secret and "LANGFUSE_SECRET_KEY" not in os.environ:
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret
        if self.langfuse_host and "LANGFUSE_BASE_URL" not in os.environ:
            os.environ["LANGFUSE_BASE_URL"] = self.langfuse_host
        if "LANGFUSE_SAMPLE_RATE" not in os.environ:
            os.environ["LANGFUSE_SAMPLE_RATE"] = str(self.langfuse_sample_rate)
        if "LANGFUSE_DEBUG" not in os.environ:
            os.environ["LANGFUSE_DEBUG"] = "true" if self.langfuse_debug else "false"
        if (
            self.langfuse_environment
            and "LANGFUSE_TRACING_ENVIRONMENT" not in os.environ
        ):
            os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = self.langfuse_environment

        # Export EDINET API key for edinet-tools SDK (reads from os.environ).
        edinet_key = self.edinet_api_key.get_secret_value()
        if edinet_key and "EDINET_API_KEY" not in os.environ:
            os.environ["EDINET_API_KEY"] = edinet_key

        # Export telemetry/system settings for third-party libraries.
        # ChromaDB and gRPC read directly from os.environ.
        if self.disable_chroma_telemetry:
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
        if self.grpc_enable_fork_support:
            os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
        if self.grpc_poll_strategy:
            os.environ["GRPC_POLL_STRATEGY"] = self.grpc_poll_strategy

        # macOS fork-safety: prevent benign Network.framework atfork SIGSEGV
        # crash dialogs in forked children (see _apply_macos_fork_safety_env).
        _apply_macos_fork_safety_env(
            enabled=self.macos_fork_safety_mitigation, platform=sys.platform
        )

        return self

    def runtime_directories(self) -> list[Path]:
        """Directories the CLI runtime should create explicitly at startup."""
        return [
            Path(self.results_dir),
            Path(self.data_cache_dir),
            Path(self.chroma_persist_directory),
            Path(self.images_dir),
            Path(self.mcp_usage_db_path).parent,
        ]

    def get_google_api_key(self) -> str:
        """
        Get Google API key securely from SecretStr field.

        Returns:
            Google API key string, or empty string if not set

        Note:
            Tests should mock this method or reload the config module
            after patching environment variables.
        """
        return self.google_api_key.get_secret_value()

    def get_tavily_api_key(self) -> str:
        """Get Tavily API key securely from SecretStr field."""
        return self.tavily_api_key.get_secret_value()

    def get_finnhub_api_key(self) -> str:
        """Get Finnhub API key securely from SecretStr field."""
        return self.finnhub_api_key.get_secret_value()

    def get_eodhd_api_key(self) -> str:
        """Get EODHD API key securely from SecretStr field."""
        return self.eodhd_api_key.get_secret_value()

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key securely from SecretStr field."""
        return self.openai_api_key.get_secret_value()

    def get_apac_specialist_api_key(self) -> str:
        """Get APAC Regional Specialist API key securely from SecretStr field."""
        return self.apac_specialist_api_key.get_secret_value()

    def get_langsmith_api_key(self) -> str:
        """Get LangSmith API key securely from SecretStr field."""
        return self.langsmith_api_key.get_secret_value()

    def get_fmp_api_key(self) -> str:
        """Get Financial Modeling Prep API key securely from SecretStr field."""
        return self.fmp_api_key.get_secret_value()

    def get_alpha_vantage_api_key(self) -> str:
        """Get Alpha Vantage API key securely from SecretStr field."""
        return self.alpha_vantage_api_key.get_secret_value()

    def get_claude_api_key(self) -> str | None:
        """Get Anthropic API key, or None if not configured."""
        return self.claude_api_key.get_secret_value() if self.claude_api_key else None

    def get_edinet_api_key(self) -> str:
        """Get EDINET API key securely from SecretStr field."""
        return self.edinet_api_key.get_secret_value()

    def get_langfuse_public_key(self) -> str:
        """Get Langfuse public key securely from SecretStr field."""
        return self.langfuse_public_key.get_secret_value()

    def get_langfuse_secret_key(self) -> str:
        """Get Langfuse secret key securely from SecretStr field."""
        return self.langfuse_secret_key.get_secret_value()


# --- Backwards Compatibility Alias ---
Config = Settings


# --- Module-level Singleton Instance ---
# Instantiated at import time, triggers validation
config = Settings()
