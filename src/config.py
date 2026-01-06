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

import logging
import os
import sys
from pathlib import Path

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

    if has_api_key:
        # LangSmith SDK auto-detects from Pydantic Settings - just log for visibility
        logger.info(f"LangSmith tracing enabled for project: {project_name}")


def _parse_env_file() -> dict:
    """Parse .env file to get explicitly set values (ignoring comments and blank lines).

    Preserved for backwards compatibility with existing tests.
    """
    env_file = Path(".env")
    env_values = {}

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
        default="gemini-3-pro-preview",
        validation_alias="DEEP_MODEL",
        description="Model for deep thinking/synthesis agents",
    )
    # Flash models work with langchain-google-genai 4.0.0+
    quick_think_llm: str = Field(
        default="gemini-2.0-flash",
        validation_alias="QUICK_MODEL",
        description="Model for quick thinking/data gathering agents",
    )

    # --- Debate & Risk Configuration ---
    max_debate_rounds: int = Field(
        default=2,
        ge=0,
        validation_alias="MAX_DEBATE_ROUNDS",
        description="Maximum rounds of bull/bear debate",
    )
    max_risk_discuss_rounds: int = Field(
        default=1,
        ge=0,
        validation_alias="MAX_RISK_DISCUSS_ROUNDS",
        description="Maximum rounds of risk discussion",
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

    # --- Consultant Configuration ---
    consultant_model: str = Field(
        default="gpt-4o",
        validation_alias="CONSULTANT_MODEL",
        description="OpenAI model for consultant in normal mode",
    )
    consultant_quick_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="CONSULTANT_QUICK_MODEL",
        description="OpenAI model for consultant in quick mode",
    )
    auditor_model: str | None = Field(
        default=None,
        validation_alias="AUDITOR_MODEL",
        description="Model for the auditor agent (optional)",
    )

    # --- Trading Parameters ---
    max_position_size: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        validation_alias="MAX_POSITION_SIZE",
        description="Maximum position size as fraction of portfolio",
    )
    max_daily_trades: int = Field(
        default=5,
        ge=0,
        validation_alias="MAX_DAILY_TRADES",
        description="Maximum number of trades per day",
    )
    risk_free_rate: float = Field(
        default=0.03,
        ge=0.0,
        validation_alias="RISK_FREE_RATE",
        description="Risk-free rate for calculations",
    )
    default_ticker: str = Field(
        default="AAPL",
        validation_alias="DEFAULT_TICKER",
        description="Default ticker symbol for analysis",
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
    # Retries from 3 -> 10 to aggressively handle 504/503 transient errors
    api_retry_attempts: int = Field(
        default=10,
        ge=0,
        validation_alias="API_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed API calls",
    )

    # --- Rate Limiting ---
    # Free tier: 15 RPM | Paid tier 1: 360 RPM | Tier 2: 1000+ RPM
    gemini_rpm_limit: int = Field(
        default=15,
        ge=1,
        validation_alias="GEMINI_RPM_LIMIT",
        description="Gemini API rate limit (requests per minute)",
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

    # --- Runtime Flags ---
    quiet_mode: bool = Field(
        default=False,
        validation_alias="QUIET_MODE",
        description="Suppress verbose logging output (set via CLI --quiet)",
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

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        # Load from .env file
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra environment variables (don't fail on unknown vars)
        extra="ignore",
        # Case-insensitive env var matching
        case_sensitive=False,
        # Allow mutation for CLI arg overrides (main.py sets quick_think_llm, etc.)
        frozen=False,
        # Use validation_alias for env var names
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def setup_environment(self) -> "Settings":
        """Post-initialization setup: create directories, configure logging, and export SDK settings.

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

        # Create required directories
        for directory in [
            self.results_dir,
            self.data_cache_dir,
            Path(self.chroma_persist_directory),
            self.images_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Set logging level
        log_level_value = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level_value)
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(log_level_value)

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

        # Export telemetry/system settings for third-party libraries.
        # ChromaDB and gRPC read directly from os.environ.
        if self.disable_chroma_telemetry:
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
        if self.grpc_enable_fork_support:
            os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
        if self.grpc_poll_strategy:
            os.environ["GRPC_POLL_STRATEGY"] = self.grpc_poll_strategy

        return self

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

    def get_langsmith_api_key(self) -> str:
        """Get LangSmith API key securely from SecretStr field."""
        return self.langsmith_api_key.get_secret_value()

    def get_fmp_api_key(self) -> str:
        """Get Financial Modeling Prep API key securely from SecretStr field."""
        return self.fmp_api_key.get_secret_value()

    def get_alpha_vantage_api_key(self) -> str:
        """Get Alpha Vantage API key securely from SecretStr field."""
        return self.alpha_vantage_api_key.get_secret_value()


# --- Backwards Compatibility Alias ---
Config = Settings


# --- Module-level Singleton Instance ---
# Instantiated at import time, triggers validation
config = Settings()
