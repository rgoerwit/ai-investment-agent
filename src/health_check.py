#!/usr/bin/env python3
"""
Health check script for container health monitoring.
Tests core system components without running full analysis.
Updated for Gemini 3 Migration (Nov 2025).

Run with:  poetry run python src/health_check.py
"""

import asyncio
import logging
import sys
from pathlib import Path

import structlog

# Add the repository root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = structlog.get_logger(__name__)

# Suppress noisy library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.ai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)


def get_package_version(module_name: str, package_name: str = None) -> str:
    """Get version of a package."""
    if package_name is None:
        package_name = module_name.replace("_", "-")

    try:
        mod = __import__(module_name)
        if hasattr(mod, "__version__"):
            return mod.__version__
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "unknown"


def check_python_version() -> tuple[bool, list[str]]:
    """Check if Python version meets requirements."""
    issues = []
    major, minor = sys.version_info[:2]

    if (major, minor) < (3, 10):
        issues.append(f"Python {major}.{minor} detected. Requires Python 3.10+")
        logger.error("python_version_fail", version=f"{major}.{minor}")
        return False, issues

    logger.info("python_version_ok", version=f"{major}.{minor}")
    return True, []


def check_environment_variables() -> bool:
    """Check if required environment variables are set for Gemini.

    Uses config getters to check for API keys, which properly handle
    SecretStr protection and prevent accidental key exposure in logs.

    Note: Pydantic Settings handles .env loading automatically via env_file
    in SettingsConfigDict, so no manual load_dotenv() is needed.
    """
    from src.config import config

    # Check required API keys using secure getters
    # (getters return empty string if not set, never expose the actual key)
    required_checks = [
        ("GOOGLE_API_KEY", config.get_google_api_key),
        ("FINNHUB_API_KEY", config.get_finnhub_api_key),
        ("TAVILY_API_KEY", config.get_tavily_api_key),
    ]
    missing_vars = []

    for var_name, getter in required_checks:
        if getter():
            logger.info("env_var_present", var=var_name)
        else:
            missing_vars.append(var_name)

    if missing_vars:
        logger.error("env_vars_missing", vars=missing_vars)
        logger.info("env_setup_hint")
        return False

    logger.info("env_vars_ok")
    return True


def check_imports() -> bool:
    """Check if core modules can be imported."""
    logger.info("checking_imports")
    critical_failures = []

    # Core Logic Imports
    modules_to_check = [
        ("structlog", "structlog"),
        ("langchain_core", "langchain-core"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        # UPDATED: Check for Google GenAI instead of OpenAI
        ("langchain_google_genai", "langchain-google-genai"),
        ("google.genai", "google-genai"),
        ("yfinance", "yfinance"),
        ("finnhub", "finnhub-python"),
    ]

    for mod_name, pkg_name in modules_to_check:
        try:
            __import__(mod_name)
            version = get_package_version(mod_name, pkg_name)
            logger.info("import_ok", package=pkg_name, version=version)
        except ImportError as e:
            logger.error("import_failed", package=pkg_name, error=str(e))
            critical_failures.append(pkg_name)

    # Check for ChromaDB (Optional but recommended)
    try:
        import chromadb

        logger.info("import_ok", package="chromadb")
    except ImportError:
        logger.warning("chromadb_missing")

    if critical_failures:
        logger.error("critical_import_failures", failures=critical_failures)
        return False

    return True


async def check_llm_connectivity() -> bool:
    """Test basic LLM connectivity with Gemini."""
    try:
        # UPDATED: Use ChatGoogleGenerativeAI
        from langchain_google_genai import ChatGoogleGenerativeAI

        from src.config import config

        logger.info("testing_llm_connectivity", model=config.quick_think_llm)

        llm = ChatGoogleGenerativeAI(
            model=config.quick_think_llm, temperature=0, timeout=10, max_retries=1
        )

        response = await asyncio.wait_for(
            llm.ainvoke("Respond with just the word 'OK'."), timeout=15.0
        )

        # Handle potential dict/list response from Gemini
        raw_content = response.content
        if isinstance(raw_content, dict):
            content = str(raw_content.get("text", raw_content))
        elif isinstance(raw_content, list):
            content = str(raw_content[0]) if raw_content else ""
        else:
            content = str(raw_content) if raw_content else ""
        content = content.strip()

        if "OK" in content or "ok" in content.lower():
            logger.info("llm_connectivity_ok")
            return True
        else:
            logger.warning("llm_unexpected_response", content=content)
            return False

    except asyncio.TimeoutError:
        logger.error("llm_connectivity_timeout")
        return False
    except ImportError as e:
        logger.error("llm_connectivity_import_error", error=str(e))
        return False
    except Exception as e:
        logger.error("llm_connectivity_error", error=str(e))
        return False


async def run_comprehensive_health_check() -> bool:
    """Run all health checks."""
    logger.info("health_check_started")

    python_ok, _ = check_python_version()
    env_ok = check_environment_variables()

    if not check_imports():
        return False

    # Check internal project imports to ensure no lingering OpenAI references break imports
    try:
        from src.llms import quick_thinking_llm

        logger.info("llms_import_ok")
    except ImportError as e:
        logger.error("llms_import_failed", error=str(e))
        return False

    llm_ok = await check_llm_connectivity()

    all_passed = all([python_ok, env_ok, llm_ok])

    if all_passed:
        logger.info("health_check_passed")
    else:
        logger.error("health_check_failed")

    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(run_comprehensive_health_check())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(130)
