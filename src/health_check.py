#!/usr/bin/env python3
"""
Health check script for container health monitoring.
Tests core system components without running full analysis.
Updated for Gemini 3 Migration (Nov 2025).

Run with:  poetry run python src/health_check.py
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple

# Add the repository root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

# Suppress noisy library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.ai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)


def get_package_version(module_name: str, package_name: str = None) -> str:
    """Get version of a package."""
    if package_name is None:
        package_name = module_name.replace('_', '-')
    
    try:
        mod = __import__(module_name)
        if hasattr(mod, '__version__'):
            return mod.__version__
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        return "unknown"


def check_python_version() -> Tuple[bool, List[str]]:
    """Check if Python version meets requirements."""
    issues = []
    major, minor = sys.version_info[:2]
    
    if (major, minor) < (3, 10):
        issues.append(f"Python {major}.{minor} detected. Requires Python 3.10+")
        logger.error(f"Python version: {major}.{minor} (✗)")
        return False, issues
    
    logger.info(f"Python version: {major}.{minor} (✓)")
    return True, []


def check_environment_variables() -> bool:
    """Check if required environment variables are set for Gemini."""
    env_file = repo_root / ".env"
    if env_file.exists():
        logger.info(f"Loading environment from {env_file}")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            logger.warning("python-dotenv not available, skipping .env file loading")
    
    # UPDATED: Check for Google API Key instead of OpenAI
    required_vars = [
        "GOOGLE_API_KEY", 
        "FINNHUB_API_KEY", 
        "TAVILY_API_KEY", 
    ]
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var, "")
        if not value or value.strip() == "":
            missing_vars.append(var)
        else:
            logger.info(f"{var}: Present (✓)")
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.info("Please copy .env.example to .env and add your API keys")
        return False
    
    logger.info("All required environment variables are set")
    return True


def check_imports() -> bool:
    """Check if core modules can be imported."""
    logger.info("Checking core module imports...")
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
        ("finnhub", "finnhub-python")
    ]

    for mod_name, pkg_name in modules_to_check:
        try:
            importlib = __import__(mod_name)
            version = get_package_version(mod_name, pkg_name)
            logger.info(f"Import successful: {pkg_name} {version} (✓)")
        except ImportError as e:
            logger.error(f"Import failed: {pkg_name} - {e}")
            critical_failures.append(pkg_name)

    # Check for ChromaDB (Optional but recommended)
    try:
        import chromadb
        logger.info("Import successful: chromadb (✓)")
    except ImportError:
        logger.warning("Import failed: chromadb (Memory will be disabled)")

    if critical_failures:
        logger.error(f"Critical import failures: {critical_failures}")
        return False
    
    return True


async def check_llm_connectivity() -> bool:
    """Test basic LLM connectivity with Gemini."""
    try:
        from src.config import config
        # UPDATED: Use ChatGoogleGenerativeAI
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        logger.info(f"Testing Gemini connectivity with model: {config.quick_think_llm}")
        
        llm = ChatGoogleGenerativeAI(
            model=config.quick_think_llm,
            temperature=0,
            timeout=10,
            max_retries=1
        )
        
        response = await asyncio.wait_for(
            llm.ainvoke("Respond with just the word 'OK'."),
            timeout=15.0
        )

        # Handle potential dict/list response from Gemini
        raw_content = response.content
        if isinstance(raw_content, dict):
            content = str(raw_content.get('text', raw_content))
        elif isinstance(raw_content, list):
            content = str(raw_content[0]) if raw_content else ""
        else:
            content = str(raw_content) if raw_content else ""
        content = content.strip()

        if "OK" in content or "ok" in content.lower():
            logger.info("Gemini LLM connectivity: OK (✓)")
            return True
        else:
            logger.warning(f"LLM responded but unexpected content: {content}")
            return False
            
    except asyncio.TimeoutError:
        logger.error("LLM connectivity: TIMEOUT (API too slow)")
        return False
    except ImportError as e:
        logger.error(f"LLM connectivity: Import error - {e}")
        return False
    except Exception as e:
        logger.error(f"LLM connectivity error: {e}")
        return False


async def run_comprehensive_health_check() -> bool:
    """Run all health checks."""
    logger.info("Starting Gemini System Health Check...")
    
    python_ok, _ = check_python_version()
    env_ok = check_environment_variables()
    
    if not check_imports():
        return False
        
    # Check internal project imports to ensure no lingering OpenAI references break imports
    try:
        from src.llms import quick_thinking_llm
        logger.info("Project module 'src.llms' imported successfully (✓)")
    except ImportError as e:
        logger.error(f"Failed to import src.llms: {e}")
        return False

    llm_ok = await check_llm_connectivity()
    
    all_passed = all([python_ok, env_ok, llm_ok])
    
    if all_passed:
        logger.info("✅ OVERALL HEALTH CHECK: PASSED")
    else:
        logger.error("❌ OVERALL HEALTH CHECK: FAILED")
        
    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(run_comprehensive_health_check())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(130)