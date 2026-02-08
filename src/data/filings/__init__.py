"""Official filing API module — structured data from regulatory filing systems.

Public API:
    registry: FilingRegistry singleton for fetching filing data
    FilingResult: Standardized output dataclass
    FilingFetcher: ABC for implementing country-specific fetchers

Usage:
    from src.data.filings import registry
    result = await registry.fetch("2767.T")  # Returns FilingResult or None
"""

import structlog

from src.data.filings.base import FilingFetcher, FilingResult
from src.data.filings.registry import registry

logger = structlog.get_logger(__name__)

# Auto-register available fetchers on import
_registered = False


def _auto_register() -> None:
    """Register all available filing fetchers. Called once on first import."""
    global _registered
    if _registered:
        return
    _registered = True

    # EDINET (Japan .T)
    try:
        from src.data.filings.edinet_fetcher import EdinetFetcher

        fetcher = EdinetFetcher()
        registry.register(fetcher)
    except ImportError:
        logger.debug(
            "edinet_fetcher_import_skipped", reason="edinet-tools not installed"
        )
    except Exception as e:
        logger.debug("edinet_fetcher_registration_error", error=str(e))

    # Future: DART (Korea), Companies House (UK), etc.


_auto_register()

__all__ = ["registry", "FilingResult", "FilingFetcher"]
