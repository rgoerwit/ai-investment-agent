"""Filing fetcher registry — maps ticker suffixes to filing API fetchers."""

import structlog

from src.data.filings.base import FilingFetcher, FilingResult

logger = structlog.get_logger(__name__)


class FilingRegistry:
    """Maps ticker suffixes to filing fetchers. Module-level singleton."""

    def __init__(self) -> None:
        self._fetchers: dict[str, FilingFetcher] = {}

    def register(self, fetcher: FilingFetcher) -> None:
        for suffix in fetcher.supported_suffixes:
            self._fetchers[suffix] = fetcher
            logger.info(
                "filing_fetcher_registered",
                suffix=suffix,
                source=fetcher.source_name,
            )

    def get_fetcher(self, ticker: str) -> FilingFetcher | None:
        suffix = ticker.split(".")[-1] if "." in ticker else None
        if suffix and suffix in self._fetchers:
            fetcher = self._fetchers[suffix]
            return fetcher if fetcher.is_available() else None
        return None

    async def fetch(self, ticker: str) -> FilingResult | None:
        fetcher = self.get_fetcher(ticker)
        if not fetcher:
            return None
        try:
            return await fetcher.get_filing_data(ticker)
        except Exception as e:
            logger.warning(
                "filing_fetch_error",
                ticker=ticker,
                source=fetcher.source_name,
                error=str(e),
            )
            return None

    @property
    def available_suffixes(self) -> list[str]:
        """Return list of suffixes with available fetchers."""
        return [s for s, f in self._fetchers.items() if f.is_available()]


# Module-level singleton
registry = FilingRegistry()
