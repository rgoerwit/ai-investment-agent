from typing import Any

import aiohttp
import structlog

from src.config import config
from src.error_safety import summarize_exception
from src.llm_runtime.timeouts import require_allowance

logger = structlog.get_logger(__name__)


class StockTwitsAPI:
    """
    A lightweight wrapper for the StockTwits API to fetch real-time social sentiment.
    No API key is required for public stream access, but rate limits apply.
    """

    BASE_URL = "https://api.stocktwits.com/api/2"

    async def get_sentiment(self, ticker: str) -> dict[str, Any]:
        """
        Fetch the last 30 messages for a ticker and calculate sentiment.

        Args:
            ticker: The stock symbol (e.g., 'AAPL', 'TSLA')

        Returns:
            Dictionary containing volume, sentiment counts, and message samples.
            Returns {'error': ...} if the request fails or ticker not found.
        """
        # StockTwits usually expects clean tickers (e.g., "AAPL" not "AAPL.US")
        # We attempt to strip common suffixes for better hit rates, though
        # strict international support is limited on StockTwits.
        clean_ticker = ticker.split(".")[0] if "." in ticker else ticker
        url = f"{self.BASE_URL}/streams/symbol/{clean_ticker}.json"

        headers = {"User-Agent": config.http_user_agent}

        # STOCKTWITS_HTTP_TIMEOUT_SECONDS, read at the call and shrunk to a
        # caller's enclosing deadline before the session is opened; a spent
        # deadline returns the same error payload as a request timeout.
        try:
            timeout_seconds = require_allowance(
                config.stocktwits_http_timeout_seconds, label="stocktwits"
            )
        except TimeoutError as e:
            return self._fetch_failed(ticker, e)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 404:
                        return {"error": "Symbol not found on StockTwits"}
                    if response.status == 429:
                        return {"error": "Rate limit exceeded"}
                    if response.status != 200:
                        return {"error": f"HTTP {response.status}"}

                    data = await response.json()
                    messages = data.get("messages", [])

                    return self._process_messages(messages, clean_ticker)

            except Exception as e:
                return self._fetch_failed(ticker, e)

    @staticmethod
    def _fetch_failed(ticker: str, error: Exception) -> dict[str, Any]:
        """Log a failed fetch and return the caller-facing error payload."""
        summary = summarize_exception(error, operation="stocktwits_fetch")
        logger.error("stocktwits_fetch_failed", ticker=ticker, **summary)
        return {"error": f"{summary['error_type']} (details in operator logs)"}

    def _process_messages(self, messages: list[dict], ticker: str) -> dict[str, Any]:
        """Analyze messages for Bullish/Bearish tags."""
        total = len(messages)
        bullish = 0
        bearish = 0

        sample_texts: list[str] = []

        for msg in messages:
            # Extract sentiment if tagged by the user
            entities = msg.get("entities", {})
            sentiment = entities.get("sentiment", {})

            if sentiment:
                basic = sentiment.get("basic")
                if basic == "Bullish":
                    bullish += 1
                elif basic == "Bearish":
                    bearish += 1

            # Keep a few samples for the LLM to read context
            if len(sample_texts) < 3:
                body = msg.get("body", "")
                user = msg.get("user", {}).get("username", "anon")
                sentiment_tag = (
                    f"[{sentiment.get('basic', 'Neutral')}]"
                    if sentiment
                    else "[No Tag]"
                )
                sample_texts.append(f"{sentiment_tag} {user}: {body}")

        # Calculate percentages
        sentiment_total = bullish + bearish
        bull_pct = (bullish / sentiment_total * 100) if sentiment_total > 0 else 0
        bear_pct = (bearish / sentiment_total * 100) if sentiment_total > 0 else 0

        return {
            "source": "StockTwits",
            "ticker": ticker,
            "total_messages_last_30": total,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "bullish_pct": round(bull_pct, 1),
            "bearish_pct": round(bear_pct, 1),
            "messages": sample_texts,
        }
