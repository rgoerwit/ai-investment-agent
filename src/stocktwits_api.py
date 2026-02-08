from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Default timeout for HTTP requests
_TIMEOUT = aiohttp.ClientTimeout(total=10)


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

        headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}

        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            try:
                async with session.get(
                    url, headers=headers, timeout=_TIMEOUT
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
                logger.error("stocktwits_fetch_failed", ticker=ticker, error=str(e))
                return {"error": str(e)}

    def _process_messages(self, messages: list[dict], ticker: str) -> dict[str, Any]:
        """Analyze messages for Bullish/Bearish tags."""
        total = len(messages)
        bullish = 0
        bearish = 0

        sample_texts = []

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
