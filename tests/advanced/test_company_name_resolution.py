"""
Tests for multi-source company name resolution.

Validates the 4-source fallback chain (yfinance → yahooquery → FMP → EODHD),
ticker echo rejection, normalization, fetcher methods, and agent prompt warning.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ticker_utils import (
    CompanyNameResult,
    _is_valid_company_name,
    resolve_company_name,
)


class TestIsValidCompanyName:
    """Test the name validation helper."""

    def test_valid_name(self):
        assert _is_valid_company_name("Toyota Motor Corporation", "7203.T") is True

    def test_none_rejected(self):
        assert _is_valid_company_name(None, "7203.T") is False

    def test_empty_rejected(self):
        assert _is_valid_company_name("", "7203.T") is False

    def test_whitespace_rejected(self):
        assert _is_valid_company_name("   ", "7203.T") is False

    def test_ticker_echo_rejected(self):
        """Source returns the ticker string itself as 'name' — treated as failure."""
        assert _is_valid_company_name("2154.HK", "2154.HK") is False

    def test_ticker_echo_case_insensitive(self):
        assert _is_valid_company_name("2154.hk", "2154.HK") is False

    def test_ticker_base_rejected(self):
        """Reject if name is just the ticker base without exchange suffix."""
        assert _is_valid_company_name("2154", "2154.HK") is False


class TestResolveCompanyName:
    """Test the multi-source resolution chain."""

    @pytest.mark.asyncio
    async def test_yfinance_resolves_name(self):
        """yfinance returns longName, other sources not called."""
        mock_info = {"longName": "Linklogis Inc.", "shortName": "Linklogis"}

        with patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf:
            mock_yf.return_value = "Linklogis Inc."
            result = await resolve_company_name("2154.HK")

        assert result.is_resolved is True
        assert result.source == "yfinance"
        assert "Linklogis" in result.name

    @pytest.mark.asyncio
    async def test_yfinance_fails_yahooquery_resolves(self):
        """yfinance returns None, yahooquery succeeds."""
        with (
            patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf,
            patch(
                "src.ticker_utils._try_yahooquery", new_callable=AsyncMock
            ) as mock_yq,
        ):
            mock_yf.return_value = None
            mock_yq.return_value = "Linklogis Inc."
            result = await resolve_company_name("2154.HK")

        assert result.is_resolved is True
        assert result.source == "yahooquery"
        assert "Linklogis" in result.name

    @pytest.mark.asyncio
    async def test_all_free_fail_fmp_resolves(self):
        """yfinance + yahooquery fail, FMP succeeds."""
        with (
            patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf,
            patch(
                "src.ticker_utils._try_yahooquery", new_callable=AsyncMock
            ) as mock_yq,
            patch("src.ticker_utils._try_fmp", new_callable=AsyncMock) as mock_fmp,
        ):
            mock_yf.return_value = None
            mock_yq.return_value = None
            mock_fmp.return_value = "Linklogis Inc."
            result = await resolve_company_name("2154.HK")

        assert result.is_resolved is True
        assert result.source == "fmp"

    @pytest.mark.asyncio
    async def test_all_fail_eodhd_resolves(self):
        """All free + FMP fail, EODHD succeeds."""
        with (
            patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf,
            patch(
                "src.ticker_utils._try_yahooquery", new_callable=AsyncMock
            ) as mock_yq,
            patch("src.ticker_utils._try_fmp", new_callable=AsyncMock) as mock_fmp,
            patch("src.ticker_utils._try_eodhd", new_callable=AsyncMock) as mock_eodhd,
        ):
            mock_yf.return_value = None
            mock_yq.return_value = None
            mock_fmp.return_value = None
            mock_eodhd.return_value = "Linklogis Inc."
            result = await resolve_company_name("2154.HK")

        assert result.is_resolved is True
        assert result.source == "eodhd"

    @pytest.mark.asyncio
    async def test_all_fail_returns_unresolved(self):
        """All 4 sources fail, returns unresolved result with ticker as name."""
        with (
            patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf,
            patch(
                "src.ticker_utils._try_yahooquery", new_callable=AsyncMock
            ) as mock_yq,
            patch("src.ticker_utils._try_fmp", new_callable=AsyncMock) as mock_fmp,
            patch("src.ticker_utils._try_eodhd", new_callable=AsyncMock) as mock_eodhd,
        ):
            mock_yf.return_value = None
            mock_yq.return_value = None
            mock_fmp.return_value = None
            mock_eodhd.return_value = None
            result = await resolve_company_name("2154.HK")

        assert result.is_resolved is False
        assert result.source == "unresolved"
        assert result.name == "2154.HK"

    @pytest.mark.asyncio
    async def test_ticker_echo_rejected_tries_next(self):
        """Source returns the ticker string as 'name' — skipped, tries next."""
        with (
            patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf,
            patch(
                "src.ticker_utils._try_yahooquery", new_callable=AsyncMock
            ) as mock_yq,
        ):
            # yfinance echoes ticker back
            mock_yf.return_value = "2154.HK"
            # yahooquery returns real name
            mock_yq.return_value = "Linklogis Inc."
            result = await resolve_company_name("2154.HK")

        assert result.is_resolved is True
        assert result.source == "yahooquery"

    @pytest.mark.asyncio
    async def test_name_normalized(self):
        """Legal suffixes stripped from resolved name."""
        with patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf:
            mock_yf.return_value = "Toyota Motor Corporation"
            result = await resolve_company_name("7203.T")

        assert result.is_resolved is True
        # "Corporation" suffix should be stripped by normalize_company_name
        assert result.name == "Toyota Motor"

    @pytest.mark.asyncio
    async def test_exception_in_source_continues_chain(self):
        """If a source raises an exception, chain continues to next source."""
        with (
            patch("src.ticker_utils._try_yfinance", new_callable=AsyncMock) as mock_yf,
            patch(
                "src.ticker_utils._try_yahooquery", new_callable=AsyncMock
            ) as mock_yq,
        ):
            mock_yf.side_effect = RuntimeError("network error")
            mock_yq.return_value = "HSBC Holdings"
            result = await resolve_company_name("0005.HK")

        assert result.is_resolved is True
        assert result.source == "yahooquery"


class TestFMPGetCompanyName:
    """Test FMP fetcher company name method."""

    @pytest.mark.asyncio
    async def test_fmp_get_company_name_success(self):
        """FMP returns company name from profile endpoint."""
        from src.data.fmp_fetcher import FMPFetcher

        fetcher = FMPFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[{"companyName": "Linklogis Inc.", "symbol": "2154.HK"}]
        )

        mock_response_cm = AsyncMock()
        mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_cm)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await fetcher.get_company_name("2154.HK")

        assert result == "Linklogis Inc."

    @pytest.mark.asyncio
    async def test_fmp_get_company_name_empty_response(self):
        """FMP returns empty list — returns None."""
        from src.data.fmp_fetcher import FMPFetcher

        fetcher = FMPFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])

        mock_response_cm = AsyncMock()
        mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_cm)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await fetcher.get_company_name("XXXX.HK")

        assert result is None

    @pytest.mark.asyncio
    async def test_fmp_get_company_name_no_api_key(self):
        """FMP not available (no API key) — returns None."""
        from src.data.fmp_fetcher import FMPFetcher

        fetcher = FMPFetcher(api_key=None)
        fetcher.api_key = None

        result = await fetcher.get_company_name("2154.HK")
        assert result is None


class TestEODHDGetCompanyName:
    """Test EODHD fetcher company name method."""

    @pytest.mark.asyncio
    async def test_eodhd_get_company_name_success(self):
        """EODHD returns company name from General data."""
        from src.data.eodhd_fetcher import EODHDFetcher

        fetcher = EODHDFetcher(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"Name": "Linklogis Inc.", "Code": "2154"}
        )

        mock_response_cm = AsyncMock()
        mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_cm)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await fetcher.get_company_name("2154.HK")

        assert result == "Linklogis Inc."

    @pytest.mark.asyncio
    async def test_eodhd_get_company_name_not_available(self):
        """EODHD not available — returns None."""
        from src.data.eodhd_fetcher import EODHDFetcher

        fetcher = EODHDFetcher(api_key=None)
        fetcher.api_key = None

        result = await fetcher.get_company_name("2154.HK")
        assert result is None


class TestAgentPromptWarning:
    """Test that unresolved company name injects warning into agent prompts."""

    def test_company_line_resolved(self):
        """When resolved, no warning appended."""
        from src.agents import _company_line

        line = _company_line("Toyota Motor", resolved=True)
        assert line == "Company: Toyota Motor"
        assert "WARNING" not in line

    def test_company_line_unresolved(self):
        """When unresolved, warning is appended."""
        from src.agents import _company_line

        line = _company_line("2154.HK", resolved=False)
        assert line.startswith("Company: 2154.HK")
        assert "WARNING" in line
        assert "Do NOT guess" in line

    def test_agent_state_has_company_name_resolved(self):
        """AgentState includes company_name_resolved field."""
        from src.agents import AgentState

        # Check via __annotations__ that the field exists
        annotations = AgentState.__annotations__
        assert "company_name_resolved" in annotations


class TestCompanyNameResult:
    """Test the dataclass."""

    def test_resolved_result(self):
        result = CompanyNameResult(
            name="Toyota Motor", source="yfinance", is_resolved=True
        )
        assert result.name == "Toyota Motor"
        assert result.source == "yfinance"
        assert result.is_resolved is True

    def test_unresolved_result(self):
        result = CompanyNameResult(
            name="7203.T", source="unresolved", is_resolved=False
        )
        assert result.name == "7203.T"
        assert result.source == "unresolved"
        assert result.is_resolved is False
