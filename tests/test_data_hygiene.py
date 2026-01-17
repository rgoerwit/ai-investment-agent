"""
Comprehensive tests for data hygiene improvements.

Tests cover:
1. Triangle validation (Price × Shares ≈ Market Cap)
2. Staleness detection (timestamp checking)
3. Outlier ratio caps (impossible values)
4. EODHD arbitration (API failure scenarios)
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.data.eodhd_fetcher import EODHDFetcher
from src.data.validator import FineGrainedValidator


class TestTriangleValidation:
    """Test Price × Shares ≈ Market Cap validation."""

    def setup_method(self):
        self.validator = FineGrainedValidator()

    def test_triangle_perfect_match(self):
        """Perfect triangle: 10 × 100 = 1000."""
        data = {
            "currentPrice": 10.0,
            "sharesOutstanding": 100,
            "marketCap": 1000.0,
        }
        result = self.validator._validate_triangle(data, "TEST")

        assert result.passed is True
        assert "market_cap_triangle" in result.validated_fields
        assert len(result.issues) == 0
        assert len(result.warnings) == 0

    def test_triangle_within_tolerance(self):
        """Within 15% tolerance: 10 × 100 = 1000, reported 1100."""
        data = {
            "currentPrice": 10.0,
            "sharesOutstanding": 100,
            "marketCap": 1100.0,  # 10% higher
        }
        result = self.validator._validate_triangle(data, "TEST")

        assert result.passed is True
        assert "market_cap_triangle" in result.validated_fields

    def test_triangle_pence_trap(self):
        """100x unit mismatch (UK stocks in Pence): 1000 × 100 = 100k vs 1k."""
        data = {
            "currentPrice": 1000.0,  # Actually in Pence, should be 10.0 Pounds
            "sharesOutstanding": 100,
            "marketCap": 1000.0,
        }
        result = self.validator._validate_triangle(data, "BP.L")

        assert result.passed is False
        assert len(result.issues) == 1
        assert "Unit mismatch (100x)" in result.issues[0]
        assert "Pence/Cents" in result.issues[0]

    def test_triangle_large_divergence(self):
        """Large divergence (not 100x): calc 10k vs reported 1M."""
        data = {
            "currentPrice": 10.0,
            "sharesOutstanding": 1000,
            "marketCap": 1000000.0,  # 100x larger, not Pence issue
        }
        result = self.validator._validate_triangle(data, "TEST")

        assert result.passed is True  # Warnings don't fail validation
        assert len(result.warnings) == 1
        assert "Triangle break" in result.warnings[0]
        assert "0.01" in result.warnings[0]  # Ratio (10k / 1M = 0.01)

    def test_triangle_missing_data(self):
        """Missing components: can't validate."""
        # Missing shares
        data = {"currentPrice": 10.0, "marketCap": 1000.0}
        result = self.validator._validate_triangle(data, "TEST")
        assert result.passed is True  # Can't validate, so passes
        assert "price/shares/marketCap" in result.missing_fields

        # Missing market cap
        data = {"currentPrice": 10.0, "sharesOutstanding": 100}
        result = self.validator._validate_triangle(data, "TEST")
        assert result.passed is True
        assert "price/shares/marketCap" in result.missing_fields

    def test_triangle_zero_market_cap(self):
        """Zero market cap: can't calculate ratio."""
        data = {
            "currentPrice": 10.0,
            "sharesOutstanding": 100,
            "marketCap": 0.0,
        }
        result = self.validator._validate_triangle(data, "TEST")

        assert result.passed is True  # Can't validate
        assert "price/shares/marketCap" in result.missing_fields

    def test_triangle_negative_values(self):
        """Negative values: should handle gracefully."""
        data = {
            "currentPrice": -10.0,
            "sharesOutstanding": 100,
            "marketCap": 1000.0,
        }
        result = self.validator._validate_triangle(data, "TEST")

        # Will calculate ratio and flag as broken
        assert len(result.warnings) > 0 or len(result.issues) > 0


class TestStalenessValidation:
    """Test data staleness detection."""

    def setup_method(self):
        self.validator = FineGrainedValidator()
        self.current_time = time.time()

    def test_staleness_fresh_data(self):
        """Data from 1 month ago: fresh."""
        one_month_ago = self.current_time - (30 * 24 * 60 * 60)
        data = {"lastFiscalYearEnd": one_month_ago}

        result = self.validator._validate_staleness(data, "TEST")

        assert result.passed is True
        assert len(result.warnings) == 0
        assert "staleness_check_lastFiscalYearEnd" in result.validated_fields

    def test_staleness_stale_data(self):
        """Data from 24 months ago: stale."""
        two_years_ago = self.current_time - (24 * 30.44 * 24 * 60 * 60)
        data = {"lastFiscalYearEnd": two_years_ago}

        result = self.validator._validate_staleness(data, "TEST")

        assert result.passed is True  # Warnings don't fail
        assert len(result.warnings) == 1
        assert "24 months old" in result.warnings[0]
        assert "lastFiscalYearEnd" in result.warnings[0]

    def test_staleness_borderline(self):
        """Data at exactly 18 months: edge case."""
        eighteen_months = self.current_time - (18 * 30.44 * 24 * 60 * 60)
        data = {"lastFiscalYearEnd": eighteen_months}

        result = self.validator._validate_staleness(data, "TEST")

        # At exactly 18 months, should be flagged as stale (>= threshold)
        assert len(result.warnings) == 1
        assert "18 months old" in result.warnings[0]

    def test_staleness_no_timestamp(self):
        """No timestamp fields: can't check."""
        data = {"someOtherField": 123}

        result = self.validator._validate_staleness(data, "TEST")

        assert result.passed is True
        assert len(result.warnings) == 1
        assert "No timestamp available" in result.warnings[0]

    def test_staleness_fallback_fields(self):
        """Uses fallback timestamp fields in order."""
        old_earnings = self.current_time - (24 * 30.44 * 24 * 60 * 60)
        data = {
            "earningsTimestampEnd": old_earnings,
            # lastFiscalYearEnd missing, should use earningsTimestampEnd
        }

        result = self.validator._validate_staleness(data, "TEST")

        assert len(result.warnings) == 1
        assert "earningsTimestampEnd" in result.warnings[0]

    def test_staleness_prefers_fiscal_year(self):
        """Prefers lastFiscalYearEnd over other timestamps."""
        old_fiscal = self.current_time - (24 * 30.44 * 24 * 60 * 60)
        fresh_earnings = self.current_time - (1 * 30.44 * 24 * 60 * 60)

        data = {
            "lastFiscalYearEnd": old_fiscal,
            "earningsTimestampEnd": fresh_earnings,
        }

        result = self.validator._validate_staleness(data, "TEST")

        # Should use lastFiscalYearEnd (preferred), flag as stale
        assert len(result.warnings) == 1
        assert "lastFiscalYearEnd" in result.warnings[0]


class TestOutlierRatioCaps:
    """Test outlier ratio capping logic."""

    def setup_method(self):
        self.validator = FineGrainedValidator()

    def test_outlier_gross_margin_over_100(self):
        """Gross margin > 100%: impossible, cap to None."""
        data = {"grossMargins": 1.5}  # 150%

        result = self.validator._validate_outlier_ratios(data, "TEST")

        assert data["grossMargins"] is None
        assert data["_gross_margin_capped"] is True
        assert len(result["notes"]) == 1
        assert "150.0%" in result["notes"][0]

    def test_outlier_dividend_yield_over_20(self):
        """Dividend yield > 20%: likely error."""
        data = {"dividendYield": 0.25}  # 25%

        result = self.validator._validate_outlier_ratios(data, "TEST")

        # Don't nullify, just flag
        assert data["dividendYield"] == 0.25
        assert data["_dividend_yield_suspect"] is True
        assert len(result["notes"]) == 1
        assert "25.0%" in result["notes"][0]

    def test_outlier_pe_over_1000(self):
        """P/E > 1000: earnings near-zero, treat as N/A."""
        data = {"trailingPE": 5000}

        result = self.validator._validate_outlier_ratios(data, "TEST")

        assert data["trailingPE"] is None
        assert data["_pe_capped"] is True
        assert len(result["notes"]) == 1
        assert "5000" in result["notes"][0]

    def test_outlier_multiple_caps(self):
        """Multiple outliers: cap all."""
        data = {
            "grossMargins": 2.0,  # 200%
            "dividendYield": 0.30,  # 30%
            "trailingPE": 10000,
        }

        result = self.validator._validate_outlier_ratios(data, "TEST")

        assert data["grossMargins"] is None
        assert data["trailingPE"] is None
        assert data["_gross_margin_capped"] is True
        assert data["_pe_capped"] is True
        assert data["_dividend_yield_suspect"] is True
        assert len(result["notes"]) == 3

    def test_outlier_normal_values(self):
        """Normal values: no capping."""
        data = {
            "grossMargins": 0.45,  # 45%
            "dividendYield": 0.03,  # 3%
            "trailingPE": 15,
        }

        result = self.validator._validate_outlier_ratios(data, "TEST")

        assert data["grossMargins"] == 0.45
        assert data["dividendYield"] == 0.03
        assert data["trailingPE"] == 15
        assert len(result["notes"]) == 0

    def test_outlier_edge_cases(self):
        """Edge case values at boundaries."""
        # Exactly 100% gross margin
        data = {"grossMargins": 1.0}
        result = self.validator._validate_outlier_ratios(data, "TEST")
        assert data["grossMargins"] == 1.0  # Not capped
        assert len(result["notes"]) == 0

        # Exactly 20% dividend yield
        data = {"dividendYield": 0.20}
        result = self.validator._validate_outlier_ratios(data, "TEST")
        assert data["dividendYield"] == 0.20  # Not flagged
        assert len(result["notes"]) == 0

        # Exactly 1000 P/E
        data = {"trailingPE": 1000}
        result = self.validator._validate_outlier_ratios(data, "TEST")
        assert data["trailingPE"] == 1000  # Not capped
        assert len(result["notes"]) == 0


@pytest.mark.asyncio
class TestEODHDArbitration:
    """Test EODHD arbitration for various API failure scenarios."""

    async def test_eodhd_success(self):
        """EODHD returns anchor metrics successfully."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        # Mock successful response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "MarketCapitalization": 1000000000,
                "PERatio": 15.5,
                "DividendYield": 0.025,
            }
        )

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is not None
        assert result["marketCap"] == 1000000000
        assert result["trailingPE"] == 15.5
        assert result["dividendYield"] == 0.025
        assert result["_source"] == "EODHD_anchor"

    async def test_eodhd_402_paywall(self):
        """EODHD returns 402 Payment Required."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 402

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("INTL.MX")

        assert result is None
        # Should not disable future calls (could be exchange-specific)
        assert fetcher._is_exhausted is False

    async def test_eodhd_401_auth_error(self):
        """EODHD returns 401 Auth Error."""
        fetcher = EODHDFetcher(api_key="bad_key")
        fetcher._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 401

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is None
        # Should disable future calls
        assert fetcher._is_exhausted is True

    async def test_eodhd_429_rate_limit(self):
        """EODHD returns 429 Rate Limit."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 429

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is None
        # Should disable future calls
        assert fetcher._is_exhausted is True

    async def test_eodhd_404_not_found(self):
        """EODHD returns 404 Not Found."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 404

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("UNKNOWN")

        assert result is None
        assert fetcher._is_exhausted is False

    async def test_eodhd_malformed_json(self):
        """EODHD returns 200 but malformed JSON."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is None

    async def test_eodhd_network_timeout(self):
        """EODHD request times out."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        # Mock timeout exception
        fetcher._session.get = MagicMock(
            side_effect=aiohttp.ClientError("Connection timeout")
        )

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is None

    async def test_eodhd_not_available(self):
        """EODHD not available (no API key)."""
        fetcher = EODHDFetcher(api_key=None)

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is None

    async def test_eodhd_already_exhausted(self):
        """EODHD already rate-limited from previous call."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._is_exhausted = True

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is None

    async def test_eodhd_missing_fields(self):
        """EODHD returns 200 but with missing/null fields."""
        fetcher = EODHDFetcher(api_key="test_key")
        fetcher._session = MagicMock()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                # MarketCapitalization missing
                "PERatio": None,  # Null
                "DividendYield": 0.025,
            }
        )

        mock_ctx_mgr = AsyncMock()
        mock_ctx_mgr.__aenter__.return_value = mock_resp
        fetcher._session.get = MagicMock(return_value=mock_ctx_mgr)

        result = await fetcher.verify_anchor_metrics("AAPL")

        assert result is not None
        assert result["marketCap"] is None
        assert result["trailingPE"] is None
        assert result["dividendYield"] == 0.025


class TestComprehensiveValidation:
    """Test that comprehensive validation includes new checks."""

    def test_comprehensive_includes_triangle(self):
        """Comprehensive validation includes triangle check."""
        validator = FineGrainedValidator()
        data = {
            "symbol": "TEST",
            "currentPrice": 10.0,
            "sharesOutstanding": 100,
            "marketCap": 1000.0,
        }

        overall = validator.validate_comprehensive(data, "TEST")

        # Should have triangle in results
        triangle_result = next(
            (r for r in overall.results if r.category == "triangle"), None
        )
        assert triangle_result is not None
        assert triangle_result.passed is True

    def test_comprehensive_includes_staleness(self):
        """Comprehensive validation includes staleness check."""
        validator = FineGrainedValidator()
        data = {
            "symbol": "TEST",
            "lastFiscalYearEnd": time.time() - (30 * 24 * 60 * 60),  # 1 month ago
        }

        overall = validator.validate_comprehensive(data, "TEST")

        # Should have staleness in results
        staleness_result = next(
            (r for r in overall.results if r.category == "staleness"), None
        )
        assert staleness_result is not None
        assert staleness_result.passed is True

    def test_comprehensive_applies_outlier_caps(self):
        """Comprehensive validation applies outlier caps."""
        validator = FineGrainedValidator()
        data = {
            "symbol": "TEST",
            "trailingPE": 5000,  # Outlier
        }

        overall = validator.validate_comprehensive(data, "TEST")

        # Should have capped P/E
        assert data["trailingPE"] is None
        assert data["_pe_capped"] is True
        assert "_data_quality_notes" in data
