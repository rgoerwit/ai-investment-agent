"""
Robustness Tests for Data Validation

Checks edge cases, corrupted data, and data integrity guardrails.
"""

import pytest

from src.data.validator import FineGrainedValidator


@pytest.fixture
def validator():
    return FineGrainedValidator()


class TestValidatorEdges:
    """Test edge cases for validation."""

    def test_empty_input(self, validator):
        """Validator should handle empty data gracefully."""
        data = {}
        result = validator.validate_comprehensive(data, "AAPL")

        assert result.basics_ok is False
        assert "Missing 'symbol' field" in result.results[0].issues

    def test_partial_garbage_data(self, validator):
        """Validator should identify garbage fields without crashing."""
        data = {
            "symbol": "AAPL",
            "currentPrice": "NOT_A_NUMBER",  # Garbage string
            "currency": "USD",
        }

        # Should not raise TypeError
        result = validator._validate_basics(data, "AAPL")

        # Should fail validation gracefully
        assert result.passed is False
        assert any("Invalid price" in issue for issue in result.issues) or any(
            "Missing price" in issue for issue in result.issues
        )

    def test_conflicting_data(self, validator):
        """Test detection of internal inconsistencies."""
        data = {
            "trailingPE": 10.0,
            "trailingEps": 5.0,
            "currentPrice": 1000.0,  # Implied PE = 200, Actual PE = 10 -> Conflict
            "marketCap": 1000000,
        }

        result = validator._validate_valuation(data, "AAPL")

        # Check for warning about PE/EPS inconsistency
        has_warning = any("PE/EPS inconsistency" in w for w in result.warnings)
        assert has_warning, "Should detect PE * EPS != Price conflict"

    def test_extreme_outliers(self, validator):
        """Test guardrails against extreme values."""
        data = {
            "profitMargins": 50.0,  # 5000% profit margin -> Likely error
            "grossMargins": 0.5,
        }

        result = validator._validate_profitability(data, "AAPL")

        has_warning = any("outside normal range" in w for w in result.warnings)
        assert has_warning, "Should flag 5000% profit margin"

    def test_financial_sector_exceptions(self, validator):
        """Ensure banks aren't penalized for normal bank metrics (negative OCF)."""
        data = {
            "industry": "Major Banks",
            "operatingCashflow": -5000000,  # Normal for banks (lending)
            "freeCashflow": -5000000,
        }

        # For a non-bank, this would warn. For a bank, it should be quieter.
        result = validator._validate_financial_health(data, "JPM")

        # Should NOT have the negative cash flow warning
        has_warning = any("Negative operating cash flow" in w for w in result.warnings)
        assert not has_warning, "Banks should be exempt from negative OCF warnings"

    def test_non_financial_negative_cash_flow(self, validator):
        """Ensure NON-banks ARE flagged for cash burn."""
        data = {"industry": "Software", "operatingCashflow": -5000000}

        result = validator._validate_financial_health(data, "ABC")

        has_warning = any("Negative operating cash flow" in w for w in result.warnings)
        assert has_warning, "Tech companies burning cash should be flagged"
