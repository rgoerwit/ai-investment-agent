"""
Tests for Fine-Grained Validator

Tests the fine-grained validation across 5 categories.
"""

from src.data.validator import FineGrainedValidator


class TestBasicsValidation:
    """Test basics validation."""

    def test_valid_basics(self):
        """Test validation passes for valid basics."""
        validator = FineGrainedValidator()

        data = {
            "symbol": "AAPL",
            "currentPrice": 150.0,
            "previousClose": 148.0,
            "currency": "USD",
        }

        result = validator._validate_basics(data, "AAPL")

        assert result.passed is True
        assert len(result.issues) == 0
        assert "symbol" in result.validated_fields
        assert "price" in result.validated_fields

    def test_missing_price(self):
        """Test validation fails when price is missing."""
        validator = FineGrainedValidator()

        data = {"symbol": "AAPL", "currency": "USD"}

        result = validator._validate_basics(data, "AAPL")

        assert result.passed is False
        assert len(result.issues) > 0
        assert any("price" in issue.lower() for issue in result.issues)

    def test_large_price_change(self):
        """Test warning for large price changes."""
        validator = FineGrainedValidator()

        data = {
            "symbol": "AAPL",
            "currentPrice": 200.0,
            "previousClose": 100.0,  # 100% jump
            "currency": "USD",
        }

        result = validator._validate_basics(data, "AAPL")

        assert result.passed is True  # Basics still pass
        assert len(result.warnings) > 0
        assert any("price change" in warning.lower() for warning in result.warnings)


class TestValuationValidation:
    """Test valuation metrics validation."""

    def test_valid_valuation(self):
        """Test validation passes for valid valuation metrics."""
        validator = FineGrainedValidator()

        data = {
            "trailingPE": 25.5,
            "priceToBook": 6.5,
            "pegRatio": 2.1,
            "marketCap": 2000000000000,
        }

        result = validator._validate_valuation(data, "AAPL")

        assert result.passed is True
        assert "trailingPE" in result.validated_fields
        assert "priceToBook" in result.validated_fields

    def test_negative_pe(self):
        """Test warning for negative P/E."""
        validator = FineGrainedValidator()

        data = {"trailingPE": -10.0, "priceToBook": 1.5, "marketCap": 1000000000}

        result = validator._validate_valuation(data, "AAPL")

        assert result.passed is True
        assert len(result.warnings) > 0
        assert any("negative p/e" in warning.lower() for warning in result.warnings)

    def test_extreme_pe(self):
        """Test warning for extreme P/E."""
        validator = FineGrainedValidator()

        data = {"trailingPE": 5000.0, "marketCap": 1000000000}

        result = validator._validate_valuation(data, "AAPL")

        assert result.passed is True
        assert len(result.warnings) > 0
        assert any("extreme" in warning.lower() for warning in result.warnings)


class TestProfitabilityValidation:
    """Test profitability metrics validation."""

    def test_valid_profitability(self):
        """Test validation passes for valid profitability."""
        validator = FineGrainedValidator()

        data = {
            "profitMargins": 0.25,
            "operatingMargins": 0.30,
            "grossMargins": 0.45,
            "returnOnEquity": 0.40,
            "returnOnAssets": 0.20,
        }

        result = validator._validate_profitability(data, "AAPL")

        assert result.passed is True
        assert "profitMargins" in result.validated_fields
        assert "returnOnEquity" in result.validated_fields

    def test_margin_out_of_range(self):
        """Test warning for margins outside normal range."""
        validator = FineGrainedValidator()

        data = {
            "profitMargins": 1.5,  # 150%, unusual
            "grossMargins": 0.45,
        }

        result = validator._validate_profitability(data, "AAPL")

        assert result.passed is True
        assert len(result.warnings) > 0


class TestFinancialHealthValidation:
    """Test financial health metrics validation."""

    def test_valid_financial_health(self):
        """Test validation passes for valid financial health."""
        validator = FineGrainedValidator()

        data = {
            "debtToEquity": 1.5,
            "currentRatio": 1.2,
            "quickRatio": 0.9,
            "operatingCashflow": 100000000000,
            "freeCashflow": 80000000000,
        }

        result = validator._validate_financial_health(data, "AAPL")

        assert result.passed is True
        assert "debtToEquity" in result.validated_fields
        assert "currentRatio" in result.validated_fields

    def test_negative_current_ratio(self):
        """Test error for negative current ratio."""
        validator = FineGrainedValidator()

        data = {"currentRatio": -0.5, "debtToEquity": 1.0}

        result = validator._validate_financial_health(data, "AAPL")

        assert result.passed is False
        assert len(result.issues) > 0
        assert any("current ratio" in issue.lower() for issue in result.issues)

    def test_extreme_leverage(self):
        """Test warning for extreme leverage."""
        validator = FineGrainedValidator()

        data = {"debtToEquity": 15.0, "currentRatio": 1.0}

        result = validator._validate_financial_health(data, "AAPL")

        assert result.passed is True
        assert len(result.warnings) > 0
        assert any("leverage" in warning.lower() for warning in result.warnings)


class TestGrowthValidation:
    """Test growth metrics validation."""

    def test_valid_growth(self):
        """Test validation passes for valid growth metrics."""
        validator = FineGrainedValidator()

        data = {"revenueGrowth": 0.10, "earningsGrowth": 0.15}

        result = validator._validate_growth(data, "AAPL")

        assert result.passed is True
        assert "revenueGrowth" in result.validated_fields
        assert "earningsGrowth" in result.validated_fields

    def test_extreme_growth(self):
        """Test warning for extreme growth."""
        validator = FineGrainedValidator()

        data = {
            "revenueGrowth": 15.0,  # 1500%, extreme
            "earningsGrowth": 0.20,
        }

        result = validator._validate_growth(data, "AAPL")

        assert result.passed is True
        assert len(result.warnings) > 0


class TestComprehensiveValidation:
    """Test comprehensive validation across all categories."""

    def test_comprehensive_validation(self):
        """Test comprehensive validation returns all category results."""
        validator = FineGrainedValidator()

        data = {
            # Basics
            "symbol": "AAPL",
            "currentPrice": 150.0,
            "currency": "USD",
            # Valuation
            "trailingPE": 25.5,
            "priceToBook": 6.5,
            "marketCap": 2000000000000,
            # Profitability
            "profitMargins": 0.25,
            "returnOnEquity": 0.40,
            # Financial Health
            "debtToEquity": 1.5,
            "currentRatio": 1.2,
            # Growth
            "revenueGrowth": 0.10,
        }

        validation = validator.validate_comprehensive(data, "AAPL")

        assert validation.categories_total == 5
        assert validation.basics_ok is True
        assert len(validation.results) == 5

        # Check each category result exists
        categories = [r.category for r in validation.results]
        assert "basics" in categories
        assert "valuation" in categories
        assert "profitability" in categories
        assert "financial_health" in categories
        assert "growth" in categories

    def test_validation_summary(self):
        """Test validation summary generation."""
        validator = FineGrainedValidator()

        data = {
            "symbol": "AAPL",
            "currentPrice": 150.0,
            "currency": "USD",
            "trailingPE": 25.5,
        }

        validation = validator.validate_comprehensive(data, "AAPL")
        summary = validator.get_validation_summary(validation)

        assert isinstance(summary, str)
        assert "Validation Summary" in summary
        assert "BASICS" in summary
        assert "VALUATION" in summary
