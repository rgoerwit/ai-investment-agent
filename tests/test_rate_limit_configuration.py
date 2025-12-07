"""
Test suite for configurable rate limit settings.

Ensures that GEMINI_RPM_LIMIT environment variable correctly configures
the rate limiter for different API tiers (free, paid tier 1, tier 2).
"""

import pytest
import os
import importlib
from unittest.mock import patch, MagicMock


class TestRateLimitConfiguration:
    """Test rate limit configuration from environment variables."""

    def test_default_free_tier_15_rpm(self):
        """Test that default rate limit is 15 RPM (free tier) when no env var is set.

        Skips test if GEMINI_RPM_LIMIT is set (user may be on paid tier).
        """
        env_value = os.environ.get("GEMINI_RPM_LIMIT")

        if env_value is not None:
            pytest.skip(f"GEMINI_RPM_LIMIT is set to {env_value}, skipping default test")

        import src.config
        assert src.config.config.gemini_rpm_limit == 15

    def test_paid_tier_1_360_rpm(self):
        """Test setting rate limit to 360 RPM (paid tier 1)."""
        os.environ["GEMINI_RPM_LIMIT"] = "360"

        import src.config
        importlib.reload(src.config)

        assert src.config.config.gemini_rpm_limit == 360

        # Clean up
        del os.environ["GEMINI_RPM_LIMIT"]

    def test_paid_tier_2_1000_rpm(self):
        """Test setting rate limit to 1000 RPM (paid tier 2)."""
        os.environ["GEMINI_RPM_LIMIT"] = "1000"

        import src.config
        importlib.reload(src.config)

        assert src.config.config.gemini_rpm_limit == 1000

        # Clean up
        del os.environ["GEMINI_RPM_LIMIT"]

    def test_custom_rpm_limit(self):
        """Test setting custom RPM limit."""
        os.environ["GEMINI_RPM_LIMIT"] = "500"

        import src.config
        importlib.reload(src.config)

        assert src.config.config.gemini_rpm_limit == 500

        # Clean up
        del os.environ["GEMINI_RPM_LIMIT"]


class TestRateLimiterCreation:
    """Test rate limiter creation from RPM settings."""

    def test_rate_limiter_15_rpm_calculation(self):
        """Test rate limiter RPS calculation for 15 RPM (free tier)."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(15)

        # 15 RPM with 80% safety factor:
        # RPS = (15 / 60) * 0.8 = 0.2 RPS
        expected_rps = (15 / 60.0) * 0.8
        assert limiter.requests_per_second == pytest.approx(expected_rps, abs=0.01)

        # Bucket size: max(5, int(15 * 0.1)) = max(5, 1) = 5
        assert limiter.max_bucket_size == 5

    def test_rate_limiter_360_rpm_calculation(self):
        """Test rate limiter RPS calculation for 360 RPM (paid tier 1)."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(360)

        # 360 RPM with 80% safety factor:
        # RPS = (360 / 60) * 0.8 = 4.8 RPS
        expected_rps = (360 / 60.0) * 0.8
        assert limiter.requests_per_second == pytest.approx(expected_rps, abs=0.01)

        # Bucket size: max(5, int(360 * 0.1)) = max(5, 36) = 36
        assert limiter.max_bucket_size == 36

    def test_rate_limiter_1000_rpm_calculation(self):
        """Test rate limiter RPS calculation for 1000 RPM (paid tier 2)."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(1000)

        # 1000 RPM with 80% safety factor:
        # RPS = (1000 / 60) * 0.8 = 13.33 RPS
        expected_rps = (1000 / 60.0) * 0.8
        assert limiter.requests_per_second == pytest.approx(expected_rps, abs=0.01)

        # Bucket size: max(5, int(1000 * 0.1)) = max(5, 100) = 100
        assert limiter.max_bucket_size == 100

    def test_rate_limiter_minimum_bucket_size(self):
        """Test that bucket size has minimum of 5."""
        from src.llms import _create_rate_limiter_from_rpm

        # Very low RPM (2 RPM)
        limiter = _create_rate_limiter_from_rpm(2)

        # Bucket size: max(5, int(2 * 0.1)) = max(5, 0) = 5
        assert limiter.max_bucket_size == 5


class TestSafetyMargin:
    """Test 20% safety margin in rate calculations."""

    def test_15_rpm_uses_80_percent(self):
        """Test that 15 RPM uses 80% safety margin (12 effective RPM)."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(15)

        # Effective RPS: (15 * 0.8) / 60 = 12 / 60 = 0.2
        # This leaves 20% buffer to avoid hitting the actual 15 RPM limit
        effective_rpm = limiter.requests_per_second * 60
        assert effective_rpm == pytest.approx(12.0, abs=0.1)

    def test_360_rpm_uses_80_percent(self):
        """Test that 360 RPM uses 80% safety margin (288 effective RPM)."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(360)

        # Effective RPS: (360 * 0.8) / 60 = 288 / 60 = 4.8
        effective_rpm = limiter.requests_per_second * 60
        assert effective_rpm == pytest.approx(288.0, abs=0.1)

    def test_1000_rpm_uses_80_percent(self):
        """Test that 1000 RPM uses 80% safety margin (800 effective RPM)."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(1000)

        # Effective RPS: (1000 * 0.8) / 60 = 800 / 60 = 13.33
        effective_rpm = limiter.requests_per_second * 60
        assert effective_rpm == pytest.approx(800.0, abs=0.1)


class TestGlobalRateLimiterInitialization:
    """Test that global rate limiter is initialized correctly."""

    def test_global_rate_limiter_uses_config(self):
        """Test that GLOBAL_RATE_LIMITER uses config.gemini_rpm_limit."""
        # Set specific RPM
        os.environ["GEMINI_RPM_LIMIT"] = "100"

        # Reimport to pick up new config
        import src.config
        import src.llms
        importlib.reload(src.config)
        importlib.reload(src.llms)

        # Global rate limiter should use 100 RPM
        # RPS = (100 / 60) * 0.8 = 1.33 RPS
        expected_rps = (100 / 60.0) * 0.8
        assert src.llms.GLOBAL_RATE_LIMITER.requests_per_second == pytest.approx(
            expected_rps, abs=0.01
        )

        # Clean up
        del os.environ["GEMINI_RPM_LIMIT"]
        importlib.reload(src.config)
        importlib.reload(src.llms)


class TestTierComparison:
    """Test performance differences between tiers."""

    def test_paid_tier_1_is_24x_faster_than_free(self):
        """Verify that paid tier 1 (360 RPM) is 24x faster than free (15 RPM)."""
        from src.llms import _create_rate_limiter_from_rpm

        free_limiter = _create_rate_limiter_from_rpm(15)
        paid1_limiter = _create_rate_limiter_from_rpm(360)

        # Ratio should be 360/15 = 24
        ratio = paid1_limiter.requests_per_second / free_limiter.requests_per_second
        assert ratio == pytest.approx(24.0, abs=0.1)

    def test_paid_tier_2_is_67x_faster_than_free(self):
        """Verify that paid tier 2 (1000 RPM) is ~67x faster than free (15 RPM)."""
        from src.llms import _create_rate_limiter_from_rpm

        free_limiter = _create_rate_limiter_from_rpm(15)
        paid2_limiter = _create_rate_limiter_from_rpm(1000)

        # Ratio should be 1000/15 = 66.67
        ratio = paid2_limiter.requests_per_second / free_limiter.requests_per_second
        assert ratio == pytest.approx(66.67, abs=0.1)

    def test_paid_tier_2_is_2_8x_faster_than_tier_1(self):
        """Verify that paid tier 2 (1000 RPM) is ~2.8x faster than tier 1 (360 RPM)."""
        from src.llms import _create_rate_limiter_from_rpm

        paid1_limiter = _create_rate_limiter_from_rpm(360)
        paid2_limiter = _create_rate_limiter_from_rpm(1000)

        # Ratio should be 1000/360 = 2.78
        ratio = paid2_limiter.requests_per_second / paid1_limiter.requests_per_second
        assert ratio == pytest.approx(2.78, abs=0.1)


class TestEdgeCases:
    """Test edge cases in rate limit configuration."""

    def test_very_low_rpm(self):
        """Test that very low RPM values work correctly."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(1)

        # 1 RPM with 80% safety factor:
        # RPS = (1 / 60) * 0.8 = 0.0133 RPS
        expected_rps = (1 / 60.0) * 0.8
        assert limiter.requests_per_second == pytest.approx(expected_rps, abs=0.001)

        # Minimum bucket size
        assert limiter.max_bucket_size == 5

    def test_very_high_rpm(self):
        """Test that very high RPM values work correctly."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(10000)

        # 10000 RPM with 80% safety factor:
        # RPS = (10000 / 60) * 0.8 = 133.33 RPS
        expected_rps = (10000 / 60.0) * 0.8
        assert limiter.requests_per_second == pytest.approx(expected_rps, abs=0.1)

        # Bucket size: max(5, int(10000 * 0.1)) = 1000
        assert limiter.max_bucket_size == 1000

    def test_invalid_rpm_string_raises_error(self):
        """Test that invalid RPM string in env var raises error."""
        os.environ["GEMINI_RPM_LIMIT"] = "invalid"

        with pytest.raises(ValueError):
            import src.config
            importlib.reload(src.config)

        # Clean up
        del os.environ["GEMINI_RPM_LIMIT"]


class TestLogging:
    """Test that rate limiter initialization produces correct configuration."""

    def test_rate_limiter_configuration_values(self):
        """Test that rate limiter creation produces correct configuration values."""
        from src.llms import _create_rate_limiter_from_rpm

        limiter = _create_rate_limiter_from_rpm(360)

        # Verify the limiter was configured correctly (this is what logging would show)
        # 360 RPM with 80% safety factor = 4.8 RPS
        expected_rps = (360 / 60.0) * 0.8
        assert limiter.requests_per_second == pytest.approx(expected_rps, abs=0.01)
        assert limiter.max_bucket_size == 36

    def test_different_tiers_produce_different_configurations(self):
        """Test that different tier configurations produce different rate limiters."""
        from src.llms import _create_rate_limiter_from_rpm

        # Free tier
        free_limiter = _create_rate_limiter_from_rpm(15)
        assert free_limiter.requests_per_second == pytest.approx(0.2, abs=0.01)
        assert free_limiter.max_bucket_size == 5

        # Paid tier 1
        paid1_limiter = _create_rate_limiter_from_rpm(360)
        assert paid1_limiter.requests_per_second == pytest.approx(4.8, abs=0.01)
        assert paid1_limiter.max_bucket_size == 36

        # Paid tier 2
        paid2_limiter = _create_rate_limiter_from_rpm(1000)
        assert paid2_limiter.requests_per_second == pytest.approx(13.33, abs=0.01)
        assert paid2_limiter.max_bucket_size == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
