"""
Tests for src.main CLI argument parsing.

Covers --strict flag parsing and composability with other flags.
"""

import pytest


class TestStrictModeCLI:
    """Test --strict CLI flag is wired correctly."""

    def test_strict_flag_parsed_from_cli(self):
        """--strict sets args.strict = True."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict"])
        assert args.strict is True

    def test_no_strict_flag_defaults_false(self):
        """Without --strict, args.strict = False."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK"])
        assert args.strict is False

    def test_strict_and_quick_composable(self):
        """--strict --quick can be combined without conflict."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict", "--quick"])
        assert args.strict is True
        assert args.quick is True

    def test_strict_with_quiet_composable(self):
        """--strict --quiet can be combined (batch use case)."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict", "--quiet"])
        assert args.strict is True
        assert args.quiet is True

    def test_strict_with_output_composable(self):
        """--strict with --output is valid."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "0005.HK", "--strict", "--output", "results/test.md"]
        )
        assert args.strict is True
        assert args.output == "results/test.md"

    def test_strict_quick_quiet_all_composable(self):
        """--strict --quick --quiet can all be combined (pipeline batch mode)."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "0005.HK", "--strict", "--quick", "--quiet"]
        )
        assert args.strict is True
        assert args.quick is True
        assert args.quiet is True


class TestStrictAddendaContent:
    """Sanity-check the content of _STRICT_PM_ADDENDUM and _STRICT_RM_ADDENDUM.

    No mocking needed — these are pure string checks on module-level constants.
    If content changes break these, it's a signal to update the plan doc too.
    """

    def test_pm_addendum_has_tighter_health_threshold(self):
        """PM addendum must require Financial Health ≥ 60% (tighter than normal 50%)."""
        from src.agents import _STRICT_PM_ADDENDUM

        assert "Financial Health ≥ 60%" in _STRICT_PM_ADDENDUM

    def test_pm_addendum_rejects_pfic_and_vie(self):
        """PM addendum must explicitly disqualify both PFIC and VIE."""
        from src.agents import _STRICT_PM_ADDENDUM

        assert "PFIC" in _STRICT_PM_ADDENDUM
        assert "VIE" in _STRICT_PM_ADDENDUM

    def test_rm_addendum_has_catalyst_requirement(self):
        """RM addendum must require a near-term catalyst in strict mode."""
        from src.agents import _STRICT_RM_ADDENDUM

        assert "catalyst" in _STRICT_RM_ADDENDUM.lower()

    def test_rm_addendum_weights_bear_arguments(self):
        """RM addendum must instruct to weight bear arguments more heavily."""
        from src.agents import _STRICT_RM_ADDENDUM

        assert "bear" in _STRICT_RM_ADDENDUM.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
