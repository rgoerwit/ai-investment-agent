import pytest

from src.earnings_baseline import (
    EARNINGS_BASELINE_STATUSES,
    is_material_baseline_distortion,
    is_unusable_earnings_baseline,
    requires_eps_growth_withholding,
)


@pytest.mark.parametrize("status", sorted(EARNINGS_BASELINE_STATUSES))
def test_baseline_predicates_are_consistent(status: str) -> None:
    expected_unusable = status != "DURABLE"
    expected_distortion = status not in {"DURABLE", "UNKNOWN"}

    assert is_unusable_earnings_baseline(status.lower()) is expected_unusable
    assert is_material_baseline_distortion(status.lower()) is expected_distortion
    assert requires_eps_growth_withholding(status, "RECONCILED") is expected_unusable


def test_unresolved_bridge_withholds_even_with_durable_baseline() -> None:
    assert requires_eps_growth_withholding("durable", "unresolved") is True
