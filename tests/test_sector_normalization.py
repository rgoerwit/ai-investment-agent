"""Unit tests for shared sector normalization."""

from __future__ import annotations

from src.sector_normalization import aggregate_sector_weights, normalize_sector_label
from src.validators.sector_classifier import Sector


def test_normalize_sector_label_passes_through_canonical_gics_names():
    assert normalize_sector_label("Health Care") == "Health Care"
    assert normalize_sector_label("Information Technology") == "Information Technology"
    assert normalize_sector_label("Consumer Staples") == "Consumer Staples"


def test_normalize_sector_label_maps_aliases_to_canonical_names():
    assert normalize_sector_label("Healthcare") == "Health Care"
    assert normalize_sector_label("Technology") == "Information Technology"
    assert normalize_sector_label("Tech") == "Information Technology"
    assert normalize_sector_label("Financial Services") == "Financials"
    assert normalize_sector_label("Finance") == "Financials"
    assert normalize_sector_label("Basic Materials") == "Materials"
    assert normalize_sector_label("Consumer Cyclical") == "Consumer Discretionary"
    assert normalize_sector_label("Consumer Defensive") == "Consumer Staples"
    assert normalize_sector_label("Telecom") == "Communication Services"


def test_normalize_sector_label_handles_spacing_case_and_unknowns():
    assert normalize_sector_label("  health   care ") == "Health Care"
    assert normalize_sector_label("telecommunications") == "Communication Services"
    assert normalize_sector_label("") == "Unknown"
    assert normalize_sector_label(None) == "Unknown"
    assert normalize_sector_label("Aerospace") == "Unknown"


def test_aggregate_sector_weights_merges_mixed_variants():
    assert aggregate_sector_weights(
        {
            "Healthcare": 10.0,
            "Health Care": 2.5,
            "Technology": 7.0,
            "Information Technology": 1.0,
            "Consumer Defensive": 3.0,
            "Consumer Staples": 4.0,
        }
    ) == {
        "Health Care": 12.5,
        "Information Technology": 8.0,
        "Consumer Staples": 7.0,
    }


def test_canonical_sector_outputs_match_validator_sector_enum_values():
    assert {normalize_sector_label(sector.value) for sector in Sector} == {
        sector.value for sector in Sector
    }
