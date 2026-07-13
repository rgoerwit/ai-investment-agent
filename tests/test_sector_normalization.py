"""Unit tests for shared sector normalization."""

from __future__ import annotations

from src.sector_normalization import aggregate_sector_weights, normalize_sector_label
from src.thesis_constants import SECTOR_MEDIAN_PE
from src.validators.sector_classifier import Sector, detect_sector


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


def test_sector_median_pe_keys_match_validator_sector_enum_values():
    """SECTOR_MEDIAN_PE must cover exactly the canonical Sector enum values.

    `data/metric_extraction.py` indexes SECTOR_MEDIAN_PE[sector] directly, so a
    renamed/added Sector member or a stray SECTOR_MEDIAN_PE key would KeyError at
    runtime with no other guard. Pin the two together.
    """
    assert set(SECTOR_MEDIAN_PE.keys()) == {sector.value for sector in Sector}


def test_detect_sector_resolves_vendor_consumer_cyclical_not_materials():
    """Yahoo's "Consumer Cyclical" must map to Consumer Discretionary.

    Regression: the substring keyword fallback contains "cyclical" under
    Materials, so "Consumer Cyclical" was misclassified as Materials (a
    capital-intensive profile with looser leverage thresholds).
    """
    report = "DATA_BLOCK:\nSECTOR: Consumer Cyclical\nPE: 6.57\n"
    assert detect_sector(report) is Sector.CONSUMER_DISCRETIONARY


def test_detect_sector_resolves_vendor_consumer_defensive_to_staples():
    report = "DATA_BLOCK:\nSECTOR: Consumer Defensive\n"
    assert detect_sector(report) is Sector.CONSUMER_STAPLES


def test_detect_sector_still_matches_genuine_materials():
    report = "DATA_BLOCK:\nSECTOR: Basic Materials\n"
    assert detect_sector(report) is Sector.MATERIALS
