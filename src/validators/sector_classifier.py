"""Sector classification helpers for financial red-flag detection."""

from __future__ import annotations

import re
from enum import Enum

import structlog

from src.data_block_utils import extract_last_data_block

logger = structlog.get_logger(__name__)


class Sector(Enum):
    """GICS-aligned sector classifications (Global Industry Classification Standard)."""

    ENERGY = "Energy"
    MATERIALS = "Materials"
    INDUSTRIALS = "Industrials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    HEALTH_CARE = "Health Care"
    FINANCIALS = "Financials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    COMMUNICATION_SERVICES = "Communication Services"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"


FINANCIALS_SECTORS = {Sector.FINANCIALS}
CAPITAL_INTENSIVE_SECTORS = {
    Sector.ENERGY,
    Sector.MATERIALS,
    Sector.UTILITIES,
    Sector.REAL_ESTATE,
}

_GICS_EXACT: dict[str, Sector] = {sector.value.lower(): sector for sector in Sector}

_KEYWORD_MAP: list[tuple[list[str], Sector]] = [
    (
        ["banking", "bank", "financial services", "insurance", "capital markets"],
        Sector.FINANCIALS,
    ),
    (["energy", "oil", "gas", "petroleum"], Sector.ENERGY),
    (
        [
            "materials",
            "mining",
            "chemicals",
            "shipping",
            "commodities",
            "cyclical",
            "tanker",
            "dry bulk",
        ],
        Sector.MATERIALS,
    ),
    (["utilities", "utility", "electric", "water"], Sector.UTILITIES),
    (["real estate", "reit"], Sector.REAL_ESTATE),
    (
        [
            "information technology",
            "technology",
            "software",
            "saas",
            "semiconductor",
        ],
        Sector.INFORMATION_TECHNOLOGY,
    ),
    (["health care", "healthcare", "pharmaceutical", "biotech"], Sector.HEALTH_CARE),
    (
        ["communication services", "telecom", "media", "entertainment"],
        Sector.COMMUNICATION_SERVICES,
    ),
    (
        ["consumer discretionary", "retail", "automotive", "luxury"],
        Sector.CONSUMER_DISCRETIONARY,
    ),
    (
        ["consumer staples", "grocery", "supermarket", "food", "beverage"],
        Sector.CONSUMER_STAPLES,
    ),
    (
        [
            "industrials",
            "industrial",
            "aerospace",
            "defense",
            "conglomerate",
            "general",
            "diversified",
        ],
        Sector.INDUSTRIALS,
    ),
]


def detect_sector(fundamentals_report: str) -> Sector:
    """Detect sector from a fundamentals report, preferring the last DATA_BLOCK."""
    if not fundamentals_report:
        return Sector.INDUSTRIALS

    data_block = extract_last_data_block(fundamentals_report)
    sector_match = (
        re.search(r"SECTOR:\s*(.+?)(?:\n|$)", data_block) if data_block else None
    )

    if not sector_match:
        marker_positions = [
            pos
            for pos in (
                fundamentals_report.find("### --- START DATA_BLOCK"),
                fundamentals_report.find("\nDATA_BLOCK:"),
            )
            if pos >= 0
        ]
        header_region = (
            fundamentals_report[: min(marker_positions)]
            if marker_positions
            else fundamentals_report
        )
        sector_match = re.search(
            r"SECTOR:\s*(.+?)(?:\n|$)", header_region, re.IGNORECASE
        )

    if not sector_match:
        logger.debug("no_sector_found_in_report", fallback="INDUSTRIALS")
        return Sector.INDUSTRIALS

    sector_text = sector_match.group(1).strip()
    exact_match = _GICS_EXACT.get(sector_text.lower())
    if exact_match is not None:
        return exact_match

    sector_lower = sector_text.lower()
    for keywords, sector_enum in _KEYWORD_MAP:
        if any(keyword in sector_lower for keyword in keywords):
            return sector_enum

    logger.debug("unrecognized_sector", sector_text=sector_text, fallback="INDUSTRIALS")
    return Sector.INDUSTRIALS
