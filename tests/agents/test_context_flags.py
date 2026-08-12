from src.agents.context_flags import (
    classify_large_drawdown_context,
    drawdown_flag,
    format_pm_context_flags,
    unresolved_related_tickers,
)


def _fundamentals(**fields: str) -> str:
    lines = ["### --- START DATA_BLOCK ---"]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    lines.append("### --- END DATA_BLOCK ---")
    return "\n".join(lines)


def test_large_drawdown_macro_only_classification() -> None:
    flag = classify_large_drawdown_context(
        "The stock derated on rate shock and multiple compression.",
        current=19_000,
        high=56_700,
    )
    assert flag == "LARGE_DRAWDOWN_MACRO_ONLY"


def test_large_drawdown_ignored_when_not_large() -> None:
    assert classify_large_drawdown_context("rate shock", current=80, high=100) is None


def test_large_drawdown_mixed_when_company_causes_present() -> None:
    flag = classify_large_drawdown_context(
        "The stock fell on multiple compression plus margin pressure and competition.",
        current=50,
        high=100,
    )
    assert flag == "LARGE_DRAWDOWN_MIXED"


def test_large_drawdown_company_specific_classification() -> None:
    flag = classify_large_drawdown_context(
        "Shares fell after a take-rate miss and margin erosion from competition.",
        current=19_000,
        high=56_700,
    )
    assert flag == "LARGE_DRAWDOWN_COMPANY_SPECIFIC"


def test_large_drawdown_unexplained_when_causes_not_near_decline() -> None:
    # Generic cause vocabulary is present, but the decline itself is never
    # discussed — so the drawdown is treated as uninvestigated, not "mixed".
    flag = classify_large_drawdown_context(
        "Revenue and margins are steady and the sector outlook is constructive.",
        current=19_000,
        high=56_700,
    )
    assert flag == "UNEXPLAINED_LARGE_DRAWDOWN"


def test_unresolved_related_tickers_extracts_unknown_entries() -> None:
    fundamentals = _fundamentals(
        RELATED_LISTED_TICKERS=(
            "035420.KS:Strategic_Partner:Unknown; 1234.T:Parent:42%"
        )
    )
    assert unresolved_related_tickers(fundamentals) == [
        "035420.KS:Strategic_Partner:Unknown"
    ]


def test_format_pm_context_flags_combines_drawdown_and_related_ticker() -> None:
    fundamentals = _fundamentals(
        CURRENT_PRICE="19000",
        FIFTY_TWO_WEEK_HIGH="56700",
        FIFTY_TWO_WEEK_LOW="17750",
        RELATED_LISTED_TICKERS="035420.KS:Strategic_Partner:Unknown",
    )
    section = format_pm_context_flags(
        fundamentals,
        "The sell-off reflects rate shock and multiple compression.",
    )
    assert "SUPPLEMENTAL PM FLAGS" in section
    assert "LARGE_DRAWDOWN_MACRO_ONLY" in section
    assert "52-week low 17750" in section
    assert "UNRESOLVED_RELATED_TICKERS" in section
    assert "filing-verification-required" in section


def test_drawdown_flag_shares_classification_with_narrative() -> None:
    """drawdown_flag reads DATA_BLOCK price fields and classifies report text."""
    fundamentals = (
        "### --- START DATA_BLOCK ---\n"
        "CURRENT_PRICE: 5.60\n"
        "FIFTY_TWO_WEEK_HIGH: 9.81\n"
        "### --- END DATA_BLOCK ---"
    )
    market = "Price is in a clear downtrend; technical setup bearish."

    assert drawdown_flag(fundamentals, market) == "LARGE_DRAWDOWN_MACRO_ONLY"
    # No decline discussion anywhere -> uninvestigated.
    assert drawdown_flag(fundamentals, "All quiet.") == "UNEXPLAINED_LARGE_DRAWDOWN"


def test_drawdown_flag_none_without_price_fields() -> None:
    assert drawdown_flag("no data block here", "downtrend text") is None
    assert drawdown_flag(None, "downtrend text") is None
