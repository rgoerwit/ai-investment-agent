from __future__ import annotations

from src.article_audit import (
    audit_article_citations,
    extract_source_confidence_context,
    prepend_verification_caveats,
)


def _block(body: str) -> str:
    return (
        f"### --- START DATA_BLOCK ---\n{body.rstrip()}\n### --- END DATA_BLOCK ---\n"
    )


def test_article_citation_audit_catches_mismatch() -> None:
    article = "The firm has no leverage `(NET_DEBT_EBITDA: -0.01)`."
    data_block = _block("NET_DEBT_EBITDA: 1.95")

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1
    assert "NET_DEBT_EBITDA: -0.01" in errors[0]["claim"]
    assert "NET_DEBT_EBITDA: 1.95" in errors[0]["ground_truth"]


def test_article_citation_audit_accepts_valid_percent() -> None:
    article = "Returns are solid `(ROIC_PERCENT: 17.43%)`."
    data_block = _block("ROIC_PERCENT: 17.43%")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_accepts_comma_number() -> None:
    article = "Scale is large `(MARKET_CAP: 1,000,000)`."
    data_block = _block("MARKET_CAP: 1000000")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_accepts_currency_suffix_spacing() -> None:
    article = "Cash generation was positive `(OPERATING_CASH_FLOW: R$6.56B)`."
    data_block = _block("OPERATING_CASH_FLOW: R$ 6.56B")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_ignores_non_backticked_parenthetical() -> None:
    article = "Leverage is manageable (NET_DEBT_EBITDA: 1.95, a moderate level)."
    data_block = _block("NET_DEBT_EBITDA: 1.95")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_catches_bare_parenthetical_mismatch() -> None:
    # The 3393.T regression: hallucinated 52-week low in a plain parenthetical.
    article = (
        "The stock trades near its high "
        "(FIFTY_TWO_WEEK_HIGH: 3100.00, FIFTY_TWO_WEEK_LOW: 1850.00 [unverified])."
    )
    data_block = _block(
        """
FIFTY_TWO_WEEK_HIGH: 3100.00
FIFTY_TWO_WEEK_LOW: 2545.00
"""
    )

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1
    assert "FIFTY_TWO_WEEK_LOW: 1850.00" in errors[0]["claim"]
    assert "FIFTY_TWO_WEEK_LOW: 2545.00" in errors[0]["ground_truth"]


def test_article_citation_audit_bare_parenthetical_keeps_thousands_value() -> None:
    article = "Cash flow held up (OPERATING_CASH_FLOW: 3,057M JPY)."
    data_block = _block("OPERATING_CASH_FLOW: 3,057M JPY")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_bare_parenthetical_ignores_unknown_key() -> None:
    # Bare parentheticals never produce missing-field errors — prose like
    # (NOTE: 2 caveats apply) must not be treated as a citation.
    article = "One caveat applies (NOTE: 2 caveats apply)."
    data_block = _block("ROIC_PERCENT: 17.43%")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_bare_parenthetical_ignores_prose_value() -> None:
    article = "The sector is stable (SECTOR: Technology remains dominant)."
    data_block = _block("SECTOR: Technology")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_accepts_quoted_value() -> None:
    # Run-1 false-positive class: the writer copied quoted values from the
    # editor's CORRECTED notes into citations.
    article = 'Cash flow held up `(OPERATING_CASH_FLOW: "3,057M JPY")`.'
    data_block = _block("OPERATING_CASH_FLOW: 3,057M JPY")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_accepts_unverified_tag_on_matching_value() -> None:
    article = "Rates rose (FIFTY_TWO_WEEK_LOW: 2545.00 [unverified])."
    data_block = _block("FIFTY_TWO_WEEK_LOW: 2545.00")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_flags_missing_field() -> None:
    article = "# Title\n\nUnsupported `(FAKE_FIELD: 42)`."
    data_block = _block("ROIC_PERCENT: 17.43%")

    errors = audit_article_citations(article, data_block)
    caveated = prepend_verification_caveats(article, errors)

    assert len(errors) == 1
    assert "No `FAKE_FIELD` field exists" in errors[0]["ground_truth"]
    assert caveated.startswith("## Verification Caveats")


def test_article_citation_audit_flags_only_invalid_citation() -> None:
    article = (
        "Returns are solid `(ROIC_PERCENT: 17.43%)`, "
        "but leverage is misstated `(NET_DEBT_EBITDA: -0.01)`."
    )
    data_block = _block(
        """
ROIC_PERCENT: 17.43%
NET_DEBT_EBITDA: 1.95
"""
    )

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1
    assert "NET_DEBT_EBITDA" in errors[0]["claim"]


def test_article_citation_audit_flags_non_na_cite_against_na_block_value() -> None:
    article = "Growth was strong `(REVENUE_GROWTH_TTM: 20.3%)`."
    data_block = _block("REVENUE_GROWTH_TTM: N/A")

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1
    assert "REVENUE_GROWTH_TTM: N/A" in errors[0]["ground_truth"]


def test_article_citation_audit_empty_article_has_no_errors() -> None:
    assert audit_article_citations("", _block("ROIC_PERCENT: 17.43%")) == []


def test_article_citation_audit_missing_datablock_skips() -> None:
    article = "Unsupported `(ROIC_PERCENT: 17.43%)`."
    data_block = "ROIC_PERCENT: 18.00%\nNarrative without fenced DATA_BLOCK."

    assert audit_article_citations(article, data_block) == []


def test_source_confidence_context_includes_datablock_notes_without_consultant() -> (
    None
):
    data_block = _block(
        """
OPERATING_CASH_FLOW_SOURCE: JUNIOR
OCF_FILING_REASON: filing line unavailable
"""
    )

    context = extract_source_confidence_context(data_block, None)

    assert context.startswith("=== SOURCE CONFIDENCE ===")
    assert "OPERATING_CASH_FLOW_SOURCE: JUNIOR" in context
    assert "OCF_FILING_REASON: filing line unavailable" in context
    assert "SPOT_CHECK" not in context


def test_source_confidence_context_includes_consultant_notes_without_datablock() -> (
    None
):
    consultant_review = "SPOT_CHECK: COVERAGE_GAP on operating cash flow."

    context = extract_source_confidence_context(None, consultant_review)

    assert context.startswith("=== SOURCE CONFIDENCE ===")
    assert "SPOT_CHECK: COVERAGE_GAP on operating cash flow." in context
    assert "aggregator-indicated" in context


def test_source_confidence_context_empty_inputs_returns_empty_string() -> None:
    assert extract_source_confidence_context(None, None) == ""
