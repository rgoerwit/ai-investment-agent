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
    assert caveated.startswith("# Title\n\n## Verification Caveats")


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


def test_article_citation_audit_accepts_trailing_qualifier_on_block_value() -> None:
    # Run-3 false-positive class: the DATA_BLOCK value carries a trailing
    # parenthetical qualifier the article legitimately omits.
    article = "Health is strong `(ADJUSTED_HEALTH_SCORE: 91.7%)`."
    data_block = _block("ADJUSTED_HEALTH_SCORE: 91.7% (based on 12 available points)")

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_still_flags_mismatch_despite_qualifier() -> None:
    article = "Health is strong `(ADJUSTED_HEALTH_SCORE: 92.1%)`."
    data_block = _block("ADJUSTED_HEALTH_SCORE: 91.7% (based on 12 available points)")

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1
    assert "ADJUSTED_HEALTH_SCORE: 92.1%" in errors[0]["claim"]


def test_article_citation_audit_checks_forward_pe_identity() -> None:
    article = "The forward multiple looks inexpensive."
    data_block = _block(
        """
CURRENT_PRICE: 183.00
FORWARD_EPS: 18.55
PE_RATIO_FORWARD: 9.37
"""
    )

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1
    assert errors[0]["location"] == "Forward P/E identity audit"
    assert "implies forward P/E 9.87" in errors[0]["ground_truth"]


def test_article_citation_audit_accepts_consistent_forward_pe_identity() -> None:
    article = "The forward multiple looks inexpensive."
    data_block = _block(
        """
CURRENT_PRICE: 183.00
FORWARD_EPS: 19.54
PE_RATIO_FORWARD: 9.37
"""
    )

    assert audit_article_citations(article, data_block) == []


def test_article_citation_audit_qualifier_strip_preserves_units() -> None:
    # Guard against over-normalization: a wrong magnitude suffix must still
    # be flagged even though the numeric prefix matches.
    article = "Cash flow held up `(OPERATING_CASH_FLOW: 3,057B JPY)`."
    data_block = _block("OPERATING_CASH_FLOW: 3,057M JPY (filing-derived)")

    errors = audit_article_citations(article, data_block)

    assert len(errors) == 1


def test_article_citation_audit_accepts_na_with_qualifier() -> None:
    article = "Growth data is unavailable `(REVENUE_GROWTH_TTM: N/A)`."
    data_block = _block("REVENUE_GROWTH_TTM: N/A (data vacuum: yfinance null)")

    assert audit_article_citations(article, data_block) == []


def test_verification_caveats_insert_after_h1_title() -> None:
    article = "# Big Story\n\nBody paragraph."
    errors = [
        {"claim": "Article cites (X: 1)", "ground_truth": "DATA_BLOCK shows X: 2"}
    ]

    caveated = prepend_verification_caveats(article, errors)

    assert caveated.startswith("# Big Story\n\n## Verification Caveats")
    assert "Body paragraph." in caveated
    assert caveated.index("# Big Story") < caveated.index("## Verification Caveats")


def test_verification_caveats_prepend_when_no_h1() -> None:
    article = "Body paragraph without a title."
    errors = [
        {"claim": "Article cites (X: 1)", "ground_truth": "DATA_BLOCK shows X: 2"}
    ]

    caveated = prepend_verification_caveats(article, errors)

    assert caveated.startswith("## Verification Caveats")
    assert article in caveated


def test_verification_caveats_idempotent_after_h1_insert() -> None:
    article = "# Big Story\n\nBody paragraph."
    errors = [
        {"claim": "Article cites (X: 1)", "ground_truth": "DATA_BLOCK shows X: 2"}
    ]

    once = prepend_verification_caveats(article, errors)
    twice = prepend_verification_caveats(once, errors)

    assert twice == once
    assert twice.count("## Verification Caveats") == 1


def test_verification_caveats_no_errors_returns_article_unchanged() -> None:
    article = "# Big Story\n\nBody paragraph."

    assert prepend_verification_caveats(article, []) == article


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


def test_source_confidence_context_constrains_6782_overstatements() -> None:
    data_block = _block(
        """
FORWARD_EPS: 19.54
FORWARD_EPS_SOURCE: yfinance
PE_RATIO_FORWARD: 9.37
PE_RATIO_FORWARD_SOURCE: yfinance
GUIDANCE_SOURCE_AUTHORITY: THIRD_PARTY
CAPACITY_EVIDENCE_STATUS: SECONDARY
MOAT_CFO_NI_AVG: 1.48
MOAT_CFO_NI_YEARS: 3
MOAT_CFO_NI_SOURCE: yfinance
NET_CASH_TO_MARKET_CAP: 2.1%
EARNINGS_GROWTH_FY_SOURCE: NET_INCOME_STATEMENT_PROXY
MRQ_COMPARISON_BASE_STATUS: DEPRESSED
GROWTH_DATA_QUALITY_NOTE: Newer quarter metadata exists for 2026-03-31, but statement-derived MRQ metrics remain aligned to 2025-12-31.
"""
    )

    context = extract_source_confidence_context(data_block, None)

    assert "never as management/company guidance" in context
    assert "keep every operating inference conditional" in context
    assert "different EPS estimate or provider" in context
    assert "not real/proven cash quality" in context
    assert "modest cushion, not a valuation floor" in context
    assert "not proof of a durable multi-year earnings trend" in context
    assert "base-sensitive, not structural acceleration" in context
    assert "never call that statement period the latest reported quarter" in context
    assert "Copy numeric thesis-break or review thresholds exactly" in context
    assert "Use 'margin of safety' only when a downside anchor" in context


def test_source_confidence_context_scopes_latest_actuals_and_coverage_count() -> None:
    data_block = _block(
        """
ANALYST_COVERAGE_ENGLISH: 7
LATEST_RESULTS_PERIOD: Three months ended March 31, 2026
LATEST_RESULTS_PERIOD_END: 2026-03-31
LATEST_RESULTS_EARNINGS_GROWTH_YOY: 102.5%
LATEST_RESULTS_EARNINGS_SCOPE: Net income attributable to owners of parent
LATEST_RESULTS_SOURCE_URL: https://issuer.example/results
LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY
GROWTH_DATA_QUALITY_NOTE: Newer primary results exist for Three months ended March 31, 2026; statement-derived MRQ growth remains aligned to 2025-12-31.
"""
    )

    context = extract_source_confidence_context(data_block, None)

    assert "aggregator analyst-opinion count" in context
    assert "historical actual results in their stated scope and period" in context
    assert "never present actual YoY growth as management guidance" in context
    assert "never call that statement period the latest reported quarter" in context


def test_source_confidence_context_empty_inputs_returns_empty_string() -> None:
    assert extract_source_confidence_context(None, None) == ""
