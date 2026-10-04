"""Regression tests for deterministic FLA ownership/capacity provenance."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.foreign_language_evidence import (
    _reframe_latest_results_block,
    has_foreign_language_protocol_residue,
    normalize_foreign_language_evidence,
    promote_foreign_growth_evidence,
)
from src.agents.message_utils import (
    ToolEvidenceRecord,
    evidence_record_to_tool_evidence,
    make_tool_evidence_record,
)
from src.agents.output_validation import (
    _has_valid_latest_results_block,
    validate_required_output,
)
from src.data_block_utils import extract_last_fenced_block
from src.graph.builder import _reconcile_fundamentals_evidence
from src.tooling.evidence_recorder import EvidenceRecord
from src.validators.entity_governance_card import build_card
from tests.helpers.frozen_regressions import load_frozen_regression


def _report(
    *,
    holder: str = "BenQ Materials Corp. (14.82%)",
    controller: str = "BenQ Materials Corp. (14.82%)",
    status: str = "CONTROLLED",
    basis: str = "CONSOLIDATED_SUBSIDIARY",
    relationship: str = "subsidiary",
    entity_role: str = "LISTED_SUBSIDIARY",
    related: str = "2449.TW:parent:14.82",
    source_url: str = "https://www.viscovision.com.tw/tw/investors_shareholders.html",
    capacity: str = "N/A",
    capacity_url: str = "N/A",
) -> str:
    return f"""CAPACITY_UTILIZATION: {capacity}
CAPACITY_UTILIZATION_SOURCE_URL: {capacity_url}
CAPACITY_UTILIZATION_AS_OF: 2026-Q1
FACILITY_BUILDOUT_STATUS: AT_CAPACITY

**OWNERSHIP STRUCTURE**
- Largest Shareholder: {holder}
- Controlling Shareholder: {controller}
- Control Status: {status}
- Control Basis: {basis}
- Parent Company: BenQ Materials Corp.
- Relationship: {relationship}
- ENTITY_ROLE_OBSERVED: {entity_role}
- Related Listed Tickers: {related}
- Ownership Evidence Status: CITED
- Ownership Source URL: {source_url}
- Ownership As Of: 2026-03-28
"""


def _tool(content: str, *, name: str = "web_search") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="call-1", name=name)


def _record(
    content: str,
    *,
    name: str = "web_search",
    urls: set[str] | tuple[str, ...] = (),
) -> ToolEvidenceRecord:
    return make_tool_evidence_record(
        tool_name=name,
        content=content,
        urls=urls,
    )


def _latest_results_report(
    *,
    source_url: str = "https://issuer.example/results",
    prior_earnings: str = "200",
) -> str:
    return f"""### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: FOUND
LATEST_RESULTS_PERIOD: Three months ended March 31, 2026
LATEST_RESULTS_PERIOD_END: 2026-03-31
LATEST_RESULTS_PRIOR_PERIOD: Three months ended March 31, 2025
LATEST_RESULTS_PRIOR_PERIOD_END: 2025-03-31
LATEST_RESULTS_PERIOD_MONTHS: 3
LATEST_RESULTS_CURRENCY: New dollars
LATEST_RESULTS_REPORTING_UNIT: thousands
LATEST_RESULTS_REVENUE: 1,500
LATEST_RESULTS_PRIOR_REVENUE: 1,000
LATEST_RESULTS_EARNINGS: 405
LATEST_RESULTS_PRIOR_EARNINGS: {prior_earnings}
LATEST_RESULTS_EARNINGS_SCOPE: Net income attributable to owners of parent
LATEST_RESULTS_SOURCE_URL: {source_url}
### --- END LATEST_RESULTS ---
"""


def _latest_results_evidence(source_url: str) -> str:
    return f"""DOCUMENT_METADATA: {{"source_url": "{source_url}"}}
Results comparison
 Three months ended March 31, 2026
 Three months ended March 31, 2025
2026-03-31
2025-03-31
 Currency: New dollars
Reporting unit: thousands
Revenue
1,500
1,000
Net income attributable to owners of parent
405
200
"""


def test_leading_tool_protocol_preamble_is_removed_without_losing_narrative():
    report = (
        "Legitimate local-language summary.\n"
        '{"ticker":"1401.T","purpose":"latest_results"}\n'
        "to=functions.search_foreign_sources  ðjson\n" + _latest_results_report()
    )

    normalized = normalize_foreign_language_evidence(report, [], ticker="1401.T")

    assert normalized.startswith("Legitimate local-language summary.\n### --- START")
    assert not has_foreign_language_protocol_residue(normalized)


def test_noncontiguous_protocol_residue_is_not_silently_sanitized():
    report = (
        "to=functions.search_foreign_sources\n"
        "unrecognized transport fragment\n" + _latest_results_report()
    )

    normalized = normalize_foreign_language_evidence(report, [], ticker="1401.T")

    assert has_foreign_language_protocol_residue(normalized)


def test_protocol_words_in_legitimate_evidence_are_not_residue():
    report = "The filing describes a tool function used in production.\n" + _report()

    normalized = normalize_foreign_language_evidence(report, [], ticker="AAPL")

    assert "tool function used in production" in normalized
    assert not has_foreign_language_protocol_residue(normalized)


def test_6782_equity_method_evidence_is_not_promoted_to_control():
    regression = load_frozen_regression("6782_TW_regression.json")
    ownership = regression["ownership_evidence"]
    source = ownership["sources"][0]["url"]
    report = _report(
        controller="NONE",
        status="NOT_CONTROLLED",
        basis="SIGNIFICANT_INFLUENCE_ONLY",
        relationship="equity method",
        entity_role="STANDALONE",
        related="8215.TW:significant_influence:14.82",
        source_url=source,
    )
    messages = [
        _tool(
            "BenQ Materials Corp. (8215.TW) owns 14.82% of Visco Vision "
            f"as of {ownership['captured_as_of']}. {source}"
        ),
        ToolMessage(
            content=(
                "BenQ Materials Corp. owns 14.82% of Visco Vision but has "
                "significant influence only; the investment uses the equity method "
                f"and does not confer control. {ownership['sources'][1]['url']}"
            ),
            tool_call_id="call-2",
            name="web_search",
        ),
    ]

    normalized = normalize_foreign_language_evidence(
        report, messages, ticker=regression["ticker"]
    )

    assert "Largest Shareholder: BenQ Materials Corp. (14.82%)" in normalized
    assert "Controlling Shareholder: NONE" in normalized
    assert "Control Status: NOT_CONTROLLED" in normalized
    assert "Control Basis: SIGNIFICANT_INFLUENCE_ONLY" in normalized
    assert "Related Listed Tickers: 8215.TW:significant_influence:14.82" in normalized
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert f"Ownership As Of: {ownership['captured_as_of']}" in normalized

    card = build_card(
        ticker=regression["ticker"],
        company_name="Visco Vision Inc.",
        merged_data={},
        senior_metrics={
            "listing_role": "LISTED_SUBSIDIARY",
            "related_listed_tickers": "2449.TW:parent:14.82",
        },
        fla_report=normalized,
    )
    assert card.largest_shareholder == {
        "name": "BenQ Materials Corp.",
        "pct": 14.82,
        "source": "fla_ownership",
    }
    assert card.control_status == "NOT_CONTROLLED"
    assert card.controlling_shareholder is None
    assert card.entity_role == "UNKNOWN"
    assert card.confidence == "conflict"
    assert card.related_listed == [
        {
            "ticker": "8215.TW",
            "relationship": "significant_influence",
            "pct": 14.82,
        }
    ]


def test_relationship_only_evidence_preserves_influence_without_inventing_stake():
    source = "https://issuer.example/financial-report.pdf"
    report = _report(
        holder="BenQ Materials Corp. (14.82%)",
        controller="NONE",
        status="NOT_CONTROLLED",
        basis="SIGNIFICANT_INFLUENCE_ONLY",
        relationship="significant influence",
        entity_role="STANDALONE",
        related="UNKNOWN",
        source_url=source,
    )

    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker="6782.TW",
        additional_records=[
            _record(
                "<result><url>https://noise.example/story</url>"
                "<summary>Unrelated market commentary.</summary></result>"
                f"<result><url>{source}</url>"
                "<summary>BenQ Materials Corp. is the entity with significant "
                "influence over the group.</summary></result>",
                name="search_foreign_sources",
                urls={source, "https://noise.example/story"},
            )
        ],
    )
    card = build_card(
        ticker="6782.TW",
        company_name="Visco Vision Inc.",
        merged_data={},
        senior_metrics={},
        fla_report=normalized,
    )

    assert "Largest Shareholder: UNKNOWN" in normalized
    assert "Influential Entity: BenQ Materials Corp." in normalized
    assert "Ownership Evidence Status: DISCLOSED_UNVERIFIED" in normalized
    assert card.largest_shareholder is None
    assert card.influential_entity["name"] == "BenQ Materials Corp."
    assert card.ownership_relationship == "SIGNIFICANT_INFLUENCE"
    assert card.control_status == "NOT_CONTROLLED"


def test_fundamentals_barrier_reconciles_legal_evidence_idempotently():
    source = "https://issuer.example/financial-report.pdf"
    raw_report = _report(
        holder="BenQ Materials Corp. (14.82%)",
        controller="NONE",
        status="NOT_CONTROLLED",
        basis="SIGNIFICANT_INFLUENCE_ONLY",
        relationship="significant influence",
        entity_role="STANDALONE",
        related="UNKNOWN",
        source_url=source,
    )
    initial = normalize_foreign_language_evidence(
        raw_report,
        [],
        ticker="6782.TW",
    )
    response = AIMessage(content=raw_report, name="foreign_language_analyst")
    record = EvidenceRecord(
        sequence=1,
        agent_key="legal_counsel",
        tool_name="search_foreign_sources",
        source="legal_counsel",
        content=(
            "BenQ Materials Corp. is the entity with significant influence "
            f"over the group. {source}"
        ),
        content_sha256="test",
        requested_urls=(),
        urls=(source,),
        blocked=False,
        findings=(),
        execution_status="SUCCEEDED",
        evidence_status="RESULTS_FOUND",
    )
    state = {
        "messages": [response],
        "company_of_interest": "6782.TW",
        "foreign_language_report": initial,
    }

    with patch(
        "src.runtime_services.get_current_evidence_records",
        return_value=[record],
    ):
        update = _reconcile_fundamentals_evidence(state)
        repeated = _reconcile_fundamentals_evidence({**state, **update})

    assert "DISCLOSED_UNVERIFIED" in update["foreign_language_report"]
    assert repeated == {}


def test_fundamentals_barrier_does_not_reprocess_failed_fla_message():
    raw_report = _report(source_url="https://issuer.example/report.pdf")
    state = {
        "messages": [AIMessage(content=raw_report, name="foreign_language_analyst")],
        "company_of_interest": "6782.TW",
        "foreign_language_report": raw_report,
        "artifact_statuses": {
            "foreign_language_report": {
                "complete": True,
                "ok": False,
                "content": raw_report,
            }
        },
    }

    assert _reconcile_fundamentals_evidence(state) == {}


def test_related_ticker_is_removed_when_it_does_not_appear_in_supporting_evidence():
    source = "https://example.com/shareholders"
    normalized = normalize_foreign_language_evidence(
        _report(source_url=source),
        [_tool(f"BenQ Materials Corp. owns 14.82% of Visco Vision. {source}")],
        ticker="6782.TW",
    )

    assert "Related Listed Tickers: UNKNOWN" in normalized
    assert "2449.TW" not in normalized


def test_sub_50_control_claim_needs_official_or_two_source_corroboration():
    source = "https://example.com/shareholders"
    normalized = normalize_foreign_language_evidence(
        _report(source_url=source, related="NONE"),
        [
            _tool(
                "BenQ Materials Corp. owns 14.82% and is described as a "
                f"consolidated subsidiary relationship. {source}"
            )
        ],
        ticker="6782.TW",
    )

    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: UNKNOWN" in normalized
    assert "Controlling Shareholder: UNKNOWN" in normalized
    assert "Parent Company: UNKNOWN" in normalized


def test_non_control_relationship_needs_supporting_evidence():
    source = "https://example.com/shareholders"
    normalized = normalize_foreign_language_evidence(
        _report(
            controller="NONE",
            status="NOT_CONTROLLED",
            basis="SIGNIFICANT_INFLUENCE_ONLY",
            relationship="equity method",
            related="NONE",
            source_url=source,
        ),
        [_tool(f"BenQ Materials Corp. owns 14.82% of Visco Vision. {source}")],
        ticker="6782.TW",
    )

    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: UNKNOWN" in normalized


def test_control_result_does_not_rewrite_entity_role():
    source = "https://example.com/shareholders"
    normalized = normalize_foreign_language_evidence(
        _report(
            controller="NONE",
            status="NOT_CONTROLLED",
            basis="SIGNIFICANT_INFLUENCE_ONLY",
            relationship="equity method",
            entity_role="LISTED_SUBSIDIARY",
            related="NONE",
            source_url=source,
        ),
        [
            _tool(
                "BenQ Materials Corp. owns 14.82% under the equity method "
                f"with significant influence but no control. {source}"
            )
        ],
        ticker="6782.TW",
    )

    assert "Control Status: NOT_CONTROLLED" in normalized
    assert "ENTITY_ROLE_OBSERVED: LISTED_SUBSIDIARY" in normalized


def test_two_urls_in_one_tool_message_are_not_two_source_corroboration():
    first = "https://one.example/shareholders"
    second = "https://two.example/profile"
    normalized = normalize_foreign_language_evidence(
        _report(source_url=first, related="NONE"),
        [
            _tool(
                "BenQ Materials Corp. owns 14.82%; consolidated subsidiary. "
                f"{first} {second}"
            )
        ],
        ticker="6782.TW",
    )

    assert "Control Status: UNKNOWN" in normalized


def test_two_distinct_tool_records_and_domains_can_corroborate_control():
    first = "https://one.example/shareholders"
    second = "https://two.example/profile"
    normalized = normalize_foreign_language_evidence(
        _report(source_url=first, related="NONE"),
        [
            _tool(
                f"BenQ Materials Corp. owns 14.82%; consolidated subsidiary. {first}"
            ),
            ToolMessage(
                content=(
                    "BenQ Materials Corp. owns 14.82%; consolidated subsidiary. "
                    f"{second}"
                ),
                tool_call_id="call-2",
                name="web_search",
            ),
        ],
        ticker="6782.TW",
    )

    assert "Control Status: CONTROLLED" in normalized


def test_official_filing_can_establish_sub_50_control_with_explicit_basis():
    source = "https://example.com/official-filing"
    normalized = normalize_foreign_language_evidence(
        _report(source_url=source, related="NONE"),
        [
            _tool(
                "### OFFICIAL FILING DATA\nBenQ Materials Corp. owns 14.82%; "
                f"the issuer is a consolidated subsidiary. {source}",
                name="get_official_filings",
            )
        ],
        ticker="6782.TW",
    )

    assert "Control Status: CONTROLLED" in normalized
    assert "Control Basis: CONSOLIDATED_SUBSIDIARY" in normalized


def test_unsupported_ownership_claim_is_cleared():
    normalized = normalize_foreign_language_evidence(
        _report(source_url="https://unsupported.example/claim"),
        [],
        ticker="6782.TW",
    )

    assert "Largest Shareholder: UNKNOWN" in normalized
    assert "Controlling Shareholder: UNKNOWN" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert "Related Listed Tickers: UNKNOWN" in normalized
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_not_found_ownership_stays_compact_and_is_not_rejected():
    report = """**OWNERSHIP STRUCTURE**
- Ownership Evidence Status: NOT_FOUND
- Ownership Source URL: N/A
- ENTITY_ROLE_OBSERVED: UNKNOWN
"""

    normalized = normalize_foreign_language_evidence(report, [], ticker="AAPL")

    assert "Ownership Evidence Status: NOT_FOUND" in normalized
    assert "Largest Shareholder:" not in normalized
    assert "CAPACITY_UTILIZATION" not in normalized


def test_controller_can_differ_from_largest_shareholder():
    source = "https://example.com/official-filing"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder="Passive Fund (40%)",
            controller="Founder Vehicle (10%)",
            status="CONTROLLED",
            basis="VOTING_AGREEMENT",
            relationship="subsidiary",
            related="NONE",
            source_url=source,
        ),
        [
            _tool(
                "Passive Fund owns 40%, while Founder Vehicle owns 10% and "
                f"controls voting rights through a voting agreement. {source}",
                name="get_official_filings",
            )
        ],
        ticker="TEST.T",
    )

    assert "Largest Shareholder: Passive Fund (40%)" in normalized
    assert "Controlling Shareholder: Founder Vehicle (10%)" in normalized


def test_exact_capacity_percentage_requires_matching_tool_evidence():
    source = "https://example.com/capacity"
    supported = normalize_foreign_language_evidence(
        _report(capacity="95%", capacity_url=source),
        [
            _tool(
                "BenQ Materials Corp. owns 14.82%. "
                "Visco Vision reported 95% capacity utilization. "
                f"https://www.viscovision.com.tw/tw/investors_shareholders.html {source}"
            )
        ],
        ticker="6782.TW",
    )
    unsupported = normalize_foreign_language_evidence(
        _report(capacity="95%", capacity_url=source),
        [],
        ticker="6782.TW",
    )

    assert "CAPACITY_UTILIZATION: 95%" in supported
    assert f"CAPACITY_UTILIZATION_SOURCE_URL: {source}" in supported
    assert "CAPACITY_UTILIZATION: N/A" in unsupported
    assert "CAPACITY_UTILIZATION_SOURCE_URL: N/A" in unsupported


def test_6782_broker_capacity_claim_is_preserved_as_secondary_not_primary():
    regression = load_frozen_regression("6782_TW_regression.json")
    evidence = regression["capacity_evidence"]
    supplemental = f"""<result>
<url>{evidence["source_url"]}</url>
<summary>{evidence["summary"]}</summary>
</result>"""

    normalized = normalize_foreign_language_evidence(
        _report(
            capacity=evidence["utilization"],
            capacity_url=evidence["source_url"],
        ),
        [],
        ticker=regression["ticker"],
        supplemental_evidence=supplemental,
    )

    assert "CAPACITY_UTILIZATION: 95%" in normalized
    assert "CAPACITY_EVIDENCE_STATUS: SECONDARY" in normalized
    assert "R_AND_D_CAPEX_BACKLOG_EVIDENCE: SECONDARY" in normalized
    assert "FACILITY_BUILDOUT_STATUS: N/A" in normalized


def test_latest_results_growth_is_computed_only_from_one_official_record():
    source = "https://www.twse.com.tw/results"

    normalized = normalize_foreign_language_evidence(
        _latest_results_report(source_url=source),
        [_tool(_latest_results_evidence(source), name="get_official_document")],
        ticker="TEST",
    )

    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: 50.0%" in normalized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: 102.5%" in normalized


def test_latest_results_accepts_post_inspection_ledger_record():
    source = "https://www.twse.com.tw/results"

    normalized = normalize_foreign_language_evidence(
        _latest_results_report(source_url=source),
        [],
        ticker="TEST",
        additional_records=[
            _record(
                _latest_results_evidence(source),
                name="get_official_document",
                urls={source},
            )
        ],
    )

    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: 102.5%" in normalized


@pytest.mark.parametrize(
    ("ticker", "fields", "source", "excerpt", "revenue_growth", "earnings_growth"),
    [
        (
            "IPN.PA",
            {
                "PERIOD": "H1 2026",
                "PRIOR_PERIOD": "H1 2025",
                "PERIOD_END": "2026-06-30",
                "PRIOR_PERIOD_END": "2025-06-30",
                "PERIOD_MONTHS": "6",
                "CURRENCY": "EUR",
                "REPORTING_UNIT": "€m",
                "REVENUE": "2190.2",
                "PRIOR_REVENUE": "1819.8",
                "EARNINGS": "402.6",
                "PRIOR_EARNINGS": "335.5",
                "EARNINGS_SCOPE": "IFRS Consolidated Net Profit",
            },
            "https://www.ipsen.com/press-release/ipsen-delivers-excellent-h1-2026-results-and-upgrades-its-full-year-guidance-3335743",
            """Extract of consolidated results
H1 2026
H1 2025
% change
€m
€m
Actual
CER
Total Sales
2 190.2
1 819.8
20.4 %
23.5 %
Core Operating Income
844.9
655.8
28.8 %
IFRS Consolidated Net Profit
402.6
335.5
20.0 %
The six-month periods ended 30 June 2026 and 30 June 2025.""",
            "20.4%",
            "20.0%",
        ),
        (
            "5478.TWO",
            {
                "PERIOD": "115年Q2",
                "PRIOR_PERIOD": "114年Q2",
                "PERIOD_END": "2026-06-30",
                "PRIOR_PERIOD_END": "2025-06-30",
                "PERIOD_MONTHS": "3",
                "CURRENCY": "新台幣",
                "REPORTING_UNIT": "仟元",
                "REVENUE": "1680297",
                "PRIOR_REVENUE": "1556424",
                "EARNINGS": "345215",
                "PRIOR_EARNINGS": "401316",
                "EARNINGS_SCOPE": "稅後淨利(歸屬本公司業主)",
            },
            "https://www.soft-world.com/News/NewsDetail?Sn=20394",
            """智冠科技（5478）115年上半年簡易合併損益比較表
單位：新台幣仟元
合併營收
115年Q2
115年Q1
QoQ
114年Q2
YoY
營業收入
1,680,297
1,935,421
-13%
1,556,424
7%
稅後淨利(歸屬本公司業主)
345,215
358,473
-3%
401,316
-13%""",
            "8.0%",
            "-14.0%",
        ),
    ],
)
def test_latest_results_inspected_comparative_table(
    ticker, fields, source, excerpt, revenue_growth, earnings_growth
):
    report = _latest_results_report(source_url=source)
    for field, value in fields.items():
        report = report.replace(
            f"LATEST_RESULTS_{field}: " + _field_for_test(report, field),
            f"LATEST_RESULTS_{field}: {value}",
        )
    evidence = f'DOCUMENT_METADATA: {{"source_url": "{source}"}}\n{excerpt}'
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker=ticker,
        additional_records=[
            replace(
                _record(evidence, name="get_official_document", urls={source}),
                authority="PRIMARY_ISSUER",
            )
        ],
    )
    if ticker == "5478.TWO":
        # The retained Q2/Q1/prior-Q2 table is real, but this issuer release
        # omits explicit quarter-end dates, so the claimed dates cannot promote.
        assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
        assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized
        return
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized
    assert f"LATEST_RESULTS_REVENUE_GROWTH_YOY: {revenue_growth}" in normalized
    assert f"LATEST_RESULTS_EARNINGS_GROWTH_YOY: {earnings_growth}" in normalized


def _field_for_test(report: str, field: str) -> str:
    return next(
        line.partition(": ")[2]
        for line in report.splitlines()
        if line.startswith(f"LATEST_RESULTS_{field}: ")
    )


def test_ledger_conversion_preserves_failure_status_and_normalizes_urls():
    record = EvidenceRecord(
        sequence=1,
        agent_key="foreign_language_analyst",
        tool_name="get_official_document",
        source="toolnode",
        content="A non-empty provider error body",
        content_sha256="test",
        requested_urls=("https://issuer.example/report/",),
        urls=("https://issuer.example/report/", "not-a-url"),
        blocked=False,
        findings=(),
        execution_status="FAILED",
        evidence_status="AUTH_ERROR",
        reason="FORBIDDEN",
    )

    converted = evidence_record_to_tool_evidence(record)

    assert converted.tool_name == "get_official_document"
    assert converted.content == "A non-empty provider error body"
    assert converted.urls == {"https://issuer.example/report"}
    assert converted.evidence_status == "AUTH_ERROR"
    assert converted.authority == "UNSUPPORTED"


def test_search_result_cannot_be_promoted_as_primary_latest_results():
    source = "https://issuer.example/results"

    normalized = normalize_foreign_language_evidence(
        _latest_results_report(source_url=source),
        [_tool(_latest_results_evidence(source))],
        ticker="TEST",
    )

    assert "LATEST_RESULTS_SOURCE_AUTHORITY: SECONDARY" in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: N/A" in normalized


def test_latest_results_rejects_mismatched_or_split_comparatives():
    source = "https://issuer.example/results"
    evidence = _latest_results_evidence(source)
    split = evidence.partition("Revenue")

    mismatched = normalize_foreign_language_evidence(
        _latest_results_report(source_url=source, prior_earnings="201"),
        [_tool(evidence, name="get_official_document")],
        ticker="TEST",
    )
    split_records = normalize_foreign_language_evidence(
        _latest_results_report(source_url=source),
        [
            _tool(split[0] + source, name="get_official_document"),
            ToolMessage(
                content=split[1] + split[2] + source,
                tool_call_id="call-2",
                name="get_official_document",
            ),
        ],
        ticker="TEST",
    )

    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in mismatched
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in split_records


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace("1,500\n1,000", "1,000\n1,500"),
        lambda text: text.replace("Revenue\n", "Other income\n"),
        lambda text: text.replace("March 31, 2025", "March 31, 2024"),
        lambda text: text.replace(
            "Reporting unit: thousands", "Reporting unit: millions"
        ),
        lambda text: text.replace(
            "Net income attributable to owners of parent",
            "Operating profit",
        ),
        lambda text: text.replace("Three months ended March 31, 2025\n", ""),
    ],
)
def test_latest_results_rejects_table_mutations(mutation):
    source = "https://www.twse.com.tw/results"
    normalized = normalize_foreign_language_evidence(
        _latest_results_report(source_url=source),
        [
            _tool(
                mutation(_latest_results_evidence(source)), name="get_official_document"
            )
        ],
        ticker="TEST",
    )

    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized


def test_latest_results_rejects_wrong_asserted_end_date():
    source = "https://www.twse.com.tw/results"
    report = _latest_results_report(source_url=source).replace(
        "LATEST_RESULTS_PERIOD_END: 2026-03-31",
        "LATEST_RESULTS_PERIOD_END: 2026-04-01",
    )

    normalized = normalize_foreign_language_evidence(
        report,
        [_tool(_latest_results_evidence(source), name="get_official_document")],
        ticker="TEST",
    )

    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized


def test_invalid_latest_citation_keeps_valid_guidance_artifact():
    guidance = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-07-14
SOURCE_URL: https://issuer.example/guidance
SEARCHES_COMPLETED: results_package=SUCCEEDED/RESULTS_FOUND; earnings_bridge=SUCCEEDED/RESULTS_FOUND
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
OPERATING_VS_NET_DIRECTION: UNKNOWN
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
DRIVER_TYPE: UNKNOWN
DRIVER_PERSISTENCE: N/A
EARNINGS_BASELINE_STATUS: MIXED
GUIDANCE_BRIDGE_STATUS: NOT_APPLICABLE
### --- END MANAGEMENT_GUIDANCE ---
"""
    report = guidance + _latest_results_report(source_url="N/A")

    normalized = normalize_foreign_language_evidence(report, [], ticker="9168.T")

    assert "COVERAGE_STATUS: FOUND" in normalized
    assert "LATEST_RESULTS_COVERAGE_STATUS: FOUND" in normalized
    assert "LATEST_RESULTS_REVENUE: N/A" in normalized
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
    assert validate_required_output("foreign_language_analyst", normalized)["ok"]
    promoted, _ = promote_foreign_growth_evidence("", normalized)
    assert "LATEST_RESULTS_REVENUE:" not in promoted
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY:" not in promoted


def test_final_latest_diagnostic_emits_once_after_complete_barrier():
    report = _latest_results_report(source_url="N/A")
    state = {
        "company_of_interest": "9168.T",
        "foreign_language_report": report,
        "messages": [AIMessage(content=report, name="foreign_language_analyst")],
        "artifact_statuses": {
            field: {"complete": True, "ok": True, "content": report}
            for field in (
                "foreign_language_report",
                "raw_fundamentals_data",
                "legal_report",
            )
        },
    }
    with (
        patch("src.runtime_services.get_current_evidence_records", return_value=[]),
        patch("src.graph.builder.logger.info") as log_info,
    ):
        _reconcile_fundamentals_evidence(state)

    finalized = [
        call
        for call in log_info.call_args_list
        if call.args == ("fla_latest_results_finalized",)
    ]
    assert len(finalized) == 1
    assert finalized[0].kwargs["reason_code"] == "INVALID_CITATION"


_NEXT_SECTION = (
    "\n### FOREIGN SOURCE FINDINGS FOR TEST\n\n**CONTEXT**\n- Country: Taiwan\n"
)
_CANONICAL_START = "### --- START LATEST_RESULTS ---"
_CANONICAL_END = "### --- END LATEST_RESULTS ---"


def _drifted(report: str, *, start: str, end: str | None) -> str:
    """Re-mark a canonical latest-results report the way models drifted (Oct 2026)."""
    lines = [line for line in report.splitlines() if line != _CANONICAL_END]
    lines[lines.index(_CANONICAL_START)] = start
    return "\n".join([*lines, *([end] if end else [])]) + "\n" + _NEXT_SECTION


def _normalized(report: str) -> str:
    return normalize_foreign_language_evidence(report, [], ticker="TEST")


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("### LATEST_RESULTS", None),  # bare heading, no END marker
        (_CANONICAL_START, "### END LATEST_RESULTS ---"),  # END missing its dashes
        ("### LATEST_RESULTS", "---"),  # closed by a horizontal rule
    ],
)
def test_drifted_latest_results_markers_are_reframed(start, end):
    normalized = _normalized(_drifted(_latest_results_report(), start=start, end=end))

    assert _has_valid_latest_results_block(normalized)
    assert "### FOREIGN SOURCE FINDINGS FOR TEST" in normalized
    block = extract_last_fenced_block(normalized, "LATEST_RESULTS")
    assert "FOREIGN SOURCE FINDINGS" not in block


def test_prose_inside_a_drifted_block_leaves_it_untouched():
    """Only an unbroken run of field lines is re-framed; prose means ambiguity."""
    report = _drifted(_latest_results_report(), start="### LATEST_RESULTS", end=None)
    report = report.replace(
        "LATEST_RESULTS_CURRENCY:",
        "Figures below are unaudited.\nLATEST_RESULTS_CURRENCY:",
    )

    normalized = _normalized(report)

    assert not _has_valid_latest_results_block(normalized)
    assert _CANONICAL_START not in normalized


def test_canonical_latest_results_block_is_not_reframed():
    report = _latest_results_report() + _NEXT_SECTION

    assert _reframe_latest_results_block(report) == report


def test_annotated_source_url_is_trimmed_to_the_url():
    report = _latest_results_report(
        source_url="https://dart.fss.or.kr/main.do?rcpNo=2026 (Half-Year Report 2026)"
    )

    normalized = _normalized(report)

    assert "LATEST_RESULTS_SOURCE_URL: https://dart.fss.or.kr/main.do?rcpNo=2026\n" in (
        normalized
    )
    assert _has_valid_latest_results_block(normalized)


def test_latest_results_ignores_stray_fields_outside_the_only_block():
    source = "https://www.twse.com.tw/results"
    report = (
        "LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND\n"
        "LATEST_RESULTS_SOURCE_URL: https://wrong.example (note)\n"
        + _latest_results_report(source_url=source)
    )
    normalized = normalize_foreign_language_evidence(
        report,
        [],
        ticker="TEST",
        additional_records=[
            _record(
                _latest_results_evidence(source),
                name="get_official_document",
                urls={source},
            )
        ],
    )

    assert _has_valid_latest_results_block(normalized)
    block = extract_last_fenced_block(normalized, "LATEST_RESULTS")
    assert block is not None
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in block
    promoted, _ = promote_foreign_growth_evidence("", normalized)
    assert "LATEST_RESULTS_COVERAGE_STATUS: FOUND" in promoted


def test_repeated_or_duplicate_latest_results_blocks_are_rejected():
    report = _latest_results_report()
    duplicate_field = report.replace(
        "LATEST_RESULTS_COVERAGE_STATUS: FOUND",
        "LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND\n"
        "LATEST_RESULTS_COVERAGE_STATUS: FOUND",
    )
    for ambiguous in (report + report, duplicate_field):
        normalized = _normalized(ambiguous)
        assert not _has_valid_latest_results_block(normalized)
        promoted, _ = promote_foreign_growth_evidence("", normalized)
        assert "LATEST_RESULTS_COVERAGE_STATUS" not in promoted


@pytest.mark.parametrize(
    ("source_url", "period_end"),
    [
        ("N/A", "2026-03-31"),  # FOUND without a source
        ("N/A (not published)", "2026-03-31"),  # an annotation is not a URL
        ("https://issuer.example/results", "UNKNOWN"),  # FOUND without a period
    ],
)
def test_found_without_source_or_period_cannot_promote_growth(source_url, period_end):
    report = _latest_results_report(source_url=source_url).replace(
        "LATEST_RESULTS_PERIOD_END: 2026-03-31",
        f"LATEST_RESULTS_PERIOD_END: {period_end}",
    )

    assert not _has_valid_latest_results_block(report)
    normalized = _normalized(report)
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized
    if period_end == "UNKNOWN":
        assert not _has_valid_latest_results_block(normalized)
    else:
        assert _has_valid_latest_results_block(normalized)
        assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized
        assert "LATEST_RESULTS_REVENUE: N/A" in normalized


def test_missing_required_field_in_a_drifted_block_stays_invalid():
    report = _drifted(_latest_results_report(), start="### LATEST_RESULTS", end=None)
    report = "\n".join(
        line
        for line in report.splitlines()
        if not line.startswith("LATEST_RESULTS_PRIOR_EARNINGS")
    )

    assert not _has_valid_latest_results_block(_normalized(report))
