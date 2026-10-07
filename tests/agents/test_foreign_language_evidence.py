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
from src.agents.management_guidance import normalize_management_guidance_output
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
    assert repeated == {"latest_results_reason": update["latest_results_reason"]}


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

    assert _reconcile_fundamentals_evidence(state) == {
        "latest_results_reason": "FLA_ARTIFACT_INVALID"
    }


def test_fundamentals_barrier_preserves_accepted_guidance_repair():
    preflight = "#### results_package\nSTATUS: COMPLETED\n#### earnings_bridge\nSTATUS: COMPLETED\n"
    initial_message = "Native research summary.\n" + _latest_results_report(
        source_url="N/A"
    )
    repaired_report = normalize_management_guidance_output(
        initial_message
        + "\n### --- START MANAGEMENT_GUIDANCE ---\n"
        + "COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH\n"
        + "### --- END MANAGEMENT_GUIDANCE ---\n",
        preflight,
    )
    state = {
        "company_of_interest": "TEST.T",
        "messages": [
            AIMessage(content=initial_message, name="foreign_language_analyst")
        ],
        "foreign_language_report": repaired_report,
        "management_guidance_evidence": preflight,
        "guidance_normalization": {
            "initial_reason": "MISSING_BLOCK",
            "final_action": "REPAIRED",
        },
        "structural_recovery_events": [
            {
                "originating_agent": "foreign_language_analyst",
                "outcome": "accepted_text",
            }
        ],
        "artifact_statuses": {
            "foreign_language_report": {
                "complete": True,
                "ok": True,
                "content": repaired_report,
            }
        },
    }

    with patch("src.runtime_services.get_current_evidence_records", return_value=[]):
        update = _reconcile_fundamentals_evidence(state)

    final_report = update.get("foreign_language_report", repaired_report)
    assert "COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH" in final_report
    assert "Targeted evidence did not resolve forward guidance" not in final_report


def test_fundamentals_barrier_preserves_accepted_latest_results_repair():
    initial_message = (
        "Native research summary.\n"
        "### --- START LATEST_RESULTS ---\nBROKEN\n"
        "### --- END LATEST_RESULTS ---\n"
    )
    accepted_report = initial_message + "\n" + _latest_results_report()
    state = {
        "company_of_interest": "TEST.T",
        "messages": [
            AIMessage(content=initial_message, name="foreign_language_analyst")
        ],
        "foreign_language_report": accepted_report,
        "structural_recovery_events": [
            {
                "originating_agent": "foreign_language_analyst",
                "outcome": "accepted_text",
            }
        ],
        "artifact_statuses": {
            "foreign_language_report": {
                "complete": True,
                "ok": True,
                "content": accepted_report,
            }
        },
    }

    with patch("src.runtime_services.get_current_evidence_records", return_value=[]):
        update = _reconcile_fundamentals_evidence(state)

    final_report = update.get("foreign_language_report", accepted_report)
    assert final_report.count("### --- START LATEST_RESULTS ---") == 1
    assert "LATEST_RESULTS_PERIOD_END: 2026-03-31" in final_report


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
                authority_by_url=((source, "PRIMARY_ISSUER"),),
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
        update = _reconcile_fundamentals_evidence(state)

    finalized = [
        call
        for call in log_info.call_args_list
        if call.args == ("fla_latest_results_finalized",)
    ]
    assert len(finalized) == 1
    assert finalized[0].kwargs["reason_code"] == "INVALID_CITATION"
    assert update["latest_results_reason"] == "INVALID_CITATION"


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


@pytest.mark.parametrize("primary", [False, True])
def test_control_corroboration_binds_primary_authority_to_ownership_citation(primary):
    official = "https://www.twse.com.tw/ownership"
    secondary = "https://secondary.example/ownership"
    cited = official if primary else secondary
    record = make_tool_evidence_record(
        tool_name="get_official_document",
        content="STATUS: EVIDENCE_FOUND\nBenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary. 2026-03-28",
        urls=(official, secondary),
    )
    normalized = normalize_foreign_language_evidence(
        _report(source_url=cited), [], ticker="TEST", additional_records=[record]
    )
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert f"Control Status: {'CONTROLLED' if primary else 'UNKNOWN'}" in normalized


@pytest.mark.parametrize("primary", [False, True])
def test_capacity_authority_binds_to_explicit_citation(primary):
    official = "https://www.twse.com.tw/capacity"
    secondary = "https://secondary.example/capacity"
    cited = official if primary else secondary
    report = f"CAPACITY_UTILIZATION: 85%\nCAPACITY_UTILIZATION_SOURCE_URL: {cited}\nCAPACITY_UTILIZATION_AS_OF: 2026-03-28\n"
    record = make_tool_evidence_record(
        tool_name="get_official_document",
        content="STATUS: EVIDENCE_FOUND\nCapacity utilization was 85% on 2026-03-28.",
        urls=(official, secondary),
    )
    normalized = normalize_foreign_language_evidence(
        report, [], ticker="TEST", additional_records=[record]
    )
    status = "PRIMARY" if primary else "SECONDARY"
    assert f"CAPACITY_EVIDENCE_STATUS: {status}" in normalized


def test_source_envelope_split_does_not_inherit_another_sources_primary_authority():
    from src.agents.foreign_language_evidence import _split_search_result_records

    official = "https://www.twse.com.tw/results"
    secondary = "https://secondary.example/results"
    record = make_tool_evidence_record(
        tool_name="extract_guidance_sources",
        content=f"STATUS: EVIDENCE_FOUND\n<result><url>{official}</url><content>Issuer filing</content></result><result><url>{secondary}</url><content>Secondary article</content></result>",
        urls=(official, secondary),
    )
    assert record.authority == "PRIMARY_REGISTRY"
    split = _split_search_result_records([record])
    assert [row.authority for row in split] == ["PRIMARY_REGISTRY", "SECONDARY"]
    assert dict(split[1].authority_by_url) == {secondary: "SECONDARY"}


@pytest.mark.parametrize(
    ("execution", "status", "usable", "primary"),
    [
        ("SUCCEEDED", "EVIDENCE_FOUND", True, True),
        ("SUCCEEDED", "RESULTS_FOUND", True, False),
        ("FAILED", "EVIDENCE_FOUND", False, False),
        ("BLOCKED", "EVIDENCE_FOUND", False, False),
        ("SKIPPED", "EVIDENCE_FOUND", False, False),
        ("SUCCEEDED", "UNAVAILABLE", False, False),
        ("SUCCEEDED", "AUTH_ERROR", False, False),
        ("SUCCEEDED", "INSUFFICIENT", False, False),
        ("SUCCEEDED", "NO_RESULTS", False, False),
        ("SUCCEEDED", "COVERAGE_COMPLETE_NO_MATCH", False, False),
    ],
)
@pytest.mark.parametrize("pct", ["14.82", "60"])
def test_saved_evidence_status_controls_ownership_capacity_and_downstream_consumers(
    execution, status, usable, primary, pct
):
    """Retained claim text in a failed response must never become economic evidence."""
    import hashlib
    import json

    source = "https://www.twse.com.tw/combined"
    content = (
        "STATUS: EVIDENCE_FOUND\n"
        f"BenQ Materials Corp. owns {pct}% of shares; this entity is a consolidated subsidiary. "
        "Capacity utilization was 85% in 2026-Q1. 2026-03-28"
    )
    saved = json.loads(
        json.dumps(
            {
                "tool_name": "get_official_document",
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "urls": [source],
                "authority_by_url": [[source, "PRIMARY_REGISTRY"]],
                "execution_status": execution,
                "evidence_status": status,
                "blocked": execution == "BLOCKED",
            }
        )
    )
    record = evidence_record_to_tool_evidence(saved)
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=f"BenQ Materials Corp. ({pct}%)",
            controller=f"BenQ Materials Corp. ({pct}%)",
            source_url=source,
            capacity="85%",
            capacity_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    controlled = usable and (primary or pct == "60")
    expected_control = "CONTROLLED" if controlled else "UNKNOWN"
    assert f"Control Status: {expected_control}" in normalized
    assert (
        f"Largest Shareholder: BenQ Materials Corp. ({pct}%)"
        if usable
        else "Largest Shareholder: UNKNOWN"
    ) in normalized
    capacity_status = "PRIMARY" if primary else "SECONDARY" if usable else "UNSUPPORTED"
    assert f"CAPACITY_EVIDENCE_STATUS: {capacity_status}" in normalized
    senior, _ = promote_foreign_growth_evidence("GROWTH_SCORE: 7", normalized)
    assert f"CAPACITY_EVIDENCE_STATUS: {capacity_status}" in senior
    assert f"CAPACITY_UTILIZATION: {'85%' if usable else 'N/A'}" in senior
    assert "GROWTH_SCORE: 7" in senior
    card = build_card(
        ticker="TEST",
        company_name="Test",
        merged_data={},
        senior_metrics={},
        fla_report=normalized,
    )
    assert card.control_status == expected_control
    assert (card.controlling_shareholder is not None) is controlled


@pytest.mark.parametrize("reverse", [False, True])
def test_search_envelopes_bind_claims_to_their_own_sources_through_normalization(
    reverse,
):
    """A primary hit with no claim cannot promote a different secondary hit."""
    official = "https://www.twse.com.tw/irrelevant"
    secondary = "https://secondary.example/combined"
    blocks = [
        f"<result><url>{official}</url><content>Unrelated issuer filing.</content></result>",
        f"<result><url>{secondary}</url><content>BenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary. Capacity utilization was 85% in 2026-Q1. 2026-03-28</content></result>",
    ]
    record = make_tool_evidence_record(
        tool_name="extract_guidance_sources",
        content="STATUS: EVIDENCE_FOUND\n"
        + "".join(reversed(blocks) if reverse else blocks),
        urls=(official, secondary),
    )
    normalized = normalize_foreign_language_evidence(
        _report(source_url=secondary, capacity="85%", capacity_url=secondary),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert "CAPACITY_EVIDENCE_STATUS: SECONDARY" in normalized


def test_unrelated_control_record_cannot_supply_a_cited_sources_control_basis():
    cited = "https://www.twse.com.tw/ownership"
    records = [
        _record(
            "STATUS: EVIDENCE_FOUND\nBenQ Materials Corp. owns 14.82% of shares.",
            name="get_official_document",
            urls=(cited,),
        ),
        _record(
            "STATUS: EVIDENCE_FOUND\nBenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary.",
            name="get_official_document",
            urls=("https://secondary.example/ownership",),
        ),
    ]
    normalized = normalize_foreign_language_evidence(
        _report(source_url=cited),
        [],
        ticker="TEST",
        additional_records=records,
    )
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: UNKNOWN" in normalized


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        '{"error": "unauthorized"}',
        '{"error": "provider_failure"}',
        "STATUS: UNAVAILABLE",
        "STATUS: AUTH_ERROR",
        "STATUS: NO_RESULTS",
        "STATUS: COVERAGE_COMPLETE_NO_MATCH",
    ],
)
def test_partial_provider_error_cannot_become_primary_but_valid_secondary_hit_survives(
    reverse, error
):
    official = "https://www.twse.com.tw/failed"
    secondary = "https://secondary.example/combined"
    claim = "BenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary. Capacity utilization was 85% in 2026-Q1. 2026-03-28"
    blocks = [
        f"<result><url>{official}</url><content>\n{error}\n{claim}</content></result>",
        f"<result><url>{secondary}</url><content>{claim}</content></result>",
    ]
    record = make_tool_evidence_record(
        tool_name="extract_guidance_sources",
        content="STATUS: EVIDENCE_FOUND\n"
        + "".join(reversed(blocks) if reverse else blocks),
        urls=(official, secondary),
    )
    for cited, usable in ((official, False), (secondary, True)):
        normalized = normalize_foreign_language_evidence(
            _report(source_url=cited, capacity="85%", capacity_url=cited),
            [],
            ticker="TEST",
            additional_records=[record],
        )
        assert "Control Status: UNKNOWN" in normalized
        assert (
            f"CAPACITY_EVIDENCE_STATUS: {'SECONDARY' if usable else 'UNSUPPORTED'}"
            in normalized
        )
        assert f"CAPACITY_UTILIZATION: {'85%' if usable else 'N/A'}" in normalized


@pytest.mark.asyncio
async def test_live_and_serialized_issuer_authority_reaches_governance_and_growth_after_teardown():
    import json

    from src.runtime_services import RuntimeServices, use_runtime_services
    from src.tooling.evidence_recorder import EvidenceRecorder
    from src.tooling.inspection_service import INSPECTION_SERVICE
    from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult

    source = "https://issuer.example/combined"
    content = f'STATUS: EVIDENCE_FOUND\nDOCUMENT_METADATA: {{"source_url": "{source}"}}\nBenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary. Capacity utilization was 85% in 2026-Q1. 2026-03-28'
    recorder = EvidenceRecorder()
    services = RuntimeServices(
        ToolExecutionService([recorder]),
        INSPECTION_SERVICE,
        evidence_recorder=recorder,
    )
    with use_runtime_services(services):
        services.issuer_authority.register_url(
            source, provenance="structured_company_profile"
        )
        await recorder.after(
            ToolInvocation("get_official_document", {"url": source}, "preflight"),
            ToolResult(content),
        )
        live = evidence_record_to_tool_evidence(recorder.snapshot()[0])
        saved = json.loads(json.dumps(recorder.serialized_snapshot()[0]))
    for record, primary in (
        (live, True),
        (evidence_record_to_tool_evidence(saved), True),
        (evidence_record_to_tool_evidence({**saved, "content_sha256": "stale"}), False),
    ):
        normalized = normalize_foreign_language_evidence(
            _report(source_url=source, capacity="85%", capacity_url=source),
            [],
            ticker="TEST",
            additional_records=[record],
        )
        card = build_card(
            ticker="TEST",
            company_name="Test",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        )
        assert card.control_status == ("CONTROLLED" if primary else "UNKNOWN")
        senior, _ = promote_foreign_growth_evidence("GROWTH_SCORE: 7", normalized)
        assert (
            f"CAPACITY_EVIDENCE_STATUS: {'PRIMARY' if primary else 'SECONDARY'}"
            in senior
        )
        assert "CAPACITY_UTILIZATION: 85%" in senior


@pytest.mark.parametrize(
    "outer,inner",
    [("EVIDENCE_FOUND", "RESULTS_FOUND"), ("RESULTS_FOUND", "EVIDENCE_FOUND")],
)
def test_result_envelope_cannot_promote_evidence_above_either_declared_status(
    outer, inner
):
    source = "https://www.twse.com.tw/combined"
    record = make_tool_evidence_record(
        tool_name="extract_guidance_sources",
        content=f"STATUS: {outer}\n<result><url>{source}</url><content>\nSTATUS: {inner}\nBenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary. Capacity utilization was 85% in 2026-Q1. 2026-03-28</content></result>",
        urls=(source,),
    )
    normalized = normalize_foreign_language_evidence(
        _report(source_url=source, capacity="85%", capacity_url=source),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert "CAPACITY_EVIDENCE_STATUS: SECONDARY" in normalized


@pytest.mark.parametrize("primary", [False, True])
def test_source_link_in_result_body_cannot_replace_the_result_source(primary):
    official = "https://www.twse.com.tw/reference"
    secondary = "https://secondary.example/ownership"
    source = official if primary else secondary
    record = make_tool_evidence_record(
        tool_name="extract_guidance_sources",
        content=f"STATUS: EVIDENCE_FOUND\n<result><url>{source}</url><content>BenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary. Reference: {official}</content></result>",
        urls=(source, official),
    )
    normalized = normalize_foreign_language_evidence(
        _report(source_url="N/A"),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    assert f"Control Status: {'CONTROLLED' if primary else 'UNKNOWN'}" in normalized


@pytest.mark.parametrize(
    "status",
    ["EVIDENCE_FOUND", "RESULTS_FOUND", "UNAVAILABLE", "AUTH_ERROR", "NO_RESULTS"],
)
def test_preflight_capacity_fallback_obeys_error_status(status):
    source = "https://www.twse.com.tw/capacity"
    supplemental = f"STATUS: {status}\n<result><url>{source}</url><content>Capacity utilization was 85% in 2026-Q1.</content></result>"
    normalized = normalize_foreign_language_evidence(
        _report(capacity="85%", capacity_url=source),
        [],
        ticker="TEST",
        supplemental_evidence=supplemental,
    )
    usable = status in {"EVIDENCE_FOUND", "RESULTS_FOUND"}
    assert f"CAPACITY_UTILIZATION: {'85%' if usable else 'N/A'}" in normalized
    assert (
        f"CAPACITY_EVIDENCE_STATUS: {'SECONDARY' if usable else 'UNSUPPORTED'}"
        in normalized
    )


@pytest.mark.parametrize("distinct", [False, True])
@pytest.mark.parametrize("status", ["EVIDENCE_FOUND", "RESULTS_FOUND", "AUTH_ERROR"])
def test_secondary_control_corroboration_requires_two_usable_independent_domains(
    distinct, status
):
    source = "https://secondary.example/ownership"
    other = "https://other.example/ownership" if distinct else source
    claim = "BenQ Materials Corp. owns 14.82% of shares; this entity is a consolidated subsidiary."
    records = [
        _record(
            "STATUS: RESULTS_FOUND\n" + claim,
            name="extract_guidance_sources",
            urls=(source,),
        ),
        _record(
            f"STATUS: {status}\n" + claim,
            name="extract_guidance_sources",
            urls=(other,),
        ),
    ]
    normalized = normalize_foreign_language_evidence(
        _report(source_url=source),
        [],
        ticker="TEST",
        additional_records=records,
    )
    controlled = distinct and status != "AUTH_ERROR"
    assert f"Control Status: {'CONTROLLED' if controlled else 'UNKNOWN'}" in normalized


@pytest.mark.parametrize("padding", [0, 20_001])
def test_capacity_preflight_bounds_each_source_after_splitting_combined_results(
    padding,
):
    source = "https://secondary.example/capacity"
    supplemental = (
        "STATUS: RESULTS_FOUND\n"
        + "<result><url>https://other.example/irrelevant</url><content>"
        + "x" * padding
        + "</content></result>"
        + f"<result><url>{source}</url><content>Capacity utilization was 85% in 2026-Q1.</content></result>"
    )
    normalized = normalize_foreign_language_evidence(
        _report(capacity="85%", capacity_url=source),
        [],
        ticker="TEST",
        supplemental_evidence=supplemental,
    )
    assert "CAPACITY_UTILIZATION: 85%" in normalized
    assert "CAPACITY_EVIDENCE_STATUS: SECONDARY" in normalized


@pytest.mark.parametrize(
    "supplemental",
    [
        "",
        "<result><content>Capacity utilization was 85%.</content></result>",
        "<result><url>javascript:alert(1)</url><content>Capacity utilization was 85%.</content></result>",
        "<result><url>https://secondary.example/capacity</url><content>Capacity utilization was 85%.",
    ],
)
def test_capacity_preflight_missing_or_malformed_source_cannot_promote(supplemental):
    normalized = normalize_foreign_language_evidence(
        _report(capacity="85%", capacity_url="https://secondary.example/capacity"),
        [],
        ticker="TEST",
        supplemental_evidence=supplemental,
    )
    assert "CAPACITY_UTILIZATION: N/A" in normalized
    assert "CAPACITY_EVIDENCE_STATUS: UNSUPPORTED" in normalized


@pytest.mark.parametrize("percentage", [50.01, 69.72])
@pytest.mark.parametrize(
    "unrelated",
    [
        "independent non-executive directors oversee the board",
        "獨立非執行董事 oversee the board",
        "independent auditors review the accounts",
        "minority shareholders have no control",
    ],
)
def test_verified_majority_precedes_unrelated_independence(percentage, unrelated):
    source = "https://issuer.example/ownership"
    content = f"Lu Ruibo owns {percentage}% of Company shares, while {unrelated}."
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=f"Lu Ruibo ({percentage}%)",
            controller="NONE",
            status="NOT_CONTROLLED",
            relationship="independent",
            source_url=source,
        ),
        [],
        ticker="1286.HK",
        additional_records=[_record(content, urls=(source,))],
    )
    assert "Control Status: CONTROLLED" in normalized
    assert "Control Basis: MAJORITY_VOTING_RIGHTS" in normalized
    assert f"Controlling Shareholder: Lu Ruibo ({percentage:g}%)" in normalized
    assert "Relationship: UNKNOWN" in normalized
    card = build_card(
        ticker="1286.HK",
        company_name="TEST",
        merged_data={},
        senior_metrics={},
        fla_report=normalized,
    )
    assert card.control_status == "CONTROLLED"


@pytest.mark.parametrize(
    "holder", ["Lu Ruibo", "Foo and Bar Holdings", "BenQ Materials Corp."]
)
@pytest.mark.parametrize(
    "contrary",
    [
        "but has no control of the Company",
        ", and his voting rights are capped at 30%",
        "and uses the equity method for its investment",
        ", independent directors oversee the board, but is not a controlling shareholder",
    ],
)
def test_majority_conflicts_require_actual_holder_bound_source(holder, contrary):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=f"{holder} (69.72%)",
            controller="NONE",
            relationship="independent",
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[
            _record(
                f"{holder} owns 69.72% of Company shares {contrary}.", urls=(source,)
            )
        ],
    )
    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: CONFLICTING_CONTROL_EVIDENCE" in normalized


@pytest.mark.parametrize(
    "pct,content,relationship,expected,basis",
    [
        (
            30,
            "Lu Ruibo owns 30% of Company shares and is independent of the Company.",
            "independent",
            "NOT_CONTROLLED",
            "NONE",
        ),
        (
            25,
            "Lu Ruibo owns 25% of Company shares with significant influence under the equity method.",
            "equity method",
            "NOT_CONTROLLED",
            "SIGNIFICANT_INFLUENCE_ONLY",
        ),
        (
            20,
            "Lu Ruibo owns 20% of Company shares. The Company has an independent business.",
            "independent",
            "UNKNOWN",
            "UNKNOWN",
        ),
        (
            20,
            "Lu Ruibo owns 20% of Company shares; independent third parties advise the Company.",
            "independent",
            "UNKNOWN",
            "UNKNOWN",
        ),
        (
            50,
            "Lu Ruibo owns 50% of Company shares.",
            "independent",
            "UNKNOWN",
            "UNKNOWN",
        ),
        (
            69.72,
            "Lu Ruibo owns 69.72% of Company shares.",
            "equity method",
            "CONTROLLED",
            "MAJORITY_VOTING_RIGHTS",
        ),
    ],
)
def test_control_relationship_boundaries(pct, content, relationship, expected, basis):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=f"Lu Ruibo ({pct}%)",
            controller="NONE",
            relationship=relationship,
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[_record(content, urls=(source,))],
    )
    assert f"Control Status: {expected}" in normalized
    assert f"Control Basis: {basis}" in normalized


@pytest.mark.parametrize(
    "case", ["failed", "blocked", "wrong_url", "wrong_holder", "unverified"]
)
def test_unverified_majority_cannot_authorize_control(case):
    source = "https://issuer.example/ownership"
    content = (
        "Lu Ruibo owns 69.72% of Company shares and is independent of the Company."
    )
    record = _record(content, urls=(source,))
    if case in {"failed", "blocked"}:
        record = replace(record, evidence_status="UNAVAILABLE")
    elif case == "wrong_url":
        record = _record(content, urls=("https://other.example/ownership",))
    elif case == "wrong_holder":
        record = _record(content.replace("Lu Ruibo", "Other Person"), urls=(source,))
    else:
        record = _record("Lu Ruibo is independent of the Company.", urls=(source,))
    normalized = normalize_foreign_language_evidence(
        _report(
            holder="Lu Ruibo (69.72%)",
            controller="NONE",
            relationship="independent",
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    assert "Control Status: CONTROLLED" not in normalized


@pytest.mark.parametrize(
    "tail",
    [
        "and minority shareholders have no control",
        "while minority shareholders have no control",
        "and independent non-executive directors oversee the board",
    ],
)
def test_other_subject_cannot_create_majority_conflict(tail):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder="Lu Ruibo (69.72%)",
            controller="NONE",
            relationship="independent",
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[
            _record(f"Lu Ruibo owns 69.72% of Company shares, {tail}.", urls=(source,))
        ],
    )
    assert "Control Status: CONTROLLED" in normalized
    assert "Control Basis: MAJORITY_VOTING_RIGHTS" in normalized


@pytest.mark.parametrize(
    "rights", ["non-voting shares", "shares with no voting rights"]
)
def test_majority_percentage_with_explicit_nonvoting_rights_is_conflicting(rights):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder="Lu Ruibo (69.72%)",
            controller="NONE",
            relationship="independent",
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[
            _record(f"Lu Ruibo owns 69.72% of the Company's {rights}.", urls=(source,))
        ],
    )
    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: CONFLICTING_CONTROL_EVIDENCE" in normalized


@pytest.mark.parametrize("failed,blocked", [(True, False), (False, True)])
def test_failed_or_blocked_raw_ownership_record_cannot_verify_majority(failed, blocked):
    import hashlib

    source = "https://issuer.example/ownership"
    content = "Lu Ruibo owns 69.72% of Company shares."
    record = evidence_record_to_tool_evidence(
        {
            "tool_name": "get_official_document",
            "content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "urls": [source],
            "authority_by_url": [[source, "PRIMARY_ISSUER"]],
            "evidence_status": "EVIDENCE_FOUND",
            "execution_status": "FAILED" if failed else "SUCCEEDED",
            "blocked": blocked,
        }
    )
    normalized = normalize_foreign_language_evidence(
        _report(holder="Lu Ruibo (69.72%)", controller="NONE", source_url=source),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    assert "Control Status: UNKNOWN" in normalized
    assert "Ownership Evidence Status: REJECTED" in normalized


@pytest.mark.parametrize("holder", ["", "UNKNOWN", "N/A"])
def test_absent_holder_never_scans_or_binds_relationship_sentences(holder):
    from src.agents.foreign_language_evidence import _holder_sentences

    with patch(
        "src.agents.foreign_language_evidence.re.finditer",
        side_effect=AssertionError("unknown holder must stop before scanning"),
    ):
        assert (
            _holder_sentences(
                "Independent directors oversee the Company. " * 1000, holder
            )
            == []
        )


def test_captured_teck_class_votes_do_not_authorize_issuer_control():
    frozen = load_frozen_regression("teck_share_class_ownership.json")
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=frozen["holder"],
            controller="UNKNOWN",
            status="UNKNOWN",
            basis="UNKNOWN",
            relationship="independent",
            source_url=frozen["source_url"],
        ),
        [],
        ticker=frozen["ticker"],
        additional_records=[_record(frozen["content"], urls=(frozen["source_url"],))],
    )
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: UNKNOWN" in normalized
    assert "Controlling Shareholder: UNKNOWN" in normalized
    assert (
        build_card(
            ticker=frozen["ticker"],
            company_name="Teck",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        ).control_status
        == "UNKNOWN"
    )


@pytest.mark.parametrize(
    "content,expected,basis",
    [
        ("Holder owns 60% of Class A.", "UNKNOWN", "UNKNOWN"),
        ("Holder owns 60% of Class II.", "UNKNOWN", "UNKNOWN"),
        ("Holder owns 60% of Class 1.", "UNKNOWN", "UNKNOWN"),
        (
            "Holder owns 60% of Company shares in a world-class business facing a class action.",
            "CONTROLLED",
            "MAJORITY_VOTING_RIGHTS",
        ),
        (
            "Holder owns 60% of Class A shares.",
            "UNKNOWN",
            "UNKNOWN",
        ),
        (
            "| Shareholder | Class A Votes (%) | Aggregate Votes (%) |\n"
            "| --- | --- | --- |\n| Holder | 60% | 34% |",
            "UNKNOWN",
            "UNKNOWN",
        ),
        (
            "Holder owns 60% of Class A shares and holds 60% of total voting rights.",
            "CONTROLLED",
            "MAJORITY_VOTING_RIGHTS",
        ),
        (
            "Holder owns 60% of Class A shares. Holder holds 60% of total voting rights.",
            "CONTROLLED",
            "MAJORITY_VOTING_RIGHTS",
        ),
        (
            "Holder owns 60% of Class A shares. Other Holder holds 60% of total voting rights.",
            "UNKNOWN",
            "UNKNOWN",
        ),
        (
            "Holder owns 60% of Company shares. Other Fund owns Class A shares.",
            "CONTROLLED",
            "MAJORITY_VOTING_RIGHTS",
        ),
        (
            "| Shareholder | Class A Votes (%) |\n| Other Fund | 80% |\n\n"
            "Holder owns 60% of Company shares.",
            "CONTROLLED",
            "MAJORITY_VOTING_RIGHTS",
        ),
        (
            "Holder owns 60% of Class A shares and has contractual control of Company.",
            "CONTROLLED",
            "CONTRACTUAL_RIGHTS",
        ),
        (
            "Holder owns 60% of Class A shares. Other Fund has contractual control of Company.",
            "UNKNOWN",
            "UNKNOWN",
        ),
    ],
)
def test_majority_share_class_scope_is_bound_to_ownership_context(
    content, expected, basis
):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder="Holder (60%)",
            controller="Holder (60%)",
            basis="CONTRACTUAL_RIGHTS",
            source_url=source,
        ),
        [_tool(content + f"\n{source}", name="get_official_filings")],
        ticker="TEST",
    )
    assert f"Control Status: {expected}" in normalized
    assert f"Control Basis: {basis}" in normalized


@pytest.mark.parametrize(
    "holder,statement",
    [
        (
            "Lu Ruibo",
            "Mr. Lu Ruibo and his associates are interested in {pct}% of the issued shares.",
        ),
        ("Lu Ruibo", "Lu Ruibo and her associates hold {pct}% of issued shares."),
        ("Holder", "Holder and its associates own {pct}% of issued shares."),
        ("Holder", "Holder and their associates own {pct}% of issued shares."),
        (
            "Holder",
            "Holder together with its associates holds {pct}% of issued shares.",
        ),
        (
            "Holder",
            "Holder, together with its associates, holds {pct}% of issued shares.",
        ),
        ("Capital Associates", "Capital Associates owns {pct}% of issued shares."),
        (
            "Lu Ruibo",
            "Lu Ruibo holds {pct}%. Lu Ruibo is not an associate of any director.",
        ),
        ("Lu Ruibo", "Lu Ruibo holds {pct}%. Lu Ruibo does not use the equity method."),
        (
            "Lu Ruibo",
            "Lu Ruibo holds {pct}%. Lu Ruibo does not have significant influence.",
        ),
        (
            "Lu Ruibo",
            "Lu Ruibo holds {pct}%. Lu Ruibo is not accounted for as an associate.",
        ),
        (
            "Lu Ruibo",
            "Lu Ruibo holds {pct}%. Lu Ruibo discusses significant influence in the annual report.",
        ),
        ("Lu Ruibo", "Lu Ruibo holds {pct}%, while Other Fund uses the equity method."),
    ],
)
@pytest.mark.parametrize("pct", [25, 69.72])
@pytest.mark.parametrize(
    "relationship", ["associate", "equity method", "significant influence"]
)
def test_coholders_names_and_negated_accounting_do_not_establish_non_control(
    holder, statement, pct, relationship
):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=f"{holder} ({pct}%)", relationship=relationship, source_url=source
        ),
        [],
        ticker="1286.HK",
        additional_records=[_record(statement.format(pct=pct), urls=(source,))],
    )
    expected = "CONTROLLED" if pct > 50 else "UNKNOWN"
    assert f"Control Status: {expected}" in normalized
    assert "CONFLICTING_CONTROL_EVIDENCE" not in normalized
    card = build_card(
        ticker="1286.HK",
        company_name="TEST",
        merged_data={},
        senior_metrics={},
        fla_report=normalized,
    )
    assert card.control_status == expected


@pytest.mark.parametrize("pct", [25, 69.72])
@pytest.mark.parametrize(
    "accounting",
    [
        "uses the equity method for its investment",
        "has significant influence only",
        "is accounted for as an associate",
        "is classified as an associate",
        "is an associate of the Company",
    ],
)
def test_explicit_accounting_relationship_remains_authoritative(pct, accounting):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(holder=f"Holder ({pct}%)", relationship="associate", source_url=source),
        [],
        ticker="TEST",
        additional_records=[
            _record(f"Holder owns {pct}% and {accounting}.", urls=(source,))
        ],
    )
    assert (
        f"Control Status: {'UNKNOWN' if pct > 50 else 'NOT_CONTROLLED'}" in normalized
    )
    assert (
        f"Control Basis: {'CONFLICTING_CONTROL_EVIDENCE' if pct > 50 else 'SIGNIFICANT_INFLUENCE_ONLY'}"
        in normalized
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "and its associates",
        "is not an associate",
        "does not use the equity method",
        "has significant influence only",
    ],
)
@pytest.mark.parametrize("failed,blocked", [(True, False), (False, True)])
def test_relationship_words_cannot_revive_failed_or_blocked_evidence(
    phrase, failed, blocked
):
    source = "https://issuer.example/ownership"
    record = evidence_record_to_tool_evidence(
        {
            "tool_name": "get_official_document",
            "content": f"Holder owns 69.72% and {phrase}.",
            "urls": [source],
            "execution_status": "FAILED" if failed else "SUCCEEDED",
            "blocked": blocked,
            "evidence_status": "EVIDENCE_FOUND",
        }
    )
    normalized = normalize_foreign_language_evidence(
        _report(holder="Holder (69.72%)", relationship="associate", source_url=source),
        [],
        ticker="TEST",
        additional_records=[record],
    )
    assert "Control Status: UNKNOWN" in normalized
    assert "Ownership Evidence Status: REJECTED" in normalized


@pytest.mark.parametrize("pct", [None, 25, 69.72])
@pytest.mark.parametrize(
    "assertion,supported",
    [
        ("Holder uses the equity method for its investment.", True),
        ("Holder uses the equity method for its stake, as at 31 May 2026.", True),
        ("Holder uses the equity method for its stake, as at 31 may 2026.", True),
        ("Holder uses the equity method for its stake, as at May 31, 2026.", True),
        ("Holder uses the equity method for its stake, as at MAY 31, 2026.", True),
        ("As at 31 May 2026, Holder uses the equity method for its stake.", True),
        ("As at 31  May 2026, Holder uses the equity method for its stake.", True),
        ("As at 31\tMay\t2026, Holder uses the equity method for its stake.", True),
        ("Holder uses the equity method for its stake, as at 31  May 2026.", True),
        ("Holder uses the equity method for its stake, as at May\t31,\t2026.", True),
        ("Holder uses the equity method for its stake, as at 31 Feb 2026.", True),
        ("Holder has significant influence over the company, which may change.", True),
        (
            "Holder has significant influence over the company, but it might change.",
            True,
        ),
        (
            "Holder has significant influence over the company, which may be unverified.",
            False,
        ),
        (
            "Holder has significant influence over the company, but it may be untrue.",
            False,
        ),
        (
            "Holder has significant influence over the company, only if its options vest.",
            False,
        ),
        ("Holder may have significant influence over the company.", False),
        ("Holder MAY have significant influence over the company.", False),
        ("May Holder use the equity method for its investment?", False),
        ("Holder applies the equity-method for its investment.", True),
        ("Holder is accounted for as an associate.", True),
        ("Holder is treated as an associate.", True),
        ("Holder is an associate of the Company.", True),
        ("Holder has significant influence only.", True),
        ("Holder exercises significant influence over the group.", True),
        ("Holder is the entity with significant influence over the issuer.", True),
        ("Holder does not use the equity method.", False),
        ("Holder never uses the equity method.", False),
        ("Holder is not accounted for as an associate.", False),
        ("Holder has no significant influence over the group.", False),
        ("Holder does not have significant influence over the group.", False),
        ("Holder may use the equity method.", False),
        ("Holder uses the equity method if the transaction closes.", False),
        (
            "Holder uses the equity method for its investment, but this is unverified.",
            False,
        ),
        (
            "Holder has significant influence only, according to an unverified claim.",
            False,
        ),
        ("Holder uses the equity method subject to shareholder approval.", False),
        ("Holder uses the equity method, but the relationship is uncertain.", False),
        (
            "Holder uses the equity method, but Other Fund may change its investment plans.",
            True,
        ),
        (
            "Holder uses the equity method, and its associates may change their investment plans.",
            True,
        ),
        (
            "Holder has significant influence only, but unrelated financial projections are unverified.",
            True,
        ),
        ("Holder has significant influence only if its options vest.", False),
        ("Holder might be classified as an associate.", False),
        ("Holder could have significant influence over the group.", False),
        ("If Holder uses the equity method, accounting will change.", False),
        (
            "Whether Holder has significant influence over the group remains unclear.",
            False,
        ),
        ('The report quotes "Holder uses the equity method" as a hypothetical.', False),
        (
            "The report quotes 'Holder is treated as an associate' as a hypothetical.",
            False,
        ),
        ("Holder is an associate of a director.", False),
        ("Holder has significant influence in the industry.", False),
        ("Holder discusses the equity method and significant influence.", False),
        (
            "Holder and its associates are unrelated to the accounting assessment.",
            False,
        ),
        ("Holder is independent. Other Fund uses the equity method.", False),
        (
            "Holder is independent, while Other Fund has significant influence over the group.",
            False,
        ),
        (
            "Holder owns shares and its associate Other Fund uses the equity method.",
            False,
        ),
        ("Holder says Other Fund uses the equity method.", False),
        ("Holder 未採用權益法。", False),
        ("Holder 財務成本 採用權益法認列之關聯企業及合資損益之份額 489,717", False),
        ("Holder 为联营公司。", False),
        (
            "Holder 持分法による投資利益が増加し、その他の営業収益も大きく増えました。",
            False,
        ),
        ("Holder uses the equity method. 獨立非執行董事 oversee the board.", True),
        ("獨立非執行董事 oversee the board. Holder uses the equity method.", True),
        (
            "Holder does not use the equity method, but Holder has significant influence only.",
            True,
        ),
    ],
)
def test_accounting_assertion_semantics_across_stake_and_relationship_only(
    pct, assertion, supported
):
    source = "https://issuer.example/ownership"
    content = (f"Holder owns {pct}%. " if pct is not None else "") + assertion
    normalized = normalize_foreign_language_evidence(
        _report(
            holder="Holder (25%)" if pct is None else f"Holder ({pct}%)",
            relationship="equity method",
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[_record(content, urls=(source,))],
    )
    expected = (
        "UNKNOWN"
        if supported and pct is not None and pct > 50
        else "NOT_CONTROLLED"
        if supported
        else "CONTROLLED"
        if pct is not None and pct > 50
        else "UNKNOWN"
    )
    assert f"Control Status: {expected}" in normalized
    if pct is None:
        assert (
            f"Ownership Evidence Status: {'DISCLOSED_UNVERIFIED' if supported else 'REJECTED'}"
            in normalized
        )
        assert "Largest Shareholder: UNKNOWN" in normalized
    card = build_card(
        ticker="TEST",
        company_name="TEST",
        merged_data={},
        senior_metrics={},
        fla_report=normalized,
    )
    assert card.control_status == expected


@pytest.mark.parametrize(
    "content",
    [
        "Holder is interested in an acquisition worth 69.72% of revenue.",
        "Holder may be interested in 69.72% of the issued shares.",
        "Holder is interested in 69.72% of the issued shares if its options vest.",
        "Holder is interested in 69.72% of the issued shares subject to shareholder approval.",
        "Holder is interested in 69.72% of the issued shares, according to an unverified claim.",
        "Holder is interested in 69.72% of the issued shares, but this is unverified.",
        "It is uncertain whether Holder is interested in 69.72% of the issued shares.",
        'The report quotes "Holder is interested in 69.72% of the issued shares".',
        "Holder is not interested in 69.72% of the issued shares.",
        "Holder were formerly interested in 69.72% of the issued shares.",
        "Holder is interested in 69.72% of another investor's portfolio.",
        "Holder is interested in Company shares. Revenue grew 69.72%.",
    ],
)
def test_interest_without_bound_issued_share_percentage_cannot_verify_stake(content):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(holder="Holder (69.72%)", source_url=source),
        [],
        ticker="TEST",
        additional_records=[_record(content, urls=(source,))],
    )
    assert "Ownership Evidence Status: REJECTED" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert (
        build_card(
            ticker="TEST",
            company_name="TEST",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        ).control_status
        == "UNKNOWN"
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        " as at 31 May 2026",
        " as at 31 may 2026",
        " as at May 31, 2026",
        " as at MAY 31, 2026",
        " as at 31  May 2026",
        " as at May\t31,\t2026",
        ", which may change",
        ", but Other Fund may change its investment plans",
        ", and its associates may change their investment plans",
    ],
)
def test_current_interest_assertion_preserves_normal_and_independent_clauses(suffix):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(holder="Holder (69.72%)", source_url=source),
        [],
        ticker="TEST",
        additional_records=[
            _record(
                f"Holder is interested in 69.72% of the issued shares{suffix}.",
                urls=(source,),
            )
        ],
    )
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: CONTROLLED" in normalized
    assert (
        build_card(
            ticker="TEST",
            company_name="TEST",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        ).control_status
        == "CONTROLLED"
    )


def test_restored_bare_associate_defect_is_killed_by_captured_regression(monkeypatch):
    import src.agents.foreign_language_evidence as evidence

    monkeypatch.setattr(
        evidence,
        "_supports_non_control_relationship",
        lambda sentence, holder, relationship: any(
            term in sentence.casefold()
            for term in (
                "equity method",
                "equity-method",
                "significant influence",
                "associate",
            )
        ),
    )
    with pytest.raises(AssertionError):
        test_coholders_names_and_negated_accounting_do_not_establish_non_control(
            "Lu Ruibo",
            "Mr. Lu Ruibo and his associates are interested in {pct}% of the issued shares.",
            69.72,
            "associate",
        )
    test_explicit_accounting_relationship_remains_authoritative(
        25, "uses the equity method for its investment"
    )


def test_restored_class_majority_defect_is_killed_by_teck_regression(monkeypatch):
    import src.agents.foreign_language_evidence as evidence

    monkeypatch.setattr(
        evidence, "_majority_is_class_scoped", lambda records, holder, pct: False
    )
    with pytest.raises(AssertionError):
        test_captured_teck_class_votes_do_not_authorize_issuer_control()


@pytest.mark.parametrize(
    "holder",
    [
        "Acme Co. Limited",
        "Acme Corp. Holdings",
        "Acme Inc. Group",
        "Acme Ltd. Holdings",
        "Acme Limited",
        "May Holdings",
        "MAY Holdings Co. Limited",
    ],
)
@pytest.mark.parametrize(
    "assertion,supported",
    [
        ("{holder} uses the equity method for its investment.", True),
        ("{holder} is accounted for as an associate.", True),
        ("{holder} has significant influence only.", True),
        ("{holder} does not use the equity method.", False),
        ("{holder} may use the equity method.", False),
        ("{holder} uses the equity method, but this is unverified.", False),
        ("{holder} and its associates are independent investors.", False),
        (
            "{holder} does not use the equity method, but {holder} has significant influence only.",
            True,
        ),
        ("Other{holder} uses the equity method for its investment.", False),
    ],
)
@pytest.mark.parametrize("pct", [None, 25, 69.72])
def test_corporate_abbreviation_normalization_preserves_exact_accounting_actor(
    holder, assertion, supported, pct
):
    source = "https://issuer.example/ownership"
    content = (
        f"{holder} holds {pct}%. " if pct is not None else ""
    ) + assertion.format(holder=holder)
    normalized = normalize_foreign_language_evidence(
        _report(
            holder=f"{holder} ({25 if pct is None else pct}%)",
            relationship="equity method",
            source_url=source,
        ),
        [],
        ticker="TEST",
        additional_records=[_record(content, urls=(source,))],
    )
    expected = (
        "UNKNOWN"
        if supported and pct is not None and pct > 50
        else "NOT_CONTROLLED"
        if supported
        else "CONTROLLED"
        if pct is not None and pct > 50
        else "UNKNOWN"
    )
    assert f"Control Status: {expected}" in normalized
    assert (
        build_card(
            ticker="TEST",
            company_name="TEST",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        ).control_status
        == expected
    )
    if pct is None:
        assert (
            f"Ownership Evidence Status: {'DISCLOSED_UNVERIFIED' if supported else 'REJECTED'}"
            in normalized
        )


@pytest.mark.parametrize(
    "holder",
    [
        "Acme Co. Limited",
        "Acme Corp. Holdings",
        "Acme Inc. Group",
        "Acme Ltd. Holdings",
    ],
)
@pytest.mark.parametrize(
    "tail,controlled",
    [
        ("is interested in 69.72% of the issued shares", True),
        ("and its associates are interested in 69.72% of the issued shares", True),
        ("is interested in 69.72% of the issued shares if its options vest", False),
        (
            "is interested in 69.72% of the issued shares subject to shareholder approval",
            False,
        ),
        ("is not interested in 69.72% of the issued shares", False),
    ],
)
def test_corporate_abbreviations_bind_interest_and_certainty(holder, tail, controlled):
    source = "https://issuer.example/ownership"
    normalized = normalize_foreign_language_evidence(
        _report(holder=f"{holder} (69.72%)", source_url=source),
        [],
        ticker="TEST",
        additional_records=[_record(f"{holder} {tail}.", urls=(source,))],
    )
    expected = "CONTROLLED" if controlled else "UNKNOWN"
    assert f"Control Status: {expected}" in normalized
    assert (
        build_card(
            ticker="TEST",
            company_name="TEST",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        ).control_status
        == expected
    )


@pytest.mark.parametrize(
    "holder",
    [
        "Acme Co. Limited",
        "Acme Corp. Holdings",
        "Acme Inc. Group",
        "Acme Ltd. Holdings",
    ],
)
@pytest.mark.parametrize(
    "votes,controlled", [(None, False), ("60", True), ("34", False)]
)
def test_corporate_abbreviations_bind_class_scope_and_total_issuer_votes(
    holder, votes, controlled
):
    source = "https://issuer.example/ownership"
    content = f"{holder} holds 60% of Class A shares."
    if votes is not None:
        content += f" {holder} holds {votes}% of total voting rights."
    normalized = normalize_foreign_language_evidence(
        _report(holder=f"{holder} (60%)", basis="UNKNOWN", source_url=source),
        [],
        ticker="TEST",
        additional_records=[_record(content, urls=(source,))],
    )
    expected = "CONTROLLED" if controlled else "UNKNOWN"
    assert f"Control Status: {expected}" in normalized
    assert (
        build_card(
            ticker="TEST",
            company_name="TEST",
            merged_data={},
            senior_metrics={},
            fla_report=normalized,
        ).control_status
        == expected
    )


@pytest.mark.parametrize(
    "holder", ["Holder", "Acme Co. Limited", "Acme Corp. Holdings"]
)
def test_exact_actor_boundary_rejects_embedded_holder_name(holder):
    from src.agents.foreign_language_evidence import (
        _holder_sentences,
        _supports_non_control_relationship,
    )

    sentence = f"Other{holder} uses the equity method for its investment."
    assert not any(
        _supports_non_control_relationship(clause, holder, "EQUITY METHOD")
        for clause in _holder_sentences(sentence, holder)
    )
