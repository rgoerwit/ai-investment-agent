"""Regression tests for deterministic FLA ownership/capacity provenance."""

from langchain_core.messages import ToolMessage

from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
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
