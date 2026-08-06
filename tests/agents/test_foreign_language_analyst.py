"""
Tests for Foreign Language Analyst agent and Fundamentals Sync barrier.

Tests cover:
1. Prompt loading and format validation
2. Tool availability (search_foreign_sources)
3. AgentState field for foreign_language_report
4. Fundamentals sync barrier logic
5. Graph structure with new agent
"""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.analyst_nodes import (
    _build_retry_invocation_messages,
    _normalize_structured_output,
    _should_retry_output,
)
from src.agents.management_guidance import (
    _discover_local_issuer_name,
    _management_guidance_queries,
    _preload_management_guidance_evidence,
)
from tests.helpers.frozen_regressions import load_frozen_regression


class TestForeignLanguageAnalystPrompt:
    """Tests for the Foreign Language Analyst prompt configuration."""

    def test_prompt_file_exists(self):
        """Verify prompt JSON file exists."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        assert prompt_path.exists(), "Foreign Language Analyst prompt file not found"

    def test_prompt_valid_json(self):
        """Verify prompt file contains valid JSON."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        # Check required fields
        assert "agent_key" in data
        assert "agent_name" in data
        assert "system_message" in data
        assert "requires_tools" in data

        assert data["agent_key"] == "foreign_language_analyst"
        assert data["requires_tools"] is True

    def test_prompt_has_workflow_instructions(self):
        """Verify prompt contains workflow instructions."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        system_message = data["system_message"]

        # Should have workflow steps
        assert "INFER CONTEXT" in system_message
        assert "SEARCH" in system_message
        assert "EXTRACT" in system_message or "REPORT" in system_message

    def test_prompt_has_ticker_mappings(self):
        """Verify prompt contains ticker suffix to country/language mappings."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        system_message = data["system_message"]

        # Should have common suffix mappings
        assert ".T" in system_message  # Japan
        assert ".HK" in system_message  # Hong Kong
        assert ".KS" in system_message or ".KQ" in system_message  # Korea

    def test_prompt_has_fallback_instructions(self):
        """Verify prompt has fallback to premium English sources."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        system_message = data["system_message"]

        # Should mention premium sources as fallback
        assert (
            "bloomberg" in system_message.lower()
            or "morningstar" in system_message.lower()
        )


class TestForeignLanguageGuidanceRetry:
    def test_missing_guidance_block_triggers_retry(self):
        assert _should_retry_output(
            "Native filing review without a structured guidance block.",
            "foreign_language_analyst",
        )

    def test_retry_adds_targeted_evidence_correction(self):
        messages = [HumanMessage(content="Analyze TEST.T")]

        retry_messages = _build_retry_invocation_messages(
            messages,
            "foreign_language_analyst",
            "No guidance block",
        )

        assert len(retry_messages) == 2
        assert "MANAGEMENT_GUIDANCE" in retry_messages[-1].content
        assert "LATEST_RESULTS" in retry_messages[-1].content
        assert "SEARCHES_COMPLETED" in retry_messages[-1].content

    def test_valid_guidance_without_latest_results_triggers_retry(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
SEARCHES_COMPLETED: latest results release
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
### --- END MANAGEMENT_GUIDANCE ---
"""

        assert _should_retry_output(content, "foreign_language_analyst")

    def test_valid_guidance_and_latest_results_do_not_trigger_retry(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
SEARCHES_COMPLETED: latest results release
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
### --- END MANAGEMENT_GUIDANCE ---
### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND
LATEST_RESULTS_PERIOD: N/A
LATEST_RESULTS_PERIOD_END: N/A
LATEST_RESULTS_PRIOR_PERIOD: N/A
LATEST_RESULTS_PRIOR_PERIOD_END: N/A
LATEST_RESULTS_PERIOD_MONTHS: N/A
LATEST_RESULTS_CURRENCY: N/A
LATEST_RESULTS_REPORTING_UNIT: N/A
LATEST_RESULTS_REVENUE: N/A
LATEST_RESULTS_PRIOR_REVENUE: N/A
LATEST_RESULTS_EARNINGS: N/A
LATEST_RESULTS_PRIOR_EARNINGS: N/A
LATEST_RESULTS_EARNINGS_SCOPE: N/A
LATEST_RESULTS_SOURCE_URL: N/A
### --- END LATEST_RESULTS ---
"""

        assert not _should_retry_output(content, "foreign_language_analyst")

    def test_incomplete_latest_results_contract_triggers_retry(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
SEARCHES_COMPLETED: latest results release
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
### --- END MANAGEMENT_GUIDANCE ---
### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND
### --- END LATEST_RESULTS ---
"""

        assert _should_retry_output(content, "foreign_language_analyst")

    def test_fundamentals_retry_corrects_dropped_guidance_fields(self):
        messages = [HumanMessage(content="Analyze TEST.T")]
        incomplete = (
            "### --- START DATA_BLOCK ---\n"
            "RAW_HEALTH_SCORE: 7/12\n"
            "### --- END DATA_BLOCK ---"
        )

        retry_messages = _build_retry_invocation_messages(
            messages,
            "fundamentals_analyst",
            incomplete,
        )

        assert len(retry_messages) == 2
        assert "GUIDANCE_COVERAGE_STATUS" in retry_messages[-1].content
        assert "NORMALIZED_EARNINGS_AVAILABLE" in retry_messages[-1].content

    def test_japan_preflight_queries_target_figures_and_tax_bridge(self):
        queries = dict(
            _management_guidance_queries(
                "6745.T",
                "Hochiki Corporation",
                as_of=date(2026, 7, 20),
            )
        )

        assert queries["earnings_bridge"].startswith("6745 Hochiki Corporation")
        assert "2027年3月期" in queries["results_package"]
        assert "決算説明資料" in queries["results_package"]
        assert "営業利益" in queries["results_package"]
        assert "当期純利益" in queries["results_package"]
        assert "決算説明会" in queries["earnings_bridge"]
        assert "賃上げ促進税制" in queries["earnings_bridge"]

    def test_local_issuer_name_requires_ticker_matched_listing_title(self):
        payload = """
<result><title>賃上げ促進税制の一般解説</title></result>
<result><title>ホーチキ(株)【6745】：決算情報</title></result>
"""

        assert _discover_local_issuer_name(payload, "6745.T") == "ホーチキ"
        assert _discover_local_issuer_name(payload, "9999.T") is None

    @pytest.mark.parametrize(
        ("ticker", "title", "expected"),
        [
            ("KTY.WA", "Grupa Kęty S.A. (KTY.WA) - wyniki", "Grupa Kęty S.A."),
            (
                "7052.KL",
                "PADINI: PADINI HOLDINGS BERHAD (7052) | KLSE Screener",
                "PADINI HOLDINGS BERHAD",
            ),
            ("0700.HK", "騰訊控股【0700.HK】業績公告", "騰訊控股"),
        ],
    )
    def test_local_issuer_name_is_script_agnostic(self, ticker, title, expected):
        payload = f"<result><title>{title}</title></result>"

        assert _discover_local_issuer_name(payload, ticker) == expected

    def test_polish_and_malaysian_queries_use_local_filing_vocabulary(self):
        polish = dict(
            _management_guidance_queries(
                "KTY.WA", "Grupa Kęty S.A.", as_of=date(2026, 7, 20)
            )
        )
        malaysia = dict(
            _management_guidance_queries(
                "7052.KL",
                "Padini Holdings Berhad",
                as_of=date(2026, 7, 20),
            )
        )

        assert "raport okresowy" in polish["results_package"]
        assert "ulga podatkowa" in polish["earnings_bridge"]
        assert "laporan tahunan" in malaysia["results_package"]
        assert "annual report" in malaysia["results_package"]
        assert "insentif cukai" in malaysia["earnings_bridge"]

    @pytest.mark.asyncio
    async def test_preflight_records_code_owned_query_outcomes(self):
        from src.tooling.runtime import ToolResult

        calls = []

        class FakeToolService:
            async def execute(self, call, runner):
                calls.append(call)
                if call.name == "get_official_filings":
                    return ToolResult(value="EDINET filing payload")
                query = call.args["search_query"]
                return ToolResult(value=f"search evidence for {query}")

        with (
            patch(
                "src.runtime_services.get_current_tool_service",
                return_value=FakeToolService(),
            ),
            patch("src.agents.management_guidance.logger.info") as log_info,
        ):
            evidence = await _preload_management_guidance_evidence("6745.T", "ホーチキ")

        assert len(calls) == 3
        assert {call.source for call in calls} == {"preflight"}
        assert {call.agent_key for call in calls} == {"foreign_language_analyst"}
        assert "#### results_package\nSTATUS: COMPLETED" in evidence
        assert "#### earnings_bridge\nSTATUS: COMPLETED" in evidence
        assert "#### statutory_filing_api\nSTATUS: COMPLETED" in evidence
        assert "賃上げ促進税制" in evidence
        search_calls = [call for call in calls if call.name == "search_foreign_sources"]
        assert all(
            "賃上げ促進税制" in call.args["priority_terms"] for call in search_calls
        )
        telemetry = next(
            call
            for call in log_info.call_args_list
            if call.args == ("management_guidance_preflight_complete",)
        )
        assert telemetry.kwargs["evidence_chars"] > 0
        assert (
            telemetry.kwargs["call_statuses"]["results_package"]
            == "SUCCEEDED/RESULTS_FOUND"
        )

    @pytest.mark.asyncio
    async def test_preflight_extracts_registered_official_results_document(self):
        from src.tooling.runtime import ToolResult

        calls = []
        official_url = "https://www.jpx.co.jp/listing/results.pdf"

        class FakeToolService:
            async def execute(self, call, runner):
                calls.append(call)
                if call.name == "get_official_document":
                    return ToolResult(
                        value=(
                            f'DOCUMENT_METADATA: {{"source_url": "{official_url}"}}\n'
                            "Revenue 1,500; net income 405."
                        )
                    )
                if call.name == "get_official_filings":
                    return ToolResult(value="EDINET filing payload")
                if "決算説明資料" in call.args["search_query"]:
                    return ToolResult(
                        value=f"<result><url>{official_url}</url></result>"
                    )
                return ToolResult(value="native bridge evidence")

        with patch(
            "src.runtime_services.get_current_tool_service",
            return_value=FakeToolService(),
        ):
            evidence = await _preload_management_guidance_evidence("6745.T", "ホーチキ")

        official_call = next(
            call for call in calls if call.name == "get_official_document"
        )
        assert official_call.args["url"] == official_url
        assert "#### latest_results_document\nSTATUS: COMPLETED" in evidence
        assert "Revenue 1,500; net income 405." in evidence

    @pytest.mark.asyncio
    async def test_preflight_descends_bounded_official_child_paths(self):
        from src.tooling.runtime import ToolResult

        calls = []
        hub_url = "https://www.jpx.co.jp/investor-relations"
        child_url = "https://www.jpx.co.jp/results/fy-2026"

        class FakeToolService:
            async def execute(self, call, runner):
                calls.append(call)
                if call.name == "get_official_filings":
                    return ToolResult(value="EDINET filing payload")
                if call.name == "get_official_document":
                    if call.args["url"] == hub_url:
                        return ToolResult(
                            value=(
                                "STATUS: EVIDENCE_FOUND\n"
                                "DOCUMENT_METADATA: "
                                f'{{"source_url": "{hub_url}", '
                                '"candidate_paths": ["/results/fy-2026"]}\n'
                                "Investor relations index."
                            )
                        )
                    return ToolResult(
                        value=(
                            "STATUS: EVIDENCE_FOUND\n"
                            f'DOCUMENT_METADATA: {{"source_url": "{child_url}"}}\n'
                            "FY2026 revenue guidance 740 to 780."
                        )
                    )
                if "決算説明資料" in call.args["search_query"]:
                    return ToolResult(value=f"<result><url>{hub_url}</url></result>")
                return ToolResult(value="native bridge evidence")

        with patch(
            "src.runtime_services.get_current_tool_service",
            return_value=FakeToolService(),
        ):
            evidence = await _preload_management_guidance_evidence("6745.T", "ホーチキ")

        official_urls = [
            call.args["url"] for call in calls if call.name == "get_official_document"
        ]
        assert official_urls == [hub_url, child_url]
        assert "#### official_child_document_1" in evidence
        assert "FY2026 revenue guidance 740 to 780." in evidence

    @pytest.mark.asyncio
    async def test_preflight_uses_discovered_local_name_for_bridge_query(self):
        from src.tooling.runtime import ToolResult

        calls = []

        class FakeToolService:
            async def execute(self, call, runner):
                calls.append(call)
                if call.name == "get_official_filings":
                    return ToolResult(value="EDINET filing payload")
                if "決算説明資料" in call.args["search_query"]:
                    return ToolResult(
                        value=(
                            "<result><title>ホーチキ(株)【6745】：決算情報"
                            "</title><url>https://example.com/results</url></result>"
                        )
                    )
                return ToolResult(value="native bridge evidence")

        with patch(
            "src.runtime_services.get_current_tool_service",
            return_value=FakeToolService(),
        ):
            evidence = await _preload_management_guidance_evidence(
                "6745.T", "Hochiki Corporation", enable_extraction=False
            )

        bridge_call = next(
            call
            for call in calls
            if call.name == "search_foreign_sources"
            and "決算説明会" in call.args["search_query"]
        )
        assert bridge_call.args["search_query"].startswith("6745 ホーチキ ")
        assert "LOCAL_ISSUER_NAME: ホーチキ" in evidence

    def test_normalizer_overwrites_self_attested_search_coverage(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://finance.logmi.jp/articles/384869
SEARCHES_COMPLETED: everything imaginable
OPERATING_PROFIT_GUIDANCE: JPY 12.3 billion, up from JPY 12.066 billion
NET_INCOME_GUIDANCE: JPY 9.0 billion, down from JPY 9.377 billion
OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN
MATERIAL_NONOPERATING_DRIVER: YES
DRIVER_TYPE: TAX_CREDIT
DRIVER_PERSISTENCE: EXPIRING
EARNINGS_BASELINE_STATUS: DURABLE
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: COMPLETED
#### statutory_filing_api
STATUS: FAILED
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "6745.T",
            management_guidance_evidence=evidence,
        )

        assert (
            "SEARCHES_COMPLETED: results_package=SUCCEEDED/RESULTS_FOUND" in normalized
        )
        assert "statutory_filing_api=FAILED" in normalized
        assert "SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT" in normalized
        assert "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED" in normalized
        assert "GUIDANCE_BRIDGE_STATUS: RECONCILED" in normalized

    def test_normalizer_does_not_treat_a_bare_results_url_as_guidance(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://example.com/results-release
SEARCHES_COMPLETED: everything imaginable
OPERATING_PROFIT_GUIDANCE: Not explicitly provided in summary
NET_INCOME_GUIDANCE: N/A
OPERATING_VS_NET_DIRECTION: UNKNOWN
MATERIAL_NONOPERATING_DRIVER: NO
DRIVER_TYPE: NONE
EARNINGS_BASELINE_STATUS: DURABLE
NORMALIZED_EARNINGS_AVAILABLE: NO
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: COMPLETED
#### statutory_filing_api
STATUS: COMPLETED
#### guidance_extract
STATUS: INSUFFICIENT_DATA
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "6745.T",
            management_guidance_evidence=evidence,
        )

        assert "MATERIAL_NONOPERATING_DRIVER: UNKNOWN" in normalized
        assert "DRIVER_TYPE: UNKNOWN" in normalized
        assert "EARNINGS_BASELINE_STATUS: UNKNOWN" in normalized
        assert "GUIDANCE_BRIDGE_STATUS: NOT_APPLICABLE" in normalized

    def test_not_disclosed_requires_complete_source_coverage(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
SOURCE_TYPE: N/A
SOURCE_URL: N/A
OPERATING_PROFIT_GUIDANCE: N/A
NET_INCOME_GUIDANCE: N/A
OPERATING_VS_NET_DIRECTION: UNKNOWN
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
DRIVER_TYPE: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = """#### results_package
STATUS: COMPLETED
EXECUTION_STATUS: SUCCEEDED
EVIDENCE_STATUS: RESULTS_FOUND
#### earnings_bridge
STATUS: INSUFFICIENT_DATA
EXECUTION_STATUS: SUCCEEDED
EVIDENCE_STATUS: NO_RESULTS
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "TEST.T",
            management_guidance_evidence=evidence,
        )

        assert "COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH" in normalized

    def test_incomplete_causal_guidance_keeps_bridge_unresolved(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://example.com/results-release
OPERATING_PROFIT_GUIDANCE: JPY 12.3 billion, up year over year
NET_INCOME_GUIDANCE: N/A
OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN
MATERIAL_NONOPERATING_DRIVER: YES
DRIVER_TYPE: TAX_CREDIT
DRIVER_PERSISTENCE: EXPIRING
EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED
NORMALIZED_EARNINGS_AVAILABLE: NO
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: COMPLETED
#### statutory_filing_api
STATUS: COMPLETED
#### guidance_extract
STATUS: INSUFFICIENT_DATA
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "6745.T",
            management_guidance_evidence=evidence,
        )

        assert "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED" in normalized
        assert "GUIDANCE_BRIDGE_STATUS: UNRESOLVED" in normalized

    def test_missing_guidance_block_is_repaired_without_erasing_useful_report(self):
        content = (
            "### FOREIGN SOURCE FINDINGS\n"
            "The annual report confirms revenue, segment profit, and ownership "
            "details from local-language sources. Cash-flow and governance findings "
            "are included with citations. This report remains useful even though "
            "forward guidance could not be resolved.\n"
        )
        evidence = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: INSUFFICIENT_DATA
#### statutory_filing_api
STATUS: FAILED
#### guidance_extract
STATUS: INSUFFICIENT_DATA
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "KTY.WA",
            management_guidance_evidence=evidence,
        )

        assert content.strip() in normalized
        assert "COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH" in normalized
        assert "EARNINGS_BASELINE_STATUS: UNKNOWN" in normalized
        assert _should_retry_output(normalized, "foreign_language_analyst")

    def test_all_failed_preflight_preserves_report_but_marks_search_failed(self):
        content = (
            "### FOREIGN SOURCE FINDINGS\n"
            "A substantive local-source review found segment and ownership data, "
            "but every code-owned guidance retrieval call failed before evidence "
            "could be returned. Existing cited findings should remain available "
            "to downstream analysts rather than being discarded wholesale.\n"
        )
        evidence = """#### results_package
STATUS: FAILED
#### earnings_bridge
STATUS: FAILED
#### statutory_filing_api
STATUS: FAILED
#### guidance_extract
STATUS: SKIPPED
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "7052.KL",
            management_guidance_evidence=evidence,
        )

        assert content.strip() in normalized
        assert "COVERAGE_STATUS: SEARCH_FAILED" in normalized
        assert "SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT" in normalized
        assert _should_retry_output(normalized, "foreign_language_analyst")

    def test_empty_output_is_not_converted_into_a_success(self):
        evidence = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: COMPLETED
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            "",
            "6745.T",
            management_guidance_evidence=evidence,
        )

        assert normalized == ""
        assert _should_retry_output(normalized, "foreign_language_analyst")

    def test_guidance_enums_are_canonicalized_before_validation(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: found
SOURCE_TYPE: transcript
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://finance.logmi.jp/articles/384869
OPERATING_PROFIT_GUIDANCE: JPY 12.3 billion
NET_INCOME_GUIDANCE: JPY 9.0 billion
OPERATING_VS_NET_DIRECTION: op_up_net_down
MATERIAL_NONOPERATING_DRIVER: yes
DRIVER_TYPE: tax_credit
DRIVER_PERSISTENCE: expiring
EARNINGS_BASELINE_STATUS: durable
NORMALIZED_EARNINGS_AVAILABLE: no
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = """#### results_package
STATUS: COMPLETED
#### earnings_bridge
STATUS: COMPLETED
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "6745.T",
            management_guidance_evidence=evidence,
        )

        assert "DRIVER_TYPE: TAX_CREDIT" in normalized
        assert "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED" in normalized
        assert _should_retry_output(normalized, "foreign_language_analyst")

    def test_guidance_na_semantic_unknowns_are_normalized_before_promotion(self):
        content = """### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
OPERATING_VS_NET_DIRECTION: N/A
MATERIAL_NONOPERATING_DRIVER: N/A
EARNINGS_BASELINE_STATUS: N/A
NORMALIZED_EARNINGS_AVAILABLE: N/A
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = """#### results_package
STATUS: COMPLETED
EVIDENCE_STATUS: COVERAGE_COMPLETE_NO_MATCH
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "1681.HK",
            management_guidance_evidence=evidence,
        )

        assert "OPERATING_VS_NET_DIRECTION: UNKNOWN" in normalized
        assert "MATERIAL_NONOPERATING_DRIVER: UNKNOWN" in normalized
        assert "EARNINGS_BASELINE_STATUS: UNKNOWN" in normalized
        assert "NORMALIZED_EARNINGS_AVAILABLE: N/A" in normalized

    def test_broker_projection_is_labeled_third_party(self):
        regression = load_frozen_regression("6782_TW_regression.json")
        guidance = regression["guidance_evidence"]
        source_url = regression["capacity_evidence"]["source_url"]
        content = f"""### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: {guidance["source_type"]}
SOURCE_DATE: 2026-04-16
SOURCE_URL: {source_url}
NET_INCOME_GUIDANCE: EPS projection, 14% YoY
NET_INCOME_YOY: {guidance["projected_eps_growth"]}
MANAGEMENT_IDENTIFIED: {guidance["management_identified"]}
OPERATING_VS_NET_DIRECTION: UNKNOWN
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
DRIVER_TYPE: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
NORMALIZED_EARNINGS_AVAILABLE: YES
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = f"""#### results_package
STATUS: COMPLETED
<result><url>{source_url}</url><summary>Yuanta Securities research report</summary></result>
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            regression["ticker"],
            management_guidance_evidence=evidence,
        )

        assert "SOURCE_AUTHORITY: THIRD_PARTY" in normalized

    def test_company_results_guidance_search_identity_does_not_mint_primary(self):
        source_url = "https://issuer.example/investors/results-2026"
        content = f"""### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: RESULTS_RELEASE
SOURCE_URL: {source_url}
MANAGEMENT_IDENTIFIED: YES
OPERATING_PROFIT_GUIDANCE: TWD 1.2 billion
NET_INCOME_GUIDANCE: TWD 900 million
OPERATING_VS_NET_DIRECTION: SAME_DIRECTION
MATERIAL_NONOPERATING_DRIVER: NO
DRIVER_TYPE: NONE
EARNINGS_BASELINE_STATUS: DURABLE
NORMALIZED_EARNINGS_AVAILABLE: YES
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = f"""#### results_package
STATUS: COMPLETED
<result><url>{source_url}</url><summary>Issuer FY2026 results release: operating
profit guidance TWD 1.2 billion; net income guidance TWD 900 million.</summary></result>
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "TEST.TW",
            management_guidance_evidence=evidence,
        )

        assert "SOURCE_AUTHORITY: UNKNOWN" in normalized

    def test_search_only_official_registry_guidance_is_not_primary(self):
        source_url = "https://www.twse.com.tw/investors/results-2026"
        content = f"""### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: RESULTS_RELEASE
SOURCE_URL: {source_url}
MANAGEMENT_IDENTIFIED: YES
OPERATING_PROFIT_GUIDANCE: TWD 1.2 billion
NET_INCOME_GUIDANCE: TWD 900 million
OPERATING_VS_NET_DIRECTION: SAME_DIRECTION
MATERIAL_NONOPERATING_DRIVER: NO
DRIVER_TYPE: NONE
EARNINGS_BASELINE_STATUS: DURABLE
NORMALIZED_EARNINGS_AVAILABLE: YES
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = f"""#### results_package
STATUS: COMPLETED
<result><url>{source_url}</url><summary>Issuer FY2026 results release: operating
profit guidance TWD 1.2 billion; net income guidance TWD 900 million.</summary></result>
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "TEST.TW",
            management_guidance_evidence=evidence,
        )

        assert "SOURCE_AUTHORITY: UNKNOWN" in normalized

    def test_fetched_official_guidance_record_is_primary(self):
        source_url = "https://www.twse.com.tw/investors/results-2026"
        content = f"""### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: RESULTS_RELEASE
SOURCE_URL: {source_url}
MANAGEMENT_IDENTIFIED: YES
OPERATING_PROFIT_GUIDANCE: TWD 1.2 billion
NET_INCOME_GUIDANCE: TWD 900 million
OPERATING_VS_NET_DIRECTION: SAME_DIRECTION
MATERIAL_NONOPERATING_DRIVER: NO
DRIVER_TYPE: NONE
EARNINGS_BASELINE_STATUS: DURABLE
NORMALIZED_EARNINGS_AVAILABLE: YES
### --- END MANAGEMENT_GUIDANCE ---
"""
        record = SimpleNamespace(
            sequence=1,
            agent_key="foreign_language_analyst",
            tool_name="get_official_document",
            content=(
                "STATUS: EVIDENCE_FOUND\n"
                f'DOCUMENT_METADATA: {{"source_url": "{source_url}"}}\n'
                "Operating profit guidance TWD 1.2 billion; "
                "net income guidance TWD 900 million."
            ),
            content_sha256="abcdef1234567890",
            requested_urls=(source_url,),
            urls=(source_url,),
            blocked=False,
            evidence_status="EVIDENCE_FOUND",
        )

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "TEST.TW",
            management_guidance_evidence="STATUS: COMPLETED",
            evidence_messages=[],
        )
        assert "SOURCE_AUTHORITY: UNKNOWN" in normalized

        from src.agents.management_guidance import normalize_management_guidance_output

        normalized = normalize_management_guidance_output(
            content,
            "STATUS: COMPLETED",
            [record],
        )
        assert "SOURCE_AUTHORITY: PRIMARY" in normalized

    def test_guidance_values_cannot_be_split_across_records(self):
        source_url = "https://www.twse.com.tw/investors/results-2026"
        content = f"""### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: RESULTS_RELEASE
SOURCE_URL: {source_url}
MANAGEMENT_IDENTIFIED: YES
OPERATING_PROFIT_GUIDANCE: TWD 1.2 billion
NET_INCOME_GUIDANCE: TWD 900 million
### --- END MANAGEMENT_GUIDANCE ---
"""
        bound_record = SimpleNamespace(
            sequence=1,
            tool_name="get_official_document",
            content="Operating profit guidance TWD 1.2 billion.",
            content_sha256="abcdef1234567890",
            requested_urls=(source_url,),
            urls=(source_url,),
            blocked=False,
            evidence_status="EVIDENCE_FOUND",
        )
        unrelated_record = SimpleNamespace(
            sequence=2,
            tool_name="get_official_document",
            content="Net income guidance TWD 900 million.",
            content_sha256="fedcba0987654321",
            requested_urls=("https://www.twse.com.tw/other",),
            urls=("https://www.twse.com.tw/other",),
            blocked=False,
            evidence_status="EVIDENCE_FOUND",
        )

        from src.agents.management_guidance import normalize_management_guidance_output

        normalized = normalize_management_guidance_output(
            content,
            "STATUS: COMPLETED",
            [bound_record, unrelated_record],
        )
        assert "SOURCE_AUTHORITY: UNKNOWN" in normalized

    def test_official_guidance_url_without_claimed_values_is_not_primary(self):
        source_url = "https://www.twse.com.tw/investors/results-2026"
        content = f"""### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_TYPE: RESULTS_RELEASE
SOURCE_URL: {source_url}
MANAGEMENT_IDENTIFIED: YES
OPERATING_PROFIT_GUIDANCE: TWD 1.2 billion
NET_INCOME_GUIDANCE: TWD 900 million
### --- END MANAGEMENT_GUIDANCE ---
"""
        evidence = f"""#### results_package
STATUS: COMPLETED
<result><url>{source_url}</url><summary>Issuer FY2026 results release.</summary></result>
"""

        normalized = _normalize_structured_output(
            "foreign_language_analyst",
            content,
            "TEST.TW",
            management_guidance_evidence=evidence,
        )

        assert "SOURCE_AUTHORITY: UNKNOWN" in normalized


class TestSearchForeignSourcesTool:
    """Tests for the search_foreign_sources tool."""

    def test_tool_exists_in_toolkit(self):
        """Verify tool is available in toolkit."""
        from src.tools.registry import toolkit

        foreign_tools = toolkit.get_foreign_language_tools()
        assert len(foreign_tools) == 4

        tool_names = [t.name for t in foreign_tools]
        assert "search_foreign_sources" in tool_names
        assert "extract_guidance_sources" in tool_names
        assert "get_official_filings" in tool_names
        assert "get_official_document" in tool_names

    def test_tool_in_all_tools(self):
        """Verify tool is included in get_all_tools."""
        from src.tools.registry import toolkit

        all_tools = toolkit.get_all_tools()
        tool_names = [t.name for t in all_tools]
        assert "search_foreign_sources" in tool_names

    def test_tool_has_correct_description(self):
        """Verify tool has informative description."""
        from src.tools.registry import toolkit

        foreign_tools = toolkit.get_foreign_language_tools()
        tool = foreign_tools[0]

        assert "foreign" in tool.description.lower()
        assert "source" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_guidance_extraction_keeps_query_relevant_tax_passage(self):
        from unittest.mock import AsyncMock

        from src.tools.research import extract_guidance_sources

        raw = {
            "results": [
                {
                    "url": "https://finance.logmi.jp/articles/384869",
                    "raw_content": "X" * 8000
                    + "賃上げ促進税制による税額控除は今期適用されない。"
                    + "Y" * 8000,
                }
            ]
        }
        with patch(
            "src.tavily_utils.extract_tavily_inspected",
            new=AsyncMock(return_value=raw),
        ):
            output = await extract_guidance_sources.ainvoke(
                {
                    "urls": [
                        "file:///etc/passwd",
                        "https://finance.logmi.jp/articles/384869",
                    ],
                    "query": "ホーチキ 賃上げ促進税制 税額控除",
                }
            )

        assert "file:///etc/passwd" not in output
        assert "https://finance.logmi.jp/articles/384869" in output
        assert "賃上げ促進税制" in output
        assert output.rstrip().endswith("</search_results>")

    @pytest.mark.parametrize(
        ("priority_term", "passage"),
        [
            ("ulga podatkowa", "Jednorazowa ulga podatkowa podwyższyła zysk netto."),
            ("insentif cukai", "Insentif cukai tahun lalu tidak lagi terpakai."),
        ],
    )
    def test_locale_priority_terms_preserve_late_tax_passages(
        self, priority_term, passage
    ):
        from src.tools.shared import _format_and_truncate_tavily_result

        formatted = _format_and_truncate_tavily_result(
            [
                {
                    "title": "Results briefing",
                    "url": "https://example.com/results",
                    "content": "A" * 8000 + passage + "B" * 8000,
                }
            ],
            max_chars=1200,
            query="company results guidance",
            priority_terms=[priority_term],
        )

        assert priority_term.casefold() in formatted.casefold()

    def test_generic_formatter_has_no_tax_specific_bias(self):
        from src.tools.shared import _format_and_truncate_tavily_result

        formatted = _format_and_truncate_tavily_result(
            [
                {
                    "title": "Market report",
                    "url": "https://example.com/market",
                    "content": (
                        "tax credit " + "A" * 8000 + "TARGET_MARKET_SIGNAL" + "B" * 8000
                    ),
                }
            ],
            max_chars=1200,
            query="TARGET_MARKET_SIGNAL",
        )

        assert "TARGET_MARKET_SIGNAL" in formatted
        assert "tax credit" not in formatted

    @pytest.mark.asyncio
    async def test_tool_handles_no_tavily(self):
        """Test graceful handling when Tavily is not configured and DDG also empty."""
        from src.tools.research import search_foreign_sources

        # Mock tavily_tool as None AND DDG returning empty
        with patch("src.tools.shared.tavily_tool", None):
            with patch("src.tools.shared._ddg_search", return_value=[]):
                result = await search_foreign_sources.ainvoke(
                    {"ticker": "7203.T", "search_query": "Toyota 決算短信"}
                )

                assert "no results" in result.lower()

    @pytest.mark.asyncio
    async def test_output_terminates_with_search_results_closer(self):
        """The Tavily `<search_results>...</search_results>` wrapper must be
        the LAST meaningful element of the returned text.

        The heuristic content inspector
        (`_detect_search_results_breakouts`) treats the terminal closer as
        a legitimate trust-boundary footer only when nothing follows it
        (whitespace allowed). A trailing 'Note: ...' footer would surface
        every search as a `delimiter_breakout` warning at threat_level=high
        — exactly the May 2026 2364.TW false-positive pattern the user
        flagged.

        Header / metadata / 'Verify dates' note must therefore live BEFORE
        the wrapper, never after.
        """
        import re
        from unittest.mock import AsyncMock

        from src.tools.research import search_foreign_sources

        # Stub Tavily to return realistic wrapped output.
        async def fake_tavily(_args):
            return {
                "results": [
                    {
                        "title": "Toyota Q3 results",
                        "url": "https://example.com/toyota",
                        "content": "Toyota reported Q3 revenue of ¥2.4T ‪JPY‬.",
                    }
                ]
            }

        with (
            patch(
                "src.tools.shared._tavily_search_with_timeout",
                new=AsyncMock(side_effect=fake_tavily),
            ),
            patch("src.tools.shared._ddg_search", new=AsyncMock(return_value=[])),
            patch(
                "src.tools.shared.extract_company_name_async",
                new=AsyncMock(return_value="Toyota Motor Corp"),
            ),
        ):
            result = await search_foreign_sources.ainvoke(
                {"ticker": "7203.T", "search_query": "Toyota 決算短信"}
            )

        # The wrapper closer must appear, and must be terminal.
        closers = list(re.finditer(r"</search_results>", result, re.I))
        assert closers, "Output must contain a </search_results> closer"
        last_closer = closers[-1]
        # Whitespace only between last closer and end of string.
        assert result[last_closer.end() :].strip() == "", (
            "Text after the final </search_results> closer trips the "
            "delimiter-breakout heuristic. Move metadata/footers BEFORE "
            "the wrapper. Trailing text was: "
            f"{result[last_closer.end() :]!r}"
        )

    @pytest.mark.asyncio
    async def test_benign_act_as_the_phrase_does_not_trigger_block(self):
        """Reproducer for the May 2026 2373.HK incident: financial news
        routinely contains phrases like 'will act as the lead underwriter',
        which triggers the heuristic's `role_play` weight-1.5 signal.

        Pre-fix, that landed in the same `tool_output` envelope as the
        spurious `delimiter_breakout` from the post-wrapper `Note:` footer
        — combined weight 4.5 → severity 'high' → action 'block'. With the
        producer fix (closer is terminal), only the role_play signal
        remains; severity 'low' → action 'allow' → debug log only.
        """
        from unittest.mock import AsyncMock as _AsyncMock

        from src.tooling.heuristic_inspector import HeuristicInspector
        from src.tooling.inspector import InspectionEnvelope, SourceKind
        from src.tools.research import search_foreign_sources

        async def fake_tavily(_args):
            return {
                "results": [
                    {
                        "title": "Underwriting note",
                        "url": "https://example.com/x",
                        "content": (
                            "Morgan Stanley will act as the lead "
                            "underwriter for the IPO."
                        ),
                    }
                ]
            }

        with (
            patch(
                "src.tools.shared._tavily_search_with_timeout",
                new=_AsyncMock(side_effect=fake_tavily),
            ),
            patch("src.tools.shared._ddg_search", new=_AsyncMock(return_value=[])),
            patch(
                "src.tools.shared.extract_company_name_async",
                new=_AsyncMock(return_value="Test Co"),
            ),
        ):
            output = await search_foreign_sources.ainvoke(
                {"ticker": "2373.HK", "search_query": "lead underwriter"}
            )

        decision = await HeuristicInspector().inspect(
            InspectionEnvelope(
                content_text=output,
                source_kind=SourceKind.tool_output,
                source_name="search_foreign_sources",
                tool_name="search_foreign_sources",
                agent_key="foreign_language_analyst",
            )
        )

        assert decision.action == "allow", (
            f"Benign 'act as the' phrase wrongly triggered "
            f"action={decision.action} (threat_level={decision.threat_level}, "
            f"types={decision.threat_types})"
        )
        assert "delimiter_breakout" not in decision.threat_types, (
            "Producer wrapper must keep the </search_results> closer "
            "terminal so the breakout heuristic stays clean"
        )


class TestAgentStateField:
    """Tests for foreign_language_report field in AgentState."""

    def test_field_exists_in_agent_state(self):
        """Verify foreign_language_report field exists."""
        from src.agents import AgentState, InvestDebateState, RiskDebateState

        # Create a minimal state
        state = AgentState(
            messages=[],
            company_of_interest="TEST",
            company_name="Test Company",
            trade_date="2025-01-01",
            sender="test",
            market_report="",
            sentiment_report="",
            news_report="",
            raw_fundamentals_data="",
            foreign_language_report="test foreign data",
            fundamentals_report="",
            investment_debate_state=InvestDebateState(
                bull_history="",
                bear_history="",
                history="",
                current_response="",
                judge_decision="",
                count=0,
            ),
            investment_plan="",
            consultant_review="",
            trader_investment_plan="",
            risk_debate_state=RiskDebateState(
                risky_history="",
                safe_history="",
                neutral_history="",
                history="",
                latest_speaker="",
                current_risky_response="",
                current_safe_response="",
                current_neutral_response="",
                judge_decision="",
                count=0,
            ),
            final_trade_decision="",
            tools_called={},
            prompts_used={},
            red_flags=[],
            pre_screening_result="",
        )

        assert state["foreign_language_report"] == "test foreign data"


class TestFundamentalsSyncRouter:
    """Tests for the fundamentals_sync_router function."""

    def test_router_waits_for_all_analysts(self):
        """Test router returns __end__ if not all three analysts complete."""
        from src.graph import fundamentals_sync_router

        # Only Junior done
        state_junior_only = {
            "raw_fundamentals_data": "some data",
            "foreign_language_report": "",
            "legal_report": "",
        }
        result = fundamentals_sync_router(state_junior_only, {})
        assert result == "__end__"

        # Only Foreign done
        state_foreign_only = {
            "raw_fundamentals_data": "",
            "foreign_language_report": "some foreign data",
            "legal_report": "",
        }
        result = fundamentals_sync_router(state_foreign_only, {})
        assert result == "__end__"

        # Only Legal done
        state_legal_only = {
            "raw_fundamentals_data": "",
            "foreign_language_report": "",
            "legal_report": "some legal data",
        }
        result = fundamentals_sync_router(state_legal_only, {})
        assert result == "__end__"

        # Junior + Foreign done (missing Legal)
        state_junior_foreign = {
            "raw_fundamentals_data": "junior data",
            "foreign_language_report": "foreign data",
            "legal_report": "",
        }
        result = fundamentals_sync_router(state_junior_foreign, {})
        assert result == "__end__"

    def test_router_proceeds_when_all_three_complete(self):
        """Test router proceeds to Fundamentals Analyst when all three complete."""
        from src.graph import fundamentals_sync_router

        state_all_done = {
            "raw_fundamentals_data": "junior data",
            "foreign_language_report": "foreign data",
            "legal_report": "legal data",
        }
        result = fundamentals_sync_router(state_all_done, {})
        assert result == "Fundamentals Analyst"

    def test_router_handles_none_values(self):
        """Test router handles None values correctly."""
        from src.graph import fundamentals_sync_router

        state_with_none = {
            "raw_fundamentals_data": None,
            "foreign_language_report": None,
            "legal_report": None,
        }
        result = fundamentals_sync_router(state_with_none, {})
        assert result == "__end__"

    def test_router_proceeds_when_legal_failed_but_completed(self):
        """A failed legal branch should still satisfy the fundamentals barrier."""
        from src.graph import fundamentals_sync_router

        state = {
            "raw_fundamentals_data": "junior data",
            "foreign_language_report": "foreign data",
            "legal_report": "",
            "artifact_statuses": {
                "legal_report": {
                    "complete": True,
                    "ok": False,
                    "error_kind": "timeout",
                    "provider": "google",
                }
            },
        }

        result = fundamentals_sync_router(state, {})
        assert result == "Fundamentals Analyst"


class TestGraphStructure:
    """Tests for graph structure with Foreign Language Analyst."""

    @patch("src.graph.routing._is_auditor_enabled")
    def test_fan_out_includes_foreign_analyst(self, mock_auditor_enabled):
        """Test that fan_out_to_analysts includes Foreign Language Analyst."""
        from src.graph import fan_out_to_analysts

        # Disable auditor for this test to check base analyst count
        mock_auditor_enabled.return_value = False

        destinations = fan_out_to_analysts({}, {})

        assert "Foreign Language Analyst" in destinations
        assert "Value Trap Detector" in destinations
        assert (
            len(destinations) == 7
        )  # Market, Sentiment, News, Junior, Foreign, Legal, Value Trap

    def test_graph_creates_with_foreign_analyst(self):
        """Test that graph creation includes Foreign Language Analyst node."""
        from src.graph import create_trading_graph

        # Create graph with memory disabled to simplify
        graph = create_trading_graph(
            ticker="TEST", enable_memory=False, quick_mode=True
        )

        # Graph should compile without errors
        assert graph is not None


class TestSeniorFundamentalsContextInjection:
    """Tests for Senior Fundamentals Analyst receiving Foreign Language data."""

    def test_context_injection_code_path_exists(self):
        """Test that the context injection code path for foreign data exists in agents.py."""
        import inspect

        from src import agents

        # Get the source code of create_analyst_node
        source = inspect.getsource(agents.create_analyst_node)

        # Verify the foreign_language_report handling code exists
        assert "foreign_language_report" in source
        assert "foreign_data" in source
        assert "FOREIGN/ALTERNATIVE SOURCE DATA" in source

    def test_analyst_node_created_successfully(self):
        """Test that fundamentals_analyst node can be created."""
        from src.agents import create_analyst_node

        # Create the fundamentals_analyst node (no tools)
        # This just tests the factory function works
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        node_func = create_analyst_node(
            mock_llm, "fundamentals_analyst", [], "fundamentals_report"
        )

        # The node function exists and is callable
        assert callable(node_func)


class TestComputeDataConflicts:
    """Tests for the pre-Senior conflict detection function."""

    def test_ocf_discrepancy_flagged(self):
        """OCF mismatch >30% between Junior and FLA produces a conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"operatingCashflow": 19950000000, "marketCap": 130000000000}'
        fla = (
            "**FILING CASH FLOW**\n"
            "- Operating Cash Flow (Filing): ¥10.91B\n"
            "- Period: H1 2025\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "OCF" in result
        assert "PERIOD MISMATCH" in result
        assert "yfinance" in result

    def test_ocf_no_conflict_when_close(self):
        """OCF values within 30% produce no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"operatingCashflow": 10000000000}'
        fla = (
            "**FILING CASH FLOW**\n"
            "- Operating Cash Flow (Filing): ¥11.5B\n"
            "- Period: FY2024\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "OCF" not in result

    def test_peg_zero_flagged(self):
        """PEG 0.00 produces an UNRELIABLE flag."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.0}'
        result = compute_data_conflicts(junior, "")
        assert "PEG" in result
        assert "UNRELIABLE" in result

    def test_peg_near_zero_flagged(self):
        """PEG 0.02 produces an UNRELIABLE flag with implied growth."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.02}'
        result = compute_data_conflicts(junior, "")
        assert "PEG" in result
        assert "UNRELIABLE" in result
        assert "50x" in result

    def test_peg_normal_no_flag(self):
        """PEG 0.8 produces no flag."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.8}'
        result = compute_data_conflicts(junior, "")
        assert "PEG" not in result

    def test_low_analyst_count_for_large_cap(self):
        """Analyst count < 5 for >$500M market cap flags as anomaly."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 2, "marketCap": 1300000000}'
        result = compute_data_conflicts(junior, "")
        assert "ANALYST_COUNT" in result
        assert "ANOMALY" in result

    def test_low_analyst_count_small_cap_ok(self):
        """Analyst count < 5 for small cap ($200M) is not anomalous."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 2, "marketCap": 200000000}'
        result = compute_data_conflicts(junior, "")
        assert "ANALYST_COUNT" not in result

    def test_parent_company_found(self):
        """FLA finding a parent company flags the ownership gap."""
        from src.agents import compute_data_conflicts

        junior = '{"operatingCashflow": 5000000000}'
        fla = (
            "**OWNERSHIP STRUCTURE**\n"
            "- Controlling Shareholder: Bandai Namco Holdings (49.12%)\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "PARENT" in result
        assert "Bandai Namco" in result

    def test_empty_junior_returns_nothing(self):
        """No Junior data → empty result."""
        from src.agents import compute_data_conflicts

        result = compute_data_conflicts("", "some FLA data")
        assert result == ""

    def test_no_conflicts_returns_empty(self):
        """Clean data with no issues → empty result."""
        from src.agents import compute_data_conflicts

        junior = (
            '{"pegRatio": 1.2, "numberOfAnalystOpinions": 8, "marketCap": 500000000}'
        )
        result = compute_data_conflicts(junior, "")
        assert result == ""

    def test_header_present_when_conflicts_exist(self):
        """Conflict report starts with AUTOMATED CONFLICT CHECK header."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.0}'
        result = compute_data_conflicts(junior, "")
        assert "AUTOMATED CONFLICT CHECK" in result
        assert "system-generated" in result

    def test_split_quarantine_note_added_from_raw_data(self):
        """Recent split quarantine marker should produce a deterministic warning."""
        from src.agents import compute_data_conflicts

        junior = '{"_split_sensitive_metrics_quarantined": true}'

        result = compute_data_conflicts(junior, "")

        assert "SPLIT_SHARE_BASIS_MISMATCH" in result
        assert "forward EPS" in result
        assert "must be reported as N/A" in result

    def test_quarter_date_reconciliation_note_added_from_raw_data(self):
        """Quarter-date reconciliation marker should explain the newer metadata override."""
        from src.agents import compute_data_conflicts

        junior = (
            '{"latest_quarter_date": "2025-12-31", '
            '"_latest_quarter_date_source": "reconciled_most_recent_quarter"}'
        )

        result = compute_data_conflicts(junior, "")

        assert "QUARTER_DATE_RECONCILED" in result
        assert "2025-12-31" in result
        assert "LATEST_QUARTER_DATE must use the reconciled newer value" in result

    def test_statement_mrq_period_lag_is_explicit(self):
        """Newer metadata must not make an older statement MRQ read as latest."""
        from src.agents import compute_data_conflicts

        junior = json.dumps(
            {
                "latest_quarter_date": "2025-12-31",
                "_latest_quarter_date_source": "yfinance_quarterly",
                "_data_quality_notes": [
                    "Newer quarter metadata exists for 2026-03-31, but "
                    "statement-derived MRQ metrics remain aligned to 2025-12-31."
                ],
            }
        )

        result = compute_data_conflicts(junior, "")

        assert "MRQ_PERIOD_LAG" in result
        assert "period-bound trailing indicators" in result
        assert "not the latest reported quarter" in result


class TestLocalAnalystCoverageConflict:
    """Tests for conflict #5: local analyst coverage detection."""

    def test_fla_local_analyst_numeric(self):
        """FLA finds 25 local analysts vs Junior's 3 → conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3, "marketCap": 500000000}'
        fla = (
            "**LOCAL ANALYST COVERAGE**\n"
            "- Estimated Local Analysts: 25\n"
            "- Key Brokerages: Nomura, Daiwa, SMBC Nikko\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" in result
        assert "25" in result

    def test_fla_local_analyst_tier_high(self):
        """FLA reports HIGH tier → conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 5}'
        fla = "**LOCAL ANALYST COVERAGE**\n- Estimated Local Analysts: HIGH\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" in result
        assert "HIGH" in result

    def test_fla_local_analyst_tier_moderate(self):
        """FLA reports MODERATE tier → conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 2}'
        fla = "**LOCAL ANALYST COVERAGE**\n- Estimated Local Analysts: MODERATE\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" in result
        assert "MODERATE" in result

    def test_fla_local_analyst_unknown(self):
        """FLA reports UNKNOWN → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3}'
        fla = "**LOCAL ANALYST COVERAGE**\n- Estimated Local Analysts: UNKNOWN\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result

    def test_fla_local_analyst_low(self):
        """FLA reports LOW → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3}'
        fla = "**LOCAL ANALYST COVERAGE**\n- Estimated Local Analysts: LOW\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result

    def test_fla_no_local_section(self):
        """No LOCAL ANALYST section → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3}'
        fla = "**FILING CASH FLOW**\n- Operating Cash Flow (Filing): ¥10.91B\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result

    def test_fla_local_analyst_not_higher(self):
        """FLA finds 2 local analysts but Junior has 5 → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 5}'
        fla = "**LOCAL ANALYST COVERAGE**\n- Estimated Local Analysts: 2\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result


class TestPromptLoading:
    """Tests that the prompt system correctly loads Foreign Language Analyst."""

    def test_prompt_loads_via_get_prompt(self):
        """Test that get_prompt can load the foreign_language_analyst prompt."""
        from src.prompts import get_prompt

        prompt = get_prompt("foreign_language_analyst")

        assert prompt is not None
        assert prompt.agent_name == "Foreign Language Analyst"
        assert prompt.requires_tools is True

    def test_prompt_version_is_set(self):
        """Test that prompt has version metadata."""
        from src.prompts import get_prompt

        prompt = get_prompt("foreign_language_analyst")

        assert prompt.version is not None
        assert len(prompt.version) > 0


class TestValueTrapVerdictExtraction:
    """Tests for extract_value_trap_verdict() helper used in PM input assembly."""

    def test_aligned_verdict(self):
        """ALIGNED verdict with high score produces correct header."""
        from src.agents import extract_value_trap_verdict

        report = (
            "SCORE: 85\nVERDICT: ALIGNED\nTRAP_RISK: LOW\n"
            "OWNERSHIP:\n  CONCENTRATION: LOW"
        )
        result = extract_value_trap_verdict(report)
        assert "ALIGNED" in result
        assert "85/100" in result
        assert "LOW" in result

    def test_trap_verdict(self):
        """TRAP verdict surfaces correctly."""
        from src.agents import extract_value_trap_verdict

        report = "SCORE: 25\nVERDICT: TRAP\nTRAP_RISK: HIGH"
        result = extract_value_trap_verdict(report)
        assert "TRAP" in result
        assert "25/100" in result
        assert "HIGH" in result

    def test_missing_verdict_returns_empty(self):
        """Report without VALUE_TRAP_BLOCK fields returns empty string."""
        from src.agents import extract_value_trap_verdict

        result = extract_value_trap_verdict(
            "Some narrative text without structured block"
        )
        assert result == ""

    def test_empty_report_returns_empty(self):
        """Empty or None input returns empty string."""
        from src.agents import extract_value_trap_verdict

        assert extract_value_trap_verdict("") == ""
        assert extract_value_trap_verdict(None) == ""

    def test_missing_trap_risk_still_works(self):
        """SCORE + VERDICT present but TRAP_RISK missing → still produces header."""
        from src.agents import extract_value_trap_verdict

        report = "SCORE: 60\nVERDICT: WATCHABLE"
        result = extract_value_trap_verdict(report)
        assert "WATCHABLE" in result
        assert "60/100" in result
        assert "N/A" in result
