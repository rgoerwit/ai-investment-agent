import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from src.agents.consultant_nodes import (
    _invoke_consultant_with_deadline,
    create_auditor_node,
    create_legal_counsel_node,
)
from src.agents.verdict_policy import maybe_demote_buy_on_blocking_flags
from src.runtime_diagnostics import get_artifact_status
from src.validators.red_flag_detector import RedFlagDetector


class TestArtifactFallbacks:
    @pytest.mark.asyncio
    async def test_consultant_call_timeout_floor_uses_actual_provider(self):
        response = object()
        with (
            patch(
                "src.agents.consultant_nodes.floor_llm_hard_timeout",
                return_value=90.0,
            ) as floor,
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=AsyncMock(return_value=response),
            ),
        ):
            result = await _invoke_consultant_with_deadline(
                object(),
                [],
                context="External Consultant",
                provider="google",
                model_name="gemini-test",
                ticker="TEST",
                deadline=time.monotonic() + 30.0,
            )

        assert result is response
        assert floor.call_args.kwargs["provider"] == "google"

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_legal_counsel_binds_tools_and_preserves_third_pass_failure(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="legal prompt", agent_name="Legal Counsel"
        )
        tool = SimpleNamespace(
            name="search_legal_tax_disclosures",
            ainvoke=AsyncMock(),
        )
        bound_stub = SimpleNamespace(model_name="gemini-bound")
        mock_llm = SimpleNamespace(
            model_name="gemini-3-flash-preview",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        tool_responses = [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "search_legal_tax_disclosures",
                        "args": {"ticker": "TEST.T", "query": str(round_number)},
                        "id": f"legal-{round_number}",
                    }
                ],
            )
            for round_number in (1, 2, 3, 4)
        ]
        final_response = SimpleNamespace(
            content=(
                '{"pfic_status":"UNCERTAIN",'
                '"pfic_evidence":"Third-pass disclosure search failed; '
                'earlier evidence retained.","vie_structure":"N/A"}'
            ),
            tool_calls=None,
        )
        invoke_mock = AsyncMock(side_effect=[*tool_responses, final_response])
        service = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(value="FIRST", blocked=False),
                    SimpleNamespace(value="SECOND", blocked=False),
                    RuntimeError("third pass failure"),
                    SimpleNamespace(value="FOURTH", blocked=False),
                ]
            )
        )

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=invoke_mock,
            ),
            patch(
                "src.agents.consultant_nodes.get_current_tool_service",
                return_value=service,
            ),
        ):
            node = create_legal_counsel_node(mock_llm, [tool])
            result = await node(
                {
                    "company_of_interest": "TEST.T",
                    "company_name": "Test Company",
                    "company_name_resolved": True,
                    "raw_fundamentals_data": ("Sector: Industrials\nCountry: Japan"),
                },
                {},
            )

        mock_llm.bind_tools.assert_called_once_with([tool])
        assert all(
            call.args[0] is bound_stub for call in invoke_mock.await_args_list[:4]
        )
        assert invoke_mock.await_args_list[4].args[0] is mock_llm
        final_input = invoke_mock.await_args_list[4].args[1]
        assert any(
            isinstance(message, ToolMessage)
            and message.content == "TOOL_ERROR: RuntimeError"
            for message in final_input
        )
        assert result["artifact_statuses"]["legal_report"]["ok"] is True
        assert "Third-pass disclosure search failed" in result["legal_report"]

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_legal_counsel_failure_preserves_unassessed_fallback(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="legal prompt", agent_name="Legal Counsel"
        )

        provider_payload = "team-id SECRET-PROVIDER-PAYLOAD"
        # The manual loop invokes llm.ainvoke directly — mock the LLM to fail.
        mock_llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=RuntimeError(
                    "Error code: 403 - Your team "
                    f"{provider_payload} has used all available credits."
                )
            ),
            model_name="grok-4.6",
        )

        node = create_legal_counsel_node(mock_llm, [])
        with patch("src.agents.consultant_nodes.logger") as mock_logger:
            result = await node(
                {
                    "company_of_interest": "TOTL.JK",
                    "company_name": "Total Indonesia",
                    "company_name_resolved": True,
                    "raw_fundamentals_data": "Sector: Finance\nCountry: Indonesia",
                },
                {},
            )

        status = result["artifact_statuses"]["legal_report"]
        risks = RedFlagDetector.extract_legal_risks(result["legal_report"])

        assert status["complete"] is True
        assert status["ok"] is False
        assert status["error_kind"] == "quota_error"
        assert status["retryable"] is False
        assert provider_payload not in repr(result)
        assert risks["pfic_status"] is None
        assert "Legal counsel unavailable" in risks["pfic_evidence"]
        assert risks["vie_structure"] is None
        assert risks["cmic_status"] is None
        assert "Legal counsel unavailable" in risks["cmic_evidence"]

        flags = RedFlagDetector.detect_legal_flags(
            risks,
            "TOTL.JK",
            artifact_status=get_artifact_status(result, "legal_report"),
        )
        assert [flag["type"] for flag in flags] == ["LEGAL_COUNSEL_UNAVAILABLE"]
        assert flags[0]["risk_penalty"] == 0.0
        assert flags[0]["blocks_buy"] is True
        assert "PFIC, VIE, CMIC" in flags[0]["detail"]
        unavailable_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args and call.args[0] == "artifact_unavailable"
        ]
        assert len(unavailable_calls) == 1
        assert unavailable_calls[0].kwargs["artifact"] == "legal_report"
        assert "exc_info" not in unavailable_calls[0].kwargs

        pm_output = """# PORTFOLIO MANAGER VERDICT: BUY
Actual Decision: BUY
<PM_BLOCK>
VERDICT: BUY
</PM_BLOCK>"""
        demoted, changed = maybe_demote_buy_on_blocking_flags(
            pm_output,
            red_flags=flags,
            ticker="TOTL.JK",
        )
        assert changed is True
        assert "VERDICT: HOLD" in demoted
        assert "LEGAL_COUNSEL_UNAVAILABLE" in demoted

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_legal_preflight_failure_does_not_abort_legal_analysis(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="legal prompt", agent_name="Legal Counsel"
        )
        response = SimpleNamespace(
            content='{"pfic_status":"CLEAN","vie_structure":"N/A"}',
            tool_calls=None,
        )
        mock_llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=response),
            model_name="gemini-3-flash-preview",
        )

        with patch(
            "src.agents.consultant_nodes.preload_capital_structure_evidence",
            new=AsyncMock(side_effect=RuntimeError("unexpected adapter failure")),
        ):
            result = await create_legal_counsel_node(mock_llm, [])(
                {
                    "company_of_interest": "TEST",
                    "company_name": "Test Company",
                    "company_name_resolved": True,
                    "raw_fundamentals_data": "Sector: Industrials\nCountry: USA",
                },
                {},
            )

        status = result["artifact_statuses"]["legal_report"]
        capital = json.loads(result["legal_report"])["capital_structure"]
        assert status["ok"] is True
        assert capital["coverage_status"] == "SEARCH_FAILED"
        assert capital["classification"] == "UNRESOLVED"

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_context_limit_preserves_graceful_report(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )

        # The manual loop invokes llm.ainvoke directly — mock the LLM to raise a
        # context-limit error so the graceful fallback path is exercised.
        mock_llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=RuntimeError("maximum context length exceeded")
            ),
            model_name="gpt-4o",
        )

        node = create_auditor_node(mock_llm, [])
        result = await node(
            {
                "company_of_interest": "TOTL.JK",
                "company_name": "Total Indonesia",
                "company_name_resolved": True,
            },
            {},
        )

        status = result["artifact_statuses"]["auditor_report"]

        assert status["complete"] is True
        assert status["ok"] is False
        assert status["error_kind"] == "bad_request"
        assert "CONTEXT_LIMIT_EXCEEDED" in result["auditor_report"]
        assert "FORENSIC_DATA_BLOCK" in result["auditor_report"]

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_account_limit_is_concise_unavailable_artifact(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )
        limit_error = RuntimeError(
            "Error code: 403 - Your team team-id has either used all available "
            "credits or reached its monthly spending limit."
        )
        mock_llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=limit_error), model_name="grok-4.6"
        )

        with patch("src.agents.consultant_nodes.logger") as mock_logger:
            result = await create_auditor_node(mock_llm, [])(
                {
                    "company_of_interest": "LIMIT.TW",
                    "company_name": "Limit Test",
                    "company_name_resolved": True,
                },
                {},
            )

        status = result["artifact_statuses"]["auditor_report"]
        assert status["complete"] is True
        assert status["ok"] is False
        assert status["error_kind"] == "quota_error"
        assert "account credits or spending limit" in status["message"]
        assert "team-id" not in result["auditor_report"]
        assert "STATUS: UNAVAILABLE" in result["auditor_report"]
        assert "FAILURE_KIND=QUOTA_ERROR" in result["auditor_report"]
        assert not any(
            call.args and call.args[0] == "auditor_error"
            for call in mock_logger.error.call_args_list
        )
        unavailable_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args and call.args[0] == "artifact_unavailable"
        ]
        assert len(unavailable_calls) == 1
        assert unavailable_calls[0].kwargs["artifact"] == "auditor_report"
        assert unavailable_calls[0].kwargs["failure_kind"] == "quota_error"

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_binds_tools_and_runs_tool_loop(self, mock_get_prompt):
        """Regression: the auditor LLM must receive its tools so it can call them.

        Previously the loop invoked the unbound LLM, so the model never saw the
        tool schemas, never emitted tool_calls, and returned INSUFFICIENT_DATA.
        """
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )

        bound_stub = SimpleNamespace(model_name="gpt-5-mini")
        mock_llm = SimpleNamespace(
            model_name="gpt-5-mini",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        tools = [SimpleNamespace(name="get_news", ainvoke=AsyncMock())]

        # First turn emits a tool call; second turn returns the final report.
        resp_with_tool = SimpleNamespace(
            content="",
            tool_calls=[{"name": "get_news", "args": {"q": "TSMC"}, "id": "c1"}],
        )
        final_resp = SimpleNamespace(
            content="STATUS: CLEAN\nNo anomalies detected.", tool_calls=None
        )
        invoke_mock = AsyncMock(side_effect=[resp_with_tool, final_resp])
        tool_service = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(value="FILING DATA"))
        )

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling", new=invoke_mock
            ),
            patch(
                "src.agents.consultant_nodes.get_current_tool_service",
                return_value=tool_service,
            ),
            patch(
                "src.agents.consultant_nodes.validate_required_output",
                return_value={"ok": True, "missing": []},
            ),
        ):
            node = create_auditor_node(mock_llm, tools)
            result = await node(
                {
                    "company_of_interest": "2330.TW",
                    "company_name": "TSMC",
                    "company_name_resolved": True,
                },
                {},
            )

        # Tools were bound to the LLM, and the loop drove the *bound* runnable.
        mock_llm.bind_tools.assert_called_once_with(tools)
        assert invoke_mock.await_args_list[0].args[0] is bound_stub
        # The emitted tool call was executed through the tool service.
        tool_service.execute.assert_awaited_once()
        status = result["artifact_statuses"]["auditor_report"]
        assert status["complete"] is True
        assert status["ok"] is True
        assert "CLEAN" in result["auditor_report"]

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_param_error_retries_with_fallback_llm(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )

        initial_llm = SimpleNamespace(model_name="gpt-4o")
        fallback_llm = SimpleNamespace(model_name="gpt-4o")
        final_response = SimpleNamespace(content="retry success", tool_calls=None)
        invoke_mock = AsyncMock(
            side_effect=[RuntimeError("Unsupported value"), final_response]
        )

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling", new=invoke_mock
        ):
            with patch(
                "langchain_openai.ChatOpenAI", return_value=fallback_llm
            ) as mock_chat:
                node = create_auditor_node(initial_llm, [])
                result = await node(
                    {
                        "company_of_interest": "TOTL.JK",
                        "company_name": "Total Indonesia",
                        "company_name_resolved": True,
                    },
                    {},
                )

        status = result["artifact_statuses"]["auditor_report"]

        assert status["complete"] is True
        assert status["ok"] is True
        assert result["auditor_report"] == "retry success"
        assert invoke_mock.await_count == 2
        mock_chat.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_fallback_failure_stamps_fallback_provider(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )
        initial_llm = SimpleNamespace(model_name="gpt-4o")
        fallback_llm = SimpleNamespace(model_name="grok-4.6")
        invoke_mock = AsyncMock(
            side_effect=[
                RuntimeError("Unsupported value"),
                RuntimeError("Error code: 403 - account used all available credits"),
            ]
        )

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling", new=invoke_mock
            ),
            patch(
                "src.agents.consultant_nodes._create_openai_responses_fallback_llm",
                return_value=fallback_llm,
            ),
        ):
            result = await create_auditor_node(initial_llm, [])(
                {
                    "company_of_interest": "TOTL.JK",
                    "company_name": "Total Indonesia",
                    "company_name_resolved": True,
                },
                {},
            )

        status = result["artifact_statuses"]["auditor_report"]
        assert invoke_mock.await_count == 2
        assert status["provider"] == "xai"
        assert status["error_kind"] == "quota_error"
        assert status["retryable"] is False
        assert "STATUS: UNAVAILABLE" in result["auditor_report"]

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_repairs_recoverable_invalid_structure(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )

        initial_llm = SimpleNamespace(model_name="gpt-4o")
        initial_response = SimpleNamespace(
            content=(
                "## FORENSIC AUDITOR REPORT\n\n"
                "**STATUS**: INSUFFICIENT_DATA\n\n"
                "The primary filings and auditor report could not be verified.\n"
                "Data remains unavailable from authoritative source documents.\n"
            ),
            tool_calls=None,
        )
        repaired_response = SimpleNamespace(
            content=(
                "## FORENSIC AUDITOR REPORT\n\n"
                "STATUS: INSUFFICIENT_DATA\n\n"
                "FORENSIC_DATA_BLOCK:\n"
                "STATUS: INSUFFICIENT_DATA\n"
                "META: UNKNOWN | REPORT_DATE=UNKNOWN\n"
                "VERDICT: Unable to perform comprehensive forensic audit from "
                "verified primary source documents.\n"
            ),
            tool_calls=None,
        )

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling",
            new=AsyncMock(side_effect=[initial_response, repaired_response]),
        ) as invoke_mock:
            tool_service = SimpleNamespace(execute=AsyncMock())
            with patch(
                "src.agents.consultant_nodes.get_current_tool_service",
                return_value=tool_service,
            ):
                node = create_auditor_node(initial_llm, [])
                result = await node(
                    {
                        "company_of_interest": "SKT.NZ",
                        "company_name": "Sky Network Television",
                        "company_name_resolved": True,
                    },
                    {},
                )

        status = result["artifact_statuses"]["auditor_report"]

        assert status["complete"] is True
        assert status["ok"] is True
        assert "VERDICT:" in result["auditor_report"]
        assert invoke_mock.await_count == 2
        tool_service.execute.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_auditor_unrecoverable_invalid_structure_logs_preview(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )

        initial_llm = SimpleNamespace(model_name="gpt-4o")
        invalid_response = SimpleNamespace(content="nonsense output", tool_calls=None)

        with patch(
            "src.agents.runtime.invoke_with_rate_limit_handling",
            new=AsyncMock(side_effect=[invalid_response, invalid_response]),
        ):
            with patch("src.agents.consultant_nodes.logger") as mock_logger:
                node = create_auditor_node(initial_llm, [])
                result = await node(
                    {
                        "company_of_interest": "BAD.TICKER",
                        "company_name": "Bad Ticker",
                        "company_name_resolved": True,
                    },
                    {},
                )

        status = result["artifact_statuses"]["auditor_report"]
        assert status["complete"] is True
        assert status["ok"] is False

        invalid_calls = [
            call
            for call in mock_logger.error.call_args_list
            if call.args[0] == "auditor_invalid_structure"
        ]
        assert invalid_calls
        assert invalid_calls[-1].kwargs["output_preview"] == "nonsense output"
