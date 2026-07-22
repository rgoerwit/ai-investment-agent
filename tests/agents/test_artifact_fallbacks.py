from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from src.agents.consultant_nodes import create_auditor_node, create_legal_counsel_node
from src.validators.red_flag_detector import RedFlagDetector


class TestArtifactFallbacks:
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
    async def test_legal_counsel_failure_preserves_conservative_fallback(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(
            system_message="legal prompt", agent_name="Legal Counsel"
        )

        # The manual loop invokes llm.ainvoke directly — mock the LLM to fail.
        mock_llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=RuntimeError("dns failure")),
            model_name="gemini-3-flash-preview",
        )

        node = create_legal_counsel_node(mock_llm, [])
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
        assert risks["pfic_status"] == "UNCERTAIN"
        assert "Legal counsel unavailable" in risks["pfic_evidence"]

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
        assert status["error_kind"] == "application_error"
        assert "CONTEXT_LIMIT_EXCEEDED" in result["auditor_report"]
        assert "FORENSIC_DATA_BLOCK" in result["auditor_report"]

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
