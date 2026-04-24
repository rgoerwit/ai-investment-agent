from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.consultant_nodes import create_auditor_node, create_legal_counsel_node
from src.validators.red_flag_detector import RedFlagDetector


class TestArtifactFallbacks:
    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_legal_counsel_failure_preserves_conservative_fallback(
        self, mock_get_prompt
    ):
        mock_get_prompt.return_value = SimpleNamespace(system_message="legal prompt")

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
        mock_get_prompt.return_value = SimpleNamespace(system_message="auditor prompt")

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
    async def test_auditor_param_error_retries_with_fallback_llm(self, mock_get_prompt):
        mock_get_prompt.return_value = SimpleNamespace(system_message="auditor prompt")

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
        mock_get_prompt.return_value = SimpleNamespace(system_message="auditor prompt")

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
        mock_get_prompt.return_value = SimpleNamespace(system_message="auditor prompt")

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
