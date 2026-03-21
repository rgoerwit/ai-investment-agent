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
