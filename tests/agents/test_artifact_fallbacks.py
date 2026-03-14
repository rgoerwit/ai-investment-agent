from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.consultant_nodes import create_auditor_node, create_legal_counsel_node
from src.validators.red_flag_detector import RedFlagDetector


class TestArtifactFallbacks:
    @pytest.mark.asyncio
    @patch("src.agents.consultant_nodes.create_react_agent")
    @patch("src.prompts.get_prompt")
    async def test_legal_counsel_failure_preserves_conservative_fallback(
        self, mock_get_prompt, mock_create_react_agent
    ):
        mock_get_prompt.return_value = SimpleNamespace(system_message="legal prompt")
        mock_create_react_agent.return_value = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=RuntimeError("dns failure"))
        )

        node = create_legal_counsel_node(
            SimpleNamespace(model_name="gemini-3-flash-preview"),
            [],
        )
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
    @patch("src.agents.consultant_nodes.create_react_agent")
    @patch("src.prompts.get_prompt")
    async def test_auditor_context_limit_preserves_graceful_report(
        self, mock_get_prompt, mock_create_react_agent
    ):
        mock_get_prompt.return_value = SimpleNamespace(system_message="auditor prompt")
        mock_create_react_agent.return_value = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=RuntimeError("maximum context length exceeded")
            )
        )

        node = create_auditor_node(SimpleNamespace(model_name="gpt-4o"), [])
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
