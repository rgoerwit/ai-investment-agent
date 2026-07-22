"""Tests for the auditor metrics preload (v2.11) and parallel tool rounds.

The preload replaces the auditor's own get_financial_metrics tool rounds with
one deterministic hook-chain call injected into the first HumanMessage; the
loop now executes a round's tool calls concurrently. Both are fail-open: any
preload failure yields an unmodified first message and the prompt's fallback
tool budget covers it.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.consultant_nodes import (
    _AUDITOR_SNAPSHOT_MAX_CHARS,
    _preload_metrics_snapshot,
    create_auditor_node,
)
from src.forensic_budget import AuditorBudgetPolicy

_SNAPSHOT_LABEL = "PRE-LOADED AGGREGATOR SNAPSHOT"
_UNTRUSTED_MARKER = "--- BEGIN UNTRUSTED DATA [financial_api] ---"
_FULL_RUN_0719_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "full_run_auditor_discard_repros.json"
)


def _metrics_tool(payload: str):
    return SimpleNamespace(
        name="get_financial_metrics",
        ainvoke=AsyncMock(return_value=payload),
    )


def _service(result=None, side_effect=None):
    return SimpleNamespace(
        execute=AsyncMock(return_value=result, side_effect=side_effect)
    )


@pytest.mark.parametrize(
    "repro",
    json.loads(_FULL_RUN_0719_FIXTURE.read_text()),
    ids=lambda repro: Path(repro["source_artifact"]).stem,
)
def test_full_run_0719_fixture_reproduces_evidence_discard_shape(repro):
    """Pin compact snapshots of the real full-run repros, never quick artifacts."""
    assert repro["quick_mode"] is False
    assert repro["evidence_chars"] > 0
    assert "not provided" in repro["auditor_report_excerpt"].lower()


class TestPreloadMetricsSnapshot:
    @pytest.mark.asyncio
    async def test_happy_path_returns_wrapped_block(self):
        payload = json.dumps({"currentPrice": 2859.0, "currency": "JPY"})
        service = _service(SimpleNamespace(value=payload, blocked=False))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot(
                "2503.T", {"get_financial_metrics": _metrics_tool(payload)}
            )

        assert _SNAPSHOT_LABEL in block
        assert _UNTRUSTED_MARKER in block
        assert '"currentPrice": 2859.0' in block
        invocation = service.execute.await_args.args[0]
        assert invocation.name == "get_financial_metrics"
        assert invocation.args == {"ticker": "2503.T"}
        assert invocation.source == "auditor"

    @pytest.mark.asyncio
    async def test_error_payload_skips_injection(self):
        service = _service(SimpleNamespace(value='{"error": "no data"}', blocked=False))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot(
                "2503.T", {"get_financial_metrics": _metrics_tool("{}")}
            )
        assert block == ""

    @pytest.mark.asyncio
    async def test_blocked_result_skips_injection(self):
        service = _service(SimpleNamespace(value="TOOL_BLOCKED: policy", blocked=True))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot(
                "2503.T", {"get_financial_metrics": _metrics_tool("{}")}
            )
        assert block == ""

    @pytest.mark.asyncio
    async def test_non_json_payload_skips_injection(self):
        service = _service(SimpleNamespace(value="TOOL_BLOCKED: policy", blocked=False))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot(
                "2503.T", {"get_financial_metrics": _metrics_tool("{}")}
            )
        assert block == ""

    @pytest.mark.asyncio
    async def test_oversized_payload_is_truncated(self):
        payload = json.dumps({"note": "x" * (_AUDITOR_SNAPSHOT_MAX_CHARS + 5_000)})
        service = _service(SimpleNamespace(value=payload, blocked=False))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot(
                "2503.T", {"get_financial_metrics": _metrics_tool(payload)}
            )
        assert _SNAPSHOT_LABEL in block
        assert payload[:100] in block
        assert payload[-50:] not in block

    @pytest.mark.asyncio
    async def test_tool_absent_makes_no_service_call(self):
        service = _service(SimpleNamespace(value="{}", blocked=False))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot("2503.T", {})
        assert block == ""
        service.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_raising_is_contained(self):
        service = _service(side_effect=RuntimeError("hook exploded"))
        with patch(
            "src.agents.consultant_nodes.get_current_tool_service",
            return_value=service,
        ):
            block = await _preload_metrics_snapshot(
                "2503.T", {"get_financial_metrics": _metrics_tool("{}")}
            )
        assert block == ""


class TestAuditorNodeWithPreload:
    def _prompt(self):
        return SimpleNamespace(
            system_message="auditor prompt", agent_name="Forensic Auditor"
        )

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_snapshot_lands_in_first_human_message(self, mock_get_prompt):
        mock_get_prompt.return_value = self._prompt()
        payload = json.dumps({"currentPrice": 2859.0, "currency": "JPY"})
        tools = [_metrics_tool(payload)]

        bound_stub = SimpleNamespace(model_name="gpt-5-mini")
        mock_llm = SimpleNamespace(
            model_name="gpt-5-mini",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        final_resp = SimpleNamespace(content="STATUS: CLEAN", tool_calls=None)
        invoke_mock = AsyncMock(return_value=final_resp)
        service = _service(SimpleNamespace(value=payload, blocked=False))

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=invoke_mock,
            ),
            patch(
                "src.agents.consultant_nodes.get_current_tool_service",
                return_value=service,
            ),
            patch(
                "src.agents.consultant_nodes.validate_required_output",
                return_value={"ok": True, "missing": []},
            ),
        ):
            node = create_auditor_node(mock_llm, tools)
            result = await node(
                {
                    "company_of_interest": "2503.T",
                    "company_name": "Kirin Holdings",
                    "company_name_resolved": True,
                },
                {},
            )

        llm_input = invoke_mock.await_args_list[0].args[1]
        human = next(m for m in llm_input if isinstance(m, HumanMessage))
        assert _SNAPSHOT_LABEL in human.content
        assert _UNTRUSTED_MARKER in human.content
        assert '"currentPrice": 2859.0' in human.content
        assert result["artifact_statuses"]["auditor_report"]["ok"] is True

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_preload_failure_leaves_message_unchanged(self, mock_get_prompt):
        mock_get_prompt.return_value = self._prompt()
        tools = [_metrics_tool("{}")]

        bound_stub = SimpleNamespace(model_name="gpt-5-mini")
        mock_llm = SimpleNamespace(
            model_name="gpt-5-mini",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        final_resp = SimpleNamespace(content="STATUS: CLEAN", tool_calls=None)
        invoke_mock = AsyncMock(return_value=final_resp)
        service = _service(side_effect=RuntimeError("hook exploded"))

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=invoke_mock,
            ),
            patch(
                "src.agents.consultant_nodes.get_current_tool_service",
                return_value=service,
            ),
            patch(
                "src.agents.consultant_nodes.validate_required_output",
                return_value={"ok": True, "missing": []},
            ),
        ):
            node = create_auditor_node(mock_llm, tools)
            result = await node(
                {
                    "company_of_interest": "2503.T",
                    "company_name": "Kirin Holdings",
                    "company_name_resolved": True,
                },
                {},
            )

        llm_input = invoke_mock.await_args_list[0].args[1]
        human = next(m for m in llm_input if isinstance(m, HumanMessage))
        assert _SNAPSHOT_LABEL not in human.content
        assert human.content.endswith("Perform a forensic audit using your tools.")
        assert result["artifact_statuses"]["auditor_report"]["ok"] is True

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_parallel_round_preserves_order_and_contains_failures(
        self, mock_get_prompt
    ):
        """A multi-tool round yields ordered ToolMessages; one failure doesn't
        cancel the sibling call (per-task containment, no gather side effects)."""
        mock_get_prompt.return_value = self._prompt()
        tools = [
            SimpleNamespace(name="get_news", ainvoke=AsyncMock()),
            SimpleNamespace(name="search_foreign_sources", ainvoke=AsyncMock()),
        ]

        bound_stub = SimpleNamespace(model_name="gpt-5-mini")
        mock_llm = SimpleNamespace(
            model_name="gpt-5-mini",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        round_resp = SimpleNamespace(
            content="",
            tool_calls=[
                {"name": "get_news", "args": {"q": "Kirin"}, "id": "c1"},
                {"name": "search_foreign_sources", "args": {"q": "キリン"}, "id": "c2"},
            ],
        )
        final_resp = SimpleNamespace(content="STATUS: CLEAN", tool_calls=None)
        invoke_mock = AsyncMock(side_effect=[round_resp, final_resp])

        async def _execute(invocation, runner=None):
            if invocation.name == "get_news":
                raise RuntimeError("news backend down")
            return SimpleNamespace(value="NATIVE FILING DATA", blocked=False)

        service = SimpleNamespace(execute=AsyncMock(side_effect=_execute))

        with (
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=invoke_mock,
            ),
            patch(
                "src.agents.consultant_nodes.get_current_tool_service",
                return_value=service,
            ),
            patch(
                "src.agents.consultant_nodes.validate_required_output",
                return_value={"ok": True, "missing": []},
            ),
        ):
            node = create_auditor_node(mock_llm, tools)
            result = await node(
                {
                    "company_of_interest": "2503.T",
                    "company_name": "Kirin Holdings",
                    "company_name_resolved": True,
                },
                {},
            )

        second_input = invoke_mock.await_args_list[1].args[1]
        tool_msgs = [m for m in second_input if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
        assert tool_msgs[0].content.startswith("TOOL_ERROR:")
        assert tool_msgs[1].content == "NATIVE FILING DATA"
        assert result["artifact_statuses"]["auditor_report"]["ok"] is True

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_loop_caps_at_configured_tool_rounds(self, mock_get_prompt):
        """Economic backstop: a model that keeps emitting tool calls gets at
        most 2 default tool rounds (3 LLM calls), while per-tool caps still apply."""
        mock_get_prompt.return_value = self._prompt()
        tools = [SimpleNamespace(name="get_news", ainvoke=AsyncMock())]

        bound_stub = SimpleNamespace(model_name="gpt-5-mini")
        mock_llm = SimpleNamespace(
            model_name="gpt-5-mini",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        greedy_resp = SimpleNamespace(
            content="STATUS: PARTIAL_DATA",
            tool_calls=[{"name": "get_news", "args": {"q": "x"}, "id": "c1"}],
        )
        # The model never stops asking for tools; the cap must cut it off.
        invoke_mock = AsyncMock(return_value=greedy_resp)
        service = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(value="DATA", blocked=False))
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
            patch(
                "src.agents.consultant_nodes.validate_required_output",
                return_value={"ok": True, "missing": []},
            ),
        ):
            node = create_auditor_node(mock_llm, tools)
            await node(
                {
                    "company_of_interest": "2503.T",
                    "company_name": "Kirin Holdings",
                    "company_name_resolved": True,
                },
                {},
            )

        assert invoke_mock.await_count == 3
        assert service.execute.await_count == 1

    @pytest.mark.asyncio
    @patch("src.prompts.get_prompt")
    async def test_third_tool_round_failure_reaches_forced_synthesis(
        self, mock_get_prompt
    ):
        """A failed final permitted tool round remains synthesis evidence.

        Regression for the 2026-07-19 empty-draft incidents: the bounded loop
        must execute the configured third round, retain its ToolMessage, and
        then call the unbound model for a real final report.
        """
        mock_get_prompt.return_value = self._prompt()
        tool = SimpleNamespace(name="custom_forensic_check", ainvoke=AsyncMock())
        bound_stub = SimpleNamespace(model_name="gpt-5-mini-bound")
        mock_llm = SimpleNamespace(
            model_name="gpt-5-mini",
            bind_tools=MagicMock(return_value=bound_stub),
        )
        tool_responses = [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "custom_forensic_check",
                        "args": {"round": round_number},
                        "id": f"c{round_number}",
                    }
                ],
            )
            for round_number in (1, 2, 3)
        ]
        final_response = SimpleNamespace(
            content=(
                "FORENSIC_DATA_BLOCK:\n"
                "STATUS: PARTIAL_DATA\n"
                "ANOMALIES: Third-round tool failed; prior evidence retained.\n"
                "VERDICT: Complete the unavailable check before clearance.\n"
            ),
            tool_calls=None,
        )
        invoke_mock = AsyncMock(side_effect=[*tool_responses, final_response])
        service = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    SimpleNamespace(value="ROUND_ONE", blocked=False),
                    SimpleNamespace(value="ROUND_TWO", blocked=False),
                    RuntimeError("third pass failure"),
                ]
            )
        )
        policy = AuditorBudgetPolicy(
            search_calls=3,
            document_calls=2,
            filing_calls=1,
            metrics_calls=1,
            news_calls=1,
            calculation_calls=2,
            max_document_bytes=15_000_000,
            max_document_pages=250,
            max_selected_pages=12,
            max_evidence_chars=35_000,
            max_tool_iterations=3,
            max_llm_calls=5,
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
            patch(
                "src.forensic_budget.AuditorBudgetPolicy.from_settings",
                return_value=policy,
            ),
        ):
            node = create_auditor_node(mock_llm, [tool])
            result = await node(
                {
                    "company_of_interest": "TEST.T",
                    "company_name": "Test Company",
                    "company_name_resolved": True,
                },
                {},
            )

        assert invoke_mock.await_count == 4
        assert all(
            call.args[0] is bound_stub for call in invoke_mock.await_args_list[:3]
        )
        assert invoke_mock.await_args_list[3].args[0] is mock_llm
        final_input = invoke_mock.await_args_list[3].args[1]
        tool_messages = [m for m in final_input if isinstance(m, ToolMessage)]
        assert tool_messages[-1].content == "TOOL_ERROR: RuntimeError"
        status = result["artifact_statuses"]["auditor_report"]
        assert status["ok"] is True
        telemetry = result["auditor_budget"]
        assert telemetry["tool_rounds_used"] == 3
        assert telemetry["forced_synthesis_used"] is True
        assert telemetry["failed_tools"] == ["custom_forensic_check"]
        assert telemetry["synthesis_evidence_chars"] > 0
