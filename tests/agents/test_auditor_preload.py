"""Tests for the auditor metrics preload (v2.11) and parallel tool rounds.

The preload replaces the auditor's own get_financial_metrics tool rounds with
one deterministic hook-chain call injected into the first HumanMessage; the
loop now executes a round's tool calls concurrently. Both are fail-open: any
preload failure yields an unmodified first message and the prompt's fallback
tool budget covers it.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.consultant_nodes import (
    _AUDITOR_SNAPSHOT_MAX_CHARS,
    _preload_metrics_snapshot,
    create_auditor_node,
)

_SNAPSHOT_LABEL = "PRE-LOADED AGGREGATOR SNAPSHOT"
_UNTRUSTED_MARKER = "--- BEGIN UNTRUSTED DATA [financial_api] ---"


def _metrics_tool(payload: str):
    return SimpleNamespace(
        name="get_financial_metrics",
        ainvoke=AsyncMock(return_value=payload),
    )


def _service(result=None, side_effect=None):
    return SimpleNamespace(
        execute=AsyncMock(return_value=result, side_effect=side_effect)
    )


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
