from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import AIMessage, ToolMessage

from src.agents import AgentState
from src.tooling.runtime import TOOL_SERVICE, ToolInvocation

logger = structlog.get_logger(__name__)


def create_agent_tool_node(tools: list, agent_key: str):
    """
    Create a tool execution node that only processes tool_calls from a specific agent.
    """
    tool_names = {tool.name for tool in tools}
    tools_by_name = {tool.name: tool for tool in tools}

    def _tool_name(tc: dict[str, Any]) -> str:
        return tc.get("name") or tc.get("function", {}).get("name", "")

    def _tool_args(tc: dict[str, Any]) -> dict[str, Any]:
        args = tc.get("args")
        if args is None:
            args = tc.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return args if isinstance(args, dict) else {}

    async def agent_tool_node(state: AgentState, config) -> dict:
        """Execute tools for a specific agent by filtering messages."""
        messages = state.get("messages", [])

        target_message = None
        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                if getattr(msg, "name", None) != agent_key:
                    continue

                msg_tool_names = {_tool_name(tc) for tc in msg.tool_calls}
                if msg_tool_names & tool_names:
                    target_message = msg
                    break

        if target_message is None:
            logger.warning(
                "agent_tool_node_no_matching_message",
                agent_key=agent_key,
                tool_names=list(tool_names),
                message="No AIMessage found with tool_calls for this agent's tools",
            )
            return {"messages": []}

        logger.debug(
            "agent_tool_node_executing",
            agent_key=agent_key,
            tool_calls=[_tool_name(tc) for tc in target_message.tool_calls],
            tool_call_count=len(target_message.tool_calls),
            total_messages=len(messages),
        )

        result_messages: list[ToolMessage] = []
        for tc in target_message.tool_calls:
            tool_name = _tool_name(tc)
            tool_args = _tool_args(tc)
            tool_id = tc.get("id", tool_name)
            tool_fn = tools_by_name.get(tool_name)

            if not tool_fn:
                msg = ToolMessage(
                    content=f"Error: Unknown tool '{tool_name}'",
                    tool_call_id=tool_id,
                    name=tool_name,
                    status="error",
                )
                msg.additional_kwargs = {
                    "agent_key": agent_key,
                    "blocked": False,
                    "findings": [],
                }
                result_messages.append(msg)
                continue

            try:
                invocation = ToolInvocation(
                    name=tool_name,
                    args=tool_args,
                    source="toolnode",
                    agent_key=agent_key,
                )
                tool_result = await TOOL_SERVICE.execute(
                    invocation,
                    runner=lambda args, tool=tool_fn: tool.ainvoke(args),
                )
                msg = ToolMessage(
                    content=str(tool_result.value),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                msg.additional_kwargs = {
                    "agent_key": agent_key,
                    "blocked": tool_result.blocked,
                    "findings": tool_result.findings or [],
                }
                result_messages.append(msg)
            except Exception as exc:
                msg = ToolMessage(
                    content=f"Error: {exc}",
                    tool_call_id=tool_id,
                    name=tool_name,
                    status="error",
                )
                msg.additional_kwargs = {
                    "agent_key": agent_key,
                    "blocked": False,
                    "findings": [],
                }
                result_messages.append(msg)

        result = {"messages": result_messages}

        result_msg_count = len(result.get("messages", []))
        expected_count = len(target_message.tool_calls)
        logger.debug(
            "agent_tool_node_results",
            agent_key=agent_key,
            result_message_count=result_msg_count,
            tool_call_count=expected_count,
        )

        if result_msg_count != expected_count:
            logger.error(
                "agent_tool_node_message_mismatch",
                agent_key=agent_key,
                expected_tool_calls=expected_count,
                received_results=result_msg_count,
                tool_calls_requested=[
                    _tool_name(tc) for tc in target_message.tool_calls
                ],
                message="Not all tool calls resulted in ToolMessages. Agent may receive incomplete data.",
            )

        return result

    return agent_tool_node
