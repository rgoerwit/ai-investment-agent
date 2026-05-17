from __future__ import annotations

import pytest

from src.tooling.runtime import ToolCallBlocked, ToolInvocation
from src.tooling.tool_argument_policy import ToolArgumentPolicyHook
from tests.helpers.injection_corpus import load_corpus


def _call(
    *,
    name: str = "fetch_reference_content",
    payload: str,
    source: str = "editor",
) -> ToolInvocation:
    arg_name = "query" if name == "search_claim" else "url"
    return ToolInvocation(
        name=name,
        args={arg_name: payload},
        source=source,
        agent_key="security_test",
    )


@pytest.mark.security
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    load_corpus(category="egress_blocked"),
    ids=lambda case: case["id"],
)
async def test_editor_reference_policy_blocks_current_high_risk_urls(case):
    hook = ToolArgumentPolicyHook(mode="block")
    with pytest.raises(ToolCallBlocked):
        await hook.before(_call(payload=case["payload"]))


@pytest.mark.security
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    load_corpus(category="egress_query_blocked"),
    ids=lambda case: case["id"],
)
async def test_editor_search_policy_blocks_pasted_or_oversized_queries(case):
    hook = ToolArgumentPolicyHook(mode="block")
    with pytest.raises(ToolCallBlocked):
        await hook.before(_call(name="search_claim", payload=case["payload"]))


@pytest.mark.security
@pytest.mark.asyncio
async def test_non_editor_source_is_out_of_current_policy_scope():
    hook = ToolArgumentPolicyHook(mode="block")
    case = load_corpus(category="egress_blocked")[0]
    call = _call(payload=case["payload"], source="toolnode")

    result = await hook.before(call)

    assert result is call


@pytest.mark.security
@pytest.mark.asyncio
async def test_current_policy_allows_normal_looking_external_domain():
    hook = ToolArgumentPolicyHook(mode="block")
    case = load_corpus(category="egress_current_allow")[0]
    call = _call(payload=case["payload"])

    result = await hook.before(call)

    assert result is call


@pytest.mark.security
@pytest.mark.asyncio
async def test_opt_in_allowlist_blocks_normal_looking_non_allowlisted_domain():
    hook = ToolArgumentPolicyHook(
        mode="block",
        allowed_reference_domains=frozenset({"reuters.com"}),
    )
    case = load_corpus(category="egress_current_allow")[0]

    with pytest.raises(ToolCallBlocked):
        await hook.before(_call(payload=case["payload"]))


@pytest.mark.security
@pytest.mark.asyncio
async def test_opt_in_allowlist_permits_known_reference_domain():
    hook = ToolArgumentPolicyHook(
        mode="block",
        allowed_reference_domains=frozenset({"reuters.com"}),
    )
    case = load_corpus(category="egress_allowlist")[0]
    call = _call(payload=case["payload"])

    result = await hook.before(call)

    assert result is call
