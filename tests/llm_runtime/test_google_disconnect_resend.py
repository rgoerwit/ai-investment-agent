"""One SDK attempt still permits one physical resend on Google's aiohttp path.

``require_single_sdk_attempt`` pins ``HttpRetryOptions(attempts=1)``, so the
SDK's tenacity policy never retries. google-genai's ``_async_request_once``
nevertheless catches ``aiohttp.ServerDisconnectedError`` (and connector, OS and
auth transport errors), sleeps 1-10 s and sends the request again, outside that
policy. These tests drive the real Gemini chat transport through
``_agenerate`` with a fake *below* the SDK, at ``aiohttp.ClientSession._request``,
and count physical sends. The resend is bounded only by the caller's hard
deadline, which is also asserted.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from langchain_core.messages import HumanMessage

import src.llms as llms_mod
from src.async_utils import run_with_hard_timeout
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import invocation_scope, policy_for_seat

_MESSAGES = [HumanMessage(content="hi")]
_OK_BODY = {
    "candidates": [
        {
            "content": {"parts": [{"text": "ok"}], "role": "model"},
            "finishReason": "STOP",
        }
    ]
}


class _InstantSleepAsyncio:
    """The asyncio module as google-genai sees it, with an instant sleep."""

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(_seconds: float) -> None:
        return None


@pytest.fixture
def wire(monkeypatch) -> SimpleNamespace:
    """Script ``ClientSession._request``: disconnect, answer, or stall."""
    import httpx

    async def refuse(self, request):
        raise AssertionError(f"unexpected httpx request to {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)

    state = SimpleNamespace(sent=[], disconnects=1, stall=None)

    async def respond(self, method, url, **kwargs):
        del self, method, kwargs
        state.sent.append(str(url))
        if state.stall is not None:
            await state.stall.wait()
        if len(state.sent) <= state.disconnects:
            raise aiohttp.ServerDisconnectedError()
        text = json.dumps(_OK_BODY)

        async def read_text() -> str:
            return text

        return SimpleNamespace(
            status_code=200, body_segments=[{}], headers={}, text=read_text
        )

    monkeypatch.setattr(aiohttp.ClientSession, "_request", respond)
    return state


def _llm() -> Any:
    return llms_mod._TieredChatGoogleGenerativeAI(model="gemini-3.8-flash", api_key="k")


def _scope(seconds: float):
    policy = policy_for_seat(SEATS[SeatId.CONSULTANT], provider="google")
    return invocation_scope(policy, llm_timeouts.now() + seconds)


@pytest.mark.asyncio
async def test_a_server_disconnect_is_resent_once_inside_one_sdk_attempt(
    monkeypatch, wire
):
    import google.genai._api_client as api_client

    monkeypatch.setattr(api_client, "asyncio", _InstantSleepAsyncio())
    llm = _llm()
    assert llm.max_retries == 1  # the single SDK attempt the repo configures

    with _scope(200.0):
        result = await llm._agenerate(_MESSAGES)

    assert result.generations[0].message.content == "ok"
    assert len(wire.sent) == 2  # one SDK attempt, two physical sends


@pytest.mark.asyncio
async def test_two_disconnects_surface_after_exactly_two_sends(monkeypatch, wire):
    import google.genai._api_client as api_client

    monkeypatch.setattr(api_client, "asyncio", _InstantSleepAsyncio())
    wire.disconnects = 10

    with _scope(200.0), pytest.raises(Exception, match="Server disconnected"):
        await _llm()._agenerate(_MESSAGES)

    assert len(wire.sent) == 2  # the resend is not repeated; no SDK retry follows


@pytest.mark.asyncio
async def test_the_hard_deadline_cuts_off_the_pre_resend_sleep(wire):
    # The SDK's real 1-10 s sleep before the resend outlasts a 50 ms deadline.
    with _scope(200.0), pytest.raises(TimeoutError):
        await run_with_hard_timeout(
            _llm()._agenerate(_MESSAGES), timeout=0.05, label="google_resend"
        )

    assert len(wire.sent) == 1


@pytest.mark.asyncio
async def test_the_hard_deadline_cancels_a_stalled_send(wire):
    wire.stall = asyncio.Event()
    try:
        with _scope(200.0), pytest.raises(TimeoutError):
            await run_with_hard_timeout(
                _llm()._agenerate(_MESSAGES), timeout=0.05, label="google_stall"
            )
    finally:
        wire.stall.set()
        await asyncio.sleep(0)

    assert len(wire.sent) == 1
