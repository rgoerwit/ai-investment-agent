"""The article workflow runs async, under its own deadline, with owned retries.

Before October 2026 the writer called its model with a synchronous ``invoke``
on the event loop, outside every deadline (the analysis deadline's scope has
closed by the time ``--article`` runs), with SDK retries fixed on each client
(three Anthropic attempts of ``api_timeout`` each, per tier, up to three tiers)
and every fallback tier built and sent after any primary failure.

These tests drive real clients built by the seat factory. The fake sits at
``httpx2``'s client ``send`` -- below the vendor SDKs, so their own retry loops
run and are counted -- and the sync client's ``send`` blocks the thread like a
real request, so a sync call on the loop is visible as a stalled loop. Nothing
reaches the network.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

import src.config as config_module
from src.agents import runtime as runtime_mod
from src.agents.circuit_breaker import reset_circuit_breaker_for_tests
from src.agents.network_breaker import reset_network_breaker_singleton
from src.article_writer import (
    ArticleWorkflowExpired,
    ArticleWriter,
    article_workflow_deadline,
    run_article_workflow_step,
)
from src.config import Settings
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.adapters import transports
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.construction import (
    WriterSeatTier,
    build_model_for_seat,
    seat_policy_with,
)
from src.llm_runtime.rate_limits import reset_fallback_limiters_for_tests
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import (
    BudgetExhausted,
    current_enclosing_deadline,
    enclosing_deadline_at,
)
from src.service_tiers import (
    _reset_flex_capability_cache_for_tests,
    _reset_flex_health_for_tests,
)
from src.token_tracker import get_tracker

WRITER = SEATS[SeatId.ARTICLE_WRITER].callback_name
FALLBACK = SEATS[SeatId.ARTICLE_WRITER_REVIEW_FALLBACK].callback_name
MESSAGES = [SystemMessage(content="writer persona"), HumanMessage(content="write")]
STALL = "stall"


def _anthropic_body(text: str) -> dict[str, Any]:
    return {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _openai_body(text: str, path: str) -> dict[str, Any]:
    if path.endswith("/responses"):
        return {
            "id": "resp",
            "object": "response",
            "created_at": 0,
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    return {
        "id": "chat",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


class FakeVendors:
    """Replaces ``httpx2``'s client ``send``; records and scripts every request.

    A script step is a status code (an error response), ``STALL`` (wait until
    released) or ``200``. Each vendor answers with its own article text.
    """

    def __init__(self) -> None:
        self.sends: list[dict[str, Any]] = []
        self.sync_sends: list[dict[str, Any]] = []
        self.script: dict[str, list[Any]] = {"anthropic": [], "openai": []}
        self.delay = 0.0
        self.sync_block_seconds = 0.3
        self.release: asyncio.Event | None = None

    @staticmethod
    def _vendor(request: Any) -> str:
        return "anthropic" if "anthropic" in request.url.host else "openai"

    def _record(self, sink: list[dict[str, Any]], request: Any) -> str:
        vendor = self._vendor(request)
        body = json.loads(request.content or b"{}")
        timeout = request.extensions.get("timeout") or {}
        sink.append(
            {
                "vendor": vendor,
                "timeout": timeout.get("read"),
                "model": body.get("model"),
                "service_tier": body.get("service_tier", "<absent>"),
            }
        )
        return vendor

    def _response(self, vendor: str, request: Any, status: int = 200) -> Any:
        if status != 200:
            return httpx2.Response(
                status,
                json={"type": "error", "error": {"type": "api_error", "message": "x"}},
                request=request,
            )
        text = f"# {vendor.title()} Article\n\nWritten by {vendor}."
        body = (
            _anthropic_body(text)
            if vendor == "anthropic"
            else _openai_body(text, request.url.path)
        )
        return httpx2.Response(200, json=body, request=request)

    async def send(self, request: Any, **_kwargs: Any) -> Any:
        vendor = self._record(self.sends, request)
        step = self.script[vendor].pop(0) if self.script[vendor] else 200
        if step == STALL:
            if self.release is None:
                self.release = asyncio.Event()
            await self.release.wait()
            step = 200
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._response(vendor, request, int(step))

    def sync_send(self, request: Any, **_kwargs: Any) -> Any:
        # A synchronous client blocks its thread for the whole request; on the
        # event loop that parks every other task. Never the network.
        vendor = self._record(self.sync_sends, request)
        time.sleep(self.sync_block_seconds)
        return self._response(vendor, request)

    def count(self, vendor: str) -> int:
        return sum(1 for send in self.sends if send["vendor"] == vendor)

    @contextmanager
    def patched(self):
        with (
            patch.object(httpx2.AsyncClient, "send", self.send),
            patch.object(httpx2.Client, "send", self.sync_send),
        ):
            yield self


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_base_provider": "google",
        "llm_review_provider": "openai",
        "llm_regional_provider": "deepseek",
        "llm_writer_provider": "anthropic",
        "llm_operational_provider": "google",
        "llm_judge_provider": "google",
        "google_api_key": "g",
        "openai_api_key": "o",
        "claude_api_key": "a",
        "deepseek_api_key": "d",
        "anthropic_llm_prose_model": "claude-opus-4-8",
        "openai_service_tier": "auto",
        "google_service_tier": "standard",
        "llm_editor_mode": "required",
        "llm_require_review_independence": False,
        "llm_review_independence_waiver_reason": "article workflow tests",
        "llm_require_regional_independence": False,
        "llm_regional_independence_waiver_reason": "article workflow tests",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _build(seat_id: SeatId, settings: Settings, **kwargs: Any):
    model = build_model_for_seat(
        seat_id, settings=settings, plan=resolve_binding_plan(settings), **kwargs
    )
    # The process rate limiter's token bucket is not under test; its first
    # acquire would wait out a refill in real time.
    model.rate_limiter = None
    return model


class TierBuilds:
    """Lazy fallback tiers whose builds are counted (real construction)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.builds: list[str] = []
        self.chain_calls = 0

    def chain(self, **_kwargs: Any) -> list[WriterSeatTier]:
        self.chain_calls += 1

        def build() -> Any:
            self.builds.append("review_group")
            return _build(SeatId.ARTICLE_WRITER_REVIEW_FALLBACK, self.settings)

        return [WriterSeatTier("review_group", build)]


@pytest.fixture(autouse=True)
def workflow_config(monkeypatch):
    """Pin every input the budget reads; short admission floors for real time."""
    values = {
        "api_timeout": 300,
        "llm_attempts": 2,
        "llm_min_useful_attempt_seconds": 0.05,
        "llm_dispatch_margin_seconds": 0.01,
        "llm_circuit_breaker_enabled": False,
        "network_breaker_enabled": False,
        "article_workflow_timeout_seconds": 900.0,
    }
    for key, value in values.items():
        monkeypatch.setattr(config_module.config, key, value, raising=False)
    # Backoff is admitted and clamped by the runtime; its length is not under
    # test, so a permitted wait is zero.
    real_allowance = runtime_mod.retry_wait_allowance

    def no_wait(*args: Any, **kwargs: Any) -> float | None:
        allowed = real_allowance(*args, **kwargs)
        return None if allowed is None else 0.0

    monkeypatch.setattr(runtime_mod, "retry_wait_allowance", no_wait)
    monkeypatch.setattr(
        "src.article_writer.search_tavily_inspected", AsyncMock(return_value=None)
    )
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()
    reset_fallback_limiters_for_tests()
    _reset_flex_capability_cache_for_tests()
    _reset_flex_health_for_tests()
    get_tracker().reset()
    yield
    reset_circuit_breaker_for_tests()
    reset_network_breaker_singleton()
    reset_fallback_limiters_for_tests()
    _reset_flex_capability_cache_for_tests()
    _reset_flex_health_for_tests()
    get_tracker().reset()


@pytest.fixture
def vendors():
    fake = FakeVendors()
    with fake.patched():
        yield fake


def _writer(
    monkeypatch, settings: Settings, tiers: TierBuilds | None = None
) -> ArticleWriter:
    """An ArticleWriter on a real Anthropic client; fallback tiers counted."""

    def create(**kwargs: Any) -> Any:
        return _build(
            SeatId.ARTICLE_WRITER,
            settings,
            callbacks=kwargs.get("callbacks") or [],
            output_tokens=16_384,
        )

    monkeypatch.setattr("src.article_writer.create_writer_llm", create)
    chain: Callable[..., Any] = tiers.chain if tiers else MagicMock(return_value=[])
    monkeypatch.setattr("src.article_writer.writer_fallback_chain", chain)
    # Any primary failure enters the chain (the provider-scoped rule).
    monkeypatch.setattr(
        "src.article_writer.resolve_binding_plan",
        lambda _cfg: SimpleNamespace(schema="new"),
    )
    return ArticleWriter(samples_dir=None)


def _attempts(seat: str) -> list[Any]:
    return [a for a in get_tracker().call_attempts if a.agent_name == seat]


# ------------------------------------------------------------------ paths


class TestWriterPaths:
    @pytest.mark.asyncio
    async def test_primary_success_sends_one_request_and_builds_no_tier(
        self, monkeypatch, vendors
    ):
        tiers = TierBuilds(_settings())
        writer = _writer(monkeypatch, _settings(), tiers)
        assert writer.llm.max_retries == 0  # one SDK attempt per dispatch

        with article_workflow_deadline():
            article = await writer._invoke_writer(MESSAGES)

        assert article.startswith("# Anthropic Article")
        assert vendors.count("anthropic") == 1
        assert vendors.sync_sends == []
        assert tiers.chain_calls == 0 and tiers.builds == []
        # Sized at dispatch from the EDITORIAL policy: at most api_timeout.
        assert 0 < vendors.sends[0]["timeout"] <= 300
        assert writer.writer_fell_back is False
        [attempt] = _attempts(WRITER)
        assert attempt.status == "success"

    @pytest.mark.asyncio
    async def test_primary_failure_then_fallback_success(self, monkeypatch, vendors):
        settings = _settings()
        tiers = TierBuilds(settings)
        writer = _writer(monkeypatch, settings, tiers)
        vendors.script["anthropic"] = [500, 500]

        with article_workflow_deadline():
            article = await writer._invoke_writer(MESSAGES)

        assert article.startswith("# Openai Article")
        # Application attempts never exceed the configured allowance
        # (llm_attempts=2), and SDK retries do not multiply them: each
        # application attempt is exactly one request on the wire.
        assert vendors.count("anthropic") == 2
        assert vendors.count("openai") == 1
        assert tiers.builds == ["review_group"]
        assert writer.llm.max_retries == 0
        assert writer.writer_fell_back is True
        assert [a.status for a in _attempts(WRITER)] == ["failure", "failure"]
        assert [a.status for a in _attempts(FALLBACK)] == ["success"]

    @pytest.mark.asyncio
    async def test_a_success_stops_retrying_early(self, monkeypatch, vendors):
        tiers = TierBuilds(_settings())
        writer = _writer(monkeypatch, _settings(), tiers)
        vendors.script["anthropic"] = [500]

        with article_workflow_deadline():
            await writer._invoke_writer(MESSAGES)

        assert vendors.count("anthropic") == 2
        assert vendors.count("openai") == 0
        assert tiers.builds == []

    @pytest.mark.asyncio
    async def test_deadline_expiring_in_the_primary_builds_no_fallback(
        self, monkeypatch, vendors
    ):
        tiers = TierBuilds(_settings())
        writer = _writer(monkeypatch, _settings(), tiers)
        vendors.script["anthropic"] = [STALL]
        # Warm the SDK client so the stalled request is reached in time.
        vendors.script["anthropic"].insert(0, 200)
        with article_workflow_deadline():
            await writer._invoke_writer(MESSAGES)
        monkeypatch.setattr(
            config_module.config, "article_workflow_timeout_seconds", 0.5
        )

        started = time.monotonic()
        with article_workflow_deadline(), pytest.raises(BudgetExhausted) as caught:
            await writer._invoke_writer(MESSAGES)
        elapsed = time.monotonic() - started

        # Returned at the deadline with the stall still unreleased.
        assert vendors.release is not None and not vendors.release.is_set()
        assert elapsed < 2.0
        assert isinstance(caught.value.__cause__, TimeoutError)
        assert tiers.builds == []  # the factory was never called
        assert vendors.count("openai") == 0
        assert vendors.count("anthropic") == 2  # the warm-up and the stall

    @pytest.mark.asyncio
    async def test_a_spent_allowance_sends_nothing(self, monkeypatch, vendors):
        tiers = TierBuilds(_settings())
        writer = _writer(monkeypatch, _settings(), tiers)

        with (
            enclosing_deadline_at(llm_timeouts.now() - 1.0),
            pytest.raises(TimeoutError),
        ):
            await writer._invoke_writer(MESSAGES)

        assert vendors.sends == [] and vendors.sync_sends == []
        assert tiers.builds == []


# ------------------------------------------------------------- event loop


class TestEventLoop:
    @pytest.mark.asyncio
    async def test_a_concurrent_task_progresses_while_the_writer_runs(
        self, monkeypatch, vendors
    ):
        writer = _writer(monkeypatch, _settings())
        # Warm the client so the measured window is the request itself.
        with article_workflow_deadline():
            await writer._invoke_writer(MESSAGES)
        vendors.delay = 0.3
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            with article_workflow_deadline():
                await writer.write(
                    ticker="TEST",
                    company_name="Test Co",
                    report_text="Report",
                    trade_date="2026-01-01",
                )
        finally:
            task.cancel()

        assert vendors.sync_sends == []
        # A sync request would have parked the loop for the whole 0.3 s.
        assert ticks >= 10


# --------------------------------------------------------------- attempts


class TestTierAttemptsAreCounted:
    @pytest.mark.asyncio
    async def test_a_flex_reissue_is_one_attempt_with_two_dispatches(
        self, monkeypatch, vendors
    ):
        settings = _settings(openai_service_tier="flex")
        tiers = TierBuilds(settings)
        writer = _writer(monkeypatch, settings, tiers)
        vendors.script["anthropic"] = [400]  # not retryable: straight to the tier
        vendors.script["openai"] = [429, 200]  # flex capacity, then standard

        with article_workflow_deadline():
            article = await writer._invoke_writer(MESSAGES)

        assert article.startswith("# Openai Article")
        assert vendors.count("anthropic") == 1
        assert [
            s["service_tier"] for s in vendors.sends if s["vendor"] == "openai"
        ] == [
            "flex",
            "auto",
        ]
        [attempt] = _attempts(FALLBACK)
        assert attempt.status == "success"
        assert [d["phase"] for d in attempt.dispatches] == ["initial", "reissue"]
        assert [d["tier"] for d in attempt.dispatches] == ["flex", "auto"]
        assert tiers.builds == ["review_group"]


# ---------------------------------------------------------- workflow owner


class _FakeWriter:
    """Stands in for ArticleWriter in the owner; its write is scriptable."""

    def __init__(self, write: Callable[..., Any], *, fell_back: bool = False):
        self.write = write
        self.current_model_name = "gpt-test"
        self.writer_fell_back = fell_back
        self.load_voice_samples = MagicMock(return_value="")


def _args() -> SimpleNamespace:
    return SimpleNamespace(quiet=True, brief=True)


async def _handle(tmp_path, writer: _FakeWriter, editor: Any, result: dict) -> None:
    from src.output import handle_article_generation

    with (
        patch("src.article_writer.ArticleWriter", return_value=writer),
        patch("src.article_writer.ArticleEditor", return_value=editor),
    ):
        # The test's own bound: a removed workflow deadline fails, never hangs.
        # asyncio.wait, not wait_for, which would wait on a task that ignores
        # its cancellation.
        task = asyncio.ensure_future(
            handle_article_generation(
                args=_args(),
                ticker="TEST",
                company_name="Test Co",
                report_text="Report",
                trade_date="2026-01-01",
                analysis_result=result,
                resolve_article_path_fn=lambda *_a, **_k: tmp_path / "article.md",
            )
        )
        done, _ = await asyncio.wait({task}, timeout=3.0)
        assert task in done, "the article workflow outlived its own allowance"
        task.result()


def _unavailable_editor() -> Any:
    editor = MagicMock()
    editor.is_available.return_value = False
    return editor


class TestWorkflowOwner:
    @pytest.mark.asyncio
    async def test_the_writer_runs_under_the_article_allowance(
        self, monkeypatch, tmp_path
    ):
        """Its own deadline, not an analysis one: none is inherited."""
        monkeypatch.setattr(
            config_module.config, "article_workflow_timeout_seconds", 600.0
        )
        seen: list[float | None] = []

        async def write(**_kwargs: Any) -> str:
            deadline = current_enclosing_deadline()
            seen.append(None if deadline is None else deadline - llm_timeouts.now())
            return "# Draft\n\nBody."

        assert current_enclosing_deadline() is None
        await _handle(tmp_path, _FakeWriter(write), _unavailable_editor(), {})

        [remaining] = seen
        assert remaining is not None and 590.0 < remaining <= 600.0
        assert current_enclosing_deadline() is None

    @pytest.mark.asyncio
    async def test_a_late_draft_is_discarded_and_never_written(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            config_module.config, "article_workflow_timeout_seconds", 0.2
        )
        release = asyncio.Event()
        finished = asyncio.Event()

        async def write(**_kwargs: Any) -> str:
            try:
                await release.wait()
            except asyncio.CancelledError:
                # Ignores cancellation and finishes late, as a stalled SDK
                # call in a worker thread would.
                await release.wait()
            finished.set()
            return "# Late Draft\n\nArrived after the deadline."

        result: dict[str, Any] = {}
        try:
            await _handle(tmp_path, _FakeWriter(write), _unavailable_editor(), result)
        finally:
            release.set()  # the late draft now arrives, whatever happened
        await asyncio.wait_for(finished.wait(), timeout=2.0)
        await asyncio.sleep(0.05)
        assert list(tmp_path.iterdir()) == []
        assert "article_generation" not in result

    @pytest.mark.asyncio
    async def test_an_exhausted_review_keeps_the_draft_unreviewed(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            config_module.config, "article_workflow_timeout_seconds", 0.3
        )

        async def write(**_kwargs: Any) -> str:
            return "# Draft\n\nHeld before the allowance ran out."

        async def edit(**_kwargs: Any) -> tuple[str, dict[str, Any]]:
            await asyncio.Event().wait()  # outlives the allowance
            return "# Approved", {"verdict": "APPROVED"}

        editor = MagicMock()
        editor.is_available.return_value = True
        editor.edit = edit
        result: dict[str, Any] = {}
        await _handle(tmp_path, _FakeWriter(write, fell_back=True), editor, result)

        assert not (tmp_path / "article.md").exists()
        draft = tmp_path / "article.draft.md"
        assert draft.read_text(encoding="utf-8").startswith("# Draft")
        generation = result["article_generation"]
        assert generation["status"] == "REVIEW_REQUIRED"
        assert generation["editor_review_error"] is True
        assert generation["workflow_deadline_exceeded"] is True
        assert generation["writer_fell_back"] is True
        assert result["run_summary"]["article_writer_fell_back"] is True

    def test_a_review_error_never_publishes(self):
        from src.output import _article_is_publishable

        approved = {"verdict": "APPROVED", "citation_audit_status": "PASSED"}
        assert _article_is_publishable("# A", approved)
        assert not _article_is_publishable("# A", {**approved, "review_error": True})
        assert not _article_is_publishable(
            "# A", {**approved, "workflow_deadline_exceeded": True}
        )

    @pytest.mark.asyncio
    async def test_a_step_after_the_deadline_is_never_started(self):
        started = False

        async def step() -> str:
            nonlocal started
            started = True
            return "x"

        coro = step()
        with (
            enclosing_deadline_at(llm_timeouts.now() - 0.1),
            pytest.raises(ArticleWorkflowExpired),
        ):
            await run_article_workflow_step(coro, label="test")
        assert started is False


# ------------------------------------------------- direct-caller dispatch


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


class TestDirectCallerUnderAnEnclosingDeadline:
    """``_dispatch_budget`` with no invocation scope reads the enclosing deadline."""

    @pytest.fixture
    def clock(self, monkeypatch):
        fake = FakeClock()
        monkeypatch.setattr(llm_timeouts, "now", fake)
        monkeypatch.setattr(config_module.config, "llm_dispatch_margin_seconds", 2.0)
        monkeypatch.setattr(config_module.config, "llm_min_useful_attempt_seconds", 5.0)
        return fake

    def _client(self):
        return transports.new_anthropic_chat(
            model="claude-opus-4-8",
            anthropic_api_key="k",
            max_retries=0,
            timeout=300.0,
            dispatch_provider="anthropic",
            budget_timeout_class="editorial",
        )

    @pytest.mark.asyncio
    async def test_without_a_deadline_the_constructed_timeout_stands(
        self, clock, vendors
    ):
        await self._client().ainvoke(MESSAGES)
        [send] = vendors.sends
        assert send["timeout"] == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_a_bound_deadline_sizes_the_request(self, clock, vendors):
        with enclosing_deadline_at(clock() + 33.0):
            await self._client().ainvoke(MESSAGES)
        [send] = vendors.sends
        assert send["timeout"] == pytest.approx(31.0)

    @pytest.mark.asyncio
    async def test_below_a_useful_attempt_nothing_is_sent(self, clock, vendors):
        with (
            enclosing_deadline_at(clock() + 6.0),
            pytest.raises(BudgetExhausted),
        ):
            await self._client().ainvoke(MESSAGES)
        assert vendors.sends == []


# ------------------------------------------- the writer's settings arrive


class TestWriterSettingsAreHonoured:
    def test_default_writer_clients_send_one_attempt(self):
        settings = _settings()
        writer = _build(SeatId.ARTICLE_WRITER, settings)
        fallback = _build(SeatId.ARTICLE_WRITER_REVIEW_FALLBACK, settings)
        assert writer.max_retries == 0
        assert fallback.max_retries == 0

    def test_the_legacy_writer_sends_one_attempt_by_default(self):
        from src import llms

        llm = llms.create_writer_llm(
            model="claude-opus-4-8", api_key_override="a", allow_fallback=False
        )
        assert llm.max_retries == 0

    def test_canonical_overrides_reach_the_client(self):
        policy = seat_policy_with(
            SeatId.ARTICLE_WRITER,
            sampling_temperature=0.3,
            client_timeout_seconds=111.0,
        )
        llm = _build(SeatId.ARTICLE_WRITER, _settings(), execution_policy=policy)
        assert llm.max_retries == 0
        assert llm.default_request_timeout == pytest.approx(111.0)

    def test_legacy_overrides_reach_the_client_temperature_included(self):
        settings = Settings(
            _env_file=None,
            google_api_key="g",
            claude_api_key="a",
            writer_model="claude-haiku-4-5",
        )
        policy = seat_policy_with(
            SeatId.ARTICLE_WRITER,
            sampling_temperature=0.3,
            client_timeout_seconds=111.0,
        )
        llm = _build(SeatId.ARTICLE_WRITER, settings, execution_policy=policy)
        assert llm.temperature == pytest.approx(0.3)
        assert llm.max_retries == 0
        assert llm.default_request_timeout == pytest.approx(111.0)

    def test_the_seam_passes_every_setting_it_is_given(self, monkeypatch):
        from src import article_writer

        monkeypatch.setattr(article_writer, "config", _settings())
        llm = article_writer.create_writer_llm(temperature=0.3, timeout=111.0)
        assert llm.max_retries == 0
        assert llm.default_request_timeout == pytest.approx(111.0)

    def test_no_override_keeps_the_registered_policy(self):
        assert seat_policy_with(SeatId.ARTICLE_WRITER) is None


class TestWorkflowSetting:
    def test_the_shipped_default(self):
        field = Settings.model_fields["article_workflow_timeout_seconds"]
        assert field.default == 900.0

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_allowance_is_rejected(self, bad):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, article_workflow_timeout_seconds=bad)
