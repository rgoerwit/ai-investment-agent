"""One publication owner for articles: after the bounded step, never inside it.

``generate_article`` and ``ArticleWriter.write`` used to write ``output_path``
from inside the bounded work, so a model call that swallowed the workflow
deadline's cancellation could publish after its caller had been told the
workflow timed out. The fake writer LLM here does exactly that, and a
releasable stall lets the test finish it late and prove nothing is written.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.article_writer import (
    ArticleWorkflowExpired,
    ArticleWriter,
    article_workflow_deadline,
    generate_article,
    publish_article,
    run_article_workflow_step,
)
from src.atomic_files import atomic_write_text
from src.config import config

BOUND_SECONDS = 0.05
SAFETY_NET_SECONDS = 2.0
ARTICLE = "# Test Article\n\nBody."
KWARGS: dict[str, Any] = {
    "ticker": "TEST",
    "company_name": "Test Co",
    "report_text": "Report",
    "trade_date": "2026-01-01",
}


class _Both:
    def __init__(self, *patchers: Any) -> None:
        self._patchers = patchers

    def stop(self) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()


class _StubbornLLM:
    """A writer model whose call swallows cancellation until released."""

    def __init__(self, *, stall: bool) -> None:
        self.stall = stall
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellations = 0

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:
        self.entered.set()
        if self.stall:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
                await self.release.wait()
        self.finished.set()
        return MagicMock(content=ARTICLE)


@pytest.fixture
def short_workflow(monkeypatch):
    """A tiny article workflow deadline."""
    monkeypatch.setattr(config, "article_workflow_timeout_seconds", BOUND_SECONDS)


async def _drain(turns: int = 20) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


async def _finish_late(llm: _StubbornLLM) -> None:
    llm.release.set()
    await asyncio.wait_for(llm.finished.wait(), timeout=SAFETY_NET_SECONDS)
    await _drain()


def _writer_entry(entry: str, writer: ArticleWriter, target: Path):
    """The two public entry points, as awaitables publishing to *target*."""
    if entry == "write":

        async def via_write() -> str:
            with article_workflow_deadline():
                return await run_article_workflow_step(
                    writer.write(output_path=target, **KWARGS), label="test:write"
                )

        return via_write()

    async def via_generate() -> str:
        return await generate_article(output_path=target, **KWARGS)

    return via_generate()


@pytest.fixture(params=["model", "draft_task"])
def seam(request) -> str:
    return str(request.param)


@pytest.fixture(params=["write", "generate_article"])
def entry(request, monkeypatch) -> str:
    monkeypatch.setattr(
        "src.article_writer.search_tavily_inspected", AsyncMock(return_value=None)
    )
    if request.param == "generate_article":
        # generate_article builds its own writer: hand it ours.
        monkeypatch.setattr(
            "src.article_writer.ArticleWriter",
            lambda **_kw: ArticleWriter(samples_dir=None, use_github_urls=False),
        )
    return str(request.param)


def _run(entry: str, seam: str, llm: _StubbornLLM, target: Path):
    patcher: Any = patch("src.article_writer.create_writer_llm", return_value=llm)
    patcher.start()
    if seam == "draft_task":
        # The model call is cancellable but the drafting task itself swallows
        # the cancellation, as an uncooperative tool hook would, and resumes.
        async def stubborn_invoke(self: Any, messages: Any) -> str:
            result = await llm.ainvoke(messages)
            return str(result.content)

        invoke = patch.object(ArticleWriter, "_invoke_writer", stubborn_invoke)
        invoke.start()
        patcher = _Both(patcher, invoke)
    return patcher, _writer_entry(entry, ArticleWriter(samples_dir=None), target)


class TestPublication:
    @pytest.mark.asyncio
    async def test_success_writes_the_article(self, entry, seam, tmp_path):
        target = tmp_path / "out" / "article.md"
        patcher, awaitable = _run(entry, seam, _StubbornLLM(stall=False), target)
        try:
            article = await awaitable
        finally:
            patcher.stop()

        assert target.read_text(encoding="utf-8") == article
        assert "Test Article" in article
        assert [p.name for p in target.parent.iterdir()] == ["article.md"]

    @pytest.mark.asyncio
    async def test_timeout_writes_nothing_and_preserves_an_existing_file(
        self, entry, seam, short_workflow, tmp_path
    ):
        target = tmp_path / "article.md"
        target.write_text("previous", encoding="utf-8")
        llm = _StubbornLLM(stall=True)
        patcher, awaitable = _run(entry, seam, llm, target)
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(awaitable, timeout=SAFETY_NET_SECONDS)
            assert target.read_text(encoding="utf-8") == "previous"
            await _finish_late(llm)
        finally:
            llm.release.set()
            patcher.stop()

        # The orphaned invocation completed, swallowing the cancellation, and
        # still published nothing.
        assert llm.cancellations == 1
        assert target.read_text(encoding="utf-8") == "previous"
        assert [p.name for p in tmp_path.iterdir()] == ["article.md"]

    @pytest.mark.asyncio
    async def test_late_completion_writes_nothing_without_an_existing_file(
        self, entry, seam, short_workflow, tmp_path
    ):
        target = tmp_path / "article.md"
        llm = _StubbornLLM(stall=True)
        patcher, awaitable = _run(entry, seam, llm, target)
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(awaitable, timeout=SAFETY_NET_SECONDS)
            await _finish_late(llm)
        finally:
            llm.release.set()
            patcher.stop()

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_success_replaces_an_existing_file(self, entry, seam, tmp_path):
        target = tmp_path / "article.md"
        target.write_text("previous", encoding="utf-8")
        patcher, awaitable = _run(entry, seam, _StubbornLLM(stall=False), target)
        try:
            article = await awaitable
        finally:
            patcher.stop()

        assert target.read_text(encoding="utf-8") == article


class TestPublishStep:
    def test_publishes_when_no_deadline_is_bound(self, tmp_path):
        target = tmp_path / "a.md"
        publish_article("text", target)
        assert target.read_text(encoding="utf-8") == "text"

    def test_refuses_once_the_workflow_deadline_has_passed(self, monkeypatch, tmp_path):
        from src.llm_runtime import timeouts as llm_timeouts

        clock = [1_000.0]
        monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
        target = tmp_path / "a.md"
        with article_workflow_deadline():
            clock[0] += 10_000.0
            with pytest.raises(ArticleWorkflowExpired):
                publish_article("text", target)
        assert not target.exists()


class TestAtomicWrite:
    def test_failed_replace_preserves_the_destination_and_leaves_no_temp(
        self, monkeypatch, tmp_path
    ):
        target = tmp_path / "a.md"
        target.write_text("previous", encoding="utf-8")

        def boom(*_a: Any, **_k: Any) -> None:
            raise OSError("disk")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="disk"):
            atomic_write_text(target, "new")

        assert target.read_text(encoding="utf-8") == "previous"
        assert [p.name for p in tmp_path.iterdir()] == ["a.md"]

    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "x" / "y" / "a.md"
        atomic_write_text(target, "é")
        assert target.read_text(encoding="utf-8") == "é"
