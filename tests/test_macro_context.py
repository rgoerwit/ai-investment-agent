"""Tests for pre-graph cached regional macro context."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.macro_context import (
    _compute_fingerprint,
    _is_thin,
    _merge_macro_callbacks,
    _read_cache,
    _summarize,
    _write_cache,
    get_macro_context,
)
from src.token_tracker import TokenTrackingCallback


class TestIsThin:
    def test_none_is_thin(self):
        assert _is_thin(None) is True

    def test_empty_is_thin(self):
        assert _is_thin("") is True

    def test_tool_unavailable_is_thin(self):
        assert _is_thin("Tool unavailable") is True

    def test_timeout_message_is_thin(self):
        assert _is_thin("Macroeconomic news search timed out or failed.") is True

    def test_single_result_is_thin(self):
        xml = "<result>only one</result>" + ("x" * 700)
        assert _is_thin(xml) is True

    def test_two_results_with_enough_text_is_not_thin(self):
        xml = "<result>a</result>\n<result>b</result>" + ("x" * 700)
        assert _is_thin(xml) is False


class TestCacheReadWrite:
    def test_round_trip(self, tmp_path):
        with patch("src.macro_context._CACHE_DIR", tmp_path):
            generated_at = _write_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                report="brief text",
                status="generated",
            )
            cached = _read_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                ttl_hours=12,
            )
        assert cached is not None
        assert cached["report"] == "brief text"
        assert cached["generated_at"] == generated_at

    def test_trade_date_mismatch_invalidates(self, tmp_path):
        with patch("src.macro_context._CACHE_DIR", tmp_path):
            _write_cache(
                "JAPAN",
                trade_date="2026-04-17",
                fingerprint="fp123",
                report="brief text",
                status="generated",
            )
            cached = _read_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                ttl_hours=12,
            )
        assert cached is None

    def test_fingerprint_mismatch_invalidates(self, tmp_path):
        with patch("src.macro_context._CACHE_DIR", tmp_path):
            _write_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="old-fp",
                report="brief text",
                status="generated",
            )
            cached = _read_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="new-fp",
                ttl_hours=12,
            )
        assert cached is None

    def test_ttl_expiry_invalidates(self, tmp_path):
        with patch("src.macro_context._CACHE_DIR", tmp_path):
            _write_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                report="brief text",
                status="generated",
            )
            cache_file = tmp_path / "JAPAN.json"
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            payload["generated_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=13)
            ).isoformat()
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            cached = _read_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                ttl_hours=12,
            )
        assert cached is None

    def test_corrupted_cache_returns_none(self, tmp_path):
        with patch("src.macro_context._CACHE_DIR", tmp_path):
            (tmp_path / "JAPAN.json").write_text("{bad json", encoding="utf-8")
            cached = _read_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                ttl_hours=12,
            )
        assert cached is None

    def test_cache_dir_tracks_live_results_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.macro_context._CACHE_DIR", None)
        monkeypatch.setattr("src.config.config.results_dir", tmp_path)

        from src.macro_context import get_macro_context_cache_dir

        assert get_macro_context_cache_dir() == tmp_path / ".macro_context_cache"

    def test_cache_dir_uses_live_config_after_reload_under_patch(
        self, monkeypatch, tmp_path
    ):
        import src.config
        import src.macro_context as macro_context

        original_results_dir = src.config.config.results_dir

        try:
            with patch("src.config.config") as mock_config:
                mock_config.results_dir = tmp_path / "patched"
                importlib.reload(macro_context)

            monkeypatch.setattr("src.macro_context._CACHE_DIR", None)
            monkeypatch.setattr(src.config.config, "results_dir", tmp_path / "live")

            assert macro_context.get_macro_context_cache_dir() == (
                tmp_path / "live" / ".macro_context_cache"
            )
        finally:
            src.config.config.results_dir = original_results_dir
            importlib.reload(macro_context)


class TestGetMacroContext:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_fetch_and_summarize(self, tmp_path):
        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch(
                "src.macro_context._fetch_macro_raw", new_callable=AsyncMock
            ) as fetch,
            patch("src.macro_context._summarize", new_callable=AsyncMock) as summarize,
        ):
            _write_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                report="cached brief",
                status="generated",
            )
            result = await get_macro_context("7203.T", "2026-04-18")

        assert result.status == "cached"
        assert result.report == "cached brief"
        fetch.assert_not_awaited()
        summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_regional_thin_triggers_global_fallback(self, tmp_path):
        async def _fetch(_trade_date: str, region: str) -> str:
            if region == "JAPAN":
                return "too short"
            return "<result>a</result>\n<result>b</result>" + ("x" * 700)

        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch("src.macro_context._fetch_macro_raw", side_effect=_fetch) as fetch,
            patch(
                "src.macro_context._summarize",
                new=AsyncMock(
                    return_value=(
                        "summarized brief",
                        True,
                        {"agent_name": "Macro Context Analyst", "version": "1.0"},
                    )
                ),
            ),
        ):
            result = await get_macro_context("7203.T", "2026-04-18")

        assert result.status == "generated_fallback"
        assert result.report == "summarized brief"
        assert fetch.await_count == 2

    @pytest.mark.asyncio
    async def test_double_thin_returns_failed(self, tmp_path):
        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch(
                "src.macro_context._fetch_macro_raw",
                new=AsyncMock(return_value="too short"),
            ),
        ):
            result = await get_macro_context("7203.T", "2026-04-18")

        assert result.status == "failed"
        assert result.report == ""

    @pytest.mark.asyncio
    async def test_global_region_fetches_instead_of_disabling(self, tmp_path):
        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch(
                "src.macro_context._fetch_macro_raw",
                new=AsyncMock(
                    return_value="<result>a</result>\n<result>b</result>" + ("x" * 700)
                ),
            ) as fetch,
            patch(
                "src.macro_context._summarize",
                new=AsyncMock(
                    return_value=(
                        "global brief",
                        True,
                        {"agent_name": "Macro Context Analyst", "version": "1.0"},
                    )
                ),
            ),
        ):
            result = await get_macro_context("AAPL", "2026-04-18")

        assert result.region == "GLOBAL"
        assert result.status == "generated"
        assert result.report == "global brief"
        fetch.assert_awaited_once_with("2026-04-18", "GLOBAL")

    @pytest.mark.asyncio
    async def test_summary_failure_returns_failed(self, tmp_path):
        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch(
                "src.macro_context._fetch_macro_raw",
                new=AsyncMock(
                    return_value="<result>a</result>\n<result>b</result>" + ("x" * 700)
                ),
            ),
            patch("src.macro_context._summarize", new=AsyncMock(return_value="")),
        ):
            result = await get_macro_context("7203.T", "2026-04-18")

        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_cache_hit_logs_cache_path(self, tmp_path):
        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch("src.macro_context.logger") as mock_logger,
        ):
            _write_cache(
                "JAPAN",
                trade_date="2026-04-18",
                fingerprint="fp123",
                report="cached brief",
                status="generated",
            )
            await get_macro_context("7203.T", "2026-04-18")

        mock_logger.info.assert_any_call(
            "macro_context_cache_hit",
            ticker="7203.T",
            region="JAPAN",
            cache_path=str(tmp_path / "JAPAN.json"),
        )

    @pytest.mark.asyncio
    async def test_generated_logs_fallback_and_cache_path(self, tmp_path):
        with (
            patch("src.macro_context._CACHE_DIR", tmp_path),
            patch("src.macro_context._compute_fingerprint", return_value="fp123"),
            patch(
                "src.macro_context._fetch_macro_raw",
                side_effect=[
                    "too short",
                    "<result>a</result>\n<result>b</result>" + ("x" * 700),
                ],
            ),
            patch(
                "src.macro_context._summarize",
                new=AsyncMock(
                    return_value=(
                        "brief",
                        True,
                        {"agent_name": "Macro Context Analyst", "version": "1.0"},
                    )
                ),
            ),
            patch("src.macro_context.logger") as mock_logger,
        ):
            await get_macro_context("7203.T", "2026-04-18")

        generated_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and call.args[0] == "macro_context_generated"
        ]
        assert len(generated_calls) == 1
        assert generated_calls[0].kwargs["used_global_fallback"] is True
        assert generated_calls[0].kwargs["cache_path"] == str(tmp_path / "JAPAN.json")


class TestSummarizeCallbacks:
    @pytest.mark.asyncio
    async def test_token_tracking_callback_attached(self):
        prompt = MagicMock(
            agent_name="Macro Context Analyst",
            version="1.0",
            category="macro",
            requires_tools=False,
            source="local",
            system_message="system",
        )
        fake_llm = MagicMock()

        with (
            patch("src.prompts.get_prompt", return_value=prompt),
            patch(
                "src.llms.create_quick_thinking_llm",
                return_value=fake_llm,
            ) as create_llm,
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=AsyncMock(return_value=MagicMock(content="brief")),
            ),
            patch(
                "src.observability.get_current_trace_context",
                return_value=None,
            ),
        ):
            report, llm_invoked, prompt_used = await _summarize(
                "raw",
                "JAPAN",
                "2026-04-18",
            )

        callbacks = create_llm.call_args.kwargs["callbacks"]
        assert report == "brief"
        assert llm_invoked is True
        assert prompt_used["agent_name"] == "Macro Context Analyst"
        assert any(isinstance(cb, TokenTrackingCallback) for cb in callbacks)

    @pytest.mark.asyncio
    async def test_external_callbacks_forwarded(self):
        prompt = MagicMock(
            agent_name="Macro Context Analyst",
            version="1.0",
            category="macro",
            requires_tools=False,
            source="local",
            system_message="system",
        )
        fake_llm = MagicMock()
        external_callback = MagicMock()

        with (
            patch("src.prompts.get_prompt", return_value=prompt),
            patch(
                "src.llms.create_quick_thinking_llm",
                return_value=fake_llm,
            ) as create_llm,
            patch(
                "src.agents.runtime.invoke_with_rate_limit_handling",
                new=AsyncMock(return_value=MagicMock(content="brief")),
            ),
            patch(
                "src.observability.get_current_trace_context",
                return_value=None,
            ),
        ):
            await _summarize(
                "raw",
                "JAPAN",
                "2026-04-18",
                callbacks=[external_callback],
            )

        callbacks = create_llm.call_args.kwargs["callbacks"]
        assert external_callback in callbacks

    def test_active_trace_callbacks_merged_without_duplicates(self):
        shared_callback = MagicMock()
        trace_context = SimpleNamespace(callbacks=[shared_callback])

        with patch(
            "src.observability.get_current_trace_context",
            return_value=trace_context,
        ):
            callbacks = _merge_macro_callbacks([shared_callback])

        assert callbacks.count(shared_callback) == 1
        assert any(isinstance(cb, TokenTrackingCallback) for cb in callbacks)


class TestFingerprint:
    def test_fingerprint_changes_when_prompt_changes(self):
        prompt_a = MagicMock(system_message="A")
        prompt_b = MagicMock(system_message="B")

        with patch("src.prompts.get_prompt", return_value=prompt_a):
            fingerprint_a = _compute_fingerprint()
        with patch("src.prompts.get_prompt", return_value=prompt_b):
            fingerprint_b = _compute_fingerprint()

        assert fingerprint_a != fingerprint_b

    def test_fingerprint_is_stable_for_same_prompt(self):
        prompt = MagicMock(system_message="stable")
        with patch("src.prompts.get_prompt", return_value=prompt):
            assert _compute_fingerprint() == _compute_fingerprint()


class TestPromptLoading:
    def test_macro_context_prompt_loads(self):
        from src.prompts import get_prompt

        prompt = get_prompt("macro_context_analyst")
        assert prompt is not None
        assert prompt.agent_name == "Macro Context Analyst"
        assert "RATES & LIQUIDITY" in prompt.system_message
