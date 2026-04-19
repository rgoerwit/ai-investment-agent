"""Tests for pre-graph cached regional macro context."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.macro_context import (
    _compute_fingerprint,
    _is_thin,
    _read_cache,
    _write_cache,
    get_macro_context,
)


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
                new=AsyncMock(return_value="summarized brief"),
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
                new=AsyncMock(return_value="global brief"),
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
