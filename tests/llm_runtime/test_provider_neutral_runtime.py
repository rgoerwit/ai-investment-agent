"""Seats that became bindable must stop assuming a vendor.

The multi-provider migration made the operational plane bindable, but three
places kept Google/Anthropic assumptions that were invisible while everything
happened to be bound that way:

* the retrospective's flex timeout floor and its failure diagnostics,
* the health check's connectivity probe,
* the Anthropic adapter's output cap.

None produced wrong output on a Google/Anthropic binding, which is exactly why
they need tests rather than observation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.llm_runtime.adapters.base import SeatModelRequest

SRC = Path(__file__).resolve().parents[2] / "src"


def _string_kwarg_values(path: Path, func_names: set[str], kwarg: str) -> list[str]:
    """Return literal string values passed as ``kwarg`` to the named calls."""
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in func_names:
            continue
        for keyword in node.keywords:
            if keyword.arg == kwarg and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    found.append(keyword.value.value)
    return found


class TestRetrospectiveFollowsItsBinding:
    def test_no_vendor_is_hardcoded_into_floors_or_diagnostics(self):
        """AST, not text: a comment explaining the fix would name the vendor.

        `SeatId.RETROSPECTIVE` resolves through the binding plan, so a literal
        provider here gives a non-Google operational plane the wrong flex
        ceiling and mislabels every failure it reports.
        """
        literals = _string_kwarg_values(
            SRC / "retrospective.py",
            {"floor_llm_hard_timeout", "floor_llm_total_timeout", "classify_failure"},
            "provider",
        )
        # "unknown" is not a vendor assumption — it is the explicit sentinel for
        # a call site with no resolved model in scope, and is exactly what a
        # provider-neutral module should say there.
        vendors = [value for value in literals if value != "unknown"]
        assert vendors == [], (
            f"hardcoded vendor literals in retrospective.py: {vendors}"
        )

    def test_the_module_names_no_vendor_at_all_in_provider_position(self):
        source = (SRC / "retrospective.py").read_text()
        assert 'provider="google"' not in source
        assert 'provider="openai"' not in source


class TestRetrospectiveTelemetryDescribesTheResolvedSeat:
    """Behavioural, not source-text: the AST scan above proves no literal is
    present; this proves the *resolved* values actually reach telemetry.

    Provider and model must describe the **same** binding. The first cut fixed
    the provider and left `model_name=config.quick_think_llm`, which is the
    legacy settings field and not what `SeatId.RETROSPECTIVE` resolves to — so a
    failure would report an OpenAI provider beside a Gemini model name.
    """

    @pytest.mark.asyncio
    async def test_provider_and_model_come_from_the_constructed_seat(self, monkeypatch):
        import src.retrospective as retro

        captured: dict = {}

        def _fake_classify(exc, **kwargs):
            captured.update(kwargs)
            return Mock(
                provider=kwargs.get("provider"),
                kind="server_error",
                message="boom",
                retryable=False,
                endpoint_host=None,
            )

        seat_llm = Mock()
        seat_llm._llm_runtime_provider = "openai"
        seat_llm.model_name = "gpt-5.4"

        async def _explode(*args, **kwargs):
            raise RuntimeError("provider exploded")

        seat_llm.ainvoke = _explode

        monkeypatch.setattr(retro, "classify_failure", _fake_classify)
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *a, **k: seat_llm,
        )
        monkeypatch.setattr(
            retro, "get_runtime_provider", lambda llm: llm._llm_runtime_provider
        )
        monkeypatch.setattr(retro, "get_model_name", lambda llm: llm.model_name)

        # A comparison rich enough to reach the LLM call; the call then raises,
        # which is the path whose telemetry this test is about.
        await retro.generate_lesson(
            {
                "ticker": "TEST.L",
                "analysis_date": "2026-06-16",
                "verdict": "DO_NOT_INITIATE",
                "days_elapsed": 60,
                "price_return_pct": 5.9,
                "benchmark_return_pct": -20.0,
                "excess_return_pct": 25.9,
            }
        )

        assert captured.get("provider") == "openai"
        assert captured.get("model_name") == "gpt-5.4", (
            "model_name must be the resolved seat's model, not a legacy "
            "settings field — provider and model must describe one binding"
        )


class TestHealthCheckIsProviderNeutralAndCannotHang:
    def test_the_probe_uses_a_deadline_not_a_cancelling_wait(self):
        """`asyncio.wait_for` cancels then *awaits* — a socket read ignores it.

        The repo's async timeout standard forbids `wait_for` around a provider
        SDK call for exactly this reason: the health check would hang precisely
        when the provider is unreachable.
        """
        source = (SRC / "health_check.py").read_text()
        assert "run_with_hard_timeout(" in source
        assert "asyncio.wait_for(" not in source

    def test_the_connectivity_log_reports_the_resolved_seat(self):
        source = (SRC / "health_check.py").read_text()
        assert "get_runtime_provider(llm)" in source
        # The old log reported `runtime_config.quick_think_llm`, which is the
        # legacy field and is not what SeatId.HEALTH_CHECK resolves to.
        assert "quick_think_llm" not in source


class TestAnthropicHonoursTheOutputBudgetContract:
    @staticmethod
    def _request(output_tokens: int | None) -> SeatModelRequest:
        binding = Mock()
        binding.model = "claude-opus-4-5"
        binding.profile.reasoning_api_mode = None
        binding.identity.vendor_id = "anthropic"
        binding.endpoint_host = "api.anthropic.com"
        settings = Mock()
        settings.get_claude_api_key.return_value = "test-key"
        settings.api_timeout = 120
        return SeatModelRequest(
            binding=binding,
            seat=Mock(),
            quick_mode=False,
            callbacks=(),
            output_tokens=output_tokens,
            settings=settings,
        )

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            pytest.param(4096, 4096, id="explicit-cap-is-honoured"),
            pytest.param(None, 16384, id="omitted-cap-keeps-the-long-form-default"),
        ],
    )
    def test_the_requested_cap_reaches_the_client(
        self, monkeypatch, requested, expected
    ):
        import sys

        from src.llm_runtime.adapters import anthropic as adapter_mod

        captured: dict = {}

        class _FakeChatAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.rate_limiter = None

        fake_module = Mock()
        fake_module.ChatAnthropic = _FakeChatAnthropic
        monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_module)
        monkeypatch.setattr(adapter_mod, "limiter_for_binding", lambda *a, **k: None)

        adapter_mod.AnthropicAdapter().build(self._request(requested))
        assert captured["max_tokens"] == expected
