from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, call, patch

import pytest


@contextmanager
def _patch_llms_config(mock_cfg):
    """Substitute *both* config references src/llms.py actually reads.

    ``_settings_or_default`` resolves the module-level ``src.llms.config`` binding,
    while the rate-limiter and budget helpers go through
    ``src.llms.config_module.config``. Patching only the latter left every factory
    reading the real ``.env`` — so these assertions silently depended on the
    developer's OPENAI_API_KEY being set, and began failing the moment a migrated
    ``.env`` moved that key to MOONSHOT_API_KEY.
    """
    with ExitStack() as stack:
        stack.enter_context(patch("src.llms.config", mock_cfg))
        stack.enter_context(patch("src.llms.config_module.config", mock_cfg))
        yield mock_cfg


def test_quick_llm_init_logging_emits_once_per_config():
    from src.llms import _reset_init_log_cache_for_tests, create_quick_thinking_llm

    _reset_init_log_cache_for_tests()
    try:
        with patch("src.llms.create_gemini_model", return_value=MagicMock()):
            with patch("src.llms.logger.debug") as mock_debug:
                create_quick_thinking_llm(model="gemini-3-flash-preview")
                create_quick_thinking_llm(model="gemini-3-flash-preview")

        mock_debug.assert_called_once()
    finally:
        _reset_init_log_cache_for_tests()


def test_consultant_llm_init_failure_logs_stack_trace():
    from src.llms import get_consultant_llm

    with patch("src.llms.config") as mock_config:
        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "test-key"
        mock_config.consultant_model = "gpt-5.2"

        with patch(
            "src.llms.create_consultant_llm",
            side_effect=RuntimeError("boom"),
        ):
            with patch("src.llms.logger") as mock_logger:
                assert get_consultant_llm() is None

    mock_logger.error.assert_called_once()
    args = mock_logger.error.call_args.args
    kwargs = mock_logger.error.call_args.kwargs
    assert args[0] == "consultant_llm_init_failed"
    assert kwargs["model"] == "gpt-5.2"
    assert kwargs["quick_mode"] is False
    # summarize_exception fields — raw str(exc) must not appear as `error=`
    assert kwargs["error_type"] == "RuntimeError"
    assert kwargs["message_preview"] == "boom"
    assert "error" not in kwargs
    assert kwargs["exc_info"] is True


def test_quick_consultant_llm_init_failure_uses_quick_model_in_log():
    from src.llms import get_consultant_llm

    with patch("src.llms.config") as mock_config:
        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "test-key"
        mock_config.consultant_model = "gpt-5.2"
        mock_config.consultant_quick_model = "gpt-5.2-mini"

        with patch(
            "src.llms.create_consultant_llm",
            side_effect=RuntimeError("boom"),
        ):
            with patch("src.llms.logger") as mock_logger:
                assert get_consultant_llm(quick_mode=True) is None

    mock_logger.error.assert_called_once()
    kwargs = mock_logger.error.call_args.kwargs
    assert kwargs["model"] == "gpt-5.2-mini"
    assert kwargs["quick_mode"] is True


# ---------------------------------------------------------------------------
# OpenAI rate limiter tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def reset_openai_rl():
    """Reset the module-level OpenAI rate limiter state around each test."""
    from src.llms import _reset_openai_rate_limiter_for_tests

    _reset_openai_rate_limiter_for_tests()
    yield
    _reset_openai_rate_limiter_for_tests()


_OPENAI_FACTORIES = [
    ("create_consultant_llm", {"model": "gpt-4o"}),
    ("create_auditor_llm", {}),
    ("create_editor_llm", {}),
]


class TestOpenAIRateLimiter:
    """Rate-limiter wiring for consultant / auditor / editor LLMs."""

    def _make_config(self, rpm: int | None):
        mock_cfg = MagicMock()
        mock_cfg.enable_consultant = True
        mock_cfg.get_openai_api_key.return_value = "sk-test"
        mock_cfg.consultant_model = "gpt-4o"
        mock_cfg.consultant_quick_model = None
        mock_cfg.auditor_model = "gpt-4o"
        mock_cfg.auditor_quick_model = None
        mock_cfg.editor_model = "gpt-4o"
        mock_cfg.openai_rpm_limit = rpm
        # budget fields
        mock_cfg.llm_default_reasoning_reserve_tokens = 0
        mock_cfg.llm_deep_reasoning_reserve_tokens = 0
        return mock_cfg

    # ChatOpenAI is imported inside the function body, so we patch the source module.
    _CHATOPENAI_TARGET = "langchain_openai.ChatOpenAI"

    @pytest.mark.parametrize("factory_name,kwargs", _OPENAI_FACTORIES)
    def test_rate_limiter_attached_when_rpm_set(
        self, reset_openai_rl, factory_name, kwargs
    ):
        import src.llms as llms_mod

        mock_llm = MagicMock()
        mock_rl = MagicMock()

        with _patch_llms_config(self._make_config(rpm=60)):
            with patch("src.llms._create_rate_limiter_from_rpm", return_value=mock_rl):
                with patch(
                    self._CHATOPENAI_TARGET, return_value=mock_llm
                ) as MockChatOpenAI:
                    factory = getattr(llms_mod, factory_name)
                    factory(**kwargs)

        call_kwargs = MockChatOpenAI.call_args.kwargs
        assert "rate_limiter" in call_kwargs, (
            f"{factory_name}: rate_limiter not passed to ChatOpenAI"
        )
        assert call_kwargs["rate_limiter"] is mock_rl

    @pytest.mark.parametrize("factory_name,kwargs", _OPENAI_FACTORIES)
    def test_no_rate_limiter_when_rpm_unset(
        self, reset_openai_rl, factory_name, kwargs
    ):
        import src.llms as llms_mod

        mock_llm = MagicMock()

        with _patch_llms_config(self._make_config(rpm=None)):
            with patch(
                self._CHATOPENAI_TARGET, return_value=mock_llm
            ) as MockChatOpenAI:
                factory = getattr(llms_mod, factory_name)
                factory(**kwargs)

        call_kwargs = MockChatOpenAI.call_args.kwargs
        assert "rate_limiter" not in call_kwargs, (
            f"{factory_name}: rate_limiter should not be set when OPENAI_RPM_LIMIT unset"
        )

    @pytest.mark.parametrize("factory_name,kwargs", _OPENAI_FACTORIES)
    def test_unthrottled_warning_emitted_once(
        self, reset_openai_rl, factory_name, kwargs
    ):
        """debug log fires exactly once per LLM kind when no limiter is configured."""
        import src.llms as llms_mod

        with _patch_llms_config(self._make_config(rpm=None)):
            with patch(self._CHATOPENAI_TARGET, return_value=MagicMock()):
                with patch("src.llms.logger") as mock_logger:
                    factory = getattr(llms_mod, factory_name)
                    factory(**kwargs)
                    factory(**kwargs)  # second call — should NOT log again

        debug_calls = [
            c
            for c in mock_logger.debug.call_args_list
            if c.args and c.args[0] == "openai_llm_unthrottled"
        ]
        assert len(debug_calls) == 1, (
            f"{factory_name}: expected 1 unthrottled debug log, got {len(debug_calls)}"
        )

    def test_shared_limiter_across_factories(self, reset_openai_rl):
        """All three factories share the same InMemoryRateLimiter instance."""
        import src.llms as llms_mod

        mock_rl = MagicMock()
        captured: list = []

        def capture_rl(**kw):
            if "rate_limiter" in kw:
                captured.append(kw["rate_limiter"])
            return MagicMock()

        with _patch_llms_config(self._make_config(rpm=60)):
            with patch("src.llms._create_rate_limiter_from_rpm", return_value=mock_rl):
                with patch(self._CHATOPENAI_TARGET, side_effect=capture_rl):
                    llms_mod.create_consultant_llm(model="gpt-4o")
                    llms_mod.create_auditor_llm()
                    llms_mod.create_editor_llm()

        assert len(captured) == 3
        assert all(rl is mock_rl for rl in captured), (
            "All factories must share the same rate limiter instance"
        )

    def test_gemini_still_uses_global_rate_limiter(self):
        """Gemini factory is unaffected — it still passes GLOBAL_RATE_LIMITER."""
        import src.llms as llms_mod

        captured_kw: dict = {}

        def fake_gemini(**kw):
            captured_kw.update(kw)
            return MagicMock()

        with patch("src.llms._TieredChatGoogleGenerativeAI", side_effect=fake_gemini):
            with patch("src.llms.config_module.config") as mock_cfg:
                mock_cfg.gemini_rpm_limit = 15
                mock_cfg.llm_base_output_tokens = 8192
                mock_cfg.llm_default_reasoning_reserve_tokens = 0
                mock_cfg.llm_deep_reasoning_reserve_tokens = 0
                mock_cfg.get_google_api_key.return_value = "key"
                llms_mod.create_gemini_model("gemini-3-flash-preview", 0.3, 60, 3)

        assert "rate_limiter" in captured_kw
        assert captured_kw["rate_limiter"] is llms_mod.GLOBAL_RATE_LIMITER
