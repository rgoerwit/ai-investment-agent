from unittest.mock import MagicMock, patch


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
    import src.llms
    from src.llms import get_consultant_llm

    src.llms._consultant_llm_instance = None
    try:
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
        assert kwargs["error_type"] == "RuntimeError"
        assert kwargs["error"] == "boom"
        assert kwargs["exc_info"] is True
    finally:
        src.llms._consultant_llm_instance = None
