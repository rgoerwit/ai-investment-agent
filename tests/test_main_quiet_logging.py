"""Regression tests for quiet-mode structured logging."""

from __future__ import annotations

import logging
import sys

import structlog

from src.main import suppress_all_logging


def _restore_default_logging() -> None:
    """Restore the repo's default test-time logging shape after a quiet-mode test."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        stream=sys.stderr,
        level=logging.INFO,
        force=True,
    )
    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"]
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class TestSuppressAllLoggingQ0XRegression:
    """Direct regression tests for the Q0X.SI quiet-mode crash."""

    def teardown_method(self) -> None:
        _restore_default_logging()

    def test_structured_error_with_kwargs_does_not_raise_and_is_emitted(
        self, capsys
    ) -> None:
        suppress_all_logging()
        logger = structlog.get_logger("test_q0x_regression")

        logger.error("analysis_failed", ticker="Q0X.SI", error="boom")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "analysis_failed" in combined
        assert "Q0X.SI" in combined
        assert "boom" in combined

    def test_exc_info_kwarg_does_not_raise(self, capsys) -> None:
        suppress_all_logging()
        logger = structlog.get_logger("test_exc_info")

        try:
            raise ValueError("test error")
        except ValueError:
            logger.error("caught_error", ticker="TEST.SI", exc_info=True)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "caught_error" in combined
        assert "TEST.SI" in combined

    def test_warning_is_emitted(self, capsys) -> None:
        suppress_all_logging()
        logger = structlog.get_logger("test_warning_emit")

        logger.warning("llm_call_retry", attempt=1, wait_seconds="65.3")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "llm_call_retry" in combined
        assert "65.3" in combined

    def test_info_is_filtered(self, capsys) -> None:
        suppress_all_logging()
        logger = structlog.get_logger("test_info_filter")

        logger.info("this_should_be_silent", ticker="QUIET.SI")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "this_should_be_silent" not in combined
        assert "QUIET.SI" not in combined

    def test_repeated_suppress_calls_remain_safe_and_emit_warning(self, capsys) -> None:
        # Mirrors the two quiet-mode call sites in src.main.
        suppress_all_logging()
        suppress_all_logging()
        logger = structlog.get_logger("test_idempotent")

        logger.warning("llm_call_retry", attempt=1, wait_seconds="65.3")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "llm_call_retry" in combined
        assert "65.3" in combined

    def test_preexisting_stdlib_warning_is_still_emitted(self, caplog) -> None:
        logger = logging.getLogger("test_preexisting_stdlib_warning")
        logger.setLevel(logging.NOTSET)

        suppress_all_logging()
        logger.warning("stdlib_warning_visible")

        assert "stdlib_warning_visible" in caplog.text

    def test_preexisting_stdlib_error_is_still_emitted(self, caplog) -> None:
        logger = logging.getLogger("test_preexisting_stdlib_error")
        logger.setLevel(logging.NOTSET)

        suppress_all_logging()
        logger.error("stdlib_error_visible")

        assert "stdlib_error_visible" in caplog.text

    def test_preexisting_stdlib_info_is_filtered(self, caplog) -> None:
        logger = logging.getLogger("test_preexisting_stdlib_info")
        logger.setLevel(logging.NOTSET)

        suppress_all_logging()
        logger.info("stdlib_info_hidden")

        assert "stdlib_info_hidden" not in caplog.text
