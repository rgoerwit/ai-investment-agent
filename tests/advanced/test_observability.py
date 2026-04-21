"""Tests for the Langfuse observability runtime."""

from __future__ import annotations

import importlib
import sys
import threading
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


class _FakeContextManager:
    def __init__(
        self,
        label: str,
        *,
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ):
        self.label = label
        self.entered = False
        self.exited = False
        self.enter_error = enter_error
        self.exit_error = exit_error

    def __enter__(self):
        self.entered = True
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        if self.exit_error is not None:
            raise self.exit_error
        return False


class _MinimalTraceClient:
    def __init__(self, root_ctx: object):
        self._root_ctx = root_ctx

    def start_as_current_observation(self, **_: object):
        return self._root_ctx

    def get_current_trace_id(self):
        return "trace-123"

    def get_trace_url(self, *, trace_id: str):
        return f"https://langfuse/{trace_id}"


def _fake_langfuse_modules(
    *,
    client: object | None = None,
    callback_handler: object | None = None,
    propagate_attributes: object | None = None,
) -> dict[str, ModuleType]:
    langfuse_module = ModuleType("langfuse")
    langfuse_langchain_module = ModuleType("langfuse.langchain")

    if client is not None:
        langfuse_module.get_client = MagicMock(return_value=client)
    if propagate_attributes is not None:
        langfuse_module.propagate_attributes = propagate_attributes
    if callback_handler is not None:
        langfuse_langchain_module.CallbackHandler = callback_handler

    return {
        "langfuse": langfuse_module,
        "langfuse.langchain": langfuse_langchain_module,
    }


def _reload_observability():
    import src.observability as observability

    return importlib.reload(observability)


@pytest.fixture(autouse=True)
def restore_observability_module():
    """Keep module-level state aligned with the real config singleton between tests."""
    yield
    _reload_observability()


class TestObservabilityRuntime:
    def test_get_observability_runtime_returns_noop_when_disabled(self):
        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = False

            observability = _reload_observability()
            runtime = observability.get_observability_runtime()
            trace = runtime.start_analysis_trace(
                ticker="TEST.X",
                session_id="session-1",
                tags=["analysis"],
                metadata={},
                input_payload={"ticker": "TEST.X"},
            )

        assert trace.enabled is False
        assert trace.callbacks == []

    def test_get_observability_runtime_returns_noop_when_keys_missing(self):
        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = ""
            mock_config.get_langfuse_secret_key.return_value = ""

            observability = _reload_observability()
            runtime = observability.get_observability_runtime()

        assert type(runtime).__name__ == "NoopObservabilityRuntime"

    def test_start_analysis_trace_creates_root_context_and_callback(self):
        mock_client = MagicMock()
        mock_root_ctx = _FakeContextManager("root")
        mock_propagation_ctx = _FakeContextManager("propagation")
        mock_callback_cls = MagicMock(return_value=MagicMock(name="callback"))
        mock_client.start_as_current_observation.return_value = mock_root_ctx
        mock_client.get_current_trace_id.return_value = "trace-123"
        mock_client.get_trace_url.return_value = "https://langfuse/trace-123"

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=mock_callback_cls,
                    propagate_attributes=MagicMock(return_value=mock_propagation_ctx),
                ),
            ):
                observability = _reload_observability()
                runtime = observability.get_observability_runtime()
                trace = runtime.start_analysis_trace(
                    ticker="0005.HK",
                    session_id="batch-1",
                    tags=["analysis", "quick"],
                    metadata={"ticker": "0005.HK", "run_mode": "quick"},
                    input_payload={"ticker": "0005.HK"},
                )

                current = observability.get_current_trace_context()
                merged = observability.build_langchain_config(
                    metadata={"workflow": "article"}
                )
                trace.close()

        assert trace.enabled is True
        assert trace.trace_id == "trace-123"
        assert trace.trace_url == "https://langfuse/trace-123"
        assert len(trace.callbacks) == 1
        assert current is trace
        assert merged["callbacks"] == trace.callbacks
        assert merged["metadata"]["ticker"] == "0005.HK"
        assert merged["metadata"]["workflow"] == "article"
        assert mock_root_ctx.entered is True
        assert mock_root_ctx.exited is True
        assert mock_propagation_ctx.entered is True
        assert mock_propagation_ctx.exited is True
        mock_client.start_as_current_observation.assert_called_once()
        mock_callback_cls.assert_called_once_with()

    def test_start_analysis_trace_sanitizes_propagated_metadata(self):
        mock_client = MagicMock()
        mock_root_ctx = _FakeContextManager("root")
        mock_propagation_ctx = _FakeContextManager("propagation")
        mock_propagate = MagicMock(return_value=mock_propagation_ctx)
        mock_client.start_as_current_observation.return_value = mock_root_ctx
        mock_client.get_current_trace_id.return_value = "trace-123"
        mock_client.get_trace_url.return_value = "https://langfuse/trace-123"

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(return_value=MagicMock()),
                    propagate_attributes=mock_propagate,
                ),
            ):
                observability = _reload_observability()
                trace = observability.get_observability_runtime().start_analysis_trace(
                    ticker="0005.HK",
                    session_id="batch-1",
                    tags=["analysis"],
                    metadata={
                        "ticker": "0005.HK",
                        "quick_mode": True,
                        "release": "3.1.0",
                        "ignored": {"nested": "object"},
                    },
                    input_payload={"ticker": "0005.HK"},
                )
                trace.close()

        propagated_metadata = mock_propagate.call_args.kwargs["metadata"]
        assert propagated_metadata == {
            "ticker": "0005.HK",
            "quick_mode": "true",
            "release": "3.1.0",
        }

    def test_start_analysis_trace_fails_soft_on_capability_gap(self):
        mock_client = object()

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(),
                    propagate_attributes=MagicMock(return_value=nullcontext()),
                ),
            ):
                observability = _reload_observability()
                runtime = observability.get_observability_runtime()
                trace = runtime.start_analysis_trace(
                    ticker="TEST.X",
                    session_id="session-1",
                    tags=["analysis"],
                    metadata={"ticker": "TEST.X"},
                    input_payload={"ticker": "TEST.X"},
                )

        assert trace.enabled is False

    def test_start_analysis_trace_allows_missing_optional_feature_methods(self):
        mock_root_ctx = _FakeContextManager("root")
        mock_propagation_ctx = _FakeContextManager("propagation")
        mock_client = _MinimalTraceClient(mock_root_ctx)

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(return_value=MagicMock()),
                    propagate_attributes=MagicMock(return_value=mock_propagation_ctx),
                ),
            ):
                observability = _reload_observability()
                trace = observability.get_observability_runtime().start_analysis_trace(
                    ticker="TEST.X",
                    session_id="session-1",
                    tags=["analysis"],
                    metadata={"ticker": "TEST.X"},
                    input_payload={"ticker": "TEST.X"},
                )
                trace.close()

        assert trace.enabled is True
        assert mock_root_ctx.exited is True
        assert mock_propagation_ctx.exited is True

    def test_start_analysis_trace_cleans_up_root_when_propagation_enter_fails(self):
        mock_client = MagicMock()
        mock_root_ctx = _FakeContextManager("root")
        mock_propagation_ctx = _FakeContextManager(
            "propagation", enter_error=RuntimeError("propagation boom")
        )
        mock_client.start_as_current_observation.return_value = mock_root_ctx

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(return_value=MagicMock()),
                    propagate_attributes=MagicMock(return_value=mock_propagation_ctx),
                ),
            ):
                observability = _reload_observability()
                with patch.object(observability, "logger") as mock_logger:
                    trace = (
                        observability.get_observability_runtime().start_analysis_trace(
                            ticker="TEST.X",
                            session_id="session-1",
                            tags=["analysis"],
                            metadata={"ticker": "TEST.X"},
                            input_payload={"ticker": "TEST.X"},
                        )
                    )

        assert trace.enabled is False
        assert mock_root_ctx.entered is True
        assert mock_root_ctx.exited is True
        assert mock_propagation_ctx.entered is True
        assert mock_propagation_ctx.exited is False
        mock_logger.warning.assert_any_call(
            "langfuse_trace_start_failed", error="propagation boom"
        )

    def test_score_trace_prefers_score_current_trace(self):
        mock_client = MagicMock()
        mock_root_ctx = _FakeContextManager("root")
        mock_propagation_ctx = _FakeContextManager("propagation")
        mock_client.start_as_current_observation.return_value = mock_root_ctx
        mock_client.get_current_trace_id.return_value = "trace-123"
        mock_client.get_trace_url.return_value = "https://langfuse/trace-123"

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(return_value=MagicMock()),
                    propagate_attributes=MagicMock(return_value=mock_propagation_ctx),
                ),
            ):
                observability = _reload_observability()
                trace = observability.get_observability_runtime().start_analysis_trace(
                    ticker="TEST.X",
                    session_id="session-1",
                    tags=["analysis"],
                    metadata={"ticker": "TEST.X"},
                    input_payload={"ticker": "TEST.X"},
                )
                trace.score_trace(
                    name="pm_verdict",
                    value="BUY",
                    data_type="CATEGORICAL",
                )
                trace.close()

        mock_client.score_current_trace.assert_called_once()
        mock_client.create_score.assert_not_called()

    def test_score_trace_falls_back_to_create_score(self):
        mock_client = MagicMock()
        mock_root_ctx = _FakeContextManager("root")
        mock_propagation_ctx = _FakeContextManager("propagation")
        mock_client.start_as_current_observation.return_value = mock_root_ctx
        mock_client.get_current_trace_id.return_value = "trace-123"
        mock_client.get_trace_url.return_value = "https://langfuse/trace-123"
        del mock_client.score_current_trace

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(return_value=MagicMock()),
                    propagate_attributes=MagicMock(return_value=mock_propagation_ctx),
                ),
            ):
                observability = _reload_observability()
                trace = observability.get_observability_runtime().start_analysis_trace(
                    ticker="TEST.X",
                    session_id="session-1",
                    tags=["analysis"],
                    metadata={"ticker": "TEST.X"},
                    input_payload={"ticker": "TEST.X"},
                )
                trace.score_trace(
                    name="analysis_validity",
                    value=1.0,
                    data_type="BOOLEAN",
                )
                trace.close()

        mock_client.create_score.assert_called_once()

    def test_close_attempts_both_context_exits_independently(self):
        mock_client = MagicMock()
        mock_root_ctx = _FakeContextManager(
            "root", exit_error=RuntimeError("root exit")
        )
        mock_propagation_ctx = _FakeContextManager(
            "propagation", exit_error=RuntimeError("prop exit")
        )
        mock_client.start_as_current_observation.return_value = mock_root_ctx
        mock_client.get_current_trace_id.return_value = "trace-123"
        mock_client.get_trace_url.return_value = "https://langfuse/trace-123"

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            mock_config.app_release = "3.1.0"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(
                    client=mock_client,
                    callback_handler=MagicMock(return_value=MagicMock()),
                    propagate_attributes=MagicMock(return_value=mock_propagation_ctx),
                ),
            ):
                observability = _reload_observability()
                with patch.object(observability, "logger") as mock_logger:
                    trace = (
                        observability.get_observability_runtime().start_analysis_trace(
                            ticker="TEST.X",
                            session_id="session-1",
                            tags=["analysis"],
                            metadata={"ticker": "TEST.X"},
                            input_payload={"ticker": "TEST.X"},
                        )
                    )
                    trace.close()

        assert mock_root_ctx.exited is True
        assert mock_propagation_ctx.exited is True
        mock_logger.warning.assert_any_call(
            "langfuse_propagation_ctx_exit_failed", error="prop exit"
        )
        mock_logger.warning.assert_any_call(
            "langfuse_root_ctx_exit_failed", error="root exit"
        )


class TestToolObservation:
    def test_start_tool_observation_is_noop_without_active_trace(self):
        observability = _reload_observability()
        ctx = observability.start_tool_observation(
            tool_name="get_news",
            input_payload={"ticker": "0005.HK"},
        )
        assert isinstance(ctx, type(nullcontext()))


class TestFlushTraces:
    def test_flush_traces_is_noop_when_disabled(self):
        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = False
            observability = _reload_observability()
            observability.flush_traces()

    def test_flush_traces_is_noop_when_keys_missing(self):
        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = ""
            mock_config.get_langfuse_secret_key.return_value = ""
            observability = _reload_observability()
            observability.flush_traces()

    def test_flush_traces_prefers_shutdown(self):
        mock_client = MagicMock()
        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(client=mock_client),
            ):
                observability = _reload_observability()
                observability.flush_traces()

        mock_client.shutdown.assert_called_once()

    def test_flush_traces_uses_live_config_after_reload_under_patch(self):
        mock_client = MagicMock()

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            _reload_observability()

        with patch.dict(sys.modules, _fake_langfuse_modules(client=mock_client)):
            observability = _reload_observability()
            observability.flush_traces()

        mock_client.shutdown.assert_not_called()
        mock_client.flush.assert_not_called()

    def test_flush_traces_times_out_without_hanging(self):
        blocked = threading.Event()

        def _slow_shutdown():
            blocked.wait(timeout=1.0)

        mock_client = MagicMock()
        mock_client.shutdown.side_effect = _slow_shutdown

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(client=mock_client),
            ):
                observability = _reload_observability()
                with patch.object(observability, "logger") as mock_logger:
                    observability.flush_traces(timeout_seconds=0.01)

        mock_logger.warning.assert_any_call(
            "langfuse_flush_timeout",
            operation="shutdown",
            timeout_seconds=0.01,
        )

    def test_flush_traces_logs_worker_exception(self):
        mock_client = MagicMock()
        mock_client.shutdown.side_effect = RuntimeError("flush boom")

        with patch("src.config.config") as mock_config:
            mock_config.langfuse_enabled = True
            mock_config.get_langfuse_public_key.return_value = "pk-test"
            mock_config.get_langfuse_secret_key.return_value = "sk-test"
            with patch.dict(
                sys.modules,
                _fake_langfuse_modules(client=mock_client),
            ):
                observability = _reload_observability()
                with patch.object(observability, "logger") as mock_logger:
                    observability.flush_traces(timeout_seconds=0)

        mock_logger.warning.assert_any_call(
            "langfuse_flush_failed",
            error="flush boom",
        )


class TestConfigIntegration:
    def test_langfuse_defaults_are_safe(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        from src.config import Settings

        fresh_config = Settings(_env_file=None)

        assert fresh_config.langfuse_enabled is False
        assert fresh_config.langfuse_sample_rate == 1.0
        assert fresh_config.langfuse_debug is False
        assert fresh_config.langfuse_prompt_fetch_enabled is False
        assert fresh_config.langfuse_prompt_label == "production"
