"""Langfuse-first observability runtime with graceful no-op behavior."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from src.config import config

if TYPE_CHECKING:
    from src.config import Settings

logger = structlog.get_logger(__name__)

_REQUIRED_TRACE_METHODS = (
    "start_as_current_observation",
    "get_current_trace_id",
    "get_trace_url",
)

_ACTIVE_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "active_trace_context", default=None
)


def _validate_client_methods(client: object, required: tuple[str, ...]) -> None:
    missing = [name for name in required if not hasattr(client, name)]
    if missing:
        raise RuntimeError(
            "Langfuse client missing required capabilities: " + ", ".join(missing)
        )


def _supports(client: object, *methods: str) -> bool:
    return all(hasattr(client, name) for name in methods)


def _langfuse_keys_present(settings: Settings) -> bool:
    return bool(
        settings.get_langfuse_public_key() and settings.get_langfuse_secret_key()
    )


def _sanitize_trace_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Return Langfuse-compatible propagated metadata."""
    sanitized: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            sanitized[key] = "true" if value else "false"
        elif isinstance(value, str | int | float):
            sanitized[key] = str(value)
    return sanitized


def _merge_callbacks(
    *callback_groups: list[BaseCallbackHandler] | None,
) -> list[BaseCallbackHandler]:
    merged: list[BaseCallbackHandler] = []
    seen: set[int] = set()
    for group in callback_groups:
        for callback in group or []:
            marker = id(callback)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(callback)
    return merged


def get_current_trace_context() -> TraceContext | None:
    """Return the currently active trace context, if any."""
    return _ACTIVE_TRACE_CONTEXT.get()


def build_langchain_config(
    *,
    callbacks: list[BaseCallbackHandler] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build LangChain invoke config from the active trace plus any local additions.

    Returns an empty dict when tracing is unavailable.
    """
    active = get_current_trace_context()
    merged_callbacks = _merge_callbacks(active.callbacks if active else None, callbacks)
    merged_metadata: dict[str, Any] = {}
    if active and active.graph_metadata:
        merged_metadata.update(active.graph_metadata)
    if metadata:
        merged_metadata.update(metadata)

    invoke_config: dict[str, Any] = {}
    if merged_callbacks:
        invoke_config["callbacks"] = merged_callbacks
    if merged_metadata:
        invoke_config["metadata"] = merged_metadata
    if tags:
        invoke_config["tags"] = tags
    return invoke_config


@dataclass(slots=True)
class TraceContext:
    """Base trace context returned by the observability runtime."""

    enabled: bool
    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    graph_metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    trace_url: str | None = None

    def score_trace(
        self,
        *,
        name: str,
        value: float | str,
        data_type: str,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def create_deferred_score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        data_type: str,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class NoopTraceContext(TraceContext):
    """No-op trace context used when Langfuse is disabled."""

    def __init__(self) -> None:
        super().__init__(enabled=False)

    def score_trace(self, **_: Any) -> None:
        return None

    def create_deferred_score(self, **_: Any) -> None:
        return None

    def close(self) -> None:
        return None


class LangfuseTraceContext(TraceContext):
    """Active Langfuse trace context with propagated attributes and callbacks."""

    def __init__(
        self,
        *,
        client: Any,
        root_ctx: Any,
        propagation_ctx: Any,
        callbacks: list[BaseCallbackHandler],
        graph_metadata: dict[str, Any],
        trace_id: str | None,
        trace_url: str | None,
    ) -> None:
        super().__init__(
            enabled=True,
            callbacks=callbacks,
            graph_metadata=graph_metadata,
            trace_id=trace_id,
            trace_url=trace_url,
        )
        self._client = client
        self._root_ctx = root_ctx
        self._propagation_ctx = propagation_ctx
        self._context_token: Token[TraceContext | None] = _ACTIVE_TRACE_CONTEXT.set(
            self
        )
        self._closed = False

    def score_trace(
        self,
        *,
        name: str,
        value: float | str,
        data_type: str,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            if _supports(self._client, "score_current_trace"):
                self._client.score_current_trace(
                    name=name,
                    value=value,
                    data_type=data_type,
                    comment=comment,
                    metadata=metadata,
                )
                return

            trace_id = self.trace_id or self._client.get_current_trace_id()
            if not trace_id:
                return
            if _supports(self._client, "create_score"):
                self._client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type=data_type,
                    comment=comment,
                    metadata=metadata,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("langfuse_score_failed", error=str(exc), score_name=name)

    def create_deferred_score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        data_type: str,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "langfuse_deferred_score_failed",
                error=str(exc),
                score_name=name,
                trace_id=trace_id,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.trace_id = self.trace_id or self._client.get_current_trace_id()
            if self.trace_id:
                self.trace_url = self.trace_url or self._client.get_trace_url(
                    trace_id=self.trace_id
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("langfuse_trace_finalize_failed", error=str(exc))
        finally:
            _ACTIVE_TRACE_CONTEXT.reset(self._context_token)
            try:
                self._propagation_ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("langfuse_propagation_ctx_exit_failed", error=str(exc))
            try:
                self._root_ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("langfuse_root_ctx_exit_failed", error=str(exc))


class ObservabilityRuntime(Protocol):
    """Runtime interface used by CLI and background workflows."""

    def start_analysis_trace(
        self,
        *,
        ticker: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext: ...

    def start_article_trace(
        self,
        *,
        ticker: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext: ...

    def start_retrospective_trace(
        self,
        *,
        ticker: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext: ...


class NoopObservabilityRuntime:
    """Runtime that bypasses all observability work."""

    def start_analysis_trace(self, **_: Any) -> TraceContext:
        return NoopTraceContext()

    def start_article_trace(self, **_: Any) -> TraceContext:
        return NoopTraceContext()

    def start_retrospective_trace(self, **_: Any) -> TraceContext:
        return NoopTraceContext()


class LangfuseObservabilityRuntime:
    """Langfuse-backed observability runtime with lazy client initialization."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        from langfuse import get_client

        client = get_client()
        _validate_client_methods(client, _REQUIRED_TRACE_METHODS)
        self._client = client
        return client

    def _start_trace(
        self,
        *,
        observation_name: str,
        trace_name: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext:
        root_ctx: Any | None = None
        root_entered = False
        propagation_ctx: Any | None = None
        propagation_entered = False
        try:
            from langfuse import propagate_attributes
            from langfuse.langchain import CallbackHandler

            client = self._get_client()
            root_ctx = client.start_as_current_observation(
                name=observation_name,
                as_type="chain",
                input=input_payload,
                metadata={"ticker": metadata.get("ticker")},
            )
            root_ctx.__enter__()
            root_entered = True
            propagation_ctx = propagate_attributes(
                session_id=session_id,
                tags=tags,
                version=self._settings.app_release,
                trace_name=trace_name,
                metadata=_sanitize_trace_metadata(metadata),
            )
            propagation_ctx.__enter__()
            propagation_entered = True

            callbacks = [CallbackHandler()]
            trace_id = client.get_current_trace_id()
            trace_url = client.get_trace_url(trace_id=trace_id) if trace_id else None

            return LangfuseTraceContext(
                client=client,
                root_ctx=root_ctx,
                propagation_ctx=propagation_ctx,
                callbacks=callbacks,
                graph_metadata=metadata,
                trace_id=trace_id,
                trace_url=trace_url,
            )
        except Exception as exc:
            logger.warning("langfuse_trace_start_failed", error=str(exc))
            if propagation_entered and propagation_ctx is not None:
                try:
                    propagation_ctx.__exit__(None, None, None)
                except Exception as cleanup_exc:
                    logger.warning(
                        "langfuse_propagation_cleanup_failed",
                        error=str(cleanup_exc),
                    )
            if root_entered and root_ctx is not None:
                try:
                    root_ctx.__exit__(None, None, None)
                except Exception as cleanup_exc:
                    logger.warning(
                        "langfuse_root_cleanup_failed",
                        error=str(cleanup_exc),
                    )
            return NoopTraceContext()

    def start_analysis_trace(
        self,
        *,
        ticker: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext:
        return self._start_trace(
            observation_name="analysis.run",
            trace_name=f"analysis:{ticker}",
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            input_payload=input_payload,
        )

    def start_article_trace(
        self,
        *,
        ticker: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext:
        return self._start_trace(
            observation_name="article.run",
            trace_name=f"article:{ticker}",
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            input_payload=input_payload,
        )

    def start_retrospective_trace(
        self,
        *,
        ticker: str,
        session_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_payload: dict[str, Any],
    ) -> TraceContext:
        return self._start_trace(
            observation_name="retrospective.run",
            trace_name=f"retrospective:{ticker}",
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            input_payload=input_payload,
        )


def get_observability_runtime(settings: Settings | None = None) -> ObservabilityRuntime:
    """Return the enabled observability runtime or a no-op runtime."""
    settings = settings or config
    if not settings.langfuse_enabled:
        return NoopObservabilityRuntime()
    if not _langfuse_keys_present(settings):
        logger.warning(
            "langfuse_missing_keys",
            hint="Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing",
        )
        return NoopObservabilityRuntime()
    try:
        return LangfuseObservabilityRuntime(settings)
    except ImportError:
        logger.warning(
            "langfuse_not_installed",
            hint="Run 'poetry add langfuse' to enable Langfuse tracing",
        )
        return NoopObservabilityRuntime()
    except Exception as exc:
        logger.warning("langfuse_runtime_unavailable", error=str(exc))
        return NoopObservabilityRuntime()


def start_tool_observation(
    *,
    tool_name: str,
    input_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """
    Start a tool observation under the current Langfuse trace when available.

    Returns a no-op context manager when tracing is disabled or there is no
    active trace context.
    """
    active = get_current_trace_context()
    if not active or not active.enabled or not config.langfuse_enabled:
        return nullcontext()

    try:
        from langfuse import get_client

        client = get_client()
        if not client.get_current_trace_id():
            return nullcontext()
        return client.start_as_current_observation(
            name=f"tool:{tool_name}",
            as_type="tool",
            input=input_payload,
            metadata=metadata or {"tool_name": tool_name},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("langfuse_tool_observation_unavailable", error=str(exc))
        return nullcontext()


def create_deferred_score(
    *,
    trace_id: str,
    name: str,
    value: float | str,
    data_type: str,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Create a score against a previously recorded trace."""
    if not config.langfuse_enabled or not _langfuse_keys_present(config):
        return
    try:
        from langfuse import get_client

        client = get_client()
        if not _supports(client, "create_score"):
            return
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "langfuse_create_deferred_score_failed",
            error=str(exc),
            trace_id=trace_id,
            score_name=name,
        )


def flush_traces() -> None:
    """Best-effort flush/shutdown of the active Langfuse client."""
    if not config.langfuse_enabled:
        return

    try:
        from langfuse import get_client

        client = get_client()
        if hasattr(client, "shutdown"):
            client.shutdown()
        elif hasattr(client, "flush"):
            client.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("langfuse_flush_failed", error=str(exc))
