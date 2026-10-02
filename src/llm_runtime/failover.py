"""Same-vendor model failover for one retry attempt.

The degradation ladder for a seat is ``(primary, flex) -> (primary, standard) ->
(failover, standard)``. Each rung is owned by the layer that already owns the
concern:

- **What** to fail over to: ``LLM_MODEL_FAILOVERS``, a model->model map resolved
  and validated into ``ResolvedBinding.failover_model`` by the binding plan.
- **When**: ``invoke_with_rate_limit_handling`` (``src/agents/runtime.py``). A
  retry that a ``server_error``/``timeout`` would have spent on the same model is
  spent on the failover model instead. No extra calls are made.
- **How**: the tiered chat-model transports in ``src/llms.py`` mix in
  ``ModelFailoverMixin``. When ``failover_attempt()`` is active they serve the
  request from a copy bound to the failover model at the standard tier, the
  same way they already rewrite the tier for a flex fallback.

Billing follows the response's ``model_name``, so the failover model is priced
as itself. Failover is one hop and same-vendor: crossing vendors would change
independence guarantees and adapter contracts, which is a different feature.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar

import structlog
from langchain_core.runnables import RunnableBinding, RunnableSequence

logger = structlog.get_logger(__name__)

# Adapter kinds whose transports implement ModelFailoverMixin. The binding plan
# rejects a failover configured for any other adapter at startup.
FAILOVER_ADAPTER_KINDS = frozenset({"google_native", "openai_native"})

# Failure kinds that mean "this model is struggling right now". Rate limits,
# bad requests, safety blocks and output caps are not model-health signals.
FAILOVER_FAILURE_KINDS = frozenset({"server_error", "timeout"})

_FAILOVER_ATTEMPT: ContextVar[bool] = ContextVar("llm_failover_attempt", default=False)


@contextmanager
def failover_attempt() -> Iterator[None]:
    """Mark the calls made inside this block as a failover attempt."""
    token = _FAILOVER_ATTEMPT.set(True)
    try:
        yield
    finally:
        _FAILOVER_ATTEMPT.reset(token)


class ModelFailoverMixin:
    """Serve a failover attempt from a copy bound to ``failover_model``.

    Concrete transports declare ``failover_model: str | None = None`` as a
    pydantic field (pydantic only collects fields from model classes) and call
    ``_failover_delegate()`` first in ``_generate``/``_agenerate``.
    """

    # Field holding the model id, and the tier name meaning "not flex".
    _failover_model_field: ClassVar[str] = "model"
    _failover_standard_tier: ClassVar[str] = "standard"

    def _failover_delegate(self) -> Any | None:
        target = getattr(self, "failover_model", None)
        if not target or not _FAILOVER_ATTEMPT.get():
            return None
        source = getattr(self, self._failover_model_field, None)
        logger.warning(
            "llm_model_failover",
            source_model=source,
            target_model=target,
            service_tier=self._failover_standard_tier,
        )
        # The copy has no failover of its own, so it cannot recurse.
        return self.model_copy(  # type: ignore[attr-defined]
            update={
                self._failover_model_field: target,
                "service_tier": self._failover_standard_tier,
                "failover_model": None,
            }
        )


def attach_failover(model: Any, failover_model: str | None) -> Any:
    """Give a constructed transport its failover target, or fail loudly."""
    if not failover_model:
        return model
    if not isinstance(model, ModelFailoverMixin):
        raise TypeError(
            f"{type(model).__name__} cannot serve a model failover; the binding "
            "plan should have rejected this adapter"
        )
    model.failover_model = failover_model  # type: ignore[attr-defined]
    return model


def failover_target(runnable: Any) -> str | None:
    """The failover model of the chat model inside ``runnable``, if any.

    Callers pass composed runnables (``prompt | llm``, ``llm.bind_tools(...)``),
    so walk the composition rather than reading attributes off the top object.
    Only real LangChain composition types are descended: probing arbitrary
    attributes never terminates on a mock, which mints a new child per access.
    """
    seen: set[int] = set()
    stack = [runnable]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ModelFailoverMixin):
            target = getattr(node, "failover_model", None)
            return target if isinstance(target, str) and target else None
        if isinstance(node, RunnableBinding):
            stack.append(node.bound)
        elif isinstance(node, RunnableSequence):
            stack.extend(node.steps)
    return None
