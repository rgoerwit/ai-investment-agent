"""Stand in for the content inspection a Tavily search result goes through.

Tavily inspects through ``InspectionService.check_within``, which bounds
``evaluate`` by what the search left. Replacing only ``evaluate`` keeps that
real bound in the path, so a test of a stalled, failing or late inspection exercises
the production timeout, not a stub of it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

import src.tavily_utils as tavily_utils
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import (
    InspectionDecision,
    InspectionDisposition,
    InspectionEnvelope,
)

Behaviour = Callable[[Any, str], Awaitable[Any]]


def inspect_tavily_with(
    monkeypatch: pytest.MonkeyPatch,
    behaviour: Behaviour,
    *,
    enforced: InspectionDisposition = "allowed",
) -> None:
    """Route Tavily's inspection to ``behaviour(raw, query)`` behind the real
    bound; its value is approved with the *enforced* disposition."""

    class _Service(InspectionService):
        async def evaluate(
            self, envelope: InspectionEnvelope
        ) -> tuple[InspectionDecision, Any]:
            approved = await behaviour(envelope.raw_content, envelope.metadata["query"])
            decision = InspectionDecision(
                action="allow", threat_level="safe", enforced=enforced
            )
            return decision, approved

    service = _Service()
    monkeypatch.setattr(tavily_utils, "get_current_inspection_service", lambda: service)
