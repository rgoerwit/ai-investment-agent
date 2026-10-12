"""The dispatch-budget fields are declared once and inherited by every transport."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm_runtime.adapters import transports
from src.llm_runtime.dispatch import DispatchBudgetFields
from src.llms import _TieredChatGoogleGenerativeAI

_FIELDS = ("budget_timeout_class", "request_cap_override")


def _transports() -> dict[str, type]:
    return {
        "google": _TieredChatGoogleGenerativeAI,
        "openai": transports.deadline_aware_openai_cls(),
        "anthropic": transports.deadline_aware_anthropic_cls(),
    }


@pytest.mark.parametrize("name", ["google", "openai", "anthropic"])
def test_transports_inherit_the_shared_mixin_and_its_field_definitions(name):
    transport = _transports()[name]
    assert issubclass(transport, DispatchBudgetFields)
    for field in _FIELDS:
        # Inherited, not redeclared: the class body carries no annotation of its own.
        assert field not in transport.__dict__.get("__annotations__", {})
        shared = DispatchBudgetFields.model_fields[field]
        own = transport.model_fields[field]
        assert (own.annotation, own.default, own.metadata) == (
            shared.annotation,
            shared.default,
            shared.metadata,
        )


def test_request_cap_override_stays_positive_and_optional():
    assert DispatchBudgetFields().request_cap_override is None
    assert DispatchBudgetFields(request_cap_override=2.5).request_cap_override == 2.5
    for bad in (0, -1.0):
        with pytest.raises(ValidationError):
            DispatchBudgetFields(request_cap_override=bad)


@pytest.mark.parametrize("name", ["google", "openai", "anthropic"])
def test_a_constructed_transport_carries_and_validates_the_override(name):
    transport = _transports()[name]
    kwargs = {"model": "m", "api_key": "k"}
    if name != "google":
        kwargs["dispatch_provider"] = name
    model = transport(**kwargs, request_cap_override=7.0, budget_timeout_class="x")
    assert model.request_cap_override == 7.0
    assert model.budget_timeout_class == "x"
    with pytest.raises(ValidationError):
        transport(**kwargs, request_cap_override=0)
