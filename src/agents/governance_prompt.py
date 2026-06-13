"""Prompt helpers for Entity Governance Card injection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.validators.entity_governance_card import (
    EntityGovernanceCard,
    card_from_dict,
    card_to_prompt_block_from_dict,
)


def governance_card(state: Mapping[str, Any]) -> EntityGovernanceCard | None:
    """Hydrate the governance card from graph state when present."""

    return card_from_dict(state.get("entity_governance_card"))


def rendered_governance_card(state: Mapping[str, Any]) -> str:
    """Render the governance card without surrounding whitespace."""

    return card_to_prompt_block_from_dict(state.get("entity_governance_card"))


def governance_block(state: Mapping[str, Any], *, with_label: bool = False) -> str:
    """Render a prompt-ready governance card block, or an empty string."""

    rendered = rendered_governance_card(state)
    if not rendered:
        return ""
    if with_label:
        return f"\n\nENTITY GOVERNANCE CARD:\n{rendered}"
    return f"\n\n{rendered}"
