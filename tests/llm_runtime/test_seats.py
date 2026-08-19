from pathlib import Path

from src.llm_budgets import AGENT_OUTPUT_BUDGET_FRACTIONS
from src.llm_runtime.capabilities import Capability
from src.llm_runtime.seats import (
    SEATS,
    AuthorityStage,
    BindingGroup,
    ModelIntent,
    ReasoningAdjustment,
    SeatId,
)


def test_registry_covers_every_declared_seat_exactly_once() -> None:
    assert set(SEATS) == set(SeatId)
    assert all(key is spec.seat_id for key, spec in SEATS.items())


def test_prompt_and_budget_references_resolve() -> None:
    prompt_keys = {path.stem for path in Path("prompts").glob("*.json")}
    for spec in SEATS.values():
        if spec.prompt_key is not None:
            assert spec.prompt_key in prompt_keys, spec.seat_id
        if spec.budget_key is not None:
            assert spec.budget_key in AGENT_OUTPUT_BUDGET_FRACTIONS, spec.seat_id


def test_adversarial_grouping_and_authority_are_explicit() -> None:
    assert SEATS[SeatId.BULL].binding_group is BindingGroup.BASE
    assert SEATS[SeatId.BEAR].binding_group is BindingGroup.BASE
    assert SEATS[SeatId.FOREIGN_LANGUAGE].binding_group is BindingGroup.BASE
    assert SEATS[SeatId.CONSULTANT].binding_group is BindingGroup.REVIEW
    assert SEATS[SeatId.AUDITOR].binding_group is BindingGroup.REVIEW
    assert SEATS[SeatId.EDITOR].binding_group is BindingGroup.REVIEW
    assert SEATS[SeatId.EDITOR].authority_stage is AuthorityStage.EDITORIAL
    assert SEATS[SeatId.APAC].binding_group is BindingGroup.REGIONAL_REVIEW
    assert SEATS[SeatId.SEMANTIC_JUDGE].binding_group is BindingGroup.JUDGE


def test_editor_and_quality_path_requirements_are_not_silent() -> None:
    editor = SEATS[SeatId.EDITOR]
    assert Capability.TOOL_CALLING in editor.requires
    assert Capability.STRUCTURED_OUTPUT in editor.requires
    assert editor.budget_key == "Article Editor"
    assert (
        SEATS[SeatId.VALUE_TRAP].normal_reasoning_adjustment
        is ReasoningAdjustment.ONE_STEP
    )
    retry_seats = {
        SeatId.MARKET,
        SeatId.SENTIMENT,
        SeatId.NEWS,
        SeatId.JUNIOR_FUNDAMENTALS,
        SeatId.SENIOR_FUNDAMENTALS,
        SeatId.FOREIGN_LANGUAGE,
        SeatId.VALUE_TRAP,
    }
    assert all(
        SEATS[seat].retry_intent is ModelIntent.REASONING for seat in retry_seats
    )


def test_callback_names_are_unique_except_intentional_fallback_aliases() -> None:
    duplicates: dict[str, list[SeatId]] = {}
    for seat_id, spec in SEATS.items():
        duplicates.setdefault(spec.callback_name, []).append(seat_id)
    repeated = {name: ids for name, ids in duplicates.items() if len(ids) > 1}
    assert repeated == {
        "Article Writer Fallback": [
            SeatId.ARTICLE_WRITER_REVIEW_FALLBACK,
            SeatId.ARTICLE_WRITER_BASE_FALLBACK,
        ]
    }
