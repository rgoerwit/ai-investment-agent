"""L1 golden-template round-trip: each prompt's documented output ≡ its parser.

For every entry in ``PROMPT_CONTRACTS`` this pulls the *documented* output block
out of the live prompt JSON (via the repo's own block finder), substitutes the
``[...]`` placeholder tokens with realistic literals, and feeds the result to the
**real consumer** — asserting the contract's success predicate holds and every
required field is extracted. This catches the silent class of bug where a prompt
teaches one format and the parser expects another (the ``### FINAL
RECOMMENDATION`` miss, the dropped auditor ``STATUS:`` line, a renamed
``DE_RATIO:`` field) — with no LLM call.

See ``scratch/general-prompt-checking.md`` (L1) for the design.
"""

from __future__ import annotations

import re

import pytest

from src.data_block_utils import extract_last_fenced_block
from src.eval.capture_contract import NODE_CAPTURE_SPECS
from src.eval.prompt_contracts import (
    PROMPT_CONTRACTS,
    PromptContract,
    Shape,
    prompt_text,
)
from src.graph.routing import _AUDITOR_CLEAN_STATUSES
from src.prompts import get_prompt

# --- placeholder materialization ------------------------------------------------
# Prompt templates write fields as `FIELD: [placeholder]`. The regex parsers can't
# read a bracketed placeholder, so we replace each `[...]` with a realistic literal
# the parser will accept. The point is to exercise the parser against the prompt's
# own field labels — so a renamed/dropped label surfaces as a parse failure.


def _placeholder_value(content: str) -> str:
    content = content.strip()
    # 1. Prose with an explicit example number, e.g. "[..., e.g., 1.33]".
    example = re.search(r"e\.g\.,?\s*([0-9][0-9.]*)", content)
    if example:
        return example.group(1)
    # 2. Numeric range like "[0-100]".
    if re.fullmatch(r"\d+\s*-\s*\d+", content):
        return "50"
    # 3. Enum: "/"- or "|"-separated ALL-CAPS tokens — take the first.
    parts = re.split(r"\s*[/|]\s*", content)
    caps = [p.strip() for p in parts if re.fullmatch(r"[A-Z][A-Z0-9_ ]*", p.strip())]
    if len(parts) >= 2 and caps:
        return caps[0]
    # 4. Numeric placeholder ("[X.XX]", "[X]", "[Price]").
    if "X" in content or content.lower() in {"price", "number"}:
        decimal = "." in content or "XX" in content or content.lower() == "price"
        return "12.34" if decimal else "12"
    # 5. Free-text fallback.
    return "N/A"


def _materialize(text: str) -> str:
    return re.sub(r"\[([^\[\]]+)\]", lambda m: _placeholder_value(m.group(1)), text)


def _sample_for(contract: PromptContract) -> str:
    msg = prompt_text(contract.prompt_key)
    if contract.shape is Shape.FENCED_BLOCK:
        block = extract_last_fenced_block(
            msg, contract.block_name, include_markers=True
        )
        assert block, (
            f"{contract.name}: fenced block {contract.block_name!r} not found in "
            f"prompt {contract.prompt_key!r} — prompt template drifted"
        )
        return _materialize(block)
    # UNFENCED block / JSON: the parser self-locates within the whole prompt.
    return _materialize(msg)


# --- tests ----------------------------------------------------------------------


@pytest.mark.parametrize("contract", PROMPT_CONTRACTS, ids=lambda c: c.name)
def test_contract_template_roundtrips(contract: PromptContract):
    if contract.shape in (Shape.LABELED_LINE, Shape.HEADER):
        # Line-shaped contracts: the documented line form must be present in the
        # prompt; the parse itself is exercised by the legacy-form test below.
        msg = prompt_text(contract.prompt_key)
        assert re.search(contract.line_pattern, msg, re.MULTILINE), (
            f"{contract.name}: documented line form /{contract.line_pattern}/ "
            f"absent from prompt {contract.prompt_key!r}"
        )
        return

    result = contract.parser(_sample_for(contract))
    assert contract.success(result), (
        f"{contract.name}: parser {contract.parser!r} rejected its own prompt "
        f"template (success predicate failed) — prompt/parser drift"
    )
    for field in contract.required_fields:
        got = (
            result.get(field)
            if isinstance(result, dict)
            else getattr(result, field, None)
        )
        assert got not in (None, ""), (
            f"{contract.name}: required field {field!r} not extracted from the "
            f"prompt template — field renamed/dropped?"
        )


@pytest.mark.parametrize(
    "contract",
    [c for c in PROMPT_CONTRACTS if c.legacy_forms],
    ids=lambda c: c.name,
)
def test_contract_legacy_forms_still_parse(contract: PromptContract):
    for raw, predicate in contract.legacy_forms:
        assert predicate(
            contract.parser(raw)
        ), f"{contract.name}: tolerated legacy form {raw!r} no longer parses"


def test_rm_recommendation_header_classifies():
    """The exact header forms the RM prompt instructs the model to emit."""
    from src.graph.routing import _classify_rm_verdict

    assert _classify_rm_verdict("### FINAL RECOMMENDATION: BUY") == "positive"
    assert _classify_rm_verdict("### INVESTMENT RECOMMENDATION: REJECT") == "negative"


def test_auditor_clean_status_classifies():
    from src.graph.routing import parse_auditor_status

    assert parse_auditor_status("STATUS: CLEAN") in _AUDITOR_CLEAN_STATUSES


def test_every_prompt_key_resolves():
    """Every capture-spec and contract prompt_key resolves through the registry.

    Guards the Auditor key/file split (``global_forensic_auditor`` ≠
    ``auditor.json``) and any future capture-spec typo.
    """
    keys = {spec.prompt_key for spec in NODE_CAPTURE_SPECS.values() if spec.prompt_key}
    keys |= {c.prompt_key for c in PROMPT_CONTRACTS}
    unresolved = sorted(k for k in keys if get_prompt(k) is None)
    assert not unresolved, f"prompt keys did not resolve: {unresolved}"
