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
from langchain_core.messages import ToolMessage

from src.agents.management_guidance import guidance_input_reason
from src.agents.value_trap_evidence import normalize_value_trap_m_and_a_evidence
from src.data_block_utils import extract_block_field, extract_last_fenced_block
from src.eval.capture_contract import NODE_CAPTURE_SPECS
from src.eval.prompt_contracts import (
    PROMPT_CONTRACTS,
    PromptContract,
    Shape,
    prompt_text,
)
from src.graph.routing import _AUDITOR_CLEAN_STATUSES
from src.prompts import get_prompt
from src.validators.supplemental_extractors import extract_value_trap_score

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
        assert predicate(contract.parser(raw)), (
            f"{contract.name}: tolerated legacy form {raw!r} no longer parses"
        )


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


@pytest.mark.parametrize("cited_in_tool", [True, False])
def test_prompted_value_trap_m_and_a_fields_survive_or_downgrade(cited_in_tool: bool):
    """Exercise the prompt's strict M&A fields through the evidence normalizer."""
    contract = next(c for c in PROMPT_CONTRACTS if c.name == "value_trap")
    block = _sample_for(contract)
    url = "https://example.com/acquisition"
    fields = {
        "M&A_CONTEXT_EVIDENCE": "CITED",
        "M&A_CONTEXT_SOURCE_URL": url,
        "M&A_CONTEXT": "Acquired a distributor.",
    }
    for name, value in fields.items():
        pattern = rf"(?m)^{re.escape(name)}:.*$"
        assert len(re.findall(pattern, block)) == 1
        block = re.sub(
            pattern,
            lambda match, name=name, value=value: f"{name}: {value}",
            block,
        )
        assert extract_block_field(block, "VALUE_TRAP_BLOCK", name) == value

    messages = (
        [ToolMessage(content=url, tool_call_id="acquisition-source")]
        if cited_in_tool
        else []
    )
    result = normalize_value_trap_m_and_a_evidence(block, messages, ticker="TEST")
    assert len(re.findall(r"(?m)^[ \t]*M&A_CONTEXT_EVIDENCE:", result)) == 1
    extracted = extract_value_trap_score(result)
    assert extracted["m_and_a_context_evidence"] == (
        "CITED" if cited_in_tool else "UNKNOWN"
    )
    assert (url in result) == cited_in_tool


@pytest.mark.parametrize("mutation", ["indent", "rename"])
def test_value_trap_prompt_m_and_a_triplet_requires_strict_labels(mutation: str):
    contract = next(c for c in PROMPT_CONTRACTS if c.name == "value_trap")
    block = _sample_for(contract)
    for field in ("M&A_CONTEXT_EVIDENCE", "M&A_CONTEXT_SOURCE_URL", "M&A_CONTEXT"):
        assert len(re.findall(rf"(?m)^{field}:", block)) == 1
        assert extract_block_field(block, "VALUE_TRAP_BLOCK", field) is not None

    replacement = (
        "  M&A_CONTEXT_EVIDENCE:"
        if mutation == "indent"
        else "M_AND_A_CONTEXT_EVIDENCE:"
    )
    changed = block.replace("M&A_CONTEXT_EVIDENCE:", replacement, 1)
    assert changed != block
    assert (
        extract_block_field(changed, "VALUE_TRAP_BLOCK", "M&A_CONTEXT_EVIDENCE") is None
    )


def test_prompted_guidance_coverage_is_accepted_by_raw_block_check():
    template = extract_last_fenced_block(
        prompt_text("foreign_language_analyst"),
        "MANAGEMENT_GUIDANCE",
        include_markers=True,
    )
    assert template is not None
    report = _materialize(template)
    assert guidance_input_reason(report) == "UNCHANGED"

    invalid = re.sub(
        r"(?m)^COVERAGE_STATUS:.*$",
        "COVERAGE_STATUS: FOUND_PROBABLY",
        report,
    )
    assert invalid != report
    assert guidance_input_reason(invalid) == "INVALID_COVERAGE"
