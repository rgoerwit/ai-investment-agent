"""Prompt-output contract registry (L1 backbone of the prompt-drift harness).

Each :class:`PromptContract` ties one structured output block to the *real*
consumer that parses it, so a test can prove the prompt's documented template and
the parser still agree without running an LLM. See
``scratch/general-prompt-checking.md`` for the full design.

Two deliberate choices:

- Block text is located with the repo's own
  :func:`src.data_block_utils.extract_last_fenced_block` (the same finder the
  production parsers use), never with hand-copied marker literals.
- Contract prompt text is resolved from the on-disk JSON by ``agent_key`` via
  :func:`prompt_text` — **not** ``src.prompts.get_prompt``, which can apply
  ``PROMPT_<KEY>`` env / Langfuse overrides and is therefore not canonical for a
  local-prompt parity test.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.agents.fundamentals_reconciler import parse_score_breakdown

# Owning parsers — each contract points at the live consumer, not a copy.
from src.agents.output_validation import validate_required_output
from src.charts.extractors.pm_block import PMBlockData, extract_pm_block
from src.charts.extractors.valuation import _extract_params
from src.graph.routing import (
    _AUDITOR_CLEAN_STATUSES,
    _classify_rm_verdict,
    parse_auditor_status,
)
from src.ibkr.order_builder import parse_trade_block
from src.thesis_constants import HEALTH_SCORE_CRITERIA
from src.validators.metric_extractor import extract_metrics
from src.validators.supplemental_extractors import (
    extract_legal_risks,
    extract_material_events_status,
    extract_value_trap_score,
    parse_consultant_conditions,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def prompt_text(prompt_key: str) -> str:
    """Return the canonical on-disk ``system_message`` for an ``agent_key``.

    Scans ``prompts/*.json`` and matches the JSON's own ``agent_key`` field, so
    it never derives a filename from the key (the Auditor key
    ``global_forensic_auditor`` lives in ``auditor.json``). Deliberately does
    not use ``get_prompt`` — env/Langfuse overrides would make it non-canonical.
    """
    for json_file in sorted(_PROMPTS_DIR.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if data.get("agent_key") == prompt_key:
            system_message = data.get("system_message")
            if not isinstance(system_message, str):
                raise ValueError(f"{json_file} has a non-string system_message")
            return system_message
    raise KeyError(f"no prompt with agent_key={prompt_key!r} under {_PROMPTS_DIR}")


class Shape(Enum):
    """How a contract's output is located in agent text."""

    FENCED_BLOCK = "fenced"  # ### --- START X --- ... ### --- END X ---
    UNFENCED_BLOCK = "unfenced"  # TRADE_BLOCK: / FORENSIC_DATA_BLOCK: / JSON
    LABELED_LINE = "labeled"  # STATUS: CLEAN
    HEADER = "header"  # ### FINAL RECOMMENDATION: BUY


@dataclass(frozen=True)
class PromptContract:
    """One structured-output contract between a prompt and its parser."""

    name: str
    prompt_key: str
    shape: Shape
    parser: Callable[[str], Any]
    success: Callable[[Any], bool]  # parser-specific predicate over the result
    block_name: str | None = None  # FENCED/UNFENCED: passed to the finder
    line_pattern: str | None = None  # LABELED_LINE / HEADER
    required_fields: tuple[str, ...] = ()
    # (input, predicate) pairs that must keep parsing — guards tolerated legacy forms.
    legacy_forms: tuple[tuple[str, Callable[[Any], bool]], ...] = ()


def _pm_ok(result: Any) -> bool:
    return isinstance(result, PMBlockData) and result.verdict is not None


def _metrics_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("debt_to_equity") is not None


def _value_trap_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("score") is not None


def _valuation_ok(result: Any) -> bool:
    return result is not None


def _trade_ok(result: Any) -> bool:
    return result is not None


def _legal_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("pfic_status") is not None


def _auditor_clean(result: Any) -> bool:
    return result in _AUDITOR_CLEAN_STATUSES


def _rm_classified(result: Any) -> bool:
    return result in {"positive", "negative"}


def _consultant_ok(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("ok"))


PROMPT_CONTRACTS: tuple[PromptContract, ...] = (
    PromptContract(
        name="data_block",
        prompt_key="fundamentals_analyst",
        shape=Shape.FENCED_BLOCK,
        parser=extract_metrics,
        success=_metrics_ok,
        block_name="DATA_BLOCK",
        required_fields=("debt_to_equity",),
    ),
    PromptContract(
        name="pm_block",
        prompt_key="portfolio_manager",
        shape=Shape.FENCED_BLOCK,
        parser=extract_pm_block,
        success=_pm_ok,
        block_name="PM_BLOCK",
        required_fields=("verdict", "zone", "risk_tally"),
    ),
    PromptContract(
        name="valuation_params",
        prompt_key="valuation_calculator",
        shape=Shape.FENCED_BLOCK,
        parser=_extract_params,
        success=_valuation_ok,
        block_name="VALUATION_PARAMS",
    ),
    PromptContract(
        name="value_trap",
        prompt_key="value_trap_detector",
        shape=Shape.FENCED_BLOCK,
        parser=extract_value_trap_score,
        success=_value_trap_ok,
        block_name="VALUE_TRAP_BLOCK",
        required_fields=("score", "verdict"),
    ),
    PromptContract(
        name="trade_block",
        prompt_key="trader",
        shape=Shape.UNFENCED_BLOCK,
        parser=parse_trade_block,
        success=_trade_ok,
        block_name="TRADE_BLOCK",
        required_fields=("action",),
    ),
    PromptContract(
        name="auditor_status",
        prompt_key="global_forensic_auditor",
        shape=Shape.LABELED_LINE,
        parser=parse_auditor_status,
        success=_auditor_clean,
        line_pattern=r"^\s*STATUS\s*[:=]",
        legacy_forms=(("STATUS: CLEAN", _auditor_clean),),
    ),
    PromptContract(
        name="rm_verdict",
        prompt_key="research_manager",
        shape=Shape.HEADER,
        parser=_classify_rm_verdict,
        success=_rm_classified,
        line_pattern=r"#*\s*(?:FINAL|INVESTMENT)?\s*RECOMMENDATION",
        legacy_forms=(
            ("### FINAL RECOMMENDATION: BUY", lambda r: r == "positive"),
            ("### INVESTMENT RECOMMENDATION: REJECT", lambda r: r == "negative"),
        ),
    ),
    PromptContract(
        name="legal_json",
        prompt_key="legal_counsel",
        shape=Shape.UNFENCED_BLOCK,
        parser=extract_legal_risks,
        success=_legal_ok,
    ),
    PromptContract(
        name="consultant",
        prompt_key="consultant",
        shape=Shape.UNFENCED_BLOCK,
        parser=lambda text: validate_required_output("consultant", text),
        success=_consultant_ok,
    ),
    PromptContract(
        name="consultant_breach_tokens",
        prompt_key="consultant",
        shape=Shape.LABELED_LINE,
        parser=parse_consultant_conditions,
        success=lambda result: (
            isinstance(result, dict) and "has_mandate_breach" in result
        ),
        line_pattern=r"^\*\*MANDATE_BREACH\*\*:",
        legacy_forms=(
            (
                "### FINAL CONSULTANT VERDICT\n\n"
                "**MANDATE_BREACH**: NONE\n**HARD_STOP**: NONE",
                lambda r: (
                    r["has_mandate_breach"] is False and r["has_hard_stop"] is False
                ),
            ),
            (
                "MANDATE BREACH: PFIC — company classified as PFIC.",
                lambda r: r["has_mandate_breach"] is True,
            ),
            (
                "HARD STOP: RESTRICTED — NS-CMIC listed entity.",
                lambda r: r["has_hard_stop"] is True,
            ),
        ),
    ),
    PromptContract(
        name="health_score_breakdown",
        prompt_key="fundamentals_analyst",
        shape=Shape.LABELED_LINE,
        parser=parse_score_breakdown,
        success=lambda result: (
            isinstance(result, dict) and set(result) == set(HEALTH_SCORE_CRITERIA)
        ),
        line_pattern=r"^HEALTH_SCORE_BREAKDOWN:",
        legacy_forms=(
            (
                "HEALTH_SCORE_BREAKDOWN: ROE=1; ROA=0.5; OPERATING_MARGIN=0; "
                "DE_RATIO=REMOVED; NET_DEBT_EBITDA=N/A; CURRENT_RATIO=1; "
                "OCF_POSITIVE=1; FCF_POSITIVE=1; FCF_YIELD=N/A; PE_OR_PEG=1; "
                "EV_EBITDA=N/A; PB_OR_PS=0",
                lambda result: (
                    isinstance(result, dict)
                    and set(result) == set(HEALTH_SCORE_CRITERIA)
                    and result["DE_RATIO"] == "REMOVED"
                ),
            ),
        ),
    ),
    PromptContract(
        name="material_events_status",
        prompt_key="news_analyst",
        shape=Shape.LABELED_LINE,
        parser=extract_material_events_status,
        success=lambda result: result in {"FOUND", "NONE_FOUND"},
        line_pattern=r"^[\s*-]*MATERIAL_EVENTS_90D\*{0,2}:",
        legacy_forms=(
            ("MATERIAL_EVENTS_90D: NONE_FOUND", lambda r: r == "NONE_FOUND"),
            # Pre-v5.4 prose (the 6831.HK 2026-07-01 phrasing).
            (
                "No material operational events have been reported in the last "
                "90 days.",
                lambda r: r == "NONE_FOUND",
            ),
        ),
    ),
)
