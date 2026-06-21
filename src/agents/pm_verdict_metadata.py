from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Canonicalization + the verdict label set live in the neutral, dependency-free
# parser so lightweight callers (charts, reporting, IBKR) can reuse them without
# importing src.agents. Re-exported here for backward compatibility.
from src.pm_decision_parser import PMVerdict, canonicalize_pm_verdict

__all__ = [
    "PMVerdict",
    "PMVerdictMetadata",
    "PMVerdictRecovery",
    "canonicalize_pm_verdict",
    "pm_verdict_metadata_from_text",
]


class PMVerdictMetadata(BaseModel):
    verdict: PMVerdict


class PMVerdictRecovery(BaseModel):
    verdict: Literal["BUY", "HOLD", "SELL", "DO_NOT_INITIATE"]


def pm_verdict_metadata_from_text(pm_output: str) -> PMVerdictMetadata:
    from src.charts.extractors.pm_block import (
        extract_pm_block,
        extract_verdict_from_text,
    )

    block = extract_pm_block(pm_output)
    verdict = block.verdict or extract_verdict_from_text(pm_output) or "UNPARSEABLE"
    return PMVerdictMetadata(verdict=canonicalize_pm_verdict(verdict))
