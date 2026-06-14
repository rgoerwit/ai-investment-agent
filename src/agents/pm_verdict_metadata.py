from __future__ import annotations

import re
from typing import Literal, cast

from pydantic import BaseModel

PMVerdict = Literal[
    "BUY",
    "HOLD",
    "SELL",
    "DO_NOT_INITIATE",
    "UNPARSEABLE",
]


class PMVerdictMetadata(BaseModel):
    verdict: PMVerdict


class PMVerdictRecovery(BaseModel):
    verdict: Literal["BUY", "HOLD", "SELL", "DO_NOT_INITIATE"]


def canonicalize_pm_verdict(raw: str | None) -> PMVerdict:
    """Return the canonical PM verdict label for free-text or block values."""
    cleaned = (raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if cleaned in {"DO_NOT_INITIATE", "DONOTINITIATE", "DONOTINITATE", "REJECT"}:
        return "DO_NOT_INITIATE"
    if cleaned in {"BUY", "HOLD", "SELL"}:
        return cast(PMVerdict, cleaned)
    return "UNPARSEABLE"


def pm_verdict_metadata_from_text(pm_output: str) -> PMVerdictMetadata:
    from src.charts.extractors.pm_block import (
        extract_pm_block,
        extract_verdict_from_text,
    )

    block = extract_pm_block(pm_output)
    verdict = block.verdict or extract_verdict_from_text(pm_output) or "UNPARSEABLE"
    return PMVerdictMetadata(verdict=canonicalize_pm_verdict(verdict))
