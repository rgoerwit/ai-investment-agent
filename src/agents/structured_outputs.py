from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel

from src.charts.extractors.pm_block import extract_pm_block, extract_verdict_from_text

PMVerdict = Literal[
    "BUY",
    "HOLD",
    "REJECT",
    "SELL",
    "DO_NOT_INITIATE",
    "UNPARSEABLE",
]


class PMVerdictMetadata(BaseModel):
    verdict: PMVerdict


def pm_verdict_metadata_from_text(pm_output: str) -> PMVerdictMetadata:
    block = extract_pm_block(pm_output)
    verdict = block.verdict or extract_verdict_from_text(pm_output) or "UNPARSEABLE"
    if verdict == "DO_NOT_INITIATE":
        normalized: PMVerdict = "DO_NOT_INITIATE"
    elif verdict in {"BUY", "HOLD", "REJECT", "SELL"}:
        normalized = cast(PMVerdict, verdict)
    else:
        normalized = "UNPARSEABLE"
    return PMVerdictMetadata(verdict=normalized)
