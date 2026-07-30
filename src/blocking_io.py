from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from src.async_utils import run_with_hard_timeout

T = TypeVar("T")


@dataclass(frozen=True)
class BlockingCallPolicy:
    label: str
    hard_timeout_seconds: float

    def with_label(self, label: str) -> BlockingCallPolicy:
        return replace(self, label=label)


async def run_blocking_call(policy: BlockingCallPolicy, fn: Callable[[], T]) -> T:
    if policy.hard_timeout_seconds <= 0:
        raise ValueError(
            "hard_timeout_seconds must be positive, "
            f"got {policy.hard_timeout_seconds!r}"
        )
    return await run_with_hard_timeout(
        asyncio.to_thread(fn),
        timeout=policy.hard_timeout_seconds,
        label=policy.label,
    )


YFINANCE_INFO_POLICY = BlockingCallPolicy("yfinance.info", 5.0)
YAHOOQUERY_QUOTE_TYPE_POLICY = BlockingCallPolicy("yahooquery.quote_type", 5.0)
OUTPUT_COMPANY_NAME_POLICY = BlockingCallPolicy("output_company_name", 6.0)
FX_RATE_POLICY = BlockingCallPolicy("fx_rate", 8.0)
