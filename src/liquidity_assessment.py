"""Typed, deterministic liquidity assessment shared by tool, graph, and reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from src.data_block_utils import (
    build_fenced_block,
    extract_block_field,
    extract_block_number,
)
from src.thesis_constants import LIQUIDITY_MIN_USD, LIQUIDITY_PASS_USD

LiquidityStatus = Literal[
    "PASS",
    "MARGINAL",
    "FAIL_INSUFFICIENT_LIQUIDITY",
    "FAIL_IRREGULAR_TRADING",
    "INSUFFICIENT_DATA",
    "ERROR",
]

_HARD_FAIL_STATUSES = frozenset(
    {"FAIL_INSUFFICIENT_LIQUIDITY", "FAIL_IRREGULAR_TRADING"}
)
_VALID_STATUSES = frozenset(
    {
        "PASS",
        "MARGINAL",
        "FAIL_INSUFFICIENT_LIQUIDITY",
        "FAIL_IRREGULAR_TRADING",
        "INSUFFICIENT_DATA",
        "ERROR",
    }
)


def _finite_number(
    value: Any,
    *,
    maximum: float | None = None,
) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _status_matches_measurements(
    status: LiquidityStatus,
    *,
    turnover: float | None,
    zero_volume_days_pct: float | None,
    flat_price_days_pct: float | None,
) -> bool:
    """Reject corrupt handoffs whose code-owned status contradicts its metrics."""
    if status == "PASS":
        return turnover is not None and turnover >= LIQUIDITY_PASS_USD
    if status == "MARGINAL":
        return (
            turnover is not None and LIQUIDITY_MIN_USD <= turnover < LIQUIDITY_PASS_USD
        )
    if status == "FAIL_INSUFFICIENT_LIQUIDITY":
        return turnover is not None and turnover < LIQUIDITY_MIN_USD
    if status == "FAIL_IRREGULAR_TRADING":
        return bool(
            (zero_volume_days_pct is not None and zero_volume_days_pct > 15.0)
            or (flat_price_days_pct is not None and flat_price_days_pct > 30.0)
        )
    return True


@dataclass(frozen=True, slots=True)
class LiquidityAssessment:
    """One code-owned interpretation of measured USD trading liquidity."""

    status: LiquidityStatus
    average_daily_turnover_usd: float | None = None
    average_daily_volume: float | None = None
    zero_volume_days_pct: float | None = None
    flat_price_days_pct: float | None = None
    reason: str | None = None

    @property
    def hard_fail(self) -> bool:
        return self.status in _HARD_FAIL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "hard_fail": self.hard_fail,
            "average_daily_turnover_usd": self.average_daily_turnover_usd,
            "average_daily_volume": self.average_daily_volume,
            "zero_volume_days_pct": self.zero_volume_days_pct,
            "flat_price_days_pct": self.flat_price_days_pct,
            "reason": self.reason,
            "minimum_turnover_usd": LIQUIDITY_MIN_USD,
            "pass_turnover_usd": LIQUIDITY_PASS_USD,
        }

    def render_block(self) -> str:
        def number(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.6f}".rstrip("0").rstrip(".")

        return build_fenced_block(
            "LIQUIDITY_BLOCK",
            "\n".join(
                (
                    f"STATUS: {self.status}",
                    f"AVG_DAILY_TURNOVER_USD: {number(self.average_daily_turnover_usd)}",
                    f"AVG_DAILY_VOLUME: {number(self.average_daily_volume)}",
                    f"ZERO_VOLUME_DAYS_PCT: {number(self.zero_volume_days_pct)}",
                    f"FLAT_PRICE_DAYS_PCT: {number(self.flat_price_days_pct)}",
                    f"REASON: {self.reason or 'NONE'}",
                )
            ),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> LiquidityAssessment | None:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
            return None
        status = raw.get("status")
        if status not in _VALID_STATUSES:
            return None

        typed_status = cast(LiquidityStatus, status)
        turnover = _finite_number(raw.get("average_daily_turnover_usd"))
        volume = _finite_number(raw.get("average_daily_volume"))
        zero_volume_pct = _finite_number(raw.get("zero_volume_days_pct"), maximum=100.0)
        flat_price_pct = _finite_number(raw.get("flat_price_days_pct"), maximum=100.0)
        if not _status_matches_measurements(
            typed_status,
            turnover=turnover,
            zero_volume_days_pct=zero_volume_pct,
            flat_price_days_pct=flat_price_pct,
        ):
            return None

        return cls(
            status=typed_status,
            average_daily_turnover_usd=turnover,
            average_daily_volume=volume,
            zero_volume_days_pct=zero_volume_pct,
            flat_price_days_pct=flat_price_pct,
            reason=str(raw.get("reason")) if raw.get("reason") else None,
        )

    @classmethod
    def from_tool_output(cls, output: str) -> LiquidityAssessment | None:
        status = extract_block_field(output, "LIQUIDITY_BLOCK", "STATUS")
        if status not in _VALID_STATUSES:
            return None
        typed_status = cast(LiquidityStatus, status)
        turnover = _finite_number(
            extract_block_number(output, "LIQUIDITY_BLOCK", "AVG_DAILY_TURNOVER_USD")
        )
        volume = _finite_number(
            extract_block_number(output, "LIQUIDITY_BLOCK", "AVG_DAILY_VOLUME")
        )
        zero_volume_pct = _finite_number(
            extract_block_number(output, "LIQUIDITY_BLOCK", "ZERO_VOLUME_DAYS_PCT"),
            maximum=100.0,
        )
        flat_price_pct = _finite_number(
            extract_block_number(output, "LIQUIDITY_BLOCK", "FLAT_PRICE_DAYS_PCT"),
            maximum=100.0,
        )
        if not _status_matches_measurements(
            typed_status,
            turnover=turnover,
            zero_volume_days_pct=zero_volume_pct,
            flat_price_days_pct=flat_price_pct,
        ):
            return None
        reason = extract_block_field(output, "LIQUIDITY_BLOCK", "REASON")
        return cls(
            status=typed_status,
            average_daily_turnover_usd=turnover,
            average_daily_volume=volume,
            zero_volume_days_pct=zero_volume_pct,
            flat_price_days_pct=flat_price_pct,
            reason=None if reason in {None, "NONE"} else reason,
        )


def latest_liquidity_assessment(
    tool_records: Iterable[Any],
) -> LiquidityAssessment | None:
    """Return the latest parseable result from the canonical liquidity tool."""
    latest: LiquidityAssessment | None = None
    for record in tool_records:
        tool_name = getattr(record, "tool_name", None)
        content = getattr(record, "content", None)
        if tool_name != "calculate_liquidity_metrics" or not isinstance(content, str):
            continue
        latest = LiquidityAssessment.from_tool_output(content) or latest
    return latest


def liquidity_fast_fail_update(raw: Any) -> dict[str, Any]:
    """Translate a typed hard fail into the graph's existing reject contract."""
    assessment = LiquidityAssessment.from_dict(raw)
    if assessment is None or not assessment.hard_fail:
        return {}
    turnover = assessment.average_daily_turnover_usd
    turnover_text = "unavailable" if turnover is None else f"${turnover:,.0f}"
    return {
        "pre_screening_result": "REJECT",
        "red_flags": [
            {
                "type": "LIQUIDITY_HARD_FAIL",
                "severity": "CRITICAL",
                "action": "AUTO_REJECT",
                "blocks_buy": True,
                "risk_penalty": 0.0,
                "detail": (
                    f"Measured average daily turnover {turnover_text}; "
                    f"status={assessment.status}; minimum=${LIQUIDITY_MIN_USD:,}."
                ),
            }
        ],
    }
