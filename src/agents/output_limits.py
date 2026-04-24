from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_MAX_STATE_FIELD_CHARS = 200_000


def cap_state_value(
    value: str, field: str, *, max_chars: int = _MAX_STATE_FIELD_CHARS
) -> str:
    """Bound large persisted artifacts so one response cannot dominate state."""
    if len(value) <= max_chars:
        return value

    logger.warning(
        "state_field_capped",
        field=field,
        original_len=len(value),
        max_chars=max_chars,
    )
    return value[:max_chars] + "\n[...truncated]"
