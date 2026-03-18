from __future__ import annotations

import re


def _compile_named_block_pattern(block_name: str) -> re.Pattern[str]:
    escaped_name = re.escape(block_name)
    return re.compile(
        rf"### --- START {escaped_name}[^\n]*---(.+?)### --- END {escaped_name} ---",
        re.DOTALL,
    )


DATA_BLOCK_PATTERN = _compile_named_block_pattern("DATA_BLOCK")


def extract_last_fenced_block(
    report: str | None,
    block_name: str,
    *,
    include_markers: bool = False,
) -> str | None:
    """Return the last parseable fenced block for the given structured block name."""
    if not report or not isinstance(report, str):
        return None

    blocks = list(_compile_named_block_pattern(block_name).finditer(report))
    if not blocks:
        return None

    last = blocks[-1]
    return last.group(0 if include_markers else 1)


def has_parseable_fenced_block(report: str | None, block_name: str) -> bool:
    """Return True only when the named fenced block can actually be parsed."""
    return (
        extract_last_fenced_block(report, block_name, include_markers=True) is not None
    )


def extract_last_data_block(
    report: str | None, *, include_markers: bool = False
) -> str | None:
    """Return the last parseable fenced DATA_BLOCK, if present."""
    return extract_last_fenced_block(
        report, "DATA_BLOCK", include_markers=include_markers
    )


def has_parseable_data_block(report: str | None) -> bool:
    """Return True only when a fenced DATA_BLOCK can actually be parsed."""
    return has_parseable_fenced_block(report, "DATA_BLOCK")
