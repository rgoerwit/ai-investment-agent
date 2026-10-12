#!/usr/bin/env python3
"""Lay an existing .env out like .env.example without changing any value.

Reads the source and the template with python-dotenv's own parser, writes a
candidate file and prints counts only, never a key or a value. The candidate must
load to exactly the values the source loads to, or nothing is written. The
candidate replaces the output atomically, owner-only, and the output may not be
the source or the template under any name (hardlink or symlink included).
"""

import argparse
import io
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import Binding, parse_stream

from src.atomic_files import atomic_write_text

_LEADING_BLANKS = re.compile(r"(?:[ \t]*\n)*")
_SLOT = re.compile(r"^\s*#\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=")
_EXTRA_HEADER = "# --- Settings not in .env.example ---"
_CARRIED_HEADER = "# --- Comments carried over from the previous .env ---"


@dataclass(frozen=True)
class Alignment:
    text: str
    placed: int
    commented_defaults: int
    extra_keys: int
    carried: int


def _bindings(text: str) -> list[Binding]:
    return list(parse_stream(io.StringIO(text)))


def _is_setting(binding: Binding) -> bool:
    return binding.key is not None and not binding.error


def _is_comment(binding: Binding) -> bool:
    return binding.key is None and not binding.error


def _split(binding: Binding) -> tuple[str, str]:
    """Separate the blank lines the parser folds into a binding from its text."""
    raw = binding.original.string
    prefix = _LEADING_BLANKS.match(raw)
    assert prefix is not None
    return prefix.group(0), raw[prefix.end() :]


def _line(raw: str) -> str:
    return raw if raw.endswith("\n") else raw + "\n"


def _comment_out(body: str) -> str:
    return "".join(f"# {part}" for part in _line(body).splitlines(keepends=True))


def _user_text(binding: Binding) -> str:
    return _line(_split(binding)[1])


def _slot_key(binding: Binding) -> str | None:
    match = _SLOT.match(_split(binding)[1])
    return match.group(1) if match else None


def effective_values(text: str, *, interpolate: bool = True) -> dict[str, str | None]:
    return dict(dotenv_values(stream=io.StringIO(text), interpolate=interpolate))


_MISSING = object()


def mismatch_count(old: dict[str, str | None], new: dict[str, str | None]) -> int:
    return sum(
        1
        for key in old.keys() | new.keys()
        if old.get(key, _MISSING) != new.get(key, _MISSING)
    )


def align(source_text: str, template_text: str) -> Alignment:
    source = _bindings(source_text)
    template = _bindings(template_text)

    # dotenv keeps the last of a repeated key; earlier ones are carried commented out.
    last = {b.key: b for b in source if _is_setting(b)}
    superseded = [b for b in source if _is_setting(b) and last[b.key] is not b]
    active_keys = {b.key for b in template if _is_setting(b)}
    slot_keys = {k for b in template if _is_comment(b) and (k := _slot_key(b))}
    template_keys = active_keys | slot_keys

    out: list[str] = []
    placed: set[str] = set()
    commented_defaults = 0
    for binding in template:
        key = binding.key if _is_setting(binding) else None
        prefix, body = _split(binding)
        if key is not None:
            if key in last and key not in placed:
                placed.add(key)
                out.append(prefix + _user_text(last[key]))
            else:
                commented_defaults += 1
                out.append(prefix + _comment_out(body))
            continue
        slot = _slot_key(binding) if _is_comment(binding) else None
        if slot in last and slot not in active_keys and slot not in placed:
            placed.add(slot)
            out.append(prefix + _user_text(last[slot]))
        else:
            out.append(_line(binding.original.string))

    extras = [b for b in last.values() if b.key not in template_keys]
    template_comments = {_split(b)[1].strip() for b in template if _is_comment(b)}
    carried = [
        _line(_split(b)[1])
        for b in source
        if _is_comment(b)
        and _split(b)[1].strip()
        and _split(b)[1].strip() not in template_comments
    ]
    carried += [_comment_out(_split(b)[1]) for b in superseded]
    carried += [_line(_split(b)[1]) for b in source if b.error]

    if extras:
        out += ["\n", _EXTRA_HEADER + "\n"] + [_user_text(b) for b in extras]
    if carried:
        out += ["\n", _CARRIED_HEADER + "\n"] + carried
    return Alignment(
        text="".join(out),
        placed=len(placed),
        commented_defaults=commented_defaults,
        extra_keys=len(extras),
        carried=len(carried),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--template", type=Path, default=Path(".env.example"))
    parser.add_argument("--output", type=Path, default=Path(".env.aligned"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    for protected, label in ((args.source, "source"), (args.template, "template")):
        if args.output.resolve() == protected.resolve() or (
            args.output.exists() and os.path.samefile(args.output, protected)
        ):
            parser.error(f"refusing to overwrite the {label}")

    source_text = args.source.read_text(encoding="utf-8")
    result = align(source_text, args.template.read_text(encoding="utf-8"))
    # Raw values must match, and so must ${VAR} expansion, which reordering can change.
    bad = max(
        mismatch_count(
            effective_values(source_text, interpolate=interpolate),
            effective_values(result.text, interpolate=interpolate),
        )
        for interpolate in (False, True)
    )
    if bad:
        print(
            f"effective values differ for {bad} keys; nothing written", file=sys.stderr
        )
        return 2
    print(
        f"placed {result.placed}, commented-out template defaults "
        f"{result.commented_defaults}, extra keys {result.extra_keys}, "
        f"carried comments {result.carried}; effective values identical"
    )
    if not args.check:
        atomic_write_text(args.output, result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
