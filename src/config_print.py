"""Print one numeric setting for a shell caller: ``python -m src.config_print <key>``.

The shell side of a contract (``scripts/run_pipeline.sh``'s per-ticker
watchdogs) reads its defaults here so ``Settings`` stays the one source of
truth: a literal in the script would drift from the field silently. The value
is resolved exactly as the application resolves it (shell environment, then
``.env``, then the field default).

Only numeric and boolean fields are printed. Anything else, including every
secret, exits non-zero, so this can never become a way to echo a credential.
Import-time output is diverted to stderr so stdout carries the value alone.

Exit status: 0 with the value on stdout; 2 for a usage error, an unknown key or
a non-numeric field; 1 when the settings themselves fail to load.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence

USAGE = "usage: python -m src.config_print <settings_field_name>"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or not args[0].strip():
        print(USAGE, file=sys.stderr)
        return 2
    key = args[0].strip()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from src.config import Settings, config
    except Exception as exc:  # noqa: BLE001 - a shell caller needs a status, not a trace
        print(
            f"config_print: settings failed to load ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1
    if key not in Settings.model_fields:
        print(f"config_print: unknown setting {key!r}", file=sys.stderr)
        return 2
    value = getattr(config, key)
    if isinstance(value, bool):
        print("true" if value else "false")
        return 0
    if not isinstance(value, int | float):
        print(
            f"config_print: {key!r} is not a numeric or boolean setting",
            file=sys.stderr,
        )
        return 2
    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
