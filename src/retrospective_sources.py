"""Where the retrospective looks for saved prediction snapshots.

Kept out of ``src/retrospective.py`` so that module stays free of configuration
parsing: it receives directories, it does not decide them. That split is also
what lets tests drive the loader over a temporary corpus without touching
``Settings``.

Local retention moves ``*_analysis.json`` artifacts out of ``RESULTS_DIR`` at
around 120 days — which lands *inside* the 90-270 day window
``TEMPORAL_WEIGHTS`` scores highest — so the archive is not optional history for
this consumer. ``scripts/eval_longitudinal_compare.py`` solved the same problem
with ``--archive-dir``; this is the settings-driven equivalent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def parse_archive_dirs(raw: str | None) -> tuple[Path, ...]:
    """Split a path-separator-delimited setting into expanded paths.

    Blank entries are dropped; order and duplicates-by-string are preserved for
    the caller to resolve, since only it knows the primary directory.
    """
    if not raw:
        return ()
    parts = [part.strip() for part in str(raw).split(os.pathsep)]
    return tuple(Path(part).expanduser() for part in parts if part)


def resolve_retrospective_sources(cfg: Any) -> tuple[Path, ...]:
    """Return the ordered directories to scan: live results first, then archives.

    Order is the precedence rule — the loader keeps the first artifact it sees for
    a given identity, so a file present in both trees resolves to the live copy.

    A configured directory that does not exist is *reported* and then dropped: a
    typo in ``RETROSPECTIVE_ARCHIVE_DIRS`` would otherwise silently halve the
    corpus with no signal at all.
    """
    primary = Path(getattr(cfg, "results_dir", "results")).expanduser()
    sources: list[Path] = [primary]

    for candidate in parse_archive_dirs(getattr(cfg, "retrospective_archive_dirs", "")):
        if candidate == primary or candidate in sources:
            continue
        if not candidate.is_dir():
            logger.warning(
                "retrospective_archive_dir_missing",
                path=str(candidate),
                msg=(
                    "Configured in RETROSPECTIVE_ARCHIVE_DIRS but not a directory; "
                    "archived snapshots from it will not be evaluated"
                ),
            )
            continue
        sources.append(candidate)

    return tuple(sources)
