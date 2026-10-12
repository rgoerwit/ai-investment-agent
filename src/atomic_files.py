"""Atomic publication of text artifacts: the one writer of finished files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete text artifact with one filesystem replacement.

    The destination is either its previous content or the new content, never
    a partial file, and an existing destination survives a failed write. The
    content is flushed to disk before the replacement, so a crash right after
    it cannot leave the new name pointing at an empty file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
