from __future__ import annotations

from pathlib import Path

from src.config import Settings


def initialize_runtime_environment(settings: Settings) -> None:
    """Create filesystem directories required by the live runtime.

    Settings construction intentionally does not create directories. CLI and
    worker entrypoints call this explicit setup step once runtime paths are
    known and command-line overrides have been applied.
    """
    for directory in settings.runtime_directories():
        Path(directory).mkdir(parents=True, exist_ok=True)
