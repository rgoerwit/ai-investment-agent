from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

ProgressCallback = Callable[[str], None]
AnalysisRunner = Callable[..., Awaitable[dict | None]]
AnalysisSaver = Callable[..., Path]
CommandBuilder = Callable[..., str]
