"""Load committed regression inputs shared across focused test modules."""

import json
from pathlib import Path
from typing import Any

_FROZEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "frozen"


def load_frozen_regression(name: str) -> dict[str, Any]:
    return json.loads((_FROZEN_DIR / name).read_text(encoding="utf-8"))
