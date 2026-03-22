from __future__ import annotations

import hashlib
import json
from typing import Any


def prompt_digest(prompt_payload: dict[str, Any]) -> str:
    """Return a stable digest for a prompt payload."""
    canonical = json.dumps(
        prompt_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
