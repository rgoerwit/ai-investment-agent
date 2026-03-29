from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_prompt_set_digest(prompts_used: dict[str, dict[str, Any]]) -> str:
    """Return a stable digest for the effective prompt set used in a run."""
    payload = {
        key: value["digest"]
        for key, value in sorted(prompts_used.items())
        if isinstance(value, dict) and value.get("digest")
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
