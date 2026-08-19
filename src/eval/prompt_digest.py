from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_digest(payload: Any) -> str:
    """Return a stable content digest for any JSON-serializable payload.

    Named for what it does rather than for its first caller: it is a generic
    canonical-JSON hasher, and the run fingerprint uses it for thesis constants
    and binding plans as well as prompts.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Retained spelling for the baseline-capture call sites.
prompt_digest = stable_digest


def agent_prompt_payload(prompt: Any) -> dict[str, Any]:
    """The canonical digestible view of a loaded prompt.

    Shared by the baseline-capture path and the normal analysis path so a given
    prompt hashes identically in both. Includes ``system_message``, which is what
    makes an edit *without* a version bump visible — and what makes an env or
    Langfuse override visible, since those replace the text a run actually used.
    """
    return {
        "agent_key": prompt.agent_key,
        "agent_name": prompt.agent_name,
        "version": prompt.version,
        "system_message": prompt.system_message,
        "category": prompt.category,
        "requires_tools": prompt.requires_tools,
    }


def agent_prompt_digest(prompt: Any) -> str:
    """Digest a loaded prompt as it will actually be used this run.

    Returns ``""`` rather than raising when the prompt carries a field that is
    not JSON-serializable. This is provenance: it must never cost a run its
    work, and an empty digest is already how consumers spell "not recorded"
    (``_prompt_set_digest`` skips falsy entries and reports ``None`` when no
    prompt contributes one). Silence here degrades the fingerprint; an exception
    would degrade the analysis.
    """
    try:
        return stable_digest(agent_prompt_payload(prompt))
    except (TypeError, ValueError, AttributeError):
        return ""
