"""What the machine was when a verdict was formed.

The retrospective compares a past decision to a realized outcome. That comparison
is only meaningful if the two runs are comparable, and nothing in a persisted
artifact recorded whether the *tooling* had changed between them: no git SHA, no
prompt digest, no binding digest, no thesis-constant digest. ``app_release`` is a
hand-edited string. So "did the verdict move because the model moved?" was not an
answerable question.

Four axes, each reusing machinery that already exists:

* **code** — ``src.eval.git_meta.get_git_metadata``
* **prompts** — ``compute_prompt_set_digest`` over the digests that
  ``prompts_used`` now carries. Deliberately the prompts *actually used*, not the
  files on disk: ``get_prompt()`` is mutable by ``PROMPT_*`` env vars and by
  Langfuse retrieval, and a file-level digest would both miss those and report a
  change when an unused prompt moves.
* **bindings** — ``BindingPlan.telemetry``, already secret-free, already carrying
  per-seat vendor, lineage, adapter, endpoint host, model and intent. This is the
  most direct answer to "which models changed".
* **thesis** — the introspected public constants of ``src.thesis_constants``, so a
  new threshold is covered with no edit here.

Best-effort throughout: this is provenance, and provenance must never cost a run
its artifact. Every resolver degrades to ``None``.

**Provenance is not causation.** ``compare()`` reports ``CHANGED`` to say the two
runs are not directly comparable. It never asserts that the change *caused* the
outcome — that inference belongs to a human reading the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import structlog

from src.eval.prompt_digest import stable_digest

logger = structlog.get_logger(__name__)

CONTEXT_SAME = "SAME"
CONTEXT_CHANGED = "CHANGED"
CONTEXT_UNKNOWN = "UNKNOWN"

# The axes compared by `RunFingerprint.compare`. `code_dirty` is deliberately not
# among them: it is a *disqualifier*, not a comparand.
_COMPARED_AXES = (
    "code_commit",
    "prompt_set_digest",
    "binding_digest",
    "thesis_digest",
)


@dataclass(frozen=True, slots=True)
class RunFingerprint:
    """Immutable identity of the code, prompts, bindings and thresholds of a run."""

    code_commit: str | None = None
    code_dirty: bool = False
    prompt_set_digest: str | None = None
    binding_digest: str | None = None
    thesis_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_commit": self.code_commit,
            "code_dirty": self.code_dirty,
            "prompt_set_digest": self.prompt_set_digest,
            "binding_digest": self.binding_digest,
            "thesis_digest": self.thesis_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RunFingerprint | None:
        """Rebuild from a persisted snapshot; ``None`` when absent or malformed."""
        if not isinstance(payload, dict):
            return None
        return cls(
            code_commit=payload.get("code_commit"),
            code_dirty=bool(payload.get("code_dirty", False)),
            prompt_set_digest=payload.get("prompt_set_digest"),
            binding_digest=payload.get("binding_digest"),
            thesis_digest=payload.get("thesis_digest"),
        )

    def compare(self, other: RunFingerprint | None) -> str:
        """``SAME`` | ``CHANGED`` | ``UNKNOWN``. Unknown is never silently SAME.

        A dirty worktree on *either* side forces ``UNKNOWN``: two runs at one
        commit with uncommitted edits are not the same machine, and the commit
        alone cannot say how they differed.

        A missing axis on either side also forces ``UNKNOWN`` rather than being
        skipped — absence of evidence about comparability is not evidence of it.
        """
        if other is None:
            return CONTEXT_UNKNOWN
        if self.code_dirty or other.code_dirty:
            return CONTEXT_UNKNOWN

        mine = self.to_dict()
        theirs = other.to_dict()
        for axis in _COMPARED_AXES:
            if mine[axis] is None or theirs[axis] is None:
                return CONTEXT_UNKNOWN
            if mine[axis] != theirs[axis]:
                return CONTEXT_CHANGED
        return CONTEXT_SAME


def _code_metadata() -> tuple[str | None, bool]:
    try:
        from src.eval.git_meta import get_git_metadata

        meta = get_git_metadata()
    except Exception:
        return None, False
    commit = meta.get("git_commit")
    return (str(commit) if commit else None), bool(meta.get("dirty", False))


def _prompt_set_digest(prompts_used: Any) -> str | None:
    """Digest the prompts a run actually used.

    Returns ``None`` when no prompt carries a digest — an honest absence beats a
    hash of ``{}``, which is a constant and would compare ``SAME`` across two runs
    with entirely different prompts.
    """
    if not isinstance(prompts_used, dict) or not prompts_used:
        return None
    has_digest = any(
        isinstance(entry, dict) and entry.get("digest")
        for entry in prompts_used.values()
    )
    if not has_digest:
        return None
    try:
        from src.eval.prompt_provenance import compute_prompt_set_digest

        return compute_prompt_set_digest(prompts_used)
    except Exception:
        return None


def _binding_digest(settings: Any) -> str | None:
    """Digest the resolved seat bindings: vendor, lineage, model and intent."""
    try:
        from src.llm_runtime.bindings import resolve_binding_plan

        plan = resolve_binding_plan(settings)
        telemetry = plan.telemetry(settings)
    except Exception as exc:
        logger.debug("binding_digest_unavailable", reason=type(exc).__name__)
        return None
    try:
        return stable_digest(telemetry)
    except (TypeError, ValueError):
        # Telemetry is documented as secret-free and JSON-shaped; if a future
        # field is not serializable, degrade rather than break persistence.
        logger.debug("binding_digest_unserializable")
        return None


def _thesis_digest() -> str | None:
    """Digest the public thresholds, by introspection rather than a fixed list.

    A hardcoded field list is exactly how a capability table rots: a new
    threshold would be silently uncovered. Only JSON-shaped public values are
    included, so a helper function or an imported module in that namespace does
    not perturb the hash.
    """
    try:
        from src import thesis_constants

        payload = {
            name: value
            for name, value in vars(thesis_constants).items()
            if not name.startswith("_")
            and isinstance(value, int | float | str | bool | dict | list | tuple)
        }
        return stable_digest(payload)
    except Exception:
        return None


def compute_run_fingerprint(
    prompts_used: Any = None, settings: Any = None
) -> RunFingerprint:
    """Best-effort fingerprint for the current process. Never raises.

    ``prompts_used`` is per-run so it is a parameter rather than cached; the other
    three axes are process-constant and resolved once (``get_git_metadata`` shells
    out four times).
    """
    code_commit, code_dirty = _cached_code_metadata()
    return RunFingerprint(
        code_commit=code_commit,
        code_dirty=code_dirty,
        prompt_set_digest=_prompt_set_digest(prompts_used),
        binding_digest=_cached_binding_digest(settings),
        thesis_digest=_cached_thesis_digest(),
    )


@lru_cache(maxsize=1)
def _cached_code_metadata() -> tuple[str | None, bool]:
    return _code_metadata()


@lru_cache(maxsize=1)
def _cached_thesis_digest() -> str | None:
    return _thesis_digest()


def _cached_binding_digest(settings: Any) -> str | None:
    # Not lru_cached: `Settings` is mutable and is not hashable, and a stale
    # binding digest is worse than re-resolving — the same reasoning that keeps
    # the binding plan itself uncached.
    if settings is None:
        from src.config import config as settings

    return _binding_digest(settings)


def reset_fingerprint_caches() -> None:
    """Clear the process-constant caches. For tests that mutate the repo state."""
    _cached_code_metadata.cache_clear()
    _cached_thesis_digest.cache_clear()
