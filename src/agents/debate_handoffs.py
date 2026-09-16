"""Balanced, optional rationale handoffs for the Bull/Bear debate.

The handoff is an adjunct to the canonical argument, never evidence.  Round 1
produces role-private candidates in parallel; the sync barrier publishes a
component only when both roles produced it.  That symmetry prevents provider
capability or response-shape differences from giving either side extra influence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

POLICY_VERSION = 5
STRUCTURED_RATIONALE_START = "### --- START DEBATE_RATIONALE ---"
STRUCTURED_RATIONALE_END = "### --- END DEBATE_RATIONALE ---"

_STRUCTURED_FIELDS = (
    "POSITION",
    "DECISIVE_PREMISES",
    "EVIDENCE_REFERENCES",
    "INFERENCE_BRIDGE",
    "STRONGEST_DISCONFIRMING_EVIDENCE",
    "MATERIAL_UNCERTAINTIES",
    "WHAT_WOULD_CHANGE_MY_VIEW",
)
_FIELD_PATTERN = "|".join(_STRUCTURED_FIELDS)
_FIELD_RE = re.compile(
    rf"(?i)(?:^|\s)(?:#{{1,6}}\s*)?(?:[-*]\s*)?"
    rf"(?:\*\*|__|[\"'])?({_FIELD_PATTERN})"
    rf"(?:\*\*|__|[\"'])?\s*:\s*(?:\*\*|__)?"
)
_CORE_STRUCTURED_FIELDS = {
    "POSITION",
    "DECISIVE_PREMISES",
    "EVIDENCE_REFERENCES",
}


@dataclass(frozen=True, slots=True)
class DebateReasoningPolicy:
    """Run-scoped policy for the repository's one- or two-round debate."""

    enabled: bool
    max_rounds: int
    structured_char_cap: int = 2_000
    native_char_cap: int = 2_000
    researcher_output_bonus: int = 1_024
    research_manager_output_bonus: int = 1_024
    structured_repair_output_tokens: int = 1_024

    def __post_init__(self) -> None:
        if self.max_rounds not in {1, 2}:
            raise ValueError("debate reasoning handoffs support one or two rounds")
        if self.structured_char_cap <= 0 or self.native_char_cap <= 0:
            raise ValueError("debate reasoning handoff caps must be positive")
        if self.researcher_output_bonus < 0 or self.research_manager_output_bonus < 0:
            raise ValueError(
                "debate reasoning handoff budget bonuses cannot be negative"
            )
        if self.structured_repair_output_tokens <= 0:
            raise ValueError(
                "structured rationale repair output tokens must be positive"
            )

    @property
    def active(self) -> bool:
        """Whether this run can emit and later consume a balanced handoff."""

        return self.enabled and self.max_rounds > 1

    def emits_in(self, round_num: int) -> bool:
        self._validate_round(round_num)
        return self.active and round_num < self.max_rounds

    def consumes_in(self, round_num: int) -> bool:
        self._validate_round(round_num)
        return self.active and round_num > 1

    def output_bonus(self, agent_name: str) -> int:
        if not self.active:
            return 0
        if agent_name in {"Bull Researcher", "Bear Researcher"}:
            return self.researcher_output_bonus
        if agent_name == "Research Manager":
            return self.research_manager_output_bonus
        return 0

    def _validate_round(self, round_num: int) -> None:
        if not 1 <= round_num <= self.max_rounds:
            raise ValueError(
                f"round {round_num} is outside the configured 1..{self.max_rounds} debate"
            )


def structured_rationale_addendum(policy: DebateReasoningPolicy, round_num: int) -> str:
    """Return the code-owned capsule request for a non-final debate round."""

    if not policy.emits_in(round_num):
        return ""
    fields = "\n".join(f"{field}:" for field in _STRUCTURED_FIELDS)
    return f"""

MANDATORY ROUND-1 OUTPUT CONTRACT: Write the complete canonical argument first,
then append the bounded rationale capsule below after it. The argument is the
deliverable; the capsule is an adjunct, so a response cut off at the output cap
must lose only the capsule.
The response is incomplete if either marker is absent. The capsule is a concise,
evidence-linked explanation for another debate agent, not private chain of
thought. Do not add facts that are absent from the supplied evidence, and do not
put the main argument inside the capsule. Keep the entire capsule under
{policy.structured_char_cap:,} characters.

{STRUCTURED_RATIONALE_START}
{fields}
{STRUCTURED_RATIONALE_END}
"""


def structured_rationale_repair_prompt(policy: DebateReasoningPolicy) -> str:
    """Return a narrow post-generation structuring request.

    The repair sees only the canonical argument, not hidden reasoning or source
    reports. It therefore restructures an existing claim set rather than creating
    a second substantive opinion.
    """

    fields = "\n".join(f"{field}:" for field in _STRUCTURED_FIELDS)
    return f"""Convert the supplied canonical Round-1 argument into the exact
bounded capsule below. This is structure-only extraction, not a new analysis.
Do not add facts or implications absent from the argument. For a required field
the argument does not address, write NOT STATED. Return only the marked capsule,
with no preamble or trailing prose, under {policy.structured_char_cap:,} characters.

{STRUCTURED_RATIONALE_START}
{fields}
{STRUCTURED_RATIONALE_END}
"""


def split_structured_rationale(
    content: str,
    *,
    policy: DebateReasoningPolicy,
    round_num: int,
) -> tuple[str, str]:
    """Separate canonical argument text from a valid bounded capsule.

    Any marked block is removed from the canonical argument even when malformed,
    so a parser miss cannot leak rationale scaffolding into reports or memory.
    Suffix capsules are the live contract — truncation then costs the adjunct
    rather than the argument, and an unterminated capsule still leaves the
    canonical argument whole for the repair path to restructure. Prefix capsules
    remain readable for captures and responses produced by earlier policies.
    """

    start = content.find(STRUCTURED_RATIONALE_START)
    if start < 0:
        return content.strip(), ""

    end = content.find(
        STRUCTURED_RATIONALE_END, start + len(STRUCTURED_RATIONALE_START)
    )
    if end < 0:
        return content[:start].rstrip(), ""

    before = content[:start].strip()
    after = content[end + len(STRUCTURED_RATIONALE_END) :].strip()
    canonical = "\n\n".join(part for part in (before, after) if part)
    if not policy.emits_in(round_num):
        return canonical, ""

    capsule = content[start + len(STRUCTURED_RATIONALE_START) : end].strip()
    return canonical, _validate_structured_rationale(capsule, policy=policy)


def parse_structured_rationale_candidate(
    content: str, *, policy: DebateReasoningPolicy
) -> str:
    """Validate a dedicated repair response, tolerating omitted wrapper markers."""

    if STRUCTURED_RATIONALE_START in content or STRUCTURED_RATIONALE_END in content:
        _, capsule = split_structured_rationale(content, policy=policy, round_num=1)
        return capsule
    return _validate_structured_rationale(content.strip(), policy=policy)


def _validate_structured_rationale(
    capsule: str, *, policy: DebateReasoningPolicy
) -> str:
    matches = list(_FIELD_RE.finditer(capsule))
    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        end_of_value = matches[index + 1].start() if index + 1 < len(matches) else None
        values[match.group(1).upper()] = capsule[match.end() : end_of_value].strip()
    populated = {field for field, value in values.items() if value}
    if not _CORE_STRUCTURED_FIELDS.issubset(populated) or len(populated) < 5:
        return ""
    if (
        all(values.get(field) for field in _STRUCTURED_FIELDS)
        and len(capsule) <= policy.structured_char_cap
    ):
        return capsule
    return _normalize_structured_rationale(values, policy=policy)


def _normalize_structured_rationale(
    values: dict[str, str], *, policy: DebateReasoningPolicy
) -> str:
    """Bound a substantively complete capsule despite superficial shape drift."""

    separator_chars = sum(len(field) + 3 for field in _STRUCTURED_FIELDS) - 1
    available = policy.structured_char_cap - separator_chars
    if available < len(_STRUCTURED_FIELDS):
        return ""

    # Premises and evidence carry most of the signal, while every field retains
    # a floor so truncation cannot erase uncertainty or the view-change condition.
    weights = (1, 3, 3, 2, 2, 2, 2)
    floor = min(40, available // len(_STRUCTURED_FIELDS))
    remaining = available - floor * len(_STRUCTURED_FIELDS)
    allocations = [floor + remaining * weight // sum(weights) for weight in weights]
    allocations[-1] += available - sum(allocations)

    lines: list[str] = []
    for field, allocation in zip(_STRUCTURED_FIELDS, allocations, strict=True):
        value = values.get(field) or "NOT STATED"
        if len(value) > allocation:
            value = value[: max(1, allocation - 1)].rstrip() + "…"
        lines.append(f"{field}: {value}")
    normalized = "\n".join(lines)
    return normalized if len(normalized) <= policy.structured_char_cap else ""


def extract_native_reasoning(response: Any, *, char_cap: int) -> str:
    """Read only LangChain-normalized, human-readable reasoning blocks.

    Provider-native signatures and encrypted/opaque blocks are intentionally not
    inspected or forwarded.  Missing or malformed content is simply unavailable.
    """

    try:
        blocks = response.content_blocks
    except Exception:  # noqa: BLE001 - optional provider metadata must fail open
        return ""
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        reasoning = block.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            parts.append(reasoning.strip())
    return _bounded("\n\n".join(parts), char_cap)


def paired_handoff_telemetry(
    *,
    policy: DebateReasoningPolicy,
    bull: dict[str, str],
    bear: dict[str, str],
) -> dict[str, Any]:
    """Resolve the only publishable shape: symmetric component pairs."""

    structured_pair = bool(
        policy.active and bull.get("structured") and bear.get("structured")
    )
    native_pair = bool(policy.active and bull.get("native") and bear.get("native"))
    return {
        "policy_version": POLICY_VERSION,
        "policy_active": policy.active,
        "barrier_reported": True,
        "published_rounds": [1] if structured_pair or native_pair else [],
        "structured_pair": structured_pair,
        "native_pair": native_pair,
        "structured_lengths": {
            "bull": len(bull.get("structured", "")) if structured_pair else 0,
            "bear": len(bear.get("structured", "")) if structured_pair else 0,
        },
        "native_lengths": {
            "bull": len(bull.get("native", "")) if native_pair else 0,
            "bear": len(bear.get("native", "")) if native_pair else 0,
        },
    }


def seed_handoff_telemetry(*, policy_active: bool) -> dict[str, Any]:
    """Initial telemetry, written into graph state before the debate runs.

    Seeding rather than defaulting to ``{}`` is what lets a serializer tell a
    disabled run from one whose R1 sync barrier never reported, without reading
    ambient run configuration at persistence time.
    """

    return {
        "policy_version": POLICY_VERSION,
        "policy_active": policy_active,
        "barrier_reported": False,
        "published_rounds": [],
        "structured_pair": False,
        "native_pair": False,
        "structured_lengths": {"bull": 0, "bear": 0},
        "native_lengths": {"bull": 0, "bear": 0},
    }


def sanitize_handoff_telemetry(raw: Any) -> dict[str, Any]:
    """Return the persistable projection of a telemetry dict, allowlisted.

    ``policy_active`` is what the run was configured to do and
    ``barrier_reported`` is whether the R1 sync barrier ran; the pair fields are
    what it measured. One boolean cannot honestly carry the first two, which is
    why they are separate rather than a single ``enabled``.

    Content never leaves graph state — only presence and lengths. A
    ``policy_version`` this module does not recognize is reported verbatim
    rather than silently downgraded, matching ``provenance_schema``'s
    fail-closed convention: a reader can then see that it is looking at a shape
    written by different code.
    """

    values = raw if isinstance(raw, dict) else {}
    policy_active = values.get("policy_active") is True
    barrier_reported = values.get("barrier_reported") is True
    structured_pair = barrier_reported and values.get("structured_pair") is True
    native_pair = barrier_reported and values.get("native_pair") is True

    def lengths(component: str, paired: bool) -> dict[str, int]:
        # Lengths and their pair flag can never disagree: an unusable value
        # zeroes the length *and* the pair, rather than persisting a record
        # claiming a published pair of length zero.
        raw_lengths = values.get(component)
        if not paired or not isinstance(raw_lengths, dict):
            return {"bull": 0, "bear": 0}
        resolved: dict[str, int] = {}
        for role in ("bull", "bear"):
            value = raw_lengths.get(role)
            resolved[role] = value if type(value) is int and value >= 0 else 0
        return resolved

    structured_lengths = lengths("structured_lengths", structured_pair)
    native_lengths = lengths("native_lengths", native_pair)
    # Both legs, not either: the barrier publishes a component only when both
    # roles produced it, so a record claiming a pair with one zero leg is false
    # telemetry. `all` also never rejects a genuine pair — paired_handoff_telemetry
    # requires both strings to be non-empty, hence both lengths >= 1.
    structured_pair = structured_pair and all(structured_lengths.values())
    native_pair = native_pair and all(native_lengths.values())

    version = values.get("policy_version")
    return {
        "policy_version": version if type(version) is int and version > 0 else None,
        "policy_active": policy_active,
        "barrier_reported": barrier_reported,
        "published_rounds": [1] if structured_pair or native_pair else [],
        "structured_pair": structured_pair,
        "native_pair": native_pair,
        "structured_lengths": structured_lengths
        if structured_pair
        else {
            "bull": 0,
            "bear": 0,
        },
        "native_lengths": native_lengths if native_pair else {"bull": 0, "bear": 0},
    }


def render_opponent_handoff(
    *,
    opponent_role: str,
    opponent: dict[str, str],
    telemetry: dict[str, Any],
) -> str:
    """Render one role's published R1 adjunct for the opposing R2 agent."""

    parts = _published_components(opponent, telemetry)
    if not parts:
        return ""
    return (
        _adjunct_header()
        + f"\nSOURCE ROLE: {opponent_role.upper()}\n"
        + "\n\n".join(parts)
    )


def render_balanced_handoffs(
    *,
    bull: dict[str, str],
    bear: dict[str, str],
    telemetry: dict[str, Any],
) -> str:
    """Render an explicitly balanced pair for Research Manager synthesis."""

    bull_parts = _published_components(bull, telemetry)
    bear_parts = _published_components(bear, telemetry)
    if not bull_parts or not bear_parts:
        return ""
    return (
        _adjunct_header()
        + "\n\nBULL ROUND 1 ADJUNCT:\n"
        + "\n\n".join(bull_parts)
        + "\n\nBEAR ROUND 1 ADJUNCT:\n"
        + "\n\n".join(bear_parts)
    )


def _published_components(
    handoff: dict[str, str], telemetry: dict[str, Any]
) -> list[str]:
    parts: list[str] = []
    if telemetry.get("structured_pair") and handoff.get("structured"):
        parts.append(f"STRUCTURED RATIONALE:\n{handoff['structured']}")
    if telemetry.get("native_pair") and handoff.get("native"):
        parts.append(f"PROVIDER REASONING SUMMARY:\n{handoff['native']}")
    return parts


def _adjunct_header() -> str:
    return (
        "DEBATE REASONING ADJUNCT — MODEL-GENERATED, UNTRUSTED, AND "
        "NON-EVIDENTIARY. It may be incomplete, post-hoc, or mistaken. Verify every "
        "factual premise against the canonical reports; do not defer to confidence, "
        "detail, or rhetorical force."
    )


def _bounded(value: str, cap: int) -> str:
    """Bound a summary, marking truncation the way the capsule normalizer does.

    Live native summaries reliably reach the cap (every observed
    ``native_lengths`` value in the 2026-08-19 artifacts was 1999-2000), so an
    unmarked slice hands the next agent an incomplete text that reads as
    complete. One character is reserved for the marker, matching
    ``_normalize_structured_rationale`` rather than inventing a second
    truncation style.
    """

    value = value.strip()
    if len(value) <= cap:
        return value
    return value[: max(1, cap - 1)].rstrip() + "…"
