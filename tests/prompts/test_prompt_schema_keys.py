"""Every key a prompt advertises in its JSON output schema must be read by code.

The failure this exists for: ``prompts/legal_counsel.json`` advertised
``other_legal_risks`` while every consumer read ``other_regulatory_risks``
(``supplemental_extractors.extract_legal_risks``, ``supplemental_flags``,
``consultant_nodes``' fallback stub). A *schema-compliant* Legal Counsel
response therefore dropped every SDN / HFCAA / Entity-List / capital-controls
finding, and no ``REGULATORY_*`` flag fired in 4,621 saved analyses.

Nothing catches that class of drift: the prompt is valid, the parser is valid,
the L0 enum-parity test only compares status vocabularies, and the flag simply
never appears. This is the JSON-schema sibling of
``tests/graph/test_consultant_gate.py::TestGateFlagTokensAreLive``, which bans
dead red-flag *type* tokens in the routing gate.

Scope is self-limiting: only prompts embedding a parseable JSON object of three
or more keys participate, so this stays quiet for prose-only prompts.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
SRC_DIR = REPO_ROOT / "src"

# Keys deliberately requested for the human-readable persisted artifact rather
# than for a parser. They are recorded here so the guard stays honest: an
# unread key must be justified, not silently tolerated.
INFORMATIONAL_ONLY_KEYS = frozenset(
    {
        # Provenance prose for the PFIC status, retained in the raw legal_report
        # saved under source_artifacts; the status itself is what code consumes.
        "pfic_source",
        # One-sentence operator summary; deliberately not parsed.
        "assessment_notes",
        # Supplied *to* the agent by src/tools/legal.py (WITHHOLDING_TAX_RATES
        # lookup) and echoed back in its JSON. The code already has the value, so
        # there is nothing to parse out of the response.
        "withholding_rate",
    }
)


def _balanced_json_objects(text: str) -> list[dict]:
    """Return every balanced ``{...}`` span in *text* that parses as a JSON dict."""
    found: list[dict] = []
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except ValueError:
                        pass
                    else:
                        if isinstance(parsed, dict) and len(parsed) >= 3:
                            found.append(parsed)
                    break
    return found


def _src_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(SRC_DIR.rglob("*.py"))
    )


def _read_access_patterns(key: str) -> tuple[re.Pattern[str], ...]:
    """Regexes for *reading* ``key`` off a payload, not merely naming it.

    A bare ``'"key" in source'`` test is too weak: an unrelated module that
    happens to contain the same string would vouch for a misspelled prompt key.
    Requiring a mapping read (``.get("key")``, ``["key"]``, ``pop("key")``) ties
    the assurance to code that actually consumes the field.
    """
    quoted = f"[\"']{re.escape(key)}[\"']"
    # Reads only. A `"key":` dict-literal pattern was tried and removed: that is a
    # *write*, and it let a producer vouch for a field no parser consumes —
    # `withholding_rate` passed solely because `src/tools/legal.py` puts it into
    # the tool payload the agent then echoes back unparsed.
    return (
        re.compile(rf"\.get\(\s*{quoted}"),
        re.compile(rf"\.pop\(\s*{quoted}"),
        re.compile(rf"\[\s*{quoted}\s*\]"),
    )


def _is_read_by_src(key: str, source: str) -> bool:
    return any(pattern.search(source) for pattern in _read_access_patterns(key))


def _advertised_schemas() -> list[tuple[str, dict]]:
    schemas: list[tuple[str, dict]] = []
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        message = json.loads(path.read_text(encoding="utf-8")).get("system_message", "")
        for schema in _balanced_json_objects(message):
            schemas.append((path.name, schema))
    return schemas


SCHEMAS = _advertised_schemas()


def test_at_least_one_schema_is_discovered() -> None:
    """Guard the guard: a parser regression must not silently disable it."""
    assert SCHEMAS, "no prompt JSON output schema found — the extractor is broken"


@pytest.mark.parametrize(
    ("prompt_name", "schema"),
    SCHEMAS,
    ids=[f"{name}-{len(schema)}keys" for name, schema in SCHEMAS],
)
def test_every_advertised_key_is_read_by_src(prompt_name: str, schema: dict) -> None:
    source = _src_text()
    unread = [
        key
        for key in schema
        if key not in INFORMATIONAL_ONLY_KEYS and not _is_read_by_src(key, source)
    ]
    assert not unread, (
        f"{prompt_name} advertises output keys no code in src/ reads: {unread}. "
        "Either the prompt or the parser has the wrong name — a compliant "
        "response is silently discarding these fields."
    )


def test_legal_counsel_regulatory_key_matches_its_consumers() -> None:
    """Pin the specific drift that motivated this file."""
    message = json.loads(
        (PROMPTS_DIR / "legal_counsel.json").read_text(encoding="utf-8")
    )["system_message"]
    assert "other_regulatory_risks" in message
    assert "other_legal_risks" not in message


def test_guard_requires_a_read_not_a_mention() -> None:
    """A key merely *named* somewhere in src/ must not count as consumed."""
    source = "SOME_CONSTANT = 'other_legal_risks'  # a mention, not a read\n"
    assert _is_read_by_src("other_legal_risks", source) is False
    assert _is_read_by_src("other_legal_risks", 'd.get("other_legal_risks")') is True
    assert _is_read_by_src("other_legal_risks", 'd["other_legal_risks"]') is True


def test_guard_does_not_accept_a_write_as_a_read() -> None:
    """A producer emitting the field must not vouch for a consumer reading it.

    Live instance: `withholding_rate` is written into the tool payload by
    `src/tools/legal.py` and echoed back by the agent, but nothing parses it out
    of the response — so a dict-literal pattern silently exempted it.
    """
    payload_write = 'return {"withholding_rate": rate, "country": country}'
    assert _is_read_by_src("withholding_rate", payload_write) is False
