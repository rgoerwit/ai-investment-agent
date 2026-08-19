"""Keep consolidated regular expressions consolidated.

Two guards, both AST-based:

1. ``TestNoRedeclaredPatterns`` -- a module must not re-declare a pattern literal that
   already has a canonical home. This is the shape that let the same defect exist in
   four copies of the risk-tally parser while a fix was applied to only one.

2. ``TestSignedFieldsAreParsedWithASign`` -- a pattern naming a field that is observed
   to carry negative values must admit a leading sign. An unsigned class does not clip
   the sign, it fails to match at all, so the value is dropped silently.

Matched on the AST, never on file text: a comment explaining a retired pattern
necessarily contains the pattern, which is the false positive
``tests/financial/test_minor_unit_denomination.py`` documents hitting on its own
docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path("src")

# pattern literal -> where the canonical definition lives
CANONICAL_PATTERNS: dict[str, str] = {
    r'https?://[^\s<>"]+': "src.text_patterns.URL_RE",
    r"(?<=[.!?])\s+|\n+": "src.text_patterns.SENTENCE_SPLIT_RE",
    r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?": "src.text_patterns.is_safe_public_host",
    r"(?is)<result\b[^>]*>.*?</result>": "src.text_patterns.RESULT_ENVELOPE_RE",
    r"(?is)<result\b[^>]*>(.*?)</result>": "src.text_patterns.RESULT_ENVELOPE_BODY_RE",
    r"\b[A-Z0-9]{1,8}(?:[.-][A-Z0-9]{1,6})\b": (
        "src.text_patterns.EXCHANGE_QUALIFIED_TICKER_RE"
    ),
}

# module -> why this file may hold the literal.
CANONICAL_HOME_ALLOWLIST: dict[str, str] = {
    "src/text_patterns.py": "defines them",
    # data_block_utils is a separate stdlib-only leaf and deliberately keeps its own
    # number token; see its module docstring.
    "src/data_block_utils.py": "independent leaf, owns _NUMBER_TOKEN_PATTERN",
}

# DATA_BLOCK / PM_BLOCK fields observed carrying negative values across the persisted
# corpus. A regex naming one of these must allow a sign in its capture group.
SIGNED_CAPABLE_FIELDS: frozenset[str] = frozenset(
    {
        "RISK_TALLY",
        "TOTAL RISK",
        "NET_MARGIN",
        "PE_VS_SECTOR",
        "ROA_PERCENT",
        "ROIC_PERCENT",
        "NET_DEBT_EBITDA",
        "NET_CASH_TO_MARKET_CAP",
        "FREE_CASH_FLOW",
        "OPERATING_CASH_FLOW",
        "REVENUE_CAGR_3Y",
        "FCF_CAGR_3Y",
        "REVENUE_GROWTH_TTM",
        "REVENUE_GROWTH_MRQ",
        "REVENUE_GROWTH_FY",
        "EARNINGS_GROWTH_TTM",
        "EARNINGS_GROWTH_MRQ",
        "EARNINGS_GROWTH_FY",
        "ROA_5Y_AVG",
        "ROE_5Y_AVG",
        "BEAR_GROWTH_PCT",
        "BASE_GROWTH_PCT",
        "BEAR_MARGIN_DELTA_BPS",
        "BASE_MARGIN_DELTA_BPS",
    }
)

# A capture group is sign-tolerant if it opens with any of these.
_SIGN_TOLERANT_OPENERS = ("-?", "[-", "[+", "(-", "-+", "\\-", "?:-")

# A capture group is numeric if it opens with any of these (after any sign).
_NUMERIC_OPENERS = ("\\d", "[\\d", "[0-9", "?:\\d", "?:[\\d", "?:[0-9")


def _is_sign_tolerant(group: str) -> bool:
    return group.startswith(_SIGN_TOLERANT_OPENERS)


def _numeric_captures_for_signed_fields(value: str) -> list[tuple[str, str]]:
    """Yield (field, capture-group-body) for numeric captures of signed-capable fields.

    The field label is matched WITHOUT requiring a literal ``:`` immediately after it.
    That mattered: ``thesis_visualizer``'s tally pattern reads
    ``TOTAL RISK (?:COUNT|SCORE)?\\*?\\*?[:\\s]*...`` -- the colon is several tokens
    downstream -- so a ``"{field}:"`` marker silently skipped the exact site this guard
    was written for, and the guard passed against a deliberately regressed copy.

    Only the first CAPTURING group after the label is inspected, and only when it opens
    like a number: an enum alternation such as ``(STRONG|WEAK)`` needs no sign. Skipping
    non-capturing groups is load-bearing -- in the tally pattern above the first ``(`` is
    ``(?:COUNT|SCORE)``, so taking it would again miss the real capture downstream.
    """
    found: list[tuple[str, str]] = []
    for field in SIGNED_CAPABLE_FIELDS:
        index = value.find(field)
        if index == -1:
            continue
        tail = value[index + len(field) :]
        # The capture must belong to this label, not to a later one.
        next_field_at = min(
            (pos for other in SIGNED_CAPABLE_FIELDS if (pos := tail.find(other)) != -1),
            default=len(tail),
        )
        group = _first_capturing_group(tail[:next_field_at])
        if group is None:
            continue
        if group.startswith(_NUMERIC_OPENERS) or _is_sign_tolerant(group):
            found.append((field, group))
    return found


def _first_capturing_group(segment: str) -> str | None:
    """Return the body after the first unescaped capturing ``(``, or None."""
    position = 0
    while (position := segment.find("(", position)) != -1:
        escaped = position > 0 and segment[position - 1] == "\\"
        non_capturing = segment[position + 1 : position + 2] == "?"
        if not escaped and not non_capturing:
            return segment[position + 1 :]
        position += 1
    return None


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, value) for every string constant that is not a docstring."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


class TestNoRedeclaredPatterns:
    def test_no_module_redeclares_a_consolidated_pattern(self) -> None:
        offenders: list[str] = []
        for path in _source_files():
            if str(path) in CANONICAL_HOME_ALLOWLIST:
                continue
            for lineno, value in _string_literals(path):
                if value in CANONICAL_PATTERNS:
                    offenders.append(
                        f"{path}:{lineno} re-declares a consolidated pattern "
                        f"-- import {CANONICAL_PATTERNS[value]} instead"
                    )
        assert not offenders, "\n".join(offenders)

    def test_every_canonical_pattern_is_actually_defined(self) -> None:
        """Guard the guard: a stale entry here would make the scan silently vacuous."""
        home = Path("src/text_patterns.py")
        literals = {value for _lineno, value in _string_literals(home)}
        missing = {
            pattern
            for pattern, target in CANONICAL_PATTERNS.items()
            if target.startswith("src.text_patterns") and pattern not in literals
        }
        assert not missing, (
            f"CANONICAL_PATTERNS lists patterns absent from {home}: {missing}. "
            "Either the pattern moved or this table is stale."
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        """An exemption that no longer exempts anything must be removed."""
        for rel, reason in CANONICAL_HOME_ALLOWLIST.items():
            path = Path(rel)
            assert path.exists(), f"allowlisted file is gone: {rel} ({reason})"

    def test_guard_detects_a_planted_duplicate(self, tmp_path: Path) -> None:
        """The scan must fail on a real re-declaration, not merely pass on clean code."""
        planted = tmp_path / "planted.py"
        planted.write_text("import re\n_URL_RE = re.compile(r'https?://[^\\s<>\"]+')\n")
        found = [
            value
            for _lineno, value in _string_literals(planted)
            if value in CANONICAL_PATTERNS
        ]
        assert found, "the AST scan failed to see a planted duplicate"


class TestSignedFieldsAreParsedWithASign:
    @pytest.mark.parametrize("path", _source_files(), ids=str)
    def test_patterns_for_negative_capable_fields_admit_a_sign(
        self, path: Path
    ) -> None:
        offenders = [
            f"{path}:{lineno} parses {field} with an unsigned capture ({group[:26]!r}). "
            "This field is observed negative in the persisted corpus; an unsigned "
            "class drops the match entirely rather than clipping the sign."
            for lineno, value in _string_literals(path)
            for field, group in _numeric_captures_for_signed_fields(value)
            if not _is_sign_tolerant(group)
        ]
        assert not offenders, "\n".join(offenders)

    @pytest.mark.parametrize(
        ("planted", "expected"),
        [
            # The plain shape.
            (r"NET_MARGIN:\s*(\d+(?:\.\d+)?)%", "NET_MARGIN"),
            # The shape the first version of this guard MISSED: the label is separated
            # from its capture by an optional group and a character class, so there is
            # no literal "TOTAL RISK:" anywhere in the pattern. This is the real
            # thesis_visualizer defect (54 of 1,320 decisions dropped their tally).
            (
                r"\*?\*?TOTAL RISK (?:COUNT|SCORE)?\*?\*?[:\s]*\*?\*?(\d+(?:\.\d+)?)\*?\*?",
                "TOTAL RISK",
            ),
        ],
        ids=["plain_field_colon", "label_separated_from_capture"],
    )
    def test_guard_detects_a_planted_unsigned_parser(
        self, planted: str, expected: str
    ) -> None:
        flagged = [
            field
            for field, group in _numeric_captures_for_signed_fields(planted)
            if not _is_sign_tolerant(group)
        ]
        assert expected in flagged, (
            f"the sign guard failed to flag a planted unsigned parser for {expected}"
        )

    @pytest.mark.parametrize(
        "clean",
        [
            r"NET_MARGIN:\s*(-?\d+(?:\.\d+)?)%",
            r"\*?\*?TOTAL RISK (?:COUNT|SCORE)?\*?\*?[:\s]*\*?\*?(-?\d+(?:\.\d+)?)\*?\*?",
            # An enum capture is not a number and needs no sign.
            r"ROIC_PERCENT:\s*(STRONG|WEAK)",
        ],
        ids=["signed_plain", "signed_separated", "enum_capture"],
    )
    def test_guard_does_not_flag_correct_patterns(self, clean: str) -> None:
        flagged = [
            field
            for field, group in _numeric_captures_for_signed_fields(clean)
            if not _is_sign_tolerant(group)
        ]
        assert not flagged, f"false positive on a correct pattern: {flagged}"
