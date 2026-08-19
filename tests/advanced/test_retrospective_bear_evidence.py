"""Evidence is resolved from the artifact at read time, not frozen at write time.

Three rounds of prompt rules, a decline token and a grounding gate were all
compensating for a lazy quantifier five steps upstream:

    r"(?:KEY RISKS|FAILURE MODE|KILL CRITERIA|BEAR CASE).*?(?=\\n\\n|\\Z)"

``.*?`` stops at the first blank line. Real bear output uses both layouts —
heading-then-body and heading-blank-line-then-body — so on the second the regex
captured the heading alone: ``'BEAR CASE SUMMARY**:'``. Measured 2026-08-17 over
7,952 snapshots, **2,449 (31%) stored a bare label as their bear evidence**, and
because that string is non-empty every consumer downstream read it as evidence.
Counts here are point-in-time; run ``scripts/retrospective_evidence_audit.py``
for the live figure rather than trusting this prose.

The architectural repair is not a better regex. It is that the excerpt is a
*cache* of ``investment_analysis.investment_debate.bear_history`` — a ~4,500
character field still present in the same JSON — so a failed extraction is
recoverable at read time instead of frozen into the corpus. ``json.load`` has
already parsed the artifact, so the recovery reads one more key from a dict
that is already in memory.
"""

from __future__ import annotations

import json

import pytest

from src.retrospective import (
    _BEAR_PREFERRED_SECTIONS,
    _BEAR_SECTION_HEADINGS,
    BEAR_EVIDENCE_FROM_SNAPSHOT,
    BEAR_EVIDENCE_MAX_CHARS,
    BEAR_EVIDENCE_MISSING,
    BEAR_EVIDENCE_RECONSTRUCTED,
    _resolve_bear_evidence,
    extract_bear_evidence,
    has_grounding_context,
    is_usable_bear_evidence,
    load_past_snapshots,
)

# Copied from a real artifact (QCI.DE 2026-07-07) — the layout that broke. An
# invented fixture here would have tested the fixture.
BROKEN_LAYOUT = (
    "Bear Analyst (Round 1): **BEAR CASE SUMMARY**:\n"
    "\n"
    "This investment fails to meet the core thesis criteria for an "
    '"undiscovered" value-to-growth opportunity. First, the stock is '
    "over-covered, with 20 analysts tracking the name.\n"
)
WORKING_LAYOUT = (
    "Bear Analyst (Round 1): **BEAR CASE SUMMARY**:\n"
    "Philogen presents a high-risk profile that fails the stability test.\n"
)


class TestTheExtractorNeverReturnsABareHeading:
    def test_a_blank_line_after_the_heading_no_longer_truncates(self):
        """The exact defect: 2,449 snapshots captured only the heading."""
        assert extract_bear_evidence(BROKEN_LAYOUT).startswith("This investment fails")

    def test_the_working_layout_is_unchanged(self):
        assert extract_bear_evidence(WORKING_LAYOUT).startswith("Philogen presents")

    def test_key_risks_is_preferred_over_the_prose_summary(self):
        """1,247 occurrences vs the summary's prose — the enumerated form wins."""
        raw = (
            "**BEAR CASE SUMMARY**:\n\nGeneric framing prose.\n\n"
            "**KEY RISKS**:\n\n1. Cyclical peak.\n2. Governance drag.\n"
        )
        assert extract_bear_evidence(raw) == "1. Cyclical peak.\n2. Governance drag."

    def test_a_body_ends_at_the_next_heading_not_at_a_blank_line(self):
        """Blank lines are internal to a section; treating one as a terminator
        is the defect being replaced."""
        raw = (
            "**KEY RISKS**:\n\n1. Debt load.\n\n2. Customer concentration.\n\n"
            "**FAILURE MODE SCORING**\n| Mode | Score |\n"
        )
        evidence = extract_bear_evidence(raw)
        assert "Customer concentration" in evidence
        assert "Mode | Score" not in evidence

    def test_a_transcript_of_only_headings_yields_nothing(self):
        assert (
            extract_bear_evidence("Bear Analyst (Round 1): **BEAR CASE SUMMARY**:")
            == ""
        )

    def test_prose_with_no_headings_is_still_returned(self):
        raw = "The bear worries about margins and the parent company."
        assert extract_bear_evidence(raw) == raw

    def test_the_fallback_still_cannot_return_a_heading(self):
        """No recognized section carried a body, so heading lines are dropped."""
        raw = "**KILL CRITERIA**\n\n**DOWNSIDE PROBABILITY**\n\nP(loss) = 40%"
        assert extract_bear_evidence(raw) == "P(loss) = 40%"

    @pytest.mark.parametrize("raw", ["", "   \n\n  ", None])
    def test_empty_input_is_empty_output(self, raw):
        assert extract_bear_evidence(raw) == ""

    def test_the_cap_is_honored(self):
        raw = "**KEY RISKS**:\n\n" + ("x" * 2000)
        assert len(extract_bear_evidence(raw)) == BEAR_EVIDENCE_MAX_CHARS


class TestSentinelRecognitionNotALengthThreshold:
    """A terse excerpt is evidence; a 20-character heading is not.

    A length threshold would have been arbitrary and would have discarded
    genuinely short bear cases. This keys on *content* — the same shape as
    `guidance_contract_value_is_uninterpretable`: present, uninterpretable, and
    therefore to be read as absent.
    """

    @pytest.mark.parametrize(
        "stored",
        [
            "BEAR CASE SUMMARY**:",  # 1,596 occurrences
            "BEAR CASE SUMMARY**",  # 688
            "BEAR CASE SUMMARY:**",  # 142
            "BEAR CASE SUMMARY",  # 23
            "**KEY RISKS**",
            "Bear Analyst (Round 1): **BEAR CASE SUMMARY**:",
            "",
            None,
        ],
    )
    def test_headings_and_blanks_are_unusable(self, stored):
        assert not is_usable_bear_evidence(stored)

    @pytest.mark.parametrize("stored", ["Debt.", "1. Cyclical peak.", "Margins."])
    def test_short_but_real_text_is_usable(self, stored):
        assert is_usable_bear_evidence(stored), (
            "a length threshold would have discarded this; the rule is content"
        )

    def test_a_heading_followed_by_a_body_is_usable(self):
        assert is_usable_bear_evidence("**KEY RISKS**\n1. Debt.")

    @pytest.mark.parametrize("value", [42, 3.5, True, ["a"], {"k": "v"}, object()])
    def test_a_non_string_is_unusable_rather_than_a_crash(self, value):
        """JSON can hold anything, and a corrupt artifact did.

        `(text or "").strip()` raised AttributeError on every one of these,
        which would have cost the snapshot — or the run — over a field that is
        merely malformed. A number is not bear evidence; it is not an error.
        """
        assert not is_usable_bear_evidence(value)


class TestResolutionPrefersTheCacheAndRecordsProvenance:
    def _artifact(self, excerpt, raw_history):
        return {
            "prediction_snapshot": {"ticker": "T.T", "bear_risks_excerpt": excerpt},
            "investment_analysis": {"investment_debate": {"bear_history": raw_history}},
        }

    def test_a_usable_excerpt_is_never_replaced(self):
        """The cache is authoritative when it is intact — no needless rework."""
        artifact = self._artifact("1. Real recorded risk.", BROKEN_LAYOUT)
        snapshot = artifact["prediction_snapshot"]
        assert _resolve_bear_evidence(snapshot, artifact) == "1. Real recorded risk."
        assert snapshot["bear_evidence_provenance"] == BEAR_EVIDENCE_FROM_SNAPSHOT

    def test_a_heading_only_excerpt_is_reconstructed(self):
        artifact = self._artifact("BEAR CASE SUMMARY**:", BROKEN_LAYOUT)
        snapshot = artifact["prediction_snapshot"]
        resolved = _resolve_bear_evidence(snapshot, artifact)
        assert resolved.startswith("This investment fails")
        assert snapshot["bear_evidence_provenance"] == BEAR_EVIDENCE_RECONSTRUCTED
        assert snapshot["bear_risks_excerpt"] == resolved

    def test_no_recoverable_source_blanks_the_heading(self):
        """Left in place, a bare label would still read as grounding."""
        artifact = self._artifact("BEAR CASE SUMMARY**:", "")
        snapshot = artifact["prediction_snapshot"]
        assert _resolve_bear_evidence(snapshot, artifact) == ""
        assert snapshot["bear_risks_excerpt"] == ""
        assert snapshot["bear_evidence_provenance"] == BEAR_EVIDENCE_MISSING

    @pytest.mark.parametrize(
        "artifact",
        [
            {},
            {"investment_analysis": None},
            {"investment_analysis": {"investment_debate": "corrupted"}},
            {"investment_analysis": {"investment_debate": {"bear_history": 42}}},
        ],
    )
    def test_a_malformed_artifact_does_not_raise(self, artifact):
        snapshot: dict = {"bear_risks_excerpt": "BEAR CASE SUMMARY**:"}
        assert _resolve_bear_evidence(snapshot, artifact) == ""
        assert snapshot["bear_evidence_provenance"] == BEAR_EVIDENCE_MISSING

    def test_the_grounding_predicate_rejects_a_bare_heading_on_its_own(self):
        """Defense in depth, and the first version of this test got it backwards.

        It asserted that an unresolved heading *is* grounded — documenting the
        defect as if it were the contract, which would have locked it in. The
        loader normally clears such a value, but a snapshot built without going
        through `load_past_snapshots` must not be able to reintroduce it.
        """
        assert not has_grounding_context({"bear_risks_excerpt": "BEAR CASE SUMMARY**:"})
        assert has_grounding_context({"bear_risks_excerpt": "1. Real recorded risk."})

    def test_reconstruction_leaves_an_unrecoverable_snapshot_ungrounded(self):
        artifact = self._artifact("BEAR CASE SUMMARY**:", "")
        snapshot = artifact["prediction_snapshot"]
        _resolve_bear_evidence(snapshot, artifact)
        assert not has_grounding_context(snapshot)


class TestTheLoaderRepairsWithoutRewritingHistory:
    def _write(self, tmp_path, name, excerpt, raw_history):
        (tmp_path / name).write_text(
            json.dumps(
                {
                    "prediction_snapshot": {
                        "ticker": "T.T",
                        "analysis_date": "2026-01-01",
                        "bear_risks_excerpt": excerpt,
                    },
                    "investment_analysis": {
                        "investment_debate": {"bear_history": raw_history}
                    },
                }
            )
        )
        return tmp_path / name

    def test_a_broken_snapshot_is_repaired_on_load(self, tmp_path):
        self._write(
            tmp_path,
            "T.T_20260101_000000_analysis.json",
            "BEAR CASE SUMMARY**:",
            BROKEN_LAYOUT,
        )
        loaded = load_past_snapshots(None, tmp_path)
        snapshot = loaded["T.T"][0]
        assert snapshot["bear_risks_excerpt"].startswith("This investment fails")
        assert snapshot["bear_evidence_provenance"] == BEAR_EVIDENCE_RECONSTRUCTED

    def test_the_file_on_disk_is_byte_for_byte_unchanged(self, tmp_path):
        """Read-only recovery. The corpus is never rewritten."""
        path = self._write(
            tmp_path,
            "T.T_20260101_000000_analysis.json",
            "BEAR CASE SUMMARY**:",
            BROKEN_LAYOUT,
        )
        before = path.read_bytes()
        load_past_snapshots(None, tmp_path)
        assert path.read_bytes() == before

    def test_an_unrecoverable_snapshot_is_left_ungrounded(self, tmp_path):
        self._write(
            tmp_path,
            "T.T_20260101_000000_analysis.json",
            "BEAR CASE SUMMARY**:",
            "",
        )
        snapshot = load_past_snapshots(None, tmp_path)["T.T"][0]
        assert snapshot["bear_evidence_provenance"] == BEAR_EVIDENCE_MISSING
        assert not has_grounding_context(snapshot)


class TestTheHeadingVocabularyCannotDriftFromThePrompt:
    """The extractor keys on section names the bear prompt mandates.

    This is the drift risk in a section-aware parser: the prompt renames a
    heading, the extractor stops finding it, and evidence silently degrades to
    the fallback — or to nothing. The failure is quiet, which is why it needs a
    guard rather than vigilance. Same shape as
    `test_threshold_parity.py::test_trader_action_enum_matches_parser`.

    Deliberately one-directional: the prompt may describe sections this parser
    does not treat as a source (FAILURE MODE SCORING is a table, KILL CRITERIA is
    extracted separately), but every name this parser matches must still be a
    heading the prompt actually asks for.
    """

    def test_every_matched_heading_is_declared_by_the_bear_prompt(self):
        # `prompt_text` reads the on-disk JSON, which is canonical. `get_prompt`
        # would resolve a Langfuse override and make the guard environment-
        # dependent — the rule CLAUDE.md states for the L1 harness.
        from src.eval.prompt_contracts import prompt_text

        prompt = prompt_text("bear_researcher").upper()
        missing = [h for h in _BEAR_SECTION_HEADINGS if h not in prompt]
        assert missing == [], (
            f"the extractor matches headings the bear prompt no longer emits: "
            f"{missing} — evidence would silently degrade to the fallback"
        )

    def test_the_preferred_sources_are_a_subset_of_what_is_matched(self):
        assert set(_BEAR_PREFERRED_SECTIONS) <= set(_BEAR_SECTION_HEADINGS)

    def test_a_heading_with_an_inline_body_is_not_swallowed(self):
        """`KEY RISKS: Debt is high` is a body line, not a heading.

        The tail character class admits emphasis and colons in either order, so
        it must still exclude letters or a one-line section would vanish.
        """
        assert is_usable_bear_evidence("**KEY RISKS**: Debt is high and rising.")
        assert extract_bear_evidence("**KEY RISKS**: Debt is high.") == (
            "**KEY RISKS**: Debt is high."
        )


class TestASummaryStopsAtEverySubsequentSection:
    """A heading missing from the vocabulary does not merely go unfound.

    It stops *terminating* the section above it, so the preceding body absorbs
    everything that follows. The first cut omitted three mandatory bear headings
    and a BEAR CASE SUMMARY silently swallowed the counterargument — evidence
    contaminated with rebuttal prose, contradicting the section-aware contract.
    """

    @pytest.mark.parametrize(
        "boundary",
        [
            "COUNTER TO BULL ARGUMENTS",
            "CONVICTION",
            "RECOMMENDATION",
            "FAILURE MODE SCORING",
            "KILL CRITERIA",
            "DOWNSIDE PROBABILITY",
        ],
    )
    def test_the_summary_ends_at_the_next_section(self, boundary):
        raw = (
            "**BEAR CASE SUMMARY**:\n\nGenuine downside on margins.\n\n"
            f"**{boundary}**:\n\nText that is not bear evidence.\n"
        )
        evidence = extract_bear_evidence(raw)
        assert evidence == "Genuine downside on margins."
        assert "not bear evidence" not in evidence

    def test_every_prompt_heading_terminates_a_section(self):
        """The converse of the drift guard.

        The other direction checks that everything matched is in the prompt.
        This checks that everything the prompt *mandates* is matched — which is
        the direction that causes contamination when it fails.
        """
        from src.eval.prompt_contracts import prompt_text

        prompt = prompt_text("bear_researcher").upper()
        mandated = [
            "BEAR CASE SUMMARY",
            "KEY RISKS",
            "COUNTER TO BULL ARGUMENTS",
            "FAILURE MODE SCORING",
            "KILL CRITERIA",
            "DOWNSIDE PROBABILITY",
            "CONVICTION",
            "RECOMMENDATION",
        ]
        for heading in mandated:
            assert heading in prompt, f"fixture stale: {heading} left the prompt"
            assert heading in _BEAR_SECTION_HEADINGS, (
                f"{heading} is mandated by the bear prompt but does not bound a "
                f"section here, so the body above it will absorb that section"
            )
