"""How each consumer resolves evidence it could not obtain.

The rule under test: missing evidence about MERIT resolves NEUTRAL; missing
evidence about HAZARD or AUTHORITY resolves CLOSED. Scoring merit punitively
invents a failure; screening hazard permissively invents an all-clear.

This file is deliberately cross-layer. A per-consumer unit test cannot catch a
consumer that never *asked* the question — which is exactly how ``EPS_GROWTH``
came to score ``0`` on an unclassifiable earnings baseline while its sibling
consumers all resolved the same condition to ``N/A``. The table below is the
enforcement: adding a consumer that resolves absent evidence means adding a row
and picking a column.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.agents.fundamentals_reconciler import (
    withhold_criterion_award,
    withhold_eps_growth_for_unusable_baseline,
)
from src.earnings_baseline import (
    DISTORTED_EARNINGS_BASELINE_STATUSES,
    EARNINGS_BASELINE_STATUSES,
    GUIDANCE_BRIDGE_STATUSES,
    REQUIRED_GUIDANCE_CONTRACT_ENUMS,
    UNCLASSIFIED_EARNINGS_BASELINE_STATUSES,
    UNUSABLE_EARNINGS_BASELINE_STATUSES,
    eps_growth_award_disposition,
    requires_eps_growth_withholding,
)
from src.evidence_disposition import (
    AbsentEvidence,
    AwardDisposition,
    EvidenceUse,
    resolve_absent_evidence,
    rubric_award_token,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestAbsentEvidencePolicy:
    """The rule itself, stated once."""

    @pytest.mark.parametrize(
        ("use", "expected"),
        [
            (EvidenceUse.MERIT, AbsentEvidence.NEUTRAL),
            (EvidenceUse.HAZARD, AbsentEvidence.CLOSED),
            (EvidenceUse.AUTHORITY, AbsentEvidence.CLOSED),
        ],
    )
    def test_direction_per_use(self, use, expected):
        assert resolve_absent_evidence(use) is expected

    def test_policy_is_total_over_every_use(self):
        """A new use must declare a direction, not inherit a silent default."""
        for use in EvidenceUse:
            assert isinstance(resolve_absent_evidence(use), AbsentEvidence)

    def test_merit_is_the_only_neutral_use(self):
        """Guards against a later edit quietly making a hazard neutral."""
        neutral = {
            u
            for u in EvidenceUse
            if resolve_absent_evidence(u) is AbsentEvidence.NEUTRAL
        }
        assert neutral == {EvidenceUse.MERIT}


class TestRubricAwardToken:
    """``0`` stays in the denominator; ``N/A`` leaves it. One character, one gate."""

    def test_refuted_scores_zero(self):
        assert rubric_award_token(AwardDisposition.REFUTED) == "0"

    def test_unresolved_scores_na(self):
        assert rubric_award_token(AwardDisposition.UNRESOLVED) == "N/A"

    def test_keep_has_no_withheld_token(self):
        with pytest.raises(ValueError, match="does not withhold"):
            rubric_award_token(AwardDisposition.KEEP)


class TestEarningsBaselineDisposition:
    """A diagnosed distortion is evidence; an unclassifiable baseline is not."""

    def test_the_two_sets_are_disjoint_and_cover_the_union(self):
        assert not (
            DISTORTED_EARNINGS_BASELINE_STATUSES
            & UNCLASSIFIED_EARNINGS_BASELINE_STATUSES
        )
        assert (
            DISTORTED_EARNINGS_BASELINE_STATUSES
            | UNCLASSIFIED_EARNINGS_BASELINE_STATUSES
        ) == UNUSABLE_EARNINGS_BASELINE_STATUSES

    @pytest.mark.parametrize("status", sorted(DISTORTED_EARNINGS_BASELINE_STATUSES))
    def test_diagnosed_distortion_refutes(self, status):
        assert (
            eps_growth_award_disposition(
                baseline_status=status, bridge_status="RECONCILED"
            )
            is AwardDisposition.REFUTED
        )

    def test_unclassifiable_baseline_is_unresolved_not_refuted(self):
        """The regression: UNKNOWN used to score identically to TEMPORARILY_BOOSTED."""
        assert (
            eps_growth_award_disposition(
                baseline_status="UNKNOWN", bridge_status="RECONCILED"
            )
            is AwardDisposition.UNRESOLVED
        )

    def test_unresolved_bridge_still_withholds_on_a_durable_baseline(self):
        assert (
            eps_growth_award_disposition(
                baseline_status="DURABLE", bridge_status="UNRESOLVED"
            )
            is AwardDisposition.UNRESOLVED
        )

    def test_durable_and_reconciled_keeps_the_award(self):
        assert (
            eps_growth_award_disposition(
                baseline_status="DURABLE", bridge_status="RECONCILED"
            )
            is AwardDisposition.KEEP
        )

    def test_diagnosis_outranks_an_unresolved_bridge(self):
        """A bridge cannot un-diagnose a distortion the analyst positively found."""
        assert (
            eps_growth_award_disposition(
                baseline_status="MIXED", bridge_status="UNRESOLVED"
            )
            is AwardDisposition.REFUTED
        )

    def test_boolean_wrapper_still_agrees_with_the_disposition(self):
        for base in ("DURABLE", "UNKNOWN", "MIXED"):
            for bridge in ("RECONCILED", "UNRESOLVED"):
                expected = (
                    eps_growth_award_disposition(
                        baseline_status=base, bridge_status=bridge
                    )
                    is not AwardDisposition.KEEP
                )
                assert requires_eps_growth_withholding(base, bridge) is expected


class TestWithheldCriterionIsNeutralNotFailed:
    """End-to-end through the DATA_BLOCK rewrite."""

    @staticmethod
    def _body(baseline: str, bridge: str = "RECONCILED") -> str:
        return "\n".join(
            [
                "EARNINGS_BASELINE_STATUS: " + baseline,
                "GUIDANCE_BRIDGE_STATUS: " + bridge,
                "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; "
                "ROA_ROE_IMPROVING=1; GROSS_MARGIN=0; GLOBAL_EXPANSION=0; "
                "R_AND_D_CAPEX_BACKLOG=0",
                "RAW_GROWTH_SCORE: 3/6",
                "ADJUSTED_GROWTH_SCORE: 50.0% (based on 6 available points)",
            ]
        )

    def test_unclassified_baseline_removes_the_criterion_from_the_denominator(self):
        updated, changed = withhold_eps_growth_for_unusable_baseline(
            self._body("UNKNOWN")
        )
        assert changed
        assert "EPS_GROWTH=N/A" in updated
        # 2 of 5 remaining points, not 2 of 6: the criterion left the denominator.
        assert "40.0% (based on 5 available points)" in updated
        assert "NOT_ESTABLISHED" in updated

    def test_diagnosed_distortion_still_scores_zero_in_the_denominator(self):
        updated, changed = withhold_eps_growth_for_unusable_baseline(
            self._body("TEMPORARILY_BOOSTED")
        )
        assert changed
        assert "EPS_GROWTH=0" in updated
        assert "33.3% (based on 6 available points)" in updated
        assert "WITHHELD" in updated

    def test_unclassified_scores_strictly_higher_than_diagnosed(self):
        """The whole point: absence must not cost what a finding costs."""
        unknown, _ = withhold_eps_growth_for_unusable_baseline(self._body("UNKNOWN"))
        boosted, _ = withhold_eps_growth_for_unusable_baseline(
            self._body("TEMPORARILY_BOOSTED")
        )
        assert "40.0%" in unknown and "33.3%" in boosted

    def test_durable_reconciled_body_is_untouched(self):
        body = self._body("DURABLE")
        updated, changed = withhold_eps_growth_for_unusable_baseline(body)
        assert not changed and updated == body


class TestGuidanceEnumsAreNotInterchangeable:
    """Edge case: five sibling status fields, five *different* vocabularies.

    ``REQUIRED_GUIDANCE_CONTRACT_FIELDS`` all live in one DATA_BLOCK, are all
    upper-snake tokens, and several share members (``UNKNOWN``, ``N/A``,
    ``NOT_APPLICABLE``). Nothing about a bare string says which field owns it,
    so the failure mode is a value that is legal *somewhere* being written or
    read in the wrong slot -- which raises nothing and type-checks fine.
    """

    def test_baseline_and_bridge_vocabularies_are_disjoint(self):
        """Disjointness is what makes the transposition guard below meaningful."""
        assert not (EARNINGS_BASELINE_STATUSES & GUIDANCE_BRIDGE_STATUSES)

    def test_no_two_contract_fields_share_a_whole_vocabulary(self):
        """Identical enums on two fields would make a mix-up undetectable."""
        seen: list[tuple[str, frozenset[str]]] = []
        for field, allowed in REQUIRED_GUIDANCE_CONTRACT_ENUMS.items():
            for other_field, other in seen:
                assert allowed != other, (
                    f"{field} and {other_field} share a vocabulary; a value "
                    "written to the wrong field could never be detected"
                )
            seen.append((field, allowed))

    def test_transposing_baseline_and_bridge_raises_rather_than_misreporting(self):
        """The load-bearing one.

        Before the parameters were keyword-only, passing a bridge status where
        a baseline belongs fell through every branch and returned ``KEEP`` --
        silently restoring the punitive-vs-neutral defect in the *permissive*
        direction. Now it cannot be expressed.
        """
        with pytest.raises(TypeError):
            eps_growth_award_disposition("UNKNOWN", "UNRESOLVED")  # type: ignore[misc]

    def test_a_bridge_token_in_the_baseline_slot_is_not_read_as_durable(self):
        """Belt-and-braces for the keyword call: a wrong-enum value must not
        read as a *pass*. ``UNRESOLVED`` is not a baseline status, so it cannot
        be a diagnosis -- but it must not silently license the award either."""
        result = eps_growth_award_disposition(
            baseline_status="RECONCILED", bridge_status="UNRESOLVED"
        )
        assert result is AwardDisposition.UNRESOLVED

    def test_guidance_bridge_status_is_derived_and_absent_from_every_prompt(self):
        """A derived field must never be requested from a model.

        ``GUIDANCE_BRIDGE_STATUS`` is computed by
        ``normalize_management_guidance_output`` and stamped over whatever the
        model said. If a prompt began advertising it, a model-authored
        ``RECONCILED`` would satisfy the validator and *suppress* the
        conservative backfill -- converting absent evidence into apparent
        positive evidence.
        """
        offenders = [
            path.name
            for path in sorted((REPO_ROOT / "prompts").glob("*.json"))
            if "GUIDANCE_BRIDGE_STATUS: [" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            "GUIDANCE_BRIDGE_STATUS is derived; prompts must not offer its enum: "
            f"{offenders}"
        )

    def test_every_contract_field_enum_is_advertised_exactly_as_code_accepts_it(self):
        """Prompt/code enum parity for the fields models *do* author.

        An off-enum token is not a loud failure: it is uninterpretable, gets
        replaced by the conservative backfill, and quietly withholds credit.
        """
        prompts = {
            path.stem: path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "prompts").glob("*.json")
        }
        for field, allowed in REQUIRED_GUIDANCE_CONTRACT_ENUMS.items():
            for name, text in prompts.items():
                for match in re.finditer(
                    rf"{re.escape(field)}\s*:\s*\[([A-Z_/| ]+)\]", text
                ):
                    # "N/A" is itself a token containing the separator, so mask
                    # it before splitting -- a naive split on "/" shatters it
                    # into "N" and "A" and reports drift that is not there.
                    raw = match.group(1).replace("N/A", "\x00")
                    offered = {
                        token.strip().replace("\x00", "N/A")
                        for token in re.split(r"[/|]", raw)
                        if token.strip()
                    }
                    assert offered <= set(allowed), (
                        f"{name}.{field} offers {sorted(offered - set(allowed))} "
                        f"which code does not accept (enum: {sorted(allowed)})"
                    )

    def test_malformed_breakdown_returns_the_body_unchanged(self):
        body = "EARNINGS_BASELINE_STATUS: UNKNOWN\nGROWTH_SCORE_BREAKDOWN: ROE=1.5"
        updated, changed = withhold_eps_growth_for_unusable_baseline(body)
        assert not changed and updated == body

    def test_already_withheld_award_is_not_rewritten(self):
        awards = {"EPS_GROWTH": "N/A"}
        assert (
            withhold_criterion_award(awards, "EPS_GROWTH", AwardDisposition.UNRESOLVED)
            is False
        )


class TestNoSiteZeroesARubricAwardDirectly:
    """The local guard.

    ``grep 'awards['`` found exactly one rubric-award mutation in ``src/`` and it
    wrote a bare ``"0"`` for both the refuted and the unresolved case. The class
    of defect is not that many sites do this — it is that nothing stopped the
    one that did, or the next one. Scan for the shape rather than the spelling.
    """

    def test_no_bare_zero_assigned_into_an_awards_mapping(self):
        offenders: list[str] = []
        for path in (REPO_ROOT / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not isinstance(node.value, ast.Constant) or node.value.value != "0":
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and "award" in target.value.id.lower()
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        assert not offenders, (
            "rubric awards must be set through withhold_criterion_award(), which "
            "picks 0 vs N/A from an AwardDisposition. A bare '0' scores missing "
            f"evidence as a failure. Offenders: {offenders}"
        )
