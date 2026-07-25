"""Tests for the entity governance card builder + reconciler."""

import pytest

from src.validators.entity_governance_card import (
    EntityGovernanceCard,
    build_card,
    card_from_dict,
    card_to_prompt_block,
    card_to_prompt_block_from_dict,
    deterministic_hints,
    extract_merged_subset_from_raw,
    requires_structure_disclosure,
)

# ---------------------------------------------------------------------------
# Deterministic hints
# ---------------------------------------------------------------------------


class TestDeterministicHints:
    def test_english_holdings_token_fires(self):
        hints, role = deterministic_hints(
            {"longName": "Youngone Holdings Co., Ltd.", "shortName": "YoungoneHoldings"}
        )
        assert any(h.startswith("name_token:") for h in hints)
        assert role == "HOLDCO"

    def test_latin_holdings_token_is_case_insensitive(self):
        hints, role = deterministic_hints(
            {"longName": "YOUNGONE HOLDINGS CO., LTD.", "shortName": "youngone"}
        )
        assert any(h.startswith("name_token:") for h in hints)
        assert role == "HOLDCO"

    def test_holdings_token_label_is_stable_when_holding_also_matches(self):
        hints, role = deterministic_hints(
            {"longName": "Example Holdings Company", "shortName": ""}
        )
        assert "name_token:Holdings" in hints
        assert role == "HOLDCO"

    def test_summary_phrase_fires_without_english_name_token(self):
        # Non-English-marker case: longName lacks "Holdings" but summary
        # describes holding-company structure. This is the case the critique
        # explicitly flagged — token lists alone are incomplete.
        hints, role = deterministic_hints(
            {
                "longName": "ソフトバンクグループ株式会社",
                "longBusinessSummary": (
                    "SoftBank Group Corp. operates as a holding company through "
                    "its subsidiaries worldwide."
                ),
            }
        )
        assert any(h.startswith("summary:") for h in hints)
        assert role == "HOLDCO"

    def test_low_employees_is_weak_only(self):
        """Weak signal alone must NOT promote to HOLDCO — defends against
        asset-light / royalty false-positives flagged in the critique."""
        hints, role = deterministic_hints(
            {
                "longName": "Boring Industrial Co",
                "longBusinessSummary": "Makes widgets.",
                "fullTimeEmployees": 50,
                "totalRevenue": 200_000_000_000,
            }
        )
        assert "weak:low_employees_vs_revenue" in hints
        assert role == "UNKNOWN"

    def test_no_hints_for_standalone(self):
        hints, role = deterministic_hints(
            {
                "longName": "Toyota Motor Corporation",
                "longBusinessSummary": (
                    "Toyota Motor Corporation designs, manufactures, assembles, "
                    "and sells passenger vehicles, minivans, commercial vehicles, "
                    "and related parts and accessories worldwide."
                ),
            }
        )
        assert hints == []
        assert role == "UNKNOWN"


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    def _build(self, merged, senior_metrics, fla_report):
        return build_card(
            ticker="TEST.X",
            company_name=merged.get("longName", "TEST"),
            merged_data=merged,
            senior_metrics=senior_metrics,
            fla_report=fla_report,
        )

    def test_youngone_role_clean_without_promoting_minor_holder_to_control(self):
        """Role agreement stays clean; a 29.09% holder is not automatically control."""
        card = self._build(
            merged={
                "longName": "Youngone Holdings Co., Ltd.",
                "longBusinessSummary": "changed its name to Youngone Holdings",
            },
            senior_metrics={
                "listing_role": "INTERMEDIATE_HOLDCO",
                "related_listed_tickers": "111770.KS:operating_subsidiary:50.5",
                "metric_scope_payout": "SEPARATE",
                "metric_scope_ocf": "CONSOLIDATED",
                "parent_company": "YMSA (29.09%)",
            },
            fla_report=(
                "Controlling Shareholder: YMSA (29.09%)\n"
                "ENTITY_ROLE_OBSERVED: INTERMEDIATE_HOLDCO\n"
                "Related Listed Tickers: 111770.KS:operating_subsidiary:50.5"
            ),
        )
        assert card.entity_role == "INTERMEDIATE_HOLDCO"
        assert card.confidence == "clean"
        assert any(e["ticker"] == "111770.KS" for e in card.related_listed)
        assert card.metric_scope == {"payout": "SEPARATE", "ocf": "CONSOLIDATED"}
        assert card.controlling_shareholder is None
        assert card.control_status == "UNKNOWN"
        assert requires_structure_disclosure(card) is True

    def test_standalone_toyota(self):
        """No source asserts non-standard role → unresolved, no disclosure."""
        card = self._build(
            merged={
                "longName": "Toyota Motor Corporation",
                "longBusinessSummary": "Toyota Motor Corporation designs vehicles.",
            },
            senior_metrics={},
            fla_report="",
        )
        assert card.entity_role == "UNKNOWN"
        assert card.confidence == "unresolved"
        assert card.related_listed == []
        assert requires_structure_disclosure(card) is False

    def test_conflict_routes_to_conflict_confidence(self):
        """Hints fire HOLDCO but Senior says STANDALONE → conflict."""
        card = self._build(
            merged={
                "longName": "Foo Holdings Inc",
                "longBusinessSummary": "operates as a holding company through its subsidiaries",
            },
            senior_metrics={"listing_role": "STANDALONE"},
            fla_report="",
        )
        assert card.confidence == "conflict"
        assert "disagreement" in card.notes
        # Conflict alone is sufficient to require disclosure even when
        # entity_role falls back to UNKNOWN.
        assert requires_structure_disclosure(card) is True

    def test_majority_resolves_senior_dissent_when_hints_corroborate_fla(self):
        """Senior=STANDALONE but FLA=PURE_HOLDCO and hints fire HOLDCO →
        2-of-3 majority resolves to PURE_HOLDCO with confidence=unresolved
        rather than escalating to conflict. Guards against a single Senior
        miscall on a name where deterministic signals and FLA agree it is a
        holdco."""
        card = self._build(
            merged={
                "longName": "Foo Holdings Inc",
                "longBusinessSummary": "operates as a holding company through its subsidiaries",
            },
            senior_metrics={"listing_role": "STANDALONE"},
            fla_report="ENTITY_ROLE_OBSERVED: PURE_HOLDCO",
        )
        assert card.entity_role == "PURE_HOLDCO"
        assert card.confidence == "unresolved"
        assert "senior rejected" in card.notes

    def test_non_english_marker_with_senior_confirmation(self):
        """Japanese 持株会社 name; English token miss but Senior says PURE_HOLDCO."""
        card = self._build(
            merged={
                "longName": "ソフトバンクグループ株式会社",
                "longBusinessSummary": "operates as a holding company through its subsidiaries",
            },
            senior_metrics={"listing_role": "PURE_HOLDCO"},
            fla_report="",
        )
        # Two sources agree on HOLDCO_FAMILY (summary hint + Senior).
        assert card.entity_role == "PURE_HOLDCO"
        assert card.confidence == "clean"

    def test_fla_silence_does_not_invent_controller(self):
        card = self._build(
            merged={"longName": "Youngone Holdings Co., Ltd."},
            senior_metrics={"listing_role": "PURE_HOLDCO", "parent_company": None},
            fla_report="ENTITY_ROLE_OBSERVED: PURE_HOLDCO",
        )
        assert card.controlling_shareholder is None
        assert card.control_status == "UNKNOWN"

    def test_youngone_unverified_holder_is_not_promoted(self):
        card = self._build(
            merged={
                "longName": "Youngone Holdings Co., Ltd.",
                "longBusinessSummary": "operates as a holding company through its subsidiaries",
            },
            senior_metrics={
                "listing_role": "PURE_HOLDCO",
                "related_listed_tickers": "111770.KS:SUBSIDIARY:100%",
                "metric_scope_payout": "CONSOLIDATED",
                "parent_company": None,
            },
            fla_report=(
                "ENTITY_ROLE_OBSERVED: PURE_HOLDCO\n"
                "Controlling Shareholder: YMSA (29.09%)\n"
                "Related Listed Tickers: 111770.KS:SUBSIDIARY:100"
            ),
        )
        assert card.controlling_shareholder is None
        assert card.largest_shareholder is None

    def test_verified_controller_text_preserves_space_after_parenthetical_removal(self):
        card = self._build(
            merged={"longName": "Youngone Holdings Co., Ltd."},
            senior_metrics={"listing_role": "PURE_HOLDCO"},
            fla_report=(
                "ENTITY_ROLE_OBSERVED: PURE_HOLDCO\n"
                "Largest Shareholder: Sung Ki-hak (Chairman) and related parties.\n"
                "Controlling Shareholder: Sung Ki-hak (Chairman) and related parties."
                "\nControl Status: CONTROLLED\n"
                "Control Basis: VOTING_AGREEMENT\n"
                "Ownership Evidence Status: VERIFIED_OFFICIAL_FILING"
            ),
        )
        assert card.controlling_shareholder is not None
        assert (
            card.controlling_shareholder["name"] == "Sung Ki-hak and related parties."
        )

    def test_senior_parent_company_does_not_turn_holder_into_parent_fallback(self):
        card = self._build(
            merged={"longName": "Youngone Holdings Co., Ltd."},
            senior_metrics={
                "listing_role": "INTERMEDIATE_HOLDCO",
                "parent_company": "YMSA (29.09%)",
            },
            fla_report="ENTITY_ROLE_OBSERVED: INTERMEDIATE_HOLDCO",
        )
        assert card.controlling_shareholder is None

    def test_senior_parent_company_is_not_independent_control_evidence(self):
        card = self._build(
            merged={"longName": "Example Operating Co."},
            senior_metrics={
                "listing_role": "LISTED_SUBSIDIARY",
                "parent_company": "Example Holdings (60.0%)",
            },
            fla_report="ENTITY_ROLE_OBSERVED: LISTED_SUBSIDIARY",
        )
        assert card.controlling_shareholder is None
        assert card.control_status == "UNKNOWN"


# ---------------------------------------------------------------------------
# Prompt rendering + disclosure gating
# ---------------------------------------------------------------------------


class TestPromptRendering:
    def test_card_block_carries_identity_and_rules(self):
        card = EntityGovernanceCard(
            ticker="009970.KS",
            canonical_name="Youngone Holdings Co., Ltd.",
            entity_role="INTERMEDIATE_HOLDCO",
            confidence="clean",
            related_listed=[
                {
                    "ticker": "111770.KS",
                    "relationship": "operating_subsidiary",
                    "pct": 50.5,
                }
            ],
            metric_scope={"payout": "SEPARATE"},
        )
        block = card_to_prompt_block(card)
        assert "009970.KS" in block
        assert "Youngone Holdings Co., Ltd." in block
        assert "INTERMEDIATE_HOLDCO" in block
        assert "111770.KS" in block
        assert "Senior DATA_BLOCK metric scope" in block
        assert "scope conflict" in block
        assert "authoritative" in block.lower()

    def test_related_listed_dedupes_malformed_fla_prose_with_senior_edge(self):
        card = build_card(
            ticker="009970.KS",
            company_name="Youngone Holdings Co., Ltd.",
            merged_data={"longName": "Youngone Holdings Co., Ltd."},
            senior_metrics={
                "listing_role": "PURE_HOLDCO",
                "related_listed_tickers": "111770.KS:SUBSIDIARY:100%",
            },
            fla_report=(
                "ENTITY_ROLE_OBSERVED: PURE_HOLDCO\n"
                "Related Listed Tickers: 111770.KS (Youngone Corporation): Subsidiary)"
            ),
        )
        assert card.related_listed == [
            {"ticker": "111770.KS", "relationship": "SUBSIDIARY", "pct": 100.0}
        ]

    def test_card_block_from_dict_rehydrates_graph_state(self):
        block = card_to_prompt_block_from_dict(
            {
                "ticker": "009970.KS",
                "canonical_name": "Youngone Holdings Co., Ltd.",
                "entity_role": "INTERMEDIATE_HOLDCO",
                "confidence": "clean",
                "related_listed": [{"ticker": "111770.KS"}],
            }
        )
        assert "009970.KS" in block
        assert "INTERMEDIATE_HOLDCO" in block

    def test_card_from_dict_ignores_unknown_future_fields(self):
        card = card_from_dict(
            {
                "ticker": "009970.KS",
                "canonical_name": "Youngone Holdings Co., Ltd.",
                "entity_role": "INTERMEDIATE_HOLDCO",
                "confidence": "clean",
                "unknown_future_field": "ignored",
            }
        )
        assert card is not None
        assert card.canonical_name == "Youngone Holdings Co., Ltd."

    def test_disclosure_required_for_holdco_with_related(self):
        card = EntityGovernanceCard(
            ticker="X.KS",
            canonical_name="X Holdings",
            entity_role="INTERMEDIATE_HOLDCO",
            confidence="clean",
            related_listed=[{"ticker": "Y.KS", "relationship": "opco"}],
        )
        assert requires_structure_disclosure(card) is True

    def test_disclosure_not_required_for_holdco_without_related(self):
        """Berkshire-style standalone holdco with private subs → no noise."""
        card = EntityGovernanceCard(
            ticker="X",
            canonical_name="X Holdings",
            entity_role="PURE_HOLDCO",
            confidence="clean",
            related_listed=[],
        )
        assert requires_structure_disclosure(card) is False

    def test_disclosure_required_on_conflict_regardless_of_role(self):
        card = EntityGovernanceCard(
            ticker="X",
            canonical_name="X Inc",
            entity_role="UNKNOWN",
            confidence="conflict",
        )
        assert requires_structure_disclosure(card) is True


# ---------------------------------------------------------------------------
# Raw-string extractor
# ---------------------------------------------------------------------------


class TestExtractMergedSubset:
    def test_extracts_tool1_json(self):
        raw = (
            "=== RAW FINANCIAL DATA FOR 009970.KS ===\n\n"
            "### TOOL 1: get_financial_metrics\n"
            '{"longName": "Youngone Holdings Co., Ltd.", '
            '"longBusinessSummary": "changed its name to Holdings", '
            '"industry": "Apparel Manufacturing", "totalRevenue": 5000000000000, '
            '"otherField": "ignored"}\n\n'
            "### TOOL 2: ..."
        )
        subset = extract_merged_subset_from_raw(raw)
        assert subset["longName"] == "Youngone Holdings Co., Ltd."
        assert "otherField" not in subset
        assert subset["totalRevenue"] == 5000000000000

    def test_extracts_json_when_string_contains_braces(self):
        raw = (
            "### TOOL 1: get_financial_metrics\n"
            '{"longName": "Case Holdings", '
            '"longBusinessSummary": "Makes {braced} disclosures", '
            '"sector": "Financial Services"}\n'
            "### TOOL 2: other"
        )
        subset = extract_merged_subset_from_raw(raw)
        assert subset["longName"] == "Case Holdings"
        assert subset["longBusinessSummary"] == "Makes {braced} disclosures"

    def test_handles_missing_marker(self):
        assert extract_merged_subset_from_raw("no tool data here") == {}

    def test_handles_malformed_json(self):
        raw = "### TOOL 1: get_financial_metrics\n{not valid json}\n"
        assert extract_merged_subset_from_raw(raw) == {}

    def test_handles_empty_input(self):
        assert extract_merged_subset_from_raw("") == {}


# ---------------------------------------------------------------------------
# Round-trip via state dict (the path the validator actually uses)
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    def test_to_dict_and_back(self):
        """Card serializes via to_dict() and rehydrates via EntityGovernanceCard(**d)."""
        card = build_card(
            ticker="9984.T",
            company_name="SoftBank Group Corp.",
            merged_data={"longName": "SoftBank Group Corp."},
            senior_metrics={"listing_role": "PURE_HOLDCO"},
            fla_report="ENTITY_ROLE_OBSERVED: PURE_HOLDCO",
        )
        d = card.to_dict()
        assert isinstance(d, dict)
        restored = EntityGovernanceCard(**d)
        assert restored.ticker == card.ticker
        assert restored.entity_role == card.entity_role
        assert restored.confidence == card.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
