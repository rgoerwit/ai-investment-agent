import json
from types import SimpleNamespace

from src.analysis_snapshot import (
    add_validated_derivations,
    build_analysis_snapshot,
    build_pre_senior_snapshot,
    decision_claim_ids,
    project_analysis_report,
    reconcile_data_block_projection,
    refresh_analysis_snapshot,
    render_analysis_snapshot,
)
from src.article_audit import (
    audit_article_claim_support,
    audit_article_claim_usage,
    strip_claim_usage,
)
from src.pm_claim_audit import (
    reconcile_final_decision_trace,
    render_decision_trace_instruction,
    validate_decision_trace,
)
from src.tooling.structured_ingress import build_structured_ingress_record


def _fundamentals(*lines: str) -> str:
    return (
        "### --- START DATA_BLOCK ---\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n### --- END DATA_BLOCK ---"
    )


def _legacy_snapshot(
    state: dict[str, str],
    evidence: list[SimpleNamespace] | None = None,
) -> dict:
    """Build a trusted fixture snapshot; production legacy fallback is degraded."""
    return build_analysis_snapshot(state, evidence or [], degraded=False)


def _with_structured_metrics(state: dict, payload: dict | None = None) -> dict:
    updated = dict(state)
    if payload is None:
        payload = json.loads(str(updated.get("raw_fundamentals_data") or "{}"))
    updated["structured_inputs"] = {
        "raw_financial_metrics": build_structured_ingress_record(
            payload,
            agent_key="junior_fundamentals_analyst",
            tool_name="get_financial_metrics",
        )
    }
    return updated


def test_snapshot_binds_source_required_claim_to_one_evidence_record() -> None:
    url = "https://www.twse.com.tw/results"
    evidence = SimpleNamespace(
        sequence=7,
        content_sha256="abc123def456789",
        content=f"Capacity utilization was 95% as of 2025-12-31. {url}",
        urls=(url,),
        blocked=False,
        evidence_status="EVIDENCE_FOUND",
    )
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "CAPACITY_UTILIZATION: 95%",
                f"CAPACITY_UTILIZATION_SOURCE_URL: {url}",
                "CAPACITY_UTILIZATION_AS_OF: 2025-12-31",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
            )
        },
        [evidence],
    )

    claim = next(iter(snapshot["claims"].values()))
    assert claim["authority"] == "PRIMARY"
    assert claim["evidence_id"] == "evidence:7:abc123def456"
    assert claim["decision_eligible"] is True


def test_unbound_source_required_claim_is_visible_but_ineligible() -> None:
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "CAPACITY_UTILIZATION: 95%",
                "CAPACITY_UTILIZATION_SOURCE_URL: https://third.example/result",
                "CAPACITY_UTILIZATION_AS_OF: UNKNOWN",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
            )
        }
    )

    claim = next(
        claim
        for claim in snapshot["claims"].values()
        if claim["field"] == "CAPACITY_UTILIZATION"
    )
    assert claim["value"] == "N/A"
    assert claim["authority"] == "UNSUPPORTED"
    assert claim["decision_eligible"] is False
    assert snapshot["conflicts"][0]["type"] == "SOURCE_BINDING_MISSING"


def test_period_is_part_of_claim_identity_not_a_grade() -> None:
    first = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "EARNINGS_MRQ_PERIOD_END: 2025-12-31",
                "EARNINGS_GROWTH_MRQ: 102.8%",
            )
        }
    )
    second = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "EARNINGS_MRQ_PERIOD_END: 2026-03-31",
                "EARNINGS_GROWTH_MRQ: 102.8%",
            )
        }
    )

    first_claim = next(
        claim
        for claim in first["claims"].values()
        if claim["field"] == "EARNINGS_GROWTH_MRQ"
    )
    second_claim = next(
        claim
        for claim in second["claims"].values()
        if claim["field"] == "EARNINGS_GROWTH_MRQ"
    )
    assert first_claim["id"] != second_claim["id"]
    assert first_claim["period"] == "2025-12-31"
    assert second_claim["period"] == "2026-03-31"


def test_snapshot_render_and_decision_ids_exclude_unsupported_claims() -> None:
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "PE_RATIO_TTM: 12.5",
                "LATEST_RESULTS_EARNINGS_GROWTH_YOY: 80%",
                "LATEST_RESULTS_SOURCE_URL: https://third.example/result",
                "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED",
                "LATEST_RESULTS_PERIOD_END: 2025-12-31",
            )
        }
    )

    rendered = render_analysis_snapshot(snapshot)
    ids = decision_claim_ids(snapshot)
    assert "PE_RATIO_TTM: 12.5" in rendered
    assert "LATEST_RESULTS_EARNINGS_GROWTH_YOY: N/A" in rendered
    assert any("pe_ratio_ttm" in claim_id for claim_id in ids)
    assert all("latest_results_earnings" not in claim_id for claim_id in ids)


def test_missing_data_block_is_invalid_contract() -> None:
    snapshot = build_analysis_snapshot({"fundamentals_report": "prose only"})
    assert snapshot["contract_status"] == "INVALID"
    assert decision_claim_ids(snapshot) == ()


def test_post_senior_compatibility_snapshot_is_degraded_and_ineligible() -> None:
    snapshot = build_analysis_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )

    assert snapshot["contract_status"] == "DEGRADED"
    assert snapshot["contract_reason"] == "PRE_SENIOR_SNAPSHOT_UNAVAILABLE"
    assert decision_claim_ids(snapshot) == ()
    assert all(not claim["decision_eligible"] for claim in snapshot["claims"].values())


def test_decision_trace_accepts_only_eligible_claims_and_active_gates() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    claim_id = decision_claim_ids(snapshot)[0]
    red_flags = [{"type": "GROWTH_GAP", "blocks_buy": True}]
    output = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: GROWTH_GAP\n"
        "### --- END PM_BLOCK ---"
    )

    trace = validate_decision_trace(output, snapshot, red_flags)
    instruction = render_decision_trace_instruction(snapshot, red_flags)
    assert trace["status"] == "VALID"
    assert trace["decision_facts"] == [claim_id]
    assert claim_id in instruction
    assert "GROWTH_GAP" in instruction


def test_decision_trace_reads_verdict_only_from_pm_block() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    claim_id = decision_claim_ids(snapshot)[0]
    output = (
        "CONSULTANT_RESOLUTION:\n"
        "- VERDICT: UNVERIFIABLE\n\n"
        "### PORTFOLIO MANAGER VERDICT: BUY\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    trace = validate_decision_trace(output, snapshot, [])

    assert trace["status"] == "VALID"
    assert trace["verdict"] == "BUY"


def test_decision_trace_requires_cited_source_sensitive_rationale() -> None:
    url = "https://www.twse.com.tw/capacity"
    evidence = SimpleNamespace(
        sequence=8,
        content_sha256="def456abc123789",
        content=f"Capacity utilization was 95%. {url}",
        urls=(url,),
        blocked=False,
        evidence_status="EVIDENCE_FOUND",
    )
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "PE_RATIO_TTM: 12.5",
                "CAPACITY_UTILIZATION: 95%",
                f"CAPACITY_UTILIZATION_SOURCE_URL: {url}",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
            )
        },
        [evidence],
    )
    claims = {
        claim["field"]: claim_id for claim_id, claim in snapshot["claims"].items()
    }
    output = (
        "Capacity utilization is 95%, supporting near-term operating leverage.\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        f"DECISION_FACTS: {claims['PE_RATIO_TTM']}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    untraced = validate_decision_trace(output, snapshot, [])
    traced = validate_decision_trace(
        output.replace(
            claims["PE_RATIO_TTM"],
            f"{claims['PE_RATIO_TTM']}, {claims['CAPACITY_UTILIZATION']}",
        ),
        snapshot,
        [],
    )

    # Uncited source-sensitive prose is surfaced as an advisory signal but does
    # NOT structurally fail the trace — the loose marker classifier also flags
    # negated/contextual mentions, so promoting it to invalidating would
    # false-positive on benign prose (see the negation test in
    # tests/test_pm_claim_audit.py). The BUY here still has thesis support
    # (PE_RATIO_TTM), so the trace is VALID.
    assert untraced["status"] == "VALID"
    assert untraced["untraced_source_families"] == ["CAPACITY"]
    assert untraced["advisory_source_families"] == ["CAPACITY"]
    assert traced["status"] == "VALID"


def test_decision_trace_allows_explicit_source_evidence_gap() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    claim_id = decision_claim_ids(snapshot)[0]
    output = (
        "Capacity utilization is unverified and remains an evidence gap.\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    assert validate_decision_trace(output, snapshot, [])["status"] == "VALID"


def test_every_final_decision_requires_a_fact_or_active_gate() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    buy = (
        "### --- START PM_BLOCK ---\nVERDICT: BUY\n"
        "DECISION_FACTS: NONE\nDECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )
    hold = buy.replace("VERDICT: BUY", "VERDICT: HOLD")

    assert validate_decision_trace(buy, snapshot, [])["status"] == "INVALID"
    assert validate_decision_trace(hold, snapshot, [])["status"] == "INVALID"


def test_buy_citing_only_current_price_is_not_thesis_support() -> None:
    """CURRENT_PRICE is a SUPPORT claim but incidental — a BUY citing only it has
    no thesis-bearing claim and must be INVALID."""
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "PE_RATIO_TTM: 12.5",
                "CURRENT_PRICE: 178",
            )
        }
    )
    claims = {
        claim["field"]: claim_id for claim_id, claim in snapshot["claims"].items()
    }
    price_only = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        f"DECISION_FACTS: {claims['CURRENT_PRICE']}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )
    with_thesis = price_only.replace(
        claims["CURRENT_PRICE"],
        f"{claims['CURRENT_PRICE']}, {claims['PE_RATIO_TTM']}",
    )

    price_trace = validate_decision_trace(price_only, snapshot, [])
    thesis_trace = validate_decision_trace(with_thesis, snapshot, [])

    assert price_trace["status"] == "INVALID"
    assert price_trace["support_facts"] == [claims["CURRENT_PRICE"]]
    assert price_trace["thesis_support_facts"] == []
    # Adding a valuation claim provides thesis support → VALID.
    assert thesis_trace["status"] == "VALID"
    assert thesis_trace["thesis_support_facts"] == [claims["PE_RATIO_TTM"]]


def test_gate_input_cannot_independently_support_buy() -> None:
    snapshot = _legacy_snapshot({"fundamentals_report": _fundamentals("DE_RATIO: 20%")})
    claim_id = next(iter(snapshot["claims"]))
    output = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    trace = validate_decision_trace(output, snapshot, [])
    assert trace["status"] == "INVALID"
    assert trace["support_facts"] == []


def test_final_trace_reconciliation_adds_gate_after_buy_demotion() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    claim_id = decision_claim_ids(snapshot)[0]
    output = (
        "### PORTFOLIO MANAGER VERDICT: HOLD\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    reconciled, trace = reconcile_final_decision_trace(
        output,
        snapshot,
        [{"type": "DECISION_TRACE_INVALID", "blocks_buy": True}],
    )

    assert "DECISION_GATES: DECISION_TRACE_INVALID" in reconciled
    assert trace["status"] == "VALID"


def test_final_trace_reconciliation_projects_canonical_scores_into_pm_block() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    snapshot["scorecards"] = {
        "HEALTH": {"percentage": 72.4, "decision_eligible": True},
        "GROWTH": {"percentage": 66.6, "decision_eligible": True},
    }
    claim_id = decision_claim_ids(snapshot)[0]
    output = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        "HEALTH_ADJ: 95\n"
        "GROWTH_ADJ: 20\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    reconciled, trace = reconcile_final_decision_trace(output, snapshot, [])

    assert "HEALTH_ADJ: 72" in reconciled
    assert "GROWTH_ADJ: 67" in reconciled
    assert trace["status"] == "VALID"


def test_final_trace_reconciliation_clears_unregistered_pm_scores() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    claim_id = decision_claim_ids(snapshot)[0]
    output = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        "HEALTH_ADJ: 95\n"
        "GROWTH_ADJ: 95\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    reconciled, trace = reconcile_final_decision_trace(output, snapshot, [])

    assert "HEALTH_ADJ: N/A" in reconciled
    assert "GROWTH_ADJ: N/A" in reconciled
    assert trace["status"] == "VALID"


def test_unsupported_claim_reference_is_invalid() -> None:
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "CAPACITY_UTILIZATION: 95%",
                "CAPACITY_UTILIZATION_SOURCE_URL: https://third.example/result",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
                "CAPACITY_UTILIZATION_AS_OF: UNKNOWN",
            )
        }
    )
    claim_id = next(iter(snapshot["claims"]))
    output = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        f"DECISION_FACTS: {claim_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )

    trace = validate_decision_trace(output, snapshot, [])
    assert trace["status"] == "INVALID"
    assert trace["invalid_facts"] == [claim_id]


def test_article_audit_rejects_unsupported_capacity_assertion() -> None:
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "CAPACITY_UTILIZATION: 95%",
                "CAPACITY_UTILIZATION_SOURCE_URL: https://third.example/result",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
                "CAPACITY_UTILIZATION_AS_OF: UNKNOWN",
            )
        }
    )

    errors = audit_article_claim_support(
        "The plant's capacity utilization is confirmed at 95%.",
        snapshot,
    )
    assert len(errors) == 1
    assert "not assertion-eligible" in errors[0]["ground_truth"]


def test_article_audit_allows_explicitly_conditional_unsupported_claim() -> None:
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "CAPACITY_UTILIZATION: 95%",
                "CAPACITY_UTILIZATION_SOURCE_URL: https://third.example/result",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
                "CAPACITY_UTILIZATION_AS_OF: UNKNOWN",
            )
        }
    )

    errors = audit_article_claim_support(
        "A secondary source suggests high utilization, but the level is unverified.",
        snapshot,
    )
    assert errors == []


def test_article_claim_usage_binds_source_sensitive_prose_and_is_stripped() -> None:
    url = "https://issuer.example/results"
    snapshot = _legacy_snapshot(
        {
            "fundamentals_report": _fundamentals(
                "LATEST_RESULTS_REVENUE_GROWTH_YOY: 16.9%",
                f"LATEST_RESULTS_SOURCE_URL: {url}",
                "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY",
                "LATEST_RESULTS_PERIOD_END: 2025-12-31",
                "LATEST_RESULTS_COVERAGE_STATUS: FOUND",
            )
        },
        [
            SimpleNamespace(
                sequence=4,
                content_sha256="abcdef123456789",
                content="Issuer results document",
                urls=(url,),
                blocked=False,
                evidence_status="EVIDENCE_FOUND",
            )
        ],
    )
    claim_id = next(
        claim_id
        for claim_id, claim in snapshot["claims"].items()
        if claim["field"] == "LATEST_RESULTS_REVENUE_GROWTH_YOY"
    )
    sentence = "Revenue grew 16.9% in the latest validated results."
    article = (
        f"# Example\n\n{sentence}\n\n```CLAIM_USAGE\n- {claim_id} | {sentence}\n```\n"
    )

    assert audit_article_claim_usage(article, snapshot) == []
    assert "CLAIM_USAGE" not in strip_claim_usage(article)


def test_article_claim_usage_rejects_unbound_source_sensitive_number() -> None:
    snapshot = _legacy_snapshot(
        {"fundamentals_report": _fundamentals("PE_RATIO_TTM: 12.5")}
    )
    claim = next(iter(snapshot["claims"].values()))
    claim["field"] = "LATEST_RESULTS_REVENUE_GROWTH_YOY"
    claim["value"] = "16.9%"

    errors = audit_article_claim_usage(
        "Revenue grew 16.9% in the latest results.",
        snapshot,
    )

    assert len(errors) == 1
    assert errors[0]["location"] == "Canonical claim usage audit"


def test_pre_senior_snapshot_owns_statement_mrq_period_and_missing_capacity() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"revenueGrowth_MRQ": 0.168693, '
                    '"_revenueGrowth_MRQ_source": "calculated_from_quarterly", '
                    '"latest_quarter_date": "2025-12-31"}'
                ),
                "foreign_language_report": (
                    "CAPACITY_UTILIZATION: N/A\n"
                    "CAPACITY_UTILIZATION_SOURCE_URL: N/A\n"
                    "CAPACITY_UTILIZATION_AS_OF: UNKNOWN\n"
                    "CAPACITY_EVIDENCE_STATUS: UNSUPPORTED"
                ),
            }
        )
    )

    claims = {claim["field"]: claim for claim in snapshot["claims"].values()}
    assert claims["REVENUE_GROWTH_MRQ"]["value"] == "16.9%"
    assert claims["REVENUE_GROWTH_MRQ"]["period"] == "2025-12-31"
    assert claims["REVENUE_GROWTH_MRQ"]["exactness"] == "CALCULATED"
    assert claims["CAPACITY_UTILIZATION"]["value"] == "N/A"
    assert snapshot["stage"] == "PRE_SENIOR"


def test_mixed_source_mrq_metrics_do_not_share_period_provenance() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"revenueGrowth_MRQ": 0.12, '
                    '"_revenueGrowth_MRQ_source": "calculated_from_quarterly", '
                    '"earningsGrowth_MRQ": 1.03, '
                    '"_earningsGrowth_MRQ_source": "aggregator", '
                    '"latest_quarter_date": "2025-12-31"}'
                )
            }
        )
    )
    claims = {claim["field"]: claim for claim in snapshot["claims"].values()}

    assert claims["REVENUE_GROWTH_MRQ"]["period"] == "2025-12-31"
    assert claims["EARNINGS_GROWTH_MRQ"]["period"] is None

    reconciled, _ = reconcile_data_block_projection(
        "LATEST_QUARTER_DATE: 2025-12-31",
        snapshot,
    )
    assert "REVENUE_MRQ_PERIOD_END: 2025-12-31" in reconciled
    assert "EARNINGS_MRQ_PERIOD_END: UNKNOWN" in reconciled
    assert "LATEST_QUARTER_DATE: UNKNOWN" in reconciled


def test_refresh_does_not_keep_competing_periods_for_one_metric() -> None:
    prior = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"revenueGrowth_MRQ": 0.10, '
                    '"_revenueGrowth_MRQ_source": "calculated_from_quarterly", '
                    '"latest_quarter_date": "2025-09-30"}'
                )
            }
        )
    )
    refreshed = refresh_analysis_snapshot(
        prior,
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"revenueGrowth_MRQ": 0.12, '
                    '"_revenueGrowth_MRQ_source": "calculated_from_quarterly", '
                    '"latest_quarter_date": "2025-12-31"}'
                )
            }
        ),
        [],
        version=2,
    )

    revenue_claims = [
        claim
        for claim in refreshed["claims"].values()
        if claim["field"] == "REVENUE_GROWTH_MRQ"
    ]
    assert len(revenue_claims) == 1
    assert revenue_claims[0]["period"] == "2025-12-31"


def test_absent_source_required_claim_overwrites_only_the_claim_value() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics({}, {"trailingPE": 12.5})
    )
    claims = {claim["field"]: claim for claim in snapshot["claims"].values()}

    assert claims["LATEST_RESULTS_REVENUE_GROWTH_YOY"]["value"] == "N/A"
    assert claims["LATEST_RESULTS_REVENUE_GROWTH_YOY"]["coverage"] == "MISSING"

    reconciled, conflicts = reconcile_data_block_projection(
        "LATEST_RESULTS_REVENUE_GROWTH_YOY: 99%",
        snapshot,
    )
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in reconciled
    assert any(
        conflict["field"] == "LATEST_RESULTS_REVENUE_GROWTH_YOY"
        for conflict in conflicts
    )


def test_projection_never_reverse_projects_shared_family_coverage() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "foreign_language_report": (
                    "### --- START MANAGEMENT_GUIDANCE ---\n"
                    "COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH\n"
                    "GUIDANCE_PERIOD: N/A\n"
                    "REVENUE_GUIDANCE: N/A\n"
                    "NET_INCOME_GUIDANCE: N/A\n"
                    "### --- END MANAGEMENT_GUIDANCE ---\n"
                    "### --- START LATEST_RESULTS ---\n"
                    "LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND\n"
                    "LATEST_RESULTS_PERIOD: N/A\n"
                    "LATEST_RESULTS_PERIOD_END: N/A\n"
                    "LATEST_RESULTS_SOURCE_URL: N/A\n"
                    "### --- END LATEST_RESULTS ---"
                )
            },
            {"trailingPE": 12.5},
        )
    )
    body = (
        "GUIDANCE_COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH\n"
        "LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND"
    )

    reconciled, _ = reconcile_data_block_projection(body, snapshot)
    reversed_snapshot = {
        **snapshot,
        "claims": dict(reversed(tuple(snapshot["claims"].items()))),
    }
    reversed_reconciled, _ = reconcile_data_block_projection(
        body,
        reversed_snapshot,
    )

    assert "GUIDANCE_COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH" in reconciled
    assert "LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND" in reconciled
    assert reversed_reconciled == reconciled


def test_snapshot_preserves_searched_but_unresolved_coverage() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "foreign_language_report": (
                    "### --- START MANAGEMENT_GUIDANCE ---\n"
                    "COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH\n"
                    "REVENUE_GUIDANCE: N/A\n"
                    "NET_INCOME_GUIDANCE: N/A\n"
                    "### --- END MANAGEMENT_GUIDANCE ---"
                )
            },
            {"trailingPE": 12.5},
        )
    )
    guidance_claims = [
        claim
        for claim in snapshot["claims"].values()
        if claim["field"] in {"GUIDANCE_REVENUE", "GUIDANCE_NET_INCOME"}
    ]

    assert guidance_claims
    assert {claim["coverage"] for claim in guidance_claims} == {"SEARCHED_UNRESOLVED"}


def test_senior_cannot_remint_registered_capacity_or_relabel_mrq() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"earningsGrowth_MRQ": 1.028, '
                    '"_earningsGrowth_MRQ_source": "calculated_from_quarterly", '
                    '"latest_quarter_date": "2025-12-31"}'
                ),
                "foreign_language_report": (
                    "CAPACITY_UTILIZATION: N/A\n"
                    "CAPACITY_UTILIZATION_SOURCE_URL: N/A\n"
                    "CAPACITY_UTILIZATION_AS_OF: UNKNOWN\n"
                    "CAPACITY_EVIDENCE_STATUS: UNSUPPORTED"
                ),
            }
        )
    )
    body = (
        "EARNINGS_GROWTH_MRQ: 102.8% (as of 2026-03-31)\n"
        "LATEST_QUARTER_DATE: 2026-03-31\n"
        "CAPACITY_UTILIZATION: 95%\n"
        "CAPACITY_EVIDENCE_STATUS: PRIMARY"
    )

    reconciled, conflicts = reconcile_data_block_projection(body, snapshot)
    assert "EARNINGS_GROWTH_MRQ: 102.8%" in reconciled
    assert "EARNINGS_MRQ_PERIOD_END: 2025-12-31" in reconciled
    assert "LATEST_QUARTER_DATE: 2025-12-31" in reconciled
    assert "CAPACITY_UTILIZATION: N/A" in reconciled
    assert "CAPACITY_EVIDENCE_STATUS: UNSUPPORTED" in reconciled
    assert {conflict["field"] for conflict in conflicts} >= {
        "EARNINGS_GROWTH_MRQ",
        "CAPACITY_UTILIZATION",
    }


def test_score_derivation_requires_complete_coherent_breakdown() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"trailingPE": 12.5, "revenueGrowth": 0.2, "earningsGrowth": 0.3}'
                )
            }
        )
    )
    incomplete = _fundamentals(
        "ADJUSTED_GROWTH_SCORE: 80.0% (based on 5 available points)",
        "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1",
    )
    # GROSS_MARGIN has a real configured dependency (GROSS_MARGIN_PERCENT /
    # raw "grossMargins") that this snapshot doesn't supply, so it's a
    # genuine, blocking lineage gap. R_AND_D_CAPEX_BACKLOG has no configured
    # dependency at all ("advisory until its producer emits structured
    # evidence") and must NOT block eligibility on its own merely because it
    # was awarded a point — see test_structurally_unbacked_criteria_do_not_
    # veto_eligibility below for the case where it's the *only* would-be gap.
    complete = _fundamentals(
        "ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)",
        (
            "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; "
            "ROA_ROE_IMPROVING=0; GROSS_MARGIN=1; GLOBAL_EXPANSION=0; "
            "R_AND_D_CAPEX_BACKLOG=1"
        ),
    )

    assert all(
        claim["field"] != "ADJUSTED_GROWTH_SCORE"
        for claim in add_validated_derivations(snapshot, incomplete)["claims"].values()
    )
    derived = add_validated_derivations(snapshot, complete)
    score = next(
        claim
        for claim in derived["claims"].values()
        if claim["field"] == "ADJUSTED_GROWTH_SCORE"
    )
    assert score["kind"] == "DERIVED_ASSESSMENT"
    assert score["decision_eligible"] is False
    assert score["decision_role"] == "GATE_INPUT"
    assert len(score["derived_from"]) == 2
    # R_AND_D_CAPEX_BACKLOG=1 is a class-1 advisory award (no evidence producer),
    # so it is excluded from the decision score: 3/6 = 50.0% (raw 4/6 = 66.7%
    # remains as advisory_percentage). Eligibility is still False here because of
    # the class-2 GROSS_MARGIN lineage gap.
    assert score["value"] == "50.0% (based on 6 available points)"
    scorecard = derived["scorecards"]["GROWTH"]
    assert scorecard["criteria"]["GLOBAL_EXPANSION"]["award"] == "0"
    assert scorecard["criteria"]["R_AND_D_CAPEX_BACKLOG"]["award"] == "1"
    assert scorecard["lineage_gaps"] == ["GROSS_MARGIN"]
    assert scorecard["advisory_only_awards"] == ["R_AND_D_CAPEX_BACKLOG"]
    assert scorecard["advisory_percentage"] == 66.7


def test_structurally_unbacked_criteria_do_not_veto_eligibility() -> None:
    """GLOBAL_EXPANSION/R_AND_D_CAPEX_BACKLOG have no configured evidence
    producer yet; a nonzero award on them alone must not zero the whole
    scorecard the way a genuinely-unresolved dependency does. It IS excluded
    from the decision score (conservative) while the model's raw percentage is
    retained as advisory_percentage."""
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"trailingPE": 12.5, "revenueGrowth": 0.2, '
                    '"earningsGrowth": 0.3, "grossMargins": 0.4}'
                )
            }
        )
    )
    report = _fundamentals(
        "ADJUSTED_GROWTH_SCORE: 66.7% (based on 6 available points)",
        (
            "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; "
            "ROA_ROE_IMPROVING=0; GROSS_MARGIN=1; GLOBAL_EXPANSION=1; "
            "R_AND_D_CAPEX_BACKLOG=0"
        ),
    )

    derived = add_validated_derivations(snapshot, report)
    score = next(
        claim
        for claim in derived["claims"].values()
        if claim["field"] == "ADJUSTED_GROWTH_SCORE"
    )
    scorecard = derived["scorecards"]["GROWTH"]

    assert scorecard["lineage_gaps"] == []
    assert score["decision_eligible"] is True
    assert score["coverage"] == "FOUND"
    # GLOBAL_EXPANSION=1 is advisory-only: decision score drops it (3/6 = 50.0%),
    # advisory_percentage keeps the model's raw 66.7%.
    assert score["value"] == "50.0% (based on 6 available points)"
    assert scorecard["percentage"] == 50.0
    assert scorecard["advisory_percentage"] == 66.7
    assert scorecard["advisory_only_awards"] == ["GLOBAL_EXPANSION"]


def test_fully_backed_scorecard_decision_equals_advisory() -> None:
    """When no advisory-only (unbacked) positive award exists, the decision
    score is byte-identical to the model's raw percentage — parity guarantee."""
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"trailingPE": 12.5, "revenueGrowth": 0.2, '
                    '"earningsGrowth": 0.3, "grossMargins": 0.4}'
                )
            }
        )
    )
    report = _fundamentals(
        "ADJUSTED_GROWTH_SCORE: 50.0% (based on 6 available points)",
        (
            "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; "
            "ROA_ROE_IMPROVING=0; GROSS_MARGIN=1; GLOBAL_EXPANSION=0; "
            "R_AND_D_CAPEX_BACKLOG=0"
        ),
    )

    derived = add_validated_derivations(snapshot, report)
    scorecard = derived["scorecards"]["GROWTH"]

    assert scorecard["advisory_only_awards"] == []
    assert scorecard["percentage"] == scorecard["advisory_percentage"] == 50.0
    assert scorecard["decision_eligible"] is True


def test_missing_structured_ingress_fails_closed_instead_of_minting_na_truth() -> None:
    snapshot = build_pre_senior_snapshot(
        {"raw_fundamentals_data": '{"currentPrice": 178, "trailingPE": 12.7}'}
    )

    assert snapshot["contract_status"] == "INVALID"
    assert snapshot["contract_reason"] == "STRUCTURED_INPUT_REGISTRY_MISSING"
    assert not any(claim["decision_eligible"] for claim in snapshot["claims"].values())


def test_structured_ingress_wins_over_divergent_junior_relay() -> None:
    state = _with_structured_metrics(
        {"raw_fundamentals_data": '{"currentPrice": 999, "trailingPE": 88}'},
        {"currentPrice": 178, "trailingPE": 12.7},
    )

    snapshot = build_pre_senior_snapshot(state)
    claims = {claim["field"]: claim for claim in snapshot["claims"].values()}

    assert snapshot["contract_status"] == "VALID"
    assert claims["CURRENT_PRICE"]["value"] == "178"
    assert claims["PE_RATIO_TTM"]["value"] == "12.7"


def test_identity_only_structured_ingress_is_degraded() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics({}, {"currentPrice": 178.0, "marketCap": 1_000})
    )

    assert snapshot["contract_status"] == "DEGRADED"
    assert snapshot["contract_reason"] == "RAW_METRICS_NO_USABLE_ANALYTIC_FIELDS"
    assert not any(claim["decision_eligible"] for claim in snapshot["claims"].values())


def test_malformed_analytic_values_do_not_make_contract_valid() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {},
            {
                "currentPrice": 178.0,
                "trailingPE": {"hostile": "not a number"},
                "revenueGrowth": True,
            },
        )
    )

    assert snapshot["contract_status"] == "DEGRADED"
    assert snapshot["contract_reason"] == "RAW_METRICS_NO_USABLE_ANALYTIC_FIELDS"
    assert not any(claim["decision_eligible"] for claim in snapshot["claims"].values())


def test_canonical_scorecard_replaces_stale_score_detail() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics(
            {
                "raw_fundamentals_data": (
                    '{"revenueGrowth": 0.2, "earningsGrowth": 0.3, "grossMargins": 0.4}'
                )
            }
        )
    )
    report = (
        _fundamentals(
            "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1; "
            "ROA_ROE_IMPROVING=0; GROSS_MARGIN=1; GLOBAL_EXPANSION=1; "
            "R_AND_D_CAPEX_BACKLOG=1",
            "RAW_GROWTH_SCORE: 5/6",
            "ADJUSTED_GROWTH_SCORE: 83.3% (based on 6 available points)",
        )
        + "\n\n### GROWTH TRANSITION DETAIL\n**Score**: 5/6 (Adjusted: 83%)\n"
    )
    derived = add_validated_derivations(snapshot, report)

    projected = project_analysis_report(report, derived)

    assert "GROWTH_SCORE_BREAKDOWN: REVENUE_GROWTH=1; EPS_GROWTH=1;" in projected
    assert "GLOBAL_EXPANSION=1; R_AND_D_CAPEX_BACKLOG=1" in projected
    assert "RAW_GROWTH_SCORE: 5/6" in projected
    # GLOBAL_EXPANSION=1 and R_AND_D_CAPEX_BACKLOG=1 are both advisory-only
    # (no evidence producer), so the decision score excludes both: 3/6 = 50.0%.
    # RAW stays 5/6; the model's raw 83.3% survives only as advisory context.
    assert "ADJUSTED_GROWTH_SCORE: 50.0% (based on 6 available points)" in projected
    assert "GROWTH_SCORE_LINEAGE_STATUS: COMPLETE" in projected
    assert projected.count("### GROWTH TRANSITION DETAIL") == 1
    assert "**Score**: 5/6 (Adjusted: 50.0%)" in projected
    assert "**Score**: 5/6 (Adjusted: 83%)" not in projected


def test_missing_scorecard_cannot_leave_model_score_load_bearing() -> None:
    snapshot = build_pre_senior_snapshot(
        _with_structured_metrics({}, {"trailingPE": 12.5})
    )
    report = _fundamentals(
        "ADJUSTED_HEALTH_SCORE: 100%",
        "ADJUSTED_GROWTH_SCORE: 100%",
    )

    projected = project_analysis_report(report, snapshot)

    assert "ADJUSTED_HEALTH_SCORE: N/A" in projected
    assert "HEALTH_SCORE_LINEAGE_STATUS: MISSING" in projected
    assert "ADJUSTED_GROWTH_SCORE: N/A" in projected
    assert "GROWTH_SCORE_LINEAGE_STATUS: MISSING" in projected


def test_fully_rooted_health_score_has_transitive_lineage() -> None:
    payload = {
        "returnOnEquity": 0.2,
        "returnOnAssets": 0.1,
        "operatingMargins": 0.15,
        "debtToEquity": 30.0,
        "totalDebt": 100.0,
        "totalCash": 20.0,
        "ebitda": 50.0,
        "currentRatio": 2.0,
        "operatingCashflow": 30.0,
        "freeCashflow": 20.0,
        "marketCap": 200.0,
        "trailingPE": 12.5,
        "enterpriseToEbitda": 8.0,
        "priceToBook": 1.2,
    }
    snapshot = build_pre_senior_snapshot(_with_structured_metrics({}, payload))
    report = _fundamentals(
        "HEALTH_SCORE_BREAKDOWN: "
        "ROE=1; ROA=1; OPERATING_MARGIN=1; DE_RATIO=1; "
        "NET_DEBT_EBITDA=1; CURRENT_RATIO=1; OCF_POSITIVE=1; "
        "FCF_POSITIVE=1; FCF_YIELD=1; PE_OR_PEG=1; "
        "EV_EBITDA=1; PB_OR_PS=1",
        "RAW_HEALTH_SCORE: 12/12",
        "ADJUSTED_HEALTH_SCORE: 100.0% (based on 12 available points)",
    )

    derived = add_validated_derivations(snapshot, report)
    score = next(
        claim
        for claim in derived["claims"].values()
        if claim["field"] == "ADJUSTED_HEALTH_SCORE"
    )

    assert score["decision_eligible"] is True
    assert score["derived_from"]
    assert all(
        component["derived_from"]
        for component in derived["scorecards"]["HEALTH"]["criteria"].values()
    )
