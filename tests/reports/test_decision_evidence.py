from src.reporting.decision_evidence import render_decision_evidence_markdown


def test_decision_evidence_links_only_inspected_eligible_source() -> None:
    state = {
        "evidence_records": [
            {
                "sequence": 7,
                "content_sha256": "abcdef1234567890",
                "requested_urls": ["https://issuer.example/results"],
                "urls": ["https://issuer.example/results"],
                "blocked": False,
                "execution_status": "SUCCEEDED",
                "evidence_status": "EVIDENCE_FOUND",
            }
        ],
        "analysis_snapshot": {
            "claims": {
                "claim:external": {
                    "field": "GUIDANCE_REVENUE",
                    "value": "+5%",
                    "authority": "PRIMARY",
                    "coverage": "FOUND",
                    "decision_eligible": True,
                    "evidence_id": "evidence:7:abcdef123456",
                    "source_url": "https://issuer.example/results",
                },
                "claim:aggregator": {
                    "field": "PE_RATIO_TTM",
                    "value": "12.0",
                    "authority": "AGGREGATOR",
                    "coverage": "FOUND",
                    "decision_eligible": True,
                    "evidence_id": "raw:yfinance:trailingPE",
                    "source_url": None,
                },
            }
        },
        "decision_trace": {"decision_facts": ["claim:external", "claim:aggregator"]},
    }

    rendered = render_decision_evidence_markdown(state)

    assert rendered.count("[inspected source]") == 1
    assert "1 decision fact(s)" in rendered
    assert "PE_RATIO_TTM" in rendered


def test_decision_evidence_does_not_trust_claim_without_matching_record() -> None:
    state = {
        "evidence_records": [],
        "analysis_snapshot": {
            "claims": {
                "claim:external": {
                    "field": "GUIDANCE_REVENUE",
                    "value": "+5%",
                    "authority": "PRIMARY",
                    "coverage": "FOUND",
                    "decision_eligible": True,
                    "evidence_id": "evidence:7:abcdef123456",
                    "source_url": "https://issuer.example/results",
                }
            }
        },
        "decision_trace": {"decision_facts": ["claim:external"]},
    }

    rendered = render_decision_evidence_markdown(state)

    assert "[inspected source]" not in rendered
    assert "No decision fact is backed by an inspected external document" in rendered


def test_decision_evidence_uses_canonical_id_for_record_without_sequence() -> None:
    url = "https://issuer.example/results"
    state = {
        "evidence_records": [
            {
                "content_sha256": "abcdef1234567890",
                "requested_urls": [url],
                "urls": [url],
                "blocked": False,
                "execution_status": "SUCCEEDED",
                "evidence_status": "EVIDENCE_FOUND",
            }
        ],
        "analysis_snapshot": {
            "claims": {
                "claim:external": {
                    "field": "GUIDANCE_REVENUE",
                    "value": "+5%",
                    "authority": "PRIMARY",
                    "coverage": "FOUND",
                    "decision_eligible": True,
                    "evidence_id": "evidence:0:abcdef123456",
                    "source_url": url,
                }
            }
        },
        "decision_trace": {"decision_facts": ["claim:external"]},
    }

    assert "[inspected source]" in render_decision_evidence_markdown(state)


def test_decision_evidence_rejects_failed_or_url_mismatched_record() -> None:
    claim = {
        "field": "GUIDANCE_REVENUE",
        "value": "+5%",
        "authority": "PRIMARY",
        "coverage": "FOUND",
        "decision_eligible": True,
        "evidence_id": "evidence:7:abcdef123456",
        "source_url": "https://issuer.example/results",
    }
    record = {
        "sequence": 7,
        "content_sha256": "abcdef1234567890",
        "requested_urls": ["https://other.example/results"],
        "urls": ["https://other.example/results"],
        "blocked": False,
        "execution_status": "SUCCEEDED",
        "evidence_status": "EVIDENCE_FOUND",
    }
    state = {
        "evidence_records": [record],
        "analysis_snapshot": {"claims": {"claim:external": claim}},
        "decision_trace": {"decision_facts": ["claim:external"]},
    }

    assert "[inspected source]" not in render_decision_evidence_markdown(state)
    record["requested_urls"] = [claim["source_url"]]
    record["urls"] = [claim["source_url"]]
    record["execution_status"] = "FAILED"
    assert "[inspected source]" not in render_decision_evidence_markdown(state)


def test_decision_evidence_exposes_zero_links_and_buy_constraints() -> None:
    state = {
        "analysis_snapshot": {
            "claims": {
                "claim:pe": {
                    "field": "PE_RATIO_TTM",
                    "value": "12.0",
                    "authority": "AGGREGATOR",
                    "coverage": "FOUND",
                    "decision_eligible": True,
                    "evidence_id": "raw:yfinance:trailingPE",
                    "source_url": None,
                }
            }
        },
        "decision_trace": {"decision_facts": ["claim:pe"]},
        "red_flags": [
            {
                "type": "MANAGEMENT_GUIDANCE_EVIDENCE_GAP",
                "detail": "Guidance could not be verified.",
                "blocks_buy": True,
            }
        ],
    }

    rendered = render_decision_evidence_markdown(state)

    assert "No decision fact is backed by an inspected external document" in rendered
    assert "unverified review context" in rendered
    assert "not external verification" in rendered
    assert "MANAGEMENT_GUIDANCE_EVIDENCE_GAP" in rendered


def test_quick_screen_explains_expected_zero_external_evidence() -> None:
    state = {
        "analysis_snapshot": {
            "claims": {
                "claim:pe": {
                    "field": "PE_RATIO_TTM",
                    "value": "12.0",
                    "authority": "AGGREGATOR",
                    "coverage": "FOUND",
                    "decision_eligible": True,
                    "evidence_id": "raw:provider:trailingPE",
                    "source_url": None,
                }
            }
        },
        "decision_trace": {"decision_facts": ["claim:pe"]},
        "run_summary": {"quick_mode": True},
    }

    rendered = render_decision_evidence_markdown(state)

    assert "Quick screening intentionally disables" in rendered
    assert "promoted to full analysis before action" in rendered
    assert "No decision fact is backed" not in rendered


def test_decision_evidence_omits_section_without_snapshot_or_trace() -> None:
    assert render_decision_evidence_markdown({}) == ""
