"""A sourced holder name can survive when the source omits its percentage."""

from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
from src.agents.message_utils import make_tool_evidence_record


def _report(holder: str) -> str:
    return f"""**OWNERSHIP STRUCTURE**
- Largest Shareholder: {holder}
- Control Status: CONTROLLED
- Control Basis: MAJORITY_VOTING_RIGHTS
- Ownership Evidence Status: CITED
- Ownership Source URL: https://issuer.example/holders
"""


def test_sourced_holder_without_percentage_keeps_name_but_not_control() -> None:
    evidence = make_tool_evidence_record(
        tool_name="get_official_filings",
        content="Acme Holdings is the largest shareholder. https://issuer.example/holders",
        urls={"https://issuer.example/holders"},
    )

    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings"), [], ticker="TEST.T", additional_records=[evidence]
    )

    assert "Largest Shareholder: Acme Holdings" in normalized
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: UNKNOWN" in normalized


def test_name_without_largest_holder_evidence_is_rejected() -> None:
    evidence = make_tool_evidence_record(
        tool_name="get_official_filings",
        content="Acme Holdings is a shareholder. https://issuer.example/holders",
        urls={"https://issuer.example/holders"},
    )

    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings"), [], ticker="TEST.T", additional_records=[evidence]
    )

    assert "Largest Shareholder: UNKNOWN" in normalized
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_retained_yara_state_percentage_is_ownership_without_majority_control():
    # Public Oct-5 official shareholder snippet; exact rejected original claim is unavailable.
    url = "https://www.yara.com/investor-relations/share-and-debt-information/shareholders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"<result><url>{url}</url><summary>| Country | Holders | Shares | Share of total |\n| Norwegian state | 1 | 92,239,891 | 36.2% |</summary></result>",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Norwegian state (36.2%)").replace(
            "https://issuer.example/holders", url
        ),
        [],
        ticker="YAR.OL",
        additional_records=[evidence],
    )
    assert "Largest Shareholder: Norwegian state (36.2%)" in normalized
    assert "Control Status: UNKNOWN" in normalized


def test_retained_sag_native_holder_percentage_does_not_establish_control():
    url = "https://news.cnyes.com/news/id/6601874"
    content = f"頻率元件廠希華是韋僑的最大法人股東，對韋僑持股 13.63%，希華董事長曾穎堂同時是韋僑的董事長。 {url}"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources", content=content, urls={url}
    )
    normalized = normalize_foreign_language_evidence(
        _report("希華 (13.63%)").replace("https://issuer.example/holders", url),
        [],
        ticker="6417.TWO",
        additional_records=[evidence],
    )
    assert "Largest Shareholder: 希華 (13.63%)" in normalized
    assert "Control Status: UNKNOWN" in normalized


def test_asset_ownership_percentage_cannot_become_issuer_controller_percentage():
    url = "https://issuer.example/holders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"JHSF bought a 49.99% interest in a shopping centre. Its board includes Jose Auriemo. {url}",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Jose Auriemo (49.99%)"),
        [],
        ticker="JHSF3.SA",
        additional_records=[evidence],
    )
    assert "Largest Shareholder: UNKNOWN" in normalized
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_acquisition_in_same_sentence_is_not_shareholder_percentage():
    url = "https://issuer.example/holders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"Jose Auriemo led the acquisition of 49.99% of a shopping centre. {url}",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Jose Auriemo (49.99%)"),
        [],
        ticker="JHSF3.SA",
        additional_records=[evidence],
    )
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_other_holder_percentage_in_same_sentence_is_not_borrowed():
    url = "https://issuer.example/holders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"Acme Holdings owns 20%, while Other Holdings owns 36.2%. {url}",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings (36.2%)"),
        [],
        ticker="TEST",
        additional_records=[evidence],
    )
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_numeric_prefix_does_not_support_a_different_percentage():
    url = "https://issuer.example/holders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"Acme Holdings owns 36.21%. {url}",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings (36.2%)"),
        [],
        ticker="TEST",
        additional_records=[evidence],
    )
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_holder_and_percentage_cannot_be_assembled_across_search_results():
    first = "https://issuer.example/holders"
    second = "https://other.example/holders"
    content = f"<result><url>{first}</url><summary>Acme Holdings is the largest shareholder.</summary></result><result><url>{second}</url><summary>Other Holdings owns 36.2%.</summary></result>"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources", content=content, urls={first, second}
    )
    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings (36.2%)"),
        [],
        ticker="TEST",
        additional_records=[evidence],
    )
    assert "Ownership Evidence Status: REJECTED" in normalized


def test_company_share_ownership_after_acquisition_retains_majority_control():
    url = "https://issuer.example/holders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"Foo Holdings owns 60% of the company’s shares following an acquisition. {url}",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Foo Holdings (60%)"), [], ticker="TEST", additional_records=[evidence]
    )
    assert "Largest Shareholder: Foo Holdings (60%)" in normalized
    assert "Control Status: CONTROLLED" in normalized


def test_acquired_asset_ownership_does_not_establish_company_share_ownership():
    url = "https://issuer.example/holders"
    evidence = make_tool_evidence_record(
        tool_name="search_foreign_sources",
        content=f"Foo Holdings owns 60% of the acquired shopping centre. {url}",
        urls={url},
    )
    normalized = normalize_foreign_language_evidence(
        _report("Foo Holdings (60%)"), [], ticker="TEST", additional_records=[evidence]
    )
    assert "Ownership Evidence Status: REJECTED" in normalized
