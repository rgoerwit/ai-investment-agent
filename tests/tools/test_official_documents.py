from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest

from src.forensic_budget import AuditorBudgetPolicy
from src.runtime_services import (
    IssuerAuthorityRegistry,
    RuntimeServices,
    use_runtime_services,
)
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService
from src.tools import official_documents as documents


def _policy() -> AuditorBudgetPolicy:
    return AuditorBudgetPolicy(
        search_calls=3,
        document_calls=2,
        filing_calls=1,
        metrics_calls=1,
        news_calls=1,
        calculation_calls=2,
        max_document_bytes=1_000_000,
        max_document_pages=10,
        max_selected_pages=2,
        max_evidence_chars=1000,
        max_tool_iterations=2,
        max_llm_calls=4,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://links.sgx.com/1.0.0/corporate-announcements/a.pdf",
        "https://www.hkexnews.hk/listedco/report.pdf",
        "https://www.sec.gov/Archives/report.htm",
    ],
)
def test_official_host_allowlist_accepts_subdomains(url: str) -> None:
    assert documents._official_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://links.sgx.com/report.pdf",
        "https://sgx.com.evil.example/report.pdf",
        "https://127.0.0.1/report.pdf",
        "https://user:pass@sgx.com/report.pdf",
        "https://sgx.com:8443/report.pdf",
        "https://sgx.com:not-a-port/report.pdf",
    ],
)
def test_official_host_allowlist_rejects_unsafe_urls(url: str) -> None:
    assert not documents._official_url(url)


def test_operator_configured_issuer_host_is_narrowly_allowed() -> None:
    hosts = documents._configured_host_suffixes(
        "thehourglass.com, investor.example.co.jp"
    )
    assert documents._official_url(
        "https://www.thehourglass.com/investor/annual-report.pdf", hosts
    )
    assert documents._official_url(
        "https://reports.investor.example.co.jp/fy.pdf", hosts
    )
    assert not documents._official_url(
        "https://thehourglass.com.evil.example/report.pdf", hosts
    )


def test_authority_resolver_distinguishes_registry_and_issuer(monkeypatch) -> None:
    monkeypatch.setattr(
        documents.config,
        "auditor_official_document_hosts",
        "issuer.example",
    )
    assert (
        documents.resolve_source_authority("https://www.twse.com.tw/report.pdf")
        == "PRIMARY_REGISTRY"
    )
    assert (
        documents.resolve_source_authority("https://ir.issuer.example/report.pdf")
        == "PRIMARY_ISSUER"
    )
    assert (
        documents.resolve_source_authority("https://aggregator.example/report")
        == "SECONDARY"
    )


def test_run_scoped_structured_issuer_host_is_primary_only_in_its_run() -> None:
    registry = IssuerAuthorityRegistry()
    assert registry.register_url(
        "https://www.issuer.example/investors",
        provenance="yfinance_company_profile",
    )
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=InspectionService(),
        issuer_authority=registry,
    )
    issuer_document = "https://ir.issuer.example/results.pdf"

    assert not documents.is_official_document_url(issuer_document)
    with use_runtime_services(services):
        assert documents.is_official_document_url(issuer_document)
        assert documents.resolve_source_authority(issuer_document) == "PRIMARY_ISSUER"
        assert not documents.is_official_document_url(
            "https://issuer.example.evil.test/results.pdf"
        )
    assert not documents.is_official_document_url(issuer_document)


@pytest.mark.parametrize(
    "url",
    [
        "https://mopsov.twse.com.tw/mops/web/t05st01",
        "https://www.tpex.org.tw/web/regular_emerging/corporateInfo/",
        "https://disclosure2.edinet-fsa.go.jp/",
    ],
)
def test_exchange_registry_hosts_are_official(url: str) -> None:
    assert documents.is_official_document_url(url)


@pytest.mark.asyncio
async def test_document_tool_rejection_names_host_and_approved_list() -> None:
    result = await documents.get_official_document.ainvoke(
        {"url": "https://ir.zigexn.co.jp/report.pdf"}
    )

    assert result.startswith(
        "STATUS: INSUFFICIENT_DATA\nREASON: UNAPPROVED_DOCUMENT_HOST\n"
    )
    assert "REJECTED_HOST: ir.zigexn.co.jp" in result
    assert "APPROVED_HOSTS: " in result
    approved_line = next(
        line for line in result.splitlines() if line.startswith("APPROVED_HOSTS: ")
    )
    assert "edinet-fsa.go.jp" in approved_line
    assert "ir.zigexn.co.jp" not in approved_line


@pytest.mark.asyncio
async def test_document_tool_rejection_includes_registered_issuer_host() -> None:
    registry = IssuerAuthorityRegistry()
    assert registry.register_url(
        "https://zigexn.co.jp/ir", provenance="yfinance_company_profile"
    )
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=InspectionService(),
        issuer_authority=registry,
    )
    with use_runtime_services(services):
        result = await documents.get_official_document.ainvoke(
            {"url": "https://some-vendor-ir-host.example/report.pdf"}
        )

    approved_line = next(
        line for line in result.splitlines() if line.startswith("APPROVED_HOSTS: ")
    )
    assert "zigexn.co.jp" in approved_line
    assert "REJECTED_HOST: some-vendor-ir-host.example" in result


def test_auditor_document_byte_cap_default_covers_real_annual_reports() -> None:
    """Real JP/KR annual securities reports with embedded exhibits routinely
    exceed 15MB; the byte cap default must not abort legitimate filings
    before page-selection ever runs. Guards against an accidental revert.
    Checks the field default (not the env-loaded instance) since an
    operator .env override is expected to take precedence at runtime."""
    assert (
        type(documents.config).model_fields["auditor_max_document_bytes"].default
        >= 30_000_000
    )


def test_invalid_configured_hosts_are_ignored() -> None:
    assert documents._configured_host_suffixes(
        "localhost, https://issuer.com/path, good.example, 127.0.0.1"
    ) == ("good.example",)


@pytest.mark.asyncio
async def test_dns_check_rejects_private_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        documents.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                documents.socket.AF_INET,
                documents.socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 443),
            )
        ],
    )

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._ensure_public_hostname("https://issuer.example/report.pdf")

    assert exc_info.value.reason == "DOCUMENT_PRIVATE_ADDRESS"


@pytest.mark.asyncio
async def test_dns_check_rejects_non_string_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        documents.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                documents.socket.AF_INET,
                documents.socket.SOCK_STREAM,
                6,
                "",
                (2130706433, 443),
            )
        ],
    )

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._ensure_public_hostname("https://issuer.example/report.pdf")

    assert exc_info.value.reason == "DOCUMENT_DNS_FAILED"


def test_page_selection_prefers_financial_pages_and_honors_cap() -> None:
    pages = [
        "cover",
        "Statement of cash flows " + "100 200 " * 30,
        "notes",
        "Income statement " + "10 20 " * 30,
    ]
    selected = documents._select_pages(
        pages, ("statement of cash flows", "income statement"), 2
    )
    assert len(selected) == 2
    assert 1 in selected


def test_real_pypdf_backend_extracts_text_pdf() -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    line = "Statement of cash flows revenue assets auditor opinion " * 8
    stream.set_data(f"BT /F1 10 Tf 72 720 Td ({line}) Tj ET".encode())
    page[NameObject("/Contents")] = stream
    payload = io.BytesIO()
    writer.write(payload)

    result = documents._extract_with_pypdf(
        payload.getvalue(),
        max_pages=10,
        selected_pages=2,
        keywords=("statement of cash flows",),
    )

    assert result.backend == "pypdf"
    assert result.pages_selected == (0,)
    assert "Statement of cash flows" in result.text


@pytest.mark.asyncio
async def test_pdf_falls_back_to_pdftotext(monkeypatch) -> None:
    def broken_pypdf(*args, **kwargs):
        raise documents.DocumentExtractionError("PDF_MALFORMED")

    fallback = documents.ExtractedDocument(
        "useful filing text " * 30, "pdftotext", 3, (0, 1)
    )
    monkeypatch.setattr(documents, "_extract_with_pypdf", broken_pypdf)
    monkeypatch.setattr(
        documents, "_extract_with_pdftotext", AsyncMock(return_value=fallback)
    )

    result = await documents.extract_pdf_text(b"%PDF-bad", _policy(), "")

    assert result.backend == "pdftotext"


@pytest.mark.asyncio
async def test_scanned_pdf_returns_typed_reason_when_fallback_unavailable(
    monkeypatch,
) -> None:
    scanned = documents.ExtractedDocument("", "pypdf", 2, (0, 1))
    monkeypatch.setattr(documents, "_extract_with_pypdf", lambda *a, **k: scanned)
    monkeypatch.setattr(
        documents,
        "_extract_with_pdftotext",
        AsyncMock(
            side_effect=documents.DocumentExtractionError("PDF_PARSER_UNAVAILABLE")
        ),
    )

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents.extract_pdf_text(b"%PDF-scan", _policy(), "")

    assert exc_info.value.reason == "PDF_TEXT_UNAVAILABLE"


@pytest.mark.asyncio
async def test_both_pdf_backends_missing_returns_parser_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        documents,
        "_extract_with_pypdf",
        lambda *a, **k: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    monkeypatch.setattr(
        documents,
        "_extract_with_pdftotext",
        AsyncMock(
            side_effect=documents.DocumentExtractionError("PDF_PARSER_UNAVAILABLE")
        ),
    )

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents.extract_pdf_text(b"%PDF", _policy(), "")

    assert exc_info.value.reason == "PDF_PARSER_UNAVAILABLE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary_reason", "expected"),
    [
        ("PDF_MALFORMED", "PDF_MALFORMED"),
        ("EXTRACTION_TIMEOUT", "EXTRACTION_TIMEOUT"),
    ],
)
async def test_pdf_failure_reason_survives_failed_fallback(
    monkeypatch, primary_reason: str, expected: str
) -> None:
    monkeypatch.setattr(
        documents,
        "_extract_with_pypdf",
        lambda *a, **k: (_ for _ in ()).throw(
            documents.DocumentExtractionError(primary_reason)
        ),
    )
    monkeypatch.setattr(
        documents,
        "_extract_with_pdftotext",
        AsyncMock(side_effect=documents.DocumentExtractionError("PDF_MALFORMED")),
    )

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents.extract_pdf_text(b"%PDF", _policy(), "")

    assert exc_info.value.reason == expected


@pytest.mark.asyncio
async def test_page_limit_is_reported_as_bounded_partial(monkeypatch) -> None:
    partial = documents.ExtractedDocument(
        "statement data " * 30,
        "pypdf",
        400,
        (1, 4),
        "DOCUMENT_PAGE_LIMIT",
    )
    monkeypatch.setattr(documents, "_extract_with_pypdf", lambda *a, **k: partial)

    result = await documents.extract_pdf_text(b"%PDF", _policy(), "")

    assert result.reason == "DOCUMENT_PAGE_LIMIT"
    assert result.pages_total == 400


@pytest.mark.asyncio
async def test_document_tool_preserves_download_size_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        documents,
        "_download_official",
        AsyncMock(side_effect=documents.DocumentExtractionError("DOCUMENT_SIZE_LIMIT")),
    )

    result = await documents.get_official_document.ainvoke(
        {"url": "https://links.sgx.com/report.pdf"}
    )

    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_SIZE_LIMIT"


@pytest.mark.asyncio
async def test_document_tool_uses_html_fallback_and_inspects_text(monkeypatch) -> None:
    html = (
        b"<html><body><h1>Annual report</h1>"
        b'<a href="/results/fy-2026.pdf">FY 2026 results</a>'
        b'<a href="https://evil.example/results.pdf">External results</a><p>'
        + b"financial data " * 30
        + b"</p></body></html>"
    )
    monkeypatch.setattr(
        documents,
        "_download_official",
        AsyncMock(
            return_value=(
                html,
                "text/html; charset=utf-8",
                "https://www.sgx.com/report.html",
            )
        ),
    )
    inspection = InspectionService()
    inspection.check = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda envelope: envelope.content_text
    )
    monkeypatch.setattr(documents, "get_current_inspection_service", lambda: inspection)

    result = await documents.get_official_document.ainvoke(
        {"url": "https://www.sgx.com/report.html"}
    )

    assert '"backend": "html"' in result
    assert '"candidate_paths": ["/results/fy-2026.pdf"]' in result
    assert "evil.example" not in result
    assert "Annual report" in result
    inspection.check.assert_awaited_once()


@pytest.mark.parametrize("placement", ["before", "after"])
@pytest.mark.parametrize(
    "wrap", ["", "<table><tr><td colspan='2'>Unrelated complex table</td></tr></table>"]
)
def test_unchanged_annual_html_fixture_retains_legacy_rejection_and_context(
    wrap, placement
):
    import json
    from pathlib import Path

    from bs4 import BeautifulSoup

    from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
    from src.agents.message_utils import make_tool_evidence_record

    fixtures = Path(__file__).parents[1] / "fixtures"
    html = (fixtures / "official_statement_rows.html").read_bytes()
    extracted = documents._html_document_text(
        BeautifulSoup(
            wrap.encode() + html if placement == "before" else html + wrap.encode(),
            "html.parser",
        )
    )
    assert "Revenue\n79,068,022\n80,650,914" in extracted
    assert "For the year ended 31 December 2025" in extracted
    assert extracted.count("For the year ended 31 December 2025") == 2
    assert "Independent context" in extracted
    case = json.loads((fixtures / "latest_results_statements.json").read_text())[0]
    evidence = {**case["evidence"], "content": extracted}
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized


@pytest.mark.parametrize(
    "complex_cells",
    [
        "<td rowspan='2'>Revenue</td><td>140</td>",
        "<td colspan='2'>Revenue 140 100</td>",
        "<td><table><tr><td>Revenue 140 100</td></tr></table></td>",
    ],
)
def test_legacy_html_flattening_retains_complex_cells_without_claiming_geometry(
    complex_cells,
):
    from bs4 import BeautifulSoup

    text = documents._html_document_text(
        BeautifulSoup(
            f"<h2>Currency and period context</h2><table><tr>{complex_cells}</tr></table><p>After</p>",
            "html.parser",
        )
    )
    assert "HTML TABLE LAYOUT UNSUPPORTED" not in text
    assert text.count("Revenue") == 1
    assert "Currency and period context" in text and "After" in text


def test_legacy_html_flattening_does_not_invent_empty_cell_values():
    from bs4 import BeautifulSoup

    text = documents._html_document_text(
        BeautifulSoup(
            "<table><tr><td>Revenue</td><td></td><td>100</td></tr></table>",
            "html.parser",
        )
    )
    assert text == "Revenue\n100"


@pytest.mark.parametrize(
    "defect",
    [
        "scope",
        "currency",
        "scale",
        "period",
        "swapped_columns",
        "missing_cell",
        "wrong_metric",
    ],
)
def test_html_geometry_does_not_grant_authority_to_wrong_financial_context(defect):
    import json
    from pathlib import Path

    from bs4 import BeautifulSoup

    from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
    from src.agents.message_utils import make_tool_evidence_record

    fixtures = Path(__file__).parents[1] / "fixtures"
    html = (fixtures / "official_statement_rows.html").read_text()
    changes = {
        "scope": ("Profit attributable to owners of the Company", "Operating profit"),
        "currency": ("RMB’000", "USD’000"),
        "scale": ("RMB’000", "RMB million"),
        "period": ("31 December 2025", "30 June 2025"),
        "swapped_columns": ("<th>2025</th><th>2024</th>", "<th>2024</th><th>2025</th>"),
        "missing_cell": ("<td>79,068,022</td>", "<td></td>"),
        "wrong_metric": ("<td>Revenue</td>", "<td>Operating expenses</td>"),
    }
    case = json.loads((fixtures / "latest_results_statements.json").read_text())[0]

    def normalized(source):
        extracted = documents._html_document_text(BeautifulSoup(source, "html.parser"))
        return normalize_foreign_language_evidence(
            case["report"],
            [],
            ticker=case["ticker"],
            additional_records=[
                make_tool_evidence_record(**{**case["evidence"], "content": extracted})
            ],
        )

    # The unchanged annual fixture is already unsupported by the legacy reader.
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: UNSUPPORTED" in normalized(html)
    incorrect = normalized(html.replace(*changes[defect]))
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" not in incorrect
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in incorrect


def test_legacy_structural_reader_rejects_intervening_unrelated_text():
    import json
    from pathlib import Path

    from bs4 import BeautifulSoup

    from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
    from src.agents.message_utils import make_tool_evidence_record

    fixtures = Path(__file__).parents[1] / "fixtures"
    html = (fixtures / "official_statement_rows.html").read_text()
    # Keep both amounts but separate the candidate header from its row region.
    html = html.replace(
        "<tr><td>Revenue</td>",
        "</tbody></table><table><tr><td colspan='2'>Unrelated spans</td></tr></table><table><tbody><tr><td>Revenue</td>",
    )
    text = documents._html_document_text(BeautifulSoup(html, "html.parser"))
    case = json.loads((fixtures / "latest_results_statements.json").read_text())[0]
    evidence = {**case["evidence"], "content": text}
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" not in normalized
    assert "LATEST_RESULTS_REVENUE_GROWTH_YOY: N/A" in normalized


def test_offline_pypdf_layout_trial_preserves_controlled_column_geometry():
    """Synthetic bytes qualify the trial API; they cannot qualify real source PDFs."""
    import json
    import re
    from pathlib import Path

    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
    from src.agents.message_utils import make_tool_evidence_record

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    cells = [
        (72, 740, "CONSOLIDATED INCOME STATEMENT"),
        (72, 720, "For the year ended 31 December 2025"),
        (300, 700, "2025"),
        (420, 700, "2024"),
        (300, 680, "RMB'000"),
        (420, 680, "RMB'000"),
        (72, 660, "Revenue"),
        (300, 660, "79,068,022"),
        (420, 660, "80,650,914"),
        (72, 640, "Profit attributable to owners of the Company"),
        (300, 640, "4,500,698"),
        (420, 640, "3,734,429"),
    ]
    stream = DecodedStreamObject()
    stream.set_data(
        "\n".join(
            f"BT /F1 7 Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET" for x, y, text in cells
        ).encode()
    )
    page[NameObject("/Contents")] = stream
    payload = io.BytesIO()
    writer.write(payload)
    read_page = PdfReader(io.BytesIO(payload.getvalue())).pages[0]
    plain = read_page.extract_text()
    layout = read_page.extract_text(extraction_mode="layout")
    assert re.search(r"Revenue[ \t]+79,068,022[ \t]+80,650,914", layout)
    for number in ("79,068,022", "80,650,914", "4,500,698", "3,734,429"):
        assert plain.count(number) == layout.count(number) == 1
    case = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/latest_results_statements.json"
        ).read_text()
    )[0]
    evidence = {**case["evidence"], "content": layout}
    normalized = normalize_foreign_language_evidence(
        case["report"],
        [],
        ticker=case["ticker"],
        additional_records=[make_tool_evidence_record(**evidence)],
    )
    assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized


@pytest.mark.parametrize("corner", ["", "<td></td>", "<td rowspan='2'></td>"])
@pytest.mark.parametrize(
    "extra", ["", "<table><tr><td>Unrelated table</td></tr></table>"]
)
def test_controlled_1112_html_repeated_period_cells_keep_existing_support(
    corner, extra
):
    import json
    from pathlib import Path

    from bs4 import BeautifulSoup

    from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
    from src.agents.message_utils import make_tool_evidence_record

    fixtures = Path(__file__).parents[1] / "fixtures"
    case = next(
        row
        for row in json.loads((fixtures / "latest_results_october6.json").read_text())
        if row["ticker"] == "1112.HK"
    )
    # Reconstructed geometry from retained cells, never claimed as captured bytes.
    html = (
        extra
        + "<h2>FINANCIAL RESULTS</h2><table>"
        + f"<tr>{corner}<td>Six months ended 30 June</td><td>Six months ended 30 June</td></tr>"
        + "<tr><td>2026</td><td>2025</td><td>Change</td></tr><tr><td>RMB million</td><td>RMB million</td></tr><tr><td>(Unaudited)</td><td>(Unaudited)</td></tr><tr><td>Revenue</td><td>8,695.9</td><td>7,019.2</td><td>+23.9%</td></tr><tr><td>Net profit</td><td>610.6</td><td>71.0</td><td>+760.0%</td></tr></table>"
        + extra
    )

    def normalized(source):
        text = documents._html_document_text(BeautifulSoup(source, "html.parser"))
        saved = case["evidence_records"][0]
        record = make_tool_evidence_record(
            tool_name=saved["tool_name"],
            urls=saved["urls"],
            evidence_status=saved["evidence_status"],
            authority_by_url=saved["authority_by_url"],
            content=text,
        )
        return normalize_foreign_language_evidence(
            case["report"], [], ticker=case["ticker"], additional_records=[record]
        )

    # Surrounding unrelated prose can still prevent legacy structural binding.
    assert ("LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" in normalized(html)) is (
        not extra
    )
    for old, new in [
        ("2026</td><td>2025", "2025</td><td>2026"),
        ("RMB million", "USD million"),
        ("RMB million", "RMB thousand"),
        ("30 June", "31 December"),
        ("Net profit", "Operating profit"),
        ("8,695.9</td><td>7,019.2", "7,019.2</td><td>8,695.9"),
        ("<td>8,695.9</td>", "<td></td>"),
    ]:
        assert "LATEST_RESULTS_SOURCE_AUTHORITY: PRIMARY" not in normalized(
            html.replace(old, new)
        )
