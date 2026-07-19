from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest

from src.forensic_budget import AuditorBudgetPolicy
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


def test_invalid_configured_hosts_are_ignored() -> None:
    assert documents._configured_host_suffixes(
        "localhost, https://issuer.com/path, good.example, 127.0.0.1"
    ) == ("good.example",)


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
        b"<html><body><h1>Annual report</h1><p>"
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
    inspection = type(
        "Inspection",
        (),
        {"check": AsyncMock(side_effect=lambda envelope: envelope.content_text)},
    )()
    monkeypatch.setattr(documents, "get_current_inspection_service", lambda: inspection)

    result = await documents.get_official_document.ainvoke(
        {"url": "https://www.sgx.com/report.html"}
    )

    assert '"backend": "html"' in result
    assert "Annual report" in result
    inspection.check.assert_awaited_once()
