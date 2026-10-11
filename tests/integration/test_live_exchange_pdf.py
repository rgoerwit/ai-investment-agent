"""Live canary for the bounded official-exchange PDF path (``--run-integration``)."""

from __future__ import annotations

import pytest

from src.forensic_budget import AuditorBudgetPolicy
from src.tools.official_documents import (
    _allocate,
    _download_official,
    extract_pdf_text,
)

_AGS_FY2025_ANNUAL_REPORT = (
    "https://links.sgx.com/1.0.0/corporate-announcements/"
    "4OVBTVDMML1FYN2G/850856_FY2025_Annual%20Report_SGX.pdf"
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_live_sgx_annual_report_downloads_and_extracts_audit_evidence() -> None:
    policy = AuditorBudgetPolicy.from_settings()

    allocation = _allocate()

    payload, content_type, final_url = await _download_official(
        _AGS_FY2025_ANNUAL_REPORT,
        policy.max_document_bytes,
        allocation=allocation,
    )
    extracted = await extract_pdf_text(
        payload,
        policy,
        "independent auditor,audit opinion,statement of cash flows",
        allocation=allocation,
    )

    assert payload.startswith(b"%PDF")
    assert "application/pdf" in content_type.lower()
    assert final_url.startswith("https://links.sgx.com/")
    assert extracted.pages_total >= 100
    assert 1 <= len(extracted.pages_selected) <= policy.max_selected_pages
    normalized = extracted.text.casefold()
    assert "auditor" in normalized
    assert "cash flow" in normalized
