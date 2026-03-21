"""Structural tests that prevent future content-inspection bypasses.

These tests are grep-based and enforce the coverage invariant: every
external-content ingress path must route through INSPECTION_SERVICE before
returning content to callers.  They fail fast if someone removes inspection
calls or adds new bare ingress paths without wiring inspection.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent / "src"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _src(rel: str) -> Path:
    return SRC_ROOT / rel


# ---------------------------------------------------------------------------
# 1. No bare create_react_agent in agent node files
#    (must use TOOL_SERVICE loops instead)
# ---------------------------------------------------------------------------


def test_no_bare_create_react_agent_in_consultant_nodes():
    """Legal Counsel and Auditor must NOT use create_react_agent (bypasses TOOL_SERVICE)."""
    src = _read(_src("agents/consultant_nodes.py"))
    # Allow the import comment to remain but not any actual calls
    calls = re.findall(r"\bcreate_react_agent\s*\(", src)
    assert calls == [], (
        f"consultant_nodes.py still contains create_react_agent() calls: {calls}. "
        "These bypass TOOL_SERVICE and content inspection hooks."
    )


def test_create_react_agent_not_imported_in_consultant_nodes():
    """create_react_agent import should be removed once the loops are in place."""
    src = _read(_src("agents/consultant_nodes.py"))
    assert "from langgraph.prebuilt import create_react_agent" not in src, (
        "create_react_agent is still imported in consultant_nodes.py — "
        "remove the import once the manual loops are in place."
    )


# ---------------------------------------------------------------------------
# 2. Direct-ingress callsites have inspection calls
# ---------------------------------------------------------------------------


def test_tavily_utils_has_inspection():
    """tavily_search_with_timeout must call INSPECTION_SERVICE.check before returning."""
    src = _read(_src("tavily_utils.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "tavily_utils.py: tavily_search_with_timeout must call "
        "INSPECTION_SERVICE.check() before returning results."
    )


def test_editor_tools_fetch_reference_has_inspection():
    """fetch_reference_content must call INSPECTION_SERVICE.check on fetched content."""
    src = _read(_src("editor_tools.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "editor_tools.py: fetch_reference_content must call "
        "INSPECTION_SERVICE.check() before returning fetched content."
    )


def test_editor_tools_search_claim_uses_inspected_tavily_helper():
    """search_claim must rely on tavily_search_with_timeout for inspected results."""
    src = _read(_src("editor_tools.py"))
    assert "tavily_search_with_timeout" in src, (
        "editor_tools.py: search_claim must call tavily_search_with_timeout(), "
        "which inspects Tavily output before returning it."
    )


def test_news_stocktwits_has_inspection():
    """StockTwits message content must be inspected before inclusion in output."""
    src = _read(_src("tools/news.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "tools/news.py: get_social_media_sentiment must call "
        "INSPECTION_SERVICE.check() on StockTwits message content."
    )


def test_fetcher_tavily_gaps_has_inspection():
    """_fetch_tavily_gaps must call INSPECTION_SERVICE.check before extracting metrics."""
    src = _read(_src("data/fetcher.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "data/fetcher.py: _fetch_tavily_gaps must call "
        "INSPECTION_SERVICE.check() before passing web text to pattern_extractor."
    )


# ---------------------------------------------------------------------------
# 3. TOOL_SERVICE used at legal_counsel and auditor sources
# ---------------------------------------------------------------------------


def test_legal_counsel_uses_tool_service():
    """Legal Counsel tool calls must route through TOOL_SERVICE."""
    src = _read(_src("agents/consultant_nodes.py"))
    # Verify TOOL_SERVICE.execute is called with source="legal_counsel"
    assert 'source="legal_counsel"' in src, (
        "consultant_nodes.py: Legal Counsel tool calls must pass "
        'source="legal_counsel" to TOOL_SERVICE.execute().'
    )


def test_auditor_uses_tool_service():
    """Auditor tool calls must route through TOOL_SERVICE."""
    src = _read(_src("agents/consultant_nodes.py"))
    assert 'source="auditor"' in src, (
        "consultant_nodes.py: Auditor tool calls must pass "
        'source="auditor" to TOOL_SERVICE.execute().'
    )


# ---------------------------------------------------------------------------
# 4. ToolSource literal includes legal_counsel and auditor
# ---------------------------------------------------------------------------


def test_tool_source_includes_legal_counsel_and_auditor():
    """ToolSource TypeAlias must enumerate legal_counsel and auditor."""
    src = _read(_src("tooling/runtime.py"))
    assert (
        '"legal_counsel"' in src
    ), "tooling/runtime.py: ToolSource must include 'legal_counsel'."
    assert '"auditor"' in src, "tooling/runtime.py: ToolSource must include 'auditor'."


# ---------------------------------------------------------------------------
# 5. Config includes inspection settings
# ---------------------------------------------------------------------------


def test_config_has_inspection_settings():
    src = _read(_src("config.py"))
    assert (
        "untrusted_content_inspection_enabled" in src
    ), "config.py: missing untrusted_content_inspection_enabled setting."
    assert (
        "untrusted_content_inspection_mode" in src
    ), "config.py: missing untrusted_content_inspection_mode setting."
    assert (
        "untrusted_content_fail_policy" in src
    ), "config.py: missing untrusted_content_fail_policy setting."


# ---------------------------------------------------------------------------
# 6. Main.py wires inspection independently of logging
# ---------------------------------------------------------------------------


def test_main_has_configure_content_inspection():
    src = _read(_src("main.py"))
    assert (
        "configure_content_inspection_from_config" in src
    ), "main.py: configure_content_inspection_from_config() must be called at startup."
    # Must not be gated inside configure_cli_logging
    # Find the configure_cli_logging function body and ensure inspection is NOT there
    # (it should be in _setup_runtime, independently)
    cli_logging_match = re.search(
        r"def configure_cli_logging\(.*?\)\s*->.*?:\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert (
        cli_logging_match is not None
    ), "Could not locate configure_cli_logging() body."
    cli_logging_body = cli_logging_match.group(1)
    assert "configure_content_inspection_from_config" not in cli_logging_body, (
        "configure_content_inspection_from_config must NOT be inside "
        "configure_cli_logging — inspection is independent of logging verbosity."
    )


# ---------------------------------------------------------------------------
# 7. INSPECTION_SERVICE singleton exists
# ---------------------------------------------------------------------------


def test_inspection_service_singleton_exists():
    src = _read(_src("tooling/inspection_service.py"))
    assert (
        "INSPECTION_SERVICE = InspectionService()" in src
    ), "tooling/inspection_service.py: INSPECTION_SERVICE singleton must be defined."


# ---------------------------------------------------------------------------
# 8. InspectionEnvelope and SourceKind are importable from src.tooling
# ---------------------------------------------------------------------------


def test_inspection_types_exported_from_tooling():
    from src.tooling import (  # noqa: F401
        INSPECTION_SERVICE,
        CompositeInspector,
        ContentInspectionHook,
        ContentInspector,
        InspectionDecision,
        InspectionEnvelope,
        InspectionService,
        NullInspector,
        SourceKind,
        configure_content_inspection,
    )
