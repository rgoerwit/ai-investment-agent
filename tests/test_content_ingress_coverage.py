"""Structural tests that prevent future content-inspection bypasses.

These tests are grep-based and enforce the coverage invariant: every
external-content ingress path must route through INSPECTION_SERVICE before
returning content to callers.  They fail fast if someone removes inspection
calls or adds new bare ingress paths without wiring inspection.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 9. New ingress paths have inspection calls
# ---------------------------------------------------------------------------


def test_memory_retrieval_has_inspection():
    """query_similar_situations must call INSPECTION_SERVICE.check on ChromaDB results."""
    src = _read(_src("memory.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "memory.py: query_similar_situations must call "
        "INSPECTION_SERVICE.check() on ChromaDB retrieved documents."
    )


def test_memory_uses_memory_retrieval_source_kind():
    """memory.py must use SourceKind.memory_retrieval for ChromaDB results."""
    src = _read(_src("memory.py"))
    assert "SourceKind.memory_retrieval" in src, (
        "memory.py: ChromaDB retrieval inspection must use "
        "SourceKind.memory_retrieval."
    )


def test_retrospective_has_inspection():
    """format_lessons_for_injection must inspect lessons text."""
    src = _read(_src("retrospective.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "retrospective.py: format_lessons_for_injection must call "
        "INSPECTION_SERVICE.check() on formatted lessons text."
    )


def test_research_foreign_search_has_inspection():
    """search_foreign_sources must inspect merged DDG+Tavily output."""
    src = _read(_src("tools/research.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "tools/research.py: search_foreign_sources must call "
        "INSPECTION_SERVICE.check() on merged output."
    )


def test_research_uses_web_search_source_kind():
    """tools/research.py must use SourceKind.web_search for foreign search."""
    src = _read(_src("tools/research.py"))
    assert "SourceKind.web_search" in src, (
        "tools/research.py: foreign search inspection must use "
        "SourceKind.web_search."
    )


def test_research_uses_official_filing_source_kind():
    """tools/research.py must use SourceKind.official_filing for filings."""
    src = _read(_src("tools/research.py"))
    assert "SourceKind.official_filing" in src, (
        "tools/research.py: filing inspection must use " "SourceKind.official_filing."
    )


def test_macro_context_has_inspection():
    """get_macro_context must inspect cached brief on re-entry."""
    src = _read(_src("macro_context.py"))
    assert "INSPECTION_SERVICE.check" in src, (
        "macro_context.py: get_macro_context must call "
        "INSPECTION_SERVICE.check() on cached macro brief."
    )


def test_macro_context_uses_cached_context_source_kind():
    """macro_context.py must use SourceKind.cached_context for cache re-entry."""
    src = _read(_src("macro_context.py"))
    assert "SourceKind.cached_context" in src, (
        "macro_context.py: cached brief inspection must use "
        "SourceKind.cached_context."
    )


# ---------------------------------------------------------------------------
# 10. Trust boundary wrapping in agent nodes
# ---------------------------------------------------------------------------


def test_analyst_nodes_uses_format_untrusted_block():
    """analyst_nodes.py must wrap extra_context with format_untrusted_block."""
    src = _read(_src("agents/analyst_nodes.py"))
    assert "format_untrusted_block" in src, (
        "agents/analyst_nodes.py: extra_context must be wrapped with "
        "format_untrusted_block()."
    )


def test_research_nodes_uses_format_untrusted_block():
    """research_nodes.py must wrap past_insights and lessons with format_untrusted_block."""
    src = _read(_src("agents/research_nodes.py"))
    assert "format_untrusted_block" in src, (
        "agents/research_nodes.py: past_insights and lessons must be wrapped "
        "with format_untrusted_block()."
    )


def test_research_nodes_wraps_memory_retrieval():
    """past_insights must use MEMORY RETRIEVAL label."""
    src = _read(_src("agents/research_nodes.py"))
    assert "MEMORY RETRIEVAL" in src, (
        "agents/research_nodes.py: past_insights must be wrapped "
        'with format_untrusted_block(..., "MEMORY RETRIEVAL").'
    )


def test_research_nodes_wraps_retrospective_lessons():
    """lessons must use RETROSPECTIVE LESSONS label."""
    src = _read(_src("agents/research_nodes.py"))
    assert "RETROSPECTIVE LESSONS" in src, (
        "agents/research_nodes.py: lessons must be wrapped "
        'with format_untrusted_block(..., "RETROSPECTIVE LESSONS").'
    )


# ---------------------------------------------------------------------------
# 11. Prompt authority — extra_context in HumanMessage, not SystemMessage
# ---------------------------------------------------------------------------


def test_analyst_nodes_extra_context_in_human_message():
    """extra_context must be in a HumanMessage, not concatenated into SystemMessage."""
    src = _read(_src("agents/analyst_nodes.py"))
    # The pattern: HumanMessage wrapping format_untrusted_block for extra_context
    assert (
        "HumanMessage" in src
    ), "agents/analyst_nodes.py must use HumanMessage for extra_context."
    tree = ast.parse(src)
    core_assignment = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "core_system_instruction"
                for target in node.targets
            )
        ),
        None,
    )
    assert (
        core_assignment is not None
    ), "agents/analyst_nodes.py: could not locate core_system_instruction assignment."
    referenced_names = {
        node.id
        for node in ast.walk(core_assignment.value)
        if isinstance(node, ast.Name)
    }
    assert "extra_context" not in referenced_names, (
        "agents/analyst_nodes.py: extra_context must NOT be inside "
        "core_system_instruction (SystemMessage). It belongs in a HumanMessage."
    )


# ---------------------------------------------------------------------------
# 12. SourceKind includes new values
# ---------------------------------------------------------------------------


def test_source_kind_has_memory_retrieval():
    from src.tooling.inspector import SourceKind

    assert hasattr(SourceKind, "memory_retrieval")
    assert SourceKind.memory_retrieval.value == "memory_retrieval"


def test_source_kind_has_cached_context():
    from src.tooling.inspector import SourceKind

    assert hasattr(SourceKind, "cached_context")
    assert SourceKind.cached_context.value == "cached_context"


# ---------------------------------------------------------------------------
# 13. New tooling exports
# ---------------------------------------------------------------------------


def test_new_inspectors_exported_from_tooling():
    from src.tooling import (  # noqa: F401
        EscalatingInspector,
        HeuristicInspector,
        LLMJudgeInspector,
        ToolArgumentPolicyHook,
        format_untrusted_block,
    )


# ---------------------------------------------------------------------------
# 14. Backend wiring in main.py
# ---------------------------------------------------------------------------


def test_main_supports_python_backend():
    """main.py must support the 'python' backend (HeuristicInspector)."""
    src = _read(_src("main.py"))
    assert "HeuristicInspector" in src, (
        "main.py: configure_content_inspection_from_config must support "
        "'python' backend with HeuristicInspector."
    )


def test_main_supports_composite_backend():
    """main.py must support the 'composite' backend (EscalatingInspector)."""
    src = _read(_src("main.py"))
    assert "EscalatingInspector" in src, (
        "main.py: configure_content_inspection_from_config must support "
        "'composite' backend with EscalatingInspector."
    )


def test_main_registers_argument_policy_hook():
    """main.py must register ToolArgumentPolicyHook."""
    src = _read(_src("main.py"))
    assert "ToolArgumentPolicyHook" in src, (
        "main.py: configure_content_inspection_from_config must register "
        "ToolArgumentPolicyHook on TOOL_SERVICE."
    )


import ast
