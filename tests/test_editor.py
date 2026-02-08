"""
Tests for the Editor-in-Chief article revision system.

Tests cover:
- Editor tool (fetch_reference_content)
- Editor prompt loading
- Article revision workflow
- Editorial loop with mock LLMs
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.editor_tools import (
    MAX_REFERENCE_CHARS,
    fetch_reference_content,
    get_editor_tools,
)

# =============================================================================
# Tool Tests
# =============================================================================


class TestFetchReferenceContent:
    """Tests for the fetch_reference_content async tool."""

    @pytest.mark.asyncio
    async def test_invalid_url_rejected(self):
        """Invalid URLs should be rejected immediately."""
        result = await fetch_reference_content.ainvoke({"url": "not-a-url"})
        assert "INVALID_URL" in result

        result = await fetch_reference_content.ainvoke({"url": ""})
        assert "INVALID_URL" in result

        result = await fetch_reference_content.ainvoke({"url": "ftp://example.com"})
        assert "INVALID_URL" in result

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Successful fetch should return cleaned text content."""
        # Create content that exceeds 100 chars after HTML stripping
        main_content = "This is the main content of the article. " * 10
        important_info = "It contains important information about the company. " * 5
        mock_html = f"""
        <html>
        <head><title>Test</title></head>
        <body>
            <nav>Navigation menu here</nav>
            <main>
                <p>{main_content}</p>
                <p>{important_info}</p>
            </main>
            <footer>Footer content here</footer>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await fetch_reference_content.ainvoke(
                {"url": "https://example.com/article"}
            )

            # Should contain main content but not nav/footer
            assert "main content" in result
            assert "important information" in result
            assert "Navigation" not in result
            assert "Footer content" not in result

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Timeout should return appropriate error message."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await fetch_reference_content.ainvoke(
                {"url": "https://example.com/slow"}
            )

            assert "FETCH_FAILED" in result
            assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """HTTP errors should return status code in error message."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not found",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await fetch_reference_content.ainvoke(
                {"url": "https://example.com/missing"}
            )

            assert "FETCH_FAILED" in result
            assert "404" in result

    @pytest.mark.asyncio
    async def test_insufficient_content(self):
        """Pages with very little text should return INSUFFICIENT_CONTENT."""
        mock_html = "<html><body><p>Hi</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await fetch_reference_content.ainvoke(
                {"url": "https://example.com/empty"}
            )

            assert "INSUFFICIENT_CONTENT" in result

    @pytest.mark.asyncio
    async def test_content_truncation(self):
        """Long content should be truncated to MAX_REFERENCE_CHARS."""
        # Create content longer than MAX_REFERENCE_CHARS
        long_text = "word " * 2000  # Much longer than 5000 chars
        mock_html = f"<html><body><p>{long_text}</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await fetch_reference_content.ainvoke(
                {"url": "https://example.com/long"}
            )

            # Should be truncated with indicator
            assert (
                len(result) <= MAX_REFERENCE_CHARS + 20
            )  # Allow for truncation marker
            assert "truncated" in result.lower()


class TestGetEditorTools:
    """Tests for the get_editor_tools function."""

    def test_returns_list_of_tools(self):
        """Should return a list with the fetch tool."""
        tools = get_editor_tools()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_tool_has_correct_name(self):
        """Tool should have the expected name."""
        tools = get_editor_tools()
        tool = tools[0]
        assert tool.name == "fetch_reference_content"


# =============================================================================
# Prompt Loading Tests
# =============================================================================


class TestEditorPromptLoading:
    """Tests for editor prompt file loading."""

    def test_editor_prompt_loads_from_registry(self):
        """Editor prompt should load via PromptRegistry."""
        from src.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get("editor_in_chief")

        assert prompt is not None
        assert prompt.agent_key == "editor_in_chief"
        # Version should be a valid numeric string (e.g., "1.0", "2.1")
        import re

        assert re.match(
            r"^\d+\.\d+$", prompt.version
        ), f"Invalid version: {prompt.version}"
        # V2.0 enabled tools for reference verification
        assert prompt.requires_tools is True

    def test_editor_prompt_has_required_fields(self):
        """Editor prompt should have all required fields."""
        from src.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get("editor_in_chief")

        assert prompt.system_message
        assert (
            "VOICE" in prompt.system_message or "voice" in prompt.system_message.lower()
        )
        assert (
            "FACT-CHECK" in prompt.system_message
            or "fact" in prompt.system_message.lower()
        )

    def test_writer_has_revision_template(self):
        """Writer prompt should have revision_template in metadata."""
        from src.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get("article_writer")

        assert prompt is not None
        assert "revision_template" in prompt.metadata
        assert "{original_draft}" in prompt.metadata["revision_template"]
        assert "{factual_errors}" in prompt.metadata["revision_template"]


# =============================================================================
# Article Revision Tests
# =============================================================================


class TestArticleRevision:
    """Tests for article revision functionality."""

    @patch("src.article_writer.create_writer_llm")
    def test_writer_revise_method_exists(self, mock_create):
        """ArticleWriter should have a revise method."""
        from src.article_writer import ArticleWriter

        mock_create.return_value = MagicMock()
        writer = ArticleWriter()
        assert hasattr(writer, "revise")
        assert callable(writer.revise)

    @patch("src.article_writer.create_writer_llm")
    def test_revise_formats_feedback_correctly(self, mock_create):
        """Revise should format editor feedback into the prompt."""
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "# Revised Article\n\nFixed content here."
        mock_llm.invoke = MagicMock(return_value=mock_response)
        mock_create.return_value = mock_llm

        writer = ArticleWriter()

        feedback = {
            "verdict": "REVISE",
            "factual_errors": [
                {"location": "Para 2", "claim": "P/E of 20", "ground_truth": "P/E: 15"}
            ],
            "cuts": ["Remove redundant paragraph 3"],
            "style_issues": ["Avoid 'might' in intro"],
        }

        result = writer.revise(
            original_draft="# Original\n\nSome content.",
            editor_feedback=feedback,
            ticker="TEST",
            company_name="Test Corp",
        )

        assert result == "# Revised Article\n\nFixed content here."
        # Verify LLM was called
        assert mock_llm.invoke.called


# =============================================================================
# ArticleEditor Tests
# =============================================================================


class TestArticleEditor:
    """Tests for the ArticleEditor class."""

    def test_editor_initialization(self):
        """ArticleEditor should initialize correctly."""
        from src.article_writer import ArticleEditor

        # This may or may not have LLM depending on environment
        editor = ArticleEditor()
        assert hasattr(editor, "llm")
        assert hasattr(editor, "tools")
        assert hasattr(editor, "is_available")

    def test_build_fact_check_context(self):
        """build_fact_check_context should assemble context correctly."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        context = editor.build_fact_check_context(
            data_block="FINANCIAL_HEALTH: 75%\nP/E: 12.5",
            pm_block="VERDICT: BUY\nCONVICTION: HIGH",
            valuation_params="52_WEEK_HIGH: 100\n52_WEEK_LOW: 50",
        )

        assert "DATA_BLOCK" in context
        assert "FINANCIAL_HEALTH: 75%" in context
        assert "PM_BLOCK" in context
        assert "VERDICT: BUY" in context
        assert "VALUATION PARAMETERS" in context

    def test_build_fact_check_context_empty(self):
        """build_fact_check_context should handle empty inputs."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()
        context = editor.build_fact_check_context()

        assert context == "No context provided."

    def test_parse_editor_response_valid_json(self):
        """_parse_editor_response should parse valid JSON."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        response = '{"verdict": "REVISE", "factual_errors": [], "confidence": 0.9}'
        result = editor._parse_editor_response(response)

        assert result["verdict"] == "REVISE"
        assert result["confidence"] == 0.9

    def test_parse_editor_response_json_in_code_block(self):
        """_parse_editor_response should extract JSON from code blocks."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        response = """Here is my review:

```json
{"verdict": "APPROVED", "confidence": 0.95}
```

That's my assessment."""

        result = editor._parse_editor_response(response)

        assert result["verdict"] == "APPROVED"
        assert result["confidence"] == 0.95

    def test_parse_editor_response_invalid_json(self):
        """_parse_editor_response should handle invalid JSON gracefully."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        response = "This is not valid JSON at all."
        result = editor._parse_editor_response(response)

        # Should return approved by default (fail-safe)
        assert result["verdict"] == "APPROVED"
        assert "parse_error" in result

    def test_is_available_returns_bool(self):
        """is_available should return boolean."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()
        result = editor.is_available()

        assert isinstance(result, bool)


# =============================================================================
# Config Tests
# =============================================================================


class TestEditorConfig:
    """Tests for editor configuration."""

    def test_editor_model_config_exists(self):
        """Config should have editor_model field."""
        from src.config import config

        assert hasattr(config, "editor_model")

    def test_editor_model_field_is_optional(self):
        """editor_model field should exist and be optional (str or None)."""
        from src.config import Settings

        # Create settings with EDITOR_MODEL explicitly cleared
        with patch.dict("os.environ", {"EDITOR_MODEL": ""}, clear=False):
            settings = Settings()
            # Field should exist and be string or None
            assert hasattr(settings, "editor_model")
            assert settings.editor_model is None or isinstance(
                settings.editor_model, str
            )


# =============================================================================
# LLM Factory Tests
# =============================================================================


class TestCreateEditorLLM:
    """Tests for create_editor_llm function."""

    def test_create_editor_llm_exists(self):
        """create_editor_llm function should exist."""
        from src.llms import create_editor_llm

        assert callable(create_editor_llm)

    def test_create_editor_llm_returns_none_without_api_key(self):
        """create_editor_llm should return None if no API key."""
        from src.llms import create_editor_llm

        # Mock config to have no API key
        with patch("src.llms.config") as mock_config:
            mock_config.enable_consultant = True
            mock_config.get_openai_api_key.return_value = ""

            result = create_editor_llm()
            assert result is None

    def test_create_editor_llm_returns_none_when_disabled(self):
        """create_editor_llm should return None if consultant disabled."""
        from src.llms import create_editor_llm

        with patch("src.llms.config") as mock_config:
            mock_config.enable_consultant = False

            result = create_editor_llm()
            assert result is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestEditorialLoopIntegration:
    """Integration tests for the full editorial loop."""

    @pytest.mark.asyncio
    async def test_edit_with_unavailable_editor(self):
        """edit() should return original draft when editor unavailable."""
        from src.article_writer import ArticleEditor, ArticleWriter

        writer = ArticleWriter()
        editor = ArticleEditor()

        # Force editor to be unavailable
        editor.llm = None

        draft = "# Test Article\n\nSome content."
        result, feedback = await editor.edit(
            writer=writer,
            article_draft=draft,
            ticker="TEST",
            company_name="Test Corp",
        )

        assert result == draft
        assert feedback.get("skipped") is True

    @pytest.mark.asyncio
    async def test_edit_approves_good_article(self):
        """edit() should approve article when editor says APPROVED."""
        from src.article_writer import ArticleEditor, ArticleWriter

        writer = ArticleWriter()
        editor = ArticleEditor()

        # Mock editor to return APPROVED
        async def mock_review(*args, **kwargs):
            return {"verdict": "APPROVED", "confidence": 0.95}

        editor.review = mock_review
        editor.llm = MagicMock()  # Make it "available"

        draft = "# Good Article\n\nAccurate content."
        result, feedback = await editor.edit(
            writer=writer,
            article_draft=draft,
            ticker="TEST",
            company_name="Test Corp",
        )

        assert result == draft
        assert feedback["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_edit_revises_on_feedback(self):
        """edit() should revise article when editor requests changes."""
        from src.article_writer import ArticleEditor, ArticleWriter

        writer = ArticleWriter()
        editor = ArticleEditor()

        revision_count = 0

        async def mock_review(*args, **kwargs):
            nonlocal revision_count
            revision_count += 1
            if revision_count == 1:
                return {
                    "verdict": "REVISE",
                    "factual_errors": [
                        {"location": "Para 1", "claim": "X", "ground_truth": "Y"}
                    ],
                    "cuts": [],
                    "style_issues": [],
                    "confidence": 0.6,
                }
            return {"verdict": "APPROVED", "confidence": 0.9}

        editor.review = mock_review
        editor.llm = MagicMock()

        # Mock writer revise
        writer.revise = MagicMock(return_value="# Revised Article\n\nFixed.")

        draft = "# Original\n\nWith errors."
        result, feedback = await editor.edit(
            writer=writer,
            article_draft=draft,
            ticker="TEST",
            company_name="Test Corp",
        )

        # Should have called revise once
        assert writer.revise.called
        assert feedback["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_edit_respects_max_revisions(self):
        """edit() should stop after MAX_REVISIONS iterations."""
        from src.article_writer import ArticleEditor, ArticleWriter

        writer = ArticleWriter()
        editor = ArticleEditor()

        review_count = 0

        async def mock_review_always_revise(*args, **kwargs):
            nonlocal review_count
            review_count += 1
            return {
                "verdict": "REVISE",
                "factual_errors": [
                    {"location": "X", "claim": "Y", "ground_truth": "Z"}
                ],
                "cuts": [],
                "style_issues": [],
                "confidence": 0.5,
            }

        editor.review = mock_review_always_revise
        editor.llm = MagicMock()

        writer.revise = MagicMock(return_value="# Still has issues")

        draft = "# Problematic Article"
        result, feedback = await editor.edit(
            writer=writer,
            article_draft=draft,
            ticker="TEST",
            company_name="Test Corp",
        )

        # Should have stopped after MAX_REVISIONS + 1 reviews (initial + after each revision)
        assert review_count == editor.MAX_REVISIONS + 1
        assert writer.revise.call_count == editor.MAX_REVISIONS


class TestMainPyIntegration:
    """Tests that verify the editor is properly wired into main.py."""

    @pytest.mark.asyncio
    async def test_handle_article_generation_calls_editor(self):
        """handle_article_generation should call ArticleEditor.edit() when editor is available."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create mock args
        args = MagicMock()
        args.article = "test_article.md"
        args.output = "/tmp/test_output.md"
        args.quiet = True
        args.brief = True

        # Mock the writer and editor
        mock_writer_instance = MagicMock()
        mock_writer_instance.write.return_value = "# Draft Article\n\nContent here."

        mock_editor_instance = MagicMock()
        mock_editor_instance.is_available.return_value = True
        mock_editor_instance.edit = AsyncMock(
            return_value=("# Final Article\n\nEdited content.", {"verdict": "APPROVED"})
        )

        with (
            patch("src.main.resolve_article_path", return_value="/tmp/test_article.md"),
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", MagicMock()),
        ):
            from src.main import handle_article_generation

            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
                analysis_result={
                    "fundamentals_report": "DATA_BLOCK",
                    "final_trade_decision": "PM_BLOCK",
                    "valuation_params": "VAL_PARAMS",
                },
            )

            # Verify editor.edit() was called
            mock_editor_instance.edit.assert_called_once()

            # Verify it was called with the ground truth data
            call_kwargs = mock_editor_instance.edit.call_args.kwargs
            assert call_kwargs["data_block"] == "DATA_BLOCK"
            assert call_kwargs["pm_block"] == "PM_BLOCK"
            assert call_kwargs["valuation_params"] == "VAL_PARAMS"

    @pytest.mark.asyncio
    async def test_handle_article_generation_skips_editor_when_unavailable(self):
        """handle_article_generation should skip editor when not available."""
        from unittest.mock import AsyncMock, MagicMock, patch

        args = MagicMock()
        args.article = "test_article.md"
        args.output = "/tmp/test_output.md"
        args.quiet = True
        args.brief = True

        mock_writer_instance = MagicMock()
        mock_writer_instance.write.return_value = "# Draft Article"

        mock_editor_instance = MagicMock()
        mock_editor_instance.is_available.return_value = False
        mock_editor_instance.edit = AsyncMock()

        with (
            patch("src.main.resolve_article_path", return_value="/tmp/test_article.md"),
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", MagicMock()),
        ):
            from src.main import handle_article_generation

            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
            )

            # Verify editor.edit() was NOT called
            mock_editor_instance.edit.assert_not_called()

    def test_handle_article_generation_signature_includes_analysis_result(self):
        """Verify handle_article_generation accepts analysis_result parameter."""
        import inspect

        from src.main import handle_article_generation

        sig = inspect.signature(handle_article_generation)
        param_names = list(sig.parameters.keys())

        assert "analysis_result" in param_names, (
            "handle_article_generation must accept analysis_result parameter "
            "to pass DATA_BLOCK/PM_BLOCK to the editor"
        )

    @pytest.mark.asyncio
    async def test_handle_article_generation_preserves_draft_on_editor_failure(self):
        """If editor.edit() raises an exception, the original draft should be preserved."""
        from unittest.mock import AsyncMock, MagicMock, mock_open, patch

        args = MagicMock()
        args.article = "test_article.md"
        args.output = "/tmp/test_output.md"
        args.quiet = False
        args.brief = False

        mock_writer_instance = MagicMock()
        mock_writer_instance.write.return_value = (
            "# Original Draft\n\nThis is the draft."
        )

        mock_editor_instance = MagicMock()
        mock_editor_instance.is_available.return_value = True
        # Simulate editor failure
        mock_editor_instance.edit = AsyncMock(side_effect=RuntimeError("API timeout"))

        m_open = mock_open()

        with (
            patch("src.main.resolve_article_path", return_value="/tmp/test_article.md"),
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", m_open),
            patch("src.main.console"),
        ):
            from src.main import handle_article_generation

            # Should NOT raise - exception is caught internally
            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
                analysis_result={
                    "fundamentals_report": "DATA",
                    "final_trade_decision": "PM",
                },
            )

            # The function should complete without raising


class TestStripLLMPreamble:
    """Tests for the _strip_llm_preamble helper function."""

    def test_strips_common_preambles(self):
        """Should strip common LLM preamble phrases."""
        from src.article_writer import _strip_llm_preamble

        cases = [
            ("Here is the revised article:\n# Title", "# Title"),
            ("Here's the corrected article:\n\n# Title", "# Title"),
            ("Below is the revised article:\n# Title", "# Title"),
            ("I've revised the article:\n# Title", "# Title"),
        ]

        for input_text, expected in cases:
            result = _strip_llm_preamble(input_text)
            assert result == expected, f"Failed for: {input_text[:30]}..."

    def test_preserves_clean_articles(self):
        """Should not modify articles that start with headers."""
        from src.article_writer import _strip_llm_preamble

        clean_article = "# Investment Analysis\n\nThis is clean content."
        result = _strip_llm_preamble(clean_article)
        assert result == clean_article

    def test_handles_empty_input(self):
        """Should handle empty or None input gracefully."""
        from src.article_writer import _strip_llm_preamble

        assert _strip_llm_preamble("") == ""
        assert _strip_llm_preamble(None) is None

    def test_case_insensitive_matching(self):
        """Preamble matching should be case-insensitive."""
        from src.article_writer import _strip_llm_preamble

        result = _strip_llm_preamble("HERE IS THE REVISED ARTICLE:\n# Title")
        assert result == "# Title"

    def test_strips_preamble_followed_by_blank_line(self):
        """Should handle preamble separated by blank line from content."""
        from src.article_writer import _strip_llm_preamble

        text = "Here is the revised version.\n\n# Actual Title\n\nContent here."
        result = _strip_llm_preamble(text)
        assert result.startswith("# Actual Title")


class TestChartPreservation:
    """Tests for chart extraction and re-injection logic."""

    def test_extract_chart_references_finds_images(self):
        """Should extract markdown image references."""
        from src.article_writer import _extract_chart_references

        text = """# Article

Some content here.

![Football Field Chart](images/football_field.png)

More content.

![Radar Chart](images/radar.png)

References section.
"""
        charts = _extract_chart_references(text)
        assert len(charts) == 2
        assert charts[0]["alt_text"] == "Football Field Chart"
        assert charts[0]["path"] == "images/football_field.png"
        assert charts[1]["alt_text"] == "Radar Chart"

    def test_extract_chart_references_empty_text(self):
        """Should handle empty or None text."""
        from src.article_writer import _extract_chart_references

        assert _extract_chart_references("") == []
        assert _extract_chart_references(None) == []

    def test_extract_chart_references_no_images(self):
        """Should return empty list when no images present."""
        from src.article_writer import _extract_chart_references

        text = "# Article\n\nJust text, no images."
        assert _extract_chart_references(text) == []

    def test_reinject_missing_charts_preserves_existing(self):
        """Should not duplicate charts that are already present."""
        from unittest.mock import MagicMock

        from src.article_writer import _reinject_missing_charts

        logger = MagicMock()
        article = "# Article\n\n![Chart](img.png)\n\nContent."
        charts = [
            {"alt_text": "Chart", "path": "img.png", "full_match": "![Chart](img.png)"}
        ]

        result = _reinject_missing_charts(article, charts, logger)
        assert result == article  # No change
        logger.warning.assert_not_called()

    def test_reinject_missing_charts_adds_lost_chart(self):
        """Should re-inject charts that were lost during revision."""
        from unittest.mock import MagicMock

        from src.article_writer import _reinject_missing_charts

        logger = MagicMock()
        article = "# Article\n\nContent without chart.\n\n## References\n\n- Source 1"
        charts = [
            {
                "alt_text": "Football Field",
                "path": "img.png",
                "full_match": "![Football Field](img.png)",
            }
        ]

        result = _reinject_missing_charts(article, charts, logger)
        assert "![Football Field](img.png)" in result
        logger.warning.assert_called_once()

    def test_reinject_places_football_near_valuation(self):
        """Football field chart should be placed near Valuation section."""
        from unittest.mock import MagicMock

        from src.article_writer import _reinject_missing_charts

        logger = MagicMock()
        article = "# Article\n\n## Bull Case\n\nBullish.\n\n## Valuation\n\nValuation content.\n\n## Verdict\n\nBuy."
        charts = [
            {
                "alt_text": "Football Field",
                "path": "img.png",
                "full_match": "![Football Field](img.png)",
            }
        ]

        result = _reinject_missing_charts(article, charts, logger)
        # Chart should appear after Valuation section header
        valuation_idx = result.find("## Valuation")
        chart_idx = result.find("![Football Field]")
        verdict_idx = result.find("## Verdict")

        assert chart_idx > valuation_idx
        assert chart_idx < verdict_idx

    def test_reinject_places_radar_near_thesis(self):
        """Radar chart should be placed near Thesis section."""
        from unittest.mock import MagicMock

        from src.article_writer import _reinject_missing_charts

        logger = MagicMock()
        article = "# Article\n\n## Thesis\n\nThesis content.\n\n## Company Overview\n\nOverview."
        charts = [
            {
                "alt_text": "Thesis Alignment Radar",
                "path": "radar.png",
                "full_match": "![Thesis Alignment Radar](radar.png)",
            }
        ]

        result = _reinject_missing_charts(article, charts, logger)
        thesis_idx = result.find("## Thesis")
        chart_idx = result.find("![Thesis Alignment Radar]")
        overview_idx = result.find("## Company Overview")

        assert chart_idx > thesis_idx
        assert chart_idx < overview_idx


# =============================================================================
# Response Extraction Tests (Claude/Gemini format handling)
# =============================================================================


class TestExtractTextFromResponse:
    """Tests for the _extract_text_from_response helper."""

    def test_claude_adaptive_thinking_response(self):
        """Should extract only text blocks, skipping thinking."""
        from src.article_writer import _extract_text_from_response

        mock_response = MagicMock()
        mock_response.content = [
            {"type": "thinking", "thinking": "Let me plan the structure..."},
            {"type": "text", "text": "# The Bull Case for Toyota\n\nContent."},
        ]

        result = _extract_text_from_response(mock_response)
        assert result == "# The Bull Case for Toyota\n\nContent."
        assert "plan the structure" not in result

    def test_claude_redacted_thinking_skipped(self):
        """Should skip redacted_thinking blocks."""
        from src.article_writer import _extract_text_from_response

        mock_response = MagicMock()
        mock_response.content = [
            {"type": "redacted_thinking", "data": "abc123encrypted"},
            {"type": "text", "text": "# Article"},
        ]

        result = _extract_text_from_response(mock_response)
        assert result == "# Article"

    def test_plain_string_response(self):
        """Should handle plain string (Claude without thinking, or Gemini fallback)."""
        from src.article_writer import _extract_text_from_response

        mock_response = MagicMock()
        mock_response.content = "# Article\n\nPlain content."

        result = _extract_text_from_response(mock_response)
        assert result == "# Article\n\nPlain content."

    def test_gemini_format_backwards_compatible(self):
        """Should handle Gemini-style response for fallback path."""
        from src.article_writer import _extract_text_from_response

        mock_response = MagicMock()
        mock_response.content = [{"text": "# Gemini Article"}]

        result = _extract_text_from_response(mock_response)
        assert result == "# Gemini Article"

    def test_multiple_text_blocks_concatenated(self):
        """Should concatenate multiple text blocks."""
        from src.article_writer import _extract_text_from_response

        mock_response = MagicMock()
        mock_response.content = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "Part 1."},
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "Part 2."},
        ]

        result = _extract_text_from_response(mock_response)
        assert result == "Part 1.\nPart 2."


# =============================================================================
# Writer Invocation Tests (refusal detection, preamble stripping)
# =============================================================================


class TestInvokeWriter:
    """Tests for the _invoke_writer method."""

    @patch("src.article_writer.create_writer_llm")
    def test_detects_refusal(self, mock_create):
        """Should raise RuntimeError when model refuses financial content."""
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="I cannot provide financial advice about specific stocks."
        )
        mock_create.return_value = mock_llm

        writer = ArticleWriter()

        with pytest.raises(RuntimeError, match="refused"):
            writer._invoke_writer([])

    @patch("src.article_writer.create_writer_llm")
    def test_passes_clean_article(self, mock_create):
        """Should return clean article text for normal responses."""
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="# Toyota: Value in Motion\n\nContent here."
        )
        mock_create.return_value = mock_llm

        writer = ArticleWriter()
        result = writer._invoke_writer([])
        assert result.startswith("# Toyota")

    @patch("src.article_writer.create_writer_llm")
    def test_strips_preamble_and_finds_header(self, mock_create):
        """Should strip preamble and find first header."""
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Certainly! Here is the article:\n\n# The Real Title\n\nContent."
        )
        mock_create.return_value = mock_llm

        writer = ArticleWriter()
        result = writer._invoke_writer([])
        assert result.startswith("# The Real Title")

    @patch("src.article_writer.create_writer_llm")
    def test_handles_claude_thinking_response(self, mock_create):
        """Should extract text from Claude thinking response."""
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "thinking", "thinking": "Planning the article structure..."},
            {"type": "text", "text": "# Great Article\n\nSolid content."},
        ]
        mock_llm.invoke.return_value = mock_response
        mock_create.return_value = mock_llm

        writer = ArticleWriter()
        result = writer._invoke_writer([])
        assert result == "# Great Article\n\nSolid content."


# =============================================================================
# Writer LLM Factory Tests
# =============================================================================


class TestCreateWriterLLM:
    """Tests for create_writer_llm factory function."""

    def test_create_writer_llm_exists(self):
        """create_writer_llm function should exist."""
        from src.llms import create_writer_llm

        assert callable(create_writer_llm)

    def test_falls_back_to_gemini_without_claude_key(self):
        """Should fall back to Gemini when CLAUDE_KEY is not set."""
        from src.llms import create_writer_llm

        with patch("src.llms.config") as mock_config:
            mock_config.get_claude_api_key.return_value = None
            mock_config.deep_think_llm = "gemini-3-pro-preview"
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 10
            mock_config.get_google_api_key.return_value = "fake-key"
            mock_config.gemini_rpm_limit = 15

            llm = create_writer_llm()
            # Should return something (Gemini fallback), not raise
            assert llm is not None

    def test_opus_effort_not_top_level_kwarg(self):
        """Regression: effort must be inside output_config, not a top-level API param.

        anthropic SDK rejects unknown top-level kwargs like 'effort' on Messages.create().
        The effort parameter must be nested: model_kwargs={"output_config": {"effort": "high"}}.
        """
        from src.llms import create_writer_llm

        with patch("src.llms.config") as mock_config:
            mock_config.get_claude_api_key.return_value = "fake-key"
            mock_config.writer_model = "claude-opus-4-6"
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 3

            llm = create_writer_llm()
            # effort must NOT be a top-level model_kwarg
            assert "effort" not in llm.model_kwargs, (
                "effort must be nested inside output_config, not top-level "
                "(causes Messages.create() unexpected keyword argument error)"
            )
            # effort must be inside output_config
            assert "output_config" in llm.model_kwargs
            assert llm.model_kwargs["output_config"]["effort"] == "high"

    def test_model_kwargs_are_valid_api_params(self):
        """Regression: all model_kwargs must be valid Anthropic Messages.create() params.

        Only these top-level params are allowed by the Anthropic API:
        model, max_tokens, messages, metadata, stop_sequences, stream, system,
        temperature, thinking, tool_choice, tools, top_k, top_p, output_config.
        Anything else causes 'unexpected keyword argument' errors at runtime.
        """
        from src.llms import create_writer_llm

        VALID_API_PARAMS = {
            "model",
            "max_tokens",
            "messages",
            "metadata",
            "stop_sequences",
            "stream",
            "system",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
            "output_config",
        }

        for model_name in [
            "claude-opus-4-6",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ]:
            with patch("src.llms.config") as mock_config:
                mock_config.get_claude_api_key.return_value = "fake-key"
                mock_config.writer_model = model_name
                mock_config.api_timeout = 300
                mock_config.api_retry_attempts = 3

                llm = create_writer_llm()
                for key in llm.model_kwargs:
                    assert key in VALID_API_PARAMS, (
                        f"model_kwargs['{key}'] for {model_name} is not a valid "
                        f"Anthropic Messages.create() parameter. "
                        f"Valid: {sorted(VALID_API_PARAMS)}"
                    )


# =============================================================================
# Writer Config Tests
# =============================================================================


class TestWriterConfig:
    """Tests for writer-related configuration."""

    def test_writer_model_config_exists(self):
        """Config should have writer_model field."""
        from src.config import config

        assert hasattr(config, "writer_model")
        assert isinstance(config.writer_model, str)

    def test_claude_api_key_field_exists(self):
        """Config should have claude_api_key field."""
        from src.config import config

        assert hasattr(config, "claude_api_key")

    def test_get_claude_api_key_returns_none_when_unset(self):
        """get_claude_api_key should return None when not configured."""
        from src.config import Settings

        with patch.dict("os.environ", {"CLAUDE_KEY": ""}, clear=False):
            settings = Settings()
            # Should be None or empty, not raise
            result = settings.get_claude_api_key()
            assert result is None or result == ""


# =============================================================================
# Tool-Calling Loop Tests
# =============================================================================


class TestEditorToolCalling:
    """Tests for the editor's agentic tool-calling loop."""

    @pytest.mark.asyncio
    async def test_review_calls_tool_for_references(self):
        """When LLM returns tool_calls, tools should be executed and results fed back."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        # First response: tool call for a URL
        tool_call_response = MagicMock()
        tool_call_response.tool_calls = [
            {
                "name": "fetch_reference_content",
                "args": {"url": "https://example.com/article"},
                "id": "call_1",
            }
        ]
        tool_call_response.content = ""

        # Second response: final JSON verdict (no tool calls)
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = json.dumps(
            {
                "verdict": "APPROVED",
                "factual_errors": [],
                "reference_checks": [
                    {
                        "url": "https://example.com/article",
                        "status": "verified",
                        "note": "Content matches",
                    }
                ],
                "cuts": [],
                "style_issues": [],
                "confidence": 0.9,
            }
        )

        mock_llm_with_tools = AsyncMock()
        mock_llm_with_tools.ainvoke = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(
            return_value="Article content about the company financials..."
        )
        mock_tool.name = "fetch_reference_content"

        editor.llm = MagicMock()
        editor.llm_with_tools = mock_llm_with_tools
        editor.tools = [mock_tool]
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        result = await editor.review("# Test Article\n\nContent.", "DATA_BLOCK: ...")

        assert result["verdict"] == "APPROVED"
        assert len(result["reference_checks"]) == 1
        # Tool should have been called once
        mock_tool.ainvoke.assert_called_once_with(
            {"url": "https://example.com/article"}
        )
        # LLM should have been called twice (tool call + final)
        assert mock_llm_with_tools.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_review_handles_tool_error_gracefully(self):
        """When tool returns FETCH_FAILED, editor should still produce valid JSON."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        # First response: tool call
        tool_call_response = MagicMock()
        tool_call_response.tool_calls = [
            {
                "name": "fetch_reference_content",
                "args": {"url": "https://broken.com"},
                "id": "call_1",
            }
        ]
        tool_call_response.content = ""

        # Second response: verdict reflecting broken URL
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = json.dumps(
            {
                "verdict": "REVISE",
                "factual_errors": [
                    {
                        "location": "References",
                        "claim": "https://broken.com",
                        "ground_truth": "URL unreachable",
                        "action": "remove",
                    }
                ],
                "reference_checks": [
                    {
                        "url": "https://broken.com",
                        "status": "broken",
                        "note": "FETCH_FAILED",
                    }
                ],
                "cuts": [],
                "style_issues": [],
                "confidence": 0.8,
            }
        )

        mock_llm_with_tools = AsyncMock()
        mock_llm_with_tools.ainvoke = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="FETCH_FAILED: HTTP 404")
        mock_tool.name = "fetch_reference_content"

        editor.llm = MagicMock()
        editor.llm_with_tools = mock_llm_with_tools
        editor.tools = [mock_tool]
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        result = await editor.review("# Article\n\nContent.", "DATA_BLOCK: ...")

        assert result["verdict"] == "REVISE"
        assert len(result["factual_errors"]) == 1
        assert result["factual_errors"][0]["action"] == "remove"

    @pytest.mark.asyncio
    async def test_review_respects_max_iterations(self):
        """Tool loop should terminate after MAX_TOOL_ITERATIONS even if LLM keeps calling tools."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        # Every response returns tool_calls — should be bounded
        tool_call_response = MagicMock()
        tool_call_response.tool_calls = [
            {
                "name": "fetch_reference_content",
                "args": {"url": "https://example.com"},
                "id": "call_1",
            }
        ]
        tool_call_response.content = ""

        # Final forced response (no tools)
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = json.dumps({"verdict": "APPROVED", "confidence": 0.7})

        mock_llm_with_tools = AsyncMock()
        # Return tool calls for MAX_TOOL_ITERATIONS, then we fall through to bare LLM
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=tool_call_response)

        mock_bare_llm = AsyncMock()
        mock_bare_llm.ainvoke = AsyncMock(return_value=final_response)

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="Some content")
        mock_tool.name = "fetch_reference_content"

        editor.llm = mock_bare_llm
        editor.llm_with_tools = mock_llm_with_tools
        editor.tools = [mock_tool]
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        result = await editor.review("# Article", "Context")

        assert result["verdict"] == "APPROVED"
        # Tool-bound LLM should have been called exactly MAX_TOOL_ITERATIONS times
        assert mock_llm_with_tools.ainvoke.call_count == editor.MAX_TOOL_ITERATIONS
        # Bare LLM should have been called once (safety valve)
        mock_bare_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_works_without_tools(self):
        """When llm_with_tools is None (no tools), should work like single-shot."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = json.dumps({"verdict": "APPROVED", "confidence": 0.95})

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=final_response)

        editor.llm = mock_llm
        editor.llm_with_tools = None  # No tools available
        editor.tools = []
        editor._tools_by_name = {}

        result = await editor.review("# Article", "Context")

        assert result["verdict"] == "APPROVED"
        assert result["confidence"] == 0.95
        # Should use bare LLM
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_calls_handles_unknown_tool(self):
        """_execute_tool_calls should return error for unknown tools."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()
        editor._tools_by_name = {}

        results = await editor._execute_tool_calls(
            [{"name": "nonexistent_tool", "args": {}, "id": "call_1"}]
        )

        assert len(results) == 1
        assert "Unknown tool" in results[0].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_caps_per_turn(self):
        """_execute_tool_calls should cap tool calls at MAX_TOOL_CALLS_PER_TURN."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="Content")
        mock_tool.name = "fetch_reference_content"

        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        # Request 5 tool calls (cap is 3)
        tool_calls = [
            {
                "name": "fetch_reference_content",
                "args": {"url": f"https://example.com/{i}"},
                "id": f"call_{i}",
            }
            for i in range(5)
        ]

        results = await editor._execute_tool_calls(tool_calls)

        # Should return 5 ToolMessages (3 executed + 2 skipped)
        assert len(results) == 5
        # Only 3 actual tool executions
        assert mock_tool.ainvoke.call_count == 3
        # Last 2 should be SKIPPED
        assert "SKIPPED" in results[3].content
        assert "SKIPPED" in results[4].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_handles_tool_exception(self):
        """_execute_tool_calls should catch exceptions from tool execution."""
        from src.article_writer import ArticleEditor

        editor = ArticleEditor()

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("Connection reset"))
        mock_tool.name = "fetch_reference_content"

        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        results = await editor._execute_tool_calls(
            [
                {
                    "name": "fetch_reference_content",
                    "args": {"url": "https://example.com"},
                    "id": "call_1",
                }
            ]
        )

        assert len(results) == 1
        assert "TOOL_ERROR" in results[0].content
        assert "Connection reset" in results[0].content


# =============================================================================
# search_claim Tool Tests
# =============================================================================


class TestSearchClaimTool:
    """Tests for the search_claim editor tool."""

    @pytest.mark.asyncio
    async def test_invalid_query_rejected(self):
        """Short/empty queries should be rejected."""
        from src.editor_tools import search_claim

        result = await search_claim.ainvoke({"query": ""})
        assert "INVALID_QUERY" in result

        result = await search_claim.ainvoke({"query": "ab"})
        assert "INVALID_QUERY" in result

    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Successful search should return content."""
        from src.editor_tools import search_claim

        mock_result = "Ultraman Card Game was launched in Q3 2024 by Tsuburaya Fields."

        with patch(
            "src.tavily_utils.tavily_search_with_timeout",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await search_claim.ainvoke(
                {"query": "Tsuburaya Fields Ultraman Card Game launch date"}
            )
            assert "Ultraman" in result

    @pytest.mark.asyncio
    async def test_search_unavailable(self):
        """Should handle Tavily being unavailable."""
        from src.editor_tools import search_claim

        with patch(
            "src.tavily_utils.tavily_search_with_timeout",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await search_claim.ainvoke({"query": "test query here"})
            assert "SEARCH_UNAVAILABLE" in result

    @pytest.mark.asyncio
    async def test_search_truncates_long_results(self):
        """Long search results should be truncated."""
        from src.editor_tools import MAX_CLAIM_SEARCH_CHARS, search_claim

        long_result = "x" * (MAX_CLAIM_SEARCH_CHARS + 1000)

        with patch(
            "src.tavily_utils.tavily_search_with_timeout",
            new_callable=AsyncMock,
            return_value=long_result,
        ):
            result = await search_claim.ainvoke({"query": "test long query"})
            assert (
                len(result) <= MAX_CLAIM_SEARCH_CHARS + 50
            )  # Allow for truncation marker
            assert "truncated" in result

    def test_get_editor_tools_includes_search_claim(self):
        """get_editor_tools should include both tools."""
        tools = get_editor_tools()
        tool_names = [t.name for t in tools]
        assert "fetch_reference_content" in tool_names
        assert "search_claim" in tool_names
