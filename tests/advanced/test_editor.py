"""
Tests for the Editor-in-Chief article revision system.

Tests cover:
- Editor tool (fetch_reference_content)
- Editor prompt loading
- Article revision workflow
- Editorial loop with mock LLMs
"""

import json
import warnings
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest


def _fetch_reference_content_tool():
    from src.editor_tools import fetch_reference_content

    return fetch_reference_content


def _get_editor_tools():
    from src.editor_tools import get_editor_tools

    return get_editor_tools()


def _max_reference_chars() -> int:
    from src.editor_tools import MAX_REFERENCE_CHARS

    return MAX_REFERENCE_CHARS


def _create_article_editor(*, llm=None, tools=None):
    from src.article_writer import ArticleEditor

    with (
        patch("src.llms.create_editor_llm", return_value=llm),
        patch("src.editor_tools.get_editor_tools", return_value=tools or []),
    ):
        return ArticleEditor()


# =============================================================================
# Tool Tests
# =============================================================================


class TestFetchReferenceContent:
    """Tests for the fetch_reference_content async tool."""

    @pytest.mark.asyncio
    async def test_invalid_url_rejected(self):
        """Invalid URLs should be rejected immediately."""
        tool = _fetch_reference_content_tool()
        result = await tool.ainvoke({"url": "not-a-url"})
        assert "INVALID_URL" in result

        result = await tool.ainvoke({"url": ""})
        assert "INVALID_URL" in result

        result = await tool.ainvoke({"url": "ftp://example.com"})
        assert "INVALID_URL" in result

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Successful fetch should return cleaned text content."""
        tool = _fetch_reference_content_tool()
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

            result = await tool.ainvoke({"url": "https://example.com/article"})

            # Should contain main content but not nav/footer
            assert "main content" in result
            assert "important information" in result
            assert "Navigation" not in result
            assert "Footer content" not in result

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Timeout should return appropriate error message."""
        tool = _fetch_reference_content_tool()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await tool.ainvoke({"url": "https://example.com/slow"})

            assert "FETCH_FAILED" in result
            assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """HTTP errors should return status code in error message."""
        tool = _fetch_reference_content_tool()
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

            result = await tool.ainvoke({"url": "https://example.com/missing"})

            assert "FETCH_FAILED" in result
            assert "404" in result

    @pytest.mark.asyncio
    async def test_insufficient_content(self):
        """Pages with very little text should return INSUFFICIENT_CONTENT."""
        tool = _fetch_reference_content_tool()
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

            result = await tool.ainvoke({"url": "https://example.com/empty"})

            assert "INSUFFICIENT_CONTENT" in result

    @pytest.mark.asyncio
    async def test_content_truncation(self):
        """Long content should be truncated to MAX_REFERENCE_CHARS."""
        tool = _fetch_reference_content_tool()
        max_reference_chars = _max_reference_chars()
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

            result = await tool.ainvoke({"url": "https://example.com/long"})

            # Should be truncated with indicator
            assert (
                len(result) <= max_reference_chars + 20
            )  # Allow for truncation marker
            assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_redirect_to_private_target_rejected(self):
        """A public URL must not be allowed to redirect into a private target."""
        tool = _fetch_reference_content_tool()

        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {"location": "http://127.0.0.1:8080/admin"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=redirect)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await tool.ainvoke({"url": "https://example.com/redirect"})

            assert "FETCH_FAILED" in result or "INVALID_URL" in result
            assert "redirect" in result.lower()

    @pytest.mark.asyncio
    async def test_too_many_redirects_rejected(self):
        """Redirect loops should stop after the configured hop budget."""
        tool = _fetch_reference_content_tool()

        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {"location": "https://example.com/next"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=redirect)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await tool.ainvoke({"url": "https://example.com/loop"})

            assert "FETCH_FAILED" in result
            assert "redirect" in result.lower()


class TestGetEditorTools:
    """Tests for the get_editor_tools function."""

    def test_returns_list_of_tools(self):
        """Should return a list with the fetch tool."""
        tools = _get_editor_tools()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_tool_has_correct_name(self):
        """Tool should have the expected name."""
        tools = _get_editor_tools()
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
        editor = _create_article_editor()
        assert hasattr(editor, "llm")
        assert hasattr(editor, "tools")
        assert hasattr(editor, "is_available")

    def test_build_fact_check_context(self):
        """build_fact_check_context should assemble context correctly."""
        editor = _create_article_editor()

        context = editor.build_fact_check_context(
            data_block="FINANCIAL_HEALTH: 75%\nP/E: 12.5",
            pm_block="VERDICT: BUY\nCONVICTION: HIGH",
            valuation_params="52_WEEK_HIGH: 100\n52_WEEK_LOW: 50",
            governance_context=(
                "=== ENTITY GOVERNANCE CARD (authoritative for identity) ===\n"
                "Ticker: 009970.KS\nRelated listed tickers: 111770.KS"
            ),
        )

        assert "DATA_BLOCK" in context
        assert "FINANCIAL_HEALTH: 75%" in context
        assert "PM_BLOCK" in context
        assert "VERDICT: BUY" in context
        assert "VALUATION PARAMETERS" in context
        assert "ENTITY GOVERNANCE CARD" in context
        assert context.count("ENTITY GOVERNANCE CARD") == 1
        assert "111770.KS" in context

    def test_build_fact_check_context_empty(self):
        """build_fact_check_context should handle empty inputs."""
        editor = _create_article_editor()
        context = editor.build_fact_check_context()

        assert context == "No context provided."

    def test_build_fact_check_context_includes_source_confidence(self):
        """build_fact_check_context should expose weak-source warnings."""
        editor = _create_article_editor()

        context = editor.build_fact_check_context(
            data_block="""### --- START DATA_BLOCK ---
OPERATING_CASH_FLOW_SOURCE: JUNIOR
OCF_FILING_REASON: missing filing support
ANALYST_COVERAGE_DATA_QUALITY_NOTE: English count understates total coverage.
### --- END DATA_BLOCK ---
""",
            consultant_review="SPOT_CHECK: COVERAGE_GAP on OCF.",
        )

        assert "SOURCE CONFIDENCE" in context
        assert "OPERATING_CASH_FLOW_SOURCE: JUNIOR" in context
        assert "COVERAGE_GAP" in context
        assert "aggregator-indicated" in context

    def test_parse_editor_response_valid_json(self):
        """_parse_editor_response should parse valid JSON."""
        editor = _create_article_editor()

        response = '{"verdict": "REVISE", "factual_errors": [], "confidence": 0.9}'
        result = editor._parse_editor_response(response)

        assert result["verdict"] == "REVISE"
        assert result["confidence"] == 0.9

    def test_parse_editor_response_json_in_code_block(self):
        """_parse_editor_response should extract JSON from code blocks."""
        editor = _create_article_editor()

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
        editor = _create_article_editor()

        response = "This is not valid JSON at all."
        result = editor._parse_editor_response(response)

        # Malformed approval cannot publish a final article.
        assert result["verdict"] == "REVISE"
        assert "parse_error" in result

    def test_is_available_returns_bool(self):
        """is_available should return boolean."""
        editor = _create_article_editor()
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

    def test_create_editor_llm_uses_openai_responses_mode(self):
        """Editor should use the OpenAI Responses API via ChatOpenAI."""
        from src.llms import create_editor_llm

        with patch("langchain_openai.ChatOpenAI") as mock_chatgpt:
            mock_chatgpt.return_value = MagicMock()
            with patch("src.llms.config") as mock_config:
                mock_config.enable_consultant = True
                mock_config.get_openai_api_key.return_value = "test-key"
                mock_config.editor_model = "gpt-4o"
                mock_config.consultant_model = "gpt-4o"

                result = create_editor_llm()

        assert result is not None
        call_kwargs = mock_chatgpt.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["use_responses_api"] is True
        assert call_kwargs["output_version"] == "responses/v1"
        assert "temperature" not in call_kwargs

    def test_create_editor_llm_sets_reasoning_effort_for_gpt5(self):
        """Non-pro GPT-5 editor models should opt into medium reasoning effort."""
        from src.llms import create_editor_llm

        with patch("langchain_openai.ChatOpenAI") as mock_chatgpt:
            llm = MagicMock()
            mock_chatgpt.return_value = llm
            with patch("src.llms.config") as mock_config:
                mock_config.enable_consultant = True
                mock_config.get_openai_api_key.return_value = "test-key"
                mock_config.editor_model = "gpt-5"
                mock_config.consultant_model = "gpt-4o"

                create_editor_llm()

        assert mock_chatgpt.call_args.kwargs["reasoning_effort"] == "medium"
        assert mock_chatgpt.call_args.kwargs["max_completion_tokens"] == 10240
        assert llm._configured_max_completion_tokens == 8192
        assert llm._configured_api_completion_tokens == 10240


# =============================================================================
# Integration Tests
# =============================================================================


class TestEditorialLoopIntegration:
    """Integration tests for the full editorial loop."""

    def test_review_response_for_citation_errors_shape(self):
        """Citation audit feedback should match the editor feedback contract."""
        from src.article_writer import _review_response_for_citation_errors

        errors = [
            {
                "location": "DATA_BLOCK citation audit",
                "claim": "Article cites (A: 1)",
                "ground_truth": "DATA_BLOCK shows A: 2",
                "action": "Correct A.",
            },
            {
                "location": "DATA_BLOCK citation audit",
                "claim": "Article cites (B: 3)",
                "ground_truth": "DATA_BLOCK shows B: 4",
                "action": "Correct B.",
            },
        ]

        feedback = _review_response_for_citation_errors(errors)

        assert feedback["verdict"] == "REVISE"
        assert feedback["confidence"] == 1.0
        assert feedback["deterministic_citation_audit"] is True
        assert feedback["citation_audit_status"] == "FAILED"
        assert len(feedback["factual_errors"]) == 2

    @pytest.mark.asyncio
    async def test_edit_with_unavailable_editor(self):
        """edit() should return original draft when editor unavailable."""
        from src.article_writer import ArticleWriter

        writer = ArticleWriter()
        editor = _create_article_editor()

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
        from src.article_writer import ArticleWriter

        writer = ArticleWriter()
        editor = _create_article_editor()

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
        from src.article_writer import ArticleWriter

        writer = ArticleWriter()
        editor = _create_article_editor()

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
            governance_card={
                "ticker": "009970.KS",
                "canonical_name": "Youngone Holdings Co., Ltd.",
            },
        )

        # Should have called revise once
        assert writer.revise.called
        assert "009970.KS" in writer.revise.call_args.kwargs["governance_context"]
        assert feedback["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_edit_keeps_previous_draft_when_revision_fails(self):
        """A failed revision (e.g. truncated output) must not lose the draft.

        The revise call raises; edit() keeps the last complete draft, runs
        the final review, and returns instead of propagating.
        """
        from src.article_writer import ArticleWriter

        writer = ArticleWriter()
        editor = _create_article_editor()

        review_count = 0

        async def mock_review(*args, **kwargs):
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

        editor.review = mock_review
        editor.llm = MagicMock()

        writer.revise = MagicMock(
            side_effect=RuntimeError(
                "Writer LLM hit the output token limit before finishing"
            )
        )

        draft = "# Complete Draft\n\nViable content."
        result, feedback = await editor.edit(
            writer=writer,
            article_draft=draft,
            ticker="TEST",
            company_name="Test Corp",
        )

        assert result == draft, "previous complete draft must be preserved"
        assert writer.revise.call_count == 1
        # Loop review + post-loop final review both ran
        assert review_count == 2

    @pytest.mark.asyncio
    async def test_edit_respects_max_revisions(self):
        """edit() should stop after MAX_REVISIONS iterations."""
        from src.article_writer import ArticleWriter

        writer = ArticleWriter()
        editor = _create_article_editor()

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

    @pytest.mark.asyncio
    async def test_edit_short_circuits_review_for_citation_mismatch(self):
        """Citation mismatches should revise before calling the LLM editor."""
        editor = _create_article_editor()
        editor.llm = MagicMock()
        editor.review = AsyncMock(
            return_value={"verdict": "APPROVED", "confidence": 0.9}
        )

        writer = MagicMock()
        writer.revise.return_value = "# Revised\n\nLeverage `(NET_DEBT_EBITDA: 1.95)`."
        data_block = (
            "### --- START DATA_BLOCK ---\n"
            "NET_DEBT_EBITDA: 1.95\n"
            "### --- END DATA_BLOCK ---"
        )

        result, feedback = await editor.edit(
            writer=writer,
            article_draft="# Draft\n\nLeverage `(NET_DEBT_EBITDA: -0.01)`.",
            ticker="TEST",
            company_name="Test Corp",
            data_block=data_block,
        )

        assert result == "# Revised\n\nLeverage `(NET_DEBT_EBITDA: 1.95)`."
        assert feedback["verdict"] == "APPROVED"
        assert editor.review.call_count == 1
        assert writer.revise.call_count == 1
        revise_feedback = writer.revise.call_args.kwargs["editor_feedback"]
        assert revise_feedback["deterministic_citation_audit"] is True
        assert revise_feedback["citation_audit_status"] == "FAILED"
        assert revise_feedback["verdict"] == "REVISE"

    @pytest.mark.asyncio
    async def test_edit_prepends_caveats_for_persistent_citation_mismatch(self):
        """Unresolved citation mismatches should be visible after max revisions."""
        editor = _create_article_editor()
        editor.llm = MagicMock()
        editor.MAX_REVISIONS = 1
        editor.review = AsyncMock(
            return_value={"verdict": "APPROVED", "confidence": 0.9}
        )

        writer = MagicMock()
        writer.revise.return_value = (
            "# Still Wrong\n\nLeverage `(NET_DEBT_EBITDA: -0.01)`."
        )
        data_block = (
            "### --- START DATA_BLOCK ---\n"
            "NET_DEBT_EBITDA: 1.95\n"
            "### --- END DATA_BLOCK ---"
        )

        result, feedback = await editor.edit(
            writer=writer,
            article_draft="# Draft\n\nLeverage `(NET_DEBT_EBITDA: -0.01)`.",
            ticker="TEST",
            company_name="Test Corp",
            data_block=data_block,
        )

        # Caveats are QA scaffolding: they land below the article's H1 title.
        assert result.startswith("# Still Wrong\n\n## Verification Caveats")
        assert "NET_DEBT_EBITDA: 1.95" in result
        assert feedback["verdict"] == "REVISE"
        assert feedback["deterministic_citation_audit"] is True
        assert feedback["citation_audit_status"] == "FAILED"
        assert writer.revise.call_count == 1
        assert editor.review.call_count == 1


class TestMainPyIntegration:
    """Tests that verify the editor is properly wired into main.py."""

    @pytest.mark.parametrize(
        ("feedback", "expected"),
        [
            ({"verdict": "APPROVED", "citation_audit_status": "PASSED"}, True),
            ({"verdict": "APPROVED", "citation_audit_status": "FAILED"}, False),
            ({"verdict": "APPROVED"}, False),
            ({"verdict": "REVISE", "citation_audit_status": "PASSED"}, False),
        ],
    )
    def test_publication_requires_editor_and_citation_approval(
        self,
        feedback,
        expected,
    ):
        from src.output import _article_is_publishable

        assert _article_is_publishable("# Article", feedback) is expected

    def test_strict_publication_requires_valid_final_decision_trace(self):
        from src.output import _article_is_publishable

        feedback = {"verdict": "APPROVED", "citation_audit_status": "PASSED"}

        assert not _article_is_publishable(
            "# Article",
            feedback,
            decision_trace={"status": "INVALID"},
            require_decision_trace=True,
        )
        assert _article_is_publishable(
            "# Article",
            feedback,
            decision_trace={"status": "VALID", "verdict": "BUY"},
            require_decision_trace=True,
        )

    @pytest.mark.asyncio
    async def test_handle_article_generation_calls_editor(self, tmp_path):
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
            return_value=(
                "# Final Article\n\nEdited content.",
                {
                    "verdict": "APPROVED",
                    "citation_audit_status": "PASSED",
                },
            )
        )

        with (
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", MagicMock()),
        ):
            from src.output import handle_article_generation

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
                    "entity_governance_card": {
                        "ticker": "009970.KS",
                        "canonical_name": "Youngone Holdings Co., Ltd.",
                    },
                },
                resolve_article_path_fn=lambda *_args, **_kwargs: (
                    tmp_path / "test_article.md"
                ),
            )

            # Verify editor.edit() was called
            mock_editor_instance.edit.assert_called_once()

            # Verify it was called with the ground truth data
            call_kwargs = mock_editor_instance.edit.call_args.kwargs
            assert call_kwargs["data_block"] == "DATA_BLOCK"
            assert call_kwargs["pm_block"] == "PM_BLOCK"
            assert call_kwargs["valuation_params"] == "VAL_PARAMS"
            assert call_kwargs["governance_card"]["ticker"] == "009970.KS"
            assert call_kwargs["evidence_constraints"]
            assert (
                mock_writer_instance.write.call_args.kwargs["evidence_constraints"]
                == call_kwargs["evidence_constraints"]
            )
            assert mock_writer_instance.write.call_args.kwargs["output_path"] is None
            assert (tmp_path / "test_article.md").read_text() == (
                "# Final Article\n\nEdited content."
            )

    @pytest.mark.asyncio
    async def test_handle_article_generation_skips_editor_when_unavailable(
        self, tmp_path
    ):
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
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", MagicMock()),
        ):
            from src.output import handle_article_generation

            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
                resolve_article_path_fn=lambda *_args, **_kwargs: (
                    tmp_path / "test_article.md"
                ),
            )

            # Verify editor.edit() was NOT called
            mock_editor_instance.edit.assert_not_called()
            assert not (tmp_path / "test_article.md").exists()
            assert (tmp_path / "test_article.draft.md").read_text() == "# Draft Article"

    def test_handle_article_generation_signature_includes_analysis_result(self):
        """Verify handle_article_generation accepts analysis_result parameter."""
        import inspect

        from src.output import handle_article_generation

        sig = inspect.signature(handle_article_generation)
        param_names = list(sig.parameters.keys())

        assert "analysis_result" in param_names, (
            "handle_article_generation must accept analysis_result parameter "
            "to pass DATA_BLOCK/PM_BLOCK to the editor"
        )

    @pytest.mark.asyncio
    async def test_handle_article_generation_preserves_draft_on_editor_failure(
        self, tmp_path
    ):
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
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", m_open),
            patch("src.main.console"),
        ):
            from src.output import handle_article_generation

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
                resolve_article_path_fn=lambda *_a, **_k: str(
                    tmp_path / "test_article.md"
                ),
            )

        # Correct error handling counts as success: the run completes without
        # raising AND the draft is preserved for review rather than lost.
        draft = tmp_path / "test_article.draft.md"
        assert draft.exists(), "draft must be preserved when the editor fails"
        assert "This is the draft." in draft.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_handle_article_generation_surfaces_writer_fallback(self, tmp_path):
        """A silent Claude → Gemini fallback must land in the run summary."""
        from unittest.mock import MagicMock, mock_open, patch

        args = MagicMock()
        args.article = "test_article.md"
        args.output = "/tmp/test_output.md"
        args.quiet = True
        args.brief = False

        mock_writer_instance = MagicMock()
        mock_writer_instance.write.return_value = "# Draft"
        mock_writer_instance.current_model_name = "gemini-3.5-flash"
        mock_writer_instance.writer_fell_back = True

        mock_editor_instance = MagicMock()
        mock_editor_instance.is_available.return_value = False

        analysis_result = {
            "fundamentals_report": "DATA",
            "final_trade_decision": "PM",
        }

        with (
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", mock_open()),
        ):
            from src.output import handle_article_generation

            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
                analysis_result=analysis_result,
                resolve_article_path_fn=lambda *_a, **_k: str(
                    tmp_path / "test_article.md"
                ),
            )

        run_summary = analysis_result["run_summary"]
        assert run_summary["article_writer_model"] == "gemini-3.5-flash"
        assert run_summary["article_writer_fell_back"] is True

    @pytest.mark.asyncio
    async def test_handle_article_generation_patches_saved_json(self, tmp_path):
        """The writer-model stamp must reach the already-persisted analysis JSON.

        The JSON artifact is saved before article generation, so the in-memory
        run_summary stamp alone never reaches disk (the 3393.T review gap).
        """
        import json
        from unittest.mock import MagicMock, mock_open, patch

        args = MagicMock()
        args.article = "test_article.md"
        args.output = "/tmp/test_output.md"
        args.quiet = True
        args.brief = False

        mock_writer_instance = MagicMock()
        mock_writer_instance.write.return_value = "# Draft"
        mock_writer_instance.current_model_name = "gemini-3.5-flash"
        mock_writer_instance.writer_fell_back = True

        mock_editor_instance = MagicMock()
        mock_editor_instance.is_available.return_value = False

        saved_json = tmp_path / "TEST_20260704_analysis.json"
        saved_json.write_text(
            json.dumps({"run_summary": {"verdict": "HOLD"}}), encoding="utf-8"
        )
        analysis_result = {
            "fundamentals_report": "DATA",
            "final_trade_decision": "PM",
            "_saved_analysis_path": str(saved_json),
        }

        with (
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", mock_open()),
        ):
            from src.output import handle_article_generation

            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
                analysis_result=analysis_result,
                resolve_article_path_fn=lambda *_a, **_k: str(
                    tmp_path / "test_article.md"
                ),
            )

        data = json.loads(saved_json.read_text(encoding="utf-8"))
        assert data["run_summary"]["verdict"] == "HOLD"
        assert data["run_summary"]["article_writer_model"] == "gemini-3.5-flash"
        assert data["run_summary"]["article_writer_fell_back"] is True

    @pytest.mark.asyncio
    async def test_handle_article_generation_records_primary_writer_model(
        self, tmp_path
    ):
        from unittest.mock import MagicMock, mock_open, patch

        args = MagicMock()
        args.article = "test_article.md"
        args.output = "/tmp/test_output.md"
        args.quiet = True
        args.brief = False

        mock_writer_instance = MagicMock()
        mock_writer_instance.write.return_value = "# Draft"
        mock_writer_instance.current_model_name = "claude-opus-4-8"
        mock_writer_instance.writer_fell_back = False

        mock_editor_instance = MagicMock()
        mock_editor_instance.is_available.return_value = False

        analysis_result = {}

        with (
            patch(
                "src.article_writer.ArticleWriter", return_value=mock_writer_instance
            ),
            patch(
                "src.article_writer.ArticleEditor", return_value=mock_editor_instance
            ),
            patch("builtins.open", mock_open()),
        ):
            from src.output import handle_article_generation

            await handle_article_generation(
                args=args,
                ticker="TEST",
                company_name="Test Corp",
                report_text="Full report...",
                trade_date="2026-01-01",
                analysis_result=analysis_result,
                resolve_article_path_fn=lambda *_a, **_k: str(
                    tmp_path / "test_article.md"
                ),
            )

        run_summary = analysis_result["run_summary"]
        assert run_summary["article_writer_model"] == "claude-opus-4-8"
        assert run_summary["article_writer_fell_back"] is False


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

    def test_openai_responses_v1_reasoning_blocks_skipped(self):
        """OpenAI responses/v1 (writer EDITOR_MODEL fallback tier): reasoning
        blocks are excluded by type, even if they carry a text field."""
        from src.article_writer import _extract_text_from_response

        mock_response = MagicMock()
        mock_response.content = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "planning..."}],
                "text": "internal chain of thought",
            },
            {"type": "text", "text": "# GPT Fallback Article\n\nContent."},
        ]

        result = _extract_text_from_response(mock_response)
        assert result == "# GPT Fallback Article\n\nContent."
        assert "chain of thought" not in result

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
        """No CLAUDE_KEY and no usable OpenAI tier → Gemini floor of the chain."""
        from src.llms import create_writer_llm

        mock_llm = MagicMock()
        with (
            patch("src.llms.config") as mock_config,
            # Prevent ChatGoogleGenerativeAI from probing network/proxy at construction time
            patch("src.llms.ChatGoogleGenerativeAI", return_value=mock_llm),
        ):
            mock_config.get_claude_api_key.return_value = None
            # No usable OpenAI tier — the chain must resolve to the Gemini floor.
            mock_config.enable_consultant = False
            mock_config.get_openai_api_key.return_value = ""
            mock_config.deep_think_llm = "gemini-3-pro-preview"
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 10
            mock_config.get_google_api_key.return_value = "fake-key"
            mock_config.gemini_rpm_limit = 15

            llm = create_writer_llm()
            # Should return something (Gemini fallback), not raise
            assert llm is not None

    def test_opus_effort_field(self):
        """langchain-anthropic >=1.3.0: effort is a first-class ChatAnthropic field.

        The library serializes it into output_config in the API call internally.
        It must NOT appear in model_kwargs (which is passed through verbatim).
        """
        from src.llms import create_writer_llm

        with patch("src.llms.config") as mock_config:
            mock_config.get_claude_api_key.return_value = "fake-key"
            mock_config.writer_model = "claude-opus-4-6"
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 3

            llm = create_writer_llm()
            # effort is a direct field, not in model_kwargs
            assert llm.effort == "high", (
                "Opus 4.6 effort must be set as a direct ChatAnthropic field "
                "(langchain-anthropic >=1.3.0 serializes it into output_config)"
            )
            assert "effort" not in llm.model_kwargs, (
                "effort must not appear in model_kwargs — the library handles "
                "wrapping it in output_config for the API call"
            )

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-4-8-20260601",
        ],
    )
    def test_opus_4_6_plus_uses_adaptive_thinking(self, model_name):
        from src.llms import create_writer_llm

        with patch("src.llms.config") as mock_config:
            mock_config.get_claude_api_key.return_value = "fake-key"
            mock_config.writer_model = model_name
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 3

            llm = create_writer_llm()

        assert llm.thinking == {"type": "adaptive"}
        assert llm.effort == "high"
        assert llm.thinking.get("type") != "enabled"

    def test_opus_4_5_keeps_manual_thinking(self):
        from src.llms import create_writer_llm

        with patch("src.llms.config") as mock_config:
            mock_config.get_claude_api_key.return_value = "fake-key"
            mock_config.writer_model = "claude-opus-4-5"
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 3

            llm = create_writer_llm()

        assert llm.thinking == {"type": "enabled", "budget_tokens": 8192}
        assert getattr(llm, "effort", None) is None

    def test_opus_effort_serialized_into_output_config_payload(self):
        """Wire-format regression: effort must appear in output_config in the API payload.

        Verifies that langchain-anthropic's _get_request_payload() correctly
        maps self.effort -> payload["output_config"]["effort"] so that the
        Anthropic API receives the right structure regardless of how the library
        internally changed between 0.x (model_kwargs workaround) and 1.3.0+
        (first-class field with built-in serializer).
        """
        from langchain_core.messages import HumanMessage

        from src.llms import create_writer_llm

        with patch("src.llms.config") as mock_config:
            mock_config.get_claude_api_key.return_value = "fake-key"
            mock_config.writer_model = "claude-opus-4-6"
            mock_config.api_timeout = 300
            mock_config.api_retry_attempts = 3

            llm = create_writer_llm()
            payload = llm._get_request_payload(
                [HumanMessage(content="test")], stop=None
            )
            assert (
                "output_config" in payload
            ), "effort=high must produce output_config in the API payload"
            assert (
                payload["output_config"].get("effort") == "high"
            ), f"output_config must contain effort='high', got: {payload['output_config']}"

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
        editor = _create_article_editor()

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

        # Second response: no more tool calls; final verdict comes from structured pass
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = "Done reviewing."

        structured_review_llm = AsyncMock()
        structured_review_llm.ainvoke = AsyncMock(
            return_value={
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
        editor.review_llm = structured_review_llm
        editor.tools = [mock_tool]
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        result = await editor.review("# Test Article\n\nContent.", "DATA_BLOCK: ...")

        assert result["verdict"] == "APPROVED"
        assert len(result["reference_checks"]) == 1
        # Tool should have been called once
        mock_tool.ainvoke.assert_called_once_with(
            {"url": "https://example.com/article"}
        )
        # Tool-bound LLM should have been called twice (tool call + stop signal)
        assert mock_llm_with_tools.ainvoke.call_count == 2
        structured_review_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_handles_tool_error_gracefully(self):
        """When tool returns FETCH_FAILED, editor should still produce valid JSON."""
        editor = _create_article_editor()

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

        # Second response: no more tool calls; verdict comes from structured pass
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = "Done reviewing."

        structured_review_llm = AsyncMock()
        structured_review_llm.ainvoke = AsyncMock(
            return_value={
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
        editor.review_llm = structured_review_llm
        editor.tools = [mock_tool]
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        result = await editor.review("# Article\n\nContent.", "DATA_BLOCK: ...")

        assert result["verdict"] == "REVISE"
        assert len(result["factual_errors"]) == 1
        assert result["factual_errors"][0]["action"] == "remove"

    @pytest.mark.asyncio
    async def test_review_respects_max_iterations(self):
        """Tool loop should terminate after MAX_TOOL_ITERATIONS even if LLM keeps calling tools."""
        editor = _create_article_editor()

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

        structured_review_llm = AsyncMock()
        structured_review_llm.ainvoke = AsyncMock(
            return_value={"verdict": "APPROVED", "confidence": 0.7}
        )

        mock_llm_with_tools = AsyncMock()
        # Return tool calls for MAX_TOOL_ITERATIONS, then we fall through to structured review
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=tool_call_response)

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="Some content")
        mock_tool.name = "fetch_reference_content"

        editor.llm = MagicMock()
        editor.llm_with_tools = mock_llm_with_tools
        editor.review_llm = structured_review_llm
        editor.tools = [mock_tool]
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        result = await editor.review("# Article", "Context")

        assert result["verdict"] == "APPROVED"
        # Tool-bound LLM should have been called exactly MAX_TOOL_ITERATIONS times
        assert mock_llm_with_tools.ainvoke.call_count == editor.MAX_TOOL_ITERATIONS
        structured_review_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_works_without_tools(self):
        """When llm_with_tools is None (no tools), should work like single-shot."""
        editor = _create_article_editor()

        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = json.dumps({"verdict": "APPROVED", "confidence": 0.95})

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=final_response)

        editor.llm = mock_llm
        editor.llm_with_tools = None  # No tools available
        editor.review_llm = None
        editor.tools = []
        editor._tools_by_name = {}

        result = await editor.review("# Article", "Context")

        assert result["verdict"] == "APPROVED"
        assert result["confidence"] == 0.95
        # Should use bare LLM
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_review_suppresses_known_pydantic_warning_when_not_debug(
        self,
    ):
        """Known structured-output serializer noise should stay out of normal runs."""
        editor = _create_article_editor()

        async def _warn_and_return(_messages, config=None):
            warnings.warn_explicit(
                (
                    "Pydantic serializer warnings:\n"
                    "  PydanticSerializationUnexpectedValue("
                    "Expected `none` - serialized value may not be as expected)"
                ),
                UserWarning,
                filename="pydantic/main.py",
                lineno=464,
                module="pydantic.main",
            )
            return {"verdict": "APPROVED", "confidence": 0.91}

        editor.llm = MagicMock()
        editor.review_llm = AsyncMock()
        editor.review_llm.ainvoke = AsyncMock(side_effect=_warn_and_return)

        with patch(
            "src.article_writer._should_emit_editor_structured_output_warnings",
            return_value=False,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await editor._invoke_final_review([])

        assert result["verdict"] == "APPROVED"
        assert caught == []

    @pytest.mark.asyncio
    async def test_structured_review_emits_known_pydantic_warning_in_debug(self):
        """Debug runs should preserve the raw serializer warning for investigation."""
        editor = _create_article_editor()

        async def _warn_and_return(_messages, config=None):
            warnings.warn_explicit(
                (
                    "Pydantic serializer warnings:\n"
                    "  PydanticSerializationUnexpectedValue("
                    "Expected `none` - serialized value may not be as expected)"
                ),
                UserWarning,
                filename="pydantic/main.py",
                lineno=464,
                module="pydantic.main",
            )
            return {"verdict": "APPROVED", "confidence": 0.91}

        editor.llm = MagicMock()
        editor.review_llm = AsyncMock()
        editor.review_llm.ainvoke = AsyncMock(side_effect=_warn_and_return)

        with patch(
            "src.article_writer._should_emit_editor_structured_output_warnings",
            return_value=True,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await editor._invoke_final_review([])

        assert result["verdict"] == "APPROVED"
        assert len(caught) == 1
        assert "Pydantic serializer warnings" in str(caught[0].message)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_handles_unknown_tool(self):
        """_execute_tool_calls should return error for unknown tools."""
        editor = _create_article_editor()
        editor._tools_by_name = {}

        results = await editor._execute_tool_calls(
            [{"name": "nonexistent_tool", "args": {}, "id": "call_1"}]
        )

        assert len(results) == 1
        assert "Unknown tool" in results[0].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_caps_per_turn(self):
        """_execute_tool_calls should cap tool calls at MAX_TOOL_CALLS_PER_TURN."""
        editor = _create_article_editor()

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
        editor = _create_article_editor()

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

    @pytest.mark.asyncio
    async def test_execute_tool_calls_routes_through_tool_service(self):
        """_execute_tool_calls should use the shared tool execution service."""
        from src.tooling.runtime import ToolResult

        editor = _create_article_editor()

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="direct-result")
        mock_tool.name = "fetch_reference_content"
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        tool_service = Mock()
        tool_service.execute = AsyncMock()
        with patch(
            "src.article_writer.get_current_tool_service",
            return_value=tool_service,
        ):
            tool_service.execute.return_value = ToolResult(value="hooked-result")

            results = await editor._execute_tool_calls(
                [
                    {
                        "name": "fetch_reference_content",
                        "args": {"url": "https://example.com"},
                        "id": "call_1",
                    }
                ]
            )

        tool_service.execute.assert_awaited_once()
        mock_tool.ainvoke.assert_not_called()
        assert len(results) == 1
        assert results[0].content == "hooked-result"


# =============================================================================
# URL Cache Tests
# =============================================================================


class TestEditorUrlCache:
    """Tests for the session-scoped URL cache in the editorial loop."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_tool_invocation(self):
        """Second fetch of the same URL should be served from cache."""
        editor = _create_article_editor()
        editor._url_cache = {"https://reuters.com/test": "FETCH_FAILED: HTTP 401"}

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="should not be called")
        mock_tool.name = "fetch_reference_content"
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        results = await editor._execute_tool_calls(
            [
                {
                    "name": "fetch_reference_content",
                    "args": {"url": "https://reuters.com/test"},
                    "id": "call_1",
                }
            ]
        )

        assert len(results) == 1
        assert "HTTP 401" in results[0].content
        mock_tool.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_invokes_tool_and_stores(self):
        """First fetch should invoke the tool and populate the cache."""
        editor = _create_article_editor()
        editor._url_cache = {}

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="Page content here")
        mock_tool.name = "fetch_reference_content"
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        results = await editor._execute_tool_calls(
            [
                {
                    "name": "fetch_reference_content",
                    "args": {"url": "https://example.com/page"},
                    "id": "call_1",
                }
            ]
        )

        assert len(results) == 1
        assert "Page content" in results[0].content
        mock_tool.ainvoke.assert_called_once()
        assert editor._url_cache["https://example.com/page"] == "Page content here"

    @pytest.mark.asyncio
    async def test_cache_stores_errors(self):
        """Tool exceptions should also be cached to avoid retrying broken URLs."""
        editor = _create_article_editor()
        editor._url_cache = {}

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("Connection refused"))
        mock_tool.name = "fetch_reference_content"
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        results = await editor._execute_tool_calls(
            [
                {
                    "name": "fetch_reference_content",
                    "args": {"url": "https://broken.com"},
                    "id": "call_1",
                }
            ]
        )

        assert "TOOL_ERROR" in results[0].content
        assert "https://broken.com" in editor._url_cache
        assert "TOOL_ERROR" in editor._url_cache["https://broken.com"]

    @pytest.mark.asyncio
    async def test_non_fetch_tools_bypass_cache(self):
        """search_claim and other tools should not use the URL cache."""
        editor = _create_article_editor()
        editor._url_cache = {}

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="Search results")
        mock_tool.name = "search_claim"
        editor._tools_by_name = {"search_claim": mock_tool}

        await editor._execute_tool_calls(
            [
                {
                    "name": "search_claim",
                    "args": {"query": "test query"},
                    "id": "call_1",
                }
            ]
        )

        mock_tool.ainvoke.assert_called_once()
        assert len(editor._url_cache) == 0

    @pytest.mark.asyncio
    async def test_editorial_loop_clears_cache(self):
        """Cache should be empty after run_editorial_loop completes."""
        editor = _create_article_editor()

        # Mock review to approve immediately
        async def mock_review(draft, context):
            return {"verdict": "APPROVED", "confidence": 0.95}

        editor.review = mock_review
        editor.llm = MagicMock()  # Needed for is_available()

        _, feedback = await editor.edit(
            writer=MagicMock(),
            article_draft="# Test\n\nContent.",
            ticker="TEST.X",
            company_name="Test Corp",
        )

        assert feedback["verdict"] == "APPROVED"
        assert editor._url_cache == {}


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
    async def test_search_claim_preserves_blocked_sentinel_from_tavily(self):
        """Blocked Tavily content should pass through without a second inspection pass."""
        from src.editor_tools import search_claim

        with patch(
            "src.tavily_utils.tavily_search_with_timeout",
            new_callable=AsyncMock,
            return_value="TOOL_BLOCKED: suspicious content",
        ):
            result = await search_claim.ainvoke({"query": "test query here"})
            assert result == "TOOL_BLOCKED: suspicious content"

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
        tools = _get_editor_tools()
        tool_names = [t.name for t in tools]
        assert "fetch_reference_content" in tool_names
        assert "search_claim" in tool_names


# =============================================================================
# Verdict Consistency & EDITOR_NOTES Tests (July 2026 editorial-loop hardening)
# =============================================================================


class TestEnforceVerdictConsistency:
    """The verdict must follow the editor's own findings."""

    def test_clean_approval_passes_through(self):
        from src.article_writer import _enforce_verdict_consistency

        feedback = {"verdict": "APPROVED", "factual_errors": [], "confidence": 0.6}
        assert _enforce_verdict_consistency(feedback)["verdict"] == "APPROVED"

    def test_approved_with_factual_errors_coerced_to_revise(self):
        from src.article_writer import _enforce_verdict_consistency

        feedback = {
            "verdict": "APPROVED",
            "factual_errors": [{"location": "X", "claim": "Y", "ground_truth": "Z"}],
        }
        assert _enforce_verdict_consistency(feedback)["verdict"] == "REVISE"

    def test_approved_with_broken_reference_coerced_to_revise(self):
        from src.article_writer import _enforce_verdict_consistency

        feedback = {
            "verdict": "APPROVED",
            "factual_errors": [],
            "reference_checks": [{"url": "https://x", "status": "broken"}],
        }
        assert _enforce_verdict_consistency(feedback)["verdict"] == "REVISE"

    def test_verified_reference_does_not_coerce(self):
        from src.article_writer import _enforce_verdict_consistency

        feedback = {
            "verdict": "APPROVED",
            "factual_errors": [],
            "reference_checks": [{"url": "https://x", "status": "verified"}],
        }
        assert _enforce_verdict_consistency(feedback)["verdict"] == "APPROVED"

    def test_revise_verdict_untouched(self):
        from src.article_writer import _enforce_verdict_consistency

        feedback = {"verdict": "REVISE", "factual_errors": []}
        assert _enforce_verdict_consistency(feedback)["verdict"] == "REVISE"

    def test_malformed_reference_entries_do_not_raise(self):
        from src.article_writer import _enforce_verdict_consistency

        feedback = {
            "verdict": "APPROVED",
            "factual_errors": [],
            "reference_checks": ["not-a-dict", None, 42],
        }
        assert _enforce_verdict_consistency(feedback)["verdict"] == "APPROVED"


class TestFlagFailedReferences:
    """URLs the editor watched fail must not survive in the published draft."""

    def test_failed_url_in_draft_appends_error_and_forces_revise(self):
        editor = _create_article_editor()
        editor._failed_urls = {"https://dead.example/quote"}
        draft = "Body text.\n\n### References\n1. [X](https://dead.example/quote)\n"
        feedback = {"verdict": "APPROVED", "factual_errors": []}

        result = editor._flag_failed_references(draft, feedback)

        assert result["verdict"] == "REVISE"
        assert len(result["factual_errors"]) == 1
        error = result["factual_errors"][0]
        assert error["location"] == "References"
        assert "https://dead.example/quote" in error["claim"]
        assert error["action"] == "correct"

    def test_failed_url_absent_from_draft_leaves_feedback_untouched(self):
        editor = _create_article_editor()
        editor._failed_urls = {"https://dead.example/quote"}
        feedback = {"verdict": "APPROVED", "factual_errors": []}

        result = editor._flag_failed_references("No links here.", feedback)

        assert result["verdict"] == "APPROVED"
        assert result["factual_errors"] == []

    def test_no_failed_urls_is_a_noop(self):
        editor = _create_article_editor()
        editor._failed_urls = set()
        feedback = {"verdict": "APPROVED", "factual_errors": []}

        assert editor._flag_failed_references("any", feedback) is feedback

    def test_non_dict_feedback_returned_as_is(self):
        editor = _create_article_editor()
        editor._failed_urls = {"https://dead.example/quote"}

        assert (
            editor._flag_failed_references("https://dead.example/quote", None) is None
        )

    def test_existing_claim_not_duplicated(self):
        editor = _create_article_editor()
        editor._failed_urls = {"https://dead.example/quote"}
        claim = (
            "References include a URL whose verification fetch failed: "
            "https://dead.example/quote"
        )
        feedback = {
            "verdict": "REVISE",
            "factual_errors": [{"location": "References", "claim": claim}],
        }

        result = editor._flag_failed_references(
            "see https://dead.example/quote", feedback
        )

        assert len(result["factual_errors"]) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_records_failure_sentinels(self):
        editor = _create_article_editor()
        editor._failed_urls = set()
        editor._url_cache = {}

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="FETCH_FAILED: HTTP 404")
        mock_tool.name = "fetch_reference_content"
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        await editor._execute_tool_calls(
            [
                {
                    "name": "fetch_reference_content",
                    "args": {"url": "https://dead.example/quote"},
                    "id": "call_1",
                }
            ]
        )

        assert editor._failed_urls == {"https://dead.example/quote"}

    @pytest.mark.asyncio
    async def test_execute_tool_calls_ignores_successful_fetch(self):
        editor = _create_article_editor()
        editor._failed_urls = set()
        editor._url_cache = {}

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value="Plenty of real page content")
        mock_tool.name = "fetch_reference_content"
        editor._tools_by_name = {"fetch_reference_content": mock_tool}

        await editor._execute_tool_calls(
            [
                {
                    "name": "fetch_reference_content",
                    "args": {"url": "https://alive.example/page"},
                    "id": "call_1",
                }
            ]
        )

        assert editor._failed_urls == set()


class TestStripEditorNotes:
    """The revision meta block must never survive into the published article."""

    def test_strips_closed_block(self):
        from src.article_writer import _strip_editor_notes

        article = (
            "# Title\n\nBody text.\n\n## References\n- https://x\n\n"
            "```EDITOR_NOTES\nCORRECTED: fixed P/E\nCONTESTED: kept phrasing — "
            "matches writing samples\n```\n"
        )
        stripped = _strip_editor_notes(article)
        assert "EDITOR_NOTES" not in stripped
        assert "CORRECTED" not in stripped
        assert stripped.startswith("# Title")
        assert "## References" in stripped

    def test_strips_unclosed_block_at_eof(self):
        from src.article_writer import _strip_editor_notes

        article = "# Title\n\nBody.\n\n```EDITOR_NOTES\nCORRECTED: item"
        stripped = _strip_editor_notes(article)
        assert "EDITOR_NOTES" not in stripped
        assert "Body." in stripped

    def test_noop_without_block(self):
        from src.article_writer import _strip_editor_notes

        article = "# Title\n\nBody with ```python\ncode\n``` fence.\n"
        assert _strip_editor_notes(article) == article

    def test_noop_on_empty_string(self):
        from src.article_writer import _strip_editor_notes

        assert _strip_editor_notes("") == ""

    @pytest.mark.asyncio
    async def test_edit_strips_notes_from_approved_return(self):
        """A draft carrying EDITOR_NOTES must come back clean from edit()."""
        from src.article_writer import ArticleWriter

        writer = ArticleWriter()
        editor = _create_article_editor()
        editor.llm = MagicMock()
        editor.review = AsyncMock(
            return_value={"verdict": "APPROVED", "confidence": 0.9}
        )

        draft = (
            "# Title\n\nBody.\n\n## References\n- https://x\n\n"
            "```EDITOR_NOTES\nCORRECTED: item\n```\n"
        )
        result, feedback = await editor.edit(
            writer=writer,
            article_draft=draft,
            ticker="TEST",
            company_name="Test Corp",
        )
        assert feedback["verdict"] == "APPROVED"
        assert "EDITOR_NOTES" not in result
        assert "Body." in result

    @pytest.mark.asyncio
    async def test_edit_coerces_approved_with_errors_and_revises(self):
        """APPROVED + listed factual errors must trigger a revision pass."""
        editor = _create_article_editor()
        editor.llm = MagicMock()

        call_count = 0

        async def mock_review(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Self-approves despite listing an error — must be coerced
                return {
                    "verdict": "APPROVED",
                    "factual_errors": [
                        {"location": "X", "claim": "Y", "ground_truth": "Z"}
                    ],
                    "confidence": 0.95,
                }
            return {"verdict": "APPROVED", "factual_errors": [], "confidence": 0.9}

        editor.review = mock_review
        writer = MagicMock()
        writer.revise.return_value = "# Revised\n\nClean."

        result, feedback = await editor.edit(
            writer=writer,
            article_draft="# Original\n\nWith an error.",
            ticker="TEST",
            company_name="Test Corp",
        )

        assert writer.revise.called
        assert feedback["verdict"] == "APPROVED"
        assert result == "# Revised\n\nClean."


class TestEditorContextWiring:
    """New context inputs reach the editor's fact-check context."""

    def test_valuation_context_included(self):
        editor = _create_article_editor()
        context = editor.build_fact_check_context(
            data_block="DATA",
            valuation_context="VALUATION DATA (scenario target suppressed): ...",
        )
        assert "=== VALUATION CONTEXT (as given to writer) ===" in context
        assert "scenario target suppressed" in context

    def test_valuation_context_omitted_when_none(self):
        editor = _create_article_editor()
        context = editor.build_fact_check_context(data_block="DATA")
        assert "VALUATION CONTEXT" not in context

    def test_writing_samples_header_used(self):
        editor = _create_article_editor()
        context = editor.build_fact_check_context(voice_samples="Sample prose.")
        assert "=== WRITING SAMPLES (Match This Voice) ===" in context
        assert "VOICE SAMPLES" not in context

    def test_writer_public_sample_loader(self, tmp_path):
        from src.article_writer import ArticleWriter

        (tmp_path / "sample.txt").write_text("Voiceful prose.", encoding="utf-8")
        writer = ArticleWriter(samples_dir=tmp_path)
        samples = writer.load_voice_samples(max_chars=1000)
        assert "Voiceful prose." in samples
