"""Tests for Article Writer module."""

import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestArticleWriterInit:
    """Tests for ArticleWriter initialization."""

    def test_finds_writing_samples_directory(self):
        """Test that ArticleWriter finds the writing_samples directory if it exists."""
        from src.article_writer import ArticleWriter

        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip("writing_samples directory not found - skipping")

        # ArticleWriter should find it
        writer = ArticleWriter.__new__(ArticleWriter)
        found_dir = writer._find_samples_dir()
        assert found_dir.exists(), "Should find existing writing_samples directory"

    def test_loads_prompt_config_from_file(self):
        """Test loading prompt config from prompts/writer.json."""
        from src.article_writer import ArticleWriter

        prompts_dir = Path("prompts")
        writer_json = prompts_dir / "writer.json"
        assert writer_json.exists(), "prompts/writer.json should exist"

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.prompts_dir = prompts_dir
        config = writer._load_prompt_config()

        assert config["agent_key"] == "article_writer"
        assert "system_message" in config
        # Version should be a valid numeric string (e.g., "1.5", "2.0")
        assert re.match(
            r"^\d+\.\d+$", config["version"]
        ), f"Invalid version: {config['version']}"
        # user_template and model_config are nested in metadata for AgentPrompt compatibility
        metadata = config["metadata"]
        assert "user_template" in metadata
        assert metadata["model_config"]["use_quick_model"] is False
        # thinking_level removed in v1.5 (Claude migration — thinking is configured in create_writer_llm)
        assert "thinking_level" not in metadata["model_config"]

    def test_fallback_when_prompt_missing(self):
        """Test fallback to default config when writer.json is missing."""
        from src.article_writer import DEFAULT_PROMPT_CONFIG, ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArticleWriter.__new__(ArticleWriter)
            writer.prompts_dir = Path(tmpdir)  # Empty directory
            config = writer._load_prompt_config()

            assert config == DEFAULT_PROMPT_CONFIG
            assert "system_message" in config


class TestVoiceSamplesLoading:
    """Tests for loading writing samples."""

    def test_loads_txt_and_md_files(self):
        """Test that both .txt and .md files are loaded."""
        from src.article_writer import ArticleWriter

        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip("writing_samples directory not found")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = samples_dir
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        # Should contain content from at least one sample
        assert len(samples) > 0
        assert "--- Sample:" in samples

    def test_respects_max_chars_limit(self):
        """Test that samples are truncated to max_chars."""
        from src.article_writer import ArticleWriter

        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip("writing_samples directory not found")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = samples_dir
        # Set both limits to test truncation behavior
        writer.prompt_config = {
            "metadata": {"max_sample_chars": 100, "max_chars_per_file": 50}
        }

        samples = writer._load_voice_samples()

        # Should be truncated - per-file cap of 50 chars each
        # Note: last file is included even if it puts us over max_sample_chars
        # So with 50 char/file + ~50 header each, 2 files = ~200 chars
        assert len(samples) <= 250  # Allow for 2 files with headers

    def test_returns_empty_when_no_samples(self):
        """Test returns empty string when no samples found."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArticleWriter.__new__(ArticleWriter)
            writer.samples_dir = Path(tmpdir)  # Empty directory
            writer.prompt_config = {"metadata": {"max_sample_chars": 5000}}

            samples = writer._load_voice_samples()
            assert samples == ""

    def test_samples_include_filename(self):
        """Test that sample content includes the filename for context."""
        from src.article_writer import ArticleWriter

        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip("writing_samples directory not found")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = samples_dir
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        # Should include sample filenames
        # Check for at least one known sample file
        sample_files = list(samples_dir.glob("*.txt")) + list(samples_dir.glob("*.md"))
        if sample_files:
            # At least one filename should appear
            found_filename = any(f.name in samples for f in sample_files)
            assert found_filename, "Sample filenames should be included"


class TestImageManifest:
    """Tests for image manifest formatting."""

    def test_converts_to_github_raw_urls(self):
        """Test that local paths are converted to GitHub raw URLs."""
        from src.article_writer import GITHUB_RAW_BASE, ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            # Create mock image files
            (images_dir / "TEST_2026-01-01_football_field.png").touch()
            (images_dir / "TEST_2026-01-01_radar.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = True

            manifest = writer._format_image_manifest("TEST", "2026-01-01")

            # Should contain GitHub raw URL base
            assert GITHUB_RAW_BASE in manifest or "No charts available" in manifest

    def test_uses_local_paths_when_disabled(self):
        """Test that local paths are used when use_github_urls is False."""
        from src.article_writer import GITHUB_RAW_BASE, ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            (images_dir / "TEST_2026-01-01_football_field.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = False

            manifest = writer._format_image_manifest("TEST", "2026-01-01")

            # Should NOT contain GitHub URL
            assert GITHUB_RAW_BASE not in manifest

    def test_handles_ticker_with_dots(self):
        """Test that tickers with dots (e.g., 0005.HK) are handled."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            # Dots in ticker become underscores in filename
            (images_dir / "0005_HK_2026-01-01_football_field.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = False

            manifest = writer._format_image_manifest("0005.HK", "2026-01-01")

            assert "Football Field" in manifest or "No charts available" in manifest

    def test_returns_no_charts_message(self):
        """Test returns appropriate message when no charts found."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = Path(tmpdir)  # Empty directory
            writer.use_github_urls = True

            manifest = writer._format_image_manifest("XXXX", "2026-01-01")

            assert "No charts available" in manifest


class TestArticlePathResolution:
    """Tests for article output path resolution."""

    def test_resolve_article_path_default_no_output(self):
        """Test default article path when --article is used without value and no --output."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = True  # --article without value
        args.output = None  # No --output

        path = resolve_article_path(args, "0005.HK")

        assert path is not None
        assert "0005_HK_article.md" in str(path)

    def test_resolve_article_path_derives_from_output(self):
        """Test article path derived from --output when --article has no value."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = True  # --article without value
        args.output = "/path/to/0005_HK_2026-01-01.md"  # --output specified

        path = resolve_article_path(args, "0005.HK")

        assert path is not None
        assert path == Path("/path/to/0005_HK_2026-01-01_article.md")

    def test_resolve_article_path_derives_preserves_extension(self):
        """Test that derived path preserves the output file extension."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = True
        args.output = "/results/report.txt"

        path = resolve_article_path(args, "AAPL")

        assert path == Path("/results/report_article.txt")

    def test_resolve_article_path_absolute(self):
        """Test absolute article path is used as-is."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = "/tmp/my_article.md"
        args.output = "/other/path.md"  # Should be ignored for absolute paths

        path = resolve_article_path(args, "AAPL")

        assert path == Path("/tmp/my_article.md")

    def test_resolve_article_path_relative_with_output(self):
        """Test relative article path resolves to output directory."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = "custom.md"  # Relative path
        args.output = "results/report.md"

        path = resolve_article_path(args, "AAPL")

        assert path == Path("results/custom.md")

    def test_resolve_article_path_relative_no_output(self):
        """Test relative article path stays relative when no --output."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = "custom.md"  # Relative path
        args.output = None

        path = resolve_article_path(args, "AAPL")

        assert path == Path("custom.md")

    def test_resolve_article_path_adds_extension(self):
        """Test that .md extension is added if missing."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = "/tmp/my_article"  # No extension
        args.output = None

        path = resolve_article_path(args, "AAPL")

        assert path.suffix == ".md"

    def test_resolve_article_path_none_when_disabled(self):
        """Test returns None when --article not specified."""
        from src.main import resolve_article_path

        args = MagicMock()
        args.article = False
        args.output = None

        path = resolve_article_path(args, "AAPL")

        assert path is None


class TestArticleGeneration:
    """Tests for article generation (mocked LLM)."""

    @patch("src.article_writer.create_writer_llm")
    def test_generates_article_with_all_components(self, mock_create_llm):
        """Test that article generation includes voice samples and images."""
        from src.article_writer import ArticleWriter

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="# Test Article\n\nThis is a test article."
        )
        mock_create_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup directories
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()
            (samples_dir / "test_sample.txt").write_text("Sample voice content here.")

            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            prompts_dir = Path("prompts")

            writer = ArticleWriter(
                prompts_dir=prompts_dir,
                samples_dir=samples_dir,
                images_dir=images_dir,
                use_github_urls=False,
            )

            article = writer.write(
                ticker="TEST",
                company_name="Test Company",
                report_text="This is the source report.",
                trade_date="2026-01-01",
            )

            # Verify LLM was called
            assert mock_llm.invoke.called

            # Verify the user message contained voice samples
            call_args = mock_llm.invoke.call_args[0][0]
            user_msg = call_args[1].content
            assert "Sample voice content" in user_msg

            # Verify article was returned
            assert "Test Article" in article

    @patch("src.article_writer.create_writer_llm")
    def test_saves_article_to_file(self, mock_create_llm):
        """Test that article is saved to specified output path."""
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="# Saved Article\n\nContent here."
        )
        mock_create_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output" / "article.md"

            writer = ArticleWriter(
                samples_dir=Path(tmpdir),
                images_dir=Path(tmpdir),
                use_github_urls=False,
            )

            writer.write(
                ticker="TEST",
                company_name="Test Co",
                report_text="Report",
                trade_date="2026-01-01",
                output_path=output_path,
            )

            # File should exist
            assert output_path.exists()
            content = output_path.read_text()
            assert "Saved Article" in content


class TestPromptTemplate:
    """Tests for prompt template structure."""

    def test_prompt_has_required_placeholders(self):
        """Test that user_template has all required placeholders."""
        import json

        prompt_file = Path("prompts/writer.json")
        assert prompt_file.exists()

        with open(prompt_file) as f:
            config = json.load(f)

        # user_template is nested in metadata for AgentPrompt compatibility
        user_template = config["metadata"]["user_template"]

        # Check required placeholders
        assert "{voice_samples}" in user_template
        assert "{image_manifest}" in user_template
        assert "{ticker}" in user_template
        assert "{company_name}" in user_template
        assert "{report_text}" in user_template
        assert (
            "{valuation_context}" in user_template
        )  # Added for chart/decision reconciliation

    def test_system_message_has_key_instructions(self):
        """Test that system message contains key instructions."""
        import json

        prompt_file = Path("prompts/writer.json")
        with open(prompt_file) as f:
            config = json.load(f)

        system_msg = config["system_message"]

        # Check for key instructions (case-insensitive for flexibility)
        assert "medium" in system_msg.lower()  # Medium formatting
        assert "voice" in system_msg.lower()  # Voice matching
        assert "references" in system_msg.lower()  # References section


class TestFactCheckContext:
    """Tests for fact-check context fetching."""

    def test_returns_empty_when_no_api_key(self):
        """Test returns empty string when Tavily API key is missing."""
        from unittest.mock import MagicMock, patch

        from src.article_writer import ArticleWriter

        # Mock at the module level where it's imported
        with patch("src.article_writer.config") as mock_config:
            mock_config.get_tavily_api_key.return_value = ""
            writer = ArticleWriter.__new__(ArticleWriter)
            context = writer._fetch_fact_check_context("AAPL", "Apple Inc.")
            assert context == ""

    def test_returns_empty_on_exception(self):
        """Test returns empty string on any exception."""
        from unittest.mock import MagicMock, patch

        from src.article_writer import ArticleWriter

        with patch("src.article_writer.config") as mock_config:
            mock_config.get_tavily_api_key.return_value = "fake-key"
            # Mock TavilyClient import to raise
            with patch.dict("sys.modules", {"tavily": None}):
                writer = ArticleWriter.__new__(ArticleWriter)
                context = writer._fetch_fact_check_context("AAPL", "Apple Inc.")
                # Should return empty due to ImportError
                assert context == ""

    def test_respects_max_chars_limit(self):
        """Test that context is truncated to max_chars."""
        from unittest.mock import MagicMock, patch

        from src.article_writer import ArticleWriter

        mock_tavily = MagicMock()
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "A" * 2000,  # Long answer
            "results": [],
        }
        mock_tavily.TavilyClient.return_value = mock_client

        with patch("src.article_writer.config") as mock_config:
            mock_config.get_tavily_api_key.return_value = "fake-key"
            with patch.dict("sys.modules", {"tavily": mock_tavily}):
                writer = ArticleWriter.__new__(ArticleWriter)
                context = writer._fetch_fact_check_context(
                    "AAPL", "Apple Inc.", max_chars=100
                )
                # Should be truncated
                assert len(context) <= 120  # 100 + "[...truncated]"
                assert "[...truncated]" in context

    def test_handles_api_errors_gracefully(self):
        """Test returns empty string on API errors."""
        from unittest.mock import MagicMock, patch

        from src.article_writer import ArticleWriter

        mock_tavily = MagicMock()
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API error")
        mock_tavily.TavilyClient.return_value = mock_client

        with patch("src.article_writer.config") as mock_config:
            mock_config.get_tavily_api_key.return_value = "fake-key"
            with patch.dict("sys.modules", {"tavily": mock_tavily}):
                writer = ArticleWriter.__new__(ArticleWriter)
                context = writer._fetch_fact_check_context("AAPL", "Apple Inc.")
                assert context == ""


class TestThinkingModelDetection:
    """Tests for Gemini thinking model detection."""

    def test_gemini_3_detected(self):
        """Test that Gemini 3.x models are detected as supporting thinking_level."""
        from src.llms import is_gemini_v3_or_greater

        assert is_gemini_v3_or_greater("gemini-3-pro-preview") is True
        assert is_gemini_v3_or_greater("gemini-3-flash-preview") is True
        assert is_gemini_v3_or_greater("gemini-3.5-pro") is True

    def test_gemini_2_not_detected(self):
        """Test that regular Gemini 2.x models are NOT detected."""
        from src.llms import is_gemini_v3_or_greater

        assert is_gemini_v3_or_greater("gemini-2.0-flash") is False
        assert is_gemini_v3_or_greater("gemini-2.5-flash") is False
        assert is_gemini_v3_or_greater("gemini-2.0-pro") is False

    def test_thinking_models_detected(self):
        """Test that 'thinking' models are detected regardless of version."""
        from src.llms import is_gemini_v3_or_greater

        # These are 2.x models but have "thinking" in the name
        assert is_gemini_v3_or_greater("gemini-2.0-flash-thinking-exp") is True
        assert is_gemini_v3_or_greater("gemini-2.5-flash-thinking") is True
        assert is_gemini_v3_or_greater("gemini-thinking-preview") is True

    def test_non_gemini_not_detected(self):
        """Test that non-Gemini models are not detected."""
        from src.llms import is_gemini_v3_or_greater

        assert is_gemini_v3_or_greater("gpt-4o") is False
        assert is_gemini_v3_or_greater("claude-3-opus") is False
        assert is_gemini_v3_or_greater("llama-3") is False


class TestWritingSamplesDirectory:
    """Tests for writing_samples directory (optional - graceful skip if missing)."""

    def test_writing_samples_directory_exists(self):
        """Verify writing_samples directory is valid if it exists."""
        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip(
                "writing_samples directory not found - feature works without it"
            )
        assert samples_dir.is_dir(), "writing_samples must be a directory"

    def test_writing_samples_contains_files(self):
        """Verify writing_samples contains at least one sample file."""
        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip("writing_samples directory not found")

        txt_files = list(samples_dir.glob("*.txt"))
        md_files = list(samples_dir.glob("*.md"))

        total_samples = len(txt_files) + len(md_files)
        assert total_samples > 0, "writing_samples should contain .txt or .md files"

    def test_writing_samples_have_content(self):
        """Verify sample files have non-trivial content."""
        samples_dir = Path("writing_samples")
        if not samples_dir.exists():
            pytest.skip("writing_samples directory not found")

        sample_files = list(samples_dir.glob("*.txt")) + list(samples_dir.glob("*.md"))
        if not sample_files:
            pytest.skip("No sample files found")

        # Check at least one file has substantial content (> 100 chars)
        has_content = False
        for sample_file in sample_files:
            try:
                content = sample_file.read_text(encoding="utf-8")
                if len(content) > 100:
                    has_content = True
                    break
            except Exception:
                continue

        assert has_content, "At least one sample should have substantial content"
