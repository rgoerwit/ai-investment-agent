"""Tests for Article Writer module."""

import asyncio
import re
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

        # Should contain content from at least one sample. The real
        # writing_samples/ dir carries a distilled style_profile.md, which
        # gets a distinct header from raw prose samples (see
        # _load_voice_samples) — accept either.
        assert len(samples) > 0
        assert (
            "--- Writing Sample:" in samples or "=== DISTILLED STYLE PROFILE" in samples
        )

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
        # Note: last file is included even if it puts us over max_sample_chars.
        # When a style_profile.md is present it's always the first file loaded
        # (see _load_voice_samples), so with 50 char/file + the longer
        # descriptive headers used to distinguish profile vs. raw samples,
        # 2 files land around 300 chars.
        assert len(samples) <= 350  # Allow for 2 files with headers

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


class TestVoiceSampleProfilePriority:
    """Hermetic tests locking profile priority, marker fix, and distinct headers."""

    def test_marker_matches_prompt_literal(self, tmp_path):
        """The marker the prompt quotes appears verbatim as its own line.

        Drift guard: the marker is EXTRACTED from prompts/writer.json, not
        hardcoded — if the prompt renames/drops the quoted marker or the
        loader stops emitting it, this fails.
        """
        import json

        from src.article_writer import ArticleWriter

        prompt = json.loads(Path("prompts/writer.json").read_text(encoding="utf-8"))
        match = re.search(
            r'block marked "(=== DISTILLED STYLE PROFILE ===)"',
            prompt["system_message"],
        )
        assert match, "writer.json no longer quotes the profile marker"
        marker = match.group(1)

        # Create profile and one raw sample
        profile_file = tmp_path / "style_profile.md"
        profile_file.write_text("Profile content here", encoding="utf-8")
        raw_file = tmp_path / "sample1.txt"
        raw_file.write_text("Raw sample", encoding="utf-8")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = tmp_path
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        # The prompt's exact marker must appear as a standalone line
        marker_lines = [line for line in samples.split("\n") if line.strip() == marker]
        assert (
            len(marker_lines) == 1
        ), "Marker should appear exactly once on its own line"

    def test_profile_appears_first(self, tmp_path):
        """Profile block appears before raw writing samples."""
        from src.article_writer import ArticleWriter

        profile_file = tmp_path / "style_profile.md"
        profile_file.write_text("Profile", encoding="utf-8")
        raw_file = tmp_path / "sample1.txt"
        raw_file.write_text("Raw", encoding="utf-8")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = tmp_path
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        profile_pos = samples.find("=== DISTILLED STYLE PROFILE")
        raw_pos = samples.find("--- Writing Sample:")
        assert profile_pos != -1, "Profile should be present"
        assert raw_pos != -1, "Raw sample should be present"
        assert profile_pos < raw_pos, "Profile must appear before raw samples"

    def test_raw_pool_capped_at_constant(self, tmp_path):
        """Raw writing samples limited to RAW_SAMPLES_WHEN_PROFILE_PRESENT."""
        from src.article_writer import RAW_SAMPLES_WHEN_PROFILE_PRESENT, ArticleWriter

        profile_file = tmp_path / "style_profile.md"
        profile_file.write_text("Profile", encoding="utf-8")

        # Create 6 raw files (more than the constant)
        for i in range(6):
            (tmp_path / f"sample{i}.txt").write_text(f"Sample {i}", encoding="utf-8")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = tmp_path
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        # Count raw sample blocks (should be capped)
        raw_count = samples.count("--- Writing Sample:")
        assert raw_count == RAW_SAMPLES_WHEN_PROFILE_PRESENT

    def test_profile_not_in_raw_pool(self, tmp_path):
        """Profile appears exactly once, never in the raw sample count."""
        from src.article_writer import ArticleWriter

        profile_file = tmp_path / "style_profile.md"
        profile_file.write_text("Profile", encoding="utf-8")
        (tmp_path / "sample1.txt").write_text("Sample", encoding="utf-8")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = tmp_path
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        # Profile marker should appear exactly once
        profile_count = samples.count("=== DISTILLED STYLE PROFILE ===")
        assert profile_count == 1

    def test_distinct_headers_profile_vs_raw(self, tmp_path):
        """Profile and raw samples use distinct headers."""
        from src.article_writer import ArticleWriter

        profile_file = tmp_path / "style_profile.md"
        profile_file.write_text("Profile", encoding="utf-8")
        raw_file = tmp_path / "sample1.txt"
        raw_file.write_text("Raw", encoding="utf-8")

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.samples_dir = tmp_path
        writer.prompt_config = {"metadata": {"max_sample_chars": 50000}}

        samples = writer._load_voice_samples()

        # Profile uses "=== ... ===" style, raw uses "--- ... ---" style
        assert "=== DISTILLED STYLE PROFILE ===" in samples
        assert "--- Writing Sample:" in samples
        # They should not use each other's format
        assert "--- DISTILLED" not in samples


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

    def test_local_paths_are_relative_to_article_directory(self, tmp_path, monkeypatch):
        """A nested article must not repeat its parent directory in image links."""
        from src.article_writer import ArticleWriter

        monkeypatch.chdir(tmp_path)
        article_dir = Path("scratch")
        images_dir = article_dir / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "TEST_radar.png").touch()

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.images_dir = images_dir
        writer.use_github_urls = False

        manifest = writer._format_image_manifest(
            "TEST",
            "2026-07-25",
            article_dir=article_dir,
        )

        assert "URL: images/TEST_radar.png" in manifest
        assert "URL: scratch/images/" not in manifest

    def test_local_paths_support_image_directory_outside_article_tree(self, tmp_path):
        """Custom image directories retain a portable relative link."""
        from src.article_writer import ArticleWriter

        article_dir = tmp_path / "reports"
        images_dir = tmp_path / "charts"
        article_dir.mkdir()
        images_dir.mkdir()
        (images_dir / "TEST_radar.png").touch()

        writer = ArticleWriter.__new__(ArticleWriter)
        writer.images_dir = images_dir
        writer.use_github_urls = False

        manifest = writer._format_image_manifest(
            "TEST",
            "2026-07-25",
            article_dir=article_dir,
        )

        assert "URL: ../charts/TEST_radar.png" in manifest

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

    def test_handles_raw_ticker_filename(self):
        """Test that charts saved with raw ticker (e.g., 2767.T_radar.png) are found."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            # Chart generator uses raw ticker as filename_stem
            (images_dir / "2767.T_football_field.png").touch()
            (images_dir / "2767.T_radar.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = False

            manifest = writer._format_image_manifest("2767.T", "2026-02-07")

            assert "Football Field" in manifest
            assert "Radar" in manifest

    def test_returns_no_charts_message(self):
        """Test returns appropriate message when no charts found."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = Path(tmpdir)  # Empty directory
            writer.use_github_urls = True

            manifest = writer._format_image_manifest("XXXX", "2026-01-01")

            assert "No charts available" in manifest

    def test_chart_paths_excludes_stale_suppressed_chart(self):
        """A stale football-field file (run suppressed it, e.g. DNI/SELL) must
        not resurface in the manifest when chart_paths from this run omit it."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            (images_dir / "2767.T_football_field.png").touch()  # stale, prior run
            (images_dir / "2767.T_radar.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = False

            manifest = writer._format_image_manifest(
                "2767.T",
                "2026-02-07",
                chart_paths={"radar": str(images_dir / "2767.T_radar.png")},
            )

            assert "Radar" in manifest
            assert "Football Field" not in manifest

    def test_chart_paths_includes_fresh_charts(self):
        """Charts listed in chart_paths appear in the manifest normally."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            (images_dir / "2767.T_football_field.png").touch()
            (images_dir / "2767.T_radar.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = False

            manifest = writer._format_image_manifest(
                "2767.T",
                "2026-02-07",
                chart_paths={
                    "football_field": str(images_dir / "2767.T_football_field.png"),
                    "radar": str(images_dir / "2767.T_radar.png"),
                },
            )

            assert "Football Field" in manifest
            assert "Radar" in manifest

    def test_chart_paths_none_or_empty_keeps_legacy_glob(self):
        """None/{} chart_paths follow the legacy disk glob (mirrors
        report_generator's empty-dict fallback convention)."""
        from src.article_writer import ArticleWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            (images_dir / "2767.T_football_field.png").touch()

            writer = ArticleWriter.__new__(ArticleWriter)
            writer.images_dir = images_dir
            writer.use_github_urls = False

            for chart_paths in (None, {}):
                manifest = writer._format_image_manifest(
                    "2767.T", "2026-02-07", chart_paths=chart_paths
                )
                assert "Football Field" in manifest


class TestArticlePathResolution:
    """Tests for article output path resolution."""

    def test_resolve_article_path_default_no_output(self):
        """Test default article path when --article is used without value and no --output."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = True  # --article without value
        args.output = None  # No --output

        path = resolve_article_path(args, "0005.HK")

        assert path is not None
        assert "0005_HK_article.md" in str(path)

    def test_resolve_article_path_derives_from_output(self):
        """Test article path derived from --output when --article has no value."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = True  # --article without value
        args.output = "/path/to/0005_HK_2026-01-01.md"  # --output specified

        path = resolve_article_path(args, "0005.HK")

        assert path is not None
        assert path == Path("/path/to/0005_HK_2026-01-01_article.md")

    def test_resolve_article_path_derives_preserves_extension(self):
        """Test that derived path preserves the output file extension."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = True
        args.output = "/results/report.txt"

        path = resolve_article_path(args, "AAPL")

        assert path == Path("/results/report_article.txt")

    def test_resolve_article_path_absolute(self):
        """Test absolute article path is used as-is."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = "/tmp/my_article.md"
        args.output = "/other/path.md"  # Should be ignored for absolute paths

        path = resolve_article_path(args, "AAPL")

        assert path == Path("/tmp/my_article.md")

    def test_resolve_article_path_relative_with_output(self):
        """Explicit relative article paths stay relative to cwd."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = "custom.md"  # Relative path
        args.output = "results/report.md"

        path = resolve_article_path(args, "AAPL")

        assert path == Path("custom.md")

    def test_resolve_article_path_relative_no_output(self):
        """Test relative article path stays relative when no --output."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = "custom.md"  # Relative path
        args.output = None

        path = resolve_article_path(args, "AAPL")

        assert path == Path("custom.md")

    def test_resolve_article_path_adds_extension(self):
        """Test that .md extension is added if missing."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = "/tmp/my_article"  # No extension
        args.output = None

        path = resolve_article_path(args, "AAPL")

        assert path.suffix == ".md"

    def test_resolve_article_path_none_when_disabled(self):
        """Test returns None when --article not specified."""
        from src.cli import resolve_article_path

        args = MagicMock()
        args.article = False
        args.output = None

        path = resolve_article_path(args, "AAPL")

        assert path is None


class TestArticleGeneration:
    """Tests for article generation (mocked LLM)."""

    @patch("src.article_writer.create_writer_llm")
    def test_article_writer_passes_extra_callbacks_to_llm(self, mock_create_llm):
        from src.article_writer import ArticleWriter

        mock_create_llm.return_value = MagicMock()
        tracing_callback = MagicMock()

        ArticleWriter(
            samples_dir=Path("writing_samples")
            if Path("writing_samples").exists()
            else None,
            callbacks=[tracing_callback],
        )

        callbacks = mock_create_llm.call_args.kwargs["callbacks"]
        assert tracing_callback in callbacks

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
            (images_dir / "TEST_radar.png").touch()

            prompts_dir = Path("prompts")
            output_path = Path(tmpdir) / "article.md"

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
                output_path=output_path,
            )

            # Verify LLM was called
            assert mock_llm.invoke.called

            # Verify the user message contained voice samples
            call_args = mock_llm.invoke.call_args[0][0]
            user_msg = call_args[1].content
            assert "Sample voice content" in user_msg
            assert "URL: images/TEST_radar.png" in user_msg

            # Verify article was returned
            assert "Test Article" in article
            assert output_path.read_text() == article

    @patch("src.article_writer.create_writer_llm")
    def test_writer_injects_governance_card(self, mock_create_llm):
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Test Article\n\nBody.")
        mock_create_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArticleWriter(
                samples_dir=Path(tmpdir),
                images_dir=Path(tmpdir),
                use_github_urls=False,
            )

            writer.write(
                ticker="009970.KS",
                company_name="Youngone Holdings Co., Ltd.",
                report_text="Report",
                trade_date="2026-01-01",
                governance_card={
                    "ticker": "009970.KS",
                    "canonical_name": "Youngone Holdings Co., Ltd.",
                    "entity_role": "INTERMEDIATE_HOLDCO",
                    "confidence": "clean",
                    "related_listed": [{"ticker": "111770.KS"}],
                },
            )

            user_msg = mock_llm.invoke.call_args[0][0][1].content
            assert "ENTITY GOVERNANCE CARD" in user_msg
            assert "111770.KS" in user_msg
            assert "MANDATORY OPENING DISCLOSURE" in user_msg

    @patch("src.article_writer.create_writer_llm")
    def test_writer_disclosure_directive_requires_nonstandard_structure(
        self, mock_create_llm
    ):
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Test Article\n\nBody.")
        mock_create_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ArticleWriter(
                samples_dir=Path(tmpdir),
                images_dir=Path(tmpdir),
                use_github_urls=False,
            )

            writer.write(
                ticker="7203.T",
                company_name="Toyota Motor Corporation",
                report_text="Report",
                trade_date="2026-01-01",
                governance_card={
                    "ticker": "7203.T",
                    "canonical_name": "Toyota Motor Corporation",
                    "entity_role": "STANDALONE",
                    "confidence": "clean",
                    "related_listed": [],
                },
            )

            user_msg = mock_llm.invoke.call_args[0][0][1].content
            assert "ENTITY GOVERNANCE CARD" in user_msg
            assert "MANDATORY OPENING DISCLOSURE" not in user_msg

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
        """Test returns empty string when Tavily search is unavailable."""
        from unittest.mock import patch

        from src.article_writer import ArticleWriter

        with patch("src.article_writer.search_tavily_sync_inspected") as mock_search:
            mock_search.return_value = None
            writer = ArticleWriter.__new__(ArticleWriter)
            context = writer._fetch_fact_check_context("AAPL", "Apple Inc.")
            assert context == ""

    def test_returns_empty_on_exception(self):
        """Test returns empty string on any exception."""
        from unittest.mock import patch

        from src.article_writer import ArticleWriter

        with patch(
            "src.article_writer.search_tavily_sync_inspected",
            side_effect=Exception("boom"),
        ):
            writer = ArticleWriter.__new__(ArticleWriter)
            context = writer._fetch_fact_check_context("AAPL", "Apple Inc.")
            assert context == ""

    def test_respects_max_chars_limit(self):
        """Test that context is truncated to max_chars."""
        from unittest.mock import patch

        from src.article_writer import ArticleWriter

        response = {
            "answer": "A" * 2000,  # Long answer
            "results": [],
        }

        with patch("src.article_writer.search_tavily_sync_inspected") as mock_search:
            mock_search.return_value = response
            writer = ArticleWriter.__new__(ArticleWriter)
            context = writer._fetch_fact_check_context(
                "AAPL", "Apple Inc.", max_chars=100
            )
            assert len(context) <= 120  # 100 + "[...truncated]"
            assert "[...truncated]" in context

    def test_handles_api_errors_gracefully(self):
        """Test returns empty string on API errors."""
        from unittest.mock import patch

        from src.article_writer import ArticleWriter

        with patch(
            "src.article_writer.search_tavily_sync_inspected",
            side_effect=Exception("API error"),
        ):
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


def _samples_dir():
    return Path("writing_samples") if Path("writing_samples").exists() else None


def _billing_error() -> Exception:
    return Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
    )


def _mock_llm(model: str, response_text: str | None = None, error=None):
    """LLM mock with a real model-name string so provider inference works."""
    llm = MagicMock()
    llm.model = model
    if error is not None:
        llm.invoke.side_effect = error
    else:
        llm.invoke.return_value = MagicMock(content=response_text)
    return llm


def _tier(label: str, llm):
    from src.llms import WriterTier

    return WriterTier(label, lambda: llm)


class _RecordingLogger:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def _record(self, event, **kwargs):
        self.events.append((event, kwargs))

    debug = info = warning = error = _record

    def named(self, event: str) -> list[dict]:
        return [kwargs for name, kwargs in self.events if name == event]


class TestWriterFallbackChainRuntime:
    """Runtime Claude-error path: iterate the writer fallback chain.

    Chain order (EDITOR_MODEL → Gemini floor) is covered at construction level
    in tests/test_llms_writer_fallback.py; here the chain is injected so the
    loop's behavior — recovery, caching, logging, output — is tested directly.
    """

    def _writer(self, mock_create_writer, primary):
        from src.article_writer import ArticleWriter

        mock_create_writer.return_value = primary
        return ArticleWriter(samples_dir=_samples_dir())

    @patch("src.article_writer.create_writer_llm")
    def test_billing_error_recovers_on_first_tier(self, mock_create_writer):
        """Claude billing error → EDITOR_MODEL tier writes the article."""
        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", "# Fallback Article\n\nWritten by GPT.")
        gemini = _mock_llm("gemini-3.5-flash", "# Should not be used.")
        writer = self._writer(mock_create_writer, primary)

        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[
                _tier("editor_model", gpt),
                _tier("gemini_last_resort", gemini),
            ],
        ):
            article = writer._invoke_writer([MagicMock()])

        assert "Fallback Article" in article
        assert writer.writer_fell_back is True
        assert writer.current_model_name == "gpt-5.4"
        gemini.invoke.assert_not_called()

    @patch("src.article_writer.create_writer_llm")
    def test_first_tier_failure_recovers_on_gemini_floor(self, mock_create_writer):
        """EDITOR_MODEL tier also fails → Gemini floor still produces the article."""
        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", error=Exception("openai 503"))
        gemini = _mock_llm("gemini-3.5-flash", "# Floor Article\n\nWritten by Gemini.")
        writer = self._writer(mock_create_writer, primary)

        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[
                _tier("editor_model", gpt),
                _tier("gemini_last_resort", gemini),
            ],
        ):
            article = writer._invoke_writer([MagicMock()])

        assert "Floor Article" in article
        assert writer.writer_fell_back is True
        assert writer.current_model_name == "gemini-3.5-flash"
        assert gpt.invoke.call_count == 1

    @patch("src.article_writer.create_writer_llm")
    def test_all_tiers_fail_raises_last_exception(self, mock_create_writer):
        """No silent empty article: every tier failing re-raises the last error."""
        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", error=Exception("openai 503"))
        gemini = _mock_llm("gemini-3.5-flash", error=Exception("gemini floor down"))
        writer = self._writer(mock_create_writer, primary)

        with (
            patch(
                "src.article_writer.writer_fallback_chain",
                return_value=[
                    _tier("editor_model", gpt),
                    _tier("gemini_last_resort", gemini),
                ],
            ),
            pytest.raises(Exception, match="gemini floor down"),
        ):
            writer._invoke_writer([MagicMock()])

    @patch("src.article_writer.create_writer_llm")
    def test_non_anthropic_errors_propagate_without_chain(self, mock_create_writer):
        """Non-Anthropic primary errors must NOT trigger the fallback chain."""
        primary = _mock_llm("claude-opus-4-8", error=ValueError("Some unrelated error"))
        writer = self._writer(mock_create_writer, primary)

        with patch("src.article_writer.writer_fallback_chain") as mock_chain:
            with pytest.raises(ValueError, match="Some unrelated error"):
                writer._invoke_writer([MagicMock()])
        mock_chain.assert_not_called()
        assert writer.writer_fell_back is False

    @patch("src.article_writer.create_writer_llm")
    def test_winning_tier_cached_for_subsequent_calls(self, mock_create_writer):
        """After recovery the winning tier is cached — Claude is not retried."""
        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", "# Article\n\nContent.")
        writer = self._writer(mock_create_writer, primary)

        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[_tier("editor_model", gpt)],
        ) as mock_chain:
            writer._invoke_writer([MagicMock()])
            writer._invoke_writer([MagicMock()])

        assert primary.invoke.call_count == 1
        assert gpt.invoke.call_count == 2
        mock_chain.assert_called_once()

    @patch("src.article_writer.create_writer_llm")
    def test_fallback_invoke_preserves_writer_persona_messages(
        self, mock_create_writer
    ):
        """The fallback swaps only the LLM — the writer's messages (persona
        SystemMessage included) reach the fallback tier unchanged."""
        from langchain_core.messages import SystemMessage

        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", "# Article\n\nContent.")
        writer = self._writer(mock_create_writer, primary)
        messages = [SystemMessage(content="writer persona"), MagicMock()]

        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[_tier("editor_model", gpt)],
        ):
            writer._invoke_writer(messages)

        assert gpt.invoke.call_args.args[0] is messages

    @patch("src.article_writer.create_writer_llm")
    def test_fallback_tiers_receive_tracking_callbacks(self, mock_create_writer):
        """Fallback tokens must be attributed — the chain is built with the
        writer's TokenTrackingCallback (the old single-hop passed none)."""
        from src.token_tracker import TokenTrackingCallback

        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", "# Article\n\nContent.")
        writer = self._writer(mock_create_writer, primary)

        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[_tier("editor_model", gpt)],
        ) as mock_chain:
            writer._invoke_writer([MagicMock()])

        callbacks = mock_chain.call_args.kwargs["callbacks"]
        assert isinstance(callbacks[0], TokenTrackingCallback)
        assert callbacks[0].agent_name == "Article Writer"

    @patch("src.article_writer.create_writer_llm")
    def test_recovery_logging_is_family_neutral_and_complete(
        self, mock_create_writer, monkeypatch
    ):
        """Success path emits primary-failed + attempt + succeeded with the tier
        label and real model/provider in structured fields — never a model
        family baked into a fallback event name."""
        import src.article_writer as aw

        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", "# Article\n\nContent.")
        writer = self._writer(mock_create_writer, primary)

        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[_tier("editor_model", gpt)],
        ):
            writer._invoke_writer([MagicMock()])

        assert len(recorder.named("claude_writer_primary_failed")) == 1
        attempts = recorder.named("writer_fallback_attempt")
        assert [a["tier"] for a in attempts] == ["editor_model"]
        assert attempts[0]["fallback_model"] == "gpt-5.4"
        assert attempts[0]["fallback_provider"] == "openai"
        succeeded = recorder.named("writer_fallback_succeeded")
        assert [s["tier"] for s in succeeded] == ["editor_model"]
        fallback_events = [name for name, _ in recorder.events if "fallback" in name]
        for name in fallback_events:
            assert not any(
                family in name for family in ("gemini", "gpt", "claude", "openai")
            ), f"family token in fallback event name: {name}"

    @patch("src.article_writer.create_writer_llm")
    def test_tier_failure_logging_then_recovery(self, mock_create_writer, monkeypatch):
        """A failing tier logs writer_fallback_attempt_failed and the loop
        advances — events tell the full story in order."""
        import src.article_writer as aw

        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gpt = _mock_llm("gpt-5.4", error=Exception("openai 503"))
        gemini = _mock_llm("gemini-3.5-flash", "# Article\n\nContent.")
        writer = self._writer(mock_create_writer, primary)

        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[
                _tier("editor_model", gpt),
                _tier("gemini_last_resort", gemini),
            ],
        ):
            writer._invoke_writer([MagicMock()])

        failed = recorder.named("writer_fallback_attempt_failed")
        assert [f["tier"] for f in failed] == ["editor_model"]
        assert failed[0]["fallback_model"] == "gpt-5.4"
        succeeded = recorder.named("writer_fallback_succeeded")
        assert [s["tier"] for s in succeeded] == ["gemini_last_resort"]
        assert succeeded[0]["fallback_provider"] == "google"

    @patch("src.article_writer.create_writer_llm")
    def test_tier_build_failure_is_contained_and_loop_advances(
        self, mock_create_writer, monkeypatch
    ):
        """A tier whose build() raises (availability race) is logged with empty
        model fields — no NameError — and the next tier still recovers."""
        import src.article_writer as aw
        from src.llms import WriterTier

        primary = _mock_llm("claude-opus-4-8", error=_billing_error())
        gemini = _mock_llm("gemini-3.5-flash", "# Article\n\nContent.")

        def _broken_build():
            raise RuntimeError("OpenAI writer fallback tier unavailable")

        writer = self._writer(mock_create_writer, primary)
        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)
        with patch(
            "src.article_writer.writer_fallback_chain",
            return_value=[
                WriterTier("editor_model", _broken_build),
                _tier("gemini_last_resort", gemini),
            ],
        ):
            article = writer._invoke_writer([MagicMock()])

        assert "Content." in article
        failed = recorder.named("writer_fallback_attempt_failed")
        assert [f["tier"] for f in failed] == ["editor_model"]
        assert failed[0]["fallback_model"] == ""
        assert failed[0]["fallback_provider"] == "unknown"
        assert writer.writer_fell_back is True

    @patch("src.article_writer.create_writer_llm")
    def test_primary_success_keeps_fell_back_flag_false(self, mock_create_writer):
        primary = _mock_llm("claude-opus-4-8", "# Article\n\nContent.")
        writer = self._writer(mock_create_writer, primary)

        with patch("src.article_writer.writer_fallback_chain") as mock_chain:
            writer._invoke_writer([MagicMock()])

        assert writer.writer_fell_back is False
        mock_chain.assert_not_called()

    @patch("src.article_writer.create_writer_llm")
    def test_truncated_output_raises_error(self, mock_create_writer):
        """MAX_TOKENS/LENGTH finish_reason raises RuntimeError, not silent skip."""
        primary = _mock_llm("claude-opus-4-8", "# Article\n\n[truncated]")
        primary.invoke.return_value.response_metadata = {"finish_reason": "MAX_TOKENS"}
        writer = self._writer(mock_create_writer, primary)

        with pytest.raises(RuntimeError, match="output token limit"):
            writer._invoke_writer([MagicMock()])

    @patch("src.article_writer.create_writer_llm")
    def test_normal_finish_reason_does_not_raise(self, mock_create_writer):
        """STOP/END_TURN finish_reason succeeds normally."""
        primary = _mock_llm("claude-opus-4-8", "# Article\n\nComplete.")
        primary.invoke.return_value.response_metadata = {"finish_reason": "STOP"}
        writer = self._writer(mock_create_writer, primary)

        article = writer._invoke_writer([MagicMock()])
        assert "Article" in article


class TestWriterFellBackInitStamp:
    """writer_fell_back is derived from the provider actually constructed —
    a no-CLAUDE_KEY run resolving to GPT/Gemini must stamp True in the saved
    run_summary, not the pre-chain default False."""

    def _writer_with_model(self, mock_create_writer, model: str):
        from src.article_writer import ArticleWriter

        mock_create_writer.return_value = _mock_llm(model, "# A\n\nB.")
        return ArticleWriter(samples_dir=_samples_dir())

    @patch("src.article_writer.create_writer_llm")
    def test_claude_backed_writer_stamps_false(self, mock_create_writer):
        writer = self._writer_with_model(mock_create_writer, "claude-opus-4-8")
        assert writer.writer_fell_back is False

    @patch("src.article_writer.create_writer_llm")
    def test_openai_backed_writer_stamps_true(self, mock_create_writer):
        writer = self._writer_with_model(mock_create_writer, "gpt-5.4")
        assert writer.writer_fell_back is True
        assert writer.current_model_name == "gpt-5.4"

    @patch("src.article_writer.create_writer_llm")
    def test_gemini_backed_writer_stamps_true(self, mock_create_writer):
        writer = self._writer_with_model(mock_create_writer, "gemini-3.5-flash")
        assert writer.writer_fell_back is True

    @patch("src.article_writer.create_writer_llm")
    def test_creation_log_reports_constructed_instance(
        self, mock_create_writer, monkeypatch
    ):
        """creating_articlewriter_llm logs the real provider/model — never the
        pre-chain 'gemini-fallback' guess."""
        import src.article_writer as aw

        mock_create_writer.return_value = _mock_llm("gpt-5.4", "# A\n\nB.")
        recorder = _RecordingLogger()
        monkeypatch.setattr(aw, "logger", recorder)

        from src.article_writer import ArticleWriter

        ArticleWriter(samples_dir=_samples_dir())

        creations = recorder.named("creating_articlewriter_llm")
        assert len(creations) == 1
        assert creations[0]["provider"] == "openai"
        assert creations[0]["model"] == "gpt-5.4"
        assert "gemini-fallback" not in str(recorder.events)


class TestArticleEditorTracing:
    @patch("src.article_writer.create_writer_llm")
    def test_article_writer_invokes_llm_with_tracing_config(self, mock_create_writer):
        from src.article_writer import ArticleWriter

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        mock_create_writer.return_value = mock_llm

        writer = ArticleWriter(
            samples_dir=Path("writing_samples")
            if Path("writing_samples").exists()
            else None,
            tracing_metadata={"source_trace_id": "trace-123"},
        )

        writer._invoke_with_fallback([MagicMock()])

        mock_llm.invoke.assert_called_once()
        assert mock_llm.invoke.call_args.kwargs["config"] == {
            "metadata": {
                "source_trace_id": "trace-123",
                "workflow": "article_primary",
                "component": "article_writer",
            }
        }

    @patch("src.llms.create_editor_llm")
    @patch("src.editor_tools.get_editor_tools", return_value=[])
    def test_article_editor_passes_extra_callbacks_to_llm(
        self, mock_get_tools, mock_create_llm
    ):
        from src.article_writer import ArticleEditor

        mock_create_llm.return_value = MagicMock()
        tracing_callback = MagicMock()

        ArticleEditor(callbacks=[tracing_callback])

        callbacks = mock_create_llm.call_args.kwargs["callbacks"]
        assert tracing_callback in callbacks

    @patch("src.llms.create_editor_llm")
    @patch("src.editor_tools.get_editor_tools", return_value=[])
    def test_article_editor_final_review_invokes_llm_with_tracing_config(
        self, mock_get_tools, mock_create_llm
    ):
        from src.article_writer import ArticleEditor

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock())
        mock_create_llm.return_value = mock_llm

        editor = ArticleEditor(tracing_metadata={"source_trace_id": "trace-123"})
        editor.review_llm = None
        editor._extract_review_result = MagicMock(return_value={"verdict": "APPROVED"})

        result = asyncio.run(editor._invoke_final_review([MagicMock()]))

        assert result == {"verdict": "APPROVED"}
        mock_llm.ainvoke.assert_called_once()
        assert mock_llm.ainvoke.call_args.kwargs["config"] == {
            "metadata": {
                "source_trace_id": "trace-123",
                "workflow": "editor_final_review",
                "component": "article_editor",
            }
        }
