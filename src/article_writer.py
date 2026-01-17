"""
Article Writer module for generating Medium-style articles from equity analysis reports.

Transforms detailed research reports into engaging, accessible articles
while matching the author's distinctive voice from writing samples.
"""

import json
import os
import random
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config
from src.llms import create_gemini_model, is_gemini_v3_or_greater

# Maximum characters for fact-check context (controls token usage)
MAX_FACT_CHECK_CHARS = 1500

logger = structlog.get_logger(__name__)

# Default fallback prompt config if prompts/writer.json is missing
DEFAULT_PROMPT_CONFIG = {
    "agent_key": "article_writer",
    "agent_name": "Article Writer",
    "version": "1.0",
    "system_message": (
        "You are a senior financial editor who writes engaging, accessible "
        "investment articles. Transform the source report into a Medium-style "
        "article. Match the voice in the writing samples. Use Markdown formatting. "
        "Include images where appropriate using ![desc](url) syntax. "
        "End with a References section."
    ),
    "metadata": {
        "max_sample_chars": 25000,
        "max_chars_per_file": 6000,
        "model_config": {
            "use_quick_model": False,
            "temperature": 0.7,
            "thinking_level": "high",
        },
        "user_template": (
            "WRITING SAMPLES:\n{voice_samples}\n\n"
            "AVAILABLE CHARTS:\n{image_manifest}\n\n"
            "Write an article about {ticker} ({company_name}).\n\n"
            "SOURCE REPORT:\n{report_text}"
        ),
    },
}

# GitHub raw URL base for the repository (configurable via env var)
# Users who want GitHub-hosted image links can set this to their repo
GITHUB_RAW_BASE = os.environ.get(
    "GITHUB_RAW_BASE",
    "https://raw.githubusercontent.com/rgoerwit/ai-investment-agent/main",
)


class ArticleWriter:
    """
    Generates Medium-style articles from equity analysis reports.

    Uses LLM to transform detailed research into engaging articles while:
    - Matching the author's voice from writing samples
    - Embedding charts (local paths by default, or GitHub URLs if configured)
    - Following Medium formatting conventions
    """

    def __init__(
        self,
        prompts_dir: Path | None = None,
        samples_dir: Path | None = None,
        images_dir: Path | None = None,
        use_github_urls: bool = False,
    ):
        """
        Initialize ArticleWriter.

        Args:
            prompts_dir: Directory containing writer.json prompt config
            samples_dir: Directory containing writing samples (*.md, *.txt)
            images_dir: Directory containing generated chart images
            use_github_urls: If True, convert image paths to GitHub raw URLs
                            (requires GITHUB_RAW_BASE env var for custom repos).
                            Default False uses local relative paths.
        """
        self.prompts_dir = prompts_dir or config.prompts_dir
        self.samples_dir = samples_dir or self._find_samples_dir()
        self.images_dir = images_dir or config.images_dir
        self.use_github_urls = use_github_urls

        self.prompt_config = self._load_prompt_config()
        self.llm = self._create_llm()

        logger.info(
            "ArticleWriter initialized",
            prompts_dir=str(self.prompts_dir),
            samples_dir=str(self.samples_dir),
            images_dir=str(self.images_dir),
            use_github_urls=use_github_urls,
        )

    def _find_samples_dir(self) -> Path:
        """Find the writing_samples directory."""
        # Check common locations
        candidates = [
            Path("writing_samples"),
            Path("./writing_samples"),
            Path(__file__).parent.parent / "writing_samples",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        # Return default even if it doesn't exist (will be handled later)
        return Path("writing_samples")

    def _load_prompt_config(self) -> dict:
        """Load prompt configuration from writer.json."""
        prompt_file = self.prompts_dir / "writer.json"

        if not prompt_file.exists():
            logger.warning(
                "writer.json not found, using default prompt config",
                expected_path=str(prompt_file),
            )
            return DEFAULT_PROMPT_CONFIG

        try:
            with open(prompt_file) as f:
                config_data = json.load(f)
            logger.debug(
                "Loaded writer prompt config", version=config_data.get("version")
            )
            return config_data
        except json.JSONDecodeError as e:
            logger.error("Failed to parse writer.json", error=str(e))
            return DEFAULT_PROMPT_CONFIG

    def _create_llm(self):
        """Create the LLM for article generation."""
        # model_config is nested in metadata for AgentPrompt compatibility
        metadata = self.prompt_config.get("metadata", {})
        model_config = metadata.get("model_config", {})

        # Use quick model or deep model based on config
        if model_config.get("use_quick_model", False):
            model_name = config.quick_think_llm
        else:
            model_name = config.deep_think_llm

        temperature = model_config.get("temperature", 0.7)
        thinking_level = model_config.get("thinking_level")

        # Only apply thinking_level for Gemini 3+
        if thinking_level and not is_gemini_v3_or_greater(model_name):
            thinking_level = None

        logger.info(
            "Creating ArticleWriter LLM",
            model=model_name,
            temperature=temperature,
            thinking_level=thinking_level,
        )

        return create_gemini_model(
            model_name=model_name,
            temperature=temperature,
            timeout=config.api_timeout,
            max_retries=config.api_retry_attempts,
            thinking_level=thinking_level,
        )

    def _load_voice_samples(self, max_chars: int | None = None) -> str:
        """
        Load writing samples to establish author voice.

        Args:
            max_chars: Maximum total characters to include (default from config)

        Returns:
            Concatenated samples as string, or empty string if none found
        """
        metadata = self.prompt_config.get("metadata", {})
        if max_chars is None:
            max_chars = metadata.get("max_sample_chars", 25000)
        max_per_file = metadata.get("max_chars_per_file", 6000)

        if not self.samples_dir.exists():
            logger.warning(
                "Writing samples directory not found", path=str(self.samples_dir)
            )
            return ""

        samples = []
        total_chars = 0

        # Load .txt and .md files, randomized for variety across runs
        sample_files = list(self.samples_dir.glob("*.txt")) + list(
            self.samples_dir.glob("*.md")
        )
        random.shuffle(sample_files)

        if not sample_files:
            logger.warning("No writing samples found", path=str(self.samples_dir))
            return ""

        for sample_file in sample_files:
            try:
                content = sample_file.read_text(encoding="utf-8")

                # Cap each file to max_per_file chars
                if len(content) > max_per_file:
                    content = content[:max_per_file] + "\n[...truncated]"

                samples.append(f"--- Sample: {sample_file.name} ---\n{content}")
                total_chars += len(content)

                # Check limit AFTER adding - ensures last file is included
                if total_chars >= max_chars:
                    break

            except Exception as e:
                logger.warning(
                    "Failed to read sample", file=str(sample_file), error=str(e)
                )

        logger.info(
            "Loaded writing samples",
            count=len(samples),
            total_chars=total_chars,
            files=[f.name for f in sample_files[: len(samples)]],
        )

        return "\n\n".join(samples)

    def _format_image_manifest(self, ticker: str, trade_date: str) -> str:
        """
        Create image manifest with URLs for available charts.

        Args:
            ticker: Stock ticker (e.g., "0005.HK")
            trade_date: Analysis date (e.g., "2026-01-01")

        Returns:
            Formatted manifest with image descriptions and URLs
        """
        if not self.images_dir.exists():
            logger.warning("Images directory not found", path=str(self.images_dir))
            return "No charts available."

        # Normalize ticker for filename matching (dots become underscores or dashes)
        safe_ticker_underscore = ticker.replace(".", "_").replace("/", "_")
        safe_ticker_dash = ticker.replace(".", "-").replace("/", "-")

        # Find matching images - try multiple naming conventions
        patterns = [
            # Standard pattern: TICKER_DATE_charttype
            f"{safe_ticker_underscore}_{trade_date}_football_field.*",
            f"{safe_ticker_underscore}_{trade_date}_radar.*",
            f"{safe_ticker_underscore}_*_football_field.*",  # Fallback without date
            f"{safe_ticker_underscore}_*_radar.*",
            # Output-based pattern: *TICKER*_charttype (from --output flag)
            f"*{safe_ticker_underscore}*_football_field.*",
            f"*{safe_ticker_underscore}*_radar.*",
            f"*{safe_ticker_dash}*_football_field.*",
            f"*{safe_ticker_dash}*_radar.*",
        ]

        found_images = []
        for pattern in patterns:
            matches = list(self.images_dir.glob(pattern))
            for match in matches:
                if match not in [img for img, _, _ in found_images]:
                    # Determine chart type
                    if "football_field" in match.name:
                        chart_type = "Football Field Valuation Chart"
                        description = "Shows 52-week range, analyst targets, and current price positioning"
                    elif "radar" in match.name:
                        chart_type = "Thesis Alignment Radar Chart"
                        description = (
                            "6-axis view: Health, Growth, Valuation, Regulatory, "
                            "Undiscovered, Jurisdiction scores"
                        )
                    else:
                        chart_type = "Chart"
                        description = "Analysis visualization"

                    found_images.append((match, chart_type, description))

        if not found_images:
            logger.info("No matching charts found", ticker=ticker, date=trade_date)
            return "No charts available for this ticker."

        # Format manifest
        manifest_lines = []
        for img_path, chart_type, description in found_images:
            if self.use_github_urls:
                # Convert to GitHub raw URL
                # Path relative to repo root
                try:
                    if img_path.is_absolute():
                        rel_path = img_path.relative_to(Path.cwd())
                    else:
                        rel_path = img_path
                    url = f"{GITHUB_RAW_BASE}/{rel_path}"
                except ValueError:
                    # Path is not under cwd (e.g., temp directory in tests)
                    # Fall back to using the images_dir relative path
                    rel_path = f"images/{img_path.name}"
                    url = f"{GITHUB_RAW_BASE}/{rel_path}"
            else:
                # Use local path
                url = str(img_path)

            manifest_lines.append(f"- **{chart_type}**: {description}")
            manifest_lines.append(f"  URL: {url}")
            manifest_lines.append("")

        logger.info("Formatted image manifest", chart_count=len(found_images))
        return "\n".join(manifest_lines)

    def _fetch_fact_check_context(
        self, ticker: str, company_name: str, max_chars: int = MAX_FACT_CHECK_CHARS
    ) -> str:
        """
        Fetch recent market context for fact-checking via Tavily search.

        This provides the LLM with fresh external data to cross-reference
        against the source report, catching potential outdated info.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            max_chars: Maximum characters to return (controls token usage)

        Returns:
            Recent market context as string, or empty string if unavailable
        """
        api_key = config.get_tavily_api_key()
        if not api_key:
            logger.debug("Tavily API key not available, skipping fact-check context")
            return ""

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)

            # Single focused search query
            query = f"{company_name} {ticker} stock latest news financials 2025 2026"

            logger.info("Fetching fact-check context", ticker=ticker, query=query[:50])

            # Use Tavily's search with limited results
            response = client.search(
                query=query,
                max_results=3,  # Just 3 results to keep it minimal
                search_depth="basic",  # Faster than "advanced"
                include_answer=True,  # Get a summary answer
            )

            # Build context from results
            context_parts = []

            # Include the AI-generated answer if available (most concise)
            if response.get("answer"):
                context_parts.append(f"Summary: {response['answer']}")

            # Add snippets from top results
            for result in response.get("results", [])[:3]:
                title = result.get("title", "")
                content = result.get("content", "")
                if title and content:
                    # Truncate individual snippets
                    snippet = f"- {title}: {content[:300]}"
                    context_parts.append(snippet)

            context = "\n".join(context_parts)

            # Enforce max chars
            if len(context) > max_chars:
                context = context[:max_chars] + "\n[...truncated]"

            logger.info(
                "Fetched fact-check context",
                ticker=ticker,
                chars=len(context),
                results=len(response.get("results", [])),
            )

            return context

        except ImportError:
            logger.debug("Tavily package not installed, skipping fact-check context")
            return ""
        except Exception as e:
            logger.warning(
                "Failed to fetch fact-check context",
                ticker=ticker,
                error=str(e),
            )
            return ""

    def write(
        self,
        ticker: str,
        company_name: str,
        report_text: str,
        trade_date: str,
        output_path: Path | None = None,
        valuation_context: str | None = None,
    ) -> str:
        """
        Generate an article from the analysis report.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            report_text: The full analysis report to transform
            trade_date: Date of the analysis
            output_path: Optional path to save the article
            valuation_context: Optional context about chart valuation vs decision

        Returns:
            Generated article as Markdown string
        """
        logger.info("Generating article", ticker=ticker, company=company_name)

        # Load voice samples
        voice_samples = self._load_voice_samples()
        if not voice_samples:
            voice_samples = (
                "(No writing samples available - use default professional tone)"
            )

        # Format image manifest
        image_manifest = self._format_image_manifest(ticker, trade_date)

        # Fetch fact-check context (single Tavily search, controlled size)
        fact_check_context = self._fetch_fact_check_context(ticker, company_name)

        # Get prompt templates
        system_message = self.prompt_config.get(
            "system_message", DEFAULT_PROMPT_CONFIG["system_message"]
        )
        # user_template is nested in metadata for AgentPrompt compatibility
        metadata = self.prompt_config.get("metadata", {})
        user_template = metadata.get(
            "user_template", DEFAULT_PROMPT_CONFIG["metadata"]["user_template"]
        )

        # Format user message
        # Use provided valuation_context or default message
        val_context = valuation_context or "VALUATION DATA: Not available."
        user_message = user_template.format(
            voice_samples=voice_samples,
            image_manifest=image_manifest,
            ticker=ticker,
            company_name=company_name,
            valuation_context=val_context,
            report_text=report_text,
        )

        # Append fact-check context if available
        if fact_check_context:
            user_message += (
                "\n\n---\n\n"
                "RECENT MARKET CONTEXT (for fact-checking - flag any discrepancies):\n\n"
                f"{fact_check_context}"
            )

        # Generate article
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        try:
            response = self.llm.invoke(messages)

            # Handle response.content being a list (Gemini with thinking_level)
            # or a string (standard response)
            if isinstance(response.content, list):
                # Extract text from content blocks
                text_parts = []
                for block in response.content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                article = "\n".join(text_parts)
            else:
                article = response.content

            logger.info(
                "Article generated",
                ticker=ticker,
                length=len(article),
                word_count=len(article.split()),
            )

            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(article, encoding="utf-8")
                logger.info("Article saved", path=str(output_path))

            return article

        except Exception as e:
            logger.error("Failed to generate article", ticker=ticker, error=str(e))
            raise


def generate_article(
    ticker: str,
    company_name: str,
    report_text: str,
    trade_date: str,
    output_path: Path | str | None = None,
    use_github_urls: bool = True,
    valuation_context: str | None = None,
) -> str:
    """
    Convenience function to generate an article.

    Args:
        ticker: Stock ticker symbol
        company_name: Full company name
        report_text: The full analysis report
        trade_date: Date of the analysis
        output_path: Optional path to save the article
        use_github_urls: Convert image paths to GitHub URLs
        valuation_context: Optional context about chart valuation vs decision

    Returns:
        Generated article as Markdown string
    """
    writer = ArticleWriter(use_github_urls=use_github_urls)
    return writer.write(
        ticker=ticker,
        company_name=company_name,
        report_text=report_text,
        trade_date=trade_date,
        output_path=Path(output_path) if output_path else None,
        valuation_context=valuation_context,
    )
