"""
Article Writer module for generating Medium-style articles from equity analysis reports.

Transforms detailed research reports into engaging, accessible articles
while matching the author's distinctive voice from writing samples.
"""

import json
import random
import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.article_audit import (
    audit_article_citations,
    extract_source_confidence_context,
    prepend_verification_caveats,
)
from src.config import config, get_env_value
from src.error_safety import redact_sensitive_text, summarize_exception
from src.llms import create_writer_fallback_llm, create_writer_llm
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import classify_failure, get_model_name, infer_provider
from src.runtime_services import get_current_tool_service
from src.tavily_utils import search_tavily_sync_inspected
from src.token_tracker import TokenTrackingCallback, get_tracker
from src.tooling.runtime import ToolInvocation

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
        },
        "user_template": (
            "WRITING SAMPLES:\n{voice_samples}\n\n"
            "AVAILABLE CHARTS:\n{image_manifest}\n\n"
            "Write an article about {ticker} ({company_name}).\n\n"
            "SOURCE REPORT:\n{report_text}"
        ),
    },
}

# Phrases that indicate a model refusal on financial content
_REFUSAL_INDICATORS = (
    "I cannot provide financial advice",
    "I'm not able to provide investment advice",
    "I can't offer specific investment recommendations",
)

_EDITOR_STRUCTURED_OUTPUT_WARNING_PATTERN = (
    r"(?s)^Pydantic serializer warnings:.*PydanticSerializationUnexpectedValue.*"
)


class EditorFactualError(BaseModel):
    location: str = ""
    claim: str = ""
    ground_truth: str = ""
    action: str = ""


class EditorReferenceCheck(BaseModel):
    url: str = ""
    status: str = ""
    note: str = ""


class EditorReviewResult(BaseModel):
    verdict: Literal["REVISE", "APPROVED"] = "APPROVED"
    factual_errors: list[EditorFactualError] = Field(default_factory=list)
    reference_checks: list[EditorReferenceCheck] = Field(default_factory=list)
    cuts: list[str] = Field(default_factory=list)
    style_issues: list[str] = Field(default_factory=list)
    confidence: float = 0.5


# Fenced meta block the writer appends during revisions (CORRECTED/CONTESTED
# lines). The editor reads it for contest adjudication; it must never survive
# into the published article.
_EDITOR_NOTES_PATTERN = re.compile(r"\n*```EDITOR_NOTES\b.*?(?:```|\Z)\s*", re.DOTALL)


def _strip_editor_notes(article: str) -> str:
    """Remove the fenced EDITOR_NOTES block from a (revised) article."""
    if not article or "```EDITOR_NOTES" not in article:
        return article
    stripped = _EDITOR_NOTES_PATTERN.sub("", article)
    logger.info(
        "stripped_editor_notes_block",
        removed_chars=len(article) - len(stripped),
    )
    return stripped.rstrip() + "\n"


def _enforce_verdict_consistency(feedback: dict) -> dict:
    """Coerce APPROVED to REVISE when the editor's own findings contradict it.

    The LLM applies the decision rules in the prompt, but nothing stops it from
    listing factual errors (or broken references) and still self-approving.
    The verdict must follow the findings, so enforce that in code.
    """
    if not isinstance(feedback, dict) or feedback.get("verdict") != "APPROVED":
        return feedback

    reference_checks = feedback.get("reference_checks") or []
    broken_reference = any(
        isinstance(check, dict)
        and str(check.get("status", "")).lower() in ("broken", "unsupported")
        for check in reference_checks
    )
    if feedback.get("factual_errors") or broken_reference:
        logger.warning(
            "editor_verdict_overridden",
            reason="findings_present",
            num_factual_errors=len(feedback.get("factual_errors") or []),
            broken_reference=broken_reference,
        )
        feedback["verdict"] = "REVISE"
    return feedback


def _review_response_for_citation_errors(errors: list[dict[str, str]]) -> dict:
    feedback = EditorReviewResult(
        verdict="REVISE",
        factual_errors=[EditorFactualError(**error) for error in errors],
        confidence=1.0,
    ).model_dump()
    # Internal loop metadata; intentionally not part of the editor response schema.
    feedback["deterministic_citation_audit"] = True
    return feedback


def _should_emit_editor_structured_output_warnings() -> bool:
    """Always suppress: these are known-benign Pydantic serializer warnings."""
    return False


@contextmanager
def _structured_review_warning_policy():
    """Suppress known benign Pydantic serializer warnings outside debug runs."""
    if _should_emit_editor_structured_output_warnings():
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_EDITOR_STRUCTURED_OUTPUT_WARNING_PATTERN,
            category=UserWarning,
            module=r"pydantic(\..*)?",
        )
        yield


def _extract_text_from_response(response) -> str:
    """
    Extract text content from an LLM response, handling both
    Claude (Anthropic) and Gemini (Google) response formats.

    Claude with thinking:
        [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
    Claude without thinking:
        "plain string"
    Gemini with thinking:
        [{"text": "..."}, ...]  (no type field)

    Returns:
        Concatenated text content, excluding thinking/redacted blocks.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    text_parts = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type in ("thinking", "redacted_thinking"):
                continue
            if block_type == "text":
                text_parts.append(block["text"])
            elif "text" in block:
                # Gemini format: no "type" field, just "text"
                text_parts.append(block["text"])
        elif hasattr(block, "text"):
            # LangChain internal content block objects
            text_parts.append(block.text)

    return "\n".join(text_parts)


# GitHub raw URL base for the repository (configurable via env var or .env)
# Users who want GitHub-hosted image links can set this to their repo
GITHUB_RAW_BASE = get_env_value("GITHUB_RAW_BASE") or (
    "https://raw.githubusercontent.com/rgoerwit/ai-investment-agent/main"
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
        callbacks: list[BaseCallbackHandler] | None = None,
        tracing_metadata: dict[str, object] | None = None,
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
        self._callbacks = callbacks or []
        self._tracing_metadata = dict(tracing_metadata or {})

        self.prompt_config = self._load_prompt_config()
        self.llm = self._create_llm()

        logger.info(
            "articlewriter_initialized",
            prompts_dir=str(self.prompts_dir),
            samples_dir=str(self.samples_dir),
            images_dir=str(self.images_dir),
            use_github_urls=use_github_urls,
            tracing_callbacks=len(self._callbacks),
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
                "writer_json_not_found_using_default_prompt_config",
                expected_path=str(prompt_file),
            )
            return DEFAULT_PROMPT_CONFIG

        try:
            with open(prompt_file) as f:
                config_data = json.load(f)
            logger.debug(
                "loaded_writer_prompt_config", version=config_data.get("version")
            )
            return cast(dict, config_data)
        except json.JSONDecodeError as e:
            logger.error(
                "writer_config_parse_failed",
                **summarize_exception(e, operation="article_writer:parse_config"),
            )
            return DEFAULT_PROMPT_CONFIG

    def _create_llm(self):
        """Create the LLM for article generation.

        Uses Claude (Anthropic) when CLAUDE_KEY is configured,
        falls back to Gemini otherwise. Model selection and
        thinking configuration are handled by create_writer_llm().

        Note: use_quick_model in model_config is ignored — the writer
        always uses WRITER_MODEL (Claude) or DEEP_MODEL (Gemini fallback).
        To use a cheaper Claude model, set WRITER_MODEL=claude-haiku-4-5.
        """
        metadata = self.prompt_config.get("metadata", {})
        model_config = metadata.get("model_config", {})
        temperature = model_config.get("temperature", 0.7)

        logger.info(
            "creating_articlewriter_llm",
            provider="claude" if config.get_claude_api_key() else "gemini-fallback",
            model=config.writer_model,
        )

        return create_writer_llm(
            temperature=temperature,
            timeout=config.api_timeout,
            max_retries=get_runtime_config(config).api_retry_attempts,
            callbacks=[
                TokenTrackingCallback("Article Writer", get_tracker()),
                *self._callbacks,
            ],
        )

    def _invoke_config(self, *, workflow: str) -> dict[str, object]:
        return {
            "metadata": {
                **self._tracing_metadata,
                "workflow": workflow,
                "component": "article_writer",
            }
        }

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
                "writing_samples_directory_not_found", path=str(self.samples_dir)
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
            logger.warning("no_writing_samples_found", path=str(self.samples_dir))
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
                    "failed_to_read_sample",
                    file=str(sample_file),
                    **summarize_exception(e, operation="failed_to_read_sample"),
                )

        logger.info(
            "loaded_writing_samples",
            count=len(samples),
            total_chars=total_chars,
            files=[f.name for f in sample_files[: len(samples)]],
        )

        return "\n\n".join(samples)

    def load_voice_samples(self, max_chars: int | None = None) -> str:
        """Public accessor so the editor can receive the same writing samples."""
        return self._load_voice_samples(max_chars=max_chars)

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
            logger.warning("images_directory_not_found", path=str(self.images_dir))
            return "No charts available."

        # Normalize ticker for filename matching (dots become underscores or dashes)
        safe_ticker_underscore = ticker.replace(".", "_").replace("/", "_")
        safe_ticker_dash = ticker.replace(".", "-").replace("/", "-")

        # Find matching images - try multiple naming conventions
        patterns = [
            # Raw ticker pattern: chart_node uses ticker directly as filename_stem
            f"{ticker}_football_field.*",
            f"{ticker}_radar.*",
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

        found_images: list[tuple[Path, str, str]] = []
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
            logger.info("no_matching_charts_found", ticker=ticker, date=trade_date)
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
                    rel_path = Path("images") / img_path.name
                    url = f"{GITHUB_RAW_BASE}/{rel_path}"
            else:
                # Use local path
                url = str(img_path)

            manifest_lines.append(f"- **{chart_type}**: {description}")
            manifest_lines.append(f"  URL: {url}")
            manifest_lines.append("")

        logger.info("formatted_image_manifest", chart_count=len(found_images))
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
        try:
            # Single focused search query
            query = f"{company_name} {ticker} stock latest news financials 2025 2026"

            logger.info("fetching_fact_check_context", ticker=ticker, query=query[:50])

            response = search_tavily_sync_inspected(
                query,
                profile="news_basic",
                include_answer=True,
            )
            if not isinstance(response, dict):
                return ""

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
                "fetched_fact_check_context",
                ticker=ticker,
                chars=len(context),
                results=len(response.get("results", [])),
            )

            return context

        except Exception as e:
            logger.warning(
                "failed_to_fetch_fact_check_context",
                ticker=ticker,
                **summarize_exception(
                    e, operation="failed_to_fetch_fact_check_context"
                ),
            )
            return ""

    def _invoke_with_fallback(self, messages: list):
        """Invoke LLM with automatic Claude → Gemini fallback on API errors."""
        primary_model = get_model_name(self.llm)
        primary_provider = infer_provider(
            model_name=primary_model, class_name=type(self.llm).__name__
        )
        try:
            logger.info(
                "llm_call_start",
                context="article_writer_primary",
                provider=primary_provider,
                model=primary_model,
                runnable_class=type(self.llm).__name__,
                max_attempts=1,
            )
            response = self.llm.invoke(
                messages,
                config=self._invoke_config(workflow="article_primary"),
            )
            logger.info(
                "llm_call_success",
                context="article_writer_primary",
                provider=primary_provider,
                model=primary_model,
                runnable_class=type(self.llm).__name__,
                attempt=1,
            )
            return response
        except Exception as e:
            error_str = str(e)
            primary_failure = classify_failure(
                e,
                provider=primary_provider,
                model_name=primary_model,
                class_name=type(self.llm).__name__,
            )
            get_tracker().record_failure(
                agent_name="Article Writer",
                provider=primary_failure.provider,
                failure_kind=primary_failure.kind,
                model_name=primary_model or "",
            )
            # Only fall back for Claude-specific API errors, not all exceptions
            is_anthropic_error = any(
                marker in error_str
                for marker in ("anthropic", "credit balance", "Anthropic API")
            )
            if not is_anthropic_error:
                # Check exception module for anthropic SDK errors
                mod = type(e).__module__ or ""
                is_anthropic_error = "anthropic" in mod

            if not is_anthropic_error:
                logger.error(
                    "llm_call_failed",
                    context="article_writer_primary",
                    provider=primary_failure.provider,
                    model=primary_model,
                    runnable_class=type(self.llm).__name__,
                    attempt=1,
                    max_attempts=1,
                    failure_kind=primary_failure.kind,
                    host=primary_failure.host,
                    retryable=primary_failure.retryable,
                    error_type=primary_failure.error_type,
                    root_cause_type=primary_failure.root_cause_type,
                    error_message=primary_failure.message,
                )
                raise  # Not a Claude error — propagate

            logger.warning(
                "claude_writer_failed_falling_back_to_gemini",
                failure_kind=primary_failure.kind,
                host=primary_failure.host,
                error_type=primary_failure.error_type,
                root_cause_type=primary_failure.root_cause_type,
                reason=primary_failure.message,
            )
            fallback_llm = create_writer_fallback_llm()
            self.llm = fallback_llm  # Cache for subsequent calls (e.g., retry)
            fallback_model = get_model_name(fallback_llm)
            fallback_provider = infer_provider(
                model_name=fallback_model, class_name=type(fallback_llm).__name__
            )
            try:
                logger.info(
                    "llm_call_start",
                    context="article_writer_fallback",
                    provider=fallback_provider,
                    model=fallback_model,
                    runnable_class=type(fallback_llm).__name__,
                    max_attempts=1,
                )
                response = fallback_llm.invoke(
                    messages,
                    config=cast(Any, self._invoke_config(workflow="article_fallback")),
                )
                logger.info(
                    "llm_call_success",
                    context="article_writer_fallback",
                    provider=fallback_provider,
                    model=fallback_model,
                    runnable_class=type(fallback_llm).__name__,
                    attempt=1,
                )
                return response
            except Exception as fallback_exc:
                fallback_failure = classify_failure(
                    fallback_exc,
                    provider=fallback_provider,
                    model_name=fallback_model,
                    class_name=type(fallback_llm).__name__,
                )
                get_tracker().record_failure(
                    agent_name="Article Writer Fallback",
                    provider=fallback_failure.provider,
                    failure_kind=fallback_failure.kind,
                    model_name=fallback_model or "",
                )
                logger.error(
                    "llm_call_failed",
                    context="article_writer_fallback",
                    provider=fallback_failure.provider,
                    model=fallback_model,
                    runnable_class=type(fallback_llm).__name__,
                    attempt=1,
                    max_attempts=1,
                    failure_kind=fallback_failure.kind,
                    host=fallback_failure.host,
                    retryable=fallback_failure.retryable,
                    error_type=fallback_failure.error_type,
                    root_cause_type=fallback_failure.root_cause_type,
                    error_message=fallback_failure.message,
                    same_failure_kind=fallback_failure.kind == primary_failure.kind,
                )
                raise

    def _invoke_writer(self, messages: list) -> str:
        """
        Invoke the writer LLM with provider-aware post-processing.

        Handles:
        1. Claude → Gemini fallback on API errors (billing, auth, rate limits)
        2. Thinking block extraction (Claude adaptive thinking / Gemini thinking)
        3. Preamble stripping (Claude's politeness tendency)
        4. Refusal detection (financial advice guardrails)

        Args:
            messages: List of SystemMessage/HumanMessage to send

        Returns:
            Clean article text (Markdown, starting with # Title)

        Raises:
            RuntimeError: If the model refuses to generate the article
        """
        response = self._invoke_with_fallback(messages)

        # 0. Truncation guard: a MAX_TOKENS/LENGTH finish means the model ran out
        # of output budget mid-generation (e.g. Gemini hidden reasoning eating the
        # shared completion pool). Surface it loudly instead of silently saving a
        # mid-sentence article. (June 2026 1928.T fallback failure.)
        finish_reason = str(
            (getattr(response, "response_metadata", {}) or {}).get("finish_reason", "")
        ).upper()
        if finish_reason in ("MAX_TOKENS", "LENGTH"):
            logger.warning(
                "writer_output_truncated",
                finish_reason=finish_reason,
                model=get_model_name(self.llm),
            )

        # 1. Extract text, filtering out thinking blocks
        article = _extract_text_from_response(response)

        # 2. Refusal detection (before stripping, so we catch it in raw output)
        for indicator in _REFUSAL_INDICATORS:
            if indicator.lower() in article.lower():
                logger.error(
                    "writer_llm_refused_to_generate_article",
                    indicator=indicator,
                    response_preview=redact_sensitive_text(article, max_chars=200),
                )
                raise RuntimeError(
                    f"Writer LLM refused to generate financial content. "
                    f"Detected: '{indicator}'. Consider adjusting the system prompt "
                    f"to frame the task as journalism/analysis rather than advice."
                )

        # 3. Preamble stripping — phrase-based first
        article = _strip_llm_preamble(article)

        # Fallback: if still doesn't start with #, find the first header
        stripped = article.strip()
        if stripped and not stripped.startswith("#"):
            header_idx = stripped.find("\n# ")
            if header_idx != -1:
                logger.info("stripped_non_header_preamble_from_writer_response")
                article = stripped[header_idx + 1 :]  # +1 to skip the \n
            elif stripped.find("# ") != -1:
                article = stripped[stripped.find("# ") :]

        return article

    def write(
        self,
        ticker: str,
        company_name: str,
        report_text: str,
        trade_date: str,
        output_path: Path | None = None,
        valuation_context: str | None = None,
        governance_card: dict[str, Any] | None = None,
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
        logger.info("generating_article", ticker=ticker, company=company_name)

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
        default_metadata = cast(dict[str, Any], DEFAULT_PROMPT_CONFIG["metadata"])
        metadata = self.prompt_config.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        user_template = metadata.get("user_template", default_metadata["user_template"])
        user_template = str(user_template)

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

        # Prepend entity governance card so the writer never collapses identity
        # back to a short-form name and discloses non-standard structures.
        if isinstance(governance_card, dict) and governance_card.get("ticker"):
            from src.validators.entity_governance_card import (
                card_from_dict,
                card_to_prompt_block,
                requires_structure_disclosure,
            )

            card_obj = card_from_dict(governance_card)
            if card_obj:
                card_block = card_to_prompt_block(card_obj)
                disclosure_directive = ""
                if requires_structure_disclosure(card_obj):
                    related = (
                        "; ".join(
                            f"{e.get('ticker','?')} ({e.get('relationship','?')})"
                            for e in card_obj.related_listed
                        )
                        or "operating subsidiaries"
                    )
                    disclosure_directive = (
                        "\n\nMANDATORY OPENING DISCLOSURE: The first body paragraph "
                        f"after the headline must disclose that {ticker} ({card_obj.canonical_name}) "
                        f"is a {card_obj.entity_role.replace('_', ' ').lower()}, and identify the "
                        f"related entity(s): {related}. Do not refer to this ticker by an "
                        "operating-subsidiary name. Do not skip this disclosure."
                    )
                user_message = (
                    f"{card_block}{disclosure_directive}\n\n---\n\n{user_message}"
                )

        # Generate article
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        try:
            article = self._invoke_writer(messages)

            logger.info(
                "article_generated",
                ticker=ticker,
                length=len(article),
                word_count=len(article.split()),
            )

            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(article, encoding="utf-8")
                logger.info("article_saved", path=str(output_path))

            return article

        except Exception as e:
            logger.error(
                "article_generation_failed",
                ticker=ticker,
                **summarize_exception(e, operation="article_writer:generate"),
            )
            raise

    def revise(
        self,
        original_draft: str,
        editor_feedback: dict,
        ticker: str,
        company_name: str,
        governance_context: str | None = None,
    ) -> str:
        """
        Revise an article based on Editor-in-Chief feedback.

        Args:
            original_draft: The article draft to revise
            editor_feedback: Parsed JSON feedback from the editor
            ticker: Stock ticker symbol
            company_name: Full company name

        Returns:
            Revised article as Markdown string
        """
        # Extract chart references before revision (for preservation)
        original_charts = _extract_chart_references(original_draft)

        logger.info(
            "revising_article_based_on_editor_feedback",
            ticker=ticker,
            errors=len(editor_feedback.get("factual_errors", [])),
            cuts=len(editor_feedback.get("cuts", [])),
            charts_to_preserve=len(original_charts),
        )

        # Get revision template from prompt config
        metadata = self.prompt_config.get("metadata", {})
        revision_template = metadata.get("revision_template")

        if not revision_template:
            # Fallback template if not in prompt config
            revision_template = (
                "CRITICAL: Your draft was rejected by the Editor-in-Chief.\n\n"
                "ORIGINAL DRAFT:\n{original_draft}\n\n"
                "EDITOR FEEDBACK:\n"
                "- Factual errors to fix: {factual_errors}\n"
                "- Passages to cut: {cuts}\n"
                "- Style issues: {style_issues}\n\n"
                "TASK: Rewrite the article addressing ALL feedback above.\n"
                "- Fix every factual error using the ground_truth values provided\n"
                "- Remove or tighten the passages marked for cutting\n"
                "- Eliminate hedge words and passive voice\n"
                "- MAINTAIN the INTJ analytical voice throughout\n"
                "- PRESERVE all image embeds (![...](...)): {chart_list}\n"
                "- Do not apologize or explain changes. Just produce the corrected article.\n"
                "- Start directly with '# Title'. No preamble."
            )

        # Format the revision prompt
        import json as json_module

        factual_errors = editor_feedback.get("factual_errors", [])
        cuts = editor_feedback.get("cuts", [])
        style_issues = editor_feedback.get("style_issues", [])

        # Build chart preservation list
        chart_list = ", ".join(c["full_match"] for c in original_charts) or "None"

        revision_prompt = revision_template.format(
            original_draft=original_draft,
            factual_errors=json_module.dumps(factual_errors, indent=2),
            cuts=json_module.dumps(cuts),
            style_issues=json_module.dumps(style_issues),
            chart_list=chart_list,
        )
        if governance_context:
            revision_prompt = (
                f"{governance_context}\n\n---\n\n"
                "PRESERVE AND APPLY THE ENTITY GOVERNANCE CARD ABOVE WHEN REVISING.\n\n"
                f"{revision_prompt}"
            )

        # Get system message
        system_message = self.prompt_config.get(
            "system_message", DEFAULT_PROMPT_CONFIG["system_message"]
        )

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=revision_prompt),
        ]

        try:
            article = self._invoke_writer(messages)

            # Re-inject any charts that were lost during revision
            if original_charts:
                article = _reinject_missing_charts(article, original_charts, logger)

            logger.info(
                "article_revised",
                ticker=ticker,
                length=len(article),
                word_count=len(article.split()),
            )

            return article

        except Exception as e:
            logger.error(
                "article_revision_failed",
                ticker=ticker,
                **summarize_exception(e, operation="article_writer:revise"),
            )
            raise


def _extract_chart_references(text: str) -> list[dict]:
    """
    Extract markdown image references from article text.

    Args:
        text: Article markdown content

    Returns:
        List of dicts with 'full_match', 'alt_text', 'path' for each image
    """
    import re

    if not text:
        return []

    # Match markdown image syntax: ![alt text](path)
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    matches = re.findall(pattern, text)

    return [
        {"alt_text": alt, "path": path, "full_match": f"![{alt}]({path})"}
        for alt, path in matches
    ]


def _reinject_missing_charts(
    revised_article: str,
    original_charts: list[dict],
    logger,
) -> str:
    """
    Re-inject chart references that were lost during revision.

    Attempts to place charts in logical locations based on their type.

    Args:
        revised_article: The revised article text
        original_charts: Chart references from the original draft
        logger: Logger instance

    Returns:
        Article with missing charts re-injected
    """
    if not original_charts:
        return revised_article

    # Check which charts are missing
    missing_charts = []
    for chart in original_charts:
        if chart["full_match"] not in revised_article:
            missing_charts.append(chart)

    if not missing_charts:
        return revised_article

    logger.warning(
        "re_injecting_missing_charts",
        count=len(missing_charts),
        charts=[c["alt_text"] for c in missing_charts],
    )

    # Re-inject based on chart type and logical placement
    for chart in missing_charts:
        alt_lower = chart["alt_text"].lower()
        insertion_point = None

        # Football field chart goes near Valuation section
        if "football" in alt_lower or "valuation" in alt_lower:
            for marker in ["## Valuation", "### Valuation", "# Valuation"]:
                if marker in revised_article:
                    # Insert after the header and first paragraph
                    idx = revised_article.find(marker)
                    # Find end of next paragraph (double newline)
                    para_end = revised_article.find("\n\n", idx + len(marker))
                    if para_end != -1:
                        insertion_point = para_end + 2
                    break

        # Radar chart goes near Thesis or Assessment
        elif "radar" in alt_lower or "thesis" in alt_lower or "alignment" in alt_lower:
            for marker in ["## Thesis", "### Thesis", "## Verdict", "### Verdict"]:
                if marker in revised_article:
                    idx = revised_article.find(marker)
                    para_end = revised_article.find("\n\n", idx)
                    if para_end != -1:
                        insertion_point = para_end + 2
                    break

        # Default: insert before References section
        if insertion_point is None:
            for marker in ["## References", "### References", "# References"]:
                if marker in revised_article:
                    insertion_point = revised_article.find(marker)
                    break

        # Last resort: append before end
        if insertion_point is None:
            insertion_point = len(revised_article)

        # Insert the chart
        chart_block = f"\n{chart['full_match']}\n\n"
        revised_article = (
            revised_article[:insertion_point]
            + chart_block
            + revised_article[insertion_point:]
        )

    return revised_article


def _strip_llm_preamble(text: str) -> str:
    """
    Strip common LLM preamble phrases that precede the actual article.

    Models sometimes add "Here is the revised article:" or similar text
    even when instructed not to. This ensures clean output.
    """
    if not text:
        return text

    # Common preamble patterns to strip
    preambles = [
        "Here is the revised article:",
        "Here is the revised version:",
        "Here is the corrected article:",
        "Here's the revised article:",
        "Here's the corrected article:",
        "Below is the revised article:",
        "I've revised the article:",
        "I have revised the article:",
        # Claude-typical patterns
        "Certainly! Here is the revised article:",
        "Certainly, here is the corrected article:",
        "Certainly! Here is the article:",
        "Here is the article about",
    ]

    stripped = text.strip()
    for preamble in preambles:
        if stripped.lower().startswith(preamble.lower()):
            # Remove preamble and any following whitespace
            stripped = stripped[len(preamble) :].lstrip()
            break

    # Also handle case where preamble is followed by blank line then content
    # Split on double newline and check if first part looks like preamble
    if "\n\n" in stripped:
        first_part, rest = stripped.split("\n\n", 1)
        if len(first_part) < 100 and not first_part.startswith("#"):
            # First part is short and not a header - likely preamble
            for preamble in preambles:
                if preamble.lower().rstrip(":") in first_part.lower():
                    stripped = rest
                    break

    return stripped


class ArticleEditor:
    """
    Editor-in-Chief that reviews and improves articles generated by the Writer.

    Uses GPT to fact-check against ground truth data and tighten prose
    while preserving the author's distinctive voice. The cross-model design
    (Claude writes, GPT edits) is intentional — it reduces single-model bias.
    """

    # Maximum revision iterations to prevent infinite loops
    MAX_REVISIONS = 2

    # Maximum tool-calling iterations per review to prevent runaway loops
    MAX_TOOL_ITERATIONS = 5

    # Maximum tool calls to execute per single LLM turn (prevents parallel call floods)
    MAX_TOOL_CALLS_PER_TURN = 3

    def __init__(
        self,
        *,
        callbacks: list[BaseCallbackHandler] | None = None,
        tracing_metadata: dict[str, object] | None = None,
    ):
        """Initialize the ArticleEditor."""
        from src.editor_tools import get_editor_tools
        from src.llms import create_editor_llm

        self._callbacks = callbacks or []
        self._tracing_metadata = dict(tracing_metadata or {})
        self.llm = create_editor_llm(
            callbacks=[
                TokenTrackingCallback("Article Editor", get_tracker()),
                *self._callbacks,
            ],
        )
        self.tools = get_editor_tools()
        self.prompt_config = self._load_prompt_config()
        self._url_cache: dict[str, str] = {}
        self.review_llm = self._build_structured_review_llm()

        # Build tool lookup and bind tools to LLM for agentic reference checking
        self._tools_by_name = {t.name: t for t in self.tools}
        if self.llm and self.tools:
            self.llm_with_tools: Any = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = None

        if self.llm:
            logger.info("articleeditor_initialized_with_gpt", tools=len(self.tools))
        else:
            logger.info("articleeditor_disabled")

    def _invoke_config(self, *, workflow: str) -> dict[str, object]:
        return {
            "metadata": {
                **self._tracing_metadata,
                "workflow": workflow,
                "component": "article_editor",
            }
        }

    def _build_structured_review_llm(self):
        """Build a strict structured-output review model when supported."""
        if not self.llm or not hasattr(self.llm, "with_structured_output"):
            return None

        try:
            return self.llm.with_structured_output(EditorReviewResult, strict=True)
        except Exception as exc:
            logger.warning(
                "editor_structured_output_unavailable",
                **summarize_exception(
                    exc, operation="editor_structured_output_unavailable"
                ),
            )
            return None

    def _load_prompt_config(self) -> dict:
        """Load editor prompt configuration."""
        prompt_file = config.prompts_dir / "editor.json"

        if not prompt_file.exists():
            logger.warning("editor_json_not_found_using_minimal_config")
            return {"system_message": "You are an Editor-in-Chief reviewing articles."}

        try:
            with open(prompt_file) as f:
                return cast(dict, json.load(f))
        except json.JSONDecodeError as e:
            logger.error(
                "editor_config_parse_failed",
                **summarize_exception(e, operation="article_editor:parse_config"),
            )
            return {"system_message": "You are an Editor-in-Chief reviewing articles."}

    def is_available(self) -> bool:
        """Check if the editor is available (LLM initialized)."""
        return self.llm is not None

    def build_fact_check_context(
        self,
        data_block: str | None = None,
        pm_block: str | None = None,
        valuation_params: str | None = None,
        voice_samples: str | None = None,
        governance_context: str | None = None,
        consultant_review: str | None = None,
        valuation_context: str | None = None,
    ) -> str:
        """
        Assemble fact-check context for the editor.

        Args:
            data_block: DATA_BLOCK from Senior Fundamentals (ground truth)
            pm_block: PM_BLOCK from Portfolio Manager (verdict, adjusted scores)
            valuation_params: Valuation parameters (target ranges)
            voice_samples: Writing samples for style reference
            valuation_context: The valuation summary given to the writer
                (carries the conditional/suppressed caveat when it fired)

        Returns:
            Formatted context string
        """
        context_parts = []

        if data_block:
            context_parts.append(f"=== DATA_BLOCK (Ground Truth) ===\n{data_block}")

        if pm_block:
            context_parts.append(f"=== PM_BLOCK (Final Verdict) ===\n{pm_block}")

        if valuation_params:
            context_parts.append(f"=== VALUATION PARAMETERS ===\n{valuation_params}")

        if valuation_context:
            context_parts.append(
                "=== VALUATION CONTEXT (as given to writer) ===\n"
                f"{valuation_context}"
            )

        if governance_context:
            context_parts.append(governance_context)

        source_confidence_context = extract_source_confidence_context(
            data_block,
            consultant_review,
        )
        if source_confidence_context:
            context_parts.append(source_confidence_context)

        if voice_samples:
            # Truncate voice samples to control token usage
            truncated_samples = voice_samples[:5000]
            if len(voice_samples) > 5000:
                truncated_samples += "\n...[truncated]"
            context_parts.append(
                f"=== WRITING SAMPLES (Match This Voice) ===\n{truncated_samples}"
            )

        return "\n\n".join(context_parts) if context_parts else "No context provided."

    async def _execute_tool_calls(self, tool_calls: list) -> list[ToolMessage]:
        """
        Execute tool calls from the LLM response.

        Caps execution at MAX_TOOL_CALLS_PER_TURN to prevent parallel call floods
        (OpenAI can return many parallel function calls in a single response).

        Args:
            tool_calls: List of tool call dicts from response.tool_calls

        Returns:
            List of ToolMessage results
        """
        results = []
        capped_calls = tool_calls[: self.MAX_TOOL_CALLS_PER_TURN]
        overflow_calls = tool_calls[self.MAX_TOOL_CALLS_PER_TURN :]
        if overflow_calls:
            logger.warning(
                "capping_tool_calls_per_turn",
                requested=len(tool_calls),
                cap=self.MAX_TOOL_CALLS_PER_TURN,
            )
        url_cache = getattr(self, "_url_cache", {})

        for tc in capped_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", tool_name)

            # Serve cached response for URL fetches (avoids re-hitting paywalls)
            if tool_name == "fetch_reference_content":
                cache_key = tool_args.get("url", "")
                if cache_key in url_cache:
                    logger.info("editor_url_cache_hit", url=cache_key[:80])
                    results.append(
                        ToolMessage(content=url_cache[cache_key], tool_call_id=tool_id)
                    )
                    continue

            tool_fn = self._tools_by_name.get(tool_name)
            if not tool_fn:
                logger.warning("unknown_tool_requested_by_editor", tool=tool_name)
                results.append(
                    ToolMessage(
                        content=f"ERROR: Unknown tool '{tool_name}'",
                        tool_call_id=tool_id,
                    )
                )
                continue

            try:
                logger.info(
                    "editor_calling_tool",
                    tool=tool_name,
                    args_preview=str(tool_args)[:100],
                )

                async def _run_tool(args: dict[str, Any], tool=tool_fn) -> Any:
                    return await tool.ainvoke(args)

                tool_result = await get_current_tool_service().execute(
                    ToolInvocation(
                        name=tool_name,
                        args=tool_args,
                        source="editor",
                        agent_key="article_editor",
                    ),
                    runner=_run_tool,
                )
                content = str(tool_result.value)

                # Cache URL fetch results for this editorial session
                if tool_name == "fetch_reference_content":
                    url_cache[tool_args.get("url", "")] = content

                results.append(ToolMessage(content=content, tool_call_id=tool_id))
            except Exception as e:
                logger.warning(
                    "tool_execution_failed",
                    tool=tool_name,
                    **summarize_exception(e, operation="tool_execution_failed"),
                )
                error_content = f"TOOL_ERROR: {e}"

                # Cache errors too — no point retrying a broken URL
                if tool_name == "fetch_reference_content":
                    url_cache[tool_args.get("url", "")] = error_content

                results.append(ToolMessage(content=error_content, tool_call_id=tool_id))

        # Append SKIPPED messages for overflow calls (after executed results)
        # so the LLM knows they weren't processed
        for tc in overflow_calls:
            skip_id = tc.get("id", tc.get("name", "unknown"))
            results.append(
                ToolMessage(
                    content="SKIPPED: Too many tool calls in one turn. Re-request if needed.",
                    tool_call_id=skip_id,
                )
            )

        return results

    async def review(
        self,
        article_draft: str,
        fact_check_context: str,
    ) -> dict:
        """
        Review an article draft and produce editorial feedback.

        Uses an agentic tool loop: the editor can call fetch_reference_content
        to verify URLs in the article's References section. After tool results
        are returned, the editor produces its final JSON verdict.

        Args:
            article_draft: The article text to review
            fact_check_context: Ground truth data for verification

        Returns:
            Parsed feedback dict with verdict, errors, cuts, etc.
        """
        if not self.llm:
            logger.warning("editor_not_available_approving_article_by_default")
            return {"verdict": "APPROVED", "confidence": 1.0}

        system_message = self.prompt_config.get(
            "system_message",
            "You are an Editor-in-Chief reviewing articles.",
        )

        user_message = f"""FACT-CHECK CONTEXT (Ground Truth):
{fact_check_context}

ARTICLE DRAFT TO REVIEW:
{article_draft}

Review this article carefully:
1. Verify ALL numbers against the DATA_BLOCK and PM_BLOCK ground truth
2. Check that the verdict/recommendation matches PM_BLOCK
3. Flag any unsupported claims or style issues
4. Spot-check 2-3 reference URLs using your fetch_reference_content tool

After your review (including any reference checks), respond with ONLY a JSON object:
```json
{{
  "verdict": "REVISE" or "APPROVED",
  "factual_errors": [
    {{"location": "Section name", "claim": "What article says", "ground_truth": "Correct value", "action": "correct"}}
  ],
  "reference_checks": [
    {{"url": "https://...", "status": "verified" or "broken" or "unsupported", "note": "..."}}
  ],
  "cuts": ["Section - reason for cut"],
  "style_issues": ["Specific issue with location"],
  "confidence": 0.85
}}
```

If there are no issues, use verdict "APPROVED" with empty arrays and high confidence."""

        messages: list = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        try:
            if self.llm_with_tools:
                for iteration in range(self.MAX_TOOL_ITERATIONS):
                    response = await self.llm_with_tools.ainvoke(
                        messages,
                        config=cast(
                            Any, self._invoke_config(workflow="editor_tool_review")
                        ),
                    )

                    tool_calls = getattr(response, "tool_calls", None)
                    if not tool_calls:
                        break

                    logger.info(
                        "editor_requesting_tool_calls",
                        iteration=iteration + 1,
                        num_calls=len(tool_calls),
                    )

                    messages.append(response)
                    tool_messages = await self._execute_tool_calls(tool_calls)
                    messages.extend(tool_messages)
                else:
                    logger.warning(
                        "editor_hit_max_tool_iterations_forcing_final_review",
                        max_iterations=self.MAX_TOOL_ITERATIONS,
                    )

            return await self._invoke_final_review(messages)

        except Exception as e:
            logger.error(
                "editor_review_failed",
                **summarize_exception(e, operation="article_editor:review"),
            )
            # On error, approve to avoid blocking
            return {
                "verdict": "APPROVED",
                "confidence": 0.5,
                "error": str(e),
            }

    async def _invoke_final_review(self, messages: list) -> dict:
        """Run the final strict review pass without tool calls."""
        if not self.llm:
            return {"verdict": "APPROVED", "confidence": 1.0}

        final_messages = messages + [
            HumanMessage(
                content=(
                    "Return only the final review object. "
                    "Do not call tools in this step."
                )
            )
        ]

        if self.review_llm:
            with _structured_review_warning_policy():
                result = await self.review_llm.ainvoke(
                    final_messages,
                    config=self._invoke_config(workflow="editor_structured_review"),
                )
            if isinstance(result, BaseModel):
                return result.model_dump()
            if isinstance(result, dict):
                return result

        response = await self.llm.ainvoke(
            final_messages,
            config=cast(Any, self._invoke_config(workflow="editor_final_review")),
        )
        return self._extract_review_result(response)

    def _extract_review_result(self, response) -> dict:
        """
        Extract and parse the editor's final review result from an LLM response.

        Args:
            response: LLM response object

        Returns:
            Parsed feedback dict
        """
        content = response.content

        logger.debug(
            "editor_raw_response",
            content_type=type(content).__name__,
            content_preview=str(content)[:200] if content else "EMPTY",
        )

        if isinstance(content, list):
            content = " ".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
            )

        if not content or not content.strip():
            logger.warning("editor_returned_empty_response")
            return {"verdict": "APPROVED", "confidence": 0.5, "empty_response": True}

        return self._parse_editor_response(content)

    def _parse_editor_response(self, content: str) -> dict:
        """
        Parse editor response into structured feedback.

        Args:
            content: Raw response content from editor LLM

        Returns:
            Parsed feedback dict
        """
        import re

        # Try to extract JSON from the response
        # Handle cases where JSON is wrapped in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON (greedy match for the outermost braces)
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                json_str = json_match.group(0)
            else:
                # No JSON found - try to infer intent from text
                content_lower = content.lower()
                if any(
                    phrase in content_lower
                    for phrase in [
                        "looks good",
                        "no issues",
                        "well-written",
                        "approve",
                        "no factual errors",
                        "accurate",
                    ]
                ):
                    logger.info("editor_response_appears_to_approve")
                    return {
                        "verdict": "APPROVED",
                        "confidence": 0.7,
                        "inferred": True,
                    }

                logger.warning(
                    "no_json_found_in_editor_response",
                    response_preview=redact_sensitive_text(content, max_chars=200),
                )
                return {"verdict": "APPROVED", "confidence": 0.5, "parse_error": True}

        try:
            feedback = json.loads(json_str)

            # Ensure required fields
            if "verdict" not in feedback:
                feedback["verdict"] = "APPROVED"
            if "confidence" not in feedback:
                feedback["confidence"] = 0.8

            logger.info(
                "editor_feedback_parsed_successfully",
                verdict=feedback.get("verdict"),
                num_errors=len(feedback.get("factual_errors", [])),
            )

            return cast(dict, feedback)

        except json.JSONDecodeError as e:
            logger.warning(
                "failed_to_parse_editor_json",
                **summarize_exception(e, operation="failed_to_parse_editor_json"),
                json_preview=redact_sensitive_text(
                    json_str[:200] if json_str else "EMPTY", max_chars=200
                ),
            )
            return {"verdict": "APPROVED", "confidence": 0.5, "parse_error": True}

    async def edit(
        self,
        writer: "ArticleWriter",
        article_draft: str,
        ticker: str,
        company_name: str,
        data_block: str | None = None,
        pm_block: str | None = None,
        valuation_params: str | None = None,
        voice_samples: str | None = None,
        governance_card: dict[str, Any] | None = None,
        consultant_review: str | None = None,
        valuation_context: str | None = None,
    ) -> tuple[str, dict]:
        """
        Run the full editorial loop: review -> revise -> review.

        Args:
            writer: ArticleWriter instance for revisions
            article_draft: Initial article draft
            ticker: Stock ticker symbol
            company_name: Full company name
            data_block: Ground truth financial data
            pm_block: Portfolio Manager verdict/scores
            valuation_params: Valuation parameters
            voice_samples: Writing samples for style
            valuation_context: Valuation summary given to the writer

        Returns:
            Tuple of (final_article, final_feedback)
        """
        if not self.is_available():
            logger.info("editor_not_available_returning_original_draft")
            return article_draft, {"verdict": "APPROVED", "skipped": True}

        governance_context = ""
        if isinstance(governance_card, dict):
            from src.validators.entity_governance_card import (
                card_to_prompt_block_from_dict,
            )

            governance_context = card_to_prompt_block_from_dict(governance_card)

        # Build fact-check context
        fact_check_context = self.build_fact_check_context(
            data_block=data_block,
            pm_block=pm_block,
            valuation_params=valuation_params,
            voice_samples=voice_samples,
            governance_context=governance_context,
            consultant_review=consultant_review,
            valuation_context=valuation_context,
        )

        current_draft = article_draft
        revision_count = 0

        # Session-scoped URL cache: avoids re-fetching the same URLs across
        # review iterations (e.g. paywalled sites that 401 every time).
        self._url_cache = {}
        try:
            while revision_count < self.MAX_REVISIONS:
                # Audit the article body only — the EDITOR_NOTES meta block may
                # quote the editor's (wrong) claimed values verbatim.
                citation_errors = audit_article_citations(
                    _strip_editor_notes(current_draft), data_block
                )
                if citation_errors:
                    feedback = _review_response_for_citation_errors(citation_errors)
                    logger.warning(
                        "article_citation_audit_failed",
                        ticker=ticker,
                        mismatch_count=len(citation_errors),
                    )
                else:
                    # Review the current draft (with EDITOR_NOTES intact so the
                    # editor can adjudicate CONTESTED items)
                    feedback = _enforce_verdict_consistency(
                        await self.review(current_draft, fact_check_context)
                    )

                logger.info(
                    "editor_review_complete",
                    ticker=ticker,
                    verdict=feedback.get("verdict"),
                    confidence=feedback.get("confidence"),
                    revision=revision_count,
                )

                # Check if approved
                if feedback.get("verdict") == "APPROVED":
                    feedback["revisions"] = revision_count
                    return _strip_editor_notes(current_draft), feedback

                # Revise based on feedback
                revision_count += 1
                logger.info(
                    "revising_article",
                    ticker=ticker,
                    revision=revision_count,
                    max_revisions=self.MAX_REVISIONS,
                )

                current_draft = writer.revise(
                    original_draft=current_draft,
                    editor_feedback=feedback,
                    ticker=ticker,
                    company_name=company_name,
                    governance_context=governance_context,
                )

            # Max revisions reached, do final review
            final_feedback = _enforce_verdict_consistency(
                await self.review(current_draft, fact_check_context)
            )
            current_draft = _strip_editor_notes(current_draft)
            final_citation_errors = audit_article_citations(current_draft, data_block)
            if final_citation_errors:
                current_draft = prepend_verification_caveats(
                    current_draft,
                    final_citation_errors,
                )
                final_feedback["verdict"] = "REVISE"
                final_feedback["deterministic_citation_audit"] = True
                final_feedback["factual_errors"] = final_citation_errors

            # Add revision count to feedback for caller logging
            final_feedback["revisions"] = revision_count

            logger.info(
                "editorial_loop_complete",
                ticker=ticker,
                revisions=revision_count,
                final_verdict=final_feedback.get("verdict"),
            )

            return current_draft, final_feedback
        finally:
            self._url_cache = {}


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
    from src.observability import flush_traces, get_observability_runtime

    runtime = get_observability_runtime(config)
    session_id = f"article-{ticker}-{trade_date}"
    trace_context = runtime.start_article_trace(
        ticker=ticker,
        session_id=session_id,
        tags=["article", f"ticker:{ticker}"],
        metadata={
            "ticker": ticker,
            "environment": config.environment,
            "run_mode": "article_only",
            "prompt_source": (
                "langfuse" if config.langfuse_prompt_fetch_enabled else "local"
            ),
            "release": config.app_release,
        },
        input_payload={"ticker": ticker, "trade_date": trade_date},
    )
    try:
        writer = ArticleWriter(
            use_github_urls=use_github_urls,
            callbacks=trace_context.callbacks,
            tracing_metadata={
                **trace_context.graph_metadata,
                "workflow": "article_only",
            },
        )
        return writer.write(
            ticker=ticker,
            company_name=company_name,
            report_text=report_text,
            trade_date=trade_date,
            output_path=Path(output_path) if output_path else None,
            valuation_context=valuation_context,
        )
    finally:
        trace_context.close()
        flush_traces()
