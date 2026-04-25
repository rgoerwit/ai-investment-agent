"""CLI parsing and output-path helpers for the main runtime entrypoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import config


@dataclass(frozen=True)
class OutputTargets:
    """Resolved output and chart destinations for a CLI run."""

    output_file: Path | None
    image_dir: Path
    skip_charts: bool

    @property
    def output_dir(self) -> Path:
        return self.output_file.parent if self.output_file else Path.cwd()


def _cli_logging_mode(
    args,
) -> Literal["quiet", "brief", "normal", "verbose", "debug"]:
    if getattr(args, "quiet", False):
        return "quiet"
    if getattr(args, "brief", False):
        return "brief"
    if getattr(args, "debug", False):
        return "debug"
    if getattr(args, "verbose", False):
        return "verbose"
    return "normal"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Investment Analysis System (Gemini 3 Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  poetry run python -m src.main --ticker AAPL

  # Quick analysis mode (Gemini Flash)
  poetry run python -m src.main --ticker NVDA --quick

  # Strict quality gate (tighter thresholds, fewer BUYs, token savings on rejects)
  poetry run python -m src.main --ticker 0005.HK --strict

  # Composable: strict quality bar + quick/cheap models
  poetry run python -m src.main --ticker 0005.HK --strict --quick

  # Quiet mode (markdown report only)
  poetry run python -m src.main --ticker AAPL --quiet

  # Brief mode (header, summary, decision only)
  poetry run python -m src.main --ticker AAPL --brief

  # Custom models
  poetry run python -m src.main --ticker TSLA --quick-model gemini-2.5-flash --deep-model gemini-3-pro-preview

  # Enable Langfuse tracing for this run
  poetry run python -m src.main --ticker 0005.HK --enable-langfuse

  # Batch retrospective: process all past tickers
  poetry run python -m src.main --retrospective-only

  # Activated venv alternative
  python -m src.main --ticker MSFT --quick
        """,
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=False,
        default=None,
        help="Stock ticker symbol to analyze (e.g., AAPL, NVDA, TSLA)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick analysis mode (faster, less detailed)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Apply stricter financial health criteria: tighter D/E and coverage thresholds, "
            "auto-reject REITs/ETFs, PFIC, and VIE structures, escalate value-trap warnings "
            "to rejects, and require higher conviction for BUY. Reduces BUY count and saves "
            "tokens by rejecting candidates before Bull/Bear debate. Composable with --quick."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging and output only final markdown report",
    )

    parser.add_argument(
        "--brief",
        action="store_true",
        help="Output only header, executive summary, and decision rationale",
    )

    parser.add_argument(
        "--quick-model",
        type=str,
        default=None,
        help=f"Model to use for quick analysis (default: {config.quick_think_llm})",
    )

    parser.add_argument(
        "--deep-model",
        type=str,
        default=None,
        help=f"Model to use for deep analysis (default: {config.deep_think_llm})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable high-signal application diagnostics",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable developer debug logging; use INVESTMENT_AGENT_TRACE_HTTP=1 "
            "for raw transport traces"
        ),
    )

    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent memory (ChromaDB)",
    )

    parser.add_argument(
        "--svg",
        action="store_true",
        help="Generate charts in SVG format (default: PNG)",
    )

    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Use transparent background for charts (default: white grid)",
    )

    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation entirely",
    )

    parser.add_argument(
        "--enable-langfuse",
        action="store_true",
        help=(
            "Enable Langfuse tracing for this run. Requires LANGFUSE_PUBLIC_KEY "
            "and LANGFUSE_SECRET_KEY."
        ),
    )

    parser.add_argument(
        "--trace-langfuse",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout). If set, images will default to {output_dir}/images",
    )

    parser.add_argument(
        "--imagedir",
        type=str,
        default=None,
        help="Directory for chart images. If --output is set, defaults to {output_dir}/images. If not set, defaults to 'images' in current dir.",
    )

    parser.add_argument(
        "--article",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Generate a Medium-style article from the analysis. "
            "Can specify output path (e.g., --article article.md) or use default "
            "(e.g., --article generates {ticker}_article.md in results dir)."
        ),
    )

    parser.add_argument(
        "--retrospective-only",
        action="store_true",
        help="Run retrospective evaluation on all past analyses without running "
        "a new analysis. Processes all tickers found in results directory.",
    )

    parser.add_argument(
        "--capture-baseline",
        action="store_true",
        help=(
            "Capture a versioned baseline bundle for this run under evals/captures/. "
            "Does not perform evaluation or baseline promotion."
        ),
    )

    parser.add_argument(
        "--capture-baseline-cleanup",
        action="store_true",
        help=(
            "Clean up stale inflight baseline captures under evals/captures/ and exit, "
            "or run cleanup before capture when combined with --capture-baseline."
        ),
    )

    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if (
        not args.retrospective_only
        and not args.capture_baseline_cleanup
        and not args.ticker
    ):
        parser.error(
            "--ticker is required unless --retrospective-only or "
            "--capture-baseline-cleanup is specified"
        )

    if args.debug:
        args.verbose = True

    return args


def resolve_output_paths(args) -> tuple[Path | None, Path]:
    """Determine output file and image directory based on arguments."""
    output_file = Path(args.output) if args.output else None
    output_dir = output_file.parent if output_file else Path.cwd()

    if args.imagedir:
        image_dir = Path(args.imagedir)
    elif output_file:
        image_dir = output_dir / "images"
    else:
        image_dir = Path("images")

    return output_file, image_dir


def validate_imagedir(imagedir: str) -> Path:
    """Validate image directory path."""
    return Path(imagedir)


def resolve_article_path(args, ticker: str) -> Path | None:
    """Determine article output path based on arguments."""
    if not args.article:
        return None

    if isinstance(args.article, str):
        article_path = Path(args.article)
        if not article_path.suffix:
            article_path = article_path.with_suffix(".md")

        if article_path.is_absolute():
            return article_path

        if args.output:
            output_dir = Path(args.output).parent
            return output_dir / article_path

        return article_path

    if args.article is True:
        if args.output:
            output_path = Path(args.output)
            stem = output_path.stem
            suffix = output_path.suffix or ".md"
            article_name = f"{stem}_article{suffix}"
            return output_path.parent / article_name

        safe_ticker = ticker.replace(".", "_").replace("/", "_")
        return config.results_dir / f"{safe_ticker}_article.md"

    return None


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate incompatible flag combinations."""
    if not args.quick:
        return

    chart_flags = [
        flag
        for flag, value in [("--transparent", args.transparent), ("--svg", args.svg)]
        if value
    ]
    if not chart_flags:
        return

    flags_str = " and ".join(chart_flags)
    verb = "has" if len(chart_flags) == 1 else "have"
    noun = "that flag" if len(chart_flags) == 1 else "those flags"
    print(
        f"error: {flags_str} {verb} no effect with --quick "
        f"(chart generation is skipped in quick mode). "
        f"Remove {noun} or drop --quick.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _resolve_output_targets(args: argparse.Namespace) -> OutputTargets:
    """Resolve output/image paths and derived chart behavior for this run."""
    output_file, image_dir = resolve_output_paths(args)
    skip_charts = bool(args.no_charts or (output_file is None and not args.imagedir))
    return OutputTargets(
        output_file=output_file,
        image_dir=image_dir,
        skip_charts=skip_charts,
    )
