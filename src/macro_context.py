"""Pre-graph cached regional macro brief generation and retrieval.

This cache is intentionally separate from ``MacroEventsStore``:
- ``MacroEventsStore`` stores sparse portfolio-detected discrete shocks.
- This module stores a short region-scoped regime brief with TTL-based refresh.

The cache is file-backed under ``results/.macro_context_cache`` so it stays near
other runtime artifacts without cluttering the user-facing result list.

Summarizer invocations use the same callback path as other traced LLM surfaces
so token/cost reporting and Langfuse generation tracing stay aligned with the
rest of the analysis run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from src.config import config
from src.macro_regions import infer_macro_region

logger = structlog.get_logger(__name__)

_CACHE_DIR = config.results_dir / ".macro_context_cache"
_DEFAULT_TTL_HOURS = 12
_THIN_MIN_CHARS = 500
_FINGERPRINT_SCHEMA = "MACRO_QUERY_V1\nMACRO_OUTPUT_V1"


@dataclass(frozen=True, slots=True)
class MacroContextResult:
    """Result of pre-graph macro-context lookup/generation.

    This path is advisory prompt context, not a publishable artifact. The extra
    metadata keeps saved analysis reporting honest about whether the summarizer
    actually ran.
    """

    report: str
    region: str
    status: str  # cached | generated | generated_fallback | failed
    generated_at: str | None = None
    llm_invoked: bool = False
    prompt_used: dict[str, Any] | None = None


def _cache_path(region: str) -> Path:
    return _CACHE_DIR / f"{region}.json"


def get_macro_context_cache_dir() -> Path:
    """Return the on-disk cache directory for regional macro briefs."""
    return Path(config.results_dir) / ".macro_context_cache"


def _prompt_metadata() -> dict[str, Any] | None:
    from src.prompts import get_prompt

    prompt = get_prompt("macro_context_analyst")
    if not prompt:
        return None
    return {
        "agent_name": prompt.agent_name,
        "version": prompt.version,
        "category": prompt.category,
        "requires_tools": prompt.requires_tools,
        "source": prompt.source,
        "execution_path": "pre_graph",
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _compute_fingerprint(prompt_text: str | None = None) -> str:
    """Derive cache invalidation fingerprint from the live summarizer prompt."""
    if prompt_text is None:
        from src.prompts import get_prompt

        prompt = get_prompt("macro_context_analyst")
        prompt_text = prompt.system_message if prompt else ""
    payload = f"{prompt_text}\n{_FINGERPRINT_SCHEMA}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_fresh(
    cached: dict[str, Any] | None,
    *,
    trade_date: str,
    fingerprint: str,
    ttl_hours: int,
) -> bool:
    if not cached:
        return False
    if cached.get("trade_date") != trade_date:
        return False
    if cached.get("fingerprint") != fingerprint:
        return False
    generated_at = cached.get("generated_at")
    if not generated_at:
        return False
    try:
        age = _utc_now() - datetime.fromisoformat(generated_at)
    except ValueError:
        return False
    return age < timedelta(hours=ttl_hours)


def _read_cache(
    region: str,
    *,
    trade_date: str,
    fingerprint: str,
    ttl_hours: int,
) -> dict[str, Any] | None:
    path = _cache_path(region)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "macro_context_cache_read_failed",
            region=region,
            path=str(path),
            error=str(exc),
        )
        return None

    if not _is_fresh(
        payload,
        trade_date=trade_date,
        fingerprint=fingerprint,
        ttl_hours=ttl_hours,
    ):
        return None
    return payload


def _write_cache(
    region: str,
    *,
    trade_date: str,
    fingerprint: str,
    report: str,
    status: str,
) -> str:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now_iso()
    payload = {
        "version": 1,
        "region": region,
        "trade_date": trade_date,
        "fingerprint": fingerprint,
        "generated_at": generated_at,
        "status": status,
        "report": report,
    }
    path = _cache_path(region)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return generated_at


def _merge_macro_callbacks(
    callbacks: list[BaseCallbackHandler] | None = None,
) -> list[BaseCallbackHandler]:
    """Merge token tracking, active trace callbacks, and caller overrides."""
    from src.observability import get_current_trace_context
    from src.token_tracker import TokenTrackingCallback, get_tracker

    merged: list[BaseCallbackHandler] = [
        TokenTrackingCallback(
            "Macro Context Analyst",
            get_tracker(),
            output_token_cap=900,
        )
    ]
    seen = {id(merged[0])}
    trace_context = get_current_trace_context()

    for callback in (trace_context.callbacks if trace_context else []) or []:
        if id(callback) in seen:
            continue
        merged.append(callback)
        seen.add(id(callback))

    for callback in callbacks or []:
        if id(callback) in seen:
            continue
        merged.append(callback)
        seen.add(id(callback))

    return merged


async def _fetch_macro_raw(trade_date: str, region: str) -> str:
    """Call the shared Tavily-backed macro tool with an optional region hint."""
    from src.tools.news import get_macroeconomic_news

    payload = {"trade_date": trade_date}
    if region:
        payload["region"] = region
    return await get_macroeconomic_news.ainvoke(payload)


def _is_thin(raw: str | None) -> bool:
    """Return True when raw macro results are too sparse to summarize safely."""
    text = (raw or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if "tool unavailable" in lowered or "timed out or failed" in lowered:
        return True
    return text.count("<result") < 2 or len(text) < _THIN_MIN_CHARS


async def _summarize(
    raw: str,
    region: str,
    trade_date: str,
    *,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, bool, dict[str, Any] | None]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.agents.message_utils import extract_string_content
    from src.agents.runtime import invoke_with_rate_limit_handling
    from src.llms import create_quick_thinking_llm
    from src.prompts import get_prompt

    prompt = get_prompt("macro_context_analyst")
    if not prompt:
        logger.warning("macro_context_prompt_missing")
        return "", False, None

    llm = create_quick_thinking_llm(
        temperature=0.1,
        max_output_tokens=900,
        callbacks=_merge_macro_callbacks(callbacks),
    )
    response = await invoke_with_rate_limit_handling(
        llm,
        [
            SystemMessage(content=prompt.system_message),
            HumanMessage(
                content=(
                    f"Region: {region}\n"
                    f"Date: {trade_date}\n\n"
                    f"Raw search results:\n{raw}"
                )
            ),
        ],
        context=prompt.agent_name,
    )
    return (
        extract_string_content(getattr(response, "content", response)).strip(),
        True,
        _prompt_metadata(),
    )


async def get_macro_context(
    ticker: str,
    trade_date: str,
    *,
    ttl_hours: int = _DEFAULT_TTL_HOURS,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> MacroContextResult:
    """Return a cached or freshly generated regional macro brief.

    This helper is intentionally non-fatal. Failures degrade to an empty report
    with ``status="failed"`` so the main analysis path continues.
    """
    region = infer_macro_region(ticker)
    fingerprint = _compute_fingerprint()

    cached = _read_cache(
        region,
        trade_date=trade_date,
        fingerprint=fingerprint,
        ttl_hours=ttl_hours,
    )
    if cached is not None:
        logger.info(
            "macro_context_cache_hit",
            ticker=ticker,
            region=region,
            cache_path=str(_cache_path(region)),
        )
        return MacroContextResult(
            report=str(cached.get("report", "")),
            region=region,
            status="cached",
            generated_at=cached.get("generated_at"),
        )

    try:
        raw = await _fetch_macro_raw(trade_date, region)
        status = "generated"

        if region != "GLOBAL" and _is_thin(raw):
            logger.info(
                "macro_context_regional_thin",
                ticker=ticker,
                region=region,
                trade_date=trade_date,
                cache_path=str(_cache_path(region)),
            )
            fallback_raw = await _fetch_macro_raw(trade_date, "GLOBAL")
            if not _is_thin(fallback_raw):
                raw = fallback_raw
                status = "generated_fallback"

        if _is_thin(raw):
            logger.warning(
                "macro_context_generation_failed",
                ticker=ticker,
                region=region,
                trade_date=trade_date,
                cache_path=str(_cache_path(region)),
            )
            return MacroContextResult("", region, "failed")

        report, llm_invoked, prompt_used = await _summarize(
            raw,
            region,
            trade_date,
            callbacks=callbacks,
        )
        if not report:
            logger.warning(
                "macro_context_summary_empty",
                ticker=ticker,
                region=region,
                trade_date=trade_date,
                llm_invoked=llm_invoked,
            )
            return MacroContextResult(
                "",
                region,
                "failed",
                llm_invoked=llm_invoked,
                prompt_used=prompt_used,
            )

        generated_at = _write_cache(
            region,
            trade_date=trade_date,
            fingerprint=fingerprint,
            report=report,
            status=status,
        )
        logger.info(
            "macro_context_generated",
            ticker=ticker,
            region=region,
            status=status,
            used_global_fallback=(status == "generated_fallback"),
            cache_path=str(_cache_path(region)),
            generated_at=generated_at,
        )
        return MacroContextResult(
            report=report,
            region=region,
            status=status,
            generated_at=generated_at,
            llm_invoked=llm_invoked,
            prompt_used=prompt_used,
        )
    except Exception as exc:
        logger.warning(
            "macro_context_failed",
            ticker=ticker,
            region=region,
            trade_date=trade_date,
            error=str(exc),
        )
        return MacroContextResult("", region, "failed")
