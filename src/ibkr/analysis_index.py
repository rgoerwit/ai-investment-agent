"""
Latest-analysis index and snapshot-loading helpers for IBKR workflows.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.currency_resolver import resolve_local_trading_currency
from src.data_block_utils import extract_kill_criteria
from src.error_safety import summarize_exception
from src.fx_normalization import get_fx_rate_fallback
from src.ibkr.models import AnalysisRecord, PortfolioEvidence, TradeBlockData
from src.ibkr.order_builder import parse_trade_block
from src.ibkr.reconciliation_rules import _exchange_from_ticker, _normalize_verdict
from src.pm_decision_parser import parse_final_decision_scores
from src.sector_normalization import normalize_sector_label
from src.validators.financial_rules import detect_red_flags
from src.validators.metric_extractor import extract_metrics
from src.validators.quality_flags import PEAK_OR_TRANSIENT_FLAGS
from src.validators.supplemental_flags import (
    detect_capital_efficiency_flags,
    detect_moat_flags,
)

logger = structlog.get_logger(__name__)
# Bump when the parsed AnalysisRecord schema changes so cached indexes rebuild.
# v5: added risk_tally + quality_flag_types (BUY stability gate inputs).
# v6: added PortfolioEvidence (dni_review_candidate marker, blocks_buy /
#     compliance flag types) — disposition classifier inputs.
# v7: is_quick_mode tri-state (None = snapshot predates the field — cached
#     v6 records stored the false-full default, which granted legacy
#     artifacts sell-confirmation authority).
# v8: kill_criteria (bear thesis-break triggers from the saved bear history) —
#     the fundamental exit conditions surfaced ahead of legacy downside levels.
_ANALYSIS_INDEX_VERSION = 8
_DATA_VACUUM_COVERAGE_THRESHOLD_PCT = 40.0


@dataclass(frozen=True, slots=True)
class AnalysisLoadProgress:
    """Progress update emitted while scanning/parsing saved analysis snapshots."""

    phase: str
    total_files: int
    processed_files: int
    loaded_analyses: int
    current_file: str | None = None


def _analysis_index_path(results_dir: Path) -> Path:
    """Return the sibling cache file that stores latest-per-ticker analysis records."""
    return results_dir.parent / f".{results_dir.name}.latest_analyses_index.json"


def _analysis_index_lock_path(results_dir: Path) -> Path:
    """Return the sibling lock file used to serialize index updates."""
    return results_dir.parent / f".{results_dir.name}.latest_analyses_index.lock"


# Note: directory mtime is an unreliable freshness signal — it is second-granular
# on some filesystems (APFS reports `st_mtime_ns` ending in `000000000` for
# directories) while the index persists full-precision ns, and two files written
# in the same wall-clock second leave it unchanged. So the mtime check is kept only
# as a cheap *hint*; the authoritative staleness guard is the analysis-file COUNT
# comparison below. Do NOT gate the count check behind an mtime match (a same-second
# addition would evade detection). The `*_mtime_mismatch_accepted` logs are emitted
# at debug to avoid per-load/per-save noise.


@contextmanager
def _analysis_index_lock(results_dir: Path):
    """Serialize index updates across concurrent processes."""
    lock_path = _analysis_index_lock_path(results_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_handle:
        wait_started = time.perf_counter()
        logger.debug("analysis_index_lock_waiting", path=str(lock_path))
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        logger.debug(
            "analysis_index_lock_acquired",
            path=str(lock_path),
            wait_secs=round(time.perf_counter() - wait_started, 6),
        )
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _safe_exception_fields(exc: BaseException, *, operation: str) -> dict[str, Any]:
    """Return sanitized structured exception fields for analysis-index logs."""
    return summarize_exception(exc, operation=operation, provider="local_filesystem")


def _serialize_analysis_index_entry(record: AnalysisRecord) -> dict[str, Any]:
    """Serialize an indexed latest-analysis entry with source-file validation metadata."""
    source_path = Path(record.file_path)
    source_stat = source_path.stat()
    return {
        "record": record.model_dump(mode="json"),
        "source_file": str(source_path),
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_size": source_stat.st_size,
    }


def _deserialize_analysis_record(data: dict[str, Any]) -> AnalysisRecord:
    """Deserialize an AnalysisRecord from the latest-analyses cache."""
    record = AnalysisRecord.model_validate(data)
    record.sector = normalize_sector_label(record.sector)
    record.ticker = _sanitize_ticker_key(record.ticker)
    return record


def _validate_analysis_index_entry(
    ticker: str,
    entry: dict[str, Any],
) -> AnalysisRecord | None:
    """Return the cached AnalysisRecord only if its source file still matches."""
    record_payload = entry.get("record")
    source_file = entry.get("source_file")
    source_mtime_ns = entry.get("source_mtime_ns")
    source_size = entry.get("source_size")

    if (
        not isinstance(record_payload, dict)
        or not isinstance(source_file, str)
        or not isinstance(source_mtime_ns, int)
        or not isinstance(source_size, int)
    ):
        logger.warning("analysis_index_entry_invalid", ticker=ticker)
        return None

    source_path = Path(source_file)
    try:
        stat = source_path.stat()
    except OSError:
        logger.warning(
            "analysis_index_entry_source_missing",
            ticker=ticker,
            source_file=source_file,
        )
        return None

    if stat.st_mtime_ns != source_mtime_ns or stat.st_size != source_size:
        logger.warning(
            "analysis_index_entry_stale",
            ticker=ticker,
            source_file=source_file,
        )
        return None

    return _deserialize_analysis_record(record_payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


_FILENAME_DASH_DATE_RE = re.compile(
    r"^(?P<ticker>.+?)_(\d{4}-\d{2}-\d{2})_analysis\.json$"
)
_FILENAME_TIMESTAMP_RE = re.compile(r"^(?P<ticker>.+?)_(\d{8})_(\d{6})_analysis\.json$")


def _parse_scores_from_final_decision(text: str) -> dict:
    """Extract health_adj, growth_adj, verdict, zone, risk_tally from a PM decision.

    Thin delegate to the neutral src.pm_decision_parser so the parser stays
    agents-free and is shared with the BUY stability gate. The verdict is returned
    RAW; the AnalysisRecord call site canonicalizes via _normalize_verdict (so a
    verdict like "REJECT" is preserved verbatim, not collapsed to DO_NOT_INITIATE).
    Private name retained as the existing test surface.
    """
    return parse_final_decision_scores(text)


def _should_emit_analysis_progress(processed_files: int, total_files: int) -> bool:
    """Return True when a user-facing progress update is worth emitting."""
    if total_files <= 0:
        return False
    if total_files > 20 and processed_files in {1, 5, 10, 25, 50, 100}:
        return True
    if total_files <= 20:
        return True
    if total_files <= 200:
        step = 25
    elif total_files <= 1000:
        step = 100
    else:
        step = 250
    return processed_files == total_files or processed_files % step == 0


def _sanitize_ticker_key(ticker: str) -> str:
    """Strip stray punctuation from historical run artifacts (e.g. 'GUD.AX:').

    January 2026 runs were invoked with trailing colons; their saved snapshots
    carry the malformed key, which then self-collides with the clean ticker in
    the base-symbol ambiguity guard.
    """
    cleaned = ticker.strip().rstrip(":;,").upper()
    if cleaned != ticker:
        logger.debug("analysis_ticker_key_sanitized", original=ticker, cleaned=cleaned)
    return cleaned


def _extract_filename_analysis_key(filename: str) -> str | None:
    """Extract the filename-level ticker segment from an analysis snapshot filename."""
    match = _FILENAME_DASH_DATE_RE.match(filename) or _FILENAME_TIMESTAMP_RE.match(
        filename
    )
    if not match:
        return None
    return match.group("ticker")


def _extract_filename_analysis_date(filename: str) -> str:
    """Extract YYYY-MM-DD from an analysis snapshot filename."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else ""


_PROFIT_TAKE_CAPITAL_FLAGS = frozenset(
    {"CAPITAL_IDLE_CASH_RISK", "CAPITAL_IDLE_CASH_SEVERE"}
)


def _extract_flag_types(
    data: dict[str, Any],
    ticker: str,
    *,
    source_file: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Single validator pass → (capital_flag_types, quality_flag_types).

    Runs the validators once over the saved fundamentals/value-trap reports and
    partitions the result into:
      - capital_flag_types: idle-cash capital flags (_PROFIT_TAKE_CAPITAL_FLAGS)
      - quality_flag_types: peak/transient markers (PEAK_OR_TRANSIENT_FLAGS) —
        CYCLICAL_PEAK_WARNING, TRANSIENT_STRENGTH_DISTORTION, and the
        moat/capital-efficiency bonus-suppression flags — for the BUY stability gate.

    Re-derivation is deterministic; any failure degrades both to ().
    """
    reports = data.get("reports") or {}
    fundamentals_report = (
        reports.get("fundamentals_report") or data.get("fundamentals_report") or ""
    )
    if not fundamentals_report:
        return (), ()
    value_trap_report = (
        reports.get("value_trap_report") or data.get("value_trap_report") or ""
    )
    try:
        metrics = extract_metrics(
            fundamentals_report,
            ticker=ticker,
            source_file=source_file,
        )
        red_flags, _ = detect_red_flags(metrics, ticker=ticker)
        moat = detect_moat_flags(
            fundamentals_report,
            ticker=ticker,
            base_metrics=metrics,
        )
        capital = detect_capital_efficiency_flags(
            fundamentals_report,
            ticker=ticker,
            value_trap_report=value_trap_report or None,
            base_metrics=metrics,
        )
    except Exception as exc:
        log_fields = _safe_exception_fields(exc, operation="extracting analysis flags")
        if source_file:
            log_fields["file"] = source_file
        logger.warning(
            "flag_extraction_failed",
            ticker=ticker,
            **log_fields,
        )
        return (), ()

    capital_types = tuple(
        dict.fromkeys(
            flag_type
            for flag in capital
            if (flag_type := flag.get("type")) in _PROFIT_TAKE_CAPITAL_FLAGS
        )
    )
    quality_types = tuple(
        dict.fromkeys(
            flag_type
            for flag in (*red_flags, *moat, *capital)
            if (flag_type := flag.get("type")) in PEAK_OR_TRANSIENT_FLAGS
        )
    )
    return capital_types, quality_types


_COMPLIANCE_FLAG_PREFIXES = ("PFIC_", "VIE_", "CMIC_", "REGULATORY_")


def _extract_portfolio_evidence(data: dict[str, Any]) -> PortfolioEvidence:
    """Assemble decision-layer evidence from already-persisted artifact fields.

    Reads run_summary markers and the root red_flags list — never prose. A
    malformed red_flags payload degrades to empty evidence (complete=False),
    which the disposition classifier treats conservatively.
    """
    run_summary = data.get("run_summary")
    run_summary = run_summary if isinstance(run_summary, dict) else {}
    red_flags = data.get("red_flags")
    flags_valid = isinstance(red_flags, list)
    flags = red_flags if flags_valid else []

    def _types(predicate: Callable[[dict[str, Any]], bool]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(flag["type"])
                for flag in flags
                if isinstance(flag, dict) and flag.get("type") and predicate(flag)
            )
        )

    return PortfolioEvidence(
        # The marker key exists in run_summary from July 2026 onward; its mere
        # presence (True or False) is what distinguishes a marker-aware artifact
        # from a legacy one.
        complete=flags_valid and "verdict_dni_review_candidate" in run_summary,
        dni_review_candidate=bool(run_summary.get("verdict_dni_review_candidate")),
        buy_blocking_flag_types=_types(lambda f: bool(f.get("blocks_buy"))),
        compliance_flag_types=_types(
            lambda f: str(f.get("type", "")).startswith(_COMPLIANCE_FLAG_PREFIXES)
        ),
    )


def _extract_tool1_financial_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Parse the saved Junior Tool 1 financial-metrics JSON payload."""
    raw = (
        ((data.get("source_artifacts") or {}).get("raw_fundamentals_data"))
        or data.get("raw_fundamentals_data")
        or ""
    )
    if not isinstance(raw, str) or not raw:
        return {}
    marker_idx = raw.find("get_financial_metrics")
    if marker_idx < 0:
        return {}
    body = raw[marker_idx:]
    start = body.find("{")
    if start < 0:
        return {}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(body[start:])
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coverage_as_percent(value: Any) -> float | None:
    coverage = _as_float(value)
    if coverage is None:
        return None
    return coverage * 100.0 if 0.0 <= coverage <= 1.0 else coverage


def _extract_analysis_data_quality(
    data: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Extract compact data-quality metadata used by held-position routing."""
    payload = _extract_tool1_financial_metrics(data)
    if not payload:
        return {}

    quality = payload.get("_quality")
    quality = quality if isinstance(quality, dict) else {}

    coverage_pct = _coverage_as_percent(payload.get("_coverage_pct"))
    if coverage_pct is None:
        coverage_pct = _coverage_as_percent(quality.get("coverage_pct"))

    basics_ok = quality.get("basics_ok")
    basics_ok = basics_ok if isinstance(basics_ok, bool) else None

    sources_used = payload.get("_sources_used")
    if not isinstance(sources_used, list):
        sources_used = quality.get("sources_used")
    normalized_sources = (
        [str(source) for source in sources_used if source]
        if isinstance(sources_used, list)
        else []
    )

    ibkr_identity = payload.get("_ibkr_identity_confidence")
    ibkr_probe_error = payload.get("_ibkr_probe_error_kind")
    rescue_original = payload.get("_ticker_rescue_original")
    rescue_resolved = payload.get("_ticker_rescue_resolved")
    rescue_reason = payload.get("_ticker_rescue_reason")
    rescue_ibkr_identity = payload.get("_ticker_rescue_ibkr_identity_confidence")
    rescue_ibkr_error = payload.get("_ticker_rescue_ibkr_probe_error_kind")
    current_price = snapshot.get("current_price")
    if current_price is None:
        current_price = payload.get("currentPrice")

    # Below 40%, the saved DATA_BLOCK is usually too sparse to justify an
    # executable held-position sell without a human data-quality review.
    data_vacuum = (
        basics_ok is False
        and coverage_pct is not None
        and coverage_pct < _DATA_VACUUM_COVERAGE_THRESHOLD_PCT
        and (
            current_price is None
            or not normalized_sources
            or (ibkr_identity not in (None, "", "VERIFIED"))
        )
    )

    return {
        "coverage_pct": coverage_pct,
        "basics_ok": basics_ok,
        "sources_used": normalized_sources,
        "ibkr_identity_confidence": ibkr_identity,
        "ibkr_probe_error_kind": ibkr_probe_error,
        "ticker_rescue_original": rescue_original,
        "ticker_rescue_resolved": rescue_resolved,
        "ticker_rescue_reason": rescue_reason,
        "ticker_rescue_ibkr_identity_confidence": rescue_ibkr_identity,
        "ticker_rescue_ibkr_probe_error_kind": rescue_ibkr_error,
        "data_vacuum": data_vacuum,
    }


def _build_analysis_record_from_data(
    filepath: Path, data: dict[str, Any]
) -> AnalysisRecord | None:
    """Build an AnalysisRecord from a saved analysis payload."""
    snapshot = data.get("prediction_snapshot", {})
    if snapshot.get("health_adj") is None or not snapshot.get("verdict"):
        fd_text = (data.get("final_decision") or {}).get("decision", "") or ""
        if fd_text:
            fallback = _parse_scores_from_final_decision(fd_text)
            if snapshot.get("health_adj") is None:
                snapshot = {**snapshot, "health_adj": fallback.get("health_adj")}
            if snapshot.get("growth_adj") is None:
                snapshot = {**snapshot, "growth_adj": fallback.get("growth_adj")}
            if not snapshot.get("verdict"):
                snapshot = {
                    **snapshot,
                    "verdict": _normalize_verdict(fallback.get("verdict") or ""),
                }
            if not snapshot.get("zone"):
                snapshot = {**snapshot, "zone": fallback.get("zone") or ""}
            if snapshot.get("risk_tally") is None:
                snapshot = {**snapshot, "risk_tally": fallback.get("risk_tally")}

    ticker = snapshot.get("ticker") or data.get("ticker", "")
    if not ticker:
        filename_ticker = _extract_filename_analysis_key(filepath.name)
        if filename_ticker:
            ticker = filename_ticker.replace("_", ".")
        if not ticker:
            return None
    ticker = _sanitize_ticker_key(ticker)

    trader_plan = data.get("investment_analysis", {}).get("trader_plan", "") or ""
    trade_block = parse_trade_block(trader_plan) or TradeBlockData()

    # Thesis-break triggers from the saved bear history (fenced machine block,
    # not free prose). These are the fundamental exit conditions the operator
    # sees ahead of legacy downside levels; legacy artifacts yield ().
    bear_history = (
        data.get("investment_analysis", {})
        .get("investment_debate", {})
        .get("bear_history", "")
        or ""
    )
    kill_criteria = tuple(extract_kill_criteria(bear_history))

    repaired_currency = _repair_legacy_snapshot_currency(
        snapshot,
        ticker=ticker,
        file_name=filepath.name,
    )
    currency = repaired_currency["currency"]
    fx_rate_to_usd = repaired_currency["fx_rate_to_usd"]
    currency_source = repaired_currency["currency_source"]
    currency_repaired = repaired_currency["currency_repaired"]
    currency_repair_reason = repaired_currency["currency_repair_reason"]
    macro_regime_raw = (
        data.get("macro_regime_block") or snapshot.get("regime_at_decision") or {}
    )
    macro_regime = macro_regime_raw if isinstance(macro_regime_raw, dict) else {}
    data_quality = _extract_analysis_data_quality(data, snapshot)
    capital_flag_types, quality_flag_types = _extract_flag_types(
        data,
        ticker,
        source_file=filepath.name,
    )

    return AnalysisRecord(
        ticker=ticker,
        analysis_date=snapshot.get("analysis_date", "")
        or _extract_filename_analysis_date(filepath.name),
        file_path=str(filepath),
        verdict=_normalize_verdict(snapshot.get("verdict", "") or ""),
        health_adj=snapshot.get("health_adj"),
        growth_adj=snapshot.get("growth_adj"),
        zone=snapshot.get("zone") or "",
        position_size=snapshot.get("position_size"),
        current_price=snapshot.get("current_price"),
        currency=currency,
        currency_source=currency_source,
        fx_rate_to_usd=fx_rate_to_usd,
        currency_repaired=currency_repaired,
        currency_repair_reason=currency_repair_reason,
        trade_block=trade_block,
        entry_price=snapshot.get("entry_price") or trade_block.entry_price,
        stop_price=snapshot.get("stop_price") or trade_block.stop_price,
        target_1_price=snapshot.get("target_1_price") or trade_block.target_1_price,
        target_2_price=snapshot.get("target_2_price") or trade_block.target_2_price,
        conviction=snapshot.get("conviction") or trade_block.conviction,
        sector=normalize_sector_label(snapshot.get("sector")),
        exchange=snapshot.get("exchange") or _exchange_from_ticker(ticker),
        is_quick_mode=(
            None
            if snapshot.get("is_quick_mode") is None
            else bool(snapshot["is_quick_mode"])
        ),
        capital_flag_types=capital_flag_types,
        risk_tally=snapshot.get("risk_tally"),
        quality_flag_types=quality_flag_types,
        kill_criteria=kill_criteria,
        macro_regime=macro_regime,
        data_quality=data_quality,
        m_and_a_status=(snapshot.get("m_and_a_status") or "").strip().upper(),
        evidence=_extract_portfolio_evidence(data),
    )


def _repair_legacy_snapshot_currency(
    snapshot: dict[str, Any], *, ticker: str, file_name: str
) -> dict[str, Any]:
    """Repair legacy snapshots that were incorrectly persisted as USD."""
    actual_currency = (snapshot.get("currency") or "USD").upper()
    fx_rate_to_usd = snapshot.get("fx_rate_to_usd")
    currency_source = snapshot.get("currency_source")
    currency_repaired = False
    currency_repair_reason = None

    resolution = resolve_local_trading_currency(ticker=ticker)
    suspicious_usd = actual_currency == "USD" and fx_rate_to_usd in {None, 1.0}
    missing_currency = not snapshot.get("currency")

    if (
        resolution.source == "exchange_suffix"
        and resolution.code
        and resolution.code != "USD"
        and (missing_currency or suspicious_usd)
    ):
        # Canonical fallback-table conversion (minor-unit + USD anchoring).
        # Deliberately the table, not live/cache: this reconstructs the rate a
        # past snapshot should have had, so a live rate would be wrong.
        repaired_fx_rate = get_fx_rate_fallback(resolution.code, "USD")
        logger.debug(
            "legacy_snapshot_currency_repaired",
            ticker=ticker,
            from_currency=actual_currency,
            repaired_currency=resolution.code,
            file=file_name,
        )
        return {
            "currency": resolution.code,
            "fx_rate_to_usd": repaired_fx_rate,
            "currency_source": "repair_on_load",
            "currency_repaired": True,
            "currency_repair_reason": "legacy_snapshot_usd_default",
        }

    return {
        "currency": actual_currency,
        "fx_rate_to_usd": fx_rate_to_usd,
        "currency_source": currency_source,
        "currency_repaired": currency_repaired,
        "currency_repair_reason": currency_repair_reason,
    }


def _build_analysis_record_from_file(filepath: Path) -> AnalysisRecord | None:
    """Load a saved analysis JSON and convert it to an AnalysisRecord."""
    with open(filepath) as handle:
        data = json.load(handle)
    return _build_analysis_record_from_data(filepath, data)


def _load_latest_analyses_from_index(
    results_dir: Path,
    *,
    current_dir_mtime_ns: int,
    progress: Callable[[AnalysisLoadProgress], None] | None = None,
) -> dict[str, AnalysisRecord] | None:
    """Return cached latest analyses if the results directory has not changed."""
    index_path = _analysis_index_path(results_dir)
    if not index_path.exists():
        return None

    def emit_rebuild_notice(current_file: str, total_files: int = 0) -> None:
        if progress is None:
            return
        progress(
            AnalysisLoadProgress(
                phase="rebuilding_index",
                total_files=total_files,
                processed_files=0,
                loaded_analyses=0,
                current_file=current_file,
            )
        )

    try:
        with open(index_path) as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "analysis_index_load_failed",
            path=str(index_path),
            **_safe_exception_fields(exc, operation="loading latest analyses index"),
        )
        emit_rebuild_notice(index_path.name)
        return None

    if payload.get("version") != _ANALYSIS_INDEX_VERSION:
        emit_rebuild_notice(f"{index_path.name}:version_mismatch")
        return None
    if payload.get("results_dir") != str(results_dir.resolve()):
        emit_rebuild_notice(f"{index_path.name}:path_mismatch")
        return None
    if payload.get("results_dir_mtime_ns") != current_dir_mtime_ns:
        indexed_total_files = int(payload.get("total_files") or 0)
        current_analysis_file_count = sum(
            1 for _ in results_dir.glob("*_analysis.json")
        )
        if indexed_total_files != current_analysis_file_count:
            emit_rebuild_notice(f"{index_path.name}:stale_directory_state")
            return None
        logger.debug(
            "analysis_index_mtime_mismatch_accepted",
            path=str(index_path),
            indexed_total_files=indexed_total_files,
            current_analysis_file_count=current_analysis_file_count,
            index_dir_mtime_ns=payload.get("results_dir_mtime_ns"),
            current_dir_mtime_ns=current_dir_mtime_ns,
        )

    analyses: dict[str, AnalysisRecord] = {}
    for ticker, entry in (payload.get("analyses") or {}).items():
        record = _validate_analysis_index_entry(ticker, entry)
        if record is None:
            emit_rebuild_notice(
                f"{index_path.name}:entry_invalid:{ticker}",
                total_files=int(payload.get("total_files") or 0),
            )
            return None
        # Key by the (sanitized) record ticker, not the raw cache key: legacy
        # caches may hold malformed keys like "GUD.AX:" alongside the clean
        # twin — keep whichever analysis is fresher when they merge.
        key = record.ticker
        existing = analyses.get(key)
        if existing is None or record.analysis_date >= existing.analysis_date:
            analyses[key] = record
    total_files = int(payload.get("total_files") or len(analyses))

    if progress is not None:
        progress(
            AnalysisLoadProgress(
                phase="indexed",
                total_files=total_files,
                processed_files=total_files,
                loaded_analyses=len(analyses),
                current_file=None,
            )
        )

    logger.info("analyses_loaded_from_index", count=len(analyses), path=str(index_path))
    return analyses


def _write_latest_analyses_index(
    results_dir: Path,
    analyses: dict[str, AnalysisRecord],
    *,
    total_files: int,
) -> None:
    """Persist the latest-per-ticker cache for future fast loads."""
    index_path = _analysis_index_path(results_dir)
    payload = {
        "version": _ANALYSIS_INDEX_VERSION,
        "results_dir": str(results_dir.resolve()),
        "results_dir_mtime_ns": results_dir.stat().st_mtime_ns,
        "total_files": total_files,
        "analyses": {
            ticker: _serialize_analysis_index_entry(record)
            for ticker, record in analyses.items()
        },
    }
    try:
        with _analysis_index_lock(results_dir):
            _atomic_write_json(index_path, payload)
    except OSError as exc:
        logger.warning(
            "analysis_index_write_failed",
            path=str(index_path),
            **_safe_exception_fields(exc, operation="writing latest analyses index"),
        )


def update_latest_analyses_index(
    results_dir: Path,
    record: AnalysisRecord,
    *,
    previous_dir_mtime_ns: int | None,
    analysis_file_count_before_save: int | None = None,
) -> bool:
    """Incrementally update a valid latest-analyses index after saving one analysis."""
    index_path = _analysis_index_path(results_dir)
    if previous_dir_mtime_ns is None:
        logger.info(
            "analysis_index_incremental_update_skipped",
            ticker=record.ticker,
            path=str(index_path),
            reason="missing_previous_dir_mtime",
        )
        return False
    if not index_path.exists():
        logger.info(
            "analysis_index_incremental_update_skipped",
            ticker=record.ticker,
            path=str(index_path),
            reason="index_missing",
        )
        return False

    try:
        with _analysis_index_lock(results_dir):
            try:
                with open(index_path) as handle:
                    payload = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "analysis_index_incremental_update_failed",
                    path=str(index_path),
                    **_safe_exception_fields(
                        exc,
                        operation="loading latest analyses index for incremental update",
                    ),
                )
                return False

            if payload.get("version") != _ANALYSIS_INDEX_VERSION:
                logger.info(
                    "analysis_index_incremental_update_skipped",
                    ticker=record.ticker,
                    path=str(index_path),
                    reason="version_mismatch",
                )
                return False
            if payload.get("results_dir") != str(results_dir.resolve()):
                logger.info(
                    "analysis_index_incremental_update_skipped",
                    ticker=record.ticker,
                    path=str(index_path),
                    reason="results_dir_mismatch",
                )
                return False
            indexed_total_files = int(payload.get("total_files") or 0)
            if payload.get("results_dir_mtime_ns") != previous_dir_mtime_ns:
                if (
                    analysis_file_count_before_save is not None
                    and indexed_total_files == analysis_file_count_before_save
                ):
                    logger.debug(
                        "analysis_index_incremental_update_mtime_mismatch_accepted",
                        ticker=record.ticker,
                        path=str(index_path),
                        expected_previous_dir_mtime_ns=previous_dir_mtime_ns,
                        index_dir_mtime_ns=payload.get("results_dir_mtime_ns"),
                        analysis_file_count_before_save=analysis_file_count_before_save,
                        indexed_total_files=indexed_total_files,
                        source_file=record.file_path,
                    )
                else:
                    logger.info(
                        "analysis_index_incremental_update_skipped",
                        ticker=record.ticker,
                        path=str(index_path),
                        reason="stale_directory_state",
                        expected_previous_dir_mtime_ns=previous_dir_mtime_ns,
                        index_dir_mtime_ns=payload.get("results_dir_mtime_ns"),
                        current_dir_mtime_ns=results_dir.stat().st_mtime_ns,
                        source_file=record.file_path,
                        analysis_file_count_before_save=analysis_file_count_before_save,
                        indexed_total_files=indexed_total_files,
                    )
                    return False

            analyses_payload = dict(payload.get("analyses") or {})
            analyses_payload[record.ticker] = _serialize_analysis_index_entry(record)
            updated_payload = {
                "version": _ANALYSIS_INDEX_VERSION,
                "results_dir": str(results_dir.resolve()),
                "results_dir_mtime_ns": results_dir.stat().st_mtime_ns,
                "total_files": indexed_total_files + 1,
                "analyses": analyses_payload,
            }
            _atomic_write_json(index_path, updated_payload)
    except OSError as exc:
        logger.warning(
            "analysis_index_incremental_write_failed",
            path=str(index_path),
            **_safe_exception_fields(
                exc,
                operation="writing latest analyses index incremental update",
            ),
        )
        return False
    logger.info(
        "analysis_index_incremental_updated",
        ticker=record.ticker,
        path=str(index_path),
        source_file=record.file_path,
    )
    return True


def load_latest_analyses(
    results_dir: Path,
    *,
    progress: Callable[[AnalysisLoadProgress], None] | None = None,
) -> dict[str, AnalysisRecord]:
    """Load the most recent analysis JSON for each ticker from results_dir."""
    if not results_dir.exists():
        logger.warning("results_dir_not_found", path=str(results_dir))
        return {}

    current_dir_mtime_ns = results_dir.stat().st_mtime_ns
    indexed = _load_latest_analyses_from_index(
        results_dir,
        current_dir_mtime_ns=current_dir_mtime_ns,
        progress=progress,
    )
    if indexed is not None:
        return indexed

    analyses: dict[str, AnalysisRecord] = {}
    filepaths = sorted(results_dir.glob("*_analysis.json"), reverse=True)
    total_files = len(filepaths)
    failed_files = 0
    duplicate_files = 0
    filename_duplicates_skipped = 0
    missing_ticker_files = 0
    seen_filename_keys: set[str] = set()

    if progress is not None:
        progress(
            AnalysisLoadProgress(
                phase="discovered",
                total_files=total_files,
                processed_files=0,
                loaded_analyses=0,
            )
        )

    heartbeat_state: dict[str, int | str] = {"file": "", "n": 0, "loaded": 0}
    heartbeat_stop = threading.Event()

    def _heartbeat_worker() -> None:
        while not heartbeat_stop.wait(timeout=30.0):
            if progress is not None and heartbeat_state["file"]:
                progress(
                    AnalysisLoadProgress(
                        phase="parsing",
                        total_files=total_files,
                        processed_files=int(heartbeat_state["n"]),
                        loaded_analyses=int(heartbeat_state["loaded"]),
                        current_file=str(heartbeat_state["file"]),
                    )
                )

    threading.Thread(
        target=_heartbeat_worker, daemon=True, name="index-scan-heartbeat"
    ).start()

    for processed_files, filepath in enumerate(filepaths, start=1):
        heartbeat_state["file"] = filepath.name
        heartbeat_state["n"] = processed_files
        heartbeat_state["loaded"] = len(analyses)
        filename_key = _extract_filename_analysis_key(filepath.name)

        def emit_progress(
            processed_files_: int = processed_files,
            current_file: str = filepath.name,
        ) -> None:
            if progress is None or not _should_emit_analysis_progress(
                processed_files_, total_files
            ):
                return
            progress(
                AnalysisLoadProgress(
                    phase="parsing",
                    total_files=total_files,
                    processed_files=processed_files_,
                    loaded_analyses=len(analyses),
                    current_file=current_file,
                )
            )

        if filename_key is not None and filename_key in seen_filename_keys:
            filename_duplicates_skipped += 1
            emit_progress()
            continue

        started = time.monotonic()
        try:
            record = _build_analysis_record_from_file(filepath)
        except (json.JSONDecodeError, OSError) as exc:
            failed_files += 1
            logger.warning(
                "analysis_file_unparseable",
                file=filepath.name,
                **_safe_exception_fields(exc, operation="loading analysis snapshot"),
                recommendation="delete_and_rerun_analysis",
            )
            emit_progress()
            continue
        elapsed = time.monotonic() - started
        if elapsed > 5.0:
            logger.warning(
                "analysis_file_slow_read",
                file=filepath.name,
                elapsed_s=round(elapsed, 1),
                hint="possible_spotlight_contention",
            )

        if record is None:
            missing_ticker_files += 1
            emit_progress()
            continue
        ticker = record.ticker

        if ticker in analyses:
            duplicate_files += 1
            emit_progress()
            continue

        analyses[ticker] = record
        if filename_key is not None:
            seen_filename_keys.add(filename_key)
        emit_progress()

    heartbeat_stop.set()
    logger.debug(
        "analyses_scan_complete",
        total_files=total_files,
        loaded=len(analyses),
        failed=failed_files,
        duplicates_skipped=duplicate_files,
        filename_duplicates_skipped=filename_duplicates_skipped,
        missing_ticker=missing_ticker_files,
    )
    logger.info("analyses_loaded", count=len(analyses))
    _write_latest_analyses_index(results_dir, analyses, total_files=total_files)
    if progress is not None:
        progress(
            AnalysisLoadProgress(
                phase="complete",
                total_files=total_files,
                processed_files=total_files,
                loaded_analyses=len(analyses),
                current_file=filepaths[-1].name if filepaths else None,
            )
        )
    return analyses
