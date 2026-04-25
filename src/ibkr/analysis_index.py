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

from src.error_safety import summarize_exception
from src.ibkr.models import AnalysisRecord, TradeBlockData
from src.ibkr.order_builder import parse_trade_block
from src.ibkr.reconciliation_rules import _exchange_from_ticker, _normalize_verdict

logger = structlog.get_logger(__name__)
_ANALYSIS_INDEX_VERSION = 2


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
    return AnalysisRecord.model_validate(data)


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
    """Extract health_adj, growth_adj, verdict, zone from a PM final_decision narrative."""
    result: dict = {}

    m = re.search(r"\bHEALTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Financial Health[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["health_adj"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"\bGROWTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Growth Transition[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["growth_adj"] = float(m.group(1))
        except ValueError:
            pass

    verdict_token = r"[A-Z_]+(?:[ \t][A-Z_]+)*"
    for pattern in (
        rf"\bVERDICT[:\s]+({verdict_token})",
        rf"PORTFOLIO MANAGER VERDICT:\s*({verdict_token})",
        r"\*\*Action\*\*:\s*\*\*(\w[\w_ ]*)\*\*",
    ):
        m = re.search(pattern, text)
        if m:
            result["verdict"] = m.group(1).strip().replace(" ", "_").upper()
            break

    m = re.search(r"\bZONE[:\s]+(HIGH|MODERATE|LOW)\b", text, re.IGNORECASE)
    if m:
        result["zone"] = m.group(1).upper()

    return result


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

    ticker = snapshot.get("ticker") or data.get("ticker", "")
    if not ticker:
        filename_ticker = _extract_filename_analysis_key(filepath.name)
        if filename_ticker:
            ticker = filename_ticker.replace("_", ".")
        if not ticker:
            return None

    trader_plan = data.get("investment_analysis", {}).get("trader_plan", "") or ""
    trade_block = parse_trade_block(trader_plan) or TradeBlockData()

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
        currency=snapshot.get("currency") or "USD",
        fx_rate_to_usd=snapshot.get("fx_rate_to_usd"),
        trade_block=trade_block,
        entry_price=snapshot.get("entry_price") or trade_block.entry_price,
        stop_price=snapshot.get("stop_price") or trade_block.stop_price,
        target_1_price=snapshot.get("target_1_price") or trade_block.target_1_price,
        target_2_price=snapshot.get("target_2_price") or trade_block.target_2_price,
        conviction=snapshot.get("conviction") or trade_block.conviction,
        sector=snapshot.get("sector") or "",
        exchange=snapshot.get("exchange") or _exchange_from_ticker(ticker),
        is_quick_mode=bool(snapshot.get("is_quick_mode", False)),
    )


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
        logger.info(
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
        analyses[ticker] = record
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
                    logger.info(
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

    heartbeat_state: dict[str, object] = {"file": "", "n": 0, "loaded": 0}
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
