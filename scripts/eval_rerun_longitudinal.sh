#!/bin/bash
# Longitudinal re-evaluation: re-runs the handful of tickers that have been
# analyzed most repeatedly since this repo's earliest retained runs
# (2025-12-01), so a fresh run can be diffed against the full analysis
# history for the same names to see whether codebase evolution has actually
# improved analysis quality (not just churned verdicts on noise).
#
# Full mode (not --quick) to match the mode most of that history was run in
# ("--quick is a screener, not investment-grade output" per CLAUDE.md).
#
# Wraps the whole run in `caffeinate -i` so the Mac doesn't sleep mid-batch.
# The JSON artifact for each run lands in the normal RESULTS_DIR (results/)
# via the standard save path -- this script does not need to know that path.

set -euo pipefail

# === macOS fork-safety / gRPC env (mirrors scripts/run_tickers.sh) ===
export GRPC_POLL_STRATEGY=poll
export GRPC_VERBOSITY=ERROR
if [ "$(uname)" = "Darwin" ] && \
   [ -z "${HTTP_PROXY:-}${HTTPS_PROXY:-}${http_proxy:-}${https_proxy:-}" ]; then
  export no_proxy='*' NO_PROXY='*'
fi
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"
# ======================================================================

# The 6 tickers with the most repeated runs whose history stretches back to
# the corpus's earliest retained analyses (2025-12-01) -- the widest
# possible longitudinal window. Override by passing a ticker file as $1
# (one ticker per line).
DEFAULT_TICKERS=(1681.HK PINFRA.MX AGS.BR 7740.T 8002.T 1088.HK)

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-15}"

TICKER_FILE="${1:-}"
if [[ -n "$TICKER_FILE" ]]; then
    if [[ ! -f "$TICKER_FILE" ]]; then
        echo "Ticker file not found: $TICKER_FILE" >&2
        exit 1
    fi
    mapfile -t TICKERS < <(grep -v '^[[:space:]]*#' "$TICKER_FILE" | grep -v '^[[:space:]]*$')
else
    TICKERS=("${DEFAULT_TICKERS[@]}")
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="scratch/eval_rerun_${RUN_STAMP}"
IMAGE_DIR="${OUT_DIR}/images"
LOG_FILE="${OUT_DIR}/run.log"
SUMMARY_FILE="${OUT_DIR}/SUMMARY.md"

mkdir -p "$IMAGE_DIR"

resolve_python_cmd() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        PYTHON_CMD=(python)
    elif command -v poetry &> /dev/null; then
        PYTHON_CMD=(poetry run python)
    else
        echo "Poetry not installed and no venv active" >&2
        exit 1
    fi
}
resolve_python_cmd

# Keep the Mac awake for the duration of this script (caffeinate exits with
# it via -w $$, so no manual cleanup needed).
if command -v caffeinate &> /dev/null; then
    caffeinate -i -w $$ &
    echo "caffeinate active (pid $!), pinned to this script's pid $$"
fi

{
    echo "# Longitudinal re-evaluation run"
    echo "Started: $(date)"
    echo "Tickers: ${TICKERS[*]}"
    echo ""
    echo "| Ticker | Status | Report |"
    echo "|---|---|---|"
} > "$SUMMARY_FILE"

echo "=== Longitudinal re-evaluation: ${#TICKERS[@]} tickers ===" | tee -a "$LOG_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$LOG_FILE"

n=0
failed=0
for ticker in "${TICKERS[@]}"; do
    n=$((n + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "--- [$n/${#TICKERS[@]}] $ticker ($(date)) ---" | tee -a "$LOG_FILE"

    REPORT_PATH="${OUT_DIR}/${ticker}.md"
    ANALYSIS_STARTED_AT="$(date +%s)"

    if "${PYTHON_CMD[@]}" -m src.main --ticker "$ticker" \
            --imagedir "$IMAGE_DIR" --output "$REPORT_PATH" \
            --quiet --brief >> "$LOG_FILE" 2>&1; then
        if VALIDITY_RESULT="$("${PYTHON_CMD[@]}" scripts/scan_batch_health.py \
                --modified-since "$ANALYSIS_STARTED_AT" \
                --require-publishable-ticker "$ticker" 2>> "$LOG_FILE")"; then
            echo "OK: $ticker" | tee -a "$LOG_FILE"
            echo "$VALIDITY_RESULT" >> "$LOG_FILE"
            echo "| $ticker | OK | [$ticker.md](./${ticker}.md) |" >> "$SUMMARY_FILE"
        else
            echo "INCOMPLETE: $ticker (diagnostic output retained)" | tee -a "$LOG_FILE"
            echo "$VALIDITY_RESULT" >> "$LOG_FILE"
            echo "| $ticker | INCOMPLETE | [$ticker.md](./${ticker}.md); see run.log |" >> "$SUMMARY_FILE"
            failed=$((failed + 1))
        fi
    else
        echo "FAILED_RUNTIME: $ticker" | tee -a "$LOG_FILE"
        echo "| $ticker | FAILED_RUNTIME | see run.log |" >> "$SUMMARY_FILE"
        failed=$((failed + 1))
    fi

    if [[ $n -lt ${#TICKERS[@]} ]]; then
        echo "Cooling down ${COOLDOWN_SECONDS}s..." | tee -a "$LOG_FILE"
        sleep "$COOLDOWN_SECONDS"
    fi
done

{
    echo ""
    echo "Finished: $(date)"
    echo "Processed: $n, Incomplete or failed: $failed"
    echo ""
    echo "Corresponding *_analysis.json artifacts were written to \$RESULTS_DIR"
    echo "(results/ by default) with a fresh timestamp -- diff those against"
    echo "the prior history for the same tickers under results/ and"
    echo "~/Developer/results_archive/ for the longitudinal comparison."
} | tee -a "$SUMMARY_FILE" "$LOG_FILE"

echo ""
echo "=== Done. Summary: $SUMMARY_FILE ==="
exit $((failed > 0 ? 1 : 0))
