#!/bin/bash
# End-to-end screening pipeline:
#   Stage 0: Scrape exchanges + filter by fundamentals → ticker list
#   Stage 1: Quick analysis on all candidates → identify BUYs
#   Stage 2: Full analysis on BUY tickers only → production reports
#
# Usage:
#   ./scripts/run_pipeline.sh [OPTIONS]
#
# Options:
#   --force             Re-run even if output files exist (disables resumability)
#   --skip-scrape FILE  Use existing ticker list instead of running find_gems.py
#   --include-us        Pass --include-us to find_gems.py
#   --cooldown N        Override COOLDOWN_SECONDS (default: 60)
#   --stage N           Start from stage N (0, 1, or 2). Requires prior stages' outputs.
#   --run-date DATE     Force output/resume date (YYYY-MM-DD), useful for cross-day resume
#   --buys-file FILE    Explicit BUY list path (bypasses date-based default; use with --stage 2
#                       when resuming a run that started on a previous calendar day)
#   -y, --yes           Skip confirmation prompts (run non-interactively)
#   -h, --help          Show this help message
#
# Environment:
#   COOLDOWN_SECONDS    Seconds between tickers (default: 60, use 10 for paid tier)

set -euo pipefail

# === CRITICAL FIX FOR MACOS ===
export GRPC_POLL_STRATEGY=poll
export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=""

# --- Configuration ---
DATE=$(date +%Y-%m-%d)
SCRATCH="scratch"
TICKER_LIST="${SCRATCH}/gems_${DATE}.txt"
BUY_LIST="${SCRATCH}/buys_${DATE}.txt"
COOLDOWN="${COOLDOWN_SECONDS:-60}"
FORCE=false
SKIP_SCRAPE=""
INCLUDE_US=""
START_STAGE=0
AUTO_YES=false
STRICT_FLAG=""
RUN_DATE_OVERRIDE=""
BUY_LIST_EXPLICIT=false
PYTHON_CMD=()
PYTHON_CMD_DISPLAY=""

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; }

# --- Confirmation prompt (skipped with --yes) ---
confirm() {
    local prompt="$1"
    if $AUTO_YES; then
        return 0
    fi
    echo ""
    echo -en "${BOLD}${prompt}${NC} [y/N] "
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# --- Time formatting ---
format_duration() {
    local total_secs=$1
    local hours=$((total_secs / 3600))
    local mins=$(( (total_secs % 3600) / 60 ))
    if [[ $hours -gt 0 ]]; then
        echo "${hours}h ${mins}m"
    else
        echo "${mins}m"
    fi
}

report_verdict_line() {
    local file="$1"
    awk '
        /^# / {
            line=$0
            sub(/\r$/, "", line)
            if (match(line, /: .+$/)) {
                found = 1
                print line
                exit 0
            }
        }
        END {
            if (!found) {
                exit 1
            }
        }
    ' "$file" 2>/dev/null || true
}

extract_report_verdict() {
    local file="$1"
    local line
    line=$(report_verdict_line "$file")
    if [[ -n "$line" ]]; then
        printf '%s\n' "${line##*: }"
    fi
}

report_has_verdict_header() {
    local file="$1"
    local verdict
    verdict=$(extract_report_verdict "$file")
    [[ -n "$verdict" ]]
}

resolve_python_cmd() {
    if [[ -n "${INVESTMENT_AGENT_CONTAINER:-}" ]] || [[ -f "/.dockerenv" ]] || [[ -f "/run/.containerenv" ]]; then
        PYTHON_CMD=(python)
        PYTHON_CMD_DISPLAY="python"
        return
    fi

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        PYTHON_CMD=(python)
        PYTHON_CMD_DISPLAY="python"
        return
    fi

    if command -v poetry &> /dev/null; then
        PYTHON_CMD=(poetry run python)
        PYTHON_CMD_DISPLAY="poetry run python"
        return
    fi

    fail "Poetry is not installed and no active virtual environment was detected"
    info "Either activate your venv, or install Poetry from: https://python-poetry.org/docs/#installation"
    exit 1
}

extract_date_from_path() {
    local path="$1"
    basename "$path" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1 || true
}

apply_run_date() {
    local source_label="$1"
    local new_date="$2"
    if [[ -z "$new_date" || "$new_date" == "$DATE" ]]; then
        return
    fi

    info "Cross-day resume: using date ${new_date} from ${source_label} (was ${DATE})"
    DATE="$new_date"
    if [[ -z "$SKIP_SCRAPE" ]]; then
        TICKER_LIST="${SCRATCH}/gems_${DATE}.txt"
    fi
    if ! $BUY_LIST_EXPLICIT; then
        BUY_LIST="${SCRATCH}/buys_${DATE}.txt"
    fi
}

ticker_to_dash() {
    local ticker="$1"
    printf '%s\n' "$ticker" | tr '._' '-'
}

quick_outfile_for() {
    local ticker="$1"
    local run_date="$2"
    local dash
    dash=$(ticker_to_dash "$ticker")
    printf '%s/README-%s-%s_quick.md\n' "$SCRATCH" "$dash" "$run_date"
}

count_completed_quick_reports_for_date() {
    local ticker_list="$1"
    local run_date="$2"
    local count=0
    local ticker outfile
    while IFS= read -r ticker || [[ -n "$ticker" ]]; do
        [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
        ticker=$(echo "$ticker" | xargs)
        [[ -z "$ticker" ]] && continue
        outfile=$(quick_outfile_for "$ticker" "$run_date")
        if [[ -f "$outfile" ]] && report_has_verdict_header "$outfile"; then
            count=$((count + 1))
        fi
    done < "$ticker_list"
    printf '%s\n' "$count"
}

detect_stage1_resume_date() {
    local ticker_list="$1"
    local inferred_date="$2"
    local today_date="$3"
    local inferred_count today_count

    inferred_count=$(count_completed_quick_reports_for_date "$ticker_list" "$inferred_date")
    today_count=$(count_completed_quick_reports_for_date "$ticker_list" "$today_date")

    if [[ "$today_count" -gt "$inferred_count" ]]; then
        printf '%s\n' "$today_date"
    else
        printf '%s\n' "$inferred_date"
    fi
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift ;;
        --skip-scrape)
            SKIP_SCRAPE="$2"
            shift 2 ;;
        --include-us)
            INCLUDE_US="--include-us"
            shift ;;
        --cooldown)
            COOLDOWN="$2"
            shift 2 ;;
        --stage)
            START_STAGE="$2"
            shift 2 ;;
        --run-date)
            RUN_DATE_OVERRIDE="$2"
            shift 2 ;;
        --buys-file)
            BUY_LIST="$2"
            BUY_LIST_EXPLICIT=true
            shift 2 ;;
        -y|--yes)
            AUTO_YES=true
            shift ;;
        --strict)
            STRICT_FLAG="--strict"
            shift ;;
        -h|--help)
            cat << 'HELPEOF'
Usage: ./scripts/run_pipeline.sh [OPTIONS]

End-to-end screening pipeline: scrape → filter → quick screen → extract BUYs → full analysis.

STAGES:
  0  Scrape exchanges + filter by fundamentals (find_gems.py)
  1  Quick analysis on all candidates (--quick --no-charts --brief --no-memory)
  2  Full analysis on BUY tickers only (production quality with charts)

OPTIONS:
  --force             Re-run even if output files exist (disables resumability)
  --skip-scrape FILE  Use existing ticker list instead of running find_gems.py
  --include-us        Include US exchanges in scrape phase
  --cooldown N        Seconds between ticker analyses (default: 60)
  --stage N           Start from stage N (0, 1, or 2)
  --run-date DATE     Force the pipeline/output date (YYYY-MM-DD) for cross-day resume
  --buys-file FILE    Explicit BUY list path — bypasses the date-based default
                      (scratch/buys_YYYY-MM-DD.txt). Use with --stage 2 when
                      resuming a run that crossed midnight:
                        ./scripts/run_pipeline.sh --stage 2 --buys-file scratch/buys_2026-02-24.txt
  --strict            Apply strict mode to Stage 2 full analysis (tighter D/E, reject
                      REITs/PFIC/VIE, escalate value traps, higher BUY conviction bar).
                      Stage 1 screening always runs strict regardless of this flag.
  -y, --yes           Skip confirmation prompts (run non-interactively)
  -h, --help          Show this help message

ENVIRONMENT:
  COOLDOWN_SECONDS    Default cooldown (overridden by --cooldown flag)

EXAMPLES:
  # Full pipeline from scratch
  ./scripts/run_pipeline.sh

  # Skip scraping, use existing ticker list
  ./scripts/run_pipeline.sh --skip-scrape scratch/gems_2026-02-19.txt

  # Re-run failed tickers (force mode)
  ./scripts/run_pipeline.sh --skip-scrape scratch/gems_2026-02-19.txt --force

  # Paid API tier (shorter cooldown)
  ./scripts/run_pipeline.sh --cooldown 10

  # Start from Stage 2 (BUY list must exist from prior run)
  ./scripts/run_pipeline.sh --stage 2

  # Resume a prior Stage 1 run from its original date
  ./scripts/run_pipeline.sh --stage 1 --skip-scrape scratch/gems_2026-02-24.txt

  # Resume using an explicit run date when filenames were copied/renamed
  ./scripts/run_pipeline.sh --stage 1 --skip-scrape scratch/gems_2026-02-25.txt --run-date 2026-02-24

  # Non-interactive (e.g. cron, CI, overnight run)
  caffeinate -i ./scripts/run_pipeline.sh --yes

  # Prevent macOS sleep during overnight run
  caffeinate -i ./scripts/run_pipeline.sh
HELPEOF
            exit 0 ;;
        *)
            fail "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1 ;;
    esac
done

# Resolve TICKER_LIST from --skip-scrape immediately (not deferred to stage 0 block)
# This ensures --stage 1 --skip-scrape FILE works without going through stage 0.
if [[ -n "$SKIP_SCRAPE" ]]; then
    TICKER_LIST="$SKIP_SCRAPE"
fi

# Date precedence:
# 1. explicit --run-date
# 2. explicit --buys-file date
# 3. --skip-scrape ticker-list date
if [[ -n "$RUN_DATE_OVERRIDE" ]]; then
    apply_run_date "run-date override" "$RUN_DATE_OVERRIDE"
elif $BUY_LIST_EXPLICIT; then
    apply_run_date "buys file" "$(extract_date_from_path "$BUY_LIST")"
elif [[ -n "$SKIP_SCRAPE" ]]; then
    apply_run_date "ticker list" "$(extract_date_from_path "$TICKER_LIST")"
fi

if [[ -n "$SKIP_SCRAPE" && -z "$RUN_DATE_OVERRIDE" && ! $BUY_LIST_EXPLICIT && $START_STAGE -eq 1 ]]; then
    DETECTED_STAGE1_DATE=$(detect_stage1_resume_date "$TICKER_LIST" "$DATE" "$(date +%Y-%m-%d)")
    if [[ -n "$DETECTED_STAGE1_DATE" && "$DETECTED_STAGE1_DATE" != "$DATE" ]]; then
        apply_run_date "existing quick outputs" "$DETECTED_STAGE1_DATE"
    fi
fi

# --- Ensure scratch directory exists ---
mkdir -p "$SCRATCH"

# --- Resolve Python runtime ---
resolve_python_cmd
info "Python runtime:      $PYTHON_CMD_DISPLAY"

# ============================================================
# STAGE 0: Scrape + Filter
# ============================================================
if [[ $START_STAGE -le 0 ]]; then
    echo ""
    echo "========================================"
    info "STAGE 0: Scrape + Filter"
    echo "========================================"

    if [[ -n "$SKIP_SCRAPE" ]]; then
        if [[ ! -f "$SKIP_SCRAPE" ]]; then
            fail "Ticker list not found: $SKIP_SCRAPE"
            exit 1
        fi
        TICKER_LIST="$SKIP_SCRAPE"
        info "Using existing ticker list: $TICKER_LIST"
    else
        GEMS_CMD=("${PYTHON_CMD[@]}" scripts/find_gems.py --output "$TICKER_LIST" --details "${SCRATCH}/gems_details_${DATE}.csv")
        if [[ -n "$INCLUDE_US" ]]; then
            GEMS_CMD+=("$INCLUDE_US")
        fi

        info "Running: ${GEMS_CMD[*]}"
        if "${GEMS_CMD[@]}"; then
            success "Scrape + filter complete"
        else
            fail "find_gems.py failed"
            exit 1
        fi
    fi

    TICKER_COUNT=$(grep -c '^[[:space:]]*[^[:space:]#]' "$TICKER_LIST" || echo "0")
    info "Candidates: $TICKER_COUNT tickers in $TICKER_LIST"

    if [[ "$TICKER_COUNT" -eq 0 ]]; then
        fail "No tickers to analyze"
        exit 1
    fi

    # --- Confirmation before Stage 1 ---
    # Count how many would actually be processed (not skipped by resumability)
    STAGE1_TODO=0
    while IFS= read -r ticker || [[ -n "$ticker" ]]; do
        [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
        ticker=$(echo "$ticker" | xargs)
        [[ -z "$ticker" ]] && continue
        DASH=$(echo "$ticker" | tr '._' '-')
        OUTFILE="${SCRATCH}/README-${DASH}-${DATE}_quick.md"
        if $FORCE || ! [[ -f "$OUTFILE" ]] || ! report_has_verdict_header "$OUTFILE"; then
            STAGE1_TODO=$((STAGE1_TODO + 1))
        fi
    done < "$TICKER_LIST"

    STAGE1_SKIP=$((TICKER_COUNT - STAGE1_TODO))
    STAGE1_SECS=$((STAGE1_TODO * (COOLDOWN + 120)))

    echo ""
    echo -e "${CYAN}━━━ Stage 1 Preview ━━━━━━━━━━━━━━━━━━━━${NC}"
    info "Tickers to analyze:  $STAGE1_TODO (of $TICKER_COUNT candidates)"
    if [[ $STAGE1_SKIP -gt 0 ]]; then
        info "Already completed:   $STAGE1_SKIP (will be skipped)"
    fi
    info "Est. time:           ~$(format_duration $STAGE1_SECS) (${COOLDOWN}s cooldown)"
    info "Mode:                --quick --strict --brief --no-memory"
    info "Output dir:          $SCRATCH/"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [[ $STAGE1_TODO -eq 0 ]]; then
        info "Nothing to do — all tickers already processed."
    elif ! confirm "Proceed with Stage 1 quick analysis?"; then
        warn "Aborted by user."
        exit 0
    fi
fi

# ============================================================
# STAGE 1: Quick Analysis
# ============================================================
if [[ $START_STAGE -le 1 ]]; then
    echo ""
    echo "========================================"
    info "STAGE 1: Quick Analysis (screening)"
    echo "========================================"

    if [[ ! -f "$TICKER_LIST" ]]; then
        fail "Ticker list not found: $TICKER_LIST"
        info "Run Stage 0 first, or use --skip-scrape FILE"
        exit 1
    fi

    # If entering at --stage 1, we haven't shown a preview yet
    if [[ $START_STAGE -eq 1 ]]; then
        TICKER_COUNT=$(grep -c '^[[:space:]]*[^[:space:]#]' "$TICKER_LIST" || echo "0")

        STAGE1_TODO=0
        while IFS= read -r ticker || [[ -n "$ticker" ]]; do
            [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
            ticker=$(echo "$ticker" | xargs)
            [[ -z "$ticker" ]] && continue
            DASH=$(echo "$ticker" | tr '._' '-')
            OUTFILE="${SCRATCH}/README-${DASH}-${DATE}_quick.md"
            if $FORCE || ! [[ -f "$OUTFILE" ]] || ! report_has_verdict_header "$OUTFILE"; then
                STAGE1_TODO=$((STAGE1_TODO + 1))
            fi
        done < "$TICKER_LIST"

        STAGE1_SKIP=$((TICKER_COUNT - STAGE1_TODO))
        STAGE1_SECS=$((STAGE1_TODO * (COOLDOWN + 120)))

        echo ""
        echo -e "${CYAN}━━━ Stage 1 Preview ━━━━━━━━━━━━━━━━━━━━${NC}"
        info "Tickers to analyze:  $STAGE1_TODO (of $TICKER_COUNT candidates)"
        if [[ $STAGE1_SKIP -gt 0 ]]; then
            info "Already completed:   $STAGE1_SKIP (will be skipped)"
        fi
        info "Est. time:           ~$(format_duration $STAGE1_SECS) (${COOLDOWN}s cooldown)"
        info "Mode:                --quick --strict --brief --no-memory"
        info "Output dir:          $SCRATCH/"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        if [[ $STAGE1_TODO -eq 0 ]]; then
            info "Nothing to do — all tickers already processed."
        elif ! confirm "Proceed with Stage 1 quick analysis?"; then
            warn "Aborted by user."
            exit 0
        fi
    fi

    STAGE1_PROCESSED=0
    STAGE1_SKIPPED=0
    STAGE1_FAILED=0

    while IFS= read -r ticker || [[ -n "$ticker" ]]; do
        # Skip empty lines and comments
        [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
        ticker=$(echo "$ticker" | xargs)
        [[ -z "$ticker" ]] && continue

        # Build output filename: dots and underscores become dashes
        DASH=$(echo "$ticker" | tr '._' '-')
        OUTFILE="${SCRATCH}/README-${DASH}-${DATE}_quick.md"
        LOGFILE="${SCRATCH}/${DASH}-LOG-${DATE}_quick.txt"

        # Resumability: skip if output already has a verdict line
        if ! $FORCE && [[ -f "$OUTFILE" ]] && report_has_verdict_header "$OUTFILE"; then
            STAGE1_SKIPPED=$((STAGE1_SKIPPED + 1))
            info "SKIP $ticker (already done)"
            continue
        fi

        STAGE1_PROCESSED=$((STAGE1_PROCESSED + 1))
        info "[$STAGE1_PROCESSED/$STAGE1_TODO, $TICKER_COUNT total] Quick: $ticker"

        if "${PYTHON_CMD[@]}" -m src.main \
            --ticker "$ticker" \
            --quick --strict --no-charts --quiet --brief --no-memory \
            --output "$OUTFILE" \
            2> "$LOGFILE"; then
            VERDICT=$(extract_report_verdict "$OUTFILE")
            [[ -n "$VERDICT" ]] && success "$ticker done [Verdict=${VERDICT}]" || success "$ticker done"
        else
            fail "FAILED: $ticker (see $LOGFILE)"
            STAGE1_FAILED=$((STAGE1_FAILED + 1))
        fi

        sleep "$COOLDOWN"

    done < "$TICKER_LIST"

    info "Stage 1 complete: $STAGE1_PROCESSED analyzed, $STAGE1_SKIPPED skipped, $STAGE1_FAILED failed"
fi

# ============================================================
# VERDICT EXTRACTION: Identify BUY tickers
# ============================================================
if [[ $START_STAGE -le 1 ]]; then
    echo ""
    echo "========================================"
    info "Extracting BUY verdicts"
    echo "========================================"

    > "$BUY_LIST"  # truncate

    BUY_COUNT=0
    SELL_COUNT=0
    HOLD_COUNT=0
    OTHER_COUNT=0

    while IFS= read -r ticker || [[ -n "$ticker" ]]; do
        [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
        ticker=$(echo "$ticker" | xargs)
        [[ -z "$ticker" ]] && continue

        DASH=$(echo "$ticker" | tr '._' '-')
        OUTFILE="${SCRATCH}/README-${DASH}-${DATE}_quick.md"

        if [[ ! -f "$OUTFILE" ]]; then
            warn "MISSING: $ticker (no output file)"
            OTHER_COUNT=$((OTHER_COUNT + 1))
            continue
        fi

        VERDICT=$(extract_report_verdict "$OUTFILE")
        case "$VERDICT" in
            BUY)
                echo "$ticker" >> "$BUY_LIST"
                BUY_COUNT=$((BUY_COUNT + 1))
                success "BUY: $ticker"
                ;;
            SELL)
                SELL_COUNT=$((SELL_COUNT + 1))
                info "SELL: $ticker"
                ;;
            HOLD)
                HOLD_COUNT=$((HOLD_COUNT + 1))
                info "HOLD: $ticker"
                ;;
            DO_NOT_INITIATE|DO\ NOT\ INITIATE)
                SELL_COUNT=$((SELL_COUNT + 1))
                info "DO_NOT_INITIATE: $ticker"
                ;;
            "")
                HEADER=$(report_verdict_line "$OUTFILE")
                warn "UNKNOWN: $ticker [header: ${HEADER:-<empty>}]"
                OTHER_COUNT=$((OTHER_COUNT + 1))
                ;;
            *)
                warn "UNKNOWN_VERDICT ($VERDICT): $ticker"
                OTHER_COUNT=$((OTHER_COUNT + 1))
                ;;
        esac

    done < "$TICKER_LIST"

    echo ""
    info "Verdict summary: $BUY_COUNT BUY, $HOLD_COUNT HOLD, $SELL_COUNT SELL/DNI, $OTHER_COUNT other"

    if [[ $BUY_COUNT -eq 0 ]]; then
        warn "No BUY verdicts found. Stage 2 will be skipped."
    fi
fi

# ============================================================
# STAGE 2: Full Analysis (BUYs only)
# ============================================================
if [[ $START_STAGE -le 2 ]]; then
    echo ""
    echo "========================================"
    info "STAGE 2: Full Analysis (BUYs only)"
    echo "========================================"

    if [[ ! -f "$BUY_LIST" ]]; then
        warn "No BUY list found at $BUY_LIST"
        info "Run Stages 0-1 first, or specify an existing file with --buys-file"
        info "Example (cross-day resume): --stage 2 --buys-file scratch/buys_2026-02-24.txt"
        exit 0
    fi

    BUY_TOTAL=$(grep -c '^[[:space:]]*[^[:space:]#]' "$BUY_LIST" || echo "0")
    if [[ "$BUY_TOTAL" -eq 0 ]]; then
        info "No BUY tickers to analyze. Pipeline complete."
        exit 0
    fi

    # Count how many would actually be processed
    STAGE2_TODO=0
    while IFS= read -r ticker || [[ -n "$ticker" ]]; do
        [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
        ticker=$(echo "$ticker" | xargs)
        [[ -z "$ticker" ]] && continue
        DASH=$(echo "$ticker" | tr '._' '-')
        OUTFILE="${SCRATCH}/README-${DASH}-${DATE}.md"
        if $FORCE || ! [[ -f "$OUTFILE" ]] || ! report_has_verdict_header "$OUTFILE"; then
            STAGE2_TODO=$((STAGE2_TODO + 1))
        fi
    done < "$BUY_LIST"

    STAGE2_SKIP=$((BUY_TOTAL - STAGE2_TODO))
    STAGE2_SECS=$((STAGE2_TODO * (COOLDOWN + 300)))

    echo ""
    echo -e "${CYAN}━━━ Stage 2 Preview ━━━━━━━━━━━━━━━━━━━━${NC}"
    info "BUY tickers to analyze: $STAGE2_TODO (of $BUY_TOTAL)"
    if [[ $STAGE2_SKIP -gt 0 ]]; then
        info "Already completed:     $STAGE2_SKIP (will be skipped)"
    fi
    info "Est. time:             ~$(format_duration $STAGE2_SECS) (${COOLDOWN}s cooldown)"
    info "Mode:                  Full analysis with charts"
    info "Output dir:            $SCRATCH/"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [[ $STAGE2_TODO -eq 0 ]]; then
        info "Nothing to do — all BUY tickers already have full analysis."
    elif ! confirm "Proceed with Stage 2 full analysis?"; then
        warn "Aborted by user."
        exit 0
    fi

    STAGE2_PROCESSED=0
    STAGE2_SKIPPED=0
    STAGE2_FAILED=0

    while IFS= read -r ticker || [[ -n "$ticker" ]]; do
        [[ -z "$ticker" || "$ticker" =~ ^[[:space:]]*# ]] && continue
        ticker=$(echo "$ticker" | xargs)
        [[ -z "$ticker" ]] && continue

        DASH=$(echo "$ticker" | tr '._' '-')
        OUTFILE="${SCRATCH}/README-${DASH}-${DATE}.md"
        LOGFILE="${SCRATCH}/${DASH}-LOG-${DATE}.txt"

        # Resumability: skip if full analysis output already has a verdict
        if ! $FORCE && [[ -f "$OUTFILE" ]] && report_has_verdict_header "$OUTFILE"; then
            STAGE2_SKIPPED=$((STAGE2_SKIPPED + 1))
            info "SKIP $ticker (full analysis exists)"
            continue
        fi

        STAGE2_PROCESSED=$((STAGE2_PROCESSED + 1))
        info "[$STAGE2_PROCESSED/$STAGE2_TODO, $BUY_TOTAL total] Full: $ticker"

        if "${PYTHON_CMD[@]}" -m src.main \
            --ticker "$ticker" \
            --transparent --quiet \
            $STRICT_FLAG \
            --output "$OUTFILE" \
            2> "$LOGFILE"; then
            VERDICT=$(extract_report_verdict "$OUTFILE")
            [[ -n "$VERDICT" ]] && success "$ticker done [Verdict=${VERDICT}]" || success "$ticker done"
        else
            fail "FAILED: $ticker (see $LOGFILE)"
            STAGE2_FAILED=$((STAGE2_FAILED + 1))
        fi

        sleep "$COOLDOWN"

    done < "$BUY_LIST"

    info "Stage 2 complete: $STAGE2_PROCESSED analyzed, $STAGE2_SKIPPED skipped, $STAGE2_FAILED failed"
fi

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========================================"
success "Pipeline complete!"
echo "========================================"
echo ""

if [[ -f "$TICKER_LIST" ]]; then
    TOTAL=$(grep -c '^[[:space:]]*[^[:space:]#]' "$TICKER_LIST" || echo "0")
    info "Candidates screened: $TOTAL ($TICKER_LIST)"
fi

if [[ -f "$BUY_LIST" ]]; then
    BUYS=$(grep -c '^[[:space:]]*[^[:space:]#]' "$BUY_LIST" || echo "0")
    info "BUY verdicts: $BUYS ($BUY_LIST)"

    if [[ "$BUYS" -gt 0 ]]; then
        echo ""
        info "BUY tickers:"
        while IFS= read -r t; do
            [[ -z "$t" || "$t" =~ ^[[:space:]]*# ]] && continue
            echo "  $t"
        done < "$BUY_LIST"
    fi
fi
