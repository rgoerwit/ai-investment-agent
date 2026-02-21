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

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; }

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

# --- Ensure scratch directory exists ---
mkdir -p "$SCRATCH"

# --- Check Poetry is available ---
if ! command -v poetry &> /dev/null; then
    fail "Poetry is not installed"
    info "Install from: https://python-poetry.org/docs/#installation"
    exit 1
fi

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
        GEMS_CMD="poetry run python scripts/find_gems.py --output $TICKER_LIST --details ${SCRATCH}/gems_details_${DATE}.csv"
        if [[ -n "$INCLUDE_US" ]]; then
            GEMS_CMD="$GEMS_CMD $INCLUDE_US"
        fi

        info "Running: $GEMS_CMD"
        if eval "$GEMS_CMD"; then
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
        if ! $FORCE && [[ -f "$OUTFILE" ]] && grep -qE '^# .*\): ' "$OUTFILE"; then
            STAGE1_SKIPPED=$((STAGE1_SKIPPED + 1))
            info "SKIP $ticker (already done)"
            continue
        fi

        STAGE1_PROCESSED=$((STAGE1_PROCESSED + 1))
        info "[$STAGE1_PROCESSED] Quick: $ticker"

        if poetry run python -m src.main \
            --ticker "$ticker" \
            --quick --no-charts --quiet --brief --no-memory \
            --output "$OUTFILE" \
            2> "$LOGFILE"; then
            success "$ticker done"
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

        # Title line (line 10) format: "# TICKER (Company Name): VERDICT"
        if grep -qE '^# .*\): BUY$' "$OUTFILE"; then
            echo "$ticker" >> "$BUY_LIST"
            BUY_COUNT=$((BUY_COUNT + 1))
            success "BUY: $ticker"
        elif grep -qE '^# .*\): SELL' "$OUTFILE"; then
            SELL_COUNT=$((SELL_COUNT + 1))
            info "SELL: $ticker"
        elif grep -qE '^# .*\): HOLD' "$OUTFILE"; then
            HOLD_COUNT=$((HOLD_COUNT + 1))
            info "HOLD: $ticker"
        elif grep -qE '^# .*\): DO_NOT_INITIATE' "$OUTFILE"; then
            SELL_COUNT=$((SELL_COUNT + 1))
            info "DO_NOT_INITIATE: $ticker"
        else
            # Try to extract verdict from line 10
            VERDICT=$(sed -n '10p' "$OUTFILE" | sed 's/.*): //')
            warn "${VERDICT:-UNKNOWN}: $ticker"
            OTHER_COUNT=$((OTHER_COUNT + 1))
        fi

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
        info "Run Stages 0-1 first"
        exit 0
    fi

    BUY_TOTAL=$(grep -c '^[[:space:]]*[^[:space:]#]' "$BUY_LIST" || echo "0")
    if [[ "$BUY_TOTAL" -eq 0 ]]; then
        info "No BUY tickers to analyze. Pipeline complete."
        exit 0
    fi

    info "$BUY_TOTAL BUY tickers to analyze in full"

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
        if ! $FORCE && [[ -f "$OUTFILE" ]] && grep -qE '^# .*\): ' "$OUTFILE"; then
            STAGE2_SKIPPED=$((STAGE2_SKIPPED + 1))
            info "SKIP $ticker (full analysis exists)"
            continue
        fi

        STAGE2_PROCESSED=$((STAGE2_PROCESSED + 1))
        info "[$STAGE2_PROCESSED/$BUY_TOTAL] Full: $ticker"

        if poetry run python -m src.main \
            --ticker "$ticker" \
            --transparent --quiet \
            --output "$OUTFILE" \
            2> "$LOGFILE"; then
            success "$ticker done"
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
