#!/bin/bash
# Batch ticker analysis script (master script)
# Reads tickers from scratch/sample_tickers.txt and analyzes each one sequentially
#
# This is the master script that supports all modes via flags:
#   --loud     Verbose logging to stderr (no --quiet, no --brief, unbuffered output)
#   --quick    Use quick analysis mode (adds --quick to Python command)
#   --verbose  Verbose reports (no --brief, but still --quiet for clean logs)
#
# The variant scripts (run_tickers_loud.sh, etc.) delegate to this master script.

set -euo pipefail

# === CRITICAL FIX FOR MACOS ===
# Prevents gRPC "fork_posix.cc:71" errors and hanging on Apple Silicon
export GRPC_POLL_STRATEGY=poll
export GRPC_VERBOSITY=ERROR
# ==============================

# Configuration defaults
DEFAULT_INPUT_FILE="scratch/sample_tickers.txt"
DEFAULT_OUTPUT_FILE="scratch/ticker_analysis_results.md"

# Cooldown between tickers (seconds). Configurable via environment variable.
# - Free tier (15 RPM): 60 seconds recommended (default)
# - Paid tier (360 RPM): 5-10 seconds sufficient
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-60}"

# Parse flags
LOUD_MODE=false
QUICK_MODE=false
VERBOSE_MODE=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --loud)
            LOUD_MODE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --help|-h)
            cat << 'EOF'
Usage: ./scripts/run_tickers.sh [OPTIONS] [INPUT_FILE] [OUTPUT_FILE]

Batch analyze multiple tickers from a file. RUN FROM REPO ROOT DIRECTORY.

OPTIONS:
    --loud      Verbose logging to stderr for real-time monitoring
                (removes --quiet and --brief, adds unbuffered output)
    --quick     Use quick analysis mode (faster, 1 debate round)
    --verbose   Verbose reports (removes --brief, keeps --quiet)
    -h, --help  Show this help message

ARGUMENTS:
    INPUT_FILE     File containing ticker symbols (default: scratch/sample_tickers.txt)
    OUTPUT_FILE    Output markdown file (default: scratch/ticker_analysis_results.md)

INPUT FILE FORMAT:
    One ticker per line, e.g.:
    AAPL
    MSFT
    NVDA

EXAMPLES:
    # Default mode (quiet, brief)
    ./scripts/run_tickers.sh

    # Quick mode (faster analysis)
    ./scripts/run_tickers.sh --quick

    # Loud mode with real-time log monitoring
    ./scripts/run_tickers.sh --loud 2>scratch/ticker_analysis_info.txt &
    tail -f scratch/ticker_analysis_info.txt

    # Verbose mode (detailed reports, quiet logs)
    ./scripts/run_tickers.sh --verbose

    # Custom files with quick mode
    ./scripts/run_tickers.sh --quick my_tickers.txt my_results.md

    # Paid tier with shorter cooldown (360 RPM)
    COOLDOWN_SECONDS=10 ./scripts/run_tickers.sh --quick

ENVIRONMENT VARIABLES:
    COOLDOWN_SECONDS    Seconds to wait between tickers (default: 60)
                        Free tier (15 RPM): use 60 (default)
                        Paid tier (360 RPM): use 5-10

NOTES:
    - Images are auto-saved to {OUTPUT_DIR}/images/ based on OUTPUT_FILE path
    - In loud mode, logs go to stderr (can redirect with 2>)
    - Options can be combined: --loud --quick
    - Failed analyses save debug files to {OUTPUT_DIR}/debug_failures/
    - "Soft failures" (PM failed but Python exit 0) are also captured

EOF
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
INPUT_FILE="${POSITIONAL_ARGS[0]:-$DEFAULT_INPUT_FILE}"
OUTPUT_FILE="${POSITIONAL_ARGS[1]:-$DEFAULT_OUTPUT_FILE}"

# Auto-detect imagedir from OUTPUT_FILE path
# Extract the directory part of OUTPUT_FILE and append /images
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [[ "$OUTPUT_DIR" == "." ]]; then
    IMAGE_DIR="images"
else
    IMAGE_DIR="${OUTPUT_DIR}/images"
fi

# Cleanup function for temp files on interrupt/exit
# IMPORTANT: Only cleans up on success (exit 0) or interrupt
# On failure, temp files are preserved for debugging
cleanup_temp_files() {
    local exit_code=$?
    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        if [[ $exit_code -eq 0 ]]; then
            # Success: remove temp files
            rm -f "${OUTPUT_DIR}"/.temp_analysis_*.md 2>/dev/null || true
            rm -f "${OUTPUT_DIR}"/.temp_analysis_*.log 2>/dev/null || true
        else
            # Failure: preserve temp files for debugging
            # Move to debug directory instead of deleting
            local debug_dir="${OUTPUT_DIR}/debug_$(date +%Y%m%d_%H%M%S)"
            if ls "${OUTPUT_DIR}"/.temp_analysis_*.{md,log} >/dev/null 2>&1; then
                mkdir -p "$debug_dir"
                mv "${OUTPUT_DIR}"/.temp_analysis_*.md "$debug_dir/" 2>/dev/null || true
                mv "${OUTPUT_DIR}"/.temp_analysis_*.log "$debug_dir/" 2>/dev/null || true
                echo -e "${YELLOW}[DEBUG]${NC} Temp files preserved in: $debug_dir"
            fi
        fi
    fi
    exit $exit_code
}

# Set trap to cleanup on EXIT, INT (Ctrl+C), TERM (kill)
trap cleanup_temp_files EXIT INT TERM

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Ensure image directory exists
mkdir -p "$IMAGE_DIR"

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    print_error "Input file not found: $INPUT_FILE"
    echo ""
    print_info "Creating example file at $INPUT_FILE..."
    mkdir -p "$(dirname "$INPUT_FILE")"
    cat > "$INPUT_FILE" << 'EOF'
AAPL
MSFT
NVDA
EOF
    print_success "Created example file with 3 tickers"
    print_info "Edit $INPUT_FILE and run again"
    exit 1
fi

# Count tickers
ticker_count=$(grep -c '^[[:space:]]*[^[:space:]#]' "$INPUT_FILE" || echo "0")
if [[ "$ticker_count" -eq 0 ]]; then
    print_error "No tickers found in $INPUT_FILE"
    print_info "Add ticker symbols (one per line) and try again"
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed"
    print_info "Install from: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Initialize output file with header
cat > "$OUTPUT_FILE" << EOF
# Ticker Analysis Results
Generated: $(date)
Input file: $INPUT_FILE
Total tickers: $ticker_count

---

EOF

# Build mode description for logging
MODE_DESC="default"
if $LOUD_MODE && $QUICK_MODE; then
    MODE_DESC="loud+quick"
elif $LOUD_MODE; then
    MODE_DESC="loud"
elif $QUICK_MODE && $VERBOSE_MODE; then
    MODE_DESC="quick+verbose"
elif $QUICK_MODE; then
    MODE_DESC="quick"
elif $VERBOSE_MODE; then
    MODE_DESC="verbose"
fi

print_info "Starting batch analysis..."
print_info "Mode: $MODE_DESC"
print_info "Input: $INPUT_FILE ($ticker_count tickers)"
print_info "Output: $OUTPUT_FILE"
print_info "Images: $IMAGE_DIR"
echo ""

# Process each ticker
processed=0
failed=0

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace
    ticker=$(echo "$line" | xargs)

    if [[ -z "$ticker" ]]; then
        continue
    fi

    processed=$((processed + 1))

    echo "========================================"
    print_info "[$processed/$ticker_count] Analyzing: $ticker"
    echo "========================================"

    # Temp file for this specific ticker's report (in the same dir as output to ensure relative links work)
    TEMP_REPORT="${OUTPUT_DIR}/.temp_analysis_${ticker}.md"
    TEMP_LOG="${OUTPUT_DIR}/.temp_analysis_${ticker}.log"

    # Build the Python command based on flags
    PYTHON_CMD="poetry run python"

    # Add -u flag for unbuffered output in loud mode
    if $LOUD_MODE; then
        PYTHON_CMD="$PYTHON_CMD -u"
    fi

    # Use --output to ensure charts are generated correctly
    PYTHON_CMD="$PYTHON_CMD -m src.main --ticker $ticker --imagedir $IMAGE_DIR --output $TEMP_REPORT"

    # Add --quiet unless in loud mode
    if ! $LOUD_MODE; then
        PYTHON_CMD="$PYTHON_CMD --quiet"
    fi

    # Add --brief unless in loud or verbose mode
    if ! $LOUD_MODE && ! $VERBOSE_MODE; then
        PYTHON_CMD="$PYTHON_CMD --brief"
    fi

    # Add --quick if in quick mode
    if $QUICK_MODE; then
        PYTHON_CMD="$PYTHON_CMD --quick"
    fi

    # Run analysis
    # In loud mode: logs to terminal (stderr/stdout)
    # In other modes: logs captured to temp log
    SUCCESS=false

    if $LOUD_MODE; then
        if $PYTHON_CMD; then
            SUCCESS=true
        fi
    else
        if $PYTHON_CMD > "$TEMP_LOG" 2>&1; then
            SUCCESS=true
        fi
    fi

    if $SUCCESS; then
        # Check if report contains Portfolio Manager failure (soft failure detection)
        # Python exits 0 but report shows PM didn't produce output
        if [ -f "$TEMP_REPORT" ] && grep -q "Portfolio Manager failed to produce final decision" "$TEMP_REPORT"; then
            print_error "Soft failure: $ticker (Portfolio Manager failed)"
            failed=$((failed + 1))

            # Treat as failure for debugging purposes
            local debug_dir="${OUTPUT_DIR}/debug_failures"
            mkdir -p "$debug_dir"
            cp "$TEMP_REPORT" "$debug_dir/${ticker}_report.md" 2>/dev/null || true
            if ! $LOUD_MODE && [ -f "$TEMP_LOG" ]; then
                cp "$TEMP_LOG" "$debug_dir/${ticker}_debug.log" 2>/dev/null || true
            fi
            print_info "Debug files saved to: $debug_dir"
        else
            print_success "Completed: $ticker"
        fi

        # Append report to main output file
        if [ -f "$TEMP_REPORT" ]; then
            cat "$TEMP_REPORT" >> "$OUTPUT_FILE"
            rm "$TEMP_REPORT"
        fi

        # In non-loud mode, append logs if they exist (usually empty with --quiet)
        if ! $LOUD_MODE && [ -f "$TEMP_LOG" ]; then
            cat "$TEMP_LOG" >> "$OUTPUT_FILE"
            rm "$TEMP_LOG"
        fi
    else
        print_error "Failed: $ticker (check logs and $OUTPUT_FILE for details)"
        failed=$((failed + 1))

        # On failure: preserve temp files for debugging, then append to output
        local debug_dir="${OUTPUT_DIR}/debug_failures"
        mkdir -p "$debug_dir"

        if [ -f "$TEMP_REPORT" ]; then
            # Copy to debug dir before appending (preserve for analysis)
            cp "$TEMP_REPORT" "$debug_dir/${ticker}_report.md" 2>/dev/null || true
            cat "$TEMP_REPORT" >> "$OUTPUT_FILE"
            rm "$TEMP_REPORT"
        fi

        if ! $LOUD_MODE && [ -f "$TEMP_LOG" ]; then
            # Copy to debug dir before appending
            cp "$TEMP_LOG" "$debug_dir/${ticker}_debug.log" 2>/dev/null || true
            cat "$TEMP_LOG" >> "$OUTPUT_FILE"
            rm "$TEMP_LOG"
        fi

        print_info "Debug files saved to: $debug_dir"
    fi

    # Add separator to output file
    echo "" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Delay to allow API rate limits (RPM) to reset before the next ticker
    # A single ticker can trigger ~30 LLM calls in parallel.
    # Configurable via COOLDOWN_SECONDS env var (default: 60)
    print_info "Cooling down for ${COOLDOWN_SECONDS} seconds to reset API rate limits..."
    sleep "$COOLDOWN_SECONDS"

done < "$INPUT_FILE"

echo ""
echo "========================================"
print_success "Batch analysis complete!"
echo "========================================"
echo "Total processed: $processed"
echo "Successful: $((processed - failed))"
echo "Failed: $failed"
echo "Results saved to: $OUTPUT_FILE"
echo "Images saved to: $IMAGE_DIR"
