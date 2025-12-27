#!/bin/bash
# Batch ticker analysis script - LOUD MODE
# Reads tickers from scratch/sample_tickers.txt and analyzes each one sequentially
# Emits verbose logging to stderr for monitoring with: tail -f
#
# Usage:
#   bash scripts/run_tickers_loud.sh 2>scratch/ticker_analysis_info.txt &
#   tail -f scratch/ticker_analysis_info.txt

set -euo pipefail

# === CRITICAL FIX FOR MACOS ===
# Prevents gRPC "fork_posix.cc:71" errors and hanging on Apple Silicon
export GRPC_POLL_STRATEGY=poll
export GRPC_VERBOSITY=ERROR  # Changed from INFO to ERROR - suppress gRPC noise in loud mode
# ==============================

# Configuration
DEFAULT_INPUT_FILE="scratch/sample_tickers.txt"
DEFAULT_OUTPUT_FILE="scratch/ticker_analysis_results.md"

INPUT_FILE="${1:-$DEFAULT_INPUT_FILE}"
OUTPUT_FILE="${2:-$DEFAULT_OUTPUT_FILE}"

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

# Show usage
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat << 'EOF'
Usage: ./scripts/run_tickers_loud.sh [INPUT_FILE] [OUTPUT_FILE]

Batch analyze multiple tickers with VERBOSE LOGGING to stderr.
RUN FROM REPO ROOT DIRECTORY.

ARGUMENTS:
    INPUT_FILE     File containing ticker symbols (default: scratch/sample_tickers.txt)
    OUTPUT_FILE    Output markdown file (default: scratch/ticker_analysis_results.md)

INPUT FILE FORMAT:
    One ticker per line, e.g.:
    AAPL
    MSFT
    NVDA

EXAMPLES:
    # Run with real-time log monitoring
    bash scripts/run_tickers_loud.sh 2>scratch/ticker_analysis_info.txt &
    tail -f scratch/ticker_analysis_info.txt

    # Run with logs to terminal (no background)
    ./scripts/run_tickers_loud.sh

    # Custom files with log capture
    bash scripts/run_tickers_loud.sh my_tickers.txt my_results.md 2>my_logs.txt &
    tail -f my_logs.txt

NOTES:
    - Analysis reports go to OUTPUT_FILE (stdout)
    - Progress logs go to stderr (can be redirected with 2>)
    - Use -u flag ensures unbuffered Python output for real-time logs
    - GRPC_VERBOSITY set to ERROR to reduce noise

EOF
    exit 0
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

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

print_info "Starting batch analysis..."
print_info "Input: $INPUT_FILE ($ticker_count tickers)"
print_info "Output: $OUTPUT_FILE"
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

    # Run analysis with unbuffered output
    # stdout (report) goes to file, stderr (logs) goes to terminal/redirected stderr
    if poetry run python -u -m src.main --ticker "$ticker" >> "$OUTPUT_FILE"; then
        print_success "Completed: $ticker"
    else
        print_error "Failed: $ticker (check logs and $OUTPUT_FILE for details)"
        failed=$((failed + 1))
    fi

    # Add separator to output file
    echo "" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Small delay to allow connections to close cleanly
    sleep 2

done < "$INPUT_FILE"

echo ""
echo "========================================"
print_success "Batch analysis complete!"
echo "========================================"
echo "Total processed: $processed"
echo "Successful: $((processed - failed))"
echo "Failed: $failed"
echo "Results saved to: $OUTPUT_FILE"
