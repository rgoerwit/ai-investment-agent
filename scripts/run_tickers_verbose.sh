#!/bin/bash
# Batch ticker analysis script - VERBOSE MODE
# Wrapper that calls run_tickers.sh with --verbose flag
#
# Usage:
#   ./scripts/run_tickers_verbose.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_tickers.sh" --verbose "$@"
