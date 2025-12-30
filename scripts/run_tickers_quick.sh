#!/bin/bash
# Batch ticker analysis script - QUICK MODE
# Wrapper that calls run_tickers.sh with --quick flag
#
# Usage:
#   ./scripts/run_tickers_quick.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_tickers.sh" --quick "$@"
