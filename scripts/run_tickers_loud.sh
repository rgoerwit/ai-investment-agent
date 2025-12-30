#!/bin/bash
# Batch ticker analysis script - LOUD MODE
# Wrapper that calls run_tickers.sh with --loud flag
#
# Usage:
#   bash scripts/run_tickers_loud.sh 2>scratch/ticker_analysis_info.txt &
#   tail -f scratch/ticker_analysis_info.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_tickers.sh" --loud "$@"
