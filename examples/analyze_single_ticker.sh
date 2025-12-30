#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Single Ticker Analysis Example
# ═══════════════════════════════════════════════════════════════════════════
#
# A minimal example script demonstrating how to analyze one stock ticker.
# This script includes setup validation and error handling.
#
# Usage:
#   ./examples/analyze_single_ticker.sh <TICKER>
#
# Example:
#   ./examples/analyze_single_ticker.sh 0005.HK
#
# For batch analysis of multiple tickers, see: scripts/run_tickers.sh
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

TICKER="${1:-}"
QUICK_MODE="${2:-}"  # Pass --quick as second argument for faster analysis

# ────────────────────────────────────────────────────────────────────────────
# Validation: Check Prerequisites
# ────────────────────────────────────────────────────────────────────────────

# Check if ticker provided
if [ -z "$TICKER" ]; then
    echo "❌ Error: No ticker provided"
    echo ""
    echo "Usage: $0 <TICKER> [--quick]"
    echo ""
    echo "Examples:"
    echo "  $0 0005.HK          # Analyze HSBC Holdings (Hong Kong)"
    echo "  $0 7203.T --quick   # Analyze Toyota (Japan) in quick mode"
    echo "  $0 2330.TW          # Analyze TSMC (Taiwan)"
    echo ""
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry is not installed"
    echo ""
    echo "Please install Poetry first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    echo ""
    echo "Or see: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check if dependencies are installed (check for .venv directory)
if [ ! -d ".venv" ]; then
    echo "❌ Error: Dependencies not installed"
    echo ""
    echo "Please install dependencies first:"
    echo "  poetry install"
    echo ""
    echo "For detailed setup instructions, see: README.md"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo ""
    echo "Please create .env file with your API keys:"
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your API keys"
    echo ""
    echo "Required API keys:"
    echo "  - GOOGLE_API_KEY (Google Gemini - get from https://aistudio.google.com/)"
    echo "  - FINNHUB_API_KEY (Market data - get from https://finnhub.io/)"
    echo "  - TAVILY_API_KEY (Web search - get from https://tavily.com/)"
    echo ""
    echo "For detailed setup instructions, see: README.md"
    exit 1
fi

# Check if critical API keys are set (basic validation)
if ! grep -q "^GOOGLE_API_KEY=" .env || \
   ! grep -q "^FINNHUB_API_KEY=" .env || \
   ! grep -q "^TAVILY_API_KEY=" .env; then
    echo "⚠️  Warning: Some required API keys may not be configured in .env"
    echo ""
    echo "Please ensure these are set:"
    echo "  - GOOGLE_API_KEY"
    echo "  - FINNHUB_API_KEY"
    echo "  - TAVILY_API_KEY"
    echo ""
    echo "Continuing anyway (the analysis will fail if keys are missing)..."
    echo ""
fi

# ────────────────────────────────────────────────────────────────────────────
# Setup: Environment Variables
# ────────────────────────────────────────────────────────────────────────────

# Suppress gRPC fork() warnings (macOS issue)
export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=""

# ────────────────────────────────────────────────────────────────────────────
# Main: Run Analysis
# ────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Investment Agent - Single Ticker Analysis"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Ticker:      $TICKER"
echo "Mode:        ${QUICK_MODE:-standard}"
echo "Date:        $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo ""

# Run analysis
# Using --output to ensure charts are generated and saved correctly
CMD="poetry run python -m src.main --ticker $TICKER --output results/${TICKER}.md"
if [ "$QUICK_MODE" = "--quick" ]; then
    CMD="$CMD --quick"
fi

# Run analysis
echo "▶️  Starting analysis..."
echo "   Command: $CMD"
echo ""

# Execute and capture exit code
set +e  # Don't exit on error (we want to handle it)
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Completion: Check Results
# ────────────────────────────────────────────────────────────────────────────

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
    echo ""

    # Find the most recent result file for this ticker
    TICKER_SAFE=$(echo "$TICKER" | tr '.' '_')
    RESULT_FILE=$(ls -t results/${TICKER_SAFE}_*.md 2>/dev/null | head -n1)

    if [ -n "$RESULT_FILE" ]; then
        echo "📊 Results saved to: $RESULT_FILE"
        echo ""
        echo "To view the report:"
        echo "  cat $RESULT_FILE"
        echo "  # or open in your text editor"
        echo ""
    else
        echo "⚠️  Results file not found in results/ directory"
        echo ""
    fi

    echo "Next steps:"
    echo "  1. Review the analysis report"
    echo "  2. Conduct additional due diligence"
    echo "  3. Make informed investment decision"
    echo ""
    echo "Reminder: This is a research tool, not financial advice!"

else
    echo "❌ Analysis failed with exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Missing or invalid API keys in .env"
    echo "  - Network connectivity problems"
    echo "  - Invalid ticker format"
    echo "  - Rate limiting (wait a few minutes and retry)"
    echo ""
    echo "For troubleshooting help, see: README.md"
    exit $EXIT_CODE
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
