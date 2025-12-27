#!/bin/bash
# Multi-Agent Investment Analysis System - Run Script
# Simple wrapper for analyzing individual stock tickers

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
TICKER=""
QUICK_MODE=false
VERBOSE=false

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << 'EOF'
Usage: ./scripts/run-analysis.sh --ticker SYMBOL [OPTIONS]

Analyze a single stock ticker using the multi-agent system.

OPTIONS:
    -t, --ticker SYMBOL    Stock ticker symbol (required)
    -q, --quick           Use quick analysis mode (faster, less detailed)
    -v, --verbose         Enable verbose output
    -h, --help            Show this help message

EXAMPLES:
    # Basic analysis
    ./scripts/run-analysis.sh --ticker AAPL

    # Quick analysis (faster)
    ./scripts/run-analysis.sh --ticker NVDA --quick

    # Verbose mode
    ./scripts/run-analysis.sh --ticker MSFT --verbose

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--ticker)
            if [[ -z "${2:-}" ]]; then
                print_error "Ticker symbol required after --ticker"
                exit 1
            fi
            TICKER="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate ticker is provided
if [[ -z "$TICKER" ]]; then
    print_error "Ticker symbol is required"
    echo ""
    show_usage
fi

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    print_warning ".env file not found"

    if [[ -f ".env.example" ]]; then
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env and add your API keys before running"
        print_info "Required: GOOGLE_API_KEY (get free key at https://ai.google.dev/)"
        exit 1
    else
        print_error "No .env or .env.example found. Please create .env with your API keys."
        exit 1
    fi
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed"
    print_info "Install from: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Build command arguments
ARGS="--ticker $TICKER"
if [[ "$QUICK_MODE" == "true" ]]; then
    ARGS="$ARGS --quick"
fi
if [[ "$VERBOSE" == "true" ]]; then
    ARGS="$ARGS --verbose"
fi

# Run the analysis
print_info "Starting analysis for $TICKER..."
if [[ "$QUICK_MODE" == "true" ]]; then
    print_info "Mode: Quick (2-4 minutes)"
else
    print_info "Mode: Standard (8-10 minutes)"
fi

# Execute with poetry
poetry run python -m src.main $ARGS

# Check exit status
if [[ $? -eq 0 ]]; then
    print_success "Analysis complete!"
else
    print_error "Analysis failed. Check output above for errors."
    exit 1
fi
