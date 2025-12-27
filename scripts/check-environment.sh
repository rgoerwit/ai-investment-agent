#!/bin/bash
# Environment validation script
# Checks that .env is properly configured before running analysis

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Multi-Agent Trading System - Environment Check"
echo "════════════════════════════════════════════════════════════"
echo ""

cd "$REPO_ROOT" || exit 1

# Check for .env file
print_info "Checking for .env file..."
if [[ ! -f ".env" ]]; then
    print_error "No .env file found"

    if [[ -f ".env.example" ]]; then
        echo ""
        print_info "Found .env.example - creating .env..."
        cp .env.example .env
        print_success "Created .env from template"
        echo ""
        print_warning "You must edit .env and add your API keys!"
        print_info "Required: GOOGLE_API_KEY"
        print_info "Optional: TAVILY_API_KEY, FMP_API_KEY, LANGSMITH_API_KEY"
        echo ""
        print_info "Get free Gemini API key: https://ai.google.dev/"
        exit 1
    else
        print_error "No .env or .env.example found"
        exit 1
    fi
else
    print_success "Found .env file"
fi

echo ""
print_info "Validating API keys..."

# Required variables
REQUIRED_VARS=("GOOGLE_API_KEY")

# Optional variables
OPTIONAL_VARS=("TAVILY_API_KEY" "FMP_API_KEY" "LANGSMITH_API_KEY")

errors=0
warnings=0

# Check required variables
for var in "${REQUIRED_VARS[@]}"; do
    if grep -q "^${var}=" .env && ! grep -q "^${var}=.*your_.*_here" .env; then
        # Variable exists and doesn't have placeholder value
        value=$(grep "^${var}=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [[ -n "$value" ]] && [[ ${#value} -gt 10 ]]; then
            print_success "$var configured"
        else
            print_error "$var exists but looks invalid (too short)"
            errors=$((errors + 1))
        fi
    else
        print_error "$var missing or not configured"
        errors=$((errors + 1))
    fi
done

# Check optional variables
for var in "${OPTIONAL_VARS[@]}"; do
    if grep -q "^${var}=" .env && ! grep -q "^${var}=.*your_.*_here" .env; then
        value=$(grep "^${var}=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [[ -n "$value" ]] && [[ ${#value} -gt 5 ]]; then
            print_success "$var configured (optional)"
        else
            print_warning "$var exists but looks invalid"
            warnings=$((warnings + 1))
        fi
    else
        print_warning "$var not configured (optional)"
        warnings=$((warnings + 1))
    fi
done

echo ""

# Check Python/Poetry
print_info "Checking Python environment..."

if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    print_success "Python $python_version found"
else
    print_error "Python 3 not found"
    errors=$((errors + 1))
fi

if command -v poetry &> /dev/null; then
    poetry_version=$(poetry --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    print_success "Poetry $poetry_version found"
else
    print_error "Poetry not found"
    print_info "Install from: https://python-poetry.org/docs/#installation"
    errors=$((errors + 1))
fi

echo ""
echo "════════════════════════════════════════════════════════════"

# Final verdict
if [[ $errors -gt 0 ]]; then
    echo ""
    print_error "Environment check failed ($errors errors, $warnings warnings)"
    echo ""
    print_info "Please fix the errors above before running analysis"
    print_info "Edit .env to add missing API keys"
    exit 1
else
    echo ""
    print_success "Environment check passed!"
    if [[ $warnings -gt 0 ]]; then
        echo ""
        print_warning "You have $warnings optional features not configured"
        print_info "The system will work, but some features may be limited"
    fi
    echo ""
    print_info "You're ready to run analysis:"
    echo "  ./scripts/run-analysis.sh --ticker AAPL"
    echo ""
    print_info "Or run the health check:"
    echo "  poetry run python src/health_check.py"
    echo ""
fi
