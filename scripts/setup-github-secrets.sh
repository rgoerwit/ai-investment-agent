#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# ⚠️  GITHUB ACTIONS SECRETS SETUP - ONLY FOR CI/CD USERS
# ════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: Automates uploading API keys to GitHub as encrypted secrets
#          for use in GitHub Actions workflows
#
# DO NOT USE THIS IF: You're only running locally (just use .env file)
#
# USE THIS IF: You're deploying via GitHub Actions and need to set up
#              encrypted secrets in your repository
#
# PREREQUISITES:
#   - GitHub CLI (gh) installed: https://cli.github.com/
#   - Authenticated with: gh auth login
#   - Repository admin access
#   - .env file with your API keys
#
# WHAT THIS DOES:
#   - Reads API keys from your .env file
#   - Uploads them as encrypted secrets to GitHub
#   - Makes them available to GitHub Actions workflows
#
# SAFETY:
#   - Keys are encrypted by GitHub before storage
#   - Only GitHub Actions can decrypt and use them
#   - Your local .env file is NOT modified or uploaded anywhere
#
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

show_usage() {
    cat << 'EOF'
Usage: ./scripts/setup-github-secrets.sh --repo owner/repo [OPTIONS]

Upload API keys from .env to GitHub as encrypted secrets.

REQUIRED:
    --repo OWNER/REPO    GitHub repository (e.g., "username/trading-system")

OPTIONAL:
    --env-file FILE      Path to .env file (default: .env)
    --force              Overwrite existing secrets
    --dry-run            Show what would be done without executing
    -h, --help           Show this help

EXAMPLES:
    # Upload secrets to your repository
    ./scripts/setup-github-secrets.sh --repo myusername/my-trading-system

    # Dry run to see what would happen
    ./scripts/setup-github-secrets.sh --repo myusername/my-repo --dry-run

    # Force update existing secrets
    ./scripts/setup-github-secrets.sh --repo myusername/my-repo --force

WHAT GETS UPLOADED:
    - GOOGLE_API_KEY (required)
    - TAVILY_API_KEY (optional)
    - FMP_API_KEY (optional)
    - LANGSMITH_API_KEY (optional)

These are read from your .env file and uploaded as encrypted GitHub secrets.

EOF
    exit 0
}

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

# Check prerequisites
check_prerequisites() {
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) is not installed"
        echo ""
        echo "Install from: https://cli.github.com/"
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        print_error "Not authenticated with GitHub CLI"
        echo ""
        echo "Run: gh auth login"
        exit 1
    fi

    print_success "GitHub CLI is ready"
}

# Load environment file
load_env_file() {
    local env_file="$1"

    if [[ ! -f "$env_file" ]]; then
        print_error "Environment file not found: $env_file"
        exit 1
    fi

    # Source the file
    set -a
    source "$env_file"
    set +a

    print_success "Loaded $env_file"
}

# Set a secret in GitHub
set_secret() {
    local repo="$1"
    local secret_name="$2"
    local secret_value="$3"
    local force="$4"
    local dry_run="$5"

    if [[ "$dry_run" == "true" ]]; then
        print_info "[DRY RUN] Would set: $secret_name"
        return 0
    fi

    # Check if secret already exists
    if gh secret list --repo "$repo" 2>/dev/null | grep -q "^$secret_name"; then
        if [[ "$force" == "true" ]]; then
            print_warning "Updating existing secret: $secret_name"
        else
            print_warning "Secret exists (use --force to overwrite): $secret_name"
            return 0
        fi
    else
        print_info "Setting new secret: $secret_name"
    fi

    # Set the secret
    if echo "$secret_value" | gh secret set "$secret_name" --repo "$repo"; then
        print_success "Set: $secret_name"
    else
        print_error "Failed to set: $secret_name"
        return 1
    fi
}

# Main function
main() {
    local repo=""
    local env_file=".env"
    local force="false"
    local dry_run="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --repo)
                repo="$2"
                shift 2
                ;;
            --env-file)
                env_file="$2"
                shift 2
                ;;
            --force)
                force="true"
                shift
                ;;
            --dry-run)
                dry_run="true"
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

    # Validate required arguments
    if [[ -z "$repo" ]]; then
        print_error "Repository is required"
        echo ""
        show_usage
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  GitHub Secrets Setup"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    print_info "Repository: $repo"
    print_info "Environment file: $env_file"
    if [[ "$dry_run" == "true" ]]; then
        print_warning "DRY RUN MODE - no changes will be made"
    fi
    echo ""

    # Run checks
    check_prerequisites
    cd "$REPO_ROOT" || exit 1
    load_env_file "$env_file"

    echo ""
    print_info "Setting GitHub secrets..."
    echo ""

    # Set API secrets
    local secrets_set=0
    local secrets_failed=0

    # Required secret
    if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
        if set_secret "$repo" "GOOGLE_API_KEY" "$GOOGLE_API_KEY" "$force" "$dry_run"; then
            secrets_set=$((secrets_set + 1))
        else
            secrets_failed=$((secrets_failed + 1))
        fi
    else
        print_warning "GOOGLE_API_KEY not found in $env_file"
    fi

    # Optional secrets
    for secret_name in "TAVILY_API_KEY" "FMP_API_KEY" "LANGSMITH_API_KEY"; do
        secret_value="${!secret_name:-}"
        if [[ -n "$secret_value" ]]; then
            if set_secret "$repo" "$secret_name" "$secret_value" "$force" "$dry_run"; then
                secrets_set=$((secrets_set + 1))
            else
                secrets_failed=$((secrets_failed + 1))
            fi
        fi
    done

    # Summary
    echo ""
    echo "════════════════════════════════════════════════════════════"
    if [[ "$dry_run" == "true" ]]; then
        print_info "Dry run complete!"
        print_info "Would have set $secrets_set secret(s)"
    else
        if [[ $secrets_failed -eq 0 ]]; then
            print_success "All secrets uploaded successfully!"
            echo ""
            print_info "Secrets set: $secrets_set"
            print_info "GitHub Actions workflows can now use these secrets"
        else
            print_error "Some secrets failed to upload"
            print_info "Successful: $secrets_set"
            print_info "Failed: $secrets_failed"
            exit 1
        fi
    fi
    echo ""
}

main "$@"
