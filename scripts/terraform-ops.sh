#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# ⚠️  ⚠️  ⚠️  DESTRUCTIVE TERRAFORM OPERATIONS - EXPERTS ONLY  ⚠️  ⚠️  ⚠️
# ════════════════════════════════════════════════════════════════════════════
#
# This script performs Terraform operations on Azure infrastructure.
# The 'destroy' command will DELETE ALL RESOURCES and CANNOT BE UNDONE.
#
# DO NOT RUN THIS SCRIPT UNLESS YOU:
#   ✓ Have a configured Azure subscription
#   ✓ Have set up Terraform backend (setup-terraform-backend.sh)
#   ✓ Understand what Terraform plan/apply/destroy do
#   ✓ Are comfortable with infrastructure-as-code
#   ✓ Have tested in a dev environment first
#
# IF YOU JUST WANT TO ANALYZE STOCKS LOCALLY:
#   DO NOT USE THIS SCRIPT - use ./scripts/run-analysis.sh instead
#
# COMMANDS:
#   validate  - Check Terraform configuration (safe, no changes)
#   plan      - Show what would be changed (safe, no changes)
#   apply     - Create/update infrastructure (CREATES BILLABLE RESOURCES)
#   destroy   - DELETE ALL INFRASTRUCTURE (IRREVERSIBLE)
#   output    - Show Terraform outputs (safe, no changes)
#
# SAFETY FEATURES:
#   - Requires explicit --env flag
#   - Shows confirmation prompts before destructive operations
#   - Supports --dry-run for testing
#   - No --auto-approve by default (must confirm manually)
#
# ESTIMATED COSTS (per environment):
#   - Azure Container Instance: ~$30-45/month
#   - Container Registry: ~$5/month
#   - Total: ~$35-50/month
#
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

print_danger() {
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  ⚠️  ⚠️         DANGER ZONE         ⚠️  ⚠️  ⚠️        ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
}

show_usage() {
    cat << 'EOF'
Usage: ./scripts/terraform-ops.sh COMMAND --env ENVIRONMENT [OPTIONS]

Perform Terraform operations on Azure infrastructure.

COMMANDS:
    validate    Validate Terraform configuration (safe)
    plan        Show planned infrastructure changes (safe)
    apply       Create/update infrastructure (CREATES BILLABLE RESOURCES)
    destroy     DELETE all infrastructure (IRREVERSIBLE)
    output      Show Terraform outputs (safe)

REQUIRED:
    --env ENV   Target environment: dev, staging, or prod

OPTIONS:
    --dry-run   Show what would be done without executing (for apply/destroy)
    --help      Show this help

EXAMPLES:
    # Safe operations (no changes)
    ./scripts/terraform-ops.sh validate --env dev
    ./scripts/terraform-ops.sh plan --env dev
    ./scripts/terraform-ops.sh output --env prod

    # Test what would happen (dry run)
    ./scripts/terraform-ops.sh apply --env dev --dry-run
    ./scripts/terraform-ops.sh destroy --env dev --dry-run

    # Real operations (will prompt for confirmation)
    ./scripts/terraform-ops.sh apply --env dev
    ./scripts/terraform-ops.sh destroy --env dev

SAFETY NOTES:
    - 'apply' requires confirmation before creating resources
    - 'destroy' requires typing environment name to confirm deletion
    - Use --dry-run first to see what would happen
    - Always test in 'dev' environment before 'staging' or 'prod'

EOF
    exit 0
}

check_prerequisites() {
    local missing=()

    if ! command -v terraform &> /dev/null; then
        missing+=("terraform")
    fi

    if ! command -v az &> /dev/null; then
        missing+=("azure-cli")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing[*]}"
        echo ""
        echo "Install:"
        echo "  terraform: https://developer.hashicorp.com/terraform/downloads"
        echo "  azure-cli: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi

    # Check Azure login
    if ! az account show &> /dev/null; then
        print_error "Not logged into Azure"
        echo ""
        print_info "Run: az login"
        exit 1
    fi
}

validate_environment() {
    local env="$1"

    case "$env" in
        dev|staging|prod) ;;
        *)
            print_error "Invalid environment: $env"
            echo "Must be: dev, staging, or prod"
            exit 1
            ;;
    esac

    # Check if Terraform directory exists
    local tf_dir="$REPO_ROOT/terraform/environments/$env"
    if [[ ! -d "$tf_dir" ]]; then
        print_error "Terraform directory not found: $tf_dir"
        echo ""
        print_info "Create this directory with your Terraform configuration first"
        exit 1
    fi
}

load_environment_vars() {
    local env_file="$REPO_ROOT/.env"

    if [[ -f "$env_file" ]]; then
        print_info "Loading environment variables from .env..."
        set -a
        source "$env_file"
        set +a
    else
        print_warning ".env file not found - using runtime environment only"
    fi

    # Export Terraform variables
    export TF_VAR_google_api_key="${GOOGLE_API_KEY:-}"
    export TF_VAR_tavily_api_key="${TAVILY_API_KEY:-}"
    export TF_VAR_fmp_api_key="${FMP_API_KEY:-}"
    export TF_VAR_langsmith_api_key="${LANGSMITH_API_KEY:-}"
}

cmd_validate() {
    local env="$1"
    local tf_dir="$REPO_ROOT/terraform/environments/$env"

    print_info "Validating Terraform configuration for $env..."

    cd "$tf_dir" || exit 1

    # Initialize (without backend for validation)
    terraform init -backend=false

    # Validate
    if terraform validate; then
        print_success "Terraform configuration is valid"
    else
        print_error "Terraform configuration has errors"
        exit 1
    fi
}

cmd_plan() {
    local env="$1"
    local tf_dir="$REPO_ROOT/terraform/environments/$env"

    print_info "Generating Terraform plan for $env..."

    cd "$tf_dir" || exit 1

    # Initialize with backend
    terraform init

    # Plan
    terraform plan -detailed-exitcode || {
        exit_code=$?
        if [[ $exit_code -eq 2 ]]; then
            echo ""
            print_warning "Changes detected - review plan above"
        else
            print_error "Plan failed with exit code $exit_code"
            exit $exit_code
        fi
    }
}

cmd_apply() {
    local env="$1"
    local dry_run="$2"
    local tf_dir="$REPO_ROOT/terraform/environments/$env"

    echo ""
    print_warning "About to apply Terraform changes for: $env"
    print_warning "This will CREATE or MODIFY Azure resources"
    print_warning "Estimated cost: ~$35-50/month per environment"
    echo ""

    if [[ "$dry_run" == "true" ]]; then
        print_info "DRY RUN MODE - showing plan only"
        cmd_plan "$env"
        return 0
    fi

    # Confirmation
    read -p "Type the environment name '$env' to confirm: " -r
    echo
    if [[ "$REPLY" != "$env" ]]; then
        print_error "Confirmation failed - aborting"
        exit 1
    fi

    cd "$tf_dir" || exit 1

    # Initialize
    terraform init

    # Apply (will prompt for confirmation)
    print_info "Applying Terraform changes..."
    if terraform apply; then
        print_success "Terraform apply complete!"
        echo ""
        terraform output
    else
        print_error "Terraform apply failed"
        exit 1
    fi
}

cmd_destroy() {
    local env="$1"
    local dry_run="$2"
    local tf_dir="$REPO_ROOT/terraform/environments/$env"

    echo ""
    print_danger
    echo ""
    print_error "DESTROY COMMAND - THIS WILL DELETE ALL INFRASTRUCTURE"
    print_error "Environment: $env"
    print_error "This operation CANNOT be undone!"
    echo ""

    if [[ "$dry_run" == "true" ]]; then
        print_info "DRY RUN MODE - showing what would be destroyed"
        cd "$tf_dir" || exit 1
        terraform init
        terraform plan -destroy
        return 0
    fi

    # Multiple confirmations for destroy
    read -p "Are you ABSOLUTELY SURE you want to destroy $env? (yes/no): " -r
    echo
    if [[ "$REPLY" != "yes" ]]; then
        print_info "Destroy cancelled"
        exit 0
    fi

    read -p "Type the environment name '$env' to confirm destruction: " -r
    echo
    if [[ "$REPLY" != "$env" ]]; then
        print_error "Confirmation failed - aborting"
        exit 1
    fi

    echo ""
    print_warning "Last chance to cancel (Ctrl+C now!)..."
    sleep 5
    echo ""

    cd "$tf_dir" || exit 1

    # Initialize
    terraform init

    # Destroy (will prompt for final confirmation)
    print_error "Destroying infrastructure..."
    if terraform destroy; then
        print_success "Infrastructure destroyed"
    else
        print_error "Destroy failed - check for orphaned resources"
        exit 1
    fi
}

cmd_output() {
    local env="$1"
    local tf_dir="$REPO_ROOT/terraform/environments/$env"

    print_info "Terraform outputs for $env:"
    echo ""

    cd "$tf_dir" || exit 1
    terraform init
    terraform output
}

main() {
    local command=""
    local environment=""
    local dry_run="false"

    # Parse arguments
    if [[ $# -eq 0 ]]; then
        show_usage
    fi

    command="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                environment="$2"
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --help|-h)
                show_usage
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                ;;
        esac
    done

    # Validate environment is provided
    if [[ -z "$environment" ]]; then
        print_error "Environment is required (use --env)"
        echo ""
        show_usage
    fi

    # Header
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Terraform Operations - Multi-Agent Trading System"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    print_info "Command: $command"
    print_info "Environment: $environment"
    if [[ "$dry_run" == "true" ]]; then
        print_warning "Mode: DRY RUN"
    fi
    echo ""

    # Checks
    check_prerequisites
    validate_environment "$environment"
    load_environment_vars

    # Execute command
    case "$command" in
        validate)
            cmd_validate "$environment"
            ;;
        plan)
            cmd_plan "$environment"
            ;;
        apply)
            cmd_apply "$environment" "$dry_run"
            ;;
        destroy)
            cmd_destroy "$environment" "$dry_run"
            ;;
        output)
            cmd_output "$environment"
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            echo "Valid commands: validate, plan, apply, destroy, output"
            exit 1
            ;;
    esac

    echo ""
}

main "$@"
