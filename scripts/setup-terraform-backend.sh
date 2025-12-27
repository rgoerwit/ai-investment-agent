#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# ⚠️  ⚠️  ⚠️  CREATES BILLABLE AZURE RESOURCES - DO NOT RUN CASUALLY! ⚠️  ⚠️  ⚠️
# ════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: One-time setup of Terraform remote state storage in Azure
#
# WHAT THIS CREATES (AND COSTS MONEY):
#   - Azure Resource Group
#   - Azure Storage Account (~$1-2/month)
#   - Blob Container for Terraform state
#
# ONLY USE THIS IF:
#   ✓ You're deploying this trading system to Azure using Terraform
#   ✓ You have an active Azure subscription
#   ✓ You understand Terraform remote state backends
#   ✓ You're comfortable with Azure billing
#
# DO NOT USE THIS IF:
#   ✗ You just want to run the system locally (use .env instead)
#   ✗ You don't have an Azure subscription
#   ✗ You're not familiar with Azure and Terraform
#
# ESTIMATED COST: ~$1-2/month for storage account
#
# PREREQUISITES:
#   - Azure CLI installed: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
#   - Authenticated with: az login
#   - Active Azure subscription
#   - Permissions to create resource groups
#
# WHAT HAPPENS:
#   1. Checks Azure CLI authentication
#   2. Creates a resource group for Terraform state
#   3. Creates a storage account (GLOBALLY UNIQUE NAME REQUIRED)
#   4. Creates a blob container for state files
#   5. Prints backend configuration for your Terraform
#
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
DEFAULT_RESOURCE_GROUP="tfstate-rg"
DEFAULT_LOCATION="eastus"
DEFAULT_CONTAINER="tfstate"

# Generate unique storage account name (must be globally unique across all Azure)
# Format: tfstate<random-hex> (storage account names: lowercase, alphanumeric, 3-24 chars)
DEFAULT_STORAGE_ACCOUNT="tfstate$(openssl rand -hex 4)"

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

show_usage() {
    cat << 'EOF'
Usage: ./scripts/setup-terraform-backend.sh [OPTIONS]

Create Azure backend storage for Terraform state (CREATES BILLABLE RESOURCES).

OPTIONS:
    --resource-group NAME  Resource group name (default: tfstate-rg)
    --storage-account NAME Storage account name (default: tfstate<random>)
    --location REGION      Azure region (default: eastus)
    --container NAME       Blob container name (default: tfstate)
    --dry-run              Show what would be created without executing
    -h, --help             Show this help

EXAMPLES:
    # Use defaults (dry run first!)
    ./scripts/setup-terraform-backend.sh --dry-run
    ./scripts/setup-terraform-backend.sh

    # Custom names and location
    ./scripts/setup-terraform-backend.sh \
        --resource-group my-tfstate-rg \
        --storage-account mytfstate12345 \
        --location westus2

IMPORTANT:
    - Storage account names must be GLOBALLY UNIQUE across all of Azure
    - Names must be lowercase, alphanumeric, 3-24 characters
    - This creates resources that cost ~$1-2/month
    - Use --dry-run first to see what would be created

EOF
    exit 0
}

check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed"
        echo ""
        echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        print_error "jq is not installed (required for JSON parsing)"
        echo ""
        echo "Install: brew install jq (macOS) or apt-get install jq (Linux)"
        exit 1
    fi

    print_success "Prerequisites found"
}

check_azure_login() {
    print_info "Checking Azure authentication..."

    if ! az account show &> /dev/null; then
        print_error "Not logged into Azure"
        echo ""
        print_info "Attempting Azure login..."
        if ! az login; then
            print_error "Azure login failed"
            exit 1
        fi
    fi

    local subscription_name
    subscription_name=$(az account show --query name -o tsv)
    local subscription_id
    subscription_id=$(az account show --query id -o tsv)

    print_success "Logged in to Azure"
    echo ""
    print_warning "Using subscription: $subscription_name"
    print_info "Subscription ID: $subscription_id"
    echo ""
    read -p "Is this the correct subscription? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Please switch to the correct subscription with:"
        echo "  az account set --subscription <subscription-id>"
        exit 1
    fi
}

create_resource_group() {
    local rg_name="$1"
    local location="$2"
    local dry_run="$3"

    if [[ "$dry_run" == "true" ]]; then
        print_info "[DRY RUN] Would create resource group: $rg_name in $location"
        return 0
    fi

    print_info "Creating resource group: $rg_name..."

    # Check if exists
    if az group show --name "$rg_name" &> /dev/null; then
        print_warning "Resource group already exists: $rg_name"
        return 0
    fi

    if az group create --name "$rg_name" --location "$location" --output none; then
        print_success "Created resource group: $rg_name"
    else
        print_error "Failed to create resource group"
        exit 1
    fi
}

create_storage_account() {
    local rg_name="$1"
    local storage_name="$2"
    local location="$3"
    local dry_run="$4"

    if [[ "$dry_run" == "true" ]]; then
        print_info "[DRY RUN] Would create storage account: $storage_name"
        return 0
    fi

    # Validate storage account name
    if [[ ! "$storage_name" =~ ^[a-z0-9]{3,24}$ ]]; then
        print_error "Invalid storage account name: $storage_name"
        echo "Must be lowercase, alphanumeric, 3-24 characters"
        exit 1
    fi

    print_info "Creating storage account: $storage_name..."
    print_warning "This will incur charges (~$1-2/month)"

    # Check if exists
    if az storage account show --name "$storage_name" --resource-group "$rg_name" &> /dev/null 2>&1; then
        print_warning "Storage account already exists: $storage_name"
        return 0
    fi

    if az storage account create \
        --name "$storage_name" \
        --resource-group "$rg_name" \
        --location "$location" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --min-tls-version TLS1_2 \
        --allow-blob-public-access false \
        --output none; then
        print_success "Created storage account: $storage_name"
    else
        print_error "Failed to create storage account"
        print_info "Name may already be taken globally (try another name)"
        exit 1
    fi
}

create_container() {
    local storage_name="$1"
    local container_name="$2"
    local dry_run="$3"

    if [[ "$dry_run" == "true" ]]; then
        print_info "[DRY RUN] Would create blob container: $container_name"
        return 0
    fi

    print_info "Creating blob container: $container_name..."

    if az storage container create \
        --name "$container_name" \
        --account-name "$storage_name" \
        --auth-mode login \
        --output none; then
        print_success "Created blob container: $container_name"
    else
        print_error "Failed to create blob container"
        exit 1
    fi
}

show_backend_config() {
    local rg_name="$1"
    local storage_name="$2"
    local container_name="$3"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    print_success "Terraform Backend Setup Complete!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Add this to your Terraform configuration (backend.tf):"
    echo ""
    cat << EOF
terraform {
  backend "azurerm" {
    resource_group_name  = "$rg_name"
    storage_account_name = "$storage_name"
    container_name       = "$container_name"
    key                  = "terraform.tfstate"
  }
}
EOF
    echo ""
    print_info "Run 'terraform init' in your Terraform directory to use this backend"
    echo ""
}

main() {
    local resource_group="$DEFAULT_RESOURCE_GROUP"
    local storage_account="$DEFAULT_STORAGE_ACCOUNT"
    local location="$DEFAULT_LOCATION"
    local container="$DEFAULT_CONTAINER"
    local dry_run="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --resource-group)
                resource_group="$2"
                shift 2
                ;;
            --storage-account)
                storage_account="$2"
                shift 2
                ;;
            --location)
                location="$2"
                shift 2
                ;;
            --container)
                container="$2"
                shift 2
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

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "⚠️  Terraform Backend Setup - Creates Billable Resources"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    print_info "Resource Group: $resource_group"
    print_info "Storage Account: $storage_account"
    print_info "Location: $location"
    print_info "Container: $container"
    echo ""
    if [[ "$dry_run" == "true" ]]; then
        print_warning "DRY RUN MODE - no resources will be created"
    else
        print_warning "This will create billable Azure resources (~$1-2/month)"
    fi
    echo ""

    # Checks
    check_prerequisites
    check_azure_login

    # Create resources
    create_resource_group "$resource_group" "$location" "$dry_run"
    create_storage_account "$resource_group" "$storage_account" "$location" "$dry_run"
    create_container "$storage_account" "$container" "$dry_run"

    if [[ "$dry_run" == "true" ]]; then
        echo ""
        print_info "Dry run complete - no resources were created"
        print_info "Remove --dry-run to actually create resources"
    else
        show_backend_config "$resource_group" "$storage_account" "$container"
    fi

    echo ""
}

main "$@"
