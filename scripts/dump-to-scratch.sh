#!/bin/bash
# This script creates a single text file archive of the entire repository,
# suitable for analysis by an AI. It includes all source code, configuration,
# documentation, and scripts while excluding transient state, local dependencies,
# and secrets.

set -euo pipefail

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
OUTPUT_DIR="${REPO_ROOT}/scratch"
OUTPUT_FILE="${OUTPUT_DIR}/repository-archive.txt"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --output FILE    Output file path (default: scratch/repository-archive.txt)"
    echo "  --include-large  Include large files (>1MB)"
    echo "  --include-logs   Include log files"
    echo "  --help           Show this help message"
    echo
    echo "This script creates a comprehensive archive of the repository suitable for AI analysis."
}

create_output_directory() {
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR"
        echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"
    fi
}

write_header() {
    local output_file="$1"

    cat > "$output_file" << EOF
# Multi-Agent Trading System Repository Archive
# Generated on: $TIMESTAMP
# Repository root: $REPO_ROOT

This is a comprehensive archive of the Multi-Agent Trading System repository,
created for AI analysis and code review purposes.

## Repository Structure Overview

The repository contains:
- src/: Python source code for the multi-agent trading system
- terraform/: Infrastructure as Code for Azure deployment
- scripts/: Deployment and utility scripts
- tests/: Unit and integration tests
- .github/: CI/CD workflows and GitHub configurations

## Archive Contents

Each file in this archive is prefixed with "File: <relative-path>" for easy navigation.

================================================================================

EOF
}

# Function to check if a file should be included
should_include_file() {
    local file="$1"
    local include_large="$2"
    local include_logs="$3"

    # Get file size
    local size
    if command -v stat &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            size=$(stat -f%z "$file" 2>/dev/null || echo 0)
        else
            size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        fi
    else
        size=0
    fi

    # Skip large files unless explicitly included
    if [[ "$include_large" == "false" && "$size" -gt 1048576 ]]; then
        return 1
    fi

    # Skip log files unless explicitly included
    if [[ "$include_logs" == "false" && "$file" =~ \.(log|logs)$ ]]; then
        return 1
    fi

    return 0
}

archive_repository() {
    local output_file="$1"
    local include_large="$2"
    local include_logs="$3"

    echo -e "${BLUE}Archiving repository contents...${NC}"

    cd "$REPO_ROOT"

    # Use find to locate all relevant files and process them
    # This is less restrictive than the previous version to ensure we get all important files
    find . -type f \
        ! -path './.git/*' \
        ! -path './scratch/*' \
        ! -path './.venv/*' \
        ! -path './venv/*' \
        ! -path './env/*' \
        ! -path './.terraform/*' \
        ! -path '*/__pycache__/*' \
        ! -path './.pytest_cache/*' \
        ! -path './node_modules/*' \
        ! -path './.vscode/*' \
        ! -path './.idea/*' \
        ! -path './coverage/*' \
        ! -path './htmlcov/*' \
        ! -path './results/*' \
        ! -path './chroma_db/*' \
        ! -name '.terraform.lock.hcl' \
        ! -name '*.tfstate' \
        ! -name '*.tfstate.*' \
        ! -name '*.tfplan' \
        ! -name '*.pyc' \
        ! -name '*.pyo' \
        ! -name '*.pyd' \
        ! -name '.DS_Store' \
        ! -name 'Thumbs.db' \
        ! -name '.env' \
        ! -name '.sqllite3' \
        ! \( -name '.env.*' -a -name 'env.example' \) \
        ! -name 'poetry*.lock' \
        ! -name '*.egg-info' \
        ! -name '*.whl' \
        ! -name '*.tar.gz' \
        ! -name '*.zip' \
        -print0 | \
    while IFS= read -r -d '' file; do
        # Remove leading './' from file path
        local clean_path="${file#./}"

        # Check if file should be included based on size and type
        if ! should_include_file "$file" "$include_large" "$include_logs"; then
            continue
        fi

        # Add file header
        echo "File: $clean_path" >> "$output_file"

        # Check if file is text-based and readable
        if [[ -r "$file" ]]; then
            # Use file command to check if it's text, but be more permissive
            local file_type
            file_type=$(file "$file" 2>/dev/null || echo "unknown")

            # Include text files, scripts, config files, and empty files
            if [[ "$file_type" =~ (text|script|empty|JSON|XML|YAML|HTML|CSS|JavaScript) ]] || \
               [[ "$clean_path" =~ \.(txt|md|py|sh|tf|tfvars|yml|yaml|json|xml|html|css|js|ts|conf|cfg|ini|toml|dockerfile|gitignore|env\.example)$ ]] || \
               [[ "$clean_path" =~ ^(README|LICENSE|CHANGELOG|Dockerfile|Makefile)$ ]]; then

                # Try to read the file content
                if cat "$file" >> "$output_file" 2>/dev/null; then
                    : # Success, file content added
                else
                    echo "[Error: Could not read file content]" >> "$output_file"
                fi
            else
                # For binary or unknown files, just note their existence
                echo "[Binary or non-text file - content not included]" >> "$output_file"
                echo "File type: $file_type" >> "$output_file"
            fi
        else
            echo "[File not readable or does not exist]" >> "$output_file"
        fi

        echo "" >> "$output_file"
    done

    echo -e "${GREEN}Repository archiving completed${NC}"
}

add_footer() {
    local output_file="$1"

    cat >> "$output_file" << EOF

================================================================================

## Archive Generation Complete

This archive was generated on $TIMESTAMP from the Multi-Agent Trading System repository.

Total files processed: $(grep -c "^File: " "$output_file" 2>/dev/null || echo "unknown")

For the most up-to-date version of this code, please refer to the original repository.

End of archive.
EOF
}

generate_file_list() {
    local output_file="$1"
    local list_file="${OUTPUT_DIR}/file-list.txt"

    echo -e "${BLUE}Generating file list...${NC}"

    {
        echo "# File List - Generated on $TIMESTAMP"
        echo "# Repository: Multi-Agent Trading System"
        echo ""
        echo "## Files included in archive:"
        echo ""

        grep "^File: " "$output_file" | sed 's/^File: /- /' | sort

        echo ""
        echo "## File count by type:"
        echo ""

        grep "^File: " "$output_file" | \
        sed 's/^File: //' | \
        sed 's/.*\.//' | \
        sort | uniq -c | sort -nr | \
        awk '{printf "- %s: %d files\n", $2, $1}'
    } > "$list_file"

    echo -e "${GREEN}File list saved to: $list_file${NC}"
}

print_summary() {
    local output_file="$1"

    local file_count
    file_count=$(grep -c "^File: " "$output_file" 2>/dev/null || echo 0)

    local file_size
    if [[ -f "$output_file" ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            file_size=$(stat -f%z "$output_file" 2>/dev/null || echo 0)
        else
            file_size=$(stat -c%s "$output_file" 2>/dev/null || echo 0)
        fi

        # Convert to human readable
        if command -v numfmt &> /dev/null; then
            file_size=$(numfmt --to=iec --suffix=B "$file_size")
        else
            file_size="${file_size} bytes"
        fi
    else
        file_size="unknown"
    fi

    echo
    echo -e "${GREEN}=== Archive Generation Summary ===${NC}"
    echo -e "${BLUE}Output file:${NC} $output_file"
    echo -e "${BLUE}Files archived:${NC} $file_count"
    echo -e "${BLUE}Archive size:${NC} $file_size"
    echo -e "${BLUE}Generated on:${NC} $TIMESTAMP"
    echo
    echo -e "${GREEN}Archive is ready for AI analysis!${NC}"
}

main() {
    local output_file="$OUTPUT_FILE"
    local include_large="false"
    local include_logs="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output)
                output_file="$2"
                shift 2
                ;;
            --include-large)
                include_large="true"
                shift
                ;;
            --include-logs)
                include_logs="true"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                show_usage
                exit 1
                ;;
        esac
    done

    # Ensure we're in the repository root
    if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
        echo -e "${RED}Error: This script must be run from the repository root${NC}"
        exit 1
    fi

    echo -e "${BLUE}=== Multi-Agent Trading System Archive Generator ===${NC}"
    echo -e "${BLUE}Repository root:${NC} $REPO_ROOT"
    echo -e "${BLUE}Output file:${NC} $output_file"
    echo -e "${BLUE}Include large files:${NC} $include_large"
    echo -e "${BLUE}Include log files:${NC} $include_logs"
    echo

    # Create output directory
    create_output_directory

    # Generate archive
    write_header "$output_file"
    archive_repository "$output_file" "$include_large" "$include_logs"
    add_footer "$output_file"

    # Generate file list
    generate_file_list "$output_file"

    # Print summary
    print_summary "$output_file"
}

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
