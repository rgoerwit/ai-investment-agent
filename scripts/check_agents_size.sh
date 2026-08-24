#!/usr/bin/env bash
# Keep repository guidance well below Codex's combined instruction-chain budget.
set -uo pipefail

file=${1:-AGENTS.md}
max_bytes=${2:-12288}

if [ ! -f "$file" ]; then
    printf '%s is missing; the public Codex contract is required.\n' "$file" >&2
    exit 1
fi

bytes=$(wc -c < "$file" | tr -d '[:space:]')
if [ "$bytes" -gt "$max_bytes" ]; then
    printf '%s is %s bytes; repository policy caps it at %s bytes.\n' \
        "$file" "$bytes" "$max_bytes" >&2
    printf 'Move bounded procedures into repository skills and durable explanation into public documentation.\n' >&2
    exit 1
fi
