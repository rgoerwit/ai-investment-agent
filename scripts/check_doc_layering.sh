#!/usr/bin/env bash
# Fail a commit when HUMAN-FACING DOCUMENTATION cites AGENT METADATA.
#
# Why this exists: agent instruction and configuration artifacts -- the root
# instruction file, this repo's agent directory, and the equivalents belonging to other
# coding agents -- are instructions to a tool. Committing them is ordinary; versioned
# tool configuration is fine. What is not fine is documentation citing them. A reader
# following such a pointer is sent into machinery that is not addressed to them, and
# often into a file a clone does not even contain.
#
# This is a LAYERING test, not an OBTAINABILITY test, and the two answers diverge.
# check_tracked_deps.sh asks "can the reader get this file?" -- which stops objecting
# to `.claude/rules/x.md` the moment that directory becomes tracked. This check asks
# "should a document cite it at all?", and the answer stays no either way. That is why
# the two hooks are separate rather than merged.
#
# Scope is every tracked Markdown file EXCEPT the agent-owned directories themselves,
# which are configured in .pre-commit-config.yaml. Files inside the agent layer cite
# each other freely -- that is what the layer is for.
#
# EXEMPTION, AND ITS HONEST LIMIT:
#
#     doc-layering-ok(<token>)
#
# exempts that token EVERYWHERE IN THAT ONE FILE -- it is file-and-token scoped, not
# occurrence scoped. There is deliberately no way to permit one mention of `CLAUDE.md`
# in a file while still catching the next one. So a marker is the wrong tool for
# keeping a single historical reference: reword the reference instead. No file in this
# repository currently needs a marker, and that is the intended steady state.
#
# Usage: check_doc_layering.sh FILE [FILE...]        (pre-commit passes staged files)
set -uo pipefail

# Instruction files belonging to this or any other coding agent.
AGENT_FILE_RE='CLAUDE[-.A-Za-z0-9]*\.md|AGENTS[-.A-Za-z0-9]*\.md|GEMINI[-.A-Za-z0-9]*\.md'

# Agent-owned directories. Two things this must get right, both learned the hard way:
#
#  1. The character class before the dot must ALLOW '/', or a nested reference such as
#     `foo/.claude/rules/x.md` slips through while the bare `.claude/rules/x.md` is
#     caught. It must still reject `a.claude/nope`, where the dot starts no segment.
#  2. The vendor list must cover the ecosystem, not just the agents this repo happens
#     to use. An earlier version listed claude/agents/codex and silently permitted
#     `.gemini/`, `.cursor/`, `.windsurf/` and `.aider*` -- and `.gemini` is gitignored
#     right here, so that was a live hole, not a hypothetical one. Unlike filesystem
#     roots, the set of coding agents is bounded and nameable, so enumerate it fully.
#  3. The trailing context must be any non-name character, not just `/` or `.`. An
#     earlier version required a separator and so passed the bare prose reference
#     "Read .gemini before changing this rule" -- naming ignored material with no
#     slash, which the obtainability guard cannot see either.
#
# ONE list, used both to match references and to decide which files are themselves
# agent-layer (below). The pre-commit `exclude:` and the Makefile mirror it; if they
# drift, this file is the authority and the self-skip here still holds.
AGENT_DIRS='claude|agents|codex|gemini|cursor|windsurf|aider|cline|continue|roo|junie|amp'
AGENT_DIR_RE="(^|[^A-Za-z0-9_-])\.(${AGENT_DIRS})([^A-Za-z0-9_-]|\$)"

# Flat rule files used by agents that predate the directory convention. `.aiderignore`
# is here rather than in the list above because the trailing-context rule deliberately
# refuses to match a longer word, and it is a distinct filename rather than a suffix.
AGENT_MISC_RE='\.cursorrules|\.clinerules|\.windsurfrules|\.roomodes|\.aiderignore|copilot-instructions'

status=0

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)

is_repository_root_agents_file() {
    local file="$1"
    local file_dir
    file_dir=$(CDPATH='' cd -- "$(dirname -- "$file")" 2>/dev/null && pwd -P) || return 1
    [ "$file_dir/$(basename -- "$file")" = "$repo_root/AGENTS.md" ]
}

for file in "$@"; do
    [ -f "$file" ] || continue

    # A file that IS agent layer may cite agent metadata freely. Derived from the one
    # list above, so widening the vendor set cannot leave this behind.
    if printf '%s' "$file" | grep -qE "(^|/)\.(${AGENT_DIRS})/" \
        || is_repository_root_agents_file "$file"; then
        continue
    fi

    # Tokens this file exempts by name, one marker each.
    exempt=$(grep -oE 'doc-layering-ok\([^)]+\)' "$file" 2>/dev/null \
        | sed 's/^doc-layering-ok(//; s/)$//' | sort -u)

    hits=$(grep -ohE "$AGENT_FILE_RE|$AGENT_DIR_RE|$AGENT_MISC_RE" "$file" 2>/dev/null \
        | sed -e 's#^[^A-Za-z0-9_.]##' -e 's#[^A-Za-z0-9_./-]*$##' | sort -u)

    [ -n "$hits" ] || continue

    while IFS= read -r hit; do
        [ -n "$hit" ] || continue
        if printf '%s\n' "$exempt" | grep -qxF "$hit"; then
            continue
        fi
        printf '%s: documentation cites agent metadata: %s\n' "$file" "$hit" >&2
        status=1
    done <<< "$hits"
done

if [ "$status" -ne 0 ]; then
    printf '\n' >&2
    printf 'Agent metadata is not human-facing documentation. A tracked document must\n' >&2
    printf 'not cite an agent instruction or configuration artifact -- not the root\n' >&2
    printf 'instruction file, not this agent directory, not another agent-s equivalent.\n' >&2
    printf '\n' >&2
    printf 'Fix by stating authority instead of location: "the implementation and its\n' >&2
    printf 'tests are authoritative". If the citation is genuinely warranted, note that\n' >&2
    printf 'doc-layering-ok(<token>) exempts that token throughout the whole file, so it\n' >&2
    printf 'will also silence the next, unrelated occurrence -- prefer rewording.\n' >&2
fi

exit "$status"
