#!/usr/bin/env bash
# Fail a commit when a TRACKED file refers to an UNOBTAINABLE one.
#
# Why this exists: anything committed is read by someone who cloned this repo and has
# none of the gitignored material. A tracked skill, rule or doc that points at private
# notes, personal source material, or an operator-only archive is broken on arrival for
# every reader but the author -- and broken silently, because it looks fine in the diff.
#
# The line is not "ignored" but OBTAINABLE:
#   unobtainable - an ignored INPUT the reader can never produce
#                  (private notes, personal writing samples, a local-only archive)
#   obtainable   - an ignored path the reader CREATES by using the project
#                  (analysis output, capture bundles, runtime state), or one with a
#                  tracked *.example.* template beside it
#
# A reference to an unobtainable path must be exempted EXPLICITLY AND INDIVIDUALLY:
#
#     tracked-deps-ok(path/to/the/private/thing)
#
# One marker exempts one path. An earlier version skipped a whole paragraph on a bare
# marker, which let a single caveat silently suppress checks for unrelated paths.
#
# Usage: check_tracked_deps.sh FILE [FILE...]        (pre-commit passes staged files)
set -uo pipefail

# Ignored paths a reader produces by RUNNING the project.
OBTAINABLE_RE='^(results|runtime|outputs|reports|logs|scratch|\.?venv)/'\
'|^evals/(captures|prompt_checks)/'\
'|^\.[A-Za-z0-9_.-]*latest_analyses_index\.(json|lock)$'

# Path-shaped tokens: at least one slash.
PATH_RE='/?[A-Za-z0-9_.-]+/[A-Za-z0-9_./*-]+'

# Bare filenames that could name an ignored file. Without this, a reference such as
# "the archive is FOO-2026-08.md" is invisible to the scan: it has no slash.
BARE_RE='(^|[^A-Za-z0-9_./-])[A-Za-z0-9_.-]+\.(md|json|txt|yaml|yml|toml|env|py|sh)\b'

# Bare DOTFILE names, which have no recognised extension and so match neither pattern
# above. This closes a real hole: the documentation-layering guard deliberately skips
# files that are themselves agent layer, so a tracked rule saying "read .gemini before
# changing this" pointed at ignored material with nothing to catch it.
#
# Capture EVERY dotted segment, not just the first. An earlier version stopped at the
# first dot, so `.env.old` was extracted as `.env` and then waved through because
# `.env.example` is tracked -- one file's template laundering references to siblings
# that have none. Whole-name capture also keeps the git-pattern classification honest:
# `.env.old` is matched by a literal, `.key` by the `*.key` extension glob.
DOTFILE_RE='(^|[^A-Za-z0-9_./-])\.[a-z][a-z0-9_-]+(\.[a-z0-9_-]+)*'

# An absolute path is unobtainable BY CONSTRUCTION, not by gitignore -- no reader has
# the author's machine. `git check-ignore` cannot see that, and the candidate loop
# below deliberately skips absolute paths as illustrations, so this needs its own pass.
#
# DEFAULT-DENY, deliberately. An earlier version listed the machine-local prefixes to
# reject (/Users, /home, /private, ...) and every review round found another one it had
# missed -- /Volumes, /mnt, /media. Enumerating the bad set never terminates. So every
# absolute path is machine-local unless its first segment is a standard filesystem
# location that means the same thing on any machine.
ABS_PATH_RE='(^|[^A-Za-z0-9_`:./\\~-])(/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.*-]+)+)'

# Absolute-looking strings that are NOT references to the author's disk. Keep this
# list short and justify each entry; it is the whole exception surface.
#   etc usr bin sbin dev proc sys lib -- read-only system locations that name the
#                                        same thing on any POSIX host
#   api                               -- a URL route namespace, not a filesystem path
#
# Deliberately NOT allowlisted, despite being "standard": /tmp, /var and /opt are
# WRITABLE and machine-varying -- /var/folders is a macOS scratch dir, /opt/homebrew
# exists only on some Macs, and telling a reader to run a script out of /tmp is both
# unobtainable from a clone and a place no one should source project tooling from.
ABS_ALLOW_RE='^/(etc|usr|bin|sbin|dev|proc|sys|lib|api)(/|$)'

# Home-relative and Windows forms, which the absolute matcher above cannot see.
HOME_LOCAL_RE='(^|[^A-Za-z0-9_`])(~/[A-Za-z.]|\$\{?HOME|[A-Za-z]:\\)'

# An ignored file with a TRACKED *.example.* sibling is obtainable: the template ships
# and the reader creates the instance from it. Derived, not listed, so it follows the
# convention already used in this repo.
has_example_sibling() {
    local p="$1"
    [ -f "${p}.example" ] && return 0
    case "$p" in
        *.*) [ -f "${p%.*}.example.${p##*.}" ] && return 0 ;;
    esac
    return 1
}

# `*.key` in .gitignore means "any file ending in .key", NOT "a file named .key". A
# token matched only by such an extension glob is a shape, not a reference: in this
# repo the sole `.key` occurrence is a jq field accessor inside a shell pipeline.
# Ask git which pattern matched rather than hardcoding an exception list.
matched_by_extension_glob() {
    git check-ignore -v -- "$1" 2>/dev/null \
        | awk -F'\t' '{print $1}' | grep -qE ':[0-9]+:!?\*\.'
}

status=0

for file in "$@"; do
    [ -f "$file" ] || continue

    # Paths this file exempts by name, one marker each.
    exempt=$(grep -oE 'tracked-deps-ok\([^)]+\)' "$file" 2>/dev/null \
        | sed 's/^tracked-deps-ok(//; s/)$//' | sort -u)

    # Machine-local paths. Braces on every expansion: `$VAR[...]` parses as an array
    # subscript, which shellcheck flags as SC1087 and which silently mangles the regex.
    while IFS= read -r hit; do
        [ -n "$hit" ] || continue
        printf '%s\n' "$hit" | grep -qE "$ABS_ALLOW_RE" && continue
        printf '%s\n' "$exempt" | grep -qxF "$hit" && continue
        printf '%s: refers to a machine-local path: %s\n' "$file" "$hit" >&2
        status=1
    done <<< "$(
        {
            grep -ohE "${ABS_PATH_RE}" "$file" 2>/dev/null \
                | grep -oE '/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.*-]+)+'
            grep -ohE "${HOME_LOCAL_RE}[A-Za-z0-9_./\\-]*" "$file" 2>/dev/null \
                | sed 's#^[^A-Za-z0-9_~$]##'
        } | sed 's#[.,;:)`\"]*$##' | sort -u
    )"

    candidates=$(
        {
            grep -oE "$PATH_RE" "$file" 2>/dev/null
            grep -oE "$BARE_RE" "$file" 2>/dev/null | grep -oE '[A-Za-z0-9_.-]+\.[a-z]+$'
            grep -oE "$DOTFILE_RE" "$file" 2>/dev/null \
                | grep -oE '\.[a-z][a-z0-9_-]+(\.[a-z0-9_-]+)*$'
        } | sed -e 's#^\./##' -e 's#[.,;:)`"]*$##' | sort -u
    )

    [ -n "$candidates" ] || continue

    while IFS= read -r path; do
        [ -n "$path" ] || continue
        case "$path" in *'*'*) continue ;; esac   # globs are patterns, not references
        case "$path" in /*) continue ;; esac      # absolute paths are illustrations

        if matched_by_extension_glob "$path"; then
            continue
        fi
        if printf '%s' "$path" | grep -qE "$OBTAINABLE_RE"; then
            continue
        fi
        if has_example_sibling "$path"; then
            continue
        fi
        if printf '%s\n' "$exempt" | grep -qxF "$path"; then
            continue
        fi
        if git check-ignore -q -- "$path" 2>/dev/null; then
            printf '%s: refers to unobtainable path: %s\n' "$file" "$path" >&2
            status=1
        fi
    done <<< "$candidates"
done

if [ "$status" -ne 0 ]; then
    printf '\n' >&2
    printf 'A tracked file must not refer to a gitignored input a reader cannot obtain.\n' >&2
    printf 'Fix by removing the reference, describing it generically, pointing at a\n' >&2
    printf 'tracked *.example.* template, or exempting that one path explicitly with\n' >&2
    printf 'the marker tracked-deps-ok(<path>) alongside a caveat.\n' >&2
fi

exit "$status"
