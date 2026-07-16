#!/usr/bin/env bash
# Snyk security scan for pre-commit — two modes, honest exit codes.
#
#   scripts/snyk_check.sh deps            (default) dependency SCA scan
#   scripts/snyk_check.sh container-base            base-image scan (python:3.12-slim)
#
# Skips cleanly (exit 0) when snyk is missing, unauthenticated, quota-limited, or
# returns an operational error — the "only activate when Snyk is available" contract.
#
# When Snyk IS available and finds vulnerabilities at the threshold:
#   • deps            → BLOCKS the commit (exit 1) so pre-commit reports "Failed".
#                       Override with SNYK_ADVISORY=1 to downgrade to a non-blocking note.
#   • container-base  → prints "REVIEW REQUIRED" and does NOT block (exit 0). The base
#                       image's residual findings (acl/attr) have no Debian fix; the one
#                       fixable OS CVE is cleared by the Dockerfile's `apt-get upgrade -y`,
#                       which is only observable on the BUILT image (scanned in CI, not here).
#
# Install snyk:  brew install snyk   or   npm install -g snyk       Authenticate:  snyk auth

set -uo pipefail

MODE="${1:-deps}"                 # deps | container-base
SEVERITY_THRESHOLD="medium"       # report MEDIUM and above
BASE_IMAGE="python:3.12-slim"     # scanned in container-base mode

note() { echo "[snyk] $*"; }

# ── 0. mode + per-mode blocking policy ──────────────────────────────────────
case "$MODE" in
    deps)           BLOCKING=1 ;;   # a new dependency CVE should fail the commit
    container-base) BLOCKING=0 ;;   # base image unfixable → REVIEW REQUIRED, never block
    *)
        note "unknown mode '$MODE' (expected: deps | container-base)"
        exit 2
        ;;
esac
# Explicit advisory escape hatch: downgrade any mode to non-blocking.
if [ "${SNYK_ADVISORY:-0}" = "1" ]; then
    BLOCKING=0
fi

# ── 1. availability ─────────────────────────────────────────────────────────
if ! command -v snyk >/dev/null 2>&1; then
    note "not installed — skipping (install: brew install snyk  or  npm install -g snyk)"
    exit 0
fi

# ── 2. authentication ───────────────────────────────────────────────────────
# Do NOT use `snyk whoami` as the auth probe: in recent CLI versions (>=1.129x)
# `whoami` became an --experimental-gated command that exits non-zero without
# the flag, which silently turned this hook into a no-op even for authenticated
# users. Detect the stored/exported API token directly instead.
authed=0
if [ -n "${SNYK_TOKEN:-}" ] || [ -n "${SNYK_CFG_API:-}" ] || [ -n "${SNYK_API_TOKEN:-}" ]; then
    authed=1
elif [ -n "$(snyk config get api 2>/dev/null)" ] \
     || [ -n "$(snyk config get INTERNAL_OAUTH_TOKEN_STORAGE 2>/dev/null)" ]; then
    # `snyk auth` uses an OAuth flow by default and stores the token under
    # INTERNAL_OAUTH_TOKEN_STORAGE, NOT the legacy `api` key — so checking only
    # `api` would report an OAuth-authenticated user as unauthenticated (and the
    # whole hook would silently skip). Accept either credential form.
    authed=1
fi

if [ "$authed" -ne 1 ]; then
    note "not authenticated — skipping scan (authenticate once with:  snyk auth)"
    exit 0
fi

# ── 3. scan ─────────────────────────────────────────────────────────────────
if [ "$MODE" = "container-base" ]; then
    note "scanning base image ${BASE_IMAGE} for MEDIUM+ vulnerabilities..."
    # A remote image ref is pulled by Snyk itself — no local Docker daemon needed.
    # If the CLI cannot fetch it, that surfaces as RC=2 and is handled as a skip.
    scan_output="$(snyk container test "${BASE_IMAGE}" --file=Dockerfile \
                     --severity-threshold="${SEVERITY_THRESHOLD}" 2>&1)"
    SNYK_RC=$?
else
    note "scanning dependencies for MEDIUM+ vulnerabilities..."
    scan_output="$(snyk test --severity-threshold="${SEVERITY_THRESHOLD}" 2>&1)"
    SNYK_RC=$?
fi

# A depleted plan / rate limit is a "licensing" condition, not a code problem.
# Match the documented Snyk messages so it surfaces as a skip, not a scary error:
#   "You have used your limit of ... tests", "test limit", "Rate limit hit ...",
#   plus generic quota/entitlement and HTTP 402/429 payment/rate signals.
is_license_failure() {
    printf '%s' "$scan_output" | grep -qiE \
      'used your limit|test limit|rate limit|quota|entitlement|payment required|(^|[^0-9])(402|429)([^0-9]|$)'
}

FINAL_EXIT=0

# Exit codes: 0 = clean, 1 = vulnerabilities found, 2 = error/auth/limit, 3 = no supported projects
case "${SNYK_RC}" in
    0)
        note "✅  no MEDIUM+ vulnerabilities found"
        ;;
    1)
        echo ""
        printf '%s\n' "$scan_output" | sed 's/^/[snyk] /'
        if [ "$BLOCKING" -eq 1 ]; then
            note "❌  MEDIUM+ vulnerabilities found — REMEDIATION NEEDED (commit blocked)"
            note "Fix the finding, or set SNYK_ADVISORY=1 to commit anyway, or record an"
            note "accepted risk via a .snyk policy / Snyk UI Consistent Ignores (see CLAUDE.md)."
            FINAL_EXIT=1
        else
            note "⚠️  MEDIUM+ vulnerabilities found — REVIEW REQUIRED (advisory, not blocking)"
            if [ "$MODE" = "container-base" ]; then
                note "Base-image OS findings may have no Debian fix yet; fixable ones are cleared"
                note "by the Dockerfile's 'apt-get upgrade -y' — verified on the built image in CI."
            else
                note "Advisory mode (SNYK_ADVISORY=1): commit proceeds. Fix before pushing."
            fi
        fi
        ;;
    2)
        if is_license_failure; then
            note "ℹ️  Snyk plan/test-limit reached (licensing) — skipping scan this commit."
            note "    Free tier: 400 open-source tests/month (resets monthly; public repos unlimited)."
            note "    This is not a code problem and does not block the commit."
        else
            note "⚠️  snyk exited with code 2 (error/auth/unreachable) — skipping (run 'snyk ${MODE/container-base/container} test' manually for details)"
        fi
        ;;
    3)
        note "no supported manifests detected — skipping"
        ;;
    *)
        note "⚠️  snyk exited with code ${SNYK_RC} — skipping (run snyk manually for details)"
        ;;
esac

exit "${FINAL_EXIT}"
