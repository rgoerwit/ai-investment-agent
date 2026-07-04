#!/usr/bin/env bash
# Advisory Snyk security scan for pre-commit.
#
# Runs only when snyk is installed and authenticated — silently skips otherwise.
# Findings at HIGH or CRITICAL severity are printed as remediation notes.
# The commit is NEVER blocked (always exits 0); this is a heads-up, not a gate.
#
# Install snyk:  brew install snyk   or   npm install -g snyk
# Authenticate:  snyk auth
#
# Free-tier note: the Snyk Free plan allows 400 open-source (SCA) tests/month
# on private repos; public-repo scans are unlimited. When that allowance is
# exhausted `snyk test` fails with a plan/limit ("licensing") error — this
# script detects that and reports it as a skip, never a blocking failure.

set -uo pipefail

SEVERITY_THRESHOLD="high"   # report HIGH and CRITICAL; skip LOW and MEDIUM

note() { echo "[snyk] $*"; }

# ── 1. availability ────────────────────────────────────────────────────────
if ! command -v snyk >/dev/null 2>&1; then
    note "not installed — skipping (install: brew install snyk  or  npm install -g snyk)"
    exit 0
fi

# ── 2. authentication ──────────────────────────────────────────────────────
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
    note "not authenticated — skipping advisory scan."
    note "authenticate once with:  snyk auth"
    exit 0
fi

# ── 3. scan ────────────────────────────────────────────────────────────────
note "scanning for HIGH/CRITICAL vulnerabilities..."
scan_output="$(snyk test --severity-threshold="${SEVERITY_THRESHOLD}" 2>&1)"
SNYK_RC=$?

# A depleted plan / rate limit is a "licensing" condition, not a code problem.
# Match the documented Snyk messages so it surfaces as a skip, not a scary error:
#   "You have used your limit of ... tests", "test limit", "Rate limit hit ...",
#   plus generic quota/entitlement and HTTP 402/429 payment/rate signals.
is_license_failure() {
    printf '%s' "$scan_output" | grep -qiE \
      'used your limit|test limit|rate limit|quota|entitlement|payment required|(^|[^0-9])(402|429)([^0-9]|$)'
}

# Exit codes: 0 = clean, 1 = vulnerabilities found, 2 = error/auth/limit, 3 = no supported projects
case "${SNYK_RC}" in
    0)
        note "✅  no HIGH or CRITICAL vulnerabilities found"
        ;;
    1)
        echo ""
        note "⚠️  HIGH/CRITICAL vulnerabilities found — REMEDIATION NEEDED"
        printf '%s\n' "$scan_output" | sed 's/^/[snyk] /'
        note "Commit proceeded (advisory). Fix before pushing, or record accepted risks"
        note "via Snyk UI Consistent Ignores (preferred here — see CLAUDE.md) or a .snyk policy."
        ;;
    2)
        if is_license_failure; then
            note "ℹ️  Snyk plan/test-limit reached (licensing) — skipping advisory scan this commit."
            note "    Free tier: 400 open-source tests/month (resets monthly; public repos unlimited)."
            note "    This is not a code problem and does not block the commit."
        else
            note "⚠️  snyk exited with code 2 (error/auth) — run 'snyk test' manually for details"
        fi
        ;;
    3)
        note "no supported manifests detected — skipping"
        ;;
    *)
        note "⚠️  snyk exited with code ${SNYK_RC} — run 'snyk test' manually for details"
        ;;
esac

exit 0  # advisory: never block the commit
