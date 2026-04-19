#!/bin/bash
# Local launcher for the IBKR dashboard.
#
# Starts the Flask app in the foreground and, by default, the refresh worker in
# the background. Uses module entrypoints under Poetry so it works even when
# Poetry script shims have not yet been installed into .venv/bin.
#
# Usage:
#   ./scripts/run_ibkr_dashboard.sh
#   ./scripts/run_ibkr_dashboard.sh --no-worker
#   ./scripts/run_ibkr_dashboard.sh -- --host 127.0.0.1 --port 5051
#   ./scripts/run_ibkr_dashboard.sh -- --account-id U20958465 --watchlist-name "default watchlist"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

RUN_WORKER=true
APP_ARGS=()
DASH_HOST="127.0.0.1"
DASH_PORT="5050"

resolve_python_cmd() {
    if [[ -n "${INVESTMENT_AGENT_CONTAINER:-}" ]] || [[ -f "/.dockerenv" ]] || [[ -f "/run/.containerenv" ]]; then
        PYTHON_CMD=(python)
        return
    fi

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        PYTHON_CMD=(python)
        return
    fi

    if command -v poetry >/dev/null 2>&1; then
        PYTHON_CMD=(poetry run python)
        return
    fi

    echo "Poetry is not installed and no active virtual environment was detected." >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./scripts/run_ibkr_dashboard.sh [OPTIONS] [-- APP_ARGS]

Options:
  --no-worker   Start only the Flask app
  -h, --help    Show this help message

Anything after `--` is passed to `src.web.ibkr_dashboard.app`.

Examples:
  ./scripts/run_ibkr_dashboard.sh
  ./scripts/run_ibkr_dashboard.sh --no-worker
  ./scripts/run_ibkr_dashboard.sh -- --host 127.0.0.1 --port 5051
  ./scripts/run_ibkr_dashboard.sh -- --account-id U20958465 --watchlist-name "default watchlist"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-worker)
            RUN_WORKER=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --host)
            DASH_HOST="$2"
            APP_ARGS+=("$1" "$2")
            shift 2
            ;;
        --port)
            DASH_PORT="$2"
            APP_ARGS+=("$1" "$2")
            shift 2
            ;;
        --)
            shift
            APP_ARGS=("$@")
            # Scan remaining args for --host/--port.
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --host) DASH_HOST="${2:-}"; shift 2 ;;
                    --port) DASH_PORT="${2:-}"; shift 2 ;;
                    *) shift ;;
                esac
            done
            break
            ;;
        *)
            APP_ARGS+=("$1")
            shift
            ;;
    esac
done

resolve_python_cmd

WORKER_PID=""

cleanup() {
    local exit_code=$?
    if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
        kill "${WORKER_PID}" 2>/dev/null || true
        wait "${WORKER_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

if ${RUN_WORKER}; then
    "${PYTHON_CMD[@]}" -m src.web.ibkr_dashboard.worker &
    WORKER_PID=$!
    echo "[INFO] Started dashboard worker (PID ${WORKER_PID})"
fi

echo "[INFO] Starting dashboard app at http://${DASH_HOST}:${DASH_PORT}/"
exec "${PYTHON_CMD[@]}" -m src.web.ibkr_dashboard.app ${APP_ARGS[@]+"${APP_ARGS[@]}"}
