#!/bin/bash

run_tracked_child() {
    local logfile="$1"
    shift

    if [[ -z "$logfile" ]]; then
        "$@"
        return $?
    fi

    local timeout="${PIPELINE_TICKER_TIMEOUT_SECONDS:-900}"
    local poll_seconds="${PIPELINE_TICKER_WATCHDOG_POLL_SECONDS:-5}"
    local dump_grace_seconds="${PIPELINE_TICKER_DUMP_GRACE_SECONDS:-5}"
    local term_grace_seconds="${PIPELINE_TICKER_TERM_GRACE_SECONDS:-10}"
    local timeout_record_file="${PIPELINE_TIMEOUT_RECORD_FILE:-}"
    local child_pid elapsed status
    local cmd=("$@")

    : > "$logfile"
    {
        printf '[pipeline_child_start] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '[pipeline_child_command] %s\n' "$*"
    } >> "$logfile"

    "${cmd[@]}" >> "$logfile" 2>&1 &
    child_pid=$!
    elapsed=0

    while kill -0 "$child_pid" 2>/dev/null; do
        if [[ "$elapsed" -ge "$timeout" ]]; then
            {
                printf '[pipeline_child_timeout] exceeded %ss; requesting pending-task dump\n' "$timeout"
                printf '[pipeline_child_signal] SIGUSR1 pid=%s\n' "$child_pid"
            } >> "$logfile"
            if [[ -n "$timeout_record_file" ]]; then
                printf '{"timestamp":"%s","status":"timeout","timeout_seconds":%s,"elapsed_seconds":%s,"pid":%s,"logfile":"%s"}\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    "$timeout" \
                    "$elapsed" \
                    "$child_pid" \
                    "$logfile" >> "$timeout_record_file"
            fi
            kill -USR1 "$child_pid" 2>/dev/null || true
            sleep "$dump_grace_seconds"

            if kill -0 "$child_pid" 2>/dev/null; then
                printf '[pipeline_child_signal] SIGTERM pid=%s\n' "$child_pid" >> "$logfile"
                kill -TERM "$child_pid" 2>/dev/null || true
                sleep "$term_grace_seconds"
            fi

            if kill -0 "$child_pid" 2>/dev/null; then
                printf '[pipeline_child_signal] SIGKILL pid=%s\n' "$child_pid" >> "$logfile"
                kill -KILL "$child_pid" 2>/dev/null || true
                sleep 1 # Wait for OS reaping
            fi

            wait "$child_pid" 2>/dev/null || true
            return 124
        fi
        sleep "$poll_seconds"
        elapsed=$((elapsed + poll_seconds))
    done

    wait "$child_pid"
    status=$?
    printf '[pipeline_child_exit] status=%s elapsed=%ss\n' "$status" "$elapsed" >> "$logfile"
    return "$status"
}

exit_if_interrupted_status() {
    local status="$1"
    if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
        warn "Pipeline interrupted; stopping without continuing to later stages"
        exit "$status"
    fi
}
