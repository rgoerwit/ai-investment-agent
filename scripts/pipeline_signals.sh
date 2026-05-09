#!/bin/bash

run_tracked_child() {
    local logfile="$1"
    shift

    if [[ -n "$logfile" ]]; then
        "$@" 2>"$logfile"
    else
        "$@"
    fi
}

exit_if_interrupted_status() {
    local status="$1"
    if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
        warn "Pipeline interrupted; stopping without continuing to later stages"
        exit "$status"
    fi
}
