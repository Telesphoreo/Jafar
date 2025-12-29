#!/bin/bash
# Twitter Sentiment Analysis - Background Runner
#
# Usage:
#   ./run.sh start    - Start the pipeline in background
#   ./run.sh stop     - Stop the running pipeline
#   ./run.sh status   - Check if pipeline is running
#   ./run.sh logs     - Tail the log file (live)
#   ./run.sh logs-all - View entire log file

# Strict mode: exit on error, undefined var, or pipe failure
set -euo pipefail

# Immutable configuration - prevent accidental modification
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RUN_DIR="${SCRIPT_DIR}/.run"
readonly PID_FILE="${RUN_DIR}/pipeline.pid"
readonly SHUTDOWN_TIMEOUT=10

# Ensure .run directory exists
ensure_run_dir() {
    if [[ ! -d "$RUN_DIR" ]]; then
        mkdir -p "$RUN_DIR" || {
            echo "Error: Cannot create runtime directory: $RUN_DIR" >&2
            exit 1
        }
    fi
}

# Get the most recent log file
get_latest_log_file() {
    # Find the most recent pipeline_*.log file
    local latest_log
    latest_log="$(find "$RUN_DIR" -maxdepth 1 -name 'pipeline_*.log' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
    echo "$latest_log"
}

# Generate new timestamped log filename
generate_log_filename() {
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    echo "${RUN_DIR}/pipeline_${timestamp}.log"
}

# Validate that a string is a valid PID (numeric, reasonable range)
is_valid_pid() {
    local pid="$1"
    # Must be numeric and positive
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    # Sanity check: PIDs are typically under 4194304 on Linux
    if [[ "$pid" -le 0 || "$pid" -gt 4194304 ]]; then
        return 1
    fi
    return 0
}

# Safely read PID from file with validation
read_pid_file() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi

    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')" || return 1

    if ! is_valid_pid "$pid"; then
        echo "Warning: Invalid PID in $PID_FILE, cleaning up" >&2
        safe_remove_pid_file
        return 1
    fi

    echo "$pid"
}

# Remove PID file only if it's actually our PID file
safe_remove_pid_file() {
    # Verify the file exists and is in our expected location
    if [[ -f "$PID_FILE" ]]; then
        local real_path
        real_path="$(realpath "$PID_FILE" 2>/dev/null)" || real_path="$PID_FILE"
        local expected_path
        expected_path="$(realpath "${SCRIPT_DIR}/pipeline.pid" 2>/dev/null)" || expected_path="${SCRIPT_DIR}/pipeline.pid"

        if [[ "$real_path" == "$expected_path" ]]; then
            rm -f -- "$PID_FILE"
        else
            echo "Error: PID file path mismatch, refusing to delete" >&2
            return 1
        fi
    fi
}

# Check if a process is running
is_process_running() {
    local pid="$1"
    if ! is_valid_pid "$pid"; then
        return 1
    fi
    ps -p "$pid" > /dev/null 2>&1
}

start() {
    # Ensure runtime directory exists
    ensure_run_dir

    local existing_pid
    if existing_pid="$(read_pid_file 2>/dev/null)"; then
        if is_process_running "$existing_pid"; then
            echo "Pipeline is already running (PID: $existing_pid)"
            echo "Use './run.sh logs' to watch progress"
            exit 1
        else
            echo "Cleaning up stale PID file..."
            safe_remove_pid_file
        fi
    fi

    # Generate new timestamped log file
    local log_file
    log_file="$(generate_log_filename)"

    echo "Starting pipeline in background..."
    echo "Log file: $log_file"

    # Verify we're in the right directory before running
    if [[ ! -d "$SCRIPT_DIR" ]]; then
        echo "Error: Script directory does not exist: $SCRIPT_DIR" >&2
        exit 1
    fi

    cd "$SCRIPT_DIR" || { echo "Error: Cannot change to $SCRIPT_DIR" >&2; exit 1; }

    # Run with nohup, redirect all output to log
    nohup uv run python -m src.main >> "$log_file" 2>&1 &
    local new_pid=$!

    # Verify the process actually started
    sleep 0.5
    if ! is_process_running "$new_pid"; then
        echo "Error: Pipeline failed to start. Check $log_file for details." >&2
        exit 1
    fi

    # Write PID atomically (write to temp file, then move)
    local temp_pid_file
    temp_pid_file="${PID_FILE}.tmp.$$"
    echo "$new_pid" > "$temp_pid_file"
    mv -f -- "$temp_pid_file" "$PID_FILE"

    echo "Pipeline started (PID: $new_pid)"
    echo ""
    echo "Commands:"
    echo "  ./run.sh logs     - Watch live progress"
    echo "  ./run.sh status   - Check if still running"
    echo "  ./run.sh stop     - Stop the pipeline"
}

stop() {
    local pid
    if ! pid="$(read_pid_file 2>/dev/null)"; then
        echo "No pipeline running (no valid PID file)"
        return 0
    fi

    if ! is_process_running "$pid"; then
        echo "Pipeline not running (stale PID file)"
        safe_remove_pid_file
        return 0
    fi

    echo "Stopping pipeline (PID: $pid)..."

    # Graceful shutdown: SIGTERM first
    kill -TERM "$pid" 2>/dev/null || true

    # Wait for graceful shutdown with timeout
    local waited=0
    while is_process_running "$pid" && [[ $waited -lt $SHUTDOWN_TIMEOUT ]]; do
        sleep 1
        ((waited++))
        echo "Waiting for shutdown... ($waited/${SHUTDOWN_TIMEOUT}s)"
    done

    # If still running, force kill
    if is_process_running "$pid"; then
        echo "Process did not exit gracefully, sending SIGKILL..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    if is_process_running "$pid"; then
        echo "Warning: Process $pid may still be running" >&2
    else
        echo "Pipeline stopped"
    fi

    safe_remove_pid_file
}

status() {
    local pid
    if ! pid="$(read_pid_file 2>/dev/null)"; then
        echo "Pipeline: NOT RUNNING"
        return 0
    fi

    local log_file
    log_file="$(get_latest_log_file)"

    if is_process_running "$pid"; then
        echo "Pipeline: RUNNING (PID: $pid)"
        echo ""
        # Show last few lines of log
        if [[ -n "$log_file" && -f "$log_file" ]]; then
            echo "Recent activity from: $(basename "$log_file")"
            tail -5 "$log_file" || true
        fi
    else
        echo "Pipeline: NOT RUNNING (finished or crashed)"
        safe_remove_pid_file
        if [[ -n "$log_file" && -f "$log_file" ]]; then
            echo ""
            echo "Last log entries from: $(basename "$log_file")"
            tail -10 "$log_file" || true
        fi
    fi
}

logs() {
    local log_file
    log_file="$(get_latest_log_file)"

    if [[ -z "$log_file" || ! -f "$log_file" ]]; then
        echo "No log file yet. Start the pipeline first."
        exit 1
    fi

    echo "Watching $(basename "$log_file") (Ctrl+C to stop watching)"
    echo "-------------------------------------------"
    tail -f "$log_file"
}

logs_all() {
    local log_file
    log_file="$(get_latest_log_file)"

    if [[ -z "$log_file" || ! -f "$log_file" ]]; then
        echo "No log file yet."
        exit 1
    fi

    less "$log_file"
}

# Main entry point
main() {
    local command="${1:-}"

    case "$command" in
        start)
            start
            ;;
        stop)
            stop
            ;;
        status)
            status
            ;;
        logs)
            logs
            ;;
        logs-all)
            logs_all
            ;;
        *)
            echo "Jafar - Twitter Sentiment Analysis Pipeline"
            echo ""
            echo "Usage: $0 {start|stop|status|logs|logs-all}"
            echo ""
            echo "Commands:"
            echo "  start     - Start the pipeline in background"
            echo "  stop      - Stop the running pipeline"
            echo "  status    - Check if pipeline is running + recent logs"
            echo "  logs      - Watch live progress (tail -f)"
            echo "  logs-all  - View entire log file"
            exit 1
            ;;
    esac
}

main "$@"
