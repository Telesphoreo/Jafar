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

# Get the most recent log file (by filename timestamp, not mtime)
get_latest_log_file() {
    # Sort by filename descending - filenames contain timestamps like pipeline_20260314_045800.log
    local latest_log
    latest_log="$(ls -1 "$RUN_DIR"/pipeline_*.log 2>/dev/null | sort -r | head -1)"
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
        expected_path="$(realpath "${RUN_DIR}/pipeline.pid" 2>/dev/null)" || expected_path="${RUN_DIR}/pipeline.pid"

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

    # Run with nohup, capture the actual python process PID
    # We use a subshell that writes its own PID, then execs the actual command
    nohup bash -c 'echo $$ > "'"$PID_FILE"'" && exec uv run python -m src.main' >> "$log_file" 2>&1 &

    # Give it a moment to write the PID file
    sleep 1

    # Read the actual PID that was written
    local new_pid
    if ! new_pid="$(read_pid_file 2>/dev/null)"; then
        echo "Error: Pipeline failed to start. Check $log_file for details." >&2
        exit 1
    fi

    # Verify the process is actually running
    if ! is_process_running "$new_pid"; then
        echo "Error: Pipeline exited immediately. Check $log_file for details." >&2
        safe_remove_pid_file
        exit 1
    fi

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

    # Send SIGTERM to the process
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

kill_orphan() {
    echo "Looking for orphaned pipeline processes..."

    # Find python processes running src.main
    local pids
    pids="$(pgrep -f 'python.*src\.main' 2>/dev/null || true)"

    if [[ -z "$pids" ]]; then
        echo "No orphaned pipeline processes found."
        echo ""
        echo "If the DB lock is still held, the connection may be stale."
        echo "PostgreSQL will release it when the connection times out,"
        echo "or you can restart PostgreSQL to force-release all locks."
        return 0
    fi

    echo "Found pipeline process(es):"
    echo ""
    # Show details for each PID
    for pid in $pids; do
        echo "  PID $pid:"
        ps -p "$pid" -o pid,ppid,user,etime,args --no-headers 2>/dev/null | sed 's/^/    /'
    done
    echo ""

    read -p "Kill these processes? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for pid in $pids; do
            echo "Killing PID $pid..."
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2

        # Check if any survived
        local survivors
        survivors="$(pgrep -f 'python.*src\.main' 2>/dev/null || true)"
        if [[ -n "$survivors" ]]; then
            echo "Some processes didn't die. Force killing..."
            for pid in $survivors; do
                kill -KILL "$pid" 2>/dev/null || true
            done
        fi

        # Clean up stale PID file if it exists
        safe_remove_pid_file

        echo "Done. You can now run './run.sh start'"
    else
        echo "Aborted."
    fi
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
        kill-orphan)
            kill_orphan
            ;;
        *)
            echo "Jafar - Twitter Sentiment Analysis Pipeline"
            echo ""
            echo "Usage: $0 {start|stop|status|logs|logs-all|kill-orphan}"
            echo ""
            echo "Commands:"
            echo "  start       - Start the pipeline in background"
            echo "  stop        - Stop the running pipeline"
            echo "  status      - Check if pipeline is running + recent logs"
            echo "  logs        - Watch live progress (tail -f)"
            echo "  logs-all    - View entire log file"
            echo "  kill-orphan - Find and kill orphaned pipeline processes"
            exit 1
            ;;
    esac
}

main "$@"
