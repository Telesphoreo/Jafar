#!/bin/bash
# Twitter Sentiment Analysis - Background Runner
#
# Usage:
#   ./run.sh start    - Start the pipeline in background
#   ./run.sh stop     - Stop the running pipeline
#   ./run.sh status   - Check if pipeline is running
#   ./run.sh logs     - Tail the log file (live)
#   ./run.sh logs-all - View entire log file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/pipeline.log"
PID_FILE="$SCRIPT_DIR/pipeline.pid"

start() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Pipeline is already running (PID: $PID)"
            echo "Use './run.sh logs' to watch progress"
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting pipeline in background..."
    echo "Log file: $LOG_FILE"

    # Run with nohup, redirect all output to log
    cd "$SCRIPT_DIR"
    nohup uv run python -m src.main > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"

    echo "Pipeline started (PID: $PID)"
    echo ""
    echo "Commands:"
    echo "  ./run.sh logs     - Watch live progress"
    echo "  ./run.sh status   - Check if still running"
    echo "  ./run.sh stop     - Stop the pipeline"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No pipeline running (no PID file)"
        exit 0
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping pipeline (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Pipeline stopped"
    else
        echo "Pipeline not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
}

status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Pipeline: NOT RUNNING"
        return
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Pipeline: RUNNING (PID: $PID)"
        echo ""
        # Show last few lines of log
        if [ -f "$LOG_FILE" ]; then
            echo "Recent activity:"
            tail -5 "$LOG_FILE"
        fi
    else
        echo "Pipeline: NOT RUNNING (finished or crashed)"
        rm -f "$PID_FILE"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Last log entries:"
            tail -10 "$LOG_FILE"
        fi
    fi
}

logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "No log file yet. Start the pipeline first."
        exit 1
    fi

    echo "Watching $LOG_FILE (Ctrl+C to stop watching)"
    echo "-------------------------------------------"
    tail -f "$LOG_FILE"
}

logs_all() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "No log file yet."
        exit 1
    fi

    less "$LOG_FILE"
}

case "${1:-}" in
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
        echo "Twitter Sentiment Analysis - Background Runner"
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
