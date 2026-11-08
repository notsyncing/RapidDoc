#!/bin/bash
set -euo pipefail

# 加载环境变量文件（如果存在）
if [ -f "/app/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' /app/.env | xargs)
fi

# 设置默认值
export PYTHONPATH=/app
export API_PORT=${API_PORT:-8888}
export STARTUP_WAIT_TIME=${STARTUP_WAIT_TIME:-15}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "API Port: $API_PORT"
echo "Log Level: $LOG_LEVEL"
echo "======================================"

# 启动主应用
echo "Starting RapidDoc Web API on port $API_PORT..."
echo "API Documentation: http://localhost:$API_PORT/docs"
echo "Health Check: http://localhost:$API_PORT/health"
echo "======================================"

# 启动 API 服务
cd /app

# An unusable SYCL/Level-Zero device leaves the process no way to recover:
# a lost device takes the whole Unified Runtime context with it, and a failed
# device allocation means every later call allocates against the same
# exhausted budget (the executor has already trimmed its pool and retried).
# app.py exits with RAPID_DOC_DEVICE_LOST_EXIT_CODE in either case; restart
# only that status here so the service recovers even when the container is
# started without a restart policy.  Any other exit is final (the container's
# own restart policy still applies).
RESTART_CODE=${RAPID_DOC_DEVICE_LOST_EXIT_CODE:-75}
CHILD_PID=""

shutdown() {
    trap - TERM INT
    if [ -n "$CHILD_PID" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 143
}
trap shutdown TERM INT

while true; do
    status=0
    python3 app.py &
    CHILD_PID=$!
    wait "$CHILD_PID" || status=$?
    CHILD_PID=""

    if [ "$status" -ne "$RESTART_CODE" ]; then
        exit "$status"
    fi
    echo "[supervisor] device failure (exit $status); restarting the API process in 2s..."
    sleep 2
done
