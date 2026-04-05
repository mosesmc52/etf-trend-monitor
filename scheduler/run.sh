#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------

LOG_FILE="${LOG_FILE:-/tmp/app.log}"

# Send stdout/stderr to BOTH docker logs and file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================"
echo "[$(date -Is)] ETF Trend Monitor started"
echo "Log file: $LOG_FILE"
echo "================================================"

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------

set -a
[ -f /app/.env ] && . /app/.env
set +a

cd /app

# ------------------------------------------------------------
# Run main job
# ------------------------------------------------------------

echo "[INFO] Running monitor..."

set +e
poetry run python run_monitor.py --email
EXIT_CODE=$?
set -e

echo "[INFO] Finished with exit code: $EXIT_CODE"

# ------------------------------------------------------------
# Emit structured tail (very useful for debugging)
# ------------------------------------------------------------

echo
echo "================ LOG TAIL (last 50 lines) ================"
tail -n 50 "$LOG_FILE" || true
echo "========================================================="

exit $EXIT_CODE
