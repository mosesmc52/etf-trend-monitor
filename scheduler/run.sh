#!/bin/bash
echo "[$(date)] Running ETF trend monitor..."

set -euo pipefail

set -a
[ -f /app/.env ] && . /app/.env
set +a

cd /app

poetry run python run_monitor.py --email
