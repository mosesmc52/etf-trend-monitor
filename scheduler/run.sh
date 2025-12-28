#!/bin/bash
#!/bin/bash
echo "[$(date)] Running monthly etf trend monitor..."

set -euo pipefail

set -a
[ -f /app/.env ] && . /app/.env
set +a

cd /app

poetry run python run_monitor.py
