#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/tabr_overnight_${STAMP}.log"

{
  echo "[$(date '+%F %T')] START finetune"
  bash "$ROOT/scripts/run_official_tabr_v25_finetune_8h.sh"
  echo "[$(date '+%F %T')] DONE finetune"
  echo "[$(date '+%F %T')] START scaling-freeze"
  bash "$ROOT/scripts/run_official_tabr_scaling_freeze_8h.sh"
  echo "[$(date '+%F %T')] DONE scaling-freeze"
} 2>&1 | tee "$LOG_FILE"

echo "Log saved to: $LOG_FILE"
