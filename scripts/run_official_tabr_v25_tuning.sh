#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# python scripts/export_to_official_tabr_dataset.py

cd third_party/tabular-dl-tabr-official
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python bin/go.py exp/tabr/convat_apex_anxiety/0-tuning.toml
