#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/third_party/tabular-dl-tabr-official"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python bin/go.py exp/tabr/convat_apex_anxiety/1-finetune-tuning.toml --continue
