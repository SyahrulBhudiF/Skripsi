#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/third_party/tabular-dl-tabr-official"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python bin/ensemble.py exp/tabr/convat_apex_anxiety/0-evaluation --n-ensembles 2 --ensemble-size 5
