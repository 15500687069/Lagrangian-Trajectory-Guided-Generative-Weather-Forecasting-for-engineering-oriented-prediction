#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-miniltgnet/configs/lite_plus.yaml}"
CHECKPOINT="${2:-outputs/miniltg/lite_plus/best.pt}"
SPLIT="${3:-test}"
OUT_DIR="${4:-outputs/miniltg/validation}"

python miniltgnet/scripts/run_validation_benchmark.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --output_dir "${OUT_DIR}" \
  --methods model persistence linear climatology

