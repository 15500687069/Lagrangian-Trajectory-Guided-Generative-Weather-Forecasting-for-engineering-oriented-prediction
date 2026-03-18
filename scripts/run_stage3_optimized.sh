#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARE_CONFIG="${PREPARE_CONFIG:-configs/stage1_regional.yaml}"
RECOVER_CONFIG="${RECOVER_CONFIG:-configs/stage3_recover.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/stage3_finetune.yaml}"
SANITY_CONFIG="${SANITY_CONFIG:-configs/stage3_redesign_sanity.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
RUN_SANITY="${RUN_SANITY:-1}"
SANITY_STEPS="${SANITY_STEPS:-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

echo "Repository: ${ROOT_DIR}"
run_cmd "${PYTHON_BIN}" --version

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  run_cmd "${PYTHON_BIN}" -m ltg_net.cli prepare --config "${PREPARE_CONFIG}"
fi

if [[ "${RUN_SANITY}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" scripts/sanity_check.py --config "${SANITY_CONFIG}" --steps "${SANITY_STEPS}" --backward
fi

run_cmd "${PYTHON_BIN}" -m ltg_net.cli train --config "${RECOVER_CONFIG}"
run_cmd "${PYTHON_BIN}" -m ltg_net.cli evaluate --config "${RECOVER_CONFIG}" --checkpoint outputs/stage3_recover/best.pt --split "${EVAL_SPLIT}"

run_cmd "${PYTHON_BIN}" -m ltg_net.cli train --config "${FINETUNE_CONFIG}"
run_cmd "${PYTHON_BIN}" -m ltg_net.cli evaluate --config "${FINETUNE_CONFIG}" --checkpoint outputs/stage3_finetune/best.pt --split "${EVAL_SPLIT}"

run_cmd "${PYTHON_BIN}" scripts/export_metrics_summary.py \
  --inputs outputs/stage3_recover/test_metrics.jsonl outputs/stage3_finetune/test_metrics.jsonl \
  --output outputs/stage3_metrics_summary.csv

echo ""
echo "Done. Key outputs:"
echo "  outputs/stage3_recover/"
echo "  outputs/stage3_finetune/"
echo "  outputs/stage3_metrics_summary.csv"
