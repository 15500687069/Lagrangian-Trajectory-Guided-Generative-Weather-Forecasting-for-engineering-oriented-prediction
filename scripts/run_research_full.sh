#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARE_CONFIG="${PREPARE_CONFIG:-configs/stage1_regional.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/stage1_regional.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/stage2_redesign.yaml}"
STAGE3_RECOVER_CONFIG="${STAGE3_RECOVER_CONFIG:-configs/stage3_recover.yaml}"
STAGE3_FINETUNE_CONFIG="${STAGE3_FINETUNE_CONFIG:-configs/stage3_finetune.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

run_stage() {
  local stage_name="$1"
  local cfg="$2"
  local ckpt="$3"
  run_cmd "${PYTHON_BIN}" -m ltg_net.cli train --config "${cfg}"
  run_cmd "${PYTHON_BIN}" -m ltg_net.cli evaluate --config "${cfg}" --checkpoint "${ckpt}" --split "${EVAL_SPLIT}"
}

echo "Repository: ${ROOT_DIR}"
run_cmd "${PYTHON_BIN}" --version

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  run_cmd "${PYTHON_BIN}" -m ltg_net.cli prepare --config "${PREPARE_CONFIG}"
fi

if [[ "${SKIP_STAGE1}" != "1" ]]; then
  run_stage "stage1" "${STAGE1_CONFIG}" "outputs/stage1/best.pt"
fi

if [[ "${SKIP_STAGE2}" != "1" ]]; then
  run_stage "stage2" "${STAGE2_CONFIG}" "outputs/stage2_redesign/best.pt"
fi

if [[ "${SKIP_STAGE3}" != "1" ]]; then
  run_stage "stage3_recover" "${STAGE3_RECOVER_CONFIG}" "outputs/stage3_recover/best.pt"
  run_stage "stage3_finetune" "${STAGE3_FINETUNE_CONFIG}" "outputs/stage3_finetune/best.pt"
  run_cmd "${PYTHON_BIN}" scripts/export_metrics_summary.py \
    --inputs outputs/stage3_recover/test_metrics.jsonl outputs/stage3_finetune/test_metrics.jsonl \
    --output outputs/stage3_metrics_summary.csv
fi

echo ""
echo "Done."
