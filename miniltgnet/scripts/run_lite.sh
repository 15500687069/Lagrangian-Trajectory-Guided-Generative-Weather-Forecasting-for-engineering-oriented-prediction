#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-miniltgnet/configs/lite.yaml}"
SANITY_CONFIG="${SANITY_CONFIG:-miniltgnet/configs/lite_sanity.yaml}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
RUN_SANITY="${RUN_SANITY:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

echo "Repository: ${ROOT_DIR}"
run_cmd "${PYTHON_BIN}" --version

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  run_cmd "${PYTHON_BIN}" -m miniltgnet.cli prepare --config "${CONFIG}"
fi

if [[ "${RUN_SANITY}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" -m miniltgnet.cli sanity --config "${SANITY_CONFIG}" --steps 2 --backward
fi

run_cmd "${PYTHON_BIN}" -m miniltgnet.cli train --config "${CONFIG}"
run_cmd "${PYTHON_BIN}" -m miniltgnet.cli evaluate --config "${CONFIG}" --checkpoint outputs/miniltg/lite/best.pt --split "${EVAL_SPLIT}"

echo ""
echo "Done. Output: outputs/miniltg/lite/"
