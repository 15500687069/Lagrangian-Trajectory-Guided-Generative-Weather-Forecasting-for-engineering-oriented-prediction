#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CMD=("${PYTHON_BIN}" "miniltgnet/scripts/run_seed_ablation.py")
if [[ "${SKIP_PREPARE}" == "1" ]]; then
  CMD+=("--skip_prepare")
fi

echo ">>> ${CMD[*]}"
"${CMD[@]}"
