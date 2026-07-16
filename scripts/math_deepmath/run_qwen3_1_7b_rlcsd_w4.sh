#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../_run_verl.sh" \
  "${SCRIPT_DIR}/../../configs/math_deepmath/qwen3_1_7b_rlcsd_w4.yaml" "$@"
