#!/usr/bin/env bash
set -euo pipefail

VOLUME="${RLCSD_FLASH_ATTN_VOLUME:-rlcsd-flash-attn-wheelhouse}"
REMOTE_PATH="${1:-/latest}"
LOCAL_DEST="${2:-wheelhouse}"

mkdir -p "${LOCAL_DEST}"
uvx --from modal modal volume get --force "${VOLUME}" "${REMOTE_PATH}" "${LOCAL_DEST}"

find "${LOCAL_DEST}" -maxdepth 2 -type f \( -name 'flash_attn*.whl' -o -name 'manifest.json' -o -name 'build.log' \) -print
