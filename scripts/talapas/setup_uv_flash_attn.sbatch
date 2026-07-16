#!/usr/bin/env bash
#SBATCH --job-name=rlcsd-env
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/gpfs/projects/ailab/tnguye11/sontungkieu/RLCSD}
UV_BIN=${UV_BIN:-/gpfs/home/tnguye11/.local/bin/uv}
ENV_PROFILE=${ENV_PROFILE:-rlcsd-cu13-sm80}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/ailab/tnguye11/rlcsd}
LOCAL_ENV_DIR="/tmp/${USER}_${ENV_PROFILE}"
ARTIFACT_ROOT="${REPO_ROOT}/.talapas"
ENV_ARCHIVE="${ARTIFACT_ROOT}/envs/${ENV_PROFILE}.tar.gz"
WHEELHOUSE="${ARTIFACT_ROOT}/wheelhouse/${ENV_PROFILE}"
UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"

source /etc/profile.d/modules.sh
module purge
module load cuda/13.0

CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST=8.0
export CUDAARCHS=80
export CMAKE_CUDA_ARCHITECTURES=80
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-1}
export UV_CACHE_DIR
export UV_LINK_MODE=copy

mkdir -p "${ARTIFACT_ROOT}/envs" "${WHEELHOUSE}" "${UV_CACHE_DIR}"
rm -rf "${LOCAL_ENV_DIR}" "${WHEELHOUSE:?}"/*
cleanup() {
    rm -rf "${LOCAL_ENV_DIR}"
}
trap cleanup EXIT

cd "${REPO_ROOT}"
"${UV_BIN}" venv --python 3.10 "${LOCAL_ENV_DIR}"
PYTHON="${LOCAL_ENV_DIR}/bin/python"

"${UV_BIN}" pip install --python "${PYTHON}" \
    pip setuptools wheel packaging pybind11 ninja
"${UV_BIN}" pip install --python "${PYTHON}" \
    --requirements docker/requirements-cu13.txt

"${PYTHON}" -m pip wheel -v \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "${WHEELHOUSE}" \
    flash-attn==2.8.3.post1 \
    2>&1 | tee "${WHEELHOUSE}/build.log"

FLASH_WHEEL=$(find "${WHEELHOUSE}" -maxdepth 1 -type f \
    -name 'flash_attn*.whl' -print -quit)
if [[ -z "${FLASH_WHEEL}" ]]; then
    echo "FlashAttention build did not produce a wheel" >&2
    exit 1
fi
"${UV_BIN}" pip install --python "${PYTHON}" --no-deps "${FLASH_WHEEL}"
"${PYTHON}" docker/apply_vllm_w4_patch.py
"${PYTHON}" docker/apply_vllm_w4_patch.py --check

"${PYTHON}" - <<'PY'
import json

import flash_attn
import torch
import vllm

print(
    json.dumps(
        {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "flash_attn": flash_attn.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
        },
        sort_keys=True,
    ),
    flush=True,
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the completed environment")
PY

"${PYTHON}" -m pip freeze > "${ARTIFACT_ROOT}/envs/${ENV_PROFILE}.freeze.txt"
ARCHIVE_TMP="${ENV_ARCHIVE}.tmp.${SLURM_JOB_ID}"
rm -f "${ARCHIVE_TMP}"
tar -C /tmp -czf "${ARCHIVE_TMP}" "$(basename "${LOCAL_ENV_DIR}")"
mv -f "${ARCHIVE_TMP}" "${ENV_ARCHIVE}"
sha256sum "${ENV_ARCHIVE}" > "${ENV_ARCHIVE}.sha256"

echo "RLCSD_TALAPAS_ENV_OK archive=${ENV_ARCHIVE} wheel=${FLASH_WHEEL}"
