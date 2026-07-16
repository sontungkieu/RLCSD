#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/gpfs/projects/ailab/tnguye11/sontungkieu/RLCSD}
UV_BIN=${UV_BIN:-/gpfs/home/tnguye11/.local/bin/uv}
ENV_PROFILE=${ENV_PROFILE:-rlcsd-cu13-sm80}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/ailab/tnguye11/rlcsd}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-ailab}
SLURM_PARTITION=${SLURM_PARTITION:-gpu}
SLURM_GRES=${SLURM_GRES:-gpu:nvidia_a100_80gb_pcie_3g.40gb:1}
FORCE_SETUP=${FORCE_SETUP:-0}
LOG_ROOT="${REPO_ROOT}/outputs/talapas/slurm"
SUBMIT_ROOT="${REPO_ROOT}/outputs/talapas/submissions"
ENV_ARCHIVE="${REPO_ROOT}/.talapas/envs/${ENV_PROFILE}.tar.gz"

mkdir -p "${LOG_ROOT}" "${SUBMIT_ROOT}" "${SCRATCH_ROOT}"
EXPORTS="ALL,REPO_ROOT=${REPO_ROOT},UV_BIN=${UV_BIN},ENV_PROFILE=${ENV_PROFILE},SCRATCH_ROOT=${SCRATCH_ROOT}"
DEPENDENCY_ARGS=()
SETUP_JOB_ID=""

if [[ ! -f "${ENV_ARCHIVE}" || "${FORCE_SETUP}" == "1" ]]; then
    SETUP_JOB_ID=$(sbatch --parsable \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --gres="${SLURM_GRES}" \
        --chdir="${REPO_ROOT}" \
        --export="${EXPORTS}" \
        --output="${LOG_ROOT}/setup_%j.out" \
        --error="${LOG_ROOT}/setup_%j.err" \
        scripts/talapas/setup_uv_flash_attn.sbatch)
    DEPENDENCY_ARGS=(--dependency="afterok:${SETUP_JOB_ID}")
fi

TRAIN_JOB_ID=$(sbatch --parsable \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --gres="${SLURM_GRES}" \
    --chdir="${REPO_ROOT}" \
    --export="${EXPORTS}" \
    --output="${LOG_ROOT}/train_%j.out" \
    --error="${LOG_ROOT}/train_%j.err" \
    "${DEPENDENCY_ARGS[@]}" \
    scripts/talapas/train_w4_smoke.sbatch)

SUBMIT_RECORD="${SUBMIT_ROOT}/$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "setup_job_id=${SETUP_JOB_ID}"
    echo "train_job_id=${TRAIN_JOB_ID}"
    echo "env_profile=${ENV_PROFILE}"
    echo "gres=${SLURM_GRES}"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
} > "${SUBMIT_RECORD}"

echo "setup_job_id=${SETUP_JOB_ID:-reused}"
echo "train_job_id=${TRAIN_JOB_ID}"
echo "submit_record=${SUBMIT_RECORD}"
