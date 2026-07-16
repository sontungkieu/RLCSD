#!/usr/bin/env bash
#SBATCH --job-name=rlcsd-w4-3ep
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/gpfs/projects/ailab/tnguye11/sontungkieu/RLCSD}
ENV_PROFILE=${ENV_PROFILE:-rlcsd-cu13-sm80}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/ailab/tnguye11/rlcsd}
LOCAL_ENV_DIR="/tmp/${USER}_${ENV_PROFILE}"
ENV_ARCHIVE="${REPO_ROOT}/.talapas/envs/${ENV_PROFILE}.tar.gz"
JOB_ROOT="${REPO_ROOT}/outputs/talapas/w4_smoke_3ep_job${SLURM_JOB_ID}"
DATA_DIR="${JOB_ROOT}/data"
RUN_CONFIG="${JOB_ROOT}/config.yaml"

source /etc/profile.d/modules.sh
module purge
module load cuda/13.0

CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_ROOT}/hf-cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export UV_LINK_MODE=copy

if [[ ! -f "${ENV_ARCHIVE}" ]]; then
    echo "Missing Talapas environment archive: ${ENV_ARCHIVE}" >&2
    exit 1
fi

rm -rf "${LOCAL_ENV_DIR}"
cleanup() {
    rm -rf "${LOCAL_ENV_DIR}"
}
trap cleanup EXIT
tar -C /tmp -xzf "${ENV_ARCHIVE}"
export PATH="${LOCAL_ENV_DIR}/bin:${PATH}"

mkdir -p "${DATA_DIR}" "${JOB_ROOT}/runs" "${HF_HOME}"
cd "${REPO_ROOT}"

python3 - <<'PY'
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
    raise SystemExit("CUDA is unavailable in the staged environment")
PY

nvidia-smi > "${JOB_ROOT}/nvidia-smi.txt"
python3 docker/make_smoke_data.py --out "${DATA_DIR}"
python3 - "${RUN_CONFIG}" "${DATA_DIR}" "${JOB_ROOT}/runs" <<'PY'
import sys
from pathlib import Path

import yaml

target, data_dir, output_dir = map(Path, sys.argv[1:])
source = Path("configs/smoke/qwen3_1_7b_rlcsd_w4_1step.yaml")
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config.update(
    {
        "num_epochs": 3,
        "save_steps": 1,
        "logging_steps": 1,
        "use_tensorboard": True,
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "train_files": str(data_dir / "train.parquet"),
        "val_files": str(data_dir / "val.parquet"),
        "experiment_name": "qwen3_1_7b_rlcsd_w4_talapas_3ep",
    }
)
target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

bash scripts/_run_verl.sh "${RUN_CONFIG}" \
    trainer.total_training_steps=3 \
    trainer.default_hdfs_dir=null \
    +trainer.remove_previous_ckpt_in_save=False

RUN_DIR=$(find "${JOB_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/metrics.jsonl" ]]; then
    echo "Training did not produce metrics.jsonl" >&2
    exit 1
fi

mapfile -t CHECKPOINTS < <(find "${RUN_DIR}" -type d -name 'global_step_*' | sort)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "Training did not produce a global_step checkpoint" >&2
    exit 1
fi

python3 - "${RUN_DIR}/metrics.jsonl" "${RUN_DIR}/summary.json" <<'PY'
import json
import statistics
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
if not rows:
    raise SystemExit("metrics.jsonl is empty")

def mean(key):
    values = [float(row[key]) for row in rows if key in row]
    return statistics.fmean(values) if values else None

summary = {
    "steps": [row.get("training/global_step") for row in rows],
    "step_count": len(rows),
    "mean_step_s": mean("timing_s/step"),
    "mean_gen_s": mean("timing_s/gen"),
    "mean_update_actor_s": mean("timing_s/update_actor"),
    "mean_update_weights_s": mean("timing_s/update_weights"),
    "mean_throughput": mean("perf/throughput"),
    "last_step": rows[-1],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY

printf '%s\n' "${CHECKPOINTS[@]}" > "${RUN_DIR}/checkpoints.txt"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
echo "RLCSD_TALAPAS_TRAIN_OK run_dir=${RUN_DIR} checkpoints=${#CHECKPOINTS[@]}"
