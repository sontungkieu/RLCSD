#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${RLCSD_SMOKE_CONFIG:-configs/smoke/qwen3_1_7b_rlcsd_32gb.yaml}"
MODEL="${RLCSD_SMOKE_MODEL:-Qwen/Qwen3-1.7B}"
DATA_DIR="${RLCSD_SMOKE_DATA_DIR:-/tmp/rlcsd-smoke/data}"

python3 docker/make_smoke_data.py --out "${DATA_DIR}"

bash scripts/_run_verl.sh "${CONFIG}" \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  reward.num_workers=4 \
  trainer.total_training_steps=1 \
  trainer.default_hdfs_dir=null \
  +trainer.remove_previous_ckpt_in_save=True

latest_log="$(find /tmp/rlcsd-smoke/outputs -name train_log.txt -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -n "${latest_log}" ] && grep -q "Training Progress" "${latest_log}"; then
  echo "RLCSD_MINI_EPOCH_OK log=${latest_log}"
else
  echo "RLCSD mini epoch finished but log marker was not found" >&2
  exit 1
fi
