#!/bin/bash
set -eo pipefail
set -x

# Base script: reads all params from a YAML config and launches verl training.
# Usage: bash scripts/_run_verl.sh <config.yaml> [extra hydra overrides...]

CONFIG=${1:?Usage: $0 <config.yaml> [extra overrides...]}
shift  # remaining args are extra hydra overrides

export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}
export PATH="$CUDA_HOME/bin:$PATH"
if [ -n "${HF_ENDPOINT}" ]; then export HF_ENDPOINT; fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}
# Note: do NOT set TORCH_DISTRIBUTED_DEBUG=DETAIL — its debug PG wrapper
# lacks `allgather_into_tensor_coalesced`, which FSDP2/DTensor need when
# resharding weights to vLLM. NCCL flight recorder above is enough to
# capture stack traces on collective desync.

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/third_party/verl:$(pwd):${PYTHONPATH}"

# --- Read YAML helper ---
Y() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('$1','${2:-}'))"; }

detect_n_gpus_per_node() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
        python3 - "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys

visible = [part for part in sys.argv[1].split(",") if part.strip()]
print(len(visible))
PY
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
        return
    fi
    python3 - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print("")
PY
}

METHOD=$(Y method grpo)
MODEL_PATH=$(Y model_path Qwen/Qwen3-1.7B)
USE_LORA=$(Y use_lora true)
LORA_R=$(Y lora_r 64)
LORA_ALPHA=$(Y lora_alpha 128)
LR=$(Y learning_rate 1e-5)
NUM_EPOCHS=$(Y num_epochs 30)
BS=$(Y per_device_batch_size 4)
GROUP_SIZE=$(Y group_size 4)
MAX_GRAD_NORM=$(Y max_grad_norm 1.0)
WARMUP=$(Y warmup_steps 50)
WEIGHT_DECAY=$(Y weight_decay 0.01)
MAX_TRAIN=$(Y max_train_samples 17000)
SAVE_STEPS=$(Y save_steps 200)
EVAL_STEPS=$(Y eval_steps 100)
MAX_PROMPT=$(Y max_prompt_length 2048)
MAX_RESP=$(Y max_completion_length 2048)
VAL_MAX_RESP=$(Y val_max_completion_length 38912)
ACTOR_MAX_TOKEN_LEN=$(Y actor_max_token_len_per_gpu "")
TEMP=$(Y temperature 1.0)
TOP_P=$(Y top_p 0.95)
TOP_K=$(Y top_k_sampling 20)
GPU_MEM=$(Y vllm_gpu_memory_utilization 0.6)
TP=$(Y vllm_tensor_parallel_size 2)
VLLM_ATTENTION_BACKEND=$(Y vllm_attention_backend "${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}")
STUDENT_ENABLE_THINKING=$(Y student_enable_thinking false)
VAL_ENABLE_THINKING=$(Y val_enable_thinking true)
VAL_DO_SAMPLE=$(Y val_do_sample true)
VAL_N=$(Y val_n 12)
VAL_TEMP=$(Y val_temperature 0.6)
VAL_TOP_P=$(Y val_top_p ${TOP_P})
VAL_TOP_K=$(Y val_top_k ${TOP_K})
VAL_BATCH_SIZE=$(Y val_batch_size 16)
TEACHER_ENABLE_THINKING=$(Y teacher_enable_thinking true)
THINKING_SYSTEM_PROMPT=$(Y thinking_system_prompt false)
PROJECT=$(Y project_name rlcsd)
EXPERIMENT_NAME=$(Y experiment_name ${METHOD})
OUTPUT_DIR=$(Y output_dir "")
VAL_BEFORE=$(Y val_before_train false)
USE_TENSORBOARD=$(Y use_tensorboard true)
PRIVILEGED_TEXT_MODE=$(Y privileged_text_mode solution_answer)
# OPD (plain on-policy distillation) keys; only consumed when METHOD=opd.
TEACHER_MODEL_PATH=$(Y teacher_model_path "")
TEACHER_ENABLE_RESOURCE_POOL=$(Y teacher_enable_resource_pool false)
TEACHER_N_GPUS_PER_NODE=$(Y teacher_n_gpus_per_node 0)
N_GPUS_PER_NODE=$(Y n_gpus_per_node "")
TEACHER_TP_SIZE=$(Y teacher_tp_size 1)
TEACHER_GPU_MEM=$(Y teacher_gpu_memory_utilization 0.3)
DISTILL_LOSS_MODE=$(Y distillation_loss_mode k1)
DISTILL_USE_PG=$(Y distillation_use_policy_gradient true)
DISTILL_USE_TASK_REWARDS=$(Y distillation_use_task_rewards false)
DISTILL_TOPK=$(Y distillation_topk 64)
DISTILL_LOSS_COEF=$(Y distillation_loss_coef 1.0)
DISTILL_LOSS_MAX_CLAMP=$(Y distillation_loss_max_clamp 10.0)
DISTILL_LOG_PROB_MIN_CLAMP=$(Y distillation_log_prob_min_clamp -10.0)
DISTILL_NUM_WORKERS=$(Y distillation_num_workers 8)
DATA_ROOT=$(Y data_dir ./data)
TRAIN_DATASET=$(Y train_dataset dapo_math_17k)
VAL_DATASET=$(Y val_dataset amc23,aime24,aime25)
TRAIN_FILES=$(Y train_files "")
VAL_FILES=$(Y val_files "")
REWARD_MANAGER_NAME=$(Y reward_manager_name "")
USE_CUSTOM_REWARD_FUNCTION=$(Y use_custom_reward_function true)
CUSTOM_REWARD_FUNCTION_PATH=$(Y custom_reward_function_path src/verl_reward.py)
CUSTOM_REWARD_FUNCTION_NAME=$(Y custom_reward_function_name compute_score)
NCCL_TIMEOUT=$(Y nccl_timeout 7200)
ROLLOUT_IS=$(Y rollout_is "")
ROLLOUT_IS_THRESHOLD=$(Y rollout_is_threshold "")
ROLLOUT_IS_BATCH_NORMALIZE=$(Y rollout_is_batch_normalize "")
ROLLOUT_RS=$(Y rollout_rs "")
ROLLOUT_RS_THRESHOLD=$(Y rollout_rs_threshold "")
OVERRIDE_MAX_POSITION_EMBEDDINGS=$(Y override_max_position_embeddings "")
ROPE_SCALING_TYPE=$(Y rope_scaling_type "")
ROPE_SCALING_FACTOR=$(Y rope_scaling_factor "")
ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS=$(Y rope_scaling_original_max_position_embeddings "")
KL_LOSS_COEF=$(Y kl_loss_coef "")
RESUME_DIR=""

USE_LORA_LOWER=$(echo "$USE_LORA" | tr '[:upper:]' '[:lower:]')
USE_CUSTOM_REWARD_FUNCTION_LOWER=$(echo "$USE_CUSTOM_REWARD_FUNCTION" | tr '[:upper:]' '[:lower:]')
USE_TENSORBOARD_LOWER=$(echo "$USE_TENSORBOARD" | tr '[:upper:]' '[:lower:]')

case "$METHOD" in
    grpo|opd|opsd|opsd_ectr|sdpo|rlsd|rlsd_ectr|srpo|rlcsd)
        ;;
    *)
        echo "Invalid method '${METHOD}' in ${CONFIG}. Allowed values: grpo, opd, opsd, opsd_ectr, sdpo, rlsd, rlsd_ectr, srpo, rlcsd" >&2
        exit 1
        ;;
esac

if [ "$METHOD" = "opd" ]; then
    if [ -z "$TEACHER_MODEL_PATH" ] || [ "$TEACHER_MODEL_PATH" = "None" ]; then
        echo "method=opd requires teacher_model_path in ${CONFIG}" >&2
        exit 1
    fi
fi

if [ -z "$EXPERIMENT_NAME" ] || [ "$EXPERIMENT_NAME" = "None" ]; then
    EXPERIMENT_NAME="${METHOD}"
fi

if [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" = "None" ]; then
    OUTPUT_DIR="./outputs/${PROJECT}/${EXPERIMENT_NAME}"
fi

REWARD_ARGS=()
if [ -n "$REWARD_MANAGER_NAME" ] && [ "$REWARD_MANAGER_NAME" != "None" ]; then
    REWARD_ARGS+=(reward.reward_manager.name=${REWARD_MANAGER_NAME})
fi
if [ "$USE_CUSTOM_REWARD_FUNCTION_LOWER" = "true" ] \
    && [ -n "$CUSTOM_REWARD_FUNCTION_PATH" ] && [ "$CUSTOM_REWARD_FUNCTION_PATH" != "None" ] \
    && [ -n "$CUSTOM_REWARD_FUNCTION_NAME" ] && [ "$CUSTOM_REWARD_FUNCTION_NAME" != "None" ]; then
    REWARD_ARGS+=(
        reward.custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}
        reward.custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME}
    )
fi

LORA_ARGS=(
    actor_rollout_ref.model.lora.merge=False
    actor_rollout_ref.model.lora_rank=0
)
if [ "$USE_LORA_LOWER" = "true" ]; then
    LORA_ARGS=(
        actor_rollout_ref.model.lora.merge=True
        actor_rollout_ref.model.lora_rank=${LORA_R}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    )
fi

HYDRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        resume_dir=*|+resume_dir=*)
            RESUME_DIR="${arg#*=}"
            ;;
        train_dataset=*|+train_dataset=*)
            TRAIN_DATASET="${arg#*=}"
            ;;
        val_dataset=*|+val_dataset=*)
            VAL_DATASET="${arg#*=}"
            ;;
        data_dir=*|+data_dir=*)
            DATA_ROOT="${arg#*=}"
            ;;
        train_files=*|+train_files=*)
            TRAIN_FILES="${arg#*=}"
            ;;
        val_files=*|+val_files=*)
            VAL_FILES="${arg#*=}"
            ;;
        n_gpus_per_node=*|+n_gpus_per_node=*)
            N_GPUS_PER_NODE="${arg#*=}"
            ;;
        vllm_tensor_parallel_size=*|+vllm_tensor_parallel_size=*)
            TP="${arg#*=}"
            ;;
        vllm_attention_backend=*|+vllm_attention_backend=*)
            VLLM_ATTENTION_BACKEND="${arg#*=}"
            ;;
        privileged_text_mode=*|+privileged_text_mode=*)
            PRIVILEGED_TEXT_MODE="${arg#*=}"
            ;;
        val_enable_thinking=*|+val_enable_thinking=*)
            VAL_ENABLE_THINKING="${arg#*=}"
            ;;
        use_tensorboard=*|+use_tensorboard=*)
            USE_TENSORBOARD="${arg#*=}"
            USE_TENSORBOARD_LOWER=$(echo "$USE_TENSORBOARD" | tr '[:upper:]' '[:lower:]')
            ;;
        reward_manager_name=*|+reward_manager_name=*)
            REWARD_MANAGER_NAME="${arg#*=}"
            ;;
        use_custom_reward_function=*|+use_custom_reward_function=*)
            USE_CUSTOM_REWARD_FUNCTION="${arg#*=}"
            USE_CUSTOM_REWARD_FUNCTION_LOWER=$(echo "$USE_CUSTOM_REWARD_FUNCTION" | tr '[:upper:]' '[:lower:]')
            ;;
        custom_reward_function_path=*|+custom_reward_function_path=*)
            CUSTOM_REWARD_FUNCTION_PATH="${arg#*=}"
            ;;
        custom_reward_function_name=*|+custom_reward_function_name=*)
            CUSTOM_REWARD_FUNCTION_NAME="${arg#*=}"
            ;;
        data.train_files=*)
            TRAIN_FILES="${arg#*=}"
            ;;
        data.val_files=*)
            VAL_FILES="${arg#*=}"
            ;;
        *)
            HYDRA_ARGS+=("$arg")
            ;;
    esac
done

if [ -z "${N_GPUS_PER_NODE}" ] || [ "${N_GPUS_PER_NODE}" = "None" ]; then
    N_GPUS_PER_NODE="$(detect_n_gpus_per_node)"
fi
if [ -z "${N_GPUS_PER_NODE}" ] || [ "${N_GPUS_PER_NODE}" = "0" ]; then
    N_GPUS_PER_NODE=8
    echo "Warning: could not detect GPU count; defaulting n_gpus_per_node=${N_GPUS_PER_NODE}" >&2
fi
if [ "$TP" -gt "$N_GPUS_PER_NODE" ]; then
    echo "vllm_tensor_parallel_size (${TP}) cannot exceed n_gpus_per_node (${N_GPUS_PER_NODE})" >&2
    exit 1
fi
if [ $((N_GPUS_PER_NODE % TP)) -ne 0 ]; then
    echo "n_gpus_per_node (${N_GPUS_PER_NODE}) must be divisible by vllm_tensor_parallel_size (${TP})" >&2
    exit 1
fi

NUM_DP=$((N_GPUS_PER_NODE / TP))

if [ -n "$VLLM_ATTENTION_BACKEND" ] && [ "$VLLM_ATTENTION_BACKEND" != "None" ]; then
    export VLLM_ATTENTION_BACKEND
else
    unset VLLM_ATTENTION_BACKEND
fi

ROLLOUT_BATCH=$((BS * NUM_DP))                    # prompts per rollout
ROLLOUT_SAMPLES=$((ROLLOUT_BATCH * GROUP_SIZE))    # samples per rollout

# ppo_mini_batch_size: how many samples per optimizer step.
# Default = rollout_samples (one update per rollout).
# Set smaller in yaml to get multiple updates per rollout.
PPO_MINI_BS=$(Y ppo_mini_batch_size ${ROLLOUT_BATCH})
# These token budgets are for full sequence length, so they must include prompt + response.
TRAIN_MAX_SEQ_LEN=$((MAX_PROMPT + MAX_RESP))
VAL_MAX_SEQ_LEN=$((MAX_PROMPT + VAL_MAX_RESP))
ROLLOUT_MAX_MODEL_LEN=${TRAIN_MAX_SEQ_LEN}
if [ "${VAL_MAX_SEQ_LEN}" -gt "${ROLLOUT_MAX_MODEL_LEN}" ]; then
    ROLLOUT_MAX_MODEL_LEN=${VAL_MAX_SEQ_LEN}
fi
if [ -z "${ACTOR_MAX_TOKEN_LEN}" ] || [ "${ACTOR_MAX_TOKEN_LEN}" = "None" ]; then
    ACTOR_MAX_TOKEN_LEN=${TRAIN_MAX_SEQ_LEN}
fi
LOGPROB_MAX_RESP=${MAX_RESP}
if [ "${VAL_MAX_RESP}" -gt "${LOGPROB_MAX_RESP}" ]; then
    LOGPROB_MAX_RESP=${VAL_MAX_RESP}
fi
LOGPROB_MAX_SEQ_LEN=$((MAX_PROMPT + LOGPROB_MAX_RESP))
ROLLOUT_MAX_BATCHED_TOKENS=${VAL_MAX_SEQ_LEN}

LOGGER='["console","file"]'
if [ "$USE_TENSORBOARD_LOWER" = "true" ]; then
    LOGGER='["console","tensorboard","file"]'
fi

PARQUET_ROOT="${DATA_ROOT%/}/verl"

normalize_dataset_spec() {
    local spec="$1"
    spec="$(echo "$spec" | tr -d '[:space:]')"
    spec="${spec#,}"
    spec="${spec%,}"
    while [[ "$spec" == *",,"* ]]; do
        spec="${spec//,,/,}"
    done
    printf '%s' "$spec"
}

dataset_spec_key() {
    local spec
    spec="$(normalize_dataset_spec "$1")"
    printf '%s' "${spec//,/+}"
}

normalize_privileged_text_mode() {
    local mode="$1"
    mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$mode" in
        solution+answer|solution_and_answer)
            mode="solution_answer"
            ;;
        answer|answer_only)
            mode="answer_only"
            ;;
    esac
    printf '%s' "$mode"
}

ensure_dataset_parquet() {
    local split="$1"
    local dataset_name="$2"
    local target_path="$3"

    if [ -f "$target_path" ]; then
        return 0
    fi
    echo "Missing ${split} parquet: ${target_path}" >&2
    echo "Run: python scripts/download_data.py --dataset ${dataset_name} --split ${split}" >&2
    exit 1
}

if [ -z "$TRAIN_FILES" ] || [ "$TRAIN_FILES" = "None" ]; then
    TRAIN_FILES="${PARQUET_ROOT}/${TRAIN_DATASET}/train.parquet"
    ensure_dataset_parquet train "$TRAIN_DATASET" "$TRAIN_FILES"
fi

if [ -z "$VAL_FILES" ] || [ "$VAL_FILES" = "None" ]; then
    VAL_DATASET="$(normalize_dataset_spec "$VAL_DATASET")"
    VAL_FILES="${PARQUET_ROOT}/$(dataset_spec_key "$VAL_DATASET")/val.parquet"
    ensure_dataset_parquet val "$VAL_DATASET" "$VAL_FILES"
fi

PRIVILEGED_TEXT_MODE="$(normalize_privileged_text_mode "$PRIVILEGED_TEXT_MODE")"

resolve_abs_path() {
    python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
}

TEE_ARGS=()
RESUME_ARGS=()

if [ -n "$RESUME_DIR" ] && [ "$RESUME_DIR" != "None" ]; then
    DIR="$(resolve_abs_path "${RESUME_DIR%/}")"
    if [ ! -d "$DIR" ]; then
        echo "Resume directory does not exist: $DIR" >&2
        exit 1
    fi
    if [ ! -f "$DIR/latest_checkpointed_iteration.txt" ]; then
        echo "Resume directory is missing latest_checkpointed_iteration.txt: $DIR" >&2
        exit 1
    fi

    RUN_TAG="$(basename "$DIR")"
    EXP="${EXPERIMENT_NAME}-${RUN_TAG}"
    TEE_ARGS=(-a)
    RESUME_ARGS+=(trainer.resume_mode=auto)
    export VERL_FILE_LOGGER_APPEND=1
else
    TS=$(date +%Y%m%d.%H%M%S)
    EXP="${EXPERIMENT_NAME}-${TS}"
    DIR="${OUTPUT_DIR%/}/${TS}"
    mkdir -p "$DIR"
    cp "$CONFIG" "$DIR/config.yaml"
    unset VERL_FILE_LOGGER_APPEND
fi

mkdir -p "$DIR"
if [ "$(resolve_abs_path "$CONFIG")" != "$(resolve_abs_path "$DIR/config.yaml")" ]; then
    cp "$CONFIG" "$DIR/config.yaml"
fi
if [ "$USE_TENSORBOARD_LOWER" = "true" ]; then
    export TENSORBOARD_DIR="${DIR}/tensorboard_log/"
    echo "TensorBoard log dir: ${TENSORBOARD_DIR}" >&2
else
    unset TENSORBOARD_DIR
fi
export VERL_FILE_LOGGER_PATH="${DIR}/metrics.jsonl"

# --- Build algorithm-specific overrides from YAML ---
ALGO=""
for key in beta jsd_token_clip teacher_mode \
           full_logit_distill distill_add_tail \
           alpha top_k_distill is_clip ema_decay \
           epsilon epsilon_w tau lam lam_decay_steps teacher_sync_interval \
           rlcsd_tau rlcsd_beta rlcsd_lam rlcsd_delta rlcsd_eta \
           rlcsd_residual_clip_low rlcsd_residual_clip_high rlcsd_k_max \
           opsd_ectr_alpha opsd_ectr_log_w_floor opsd_ectr_cw_tau \
           rlsd_ectr_lam \
           srpo_beta; do
    val=$(Y $key "")
    [ -n "$val" ] && [ "$val" != "None" ] && ALGO="$ALGO +actor_rollout_ref.actor.policy_loss.${key}=${val}"
done
if [ "$METHOD" != "grpo" ] && [ "$METHOD" != "opd" ]; then
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.privileged_text_mode=${PRIVILEGED_TEXT_MODE}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.teacher_enable_thinking=${TEACHER_ENABLE_THINKING}"
    ALGO="$ALGO +actor_rollout_ref.rollout.custom.thinking_system_prompt=${THINKING_SYSTEM_PROMPT}"
fi

# --- OPD-specific distillation overrides ---
DISTILL_ARGS=()
if [ "$METHOD" = "opd" ]; then
    DISTILL_ARGS=(
        distillation.enabled=True
        distillation.num_workers=${DISTILL_NUM_WORKERS}
        distillation.teacher_model.enable_resource_pool=${TEACHER_ENABLE_RESOURCE_POOL}
        distillation.teacher_model.n_gpus_per_node=${TEACHER_N_GPUS_PER_NODE}
        distillation.teacher_model.nnodes=1
        distillation.teacher_model.model_path=${TEACHER_MODEL_PATH}
        distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP_SIZE}
        distillation.teacher_model.inference.name=vllm
        distillation.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM}
        distillation.teacher_model.inference.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
        distillation.teacher_model.inference.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}
        distillation.distillation_loss.loss_mode=${DISTILL_LOSS_MODE}
        distillation.distillation_loss.topk=${DISTILL_TOPK}
        distillation.distillation_loss.use_task_rewards=${DISTILL_USE_TASK_REWARDS}
        distillation.distillation_loss.use_policy_gradient=${DISTILL_USE_PG}
        distillation.distillation_loss.distillation_loss_coef=${DISTILL_LOSS_COEF}
        distillation.distillation_loss.loss_max_clamp=${DISTILL_LOSS_MAX_CLAMP}
        distillation.distillation_loss.log_prob_min_clamp=${DISTILL_LOG_PROB_MIN_CLAMP}
    )
fi

ROLLOUT_CORR=""
for key in rollout_is rollout_is_threshold rollout_is_batch_normalize rollout_rs rollout_rs_threshold; do
    val=$(Y $key "")
    [ -n "$val" ] && [ "$val" != "None" ] && ROLLOUT_CORR="$ROLLOUT_CORR algorithm.rollout_correction.${key}=${val}"
done

MODEL_OVERRIDE=""
if [ -n "$OVERRIDE_MAX_POSITION_EMBEDDINGS" ] && [ "$OVERRIDE_MAX_POSITION_EMBEDDINGS" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.max_position_embeddings=${OVERRIDE_MAX_POSITION_EMBEDDINGS}"
fi
if [ -n "$ROPE_SCALING_TYPE" ] && [ "$ROPE_SCALING_TYPE" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.type=${ROPE_SCALING_TYPE}"
fi
if [ -n "$ROPE_SCALING_FACTOR" ] && [ "$ROPE_SCALING_FACTOR" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.factor=${ROPE_SCALING_FACTOR}"
fi
if [ -n "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" ] && [ "$ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS" != "None" ]; then
    MODEL_OVERRIDE="$MODEL_OVERRIDE +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=${ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
fi

# --- Use custom main for privileged-teacher distillation, standard for grpo / opd ---
MAIN=src.self_distill_main
case "$METHOD" in
    grpo|opd) MAIN=verl.trainer.main_ppo ;;
esac

WORKER_IMPL=$(Y use_legacy_worker_impl "")
if [ -z "$WORKER_IMPL" ] || [ "$WORKER_IMPL" = "None" ]; then
    case "$METHOD" in
        grpo|opd) WORKER_IMPL=disable ;;
        *)        WORKER_IMPL=enable ;;
    esac
fi

python3 -m $MAIN \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${RESUME_ARGS[@]}" \
    trainer.val_before_train=${VAL_BEFORE} \
    trainer.use_legacy_worker_impl=${WORKER_IMPL} \
    trainer.critic_warmup=0 \
    trainer.logger=${LOGGER} \
    trainer.project_name=${PROJECT} \
    trainer.experiment_name=${EXP} \
    trainer.default_local_dir=${DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_STEPS} \
    trainer.test_freq=${EVAL_STEPS} \
    trainer.total_epochs=${NUM_EPOCHS} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${ROLLOUT_BATCH} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.train_max_samples=${MAX_TRAIN} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    +data.apply_chat_template_kwargs.enable_thinking=${STUDENT_ENABLE_THINKING} \
    +data.val_apply_chat_template_kwargs.enable_thinking=${VAL_ENABLE_THINKING} \
    +data.thinking_system_prompt=${THINKING_SYSTEM_PROMPT} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP} \
    actor_rollout_ref.actor.grad_clip=${MAX_GRAD_NORM} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$(if [ -n "$KL_LOSS_COEF" ] && [ "$KL_LOSS_COEF" != "None" ]; then echo "$KL_LOSS_COEF"; else case "$METHOD" in rlsd|sdpo|srpo|opsd|opd) echo "0";; *) echo "0.001";; esac; fi) \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=$(case "$METHOD" in grpo|opd) echo "vanilla";; *) echo "$METHOD";; esac) \
    actor_rollout_ref.rollout.temperature=${TEMP} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_SEQ_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS} \
    +actor_rollout_ref.rollout.custom.val_response_length=${VAL_MAX_RESP} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_SEQ_LEN} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    "${REWARD_ARGS[@]}" \
    ${MODEL_OVERRIDE} \
    ${ALGO} \
    ${ROLLOUT_CORR} \
    "${DISTILL_ARGS[@]}" \
    "${HYDRA_ARGS[@]}" \
    2>&1 | tee "${TEE_ARGS[@]}" "${DIR}/train_log.txt"
