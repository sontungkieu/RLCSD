# RLCSD 3090 Ti Standard 16k Benchmark

Run ID: `rlcsd_3090ti_16k_20260702_1224`

Remote host: `root@174.94.157.109:36553`

Local archive: `/home/tung/RLCSD/experiments/rlcsd_3090ti_16k_20260702_1224`

## Scope

This archive stores the pulled benchmark artifacts, logs, configuration files,
GPU monitor traces, TensorBoard event file, run outputs, and cleanup verification
for the RLCSD standard 16k/40k-context benchmark.

The raw remote parquet files were not copied into this local archive. They were
downloaded inside the private remote benchmark workspace and that workspace was
removed during cleanup to restore the remote environment. The dataset manifest
and download logs are preserved under `artifacts/data_files.txt`,
`artifacts/download_train_data.txt`, and `artifacts/download_val_data.txt`.

Recorded remote dataset files:

- `deepmath_filtered_level5_7/train.parquet`: `1031585556` bytes.
- `amc23+aime24+aime25/val.parquet`: `105032` bytes.

## Config Under Test

The standard training attempt kept the important long-context knobs:

- `max_prompt_length=2048`
- `max_completion_length=16384`
- `val_max_completion_length=38912`
- `actor_rollout_ref.rollout.max_model_len=40960`
- `actor_rollout_ref.rollout.max_num_batched_tokens=40960`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960`
- `train_batch_size=8`
- `rollout.n=8`
- `ppo_mini_batch_size=16`
- `vllm_gpu_memory_utilization=0.6`

The exact generated YAML is saved at:
`artifacts/qwen3_1_7b_rlcsd_standard_16k_train_bench.yaml`.

## vLLM Results

Direct 16k generation benchmark:

| name | requested backend | observed backend | mean generated tok/s | mean elapsed s | generated tokens | exit |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `flash_attn_16k_fp16` | `FLASH_ATTN` | `FLASH_ATTN` | `456.0938` | `287.3868` | `131072` | `0` |
| `flashinfer_16k_fp16` | `FLASHINFER` | `FLASH_ATTN` | `452.7715` | `289.4890` | `131072` | `0` |

40k-context, 16k-generation benchmark:

| name | requested backend | observed backend | mean generated tok/s | mean elapsed s | generated tokens | exit |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `flash_attn_40kctx_16kgen` | `FLASH_ATTN` | `FLASH_ATTN` | `456.3287` | `287.2315` | `131072` | `0` |

The `FLASHINFER` request did not actually switch backend in this environment;
the observed backend stayed `FLASH_ATTN`.

## Training Attempt Result

The one-step standard training run did not complete.

It passed dependency setup, dataset loading, model initialization, and vLLM
rollout startup. It then failed after about `16m16s` during actor log-prob
computation:

- failing phase: `old_log_prob -> actor_rollout_compute_log_prob -> lm_head`
- error: CUDA OOM
- requested allocation: `26.08 GiB`
- GPU total capacity reported by PyTorch: `23.55 GiB`
- `metrics.jsonl`: `0` lines, so no completed training metric row was emitted

GPU monitor for the SDPA fallback run:

- samples: `219`
- max VRAM used: `21138 MiB`
- max GPU utilization: `100%`

The full FlashAttention2 training path could not be tested because
`flash-attn==2.8.3.post1` had no matching prebuilt wheel for the current
`torch2.9/cu12` stack, and source build was killed during `nvcc` compilation.

## Cleanup Verification

The final remote verification is stored in
`reports/remote_cleanup_verify.txt`.

Observed final state:

- virtual try-on backend health: OK on port `8000`
- virtual try-on frontend: HTTP 200 on port `8080`
- GPU memory after cleanup: `4 MiB / 24564 MiB`
- removed benchmark workspace: `/workspace/rlcsd_3090ti_16k_20260702_1224`
- removed temp dir: `/workspace/rt16k1224`
- removed temp dir: `/workspace/tmp16k1224`
- residual RLCSD/Ray/vLLM/SelfDistill processes: none listed

## Archive Layout

- `artifacts/`: command logs, dependency install logs, vLLM benchmark JSON/stdout,
  GPU monitor CSVs, generated config, exit codes, and dataset manifest.
- `train_outputs/`: copied run output folders, configs, train logs, empty
  `metrics.jsonl`, and TensorBoard event file for the SDPA fallback attempt.
- `reports/file_manifest.txt`: complete file list for this archive.
- `reports/remote_cleanup_verify.txt`: final remote health and cleanup check.
- `reports/du.txt`: archive disk usage.

## Repo Checklist

- Code changed in this task: N/A.
- README review/update: N/A, no runtime behavior or command contract changed.
- PDF review/build: N/A, no `pdf/` directory found for this repo checkout.
- LaTeX cache cleanup: N/A.
- Mindmap update: N/A, no mindmap files found.
- `milestones.md` review/update: N/A, file not present in this checkout.
- `temp/` notes: N/A.
- `plan.md` / `plan_next...`: N/A, no active plan file present.
