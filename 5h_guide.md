# 5h Guide: Remote 3090 Ti RLCSD Smoke Run and vLLM Generation Benchmark

Guide này dành cho một agent khác chạy trên remote server 3090 Ti. Agent phải coi đây là runbook độc lập: không giả định đã biết các benchmark hoặc patch logging trước đó.

Mục tiêu trong 5 giờ:

- Setup RLCSD trong một thư mục riêng trên remote, không cài hoặc ghi ra hệ thống chung.
- Không động vào data server hoặc shared model/data mounts. Không ghi, xóa, chmod, chown, rsync, hoặc tạo cache ở các path ngoài workspace riêng.
- Chạy vài iteration RLCSD với log TensorBoard/JSONL/stdout đầy đủ.
- Benchmark một số config sinh của vLLM trên prompt thật lấy từ dataset RLCSD, gồm throughput và memory.
- Kéo toàn bộ log/artifact nhỏ về local.
- Cleanup remote để server gần như trở lại trạng thái ban đầu.

## 0. Bối cảnh repo cần biết

Repo chính: `https://github.com/sontungkieu/RLCSD.git`

Commit nên dùng: `08688d2 feat(diagnostics): add scheduled RLCSD training metrics`

Entrypoint RLCSD:

- Script launcher: `scripts/_run_verl.sh`
- Script tiện dùng: `scripts/math_deepmath/run_qwen3_1_7b_rlcsd.sh`
- Main training cho RLCSD: `src.self_distill_main`
- Config gốc Qwen3-1.7B math: `configs/math_deepmath/qwen3_1_7b_rlcsd.yaml`

Config gốc có:

- `method: rlcsd`
- `max_prompt_length: 2048`
- `max_completion_length: 16384`
- `teacher_mode: snapshot`
- `teacher_sync_interval: 10`
- `rlcsd_k_max: 4`
- `use_tensorboard: true`
- model path math mặc định là `/workspace/storage-shared/models/Qwen3-1.7B`

Trên remote 3090 Ti, không dùng path `/workspace/storage-shared/...` vì yêu cầu không động vào data server. Override model sang `Qwen/Qwen3-1.7B` và đặt toàn bộ HF/vLLM/cache/output trong workspace riêng.

Logging hiện có:

- Coarse timers: `timing_s/step`, `timing_s/gen`, `timing_s/old_log_prob`, `timing_s/teacher_log_prob`, `timing_s/update_actor`, `timing_s/update_weights`, `timing_s/reward`, `timing_s/adv`, `timing_s/testing`, `timing_s/save_checkpoint`.
- Actor detailed timers: `actor_timing_s/student_forward`, `actor_timing_s/teacher_correct_forward`, `actor_timing_s/teacher_wrong_multi_forward`, `actor_timing_s/teacher_forward`, `actor_timing_s/policy_loss`, `actor_timing_s/backward`, `actor_timing_s/optimizer_step`, `actor_timing_s/teacher_snapshot_update`.
- Memory/system: `gpu/*`, `sys/*`, `proc/*`, `ray/*`.
- Token/RLCSD: `actor_tokens/*`, `rlcsd/*`.
- Detailed diagnostics default: steps 1-10, then every 10th step. For smoke runs, override `diagnostics.detailed_every_n_steps=1`.

RLCSD profiling interpretation:

- `teacher_log_prob` is not the full teacher cost for RLCSD.
- The real teacher-forward hotspot is often inside `update_actor`, split by actor timers.
- Compare at least:
  - `timing_s/gen`
  - `timing_s/old_log_prob`
  - `timing_s/update_actor`
  - `actor_timing_s/student_forward`
  - `actor_timing_s/teacher_correct_forward`
  - `actor_timing_s/teacher_wrong_multi_forward`
  - `actor_timing_s/backward`
  - `timing_s/update_weights`

## 1. Non-negotiable safety rules

Do not do these:

- Do not use `sudo`, `apt`, system package managers, or global `pip install`.
- Do not write to `/data`, `/mnt/data`, `/workspace/storage-shared`, `/datasets`, `/shared`, `/scratch/shared`, model-store mounts, or any path that looks like a shared data server.
- Do not run `rm -rf` on any path unless it is inside the workspace created for this task.
- Do not run `rsync --delete` against remote or shared paths.
- Do not print or store secrets.
- Do not modify shared datasets, model files, system CUDA, driver, Ray cluster service, or global caches.
- Do not leave Ray/vLLM/python processes running after the task.

Allowed:

- Create one isolated workspace under `$HOME` or another user-owned non-shared path.
- Download HF model/dataset into that workspace if internet is available.
- Read GPU/system info.
- Use only the current user's processes and files.

If the server cannot download model/data without using data server paths, stop and ask the user. Do not silently fall back to shared data.

## 2. 5-hour schedule

Target schedule:

| Time | Work |
|---:|---|
| 0:00-0:20 | Connect, create workspace, capture before-state, start command logging |
| 0:20-1:20 | Clone repo, create venv, install dependencies in workspace |
| 1:20-1:45 | Download minimal RLCSD train/val parquet into workspace |
| 1:45-2:45 | Run 2-4 RLCSD iterations, collect train logs and diagnostics |
| 2:45-3:50 | Run vLLM generation benchmarks on real RLCSD prompts |
| 3:50-4:35 | Pull logs to local and parse summaries |
| 4:35-5:00 | Cleanup remote, capture after-state, final report |

If setup exceeds 90 minutes, stop full training attempt and prioritize:

1. Import checks and exact blocker log.
2. vLLM generation benchmark if vLLM works.
3. Cleanup and report.

## 3. Local variables

Run these on local before SSH. Replace `USER@HOST`.

```bash
export REMOTE='USER@HOST'
export RUN_ID="rlcsd_3090ti_$(date +%Y%m%d_%H%M%S)"
export REMOTE_HOME="$(ssh "$REMOTE" 'printf %s "$HOME"')"
export REMOTE_ROOT="$REMOTE_HOME/$RUN_ID"
export LOCAL_ARCHIVE="$PWD/remote_artifacts/${RUN_ID}"
mkdir -p "$LOCAL_ARCHIVE"
```

Use this guide as source of truth. Copy it to the local archive for provenance:

```bash
cp 5h_guide.md "$LOCAL_ARCHIVE/"
```

When opening a remote shell, pass or re-export the same `RUN_ID`:

```bash
ssh "$REMOTE"
export RUN_ID='<paste the local RUN_ID value here>'
```

## 4. Remote preflight and isolated workspace

Run on remote.

```bash
: "${RUN_ID:?Set RUN_ID from the local shell and pass the same value to remote}"
export REMOTE_ROOT="$HOME/${RUN_ID}"
mkdir -p "$REMOTE_ROOT"/{artifacts,cache,tmp,tools,logs}
chmod 700 "$REMOTE_ROOT"

export HF_HOME="$REMOTE_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$REMOTE_ROOT/cache/xdg"
export UV_CACHE_DIR="$REMOTE_ROOT/cache/uv"
export PIP_CACHE_DIR="$REMOTE_ROOT/cache/pip"
export TRITON_CACHE_DIR="$REMOTE_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$REMOTE_ROOT/cache/vllm"
export TMPDIR="$REMOTE_ROOT/tmp"

exec > >(tee -a "$REMOTE_ROOT/logs/session.log") 2>&1
set -euxo pipefail

{
  echo "run_id=$RUN_ID"
  echo "remote_root=$REMOTE_ROOT"
  date -Is
  hostname
  whoami
  pwd
  uname -a
  df -h
  mount | sed -n '1,120p'
  nvidia-smi || true
  env | sort | grep -E '^(HF_|TRANSFORMERS_|HF_DATASETS_|XDG_CACHE_HOME|UV_CACHE_DIR|PIP_CACHE_DIR|TRITON_CACHE_DIR|VLLM_CACHE_ROOT|TMPDIR|CUDA|NVIDIA|PATH)=' || true
} | tee "$REMOTE_ROOT/artifacts/remote_state_before.txt"
```

Path guard for cleanup:

```bash
safe_rm() {
  target="$1"
  case "$target" in
    "$REMOTE_ROOT"/*|"$REMOTE_ROOT") rm -rf "$target" ;;
    *) echo "REFUSE rm outside REMOTE_ROOT: $target" >&2; return 2 ;;
  esac
}
```

Keep a change log:

```bash
cat > "$REMOTE_ROOT/artifacts/remote_changes.md" <<EOF
# Remote changes log

- Created workspace: $REMOTE_ROOT
- All caches, venvs, model downloads, dataset downloads, generated scripts, logs, and outputs must stay below this path.
- Forbidden: writes to data server/shared model paths.

EOF
```

## 5. Clone repo at the expected commit

Run on remote.

```bash
cd "$REMOTE_ROOT"
git clone https://github.com/sontungkieu/RLCSD.git repo
cd repo
git checkout 08688d2
git status --short --branch | tee "$REMOTE_ROOT/artifacts/git_status_initial.txt"
git log -1 --oneline | tee "$REMOTE_ROOT/artifacts/git_commit.txt"
```

Do not edit tracked repo files unless a runtime blocker requires a hotfix. If any hotfix is needed, save it as a patch:

```bash
git diff > "$REMOTE_ROOT/artifacts/remote_hotfix.patch"
```

Do not commit or push from remote unless the user explicitly asks.

## 6. Create isolated Python environment

Run on remote. Prefer venv under workspace. Do not use global pip.

```bash
cd "$REMOTE_ROOT/repo"
python3 -m venv "$REMOTE_ROOT/.venv"
source "$REMOTE_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r third_party/verl/requirements.txt
python -m pip check || true
```

Minimal import check:

```bash
python - <<'PY' | tee "$REMOTE_ROOT/artifacts/import_check.txt"
mods = ["torch", "transformers", "datasets", "pyarrow", "vllm", "ray", "tensordict", "hydra", "tensorboard", "psutil"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"OK {m} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"FAIL {m}: {type(exc).__name__}: {exc}")
PY
```

Important dependency note: prior runs failed before training when `tensordict` was missing. Do not launch training until `tensordict`, `torchdata`, `ray`, `hydra`, `tensorboard`, and `vllm` import cleanly.

## 7. Download minimal dataset into workspace only

Use the RLCSD HF dataset. Do not use shared dataset paths.

```bash
cd "$REMOTE_ROOT/repo"
source "$REMOTE_ROOT/.venv/bin/activate"
mkdir -p "$REMOTE_ROOT/runtime/data/verl"

python scripts/download_data.py \
  --dataset deepmath_filtered_level5_7 \
  --split train \
  --dst "$REMOTE_ROOT/runtime/data/verl"

python scripts/download_data.py \
  --dataset 'amc23+aime24+aime25' \
  --split val \
  --dst "$REMOTE_ROOT/runtime/data/verl"

find "$REMOTE_ROOT/runtime/data" -maxdepth 4 -type f -printf '%p %s\n' \
  | tee "$REMOTE_ROOT/artifacts/data_files.txt"
```

Expected files:

- `$REMOTE_ROOT/runtime/data/verl/deepmath_filtered_level5_7/train.parquet`
- `$REMOTE_ROOT/runtime/data/verl/amc23+aime24+aime25/val.parquet`

## 8. Build a 3090 Ti smoke config

Do not edit the tracked YAML. Create a temporary config under runtime.

This smoke config keeps input/output at 2048/2048 for the target case, disables checkpoint/eval, uses Qwen HF model with local cache, and forces all output into `$REMOTE_ROOT/runtime`.

```bash
cd "$REMOTE_ROOT/repo"
source "$REMOTE_ROOT/.venv/bin/activate"
mkdir -p "$REMOTE_ROOT/runtime/configs" "$REMOTE_ROOT/runtime/outputs"

python - <<'PY'
import os
from pathlib import Path
import yaml

remote_root = Path(os.environ["REMOTE_ROOT"])
src = Path("configs/math_deepmath/qwen3_1_7b_rlcsd.yaml")
dst = remote_root / "runtime/configs/qwen3_1_7b_rlcsd_3090ti_smoke.yaml"
cfg = yaml.safe_load(src.read_text())
cfg.update({
    "model_path": "Qwen/Qwen3-1.7B",
    "output_dir": str(remote_root / "runtime/outputs/rlcsd_3090ti_smoke"),
    "data_dir": str(remote_root / "runtime/data"),
    "num_epochs": 1,
    "per_device_batch_size": 1,
    "group_size": 4,
    "ppo_mini_batch_size": 4,
    "max_train_samples": 8,
    "save_steps": -1,
    "eval_steps": -1,
    "val_before_train": False,
    "max_prompt_length": 2048,
    "max_completion_length": 2048,
    "val_max_completion_length": 2048,
    "actor_max_token_len_per_gpu": 4096,
    "vllm_gpu_memory_utilization": 0.45,
    "vllm_tensor_parallel_size": 1,
    "rlcsd_k_max": 2,
    "experiment_name": "qwen3_1_7b_rlcsd_3090ti_smoke",
})
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY

cat "$REMOTE_ROOT/runtime/configs/qwen3_1_7b_rlcsd_3090ti_smoke.yaml" \
  | tee "$REMOTE_ROOT/artifacts/smoke_config.yaml"
```

If 2048/2048 OOMs, do one fallback run at 1024/1024, but keep the failed 2048/2048 log. Do not hide the OOM.

Fallback edits:

- `max_prompt_length: 1024`
- `max_completion_length: 1024`
- `val_max_completion_length: 1024`
- `actor_max_token_len_per_gpu: 2048`
- keep all other isolation settings.

## 9. Run a few RLCSD iterations

Use one GPU on 3090 Ti. Keep logs in workspace. Force detailed diagnostics on every step.

```bash
cd "$REMOTE_ROOT/repo"
source "$REMOTE_ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# background GPU monitor
(
  while true; do
    date -Is
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits
    sleep 1
  done
) > "$REMOTE_ROOT/artifacts/train_gpu_monitor.csv" 2>&1 &
GPU_MON_PID=$!
echo "$GPU_MON_PID" > "$REMOTE_ROOT/artifacts/train_gpu_monitor.pid"

set +e
bash scripts/_run_verl.sh "$REMOTE_ROOT/runtime/configs/qwen3_1_7b_rlcsd_3090ti_smoke.yaml" \
  trainer.total_training_steps=4 \
  diagnostics.detailed_first_n_steps=4 \
  diagnostics.detailed_every_n_steps=1 \
  2>&1 | tee "$REMOTE_ROOT/artifacts/rlcsd_train_stdout.txt"
TRAIN_RC=${PIPESTATUS[0]}
set -e
kill "$GPU_MON_PID" || true
echo "$TRAIN_RC" | tee "$REMOTE_ROOT/artifacts/rlcsd_train_exit_code.txt"
```

Expected outputs:

- `train_log.txt` under the run output dir.
- `metrics.jsonl` under the run output dir.
- TensorBoard events under `tensorboard_log/`.
- `config.yaml` copy under the run output dir.
- GPU monitor CSV in `$REMOTE_ROOT/artifacts/train_gpu_monitor.csv`.

Find and copy compact train artifacts:

```bash
find "$REMOTE_ROOT/runtime/outputs" -type f \
  \( -name 'metrics.jsonl' -o -name 'train_log.txt' -o -name 'config.yaml' -o -name 'events.out.tfevents*' \) \
  -printf '%p\n' | tee "$REMOTE_ROOT/artifacts/train_artifact_paths.txt"

python - <<'PY' | tee "$REMOTE_ROOT/artifacts/rlcsd_train_summary.txt"
import json
import os
from pathlib import Path

root = Path(os.environ["REMOTE_ROOT"]) / "runtime/outputs"
metrics_files = list(root.rglob("metrics.jsonl"))
print(f"metrics_files={len(metrics_files)}")
keys = [
    "training/global_step",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/teacher_log_prob",
    "timing_s/update_actor",
    "actor_timing_s/student_forward",
    "actor_timing_s/teacher_correct_forward",
    "actor_timing_s/teacher_wrong_multi_forward",
    "actor_timing_s/policy_loss",
    "actor_timing_s/backward",
    "actor_timing_s/optimizer_step",
    "timing_s/update_weights",
    "gpu/mem_allocated_gb_after_update_actor_max",
    "sys/cpu_percent",
    "proc/rss_gb",
    "rlcsd/valid_group_ratio",
    "rlcsd/effective_k_mean",
]
for path in metrics_files:
    print(f"\n== {path} ==")
    for line in path.read_text().splitlines()[-8:]:
        row = json.loads(line)
        out = {k: row.get(k) for k in keys if k in row}
        print(json.dumps(out, ensure_ascii=False))
PY
```

If training fails:

- Preserve stdout, `train_log.txt`, traceback, GPU monitor, and exit code.
- If failure is dependency import, record exact missing module and do not burn time on unrelated tuning.
- If failure is OOM, run the 1024/1024 fallback once, then continue to vLLM benchmark.
- If Ray leaves processes behind, run `ray stop --force` from the venv.

## 10. vLLM generation benchmark on real RLCSD prompts

Benchmark should use real prompts from the downloaded RLCSD parquet, not synthetic repeated prompts. Force generation to 2048 tokens for the target case with `ignore_eos=True` so output length is comparable.

Create benchmark script outside the repo tree.

```bash
cat > "$REMOTE_ROOT/tools/bench_vllm_generation.py" <<'PY'
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_text(row):
    for key in ("problem", "question", "prompt", "query"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raw = row.get("raw_prompt")
    if raw is not None:
        return raw
    raise KeyError(f"Cannot find prompt-like field. Keys={list(row.keys())}")


def build_prompt(tokenizer, item, max_prompt_tokens):
    if isinstance(item, list):
        text = tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
    elif isinstance(item, str):
        text = item
    else:
        text = str(item)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids = ids[-max_prompt_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=False), len(ids)


def llm_kwargs_for_version(kwargs):
    sig = inspect.signature(LLM)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    ap.add_argument("--backend", default="")
    ap.add_argument("--speculative-json", default="")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    if args.backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.backend

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    table = pq.read_table(args.parquet)
    rows = table.slice(0, max(args.num_prompts * 3, args.num_prompts)).to_pylist()
    prompts = []
    prompt_token_counts = []
    for row in rows:
        try:
            prompt, n_tok = build_prompt(tokenizer, extract_text(row), args.max_prompt_tokens)
        except Exception:
            continue
        prompts.append(prompt)
        prompt_token_counts.append(n_tok)
        if len(prompts) >= args.num_prompts:
            break
    if not prompts:
        raise RuntimeError("No usable prompts extracted from parquet")

    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_prompt_tokens + args.max_new_tokens,
    }
    if args.speculative_json:
        llm_kwargs.update(json.loads(args.speculative_json))
    llm = LLM(**llm_kwargs_for_version(llm_kwargs))
    sampling = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )

    records = []
    for idx in range(args.warmup + args.iters):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - started
        generated_tokens = 0
        for out_item in outputs:
            generated_tokens += len(out_item.outputs[0].token_ids)
        row = {
            "iter": idx - args.warmup,
            "warmup": idx < args.warmup,
            "backend": args.backend or os.environ.get("VLLM_ATTENTION_BACKEND", ""),
            "speculative_json": args.speculative_json,
            "num_prompts": len(prompts),
            "prompt_tokens_sum": sum(prompt_token_counts),
            "prompt_tokens_mean": sum(prompt_token_counts) / len(prompt_token_counts),
            "generated_tokens": generated_tokens,
            "elapsed_s": elapsed,
            "gen_tok_s": generated_tokens / elapsed if elapsed > 0 else None,
            "dtype": args.dtype,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
        }
        print(json.dumps(row), flush=True)
        records.append(row)

    measured = [r for r in records if not r["warmup"]]
    summary = {
        "backend": args.backend or os.environ.get("VLLM_ATTENTION_BACKEND", ""),
        "speculative_json": args.speculative_json,
        "dtype": args.dtype,
        "num_measured_iters": len(measured),
        "mean_gen_tok_s": sum(r["gen_tok_s"] for r in measured) / len(measured),
        "mean_elapsed_s": sum(r["elapsed_s"] for r in measured) / len(measured),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in measured) / len(measured),
        "mean_prompt_tokens": sum(prompt_token_counts) / len(prompt_token_counts),
    }
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
PY
```

Benchmark configs to try on 3090 Ti:

1. `FLASH_ATTN`, fp16, no speculative.
2. `FLASHINFER`, fp16, no speculative, if installed/supported.
3. `XFORMERS`, fp16, no speculative, if installed/supported.
4. n-gram/prompt-lookup speculative with whichever vLLM kwarg style the installed vLLM supports. Try both JSONs below; failed unsupported configs should be logged as skipped, not hidden.

Run loop:

```bash
cd "$REMOTE_ROOT/repo"
source "$REMOTE_ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
BENCH_DIR="$REMOTE_ROOT/artifacts/vllm_bench"
mkdir -p "$BENCH_DIR"
PARQUET="$REMOTE_ROOT/runtime/data/verl/deepmath_filtered_level5_7/train.parquet"

run_bench() {
  name="$1"; shift
  (
    while true; do
      date -Is
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits
      sleep 1
    done
  ) > "$BENCH_DIR/${name}_gpu.csv" 2>&1 &
  mon_pid=$!
  set +e
  python "$REMOTE_ROOT/tools/bench_vllm_generation.py" "$@" \
    2>&1 | tee "$BENCH_DIR/${name}.stdout.txt"
  rc=${PIPESTATUS[0]}
  set -e
  kill "$mon_pid" || true
  echo "$rc" > "$BENCH_DIR/${name}.exit_code"
}

run_bench flash_attn_fp16 \
  --model Qwen/Qwen3-1.7B \
  --parquet "$PARQUET" \
  --out "$BENCH_DIR/flash_attn_fp16.json" \
  --backend FLASH_ATTN \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --num-prompts 8 \
  --max-prompt-tokens 2048 \
  --max-new-tokens 2048 \
  --warmup 1 \
  --iters 4

run_bench flashinfer_fp16 \
  --model Qwen/Qwen3-1.7B \
  --parquet "$PARQUET" \
  --out "$BENCH_DIR/flashinfer_fp16.json" \
  --backend FLASHINFER \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --num-prompts 8 \
  --max-prompt-tokens 2048 \
  --max-new-tokens 2048 \
  --warmup 1 \
  --iters 4

run_bench xformers_fp16 \
  --model Qwen/Qwen3-1.7B \
  --parquet "$PARQUET" \
  --out "$BENCH_DIR/xformers_fp16.json" \
  --backend XFORMERS \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --num-prompts 8 \
  --max-prompt-tokens 2048 \
  --max-new-tokens 2048 \
  --warmup 1 \
  --iters 4

run_bench ngram_old_kwargs_fp16 \
  --model Qwen/Qwen3-1.7B \
  --parquet "$PARQUET" \
  --out "$BENCH_DIR/ngram_old_kwargs_fp16.json" \
  --backend FLASH_ATTN \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --num-prompts 8 \
  --max-prompt-tokens 2048 \
  --max-new-tokens 2048 \
  --warmup 1 \
  --iters 4 \
  --speculative-json '{"speculative_model":"[ngram]","num_speculative_tokens":5,"ngram_prompt_lookup_max":4,"ngram_prompt_lookup_min":2}'

run_bench ngram_spec_config_fp16 \
  --model Qwen/Qwen3-1.7B \
  --parquet "$PARQUET" \
  --out "$BENCH_DIR/ngram_spec_config_fp16.json" \
  --backend FLASH_ATTN \
  --dtype float16 \
  --gpu-memory-utilization 0.60 \
  --num-prompts 8 \
  --max-prompt-tokens 2048 \
  --max-new-tokens 2048 \
  --warmup 1 \
  --iters 4 \
  --speculative-json '{"speculative_config":{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}}'
```

If a backend/speculative mode fails:

- Keep stdout and exit code.
- Mark it as skipped/failed in the summary with the exact exception.
- Do not spend more than 20 minutes fixing one optional vLLM backend.

Summarize benchmark JSONs:

```bash
python - <<'PY' | tee "$REMOTE_ROOT/artifacts/vllm_bench_summary.jsonl"
import json
import os
from pathlib import Path

bench = Path(os.environ["REMOTE_ROOT"]) / "artifacts/vllm_bench"
for path in sorted(bench.glob("*.json")):
    try:
        obj = json.loads(path.read_text())
        row = {"name": path.stem, **obj["summary"]}
    except Exception as exc:
        row = {"name": path.stem, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(row, ensure_ascii=False))

for exit_path in sorted(bench.glob("*.exit_code")):
    print(json.dumps({"name": exit_path.name.replace(".exit_code", ""), "exit_code": exit_path.read_text().strip()}))
PY
```

## 11. Pull logs and artifacts to local

Run from local. Do not pull model cache or venv.

```bash
mkdir -p "$LOCAL_ARCHIVE"
rsync -avz \
  --include='*/' \
  --include='*.txt' \
  --include='*.log' \
  --include='*.json' \
  --include='*.jsonl' \
  --include='*.yaml' \
  --include='*.csv' \
  --include='*.patch' \
  --include='events.out.tfevents*' \
  --exclude='*' \
  "$REMOTE:$REMOTE_ROOT/artifacts/" \
  "$LOCAL_ARCHIVE/artifacts/"

rsync -avz \
  --include='*/' \
  --include='metrics.jsonl' \
  --include='train_log.txt' \
  --include='config.yaml' \
  --include='events.out.tfevents*' \
  --exclude='*' \
  "$REMOTE:$REMOTE_ROOT/runtime/outputs/" \
  "$LOCAL_ARCHIVE/train_outputs/"
```

Local quick parse:

```bash
find "$LOCAL_ARCHIVE" -type f | sort > "$LOCAL_ARCHIVE/local_manifest.txt"

python - <<'PY'
import json
import os
from pathlib import Path

archive = Path(os.environ["LOCAL_ARCHIVE"])
for mf in archive.rglob("metrics.jsonl"):
    print(f"\n== {mf} ==")
    for line in mf.read_text().splitlines()[-8:]:
        row = json.loads(line)
        keys = [
            "training/global_step",
            "timing_s/step",
            "timing_s/gen",
            "timing_s/old_log_prob",
            "timing_s/update_actor",
            "actor_timing_s/student_forward",
            "actor_timing_s/teacher_correct_forward",
            "actor_timing_s/teacher_wrong_multi_forward",
            "actor_timing_s/backward",
            "timing_s/update_weights",
            "sys/cpu_percent",
            "proc/rss_gb",
        ]
        print(json.dumps({k: row.get(k) for k in keys if k in row}, ensure_ascii=False))

summary = archive / "artifacts/vllm_bench_summary.jsonl"
if summary.exists():
    print(f"\n== {summary} ==")
    print(summary.read_text())
PY
```

## 12. Cleanup remote

Run only after artifacts are pulled and verified locally.

Remote cleanup:

```bash
set -euxo pipefail
safe_rm() {
  target="$1"
  case "$target" in
    "$REMOTE_ROOT"/*|"$REMOTE_ROOT") rm -rf "$target" ;;
    *) echo "REFUSE rm outside REMOTE_ROOT: $target" >&2; return 2 ;;
  esac
}
source "$REMOTE_ROOT/.venv/bin/activate" || true
ray stop --force || true
deactivate || true

{
  date -Is
  nvidia-smi || true
  ps -fu "$USER" | grep -E "$REMOTE_ROOT|ray|vllm|python" || true
  df -h
} | tee "$REMOTE_ROOT/artifacts/remote_state_before_cleanup.txt"
```

Pull `remote_state_before_cleanup.txt` to local before deletion:

```bash
rsync -avz "$REMOTE:$REMOTE_ROOT/artifacts/remote_state_before_cleanup.txt" "$LOCAL_ARCHIVE/artifacts/"
```

Then delete only the isolated workspace:

```bash
safe_rm "$REMOTE_ROOT"
```

Verify no task-owned paths/processes remain:

```bash
test ! -e "$REMOTE_ROOT" && echo "workspace removed"
ps -fu "$USER" | grep "$RUN_ID" || true
nvidia-smi || true
```

Save the final local note:

```bash
cat > "$LOCAL_ARCHIVE/final_remote_cleanup_note.txt" <<EOF
Remote workspace requested for deletion: $REMOTE_ROOT
Deletion verified locally by agent at: $(date -Is)
Remaining remote state should be checked with: ssh $REMOTE 'test ! -e "$REMOTE_ROOT" && echo removed'
EOF
```

## 13. Required final report

Final report must include:

- Remote hostname, GPU model, driver/CUDA from `nvidia-smi`.
- Repo commit used.
- Workspace path created and confirmation it was removed.
- Exact files pulled to local.
- Training status:
  - exit code
  - number of completed steps
  - whether 2048/2048 succeeded
  - fallback used or not
  - last-step coarse timing table
  - actor detailed timing table if present
  - peak GPU memory from CSV
  - CPU/RAM metrics from `metrics.jsonl`
- vLLM benchmark table:
  - config name
  - backend
  - dtype
  - speculative config
  - mean prompt tokens
  - generated tokens
  - mean elapsed seconds
  - mean generated tok/s
  - peak GPU memory
  - exit status/failure reason
- Any remote hotfix patch path, or `N/A`.
- Any server path outside `$REMOTE_ROOT` touched, expected answer: `N/A`.
- Cleanup confirmation.

## 14. Decision rules while running

Use these rules instead of guessing:

- If dependency import fails, fix dependencies in the workspace venv only.
- If installing dependencies risks system/global changes, stop.
- If model/data download wants to write outside `$REMOTE_ROOT`, stop.
- If 2048/2048 training OOMs, preserve logs and run one 1024/1024 fallback.
- If a vLLM backend is unsupported, mark it failed and continue to the next config.
- If a speculative config is unsupported by installed vLLM, keep the exception and continue.
- If the server appears shared or busy, reduce benchmark prompts/iters before disrupting other users.
- If cleanup cannot prove `$REMOTE_ROOT` was removed, report that as a blocker.

## 15. Minimal success criteria

Acceptable completion within 5 hours:

- Local archive contains remote preflight, command/session logs, train stdout or failure log, vLLM benchmark logs, GPU monitor CSVs, and cleanup note.
- At least one of these succeeded:
  - RLCSD smoke run produced `metrics.jsonl` with at least 2 train steps.
  - vLLM baseline `FLASH_ATTN` fp16 benchmark produced JSON summary on real RLCSD prompts.
- No writes outside the isolated remote workspace.
- Remote workspace removed after pulling logs.
