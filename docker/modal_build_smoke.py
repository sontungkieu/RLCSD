import os
import subprocess
import time

import modal


BUILD_ARGS = {
    "FLASH_ATTN_CUDA_ARCHS": os.environ.get("FLASH_ATTN_CUDA_ARCHS", "100;120"),
    "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "10.0;12.0+PTX"),
    "CUDAARCHS": os.environ.get("CUDAARCHS", "100;120"),
    "CMAKE_CUDA_ARCHITECTURES": os.environ.get("CMAKE_CUDA_ARCHITECTURES", "100;120"),
    "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
    "INSTALL_FLASH_ATTN": os.environ.get("INSTALL_FLASH_ATTN", "0"),
}

GPU = os.environ.get("RLCSD_MODAL_GPU", "B200")
IMAGE_ID = os.environ.get("RLCSD_MODAL_IMAGE_ID")

if IMAGE_ID:
    image = (
        modal.Image.from_id(IMAGE_ID)
        .add_local_file(
            "src/w4_self_speculative.py",
            "/workspace/RLCSD/src/w4_self_speculative.py",
        )
        .add_local_file(
            "scripts/_run_verl.sh",
            "/workspace/RLCSD/scripts/_run_verl.sh",
        )
        .add_local_file(
            "configs/smoke/qwen3_1_7b_rlcsd_w4_1step.yaml",
            "/workspace/RLCSD/configs/smoke/qwen3_1_7b_rlcsd_w4_1step.yaml",
        )
        .add_local_file(
            "third_party/verl/verl/workers/rollout/vllm_rollout/utils.py",
            "/workspace/RLCSD/third_party/verl/verl/workers/rollout/vllm_rollout/utils.py",
        )
    )
else:
    image = modal.Image.from_dockerfile(
        "docker/Dockerfile.rlc-runtime",
        context_dir=".",
        build_args=BUILD_ARGS,
    )
app = modal.App("rlcsd-dockerfile-build-smoke", image=image)
hf_cache = modal.Volume.from_name("rlcsd-hf-cache", create_if_missing=True)


def _refresh_w4_drafter(worker) -> float:
    drafter = worker.model_runner.drafter
    return drafter.refresh_from_target()


def _actor_update_completed(metric_row: dict[str, object]) -> bool:
    explicit_skip = metric_row.get("actor/skip_update")
    if explicit_skip is not None:
        return float(explicit_skip) == 0.0

    positive_metrics = (
        "actor/grad_norm",
        "actor_timing_s/backward",
        "actor_timing_s/optimizer_step",
        "timing_s/update_actor",
        "rlcsd/valid_group_ratio",
    )
    return all(
        key in metric_row and float(metric_row[key]) > 0.0 for key in positive_metrics
    )


@app.function(gpu=GPU, timeout=60 * 30)
def smoke() -> None:
    subprocess.run(["python3", "/workspace/RLCSD/docker/smoke_test.py"], check=True)


@app.function(gpu=GPU, timeout=60 * 90)
def mini_epoch() -> None:
    model_path = "/tmp/tiny-qwen2-random"
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/hf")
    subprocess.run(
        [
            "python3",
            "/workspace/RLCSD/docker/make_tiny_qwen2_model.py",
            "--out",
            model_path,
        ],
        check=True,
        env=env,
    )
    env["RLCSD_SMOKE_MODEL"] = model_path
    subprocess.run(
        ["bash", "/workspace/RLCSD/docker/run_mini_epoch.sh"], check=True, env=env
    )


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def w4_probe() -> None:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import torch
    from vllm import LLM, SamplingParams

    model = os.environ.get("RLCSD_W4_PROBE_MODEL", "Qwen/Qwen3-1.7B")
    os.environ.setdefault("RLCSD_W4_GROUP_SIZE", "128")
    started = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        speculative_config={
            "method": "custom_class",
            "model": "src.w4_self_speculative.W4SelfSpeculativeProposer",
            "num_speculative_tokens": 2,
        },
    )
    loaded_seconds = time.perf_counter() - started
    prompts = [
        "Solve 17 * 23 step by step.",
        "Find the derivative of x^3 + 2x.",
        "If a triangle has sides 3, 4, 5, explain its area.",
        "Compute the sum of integers from 1 through 100.",
    ]
    generated_at = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    generation_seconds = time.perf_counter() - generated_at
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    refreshed_at = time.perf_counter()
    refresh_seconds = llm.collective_rpc(_refresh_w4_drafter)[0]
    refresh_rpc_seconds = time.perf_counter() - refreshed_at
    regenerated_at = time.perf_counter()
    refreshed_outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    regenerated_seconds = time.perf_counter() - regenerated_at
    refreshed_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in refreshed_outputs
    )
    print(
        "RLCSD_W4_PROBE="
        f"model={model} gpu={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)} "
        f"load_s={loaded_seconds:.3f} generation_s={generation_seconds:.3f} "
        f"output_tokens={output_tokens} refresh_s={refresh_seconds:.3f} "
        f"refresh_rpc_s={refresh_rpc_seconds:.3f} "
        f"regeneration_s={regenerated_seconds:.3f} "
        f"refreshed_output_tokens={refreshed_output_tokens}",
        flush=True,
    )


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def w4_train_smoke() -> None:
    import json

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/.cache/huggingface")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RAY_DEDUP_LOGS", "0")
    root = "/workspace/RLCSD"
    subprocess.run(
        [
            "python3",
            "docker/make_smoke_data.py",
            "--out",
            "/tmp/rlcsd-smoke-w4/data",
        ],
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "bash",
            "scripts/_run_verl.sh",
            "configs/smoke/qwen3_1_7b_rlcsd_w4_1step.yaml",
            "trainer.total_training_steps=1",
            "trainer.default_hdfs_dir=null",
            "+trainer.remove_previous_ckpt_in_save=True",
        ],
        cwd=root,
        env=env,
        check=True,
    )

    from pathlib import Path

    output_root = Path("/tmp/rlcsd-smoke-w4/outputs")
    logs = sorted(
        output_root.rglob("train_log.txt"), key=lambda path: path.stat().st_mtime
    )
    metrics = sorted(
        output_root.rglob("metrics.jsonl"), key=lambda path: path.stat().st_mtime
    )
    if not logs or not metrics:
        raise RuntimeError(
            "W4 training smoke did not produce train_log.txt and metrics.jsonl"
        )
    log_lines = logs[-1].read_text(errors="replace").splitlines()
    refresh_lines = [line for line in log_lines if "RLCSD_W4_REFRESH" in line]
    if len(refresh_lines) < 2:
        raise RuntimeError(
            "W4 training smoke requires initial and post-update drafter refreshes; "
            f"found {len(refresh_lines)} marker(s)"
        )
    metric_rows = [
        json.loads(line)
        for line in metrics[-1].read_text().splitlines()
        if line.strip()
    ]
    if not metric_rows:
        raise RuntimeError("W4 training smoke produced an empty metrics.jsonl")
    if not _actor_update_completed(metric_rows[-1]):
        raise RuntimeError(
            "W4 training smoke has no evidence of a completed actor update"
        )
    print("RLCSD_W4_TRAIN_REFRESH_LINES=" + " | ".join(refresh_lines), flush=True)
    print("RLCSD_W4_TRAIN_METRICS_JSONL_BEGIN", flush=True)
    print(metrics[-1].read_text(), flush=True)
    print("RLCSD_W4_TRAIN_METRICS_JSONL_END", flush=True)
    print("RLCSD_W4_TRAIN_SMOKE_OK", flush=True)


if __name__ == "__main__":
    task = os.environ.get("RLCSD_MODAL_TASK", "smoke")
    if task == "smoke":
        smoke.remote()
    elif task == "mini_epoch":
        mini_epoch.remote()
    elif task == "w4_probe":
        w4_probe.remote()
    elif task == "w4_train_smoke":
        w4_train_smoke.remote()
    else:
        raise SystemExit(f"Unsupported RLCSD_MODAL_TASK={task!r}")
