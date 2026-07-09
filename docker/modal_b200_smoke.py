import os
import subprocess

import modal


IMAGE = os.environ.get(
    "RLCSD_IMAGE",
    "codemaivanngu/rlcsd:b200-cu13-vllm024-sm100-sm120",
)
GPU = os.environ.get("RLCSD_MODAL_GPU", "B200")
FORCE_BUILD = os.environ.get("RLCSD_MODAL_FORCE_BUILD", "0").lower() in {
    "1",
    "true",
    "yes",
}

image = modal.Image.from_registry(IMAGE, force_build=FORCE_BUILD).entrypoint([])
app = modal.App("rlcsd-b200-container-smoke", image=image)


@app.function(gpu=GPU, timeout=60 * 30)
def smoke() -> None:
    subprocess.run(
        ["python3", "/workspace/RLCSD/docker/smoke_test.py"],
        check=True,
    )


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
        ["bash", "/workspace/RLCSD/docker/run_mini_epoch.sh"],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    task = os.environ.get("RLCSD_MODAL_TASK", "smoke")
    if task == "smoke":
        smoke.remote()
    elif task == "mini_epoch":
        mini_epoch.remote()
    else:
        raise SystemExit(f"Unsupported RLCSD_MODAL_TASK={task!r}")
