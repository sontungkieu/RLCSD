import os
import subprocess

import modal


BUILD_ARGS = {
    "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "10.0;12.0+PTX"),
    "CUDAARCHS": os.environ.get("CUDAARCHS", "100;120"),
    "CMAKE_CUDA_ARCHITECTURES": os.environ.get("CMAKE_CUDA_ARCHITECTURES", "100;120"),
    "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
    "INSTALL_FLASH_ATTN": os.environ.get("INSTALL_FLASH_ATTN", "0"),
}

GPU = os.environ.get("RLCSD_MODAL_GPU", "B200")

image = modal.Image.from_dockerfile(
    "docker/Dockerfile.rlc-runtime",
    context_dir=".",
    build_args=BUILD_ARGS,
)
app = modal.App("rlcsd-dockerfile-build-smoke", image=image)


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
    subprocess.run(["bash", "/workspace/RLCSD/docker/run_mini_epoch.sh"], check=True, env=env)


if __name__ == "__main__":
    task = os.environ.get("RLCSD_MODAL_TASK", "smoke")
    if task == "smoke":
        smoke.remote()
    elif task == "mini_epoch":
        mini_epoch.remote()
    else:
        raise SystemExit(f"Unsupported RLCSD_MODAL_TASK={task!r}")
