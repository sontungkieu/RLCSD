import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal


FLASH_ATTN_VERSION = os.environ.get("FLASH_ATTN_VERSION", "2.8.3.post1")
FLASH_ATTN_CUDA_ARCHS = os.environ.get("FLASH_ATTN_CUDA_ARCHS", "100")
TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", "10.0")
CUDAARCHS = os.environ.get("CUDAARCHS", "100")
CMAKE_CUDA_ARCHITECTURES = os.environ.get("CMAKE_CUDA_ARCHITECTURES", "100")
MAX_JOBS = os.environ.get("MAX_JOBS", "8")
NVCC_THREADS = os.environ.get("NVCC_THREADS", "1")
RUN_IMPORT_CHECK = os.environ.get("RLCSD_FLASH_ATTN_IMPORT_CHECK", "1") not in {
    "0",
    "false",
    "False",
}

GPU = os.environ.get("RLCSD_MODAL_WHEEL_GPU", "").strip() or None
CPU = float(os.environ.get("RLCSD_MODAL_WHEEL_CPU", "16"))
MEMORY = int(os.environ.get("RLCSD_MODAL_WHEEL_MEMORY", "65536"))
TIMEOUT = int(os.environ.get("RLCSD_MODAL_WHEEL_TIMEOUT", str(4 * 60 * 60)))
VOLUME_NAME = os.environ.get("RLCSD_FLASH_ATTN_VOLUME", "rlcsd-flash-attn-wheelhouse")

BUILD_ARGS = {
    "FLASH_ATTN_CUDA_ARCHS": FLASH_ATTN_CUDA_ARCHS,
    "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST,
    "CUDAARCHS": CUDAARCHS,
    "CMAKE_CUDA_ARCHITECTURES": CMAKE_CUDA_ARCHITECTURES,
    "MAX_JOBS": MAX_JOBS,
    "INSTALL_FLASH_ATTN": "0",
}

image = modal.Image.from_dockerfile(
    "docker/Dockerfile.rlc-runtime",
    context_dir=".",
    build_args=BUILD_ARGS,
)
wheel_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("rlcsd-flash-attn-wheel-build")

function_kwargs = {
    "image": image,
    "cpu": CPU,
    "memory": MEMORY,
    "timeout": TIMEOUT,
    "volumes": {"/wheelhouse": wheel_volume},
}
if GPU:
    function_kwargs["gpu"] = GPU


def _run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


@app.function(**function_kwargs)
def build_wheel(
    flash_attn_version: str = FLASH_ATTN_VERSION,
    flash_attn_cuda_archs: str = FLASH_ATTN_CUDA_ARCHS,
    torch_cuda_arch_list: str = TORCH_CUDA_ARCH_LIST,
    cudaarchs: str = CUDAARCHS,
    cmake_cuda_architectures: str = CMAKE_CUDA_ARCHITECTURES,
    max_jobs: str = MAX_JOBS,
    nvcc_threads: str = NVCC_THREADS,
    run_import_check: bool = RUN_IMPORT_CHECK,
    builder_gpu: str | None = GPU,
    builder_cpu: float = CPU,
    builder_memory: int = MEMORY,
) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("/wheelhouse") / "runs" / run_id
    latest_dir = Path("/wheelhouse") / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "build.log"

    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": f"/usr/local/cuda/bin:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}",
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTN_CUDA_ARCHS": flash_attn_cuda_archs,
            "TORCH_CUDA_ARCH_LIST": torch_cuda_arch_list,
            "CUDAARCHS": cudaarchs,
            "CMAKE_CUDA_ARCHITECTURES": cmake_cuda_architectures,
            "MAX_JOBS": max_jobs,
            "NVCC_THREADS": nvcc_threads,
            "PIP_NO_CACHE_DIR": "0",
        }
    )

    import torch

    manifest = {
        "run_id": run_id,
        "package": f"flash-attn=={flash_attn_version}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "flash_attn_cuda_archs": flash_attn_cuda_archs,
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "cudaarchs": cudaarchs,
        "cmake_cuda_architectures": cmake_cuda_architectures,
        "max_jobs": max_jobs,
        "nvcc_threads": nvcc_threads,
        "gpu": builder_gpu,
        "cpu": builder_cpu,
        "memory_mb": builder_memory,
        "run_import_check": run_import_check,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }

    start = time.monotonic()
    wheel_cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "-v",
        "--no-build-isolation",
        "--no-deps",
        "--wheel-dir",
        str(run_dir),
        f"flash-attn=={flash_attn_version}",
    ]

    try:
        rc = _run_logged(wheel_cmd, env, log_path)
        manifest["pip_wheel_returncode"] = rc
        wheels = sorted(str(path) for path in run_dir.glob("flash_attn*.whl"))
        manifest["wheels"] = wheels

        if rc != 0:
            manifest["status"] = "failed"
            raise RuntimeError(f"pip wheel failed with return code {rc}")
        if not wheels:
            manifest["status"] = "failed"
            raise RuntimeError("pip wheel succeeded but produced no flash_attn*.whl")

        if run_import_check:
            install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                wheels[-1],
            ]
            import_cmd = [
                sys.executable,
                "-c",
                "import flash_attn; print('flash_attn_import_ok', getattr(flash_attn, '__version__', 'unknown'))",
            ]
            install_rc = _run_logged(install_cmd, env, log_path)
            import_rc = _run_logged(import_cmd, env, log_path)
            manifest["import_check"] = {
                "install_returncode": install_rc,
                "import_returncode": import_rc,
            }
            if install_rc != 0 or import_rc != 0:
                manifest["status"] = "failed"
                raise RuntimeError("flash-attn wheel import check failed")

        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
        for wheel in wheels:
            shutil.copy2(wheel, latest_dir / Path(wheel).name)

        manifest["status"] = "success"
        manifest["latest_dir"] = str(latest_dir)
        return manifest
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_s"] = round(time.monotonic() - start, 3)
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if latest_dir.exists():
            shutil.copy2(manifest_path, latest_dir / "manifest.json")
            shutil.copy2(log_path, latest_dir / "build.log")
        wheel_volume.commit()
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


@app.function(image=image, volumes={"/wheelhouse": wheel_volume}, timeout=5 * 60)
def list_wheels() -> list[str]:
    root = Path("/wheelhouse")
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )
