import importlib
import os
import subprocess
import sys


def check_import(module_name: str, required: bool = True) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        if required:
            raise
        print(f"{module_name}=optional-missing ({exc.__class__.__name__}: {exc})")
        return
    version = getattr(module, "__version__", "ok")
    print(f"{module_name}={version}")


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '')}")
    print(f"CUDAARCHS={os.environ.get('CUDAARCHS', '')}")

    subprocess.run(["nvidia-smi"], check=True)

    import torch

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device("cuda")
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print(f"gpu_capability={torch.cuda.get_device_capability(0)}")
    x = torch.randn((256, 256), device=device, dtype=torch.float16)
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"matmul_checksum={float(y.float().mean().cpu()):.6f}")

    required_modules = [
        "vllm",
        "transformers",
        "ray",
        "deepspeed",
        "hydra",
        "datasets",
        "peft",
        "tensordict",
        "verl",
    ]
    for module_name in required_modules:
        check_import(module_name)
    check_import("flash_attn", required=False)

    print("RLCSD_SMOKE_OK")


if __name__ == "__main__":
    main()
