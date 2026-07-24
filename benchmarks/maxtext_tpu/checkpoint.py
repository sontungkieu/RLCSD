"""HF-to-MaxText checkpoint conversion with a provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.maxtext_tpu.matrix import get_case


CPU_ONLY_ENV_OVERRIDES = {
    "JAX_PLATFORMS": "cpu",
    "PJRT_DEVICE": "CPU",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONUNBUFFERED": "1",
}

TPU_RUNTIME_ENV_KEYS = {
    "COLAB_TPU_ADDR",
    "CLOUD_TPU_TASK_ID",
    "TPU_ACCELERATOR_TYPE",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_ML_PLATFORM",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_BOUNDS",
    "TPU_VISIBLE_CHIPS",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "XRT_TPU_CONFIG",
}


def _cpu_only_environment(
    base_environment: dict[str, str] | None = None,
    *,
    simulated_cpu_devices: int = 16,
) -> dict[str, str]:
    """Return an environment that cannot attach checkpoint conversion to TPU.

    MaxText imports JAX before its CLI sets ``XLA_FLAGS``. Supplying the
    simulated CPU device count here ensures XLA observes it at process start.
    TPU discovery variables are removed so TensorFlow, JAX, or torch-xla
    imports in the conversion stack cannot attach to the live TPU runtime.
    """

    if simulated_cpu_devices < 1:
        raise ValueError("simulated_cpu_devices must be positive")

    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    for key in TPU_RUNTIME_ENV_KEYS:
        environment.pop(key, None)

    device_flag_prefix = "--xla_force_host_platform_device_count="
    xla_flags = [
        token
        for token in shlex.split(environment.get("XLA_FLAGS", ""))
        if not token.startswith(device_flag_prefix)
    ]
    xla_flags.append(f"{device_flag_prefix}{simulated_cpu_devices}")
    environment["XLA_FLAGS"] = shlex.join(xla_flags)
    environment.update(CPU_ONLY_ENV_OVERRIDES)
    return environment


def _resolve_revision(model_id: str, requested_revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve revision for {model_id}.")
    return info.sha


def _maxtext_base_config() -> Path:
    import maxtext

    path = Path(maxtext.__file__).resolve().parent / "configs" / "base.yml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Installed MaxText base config was not found at {path}."
        )
    return path


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        total_bytes += size
    return digest.hexdigest(), file_count, total_bytes


def main() -> None:
    # Importing MaxText can initialize JAX. Apply this before resolving the
    # installed MaxText config so the outer CLI process stays on CPU as well as
    # the actual conversion subprocess.
    os.environ.update(CPU_ONLY_ENV_OVERRIDES)

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--simulated-cpu-devices", type=int, default=16)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Audit an existing output instead of running conversion.",
    )
    args = parser.parse_args()

    case = get_case(args.case)
    output_dir = args.output_dir.resolve()
    items_dir = output_dir / "0" / "items"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.reuse:
        raise FileExistsError(
            f"{output_dir} is not empty; use a fresh path or --reuse."
        )

    resolved_revision = _resolve_revision(case.model.model_id, args.hf_revision)
    started = time.perf_counter()
    command: list[str] = []
    conversion_env = _cpu_only_environment(
        simulated_cpu_devices=args.simulated_cpu_devices
    )
    if not args.reuse:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "maxtext.checkpoint_conversion.to_maxtext",
            str(_maxtext_base_config()),
            f"model_name={case.model.model_id.rsplit('/', 1)[-1].lower()}",
            f"base_output_directory={output_dir}",
            "hardware=cpu",
            "scan_layers=false",
            "use_multimodal=false",
            "skip_jax_distributed_system=true",
            "--lazy_load_tensors=true",
            f"--simulated_cpu_devices_count={args.simulated_cpu_devices}",
            f"--hf_model_path={case.model.model_id}",
            f"--revision={resolved_revision}",
            "--save_dtype=bfloat16",
        ]
        # Authentication remains in the environment; it is never serialized.
        subprocess.run(command, env=conversion_env, check=True)

    if not items_dir.is_dir():
        raise FileNotFoundError(
            f"Converted Orbax items directory was not found: {items_dir}"
        )
    tree_sha256, file_count, total_bytes = _tree_digest(items_dir)
    manifest = {
        "schema_version": 1,
        "kind": "hf_to_maxtext_checkpoint",
        "source_case_id": case.case_id,
        "model_key": case.model.key,
        "model_id": case.model.model_id,
        "requested_hf_revision": args.hf_revision,
        "resolved_hf_revision": resolved_revision,
        "checkpoint_items_path": str(items_dir),
        "scan_layers": False,
        "save_dtype": "bfloat16",
        "simulated_cpu_devices": args.simulated_cpu_devices,
        "elapsed_s": time.perf_counter() - started,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_metadata_sha256": tree_sha256,
        "reused_existing_output": bool(args.reuse),
        "command": command,
        "runtime_environment": {
            "jax_platforms": conversion_env.get("JAX_PLATFORMS", "cpu"),
            "pjrt_device": conversion_env.get("PJRT_DEVICE", "CPU"),
            "xla_flags": conversion_env["XLA_FLAGS"],
            "tpu_runtime_keys_present": sorted(
                key
                for key in TPU_RUNTIME_ENV_KEYS
                if key in conversion_env
            ),
        },
        "contains_secret": False,
    }
    manifest_path = output_dir / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
