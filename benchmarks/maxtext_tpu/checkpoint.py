"""HF-to-MaxText checkpoint conversion with a provenance manifest."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

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

LIBTPU_IMPORT_BLOCKER_SOURCE = """\
raise ImportError(
    "libtpu is intentionally unavailable during CPU-only checkpoint conversion"
)
"""


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


def _activate_cpu_only_environment(
    *, simulated_cpu_devices: int
) -> dict[str, str]:
    """Apply CPU isolation before this CLI imports MaxText or JAX."""

    environment = _cpu_only_environment(
        simulated_cpu_devices=simulated_cpu_devices
    )
    for key in TPU_RUNTIME_ENV_KEYS:
        os.environ.pop(key, None)
    for key in (*CPU_ONLY_ENV_OVERRIDES, "XLA_FLAGS"):
        os.environ[key] = environment[key]
    return environment


@contextlib.contextmanager
def _block_libtpu_import(
    environment: dict[str, str],
) -> Iterator[dict[str, str]]:
    """Hide installed libtpu from the CPU-only conversion process.

    Recent JAX versions discover Cloud TPU from the device plus the installed
    ``libtpu`` module, even after TPU environment variables are removed and
    ``JAX_PLATFORMS=cpu`` is set.  A leading import blocker makes JAX's
    optional ``import libtpu`` take its supported ImportError path without
    changing the TPU environment used by later parity and training processes.
    """

    if "libtpu" in sys.modules:
        raise RuntimeError(
            "libtpu was imported before CPU checkpoint isolation"
        )

    with tempfile.TemporaryDirectory(prefix="rlcsd-cpu-libtpu-block-") as temp:
        blocker_dir = Path(temp)
        (blocker_dir / "libtpu.py").write_text(
            LIBTPU_IMPORT_BLOCKER_SOURCE,
            encoding="utf-8",
        )
        blocker_path = str(blocker_dir)
        child_environment = dict(environment)
        inherited_pythonpath = child_environment.get("PYTHONPATH", "")
        child_environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (blocker_path, inherited_pythonpath)
            if part
        )
        child_environment["RLCSD_LIBTPU_IMPORT_BLOCKED"] = "1"
        sys.path.insert(0, blocker_path)
        try:
            yield child_environment
        finally:
            if blocker_path in sys.path:
                sys.path.remove(blocker_path)


def _resolve_revision(model_id: str, requested_revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve revision for {model_id}.")
    return info.sha


def _emit_conversion_milestone(name: str, **details: object) -> None:
    payload = {"milestone": name, **details}
    print(
        "RLCSD_CHECKPOINT_CONVERSION_MILESTONE "
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )


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

    # Importing MaxText can initialize JAX. Apply the complete isolation,
    # including XLA's simulated device flag and removal of TPU discovery
    # variables, before resolving the installed MaxText config.
    conversion_env = _activate_cpu_only_environment(
        simulated_cpu_devices=args.simulated_cpu_devices
    )
    _emit_conversion_milestone(
        "cpu_environment_activated",
        jax_platforms=conversion_env["JAX_PLATFORMS"],
        pjrt_device=conversion_env["PJRT_DEVICE"],
        simulated_cpu_devices=args.simulated_cpu_devices,
    )

    case = get_case(args.case)
    output_dir = args.output_dir.resolve()
    items_dir = output_dir / "0" / "items"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.reuse:
        raise FileExistsError(
            f"{output_dir} is not empty; use a fresh path or --reuse."
        )

    resolved_revision = _resolve_revision(case.model.model_id, args.hf_revision)
    _emit_conversion_milestone(
        "hf_revision_resolved",
        model_id=case.model.model_id,
        resolved_revision=resolved_revision,
    )
    started = time.perf_counter()
    command: list[str] = []
    if not args.reuse:
        output_dir.mkdir(parents=True, exist_ok=True)
        with _block_libtpu_import(conversion_env) as blocked_env:
            _emit_conversion_milestone(
                "libtpu_import_blocked",
                blocker_enabled=(
                    blocked_env.get("RLCSD_LIBTPU_IMPORT_BLOCKED") == "1"
                ),
            )
            base_config = _maxtext_base_config()
            _emit_conversion_milestone(
                "maxtext_base_config_resolved",
                base_config=str(base_config),
            )
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "maxtext.checkpoint_conversion.to_maxtext",
                str(base_config),
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
            _emit_conversion_milestone(
                "conversion_subprocess_starting",
                child_libtpu_import_blocked=(
                    blocked_env.get("RLCSD_LIBTPU_IMPORT_BLOCKED") == "1"
                ),
                command_module="maxtext.checkpoint_conversion.to_maxtext",
            )
            # Authentication remains in the environment; it is never
            # serialized. The libtpu blocker is scoped to conversion only.
            subprocess.run(command, env=blocked_env, check=True)

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
            "libtpu_import_blocked": not args.reuse,
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
