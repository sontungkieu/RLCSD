from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

from benchmarks.maxtext_tpu import checkpoint


def test_checkpoint_cli_forces_outer_process_to_cpu_before_resolution(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "checkpoint"
    (output_dir / "0" / "items").mkdir(parents=True)
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-8")
    monkeypatch.setenv("XRT_TPU_CONFIG", "localservice;0;localhost:51011")
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_dump_to=/tmp/xla --xla_force_host_platform_device_count=2"
    )

    def resolve_revision(model_id, requested_revision):
        assert model_id == "Qwen/Qwen3-1.7B"
        assert requested_revision == "main"
        assert checkpoint.os.environ["JAX_PLATFORMS"] == "cpu"
        assert checkpoint.os.environ["PJRT_DEVICE"] == "CPU"
        assert checkpoint.os.environ["XLA_FLAGS"] == (
            "--xla_dump_to=/tmp/xla --xla_force_host_platform_device_count=16"
        )
        assert not any(
            key in checkpoint.os.environ
            for key in checkpoint.TPU_RUNTIME_ENV_KEYS
        )
        return "a" * 40

    monkeypatch.setattr(checkpoint, "_resolve_revision", resolve_revision)
    monkeypatch.setattr(
        checkpoint.sys,
        "argv",
        [
            "checkpoint",
            "--case",
            "qwen3_1_7b-bs12-pp2xtp4",
            "--output-dir",
            str(output_dir),
            "--reuse",
        ],
    )

    checkpoint.main()

    manifest = json.loads(
        (output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    runtime_environment = manifest["runtime_environment"]
    assert runtime_environment["jax_platforms"] == "cpu"
    assert runtime_environment["pjrt_device"] == "CPU"
    assert runtime_environment["xla_flags"].endswith(
        "--xla_force_host_platform_device_count=16"
    )
    assert runtime_environment["libtpu_import_blocked"] is False
    assert runtime_environment["tpu_runtime_keys_present"] == []


def test_checkpoint_cpu_environment_overrides_inherited_tpu_without_mutation():
    inherited = {
        "JAX_PLATFORMS": "tpu",
        "PJRT_DEVICE": "TPU",
        "TPU_ACCELERATOR_TYPE": "v5litepod-8",
        "TPU_WORKER_HOSTNAMES": "localhost",
        "XLA_FLAGS": "--xla_dump_to=/tmp/xla --xla_force_host_platform_device_count=2",
        "XRT_TPU_CONFIG": "localservice;0;localhost:51011",
        "HF_HOME": "/tmp/hf",
    }
    isolated = checkpoint._cpu_only_environment(
        inherited, simulated_cpu_devices=8
    )

    assert isolated["JAX_PLATFORMS"] == "cpu"
    assert isolated["PJRT_DEVICE"] == "CPU"
    assert isolated["PYTHONFAULTHANDLER"] == "1"
    assert isolated["PYTHONUNBUFFERED"] == "1"
    assert isolated["HF_HOME"] == "/tmp/hf"
    assert isolated["XLA_FLAGS"] == (
        "--xla_dump_to=/tmp/xla --xla_force_host_platform_device_count=8"
    )
    assert "TPU_ACCELERATOR_TYPE" not in isolated
    assert "TPU_WORKER_HOSTNAMES" not in isolated
    assert "XRT_TPU_CONFIG" not in isolated
    assert inherited["JAX_PLATFORMS"] == "tpu"
    assert inherited["PJRT_DEVICE"] == "TPU"
    assert inherited["TPU_ACCELERATOR_TYPE"] == "v5litepod-8"
    assert inherited["XRT_TPU_CONFIG"] == "localservice;0;localhost:51011"


def test_libtpu_import_blocker_precedes_installed_module(
    monkeypatch, tmp_path
):
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "libtpu.py").write_text(
        "SENTINEL = 'installed-libtpu'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(installed))
    environment = {"PYTHONPATH": str(installed)}

    with checkpoint._block_libtpu_import(environment) as blocked:
        blocker = blocked["PYTHONPATH"].split(os.pathsep)[0]
        assert blocker != str(installed)
        assert blocked["RLCSD_LIBTPU_IMPORT_BLOCKED"] == "1"
        assert sys.path[0] == blocker
        with pytest.raises(ImportError, match="CPU-only checkpoint"):
            importlib.import_module("libtpu")

    assert blocker not in sys.path
    sys.modules.pop("libtpu", None)
    assert importlib.import_module("libtpu").SENTINEL == "installed-libtpu"
    sys.modules.pop("libtpu", None)


def test_maxtext_base_config_discovery_does_not_execute_package(
    monkeypatch, tmp_path
):
    package = tmp_path / "maxtext"
    config = package / "configs" / "base.yml"
    config.parent.mkdir(parents=True)
    config.write_text("base_config: true\n", encoding="utf-8")
    (package / "__init__.py").write_text(
        "raise RuntimeError('maxtext package executed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("maxtext", None)

    assert checkpoint._maxtext_base_config() == config.resolve()
    assert "maxtext" not in sys.modules


def test_checkpoint_conversion_command_declares_cpu_hardware(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "checkpoint"
    items_dir = output_dir / "0" / "items"
    captured = {}

    monkeypatch.setattr(checkpoint, "_resolve_revision", lambda *_: "b" * 40)
    monkeypatch.setattr(
        checkpoint, "_maxtext_base_config", lambda: tmp_path / "base.yml"
    )

    def run(command, *, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        items_dir.mkdir(parents=True)

    monkeypatch.setattr(checkpoint.subprocess, "run", run)
    monkeypatch.setattr(
        checkpoint.sys,
        "argv",
        [
            "checkpoint",
            "--case",
            "qwen3_1_7b-bs12-pp2xtp4",
            "--output-dir",
            str(output_dir),
            "--simulated-cpu-devices",
            "8",
        ],
    )

    checkpoint.main()

    assert captured["check"] is True
    assert captured["command"][:4] == [
        checkpoint.sys.executable,
        "-X",
        "faulthandler",
        "-m",
    ]
    assert "hardware=cpu" in captured["command"]
    assert "--simulated_cpu_devices_count=8" in captured["command"]
    assert captured["env"]["JAX_PLATFORMS"] == "cpu"
    assert captured["env"]["PJRT_DEVICE"] == "CPU"
    assert captured["env"]["XLA_FLAGS"].endswith(
        "--xla_force_host_platform_device_count=8"
    )
    assert captured["env"]["RLCSD_LIBTPU_IMPORT_BLOCKED"] == "1"
    assert os.path.basename(
        captured["env"]["PYTHONPATH"].split(os.pathsep)[0]
    ).startswith(
        "rlcsd-cpu-libtpu-block-"
    )
    assert not any(
        key in captured["env"] for key in checkpoint.TPU_RUNTIME_ENV_KEYS
    )

    manifest = json.loads(
        (output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runtime_environment"]["tpu_runtime_keys_present"] == []
    assert manifest["runtime_environment"]["libtpu_import_blocked"] is True
