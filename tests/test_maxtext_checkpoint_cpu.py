from __future__ import annotations

import json

from benchmarks.maxtext_tpu import checkpoint


def test_checkpoint_cli_forces_outer_process_to_cpu_before_resolution(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "checkpoint"
    (output_dir / "0" / "items").mkdir(parents=True)
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("PJRT_DEVICE", "TPU")

    def resolve_revision(model_id, requested_revision):
        assert model_id == "Qwen/Qwen3-1.7B"
        assert requested_revision == "main"
        assert checkpoint.os.environ["JAX_PLATFORMS"] == "cpu"
        assert checkpoint.os.environ["PJRT_DEVICE"] == "CPU"
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
    assert manifest["runtime_environment"] == {
        "jax_platforms": "cpu",
        "pjrt_device": "CPU",
    }


def test_checkpoint_cpu_environment_overrides_inherited_tpu_without_mutation():
    inherited = {
        "JAX_PLATFORMS": "tpu",
        "PJRT_DEVICE": "TPU",
        "HF_HOME": "/tmp/hf",
    }
    isolated = checkpoint._cpu_only_environment(inherited)

    assert isolated["JAX_PLATFORMS"] == "cpu"
    assert isolated["PJRT_DEVICE"] == "CPU"
    assert isolated["HF_HOME"] == "/tmp/hf"
    assert inherited["JAX_PLATFORMS"] == "tpu"
    assert inherited["PJRT_DEVICE"] == "TPU"
