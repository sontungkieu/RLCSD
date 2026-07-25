from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.maxtext_tpu import dependencies


def test_requirements_preserve_provider_libtpu():
    requirements = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "maxtext_tpu"
        / "requirements-tpu.txt"
    ).read_text(encoding="utf-8")

    assert "maxtext==0.2.3" in requirements
    assert "pathwaysutils>=0.1.8,<0.2" in requirements
    assert "maxtext[tpu]" not in requirements.lower()
    assert not any(
        line.strip().lower().startswith("libtpu")
        for line in requirements.splitlines()
    )


def test_pathways_auxiliary_requirements_cover_unconditional_imports():
    requirements = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "maxtext_tpu"
        / "requirements-config-runtime.txt"
    ).read_text(encoding="utf-8")

    assert "fastapi>=0.115,<1" in requirements
    assert "omegaconf>=2.3,<3" in requirements
    assert "uvicorn>=0.30,<1" in requirements
    assert not any(
        line.strip().lower().startswith(("jax", "jaxlib", "libtpu"))
        for line in requirements.splitlines()
    )


def test_provider_runtime_safe_install_disables_dependency_resolution():
    command = dependencies.provider_runtime_safe_pip_install_command(
        "/usr/bin/python3",
        "/tmp/requirements-tpu.txt",
    )

    assert command[:4] == ["/usr/bin/python3", "-m", "pip", "install"]
    assert "--no-deps" in command
    assert command[-2:] == ["-r", "/tmp/requirements-tpu.txt"]
    assert "--upgrade" not in command


def test_auxiliary_install_resolves_only_if_needed():
    command = dependencies.auxiliary_pip_install_command(
        "/usr/bin/python3",
        "/tmp/requirements-config-runtime.txt",
    )

    assert command[:4] == ["/usr/bin/python3", "-m", "pip", "install"]
    assert "--no-deps" not in command
    assert command[-2:] == ["-r", "/tmp/requirements-config-runtime.txt"]
    assert command[
        command.index("--upgrade-strategy") + 1
    ] == "only-if-needed"


def test_tpu_runtime_guard_accepts_unchanged_versions():
    versions = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.17"}
    dependencies.require_unchanged_tpu_runtime(versions, versions.copy())


def test_tpu_runtime_guard_rejects_libtpu_replacement():
    before = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.17"}
    after = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.42.1"}

    with pytest.raises(RuntimeError, match="libtpu"):
        dependencies.require_unchanged_tpu_runtime(before, after)
