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
    assert "maxtext[tpu]" not in requirements.lower()
    assert not any(
        line.strip().lower().startswith("libtpu")
        for line in requirements.splitlines()
    )


def test_tpu_runtime_guard_accepts_unchanged_versions():
    versions = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.17"}
    dependencies.require_unchanged_tpu_runtime(versions, versions.copy())


def test_tpu_runtime_guard_rejects_libtpu_replacement():
    before = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.17"}
    after = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.42.1"}

    with pytest.raises(RuntimeError, match="libtpu"):
        dependencies.require_unchanged_tpu_runtime(before, after)
