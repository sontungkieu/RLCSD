from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.maxtext_tpu import config_import_smoke, dependencies


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

    assert "aqtp==0.9.0" in requirements
    assert "tokamax==0.0.12" in requirements
    assert "tiktoken==0.13.0" in requirements
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


def test_config_discovery_smoke_does_not_import_maxtext(capsys):
    imported = []
    discovered = []

    def fake_import(module_name):
        imported.append(module_name)
        return object()

    def fake_discover(module_name):
        discovered.append(module_name)
        return object()

    result = config_import_smoke.run_config_discovery_smoke(
        fake_import,
        fake_discover,
    )
    output = capsys.readouterr().out

    assert result == (
        config_import_smoke.SAFE_IMPORT_MODULES,
        config_import_smoke.DISCOVERY_MODULES,
    )
    assert imported == list(config_import_smoke.SAFE_IMPORT_MODULES)
    assert discovered == list(config_import_smoke.DISCOVERY_MODULES)
    assert not set(imported) & set(config_import_smoke.DISCOVERY_MODULES)
    for module_name in config_import_smoke.SAFE_IMPORT_MODULES:
        assert (
            f'RLCSD_MAXTEXT_IMPORT_START {{"module": "{module_name}"}}'
            in output
        )
        assert (
            f'RLCSD_MAXTEXT_IMPORT_OK {{"module": "{module_name}"}}'
            in output
        )
    for module_name in config_import_smoke.DISCOVERY_MODULES:
        assert (
            f'RLCSD_MAXTEXT_MODULE_SPEC_START {{"module": "{module_name}"}}'
            in output
        )
        assert (
            f'RLCSD_MAXTEXT_MODULE_SPEC_OK {{"module": "{module_name}"}}'
            in output
        )
    assert '"runtime_import_executed": false' in output
    assert '"ok": true' in output


def test_dotted_module_discovery_does_not_execute_parent(
    tmp_path,
    monkeypatch,
):
    package_root = tmp_path / "_rlcsd_spec_smoke"
    configs_root = package_root / "configs"
    configs_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        "raise AssertionError('parent package executed')\n",
        encoding="utf-8",
    )
    (configs_root / "__init__.py").write_text("", encoding="utf-8")
    (configs_root / "pyconfig.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    spec = config_import_smoke._find_spec_without_import(
        "_rlcsd_spec_smoke.configs.pyconfig"
    )

    assert spec is not None
    assert "_rlcsd_spec_smoke" not in sys.modules
