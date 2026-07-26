"""Smoke-import safe dependencies and discover MaxText without executing it."""

from __future__ import annotations

import faulthandler
import importlib
import json
from collections.abc import Callable
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType


SAFE_IMPORT_MODULES = (
    "aqt.jax.v2",
    "tokamax",
    "tiktoken",
    "pathwaysutils",
)
DISCOVERY_MODULES = (
    "maxtext",
    "maxtext.configs.pyconfig",
)


def _emit(event: str, module_name: str) -> None:
    print(
        f"{event} "
        + json.dumps({"module": module_name}, sort_keys=True),
        flush=True,
    )


def _find_spec_without_import(module_name: str) -> ModuleSpec | None:
    """Resolve a dotted module without importing any parent package."""

    search_path: list[str] | None = None
    spec: ModuleSpec | None = None
    components: list[str] = []
    for component in module_name.split("."):
        components.append(component)
        spec = PathFinder.find_spec(".".join(components), search_path)
        if spec is None:
            return None
        if len(components) < len(module_name.split(".")):
            locations = spec.submodule_search_locations
            if locations is None:
                return None
            search_path = list(locations)
    return spec


def run_config_discovery_smoke(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
    discover_module: Callable[[str], object | None] = _find_spec_without_import,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Import safe dependencies and discover MaxText with durable markers."""

    imported: list[str] = []
    for module_name in SAFE_IMPORT_MODULES:
        _emit("RLCSD_MAXTEXT_IMPORT_START", module_name)
        import_module(module_name)
        imported.append(module_name)
        _emit("RLCSD_MAXTEXT_IMPORT_OK", module_name)

    discovered: list[str] = []
    for module_name in DISCOVERY_MODULES:
        _emit("RLCSD_MAXTEXT_MODULE_SPEC_START", module_name)
        if discover_module(module_name) is None:
            raise ModuleNotFoundError(
                f"Could not discover MaxText module without importing it: "
                f"{module_name}"
            )
        discovered.append(module_name)
        _emit("RLCSD_MAXTEXT_MODULE_SPEC_OK", module_name)

    print(
        "RLCSD_MAXTEXT_CONFIG_DISCOVERY_SMOKE "
        + json.dumps(
            {
                "discovered_modules": discovered,
                "imported_modules": imported,
                "ok": True,
                "runtime_import_executed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return tuple(imported), tuple(discovered)


def main() -> None:
    faulthandler.enable()
    run_config_discovery_smoke()


if __name__ == "__main__":
    main()
