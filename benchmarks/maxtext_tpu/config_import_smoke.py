"""Emit durable progress evidence while smoke-importing MaxText config modules."""

from __future__ import annotations

import faulthandler
import importlib
import json
from collections.abc import Callable
from types import ModuleType


SMOKE_MODULES = (
    "aqt.jax.v2",
    "tokamax",
    "tiktoken",
    "pathwaysutils",
    "maxtext",
    "maxtext.configs.pyconfig",
)


def _emit(event: str, module_name: str) -> None:
    print(
        f"{event} "
        + json.dumps({"module": module_name}, sort_keys=True),
        flush=True,
    )


def run_import_smoke(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> tuple[str, ...]:
    """Import each dependency with flushed before/after markers."""

    imported: list[str] = []
    for module_name in SMOKE_MODULES:
        _emit("RLCSD_MAXTEXT_IMPORT_START", module_name)
        import_module(module_name)
        imported.append(module_name)
        _emit("RLCSD_MAXTEXT_IMPORT_OK", module_name)
    print(
        "RLCSD_MAXTEXT_CONFIG_IMPORT_SMOKE "
        + json.dumps({"modules": imported, "ok": True}, sort_keys=True),
        flush=True,
    )
    return tuple(imported)


def main() -> None:
    faulthandler.enable()
    run_import_smoke()


if __name__ == "__main__":
    main()
