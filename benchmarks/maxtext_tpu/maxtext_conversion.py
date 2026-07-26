"""Compatibility entry point for the installed MaxText converter."""

from __future__ import annotations

import json
import runpy


HF_ID_COMPATIBILITY = {
    # MaxText 0.2.3 ships the Qwen3-1.7B model config and conversion mappings,
    # but accidentally omits this model from maxtext.utils.globals.HF_IDS.
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def _register_hf_id_compatibility(hf_ids: dict[str, str]) -> list[str]:
    registered = []
    for model_name, model_id in HF_ID_COMPATIBILITY.items():
        existing = hf_ids.get(model_name)
        if existing not in (None, model_id):
            raise RuntimeError(
                f"MaxText HF_IDS maps {model_name!r} to unexpected "
                f"value {existing!r}."
            )
        hf_ids[model_name] = model_id
        registered.append(model_name)
    return registered


def _validate_installed_conversion_support(
    model_names: list[str],
) -> dict[str, list[str]]:
    from maxtext.checkpoint_conversion.utils.hf_model_configs import (
        HF_MODEL_CONFIGS,
    )
    from maxtext.checkpoint_conversion.utils.param_mapping import (
        HOOK_FNS,
        PARAM_MAPPING,
    )

    tables = {
        "hf_model_configs": HF_MODEL_CONFIGS,
        "hook_fns": HOOK_FNS,
        "param_mapping": PARAM_MAPPING,
    }
    missing = {
        table_name: [
            model_name
            for model_name in model_names
            if model_name not in table
        ]
        for table_name, table in tables.items()
    }
    missing = {
        table_name: names
        for table_name, names in missing.items()
        if names
    }
    if missing:
        raise RuntimeError(
            "Installed MaxText converter is missing compatibility tables: "
            + json.dumps(missing, sort_keys=True)
        )
    return {
        table_name: sorted(model_names)
        for table_name in tables
    }


def main() -> None:
    from maxtext.utils.globals import HF_IDS

    registered = _register_hf_id_compatibility(HF_IDS)
    validated_tables = _validate_installed_conversion_support(registered)
    print(
        "RLCSD_MAXTEXT_HF_ID_COMPATIBILITY "
        + json.dumps(
            {
                "registered": registered,
                "validated_tables": validated_tables,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    runpy.run_module(
        "maxtext.checkpoint_conversion.to_maxtext",
        run_name="__main__",
        alter_sys=True,
    )


if __name__ == "__main__":
    main()
