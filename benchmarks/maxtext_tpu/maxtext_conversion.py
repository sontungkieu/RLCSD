"""Compatibility entry point for the installed MaxText converter."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from typing import Any

import numpy as np


HF_ID_COMPATIBILITY = {
    # MaxText 0.2.3 ships the Qwen3-1.7B model config and conversion mappings,
    # but accidentally omits this model from maxtext.utils.globals.HF_IDS.
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

_SCANNED_LAYER_PREFIX = "params-decoder-layers-"
_PIPELINE_LAYER_PREFIX = (
    "params-decoder-pipeline_module-layers-layers_{local_layer}-"
)


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


def _pipeline_layout(maxtext_config: Any) -> tuple[int, int] | None:
    pipeline_stages = int(
        getattr(maxtext_config, "ici_pipeline_parallelism", 1)
    ) * int(getattr(maxtext_config, "dcn_pipeline_parallelism", 1))
    if pipeline_stages <= 1:
        return None
    if bool(getattr(maxtext_config, "scan_layers_per_stage", False)):
        raise RuntimeError(
            "Qwen pipeline converter compatibility currently requires "
            "scan_layers_per_stage=false."
        )
    pipeline_repeats = int(
        getattr(maxtext_config, "num_pipeline_repeats", 1)
    )
    if pipeline_repeats != 1:
        raise RuntimeError(
            "Qwen pipeline converter compatibility currently requires "
            "num_pipeline_repeats=1."
        )
    layers_per_stage = int(
        getattr(maxtext_config, "num_layers_per_pipeline_stage", 0)
    )
    if layers_per_stage <= 0:
        raise RuntimeError(
            "num_layers_per_pipeline_stage must be positive for pipeline "
            "checkpoint conversion."
        )
    return pipeline_stages, layers_per_stage


def _pipeline_layer_key(base_key: str, local_layer: int) -> str:
    if not base_key.startswith(_SCANNED_LAYER_PREFIX):
        raise ValueError(f"Not a scanned Qwen layer key: {base_key!r}")
    suffix = base_key.removeprefix(_SCANNED_LAYER_PREFIX)
    return _PIPELINE_LAYER_PREFIX.format(
        local_layer=local_layer
    ) + suffix


def _pipeline_qwen_param_mapping(
    base_mapping: dict[str, Any],
    *,
    num_hidden_layers: int,
    pipeline_stages: int,
    layers_per_stage: int,
) -> dict[str, Any]:
    expected_layers = pipeline_stages * layers_per_stage
    if num_hidden_layers != expected_layers:
        raise RuntimeError(
            "Pipeline layout does not cover every Qwen layer: "
            f"num_hidden_layers={num_hidden_layers}, "
            f"pipeline_stages={pipeline_stages}, "
            f"layers_per_stage={layers_per_stage}."
        )

    remapped: dict[str, Any] = {}
    for key, value in base_mapping.items():
        if not key.startswith(_SCANNED_LAYER_PREFIX):
            remapped[key] = value
            continue
        if not isinstance(value, list) or len(value) != num_hidden_layers:
            raise RuntimeError(
                "Expected a scanned Qwen mapping with one source per layer "
                f"for {key!r}; received {type(value).__name__}."
            )
        for local_layer in range(layers_per_stage):
            remapped[
                _pipeline_layer_key(key, local_layer)
            ] = tuple(
                value[stage * layers_per_stage + local_layer]
                for stage in range(pipeline_stages)
            )
    return remapped


def _stack_pipeline_stages_hook(
    base_hook: Any,
    *,
    pipeline_stages: int,
) -> Callable[[tuple[Any, ...], tuple[int, ...]], np.ndarray]:
    def stack_pipeline_stages(
        stage_tensors: tuple[Any, ...],
        target_shape: tuple[int, ...],
    ) -> np.ndarray:
        if (
            not isinstance(stage_tensors, tuple)
            or len(stage_tensors) != pipeline_stages
        ):
            raise RuntimeError(
                "Expected one Hugging Face tensor per pipeline stage."
            )
        if not target_shape or target_shape[0] != pipeline_stages:
            raise RuntimeError(
                "Expected the MaxText pipeline stage axis at dimension 0; "
                f"received target_shape={target_shape!r}."
            )
        hooks = (
            base_hook
            if isinstance(base_hook, list)
            else [base_hook]
        )
        stage_shape = tuple(target_shape[1:])
        converted = []
        for tensor in stage_tensors:
            for hook in hooks:
                if hook is not None:
                    tensor = hook(tensor, stage_shape)
            converted.append(tensor)
        return np.stack(converted, axis=0)

    return stack_pipeline_stages


def _pipeline_qwen_hook_mapping(
    base_hooks: dict[str, Any],
    base_mapping: dict[str, Any],
    *,
    pipeline_stages: int,
    layers_per_stage: int,
) -> dict[str, Any]:
    remapped = {
        key: hook
        for key, hook in base_hooks.items()
        if not key.startswith(_SCANNED_LAYER_PREFIX)
    }
    for key in base_mapping:
        if not key.startswith(_SCANNED_LAYER_PREFIX):
            continue
        base_hook = base_hooks.get(key)
        for local_layer in range(layers_per_stage):
            remapped[_pipeline_layer_key(key, local_layer)] = (
                _stack_pipeline_stages_hook(
                    base_hook,
                    pipeline_stages=pipeline_stages,
                )
            )
    return remapped


def _wrap_qwen_pipeline_mapping(
    base_mapping_fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def mapping(config, maxtext_config, scan_layers=False):
        base_mapping = base_mapping_fn(
            config, maxtext_config, scan_layers
        )
        layout = _pipeline_layout(maxtext_config)
        if not scan_layers or layout is None:
            return base_mapping
        pipeline_stages, layers_per_stage = layout
        return _pipeline_qwen_param_mapping(
            base_mapping,
            num_hidden_layers=int(config["num_hidden_layers"]),
            pipeline_stages=pipeline_stages,
            layers_per_stage=layers_per_stage,
        )

    return mapping


def _wrap_qwen_pipeline_hooks(
    base_hook_fn: Callable[..., dict[str, Any]],
    base_mapping_fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def hooks(
        config,
        maxtext_config,
        scan_layers=False,
        saving_to_hf=False,
    ):
        base_hooks = base_hook_fn(
            config,
            maxtext_config,
            scan_layers,
            saving_to_hf=saving_to_hf,
        )
        layout = _pipeline_layout(maxtext_config)
        if not scan_layers or layout is None:
            return base_hooks
        pipeline_stages, layers_per_stage = layout
        base_mapping = base_mapping_fn(
            config, maxtext_config, scan_layers
        )
        return _pipeline_qwen_hook_mapping(
            base_hooks,
            base_mapping,
            pipeline_stages=pipeline_stages,
            layers_per_stage=layers_per_stage,
        )

    return hooks


def _install_qwen_pipeline_conversion_compatibility(
    model_names: list[str],
) -> dict[str, Any]:
    from maxtext.checkpoint_conversion.utils import param_mapping

    installed = []
    for model_name in model_names:
        mapping_fn = param_mapping.PARAM_MAPPING[model_name]
        hook_fn = param_mapping.HOOK_FNS[model_name]
        param_mapping.PARAM_MAPPING[model_name] = (
            _wrap_qwen_pipeline_mapping(mapping_fn)
        )
        param_mapping.HOOK_FNS[model_name] = (
            _wrap_qwen_pipeline_hooks(hook_fn, mapping_fn)
        )
        installed.append(model_name)
    return {
        "installed": installed,
        "layout": "contiguous_stage_blocks",
        "scan_layers_per_stage": False,
    }


def main() -> None:
    from maxtext.utils.globals import HF_IDS

    registered = _register_hf_id_compatibility(HF_IDS)
    validated_tables = _validate_installed_conversion_support(registered)
    pipeline_compatibility = (
        _install_qwen_pipeline_conversion_compatibility(registered)
    )
    print(
        "RLCSD_MAXTEXT_HF_ID_COMPATIBILITY "
        + json.dumps(
            {
                "registered": registered,
                "validated_tables": validated_tables,
                "pipeline_compatibility": pipeline_compatibility,
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
