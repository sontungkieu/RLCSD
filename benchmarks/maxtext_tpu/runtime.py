"""Shared runtime helpers for the MaxText TPU benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import (
    DATASET_REPO,
    DATASET_REVISION,
    EXPECTED_TPU_DEVICES,
    BenchmarkCase,
)

BOXED_ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_checkpoint_manifest(
    checkpoint_items: Path, case: BenchmarkCase
) -> dict[str, Any]:
    manifest_path = checkpoint_items.resolve().parents[1] / ("checkpoint_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint provenance manifest is required: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != case.model.model_id:
        raise ValueError(
            "Checkpoint model_id does not match the benchmark model: "
            f"{manifest.get('model_id')} != {case.model.model_id}."
        )
    if not manifest.get("resolved_hf_revision"):
        raise ValueError("Checkpoint manifest has no resolved HF revision.")
    if manifest.get("scan_layers") is not False:
        raise ValueError(
            "Checkpoint manifest must declare scan_layers=false."
        )
    if manifest.get("maxtext_pipeline") != case.maxtext_pipeline_kwargs:
        raise ValueError(
            "Checkpoint MaxText pipeline structure does not match the "
            f"benchmark case: {manifest.get('maxtext_pipeline')} != "
            f"{case.maxtext_pipeline_kwargs}."
        )
    return manifest


def require_tpu_v5e8() -> tuple[list[Any], dict[str, Any]]:
    import jax

    devices = list(jax.devices())
    kinds = [str(getattr(device, "device_kind", "")) for device in devices]
    kinds_lower = [kind.lower() for kind in kinds]
    is_v5e = bool(kinds_lower) and all(
        "v5e" in kind or "v5 lite" in kind for kind in kinds_lower
    )
    evidence = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": [str(device) for device in devices],
        "device_kinds": kinds,
        "v5e_kind_match": is_v5e,
    }
    if (
        evidence["backend"] != "tpu"
        or len(devices) != EXPECTED_TPU_DEVICES
        or not is_v5e
    ):
        raise RuntimeError(
            "Expected a TPU v5e-8 runtime; observed "
            f"{json.dumps(evidence, sort_keys=True)}"
        )
    return devices, evidence


def memory_snapshot(label: str) -> dict[str, Any]:
    import jax
    import psutil

    process = psutil.Process(os.getpid())
    device_rows = []
    for device in jax.devices():
        try:
            stats = device.memory_stats() or {}
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # diagnostic path
            stats = {"error": f"{type(exc).__name__}: {exc}"}
        device_rows.append(
            {
                "device": str(device),
                "bytes_in_use": stats.get("bytes_in_use"),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                "bytes_limit": stats.get("bytes_limit"),
            }
        )
    return {
        "label": label,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_rss_bytes": process.memory_info().rss,
        "devices": device_rows,
    }


def _path_text(path: tuple[Any, ...]) -> str:
    parts = []
    for item in path:
        value = None
        for attribute in ("key", "idx", "name"):
            if hasattr(item, attribute):
                value = getattr(item, attribute)
                break
        parts.append(str(item if value is None else value))
    return ".".join(parts)


def parameter_sharding_report(model: Any, devices: list[Any]) -> dict[str, Any]:
    import jax
    import numpy as np
    from flax import nnx

    state = nnx.state(model, nnx.Param).to_pure_dict()
    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(state)
    bytes_by_device = {str(device): 0 for device in devices}
    logical_bytes = 0
    parameter_count = 0
    stage_sharded_leaf_count = 0
    tensor_sharded_leaf_count = 0
    representative_specs = []
    leaf_count = 0
    for path, array in flat_with_path:
        if not hasattr(array, "shape") or not hasattr(array, "dtype"):
            continue
        leaf_count += 1
        itemsize = int(np.dtype(array.dtype).itemsize)
        logical_bytes += int(array.size) * itemsize
        parameter_count += int(array.size)
        sharding = getattr(array, "sharding", None)
        spec_text = repr(getattr(sharding, "spec", sharding))
        has_stage = "stage" in spec_text
        has_tensor = "tensor" in spec_text
        stage_sharded_leaf_count += int(has_stage)
        tensor_sharded_leaf_count += int(has_tensor)
        if (has_stage or has_tensor) and len(representative_specs) < 30:
            representative_specs.append(
                {
                    "path": _path_text(path),
                    "shape": list(array.shape),
                    "spec": spec_text,
                    "shard_shapes": sorted(
                        {
                            tuple(shard.data.shape)
                            for shard in getattr(array, "addressable_shards", [])
                        }
                    ),
                }
            )
        for shard in getattr(array, "addressable_shards", []):
            bytes_by_device[str(shard.device)] += int(shard.data.size) * itemsize
    return {
        "leaf_count": leaf_count,
        "logical_parameter_count": parameter_count,
        "logical_parameter_bytes": logical_bytes,
        "stage_sharded_leaf_count": stage_sharded_leaf_count,
        "tensor_sharded_leaf_count": tensor_sharded_leaf_count,
        "physical_parameter_bytes_by_device": bytes_by_device,
        "representative_specs": representative_specs,
    }


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def _problem_from_row(row: Mapping[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, Mapping) and extra_info.get("problem"):
        return str(extra_info["problem"]).strip()
    prompt = row.get("prompt") or []
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", "")).strip()
    return str(prompt).strip()


def _tokenize_prompt(tokenizer: Any, problem: str, enable_thinking: bool) -> list[int]:
    messages = [
        {
            "role": "user",
            "content": (f"Problem: {problem.strip()}\n\n{BOXED_ANSWER_INSTRUCTION}"),
        }
    ]
    tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    return _normalize_token_ids(tokens)


def load_real_batch(
    case: BenchmarkCase,
    tokenizer: Any,
    *,
    cache_dir: Path,
    enable_thinking: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    started = time.perf_counter()
    info = HfApi().dataset_info(
        DATASET_REPO,
        revision=DATASET_REVISION,
        files_metadata=True,
    )
    if info.sha != DATASET_REVISION:
        raise RuntimeError(
            f"Dataset resolved to {info.sha}, expected {DATASET_REVISION}."
        )
    sibling = next(
        (item for item in info.siblings if item.rfilename == case.model.dataset_file),
        None,
    )
    if sibling is None:
        raise FileNotFoundError(case.model.dataset_file)
    parquet_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=case.model.dataset_file,
            revision=DATASET_REVISION,
            repo_type="dataset",
            local_dir=cache_dir,
        )
    )
    parquet_file = pq.ParquetFile(parquet_path)
    selected: list[dict[str, Any]] = []
    scanned_rows = 0
    for record_batch in parquet_file.iter_batches(
        batch_size=128,
        columns=["prompt", "data_source", "extra_info"],
    ):
        for row in record_batch.to_pylist():
            row_index = scanned_rows
            scanned_rows += 1
            problem = _problem_from_row(row)
            if not problem:
                continue
            token_ids = _tokenize_prompt(tokenizer, problem, enable_thinking)
            if len(token_ids) > case.sequence_length:
                continue
            selected.append(
                {
                    "row_index": row_index,
                    "data_source": str(row.get("data_source", "")),
                    "problem_sha256": hashlib.sha256(
                        problem.encode("utf-8")
                    ).hexdigest(),
                    "prompt_token_count": len(token_ids),
                    "token_ids": token_ids,
                }
            )
            if len(selected) == case.global_batch_size:
                break
        if len(selected) == case.global_batch_size:
            break
    if len(selected) != case.global_batch_size:
        raise RuntimeError(
            f"Found {len(selected)} usable rows after scanning {scanned_rows}."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id must be configured.")
    token_matrix = np.full(
        (case.global_batch_size, case.sequence_length),
        pad_id,
        dtype=np.int32,
    )
    segment_ids = np.zeros_like(token_matrix)
    for row_index, item in enumerate(selected):
        width = len(item["token_ids"])
        token_matrix[row_index, :width] = item["token_ids"]
        segment_ids[row_index, :width] = 1

    summary = {
        "repo_id": DATASET_REPO,
        "requested_revision": DATASET_REVISION,
        "resolved_revision": info.sha,
        "dataset_file": case.model.dataset_file,
        "file_size_bytes": parquet_path.stat().st_size,
        "declared_file_size_bytes": getattr(sibling, "size", None),
        "parquet_num_rows": parquet_file.metadata.num_rows,
        "download_and_tokenize_s": time.perf_counter() - started,
        "scanned_rows": scanned_rows,
        "global_batch_size": case.global_batch_size,
        "sequence_length": case.sequence_length,
        "actual_non_pad_tokens": int(segment_ids.sum()),
        "selected_prompts": [
            {key: value for key, value in row.items() if key != "token_ids"}
            for row in selected
        ],
    }
    return token_matrix, segment_ids, summary


def create_parallel_config(case: BenchmarkCase) -> Any:
    from tunix.models.maxtext_parallelism import MaxTextPipelineConfig

    config = MaxTextPipelineConfig(
        pipeline_parallelism=case.layout.pipeline_parallelism,
        tensor_parallelism=case.layout.tensor_parallelism,
        num_layers_per_pipeline_stage=(case.num_layers_per_pipeline_stage),
        num_pipeline_microbatches=case.num_pipeline_microbatches,
    )
    config.validate_batch_size(case.global_batch_size)
    return config


def load_model(
    case: BenchmarkCase,
    *,
    devices: list[Any],
    checkpoint_items: Path | None,
    allow_random_weights: bool,
) -> tuple[Any, Any, str | None, str]:
    import jax
    from tunix.models import automodel

    if checkpoint_items is None and not allow_random_weights:
        raise ValueError(
            "A converted checkpoint is required. Use --allow-random-weights "
            "only for an explicitly labelled infrastructure smoke test."
        )
    if checkpoint_items is not None and not checkpoint_items.is_dir():
        raise FileNotFoundError(checkpoint_items)
    if checkpoint_items is not None:
        read_checkpoint_manifest(checkpoint_items, case)

    parallel_config = create_parallel_config(case)
    mesh = parallel_config.create_mesh(devices=devices)
    with jax.set_mesh(mesh):
        model, resolved_path = automodel.AutoModel.from_pretrained(
            model_id=case.model.model_id,
            model_path=(
                str(checkpoint_items.resolve())
                if checkpoint_items is not None
                else None
            ),
            mesh=mesh,
            model_source=automodel.ModelSource.MAXTEXT,
            maxtext_pipeline_config=parallel_config,
            per_device_batch_size=(case.global_batch_size / len(devices)),
            max_target_length=case.sequence_length,
            steps=1,
            dataset_type="synthetic",
            # MaxText 0.2.3 rejects load_parameters_path unless
            # checkpointing is enabled. Keep it disabled only for the
            # explicitly labelled random-weight infrastructure smoke path.
            enable_checkpointing=checkpoint_items is not None,
            enable_dropout=False,
            attention="dot_product",
            remat_policy="full",
            scan_layers=False,
            scan_layers_per_stage=False,
        )
    weights = (
        "converted_pretrained_checkpoint"
        if checkpoint_items is not None
        else "random_shape_correct"
    )
    return model, mesh, resolved_path, weights


def repo_config_provenance(case: BenchmarkCase, repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / case.model.rlcsd_config_path
    content = config_path.read_bytes()
    return {
        "rlcsd_config_path": case.model.rlcsd_config_path,
        "rlcsd_config_sha256": hashlib.sha256(content).hexdigest(),
    }
