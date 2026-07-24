"""Forward and core training-step benchmarks on TPU v5e-8."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import get_case
from benchmarks.maxtext_tpu.runtime import (
    load_model,
    load_real_batch,
    memory_snapshot,
    parameter_sharding_report,
    read_checkpoint_manifest,
    repo_config_provenance,
    require_tpu_v5e8,
    write_json,
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _timed_call(function: Any, *args: Any) -> tuple[float, Any]:
    import jax

    started = time.perf_counter()
    result = function(*args)
    jax.block_until_ready(result)
    return time.perf_counter() - started, result


def _decode_capability_report() -> dict[str, Any]:
    from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter

    signature = inspect.signature(TunixMaxTextAdapter.__call__)
    has_init_cache = hasattr(TunixMaxTextAdapter, "init_cache")
    return {
        "schema_version": 1,
        "phase": "decode_capability",
        "adapter": ("maxtext.integration.tunix.tunix_adapter.TunixMaxTextAdapter"),
        "call_parameters": list(signature.parameters),
        "has_init_cache": has_init_cache,
        "kv_cache_decode_supported": False,
        "reason": (
            "The adapter accepts a cache placeholder but has no init_cache "
            "method and returns logits,None. Tunix Sampler KV-cache decode "
            "must not be claimed for this adapter."
        ),
        "throughput_available": False,
    }


def _run_forward(
    model: Any,
    mesh: Any,
    tokens: Any,
    positions: Any,
    segment_ids: Any,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax import nnx

    @nnx.jit
    def forward_fn(module, input_tokens, input_positions, input_segments):
        logits, _ = module(
            input_tokens,
            input_positions,
            None,
            None,
            decoder_segment_ids=input_segments,
        )
        return jnp.sum(logits[:, -1, :].astype(jnp.float32))

    logical_axis_rules = model.base.config.logical_axis_rules
    with jax.set_mesh(mesh), nn.partitioning.axis_rules(logical_axis_rules):
        compile_s, compile_checksum = _timed_call(
            forward_fn, model, tokens, positions, segment_ids
        )
        warmup_rows = [
            _timed_call(forward_fn, model, tokens, positions, segment_ids)
            for _ in range(warmups)
        ]
        timed_rows = [
            _timed_call(forward_fn, model, tokens, positions, segment_ids)
            for _ in range(repeats)
        ]
    return {
        "compile_and_first_step_s": compile_s,
        "compile_checksum": float(compile_checksum),
        "warmup_step_s": [row[0] for row in warmup_rows],
        "timed_step_s": [row[0] for row in timed_rows],
        "timed_checksums": [float(row[1]) for row in timed_rows],
        "mean_timed_step_s": _mean([row[0] for row in timed_rows]),
    }


def _run_core_train_step(
    model: Any,
    mesh: Any,
    tokens: Any,
    positions: Any,
    segment_ids: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import nnx

    optimizer_started = time.perf_counter()
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        wrt=nnx.Param,
    )
    optimizer_init_s = time.perf_counter() - optimizer_started

    def loss_fn(module, input_tokens, input_positions, input_segments):
        logits, _ = module(
            input_tokens,
            input_positions,
            None,
            None,
            decoder_segment_ids=input_segments,
        )
        prediction_logits = logits[:, :-1, :].astype(jnp.float32)
        targets = input_tokens[:, 1:]
        mask = input_segments[:, 1:].astype(jnp.float32)
        selected_logits = jnp.take_along_axis(
            prediction_logits, targets[..., None], axis=-1
        )[..., 0]
        negative_log_likelihood = (
            jax.nn.logsumexp(prediction_logits, axis=-1) - selected_logits
        )
        return jnp.sum(negative_log_likelihood * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    @nnx.jit
    def train_step(module, opt, input_tokens, input_positions, input_segments):
        loss, grads = nnx.value_and_grad(loss_fn)(
            module,
            input_tokens,
            input_positions,
            input_segments,
        )
        grad_norm = optax.global_norm(
            jax.tree.map(lambda value: value.astype(jnp.float32), grads)
        )
        opt.update(module, grads)
        return loss, grad_norm

    logical_axis_rules = model.base.config.logical_axis_rules
    with jax.set_mesh(mesh), nn.partitioning.axis_rules(logical_axis_rules):
        compile_s, compile_result = _timed_call(
            train_step,
            model,
            optimizer,
            tokens,
            positions,
            segment_ids,
        )
        warmup_rows = [
            _timed_call(
                train_step,
                model,
                optimizer,
                tokens,
                positions,
                segment_ids,
            )
            for _ in range(warmups)
        ]
        timed_rows = [
            _timed_call(
                train_step,
                model,
                optimizer,
                tokens,
                positions,
                segment_ids,
            )
            for _ in range(repeats)
        ]

    def pair_to_floats(pair):
        return {"loss": float(pair[0]), "grad_norm": float(pair[1])}

    timing = {
        "optimizer_init_s": optimizer_init_s,
        "compile_and_first_step_s": compile_s,
        "compile_result": pair_to_floats(compile_result),
        "warmup_step_s": [row[0] for row in warmup_rows],
        "warmup_results": [pair_to_floats(row[1]) for row in warmup_rows],
        "timed_step_s": [row[0] for row in timed_rows],
        "timed_results": [pair_to_floats(row[1]) for row in timed_rows],
        "mean_timed_step_s": _mean([row[0] for row in timed_rows]),
    }
    return timing, optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("forward", "core-train-step", "decode-capability"),
    )
    parser.add_argument("--case")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-items", type=Path)
    parser.add_argument("--allow-random-weights", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    if args.mode == "decode-capability":
        write_json(args.output, _decode_capability_report())
        return
    if not args.case:
        parser.error("--case is required for forward/core-train-step")
    if args.warmups < 1 or args.repeats < 1:
        parser.error("--warmups and --repeats must be positive")

    case = get_case(args.case)
    devices, environment = require_tpu_v5e8()
    memory = [memory_snapshot("before_tokenizer")]
    checkpoint_manifest = (
        read_checkpoint_manifest(args.checkpoint_items, case)
        if args.checkpoint_items is not None
        else None
    )

    import jax.numpy as jnp
    from transformers import AutoTokenizer

    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        case.model.model_id,
        revision=(
            checkpoint_manifest["resolved_hf_revision"]
            if checkpoint_manifest is not None
            else "main"
        ),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_s = time.perf_counter() - tokenizer_started
    token_matrix, segment_matrix, dataset = load_real_batch(
        case,
        tokenizer,
        cache_dir=args.cache_dir,
        enable_thinking=True,
    )

    memory.append(memory_snapshot("before_model"))
    model_started = time.perf_counter()
    model, mesh, resolved_path, weights = load_model(
        case,
        devices=devices,
        checkpoint_items=args.checkpoint_items,
        allow_random_weights=args.allow_random_weights,
    )
    model_init_s = time.perf_counter() - model_started
    memory.append(memory_snapshot("after_model"))
    sharding = parameter_sharding_report(model, devices)
    if (
        sharding["stage_sharded_leaf_count"] < 1
        or sharding["tensor_sharded_leaf_count"] < 1
    ):
        raise RuntimeError(
            "Model parameters are not physically sharded over both stage "
            f"and tensor axes: {json.dumps(sharding, sort_keys=True)}"
        )

    tokens = jnp.asarray(token_matrix)
    segment_ids = jnp.asarray(segment_matrix)
    positions = jnp.broadcast_to(
        jnp.arange(case.sequence_length, dtype=jnp.int32),
        token_matrix.shape,
    )

    optimizer = None
    if args.mode == "forward":
        timing = _run_forward(
            model,
            mesh,
            tokens,
            positions,
            segment_ids,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        workload_kind = "full_sequence_forward"
    else:
        timing, optimizer = _run_core_train_step(
            model,
            mesh,
            tokens,
            positions,
            segment_ids,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        workload_kind = "causal_lm_forward_backward_adamw"
    memory.append(memory_snapshot("after_timed_steps"))

    padded_tokens_per_step = case.global_batch_size * case.sequence_length
    actual_target_tokens = int(segment_matrix[:, 1:].sum())
    mean_step_s = timing["mean_timed_step_s"]
    summary = {
        "schema_version": 1,
        "ok": True,
        "phase": args.mode,
        "case": case.as_dict(),
        "workload_kind": workload_kind,
        "is_rlcsd_end_to_end": False,
        "weights": weights,
        "checkpoint_items": resolved_path,
        "checkpoint_provenance": (
            {
                key: checkpoint_manifest[key]
                for key in (
                    "model_id",
                    "resolved_hf_revision",
                    "save_dtype",
                    "tree_metadata_sha256",
                )
            }
            if checkpoint_manifest is not None
            else None
        ),
        "environment": environment,
        "parameter_sharding": sharding,
        "dataset": dataset,
        "config": repo_config_provenance(case, Path(__file__).resolve().parents[2]),
        "timing": {
            "tokenizer_load_s": tokenizer_load_s,
            "model_init_s": model_init_s,
            **timing,
        },
        "throughput": {
            "padded_tokens_per_s": padded_tokens_per_step / mean_step_s,
            "actual_target_tokens_per_s": (actual_target_tokens / mean_step_s),
            "examples_per_s": case.global_batch_size / mean_step_s,
        },
        "memory": memory,
        "measurement_boundary": {
            "compile_excluded": True,
            "warmup_excluded": True,
            "timed_repeats": args.repeats,
            "includes_decode": False,
            "includes_verifier": False,
            "includes_teacher_correct_wrong_passes": False,
            "includes_rlcsd_policy_loss": False,
        },
    }
    write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    del optimizer


if __name__ == "__main__":
    main()
