"""Logits parity gate for a converted HF -> MaxText checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.maxtext_tpu.matrix import get_case
from benchmarks.maxtext_tpu.runtime import (
    load_model,
    load_real_batch,
    memory_snapshot,
    read_checkpoint_manifest,
    require_tpu_v5e8,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint-items", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-abs-atol", type=float, default=2.0)
    parser.add_argument("--mean-abs-atol", type=float, default=0.15)
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--top-k-overlap-min", type=float, default=0.9)
    args = parser.parse_args()

    case = get_case(args.case)
    manifest = read_checkpoint_manifest(args.checkpoint_items, case)
    revision = str(manifest["resolved_hf_revision"])

    devices, environment = require_tpu_v5e8()
    memory = [memory_snapshot("before_parity_models")]

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        case.model.model_id,
        revision=revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_matrix, segment_matrix, dataset = load_real_batch(
        case,
        tokenizer,
        cache_dir=args.cache_dir,
        enable_thinking=True,
    )
    sample_width = int(segment_matrix[0].sum())
    if sample_width < 2:
        raise RuntimeError("Parity prompt must contain at least two tokens.")
    last_index = sample_width - 1

    hf_started = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        case.model.model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    hf_model.eval()
    with torch.inference_mode():
        hf_outputs = hf_model(
            input_ids=torch.as_tensor(
                token_matrix[:1, :sample_width], dtype=torch.long
            ),
            attention_mask=torch.as_tensor(
                segment_matrix[:1, :sample_width], dtype=torch.long
            ),
        )
        hf_logits = hf_outputs.logits[0, last_index].float().cpu().numpy()
    hf_elapsed_s = time.perf_counter() - hf_started
    del hf_outputs, hf_model

    maxtext_started = time.perf_counter()
    maxtext_model, mesh, resolved_path, weights = load_model(
        case,
        devices=devices,
        checkpoint_items=args.checkpoint_items,
        allow_random_weights=False,
    )
    # Parity needs one real prompt, not the case's dense training shape.  This
    # is essential for the original 40,960-token model budget: materializing
    # [16, 40960, vocab] logits would turn a one-position parity check into an
    # unrelated memory stress test.
    parity_tokens = token_matrix[:1, :sample_width]
    parity_segments = segment_matrix[:1, :sample_width]
    positions = jnp.broadcast_to(
        jnp.arange(sample_width, dtype=jnp.int32),
        parity_tokens.shape,
    )
    with jax.set_mesh(mesh):
        maxtext_outputs, _ = maxtext_model(
            jnp.asarray(parity_tokens),
            positions,
            None,
            None,
            decoder_segment_ids=jnp.asarray(parity_segments),
        )
        maxtext_logits = np.asarray(
            jax.device_get(maxtext_outputs[0, last_index]),
            dtype=np.float32,
        )
    maxtext_elapsed_s = time.perf_counter() - maxtext_started
    memory.append(memory_snapshot("after_parity_models"))

    if hf_logits.shape != maxtext_logits.shape:
        raise ValueError(
            f"Logit shapes differ: HF {hf_logits.shape}, "
            f"MaxText {maxtext_logits.shape}."
        )
    difference = np.abs(hf_logits - maxtext_logits)
    denominator = np.linalg.norm(hf_logits) * np.linalg.norm(maxtext_logits)
    cosine = float(np.dot(hf_logits, maxtext_logits) / denominator)
    hf_top = set(np.argpartition(hf_logits, -args.top_k)[-args.top_k :])
    maxtext_top = set(np.argpartition(maxtext_logits, -args.top_k)[-args.top_k :])
    top_k_overlap = len(hf_top & maxtext_top) / args.top_k
    metrics = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "cosine_similarity": cosine,
        "top_k": args.top_k,
        "top_k_overlap": top_k_overlap,
    }
    thresholds = {
        "max_abs_atol": args.max_abs_atol,
        "mean_abs_atol": args.mean_abs_atol,
        "cosine_min": args.cosine_min,
        "top_k_overlap_min": args.top_k_overlap_min,
    }
    checks = {
        "max_abs": metrics["max_abs_error"] <= args.max_abs_atol,
        "mean_abs": metrics["mean_abs_error"] <= args.mean_abs_atol,
        "cosine": cosine >= args.cosine_min,
        "top_k_overlap": top_k_overlap >= args.top_k_overlap_min,
    }
    summary = {
        "schema_version": 1,
        "phase": "checkpoint_logits_parity",
        "ok": all(checks.values()),
        "case": case.as_dict(),
        "weights": weights,
        "checkpoint_items": resolved_path,
        "resolved_hf_revision": revision,
        "environment": environment,
        "dataset": dataset,
        "sample": {
            "dataset_row_index": dataset["selected_prompts"][0]["row_index"],
            "prompt_token_count": sample_width,
            "compared_position": last_index,
            "maxtext_forward_batch_size": 1,
            "maxtext_forward_sequence_length": sample_width,
        },
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "timing": {
            "hf_cpu_load_and_forward_s": hf_elapsed_s,
            "maxtext_tpu_load_and_forward_s": maxtext_elapsed_s,
        },
        "memory": memory,
    }
    write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    if not summary["ok"]:
        raise RuntimeError(f"Checkpoint logits parity failed: {checks}")


if __name__ == "__main__":
    main()
