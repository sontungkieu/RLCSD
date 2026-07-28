"""KV-cache rollout generation for the original RLCSD contract.

This phase intentionally runs before the actor/teacher phase.  MaxText's
``OfflineEngine`` owns the TPU mesh while it performs cached autoregressive
decode; its compact JSONL output is then consumed by the actor step after the
engine has been released.  Sequential phases let both inference and training
use all eight v5e chips without claiming unsupported concurrent colocation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import ORIGINAL_RLCSD_CASE
from benchmarks.maxtext_tpu.rlcsd_contract import (
    OriginalRlcsdContract,
    load_original_contract,
)
from benchmarks.maxtext_tpu.runtime import (
    _normalize_token_ids,
    _problem_from_row,
    _tokenize_prompt,
    read_checkpoint_manifest,
    require_tpu_v5e8,
    write_json,
)
from src.opsd_format import extract_boxed_answer
from src.verl_reward import compute_score


@dataclasses.dataclass(frozen=True, slots=True)
class PromptRecord:
    prompt_id: str
    row_index: int
    data_source: str
    problem: str
    ground_truth: str
    reference_solution: str
    prompt_token_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["prompt_token_ids"] = list(self.prompt_token_ids)
        return row


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutSample:
    prompt_id: str
    row_index: int
    generation_index: int
    data_source: str
    problem: str
    ground_truth: str
    reference_solution: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_text: str
    old_log_probs: tuple[float, ...]
    reward: float
    answer: str | None

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> RolloutSample:
        return cls(
            prompt_id=str(row["prompt_id"]),
            row_index=int(row["row_index"]),
            generation_index=int(row["generation_index"]),
            data_source=str(row["data_source"]),
            problem=str(row["problem"]),
            ground_truth=str(row["ground_truth"]),
            reference_solution=str(row["reference_solution"]),
            prompt_token_ids=tuple(int(value) for value in row["prompt_token_ids"]),
            response_token_ids=tuple(int(value) for value in row["response_token_ids"]),
            response_text=str(row["response_text"]),
            old_log_probs=tuple(float(value) for value in row["old_log_probs"]),
            reward=float(row["reward"]),
            answer=(None if row.get("answer") is None else str(row["answer"])),
        )

    def as_dict(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        for key in (
            "prompt_token_ids",
            "response_token_ids",
            "old_log_probs",
        ):
            row[key] = list(row[key])
        return row

    def validate(self, contract: OriginalRlcsdContract) -> None:
        if not self.prompt_id:
            raise ValueError("rollout sample has an empty prompt_id")
        if not self.problem or not self.reference_solution:
            raise ValueError(
                f"{self.prompt_id}: problem/reference solution is required"
            )
        if len(self.prompt_token_ids) > contract.max_prompt_length:
            raise ValueError(
                f"{self.prompt_id}: prompt has {len(self.prompt_token_ids)} "
                f"tokens, limit={contract.max_prompt_length}"
            )
        if not self.response_token_ids:
            raise ValueError(f"{self.prompt_id}: empty generated response")
        if len(self.response_token_ids) > contract.max_completion_length:
            raise ValueError(
                f"{self.prompt_id}: response has "
                f"{len(self.response_token_ids)} tokens, "
                f"limit={contract.max_completion_length}"
            )
        if len(self.old_log_probs) != len(self.response_token_ids):
            raise ValueError(
                f"{self.prompt_id}: response/log-prob lengths differ: "
                f"{len(self.response_token_ids)} != {len(self.old_log_probs)}"
            )
        if not math.isfinite(self.reward):
            raise ValueError(f"{self.prompt_id}: non-finite reward")
        if not all(math.isfinite(value) for value in self.old_log_probs):
            raise ValueError(f"{self.prompt_id}: non-finite old log-prob")


def _ground_truth_from_row(row: Mapping[str, Any]) -> str:
    reward_model = row.get("reward_model") or {}
    if isinstance(reward_model, Mapping):
        value = reward_model.get("ground_truth", "")
        if value is not None:
            return str(value).strip()
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, Mapping):
        for key in ("answer", "ground_truth"):
            if extra_info.get(key) is not None:
                return str(extra_info[key]).strip()
    return ""


def _solution_from_row(row: Mapping[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, Mapping):
        value = extra_info.get("solution", "")
        if value is not None:
            return str(value).strip()
    return ""


def load_original_prompt_records(
    contract: OriginalRlcsdContract,
    tokenizer: Any,
    *,
    cache_dir: Path,
    max_records: int | None = None,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Load the deterministic prefix of usable original-dataset prompts.

    ``max_records=None`` materializes the original 60,000-sample training
    prefix.  A caller that only needs one rollout may request exactly 64.
    Rows are never shuffled, matching ``scripts/_run_verl.sh``.
    """

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    started = time.perf_counter()
    info = HfApi().dataset_info(
        contract.dataset_repo,
        revision=contract.dataset_revision,
        files_metadata=True,
    )
    if info.sha != contract.dataset_revision:
        raise RuntimeError(
            f"Dataset resolved to {info.sha}, expected {contract.dataset_revision}."
        )
    sibling = next(
        (item for item in info.siblings if item.rfilename == contract.dataset_file),
        None,
    )
    if sibling is None:
        raise FileNotFoundError(contract.dataset_file)
    parquet_path = Path(
        hf_hub_download(
            repo_id=contract.dataset_repo,
            filename=contract.dataset_file,
            revision=contract.dataset_revision,
            repo_type="dataset",
            local_dir=cache_dir,
        )
    )
    parquet_file = pq.ParquetFile(parquet_path)
    # ``schema.names`` exposes Parquet leaf names for nested structs; the
    # dataset contract is expressed in top-level Arrow columns.
    available_columns = set(parquet_file.schema_arrow.names)
    requested_columns = [
        key
        for key in ("prompt", "data_source", "extra_info", "reward_model")
        if key in available_columns
    ]
    required_columns = {"prompt", "data_source", "extra_info"}
    missing_columns = sorted(required_columns - set(requested_columns))
    if missing_columns:
        raise ValueError(
            f"{contract.dataset_file} is missing columns: {missing_columns}"
        )

    target_count = (
        contract.max_train_samples if max_records is None else int(max_records)
    )
    if target_count < contract.rollout_prompt_batch_size:
        raise ValueError(
            f"max_records must be at least {contract.rollout_prompt_batch_size}"
        )
    if target_count > contract.max_train_samples:
        raise ValueError(
            f"max_records={target_count} exceeds original cap "
            f"{contract.max_train_samples}"
        )

    selected: list[PromptRecord] = []
    scanned_rows = 0
    skipped: dict[str, int] = {
        "empty_problem": 0,
        "empty_ground_truth": 0,
        "empty_reference_solution": 0,
        "overlong_prompt": 0,
    }
    for record_batch in parquet_file.iter_batches(
        batch_size=128,
        columns=requested_columns,
    ):
        for row in record_batch.to_pylist():
            row_index = scanned_rows
            scanned_rows += 1
            problem = _problem_from_row(row)
            ground_truth = _ground_truth_from_row(row)
            solution = _solution_from_row(row)
            if not problem:
                skipped["empty_problem"] += 1
                continue
            if not ground_truth:
                skipped["empty_ground_truth"] += 1
                continue
            if contract.privileged_text_mode == "solution_answer" and not solution:
                skipped["empty_reference_solution"] += 1
                continue
            token_ids = _tokenize_prompt(
                tokenizer,
                problem,
                contract.student_enable_thinking,
            )
            if len(token_ids) > contract.max_prompt_length:
                skipped["overlong_prompt"] += 1
                continue
            prompt_id = hashlib.sha256(f"{row_index}\0{problem}".encode()).hexdigest()[
                :24
            ]
            selected.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    row_index=row_index,
                    data_source=str(row.get("data_source", "")),
                    problem=problem,
                    ground_truth=ground_truth,
                    reference_solution=solution,
                    prompt_token_ids=tuple(token_ids),
                )
            )
            if len(selected) == target_count:
                break
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(
            f"Found {len(selected)} valid prompts after scanning "
            f"{scanned_rows}; required {target_count}."
        )
    return selected, {
        "repo_id": contract.dataset_repo,
        "requested_revision": contract.dataset_revision,
        "resolved_revision": info.sha,
        "dataset_file": contract.dataset_file,
        "file_size_bytes": parquet_path.stat().st_size,
        "declared_file_size_bytes": getattr(sibling, "size", None),
        "parquet_num_rows": parquet_file.metadata.num_rows,
        "scanned_rows": scanned_rows,
        "requested_prompt_count": target_count,
        "selected_prompt_count": len(selected),
        "shuffle": False,
        "skipped": skipped,
        "load_s": time.perf_counter() - started,
    }


def _maxtext_base_config_path() -> Path:
    import maxtext

    path = Path(maxtext.__file__).resolve().parent / "configs" / "base.yml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_offline_inference_config(
    contract: OriginalRlcsdContract,
    *,
    checkpoint_items: Path,
    validation: bool = False,
) -> Any:
    """Create a full-mesh TP8 MaxText KV-cache inference configuration."""

    from maxtext.configs import pyconfig

    target_length = (
        contract.val_max_sequence_length
        if validation
        else contract.train_max_sequence_length
    )
    sampling_temperature = (
        contract.val_temperature if validation else contract.temperature
    )
    sampling_top_k = contract.val_top_k if validation else contract.top_k
    sampling_top_p = contract.val_top_p if validation else contract.top_p
    argv = [
        "",
        str(_maxtext_base_config_path()),
        "model_name=qwen3-1.7b",
        f"tokenizer_path={contract.model_id}",
        "tokenizer_type=huggingface",
        f"load_parameters_path={checkpoint_items.resolve()}",
        "enable_checkpointing=true",
        "async_checkpointing=false",
        "dataset_type=synthetic",
        "steps=1",
        "per_device_batch_size=1",
        f"max_prefill_predict_length={contract.max_prompt_length}",
        f"max_target_length={target_length}",
        "ici_pipeline_parallelism=1",
        "ici_tensor_parallelism=8",
        "ici_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        "scan_layers=true",
        "scan_layers_per_stage=false",
        "attention=dot_product",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "return_log_prob=true",
        "decode_sampling_strategy=composite",
        f"decode_sampling_temperature={sampling_temperature}",
        f"decode_sampling_top_k={sampling_top_k}",
        f"decode_sampling_nucleus_p={sampling_top_p}",
        "skip_jax_distributed_system=true",
    ]
    config = pyconfig.initialize(argv)
    observed = {
        "max_prefill_predict_length": int(config.max_prefill_predict_length),
        "max_target_length": int(config.max_target_length),
        "ici_tensor_parallelism": int(config.ici_tensor_parallelism),
        "return_log_prob": bool(config.return_log_prob),
    }
    expected = {
        "max_prefill_predict_length": contract.max_prompt_length,
        "max_target_length": target_length,
        "ici_tensor_parallelism": 8,
        "return_log_prob": True,
    }
    if observed != expected:
        raise RuntimeError(
            f"MaxText inference config drift: observed={observed}, expected={expected}"
        )
    return config


def create_offline_engine(
    config: Any,
    tokenizer: Any,
    *,
    seed: int,
    params: Any | None = None,
    mesh: Any | None = None,
) -> Any:
    """Instantiate the MaxText engine lazily so CPU validation stays light."""

    # MaxText 0.2.3's OfflineEngine can run without the optional Jetstream
    # serving package when its documented cloud-decoupled mode is enabled.
    # Its OfflineEngine still imports prefill_packing eagerly even when batch
    # prefill is disabled, while that module intentionally rejects decoupled
    # mode. Install the non-packed processor used by OfflineEngine's default
    # path before importing the engine; packed prefill remains fail-fast.
    # RLCSD supplies both the tokenizer and EOS ids below, while the real
    # MaxText model, KV-cache prefill, and autoregressive decode remain active.
    os.environ["DECOUPLE_GCLOUD"] = "TRUE"
    _install_decoupled_prefill_compat()
    import jax
    from maxtext.inference.offline_engine import OfflineEngine

    eos_ids = sorted(
        {
            int(value)
            for value in (
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
            if value is not None and int(value) >= 0
        }
    )
    if not eos_ids:
        raise ValueError("Tokenizer exposes no usable EOS token ids.")
    return OfflineEngine(
        config=config,
        tokenizer=tokenizer,
        eos_ids=eos_ids,
        prefill_lengths=[64, 128, 256, 512, 1024, 2048],
        min_decode_steps=1,
        enable_batch_prefill=False,
        rng=jax.random.PRNGKey(seed),
        params=_as_maxtext_inference_params(params),
        mesh=mesh,
    )


def _as_maxtext_inference_params(params: Any | None) -> Any | None:
    """Convert Flax NNX parameter state to MaxEngine's pure-dict pytree."""

    if params is None:
        return None
    to_pure_dict = getattr(params, "to_pure_dict", None)
    if callable(to_pure_dict):
        return to_pure_dict()
    return params


def update_offline_engine_params(engine: Any, params: Any) -> None:
    """Refresh rollout weights using the pytree shape MaxEngine expects."""

    engine.update_params(_as_maxtext_inference_params(params))


def _install_decoupled_prefill_compat() -> None:
    """Provide OfflineEngine's non-packed prefill API without Jetstream."""

    module_name = "maxtext.input_pipeline.packing.prefill_packing"
    if module_name in sys.modules:
        return

    compat_module = types.ModuleType(module_name)

    class PrefillProcessor:
        """Minimal stock-compatible processor for non-packed prefill."""

        def __init__(self, engine: Any):
            self.engine = engine

        def _process(
            self,
            params: Any,
            tokens: Any,
            slot: int,
            true_length: int,
            decode_state: Any,
            rng: Any,
            return_prompt_logp: bool = False,
        ) -> tuple[Any, Any]:
            prefill_result, first_token = self.engine.prefill(
                params=params,
                padded_tokens=tokens,
                true_length=true_length,
                rng=rng,
                return_prompt_logp=return_prompt_logp,
            )
            decode_state = self.engine.insert(
                prefill_result,
                decode_state,
                slot,
            )
            if return_prompt_logp:
                decode_state["prompt_logp"] = prefill_result["prompt_logp"]
            return first_token, decode_state

    class BatchedPrefillProcessor:
        def __init__(self, *_: Any, **__: Any):
            raise RuntimeError(
                "Batch prefill requires Jetstream and is disabled for the "
                "RLCSD DECOUPLE_GCLOUD rollout path."
            )

    compat_module.PrefillProcessor = PrefillProcessor
    compat_module.BatchedPrefillProcessor = BatchedPrefillProcessor
    compat_module._RLCSD_DECOUPLED_COMPAT = True
    sys.modules[module_name] = compat_module


def generate_rollout_samples(
    prompts: Sequence[PromptRecord],
    contract: OriginalRlcsdContract,
    tokenizer: Any,
    engine: Any,
    *,
    micro_batch_size: int,
    seed: int,
) -> list[RolloutSample]:
    """Generate and verify exactly 64×8 on-policy samples."""

    import jax
    import numpy as np
    from maxtext.inference.offline_engine import InputData

    if len(prompts) != contract.rollout_prompt_batch_size:
        raise ValueError(
            f"Expected {contract.rollout_prompt_batch_size} prompts, "
            f"received {len(prompts)}"
        )
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")

    requests: list[tuple[PromptRecord, int, Any]] = []
    for prompt in prompts:
        for generation_index in range(contract.group_size):
            request_id = f"{prompt.prompt_id}:{generation_index}"
            requests.append(
                (
                    prompt,
                    generation_index,
                    InputData(
                        id=request_id,
                        tokens=np.asarray(
                            prompt.prompt_token_ids,
                            dtype=np.int32,
                        ),
                        true_length=len(prompt.prompt_token_ids),
                    ),
                )
            )

    samples: list[RolloutSample] = []
    rng = jax.random.PRNGKey(seed)
    for start in range(0, len(requests), micro_batch_size):
        chunk = requests[start : start + micro_batch_size]
        rng, chunk_rng = jax.random.split(rng)
        results = engine.batch_inference(
            [request[2] for request in chunk],
            desc=f"rlcsd-rollout-{start // micro_batch_size}",
            rng=chunk_rng,
        )
        by_id = {result.index: result for result in results}
        for prompt, generation_index, request in chunk:
            result = by_id[str(request.id)]
            prompt_length = int(result.prompt_length)
            all_token_ids = _normalize_token_ids(result.token_ids)
            all_log_probs = [
                float(value) for value in np.asarray(result.logprobs).reshape(-1)
            ]
            response_ids = all_token_ids[prompt_length:]
            response_log_probs = all_log_probs[prompt_length:]
            if len(response_ids) > contract.max_completion_length:
                response_ids = response_ids[: contract.max_completion_length]
                response_log_probs = response_log_probs[
                    : contract.max_completion_length
                ]
            response_text = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )
            reward_result = compute_score(
                data_source=prompt.data_source,
                solution_str=response_text,
                ground_truth=prompt.ground_truth,
                extra_info={"problem": prompt.problem},
            )
            reward = float(
                reward_result.get("score", 0.0)
                if isinstance(reward_result, Mapping)
                else reward_result
            )
            sample = RolloutSample(
                prompt_id=prompt.prompt_id,
                row_index=prompt.row_index,
                generation_index=generation_index,
                data_source=prompt.data_source,
                problem=prompt.problem,
                ground_truth=prompt.ground_truth,
                reference_solution=prompt.reference_solution,
                prompt_token_ids=prompt.prompt_token_ids,
                response_token_ids=tuple(response_ids),
                response_text=response_text,
                old_log_probs=tuple(response_log_probs),
                reward=reward,
                answer=extract_boxed_answer(response_text),
            )
            sample.validate(contract)
            samples.append(sample)
    validate_rollout_samples(samples, contract)
    return samples


def validate_rollout_samples(
    samples: Sequence[RolloutSample],
    contract: OriginalRlcsdContract,
) -> dict[str, Any]:
    if len(samples) != contract.rollout_sample_batch_size:
        raise ValueError(
            f"Expected {contract.rollout_sample_batch_size} rollout samples, "
            f"received {len(samples)}"
        )
    groups: dict[str, list[RolloutSample]] = {}
    for sample in samples:
        sample.validate(contract)
        groups.setdefault(sample.prompt_id, []).append(sample)
    if len(groups) != contract.rollout_prompt_batch_size:
        raise ValueError(
            f"Expected {contract.rollout_prompt_batch_size} prompt groups, "
            f"received {len(groups)}"
        )
    bad_groups = {
        prompt_id: sorted(item.generation_index for item in group)
        for prompt_id, group in groups.items()
        if len(group) != contract.group_size
        or sorted(item.generation_index for item in group)
        != list(range(contract.group_size))
    }
    if bad_groups:
        raise ValueError(
            "Rollout group-size/index contract failed: "
            + json.dumps(bad_groups, sort_keys=True)
        )
    return {
        "prompt_group_count": len(groups),
        "group_size": contract.group_size,
        "sample_count": len(samples),
        "positive_sample_count": sum(sample.reward > 0 for sample in samples),
        "negative_sample_count": sum(sample.reward <= 0 for sample in samples),
        "response_token_count": sum(
            len(sample.response_token_ids) for sample in samples
        ),
    }


def write_rollout_jsonl(path: Path, samples: Iterable[RolloutSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.as_dict(), sort_keys=True) + "\n")
    temporary.replace(path)


def read_rollout_jsonl(path: Path) -> list[RolloutSample]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(RolloutSample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-items", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-micro-batch-size", type=int, default=16)
    args = parser.parse_args()

    contract = load_original_contract()
    devices, environment = require_tpu_v5e8()
    del devices
    checkpoint_manifest = read_checkpoint_manifest(
        args.checkpoint_items,
        ORIGINAL_RLCSD_CASE,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract.model_id,
        revision=checkpoint_manifest["resolved_hf_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, dataset = load_original_prompt_records(
        contract,
        tokenizer,
        cache_dir=args.cache_dir,
        max_records=contract.rollout_prompt_batch_size,
    )
    config = build_offline_inference_config(
        contract,
        checkpoint_items=args.checkpoint_items,
    )
    started = time.perf_counter()
    engine = create_offline_engine(config, tokenizer, seed=args.seed)
    samples = generate_rollout_samples(
        prompts,
        contract,
        tokenizer,
        engine,
        micro_batch_size=args.rollout_micro_batch_size,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    summary = validate_rollout_samples(samples, contract)
    output_jsonl = args.output_dir / "rollouts.jsonl"
    write_rollout_jsonl(output_jsonl, samples)
    write_json(
        args.output_dir / "rollout_manifest.json",
        {
            "schema_version": 1,
            "ok": True,
            "phase": "rlcsd_rollout",
            "contract": contract.as_dict(),
            "case": ORIGINAL_RLCSD_CASE.as_dict(),
            "weights": "converted_pretrained_checkpoint",
            "checkpoint_provenance": {
                key: checkpoint_manifest.get(key)
                for key in (
                    "model_id",
                    "resolved_hf_revision",
                    "save_dtype",
                    "tree_metadata_sha256",
                )
            },
            "environment": environment,
            "dataset": dataset,
            "rollout": {
                **summary,
                "decode_backend": "maxtext_offline_engine_kv_cache",
                "temperature": contract.temperature,
                "top_p": contract.top_p,
                "top_k": contract.top_k,
                "max_prompt_length": contract.max_prompt_length,
                "max_completion_length": contract.max_completion_length,
                "elapsed_s": elapsed,
                "output_jsonl": str(output_jsonl.resolve()),
                "output_sha256": hashlib.sha256(output_jsonl.read_bytes()).hexdigest(),
            },
            "measurement_boundary": {
                "includes_decode": True,
                "includes_verifier": True,
                "includes_teacher_correct_wrong_passes": False,
                "includes_rlcsd_policy_loss": False,
                "includes_optimizer_update": False,
            },
        },
    )


if __name__ == "__main__":
    main()
