"""Resumable original-contract RLCSD training on one TPU v5e-8 host.

The reference run is far longer than one Kaggle TPU session.  This entrypoint
therefore treats a Kaggle notebook as a resumable shard of one logical run.  A
shard may stop only between validation prompts or rollout batches, writes a
full model/optimizer/teacher checkpoint, and never claims 30-epoch completion
until every reference epoch and validation gate is present in the ledger.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import ORIGINAL_RLCSD_CASE
from benchmarks.maxtext_tpu.rlcsd_actor import (
    OriginalRlcsdActorState,
    create_original_rlcsd_optimizer,
    run_original_rlcsd_updates,
)
from benchmarks.maxtext_tpu.rlcsd_contract import (
    OriginalRlcsdContract,
    load_original_contract,
)
from benchmarks.maxtext_tpu.rlcsd_data import (
    build_teacher_assignments,
    compute_grpo_advantages,
    iter_actor_minibatches,
    pack_actor_minibatch,
)
from benchmarks.maxtext_tpu.rlcsd_rollout import (
    PromptRecord,
    build_offline_inference_config,
    create_offline_engine,
    generate_rollout_samples,
    load_original_prompt_records,
    update_offline_engine_params,
)
from benchmarks.maxtext_tpu.runtime import (
    _normalize_token_ids,
    _tokenize_prompt,
    load_model,
    parameter_sharding_report,
    read_checkpoint_manifest,
    require_tpu_v5e8,
    write_json,
)
from src.opsd_format import extract_boxed_answer, grade_boxed_answer

EVAL_SOURCES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "amc23": (
        "AI-MO/aimo-validation-amc",
        "69d78a4a2c840e82d69af6bc742bda09005f6316",
        ("data/train-00000-of-00001.parquet",),
    ),
    "aime24": (
        "AI-MO/aimo-validation-aime",
        "13f9e12f613e720c2a2b2f345dd04b998a29494d",
        ("data/train-00000-of-00001.parquet",),
    ),
    "aime25": (
        "opencompass/AIME2025",
        "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        ("aime2025-I.jsonl", "aime2025-II.jsonl"),
    ),
}


def _contract_digest(contract: OriginalRlcsdContract) -> str:
    encoded = json.dumps(
        contract.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class EvalRecord:
    dataset: str
    record_id: str
    problem: str
    answer: str
    prompt_token_ids: tuple[int, ...]


@dataclasses.dataclass(slots=True)
class TrainingProgress:
    """Small JSON state coupled to the latest Orbax checkpoint."""

    schema_version: int
    contract_sha256: str
    phase: str
    active_validation_step: int | None
    pretrain_validation_dataset_index: int
    pretrain_validation_record_index: int
    pretrain_validation_complete: bool
    epoch_index: int
    batch_index_in_epoch: int
    rollout_step: int
    optimizer_step: int
    prompts_consumed: int
    periodic_checkpoint_steps: list[int]
    periodic_validation_steps: list[int]

    @classmethod
    def initial(cls, contract: OriginalRlcsdContract) -> TrainingProgress:
        return cls(
            schema_version=1,
            contract_sha256=_contract_digest(contract),
            phase="pretrain_validation",
            active_validation_step=0,
            pretrain_validation_dataset_index=0,
            pretrain_validation_record_index=0,
            pretrain_validation_complete=False,
            epoch_index=0,
            batch_index_in_epoch=0,
            rollout_step=0,
            optimizer_step=0,
            prompts_consumed=0,
            periodic_checkpoint_steps=[],
            periodic_validation_steps=[],
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        contract: OriginalRlcsdContract,
    ) -> TrainingProgress:
        raw = json.loads(path.read_text(encoding="utf-8"))
        progress = cls(**raw)
        if progress.contract_sha256 != _contract_digest(contract):
            raise ValueError("resume progress belongs to a different contract")
        return progress

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def original_steps_per_epoch(contract: OriginalRlcsdContract) -> int:
    """Match VERL StatefulDataLoader(drop_last=True)."""

    return contract.max_train_samples // contract.rollout_prompt_batch_size


def original_total_rollout_steps(contract: OriginalRlcsdContract) -> int:
    return original_steps_per_epoch(contract) * contract.num_epochs


def prompt_batch_for_progress(
    prompts: Sequence[PromptRecord],
    progress: TrainingProgress,
    contract: OriginalRlcsdContract,
) -> Sequence[PromptRecord]:
    steps_per_epoch = original_steps_per_epoch(contract)
    if not 0 <= progress.batch_index_in_epoch < steps_per_epoch:
        raise ValueError("batch_index_in_epoch is outside the original epoch")
    start = progress.batch_index_in_epoch * contract.rollout_prompt_batch_size
    end = start + contract.rollout_prompt_batch_size
    batch = prompts[start:end]
    if len(batch) != contract.rollout_prompt_batch_size:
        raise ValueError("drop-last prompt batch is incomplete")
    return batch


def advance_training_progress(
    progress: TrainingProgress,
    contract: OriginalRlcsdContract,
    *,
    optimizer_step: int,
) -> None:
    progress.rollout_step += 1
    progress.optimizer_step = int(optimizer_step)
    progress.prompts_consumed += contract.rollout_prompt_batch_size
    progress.batch_index_in_epoch += 1
    if progress.batch_index_in_epoch == original_steps_per_epoch(contract):
        progress.epoch_index += 1
        progress.batch_index_in_epoch = 0
    progress.phase = (
        "complete" if progress.epoch_index == contract.num_epochs else "training"
    )


def _problem_answer(row: Mapping[str, Any]) -> tuple[str, str]:
    problem = row.get("problem", row.get("question", ""))
    answer = row.get("answer", row.get("ground_truth", ""))
    return str(problem).strip(), str(answer).strip()


def _load_eval_records(
    dataset_name: str,
    tokenizer: Any,
    contract: OriginalRlcsdContract,
    *,
    cache_dir: Path,
) -> tuple[list[EvalRecord], dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo_id, revision, filenames = EVAL_SOURCES[dataset_name]
    info = HfApi().dataset_info(repo_id, revision=revision)
    if info.sha != revision:
        raise RuntimeError(
            f"{dataset_name} resolved to {info.sha}, expected {revision}"
        )
    rows: list[Mapping[str, Any]] = []
    downloaded: list[str] = []
    for filename in filenames:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="dataset",
                local_dir=cache_dir / "eval" / dataset_name,
            )
        )
        downloaded.append(str(path))
        if path.suffix == ".parquet":
            rows.extend(pq.read_table(path).to_pylist())
        else:
            with path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())

    records: list[EvalRecord] = []
    for index, row in enumerate(rows):
        problem, answer = _problem_answer(row)
        if dataset_name == "aime24" and "2024" not in str(row.get("url", "")):
            continue
        if not problem or not answer:
            continue
        token_ids = _tokenize_prompt(
            tokenizer,
            problem,
            contract.val_enable_thinking,
        )
        if len(token_ids) > contract.max_prompt_length:
            continue
        records.append(
            EvalRecord(
                dataset=dataset_name,
                record_id=f"{dataset_name}:{index}",
                problem=problem,
                answer=answer,
                prompt_token_ids=tuple(token_ids),
            )
        )
    if not records:
        raise RuntimeError(f"{dataset_name} produced no usable records")
    return records, {
        "dataset": dataset_name,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "files": list(filenames),
        "downloaded_files": downloaded,
        "record_count": len(records),
    }


def _generate_validation_samples(
    engine: Any,
    tokenizer: Any,
    record: EvalRecord,
    contract: OriginalRlcsdContract,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    import jax
    import numpy as np
    from maxtext.inference.offline_engine import InputData

    requests = [
        InputData(
            id=f"{record.record_id}:{index}",
            tokens=np.asarray(record.prompt_token_ids, dtype=np.int32),
            true_length=len(record.prompt_token_ids),
        )
        for index in range(contract.val_n)
    ]
    results = engine.batch_inference(
        requests,
        desc=f"validation-{record.record_id}",
        rng=jax.random.PRNGKey(seed),
    )
    rows = []
    for generation_index, result in enumerate(results):
        token_ids = _normalize_token_ids(result.token_ids)
        response_ids = token_ids[int(result.prompt_length) :]
        response_ids = response_ids[: contract.val_max_completion_length]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        prediction = extract_boxed_answer(response)
        rows.append(
            {
                "dataset": record.dataset,
                "record_id": record.record_id,
                "generation_index": generation_index,
                "response_token_count": len(response_ids),
                "prediction": prediction,
                "correct": grade_boxed_answer(prediction, record.answer),
            }
        )
    if len(rows) != contract.val_n:
        raise RuntimeError("validation generation count drifted from val_n")
    return rows


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _deadline_reached(
    started: float,
    deadline_seconds: int,
    *,
    safety_seconds: int,
) -> bool:
    return time.perf_counter() - started >= deadline_seconds - safety_seconds


def _checkpoint_manager(checkpoint_dir: Path, contract: OriginalRlcsdContract) -> Any:
    from maxtext.common import checkpointing

    return checkpointing.create_orbax_checkpoint_manager(
        str(checkpoint_dir),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=contract.save_steps,
        dataset_type=None,
        max_num_checkpoints_to_keep=2,
        checkpoint_storage_concurrent_gb=16,
    )


def _save_training_checkpoint(
    *,
    checkpoint_manager: Any,
    checkpoint_dir: Path,
    model: Any,
    actor_state: OriginalRlcsdActorState,
    progress: TrainingProgress,
    reason: str,
) -> dict[str, Any]:
    from flax import nnx
    from maxtext.common import checkpointing, train_state_nnx

    state = train_state_nnx.TrainStateNNX(model, actor_state.optimizer)
    nnx_state = nnx.state(state)
    linen_state = train_state_nnx.to_linen_checkpoint_dict(nnx_state.to_pure_dict())
    step = progress.rollout_step
    saved = checkpointing.save_checkpoint(
        checkpoint_manager,
        step,
        linen_state,
        force=True,
    )
    checkpoint_manager.wait_until_finished()
    step_dir = checkpoint_dir / str(step)
    teacher_dir = step_dir / "teacher_items"
    checkpointing.save_params_to_path(
        str(teacher_dir),
        actor_state.teacher_params,
    )
    progress_path = step_dir / "training_progress.json"
    write_json(progress_path, progress.as_dict())
    write_json(checkpoint_dir / "latest_progress.json", progress.as_dict())
    return {
        "step": step,
        "optimizer_step": actor_state.global_update_step,
        "reason": reason,
        "checkpoint_manager_saved": bool(saved),
        "items": str((step_dir / "items").resolve()),
        "teacher_items": str(teacher_dir.resolve()),
        "progress": str(progress_path.resolve()),
    }


def _restore_training_checkpoint(
    *,
    checkpoint_dir: Path,
    model: Any,
    mesh: Any,
    contract: OriginalRlcsdContract,
) -> tuple[OriginalRlcsdActorState, TrainingProgress]:
    from flax import nnx
    from maxtext.common import checkpointing, train_state_nnx

    progress = TrainingProgress.from_path(
        checkpoint_dir / "latest_progress.json",
        contract,
    )
    optimizer, _ = create_original_rlcsd_optimizer(model, contract)
    train_state = train_state_nnx.TrainStateNNX(model, optimizer)
    target = nnx.state(train_state)
    manager = _checkpoint_manager(checkpoint_dir, contract)
    restored, _ = checkpointing.load_state_if_possible(
        manager,
        None,
        "",
        "",
        16,
        target,
        dataset_type="synthetic",
    )
    if not restored:
        raise RuntimeError("resume directory has no restorable checkpoint")
    restored_state = restored["items"]
    nnx.replace_by_pure_dict(target, restored_state.to_pure_dict())
    nnx.update(train_state, target)
    _, params, _ = nnx.split(model, nnx.Param, ...)
    teacher_path = checkpoint_dir / str(progress.rollout_step) / "teacher_items"
    teacher_params = checkpointing.load_params_from_path(
        str(teacher_path),
        params,
        16,
    )
    actor_state = OriginalRlcsdActorState(
        optimizer=optimizer,
        teacher_params=teacher_params,
        global_update_step=progress.optimizer_step,
    )
    manager.close()
    del mesh
    return actor_state, progress


def _session_manifest(
    *,
    contract: OriginalRlcsdContract,
    progress: TrainingProgress,
    environment: Mapping[str, Any],
    dataset: Mapping[str, Any],
    sharding: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    validation_sources: Sequence[Mapping[str, Any]],
    update_reports: Sequence[Mapping[str, Any]],
    stopped_reason: str,
) -> dict[str, Any]:
    complete = (
        progress.epoch_index == contract.num_epochs
        and progress.pretrain_validation_complete
    )
    return {
        "schema_version": 1,
        "ok": True,
        "phase": "rlcsd_end_to_end" if complete else "rlcsd_training_shard",
        "is_rlcsd_end_to_end": complete,
        "is_resumable_original_contract_shard": not complete,
        "contract": contract.as_dict(),
        "case": ORIGINAL_RLCSD_CASE.as_dict(),
        "weights": "converted_pretrained_checkpoint",
        "environment": dict(environment),
        "dataset": dict(dataset),
        "parameter_sharding": dict(sharding),
        "progress": progress.as_dict(),
        "training": {
            "num_epochs_completed": progress.epoch_index,
            "max_train_samples": contract.max_train_samples,
            "steps_per_epoch": original_steps_per_epoch(contract),
            "total_rollout_steps": original_total_rollout_steps(contract),
            "rollout_steps_completed": progress.rollout_step,
            "optimizer_steps_completed": progress.optimizer_step,
            "prompts_consumed": progress.prompts_consumed,
            "session_update_reports": list(update_reports),
        },
        "validation": {
            "val_before_train": contract.val_before_train,
            "val_n": contract.val_n,
            "completed_datasets": (
                list(contract.val_datasets)
                if progress.pretrain_validation_complete
                else []
            ),
            "sources": list(validation_sources),
            "periodic_validation_steps": progress.periodic_validation_steps,
        },
        "checkpoints": {
            "save_steps": contract.save_steps,
            "saved_checkpoint_count": len(checkpoints),
            "periodic_checkpoint_steps": progress.periodic_checkpoint_steps,
            "session_checkpoints": list(checkpoints),
        },
        "stopped_reason": stopped_reason,
        "measurement_boundary": {
            "includes_decode": progress.rollout_step > 0,
            "includes_verifier": progress.rollout_step > 0,
            "includes_teacher_correct_wrong_passes": progress.optimizer_step > 0,
            "includes_rlcsd_policy_loss": progress.optimizer_step > 0,
            "includes_backward": progress.optimizer_step > 0,
            "includes_optimizer_update": progress.optimizer_step > 0,
            "full_30_epoch_training": complete,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-items", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--resume-checkpoint-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-micro-batch-size", type=int, default=16)
    parser.add_argument("--session-deadline-seconds", type=int, default=28_800)
    parser.add_argument("--deadline-safety-seconds", type=int, default=900)
    parser.add_argument("--session-max-rollout-batches", type=int)
    args = parser.parse_args()

    started = time.perf_counter()
    contract = load_original_contract()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = read_checkpoint_manifest(
        args.checkpoint_items,
        ORIGINAL_RLCSD_CASE,
    )
    devices, environment = require_tpu_v5e8()

    from flax import nnx
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
        max_records=contract.max_train_samples,
    )
    model, mesh, _, weights = load_model(
        ORIGINAL_RLCSD_CASE,
        devices=devices,
        checkpoint_items=args.checkpoint_items,
        allow_random_weights=False,
    )
    if weights != "converted_pretrained_checkpoint":
        raise RuntimeError("original training requires converted pretrained weights")
    sharding = parameter_sharding_report(model, devices)
    output_checkpoint_dir = args.output_dir / "checkpoints"
    if args.resume_checkpoint_dir:
        actor_state, progress = _restore_training_checkpoint(
            checkpoint_dir=args.resume_checkpoint_dir,
            model=model,
            mesh=mesh,
            contract=contract,
        )
    else:
        optimizer, _ = create_original_rlcsd_optimizer(model, contract)
        actor_state = OriginalRlcsdActorState(
            optimizer=optimizer,
            teacher_params=None,
            global_update_step=0,
        )
        progress = TrainingProgress.initial(contract)
    checkpoint_manager = _checkpoint_manager(
        output_checkpoint_dir,
        contract,
    )

    validation_sources: list[dict[str, Any]] = []
    validation_path = args.output_dir / "validation_samples.jsonl"
    if progress.phase in {"pretrain_validation", "periodic_validation"}:
        eval_config = build_offline_inference_config(
            contract,
            checkpoint_items=args.checkpoint_items,
            validation=True,
        )
        _, current_params, _ = nnx.split(model, nnx.Param, ...)
        eval_engine = create_offline_engine(
            eval_config,
            tokenizer,
            seed=args.seed,
            params=current_params,
            mesh=mesh,
        )
        for dataset_index in range(
            progress.pretrain_validation_dataset_index,
            len(contract.val_datasets),
        ):
            dataset_name = contract.val_datasets[dataset_index]
            records, source = _load_eval_records(
                dataset_name,
                tokenizer,
                contract,
                cache_dir=args.cache_dir,
            )
            validation_sources.append(source)
            record_start = (
                progress.pretrain_validation_record_index
                if dataset_index == progress.pretrain_validation_dataset_index
                else 0
            )
            for record_index in range(record_start, len(records)):
                if _deadline_reached(
                    started,
                    args.session_deadline_seconds,
                    safety_seconds=args.deadline_safety_seconds,
                ):
                    stopped_reason = "session_deadline_before_validation_prompt"
                    break
                rows = _generate_validation_samples(
                    eval_engine,
                    tokenizer,
                    records[record_index],
                    contract,
                    seed=args.seed + dataset_index * 100_000 + record_index,
                )
                _append_jsonl(validation_path, rows)
                progress.pretrain_validation_dataset_index = dataset_index
                progress.pretrain_validation_record_index = record_index + 1
                write_json(
                    args.output_dir / "live_progress.json",
                    progress.as_dict(),
                )
                if _deadline_reached(
                    started,
                    args.session_deadline_seconds,
                    safety_seconds=args.deadline_safety_seconds,
                ):
                    stopped_reason = "session_deadline_after_validation_prompt"
                    break
            else:
                progress.pretrain_validation_dataset_index = dataset_index + 1
                progress.pretrain_validation_record_index = 0
                continue
            break
        else:
            if progress.active_validation_step == 0:
                progress.pretrain_validation_complete = True
            elif progress.active_validation_step is not None:
                progress.periodic_validation_steps.append(
                    progress.active_validation_step
                )
            progress.active_validation_step = None
            progress.pretrain_validation_dataset_index = 0
            progress.pretrain_validation_record_index = 0
            progress.phase = "training"
        del eval_engine

    update_reports: list[dict[str, Any]] = []
    session_rollout_batches = 0
    stopped_reason = (
        "pretrain_validation_incomplete"
        if not progress.pretrain_validation_complete
        else "session_deadline"
    )
    if progress.phase == "training":
        train_config = build_offline_inference_config(
            contract,
            checkpoint_items=args.checkpoint_items,
            validation=False,
        )
        _, current_params, _ = nnx.split(model, nnx.Param, ...)
        train_engine = create_offline_engine(
            train_config,
            tokenizer,
            seed=args.seed,
            params=current_params,
            mesh=mesh,
        )
        while progress.epoch_index < contract.num_epochs:
            if _deadline_reached(
                started,
                args.session_deadline_seconds,
                safety_seconds=args.deadline_safety_seconds,
            ):
                stopped_reason = "session_deadline_before_rollout_batch"
                break
            if (
                args.session_max_rollout_batches is not None
                and session_rollout_batches >= args.session_max_rollout_batches
            ):
                stopped_reason = "explicit_session_rollout_batch_boundary"
                break
            prompt_batch = prompt_batch_for_progress(prompts, progress, contract)
            samples = generate_rollout_samples(
                prompt_batch,
                contract,
                tokenizer,
                train_engine,
                micro_batch_size=args.rollout_micro_batch_size,
                seed=args.seed + progress.rollout_step,
            )
            advantages = compute_grpo_advantages(samples, contract)
            assignments, alignment = build_teacher_assignments(
                samples,
                contract,
                seed=args.seed + progress.rollout_step,
            )
            if not assignments:
                raise RuntimeError("rollout produced no valid mixed RLCSD group")
            packed_batches = [
                pack_actor_minibatch(batch, advantages, tokenizer, contract)
                for batch in iter_actor_minibatches(assignments, contract)
            ]
            report, actor_state = run_original_rlcsd_updates(
                model=model,
                mesh=mesh,
                packed_batches=packed_batches,
                contract=contract,
                max_updates=None,
                actor_state=actor_state,
            )
            report["alignment"] = alignment
            update_reports.append(report)
            advance_training_progress(
                progress,
                contract,
                optimizer_step=actor_state.global_update_step,
            )
            session_rollout_batches += 1
            _, current_params, _ = nnx.split(model, nnx.Param, ...)
            update_offline_engine_params(train_engine, current_params)
            if progress.rollout_step % contract.save_steps == 0:
                progress.periodic_checkpoint_steps.append(progress.rollout_step)
            if progress.rollout_step % contract.eval_steps == 0:
                progress.phase = "periodic_validation"
                progress.active_validation_step = progress.rollout_step
                progress.pretrain_validation_dataset_index = 0
                progress.pretrain_validation_record_index = 0
                # The next session executes validation before another rollout.
                stopped_reason = "periodic_validation_boundary"
                break
        del train_engine

    if actor_state.teacher_params is None:
        from benchmarks.maxtext_tpu.rlcsd_actor import _snapshot_params

        actor_state.teacher_params = _snapshot_params(model)
    checkpoint_rows = [
        _save_training_checkpoint(
            checkpoint_manager=checkpoint_manager,
            checkpoint_dir=output_checkpoint_dir,
            model=model,
            actor_state=actor_state,
            progress=progress,
            reason=stopped_reason,
        )
    ]
    manifest = _session_manifest(
        contract=contract,
        progress=progress,
        environment=environment,
        dataset=dataset,
        sharding=sharding,
        checkpoints=checkpoint_rows,
        validation_sources=validation_sources,
        update_reports=update_reports,
        stopped_reason=stopped_reason,
    )
    write_json(args.output_dir / "training_session.json", manifest)
    print("RLCSD_ORIGINAL_TRAINING_SESSION " + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
