"""Fail-closed audit of benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import DATASET_REVISION
from benchmarks.maxtext_tpu.rlcsd_contract import load_original_contract


def _positive_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_update_rows(rows: Any) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    required = (
        "loss",
        "grad_norm",
        "learning_rate",
        "policy_execution_s",
        "optimizer_update_s",
        "response_tokens_per_s",
    )
    return all(
        isinstance(row, dict)
        and all(_finite(row.get(key)) for key in required)
        and row.get("grad_norm", 0) > 0
        and row.get("policy_execution_s", 0) > 0
        and row.get("optimizer_update_s", 0) > 0
        and row.get("optimizer_update_applied") is True
        and row.get("response_token_count", 0) > 0
        and row.get("response_tokens_per_s", 0) > 0
        for row in rows
    )


def audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    phase = payload.get("phase")
    checks: dict[str, bool] = {
        "payload_ok": payload.get("ok") is True,
    }
    environment = payload.get("environment", {})
    dataset = payload.get("dataset", {})
    if phase != "decode_capability":
        checks.update(
            {
                "jax_tpu": environment.get("backend") == "tpu",
                "eight_devices": environment.get("device_count") == 8,
                "v5e_kind": environment.get("v5e_kind_match") is True,
                "pinned_dataset": (
                    dataset.get("resolved_revision") == DATASET_REVISION
                ),
                "real_checkpoint": (
                    payload.get("weights") == "converted_pretrained_checkpoint"
                ),
            }
        )
    if phase in {"forward", "core-train-step"}:
        boundary = payload.get("measurement_boundary", {})
        sharding = payload.get("parameter_sharding", {})
        timing = payload.get("timing", {})
        throughput = payload.get("throughput", {})
        checks.update(
            {
                "compile_excluded": (boundary.get("compile_excluded") is True),
                "warmup_excluded": (boundary.get("warmup_excluded") is True),
                "timed_rows": bool(timing.get("timed_step_s")),
                "positive_throughput": (throughput.get("padded_tokens_per_s", 0) > 0),
                "stage_sharding_observed": (
                    sharding.get("stage_sharded_leaf_count", 0) > 0
                ),
                "tensor_sharding_observed": (
                    sharding.get("tensor_sharded_leaf_count", 0) > 0
                ),
            }
        )
    elif phase == "checkpoint_logits_parity":
        checks["parity_checks"] = all(payload.get("checks", {}).values())
    elif phase == "decode_capability":
        checks = {
            "capability_report": True,
            "decode_supported": (payload.get("kv_cache_decode_supported") is True),
            "throughput_available": (payload.get("throughput_available") is True),
        }
    elif phase == "rlcsd_rollout":
        expected_contract = load_original_contract().as_dict()
        rollout = payload.get("rollout", {})
        case = payload.get("case", {})
        boundary = payload.get("measurement_boundary", {})
        checks.update(
            {
                "original_contract_exact": (
                    payload.get("contract") == expected_contract
                ),
                "original_profile": (
                    case.get("measurement_profile") == "original_rlcsd"
                    and case.get("sequence_length") == 40960
                ),
                "rollout_prompt_batch_64": (rollout.get("prompt_group_count") == 64),
                "rollout_group_size_8": rollout.get("group_size") == 8,
                "rollout_sample_batch_512": (rollout.get("sample_count") == 512),
                "kv_cache_decode": (
                    rollout.get("decode_backend") == "maxtext_offline_engine_kv_cache"
                ),
                "verifier_executed": boundary.get("includes_verifier") is True,
                "rollout_artifact_hash": bool(rollout.get("output_sha256")),
                "not_actor_update": (
                    boundary.get("includes_optimizer_update") is False
                ),
            }
        )
    elif phase == "rlcsd_original_update":
        expected_contract = load_original_contract().as_dict()
        rollout = payload.get("rollout", {})
        alignment = payload.get("alignment", {})
        summary = alignment.get("summary", {})
        case = payload.get("case", {})
        boundary = payload.get("measurement_boundary", {})
        timed_rows = payload.get("timed_updates")
        update_count = payload.get("update_count")
        expected_updates = payload.get("expected_full_update_count")
        expected_syncs = (
            math.ceil(update_count / expected_contract["teacher_sync_interval"])
            if isinstance(update_count, int) and update_count > 0
            else None
        )
        checks.update(
            {
                "original_contract_exact": (
                    payload.get("contract") == expected_contract
                ),
                "original_core_update_claim": (
                    payload.get("is_original_rlcsd_core_update") is True
                    and payload.get("is_rlcsd_end_to_end") is False
                ),
                "not_diagnostic_cap": (payload.get("diagnostic_update_cap") is None),
                "original_profile": (
                    case.get("measurement_profile") == "original_rlcsd"
                    and case.get("sequence_length") == 40960
                    and case.get("global_batch_size") == 16
                    and case.get("parallelism", {}).get("tensor_parallelism") == 8
                ),
                "rollout_prompt_batch_64": (rollout.get("prompt_group_count") == 64),
                "rollout_group_size_8": rollout.get("group_size") == 8,
                "rollout_sample_batch_512": (rollout.get("sample_count") == 512),
                "teacher_alignment_nonempty": (
                    alignment.get("ok") is True
                    and alignment.get("assignment_count", 0) > 0
                    and summary.get("group_count") == 64
                ),
                "full_valid_actor_batch_consumed": (
                    isinstance(update_count, int)
                    and update_count == expected_updates
                    and update_count >= 2
                ),
                "snapshot_sync_interval_10": (
                    payload.get("teacher_sync_event_count") == expected_syncs
                ),
                "compile_excluded": boundary.get("compile_excluded") is True,
                "warmup_excluded": boundary.get("warmup_excluded") is True,
                "decode_included": boundary.get("includes_decode") is True,
                "verifier_included": (boundary.get("includes_verifier") is True),
                "teacher_paths_included": (
                    boundary.get("includes_teacher_correct_wrong_passes") is True
                ),
                "rlcsd_loss_included": (
                    boundary.get("includes_rlcsd_policy_loss") is True
                ),
                "backward_included": (boundary.get("includes_backward") is True),
                "optimizer_included": (
                    boundary.get("includes_optimizer_update") is True
                ),
                "not_full_30_epoch_claim": (
                    boundary.get("full_30_epoch_training") is False
                ),
                "timed_updates_finite": _finite_update_rows(timed_rows),
                "timed_throughput": _positive_finite(
                    payload.get("mean_timed_response_tokens_per_s")
                ),
                "nonzero_lr_adamw_update": (
                    payload.get("nonzero_learning_rate_update_count", 0) > 0
                ),
                "tensor_sharding_observed": (
                    payload.get("parameter_sharding", {}).get(
                        "tensor_sharded_leaf_count",
                        0,
                    )
                    > 0
                ),
            }
        )
    elif phase == "rlcsd_end_to_end":
        expected_contract = load_original_contract().as_dict()
        boundary = payload.get("measurement_boundary", {})
        training = payload.get("training", {})
        validation = payload.get("validation", {})
        checkpoints = payload.get("checkpoints", {})
        total_rollout_steps = (
            expected_contract["max_train_samples"]
            // expected_contract["rollout_prompt_batch_size"]
        ) * expected_contract["num_epochs"]
        expected_periodic_steps = list(
            range(
                expected_contract["eval_steps"],
                total_rollout_steps + 1,
                expected_contract["eval_steps"],
            )
        )
        checks.update(
            {
                "original_contract_exact": (
                    payload.get("contract") == expected_contract
                ),
                "declared_end_to_end": (payload.get("is_rlcsd_end_to_end") is True),
                "rollout_included": boundary.get("includes_decode") is True,
                "verifier_included": (boundary.get("includes_verifier") is True),
                "teacher_paths_included": (
                    boundary.get("includes_teacher_correct_wrong_passes") is True
                ),
                "rlcsd_loss_included": (
                    boundary.get("includes_rlcsd_policy_loss") is True
                ),
                "optimizer_included": (
                    boundary.get("includes_optimizer_update") is True
                ),
                "thirty_epochs_completed": (
                    training.get("num_epochs_completed")
                    == expected_contract["num_epochs"]
                ),
                "train_sample_cap_60000": (
                    training.get("max_train_samples")
                    == expected_contract["max_train_samples"]
                ),
                "all_drop_last_rollout_steps_completed": (
                    training.get("total_rollout_steps") == total_rollout_steps
                    and training.get("rollout_steps_completed") == total_rollout_steps
                    and training.get("prompts_consumed")
                    == total_rollout_steps
                    * expected_contract["rollout_prompt_batch_size"]
                    and training.get("optimizer_steps_completed", 0) > 0
                ),
                "validation_suites_complete": (
                    sorted(validation.get("completed_datasets", []))
                    == sorted(expected_contract["val_datasets"])
                ),
                "validation_schedule_complete": (
                    validation.get("val_before_train") is True
                    and validation.get("val_n") == expected_contract["val_n"]
                    and validation.get("periodic_validation_steps")
                    == expected_periodic_steps
                ),
                "save_interval_50": (
                    checkpoints.get("save_steps") == expected_contract["save_steps"]
                    and checkpoints.get("saved_checkpoint_count", 0) > 0
                    and checkpoints.get("periodic_checkpoint_steps")
                    == expected_periodic_steps
                ),
                "full_training_boundary": (
                    boundary.get("full_30_epoch_training") is True
                ),
            }
        )
    elif phase == "rlcsd_training_shard":
        expected_contract = load_original_contract().as_dict()
        progress = payload.get("progress", {})
        training = payload.get("training", {})
        validation = payload.get("validation", {})
        checkpoints = payload.get("checkpoints", {})
        case = payload.get("case", {})
        checks.update(
            {
                "original_contract_exact": (
                    payload.get("contract") == expected_contract
                ),
                "declared_resumable_shard": (
                    payload.get("is_resumable_original_contract_shard") is True
                    and payload.get("is_rlcsd_end_to_end") is False
                ),
                "original_profile": (
                    case.get("measurement_profile") == "original_rlcsd"
                    and case.get("sequence_length") == 40960
                    and case.get("parallelism", {}).get("tensor_parallelism") == 8
                ),
                "full_contract_target_retained": (
                    training.get("max_train_samples")
                    == expected_contract["max_train_samples"]
                    and training.get("total_rollout_steps")
                    == (
                        expected_contract["max_train_samples"]
                        // expected_contract["rollout_prompt_batch_size"]
                    )
                    * expected_contract["num_epochs"]
                ),
                "progress_is_bounded": (
                    isinstance(progress.get("epoch_index"), int)
                    and 0 <= progress["epoch_index"] <= expected_contract["num_epochs"]
                    and isinstance(progress.get("rollout_step"), int)
                    and progress["rollout_step"] >= 0
                    and isinstance(progress.get("optimizer_step"), int)
                    and progress["optimizer_step"] >= 0
                ),
                "validation_contract_retained": (
                    validation.get("val_before_train") is True
                    and validation.get("val_n") == expected_contract["val_n"]
                ),
                "save_interval_retained": (
                    checkpoints.get("save_steps") == expected_contract["save_steps"]
                ),
                "resumable_checkpoint_written": (
                    checkpoints.get("saved_checkpoint_count", 0) > 0
                    and bool(checkpoints.get("session_checkpoints"))
                ),
                "not_full_training_claim": (
                    payload.get("measurement_boundary", {}).get(
                        "full_30_epoch_training"
                    )
                    is False
                ),
            }
        )
    verdict = all(checks.values())
    return {
        "schema_version": 1,
        "phase": phase,
        "verdict": verdict,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for input_path in args.inputs:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": str(input_path),
                **audit_payload(payload),
            }
        )
    report = {
        "schema_version": 1,
        "all_valid": all(row["verdict"] for row in rows),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not report["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
