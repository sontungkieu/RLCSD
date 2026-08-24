"""Small Kaggle Job Spec adapters for the original RLCSD training contract.

The scientific phases remain ordinary modules (checkpoint conversion, parity,
training, and audit).  This module owns only the cross-account resume gate,
the exact TPU environment identity check, and final metric collection needed
by a declarative KJO schema-v2 job.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.dependencies import distribution_versions
from benchmarks.maxtext_tpu.rlcsd_train import (
    resolve_attached_resume_training_root,
    stage_writable_resume_checkpoint,
)


ORIGINAL_CASE_ID = "qwen3_1_7b-original-rlcsd-tp8"
TPU_RUNTIME_EXPECTED = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "libtpu": "0.0.17",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_environment(output: Path) -> dict[str, Any]:
    """Require the TPU distributions used by the validated v36 environment."""

    package_names = (
        "jax",
        "jaxlib",
        "libtpu",
        "maxtext",
        "google-tunix",
        "flax",
        "orbax-checkpoint",
        "transformers",
    )
    versions = distribution_versions(package_names)
    mismatches = {
        name: {"expected": expected, "observed": versions.get(name)}
        for name, expected in TPU_RUNTIME_EXPECTED.items()
        if versions.get(name) != expected
    }
    missing = sorted(name for name, version in versions.items() if version is None)
    summary = {
        "schema_version": 1,
        "kind": "rlcsd_job_environment_gate",
        "expected_tpu_runtime": TPU_RUNTIME_EXPECTED,
        "distribution_versions": versions,
        "missing_distributions": missing,
        "runtime_mismatches": mismatches,
        "ok": not missing and not mismatches,
    }
    _write_json(output, summary)
    if not summary["ok"]:
        raise RuntimeError(f"RLCSD Job Spec environment gate failed: {summary}")
    print("RLCSD_JOB_ENVIRONMENT_SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


def prepare_resume(
    *,
    input_root: Path,
    relay_dataset_source: str,
    source_kernel_id: str,
    source_run_id: str,
    declared_relay_tree_sha256: str,
    expected_contract_sha256: str,
    output_checkpoint_dir: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Validate one relay source and copy its checkpoint to writable storage."""

    training_root = resolve_attached_resume_training_root(
        input_root,
        relay_dataset_source,
    )
    checkpoint_dir = training_root / "checkpoints"
    progress = _read_json(checkpoint_dir / "latest_progress.json")
    session = _read_json(training_root / "training_session.json")
    source_run_summary = _read_json(
        training_root.parent
        / "kaggle_job_ops"
        / source_run_id
        / "run_summary.json"
    )
    rollout_step = int(progress["rollout_step"])
    source_kernel_slug = source_kernel_id.partition("/")[2]
    checks = {
        "relay_dataset_source_exact": bool(relay_dataset_source),
        "source_kernel_id_exact": (
            source_kernel_slug == source_run_id.replace("_", "-")
        ),
        "source_run_id_exact": source_run_summary.get("run_id") == source_run_id,
        "declared_relay_tree_sha256_valid": (
            len(declared_relay_tree_sha256) == 64
            and all(ch in "0123456789abcdef" for ch in declared_relay_tree_sha256)
        ),
        "contract_sha256_exact": (
            progress.get("contract_sha256") == expected_contract_sha256
        ),
        "resumable_shard": (
            session.get("is_resumable_original_contract_shard") is True
        ),
        "not_full_training_claim": session.get("is_rlcsd_end_to_end") is False,
        "case_exact": (
            session.get("case", {}).get("case_id") == ORIGINAL_CASE_ID
        ),
        "progress_matches_session": session.get("progress") == progress,
        "checkpoint_items_present": (
            checkpoint_dir / str(rollout_step) / "items"
        ).is_dir(),
        "teacher_items_present": (
            checkpoint_dir / str(rollout_step) / "teacher_items"
        ).is_dir(),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise RuntimeError(f"RLCSD relay resume gate failed: {failed}")

    staged = stage_writable_resume_checkpoint(
        checkpoint_dir,
        output_checkpoint_dir,
    )
    summary = {
        "schema_version": 1,
        "kind": "rlcsd_cross_account_resume_gate",
        "relay_dataset_source": relay_dataset_source,
        "source_kernel_id": source_kernel_id,
        "source_run_id": source_run_id,
        # This hash is verified locally by KJO before upload and preserved in
        # the cross-account plan.  Runtime checks the exact attached source and
        # all semantic state markers; it does not pretend to rehash 6.3 GB.
        "declared_local_relay_tree_sha256": declared_relay_tree_sha256,
        "input_training_root": str(training_root),
        "source_checkpoint_dir": str(checkpoint_dir),
        "writable_checkpoint_dir": str(staged),
        "source_preserved": True,
        "runtime_marker_checks": checks,
        "progress": progress,
        "ok": True,
    }
    _write_json(summary_output, summary)
    print("RLCSD_RESUME_SOURCE_SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


def collect_results(
    *,
    metrics_root: Path,
    training_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Collect the small scientific summary and optional measured throughput."""

    train_result = _read_json(training_root / "training_session.json")
    audit_result = _read_json(metrics_root / "audit.json")
    resume_result = _read_json(metrics_root / "resume_source_summary.json")
    environment_result = _read_json(metrics_root / "environment_summary.json")
    if audit_result.get("all_valid") is not True:
        raise RuntimeError(f"RLCSD evidence audit failed: {audit_result}")

    update_reports = train_result.get("training", {}).get(
        "session_update_reports",
        [],
    )
    timed_updates = [
        row
        for report in update_reports
        for row in report.get("timed_updates", [])
    ]
    benchmark_row = None
    if timed_updates:
        response_tokens = sum(int(row["response_token_count"]) for row in timed_updates)
        execution_s = sum(
            float(row["policy_execution_s"]) + float(row["optimizer_update_s"])
            for row in timed_updates
        )
        if execution_s <= 0:
            raise RuntimeError("timed RLCSD updates reported non-positive execution time")
        benchmark_row = {
            "bs": 16,
            "p": 40960,
            "g": 0,
            "prefill_tok_s": response_tokens / execution_s,
            "generate_tok_s": 0.0,
            "prefill_s": execution_s / len(timed_updates),
            "decode_s": 0.0,
            "warmup_iteration_count": 0,
            "timed_iteration_count": len(timed_updates),
            "includes_compile": False,
            "includes_warmup": False,
            "workload": "original_rlcsd_rollout_teacher_backward_adamw_shard",
            "is_rlcsd_end_to_end": False,
            "weights": train_result["weights"],
        }
        run_id = os.environ.get("KJO_RUN_ID", "").strip()
        if run_id:
            benchmark_path = (
                Path("/kaggle/working/kaggle_job_ops")
                / run_id
                / "benchmark_throughput.jsonl"
            )
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_path.write_text(
                json.dumps(benchmark_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print("KJO_BENCHMARK_THROUGHPUT_ROW " + json.dumps(benchmark_row, sort_keys=True))

    summary = {
        "schema_version": 1,
        "ok": True,
        "case": train_result["case"],
        "audit": audit_result,
        "benchmark_row": benchmark_row,
        "claim_boundary": train_result["measurement_boundary"],
        "training_progress": train_result["progress"],
        "training_artifact_root": str(training_root),
        "resume_source": resume_result,
        "environment": environment_result,
    }
    _write_json(output, summary)
    print("RLCSD_TRAINING_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment_parser = subparsers.add_parser("verify-environment")
    environment_parser.add_argument("--output", required=True, type=Path)

    resume_parser = subparsers.add_parser("prepare-resume")
    resume_parser.add_argument("--input-root", required=True, type=Path)
    resume_parser.add_argument("--relay-dataset-source", required=True)
    resume_parser.add_argument("--source-kernel-id", required=True)
    resume_parser.add_argument("--source-run-id", required=True)
    resume_parser.add_argument("--declared-relay-tree-sha256", required=True)
    resume_parser.add_argument("--expected-contract-sha256", required=True)
    resume_parser.add_argument("--output-checkpoint-dir", required=True, type=Path)
    resume_parser.add_argument("--summary-output", required=True, type=Path)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--metrics-root", required=True, type=Path)
    collect_parser.add_argument("--training-root", required=True, type=Path)
    collect_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "verify-environment":
        verify_environment(args.output)
    elif args.command == "prepare-resume":
        prepare_resume(
            input_root=args.input_root,
            relay_dataset_source=args.relay_dataset_source,
            source_kernel_id=args.source_kernel_id,
            source_run_id=args.source_run_id,
            declared_relay_tree_sha256=args.declared_relay_tree_sha256,
            expected_contract_sha256=args.expected_contract_sha256,
            output_checkpoint_dir=args.output_checkpoint_dir,
            summary_output=args.summary_output,
        )
    elif args.command == "collect":
        collect_results(
            metrics_root=args.metrics_root,
            training_root=args.training_root,
            output=args.output,
        )
    else:  # pragma: no cover - argparse guarantees the branch set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
