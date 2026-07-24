"""Fail-closed audit of benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import DATASET_REVISION


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
    elif phase == "rlcsd_end_to_end":
        boundary = payload.get("measurement_boundary", {})
        checks.update(
            {
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
