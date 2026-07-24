from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp

from benchmarks.maxtext_tpu.audit import audit_payload
from benchmarks.maxtext_tpu.matrix import (
    DATASET_REVISION,
    get_case,
    iter_cases,
    validate_repo_configs,
)
from benchmarks.maxtext_tpu.rlcsd_jax import rlcsd_policy_loss
from benchmarks.maxtext_tpu.runtime import read_checkpoint_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_matrix_has_three_models_two_layouts_and_three_batches():
    cases = list(iter_cases())
    assert len(cases) == 18
    assert {case.model.key for case in cases} == {
        "qwen3_1_7b",
        "qwen3_4b",
        "qwen3_8b",
    }
    assert {case.global_batch_size for case in cases} == {12, 16, 64}
    assert {case.layout.label for case in cases} == {
        "PP2xTP4",
        "PP4xTP2",
    }
    for case in cases:
        case.validate()
        assert case.layout.required_device_count == 8


def test_pp4_batch12_uses_four_microbatches():
    case = get_case("qwen3_8b-bs12-pp4xtp2")
    assert case.num_pipeline_microbatches == 4
    assert case.global_batch_size % case.num_pipeline_microbatches == 0


def test_matrix_matches_checked_in_rlcsd_configs():
    validate_repo_configs(REPO_ROOT)


def test_checkpoint_manifest_is_reusable_across_same_model_cases(tmp_path):
    checkpoint_root = tmp_path / "qwen3_8b"
    items = checkpoint_root / "0" / "items"
    items.mkdir(parents=True)
    (checkpoint_root / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "source_case_id": "qwen3_8b-bs12-pp2xtp4",
                "model_id": "Qwen/Qwen3-8B",
                "resolved_hf_revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    manifest = read_checkpoint_manifest(items, get_case("qwen3_8b-bs64-pp4xtp2"))
    assert manifest["source_case_id"] == "qwen3_8b-bs12-pp2xtp4"


def test_rlcsd_jax_zero_contrast_reduces_to_ppo_path():
    loss, metrics = rlcsd_policy_loss(
        old_log_prob=jnp.zeros((1, 2)),
        log_prob=jnp.zeros((1, 2)),
        advantages=jnp.asarray([[1.0, -1.0]]),
        response_mask=jnp.ones((1, 2)),
        teacher_correct_log_prob=jnp.zeros((1, 2)),
        teacher_wrong_multi_log_prob=jnp.zeros((1, 2, 2)),
        teacher_wrong_multi_valid_mask=jnp.ones((1, 2)),
    )
    assert math.isclose(float(loss), 0.0, abs_tol=1e-6)
    assert float(metrics["selected_token_count"]) == 0.0
    assert float(metrics["k_valid_mean"]) == 2.0


def test_rlcsd_jax_modulates_positive_advantage():
    loss, metrics = rlcsd_policy_loss(
        old_log_prob=jnp.zeros((1, 1)),
        log_prob=jnp.zeros((1, 1)),
        advantages=jnp.ones((1, 1)),
        response_mask=jnp.ones((1, 1)),
        teacher_correct_log_prob=jnp.ones((1, 1)),
        teacher_wrong_multi_log_prob=jnp.zeros((1, 2, 1)),
        teacher_wrong_multi_valid_mask=jnp.ones((1, 2)),
        tau=1e-3,
        beta=1.0,
        lam=0.5,
        delta=0.02,
    )
    assert math.isclose(float(loss), -1.5, abs_tol=1e-5)
    assert float(metrics["selected_token_count"]) == 1.0


def _forward_payload(weights: str) -> dict:
    return {
        "ok": True,
        "phase": "forward",
        "weights": weights,
        "environment": {
            "backend": "tpu",
            "device_count": 8,
            "v5e_kind_match": True,
        },
        "dataset": {"resolved_revision": DATASET_REVISION},
        "measurement_boundary": {
            "compile_excluded": True,
            "warmup_excluded": True,
        },
        "timing": {"timed_step_s": [0.1, 0.1]},
        "throughput": {"padded_tokens_per_s": 100.0},
        "parameter_sharding": {
            "stage_sharded_leaf_count": 1,
            "tensor_sharded_leaf_count": 1,
        },
    }


def test_audit_rejects_random_weights_for_scientific_result():
    report = audit_payload(_forward_payload("random_shape_correct"))
    assert report["verdict"] is False
    assert "real_checkpoint" in report["failed_checks"]


def test_audit_accepts_evidence_complete_forward():
    report = audit_payload(_forward_payload("converted_pretrained_checkpoint"))
    assert report["verdict"] is True


def test_decode_capability_is_not_decode_throughput():
    report = audit_payload(
        {
            "phase": "decode_capability",
            "kv_cache_decode_supported": False,
            "throughput_available": False,
        }
    )
    assert report["verdict"] is False
    assert set(report["failed_checks"]) == {
        "decode_supported",
        "throughput_available",
    }
