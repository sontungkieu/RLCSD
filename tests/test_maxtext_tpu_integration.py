from __future__ import annotations

import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import jax.numpy as jnp
import pytest

from benchmarks.maxtext_tpu import runtime
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
    source_case = get_case("qwen3_8b-bs12-pp2xtp4")
    (checkpoint_root / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "source_case_id": source_case.case_id,
                "model_id": "Qwen/Qwen3-8B",
                "resolved_hf_revision": "a" * 40,
                "scan_layers": True,
                "maxtext_pipeline": source_case.maxtext_pipeline_kwargs,
            }
        ),
        encoding="utf-8",
    )
    manifest = read_checkpoint_manifest(items, get_case("qwen3_8b-bs64-pp2xtp4"))
    assert manifest["source_case_id"] == "qwen3_8b-bs12-pp2xtp4"


def test_checkpoint_manifest_rejects_different_pipeline_structure(tmp_path):
    checkpoint_root = tmp_path / "qwen3_8b"
    items = checkpoint_root / "0" / "items"
    items.mkdir(parents=True)
    source_case = get_case("qwen3_8b-bs12-pp2xtp4")
    (checkpoint_root / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "source_case_id": source_case.case_id,
                "model_id": source_case.model.model_id,
                "resolved_hf_revision": "a" * 40,
                "scan_layers": True,
                "maxtext_pipeline": source_case.maxtext_pipeline_kwargs,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pipeline structure"):
        read_checkpoint_manifest(items, get_case("qwen3_8b-bs12-pp4xtp2"))


@pytest.mark.parametrize(
    ("use_checkpoint", "expected_enable_checkpointing"),
    [(True, True), (False, False)],
)
def test_load_model_enables_maxtext_checkpoint_loading_only_when_needed(
    tmp_path,
    monkeypatch,
    use_checkpoint,
    expected_enable_checkpointing,
):
    captured = {}

    class FakeParallelConfig:
        def create_mesh(self, *, devices):
            assert devices == ["tpu"]
            return "mesh"

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return "model", kwargs["model_path"]

    fake_jax = types.ModuleType("jax")
    fake_jax.set_mesh = lambda mesh: nullcontext()
    fake_automodel = types.ModuleType("tunix.models.automodel")
    fake_automodel.AutoModel = FakeAutoModel
    fake_automodel.ModelSource = types.SimpleNamespace(MAXTEXT="maxtext")
    fake_models = types.ModuleType("tunix.models")
    fake_models.automodel = fake_automodel
    fake_tunix = types.ModuleType("tunix")
    fake_tunix.models = fake_models

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "tunix", fake_tunix)
    monkeypatch.setitem(sys.modules, "tunix.models", fake_models)
    monkeypatch.setitem(
        sys.modules,
        "tunix.models.automodel",
        fake_automodel,
    )
    monkeypatch.setattr(
        runtime,
        "create_parallel_config",
        lambda case: FakeParallelConfig(),
    )
    monkeypatch.setattr(
        runtime,
        "read_checkpoint_manifest",
        lambda checkpoint_items, case: {},
    )

    checkpoint_items = None
    if use_checkpoint:
        checkpoint_items = tmp_path / "checkpoint" / "0" / "items"
        checkpoint_items.mkdir(parents=True)

    runtime.load_model(
        get_case("qwen3_1_7b-bs12-pp2xtp4"),
        devices=["tpu"],
        checkpoint_items=checkpoint_items,
        allow_random_weights=not use_checkpoint,
    )

    assert captured["enable_checkpointing"] is expected_enable_checkpointing
    assert captured["scan_layers"] is True
    assert captured["model_path"] == (
        str(checkpoint_items.resolve()) if checkpoint_items else None
    )


def test_original_tp8_uses_ordinary_maxtext_mesh_without_pipeline_config(
    tmp_path,
    monkeypatch,
):
    captured = {}
    axis_names = (
        "diloco",
        "data",
        "stage",
        "fsdp",
        "fsdp_transpose",
        "context",
        "context_autoregressive",
        "tensor",
        "tensor_sequence",
        "expert",
        "autoregressive",
    )

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(**kwargs):
            captured["model_kwargs"] = kwargs
            return "model", kwargs["model_path"]

    def fake_create_mesh(mesh_shape, mesh_axis_names, *, devices):
        captured["mesh_shape"] = mesh_shape
        captured["mesh_axis_names"] = mesh_axis_names
        captured["devices"] = devices
        return "ordinary-tp8-mesh"

    fake_jax = types.ModuleType("jax")
    fake_jax.set_mesh = lambda mesh: nullcontext()
    fake_automodel = types.ModuleType("tunix.models.automodel")
    fake_automodel.AutoModel = FakeAutoModel
    fake_automodel.ModelSource = types.SimpleNamespace(MAXTEXT="maxtext")
    fake_parallelism = types.ModuleType("tunix.models.maxtext_parallelism")
    fake_parallelism.MAXTEXT_MESH_AXIS_NAMES = axis_names
    fake_mesh = types.ModuleType("tunix.utils.mesh")
    fake_mesh.create_mesh = fake_create_mesh
    fake_utils = types.ModuleType("tunix.utils")
    fake_utils.mesh = fake_mesh
    fake_models = types.ModuleType("tunix.models")
    fake_models.automodel = fake_automodel
    fake_models.maxtext_parallelism = fake_parallelism
    fake_tunix = types.ModuleType("tunix")
    fake_tunix.models = fake_models
    fake_tunix.utils = fake_utils

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "tunix", fake_tunix)
    monkeypatch.setitem(sys.modules, "tunix.models", fake_models)
    monkeypatch.setitem(
        sys.modules,
        "tunix.models.automodel",
        fake_automodel,
    )
    monkeypatch.setitem(
        sys.modules,
        "tunix.models.maxtext_parallelism",
        fake_parallelism,
    )
    monkeypatch.setitem(sys.modules, "tunix.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "tunix.utils.mesh", fake_mesh)
    monkeypatch.setattr(
        runtime,
        "read_checkpoint_manifest",
        lambda checkpoint_items, case: {},
    )

    checkpoint_items = tmp_path / "checkpoint" / "0" / "items"
    checkpoint_items.mkdir(parents=True)
    case = get_case("qwen3_1_7b-original-rlcsd-tp8")
    devices = [f"tpu-{index}" for index in range(8)]

    model, mesh, _, weights = runtime.load_model(
        case,
        devices=devices,
        checkpoint_items=checkpoint_items,
        allow_random_weights=False,
    )

    assert model == "model"
    assert mesh == "ordinary-tp8-mesh"
    assert weights == "converted_pretrained_checkpoint"
    assert captured["mesh_axis_names"] == axis_names
    assert captured["mesh_shape"][axis_names.index("stage")] == 1
    assert captured["mesh_shape"][axis_names.index("tensor")] == 8
    assert captured["devices"] == devices
    assert captured["model_kwargs"]["maxtext_pipeline_config"] is None
    assert captured["model_kwargs"]["ici_pipeline_parallelism"] == 1
    assert captured["model_kwargs"]["ici_tensor_parallelism"] == 8


def test_pipeline_layout_still_uses_tunix_pipeline_config(monkeypatch):
    captured = {}

    class FakePipelineConfig:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def validate_batch_size(self, global_batch_size):
            captured["global_batch_size"] = global_batch_size

    fake_parallelism = types.ModuleType("tunix.models.maxtext_parallelism")
    fake_parallelism.MaxTextPipelineConfig = FakePipelineConfig
    fake_models = types.ModuleType("tunix.models")
    fake_models.maxtext_parallelism = fake_parallelism
    fake_tunix = types.ModuleType("tunix")
    fake_tunix.models = fake_models
    monkeypatch.setitem(sys.modules, "tunix", fake_tunix)
    monkeypatch.setitem(sys.modules, "tunix.models", fake_models)
    monkeypatch.setitem(
        sys.modules,
        "tunix.models.maxtext_parallelism",
        fake_parallelism,
    )

    config = runtime.create_parallel_config(get_case("qwen3_1_7b-bs12-pp2xtp4"))

    assert isinstance(config, FakePipelineConfig)
    assert captured["kwargs"]["pipeline_parallelism"] == 2
    assert captured["kwargs"]["tensor_parallelism"] == 4
    assert captured["global_batch_size"] == 12


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
