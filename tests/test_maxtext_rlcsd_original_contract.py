from __future__ import annotations

import dataclasses
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from benchmarks.maxtext_tpu.audit import audit_payload
from benchmarks.maxtext_tpu.matrix import ORIGINAL_RLCSD_CASE, get_case
from benchmarks.maxtext_tpu.rlcsd_actor import (
    plan_actor_update_chunks,
    plan_teacher_snapshot_windows,
    plan_token_budget_chunks,
)
from benchmarks.maxtext_tpu.rlcsd_contract import load_original_contract
from benchmarks.maxtext_tpu.rlcsd_data import (
    build_teacher_assignments,
    compute_grpo_advantages,
    pack_actor_minibatch,
)
from benchmarks.maxtext_tpu.rlcsd_rollout import (
    RolloutSample,
    _install_decoupled_prefill_compat,
    _install_offline_engine_size_one_int_compat,
    _register_result_tokens_pytree,
    create_offline_engine,
    update_offline_engine_params,
    validate_rollout_samples,
)
from benchmarks.maxtext_tpu.rlcsd_train import (
    TrainingProgress,
    advance_training_progress,
    original_steps_per_epoch,
    original_total_rollout_steps,
    resolve_attached_resume_training_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        text = "\n".join(item["content"] for item in messages)
        width = 5 + (len(text) % 7)
        return list(range(10, 10 + width))


def _sample(
    prompt_index: int,
    generation_index: int,
    *,
    reward: float,
    answer: str | None,
) -> RolloutSample:
    return RolloutSample(
        prompt_id=f"prompt-{prompt_index:02d}",
        row_index=prompt_index,
        generation_index=generation_index,
        data_source="openai/gsm8k",
        problem=f"problem {prompt_index}",
        ground_truth="42",
        reference_solution="reference solution",
        prompt_token_ids=(10, 11, 12),
        response_token_ids=(20, 21),
        response_text=(
            f"reasoning \\boxed{{{answer}}}" if answer is not None else "reasoning"
        ),
        old_log_probs=(-0.1, -0.2),
        reward=reward,
        answer=answer,
    )


def _mixed_group(prompt_index: int) -> list[RolloutSample]:
    return [
        _sample(
            prompt_index,
            generation_index,
            reward=1.0 if generation_index < 2 else 0.0,
            answer="42" if generation_index < 2 else str(generation_index),
        )
        for generation_index in range(8)
    ]


def test_original_contract_derives_launcher_values_and_full_yaml_controls():
    contract = load_original_contract(REPO_ROOT)
    assert contract.rollout_prompt_batch_size == 64
    assert contract.group_size == 8
    assert contract.rollout_sample_batch_size == 512
    assert contract.ppo_mini_batch_size == 16
    assert contract.ppo_micro_batch_size_per_worker == 4
    assert contract.max_prompt_length == 2048
    assert contract.max_completion_length == 16384
    assert contract.train_max_sequence_length == 18432
    assert contract.val_max_sequence_length == 40960
    assert contract.actor_max_token_len_per_worker == 40960
    assert contract.num_epochs == 30
    assert contract.max_train_samples == 60000
    assert contract.val_datasets == ("amc23", "aime24", "aime25")
    assert contract.save_steps == contract.eval_steps == 50


def test_original_case_is_not_the_seq512_canary_matrix():
    original = get_case("qwen3_1_7b-original-rlcsd-tp8")
    canary = get_case("qwen3_1_7b-bs12-pp2xtp4")
    assert original == ORIGINAL_RLCSD_CASE
    assert original.sequence_length == 40960
    assert original.measurement_profile == "original_rlcsd"
    assert original.as_dict()["is_original_rlcsd_config"] is True
    assert canary.sequence_length == 512
    assert canary.measurement_profile == "core_step_canary"
    assert canary.as_dict()["is_original_rlcsd_config"] is False


def _write_resume_markers(training_root: Path) -> None:
    (training_root / "checkpoints").mkdir(parents=True)
    (training_root / "training_session.json").write_text("{}\n", encoding="utf-8")
    (training_root / "checkpoints" / "latest_progress.json").write_text(
        "{}\n",
        encoding="utf-8",
    )


def test_resume_root_resolves_kaggle_notebook_output_namespace(tmp_path):
    input_root = tmp_path / "input"
    expected = (
        input_root
        / "notebooks"
        / "anhhaphan"
        / "rlcsd-qwen3-1p7b-original-contract-tp8-train-v31"
        / "output"
        / "rlcsd_original_training"
    )
    _write_resume_markers(expected)
    _write_resume_markers(
        input_root
        / "notebooks"
        / "other-owner"
        / "rlcsd-qwen3-1p7b-original-contract-tp8-train-v31"
        / "output"
        / "rlcsd_original_training"
    )

    resolved = resolve_attached_resume_training_root(
        input_root,
        "anhhaphan/rlcsd-qwen3-1p7b-original-contract-tp8-train-v31",
    )

    assert resolved == expected


def test_resume_root_fails_closed_on_ambiguous_exact_source(tmp_path):
    input_root = tmp_path / "input"
    source_root = (
        input_root
        / "notebooks"
        / "anhhaphan"
        / "rlcsd-qwen3-1p7b-original-contract-tp8-train-v31"
    )
    _write_resume_markers(source_root / "rlcsd_original_training")
    _write_resume_markers(source_root / "output" / "rlcsd_original_training")

    with pytest.raises(FileNotFoundError, match="found 2"):
        resolve_attached_resume_training_root(
            input_root,
            "anhhaphan/rlcsd-qwen3-1p7b-original-contract-tp8-train-v31",
        )


def test_offline_engine_decouples_optional_jetstream_before_import(monkeypatch):
    import benchmarks.maxtext_tpu.rlcsd_rollout as rollout_module

    observed = {}

    class FakeNnxState:
        def __init__(self):
            self.to_pure_dict_calls = 0

        def to_pure_dict(self):
            self.to_pure_dict_calls += 1
            return {"base": {"decoder": {"weight": "array"}}}

    class FakeOfflineEngine:
        def __init__(self, **kwargs):
            observed["decoupled_at_import"] = os.environ.get("DECOUPLE_GCLOUD")
            observed["kwargs"] = kwargs

    class FakeInferenceWorker:
        pass

    offline_engine_module = types.ModuleType(
        "maxtext.inference.offline_engine"
    )
    offline_engine_module.InferenceWorker = FakeInferenceWorker
    offline_engine_module.OfflineEngine = FakeOfflineEngine
    monkeypatch.setitem(
        sys.modules,
        "maxtext.inference.offline_engine",
        offline_engine_module,
    )
    monkeypatch.setattr(
        rollout_module,
        "_register_decoupled_result_tokens_pytree",
        lambda: True,
    )
    monkeypatch.delenv("DECOUPLE_GCLOUD", raising=False)

    class Tokenizer:
        eos_token_id = 1

        @staticmethod
        def convert_tokens_to_ids(token):
            assert token == "<|im_end|>"
            return 2

    params = FakeNnxState()
    engine = create_offline_engine(
        object(),
        Tokenizer(),
        seed=7,
        params=params,
        mesh="mesh",
    )

    assert isinstance(engine, FakeOfflineEngine)
    assert observed["decoupled_at_import"] == "TRUE"
    assert observed["kwargs"]["eos_ids"] == [1, 2]
    assert observed["kwargs"]["tokenizer"].eos_token_id == 1
    assert observed["kwargs"]["params"] == {
        "params": {"decoder": {"weight": "array"}}
    }
    assert params.to_pure_dict_calls == 1
    assert observed["kwargs"]["mesh"] == "mesh"


def test_offline_engine_param_refresh_converts_nnx_state_to_pure_dict():
    observed = {}

    class FakeNnxState:
        @staticmethod
        def to_pure_dict():
            return {"base": {"decoder": {"weight": "updated-array"}}}

    class Engine:
        @staticmethod
        def update_params(params):
            observed["params"] = params

    update_offline_engine_params(Engine(), FakeNnxState())

    assert observed["params"] == {
        "params": {"decoder": {"weight": "updated-array"}}
    }


def test_offline_engine_param_refresh_preserves_linen_params_tree():
    observed = {}
    linen_params = {"params": {"decoder": {"weight": "array"}}}

    class Engine:
        @staticmethod
        def update_params(params):
            observed["params"] = params

    update_offline_engine_params(Engine(), linen_params)

    assert observed["params"] is linen_params


def test_decoupled_prefill_compat_matches_offline_engine_default_path(
    monkeypatch,
):
    module_name = "maxtext.input_pipeline.packing.prefill_packing"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    _install_decoupled_prefill_compat()
    module = sys.modules[module_name]

    class Engine:
        @staticmethod
        def prefill(**kwargs):
            assert kwargs == {
                "params": "params",
                "padded_tokens": "tokens",
                "true_length": 3,
                "rng": "rng",
                "return_prompt_logp": True,
            }
            return {"prompt_logp": "prompt-logp"}, "first-token"

        @staticmethod
        def insert(prefill_result, decode_state, slot):
            assert prefill_result == {"prompt_logp": "prompt-logp"}
            assert slot == 5
            return {**decode_state, "inserted": True}

    processor = module.PrefillProcessor(Engine())
    first_token, decode_state = processor._process(
        "params",
        "tokens",
        5,
        3,
        {"state": True},
        "rng",
        return_prompt_logp=True,
    )

    assert module._RLCSD_DECOUPLED_COMPAT is True
    assert first_token == "first-token"
    assert decode_state == {
        "state": True,
        "inserted": True,
        "prompt_logp": "prompt-logp",
    }
    with pytest.raises(RuntimeError, match="Batch prefill requires Jetstream"):
        module.BatchedPrefillProcessor(Engine())


def test_decoupled_result_tokens_are_valid_jax_jit_outputs():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    class ResultTokens:
        def __init__(
            self,
            *,
            data,
            tokens_idx,
            valid_idx,
            length_idx,
            log_prob,
            samples_per_slot,
        ):
            self.data = data
            self.tokens_idx = tokens_idx
            self.valid_idx = valid_idx
            self.length_idx = length_idx
            self.log_prob = log_prob
            self.samples_per_slot = samples_per_slot

    assert _register_result_tokens_pytree(ResultTokens, jax.tree_util) is True
    assert _register_result_tokens_pytree(ResultTokens, jax.tree_util) is False

    @jax.jit
    def build_result(value):
        return ResultTokens(
            data=value + 1,
            tokens_idx=(0, 1),
            valid_idx=(1, 2),
            length_idx=(2, 3),
            log_prob=value - 1,
            samples_per_slot=1,
        )

    result = build_result(jnp.asarray([[2]], dtype=jnp.int32))
    assert result.data.tolist() == [[3]]
    assert result.log_prob.tolist() == [[1]]
    assert result.tokens_idx == (0, 1)
    assert result.valid_idx == (1, 2)
    assert result.length_idx == (2, 3)
    assert result.samples_per_slot == 1


def test_offline_engine_detokenization_accepts_size_one_token_arrays():
    class FutureNumpyArray:
        shape = (1,)
        size = 1

        @staticmethod
        def item():
            return 17

        @staticmethod
        def __int__():
            raise TypeError(
                "only 0-dimensional arrays can be converted to Python scalars"
            )

    class OfflineEngine:
        @staticmethod
        def background_detokenization():
            return int(FutureNumpyArray())

    with pytest.raises(TypeError, match="0-dimensional arrays"):
        OfflineEngine.background_detokenization()

    assert _install_offline_engine_size_one_int_compat(OfflineEngine) is True
    assert _install_offline_engine_size_one_int_compat(OfflineEngine) is False
    assert OfflineEngine.background_detokenization() == 17


def test_offline_engine_compat_targets_worker_without_gating_result_tokens(
    monkeypatch,
):
    import benchmarks.maxtext_tpu.rlcsd_rollout as rollout_module

    class FutureNumpyArray:
        shape = (1,)
        size = 1

        @staticmethod
        def item():
            return 23

        @staticmethod
        def __int__():
            raise TypeError(
                "only 0-dimensional arrays can be converted to Python scalars"
            )

    class InferenceWorker:
        @staticmethod
        def background_detokenization():
            return int(FutureNumpyArray())

    class OfflineEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    offline_engine_module = types.ModuleType(
        "maxtext.inference.offline_engine"
    )
    offline_engine_module.InferenceWorker = InferenceWorker
    offline_engine_module.OfflineEngine = OfflineEngine
    maxtext_package = types.ModuleType("maxtext")
    maxtext_package.__path__ = []
    inference_package = types.ModuleType("maxtext.inference")
    inference_package.__path__ = []
    fake_jax = types.ModuleType("jax")
    fake_jax.random = types.SimpleNamespace(
        PRNGKey=lambda seed: ("fake-prng-key", seed)
    )

    fake_modules = {
        "jax": fake_jax,
        "maxtext": maxtext_package,
        "maxtext.inference": inference_package,
        "maxtext.inference.offline_engine": offline_engine_module,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    registrations = []
    monkeypatch.setattr(
        rollout_module,
        "_register_decoupled_result_tokens_pytree",
        lambda: registrations.append(True) or True,
    )

    class Tokenizer:
        eos_token_id = 1

        @staticmethod
        def convert_tokens_to_ids(_token):
            return 2

    engine = create_offline_engine(
        types.SimpleNamespace(),
        Tokenizer(),
        seed=7,
    )

    assert registrations == [True]
    assert InferenceWorker.background_detokenization() == 23
    assert isinstance(engine, OfflineEngine)


def test_rollout_gate_requires_exactly_64_groups_of_8():
    contract = load_original_contract(REPO_ROOT)
    samples = [
        sample for prompt_index in range(64) for sample in _mixed_group(prompt_index)
    ]
    summary = validate_rollout_samples(samples, contract)
    assert summary["prompt_group_count"] == 64
    assert summary["group_size"] == 8
    assert summary["sample_count"] == 512


def test_teacher_alignment_is_self_excluded_and_packs_original_budgets():
    contract = load_original_contract(REPO_ROOT)
    samples = _mixed_group(0)
    advantages = compute_grpo_advantages(samples, contract)
    assignments, summary = build_teacher_assignments(
        samples,
        contract,
        seed=42,
    )
    assert len(assignments) == 8
    assert summary["valid_group_count"] == 1
    for assignment in assignments:
        sample_key = (
            assignment.sample.prompt_id,
            assignment.sample.generation_index,
        )
        used = [
            (
                item.prompt_id,
                item.generation_index,
            )
            for item in (
                assignment.correct_context,
                *assignment.wrong_contexts,
            )
        ]
        assert sample_key not in used
        assert 1 <= len(assignment.wrong_contexts) <= 4

    packed = pack_actor_minibatch(
        assignments,
        advantages,
        FakeTokenizer(),
        contract,
    )
    assert packed["policy_input_ids"].shape[0] == 8
    assert packed["packing"]["policy_width"] <= 18432
    assert packed["packing"]["teacher_correct_width"] <= 40960
    assert packed["packing"]["teacher_wrong_width"] <= 40960
    assert np.all(packed["response_mask"].sum(axis=1) == 2)
    assert np.all(packed["teacher_wrong_valid_mask"].sum(axis=1) >= 1)


def test_token_budget_scheduler_never_exceeds_40960_or_four_rows():
    segments = np.zeros((6, 22000), dtype=np.int8)
    segments[0, 2000:] = 1
    segments[1, 3000:] = 1
    segments[2, 18000:] = 1
    segments[3, 19000:] = 1
    segments[4, 1000:5000] = 1
    segments[5, 2000:6000] = 1
    chunks = plan_token_budget_chunks(
        segments,
        token_budget=40960,
        max_rows=4,
    )
    assert [row for chunk in chunks for row in chunk.rows] == list(range(6))
    assert all(chunk.padded_token_count <= 40960 for chunk in chunks)
    assert all(len(chunk.rows) <= 4 for chunk in chunks)


def test_teacher_aware_workload_scheduler_can_split_below_row_cap():
    contract = dataclasses.replace(
        load_original_contract(REPO_ROOT),
        actor_max_token_len_per_worker=10,
    )
    segments = np.ones((4, 6), dtype=np.int8)
    packed = {
        "policy_segment_ids": segments,
        "teacher_correct_segment_ids": segments,
        "teacher_wrong_segment_ids": segments[:, None, :],
        "teacher_wrong_valid_mask": np.ones((4, 1), dtype=bool),
    }
    chunks = plan_actor_update_chunks(packed, contract)
    assert sorted(row for chunk in chunks for row in chunk.rows) == [0, 1, 2, 3]
    assert len(chunks) == 4


def test_snapshot_teacher_window_survives_rollout_batch_boundary():
    assert plan_teacher_snapshot_windows(0, 32, 10) == [
        (0, 10, True),
        (10, 10, True),
        (20, 10, True),
        (30, 2, True),
    ]
    assert plan_teacher_snapshot_windows(32, 32, 10) == [
        (32, 8, False),
        (40, 10, True),
        (50, 10, True),
        (60, 4, True),
    ]


def _original_update_payload() -> dict:
    contract = load_original_contract(REPO_ROOT)
    return {
        "ok": True,
        "phase": "rlcsd_original_update",
        "is_rlcsd_end_to_end": False,
        "is_original_rlcsd_core_update": True,
        "contract": contract.as_dict(),
        "case": ORIGINAL_RLCSD_CASE.as_dict(),
        "weights": "converted_pretrained_checkpoint",
        "environment": {
            "backend": "tpu",
            "device_count": 8,
            "v5e_kind_match": True,
        },
        "dataset": {"resolved_revision": contract.dataset_revision},
        "rollout": {
            "prompt_group_count": 64,
            "group_size": 8,
            "sample_count": 512,
        },
        "alignment": {
            "ok": True,
            "assignment_count": 32,
            "summary": {"group_count": 64},
        },
        "diagnostic_update_cap": None,
        "update_count": 2,
        "expected_full_update_count": 2,
        "teacher_sync_event_count": 1,
        "nonzero_learning_rate_update_count": 1,
        "mean_timed_response_tokens_per_s": 12.0,
        "timed_updates": [
            {
                "loss": 1.0,
                "grad_norm": 2.0,
                "learning_rate": 2e-8,
                "policy_execution_s": 3.0,
                "optimizer_update_s": 0.2,
                "optimizer_update_applied": True,
                "response_token_count": 100,
                "response_tokens_per_s": 31.25,
            }
        ],
        "parameter_sharding": {"tensor_sharded_leaf_count": 1},
        "measurement_boundary": {
            "compile_excluded": True,
            "warmup_excluded": True,
            "includes_decode": True,
            "includes_verifier": True,
            "includes_teacher_correct_wrong_passes": True,
            "includes_rlcsd_policy_loss": True,
            "includes_backward": True,
            "includes_optimizer_update": True,
            "full_30_epoch_training": False,
        },
    }


def test_original_update_audit_accepts_complete_core_update():
    report = audit_payload(_original_update_payload())
    assert report["verdict"] is True


def test_original_update_audit_rejects_seq512_relabeling_and_caps():
    payload = _original_update_payload()
    payload["case"] = dataclasses.replace(
        ORIGINAL_RLCSD_CASE,
        sequence_length=512,
    ).as_dict()
    payload["diagnostic_update_cap"] = 2
    report = audit_payload(payload)
    assert report["verdict"] is False
    assert "original_profile" in report["failed_checks"]
    assert "not_diagnostic_cap" in report["failed_checks"]


def test_end_to_end_audit_rejects_core_update_without_30_epochs():
    payload = _original_update_payload()
    payload.update(
        {
            "phase": "rlcsd_end_to_end",
            "is_rlcsd_end_to_end": True,
            "measurement_boundary": {
                **payload["measurement_boundary"],
                "full_30_epoch_training": False,
            },
        }
    )
    report = audit_payload(payload)
    assert report["verdict"] is False
    assert "thirty_epochs_completed" in report["failed_checks"]
    assert "full_training_boundary" in report["failed_checks"]


def test_original_training_progress_matches_drop_last_30_epoch_contract():
    contract = load_original_contract(REPO_ROOT)
    progress = TrainingProgress.initial(contract)
    assert original_steps_per_epoch(contract) == 937
    assert original_total_rollout_steps(contract) == 28_110
    progress.pretrain_validation_complete = True
    progress.phase = "training"
    for rollout_step in range(original_steps_per_epoch(contract)):
        advance_training_progress(
            progress,
            contract,
            optimizer_step=(rollout_step + 1) * 32,
        )
    assert progress.epoch_index == 1
    assert progress.batch_index_in_epoch == 0
    assert progress.rollout_step == 937
    assert progress.prompts_consumed == 59_968


def test_resumable_training_shard_audit_preserves_full_target_without_claiming_it():
    contract = load_original_contract(REPO_ROOT)
    progress = TrainingProgress.initial(contract)
    payload = {
        "ok": True,
        "phase": "rlcsd_training_shard",
        "is_rlcsd_end_to_end": False,
        "is_resumable_original_contract_shard": True,
        "contract": contract.as_dict(),
        "case": ORIGINAL_RLCSD_CASE.as_dict(),
        "weights": "converted_pretrained_checkpoint",
        "environment": {
            "backend": "tpu",
            "device_count": 8,
            "v5e_kind_match": True,
        },
        "dataset": {"resolved_revision": contract.dataset_revision},
        "progress": progress.as_dict(),
        "training": {
            "max_train_samples": 60_000,
            "total_rollout_steps": 28_110,
        },
        "validation": {
            "val_before_train": True,
            "val_n": 12,
        },
        "checkpoints": {
            "save_steps": 50,
            "saved_checkpoint_count": 1,
            "session_checkpoints": [{"step": 0}],
        },
        "measurement_boundary": {"full_30_epoch_training": False},
    }
    report = audit_payload(payload)
    assert report["verdict"] is True
