"""Executable contract for the original Qwen3-1.7B RLCSD experiment.

The existing MaxText matrix is an infrastructure canary.  This module instead
loads the checked-in RLCSD YAML and derives the effective values produced by
``scripts/_run_verl.sh`` on the reference eight-worker topology.  Keeping the
derivation executable prevents a short-sequence canary from being relabelled as
an original-config reproduction.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

REFERENCE_CONFIG_PATH = "configs/math_deepmath/qwen3_1_7b_rlcsd.yaml"
REFERENCE_MODEL_ID = "Qwen/Qwen3-1.7B"
REFERENCE_DATASET_REPO = "Leyiii/RLCSD"
REFERENCE_DATASET_REVISION = "33d7de919af5b03257ff92c30303fddf9afdda4a"
REFERENCE_DATASET_FILE = "deepmath_filtered_level5_7/train.parquet"
REFERENCE_WORKER_COUNT = 8


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Read the repository's flat experiment YAML without a YAML dependency."""

    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected key: value")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key or not raw_value.strip():
            raise ValueError(
                f"{path}:{line_number}: nested/empty YAML is not supported"
            )
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        values[key] = _parse_scalar(raw_value)
    return values


@dataclasses.dataclass(frozen=True, slots=True)
class OriginalRlcsdContract:
    """Reference RLCSD values plus launcher-derived effective dimensions."""

    config_path: str
    config_sha256: str
    model_id: str
    dataset_repo: str
    dataset_revision: str
    dataset_file: str
    method: str
    use_lora: bool
    learning_rate: float
    num_epochs: int
    per_device_batch_size: int
    reference_data_parallel_workers: int
    rollout_prompt_batch_size: int
    group_size: int
    rollout_sample_batch_size: int
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_worker: int
    max_grad_norm: float
    warmup_steps: int
    weight_decay: float
    max_train_samples: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    max_prompt_length: int
    max_completion_length: int
    train_max_sequence_length: int
    val_max_completion_length: int
    val_max_sequence_length: int
    max_model_length: int
    actor_max_token_len_per_worker: int
    temperature: float
    top_p: float
    top_k: int
    student_enable_thinking: bool
    val_enable_thinking: bool
    val_do_sample: bool
    val_n: int
    val_temperature: float
    val_top_p: float
    val_top_k: int
    val_batch_size: int
    teacher_enable_thinking: bool
    vllm_gpu_memory_utilization: float
    vllm_tensor_parallel_size: int
    privileged_text_mode: str
    rollout_is: str
    rollout_is_threshold: float
    kl_loss_coef: float
    epsilon: float
    rlcsd_tau: float
    rlcsd_beta: float
    rlcsd_lam: float
    rlcsd_delta: float
    rlcsd_eta: float
    rlcsd_residual_clip_low: float
    rlcsd_residual_clip_high: float
    rlcsd_k_max: int
    teacher_sync_interval: int
    teacher_mode: str
    val_datasets: tuple[str, ...]
    output_dir: str
    data_dir: str
    use_tensorboard: bool
    project_name: str
    experiment_name: str
    val_before_train: bool

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        *,
        reference_worker_count: int = REFERENCE_WORKER_COUNT,
    ) -> OriginalRlcsdContract:
        config_path = repo_root / REFERENCE_CONFIG_PATH
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        raw = config_path.read_bytes()
        values = load_flat_yaml(config_path)
        required = {
            "method",
            "use_lora",
            "learning_rate",
            "num_epochs",
            "per_device_batch_size",
            "group_size",
            "ppo_mini_batch_size",
            "max_grad_norm",
            "warmup_steps",
            "weight_decay",
            "max_train_samples",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "max_prompt_length",
            "max_completion_length",
            "val_max_completion_length",
            "actor_max_token_len_per_gpu",
            "temperature",
            "top_p",
            "top_k_sampling",
            "student_enable_thinking",
            "val_enable_thinking",
            "val_do_sample",
            "val_n",
            "val_temperature",
            "val_top_p",
            "val_top_k",
            "val_batch_size",
            "teacher_enable_thinking",
            "vllm_gpu_memory_utilization",
            "vllm_tensor_parallel_size",
            "privileged_text_mode",
            "rollout_is",
            "rollout_is_threshold",
            "kl_loss_coef",
            "epsilon",
            "rlcsd_tau",
            "rlcsd_beta",
            "rlcsd_lam",
            "rlcsd_delta",
            "rlcsd_eta",
            "rlcsd_residual_clip_low",
            "rlcsd_residual_clip_high",
            "rlcsd_k_max",
            "teacher_sync_interval",
            "teacher_mode",
            "train_dataset",
            "val_dataset",
            "output_dir",
            "data_dir",
            "use_tensorboard",
            "project_name",
            "experiment_name",
            "val_before_train",
        }
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"{config_path} is missing required keys: {missing}")

        prompt_batch = int(values["per_device_batch_size"]) * int(
            reference_worker_count
        )
        rollout_samples = prompt_batch * int(values["group_size"])
        train_max = int(values["max_prompt_length"]) + int(
            values["max_completion_length"]
        )
        val_max = int(values["max_prompt_length"]) + int(
            values["val_max_completion_length"]
        )
        contract = cls(
            config_path=REFERENCE_CONFIG_PATH,
            config_sha256=hashlib.sha256(raw).hexdigest(),
            model_id=REFERENCE_MODEL_ID,
            dataset_repo=REFERENCE_DATASET_REPO,
            dataset_revision=REFERENCE_DATASET_REVISION,
            dataset_file=(f"{values['train_dataset']}/train.parquet"),
            method=str(values["method"]),
            use_lora=bool(values["use_lora"]),
            learning_rate=float(values["learning_rate"]),
            num_epochs=int(values["num_epochs"]),
            per_device_batch_size=int(values["per_device_batch_size"]),
            reference_data_parallel_workers=int(reference_worker_count),
            rollout_prompt_batch_size=prompt_batch,
            group_size=int(values["group_size"]),
            rollout_sample_batch_size=rollout_samples,
            ppo_mini_batch_size=int(values["ppo_mini_batch_size"]),
            # ``scripts/_run_verl.sh`` fixes this value rather than reading it
            # from the YAML.
            ppo_micro_batch_size_per_worker=4,
            max_grad_norm=float(values["max_grad_norm"]),
            warmup_steps=int(values["warmup_steps"]),
            weight_decay=float(values["weight_decay"]),
            max_train_samples=int(values["max_train_samples"]),
            save_steps=int(values["save_steps"]),
            eval_steps=int(values["eval_steps"]),
            logging_steps=int(values["logging_steps"]),
            max_prompt_length=int(values["max_prompt_length"]),
            max_completion_length=int(values["max_completion_length"]),
            train_max_sequence_length=train_max,
            val_max_completion_length=int(values["val_max_completion_length"]),
            val_max_sequence_length=val_max,
            max_model_length=max(train_max, val_max),
            actor_max_token_len_per_worker=int(values["actor_max_token_len_per_gpu"]),
            temperature=float(values["temperature"]),
            top_p=float(values["top_p"]),
            top_k=int(values["top_k_sampling"]),
            student_enable_thinking=bool(values["student_enable_thinking"]),
            val_enable_thinking=bool(values["val_enable_thinking"]),
            val_do_sample=bool(values["val_do_sample"]),
            val_n=int(values["val_n"]),
            val_temperature=float(values["val_temperature"]),
            val_top_p=float(values["val_top_p"]),
            val_top_k=int(values["val_top_k"]),
            val_batch_size=int(values["val_batch_size"]),
            teacher_enable_thinking=bool(values["teacher_enable_thinking"]),
            vllm_gpu_memory_utilization=float(values["vllm_gpu_memory_utilization"]),
            vllm_tensor_parallel_size=int(values["vllm_tensor_parallel_size"]),
            privileged_text_mode=str(values["privileged_text_mode"]),
            rollout_is=str(values["rollout_is"]),
            rollout_is_threshold=float(values["rollout_is_threshold"]),
            kl_loss_coef=float(values["kl_loss_coef"]),
            epsilon=float(values["epsilon"]),
            rlcsd_tau=float(values["rlcsd_tau"]),
            rlcsd_beta=float(values["rlcsd_beta"]),
            rlcsd_lam=float(values["rlcsd_lam"]),
            rlcsd_delta=float(values["rlcsd_delta"]),
            rlcsd_eta=float(values["rlcsd_eta"]),
            rlcsd_residual_clip_low=float(values["rlcsd_residual_clip_low"]),
            rlcsd_residual_clip_high=float(values["rlcsd_residual_clip_high"]),
            rlcsd_k_max=int(values["rlcsd_k_max"]),
            teacher_sync_interval=int(values["teacher_sync_interval"]),
            teacher_mode=str(values["teacher_mode"]),
            val_datasets=tuple(
                part.strip()
                for part in str(values["val_dataset"]).split(",")
                if part.strip()
            ),
            output_dir=str(values["output_dir"]),
            data_dir=str(values["data_dir"]),
            use_tensorboard=bool(values["use_tensorboard"]),
            project_name=str(values["project_name"]),
            experiment_name=str(values["experiment_name"]),
            val_before_train=bool(values["val_before_train"]),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        expected = {
            "method": (self.method, "rlcsd"),
            "model_id": (self.model_id, REFERENCE_MODEL_ID),
            "dataset_file": (self.dataset_file, REFERENCE_DATASET_FILE),
            "worker_count": (self.reference_data_parallel_workers, 8),
            "per_device_batch_size": (self.per_device_batch_size, 8),
            "rollout_prompt_batch_size": (self.rollout_prompt_batch_size, 64),
            "group_size": (self.group_size, 8),
            "rollout_sample_batch_size": (
                self.rollout_sample_batch_size,
                512,
            ),
            "ppo_mini_batch_size": (self.ppo_mini_batch_size, 16),
            "ppo_micro_batch_size_per_worker": (
                self.ppo_micro_batch_size_per_worker,
                4,
            ),
            "max_prompt_length": (self.max_prompt_length, 2048),
            "max_completion_length": (self.max_completion_length, 16384),
            "train_max_sequence_length": (
                self.train_max_sequence_length,
                18432,
            ),
            "val_max_completion_length": (
                self.val_max_completion_length,
                38912,
            ),
            "max_model_length": (self.max_model_length, 40960),
            "actor_max_token_len_per_worker": (
                self.actor_max_token_len_per_worker,
                40960,
            ),
            "num_epochs": (self.num_epochs, 30),
            "learning_rate": (self.learning_rate, 1e-6),
            "max_grad_norm": (self.max_grad_norm, 1.0),
            "warmup_steps": (self.warmup_steps, 50),
            "weight_decay": (self.weight_decay, 0.01),
            "max_train_samples": (self.max_train_samples, 60000),
            "save_steps": (self.save_steps, 50),
            "eval_steps": (self.eval_steps, 50),
            "logging_steps": (self.logging_steps, 5),
            "temperature": (self.temperature, 1.0),
            "top_p": (self.top_p, 0.95),
            "top_k": (self.top_k, 20),
            "val_datasets": (
                self.val_datasets,
                ("amc23", "aime24", "aime25"),
            ),
            "val_n": (self.val_n, 12),
            "val_temperature": (self.val_temperature, 0.6),
            "val_top_p": (self.val_top_p, 0.95),
            "val_top_k": (self.val_top_k, 20),
            "val_batch_size": (self.val_batch_size, 16),
            "privileged_text_mode": (
                self.privileged_text_mode,
                "solution_answer",
            ),
            "rollout_is": (self.rollout_is, "token"),
            "rollout_is_threshold": (self.rollout_is_threshold, 2.0),
            "kl_loss_coef": (self.kl_loss_coef, 0.0),
            "epsilon": (self.epsilon, 0.2),
            "rlcsd_tau": (self.rlcsd_tau, 0.02),
            "rlcsd_beta": (self.rlcsd_beta, 1.0),
            "rlcsd_lam": (self.rlcsd_lam, 0.5),
            "rlcsd_delta": (self.rlcsd_delta, 0.02),
            "rlcsd_eta": (self.rlcsd_eta, 1.0),
            "rlcsd_residual_clip_low": (
                self.rlcsd_residual_clip_low,
                -2.0,
            ),
            "rlcsd_residual_clip_high": (
                self.rlcsd_residual_clip_high,
                2.0,
            ),
            "rlcsd_k_max": (self.rlcsd_k_max, 4),
            "teacher_sync_interval": (self.teacher_sync_interval, 10),
            "teacher_mode": (self.teacher_mode, "snapshot"),
            "vllm_tensor_parallel_size": (
                self.vllm_tensor_parallel_size,
                1,
            ),
            "vllm_gpu_memory_utilization": (
                self.vllm_gpu_memory_utilization,
                0.6,
            ),
            "student_enable_thinking": (
                self.student_enable_thinking,
                True,
            ),
            "teacher_enable_thinking": (
                self.teacher_enable_thinking,
                True,
            ),
            "val_enable_thinking": (self.val_enable_thinking, True),
            "val_do_sample": (self.val_do_sample, True),
            "use_tensorboard": (self.use_tensorboard, True),
            "val_before_train": (self.val_before_train, True),
        }
        mismatches = {
            key: {"observed": observed, "expected": wanted}
            for key, (observed, wanted) in expected.items()
            if observed != wanted
        }
        if self.use_lora:
            mismatches["use_lora"] = {"observed": True, "expected": False}
        if self.ppo_mini_batch_size <= 0 or (
            self.rollout_sample_batch_size % self.ppo_mini_batch_size
        ):
            mismatches["ppo_mini_batch_divisibility"] = {
                "observed": self.ppo_mini_batch_size,
                "expected": "positive divisor of rollout_sample_batch_size",
            }
        if self.actor_max_token_len_per_worker < self.train_max_sequence_length:
            mismatches["actor_token_budget"] = {
                "observed": self.actor_max_token_len_per_worker,
                "expected": f">={self.train_max_sequence_length}",
            }
        if mismatches:
            raise ValueError(
                "Original RLCSD config contract drifted: "
                + json.dumps(mismatches, sort_keys=True)
            )

    @property
    def required_runtime_gates(self) -> tuple[str, ...]:
        return (
            "converted_pretrained_checkpoint",
            "original_config_exact",
            "kv_cache_rollout_decode",
            "rollout_prompt_batch_size_64",
            "rollout_group_size_8",
            "rollout_sample_batch_size_512",
            "math_verifier",
            "teacher_correct_context",
            "teacher_wrong_k_marginal_contexts",
            "snapshot_teacher_sync_interval_10",
            "teacher_aware_dynamic_batching",
            "token_rollout_is_threshold_2",
            "rlcsd_policy_loss",
            "backward",
            "adamw_update",
            "finite_loss",
        )

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "contract_id": "qwen3_1_7b_rlcsd_original",
                "required_runtime_gates": list(self.required_runtime_gates),
                "is_canary": False,
            }
        )
        payload["val_datasets"] = list(self.val_datasets)
        return payload


def load_original_contract(repo_root: Path | None = None) -> OriginalRlcsdContract:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return OriginalRlcsdContract.from_repo(repo_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = (
        json.dumps(
            load_original_contract(args.repo_root).as_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
