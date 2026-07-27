"""RLCSD rollout alignment and model-input packing.

The functions here mirror the reference ``src.self_distill_main`` semantics:
mixed-outcome rollout groups only, self-excluded sibling contexts, one positive
teacher context and a uniform marginal over up to K negative contexts.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.maxtext_tpu.rlcsd_contract import OriginalRlcsdContract
from benchmarks.maxtext_tpu.rlcsd_rollout import RolloutSample
from src.opsd_format import build_teacher_messages


def sample_key(sample: RolloutSample) -> str:
    return f"{sample.prompt_id}:{sample.generation_index}"


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherAssignment:
    sample: RolloutSample
    correct_context: RolloutSample
    wrong_contexts: tuple[RolloutSample, ...]

    def validate(self, contract: OriginalRlcsdContract) -> None:
        if self.correct_context.reward <= 0:
            raise ValueError("correct teacher context must have positive reward")
        if not self.wrong_contexts:
            raise ValueError("at least one wrong teacher context is required")
        if len(self.wrong_contexts) > contract.rlcsd_k_max:
            raise ValueError(f"wrong teacher K exceeds {contract.rlcsd_k_max}")
        if any(item.reward > 0 or item.answer is None for item in self.wrong_contexts):
            raise ValueError("wrong teacher contexts must be boxed-answer negatives")
        used = [sample_key(self.correct_context)] + [
            sample_key(item) for item in self.wrong_contexts
        ]
        if sample_key(self.sample) in used:
            raise ValueError("teacher sibling selection included the sample itself")
        if len(set(used[1:])) != len(used[1:]):
            raise ValueError("wrong teacher contexts contain duplicates")
        if any(item.prompt_id != self.sample.prompt_id for item in used_samples(self)):
            raise ValueError("teacher contexts crossed prompt groups")


def used_samples(assignment: TeacherAssignment) -> tuple[RolloutSample, ...]:
    return (assignment.correct_context, *assignment.wrong_contexts)


def compute_grpo_advantages(
    samples: Sequence[RolloutSample],
    contract: OriginalRlcsdContract,
) -> dict[str, float]:
    """Match VERL's outcome-GRPO mean/sample-std normalization."""

    groups: dict[str, list[RolloutSample]] = {}
    for sample in samples:
        groups.setdefault(sample.prompt_id, []).append(sample)
    advantages: dict[str, float] = {}
    for prompt_id, group in groups.items():
        if len(group) != contract.group_size:
            raise ValueError(
                f"{prompt_id}: expected group size {contract.group_size}, "
                f"received {len(group)}"
            )
        rewards = [float(item.reward) for item in group]
        mean = statistics.fmean(rewards)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
        for item in group:
            value = (float(item.reward) - mean) / (std + 1e-6)
            if not math.isfinite(value):
                raise ValueError(f"{sample_key(item)}: non-finite advantage")
            advantages[sample_key(item)] = value
    return advantages


def build_teacher_assignments(
    samples: Sequence[RolloutSample],
    contract: OriginalRlcsdContract,
    *,
    seed: int,
) -> tuple[list[TeacherAssignment], dict[str, Any]]:
    """Select correct/wrong non-self siblings using reference RLCSD rules."""

    groups: dict[str, list[RolloutSample]] = {}
    for sample in samples:
        groups.setdefault(sample.prompt_id, []).append(sample)
    rng = random.Random(seed)
    assignments: list[TeacherAssignment] = []
    all_same_outcome = 0
    invalid_negative_answer = 0
    self_excluded_skip = 0
    valid_group_count = 0

    for prompt_id in sorted(groups):
        group = sorted(groups[prompt_id], key=lambda item: item.generation_index)
        if len(group) != contract.group_size:
            raise ValueError(f"{prompt_id}: expected {contract.group_size} generations")
        positives = [item for item in group if item.reward > 0]
        negatives = [item for item in group if item.reward <= 0]
        valid_negatives = [item for item in negatives if item.answer is not None]
        invalid_negative_answer += len(negatives) - len(valid_negatives)
        if not positives or not valid_negatives:
            all_same_outcome += 1
            continue

        group_assignment_count = 0
        for sample in group:
            if sample.reward > 0:
                correct_pool = [
                    item for item in positives if sample_key(item) != sample_key(sample)
                ]
                wrong_pool = list(valid_negatives)
            else:
                correct_pool = list(positives)
                wrong_pool = [
                    item
                    for item in valid_negatives
                    if sample_key(item) != sample_key(sample)
                ]
            if not correct_pool or not wrong_pool:
                self_excluded_skip += 1
                continue

            correct_context = rng.choice(correct_pool)
            chosen_wrong = rng.choice(wrong_pool)
            remaining = [
                item
                for item in wrong_pool
                if sample_key(item) != sample_key(chosen_wrong)
            ]
            rng.shuffle(remaining)
            wrong_contexts = (
                chosen_wrong,
                *remaining[: max(contract.rlcsd_k_max - 1, 0)],
            )
            assignment = TeacherAssignment(
                sample=sample,
                correct_context=correct_context,
                wrong_contexts=tuple(wrong_contexts),
            )
            assignment.validate(contract)
            assignments.append(assignment)
            group_assignment_count += 1
        if group_assignment_count:
            valid_group_count += 1

    return assignments, {
        "group_count": len(groups),
        "valid_group_count": valid_group_count,
        "all_same_or_unusable_group_count": all_same_outcome,
        "invalid_negative_answer_count": invalid_negative_answer,
        "self_excluded_sampling_skip_count": self_excluded_skip,
        "valid_sample_count": len(assignments),
        "skipped_sample_count": len(samples) - len(assignments),
        "effective_k_mean": (
            statistics.fmean(len(item.wrong_contexts) for item in assignments)
            if assignments
            else 0.0
        ),
    }


def _chat_template_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> list[int]:
    values = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]


def _teacher_prompt_tokens(
    tokenizer: Any,
    *,
    target: RolloutSample,
    context: RolloutSample,
    contract: OriginalRlcsdContract,
) -> list[int]:
    answer = context.answer
    if answer is None and context.reward > 0:
        answer = context.ground_truth
    if answer is None:
        raise ValueError(f"{sample_key(context)}: teacher context has no boxed answer")
    _, messages, effective_mode = build_teacher_messages(
        problem=target.problem,
        answer=answer,
        solution=context.response_text,
        mode=contract.privileged_text_mode,
    )
    if effective_mode != contract.privileged_text_mode:
        raise ValueError(
            f"Teacher prompt mode drifted: {effective_mode} != "
            f"{contract.privileged_text_mode}"
        )
    return _chat_template_tokens(
        tokenizer,
        messages,
        enable_thinking=contract.teacher_enable_thinking,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PackedSequenceBatch:
    input_ids: Any
    segment_ids: Any
    position_ids: Any
    response_target_positions: Any


def _pack_left_padded_sequences(
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    *,
    pad_id: int,
    total_token_budget: int,
    response_width: int,
) -> PackedSequenceBatch:
    import numpy as np

    if len(prompt_ids) != len(response_ids) or not prompt_ids:
        raise ValueError("prompt/response sequence lists must be non-empty and aligned")
    trimmed_prompts: list[list[int]] = []
    for prompt, response in zip(prompt_ids, response_ids, strict=True):
        prompt = [int(value) for value in prompt]
        response = [int(value) for value in response]
        if not response:
            raise ValueError("cannot pack an empty response")
        allowed_prompt = total_token_budget - len(response)
        if allowed_prompt < 1:
            raise ValueError(
                f"response length {len(response)} leaves no prefix token "
                f"inside token budget {total_token_budget}"
            )
        if len(prompt) > allowed_prompt:
            prompt = prompt[-allowed_prompt:]
        trimmed_prompts.append(prompt)

    prompt_width = max(len(values) for values in trimmed_prompts)
    total_width = prompt_width + response_width
    if total_width > total_token_budget:
        raise ValueError(
            f"packed width {total_width} exceeds budget {total_token_budget}"
        )
    batch_size = len(trimmed_prompts)
    input_ids = np.full((batch_size, total_width), pad_id, dtype=np.int32)
    segment_ids = np.zeros_like(input_ids)
    position_ids = np.zeros_like(input_ids)
    target_positions = np.zeros(
        (batch_size, response_width),
        dtype=np.int32,
    )
    for row, (prompt, response) in enumerate(
        zip(trimmed_prompts, response_ids, strict=True)
    ):
        prompt_offset = prompt_width - len(prompt)
        input_ids[row, prompt_offset:prompt_width] = prompt
        response_count = len(response)
        input_ids[row, prompt_width : prompt_width + response_count] = response
        segment_ids[
            row,
            prompt_offset : prompt_width + response_count,
        ] = 1
        position_ids[
            row,
            prompt_offset : prompt_width + response_count,
        ] = np.arange(len(prompt) + response_count, dtype=np.int32)
        target_positions[row, :response_count] = (
            prompt_width - 1 + np.arange(response_count, dtype=np.int32)
        )
    return PackedSequenceBatch(
        input_ids=input_ids,
        segment_ids=segment_ids,
        position_ids=position_ids,
        response_target_positions=target_positions,
    )


def pack_actor_minibatch(
    assignments: Sequence[TeacherAssignment],
    advantages: Mapping[str, float],
    tokenizer: Any,
    contract: OriginalRlcsdContract,
) -> dict[str, Any]:
    """Create policy/correct/wrong tensors for one PPO mini-batch."""

    import numpy as np

    if not assignments:
        raise ValueError("actor mini-batch is empty")
    if len(assignments) > contract.ppo_mini_batch_size:
        raise ValueError(
            f"actor mini-batch has {len(assignments)} samples, "
            f"limit={contract.ppo_mini_batch_size}"
        )
    for assignment in assignments:
        assignment.validate(contract)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id must be configured.")

    response_ids = [
        list(assignment.sample.response_token_ids) for assignment in assignments
    ]
    response_width = max(len(values) for values in response_ids)
    if response_width > contract.max_completion_length:
        raise ValueError("response width exceeds original completion contract")

    policy = _pack_left_padded_sequences(
        [list(assignment.sample.prompt_token_ids) for assignment in assignments],
        response_ids,
        pad_id=int(pad_id),
        total_token_budget=contract.train_max_sequence_length,
        response_width=response_width,
    )
    correct = _pack_left_padded_sequences(
        [
            _teacher_prompt_tokens(
                tokenizer,
                target=assignment.sample,
                context=assignment.correct_context,
                contract=contract,
            )
            for assignment in assignments
        ],
        response_ids,
        pad_id=int(pad_id),
        total_token_budget=contract.actor_max_token_len_per_worker,
        response_width=response_width,
    )

    k_width = max(len(assignment.wrong_contexts) for assignment in assignments)
    wrong_prompt_ids: list[list[int]] = []
    wrong_response_ids: list[list[int]] = []
    wrong_valid = np.zeros((len(assignments), k_width), dtype=np.bool_)
    for row, assignment in enumerate(assignments):
        contexts = list(assignment.wrong_contexts)
        if not contexts:
            raise ValueError("RLCSD assignment has no wrong contexts")
        for column in range(k_width):
            is_valid = column < len(contexts)
            context = contexts[column] if is_valid else contexts[0]
            wrong_prompt_ids.append(
                _teacher_prompt_tokens(
                    tokenizer,
                    target=assignment.sample,
                    context=context,
                    contract=contract,
                )
            )
            wrong_response_ids.append(response_ids[row])
            wrong_valid[row, column] = is_valid
    wrong_flat = _pack_left_padded_sequences(
        wrong_prompt_ids,
        wrong_response_ids,
        pad_id=int(pad_id),
        total_token_budget=contract.actor_max_token_len_per_worker,
        response_width=response_width,
    )

    batch_size = len(assignments)
    response_mask = np.zeros((batch_size, response_width), dtype=np.float32)
    old_log_probs = np.zeros_like(response_mask)
    advantage_tokens = np.zeros_like(response_mask)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    for row, assignment in enumerate(assignments):
        sample = assignment.sample
        count = len(sample.response_token_ids)
        response_mask[row, :count] = 1.0
        old_log_probs[row, :count] = sample.old_log_probs
        advantage_tokens[row, :count] = float(advantages[sample_key(sample)])
        rewards[row] = float(sample.reward)

    def reshape_wrong(value: Any) -> Any:
        return value.reshape((batch_size, k_width, *value.shape[1:]))

    return {
        "schema_version": 1,
        "sample_keys": [sample_key(assignment.sample) for assignment in assignments],
        "policy_input_ids": policy.input_ids,
        "policy_segment_ids": policy.segment_ids,
        "policy_position_ids": policy.position_ids,
        "policy_response_target_positions": (policy.response_target_positions),
        "teacher_correct_input_ids": correct.input_ids,
        "teacher_correct_segment_ids": correct.segment_ids,
        "teacher_correct_position_ids": correct.position_ids,
        "teacher_correct_response_target_positions": (
            correct.response_target_positions
        ),
        "teacher_wrong_input_ids": reshape_wrong(wrong_flat.input_ids),
        "teacher_wrong_segment_ids": reshape_wrong(wrong_flat.segment_ids),
        "teacher_wrong_position_ids": reshape_wrong(wrong_flat.position_ids),
        "teacher_wrong_response_target_positions": reshape_wrong(
            wrong_flat.response_target_positions
        ),
        "teacher_wrong_valid_mask": wrong_valid,
        "old_log_probs": old_log_probs,
        "advantages": advantage_tokens,
        "response_mask": response_mask,
        "rewards": rewards,
        "packing": {
            "policy_width": int(policy.input_ids.shape[1]),
            "teacher_correct_width": int(correct.input_ids.shape[1]),
            "teacher_wrong_width": int(wrong_flat.input_ids.shape[1]),
            "response_width": response_width,
            "effective_k": wrong_valid.sum(axis=1).astype(int).tolist(),
        },
    }


def iter_actor_minibatches(
    assignments: Sequence[TeacherAssignment],
    contract: OriginalRlcsdContract,
) -> Sequence[Sequence[TeacherAssignment]]:
    return [
        assignments[start : start + contract.ppo_mini_batch_size]
        for start in range(0, len(assignments), contract.ppo_mini_batch_size)
    ]


def alignment_manifest(
    assignments: Sequence[TeacherAssignment],
    summary: Mapping[str, Any],
    contract: OriginalRlcsdContract,
) -> dict[str, Any]:
    for assignment in assignments:
        assignment.validate(contract)
    return {
        "schema_version": 1,
        "ok": bool(assignments),
        "phase": "rlcsd_teacher_alignment",
        "contract_id": contract.as_dict()["contract_id"],
        "summary": dict(summary),
        "assignment_count": len(assignments),
        "assignments": [
            {
                "sample": sample_key(assignment.sample),
                "correct_context": sample_key(assignment.correct_context),
                "wrong_contexts": [
                    sample_key(item) for item in assignment.wrong_contexts
                ],
            }
            for assignment in assignments
        ],
    }


def assert_json_serializable_manifest(payload: Mapping[str, Any]) -> None:
    json.dumps(payload, sort_keys=True, allow_nan=False)
