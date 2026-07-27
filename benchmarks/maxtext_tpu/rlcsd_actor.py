"""Original-contract RLCSD teacher passes and actor updates on MaxText.

This module consumes one complete 64-prompt x 8-generation rollout batch.  It
uses the full eight-device TP mesh sequentially:

1. freeze the current actor logically and precompute teacher log-probabilities
   for the next ``teacher_sync_interval`` PPO updates;
2. run token-budgeted policy micro-batches and accumulate their gradients;
3. apply one clipped AdamW update per PPO mini-batch.

Precomputing each teacher window before mutating the actor implements snapshot
teacher semantics without keeping a second 1.7B model resident in TPU memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.maxtext_tpu.matrix import ORIGINAL_RLCSD_CASE
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
from benchmarks.maxtext_tpu.rlcsd_jax import rlcsd_policy_loss
from benchmarks.maxtext_tpu.rlcsd_rollout import (
    read_rollout_jsonl,
    validate_rollout_samples,
)
from benchmarks.maxtext_tpu.runtime import (
    load_model,
    memory_snapshot,
    parameter_sharding_report,
    read_checkpoint_manifest,
    require_tpu_v5e8,
    write_json,
)


@dataclasses.dataclass(frozen=True, slots=True)
class TokenBudgetChunk:
    rows: tuple[int, ...]
    start_column: int
    end_column: int
    padded_token_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherLogProbBatch:
    correct: Any
    wrong: Any
    compile_warmup_s: tuple[float, ...]
    execution_s: tuple[float, ...]


@dataclasses.dataclass(slots=True)
class OriginalRlcsdActorState:
    """State that must survive rollout-batch and Kaggle-session boundaries."""

    optimizer: Any
    teacher_params: Any
    global_update_step: int


def original_rlcsd_schedule(contract: OriginalRlcsdContract) -> Any:
    import optax

    return optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=contract.learning_rate,
        warmup_steps=contract.warmup_steps,
    )


def create_original_rlcsd_optimizer(
    model: Any,
    contract: OriginalRlcsdContract,
) -> tuple[Any, Any]:
    """Create the reference AdamW and constant-after-warmup schedule once."""

    import optax
    from flax import nnx

    schedule = original_rlcsd_schedule(contract)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(contract.max_grad_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=contract.weight_decay,
            ),
        ),
        wrt=nnx.Param,
    )
    return optimizer, schedule


def _snapshot_params(model: Any) -> Any:
    """Materialize a distinct, sharded snapshot of all trainable parameters."""

    import jax
    from flax import nnx

    _, params, _ = nnx.split(model, nnx.Param, ...)
    snapshot = jax.tree.map(lambda value: value.copy(), params)
    return jax.block_until_ready(snapshot)


def plan_teacher_snapshot_windows(
    global_update_start: int,
    update_count: int,
    sync_interval: int,
) -> list[tuple[int, int, bool]]:
    """Return ``(global_start, count, refresh_before)`` snapshot windows."""

    if global_update_start < 0 or update_count < 0 or sync_interval < 1:
        raise ValueError("invalid teacher snapshot window arguments")
    windows = []
    cursor = 0
    while cursor < update_count:
        global_step = global_update_start + cursor
        offset = global_step % sync_interval
        count = min(sync_interval - offset, update_count - cursor)
        windows.append((global_step, count, offset == 0))
        cursor += count
    return windows


def plan_token_budget_chunks(
    segment_ids: Any,
    *,
    token_budget: int,
    max_rows: int,
) -> list[TokenBudgetChunk]:
    """Greedily group left-padded rows without exceeding the token budget."""

    import numpy as np

    segments = np.asarray(segment_ids)
    if segments.ndim != 2 or segments.shape[0] == 0:
        raise ValueError("segment_ids must be a non-empty rank-2 array")
    if token_budget < 1 or max_rows < 1:
        raise ValueError("token_budget and max_rows must be positive")

    bounds: list[tuple[int, int]] = []
    for row in segments:
        active = np.flatnonzero(row)
        if active.size == 0:
            raise ValueError("cannot schedule a row without active tokens")
        bounds.append((int(active[0]), int(active[-1]) + 1))

    chunks: list[TokenBudgetChunk] = []
    pending: list[int] = []
    for row_index in range(len(bounds)):
        candidate = [*pending, row_index]
        start = min(bounds[index][0] for index in candidate)
        end = max(bounds[index][1] for index in candidate)
        padded = len(candidate) * (end - start)
        if pending and (len(candidate) > max_rows or padded > token_budget):
            start = min(bounds[index][0] for index in pending)
            end = max(bounds[index][1] for index in pending)
            chunks.append(
                TokenBudgetChunk(
                    rows=tuple(pending),
                    start_column=start,
                    end_column=end,
                    padded_token_count=len(pending) * (end - start),
                )
            )
            pending = [row_index]
        else:
            pending = candidate
        single_width = bounds[row_index][1] - bounds[row_index][0]
        if single_width > token_budget:
            raise ValueError(
                f"row {row_index} requires {single_width} tokens, budget={token_budget}"
            )
    if pending:
        start = min(bounds[index][0] for index in pending)
        end = max(bounds[index][1] for index in pending)
        chunks.append(
            TokenBudgetChunk(
                rows=tuple(pending),
                start_column=start,
                end_column=end,
                padded_token_count=len(pending) * (end - start),
            )
        )
    if any(
        len(chunk.rows) > max_rows or chunk.padded_token_count > token_budget
        for chunk in chunks
    ):
        raise AssertionError("internal token-budget scheduler error")
    return chunks


def _karmarkar_karp_partitions(
    workloads: Sequence[int],
    partition_count: int,
) -> list[list[int]]:
    """Mirror VERL's unequal-size Karmarkar-Karp workload partitioner."""

    if not 1 <= partition_count <= len(workloads):
        raise ValueError("partition_count must be within the workload count")

    class PartitionSet:
        def __init__(self) -> None:
            self.total = 0
            self.items: list[tuple[int, int]] = []

        def add(self, index: int, value: int) -> None:
            self.items.append((index, value))
            self.total += value

        def merge(self, other: PartitionSet) -> None:
            for index, value in other.items:
                self.add(index, value)

        def __lt__(self, other: PartitionSet) -> bool:
            return (self.total, len(self.items), self.items) < (
                other.total,
                len(other.items),
                other.items,
            )

    class State:
        def __init__(self, index: int, value: int) -> None:
            self.sets = [PartitionSet() for _ in range(partition_count)]
            self.sets[0].add(index, value)
            self.sets.sort(reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].total - self.sets[-1].total

        def merge(self, other: State) -> None:
            for index in range(partition_count):
                self.sets[index].merge(other.sets[-1 - index])
            self.sets.sort(reverse=True)

        def __lt__(self, other: State) -> bool:
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    queue = [
        State(index, int(value))
        for value, index in sorted(
            (int(value), index) for index, value in enumerate(workloads)
        )
    ]
    heapq.heapify(queue)
    while len(queue) > 1:
        first = heapq.heappop(queue)
        first.merge(heapq.heappop(queue))
        heapq.heappush(queue, first)
    partitions = [
        sorted(index for index, _ in partition.items) for partition in queue[0].sets
    ]
    if len(partitions) != partition_count or any(not row for row in partitions):
        raise AssertionError("invalid Karmarkar-Karp partition")
    return partitions


def plan_actor_update_chunks(
    packed: Mapping[str, Any],
    contract: OriginalRlcsdContract,
) -> list[TokenBudgetChunk]:
    """Mirror reference teacher-aware dynamic actor micro-batching."""

    import numpy as np

    policy_lengths = np.asarray(packed["policy_segment_ids"]).sum(axis=1)
    correct_lengths = np.asarray(packed["teacher_correct_segment_ids"]).sum(axis=1)
    wrong_lengths = np.asarray(packed["teacher_wrong_segment_ids"]).sum(axis=2)
    wrong_valid = np.asarray(packed["teacher_wrong_valid_mask"], dtype=bool)

    def workload(lengths: Any) -> Any:
        lengths = np.asarray(lengths, dtype=np.int64)
        return 24576 * lengths + lengths**2

    effective_lengths = np.maximum(
        np.maximum(policy_lengths, correct_lengths),
        np.where(wrong_valid, wrong_lengths, 0).max(axis=1),
    )
    if int(effective_lengths.max()) > contract.actor_max_token_len_per_worker:
        raise ValueError("teacher-aware sequence exceeds actor token budget")
    effective_workloads = (
        workload(policy_lengths)
        + workload(correct_lengths)
        + np.where(wrong_valid, workload(wrong_lengths), 0).sum(axis=1)
    )
    max_workload = int(workload(contract.actor_max_token_len_per_worker))
    partition_count = min(
        len(policy_lengths),
        max(
            1,
            math.ceil(int(effective_workloads.sum()) / max(max_workload, 1)),
        ),
    )
    partitions = _karmarkar_karp_partitions(
        effective_workloads.astype(int).tolist(),
        partition_count,
    )
    partitions.sort(
        key=lambda rows: (
            sum(int(effective_workloads[row]) for row in rows),
            rows[0],
        ),
        reverse=True,
    )
    partitions = partitions[::2][::-1] + partitions[1::2]

    segments = np.asarray(packed["policy_segment_ids"])
    chunks = []
    for rows in partitions:
        active = [np.flatnonzero(segments[row]) for row in rows]
        start = min(int(indices[0]) for indices in active)
        end = max(int(indices[-1]) + 1 for indices in active)
        chunks.append(
            TokenBudgetChunk(
                rows=tuple(rows),
                start_column=start,
                end_column=end,
                padded_token_count=len(rows) * (end - start),
            )
        )
    return chunks


def _trim_chunk(
    input_ids: Any,
    position_ids: Any,
    segment_ids: Any,
    target_positions: Any,
    response_mask: Any,
    chunk: TokenBudgetChunk,
) -> tuple[Any, Any, Any, Any, Any]:
    import numpy as np

    rows = np.asarray(chunk.rows, dtype=np.int32)
    start = chunk.start_column
    end = chunk.end_column
    trimmed_targets = np.asarray(target_positions)[rows] - start
    mask = np.asarray(response_mask)[rows]
    trimmed_targets = np.where(mask > 0, trimmed_targets, 0)
    if np.any(trimmed_targets[mask > 0] < 0):
        raise ValueError("response target fell before trimmed sequence")
    if np.any(trimmed_targets[mask > 0] >= end - start - 1):
        raise ValueError("response target fell outside trimmed sequence")
    return (
        np.asarray(input_ids)[rows, start:end],
        np.asarray(position_ids)[rows, start:end],
        np.asarray(segment_ids)[rows, start:end],
        trimmed_targets.astype(np.int32),
        mask,
    )


def _block_tree(value: Any) -> Any:
    import jax

    return jax.block_until_ready(value)


def _shape_signature(values: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(size) for size in value.shape) for value in values)


def _make_response_log_prob_fn() -> Any:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    @nnx.jit
    def response_log_prob_fn(
        module: Any,
        input_ids: Any,
        position_ids: Any,
        segment_ids: Any,
        target_positions: Any,
        response_mask: Any,
    ) -> Any:
        logits, _ = module(
            input_ids,
            position_ids,
            None,
            None,
            decoder_segment_ids=segment_ids,
        )
        batch_indices = jnp.arange(input_ids.shape[0])[:, None]
        response_logits = logits[batch_indices, target_positions, :].astype(jnp.float32)
        response_tokens = input_ids[
            batch_indices,
            target_positions + 1,
        ]
        selected_logits = jnp.take_along_axis(
            response_logits,
            response_tokens[..., None],
            axis=-1,
        )[..., 0]
        log_probs = selected_logits - jax.nn.logsumexp(
            response_logits,
            axis=-1,
        )
        return jnp.where(response_mask.astype(bool), log_probs, 0.0)

    return response_log_prob_fn


def _execute_response_chunks(
    *,
    model: Any,
    input_ids: Any,
    position_ids: Any,
    segment_ids: Any,
    target_positions: Any,
    response_mask: Any,
    chunks: Sequence[TokenBudgetChunk],
    response_fn: Any,
    warmed_signatures: set[tuple[tuple[int, ...], ...]],
) -> tuple[Any, list[float], list[float]]:
    import numpy as np

    output = np.zeros(np.asarray(response_mask).shape, dtype=np.float32)
    compile_warmup_s: list[float] = []
    execution_s: list[float] = []
    for chunk in chunks:
        values = _trim_chunk(
            input_ids,
            position_ids,
            segment_ids,
            target_positions,
            response_mask,
            chunk,
        )
        signature = _shape_signature(values)
        if signature not in warmed_signatures:
            started = time.perf_counter()
            _block_tree(response_fn(model, *values))
            compile_warmup_s.append(time.perf_counter() - started)
            warmed_signatures.add(signature)
        started = time.perf_counter()
        result = _block_tree(response_fn(model, *values))
        execution_s.append(time.perf_counter() - started)
        output[list(chunk.rows)] = np.asarray(result, dtype=np.float32)
    return output, compile_warmup_s, execution_s


def compute_teacher_log_probs(
    model: Any,
    packed: Mapping[str, Any],
    contract: OriginalRlcsdContract,
    *,
    response_fn: Any,
    warmed_signatures: set[tuple[tuple[int, ...], ...]],
) -> TeacherLogProbBatch:
    """Run frozen-snapshot correct and K-marginal wrong teacher passes."""

    import numpy as np

    response_mask = np.asarray(packed["response_mask"])
    correct_chunks = plan_token_budget_chunks(
        packed["teacher_correct_segment_ids"],
        token_budget=contract.actor_max_token_len_per_worker,
        max_rows=contract.ppo_micro_batch_size_per_worker,
    )
    correct, compile_rows, execution_rows = _execute_response_chunks(
        model=model,
        input_ids=packed["teacher_correct_input_ids"],
        position_ids=packed["teacher_correct_position_ids"],
        segment_ids=packed["teacher_correct_segment_ids"],
        target_positions=packed["teacher_correct_response_target_positions"],
        response_mask=response_mask,
        chunks=correct_chunks,
        response_fn=response_fn,
        warmed_signatures=warmed_signatures,
    )

    wrong_valid = np.asarray(packed["teacher_wrong_valid_mask"], dtype=bool)
    valid_coordinates = list(zip(*np.nonzero(wrong_valid), strict=True))
    if not valid_coordinates:
        raise ValueError("teacher wrong-context tensor has no valid rows")
    wrong_inputs = np.stack(
        [
            packed["teacher_wrong_input_ids"][row, column]
            for row, column in valid_coordinates
        ]
    )
    wrong_positions = np.stack(
        [
            packed["teacher_wrong_position_ids"][row, column]
            for row, column in valid_coordinates
        ]
    )
    wrong_segments = np.stack(
        [
            packed["teacher_wrong_segment_ids"][row, column]
            for row, column in valid_coordinates
        ]
    )
    wrong_targets = np.stack(
        [
            packed["teacher_wrong_response_target_positions"][row, column]
            for row, column in valid_coordinates
        ]
    )
    wrong_response_mask = np.stack([response_mask[row] for row, _ in valid_coordinates])
    wrong_chunks = plan_token_budget_chunks(
        wrong_segments,
        token_budget=contract.actor_max_token_len_per_worker,
        max_rows=contract.ppo_micro_batch_size_per_worker,
    )
    wrong_flat, wrong_compile, wrong_execution = _execute_response_chunks(
        model=model,
        input_ids=wrong_inputs,
        position_ids=wrong_positions,
        segment_ids=wrong_segments,
        target_positions=wrong_targets,
        response_mask=wrong_response_mask,
        chunks=wrong_chunks,
        response_fn=response_fn,
        warmed_signatures=warmed_signatures,
    )
    wrong = np.zeros(
        (*wrong_valid.shape, response_mask.shape[1]),
        dtype=np.float32,
    )
    for flat_index, (row, column) in enumerate(valid_coordinates):
        wrong[row, column] = wrong_flat[flat_index]
    return TeacherLogProbBatch(
        correct=correct,
        wrong=wrong,
        compile_warmup_s=(*compile_rows, *wrong_compile),
        execution_s=(*execution_rows, *wrong_execution),
    )


def _finite_metric_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        for key in ("loss", "grad_norm", "learning_rate"):
            if not math.isfinite(float(row[key])):
                return False
    return True


def run_original_rlcsd_updates(
    *,
    model: Any,
    mesh: Any,
    packed_batches: Sequence[Mapping[str, Any]],
    contract: OriginalRlcsdContract,
    max_updates: int | None,
    actor_state: OriginalRlcsdActorState | None = None,
) -> tuple[dict[str, Any], OriginalRlcsdActorState]:
    """Execute snapshot-teacher RLCSD PPO updates over one rollout batch."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from flax.linen import partitioning

    if not packed_batches:
        raise ValueError("no valid RLCSD PPO mini-batches")
    if max_updates is not None:
        if max_updates < 1:
            raise ValueError("max_updates must be positive")
        packed_batches = packed_batches[:max_updates]

    import optax

    if actor_state is None:
        optimizer, schedule = create_original_rlcsd_optimizer(model, contract)
        actor_state = OriginalRlcsdActorState(
            optimizer=optimizer,
            teacher_params=None,
            global_update_step=0,
        )
    else:
        optimizer = actor_state.optimizer
        schedule = original_rlcsd_schedule(contract)
    graphdef, _, _ = nnx.split(model, nnx.Param, ...)

    def loss_wrapper(
        diff_params: Any,
        fixed_rest: Any,
        input_ids: Any,
        position_ids: Any,
        segment_ids: Any,
        target_positions: Any,
        response_mask: Any,
        old_log_probs: Any,
        advantages: Any,
        teacher_correct: Any,
        teacher_wrong: Any,
        teacher_wrong_valid: Any,
        loss_scale_factor: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        local_model = nnx.merge(
            graphdef,
            diff_params,
            fixed_rest,
            copy=True,
        )
        logits, _ = local_model(
            input_ids,
            position_ids,
            None,
            None,
            decoder_segment_ids=segment_ids,
        )
        batch_indices = jnp.arange(input_ids.shape[0])[:, None]
        response_logits = logits[
            batch_indices,
            target_positions,
            :,
        ].astype(jnp.float32)
        response_tokens = input_ids[
            batch_indices,
            target_positions + 1,
        ]
        selected_logits = jnp.take_along_axis(
            response_logits,
            response_tokens[..., None],
            axis=-1,
        )[..., 0]
        log_prob = selected_logits - jax.nn.logsumexp(
            response_logits,
            axis=-1,
        )
        log_prob = jnp.where(
            response_mask.astype(bool),
            log_prob,
            0.0,
        )
        rollout_is = (
            jnp.minimum(
                jnp.exp(
                    jnp.clip(
                        log_prob - old_log_probs,
                        -20.0,
                        20.0,
                    )
                ),
                contract.rollout_is_threshold,
            )
            * response_mask
        )
        loss, metrics = rlcsd_policy_loss(
            old_log_prob=old_log_probs,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            teacher_correct_log_prob=teacher_correct,
            teacher_wrong_multi_log_prob=teacher_wrong,
            teacher_wrong_multi_valid_mask=teacher_wrong_valid,
            epsilon=contract.epsilon,
            tau=contract.rlcsd_tau,
            beta=contract.rlcsd_beta,
            lam=contract.rlcsd_lam,
            delta=contract.rlcsd_delta,
            eta=contract.rlcsd_eta,
            residual_clip_low=contract.rlcsd_residual_clip_low,
            residual_clip_high=contract.rlcsd_residual_clip_high,
            rollout_is_weights=rollout_is,
        )
        return loss * loss_scale_factor, metrics

    grad_fn = jax.jit(
        jax.value_and_grad(
            loss_wrapper,
            argnums=0,
            has_aux=True,
        )
    )
    response_fn = _make_response_log_prob_fn()
    teacher_warmed: set[tuple[tuple[int, ...], ...]] = set()
    actor_warmed: set[tuple[tuple[int, ...], ...]] = set()
    teacher_compile_s: list[float] = []
    teacher_execution_s: list[float] = []
    actor_compile_s: list[float] = []
    update_rows: list[dict[str, Any]] = []
    teacher_sync_events = 0

    logical_axis_rules = model.base.config.logical_axis_rules
    with jax.set_mesh(mesh), partitioning.axis_rules(logical_axis_rules):
        cursor = 0
        windows = plan_teacher_snapshot_windows(
            actor_state.global_update_step,
            len(packed_batches),
            contract.teacher_sync_interval,
        )
        for global_step, window_size, refresh_before in windows:
            if actor_state.teacher_params is None or refresh_before:
                actor_state.teacher_params = _snapshot_params(model)
                teacher_sync_events += 1
            window = packed_batches[cursor : cursor + window_size]
            _, _, current_rest = nnx.split(model, nnx.Param, ...)
            teacher_model = nnx.merge(
                graphdef,
                actor_state.teacher_params,
                current_rest,
                copy=True,
            )
            # All teacher logits in this window are produced before the first
            # actor mutation and from the persisted snapshot state.  This keeps
            # a step-30 snapshot active for steps 30..39 even when a rollout
            # batch ends at step 32.
            teacher_rows = [
                compute_teacher_log_probs(
                    teacher_model,
                    packed,
                    contract,
                    response_fn=response_fn,
                    warmed_signatures=teacher_warmed,
                )
                for packed in window
            ]
            teacher_compile_s.extend(
                value for row in teacher_rows for value in row.compile_warmup_s
            )
            teacher_execution_s.extend(
                value for row in teacher_rows for value in row.execution_s
            )

            for local_index, (packed, teacher) in enumerate(
                zip(window, teacher_rows, strict=True)
            ):
                update_index = global_step + local_index
                _, params, rest = nnx.split(model, nnx.Param, ...)
                policy_chunks = plan_actor_update_chunks(packed, contract)
                accumulated_grads = None
                metric_sums: dict[str, float] = {}
                execution_s = 0.0
                for chunk in policy_chunks:
                    values = _trim_chunk(
                        packed["policy_input_ids"],
                        packed["policy_position_ids"],
                        packed["policy_segment_ids"],
                        packed["policy_response_target_positions"],
                        packed["response_mask"],
                        chunk,
                    )
                    rows = np.asarray(chunk.rows, dtype=np.int32)
                    arguments = (
                        params,
                        rest,
                        *values,
                        np.asarray(packed["old_log_probs"])[rows],
                        np.asarray(packed["advantages"])[rows],
                        np.asarray(teacher.correct)[rows],
                        np.asarray(teacher.wrong)[rows],
                        np.asarray(packed["teacher_wrong_valid_mask"])[rows],
                        np.asarray(
                            len(chunk.rows) / contract.ppo_mini_batch_size,
                            dtype=np.float32,
                        ),
                    )
                    signature = _shape_signature(arguments[2:])
                    if signature not in actor_warmed:
                        started = time.perf_counter()
                        _block_tree(grad_fn(*arguments))
                        actor_compile_s.append(time.perf_counter() - started)
                        actor_warmed.add(signature)
                    started = time.perf_counter()
                    (loss_and_metrics, grads) = _block_tree(grad_fn(*arguments))
                    execution_s += time.perf_counter() - started
                    loss, metrics = loss_and_metrics
                    accumulated_grads = (
                        grads
                        if accumulated_grads is None
                        else jax.tree.map(
                            lambda left, right: left + right,
                            accumulated_grads,
                            grads,
                        )
                    )
                    metric_sums["loss"] = metric_sums.get("loss", 0.0) + float(loss)
                    for key, value in metrics.items():
                        if key == "loss":
                            continue
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                if accumulated_grads is None:
                    raise AssertionError("actor produced no gradients")
                grad_norm = float(
                    optax.global_norm(
                        jax.tree.map(
                            lambda value: value.astype(jnp.float32),
                            accumulated_grads,
                        )
                    )
                )
                update_started = time.perf_counter()
                optimizer.update(model, accumulated_grads)
                _block_tree(nnx.state(model, nnx.Param))
                optimizer_update_s = time.perf_counter() - update_started
                learning_rate = float(schedule(update_index))
                response_tokens = int(np.asarray(packed["response_mask"]).sum())
                row = {
                    "update_index": update_index,
                    "loss": metric_sums["loss"],
                    "grad_norm": grad_norm,
                    "learning_rate": learning_rate,
                    "response_token_count": response_tokens,
                    "policy_micro_batch_count": len(policy_chunks),
                    "micro_batching": ("verl_teacher_aware_karmarkar_karp_workload"),
                    "policy_execution_s": execution_s,
                    "optimizer_update_s": optimizer_update_s,
                    "optimizer_update_applied": True,
                    "response_tokens_per_s": (
                        response_tokens / max(execution_s + optimizer_update_s, 1e-12)
                    ),
                    "selected_token_count": metric_sums.get(
                        "selected_token_count",
                        0.0,
                    ),
                    "teacher_snapshot_window": (
                        update_index // contract.teacher_sync_interval
                    ),
                }
                update_rows.append(row)
            cursor += window_size

    actor_state.global_update_step += len(update_rows)

    if not _finite_metric_rows(update_rows):
        raise RuntimeError("RLCSD actor produced non-finite metrics")
    timed_rows = update_rows[1:] if len(update_rows) > 1 else []
    return (
        {
            "schema_version": 1,
            "ok": bool(timed_rows),
            "phase": "rlcsd_original_update",
            "teacher_sync_event_count": teacher_sync_events,
            "global_update_start": (actor_state.global_update_step - len(update_rows)),
            "global_update_end": actor_state.global_update_step,
            "teacher_compile_warmup_s": teacher_compile_s,
            "teacher_execution_s": teacher_execution_s,
            "actor_compile_warmup_s": actor_compile_s,
            "compile_and_warmup_update": (update_rows[0] if update_rows else None),
            "timed_updates": timed_rows,
            "update_count": len(update_rows),
            "finite_metrics": True,
            "nonzero_learning_rate_update_count": sum(
                row["learning_rate"] > 0 for row in update_rows
            ),
            "mean_timed_response_tokens_per_s": (
                statistics.fmean(row["response_tokens_per_s"] for row in timed_rows)
                if timed_rows
                else None
            ),
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
        },
        actor_state,
    )


def _load_and_validate_rollout(
    rollout_path: Path,
    rollout_manifest_path: Path,
    contract: OriginalRlcsdContract,
) -> tuple[list[Any], dict[str, Any]]:
    manifest = json.loads(rollout_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ok") is not True or manifest.get("phase") != "rlcsd_rollout":
        raise ValueError("rollout manifest is not a successful rollout phase")
    if manifest.get("contract") != contract.as_dict():
        raise ValueError("rollout manifest does not match the current contract")
    expected_hash = manifest.get("rollout", {}).get("output_sha256")
    observed_hash = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
    if expected_hash != observed_hash:
        raise ValueError(f"rollout hash mismatch: {observed_hash} != {expected_hash}")
    samples = read_rollout_jsonl(rollout_path)
    validate_rollout_samples(samples, contract)
    return samples, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--rollout-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-items", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-updates",
        type=int,
        help=(
            "Explicit diagnostic cap. Omit for every valid PPO mini-batch; "
            "capped runs cannot pass the original-update audit."
        ),
    )
    args = parser.parse_args()

    contract = load_original_contract()
    samples, rollout_manifest = _load_and_validate_rollout(
        args.rollouts,
        args.rollout_manifest,
        contract,
    )
    advantages = compute_grpo_advantages(samples, contract)
    assignments, alignment = build_teacher_assignments(
        samples,
        contract,
        seed=args.seed,
    )
    if len(assignments) < contract.ppo_mini_batch_size * 2:
        raise RuntimeError(
            "At least two valid PPO mini-batches are required to separate "
            "compile/warmup from a timed update."
        )

    checkpoint_manifest = read_checkpoint_manifest(
        args.checkpoint_items,
        ORIGINAL_RLCSD_CASE,
    )
    devices, environment = require_tpu_v5e8()
    memory = [memory_snapshot("before_model")]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract.model_id,
        revision=checkpoint_manifest["resolved_hf_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    packed_batches = [
        pack_actor_minibatch(batch, advantages, tokenizer, contract)
        for batch in iter_actor_minibatches(assignments, contract)
    ]
    model, mesh, resolved_path, weights = load_model(
        ORIGINAL_RLCSD_CASE,
        devices=devices,
        checkpoint_items=args.checkpoint_items,
        allow_random_weights=False,
    )
    memory.append(memory_snapshot("after_model"))
    sharding = parameter_sharding_report(model, devices)
    result, actor_state = run_original_rlcsd_updates(
        model=model,
        mesh=mesh,
        packed_batches=packed_batches,
        contract=contract,
        max_updates=args.max_updates,
    )
    memory.append(memory_snapshot("after_updates"))
    del actor_state

    expected_full_updates = math.ceil(len(assignments) / contract.ppo_mini_batch_size)
    capped = args.max_updates is not None and args.max_updates < expected_full_updates
    result.update(
        {
            "ok": (
                result["ok"]
                and result["nonzero_learning_rate_update_count"] > 0
                and not capped
            ),
            "is_rlcsd_end_to_end": False,
            "is_original_rlcsd_core_update": not capped,
            "contract": contract.as_dict(),
            "case": ORIGINAL_RLCSD_CASE.as_dict(),
            "weights": weights,
            "resolved_model_path": resolved_path,
            "environment": environment,
            "dataset": rollout_manifest["dataset"],
            "checkpoint_provenance": {
                key: checkpoint_manifest.get(key)
                for key in (
                    "model_id",
                    "resolved_hf_revision",
                    "save_dtype",
                    "tree_metadata_sha256",
                )
            },
            "rollout": rollout_manifest["rollout"],
            "alignment": alignment,
            "expected_full_update_count": expected_full_updates,
            "diagnostic_update_cap": args.max_updates,
            "parameter_sharding": sharding,
            "memory": memory,
        }
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
