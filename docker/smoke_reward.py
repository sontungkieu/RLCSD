from __future__ import annotations

import itertools


_COUNTER = itertools.count()


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    del data_source, ground_truth, extra_info
    step = next(_COUNTER)
    reward = 1.0 if step % 2 == 0 else 0.0
    formatted = 1.0 if "\\boxed" in solution_str else 0.0
    return {
        "score": reward,
        "acc": reward,
        "formatted": formatted,
        "smoke_forced_reward": reward,
    }
