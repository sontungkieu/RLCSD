from __future__ import annotations

import argparse
from pathlib import Path


def make_rows() -> list[dict]:
    problem = "Compute 1 + 1. Put the final answer in \\\\boxed{}."
    solution = "1 + 1 = 2, so the answer is \\\\boxed{2}."
    prompt = [{"role": "user", "content": problem}]
    return [
        {
            "data_source": "dapo_math_17k",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "2"},
            "extra_info": {
                "index": idx,
                "problem": problem,
                "solution": solution,
                "answer": "2",
            },
        }
        for idx in range(2)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/rlcsd-smoke/data")
    args = parser.parse_args()

    from datasets import Dataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    Dataset.from_list(rows).to_parquet(str(out_dir / "train.parquet"))
    Dataset.from_list(rows[:1]).to_parquet(str(out_dir / "val.parquet"))
    print(f"wrote smoke parquet files under {out_dir}")


if __name__ == "__main__":
    main()
