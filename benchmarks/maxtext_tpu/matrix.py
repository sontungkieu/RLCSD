"""Canonical RLCSD/MaxText benchmark matrix.

The matrix is deliberately pure Python so it can be validated without a TPU or
the heavyweight JAX/MaxText environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

DATASET_REPO = "Leyiii/RLCSD"
DATASET_REVISION = "33d7de919af5b03257ff92c30303fddf9afdda4a"
SEQUENCE_LENGTH = 512
EXPECTED_TPU_DEVICES = 8
TUNIX_PIPELINE_COMMIT = "50f5752a17edec56e2aa30aabfc03859949adf6f"


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    model_id: str
    rlcsd_config_path: str
    dataset_file: str
    num_decoder_layers: int


@dataclasses.dataclass(frozen=True, slots=True)
class LayoutSpec:
    label: str
    pipeline_parallelism: int
    tensor_parallelism: int

    @property
    def required_device_count(self) -> int:
        return self.pipeline_parallelism * self.tensor_parallelism


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkCase:
    model: ModelSpec
    layout: LayoutSpec
    global_batch_size: int
    sequence_length: int
    num_pipeline_microbatches: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.model.key}-bs{self.global_batch_size}-{self.layout.label.lower()}"
        )

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self.model.num_decoder_layers // self.layout.pipeline_parallelism

    @property
    def maxtext_pipeline_kwargs(self) -> dict[str, int]:
        """Return the MaxText structure-defining pipeline overrides."""

        return {
            "ici_pipeline_parallelism": self.layout.pipeline_parallelism,
            "ici_tensor_parallelism": self.layout.tensor_parallelism,
            "ici_data_parallelism": 1,
            "ici_fsdp_parallelism": 1,
            "num_layers_per_pipeline_stage": self.num_layers_per_pipeline_stage,
            "num_pipeline_microbatches": self.num_pipeline_microbatches,
        }

    def validate(self) -> None:
        if self.layout.required_device_count != EXPECTED_TPU_DEVICES:
            raise ValueError(
                f"{self.case_id} requires {self.layout.required_device_count} "
                f"devices, expected {EXPECTED_TPU_DEVICES}."
            )
        if self.model.num_decoder_layers % self.layout.pipeline_parallelism:
            raise ValueError(
                f"{self.case_id}: decoder layers are not divisible by stages."
            )
        if self.num_pipeline_microbatches % self.layout.pipeline_parallelism:
            raise ValueError(
                f"{self.case_id}: microbatches must be divisible by stages."
            )
        if self.global_batch_size % self.num_pipeline_microbatches:
            raise ValueError(
                f"{self.case_id}: batch must be divisible by microbatches."
            )

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "case_id": self.case_id,
            "model_key": self.model.key,
            "model_id": self.model.model_id,
            "rlcsd_config_path": self.model.rlcsd_config_path,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "dataset_file": self.model.dataset_file,
            "global_batch_size": self.global_batch_size,
            "sequence_length": self.sequence_length,
            "num_decoder_layers": self.model.num_decoder_layers,
            "parallelism": {
                "label": self.layout.label,
                "pipeline_parallelism": self.layout.pipeline_parallelism,
                "tensor_parallelism": self.layout.tensor_parallelism,
                "num_layers_per_pipeline_stage": (self.num_layers_per_pipeline_stage),
                "num_pipeline_microbatches": (self.num_pipeline_microbatches),
                "required_device_count": self.layout.required_device_count,
            },
            "tunix_pipeline_commit": TUNIX_PIPELINE_COMMIT,
        }


MODELS = (
    ModelSpec(
        key="qwen3_1_7b",
        model_id="Qwen/Qwen3-1.7B",
        rlcsd_config_path=("configs/math_deepmath/qwen3_1_7b_rlcsd.yaml"),
        dataset_file="deepmath_filtered_level5_7/train.parquet",
        num_decoder_layers=28,
    ),
    ModelSpec(
        key="qwen3_4b",
        model_id="Qwen/Qwen3-4B",
        rlcsd_config_path="configs/math_deepmath/qwen3_4b_rlcsd.yaml",
        dataset_file="deepmath_filtered_level6_8/train.parquet",
        num_decoder_layers=36,
    ),
    ModelSpec(
        key="qwen3_8b",
        model_id="Qwen/Qwen3-8B",
        rlcsd_config_path="configs/math_deepmath/qwen3_8b_rlcsd.yaml",
        dataset_file="deepmath_filtered_level7_10/train.parquet",
        num_decoder_layers=36,
    ),
)

LAYOUTS = (
    LayoutSpec(
        label="PP2xTP4",
        pipeline_parallelism=2,
        tensor_parallelism=4,
    ),
    LayoutSpec(
        label="PP4xTP2",
        pipeline_parallelism=4,
        tensor_parallelism=2,
    ),
)

BATCH_SIZES = (12, 16, 64)


def _microbatch_count(layout: LayoutSpec, batch_size: int) -> int:
    # PP4xTP2 used eight microbatches in the batch-16/64 forward runs. Batch
    # 12 cannot be divided by eight, so it uses four. This is intentional and
    # must remain visible in comparisons.
    if layout.pipeline_parallelism == 4:
        return 4 if batch_size == 12 else 8
    return 4


def iter_cases() -> Iterable[BenchmarkCase]:
    for model in MODELS:
        for batch_size in BATCH_SIZES:
            for layout in LAYOUTS:
                case = BenchmarkCase(
                    model=model,
                    layout=layout,
                    global_batch_size=batch_size,
                    sequence_length=SEQUENCE_LENGTH,
                    num_pipeline_microbatches=_microbatch_count(layout, batch_size),
                )
                case.validate()
                yield case


def get_case(case_id: str) -> BenchmarkCase:
    for case in iter_cases():
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in iter_cases())
    raise KeyError(f"Unknown case {case_id!r}. Available cases: {choices}")


def validate_repo_configs(repo_root: Path) -> None:
    for model in MODELS:
        config_path = repo_root / model.rlcsd_config_path
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config_text = config_path.read_text(encoding="utf-8")
        required_lines = {
            "method: rlcsd",
            f"train_dataset: {model.dataset_file.rsplit('/', 1)[0]}",
        }
        missing = sorted(line for line in required_lines if line not in config_text)
        if missing:
            raise ValueError(f"{config_path} does not match the matrix: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument(
        "--validate-repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    validate_repo_configs(args.validate_repo_root)
    cases = [get_case(args.case)] if args.case else list(iter_cases())
    payload = {
        "schema_version": 1,
        "case_count": len(cases),
        "cases": [case.as_dict() for case in cases],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
