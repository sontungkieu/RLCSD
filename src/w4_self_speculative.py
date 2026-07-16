"""Online W4 self-drafter for vLLM 0.24 speculative decoding.

The target actor remains BF16. A separate W4 copy is refreshed after each
complete veRL weight transfer and uses vLLM's native draft-model path.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from copy import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.scalar_type import scalar_types
from vllm.v1.spec_decode.draft_model import DraftModelProposer


logger = logging.getLogger(__name__)

_W4_BITS = 4
_W4_PACK_FACTOR = 32 // _W4_BITS
_AWQ_INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_awq_columns(values: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values in the checkpoint layout expected by AutoAWQ."""
    size_k, size_n = values.shape
    if size_n % _W4_PACK_FACTOR:
        raise ValueError(f"AWQ output dimension must be divisible by 8, got {size_n}")

    order = torch.tensor(_AWQ_INTERLEAVE, device=values.device)
    interleaved = (
        values.reshape(-1, _W4_PACK_FACTOR)
        .index_select(1, order)
        .reshape(size_k, size_n)
        .to(torch.int32)
    )
    packed = torch.zeros(
        size_k,
        size_n // _W4_PACK_FACTOR,
        dtype=torch.int32,
        device=values.device,
    )
    for index in range(_W4_PACK_FACTOR):
        packed.bitwise_or_(interleaved[:, index::_W4_PACK_FACTOR] << (_W4_BITS * index))
    return packed.contiguous()


def _unwrap_model(model: nn.Module) -> nn.Module:
    seen: set[int] = set()
    while id(model) not in seen:
        seen.add(id(model))
        unwrap = getattr(model, "unwrap", None)
        if not callable(unwrap):
            break
        unwrapped = unwrap()
        if unwrapped is model:
            break
        model = unwrapped
    return model


class W4SelfSpeculativeProposer(DraftModelProposer):
    """RTN W4 draft copy of the live actor, backed by AutoAWQ kernels."""

    uses_native_draft_model_path = True

    def __init__(self, vllm_config: VllmConfig):
        # vLLM's custom-class API only passes VllmConfig. The small runner hook
        # calls bind_runner before model loading so the native proposer can init.
        self._bootstrap_vllm_config = vllm_config
        self._is_bound = False
        self._target_model: nn.Module | None = None
        self._draft_vllm_config: VllmConfig | None = None
        self._awq_checkpoint_shapes: dict[str, torch.Size] = {}
        self.last_refresh_seconds = 0.0

    def bind_runner(self, runner) -> None:
        if self._is_bound:
            return
        super().__init__(
            vllm_config=self._bootstrap_vllm_config,
            device=runner.device,
            runner=runner,
        )
        self._reserve_speculative_scheduler_slots()
        self.group_size = int(os.environ.get("RLCSD_W4_GROUP_SIZE", "128"))
        if self.group_size not in (32, 64, 128):
            raise ValueError(
                "RLCSD_W4_GROUP_SIZE must be one of 32, 64, or 128, "
                f"got {self.group_size}"
            )
        self._is_bound = True
        logger.info(
            "Initialized W4 self-speculative proposer: gamma=%d group_size=%d",
            self.num_speculative_tokens,
            self.group_size,
        )

    def _reserve_speculative_scheduler_slots(self) -> None:
        scheduler_config = self.vllm_config.scheduler_config
        slot_delta = (
            self.speculative_config.max_num_new_slots_for_drafting
            * scheduler_config.max_num_seqs
        )
        max_scheduled = scheduler_config.max_num_batched_tokens - slot_delta
        if max_scheduled <= 0:
            raise ValueError(
                "W4 speculative decoding leaves no scheduler token budget: "
                f"max_num_batched_tokens={scheduler_config.max_num_batched_tokens}, "
                f"required_draft_slots={slot_delta}"
            )
        current = scheduler_config.max_num_scheduled_tokens
        if current is None or current > max_scheduled:
            scheduler_config.max_num_scheduled_tokens = max_scheduled
            logger.info(
                "Reserved %d W4 draft slots: max_num_scheduled_tokens=%d",
                slot_delta,
                max_scheduled,
            )

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        quant_config = AutoAWQConfig(
            weight_bits=4,
            group_size=self.group_size,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=[],
            full_config={
                "bits": 4,
                "group_size": self.group_size,
                "zero_point": True,
                "quant_method": "awq",
            },
        )
        # vLLM 0.24's ModelConfig stores the derived model_arch_config in
        # __dict__, which its generic replace helper cannot clone.
        model_config = copy(base.model_config)
        model_config.quantization = "auto_awq"
        return replace(base, quant_config=quant_config, model_config=model_config)

    def load_model(self, target_model: nn.Module) -> None:
        self._target_model = _unwrap_model(target_model)
        super().load_model(self._target_model)

    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader.utils import (
            initialize_model,
            process_weights_after_loading,
        )
        from vllm.model_executor.model_loader.reload import (
            record_metadata_for_reloading,
        )

        if self._target_model is None:
            raise RuntimeError(
                "W4 target model is not available during draft initialization"
            )

        draft_config = self._create_draft_vllm_config()
        self._draft_vllm_config = draft_config
        with set_model_tag("w4_self_drafter"):
            model = initialize_model(vllm_config=draft_config, prefix="draft_model")
        model = model.to(device=self.device)
        with torch.device(self.device):
            record_metadata_for_reloading(model)
        self._cast_float32_parameters(model)

        state = self._build_awq_state(model, self._target_model)
        self._load_named_state(model, state)
        process_weights_after_loading(model, draft_config.model_config, self.device)
        logger.info("Loaded W4 self-drafter from %d tensors", len(state))
        return model

    @torch.inference_mode()
    def refresh_from_target(self) -> float:
        """Requantize the draft after a complete actor-to-vLLM weight sync."""
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        if (
            self.model is None
            or self._target_model is None
            or self._draft_vllm_config is None
        ):
            raise RuntimeError("W4 self-drafter has not finished loading")

        start = time.perf_counter()
        state = self._build_awq_state(self.model, self._target_model)
        initialize_layerwise_reload(self.model)
        self._load_named_state(self.model, state)
        finalize_layerwise_reload(self.model, self._draft_vllm_config.model_config)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_refresh_seconds = time.perf_counter() - start
        logger.info(
            "Refreshed W4 self-drafter from %d tensors in %.3fs",
            len(state),
            self.last_refresh_seconds,
        )
        return self.last_refresh_seconds

    def _build_awq_state(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
    ) -> list[tuple[str, torch.Tensor]]:
        target_model = _unwrap_model(target_model)
        target_modules = dict(target_model.named_modules())
        target_params = dict(target_model.named_parameters())
        quantized_names: set[str] = set()
        state: list[tuple[str, torch.Tensor]] = []

        for module_name, draft_module in draft_model.named_modules():
            if not all(
                hasattr(draft_module, name) for name in ("qweight", "qzeros", "scales")
            ):
                continue
            target_module = target_modules.get(module_name)
            target_weight = getattr(target_module, "weight", None)
            if target_weight is None:
                raise KeyError(
                    f"Missing target linear weight for W4 module {module_name}"
                )

            qweight, scales, qzeros = self._quantize_weight(target_weight)
            tensors = {
                f"{module_name}.qweight": qweight,
                f"{module_name}.qzeros": qzeros,
                f"{module_name}.scales": scales.to(dtype=draft_module.scales.dtype),
            }
            for name, tensor in tensors.items():
                parameter_name = name.rsplit(".", 1)[-1]
                expected_shape = self._awq_checkpoint_shapes.get(name)
                if expected_shape is None:
                    expected_shape = dict(draft_module.named_parameters(recurse=False))[
                        parameter_name
                    ].shape
                    self._awq_checkpoint_shapes[name] = expected_shape
                if tensor.shape != expected_shape:
                    raise ValueError(
                        f"W4 tensor shape mismatch for {name}: "
                        f"generated={tuple(tensor.shape)} expected={tuple(expected_shape)}"
                    )
                state.append((name, tensor))
                quantized_names.add(name)

        for name, draft_param in draft_model.named_parameters():
            if name in quantized_names:
                continue
            target_param = target_params.get(name)
            if target_param is None:
                continue
            if target_param.shape != draft_param.shape:
                raise ValueError(
                    f"Non-quantized tensor shape mismatch for {name}: "
                    f"target={tuple(target_param.shape)} draft={tuple(draft_param.shape)}"
                )
            state.append((name, target_param.detach()))

        state.sort(key=lambda item: item[0])
        if not quantized_names:
            raise RuntimeError(
                "No AutoAWQ linear modules were found in the W4 draft model"
            )
        return state

    @staticmethod
    def _load_named_state(
        model: nn.Module,
        state: list[tuple[str, torch.Tensor]],
    ) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        parameters = dict(model.named_parameters(remove_duplicate=False))
        for name, tensor in state:
            parameter = parameters.get(name)
            if parameter is None:
                raise KeyError(f"Missing W4 draft parameter {name}")
            weight_loader = getattr(
                parameter,
                "weight_loader",
                default_weight_loader,
            )
            weight_loader(parameter, tensor)

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transposed = weight.detach().t().contiguous().float()
        dequantized, values, scales, zero_points = quantize_weights(
            transposed,
            quant_type=scalar_types.uint4,
            group_size=self.group_size,
            zero_points=True,
        )
        del dequantized, transposed
        if scales is None or zero_points is None:
            raise RuntimeError(
                "Asymmetric W4 quantization did not produce scales/zero points"
            )
        return (
            _pack_awq_columns(values),
            scales,
            _pack_awq_columns(zero_points),
        )

    def _cast_float32_parameters(self, model: nn.Module) -> None:
        target_dtype = self.vllm_config.model_config.dtype
        for parameter in model.parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(dtype=target_dtype)


def iter_w4_parameters(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Expose packed W4 parameters for diagnostics without materializing copies."""
    for name, parameter in model.named_parameters():
        if name.endswith((".qweight", ".qzeros", ".scales")):
            yield name, parameter
