#!/usr/bin/env python3
"""Install the minimal vLLM 0.24 hook needed by the W4 self-drafter."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path


SUPPORTED_VLLM_VERSION = "0.24.0"
RUNNER_RELATIVE_PATH = Path("vllm/v1/worker/gpu_model_runner.py")
PATCH_MARKER = "RLCSD native custom draft-model path"

ANCHOR = """            if self.speculative_config.method == \"custom_class\":
                self.drafter = create_custom_proposer(  # type: ignore[assignment]
                    self.vllm_config
                )
"""

REPLACEMENT = f"""            if self.speculative_config.method == \"custom_class\":
                self.drafter = create_custom_proposer(  # type: ignore[assignment]
                    self.vllm_config
                )
                # {PATCH_MARKER}. Custom proposers that opt in can reuse
                # vLLM's native DraftModelProposer scheduling and KV-cache path.
                if getattr(self.drafter, \"uses_native_draft_model_path\", False):
                    self.speculative_config.method = \"draft_model\"
                if hasattr(self.drafter, \"bind_runner\"):
                    self.drafter.bind_runner(self)
"""


def discover_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("vLLM is not installed")
    return Path(next(iter(spec.submodule_search_locations))).parent


def patch_runner(vllm_root: Path, *, check_only: bool = False) -> bool:
    runner_path = vllm_root / RUNNER_RELATIVE_PATH
    source = runner_path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return False
    if check_only:
        raise RuntimeError(f"W4 hook is not installed in {runner_path}")
    if source.count(ANCHOR) != 1:
        raise RuntimeError(
            "Unsupported gpu_model_runner.py layout: expected exactly one "
            "vLLM 0.24 custom proposer anchor"
        )
    runner_path.write_text(source.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.vllm_root is None:
        installed_version = importlib.metadata.version("vllm")
        if installed_version != SUPPORTED_VLLM_VERSION:
            raise RuntimeError(
                f"W4 hook supports vLLM {SUPPORTED_VLLM_VERSION}, "
                f"found {installed_version}"
            )
        vllm_root = discover_vllm_root()
    else:
        vllm_root = args.vllm_root.resolve()

    changed = patch_runner(vllm_root, check_only=args.check)
    action = "installed" if changed else "already installed"
    print(f"RLCSD W4 vLLM hook {action}: {vllm_root / RUNNER_RELATIVE_PATH}")


if __name__ == "__main__":
    main()
