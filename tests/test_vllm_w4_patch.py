from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "docker" / "apply_vllm_w4_patch.py"
SPEC = importlib.util.spec_from_file_location("apply_vllm_w4_patch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH_MODULE)


class VllmW4PatchTest(unittest.TestCase):
    def test_patch_is_checked_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runner = root / PATCH_MODULE.RUNNER_RELATIVE_PATH
            runner.parent.mkdir(parents=True)
            runner.write_text(PATCH_MODULE.ANCHOR, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not installed"):
                PATCH_MODULE.patch_runner(root, check_only=True)

            self.assertTrue(PATCH_MODULE.patch_runner(root))
            self.assertIn(PATCH_MODULE.PATCH_MARKER, runner.read_text(encoding="utf-8"))
            self.assertFalse(PATCH_MODULE.patch_runner(root))
            self.assertFalse(PATCH_MODULE.patch_runner(root, check_only=True))

    def test_patch_rejects_unknown_runner_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runner = root / PATCH_MODULE.RUNNER_RELATIVE_PATH
            runner.parent.mkdir(parents=True)
            runner.write_text("# incompatible runner\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                PATCH_MODULE.patch_runner(root)


if __name__ == "__main__":
    unittest.main()
