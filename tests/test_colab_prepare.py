import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import venv
from unittest.mock import patch
from contextlib import ExitStack

MODULE = Path(__file__).resolve().parents[1] / "scripts/colab_prepare.py"
spec = importlib.util.spec_from_file_location("colab_prepare", MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def make_cache(self, root):
        (root / prepare.MODEL_NAME).write_bytes(b"model-fixture")
        prepare.write_json(root / "quality_report.json", {"status": "validated", "settings": prepare.settings(),
                           "models": {"Q4_K_M": {"size_bytes": 13, "sha256": prepare.sha256(root / prepare.MODEL_NAME)}}})
        (root / "smoke_prompt.txt").write_text("sample prompt")
        manifest = {"status": "validated", "settings_fingerprint": prepare.fingerprint(),
                    "model": {"filename": prepare.MODEL_NAME, "size_bytes": 13,
                              "sha256": prepare.sha256(root / prepare.MODEL_NAME)},
                    "quality_report": {"filename": "quality_report.json", "sha256": prepare.sha256(root / "quality_report.json")},
                    "smoke_prompt": {"filename": "smoke_prompt.txt", "sha256": prepare.sha256(root / "smoke_prompt.txt")}}
        prepare.write_json(root / "manifest.json", manifest)
        (root / "SHA256SUMS").write_text("".join(f"{prepare.sha256(root / name)}  {name}\n" for name in [prepare.MODEL_NAME, "quality_report.json", "manifest.json"]))
        prepare.write_json(root / "COMPLETE.json", {"manifest_sha256": prepare.sha256(root / "manifest.json")})

    def test_cache_accepts_only_complete_matching_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_cache(root)
            self.assertTrue(prepare.cache_valid(root)[0])
            self.assertFalse(prepare.cache_valid(root, "different-version")[0])
            (root / prepare.MODEL_NAME).write_bytes(b"model-corrupt")
            self.assertFalse(prepare.cache_valid(root)[0])

    def test_incomplete_copy_or_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_cache(root)
            (root / "COMPLETE.json").unlink()
            self.assertFalse(prepare.cache_valid(root)[0])
            self.make_cache(root)
            (root / "quality_report.json").unlink()
            self.assertFalse(prepare.cache_valid(root)[0])

    def test_missing_drive_fails_without_mount_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                prepare.check_environment(Path(folder) / "missing", 100 * 2**30, 10 * 2**30)
            with self.assertRaisesRegex(RuntimeError, "15 GiB"):
                prepare.check_environment(Path(folder), 14 * 2**30, 10 * 2**30)
            with self.assertRaisesRegex(RuntimeError, "6 GiB"):
                prepare.check_environment(Path(folder), 100 * 2**30, 5 * 2**30)

    def test_unvalidated_output_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(prepare, "ARTIFACTS", Path(folder)):
                with self.assertRaisesRegex(RuntimeError, "Refusing to publish"):
                    prepare.stage8_publish()

    def test_stale_successful_report_cannot_publish_new_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_cache(root)
            (root / prepare.MODEL_NAME).write_bytes(b"different-model-bytes")
            with patch.object(prepare, "ARTIFACTS", root):
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    prepare.stage8_publish()
            self.make_cache(root)
            quality = prepare.read_json(root / "quality_report.json")
            quality["settings"]["seed"] = 99
            with self.assertRaisesRegex(RuntimeError, "stale"):
                prepare.validate_quality_binding(quality, root / prepare.MODEL_NAME)

    def test_successful_publish_binds_local_and_drive_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, drive = root / "work", root / "drive"
            artifacts, logs = work / "artifacts", work / "logs"
            artifacts.mkdir(parents=True)
            logs.mkdir()
            (logs / "sample.log").write_text("diagnostic output")
            inputs = work / "validation_inputs"
            inputs.mkdir()
            (inputs / "brief_explanation.txt").write_text("official rendered template")
            (artifacts / prepare.MODEL_NAME).write_bytes(b"model-fixture")
            (artifacts / "python-requirements-resolved.txt").write_text("example==1")
            prepare.write_json(artifacts / "quality_report.json", {
                "status": "validated", "settings": prepare.settings(),
                "models": {"Q4_K_M": {"size_bytes": 13, "sha256": prepare.sha256(artifacts / prepare.MODEL_NAME)},
                           "BF16": {"sha256": "bf16-checkpoint"}}})
            with ExitStack() as stack:
                for name, value in {"WORK": work, "DRIVE": drive, "ARTIFACTS": artifacts, "LOGS": logs,
                                    "STATE": {"cache_hit": False, "conversion": {"bf16": {"sha256": "bf16-checkpoint"}}}}.items():
                    stack.enter_context(patch.object(prepare, name, value))
                prepare.stage8_publish()
                self.assertTrue(prepare.cache_valid(prepare.cache_dir())[0])
                self.assertEqual(prepare.read_json(artifacts / "COMPLETE.json")["manifest_sha256"],
                                 prepare.sha256(artifacts / "manifest.json"))
                prepare.stage3_cache()
                self.assertTrue(prepare.STATE["cache_hit"])
                self.assertTrue((artifacts / "COMPLETE.json").exists())

    def test_ppl_requires_full_run_and_finite_estimate(self):
        prefix = "calculating perplexity over 4 chunks, n_ctx=512, batch_size=512, n_seq=1\n"
        self.assertEqual(prepare.parse_ppl(prefix, "Final estimate: PPL = 5.25 +/- 0.2"), 5.25)
        for value in ["nan", "inf", "-1"]:
            with self.assertRaises(RuntimeError):
                prepare.parse_ppl(prefix, f"Final estimate: PPL = {value}")
        with self.assertRaises(RuntimeError):
            prepare.parse_ppl("", "Final estimate: PPL = 5.25")

    def test_empty_and_template_outputs_fail(self):
        self.assertEqual(prepare.extract_response("你好 [end of text]\n"), "你好")
        for value in ["", " [end of text]\n", "<think>reasoning", "<|im_start|>assistant"]:
            with self.assertRaises(RuntimeError):
                prepare.extract_response(value)

    def test_notebook_embeds_source_and_all_stages(self):
        nb = json.loads((MODULE.parents[1] / "notebooks/Qwen3_RK3588_GPU_Prepare.ipynb").read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertIn(prepare.MODEL_REVISION, code)
        self.assertIn("def stage8_publish", code)
        for number in range(1, 9):
            self.assertIn(f"run_stage({number})", code)
        self.assertNotIn("/Users/a/", code)

    def test_partial_venv_recovers_without_ensurepip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / ".venv"
            venv.EnvBuilder(with_pip=False, system_site_packages=False).create(root)
            self.assertIn("include-system-site-packages = false", (root / "pyvenv.cfg").read_text())
            target = root / "bin/python"
            result = subprocess.CompletedProcess([], 0, "pip inherited", "")
            with patch.object(prepare, "PYTHON", target), patch.object(prepare.venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip must not run")), patch.object(prepare.subprocess, "run", return_value=result):
                command = prepare.prepare_python_environment()
            self.assertEqual(command, [str(target), "-m", "pip"])
            self.assertIn("include-system-site-packages = true", (root / "pyvenv.cfg").read_text())

    def test_host_pip_can_manage_pipless_venv(self):
        failed = subprocess.CompletedProcess([], 1, "", "no pip")
        success = subprocess.CompletedProcess([], 0, "pip", "")
        with patch.object(prepare.venv, "EnvBuilder") as builder, patch.object(prepare.subprocess, "run", side_effect=[failed, success]):
            command = prepare.prepare_python_environment()
            builder.assert_called_once_with(with_pip=False, system_site_packages=True)
            self.assertIn("--python", command)

    def test_failed_stage_blocks_later_actions_and_can_retry(self):
        first = unittest.mock.Mock(side_effect=RuntimeError("missing dependency"))
        next_stage = unittest.mock.Mock()
        state = {"settings_fingerprint": prepare.fingerprint(), "completed_stages": [1]}
        with patch.object(prepare, "STATE", state), patch.object(prepare, "save_state"), patch.object(prepare, "STAGES", [lambda: None, first, next_stage] + [lambda: None] * 5):
            with self.assertRaisesRegex(RuntimeError, "missing dependency"):
                prepare.run_stage(2)
            self.assertEqual(state["failed_stage"], 2)
            with self.assertRaisesRegex(RuntimeError, "prerequisite"):
                prepare.run_stage(3)
            next_stage.assert_not_called()
            first.side_effect = None
            prepare.run_stage(2)
            prepare.run_stage(3)
            self.assertEqual(state["completed_stages"], [1, 2, 3])
            self.assertNotIn("failed_stage", state)


if __name__ == "__main__":
    unittest.main()
