#!/usr/bin/env python3
"""Focused checks for the resumable experiment-suite orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment_suite import (  # noqa: E402
    ExecutionResult,
    ExperimentSuite,
    FUTURE_RESIDENT_MODEL_HOOK,
    INDEX_FILENAME,
    SuiteConfigError,
    SuiteError,
    build_plan,
    derive_seed,
)


def _config(runner: Path, *, asset: Path | None = None) -> dict[str, object]:
    return {
        "schema_version": "digital-fly-suite-1",
        "name": "test-suite",
        "protocol_id": "test-protocol",
        "master_seed": 20260921,
        "runner": {
            "script": str(runner),
            "base_args": ["--duration", "0.001"],
            "assets": [] if asset is None else [str(asset)],
        },
        "resources": {
            "max_vram_gib": 2.0,
            "max_wall_time_s": 30,
            "max_output_bytes": 4096,
        },
        "conditions": [
            {"id": "loom", "family": "loom", "stimulus_name": "looming_grounded"},
            {"id": "static", "family": "static", "stimulus_name": "static_grounded"},
            {"id": "sham", "family": "sham", "stimulus_name": "sham_grounded"},
            {"id": "loom-ablation", "family": "ablation", "metadata": {"lesion": "loom-input"}},
        ],
    }


def _child_manifest(seed: int, renderer: str) -> dict[str, object]:
    return {
        "schema_version": "digital-fly-run-1",
        "seed": seed,
        "requested_renderer_backend": renderer,
        "effective_execution": {"renderer_backend": renderer},
    }


class ExperimentSuiteTest(unittest.TestCase):
    def test_condition_seed_is_stable_uint32_and_trial_specific(self) -> None:
        first = derive_seed(20260921, "opaque-loom", protocol_id="escape-e14-v1")
        second = derive_seed(20260921, "opaque-loom", protocol_id="escape-e14-v1")
        different_trial = derive_seed(20260921, "opaque-static", protocol_id="escape-e14-v1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_trial)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 0xFFFFFFFF)

    def test_plan_records_all_conditions_assets_and_future_resident_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "immutable.bin"
            asset.write_bytes(b"fixed-model-init")
            plan = build_plan(_config(ROOT / "scripts" / "run_digital_fly.py", asset=asset), out=root / "out")
            self.assertEqual([item.family for item in plan.conditions], ["loom", "static", "sham", "ablation"])
            self.assertEqual(len(plan.assets), 2)  # runner script + explicit immutable asset
            self.assertTrue(all(len(item["sha256"]) == 64 for item in plan.assets))
            self.assertTrue(all(condition.model_initialization_key for condition in plan.conditions))
            manifest = plan.to_dict()
            self.assertTrue(manifest["reuse"]["immutable_assets"]["possible"])
            self.assertFalse(manifest["reuse"]["model_initialization"]["possible"])
            self.assertIn(FUTURE_RESIDENT_MODEL_HOOK, manifest["reuse"]["model_initialization"]["future_hook"])

    def test_controlled_seed_and_output_arguments_fail_closed(self) -> None:
        config = _config(ROOT / "scripts" / "run_digital_fly.py")
        config["conditions"] = [{"id": "bad", "family": "loom", "runner_args": ["--seed", "7"]}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SuiteConfigError, "suite-owned option --seed"):
                build_plan(config, out=Path(tmp) / "out")

    def test_run_writes_index_and_resumes_only_a_valid_completed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "suite"
            plan = build_plan(_config(ROOT / "scripts" / "run_digital_fly.py"), out=out)
            calls: list[tuple[str, ...]] = []

            def fake_executor(command, cwd, caps, stdout_path, stderr_path):
                del cwd, caps
                calls.append(tuple(command))
                attempt_dir = Path(command[command.index("--out") + 1])
                seed = int(command[command.index("--seed") + 1])
                renderer = command[command.index("--renderer-backend") + 1]
                (attempt_dir / "manifest.json").write_text(
                    json.dumps(_child_manifest(seed, renderer)),
                    encoding="utf-8",
                )
                stdout_path.write_text('{"status":"OK"}\n', encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return ExecutionResult(0, 0.001, output_bytes=16)

            first = ExperimentSuite(plan, out, executor=fake_executor).run()
            self.assertEqual(first["status"], "complete")
            self.assertEqual(len(calls), 4)
            index_path = out / INDEX_FILENAME
            saved = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["plan_hash"], plan.plan_hash)
            self.assertTrue(all(item["status"] == "complete" for item in saved["conditions"]))

            resumed = ExperimentSuite(plan, out, executor=fake_executor).run()
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(len(calls), 4, "valid completed attempts must not be rerun")
            self.assertTrue(all(item["status"] == "skipped" for item in resumed["conditions"]))

    def test_failed_attempt_is_retained_and_retry_gets_a_new_attempt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "suite"
            config = _config(ROOT / "scripts" / "run_digital_fly.py")
            config["conditions"] = [{"id": "loom", "family": "loom"}]
            plan = build_plan(config, out=out)

            def fail_executor(command, cwd, caps, stdout_path, stderr_path):
                del command, cwd, caps
                stdout_path.write_text("failed\n", encoding="utf-8")
                stderr_path.write_text("boom\n", encoding="utf-8")
                return ExecutionResult(9, 0.001, output_bytes=10)

            first = ExperimentSuite(plan, out, executor=fail_executor).run()
            self.assertEqual(first["status"], "error")
            first_attempt = Path(first["conditions"][0]["latest_attempt"])

            def pass_executor(command, cwd, caps, stdout_path, stderr_path):
                del cwd, caps
                attempt_dir = Path(command[command.index("--out") + 1])
                seed = int(command[command.index("--seed") + 1])
                (attempt_dir / "manifest.json").write_text(
                    json.dumps(_child_manifest(seed, "analytic_reference")),
                    encoding="utf-8",
                )
                stdout_path.write_text("ok\n", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return ExecutionResult(0, 0.001, output_bytes=3)

            second = ExperimentSuite(plan, out, executor=pass_executor).run()
            self.assertEqual(second["status"], "complete")
            attempts = second["conditions"][0]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertNotEqual(first_attempt, Path(attempts[-1]["attempt_dir"]))
            self.assertEqual(attempts[0]["status"], "error")
            self.assertEqual(attempts[1]["status"], "complete")

    def test_vram_argument_cannot_exceed_suite_cap(self) -> None:
        config = _config(ROOT / "scripts" / "run_digital_fly.py")
        config["runner"] = {
            "script": str(ROOT / "scripts" / "run_digital_fly.py"),
            "base_args": ["--max-vram-gib", "3"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SuiteConfigError, "exceeds suite resource cap"):
                build_plan(config, out=Path(tmp) / "out")

    def test_renderer_backend_is_global_explicit_and_part_of_model_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            analytic = build_plan(_config(ROOT / "scripts" / "run_digital_fly.py"), out=out)
            warp = build_plan(
                _config(ROOT / "scripts" / "run_digital_fly.py"),
                out=out,
                renderer_backend="warp_cuda",
            )
            self.assertEqual(analytic.requested_renderer_backend, "analytic_reference")
            self.assertEqual(analytic.effective_renderer_backend, "analytic_reference")
            self.assertTrue(analytic.renderer_provenance["fail_closed"])
            self.assertEqual(warp.requested_renderer_backend, "warp_cuda")
            self.assertEqual(warp.effective_renderer_backend, "warp_cuda")
            self.assertNotEqual(analytic.model_initialization_key, warp.model_initialization_key)
            self.assertNotEqual(analytic.plan_hash, warp.plan_hash)

    def test_warp_backend_is_passed_to_launcher_and_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "suite"
            plan = build_plan(
                _config(ROOT / "scripts" / "run_digital_fly.py"),
                out=out,
                renderer_backend="warp_cuda",
            )
            commands: list[tuple[str, ...]] = []

            def fake_executor(command, cwd, caps, stdout_path, stderr_path):
                del cwd, caps
                commands.append(tuple(command))
                attempt_dir = Path(command[command.index("--out") + 1])
                seed = int(command[command.index("--seed") + 1])
                renderer = command[command.index("--renderer-backend") + 1]
                (attempt_dir / "manifest.json").write_text(
                    json.dumps(_child_manifest(seed, renderer)), encoding="utf-8",
                )
                stdout_path.write_text("ok\n", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return ExecutionResult(0, 0.001, output_bytes=3)

            result = ExperimentSuite(plan, out, executor=fake_executor).run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(commands[0][commands[0].index("--renderer-backend") + 1], "warp_cuda")
            self.assertEqual(result["requested_renderer_backend"], "warp_cuda")
            self.assertEqual(result["effective_renderer_backend"], "warp_cuda")

    def test_warp_without_launcher_interface_fails_in_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "opaque_runner.py"
            runner.write_text("print('no renderer interface')\n", encoding="utf-8")
            config = _config(runner)
            with self.assertRaisesRegex(SuiteConfigError, "refusing analytic fallback"):
                build_plan(config, out=root / "out", renderer_backend="warp_cuda")

    def test_warp_child_reporting_analytic_is_an_error_not_a_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "suite"
            plan = build_plan(
                _config(ROOT / "scripts" / "run_digital_fly.py"),
                out=out,
                renderer_backend="warp_cuda",
            )

            def fake_executor(command, cwd, caps, stdout_path, stderr_path):
                del cwd, caps
                attempt_dir = Path(command[command.index("--out") + 1])
                seed = int(command[command.index("--seed") + 1])
                child = _child_manifest(seed, "warp_cuda")
                child["effective_execution"] = {"renderer_backend": "analytic_reference"}
                (attempt_dir / "manifest.json").write_text(
                    json.dumps(child), encoding="utf-8",
                )
                stdout_path.write_text("ok\n", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return ExecutionResult(0, 0.001, output_bytes=3)

            result = ExperimentSuite(plan, out, executor=fake_executor).run()
            self.assertEqual(result["status"], "error")
            self.assertIn("effective renderer mismatch", result["conditions"][0]["error"])

    def test_renderer_tampering_invalidates_resume_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "suite"
            plan = build_plan(_config(ROOT / "scripts" / "run_digital_fly.py"), out=out)
            calls = 0

            def fake_executor(command, cwd, caps, stdout_path, stderr_path):
                nonlocal calls
                del cwd, caps
                calls += 1
                attempt_dir = Path(command[command.index("--out") + 1])
                seed = int(command[command.index("--seed") + 1])
                renderer = command[command.index("--renderer-backend") + 1]
                (attempt_dir / "manifest.json").write_text(
                    json.dumps(_child_manifest(seed, renderer)), encoding="utf-8",
                )
                stdout_path.write_text("ok\n", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return ExecutionResult(0, 0.001, output_bytes=3)

            ExperimentSuite(plan, out, executor=fake_executor).run()
            index_path = out / INDEX_FILENAME
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["conditions"][0]["attempts"][-1]["effective_renderer_backend"] = "warp_cuda"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            resumed = ExperimentSuite(plan, out, executor=fake_executor).run()
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(calls, 5)
            self.assertEqual(len(resumed["conditions"][0]["attempts"]), 2)

    def test_asset_mutation_after_plan_creation_fails_before_child_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "immutable.bin"
            asset.write_bytes(b"before")
            plan = build_plan(_config(ROOT / "scripts" / "run_digital_fly.py", asset=asset), out=root / "out")
            asset.write_bytes(b"after")
            with self.assertRaisesRegex(SuiteError, "immutable asset changed"):
                # The verification happens before the executor is entered.
                ExperimentSuite(plan, root / "out").run()


if __name__ == "__main__":
    unittest.main()
