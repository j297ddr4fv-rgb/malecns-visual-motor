import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flygym_online_neural_worker_eye_parity_state_v2 as worker_module


class ResidentWorkerHelpersTests(unittest.TestCase):
    def test_parse_cli_resident_is_opt_in(self):
        cold = worker_module._parse_cli(["--stimulus-condition", "left", "--out", "/data/artifacts/by-project/malecns-gpu/parse-only"])
        resident = worker_module._parse_cli(["--stimulus-condition", "left", "--out", "/data/artifacts/by-project/malecns-gpu/parse-only", "--resident"])
        self.assertFalse(cold.resident)
        self.assertTrue(resident.resident)

    def test_reset_case_accepts_one_complete_adapter_family(self):
        with tempfile.TemporaryDirectory() as raw:
            schedule = Path(raw) / "schedule.json"
            schedule.write_text("{}", encoding="utf-8")
            sha256 = worker_module._sha256_file(schedule)
            argv, adapter_env = worker_module._resident_reset_fields({
                "argv": ["--stimulus-condition", "left"],
                "adapter_env": {
                    "FLYGYM_EXTENSION_SCHEDULE_PATH": str(schedule),
                    "FLYGYM_EXTENSION_SCHEDULE_SHA256": sha256,
                    "FLYGYM_EXTENSION_CASE_ID": "case-2",
                },
            })
        self.assertEqual(argv, ["--stimulus-condition", "left"])
        self.assertEqual(adapter_env["FLYGYM_EXTENSION_CASE_ID"], "case-2")
        self.assertEqual(adapter_env["FLYGYM_EXTENSION_SCHEDULE_SHA256"], sha256)

    def test_reset_case_rejects_partial_or_unapproved_adapter_env(self):
        with self.assertRaises(worker_module.WorkerError):
            worker_module._resident_reset_fields({
                "argv": [],
                "adapter_env": {"FLYGYM_HIGHCONTRAST_CASE_ID": "case"},
            })
        with self.assertRaises(worker_module.WorkerError):
            worker_module._resident_reset_fields({
                "argv": [],
                "adapter_env": {"PATH": "/tmp/override"},
            })

    def test_angular_stimulus_can_change_without_changing_model_identity(self):
        first = {"worker_config": {"sha256": "config"}, "angular_stimulus_spec": {"sha256": "stimulus-a"}}
        second = {"worker_config": {"sha256": "config"}, "angular_stimulus_spec": {"sha256": "stimulus-b"}}
        self.assertEqual(worker_module._resident_asset_identity(first), worker_module._resident_asset_identity(second))
        second["worker_config"] = {"sha256": "other-model-config"}
        self.assertNotEqual(worker_module._resident_asset_identity(first), worker_module._resident_asset_identity(second))

    def test_clear_flyvis_case_timing_resets_totals_and_last_frame(self):
        class FakeAdapter:
            _timing_totals = {"frames": 4, "step_s": 0.5}
            _last_frame_timing = {"step_s": 0.1}

        adapter = FakeAdapter()
        worker_module._clear_flyvis_case_timing(adapter)
        self.assertEqual(adapter._timing_totals, {"frames": 0, "step_s": 0.0})
        self.assertIsNone(adapter._last_frame_timing)


class ResidentWorkerMainTests(unittest.TestCase):
    def _run_main(self, incoming: str):
        created = []

        class FakeWorker:
            def __init__(self, args, reuse_from=None):
                self._state = "running"
                self.capture_mode = "light"
                self.capture_flags = {}
                self.last_seq = None
                self.last_time_s = None
                self.reuse_from = reuse_from
                created.append(self)

            def ready_record(self):
                return {"protocol": worker_module.PROTOCOL, "type": "ready"}

            def process_frame(self, request):
                if self._state != "running":
                    raise AssertionError("stopped fake worker received a frame")
                return {"protocol": worker_module.PROTOCOL, "type": "frame_result"}

            def stop(self):
                self._state = "stopped"
                return {"protocol": worker_module.PROTOCOL, "type": "stopped"}

            def fail(self, exc):
                self._state = "failed"
                return {"protocol": worker_module.PROTOCOL, "type": "error", "error": str(exc)}

        first_args = type("Args", (), {"resident": True})()

        def parse_args(argv=None):
            return first_args if argv is None else type("Args", (), {"resident": False})()

        protocol_output = io.StringIO()
        with patch.object(worker_module, "_parse_cli", side_effect=parse_args), \
                patch.object(worker_module, "OnlineNeuralWorker", FakeWorker), \
                patch.object(worker_module.sys, "stdin", io.StringIO(incoming)), \
                patch.object(worker_module.sys, "stdout", protocol_output), \
                patch.dict(worker_module.os.environ, {}, clear=False), \
                patch.object(worker_module.sys, "stderr", io.StringIO()):
            result = worker_module.main()
        records = [json.loads(line) for line in protocol_output.getvalue().splitlines()]
        return result, records, created

    def test_resident_stop_reset_and_shutdown_reuses_stopped_worker(self):
        incoming = "\n".join((
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "stop"}),
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "reset_case",
                        "argv": ["--stimulus-condition", "right"], "adapter_env": {}}),
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "stop"}),
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "shutdown"}),
            "",
        ))
        result, records, created = self._run_main(incoming)
        self.assertEqual(result, 0)
        self.assertEqual([record["type"] for record in records],
                         ["ready", "stopped", "ready", "stopped", "shutdown"])
        self.assertEqual(len(created), 2)
        self.assertIs(created[1].reuse_from, created[0])
        self.assertEqual(created[0]._state, "stopped")

    def test_resident_rejects_frame_after_stop_without_invoking_worker(self):
        incoming = "\n".join((
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "stop"}),
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "frame"}),
            json.dumps({"protocol": worker_module.PROTOCOL, "type": "shutdown"}),
            "",
        ))
        result, records, created = self._run_main(incoming)
        self.assertEqual(result, 0)
        self.assertEqual([record["type"] for record in records],
                         ["ready", "stopped", "error", "shutdown"])
        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
