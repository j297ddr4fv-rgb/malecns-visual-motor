import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_flygym_free_body_online_lowcontrast_v2 as runner


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.pid = 1
        self.code = None

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.code = 0
        return 0


class ResidentWorkerClientTests(unittest.TestCase):
    def test_reuses_process_and_rebinds_schedule_without_model_replica(self):
        worker = FakeProcess()
        pool = runner.ResidentNeuralWorker()
        command = ["python", "bootstrap_retina_extension.py", "worker.py",
                   "schedule-a", "hash-a", "case-a", "--config", "config-a"]
        with tempfile.TemporaryDirectory() as raw, patch.object(runner.subprocess, "Popen", return_value=worker) as spawn:
            root = Path(raw)
            self.assertIs(pool.start_case(command, {}, root / "first.log"), worker)
            second = command.copy()
            second[3:6] = ["schedule-b", "hash-b", "case-b"]
            self.assertIs(pool.start_case(second, {}, root / "second.log"), worker)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(spawn.call_args.args[0][-1], "--resident")
            request = json.loads(worker.stdin.getvalue())
            self.assertEqual(request["type"], "reset_case")
            self.assertEqual(request["argv"], ["--config", "config-a"])
            self.assertEqual(request["adapter_env"]["FLYGYM_EXTENSION_CASE_ID"], "case-b")

    def test_rejects_bootstrap_change(self):
        pool = runner.ResidentNeuralWorker()
        command = ["python", "bootstrap_retina_extension.py", "worker.py", "a", "b", "c", "--config", "cfg"]
        with tempfile.TemporaryDirectory() as raw, patch.object(runner.subprocess, "Popen", return_value=FakeProcess()):
            pool.start_case(command, {}, Path(raw) / "first.log")
            command[1] = "other_bootstrap.py"
            with self.assertRaises(runner.RunnerError):
                pool.start_case(command, {}, Path(raw) / "second.log")

    def test_close_uses_shutdown_and_reaps_process(self):
        pool = runner.ResidentNeuralWorker()
        worker = FakeProcess()
        pool.process = worker
        with patch.object(runner, "_protocol_read", return_value=('{"type":"shutdown"}', {"type": "shutdown"})):
            pool.close()
        self.assertEqual(worker.code, 0)
        self.assertIsNone(pool.process)


if __name__ == "__main__":
    unittest.main()
