from __future__ import annotations

import hashlib
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_leg_muscle_backends import ComparisonError, compare_runs, main


CSR_FILES = (
    "ids.npy", "offsets.npy", "target.npy", "weight.npy", "sign.npy",
    "annotations.npz", "tables.json", "meta.json",
)
DT_S = 0.0005
TICKS = 3
NEURONS = 4


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _body_state(time_s: float, *, q_rad: float = 0.1, x_m: float = 0.0,
                muscle_force_N: float = 1e-6) -> dict:
    return {
        "root": {
            "position_m": [x_m, 0.0, 0.001],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_mps": [0.0, 0.0, 0.0],
        },
        "joints": {
            "LF_tibia_pitch": {"q_rad": q_rad, "qd_rad_s": 0.0, "torque_Nm": 0.0},
        },
        "feet": {
            "LF": {
                "position_m": [0.001, 0.0, 0.0001],
                "contact": True,
                "normal_force_N": 1e-6,
                "tangential_force_N": [0.0, 0.0, 0.0],
                "contact_force_N": [0.0, 0.0, 1e-6],
            },
        },
        "body_contacts": [],
        "body_ground_contact": False,
        "center_of_mass_position_m": [x_m, 0.0, 0.001],
        "energy": {"kinetic_J": 0.0, "potential_J": 1e-8, "total_J": 1e-8},
        "time_s": time_s,
        "muscles": {"LF_tibia_extensor": {
            "activation": 0.2,
            "excitation": 0.3,
            "length_m": 0.00018,
            "velocity_mps": 0.0,
            "active_force_N": muscle_force_N,
            "passive_force_N": 0.0,
            "total_force_N": muscle_force_N,
            "power_W": 0.0,
        }},
    }


def _snapshot(time_s: float, *, q_rad: float = 0.1, x_m: float = 0.0,
              muscle_force_N: float = 1e-6) -> dict:
    return {
        "schema_version": 1,
        "mjcf_sha256": "a" * 64,
        "time_s": time_s,
        "qpos": [0.0, 0.0, 0.001, 1.0, 0.0, 0.0, 0.0, q_rad],
        "qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "qacc_warmstart": [0.0] * 7,
        "joint_rest_q_rad": {"LF_tibia_pitch": 0.1},
        "telemetry": _body_state(time_s, q_rad=q_rad, x_m=x_m,
                                 muscle_force_N=muscle_force_N),
    }


def _write_run(directory: Path, *, candidate: bool) -> None:
    directory.mkdir()
    source_bytes = b"same executed source snapshot\n"
    (directory / "sources").mkdir()
    (directory / "sources" / "runner.py").write_bytes(source_bytes)
    source_hashes = {"runner.py": _sha(source_bytes)}
    csr_hashes = {name: _sha(("csr:" + name).encode()) for name in CSR_FILES}
    backend_request = "torch_cuda" if candidate else "cpu"
    backend_actual = "torch_cuda" if candidate else "numpy_cpu"
    device = "cuda" if candidate else "cpu"
    engine_request = "torch" if candidate else "cpu"
    dense = bytes([0] * (TICKS * NEURONS))
    dense_sha = _sha(dense)

    count_rows = [
        {"tick": tick, "t_s": tick * DT_S, "total_spikes": 0, "target_spikes": 0}
        for tick in range(TICKS)
    ]
    counts_raw = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                         for row in count_rows).encode()
    (directory / "neural_spike_counts.jsonl").write_bytes(counts_raw)

    telemetry = []
    for tick in (0, TICKS - 1):
        time_s = (tick + 1) * DT_S
        telemetry.append({
            "time_s": time_s,
            "muscles": {"LF_tibia_extensor": {
                "activation": 0.2, "excitation": 0.3, "length_m": 0.00018,
                "velocity_mps": 0.0, "active_force_N": 1e-6,
                "passive_force_N": 0.0, "total_force_N": 1e-6, "power_W": 0.0,
            }},
            "body": _body_state(time_s),
            "neural": {"source": "malecns", "intervention_active": False,
                       "total_spikes": 0, "target_spikes": 0, "tick": tick},
        })
    telemetry_raw = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                            for row in telemetry).encode()
    (directory / "telemetry.jsonl").write_bytes(telemetry_raw)

    initial = _snapshot(0.0)
    final = _snapshot(TICKS * DT_S)
    summary = {
        "case": "sham", "leg": "LF", "source": "malecns",
        "evidence_class": "digital_simulation", "biological_validation": "NOT_VALIDATED",
        "autonomous_behavior": False, "control_authority": "sham_no_external_neural_drive",
        "backend": f"{backend_actual}+mujoco_cpu", "physics_backend": "mujoco_cpu",
        "neural_backend_requested": backend_request,
        "neural_backend_actual": backend_actual, "neural_device_actual": device,
        "seed": 7, "max_vram_gib": 15.5,
        "neural_observation_dtype": "uint8", "neural_spike_encoding": "tick-major dense uint8 C-order",
        "neural_spike_shape": [TICKS, NEURONS], "full_neural_trace_ticks": TICKS,
        "neural_spike_sha256": dense_sha, "gpu_used": candidate,
        "fixed_root": False, "duration_s": TICKS * DT_S, "neural_dt_s": DT_S,
        "wall_s": 2.0 if candidate else 1.0,
        "selector": "synthetic selector", "selected_neuron_count": 1,
        "muscle_group": "tibia_extensor", "selected_channel": "LF_tibia_extensor",
        "selected_body_ids": [1001], "intervention_window_s": [0.0, 0.0],
        "effective_intervention_window_s": None, "current_mV_per_ms": 2.0,
        "prescribed_rate_hz": None, "total_spikes": 0, "target_spikes": 0,
        "maximum_selected_active_muscle_force_N": 1e-6,
        "selected_joint": "LF_tibia_pitch", "selected_joint_delta_rad": 0.0,
        "root_displacement_m": [0.0, 0.0, 0.0],
        "center_of_mass_displacement_m": [0.0, 0.0, 0.0],
        "all_ground_contact_impulse_Ns": [0.0, 0.0, 0.0],
        "contact_impulse_sampling_dt_s": DT_S,
        "nonfoot_ground_contact_fraction": 0.0,
        "peak_foot_tangential_force_N": 0.0,
        "summed_foot_tangential_impulse_Ns": [0.0, 0.0, 0.0],
        "summed_foot_normal_impulse_Ns": 0.0,
        "initial": initial, "final": final,
        "output_name": "candidate-label" if candidate else "reference-label",
    }
    neural = {
        "schema_version": 1, "generated_at": "different per run",
        "backend_requested": engine_request, "backend": backend_actual,
        "device_requested": device, "device": device, "max_vram_gib": 15.5,
        "dt_ms": 0.5, "gain": 1.0, "kc_gain": 1.0, "seed": 7,
        "n": NEURONS, "edges": 4, "delay_steps": 1, "ref_steps": 2,
        "include_flyvis_drive_echo": False, "capabilities": {"cpu": not candidate},
        "triton_timing": {},
        "csr": {"path": f"/different/path/{'candidate' if candidate else 'reference'}",
                "exists": True, "valid": True, "blockers": [], "neurons": NEURONS,
                "edges": 4, "synapses": 4,
                "shapes": {"ids": [NEURONS], "offsets": [NEURONS + 1]},
                "dtypes": {"ids": "int64", "offsets": "int64"},
                "metadata": {"meta": {"dataset": "synthetic-v1"}}},
    }
    pipeline = {
        "schema": "articulated-neural-muscle-pipeline-1",
        "evidence_class": "digital_simulation", "biological_validation": "NOT_VALIDATED",
        "synthetic_gait": False, "body_traction_command": False,
        "rate_filter_tau_s": 0.02,
        "body": {"fixed_root": False, "config_sha256": "b" * 64,
                 "mjcf_sha256": "a" * 64, "muscle_channel_count": 1,
                 "joint_names": ["LF_tibia_pitch"]},
    }
    manifest = {
        "pipeline": pipeline, "neural": neural,
        "summary": {key: value for key, value in summary.items() if key not in ("initial", "final")},
        "source_sha256": source_hashes, "source_snapshot_directory": "sources",
        "csr_sha256": csr_hashes,
        "neural_backend_preflight": {
            "requested_backend": backend_request, "requires_cuda": candidate,
            "gate_checked": candidate, "gate_passed": True, "blockers": [],
        },
        "telemetry_sha256": _sha(telemetry_raw),
        "neural_counts_sha256": _sha(counts_raw),
    }
    _write_json(directory / "summary.json", summary)
    _write_json(directory / "manifest.json", manifest)


class CompareLegMuscleBackendsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "cpu-reference-label-a"
        self.candidate = self.root / "cuda-candidate-label-b"
        _write_run(self.reference, candidate=False)
        _write_run(self.candidate, candidate=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _rewrite_candidate(self, name: str, data: object) -> None:
        _write_json(self.candidate / name, data)

    def _read_json(self, directory: Path, name: str) -> dict:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    def _refresh_telemetry_digest(self) -> None:
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["telemetry_sha256"] = _sha((self.candidate / "telemetry.jsonl").read_bytes())
        self._rewrite_candidate("manifest.json", manifest)

    def test_matched_synthetic_runs_pass_and_inputs_remain_read_only(self) -> None:
        before = {
            path: _sha(path.read_bytes())
            for directory in (self.reference, self.candidate)
            for path in directory.rglob("*") if path.is_file()
        }
        report = compare_runs(self.reference, self.candidate)
        after = {
            path: _sha(path.read_bytes())
            for directory in (self.reference, self.candidate)
            for path in directory.rglob("*") if path.is_file()
        }
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["neural_spike_trace"]["exact_hash_match"])
        self.assertFalse(report["performance_equivalence"])
        self.assertEqual(report["biological_validation"], "NOT_VALIDATED")
        self.assertEqual(before, after)
        self.assertEqual(report["trajectory_comparison"]["telemetry_samples"], 2)
        errors = report["trajectory_comparison"]["max_absolute_error_by_category"]
        for category in ("joints", "muscles", "root", "contact"):
            self.assertGreater(errors[category]["numeric_leaves"], 0)

    def test_neural_hash_mismatch_is_a_numerical_failure_report(self) -> None:
        summary = self._read_json(self.candidate, "summary.json")
        manifest = self._read_json(self.candidate, "manifest.json")
        summary["neural_spike_sha256"] = "f" * 64
        manifest["summary"]["neural_spike_sha256"] = "f" * 64
        self._rewrite_candidate("summary.json", summary)
        self._rewrite_candidate("manifest.json", manifest)

        report = compare_runs(self.reference, self.candidate)
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse(report["neural_spike_trace"]["exact_hash_match"])
        self.assertEqual(report["trajectory_comparison"]["numerical_failures"][0]["category"], "neural")

    def test_trajectory_tamper_is_reported_with_category_error(self) -> None:
        path = self.candidate / "telemetry.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["muscles"]["LF_tibia_extensor"]["active_force_N"] += 0.01
        rows[-1]["body"]["joints"]["LF_tibia_pitch"]["q_rad"] += 1e-4
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        self._refresh_telemetry_digest()

        report = compare_runs(self.reference, self.candidate)
        self.assertEqual(report["status"], "FAILED")
        errors = report["trajectory_comparison"]["max_absolute_error_by_category"]
        self.assertFalse(errors["muscles"]["within_tolerance"])
        self.assertFalse(errors["joints"]["within_tolerance"])

    def test_seed_and_gain_mismatches_reject_the_pair(self) -> None:
        summary = self._read_json(self.candidate, "summary.json")
        manifest = self._read_json(self.candidate, "manifest.json")
        summary["seed"] = 9
        manifest["summary"]["seed"] = 9
        manifest["neural"]["seed"] = 9
        manifest["neural"]["gain"] = 2.0
        self._rewrite_candidate("summary.json", summary)
        self._rewrite_candidate("manifest.json", manifest)
        with self.assertRaisesRegex(ComparisonError, "integration controls differ|control 'seed'"):
            compare_runs(self.reference, self.candidate)

    def test_dt_mismatch_rejects_the_pair(self) -> None:
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["neural"]["dt_ms"] = 0.25
        self._rewrite_candidate("manifest.json", manifest)
        with self.assertRaisesRegex(ComparisonError, "dt_s disagrees"):
            compare_runs(self.reference, self.candidate)

    def test_different_source_snapshots_are_rejected(self) -> None:
        manifest = self._read_json(self.candidate, "manifest.json")
        changed_source = b"changed source snapshot\n"
        (self.candidate / "sources" / "runner.py").write_bytes(changed_source)
        manifest["source_sha256"]["runner.py"] = _sha(changed_source)
        self._rewrite_candidate("manifest.json", manifest)
        with self.assertRaisesRegex(ComparisonError, "source_sha256 maps differ"):
            compare_runs(self.reference, self.candidate)

    def test_candidate_that_reports_cpu_is_rejected(self) -> None:
        summary = self._read_json(self.candidate, "summary.json")
        summary.update(neural_backend_requested="cpu", neural_backend_actual="numpy_cpu",
                       neural_device_actual="cpu", backend="numpy_cpu+mujoco_cpu", gpu_used=False)
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["summary"].update(neural_backend_requested="cpu", neural_backend_actual="numpy_cpu",
                                   neural_device_actual="cpu", backend="numpy_cpu+mujoco_cpu", gpu_used=False)
        manifest["neural"].update(backend_requested="cpu", backend="numpy_cpu",
                                   device_requested="cpu", device="cpu")
        manifest["neural_backend_preflight"] = {
            "requested_backend": "cpu", "requires_cuda": False,
            "gate_checked": False, "gate_passed": True, "blockers": [],
        }
        self._rewrite_candidate("summary.json", summary)
        self._rewrite_candidate("manifest.json", manifest)
        with self.assertRaisesRegex(ComparisonError, "candidate requested backend"):
            compare_runs(self.reference, self.candidate)

    def test_corrupt_telemetry_digest_is_a_clear_error(self) -> None:
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["telemetry_sha256"] = "0" * 64
        self._rewrite_candidate("manifest.json", manifest)
        with self.assertRaisesRegex(ComparisonError, "telemetry.jsonl SHA-256"):
            compare_runs(self.reference, self.candidate)

    def test_full_feedback_stream_checks_hashes_and_causal_alignment(self) -> None:
        for directory in (self.reference, self.candidate):
            counts_path = directory/'neural_spike_counts.jsonl'
            counts = [json.loads(line) for line in counts_path.read_text().splitlines()]
            for row in counts:
                row['afferent_spikes'] = 0
            counts_path.write_text(''.join(json.dumps(row)+'\n' for row in counts))
            summary = self._read_json(directory, 'summary.json')
            summary.update(sensory_feedback_mode='on', afferent_spikes=0)
            manifest = self._read_json(directory, 'manifest.json')
            manifest['summary'].update(sensory_feedback_mode='on', afferent_spikes=0)
            manifest['neural_counts_sha256'] = _sha(counts_path.read_bytes())
            rows = [{'tick': tick, 'source_time_s': tick*DT_S, 'applied_at_s': tick*DT_S,
                     'afferent_spikes': 0, 'currents_mV_per_ms': {'fixture_sensory': .1}}
                    for tick in range(TICKS)]
            feedback_path = directory/'sensory_feedback.jsonl'
            feedback_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            manifest['sensory_feedback_sha256'] = _sha(feedback_path.read_bytes())
            _write_json(directory/'summary.json', summary)
            _write_json(directory/'manifest.json', manifest)
        self.assertEqual(compare_runs(self.reference, self.candidate)['status'], 'PASS')
        feedback_path = self.candidate/'sensory_feedback.jsonl'
        rows = [json.loads(line) for line in feedback_path.read_text().splitlines()]
        rows[-1]['source_time_s'] += DT_S
        feedback_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(ComparisonError, 'SHA-256'):
            compare_runs(self.reference, self.candidate)
        manifest = self._read_json(self.candidate, 'manifest.json')
        manifest['sensory_feedback_sha256'] = _sha(feedback_path.read_bytes())
        self._rewrite_candidate('manifest.json', manifest)
        with self.assertRaisesRegex(ComparisonError, 'causality/time alignment'):
            compare_runs(self.reference, self.candidate)

    def test_cli_writes_report_exclusively_outside_inputs(self) -> None:
        report_path = self.root / "comparison.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.reference), str(self.candidate), "--out", str(report_path)]), 0)
        original = report_path.read_bytes()
        self.assertEqual(json.loads(original)["status"], "PASS")
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            self.assertEqual(main([str(self.reference), str(self.candidate), "--out", str(report_path)]), 2)
        self.assertIn("refusing to overwrite", stderr.getvalue())
        self.assertEqual(report_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
