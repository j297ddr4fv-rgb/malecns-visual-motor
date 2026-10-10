#!/usr/bin/env python3
"""CPU-only tests for the flight artifact finalizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from finalize_flight_artifact import (  # noqa: E402
    FinalizationError,
    finalize_flight_artifact,
)


class FinalizeFlightArtifactTest(unittest.TestCase):
    def _bundle(self, directory: Path) -> dict[str, bytes]:
        rows = [
            {"time_s": 0.0, "phase": "takeoff", "position_m": [0.0, 0.0, 0.0]},
            {"time_s": 0.1, "phase": "sustained", "position_m": [0.0, 0.0, 0.4]},
        ]
        manifest = {
            "schema_version": "digital-fly-run-1",
            "seed": 7,
            "provenance": {"evidence_class": "digital_simulation"},
            "outputs": {
                "manifest": "manifest.json",
                "trajectory": "trajectory.json",
                "telemetry": "telemetry.jsonl",
                "validation": "validation.json",
                "video": "flight.mp4",
            },
            "simulation_duration_s": 0.1,
            "playback": {"playback_fps": 30.0},
        }
        trajectory = {
            "schema_version": "digital-fly-trajectory-1",
            "metadata": {
                "evidence_class": "digital_simulation",
                "manifest_path": "manifest.json",
            },
            "trajectory": rows,
        }
        validation = {
            "schema_version": "flight-validation-1",
            "decision": "NOT_VALIDATED",
            "evidence": {"class": "digital_simulation", "source": "trajectory.json"},
        }
        payloads = {
            "manifest.json": json.dumps(manifest, sort_keys=True).encode("utf-8"),
            "trajectory.json": json.dumps(trajectory, sort_keys=True).encode("utf-8"),
            "telemetry.jsonl": b"\n".join(json.dumps(row, sort_keys=True).encode("utf-8") for row in rows) + b"\n",
            "validation.json": json.dumps(validation, sort_keys=True).encode("utf-8"),
            "flight.mp4": b"tiny test video bytes",
        }
        for name, content in payloads.items():
            (directory / name).write_bytes(content)
        return payloads

    def _ffprobe_result(self, video_size: int) -> subprocess.CompletedProcess[str]:
        payload = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 160,
                    "height": 96,
                    "r_frame_rate": "30/1",
                    "avg_frame_rate": "30/1",
                    "duration": "0.100000",
                    "nb_frames": "3",
                }
            ],
            "format": {"duration": "0.100000", "size": str(video_size)},
        }
        return subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=json.dumps(payload), stderr=""
        )

    def test_finalizes_valid_bundle_with_probe_mapping_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payloads = self._bundle(directory)
            with mock.patch(
                "finalize_flight_artifact.subprocess.run",
                return_value=self._ffprobe_result(len(payloads["flight.mp4"])),
            ) as run:
                integrity = finalize_flight_artifact(directory)

            self.assertTrue((directory / "video_probe.json").is_file())
            self.assertTrue((directory / "integrity.json").is_file())
            command = run.call_args.args[0]
            self.assertIn("-count_frames", command)
            self.assertIn("-show_entries", command)

            probe = json.loads((directory / "video_probe.json").read_text(encoding="utf-8"))
            self.assertEqual(probe["streams"][0]["codec_name"], "h264")
            self.assertEqual(probe["width"], 160)
            self.assertEqual(probe["height"], 96)
            self.assertEqual(probe["fps"], 30.0)
            self.assertEqual(probe["duration_s"], 0.1)
            self.assertEqual(probe["frame_count"], 3)
            self.assertEqual(probe["mapping"]["simulation"]["duration_s"], 0.1)
            self.assertEqual(probe["mapping"]["playback"]["fps"], 30.0)
            self.assertEqual(probe["mapping"]["frame_to_trajectory"]["kind"], "time_resampled")

            self.assertEqual(integrity["files"]["flight.mp4"]["size_bytes"], len(payloads["flight.mp4"]))
            expected_hash = hashlib.sha256(payloads["trajectory.json"]).hexdigest()
            self.assertEqual(integrity["hashes"]["trajectory.json"], expected_hash)
            self.assertEqual(integrity["artifacts"]["manifest"]["path"], "manifest.json")
            self.assertEqual(integrity["sizes"]["validation.json"], len(payloads["validation.json"]))
            self.assertEqual(integrity["provenance"]["evidence_class"], "digital_simulation")
            self.assertEqual(
                integrity["provenance"]["component_labels"]["evidence_class"],
                "digital_simulation",
            )
            self.assertFalse(integrity["resources"]["declared"])
            self.assertEqual(integrity["evidence"]["class"], "digital_simulation")
            self.assertFalse(integrity["evidence"]["real_biological"])
            self.assertEqual(
                integrity["evidence"]["claim_boundary"],
                "digital_simulation; not biological validation",
            )

    def test_failed_validation_does_not_create_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payloads = self._bundle(directory)
            (directory / "telemetry.jsonl").write_bytes(payloads["telemetry.jsonl"] + b"{}\n")
            with self.assertRaisesRegex(FinalizationError, "row count mismatch"):
                finalize_flight_artifact(directory)
            self.assertFalse((directory / "video_probe.json").exists())
            self.assertFalse((directory / "integrity.json").exists())

    def test_missing_or_empty_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._bundle(directory)
            (directory / "validation.json").write_bytes(b"")
            with self.assertRaisesRegex(FinalizationError, "validation file is empty"):
                finalize_flight_artifact(directory)
            self.assertFalse((directory / "video_probe.json").exists())

    def test_bad_ffprobe_and_size_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payloads = self._bundle(directory)
            bad_probe = subprocess.CompletedProcess(
                args=["ffprobe"], returncode=0,
                stdout=json.dumps({"streams": [], "format": {}}), stderr=""
            )
            with mock.patch("finalize_flight_artifact.subprocess.run", return_value=bad_probe):
                with self.assertRaisesRegex(FinalizationError, "no video streams"):
                    finalize_flight_artifact(directory)
            self.assertFalse((directory / "video_probe.json").exists())

            mismatched_size = self._ffprobe_result(len(payloads["flight.mp4"]) + 1)
            with mock.patch("finalize_flight_artifact.subprocess.run", return_value=mismatched_size):
                with self.assertRaisesRegex(FinalizationError, "format size"):
                    finalize_flight_artifact(directory)
            self.assertFalse((directory / "integrity.json").exists())

    def test_manifest_reference_and_evidence_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payloads = self._bundle(directory)
            manifest = json.loads(payloads["manifest.json"])
            manifest["outputs"]["trajectory"] = "other-run/trajectory.json"
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "finalize_flight_artifact.subprocess.run",
                return_value=self._ffprobe_result(len(payloads["flight.mp4"])),
            ):
                with self.assertRaisesRegex(FinalizationError, "manifest.outputs.trajectory"):
                    finalize_flight_artifact(directory)

    def test_manifest_must_declare_validation_and_video_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._bundle(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            del manifest["outputs"]["video"]
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "missing required 'video' reference"):
                finalize_flight_artifact(directory)
            self.assertFalse((directory / "video_probe.json").exists())
            self.assertFalse((directory / "integrity.json").exists())

    def test_non_digital_evidence_is_rejected_for_digital_flight_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._bundle(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            trajectory = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
            validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            manifest["provenance"]["evidence_class"] = "real_biological"
            trajectory["metadata"]["evidence_class"] = "real_biological"
            validation["evidence"]["class"] = "real_biological"
            validation["evidence"]["real_biological"] = True
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (directory / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
            (directory / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "unsupported evidence class"):
                finalize_flight_artifact(directory)

    def test_requested_cuda_fallback_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._bundle(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest.update(
                {
                    "requested_backends": {"neural": "malecns", "vision": "flyvis"},
                    "requested_execution": {"neural_device": "cuda", "vision_device": "cuda:0"},
                    "effective_execution": {
                        "neural_device": "cpu",
                        "neural_backend": "numpy_cpu",
                        "vision_device": "cpu",
                    },
                    "effective_backends": {"neural": "numpy_cpu", "vision": "deterministic_stub"},
                    "components": {
                        "neural": {"is_stub": True, "effective_device": "cpu", "effective_backend": "numpy_cpu"},
                        "vision": {"is_stub": True, "runtime": {"device": "cpu"}},
                    },
                }
            )
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "CUDA neural execution"):
                finalize_flight_artifact(directory)

    def test_integrity_retains_component_and_resource_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payloads = self._bundle(directory)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            manifest.update(
                {
                    "requested_backends": {"neural": "malecns", "vision": "flyvis", "body": "reduced"},
                    "effective_backends": {"neural": "torch_cuda", "vision": "flyvis", "body": "reduced_6dof"},
                    "provenance": {
                        **manifest["provenance"],
                        "world": "handbuilt",
                        "vision": "fitted",
                        "neural": "connectome",
                        "body": "handbuilt",
                    },
                    "components": {
                        "world": {"is_stub": True, "source": "handbuilt", "backend": "virtual_world"},
                        "neural": {
                            "is_stub": False,
                            "source": "connectome",
                            "effective_backend": "torch_cuda",
                            "effective_device": "cuda",
                            "runtime": {"backend": "torch_cuda", "device": "cuda", "n": 176422, "edges": 6287749},
                        },
                    },
                    "resource_limits": {"cgroup_memory_max_bytes": 5368709120, "torch_allocator_max_vram_gib": 15.5},
                    "resource_usage": {"peak_host_rss_bytes": 1234, "cuda": {"peak_allocated_bytes": 5678}},
                }
            )
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "finalize_flight_artifact.subprocess.run",
                return_value=self._ffprobe_result(len(payloads["flight.mp4"])),
            ):
                integrity = finalize_flight_artifact(directory)
            self.assertEqual(integrity["provenance"]["component_labels"]["neural"], "connectome")
            self.assertEqual(integrity["provenance"]["components"]["world"]["source"], "handbuilt")
            self.assertEqual(integrity["provenance"]["components"]["neural"]["runtime"]["n"], 176422)
            self.assertEqual(integrity["resources"]["limits"]["cgroup_memory_max_bytes"], 5368709120)
            self.assertEqual(integrity["resources"]["usage"]["cuda"]["peak_allocated_bytes"], 5678)


if __name__ == "__main__":
    unittest.main()
