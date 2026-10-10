from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_courtship_visual import (  # noqa: E402
    DEFAULT_CRITERIA,
    TrialSpec,
    _camera_config,
    analyze_courtship_visual,
)


def _vector_record(values: tuple[float, ...]) -> dict[str, object]:
    raw = struct.pack("<" + "f" * len(values), *values)
    compressed = zlib.compress(raw)
    return {
        "encoding": "zlib+base64",
        "lossless": True,
        "dtype": "float32",
        "shape": [len(values)],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


class CourtshipVisualAnalyzerTests(unittest.TestCase):
    def test_manifest_camera_takes_precedence_over_row_geometry(self) -> None:
        declared = {"hfov_rad": 1.23, "vfov_rad": 0.72}
        row_camera = {"hfov_rad": 2.0, "vfov_rad": 1.0}
        selected = _camera_config(
            {"courtship_visual": {"camera": declared}},
            {"world": {"geometry": {"camera": row_camera}}},
        )
        self.assertEqual(selected, declared)

    def _write_run(
        self,
        root: Path,
        condition: str,
        seed: int,
        *,
        turn: bool,
        retinal_signal: bool,
        compact: bool = False,
        camera_in_manifest: bool = False,
        omit_world_geometry: bool = False,
        frames: int = 24,
    ) -> Path:
        run = root / f"{condition}-seed{seed}"
        run.mkdir()
        camera = {
            "hfov_rad": 1.75,
            "vfov_rad": 1.20,
            "near_m": 0.01,
            "far_m": 8.0,
        }
        manifest = {
            "schema_version": "digital-fly-run-1",
            "execution_policy": "strict-free-behavior-escape-experiment",
            "seed": seed,
            "provenance": {"evidence_class": "digital_simulation"},
            "autonomy_policy": {"strict": True},
            "courtship_visual": {
                "schema_version": "courtship-visual-assay-1",
                "target_source": "independent_world_trajectory",
                "target_condition": condition,
                "target_id": "female_fly",
                **({"camera": camera} if camera_in_manifest else {}),
            },
        }
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows = []
        base_target = [0.8, 0.5, 0.02]
        for index in range(frames):
            time_s = index * 0.005
            pre_yaw = 0.0 if index < 4 or not turn else min(0.35, (index - 3) * 0.018)
            post_yaw = pre_yaw
            if turn and index >= 4:
                post_yaw = min(0.35, (index - 2) * 0.018)
            pose = {"position_m": [0.0, 0.0, 0.02], "orientation_rad": [0.0, 0.0, pre_yaw]}
            target = {
                "id": "female_fly",
                "time_s": time_s,
                "active": True,
                "renderable": condition != "hidden",
                "position_m": base_target,
                "velocity_mps": [0.0, 0.0, 0.0],
                "size_m": [0.10, 0.10, 0.10],
                "visual_only": True,
                "non_interacting": True,
            }
            pixels = ([0.10] * 8, [0.10] * 8)
            if retinal_signal and condition != "hidden":
                pixels = ([0.30] * 8, [0.25] * 8)
            row = {
                "frame_index": index,
                "frame_time_s": time_s,
                "time_s": time_s + 0.005,
                "dt_s": 0.005,
                "position_m": [0.0, 0.0, 0.02],
                "velocity_mps": [0.0, 0.0, 0.0],
                "orientation_rad": [0.0, 0.0, post_yaw],
                "yaw_rate": (post_yaw - pre_yaw) / 0.005,
                "grounded": True,
                "body_before": pose,
                "stimulus_state": {"female_fly": target},
                "visual": {"pixels": [list(pixels[0]), list(pixels[1])]},
            }
            if not omit_world_geometry:
                row["world"] = {"geometry": {"camera": camera}}
            if compact:
                row["visual"] = {"pixels": None}
                row["visual_input"] = {
                    "left_columns": _vector_record(pixels[0]),
                    "right_columns": _vector_record(pixels[1]),
                }
            rows.append(row)
        with (run / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return run

    def test_matched_hidden_retina_and_orientation_can_support_narrow_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trials = []
            for seed in (1729, 1731, 1732):
                moving = self._write_run(root, "moving", seed, turn=True, retinal_signal=True)
                static = self._write_run(root, "static", seed, turn=False, retinal_signal=True)
                hidden = self._write_run(root, "hidden", seed, turn=False, retinal_signal=False)
                trials.extend([
                    TrialSpec(moving, condition="moving", label=f"moving-{seed}"),
                    TrialSpec(static, condition="static", label=f"static-{seed}"),
                    TrialSpec(hidden, condition="hidden", label=f"hidden-{seed}"),
                ])
            report = analyze_courtship_visual(trials)

        self.assertEqual(report["schema_version"], "courtship-visual-analysis-1")
        self.assertEqual(report["verdict"], "VISUAL_ORIENTATION_TRACKING_SUPPORTED")
        self.assertEqual(report["criterion_results"]["moving_target_visible_vs_hidden"], "PASS")
        self.assertEqual(report["criterion_results"]["moving_target_orientation_vs_hidden"], "PASS")
        self.assertEqual(report["criterion_results"]["moving_target_tracking_vs_static"], "PASS")
        self.assertEqual(report["counts"]["matched_moving_hidden_pair_count"], 3)
        self.assertEqual(report["condition_effects"]["moving_vs_static"]["pair_count"], 3)
        moving_reports = [item for item in report["trials"] if item["condition"] == "moving"]
        self.assertTrue(all(item["retinal_visibility_vs_hidden"]["both_eyes_confirmed"] for item in moving_reports))
        self.assertGreater(
            report["condition_effects"]["moving_vs_hidden"]["median_moving_minus_hidden_rad"],
            0.1,
        )
        self.assertEqual(report["assay_scope"], "visual orientation and tracking only")
        self.assertEqual(report["full_courtship_endpoint"], "not_assessed")
        self.assertEqual(report["P1_state"]["status"], "not_measured_or_inferred_by_this_analyzer")

    def test_renderable_fov_without_retinal_difference_is_not_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for seed in (1, 2, 3):
                runs.append(TrialSpec(
                    self._write_run(root, "moving", seed, turn=False, retinal_signal=False),
                    condition="moving",
                ))
                runs.append(TrialSpec(
                    self._write_run(root, "hidden", seed, turn=False, retinal_signal=False),
                    condition="hidden",
                ))
            report = analyze_courtship_visual(runs)

        moving = next(item for item in report["trials"] if item["condition"] == "moving")
        self.assertGreater(moving["target_in_camera_fov_frame_count"], 0)
        self.assertFalse(moving["retinal_visibility_vs_hidden"]["visible_confirmed"])
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertEqual(report["criterion_results"]["moving_target_visible_vs_hidden"], "FAIL")

    def test_static_orientation_alone_does_not_pass_motion_tracking_criterion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trials = []
            for seed in (41, 42):
                moving = self._write_run(root, "moving", seed, turn=True, retinal_signal=True)
                static = self._write_run(root, "static", seed, turn=True, retinal_signal=True)
                hidden = self._write_run(root, "hidden", seed, turn=False, retinal_signal=False)
                trials.extend([
                    TrialSpec(moving, condition="moving"),
                    TrialSpec(static, condition="static"),
                    TrialSpec(hidden, condition="hidden"),
                ])
            report = analyze_courtship_visual(trials)

        self.assertEqual(
            report["criterion_results"]["moving_target_orientation_vs_hidden"], "PASS",
        )
        self.assertEqual(
            report["criterion_results"]["moving_target_tracking_vs_static"], "FAIL",
        )
        self.assertEqual(
            report["verdict"],
            "VISUAL_ORIENTATION_TRACKING_NOT_SUPPORTED_BY_PREDEFINED_CRITERIA",
        )

    def test_compact_lossless_eye_arrays_are_decoded_and_hash_only_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            moving = self._write_run(
                root, "moving", 7, turn=False, retinal_signal=True, compact=True,
                camera_in_manifest=True, omit_world_geometry=True,
            )
            hidden = self._write_run(
                root, "hidden", 7, turn=False, retinal_signal=False, compact=True,
                camera_in_manifest=True, omit_world_geometry=True,
            )
            report = analyze_courtship_visual(
                [TrialSpec(moving, condition="moving"), TrialSpec(hidden, condition="hidden")],
                criteria={
                    "minimum_paired_seeds": 1,
                    "minimum_target_in_fov_frames_per_trial": 2,
                    "minimum_confirmed_visible_frames_per_trial": 2,
                    "minimum_pixel_delta_rms_per_eye": 0.01,
                },
            )
        moving_report = next(item for item in report["trials"] if item["condition"] == "moving")
        self.assertTrue(moving_report["retinal_visibility_vs_hidden"]["both_eyes_confirmed"])
        self.assertEqual(moving_report["camera"]["hfov_rad"], 1.75)

    def test_trial_summarizes_lc10a_and_pc1_labeled_spikes_without_state_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._write_run(
                root, "moving", 31, turn=False, retinal_signal=True, frames=3,
            )
            telemetry_path = run / "telemetry.jsonl"
            rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
            window_s = (0.005, 0.010, 0.005)
            lc10a_spikes = (1, 2, 3)
            pc1_spikes = (0, 1, 0)
            lc10a_rates = (10.0, 20.0, 30.0)
            pc1_rates = (0.0, 6.0, 12.0)
            for index, row in enumerate(rows):
                row["neural"] = {
                    "named_outputs": {
                        "population_spike_counts": {
                            "LC10a": lc10a_spikes[index],
                            "prefix:pC1_": pc1_spikes[index],
                        },
                        "population_rates_hz": {
                            "LC10a": lc10a_rates[index],
                            "prefix:pC1_": pc1_rates[index],
                        },
                    }
                }
                row["neural_window"] = {"window_dt_s": window_s[index]}
            telemetry_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            report = analyze_courtship_visual([TrialSpec(run, condition="moving")])

        trial = report["trials"][0]
        populations = trial["neural_population_summary"]
        self.assertEqual(populations["LC10a"]["total_spikes"], 6)
        self.assertAlmostEqual(populations["LC10a"]["mean_rate_hz"], 20.0)
        self.assertEqual(populations["pC1_labeled_population_proxy"]["total_spikes"], 1)
        self.assertAlmostEqual(populations["pC1_labeled_population_proxy"]["mean_rate_hz"], 6.0)
        self.assertIn("not a validated P1 state or arousal", populations["pC1_labeled_population_proxy"]["interpretation"])
        self.assertEqual(report["P1_state"]["status"], "not_measured_or_inferred_by_this_analyzer")

    def test_paths_and_condition_mapping_api(self) -> None:
        self.assertGreaterEqual(DEFAULT_CRITERIA["minimum_paired_seeds"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            moving = self._write_run(root, "moving", 9, turn=False, retinal_signal=True)
            hidden = self._write_run(root, "hidden", 9, turn=False, retinal_signal=False)
            report = analyze_courtship_visual(
                {"moving": moving, "hidden": hidden},
                criteria={"minimum_paired_seeds": 1},
            )
        self.assertEqual(report["counts"]["matched_moving_hidden_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
