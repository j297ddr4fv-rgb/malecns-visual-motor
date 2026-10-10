#!/usr/bin/env python3
"""Run one separately authorized, offline recorded-input tethered-ball case.

The runner does not start a neural worker. It decodes the saved P/DNp15 rows
with the frozen public decoder, then supplies those recorded commands to the
existing ball session. Physics execution requires a case-specific root review.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-tethered-ball-recorded-input-replay-20261007-v1"
)
PARENT_BALL_BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-tethered-treadmill-pilot-20261007-v1/compile-preflight-v8"
)
FRAME_COUNT = 2000
FRAME_DT_S = 0.005
CONTROL_DT_S = 0.001
NATIVE_DT_S = 0.0001
CONTROL_TICKS_PER_FRAME = 5
NATIVE_STEPS_PER_CONTROL_TICK = 10
SEED = 1729
CANONICAL_XML_SHA256 = "9b301479a1aac6763ff98bf3811fcbe0c909020fecdb81a1b20b90b6e22bf69a"
RESOURCE_LIMITS = {"memory_max": "7G", "cpu_quota": "200%", "runtime_max_seconds": 1800, "gpu": False}


class ReplayError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"expected JSON object at {path}")
    return value


def validate_frozen_sources(source_freeze: dict[str, Any]) -> None:
    mismatches = []
    for raw_path, expected in source_freeze.get("files", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            mismatches.append({"path": raw_path, "reason": "missing"})
            continue
        observed = sha256(path)
        if observed != expected.get("sha256"):
            mismatches.append({"path": raw_path, "expected": expected.get("sha256"), "observed": observed})
    if mismatches:
        raise ReplayError(f"source/asset hash mismatch: {mismatches[:8]}")


def authorization_for_case(
    review_path: Path, case_id: str, protocol_sha: str, source_freeze_sha: str,
    schedule_sha: str, run_id: str,
) -> dict[str, Any]:
    review = read_json(review_path)
    expected = {
        "schema_version": "flygym-tethered-ball-recorded-input-root-review-v1",
        "status": "authorized",
        "authorized_by": "root",
        "case_id": case_id,
        "protocol_sha256": protocol_sha,
        "source_freeze_sha256": source_freeze_sha,
        "decoded_schedule_sha256": schedule_sha,
        "canonical_world_xml_sha256": CANONICAL_XML_SHA256,
        "run_id": run_id,
        "resource_limits": RESOURCE_LIMITS,
    }
    mismatches = {key: {"expected": value, "observed": review.get(key)}
                  for key, value in expected.items() if review.get(key) != value}
    if mismatches:
        raise PermissionError(f"root review is not bound to this frozen case: {mismatches}")
    return {
        **review,
        "root_review_path": str(review_path.resolve()),
        "root_review_file_sha256": sha256(review_path),
    }


class RecordedInputTetheredBallSession:
    """Per-tick P/d adapter around the unchanged, frozen ball session."""

    def __new__(cls, *args, **kwargs):
        from flygym_tethered_treadmill_session_v1 import TetheredBallSessionV1

        class _Adapter(TetheredBallSessionV1):
            def step_recorded_control_tick(
                self, leg_power: float, signed_d: float, soma_side: str
            ) -> dict[str, Any]:
                p = float(leg_power)
                d = float(signed_d)
                if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                    raise ValueError("recorded P must be finite and in [0,1]")
                if not math.isfinite(d) or not -1.0 <= d <= 1.0:
                    raise ValueError("recorded d must be finite and in [-1,1]")
                if soma_side not in {"left", "right"}:
                    raise ValueError("soma_side must follow the accepted held-side rule")

                # The frozen base step rejects a variable P and records d from
                # self.config, so update only these command fields in this
                # subclass before delegating to the unmodified physics step.
                self.config = replace(self.config, leg_power_p=p, differential_ap_d=d)
                self.controller.gesture_config = replace(
                    self.controller.gesture_config,
                    dna02_command=abs(d),
                    dng13_command=abs(d),
                    dna02_soma_side=soma_side,
                    dng13_soma_side=soma_side,
                )
                result = super().step_control_tick((p, p))
                result["recorded_input"] = {
                    "leg_power_P": p,
                    "signed_d": d,
                    "decoded_gesture_amplitude": abs(d),
                    "held_soma_side": soma_side,
                }
                return result

        return _Adapter(*args, **kwargs)


def run_id_for(case_id: str) -> str:
    return f"recorded-ball-{case_id}-T10s-seed{SEED}"


def _window_summary(trace: dict[str, np.ndarray], start_s: float, stop_s: float) -> dict[str, Any]:
    times = np.asarray(trace["time_s"])
    mask = (times >= start_s) & (times < stop_s)
    if not np.any(mask):
        raise ReplayError(f"no native samples in analysis window [{start_s},{stop_s})")
    omega = np.asarray(trace["ball_body_angvel_world_rad_s"])[mask]
    forces = np.asarray(trace["foot_contact_fz_up_native"])[mask]
    return {
        "window_s_half_open": [float(start_s), float(stop_s)],
        "native_sample_count": int(mask.sum()),
        "mean_ball_angular_velocity_world_rad_s": np.mean(omega, axis=0).astype(float).tolist(),
        "sd_ball_angular_velocity_world_rad_s": np.std(omega, axis=0).astype(float).tolist(),
        "max_abs_ball_angular_velocity_world_rad_s": float(np.max(np.abs(omega))),
        "positive_Fz_fraction_by_leg": np.mean(forces > 0.0, axis=0).astype(float).tolist(),
        "mean_positive_Fz_by_leg_native": np.mean(np.maximum(forces, 0.0), axis=0).astype(float).tolist(),
    }


def execute(case_id: str, review_path: Path) -> Path:
    if case_id not in {"c1_static", "c1_positive60", "c1_negative60"}:
        raise ReplayError(f"unknown case: {case_id}")
    source_freeze_path = BUNDLE / "source_freeze.json"
    protocol_path = BUNDLE / "protocol.json"
    prepare_manifest_path = BUNDLE / "prepare-manifest.json"
    source_freeze = read_json(source_freeze_path)
    protocol = read_json(protocol_path)
    prepared_manifest = read_json(prepare_manifest_path)
    source_freeze_sha = sha256(source_freeze_path)
    protocol_sha = sha256(protocol_path)
    if protocol.get("source_freeze_sha256") != source_freeze_sha:
        raise ReplayError("protocol no longer binds the current source freeze")
    if prepared_manifest.get("status") != "prepared_only_no_engine_no_physics_no_neural_forward":
        raise ReplayError("preparation manifest is not the expected no-step record")
    if prepared_manifest.get("protocol_sha256") != protocol_sha:
        raise ReplayError("protocol hash differs from the frozen preparation manifest")
    validate_frozen_sources(source_freeze)

    case_protocol = protocol["cases"][case_id]
    schedule_path = Path(case_protocol["decoded_schedule_path"])
    if sha256(schedule_path) != case_protocol["decoded_schedule_sha256"]:
        raise ReplayError("decoded command schedule hash mismatch")
    run_id = run_id_for(case_id)
    authorization = authorization_for_case(
        review_path, case_id, protocol_sha, source_freeze_sha,
        case_protocol["decoded_schedule_sha256"], run_id,
    )

    output_dir = BUNDLE / "runs" / case_id
    if output_dir.exists():
        raise ReplayError(f"refusing to overwrite an existing capture: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    record_path = output_dir / "run-record.json"
    write_json(record_path, {
        "schema_version": "flygym-tethered-ball-recorded-input-run-v1",
        "status": "starting",
        "case_id": case_id,
        "run_id": run_id,
        "protocol_sha256": protocol_sha,
        "source_freeze_sha256": source_freeze_sha,
        "decoded_schedule_sha256": case_protocol["decoded_schedule_sha256"],
        "root_review_sha256": authorization["root_review_file_sha256"],
        "run_classification": protocol["classification"],
        "neural_worker_started": False,
        "neural_forward_called": False,
        "physics_execution_authorized": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })

    session = None
    try:
        from flygym_tethered_treadmill_session_v1 import TetheredPilotConfig
        import run_flygym_tethered_treadmill_pilot as ball_runner

        with np.load(schedule_path, allow_pickle=False) as saved:
            schedule = {name: saved[name].copy() for name in saved.files}
        if (schedule["seq"].shape != (FRAME_COUNT,)
                or not np.array_equal(schedule["seq"], np.arange(FRAME_COUNT))
                or not np.array_equal(schedule["frame_time_s"], np.arange(FRAME_COUNT) * FRAME_DT_S)):
            raise ReplayError("saved schedule sequence/clock is not exactly 2000 × 5 ms")
        if (not np.isfinite(schedule["leg_power"]).all()
                or not np.isfinite(schedule["signed_ap_d"]).all()
                or not np.isfinite(schedule["dn_p15_left_rate_hz"]).all()
                or not np.isfinite(schedule["dn_p15_right_rate_hz"]).all()):
            raise ReplayError("saved input schedule contains nonfinite values")

        config = TetheredPilotConfig(
            duration_s=10.0,
            seed=SEED,
            leg_power_p=0.0,
            differential_ap_d=0.0,
        )
        session = RecordedInputTetheredBallSession(config)
        session.run_id = run_id
        raw_xml = session.world.mjcf_root.to_xml()
        raw_xml_path = output_dir / "actual_world.raw.xml"
        raw_xml_path.write_text(raw_xml, encoding="utf-8")
        canonical_xml, canonical_audit = ball_runner.canonicalize_world_xml(raw_xml)
        canonical_path = output_dir / "actual_world.canonical.xml"
        canonical_path.write_text(canonical_xml, encoding="utf-8")
        observed_xml_sha = sha256(canonical_path)
        preflight = {
            "schema_version": "flygym-tethered-ball-recorded-input-preflight-v1",
            "status": "pass" if observed_xml_sha == CANONICAL_XML_SHA256 else "rejected",
            "case_id": case_id,
            "run_id": run_id,
            "expected_canonical_world_xml_sha256": CANONICAL_XML_SHA256,
            "observed_canonical_world_xml_sha256": observed_xml_sha,
            "raw_world_xml_sha256": sha256(raw_xml_path),
            "canonicalization_audit": canonical_audit,
            "session_time_s_before_first_step": session.time_s,
            "control_ticks_before_first_step": session._control_samples,
            "native_samples_before_first_step": session._native_samples,
            "physics_steps_before_first_step": session._physics_step_count,
            "session_config": asdict(config),
            "physics_steps_called_by_preflight": 0,
        }
        write_json(output_dir / "preflight.json", preflight)
        if observed_xml_sha != CANONICAL_XML_SHA256:
            raise ReplayError("runtime world MJCF differs from the previously reviewed canonical tethered-ball model")
        if (session.time_s != 0.0 or session._control_samples != 0
                or session._native_samples != 0 or session._physics_step_count != 0):
            raise ReplayError("the run gate must precede every physics step")

        write_json(output_dir / "authorization_record.json", authorization)
        frame_log_path = output_dir / "input_frames.jsonl"
        tick_log_path = output_dir / "control_ticks.jsonl"
        controls_done = 0
        physics_started = time.monotonic()
        with frame_log_path.open("x", encoding="utf-8", buffering=1) as frame_log, \
                tick_log_path.open("x", encoding="utf-8", buffering=1) as tick_log:
            for seq in range(FRAME_COUNT):
                frame_time = float(schedule["frame_time_s"][seq])
                if not math.isclose(session.time_s, frame_time, rel_tol=0.0, abs_tol=1e-9):
                    raise ReplayError(f"session clock differs from recorded frame {seq}")
                power = float(schedule["leg_power"][seq])
                signed_d = float(schedule["signed_ap_d"][seq])
                soma_side = str(schedule["ap_soma_side"][seq])
                frame_log.write(json.dumps({
                    "seq": seq,
                    "frame_time_s": frame_time,
                    "recorded_leg_power_P": power,
                    "recorded_DNp15_L_rate_hz": float(schedule["dn_p15_left_rate_hz"][seq]),
                    "recorded_DNp15_R_rate_hz": float(schedule["dn_p15_right_rate_hz"][seq]),
                    "decoder_ema_left_hz": float(schedule["ema_left_rate_hz"][seq]),
                    "decoder_ema_right_hz": float(schedule["ema_right_rate_hz"][seq]),
                    "decoder_raw_opponent_signal": float(schedule["raw_opponent_signal"][seq]),
                    "decoded_signed_d": signed_d,
                    "decoded_command_amplitude": float(schedule["ap_command_amplitude"][seq]),
                    "held_soma_side": soma_side,
                    "hold_control_ticks": CONTROL_TICKS_PER_FRAME,
                    "stimulus_condition_passed_to_controller": False,
                }, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")

                for tick_in_frame in range(CONTROL_TICKS_PER_FRAME):
                    expected_start = frame_time + tick_in_frame * CONTROL_DT_S
                    if not math.isclose(session.time_s, expected_start, rel_tol=0.0, abs_tol=1e-9):
                        raise ReplayError(f"body control clock split at source frame {seq}, tick {tick_in_frame}")
                    native_first = int(session._native_samples)
                    started = time.monotonic()
                    result = session.step_recorded_control_tick(power, signed_d, soma_side)
                    elapsed = time.monotonic() - started
                    native_last = int(session._native_samples)
                    if int(result["native_samples_added"]) != NATIVE_STEPS_PER_CONTROL_TICK:
                        raise ReplayError("each 1 ms tick must advance exactly ten 100 us native intervals")
                    if not math.isclose(session.time_s, expected_start + CONTROL_DT_S, rel_tol=0.0, abs_tol=1e-9):
                        raise ReplayError("control tick ended outside its scheduled 1 ms interval")
                    if (result["descending_signal_lr"] != [power, power]
                            or result["ap_differential_command"] != signed_d
                            or result["ap_dna02_command"] != abs(signed_d)
                            or result["ap_dng13_command"] != abs(signed_d)
                            or result["ap_soma_side"] != soma_side):
                        raise ReplayError("adapter did not replay the recorded P/d command exactly")
                    row = {
                        "seq": seq,
                        "tick_in_frame": tick_in_frame,
                        "control_time_s": expected_start,
                        "control_end_time_s": float(session.time_s),
                        "source_frame_P": power,
                        "source_frame_signed_d": signed_d,
                        "held_soma_side": soma_side,
                        "actual_descending_signal_lr": [power, power],
                        "actual_signed_AP_d": signed_d,
                        "actual_AP_amplitude": abs(signed_d),
                        "native_sample_indices_halfopen": [native_first, native_last],
                        "native_samples_added": int(result["native_samples_added"]),
                        "position_targets_rad": result["position_targets_rad"],
                        "controller_telemetry": result["controller_telemetry"],
                        "step_wall_s": elapsed,
                    }
                    tick_log.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
                    controls_done += 1
                if seq % 200 == 199 or seq == FRAME_COUNT - 1:
                    print(f"{case_id}: {seq + 1}/{FRAME_COUNT} frames; t={session.time_s:.3f}s", file=sys.stderr, flush=True)

        if (controls_done != FRAME_COUNT * CONTROL_TICKS_PER_FRAME
                or session._native_samples != FRAME_COUNT * CONTROL_TICKS_PER_FRAME * NATIVE_STEPS_PER_CONTROL_TICK + 1
                or session._physics_step_count != FRAME_COUNT * CONTROL_TICKS_PER_FRAME * NATIVE_STEPS_PER_CONTROL_TICK
                or not math.isclose(session.time_s, 10.0, rel_tol=0.0, abs_tol=1e-8)):
            raise ReplayError("capture did not meet the 10 s / 10,000 controls / 100,000 physics-step contract")

        session.save_trajectory(output_dir / "trajectory.npz")
        used_trace = {key: value[:session._native_samples] for key, value in session.trace.items()}
        windows = {
            "early_5p0_5p5": _window_summary(used_trace, 5.0, 5.5),
            "late_6_10": _window_summary(used_trace, 6.0, 10.0),
        }
        summary = {
            "schema_version": "flygym-tethered-ball-recorded-input-summary-v1",
            "status": "complete",
            "case_id": case_id,
            "run_id": run_id,
            "protocol_sha256": protocol_sha,
            "source_freeze_sha256": source_freeze_sha,
            "decoded_schedule_sha256": case_protocol["decoded_schedule_sha256"],
            "root_review_sha256": authorization["root_review_file_sha256"],
            "input_frame_count": FRAME_COUNT,
            "control_tick_count": controls_done,
            "native_sample_count": session._native_samples,
            "physics_step_count": session._physics_step_count,
            "final_time_s": float(session.time_s),
            "elapsed_physics_wall_s": time.monotonic() - physics_started,
            "trajectory_sha256": sha256(output_dir / "trajectory.npz"),
            "contacts_sha256": sha256(output_dir / "contacts.npz"),
            "input_frames_sha256": sha256(frame_log_path),
            "control_ticks_sha256": sha256(tick_log_path),
            "canonical_world_xml_sha256": sha256(output_dir / "actual_world.canonical.xml"),
            "windows": windows,
            "authority": {
                "root_qpos_write_count_max_after_initialization": int(np.max(used_trace["root_qpos_write_count"])),
                "root_qvel_write_count_max": int(np.max(used_trace["root_qvel_write_count"])),
                "root_external_wrench_abs_max_native": float(np.max(np.abs(used_trace["root_external_wrench_native"]))),
                "ball_external_wrench_abs_max_native": float(np.max(np.abs(used_trace["ball_external_wrench_native"]))),
                "qfrc_applied_abs_max_native": float(np.max(np.abs(used_trace["qfrc_applied_native"]))),
                "thorax_ground_contact_count": int(np.sum(used_trace["thorax_ground_contact"])),
                "nonleg_ball_contact_count": int(np.sum(used_trace["nonleg_ball_contact"])),
            },
            "interpretation": "recorded fixed-head neural outputs drove an engineered AP template adapter and tethered-ball plant; ball angular velocity is a mechanics response, not biological turning validation",
            "condition_label_or_stimulus_direction_sent_to_controller": False,
            "neural_worker_or_neural_forward": False,
        }
        controller_metadata = dict(session.controller.provenance)
        controller_metadata["recorded_input_replay"] = {
            "input_mode": "offline recorded neural outputs from the original fixed-head c=1 source",
            "source_condition_label_sent_to_controller": False,
            "CPG_input": "symmetric (P,P), with P taken from original worker frames",
            "AP_input": "signed decoded d, with both abstract gesture amplitudes abs(d) and held selected soma side",
            "DNP15_decoder_source_sha256": protocol["inputs"]["decoder"]["source_sha256"],
            "source_fixed_head_rows": FRAME_COUNT,
            "held_control_ticks_per_neural_frame": CONTROL_TICKS_PER_FRAME,
            "neural_worker_started": False,
            "neural_forward_called": False,
            "closed_loop": False,
        }
        write_json(output_dir / "controller_metadata.json", controller_metadata)
        summary["controller_metadata_sha256"] = sha256(output_dir / "controller_metadata.json")
        write_json(output_dir / "summary.json", summary)
        write_json(record_path, {
            **read_json(record_path),
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "summary_sha256": sha256(output_dir / "summary.json"),
            "trajectory_sha256": summary["trajectory_sha256"],
            "contacts_sha256": summary["contacts_sha256"],
            "native_sample_count": session._native_samples,
            "control_tick_count": controls_done,
        })
        print(json.dumps({"status": "complete", "case_id": case_id, "run_dir": str(output_dir), **summary}, sort_keys=True))
        return output_dir
    except Exception as exc:
        write_json(output_dir / "failure.json", {
            "schema_version": "flygym-tethered-ball-recorded-input-failure-v1",
            "status": "failed_partial_preserved",
            "case_id": case_id,
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "time_s": None if session is None else float(session.time_s),
            "native_samples": None if session is None else int(session._native_samples),
            "control_samples": None if session is None else int(session._control_samples),
            "physics_steps": None if session is None else int(session._physics_step_count),
            "protocol_sha256": protocol_sha,
            "source_freeze_sha256": source_freeze_sha,
            "decoded_schedule_sha256": case_protocol["decoded_schedule_sha256"],
        })
        raise
    finally:
        if session is not None:
            session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute exactly one root-authorized CPU case")
    parser.add_argument("--case", choices=("c1_static", "c1_positive60", "c1_negative60"))
    parser.add_argument("--root-review", type=Path)
    args = parser.parse_args()
    if not args.execute or args.case is None or args.root_review is None:
        parser.error("execution requires --execute, --case, and a case-specific --root-review JSON")
    execute(args.case, args.root_review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
