#!/usr/bin/env python3
"""Summarize the frozen standardized FlyGym regression bundle.

This is a read-only summarizer. It consumes the bundle input/suite manifests,
the neural runner's per-case records, and the separately recorded mechanics
captures. It never loads a simulator or model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "flygym_standardized_acceptance_v1.json"
DEFAULT_BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "standardized-repair-20261007-v1"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the mechanics freeze's canonical encoding."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def first(mapping: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def resolve_path(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def max_abs_numeric(value: Any) -> float | None:
    scalar = numeric(value)
    if scalar is not None:
        return abs(scalar)
    if isinstance(value, dict):
        for key in ("max_abs", "max_abs_native", "peak_abs", "value"):
            candidate = max_abs_numeric(value.get(key))
            if candidate is not None:
                return candidate
        values = [max_abs_numeric(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        values = [max_abs_numeric(item) for item in value]
    else:
        return None
    valid = [item for item in values if item is not None]
    return max(valid) if valid else None


def case_id(row: dict[str, Any]) -> str:
    value = first(row, "case_id", "id", "name")
    return str(value) if value is not None else ""


def manifest_cases(document: Any, key: str) -> list[dict[str, Any]]:
    rows = document.get(key, []) if isinstance(document, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def index_cases(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case_id(row): row for row in rows if case_id(row)}


def suite_group_for(suite_manifest: dict[str, Any], row: dict[str, Any]) -> str | None:
    cid = case_id(row)
    groups = suite_manifest.get("case_groups", {})
    if isinstance(groups, dict):
        value = groups.get(cid)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return first(value, "assay_group", "source_assay", "group")
    value = first(row, "assay_group", "source_assay", "original_assay", "group")
    return str(value) if value is not None else None


def case_output_dir(bundle: Path, row: dict[str, Any], group: str | None) -> Path | None:
    explicit = resolve_path(first(row, "run_output_dir", "run_dir", "output_dir"), bundle)
    if explicit is not None and explicit.exists():
        return explicit
    cid = case_id(row)
    assay_id = first(row, "assay_id", "_suite_assay_id", default="standardized-repair-v1")
    candidates = [bundle / "runs" / str(assay_id) / cid]
    if group:
        candidates.append(bundle / "runs" / group / cid)
    candidates.extend((bundle / "runs" / cid, bundle / "runs" / "neural" / cid))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def read_optional_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def path_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    record: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        record["exists"] = True
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    else:
        record["exists"] = False
    return record


def recorded_file_hash(record: Any, *keys: str) -> str | None:
    value = first(record, *keys)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        digest = first(value, "sha256", "hash")
        return digest if isinstance(digest, str) else None
    return None


def compare_hash(actual: str | None, expected: str | None) -> bool | None:
    if not actual or not expected:
        return None
    return actual.lower() == expected.lower()


def window_records(summary: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    windows = summary.get(key, {})
    return {str(name): value for name, value in windows.items() if isinstance(value, dict)} if isinstance(windows, dict) else {}


def neural_metrics(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize old and v1 per-window fields without conflating raw/gated values."""
    raw_p = first(value, "mean_P", "mean_raw_P", "raw_P")
    control_p = first(value, "mean_control_P", "control_power", "mean_control_power")
    gated_d = first(value, "mean_gated_signed_d", "mean_signed_d", "gated_d")
    ungated_d = first(value, "mean_ungated_signed_d", "ungated_signed_d")
    raw_opponent = first(value, "mean_raw_opponent_signal", "raw_opponent_signal")
    return {
        "frame_count": first(value, "frames", "frame_count", "visual_frames"),
        "raw_P": raw_p,
        "control_power": control_p,
        "static_gate_fraction": first(
            value, "static_gate_fraction", "mean_static_gate_fraction", "gate_active_fraction"
        ),
        "raw_DNp15_L_Hz": first(value, "mean_DNp15_L_Hz", "raw_DNp15_L_Hz", "dn_p15_left_rate_hz"),
        "raw_DNp15_R_Hz": first(value, "mean_DNp15_R_Hz", "raw_DNp15_R_Hz", "dn_p15_right_rate_hz"),
        "raw_opponent_signal": raw_opponent,
        "ungated_signed_d": ungated_d,
        "gated_d": gated_d,
        "legacy_mean_abs_d": first(value, "mean_abs_d"),
        "legacy_shadow_d": first(value, "mean_legacy_shadow_d", "legacy_shadow_d"),
        "window_s_half_open": first(value, "window_s_half_open", "window_s", "interval_s"),
    }


def ball_metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "omega_world_mean_rad_s": first(
            value, "ball_body_angvel_world_rad_s_mean", "omega_world_mean_rad_s", "mean_xyz"
        ),
        "omega_world_sd_rad_s": first(
            value, "ball_body_angvel_world_rad_s_sd", "omega_world_sd_rad_s", "sd_xyz"
        ),
        "max_abs_angular_velocity_rad_s": first(
            value, "max_abs_ball_angular_velocity_rad_s", "max_abs_angular_velocity_rad_s"
        ),
        "positive_fz_fraction_by_leg": first(
            value, "positive_fz_fraction_by_leg", "foot_contact_fraction_by_leg"
        ),
        "mean_positive_fz_by_leg_native": first(value, "mean_positive_fz_by_leg_native"),
        "native_samples": first(value, "native_samples", "sample_count"),
        "window_s_half_open": first(value, "window_s_half_open", "window_s", "interval_s"),
    }


def normalize_neural_case(
    bundle: Path,
    row: dict[str, Any],
    group: str | None,
    expected: dict[str, Any],
) -> dict[str, Any]:
    cid = case_id(row)
    out_dir = case_output_dir(bundle, row, group)
    summary_path = out_dir / "case_summary.json" if out_dir else None
    record_path = out_dir / "run-record.json" if out_dir else None
    integrity_path = out_dir / "run-integrity.json" if out_dir else None
    summary = read_optional_json(summary_path)
    record = read_optional_json(record_path)
    integrity = read_optional_json(integrity_path)
    summary = summary if isinstance(summary, dict) else {}
    record = record if isinstance(record, dict) else {}
    status = first(summary, "status", default=first(record, "status", default="missing"))
    source_windows = window_records(summary, "neural_windows")
    ball_windows = window_records(summary, "ball_windows")
    normalized_windows: dict[str, Any] = {}
    for name in sorted(set(source_windows) | set(ball_windows)):
        neural = neural_metrics(source_windows.get(name, {})) if name in source_windows else None
        ball = ball_metrics(ball_windows.get(name, {})) if name in ball_windows else None
        normalized_windows[name] = {"neural": neural, "ball": ball}

    input_entry = first(record, "input_npz", default={})
    row_input = first(row, "input_path", "input_npz", "path")
    input_path_value = first(row_input, "path") if isinstance(row_input, dict) else row_input
    manifest_input_path = resolve_path(input_path_value, Path(first(row, "_manifest_dir", default=bundle)))
    record_input_path = resolve_path(first(input_entry, "path"), bundle)
    actual_input_path = manifest_input_path if manifest_input_path and manifest_input_path.is_file() else record_input_path
    expected_input_hash = first(row, "input_sha256", "input_npz_sha256", "sha256", default=recorded_file_hash(row_input, "sha256") or recorded_file_hash(input_entry, "sha256"))
    actual_input_hash = sha256_file(actual_input_path) if actual_input_path and actual_input_path.is_file() else None
    record_input_hash = recorded_file_hash(input_entry, "sha256") or first(summary, "input_npz_sha256")

    artifacts: dict[str, Any] = {
        "input": path_record(actual_input_path),
        "input_manifest": None,
        "case_summary": path_record(summary_path),
        "run_record": path_record(record_path),
        "run_integrity": path_record(integrity_path),
        "decoder_policy": path_record(out_dir / "decoder-policy.json" if out_dir else None),
        "repair_freeze": path_record(bundle / "repair-source-freeze.json"),
    }
    decoder_policy = read_optional_json(out_dir / "decoder-policy.json") if out_dir else None
    input_manifest_record = first(record, "input_manifest", default={})
    if isinstance(input_manifest_record, dict):
        record_manifest_path = resolve_path(first(input_manifest_record, "path"), bundle)
        if record_manifest_path is None:
            record_manifest_path = resolve_path(first(row, "_input_manifest_path"), bundle)
        artifacts["input_manifest"] = path_record(record_manifest_path)
        if isinstance(artifacts["input_manifest"], dict):
            recorded_manifest_hash = recorded_file_hash(input_manifest_record, "sha256") or first(record, "input_manifest_sha256") or first(summary, "input_manifest_sha256") or first(row, "input_manifest_sha256")
            artifacts["input_manifest"]["recorded_sha256"] = recorded_manifest_hash
            artifacts["input_manifest"]["hash_matches_record"] = compare_hash(artifacts["input_manifest"].get("sha256"), recorded_manifest_hash)
    for name, recorded in (
        ("trajectory", first(record, "trajectory_sha256", default=first(summary, "trajectory_sha256"))),
        ("contacts", first(record, "contacts_sha256", default=first(summary, "contacts_sha256"))),
        ("decoded_frames", first(summary, "decoded_frames_sha256")),
        ("control_ticks", first(summary, "control_ticks_sha256")),
    ):
        candidate = out_dir / ("trajectory.npz" if name == "trajectory" else "contacts.npz" if name == "contacts" else "decoded_frames.jsonl" if name == "decoded_frames" else "control_ticks.jsonl") if out_dir else None
        artifacts[name] = path_record(candidate)
        if isinstance(artifacts[name], dict):
            artifacts[name]["recorded_sha256"] = recorded
            artifacts[name]["hash_matches_record"] = compare_hash(artifacts[name].get("sha256"), recorded)

    recorded_summary_hash = first(record, "summary_sha256", default=first(summary, "case_summary_sha256"))
    if isinstance(artifacts.get("case_summary"), dict):
        artifacts["case_summary"]["recorded_sha256"] = recorded_summary_hash
        artifacts["case_summary"]["hash_matches_record"] = compare_hash(artifacts["case_summary"].get("sha256"), recorded_summary_hash)

    if isinstance(artifacts["input"], dict):
        artifacts["input"]["manifest_sha256"] = expected_input_hash
        artifacts["input"]["recorded_sha256"] = record_input_hash
        artifacts["input"]["hash_matches_manifest"] = compare_hash(actual_input_hash, str(expected_input_hash) if expected_input_hash else None)
        artifacts["input"]["hash_matches_record"] = compare_hash(actual_input_hash, str(record_input_hash) if record_input_hash else None)

    worker_count = first(summary, "worker_frame_count", default=first(record, "worker_artifact_audit", default={}))
    if isinstance(worker_count, dict):
        worker_count = first(worker_count, "frame_count")
    neural_ticks = first(summary, "worker_neural_ticks", "neural_ticks")
    if neural_ticks is None:
        neural_ticks = first(record, "worker_neural_ticks")
    body_ticks = first(summary, "body_control_ticks", default=first(record, "body_control_ticks"))
    physics_steps = first(summary, "body_physics_steps", default=first(record, "body_physics_steps"))
    native_samples = first(summary, "body_native_samples", default=first(record, "body_native_samples"))
    closed_loop = first(summary, "closed_loop", default=first(record, "closed_loop", default=False))
    if isinstance(closed_loop, dict):
        closed_loop = first(closed_loop, "enabled", "is_closed_loop", default=False)
    decoder_policy = read_optional_json(out_dir / "decoder-policy.json") if out_dir else None
    if not isinstance(decoder_policy, dict):
        decoder_policy = first(summary, "decoder", default=record.get("decoder_provenance", {}))
    candidate_policy = decoder_policy.get("candidate", decoder_policy) if isinstance(decoder_policy, dict) else {}
    gate_policy = candidate_policy.get("power_gate", {}) if isinstance(candidate_policy, dict) else {}
    recorded_repair_freeze = first(record, "source_hashes_after", default={})
    recorded_repair_freeze_hash = recorded_file_hash(recorded_repair_freeze.get("repair_freeze"), "sha256") if isinstance(recorded_repair_freeze, dict) else None
    if isinstance(artifacts.get("repair_freeze"), dict):
        artifacts["repair_freeze"]["recorded_sha256"] = recorded_repair_freeze_hash
        artifacts["repair_freeze"]["hash_matches_record"] = compare_hash(artifacts["repair_freeze"].get("sha256"), recorded_repair_freeze_hash)

    checks = {
        "status_complete": status == "complete",
        "neural_frame_count": worker_count == expected.get("visual_frame_count"),
        "neural_tick_count": neural_ticks == expected.get("expected_neural_ticks", 8000),
        "body_control_ticks": body_ticks == expected.get("expected_body_control_ticks"),
        "body_physics_steps": physics_steps == expected.get("expected_body_physics_steps"),
        "body_native_samples": native_samples == expected.get("expected_body_native_samples"),
        "input_hash_matches_manifest": (artifacts.get("input") or {}).get("hash_matches_manifest") is True,
        "trajectory_hash_matches_record": (artifacts.get("trajectory") or {}).get("hash_matches_record") is True,
        "contacts_hash_matches_record": (artifacts.get("contacts") or {}).get("hash_matches_record") is True,
        "case_summary_hash_matches_record": (artifacts.get("case_summary") or {}).get("hash_matches_record") is True,
        "input_manifest_hash_matches_record": (artifacts.get("input_manifest") or {}).get("hash_matches_record") is True,
        "repair_freeze_hash_matches_record": (artifacts.get("repair_freeze") or {}).get("hash_matches_record") is True,
        "core_freeze_hash_present": isinstance(first(summary, "core_freeze_sha256", default=first(record, "source_hashes_after", default={}).get("core_freeze", {}).get("sha256") if isinstance(record.get("source_hashes_after"), dict) else None), str),
        "stimulus_label_not_sent": first(record, "stimulus_label_sent_to_decoder_or_controller", default=not first(summary, "stimulus_label_or_target_to_controller", default=False)) is False,
        "routing_label_matches_contract": first(row, "routing_label", "worker_condition", default="static") == expected.get("routing_label", "static"),
        "seed_matches_contract": first(row, "seed", default=first(record, "body_config", default={}).get("seed") if isinstance(record.get("body_config"), dict) else expected.get("seed")) == expected.get("seed"),
        "closed_loop_false": closed_loop is False,
        "static_gate_threshold_reported": numeric(gate_policy.get("p_deadband")) is not None if isinstance(gate_policy, dict) else False,
        "fixed_normalization_matches_contract": numeric(candidate_policy.get("normalization_hz")) == numeric(expected.get("_readout_contract", {}).get("fixed_opponent_normalization_hz")) if isinstance(candidate_policy, dict) else False,
        "zero_gray_policy_reported": first(candidate_policy, "baseline_mode") == "zero_gray" if isinstance(candidate_policy, dict) else False,
        "legacy_shadow_diagnostic_only": decoder_policy.get("shadow_controls_body") is False if isinstance(decoder_policy, dict) and "shadow_controls_body" in decoder_policy else True,
    }
    # The new runner records static-window controls alongside raw channels.
    required_fields = {
        "raw_P", "control_power", "static_gate_fraction", "raw_DNp15_L_Hz",
        "raw_DNp15_R_Hz", "raw_opponent_signal", "ungated_signed_d", "gated_d",
    }
    reporting_fields_present = True
    for name in ("gray_0_1", "early_1_1p5", "late_2_4"):
        window_metrics = normalized_windows.get(name, {}).get("neural")
        if window_metrics is None or any(window_metrics.get(key) is None for key in required_fields):
            reporting_fields_present = False
            break
    checks["required_neural_readouts_present"] = reporting_fields_present
    checks["integrity_record_present"] = integrity_path is not None and integrity_path.is_file()
    checks["input_path_hash_matches_run_record"] = compare_hash(actual_input_hash, record_input_hash) is True
    source_hashes_before = record.get("source_hashes_before", {}) if isinstance(record.get("source_hashes_before"), dict) else {}
    source_hashes_after = record.get("source_hashes_after", {}) if isinstance(record.get("source_hashes_after"), dict) else {}
    def source_digest_map(value: dict[str, Any]) -> dict[str, str]:
        return {name: str(entry["sha256"]) for name, entry in value.items() if isinstance(entry, dict) and isinstance(entry.get("sha256"), str)}
    source_hashes_unchanged = bool(source_digest_map(source_hashes_before)) and source_digest_map(source_hashes_before) == source_digest_map(source_hashes_after)
    checks["source_hashes_unchanged"] = source_hashes_unchanged
    return {
        "case_id": cid,
        "cohort": "neural",
        "assay_group": group,
        "condition": first(row, "condition", "stimulus_metadata", default={}),
        "notes": first(row, "notes"),
        "source_case_id": first(row, "source_case_id"),
        "source_bundle": first(row, "source_bundle"),
        "source_manifest": first(row, "source_manifest"),
        "status": status,
        "closed_loop": bool(closed_loop),
        "closed_loop_basis": "fixed prerecorded retinal input followed by separate body/ball replay",
        "routing_label": first(row, "routing_label", "worker_condition", default="static"),
        "seed": first(row, "seed", default=first(record, "body_config", default={}).get("seed") if isinstance(record.get("body_config"), dict) else None),
        "counts": {
            "visual_frames": worker_count,
            "neural_ticks": neural_ticks,
            "body_control_ticks": body_ticks,
            "body_physics_steps": physics_steps,
            "body_native_samples": native_samples,
        },
        "hashes": artifacts,
        "core_freeze_sha256": first(summary, "core_freeze_sha256", default=first(record, "source_hashes_after", default={}).get("core_freeze", {}).get("sha256") if isinstance(record.get("source_hashes_after"), dict) else None),
        "decoder_policy": decoder_policy,
        "repair_freeze_sha256": first(source_hashes_after, "repair_freeze", default={}).get("sha256") if isinstance(source_hashes_after.get("repair_freeze"), dict) else None,
        "windows": normalized_windows,
        "checks": checks,
        "complete_for_engineering": bool(checks) and all(checks.values()),
        "integrity_record": integrity if integrity else None,
    }


def mechanics_output_dir(bundle: Path, row: dict[str, Any]) -> Path:
    explicit = resolve_path(first(row, "run_output_dir", "run_dir", "output_dir", "case_dir"), bundle)
    if explicit is not None:
        return explicit
    return bundle / "mechanics" / "mechanics_cases" / case_id(row)


def resolve_mechanics_file(
    case_dir: Path, row: dict[str, Any], row_keys: tuple[str, ...], candidates: tuple[str, ...], base: Path
) -> Path | None:
    explicit = resolve_path(first(row, *row_keys), base)
    if explicit is not None:
        return explicit
    return next((case_dir / name for name in candidates if (case_dir / name).is_file()), case_dir / candidates[0])


def normalize_mechanics_window(name: str, value: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    # Keep the worker's raw metrics intact and add only explicit, unambiguous aliases.
    body_yaw = first(value, "body_yaw", "thorax_yaw", "thorax_heading_yaw", "yaw_metrics")
    if body_yaw is None:
        body_yaw = {
            key: value[key]
            for key in (
                "thorax_heading_yaw_rad", "body_yaw_change_rad", "body_yaw_change_deg",
                "yaw_change_rad", "yaw_change_deg", "yaw_start_rad", "yaw_end_rad",
                "thorax_heading_yaw_change_rad", "thorax_heading_yaw_start_rad", "thorax_heading_yaw_end_rad",
                "mean_body_yaw_rate_rad_s",
            )
            if key in value
        }
    return {
        "mechanics": {
            "constant_P": first(condition, "P", "constant_P", "leg_power_p"),
            "d": first(condition, "d", "differential_ap_d", "ap_d"),
            "body_yaw": body_yaw,
            "body_yaw_change_deg": first(value, "thorax_heading_yaw_change_deg_0p5_3s", "body_yaw_change_deg", "yaw_change_deg"),
            "body_yaw_change_rad": first(value, "thorax_heading_yaw_change_rad_0p5_3s", "body_yaw_change_rad", "yaw_change_rad", default=(numeric(value.get("thorax_heading_yaw_end_rad")) - numeric(value.get("thorax_heading_yaw_start_rad")) if numeric(value.get("thorax_heading_yaw_end_rad")) is not None and numeric(value.get("thorax_heading_yaw_start_rad")) is not None else None)),
            "contact": first(value, "contact", "contacts", "contact_metrics", "contact_summary"),
            "per_leg_contact": first(value, "per_leg_contact", "foot_contact_fraction_by_leg", "positive_fz_fraction_by_leg", "loaded_fz_support_fraction_by_leg", "loaded_fraction_by_leg"),
            "at_least_two_loaded_legs_fraction": first(value, "at_least_two_loaded_legs_fraction", "fraction_samples_at_least_two_loaded_legs"),
            "sample_count": first(value, "sample_count", "samples", "native_samples"),
            "window_s_half_open": first(value, "window_s_half_open", "window_s", "interval_s"),
            "source_metrics": value,
        }
    }


def normalize_mechanics_case(bundle: Path, row: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    cid = case_id(row)
    case_dir = mechanics_output_dir(bundle, row)
    suite_dir = bundle / "mechanics"
    base_dir = bundle
    record_path = resolve_mechanics_file(case_dir, row, ("run_record_path",), ("run-record.json", "run_record.json"), base_dir)
    manifest_path = resolve_mechanics_file(case_dir, row, ("manifest_path",), ("manifest.json", "case-manifest.json"), base_dir)
    summary_path = resolve_mechanics_file(case_dir, row, ("summary_path", "analysis_summary_path"), ("case_summary.json", "summary.json", "analysis-summary.json"), base_dir)
    integrity_path = resolve_mechanics_file(case_dir, row, ("integrity_path",), ("run-integrity.json", "integrity.json"), base_dir)
    trajectory_path = resolve_mechanics_file(case_dir, row, ("trajectory_path",), ("trajectory.npz",), base_dir)
    contacts_path = resolve_mechanics_file(case_dir, row, ("contacts_path",), ("contacts.npz",), base_dir)
    record = read_optional_json(record_path)
    manifest = read_optional_json(manifest_path)
    summary = read_optional_json(summary_path)
    integrity = read_optional_json(integrity_path)
    record = record if isinstance(record, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    cohort_manifest_path = suite_dir / "cohort-manifest.json"
    cohort_manifest = read_optional_json(cohort_manifest_path)
    cohort_manifest = cohort_manifest if isinstance(cohort_manifest, dict) else {}
    analysis_manifest_ref = cohort_manifest.get("analysis_manifest")
    analysis_manifest_path = resolve_path(first(analysis_manifest_ref, "path"), bundle)
    if analysis_manifest_path is None:
        analysis_manifest_path = suite_dir / "analysis" / "manifest.json"
    analysis_manifest = read_optional_json(analysis_manifest_path)
    analysis_manifest = analysis_manifest if isinstance(analysis_manifest, dict) else {}
    analysis_outputs = analysis_manifest.get("outputs", {})
    if not isinstance(analysis_outputs, dict):
        analysis_outputs = {}
    event_name = f"events-{cid}.json"
    event_entry = analysis_outputs.get(event_name)
    event_path = resolve_path(first(event_entry, "path"), bundle)
    if event_path is None:
        event_path = analysis_manifest_path.parent / event_name
    event_summary = read_optional_json(event_path)
    event_summary = event_summary if isinstance(event_summary, dict) else {}
    effective_config_path = case_dir / "effective_config.json"
    effective = read_optional_json(effective_config_path)
    effective = effective if isinstance(effective, dict) else {}
    condition = first(row, "condition", default={})
    condition = condition if isinstance(condition, dict) else {}
    merged_condition = {**effective, **record, **condition}
    if first(merged_condition, "P", "constant_P", "leg_power_p") is None and row.get("constant_P") is not None:
        merged_condition["constant_P"] = row["constant_P"]
    if first(merged_condition, "d", "differential_ap_d", "ap_d") is None and row.get("d") is not None:
        merged_condition["d"] = row["d"]
    metrics_source = first(summary, "windows", "window_metrics", "mechanics_windows", "metrics_by_window", "analysis_windows")
    if not isinstance(metrics_source, dict):
        metrics_source = first(manifest, "windows", "window_metrics", "mechanics_windows", "metrics_by_window", "analysis_windows")
    if not isinstance(metrics_source, dict):
        metrics_source = first(record, "windows", "window_metrics", "mechanics_windows", "metrics_by_window", "analysis_windows")
    if not isinstance(metrics_source, dict) and isinstance(record.get("window_summary"), dict):
        metrics_source = {"analysis_0p5_3": record["window_summary"]}
    windows = {
        str(name): normalize_mechanics_window(str(name), value, merged_condition)
        for name, value in (metrics_source.items() if isinstance(metrics_source, dict) else [])
        if isinstance(value, dict)
    }

    counts_source = first(summary, "counts", default=first(record, "counts", default=first(manifest, "counts", default={})))
    if not isinstance(counts_source, dict):
        counts_source = {}
    counts = {
        "control_ticks": first(counts_source, "control_ticks", "control_tick_count", default=first(record, "control_ticks", "control_tick_count")),
        "physics_steps": first(counts_source, "physics_steps", "physics_step_count", default=first(record, "physics_steps", "physics_step_count")),
        "native_samples": first(counts_source, "native_samples", "native_sample_count", default=first(record, "native_samples", "native_sample_count")),
    }
    recorded_hashes = first(record, "hashes", "artifact_hashes", "file_hashes", default=first(manifest, "hashes", "artifact_hashes", "file_hashes", default={}))
    if not isinstance(recorded_hashes, dict):
        recorded_hashes = {}
    capture_files = record.get("capture_files", {}) if isinstance(record.get("capture_files"), dict) else {}
    nested_files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
    source_freeze_document = read_optional_json(suite_dir / "source_freeze.json")
    source_freeze_document = source_freeze_document if isinstance(source_freeze_document, dict) else {}
    protocol_document = read_optional_json(suite_dir / "protocol.json")
    protocol_document = protocol_document if isinstance(protocol_document, dict) else {}
    artifacts = {
        "run_record": path_record(record_path),
        "manifest": path_record(manifest_path),
        "summary": path_record(summary_path),
        "integrity": path_record(integrity_path),
        "effective_config": path_record(effective_config_path),
        "trajectory": path_record(trajectory_path),
        "contacts": path_record(contacts_path),
        "source_freeze": path_record(suite_dir / "source_freeze.json"),
        "protocol": path_record(suite_dir / "protocol.json"),
        "cohort_manifest": path_record(cohort_manifest_path),
        "analysis_manifest": path_record(analysis_manifest_path),
        "gait_events": path_record(event_path),
    }
    if isinstance(artifacts.get("analysis_manifest"), dict):
        analysis_manifest_expected_hash = recorded_file_hash(analysis_manifest_ref, "sha256")
        artifacts["analysis_manifest"]["recorded_sha256"] = analysis_manifest_expected_hash
        artifacts["analysis_manifest"]["hash_matches_cohort_manifest"] = compare_hash(
            artifacts["analysis_manifest"].get("sha256"),
            analysis_manifest_expected_hash,
        )
    aliases_by_artifact = {
        "run_record": ("run_record_sha256", "run-record.json", "run_record.json"),
        "manifest": ("manifest_sha256", "manifest.json", "case-manifest.json"),
        "summary": ("summary_sha256", "case_summary_sha256", "case_summary.json", "summary.json", "analysis-summary.json"),
        "integrity": ("integrity_sha256", "run-integrity.json", "integrity.json"),
        "effective_config": ("effective_config_sha256", "effective_config.json"),
        "trajectory": ("trajectory_sha256", "trajectory.npz"),
        "contacts": ("contacts_sha256", "contacts.npz"),
        "source_freeze": ("source_freeze_sha256", "source_freeze.json"),
        "protocol": ("protocol_sha256", "protocol.json"),
    }
    for key, aliases in aliases_by_artifact.items():
        expected_hash = first(recorded_hashes, *aliases)
        if expected_hash is None and key in {"manifest", "trajectory", "contacts", "effective_config"}:
            capture_name = {
                "manifest": "manifest.json",
                "trajectory": "trajectory.npz",
                "contacts": "contacts.npz",
                "effective_config": "effective_config.json",
            }[key]
            expected_hash = recorded_file_hash(capture_files.get(capture_name), "sha256")
        if expected_hash is None and key == "source_freeze":
            expected_hash = recorded_file_hash(cohort_manifest.get("source_freeze"), "sha256")
        if expected_hash is None and key == "protocol":
            expected_hash = recorded_file_hash(cohort_manifest.get("protocol"), "sha256")
        if expected_hash is None and key == "cohort_manifest":
            expected_hash = first(row, "cohort_manifest_sha256")
        if expected_hash is None and key == "gait_events":
            expected_hash = recorded_file_hash(event_entry, "sha256")
        if expected_hash is None and isinstance(nested_files, dict):
            nested_entry = next((nested_files[name] for name in aliases if name in nested_files), None)
            expected_hash = recorded_file_hash(nested_entry, "sha256")
        if expected_hash is None:
            expected_hash = first(row, key + "_sha256")
        artifact = artifacts[key]
        if isinstance(artifact, dict):
            artifact["recorded_sha256"] = expected_hash
            artifact["hash_matches_record"] = compare_hash(artifact.get("sha256"), expected_hash)

    gait_event_artifact = artifacts.get("gait_events")
    if isinstance(gait_event_artifact, dict):
        gait_event_expected_hash = recorded_file_hash(event_entry, "sha256")
        gait_event_artifact["recorded_sha256"] = gait_event_expected_hash
        gait_event_artifact["hash_matches_record"] = compare_hash(
            gait_event_artifact.get("sha256"),
            gait_event_expected_hash,
        )

    protocol_artifact = artifacts.get("protocol")
    protocol_file_sha256 = (
        protocol_artifact.get("sha256") if isinstance(protocol_artifact, dict) else None
    )
    protocol_canonical_sha256 = (
        canonical_json_sha256(protocol_document) if protocol_document else None
    )
    protocol_file_sha256_from_cohort = recorded_file_hash(cohort_manifest.get("protocol"), "sha256")
    protocol_canonical_sha256_from_freeze = first(source_freeze_document, "protocol_sha256")
    if isinstance(protocol_artifact, dict):
        protocol_artifact.update({
            "protocol_file_sha256": protocol_file_sha256,
            "protocol_canonical_sha256": protocol_canonical_sha256,
            "recorded_file_sha256_from_cohort_manifest": protocol_file_sha256_from_cohort,
            "recorded_canonical_sha256_from_source_freeze": protocol_canonical_sha256_from_freeze,
            "file_hash_matches_cohort_manifest": compare_hash(
                protocol_file_sha256,
                protocol_file_sha256_from_cohort,
            ),
            "canonical_hash_matches_source_freeze": compare_hash(
                protocol_canonical_sha256,
                protocol_canonical_sha256_from_freeze,
            ),
        })

    status = first(summary, "status", default=first(record, "status", default=first(manifest, "status", default=first(row, "status", default="missing"))))
    closed_loop = first(summary, "closed_loop", default=first(record, "closed_loop", default=first(manifest, "closed_loop", default=False)))
    no_neural = first(row, "no_neural", "no_neural_input", default=first(summary, "no_neural", "no_neural_input", default=first(record, "no_neural", "no_neural_input", default=first(manifest, "no_neural", "no_neural_input", default=record.get("neural_forward_called") is False))))
    no_vision = first(row, "no_vision", "no_visual_input", default=first(summary, "no_vision", "no_visual_input", default=first(record, "no_vision", "no_visual_input", default=first(manifest, "no_vision", "no_visual_input", default=False))))
    same_initial = first(summary, "initial_state_and_phase_matched", "shared_initial_state_and_phase", default=first(record, "initial_state_and_phase_matched", "shared_initial_state_and_phase", default=first(manifest, "initial_state_and_phase_matched", "shared_initial_state_and_phase", default=first(row, "initial_state_and_phase_matched", "shared_initial_state_and_phase"))))
    authority = first(summary, "authority", default=first(record, "authority", default=first(manifest, "authority", default={})))
    if not isinstance(authority, dict):
        authority = {}
    expected_case = next((item for item in expected.get("conditions", []) if item.get("case_id") == cid), {})
    expected_counts = expected
    run_window_summary = record.get("window_summary", {}) if isinstance(record.get("window_summary"), dict) else {}
    root_qpos_writes = first(authority, "root_qpos_write_count", "root_qpos_writes", "root_qpos_write_count_max", default=first(run_window_summary, "root_qpos_write_count_max", default=first(record, "root_state_writes_after_initialization", "root_qpos_write_count", "root_qpos_writes")))
    root_qvel_writes = first(authority, "root_qvel_write_count", "root_qvel_writes", "root_qvel_write_count_max", default=first(run_window_summary, "root_qvel_write_count_max", default=first(record, "root_state_writes_after_initialization", "root_qvel_write_count", "root_qvel_writes")))
    external_peaks = [
        first(authority, "root_external_wrench_peak", "root_external_wrench_peak_abs", "root_external_wrench_peak_native", "root_external_wrench_max_abs_native", "root_external_wrench_max_abs", default=first(run_window_summary, "root_external_wrench_peak_native", default=first(record, "root_external_wrench_peak", "root_external_wrench_peak_abs"))),
        first(authority, "body_external_wrench_peak", "body_external_wrench_peak_abs", "body_external_wrench_peak_native", "body_external_wrench_max_abs_native", "body_external_wrench_max_abs", default=first(run_window_summary, "body_external_wrench_peak_native", default=first(record, "body_external_wrench_peak", "body_external_wrench_peak_abs"))),
    ]
    qfrc_peak = first(authority, "qfrc_applied_peak", "qfrc_applied_peak_abs", "generalized_applied_force_peak_native", "qfrc_applied_max_abs_native", "qfrc_applied_max_abs", default=first(run_window_summary, "generalized_applied_force_peak_native", default=first(record, "qfrc_applied_peak", "qfrc_applied_peak_abs")))
    qpos_qvel_zero = root_qpos_writes == 0 and root_qvel_writes == 0
    external_wrenches_zero = len(external_peaks) == 2 and all(max_abs_numeric(value) == 0.0 for value in external_peaks)
    qfrc_zero = max_abs_numeric(qfrc_peak) == 0.0
    expected_window = expected.get("analysis_window_s")
    actual_window = first(record, "analysis_window_s", default=first(summary, "analysis_window_s", default=first(manifest, "analysis_window_s")))
    actual_duration = numeric(first(record, "duration_s", default=first(record.get("body_config", {}), "duration_s", default=first(summary, "duration_s", default=first(manifest, "duration_s")))))
    actual_seed = first(record, "seed", default=first(record.get("body_config", {}), "seed", default=first(record.get("condition", {}), "seed", default=first(summary, "seed", default=first(manifest, "seed", default=row.get("seed"))))))
    qpos_hash = first(record, "matched_initial_qpos_sha256", "initial_qpos_sha256", default=first(run_window_summary, "initial_qpos_sha256", default=first(record.get("initial_state", {}), "qpos_sha256")))
    qvel_hash = first(record, "matched_initial_qvel_sha256", "initial_qvel_sha256", default=first(run_window_summary, "initial_qvel_sha256", default=first(record.get("initial_state", {}), "qvel_sha256")))
    phase_hash = first(record, "cpg_phase_trace_sha256", "initial_cpg_phase_sha256", default=first(run_window_summary, "cpg_phase_trace_sha256"))
    record_condition = record.get("condition", {}) if isinstance(record.get("condition"), dict) else {}
    recorded_p = numeric(first(run_window_summary, "P", default=first(record, "P", "constant_P")))
    if recorded_p is None:
        power_pair = first(record_condition, "P_left_right")
        if isinstance(power_pair, list) and len(power_pair) == 2 and numeric(power_pair[0]) == numeric(power_pair[1]):
            recorded_p = numeric(power_pair[0])
    recorded_d = numeric(first(run_window_summary, "d", default=first(record, "d", default=first(record_condition, "differential_ap_d"))))
    yaw_deg = first(run_window_summary, "thorax_heading_yaw_change_deg_0p5_3s", default=first(record, "thorax_heading_yaw_change_deg_0p5_3s", "body_yaw_change_deg_0p5_3s"))
    yaw_rad = first(run_window_summary, "thorax_heading_yaw_change_rad_0p5_3s", "thorax_heading_yaw_change_rad_analysis", default=first(record, "thorax_heading_yaw_change_rad_0p5_3s", "body_yaw_change_rad_0p5_3s"))
    record_metrics = {
        "thorax_heading_yaw_change_deg_0p5_3s": yaw_deg,
        "body_yaw_change_deg": yaw_deg,
        "body_yaw_change_rad": yaw_rad,
        "loaded_fz_support_fraction_by_leg": first(run_window_summary, "loaded_fraction_by_leg", default=first(record, "loaded_fz_support_fraction_by_leg")),
        "at_least_two_loaded_legs_fraction": first(run_window_summary, "fraction_samples_with_at_least_two_loaded_legs", "fraction_samples_with_at_least_two_loaded_legs", default=first(record, "at_least_two_loaded_legs_fraction", "fraction_samples_with_at_least_two_loaded_legs")),
        "thorax_ground_contact_count": first(run_window_summary, "thorax_ground_contact_samples", default=first(record, "thorax_ground_contact_count", "thorax_contact_count")),
        "nonleg_ground_contact_count": first(run_window_summary, "nonleg_ground_contact_samples", default=first(record, "nonleg_ground_contact_count", "nonleg_contact_count")),
        "analysis_window_s_half_open": expected_window,
    }
    analysis_window_key = next((name for name in windows if "0p5_3" in name or "0.5_3" in name or name == "analysis_0p5_3"), "analysis_0p5_3")
    existing_analysis = windows.get(analysis_window_key)
    if not isinstance(existing_analysis, dict):
        existing_analysis = normalize_mechanics_window(analysis_window_key, record_metrics, merged_condition)
        windows[analysis_window_key] = existing_analysis
    else:
        target_metrics = existing_analysis.get("mechanics")
        if isinstance(target_metrics, dict):
            fallback_metrics = normalize_mechanics_window(analysis_window_key, record_metrics, merged_condition)["mechanics"]
            for key, value in fallback_metrics.items():
                if target_metrics.get(key) is None and value is not None:
                    target_metrics[key] = value
    checks = {
        "status_complete": status == "complete",
        "control_ticks_match": counts["control_ticks"] == expected_counts.get("expected_control_ticks"),
        "physics_steps_match": counts["physics_steps"] == expected_counts.get("expected_physics_steps"),
        "native_samples_match": counts["native_samples"] == expected_counts.get("expected_native_samples"),
        "no_neural_input": no_neural is True,
        "no_vision_input": no_vision is True,
        "closed_loop_false": closed_loop is False,
        "initial_state_and_phase_matched": same_initial is True,
        "matched_initial_qpos_hash_present": isinstance(qpos_hash, str) and bool(qpos_hash),
        "matched_initial_qvel_hash_present": isinstance(qvel_hash, str) and bool(qvel_hash),
        "matched_phase_hash_present": isinstance(phase_hash, str) and bool(phase_hash),
        "analysis_window_matches_contract": actual_window == expected_window,
        "duration_matches_contract": actual_duration == numeric(expected.get("duration_s")),
        "seed_matches_contract": actual_seed == expected.get("seed"),
        "trajectory_hash_recorded": artifacts.get("trajectory", {}).get("sha256") is not None,
        "contacts_hash_recorded": artifacts.get("contacts", {}).get("sha256") is not None,
        "source_freeze_present": artifacts.get("source_freeze", {}).get("exists") is True,
        "source_freeze_hash_matches_record": artifacts.get("source_freeze", {}).get("hash_matches_record") is True,
        "protocol_file_hash_matches_cohort_manifest": artifacts.get("protocol", {}).get("file_hash_matches_cohort_manifest") is True,
        "protocol_canonical_hash_matches_source_freeze": artifacts.get("protocol", {}).get("canonical_hash_matches_source_freeze") is True,
        "analysis_manifest_hash_matches_cohort_manifest": artifacts.get("analysis_manifest", {}).get("hash_matches_cohort_manifest") is True,
        "condition_matches_contract": (
            recorded_p == numeric(expected.get("constant_P"))
            and recorded_d == numeric(expected_case.get("d"))
        ),
        "cohort_manifest_complete": cohort_manifest.get("status") == "complete",
        "gait_event_hash_matches_manifest": artifacts.get("gait_events", {}).get("hash_matches_record") is True,
        "root_qpos_qvel_writes_zero": qpos_qvel_zero,
        "external_wrenches_zero": external_wrenches_zero,
        "generalized_applied_force_zero": qfrc_zero,
        "authority_limits_match": qpos_qvel_zero and external_wrenches_zero and qfrc_zero,
    }
    field_pending = [name for name, passed in checks.items() if not passed]
    yaw_fields = {key: value for source in (record, run_window_summary, summary, manifest) for key, value in source.items() if "yaw" in key.lower() or "heading" in key.lower() or "loaded_fz" in key.lower()}
    event_per_leg = event_summary.get("per_leg", {}) if isinstance(event_summary.get("per_leg"), dict) else {}
    gait_event_summary = {
        "body_yaw": event_summary.get("body_yaw"),
        "six_leg_measured_cycle_gate": event_summary.get("six_leg_measured_cycle_gate"),
        "per_leg": {
            leg: {
                "complete_lift_touchdown_events_analysis_window": first(first(metrics, "event_counts", default={}), "complete_lift_touchdown_events_analysis_window"),
                "phase_associated_complete_events_analysis_window": first(first(metrics, "event_counts", default={}), "phase_associated_complete_events_analysis_window"),
                "startup_events_0_to_0p5s": first(first(metrics, "event_counts", default={}), "startup_events_0_to_0p5s"),
                "terminal_or_incomplete_event_tails": first(first(metrics, "event_counts", default={}), "terminal_or_incomplete_event_tails"),
            }
            for leg, metrics in event_per_leg.items() if isinstance(metrics, dict)
        },
        "interpretation_limit": "single-seed engineering event report; not a biological walking or neural gait acceptance result",
    }
    return {
        "case_id": cid,
        "cohort": "mechanics",
        "assay_group": "mechanics-yaw-calibration",
        "condition": merged_condition,
        "status": status,
        "closed_loop": bool(closed_loop),
        "closed_loop_basis": "independent free-ground mechanical run; no neural or visual input",
        "neural_input": bool(not no_neural),
        "vision_input": bool(not no_vision),
        "counts": counts,
        "hashes": artifacts,
        "matched_initial_state_and_phase_hashes": {
            "qpos_sha256": qpos_hash,
            "qvel_sha256": qvel_hash,
            "cpg_phase_sha256": phase_hash,
        },
        "windows": windows,
        "yaw_summary_fields": yaw_fields,
        "gait_event_summary": gait_event_summary,
        "checks": checks,
        "fields_pending": field_pending,
        "complete_for_engineering": all(checks.values()),
        "run_record": record if record else None,
        "case_manifest": manifest if manifest else None,
        "integrity_record": integrity if integrity else None,
    }


def get_number(case: dict[str, Any], window: str, section: str, field: str) -> float | None:
    value = case.get("windows", {}).get(window, {}).get(section, {}).get(field)
    return numeric(value)


def make_comparisons(cases: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in cases}
    comparisons: list[dict[str, Any]] = []
    rules = config.get("comparisons", [])
    for rule in rules if isinstance(rules, list) else []:
        if not isinstance(rule, dict):
            continue
        group = rule.get("assay_group")
        baseline_id = rule.get("baseline_case_id")
        baseline = by_id.get(baseline_id)
        targets = rule.get("case_ids", [])
        if not isinstance(targets, list):
            continue
        for target_id in targets:
            target = by_id.get(target_id)
            if target is None or baseline is None:
                comparisons.append({
                    "assay_group": group,
                    "baseline_case_id": baseline_id,
                    "case_id": target_id,
                    "status": "missing_case",
                })
                continue
            window_names = sorted(set(baseline.get("windows", {})) & set(target.get("windows", {})))
            window_deltas: dict[str, Any] = {}
            for window in window_names:
                base_section = "neural" if target["cohort"] == "neural" else "mechanics"
                target_section = base_section
                metrics = (
                    ["raw_P", "control_power", "static_gate_fraction", "raw_DNp15_L_Hz", "raw_DNp15_R_Hz", "raw_opponent_signal", "ungated_signed_d", "gated_d"]
                    if target["cohort"] == "neural"
                    else ["constant_P", "d", "body_yaw_change_deg", "body_yaw_change_rad", "at_least_two_loaded_legs_fraction"]
                )
                deltas: dict[str, Any] = {}
                for metric in metrics:
                    b = get_number(baseline, window, base_section, metric)
                    t = get_number(target, window, target_section, metric)
                    if b is not None and t is not None:
                        deltas[metric] = {"baseline": b, "target": t, "delta_target_minus_baseline": t - b}
                if target["cohort"] == "mechanics":
                    for metric in ("per_leg_contact",):
                        base_values = baseline.get("windows", {}).get(window, {}).get("mechanics", {}).get(metric)
                        target_values = target.get("windows", {}).get(window, {}).get("mechanics", {}).get(metric)
                        if isinstance(base_values, list) and isinstance(target_values, list) and len(base_values) == len(target_values):
                            deltas[metric] = {
                                "baseline": base_values,
                                "target": target_values,
                                "delta_target_minus_baseline": [float(t) - float(b) for b, t in zip(base_values, target_values)],
                            }
                else:
                    base_contact = baseline.get("windows", {}).get(window, {}).get("ball", {}).get("positive_fz_fraction_by_leg")
                    target_contact = target.get("windows", {}).get(window, {}).get("ball", {}).get("positive_fz_fraction_by_leg")
                    if isinstance(base_contact, list) and isinstance(target_contact, list) and len(base_contact) == len(target_contact):
                        deltas["positive_fz_fraction_by_leg"] = {
                            "baseline": base_contact,
                            "target": target_contact,
                            "delta_target_minus_baseline": [float(t) - float(b) for b, t in zip(base_contact, target_contact)],
                        }
                if target["cohort"] == "neural":
                    base_omega = baseline.get("windows", {}).get(window, {}).get("ball", {}).get("omega_world_mean_rad_s")
                    target_omega = target.get("windows", {}).get(window, {}).get("ball", {}).get("omega_world_mean_rad_s")
                    if isinstance(base_omega, list) and isinstance(target_omega, list) and len(base_omega) == len(target_omega):
                        deltas["ball_omega_world_rad_s"] = {
                            "baseline": base_omega,
                            "target": target_omega,
                            "delta_target_minus_baseline": [float(t) - float(b) for b, t in zip(base_omega, target_omega)],
                        }
                window_deltas[window] = deltas
            comparisons.append({
                "assay_group": group,
                "baseline_case_id": baseline_id,
                "case_id": target_id,
                "status": "descriptive_only",
                "interpretation": "single-seed engineering difference; not a statistical or biological effect",
                "windows": window_deltas,
            })
    return comparisons


def make_within_case_gray_deltas(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scalar_metrics = ("raw_P", "control_power", "raw_DNp15_L_Hz", "raw_DNp15_R_Hz", "raw_opponent_signal", "ungated_signed_d", "gated_d", "static_gate_fraction")
    for case in cases:
        if case.get("cohort") != "neural":
            continue
        baseline = case.get("windows", {}).get("gray_0_1", {})
        baseline_neural = baseline.get("neural", {}) if isinstance(baseline, dict) else {}
        window_deltas: dict[str, Any] = {}
        for window_name in ("early_1_1p5", "late_2_4"):
            target = case.get("windows", {}).get(window_name, {})
            target_neural = target.get("neural", {}) if isinstance(target, dict) else {}
            deltas: dict[str, Any] = {}
            for metric in scalar_metrics:
                base_value = numeric(baseline_neural.get(metric))
                target_value = numeric(target_neural.get(metric))
                if base_value is not None and target_value is not None:
                    deltas[metric] = {
                        "gray_baseline": base_value,
                        "window_value": target_value,
                        "delta_window_minus_gray": target_value - base_value,
                    }
            base_ball = baseline.get("ball", {}).get("omega_world_mean_rad_s") if isinstance(baseline, dict) else None
            target_ball = target.get("ball", {}).get("omega_world_mean_rad_s") if isinstance(target, dict) else None
            if isinstance(base_ball, list) and isinstance(target_ball, list) and len(base_ball) == len(target_ball):
                deltas["ball_omega_world_rad_s"] = {
                    "gray_baseline": base_ball,
                    "window_value": target_ball,
                    "delta_window_minus_gray": [float(t) - float(b) for b, t in zip(base_ball, target_ball)],
                }
            base_contact = baseline.get("ball", {}).get("positive_fz_fraction_by_leg") if isinstance(baseline, dict) else None
            target_contact = target.get("ball", {}).get("positive_fz_fraction_by_leg") if isinstance(target, dict) else None
            if isinstance(base_contact, list) and isinstance(target_contact, list) and len(base_contact) == len(target_contact):
                deltas["positive_fz_fraction_by_leg"] = {
                    "gray_baseline": base_contact,
                    "window_value": target_contact,
                    "delta_window_minus_gray": [float(t) - float(b) for b, t in zip(base_contact, target_contact)],
                }
            window_deltas[window_name] = deltas
        results.append({"case_id": case.get("case_id"), "assay_group": case.get("assay_group"), "baseline_window": "gray_0_1", "windows": window_deltas})
    return results


def build_markdown(result: dict[str, Any]) -> str:
    counts = result.get("counts", {})
    lines = [
        "# Standardized suite v1 summary",
        "",
        f"- Suite: `{result.get('suite_id')}`",
        f"- Status: **{result.get('status')}**",
        f"- Cases: {counts.get('complete', 0)}/{counts.get('expected_total', 0)} complete "
        f"({counts.get('neural_complete', 0)}/{counts.get('expected_neural', 0)} neural; "
        f"{counts.get('mechanics_complete', 0)}/{counts.get('expected_mechanics', 0)} mechanics)",
        "- Evidence class: `digital_simulation`; one-seed engineering regression only.",
        "- Natural-behavior and statistical-validation acceptance: **not evaluated**.",
        "",
        "## Case status",
        "",
        "| Cohort | Group | Case | Status | Complete for engineering | Closed-loop |",
        "|---|---|---|---|---:|---:|",
    ]
    for case in result.get("cases", []):
        lines.append(
            f"| {case.get('cohort')} | {case.get('assay_group') or '—'} | `{case.get('case_id')}` "
            f"| {case.get('status')} | {case.get('complete_for_engineering')} | {case.get('closed_loop')} |"
        )
    lines.extend(("", "## Descriptive control comparisons", ""))
    comparisons = result.get("comparisons", [])
    if not comparisons:
        lines.append("No configured baseline comparison could be computed.")
    else:
        lines.extend(("| Group | Base | Case | Status |", "|---|---|---|---|"))
        for comparison in comparisons:
            lines.append(
                f"| {comparison.get('assay_group')} | `{comparison.get('baseline_case_id')}` "
                f"| `{comparison.get('case_id')}` | {comparison.get('status')} |"
            )
    lines.extend((
        "",
        "The JSON output contains per-window raw `P`, final control power, raw DNp15 rates, raw opponent signal, "
        "ungated and gated `d`, static-gate activity, ball world angular velocity, contact fractions, counts, and hashes where recorded.",
        "Ball angular velocity is a tethered mechanics readout, not free-fly yaw. Contact fractions do not establish complete gait cycles.",
        "Historical ten-second stripe windows are not pooled with the four-second v1 schedule.",
        "",
    ))
    return "\n".join(lines)


def summarize(bundle: Path, input_manifest_path: Path, suite_manifest_path: Path, config_path: Path) -> dict[str, Any]:
    input_manifest = read_json(input_manifest_path)
    suite_manifest = read_json(suite_manifest_path) if suite_manifest_path.is_file() else {}
    config = read_json(config_path)
    if not isinstance(input_manifest, dict) or not isinstance(config, dict):
        raise ValueError("input manifest and acceptance config must be JSON objects")
    if not isinstance(suite_manifest, dict):
        suite_manifest = {}
    input_rows = manifest_cases(input_manifest, "cases")
    suite_rows = manifest_cases(suite_manifest, "cases")
    neural_rows = index_cases(input_rows)
    for row in suite_rows:
        cid = case_id(row)
        if cid and cid not in neural_rows:
            neural_rows[cid] = row
        elif cid:
            neural_rows[cid] = {**neural_rows[cid], **row}
    for row in neural_rows.values():
        row["_manifest_dir"] = str(input_manifest_path.parent)
        row["_input_manifest_path"] = str(input_manifest_path)
        row["_suite_assay_id"] = first(
            row,
            "assay_id",
            default=first(input_manifest, "assay_id", default="standardized-repair-v1"),
        )
    expected = {**config.get("neural_contract", {}), "_readout_contract": config.get("readout_contract", {})}
    neural = [normalize_neural_case(bundle, row, suite_group_for(suite_manifest, row), expected) for row in neural_rows.values()]
    mech_rows = manifest_cases(suite_manifest, "mechanics_cases") or manifest_cases(input_manifest, "mechanics_cases")
    mechanics_contract = config.get("mechanics_contract", {})
    mechanics = [normalize_mechanics_case(bundle, row, mechanics_contract) for row in mech_rows]
    mechanics_cohort_manifest = read_optional_json(bundle / "mechanics" / "cohort-manifest.json")
    mechanics_cohort_manifest = mechanics_cohort_manifest if isinstance(mechanics_cohort_manifest, dict) else {}
    mechanics_analysis_manifest_ref = mechanics_cohort_manifest.get("analysis_manifest")
    mechanics_analysis_manifest_path = resolve_path(first(mechanics_analysis_manifest_ref, "path"), bundle)
    if mechanics_analysis_manifest_path is None:
        mechanics_analysis_manifest_path = bundle / "mechanics" / "analysis" / "manifest.json"
    mechanics_analysis_summary_path = mechanics_analysis_manifest_path.parent / "summary.json"
    mechanics_analysis_summary = read_optional_json(mechanics_analysis_summary_path)
    mechanics_analysis_manifest = read_optional_json(mechanics_analysis_manifest_path)
    cases = neural + mechanics
    expected_counts = config.get("expected_counts", {})
    expected_neural = int(expected_counts.get("neural_cases", 0))
    expected_mechanics = int(expected_counts.get("mechanics_cases", 0))
    expected_total = int(expected_counts.get("total_cases", expected_neural + expected_mechanics))
    complete_neural = sum(case.get("complete_for_engineering", False) for case in neural)
    complete_mechanics = sum(case.get("complete_for_engineering", False) for case in mechanics)
    complete = complete_neural + complete_mechanics
    missing_expected = len(neural) != expected_neural or len(mechanics) != expected_mechanics or len(cases) != expected_total
    neural_freeze_ids = sorted({case.get("core_freeze_sha256") for case in neural if case.get("core_freeze_sha256")})
    neural_freeze_consistent = len(neural) == expected_neural and len(neural_freeze_ids) == 1 and all(case.get("core_freeze_sha256") for case in neural)
    neural_repair_freeze_ids = sorted({case.get("repair_freeze_sha256") for case in neural if case.get("repair_freeze_sha256")})
    neural_repair_freeze_consistent = len(neural) == expected_neural and len(neural_repair_freeze_ids) == 1 and all(case.get("repair_freeze_sha256") for case in neural)
    mechanics_freeze_ids = sorted({case.get("hashes", {}).get("source_freeze", {}).get("sha256") for case in mechanics if case.get("hashes", {}).get("source_freeze", {}).get("sha256")})
    mechanics_freeze_consistent = len(mechanics) == expected_mechanics and len(mechanics_freeze_ids) == 1 and all(case.get("hashes", {}).get("source_freeze", {}).get("sha256") for case in mechanics)
    qpos_hashes = {case.get("matched_initial_state_and_phase_hashes", {}).get("qpos_sha256") for case in mechanics}
    qvel_hashes = {case.get("matched_initial_state_and_phase_hashes", {}).get("qvel_sha256") for case in mechanics}
    phase_hashes = {case.get("matched_initial_state_and_phase_hashes", {}).get("cpg_phase_sha256") for case in mechanics}
    mechanics_pairing_consistent = (
        len(mechanics) == expected_mechanics
        and len(qpos_hashes) == len(qvel_hashes) == len(phase_hashes) == 1
        and None not in qpos_hashes
        and None not in qvel_hashes
        and None not in phase_hashes
    )
    status = "complete" if complete == expected_total and not missing_expected and neural_freeze_consistent and neural_repair_freeze_consistent and mechanics_freeze_consistent and mechanics_pairing_consistent else "incomplete_or_mismatch"
    comparisons = make_comparisons(cases, config)
    within_case_gray_deltas = make_within_case_gray_deltas(cases)
    return {
        "schema_version": "flygym-standardized-suite-summary-v1",
        "suite_id": config.get("suite_id"),
        "evidence_class": config.get("evidence_class"),
        "classification": config.get("classification"),
        "status": status,
        "counts": {
            "expected_neural": expected_neural,
            "expected_mechanics": expected_mechanics,
            "expected_total": expected_total,
            "neural_present": len(neural),
            "mechanics_present": len(mechanics),
            "present_total": len(cases),
            "neural_complete": complete_neural,
            "mechanics_complete": complete_mechanics,
            "complete": complete,
        },
        "freeze_and_pairing_review": {
            "neural_core_freeze_ids": neural_freeze_ids,
            "neural_core_freeze_consistent": neural_freeze_consistent,
            "neural_repair_freeze_ids": neural_repair_freeze_ids,
            "neural_repair_freeze_consistent": neural_repair_freeze_consistent,
            "mechanics_source_freeze_ids": mechanics_freeze_ids,
            "mechanics_source_freeze_consistent": mechanics_freeze_consistent,
            "mechanics_initial_state_and_phase_matched": mechanics_pairing_consistent,
        },
        "manifest_hashes": {
            "input_manifest": path_record(input_manifest_path),
            "suite_manifest": path_record(suite_manifest_path),
            "acceptance_config": path_record(config_path),
        },
        "manifest_status": {
            "input_manifest": first(input_manifest, "status"),
            "suite_manifest": first(suite_manifest, "status"),
        },
        "historical_mechanics_reference_only": first(
            suite_manifest, "historical_mechanics_reference", "historical_reference"
        ),
        "mechanics_analysis": {
            "summary": path_record(mechanics_analysis_summary_path),
            "manifest": path_record(mechanics_analysis_manifest_path),
            "results": mechanics_analysis_summary if isinstance(mechanics_analysis_summary, dict) else None,
            "manifest_data": mechanics_analysis_manifest if isinstance(mechanics_analysis_manifest, dict) else None,
        },
        "acceptance_boundary": {
            "engineering_regression_only": True,
            "statistical_validation": False,
            "natural_behavior_success_claim_allowed": False,
            "neural_closed_loop": False,
            "mechanics_independent": True,
        },
        "cases": sorted(cases, key=lambda case: (case.get("cohort", ""), case.get("assay_group") or "", case.get("case_id") or "")),
        "comparisons": comparisons,
        "within_case_gray_deltas": within_case_gray_deltas,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="standardized bundle root")
    parser.add_argument("--input-manifest", type=Path, help="defaults to <bundle>/input-manifest.json")
    parser.add_argument("--suite-manifest", type=Path, help="defaults to <bundle>/suite-manifest.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="acceptance policy JSON")
    parser.add_argument("--output-json", type=Path, help="defaults to <bundle>/analysis/standardized-suite-summary.json")
    parser.add_argument("--output-md", type=Path, help="defaults to <bundle>/analysis/standardized-suite-summary.md")
    parser.add_argument("--strict", action="store_true", help="exit 2 if any expected case or engineering check is incomplete")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    input_manifest = (args.input_manifest or (bundle / "input-manifest.json")).expanduser().resolve()
    suite_manifest = (args.suite_manifest or (bundle / "suite-manifest.json")).expanduser().resolve()
    config = args.config.expanduser().resolve()
    output_json = (args.output_json or (bundle / "analysis" / "standardized-suite-summary.json")).expanduser().resolve()
    output_md = (args.output_md or (bundle / "analysis" / "standardized-suite-summary.md")).expanduser().resolve()
    try:
        result = summarize(bundle, input_manifest, suite_manifest, config)
        dump_json(output_json, result)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(result), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False))
    if args.strict and result["status"] != "complete":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
