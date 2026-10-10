#!/usr/bin/env python3
"""Compare recorded reduced-body and articulated optomotor stripe runs.

The analyzer is read-only with respect to run inputs.  It streams telemetry,
checks the run protocol and declared output digests, and writes a report plus a
standalone PNG into a new output directory.  A seed is a deterministic digital
simulation replicate; frames are never treated as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "articulated-optomotor-comparison-1"
EXPECTED_DURATION_S = 7.5
EXPECTED_DT_S = 0.005
EXPECTED_FRAMES = 1500
CONDITIONS = ("left", "right", "static")
LEGS = ("fl/L", "fl/R", "ml/L", "ml/R", "hl/L", "hl/R")
LEG_POOLS = tuple(f"superclass:vnc_motor&subclass:{leg}" for leg in LEGS)
HASH_CHUNK_BYTES = 1024 * 1024
MIN_SEEDS_FOR_TEST = 5
REPEAT_SUFFIX = re.compile(r"(?:^|[-_])repeat[-_]?([0-9]+)$", re.IGNORECASE)


class AnalysisError(ValueError):
    """Raised when an input run is incomplete or violates the comparison protocol."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{where} must be a JSON object")
    return value


def _number(value: Any, where: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise AnalysisError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise AnalysisError(f"{where} must be {qualifier}")
    return result


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{where} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{where} must be an integer") from exc
    if result != value:
        raise AnalysisError(f"{where} must be an integer")
    return result


def _json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {where} at {path}: {exc}") from exc
    return dict(_mapping(value, where))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _normalize_condition(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise AnalysisError(f"{where} is missing a stimulus condition")
    normalized = value.lower().strip()
    if normalized.startswith("stripe_"):
        normalized = normalized.removeprefix("stripe_")
    if normalized not in CONDITIONS:
        raise AnalysisError(f"{where} has unsupported stimulus condition {value!r}")
    return normalized


def _read_seed(manifest: Mapping[str, Any], where: str) -> int:
    protocol = manifest.get("protocol")
    protocol_seed = protocol.get("seed") if isinstance(protocol, Mapping) else None
    candidates = (manifest.get("seed"), protocol_seed)
    for value in candidates:
        if value is not None:
            return _integer(value, f"{where}.seed")
    raise AnalysisError(f"{where} is missing the trial seed")


def _condition(manifest: Mapping[str, Any], where: str) -> str:
    candidates: list[Any] = [manifest.get("condition"), manifest.get("stimulus_condition")]
    protocol = manifest.get("protocol")
    if isinstance(protocol, Mapping):
        candidates.extend((protocol.get("condition"), protocol.get("stimulus_condition")))
    scenario_results = manifest.get("scenario_results")
    if isinstance(scenario_results, Mapping):
        free = scenario_results.get("free-behavior")
        if isinstance(free, Mapping):
            metrics = free.get("metrics")
            if isinstance(metrics, Mapping):
                candidates.append(metrics.get("stimulus_condition"))
    for candidate in candidates:
        if candidate is not None:
            return _normalize_condition(candidate, where)
    raise AnalysisError(f"{where} is missing a stimulus condition")


def _mode(manifest: Mapping[str, Any]) -> str:
    for key in ("sensory_mode", "feedback_mode", "mode"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value.lower()
    for parent_key, child_key in (("sensory", "mode"), ("sensory_feedback", "mode"),
                                  ("intervention", "mode"),
                                  ("protocol", "sensory_mode")):
        parent = manifest.get(parent_key)
        if isinstance(parent, Mapping):
            value = parent.get(child_key)
            if isinstance(value, str) and value:
                return value.lower()
    return "unspecified"


def _status(manifest: Mapping[str, Any]) -> str | None:
    for key in ("status", "run_status", "outcome"):
        value = manifest.get(key)
        if isinstance(value, str):
            return value.strip().upper()
    return None


def _repeat_index(manifest: Mapping[str, Any], run_dir: Path) -> int | None:
    for key in ("repeat_index", "replicate_index"):
        value = manifest.get(key)
        if value is not None:
            result = _integer(value, f"{run_dir}.{key}")
            if result <= 0:
                raise AnalysisError(f"{run_dir}.{key} must be positive")
            return result
    repeat = manifest.get("repeat")
    if isinstance(repeat, Mapping) and repeat.get("index") is not None:
        result = _integer(repeat["index"], f"{run_dir}.repeat.index")
        if result <= 0:
            raise AnalysisError(f"{run_dir}.repeat.index must be positive")
        return result
    match = REPEAT_SUFFIX.search(run_dir.name)
    return int(match.group(1)) if match else None


def _duration_contract(manifest: Mapping[str, Any], where: str) -> tuple[float, float, int]:
    duration = dt = frames = None
    protocol = manifest.get("protocol")
    if isinstance(protocol, Mapping):
        duration = protocol.get("duration_s", duration)
        dt = protocol.get("frame_dt_s", protocol.get("flyvis_dt_s",
                             protocol.get("dt_s", dt)))
        frames = protocol.get("frame_count", protocol.get(
            "frames", protocol.get("flyvis_frames", frames)))
    for parent_key in ("run", "simulation", "assay"):
        parent = manifest.get(parent_key)
        if isinstance(parent, Mapping):
            duration = parent.get("duration_s", duration)
            dt = parent.get("frame_dt_s", parent.get("dt_s", dt))
            frames = parent.get("frame_count", parent.get("frames", frames))
    duration = manifest.get("duration_s", duration)
    dt = manifest.get("frame_dt_s", manifest.get("dt_s", dt))
    frames = manifest.get("frame_count", manifest.get("frames", frames))
    scenario_results = manifest.get("scenario_results")
    if isinstance(scenario_results, Mapping):
        free = scenario_results.get("free-behavior")
        if isinstance(free, Mapping):
            metrics = free.get("metrics")
            if isinstance(metrics, Mapping):
                duration = metrics.get("duration_s", duration)
                dt = metrics.get("dt_s", dt)
                frames = metrics.get("frame_count", metrics.get("sample_count", frames))
    if duration is None or dt is None or frames is None:
        raise AnalysisError(f"{where} does not declare duration, output dt, and frame count")
    duration_value = _number(duration, f"{where}.duration_s", nonnegative=True)
    dt_value = _number(dt, f"{where}.frame_dt_s", nonnegative=True)
    frame_value = _integer(frames, f"{where}.frame_count")
    if (not math.isclose(duration_value, EXPECTED_DURATION_S, abs_tol=1e-8, rel_tol=0)
            or not math.isclose(dt_value, EXPECTED_DT_S, abs_tol=1e-10, rel_tol=0)
            or frame_value != EXPECTED_FRAMES):
        raise AnalysisError(
            f"{where} must be {EXPECTED_DURATION_S:g}s at {EXPECTED_DT_S:g}s with "
            f"{EXPECTED_FRAMES} frames; found {duration_value:g}s, {dt_value:g}s, "
            f"{frame_value} frames")
    return duration_value, dt_value, frame_value


def _cuda_and_protocol(manifest: Mapping[str, Any], model: str, where: str) -> None:
    """Fail closed on a wrong assay, non-CUDA backend, or a reported fallback."""
    schema = manifest.get("schema_version")
    if schema == "digital-fly-run-1":
        if manifest.get("execution_policy") != "strict-free-behavior-escape-experiment":
            raise AnalysisError(f"{where} is not a strict free-behavior run")
        backend = _mapping(manifest.get("effective_backends"), f"{where}.effective_backends")
        expected_neural = "triton" if model == "historical_triton" else "torch"
        for key, expected in (("renderer", "warp_cuda"), ("vision", "flyvis"),
                              ("neural", expected_neural), ("autopilot", "prohibited")):
            actual = backend.get(key)
            acceptable = (actual in ("torch", "torch_cuda") if key == "neural"
                          and model != "historical_triton" else actual == expected)
            if not acceptable:
                raise AnalysisError(f"{where} effective_backends.{key} must be {expected!r}")
        if model != "historical_triton" and backend.get("body") != "flight_dynamics_reduced_6dof":
            raise AnalysisError(f"{where} is not the current reduced-body baseline")
        components = _mapping(manifest.get("components"), f"{where}.components")
        neural = _mapping(components.get("neural"), f"{where}.components.neural")
        if neural.get("is_stub") is True:
            raise AnalysisError(f"{where} reports a neural stub")
        runtime = neural.get("runtime")
        device = None
        if isinstance(runtime, Mapping):
            device = runtime.get("device")
        if device is None:
            device = neural.get("device")
        if not isinstance(device, str) or not device.lower().startswith("cuda"):
            raise AnalysisError(f"{where} does not prove actual CUDA neural execution")
        ground = _mapping(manifest.get("ground_walking"), f"{where}.ground_walking")
        config = _mapping(ground.get("config"), f"{where}.ground_walking.config")
        if (ground.get("no_behavior_policy") is not True
                or config.get("enabled") is not True or config.get("ground_only") is not True):
            raise AnalysisError(f"{where} does not match the reduced ground-crawl protocol")
        if (model == "historical_triton"
                and ground.get("evidence_class") != "reduced_contact_gated_neural_motor_actuator"):
            raise AnalysisError(f"{where} does not match the historical reduced ground-crawl assay")
        components = _mapping(manifest.get("components"), f"{where}.components")
        experiment = _mapping(components.get("experiment"), f"{where}.components.experiment")
        if experiment.get("name") != "strict-optomotor-ground-crawl-assay":
            raise AnalysisError(f"{where} is not the strict optomotor ground-crawl assay")
        scenario_results = _mapping(manifest.get("scenario_results"), f"{where}.scenario_results")
        free = _mapping(scenario_results.get("free-behavior"), f"{where}.free-behavior")
        metrics = _mapping(free.get("metrics"), f"{where}.free-behavior.metrics")
        if metrics.get("stimulus_interaction") != "visual_only_non_interacting":
            raise AnalysisError(f"{where} stimulus is not a visual-only non-interacting panel")
    elif schema == "articulated-optomotor-run-1":
        effective = _mapping(manifest.get("effective_backends"), f"{where}.effective_backends")
        neural = effective.get("neural")
        if not isinstance(neural, str) or "torch" not in neural.lower():
            raise AnalysisError(f"{where} is not the required Torch CUDA neural run")
        execution = _mapping(manifest.get("cuda_execution"), f"{where}.cuda_execution")
        if execution.get("verified") is not True:
            raise AnalysisError(f"{where} does not verify actual CUDA execution")
        if execution.get("fallback_permitted") is not False:
            raise AnalysisError(f"{where} does not prohibit backend fallback")
        if execution.get("fallback_used") is not False:
            raise AnalysisError(f"{where} does not explicitly record fallback_used=false")
        renderer_state = _mapping(execution.get("renderer"), f"{where}.cuda_execution.renderer")
        vision_state = _mapping(execution.get("vision"), f"{where}.cuda_execution.vision")
        neural_state = _mapping(execution.get("neural"), f"{where}.cuda_execution.neural")
        if renderer_state.get("effective") != "warp_cuda" or renderer_state.get(
                "actual_cuda_tensor_seen") is not True:
            raise AnalysisError(f"{where} does not verify an actual CUDA renderer tensor")
        if renderer_state.get("requested") != "warp_cuda":
            raise AnalysisError(f"{where} did not request a Warp CUDA renderer")
        vision_device = vision_state.get("device")
        neural_device = neural_state.get("device")
        if not isinstance(vision_device, str) or not vision_device.lower().startswith("cuda:"):
            raise AnalysisError(f"{where} FlyVis stage is not on an explicit CUDA device")
        if not isinstance(neural_device, str) or neural_device.lower() != "cuda":
            raise AnalysisError(f"{where} does not prove CUDA on the neural stage")
        if (vision_state.get("effective") != "flyvis"
                or neural_state.get("effective") != "torch_cuda"):
            raise AnalysisError(f"{where} did not execute FlyVis and Torch CUDA as requested")
        if (effective.get("renderer") != "warp_cuda"
                or effective.get("vision") != "flyvis"
                or effective.get("vision_device") != vision_device
                or effective.get("neural_device") != neural_device):
            raise AnalysisError(f"{where} renderer is not Warp CUDA")
        protocol = _mapping(manifest.get("protocol"), f"{where}.protocol")
        if protocol.get("stimulus_name") != "stripe_panel":
            raise AnalysisError(f"{where} does not use the recorded stripe panel stimulus")
        if (not math.isclose(_number(protocol.get("neural_gain"),
                                    f"{where}.protocol.neural_gain"), 1.2,
                            rel_tol=0, abs_tol=1e-6)
                or protocol.get("cns_dt_s") != 0.0005
                or protocol.get("cns_ticks_per_frame") != 10
                or protocol.get("stimulus_anchor") != "world"
                or protocol.get("synthetic_gait") is not False
                or protocol.get("body_traction_command") is not False
                or protocol.get("direct_motor_input") is not False):
            raise AnalysisError(f"{where} differs from the articulated optomotor protocol")
        feedback = _mapping(manifest.get("sensory_feedback"), f"{where}.sensory_feedback")
        mode = _mode(manifest)
        expected_legacy = (["jo", "haltere", "wing", "leg"] if mode == "legacy"
                           else ["jo", "haltere", "wing"])
        if feedback.get("legacy_groups_applied") != expected_legacy:
            raise AnalysisError(f"{where} applies unexpected legacy mechanosensory groups")
        expected_masked = ([] if mode == "legacy" else
                           ["leg", "leg_mechanosensory", "leg_tactile", "leg_proprioceptive"])
        if feedback.get("legacy_leg_aliases_masked") != expected_masked:
            raise AnalysisError(f"{where} does not record the expected legacy leg-input mask")
        if feedback.get("articulated_leg_current_applied") is not (mode == "on"):
            raise AnalysisError(f"{where} articulated leg-current mode does not match its mode label")
        body_source = feedback.get("body_state_source")
        if not isinstance(body_source, str) or not body_source.strip():
            raise AnalysisError(f"{where} does not describe its articulated-body sensory input")
        mapping_audit = _mapping(manifest.get("mapping"), f"{where}.mapping")
        overlap = _mapping(mapping_audit.get("leg_target_overlap"),
                           f"{where}.mapping.leg_target_overlap")
        if _integer(overlap.get("dense_neuron_overlap_count"),
                    f"{where}.mapping.leg_target_overlap.dense_neuron_overlap_count") != 0:
            raise AnalysisError(f"{where} has overlap between legacy and articulated leg targets")
        expected_frames = _integer(protocol.get("flyvis_frames"), f"{where}.protocol.flyvis_frames")
        expected_ticks = _integer(protocol.get("cns_ticks"), f"{where}.protocol.cns_ticks")
        if (_integer(vision_state.get("frames_stepped"), f"{where}.cuda_execution.vision.frames_stepped")
                != expected_frames
                or _integer(neural_state.get("ticks_stepped"), f"{where}.cuda_execution.neural.ticks_stepped")
                != expected_ticks
                or expected_ticks != expected_frames * 10):
            raise AnalysisError(f"{where} actual CUDA frame/tick counts do not match the protocol")
        if _mode(manifest) not in {"legacy", "off", "on"}:
            raise AnalysisError(f"{where} has an unsupported articulated sensory mode")
    else:
        raise AnalysisError(f"{where} has unsupported manifest schema {schema!r}")

    # All recognized explicit fallback fields must agree with the effective path.
    if manifest.get("fallback_used") is True or manifest.get("cpu_fallback") is True:
        raise AnalysisError(f"{where} reports a CPU/backend fallback")


def _declared_output_hashes(manifest: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Return output-relative path -> (sha256, provenance key) from known contracts."""
    found: dict[str, tuple[str, str]] = {}
    for field in ("output_hashes", "artifact_hashes", "artifacts", "outputs"):
        container = manifest.get(field)
        if not isinstance(container, Mapping):
            continue
        for name, value in container.items():
            relpath: Any = name
            digest: Any = None
            if isinstance(value, Mapping):
                relpath = value.get("path", value.get("file", name))
                digest = value.get("sha256", value.get("sha256_hex"))
                if value.get("role") not in (None, "output", "artifact"):
                    continue
            elif isinstance(value, str) and len(value) == 64:
                digest = value
            if (isinstance(relpath, str) and isinstance(digest, str)
                    and len(digest) == 64
                    and all(character in "0123456789abcdefABCDEF" for character in digest)):
                found[Path(relpath).as_posix()] = (digest.lower(), field)
    return found


def verify_declared_outputs(run_dir: Path, manifest: Mapping[str, Any], *, required: bool) -> dict[str, str]:
    declared = _declared_output_hashes(manifest)
    actual: dict[str, str] = {}
    if required:
        needed = {"telemetry.jsonl", "summary.json", "performance.json"}
        if (run_dir / "neural_spike_counts.jsonl").is_file():
            needed.add("neural_spike_counts.jsonl")
        absent = sorted(name for name in needed if name not in declared)
        if absent:
            raise AnalysisError(
                f"{run_dir}: manifest is missing output SHA-256 values for {', '.join(absent)}")
    for relpath, (expected, source_field) in declared.items():
        candidate = Path(relpath)
        if candidate.is_absolute():
            # Historical digital-fly manifests used absolute paths without hashes;
            # any hashed output must be relocatable and contained in the run.
            raise AnalysisError(f"{run_dir}: hashed output path must be relative: {relpath}")
        resolved = (run_dir / candidate).resolve()
        try:
            resolved.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise AnalysisError(f"{run_dir}: output hash path escapes the run directory") from exc
        if not resolved.is_file():
            raise AnalysisError(f"{run_dir}: hashed output is missing: {relpath}")
        observed = sha256_file(resolved)
        if observed.lower() != expected.lower():
            raise AnalysisError(f"{run_dir}: {source_field} digest mismatch for {relpath}")
        actual[relpath] = observed
    return actual


def verify_source_snapshots(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    if manifest.get("schema_version") != "articulated-optomotor-run-1":
        return {}
    hashes = _mapping(manifest.get("source_sha256"), f"{run_dir}.source_sha256")
    snapshot_name = manifest.get("source_snapshot_directory")
    if snapshot_name != "sources":
        raise AnalysisError(f"{run_dir}: source snapshot directory must be 'sources'")
    snapshot_root = run_dir / "sources"
    verified: dict[str, str] = {}
    for relpath, expected in hashes.items():
        if not isinstance(relpath, str) or not isinstance(expected, str) or len(expected) != 64:
            raise AnalysisError(f"{run_dir}: malformed source SHA-256 entry")
        candidate = Path(relpath)
        if candidate.is_absolute():
            raise AnalysisError(f"{run_dir}: source snapshot path must be relative")
        resolved = (snapshot_root / candidate).resolve()
        try:
            resolved.relative_to(snapshot_root.resolve())
        except ValueError as exc:
            raise AnalysisError(f"{run_dir}: source snapshot path escapes the run directory") from exc
        if not resolved.is_file():
            raise AnalysisError(f"{run_dir}: source snapshot is missing: {relpath}")
        actual = sha256_file(resolved)
        if actual.lower() != expected.lower():
            raise AnalysisError(f"{run_dir}: source snapshot digest mismatch for {relpath}")
        verified[relpath] = actual
    if not verified:
        raise AnalysisError(f"{run_dir}: articulated manifest declares no source snapshots")
    return verified


def discover_runs(root: Path, *, label: str, seed: int,
                  skipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise AnalysisError(f"{label} input is not a directory: {root}")
    manifest_dirs = {path.parent for path in root.rglob("manifest.json")}
    telemetry_dirs = {path.parent for path in root.rglob("telemetry.jsonl")}
    directories = sorted(manifest_dirs | telemetry_dirs)
    if not directories:
        raise AnalysisError(f"{label} input contains no run manifests or telemetry: {root}")
    runs: list[dict[str, Any]] = []
    for run_dir in directories:
        manifest_path = run_dir / "manifest.json"
        telemetry_path = run_dir / "telemetry.jsonl"
        if not manifest_path.is_file():
            raise AnalysisError(f"{run_dir}: telemetry exists without a manifest")
        manifest = _json(manifest_path, "manifest")
        if _status(manifest) in {"ERROR", "FAILED", "ABORTED"}:
            skipped.append({"path": str(run_dir), "model": label,
                            "status": _status(manifest),
                            "reason": str(manifest.get("error", manifest.get("failure", "run marked failed")))})
            continue
        if not telemetry_path.is_file():
            raise AnalysisError(f"{run_dir}: completed/non-error run is missing telemetry.jsonl")
        this_seed = _read_seed(manifest, str(run_dir))
        condition = _condition(manifest, str(run_dir))
        _duration_contract(manifest, str(run_dir))
        _cuda_and_protocol(manifest, label, str(run_dir))
        is_articulated = manifest.get("schema_version") == "articulated-optomotor-run-1"
        if this_seed != seed:
            continue
        hashes = verify_declared_outputs(run_dir, manifest, required=is_articulated)
        source_hashes = verify_source_snapshots(run_dir, manifest)
        runs.append({"path": run_dir, "manifest_path": manifest_path,
                     "telemetry_path": telemetry_path, "manifest": manifest,
                     "seed": this_seed, "condition": condition,
                     "mode": _mode(manifest) if is_articulated else label,
                     "model": label, "output_hashes": hashes,
                     "repeat_index": _repeat_index(manifest, run_dir),
                     "source_hashes_verified": source_hashes,
                     "manifest_sha256": sha256_file(manifest_path),
                     "telemetry_sha256": None})
    if not runs:
        raise AnalysisError(f"{label} has no valid runs for seed {seed}")
    return runs


def _vec3(value: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AnalysisError(f"{where} must contain three finite numbers")
    return tuple(_number(part, f"{where}[{index}]") for index, part in enumerate(value))  # type: ignore[return-value]


def _root_state(row: Mapping[str, Any], where: str) -> tuple[tuple[float, float, float], float]:
    # Reduced-body rows keep position/orientation at top level. Articulated rows
    # use body.root, matching the established articulated-body telemetry contract.
    if "position_m" in row and "orientation_rad" in row:
        position = _vec3(row["position_m"], f"{where}.position_m")
        orientation = _vec3(row["orientation_rad"], f"{where}.orientation_rad")
        return position, orientation[2]
    body = _mapping(row.get("body"), f"{where}.body")
    if isinstance(body.get("telemetry"), Mapping):
        body = _mapping(body["telemetry"], f"{where}.body.telemetry")
    root = _mapping(body.get("root"), f"{where}.body.root")
    position_value = root.get("position_m", root.get("position"))
    if position_value is None:
        raise AnalysisError(f"{where}.body.root.position_m is missing")
    position = _vec3(position_value, f"{where}.body.root.position_m")
    if "yaw_rad" in root:
        yaw = _number(root["yaw_rad"], f"{where}.body.root.yaw_rad")
    elif "orientation_rad" in root:
        yaw = _vec3(root["orientation_rad"], f"{where}.body.root.orientation_rad")[2]
    elif "orientation_rpy_rad" in root:
        yaw = _vec3(root["orientation_rpy_rad"],
                    f"{where}.body.root.orientation_rpy_rad")[2]
    elif "orientation_rad" in body:
        yaw = _vec3(body["orientation_rad"], f"{where}.body.orientation_rad")[2]
    elif "orientation_rpy_rad" in body:
        yaw = _vec3(body["orientation_rpy_rad"],
                    f"{where}.body.orientation_rpy_rad")[2]
    else:
        raise AnalysisError(f"{where}.body.root must expose yaw_rad or orientation_rad")
    return position, yaw


def _initial_pose(run: Mapping[str, Any], first_row: Mapping[str, Any]) -> tuple[tuple[float, float, float], float]:
    manifest = run["manifest"]
    summary_path = run["path"] / "summary.json"
    initial_state = manifest.get("initial_state")
    root_pose = initial_state.get("root_pose") if isinstance(initial_state, Mapping) else None
    pose_candidates: list[Any] = [manifest.get("initial_pose"), root_pose,
                                   manifest.get("initial_state"), first_row.get("initial_pose")]
    if summary_path.is_file():
        # The new runner may keep initial state in summary.json; the file is small.
        summary = _json(summary_path, "summary")
        pose_candidates.extend((summary.get("initial_pose"), summary.get("initial_root"),
                                summary.get("initial_state"), summary.get("initial_root_pose")))
    body_before = first_row.get("body_before")
    if isinstance(body_before, Mapping):
        pose_candidates.insert(0, body_before)
    for pose_value in pose_candidates:
        if not isinstance(pose_value, Mapping):
            continue
        pose = pose_value
        if isinstance(pose.get("telemetry"), Mapping):
            pose = pose["telemetry"]
        if isinstance(pose.get("root"), Mapping):
            pose = pose["root"]
        elif isinstance(pose.get("body"), Mapping):
            pose = pose["body"]
            if isinstance(pose.get("root"), Mapping):
                pose = pose["root"]
        position_value = pose.get("position_m", pose.get("position"))
        yaw_value = pose.get("yaw_rad")
        if yaw_value is None and "orientation_rad" in pose:
            yaw_value = _vec3(pose["orientation_rad"], "initial_pose.orientation_rad")[2]
        if yaw_value is None and "orientation_rpy_rad" in pose:
            yaw_value = _vec3(pose["orientation_rpy_rad"],
                              "initial_pose.orientation_rpy_rad")[2]
        if position_value is not None and yaw_value is not None:
            return (_vec3(position_value, "initial_pose.position_m"),
                    _number(yaw_value, "initial_pose.yaw_rad"))
    # Legacy reduced-body trajectories explicitly record body_before on frame 0.
    if run["manifest"].get("schema_version") == "digital-fly-run-1":
        raise AnalysisError(f"{run['path']}: true pre-step initial pose is absent")
    raise AnalysisError(f"{run['path']}: articulated manifest/summary lacks the true initial pose")


def _population_counts(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    neural = row.get("neural")
    if not isinstance(neural, Mapping):
        return None
    direct = neural.get("annotated_leg_spikes_by_leg")
    if direct is None:
        direct = neural.get("annotated_leg_spikes")
    if direct is None:
        direct = neural.get("motor_spikes_by_leg", neural.get("mapped_motor_spikes_by_leg"))
    if isinstance(direct, Mapping):
        return direct
    outputs = neural.get("named_outputs")
    if isinstance(outputs, Mapping) and isinstance(outputs.get("population_spike_counts"), Mapping):
        return outputs["population_spike_counts"]
    if isinstance(neural.get("population_spike_counts"), Mapping):
        return neural["population_spike_counts"]
    return None


def _mapped_leg_counts(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    neural = row.get("neural")
    if not isinstance(neural, Mapping):
        return None
    value = neural.get("mapped_motor_spikes_by_leg", neural.get("mapped_leg_spikes"))
    return value if isinstance(value, Mapping) else None


def _afferent_leg_counts(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    neural = row.get("neural")
    if not isinstance(neural, Mapping):
        return None
    value = neural.get("articulated_leg_sensory_spikes_by_leg")
    return value if isinstance(value, Mapping) else None


def _ensure_finite_tree(value: Any, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite_tree(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite_tree(item, f"{where}[{index}]")
        return
    if isinstance(value, (int, float)):
        _number(value, where)


def _leg_value(counts: Mapping[str, Any], pool: str, where: str) -> float:
    leg = pool.rsplit(":", 1)[-1]
    abbreviation = {"fl/L": "LF", "fl/R": "RF", "ml/L": "LM",
                    "ml/R": "RM", "hl/L": "LH", "hl/R": "RH"}[leg]
    for key in (pool, leg, leg.replace("/", "_"), leg.replace("/", "-"), abbreviation):
        if key in counts:
            return _number(counts[key], f"{where}.{key}", nonnegative=True)
    raise AnalysisError(f"{where} is missing per-leg motor count {pool}")


def _articulated_body_parts(row: Mapping[str, Any], where: str) -> Mapping[str, Any] | None:
    body = row.get("body")
    if not isinstance(body, Mapping):
        return None
    if isinstance(body.get("telemetry"), Mapping):
        return body["telemetry"]
    return body


def _feet_contacts(body: Mapping[str, Any]) -> tuple[dict[str, int] | None, int | None, int | None, float | None]:
    feet_value = body.get("feet")
    if not isinstance(feet_value, Mapping):
        return None, None, None, None
    contacts: dict[str, int] = {}
    force = 0.0
    force_known = False
    for name, raw_foot in feet_value.items():
        if not isinstance(raw_foot, Mapping):
            continue
        contact = raw_foot.get("contact", raw_foot.get("in_contact"))
        if isinstance(contact, bool):
            contacts[str(name)] = int(contact)
        normal = raw_foot.get("normal_force_N", raw_foot.get("normal_force"))
        if normal is not None:
            force += _number(normal, f"body.feet.{name}.normal_force_N", nonnegative=True)
            force_known = True
    if len(contacts) != 6:
        contacts_out = None
    else:
        contacts_out = contacts
    count = sum(contacts.values()) if contacts_out is not None else None
    body_ground = body.get("body_ground_contact", body.get("nonfoot_ground_contact"))
    nonfoot = int(body_ground) if isinstance(body_ground, bool) else None
    if nonfoot is None:
        contacts_list = body.get("body_contacts", body.get("contacts"))
        if isinstance(contacts_list, list):
            nonfoot_count = 0
            classifiable = True
            for contact in contacts_list:
                if not isinstance(contact, Mapping) or contact.get("ground_contact") is not True:
                    continue
                if contact.get("foot_ground_contact") is True:
                    continue
                part = str(contact.get("part", contact.get("body_part", ""))).lower()
                if "foot" in part:
                    continue
                geom_names = " ".join(str(contact.get(key, "")) for key in
                                       ("geom1", "geom2", "geom1_name", "geom2_name")).lower()
                if "foot" in geom_names:
                    continue
                if not part and not geom_names:
                    classifiable = False
                    break
                nonfoot_count += 1
            if classifiable:
                nonfoot = nonfoot_count
    return contacts_out, count, nonfoot, force if force_known else None


def _muscle_forces(row: Mapping[str, Any], where: str) -> tuple[float | None, dict[str, float] | None]:
    muscles = row.get("muscles")
    if not isinstance(muscles, Mapping):
        return None, None
    explicit = muscles.get("active_force_by_leg_N", muscles.get("force_by_leg_N"))
    if isinstance(explicit, Mapping):
        per_leg = {str(key): _number(value, f"{where}.muscles.{key}", nonnegative=True)
                   for key, value in explicit.items()}
        if all(leg in per_leg for leg in ("LF", "RF", "LM", "RM", "LH", "RH")):
            return sum(per_leg.values()), per_leg
        return sum(per_leg.values()), per_leg
    active_forces: list[float] = []
    by_leg: dict[str, float] = {}
    for name, raw in muscles.items():
        if not isinstance(raw, Mapping) or "active_force_N" not in raw:
            continue
        value = _number(raw["active_force_N"], f"{where}.muscles.{name}.active_force_N",
                        nonnegative=True)
        active_forces.append(value)
        key = str(name).split("_", 1)[0]
        aliases = {"LF": "LF", "RF": "RF", "LM": "LM", "RM": "RM", "LH": "LH", "RH": "RH"}
        if key in aliases:
            by_leg[key] = by_leg.get(key, 0.0) + value
    return (sum(active_forces), by_leg if by_leg else None) if active_forces else (None, None)


def _parse_line(raw: bytes, path: Path, line_no: int) -> Mapping[str, Any]:
    if not raw.strip():
        raise AnalysisError(f"{path}: blank telemetry line at {line_no}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{path}: invalid JSON at line {line_no}: {exc}") from exc
    return _mapping(value, f"{path.name}:{line_no}")


def verify_neural_tick_counts(run: Mapping[str, Any]) -> dict[str, Any]:
    """Stream the 0.5 ms count log and reconcile all six-leg totals."""
    path = run["path"] / "neural_spike_counts.jsonl"
    expected_ticks = EXPECTED_FRAMES * 10
    totals = {"neural_spikes": 0, "annotated_leg_spikes": 0,
              "mapped_motor_spikes": 0, "articulated_leg_sensory_spikes": 0}
    per_leg = {
        "annotated_leg_spikes_by_leg": {leg: 0 for leg in ("LF", "RF", "LM", "RM", "LH", "RH")},
        "mapped_motor_spikes_by_leg": {leg: 0 for leg in ("LF", "RF", "LM", "RM", "LH", "RH")},
        "articulated_leg_sensory_spikes_by_leg": {leg: 0 for leg in ("LF", "RF", "LM", "RM", "LH", "RH")},
    }
    count = 0
    try:
        with path.open("rb") as handle:
            for index, raw in enumerate(handle):
                row = _parse_line(raw, path, index + 1)
                tick = _integer(row.get("tick"), f"{path}.tick")
                if tick != index:
                    raise AnalysisError(f"{path}: tick {tick} occurs at row {index}")
                t_s = _number(row.get("t_s"), f"{path}.t_s")
                if not math.isclose(t_s, tick * 0.0005, rel_tol=0, abs_tol=1e-10):
                    raise AnalysisError(f"{path}: tick {tick} has an unexpected timestamp")
                row_fields = {"neural_spikes": "total_spikes",
                              "annotated_leg_spikes": "annotated_leg_spikes",
                              "mapped_motor_spikes": "mapped_motor_spikes",
                              "articulated_leg_sensory_spikes": "articulated_leg_sensory_spikes"}
                for field, row_field in row_fields.items():
                    totals[field] += _integer(row.get(row_field), f"{path}.{row_field}")
                for field, values in per_leg.items():
                    declared = _mapping(row.get(field), f"{path}.{field}")
                    for leg in values:
                        values[leg] += _integer(declared.get(leg), f"{path}.{field}.{leg}")
                count += 1
    except OSError as exc:
        raise AnalysisError(f"cannot read neural count log {path}: {exc}") from exc
    if count != expected_ticks:
        raise AnalysisError(f"{path}: has {count} ticks; expected {expected_ticks}")
    manifest_counts = _mapping(run["manifest"].get("counts"), f"{run['path']}.counts")
    aliases = {"neural_spikes": "neural_spikes",
               "annotated_leg_spikes": "annotated_leg_spikes",
               "mapped_motor_spikes": "mapped_motor_spikes",
               "articulated_leg_sensory_spikes": "articulated_leg_sensory_spikes"}
    for computed, declared_key in aliases.items():
        if _integer(manifest_counts.get(declared_key),
                    f"{run['path']}.counts.{declared_key}") != totals[computed]:
            raise AnalysisError(f"{path}: {computed} total disagrees with the manifest")
    for field, values in per_leg.items():
        declared = _mapping(manifest_counts.get(field), f"{run['path']}.counts.{field}")
        if {leg: _integer(declared.get(leg), f"{run['path']}.counts.{field}.{leg}")
                for leg in values} != values:
            raise AnalysisError(f"{path}: {field} totals disagree with the manifest")
    return {**totals, **per_leg, "ticks": count}


def analyze_trial(run: dict[str, Any]) -> dict[str, Any]:
    """Stream one trial and summarize only trial-level, not frame-level, outcomes."""
    path: Path = run["telemetry_path"]
    digest = hashlib.sha256()
    frame_count = 0
    first_row: Mapping[str, Any] | None = None
    initial_position: tuple[float, float, float] | None = None
    initial_yaw: float | None = None
    previous_yaw: float | None = None
    unwrapped_yaw: float | None = None
    final_position: tuple[float, float, float] | None = None
    per_leg_spikes = {leg: 0.0 for leg in LEGS}
    mapped_motor_spikes = 0.0
    mapped_motor_frames = 0
    mapped_leg_spikes = {leg: 0.0 for leg in ("LF", "RF", "LM", "RM", "LH", "RH")}
    mapped_leg_frames = 0
    afferent_leg_spikes = {leg: 0.0 for leg in ("LF", "RF", "LM", "RM", "LH", "RH")}
    afferent_leg_frames = 0
    afferent_spikes = 0.0
    afferent_frames = 0
    leg_counts_known = True
    force_sum = 0.0
    force_frames = 0
    force_peak = 0.0
    per_leg_force_sum: dict[str, float] = {}
    per_leg_force_peak: dict[str, float] = {}
    joint_delta_final: dict[str, float] | None = None
    joint_delta_peak: dict[str, float] = {}
    foot_contact_sum: dict[str, int] = {}
    contact_frames = body_contact_frames = nonfoot_contact_total = 0
    contact_known = nonfoot_known = True
    foot_force_sum = 0.0
    foot_force_frames = 0
    conditions_seen: set[str] = set()
    try:
        with path.open("rb") as handle:
            for index, raw in enumerate(handle):
                digest.update(raw)
                row = _parse_line(raw, path, index + 1)
                frame = _integer(row.get("frame_index", index), f"{path}.frame_index")
                if frame != index:
                    raise AnalysisError(f"{path}: frame_index {frame} occurs at row {index}")
                timestamp = _number(row.get("frame_time_s"), f"{path}.frame_time_s")
                expected_time = index * EXPECTED_DT_S
                if not math.isclose(timestamp, expected_time, rel_tol=0, abs_tol=1e-8):
                    raise AnalysisError(f"{path}: frame {index} has unexpected time {timestamp:g}s")
                if run["manifest"].get("schema_version") == "articulated-optomotor-run-1":
                    clock = _mapping(row.get("timestamp"), f"{path}.timestamp")
                    if (clock.get("clock") != "control"
                            or _integer(clock.get("tick"), f"{path}.timestamp.tick") != index
                            or not math.isclose(_number(clock.get("rate_hz"),
                                                        f"{path}.timestamp.rate_hz"),
                                                200.0, rel_tol=0, abs_tol=1e-9)
                            or not math.isclose(_number(clock.get("t_s"),
                                                        f"{path}.timestamp.t_s"),
                                                expected_time, rel_tol=0, abs_tol=1e-8)):
                        raise AnalysisError(f"{path}: frame {index} control clock differs from protocol")
                    feedback_row = _mapping(row.get("sensory_feedback"),
                                            f"{path}.sensory_feedback")
                    expected_mode = _mode(run["manifest"])
                    if feedback_row.get("mode") != expected_mode:
                        raise AnalysisError(f"{path}: frame {index} sensory mode differs from manifest")
                    for field in ("source_time_s", "applied_at_s"):
                        observed = _number(feedback_row.get(field),
                                           f"{path}.sensory_feedback.{field}")
                        if not math.isclose(observed, expected_time, rel_tol=0, abs_tol=1e-8):
                            raise AnalysisError(
                                f"{path}: frame {index} feedback {field} is out of sync with body time")
                    encoder_ticks = feedback_row.get("articulated_leg_encoder_ticks", [])
                    if expected_mode == "on":
                        if not isinstance(encoder_ticks, list) or len(encoder_ticks) != 10:
                            raise AnalysisError(
                                f"{path}: frame {index} lacks ten articulated-leg encoder ticks")
                        for substep, encoded in enumerate(encoder_ticks):
                            encoded = _mapping(encoded, f"{path}.articulated_leg_encoder_ticks")
                            tick = index * 10 + substep
                            for field in ("tick",):
                                if _integer(encoded.get(field), f"{path}.encoder.{field}") != tick:
                                    raise AnalysisError(f"{path}: articulated afferent tick sequence has a gap")
                            tick_time = tick * 0.0005
                            for field in ("source_time_s", "applied_at_s"):
                                observed = _number(encoded.get(field), f"{path}.encoder.{field}")
                                if not math.isclose(observed, tick_time, rel_tol=0, abs_tol=1e-10):
                                    raise AnalysisError(
                                        f"{path}: articulated afferent {field} is out of sync at tick {tick}")
                            if encoded.get("input_applied") is not True:
                                raise AnalysisError(f"{path}: articulated afferent input was not applied")
                condition_value = row.get("stimulus_condition", row.get("condition"))
                if condition_value is not None:
                    conditions_seen.add(_normalize_condition(condition_value, str(path)))
                position, yaw = _root_state(row, str(path))
                body_state = row.get("body")
                if body_state is not None:
                    _ensure_finite_tree(body_state, f"{path}:frame {index}.body")
                if row.get("muscles") is not None:
                    _ensure_finite_tree(row["muscles"], f"{path}:frame {index}.muscles")
                if first_row is None:
                    first_row = row
                    initial_position, initial_yaw = _initial_pose(run, row)
                    previous_yaw = initial_yaw
                    unwrapped_yaw = initial_yaw
                assert initial_position is not None and initial_yaw is not None
                assert previous_yaw is not None and unwrapped_yaw is not None
                delta_yaw = (yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
                unwrapped_yaw += delta_yaw
                previous_yaw = yaw
                final_position = position

                counts = _population_counts(row)
                if counts is None:
                    leg_counts_known = False
                else:
                    try:
                        values = {leg: _leg_value(counts, pool, f"{path}:frame {index}")
                                  for leg, pool in zip(LEGS, LEG_POOLS)}
                    except AnalysisError:
                        # Current articulated rows publish annotated counts per leg.
                        direct = row.get("neural")
                        if isinstance(direct, Mapping):
                            direct = direct.get("annotated_leg_spikes_by_leg")
                        if isinstance(direct, Mapping) and all(leg in direct for leg in LEGS):
                            values = {leg: _number(direct[leg], f"{path}.motor_spikes.{leg}",
                                                   nonnegative=True) for leg in LEGS}
                        else:
                            leg_counts_known = False
                            values = {}
                    for leg, value in values.items():
                        per_leg_spikes[leg] += value
                neural = row.get("neural")
                if isinstance(neural, Mapping):
                    if run["manifest"].get("schema_version") == "digital-fly-run-1":
                        outputs = _mapping(neural.get("named_outputs"),
                                           f"{path}.neural.named_outputs")
                        if (_integer(outputs.get("neural_window_steps"),
                                     f"{path}.neural_window_steps") != 10
                                or outputs.get("spikes_are_window_counts") is not True
                                or not math.isclose(_number(outputs.get("neural_window_dt_s"),
                                                            f"{path}.neural_window_dt_s"),
                                                    EXPECTED_DT_S, rel_tol=0, abs_tol=1e-10)
                                or outputs.get("rates_semantics") != "window_mean_hz"):
                            raise AnalysisError(
                                f"{path}: reduced-body spike counts do not cover the full 5 ms window")
                    if run["manifest"].get("schema_version") == "articulated-optomotor-run-1":
                        if _integer(neural.get("tick_count"), f"{path}.neural.tick_count") != 10:
                            raise AnalysisError(f"{path}: frame {index} does not contain ten neural ticks")
                    count_value = neural.get("mapped_motor_spikes")
                    if count_value is not None:
                        mapped_motor_spikes += _number(count_value, f"{path}.mapped_motor_spikes",
                                                        nonnegative=True)
                        mapped_motor_frames += 1

                    afferent_value = neural.get("articulated_leg_sensory_spikes")
                    if afferent_value is not None:
                        afferent_spikes += _number(afferent_value, f"{path}.articulated_leg_sensory_spikes",
                                                   nonnegative=True)
                        afferent_frames += 1
                    afferent_counts = _afferent_leg_counts(row)
                    if afferent_counts is not None:
                        current_afferent = {
                            leg: _number(afferent_counts[leg], f"{path}.afferent_spikes.{leg}",
                                         nonnegative=True)
                            for leg in ("LF", "RF", "LM", "RM", "LH", "RH")
                            if leg in afferent_counts
                        }
                        if len(current_afferent) != 6:
                            raise AnalysisError(f"{path}: articulated leg afferent counts must contain six legs")
                        for leg, value in current_afferent.items():
                            afferent_leg_spikes[leg] += value
                        afferent_leg_frames += 1
                    mapped_counts = _mapped_leg_counts(row)
                    if mapped_counts is not None:
                        current_mapped = {
                            leg: _number(mapped_counts[leg], f"{path}.mapped_leg_spikes.{leg}",
                                         nonnegative=True)
                            for leg in ("LF", "RF", "LM", "RM", "LH", "RH")
                            if leg in mapped_counts
                        }
                        if len(current_mapped) != 6:
                            raise AnalysisError(f"{path}: mapped_leg_spikes must contain all six legs")
                        for leg, value in current_mapped.items():
                            mapped_leg_spikes[leg] += value
                        mapped_leg_frames += 1
                        current_mapped_total = sum(current_mapped.values())
                        if count_value is not None and not math.isclose(
                                _number(count_value, f"{path}.mapped_motor_spikes"),
                                current_mapped_total, rel_tol=1e-9, abs_tol=1e-6):
                            raise AnalysisError(
                                f"{path}: mapped total disagrees with the six per-leg mapped counts")
                        if count_value is None:
                            mapped_motor_spikes += current_mapped_total
                            mapped_motor_frames += 1

                body = _articulated_body_parts(row, str(path))
                if body is not None:
                    feet, foot_count, nonfoot_count, foot_force = _feet_contacts(body)
                    if feet is None or foot_count is None:
                        contact_known = False
                    else:
                        for leg, present in feet.items():
                            foot_contact_sum[leg] = foot_contact_sum.get(leg, 0) + present
                        contact_frames += int(foot_count > 0)
                    if nonfoot_count is None:
                        nonfoot_known = False
                    else:
                        nonfoot_contact_total += nonfoot_count
                        body_contact_frames += int(nonfoot_count > 0)
                    if foot_force is not None:
                        foot_force_sum += foot_force
                        foot_force_frames += 1
                    total_force, by_leg_force = _muscle_forces(row, f"{path}:frame {index}")
                    if total_force is not None:
                        force_sum += total_force
                        force_frames += 1
                        force_peak = max(force_peak, total_force)
                    if by_leg_force:
                        for leg, value in by_leg_force.items():
                            per_leg_force_sum[leg] = per_leg_force_sum.get(leg, 0.0) + value
                    peak_by_leg = row["muscles"].get("peak_active_force_by_leg_N") \
                        if isinstance(row.get("muscles"), Mapping) else None
                    if isinstance(peak_by_leg, Mapping):
                        for leg, value in peak_by_leg.items():
                            parsed = _number(value, f"{path}.peak_active_force_by_leg_N.{leg}",
                                              nonnegative=True)
                            per_leg_force_peak[leg] = max(per_leg_force_peak.get(leg, 0.0), parsed)
                joint_delta = row.get("joint_delta_rad")
                if isinstance(joint_delta, Mapping):
                    current_delta = {str(key): _number(value, f"{path}.joint_delta_rad.{key}")
                                     for key, value in joint_delta.items()}
                    joint_delta_final = current_delta
                    for joint, value in current_delta.items():
                        joint_delta_peak[joint] = max(joint_delta_peak.get(joint, 0.0), abs(value))
                frame_count += 1
    except OSError as exc:
        raise AnalysisError(f"cannot read telemetry {path}: {exc}") from exc

    if frame_count != EXPECTED_FRAMES:
        raise AnalysisError(f"{path}: has {frame_count} rows; expected {EXPECTED_FRAMES}")
    if conditions_seen and conditions_seen != {run["condition"]}:
        raise AnalysisError(f"{path}: telemetry stimulus condition does not match its manifest")
    if initial_position is None or initial_yaw is None or final_position is None or unwrapped_yaw is None:
        raise AnalysisError(f"{path}: missing body state")
    expected_digest = run.get("output_hashes", {}).get("telemetry.jsonl")
    if expected_digest is not None and digest.hexdigest().lower() != expected_digest.lower():
        raise AnalysisError(f"{path}: streamed telemetry SHA-256 differs from manifest digest")

    dx = final_position[0] - initial_position[0]
    dy = final_position[1] - initial_position[1]
    forward = dx * math.cos(initial_yaw) + dy * math.sin(initial_yaw)
    lateral = -dx * math.sin(initial_yaw) + dy * math.cos(initial_yaw)
    schema = run["manifest"].get("schema_version")
    if schema == "articulated-optomotor-run-1":
        if not leg_counts_known or mapped_motor_frames != frame_count or mapped_leg_frames != frame_count:
            raise AnalysisError(f"{run['path']}: articulated telemetry lacks mapped six-leg motor counts")
        if afferent_frames != frame_count or afferent_leg_frames != frame_count:
            raise AnalysisError(f"{run['path']}: articulated telemetry lacks six-leg afferent counts")
        if force_frames == 0:
            raise AnalysisError(f"{run['path']}: articulated telemetry lacks muscle-force summaries")
        if not contact_known or not nonfoot_known:
            raise AnalysisError(f"{run['path']}: articulated telemetry lacks support/contact fields")
        manifest_counts = _mapping(run["manifest"].get("counts"), f"{run['path']}.counts")
        if _integer(manifest_counts.get("mapped_motor_spikes"),
                    f"{run['path']}.counts.mapped_motor_spikes") != int(mapped_motor_spikes):
            raise AnalysisError(f"{run['path']}: mapped motor total disagrees with the manifest")
        declared_mapped = _mapping(manifest_counts.get("mapped_motor_spikes_by_leg"),
                                   f"{run['path']}.counts.mapped_motor_spikes_by_leg")
        declared_annotated = _mapping(manifest_counts.get("annotated_leg_spikes_by_leg"),
                                      f"{run['path']}.counts.annotated_leg_spikes_by_leg")
        for leg, mapped_total in mapped_leg_spikes.items():
            if _integer(declared_mapped.get(leg), f"{run['path']}.counts.mapped.{leg}") != int(mapped_total):
                raise AnalysisError(f"{run['path']}: mapped {leg} total disagrees with the manifest")
        for abbreviation, leg in zip(("LF", "RF", "LM", "RM", "LH", "RH"), LEGS):
            if _integer(declared_annotated.get(abbreviation),
                        f"{run['path']}.counts.annotated.{abbreviation}") != int(per_leg_spikes[leg]):
                raise AnalysisError(
                    f"{run['path']}: annotated six-leg count for {abbreviation} disagrees with the manifest")
        tick_totals = verify_neural_tick_counts(run)
        if tick_totals["mapped_motor_spikes"] != int(mapped_motor_spikes):
            raise AnalysisError(f"{run['path']}: per-tick and per-frame mapped motor totals disagree")
        if tick_totals["annotated_leg_spikes_by_leg"] != {
                leg: int(per_leg_spikes[LEGS[index]])
                for index, leg in enumerate(("LF", "RF", "LM", "RM", "LH", "RH"))}:
            raise AnalysisError(f"{run['path']}: per-tick and per-frame annotated leg counts disagree")
        if tick_totals["annotated_leg_spikes"] != int(sum(per_leg_spikes.values())):
            raise AnalysisError(f"{run['path']}: total annotated spikes disagree with per-frame counts")
        if tick_totals["mapped_motor_spikes_by_leg"] != {
                leg: int(mapped_leg_spikes[leg]) for leg in mapped_leg_spikes}:
            raise AnalysisError(f"{run['path']}: per-tick and per-frame mapped leg counts disagree")
        if tick_totals["articulated_leg_sensory_spikes_by_leg"] != {
                leg: int(afferent_leg_spikes[leg]) for leg in afferent_leg_spikes}:
            raise AnalysisError(f"{run['path']}: per-tick and per-frame afferent counts disagree")
        if tick_totals["articulated_leg_sensory_spikes"] != int(afferent_spikes):
            raise AnalysisError(f"{run['path']}: per-tick and per-frame afferent totals disagree")

    performance_summary: dict[str, Any] = {}
    performance_path = run["path"] / "performance.json"
    if performance_path.is_file():
        performance = _json(performance_path, "performance")
        stages = performance.get("stages_s", performance.get("stage_wall_s",
                              performance.get("stage_wall_times_s")))
        if isinstance(stages, Mapping):
            performance_summary["stages_wall_s"] = {
                str(key): _number(value, f"performance.{key}", nonnegative=True)
                for key, value in stages.items()
            }
        for key in ("simulation_loop_s", "frame_total_s", "loop_wall_s", "wall_s"):
            if performance.get(key) is not None:
                performance_summary["simulation_wall_s"] = _number(
                    performance[key], f"performance.{key}", nonnegative=True)
                break
    if not performance_summary:
        manifest = run["manifest"]
        scenario_results = manifest.get("scenario_results")
        if isinstance(scenario_results, Mapping):
            free = scenario_results.get("free-behavior")
            if isinstance(free, Mapping) and isinstance(free.get("metrics"), Mapping):
                stages = free["metrics"].get("stage_wall_s")
                if isinstance(stages, Mapping):
                    performance_summary["stages_wall_s"] = {
                        str(key): _number(value, f"manifest.stage_wall_s.{key}", nonnegative=True)
                        for key, value in stages.items()
                    }
                    if stages.get("frame_total") is not None:
                        performance_summary["simulation_wall_s"] = _number(
                            stages["frame_total"], "manifest.stage_wall_s.frame_total",
                            nonnegative=True)
        summary_path = run["path"] / "summary.json"
        if not performance_summary and summary_path.is_file():
            summary = _json(summary_path, "summary")
            timing = summary.get("timing")
            if isinstance(timing, Mapping):
                stages = timing.get("stages_s", timing.get("stage_wall_s", timing))
                if isinstance(stages, Mapping):
                    performance_summary["stages_wall_s"] = {
                        str(key): _number(value, f"summary.timing.{key}", nonnegative=True)
                        for key, value in stages.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                    for key in ("simulation_loop_s", "frame_total_s", "wall_s"):
                        if timing.get(key) is not None:
                            performance_summary["simulation_wall_s"] = _number(
                                timing[key], f"summary.timing.{key}", nonnegative=True)
                            break

    return {
        "run_id": run["path"].name,
        "path": str(run["path"]),
        "model": run["model"],
        "mode": run["mode"],
        "repeat_index": run.get("repeat_index"),
        "sensory_feedback_contract": run["manifest"].get("sensory_feedback"),
        "seed": run["seed"],
        "condition": run["condition"],
        "frames": frame_count,
        "duration_s": EXPECTED_DURATION_S,
        "dt_s": EXPECTED_DT_S,
        "initial_pose": {"position_m": list(initial_position),
                          "yaw_deg": math.degrees(initial_yaw)},
        "initial_body_height_mm": 1000.0 * initial_position[2],
        "initial_root_vs_camera_reference": (
            run["manifest"].get("protocol", {}).get("initial_root_vs_original_camera")
            if isinstance(run["manifest"].get("protocol"), Mapping) else None),
        "forward_displacement_mm": 1000.0 * forward,
        "lateral_displacement_mm": 1000.0 * lateral,
        "net_yaw_change_deg": math.degrees(unwrapped_yaw - initial_yaw),
        "final_yaw_unwrapped_deg": math.degrees(unwrapped_yaw),
        "annotated_leg_spikes": sum(per_leg_spikes.values()) if leg_counts_known else None,
        "mapped_motor_spikes": mapped_motor_spikes if mapped_motor_frames == frame_count else None,
        "mapped_leg_motor_spikes": ({leg: mapped_leg_spikes[leg]
                                      for leg in mapped_leg_spikes}
                                     if mapped_leg_frames == frame_count else None),
        "articulated_leg_sensory_spikes": afferent_spikes if afferent_frames == frame_count else None,
        "articulated_leg_sensory_spikes_by_leg": ({leg: afferent_leg_spikes[leg]
                                                    for leg in afferent_leg_spikes}
                                                   if afferent_leg_frames == frame_count else None),
        "six_leg_motor_spikes": ({leg: per_leg_spikes[leg] for leg in LEGS}
                                  if leg_counts_known else None),
        "six_leg_count_semantics": (
            "sum of recorded six-pool 5 ms window spike counts; reduced-body manifests "
            "declare ten 0.5 ms neural steps per output frame"
            if leg_counts_known else None),
        "mean_total_active_muscle_force_N": force_sum / force_frames if force_frames else None,
        "peak_total_active_muscle_force_N": force_peak if force_frames else None,
        "mean_active_muscle_force_by_leg_N": {
            key: value / force_frames for key, value in per_leg_force_sum.items()
        } if per_leg_force_sum else None,
        "peak_active_muscle_force_by_leg_N": per_leg_force_peak or None,
        "final_joint_delta_rad": joint_delta_final,
        "peak_abs_joint_delta_rad": joint_delta_peak or None,
        "mean_peak_abs_joint_delta_rad": (
            statistics.mean(joint_delta_peak.values()) if joint_delta_peak else None),
        "mean_final_abs_joint_delta_rad": (
            statistics.mean(map(abs, joint_delta_final.values())) if joint_delta_final else None),
        "foot_contact_fraction_any": contact_frames / frame_count if contact_known else None,
        "foot_contact_frames_by_leg": foot_contact_sum if contact_known else None,
        "nonfoot_ground_contact_frames": body_contact_frames if nonfoot_known else None,
        "nonfoot_ground_contact_events": nonfoot_contact_total if nonfoot_known else None,
        "mean_total_foot_normal_force_N": foot_force_sum / foot_force_frames if foot_force_frames else None,
        "wall_timing": performance_summary or None,
        "integrity": {
            "manifest_sha256": run["manifest_sha256"],
            "telemetry_sha256": digest.hexdigest(),
            "declared_output_hashes_verified": run["output_hashes"],
            "source_snapshot_hashes_verified": run.get("source_hashes_verified", {}),
            "historical_artifact_hash_limitation": (
                "manifest has no output digests; computed hashes identify the analyzed files"
                if not run["output_hashes"] else None),
        },
    }


def _permutation_summary(values: list[float]) -> dict[str, Any]:
    count = len(values)
    base = {"n_seeds": count,
            "seed_unit": "deterministic digital-simulation seed; not a biological replicate",
            "confidence_interval": None}
    if count < MIN_SEEDS_FOR_TEST:
        base.update({"exact_two_sided_sign_flip_p": None,
                     "test_status": "not_estimable_low_seed_count",
                     "minimum_seed_count_for_test": MIN_SEEDS_FOR_TEST})
        return base
    if count > 20:
        base.update({"exact_two_sided_sign_flip_p": None,
                     "test_status": "not_computed_exact_enumeration_limit",
                     "maximum_seed_count_for_exact_test": 20})
        return base
    observed = abs(statistics.mean(values))
    extreme = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=count):
        statistic = abs(statistics.mean(value * sign for value, sign in zip(values, signs)))
        total += 1
        if statistic >= observed - 1e-12:
            extreme += 1
    base.update({"exact_two_sided_sign_flip_p": extreme / total,
                 "test_status": "exact_sign_flip_permutation_descriptive_seedset_only",
                 "statistic": "absolute mean paired seed difference",
                 "permutations": total})
    return base


def _metric_delta(reference: float | None, comparison: float | None) -> dict[str, Any]:
    if reference is None or comparison is None:
        return {"reference": reference, "comparison": comparison,
                "difference": None, "percent_change": None}
    delta = comparison - reference
    percent = (100.0 * delta / abs(reference)) if abs(reference) > 1e-15 else None
    return {"reference": reference, "comparison": comparison,
            "difference": delta, "percent_change": percent}


def _paired_contrast(reference: list[dict[str, Any]], comparison: list[dict[str, Any]],
                     *, label: str) -> dict[str, Any]:
    reference_by_key = {(row["seed"], row["condition"]): row for row in reference}
    comparison_by_key = {(row["seed"], row["condition"]): row for row in comparison}
    common = sorted(reference_by_key.keys() & comparison_by_key.keys())
    metrics = ("initial_body_height_mm", "forward_displacement_mm", "lateral_displacement_mm",
               "net_yaw_change_deg", "annotated_leg_spikes", "mapped_motor_spikes",
               "articulated_leg_sensory_spikes", "mean_total_active_muscle_force_N",
               "mean_total_foot_normal_force_N", "foot_contact_fraction_any",
               "nonfoot_ground_contact_frames", "mean_peak_abs_joint_delta_rad",
               "mean_final_abs_joint_delta_rad")
    paired_rows = []
    for key in common:
        a, b = reference_by_key[key], comparison_by_key[key]
        paired_rows.append({"seed": key[0], "condition": key[1],
                            "metrics": {metric: _metric_delta(a.get(metric), b.get(metric))
                                        for metric in metrics}})
    aggregate: dict[str, Any] = {}
    for metric in metrics:
        deltas_by_seed: dict[int, list[float]] = {}
        percentages_by_seed: dict[int, list[float]] = {}
        for row in paired_rows:
            item = row["metrics"][metric]
            if item["difference"] is not None:
                deltas_by_seed.setdefault(row["seed"], []).append(item["difference"])
            if item["percent_change"] is not None:
                percentages_by_seed.setdefault(row["seed"], []).append(item["percent_change"])
        seed_deltas = [statistics.mean(values) for _, values in sorted(deltas_by_seed.items())]
        seed_percentages = [statistics.mean(values) for _, values in sorted(percentages_by_seed.items())]
        aggregate[metric] = {
            "paired_seed_count": len(seed_deltas),
            "paired_trial_count": sum(map(len, deltas_by_seed.values())),
            "mean_difference_across_seed_means": statistics.mean(seed_deltas) if seed_deltas else None,
            "median_difference_across_seed_means": statistics.median(seed_deltas) if seed_deltas else None,
            "mean_percent_change_across_seed_means": (
                statistics.mean(seed_percentages) if seed_percentages else None),
            **_permutation_summary(seed_deltas),
        }
    return {"contrast": label,
            "direction": "comparison minus reference",
            "paired_trials": paired_rows,
            "aggregate_by_metric": aggregate,
            "confidence_intervals": "omitted; reported differences are descriptive seed-set summaries"}


def _left_right(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["condition"]] = row
    effects: list[dict[str, Any]] = []
    metric_names = ("forward_displacement_mm", "lateral_displacement_mm", "net_yaw_change_deg",
                    "annotated_leg_spikes", "mapped_motor_spikes",
                    "articulated_leg_sensory_spikes", "mean_total_active_muscle_force_N",
                    "mean_peak_abs_joint_delta_rad")
    for seed, conditions in sorted(by_seed.items()):
        if "left" not in conditions or "right" not in conditions:
            continue
        left, right = conditions["left"], conditions["right"]
        effects.append({"seed": seed, "direction": "left minus right",
                        "metrics": {name: _metric_delta(right.get(name), left.get(name))
                                    for name in metric_names}})
    aggregate: dict[str, Any] = {}
    for metric in metric_names:
        values = [entry["metrics"][metric]["difference"] for entry in effects
                  if entry["metrics"][metric]["difference"] is not None]
        aggregate[metric] = {"mean_left_minus_right": statistics.mean(values) if values else None,
                             "median_left_minus_right": statistics.median(values) if values else None,
                             **_permutation_summary(values)}
    return {"model": label, "paired_seed_effects": effects,
            "aggregate_by_metric": aggregate,
            "interpretation": "descriptive directional contrast across deterministic simulation seeds"}


def _moving_minus_static(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["condition"]] = row
    metric_names = ("forward_displacement_mm", "lateral_displacement_mm",
                    "net_yaw_change_deg", "annotated_leg_spikes", "mapped_motor_spikes",
                    "articulated_leg_sensory_spikes")
    effects: list[dict[str, Any]] = []
    for seed, conditions in sorted(by_seed.items()):
        if not all(condition in conditions for condition in CONDITIONS):
            continue
        left, right, static = (conditions[name] for name in CONDITIONS)
        metrics: dict[str, Any] = {}
        for name in metric_names:
            left_value, right_value, static_value = (
                left.get(name), right.get(name), static.get(name))
            moving_mean = (statistics.mean((left_value, right_value))
                           if left_value is not None and right_value is not None else None)
            metrics[name] = _metric_delta(static_value, moving_mean)
        effects.append({"seed": seed, "direction": "mean(left,right) minus static",
                        "metrics": metrics})
    aggregate = {}
    for metric in metric_names:
        values = [effect["metrics"][metric]["difference"] for effect in effects
                  if effect["metrics"][metric]["difference"] is not None]
        aggregate[metric] = {
            "mean_moving_minus_static": statistics.mean(values) if values else None,
            "median_moving_minus_static": statistics.median(values) if values else None,
            **_permutation_summary(values),
        }
    return {"model": label, "paired_seed_effects": effects,
            "aggregate_by_metric": aggregate,
            "interpretation": (
                "descriptive stripe-condition contrast against the within-seed static baseline; "
                "does not isolate causality or establish biological optomotor turning")}


def _select_historical_primary(runs: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the canonical no-repeat run primary; retain named repeats as context."""
    primary: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_condition.setdefault(run["condition"], []).append(run)
    for condition in CONDITIONS:
        group = by_condition.get(condition, [])
        canonical = [run for run in group if run["repeat_index"] is None]
        if len(canonical) != 1:
            raise AnalysisError(
                f"historical Triton {condition}: expected one no-suffix primary run, "
                f"found {len(canonical)}")
        primary.extend(canonical)
        for run in group:
            if run is canonical[0]:
                continue
            if run["repeat_index"] is None:
                raise AnalysisError(f"historical Triton {condition}: ambiguous primary run")
            repeats.append(run)
            skipped.append({"path": str(run["path"]), "model": "historical_triton",
                            "status": "UNSELECTED_CONTEXT_ONLY",
                            "repeat_index": run["repeat_index"],
                            "reason": "canonical no-suffix primary selected; run is summarized only in historical_repeat_context"})
    return primary, repeats


def _repeat_drift_context(primary: list[dict[str, Any]], repeats: list[dict[str, Any]]) -> dict[str, Any]:
    primary_by_condition = {row["condition"]: row for row in primary}
    repeat_by_condition = {row["condition"]: row for row in repeats}
    metric_names = ("forward_displacement_mm", "lateral_displacement_mm", "net_yaw_change_deg",
                    "annotated_leg_spikes")
    per_trial = []
    for condition, repeated in sorted(repeat_by_condition.items()):
        original = primary_by_condition.get(condition)
        if original is None:
            continue
        per_trial.append({"seed": original["seed"], "condition": condition,
                          "repeat_index": repeated.get("repeat_index"),
                          "primary_run": original["run_id"], "repeat_run": repeated["run_id"],
                          "metrics_repeat_minus_primary": {
                              metric: _metric_delta(original.get(metric), repeated.get(metric))
                              for metric in metric_names
                          },
                          "primary_manifest_sha256": original["integrity"]["manifest_sha256"],
                          "repeat_manifest_sha256": repeated["integrity"]["manifest_sha256"]})
    yaw_pair = None
    primary_rows = {row["condition"]: row for row in primary}
    if all(condition in primary_rows and condition in repeat_by_condition
           for condition in ("left", "right")):
        primary_effect = (primary_rows["left"]["net_yaw_change_deg"]
                          - primary_rows["right"]["net_yaw_change_deg"])
        repeat_effect = (repeat_by_condition["left"]["net_yaw_change_deg"]
                         - repeat_by_condition["right"]["net_yaw_change_deg"])
        yaw_pair = {"primary_left_minus_right_deg": primary_effect,
                    "repeat_left_minus_right_deg": repeat_effect,
                    "repeat_minus_primary_directional_effect_deg": repeat_effect - primary_effect,
                    "interpretation": "single-seed repeat drift; descriptive only"}
    return {"selection": "canonical historical runs without a repeat suffix are primary",
            "repeat_runs": [{"run_id": row["run_id"], "path": row["path"],
                             "seed": row["seed"], "condition": row["condition"],
                             "repeat_index": row.get("repeat_index"),
                             "manifest_sha256": row["integrity"]["manifest_sha256"],
                             "telemetry_sha256": row["integrity"]["telemetry_sha256"]}
                            for row in repeats],
            "repeat_minus_primary_by_condition": per_trial,
            "left_right_yaw_effect_drift": yaw_pair,
            "inference": "contextual repeat drift only; not a second independent seed"}


def build_report(run_summaries: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in run_summaries:
        key = row["model"] if row["model"] != "articulated" else f"articulated_{row['mode']}"
        by_model.setdefault(key, []).append(row)
    current = by_model.get("reduced_torch", [])
    articulated_modes = {row["mode"] for row in run_summaries if row["model"] == "articulated"}
    contrasts: dict[str, Any] = {}
    for mode in sorted(articulated_modes):
        articulated = [row for row in run_summaries
                       if row["model"] == "articulated" and row["mode"] == mode]
        contrasts[f"reduced_torch_vs_articulated_{mode}"] = _paired_contrast(
            current, articulated, label=f"reduced_torch -> articulated_{mode}")
    if "on" in articulated_modes and "off" in articulated_modes:
        off = [row for row in run_summaries if row["model"] == "articulated" and row["mode"] == "off"]
        on = [row for row in run_summaries if row["model"] == "articulated" and row["mode"] == "on"]
        contrasts["articulated_off_vs_on"] = _paired_contrast(
            off, on, label="articulated_off -> articulated_on")

    left_right = {}
    moving_vs_static = {}
    for model, rows in by_model.items():
        left_right[model] = _left_right(rows, model)
        moving_vs_static[model] = _moving_minus_static(rows, model)
    historical = {name: rows for name, rows in by_model.items()
                  if name in {"historical_triton", "reduced_torch"}}
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "digital_simulation",
        "biological_validation": "NOT_VALIDATED",
        "interpretation_policy": (
            "Trials are deterministic simulation seeds. Frames are summarized within trial and are not "
            "independent replicates. For fewer than five paired seeds, exact tests are not computed; "
            "no biological or population-level significance is claimed."),
        "protocol": {"duration_s": EXPECTED_DURATION_S, "output_dt_s": EXPECTED_DT_S,
                     "frames_per_trial": EXPECTED_FRAMES,
                     "conditions": list(CONDITIONS),
                     "primary_contrast": "current reduced Torch CUDA vs articulated on-mode",
                     "secondary_contrasts": ["articulated legacy/off/on modes when available"],
                     "paired_direction": "left minus right"},
        "trials": run_summaries,
        "condition_orientation_note": (
            "The retained stripe_left/stripe_right labels encode world-Y stimulus velocity signs (+Y/-Y). "
            "The camera's +Y-right convention and the articulated body's +Y-anatomical-left convention "
            "leave body-turn handedness unresolved; yaw is reported with its signed world convention, and "
            "these contrasts do not imply a biological turn direction."),
        "initial_body_height_note": (
            "Body initial height is model-dependent in these recorded runs. Current reduced-body "
            "runs start at 1.2 mm, while the articulated body starts at its recorded root height; "
            "displacement is always measured from each trial's own true pre-step initial pose."),
        "wall_time_comparison_note": (
            "Wall-time stage scopes differ: current reduced-body rows use the compact telemetry path, "
            "while articulated rows write the full body and per-tick spike traces and include final "
            "device synchronization. These measurements describe each recorded pipeline and do not "
            "isolate mechanics as the cause of a timing difference."),
        "left_right_by_seed": left_right,
        "moving_mean_minus_static_by_seed": moving_vs_static,
        "paired_model_contrasts": contrasts,
        "historical_context": {
            "models": {name: [{"seed": row["seed"], "condition": row["condition"],
                                "forward_displacement_mm": row["forward_displacement_mm"],
                                "lateral_displacement_mm": row["lateral_displacement_mm"],
                                "net_yaw_change_deg": row["net_yaw_change_deg"]}
                               for row in rows]
                       for name, rows in historical.items()},
            "source_snapshot_note": (
                "Historical reduced-body manifests do not contain a complete source snapshot; "
                "their manifest and telemetry hashes are recorded for provenance."),
        },
        "skipped_runs": skipped,
    }


def plot_report(report: Mapping[str, Any], path: Path) -> None:
    """Create a standalone descriptive PNG without treating frames as samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise AnalysisError("matplotlib is required to write the requested PNG") from exc
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in report["trials"]:
        model = row["model"] if row["model"] != "articulated" else f"articulated_{row['mode']}"
        groups.setdefault(model, []).append(row)
    metrics = (("forward_displacement_mm", "Forward displacement (mm)"),
               ("lateral_displacement_mm", "Lateral displacement (mm)"),
               ("net_yaw_change_deg", "Unwrapped yaw change (deg)"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    models = sorted(groups)
    width = 0.78 / max(len(models), 1)
    colors = plt.get_cmap("tab10")
    for axis, (metric, title) in zip(axes, metrics):
        for model_index, model in enumerate(models):
            entries = groups[model]
            x = {condition: index for index, condition in enumerate(CONDITIONS)}
            offset = (model_index - (len(models) - 1) / 2) * width
            for condition in CONDITIONS:
                values = [float(row[metric]) for row in entries
                          if row["condition"] == condition and row.get(metric) is not None]
                if values:
                    axis.bar(x[condition] + offset, statistics.mean(values), width=width,
                             color=colors(model_index % 10),
                             label=model if condition == CONDITIONS[0] else None,
                             alpha=0.88)
                    axis.text(x[condition] + offset, statistics.mean(values), f"n={len(values)}",
                              ha="center", va="bottom", fontsize=7, rotation=90)
        axis.axhline(0, color="#333333", linewidth=0.7)
        axis.set_xticks(range(len(CONDITIONS)),
                        ("stripe_left\n(world +Y)", "stripe_right\n(world -Y)", "stripe_static"))
        axis.set_title(title)
        axis.set_xlabel("stripe condition")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Optomotor ground-crawl comparison — descriptive seed summaries")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_inputs(historical_triton: Path, reduced_torch: Path, articulated_root: Path,
                   *, seed: int, articulated_off_root: Path | None = None,
                   articulated_legacy_root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    skipped: list[dict[str, Any]] = []
    triton_all = discover_runs(historical_triton, label="historical_triton", seed=seed, skipped=skipped)
    triton, repeat_runs = _select_historical_primary(triton_all, skipped)
    reduced = discover_runs(reduced_torch, label="reduced_torch", seed=seed, skipped=skipped)
    articulated = discover_runs(articulated_root, label="articulated", seed=seed, skipped=skipped)
    for expected_mode, extra_root in (("off", articulated_off_root),
                                      ("legacy", articulated_legacy_root)):
        if extra_root is None:
            continue
        extra_runs = discover_runs(extra_root, label="articulated", seed=seed, skipped=skipped)
        if any(run["mode"] != expected_mode for run in extra_runs):
            raise AnalysisError(
                f"{expected_mode} comparison root contains a trial with another sensory mode")
        articulated.extend(extra_runs)
    selected = triton + reduced + articulated
    # Exactly one completed trial per seed/condition/mode avoids accidental
    # selection of an arbitrary repeat or a failed first attempt.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in selected:
        key = (run["model"], run["condition"], run["mode"])
        if key in unique:
            raise AnalysisError(
                f"multiple completed runs match model/condition/mode {key}; select a single run root")
        unique[key] = run
    by_model_condition: dict[tuple[str, str], int] = {}
    for run in selected:
        by_model_condition[(run["model"], run["condition"])] = 1
    required = {"historical_triton", "reduced_torch"}
    for model in required:
        found = {run["condition"] for run in selected if run["model"] == model}
        if found != set(CONDITIONS):
            raise AnalysisError(f"{model} requires left/right/static; found {sorted(found)}")
    modes = {run["mode"] for run in articulated}
    found_on = {run["condition"] for run in articulated if run["mode"] == "on"}
    if found_on != set(CONDITIONS):
        raise AnalysisError(f"articulated mode 'on' requires left/right/static; found {sorted(found_on)}")
    if "on" not in modes:
        raise AnalysisError("articulated root has no completed mode='on' set")
    summaries = [analyze_trial(run) for run in selected]
    repeat_summaries = [analyze_trial(run) for run in repeat_runs]
    report = build_report(summaries, skipped)
    report["historical_repeat_context"] = _repeat_drift_context(
        [row for row in summaries if row["model"] == "historical_triton"], repeat_summaries)
    return report, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-triton", type=Path, required=True,
                        help="historical strict 7.5 s reduced-body Triton run root")
    parser.add_argument("--reduced-torch", type=Path, required=True,
                        help="current strict 7.5 s reduced-body Torch CUDA run root")
    parser.add_argument("--articulated", type=Path, required=True,
                        help="articulated optomotor output root; may contain legacy/off/on modes")
    parser.add_argument("--articulated-off", type=Path,
                        help="optional separate root for matched articulated off-mode trials")
    parser.add_argument("--articulated-legacy", type=Path,
                        help="optional separate root for articulated legacy mechanism controls")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory for comparison.json and comparison.png")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f"--out already exists; refusing to overwrite: {out}")
    report, _ = analyze_inputs(args.historical_triton, args.reduced_torch,
                               args.articulated, seed=args.seed,
                               articulated_off_root=args.articulated_off,
                               articulated_legacy_root=args.articulated_legacy)
    out.mkdir(parents=True)
    json_path = out / "comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    plot_path = out / "comparison.png"
    plot_report(report, plot_path)
    print(json.dumps({"report": str(json_path), "plot": str(plot_path),
                      "trials": len(report["trials"]), "skipped_runs": len(report["skipped_runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
