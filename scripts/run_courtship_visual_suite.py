#!/usr/bin/env python3
"""Run a fail-closed strict CUDA visual-courtship assay matrix.

The default protocol runs matched moving, static, and hidden female-target
conditions for each seed.  A separate ``--courtship-movie`` mode runs one
open-loop visual probe and deliberately does not load a scene manifest.  Each
trial runs in its own user-systemd unit with ``MemoryMax=5G``.  Raw launcher
manifests, compact telemetry, trajectories, commands, and logs are retained.

This is a visual orientation/tracking assay, not evidence of complete or
autonomous courtship. CPU, stub-vision, analytic-renderer, and other fallback
runs are rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_digital_fly.py"
ANALYZER = ROOT / "scripts" / "analyze_courtship_visual.py"
PYTHON = Path("/data/environments/projects/flyvis-opticlobe/bin/python")
DEFAULT_CONFIG = ROOT / "config" / "courtship_visual_experiment.json"
DEFAULT_STIMULUS_CONFIG = ROOT / "config" / "courtship_visual_stimuli.json"
DEFAULT_SCENE_MANIFEST = ROOT / "config" / "courtship_female_scene.json"
DEFAULT_TRAJECTORY = ROOT / "config" / "courtship_simon2010_female_trajectory.json"
DEFAULT_GEOMETRY = ROOT / "config" / "courtship_target_geometry.json"
DEFAULT_MOVIE = Path("/data/datasets/raw/malecns-gpu/courtship/simon2010/movS2.avi")
DEFAULT_OUT_ROOT = Path("/data/artifacts/by-project/malecns-gpu/experiments")
FRAME_DT_S = 0.005
DURATION_S = 2.5
MAX_VRAM_GIB = 15.5
MEMORY_MAX = "5G"
DEFAULT_SEEDS = (1729, 1731)
TARGET_CONDITIONS = ("moving", "static", "hidden")


@dataclass(frozen=True)
class TrialSpec:
    seed: int
    condition: str
    label: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return raw


def _resolve_file(path: Path, description: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{description} is missing: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{description} is not a regular file: {resolved}")
    return resolved


def _validate_config(config_path: Path, stimulus_config_path: Path,
                     trajectory_path: Path | None, duration_s: float,
                     geometry_path: Path | None = None
                     ) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    config = _read_json(config_path, "courtship experiment config")
    if config.get("schema_version") != "courtship-visual-experiment-1":
        raise RuntimeError("experiment config must use schema_version courtship-visual-experiment-1")
    if config.get("stimulus_name") != "female_placeholder":
        raise RuntimeError("courtship experiment config must select stimulus_name=female_placeholder")
    experiment = config.get("experiment") or {}
    if not isinstance(experiment, Mapping):
        raise RuntimeError("courtship experiment config experiment section must be an object")
    if experiment.get("resting_start") is not True:
        raise RuntimeError("courtship visual assay requires experiment.resting_start=true")
    if experiment.get("stimulus_interaction", "visual_only_non_interacting") != "visual_only_non_interacting":
        raise RuntimeError("courtship visual assay requires visual_only_non_interacting stimuli")
    if experiment.get("allow_legacy_wing_fallback") is not False:
        raise RuntimeError("courtship visual assay must disable legacy wing fallback")
    baseline = experiment.get("visual_baseline") or {}
    if (baseline.get("mode") != "initial_scene_adapted"
            or baseline.get("feed_malecns") is not False
            or baseline.get("advance_body") is not False):
        raise RuntimeError("visual baseline must adapt FlyVis without feeding MaleCNS or advancing the body")
    camera = experiment.get("camera")
    if not isinstance(camera, Mapping):
        raise RuntimeError("courtship experiment.camera must define the courtship-only camera contract")
    expected_camera: dict[str, float] = {}
    for name, value in camera.items():
        if isinstance(value, bool):
            raise RuntimeError(f"courtship camera field {name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"courtship camera field {name} must be numeric") from exc
        if not math.isfinite(number):
            raise RuntimeError(f"courtship camera field {name} must be finite")
        expected_camera[str(name)] = number
    near_m = expected_camera.get("near_m")
    eye_separation_m = expected_camera.get("eye_separation_m")
    if near_m is None or not 0.0 < near_m < 0.001:
        raise RuntimeError("courtship camera near_m must be present and less than 1 mm")
    if eye_separation_m is None or not 0.0 < eye_separation_m < 0.002:
        raise RuntimeError("courtship camera eye_separation_m must be present and less than 2 mm")
    if expected_camera.get("far_m", 8.0) <= near_m:
        raise RuntimeError("courtship camera far_m must be greater than near_m")
    scenario = (config.get("scenarios") or {}).get("courtship_visual")
    if not isinstance(scenario, Mapping):
        raise RuntimeError("experiment config must define scenarios.courtship_visual")
    try:
        scenario_duration = float(scenario.get("duration_s", float("nan")))
        scenario_dt = float(scenario.get("dt_s", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("courtship scenario duration and dt must be numeric") from exc
    if not math.isfinite(scenario_duration) or abs(scenario_duration - duration_s) > 1.0e-12:
        raise RuntimeError("courtship scenario duration must match the suite duration")
    if not math.isfinite(scenario_dt) or abs(scenario_dt - FRAME_DT_S) > 1.0e-12:
        raise RuntimeError("courtship scenario must use the 5 ms frame grid")
    wind = ((config.get("world") or {}).get("wind") or {})
    base = wind.get("base_mps", (0.0, 0.0, 0.0))
    if (not isinstance(base, (list, tuple)) or len(base) != 3
            or any(not math.isfinite(float(value)) or abs(float(value)) > 1.0e-12 for value in base)
            or float(wind.get("spatial_amplitude_mps", 0.0)) != 0.0
            or wind.get("gusts", [])):
        raise RuntimeError("courtship suite requires zero base/spatial/gust wind")

    stimuli = _read_json(stimulus_config_path, "courtship stimulus config")
    if stimuli.get("schema_version") != "physical-stimulus-protocol-1":
        raise RuntimeError("courtship stimulus config schema is not physical-stimulus-protocol-1")
    placeholder = (stimuli.get("stimuli") or {}).get("female_placeholder")
    if not isinstance(placeholder, Mapping):
        raise RuntimeError("stimulus config must define female_placeholder")
    if (placeholder.get("renderable") is not False
            or placeholder.get("collidable") is not False
            or placeholder.get("visual_only") is not True
            or placeholder.get("non_interacting") is not True):
        raise RuntimeError("female_placeholder must be invisible, non-colliding, and non-interacting")

    if trajectory_path is not None:
        trajectory = _read_json(trajectory_path, "female target trajectory")
        if trajectory.get("schema_version") != "female-target-trajectory-1":
            raise RuntimeError("target trajectory schema must be female-target-trajectory-1")
        keyframes = trajectory.get("keyframes")
        if not isinstance(keyframes, list) or not keyframes:
            raise RuntimeError("target trajectory must have at least one keyframe")
        if geometry_path is not None:
            sys.path.insert(0, str(ROOT / "scripts"))
            try:
                from courtship_target import FemaleTargetGeometry  # noqa: PLC0415
                FemaleTargetGeometry.from_json(geometry_path)
            except (ImportError, ValueError) as exc:
                raise RuntimeError(f"invalid female target geometry config {geometry_path}: {exc}") from exc
    return config, stimuli, expected_camera


def _validate_scene_manifest(path: Path, *, expected_object_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # Use the production parser so the suite catches missing geometry assets,
    # unsupported manifest fields, and a mismatched stimulus binding before
    # starting any GPU work.
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from scene_asset_manifest import load_scene_manifest  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(f"scene manifest parser is unavailable: {exc}") from exc
    try:
        manifest = load_scene_manifest(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid female scene manifest {path}: {exc}") from exc
    bindings = manifest.simulation.get("stimulus_object_ids") or {}
    object_id = bindings.get("female_fly")
    if not object_id:
        raise RuntimeError("scene manifest must bind stimulus id female_fly to a visual mesh")
    if object_id != expected_object_id:
        raise RuntimeError(
            f"scene female_fly binding must reference geometry object_id {expected_object_id!r}, "
            f"got {object_id!r}"
        )
    asset = next((item for item in manifest.assets if item.object_id == object_id), None)
    if asset is None:
        raise RuntimeError(f"female_fly binding references absent scene asset {object_id!r}")
    if asset.role not in {"visual", "visual_collision"}:
        raise RuntimeError("female_fly scene asset must have a visual role")
    if not asset.path.is_file():
        raise RuntimeError(f"female fly mesh asset is missing: {asset.path}")
    if asset.integrity_sha256 and sha256_file(asset.path) != asset.integrity_sha256:
        raise RuntimeError(f"female fly mesh SHA-256 does not match scene manifest: {asset.path}")
    return manifest.to_dict(), {
        "asset_object_id": asset.object_id,
        "asset_path": str(asset.path),
        "asset_sha256": sha256_file(asset.path),
        "asset_manifest_sha256": sha256_file(path),
    }


def runtime_environment() -> dict[str, str]:
    if not PYTHON.is_file():
        raise RuntimeError(f"registered FlyVis GPU environment is missing: {PYTHON}")
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT / "scripts"),
        "FLYVIS_ROOT_DIR": "/data/models/flyvis",
        "WARP_CACHE_PATH": "/data/cache/malecns-gpu/warp",
    })
    return env


def _run_json(command: list[str], *, env: Mapping[str, str], timeout_s: float = 90.0
              ) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        command, cwd=ROOT, env=dict(env), check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"preflight command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"preflight did not return JSON: {completed.stdout[-4000:]}") from exc
    if not isinstance(report, dict):
        raise RuntimeError("preflight report must be a JSON object")
    return report, completed.stderr


def gpu_and_memory_preflight(env: Mapping[str, str]) -> dict[str, Any]:
    report, stderr = _run_json([
        str(PYTHON), str(ROOT / "scripts" / "twin_preflight.py"),
        "--device", "cuda", "--backend", "triton", "--json",
    ], env=env)
    gate = report.get("gate") or {}
    if not report.get("gate_passed") or not gate.get("open"):
        raise RuntimeError(f"MaleCNS CUDA gate is closed: {gate}")
    capabilities = report.get("capabilities") or {}
    for name in ("flyvis", "torch", "triton"):
        capability = capabilities.get(name) or {}
        if not capability.get("installed") or not capability.get("available"):
            raise RuntimeError(f"required CUDA capability {name!r} is unavailable: {capability}")
    gpu_fact = (gate.get("facts") or {}).get("gpu") or {}
    gpu_query = str(gpu_fact.get("nvidia-smi", ""))
    if not gpu_query or gpu_query.startswith(("missing", "error:", "exit ")):
        raise RuntimeError(f"CUDA preflight did not verify a live NVIDIA GPU: {gpu_fact}")

    systemd_run = shutil.which("systemd-run")
    if not systemd_run:
        raise RuntimeError("systemd-run is unavailable; cannot enforce MemoryMax=5G")
    cap_check = subprocess.run(
        [systemd_run, "--user", "--wait", "--pipe", "--collect",
         "--property=MemoryMax=5G", "/usr/bin/true"],
        cwd=ROOT, env=dict(env), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=45,
    )
    if cap_check.returncode != 0:
        raise RuntimeError(
            "user systemd could not enforce the MemoryMax=5G preflight unit: "
            f"{cap_check.stderr[-2000:]}"
        )
    report["resource_cap_preflight"] = {
        "MemoryMax": MEMORY_MAX,
        "systemd_run_user_check": "passed",
        "systemd_run_stderr": cap_check.stderr.strip(),
    }
    report["preflight_stderr"] = stderr.strip()
    report["environment"] = {
        "python": str(PYTHON),
        "flyvis_root": env["FLYVIS_ROOT_DIR"],
        "warp_cache": env["WARP_CACHE_PATH"],
    }
    return report


def _source_hashes() -> dict[str, str]:
    names = (
        "scripts/run_courtship_visual_suite.py",
        "scripts/run_digital_fly.py",
        "scripts/courtship_target.py",
        "scripts/courtship_movie.py",
        "scripts/scene_asset_manifest.py",
        "scripts/gltf_mesh_loader.py",
        "scripts/malecns_engine.py",
        "scripts/neuromuscular_adapter.py",
        "scripts/flyvis_runtime_adapter.py",
        "scripts/virtual_world.py",
        "scripts/warp_world_backend.py",
        "scripts/flight_dynamics.py",
        "config/neuromuscular_map.json",
        "config/courtship_target_geometry.json",
    )
    missing = [name for name in names if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"courtship suite source inputs are missing: {missing}")
    return {name: sha256_file(ROOT / name) for name in names}


def _snapshot_inputs(out: Path, files: Mapping[str, Path]) -> dict[str, Any]:
    snapshot_root = out / "inputs"
    records: dict[str, Any] = {}
    for key, source in files.items():
        source = source.resolve(strict=True)
        destination = snapshot_root / key / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        snapshot_hash = sha256_file(destination)
        if source_hash != snapshot_hash:
            raise RuntimeError(f"input changed while snapshotting: {source}")
        records[key] = {
            "source_path": str(source),
            "snapshot_path": str(destination),
            "sha256": snapshot_hash,
        }
    return records


def _record_hashed_inputs(files: Mapping[str, Path]) -> dict[str, Any]:
    return {
        key: {
            "source_path": str(path.resolve()),
            "snapshot_path": None,
            "sha256": sha256_file(path),
            "snapshot_policy": "hash_only_large_or_external_asset",
        }
        for key, path in files.items()
    }


def _assert_unchanged(files: Mapping[str, Path], expected: Mapping[str, str], label: str) -> None:
    for key, path in files.items():
        current = sha256_file(path)
        if current != expected[key]:
            raise RuntimeError(f"{label}: input {key} changed after suite snapshot")


def make_trial_specs(seeds: Sequence[int], conditions: Sequence[str]) -> tuple[TrialSpec, ...]:
    return tuple(
        TrialSpec(seed=int(seed), condition=condition, label=f"seed-{int(seed)}-{condition}")
        for seed in seeds for condition in conditions
    )


def _base_launcher_args(*, config: Path, stimulus_config: Path, run_dir: Path,
                        seed: int, duration_s: float) -> list[str]:
    return [
        str(PYTHON), str(RUNNER),
        "--scenario", "courtship_visual",
        "--mode", "free-behavior",
        "--config", str(config),
        "--out", str(run_dir),
        "--seed", str(seed),
        "--duration", str(duration_s),
        "--stimulus-config", str(stimulus_config),
        "--neural", "malecns", "--vision", "flyvis", "--body", "reduced",
        "--renderer-backend", "warp_cuda",
        "--neural-device", "cuda", "--neural-backend", "triton",
        "--vision-device", "cuda:0", "--max-vram-gib", str(MAX_VRAM_GIB),
        "--telemetry-mode", "compact",
    ]


def build_trial_command(*, spec: TrialSpec, trial_dir: Path, config: Path,
                        stimulus_config: Path, scene_manifest: Path | None,
                        trajectory: Path | None, movie: Path | None,
                        duration_s: float) -> list[str]:
    run_dir = trial_dir / "run"
    command = _base_launcher_args(
        config=config, stimulus_config=stimulus_config,
        run_dir=run_dir, seed=spec.seed, duration_s=duration_s,
    )
    if movie is not None:
        if scene_manifest is not None or trajectory is not None:
            raise ValueError("movie trial cannot include a scene manifest or target trajectory")
        command += ["--courtship-movie", str(movie)]
        return command
    if scene_manifest is None or trajectory is None:
        raise ValueError("target trial requires both a female scene manifest and trajectory")
    if spec.condition not in TARGET_CONDITIONS:
        raise ValueError(f"unknown courtship target condition: {spec.condition}")
    command += [
        "--scene-manifest", str(scene_manifest),
        "--courtship-target-trajectory", str(trajectory),
        "--courtship-target-condition", spec.condition,
    ]
    return command


def _check_manifest(manifest: Mapping[str, Any], *, spec: TrialSpec,
                    config: Path, stimulus_config: Path,
                    scene_manifest: Path | None, trajectory: Path | None,
                    movie: Path | None, trajectory_sha256: str | None,
                    movie_sha256: str | None,
                    expected_camera: Mapping[str, float]) -> None:
    if manifest.get("schema_version") != "digital-fly-run-1":
        raise RuntimeError(f"{spec.label}: unexpected launcher manifest schema")
    if int(manifest.get("seed", -1)) != spec.seed:
        raise RuntimeError(f"{spec.label}: manifest seed mismatch")
    if Path(str(manifest.get("config", ""))).resolve() != config.resolve():
        raise RuntimeError(f"{spec.label}: config path mismatch")
    if manifest.get("execution_policy") != "strict-free-behavior-escape-experiment":
        raise RuntimeError(f"{spec.label}: run was not strict free-behavior")
    if manifest.get("neural_intervention") is not None:
        raise RuntimeError(f"{spec.label}: unexpected direct neural intervention")
    expected_class = (
        "courtship_open_loop_visual_probe" if movie is not None
        else "courtship_closed_loop_visual_assay"
    )
    if manifest.get("run_classification") != expected_class:
        raise RuntimeError(f"{spec.label}: courtship run classification mismatch")
    autonomy = manifest.get("autonomy_policy") or {}
    if (autonomy.get("strict") is not True
            or autonomy.get("control_authority") != "MaleCNS"
            or autonomy.get("counts_as_autonomous_courtship") is not False):
        raise RuntimeError(f"{spec.label}: strict MaleCNS control policy is missing")
    declared_backends = manifest.get("effective_backends") or {}
    if (declared_backends.get("autopilot") != "prohibited"
            or declared_backends.get("flight_state_drive") != "disabled"
            or declared_backends.get("flight_body_stabilizer") != "disabled"):
        raise RuntimeError(f"{spec.label}: strict run enabled a prohibited controller or stabilizer")
    if manifest.get("requested_backends") != {
            "neural": "malecns", "vision": "flyvis", "body": "reduced"}:
        raise RuntimeError(f"{spec.label}: requested backend set changed")
    if manifest.get("requested_renderer_backend") != "warp_cuda":
        raise RuntimeError(f"{spec.label}: requested renderer is not Warp CUDA")
    effective = manifest.get("effective_backends") or {}
    execution = manifest.get("effective_execution") or {}
    if (effective.get("neural") != "triton"
            or effective.get("vision") in (None, "stub", "unknown")
            or str(effective.get("vision", "")).lower().find("stub") >= 0
            or effective.get("renderer") != "warp_cuda"
            or execution.get("renderer_backend") != "warp_cuda"
            or execution.get("neural_device") != "cuda"
            or execution.get("neural_backend") != "triton"
            or not str(execution.get("vision_device", "")).startswith("cuda:")):
        raise RuntimeError(
            f"{spec.label}: requested CUDA path did not take effect: "
            f"effective_backends={effective!r}, effective_execution={execution!r}"
        )
    requested = manifest.get("requested_execution") or {}
    if (requested.get("strict") is not True
            or requested.get("mode") != "strict-free-behavior-escape-experiment"
            or requested.get("renderer_backend") != "warp_cuda"
            or requested.get("max_vram_gib") != MAX_VRAM_GIB
            or execution.get("max_vram_gib") != MAX_VRAM_GIB
            or requested.get("neural_device") != "cuda"
            or requested.get("neural_backend") != "triton"
            or requested.get("vision_device") != "cuda:0"):
        raise RuntimeError(f"{spec.label}: explicit CUDA/15.5 GiB request missing from manifest")
    if requested.get("telemetry_mode") != "compact":
        raise RuntimeError(f"{spec.label}: compact telemetry was not requested")
    telemetry_contract = manifest.get("telemetry_contract") or {}
    if telemetry_contract.get("mode") != "compact":
        raise RuntimeError(f"{spec.label}: compact telemetry contract mismatch")
    if Path(str(requested.get("stimulus_config", ""))).resolve() != stimulus_config.resolve():
        raise RuntimeError(f"{spec.label}: stimulus config path mismatch")
    courtship = manifest.get("courtship_visual") or {}
    effective_camera = courtship.get("camera") or {}
    if not isinstance(effective_camera, Mapping):
        raise RuntimeError(f"{spec.label}: manifest omitted the effective courtship camera")
    for name, expected in expected_camera.items():
        try:
            actual = float(effective_camera.get(name))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{spec.label}: manifest camera field {name} is missing or invalid") from exc
        if not math.isfinite(actual) or abs(actual - expected) > 1.0e-12:
            raise RuntimeError(
                f"{spec.label}: manifest camera {name}={actual!r} differs from configured {expected!r}"
            )
    if (float(effective_camera.get("near_m", float("inf"))) >= 0.001
            or float(effective_camera.get("eye_separation_m", float("inf"))) >= 0.002):
        raise RuntimeError(f"{spec.label}: manifest camera would clip/offset the nearby target")
    if movie is None:
        if (courtship.get("schema_version") != "courtship-visual-assay-1"
                or courtship.get("target_source") != "independent_world_trajectory"
                or courtship.get("target_condition") != spec.condition
                or courtship.get("trajectory_sha256") != trajectory_sha256
                or courtship.get("control_authority") != "MaleCNS"
                or courtship.get("target_labels_enter_control") is not False
                or not isinstance(courtship.get("asset_binding"), Mapping)):
            raise RuntimeError(f"{spec.label}: closed-loop courtship manifest contract mismatch")
        if Path(str(courtship.get("trajectory_path", ""))).resolve() != trajectory.resolve():
            raise RuntimeError(f"{spec.label}: trajectory path mismatch")
        if (Path(str(requested.get("scene_manifest", ""))).resolve()
                != scene_manifest.resolve()):
            raise RuntimeError(f"{spec.label}: scene manifest path mismatch")
        if (Path(str(requested.get("courtship_target_trajectory", ""))).resolve()
                != trajectory.resolve()
                or requested.get("courtship_target_condition") != spec.condition):
            raise RuntimeError(f"{spec.label}: requested target condition/trajectory mismatch")
        if requested.get("courtship_movie") is not None:
            raise RuntimeError(f"{spec.label}: unexpected movie input in target trial")
    else:
        if (courtship.get("schema_version") != "courtship-visual-assay-1"
                or courtship.get("target_source") != "open_loop_movie"
                or courtship.get("movie_sha256") != movie_sha256
                or courtship.get("control_authority") != "MaleCNS"):
            raise RuntimeError(f"{spec.label}: open-loop movie manifest contract mismatch")
        if Path(str(courtship.get("movie_path", ""))).resolve() != movie.resolve():
            raise RuntimeError(f"{spec.label}: movie path mismatch")
        if requested.get("scene_manifest") is not None:
            raise RuntimeError(f"{spec.label}: movie mode unexpectedly loaded a scene manifest")
        if requested.get("courtship_target_trajectory") is not None:
            raise RuntimeError(f"{spec.label}: movie mode unexpectedly loaded a target trajectory")
        if Path(str(requested.get("courtship_movie", ""))).resolve() != movie.resolve():
            raise RuntimeError(f"{spec.label}: requested movie path mismatch")


def _validate_telemetry(path: Path, *, label: str, duration_s: float) -> tuple[int, str]:
    digest = hashlib.sha256()
    expected_frames = round(duration_s / FRAME_DT_S)
    count = 0
    previous_time: float | None = None
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"{label}: invalid telemetry JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(f"{label}: telemetry line {line_number} must be an object")
            if row.get("scenario") != "free-behavior":
                raise RuntimeError(f"{label}: telemetry escaped the strict free-behavior scenario")
            frame_index = row.get("frame_index")
            frame_time = row.get("frame_time_s")
            if (isinstance(frame_index, bool) or not isinstance(frame_index, int)
                    or frame_index != count):
                raise RuntimeError(f"{label}: telemetry frame index discontinuity on line {line_number}")
            try:
                timestamp = float(frame_time)
                dt_s = float(row.get("dt_s"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{label}: telemetry time fields are missing or invalid") from exc
            if not math.isfinite(dt_s) or abs(dt_s - FRAME_DT_S) > 1.0e-12:
                raise RuntimeError(f"{label}: telemetry frame dt is not 5 ms")
            if not math.isfinite(timestamp) or abs(timestamp - count * FRAME_DT_S) > 1.0e-9:
                raise RuntimeError(f"{label}: telemetry timestamp is off the 5 ms frame grid")
            if previous_time is not None and abs(timestamp - previous_time - FRAME_DT_S) > 1.0e-9:
                raise RuntimeError(f"{label}: telemetry timestamps are discontinuous")
            previous_time = timestamp
            count += 1
    if count != expected_frames:
        raise RuntimeError(f"{label}: expected {expected_frames} telemetry frames, got {count}")
    return count, digest.hexdigest()


def run_trial(*, out: Path, spec: TrialSpec, config: Path,
              stimulus_config: Path, scene_manifest: Path | None,
              trajectory: Path | None, movie: Path | None,
              duration_s: float, env: Mapping[str, str], timeout_s: float) -> Path:
    trial = out / "trials" / f"seed-{spec.seed}" / spec.condition
    trial.mkdir(parents=True, exist_ok=False)
    run_dir = trial / "run"
    unit = f"malecns-courtship-{spec.seed}-{spec.condition}-{uuid.uuid4().hex[:8]}"
    systemd_run = shutil.which("systemd-run")
    if not systemd_run:
        raise RuntimeError("systemd-run disappeared after successful resource preflight")
    launcher_args = build_trial_command(
        spec=spec, trial_dir=trial, config=config,
        stimulus_config=stimulus_config, scene_manifest=scene_manifest,
        trajectory=trajectory, movie=movie, duration_s=duration_s,
    )
    command = [
        systemd_run, "--user", "--wait", "--pipe", "--collect",
        f"--unit={unit}",
        f"--property=MemoryMax={MEMORY_MAX}",
        "--property=LimitMEMLOCK=64M",
        "--property=RuntimeMaxSec=900",
        f"--working-directory={ROOT}",
        f"--setenv=PYTHONPATH={env['PYTHONPATH']}",
        f"--setenv=FLYVIS_ROOT_DIR={env['FLYVIS_ROOT_DIR']}",
        f"--setenv=WARP_CACHE_PATH={env['WARP_CACHE_PATH']}",
        *launcher_args,
    ]
    write_json(trial / "command.json", {
        "argv": command,
        "launcher_argv": launcher_args,
        "environment": {key: env[key] for key in (
            "PYTHONPATH", "FLYVIS_ROOT_DIR", "WARP_CACHE_PATH",
        )},
        "host_memory_limit": MEMORY_MAX,
        "max_vram_gib": MAX_VRAM_GIB,
        "timeout_s": timeout_s,
    })
    started = time.monotonic()
    try:
        with (trial / "stdout.log").open("w", encoding="utf-8") as stdout, \
                (trial / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=ROOT, env=dict(env), stdout=stdout, stderr=stderr,
                text=True, check=False, timeout=timeout_s,
            )
    except subprocess.TimeoutExpired as exc:
        write_json(trial / "process.json", {
            "return_code": None, "timed_out": True,
            "wall_time_s": time.monotonic() - started, "systemd_unit": unit,
        })
        raise RuntimeError(f"{spec.label} timed out; see {trial / 'stderr.log'}") from exc
    write_json(trial / "process.json", {
        "return_code": completed.returncode,
        "wall_time_s": time.monotonic() - started,
        "systemd_unit": unit,
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"{spec.label} failed with exit {completed.returncode}; see {trial / 'stderr.log'}"
        )
    for filename in ("manifest.json", "telemetry.jsonl", "trajectory.json"):
        if not (run_dir / filename).is_file():
            raise RuntimeError(f"{spec.label}: successful launcher omitted {run_dir / filename}")
    return trial


def _record_trial(trial: Path, *, spec: TrialSpec, config: Path,
                  stimulus_config: Path, scene_manifest: Path | None,
                  trajectory: Path | None, movie: Path | None,
                  duration_s: float, trajectory_sha256: str | None,
                  movie_sha256: str | None, source_hashes: Mapping[str, str],
                  expected_camera: Mapping[str, float]) -> dict[str, Any]:
    run_dir = trial / "run"
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path, "launcher manifest")
    _check_manifest(
        manifest, spec=spec, config=config, stimulus_config=stimulus_config,
        scene_manifest=scene_manifest, trajectory=trajectory, movie=movie,
        trajectory_sha256=trajectory_sha256, movie_sha256=movie_sha256,
        expected_camera=expected_camera,
    )
    frame_count, telemetry_sha256 = _validate_telemetry(
        run_dir / "telemetry.jsonl", label=spec.label, duration_s=duration_s,
    )
    trajectory_path = run_dir / "trajectory.json"
    trajectory_data = _read_json(trajectory_path, "launcher trajectory")
    trajectory_metadata = trajectory_data.get("metadata") or {}
    if trajectory_metadata.get("telemetry_mode") != "compact":
        raise RuntimeError(f"{spec.label}: compact trajectory metadata is missing")
    scenario_records = trajectory_data.get("scenarios")
    if isinstance(scenario_records, Mapping):
        scenario_summary = scenario_records.get("courtship_visual") or {}
        recorded_frames = scenario_summary.get("frames")
        if isinstance(recorded_frames, list) and len(recorded_frames) != frame_count:
            raise RuntimeError(f"{spec.label}: trajectory frame count disagrees with telemetry")
    return {
        "label": spec.label,
        "seed": spec.seed,
        "condition": spec.condition,
        "trial_path": str(trial),
        "run_path": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "telemetry_path": str(run_dir / "telemetry.jsonl"),
        "telemetry_sha256": telemetry_sha256,
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "frame_count": frame_count,
        "duration_s": duration_s,
        "dt_s": FRAME_DT_S,
        "requested_backends": manifest.get("requested_backends"),
        "effective_backends": manifest.get("effective_backends"),
        "effective_execution": manifest.get("effective_execution"),
        "courtship_visual": manifest.get("courtship_visual"),
        "source_sha256": dict(source_hashes),
    }


def _run_analyzer_if_available(out: Path, run_summaries: Sequence[Mapping[str, Any]],
                               env: Mapping[str, str]) -> dict[str, Any]:
    command = [str(PYTHON), str(ANALYZER)]
    for trial in run_summaries:
        command += ["--trial", f"{trial['label']}={trial['condition']}={trial['run_path']}"]
    analysis_path = out / "courtship_visual_analysis.json"
    command += ["--output", str(analysis_path)]
    if not ANALYZER.is_file():
        return {
            "status": "not_available",
            "expected_command": command,
            "note": "raw trials are complete; run this analyzer command after its CLI is available",
        }
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=dict(env), check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed_nonblocking",
            "command": command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_path": str(analysis_path),
        }
    status = "completed" if completed.returncode == 0 and analysis_path.is_file() else "failed_nonblocking"
    result = {
        "status": status,
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_path": str(analysis_path),
    }
    if analysis_path.is_file():
        result["output_sha256"] = sha256_file(analysis_path)
    result["analyzer_sha256"] = sha256_file(ANALYZER)
    return result


def execute_suite(*, config_path: Path = DEFAULT_CONFIG,
                  stimulus_config_path: Path = DEFAULT_STIMULUS_CONFIG,
                  scene_manifest_path: Path | None = None,
                  trajectory_path: Path = DEFAULT_TRAJECTORY,
                  movie_path: Path | None = None,
                  seeds: Sequence[int] = DEFAULT_SEEDS,
                  conditions: Sequence[str] = TARGET_CONDITIONS,
                  duration_s: float = DURATION_S,
                  out: Path | None = None,
                  timeout_s: float = 960.0) -> dict[str, Any]:
    conditions = tuple(conditions)
    if duration_s != DURATION_S:
        raise ValueError("the locked courtship visual assay duration is 2.5 seconds")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence of distinct integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 0xFFFFFFFF
           for seed in seeds):
        raise ValueError("each seed must be an unsigned 32-bit integer")
    if timeout_s <= 0.0 or not math.isfinite(timeout_s):
        raise ValueError("timeout_s must be finite and positive")
    if movie_path is not None and scene_manifest_path is not None:
        raise ValueError("movie mode cannot include a scene manifest")
    if movie_path is None:
        if not conditions or len(set(conditions)) != len(conditions):
            raise ValueError("target conditions must be a nonempty sequence without duplicates")
        invalid = sorted(set(conditions) - set(TARGET_CONDITIONS))
        if invalid:
            raise ValueError(f"unknown courtship target conditions: {invalid}")
    else:
        if conditions not in (TARGET_CONDITIONS, (), None):
            raise ValueError("movie mode runs a single movie trial, not a target condition matrix")
        conditions = ()

    config_path = _resolve_file(config_path, "courtship experiment config")
    stimulus_config_path = _resolve_file(stimulus_config_path, "courtship stimulus config")
    if movie_path is None:
        scene_manifest_path = _resolve_file(
            scene_manifest_path or DEFAULT_SCENE_MANIFEST, "female scene manifest",
        )
        trajectory_path = _resolve_file(trajectory_path, "female target trajectory")
        geometry_path = _resolve_file(DEFAULT_GEOMETRY, "female target geometry config")
        sys.path.insert(0, str(ROOT / "scripts"))
        from courtship_target import FemaleTargetGeometry  # noqa: PLC0415
        geometry = FemaleTargetGeometry.from_json(geometry_path)
        _, scene_asset = _validate_scene_manifest(
            scene_manifest_path, expected_object_id=geometry.asset_object_id,
        )
        trajectory_sha256 = sha256_file(trajectory_path)
        movie = None
        movie_sha256 = None
        _, _, expected_camera = _validate_config(
            config_path, stimulus_config_path, trajectory_path, duration_s, geometry_path,
        )
    else:
        if scene_manifest_path is not None:
            raise ValueError("movie mode must not use --scene-manifest")
        movie = _resolve_file(movie_path, "courtship movie")
        trajectory_path = None
        trajectory_sha256 = None
        movie_sha256 = sha256_file(movie)
        _, _, expected_camera = _validate_config(config_path, stimulus_config_path, None, duration_s)
        scene_asset = None
        geometry_path = None

    source_files = {
        "config": config_path,
        "stimulus_config": stimulus_config_path,
    }
    if movie is None:
        source_files.update({
            "scene_manifest": scene_manifest_path,
            "trajectory": trajectory_path,
            "target_geometry": geometry_path,
        })
    else:
        source_files["movie"] = movie
    source_files = {key: path for key, path in source_files.items() if path is not None}
    input_hashes = {key: sha256_file(path) for key, path in source_files.items()}
    snapshot_source_files = {key: path for key, path in source_files.items() if key != "movie"}
    hash_only_files = {key: path for key, path in source_files.items() if key == "movie"}
    source_hashes = _source_hashes()
    out = (out or _default_output_path()).expanduser().resolve()
    if out.exists():
        raise ValueError(f"output directory already exists; refusing to overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)

    try:
        env = runtime_environment()
        preflight = gpu_and_memory_preflight(env)
        write_json(out / "preflight.json", preflight)
        input_provenance = _snapshot_inputs(out, snapshot_source_files)
        snapshot_hashes = {key: value["sha256"] for key, value in input_provenance.items()}
        if snapshot_hashes != {key: input_hashes[key] for key in snapshot_source_files}:
            raise RuntimeError("an input changed while the courtship suite snapshots were created")
        input_provenance.update(_record_hashed_inputs(hash_only_files))
        if scene_asset is not None:
            input_provenance["female_mesh_asset"] = scene_asset

        def assert_mesh_unchanged(label: str) -> None:
            if scene_asset is not None:
                asset_path = Path(scene_asset["asset_path"])
                if sha256_file(asset_path) != scene_asset["asset_sha256"]:
                    raise RuntimeError(f"{label}: female mesh asset changed after suite snapshot")

        trials = ((TrialSpec(int(seeds[0]), "movie", f"seed-{int(seeds[0])}-movie"),)
                  if movie is not None else make_trial_specs(seeds, conditions))
        run_summaries: list[dict[str, Any]] = []
        for spec in trials:
            _assert_unchanged(source_files, input_hashes, f"before {spec.label}")
            assert_mesh_unchanged(f"before {spec.label}")
            if _source_hashes() != source_hashes:
                raise RuntimeError(f"courtship source tree changed before trial {spec.label}")
            trial = run_trial(
                out=out, spec=spec, config=config_path,
                stimulus_config=stimulus_config_path,
                scene_manifest=scene_manifest_path, trajectory=trajectory_path,
                movie=movie, duration_s=duration_s, env=env, timeout_s=timeout_s,
            )
            if _source_hashes() != source_hashes:
                raise RuntimeError(f"courtship source tree changed during trial {spec.label}")
            _assert_unchanged(source_files, input_hashes, f"after {spec.label}")
            assert_mesh_unchanged(f"after {spec.label}")
            run_summaries.append(_record_trial(
                trial, spec=spec, config=config_path,
                stimulus_config=stimulus_config_path,
                scene_manifest=scene_manifest_path, trajectory=trajectory_path,
                movie=movie, duration_s=duration_s,
                trajectory_sha256=trajectory_sha256, movie_sha256=movie_sha256,
                source_hashes=source_hashes, expected_camera=expected_camera,
            ))
        analysis = (
            {"status": "not_applicable", "note": "open-loop movie probes are retained for dedicated analysis"}
            if movie is not None else _run_analyzer_if_available(out, run_summaries, env)
        )
        report = {
            "schema_version": "male-cns-courtship-visual-suite-1",
            "created_utc": now_utc(),
            "evidence_class": "digital_simulation",
            "experiment": (
                "strict_open_loop_courtship_visual_movie_probe" if movie is not None
                else "strict_courtship_visual_target_condition_matrix"
            ),
            "classification": (
                "courtship_open_loop_visual_probe" if movie is not None
                else "courtship_closed_loop_visual_assay"
            ),
            "biological_claim": False,
            "interpretation_boundary": (
                "This assay records visual orientation/tracking responses under the specified stimulus. "
                "It does not claim complete or autonomous courtship. Movie mode is open-loop and its "
                "visual input is independent of the simulated male pose."
            ),
            "execution": {
                "mode": "strict free-behavior",
                "neural_backend": "MaleCNS Triton CUDA",
                "vision_backend": "FlyVis CUDA",
                "renderer_backend": "Warp CUDA",
                "body_backend": "reduced 6-DoF flight dynamics",
                "frame_dt_s": FRAME_DT_S,
                "duration_s": duration_s,
                "seeds": list(dict.fromkeys(spec.seed for spec in trials)),
                "conditions": list(dict.fromkeys(spec.condition for spec in trials)),
                "condition_order": (
                    "serial; target conditions in CLI order within each seed: "
                    + ", ".join(dict.fromkeys(spec.condition for spec in trials))
                    if movie is None else "single serial open-loop movie trial"
                ),
                "host_memory_limit": "MemoryMax=5G per trial unit",
                "vram_limit_gib": MAX_VRAM_GIB,
                "fallback_policy": "fail closed; CPU, stub, and analytic renderer runs are rejected",
            },
            "inputs": {
                **input_provenance,
                "source_sha256": source_hashes,
            },
            "preflight": {
                "path": str(out / "preflight.json"),
                "sha256": sha256_file(out / "preflight.json"),
                "gate_passed": bool(preflight.get("gate_passed")),
                "memory_cap_check": preflight.get("resource_cap_preflight"),
            },
            "analysis": analysis,
            "run_summaries": run_summaries,
            "raw_trials_retained": True,
        }
        write_json(out / "suite_report.json", report)
        return report
    except Exception as exc:
        write_json(out / "failure.json", {
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "created_utc": now_utc(),
            "trial_directories_present": [
                str(path.relative_to(out)) for path in sorted((out / "trials").glob("seed-*/*"))
                if path.is_dir()
            ] if (out / "trials").exists() else [],
        })
        raise


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUT_ROOT / f"courtship-visual-{stamp}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stimulus-config", type=Path, default=DEFAULT_STIMULUS_CONFIG)
    parser.add_argument(
        "--scene-manifest", type=Path, default=None,
        help=f"female glTF scene manifest (default: {DEFAULT_SCENE_MANIFEST}; omitted for movie mode)",
    )
    parser.add_argument("--courtship-target-trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--courtship-movie", type=Path, default=None,
                        help="optional open-loop movie; runs exactly one trial and uses no scene manifest")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", choices=TARGET_CONDITIONS, nargs="+",
                        default=list(TARGET_CONDITIONS),
                        help="target matrix conditions; movie mode accepts only the default and runs one movie trial")
    parser.add_argument("--duration-s", type=float, choices=(2.5,), default=DURATION_S)
    parser.add_argument("--out", type=Path, default=_default_output_path(),
                        help="new suite directory under /data/artifacts/by-project/malecns-gpu/experiments")
    parser.add_argument("--timeout-s", type=float, default=960.0,
                        help="wall timeout per trial (systemd unit is capped at 900 s)")
    args = parser.parse_args(argv)
    try:
        if args.courtship_movie is not None and args.scene_manifest is not None:
            raise ValueError("--courtship-movie mode must not include --scene-manifest")
        report = execute_suite(
            config_path=args.config,
            stimulus_config_path=args.stimulus_config,
            scene_manifest_path=args.scene_manifest,
            trajectory_path=args.courtship_target_trajectory,
            movie_path=args.courtship_movie,
            seeds=tuple(args.seeds),
            conditions=tuple(args.conditions),
            duration_s=args.duration_s,
            out=args.out,
            timeout_s=args.timeout_s,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"courtship visual suite failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "suite_report": str(args.out.expanduser().resolve() / "suite_report.json"),
        "trial_count": len(report["run_summaries"]),
        "analysis_status": report["analysis"].get("status"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
