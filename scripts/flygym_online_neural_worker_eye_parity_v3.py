#!/usr/bin/env python3
"""Synchronous FlyGym pose -> fitted FlyVis -> MaleCNS -> leg-drive worker.

The process owns only the virtual visual scene and neural/motor-readout stages.
It never imports the digital-fly body, autopilot, state-drive, stabilizer, or
mission runner.  A separate FlyGym process supplies the already mapped
VirtualWorld-convention eye-midpoint pose at 200 Hz and receives one decoded
bilateral leg drive for the same frame.

Protocol: ``flygym-online-v1`` JSON Lines on stdin/stdout.  Diagnostics and
Python/backend output are redirected to stderr; stdout is reserved for
protocol records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "config" / "flygym_online_neural_worker.json"
ARTIFACT_ROOT = Path("/data/artifacts/by-project/malecns-gpu")
PROTOCOL = "flygym-online-v1"
FRAME_DT_S = 0.005
NEURAL_DT_S = 0.0005
NEURAL_STEPS_PER_FRAME = 10
NEURAL_HZ = 2000.0
FLYVIS_HZ = 200.0
MAX_HOST_MEMORY_BYTES = 7 * 1024**3
FROZEN_V2_PARENT_SHA256 = "23579a136547698b7eec9cd4a0ca2c3a8f8d5d0144601b08f67dcac383cc7ccb"
MAX_CAPTURE_FRAMES = 4000  # 20 s at 200 Hz; at most ~23 MiB of float32 eye arrays.
EYE_AZIMUTH_PARITY_POLICY = {"L": -1, "R": 1}

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class WorkerError(RuntimeError):
    """Invalid configuration, backend, or protocol input."""


def _emit(record: Mapping[str, Any], stream: Any) -> None:
    text = json.dumps(record, allow_nan=False, separators=(",", ":"))
    stream.write(text + "\n")
    stream.flush()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_file_record(path: Path, *, hash_limit_bytes: int = 1024 * 1024) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    size = path.stat().st_size
    result: dict[str, Any] = {"path": str(path), "exists": True, "size_bytes": size}
    if size <= hash_limit_bytes:
        result["sha256"] = _sha256_file(path)
    else:
        result["sha256"] = None
        result["hash_skipped_reason"] = "file exceeds bounded provenance hash limit"
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return _sha256_bytes(array.tobytes(order="C"))


def _host_memory_cgroup_limit_bytes() -> int | None:
    """Read the effective cgroup memory ceiling without changing it."""

    try:
        entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    v2_path: str | None = None
    v1_path: str | None = None
    for line in entries:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and not parts[1]:
            v2_path = parts[2]
        elif "memory" in parts[1].split(","):
            v1_path = parts[2]
    if v2_path is not None:
        root = Path("/sys/fs/cgroup")
        relative = v2_path.lstrip("/")
        current = root / relative
        name = "memory.max"
    elif v1_path is not None:
        root = Path("/sys/fs/cgroup/memory")
        relative = v1_path.lstrip("/")
        current = root / relative
        name = "memory.limit_in_bytes"
    else:
        return None
    candidates: list[int] = []
    try:
        while current == root or root in current.parents:
            file = current / name
            if file.is_file():
                raw = file.read_text(encoding="ascii").strip()
                if raw != "max":
                    value = int(raw)
                    if value > 0 and value < (1 << 60):
                        candidates.append(value)
            if current == root:
                break
            current = current.parent
    except (OSError, ValueError):
        return None
    return min(candidates) if candidates else None


def _finite_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkerError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise WorkerError(f"{label} must be finite")
    return out


def _vec(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise WorkerError(f"{label} must be a numeric vector of length {length}")
    try:
        values = tuple(_finite_float(item, f"{label}[{index}]")
                       for index, item in enumerate(value))
    except TypeError as exc:
        raise WorkerError(f"{label} must be a numeric vector of length {length}") from exc
    if len(values) != length:
        raise WorkerError(f"{label} must have length {length}; got {len(values)}")
    return values


def _quat_to_rpy_wxyz(quaternion: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-12:
        raise WorkerError("pose.quaternion_wxyz must have nonzero finite norm")
    if abs(norm - 1.0) > 1e-3:
        raise WorkerError(f"pose.quaternion_wxyz must be normalized (norm={norm:g})")
    w, x, y, z = (component / norm for component in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    # VirtualWorld's Pose pitch is defined by its camera basis as
    # Rz(yaw) * Ry(-pitch) * Rx(roll); it is the negative of the conventional
    # quaternion-to-Euler Y angle.
    pitch = -math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _timestamp(t_s: float) -> Any:
    from twin_contracts import Timestamp

    tick = int(round(t_s * NEURAL_HZ))
    if abs(t_s - tick / NEURAL_HZ) > 1e-8:
        raise WorkerError(f"time_s={t_s:g} is not aligned to the 2 kHz neural clock")
    return Timestamp(t_s, tick, "neural", NEURAL_HZ)


def _resolve_ref(config_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError(f"config.{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise WorkerError(f"config.{field} does not exist: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkerError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{label} root must be a JSON object")
    return value


def _load_angular_stimulus_spec(path: Path) -> tuple[Any, dict[str, Any], bytes]:
    """Load the frozen optional angular-field specification without defaults."""

    try:
        raw_bytes = path.read_bytes()
        document = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise WorkerError(f"could not read angular stimulus spec {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "angular-optomotor-stimulus-spec-1":
        raise WorkerError("angular stimulus spec must use schema_version angular-optomotor-stimulus-spec-1")
    if set(document) != {"schema_version", "display_geometry", "stimulus"}:
        raise WorkerError("angular stimulus spec has unexpected or missing top-level fields")
    if document.get("display_geometry") != "fixed_arena_infinite_distance_angular_field":
        raise WorkerError("angular stimulus spec has an unsupported display_geometry")
    stimulus_values = document.get("stimulus")
    if not isinstance(stimulus_values, dict):
        raise WorkerError("angular stimulus spec must contain a stimulus object")
    expected = {
        "name", "contrast", "head_direction_sign", "wavelength_deg",
        "temporal_frequency_hz", "mean_luminance", "azimuth_extent_deg",
        "elevation_extent_deg", "onset_s", "duration_s", "phase_at_onset_rad",
    }
    if set(stimulus_values) != expected:
        missing = sorted(expected - set(stimulus_values))
        extra = sorted(set(stimulus_values) - expected)
        raise WorkerError(f"angular stimulus fields mismatch; missing={missing}, extra={extra}")
    if (isinstance(stimulus_values.get("head_direction_sign"), bool)
            or not isinstance(stimulus_values.get("head_direction_sign"), int)):
        raise WorkerError("angular stimulus head_direction_sign must be integer +1 or -1")
    try:
        from standard_optomotor_stimulus import AngularOptomotorStimulus

        stimulus = AngularOptomotorStimulus(**stimulus_values)
    except (TypeError, ValueError) as exc:
        raise WorkerError(f"invalid angular stimulus parameters: {exc}") from exc
    for label, value in (
        ("onset_s", stimulus.onset_s),
        ("duration_s", stimulus.duration_s),
        ("end_s", stimulus.end_s),
    ):
        frames = value / FRAME_DT_S
        if not math.isclose(frames, round(frames), rel_tol=0.0, abs_tol=1e-8):
            raise WorkerError(f"angular stimulus {label} must align to the 5 ms visual frame")
    return stimulus, document, raw_bytes


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stimulus-condition", choices=("left", "right", "static", "angular"), required=True)
    parser.add_argument("--stimulus-onset-s", type=float, default=None)
    parser.add_argument(
        "--angular-stimulus-spec", type=Path, default=None,
        help="frozen angular-optomotor-stimulus-spec-1 JSON; requires --stimulus-condition angular",
    )
    parser.add_argument(
        "--stimulus-speed-mm-s", type=float, default=None,
        help="optional non-negative panel-speed magnitude in mm/s; preserves the configured left/right velocity direction",
    )
    parser.add_argument("--static-warmup-s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-frames", type=int, default=None,
                        help=f"full binocular-column capture limit (config default; safe cap {MAX_CAPTURE_FRAMES} frames/20 s)")
    parser.add_argument("--eye-separation-m", type=float, default=None,
                        help="measured left/right optical-origin baseline; defaults to worker config")
    parser.add_argument("--neural-device", choices=("cuda",), default="cuda")
    parser.add_argument("--neural-backend", choices=("triton",), default="triton")
    parser.add_argument("--vision-device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--max-vram-gib", type=float, default=8.0)
    parser.add_argument("--csr-path", type=Path, default=None)
    parser.add_argument(
        "--neural-intervention-config", type=Path, default=None,
        help="validated direct neural-input schedule; requires --allow-neural-intervention",
    )
    parser.add_argument(
        "--allow-neural-intervention", action="store_true",
        help="explicitly opt in to the supplied direct neural-input schedule",
    )
    parser.add_argument(
        "--deterministic-vision", action="store_true",
        help="request strict PyTorch determinism before FlyVis CUDA initialization",
    )
    parser.add_argument(
        "--deterministic-neural-delivery", action="store_true",
        help="request the engine's opt-in deterministic Triton delivery path",
    )
    parser.add_argument(
        "--debug-neural-state-frames", type=int, default=0,
        help="save initial and first N detailed neural-state captures (0 disables; maximum 64)",
    )
    parser.add_argument(
        "--capture-visual-drive", action="store_true",
        help="save exact fitted FlyVis T4/T5 drive values for every captured frame",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--standard-visual-input", type=Path)
    parser.add_argument("--allow-standard-visual-input", action="store_true")
    parser.add_argument("--capture-motor-channels", action="store_true")
    parser.add_argument("--capture-flyvis-state", action="store_true")
    return parser.parse_args()


class OnlineNeuralWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.seed < 0:
            raise WorkerError("--seed must be non-negative")
        if not math.isfinite(args.max_vram_gib) or not 0 < args.max_vram_gib <= 8.0:
            raise WorkerError("--max-vram-gib must be in (0, 8] for this bounded worker")
        self.host_memory_limit_bytes = _host_memory_cgroup_limit_bytes()
        if (self.host_memory_limit_bytes is None
                or self.host_memory_limit_bytes > MAX_HOST_MEMORY_BYTES):
            limit_text = "unknown" if self.host_memory_limit_bytes is None else f"{self.host_memory_limit_bytes} B"
            raise WorkerError(
                "worker must inherit the registered <=7 GiB host-memory cgroup cap "
                f"(observed effective limit: {limit_text})"
            )
        angular_spec_argument = getattr(args, "angular_stimulus_spec", None)
        if (args.stimulus_condition == "angular") != (angular_spec_argument is not None):
            raise WorkerError(
                "--angular-stimulus-spec is required exactly when --stimulus-condition angular is selected"
            )
        self.angular_stimulus = None
        self.angular_stimulus_spec_document = None
        self.angular_stimulus_spec_raw_bytes = None
        self.angular_stimulus_spec_path: Path | None = None
        self.angular_stimulus_spec_sha256: str | None = None
        if angular_spec_argument is not None:
            self.angular_stimulus_spec_path = angular_spec_argument.expanduser().resolve()
            (self.angular_stimulus, self.angular_stimulus_spec_document,
             self.angular_stimulus_spec_raw_bytes) = _load_angular_stimulus_spec(
                self.angular_stimulus_spec_path
            )
            if args.stimulus_onset_s is not None and not math.isclose(
                float(args.stimulus_onset_s), self.angular_stimulus.onset_s,
                rel_tol=0.0, abs_tol=1e-9,
            ):
                raise WorkerError("--stimulus-onset-s must match the angular stimulus spec timeline")
            args.stimulus_onset_s = self.angular_stimulus.onset_s
            self.angular_stimulus_spec_sha256 = _sha256_bytes(self.angular_stimulus_spec_raw_bytes)
        elif args.stimulus_onset_s is None:
            args.stimulus_onset_s = 1.0
        if not math.isfinite(args.stimulus_onset_s) or args.stimulus_onset_s < 0:
            raise WorkerError("--stimulus-onset-s must be finite and non-negative")
        if abs(args.stimulus_onset_s / FRAME_DT_S - round(args.stimulus_onset_s / FRAME_DT_S)) > 1e-8:
            raise WorkerError("--stimulus-onset-s must align to the 5 ms visual frame")
        if args.stimulus_speed_mm_s is not None:
            if args.stimulus_condition == "angular":
                raise WorkerError("--stimulus-speed-mm-s applies only to physical stripe conditions")
            speed_mm_s = _finite_float(args.stimulus_speed_mm_s, "stimulus_speed_mm_s")
            if speed_mm_s < 0.0:
                raise WorkerError("--stimulus-speed-mm-s must be non-negative")
            if args.stimulus_condition == "static" and speed_mm_s != 0.0:
                raise WorkerError("static stimulus speed override must be exactly 0 mm/s")

        self.config_path = args.config.expanduser().resolve()
        self.config = _load_json(self.config_path, "worker config")
        if self.config.get("schema_version") != "flygym-online-neural-worker-1":
            raise WorkerError("unsupported worker config schema_version")
        self._load_neural_intervention_schedule(args)
        configured_max_frames = self.config.get("max_frames", 400)
        max_frames_value = configured_max_frames if args.max_frames is None else args.max_frames
        if (isinstance(max_frames_value, bool) or not isinstance(max_frames_value, int)
                or not 1 <= max_frames_value <= MAX_CAPTURE_FRAMES):
            raise WorkerError(
                f"--max-frames/config max_frames must be in [1, {MAX_CAPTURE_FRAMES}] "
                "to keep the full binocular capture bounded"
            )
        self.world_config_path = _resolve_ref(self.config_path, self.config.get("world_config"), "world_config")
        self.stimulus_config_path = _resolve_ref(self.config_path, self.config.get("stimulus_config"), "stimulus_config")
        self.scene_manifest_path = _resolve_ref(self.config_path, self.config.get("scene_manifest"), "scene_manifest")

        geometry = self.config.get("eye_geometry", {})
        if not isinstance(geometry, Mapping):
            raise WorkerError("config.eye_geometry must be an object")
        configured_separation = geometry.get("eye_separation_m")
        separation = configured_separation if args.eye_separation_m is None else args.eye_separation_m
        if separation is None:
            raise WorkerError("eye separation must be specified in config or --eye-separation-m")
        self.eye_separation_m = _finite_float(separation, "eye_separation_m")
        if not 0.0 <= self.eye_separation_m <= 0.01:
            raise WorkerError("eye_separation_m must be in [0, 0.01] m")
        self.eye_geometry = dict(geometry)
        self.eye_geometry["eye_separation_m"] = self.eye_separation_m
        self.eye_geometry["cli_override"] = args.eye_separation_m is not None

        baseline_default = (
            self.angular_stimulus.onset_s if self.angular_stimulus is not None
            else self.config.get("static_warmup_s", 1.0)
        )
        self.static_warmup_s = _finite_float(
            baseline_default if args.static_warmup_s is None else args.static_warmup_s,
            "static_warmup_s",
        )
        if self.static_warmup_s <= 0 or abs(self.static_warmup_s / FRAME_DT_S - round(self.static_warmup_s / FRAME_DT_S)) > 1e-8:
            raise WorkerError("static warmup must be positive and aligned to the 5 ms visual frame")
        if self.angular_stimulus is not None:
            if not math.isclose(self.static_warmup_s, self.angular_stimulus.onset_s,
                                rel_tol=0.0, abs_tol=1e-9):
                raise WorkerError("angular-field static warmup must equal the spec's pre-stimulus gray duration")
            if max_frames_value * FRAME_DT_S + 1e-9 < self.angular_stimulus.end_s:
                raise WorkerError(
                    "worker frame limit ends before the complete angular moving interval; refusing a partial calibration"
                )

        self.seed = int(args.seed)
        self._configure_deterministic_vision(
            bool(getattr(args, "deterministic_vision", False)), self.seed,
        )
        self.deterministic_neural_delivery_requested = bool(
            getattr(args, "deterministic_neural_delivery", False)
        )
        self.condition = str(args.stimulus_condition)
        self.stimulus_onset_s = float(args.stimulus_onset_s)
        self.stimulus_speed_override_mm_s = (
            None if args.stimulus_speed_mm_s is None
            else float(args.stimulus_speed_mm_s)
        )
        self.max_vram_gib = float(args.max_vram_gib)
        self.max_frames = int(max_frames_value)
        debug_frames = getattr(args, "debug_neural_state_frames", 0)
        if (isinstance(debug_frames, bool) or not isinstance(debug_frames, int)
                or not 0 <= debug_frames <= 64):
            raise WorkerError("--debug-neural-state-frames must be an integer in [0, 64]")
        if debug_frames > self.max_frames:
            raise WorkerError("--debug-neural-state-frames cannot exceed the worker frame limit")
        self.debug_neural_state_frames = debug_frames
        self.capture_visual_drive = bool(getattr(args, "capture_visual_drive", False))
        native_path = getattr(args, "standard_visual_input", None)
        if bool(native_path) != bool(getattr(args, "allow_standard_visual_input", False)):
            raise WorkerError("standard visual input requires an explicit file and opt-in flag")
        self.native_input_path = native_path.expanduser().resolve() if native_path else None
        self.native_input = None
        self.native_input_metadata = None
        if self.angular_stimulus is not None and self.native_input_path is not None:
            raise WorkerError("angular stimulus specs cannot be combined with recorded native visual columns")
        if self.native_input_path is not None:
            with np.load(self.native_input_path, allow_pickle=False) as archive:
                self.native_input = {key: archive[key].copy() for key in
                                     ("left_columns", "right_columns", "time_s", "column_uv")}
                self.native_input_metadata = json.loads(str(archive["metadata_json"].item()))
            for eye in ("left_columns", "right_columns"):
                values = self.native_input[eye]
                if (values.shape != (self.max_frames, 721) or values.dtype != np.float32
                        or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1)):
                    raise WorkerError("standard visual columns must be finite float32 [frames,721] in [0,1]")
                values.setflags(write=False)
            if not np.allclose(self.native_input["time_s"], np.arange(self.max_frames)*FRAME_DT_S,
                               rtol=0, atol=1e-9):
                raise WorkerError("standard visual input must have the exact requested 5 ms clock")
            if (self.native_input_metadata.get("condition") != self.condition
                    or abs(float(self.native_input_metadata.get("onset_s", -1))-self.stimulus_onset_s) > 1e-9):
                raise WorkerError("standard visual condition/onset does not match the worker")
        self.capture_motor_channels = bool(getattr(args, "capture_motor_channels", False))
        self.capture_flyvis_state = bool(getattr(args, "capture_flyvis_state", False))
        self.vision_device = str(args.vision_device)
        self.neural_device = str(args.neural_device)
        self.neural_backend = str(args.neural_backend)
        self.out_dir = args.out.expanduser().resolve()
        if (not self.out_dir.is_absolute() or self.out_dir == ARTIFACT_ROOT
                or not self.out_dir.is_relative_to(ARTIFACT_ROOT)):
            raise WorkerError(f"--out must be a child path under {ARTIFACT_ROOT}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any(self.out_dir.iterdir()):
            raise WorkerError(f"--out must be a new or empty directory: {self.out_dir}")
        reserved = ("frames.jsonl", "provenance.json", "integrity.json", "failure.json",
                    "visual_drive.npz", "angular_stimulus_spec.json")
        conflicts = [name for name in reserved if (self.out_dir / name).exists()]
        if conflicts:
            raise WorkerError(f"output directory already contains reserved worker artifacts: {conflicts}")
        self.angular_stimulus_spec_copy_path: Path | None = None
        if self.angular_stimulus is not None:
            assert self.angular_stimulus_spec_path is not None
            assert self.angular_stimulus_spec_raw_bytes is not None
            if self.angular_stimulus_spec_path.read_bytes() != self.angular_stimulus_spec_raw_bytes:
                raise WorkerError("angular stimulus spec changed after validation")
            self.angular_stimulus_spec_copy_path = self.out_dir / "angular_stimulus_spec.json"
            self.angular_stimulus_spec_copy_path.write_bytes(self.angular_stimulus_spec_raw_bytes)
        self.frames_path = self.out_dir / "frames.jsonl"
        self.provenance_path = self.out_dir / "provenance.json"
        self.integrity_path = self.out_dir / "integrity.json"
        self.failure_path = self.out_dir / "failure.json"
        self.eye_columns_path = self.out_dir / "eye_columns.npz"
        self.visual_drive_path = self.out_dir / "visual_drive.npz"
        self.visual_drive_file_record: dict[str, Any] | None = None
        self.visual_drive_capture_count = 0
        self.visual_drive_target_spikes_captured = 0
        self.debug_neural_state_dir = self.out_dir / "debug-neural-state"
        self.debug_neural_state_initial_record: dict[str, Any] | None = None
        self.debug_neural_state_initial_metadata: dict[str, Any] | None = None
        self.debug_neural_state_frame_records: list[dict[str, Any]] = []

        from virtual_world import CameraConfig, PhysicalStimulus, VirtualWorld, load_stimuli

        base_world = _load_json(self.world_config_path, "world config")
        if not isinstance(base_world.get("world"), Mapping):
            raise WorkerError("world config lacks an object-valued 'world'")
        experiment_config = base_world.get("experiment", {})
        if not isinstance(experiment_config, Mapping):
            raise WorkerError("world config experiment must be an object when provided")
        configured_neural_gain = experiment_config.get("neural_gain")
        self.configured_neural_gain = (
            None if configured_neural_gain is None
            else _finite_float(configured_neural_gain, "experiment.neural_gain")
        )
        if self.configured_neural_gain is not None and self.configured_neural_gain <= 0.0:
            raise WorkerError("experiment.neural_gain must be positive")
        selected_stimuli: dict[str, Any]
        if self.angular_stimulus is not None:
            self.stimulus = None
            self.stimulus_velocity_virtualworld_mps = None
            self.stimulus_speed_effective_mm_s = None
            selected_stimuli = {}
        else:
            _load_json(self.stimulus_config_path, "stimulus config")
            names = self.config.get("condition_stimulus_names", {})
            if not isinstance(names, Mapping) or self.condition not in names:
                raise WorkerError(f"config has no visual stimulus mapping for {self.condition!r}")
            condition_name = str(names[self.condition])
            source_stimuli = load_stimuli(self.stimulus_config_path, seed=self.seed)
            if condition_name not in source_stimuli:
                raise WorkerError(f"stimulus config has no condition {condition_name!r}")
            source_stimulus = source_stimuli[condition_name]
            selected_payload = source_stimulus.to_dict()
            if self.stimulus_speed_override_mm_s is not None:
                configured_velocity = np.asarray(source_stimulus.velocity_mps, dtype=np.float64)
                configured_speed = float(np.linalg.norm(configured_velocity))
                requested_speed_mps = self.stimulus_speed_override_mm_s / 1000.0
                if requested_speed_mps == 0.0:
                    effective_velocity = np.zeros(3, dtype=np.float64)
                elif configured_speed <= 0.0:
                    raise WorkerError(
                        "a positive speed override requires a non-zero configured left/right direction"
                    )
                else:
                    effective_velocity = configured_velocity * (requested_speed_mps / configured_speed)
                selected_payload["velocity_mps"] = effective_velocity.tolist()
            # The panel exists at t=0. A separate scene-pattern clock holds it at
            # its initial pose until body time reaches the requested onset.
            selected_payload["start_time_s"] = 0.0
            selected_payload["seed"] = self.seed
            self.stimulus = PhysicalStimulus.from_mapping(selected_payload, seed=self.seed)
            self.stimulus_velocity_virtualworld_mps = [
                float(value) for value in self.stimulus.velocity_mps
            ]
            self.stimulus_speed_effective_mm_s = float(np.linalg.norm(
                np.asarray(self.stimulus.velocity_mps, dtype=np.float64)
            ) * 1000.0)
            selected_stimuli = {self.stimulus.id: self.stimulus}

        camera_cfg = self.config.get("camera", {})
        if not isinstance(camera_cfg, Mapping):
            raise WorkerError("config.camera must be an object")
        camera_kwargs = dict(camera_cfg)
        camera_kwargs["eye_separation_m"] = self.eye_separation_m
        try:
            camera = CameraConfig(**camera_kwargs)
        except (TypeError, ValueError) as exc:
            raise WorkerError(f"invalid camera config: {exc}") from exc
        sanitized_world = {
            "schema_version": "flygym-online-neural-scene-1",
            "seed": self.seed,
            "world": dict(base_world["world"]),
            # This sensor process has no scenario, goal, mission, or body loop.
            "scenarios": {},
        }
        if str(self.config.get("renderer_backend", "virtual_world_raycast")) != "virtual_world_raycast":
            raise WorkerError("this worker only accepts the explicit VirtualWorld reference renderer")
        self.world = VirtualWorld.from_mapping(
            sanitized_world, camera=camera, stimuli=selected_stimuli,
            scene_path=self.scene_manifest_path,
        )
        if getattr(self.world, "backend_name", None) != "virtual_world_raycast":
            raise WorkerError("VirtualWorld reference renderer did not initialize as requested")

        from flyvis_runtime_adapter_eye_parity_v2 import FlyVisRuntimeAdapter

        flyvis_cfg = self.config.get("flyvis", {})
        if not isinstance(flyvis_cfg, Mapping):
            raise WorkerError("config.flyvis must be an object")
        flyvis_kwargs = dict(flyvis_cfg)
        flyvis_kwargs.update({
            "device": self.vision_device,
            "max_vram_gib": self.max_vram_gib,
            "eye_azimuth_parity": dict(EYE_AZIMUTH_PARITY_POLICY),
        })
        self.flyvis = FlyVisRuntimeAdapter(**flyvis_kwargs)
        if self.flyvis.device != self.vision_device or self.flyvis.name != "flyvis":
            raise WorkerError(
                f"requested FlyVis {self.vision_device}, got {self.flyvis.name}/{self.flyvis.device}"
            )
        if self.flyvis.n_columns != 721:
            raise WorkerError(f"FlyVis must expose exactly 721 columns per eye, got {self.flyvis.n_columns}")
        direction_pair = self.flyvis.column_directions
        self.directions = {
            "left": np.ascontiguousarray(direction_pair["L"], dtype=np.float32),
            "right": np.ascontiguousarray(direction_pair["R"], dtype=np.float32),
        }
        for eye, direction in self.directions.items():
            if direction.shape != (721, 3) or not np.all(np.isfinite(direction)):
                raise WorkerError(f"FlyVis {eye} directions are not the exact finite (721,3) mapping")

        from neuromuscular_adapter import NeuromuscularAdapter

        map_path = Path(__file__).resolve().parents[1] / "config" / "neuromuscular_map.json"
        csr_path = args.csr_path.expanduser().resolve() if args.csr_path else self.flyvis.csr_path.resolve()
        if csr_path != self.flyvis.csr_path.resolve():
            raise WorkerError(
                "MaleCNS and FlyVis must use the same registered CSR annotations for target mapping"
            )
        self.neuromuscular = NeuromuscularAdapter(
            map_path=map_path, csr_path=csr_path, strict=True,
        )
        if "leg_power" not in self.neuromuscular.channels or "leg_steering" not in self.neuromuscular.channels:
            raise WorkerError("strict neuromuscular map lacks leg_power or leg_steering")
        report_populations: dict[str, str] = {}
        for channel in self.neuromuscular.channels.values():
            for term in channel.terms:
                report_populations[str(term.spec)] = str(term.spec)
        if not report_populations:
            raise WorkerError("strict neuromuscular map resolved no motor populations")
        raw_diagnostic_populations = self.config.get("diagnostic_populations")
        if not isinstance(raw_diagnostic_populations, Mapping) or not raw_diagnostic_populations:
            raise WorkerError("config.diagnostic_populations must be a non-empty name-to-selector object")
        self.diagnostic_population_specs: dict[str, str] = {}
        for raw_name, raw_spec in raw_diagnostic_populations.items():
            name = str(raw_name).strip()
            spec = str(raw_spec).strip() if isinstance(raw_spec, str) else ""
            if not name or not spec:
                raise WorkerError("diagnostic population names and selectors must be non-empty strings")
            if name in report_populations and report_populations[name] != spec:
                raise WorkerError(f"diagnostic population name conflicts with a motor report: {name!r}")
            self.diagnostic_population_specs[name] = spec
            report_populations[name] = spec

        import malecns_engine as malecns_engine_module
        from malecns_engine import create_engine

        engine_kwargs: dict[str, Any] = {
            "seed": self.seed,
            "device": self.neural_device,
            "backend": self.neural_backend,
            "csr_path": csr_path,
            "max_vram_gib": self.max_vram_gib,
            "dt_ms": NEURAL_DT_S * 1000.0,
            "flyvis_units": str(self.flyvis.units),
            "report_populations": report_populations,
            "include_flyvis_drive_echo": False,
            "deterministic_delivery": self.deterministic_neural_delivery_requested,
        }
        if self.configured_neural_gain is not None:
            engine_kwargs["gain"] = self.configured_neural_gain
        self.engine = create_engine(
            **engine_kwargs,
        )
        engine_delivery = self.engine.manifest().get("deterministic_delivery")
        if not isinstance(engine_delivery, Mapping):
            raise WorkerError("MaleCNS manifest omitted actual deterministic_delivery metadata")
        if (not isinstance(engine_delivery.get("requested"), bool)
                or engine_delivery["requested"] != self.deterministic_neural_delivery_requested
                or not isinstance(engine_delivery.get("enabled"), bool)
                or engine_delivery["enabled"] != self.deterministic_neural_delivery_requested):
            raise WorkerError("MaleCNS deterministic-delivery mode did not match the explicit worker request")
        self.deterministic_neural_delivery = _json_safe(dict(engine_delivery))
        if self.engine.device != self.neural_device or self.engine.backend != self.neural_backend:
            raise WorkerError(
                "MaleCNS backend/device selection did not match the explicit request: "
                f"requested {self.neural_backend}/{self.neural_device}, "
                f"got {self.engine.backend}/{self.engine.device}"
            )
        if not callable(getattr(self.engine, "step_window", None)):
            raise WorkerError("selected MaleCNS engine lacks the required step_window API")
        self._initialize_neural_intervention_membership()
        self._initialize_diagnostic_membership()
        self.neural_parameters = {
            "gain": float(self.engine.gain),
            "gain_configured_by_world_experiment": self.configured_neural_gain,
            "gain_source": (
                f"{self.world_config_path}#/experiment/neural_gain"
                if self.configured_neural_gain is not None
                else "MaleCNSEngine.DEFAULT_GAIN"
            ),
            "default_gain": float(malecns_engine_module.DEFAULT_GAIN),
            "kc_gain": float(self.engine.kc_gain),
            "default_kc_gain": float(malecns_engine_module.DEFAULT_KC_GAIN),
            "v_rest_mv": float(malecns_engine_module.V_REST_MV),
            "v_threshold_mv": float(malecns_engine_module.V_THR_MV),
            "v_reset_mv": float(malecns_engine_module.V_RESET_MV),
            "w_syn_mv": float(malecns_engine_module.W_SYN_MV),
            "tau_m_ms": float(malecns_engine_module.TAU_M_MS),
            "tau_s_ms": float(malecns_engine_module.TAU_S_MS),
            "delay_ms": float(malecns_engine_module.DELAY_MS),
            "refractory_ms": float(malecns_engine_module.REFRACTORY_MS),
        }
        self._initialize_debug_neural_state_capture()

        target_indices = np.ascontiguousarray(self.flyvis.target_indices, dtype=np.int64)
        if target_indices.ndim != 1 or target_indices.size != self.flyvis.n_t4t5:
            raise WorkerError("FlyVis target indices do not match the actual T4/T5 output width")
        target_indices.setflags(write=False)
        self.target_indices = target_indices
        self.target_indices_hash = _array_sha256(target_indices)
        if self.native_input is not None and not np.array_equal(
                self.native_input["column_uv"], self.flyvis.column_uv):
            raise WorkerError("standard visual columns disagree with actual Stimulus.input_index UV order")
        self.motor_channel_capture = None
        if self.capture_motor_channels:
            from motor_channel_capture import MotorChannelCapture
            self.motor_channel_capture = MotorChannelCapture(
                self.engine, self.max_frames, FRAME_DT_S, self.out_dir,
                ROOT / "config/leg_muscle_map.json",
            )
        self.flyvis_state_rows = (
            np.empty((self.max_frames, len(self.flyvis._mapping_flat_nodes)), dtype=np.float32)
            if self.capture_flyvis_state else None
        )
        self.flyvis_baseline_gather = None
        self.flyvis_state_count = 0
        self.last_seq: int | None = None
        self.last_time_s: float | None = None
        self.frame_count = 0
        self._captured_seqs = np.empty(self.max_frames, dtype=np.int64)
        self._captured_body_times = np.empty(self.max_frames, dtype=np.float64)
        self._captured_pattern_times = np.empty(self.max_frames, dtype=np.float64)
        self._captured_left_columns = np.empty((self.max_frames, 721), dtype=np.float32)
        self._captured_right_columns = np.empty((self.max_frames, 721), dtype=np.float32)
        self._captured_visual_drive_values = (
            np.empty((self.max_frames, self.flyvis.n_t4t5), dtype=np.float32)
            if self.capture_visual_drive else None
        )
        self._captured_visual_target_spike_counts = (
            np.empty((self.max_frames, self.flyvis.n_t4t5), dtype=np.uint16)
            if self.capture_visual_drive else None
        )
        self._captured_visual_drive_units: str | None = (
            str(self.flyvis.units) if self.capture_visual_drive else None
        )
        self.warmup_record: dict[str, Any] | None = None
        self.frames_handle = self.frames_path.open("x", encoding="utf-8", buffering=1)

        self.source_assets = self._source_assets(csr_path=csr_path, map_path=map_path)
        if self.angular_stimulus is not None:
            assert self.angular_stimulus_spec_path is not None
            assert self.angular_stimulus_spec_copy_path is not None
            captured_spec_record = _small_file_record(self.angular_stimulus_spec_copy_path)
            if captured_spec_record.get("sha256") != self.angular_stimulus_spec_sha256:
                raise WorkerError("captured angular stimulus spec copy failed its SHA-256 check")
            self.source_assets["angular_stimulus_spec"] = {
                "source": _small_file_record(self.angular_stimulus_spec_path),
                "captured_copy": captured_spec_record,
                "sha256": self.angular_stimulus_spec_sha256,
            }
        if self.native_input_path is not None:
            self.source_assets["standard_visual_input"] = _small_file_record(self.native_input_path)
        if self.capture_motor_channels:
            self.source_assets["motor_channel_mapping"] = _small_file_record(ROOT / "config/leg_muscle_map.json")
            self.source_assets["motor_channel_observer"] = _small_file_record(ROOT / "scripts/motor_channel_capture.py")
        self.effective_backends = {
            "renderer": (
                "fixed_arena_angular_field_cpu" if self.angular_stimulus is not None
                else (self.world.backend_name if self.native_input is None
                      else "flyvis_native_recorded_columns")
            ),
            "vision": self.flyvis.name,
            "vision_device": self.flyvis.device,
            "neural": self.engine.backend,
            "neural_device": self.engine.device,
            "neural_source": "connectome",
            "motor_decoder": "NeuromuscularAdapter",
        }
        self.provenance = self._build_provenance()
        self.provenance["standard_visual_input"] = self._standard_visual_manifest()
        if self.native_input is not None:
            self.provenance["renderer"].update({
                "backend": "flyvis_native_recorded_columns",
                "role": "standard visual calibration; physical scene raycasting bypassed",
            })
            self.provenance["causal_boundary"]["stimulus_condition_role"] = "selects standard recorded image sequence only"
            self.provenance["visual_capture"]["input_mode"] = "standard recorded columns; pose does not select the images"
        self.provenance["motor_channel_capture"] = self._motor_capture_manifest()
        self.provenance["flyvis_state_capture"] = {"enabled": self.capture_flyvis_state,
            "semantics": "native mapped gather state and static baseline; observer only"}
        self.provenance_path.write_text(
            json.dumps(self.provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _configure_deterministic_vision(self, requested: bool, seed: int) -> None:
        self.deterministic_vision_requested = bool(requested)
        self.deterministic_vision_settings_applied = False
        if not requested:
            return
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=False)
            cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn is not None:
                if hasattr(cudnn, "benchmark"): cudnn.benchmark = False
                if hasattr(cudnn, "deterministic"): cudnn.deterministic = True
        except Exception as exc:
            raise WorkerError(
                f"--deterministic-vision failed without fallback: {type(exc).__name__}: {exc}"
            ) from exc
        self.deterministic_vision_settings_applied = True

    def _deterministic_vision_manifest(self) -> dict[str, Any]:
        torch = sys.modules.get("torch")
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        query = getattr(torch, "are_deterministic_algorithms_enabled", None)
        return {
            "requested": self.deterministic_vision_requested,
            "effective_flags": {"settings_applied": self.deterministic_vision_settings_applied,
                "torch_deterministic_algorithms": bool(query()) if callable(query) else None,
                "cudnn_benchmark": getattr(cudnn, "benchmark", None),
                "cudnn_deterministic": getattr(cudnn, "deterministic", None),
                "cuda_seeded_by_worker": self.deterministic_vision_settings_applied},
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
            "custom_triton_repeatability_guaranteed": False,
            "scope_note": "Torch settings do not guarantee repeatability of custom Triton kernels",
        }

    @staticmethod
    def _debug_array_copy(value: Any) -> np.ndarray:
        if callable(getattr(value, "detach", None)):
            return value.detach().cpu().numpy().copy()
        return np.asarray(value).copy()

    def _debug_neural_state_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        sources = {
            "current_gpu": "_triton_current_t",
            "probability_gpu": "_triton_probability_t",
            "v": "v",
            "g": "g",
            "ref_until": "ref_until",
            "ring": "ring",
            "spk": "spk",
            "spike_tick": "spike_tick",
            "fixed_delivery_accumulator": "_triton_fixed_delivery_accumulator",
        }
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for key, attribute in sources.items():
            source = getattr(self.engine, attribute, None)
            if source is None and key == "fixed_delivery_accumulator":
                continue
            if source is None:
                raise WorkerError(f"debug neural-state capture lacks engine array {attribute}")
            array = self._debug_array_copy(source)
            arrays[key] = array
            metadata[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
            }
        return arrays, metadata

    def _debug_write_npz(
        self, path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = dict(metadata)
        payload["arrays"] = {
            name: {
                "shape": list(np.asarray(array).shape),
                "dtype": str(np.asarray(array).dtype),
                "sha256": _array_sha256(array),
            }
            for name, array in arrays.items()
        }
        np.savez_compressed(
            path, **arrays,
            metadata_json=np.asarray(json.dumps(payload, sort_keys=True, allow_nan=False)),
        )
        return {
            **payload,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }

    def _initialize_debug_neural_state_capture(self) -> None:
        if self.debug_neural_state_frames == 0:
            return
        self.debug_neural_state_dir.mkdir(parents=True, exist_ok=False)
        arrays, array_metadata = self._debug_neural_state_arrays()
        weight_host = np.asarray(getattr(self.engine, "_weight", None))
        if weight_host.ndim != 1 or weight_host.size == 0:
            raise WorkerError("debug initial capture lacks the engine's host synaptic weights")
        weight_gpu_source = getattr(self.engine, "_weight_t", None)
        if weight_gpu_source is None:
            raise WorkerError("debug initial capture lacks the engine's GPU synaptic weights")
        weight_gpu = self._debug_array_copy(weight_gpu_source)
        host_input_hashes = {"current": None, "probability": None}
        self.debug_neural_state_initial_metadata = {
            "schema_version": "flygym-online-neural-state-capture-1",
            "capture_kind": "initial_before_first_step_window",
            "frame_index": None,
            "seq": None,
            "time_s": None,
            "neural_window_start_step_index": None,
            "neural_window_end_step_index": None,
            "engine_next_step_index": int(self.engine._step_index),
            "host_held_input_sha256": host_input_hashes,
            "initial_state_is_frame_input": False,
            "pinned_input_staging": bool(getattr(self.engine, "_triton_pinned_input_staging", False)),
            "weights": {
                "host_shape": list(weight_host.shape),
                "host_dtype": str(weight_host.dtype),
                "host_sha256": _array_sha256(weight_host),
                "gpu_shape": list(weight_gpu.shape),
                "gpu_dtype": str(weight_gpu.dtype),
                "gpu_sha256": _array_sha256(weight_gpu),
                "host_gpu_equal": bool(np.array_equal(weight_host, weight_gpu)),
            },
            "extra_device_to_host_capture": True,
            "timing_equivalent_to_normal_run": False,
        }
        self.debug_neural_state_initial_record = self._debug_write_npz(
            self.debug_neural_state_dir / "initial.npz", arrays,
            {**self.debug_neural_state_initial_metadata, "arrays": array_metadata},
        )

    def _capture_debug_neural_state_frame(
        self, *, seq: int, time_s: float, named_outputs: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if self.frame_count >= self.debug_neural_state_frames:
            return None
        host_current = getattr(self.engine, "_triton_current_host", None)
        host_probability = getattr(self.engine, "_triton_probability_host", None)
        if host_current is None or host_probability is None:
            raise WorkerError("debug capture lacks the engine's held host current/probability arrays")
        host_current = np.asarray(host_current)
        host_probability = np.asarray(host_probability)
        arrays, array_metadata = self._debug_neural_state_arrays()
        current_gpu = arrays["current_gpu"]
        probability_gpu = arrays["probability_gpu"]
        last_step = int(named_outputs.get("step_index", -1))
        window_steps = int(named_outputs.get("neural_window_steps", -1))
        if last_step < 0 or window_steps < 1:
            raise WorkerError("debug capture lacks MaleCNS window step metadata")
        host_hashes = {
            "current": _array_sha256(host_current),
            "probability": _array_sha256(host_probability),
        }
        metadata = {
            "schema_version": "flygym-online-neural-state-capture-1",
            "capture_kind": "post_step_window_before_neuromuscular_decode",
            "frame_index": int(self.frame_count),
            "seq": int(seq),
            "time_s": float(time_s),
            "neural_window_start_step_index": last_step - window_steps + 1,
            "neural_window_end_step_index": last_step,
            "engine_next_step_index": int(self.engine._step_index),
            "host_held_input_sha256": host_hashes,
            "host_gpu_input_array_equal": {
                "current": bool(np.array_equal(host_current, current_gpu)),
                "probability": bool(np.array_equal(host_probability, probability_gpu)),
            },
            "host_gpu_input_shapes_and_dtypes_match": {
                "current": bool(host_current.shape == current_gpu.shape
                                and host_current.dtype == current_gpu.dtype),
                "probability": bool(host_probability.shape == probability_gpu.shape
                                    and host_probability.dtype == probability_gpu.dtype),
            },
            "pinned_input_staging": bool(getattr(self.engine, "_triton_pinned_input_staging", False)),
            "pinned_input_reason": getattr(self.engine, "_triton_pinned_input_reason", None),
            "extra_device_to_host_capture": True,
            "timing_equivalent_to_normal_run": False,
        }
        record = self._debug_write_npz(
            self.debug_neural_state_dir / f"frame-{self.frame_count:04d}.npz",
            arrays, {**metadata, "arrays": array_metadata},
        )
        self.debug_neural_state_frame_records.append(record)
        return record

    def _debug_neural_state_manifest(self) -> dict[str, Any]:
        enabled = self.debug_neural_state_frames > 0
        return {
            "enabled": enabled,
            "requested_frame_captures": int(self.debug_neural_state_frames),
            "maximum_frame_captures": 64,
            "artifact_dir": str(self.debug_neural_state_dir) if enabled else None,
            "initial_capture": self.debug_neural_state_initial_record,
            "captured_frames": list(self.debug_neural_state_frame_records),
            "pinned_input_staging": bool(getattr(self.engine, "_triton_pinned_input_staging", False)),
            "extra_device_to_host_capture": enabled,
            "timing_equivalent_to_normal_run": not enabled,
            "scope_note": (
                "debug tensor copies and NPZ compression perturb timing; exclude debug captures "
                "from runtime or performance comparisons"
            ) if enabled else "disabled; no debug-state copies are made",
        }

    def _load_neural_intervention_schedule(self, args: argparse.Namespace) -> None:
        schedule_path_arg = getattr(args, "neural_intervention_config", None)
        opted_in = bool(getattr(args, "allow_neural_intervention", False))
        if schedule_path_arg is None and opted_in:
            raise WorkerError("--allow-neural-intervention requires --neural-intervention-config")
        if schedule_path_arg is not None and not opted_in:
            raise WorkerError(
                "--neural-intervention-config requires explicit --allow-neural-intervention opt-in"
            )
        self.neural_intervention_schedule = None
        self.neural_intervention_config_path: Path | None = None
        self.neural_intervention_source_record: dict[str, Any] | None = None
        self.neural_intervention_selector_counts: dict[str, int] = {}
        self.neural_intervention_selector_membership: dict[str, dict[str, Any]] = {}
        if schedule_path_arg is None:
            return

        schedule_path = schedule_path_arg.expanduser().resolve()
        try:
            source_bytes = schedule_path.read_bytes()
            from neural_intervention import load_schedule

            schedule = load_schedule(schedule_path)
            if schedule_path.read_bytes() != source_bytes:
                raise WorkerError("neural intervention schedule changed while it was being loaded")
        except WorkerError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise WorkerError(f"could not load neural intervention schedule {schedule_path}: {exc}") from exc
        if not schedule.enabled:
            raise WorkerError("--neural-intervention-config must contain an enabled intervention schedule")
        if schedule.opt_in_required and not opted_in:
            raise WorkerError("neural intervention schedule requires explicit opt-in")
        self.neural_intervention_schedule = schedule
        self.neural_intervention_config_path = schedule_path
        self.neural_intervention_source_record = {
            "path": str(schedule_path),
            "source_size_bytes": len(source_bytes),
            "source_sha256": _sha256_bytes(source_bytes),
        }

    def _initialize_neural_intervention_membership(self) -> None:
        schedule = self.neural_intervention_schedule
        if schedule is None:
            return
        populations = getattr(self.engine, "populations", None)
        if populations is None or not callable(getattr(populations, "resolve", None)):
            raise WorkerError("neural-intervention audit requires MaleCNS PopulationIndex.resolve")
        for entry in schedule.entries:
            if entry.expected_match_count != 1:
                raise WorkerError(
                    f"neural-intervention selector {entry.selector!r} must declare "
                    "expected_match_count=1"
                )
            try:
                resolved = np.ascontiguousarray(
                    np.asarray(populations.resolve(entry.selector, strict=True), dtype=np.int64)
                )
            except Exception as exc:
                raise WorkerError(
                    f"neural-intervention selector {entry.selector!r} did not resolve: {exc}"
                ) from exc
            if resolved.ndim != 1 or resolved.size != 1:
                raise WorkerError(
                    f"neural-intervention selector {entry.selector!r} resolved "
                    f"{resolved.size} cells; expected exactly one"
                )
            self.neural_intervention_selector_counts[entry.selector] = 1
            ids = np.asarray(self.engine.graph.ids)
            self.neural_intervention_selector_membership[entry.selector] = {
                "count": 1,
                "csr_indices": [int(value) for value in resolved],
                "neuron_ids": [int(ids[index]) for index in resolved],
                "csr_indices_sha256": _array_sha256(resolved),
                "neuron_ids_sha256": _array_sha256(np.asarray(ids[resolved])),
            }

    def _neural_intervention_manifest(self) -> dict[str, Any]:
        schedule = self.neural_intervention_schedule
        if schedule is None:
            return {
                "enabled": False,
                "classification": "none",
                "schedule": None,
                "active_input_channels": {},
                "body_or_motor_command_injected": False,
                "autonomous_behavior_claim": False,
            }
        return {
            "enabled": True,
            "classification": schedule.classification,
            "intervention_type": schedule.intervention_type,
            "schedule_name": schedule.name,
            "opt_in_required": schedule.opt_in_required,
            "counts_as_autonomous_foraging": schedule.counts_as_autonomous_foraging,
            "prohibited_outputs": list(schedule.prohibited_outputs),
            "source_config": dict(self.neural_intervention_source_record or {}),
            "entries": [
                {
                    "selector": entry.selector,
                    "expected_match_count": entry.expected_match_count,
                    "units": entry.units,
                    "start_s": entry.start_s,
                    "end_s": entry.end_s,
                    "value": entry.value,
                }
                for entry in schedule.entries
            ],
            "resolved_selector_counts": dict(self.neural_intervention_selector_counts),
            "resolved_selector_membership": _json_safe(self.neural_intervention_selector_membership),
            "schedule_time_basis": "frame_time_s at the start of each 5 ms neural input window",
            "active_input_channels": {
                "poisson_hz": "SensoryFrame.channels.poisson_hz",
                "current_mV_per_ms": "SensoryFrame.channels.current_mV_per_ms",
            },
            "body_or_motor_command_injected": False,
            "autonomous_behavior_claim": False,
            "visual_following_claim": False,
            "optogenetic_equivalence_claim": False,
            "biological_calibration_claim": False,
            "interpretation": (
                "forced neural-input diagnostic; not autonomous behavior, visual following, "
                "calibrated optogenetics, or an optogenetic-equivalence claim"
            ),
        }

    def _frame_neural_intervention(self, time_s: float) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        schedule = self.neural_intervention_schedule
        if schedule is None:
            return {}, {}, {
                "enabled": False,
                "classification": "none",
                "active_poisson_hz": {},
                "active_current_mV_per_ms": {},
                "body_or_motor_command_injected": False,
                "autonomous_behavior_claim": False,
            }
        rates = schedule.rates_at(time_s)
        currents = schedule.currents_at(time_s)
        record = {
            "enabled": True,
            "classification": schedule.classification,
            "intervention_type": schedule.intervention_type,
            "schedule_name": schedule.name,
            "time_s": float(time_s),
            "source_config": dict(self.neural_intervention_source_record or {}),
            "active_poisson_hz": dict(rates),
            "active_current_mV_per_ms": dict(currents),
            "resolved_selector_counts": dict(self.neural_intervention_selector_counts),
            "counts_as_autonomous_foraging": False,
            "body_or_motor_command_injected": False,
            "autonomous_behavior_claim": False,
            "visual_following_claim": False,
            "optogenetic_equivalence_claim": False,
            "biological_calibration_claim": False,
            "interpretation": (
                "forced neural-input diagnostic; not autonomous behavior, visual following, "
                "calibrated optogenetics, or an optogenetic-equivalence claim"
            ),
        }
        return rates, currents, record

    def _initialize_diagnostic_membership(self) -> None:
        """Freeze the exact selector membership used by frame diagnostics."""

        graph = self.engine.graph
        ids = np.asarray(graph.ids)
        tables = graph.tables
        annotations = graph.annotations
        type_labels = np.asarray(tables["types"]).astype(str)
        side_labels = np.asarray(tables["sides"]).astype(str)
        nt_labels = np.asarray(tables["nts"]).astype(str)
        type_idx = np.asarray(annotations["typeIdx"])
        side_idx = np.asarray(annotations["sideIdx"])
        nt_idx = np.asarray(annotations["ntIdx"])

        membership: dict[str, dict[str, Any]] = {}
        selected_arrays: list[np.ndarray] = []
        for name, spec in self.diagnostic_population_specs.items():
            if name not in self.engine.report_populations:
                raise WorkerError(f"MaleCNS did not resolve diagnostic population {name!r}")
            selected = np.ascontiguousarray(
                np.asarray(self.engine.report_populations[name], dtype=np.int64)
            )
            if selected.ndim != 1 or selected.size == 0:
                raise WorkerError(f"diagnostic population {name!r} resolved to no CSR neurons")
            if (np.any(selected < 0) or np.any(selected >= self.engine.n)
                    or np.unique(selected).size != selected.size):
                raise WorkerError(f"diagnostic population {name!r} has invalid CSR membership")
            selected_arrays.append(selected)
            members = []
            for index in selected.tolist():
                members.append({
                    "csr_index": int(index),
                    "neuron_id": int(ids[index]),
                    "type_annotation": str(type_labels[type_idx[index]]),
                    "side_annotation": str(side_labels[side_idx[index]]),
                    "nt_annotation": str(nt_labels[nt_idx[index]]),
                })
            membership[name] = {
                "selector": spec,
                "count": int(selected.size),
                "csr_indices": [int(value) for value in selected],
                "neuron_ids": [int(ids[index]) for index in selected],
                "side_annotations": [str(side_labels[side_idx[index]]) for index in selected],
                "members": members,
                "csr_indices_sha256": _array_sha256(selected),
                "neuron_ids_sha256": _array_sha256(np.asarray(ids[selected])),
                "side_annotation_semantics": (
                    "verbatim registered CSR side annotation; no anatomical laterality is inferred"
                ),
            }

        union = np.unique(np.concatenate(selected_arrays)).astype(np.int64, copy=False)
        if union.size == 0:
            raise WorkerError("diagnostic selectors resolved to an empty CSR membership union")
        self.diagnostic_population_membership = membership
        self.diagnostic_neuron_indices = np.ascontiguousarray(union)
        self.diagnostic_neuron_ids = np.ascontiguousarray(ids[union])
        self.diagnostic_neuron_side_annotations = {
            str(int(index)): str(side_labels[side_idx[index]]) for index in union
        }

        # MaleCNSEngine.v is its live per-neuron membrane-voltage state for
        # both the NumPy and Torch/Triton backends. The index tensor is created
        # once; frame sampling uses a read-only gather and never writes v.
        state = getattr(self.engine, "v", None)
        if tuple(getattr(state, "shape", ())) != (int(self.engine.n),):
            raise WorkerError("MaleCNS live membrane-voltage state is unavailable or has the wrong shape")
        self._diagnostic_device_indices = None
        if callable(getattr(state, "index_select", None)):
            torch = getattr(self.engine, "_torch", None)
            if torch is None:
                raise WorkerError("MaleCNS tensor voltage state lacks its registered Torch backend")
            try:
                self._diagnostic_device_indices = torch.as_tensor(
                    self.diagnostic_neuron_indices,
                    dtype=torch.int64,
                    device=state.device,
                )
            except Exception as exc:
                raise WorkerError(f"could not prepare read-only diagnostic voltage gather: {exc}") from exc

    def _diagnostic_identity_fields(self) -> dict[str, Any]:
        return {
            "diagnostic_population_specs": dict(self.diagnostic_population_specs),
            "diagnostic_population_membership": _json_safe(self.diagnostic_population_membership),
            "diagnostic_neuron_indices": [int(value) for value in self.diagnostic_neuron_indices],
            "diagnostic_neuron_ids": [int(value) for value in self.diagnostic_neuron_ids],
            "diagnostic_neuron_side_annotations": dict(self.diagnostic_neuron_side_annotations),
            "diagnostic_side_annotation_semantics": (
                "stored CSR annotation selected by the exact /L or /R selector; "
                "not an inferred anatomical laterality claim"
            ),
        }

    def _sample_diagnostic_membrane_vm(self) -> tuple[dict[str, float], dict[str, float], float]:
        """Read each selected live Vm once, after the final tick of this window."""

        state = self.engine.v
        if self._diagnostic_device_indices is not None:
            values = state.index_select(0, self._diagnostic_device_indices).detach().cpu().numpy()
        else:
            values = np.asarray(state[self.diagnostic_neuron_indices])
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.diagnostic_neuron_indices.shape or not np.all(np.isfinite(values)):
            raise WorkerError("MaleCNS diagnostic Vm gather returned invalid values")
        by_index = {
            str(int(index)): _finite_float(value, f"diagnostic Vm at CSR index {int(index)}")
            for index, value in zip(self.diagnostic_neuron_indices, values)
        }
        population_means: dict[str, float] = {}
        for name, record in self.diagnostic_population_membership.items():
            members = [by_index[str(index)] for index in record["csr_indices"]]
            if not members:
                raise WorkerError(f"diagnostic population {name!r} has no Vm samples")
            population_means[name] = float(np.mean(members, dtype=np.float64))
        return by_index, population_means, float(self.engine._last_timestamp.t_s)

    def _source_assets(self, *, csr_path: Path, map_path: Path) -> dict[str, Any]:
        runtime = self.flyvis.manifest()
        view_path = Path(runtime["model_root"]) / "results" / Path(str(runtime["view"]))
        model_files = {}
        for name in ("best_chkpt", "_meta.yaml"):
            path = view_path / name
            if path.is_file():
                model_files[name] = _small_file_record(path)

        scene = _load_json(self.scene_manifest_path, "scene manifest")
        asset_root_value = scene.get("asset_root")
        asset_root = Path(str(asset_root_value)).expanduser()
        if not asset_root.is_absolute():
            asset_root = (self.scene_manifest_path.parent / asset_root).resolve()
        panel_records: dict[str, Any] = {}
        for asset in scene.get("assets", []):
            if not isinstance(asset, Mapping) or not isinstance(asset.get("path"), str):
                continue
            path = (asset_root / str(asset["path"])).resolve()
            record = _small_file_record(path)
            integrity = asset.get("integrity", {})
            declared = integrity.get("sha256") if isinstance(integrity, Mapping) else None
            record["declared_sha256"] = declared
            if declared and record.get("sha256") and declared != record["sha256"]:
                raise WorkerError(f"scene asset integrity mismatch: {path}")
            panel_records[str(asset.get("object_id", path.name))] = record

        try:
            csr_record = self.engine.manifest()["csr"]
        except Exception as exc:
            raise WorkerError(f"could not describe loaded MaleCNS CSR assets: {exc}") from exc
        csr_files = {}
        for name, exists in csr_record.get("files", {}).items():
            if exists:
                csr_files[name] = _small_file_record(csr_path / str(name))
        target_path = Path(runtime["target_path"])
        return {
            "worker_config": _small_file_record(self.config_path),
            "world_config": _small_file_record(self.world_config_path),
            "stimulus_config": _small_file_record(self.stimulus_config_path),
            "scene_manifest": _small_file_record(self.scene_manifest_path),
            "visual_scene_assets": panel_records,
            "flyvis": {
                "environment_path": self.flyvis.preflight.get("environment"),
                "model_root": runtime.get("model_root"),
                "view": runtime.get("view"),
                "view_path": str(view_path),
                "model_files": model_files,
                "target_mapping": _small_file_record(target_path),
                "adapter_source": _small_file_record(SCRIPTS / "flyvis_runtime_adapter_eye_parity_v2.py"),
                "legacy_adapter_source": _small_file_record(SCRIPTS / "flyvis_runtime_adapter.py"),
                "eye_coordinate_policy": runtime.get("eye_coordinate_policy"),
                "n_columns_per_eye": runtime.get("n_columns_per_eye"),
                "n_targets": runtime.get("n_targets"),
                "direction_hashes": {
                    eye: runtime.get("column_directions", {}).get(eye, {}).get("sha256")
                    for eye in ("L", "R")
                },
            },
            "worker_source": {
                "candidate": _small_file_record(Path(__file__).resolve()),
                "legacy": _small_file_record(SCRIPTS / "flygym_online_neural_worker.py"),
                "parent_v2": {
                    **_small_file_record(SCRIPTS / "flygym_online_neural_worker_eye_parity_v2.py"),
                    "frozen_sha256": FROZEN_V2_PARENT_SHA256,
                    "resource_only_changes_in_v3": True,
                },
                "eye_coordinate_policy": dict(EYE_AZIMUTH_PARITY_POLICY),
            },
            "malecns": {
                "csr_path": str(csr_path),
                "csr_preflight": csr_record,
                "csr_files": csr_files,
                "neuromuscular_map": _small_file_record(map_path),
                "sensory_map": _small_file_record(Path(self.engine.sensory_map_path)),
                "flyvis_target_indices_sha256": self.target_indices_hash,
            },
        }

    def _angular_stimulus_manifest(self) -> dict[str, Any] | None:
        if self.angular_stimulus is None:
            return None
        return {
            "spec_schema_version": "angular-optomotor-stimulus-spec-1",
            "spec_path": str(self.angular_stimulus_spec_path),
            "captured_spec_path": str(self.angular_stimulus_spec_copy_path),
            "spec_sha256": self.angular_stimulus_spec_sha256,
            "name": self.angular_stimulus.name,
            "contrast_study_parameter_c": self.angular_stimulus.contrast,
            "head_direction_sign": self.angular_stimulus.head_direction_sign,
            "head_direction_frame": "initial FlyGym head azimuth atan2(head_y, head_x); +head_Y is fly-left",
            "head_angular_speed_deg_s": self.angular_stimulus.angular_speed_deg_s,
            "arena_angular_speed_deg_s": self.angular_stimulus.arena_angular_speed_deg_s,
            "angular_speed_units": "degrees per second",
            "wavelength_deg": self.angular_stimulus.wavelength_deg,
            "temporal_frequency_hz": self.angular_stimulus.temporal_frequency_hz,
            "onset_s": self.angular_stimulus.onset_s,
            "duration_s": self.angular_stimulus.duration_s,
            "end_exclusive_s": self.angular_stimulus.end_s,
            "display_geometry": "fixed arena angular field; infinite-distance angular-screen approximation",
            "translation_parallax": "none; head position does not change arena angles",
            "head_ray_to_arena_transform": (
                "VirtualWorld._camera_basis(current_pose): forward*head_x - right*head_y + up*head_z; "
                "sample remains fixed in arena coordinates as body yaw changes"
            ),
            "stimulus_description": self.angular_stimulus.to_dict(),
            "motor_authority": "none; angular stimulus selects pixels only",
        }

    def _stimulus_manifest(self) -> dict[str, Any]:
        if self.angular_stimulus is not None:
            return self.angular_stimulus.to_dict()
        assert self.stimulus is not None
        return self.stimulus.to_dict()

    def _build_provenance(self) -> dict[str, Any]:
        angular = self.angular_stimulus is not None
        stimulus_clock = (
            {
                "body_and_neural_clock": "frame_time_s",
                "pattern_clock": "absolute frame_time_s; spec onset/duration gate grating, not body time",
                "pre_stimulus_interval_s": [0.0, self.angular_stimulus.onset_s],
                "moving_grating_interval_s": [self.angular_stimulus.onset_s,
                                               self.angular_stimulus.end_s],
                "post_stimulus": "uniform mean gray if later requested by total duration",
                "static_visual_warmup_s": self.static_warmup_s,
                "angular_spec_owns_timeline": True,
            }
            if angular else {
                "body_and_neural_clock": "frame_time_s",
                "pattern_clock": "max(0, frame_time_s - stimulus_onset_s)",
                "physical_stimulus_start_time_s": float(self.stimulus.start_time_s),
                "panel_visible_during_static_warmup": True,
                "panel_motion_begins_at_body_time_s": self.stimulus_onset_s,
            }
        )
        return {
            "schema_version": "flygym-online-neural-provenance-1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "seed": self.seed,
            "condition": self.condition,
            "stimulus_mode": "fixed_arena_angular_field" if angular else "physical_world_panel",
            "stimulus": self._stimulus_manifest(),
            "angular_stimulus_spec": self._angular_stimulus_manifest(),
            "stimulus_speed_override_mm_s": self.stimulus_speed_override_mm_s,
            "stimulus_speed_effective_mm_s": self.stimulus_speed_effective_mm_s,
            "stimulus_velocity_frame": (
                None if angular else "VirtualWorld scene-coordinate frame"
            ),
            "stimulus_velocity_virtualworld_mps": self.stimulus_velocity_virtualworld_mps,
            "stimulus_onset_s": self.stimulus_onset_s,
            "stimulus_clock": stimulus_clock,
            "static_visual_warmup": {
                "duration_s": self.static_warmup_s,
                "male_cns_invoked": False,
                "neuromuscular_decoder_invoked": False,
                "body_stepped_or_forced_to_settle": False,
                "method": "FlyVisRuntimeAdapter.preadapt_static_scene at first received body pose",
                "visible_scene_render_time_s": 0.0,
                "visible_scene_pattern_time_s": 0.0,
                "flyvis_recurrent_state_effect": (
                    "advance the fitted FlyVis state over the fixed scene frames, retain the final state and "
                    "baseline activity, then reset only its exposed frame index to zero"
                ),
                "male_cns_initial_step_index": int(self.engine._step_index),
                "male_cns_steps_during_warmup": 0,
            },
            "neural_intervention": self._neural_intervention_manifest(),
            "deterministic_vision": self._deterministic_vision_manifest(),
            "deterministic_neural_delivery": self.deterministic_neural_delivery,
            "debug_neural_state": self._debug_neural_state_manifest(),
            "visual_drive_capture": self._visual_drive_capture_manifest(),
            "standard_visual_input": self._standard_visual_manifest(),
            "motor_channel_capture": self._motor_capture_manifest(),
            "flyvis_state_capture": {"enabled": self.capture_flyvis_state},
            "pose_and_eye_geometry": {
                "input_pose_frame": "VirtualWorld convention; transformed by FlyGym body bridge",
                "position_semantics": "eye midpoint; worker never applies a head offset",
                "eye_geometry": self.eye_geometry,
                "ray_directions": "exact fitted FlyVis column_directions, in returned input_index order",
                "native_eye_camera_euler_extrinsics_applied_separately": False,
                "limitation": (
                    "ray orientation uses the fitted FlyVis L/R head-frame direction grid; "
                    "the native FlyGym camera Euler extrinsics are not applied a second time"
                ),
            },
            "effective_backends": self.effective_backends,
            "source_code": {
                "worker_script": _small_file_record(Path(__file__).resolve()),
                "angular_stimulus_helper": (
                    _small_file_record(SCRIPTS / "standard_optomotor_stimulus.py")
                    if angular else None
                ),
            },
            "resource_caps": {
                "max_vram_gib_per_backend": self.max_vram_gib,
                "host_memory_limit_bytes": self.host_memory_limit_bytes,
                "host_memory_policy": "verified inherited cgroup limit, at most 7 GiB",
            },
            "causal_boundary": {
                "stimulus_condition_role": "selects only the visual scene at worker startup",
                "motor_source": "NeuromuscularAdapter leg_power/leg_steering only",
                "left_drive_equation": "clip(leg_power - 0.5 * leg_steering, 0, 1)",
                "right_drive_equation": "clip(leg_power + 0.5 * leg_steering, 0, 1)",
                "drive_gain_or_offset": "none",
                "future_interpolation_or_stale_outputs": False,
                "autopilot_state_drive_stabilizer_mission": "not imported or invoked",
                "mechanosensory_feedback": {
                    "status": "unsupported_and_not_injected",
                    "reason": "FlyGym online pose protocol supplies no mapped contact/joint afferent channels",
                },
            },
            "renderer": {
                "backend": self.effective_backends["renderer"],
                "role": (
                    "CPU sampler for frozen fixed-arena angular stimulus; VirtualWorld raycasting is bypassed"
                    if angular else "explicit analytic reference renderer; not a FlyVis stub"
                ),
                "columns_per_eye": 721,
                "direction_hashes": {
                    eye: _array_sha256(self.directions[eye]) for eye in ("left", "right")
                },
            },
            "visual_capture": {
                "max_frames": self.max_frames,
                "array_schema": "seq,frame_time_s,pattern_time_s,left_columns,right_columns",
                "dtype": "float64 timestamps; float32 luminance columns in [0,1]",
                "expected_shape_per_eye": [self.max_frames, 721],
                "path": str(self.eye_columns_path),
                "sha256": None,
            },
            "flyvis": _json_safe(self.flyvis.manifest()),
            "malecns": {
                "backend": self.engine.backend,
                "device": self.engine.device,
                "seed": self.engine.seed,
                "dt_ms": self.engine.dt_ms,
                "model_parameters": self.neural_parameters,
                "max_vram_gib": self.engine.max_vram_gib,
                "n_neurons": self.engine.n,
                "n_edges": self.engine.e,
                "report_population_sizes": {
                    name: int(ids.size) for name, ids in self.engine.report_populations.items()
                },
            },
            **self._diagnostic_identity_fields(),
            "neuromuscular": {
                "map_path": str(self.neuromuscular.map_path),
                "map_sha256": self.neuromuscular.map_sha256,
                "calibration_id": self.neuromuscular.calibration_id,
                "channels": {
                    name: {
                        "units": channel.units,
                        "tau_ms": channel.tau_ms,
                        "terms": [term.spec for term in channel.terms],
                    }
                    for name, channel in self.neuromuscular.channels.items()
                },
            },
            "source_assets": self.source_assets,
            "files": {
                "frames_jsonl": str(self.frames_path),
                "eye_columns_npz": str(self.eye_columns_path),
                "provenance_json": str(self.provenance_path),
                "integrity_json": str(self.integrity_path),
                "angular_stimulus_spec_json": (
                    str(self.angular_stimulus_spec_copy_path)
                    if self.angular_stimulus_spec_copy_path is not None else None
                ),
            },
        }

    def ready_record(self) -> dict[str, Any]:
        angular = self.angular_stimulus is not None
        return {
            "protocol": PROTOCOL,
            "type": "ready",
            "effective_backends": self.effective_backends,
            "source": "connectome",
            "source_assets": self.source_assets,
            "seed": self.seed,
            "stimulus_condition": self.condition,
            "stimulus_mode": "fixed_arena_angular_field" if angular else "physical_world_panel",
            "angular_stimulus": self._angular_stimulus_manifest(),
            "angular_stimulus_spec_sha256": self.angular_stimulus_spec_sha256,
            "stimulus_speed_override_mm_s": self.stimulus_speed_override_mm_s,
            "stimulus_speed_effective_mm_s": self.stimulus_speed_effective_mm_s,
            "stimulus_velocity_frame": None if angular else "VirtualWorld scene-coordinate frame",
            "stimulus_velocity_virtualworld_mps": self.stimulus_velocity_virtualworld_mps,
            "stimulus_onset_s": self.stimulus_onset_s,
            "stimulus_clock": (
                "pattern_time_s=frame_time_s; angular spec controls gray/grating intervals"
                if angular else
                "pattern_time_s=max(0,time_s-stimulus_onset_s); neural/body time_s is unchanged"
            ),
            "static_warmup_s": self.static_warmup_s,
            "eye_separation_m": self.eye_separation_m,
            "eye_geometry_source": self.eye_geometry.get("source"),
            "n_columns_per_eye": int(self.flyvis.n_columns),
            "n_t4t5_targets": int(self.flyvis.n_t4t5),
            "flyvis_target_units": self.flyvis.units,
            "flyvis_target_indices_sha256": self.target_indices_hash,
            "eye_column_hashes": {
                eye: _array_sha256(self.directions[eye]) for eye in ("left", "right")
            },
            "frame_dt_s": FRAME_DT_S,
            "neural_tick_dt_s": NEURAL_DT_S,
            "neural_steps_per_frame": NEURAL_STEPS_PER_FRAME,
            "malecns_model_parameters": self.neural_parameters,
            "max_frames": self.max_frames,
            "eye_columns_artifact": str(self.eye_columns_path),
            "male_cns_initial_step_index": int(self.engine._step_index),
            "male_cns_steps_during_visual_warmup": 0,
            "flyvis_exposed_frame_index_before_first_live_frame": 0,
            **self._diagnostic_identity_fields(),
            "neural_intervention": self._neural_intervention_manifest(),
            "deterministic_vision": self._deterministic_vision_manifest(),
            "deterministic_neural_delivery": self.deterministic_neural_delivery,
            "debug_neural_state": self._debug_neural_state_manifest(),
            "visual_drive_capture": self._visual_drive_capture_manifest(),
            "standard_visual_input": self._standard_visual_manifest(),
            "motor_channel_capture": self._motor_capture_manifest(),
            "flyvis_state_capture": {"enabled": self.capture_flyvis_state},
            "motor_source": "decoded_leg_power_leg_steering_mix",
            "mechanosensory_feedback": "unsupported_and_not_injected",
            "artifact_dir": str(self.out_dir),
            "host_memory_limit_bytes": self.host_memory_limit_bytes,
        }

    def _validate_frame(self, request: Any) -> tuple[int, float, Any, tuple[float, ...]]:
        if not isinstance(request, Mapping):
            raise WorkerError("frame record must be a JSON object")
        if request.get("protocol") != PROTOCOL or request.get("type") != "frame":
            raise WorkerError("expected flygym-online-v1 frame record")
        raw_seq = request.get("seq")
        if isinstance(raw_seq, bool) or not isinstance(raw_seq, int) or raw_seq < 0:
            raise WorkerError("frame.seq must be a non-negative integer")
        seq = int(raw_seq)
        if self.last_seq is not None and seq != self.last_seq + 1:
            raise WorkerError(f"frame.seq must be exactly {self.last_seq + 1}; got {seq}")
        t_s = _finite_float(request.get("time_s"), "frame.time_s")
        if t_s < 0:
            raise WorkerError("frame.time_s must be non-negative")
        dt_s = _finite_float(request.get("dt_s"), "frame.dt_s")
        if abs(dt_s - FRAME_DT_S) > 1e-9:
            raise WorkerError(f"frame.dt_s must equal {FRAME_DT_S:g}; got {dt_s:g}")
        if abs(t_s / FRAME_DT_S - round(t_s / FRAME_DT_S)) > 1e-8:
            raise WorkerError("frame.time_s must align to the 5 ms visual clock")
        if self.last_time_s is None:
            if abs(t_s) > 1e-9:
                raise WorkerError("first frame must start at time_s=0")
        elif abs(t_s - (self.last_time_s + FRAME_DT_S)) > 1e-8:
            raise WorkerError(
                f"frame.time_s must be the next 5 ms sample after {self.last_time_s:g}; got {t_s:g}"
            )
        pose_raw = request.get("pose")
        if not isinstance(pose_raw, Mapping):
            raise WorkerError("frame.pose must be an object")
        position = _vec(pose_raw.get("position_m"), 3, "pose.position_m")
        quaternion = _vec(pose_raw.get("quaternion_wxyz"), 4, "pose.quaternion_wxyz")
        rpy_from_quat = _quat_to_rpy_wxyz(quaternion)
        yaw = _finite_float(pose_raw.get("yaw_rad"), "pose.yaw_rad")
        pitch = _finite_float(pose_raw.get("pitch_rad"), "pose.pitch_rad")
        roll = _finite_float(pose_raw.get("roll_rad"), "pose.roll_rad")
        rpy_expected = (roll, pitch, yaw)
        if any(abs(_angle_delta(a, b)) > 5e-4 for a, b in zip(rpy_from_quat, rpy_expected)):
            raise WorkerError("pose quaternion_wxyz disagrees with yaw/pitch/roll")
        velocity = _vec(pose_raw.get("velocity_mps"), 3, "pose.velocity_mps")
        angular = _vec(pose_raw.get("angular_velocity_radps"), 3, "pose.angular_velocity_radps")
        from virtual_world import Pose

        pose = Pose(position, yaw, pitch, roll, velocity, angular)
        return seq, t_s, pose, quaternion

    def _pattern_time(self, body_time_s: float) -> float:
        if self.angular_stimulus is not None:
            return float(body_time_s)
        return max(0.0, float(body_time_s) - self.stimulus_onset_s)

    def _standard_visual_manifest(self) -> dict[str, Any]:
        if self.angular_stimulus is not None:
            return {
                "enabled": True,
                "mode": "fixed_arena_angular_field_cpu",
                "file": _small_file_record(self.angular_stimulus_spec_copy_path),
                "spec_sha256": self.angular_stimulus_spec_sha256,
                "metadata": self._angular_stimulus_manifest(),
                "classification": "live_pose_rendered_angular_field",
                "native_recorded_columns_used": False,
            }
        return {"enabled": self.native_input is not None,
                "mode": "flyvis_native_standard_columns" if self.native_input is not None else "world_raycast",
                "file": _small_file_record(self.native_input_path) if self.native_input_path else None,
                "metadata": self.native_input_metadata,
                "classification": "controlled_visual_calibration" if self.native_input is not None else "scene_input"}

    def _motor_capture_manifest(self) -> dict[str, Any]:
        return (self.motor_channel_capture.metadata() if self.motor_channel_capture is not None
                else {"enabled": False})

    def _capture_flyvis_state_row(self) -> None:
        from flyvis_runtime_adapter_eye_parity_v2 import _state_activity_native
        activity = _state_activity_native(self.flyvis._state)
        rows = self.flyvis._mapping_eye_buffer
        nodes = self.flyvis._mapping_node_buffer
        raw = activity[rows, nodes].detach().cpu().numpy().copy()
        if raw.shape != (self.flyvis_state_rows.shape[1],) or not np.all(np.isfinite(raw)):
            raise WorkerError("native FlyVis gather state has invalid shape or values")
        if self.flyvis_state_count != self.frame_count:
            raise WorkerError("FlyVis state recording is not sequential")
        if self.flyvis_baseline_gather is None:
            baseline = self.flyvis._baseline_for_activity(activity)
            if baseline is None:
                raise WorkerError("native state calibration requires an actual static baseline")
            self.flyvis_baseline_gather = baseline[rows, nodes].detach().cpu().numpy().copy()
        self.flyvis_state_rows[self.frame_count] = raw
        self.flyvis_state_count += 1

    def _write_flyvis_state_artifact(self) -> dict[str, Any] | None:
        if not self.capture_flyvis_state:
            return None
        if self.flyvis_state_count != self.frame_count or self.flyvis_baseline_gather is None:
            raise WorkerError("native FlyVis state rows do not match completed frames")
        path = self.out_dir / "flyvis_state.npz"
        metadata = {"schema_version": "flyvis-native-state-gathers-1",
                    "output_gain": float(self.flyvis.output_gain),
                    "state_units": "FlyVis internal activity; biological units uncalibrated",
                    "activation_reference": "ReLU(activity)-ReLU(static_baseline), before target reduction",
                    "observer_only": True, "saved_frames": self.frame_count}
        np.savez_compressed(path, gather_activity=self.flyvis_state_rows[:self.frame_count],
            baseline_gather_activity=self.flyvis_baseline_gather,
            mapping_target_rows=self.flyvis._mapping_target_rows,
            mapping_group_sizes=self.flyvis._mapping_group_sizes,
            mapping_eye_rows=self.flyvis._mapping_eye_rows,
            mapping_flat_nodes=self.flyvis._mapping_flat_nodes,
            target_indices=self.target_indices,
            seq=self._captured_seqs[:self.frame_count],
            time_s=self._captured_body_times[:self.frame_count],
            metadata_json=np.asarray(json.dumps(metadata)))
        return {"path": str(path), "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path)}

    def _render_columns(self, pose: Any, body_time_s: float) -> tuple[np.ndarray, np.ndarray, float]:
        pattern_time_s = self._pattern_time(body_time_s)
        if self.native_input is not None:
            index = round(body_time_s / FRAME_DT_S)
            if not 0 <= index < self.max_frames:
                raise WorkerError("standard visual input clock exceeds its captured frames")
            return (self.native_input["left_columns"][index],
                    self.native_input["right_columns"][index], pattern_time_s)
        if self.angular_stimulus is not None:
            # Convert the fitted FlyVis head-frame rays through the current
            # VirtualWorld pose basis into the fixed arena before sampling.
            # This preserves retinal image motion under yaw; the field never
            # follows the head. Eye origins are intentionally irrelevant to
            # this infinite-distance angular-screen approximation.
            forward, right, up = self.world._camera_basis(pose)
            forward = np.asarray(forward, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)
            up = np.asarray(up, dtype=np.float64)

            def in_arena(head_directions: np.ndarray) -> np.ndarray:
                head = np.asarray(head_directions, dtype=np.float64)
                return (head[:, 0:1] * forward[None, :]
                        - head[:, 1:2] * right[None, :]
                        + head[:, 2:3] * up[None, :])

            left = np.ascontiguousarray(self.angular_stimulus.render_world_directions(
                in_arena(self.directions["left"]), body_time_s,
            ), dtype=np.float32)
            right = np.ascontiguousarray(self.angular_stimulus.render_world_directions(
                in_arena(self.directions["right"]), body_time_s,
            ), dtype=np.float32)
            if left.shape != (721,) or right.shape != (721,):
                raise WorkerError(f"angular renderer returned non-721 eye columns: {left.shape}, {right.shape}")
            if (not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
                    or np.any(left < 0) or np.any(left > 1)
                    or np.any(right < 0) or np.any(right > 1)):
                raise WorkerError("angular renderer returned invalid luminance columns")
            return left, right, pattern_time_s
        assert self.stimulus is not None
        left_raw = self.world.render_eye_directions(
            pose, self.directions["left"], eye="left", timestamp_s=pattern_time_s,
            stimulus=self.stimulus,
        )
        right_raw = self.world.render_eye_directions(
            pose, self.directions["right"], eye="right", timestamp_s=pattern_time_s,
            stimulus=self.stimulus,
        )
        left = np.ascontiguousarray(np.asarray(left_raw, dtype=np.float32))
        right = np.ascontiguousarray(np.asarray(right_raw, dtype=np.float32))
        if left.shape != (721,) or right.shape != (721,):
            raise WorkerError(f"VirtualWorld returned non-721 eye columns: {left.shape}, {right.shape}")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise WorkerError("rendered eye columns contain non-finite luminance")
        if np.any(left < 0) or np.any(left > 1) or np.any(right < 0) or np.any(right > 1):
            raise WorkerError("rendered eye luminance must be in [0,1]")
        return left, right, pattern_time_s

    def _warm_static_at_first_pose(self, pose: Any) -> None:
        left, right, pattern_time_s = self._render_columns(pose, 0.0)
        method = getattr(self.flyvis, "preadapt_static_scene", None)
        if not callable(method):
            raise WorkerError("real FlyVis adapter lacks static-scene preadaptation; no fallback is allowed")
        self.warmup_record = _json_safe(method(left, right, seconds=self.static_warmup_s))
        if not isinstance(self.warmup_record, dict):
            raise WorkerError("FlyVis static-scene preadaptation returned invalid metadata")
        self.warmup_record["body_time_s"] = 0.0
        self.warmup_record["render_time_s"] = pattern_time_s
        self.warmup_record["pattern_time_s"] = pattern_time_s
        if self.angular_stimulus is not None:
            self.warmup_record["panel_visible"] = False
            self.warmup_record["angular_field_present"] = True
            self.warmup_record["grating_active"] = False
            self.warmup_record["angular_stimulus_spec_sha256"] = self.angular_stimulus_spec_sha256
        elif self.native_input is not None:
            self.warmup_record["panel_visible"] = False
            self.warmup_record["standard_visual_input"] = self._standard_visual_manifest()
        else:
            assert self.stimulus is not None
            warmup_stimulus_state = self.stimulus.state_at(pattern_time_s)
            self.warmup_record["panel_visible"] = bool(
                warmup_stimulus_state.active and warmup_stimulus_state.renderable
            )
            if not self.warmup_record["panel_visible"]:
                raise WorkerError("the static FlyVis warmup scene does not contain the selected panel")
        self.provenance["static_visual_warmup"]["result"] = self.warmup_record
        self.provenance_path.write_text(
            json.dumps(self.provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def process_frame(self, request: Any) -> dict[str, Any]:
        if self.frame_count >= self.max_frames:
            raise WorkerError(f"frame limit {self.max_frames} reached; stop the worker before sending another frame")
        seq, t_s, pose, quaternion = self._validate_frame(request)
        if self.frame_count == 0:
            self._warm_static_at_first_pose(pose)

        left_columns, right_columns, pattern_time_s = self._render_columns(pose, t_s)
        if self.angular_stimulus is not None:
            stimulus_state = {"fixed_arena_angular_field": {
                "active": True,
                "renderable": True,
                "grating_active": self.angular_stimulus.onset_s <= t_s < self.angular_stimulus.end_s,
                "body_time_s": t_s,
                "pattern_time_s": pattern_time_s,
                "spec_sha256": self.angular_stimulus_spec_sha256,
                "display_geometry": "fixed_arena_infinite_distance_angular_field",
                "head_angular_speed_deg_s": self.angular_stimulus.angular_speed_deg_s,
                "arena_angular_speed_deg_s": self.angular_stimulus.arena_angular_speed_deg_s,
            }}
        elif self.native_input is None:
            assert self.stimulus is not None
            stimulus_state = self.world.stimulus_state_at(pattern_time_s, self.stimulus)
        else:
            stimulus_state = {"native_standard_columns": {
                "active": True, "renderable": True,
                "input_source": str(self.native_input_path),
                "physical_scene_rendered": False,
                "body_time_s": t_s, "pattern_time_s": pattern_time_s}}
        if not stimulus_state:
            raise WorkerError("selected visual panel is absent from the pattern scene")
        panel_state = next(iter(stimulus_state.values()))
        if not bool(panel_state.get("active")) or not bool(panel_state.get("renderable")):
            raise WorkerError("selected visual panel is not active/renderable on the pattern clock")
        self._captured_seqs[self.frame_count] = seq
        self._captured_body_times[self.frame_count] = t_s
        self._captured_pattern_times[self.frame_count] = pattern_time_s
        self._captured_left_columns[self.frame_count] = left_columns
        self._captured_right_columns[self.frame_count] = right_columns
        left_eye_hash = _array_sha256(left_columns)
        right_eye_hash = _array_sha256(right_columns)
        visual_output = self.flyvis.step_detailed(left_columns, right_columns)
        if self.capture_flyvis_state:
            self._capture_flyvis_state_row()
        if int(visual_output.frame_index) != self.frame_count:
            raise WorkerError(
                f"FlyVis frame index should be {self.frame_count}; got {visual_output.frame_index}"
            )
        values = np.ascontiguousarray(np.asarray(visual_output.drive, dtype=np.float32))
        target_indices = np.asarray(visual_output.target_indices)
        if values.shape != self.target_indices.shape or not np.array_equal(target_indices, self.target_indices):
            raise WorkerError("FlyVis drive vector or target order changed after initialization")
        if not np.all(np.isfinite(values)):
            raise WorkerError("FlyVis T4/T5 target drive contains non-finite values")
        units = str(visual_output.units)
        if units not in {"poisson_hz", "current_mV_per_ms"}:
            raise WorkerError(f"MaleCNS does not support FlyVis units {units!r}")
        visual_drive_hash = _array_sha256(values)
        visual_drive_mean = float(np.mean(values, dtype=np.float64))
        visual_drive_abs_mean = float(np.mean(np.abs(values), dtype=np.float64))
        visual_drive_std = float(np.std(values, dtype=np.float64))
        visual_drive_capture_row = self._capture_visual_drive_frame(
            seq=seq, time_s=t_s, values=values, units=units,
        )
        ts = _timestamp(t_s)
        from twin_contracts import SensoryFrame, SourceLabel

        # Only the actual sparse fitted T4/T5-to-MaleCNS mapping enters the
        # engine. Pose drives the renderer; no body/contact afferent is guessed.
        flyvis_contract = {
            "values": values,
            "target_indices": self.target_indices,
            "units": units,
        }
        intervention_rates, intervention_currents, neural_intervention_record = (
            self._frame_neural_intervention(t_s)
        )
        sensory_channels: dict[str, Any] = {"flyvis_t4t5": flyvis_contract}
        if intervention_rates:
            sensory_channels["poisson_hz"] = dict(intervention_rates)
        if intervention_currents:
            sensory_channels["current_mV_per_ms"] = dict(intervention_currents)
        sensory = SensoryFrame(
            timestamp=ts,
            visual=None,
            body=None,
            channels=sensory_channels,
            source=SourceLabel.FITTED,
        )
        raw_neural = self.engine.step_window(
            sensory, ts, steps=NEURAL_STEPS_PER_FRAME,
        )
        if raw_neural is None:
            raise WorkerError("MaleCNS step_window returned no output; scalar/stub fallback is prohibited")
        if self.motor_channel_capture is not None:
            self.motor_channel_capture.capture(raw_neural.spikes, seq, t_s)
        named = dict(raw_neural.named_outputs)
        if int(named.get("neural_window_steps", -1)) != NEURAL_STEPS_PER_FRAME:
            raise WorkerError("MaleCNS did not integrate exactly ten 0.5 ms ticks")
        if visual_drive_capture_row is not None:
            target_spike_capture = self._capture_visual_target_spikes(
                raw_neural.spikes, row_index=int(visual_drive_capture_row["row_index"]),
            )
            visual_drive_capture_row["target_spike_counts"] = target_spike_capture
        expected_end_s = t_s + FRAME_DT_S
        reported_start = named.get("window_start_timestamp")
        reported_end = named.get("window_end_timestamp")
        if not isinstance(reported_start, Mapping) or not isinstance(reported_end, Mapping):
            raise WorkerError("MaleCNS window omitted explicit start/end timestamps")
        if abs(float(reported_start.get("t_s", math.nan)) - t_s) > 1e-8:
            raise WorkerError("MaleCNS window start timestamp does not match the input frame")
        if abs(float(reported_end.get("t_s", math.nan)) - (expected_end_s - NEURAL_DT_S)) > 1e-8:
            raise WorkerError("MaleCNS window end tick does not match the tenth 0.5 ms tick")
        debug_neural_state_record = self._capture_debug_neural_state_frame(
            seq=seq, time_s=t_s, named_outputs=named,
        )
        if raw_neural.timestamp != ts:
            raw_neural = raw_neural.with_timestamp(ts, named_outputs=named)
        motor_output = self.neuromuscular.step(
            raw_neural, dt_ms=FRAME_DT_S * 1000.0, timestamp=ts,
        )
        leg_power = _finite_float(motor_output.leg_power, "leg_power")
        leg_steering = _finite_float(motor_output.leg_steering, "leg_steering")
        left_drive = min(1.0, max(0.0, leg_power - 0.5 * leg_steering))
        right_drive = min(1.0, max(0.0, leg_power + 0.5 * leg_steering))
        if (abs(left_drive - float(motor_output.left_leg_power)) > 1e-9
                or abs(right_drive - float(motor_output.right_leg_power)) > 1e-9):
            raise WorkerError("NeuromuscularAdapter bilateral leg mix changed unexpectedly")
        population_rates = named.get("population_rates_hz")
        population_spikes = named.get("population_spike_counts")
        if not isinstance(population_rates, Mapping) or not isinstance(population_spikes, Mapping):
            raise WorkerError("MaleCNS output omitted requested population rates/spike counts")
        safe_rates = {str(key): _finite_float(value, f"population_rates_hz[{key}]")
                      for key, value in population_rates.items()}
        safe_spikes: dict[str, int] = {}
        for key, value in population_spikes.items():
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise WorkerError(f"population_spike_counts[{key}] is not a non-negative integer")
            safe_spikes[str(key)] = int(value)
        neural_last_step = int(named.get("step_index", -1))
        expected_last_step = self.frame_count * NEURAL_STEPS_PER_FRAME + NEURAL_STEPS_PER_FRAME - 1
        if neural_last_step != expected_last_step:
            raise WorkerError(
                "MaleCNS neural steps are not contiguous from step zero: "
                f"expected window ending at {expected_last_step}, got {neural_last_step}"
            )
        diagnostic_vm_by_index, diagnostic_population_vm_means, diagnostic_vm_time_s = (
            self._sample_diagnostic_membrane_vm()
        )
        expected_diagnostic_vm_time_s = t_s + (NEURAL_STEPS_PER_FRAME - 1) * NEURAL_DT_S
        if abs(diagnostic_vm_time_s - expected_diagnostic_vm_time_s) > 1e-8:
            raise WorkerError(
                "diagnostic Vm sample time does not match the final integrated 0.5 ms tick"
            )

        reply: dict[str, Any] = {
            "protocol": PROTOCOL,
            "type": "drive",
            "seq": seq,
            "time_s": t_s,
            "render_time_s": pattern_time_s,
            "pattern_time_s": pattern_time_s,
            "neural_window_end_s": expected_end_s,
            "left_descending_drive": left_drive,
            "right_descending_drive": right_drive,
            "leg_power": leg_power,
            "leg_steering": leg_steering,
            "population_rates_hz": safe_rates,
            "population_spike_counts": safe_spikes,
            "neural_intervention": neural_intervention_record,
            **({"debug_neural_state": debug_neural_state_record}
               if debug_neural_state_record is not None else {}),
            **({"visual_drive_capture_row": visual_drive_capture_row}
               if visual_drive_capture_row is not None else {}),
            "diagnostic_neuron_vm_mv_by_csr_index": diagnostic_vm_by_index,
            "diagnostic_population_vm_mv_mean": diagnostic_population_vm_means,
            "diagnostic_population_vm_sample_count": {
                name: int(record["count"])
                for name, record in self.diagnostic_population_membership.items()
            },
            "diagnostic_vm_sample_time_s": diagnostic_vm_time_s,
            "diagnostic_vm_sample_offset_within_window_s": diagnostic_vm_time_s - t_s,
            "diagnostic_vm_sample_semantics": (
                "instantaneous engine.v after the tenth 0.5 ms tick; "
                "sample time is 4.5 ms into the 5 ms frame window, not a 5 ms mean"
            ),
            "vision_frame_id": int(visual_output.frame_index),
            "neural_frame_id": neural_last_step,
            "neural_window_start_step_index": neural_last_step - NEURAL_STEPS_PER_FRAME + 1,
            "neural_window_end_step_index": neural_last_step,
            "eye_column_hashes": {"left": left_eye_hash, "right": right_eye_hash},
            "visual_drive_hash": visual_drive_hash,
            "visual_drive_mean": visual_drive_mean,
            "visual_drive_abs_mean": visual_drive_abs_mean,
            "visual_drive_std": visual_drive_std,
            "flyvis_target_indices_sha256": self.target_indices_hash,
            "flyvis_target_units": units,
            "effective_backends": self.effective_backends,
            "source": {
                "vision": "fitted_flyvis",
                "neural": "connectome_malecns",
                "motor_readout": "NeuromuscularAdapter",
                "mechanosensory_feedback": "unsupported_and_not_injected",
            },
            "artifact_dir": str(self.out_dir),
        }
        record = {
            **reply,
            "input_pose": _json_safe(request["pose"]),
            "input_quaternion_wxyz": list(quaternion),
            "body_time_s": t_s,
            "render_time_s": pattern_time_s,
            "pattern_time_s": pattern_time_s,
            "visual_drive_mean": visual_drive_mean,
            "visual_drive_abs_mean": visual_drive_abs_mean,
            "visual_drive_std": visual_drive_std,
            "stimulus_condition": self.condition,
            "stimulus_state": {
                "body_time_s": t_s,
                "pattern_time_s": pattern_time_s,
                "state": _json_safe(stimulus_state),
            },
            "population_report_count": len(safe_rates),
            "motor_output_source": str(motor_output.source),
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        row_hash = _sha256_bytes(canonical)
        reply["artifact_checksums"] = {"frame_record_content_sha256": row_hash}
        record["artifact_checksums"] = reply["artifact_checksums"]
        self.frames_handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.frames_handle.flush()
        self.frame_count += 1
        self.last_seq = seq
        self.last_time_s = t_s
        return reply

    def stop(self) -> dict[str, Any]:
        self.frames_handle.flush()
        os.fsync(self.frames_handle.fileno())
        self.frames_handle.close()
        eye_columns_record = self._write_eye_columns_artifact()
        if (self.capture_visual_drive
                and self.visual_drive_target_spikes_captured != self.visual_drive_capture_count):
            raise WorkerError("visual-drive and target-spike capture row counts disagree")
        visual_drive_record = self._write_visual_drive_artifact()
        motor_record = (self.motor_channel_capture.finalize()
                        if self.motor_channel_capture is not None else None)
        state_record = self._write_flyvis_state_artifact()
        self.provenance["visual_capture"].update({
            "frames_saved": self.frame_count,
            "expected_shape_per_eye": [self.frame_count, 721],
            "size_bytes": eye_columns_record["size_bytes"],
            "sha256": eye_columns_record["sha256"],
        })
        self.provenance["debug_neural_state"] = self._debug_neural_state_manifest()
        self.provenance["visual_drive_capture"] = self._visual_drive_capture_manifest()
        self.provenance["motor_channel_capture"] = self._motor_capture_manifest()
        self.provenance["flyvis_state_capture"]["artifact"] = state_record
        self.provenance_path.write_text(
            json.dumps(self.provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        integrity = {
            "schema_version": "flygym-online-neural-integrity-1",
            "protocol": PROTOCOL,
            "frame_count": self.frame_count,
            "last_seq": self.last_seq,
            "last_time_s": self.last_time_s,
            "files": {
                "frames.jsonl": {
                    "path": str(self.frames_path),
                    "size_bytes": self.frames_path.stat().st_size,
                    "sha256": _sha256_file(self.frames_path),
                },
                "provenance.json": {
                    "path": str(self.provenance_path),
                    "size_bytes": self.provenance_path.stat().st_size,
                    "sha256": _sha256_file(self.provenance_path),
                },
                **({"angular_stimulus_spec.json": _small_file_record(
                    self.angular_stimulus_spec_copy_path)}
                   if self.angular_stimulus_spec_copy_path is not None else {}),
                "eye_columns.npz": eye_columns_record,
                **({"visual_drive.npz": visual_drive_record}
                   if visual_drive_record is not None else {}),
                **({"muscle_channels.npz": motor_record} if motor_record is not None else {}),
                **({"muscle_channels_membership.json": _small_file_record(
                    self.motor_channel_capture.membership_path)} if motor_record is not None else {}),
                **({"flyvis_state.npz": state_record} if state_record is not None else {}),
            },
            **({"debug_neural_state_files": [
                {key: record[key] for key in ("path", "size_bytes", "sha256")}
                for record in (
                    ([self.debug_neural_state_initial_record]
                     if self.debug_neural_state_initial_record is not None else [])
                    + self.debug_neural_state_frame_records
                )
            ]} if self.debug_neural_state_frames else {}),
        }
        self.integrity_path.write_text(
            json.dumps(integrity, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "protocol": PROTOCOL,
            "type": "stopped",
            "frame_count": self.frame_count,
            "last_seq": self.last_seq,
            "last_time_s": self.last_time_s,
            "artifact_checksums": {
                "frames_jsonl_sha256": integrity["files"]["frames.jsonl"]["sha256"],
                "provenance_json_sha256": integrity["files"]["provenance.json"]["sha256"],
                "eye_columns_npz_sha256": eye_columns_record["sha256"],
                **({"visual_drive_npz_sha256": visual_drive_record["sha256"]}
                   if visual_drive_record is not None else {}),
                "integrity_json_sha256": _sha256_file(self.integrity_path),
            },
        }

    def _write_eye_columns_artifact(self) -> dict[str, Any]:
        count = self.frame_count
        np.savez_compressed(
            self.eye_columns_path,
            seq=self._captured_seqs[:count],
            frame_time_s=self._captured_body_times[:count],
            pattern_time_s=self._captured_pattern_times[:count],
            left_columns=self._captured_left_columns[:count],
            right_columns=self._captured_right_columns[:count],
            schema_version=np.asarray("flygym-online-eye-columns-1"),
        )
        return {
            "path": str(self.eye_columns_path),
            "size_bytes": self.eye_columns_path.stat().st_size,
            "sha256": _sha256_file(self.eye_columns_path),
        }

    def _capture_visual_drive_frame(
        self, *, seq: int, time_s: float, values: np.ndarray, units: str,
    ) -> dict[str, Any] | None:
        if self._captured_visual_drive_values is None:
            return None
        if self.visual_drive_capture_count >= self.max_frames:
            raise WorkerError("visual-drive capture exceeded the configured frame limit")
        if values.shape != (self.flyvis.n_t4t5,) or values.dtype != np.float32:
            raise WorkerError("visual-drive capture requires the exact float32 fitted target vector")
        if self._captured_visual_drive_units is not None and units != self._captured_visual_drive_units:
            raise WorkerError("FlyVis target units changed during visual-drive capture")
        self._captured_visual_drive_units = str(units)
        row_index = self.visual_drive_capture_count
        self._captured_visual_drive_values[row_index] = values
        self.visual_drive_capture_count += 1
        return {
            "row_index": int(row_index),
            "seq": int(seq),
            "time_s": float(time_s),
            "units": str(units),
            "target_indices_sha256": self.target_indices_hash,
        }

    def _capture_visual_target_spikes(self, dense_counts: Any, *, row_index: int) -> dict[str, Any]:
        if self._captured_visual_target_spike_counts is None:
            raise WorkerError("target spike capture is unavailable without visual-drive capture")
        try:
            counts = np.asarray(dense_counts)
        except Exception as exc:
            raise WorkerError(f"step_window did not return host spike counts: {exc}") from exc
        if (counts.shape != (self.engine.n,)
                or not (np.issubdtype(counts.dtype, np.integer)
                        or np.issubdtype(counts.dtype, np.floating))):
            raise WorkerError("step_window spike counts must be a full numeric host neuron vector")
        if (not np.all(np.isfinite(counts)) or np.any(counts < 0)
                or np.any(counts > NEURAL_STEPS_PER_FRAME)
                or np.any(counts != np.floor(counts))):
            raise WorkerError(
                f"step_window spike counts must be integer values in [0, {NEURAL_STEPS_PER_FRAME}]"
            )
        if row_index != self.visual_drive_target_spikes_captured:
            raise WorkerError("target spike rows are not aligned with visual-drive capture rows")
        target_counts = counts[self.target_indices]
        self._captured_visual_target_spike_counts[row_index] = target_counts.astype(np.uint16)
        self.visual_drive_target_spikes_captured += 1
        return {
            "row_index": int(row_index),
            "shape": [int(self.flyvis.n_t4t5)],
            "dtype": "uint16",
            "window_steps": NEURAL_STEPS_PER_FRAME,
            "maximum_count_per_neuron": NEURAL_STEPS_PER_FRAME,
        }

    def _write_visual_drive_artifact(self) -> dict[str, Any] | None:
        if self._captured_visual_drive_values is None:
            return None
        count = self.visual_drive_target_spikes_captured
        units = self._captured_visual_drive_units or str(self.flyvis.units)
        ordering = (
            "drive_values rows follow seq/time_s; columns follow target_indices exactly in "
            "FlyVisRuntimeAdapter output order"
        )
        metadata = {
            "schema_version": "flygym-online-visual-drive-1",
            "frames_saved": count,
            "drive_values_shape": [count, int(self.flyvis.n_t4t5)],
            "drive_values_dtype": "float32",
            "target_indices_shape": [int(self.target_indices.size)],
            "target_spike_counts_shape": [count, int(self.flyvis.n_t4t5)],
            "target_spike_counts_dtype": "uint16",
            "target_spike_count_window_steps": NEURAL_STEPS_PER_FRAME,
            "target_spike_count_window_dt_s": FRAME_DT_S,
            "target_spike_count_maximum_per_neuron_per_window": NEURAL_STEPS_PER_FRAME,
            "target_spike_count_semantics": (
                "exact per-neuron spike counts over this frame's ten 0.5 ms MaleCNS ticks"
            ),
            "target_indices_sha256": self.target_indices_hash,
            "ordering": ordering,
        }
        np.savez_compressed(
            self.visual_drive_path,
            drive_values=self._captured_visual_drive_values[:count],
            target_spike_counts=self._captured_visual_target_spike_counts[:count],
            target_indices=self.target_indices,
            time_s=self._captured_body_times[:count],
            seq=self._captured_seqs[:count],
            units=np.asarray(units),
            schema_version=np.asarray("flygym-online-visual-drive-1"),
            ordering=np.asarray(ordering),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        )
        record = {
            "path": str(self.visual_drive_path),
            "size_bytes": int(self.visual_drive_path.stat().st_size),
            "sha256": _sha256_file(self.visual_drive_path),
            "frames_saved": count,
            "input_drive_rows_captured": self.visual_drive_capture_count,
            "drive_values_shape": [count, int(self.flyvis.n_t4t5)],
            "drive_values_dtype": "float32",
            "target_indices_shape": [int(self.target_indices.size)],
            "target_indices_dtype": str(self.target_indices.dtype),
            "target_indices_sha256": self.target_indices_hash,
            "target_spike_counts_shape": [count, int(self.flyvis.n_t4t5)],
            "target_spike_counts_dtype": "uint16",
            "target_spike_count_window_steps": NEURAL_STEPS_PER_FRAME,
            "target_spike_count_window_dt_s": FRAME_DT_S,
            "target_spike_count_maximum_per_neuron_per_window": NEURAL_STEPS_PER_FRAME,
            "units": units,
            "ordering": ordering,
        }
        self.visual_drive_file_record = record
        return record

    def _visual_drive_capture_manifest(self) -> dict[str, Any]:
        enabled = self._captured_visual_drive_values is not None
        record = self.visual_drive_file_record
        return {
            "enabled": enabled,
            "path": str(self.visual_drive_path) if enabled else None,
            "file_sha256": None if record is None else record["sha256"],
            "size_bytes": None if record is None else record["size_bytes"],
            "saved_frames": self.visual_drive_target_spikes_captured if enabled else 0,
            "input_drive_rows_captured": self.visual_drive_capture_count if enabled else 0,
            "drive_values_shape": (
                [self.visual_drive_target_spikes_captured, int(self.flyvis.n_t4t5)] if enabled else None
            ),
            "maximum_drive_values_shape": (
                [self.max_frames, int(self.flyvis.n_t4t5)] if enabled else None
            ),
            "drive_values_dtype": "float32" if enabled else None,
            "target_indices_shape": [int(self.target_indices.size)] if enabled else None,
            "target_indices_sha256": self.target_indices_hash if enabled else None,
            "target_spike_counts_shape": (
                [self.visual_drive_target_spikes_captured, int(self.flyvis.n_t4t5)]
                if enabled else None
            ),
            "target_spike_counts_dtype": "uint16" if enabled else None,
            "target_spike_count_window_steps": NEURAL_STEPS_PER_FRAME if enabled else None,
            "target_spike_count_window_dt_s": FRAME_DT_S if enabled else None,
            "target_spike_count_maximum_per_neuron_per_window": (
                NEURAL_STEPS_PER_FRAME if enabled else None
            ),
            "units": self._captured_visual_drive_units if enabled else None,
            "ordering": (
                "rows follow seq/time_s; columns follow target_indices exactly in "
                "FlyVisRuntimeAdapter output order"
            ) if enabled else None,
            "file_record": record,
        }

    def fail(self, exc: BaseException) -> dict[str, Any]:
        message = f"{type(exc).__name__}: {exc}"
        failure = {
            "schema_version": "flygym-online-neural-failure-1",
            "protocol": PROTOCOL,
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "last_seq": self.last_seq,
            "last_time_s": self.last_time_s,
            "error": message,
        }
        try:
            if not self.frames_handle.closed:
                self.frames_handle.flush()
                os.fsync(self.frames_handle.fileno())
                self.frames_handle.close()
            if not self.eye_columns_path.exists():
                eye_columns_record = self._write_eye_columns_artifact()
                self.provenance["visual_capture"].update({
                    "frames_saved": self.frame_count,
                    "expected_shape_per_eye": [self.frame_count, 721],
                    "size_bytes": eye_columns_record["size_bytes"],
                    "sha256": eye_columns_record["sha256"],
                })
                self.provenance_path.write_text(
                    json.dumps(self.provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            self.provenance["debug_neural_state"] = self._debug_neural_state_manifest()
            if self._captured_visual_drive_values is not None and self.visual_drive_file_record is None:
                self._write_visual_drive_artifact()
            self.provenance["visual_drive_capture"] = self._visual_drive_capture_manifest()
            self.provenance_path.write_text(
                json.dumps(self.provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self.failure_path.write_text(
                json.dumps(failure, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return {"protocol": PROTOCOL, "type": "error", "error": message,
                "last_seq": self.last_seq, "last_time_s": self.last_time_s}


def main() -> int:
    protocol_stdout = sys.stdout
    args = _parse_cli()
    # Native libraries may log to stdout during context/kernel setup. Keep the
    # wire clean while preserving one explicit stream for protocol JSONL.
    sys.stdout = sys.stderr
    worker: OnlineNeuralWorker | None = None
    try:
        worker = OnlineNeuralWorker(args)
        _emit(worker.ready_record(), protocol_stdout)
        for line_number, line in enumerate(sys.stdin, start=1):
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except ValueError as exc:
                raise WorkerError(f"stdin line {line_number} is malformed JSON: {exc}") from exc
            if isinstance(request, Mapping) and request.get("protocol") == PROTOCOL and request.get("type") == "stop":
                _emit(worker.stop(), protocol_stdout)
                return 0
            _emit(worker.process_frame(request), protocol_stdout)
        raise WorkerError("stdin closed without an explicit stop record")
    except BaseException as exc:
        if worker is not None:
            _emit(worker.fail(exc), protocol_stdout)
        else:
            _emit({"protocol": PROTOCOL, "type": "error",
                   "error": f"{type(exc).__name__}: {exc}"}, protocol_stdout)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
