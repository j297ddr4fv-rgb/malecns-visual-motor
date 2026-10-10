#!/usr/bin/env python3
"""Replay a sparse MaleCNS trace through the neuromuscular/body boundary.

This module is an *offline bridge diagnostic*.  It does not run a neural
network, infer a threat label, or synthesize a behavioural command from the
stimulus.  It reads the sparse trace written by :mod:`lif_flyb`, resolves the
configured selectors against the CSR annotations, aggregates the recorded
events into exact 5 ms ``NeuralOutput`` windows, and sends those outputs to
``NeuromuscularAdapter`` and the reduced combined wing/leg dynamics.

The grounded ``looming_grounded`` stimulus is deliberately a separate
geometry stream.  It is available to the renderer and the collision/clearance
measurement only.  No stimulus state, contact time, direction, or label is
included in the neural input or body command.  The resulting artifact is
therefore not autonomous closed-loop evidence; it is a reproducible
trace-to-body wiring diagnostic.

The board is never an actuator or a termination condition.  OBB overlap is
reported as a *counterfactual would-contact* measurement: the geometry says
the palm would overlap the body at that sampled state, but no palm force,
contact impulse, body stop, or collision response is applied.  The replay
always consumes every complete 5 ms window present in the trace.

The sparse trace format is the one emitted by ``lif_flyb.py``::

    times   (T,) float milliseconds; each value is the end of one trace frame
    offsets (T+1,) int; CSR-like frame boundaries into ``neurons``
    neurons  (N,) int dense CSR neuron indices

Counts and rates are exact with respect to those recorded sparse frame events.
If the upstream trace stored a per-frame union, repeated spikes of one neuron
within that frame cannot be reconstructed and are not silently invented.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neuromuscular_adapter import NeuromuscularAdapter  # noqa: E402


SCHEMA_VERSION = "escape-trace-replay-1"
DEFAULT_CSR = Path("/data/cache/malecns-gpu/csr-flyb")
DEFAULT_MAP = ROOT / "config" / "neuromuscular_map.json"
DEFAULT_WORLD = ROOT / "config" / "scenarios.json"
DEFAULT_STIMULUS = ROOT / "config" / "escape_stimuli.json"
DEFAULT_WINDOW_MS = 5.0
DEFAULT_BODY_SIZE_M = (0.0030, 0.0015, 0.0012)
DEFAULT_BODY_GROUND_Z_M = 0.0012


class ReplayError(ValueError):
    """Raised when a sparse replay input or bridge contract is invalid."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ReplayError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ReplayError(f"{name} must be finite")
    return result


def _vec3(value: Iterable[Any], name: str) -> tuple[float, float, float]:
    try:
        values = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise ReplayError(f"{name} must contain three finite values") from exc
    if len(values) != 3:
        raise ReplayError(f"{name} must contain three finite values")
    return values  # type: ignore[return-value]


def _json_safe(value: Any) -> Any:
    """Convert NumPy/dataclass-adjacent values into finite JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    return repr(value)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReplayError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReplayError(f"JSON metadata {path} must contain an object")
    return value


@dataclass(frozen=True)
class SparseSpikeTrace:
    """Validated sparse frames from a ``lif_flyb`` ``spikes.npz`` file."""

    times_ms: np.ndarray
    offsets: np.ndarray
    neurons: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_ms, dtype=np.float64).reshape(-1)
        offsets = np.asarray(self.offsets, dtype=np.int64).reshape(-1)
        neurons = np.asarray(self.neurons, dtype=np.int64).reshape(-1)
        if times.size == 0:
            raise ReplayError("sparse spike trace has no frames")
        if not np.all(np.isfinite(times)) or np.any(times < 0.0):
            raise ReplayError("trace times_ms must be finite and non-negative")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ReplayError("trace times_ms must be strictly increasing")
        if offsets.size != times.size + 1:
            raise ReplayError(
                f"trace offsets length {offsets.size} must equal frames+1 ({times.size + 1})"
            )
        if offsets[0] != 0 or offsets[-1] != neurons.size or np.any(np.diff(offsets) < 0):
            raise ReplayError("trace offsets must be monotone, start at zero, and end at neurons.size")
        if np.any(neurons < 0):
            raise ReplayError("trace neuron indices must be non-negative")
        object.__setattr__(self, "times_ms", times)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "neurons", neurons)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "source", None if self.source is None else str(self.source))

    @property
    def frame_count(self) -> int:
        return int(self.times_ms.size)

    @property
    def spike_event_count(self) -> int:
        return int(self.neurons.size)

    @property
    def duration_ms(self) -> float:
        """End time of the final recorded trace frame."""

        return float(self.times_ms[-1])

    @property
    def trace_frame_ms(self) -> Optional[float]:
        timing = self.metadata.get("timing_contract")
        if isinstance(timing, Mapping):
            raw = timing.get("trace_frame_ms")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value > 0.0:
                return value
        if self.times_ms.size > 1:
            diffs = np.diff(self.times_ms)
            candidate = float(np.median(diffs))
            if math.isfinite(candidate) and candidate > 0.0:
                return candidate
        return None

    def frame(self, index: int) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.neurons[start:end]

    @classmethod
    def from_npz(cls, path: str | Path) -> "SparseSpikeTrace":
        requested = Path(path)
        source = requested / "spikes.npz" if requested.is_dir() else requested
        if not source.exists():
            raise ReplayError(f"sparse trace not found: {source}")
        try:
            with np.load(source, allow_pickle=False) as loaded:
                missing = [key for key in ("times", "offsets", "neurons") if key not in loaded.files]
                if missing:
                    raise ReplayError(f"{source} is missing sparse trace arrays: {', '.join(missing)}")
                times = np.array(loaded["times"], copy=True)
                offsets = np.array(loaded["offsets"], copy=True)
                neurons = np.array(loaded["neurons"], copy=True)
        except ReplayError:
            raise
        except (OSError, ValueError) as exc:
            raise ReplayError(f"cannot load sparse trace {source}: {exc}") from exc

        metadata: Mapping[str, Any] = {}
        summary = source.parent / "summary.json"
        if summary.exists():
            metadata = _load_json(summary)
        return cls(times, offsets, neurons, metadata, str(source.resolve()))


def load_sparse_trace(path: str | Path) -> SparseSpikeTrace:
    """Load and validate a sparse ``lif_flyb`` trace."""

    return SparseSpikeTrace.from_npz(path)


@dataclass(frozen=True)
class PopulationWindow:
    """One exact aggregation window in the neural 5 ms clock."""

    index: int
    start_ms: float
    end_ms: float
    rates_hz: Mapping[str, float]
    spike_counts: Mapping[str, int]

    @property
    def dt_ms(self) -> float:
        return float(self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "start_ms": float(self.start_ms),
            "end_ms": float(self.end_ms),
            "dt_ms": float(self.dt_ms),
            "population_rates_hz": dict(self.rates_hz),
            "population_spike_counts": dict(self.spike_counts),
        }


def configured_selectors(adapter: NeuromuscularAdapter) -> tuple[str, ...]:
    """Return unique selector terms in map order, preserving provenance."""

    selectors: list[str] = []
    seen: set[str] = set()
    for channel in adapter.channels.values():
        for term in channel.terms:
            selector = str(term.spec)
            if selector not in seen:
                selectors.append(selector)
                seen.add(selector)
    return tuple(selectors)


def _window_index(time_ms: float, window_ms: float) -> int:
    """Assign a frame-end timestamp to its half-open window.

    ``lif_flyb`` records the end of each trace frame.  Thus a frame ending
    exactly at 5 ms belongs to [0, 5], while a frame ending just after 5 ms
    belongs to the next window.  The tiny tolerance only removes floating
    point noise at exact boundaries.
    """

    if time_ms <= 1.0e-9:
        return 0
    return max(0, int(math.ceil((time_ms - 1.0e-9) / window_ms) - 1))


def aggregate_population_windows(
    trace: SparseSpikeTrace | str | Path,
    adapter: NeuromuscularAdapter,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    selectors: Optional[Sequence[str]] = None,
) -> tuple[PopulationWindow, ...]:
    """Aggregate sparse selector events into exact windows and per-population Hz.

    A selector's rate is ``count / (resolved_neurons * 0.005 s)``.  The
    selector counts are independent, so overlapping map terms remain
    auditable and are not renormalized against one another.
    """

    trace = trace if isinstance(trace, SparseSpikeTrace) else load_sparse_trace(trace)
    window_ms = _finite(window_ms, "window_ms")
    if window_ms <= 0.0:
        raise ReplayError("window_ms must be positive")
    if not math.isclose(window_ms, DEFAULT_WINDOW_MS, rel_tol=0.0, abs_tol=1.0e-9):
        raise ReplayError("the escape replay bridge requires an exact 5 ms window")
    selected = tuple(str(item) for item in (selectors or configured_selectors(adapter)))
    if not selected:
        raise ReplayError("neuromuscular map contains no selectors")

    selector_indices: dict[str, np.ndarray] = {}
    for selector in selected:
        indices = np.asarray(adapter.resolve(selector), dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ReplayError(f"selector resolves to zero CSR neurons: {selector}")
        selector_indices[selector] = np.unique(indices)

    n_windows = int(math.floor((trace.duration_ms + 1.0e-9) / window_ms))
    if n_windows <= 0:
        raise ReplayError(
            f"trace duration {trace.duration_ms:g} ms is shorter than one {window_ms:g} ms window"
        )
    counts = {selector: np.zeros(n_windows, dtype=np.int64) for selector in selected}
    for frame_index, time_ms in enumerate(trace.times_ms):
        window = _window_index(float(time_ms), window_ms)
        if window < 0 or window >= n_windows:
            # A final incomplete frame is intentionally excluded: every
            # NeuralOutput produced below represents exactly window_ms.
            continue
        frame_neurons = trace.frame(frame_index)
        if frame_neurons.size == 0:
            continue
        for selector, indices in selector_indices.items():
            counts[selector][window] += int(np.count_nonzero(np.isin(frame_neurons, indices)))

    interval_s = window_ms / 1000.0
    windows: list[PopulationWindow] = []
    for index in range(n_windows):
        count_map = {selector: int(values[index]) for selector, values in counts.items()}
        rate_map = {
            selector: float(count_map[selector] / (selector_indices[selector].size * interval_s))
            for selector in selected
        }
        windows.append(PopulationWindow(
            index=index,
            start_ms=index * window_ms,
            end_ms=(index + 1) * window_ms,
            rates_hz=rate_map,
            spike_counts=count_map,
        ))
    return tuple(windows)


def neural_output_from_window(window: PopulationWindow) -> Any:
    """Package one population window as the shared ``NeuralOutput`` contract."""

    try:
        from twin_contracts import NeuralOutput, SourceLabel, Timestamp
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ReplayError("twin_contracts is required for NeuralOutput replay") from exc
    rate_hz = 1000.0 / window.dt_ms
    timestamp = Timestamp(
        t_s=window.end_ms / 1000.0,
        tick=window.index + 1,
        clock="neural_5ms",
        rate_hz=rate_hz,
    )
    return NeuralOutput(
        timestamp=timestamp,
        rates_hz=None,
        spikes=None,
        named_outputs={
            "population_rates_hz": dict(window.rates_hz),
            "population_spike_counts": dict(window.spike_counts),
            "aggregation": {
                "window_ms": float(window.dt_ms),
                "count_semantics": "recorded_sparse_trace_frame_events",
                "exact_window": True,
            },
        },
        source=SourceLabel.CONNECTOME,
    )


@dataclass(frozen=True)
class OBBMetric:
    """Counterfactual signed separating-axis margin for two OBBs.

    ``clearance_m > 0`` means the geometry is separated, ``== 0`` means it
    touches, and ``< 0`` means the geometry overlaps.  The metric is the exact SAT margin for
    the 15 separating-axis candidates (three axes from each box and nine cross
    products); it is not a sphere or AABB approximation.  This is an
    observation-only geometry result, not a physical collision response.
    For a separated pair the smallest positive projected gap is retained,
    while for an overlap the least penetration depth is reported with a
    negative sign.
    """

    would_contact: bool
    clearance_m: float
    axis_m: tuple[float, float, float]
    axis_kind: str

    @property
    def colliding(self) -> bool:
        """Compatibility alias; callers must treat this as counterfactual."""

        return self.would_contact

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "oriented_box_separating_axis_15",
            "would_contact": bool(self.would_contact),
            "counterfactual": True,
            "physical_effect_applied": False,
            "termination_applied": False,
            "clearance_m": float(self.clearance_m),
            "axis_m": list(self.axis_m),
            "axis_kind": self.axis_kind,
            "signed_clearance_definition": "positive=separated, zero=touching, negative=counterfactual_overlap",
        }


def _unit_axis(value: Iterable[Any], name: str) -> tuple[float, float, float]:
    vector = _vec3(value, name)
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1.0e-12:
        raise ReplayError(f"{name} must be non-zero")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def obb_collision_clearance(
    center_a_m: Iterable[Any],
    half_size_a_m: Iterable[Any],
    axes_a: Sequence[Iterable[Any]],
    center_b_m: Iterable[Any],
    half_size_b_m: Iterable[Any],
    axes_b: Sequence[Iterable[Any]],
) -> OBBMetric:
    """Evaluate exact OBB intersection and signed SAT clearance."""

    center_a = _vec3(center_a_m, "center_a_m")
    center_b = _vec3(center_b_m, "center_b_m")
    half_a = _vec3(half_size_a_m, "half_size_a_m")
    half_b = _vec3(half_size_b_m, "half_size_b_m")
    if any(value <= 0.0 for value in half_a + half_b):
        raise ReplayError("OBB half sizes must be positive")
    if len(axes_a) != 3 or len(axes_b) != 3:
        raise ReplayError("each OBB must have three axes")
    basis_a = tuple(_unit_axis(axis, f"axes_a[{index}]") for index, axis in enumerate(axes_a))
    basis_b = tuple(_unit_axis(axis, f"axes_b[{index}]") for index, axis in enumerate(axes_b))
    for label, basis in (("axes_a", basis_a), ("axes_b", basis_b)):
        for first in range(3):
            for second in range(first + 1, 3):
                if abs(_dot(basis[first], basis[second])) > 1.0e-6:
                    raise ReplayError(f"{label} must be an orthonormal basis")
    delta = tuple(center_b[index] - center_a[index] for index in range(3))

    candidates: list[tuple[tuple[float, float, float], str]] = [
        (axis, f"A{index}") for index, axis in enumerate(basis_a)
    ] + [
        (axis, f"B{index}") for index, axis in enumerate(basis_b)
    ]
    for ia, axis_a in enumerate(basis_a):
        for ib, axis_b in enumerate(basis_b):
            cross = _cross(axis_a, axis_b)
            norm = math.sqrt(_dot(cross, cross))
            if norm > 1.0e-10:
                candidates.append((tuple(component / norm for component in cross), f"A{ia}xB{ib}"))

    separating: list[tuple[float, tuple[float, float, float], str]] = []
    penetrating: list[tuple[float, tuple[float, float, float], str]] = []
    for axis, kind in candidates:
        radius_a = sum(half_a[index] * abs(_dot(axis, basis_a[index])) for index in range(3))
        radius_b = sum(half_b[index] * abs(_dot(axis, basis_b[index])) for index in range(3))
        separation = abs(_dot(delta, axis)) - radius_a - radius_b
        oriented_axis = axis
        if _dot(delta, oriented_axis) < 0.0:
            oriented_axis = tuple(-component for component in oriented_axis)
        if separation > 1.0e-12:
            separating.append((separation, oriented_axis, kind))
        else:
            penetrating.append((-separation, oriented_axis, kind))

    if separating:
        clearance, axis, kind = min(separating, key=lambda item: item[0])
        return OBBMetric(False, float(clearance), axis, kind)
    penetration, axis, kind = min(penetrating, key=lambda item: item[0])
    return OBBMetric(True, -float(penetration), axis, kind)


# Explicit aliases make the geometry-only boundary discoverable to callers.
exact_obb_collision_clearance = obb_collision_clearance
sat_obb_metric = obb_collision_clearance


def _rpy_from_quaternion(value: Iterable[Any]) -> tuple[float, float, float]:
    q = _vec4(value, "orientation_quat")
    w, x, y, z = q
    norm = math.sqrt(sum(item * item for item in q))
    if norm <= 1.0e-15:
        return (0.0, 0.0, 0.0)
    w, x, y, z = (item / norm for item in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def _vec4(value: Iterable[Any], name: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise ReplayError(f"{name} must contain four finite values") from exc
    if len(values) != 4:
        raise ReplayError(f"{name} must contain four finite values")
    return values  # type: ignore[return-value]


def _rpy_axes(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    """Body-frame x/y/z axes in world coordinates for Z-Y-X Euler angles."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, sy * cp, -sp),
        (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr),
    )


def _extract(mapping_or_object: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(mapping_or_object, Mapping):
        for name in names:
            if name in mapping_or_object:
                return mapping_or_object[name]
    else:
        for name in names:
            if hasattr(mapping_or_object, name):
                return getattr(mapping_or_object, name)
    return default


def _state_from_body(output: Any, body: Any) -> Any:
    state = _extract(output, ("state", "body_state"))
    if state is None:
        state = _extract(body, ("state", "body_state"))
    if state is None:
        state = output if output is not None else body
    return state


def _pose_from_state(state: Any) -> Any:
    try:
        from virtual_world import Pose
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ReplayError("virtual_world is required for pose/collision telemetry") from exc
    position = _extract(state, ("position_m", "position"), (0.0, 0.0, DEFAULT_BODY_GROUND_Z_M))
    velocity = _extract(state, ("velocity_mps", "velocity"), (0.0, 0.0, 0.0))
    angular = _extract(state, ("angular_velocity_rad_s", "angular_velocity_radps", "angular_velocity"), (0.0, 0.0, 0.0))
    orientation = _extract(state, ("orientation_rad", "orientation_rpy_rad", "attitude_rpy_rad"))
    if orientation is None:
        quaternion = _extract(state, ("orientation_quat", "attitude_quat"), (1.0, 0.0, 0.0, 0.0))
        roll, pitch, yaw = _rpy_from_quaternion(quaternion)
    else:
        roll, pitch, yaw = _vec3(orientation, "orientation_rad")
    return Pose(
        _vec3(position, "position_m"),
        yaw_rad=yaw,
        pitch_rad=pitch,
        roll_rad=roll,
        velocity_mps=_vec3(velocity, "velocity_mps"),
        angular_velocity_radps=_vec3(angular, "angular_velocity_radps"),
    )


def _body_obb_metric(pose: Any, stimulus_state: Any, body_size_m: Sequence[float]) -> OBBMetric:
    body_axes = _rpy_axes(pose.roll_rad, pose.pitch_rad, pose.yaw_rad)
    stimulus_geometry = stimulus_state.geometry
    return obb_collision_clearance(
        pose.position_m,
        tuple(value * 0.5 for value in _vec3(body_size_m, "body_size_m")),
        body_axes,
        stimulus_geometry.center_m,
        tuple(value * 0.5 for value in stimulus_geometry.size_m),
        stimulus_geometry.basis,
    )


def _default_grounded_stimulus(path: Path = DEFAULT_STIMULUS, name: str = "looming_grounded") -> Any:
    try:
        from virtual_world import PhysicalStimulus
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ReplayError("virtual_world is required for the grounded stimulus") from exc
    if path.exists():
        loaded = _load_json(path)
        raw_stimuli = loaded.get("stimuli", {})
        if isinstance(raw_stimuli, Mapping) and name in raw_stimuli:
            return PhysicalStimulus.from_mapping(raw_stimuli[name], seed=loaded.get("seed"))
    # Keep a small explicit fallback so a unit-test checkout without the
    # protocol file remains deterministic.  This is still geometry-only.
    return PhysicalStimulus.from_mapping({
        "id": "palm_looming_grounded",
        "mode": "looming",
        "start_position_m": [1.0, 0.0, DEFAULT_BODY_GROUND_Z_M],
        "speed_mps": 0.5,
        "direction_m": [-1.0, 0.0, 0.0],
        "size_m": [0.09, 0.09, 0.02],
        "contact_radius_m": DEFAULT_BODY_GROUND_Z_M,
        "start_time_s": 0.0,
        "duration_s": 2.0,
        "texture": "dark_palm",
        "albedo": 0.12,
        "renderable": True,
        "collidable": True,
    })


def _make_combined_body(body_ground_z_m: float = DEFAULT_BODY_GROUND_Z_M) -> Any:
    """Create the currently available one-state wing+leg reduced bridge."""

    try:
        import flight_dynamics as dynamics
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ReplayError("flight_dynamics is required for body replay") from exc
    body_type = None
    for name in ("WingLegFlightDynamics", "CombinedFlightDynamics", "FlightDynamicsWithLegs", "WingLegBridge"):
        candidate = getattr(dynamics, name, None)
        if candidate is not None:
            body_type = candidate
            break
    if body_type is None:
        raise ReplayError("combined wing+leg dynamics interface is unavailable")
    config = dynamics.FlightDynamicsConfig(
        ground_contact_enabled=True,
        ground_plane_z_m=float(body_ground_z_m),
        max_dt_s=0.005,
    )
    state = dynamics.FlightState(position_m=(0.0, 0.0, float(body_ground_z_m)))
    jump_config = dynamics.JumpLegConfig(
        ground_contact_enabled=True,
        ground_plane_z_m=float(body_ground_z_m),
        max_dt_s=0.005,
    )
    return body_type(config=config, state=state, jump_config=jump_config, strict_neural=True)


@dataclass(frozen=True)
class ReplayConfig:
    """Run-time options for the diagnostic bridge."""

    window_ms: float = DEFAULT_WINDOW_MS
    body_size_m: tuple[float, float, float] = DEFAULT_BODY_SIZE_M
    body_ground_z_m: float = DEFAULT_BODY_GROUND_Z_M
    render: bool = False
    render_width: int = 32
    render_height: int = 24

    def __post_init__(self) -> None:
        window = _finite(self.window_ms, "window_ms")
        if window <= 0.0:
            raise ReplayError("window_ms must be positive")
        object.__setattr__(self, "window_ms", window)
        size = _vec3(self.body_size_m, "body_size_m")
        if any(item <= 0.0 for item in size):
            raise ReplayError("body_size_m values must be positive")
        object.__setattr__(self, "body_size_m", size)
        object.__setattr__(self, "body_ground_z_m", _finite(self.body_ground_z_m, "body_ground_z_m"))
        if int(self.render_width) <= 0 or int(self.render_height) <= 0:
            raise ReplayError("render resolution must be positive")
        object.__setattr__(self, "render_width", int(self.render_width))
        object.__setattr__(self, "render_height", int(self.render_height))


@dataclass
class ReplayResult:
    """In-memory replay result and its auditable diagnostic artifact."""

    metadata: Mapping[str, Any]
    telemetry: list[Mapping[str, Any]]
    neural_outputs: list[Any]
    frames: list[Any] = field(default_factory=list)

    def to_dict(self, *, include_frames: bool = False) -> dict[str, Any]:
        out = {
            "schema_version": SCHEMA_VERSION,
            "metadata": _json_safe(self.metadata),
            "telemetry": _json_safe(self.telemetry),
            "neural_outputs": [_json_safe(output.to_dict()) for output in self.neural_outputs],
        }
        if include_frames:
            out["frames"] = _json_safe(self.frames)
        return out


class EscapeTraceReplayer:
    """Run a sparse-trace replay through adapter, body, and geometry logging."""

    def __init__(
        self,
        trace: SparseSpikeTrace | str | Path,
        *,
        adapter: Optional[NeuromuscularAdapter] = None,
        map_path: str | Path = DEFAULT_MAP,
        csr_path: str | Path = DEFAULT_CSR,
        population_index: Any = None,
        stimulus: Any = None,
        stimulus_path: str | Path = DEFAULT_STIMULUS,
        stimulus_name: str = "looming_grounded",
        world: Any = None,
        world_path: str | Path = DEFAULT_WORLD,
        body: Any = None,
        config: Optional[ReplayConfig] = None,
    ) -> None:
        self.trace = trace if isinstance(trace, SparseSpikeTrace) else load_sparse_trace(trace)
        self.config = config or ReplayConfig()
        self.adapter = adapter or NeuromuscularAdapter(
            map_path=map_path,
            csr_path=csr_path,
            population_index=population_index,
            strict=True,
        )
        self.selectors = configured_selectors(self.adapter)
        for selector in self.selectors:
            if self.adapter.resolve(selector).size == 0:
                raise ReplayError(f"selector resolves to zero CSR neurons: {selector}")
        if stimulus is None:
            self.stimulus = _default_grounded_stimulus(Path(stimulus_path), stimulus_name)
        else:
            try:
                from virtual_world import PhysicalStimulus
                self.stimulus = (
                    stimulus if isinstance(stimulus, PhysicalStimulus)
                    else PhysicalStimulus.from_mapping(stimulus)
                )
            except (ImportError, TypeError, ValueError) as exc:
                raise ReplayError(f"invalid physical stimulus: {exc}") from exc
        self.world = world
        if self.world is None:
            try:
                from virtual_world import VirtualWorld
                self.world = VirtualWorld.from_json(world_path)
            except (ImportError, OSError, ValueError) as exc:
                raise ReplayError(f"cannot load rendering world {world_path}: {exc}") from exc
        self.body = body if body is not None else _make_combined_body(self.config.body_ground_z_m)
        self._windows: Optional[tuple[PopulationWindow, ...]] = None

    @property
    def windows(self) -> tuple[PopulationWindow, ...]:
        if self._windows is None:
            self._windows = aggregate_population_windows(
                self.trace,
                self.adapter,
                window_ms=self.config.window_ms,
                selectors=self.selectors,
            )
        return self._windows

    def _step_body(self, output: Any) -> Any:
        seconds = self.config.window_ms / 1000.0
        if seconds > 0.005 + 1.0e-12:
            raise ReplayError("combined body bridge max_dt_s is 5 ms; choose a 5 ms replay window")
        for name in ("step_with_jump", "step_wing_leg", "step_combined"):
            method = getattr(self.body, name, None)
            if callable(method):
                return method(output, dt_s=seconds, strict_neural=True)
        method = getattr(self.body, "step", None)
        if callable(method):
            return method(output, dt_s=seconds)
        raise ReplayError("body does not expose a combined step/step_with_jump method")

    def run(self) -> ReplayResult:
        neural_outputs: list[Any] = []
        telemetry: list[Mapping[str, Any]] = []
        frames: list[Any] = []
        for window in self.windows:
            neural = neural_output_from_window(window)
            neural_outputs.append(neural)
            decoded = self.adapter.step(neural, dt_ms=window.dt_ms, timestamp=neural.timestamp)
            body_output = self._step_body(decoded)
            state = _state_from_body(body_output, self.body)
            pose = _pose_from_state(state)
            stimulus_state = self.stimulus.state_at(window.end_ms / 1000.0, pose)
            metric = _body_obb_metric(pose, stimulus_state, self.config.body_size_m)
            rendered = None
            if self.config.render:
                rendered = self.world.render_stereo(
                    pose,
                    width=self.config.render_width,
                    height=self.config.render_height,
                    timestamp_s=window.end_ms / 1000.0,
                    stimulus=self.stimulus,
                )
                frames.append(rendered.to_dict(include_pixels=True))
            jump_force = _extract(body_output, ("jump_force_body_N", "jump_force_N", "jump_force"), (0.0, 0.0, 0.0))
            row = {
                "schema_version": SCHEMA_VERSION,
                "time_s": float(window.end_ms / 1000.0),
                "dt_s": float(window.dt_ms / 1000.0),
                "neural": window.to_dict(),
                "neuromuscular": {
                    "ttmn_activity": float(decoded.channel("ttmn_activity")),
                    "ttmn_motor": float(decoded.channel("jump_motor")),
                    "jump_event_dnp01": float(decoded.channel("jump")),
                    "wing_motor_left": float(decoded.channel("wing_motor_left")),
                    "wing_motor_right": float(decoded.channel("wing_motor_right")),
                    "wing_drive_left": float(decoded.left_wing_power),
                    "wing_drive_right": float(decoded.right_wing_power),
                    "leg_power": float(decoded.leg_power),
                    "leg_steering": float(decoded.leg_steering),
                    "saturated": bool(decoded.saturated),
                    "source": str(decoded.source),
                },
                "jump_force_body_N": list(_vec3(jump_force, "jump_force_body_N")),
                "body_output": _json_safe(body_output.to_dict() if hasattr(body_output, "to_dict") else body_output),
                "pose": _json_safe(pose.to_dict()),
                "obb_counterfactual": {
                    "body_size_m": list(self.config.body_size_m),
                    "stimulus_id": stimulus_state.id,
                    "stimulus_geometry": _json_safe(stimulus_state.geometry),
                    **metric.to_dict(),
                },
                "stimulus": {
                    "use": "rendering_and_counterfactual_geometry_evaluation_only",
                    "control_authority": "none",
                    "forbidden_for_control": True,
                    "physical_effect_applied": False,
                    "termination_applied": False,
                    "state": _json_safe(stimulus_state.to_dict()),
                },
                "replay_policy": {
                    "stimulus_force_applied": False,
                    "stimulus_contact_impulse_applied": False,
                    "stop_on_would_contact": False,
                    "continues_after_would_contact": True,
                },
            }
            telemetry.append(row)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "evidence_class": "diagnostic_trace_replay_bridge",
            "autonomous_closed_loop_evidence": False,
            "autonomous_behavior_claim": False,
            "trace_source": self.trace.source,
            "trace_frame_count": self.trace.frame_count,
            "trace_sparse_event_count": self.trace.spike_event_count,
            "trace_frame_ms": self.trace.trace_frame_ms,
            "window_ms": self.config.window_ms,
            "window_count": len(self.windows),
            "window_semantics": "exact 5 ms aggregate of recorded sparse trace frame events",
            "selector_count": len(self.selectors),
            "selectors": list(self.selectors),
            "adapter": _json_safe(self.adapter.metadata()),
            "body": _json_safe(getattr(self.body, "provenance", {})),
            "stimulus": {
                "id": self.stimulus.id,
                "kind": "grounded_physical_stimulus",
                "source": "looming_grounded",
                "forbidden_for_control": True,
                "control_authority": "none",
                "used_for": ["rendering", "counterfactual_exact_obb_clearance"],
                "physical_effect_applied": False,
                "termination_applied": False,
                "spec": _json_safe(self.stimulus.to_dict()),
            },
            "replay_policy": {
                "stimulus_force_applied": False,
                "stimulus_contact_impulse_applied": False,
                "stop_on_would_contact": False,
                "continues_after_would_contact": True,
                "replayed_duration_ms": float(self.windows[-1].end_ms),
            },
            "control_path": [
                "sparse_lif_trace",
                "exact_5ms_population_aggregate",
                "NeuralOutput",
                "NeuromuscularAdapter",
                "WingLegFlightDynamics",
            ],
            "prohibited_control_inputs": [
                "stimulus_state",
                "collision_time",
                "stimulus_direction",
                "behavior_label",
                "phase_or_goal",
                "autopilot_or_stabilizer",
            ],
        }
        return ReplayResult(metadata, telemetry, neural_outputs, frames)


def replay_escape_trace(trace: SparseSpikeTrace | str | Path, **kwargs: Any) -> ReplayResult:
    """Functional wrapper for :class:`EscapeTraceReplayer`."""

    return EscapeTraceReplayer(trace, **kwargs).run()


def write_replay_artifact(result: ReplayResult, output: str | Path, *, include_frames: bool = False) -> Path:
    """Write ``replay.json`` plus line-oriented telemetry to a file/directory."""

    target = Path(output)
    if target.suffix.lower() == ".json":
        target.parent.mkdir(parents=True, exist_ok=True)
        json_path = target
        directory = target.parent
    else:
        target.mkdir(parents=True, exist_ok=True)
        directory = target
        json_path = target / "replay.json"
    document = result.to_dict(include_frames=include_frames)
    json_path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    telemetry_path = directory / "telemetry.jsonl"
    with telemetry_path.open("w", encoding="utf-8") as stream:
        for row in document["telemetry"]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return json_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="lif_flyb spikes.npz or its artifact directory")
    parser.add_argument("--csr", type=Path, default=DEFAULT_CSR)
    parser.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--stimulus-config", type=Path, default=DEFAULT_STIMULUS)
    parser.add_argument("--stimulus-name", default="looming_grounded")
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--body-size-m", nargs=3, type=float, default=DEFAULT_BODY_SIZE_M)
    parser.add_argument("--body-ground-z-m", type=float, default=DEFAULT_BODY_GROUND_Z_M)
    parser.add_argument("--render", action="store_true", help="retain low-resolution stereo frames for audit only")
    parser.add_argument("--render-width", type=int, default=32)
    parser.add_argument("--render-height", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None, help="JSON file or artifact directory")
    parser.add_argument("--include-frames", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = replay_escape_trace(
        args.trace,
        map_path=args.map_path,
        csr_path=args.csr,
        stimulus_path=args.stimulus_config,
        stimulus_name=args.stimulus_name,
        world_path=args.world,
        config=ReplayConfig(
            window_ms=args.window_ms,
            body_size_m=tuple(args.body_size_m),
            body_ground_z_m=args.body_ground_z_m,
            render=args.render,
            render_width=args.render_width,
            render_height=args.render_height,
        ),
    )
    if args.out is not None:
        path = write_replay_artifact(result, args.out, include_frames=args.include_frames)
        print(f"artifact: {path}")
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "windows": len(result.telemetry),
        "first_ttmn_ms": next((row["time_s"] * 1000.0 for row in result.telemetry if row["neuromuscular"]["ttmn_motor"] > 0.0), None),
        "first_would_contact_ms": next((row["time_s"] * 1000.0 for row in result.telemetry if row["obb_counterfactual"]["would_contact"]), None),
        "replayed_full_trace": bool(result.metadata["replay_policy"]["continues_after_would_contact"]),
        "autonomous_closed_loop_evidence": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
