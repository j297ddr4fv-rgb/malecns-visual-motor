#!/usr/bin/env python3
"""Machine-readable validation for digital-fly flight traces.

The validator is intentionally a small, dependency-light boundary around
flight telemetry.  It accepts JSON/JSONL, NPZ, and CSV traces, normalises the
common schemas emitted by this repository, and reports two different things:

``mechanical_status``
    Whether the supplied numbers satisfy the configured engineering checks.
``validation_status``
    Whether those checks are allowed to support a biological claim.  Synthetic
    fixtures and digital simulations are useful for testing the validator but
    are never promoted to real-fly validation.

``engineering_decision``
    A separate completion gate over takeoff, sustained airborne duration,
    hover, forward displacement, controlled yaw, bounded attitude/speed, gust
    recovery, and landing.  Optional visual/neural diagnostics remain in the
    report but do not make a compact body trajectory fail closed.

This distinction is important: the current repository contains digital-body
and external-body experiments, not a calibrated in-vivo flight recording.
Consequently an existing artifact can have useful PASS/FAIL engineering
numbers while the top-level decision remains ``NOT_VALIDATED``.

The module has no mandatory third-party dependency.  NumPy is imported only
when an NPZ file is actually read.  Public helpers are deliberately plain:

``load_trace(path)``
    Read and normalise one input file.
``validate_trace(trace, ...)``
    Evaluate all checks and return a JSON-ready dictionary.
``validate_file(path, ...)``
    Convenience wrapper for callers and tests.

The CLI prints JSON by default::

    python3 scripts/validate_digital_fly.py artifacts/fly-scale/closed-loop-g0.3/flight.json

That command reports engineering measurements, but its decision is
``NOT_VALIDATED`` because the artifact is a digital simulation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence


__all__ = [
    "EvidenceClass",
    "Trace",
    "ValidationThresholds",
    "classify_evidence",
    "load_trace",
    "validate_trace",
    "validate_file",
    "validate_paths",
    "load_data",
    "validate",
    "metric_takeoff",
    "metric_airborne_duration",
    "metric_hover",
    "metric_forward_displacement",
    "metric_controlled_yaw",
    "metric_bounded_attitude_speed",
    "metric_attitude",
    "metric_visual_steering",
    "metric_gust_recovery",
    "metric_landing",
    "metric_finite_values",
    "metric_collision_free",
    "metric_resource_limits",
    "metric_saccades",
    "metric_latency",
    "metric_closed_loop_source",
    "ENGINEERING_METRIC_FUNCTIONS",
    "ENGINEERING_REQUIRED_METRICS",
]


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"
STATUS_NOT_VALIDATED = "NOT_VALIDATED"


class EvidenceClass:
    """Stable evidence labels used in output and on the command line."""

    REAL_BIOLOGICAL = "real_biological"
    DIGITAL_SIMULATION = "digital_simulation"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    UNKNOWN = "unknown"

    ALL = {
        REAL_BIOLOGICAL,
        DIGITAL_SIMULATION,
        SYNTHETIC_FIXTURE,
        UNKNOWN,
    }


@dataclass(frozen=True)
class ValidationThresholds:
    """Default engineering gates.

    The takeoff/hover/gust values follow ``config/scenarios.json``.  The
    saccade ranges follow the real-fly comparison in ``docs/EXPERIMENT_LOG.md``
    E18.  The other values are deliberately conservative, explicit defaults;
    callers may replace them through ``--thresholds`` or a mapping.
    """

    takeoff_min_altitude_m: float = 0.35
    takeoff_min_gain_m: float = 0.20
    takeoff_sustain_s: float = 0.20
    airborne_min_duration_s: float = 1.00
    airborne_min_altitude_m: float = 0.35
    airborne_max_gap_s: float = 0.50
    hover_max_altitude_error_m: float = 0.035
    hover_max_horizontal_drift_m: float = 0.08
    forward_min_displacement_m: float = 0.20
    controlled_yaw_min_change_rad: float = 0.20
    attitude_rms_deg: float = 10.0
    attitude_max_deg: float = 20.0
    speed_max_mps: float = 5.0
    angular_rate_max_dps: float = 4000.0
    visual_min_correlation: float = 0.20
    visual_min_direction_accuracy: float = 0.70
    visual_min_cue: float = 1.0e-9
    gust_final_speed_mps: float = 0.35
    gust_recovery_s: float = 1.00
    landing_max_vertical_speed_mps: float = 0.25
    landing_max_impact_accel_mps2: float = 50.0
    landing_max_target_distance_m: float = 0.25
    saccade_detection_rate_dps: float = 500.0
    saccade_frequency_min_hz: float = 3.0
    saccade_frequency_max_hz: float = 10.0
    saccade_amplitude_min_deg: float = 15.0
    saccade_amplitude_max_deg: float = 90.0
    saccade_peak_rate_min_dps: float = 1000.0
    saccade_peak_rate_max_dps: float = 4000.0
    latency_max_ms: float = 50.0
    closed_loop_source_ratio_min: float = 0.80

    def __post_init__(self) -> None:
        for field_info in fields(self):
            value = float(getattr(self, field_info.name))
            if not math.isfinite(value):
                raise ValueError(f"threshold {field_info.name} must be finite")
            if "min_" in field_info.name and value < 0:
                raise ValueError(f"threshold {field_info.name} must be non-negative")
            if "max_" in field_info.name and value < 0:
                raise ValueError(f"threshold {field_info.name} must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ValidationThresholds":
        if value is None:
            return cls()
        source: Mapping[str, Any] = value
        # Accept either a direct threshold object or a JSON report containing
        # a ``thresholds`` object.
        nested = source.get("thresholds")
        if isinstance(nested, Mapping):
            source = nested
        kwargs: dict[str, Any] = {}
        names = {field_info.name for field_info in fields(cls)}
        aliases = {
            # Keep the command-line/configuration vocabulary forgiving.  The
            # canonical names are intentionally explicit in the report, but
            # old notebooks commonly use the reverse min/max order.
            "min_airborne_duration_s": "airborne_min_duration_s",
            "airborne_duration_min_s": "airborne_min_duration_s",
            "airborne_min_s": "airborne_min_duration_s",
            "airborne_height_min_m": "airborne_min_altitude_m",
            "min_airborne_altitude_m": "airborne_min_altitude_m",
            "max_airborne_gap_s": "airborne_max_gap_s",
            "forward_displacement_min_m": "forward_min_displacement_m",
            "min_forward_displacement_m": "forward_min_displacement_m",
            "yaw_min_change_rad": "controlled_yaw_min_change_rad",
            "min_yaw_change_rad": "controlled_yaw_min_change_rad",
            "controlled_yaw_min_rad": "controlled_yaw_min_change_rad",
            "max_speed_mps": "speed_max_mps",
            "speed_limit_mps": "speed_max_mps",
            "max_angular_rate_dps": "angular_rate_max_dps",
            "angular_rate_limit_dps": "angular_rate_max_dps",
            "max_landing_target_distance_m": "landing_max_target_distance_m",
            "landing_target_max_distance_m": "landing_max_target_distance_m",
            "pad_support_max_distance_m": "landing_max_target_distance_m",
        }
        for key, raw in source.items():
            name = aliases.get(_normal_key(key), _normal_key(key))
            if name in names:
                kwargs[name] = float(raw)
        return cls(**kwargs)


@dataclass
class Trace:
    """Normalised time-series data and provenance metadata."""

    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    source: Optional[str] = None
    evidence_class: str = EvidenceClass.UNKNOWN
    time_inferred: bool = False
    # Loading deliberately keeps malformed/non-finite input observable.  The
    # scalar helpers below return ``None`` for NaN/Inf so that an individual
    # optional channel can be skipped, but silently dropping such a value from
    # a trajectory would make a partial trace look healthy.  These paths are
    # carried into validation as a hard finite-data gate.
    integrity_issues: tuple[str, ...] = ()

    @property
    def n_samples(self) -> int:
        return len(self.rows)

    @property
    def duration_s(self) -> Optional[float]:
        """Return the longest recorded duration.

        ``run_digital_fly.py`` concatenates scenario-local trajectories.  A
        flat ``max(time)-min(time)`` calculation therefore reports the local
        duration of one scenario even when the file contains several runs,
        and can also span an artificial gap between scenarios.  Keep the
        single-trace behaviour while making the multi-scenario case useful to
        the report.
        """

        groups: dict[str, list[float]] = {}
        for row in self.rows:
            time = _as_float(row.get("time_s"))
            if time is None:
                continue
            scenario = str(row.get("scenario", "")).strip()
            groups.setdefault(scenario, []).append(time)
        durations = [max(values) - min(values) for values in groups.values() if len(values) >= 2]
        return max(durations) if durations else None

    @property
    def scenarios(self) -> tuple[str, ...]:
        """Scenario labels in first-seen order (empty label omitted)."""

        seen: list[str] = []
        for row in self.rows:
            label = str(row.get("scenario", "")).strip()
            if label and label not in seen:
                seen.append(label)
        return tuple(seen)

    def copy(self) -> "Trace":
        return Trace(
            rows=[dict(row) for row in self.rows],
            metadata=dict(self.metadata),
            source=self.source,
            evidence_class=self.evidence_class,
            time_inferred=self.time_inferred,
            integrity_issues=tuple(self.integrity_issues),
        )


# ---------------------------------------------------------------------------
# Input loading and schema normalisation
# ---------------------------------------------------------------------------


def _normal_key(value: Any) -> str:
    """Make a key comparable across camelCase, spaces, and hyphens."""

    text = str(value).strip()
    out: list[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            out.append(char.lower())
        else:
            out.append("_")
    result = "".join(out)
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    # NumPy scalar values expose ``item``; keeping this duck-typed avoids a
    # mandatory NumPy import in the JSON/CSV paths.
    try:
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "active", "contact", "landed"}:
        return True
    if text in {"0", "false", "no", "n", "off", "inactive", "none"}:
        return False
    return None


def _as_sequence(value: Any) -> Optional[list[Any]]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        result = list(value)
    except TypeError:
        return None
    return result


def _finite_sequence(value: Any, length: int) -> Optional[list[float]]:
    sequence = _as_sequence(value)
    if sequence is None or len(sequence) != length:
        return None
    result: list[float] = []
    for item in sequence:
        number = _as_float(item)
        if number is None:
            return None
        result.append(number)
    return result


def _first_value(mapping: Mapping[str, Any], *names: str) -> Any:
    normalised = {_normal_key(key): value for key, value in mapping.items()}
    for name in names:
        key = _normal_key(name)
        if key in normalised:
            return normalised[key]
    return None


def _recursive_find(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    wanted = {_normal_key(name) for name in names}
    stack: list[Any] = [mapping]
    while stack:
        current = stack.pop()
        if not isinstance(current, Mapping):
            continue
        for key, value in current.items():
            if _normal_key(key) in wanted:
                return value
            if isinstance(value, Mapping):
                stack.append(value)
    return None


def _radians(value: Any, *, degrees: bool = False) -> Optional[float]:
    number = _as_float(value)
    if number is None:
        return None
    return math.radians(number) if degrees else number


def _canonicalise_row(raw_row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten common vector fields and add canonical scalar aliases."""

    # ``twin_contracts`` serialises a control sample as a bundle containing
    # ``timestamp``, ``body``, ``visual``, ``motor`` and ``diagnostics``.
    # Flatten those records here so the validator can consume the contract
    # directly without importing it (and without making that module a runtime
    # dependency).  Keep the nested objects in the row as well for audit/debug
    # output.  Command provenance is preferred over a diagnostics/body source:
    # ``StepDiagnostics.source=handbuilt`` does not mean the motor command was
    # hand-built.
    expanded: dict[str, Any] = dict(raw_row)
    nested_command_source: Any = None
    nested_command: Optional[Mapping[str, Any]] = None
    for nested_name in ("motor", "motor_command", "command", "actuator", "actuators"):
        nested = raw_row.get(nested_name)
        if isinstance(nested, Mapping):
            nested_command = nested
            nested_command_source = nested.get("source", nested_command_source)
            for key, value in nested.items():
                expanded.setdefault(str(key), value)
    for nested_name in ("body", "world", "visual", "visual_observation", "sensory", "sensory_frame", "diagnostics", "step_diagnostics", "counters", "metrics", "scenario_metrics"):
        nested = raw_row.get(nested_name)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                expanded.setdefault(str(key), value)
                # Contract metadata/counters are useful as row-level aliases.
                if isinstance(value, Mapping):
                    for child_key, child_value in value.items():
                        expanded.setdefault(str(child_key), child_value)
    if nested_command_source is not None:
        expanded["command_source"] = nested_command_source

    row: dict[str, Any] = dict(expanded)
    normalised = {_normal_key(key): value for key, value in expanded.items()}

    def value(*names: str) -> Any:
        for name in names:
            key = _normal_key(name)
            if key in normalised:
                return normalised[key]
        return None

    def put(name: str, raw: Any) -> None:
        if raw is not None and name not in row:
            row[name] = raw

    def put_numbers(specs: Mapping[str, Sequence[str]]) -> None:
        for name, aliases in specs.items():
            put(name, _as_float(value(*aliases)))

    def put_angular(name: str, aliases: Sequence[str], *, degrees: bool = False) -> None:
        put(name, _radians(value(*aliases), degrees=degrees))

    # Time.  ``frame`` is deliberately not treated as seconds; NPZ metadata
    # supplies dt when an index must be converted.
    timestamp_value = value("timestamp")
    if isinstance(timestamp_value, Mapping):
        timestamp_value = _first_value(timestamp_value, "t_s", "time_s", "timestamp_s", "time")
    time_raw = timestamp_value if timestamp_value is not None else value("time_s", "t_s", "timestamp_s", "time", "t")
    if time_raw is not None:
        put("time_s", _as_float(time_raw))

    # Position and velocity vectors.
    position = value("position_m", "position", "pos", "trajectory", "traj")
    position_values = _finite_sequence(position, 3)
    if position_values is not None:
        put("x_m", position_values[0])
        put("y_m", position_values[1])
        put("z_m", position_values[2])
    put_numbers({
        "x_m": ("x_m", "x", "px", "position_x", "position_x_m", "pos_x"),
        "y_m": ("y_m", "y", "py", "position_y", "position_y_m", "pos_y"),
        "z_m": ("z_m", "z", "pz", "altitude_m", "altitude", "height", "position_z", "position_z_m", "pos_z"),
    })
    ground_distance = _as_float(value("ground_distance_m", "ground_distance"))
    if _as_float(value("z_m", "z", "altitude_m", "altitude")) is None and ground_distance is not None:
        put("z_m", ground_distance)

    velocity = value("velocity_mps", "velocity", "linear_velocity", "vel")
    velocity_values = _finite_sequence(velocity, 3)
    if velocity_values is not None:
        put("vx_mps", velocity_values[0])
        put("vy_mps", velocity_values[1])
        put("vz_mps", velocity_values[2])
    put_numbers({
        "vx_mps": ("vx_mps", "vx", "velocity_x", "velocity_x_mps"),
        "vy_mps": ("vy_mps", "vy", "velocity_y", "velocity_y_mps"),
        "vz_mps": ("vz_mps", "vz", "vertical_speed", "vertical_velocity", "vertical_speed_mps", "velocity_z", "velocity_z_mps"),
    })
    speed = _as_float(value("speed_mps", "speed", "airspeed", "forward_speed", "airspeed_mps"))
    if speed is not None:
        put("speed_mps", abs(speed))

    # Orientation can be scalar Euler values or a scalar-first quaternion.
    quaternion = value("orientation_quat", "attitude_quat", "quaternion", "quat")
    quaternion_values = _finite_sequence(quaternion, 4)
    if quaternion_values is not None:
        put("qw", quaternion_values[0])
        put("qx", quaternion_values[1])
        put("qy", quaternion_values[2])
        put("qz", quaternion_values[3])
    orientation_vector = _finite_sequence(value("orientation_rad", "orientation_rpy_rad", "attitude_rpy_rad", "rpy_rad"), 3)
    if orientation_vector is not None:
        # Contract order is roll, pitch, yaw.
        put("roll_rad", orientation_vector[0])
        put("pitch_rad", orientation_vector[1])
        put("yaw_rad", orientation_vector[2])
    angle_specs = {
        "yaw_rad": ("yaw_rad", "yaw", "heading_rad", "heading", "psi"),
        "pitch_rad": ("pitch_rad", "pitch", "theta"),
        "roll_rad": ("roll_rad", "roll", "phi"),
    }
    for canonical, names in angle_specs.items():
        put_angular(canonical, names)
    for canonical, names in {
        "yaw_rad": ("yaw_deg", "heading_deg"),
        "pitch_rad": ("pitch_deg",),
        "roll_rad": ("roll_deg",),
    }.items():
        put_angular(canonical, names, degrees=True)

    angular_velocity = value("angular_velocity_radps", "angular_velocity_rad_s", "angular_velocity", "omega_vector", "omega_vec")
    angular_values = _finite_sequence(angular_velocity, 3)
    if angular_values is not None:
        put("roll_rate_radps", angular_values[0])
        put("pitch_rate_radps", angular_values[1])
        put("yaw_rate_radps", angular_values[2])
    rate_specs = {
        "yaw_rate_radps": ("yaw_rate_radps", "yaw_rate_rad_s", "yaw_rate", "omega", "angular_velocity_z", "omega_z", "turn_rate"),
        "pitch_rate_radps": ("pitch_rate_radps", "pitch_rate_rad_s", "pitch_rate", "omega_pitch"),
        "roll_rate_radps": ("roll_rate_radps", "roll_rate_rad_s", "roll_rate", "omega_roll"),
    }
    for canonical, names in rate_specs.items():
        put_angular(canonical, names)
    for canonical, names in {
        "yaw_rate_radps": ("yaw_rate_dps", "yaw_rate_deg_s", "omega_dps", "turn_rate_dps"),
        "pitch_rate_radps": ("pitch_rate_dps",),
        "roll_rate_radps": ("roll_rate_dps",),
    }.items():
        put_angular(canonical, names, degrees=True)

    acceleration = value("acceleration_mps2", "acceleration", "linear_acceleration")
    acceleration_values = _finite_sequence(acceleration, 3)
    if acceleration_values is not None:
        put("ax_mps2", acceleration_values[0])
        put("ay_mps2", acceleration_values[1])
        put("az_mps2", acceleration_values[2])
    put_numbers({
        "ax_mps2": ("ax_mps2", "acceleration_x", "accel_x"),
        "ay_mps2": ("ay_mps2", "acceleration_y", "accel_y"),
        "az_mps2": ("az_mps2", "acceleration_z", "accel_z", "vertical_acceleration"),
    })

    # Target and visual aliases.  The target position is kept separate from
    # the measured position so hover/attitude errors do not accidentally use
    # the first sample as a target when one was supplied.
    target_position = value("target_position_m", "target_position", "goal_position_m", "goal_position", "reference_position_m", "desired_position", "setpoint_position")
    target_values = _finite_sequence(target_position, 3)
    if target_values is not None:
        put("target_x_m", target_values[0])
        put("target_y_m", target_values[1])
        put("target_z_m", target_values[2])
    put_numbers({
        "target_x_m": ("target_x_m", "target_x", "desired_x", "setpoint_x"),
        "target_y_m": ("target_y_m", "target_y", "desired_y", "setpoint_y"),
        "target_z_m": ("target_z_m", "target_z", "target_altitude_m", "target_altitude", "desired_z", "desired_altitude", "setpoint_z"),
    })
    for canonical, names in {
        "target_yaw_rad": ("target_yaw_rad", "target_yaw", "goal_yaw_rad", "reference_yaw_rad", "desired_yaw", "heading_target_rad"),
        "target_pitch_rad": ("target_pitch_rad", "target_pitch", "desired_pitch"),
        "target_roll_rad": ("target_roll_rad", "target_roll", "desired_roll"),
    }.items():
        put_angular(canonical, names)
    for canonical, names in {
        "target_yaw_rad": ("target_yaw_deg", "goal_yaw_deg", "desired_yaw_deg"),
        "target_pitch_rad": ("target_pitch_deg", "desired_pitch_deg"),
        "target_roll_rad": ("target_roll_deg", "desired_roll_deg"),
    }.items():
        put_angular(canonical, names, degrees=True)

    target_distance = _as_float(value("target_distance_m", "target_distance", "distance_to_target_m", "distance_to_pad_m"))
    if target_distance is not None:
        put("target_distance_m", abs(target_distance))
    target_on_target = _as_bool(value("target_on_target", "on_target", "pad_support", "supported_by_pad"))
    if target_on_target is not None:
        put("target_on_target", target_on_target)
    target_contact = _as_bool(value("target_contact", "pad_contact", "landing_target_contact"))
    if target_contact is not None:
        put("target_contact", target_contact)

    visual = value(
        "visual_direction", "visual_direction_signal", "stimulus_direction",
        "visual_yaw", "visual_flow", "optic_flow", "flow_yaw", "flow",
        "turn_cue", "visual_cue", "horizontal_motion",
    )
    if visual is None:
        optic_vector = _finite_sequence(value("optic_flow_rad_s", "optic_flow_vector"), 3)
        if optic_vector is not None:
            visual = optic_vector[2]
    if visual is not None:
        parsed = _direction_number(visual)
        if parsed is not None:
            put("visual_direction", parsed)
    confidence = _as_float(value("visual_confidence", "confidence", "cue_confidence"))
    if confidence is not None:
        put("visual_confidence", max(0.0, min(1.0, confidence)))

    # Events, collision, contact, and provenance.
    gust = value("gust_active", "gust", "wind_gust", "gust_flag", "wind_active")
    gust_bool = _as_bool(gust)
    if gust_bool is not None:
        put("gust_active", gust_bool)
    gust_speed = _as_float(value("gust_speed_mps", "wind_speed_mps", "gust_mps"))
    if gust_speed is not None:
        put("gust_speed_mps", abs(gust_speed))
    wind = _finite_sequence(value("wind_mps", "wind", "wind_vector_mps"), 3)
    if wind is not None:
        wind_speed = math.sqrt(sum(component * component for component in wind))
        put("wind_speed_mps", wind_speed)
        # Keep a separate alias for gust detection.  It is not marked active
        # here: a non-zero base wind is not necessarily a configured gust.
        if _as_float(row.get("gust_speed_mps")) is None:
            put("gust_speed_mps", wind_speed)
    contact = _as_bool(value("contact", "ground_contact", "grounded", "landed", "on_ground"))
    if contact is not None:
        put("contact", contact)
    collision = _as_float(value("collision_count", "collisions", "collision"))
    if collision is None and _as_bool(value("collision", "collision_event", "hit")) is True:
        collision = 1.0
    if collision is not None:
        # Preserve negative values for the collision gate to reject; clamping
        # here would turn corrupt telemetry into an apparent zero-collision
        # run.
        put("collision_count", collision)
    impact = _as_float(value("impact_acceleration_mps2", "impact_accel_mps2", "impact_acceleration", "impact_accel"))
    if impact is not None:
        put("impact_accel_mps2", abs(impact))
    if impact is None:
        impact_g = _as_float(value("impact_g", "impact_deceleration_g", "peak_impact_g", "impact_accel_g"))
        if impact_g is not None:
            put("impact_accel_mps2", abs(impact_g) * 9.80665)
    if _as_float(row.get("impact_accel_mps2")) is None:
        peak_impact = _as_float(value("peak_impact_acceleration", "peak_impact_accel", "peak_acceleration", "peak_accel_mps2", "deceleration_mps2"))
        if peak_impact is not None:
            put("impact_accel_mps2", abs(peak_impact))
    force = value("contact_force_n", "contact_force", "normal_force_n", "impact_force_n", "force_n")
    force_values = _finite_sequence(force, 3)
    if force_values is not None:
        put("contact_force_norm_n", math.sqrt(sum(item * item for item in force_values)))
    else:
        force_scalar = _as_float(force)
        if force_scalar is not None:
            put("contact_force_norm_n", abs(force_scalar))
    source = value("command_source", "control_source", "source", "source_label", "provenance", "readout_source", "closed_loop_source")
    if source is not None:
        if isinstance(source, Mapping):
            source = _first_value(source, "command", "control", "motor", "readout", "source", "label")
        source = getattr(source, "value", source)
        if source is not None:
            row["source"] = str(source)
    if nested_command is not None:
        command_values = {_normal_key(key): value for key, value in nested_command.items()}
        command_yaw = None
        for key in ("yaw_rate_rad_s", "yaw_rate_radps", "yaw_rate", "yaw_command", "yaw"):
            if key in command_values:
                command_yaw = _as_float(command_values[key])
                if command_yaw is not None:
                    break
        if command_yaw is None:
            for key in ("yaw_rate_dps", "yaw_rate_deg_s"):
                if key in command_values:
                    command_yaw = _radians(command_values[key], degrees=True)
                    if command_yaw is not None:
                        break
        if command_yaw is not None:
            # Body angular velocity and motor command are separate signals in
            # the contract; the latter is the most direct steering response.
            row["steering_response"] = command_yaw
    safety = _as_bool(value("safety_active", "safety_override", "manual_override"))
    if safety is not None:
        put("safety_active", safety)

    # If a derived speed is absent, velocity supplies it.  This happens for
    # the compact NPZ files emitted by fly_world.py.
    if _as_float(row.get("speed_mps")) is None:
        velocity_components = [_as_float(row.get(name)) for name in ("vx_mps", "vy_mps", "vz_mps")]
        if all(component is not None for component in velocity_components):
            put("speed_mps", math.sqrt(sum(float(component) ** 2 for component in velocity_components)))
    return row


def _direction_number(value: Any) -> Optional[float]:
    number = _as_float(value)
    if number is not None:
        return number
    text = str(value).strip().lower()
    if text in {"left", "l", "ccw", "counterclockwise", "counter_clockwise", "+", "positive"}:
        return 1.0
    if text in {"right", "r", "cw", "clockwise", "-", "negative"}:
        return -1.0
    return None


def _rows_from_mapping(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract records or columnar arrays from a JSON-like mapping."""

    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(value, (list, tuple, Mapping)):
            metadata[str(key)] = value
    for key in (
        "metadata", "meta", "config", "scenario", "acceptance", "world",
        "provenance", "manifest", "resource_limits", "resource_usage",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            metadata[str(key)] = value

    # Some producers wrap columnar telemetry in ``data`` or ``telemetry``.
    # Recurse only into a mapping (not a row-list, handled below), then retain
    # top-level provenance/configuration alongside the nested metadata.
    for nested_key in ("data", "telemetry", "samples", "trajectory"):
        nested_payload = payload.get(nested_key)
        if isinstance(nested_payload, Mapping):
            nested_rows, nested_metadata = _rows_from_mapping(nested_payload)
            if nested_rows:
                merged = dict(metadata)
                merged.update(nested_metadata)
                return nested_rows, merged

    # A few runners store a multi-scenario trajectory as
    # ``{"trajectory": {"hover": [...], "landing": [...]}}`` instead of
    # one flat list.  Accept that shape as well and add the scenario label
    # only when a row did not already carry one.
    for nested_key in ("trajectory", "scenarios", "runs", "scenario_results"):
        nested_payload = payload.get(nested_key)
        if not isinstance(nested_payload, Mapping):
            continue
        grouped_rows: list[dict[str, Any]] = []
        grouped_metadata: dict[str, Any] = {}
        for scenario_name, scenario_payload in nested_payload.items():
            if isinstance(scenario_payload, list) and (not scenario_payload or isinstance(scenario_payload[0], Mapping)):
                candidate_rows = [dict(row) for row in scenario_payload if isinstance(row, Mapping)]
                candidate_metadata: dict[str, Any] = {}
            elif isinstance(scenario_payload, Mapping):
                candidate_rows, candidate_metadata = _rows_from_mapping(scenario_payload)
            else:
                continue
            for row in candidate_rows:
                row.setdefault("scenario", str(scenario_name))
            grouped_rows.extend(candidate_rows)
            grouped_metadata.update(candidate_metadata)
        if grouped_rows:
            merged = dict(metadata)
            merged.update(grouped_metadata)
            return grouped_rows, merged

    records: Any = None
    records_key: Optional[str] = None
    for key in ("records", "rows", "samples", "telemetry", "steps", "diagnostics", "data", "trajectory"):
        candidate = payload.get(key)
        if isinstance(candidate, list) and (not candidate or isinstance(candidate[0], Mapping)):
            records = candidate
            records_key = key
            break
    if records is not None:
        # With row-oriented JSON every other top-level field is metadata,
        # including list-valued windows/targets.  Treating those lists as
        # columns would silently turn ``hover_target_position_m`` into three
        # bogus samples (and was a particularly easy source of false hover
        # failures).
        metadata = {str(key): value for key, value in payload.items() if key != records_key}
        return [dict(row) for row in records if isinstance(row, Mapping)], metadata

    # Columnar JSON: choose the longest one-dimensional list as the row count.
    columns: dict[str, list[Any]] = {}
    lengths: list[int] = []
    for key, value in payload.items():
        if isinstance(value, list) and (not value or not isinstance(value[0], Mapping)):
            columns[str(key)] = value
            lengths.append(len(value))
    n = max(lengths, default=0)
    if n:
        rows: list[dict[str, Any]] = []
        for index in range(n):
            row = {key: values[index] for key, values in columns.items() if index < len(values)}
            rows.append(row)
        return rows, metadata
    return [], metadata


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - exotic NumPy object
            pass
    if isinstance(value, Mapping):
        return {str(key): _decode_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode_json_value(item) for item in value]
    return value


# Dense image/neural payloads are useful provenance but are not flight-state
# channels.  Skipping them keeps validation of a multi-minute artifact cheap;
# all pose, command, diagnostic, resource and metadata scalars still pass
# through the finite-value audit.  Dense arrays have their own producer-side
# contract (and, when present, are represented by compact summary fields).
_FINITE_AUDIT_SKIP_KEYS = {
    "pixels", "left", "right", "render", "rates_hz", "spikes",
    "dense_rates", "dense_spikes", "activity", "responses",
}

_ROW_FINITE_KEYS = {
    "time_s", "t", "timestamp", "frame_time_s", "position_m", "position", "x_m", "y_m", "z_m",
    "velocity_mps", "velocity", "vx_mps", "vy_mps", "vz_mps", "speed_mps", "speed", "vertical_speed",
    "orientation_quat", "orientation_rad", "roll_rad", "pitch_rad", "yaw_rad", "roll", "pitch", "yaw",
    "angular_velocity_rad_s", "angular_velocity_radps", "roll_rate_radps", "pitch_rate_radps", "yaw_rate_radps",
    "acceleration_mps2", "ax_mps2", "ay_mps2", "az_mps2", "target_position_m", "target_distance_m",
    "target_bearing_rad", "target_yaw_rad", "wind_mps", "wind_speed_mps", "gust_speed_mps", "gust_active",
    "visual_direction", "visual_confidence", "collision_count", "collision", "collision_event", "hit",
    "contact", "grounded", "ground_distance_m", "impact_accel_mps2", "impact_acceleration_mps2",
    "contact_force_norm_n", "mass_kg", "body_mass_kg", "forward_displacement_m", "yaw_change_rad",
    "max_speed_mps", "max_angular_rate_dps", "input_finite", "readout_finite", "connectome_finite",
    "body", "world", "motor", "actuator", "actuators", "diagnostics", "step_diagnostics", "counters",
}


def _collect_nonfinite(value: Any, path: str, issues: list[str], *, limit: int = 64) -> None:
    """Collect paths containing NaN/Inf without importing NumPy.

    JSON accepts ``NaN``/``Infinity`` by default and NumPy scalar values can
    also reach this module through NPZ decoding.  A bounded list is enough for
    an auditable report while avoiding an enormous error payload for a corrupt
    dense trace.
    """

    if len(issues) >= limit:
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            if not math.isfinite(float(value)):
                issues.append(path)
        except (TypeError, ValueError, OverflowError):
            issues.append(path)
        return
    # NumPy scalar values expose ``item``.  Do this before sequence handling
    # so a scalar array does not become an opaque object in the audit.
    if hasattr(value, "item") and not isinstance(value, (Mapping, list, tuple)):
        try:
            item = value.item()
        except Exception:  # pragma: no cover - exotic array-like value
            item = None
        if item is not None and item is not value:
            _collect_nonfinite(item, path, issues, limit=limit)
            return
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _normal_key(key_text) in _FINITE_AUDIT_SKIP_KEYS:
                continue
            _collect_nonfinite(item, f"{path}.{key_text}", issues, limit=limit)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_nonfinite(item, f"{path}[{index}]", issues, limit=limit)
        return
    # Avoid a mandatory NumPy dependency while still auditing simple array
    # objects returned by callers that construct Trace in memory.
    if hasattr(value, "tolist"):
        try:
            _collect_nonfinite(value.tolist(), path, issues, limit=limit)
        except Exception:  # pragma: no cover - unsupported array-like value
            pass


def _nonfinite_issues(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    # Audit the compact state/control contract rather than recursively walking
    # every rendered pixel and repeated neural provenance object.  Unknown
    # top-level numeric fields are still included, so adding a future scalar
    # telemetry channel cannot silently bypass the gate.
    for index, row in enumerate(rows):
        for key, item in row.items():
            key_text = str(key)
            if _normal_key(key_text) in _FINITE_AUDIT_SKIP_KEYS:
                continue
            if key_text in _ROW_FINITE_KEYS or not isinstance(item, (Mapping, list, tuple)):
                _collect_nonfinite(item, f"rows[{index}].{key_text}", issues)
            elif _normal_key(key_text) in {"body", "world", "motor", "actuator", "actuators", "diagnostics", "step_diagnostics", "counters"}:
                _collect_nonfinite(item, f"rows[{index}].{key_text}", issues)
            if len(issues) >= 64:
                break
        if len(issues) >= 64:
            break
    _collect_nonfinite(metadata, "metadata", issues)
    return tuple(issues)


def _metadata_from_npz(scalars: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in scalars.items():
        metadata[str(key)] = _decode_json_value(value)
    # ``meta`` is frequently a JSON string in this repository.
    for key in ("meta", "metadata", "config"):
        raw = metadata.get(key)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, Mapping):
                metadata[key] = parsed
    return metadata


def _load_npz(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("reading NPZ requires numpy; install it outside this validator") from exc
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {str(key): loaded[key] for key in loaded.files}
        scalar_values: dict[str, Any] = {}
        column_values: dict[str, Any] = {}
        n = 0
        for key, value in arrays.items():
            if getattr(value, "ndim", 0) == 0:
                scalar_values[key] = value
                continue
            column_values[key] = value
            if getattr(value, "shape", ()):
                n = max(n, int(value.shape[0]))
        metadata = _metadata_from_npz(scalar_values)
        rows: list[dict[str, Any]] = []
        for index in range(n):
            row: dict[str, Any] = {}
            for key, value in column_values.items():
                if not getattr(value, "shape", ()) or index >= int(value.shape[0]):
                    continue
                cell = value[index]
                row[key] = _decode_json_value(cell)
            rows.append(row)
        # A scalar trajectory is unusual but useful in hand-written fixtures.
        if not rows and column_values:
            rows = [{key: _decode_json_value(value) for key, value in column_values.items()}]
        return rows, metadata


def _parse_csv_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    parsed_bool = _as_bool(text)
    if parsed_bool is not None and text.lower() in {"true", "false", "yes", "no", "on", "off"}:
        return parsed_bool
    number = _as_float(text)
    if number is None and text.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        # Preserve the non-finite token as a float so the finite-data metric
        # can report it explicitly instead of treating it as an arbitrary
        # string/missing optional channel.
        try:
            return float(text)
        except ValueError:  # pragma: no cover - guarded token set
            pass
    return number if number is not None else text


def _load_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # Ignore comment lines while retaining simple ``# key: value`` hints.
        content: list[str] = []
        for line in handle:
            if line.lstrip().startswith("#"):
                comment = line.lstrip()[1:].strip()
                if ":" in comment:
                    key, raw = comment.split(":", 1)
                    metadata[key.strip()] = _parse_csv_value(raw)
                continue
            content.append(line)
    if not content:
        return rows, metadata
    reader = csv.DictReader(content)
    for raw in reader:
        rows.append({str(key): _parse_csv_value(value or "") for key, value in raw.items() if key is not None})
    return rows, metadata


def _manifest_trajectory_path(manifest_path: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    """Resolve a trajectory member from a runner manifest when available."""

    outputs = payload.get("outputs")
    candidates: list[Any] = []
    explicit = False
    if isinstance(outputs, Mapping):
        for key in ("trajectory", "trajectory_path"):
            if outputs.get(key) is not None:
                explicit = True
                candidates.append(outputs.get(key))
    validation = payload.get("validation")
    if isinstance(validation, Mapping):
        for key in ("input", "trajectory", "trajectory_path"):
            if validation.get(key) is not None:
                explicit = True
                candidates.append(validation.get(key))
    if not explicit:
        candidates.append("trajectory.json")
    for candidate in candidates:
        if not isinstance(candidate, (str, Path)) or not str(candidate).strip():
            continue
        value = Path(str(candidate))
        if not value.is_absolute():
            value = manifest_path.parent / value
        try:
            if value.is_file() and value.resolve() != manifest_path.resolve():
                return value
        except OSError:
            continue
    return None


def _looks_like_manifest(payload: Mapping[str, Any]) -> bool:
    schema = str(payload.get("schema_version", "")).strip().lower()
    return (
        isinstance(payload.get("outputs"), Mapping)
        and ("trajectory" in payload.get("outputs", {}) or "trajectory_path" in payload.get("outputs", {}))
    ) or schema.startswith("digital-fly-run") or (
        isinstance(payload.get("resource_limits"), Mapping)
        and isinstance(payload.get("scenario_results"), Mapping)
    )


def _attach_manifest_metadata(source_path: Path, metadata: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Attach a sibling/declared manifest for resource and provenance checks."""

    merged = dict(metadata)
    candidates: list[Path] = []
    declared = _recursive_find(metadata, ("manifest_path", "manifest_file", "run_manifest"))
    if isinstance(declared, (str, Path)) and str(declared).strip():
        candidate = Path(str(declared))
        if not candidate.is_absolute():
            candidate = source_path.parent / candidate
        candidates.append(candidate)
    if source_path.name.lower() != "manifest.json":
        candidates.append(source_path.parent / "manifest.json")
    for candidate in candidates:
        try:
            if not candidate.is_file() or candidate.resolve() == source_path.resolve():
                continue
            with candidate.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(manifest, Mapping):
            merged["manifest"] = dict(manifest)
            issues = _nonfinite_issues([], {"manifest": manifest})
            return merged, issues
    return merged, ()


def classify_evidence(
    metadata: Mapping[str, Any] | None = None,
    *,
    source: str | Path | None = None,
    override: str | None = None,
) -> str:
    """Classify provenance conservatively.

    A real biological claim must be explicit.  A path that merely contains
    ``real`` is not enough; callers should put ``evidence_class`` in metadata
    or pass ``--evidence-class real_biological`` after reviewing the data.
    """

    data = metadata or {}
    source_text = str(source).replace("\\", "/").lower() if source is not None else ""
    explicit_hint = _recursive_find(data, ("evidence_class", "evidence", "data_origin", "source_type", "validation_scope"))
    if isinstance(explicit_hint, Mapping):
        explicit_hint = _first_value(explicit_hint, "class", "evidence_class", "type", "kind")
    synthetic_hint = _recursive_find(data, ("synthetic_fixture", "is_fixture", "fixture"))
    explicit_text = _normal_key(explicit_hint) if explicit_hint is not None else ""
    # Synthetic fixtures are a hard boundary.  Even an accidental
    # ``--evidence-class real_biological`` must not turn a test file into a
    # biological claim.
    if explicit_text in {"synthetic", "synthetic_fixture", "fixture", "test_fixture"} or _as_bool(synthetic_hint) is True or "/tests/fixtures/" in source_text or source_text.startswith("tests/fixtures/") or source_text.endswith("fixture.json"):
        return EvidenceClass.SYNTHETIC_FIXTURE
    if override:
        value = str(override).strip().lower()
        if value not in EvidenceClass.ALL:
            raise ValueError(f"unknown evidence class {override!r}; choose one of {sorted(EvidenceClass.ALL)}")
        return value
    if explicit_hint is not None:
        text = _normal_key(explicit_hint)
        if text in {"real", "real_fly", "real_biological", "biological", "in_vivo", "in_vivo_recording", "wetlab"}:
            return EvidenceClass.REAL_BIOLOGICAL
        if text in {"synthetic", "synthetic_fixture", "fixture", "test_fixture"}:
            return EvidenceClass.SYNTHETIC_FIXTURE
        if text in {"digital", "digital_twin", "digital_simulation", "simulation", "simulated"}:
            return EvidenceClass.DIGITAL_SIMULATION
    if source is not None:
        # Repository artifacts are generated by the digital body/world
        # experiments documented in E17--E20.  Treat them as simulation data
        # by default rather than leaving provenance ambiguous.
        if "/artifacts/" in source_text or source_text.startswith("artifacts/") or source_text.endswith("/flight.json") or source_text.endswith("/flight.npz"):
            return EvidenceClass.DIGITAL_SIMULATION
    return EvidenceClass.UNKNOWN


def _fill_time(rows: list[dict[str, Any]], metadata: Mapping[str, Any]) -> bool:
    if not rows:
        return False
    existing = [_as_float(row.get("time_s")) for row in rows]
    if all(value is not None for value in existing):
        return False
    dt = _as_float(_recursive_find(metadata, ("dt_s", "dt", "sample_period_s", "period_s")))
    frame_ms = _as_float(_recursive_find(metadata, ("frame_ms", "sample_ms", "frame_period_ms")))
    if dt is None and frame_ms is not None:
        dt = frame_ms / 1000.0
    if dt is None:
        sample_rate = _as_float(_recursive_find(metadata, ("sample_rate_hz", "frequency_hz", "fps")))
        if sample_rate and sample_rate > 0:
            dt = 1.0 / sample_rate
    if dt is None or not math.isfinite(dt) or dt <= 0:
        return False
    for index, row in enumerate(rows):
        if _as_float(row.get("time_s")) is None:
            row["time_s"] = index * dt
    return True


def load_trace(
    path: str | Path,
    *,
    evidence_class: str | None = None,
) -> Trace:
    """Load and normalise one JSON/JSONL/NPZ/CSV trace."""

    source_path = Path(path)
    if source_path.is_dir():
        # A complete runner artifact is commonly handed to the validator as a
        # directory.  Prefer its trajectory, then use the manifest resolver
        # below for future bundles that expose a different output location.
        trajectory_candidate = source_path / "trajectory.json"
        manifest_candidate = source_path / "manifest.json"
        if trajectory_candidate.is_file():
            source_path = trajectory_candidate
        elif manifest_candidate.is_file():
            source_path = manifest_candidate
        else:
            raise ValueError(f"artifact directory has no trajectory.json or manifest.json: {source_path}")
    # A runner manifest explicitly names trajectory.json as the authoritative
    # playback/validation input.  If a large telemetry sidecar survives an
    # interrupted JSON serialization but that trajectory member is missing,
    # fail closed before reading hundreds of megabytes of telemetry.  Stand-
    # alone JSONL files without a manifest remain supported below.
    if source_path.suffix.lower() in {".jsonl", ".ndjson"}:
        sibling_manifest = source_path.parent / "manifest.json"
        try:
            with sibling_manifest.open("r", encoding="utf-8") as handle:
                sibling_payload = json.load(handle)
        except (OSError, UnicodeError, ValueError):
            sibling_payload = None
        if isinstance(sibling_payload, Mapping) and _looks_like_manifest(sibling_payload):
            outputs = sibling_payload.get("outputs")
            declared = outputs.get("trajectory") if isinstance(outputs, Mapping) else None
            if declared is None:
                validation = sibling_payload.get("validation")
                declared = validation.get("input") if isinstance(validation, Mapping) else None
            if isinstance(declared, (str, Path)) and str(declared).strip():
                expected = Path(str(declared))
                if not expected.is_absolute():
                    expected = source_path.parent / expected
                if not expected.is_file():
                    metadata = {"manifest": dict(sibling_payload), "artifact_integrity": {"trajectory_missing": str(expected)}}
                    evidence = classify_evidence(metadata, source=source_path, override=evidence_class)
                    return Trace(
                        rows=[], metadata=metadata, source=str(source_path), evidence_class=evidence,
                        integrity_issues=(f"declared trajectory file is missing: {expected}",),
                    )
    suffix = source_path.suffix.lower()
    if suffix == ".npz":
        raw_rows, metadata = _load_npz(source_path)
    elif suffix == ".csv":
        raw_rows, metadata = _load_csv(source_path)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        if suffix in {".jsonl", ".ndjson"}:
            raw_rows = []
            metadata = {}
            with source_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    parsed = json.loads(line)
                    if isinstance(parsed, Mapping):
                        raw_rows.append(dict(parsed))
        else:
            with source_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                raw_rows = [dict(row) for row in payload if isinstance(row, Mapping)]
                metadata = {}
            elif isinstance(payload, Mapping):
                # Accepting ``manifest.json`` directly makes the validator
                # usable at the artifact boundary and keeps it compatible
                # with future bundles whose manifest points at a renamed or
                # nested trajectory member.
                if _looks_like_manifest(payload):
                    trajectory_path = _manifest_trajectory_path(source_path, payload)
                    if trajectory_path is not None:
                        nested = load_trace(trajectory_path, evidence_class=evidence_class)
                        nested_metadata = dict(nested.metadata)
                        nested_metadata["manifest"] = dict(payload)
                        nested.metadata = nested_metadata
                        nested.source = str(source_path)
                        nested.evidence_class = classify_evidence(
                            nested_metadata, source=source_path, override=evidence_class or nested.evidence_class
                        )
                        nested.integrity_issues = tuple(
                            dict.fromkeys(
                                [*nested.integrity_issues, *_nonfinite_issues([], {"manifest": payload})]
                            )
                        )
                        return nested
                    metadata = {"manifest": dict(payload), "artifact_integrity": {"trajectory_missing": True}}
                    evidence = classify_evidence(metadata, source=source_path, override=evidence_class)
                    return Trace(
                        rows=[], metadata=metadata, source=str(source_path), evidence_class=evidence,
                        integrity_issues=("manifest does not resolve to an existing trajectory file",),
                    )
                raw_rows, metadata = _rows_from_mapping(payload)
            else:
                raise ValueError(f"JSON root must be an object or list, got {type(payload).__name__}")
    else:
        raise ValueError(f"unsupported trace format {source_path.suffix!r}; use JSON, JSONL, NPZ, or CSV")

    metadata, manifest_issues = _attach_manifest_metadata(source_path, metadata)
    rows = [_canonicalise_row(row) for row in raw_rows]
    inferred = _fill_time(rows, metadata)
    if rows:
        # Stable chronological order is useful when a producer writes chunks
        # out of order.  Rows without time remain in their input order.
        if all(_as_float(row.get("time_s")) is not None for row in rows):
            rows.sort(key=lambda row: float(row["time_s"]))
    evidence = classify_evidence(metadata, source=source_path, override=evidence_class)
    issues = tuple(dict.fromkeys([*_nonfinite_issues(raw_rows, metadata), *manifest_issues]))
    return Trace(
        rows=rows,
        metadata=metadata,
        source=str(source_path),
        evidence_class=evidence,
        time_inferred=inferred,
        integrity_issues=issues,
    )


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _numeric_series(trace: Trace, key: str, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> tuple[list[float], list[float]]:
    source_rows = rows if rows is not None else trace.rows
    values: list[float] = []
    times: list[float] = []
    for index, row in enumerate(source_rows):
        value = _as_float(row.get(key))
        if value is None:
            continue
        time = _as_float(row.get("time_s"))
        if time is None:
            continue
        values.append(value)
        times.append(time)
    return times, values


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _series_with_derived_speed(trace: Trace, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> tuple[list[float], list[float]]:
    source_rows = rows if rows is not None else trace.rows
    times: list[float] = []
    speeds: list[float] = []
    for index, row in enumerate(source_rows):
        time = _as_float(row.get("time_s"))
        speed = _as_float(row.get("speed_mps"))
        if speed is None:
            components = [_as_float(row.get(name)) for name in ("vx_mps", "vy_mps", "vz_mps")]
            if all(component is not None for component in components):
                speed = math.sqrt(sum(float(component) ** 2 for component in components))
        if speed is None and index > 0:
            previous = source_rows[index - 1]
            previous_position = [_as_float(previous.get(name)) for name in ("x_m", "y_m", "z_m")]
            position = [_as_float(row.get(name)) for name in ("x_m", "y_m", "z_m")]
            previous_time = _as_float(previous.get("time_s"))
            if all(item is not None for item in previous_position + position) and time is not None and previous_time is not None and time > previous_time:
                speed = math.sqrt(sum((float(position[component]) - float(previous_position[component])) ** 2 for component in range(3))) / (time - previous_time)
        if time is not None and speed is not None:
            times.append(time)
            speeds.append(abs(speed))
    return times, speeds


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _rms(values: Sequence[float]) -> Optional[float]:
    return math.sqrt(statistics.fmean([value * value for value in values])) if values else None


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 2:
        return None
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    da = [value - mean_a for value in a]
    db = [value - mean_b for value in b]
    denom = math.sqrt(sum(value * value for value in da) * sum(value * value for value in db))
    if denom <= 1.0e-15:
        return None
    return sum(x * y for x, y in zip(da, db)) / denom


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _angle_delta(current: float, previous: float) -> float:
    return _wrap_angle(current - previous)


def _target_meta(trace: Trace, *names: str) -> Any:
    return _recursive_find(trace.metadata, names)


_SCENARIO_ALIASES: dict[str, set[str] | None] = {
    # ``run_digital_fly.py`` uses the labels on the left.  The additional
    # spellings make hand-authored trajectories and older pilots work without
    # changing the runner's output contract.
    "takeoff": {"takeoff", "launch", "vertical_ramp", "vertical-ramp", "ramp"},
    "takeoff_success": {"takeoff", "launch", "vertical_ramp", "vertical-ramp", "ramp"},
    "airborne": None,
    "airborne_duration": None,
    "hover": {"hover", "hover_hold", "hover-hold", "sustained", "sustained_flight", "sustained-flight", "hold"},
    "hover_drift": {"hover", "hover_hold", "hover-hold", "sustained", "sustained_flight", "sustained-flight", "hold"},
    "forward": {"forward", "forward_flight", "forward-flight", "forward_target", "forward-target", "front_flight", "front-flight", "flight", "flight_demo", "flight-demo", "cruise", "translation", "displacement", "forward_ramp", "forward-ramp", "position_ramp", "position-ramp"},
    "forward_displacement": {"forward", "forward_flight", "forward-flight", "forward_target", "forward-target", "front_flight", "front-flight", "flight", "flight_demo", "flight-demo", "cruise", "translation", "displacement", "forward_ramp", "forward-ramp", "position_ramp", "position-ramp"},
    "controlled_yaw": {"yaw", "yaw_step", "yaw-step", "turn", "turning", "yaw_turn", "yaw-turn", "control", "controlled_yaw", "controlled-yaw"},
    "yaw": {"yaw", "yaw_step", "yaw-step", "turn", "turning", "yaw_turn", "yaw-turn", "control", "controlled_yaw", "controlled-yaw"},
    "bounded_attitude_speed": None,
    "attitude": None,
    "attitude_error": None,
    "gust": {"gust", "gust_recovery", "gust-recovery", "wind", "wind_gust", "wind-gust"},
    "gust_recovery": {"gust", "gust_recovery", "gust-recovery", "wind", "wind_gust", "wind-gust"},
    "landing": {"landing", "landing_speed_impact", "landing-speed-impact", "touchdown", "approach", "descent", "descent_landing", "settle"},
    "landing_speed_impact": {"landing", "landing_speed_impact", "landing-speed-impact", "touchdown", "approach", "descent", "descent_landing", "settle"},
    # The remaining diagnostics are run over the supplied trace.  They are
    # intentionally not scenario-gated because they are often recorded in a
    # dedicated short window without a scenario label.
    "visual_steering": None,
    "visual_steering_direction": None,
    "saccades": None,
    "latency": None,
    "closed_loop": None,
}


def _scenario_key(value: Any) -> str:
    return _normal_key(value)


def _scenario_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by scenario while preserving chronological order per run."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get("scenario", "")).strip()
        key = _scenario_key(label) if label else ""
        groups.setdefault(key, []).append(dict(row))
    for values in groups.values():
        if all(_as_float(row.get("time_s")) is not None for row in values):
            values.sort(key=lambda row: float(row["time_s"]))
    return groups


def _has_scenario_labels(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(_row_labels(row) for row in rows)


def _row_labels(row: Mapping[str, Any]) -> set[str]:
    """Return normalized run and mission-phase labels for one telemetry row."""

    labels: set[str] = set()
    for key in ("scenario", "config_scenario", "mission_phase", "phase"):
        value = row.get(key)
        if isinstance(value, Mapping):
            value = _first_value(value, "label", "name", "phase", "kind")
        if value is not None and str(value).strip():
            labels.add(_scenario_key(value))
    return labels


def _phase_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise mission-phase labels without treating labels as PASS evidence."""

    phase_names: list[str] = []
    for row in rows:
        for key in ("mission_phase", "phase"):
            value = row.get(key)
            if isinstance(value, Mapping):
                value = _first_value(value, "label", "name", "phase", "kind")
            if value is not None and str(value).strip() and str(value) not in phase_names:
                phase_names.append(str(value))
    required = ("takeoff", "forward", "turn", "hover", "gust-recovery", "landing")
    present: dict[str, bool] = {}
    for phase in required:
        aliases = _SCENARIO_ALIASES.get(
            "controlled_yaw" if phase == "turn" else "forward_displacement" if phase == "forward" else "gust_recovery" if phase == "gust-recovery" else phase,
            {phase},
        )
        wanted = {_scenario_key(alias) for alias in (aliases or {phase})}
        present[phase] = any(_row_labels(row) & wanted for row in rows)
    return {"labels": phase_names, "required": list(required), "present": present, "missing": [name for name, value in present.items() if not value]}


def _row_window(trace: Trace, name: str, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> list[dict[str, Any]]:
    """Select an explicit ``<name>_window_s`` or scenario-labelled window."""

    source_rows = [dict(row) for row in (rows if rows is not None else trace.rows)]
    window = _target_meta(trace, f"{name}_window_s", f"{name}_window")
    window_values = _finite_sequence(window, 2)
    if window_values is not None:
        start, end = window_values
        window_rows = [row for row in source_rows if (time := _as_float(row.get("time_s"))) is not None and start <= time <= end]
        requested_scenario = _target_meta(trace, f"{name}_scenario", f"{name}_run", f"{name}_phase")
        if requested_scenario is not None:
            requested = _scenario_key(requested_scenario)
            window_rows = [row for row in window_rows if requested in _row_labels(row)]
        elif _has_scenario_labels(window_rows):
            aliases = _SCENARIO_ALIASES.get(name)
            if aliases is not None:
                normalised_aliases = {_scenario_key(alias) for alias in aliases}
                labelled_window = [row for row in window_rows if _row_labels(row) & normalised_aliases]
                if labelled_window:
                    window_rows = labelled_window
        return window_rows
    aliases = _SCENARIO_ALIASES.get(name, {_normal_key(name), _normal_key(name).replace("_", "-")})
    if aliases is None:
        return source_rows
    normalised_aliases = {_scenario_key(alias) for alias in aliases}
    labelled = [row for row in source_rows if _row_labels(row) & normalised_aliases]
    if labelled:
        return labelled
    # A multi-scenario runner trace must not let (say) the hover samples be
    # mistaken for a takeoff simply because a requested scenario is absent.
    # Unlabelled legacy traces retain the old full-trace fallback.
    return [] if _has_scenario_labels(source_rows) else source_rows


def _result(
    name: str,
    *,
    status: str,
    value: Any = None,
    unit: Optional[str] = None,
    threshold: Any = None,
    reason: str = "",
    available: bool = True,
    evidence_class: str = EvidenceClass.UNKNOWN,
    details: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    mechanical = status
    if not available:
        mechanical = STATUS_SKIP
    claim_status = mechanical if evidence_class == EvidenceClass.REAL_BIOLOGICAL else (
        STATUS_SKIP if mechanical == STATUS_SKIP else STATUS_NOT_VALIDATED
    )
    payload: dict[str, Any] = {
        "name": name,
        "status": mechanical,
        "mechanical_status": mechanical,
        "validation_status": claim_status,
        "claim_status": claim_status,
        "status_for_claim": claim_status,
        "validated": bool(evidence_class == EvidenceClass.REAL_BIOLOGICAL and mechanical == STATUS_PASS),
        "value": value,
        "measurements": value,
        "unit": unit,
        "threshold": threshold,
        "reason": reason,
    }
    if details:
        payload["details"] = dict(details)
    return payload


def _skip(name: str, reason: str, evidence_class: str) -> dict[str, Any]:
    return _result(name, status=STATUS_SKIP, reason=reason, available=False, evidence_class=evidence_class)


def _collision_count(trace: Trace, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> float:
    source_rows = rows if rows is not None else trace.rows
    values: list[float] = []
    for row in source_rows:
        count = _as_float(row.get("collision_count"))
        if count is None and isinstance(row.get("collisions"), Sequence) and not isinstance(row.get("collisions"), (str, bytes)):
            count = float(len(row["collisions"]))  # type: ignore[arg-type]
        if count is None and _as_bool(row.get("collision")) is True:
            count = 1.0
        if count is not None:
            values.append(count)
    metadata_count = _target_meta(trace, "collision_count", "collisions", "max_collision_count")
    meta_value = _as_float(metadata_count)
    if meta_value is None and isinstance(metadata_count, Sequence) and not isinstance(metadata_count, (str, bytes)):
        meta_value = float(len(metadata_count))
    return max([0.0, *(values or [0.0]), *( [meta_value] if meta_value is not None else [])])


def _duration(times: Sequence[float]) -> Optional[float]:
    if len(times) < 2:
        return None
    return max(times) - min(times)


def _ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = [dict(row) for row in rows]
    if all(_as_float(row.get("time_s")) is not None for row in ordered):
        ordered.sort(key=lambda row: float(row["time_s"]))
    return ordered


def _airborne_flag(row: Mapping[str, Any], min_altitude_m: float) -> Optional[bool]:
    """Infer airborne state from explicit state first, then compact pose data."""

    altitude = _as_float(row.get("z_m"))
    for key in ("airborne", "is_airborne", "in_flight"):
        value = _as_bool(row.get(key))
        if value is not None:
            # An explicit airborne bit still cannot establish the configured
            # altitude envelope when the pose channel is absent.  Returning
            # ``None`` here makes the enclosing metric SKIP rather than
            # converting a missing altitude into a passing interval.
            if altitude is None:
                return None
            return bool(value and altitude >= min_altitude_m)
    grounded = _as_bool(row.get("grounded"))
    contact = _as_bool(row.get("contact"))
    if grounded is True or contact is True:
        return False
    if grounded is False or contact is False:
        return altitude >= min_altitude_m if altitude is not None else None
    mode = str(row.get("flight_mode", row.get("mode", ""))).strip().lower()
    if mode in {"flight", "airborne", "cruise", "hover", "takeoff", "turn", "recover"}:
        return altitude >= min_altitude_m if altitude is not None else None
    if mode in {"grounded", "landing_contact", "landed", "on_ground"}:
        return False
    return altitude >= min_altitude_m if altitude is not None else None


def _airborne_segments(rows: Sequence[Mapping[str, Any]], thresholds: ValidationThresholds) -> list[dict[str, Any]]:
    """Return contiguous airborne intervals, independently per scenario."""

    segments: list[dict[str, Any]] = []
    for scenario, scenario_rows in _scenario_groups(rows).items():
        ordered = _ordered_rows(scenario_rows)
        current_start: Optional[float] = None
        previous_time: Optional[float] = None
        current_end: Optional[float] = None
        for row in ordered:
            time = _as_float(row.get("time_s"))
            airborne = _airborne_flag(row, thresholds.airborne_min_altitude_m)
            contiguous = (
                airborne is True
                and time is not None
                and (previous_time is None or time - previous_time <= thresholds.airborne_max_gap_s + 1.0e-12)
            )
            if contiguous:
                if current_start is None:
                    current_start = float(time)
                current_end = float(time)
            elif current_start is not None and current_end is not None:
                segments.append({
                    "scenario": scenario,
                    "start_s": current_start,
                    "end_s": current_end,
                    "duration_s": max(0.0, current_end - current_start),
                })
                current_start = None
                current_end = None
            if time is not None:
                previous_time = float(time)
        if current_start is not None and current_end is not None:
            segments.append({
                "scenario": scenario,
                "start_s": current_start,
                "end_s": current_end,
                "duration_s": max(0.0, current_end - current_start),
            })
    return segments


def _orientation_rpy(row: Mapping[str, Any]) -> Optional[tuple[float, float, float]]:
    values = (
        _as_float(row.get("roll_rad")),
        _as_float(row.get("pitch_rad")),
        _as_float(row.get("yaw_rad")),
    )
    if all(value is not None for value in values):
        return tuple(float(value) for value in values)  # type: ignore[return-value]
    return _euler_from_quaternion(row)


def _position_xy(row: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    x = _as_float(row.get("x_m"))
    y = _as_float(row.get("y_m"))
    return (float(x), float(y)) if x is not None and y is not None else None


def _scalar_metadata(trace: Trace, *names: str) -> Optional[float]:
    value = _target_meta(trace, *names)
    return _as_float(value)


def _collision_metadata_value(trace: Trace) -> tuple[bool, Optional[float]]:
    """Return whether an explicit aggregate collision count is available."""

    value = _target_meta(trace, "collision_count", "max_collision_count")
    number = _as_float(value)
    if number is not None:
        return True, number
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        values = [_as_float(item) for item in value]
        if all(item is not None for item in values):
            return True, float(len(values))
    events = _target_meta(trace, "collisions", "collision_events")
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes, Mapping)):
        return True, float(len(events))
    return False, None


def _collision_observation(row: Mapping[str, Any]) -> tuple[bool, Optional[float]]:
    """Read one row's collision signal without treating absence as zero."""

    if "collision_count" in row:
        value = _as_float(row.get("collision_count"))
        return True, value
    collisions = row.get("collisions")
    if isinstance(collisions, Sequence) and not isinstance(collisions, (str, bytes, Mapping)):
        return True, float(len(collisions))
    if "collision" in row or "collision_event" in row or "hit" in row:
        value = _as_bool(row.get("collision", row.get("collision_event", row.get("hit"))))
        return (value is not None), (1.0 if value is True else 0.0 if value is False else None)
    return False, None


def metric_finite_values(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Reject NaN/Inf instead of allowing scalar parsing to hide it."""

    del thresholds
    name = "finite_values"
    if not trace.rows:
        return _skip(name, "no trajectory rows to audit", trace.evidence_class)
    issues = list(trace.integrity_issues)
    issues.extend(_nonfinite_issues(trace.rows, trace.metadata))
    issues = list(dict.fromkeys(issues))
    if issues:
        return _result(
            name,
            status=STATUS_FAIL,
            value={"finite": False, "nonfinite_count": len(issues), "paths": issues[:16]},
            unit="boolean",
            threshold={"nonfinite_count": 0},
            reason="trajectory or artifact metadata contains NaN/Infinity",
            evidence_class=trace.evidence_class,
        )
    missing_time = [index for index, row in enumerate(trace.rows) if _as_float(row.get("time_s")) is None]
    if missing_time:
        return _skip(
            name,
            f"timestamp is missing for {len(missing_time)} trajectory row(s)",
            trace.evidence_class,
        )
    return _result(
        name,
        status=STATUS_PASS,
        value={"finite": True, "nonfinite_count": 0, "n_samples": len(trace.rows)},
        unit="boolean",
        threshold={"nonfinite_count": 0},
        reason="all audited trajectory and artifact values are finite",
        evidence_class=trace.evidence_class,
    )


def metric_collision_free(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Require explicit, complete collision telemetry before claiming zero hits."""

    del thresholds
    name = "collision_free"
    if not trace.rows:
        return _skip(name, "no trajectory rows to audit for collisions", trace.evidence_class)
    observed = 0
    counts: list[float] = []
    invalid_rows: list[int] = []
    for index, row in enumerate(trace.rows):
        present, value = _collision_observation(row)
        if not present:
            continue
        observed += 1
        if value is None or not math.isfinite(float(value)) or value < 0.0:
            invalid_rows.append(index)
        else:
            counts.append(float(value))
    aggregate_present, aggregate = _collision_metadata_value(trace)
    if aggregate_present and aggregate is not None:
        if not math.isfinite(float(aggregate)) or aggregate < 0.0:
            invalid_rows.append(-1)
        else:
            counts.append(float(aggregate))
    if invalid_rows:
        return _result(
            name,
            status=STATUS_FAIL,
            value={"collision_count": None, "invalid_rows": invalid_rows[:16]},
            unit="count",
            threshold={"max_collisions": 0},
            reason="collision telemetry is non-finite or negative",
            evidence_class=trace.evidence_class,
        )
    # An aggregate manifest count cannot fill an uninstrumented row: a later
    # collision would be invisible.  Complete row coverage is required for a
    # zero-collision claim; aggregate data remains useful for traces with no
    # rows only through the separate artifact/trajectory gates.
    if observed < len(trace.rows):
        return _skip(
            name,
            "collision telemetry is missing for one or more trajectory rows",
            trace.evidence_class,
        )
    if not counts:
        return _skip(name, "no collision_count/collision-event telemetry", trace.evidence_class)
    collision_count = max(counts)
    passed = collision_count <= 0.0
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"collision_count": collision_count, "observed_rows": observed, "total_rows": len(trace.rows)},
        unit="count",
        threshold={"max_collisions": 0},
        reason="all recorded collision counts are zero" if passed else "one or more collisions were recorded",
        evidence_class=trace.evidence_class,
    )


def _resource_value(mapping: Mapping[str, Any], names: Sequence[str], *, gib: bool = False) -> Optional[float]:
    """Find a finite resource scalar in a mapping using explicit aliases."""

    value = _first_value(mapping, *names)
    if value is None:
        for nested in mapping.values():
            if isinstance(nested, Mapping):
                value = _first_value(nested, *names)
                if value is not None:
                    break
    number = _as_float(value)
    if number is None:
        return None
    return number * (1024.0 ** 3) if gib else number


def _resource_budget(trace: Trace) -> tuple[dict[str, Any], str]:
    limits_raw = _recursive_find(trace.metadata, ("resource_limits", "resource_limit", "limits"))
    usage_raw = _recursive_find(trace.metadata, ("resource_usage", "resource_use", "usage"))
    limits = limits_raw if isinstance(limits_raw, Mapping) else {}
    usage = usage_raw if isinstance(usage_raw, Mapping) else {}
    host_limit = _resource_value(
        limits,
        ("cgroup_memory_max_bytes", "ram_limit_bytes", "host_memory_max_bytes", "memory_limit_bytes"),
    )
    if host_limit is None:
        host_limit = _resource_value(
            limits,
            ("cgroup_memory_max_gib", "cgroup_memory_max_gb", "ram_limit_gib", "ram_limit_gb", "host_memory_max_gib", "host_memory_max_gb"),
            gib=True,
        )
    vram_limit = _resource_value(
        limits,
        ("torch_allocator_max_vram_bytes", "vram_limit_bytes", "gpu_memory_limit_bytes"),
    )
    if vram_limit is None:
        vram_limit = _resource_value(
            limits,
            ("torch_allocator_max_vram_gib", "torch_allocator_max_vram_gb", "vram_limit_gib", "vram_limit_gb", "gpu_memory_limit_gib", "gpu_memory_limit_gb", "max_vram_gib", "max_vram_gb"),
            gib=True,
        )
    host_peak = _resource_value(
        usage,
        ("peak_host_rss_bytes", "peak_rss_bytes", "peak_host_memory_bytes", "peak_memory_bytes"),
    )
    if host_peak is None:
        host_peak = _resource_value(usage, ("peak_host_rss_gib", "peak_host_rss_gb", "peak_rss_gib", "peak_rss_gb"), gib=True)
    cuda = usage.get("cuda") if isinstance(usage.get("cuda"), Mapping) else {}
    vram_peak = _resource_value(
        usage,
        ("peak_vram_reserved_bytes", "peak_vram_allocated_bytes", "peak_gpu_memory_bytes"),
    )
    if vram_peak is None:
        vram_peak = _resource_value(
            cuda,
            ("peak_reserved_bytes", "peak_allocated_bytes", "peak_memory_bytes"),
        )
    if vram_peak is None:
        vram_peak = _resource_value(usage, ("peak_vram_reserved_gib", "peak_vram_reserved_gb", "peak_vram_allocated_gib", "peak_vram_allocated_gb"), gib=True)
    return {
        "host_limit_bytes": host_limit,
        "host_peak_bytes": host_peak,
        "vram_limit_bytes": vram_limit,
        "vram_peak_bytes": vram_peak,
        "has_limits": bool(limits),
        "has_usage": bool(usage),
    }, "manifest/resource metadata"


def metric_resource_limits(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Check measured host/VRAM peaks against declared artifact limits."""

    del thresholds
    name = "resource_limits"
    observation, source = _resource_budget(trace)
    checks: list[dict[str, Any]] = []
    host_limit = observation["host_limit_bytes"]
    host_peak = observation["host_peak_bytes"]
    vram_limit = observation["vram_limit_bytes"]
    vram_peak = observation["vram_peak_bytes"]
    if host_limit is not None and host_peak is not None:
        checks.append({"resource": "host_rss", "peak_bytes": host_peak, "limit_bytes": host_limit, "passed": host_peak <= host_limit})
    if vram_limit is not None and vram_peak is not None:
        checks.append({"resource": "vram_reserved_or_allocated", "peak_bytes": vram_peak, "limit_bytes": vram_limit, "passed": vram_peak <= vram_limit})
    if not checks:
        return _skip(
            name,
            "resource limits and measured peak usage are required; no complete budget pair was found",
            trace.evidence_class,
        )
    passed = all(bool(check["passed"]) for check in checks)
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"checks": checks, "host_headroom_bytes": host_limit - host_peak if host_limit is not None and host_peak is not None else None, "vram_headroom_bytes": vram_limit - vram_peak if vram_limit is not None and vram_peak is not None else None},
        unit="bytes",
        threshold={"declared_limits": True},
        reason="measured resource peaks remain within declared limits" if passed else "measured resource peak exceeds a declared limit",
        evidence_class=trace.evidence_class,
        details={"source": source, "has_limits": observation["has_limits"], "has_usage": observation["has_usage"]},
    )


# ---------------------------------------------------------------------------
# Individual machine checks
# ---------------------------------------------------------------------------


def metric_takeoff(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "takeoff_success"
    rows = _row_window(trace, "takeoff")
    if rows and any(_as_float(row.get("time_s")) is None or _as_float(row.get("z_m")) is None for row in rows):
        return _skip(name, "takeoff window has missing timestamp or altitude samples", trace.evidence_class)
    times, altitude = _numeric_series(trace, "z_m", rows)
    if len(altitude) < 2:
        return _skip(name, "no altitude/position-z time series", trace.evidence_class)
    start = altitude[0]
    max_altitude = max(altitude)
    final_altitude = altitude[-1]
    gain = max_altitude - start
    crossing = next((index for index, value in enumerate(altitude) if value >= thresholds.takeoff_min_altitude_m), None)
    sustain_s = 0.0
    if crossing is not None:
        crossing_time = times[crossing]
        last_good = crossing
        for index in range(crossing, len(altitude)):
            if altitude[index] + 1.0e-12 < thresholds.takeoff_min_altitude_m:
                break
            if index > crossing and times[index] - times[index - 1] > thresholds.airborne_max_gap_s + 1.0e-12:
                break
            last_good = index
        sustain_s = max(0.0, times[last_good] - crossing_time)
    direct = _target_meta(trace, "takeoff_success", "takeoff_passed")
    direct_bool = _as_bool(direct)
    passed = (
        max_altitude >= thresholds.takeoff_min_altitude_m
        and gain >= thresholds.takeoff_min_gain_m
        and sustain_s >= thresholds.takeoff_sustain_s
        and (direct_bool is not False)
        and _collision_count(trace, rows) <= 0.0
    )
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"start_altitude_m": start, "max_altitude_m": max_altitude, "final_altitude_m": final_altitude, "gain_m": gain, "sustain_s": sustain_s},
        unit="m; s",
        threshold={"min_altitude_m": thresholds.takeoff_min_altitude_m, "min_gain_m": thresholds.takeoff_min_gain_m, "sustain_s": thresholds.takeoff_sustain_s, "max_collisions": 0},
        reason="altitude crossed and was sustained" if passed else "altitude, sustain, or collision gate failed",
        evidence_class=trace.evidence_class,
        details={"n_samples": len(altitude), "window": "takeoff_window_s or full trace"},
    )


def metric_airborne_duration(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Require one continuous, timestamped airborne interval."""

    name = "airborne_duration"
    rows = _row_window(trace, name)
    direct = _scalar_metadata(
        trace,
        "airborne_duration_s",
        "sustained_airborne_duration_s",
        "flight_duration_s",
        "sustained_flight_duration_s",
        "airborne_s",
    )
    if rows and any(_as_float(row.get("time_s")) is None or _as_float(row.get("z_m")) is None for row in rows):
        return _skip(name, "airborne window has missing timestamp or altitude samples", trace.evidence_class)
    segments = _airborne_segments(rows, thresholds) if rows else []
    durations = [float(segment["duration_s"]) for segment in segments]
    duration = max(durations, default=direct)
    if duration is None:
        return _skip(name, "airborne duration requires timestamped flight/contact or altitude samples", trace.evidence_class)
    if direct is None and not any(value > 0.0 for value in durations):
        return _skip(name, "airborne duration requires at least two separated airborne timestamps", trace.evidence_class)
    collisions = _collision_count(trace, rows)
    passed = duration >= thresholds.airborne_min_duration_s and collisions <= 0.0
    scenario_durations: dict[str, float] = {}
    for segment in segments:
        key = str(segment["scenario"] or "trace")
        scenario_durations[key] = max(scenario_durations.get(key, 0.0), float(segment["duration_s"]))
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={
            "duration_s": duration,
            "max_contiguous_duration_s": duration,
            "scenario_durations_s": scenario_durations,
            "segments": segments,
            "collision_count": collisions,
        },
        unit="s",
        threshold={"min_duration_s": thresholds.airborne_min_duration_s, "min_altitude_m": thresholds.airborne_min_altitude_m, "max_gap_s": thresholds.airborne_max_gap_s, "max_collisions": 0},
        reason="airborne interval is sustained without collision" if passed else "airborne duration or collision gate failed",
        evidence_class=trace.evidence_class,
        details={"n_samples": len(rows), "source": "explicit duration or contiguous airborne rows"},
    )


def metric_hover(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "hover_drift"
    rows = _row_window(trace, "hover")
    if rows and any(
        _as_float(row.get(key)) is None
        for row in rows
        for key in ("time_s", "x_m", "y_m", "z_m")
    ):
        return _skip(name, "hover window has missing timestamp or position samples", trace.evidence_class)
    times = [_as_float(row.get("time_s")) for row in rows]
    x = [_as_float(row.get("x_m")) for row in rows]
    y = [_as_float(row.get("y_m")) for row in rows]
    z = [_as_float(row.get("z_m")) for row in rows]
    if not all(any(value is not None for value in series) for series in (x, y, z)):
        return _skip(name, "hover requires x/y/z position and target or initial pose", trace.evidence_class)
    valid = [(float(xv), float(yv), float(zv), float(tv)) for xv, yv, zv, tv in zip(x, y, z, times) if xv is not None and yv is not None and zv is not None and tv is not None]
    if len(valid) < 2:
        return _skip(name, "fewer than two complete hover samples", trace.evidence_class)
    target_position = _target_meta(trace, "hover_target_position_m", "target_position_m", "hover_target", "target")
    target_values = _finite_sequence(target_position, 3)
    target_x = target_values[0] if target_values is not None else _as_float(_target_meta(trace, "hover_target_x_m", "target_x_m", "target_x"))
    target_y = target_values[1] if target_values is not None else _as_float(_target_meta(trace, "hover_target_y_m", "target_y_m", "target_y"))
    target_z = target_values[2] if target_values is not None else _as_float(_target_meta(trace, "hover_target_z_m", "target_z_m", "target_altitude_m", "target_altitude"))
    row_targets = [
        (
            _as_float(row.get("target_x_m")),
            _as_float(row.get("target_y_m")),
            _as_float(row.get("target_z_m")),
        )
        for row in rows
    ]
    complete_row_targets = [tuple(float(value) for value in target) for target in row_targets if all(value is not None for value in target)]
    if complete_row_targets:
        if target_x is None:
            target_x = statistics.median(target[0] for target in complete_row_targets)
        if target_y is None:
            target_y = statistics.median(target[1] for target in complete_row_targets)
        if target_z is None:
            target_z = statistics.median(target[2] for target in complete_row_targets)
    if target_x is None or target_y is None or target_z is None:
        # Runner trajectories intentionally stay compact and do not repeat
        # the configured setpoint in every row.  For a labelled multi-run
        # trace, a median pose is a robust estimate of the hold setpoint and
        # avoids mistaking one integration transient for the target.  Keep
        # the first-sample convention for old unlabelled fixtures.
        if _has_scenario_labels(rows):
            inferred = (
                statistics.median(item[0] for item in valid),
                statistics.median(item[1] for item in valid),
                statistics.median(item[2] for item in valid),
            )
        else:
            inferred = (valid[0][0], valid[0][1], valid[0][2])
        if target_x is None:
            target_x = inferred[0]
        if target_y is None:
            target_y = inferred[1]
        if target_z is None:
            target_z = inferred[2]
    target_inferred = target_position is None and not complete_row_targets
    horizontal = [math.hypot(xv - target_x, yv - target_y) for xv, yv, _, _ in valid]
    altitude_error = [abs(zv - target_z) for _, _, zv, _ in valid]
    max_drift = max(horizontal)
    p95_drift = _percentile(horizontal, 0.95)
    max_altitude_error = max(altitude_error)
    passed = max_drift <= thresholds.hover_max_horizontal_drift_m and max_altitude_error <= thresholds.hover_max_altitude_error_m
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"max_horizontal_drift_m": max_drift, "p95_horizontal_drift_m": p95_drift, "max_altitude_error_m": max_altitude_error, "target_m": [target_x, target_y, target_z]},
        unit="m",
        threshold={"max_horizontal_drift_m": thresholds.hover_max_horizontal_drift_m, "max_altitude_error_m": thresholds.hover_max_altitude_error_m},
        reason="drift and altitude error within hover envelope" if passed else "hover drift or altitude error exceeds envelope",
        evidence_class=trace.evidence_class,
        details={"n_samples": len(valid), "target_inferred": target_inferred, "target_estimator": "median labelled pose" if _has_scenario_labels(rows) else "first sample"},
    )


def metric_forward_displacement(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Measure net translation along the initial/declared forward axis."""

    name = "forward_displacement"
    rows = _row_window(trace, name)
    direct = _scalar_metadata(
        trace,
        "forward_displacement_m",
        "forward_distance_m",
        "net_forward_displacement_m",
        "forward_displacement",
    )
    # A generic hover/landing trace can contain arbitrary horizontal jitter;
    # do not reinterpret that as a forward-flight test.  Unlabelled traces
    # need either an explicit forward metric/axis or a per-row forward command
    # to establish intent.
    if direct is None and not _has_scenario_labels(rows):
        has_forward_intent = _target_meta(trace, "forward_axis_xy", "forward_axis", "forward_heading_rad", "forward_sign") is not None
        has_forward_intent = has_forward_intent or any(
            any(key in row for key in ("forward_axis_xy", "forward_axis", "forward_heading_rad", "forward_command", "forward_speed_mps", "target_forward_m"))
            for row in rows
        )
        if not has_forward_intent:
            return _skip(name, "no forward-flight scenario, axis, or explicit displacement measurement", trace.evidence_class)
    if direct is None and rows and any(
        _as_float(row.get(key)) is None
        for row in rows
        for key in ("time_s", "x_m", "y_m")
    ):
        return _skip(name, "forward window has missing timestamp or horizontal position samples", trace.evidence_class)
    axis_value = _target_meta(trace, "forward_axis_xy", "forward_axis", "heading_axis")
    if axis_value is None:
        for row in rows:
            axis_value = row.get("forward_axis_xy", row.get("forward_axis", row.get("heading_axis")))
            if axis_value is not None:
                break
    axis: Optional[tuple[float, float]] = None
    axis_values = _finite_sequence(axis_value, 2)
    if axis_values is None:
        axis_three = _finite_sequence(axis_value, 3)
        if axis_three is not None:
            axis_values = axis_three[:2]
    if axis_values is not None:
        norm = math.hypot(axis_values[0], axis_values[1])
        if norm > 1.0e-12:
            axis = (axis_values[0] / norm, axis_values[1] / norm)

    positions: list[tuple[float, float, float]] = []
    for row in _ordered_rows(rows):
        position = _position_xy(row)
        time = _as_float(row.get("time_s"))
        if position is not None and time is not None:
            positions.append((float(time), position[0], position[1]))
    if direct is None and len(positions) < 2:
        return _skip(name, "forward displacement requires two timestamped horizontal positions or explicit distance", trace.evidence_class)
    if direct is None:
        first = positions[0]
        last = positions[-1]
        if axis is None:
            first_rpy = _orientation_rpy(_ordered_rows(rows)[0]) if rows else None
            heading = first_rpy[2] if first_rpy is not None else _scalar_metadata(trace, "forward_heading_rad", "initial_heading_rad", "heading_rad")
            if heading is None:
                heading = _as_float(next((row.get("forward_heading_rad") for row in rows if row.get("forward_heading_rad") is not None), None))
            if heading is None:
                heading = 0.0  # contract default: +x is forward
            axis = (math.cos(heading), math.sin(heading))
        displacement = (last[1] - first[1]) * axis[0] + (last[2] - first[2]) * axis[1]
        forward_sign = _scalar_metadata(trace, "forward_sign", "forward_direction_sign")
        if forward_sign is not None and forward_sign < 0:
            displacement = -displacement
        direct = displacement
    passed = direct >= thresholds.forward_min_displacement_m and _collision_count(trace, rows) <= 0.0
    first_position = positions[0][1:] if positions else None
    last_position = positions[-1][1:] if positions else None
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={
            "displacement_m": direct,
            "start_xy_m": list(first_position) if first_position is not None else None,
            "end_xy_m": list(last_position) if last_position is not None else None,
            "forward_axis_xy": list(axis) if axis is not None else None,
        },
        unit="m",
        threshold={"min_displacement_m": thresholds.forward_min_displacement_m, "max_collisions": 0},
        reason="forward displacement reached the required distance" if passed else "forward displacement or collision gate failed",
        evidence_class=trace.evidence_class,
        details={"n_samples": len(positions), "source": "explicit metric or net position projection"},
    )


def metric_controlled_yaw(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Require a commanded/measured yaw step with a non-trivial response."""

    name = "controlled_yaw"
    rows = _row_window(trace, name)
    direct = _scalar_metadata(trace, "controlled_yaw_change_rad", "yaw_change_rad", "net_yaw_change_rad", "controlled_yaw_change", "yaw_change")
    ordered = _ordered_rows(rows)
    if direct is None and not _has_scenario_labels(rows):
        has_yaw_intent = _target_meta(trace, "controlled_yaw_success", "yaw_step_success", "yaw_controlled", "yaw_sign") is not None
        has_yaw_intent = has_yaw_intent or any(
            any(key in row for key in ("yaw_command", "yaw_control", "commanded_yaw_rate_radps", "steering_response"))
            for row in rows
        )
        if not has_yaw_intent:
            return _skip(name, "no yaw-step scenario, command, or explicit yaw-change measurement", trace.evidence_class)
    yaw_series: list[tuple[float, float]] = []
    rate_series: list[tuple[float, float]] = []
    command_series: list[tuple[float, float]] = []
    if direct is None and rows and any(_as_float(row.get("time_s")) is None for row in rows):
        return _skip(name, "yaw window has missing timestamp samples", trace.evidence_class)
    command_keys = (
        "steering_response", "yaw_command", "yaw_control", "yaw_rate_command_radps",
        "commanded_yaw_rate_radps", "commanded_yaw_rate_rad_s", "motor_yaw_rate_radps",
    )
    for row in ordered:
        time = _as_float(row.get("time_s"))
        if time is None:
            continue
        rpy = _orientation_rpy(row)
        if rpy is not None:
            yaw_series.append((float(time), rpy[2]))
        rate = _first_present(row, ("yaw_rate_radps", "yaw_rate_rad_s", "yaw_rate", "turn_rate_radps"))
        if rate is not None:
            rate_series.append((float(time), rate))
        command = _first_present(row, command_keys)
        if command is not None:
            command_series.append((float(time), command))
    actual_change: Optional[float] = direct
    signed_change: Optional[float] = None
    if actual_change is None and len(yaw_series) >= 2:
        accumulated = 0.0
        for (_, previous), (_, current) in zip(yaw_series, yaw_series[1:]):
            accumulated += _angle_delta(current, previous)
        signed_change = accumulated
        actual_change = abs(accumulated)
    if actual_change is None and len(rate_series) >= 2:
        integrated = 0.0
        for (previous_t, previous_rate), (current_t, current_rate) in zip(rate_series, rate_series[1:]):
            dt = current_t - previous_t
            if dt > 0:
                integrated += 0.5 * (previous_rate + current_rate) * dt
        signed_change = integrated
        actual_change = abs(integrated)
    if actual_change is None:
        return _skip(name, "controlled yaw requires measured yaw/rate samples or explicit yaw change", trace.evidence_class)

    commands = [value for _, value in command_series]
    command_present = bool(commands)
    nonzero_commands = [value for value in commands if abs(value) > 1.0e-12]
    command_ok = not command_present or bool(nonzero_commands)
    if command_present and nonzero_commands and signed_change is not None and abs(signed_change) > 1.0e-12:
        # A yaw command is a control test, not merely an arbitrary heading
        # change.  Reject a clear polarity reversal while allowing a direct
        # scalar measurement (which has no signed convention) to stand alone.
        command_sign = math.copysign(1.0, sum(nonzero_commands))
        actual_sign = math.copysign(1.0, signed_change)
        if command_sign != actual_sign:
            command_ok = False
    direct_flag = _as_bool(_target_meta(trace, "controlled_yaw_success", "yaw_step_success", "yaw_controlled"))
    if direct_flag is False:
        command_ok = False
    passed = actual_change >= thresholds.controlled_yaw_min_change_rad and command_ok and _collision_count(trace, rows) <= 0.0
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={
            "yaw_change_rad": actual_change,
            "yaw_change_deg": math.degrees(actual_change),
            "command_samples": len(commands),
            "nonzero_command_samples": len(nonzero_commands),
            "signed_yaw_change_rad": signed_change,
            "collision_count": _collision_count(trace, rows),
        },
        unit="rad",
        threshold={"min_change_rad": thresholds.controlled_yaw_min_change_rad, "max_collisions": 0},
        reason="yaw response reached the commanded step" if passed else "controlled yaw response or collision gate failed",
        evidence_class=trace.evidence_class,
        details={"source": "explicit metric, yaw series, or integrated yaw rate"},
    )


def metric_bounded_attitude_speed(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    """Bound roll/pitch, translational speed, and reported angular rates."""

    name = "bounded_attitude_speed"
    rows = _row_window(trace, name)
    attitude: list[float] = []
    speed_values: list[float] = []
    rate_values: list[float] = []
    n_attitude = 0
    n_speed = 0
    for scenario_rows in _scenario_groups(rows).values():
        ordered = _ordered_rows(scenario_rows)
        previous_position: Optional[tuple[float, float, float]] = None
        previous_time: Optional[float] = None
        for row in ordered:
            rpy = _orientation_rpy(row)
            if rpy is not None:
                # Yaw is deliberately not included in this bound: controlled
                # yaw is a separate completion gate and may legitimately turn
                # farther than a roll/pitch stabilization budget.
                attitude.append(math.degrees(max(abs(rpy[0]), abs(rpy[1]))))
                n_attitude += 1
                rates = [_as_float(row.get(key)) for key in ("roll_rate_radps", "pitch_rate_radps", "yaw_rate_radps")]
                rates = [abs(float(value)) for value in rates if value is not None]
                if rates:
                    rate_values.append(math.degrees(max(rates)))
            speed = _as_float(row.get("speed_mps"))
            if speed is None:
                components = [_as_float(row.get(key)) for key in ("vx_mps", "vy_mps", "vz_mps")]
                if all(value is not None for value in components):
                    speed = math.sqrt(sum(float(value) ** 2 for value in components))
            if speed is None:
                position = row.get("x_m"), row.get("y_m"), row.get("z_m")
                position_values = [_as_float(value) for value in position]
                time = _as_float(row.get("time_s"))
                if all(value is not None for value in position_values) and previous_position is not None and previous_time is not None and time is not None and time > previous_time:
                    speed = math.sqrt(sum((float(position_values[index]) - previous_position[index]) ** 2 for index in range(3))) / (time - previous_time)
                if all(value is not None for value in position_values) and time is not None:
                    previous_position = tuple(float(value) for value in position_values)  # type: ignore[assignment]
                    previous_time = float(time)
            else:
                # Keep the derivative baseline warm even when a compact row
                # supplies speed explicitly; a later row may omit it.
                position_values = [_as_float(row.get(key)) for key in ("x_m", "y_m", "z_m")]
                time = _as_float(row.get("time_s"))
                if all(value is not None for value in position_values) and time is not None:
                    previous_position = tuple(float(value) for value in position_values)  # type: ignore[assignment]
                    previous_time = float(time)
            if speed is not None:
                speed_values.append(abs(float(speed)))
                n_speed += 1
    if rows and any(_as_float(row.get("time_s")) is None for row in rows):
        return _skip(name, "attitude/speed window has missing timestamp samples", trace.evidence_class)
    if rows and (n_attitude < len(rows) or n_speed < len(rows)):
        return _skip(
            name,
            "attitude and speed limits require complete samples for the selected window",
            trace.evidence_class,
        )
    direct_max_attitude = _scalar_metadata(trace, "max_roll_pitch_deg", "max_attitude_deg", "peak_attitude_deg")
    direct_max_speed = _scalar_metadata(trace, "max_speed_mps", "peak_speed_mps", "speed_peak_mps")
    direct_max_rate = _scalar_metadata(trace, "max_angular_rate_dps", "peak_angular_rate_dps")
    if direct_max_attitude is not None and not attitude:
        attitude.append(abs(direct_max_attitude))
    if direct_max_speed is not None and not speed_values:
        speed_values.append(abs(direct_max_speed))
    if direct_max_rate is not None and not rate_values:
        rate_values.append(abs(direct_max_rate))
    if not attitude or not speed_values:
        return _skip(name, "bounded attitude/speed requires complete attitude and speed/velocity samples", trace.evidence_class)
    max_attitude = max(attitude)
    max_speed = max(speed_values)
    max_rate = max(rate_values) if rate_values else None
    passed = (
        max_attitude <= thresholds.attitude_max_deg
        and max_speed <= thresholds.speed_max_mps
        and (max_rate is None or max_rate <= thresholds.angular_rate_max_dps)
        and _collision_count(trace, rows) <= 0.0
    )
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"max_roll_pitch_deg": max_attitude, "max_speed_mps": max_speed, "max_angular_rate_dps": max_rate, "n_attitude_samples": n_attitude, "n_speed_samples": n_speed, "collision_count": _collision_count(trace, rows)},
        unit="deg; m/s; deg/s",
        threshold={"max_roll_pitch_deg": thresholds.attitude_max_deg, "max_speed_mps": thresholds.speed_max_mps, "max_angular_rate_dps": thresholds.angular_rate_max_dps, "max_collisions": 0},
        reason="attitude, speed, and angular-rate bounds are respected" if passed else "attitude, speed, angular-rate, or collision bound failed",
        evidence_class=trace.evidence_class,
    )


def _euler_from_quaternion(row: Mapping[str, Any]) -> Optional[tuple[float, float, float]]:
    values = [_as_float(row.get(key)) for key in ("qw", "qx", "qy", "qz")]
    if not all(value is not None for value in values):
        return None
    w, x, y, z = (float(value) for value in values)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-15:
        return None
    w, x, y, z = (component / norm for component in (w, x, y, z))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def metric_attitude(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "attitude_error"
    rows = _row_window(trace, "attitude")
    values: list[tuple[float, float, float, float]] = []
    inferred_target: Optional[tuple[float, float, float]] = None
    for row in rows:
        time = _as_float(row.get("time_s"))
        if time is None:
            continue
        euler = (
            _as_float(row.get("roll_rad")),
            _as_float(row.get("pitch_rad")),
            _as_float(row.get("yaw_rad")),
        )
        if not all(value is not None for value in euler):
            quaternion_euler = _euler_from_quaternion(row)
            if quaternion_euler is None:
                continue
            euler = quaternion_euler
        targets = (
            _as_float(row.get("target_roll_rad")),
            _as_float(row.get("target_pitch_rad")),
            _as_float(row.get("target_yaw_rad")),
        )
        if not all(value is not None for value in targets):
            meta_target = _target_meta(trace, "target_attitude_rpy_rad", "attitude_target_rpy_rad", "target_rpy_rad")
            meta_values = _finite_sequence(meta_target, 3)
            if meta_values is not None:
                targets = tuple(meta_values)  # type: ignore[assignment]
            else:
                metadata_targets = (
                    _as_float(_target_meta(trace, "target_roll_rad", "target_roll")),
                    _as_float(_target_meta(trace, "target_pitch_rad", "target_pitch")),
                    _as_float(_target_meta(trace, "target_yaw_rad", "target_yaw")),
                )
                targets = tuple(
                    metadata_targets[index] if metadata_targets[index] is not None else None
                    for index in range(3)
                )  # type: ignore[assignment]
        if inferred_target is None:
            inferred_target = (float(euler[0]), float(euler[1]), float(euler[2]))
        # Fill only missing target components from the first observed pose.
        # This preserves an explicit yaw setpoint even when roll/pitch are
        # omitted, and avoids the old error of reusing metric values as angles.
        targets = tuple(
            float(targets[index]) if targets[index] is not None else inferred_target[index]
            for index in range(3)
        )
        error = math.degrees(math.sqrt(sum(_angle_delta(float(angle), float(target)) ** 2 for angle, target in zip(euler, targets))))
        values.append((float(time), error, math.degrees(abs(_angle_delta(float(euler[2]), float(targets[2])))), math.degrees(abs(_angle_delta(float(euler[0]), float(targets[0]))))))
    if len(values) < 2:
        return _skip(name, "no complete Euler/quaternion attitude time series", trace.evidence_class)
    errors = [item[1] for item in values]
    rms = _rms(errors)
    maximum = max(errors)
    p95 = _percentile(errors, 0.95)
    passed = rms is not None and rms <= thresholds.attitude_rms_deg and maximum <= thresholds.attitude_max_deg
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"rms_deg": rms, "max_deg": maximum, "p95_deg": p95},
        unit="deg",
        threshold={"rms_deg": thresholds.attitude_rms_deg, "max_deg": thresholds.attitude_max_deg},
        reason="attitude error within envelope" if passed else "attitude error exceeds envelope",
        evidence_class=trace.evidence_class,
        details={"n_samples": len(values), "target": "explicit setpoint or initial pose"},
    )


def metric_visual_steering(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "visual_steering_direction"
    rows = _row_window(trace, "visual_steering")
    pairs: list[tuple[float, float, float]] = []
    expected_sign = _as_float(_target_meta(trace, "visual_to_yaw_sign", "yaw_sign", "expected_visual_sign")) or 1.0
    for row in rows:
        cue = _as_float(row.get("visual_direction"))
        response = _first_present(row, ("steering_response", "yaw_command", "yaw_change", "heading_change", "yaw_rate_radps", "turn_rate_radps"))
        confidence = _as_float(row.get("visual_confidence"))
        if cue is None or response is None or abs(cue) < thresholds.visual_min_cue:
            continue
        if confidence is not None and confidence <= 0:
            continue
        pairs.append((cue * expected_sign, response, confidence if confidence is not None else 1.0))
    if len(pairs) < 3:
        return _skip(name, "requires at least three non-zero visual-cue/turn-response samples", trace.evidence_class)
    cue_values = [item[0] for item in pairs]
    response_values = [item[1] for item in pairs]
    # Exclude genuinely zero response samples from the directional accuracy
    # denominator, but retain them for correlation where informative.
    response_scale = _percentile([abs(value) for value in response_values], 0.75) or 1.0
    response_epsilon = max(1.0e-12, 0.01 * response_scale)
    directional = [cue * response > 0 for cue, response, _ in pairs if abs(response) >= response_epsilon]
    if not directional:
        return _skip(name, "visual cue has no measurable steering response", trace.evidence_class)
    accuracy = sum(1 for value in directional if value) / len(directional)
    correlation = _correlation(cue_values, response_values)
    passed = correlation is not None and correlation >= thresholds.visual_min_correlation and accuracy >= thresholds.visual_min_direction_accuracy
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"correlation": correlation, "direction_accuracy": accuracy, "n_pairs": len(pairs), "expected_sign": expected_sign},
        unit="1",
        threshold={"min_correlation": thresholds.visual_min_correlation, "min_direction_accuracy": thresholds.visual_min_direction_accuracy},
        reason="visual cue and steering response have the expected polarity" if passed else "visual-to-steering polarity/correlation gate failed",
        evidence_class=trace.evidence_class,
    )


def _gust_bounds(trace: Trace, rows: Sequence[Mapping[str, Any]]) -> Optional[tuple[float, float]]:
    active_times = [float(row["time_s"]) for row in rows if _as_bool(row.get("gust_active")) is True and _as_float(row.get("time_s")) is not None]
    if not active_times:
        # The compact runner contract records ``wind_mps`` but does not need
        # to repeat a dense wind/gust object in every row.  In a row-labelled
        # gust scenario, a clearly non-zero wind interval is sufficient to
        # locate the perturbation; a base wind below this guard is not treated
        # as a gust.
        wind_rows: list[tuple[float, float]] = []
        for row in rows:
            time = _as_float(row.get("time_s"))
            wind_speed = _as_float(row.get("gust_speed_mps", row.get("wind_speed_mps")))
            if wind_speed is not None and time is not None and wind_speed > 0.10:
                wind_rows.append((float(time), abs(float(wind_speed))))
        if wind_rows:
            active_times = [time for time, _ in wind_rows]
    if active_times:
        # Rows are samples, so use the last active timestamp as the end.  The
        # next sample is a better approximation when an explicit dt exists.
        start, end = min(active_times), max(active_times)
        dt = _as_float(_target_meta(trace, "dt_s", "dt"))
        if dt is None and len(active_times) > 1:
            dt = min(b - a for a, b in zip(active_times, active_times[1:]) if b > a)
        return start, end + (dt or 0.0)
    start_raw = _target_meta(trace, "gust_start_s", "gust_start", "gust_at_s", "gust_at", "wind_gust_start_s")
    start = _as_float(start_raw)
    if start is None:
        return None
    # fly_world.py records ``gust_at`` as a frame index.  If it is outside the
    # trace's time range, convert using the frame period.
    all_times = [_as_float(row.get("time_s")) for row in rows]
    all_times = [value for value in all_times if value is not None]
    if all_times and start > max(all_times):
        frame_ms = _as_float(_target_meta(trace, "frame_ms", "sample_ms"))
        dt = _as_float(_target_meta(trace, "dt_s", "dt"))
        period = frame_ms / 1000.0 if frame_ms is not None else dt
        if period is not None:
            start *= period
    duration = _as_float(_target_meta(trace, "gust_duration_s", "gust_duration"))
    if duration is None:
        end_raw = _as_float(_target_meta(trace, "gust_end_s", "gust_end"))
        if end_raw is not None:
            duration = max(0.0, end_raw - start)
    if duration is None:
        duration = 0.25
    return start, start + max(0.0, duration)


def metric_gust_recovery(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "gust_recovery"
    rows = _row_window(trace, "gust_recovery")
    bounds = _gust_bounds(trace, rows)
    if bounds is None:
        return _skip(name, "no gust-active samples or gust onset metadata", trace.evidence_class)
    times, speeds = _series_with_derived_speed(trace, rows)
    if len(speeds) < 2:
        return _skip(name, "gust recovery requires speed/velocity/position samples", trace.evidence_class)
    if rows and (len(times) != len(rows) or any(_as_float(row.get("time_s")) is None for row in rows)):
        return _skip(name, "gust recovery requires complete timestamped speed samples", trace.evidence_class)
    start, end = bounds
    final_window_start = max(end, times[-1] - max(0.2, (times[-1] - times[0]) * 0.2))
    final_values = [speed for time, speed in zip(times, speeds) if time >= final_window_start]
    # The recovery gate is an envelope, so a mean cannot hide one late speed
    # excursion.  Report the worst final-window speed.
    final_speed = max(final_values) if final_values else speeds[-1]
    baseline_values = [speed for time, speed in zip(times, speeds) if time < start]
    baseline_speed = _mean(baseline_values)
    recovery_time: Optional[float] = None
    for index, time in enumerate(times):
        if time < end or speeds[index] > thresholds.gust_final_speed_mps:
            continue
        # Require the following samples to remain under the bound for a short
        # dwell so a single lucky crossing does not count as recovery.
        dwell_end = time + min(0.20, max(0.02, thresholds.gust_recovery_s / 4.0))
        dwell = [speed for later, speed in zip(times[index:], speeds[index:]) if later <= dwell_end]
        if dwell and max(dwell) <= thresholds.gust_final_speed_mps:
            recovery_time = time - end
            break
    collisions = _collision_count(trace, rows)
    passed = final_speed is not None and final_speed <= thresholds.gust_final_speed_mps and recovery_time is not None and recovery_time <= thresholds.gust_recovery_s and collisions <= 0.0
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"gust_start_s": start, "gust_end_s": end, "recovery_time_s": recovery_time, "final_speed_mps": final_speed, "baseline_speed_mps": baseline_speed, "collision_count": collisions},
        unit="s / m/s",
        threshold={"max_recovery_s": thresholds.gust_recovery_s, "max_final_speed_mps": thresholds.gust_final_speed_mps, "max_collisions": 0},
        reason="speed recovered after gust without collision" if passed else "gust recovery time, speed, or collision gate failed",
        evidence_class=trace.evidence_class,
    )


def _contact_index(rows: Sequence[Mapping[str, Any]], ground_z: float = 0.0) -> Optional[int]:
    for index, row in enumerate(rows):
        contact = _as_bool(row.get("contact"))
        if contact is True:
            return index
    z_values = [_as_float(row.get("z_m")) for row in rows]
    for index, value in enumerate(z_values):
        if value is not None and value <= ground_z + 1.0e-3:
            return index
    return None


def _landing_timing_issue(trace: Trace, ground_z: float, landing_start_s: Optional[float]) -> Optional[dict[str, Any]]:
    """Detect an airborne body touching down before the declared landing phase."""

    if landing_start_s is None:
        return None
    saw_airborne = False
    early: Optional[dict[str, Any]] = None
    ordered = _ordered_rows(trace.rows)
    for row in ordered:
        time = _as_float(row.get("time_s"))
        if time is None:
            continue
        airborne = _airborne_flag(row, 0.0)
        if airborne is True:
            saw_airborne = True
        explicit_contact = _as_bool(row.get("contact"))
        if explicit_contact is not None:
            # A body can leave the support plane with effectively zero
            # clearance on its first airborne sample.  The runner's explicit
            # false contact flag is authoritative; re-inferring contact from
            # that zero clearance creates a fictitious early touchdown at
            # liftoff.
            contact = explicit_contact
        else:
            z = _as_float(row.get("z_m"))
            ground_distance = _as_float(row.get("ground_distance_m"))
            contact = (z is not None and z <= ground_z + 1.0e-3) or (
                ground_distance is not None and ground_distance <= 1.0e-3
            )
        if saw_airborne and contact and time < landing_start_s - 1.0e-9:
            early = {
                "time_s": float(time),
                "phase": next(iter(_row_labels(row)), ""),
                "landing_start_s": landing_start_s,
            }
            break
    return early


def metric_landing(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "landing_speed_impact"
    rows = _ordered_rows(_row_window(trace, "landing_speed_impact"))
    if rows and any(_as_float(row.get("time_s")) is None for row in rows):
        return _skip(name, "landing window has missing timestamp samples", trace.evidence_class)
    ground_z = _as_float(_target_meta(trace, "ground_z_m", "ground_z", "ground_altitude")) or 0.0
    landing_phase_times = [
        float(row["time_s"])
        for row in trace.rows
        if _as_float(row.get("time_s")) is not None and _row_labels(row) & {
            _scenario_key(alias) for alias in _SCENARIO_ALIASES["landing_speed_impact"]
        }
    ]
    landing_start_s = min(landing_phase_times) if landing_phase_times else _scalar_metadata(
        trace, "landing_start_s", "landing_phase_start_s", "touchdown_start_s"
    )
    timing_issue = _landing_timing_issue(trace, ground_z, landing_start_s)
    if timing_issue is not None:
        return _result(
            name,
            status=STATUS_FAIL,
            value={"landing_timing": timing_issue},
            unit="s",
            threshold={"contact_not_before_landing_phase": True},
            reason="airborne body contacted the ground before the declared landing phase",
            evidence_class=trace.evidence_class,
        )
    contact_index = _contact_index(rows, ground_z)
    if contact_index is None:
        return _skip(name, "no ground contact/landing sample", trace.evidence_class)
    if contact_index == 0:
        return _skip(name, "landing contact has no preceding airborne sample", trace.evidence_class)
    contact_row = rows[contact_index]
    target_distance = _as_float(contact_row.get("target_distance_m"))
    target_support = _as_bool(contact_row.get("target_on_target", contact_row.get("target_contact")))
    target_observed = any(
        _as_float(row.get("target_distance_m")) is not None
        or _as_bool(row.get("target_on_target", row.get("target_contact"))) is not None
        for row in rows
    )
    if not target_observed:
        return _skip(name, "landing requires pad/target support distance or contact telemetry", trace.evidence_class)
    if target_distance is None and target_support is None:
        return _skip(name, "landing pad/target support telemetry is incomplete", trace.evidence_class)
    # Contact resolution normally clamps the post-impact vertical velocity to
    # zero.  Judge touchdown with the immediately preceding airborne sample;
    # otherwise every hard landing would appear perfectly gentle.
    previous = rows[contact_index - 1]
    previous_contact = _as_bool(previous.get("contact")) is True
    landing_speed = None if previous_contact else _as_float(previous.get("vz_mps"))
    if landing_speed is None:
        landing_speed = _as_float(contact_row.get("vz_mps"))
    if landing_speed is None and contact_index > 0:
        current_z = _as_float(contact_row.get("z_m"))
        previous_z = _as_float(previous.get("z_m"))
        current_t = _as_float(contact_row.get("time_s"))
        previous_t = _as_float(previous.get("time_s"))
        if None not in (current_z, previous_z, current_t, previous_t) and float(current_t) > float(previous_t):
            landing_speed = (float(current_z) - float(previous_z)) / (float(current_t) - float(previous_t))
    if landing_speed is None:
        return _skip(name, "landing contact exists but vertical speed is unavailable", trace.evidence_class)
    landing_speed = abs(landing_speed)
    impact = _as_float(contact_row.get("impact_accel_mps2"))
    if impact is None:
        impact_candidates = [_as_float(row.get("impact_accel_mps2")) for row in rows[max(0, contact_index - 2): min(len(rows), contact_index + 3)]]
        impact_candidates = [value for value in impact_candidates if value is not None]
        if impact_candidates:
            impact = max(abs(value) for value in impact_candidates)
    if impact is None:
        # Some body adapters expose only the vertical acceleration channel.
        # Use its contact neighbourhood as an impact proxy and say so in the
        # details rather than silently treating a missing impact channel as 0.
        acceleration_candidates = [_as_float(row.get("az_mps2")) for row in rows[max(0, contact_index - 2): min(len(rows), contact_index + 3)]]
        acceleration_candidates = [value for value in acceleration_candidates if value is not None]
        if acceleration_candidates:
            impact = max(abs(value) for value in acceleration_candidates)
    if impact is None:
        # Convert a contact-force trace when mass is explicitly supplied.
        force = [_as_float(row.get("contact_force_norm_n")) for row in rows[max(0, contact_index - 2): min(len(rows), contact_index + 3)]]
        force = [value for value in force if value is not None]
        mass = _as_float(_target_meta(trace, "mass_kg", "body_mass_kg"))
        if mass is None:
            mass = _as_float(contact_row.get("mass_kg", contact_row.get("body_mass_kg")))
        if force and mass and mass > 0:
            impact = max(force) / mass
    if impact is None:
        return _skip(name, "landing impact acceleration/force is unavailable", trace.evidence_class)
    collisions = _collision_count(trace, rows)
    pad_ok = (
        target_support is not False
        and (target_distance is None or target_distance <= thresholds.landing_max_target_distance_m)
    )
    passed = landing_speed <= thresholds.landing_max_vertical_speed_mps and abs(impact) <= thresholds.landing_max_impact_accel_mps2 and collisions <= 0.0 and pad_ok
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"contact_index": contact_index, "landing_speed_mps": landing_speed, "impact_accel_mps2": abs(impact), "collision_count": collisions, "target_distance_m": target_distance, "target_on_target": target_support, "pad_supported": pad_ok},
        unit="m/s; m/s^2",
        threshold={"max_vertical_speed_mps": thresholds.landing_max_vertical_speed_mps, "max_impact_accel_mps2": thresholds.landing_max_impact_accel_mps2, "max_target_distance_m": thresholds.landing_max_target_distance_m, "max_collisions": 0},
        reason="landing speed, impact, and pad support are within envelope" if passed else "landing speed, impact, pad support, or collision gate failed",
        evidence_class=trace.evidence_class,
    )


def _angular_rate_series(trace: Trace, rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    times: list[float] = []
    rates: list[float] = []
    for index, row in enumerate(rows):
        time = _as_float(row.get("time_s"))
        rate = _as_float(row.get("yaw_rate_radps"))
        if rate is None and index > 0:
            current = _as_float(row.get("yaw_rad"))
            previous = _as_float(rows[index - 1].get("yaw_rad"))
            previous_time = _as_float(rows[index - 1].get("time_s"))
            if None not in (current, previous, time, previous_time) and float(time) > float(previous_time):
                rate = _angle_delta(float(current), float(previous)) / (float(time) - float(previous_time))
        if rate is not None and time is not None:
            times.append(time)
            rates.append(rate)
    return times, rates


def metric_saccades(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "saccades"
    rows = _row_window(trace, "saccades")
    times, rates = _angular_rate_series(trace, rows)
    if len(rates) < 2:
        return _skip(name, "no yaw-rate or yaw time series", trace.evidence_class)
    threshold = math.radians(thresholds.saccade_detection_rate_dps)
    events: list[tuple[int, int]] = []
    start: Optional[int] = None
    for index, rate in enumerate(rates):
        if abs(rate) >= threshold:
            if start is None:
                start = index
        elif start is not None:
            events.append((start, index - 1))
            start = None
    if start is not None:
        events.append((start, len(rates) - 1))
    # Merge one-sample gaps caused by a threshold crossing at a pulse's peak.
    merged: list[tuple[int, int]] = []
    for event in events:
        if merged and event[0] - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], event[1])
        else:
            merged.append(event)
    events = merged
    dt_values = [b - a for a, b in zip(times, times[1:]) if b > a]
    dt = statistics.median(dt_values) if dt_values else None
    amplitudes: list[float] = []
    for start_index, end_index in events:
        integral = 0.0
        integration_end = min(len(times) - 1, end_index + 1)
        for index in range(start_index, integration_end):
            step = times[index + 1] - times[index]
            if step > 0:
                integral += 0.5 * (abs(rates[index]) + abs(rates[index + 1])) * step
        if start_index == end_index and dt:
            integral = abs(rates[start_index]) * dt
        elif integration_end == start_index and dt:
            integral = abs(rates[start_index]) * dt
        amplitudes.append(math.degrees(integral))
    duration = _duration(times)
    metadata_count = _as_float(_target_meta(trace, "n_saccades", "saccade_count"))
    count = len(events) if events else (int(metadata_count) if metadata_count is not None else 0)
    metadata_frequency = _as_float(_target_meta(trace, "saccade_frequency_hz", "saccade_rate_hz"))
    frequency = metadata_frequency if metadata_frequency is not None else (count / duration if duration and duration > 0 else None)
    median_amplitude = statistics.median(amplitudes) if amplitudes else _as_float(_target_meta(trace, "saccade_amplitude_deg", "median_saccade_amplitude_deg"))
    peak_rate = math.degrees(max(abs(rate) for rate in rates))
    if not events and _as_float(_target_meta(trace, "saccade_peak_rate_dps", "peak_angular_velocity_dps")) is not None:
        peak_rate = _as_float(_target_meta(trace, "saccade_peak_rate_dps", "peak_angular_velocity_dps")) or peak_rate
    passed = (
        frequency is not None and median_amplitude is not None
        and thresholds.saccade_frequency_min_hz <= frequency <= thresholds.saccade_frequency_max_hz
        and thresholds.saccade_amplitude_min_deg <= median_amplitude <= thresholds.saccade_amplitude_max_deg
        and thresholds.saccade_peak_rate_min_dps <= peak_rate <= thresholds.saccade_peak_rate_max_dps
    )
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"count": count, "frequency_hz": frequency, "median_amplitude_deg": median_amplitude, "peak_rate_dps": peak_rate, "detected_events": len(events)},
        unit="Hz; deg; deg/s",
        threshold={"frequency_hz": [thresholds.saccade_frequency_min_hz, thresholds.saccade_frequency_max_hz], "amplitude_deg": [thresholds.saccade_amplitude_min_deg, thresholds.saccade_amplitude_max_deg], "peak_rate_dps": [thresholds.saccade_peak_rate_min_dps, thresholds.saccade_peak_rate_max_dps]},
        reason="saccade timing, amplitude, and peak rate are in the comparison range" if passed else "one or more saccade range gates failed",
        evidence_class=trace.evidence_class,
        details={"detection_threshold_dps": thresholds.saccade_detection_rate_dps, "duration_s": duration},
    )


def _latency_values(trace: Trace, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    for names, scale in (
        (("latency_ms", "end_to_end_latency_ms", "e2e_latency_ms", "latencies_ms"), 1.0),
        (("latency_s", "end_to_end_latency_s", "latencies_s"), 1000.0),
    ):
        direct = _target_meta(trace, *names)
        if isinstance(direct, (list, tuple)):
            return [value * scale for item in direct if (value := _as_float(item)) is not None]
        number = _as_float(direct)
        if number is not None:
            return [number * scale]
    values: list[float] = []
    for row in rows:
        for input_key, output_key in (
            ("input_timestamp_s", "output_timestamp_s"),
            ("stimulus_time_s", "response_time_s"),
            ("visual_onset_s", "motor_onset_s"),
            ("input_time_s", "output_time_s"),
        ):
            start = _as_float(row.get(input_key))
            end = _as_float(row.get(output_key))
            if start is not None and end is not None and end >= start:
                values.append((end - start) * 1000.0)
                break
        if _as_float(row.get("latency_ms")) is not None:
            values.append(float(_as_float(row.get("latency_ms")) or 0.0))
        elif _as_float(row.get("latency_s")) is not None:
            values.append(float(_as_float(row.get("latency_s")) or 0.0) * 1000.0)
    if values:
        return values
    # Infer onset latency from a visual cue and a yaw/steering response.  This
    # is intentionally conservative and is reported as inferred in details.
    cue: list[tuple[float, float]] = []
    response: list[tuple[float, float]] = []
    for row in rows:
        time = _as_float(row.get("time_s"))
        visual = _as_float(row.get("visual_direction"))
        turn = _first_present(row, ("steering_response", "yaw_command", "yaw_change", "heading_change", "yaw_rate_radps"))
        if time is not None and visual is not None:
            cue.append((time, abs(visual)))
        if time is not None and turn is not None:
            response.append((time, abs(turn)))
    if not cue or not response:
        return []
    cue_peak = _percentile([value for _, value in cue], 0.90) or 0.0
    response_peak = _percentile([value for _, value in response], 0.90) or 0.0
    cue_threshold = max(1.0e-9, cue_peak * 0.10)
    response_threshold = max(1.0e-9, response_peak * 0.10)
    cue_onset = next((time for time, value in cue if value >= cue_threshold), None)
    if cue_onset is None:
        return []
    response_onset = next((time for time, value in response if time >= cue_onset and value >= response_threshold), None)
    if response_onset is None:
        return []
    return [max(0.0, (response_onset - cue_onset) * 1000.0)]


def metric_latency(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "end_to_end_latency"
    rows = _row_window(trace, "latency")
    values = _latency_values(trace, rows)
    if not values:
        return _skip(name, "no paired input/output timestamps or onset signals", trace.evidence_class)
    median = statistics.median(values)
    p95 = _percentile(values, 0.95)
    passed = p95 is not None and p95 <= thresholds.latency_max_ms
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"median_ms": median, "p95_ms": p95, "max_ms": max(values), "samples": len(values)},
        unit="ms",
        threshold={"max_p95_ms": thresholds.latency_max_ms},
        reason="end-to-end latency is within control-loop budget" if passed else "end-to-end latency exceeds control-loop budget",
        evidence_class=trace.evidence_class,
        details={"source": "explicit timestamps/metadata or inferred onset"},
    )


def _source_is_closed(source: str) -> bool:
    source = getattr(source, "value", source)
    text = _normal_key(source)
    if not text:
        return False
    if any(token in text for token in ("safety", "manual", "fallback", "open_loop", "openloop", "scripted", "hand_built", "grounded_hold")):
        return False
    # twin_contracts.SourceLabel uses the compact value ``fitted`` for a
    # fitted motor decoder and ``handbuilt`` for external glue/body models.
    # Only the former belongs in the closed-loop numerator.
    return any(token in text for token in ("connectome", "neural", "fitted", "readout", "vision", "sensory", "closed_loop", "brain", "flyvis"))


def metric_closed_loop_source(trace: Trace, thresholds: ValidationThresholds) -> dict[str, Any]:
    name = "closed_loop_source_ratio"
    rows = _row_window(trace, "closed_loop")
    labels: list[str] = []
    for row in rows:
        source = row.get("source")
        if source is None:
            closed = _as_bool(row.get("closed_loop"))
            if closed is not None:
                labels.append("closed_loop" if closed else "unknown")
                continue
            if _as_bool(row.get("safety_active")) is True:
                labels.append("safety_layer")
                continue
            if _as_bool(row.get("connectome_finite")) is True and _as_bool(row.get("readout_finite")) is True:
                labels.append("fitted_readout")
                continue
            continue
        labels.append(str(source))
    if not labels:
        counts = _target_meta(trace, "source_counts", "closed_loop_source_counts")
        if isinstance(counts, Mapping):
            labels = [str(key) for key, raw in counts.items() for _ in range(max(0, int(_as_float(raw) or 0)))]
    if not labels:
        fraction = _as_float(_target_meta(trace, "closed_loop_source_ratio", "closed_loop_fraction", "neural_source_ratio"))
        if fraction is None:
            return _skip(name, "no per-sample source/provenance or source-count metadata", trace.evidence_class)
        ratio = max(0.0, min(1.0, fraction))
        counts = {"closed_loop": ratio, "other": 1.0 - ratio}
        denominator = 1
    else:
        closed_count = sum(1 for source in labels if _source_is_closed(source))
        ratio = closed_count / len(labels)
        counts: dict[str, int] = {}
        for source in labels:
            key = _normal_key(source) or "unknown"
            counts[key] = counts.get(key, 0) + 1
        denominator = len(labels)
    passed = ratio >= thresholds.closed_loop_source_ratio_min
    return _result(
        name,
        status=STATUS_PASS if passed else STATUS_FAIL,
        value={"ratio": ratio, "closed_loop_samples": ratio * denominator, "total_samples": denominator, "source_counts": counts},
        unit="fraction",
        threshold={"min_ratio": thresholds.closed_loop_source_ratio_min},
        reason="control commands are predominantly closed-loop sourced" if passed else "too much control output is open-loop/manual/safety sourced",
        evidence_class=trace.evidence_class,
    )


ENGINEERING_REQUIRED_METRICS = (
    "finite_values",
    "collision_free",
    "takeoff_success",
    "airborne_duration",
    "hover_drift",
    "forward_displacement",
    "controlled_yaw",
    "bounded_attitude_speed",
    "gust_recovery",
    "landing_speed_impact",
    "resource_limits",
)

ENGINEERING_METRIC_FUNCTIONS = {
    "finite_values": metric_finite_values,
    "collision_free": metric_collision_free,
    "takeoff_success": metric_takeoff,
    "airborne_duration": metric_airborne_duration,
    "hover_drift": metric_hover,
    "forward_displacement": metric_forward_displacement,
    "controlled_yaw": metric_controlled_yaw,
    "bounded_attitude_speed": metric_bounded_attitude_speed,
    "gust_recovery": metric_gust_recovery,
    "landing_speed_impact": metric_landing,
    "resource_limits": metric_resource_limits,
}

# Legacy/diagnostic checks remain useful for visual steering, saccades,
# latency, and provenance.  They are intentionally not engineering-completion
# gates: a compact body/control trajectory can prove the flight envelope
# without carrying dense visual or neural arrays.
DIAGNOSTIC_METRIC_FUNCTIONS = {
    "attitude_error": metric_attitude,
    "visual_steering_direction": metric_visual_steering,
    "saccades": metric_saccades,
    "end_to_end_latency": metric_latency,
    "closed_loop_source_ratio": metric_closed_loop_source,
}

METRIC_FUNCTIONS = {**ENGINEERING_METRIC_FUNCTIONS, **DIAGNOSTIC_METRIC_FUNCTIONS}

# Descriptive aliases used by small notebooks and older validation scripts.
# The canonical report keys remain the names in ``METRIC_FUNCTIONS``.
compute_takeoff_success = metric_takeoff
compute_airborne_duration = metric_airborne_duration
compute_hover_drift = metric_hover
compute_forward_displacement = metric_forward_displacement
compute_controlled_yaw = metric_controlled_yaw
compute_bounded_attitude_speed = metric_bounded_attitude_speed
compute_attitude_error = metric_attitude
compute_visual_steering_direction = metric_visual_steering
compute_gust_recovery = metric_gust_recovery
compute_landing_speed_impact = metric_landing
compute_finite_values = metric_finite_values
compute_collision_free = metric_collision_free
compute_resource_limits = metric_resource_limits
compute_saccade_dynamics = metric_saccades
compute_end_to_end_latency = metric_latency
compute_closed_loop_source_ratio = metric_closed_loop_source


# ---------------------------------------------------------------------------
# Report/decision assembly and CLI
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:  # pragma: no cover
            return str(value)
    return value


def validate_trace(
    trace: Trace,
    *,
    thresholds: ValidationThresholds | Mapping[str, Any] | None = None,
    evidence_class: str | None = None,
) -> dict[str, Any]:
    """Run every metric and assemble a conservative go/no-go report."""

    config = thresholds if isinstance(thresholds, ValidationThresholds) else ValidationThresholds.from_mapping(thresholds)
    trace_evidence = trace.evidence_class if trace.evidence_class != EvidenceClass.UNKNOWN else None
    evidence = classify_evidence(trace.metadata, source=trace.source, override=evidence_class or trace_evidence)
    # Metric payloads carry their own claim status.  Evaluate them against the
    # same effective provenance used by the top-level decision so an explicit
    # real-data attestation is reflected consistently (while the classifier's
    # synthetic-fixture hard boundary still wins).
    evaluated_trace = trace.copy()
    evaluated_trace.rows = [_canonicalise_row(row) for row in evaluated_trace.rows]
    evaluated_trace.time_inferred = _fill_time(evaluated_trace.rows, evaluated_trace.metadata) or evaluated_trace.time_inferred
    evaluated_trace.evidence_class = evidence
    metrics: dict[str, dict[str, Any]] = {}
    for name, function in METRIC_FUNCTIONS.items():
        try:
            metrics[name] = function(evaluated_trace, config)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            # A malformed optional field should produce an auditable SKIP, not
            # a false pass or a validator crash.
            metrics[name] = _skip(name, f"metric input malformed: {exc}", evidence)
    engineering_statuses = [metrics[name]["mechanical_status"] for name in ENGINEERING_REQUIRED_METRICS]
    if any(status == STATUS_FAIL for status in engineering_statuses):
        engineering_decision = "NO-GO"
    elif all(status == STATUS_PASS for status in engineering_statuses):
        engineering_decision = "GO"
    else:
        engineering_decision = STATUS_NOT_VALIDATED
    # ``mechanical_decision`` is retained as a compatibility alias, but now
    # means the explicit engineering completion gate rather than the optional
    # visual/neural diagnostics.
    mechanical_decision = engineering_decision
    real_evidence = evidence == EvidenceClass.REAL_BIOLOGICAL
    if not real_evidence:
        decision = STATUS_NOT_VALIDATED
        validation_status = STATUS_NOT_VALIDATED
        reason = "real biological flight data is not attested; measurements are toolchain-only"
        if evaluated_trace.integrity_issues:
            reason += "; artifact integrity is incomplete: " + "; ".join(evaluated_trace.integrity_issues[:3])
    elif engineering_decision == "NO-GO":
        decision = "NO-GO"
        validation_status = STATUS_FAIL
        reason = "at least one required engineering flight metric failed"
    elif engineering_decision != "GO":
        decision = STATUS_NOT_VALIDATED
        validation_status = STATUS_NOT_VALIDATED
        reason = "one or more required engineering flight metrics are missing"
    else:
        decision = "GO"
        validation_status = STATUS_PASS
        reason = "all required engineering flight metrics pass on explicitly attested real biological data"
    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario, rows in _scenario_groups(evaluated_trace.rows).items():
        times = [_as_float(row.get("time_s")) for row in rows]
        times = [float(value) for value in times if value is not None]
        scenario_results[scenario or "trace"] = {
            "sample_count": len(rows),
            "start_s": min(times) if times else None,
            "end_s": max(times) if times else None,
            "duration_s": (max(times) - min(times)) if len(times) >= 2 else None,
        }
    scenario_names = list(evaluated_trace.scenarios)
    metadata_scenarios = _target_meta(evaluated_trace, "scenario_names", "scenarios")
    if not scenario_names and isinstance(metadata_scenarios, (list, tuple)):
        scenario_names = [str(value) for value in metadata_scenarios]
    phase_coverage = _phase_coverage(evaluated_trace.rows)
    report = {
        "schema_version": "flight-validation-1",
        "decision": decision,
        "status": decision,
        "decision_code": decision.replace("-", "_"),
        "go_no_go": decision,
        "overall_status": validation_status,
        "validation_status": validation_status,
        "biological_decision": decision,
        # Keep the biological boundary obvious to simple consumers that do
        # not inspect the engineering sub-report.  Digital/synthetic/unknown
        # traces are always NOT_VALIDATED, even when every mechanical gate is
        # green.
        "biological_status": validation_status,
        "biological_validation": decision,
        "biological_validated": bool(real_evidence and decision == "GO"),
        "engineering_decision": engineering_decision,
        "engineering_status": engineering_decision,
        "engineering_decision_code": engineering_decision.replace("-", "_"),
        "engineering_go": engineering_decision == "GO",
        "mechanical_decision": mechanical_decision,
        "reason": reason,
        "evidence": {
            "class": evidence,
            "real_biological": real_evidence,
            "synthetic_fixture": evidence == EvidenceClass.SYNTHETIC_FIXTURE,
            "source": trace.source,
            "time_inferred": evaluated_trace.time_inferred,
            "n_samples": evaluated_trace.n_samples,
            "duration_s": evaluated_trace.duration_s,
            "integrity_issues": list(evaluated_trace.integrity_issues),
        },
        "thresholds": config.as_dict(),
        "metrics": metrics,
        "engineering": {
            "decision": engineering_decision,
            "status": engineering_decision,
            "required_metrics": list(ENGINEERING_REQUIRED_METRICS),
            "optional_diagnostics": list(DIAGNOSTIC_METRIC_FUNCTIONS),
            "passed_metrics": [name for name in ENGINEERING_REQUIRED_METRICS if metrics[name]["mechanical_status"] == STATUS_PASS],
            "failed_metrics": [name for name in ENGINEERING_REQUIRED_METRICS if metrics[name]["mechanical_status"] == STATUS_FAIL],
            "skipped_metrics": [name for name in ENGINEERING_REQUIRED_METRICS if metrics[name]["mechanical_status"] == STATUS_SKIP],
            "diagnostic_failures": [name for name in DIAGNOSTIC_METRIC_FUNCTIONS if metrics[name]["mechanical_status"] == STATUS_FAIL],
            "phase_coverage": phase_coverage,
        },
        "scenario_names": scenario_names,
        "phase_coverage": phase_coverage,
        "scenario_results": scenario_results,
        "required_metrics": list(ENGINEERING_REQUIRED_METRICS),
        "required_engineering_metrics": list(ENGINEERING_REQUIRED_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRIC_FUNCTIONS),
        "limitations": [
            "Digital simulations and synthetic fixtures cannot establish biological validity.",
            "A GO decision requires explicit real_biological evidence metadata or CLI attestation.",
            "Engineering GO requires finite values, complete collision telemetry, the flight-envelope metrics, and a measured resource budget; optional visual/neural diagnostics remain visible but do not gate compact-body validation.",
            "Missing channels are SKIP/NOT_VALIDATED; they are never imputed as biological success.",
        ],
    }
    # Compatibility aliases make reports convenient for notebooks while the
    # canonical names remain stable.
    report["metric_aliases"] = {
        "takeoff": "takeoff_success",
        "airborne": "airborne_duration",
        "sustained_airborne_duration": "airborne_duration",
        "hover": "hover_drift",
        "forward_flight": "forward_displacement",
        "yaw_control": "controlled_yaw",
        "attitude_speed_bounds": "bounded_attitude_speed",
        "gust": "gust_recovery",
        "landing": "landing_speed_impact",
        "visual_steering": "visual_steering_direction",
        "saccade_dynamics": "saccades",
        "latency": "end_to_end_latency",
        "closed_loop_provenance": "closed_loop_source_ratio",
        "finite": "finite_values",
        "collisions": "collision_free",
        "resources": "resource_limits",
    }
    return _json_safe(report)


def validate_file(
    path: str | Path,
    *,
    thresholds: ValidationThresholds | Mapping[str, Any] | None = None,
    evidence_class: str | None = None,
) -> dict[str, Any]:
    trace = load_trace(path, evidence_class=evidence_class)
    return validate_trace(trace, thresholds=thresholds, evidence_class=evidence_class)


# Short aliases for command-loop integrations.
load_data = load_trace
validate = validate_file


def validate_paths(
    paths: Sequence[str | Path],
    *,
    thresholds: ValidationThresholds | Mapping[str, Any] | None = None,
    evidence_class: str | None = None,
) -> dict[str, Any]:
    """Validate one path, or return a multi-run wrapper for several paths."""

    if not paths:
        empty_trace = Trace(rows=[], metadata={}, source=None, evidence_class=evidence_class or EvidenceClass.UNKNOWN)
        report = validate_trace(empty_trace, thresholds=thresholds, evidence_class=evidence_class)
        report["reason"] = "no trace supplied; all metrics are SKIP"
        return report
    if len(paths) == 1:
        return validate_file(paths[0], thresholds=thresholds, evidence_class=evidence_class)
    reports = [validate_file(path, thresholds=thresholds, evidence_class=evidence_class) for path in paths]
    engineering_decisions = [report.get("engineering_decision", report.get("mechanical_decision", STATUS_NOT_VALIDATED)) for report in reports]
    if any(decision == "NO-GO" for decision in engineering_decisions):
        engineering_decision = "NO-GO"
    elif all(decision == "GO" for decision in engineering_decisions):
        engineering_decision = "GO"
    else:
        engineering_decision = STATUS_NOT_VALIDATED
    biological_decisions = [report["decision"] for report in reports]
    if any(decision == "NO-GO" for decision in biological_decisions):
        decision = "NO-GO"
    elif all(decision == "GO" for decision in biological_decisions):
        decision = "GO"
    else:
        decision = STATUS_NOT_VALIDATED
    return {
        "schema_version": "flight-validation-batch-1",
        "decision": decision,
        "status": decision,
        "decision_code": decision.replace("-", "_"),
        "go_no_go": decision,
        "overall_status": STATUS_NOT_VALIDATED if decision == STATUS_NOT_VALIDATED else decision,
        "validation_status": STATUS_NOT_VALIDATED if decision == STATUS_NOT_VALIDATED else decision,
        "biological_decision": decision,
        "biological_status": STATUS_NOT_VALIDATED if decision == STATUS_NOT_VALIDATED else decision,
        "biological_validation": decision,
        "biological_validated": decision == "GO",
        "engineering_decision": engineering_decision,
        "engineering_status": engineering_decision,
        "engineering_decision_code": engineering_decision.replace("-", "_"),
        "engineering_go": engineering_decision == "GO",
        "mechanical_decision": engineering_decision,
        "runs": reports,
        "engineering": {
            "decision": engineering_decision,
            "required_metrics": list(ENGINEERING_REQUIRED_METRICS),
            "required_engineering_metrics": list(ENGINEERING_REQUIRED_METRICS),
            "reason": "every supplied run must pass every required engineering metric for GO",
        },
        "reason": "biological and engineering decisions are reported separately; every supplied run must pass for batch GO",
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="JSON/JSONL/NPZ/CSV flight trace(s)")
    parser.add_argument("--input", "-i", dest="input_paths", action="append", type=Path, help="additional trace path (repeatable)")
    parser.add_argument("--evidence-class", choices=sorted(EvidenceClass.ALL), help="explicit provenance attestation; real_biological requires user review")
    parser.add_argument("--thresholds", type=Path, help="JSON file containing threshold overrides")
    parser.add_argument("--output", "-o", type=Path, help="write the JSON report here instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit one compact JSON line")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    paths = list(args.paths) + list(args.input_paths or [])
    threshold_mapping: Optional[Mapping[str, Any]] = None
    if args.thresholds:
        with args.thresholds.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, Mapping):
            raise SystemExit("--thresholds JSON must contain an object")
        threshold_mapping = parsed
    try:
        report = validate_paths(paths, thresholds=threshold_mapping, evidence_class=args.evidence_class)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        error = {"schema_version": "flight-validation-1", "decision": "ERROR", "error": str(exc)}
        text = json.dumps(error, ensure_ascii=False, separators=(",", ":") if args.compact else None)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 2
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    # Missing real data is deliberately not a process error.  This lets a CI
    # smoke test run without turning an honest SKIP into a false failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
