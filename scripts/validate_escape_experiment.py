#!/usr/bin/env python3
"""Strict, CPU-only acceptance checks for a digital escape experiment.

This validator is intentionally independent from the flight runner and from
``validate_digital_fly.py``.  It checks a causal chain in an exported
trajectory, rather than trusting a runner's status flags::

    stimulus -> neural response -> motor transfer -> body escape

The four evidence levels reported by :func:`validate_escape_experiment` are
``NEURAL_RESPONSE``, ``MOTOR_TRANSFER``, ``PHYSICAL_ESCAPE`` and
``CAUSAL_ESCAPE``.  A passing digital trace is still reported as
``NOT_VALIDATED`` at the biological boundary.  A single successful run can
never establish ``CAUSAL_ESCAPE``; same-seed sham/matched, mirror and
lesion/ablation runs are required.

Only the Python standard library is used.  JSON and JSONL inputs are
supported, and the normaliser accepts the row envelopes emitted by the
project (``trajectory``, ``records``, ``rows`` and ``telemetry``) as well as
small hand-authored CPU fixtures.  Missing provenance is a hard failure, not
an implicit pass.

For the strict escape protocol the palm may be visual-only/non-interacting.
The geometry-derived contact timestamp is then a counterfactual overlap
deadline, not a physics collision event.  Such traces must still cover the
configured full simulation duration, and a judged response must lift off
before the counterfactual contact/decision deadlines.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "escape-validation-1"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_VALIDATED = "NOT_VALIDATED"

NEURAL_RESPONSE = "NEURAL_RESPONSE"
MOTOR_TRANSFER = "MOTOR_TRANSFER"
PHYSICAL_ESCAPE = "PHYSICAL_ESCAPE"
CAUSAL_ESCAPE = "CAUSAL_ESCAPE"
EVIDENCE_LEVELS = (NEURAL_RESPONSE, MOTOR_TRANSFER, PHYSICAL_ESCAPE, CAUSAL_ESCAPE)


class EscapeValidationError(ValueError):
    """Raised when an input file is malformed or cannot be loaded."""


@dataclass(frozen=True)
class EscapeThresholds:
    """Numerical gates used by the validator.

    These are engineering defaults, not biological constants.  An
    experiment may override them under ``manifest.acceptance`` or
    ``manifest.thresholds``; the resolved values are retained in the report.
    """

    neural_response_delta: float = 1.0e-9
    motor_response_delta: float = 1.0e-9
    motion_displacement_m: float = 1.0e-5
    liftoff_height_m: float = 1.0e-3
    escape_projection_m: float = 1.0e-3
    control_effect_projection_m: float = 1.0e-3
    max_global_active_fraction: float = 0.50
    max_global_saturation_fraction: float = 0.50
    max_integration_accel_mps2: float = 1.0e3
    integration_tolerance_m: float = 1.0e-3
    time_tolerance_s: float = 1.0e-9
    minimum_control_rows: int = 2
    # These are deliberately optional.  A trace without an explicit protocol
    # deadline is not allowed to pass the physical gate; ``None`` is retained
    # in the resolved report so a missing timing contract is visible rather
    # than silently replaced by a convenient default.
    decision_deadline_s: float | None = None
    planned_contact_time_s: float | None = None

    @classmethod
    def from_sources(
        cls,
        manifest: Mapping[str, Any] | None = None,
        override: Mapping[str, Any] | None = None,
    ) -> "EscapeThresholds":
        source: dict[str, Any] = {}
        if isinstance(manifest, Mapping):
            # Timing is often recorded beside the acceptance block (for
            # example under ``protocol`` or ``timing_deadlines``).  Include
            # the top-level mapping as well; only known dataclass fields are
            # consumed below.
            source.update(manifest)
            for key in ("thresholds", "acceptance", "escape_acceptance"):
                value = manifest.get(key)
                if isinstance(value, Mapping):
                    source.update(value)
        if isinstance(override, Mapping):
            source.update(override)

        aliases = {
            "neural_delta_min": "neural_response_delta",
            "min_neural_response_delta": "neural_response_delta",
            "neural_response_min_delta": "neural_response_delta",
            "motor_delta_min": "motor_response_delta",
            "min_motor_response_delta": "motor_response_delta",
            "motion_threshold_m": "motion_displacement_m",
            "motion_min_displacement_m": "motion_displacement_m",
            "min_liftoff_height_m": "liftoff_height_m",
            "takeoff_height_m": "liftoff_height_m",
            "min_escape_projection_m": "escape_projection_m",
            "min_displacement_projection_m": "escape_projection_m",
            "max_active_fraction": "max_global_active_fraction",
            "global_active_fraction_max": "max_global_active_fraction",
            "max_saturation_fraction": "max_global_saturation_fraction",
            "global_saturation_fraction_max": "max_global_saturation_fraction",
            "max_accel_mps2": "max_integration_accel_mps2",
            "integration_max_accel_mps2": "max_integration_accel_mps2",
            "position_tolerance_m": "integration_tolerance_m",
            "time_tolerance": "time_tolerance_s",
            "min_control_rows": "minimum_control_rows",
            "decision_deadline": "decision_deadline_s",
            "latest_decision_s": "decision_deadline_s",
            "latest_decision": "decision_deadline_s",
            "protocol_decision_deadline_s": "decision_deadline_s",
            "protocol_decision_deadline": "decision_deadline_s",
            "escape_decision_deadline_s": "decision_deadline_s",
            "liftoff_deadline_s": "decision_deadline_s",
            "planned_contact_time": "planned_contact_time_s",
            "first_contact_time_s": "planned_contact_time_s",
            "physical_contact_time_s": "planned_contact_time_s",
            "planned_physical_contact_time_s": "planned_contact_time_s",
            "expected_surface_contact_time_s": "planned_contact_time_s",
            "expected_contact_time_s": "planned_contact_time_s",
            "impact_time_s": "planned_contact_time_s",
            "collision_time_s": "planned_contact_time_s",
            "contact_deadline_s": "planned_contact_time_s",
        }
        aliases.update({
            f"{name[:-2] if name.endswith('_s') else name}_ms": target
            for name, target in tuple(aliases.items())
            if target in {"decision_deadline_s", "planned_contact_time_s"}
        })
        aliases = {_normal_key(key): value for key, value in aliases.items()}
        names = {item for item in cls.__dataclass_fields__}
        canonical_names = {_normal_key(item): item for item in names}
        kwargs: dict[str, Any] = {}
        for raw_key, raw_value in source.items():
            key = _normal_key(raw_key)
            key = aliases.get(key, key)
            key = canonical_names.get(_normal_key(key))
            if key not in names:
                continue
            if key == "minimum_control_rows":
                try:
                    value = int(raw_value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if value >= 1:
                    kwargs[key] = value
                continue
            number = _finite(raw_value)
            normalized_raw = _normal_key(raw_key)
            if normalized_raw.endswith("ms"):
                number = None if number is None else number / 1000.0
            if number is not None and number >= 0.0:
                kwargs[key] = number
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float | int | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class ExperimentBundle:
    """Normalised trajectory, telemetry and manifest for one batch."""

    rows: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    telemetry_rows: tuple[Mapping[str, Any], ...] = ()
    source: str | None = None
    trajectory_source: str | None = None
    manifest_source: str | None = None
    telemetry_source: str | None = None
    condition: str | None = None
    time_inferred: bool = False


# Keys below are deliberately conservative.  They are checked in command or
# authority contexts; a harmless phase label in a trajectory is not itself a
# phase controller.
_AUTHORITY_NAMES = {
    "phaseauthority",
    "phasecontroller",
    "phasecommand",
    "phasedrive",
    "goalauthority",
    "goalcontroller",
    "goalcommand",
    "goaldrive",
    "statedrive",
    "flightstatedrive",
    "stabilizer",
    "flightbodystabilizer",
    "autopilot",
    "autocommand",
    "pilot",
    "safetycontroller",
    "targetfollower",
    "waypointcontroller",
}
_OFF_TEXT = {
    "0",
    "false",
    "no",
    "off",
    "none",
    "null",
    "disabled",
    "inactive",
    "log_only",
    "telemetry_only",
    "observation_only",
    "not_used",
    "prohibited",
    "record_only",
    "monitor_only",
    "monitor_and_abort_only",
}
_METADATA_KEY_NAMES = {
    "stimulusid",
    "stimuluslabel",
    "stimulusname",
    "condition",
    "variant",
    "batchid",
    "runid",
    "mirror",
    "mirrored",
    "lesion",
    "ablation",
    "sham",
    "metadata",
    "scenario",
    "phase",
    "onset",
    "onsets",
    "radius",
    "stimulusmetadata",
}
_FORBIDDEN_SIGNAL_KEYS = {
    "collisiontime",
    "collisiontimes",
    "timetocollision",
    "ttc",
    "palmrange",
    "palmdistance",
    "palmvelocity",
    "palmangle",
    "palmazimuth",
    "palmrangegroundtruth",
    "futureframe",
    "futureframes",
    "futurestate",
    "targetvisibility",
    "targetdistance",
    "targetbearing",
    "desiredtrajectory",
    "postimpact",
    "postimpactflag",
    "escapelabel",
    "reward",
    "successbit",
    "failurebit",
    "conditiongain",
    "conditionrate",
    "visualtonicstate",
    "goalposition",
    "goalpositionm",
    "goalyaw",
    "phaseconditionedgain",
}
_CONTAINER_NAMES = {
    "body",
    "pose",
    "world",
    "physics",
    "dynamics",
    "sensory",
    "sensor",
    "visual",
    "retina",
    "control",
    "motor",
    "command",
    "neural",
    "cns",
    "network",
    "activity",
    "diagnostics",
    "provenance",
    "authority",
    "events",
}


def _normal_key(value: Any) -> str:
    """Normalise a key for tolerant schema matching."""

    text = str(value).strip().lower()
    return "".join(character for character in text if character.isalnum())


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EscapeValidationError(f"cannot read {path}: {exc}") from exc
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_json_constant)
            except (TypeError, ValueError) as exc:
                raise EscapeValidationError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise EscapeValidationError(
                    f"JSONL record at {path}:{line_number} is not an object"
                )
            rows.append(value)
        return rows
    try:
        return json.loads(text, parse_constant=_json_constant)
    except (TypeError, ValueError) as exc:
        raise EscapeValidationError(f"invalid JSON at {path}: {exc}") from exc


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EscapeValidationError(f"{label} must be a JSON object")
    return value


def _extract_rows(payload: Any, label: str = "trajectory") -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Extract row records and a shallow metadata envelope."""

    if isinstance(payload, list):
        rows = payload
        rows_value = rows
        metadata: dict[str, Any] = {}
    elif isinstance(payload, Mapping):
        metadata_value = payload.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        rows_value: Any = None
        for key in ("trajectory", "records", "rows", "samples", "telemetry", "trace"):
            if key not in payload:
                continue
            candidate = payload[key]
            if isinstance(candidate, list):
                rows_value = candidate
                break
            if isinstance(candidate, Mapping):
                # A mapping keyed by condition/scenario is flattened only when
                # each value is a list of row objects.
                flattened: list[Mapping[str, Any]] = []
                for group, group_rows in candidate.items():
                    if not isinstance(group_rows, list):
                        flattened = []
                        break
                    for row in group_rows:
                        if not isinstance(row, Mapping):
                            raise EscapeValidationError(
                                f"{label}.{key}[{group!r}] contains a non-object row"
                            )
                        item = dict(row)
                        item.setdefault("batch_id", str(group))
                        flattened.append(item)
                if flattened or not candidate:
                    rows_value = flattened
                    break
                nested_rows, nested_metadata = _extract_rows(candidate, label)
                metadata.update(nested_metadata)
                rows_value = nested_rows
                break
        if rows_value is None and isinstance(payload.get("data"), (Mapping, list)):
            nested_rows, nested_metadata = _extract_rows(payload["data"], label)
            metadata.update(nested_metadata)
            rows_value = nested_rows
        if rows_value is None:
            # A one-row record is useful for explicit event fixtures, but only
            # accept it when it has a time or a body/neural/motor field.
            if any(key in payload for key in ("time_s", "timestamp", "position_m", "neural", "motor")):
                rows_value = [payload]
            else:
                raise EscapeValidationError(
                    f"{label} has no trajectory/rows/records/telemetry field"
                )
    else:
        raise EscapeValidationError(f"{label} must be a JSON object or array")

    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows_value):
        if not isinstance(row, Mapping):
            raise EscapeValidationError(f"{label} row {index} is not an object")
        rows.append(row)
    if not rows:
        raise EscapeValidationError(f"{label} contains no rows")
    return rows, metadata


def _resolve_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _manifest_output(manifest: Mapping[str, Any], base: Path, name: str) -> Path | None:
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        result = _resolve_path(base, outputs.get(name))
        if result is not None:
            return result
    for key in (name, f"{name}_path", f"{name}Path"):
        result = _resolve_path(base, manifest.get(key))
        if result is not None:
            return result
    return None


def _find_in_dir(directory: Path, manifest: Mapping[str, Any], name: str) -> Path | None:
    named = _manifest_output(manifest, directory, name)
    if named is not None and named.is_file():
        return named
    candidates = {
        "trajectory": ("trajectory.json", "trajectory.jsonl", "records.json", "flight.json"),
        "telemetry": ("telemetry.jsonl", "telemetry.json", "trace.jsonl"),
        "manifest": ("manifest.json",),
    }
    for filename in candidates.get(name, ()):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def load_bundle(
    trajectory: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    manifest: str | Path | Mapping[str, Any] | None = None,
    telemetry: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    condition: str | None = None,
) -> ExperimentBundle:
    """Load one trajectory/telemetry + manifest bundle.

    ``trajectory`` may be a run directory, JSON/JSONL path, parsed payload or
    row list.  A manifest is mandatory for a normal run; an inline manifest
    embedded in a trajectory envelope is accepted only when no separate
    manifest was supplied.
    """

    trajectory_path: Path | None = None
    manifest_path: Path | None = None
    telemetry_path: Path | None = None

    if isinstance(trajectory, (str, Path)):
        trajectory_path = Path(trajectory).expanduser()
        if trajectory_path.is_dir():
            directory = trajectory_path
            if manifest is None:
                manifest_path = _find_in_dir(directory, {}, "manifest")
                if manifest_path is None:
                    raise EscapeValidationError(f"run directory has no manifest.json: {directory}")
                manifest_payload = _read_json(manifest_path)
            elif isinstance(manifest, (str, Path)):
                manifest_path = Path(manifest).expanduser()
                if manifest_path.is_dir():
                    manifest_path = _find_in_dir(manifest_path, {}, "manifest")
                if manifest_path is None:
                    raise EscapeValidationError("manifest directory has no manifest.json")
                manifest_payload = _read_json(manifest_path)
            else:
                manifest_payload = manifest
            manifest_payload = _as_mapping(manifest_payload, "manifest")
            trajectory_path = _find_in_dir(directory, manifest_payload, "trajectory")
            if trajectory_path is None:
                raise EscapeValidationError(f"run directory has no trajectory file: {directory}")
            if telemetry is None:
                telemetry_path = _find_in_dir(directory, manifest_payload, "telemetry")
        else:
            payload = _read_json(trajectory_path)
            if isinstance(manifest, (str, Path)):
                manifest_path = Path(manifest).expanduser()
                manifest_payload = _as_mapping(_read_json(manifest_path), "manifest")
            elif manifest is None and isinstance(payload, Mapping) and isinstance(payload.get("manifest"), Mapping):
                manifest_payload = payload["manifest"]
            else:
                manifest_payload = manifest
            if manifest_payload is None:
                raise EscapeValidationError("manifest is required (pass --manifest or include an inline manifest)")
            manifest_payload = _as_mapping(manifest_payload, "manifest")
            # Keep the parsed payload below; avoid reading it twice.
    else:
        payload = trajectory
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest).expanduser()
            manifest_payload = _as_mapping(_read_json(manifest_path), "manifest")
        elif isinstance(manifest, Mapping):
            manifest_payload = manifest
        elif isinstance(payload, Mapping) and isinstance(payload.get("manifest"), Mapping):
            manifest_payload = payload["manifest"]
        else:
            raise EscapeValidationError("manifest is required for an inline trajectory")
        manifest_payload = _as_mapping(manifest_payload, "manifest")

    if trajectory_path is not None:
        payload = _read_json(trajectory_path)
    rows, metadata = _extract_rows(payload, "trajectory")
    if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping):
        metadata = dict(payload["metadata"])
    # Preserve batch/control declarations that live beside the row envelope;
    # they are part of the input contract, not transient loader state.
    if isinstance(payload, Mapping):
        for key in ("batches", "controls", "matched_controls", "comparison_runs"):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]

    if telemetry is None and telemetry_path is not None:
        telemetry = telemetry_path
    telemetry_rows: tuple[Mapping[str, Any], ...] = ()
    telemetry_source: str | None = None
    if telemetry is not None:
        if isinstance(telemetry, (str, Path)):
            telemetry_path = Path(telemetry).expanduser()
            telemetry_payload = _read_json(telemetry_path)
            telemetry_source = str(telemetry_path)
        else:
            telemetry_payload = telemetry
        trows, _ = _extract_rows(telemetry_payload, "telemetry")
        telemetry_rows = tuple(trows)

    inferred_condition = condition
    if inferred_condition is None:
        for source in (manifest_payload, metadata):
            if not isinstance(source, Mapping):
                continue
            for key in ("condition", "variant", "batch", "batch_id", "run_label", "label", "intervention"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    inferred_condition = value.strip()
                    break
            if inferred_condition:
                break

    return ExperimentBundle(
        rows=tuple(rows),
        manifest=dict(manifest_payload),
        metadata=metadata,
        telemetry_rows=telemetry_rows,
        source=str(trajectory_path.parent if trajectory_path and trajectory_path.is_file() else trajectory_path) if trajectory_path else None,
        trajectory_source=str(trajectory_path) if trajectory_path else None,
        manifest_source=str(manifest_path) if manifest_path else None,
        telemetry_source=telemetry_source,
        condition=inferred_condition,
    )


load_experiment = load_bundle
load_trajectory = load_bundle


def _lookup_normalised(mapping: Mapping[str, Any], names: Iterable[str], *, nested: bool = True) -> tuple[Any, str | None]:
    wanted = {_normal_key(name) for name in names}
    for actual_key, value in mapping.items():
        if _normal_key(actual_key) in wanted:
            return value, str(actual_key)
    if nested:
        for actual_key, value in mapping.items():
            if _normal_key(actual_key) not in {_normal_key(name) for name in _CONTAINER_NAMES}:
                continue
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    if _normal_key(child_key) in wanted:
                        return child_value, f"{actual_key}.{child_key}"
    return None, None


def _vector(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        labels = (("x", "front", "forward"), ("y", "left", "lateral"), ("z", "up", "vertical"))
        result: list[float] = []
        for keys in labels:
            found = None
            for key in keys:
                if key in value:
                    found = _finite(value[key])
                    break
            if found is None:
                return None
            result.append(found)
        return tuple(result)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        result = tuple(_finite(item) for item in value[:3])
        if any(item is None for item in result):
            return None
        return result  # type: ignore[return-value]
    return None


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float] | None:
    length = _norm(vector)
    if not math.isfinite(length) or length <= 1.0e-15:
        return None
    return tuple(float(item) / length for item in vector)  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _time(row: Mapping[str, Any], index: int, metadata: Mapping[str, Any], manifest: Mapping[str, Any]) -> float | None:
    value, key = _lookup_normalised(row, ("time_s", "t_s", "timestamp_s", "time", "t", "time_ms", "t_ms", "timestamp_ms"))
    if isinstance(value, Mapping):
        nested_key = "t_s"
        for candidate in ("t_s", "time_s", "seconds", "t_ms", "time_ms", "milliseconds"):
            if candidate in value:
                nested_key = candidate
                value = value[candidate]
                break
        key = nested_key
    number = _finite(value)
    if number is not None:
        if key and _normal_key(key) in {"timems", "tms", "timestampms", "milliseconds"}:
            return number / 1000.0
        return number
    frame, _ = _lookup_normalised(row, ("frame_index", "frame", "index"))
    frame_value = _finite(frame)
    if frame_value is None:
        return None
    rate = None
    for source in (metadata, manifest):
        if not isinstance(source, Mapping):
            continue
        for key in ("frame_rate_hz", "control_rate_hz", "rate_hz", "fps"):
            rate = _finite(source.get(key))
            if rate is not None and rate > 0:
                break
        if rate is not None and rate > 0:
            break
    return frame_value / rate if rate and rate > 0 else None


def _vector_from_row(
    row: Mapping[str, Any],
    vector_names: Sequence[str],
    component_names: Sequence[str],
) -> tuple[float, float, float] | None:
    value, _ = _lookup_normalised(row, vector_names)
    result = _vector(value)
    if result is not None:
        return result
    components: list[float] = []
    for key in component_names:
        item, _ = _lookup_normalised(row, (key,))
        number = _finite(item)
        if number is None:
            return None
        components.append(number)
    return tuple(components) if len(components) == 3 else None  # type: ignore[return-value]


def _position(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    return _vector_from_row(row, ("position_m", "position", "pose_m", "pose_position_m"), ("x_m", "y_m", "z_m"))


def _velocity(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    return _vector_from_row(row, ("velocity_mps", "velocity", "linear_velocity_mps", "body_velocity_mps"), ("vx_mps", "vy_mps", "vz_mps"))


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "on", "1", "active", "airborne", "flight"}:
            return True
        if text in {"false", "no", "off", "0", "inactive", "grounded", "none"}:
            return False
    return None


def _contact(row: Mapping[str, Any]) -> bool | None:
    value, _ = _lookup_normalised(row, ("contact", "ground_contact", "grounded", "on_ground"))
    return _boolean(value)


def _collision(row: Mapping[str, Any]) -> tuple[bool | None, bool]:
    value, _ = _lookup_normalised(row, ("collision_count", "collisions", "collision", "impact"))
    if value is None:
        return None, False
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value), True
    number = _finite(value)
    if number is not None:
        return number > 0.0, True
    boolean = _boolean(value)
    return boolean, boolean is not None


def _scalar(value: Any) -> float | None:
    number = _finite(value)
    if number is not None:
        return abs(number)
    if isinstance(value, Mapping):
        # Prefer semantically meaningful scalar activity fields.  ``active``
        # is handled before arbitrary nested values so a vector of metadata
        # cannot accidentally manufacture a response.
        for key in (
            "response_score",
            "activity",
            "population_rates_hz",
            "population_rates",
            "population_spike_counts",
            "spike_counts",
            "named_outputs",
            "active_count",
            "active_neurons",
            "spikes",
            "spike_count",
            "total_spikes",
            "mean_rate_hz",
            "rate_hz",
            "firing_rate_hz",
            "value",
        ):
            if key in value:
                result = _scalar(value[key])
                if result is not None:
                    return result
        numeric_children = [
            abs(number)
            for child in value.values()
            for number in [_finite(child)]
            if number is not None
        ]
        if numeric_children:
            return max(numeric_children)
        return None
    if isinstance(value, (list, tuple)):
        values = [_finite(item) for item in value]
        values = [abs(item) for item in values if item is not None]
        if not values:
            return None
        # A population vector is represented by an active-neuron count where
        # possible; this preserves the meaning of sparse-vs-global activity.
        return float(sum(item > 0.0 for item in values))
    return None


def _neural_value(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    value, path = _lookup_normalised(
        row,
        (
            "neural_activity",
            "neural_response",
            "neural_signal",
            "spike_count",
            "spikes",
            "rates_hz",
            "active_neurons",
            "active_count",
            "mean_rate_hz",
            "activity",
            "active_neurons",
            "spike_count",
            "population_rates_hz",
            "population_spike_counts",
            "named_outputs",
        ),
    )
    if value is not None:
        return _scalar(value), path
    return None, None


def _population_signal(row: Mapping[str, Any], populations: Sequence[str]) -> tuple[float | None, str | None]:
    """Read a named population event from compact ``named_outputs`` maps."""

    wanted = tuple(_normal_key(name) for name in populations)

    def walk(value: Any, path: str, depth: int = 0) -> tuple[float | None, str | None]:
        if depth > 6:
            return None, None
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = _normal_key(key)
                if any(normalized == name or normalized.startswith(name) for name in wanted):
                    number = _scalar(child)
                    if number is not None:
                        return number, f"{path}.{key}" if path else str(key)
            for key, child in value.items():
                if isinstance(child, Mapping):
                    result = walk(child, f"{path}.{key}" if path else str(key), depth + 1)
                    if result[0] is not None:
                        return result
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                result = walk(child, f"{path}[{index}]", depth + 1)
                if result[0] is not None:
                    return result
        return None, None

    return walk(row, "")


def _motor_value(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    value, path = _lookup_normalised(
        row,
        (
            "motor_output",
            "motor_command",
            "escape_command",
            "motor_activity",
            "command",
            "thrust",
            "wing_power",
            "yaw_rate_rad_s",
            "steering",
            "steering_response",
            "jump",
        ),
    )
    if value is not None:
        return _scalar(value), path
    # A compact motor mapping often contains only actuator channels.
    for container in ("motor", "control", "command", "actuators"):
        child, child_path = _lookup_normalised(row, (container,), nested=False)
        if isinstance(child, Mapping):
            values: list[float] = []
            for key, item in child.items():
                if _normal_key(key) in {"source", "provenance", "mode", "safetyactive", "safetyreason"}:
                    continue
                number = _scalar(item)
                if number is not None:
                    values.append(number)
            if values:
                return max(values), child_path or container
    population_value, population_path = _population_signal(
        row,
        ("TTMn", "DLM", "DLMn", "motor", "wing_motor"),
    )
    if population_value is not None:
        return population_value, population_path
    return None, None


def _neural_fraction(row: Mapping[str, Any]) -> tuple[float | None, float | None, bool | None, str | None]:
    """Return active fraction, saturation fraction, global saturation flag and path."""

    value, path = _lookup_normalised(row, ("network", "neural_network"), nested=False)
    network = value if isinstance(value, Mapping) else None
    if network is None:
        value, path = _lookup_normalised(row, ("neural", "cns"), nested=False)
        network = value if isinstance(value, Mapping) else None
    if network is None:
        network = row
        path = "row"

    # Compact MaleCNS rows put counts in ``neural.named_outputs``.  Treat that
    # record as the network summary, while retaining the outer path for the
    # audit trail.  No dense vector is required for this CPU-side gate.
    if isinstance(network.get("named_outputs"), Mapping):
        nested = dict(network["named_outputs"])
        for key, value in network.items():
            if key != "named_outputs" and key not in nested:
                nested[key] = value
        network = nested

    neuron_count = None
    active_count = None
    active_fraction = None
    saturation_fraction = None
    global_saturation = None
    for key in ("neuron_count", "n_neurons", "total_neurons", "population_size"):
        neuron_count = _finite(network.get(key))
        if neuron_count is not None:
            break
    for key in ("active_neuron_count", "active_count", "active_neurons", "n_active"):
        active_count = _finite(network.get(key))
        if active_count is not None:
            break
    for key in ("active_fraction", "global_active_fraction", "fraction_active"):
        active_fraction = _finite(network.get(key))
        if active_fraction is not None:
            break
    for key in ("saturation_fraction", "global_saturation_fraction", "fraction_saturated"):
        saturation_fraction = _finite(network.get(key))
        if saturation_fraction is not None:
            break
    for key in ("global_saturation", "saturated", "network_saturated"):
        if key in network:
            global_saturation = _boolean(network[key])
            break
    if active_fraction is None and neuron_count and active_count is not None:
        active_fraction = active_count / neuron_count
    if saturation_fraction is None and network.get("saturated_count") is not None and neuron_count:
        saturated_count = _finite(network.get("saturated_count"))
        if saturated_count is not None:
            saturation_fraction = saturated_count / neuron_count
    if saturation_fraction is None and global_saturation is False:
        # A false global-saturation flag is direct evidence that this trace
        # did not report a network-wide saturated state.  Keep the fraction
        # explicit in the normalized report as zero, while still requiring an
        # active-fraction/size channel independently.
        saturation_fraction = 0.0
    if active_fraction is None:
        # A vector is sufficient to estimate an active fraction, but it is
        # still not sufficient to infer saturation without a declared cap.
        rates = None
        for key in ("rates_hz", "activity_vector", "neural_rates"):
            candidate = network.get(key)
            if isinstance(candidate, (list, tuple)):
                rates = candidate
                break
        if rates is not None and rates:
            finite_values = [_finite(item) for item in rates]
            if all(item is not None for item in finite_values):
                active_fraction = sum(float(item) > 0.0 for item in finite_values) / len(finite_values)  # type: ignore[arg-type]
                neuron_count = float(len(finite_values))
    if active_fraction is not None and not 0.0 <= active_fraction <= 1.0:
        active_fraction = None
    if saturation_fraction is not None and not 0.0 <= saturation_fraction <= 1.0:
        saturation_fraction = None
    return active_fraction, saturation_fraction, global_saturation, path


def _row_times(bundle: ExperimentBundle) -> tuple[list[float], list[str]]:
    times: list[float] = []
    errors: list[str] = []
    for index, row in enumerate(bundle.rows):
        value = _time(row, index, bundle.metadata, bundle.manifest)
        if value is None:
            errors.append(f"row {index} has no finite time_s/timestamp")
        else:
            times.append(value)
    if len(times) == len(bundle.rows):
        for index, (previous, current) in enumerate(zip(times, times[1:]), 1):
            if current <= previous:
                errors.append(
                    f"timestamps are not strictly increasing at row {index - 1}->{index}"
                )
    return times, errors


def _declared_off(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        return value.strip().lower() in _OFF_TEXT
    if isinstance(value, Mapping):
        for key in ("enabled", "active", "authority", "writes_behavior", "writes_control", "used"):
            if key in value:
                return _declared_off(value[key])
        for key in ("source", "mode", "status"):
            if key in value:
                return _declared_off(value[key])
        return False
    return False


def _policy_explicitly_prohibits(policy: Mapping[str, Any], logical_name: str) -> bool:
    """Recognise the strict runner's explicit prohibited-path contract."""

    strict = _boolean(policy.get("strict"))
    fail_closed = _boolean(policy.get("fail_closed"))
    authority = str(policy.get("control_authority", "")).strip().lower()
    if strict is not True or fail_closed is not True or authority not in {"malecns", "neural", "connectome"}:
        return False
    paths = policy.get("prohibited_paths", ())
    if not isinstance(paths, (list, tuple, set)):
        return False
    tokens = {
        "phase": ("phase", "mission_phase"),
        "goal": ("goal", "mission_goal"),
        "state_drive": ("state_drive", "flight_state_drive"),
        "stabilizer": ("stabilizer", "flight_body_stabilizer"),
        "autopilot": ("autopilot", "flight_autopilot"),
    }.get(logical_name, (logical_name,))
    text = " ".join(str(item).lower() for item in paths)
    normalized_text = _normal_key(text)
    return any(token in text or _normal_key(token) in normalized_text for token in tokens)


def _authority_audit(bundle: ExperimentBundle) -> dict[str, Any]:
    manifest = bundle.manifest
    reasons: list[str] = []
    violations: list[str] = []
    declaration: Mapping[str, Any] | None = None
    for key in (
        "behavior_authority",
        "control_authority",
        "authority",
        "input_authority",
        "closed_loop_authority",
        # The strict runner records this as an autonomy policy rather than a
        # flat authority map.  It is accepted only when it is explicitly
        # strict/fail-closed and names the allowed neural control authority.
        "autonomy_policy",
    ):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            declaration = value
            break

    aliases = {
        "phase": ("phase", "phase_authority", "phase_controller"),
        "goal": ("goal", "goal_authority", "goal_controller"),
        "state_drive": ("state_drive", "flight_state_drive", "state_drive_authority"),
        "stabilizer": ("stabilizer", "flight_body_stabilizer", "stabilizer_authority"),
        "autopilot": ("autopilot", "auto_pilot", "autopilot_authority"),
    }
    missing: list[str] = []
    if declaration is None:
        missing.extend(aliases)
        reasons.append("manifest lacks an explicit behavior/control authority declaration")
    else:
        for logical_name, candidates in aliases.items():
            found = None
            for candidate in candidates:
                for key, value in declaration.items():
                    if _normal_key(key) == _normal_key(candidate):
                        found = value
                        break
                if found is not None:
                    break
            if found is None and _policy_explicitly_prohibits(declaration, logical_name):
                found = "prohibited"
            if found is None:
                missing.append(logical_name)
            elif not _declared_off(found):
                violations.append(f"{logical_name} has behavior/control authority")

    # Direct component declarations are only allowed when explicitly disabled
    # or telemetry-only.  `phase` and `goal` labels remain legal as labels;
    # only their authority-shaped fields are inspected here.
    def walk(value: Any, path: str, parent: str | None = None) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = _normal_key(key)
                child_path = f"{path}.{key}" if path else key
                if normalized in _AUTHORITY_NAMES:
                    if not _declared_off(child):
                        violations.append(f"{child_path} is active")
                elif normalized in {"goalauthority", "phaseauthority", "goalcontroller", "phasecontroller"}:
                    if not _declared_off(child):
                        violations.append(f"{child_path} is active")
                walk(child, child_path, normalized)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", parent)

    walk(manifest, "manifest")
    for key in (
        "goal",
        "goals",
        "mission_goal",
        "mission_goals",
        "goal_position_m",
        "goal_yaw_rad",
        "desired_trajectory",
        "target_follower",
    ):
        if key in manifest and not _declared_off(manifest[key]):
            violations.append(f"manifest.{key} supplies a behavior goal/trajectory")
    for index, row in enumerate(bundle.rows):
        # A phase label is data.  A goal/state-drive/autopilot object in a
        # control path is authority and is rejected unless explicitly off.
        for container_name in ("control", "motor", "command", "actuators", "sensory"):
            child, _ = _lookup_normalised(row, (container_name,), nested=False)
            if not isinstance(child, Mapping):
                continue
            for key, value in child.items():
                normalized = _normal_key(key)
                if normalized in _AUTHORITY_NAMES or normalized in {
                    "phase", "phaselabel", "goal", "goalposition", "goalpositionm", "goalerror",
                    "goalyaw", "phasecommand", "stateinput", "statedrive", "flightstatedrive",
                    "stabilizer", "flightbodystabilizer", "autopilot",
                }:
                    if not _declared_off(value):
                        violations.append(f"row {index} control field {key!r} has authority")
                if normalized in _FORBIDDEN_SIGNAL_KEYS:
                    violations.append(f"row {index} control field {key!r} is a forbidden future/target signal")
        for key, value in row.items():
            normalized = _normal_key(key)
            if normalized in {"goalpositionm", "goalyaw", "goal", "targettrajectory", "desiredtrajectory"}:
                if not _declared_off(value):
                    violations.append(f"row {index} field {key!r} supplies a goal/trajectory")
        for key in ("autopilot", "flight_state_drive", "flight_body_stabilizer", "stabilizer", "state_drive", "goal_command", "pose_write"):
            if key not in row:
                continue
            if not _declared_off(row[key]):
                violations.append(f"row {index} field {key!r} is active")

    if missing:
        reasons.append("missing explicit off declarations for: " + ", ".join(missing))
    if violations:
        reasons.extend(violations)
    return {
        "passed": not missing and not violations,
        "missing": missing,
        "violations": violations,
        "declaration_present": declaration is not None,
        "reasons": reasons,
    }


def _stimulus_mapping(manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for source in (manifest, metadata):
        for key in (
            "stimulus",
            "stimulus_metadata",
            "stimulus_spec",
            "experiment_stimulus",
            "stimulus_overrides",
            "physical_stimulus",
            "cue",
            "threat",
            "loom",
            "escape_stimulus",
        ):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, Mapping):
                # Runner manifests wrap the physical stimulus in a
                # ``physical_stimulus`` envelope and keep the resolved
                # geometry under ``manifest``.  Preserve both: the wrapper
                # carries the interaction contract while the nested mapping
                # carries start/velocity/size/contact metadata.
                nested_manifest = value.get("manifest")
                if isinstance(nested_manifest, Mapping):
                    merged = dict(nested_manifest)
                    merged.update(dict(value))
                    return merged
                return value
        # A world/experiment envelope may keep the physical stimulus under a
        # nested key.  Only accept an explicitly named stimulus object; do not
        # treat arbitrary world geometry or goals as a cue.
        for parent_key in ("experiment", "escape_experiment", "world"):
            parent = source.get(parent_key) if isinstance(source, Mapping) else None
            if isinstance(parent, Mapping):
                for key in ("stimulus", "stimulus_metadata", "stimulus_spec", "escape_stimulus"):
                    value = parent.get(key)
                    if isinstance(value, Mapping):
                        return value
    return None


def _mapping_paths(value: Any, path: str, *, max_depth: int = 6) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield manifest/metadata mappings with stable provenance paths.

    Timing declarations have appeared in several protocol envelopes over the
    lifetime of the experiment (``protocol``, ``acceptance``, ``experiment``
    and ``stimulus_overrides``).  Walking mappings here keeps the timing
    resolver tolerant without treating trajectory rows as declarations.
    """

    if not isinstance(value, Mapping):
        return
    yield path, value
    if max_depth <= 0:
        return
    for key, child in value.items():
        if isinstance(child, Mapping):
            child_path = f"{path}.{key}" if path else str(key)
            yield from _mapping_paths(child, child_path, max_depth=max_depth - 1)


def _time_fields(
    source: Mapping[str, Any],
    seconds_names: Iterable[str],
    milliseconds_names: Iterable[str],
) -> tuple[float | None, str | None]:
    return next(iter(_time_field_values(source, seconds_names, milliseconds_names)), (None, None))


def _time_field_values(
    source: Mapping[str, Any],
    seconds_names: Iterable[str],
    milliseconds_names: Iterable[str],
) -> list[tuple[float, str]]:
    seconds = {_normal_key(name) for name in seconds_names}
    milliseconds = {_normal_key(name) for name in milliseconds_names}
    values: list[tuple[float, str]] = []
    for key, value in source.items():
        normalized = _normal_key(key)
        if normalized not in seconds and normalized not in milliseconds:
            continue
        number = _finite(value)
        if number is not None and number >= 0.0:
            scaled = number if normalized in seconds else number / 1000.0
            values.append((scaled, str(key)))
    return values


def _body_radius(bundle: ExperimentBundle) -> tuple[float | None, str | None]:
    """Resolve the body/contact radius used by the physical deadline check."""

    names = ("body_radius_m", "body_radius", "palm_radius_m", "contact_radius_m")
    # Prefer an explicitly declared body/world radius.  A stimulus contact
    # radius is only a fallback because some older manifests encoded the body
    # radius on the stimulus object.
    fallback: tuple[float | None, str | None] = (None, None)
    for source_name, source in (("manifest", bundle.manifest), ("metadata", bundle.metadata)):
        for path, mapping in _mapping_paths(source, source_name):
            for key, value in mapping.items():
                normalized = _normal_key(key)
                if normalized not in {_normal_key(name) for name in names}:
                    continue
                number = _finite(value)
                if number is None or number < 0.0:
                    continue
                location = f"{path}.{key}"
                if normalized == "contactradiusm" or normalized == "contactradius":
                    fallback = (number, location)
                else:
                    return number, location
    return fallback


def _stimulus_geometry(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise the small geometry subset needed for first contact timing."""

    start = None
    start_key = None
    for key in ("start_position_m", "start_point_m", "start_m", "center_m", "position_m"):
        value = _vector(stimulus.get(key))
        if value is not None:
            start, start_key = value, key
            break
    direction = None
    direction_key = None
    for key in ("direction_m", "direction", "normal_m", "approach_direction_m"):
        value = _unit(_vector(stimulus.get(key)) or ())
        if value is not None:
            direction, direction_key = value, key
            break
    velocity = None
    velocity_key = None
    for key in ("velocity_mps", "velocity"):
        value = _vector(stimulus.get(key))
        if value is not None:
            velocity, velocity_key = value, key
            break
    if velocity is None:
        speed = None
        speed_key = None
        for key in ("speed_mps", "speed"):
            value = _finite(stimulus.get(key))
            if value is not None and value > 0.0:
                speed, speed_key = value, key
                break
        if speed is not None and direction is not None:
            velocity = tuple(speed * item for item in direction)
            velocity_key = speed_key
    if direction is None and velocity is not None:
        # A compact moving-stimulus declaration may provide only velocity;
        # for a board moving along its normal this is the physical contact
        # axis and therefore the least-assumptive orientation.
        direction = _unit(velocity)
        direction_key = velocity_key
    size = None
    size_key = None
    for key in ("size_m", "size", "dimensions_m"):
        value = _vector(stimulus.get(key))
        if value is not None:
            size, size_key = value, key
            break
    thickness = None
    thickness_key = None
    for key in ("thickness_m", "thickness", "depth_m", "depth"):
        value = _finite(stimulus.get(key))
        if value is not None and value >= 0.0:
            thickness, thickness_key = value, key
            break
    start_time = 0.0
    start_time_key = None
    # ``onset_s`` is a sensory presentation time in the trajectory contract;
    # it is not necessarily the physical geometry's motion start.  Use it
    # only for stimulus timing, never to inflate first-contact time.
    for key in ("start_time_s", "start_time", "start_s", "time_start_s"):
        value = _finite(stimulus.get(key))
        if value is not None and value >= 0.0:
            start_time, start_time_key = value, key
            break
    if start_time_key is None:
        for key in ("start_time_ms", "start_ms", "onset_ms"):
            value = _finite(stimulus.get(key))
            if value is not None and value >= 0.0:
                start_time, start_time_key = value / 1000.0, key
                break
    return {
        "start_position": start,
        "start_position_key": start_key,
        "direction": direction,
        "direction_key": direction_key,
        "velocity": velocity,
        "velocity_key": velocity_key,
        "size": size,
        "size_key": size_key,
        "thickness": thickness,
        "thickness_key": thickness_key,
        "start_time_s": start_time,
        "start_time_key": start_time_key,
    }


def _box_half_extent_along(size: Sequence[float], direction: Sequence[float] | None, axis: Sequence[float]) -> float:
    """Return a conservative oriented-box half extent along ``axis``."""

    if direction is None:
        return 0.5 * max(abs(float(item)) for item in size)
    normal = _unit(direction)
    if normal is None:
        return 0.5 * max(abs(float(item)) for item in size)
    up_reference = (0.0, 0.0, 1.0)
    if abs(_dot(normal, up_reference)) > 0.98:
        up_reference = (0.0, 1.0, 0.0)
    right = _unit((
        up_reference[1] * normal[2] - up_reference[2] * normal[1],
        up_reference[2] * normal[0] - up_reference[0] * normal[2],
        up_reference[0] * normal[1] - up_reference[1] * normal[0],
    ))
    if right is None:
        return 0.5 * max(abs(float(item)) for item in size)
    up = _unit((
        normal[1] * right[2] - normal[2] * right[1],
        normal[2] * right[0] - normal[0] * right[2],
        normal[0] * right[1] - normal[1] * right[0],
    ))
    if up is None:
        return 0.5 * max(abs(float(item)) for item in size)
    axes = (right, up, normal)
    half = tuple(0.5 * abs(float(item)) for item in size)
    return sum(abs(_dot(axis, basis)) * half[index] for index, basis in enumerate(axes))


def _stimulus_interaction_contract(bundle: ExperimentBundle) -> dict[str, Any]:
    """Resolve whether the stimulus can physically interact with the body.

    The escape protocol deliberately renders the palm while setting it
    non-interacting.  This contract is separate from the counterfactual
    geometry timestamp: a theoretical surface overlap is not a physics
    collision and must never be reported as one.
    """

    visual_only_tokens = {
        "visualonly",
        "visualonlynoninteracting",
        "noninteracting",
        "recordonly",
        "telemetryonly",
        "counterfactualtelemetryonly",
        "counterfactual",
        "none",
        "disabled",
        "off",
    }
    interactive_tokens = {
        "interactive",
        "collidable",
        "physical",
        "physics",
        "collision",
        "enabled",
        "on",
    }
    visual_keys = {"visualonly", "noninteracting", "noninteractive"}
    collidable_keys = {
        "collidable",
        "collisionenabled",
        "physicscollisionenabled",
        "physicalinteraction",
        "interacting",
    }
    interaction_keys = {
        "stimulusinteraction",
        "interaction",
        "contact",
        "interactionmode",
        "physicsinteraction",
    }
    visual_evidence: list[str] = []
    interactive_evidence: list[str] = []
    for source_name, source in (("manifest", bundle.manifest), ("metadata", bundle.metadata)):
        for path, mapping in _mapping_paths(source, source_name):
            for raw_key, value in mapping.items():
                key = _normal_key(raw_key)
                location = f"{path}.{raw_key}"
                if key in visual_keys:
                    state = _boolean(value)
                    if state is True:
                        visual_evidence.append(location)
                    continue
                if key in collidable_keys:
                    state = _boolean(value)
                    if state is False:
                        visual_evidence.append(location)
                    elif state is True:
                        interactive_evidence.append(location)
                    continue
                if key in interaction_keys and isinstance(value, str):
                    token = _normal_key(value)
                    if token in visual_only_tokens or any(item in token for item in ("visualonly", "noninteract", "counterfactual")):
                        visual_evidence.append(f"{location}={value}")
                    elif token in interactive_tokens or any(item in token for item in ("collid", "interact", "physical")):
                        interactive_evidence.append(f"{location}={value}")

    # Explicit visual-only/non-interacting evidence wins over a generic
    # ``collidable`` value because the strict free-behavior protocol forces
    # the latter off at construction time.
    if visual_evidence:
        mode = "visual_only_non_interacting"
    elif interactive_evidence:
        mode = "interactive_collidable"
    else:
        mode = "unknown"
    return {
        "mode": mode,
        "visual_only": mode == "visual_only_non_interacting",
        "collision_required": mode == "interactive_collidable",
        "declared": mode != "unknown",
        "visual_evidence": sorted(set(visual_evidence)),
        "interactive_evidence": sorted(set(interactive_evidence)),
    }


def _configured_duration(bundle: ExperimentBundle) -> dict[str, Any]:
    """Return the required full simulation duration from the manifest.

    The largest explicit run/scenario duration is used.  Stimulus lifetime
    can be shorter than the simulation, and accepting the smaller value would
    allow a trace to stop immediately after the counterfactual contact point.
    """

    seconds_names = (
        "duration_s",
        "simulation_duration_s",
        "configured_duration_s",
        "scenario_duration_s",
        "run_duration_s",
        "full_duration_s",
        "mission_duration_s",
        "requested_duration_s",
        "protocol_duration_s",
    )
    milliseconds_names = tuple(name.replace("_s", "_ms") for name in seconds_names if name.endswith("_s")) + (
        "duration_ms",
        "simulation_duration_ms",
        "scenario_duration_ms",
        "run_duration_ms",
        "full_duration_ms",
        "mission_duration_ms",
        "requested_duration_ms",
    )
    candidates: list[tuple[float, str]] = []
    seen: set[int] = set()
    for source_name, source in (("manifest", bundle.manifest), ("metadata", bundle.metadata)):
        for path, mapping in _mapping_paths(source, source_name):
            if id(mapping) in seen:
                continue
            seen.add(id(mapping))
            for value, key in _time_field_values(mapping, seconds_names, milliseconds_names):
                if value >= 0.0:
                    candidates.append((value, f"{path}.{key}"))
    # A run manifest may carry only the source config path while the selected
    # stimulus manifest carries its own (shorter) lifetime.  Resolve the
    # referenced config read-only when available so a free-behavior trace is
    # checked against the full scenario duration (2.5 s in the core config),
    # not merely the palm's 2.0 s render duration.
    config_values: list[Any] = []
    for source in (bundle.manifest, bundle.metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("config", "config_path", "world_config", "experiment_config"):
            value = source.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                config_values.append(value)
    config_paths: list[Path] = []
    for value in config_values:
        raw_path = Path(value).expanduser()
        config_paths.append(raw_path)
        if not raw_path.is_absolute():
            if bundle.manifest_source:
                config_paths.append(Path(bundle.manifest_source).expanduser().parent / raw_path)
            if bundle.trajectory_source:
                config_paths.append(Path(bundle.trajectory_source).expanduser().parent / raw_path)
            config_paths.append(Path.cwd() / raw_path)
    seen_paths: set[str] = set()
    for config_path in config_paths:
        path_key = str(config_path)
        if path_key in seen_paths or not config_path.is_file():
            continue
        seen_paths.add(path_key)
        try:
            config_payload = _read_json(config_path)
        except EscapeValidationError:
            continue
        for path, mapping in _mapping_paths(config_payload, f"config:{config_path}"):
            for value, key in _time_field_values(mapping, seconds_names, milliseconds_names):
                if value >= 0.0:
                    candidates.append((value, f"{path}.{key}"))
    if not candidates:
        return {
            "duration_s": None,
            "source": None,
            "candidates": [],
        }
    duration, source = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "duration_s": duration,
        "source": source,
        "candidates": [{"duration_s": value, "source": path} for value, path in candidates],
    }


def _timing_deadlines(
    bundle: ExperimentBundle,
    thresholds: EscapeThresholds,
    positions: Sequence[tuple[float, float, float] | None],
) -> dict[str, Any]:
    """Resolve protocol and geometry contact deadlines, fail-closed.

    ``decision_deadline_s`` is the latest permitted body liftoff time.  The
    counterfactual contact time is a separate contract: it is read from an
    explicit protocol field when present, or derived from declared
    moving-stimulus geometry.  For a visual-only stimulus, the absence of a
    physics collision row is valid; a trace that ends before the contact or
    full-duration deadline is not evidence of avoidance.
    """

    decision_seconds = (
        "decision_deadline_s",
        "decision_deadline",
        "latest_decision_s",
        "latest_decision",
        "protocol_decision_deadline_s",
        "protocol_decision_deadline",
        "escape_decision_deadline_s",
        "liftoff_deadline_s",
        "response_deadline_s",
        "go_no_go_deadline_s",
    )
    decision_milliseconds = tuple(name.replace("_s", "_ms") for name in decision_seconds if name.endswith("_s")) + (
        "decision_deadline_ms",
        "latest_decision_ms",
        "protocol_decision_deadline_ms",
        "escape_decision_deadline_ms",
        "liftoff_deadline_ms",
        "response_deadline_ms",
        "go_no_go_deadline_ms",
    )
    contact_seconds = (
        "planned_contact_time_s",
        "planned_physical_contact_time_s",
        "first_contact_time_s",
        "physical_contact_time_s",
        "expected_surface_contact_time_s",
        "expected_contact_time_s",
        "surface_contact_deadline_s",
        "planned_contact_deadline_s",
        "counterfactual_contact_time_s",
        "counterfactual_contact_deadline_s",
        "contact_deadline_s",
        "impact_time_s",
        "collision_time_s",
        "surface_contact_time_s",
    )
    contact_milliseconds = tuple(name.replace("_s", "_ms") for name in contact_seconds if name.endswith("_s")) + (
        "planned_contact_time_ms",
        "planned_physical_contact_time_ms",
        "first_contact_time_ms",
        "physical_contact_time_ms",
        "expected_surface_contact_time_ms",
        "expected_contact_time_ms",
        "surface_contact_deadline_ms",
        "planned_contact_deadline_ms",
        "counterfactual_contact_time_ms",
        "counterfactual_contact_deadline_ms",
        "contact_deadline_ms",
        "impact_time_ms",
        "collision_time_ms",
        "surface_contact_time_ms",
    )

    decision: float | None = thresholds.decision_deadline_s
    decision_source = "thresholds.decision_deadline_s" if decision is not None else None
    planned: float | None = thresholds.planned_contact_time_s
    planned_source = "thresholds.planned_contact_time_s" if planned is not None else None
    contexts: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[int] = set()
    for source_name, source in (("manifest", bundle.manifest), ("metadata", bundle.metadata)):
        for path, mapping in _mapping_paths(source, source_name):
            if id(mapping) in seen:
                continue
            seen.add(id(mapping))
            contexts.append((path, mapping))
    planned_candidates: list[tuple[int, int, float, str]] = []
    primary_contact_fields = {
        _normal_key(name)
        for name in (
            "planned_contact_time_s",
            "planned_physical_contact_time_s",
            "first_contact_time_s",
            "physical_contact_time_s",
            "expected_surface_contact_time_s",
            "expected_contact_time_s",
            "surface_contact_deadline_s",
            "planned_contact_deadline_s",
            "counterfactual_contact_time_s",
            "counterfactual_contact_deadline_s",
            "contact_deadline_s",
        )
    }
    primary_contact_fields_ms = {
        _normal_key(name)
        for name in (
            "planned_contact_time_ms",
            "planned_physical_contact_time_ms",
            "first_contact_time_ms",
            "physical_contact_time_ms",
            "expected_surface_contact_time_ms",
            "expected_contact_time_ms",
            "surface_contact_deadline_ms",
            "planned_contact_deadline_ms",
            "counterfactual_contact_time_ms",
            "counterfactual_contact_deadline_ms",
            "contact_deadline_ms",
        )
    }
    order = 0
    for path, mapping in contexts:
        if decision is None:
            value, key = _time_fields(mapping, decision_seconds, decision_milliseconds)
            if value is not None:
                decision, decision_source = value, f"{path}.{key}"
        for value, key in _time_field_values(mapping, contact_seconds, contact_milliseconds):
            normalized_key = _normal_key(key)
            rank = 0 if normalized_key in primary_contact_fields or normalized_key in primary_contact_fields_ms else 1
            planned_candidates.append((rank, order, value, f"{path}.{key}"))
            order += 1
    if planned is None and planned_candidates:
        _, _, planned, planned_source = min(planned_candidates, key=lambda item: (item[0], item[1]))

    stimulus = _stimulus_mapping(bundle.manifest, bundle.metadata)
    geometry: dict[str, Any] = {"available": False, "reasons": []}
    if planned is None and isinstance(stimulus, Mapping) and positions and positions[0] is not None:
        parsed = _stimulus_geometry(stimulus)
        start = parsed["start_position"]
        velocity = parsed["velocity"]
        if start is None:
            geometry["reasons"].append("stimulus start position is missing")
        if velocity is None:
            geometry["reasons"].append("stimulus velocity/speed is missing")
        if start is not None and velocity is not None:
            speed = _norm(velocity)
            travel = _unit(velocity)
            if travel is None or speed <= 0.0:
                geometry["reasons"].append("stimulus approach speed is not positive")
            else:
                body_position = positions[0]
                range_along = _dot(
                    [body_position[index] - start[index] for index in range(3)], travel  # type: ignore[index]
                )
                half_extent = None
                if parsed["thickness"] is not None:
                    half_extent = 0.5 * float(parsed["thickness"])
                elif parsed["size"] is not None:
                    half_extent = _box_half_extent_along(
                        parsed["size"], parsed["direction"], travel
                    )
                else:
                    geometry["reasons"].append("stimulus thickness/size is missing")
                radius, radius_source = _body_radius(bundle)
                if radius is None:
                    geometry["reasons"].append("body/contact radius is missing")
                if half_extent is not None and radius is not None and range_along > 0.0:
                    contact_distance = range_along - half_extent - radius
                    if contact_distance < 0.0:
                        geometry["reasons"].append("initial body/stimulus geometry is already intersecting")
                    else:
                        planned = float(parsed["start_time_s"]) + contact_distance / speed
                        planned_source = "derived:stimulus_geometry"
                        geometry.update({
                            "available": True,
                            "start_position_m": start,
                            "velocity_mps": velocity,
                            "speed_mps": speed,
                            "travel_direction": travel,
                            "range_along_m": range_along,
                            "half_extent_along_m": half_extent,
                            "body_radius_m": radius,
                            "body_radius_source": radius_source,
                            "contact_distance_m": contact_distance,
                            "start_time_s": parsed["start_time_s"],
                            "contact_time_s": planned,
                            "stimulus_start_source": parsed["start_position_key"],
                            "stimulus_velocity_source": parsed["velocity_key"],
                            "stimulus_size_source": parsed["thickness_key"] or parsed["size_key"],
                        })
    elif planned is not None:
        geometry["available"] = True
        geometry["contact_time_s"] = planned

    return {
        "decision_deadline_s": decision,
        "decision_deadline_source": decision_source,
        "planned_contact_time_s": planned,
        "planned_contact_time_source": planned_source,
        "geometry": geometry,
    }


def _flatten_scalars(value: Any, *, key: str | None = None) -> tuple[set[str], set[float]]:
    strings: set[str] = set()
    numbers: set[float] = set()
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            strings.add(str(child_value)) if isinstance(child_value, str) else None
            child_strings, child_numbers = _flatten_scalars(child_value, key=str(child_key))
            strings.update(child_strings)
            numbers.update(child_numbers)
    elif isinstance(value, str):
        strings.add(value)
    else:
        number = _finite(value)
        if number is not None:
            numbers.add(number)
    return strings, numbers


def _iter_signal_maps(row: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for container in ("sensory", "sensor", "visual", "retina", "control", "motor", "command", "actuators"):
        value, path = _lookup_normalised(row, (container,), nested=False)
        if isinstance(value, Mapping):
            root = path or container
            stack: list[tuple[str, Mapping[str, Any], int]] = [(root, value, 0)]
            while stack:
                current_path, current, depth = stack.pop()
                yield current_path, current
                if depth >= 6:
                    continue
                for child_key, child_value in current.items():
                    if isinstance(child_value, Mapping):
                        stack.append((f"{current_path}.{child_key}", child_value, depth + 1))


def _routing_declaration(manifest: Mapping[str, Any]) -> tuple[bool | None, bool | None, Mapping[str, Any] | None]:
    routing: Mapping[str, Any] | None = None
    for key in ("stimulus_routing", "data_lineage", "lineage", "routing", "signal_routing", "provenance"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            # Do not use a generic provenance map unless it carries routing
            # flags; component source labels are not lineage evidence.
            if key == "provenance" and not any("metadata" in _normal_key(k) or "routing" in _normal_key(k) for k in value):
                continue
            routing = value
            break
    if routing is None:
        policy = manifest.get("autonomy_policy")
        if isinstance(policy, Mapping):
            record_only = all(
                str(policy.get(key, "")).strip().lower() in {"record_only", "telemetry_only", "prohibited"}
                for key in ("stimulus_labels", "stimulus_direction", "future_state")
            )
            if record_only and _boolean(policy.get("strict")) is True and _boolean(policy.get("fail_closed")) is True:
                return False, False, policy
        return None, None, None
    def find(names: Iterable[str]) -> bool | None:
        wanted = {_normal_key(name) for name in names}
        for key, value in routing.items():
            if _normal_key(key) in wanted:
                return _boolean(value)
        return None
    return (
        find(("metadata_to_sensory", "stimulus_metadata_to_sensory", "metadata_flows_to_sensory")),
        find(("metadata_to_control", "stimulus_metadata_to_control", "metadata_flows_to_control")),
        routing,
    )


def _stimulus_leakage_audit(bundle: ExperimentBundle) -> dict[str, Any]:
    stimulus = _stimulus_mapping(bundle.manifest, bundle.metadata)
    reasons: list[str] = []
    leaks: list[str] = []
    if stimulus is None:
        reasons.append("missing stimulus metadata mapping")
        stimulus_strings: set[str] = set()
    else:
        stimulus_strings, _ = _flatten_scalars(stimulus)
    to_sensory, to_control, routing = _routing_declaration(bundle.manifest)
    if to_sensory is not False:
        reasons.append("stimulus metadata -> sensory routing is not explicitly false")
    if to_control is not False:
        reasons.append("stimulus metadata -> control routing is not explicitly false")
    if routing is None:
        reasons.append("missing stimulus routing/data-lineage declaration")

    rows_to_check: list[tuple[str, Mapping[str, Any]]] = [("trajectory", row) for row in bundle.rows]
    rows_to_check.extend(("telemetry", row) for row in bundle.telemetry_rows)
    for index, (row_kind, row) in enumerate(rows_to_check):
        for path, signal in _iter_signal_maps(row):
            for raw_key, value in signal.items():
                normalized = _normal_key(raw_key)
                if normalized in _METADATA_KEY_NAMES or normalized.startswith("stimulusmetadata"):
                    # Physical sensory values such as x/y/z are not in this
                    # set; IDs/conditions/labels are metadata by definition.
                    leaks.append(f"{row_kind} row {index} {path}.{raw_key} looks like stimulus metadata")
                if normalized in _FORBIDDEN_SIGNAL_KEYS:
                    leaks.append(f"{row_kind} row {index} {path}.{raw_key} is a forbidden stimulus/target signal")
                if isinstance(value, str) and value in stimulus_strings:
                    leaks.append(f"{row_kind} row {index} {path}.{raw_key} repeats a stimulus metadata value")
    return {
        "passed": stimulus is not None and to_sensory is False and to_control is False and routing is not None and not leaks,
        "routing": {
            "metadata_to_sensory": to_sensory,
            "metadata_to_control": to_control,
        },
        "leaks": leaks,
        "reasons": reasons,
    }


def _telemetry_audit(bundle: ExperimentBundle, thresholds: EscapeThresholds) -> dict[str, Any]:
    """Check that an optional telemetry sidecar is the same experiment."""

    if not bundle.telemetry_rows:
        return {
            "present": False,
            "passed": True,
            "reasons": [],
            "n_rows": 0,
        }
    reasons: list[str] = []
    if len(bundle.telemetry_rows) != len(bundle.rows):
        reasons.append(
            f"telemetry row count {len(bundle.telemetry_rows)} does not match trajectory row count {len(bundle.rows)}"
        )
    telemetry_bundle = ExperimentBundle(
        rows=bundle.telemetry_rows,
        manifest=bundle.manifest,
        metadata=bundle.metadata,
    )
    trajectory_times, trajectory_errors = _row_times(bundle)
    telemetry_times, telemetry_errors = _row_times(telemetry_bundle)
    if trajectory_errors or telemetry_errors:
        reasons.extend(trajectory_errors)
        reasons.extend("telemetry " + error for error in telemetry_errors)
    if not trajectory_errors and not telemetry_errors and len(trajectory_times) == len(telemetry_times):
        if any(abs(a - b) > thresholds.time_tolerance_s for a, b in zip(trajectory_times, telemetry_times)):
            reasons.append("telemetry timestamps do not match trajectory timestamps")
    if len(bundle.telemetry_rows) == len(bundle.rows):
        for index, (trajectory_row, telemetry_row) in enumerate(zip(bundle.rows, bundle.telemetry_rows)):
            trajectory_position = _position(trajectory_row)
            telemetry_position = _position(telemetry_row)
            if trajectory_position is not None and telemetry_position is not None:
                if _norm([trajectory_position[k] - telemetry_position[k] for k in range(3)]) > thresholds.integration_tolerance_m:
                    reasons.append(f"telemetry body position differs at row {index}")
                    break
    return {
        "present": True,
        "passed": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "n_rows": len(bundle.telemetry_rows),
    }


def _stimulus_onset(bundle: ExperimentBundle, times: Sequence[float]) -> tuple[float | None, str | None]:
    stimulus = _stimulus_mapping(bundle.manifest, bundle.metadata)
    for source, label in ((stimulus, "manifest.stimulus"), (bundle.manifest, "manifest"), (bundle.metadata, "trajectory.metadata")):
        if not isinstance(source, Mapping):
            continue
        for key in ("onset_s", "start_s", "onset_time_s", "start_time_s", "t_onset_s"):
            value = _finite(source.get(key))
            if value is not None:
                return value, f"{label}.{key}"
        for key in ("onset_ms", "start_ms", "onset_time_ms", "start_time_ms", "t_onset_ms"):
            value = _finite(source.get(key))
            if value is not None:
                return value / 1000.0, f"{label}.{key}"
    active_values: list[bool | None] = []
    magnitudes: list[float | None] = []
    for row in bundle.rows:
        value, _ = _lookup_normalised(row, ("stimulus_active", "cue_active", "threat_active", "loom_active", "stimulus_present"))
        active_values.append(_boolean(value))
        value, _ = _lookup_normalised(row, ("stimulus_magnitude", "cue_magnitude", "loom_radius_deg", "loom_size", "cue"))
        magnitudes.append(_scalar(value))
    for index, value in enumerate(active_values):
        if value is True and (index == 0 or active_values[index - 1] is not True):
            return times[index] if index < len(times) else None, "row.stimulus_active"
    for index, value in enumerate(magnitudes):
        if value is not None and value > 0.0 and (index == 0 or not magnitudes[index - 1] or magnitudes[index - 1] <= 0.0):
            return times[index] if index < len(times) else None, "row.stimulus_magnitude"
    return None, None


def _onset_from_values(
    times: Sequence[float],
    values: Sequence[float | None],
    *,
    start_index: int = 0,
    delta: float,
) -> tuple[float | None, int | None, float | None]:
    finite_values = [value for value in values[: max(0, start_index)] if value is not None]
    if not finite_values:
        finite_values = [value for value in values if value is not None][: max(1, min(3, len(values)))]
    if not finite_values:
        return None, None, None
    baseline = sum(finite_values) / len(finite_values)
    for index in range(max(0, start_index), len(values)):
        value = values[index]
        if value is None:
            continue
        if value > baseline + delta:
            return times[index], index, baseline
    return None, None, baseline


def _motion_values(rows: Sequence[Mapping[str, Any]], positions: Sequence[tuple[float, float, float] | None]) -> list[float | None]:
    first = next((value for value in positions if value is not None), None)
    if first is None:
        return [None] * len(rows)
    result: list[float | None] = []
    for row, position in zip(rows, positions):
        if position is not None:
            result.append(_norm([position[k] - first[k] for k in range(3)]))
            continue
        velocity = _velocity(row)
        result.append(_norm(velocity) if velocity is not None else None)
    return result


def _temporal_evidence(bundle: ExperimentBundle, thresholds: EscapeThresholds) -> dict[str, Any]:
    times, time_errors = _row_times(bundle)
    if time_errors:
        return {
            "passed": False,
            "reasons": time_errors,
            "times": times,
            "stimulus_onset_s": None,
            "neural_onset_s": None,
            "motor_onset_s": None,
            "motion_onset_s": None,
        }
    neural_values: list[float | None] = []
    motor_values: list[float | None] = []
    positions: list[tuple[float, float, float] | None] = []
    neural_paths: list[str | None] = []
    motor_paths: list[str | None] = []
    for row in bundle.rows:
        neural, neural_path = _neural_value(row)
        motor, motor_path = _motor_value(row)
        neural_values.append(neural)
        motor_values.append(motor)
        positions.append(_position(row))
        neural_paths.append(neural_path)
        motor_paths.append(motor_path)
    stimulus_onset, stimulus_source = _stimulus_onset(bundle, times)
    reasons: list[str] = []
    neural_reasons: list[str] = []
    motor_reasons: list[str] = []
    motion_reasons: list[str] = []
    if stimulus_onset is None:
        neural_reasons.append("stimulus onset is not explicitly present")
    if not any(value is not None for value in neural_values):
        neural_reasons.append("no neural activity/spike/rate channel")
    if not any(value is not None for value in motor_values):
        motor_reasons.append("no motor/control channel")
    if not any(value is not None for value in positions):
        motion_reasons.append("no body position channel")
    if neural_reasons or motor_reasons or motion_reasons:
        reasons.extend(neural_reasons)
        reasons.extend(motor_reasons)
        reasons.extend(motion_reasons)
        return {
            "passed": False,
            "reasons": reasons,
            "neural_reasons": neural_reasons,
            "motor_reasons": motor_reasons,
            "motion_reasons": motion_reasons,
            "times": times,
            "stimulus_onset_s": stimulus_onset,
            "stimulus_source": stimulus_source,
            "neural_onset_s": None,
            "motor_onset_s": None,
            "motion_onset_s": None,
            "neural_values": neural_values,
            "motor_values": motor_values,
        }
    start_index = 0
    if stimulus_onset is not None:
        start_index = next((index for index, time in enumerate(times) if time >= stimulus_onset), 0)
    neural_onset, neural_index, neural_baseline = _onset_from_values(
        times, neural_values, start_index=start_index, delta=thresholds.neural_response_delta
    )
    motor_onset, motor_index, motor_baseline = _onset_from_values(
        times, motor_values, start_index=start_index, delta=thresholds.motor_response_delta
    )
    motion_values = _motion_values(bundle.rows, positions)
    motion_onset, motion_index, motion_baseline = _onset_from_values(
        times, motion_values, start_index=start_index, delta=thresholds.motion_displacement_m
    )
    if neural_onset is None:
        neural_reasons.append("neural activity does not rise above its pre-stimulus baseline")
    if motor_onset is None:
        motor_reasons.append("motor output does not rise above its baseline")
    if motion_onset is None:
        motion_reasons.append("body motion does not rise above its baseline")
    if neural_onset is not None and stimulus_onset is not None and neural_onset <= stimulus_onset:
        # A sample exactly at onset is allowed only when the row carries a
        # post-onset time; a response pre-dating onset is not causal.
        if neural_onset < stimulus_onset:
            neural_reasons.append("neural response precedes the declared stimulus onset")
    if neural_index is not None and motor_index is not None and neural_index >= motor_index:
        motor_reasons.append("motor onset is not strictly after neural onset")
    if motor_index is not None and motion_index is not None and motor_index >= motion_index:
        motion_reasons.append("body motion is not strictly after motor onset")
    if neural_index is not None and motion_index is not None and neural_index >= motion_index:
        motion_reasons.append("neural activity does not precede body motion")
    # A declared motor source is needed in addition to a numeric value.  A
    # plain position/setpoint trace must not be promoted to motor transfer.
    source_values: list[str] = []
    for row in bundle.rows:
        source, _ = _lookup_normalised(row, ("motor_source", "control_source", "source", "neural_source"))
        if isinstance(source, str) and source.strip():
            source_values.append(source.strip())
        motor, _ = _lookup_normalised(row, ("motor", "control", "command"), nested=False)
        if isinstance(motor, Mapping):
            nested_source, _ = _lookup_normalised(motor, ("source", "neural_source", "provenance"), nested=False)
            if isinstance(nested_source, str) and nested_source.strip():
                source_values.append(nested_source.strip())
    if not source_values:
        motor_reasons.append("motor/control provenance source is missing")
    reasons.extend(neural_reasons)
    reasons.extend(motor_reasons)
    reasons.extend(motion_reasons)
    return {
        "passed": not reasons,
        "reasons": reasons,
        "neural_reasons": neural_reasons,
        "motor_reasons": motor_reasons,
        "motion_reasons": motion_reasons,
        "times": times,
        "stimulus_onset_s": stimulus_onset,
        "stimulus_source": stimulus_source,
        "neural_onset_s": neural_onset,
        "motor_onset_s": motor_onset,
        "motion_onset_s": motion_onset,
        "neural_index": neural_index,
        "motor_index": motor_index,
        "motion_index": motion_index,
        "neural_baseline": neural_baseline,
        "motor_baseline": motor_baseline,
        "motion_baseline": motion_baseline,
        "neural_values": neural_values,
        "motor_values": motor_values,
        "motion_values": motion_values,
        "neural_paths": neural_paths,
        "motor_paths": motor_paths,
        "motor_sources": sorted(set(source_values)),
    }


def _network_audit(bundle: ExperimentBundle, thresholds: EscapeThresholds) -> dict[str, Any]:
    active_values: list[float] = []
    saturation_values: list[float] = []
    global_saturated_rows: list[int] = []
    missing_rows: list[int] = []
    paths: list[str] = []
    for index, row in enumerate(bundle.rows):
        active, saturated, global_saturated, path = _neural_fraction(row)
        if path:
            paths.append(path)
        if active is None or (saturated is None and global_saturated is None):
            missing_rows.append(index)
        if active is not None:
            active_values.append(active)
        if saturated is not None:
            saturation_values.append(saturated)
        if global_saturated is True:
            global_saturated_rows.append(index)
    reasons: list[str] = []
    if missing_rows:
        reasons.append(
            "network-wide neuron count/active fraction/saturation evidence is missing at rows "
            + ", ".join(str(index) for index in missing_rows[:5])
        )
    if not active_values:
        reasons.append("no finite global active-fraction evidence")
    if not saturation_values:
        reasons.append("no finite global saturation-fraction evidence")
    max_active = max(active_values) if active_values else None
    max_saturation = max(saturation_values) if saturation_values else None
    if max_active is not None and max_active > thresholds.max_global_active_fraction:
        reasons.append(
            f"global active fraction {max_active:.6g} exceeds {thresholds.max_global_active_fraction:.6g}"
        )
    if max_saturation is not None and max_saturation > thresholds.max_global_saturation_fraction:
        reasons.append(
            f"global saturation fraction {max_saturation:.6g} exceeds {thresholds.max_global_saturation_fraction:.6g}"
        )
    if global_saturated_rows:
        reasons.append("network declares global saturation at rows " + ", ".join(map(str, global_saturated_rows[:5])))
    return {
        "passed": not reasons,
        "reasons": reasons,
        "max_global_active_fraction": max_active,
        "max_global_saturation_fraction": max_saturation,
        "global_saturation_rows": global_saturated_rows,
        "missing_rows": missing_rows,
        "paths": sorted(set(paths)),
    }


def _authority_body_declaration(manifest: Mapping[str, Any]) -> tuple[bool, list[str], Mapping[str, Any] | None]:
    declaration: Mapping[str, Any] | None = None
    for key in ("body_state_authority", "pose_authority", "body_integrator", "body", "dynamics"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            declaration = value
            break
    reasons: list[str] = []
    if declaration is None:
        return False, ["manifest lacks body-state/integrator authority declaration"], None
    source = None
    for key in ("source", "state_source", "pose_source", "authority", "integrator"):
        if key in declaration:
            source = declaration[key]
            break
    if source is None or str(source).strip().lower() not in {"integrator", "physics", "dynamics", "body_integrator", "simulated_body", "handbuilt_integrator"}:
        reasons.append("body state source is not explicitly an integrator/physics state")
    for key in ("direct_pose_write", "writes_pose", "pose_write", "teleport", "teleports", "set_position_directly"):
        if key in declaration and not _declared_off(declaration[key]):
            reasons.append(f"body declaration permits {key}")
    # An explicit empty event list is preferable to inferring absence from no
    # field.  Require either the declaration flags or event lists.
    has_event_evidence = any(key in declaration for key in ("direct_pose_write", "pose_write_events", "teleport_events", "teleports", "pose_writes"))
    if not has_event_evidence:
        reasons.append("body declaration has no explicit pose-write/teleport evidence")
    return not reasons, reasons, declaration


def _body_integrity_audit(bundle: ExperimentBundle, temporal: Mapping[str, Any], thresholds: EscapeThresholds) -> dict[str, Any]:
    passed_declaration, declaration_reasons, declaration = _authority_body_declaration(bundle.manifest)
    direct_events: list[str] = []
    numeric_violations: list[str] = []
    positions = [_position(row) for row in bundle.rows]
    velocities = [_velocity(row) for row in bundle.rows]
    times = list(temporal.get("times", ()))
    for index, row in enumerate(bundle.rows):
        for key in ("direct_pose_write", "pose_write", "teleport", "teleported", "set_position_directly"):
            if key in row and not _declared_off(row[key]):
                direct_events.append(f"row {index}.{key}")
        for key in ("pose_source", "position_source", "state_source", "body_state_source"):
            if key in row:
                source = str(row[key]).strip().lower()
                if source not in {"integrator", "physics", "dynamics", "body_integrator", "simulated_body", "handbuilt_integrator"}:
                    direct_events.append(f"row {index}.{key}={source}")
        events, _ = _lookup_normalised(row, ("events", "state_events"), nested=False)
        if isinstance(events, list):
            for event in events:
                if isinstance(event, Mapping):
                    kind = str(event.get("type", event.get("kind", ""))).lower()
                    if any(token in kind for token in ("teleport", "pose_write", "set_position", "direct_pose")):
                        direct_events.append(f"row {index}.events.{kind}")
                elif isinstance(event, str) and any(token in event.lower() for token in ("teleport", "pose_write", "set_position", "direct_pose")):
                    direct_events.append(f"row {index}.events.{event}")
    for index in range(1, min(len(positions), len(times))):
        if positions[index] is None or positions[index - 1] is None:
            continue
        dt = times[index] - times[index - 1]
        if dt <= 0.0:
            continue
        step = _norm([positions[index][axis] - positions[index - 1][axis] for axis in range(3)])  # type: ignore[index]
        velocity = velocities[index - 1] or velocities[index]
        if velocity is None:
            numeric_violations.append(f"row {index} has position change but no velocity")
            continue
        speed = _norm(velocity)
        max_step = speed * dt + 0.5 * thresholds.max_integration_accel_mps2 * dt * dt + thresholds.integration_tolerance_m
        if step > max_step:
            numeric_violations.append(
                f"row {index} displacement {step:.6g} m exceeds integrated bound {max_step:.6g} m"
            )
    reasons = list(declaration_reasons)
    if direct_events:
        reasons.append("direct pose write/teleport event observed: " + ", ".join(direct_events[:5]))
    if numeric_violations:
        reasons.extend(numeric_violations)
    return {
        "passed": passed_declaration and not direct_events and not numeric_violations,
        "reasons": reasons,
        "declaration": declaration,
        "direct_pose_events": direct_events,
        "numeric_violations": numeric_violations,
    }


def _ground_and_escape_audit(bundle: ExperimentBundle, thresholds: EscapeThresholds) -> dict[str, Any]:
    rows = bundle.rows
    positions = [_position(row) for row in rows]
    times, time_errors = _row_times(bundle)
    reasons: list[str] = list(time_errors)
    ground = None
    for source in (bundle.manifest, bundle.metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("ground_plane_z_m", "ground_z_m", "floor_z_m", "ground_plane"):
            ground = _finite(source.get(key))
            if ground is not None:
                break
        if ground is not None:
            break
        world = source.get("world")
        if isinstance(world, Mapping):
            ground = _finite(world.get("ground_plane_z_m", world.get("ground_z_m")))
            if ground is not None:
                break
    if ground is None:
        reasons.append("ground plane z is not declared")
    if any(position is None for position in positions):
        reasons.append("body position is missing/non-finite in one or more rows")

    contacts = [_contact(row) for row in rows]
    collisions = [_collision(row) for row in rows]
    interaction = _stimulus_interaction_contract(bundle)
    collision_channel_present = bool(collisions) and all(present for _, present in collisions)
    collision_observed = any(present and value for value, present in collisions)
    if any(value is None for value in contacts):
        reasons.append("contact/grounded channel is missing in one or more rows")
    if not interaction["declared"]:
        reasons.append(
            "stimulus interaction contract is missing (declare visual_only/non_interacting or interactive/collidable)"
        )
    if interaction["collision_required"] and not collision_channel_present:
        reasons.append("physics collision channel is missing for an interactive stimulus")

    liftoff_index: int | None = None
    if ground is not None:
        for index, (position, contact) in enumerate(zip(positions, contacts)):
            if position is None:
                continue
            airborne = contact is False
            if airborne and position[2] > ground + thresholds.liftoff_height_m:
                liftoff_index = index
                break
    collision_index = next((index for index, (value, present) in enumerate(collisions) if present and value), None)
    liftoff_time = times[liftoff_index] if liftoff_index is not None and liftoff_index < len(times) else None
    collision_time = times[collision_index] if collision_index is not None and collision_index < len(times) else None
    if liftoff_index is None:
        reasons.append("no explicit ground-to-air liftoff transition")
    initial_grounded = bool(
        contacts
        and contacts[0] is True
        and positions[0] is not None
        and ground is not None
        and positions[0][2] <= ground + thresholds.liftoff_height_m
    )
    if not initial_grounded:
        reasons.append("trace does not begin in a declared grounded state")
    if collision_index is not None:
        collision_label = "visual-only stimulus reports a physics collision" if interaction["visual_only"] else "physics collision observed"
        reasons.append(
            collision_label
            + (f" at t={collision_time:.6g}s" if collision_time is not None else "")
        )
    if collision_index is not None and liftoff_index is not None and liftoff_index >= collision_index:
        reasons.append("body collides before it leaves the ground")

    timing = _timing_deadlines(bundle, thresholds, positions)
    decision_deadline = timing.get("decision_deadline_s")
    planned_contact = timing.get("planned_contact_time_s")
    duration_contract = _configured_duration(bundle)
    configured_duration = duration_contract.get("duration_s")
    if decision_deadline is None:
        reasons.append("configured protocol decision/liftoff deadline is missing")
    if planned_contact is None:
        reasons.append("planned physical contact deadline is missing")
    if configured_duration is None:
        reasons.append("configured full simulation duration is missing")
    if times:
        required_end = None
        if decision_deadline is not None:
            required_end = decision_deadline
        if planned_contact is not None:
            required_end = planned_contact if required_end is None else max(required_end, planned_contact)
        if configured_duration is not None:
            required_end = configured_duration if required_end is None else max(required_end, configured_duration)
        if required_end is not None and times[-1] < required_end - thresholds.time_tolerance_s:
            reasons.append(
                f"trajectory ends at {times[-1]:.6g}s before the required full duration/contact deadline {required_end:.6g}s"
            )
    if liftoff_time is not None and decision_deadline is not None and liftoff_time >= decision_deadline:
        reasons.append(
            f"liftoff at {liftoff_time:.6g}s is not before configured decision deadline {decision_deadline:.6g}s"
        )
    if liftoff_time is not None and planned_contact is not None and liftoff_time >= planned_contact:
        reasons.append(
            f"liftoff at {liftoff_time:.6g}s is not before planned physical contact {planned_contact:.6g}s"
        )

    vertical_clearance = None
    if liftoff_index is not None and liftoff_index < len(positions) and positions[liftoff_index] is not None and ground is not None:
        vertical_clearance = positions[liftoff_index][2] - ground  # type: ignore[index]
    if vertical_clearance is None:
        reasons.append("vertical liftoff clearance cannot be computed")
    elif vertical_clearance <= thresholds.liftoff_height_m:
        reasons.append(
            f"vertical liftoff clearance {vertical_clearance:.6g} m is not > {thresholds.liftoff_height_m:.6g} m"
        )

    # The initial physical gate is deliberately about a timely vertical jump
    # before the counterfactual contact deadline.  For visual-only stimuli no
    # physics collision channel is required.  A later horizontal flight away
    # from the stimulus is a separate, stronger metric and must not turn an
    # otherwise valid initial jump into a PHYSICAL_ESCAPE failure.
    stimulus = _stimulus_mapping(bundle.manifest, bundle.metadata)
    away_vector: tuple[float, float, float] | None = None
    direction_source: str | None = None
    if isinstance(stimulus, Mapping):
        for key in ("away_direction_m", "escape_direction_m", "expected_escape_direction_m"):
            away_vector = _unit(_vector(stimulus.get(key)) or ())
            if away_vector is not None:
                direction_source = f"stimulus.{key}"
                break
        if away_vector is None:
            for key in (
                "position_m",
                "source_position_m",
                "start_position_m",
                "threat_position_m",
                "stimulus_position_m",
            ):
                source_position = _vector(stimulus.get(key))
                if source_position is not None and positions and positions[0] is not None:
                    away_vector = _unit([positions[0][axis] - source_position[axis] for axis in range(3)])
                    if away_vector is not None:
                        direction_source = f"stimulus.{key}->body"
                        break
            if away_vector is None:
                approach = _vector(stimulus.get("approach_direction_m", stimulus.get("direction_m")))
                if approach is not None:
                    away_vector = _unit([-item for item in approach])
                    # A velocity/approach vector points from the threat toward
                    # the body; inverting it gives the direction away from
                    # the threat.  A protocol may instead declare
                    # direction_semantics=away, in which case retain it.
                    semantics = str(stimulus.get("direction_semantics", "approach")).lower()
                    if semantics in {"away", "escape", "source_to_body"}:
                        away_vector = _unit(approach)
                        direction_source = "stimulus.direction_m (declared away)"
                    else:
                        direction_source = "stimulus.approach/direction_m (inverted)"
    if away_vector is None:
        for source in (bundle.manifest, bundle.metadata):
            if not isinstance(source, Mapping):
                continue
            for key in ("away_direction_m", "escape_direction_m", "expected_escape_direction_m"):
                vector = _unit(_vector(source.get(key)) or ())
                if vector is not None:
                    away_vector = vector
                    direction_source = key
                    break
            if away_vector is not None:
                break

    displacement: tuple[float, float, float] | None = None
    full_projection = None
    horizontal_away_vector = None
    horizontal_projection = None
    if positions and positions[0] is not None and positions[-1] is not None:
        displacement = tuple(positions[-1][axis] - positions[0][axis] for axis in range(3))  # type: ignore[index]
        if away_vector is not None:
            full_projection = _dot(displacement, away_vector)
            horizontal_away_vector = _unit((away_vector[0], away_vector[1], 0.0))
            if horizontal_away_vector is not None:
                horizontal_projection = _dot(
                    (displacement[0], displacement[1], 0.0), horizontal_away_vector
                )

    horizontal_reasons: list[str] = []
    if away_vector is None:
        horizontal_reasons.append("stimulus source/away direction is not declared")
    elif horizontal_away_vector is None:
        horizontal_reasons.append("stimulus away direction has no horizontal component")
    elif horizontal_projection is None:
        horizontal_reasons.append("cannot compute final horizontal displacement projection")
    elif horizontal_projection <= thresholds.escape_projection_m:
        horizontal_reasons.append(
            f"horizontal displacement projection away from stimulus {horizontal_projection:.6g} m is not > {thresholds.escape_projection_m:.6g} m"
        )
    horizontal_passed = not horizontal_reasons
    vertical_passed = not reasons
    duration_covered = bool(
        configured_duration is not None
        and times
        and times[-1] >= configured_duration - thresholds.time_tolerance_s
    )
    counterfactual_contact_reached = bool(
        planned_contact is not None
        and times
        and times[-1] >= planned_contact - thresholds.time_tolerance_s
    )
    return {
        "passed": vertical_passed,
        "reasons": list(dict.fromkeys(reasons)),
        "ground_plane_z_m": ground,
        "initial_grounded": initial_grounded,
        "liftoff_index": liftoff_index,
        "liftoff_time_s": liftoff_time,
        "collision_index": collision_index,
        "collision_time_s": collision_time,
        # ``None`` means no physics collision channel was required/observed
        # for a visual-only stimulus; it is intentionally not a claim that a
        # collision-free physics trace was measured.
        "collision_free": (
            None
            if interaction["visual_only"] and not collision_channel_present
            else collision_index is None and collision_channel_present
        ),
        "stimulus_interaction": interaction,
        "interaction_mode": interaction["mode"],
        "physics_collision_observed": collision_observed,
        "physics_collision_channel_present": collision_channel_present,
        "physics_collision": {
            "mode": interaction["mode"],
            "observed": collision_observed,
            "channel_present": collision_channel_present,
            "channel_required": interaction["collision_required"],
            "passed": not collision_observed and (
                collision_channel_present or not interaction["collision_required"]
            ),
            "contract": interaction,
        },
        "liftoff_before_collision": (
            liftoff_index is not None and (collision_index is None or liftoff_index < collision_index)
        ),
        "decision_deadline_s": decision_deadline,
        "decision_deadline_source": timing.get("decision_deadline_source"),
        "planned_contact_time_s": planned_contact,
        "planned_contact_time_source": timing.get("planned_contact_time_source"),
        "timing_geometry": timing.get("geometry"),
        "configured_duration_s": configured_duration,
        "configured_duration_source": duration_contract.get("source"),
        "configured_duration_candidates": duration_contract.get("candidates", []),
        "duration_coverage": {
            "configured_duration_s": configured_duration,
            "source": duration_contract.get("source"),
            "trace_end_time_s": times[-1] if times else None,
            "covered": duration_covered,
            "passed": duration_covered,
        },
        "trace_end_time_s": times[-1] if times else None,
        "vertical_clearance_m": vertical_clearance,
        "counterfactual_contact_time_s": planned_contact,
        "counterfactual_overlap_reached": counterfactual_contact_reached,
        "counterfactual_contact": {
            "time_s": planned_contact,
            "source": timing.get("planned_contact_time_source"),
            "trace_reached": counterfactual_contact_reached,
            "overlap_is_counterfactual": True,
            "is_physics_collision": False,
            "physics_collision_observed": collision_observed,
        },
        "initial_jump_avoidance": {
            "status": STATUS_PASS if vertical_passed else STATUS_FAIL,
            "passed": vertical_passed,
            "reasons": list(dict.fromkeys(reasons)),
            "liftoff_time_s": liftoff_time,
            "vertical_clearance_m": vertical_clearance,
            "collision_free": (
                None
                if interaction["visual_only"] and not collision_channel_present
                else collision_index is None and collision_channel_present
            ),
            "decision_deadline_s": decision_deadline,
            "planned_contact_time_s": planned_contact,
            "configured_duration_s": configured_duration,
            "physics_collision": {
                "observed": collision_observed,
                "channel_present": collision_channel_present,
                "channel_required": interaction["collision_required"],
            },
        },
        "away_direction": away_vector,
        "away_direction_source": direction_source,
        "net_displacement_m": displacement,
        # Keep the historical key as the full-vector projection, but expose
        # the horizontal metric explicitly.  Controls use the latter.
        "net_displacement_projection_m": full_projection,
        "horizontal_away_flight": {
            "status": STATUS_PASS if horizontal_passed else STATUS_FAIL,
            "passed": horizontal_passed,
            "reasons": horizontal_reasons,
            "away_direction_horizontal": horizontal_away_vector,
            "projection_m": horizontal_projection,
            "threshold_m": thresholds.escape_projection_m,
        },
    }


def _status(name: str, passed: bool, reasons: Sequence[str], checks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "established": bool(passed),
        "reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "checks": dict(checks or {}),
    }


def _validate_one(bundle: ExperimentBundle, thresholds: EscapeThresholds) -> dict[str, Any]:
    authority = _authority_audit(bundle)
    leakage = _stimulus_leakage_audit(bundle)
    telemetry = _telemetry_audit(bundle, thresholds)
    temporal = _temporal_evidence(bundle, thresholds)
    network = _network_audit(bundle, thresholds)
    body = _body_integrity_audit(bundle, temporal, thresholds)
    physical = _ground_and_escape_audit(bundle, thresholds)

    common_reasons: list[str] = []
    if not authority["passed"]:
        common_reasons.append("behavior-authority audit failed")
    if not leakage["passed"]:
        common_reasons.append("stimulus metadata lineage audit failed")
    if not telemetry["passed"]:
        common_reasons.append("trajectory/telemetry consistency audit failed")
    neural_reasons = common_reasons + list(temporal.get("reasons", ())) + list(network.get("reasons", ()))
    neural = _status(
        NEURAL_RESPONSE,
        authority["passed"] and leakage["passed"] and telemetry["passed"] and temporal.get("stimulus_onset_s") is not None
        and temporal.get("neural_onset_s") is not None
        and temporal.get("neural_index") is not None
        and not list(temporal.get("neural_reasons", ()))
        and network["passed"],
        common_reasons + list(temporal.get("neural_reasons", ())) + list(network.get("reasons", ())),
        {"authority": authority, "stimulus_lineage": leakage, "telemetry": telemetry, "temporal": temporal, "network": network},
    )
    motor_reasons = common_reasons + list(temporal.get("motor_reasons", ()))
    motor_pass = (
        authority["passed"] and leakage["passed"] and telemetry["passed"] and temporal.get("neural_index") is not None
        and temporal.get("motor_index") is not None
        and temporal.get("neural_index") < temporal.get("motor_index")
        and bool(temporal.get("motor_sources"))
    )
    motor = _status(MOTOR_TRANSFER, motor_pass, motor_reasons, {"authority": authority, "stimulus_lineage": leakage, "telemetry": telemetry, "temporal": temporal, "body_integrity": body})
    physical_reasons = common_reasons + list(temporal.get("motion_reasons", ())) + list(physical.get("reasons", ())) + list(body.get("reasons", ())) + list(network.get("reasons", ()))
    physical_pass = (
        authority["passed"] and leakage["passed"] and telemetry["passed"] and temporal.get("neural_index") is not None
        and temporal.get("motion_index") is not None
        and temporal.get("neural_index") < temporal.get("motion_index")
        and physical["passed"] and body["passed"] and network["passed"]
    )
    physical_status = _status(
        PHYSICAL_ESCAPE,
        physical_pass,
        physical_reasons,
        {
            "authority": authority,
            "stimulus_lineage": leakage,
            "telemetry": telemetry,
            "temporal": temporal,
            "network": network,
            "body_integrity": body,
            "ground_escape": physical,
            "initial_jump_avoidance": physical.get("initial_jump_avoidance"),
            "horizontal_away_flight": physical.get("horizontal_away_flight"),
        },
    )
    return {
        "condition": bundle.condition,
        "source": bundle.source,
        "n_rows": len(bundle.rows),
        "time_inferred": bundle.time_inferred,
        "authority": authority,
        "stimulus_lineage": leakage,
        "telemetry": telemetry,
        "temporal": temporal,
        "network": network,
        "body_integrity": body,
        "ground_escape": physical,
        "categories": {
            NEURAL_RESPONSE: neural,
            MOTOR_TRANSFER: motor,
            PHYSICAL_ESCAPE: physical_status,
        },
        "summary": {
            "neural_response": neural["established"],
            "motor_transfer": motor["established"],
            "physical_escape": physical_status["established"],
            "net_displacement_projection_m": physical.get("net_displacement_projection_m"),
            "horizontal_away_projection_m": physical.get("horizontal_away_flight", {}).get("projection_m"),
            "horizontal_away_flight": physical.get("horizontal_away_flight", {}).get("passed", False),
            "liftoff_time_s": physical.get("liftoff_time_s"),
            "decision_deadline_s": physical.get("decision_deadline_s"),
            "planned_contact_time_s": physical.get("planned_contact_time_s"),
            "configured_duration_s": physical.get("configured_duration_s"),
            "counterfactual_contact_reached": physical.get("counterfactual_contact", {}).get("trace_reached", False),
            "physics_collision_observed": physical.get("physics_collision_observed", False),
            "neural_onset_s": temporal.get("neural_onset_s"),
            "motor_onset_s": temporal.get("motor_onset_s"),
            "motion_onset_s": temporal.get("motion_onset_s"),
        },
    }


def _extract_seed(bundle: ExperimentBundle) -> Any:
    for source in (bundle.manifest, bundle.metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("seed", "random_seed", "rng_seed"):
            if key in source and source[key] is not None:
                value = source[key]
                if isinstance(value, (int, str)) and not isinstance(value, bool):
                    return value
                number = _finite(value)
                if number is not None:
                    return int(number) if number.is_integer() else number
    return None


def _normalised_stimulus_for_role(bundle: ExperimentBundle, role: str) -> Any:
    """Return stimulus metadata with only the declared intervention removed.

    A mirror batch is allowed to change the signed source geometry; onset,
    size, timing and other matched fields must still agree.  Sham and
    lesion/ablation batches retain the stimulus geometry and therefore use
    the stricter normalisation above.
    """

    stimulus = _stimulus_mapping(bundle.manifest, bundle.metadata)
    if stimulus is None:
        return None
    if role != "mirror":
        return _strip_for_match(stimulus)
    value = _strip_for_match(stimulus)
    if isinstance(value, Mapping):
        result = dict(value)
        for key in list(result):
            normalized = _normal_key(key)
            if normalized in {
                "positionm",
                "sourcepositionm",
                "threatpositionm",
                "stimuluspositionm",
                "awaydirectionm",
                "escapedirectionm",
                "expectedescapedirectionm",
                "approachdirectionm",
            }:
                result.pop(key, None)
        return result
    return value


def _strip_for_match(value: Any, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            normalized = _normal_key(raw_key)
            if normalized in {
                "condition", "variant", "batchid", "runid", "intervention", "ablation", "lesion",
                "mirror", "mirrored", "label", "name", "id", "mode", "kind", "type",
                "renderable", "collidable", "texture",
            }:
                continue
            result[str(raw_key)] = _strip_for_match(child, str(raw_key))
        return result
    if isinstance(value, list):
        return [_strip_for_match(item, key) for item in value]
    number = _finite(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return value


def _condition_name(bundle: ExperimentBundle, fallback: str | None = None) -> str:
    if bundle.condition:
        return bundle.condition.strip().lower()
    for source in (bundle.manifest, bundle.metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("condition", "variant", "batch", "batch_id", "intervention", "label", "run_label"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return (fallback or "candidate").strip().lower()


def _inline_bundle(value: Any, parent: ExperimentBundle, condition: str | None = None) -> ExperimentBundle:
    if isinstance(value, ExperimentBundle):
        if condition is not None:
            value.condition = condition
        return value
    if isinstance(value, (str, Path)):
        try:
            return load_bundle(value, condition=condition)
        except EscapeValidationError:
            # A block manifest commonly names one shared immutable protocol
            # manifest while controls are exported as trajectory-only files.
            # Reuse that manifest only as an explicit parent contract; the
            # comparison still requires same-seed/grid/stimulus evidence.
            local_manifest = dict(parent.manifest)
            if condition is not None:
                local_manifest["condition"] = condition
            return load_bundle(value, local_manifest, condition=condition)
    if isinstance(value, Mapping):
        if any(key in value for key in ("trajectory", "rows", "records", "samples", "telemetry")):
            manifest_value = value.get("manifest")
            local_manifest: Any = (
                manifest_value
                if isinstance(manifest_value, (str, Path, Mapping))
                else dict(parent.manifest)
            )
            if condition is not None:
                if isinstance(local_manifest, Mapping):
                    local_manifest = dict(local_manifest)
                    local_manifest["condition"] = condition
            return load_bundle(value, local_manifest, condition=condition)
        rows = value.get("data")
        if isinstance(rows, list):
            local_manifest = dict(parent.manifest)
            local_manifest["condition"] = condition or value.get("condition", "control")
            return load_bundle(rows, local_manifest, condition=condition)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        local_manifest = dict(parent.manifest)
        local_manifest["condition"] = condition or "control"
        return load_bundle(value, local_manifest, condition=condition)
    raise EscapeValidationError("control batch must be a path, trajectory envelope or row list")


def _control_entries(candidate: ExperimentBundle, controls: Any) -> list[tuple[str, ExperimentBundle]]:
    entries: list[tuple[str, ExperimentBundle]] = [("candidate", candidate)]
    values: Any = controls
    if values is None:
        for source in (candidate.manifest, candidate.metadata):
            if isinstance(source, Mapping):
                for key in ("batches", "controls", "matched_controls", "comparison_runs"):
                    if key in source:
                        values = source[key]
                        break
            if values is not None:
                break
    if values is None:
        # A combined telemetry export may carry several pre-labelled batches
        # in one row stream.  Split only on explicit batch/condition fields;
        # ordinary scenario/phase labels are not comparison batches.
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in candidate.rows:
            label = None
            for key in ("batch_id", "trial_condition", "condition", "variant", "intervention"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    label = value.strip()
                    break
            if label is None:
                grouped = {}
                break
            grouped.setdefault(label, []).append(row)
        if len(grouped) > 1:
            values = {
                label: {"trajectory": rows}
                for label, rows in grouped.items()
            }
    if values is None:
        return entries
    if isinstance(values, Mapping):
        if any(key in values for key in ("batches", "controls", "runs")):
            for key in ("batches", "controls", "runs"):
                if key in values:
                    values = values[key]
                    break
        else:
            values = list(values.items())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        iterable = list(values)
    else:
        iterable = [values]
    for index, item in enumerate(iterable):
        label = f"control-{index}"
        value = item
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
            label, value = item[0], item[1]
        elif isinstance(item, Mapping):
            for key in ("condition", "variant", "label", "name", "batch_id", "id", "intervention"):
                if isinstance(item.get(key), str) and item[key].strip():
                    label = item[key].strip()
                    break
            if any(key in item for key in ("trajectory", "rows", "records", "samples", "data", "path", "directory")):
                value = item.get("trajectory", item.get("rows", item.get("records", item.get("data", item.get("path", item.get("directory"))))))
                if isinstance(item.get("manifest"), (str, Path, Mapping)):
                    value = {"trajectory": value, "manifest": item["manifest"]}
        try:
            bundle = _inline_bundle(value, candidate, label)
        except EscapeValidationError:
            # Preserve the malformed control as a synthetic empty bundle in
            # the report; the causal gate will fail closed with its label.
            local_manifest = dict(candidate.manifest)
            local_manifest["condition"] = label
            bundle = ExperimentBundle(rows=(), manifest=local_manifest, condition=label)
        entries.append((label, bundle))
    return entries


def _compare_batches(entries: Sequence[tuple[str, ExperimentBundle]], results: Mapping[str, Mapping[str, Any]], thresholds: EscapeThresholds) -> dict[str, Any]:
    # Normalise aliases while retaining original labels in the report.
    categories: dict[str, tuple[str, ExperimentBundle, Mapping[str, Any]]] = {}
    for label, bundle in entries:
        condition = _condition_name(bundle, label)
        result = results.get(label, {})
        if condition == "candidate" and "candidate" not in categories:
            categories["candidate"] = (label, bundle, result)
        elif condition not in categories:
            categories[condition] = (label, bundle, result)
    def find(predicate: Any) -> tuple[str, ExperimentBundle, Mapping[str, Any]] | None:
        for condition, item in categories.items():
            if predicate(condition):
                return item
        return None
    candidate = categories.get("candidate") or find(lambda name: any(token in name for token in ("stimulus", "full", "escape", "control")))
    sham = find(lambda name: name in {"sham", "matched", "matched_control", "no_stimulus", "nostimulus", "control_sham"} or "sham" in name or "matched" in name)
    mirror = find(lambda name: "mirror" in name or "counter" in name)
    lesion = find(lambda name: "lesion" in name or "ablation" in name or "knockout" in name or "silenc" in name)
    missing: list[str] = []
    if candidate is None:
        missing.append("candidate/stimulus")
    if sham is None:
        missing.append("sham_or_matched_control")
    if mirror is None:
        missing.append("mirror_control")
    if lesion is None:
        missing.append("lesion_or_ablation_control")
    reasons: list[str] = []
    pairwise: dict[str, Any] = {}
    if missing:
        reasons.append("missing required comparison batches: " + ", ".join(missing))
    reference = candidate
    if reference is not None:
        ref_label, ref_bundle, ref_result = reference
        ref_times, ref_time_errors = _row_times(ref_bundle)
        ref_seed = _extract_seed(ref_bundle)
        for role, item in (("sham", sham), ("mirror", mirror), ("lesion_or_ablation", lesion)):
            if item is None:
                continue
            label, bundle, result = item
            times, time_errors = _row_times(bundle)
            same_seed = ref_seed is not None and _extract_seed(bundle) == ref_seed
            same_time_grid = (
                not ref_time_errors and not time_errors and len(ref_times) == len(times)
                and all(abs(a - b) <= thresholds.time_tolerance_s for a, b in zip(ref_times, times))
            )
            ref_stimulus = _normalised_stimulus_for_role(ref_bundle, role)
            same_stimulus = ref_stimulus is not None and ref_stimulus == _normalised_stimulus_for_role(bundle, role)
            ref_pos = _position(ref_bundle.rows[0]) if ref_bundle.rows else None
            pos = _position(bundle.rows[0]) if bundle.rows else None
            same_initial_body = ref_pos is not None and pos is not None and _norm([ref_pos[i] - pos[i] for i in range(3)]) <= thresholds.integration_tolerance_m
            passed = same_seed and same_time_grid and same_stimulus and same_initial_body and bool(result)
            pairwise[role] = {
                "label": label,
                "same_seed": same_seed,
                "same_time_grid": same_time_grid,
                "same_stimulus_after_intervention": same_stimulus,
                "same_initial_body": same_initial_body,
                "passed": passed,
                "time_errors": time_errors,
            }
            if not passed:
                reasons.append(f"{role} batch is not comparable to candidate")

    def projection(item: tuple[str, ExperimentBundle, Mapping[str, Any]] | None) -> float | None:
        if item is None:
            return None
        ground_escape = item[2].get("ground_escape", {})
        horizontal = ground_escape.get("horizontal_away_flight", {})
        value = horizontal.get("projection_m")
        if value is None:
            # Compatibility with reports produced before the split.  New
            # reports always take the horizontal metric above so a vertical
            # jump cannot manufacture a causal effect.
            value = ground_escape.get("net_displacement_projection_m")
        return _finite(value)
    candidate_projection = projection(candidate)
    sham_projection = projection(sham)
    lesion_projection = projection(lesion)
    candidate_horizontal = bool(
        candidate is not None
        and candidate[2].get("ground_escape", {}).get("horizontal_away_flight", {}).get("passed", False)
    )
    if candidate is not None and not candidate_horizontal:
        reasons.append("candidate horizontal away-flight metric is not established")
    control_delta = None if candidate_projection is None or sham_projection is None else candidate_projection - sham_projection
    lesion_delta = None if candidate_projection is None or lesion_projection is None else candidate_projection - lesion_projection
    if control_delta is None or control_delta <= thresholds.control_effect_projection_m:
        reasons.append("candidate escape projection is not greater than sham/matched control by the required effect")
    if lesion_delta is None or lesion_delta <= thresholds.control_effect_projection_m:
        reasons.append("lesion/ablation does not remove the escape projection")

    mirror_reversal = False
    mirror_axis = None
    if candidate is not None and mirror is not None:
        axis_value = None
        for source in (candidate[1].manifest, candidate[1].metadata):
            if isinstance(source, Mapping):
                axis_value = source.get("mirror_axis_m", source.get("mirror_axis", source.get("escape_axis_m")))
                if axis_value is not None:
                    break
        axis = _unit(_vector(axis_value) or ()) if axis_value is not None else None
        if axis is None:
            axis = (1.0, 0.0, 0.0)
            mirror_axis = "default x axis"
        else:
            mirror_axis = axis_value
        def signed(item: tuple[str, ExperimentBundle, Mapping[str, Any]]) -> float | None:
            rows = item[1].rows
            if not rows:
                return None
            start = _position(rows[0])
            end = _position(rows[-1])
            if start is None or end is None:
                return None
            return _dot([end[i] - start[i] for i in range(3)], axis)  # type: ignore[arg-type]
        candidate_signed = signed(candidate)
        mirror_signed = signed(mirror)
        mirror_reversal = (
            candidate_signed is not None and mirror_signed is not None
            and abs(candidate_signed) > thresholds.control_effect_projection_m
            and abs(mirror_signed) > thresholds.control_effect_projection_m
            and candidate_signed * mirror_signed < 0.0
        )
        if not mirror_reversal:
            reasons.append("mirror batch does not reverse the signed escape displacement")
    else:
        candidate_signed = None
        mirror_signed = None

    # `reasons` includes effect failures; expose the structural result
    # separately because an incompatible batch must not be masked by a strong
    # numerical difference.
    structural_passed = bool(not missing and all(item.get("passed") for item in pairwise.values()))
    causal_passed = (
        structural_passed
        and candidate_horizontal
        and candidate is not None
        and sham is not None
        and mirror is not None
        and lesion is not None
        and not any(
            reason.startswith((
                "candidate escape",
                "candidate horizontal",
                "lesion/ablation",
                "mirror batch",
            ))
            for reason in reasons
        )
    )
    return {
        "passed": causal_passed,
        "structural_comparability": structural_passed,
        "missing_batches": missing,
        "batches": {key: value[0] for key, value in categories.items()},
        "pairwise": pairwise,
        "effects": {
            "candidate_projection_m": candidate_projection,
            "sham_projection_m": sham_projection,
            "lesion_projection_m": lesion_projection,
            "candidate_horizontal_away_flight": candidate_horizontal,
            "candidate_minus_sham_m": control_delta,
            "candidate_minus_lesion_m": lesion_delta,
            "mirror_reversal": mirror_reversal,
            "mirror_axis": mirror_axis,
            "candidate_signed_displacement_m": candidate_signed,
            "mirror_signed_displacement_m": mirror_signed,
        },
        "reasons": list(dict.fromkeys(reasons)),
    }


def validate_escape_experiment(
    trajectory: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | ExperimentBundle,
    manifest: str | Path | Mapping[str, Any] | None = None,
    telemetry: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    controls: Any = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one candidate trajectory and optional comparison batches."""

    if isinstance(trajectory, ExperimentBundle):
        bundle = trajectory
    else:
        bundle = load_bundle(trajectory, manifest, telemetry)
    resolved = EscapeThresholds.from_sources(bundle.manifest, thresholds)
    candidate_result = _validate_one(bundle, resolved)
    entries = _control_entries(bundle, controls)
    all_batch_results: dict[str, Any] = {"candidate": candidate_result}
    for label, control_bundle in entries[1:]:
        if not control_bundle.rows:
            all_batch_results[label] = {"condition": label, "categories": {}, "ground_escape": {}, "reasons": ["control batch could not be loaded"]}
            continue
        control_result = _validate_one(control_bundle, resolved)
        all_batch_results[label] = control_result
    comparisons = _compare_batches(entries, all_batch_results, resolved)
    category_map = dict(candidate_result["categories"])
    causal_reasons = [
        "CAUSAL_ESCAPE requires comparable same-seed sham/matched, mirror, and lesion/ablation batches",
    ]
    if not candidate_result["categories"][NEURAL_RESPONSE]["established"]:
        causal_reasons.append("NEURAL_RESPONSE is not established")
    if not candidate_result["categories"][MOTOR_TRANSFER]["established"]:
        causal_reasons.append("MOTOR_TRANSFER is not established")
    if not candidate_result["categories"][PHYSICAL_ESCAPE]["established"]:
        causal_reasons.append("PHYSICAL_ESCAPE is not established")
    causal_reasons.extend(comparisons.get("reasons", ()))
    causal = _status(CAUSAL_ESCAPE, bool(
        candidate_result["categories"][NEURAL_RESPONSE]["established"]
        and candidate_result["categories"][MOTOR_TRANSFER]["established"]
        and candidate_result["categories"][PHYSICAL_ESCAPE]["established"]
        and comparisons["passed"]
    ), causal_reasons, {"comparisons": comparisons, "candidate": candidate_result["summary"]})
    category_map[CAUSAL_ESCAPE] = causal
    all_passed = all(item["established"] for item in category_map.values())
    declared_class = None
    for source in (bundle.manifest, bundle.metadata):
        if isinstance(source, Mapping):
            value = source.get("evidence_class")
            if value is None and isinstance(source.get("provenance"), Mapping):
                value = source["provenance"].get("evidence_class")
            if isinstance(value, str) and value.strip():
                declared_class = value.strip()
                break
    evidence_class = declared_class or "digital_simulation"
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": STATUS_NOT_VALIDATED,
        "biological_decision": STATUS_NOT_VALIDATED,
        "decision_code": STATUS_NOT_VALIDATED,
        "status": STATUS_PASS if all_passed else STATUS_FAIL,
        "overall_status": STATUS_PASS if all_passed else STATUS_FAIL,
        "validation_status": STATUS_NOT_VALIDATED,
        "biological_validation": False,
        "biological_claim": False,
        "causal_claim": False,
        "evidence": {
            "class": evidence_class,
            "real_biological": False,
            "digital_only": True,
            "single_success_is_not_biological_validation": True,
        },
        "interpretation": (
            "This report is a digital/engineering evidence gate. Even when all four "
            "categories pass, it does not establish biological validation."
        ),
        "thresholds": resolved.to_dict(),
        "inputs": {
            "trajectory": bundle.trajectory_source or bundle.source,
            "telemetry": bundle.telemetry_source,
            "manifest": bundle.manifest_source,
            "n_rows": len(bundle.rows),
            "seed": _extract_seed(bundle),
        },
        "categories": category_map,
        "criteria": category_map,
        "controls": {
            "batches": all_batch_results,
            "comparability": comparisons,
        },
        "authority": candidate_result["authority"],
        "stimulus_lineage": candidate_result["stimulus_lineage"],
        "telemetry": candidate_result["telemetry"],
        "temporal": candidate_result["temporal"],
        "network": candidate_result["network"],
        "body_integrity": candidate_result["body_integrity"],
        "ground_escape": candidate_result["ground_escape"],
        "metrics": {
            "initial_jump_avoidance": candidate_result["ground_escape"].get("initial_jump_avoidance"),
            "horizontal_away_flight": candidate_result["ground_escape"].get("horizontal_away_flight"),
            "liftoff_time_s": candidate_result["ground_escape"].get("liftoff_time_s"),
            "decision_deadline_s": candidate_result["ground_escape"].get("decision_deadline_s"),
            "planned_contact_time_s": candidate_result["ground_escape"].get("planned_contact_time_s"),
            "configured_duration_s": candidate_result["ground_escape"].get("configured_duration_s"),
            "counterfactual_contact": candidate_result["ground_escape"].get("counterfactual_contact"),
            "physics_collision": candidate_result["ground_escape"].get("physics_collision"),
            "duration_coverage": candidate_result["ground_escape"].get("duration_coverage"),
        },
    }
    # Stable aliases make reports easier to consume without changing the four
    # canonical evidence names.
    report["evidence_levels"] = {name: category_map[name]["status"] for name in EVIDENCE_LEVELS}
    return report


validate_escape = validate_escape_experiment
validate = validate_escape_experiment
validate_paths = validate_escape_experiment


def validate_file(
    trajectory: str | Path,
    manifest: str | Path | Mapping[str, Any] | None = None,
    telemetry: str | Path | None = None,
    *,
    controls: Any = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_escape_experiment(
        trajectory,
        manifest,
        telemetry,
        controls=controls,
        thresholds=thresholds,
    )


def write_report(report: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise EscapeValidationError(f"cannot write report {path}: {exc}") from exc
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="trajectory.json/jsonl or run directory")
    parser.add_argument("--manifest", help="manifest.json (normally inferred from a run directory)")
    parser.add_argument("--telemetry", help="telemetry.json/jsonl (normally inferred)")
    parser.add_argument("--controls", help="JSON file containing sham/mirror/lesion batches")
    parser.add_argument("-o", "--output", help="write a JSON report to this path")
    parser.add_argument("--compact", action="store_true", help="print compact JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        controls = _read_json(Path(args.controls).expanduser()) if args.controls else None
        report = validate_escape_experiment(
            args.trajectory,
            args.manifest,
            args.telemetry,
            controls=controls,
        )
        if args.output:
            write_report(report, args.output)
        print(json.dumps(report, sort_keys=True, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
        # A generated FAIL report is useful in CI and is intentionally
        # non-zero; malformed input is also a non-zero exit.
        return 0 if report["status"] == STATUS_PASS else 1
    except (EscapeValidationError, OSError, ValueError) as exc:
        print(f"validate_escape_experiment: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_NOT_VALIDATED",
    "NEURAL_RESPONSE",
    "MOTOR_TRANSFER",
    "PHYSICAL_ESCAPE",
    "CAUSAL_ESCAPE",
    "EVIDENCE_LEVELS",
    "EscapeValidationError",
    "EscapeThresholds",
    "ExperimentBundle",
    "load_bundle",
    "load_experiment",
    "load_trajectory",
    "validate_escape_experiment",
    "validate_escape",
    "validate",
    "validate_paths",
    "validate_file",
    "write_report",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
