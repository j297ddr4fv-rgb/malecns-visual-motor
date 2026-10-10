#!/usr/bin/env python3
"""Compare two same-seed digital-flight trajectory bundles.

This module is deliberately an *ablation comparator*, not a flight validator.
It checks that a candidate and an ablation are the same experiment up to an
explicit intervention, then reports compact differences in takeoff,
airborne-time, yaw, and steering channels.  A report from this tool never
supports a biological-validation claim; its ``decision`` is always
``NOT_VALIDATED``.

The command accepts either a run directory (containing ``manifest.json`` and
``trajectory.json``), a trajectory/manifest JSON file, or a JSONL trajectory
file::

    python3 scripts/compare_flight_ablation.py candidate/ ablation/ \
        --output ablation_report.json

The input contract is intentionally compatible with the compact traces used by
``run_digital_fly.py``.  The implementation also understands the older
``records``/``samples``/``rows`` and ``t``/``yaw``/``px`` style records used by
the flight experiments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


__all__ = [
    "AblationComparisonError",
    "TrajectoryBundle",
    "load_bundle",
    "load_trajectory_bundle",
    "compare_bundles",
    "compare_runs",
    "compare_trajectories",
    "compare_flight_ablation",
    "write_ablation_report",
    "main",
]


REPORT_SCHEMA_VERSION = "flight-ablation-report-1"
NOT_VALIDATED = "NOT_VALIDATED"

# A provenance map may contain a run-specific label for the intervention.  It
# is compared after these fields are removed; all other provenance fields must
# match exactly.  In particular, a changed world/model/backend/source remains
# a hard incompatibility.
_INTERVENTION_KEYS = {
    "ablation",
    "ablation_label",
    "condition",
    "control",
    "intervention",
    "intervention_label",
    "label",
    "manipulation",
    "perturbation",
    "run_id",
    "run_label",
    "variant",
}

# These are common names for the deliberately changed input/control path.  We
# permit them to vary only as part of an explicitly declared intervention; the
# rest of the provenance contract remains strict.
_INTERVENTION_CHANNEL_KEYS = {
    "cue",
    "drive",
    "input_drive",
    "neural_drive",
    "neural_input",
    "path_role",
    "steering",
    "steering_mode",
    "steering_response",
    "visual_cue",
    "visual_drive",
    "visual_input",
}

_ROW_KEYS = ("trajectory", "records", "rows", "samples", "telemetry", "trace")
_NESTED_ROW_KEYS = ("body", "motor", "visual", "control", "diagnostics")
_TRUE_STRINGS = {"1", "true", "yes", "y", "on", "contact", "grounded"}
_FALSE_STRINGS = {"0", "false", "no", "n", "off", "airborne", "flight"}
_VOLATILE_MANIFEST_KEYS = {
    "outputs",
    "validation",
    "scenario_results",
    "resource_usage",
    "resource_limits",
    "created_at",
    "generated_at",
    "timestamp",
    "updated_at",
    "wall_time_s",
    "run_id",
    "run_label",
    "label",
    "ablation",
    "ablation_label",
    "condition",
    "control",
    "intervention",
    "intervention_label",
    "manipulation",
    "perturbation",
    "variant",
    # Provenance is compared separately after intervention fields are
    # normalised; retaining the raw nested map here would reintroduce the
    # deliberate candidate/ablation difference.
    "provenance",
}


class AblationComparisonError(ValueError):
    """Raised when two bundles cannot be compared safely."""


@dataclass(frozen=True)
class TrajectoryBundle:
    """Normalised input bundle used by :func:`compare_bundles`."""

    source: Path | None
    trajectory_source: Path | None
    manifest_source: Path | None
    payload: Mapping[str, Any]
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    times: tuple[float, ...]
    scenarios: tuple[str, ...]
    scenario_names: tuple[str, ...]
    seed: int | str
    provenance: Mapping[str, Any]
    shared_provenance: Mapping[str, Any]
    intervention: Any
    ground_plane_z_m: float
    takeoff_threshold_m: float
    airborne_threshold_m: float
    source_label: str


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_scalar(value: Any) -> Any:
    """Return a deterministic JSON-compatible scalar when possible."""

    if isinstance(value, (str, int, bool)) or value is None:
        return value
    number = _finite(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return str(value)


def _normalise_json(value: Any, omitted_keys: frozenset[str] = frozenset()) -> Any:
    """Copy JSON-like data, optionally dropping mapping keys case-insensitively."""
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_json(item, omitted_keys)
            for key, item in value.items()
            if str(key).lower() not in omitted_keys
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, omitted_keys) for item in value]
    return _json_scalar(value)


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AblationComparisonError(f"cannot read {path}: {exc}") from exc
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AblationComparisonError(
                    f"invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise AblationComparisonError(
                    f"JSONL row {line_number} in {path} is not an object"
                )
            rows.append(value)
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AblationComparisonError(f"invalid JSON in {path}: {exc}") from exc


def _resolve_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _manifest_output(manifest: Mapping[str, Any], base: Path, name: str) -> Path | None:
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        path = _resolve_path(base, outputs.get(name))
        if path is not None:
            return path
    for key in (name, f"{name}_path"):
        path = _resolve_path(base, manifest.get(key))
        if path is not None:
            return path
    return None


def _find_trajectory_for_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    base = manifest_path.parent
    candidate = _manifest_output(manifest, base, "trajectory")
    if candidate is not None and candidate.is_file():
        return candidate
    for name in ("trajectory.json", "trajectory.jsonl", "flight.json", "records.json"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    raise AblationComparisonError(
        f"manifest {manifest_path} does not identify an existing trajectory file"
    )


def _extract_rows(payload: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Extract rows and top-level metadata from common trajectory envelopes."""

    if isinstance(payload, list):
        rows = payload
        metadata: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        metadata_value = payload.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        rows = None
        for key in _ROW_KEYS:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, list):
                rows = value
                break
            if isinstance(value, Mapping):
                # A nested trajectory envelope (for example
                # {"trajectory": {"rows": [...]}}) is different from a
                # mapping keyed by scenario names.  Recurse when the mapping
                # itself carries one of the known row fields.
                if any(nested_key in value for nested_key in _ROW_KEYS) or "data" in value:
                    nested_rows, nested_metadata = _extract_rows(value)
                    merged_metadata = dict(metadata)
                    merged_metadata.update(nested_metadata)
                    return nested_rows, merged_metadata
                # Keyed scenario traces are flattened deterministically.  The
                # key is injected only when the row itself has no label.
                flattened: list[Mapping[str, Any]] = []
                for scenario, grouped in value.items():
                    if not isinstance(grouped, list):
                        continue
                    for row in grouped:
                        if not isinstance(row, Mapping):
                            raise AblationComparisonError(
                                f"{key}[{scenario!r}] contains a non-object row"
                            )
                        copy = dict(row)
                        copy.setdefault("scenario", str(scenario))
                        flattened.append(copy)
                if flattened or not value:
                    rows = flattened
                    break
        if rows is None:
            # A nested ``data`` envelope is common in exported reports.
            data = payload.get("data")
            if isinstance(data, Mapping) or isinstance(data, list):
                return _extract_rows(data)
            # A small number of exporters use columnar JSON rather than a row
            # list.  Require a time column so that unrelated metadata lists do
            # not accidentally become trajectory rows.
            time_column = None
            for time_key in ("time_s", "timestamp_s", "time", "t"):
                value = payload.get(time_key)
                if isinstance(value, list):
                    time_column = (time_key, value)
                    break
            if time_column is not None:
                count = len(time_column[1])
                if count == 0:
                    raise AblationComparisonError("columnar trajectory contains no rows")
                columns = {
                    str(key): value
                    for key, value in payload.items()
                    if isinstance(value, list) and len(value) == count
                }
                rows = [
                    {key: column[index] for key, column in columns.items()}
                    for index in range(count)
                ]
            else:
                raise AblationComparisonError(
                    "trajectory bundle has no rows/trajectory/records/samples/telemetry field"
                )
    else:
        raise AblationComparisonError("trajectory bundle must be a JSON object or array")

    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AblationComparisonError(f"trajectory row {index} is not an object")
        result.append(row)
    if not result:
        raise AblationComparisonError("trajectory bundle contains no rows")
    return result, metadata


def _merge_metadata(payload: Any, manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(manifest, Mapping):
        merged.update({str(k): v for k, v in manifest.items()})
        manifest_metadata = manifest.get("metadata")
        if isinstance(manifest_metadata, Mapping):
            merged.update({str(k): v for k, v in manifest_metadata.items()})
    if isinstance(payload, Mapping):
        for key in ("seed", "scenario", "scenario_names", "scenarios", "provenance", "intervention", "ablation", "condition", "variant", "ground_plane_z_m", "ground_z_m", "takeoff_threshold_m", "airborne_threshold_m", "frame_ms", "time_step_s"):
            if key in payload:
                merged[key] = payload[key]
    merged.update({str(k): v for k, v in metadata.items()})
    return merged


def _candidate_manifest(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("manifest"), Mapping):
        return payload["manifest"]
    return {}


def _extract_seed(payload: Any, manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> int | str:
    for source in (manifest, payload if isinstance(payload, Mapping) else {}, metadata):
        for key in ("seed", "random_seed", "rng_seed"):
            if not isinstance(source, Mapping) or key not in source:
                continue
            value = source[key]
            if value is None or isinstance(value, bool):
                continue
            number = _finite(value)
            if number is not None and number.is_integer():
                return int(number)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise AblationComparisonError("trajectory bundle is missing required seed metadata")


def _provenance_source(payload: Any, manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in (manifest, payload if isinstance(payload, Mapping) else {}, metadata):
        if not isinstance(source, Mapping):
            continue
        value = source.get("provenance")
        if isinstance(value, Mapping) and value:
            return value
    # Keep the evidence class visible even in a small hand-authored bundle.
    for source in (manifest, payload if isinstance(payload, Mapping) else {}, metadata):
        if isinstance(source, Mapping) and source.get("evidence_class") is not None:
            return {"evidence_class": source["evidence_class"]}
    raise AblationComparisonError("trajectory bundle is missing required provenance metadata")


def _provenance_for_comparison(value: Any, *, intervention_declared: bool) -> Any:
    """Remove only declared intervention fields from a provenance map."""

    omitted = _INTERVENTION_KEYS
    if intervention_declared:
        omitted = omitted | _INTERVENTION_CHANNEL_KEYS
    return _normalise_json(value, omitted)


def _extract_intervention(payload: Any, manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> Any:
    values: dict[str, Any] = {}
    for source_name, source in (
        ("manifest", manifest),
        ("trajectory", payload if isinstance(payload, Mapping) else {}),
        ("metadata", metadata),
    ):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "ablation",
            "ablation_label",
            "condition",
            "control",
            "intervention",
            "intervention_label",
            "label",
            "manipulation",
            "perturbation",
            "run_label",
            "variant",
        ):
            if key in source:
                values[f"{source_name}.{key}"] = _normalise_json(source[key])
        provenance = source.get("provenance")
        if isinstance(provenance, Mapping):
            for key in _INTERVENTION_KEYS:
                if key in provenance:
                    values[f"{source_name}.provenance.{key}"] = _normalise_json(provenance[key])
    return values or None


def _extract_global_scenarios(payload: Any, manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, ...]:
    for source in (manifest, payload if isinstance(payload, Mapping) else {}, metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("scenario_names", "scenarios"):
            value = source.get(key)
            if isinstance(value, (list, tuple)) and value:
                return tuple(str(item) for item in value)
            if isinstance(value, str) and value:
                return (value,)
        value = source.get("scenario")
        if isinstance(value, str) and value:
            return (value,)
    return ()


def _row_scenario(row: Mapping[str, Any]) -> str | None:
    for key in ("scenario", "phase", "scene"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_scenarios(rows: Sequence[Mapping[str, Any]], global_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = tuple(str(name) for name in global_names)
    row_values = [_row_scenario(row) for row in rows]
    if any(value is None for value in row_values):
        if len(names) == 1:
            row_values = [value or names[0] for value in row_values]
        else:
            missing = [index for index, value in enumerate(row_values) if value is None]
            raise AblationComparisonError(
                "trajectory rows are missing scenario labels at indices "
                + ", ".join(str(index) for index in missing[:5])
            )
    sequence = tuple(str(value) for value in row_values)
    observed: list[str] = []
    for value in sequence:
        if value not in observed:
            observed.append(value)
    if names:
        # Scenario names are an order-sensitive declaration.  A manifest that
        # lists the same names in another order is not the same experiment.
        if tuple(names) != tuple(observed):
            raise AblationComparisonError(
                f"declared scenarios {list(names)!r} do not match row scenarios {observed!r}"
            )
    else:
        names = tuple(observed)
    if not names:
        raise AblationComparisonError("trajectory bundle is missing required scenario metadata")
    return sequence, names


def _lookup(row: Mapping[str, Any], names: Sequence[str]) -> tuple[Any, str | None]:
    for name in names:
        if name in row:
            return row[name], name
    for nested_name in _NESTED_ROW_KEYS:
        nested = row.get(nested_name)
        if not isinstance(nested, Mapping):
            continue
        for name in names:
            if name in nested:
                return nested[name], f"{nested_name}.{name}"
    return None, None


def _vector_component(value: Any, index: int) -> float | None:
    if isinstance(value, Mapping):
        labels = ("x", "y", "z") if index < 3 else ()
        orientation_labels = ("roll", "pitch", "yaw") if index < 3 else ()
        for key in (labels[index], orientation_labels[index]):
            if key in value:
                number = _finite(value[key])
                if number is not None:
                    return number
        return None
    if isinstance(value, (list, tuple)) and len(value) > index:
        return _finite(value[index])
    return None


def _time_from_row(row: Mapping[str, Any], index: int, metadata: Mapping[str, Any]) -> float | None:
    value, _ = _lookup(row, ("time_s", "timestamp_s", "timestamp", "time", "t"))
    if isinstance(value, Mapping):
        value = value.get("t_s", value.get("time_s", value.get("seconds")))
    number = _finite(value)
    if number is not None:
        return number
    frame, _ = _lookup(row, ("frame", "frame_index", "index"))
    frame_number = _finite(frame)
    frame_ms = _finite(metadata.get("frame_ms"))
    if frame_number is not None and frame_ms is not None:
        return frame_number * frame_ms / 1000.0
    time_step = _finite(metadata.get("time_step_s"))
    if time_step is not None:
        return index * time_step
    return None


def _validate_times(
    rows: Sequence[Mapping[str, Any]],
    times: Sequence[float],
    scenarios: Sequence[str] | None = None,
) -> None:
    if len(times) != len(rows):
        raise AblationComparisonError("trajectory time grid length does not match row count")
    for index, time in enumerate(times):
        if not math.isfinite(time):
            raise AblationComparisonError(f"trajectory time at row {index} is not finite")
        if index and time <= times[index - 1]:
            # The repository's multi-scenario runner may reset local time to
            # zero at a labelled scenario boundary.  Such a reset is valid
            # only when the scenario label changes; duplicates/decreases
            # within one scenario remain a hard error.
            if (
                scenarios is not None
                and index < len(scenarios)
                and scenarios[index] != scenarios[index - 1]
                and time >= 0.0
            ):
                continue
            raise AblationComparisonError(
                f"trajectory timestamps must increase strictly (row {index - 1}={times[index - 1]!r}, row {index}={time!r})"
            )


def _number_from_mapping(source: Mapping[str, Any], names: Sequence[str], default: float) -> float:
    for name in names:
        value = _finite(source.get(name))
        if value is not None:
            return value
    return default


def _build_bundle(
    source: Path | None,
    trajectory_source: Path | None,
    manifest_source: Path | None,
    payload: Any,
    manifest: Mapping[str, Any],
    source_label: str | None = None,
) -> TrajectoryBundle:
    rows, row_metadata = _extract_rows(payload)
    metadata = _merge_metadata(payload, manifest, row_metadata)
    seed = _extract_seed(payload, manifest, metadata)
    provenance = _provenance_source(payload, manifest, metadata)
    intervention = _extract_intervention(payload, manifest, metadata)
    shared_provenance = _provenance_for_comparison(
        provenance,
        intervention_declared=intervention is not None,
    )
    global_scenarios = _extract_global_scenarios(payload, manifest, metadata)
    scenarios, scenario_names = _extract_scenarios(rows, global_scenarios)
    times: list[float] = []
    for index, row in enumerate(rows):
        value = _time_from_row(row, index, metadata)
        if value is None:
            raise AblationComparisonError(
                f"trajectory row {index} has no finite time_s/t/timestamp or frame+frame_ms"
            )
        times.append(value)
    _validate_times(rows, times, scenarios)

    ground_plane = _number_from_mapping(
        metadata,
        ("ground_plane_z_m", "ground_z_m", "ground_altitude_m", "ground_height_m"),
        0.0,
    )
    takeoff_threshold = _number_from_mapping(
        metadata,
        ("takeoff_threshold_m", "takeoff_min_altitude_m"),
        0.35,
    )
    airborne_threshold = _number_from_mapping(
        metadata,
        ("airborne_threshold_m", "airborne_min_altitude_m"),
        0.10,
    )
    if takeoff_threshold < 0 or airborne_threshold < 0:
        raise AblationComparisonError("altitude thresholds must be non-negative")
    return TrajectoryBundle(
        source=source,
        trajectory_source=trajectory_source,
        manifest_source=manifest_source,
        payload=payload if isinstance(payload, Mapping) else {"trajectory": rows},
        manifest=manifest,
        metadata=metadata,
        rows=tuple(rows),
        times=tuple(times),
        scenarios=scenarios,
        scenario_names=scenario_names,
        seed=seed,
        provenance=provenance,
        shared_provenance=shared_provenance if isinstance(shared_provenance, Mapping) else {},
        intervention=intervention,
        ground_plane_z_m=ground_plane,
        takeoff_threshold_m=takeoff_threshold,
        airborne_threshold_m=airborne_threshold,
        source_label=source_label or (str(source) if source is not None else "<memory>"),
    )


def _payload_and_manifest_from_path(path: Path) -> tuple[Any, Mapping[str, Any], Path | None, Path | None]:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        manifest: Mapping[str, Any] = {}
        if manifest_path.is_file():
            manifest_value = _read_json(manifest_path)
            if not isinstance(manifest_value, Mapping):
                raise AblationComparisonError(f"manifest {manifest_path} is not a JSON object")
            manifest = manifest_value
        if manifest_path.is_file():
            trajectory_path = _find_trajectory_for_manifest(manifest_path, manifest)
        else:
            trajectory_path = next(
                (path / name for name in ("trajectory.json", "trajectory.jsonl", "flight.json") if (path / name).is_file()),
                None,
            )
            if trajectory_path is None:
                raise AblationComparisonError(f"bundle directory {path} has no trajectory file")
        payload = _read_json(trajectory_path)
        embedded = _candidate_manifest(payload)
        if embedded:
            manifest = embedded
        return payload, manifest, trajectory_path, manifest_path if manifest_path.is_file() else None

    if not path.is_file():
        raise AblationComparisonError(f"input bundle does not exist: {path}")
    payload = _read_json(path)
    if path.name == "manifest.json":
        if not isinstance(payload, Mapping):
            raise AblationComparisonError(f"manifest {path} is not a JSON object")
        trajectory_path = _find_trajectory_for_manifest(path, payload)
        return _read_json(trajectory_path), payload, trajectory_path, path
    manifest_path = path.parent / "manifest.json"
    manifest: Mapping[str, Any] = {}
    if manifest_path.is_file():
        value = _read_json(manifest_path)
        if not isinstance(value, Mapping):
            raise AblationComparisonError(f"manifest {manifest_path} is not a JSON object")
        manifest = value
    embedded = _candidate_manifest(payload)
    if embedded:
        manifest = embedded
    return payload, manifest, path, manifest_path if manifest_path.is_file() else None


def load_bundle(value: str | Path | Mapping[str, Any], *, source_label: str | None = None) -> TrajectoryBundle:
    """Load and normalise one trajectory bundle.

    ``Mapping`` input is useful for callers/tests that already have an in-memory
    JSON envelope.  It still must carry seed, scenario and provenance metadata.
    """

    if isinstance(value, Mapping):
        payload = value
        manifest = _candidate_manifest(payload)
        return _build_bundle(None, None, None, payload, manifest, source_label=source_label)
    path = Path(value).expanduser().resolve()
    payload, manifest, trajectory_source, manifest_source = _payload_and_manifest_from_path(path)
    return _build_bundle(path, trajectory_source, manifest_source, payload, manifest, source_label=source_label)


load_trajectory_bundle = load_bundle


def _normalise_seed(seed: int | str) -> int | str:
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    value = str(seed).strip()
    try:
        return int(value)
    except ValueError:
        return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_normalise_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_text(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _manifest_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep stable run identity while dropping per-run output/measurement data."""
    return _normalise_json(value, _VOLATILE_MANIFEST_KEYS)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _verify_compatibility(candidate: TrajectoryBundle, ablation: TrajectoryBundle, tolerance: float) -> dict[str, Any]:
    candidate_seed = _normalise_seed(candidate.seed)
    ablation_seed = _normalise_seed(ablation.seed)
    if candidate_seed != ablation_seed:
        raise AblationComparisonError(
            f"seed mismatch: candidate={candidate_seed!r}, ablation={ablation_seed!r}"
        )
    if len(candidate.times) != len(ablation.times):
        raise AblationComparisonError(
            f"time-grid length mismatch: candidate={len(candidate.times)}, ablation={len(ablation.times)}"
        )
    if tolerance < 0 or not math.isfinite(tolerance):
        raise AblationComparisonError("time tolerance must be a finite non-negative number")
    time_deltas = [abs(a - b) for a, b in zip(candidate.times, ablation.times)]
    max_time_delta = max(time_deltas, default=0.0)
    if max_time_delta > tolerance:
        raise AblationComparisonError(
            f"time-grid mismatch: maximum absolute delta {max_time_delta:.12g}s exceeds tolerance {tolerance:.12g}s"
        )
    if candidate.scenario_names != ablation.scenario_names or candidate.scenarios != ablation.scenarios:
        raise AblationComparisonError(
            "scenario mismatch: candidate and ablation must have the same declared and row-level scenario sequence"
        )
    if candidate.shared_provenance != ablation.shared_provenance:
        raise AblationComparisonError(
            "provenance mismatch: candidate and ablation differ outside their declared intervention metadata"
        )
    candidate_identity = _manifest_identity(candidate.manifest)
    ablation_identity = _manifest_identity(ablation.manifest)
    if candidate_identity != ablation_identity:
        raise AblationComparisonError(
            "provenance mismatch: stable manifest run identity differs between candidate and ablation"
        )
    return {
        "passed": True,
        "seed": {"candidate": candidate_seed, "ablation": ablation_seed, "matched": True},
        "time_grid": {
            "sample_count": len(candidate.times),
            "candidate_start_s": candidate.times[0],
            "candidate_end_s": candidate.times[-1],
            "ablation_start_s": ablation.times[0],
            "ablation_end_s": ablation.times[-1],
            "max_abs_delta_s": max_time_delta,
            "tolerance_s": tolerance,
            "matched": True,
        },
        "scenario": {
            "candidate": list(candidate.scenario_names),
            "ablation": list(ablation.scenario_names),
            "matched": True,
        },
        "provenance": {
            "matched": True,
            "shared_fingerprint_sha256": _hash_text(candidate.shared_provenance),
            "shared_keys": sorted(str(key) for key in candidate.shared_provenance),
            "manifest_identity_sha256": _hash_text(candidate_identity),
            "candidate_intervention": candidate.intervention,
            "ablation_intervention": ablation.intervention,
        },
    }


def _altitude(row: Mapping[str, Any], ground_plane_z_m: float) -> tuple[float | None, str | None]:
    value, source = _lookup(row, ("altitude_m", "height_m", "z_m", "height", "altitude"))
    number = _finite(value)
    if number is not None:
        return number, source
    value, source = _lookup(row, ("position_m", "position", "pos", "xyz", "traj"))
    component = _vector_component(value, 2)
    if component is not None:
        return component - ground_plane_z_m, source
    value, source = _lookup(row, ("z", "pz"))
    number = _finite(value)
    if number is not None:
        return number - ground_plane_z_m, source
    return None, None


def _collision_count(row: Mapping[str, Any]) -> float:
    value, _ = _lookup(row, ("collision_count", "collisions", "collision"))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _finite(value)
    return max(0.0, number) if number is not None else 0.0


def _contiguous_true_runs(
    states: Sequence[bool | None], times: Sequence[float], scenarios: Sequence[str]
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, state in enumerate(states):
        if state is True:
            if start is None:
                start = index
            elif scenarios[index] != scenarios[index - 1] or times[index] - times[index - 1] > 0.5:
                runs.append((start, index - 1))
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(states) - 1))
    return runs


def _takeoff_metrics(bundle: TrajectoryBundle) -> dict[str, Any]:
    values = [_altitude(row, bundle.ground_plane_z_m)[0] for row in bundle.rows]
    available = [(index, value) for index, value in enumerate(values) if value is not None]
    collisions = max((_collision_count(row) for row in bundle.rows), default=0.0)
    base: dict[str, Any] = {
        "available": bool(available),
        "threshold_m": bundle.takeoff_threshold_m,
        "collision_count": int(collisions) if collisions.is_integer() else collisions,
    }
    if not available:
        base.update({
            "start_altitude_m": None,
            "max_altitude_m": None,
            "altitude_gain_m": None,
            "threshold_crossing_time_s": None,
            "threshold_dwell_s": None,
            "takeoff_detected": False,
        })
        return base
    start_index, start_altitude = available[0]
    max_index, max_altitude = max(available, key=lambda item: item[1])
    gain = max_altitude - start_altitude
    runs = _contiguous_true_runs(
        [value is not None and value >= bundle.takeoff_threshold_m for value in values],
        bundle.times,
        bundle.scenarios,
    )
    crossing_index = runs[0][0] if runs else None
    crossing_time = bundle.times[crossing_index] if crossing_index is not None else None
    dwell = bundle.times[runs[0][1]] - crossing_time if runs else 0.0
    base.update({
        "start_altitude_m": start_altitude,
        "max_altitude_m": max_altitude,
        "max_altitude_time_s": bundle.times[max_index],
        "altitude_gain_m": gain,
        "threshold_crossing_time_s": crossing_time,
        "threshold_dwell_s": dwell,
        "takeoff_detected": bool(
            gain >= 0.20 and crossing_index is not None and dwell >= 0.20 and collisions == 0
        ),
        "start_row": start_index,
    })
    return base


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    number = _finite(value)
    if number is not None and number in (0.0, 1.0):
        return bool(number)
    return None


def _contact(row: Mapping[str, Any]) -> bool | None:
    value, _ = _lookup(row, ("contact", "grounded", "on_ground", "landed"))
    return _as_bool(value)


def _airborne_metrics(bundle: TrajectoryBundle) -> dict[str, Any]:
    altitude_values = [_altitude(row, bundle.ground_plane_z_m)[0] for row in bundle.rows]
    states: list[bool | None] = []
    for row, altitude in zip(bundle.rows, altitude_values):
        contact = _contact(row)
        if contact is not None:
            state = not contact
            # An explicit contact channel is authoritative, but a reported
            # near-ground point should not be advertised as sustained flight.
            if altitude is not None:
                state = state and altitude >= bundle.airborne_threshold_m
            states.append(state)
        elif altitude is not None:
            states.append(altitude >= bundle.airborne_threshold_m)
        else:
            states.append(None)
    runs = _contiguous_true_runs(states, bundle.times, bundle.scenarios)
    if not runs:
        return {
            "available": any(value is not None for value in altitude_values) or any(value is not None for value in states),
            "airborne_threshold_m": bundle.airborne_threshold_m,
            "airborne_duration_s": 0.0,
            "airborne_start_time_s": None,
            "airborne_end_time_s": None,
            "airborne_fraction": 0.0,
            "airborne_detected": False,
            "max_contiguous_duration_s": 0.0,
        }
    best_start, best_end = max(runs, key=lambda item: bundle.times[item[1]] - bundle.times[item[0]])
    duration = bundle.times[best_end] - bundle.times[best_start]
    true_count = sum(value is True for value in states)
    return {
        "available": True,
        "airborne_threshold_m": bundle.airborne_threshold_m,
        "airborne_duration_s": duration,
        "airborne_start_time_s": bundle.times[best_start],
        "airborne_end_time_s": bundle.times[best_end],
        "airborne_fraction": true_count / len(states),
        "airborne_detected": duration > 0.0 or len(runs) > 0,
        "max_contiguous_duration_s": duration,
        "airborne_start_row": best_start,
        "airborne_end_row": best_end,
    }


def _yaw_metrics(bundle: TrajectoryBundle) -> dict[str, Any]:
    values: list[tuple[float, float]] = []
    value_scenarios: list[str] = []
    for index, (time, row) in enumerate(zip(bundle.times, bundle.rows)):
        value, _ = _lookup(row, ("yaw", "yaw_rad"))
        yaw = _finite(value)
        if yaw is None:
            value, _ = _lookup(row, ("orientation_rad", "orientation", "rpy"))
            yaw = _vector_component(value, 2)
        if yaw is not None:
            values.append((time, yaw))
            value_scenarios.append(bundle.scenarios[index])
    if not values:
        return {
            "available": False,
            "sample_count": 0,
            "start_yaw_rad": None,
            "end_yaw_rad": None,
            "net_yaw_change_rad": None,
            "max_abs_yaw_change_rad": None,
            "yaw_range_rad": None,
        }
    unwrapped = [values[0][1]]
    for value_index, (_, value) in enumerate(values[1:], 1):
        previous = unwrapped[-1]
        if value_index < len(value_scenarios) and value_scenarios[value_index] != value_scenarios[value_index - 1]:
            unwrapped.append(value)
            continue
        delta = (value - previous + math.pi) % (2.0 * math.pi) - math.pi
        unwrapped.append(previous + delta)
    start = unwrapped[0]
    changes = [value - start for value in unwrapped]
    rate_values: list[float] = []
    for index, ((time_a, _), (time_b, _)) in enumerate(zip(values, values[1:])):
        dt = time_b - time_a
        if dt > 0:
            rate_values.append((unwrapped[index + 1] - unwrapped[index]) / dt)
    return {
        "available": True,
        "sample_count": len(unwrapped),
        "start_yaw_rad": start,
        "end_yaw_rad": unwrapped[-1],
        "net_yaw_change_rad": unwrapped[-1] - start,
        "max_abs_yaw_change_rad": max(abs(change) for change in changes),
        "yaw_range_rad": max(unwrapped) - min(unwrapped),
        "peak_abs_yaw_rate_rad_s": max((abs(value) for value in rate_values), default=None),
    }


def _steering_value(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    direct_names = (
        "steering_response",
        "steering",
        "wing_steering",
        "flight_steering",
        "flight_yaw",
        "yaw_command",
        "steering_command",
    )
    value, source = _lookup(row, direct_names)
    number = _finite(value)
    if number is not None:
        return number, source
    right, right_source = _lookup(row, ("right_wing", "wing_drive_right", "wmR", "right_motor"))
    left, left_source = _lookup(row, ("left_wing", "wing_drive_left", "wmL", "left_motor"))
    right_number = _finite(right)
    left_number = _finite(left)
    if right_number is not None and left_number is not None:
        return right_number - left_number, f"{right_source or 'right'}-{left_source or 'left'}"
    value, source = _lookup(row, ("yaw_rate", "yaw_rate_rad_s", "omega", "angular_velocity_rad_s"))
    number = _finite(value)
    if number is None and isinstance(value, (list, tuple, Mapping)):
        number = _vector_component(value, 2)
    if number is not None:
        return number, source
    return None, None


def _visual_cue_value(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    value, source = _lookup(
        row,
        ("visual_direction", "visual_cue", "optic_flow_rad_s", "flow_yaw", "cue", "flow"),
    )
    number = _finite(value)
    if number is not None:
        return number, source
    if isinstance(value, (list, tuple, Mapping)):
        number = _vector_component(value, 0)
        if number is not None:
            return number, source
    return None, None


def _signal_summary(values: Sequence[tuple[float, float]], source_fields: Sequence[str]) -> dict[str, Any]:
    if not values:
        return {
            "available": False,
            "sample_count": 0,
            "source_field": None,
            "mean": None,
            "mean_abs": None,
            "rms": None,
            "peak_abs": None,
            "integral": None,
            "final": None,
            "nonzero_count": 0,
        }
    numbers = [value for _, value in values]
    mean = sum(numbers) / len(numbers)
    integral = 0.0
    for (time_a, value_a), (time_b, value_b) in zip(values, values[1:]):
        delta_t = time_b - time_a
        if delta_t > 0.0:
            integral += 0.5 * (value_a + value_b) * delta_t
    return {
        "available": True,
        "sample_count": len(numbers),
        "source_field": Counter(source_fields).most_common(1)[0][0] if source_fields else None,
        "mean": mean,
        "mean_abs": sum(abs(value) for value in numbers) / len(numbers),
        "rms": math.sqrt(sum(value * value for value in numbers) / len(numbers)),
        "peak_abs": max(abs(value) for value in numbers),
        "integral": integral,
        "final": numbers[-1],
        "nonzero_count": sum(abs(value) > 1.0e-12 for value in numbers),
    }


def _steering_metrics(bundle: TrajectoryBundle) -> dict[str, Any]:
    responses: list[tuple[float, float]] = []
    response_sources: list[str] = []
    cues: list[tuple[float, float]] = []
    cue_sources: list[str] = []
    for time, row in zip(bundle.times, bundle.rows):
        value, source = _steering_value(row)
        if value is not None:
            responses.append((time, value))
            if source is not None:
                response_sources.append(source)
        cue, cue_source = _visual_cue_value(row)
        if cue is not None:
            cues.append((time, cue))
            if cue_source is not None:
                cue_sources.append(cue_source)
    response = _signal_summary(responses, response_sources)
    cue = _signal_summary(cues, cue_sources)
    response["visual_cue"] = cue
    return response


def _numeric_deltas(candidate: Mapping[str, Any], ablation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(candidate) | set(ablation)):
        left_mapping = candidate.get(key)
        right_mapping = ablation.get(key)
        if isinstance(left_mapping, Mapping) and isinstance(right_mapping, Mapping):
            nested = _numeric_deltas(left_mapping, right_mapping)
            if nested:
                result[key] = nested
            continue
        left = _finite(candidate.get(key))
        right = _finite(ablation.get(key))
        if left is not None and right is not None:
            result[key] = right - left
    return result


def _absolute_deltas(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _absolute_deltas(item)
        else:
            number = _finite(item)
            if number is not None:
                result[key] = abs(number)
    return result


def _has_nonzero_delta(value: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(value, Mapping):
        return any(_has_nonzero_delta(item, tolerance) for item in value.values())
    number = _finite(value)
    return number is not None and abs(number) > tolerance


def _metric_comparison(candidate: Mapping[str, Any], ablation: Mapping[str, Any]) -> dict[str, Any]:
    delta = _numeric_deltas(candidate, ablation)
    absolute = _absolute_deltas(delta)
    return {
        "candidate": candidate,
        "ablation": ablation,
        "delta": delta,
        # ``difference`` is a convenient explicit alias for consumers that do
        # not want to know whether the report uses delta terminology.
        "difference": delta,
        "absolute_delta": absolute,
    }


def _response_effect(steering: Mapping[str, Any], tolerance: float = 1.0e-12) -> dict[str, Any]:
    candidate_mean_abs = _finite(steering.get("candidate", {}).get("mean_abs"))
    ablation_mean_abs = _finite(steering.get("ablation", {}).get("mean_abs"))
    candidate_mean = _finite(steering.get("candidate", {}).get("mean"))
    ablation_mean = _finite(steering.get("ablation", {}).get("mean"))
    if candidate_mean_abs is None or ablation_mean_abs is None:
        return {
            "available": False,
            "removed": None,
            "reversed": None,
            "removed_or_reversed": None,
            "ablation_to_candidate_abs_ratio": None,
        }
    ratio = ablation_mean_abs / candidate_mean_abs if candidate_mean_abs > tolerance else None
    reversed_response = bool(
        candidate_mean is not None and ablation_mean is not None
        and abs(candidate_mean) > tolerance and abs(ablation_mean) > tolerance
        and candidate_mean * ablation_mean < 0
    )
    removed = bool(candidate_mean_abs > tolerance and ablation_mean_abs <= 0.5 * candidate_mean_abs)
    return {
        "available": True,
        "removed": removed,
        "reversed": reversed_response,
        "removed_or_reversed": removed or reversed_response,
        "ablation_to_candidate_abs_ratio": ratio,
    }


def _input_summary(bundle: TrajectoryBundle) -> dict[str, Any]:
    duration = 0.0
    for index in range(1, len(bundle.times)):
        if bundle.scenarios[index] == bundle.scenarios[index - 1]:
            delta_t = bundle.times[index] - bundle.times[index - 1]
            if delta_t > 0.0:
                duration += delta_t
    return {
        "path": bundle.source_label,
        "trajectory_path": str(bundle.trajectory_source) if bundle.trajectory_source is not None else None,
        "manifest_path": str(bundle.manifest_source) if bundle.manifest_source is not None else None,
        "trajectory_sha256": _sha256(bundle.trajectory_source),
        "manifest_sha256": _sha256(bundle.manifest_source),
        "seed": _normalise_seed(bundle.seed),
        "scenario_names": list(bundle.scenario_names),
        "sample_count": len(bundle.rows),
        "duration_s": duration,
        "evidence_class": bundle.provenance.get("evidence_class"),
        "intervention": bundle.intervention,
    }


def compare_bundles(
    candidate: str | Path | Mapping[str, Any],
    ablation: str | Path | Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    time_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Compare two compatible candidate/ablation trajectory bundles.

    The function raises :class:`AblationComparisonError` before writing any
    report if the runs are not compatible.  ``output_path`` is optional for
    library callers and causes an ``ablation_report.json`` to be written when
    supplied.
    """

    candidate_bundle = load_bundle(candidate, source_label="candidate" if isinstance(candidate, Mapping) else None)
    ablation_bundle = load_bundle(ablation, source_label="ablation" if isinstance(ablation, Mapping) else None)
    compatibility = _verify_compatibility(candidate_bundle, ablation_bundle, time_tolerance)

    candidate_metrics = {
        "takeoff": _takeoff_metrics(candidate_bundle),
        "airborne": _airborne_metrics(candidate_bundle),
        "yaw": _yaw_metrics(candidate_bundle),
        "steering": _steering_metrics(candidate_bundle),
    }
    ablation_metrics = {
        "takeoff": _takeoff_metrics(ablation_bundle),
        "airborne": _airborne_metrics(ablation_bundle),
        "yaw": _yaw_metrics(ablation_bundle),
        "steering": _steering_metrics(ablation_bundle),
    }
    metrics = {
        name: _metric_comparison(candidate_metrics[name], ablation_metrics[name])
        for name in candidate_metrics
    }
    steering_effect = _response_effect(metrics["steering"])
    candidate_yaw = metrics["yaw"]["candidate"]
    ablation_yaw = metrics["yaw"]["ablation"]
    candidate_yaw_abs = _finite(candidate_yaw.get("max_abs_yaw_change_rad"))
    ablation_yaw_abs = _finite(ablation_yaw.get("max_abs_yaw_change_rad"))
    yaw_effect: dict[str, Any]
    if candidate_yaw_abs is None or ablation_yaw_abs is None:
        yaw_effect = {"available": False, "removed_or_reversed": None}
    else:
        candidate_delta = _finite(candidate_yaw.get("net_yaw_change_rad"))
        ablation_delta = _finite(ablation_yaw.get("net_yaw_change_rad"))
        yaw_removed = candidate_yaw_abs > 1.0e-12 and ablation_yaw_abs <= 0.5 * candidate_yaw_abs
        yaw_reversed = bool(
            candidate_delta is not None and ablation_delta is not None
            and abs(candidate_delta) > 1.0e-12 and abs(ablation_delta) > 1.0e-12
            and candidate_delta * ablation_delta < 0
        )
        yaw_effect = {
            "available": True,
            "removed": yaw_removed,
            "reversed": yaw_reversed,
            "removed_or_reversed": yaw_removed or yaw_reversed,
            "ablation_to_candidate_abs_ratio": (
                ablation_yaw_abs / candidate_yaw_abs if candidate_yaw_abs > 1.0e-12 else None
            ),
        }
    all_differences = {
        name: metrics[name]["delta"] for name in metrics
    }
    effect_observed = any(_has_nonzero_delta(values) for values in all_differences.values())
    causal_status = "DIGITAL_DIFFERENCE_OBSERVED" if effect_observed else "NO_DIGITAL_DIFFERENCE_OBSERVED"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "decision": NOT_VALIDATED,
        "validation_status": NOT_VALIDATED,
        "biological_validation": False,
        "biological_claim": False,
        "evidence": {
            "class": "digital_simulation",
            "real_biological": False,
            "biological_claim": False,
            "candidate_declared_class": candidate_bundle.provenance.get("evidence_class"),
            "ablation_declared_class": ablation_bundle.provenance.get("evidence_class"),
        },
        "interpretation": (
            "This is a same-seed digital trajectory comparison. It reports engineering/control "
            "differences only and is not biological validation."
        ),
        "inputs": {
            "candidate": _input_summary(candidate_bundle),
            "ablation": _input_summary(ablation_bundle),
        },
        "compatibility": compatibility,
        "metrics": metrics,
        "differences": all_differences,
        "causal_control_evidence": {
            "status": causal_status,
            "same_seed": True,
            "same_time_grid": True,
            "same_scenario": True,
            "same_shared_provenance": True,
            "steering_response": steering_effect,
            "yaw_response": yaw_effect,
            "effect_observed": effect_observed,
            "caveat": (
                "A matched digital ablation supports a controlled-control comparison, "
                "but does not establish biological validity or by itself prove mechanism."
            ),
        },
    }
    if output_path is not None:
        write_ablation_report(report, output_path)
    return report


def compare_trajectories(
    candidate: str | Path | Mapping[str, Any],
    ablation: str | Path | Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    time_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Alias for :func:`compare_bundles` with a trajectory-oriented name."""

    return compare_bundles(candidate, ablation, output_path, time_tolerance=time_tolerance)


def compare_flight_ablation(
    candidate: str | Path | Mapping[str, Any],
    ablation: str | Path | Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    time_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Descriptive alias for callers that want an explicit flight name."""

    return compare_bundles(candidate, ablation, output_path, time_tolerance=time_tolerance)


compare_runs = compare_bundles


def write_ablation_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write one JSON-safe compact ablation report and return its path."""

    path = Path(output_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise AblationComparisonError(f"cannot write ablation report {path}: {exc}") from exc
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_pos", nargs="?", help="candidate trajectory bundle directory or JSON/JSONL file")
    parser.add_argument("ablation_pos", nargs="?", help="ablation trajectory bundle directory or JSON/JSONL file")
    parser.add_argument(
        "--candidate",
        dest="candidate_opt",
        help="candidate trajectory bundle (alternative to the first positional argument)",
    )
    parser.add_argument(
        "--ablation",
        dest="ablation_opt",
        help="ablation trajectory bundle (alternative to the second positional argument)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ablation_report.json",
        help="output report path (default: ablation_report.json)",
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=1.0e-9,
        help="maximum allowed absolute timestamp mismatch in seconds (default: 1e-9)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    candidate = args.candidate_opt or args.candidate_pos
    ablation = args.ablation_opt or args.ablation_pos
    if not candidate or not ablation:
        parser.error("candidate and ablation bundles are required")
    try:
        report = compare_bundles(
            candidate,
            ablation,
            args.output,
            time_tolerance=args.time_tolerance,
        )
    except (AblationComparisonError, OSError, ValueError) as exc:
        print(f"compare_flight_ablation: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(Path(args.output).expanduser()),
        "status": report["causal_control_evidence"]["status"],
        "decision": report["decision"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
