#!/usr/bin/env python3
"""Finalize and audit a rendered digital-fly flight artifact bundle.

The simulator and renderer deliberately produce the scientific inputs first.
This module is a small, read-only verifier/finalizer that runs afterwards.  It
does not rewrite ``manifest.json``, ``trajectory.json``, ``telemetry.jsonl``,
``validation.json`` or ``flight.mp4``.  Instead it probes the video and writes
the two derived sidecars ``video_probe.json`` and ``integrity.json``.

Typical use::

    python3 scripts/finalize_flight_artifact.py artifacts/my-run

The five input files are required.  In addition to checking that they are
present and non-empty, the verifier checks JSON/JSONL syntax, checks that the
trajectory and telemetry contain the same rows, requires the manifest to name
all five members, checks the output/evidence/provenance declarations, and
fails if ffprobe cannot establish a non-empty video stream with finite
dimensions, frame rate, duration and frame count.  The integrity sidecar also
retains compact provenance, resource-envelope and evidence summaries; it does
not turn a connectome-constrained engineering simulation into biological
validation.

Only Python's standard library and the external ``ffprobe`` executable are
needed.  The output JSON retains ffprobe's raw ``streams`` and ``format``
objects so that it remains compatible with the repository's documented probe
commands, and adds normalized scalar fields plus an explicit simulation to
playback mapping.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, NoReturn, Optional


SCHEMA_VERSION = "flight-bundle-integrity-1"
VIDEO_PROBE_SCHEMA_VERSION = "flight-video-probe-1"
EXPECTED_EVIDENCE_CLASS = "digital_simulation"

# These are intentionally fixed.  A bundle with a renamed or missing input is
# not auditable merely because another similarly named file happens to exist.
REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("manifest", "manifest.json"),
    ("trajectory", "trajectory.json"),
    ("telemetry", "telemetry.jsonl"),
    ("validation", "validation.json"),
    ("video", "flight.mp4"),
)

# These are the names used by the delivery manifest.  A finalised bundle is
# not a loose collection of files: the manifest must point at every member so
# that copying a directory cannot silently detach validation or video from the
# run that produced them.
REQUIRED_OUTPUTS: tuple[tuple[str, str], ...] = REQUIRED_FILES


class FinalizationError(RuntimeError):
    """Raised when a flight bundle cannot be finalized safely."""


def _fail(message: str) -> NoReturn:
    raise FinalizationError(message)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value}")


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    """Convert a probe/declaration value to a finite float."""

    if isinstance(value, bool):
        _fail(f"{label} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalizationError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        _fail(f"{label} must be {qualifier}, got {value!r}")
    return number


def _finite_int(value: Any, label: str, *, positive: bool = False) -> int:
    """Convert a ffprobe integer field without accepting fractional values."""

    if isinstance(value, bool):
        _fail(f"{label} must be an integer, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalizationError(f"{label} must be an integer, got {value!r}") from exc
    if not math.isfinite(number) or number != math.trunc(number):
        _fail(f"{label} must be an integer, got {value!r}")
    result = int(number)
    if positive and result <= 0:
        _fail(f"{label} must be a positive integer, got {value!r}")
    return result


def _read_required_files(run_dir: Path) -> dict[str, Path]:
    """Resolve required bundle members and reject missing/empty/non-files."""

    if not run_dir.exists():
        _fail(f"run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        _fail(f"run path is not a directory: {run_dir}")

    paths: dict[str, Path] = {}
    for logical_name, filename in REQUIRED_FILES:
        path = run_dir / filename
        # is_file() rejects directories and broken symlinks.  A symlink to a
        # regular file is allowed, but all later references are resolved and
        # still have to point at the requested member.
        if not path.is_file():
            _fail(f"missing required {logical_name} file: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise FinalizationError(f"cannot stat required {logical_name} file {path}: {exc}") from exc
        if size <= 0:
            _fail(f"required {logical_name} file is empty: {path}")
        paths[logical_name] = path
    return paths


def _load_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise FinalizationError(f"invalid {description} JSON at {path}: {exc}") from exc
    if value is None:
        _fail(f"{description} JSON is null: {path}")
    return value


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{description} must be a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    """Canonical form used only for comparing already parsed JSON rows."""

    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"JSON contains a non-finite or unsupported value: {exc}") from exc


def _read_telemetry(path: Path) -> list[Mapping[str, Any]]:
    """Read strict JSONL telemetry; blank records are not silently ignored."""

    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    _fail(f"telemetry contains an empty record at line {line_number}")
                try:
                    value = json.loads(raw_line, parse_constant=_reject_json_constant)
                except (UnicodeError, ValueError) as exc:
                    raise FinalizationError(
                        f"invalid telemetry JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    _fail(f"telemetry record at line {line_number} is not a JSON object")
                # Dumping here also rejects NaN/Infinity, which Python's JSON
                # decoder otherwise accepts by default.
                _canonical_json(value)
                rows.append(value)
    except OSError as exc:
        raise FinalizationError(f"cannot read telemetry {path}: {exc}") from exc
    if not rows:
        _fail(f"telemetry contains no records: {path}")
    return rows


def _trajectory_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows_value = payload.get("trajectory")
    if not isinstance(rows_value, list) or not rows_value:
        _fail("trajectory.json must contain a non-empty 'trajectory' array")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows_value):
        if not isinstance(row, Mapping):
            _fail(f"trajectory row {index} is not a JSON object")
        _canonical_json(row)
        rows.append(row)
    return rows


def _trajectory_times(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    times: list[float] = []
    for index, row in enumerate(rows):
        if "time_s" not in row:
            _fail(f"trajectory row {index} is missing time_s; simulation/playback mapping is unsafe")
        times.append(_finite_float(row["time_s"], f"trajectory row {index} time_s"))
    for index, (previous, current) in enumerate(zip(times, times[1:]), start=1):
        if current <= previous:
            _fail(
                "trajectory timestamps must be strictly increasing "
                f"(row {index - 1}={previous:g}, row {index}={current:g})"
            )
    return times[0], times[-1], max(0.0, times[-1] - times[0])


def _same_file_reference(value: Any, expected_path: Path, expected_name: str) -> bool:
    """Match absolute, run-relative, portable-name, or safe copied-run references."""

    if not isinstance(value, (str, os.PathLike)):
        return False
    raw = str(value).strip()
    if not raw:
        return False
    candidate = Path(raw).expanduser()
    if candidate.name != expected_name:
        return False
    if candidate.is_absolute():
        try:
            return candidate.resolve(strict=False) == expected_path.resolve(strict=False)
        except OSError:
            return False
    try:
        if (expected_path.parent / candidate).resolve(strict=False) == expected_path.resolve(strict=False):
            return True
        # A relative provenance path such as ``artifacts/run/trajectory.json``
        # is common in validator output when the command was run from the
        # repository root.  Accept it only when it resolves to this exact
        # file, not merely because its basename happens to match.
        if (Path.cwd() / candidate).resolve(strict=False) == expected_path.resolve(strict=False):
            return True
    except OSError:
        pass
    # Simple names are the portable form used when a run directory is copied.
    # A path containing a directory component that did not resolve above is a
    # concrete mismatch and must fail closed.
    if len(candidate.parts) == 1 and candidate.name == expected_name:
        return True
    if any(part in {".", ".."} for part in candidate.parts):
        return False
    expected_parts = expected_path.resolve(strict=False).parts
    return len(candidate.parts) <= len(expected_parts) and expected_parts[-len(candidate.parts) :] == candidate.parts


def _is_cuda_label(value: Any) -> bool:
    """Return whether a backend/device declaration names a CUDA execution."""

    if not isinstance(value, str):
        return False
    label = value.strip().lower()
    return label.startswith("cuda") or "_cuda" in label or label.endswith("-cuda")


def _check_backend_declarations(manifest: Mapping[str, Any]) -> None:
    """Reject requested CUDA fallbacks; neural declarations also require a CUDA backend."""

    requested_backends_value = manifest.get("requested_backends")
    requested_backends = (
        _require_mapping(requested_backends_value, "manifest.requested_backends")
        if requested_backends_value is not None else {}
    )
    requested_execution_value = manifest.get("requested_execution")
    requested_execution = (
        _require_mapping(requested_execution_value, "manifest.requested_execution")
        if requested_execution_value is not None else {}
    )
    effective_execution_value = manifest.get("effective_execution")
    effective_execution = (
        _require_mapping(effective_execution_value, "manifest.effective_execution")
        if effective_execution_value is not None else {}
    )
    effective_backends_value = manifest.get("effective_backends")
    effective_backends = (
        _require_mapping(effective_backends_value, "manifest.effective_backends")
        if effective_backends_value is not None else {}
    )
    components_value = manifest.get("components")
    components = (
        _require_mapping(components_value, "manifest.components")
        if components_value is not None else {}
    )

    for name in ("neural", "vision"):
        device_key = f"{name}_device"
        if not _is_cuda_label(requested_execution.get(device_key)):
            continue
        component_value = components.get(name)
        component = (
            _require_mapping(component_value, f"manifest.components.{name}")
            if component_value is not None else {}
        )
        runtime_value = component.get("runtime")
        runtime = runtime_value if isinstance(runtime_value, Mapping) else {}
        if component.get("is_stub") is True:
            _fail(f"manifest requests CUDA {name} execution but components.{name} is a stub")

        fallback_device = runtime.get("device")
        if name == "neural":
            fallback_device = component.get("effective_device", fallback_device)
        effective_device = effective_execution.get(device_key, fallback_device)
        if not _is_cuda_label(effective_device):
            _fail(
                f"manifest requests CUDA {name} execution but effective {name} device "
                f"is {effective_device!r}"
            )
        if name == "neural":
            fallback_backend = runtime.get("backend", effective_backends.get("neural"))
            fallback_backend = component.get("effective_backend", fallback_backend)
            effective_backend = effective_execution.get("neural_backend", fallback_backend)
            if not _is_cuda_label(effective_backend):
                _fail(
                    "manifest requests CUDA neural execution but effective neural backend "
                    f"is not explicitly CUDA: {effective_backend!r}"
                )


def _check_declared_references(
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    validation: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    """Cross-check required members, evidence class, seed, run ID, and backend declarations."""

    outputs = manifest.get("outputs")
    outputs_map = _require_mapping(outputs, "manifest.outputs")
    for logical_name, filename in REQUIRED_OUTPUTS:
        if logical_name not in outputs_map or outputs_map[logical_name] is None:
            _fail(f"manifest.outputs is missing required {logical_name!r} reference")
        if not _same_file_reference(
            outputs_map[logical_name], paths["manifest"].parent / filename, filename
        ):
            _fail(f"manifest.outputs.{logical_name} does not reference {filename!r} in this run")

    metadata = trajectory.get("metadata")
    metadata_map = _require_mapping(metadata, "trajectory.metadata")
    def check_references(references: Iterable[tuple[Any, Path, str, str]]) -> None:
        for value, expected_path, filename, label in references:
            if value is not None and not _same_file_reference(value, expected_path, filename):
                _fail(f"{label} does not reference this {filename}")

    check_references(
        (metadata_map.get(key), paths["manifest"], "manifest.json", f"trajectory.metadata.{key}")
        for key in ("manifest", "manifest_path")
    )
    evidence = validation.get("evidence")
    evidence_map = _require_mapping(evidence, "validation.evidence")
    manifest_validation = manifest.get("validation")
    references = [
        (evidence_map.get("source"), paths["trajectory"], "trajectory.json", "validation.evidence.source")
    ]
    if isinstance(manifest_validation, Mapping):
        references.append((manifest_validation.get("input"), paths["trajectory"], "trajectory.json", "manifest.validation.input"))
    references.extend(
        (validation.get(key), paths["trajectory"], "trajectory.json", f"validation.{key}")
        for key in ("input", "trajectory", "trajectory_path")
    )
    check_references(references)

    provenance = _require_mapping(manifest.get("provenance"), "manifest.provenance")
    evidence_class = provenance.get("evidence_class")
    if not isinstance(evidence_class, str) or not evidence_class.strip():
        _fail("manifest.provenance.evidence_class must be a non-empty string")
    evidence_class = evidence_class.strip()
    trajectory_metadata = metadata_map
    metadata_class = trajectory_metadata.get("evidence_class")
    validation_class = evidence_map.get("class")
    if not isinstance(metadata_class, str) or not metadata_class.strip():
        _fail("trajectory.metadata.evidence_class must be a non-empty string")
    if not isinstance(validation_class, str) or not validation_class.strip():
        _fail("validation.evidence.class must be a non-empty string")
    declared_classes = [
        ("manifest.provenance.evidence_class", evidence_class),
        ("trajectory.metadata.evidence_class", metadata_class.strip()),
        ("validation.evidence.class", validation_class.strip()),
    ]
    first_name, first_value = declared_classes[0]
    for name, value in declared_classes[1:]:
        if value != first_value:
            _fail(f"evidence class mismatch: {first_name}={first_value!r}, {name}={value!r}")
    if first_value != EXPECTED_EVIDENCE_CLASS:
        _fail(
            "unsupported evidence class for a digital-flight bundle: "
            f"{first_value!r}; expected {EXPECTED_EVIDENCE_CLASS!r}"
        )

    # A digital artifact may be mechanically useful, but it must never be
    # finalized as biological evidence.  Preserve the explicit boundary in
    # the sidecar and reject contradictory declarations here.
    if evidence_map.get("real_biological") is True:
        _fail("validation.evidence.real_biological cannot be true for digital_simulation")
    if isinstance(manifest_validation, Mapping):
        expected_class = manifest_validation.get("expected_evidence_class")
        if expected_class is not None and str(expected_class) != EXPECTED_EVIDENCE_CLASS:
            _fail(
                "manifest.validation.expected_evidence_class must be "
                f"{EXPECTED_EVIDENCE_CLASS!r}"
            )
        if manifest_validation.get("biological_claim") is True:
            _fail("manifest.validation.biological_claim must be false for digital_simulation")

    sources = (
        ("manifest", manifest),
        ("trajectory.metadata", trajectory_metadata),
        ("validation", validation),
    )
    for field in ("seed", "run_id"):
        declarations = [
            (f"{source_name}.{field}", str(source[field]) if field == "run_id" else source[field])
            for source_name, source in sources
            if source.get(field) is not None
        ]
        if declarations:
            first_name, first_value = declarations[0]
            for name, value in declarations[1:]:
                if value != first_value:
                    _fail(f"{field} mismatch: {first_name}={first_value!r}, {name}={value!r}")

    _check_backend_declarations(manifest)


def _check_trajectory_telemetry(
    trajectory_rows: Sequence[Mapping[str, Any]], telemetry_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(trajectory_rows) != len(telemetry_rows):
        _fail(
            "trajectory/telemetry row count mismatch: "
            f"trajectory={len(trajectory_rows)}, telemetry={len(telemetry_rows)}"
        )
    for index, (trajectory_row, telemetry_row) in enumerate(zip(trajectory_rows, telemetry_rows)):
        if _canonical_json(trajectory_row) != _canonical_json(telemetry_row):
            _fail(f"trajectory/telemetry row mismatch at index {index}")


def _sha256_and_size(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise FinalizationError(f"cannot hash {path}: {exc}") from exc
    if size <= 0:
        _fail(f"required input became empty while hashing: {path}")
    return {
        "path": path.name,
        "size_bytes": size,
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _run_ffprobe(video_path: Path, ffprobe: os.PathLike[str] | str = "ffprobe") -> Mapping[str, Any]:
    """Run ffprobe with frame counting enabled and parse its JSON output."""

    executable = str(ffprobe)
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_type,codec_name,codec_long_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except (OSError, ValueError) as exc:
        raise FinalizationError(f"could not run ffprobe for {video_path}: {exc}") from exc
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    if completed.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"exit status {completed.returncode}"
        _fail(f"ffprobe failed for {video_path}: {detail}")
    if not stdout.strip():
        _fail(f"ffprobe returned empty output for {video_path}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FinalizationError(f"ffprobe returned invalid JSON for {video_path}: {exc}") from exc
    return _require_mapping(payload, "ffprobe output")


def _parse_rate(value: Any, label: str) -> Optional[float]:
    if value is None or str(value).strip() in {"", "N/A", "0/0", "0"}:
        return None
    text = str(value).strip()
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", 1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            if denominator == 0.0:
                return None
            result = numerator / denominator
        else:
            result = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalizationError(f"{label} is not a valid frame rate: {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def _normalize_probe(raw: Mapping[str, Any], video_path: Path) -> dict[str, Any]:
    streams_value = raw.get("streams")
    if not isinstance(streams_value, list) or not streams_value:
        _fail("ffprobe found no video streams")
    streams: list[Mapping[str, Any]] = []
    for index, stream in enumerate(streams_value):
        stream_map = _require_mapping(stream, f"ffprobe stream {index}")
        streams.append(stream_map)
    video_streams = [stream for stream in streams if str(stream.get("codec_type", "video")) == "video"]
    if not video_streams:
        _fail("ffprobe found no usable video stream")
    if len(video_streams) > 1:
        # The command selects v:0, but silently choosing among multiple video
        # streams would make a frame mapping ambiguous for an audit bundle.
        _fail(f"ffprobe found {len(video_streams)} video streams; expected exactly one")
    stream = video_streams[0]
    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec.strip() or codec.strip().upper() == "N/A":
        _fail("ffprobe video stream has no codec_name")
    width = _finite_int(stream.get("width"), "video width", positive=True)
    height = _finite_int(stream.get("height"), "video height", positive=True)

    avg_rate = _parse_rate(stream.get("avg_frame_rate"), "avg_frame_rate")
    nominal_rate = _parse_rate(stream.get("r_frame_rate"), "r_frame_rate")
    fps = avg_rate or nominal_rate
    if fps is None:
        _fail("ffprobe video stream has no positive frame rate")
    if avg_rate is not None and nominal_rate is not None:
        tolerance = max(1.0e-6, 1.0e-3 * max(abs(avg_rate), abs(nominal_rate)))
        if abs(avg_rate - nominal_rate) > tolerance:
            _fail(f"video frame rate is not stable: avg_frame_rate={avg_rate:g}, r_frame_rate={nominal_rate:g}")

    format_map = raw.get("format")
    format_map = _require_mapping(format_map, "ffprobe format") if format_map is not None else {}
    duration_value = stream.get("duration")
    if duration_value in (None, "N/A", "") or _number_if_finite(duration_value) in (None, 0.0):
        duration_value = format_map.get("duration")
    duration_s = _finite_float(duration_value, "video duration", positive=True)
    stream_duration = _number_if_finite(stream.get("duration"))
    format_duration = _number_if_finite(format_map.get("duration"))
    if stream_duration is not None and format_duration is not None and stream_duration > 0.0 and format_duration > 0.0:
        duration_tolerance = max(1.0e-6, 1.0e-3 * max(stream_duration, format_duration))
        if abs(stream_duration - format_duration) > duration_tolerance:
            _fail(
                "ffprobe stream/format duration mismatch: "
                f"stream={stream_duration:g}, format={format_duration:g}"
            )

    frame_value = stream.get("nb_read_frames")
    if frame_value in (None, "N/A", "") or _number_if_finite(frame_value) in (None, 0.0):
        frame_value = stream.get("nb_frames")
    frame_count = _finite_int(frame_value, "video frame count", positive=True)

    reported_size = format_map.get("size")
    if reported_size not in (None, "", "N/A"):
        format_size = _finite_int(reported_size, "ffprobe format size", positive=True)
        actual_size = video_path.stat().st_size
        if format_size != actual_size:
            _fail(f"ffprobe format size {format_size} does not match flight.mp4 size {actual_size}")

    # Preserve all raw ffprobe fields, including streams/format, and add a
    # normalized view that does not require callers to parse rational strings.
    # Fill the two common sidecar fields when an ffprobe build exposes the
    # counted frames as ``nb_read_frames`` or keeps duration only at format
    # level.  This keeps the documented jq probe (which reads nb_frames) useful
    # without discarding any raw fields.
    normalized: dict[str, Any] = dict(raw)
    normalized_streams = [dict(item) for item in streams]
    normalized_stream = normalized_streams[streams.index(stream)]
    normalized_stream.setdefault("codec_name", str(codec))
    normalized_stream.setdefault("width", width)
    normalized_stream.setdefault("height", height)
    normalized_stream.setdefault("avg_frame_rate", str(stream.get("avg_frame_rate") or fps))
    normalized_stream.setdefault("r_frame_rate", str(stream.get("r_frame_rate") or fps))
    normalized_stream["nb_frames"] = str(frame_count)
    normalized_stream["duration"] = str(duration_s)
    normalized["streams"] = normalized_streams
    normalized_format = dict(format_map)
    normalized_format.setdefault("duration", str(duration_s))
    normalized_format.setdefault("size", str(video_path.stat().st_size))
    normalized["format"] = normalized_format
    normalized.update(
        schema_version=VIDEO_PROBE_SCHEMA_VERSION,
        video=video_path.name,
        codec=str(codec),
        codec_name=str(codec),
        width=width,
        height=height,
        dimensions={"width": width, "height": height},
        fps=fps,
        playback_fps=fps,
        duration_s=duration_s,
        duration=duration_s,
        video_duration_s=duration_s,
        frame_count=frame_count,
        video_frame_count=frame_count,
        video_stream={
            "codec": str(codec),
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration_s,
            "frame_count": frame_count,
        },
    )
    return normalized


def _number_if_finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_number(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in mapping:
            value = _number_if_finite(mapping[key])
            if value is not None and value > 0.0:
                return value
    return None


def _declared_simulation_duration(
    manifest: Mapping[str, Any], trajectory: Mapping[str, Any], validation: Mapping[str, Any], trace_span_s: float
) -> tuple[float, str]:
    """Find a declared simulation duration without confusing sample cadence for it."""

    metadata = trajectory.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    for source, mapping in (
        ("trajectory.metadata", metadata_map),
        ("manifest", manifest),
        ("validation", validation),
    ):
        candidate = _first_number(
            mapping,
            ("simulation_duration_s", "mission_duration_s", "sim_duration_s", "duration_s"),
        )
        if candidate is not None:
            return candidate, source
        for nested_name in ("simulation", "mission", "timing", "mapping"):
            nested = mapping.get(nested_name)
            if isinstance(nested, Mapping):
                candidate = _first_number(
                    nested,
                    ("simulation_duration_s", "mission_duration_s", "sim_duration_s", "duration_s"),
                )
                if candidate is not None:
                    return candidate, f"{source}.{nested_name}"

    scenario_results = manifest.get("scenario_results")
    if isinstance(scenario_results, Mapping) and scenario_results:
        durations: list[float] = []
        for name, result in scenario_results.items():
            if not isinstance(result, Mapping):
                continue
            value = _first_number(result, ("duration_s", "simulation_duration_s"))
            if value is None and isinstance(result.get("metrics"), Mapping):
                value = _first_number(result["metrics"], ("duration_s", "simulation_duration_s"))
            if value is not None:
                durations.append(value)
        if durations:
            return sum(durations), "manifest.scenario_results"

    if trace_span_s > 0.0:
        return trace_span_s, "trajectory.time_s span"
    # A one-sample run can still carry a declared duration in a row or in a
    # validation report.  Search rows only for explicit names, never arbitrary
    # ``duration_s`` fields from nested diagnostics.
    rows_value = trajectory.get("trajectory")
    if isinstance(rows_value, list) and rows_value:
        row = rows_value[-1]
        if isinstance(row, Mapping):
            candidate = _first_number(row, ("simulation_duration_s", "mission_duration_s"))
            if candidate is not None:
                return candidate, "trajectory row"
    return trace_span_s, "trajectory.time_s span"


def _declared_playback(
    manifest: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], str]:
    """Read optional explicit playback declarations.

    ``frame_rate_hz`` in trajectory metadata is a simulation/control cadence,
    not a playback rate, so it is intentionally excluded here.  Producers can
    declare video settings under a ``video``, ``render`` or ``playback`` map,
    or with the explicit top-level aliases supported below.
    """

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for source, root in (("trajectory.metadata", trajectory.get("metadata")), ("manifest", manifest)):
        if not isinstance(root, Mapping):
            continue
        for key in ("playback", "video", "render", "video_mapping", "mapping"):
            nested = root.get(key)
            if isinstance(nested, Mapping):
                candidates.append((f"{source}.{key}", nested))
        candidates.append((source, root))

    fps: Optional[float] = None
    duration: Optional[float] = None
    playback_rate: Optional[float] = None
    frame_count: Optional[int] = None
    source = "derived from ffprobe"
    for candidate_source, mapping in candidates:
        is_nested_playback = candidate_source.rsplit(".", 1)[-1] in {
            "playback", "video", "render", "video_mapping"
        }
        fps_keys = ("playback_fps", "render_fps", "output_fps", "video_fps") + (("fps",) if is_nested_playback else ())
        duration_keys = ("playback_duration_s", "video_duration_s") + (("duration_s",) if is_nested_playback else ())
        frame_count_keys = ("playback_frame_count", "video_frame_count")
        if is_nested_playback:
            frame_count_keys += ("frame_count", "video_frames")
        if fps is None:
            fps = _first_number(mapping, fps_keys)
        if duration is None:
            duration = _first_number(mapping, duration_keys)
        if playback_rate is None:
            playback_rate = _first_number(mapping, ("playback_rate", "speed", "time_scale"))
        if frame_count is None:
            for frame_key in frame_count_keys:
                if frame_key in mapping:
                    frame_count = _finite_int(
                        mapping[frame_key], f"{candidate_source}.{frame_key}", positive=True
                    )
                    break
        if fps is not None or duration is not None or playback_rate is not None or frame_count is not None:
            source = candidate_source
    return fps, duration, playback_rate, frame_count, source


def _mapping_payload(
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    validation: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    trace_start_s: float,
    trace_end_s: float,
    trace_span_s: float,
    video_probe: Mapping[str, Any],
) -> dict[str, Any]:
    simulation_duration_s, simulation_source = _declared_simulation_duration(
        manifest, trajectory, validation, trace_span_s
    )
    if simulation_duration_s < 0.0 or not math.isfinite(simulation_duration_s):
        _fail(f"declared simulation duration is invalid: {simulation_duration_s!r}")
    if trace_span_s > simulation_duration_s + max(1.0e-9, 1.0e-6 * max(1.0, simulation_duration_s)):
        _fail(
            "trajectory time span exceeds declared simulation duration: "
            f"span={trace_span_s:g}, declared={simulation_duration_s:g}"
        )

    video_fps = _finite_float(video_probe["fps"], "normalized video fps", positive=True)
    video_duration_s = _finite_float(video_probe["duration_s"], "normalized video duration", positive=True)
    video_frame_count = _finite_int(video_probe["frame_count"], "normalized video frame count", positive=True)
    declared_fps, declared_playback_duration, declared_rate, declared_frame_count, declared_source = _declared_playback(
        manifest, trajectory
    )
    if declared_fps is not None:
        tolerance = max(1.0e-6, 1.0e-3 * max(abs(declared_fps), abs(video_fps)))
        if abs(declared_fps - video_fps) > tolerance:
            _fail(f"declared playback FPS {declared_fps:g} does not match video FPS {video_fps:g}")
    if declared_playback_duration is not None:
        tolerance = max(1.0e-6, 1.0e-3 * max(abs(declared_playback_duration), abs(video_duration_s)))
        if abs(declared_playback_duration - video_duration_s) > tolerance:
            _fail(
                "declared playback duration "
                f"{declared_playback_duration:g} does not match video duration {video_duration_s:g}"
            )
    if declared_frame_count is not None and declared_frame_count != video_frame_count:
        _fail(
            f"declared playback frame count {declared_frame_count} does not match video frame count {video_frame_count}"
        )

    # ``playback_rate`` is expressed as simulation seconds per video second;
    # this is unambiguous even when a video was intentionally slowed down.
    derived_rate = simulation_duration_s / video_duration_s if video_duration_s > 0.0 else None
    if declared_rate is not None:
        rate_tolerance = max(1.0e-6, 1.0e-3 * max(abs(declared_rate), abs(derived_rate or declared_rate)))
        if derived_rate is not None and abs(declared_rate - derived_rate) > rate_tolerance:
            _fail(
                "declared playback rate "
                f"{declared_rate:g} does not match simulation/video duration ratio {derived_rate:g}"
            )
    effective_rate = declared_rate if declared_rate is not None else derived_rate

    frame_map_kind = "one_to_one" if video_frame_count == len(trajectory_rows) else "time_resampled"
    mapping: dict[str, Any] = {
        "schema_version": "flight-simulation-playback-mapping-1",
        "simulation": {
            "duration_s": simulation_duration_s,
            "time_start_s": trace_start_s,
            "time_end_s": trace_end_s,
            "time_span_s": trace_span_s,
            "trajectory_rows": len(trajectory_rows),
            "duration_source": simulation_source,
        },
        "playback": {
            "duration_s": video_duration_s,
            "fps": video_fps,
            "frame_count": video_frame_count,
            "simulation_seconds_per_video_second": effective_rate,
            "declared_fps": declared_fps,
            "declared_duration_s": declared_playback_duration,
            "declared_rate": declared_rate,
            "declared_frame_count": declared_frame_count,
            "declaration_source": declared_source,
        },
        "frame_to_trajectory": {
            "kind": frame_map_kind,
            "video_frames": video_frame_count,
            "trajectory_rows": len(trajectory_rows),
            "trajectory_time_start_s": trace_start_s,
            "trajectory_time_end_s": trace_end_s,
            "simulation_duration_s": simulation_duration_s,
            "playback_duration_s": video_duration_s,
            "rule": "linear time mapping over the declared simulation interval",
        },
    }
    # Useful top-level aliases keep the sidecar convenient for simple jq
    # checks, while the nested mapping remains the canonical declaration.
    mapping["simulation_duration_s"] = simulation_duration_s
    mapping["playback_duration_s"] = video_duration_s
    mapping["playback_rate"] = effective_rate
    mapping["trajectory_rows"] = len(trajectory_rows)
    mapping["video_frames"] = video_frame_count
    return mapping


def _delivery_metadata(
    manifest: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    """Project compact provenance, resource, and evidence metadata into integrity.json."""

    provenance = _require_mapping(manifest.get("provenance"), "manifest.provenance")
    evidence = _require_mapping(validation.get("evidence"), "validation.evidence")
    manifest_validation = manifest.get("validation")
    manifest_validation_map = (
        _require_mapping(manifest_validation, "manifest.validation")
        if manifest_validation is not None else {}
    )

    components_value = manifest.get("components")
    components = (
        _require_mapping(components_value, "manifest.components")
        if components_value is not None else {}
    )
    component_summary: dict[str, Any] = {}
    for name, raw_component in components.items():
        if not isinstance(raw_component, Mapping):
            continue
        summary = {
            key: raw_component[key]
            for key in ("is_stub", "source", "backend", "effective_backend", "effective_device", "reduced")
            if key in raw_component
        }
        runtime = raw_component.get("runtime")
        if isinstance(runtime, Mapping):
            runtime_summary = {
                key: runtime[key]
                for key in ("backend", "device", "n", "neurons", "edges", "synapses", "n_targets", "units")
                if key in runtime
            }
            if runtime_summary:
                summary["runtime"] = runtime_summary
        if summary:
            component_summary[str(name)] = summary

    resource_limits = manifest.get("resource_limits")
    if resource_limits is not None:
        resource_limits = _require_mapping(resource_limits, "manifest.resource_limits")
    resource_usage = manifest.get("resource_usage")
    if resource_usage is not None:
        resource_usage = _require_mapping(resource_usage, "manifest.resource_usage")

    evidence_class = str(evidence.get("class", provenance.get("evidence_class")))
    return {
        "provenance": {
            "evidence_class": str(provenance.get("evidence_class")),
            "component_labels": dict(provenance),
            "requested_backends": manifest.get("requested_backends"),
            "effective_backends": manifest.get("effective_backends"),
            "components": component_summary,
        },
        "resources": {
            "declared": resource_limits is not None or resource_usage is not None,
            "limits": resource_limits,
            "usage": resource_usage,
        },
        "evidence": {
            "class": evidence_class,
            "real_biological": evidence.get("real_biological", False),
            "synthetic_fixture": evidence.get("synthetic_fixture", False),
            "decision": validation.get("decision"),
            "validation_status": validation.get("validation_status", validation.get("status")),
            "mechanical_decision": validation.get("mechanical_decision"),
            "biological_claim": manifest_validation_map.get("biological_claim", False),
            "claim_boundary": "digital_simulation; not biological validation",
            "limitations": validation.get("limitations"),
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one derived sidecar atomically in its target directory."""

    temporary_path: Optional[Path] = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as exc:
        raise FinalizationError(f"cannot write derived sidecar {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def finalize_flight_artifact(
    run_dir: os.PathLike[str] | str,
    *,
    ffprobe: os.PathLike[str] | str = "ffprobe",
) -> dict[str, Any]:
    """Finalize ``run_dir`` and return the generated integrity payload.

    All required inputs are read and verified before either derived output is
    replaced.  If verification fails, existing ``video_probe.json`` and
    ``integrity.json`` are left untouched.
    """

    directory = Path(run_dir).expanduser()
    paths = _read_required_files(directory)
    manifest = _require_mapping(_load_json(paths["manifest"], "manifest"), "manifest")
    trajectory = _require_mapping(_load_json(paths["trajectory"], "trajectory"), "trajectory")
    validation = _require_mapping(_load_json(paths["validation"], "validation"), "validation")
    trajectory_rows = _trajectory_rows(trajectory)
    telemetry_rows = _read_telemetry(paths["telemetry"])
    _check_trajectory_telemetry(trajectory_rows, telemetry_rows)
    trace_start_s, trace_end_s, trace_span_s = _trajectory_times(trajectory_rows)
    _check_declared_references(manifest, trajectory, validation, paths)

    # Hash all scientific/video inputs before writing any generated sidecar.
    file_records = {logical: _sha256_and_size(path) for logical, path in paths.items()}
    raw_probe = _run_ffprobe(paths["video"], ffprobe=ffprobe)
    normalized_probe = _normalize_probe(raw_probe, paths["video"])
    mapping = _mapping_payload(
        manifest,
        trajectory,
        validation,
        trajectory_rows,
        trace_start_s,
        trace_end_s,
        trace_span_s,
        normalized_probe,
    )
    normalized_probe["mapping"] = mapping
    normalized_probe["simulation"] = mapping["simulation"]
    normalized_probe["playback"] = mapping["playback"]
    normalized_probe["frame_to_trajectory"] = mapping["frame_to_trajectory"]
    normalized_probe["simulation_duration_s"] = mapping["simulation_duration_s"]
    normalized_probe["playback_duration_s"] = mapping["playback_duration_s"]
    normalized_probe["playback_rate"] = mapping["playback_rate"]

    video_probe_path = directory / "video_probe.json"
    integrity_path = directory / "integrity.json"
    # Probe is written first because integrity records its generated hash.  It
    # is intentionally separate from the five required scientific/video
    # records, so integrity never hashes itself.
    _atomic_write_json(video_probe_path, normalized_probe)
    video_probe_record = _sha256_and_size(video_probe_path)

    files_by_filename = {record["path"]: dict(record) for record in file_records.values()}
    artifacts = {logical: dict(record) for logical, record in file_records.items()}
    hashes = {record["path"]: record["sha256"] for record in file_records.values()}
    sizes = {record["path"]: record["size_bytes"] for record in file_records.values()}
    delivery_metadata = _delivery_metadata(manifest, validation)
    integrity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_directory": directory.name,
        "complete": True,
        "files": files_by_filename,
        "artifacts": artifacts,
        "hashes": hashes,
        "sizes": sizes,
        "generated_files": {"video_probe.json": video_probe_record},
        "video_probe": {
            "path": "video_probe.json",
            "sha256": video_probe_record["sha256"],
            "size_bytes": video_probe_record["size_bytes"],
        },
        # These are projections of already-hashed declarations.  They make
        # the delivery boundary reviewable without pretending to be new
        # measurements or biological validation.
        "provenance": delivery_metadata["provenance"],
        "resources": delivery_metadata["resources"],
        "evidence": delivery_metadata["evidence"],
        "mapping": mapping,
    }
    _atomic_write_json(integrity_path, integrity)
    return integrity


# Short aliases make the module pleasant to use from tests and small scripts.
finalize_run = finalize_flight_artifact
finalize_artifact = finalize_flight_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="completed flight artifact directory")
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable (default: ffprobe on PATH)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        integrity = finalize_flight_artifact(args.run_dir, ffprobe=args.ffprobe)
    except FinalizationError as exc:
        print(f"finalize_flight_artifact: ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_directory": integrity["run_directory"],
                "video_probe": str(Path(args.run_dir) / "video_probe.json"),
                "integrity": str(Path(args.run_dir) / "integrity.json"),
                "complete": integrity["complete"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())
