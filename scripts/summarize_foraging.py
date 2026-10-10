#!/usr/bin/env python3
"""Validate and summarize completed elevated-food digital-fly runs.

This read-only analyzer accepts explicitly named experiment runs, audits their
strict-run and food-cue provenance, checks trajectory/telemetry frame identity,
and reports recorded movement, contact, neural, and motor events. Its report
describes simulator artifacts only. Odor input, contact-gated sugar input,
motor readouts, and movement are not evidence of biological behavior or food
ingestion.

Example::

    python3 scripts/summarize_foraging.py \
        --intact runs/intact --odor-ablated runs/odor-off \
        --contact-positive runs/contact-positive \
        --no-food runs/no-food \
        --output-json /data/artifacts/by-project/malecns-gpu/experiments/foraging-20260925/summary.json \
        --output-markdown /data/artifacts/by-project/malecns-gpu/experiments/foraging-20260925/summary.md

The no-food run is optional. The analyzer never changes input artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "elevated-food-foraging-summary-1"
STRICT_POLICY = "strict-free-behavior-escape-experiment"
NOT_VALIDATED = "NOT_VALIDATED"
ROLES = ("intact", "odor_ablated", "contact_positive", "contact_taste_ablated", "no_food")
EXPECTED_ABLATIONS = {
    "intact": (),
    "odor_ablated": ("odor",),
    "contact_positive": (),  # The contact arm may also be odor-ablated when paired with its taste control.
    "contact_taste_ablated": ("taste",),
    "no_food": (),
}
PAIR_EXPERIMENT_IGNORES = {"name", "description"}
CONTACT_EXPERIMENT_IGNORES = {"name", "description", "initial_pose", "resting_start"}


class ForagingSummaryError(ValueError):
    """Raised when an input run is incomplete or cannot be safely compared."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON number {value!r}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ForagingSummaryError(f"cannot read JSON artifact {path}: {exc}") from exc


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _required_number(value: Any, label: str, where: str) -> float:
    result = _number(value)
    if result is None:
        raise ForagingSummaryError(f"{where}: {label} must be a finite number")
    return result


def _vec3(value: Any, label: str, where: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ForagingSummaryError(f"{where}: {label} must be a 3-vector")
    return tuple(_required_number(item, f"{label}[{index}]", where) for index, item in enumerate(value))  # type: ignore[return-value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ForagingSummaryError(f"cannot hash provenance file {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_output(run_dir: Path, manifest: Mapping[str, Any], name: str,
                 fallback: str) -> Path:
    raw = _mapping(manifest.get("outputs")).get(name)
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = run_dir / candidate
    else:
        candidate = run_dir / fallback
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run_dir.resolve())
    except (OSError, ValueError) as exc:
        raise ForagingSummaryError(
            f"{run_dir}: {name} artifact is missing or resolves outside the run directory: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise ForagingSummaryError(f"{run_dir}: {name} artifact is not a file: {resolved}")
    return resolved


def _read_telemetry(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ForagingSummaryError(f"invalid telemetry JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, Mapping):
                    raise ForagingSummaryError(f"telemetry row {line_no} in {path} is not an object")
                rows.append(row)
    except OSError as exc:
        raise ForagingSummaryError(f"cannot read telemetry {path}: {exc}") from exc
    if not rows:
        raise ForagingSummaryError(f"telemetry {path} contains no frame rows")
    return rows


def _trajectory_rows(payload: Any, path: Path) -> list[Mapping[str, Any]]:
    rows = payload.get("trajectory") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise ForagingSummaryError(f"trajectory {path} must contain a nonempty row list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ForagingSummaryError(f"trajectory {path} contains a non-object row")
    return list(rows)


def _identity_value(row: Mapping[str, Any], key: str, where: str) -> Any:
    value = row.get(key)
    if value is None:
        raise ForagingSummaryError(f"{where}: frame identity is missing {key}")
    return value


def _check_streams(trajectory: Sequence[Mapping[str, Any]], telemetry: Sequence[Mapping[str, Any]],
                   run_dir: Path) -> list[dict[str, Any]]:
    if len(trajectory) != len(telemetry):
        raise ForagingSummaryError(
            f"{run_dir}: trajectory has {len(trajectory)} rows but telemetry has {len(telemetry)}"
        )
    frames: list[dict[str, Any]] = []
    prior_frame: int | None = None
    prior_input_time: float | None = None
    prior_state_time: float | None = None
    reference_dt: float | None = None
    for index, (traj, telem) in enumerate(zip(trajectory, telemetry)):
        where = f"{run_dir} row {index}"
        frame_t = _identity_value(traj, "frame_index", where)
        frame_m = _identity_value(telem, "frame_index", where)
        if (isinstance(frame_t, bool) or not isinstance(frame_t, int)
                or isinstance(frame_m, bool) or not isinstance(frame_m, int) or frame_t != frame_m):
            raise ForagingSummaryError(f"{where}: trajectory/telemetry frame_index mismatch")
        if frame_t != index or (prior_frame is not None and frame_t != prior_frame + 1):
            raise ForagingSummaryError(f"{where}: frame indices must be contiguous from zero")
        input_time_t = _required_number(traj.get("frame_time_s"), "frame_time_s", where)
        input_time_m = _required_number(telem.get("frame_time_s"), "frame_time_s", where)
        state_time_t = _required_number(traj.get("time_s"), "time_s", where)
        state_time_m = _required_number(telem.get("time_s"), "time_s", where)
        for label, left, right in (
            ("frame_time_s", input_time_t, input_time_m), ("time_s", state_time_t, state_time_m),
        ):
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9):
                raise ForagingSummaryError(f"{where}: trajectory/telemetry {label} mismatch")
        dt_s = _required_number(traj.get("dt_s"), "dt_s", where)
        if dt_s <= 0 or state_time_t < input_time_t or not math.isclose(
                state_time_t - input_time_t, dt_s, rel_tol=0.0, abs_tol=1e-8):
            raise ForagingSummaryError(f"{where}: frame interval identity is inconsistent")
        if reference_dt is None:
            reference_dt = dt_s
        elif not math.isclose(dt_s, reference_dt, rel_tol=0.0, abs_tol=1e-12):
            raise ForagingSummaryError(f"{where}: dt_s varies within the run")
        if not math.isclose(input_time_t, frame_t * dt_s, rel_tol=0.0, abs_tol=1e-8):
            raise ForagingSummaryError(f"{where}: frame_time_s does not match frame_index and dt_s")
        for key in ("scenario", "config_scenario"):
            if key in traj and key in telem and traj.get(key) != telem.get(key):
                raise ForagingSummaryError(f"{where}: trajectory/telemetry {key} differs")
        if prior_input_time is not None and input_time_t <= prior_input_time:
            raise ForagingSummaryError(f"{where}: frame input times are not strictly increasing")
        if prior_state_time is not None and state_time_t <= prior_state_time:
            raise ForagingSummaryError(f"{where}: body-state times are not strictly increasing")
        frames.append({
            "frame_index": frame_t,
            "frame_time_s": input_time_t,
            "state_time_s": state_time_t,
            "dt_s": dt_s,
        })
        prior_frame, prior_input_time, prior_state_time = frame_t, input_time_t, state_time_t
    return frames


def _ablation_list(value: Any, label: str, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ForagingSummaryError(f"{where}: {label} must be a list of sense names")
    normalized = tuple(sorted(item.strip().lower() for item in value))
    if len(set(normalized)) != len(normalized):
        raise ForagingSummaryError(f"{where}: {label} contains duplicates")
    if set(normalized) - {"odor", "taste"}:
        raise ForagingSummaryError(f"{where}: {label} includes an unknown food sense")
    return normalized


def _resolve_food_config(run_dir: Path, manifest: Mapping[str, Any],
                         food: Mapping[str, Any]) -> tuple[Mapping[str, Any], str, Path]:
    where = str(run_dir)
    requested = _mapping(manifest.get("requested_execution"))
    path_raw = requested.get("food_config")
    requested_hash = requested.get("food_config_sha256")
    component_path = food.get("config_path")
    component_hash = food.get("config_sha256")
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise ForagingSummaryError(f"{where}: requested_execution.food_config is missing")
    if not isinstance(component_path, str) or not component_path.strip():
        raise ForagingSummaryError(f"{where}: components.food_sensing.config_path is missing")
    if not isinstance(requested_hash, str) or not isinstance(component_hash, str):
        raise ForagingSummaryError(f"{where}: food config SHA-256 provenance is missing")
    if requested_hash.lower() != component_hash.lower():
        raise ForagingSummaryError(f"{where}: requested and component food config hashes differ")
    path_a = Path(path_raw).expanduser()
    path_b = Path(component_path).expanduser()
    if not path_a.is_absolute():
        path_a = run_dir / path_a
    if not path_b.is_absolute():
        path_b = run_dir / path_b
    try:
        resolved_a, resolved_b = path_a.resolve(strict=True), path_b.resolve(strict=True)
    except OSError as exc:
        raise ForagingSummaryError(f"{where}: food config source is unavailable: {exc}") from exc
    if resolved_a != resolved_b or not resolved_a.is_file():
        raise ForagingSummaryError(f"{where}: requested/component food config paths disagree")
    actual_hash = _sha256_file(resolved_a)
    if actual_hash.lower() != requested_hash.lower():
        raise ForagingSummaryError(f"{where}: food config file hash does not match manifest provenance")
    config = _read_json(resolved_a)
    embedded = food.get("config")
    if not isinstance(config, Mapping) or not isinstance(embedded, Mapping) or dict(config) != dict(embedded):
        raise ForagingSummaryError(f"{where}: embedded food config differs from the source file")
    return config, actual_hash, resolved_a


def _validate_strict_manifest(role: str, run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    where = f"{role} run {run_dir}"
    if manifest.get("schema_version") != "digital-fly-run-1":
        raise ForagingSummaryError(f"{where}: unsupported or missing run manifest schema")
    if manifest.get("execution_policy") != STRICT_POLICY:
        raise ForagingSummaryError(f"{where}: execution_policy is not the strict free-behavior policy")
    policy = _mapping(manifest.get("autonomy_policy"))
    if policy.get("strict") is not True or policy.get("control_authority") != "MaleCNS" or policy.get("fail_closed") is not True:
        raise ForagingSummaryError(f"{where}: strict MaleCNS autonomy provenance is incomplete")
    requested = _mapping(manifest.get("requested_execution"))
    requested_backends = _mapping(manifest.get("requested_backends"))
    effective_backends = _mapping(manifest.get("effective_backends"))
    components = _mapping(manifest.get("components"))
    if requested.get("strict") is not True or requested.get("mode") != STRICT_POLICY:
        raise ForagingSummaryError(f"{where}: requested_execution does not record strict mode")
    if requested_backends.get("neural") != "malecns":
        raise ForagingSummaryError(f"{where}: requested neural backend is not MaleCNS")
    neural = _mapping(components.get("neural"))
    if neural.get("is_stub") is not False or effective_backends.get("neural") in (None, "deterministic_neural_stub"):
        raise ForagingSummaryError(f"{where}: effective neural backend is missing or stubbed")
    for name in ("requested_backends", "effective_backends"):
        if not _mapping(manifest.get(name)):
            raise ForagingSummaryError(f"{where}: {name} identity is missing")
    limits = _mapping(manifest.get("resource_limits"))
    max_vram = _number(requested.get("max_vram_gib"))
    if max_vram is None or max_vram <= 0:
        raise ForagingSummaryError(f"{where}: requested_execution.max_vram_gib must be positive")
    allocator_cap = _number(limits.get("torch_allocator_max_vram_gib"))
    if allocator_cap is not None and allocator_cap > max_vram + 1e-9:
        raise ForagingSummaryError(f"{where}: allocator VRAM cap exceeds requested VRAM cap")
    caps = {
        "requested_max_vram_gib": max_vram,
        "torch_allocator_max_vram_gib": allocator_cap,
        "cgroup_memory_max_bytes": _number(limits.get("cgroup_memory_max_bytes")),
    }
    identity = {
        "seed": manifest.get("seed"),
        "execution_policy": manifest.get("execution_policy"),
        "requested_backends": dict(requested_backends),
        "effective_backends": dict(effective_backends),
        "requested_renderer_backend": manifest.get("requested_renderer_backend"),
        "effective_renderer_backend": _mapping(manifest.get("effective_execution")).get("renderer_backend"),
        "resource_caps": caps,
    }
    if identity["seed"] is None:
        raise ForagingSummaryError(f"{where}: manifest seed is missing")
    if not isinstance(identity["requested_renderer_backend"], str) or not isinstance(identity["effective_renderer_backend"], str):
        raise ForagingSummaryError(f"{where}: renderer identity is incomplete")
    return identity


def _first_event(rows: Sequence[Mapping[str, Any]], predicate: Any, *, sensory: bool = False) -> dict[str, Any]:
    for row in rows:
        if predicate(row):
            food = _mapping(row.get("food_cue"))
            event_time = _number(food.get("time_s")) if sensory else _number(row.get("time_s"))
            return {
                "status": "observed",
                "time_s": event_time,
                "frame_index": row.get("frame_index"),
                "frame_time_s": row.get("frame_time_s"),
                "state_time_s": row.get("time_s"),
            }
    return {"status": "not_observed", "time_s": None, "frame_index": None,
            "frame_time_s": None, "state_time_s": None}


def _rate_sum(value: Any, label: str, where: str) -> float:
    if not isinstance(value, Mapping):
        raise ForagingSummaryError(f"{where}: {label} must be a selector-to-rate mapping")
    total = 0.0
    for key, raw in value.items():
        rate = _required_number(raw, f"{label}.{key}", where)
        if rate < 0:
            raise ForagingSummaryError(f"{where}: {label}.{key} must be nonnegative")
        total += rate
    if not math.isfinite(total):
        raise ForagingSummaryError(f"{where}: {label} total rate is not finite")
    return total


def _numeric_sum(value: Any) -> float | None:
    if isinstance(value, Mapping):
        samples = [_numeric_sum(child) for child in value.values()]
    elif isinstance(value, (list, tuple)):
        samples = [_numeric_sum(child) for child in value]
    else:
        number = _number(value)
        return number
    if any(sample is None for sample in samples):
        return None
    return sum(float(sample) for sample in samples)


def _summarize_population_events(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_frame: list[dict[str, Any]] = []
    population_names: set[str] = set()
    for row in rows:
        named = _mapping(_mapping(row.get("neural")).get("named_outputs"))
        counts = _mapping(named.get("population_spike_counts"))
        rates = _mapping(named.get("population_rates_hz"))
        mn9_counts = {str(name): _numeric_sum(value) for name, value in counts.items()
                      if "mn9" in str(name).casefold()}
        mn9_rates = {str(name): _numeric_sum(value) for name, value in rates.items()
                     if "mn9" in str(name).casefold()}
        population_names.update(mn9_counts)
        population_names.update(mn9_rates)
        per_frame.append({"row": row, "counts": mn9_counts, "rates": mn9_rates})

    def positive_mn9(record: Mapping[str, Any], field: str) -> bool:
        values = record[field].values()
        return any(value is not None and value > 0 for value in values)

    count_rows = [record for record in per_frame if record["counts"]]
    rate_rows = [record for record in per_frame if record["rates"]]
    max_counts = [value for record in count_rows for value in record["counts"].values() if value is not None]
    max_rates = [value for record in rate_rows for value in record["rates"].values() if value is not None]
    count_totals: dict[str, float] = {}
    for record in count_rows:
        for name, value in record["counts"].items():
            if value is not None:
                count_totals[name] = count_totals.get(name, 0.0) + value
    return {
        "MN9_population_keys": sorted(population_names),
        "first_MN9_spike": _first_event(rows, lambda row: any(
            value is not None and value > 0 for value in _mn9_values(row, "population_spike_counts")
        )),
        "first_MN9_positive_rate": _first_event(rows, lambda row: any(
            value is not None and value > 0 for value in _mn9_values(row, "population_rates_hz")
        )),
        "MN9_spike_count_sample_count": len(count_rows),
        "MN9_rate_sample_count": len(rate_rows),
        "MN9_total_spikes_by_population": count_totals,
        "MN9_max_recorded_spike_count_value": max(max_counts) if max_counts else None,
        "MN9_max_recorded_rate_hz": max(max_rates) if max_rates else None,
    }


def _mn9_values(row: Mapping[str, Any], key: str) -> list[float | None]:
    named = _mapping(_mapping(row.get("neural")).get("named_outputs"))
    values = _mapping(named.get(key))
    return [_numeric_sum(value) for name, value in values.items() if "mn9" in str(name).casefold()]


def _summarize_food_neuron_outputs(rows: Sequence[Mapping[str, Any]],
                                  selectors: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for selector in sorted(selectors):
        counts_seen: list[float] = []
        rates_seen: list[float] = []
        first_spike: dict[str, Any] | None = None
        first_rate: dict[str, Any] | None = None
        for row in rows:
            named = _mapping(_mapping(row.get("neural")).get("named_outputs"))
            counts = _mapping(named.get("population_spike_counts"))
            rates = _mapping(named.get("population_rates_hz"))
            if selector not in counts or selector not in rates:
                raise ForagingSummaryError(
                    f"frame {row.get('frame_index')}: named neural outputs omit food selector {selector!r}"
                )
            count = _required_number(counts[selector], f"population_spike_counts.{selector}", f"frame {row.get('frame_index')}")
            rate = _required_number(rates[selector], f"population_rates_hz.{selector}", f"frame {row.get('frame_index')}")
            if count < 0 or rate < 0:
                raise ForagingSummaryError(f"frame {row.get('frame_index')}: food neuron outputs cannot be negative")
            counts_seen.append(count)
            rates_seen.append(rate)
            event = {"time_s": row.get("time_s"), "frame_time_s": row.get("frame_time_s"),
                     "frame_index": row.get("frame_index")}
            if first_spike is None and count > 0:
                first_spike = event
            if first_rate is None and rate > 0:
                first_rate = event
        result[selector] = {
            "observed_samples": len(counts_seen),
            "total_spikes": sum(counts_seen),
            "maximum_spikes_per_window": max(counts_seen) if counts_seen else None,
            "first_positive_spike_count": first_spike or {"status": "not_observed", "time_s": None, "frame_index": None},
            "maximum_rate_hz": max(rates_seen) if rates_seen else None,
            "first_positive_rate": first_rate or {"status": "not_observed", "time_s": None, "frame_index": None},
        }
    return result


def _food_rate_summary(rows: Sequence[Mapping[str, Any]], field: str, *,
                       selectors: set[str] | None = None, sensory: bool = True) -> dict[str, Any]:
    role_frames: list[tuple[Mapping[str, Any], float]] = []
    for row in rows:
        cue = _mapping(row.get("food_cue"))
        rate_map = _mapping(cue.get(field))
        if selectors is not None:
            rate_map = {key: rate_map[key] for key in selectors if key in rate_map}
        role_frames.append((row, _rate_sum(rate_map, field, f"frame {row.get('frame_index')}")))
    positive = [(row, total) for row, total in role_frames if total > 0]
    max_total = max((total for _, total in role_frames), default=0.0)
    return {
        "first_positive": _first_event(rows, lambda row: _rate_sum(
            {key: value for key, value in _mapping(_mapping(row.get("food_cue")).get(field)).items()
             if selectors is None or key in selectors},
            field, f"frame {row.get('frame_index')}"
        ) > 0, sensory=sensory),
        "max_total_rate_hz": max_total,
        "positive_frame_count": len(positive),
        "frame_count": len(rows),
    }


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _cue_config_fields(config: Mapping[str, Any], where: str) -> tuple[set[str], set[str], str, tuple[float, float, float]]:
    field = _mapping(config.get("field"))
    contact_surface = config.get("contact_surface_id")
    if not isinstance(contact_surface, str) or not contact_surface:
        raise ForagingSummaryError(f"{where}: food config contact_surface_id is missing")
    center = _vec3(field.get("center_m"), "field.center_m", where)
    odor = _mapping(config.get("odor_selector_weights_hz_per_unit"))
    odor_names: set[str] = set()
    for side in ("left", "right"):
        side_map = _mapping(odor.get(side))
        if not side_map:
            raise ForagingSummaryError(f"{where}: food config odor selectors are incomplete")
        odor_names.update(str(key) for key in side_map)
    taste = _mapping(config.get("sugar_selector_rates_hz"))
    if not taste:
        raise ForagingSummaryError(f"{where}: food config sugar selectors are missing")
    return odor_names, {str(key) for key in taste}, contact_surface, center


def _audit_food_rows(role: str, rows: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]],
                     config: Mapping[str, Any], component: Mapping[str, Any],
                     ablations: tuple[str, ...]) -> None:
    where = f"{role} run"
    odor_names, taste_names, surface_id, _ = _cue_config_fields(config, where)
    expected_counts = _mapping(config.get("selector_expected_counts"))
    actual_counts = _mapping(component.get("selector_counts"))
    if expected_counts and dict(expected_counts) != dict(actual_counts):
        raise ForagingSummaryError(f"{where}: resolved selector_counts differ from food config expectations")
    if component.get("contact_surface_id") != surface_id:
        raise ForagingSummaryError(f"{where}: food contact_surface_id differs from source config")
    for row, frame in zip(rows, frames):
        index = frame["frame_index"]
        cue = row.get("food_cue")
        if not isinstance(cue, Mapping):
            raise ForagingSummaryError(f"{where} frame {index}: food_cue is missing")
        cue_time = _required_number(cue.get("time_s"), "food_cue.time_s", f"{where} frame {index}")
        if not math.isclose(cue_time, frame["frame_time_s"], rel_tol=0.0, abs_tol=1e-9):
            raise ForagingSummaryError(f"{where} frame {index}: food_cue input time differs from frame_time_s")
        if cue.get("contact_surface_id") != surface_id:
            raise ForagingSummaryError(f"{where} frame {index}: food cue surface identity mismatch")
        cue_ablations = _ablation_list(cue.get("ablated_senses"), "food_cue.ablated_senses", f"{where} frame {index}")
        if cue_ablations != ablations:
            raise ForagingSummaryError(f"{where} frame {index}: cue ablation identity differs from role")
        for name in ("physical_contact", "edible_region_contact", "contact"):
            if type(cue.get(name)) is not bool:
                raise ForagingSummaryError(f"{where} frame {index}: food_cue.{name} must be bool")
        if cue["contact"] is not (cue["physical_contact"] and cue["edible_region_contact"]):
            raise ForagingSummaryError(f"{where} frame {index}: food cue contact gate is inconsistent")
        for name in ("concentration", "left_sample", "right_sample", "mouth_distance_m"):
            value = _required_number(cue.get(name), f"food_cue.{name}", f"{where} frame {index}")
            if value < 0:
                raise ForagingSummaryError(f"{where} frame {index}: food_cue.{name} must be nonnegative")
        raw_odor = _mapping(cue.get("raw_odor_poisson_hz"))
        raw_taste = _mapping(cue.get("raw_taste_poisson_hz"))
        cns = _mapping(cue.get("cns_poisson_hz"))
        if set(raw_odor) != odor_names or set(raw_taste) != taste_names or set(cns) != odor_names | taste_names:
            raise ForagingSummaryError(f"{where} frame {index}: food selector maps disagree with configured selectors")
        odor_config = _mapping(config.get("odor_selector_weights_hz_per_unit"))
        taste_config = _mapping(config.get("sugar_selector_rates_hz"))
        antenna_samples = {"left": _required_number(cue.get("left_sample"), "food_cue.left_sample", f"{where} frame {index}"),
                           "right": _required_number(cue.get("right_sample"), "food_cue.right_sample", f"{where} frame {index}")}
        odor_weights: dict[str, float] = {}
        for side in ("left", "right"):
            for selector, weight in _mapping(odor_config.get(side)).items():
                odor_weights[str(selector)] = antenna_samples[side] * _required_number(weight, f"odor weight {selector}", where)
        for selector, expected_rate in odor_weights.items():
            if not math.isclose(_required_number(raw_odor[selector], f"raw odor rate {selector}", f"{where} frame {index}"),
                                expected_rate, rel_tol=1e-9, abs_tol=1e-9):
                raise ForagingSummaryError(f"{where} frame {index}: raw odor rate does not match sampled antenna concentration")
        for sense, selectors, raw_map in (("odor", odor_names, raw_odor), ("taste", taste_names, raw_taste)):
            for selector in selectors:
                raw_rate = _required_number(raw_map[selector], f"raw {sense} rate {selector}", f"{where} frame {index}")
                cns_rate = _required_number(cns[selector], f"CNS {sense} rate {selector}", f"{where} frame {index}")
                expected_rate = 0.0 if sense in ablations else raw_rate
                if not math.isclose(cns_rate, expected_rate, rel_tol=0.0, abs_tol=1e-9):
                    raise ForagingSummaryError(f"{where} frame {index}: raw-to-CNS {sense} ablation seam is inconsistent")
                if raw_rate < 0:
                    raise ForagingSummaryError(f"{where} frame {index}: raw {sense} rate must be nonnegative")
        for selector in taste_names:
            expected_rate = _required_number(taste_config[selector], f"configured sugar rate {selector}", where) if cue["contact"] else 0.0
            if not math.isclose(_required_number(raw_taste[selector], f"raw sugar rate {selector}", f"{where} frame {index}"),
                                expected_rate, rel_tol=0.0, abs_tol=1e-9):
                raise ForagingSummaryError(f"{where} frame {index}: raw sugar rate does not match the contact gate")


def _position(row: Mapping[str, Any], where: str) -> tuple[float, float, float]:
    raw = row.get("position_m")
    if raw is None:
        raw = _mapping(row.get("body")).get("position_m")
    return _vec3(raw, "position_m", where)


def _bool_series(rows: Sequence[Mapping[str, Any]], key: str, where: str) -> list[bool]:
    values: list[bool] = []
    for row in rows:
        value = row.get(key)
        if type(value) is not bool:
            raise ForagingSummaryError(f"{where} frame {row.get('frame_index')}: {key} must be bool")
        values.append(value)
    return values


def _trajectory_summary(rows: Sequence[Mapping[str, Any]], center: Sequence[float],
                        run_dir: Path, contact_surface_id: str) -> dict[str, Any]:
    samples: list[tuple[Mapping[str, Any], tuple[float, float, float], float]] = []
    for row in rows:
        point = _position(row, str(run_dir))
        samples.append((row, point, _distance(point, center)))
    first_row, first_position, first_distance = samples[0]
    last_row, last_position, last_distance = samples[-1]
    nearest_row, nearest_position, min_distance = min(samples, key=lambda item: item[2])
    z_values = [point[2] for _, point, _ in samples]
    distances = [distance for _, _, distance in samples]
    ground_flags = _bool_series(rows, "grounded", str(run_dir))
    target_flags = _bool_series(rows, "target_contact", str(run_dir))
    initial_grounded = ground_flags[0]
    first_airborne_index = next((i for i, value in enumerate(ground_flags) if not value), None)
    landing_index = None
    if first_airborne_index is not None:
        landing_index = next((i for i in range(first_airborne_index + 1, len(ground_flags)) if ground_flags[i]), None)
    target_index = next((i for i, value in enumerate(target_flags) if value), None)
    target_surface_index = next((i for i, row in enumerate(rows)
                                 if target_flags[i] and row.get("contact_surface_id") == contact_surface_id), None)

    def state_event(index: int | None) -> dict[str, Any]:
        if index is None:
            return {"status": "not_observed", "time_s": None, "frame_index": None}
        row = rows[index]
        return {"status": "observed", "time_s": row.get("time_s"), "frame_index": row.get("frame_index")}

    return {
        "sourceward_progress_3d_m": {
            "distance_reference": "configured odor field center",
            "initial_distance_m": first_distance,
            "initial_frame_index": first_row.get("frame_index"),
            "initial_state_time_s": first_row.get("time_s"),
            "minimum_distance_m": min_distance,
            "minimum_frame_index": nearest_row.get("frame_index"),
            "minimum_state_time_s": nearest_row.get("time_s"),
            "final_distance_m": last_distance,
            "final_frame_index": last_row.get("frame_index"),
            "final_state_time_s": last_row.get("time_s"),
            "initial_minus_minimum_m": first_distance - min_distance,
            "initial_minus_final_m": first_distance - last_distance,
        },
        "body_position_m": {
            "initial": list(first_position), "minimum_distance_position": list(nearest_position),
            "final": list(last_position),
        },
        "ascent_m": {
            "initial_z_world_m": z_values[0],
            "maximum_z_world_m": max(z_values),
            "final_z_world_m": z_values[-1],
            "maximum_above_initial_m": max(z_values) - z_values[0],
            "final_above_initial_m": z_values[-1] - z_values[0],
        },
        "landing_and_contact": {
            "grounded_at_first_recorded_state": initial_grounded,
            "first_airborne_state": state_event(first_airborne_index),
            "first_landing_after_airborne": state_event(landing_index),
            "first_target_contact": state_event(target_index),
            "first_declared_food_surface_target_contact": state_event(target_surface_index),
            "grounded_frame_count": sum(ground_flags),
            "target_contact_frame_count": sum(target_flags),
        },
        "frame_count": len(rows),
    }


def _summarize_run(role: str, source: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = Path(source).expanduser().resolve()
    if not run_dir.is_dir():
        raise ForagingSummaryError(f"{role}: run directory does not exist: {run_dir}")
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ForagingSummaryError(f"{role}: manifest root must be an object")
    identity = _validate_strict_manifest(role, run_dir, manifest)
    trajectory_path = _safe_output(run_dir, manifest, "trajectory", "trajectory.json")
    telemetry_path = _safe_output(run_dir, manifest, "telemetry", "telemetry.jsonl")
    trajectory = _trajectory_rows(_read_json(trajectory_path), trajectory_path)
    telemetry = _read_telemetry(telemetry_path)
    frames = _check_streams(trajectory, telemetry, run_dir)
    requested = _mapping(manifest.get("requested_execution"))
    requested_frame_count = requested.get("frame_count")
    if requested_frame_count is not None and requested_frame_count != len(frames):
        raise ForagingSummaryError(f"{role}: requested frame_count differs from recorded frame count")
    requested_ablations = _ablation_list(
        requested.get("food_cue_ablations", []), "requested food_cue_ablations", str(run_dir)
    )
    expected_ablations = EXPECTED_ABLATIONS[role]
    if role == "contact_positive":
        if requested_ablations not in ((), ("odor",)):
            raise ForagingSummaryError("contact_positive: only an optional matched odor ablation is permitted")
        expected_ablations = requested_ablations
    elif role == "contact_taste_ablated":
        if requested_ablations not in (("taste",), ("odor", "taste")):
            raise ForagingSummaryError("contact_taste_ablated: taste must be ablated, with optional matched odor ablation")
        expected_ablations = requested_ablations
    else:
        expected_ablations = EXPECTED_ABLATIONS[role]
        if requested_ablations != expected_ablations:
            raise ForagingSummaryError(f"{role}: requested food ablations are {requested_ablations!r}, expected {expected_ablations!r}")
    effective = _mapping(manifest.get("effective_execution"))
    if _ablation_list(effective.get("food_cue_ablations", []), "effective food_cue_ablations", str(run_dir)) != expected_ablations:
        raise ForagingSummaryError(f"{role}: effective food ablations disagree with its role")
    components = _mapping(manifest.get("components"))
    food_component_raw = components.get("food_sensing")
    food_component = _mapping(food_component_raw)
    has_food_config = role != "no_food"
    if has_food_config:
        if not food_component:
            raise ForagingSummaryError(f"{role}: food-sensing component provenance is missing")
        config, food_hash, config_path = _resolve_food_config(run_dir, manifest, food_component)
        if food_component.get("source") != "handbuilt" or food_component.get("input_to_malecns") != "poisson_hz":
            raise ForagingSummaryError(f"{role}: food-sensing source/input provenance is unexpected")
        _audit_food_rows(role, telemetry, frames, config, food_component, expected_ablations)
        if _ablation_list(food_component.get("ablated_senses"), "components.food_sensing.ablated_senses", str(run_dir)) != expected_ablations:
            raise ForagingSummaryError(f"{role}: component food-sense ablation provenance disagrees with its role")
        _, _, surface_id, odor_center = _cue_config_fields(config, str(run_dir))
    else:
        config, food_hash, config_path, surface_id, odor_center = None, None, None, None, None
        if food_component_raw is not None or "food_config" in requested or "food_config_sha256" in requested:
            raise ForagingSummaryError("no_food: food-sensing manifest/config is present")
        if any("food_cue" in row for row in telemetry):
            raise ForagingSummaryError("no_food: telemetry contains food_cue rows")
    for row in telemetry:
        if row.get("execution_policy") != STRICT_POLICY:
            raise ForagingSummaryError(f"{role}: telemetry row does not identify strict execution")
    trajectory_summary = None
    events: dict[str, Any] = {}
    if has_food_config:
        assert odor_center is not None and surface_id is not None
        trajectory_summary = _trajectory_summary(telemetry, odor_center, run_dir, surface_id)
        odor_names, taste_names, _, _ = _cue_config_fields(config, str(run_dir))
        events = {
            "odor_input_raw": _food_rate_summary(telemetry, "raw_odor_poisson_hz", selectors=odor_names),
            "odor_input_to_CNS": _food_rate_summary(telemetry, "cns_poisson_hz", selectors=odor_names),
            "sugar_input_raw": _food_rate_summary(telemetry, "raw_taste_poisson_hz", selectors=taste_names),
            "sugar_input_to_CNS": _food_rate_summary(telemetry, "cns_poisson_hz", selectors=taste_names),
            "food_physical_contact": _first_event(telemetry, lambda row: _mapping(row.get("food_cue")).get("physical_contact") is True, sensory=True),
            "edible_region_contact": _first_event(telemetry, lambda row: _mapping(row.get("food_cue")).get("edible_region_contact") is True, sensory=True),
            "food_cue_contact": _first_event(telemetry, lambda row: _mapping(row.get("food_cue")).get("contact") is True, sensory=True),
            "MN9": _summarize_population_events(telemetry),
            "mouth_motor_output": _first_event(telemetry, lambda row: (_number(_mapping(row.get("nm_out")).get("mouth")) or 0.0) > 0),
            "wing_motor_output": _first_event(telemetry, lambda row: (
                (_number(_mapping(row.get("nm_out")).get("left_wing_power")) or 0.0) > 0
                or (_number(_mapping(row.get("nm_out")).get("right_wing_power")) or 0.0) > 0
                or (_number(row.get("wing_power")) or 0.0) > 0
            )),
            "takeoff": trajectory_summary["landing_and_contact"]["first_airborne_state"],
        }
    frame_identity = {
        "frame_count": len(frames),
        "first_frame_index": frames[0]["frame_index"],
        "last_frame_index": frames[-1]["frame_index"],
        "start_frame_time_s": frames[0]["frame_time_s"],
        "end_frame_time_s": frames[-1]["frame_time_s"],
        "end_state_time_s": frames[-1]["state_time_s"],
        "dt_s": frames[0]["dt_s"],
        "trajectory_telemetry_identity_checked": True,
    }
    summary = {
        "role": role,
        "run_directory": str(run_dir),
        "identity": identity,
        "frame_identity": frame_identity,
        "food_config": None if not has_food_config else {
            "path": str(config_path), "sha256": food_hash,
            "contact_surface_id": surface_id, "selector_counts": dict(_mapping(food_component.get("selector_counts"))),
            "ablated_senses": list(expected_ablations),
        },
        "trajectory": trajectory_summary,
        "events": events,
        "artifacts": {
            "manifest_sha256": _sha256_file(manifest_path),
            "trajectory_sha256": _sha256_file(trajectory_path),
            "telemetry_sha256": _sha256_file(telemetry_path),
        },
    }
    internals = {
        "manifest": manifest,
        "experiment": _mapping(components.get("experiment")),
        "world_scene": _mapping(_mapping(components.get("world")).get("scene_manifests")),
        "food_hash": food_hash,
        "frames": frames,
        "rows": telemetry,
        "odor_center": odor_center,
        "contact_surface_id": surface_id,
    }
    return summary, internals


def _without_keys(value: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def _request_identity(internal: Mapping[str, Any], *, no_food: bool = False) -> dict[str, Any]:
    manifest = _mapping(internal.get("manifest"))
    requested = dict(_mapping(manifest.get("requested_execution")))
    requested.pop("food_cue_ablations", None)
    if no_food:
        requested.pop("food_config", None)
        requested.pop("food_config_sha256", None)
    return requested


def _check_request_pair(runs: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
                        left_role: str, right_role: str, *, no_food: bool = False) -> None:
    left, right = runs[left_role][1], runs[right_role][1]
    if _request_identity(left, no_food=no_food) != _request_identity(right, no_food=no_food):
        raise ForagingSummaryError(f"{left_role} and {right_role} requested execution inputs differ beyond declared food intervention")
    left_config = _mapping(left.get("manifest")).get("config")
    right_config = _mapping(right.get("manifest")).get("config")
    if left_config != right_config:
        raise ForagingSummaryError(f"{left_role} and {right_role} main experiment config paths differ")


def _check_matched_inputs(runs: Mapping[str, tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    required = ("intact", "odor_ablated")
    for role in required:
        if role not in runs:
            raise ForagingSummaryError(f"required run role is missing: {role}")
    summaries = {role: pair[0] for role, pair in runs.items()}
    internals = {role: pair[1] for role, pair in runs.items()}
    base = summaries["intact"]["identity"]
    for role, summary in summaries.items():
        identity = summary["identity"]
        for key in ("seed", "execution_policy", "requested_backends", "effective_backends",
                    "requested_renderer_backend", "effective_renderer_backend"):
            left, right = identity[key], base[key]
            if key == "effective_backends":
                left = {name: value for name, value in left.items() if name != "food_sensing"}
                right = {name: value for name, value in right.items() if name != "food_sensing"}
            if left != right:
                raise ForagingSummaryError(f"{role}: {key} differs from intact run")
        if identity["resource_caps"] != base["resource_caps"]:
            raise ForagingSummaryError(f"{role}: recorded resource caps differ from intact run")
        if internals[role]["world_scene"] != internals["intact"]["world_scene"]:
            raise ForagingSummaryError(f"{role}: physical scene/world identity differs from intact run")
    exact_pair_experiment = _without_keys(internals["intact"]["experiment"], PAIR_EXPERIMENT_IGNORES)
    if _without_keys(internals["odor_ablated"]["experiment"], PAIR_EXPERIMENT_IGNORES) != exact_pair_experiment:
        raise ForagingSummaryError("intact and odor-ablated experiment inputs differ beyond descriptive labels")
    _check_request_pair(runs, "intact", "odor_ablated")
    if summaries["intact"]["frame_identity"] != summaries["odor_ablated"]["frame_identity"]:
        raise ForagingSummaryError("intact and odor-ablated frame identities differ")
    matched_pairs = ["intact_vs_odor_ablated"]
    if "contact_positive" in internals:
        contact_experiment = _without_keys(internals["contact_positive"]["experiment"], CONTACT_EXPERIMENT_IGNORES)
        baseline_without_start = _without_keys(internals["intact"]["experiment"], CONTACT_EXPERIMENT_IGNORES)
        if contact_experiment != baseline_without_start:
            raise ForagingSummaryError("contact-positive experiment differs beyond its declared start-pose control")
    if "contact_taste_ablated" in internals:
        if "contact_positive" not in internals:
            raise ForagingSummaryError("contact-taste-ablated requires its matched contact-positive run")
        taste_experiment = _without_keys(internals["contact_taste_ablated"]["experiment"], PAIR_EXPERIMENT_IGNORES)
        positive_experiment = _without_keys(internals["contact_positive"]["experiment"], PAIR_EXPERIMENT_IGNORES)
        if taste_experiment != positive_experiment:
            raise ForagingSummaryError("contact-positive and contact-taste-ablated experiment inputs differ")
        _check_request_pair(runs, "contact_positive", "contact_taste_ablated")
        positive_ablations = tuple(summaries["contact_positive"]["food_config"].get("ablated_senses", []))
        taste_ablations = tuple(summaries["contact_taste_ablated"]["food_config"].get("ablated_senses", []))
        if set(taste_ablations) != set(positive_ablations) | {"taste"}:
            raise ForagingSummaryError("contact-taste-ablated must add exactly taste to contact-positive ablations")
        if summaries["contact_positive"]["frame_identity"] != summaries["contact_taste_ablated"]["frame_identity"]:
            raise ForagingSummaryError("contact-positive and contact-taste-ablated frame identities differ")
        matched_pairs.append("contact_positive_vs_contact_taste_ablated")
    if "no_food" in internals:
        no_food_experiment = _without_keys(internals["no_food"]["experiment"], PAIR_EXPERIMENT_IGNORES)
        if no_food_experiment != exact_pair_experiment:
            raise ForagingSummaryError("no-food experiment inputs differ beyond descriptive labels")
        if summaries["no_food"]["frame_identity"] != summaries["intact"]["frame_identity"]:
            raise ForagingSummaryError("no-food and intact frame identities differ")
        _check_request_pair(runs, "intact", "no_food", no_food=True)
    if "no_food" in internals:
        no_food_experiment = _without_keys(internals["no_food"]["experiment"], PAIR_EXPERIMENT_IGNORES)
        if no_food_experiment != exact_pair_experiment:
            raise ForagingSummaryError("no-food experiment inputs differ beyond descriptive labels")
    food_roles = [role for role in ("intact", "odor_ablated", "contact_positive", "contact_taste_ablated")
                  if role in internals]
    food_hashes = {role: internals[role]["food_hash"] for role in food_roles}
    if len(set(food_hashes.values())) != 1:
        raise ForagingSummaryError("food-config SHA-256 differs across the three food-present roles")
    return {
        "seed_backend_policy_caps_matched": True,
        "world_scene_identity_matched": True,
        "matched_pairs_with_identical_frame_identity": matched_pairs + (["intact_vs_no_food"] if "no_food" in internals else []),
        "intact_vs_odor_ablated_inputs_matched_except_declared_odor_ablation": True,
        "contact_positive_start_pose_control_checked": "contact_positive" in runs,
        "food_config_hashes_match_across_food_roles": True,
        "food_config_sha256": next(iter(food_hashes.values())),
        "no_food_run_included": "no_food" in runs,
    }


def analyze_runs(sources: Mapping[str, str | Path]) -> dict[str, Any]:
    unknown = set(sources) - set(ROLES)
    if unknown:
        raise ForagingSummaryError(f"unknown run role(s): {', '.join(sorted(unknown))}")
    if any(role not in sources for role in ("intact", "odor_ablated")):
        raise ForagingSummaryError("intact and odor-ablated runs are required")
    runs = {role: _summarize_run(role, path) for role, path in sources.items()}
    if "no_food" in runs:
        food_run = runs["intact"][1]
        no_food_summary = runs["no_food"][0]
        no_food_internal = runs["no_food"][1]
        center = food_run["odor_center"]
        surface_id = food_run["contact_surface_id"]
        if center is None or surface_id is None:
            raise ForagingSummaryError("cannot summarize no-food distance without the matched food reference geometry")
        no_food_summary["trajectory"] = _trajectory_summary(
            no_food_internal["rows"], center, Path(no_food_summary["run_directory"]), surface_id
        )
        no_food_summary["events"] = {
            "MN9": _summarize_population_events(no_food_internal["rows"]),
            "mouth_motor_output": _first_event(no_food_internal["rows"], lambda row: (_number(_mapping(row.get("nm_out")).get("mouth")) or 0.0) > 0),
            "wing_motor_output": _first_event(no_food_internal["rows"], lambda row: (
                (_number(_mapping(row.get("nm_out")).get("left_wing_power")) or 0.0) > 0
                or (_number(_mapping(row.get("nm_out")).get("right_wing_power")) or 0.0) > 0
                or (_number(row.get("wing_power")) or 0.0) > 0
            )),
            "takeoff": no_food_summary["trajectory"]["landing_and_contact"]["first_airborne_state"],
        }
    for role in ("intact", "odor_ablated", "contact_positive", "contact_taste_ablated"):
        if role not in runs:
            continue
        internal = runs[role][1]
        food_component = _mapping(_mapping(internal["manifest"].get("components")).get("food_sensing"))
        config = _mapping(food_component.get("config"))
        odor_names, taste_names, _, _ = _cue_config_fields(config, role)
        runs[role][0]["events"]["food_receptor_population_outputs"] = {
            "odor": _summarize_food_neuron_outputs(internal["rows"], sorted(odor_names)),
            "sugar": _summarize_food_neuron_outputs(internal["rows"], sorted(taste_names)),
        }
    compatibility = _check_matched_inputs(runs)
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": NOT_VALIDATED,
        "evidence_class": "digital_simulation",
        "biological_claim": False,
        "causal_claim": False,
        "actual_ingestion": "not_modeled",
        "scope": "Descriptive summary of strict digital-fly food-cue, neural/motor, contact, and trajectory artifacts.",
        "compatibility": compatibility,
        "runs": {role: pair[0] for role, pair in runs.items()},
        "interpretation_limits": [
            "The odor field and receptor rates are hand-built simulator inputs. Sourceward distance changes do not establish odor-guided search or causation.",
            "Sugar rates are a contact-gated neural input proxy defined by declared support-surface and mouth-sphere geometry; they do not establish feeding or ingestion.",
            "MN9 population summaries, mouth and wing channels, takeoff, landing, and contact are recorded model readouts, not validated biological events.",
            "A missing event means it was not observed in the recorded frames; it is not evidence that the event cannot occur.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Elevated-food foraging experiment summary",
        "",
        f"Evidence: `{report.get('evidence_class')}`; decision: `{report.get('decision')}`.",
        "",
        "This report describes digital-simulation artifacts only. It makes no biological or causal claim. Actual ingestion is **not modeled**.",
        "",
        "## Run matrix",
        "",
        "| Role | Seed | Frames | Food config SHA-256 |",
        "|---|---:|---:|---|",
    ]
    runs = _mapping(report.get("runs"))
    for role, raw in runs.items():
        run = _mapping(raw)
        identity = _mapping(run.get("identity"))
        frame = _mapping(run.get("frame_identity"))
        food = _mapping(run.get("food_config"))
        lines.append(f"| {role} | {identity.get('seed')} | {frame.get('frame_count')} | {food.get('sha256', 'not configured')} |")
    lines.extend([
        "",
        "## Compatibility checks",
        "",
    ])
    for key, value in _mapping(report.get("compatibility")).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Recorded outcomes",
        "",
        "Distances are 3D Euclidean distances from the recorded body center to the configured odor-field center. Ascent is relative to the first recorded body-state sample.",
        "",
        "| Role | Initial distance (m) | Minimum (m) | Final (m) | Max ascent (m) | First raw odor (s) | First CNS odor (s) | First CNS sugar (s) | Takeoff (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for role, raw in runs.items():
        run = _mapping(raw)
        trajectory = _mapping(run.get("trajectory"))
        progress = _mapping(_mapping(trajectory.get("sourceward_progress_3d_m")))
        ascent = _mapping(trajectory.get("ascent_m"))
        events = _mapping(run.get("events"))
        odor = _mapping(events.get("odor_input_raw"))
        cns_odor = _mapping(events.get("odor_input_to_CNS"))
        cns_sugar = _mapping(events.get("sugar_input_to_CNS"))
        takeoff = _mapping(events.get("takeoff"))
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            role, progress.get("initial_distance_m", "—"), progress.get("minimum_distance_m", "—"),
            progress.get("final_distance_m", "—"), ascent.get("maximum_above_initial_m", "—"),
            _display_time(_mapping(odor.get("first_positive"))),
            _display_time(_mapping(cns_odor.get("first_positive"))),
            _display_time(_mapping(cns_sugar.get("first_positive"))), _display_time(takeoff),
        ))
    lines.extend([
        "",
        "The first odor time marks a recorded receptor-rate input. The first sugar time marks a contact-gated rate sent to the CNS input seam. Temporal order alongside movement is descriptive and does not establish odor-induced search. MN9 and mouth values are recorded model readouts. Neither a positive mouth channel nor a contact record demonstrates ingestion.",
        "",
        "### Per-run contact and motor events",
        "",
    ])
    for role, raw in runs.items():
        run = _mapping(raw)
        events = _mapping(run.get("events"))
        trajectory = _mapping(run.get("trajectory"))
        contact = _mapping(trajectory.get("landing_and_contact"))
        mn9 = _mapping(events.get("MN9"))
        sugar_cns = _mapping(events.get("sugar_input_to_CNS"))
        mn9_totals = _mapping(mn9.get("MN9_total_spikes_by_population"))
        lines.extend([
            f"#### {role}",
            "",
            f"- Declared food-sense ablations: `{_mapping(run.get('food_config')).get('ablated_senses', [])}`.",
            f"- First food-support contact at cue sampling: {_display_time(_mapping(events.get('food_physical_contact')))} s; first body target-surface contact: {_display_time(_mapping(contact.get('first_declared_food_surface_target_contact')))} s.",
            f"- First edible-region contact: {_display_time(_mapping(events.get('edible_region_contact')))} s; first gated cue contact: {_display_time(_mapping(events.get('food_cue_contact')))} s.",
            f"- First MN9 population spike: {_display_time(_mapping(mn9.get('first_MN9_spike')))} s; first mouth output: {_display_time(_mapping(events.get('mouth_motor_output')))} s.",
            f"- First wing output: {_display_time(_mapping(events.get('wing_motor_output')))} s; first landing after an airborne frame: {_display_time(_mapping(contact.get('first_landing_after_airborne')))} s.",
            f"- CNS sugar input was positive in {sugar_cns.get('positive_frame_count', 0)}/{sugar_cns.get('frame_count', _mapping(run.get('frame_identity')).get('frame_count', 0))} frames; MN9 recorded spike totals by population: `{dict(mn9_totals)}`.",
            "- Configured sugar-selector neural spike totals: " + ", ".join(
                f"{name}={item.get('total_spikes')}" for name, item in _mapping(
                    _mapping(events.get("food_receptor_population_outputs")).get("sugar")
                ).items()
            ) + ".",
            "",
        ])
    lines.extend(["## Limitations", ""])
    for item in report.get("interpretation_limits", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _display_time(event: Mapping[str, Any]) -> str:
    value = event.get("time_s")
    return "not observed" if _number(value) is None else f"{_number(value):.6g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intact", required=True, help="matched intact food-cue run directory")
    parser.add_argument("--odor-ablated", required=True, help="matched run with odor input ablated")
    parser.add_argument("--contact-positive", help="optional food-present support-contact positive-control run")
    parser.add_argument("--contact-taste-ablated", help="optional matched contact-positive run with taste input ablated")
    parser.add_argument("--no-food", help="optional matched run with the food cue disabled")
    parser.add_argument("--output-json", required=True, type=Path, help="JSON report output path")
    parser.add_argument("--output-markdown", required=True, type=Path, help="Markdown report output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = {
        "intact": args.intact,
        "odor_ablated": args.odor_ablated,
    }
    if args.contact_positive:
        sources["contact_positive"] = args.contact_positive
    if args.no_food:
        sources["no_food"] = args.no_food
    if args.contact_taste_ablated:
        sources["contact_taste_ablated"] = args.contact_taste_ablated
    try:
        report = analyze_runs(sources)
        json_text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        markdown_text = render_markdown(report)
        if args.output_json.resolve() == args.output_markdown.resolve():
            raise ForagingSummaryError("JSON and Markdown output paths must be different")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text, encoding="utf-8")
        args.output_markdown.write_text(markdown_text, encoding="utf-8")
    except (ForagingSummaryError, OSError) as exc:
        print(f"summarize_foraging: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"decision": report["decision"], "json": str(args.output_json),
                      "markdown": str(args.output_markdown), "run_roles": list(report["runs"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
