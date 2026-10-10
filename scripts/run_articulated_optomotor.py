#!/usr/bin/env python3
"""Run the strict world-anchored stripe assay with articulated six-leg physics.

The renderer, exact FlyVis columns, FlyVis-to-MaleCNS mapping and MaleCNS
construction are delegated to the existing strict runner. This module owns
the neural 0.5 ms loop and sends every individual dense neuron-spike vector to
the established Hill-muscle/MuJoCo pipeline. It never generates a gait,
chooses a motor direction, or applies a body traction command.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_digital_fly import (  # noqa: E402
    DigitalFlyRunner,
    RunOptions,
    _array_sha256,
    _flyvis_drive_contract,
    _make_contract_body,
    _rpy_from_quat,
    _substep_sensory,
    _timestamp,
)
from twin_contracts import SensoryFrame, SourceLabel, VisualObservation  # noqa: E402
from virtual_world import Pose  # noqa: E402
from articulated_leg_body import ArticulatedLegBody  # noqa: E402
from leg_muscle_decoder import LegMuscleDecoder  # noqa: E402
from leg_muscle_pipeline import LegMusclePipeline  # noqa: E402
from leg_muscle_parameters import (  # noqa: E402
    DEFAULT_CONFIG_PATH as DEFAULT_MUSCLE_CONFIG_PATH,
    LegMuscleParameters,
)
from leg_sensory_feedback import LEGS as LEG_NAMES  # noqa: E402
from leg_sensory_feedback import LegSensoryFeedback  # noqa: E402


SCHEMA_VERSION = "articulated-optomotor-run-1"
FRAME_DT_S = 0.005
CNS_DT_S = 0.0005
CNS_TICKS_PER_FRAME = 10
MAX_DURATION_S = 7.5
MAX_HOST_MEMORY_BYTES = 5 * 1024**3
MAX_VRAM_GIB = 15.5
DEFAULT_DURATION_S = 7.5
DEFAULT_OUT = Path(
    "/data/artifacts/by-project/malecns-gpu/experiments/"
    "articulated-optomotor-20261003-v1"
)
CONFIG_PATH = ROOT / "config" / "optomotor_ground_crawl_experiment.json"
STIMULUS_PATH = ROOT / "config" / "optomotor_pilot_stimuli.json"
SCENE_PATH = ROOT / "config" / "optomotor_panel_scene.json"
CONDITION_STIMULI = {
    "left": "stripe_left",
    "right": "stripe_right",
    "static": "stripe_static",
}
LEGACY_LEG_ALIASES = (
    "leg", "leg_mechanosensory", "leg_tactile", "leg_proprioceptive",
)
LEGACY_NONLEG_GROUPS = ("jo", "haltere", "wing")
LEGACY_CONTACT_KEYS = {
    "LF": "front_left", "RF": "front_right",
    "LM": "middle_left", "RM": "middle_right",
    "LH": "hind_left", "RH": "hind_right",
}


class ArticulatedOptomotorError(RuntimeError):
    """The requested strict articulated trial could not be completed safely."""


def validate_duration(duration_s: float) -> int:
    """Return frame count for a bounded, 5 ms-aligned trial duration."""

    try:
        duration = float(duration_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("duration must be a finite number") from exc
    if not math.isfinite(duration) or duration < FRAME_DT_S or duration > MAX_DURATION_S:
        raise ValueError(f"duration must be in [{FRAME_DT_S:g}, {MAX_DURATION_S:g}] seconds")
    frames = int(round(duration / FRAME_DT_S))
    if abs(duration - frames * FRAME_DT_S) > 1e-10:
        raise ValueError("duration must be an exact multiple of the 5 ms FlyVis frame period")
    return frames


def neural_tick_timestamp(tick: int) -> Any:
    """Build the exact 2 kHz neural timestamp for a nonnegative integer tick."""

    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError("neural tick must be a nonnegative integer")
    value = int(tick)
    return _timestamp(value * CNS_DT_S, clock="neural", rate_hz=1.0 / CNS_DT_S)


def canonical_spikes(value: Any, population_size: int) -> np.ndarray:
    """Validate one actual binary per-neuron observation in dense CSR order."""

    if isinstance(population_size, bool) or int(population_size) <= 0:
        raise ValueError("population_size must be a positive integer")
    try:
        dense = np.asarray(value)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("neural spikes must be a dense numeric vector") from exc
    if dense.ndim != 1 or dense.size != int(population_size):
        raise ValueError(
            f"neural spikes must have shape ({int(population_size)},), got {dense.shape}"
        )
    if dense.dtype.kind not in "biuf" or not np.isfinite(dense).all():
        raise ValueError("neural spikes must be finite numeric values")
    if np.any((dense != 0) & (dense != 1)):
        raise ValueError("each per-neuron neural observation must be binary")
    return np.ascontiguousarray(dense, dtype=np.uint8)


def mask_legacy_leg_drives(
    population_drive: Mapping[str, Any], mode: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Mask every legacy leg alias for off/on while preserving the source record."""

    if mode not in {"legacy", "off", "on"}:
        raise ValueError("sensory feedback mode must be legacy, off, or on")
    copied = dict(population_drive)
    masked: dict[str, float] = {}
    if mode in {"off", "on"}:
        for name in LEGACY_LEG_ALIASES:
            if name in copied:
                masked[name] = float(copied[name])
                copied[name] = 0.0
    return copied, masked


def legacy_contact_mapping(feet: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """Translate articulated leg labels into the existing feedback vocabulary."""

    missing = set(LEGACY_CONTACT_KEYS) - set(feet)
    if missing:
        raise ValueError(f"missing articulated foot records: {sorted(missing)}")
    return {
        LEGACY_CONTACT_KEYS[leg]: bool(feet[leg]["contact"])
        for leg in LEG_NAMES
    }


def mode_contract(mode: str) -> dict[str, Any]:
    """Describe the exact legacy and articulated sensory sources for a mode."""

    if mode not in {"legacy", "off", "on"}:
        raise ValueError("sensory feedback mode must be legacy, off, or on")
    return {
        "mode": mode,
        "legacy_groups_applied": (
            ["jo", "haltere", "wing", "leg"] if mode == "legacy"
            else list(LEGACY_NONLEG_GROUPS)
        ),
        "legacy_leg_aliases_masked": list(LEGACY_LEG_ALIASES) if mode != "legacy" else [],
        "articulated_leg_current_applied": mode == "on",
        "articulated_leg_current_source": "leg_sensory_feedback.LegSensoryFeedback.encode",
        "body_state_source": "current ArticulatedLegBody telemetry: legacy aggregates at 5 ms boundaries; articulated afferents at 0.5 ms boundaries",
        "legacy_aggregate_source_cadence_s": FRAME_DT_S,
        "articulated_leg_source_cadence_s": CNS_DT_S,
        "legacy_wing_joint_state": "zero-power/no-flapping; articulated body has no wings",
    }


def per_leg_indices(decoder: LegMuscleDecoder) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resolve all annotated leg pools and the exact mapped-muscle subset."""

    all_pools = {
        leg: np.asarray(decoder._leg_indices[leg], dtype=np.int64)
        for leg in LEG_NAMES
    }
    mapped_sets: dict[str, set[int]] = {leg: set() for leg in LEG_NAMES}
    for channel in decoder.channels:
        leg = str(channel["leg"])
        mapped_sets[leg].update(
            int(index) for index in decoder._channel_indices[str(channel["name"])].tolist()
        )
    mapped = {
        leg: np.asarray(sorted(mapped_sets[leg]), dtype=np.int64)
        for leg in LEG_NAMES
    }
    for leg in LEG_NAMES:
        if not set(mapped[leg].tolist()).issubset(set(all_pools[leg].tolist())):
            raise ArticulatedOptomotorError(f"mapped muscle pool escaped annotated leg pool {leg}")
    return all_pools, mapped


def build_motor_unit_mapping(
    engine_index: Any,
    decoder: LegMuscleDecoder,
    all_pools: Mapping[str, np.ndarray],
    mapped_pools: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the exact dense-index/body-ID map and sparse-event lookup.

    This uses the selected engine's own index and validates that the decoder
    adapter points back to it. Speed class is kept explicitly unknown because
    the source population has no fast/slow labels.
    """

    adapter = decoder.population_index
    adapter_source = getattr(adapter, "_source", adapter)
    if adapter_source is not engine_index:
        raise ArticulatedOptomotorError(
            "motor-unit decoder is not using this MaleCNS engine's population index"
        )
    engine_ids = np.asarray(getattr(engine_index, "ids", None))
    engine_n = int(getattr(engine_index, "n", -1))
    if (engine_ids.ndim != 1 or engine_n <= 0 or engine_ids.size != engine_n
            or engine_ids.dtype.kind not in "iu"):
        raise ArticulatedOptomotorError("engine population index has invalid dense body IDs")
    if np.unique(engine_ids).size != engine_n:
        raise ArticulatedOptomotorError("engine population index body IDs are not unique")
    adapter_ids = np.asarray(getattr(adapter, "ids", None))
    if adapter_ids.shape != engine_ids.shape or not np.array_equal(adapter_ids, engine_ids):
        raise ArticulatedOptomotorError(
            "motor-unit decoder adapter body IDs differ from the engine dense order"
        )
    body_resolver = getattr(engine_index, "resolve", None)
    if not callable(body_resolver):
        raise ArticulatedOptomotorError("engine population index cannot resolve body IDs")

    normalized_pools: dict[str, np.ndarray] = {}
    dense_leg_by_index: dict[int, str] = {}
    for leg in LEG_NAMES:
        if leg not in all_pools or leg not in mapped_pools:
            raise ArticulatedOptomotorError(f"motor-unit mapping is missing the {leg} leg pool")
        pool = np.asarray(all_pools[leg])
        mapped = np.asarray(mapped_pools[leg])
        if pool.ndim != 1 or pool.dtype.kind not in "iu":
            raise ArticulatedOptomotorError(f"annotated dense indices are invalid for {leg}")
        if mapped.ndim != 1 or mapped.dtype.kind not in "iu":
            raise ArticulatedOptomotorError(f"mapped dense indices are invalid for {leg}")
        pool = np.unique(pool.astype(np.int64, copy=False))
        mapped = np.unique(mapped.astype(np.int64, copy=False))
        if (pool.size and (int(pool[0]) < 0 or int(pool[-1]) >= engine_n)):
            raise ArticulatedOptomotorError(f"annotated dense index is out of range for {leg}")
        if (mapped.size and (int(mapped[0]) < 0 or int(mapped[-1]) >= engine_n)):
            raise ArticulatedOptomotorError(f"mapped dense index is out of range for {leg}")
        if not np.isin(mapped, pool, assume_unique=True).all():
            raise ArticulatedOptomotorError(f"mapped motor pool escaped annotated leg pool {leg}")
        normalized_pools[leg] = pool
        for index in pool.tolist():
            if index in dense_leg_by_index:
                raise ArticulatedOptomotorError(
                    f"annotated motor neuron dense index {index} belongs to multiple legs"
                )
            dense_leg_by_index[int(index)] = leg

    # Resolve only the recorded annotated leg motor units. Resolving every
    # population ID would repeatedly scan the full 176k-neuron ID array.
    for dense_index in dense_leg_by_index:
        body_id = int(engine_ids[dense_index])
        resolved = np.asarray(body_resolver(f"body:{body_id}", strict=True), dtype=np.int64)
        if resolved.shape != (1,) or int(resolved[0]) != dense_index:
            raise ArticulatedOptomotorError(
                f"engine body-ID resolution disagrees with dense index {dense_index}"
            )

    memberships: dict[int, list[str]] = {index: [] for index in dense_leg_by_index}
    channel_rows: list[dict[str, Any]] = []
    observed_channel_union: dict[str, set[int]] = {leg: set() for leg in LEG_NAMES}
    for channel in decoder.channels:
        name = str(channel["name"])
        leg = str(channel["leg"])
        if leg not in LEG_NAMES or name not in decoder._channel_indices:
            raise ArticulatedOptomotorError(f"decoder channel metadata is invalid: {name!r}")
        indices = np.asarray(decoder._channel_indices[name])
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ArticulatedOptomotorError(f"dense indices are invalid for channel {name}")
        indices = np.unique(indices.astype(np.int64, copy=False))
        if indices.size and (int(indices[0]) < 0 or int(indices[-1]) >= engine_n):
            raise ArticulatedOptomotorError(f"channel {name} has an out-of-range dense index")
        if not np.isin(indices, normalized_pools[leg], assume_unique=True).all():
            raise ArticulatedOptomotorError(f"channel {name} escaped annotated leg pool {leg}")
        for index in indices.tolist():
            memberships[int(index)].append(name)
            observed_channel_union[leg].add(int(index))
        channel_rows.append({
            "name": name,
            "leg": leg,
            "group": str(channel["group"]),
            "type_name": str(channel["type_name"]),
            "selector": str(decoder._channel_specs[name]),
            "motor_unit_count": int(indices.size),
            "dense_indices": [int(value) for value in indices.tolist()],
            "body_ids": [int(value) for value in engine_ids[indices].tolist()],
            "motor_unit_speed_classification": "unknown",
        })

    for leg in LEG_NAMES:
        expected_mapped = set(normalized_pools[leg].tolist()) & observed_channel_union[leg]
        if expected_mapped != set(normalized_pools[leg].tolist()) & set(
                np.asarray(mapped_pools[leg], dtype=np.int64).tolist()):
            raise ArticulatedOptomotorError(
                f"mapped pool disagrees with exact channel assignments for {leg}"
            )

    units: list[dict[str, Any]] = []
    for dense_index in sorted(dense_leg_by_index):
        channels = sorted(memberships[dense_index])
        units.append({
            "body_id": int(engine_ids[dense_index]),
            "dense_index": int(dense_index),
            "leg": dense_leg_by_index[dense_index],
            "muscle_channels": channels,
            "muscle_groups": sorted({
                str(channel["group"]) for channel in decoder.channels
                if str(channel["name"]) in channels
            }),
            "motor_unit_speed_classification": "unknown",
        })
    counts = {
        "annotated_motor_units_by_leg": {
            leg: int(normalized_pools[leg].size) for leg in LEG_NAMES
        },
        "mapped_motor_units_by_leg": {
            leg: int(len(observed_channel_union[leg])) for leg in LEG_NAMES
        },
        "unmapped_motor_units_by_leg": {
            leg: int(normalized_pools[leg].size - len(observed_channel_union[leg]))
            for leg in LEG_NAMES
        },
        "annotated_motor_units_total": len(units),
        "mapped_motor_units_total": sum(len(names) > 0 for names in memberships.values()),
        "unmapped_motor_units_total": sum(not names for names in memberships.values()),
    }
    mapping = {
        "schema_version": "articulated-leg-motor-unit-map-1",
        "index_source": "constructed MaleCNS engine's own dense PopulationIndex",
        "dense_neuron_count": engine_n,
        "body_id_field": "PopulationIndex.ids[dense_index]",
        "motor_unit_speed_classification": "unknown; no fast/slow labels were supplied",
        "channels": channel_rows,
        "counts": counts,
        "units": units,
    }
    context = {
        "dense_indices": np.asarray([unit["dense_index"] for unit in units], dtype=np.int64),
        "body_ids": np.asarray([unit["body_id"] for unit in units], dtype=np.int64),
        "mapped": np.asarray([bool(unit["muscle_channels"]) for unit in units], dtype=bool),
        "legs": np.asarray([unit["leg"] for unit in units], dtype=object),
    }
    return mapping, context


def sparse_motor_unit_event(
    spikes: np.ndarray,
    mapping_context: Mapping[str, np.ndarray],
    *,
    tick: int,
    neural_t_s: float,
    physical_t_s: float,
    annotated_counts: Mapping[str, int],
    mapped_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Encode already-validated full spikes as one sparse, aligned tick row."""

    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError("motor-unit event tick must be a nonnegative integer")
    expected_time = int(tick) * CNS_DT_S
    for label, value in (("neural_t_s", neural_t_s), ("physical_t_s", physical_t_s)):
        if not math.isfinite(float(value)) or abs(float(value) - expected_time) > 1e-9:
            raise ValueError(f"motor-unit event {label} is not aligned to neural tick {tick}")
    dense_indices = np.asarray(mapping_context["dense_indices"], dtype=np.int64)
    body_ids = np.asarray(mapping_context["body_ids"], dtype=np.int64)
    mapped = np.asarray(mapping_context["mapped"], dtype=bool)
    dense = np.asarray(spikes)
    if (dense.ndim != 1 or dense_indices.shape != body_ids.shape
            or body_ids.shape != mapped.shape
            or (dense_indices.size and int(dense_indices.max()) >= dense.size)):
        raise ValueError("motor-unit event mapping does not fit the validated spike vector")
    active_mask = dense[dense_indices] == 1
    active_body_ids = body_ids[active_mask]
    mapped_active_count = int(np.count_nonzero(active_mask & mapped))
    active_by_leg = {leg: int(annotated_counts.get(leg, 0)) for leg in LEG_NAMES}
    mapped_by_leg = {leg: int(mapped_counts.get(leg, 0)) for leg in LEG_NAMES}
    if sum(active_by_leg.values()) != int(active_body_ids.size):
        raise ValueError("sparse motor-unit events do not reconcile with annotated per-leg counts")
    if sum(mapped_by_leg.values()) != mapped_active_count:
        raise ValueError("sparse motor-unit events do not reconcile with mapped per-leg counts")
    if any(mapped_by_leg[leg] > active_by_leg[leg] for leg in LEG_NAMES):
        raise ValueError("mapped per-leg motor counts exceed annotated counts")
    unmapped_by_leg = {
        leg: active_by_leg[leg] - mapped_by_leg[leg] for leg in LEG_NAMES
    }
    return {
        "tick": int(tick),
        "neural_t_s": float(neural_t_s),
        "physical_t_s": float(physical_t_s),
        "physical_interval_end_t_s": float(physical_t_s) + CNS_DT_S,
        "active_motor_unit_count": int(active_body_ids.size),
        "mapped_active_motor_unit_count": mapped_active_count,
        "unmapped_active_motor_unit_count": int(active_body_ids.size) - mapped_active_count,
        "active_by_leg": active_by_leg,
        "mapped_active_by_leg": mapped_by_leg,
        "unmapped_active_by_leg": unmapped_by_leg,
        "active_body_ids": [int(value) for value in active_body_ids.tolist()],
    }


def leg_sensory_indices(encoder: LegSensoryFeedback, index: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return disjoint sensory targets by leg and their full union."""

    provenance = encoder.provenance()
    selector_names = tuple(sorted(provenance["selectors"]))
    by_leg: dict[str, list[int]] = {leg: [] for leg in LEG_NAMES}
    for selector in selector_names:
        resolved = np.asarray(index.resolve(selector), dtype=np.int64)
        nerve_side = selector.rsplit("&nerve:", 1)[-1]
        leg = next((name for name, spec in encoder._leg_selectors.items()
                    if selector in spec.values()), None)
        if leg is None:
            raise ArticulatedOptomotorError(f"could not assign sensory target selector to leg: {selector}")
        by_leg[leg].extend(int(value) for value in resolved.tolist())
    arrays = {leg: np.asarray(sorted(set(values)), dtype=np.int64) for leg, values in by_leg.items()}
    joined = np.concatenate([arrays[leg] for leg in LEG_NAMES])
    if np.unique(joined).size != joined.size:
        raise ArticulatedOptomotorError("articulated per-leg sensory targets overlap")
    return np.unique(joined), arrays


def selector_overlap_count(
    engine_index: Any, legacy_resolver: Any, articulated_indices: np.ndarray,
) -> dict[str, Any]:
    """Measure old aggregate-leg / new afferent target overlap by dense index."""

    old_leg_spec = next((item for item in legacy_resolver.specs if item.name == "leg"), None)
    if old_leg_spec is None:
        raise ArticulatedOptomotorError("legacy sensory map has no aggregate leg selector")
    old_parts = [np.asarray(engine_index.resolve(selector, strict=False), dtype=np.int64)
                 for selector in old_leg_spec.selectors]
    old_indices = np.unique(np.concatenate(old_parts)) if old_parts else np.empty(0, dtype=np.int64)
    overlap = np.intersect1d(old_indices, articulated_indices, assume_unique=True)
    return {
        "legacy_aggregate_leg_target_count": int(old_indices.size),
        "articulated_leg_target_count": int(articulated_indices.size),
        "dense_neuron_overlap_count": int(overlap.size),
        "overlap_body_id_sha256": _integer_sha256(engine_index.ids[overlap]),
        "interpretation": "overlap is recorded; legacy leg drive is zero in off/on modes",
    }


def _integer_sha256(values: Sequence[Any]) -> str:
    payload = json.dumps(sorted(int(value) for value in values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(_json_safe(value), allow_nan=False, separators=(",", ":")) + "\n")


def _cgroup_memory_limit_bytes(
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """Find the tightest finite cgroup-v2 memory.max for this process."""

    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    relative = next((line.partition("::")[2] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return None
    base = cgroup_root.resolve()
    current = (base / relative.lstrip("/")).resolve()
    if not current.is_relative_to(base):
        return None
    finite: list[int] = []
    while current == base or current.is_relative_to(base):
        limit_file = current / "memory.max"
        try:
            raw = limit_file.read_text(encoding="ascii").strip()
        except OSError:
            raw = ""
        if raw and raw != "max":
            try:
                limit = int(raw)
            except ValueError:
                return None
            if limit > 0:
                finite.append(limit)
        if current == base:
            break
        current = current.parent
    return min(finite) if finite else None


def _require_resource_scope() -> dict[str, Any]:
    limit = _cgroup_memory_limit_bytes()
    if limit is None or limit > MAX_HOST_MEMORY_BYTES:
        raise ArticulatedOptomotorError(
            "run inside a cgroup-v2 scope with finite MemoryMax no greater than 5 GiB; "
            f"observed memory.max={limit!r}"
        )
    return {
        "host_memory_limit_bytes": limit,
        "host_memory_limit_gib": limit / 1024**3,
        "required_host_memory_limit_gib": 5.0,
        "max_vram_gib": MAX_VRAM_GIB,
        "scope_proof": "current process cgroup-v2 memory.max",
    }


def _pose_from_telemetry(body_record: Mapping[str, Any]) -> tuple[Pose, tuple[float, float, float]]:
    root = body_record["root"]
    rpy = _rpy_from_quat(root["quaternion_wxyz"])
    pose = Pose(
        position_m=tuple(float(value) for value in root["position_m"]),
        yaw_rad=rpy[2], pitch_rad=rpy[1], roll_rad=rpy[0],
        velocity_mps=tuple(float(value) for value in root["linear_velocity_mps"]),
        angular_velocity_radps=tuple(float(value) for value in root["angular_velocity_radps"]),
    )
    return pose, rpy


class DecoderPopulationIndex:
    """Preserve engine ordering while allowing audited empty channels."""

    def __init__(self, source: Any):
        self._source = source
        self.ids = source.ids
        self.n = source.n
        self.annotations = source.annotations
        self.tables = source.tables

    def resolve(self, spec: str) -> np.ndarray:
        return self._source.resolve(spec, strict=False)


def _make_pipeline(
    engine: Any,
    muscle_config_path: str | Path | None = None,
    muscle_force_scale: float = 1.0,
) -> LegMusclePipeline:
    body = ArticulatedLegBody()
    # The adapter preserves the engine's dense order while allowing the
    # decoder's documented empty exact channels. The selected engine index
    # must remain the one used by both neural observations and muscle mapping.
    decoder = LegMuscleDecoder(population_index=DecoderPopulationIndex(engine.populations))
    return LegMusclePipeline(
        decoder=decoder,
        body=body,
        muscle_config_path=muscle_config_path,
        muscle_force_scale=muscle_force_scale,
    )


def validate_muscle_options(
    muscle_config_path: str | Path | None,
    muscle_force_scale: float,
) -> tuple[Path, float, str]:
    """Strictly validate selected muscle parameters before runtime startup."""

    try:
        loader = LegMuscleParameters(
            config_path=muscle_config_path,
            force_scale=muscle_force_scale,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid muscle profile options: {exc}") from exc
    selected_path = loader.config_path.expanduser().resolve()
    if not selected_path.is_file():
        raise ValueError(f"muscle config is not a regular file: {selected_path}")
    return selected_path, loader.force_scale, _sha256_file(selected_path)


def _parse_force_scale(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("muscle force scale must be a number in (0, 10]") from exc
    if not math.isfinite(result) or result <= 0.0 or result > 10.0:
        raise argparse.ArgumentTypeError("muscle force scale must be finite and in (0, 10]")
    return result


def _new_runner(out: Path, seed: int, duration_s: float, condition: str) -> DigitalFlyRunner:
    if condition not in CONDITION_STIMULI:
        raise ValueError(f"condition must be one of {', '.join(CONDITION_STIMULI)}")
    options = RunOptions(
        config=CONFIG_PATH.resolve(),
        out=out,
        mode="free-behavior",
        free_behavior=True,
        neural="malecns",
        vision="flyvis",
        body="reduced",
        seed=int(seed),
        neural_device="cuda",
        neural_backend="torch",
        neural_gain=None,
        vision_device="cuda:0",
        max_vram_gib=MAX_VRAM_GIB,
        duration_override_s=float(duration_s),
        telemetry_mode="minimal",
        renderer_backend="warp_cuda",
        scene_manifest=SCENE_PATH.resolve(),
        stimulus_config=STIMULUS_PATH.resolve(),
        stimulus_condition=CONDITION_STIMULI[condition],
    )
    return DigitalFlyRunner(options)


def _source_inventory(
    runner: DigitalFlyRunner,
    muscle_config_path: str | Path | None = None,
    *,
    expected_muscle_config_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Path]]:
    relative_paths = (
        "scripts/run_articulated_optomotor.py",
        "scripts/run_digital_fly.py",
        "scripts/virtual_world.py",
        "scripts/warp_world_backend.py",
        "scripts/world_renderer_opt.py",
        "scripts/scene_asset_manifest.py",
        "scripts/gltf_mesh_loader.py",
        "scripts/flight_dynamics.py",
        "scripts/stereo_flyvis.py",
        "scripts/flyvis_runtime_adapter.py",
        "scripts/flyvis_control_readout.py",
        "scripts/malecns_engine.py",
        "scripts/malecns_gate.py",
        "scripts/neuromuscular_adapter.py",
        "scripts/sensory_feedback.py",
        "scripts/leg_sensory_feedback.py",
        "scripts/leg_muscle_decoder.py",
        "scripts/leg_muscle_dynamics.py",
        "scripts/leg_muscle_parameters.py",
        "scripts/leg_muscle_pipeline.py",
        "scripts/articulated_leg_body.py",
        "scripts/population_selectors.py",
        "scripts/twin_contracts.py",
        "config/optomotor_ground_crawl_experiment.json",
        "config/optomotor_pilot_stimuli.json",
        "config/optomotor_panel_scene.json",
        "config/sensory_map.json",
        "config/leg_sensory_feedback.json",
        "config/leg_muscle_map.json",
        "config/leg_muscle_parameters.json",
        "config/articulated_leg_body.json",
    )
    sources = {relative: ROOT / relative for relative in relative_paths}
    selected_path = (
        Path(muscle_config_path).expanduser().resolve()
        if muscle_config_path is not None
        else DEFAULT_MUSCLE_CONFIG_PATH.resolve()
    )
    selected_key = _muscle_config_source_key(selected_path)
    if selected_key != "config/leg_muscle_parameters.json":
        sources[selected_key] = selected_path
    for relative, path in sources.items():
        if not path.is_file():
            raise ArticulatedOptomotorError(f"required source/config file is missing: {path}")
    hashes = {relative: _sha256_file(path) for relative, path in sources.items()}
    if (expected_muscle_config_sha256 is not None
            and hashes[selected_key] != expected_muscle_config_sha256):
        raise ArticulatedOptomotorError(
            "selected muscle config changed after preflight validation"
        )
    return hashes, sources


def _muscle_config_source_key(path: Path) -> str:
    """Map the chosen profile to a fixed, path-safe bundle inventory key."""

    if path.resolve() == DEFAULT_MUSCLE_CONFIG_PATH.resolve():
        return "config/leg_muscle_parameters.json"
    return "selected-config/leg_muscle_parameters.json"


def _snapshot_sources(out: Path, hashes: Mapping[str, str], sources: Mapping[str, Path]) -> None:
    snapshot_root = out / "sources"
    for relative, path in sources.items():
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if _sha256_file(destination) != hashes[relative]:
            raise ArticulatedOptomotorError(f"source snapshot hash mismatch: {relative}")


def _verify_source_hashes(
    hashes: Mapping[str, str],
    sources: Mapping[str, Path],
    *,
    phase: str,
) -> None:
    current = {relative: _sha256_file(path) for relative, path in sources.items()}
    changed = sorted(key for key in hashes if current.get(key) != hashes[key])
    if changed:
        raise ArticulatedOptomotorError(
            f"source files changed {phase}: {', '.join(changed)}"
        )


def _runtime_mapping_audit(
    runner: DigitalFlyRunner, encoder: LegSensoryFeedback, pipeline: LegMusclePipeline,
) -> dict[str, Any]:
    engine = runner.neural_kind.engine
    sensory_union, sensory_by_leg = leg_sensory_indices(encoder, engine.populations)
    all_pools, mapped = per_leg_indices(pipeline.decoder)
    overlap = selector_overlap_count(engine.populations, runner.sensory_feedback.resolver, sensory_union)
    return {
        "annotated_leg_pool_counts": {leg: int(all_pools[leg].size) for leg in LEG_NAMES},
        "mapped_muscle_motor_counts": {leg: int(mapped[leg].size) for leg in LEG_NAMES},
        "annotated_leg_total": int(sum(values.size for values in all_pools.values())),
        "mapped_motor_total": int(sum(values.size for values in mapped.values())),
        "unmapped_annotated_leg_total": int(
            sum(all_pools[leg].size - mapped[leg].size for leg in LEG_NAMES)
        ),
        "articulated_leg_sensory_counts": {
            leg: int(sensory_by_leg[leg].size) for leg in LEG_NAMES
        },
        "articulated_leg_sensory_total": int(sensory_union.size),
        "leg_target_overlap": overlap,
        "index_source": "the constructed MaleCNS engine's own dense PopulationIndex",
        "all_pool_and_mapped_indices": (all_pools, mapped),
        "sensory_indices": sensory_union,
        "sensory_indices_by_leg": sensory_by_leg,
    }


def _sensory_frame_at_tick(
    frame: SensoryFrame, timestamp: Any, leg_record: Mapping[str, Any],
) -> SensoryFrame:
    """Hold 5 ms visual/legacy input and refresh articulated current per tick."""

    held = _substep_sensory(frame, timestamp)
    channels = dict(held.channels)
    currents = leg_record.get("currents_mV_per_ms")
    if not isinstance(currents, Mapping):
        raise ArticulatedOptomotorError("per-leg sensory encoder did not return a current mapping")
    channels["current_mV_per_ms"] = dict(currents)
    return SensoryFrame(
        timestamp=timestamp,
        visual=held.visual,
        body=held.body,
        channels=channels,
        source=held.source,
    )


def _muscle_frame_summary(
    frame_muscle_samples: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    names = sorted(frame_muscle_samples[-1]) if frame_muscle_samples else []
    fields = ("active_force_N", "passive_force_N", "total_force_N", "excitation", "activation")
    per_channel_peak: dict[str, dict[str, float]] = {}
    for name in names:
        per_channel_peak[name] = {
            field: max(float(sample[name][field]) for sample in frame_muscle_samples)
            for field in fields
        }
    active_end_by_leg: dict[str, float] = {}
    active_peak_by_leg: dict[str, float] = {}
    excitation_peak_by_leg: dict[str, float] = {}
    for leg in LEG_NAMES:
        channels = [name for name in names if name.startswith(f"{leg}_")]
        active_end_by_leg[leg] = math.fsum(
            float(frame_muscle_samples[-1][name]["active_force_N"]) for name in channels
        ) if frame_muscle_samples else 0.0
        active_peak_by_leg[leg] = max((
            math.fsum(float(sample[name]["active_force_N"]) for name in channels)
            for sample in frame_muscle_samples
        ), default=0.0)
        excitation_peak_by_leg[leg] = max(
            (float(sample[name]["excitation"])
             for sample in frame_muscle_samples for name in channels), default=0.0,
        )
    return {
        "active_force_by_leg_N": active_end_by_leg,
        "peak_active_force_by_leg_N": active_peak_by_leg,
        "peak_excitation_by_leg": excitation_peak_by_leg,
        "peak_by_muscle": per_channel_peak,
    }


def _initial_root_pose(body_record: Mapping[str, Any]) -> dict[str, Any]:
    pose, rpy = _pose_from_telemetry(body_record)
    return {
        "position_m": list(pose.position_m),
        "orientation_quat_wxyz": list(body_record["root"]["quaternion_wxyz"]),
        "orientation_rpy_rad": list(rpy),
        "yaw_rad": float(rpy[2]),
        "velocity_mps": list(pose.velocity_mps),
        "angular_velocity_rad_s": list(pose.angular_velocity_radps),
    }


def _build_sensor_frame(
    runner: DigitalFlyRunner,
    pipeline: LegMusclePipeline,
    encoder: LegSensoryFeedback,
    sensory_mode: str,
    body_record: Mapping[str, Any],
    pose: Pose,
    frame_index: int,
    visual_output: Any,
    sampling: Mapping[str, Any],
) -> tuple[SensoryFrame, dict[str, Any], dict[str, Any]]:
    t_s = frame_index * FRAME_DT_S
    ts = _timestamp(t_s, clock="control", rate_hz=1.0 / FRAME_DT_S)
    root = body_record["root"]
    body_contract = _make_contract_body(
        type("State", (), {
            "position_m": root["position_m"],
            "velocity_mps": root["linear_velocity_mps"],
            "angular_velocity_rad_s": root["angular_velocity_radps"],
            "orientation_quat": root["quaternion_wxyz"],
        })(), ts,
    )
    feet_contacts = {leg: bool(body_record["feet"][leg]["contact"]) for leg in LEG_NAMES}
    legacy_contacts = legacy_contact_mapping(body_record["feet"])
    wind = runner.world.wind_at(pose.position_m, t_s)
    old_mech = runner.sensory_feedback.step(
        body_contract,
        world_wind=wind,
        ground_contact=legacy_contacts,
        joint_state=runner._feedback_joint_state(None),
    )
    old_population, old_leg_values = mask_legacy_leg_drives(
        old_mech.population_drive, sensory_mode,
    )
    visual = VisualObservation(
        timestamp=ts,
        pixels=None,
        optic_flow_rad_s=(0.0, 0.0, 0.0),
        sensor_id="virtual-world-stereo-flyvis",
        source=SourceLabel.FITTED,
        metadata={
            **dict(getattr(visual_output, "metadata", {})),
            "world_sensor_backend": getattr(runner.world, "backend_name", "unknown"),
            "flyvis_sampling": dict(sampling),
            "free_behavior_directional_readout": "record_only",
        },
    )
    flyvis_channels = _flyvis_drive_contract(
        runner.flyvis,
        visual_output.t4t5_drive,
        mapping_override=runner._flyvis_engine_mapping,
    )
    sensory_channels: dict[str, Any] = {
        "optic_flow_x_rad_s": 0.0,
        "optic_flow_y_rad_s": 0.0,
        **flyvis_channels,
        **{f"mech:{key}": value for key, value in old_mech.channels.items()},
        **{f"mech_raw:{key}": value for key, value in old_mech.raw_channels.items()},
        **{f"population:{key}": value for key, value in old_population.items()},
        "population_drive_units": str(old_mech.population_drive_units),
        "population_drive_range": list(old_mech.population_drive_range),
        "current_mV_per_ms": {},
    }
    frame = SensoryFrame(
        timestamp=ts,
        visual=visual,
        body=body_contract,
        channels=sensory_channels,
        source=SourceLabel.HANDBUILT,
    )
    feedback_record = {
        "mode": sensory_mode,
        "legacy_population_drive_raw": dict(old_mech.population_drive),
        "legacy_population_drive_applied": dict(old_population),
        "legacy_leg_values_masked": old_leg_values,
        "legacy_body_contact_by_foot": feet_contacts,
        "legacy_body_contact_mapping": legacy_contacts,
        "legacy_world_wind_mps": list(wind),
        "articulated_leg_encoder_ticks": [],
        "source_time_s": float(body_record["time_s"]),
        "applied_at_s": t_s,
    }
    return frame, feedback_record, body_contract.to_dict()


def _world_runtime_manifests(runner: DigitalFlyRunner) -> dict[str, Any]:
    engine = runner.neural_kind.engine
    renderer = getattr(runner.world, "renderer", None)
    renderer_manifest = renderer.manifest() if callable(getattr(renderer, "manifest", None)) else {}
    flyvis_manifest = runner.flyvis_runtime_adapter.manifest()
    neural_manifest = engine.manifest()
    world_manifest = runner.world.manifest()
    return {
        "world": _json_safe(world_manifest),
        "renderer": _json_safe(renderer_manifest),
        "flyvis": _json_safe(flyvis_manifest),
        "malecns": _json_safe(neural_manifest),
    }


def _validate_effective_backends(runner: DigitalFlyRunner) -> dict[str, Any]:
    engine = runner.neural_kind.engine
    runtime = runner.flyvis_runtime_adapter
    if getattr(runner.world, "backend_name", None) != "warp_cuda":
        raise ArticulatedOptomotorError("requested Warp CUDA renderer did not remain selected")
    if (getattr(engine, "backend", None), getattr(engine, "device", None)) != ("torch_cuda", "cuda"):
        raise ArticulatedOptomotorError("MaleCNS did not select the exact Torch CUDA backend")
    device = str(getattr(runtime, "device", "")).lower()
    if not device.startswith("cuda:"):
        raise ArticulatedOptomotorError("FlyVis did not select an explicit CUDA device")
    effective_gain = float(getattr(engine, "gain", math.nan))
    if not math.isclose(effective_gain, 1.2, rel_tol=0.0, abs_tol=float(np.finfo(np.float32).eps)):
        raise ArticulatedOptomotorError("MaleCNS engine gain is not the configured 1.2")
    if float(getattr(engine, "max_vram_gib", math.nan)) > MAX_VRAM_GIB:
        raise ArticulatedOptomotorError("MaleCNS engine VRAM cap exceeds 15.5 GiB")
    torch = getattr(runtime, "torch", None)
    if torch is None or not bool(torch.cuda.is_available()):
        raise ArticulatedOptomotorError("FlyVis Torch reports CUDA unavailable")
    if not bool(engine._torch.cuda.is_available()):
        raise ArticulatedOptomotorError("MaleCNS Torch reports CUDA unavailable")
    return {
        "renderer": {"requested": "warp_cuda", "effective": "warp_cuda",
                     "device": str(getattr(runner.world.renderer, "device_name", "unknown")),
                     "actual_cuda_tensor_seen": False},
        "vision": {"requested": "flyvis", "effective": "flyvis", "device": device,
                   "torch_cuda_available": True, "frames_stepped": 0},
        "neural": {"requested": "torch_cuda", "effective": "torch_cuda", "device": "cuda",
                   "torch_cuda_available": True, "ticks_stepped": 0,
                   "configured_gain": 1.2, "effective_gain": effective_gain},
        "fallback_permitted": False,
        "fallback_used": False,
    }


def _asset_hashes() -> dict[str, str]:
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    root = Path(str(scene["asset_root"]))
    result: dict[str, str] = {}
    for asset in scene.get("assets", []):
        path = root / str(asset["path"])
        if not path.is_file():
            raise ArticulatedOptomotorError(f"scene asset is missing: {path}")
        result[str(path)] = _sha256_file(path)
    return result


def _csr_hashes(path: Path) -> dict[str, str]:
    expected = ("annotations.npz", "ids.npy", "meta.json", "offsets.npy",
                "retina.npy", "sign.npy", "tables.json", "target.npy", "weight.npy")
    result: dict[str, str] = {}
    for name in expected:
        candidate = path / name
        if not candidate.is_file():
            raise ArticulatedOptomotorError(f"MaleCNS CSR source is missing: {candidate}")
        result[name] = _sha256_file(candidate)
    return result


def _root_camera_difference(initial_root: Mapping[str, Any]) -> dict[str, Any]:
    target = [0.0, 0.0, 0.0012]
    position = list(initial_root["position_m"])
    return {
        "original_reduced_runner_camera_position_m": target,
        "articulated_root_position_m": position,
        "articulated_minus_original_z_m": position[2] - target[2],
        "camera_uses_body_root_directly": True,
        "camera_offset_applied": [0.0, 0.0, 0.0],
    }


def run_trial(
    out: Path,
    *,
    seed: int = 1729,
    duration_s: float = DEFAULT_DURATION_S,
    condition: str = "left",
    sensory_feedback: str = "legacy",
    muscle_config_path: str | Path | None = None,
    muscle_force_scale: float = 1.0,
    record_motor_units: bool = False,
) -> dict[str, Any]:
    """Execute one paired stripe condition and write a hash-audited bundle."""

    frame_count = validate_duration(duration_s)
    if condition not in CONDITION_STIMULI:
        raise ValueError(f"condition must be one of {', '.join(CONDITION_STIMULI)}")
    if sensory_feedback not in {"legacy", "off", "on"}:
        raise ValueError("sensory_feedback must be legacy, off, or on")
    if not isinstance(record_motor_units, bool):
        raise ValueError("record_motor_units must be a boolean")
    selected_muscle_path, selected_force_scale, preflight_muscle_hash = validate_muscle_options(
        muscle_config_path,
        muscle_force_scale,
    )
    out = Path(out).expanduser().resolve()
    documents = (Path.home() / "Documents").resolve()
    if out == documents or documents in out.parents:
        raise ValueError("generated project artifacts must not be written under ~/Documents")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing trial directory: {out}")

    resource_report = _require_resource_scope()
    started_setup = time.perf_counter()
    runner = _new_runner(out, seed, duration_s, condition)
    source_hashes, source_paths = _source_inventory(
        runner,
        selected_muscle_path,
        expected_muscle_config_sha256=preflight_muscle_hash,
    )
    out.mkdir(parents=True, exist_ok=False)
    _snapshot_sources(out, source_hashes, source_paths)
    runtime_before = _validate_effective_backends(runner)
    neural = runner._new_neural("free-behavior")
    if neural is not runner.neural_kind:
        raise ArticulatedOptomotorError("strict runner returned an unexpected MaleCNS adapter")
    pipeline = _make_pipeline(
        neural.engine,
        muscle_config_path=selected_muscle_path,
        muscle_force_scale=selected_force_scale,
    )
    encoder = LegSensoryFeedback(index=neural.engine.populations)
    mapping_audit = _runtime_mapping_audit(runner, encoder, pipeline)
    all_pool, mapped = mapping_audit.pop("all_pool_and_mapped_indices")
    sensory_indices = mapping_audit.pop("sensory_indices")
    sensory_by_leg = mapping_audit.pop("sensory_indices_by_leg")
    n_neurons = int(neural.engine.n)
    if n_neurons != int(pipeline.decoder.population_index.n):
        raise ArticulatedOptomotorError("MaleCNS and leg decoder dense neuron spaces differ")
    motor_unit_mapping: dict[str, Any] | None = None
    motor_unit_context: dict[str, np.ndarray] | None = None
    motor_unit_mapping_path: Path | None = None
    motor_unit_mapping_sha256_before_loop: str | None = None
    if record_motor_units:
        motor_unit_mapping, motor_unit_context = build_motor_unit_mapping(
            neural.engine.populations, pipeline.decoder, all_pool, mapped,
        )
        motor_unit_mapping_path = out / "motor_unit_mapping.json"
        motor_unit_mapping_path.write_text(
            json.dumps(motor_unit_mapping, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        motor_unit_mapping_sha256_before_loop = _sha256_file(motor_unit_mapping_path)

    runner.flyvis.reset()
    pipeline.body_snapshot = pipeline.body.snapshot()
    initial_body = pipeline.body_snapshot["telemetry"]
    initial_mujoco_time_s = float(pipeline.body.data.time)
    initial_pose = _initial_root_pose(initial_body)
    # The authored free-root model starts at 1.275 mm. At the old 1.2 mm camera
    # height its 75 µm feet begin 53 µm into the floor, so preserve the
    # physically authored initial model height and record the 75 µm mismatch.
    try:
        from flight_dynamics import FlightState
    except ImportError as exc:
        raise ArticulatedOptomotorError("strict initial-scene adaptation requires FlightState") from exc
    adaptation_state = FlightState(
        time_s=0.0,
        position_m=tuple(initial_pose["position_m"]),
        orientation_quat=tuple(initial_pose["orientation_quat_wxyz"]),
        velocity_mps=tuple(initial_pose["velocity_mps"]),
        angular_velocity_rad_s=tuple(initial_pose["angular_velocity_rad_s"]),
    )
    adaptation_report = runner._preadapt_free_visual_scene(
        type("ArticulatedStateAdapter", (), {"state": adaptation_state})(),
    )
    neural.reset()
    runner.sensory_feedback.reset()
    setup_s = time.perf_counter() - started_setup
    _verify_source_hashes(source_hashes, source_paths, phase="during setup")

    (out / "source_sha256.json").write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    csr_hashes_before = _csr_hashes(Path(neural.engine.csr_path))

    frame_dt_values = {
        key: 0.0 for key in (
            "render_s", "flyvis_s", "feedback_s", "neural_s",
            "muscle_physics_s", "telemetry_write_s", "spike_count_write_s",
        )
    }
    if record_motor_units:
        frame_dt_values["motor_unit_spike_write_s"] = 0.0
    neural_spike_digest = hashlib.sha256()
    totals = {
        "neural_spikes": 0,
        "annotated_leg_spikes": 0,
        "mapped_motor_spikes": 0,
        "articulated_leg_sensory_spikes": 0,
        "ticks": 0,
    }
    annotated_by_leg = {leg: 0 for leg in LEG_NAMES}
    mapped_by_leg = {leg: 0 for leg in LEG_NAMES}
    afferent_by_leg = {leg: 0 for leg in LEG_NAMES}
    per_muscle_max: dict[str, dict[str, float]] = {}
    per_joint_peak_passive_spring_torque: dict[str, float] = {}
    nonfoot_contact_ticks = 0
    renderer_cuda_tensor_seen = False
    frame_stage_rows: list[dict[str, Any]] = []
    loop_started = time.perf_counter()
    current_body = initial_body
    initial_joints = {name: float(row["q_rad"]) for name, row in current_body["joints"].items()}

    try:
        with ExitStack() as output_stack:
            telemetry_fh = output_stack.enter_context(
                (out / "telemetry.jsonl").open("w", encoding="utf-8", buffering=1)
            )
            counts_fh = output_stack.enter_context(
                (out / "neural_spike_counts.jsonl").open("w", encoding="utf-8", buffering=1)
            )
            motor_units_fh = (
                output_stack.enter_context(
                    (out / "motor_unit_spikes.jsonl").open("w", encoding="utf-8", buffering=1)
                )
                if record_motor_units else None
            )
            for frame_index in range(frame_count):
                frame_started = time.perf_counter()
                frame_time_s = frame_index * FRAME_DT_S
                pose, rpy = _pose_from_telemetry(current_body)

                stage = time.perf_counter()
                left_columns, right_columns, sampling = runner._free_exact_flyvis_columns(
                    pose, frame_time_s,
                )
                render_devices = [str(getattr(value, "device", "")).lower()
                                  for value in (left_columns, right_columns)]
                if any(not device.startswith("cuda:") for device in render_devices):
                    raise ArticulatedOptomotorError(
                        "Warp exact-column render did not return two CUDA-resident eye tensors"
                    )
                renderer_cuda_tensor_seen = True
                frame_dt_values["render_s"] += time.perf_counter() - stage

                stage = time.perf_counter()
                visual_output = runner._process_exact_flyvis_columns(
                    left_columns, right_columns, frame_time_s,
                )
                frame_dt_values["flyvis_s"] += time.perf_counter() - stage
                runtime_before["vision"]["frames_stepped"] += 1

                stage = time.perf_counter()
                sensor_frame, feedback_record, body_contract = _build_sensor_frame(
                    runner, pipeline, encoder, sensory_feedback, current_body, pose,
                    frame_index, visual_output, sampling,
                )
                frame_dt_values["feedback_s"] += time.perf_counter() - stage

                drive = np.asarray(visual_output.t4t5_drive, dtype=np.float32)
                drive_summary = {
                    "count": int(drive.size),
                    "min": float(drive.min()) if drive.size else 0.0,
                    "max": float(drive.max()) if drive.size else 0.0,
                    "mean": float(drive.mean()) if drive.size else 0.0,
                    "sha256": _array_sha256(drive),
                    "units": str(getattr(runner.flyvis_runtime_adapter, "units", "unknown")),
                }
                stimulus_state = runner.world.stimulus_state_at(
                    frame_time_s, reference_pose=pose,
                )

                per_leg_annotated_frame = {leg: 0 for leg in LEG_NAMES}
                per_leg_mapped_frame = {leg: 0 for leg in LEG_NAMES}
                per_leg_afferent_frame = {leg: 0 for leg in LEG_NAMES}
                frame_muscle_samples: list[Mapping[str, Mapping[str, Any]]] = []
                frame_spikes = 0
                frame_annotated = 0
                frame_mapped = 0
                frame_afferent = 0
                for substep in range(CNS_TICKS_PER_FRAME):
                    tick = frame_index * CNS_TICKS_PER_FRAME + substep
                    timestamp = neural_tick_timestamp(tick)
                    if abs(timestamp.t_s - (frame_time_s + substep * CNS_DT_S)) > 1e-12:
                        raise ArticulatedOptomotorError("CNS tick clock drifted from the 5 ms frame")
                    expected_source_time = frame_time_s + substep * CNS_DT_S
                    if abs(float(current_body["time_s"]) - expected_source_time) > 1e-9:
                        raise ArticulatedOptomotorError(
                            "articulated sensory source body is not at the current CNS boundary"
                        )
                    feedback_tick_started = time.perf_counter()
                    leg_record = encoder.encode(
                        current_body,
                        timestamp.t_s,
                        enabled=True,
                        silenced=sensory_feedback != "on",
                    )
                    feedback_record["articulated_leg_encoder_ticks"].append({
                        "tick": tick,
                        "source_time_s": float(leg_record["source_time_s"]),
                        "applied_at_s": float(leg_record["applied_at_s"]),
                        "input_applied": bool(leg_record["input_applied"]),
                        "currents_mV_per_ms": dict(leg_record["currents_mV_per_ms"]),
                        "derived_readouts": leg_record["derived_readouts"],
                    })
                    cns_frame = _sensory_frame_at_tick(sensor_frame, timestamp, leg_record)
                    frame_dt_values["feedback_s"] += time.perf_counter() - feedback_tick_started
                    stage = time.perf_counter()
                    neural_output = neural.step(cns_frame, timestamp)
                    spikes = canonical_spikes(neural_output.spikes, n_neurons)
                    frame_dt_values["neural_s"] += time.perf_counter() - stage
                    neural_spike_digest.update(spikes.tobytes(order="C"))
                    spike_count = int(spikes.sum(dtype=np.uint64))
                    annotated_counts = {leg: int(spikes[all_pool[leg]].sum(dtype=np.uint64))
                                        for leg in LEG_NAMES}
                    mapped_counts = {leg: int(spikes[mapped[leg]].sum(dtype=np.uint64))
                                     for leg in LEG_NAMES}
                    afferent_counts = {leg: int(spikes[sensory_by_leg[leg]].sum(dtype=np.uint64))
                                       for leg in LEG_NAMES}
                    annotated_count = sum(annotated_counts.values())
                    mapped_count = sum(mapped_counts.values())
                    afferent_count = sum(afferent_counts.values())
                    for leg in LEG_NAMES:
                        per_leg_annotated_frame[leg] += annotated_counts[leg]
                        per_leg_mapped_frame[leg] += mapped_counts[leg]
                        per_leg_afferent_frame[leg] += afferent_counts[leg]
                        annotated_by_leg[leg] += annotated_counts[leg]
                        mapped_by_leg[leg] += mapped_counts[leg]
                        afferent_by_leg[leg] += afferent_counts[leg]
                    frame_spikes += spike_count
                    frame_annotated += annotated_count
                    frame_mapped += mapped_count
                    frame_afferent += afferent_count
                    totals["neural_spikes"] += spike_count
                    totals["annotated_leg_spikes"] += annotated_count
                    totals["mapped_motor_spikes"] += mapped_count
                    totals["articulated_leg_sensory_spikes"] += afferent_count
                    totals["ticks"] += 1

                    count_started = time.perf_counter()
                    _json_line(counts_fh, {
                        "tick": tick,
                        "t_s": timestamp.t_s,
                        "clock": timestamp.clock,
                        "total_spikes": spike_count,
                        "annotated_leg_spikes": annotated_count,
                        "annotated_leg_spikes_by_leg": annotated_counts,
                        "mapped_motor_spikes": mapped_count,
                        "mapped_motor_spikes_by_leg": mapped_counts,
                        "articulated_leg_sensory_spikes": afferent_count,
                        "articulated_leg_sensory_spikes_by_leg": afferent_counts,
                    })
                    frame_dt_values["spike_count_write_s"] += time.perf_counter() - count_started

                    if motor_units_fh is not None:
                        motor_event_started = time.perf_counter()
                        if motor_unit_context is None:
                            raise ArticulatedOptomotorError(
                                "motor-unit event recording has no validated mapping"
                            )
                        _json_line(motor_units_fh, sparse_motor_unit_event(
                            spikes,
                            motor_unit_context,
                            tick=tick,
                            neural_t_s=timestamp.t_s,
                            physical_t_s=float(current_body["time_s"]),
                            annotated_counts=annotated_counts,
                            mapped_counts=mapped_counts,
                        ))
                        frame_dt_values["motor_unit_spike_write_s"] += (
                            time.perf_counter() - motor_event_started
                        )

                    stage = time.perf_counter()
                    physical = pipeline.step(dt_s=CNS_DT_S, spikes=spikes)
                    frame_dt_values["muscle_physics_s"] += time.perf_counter() - stage
                    expected_time = (tick + 1) * CNS_DT_S
                    if (abs(float(physical["time_s"]) - expected_time) > 1e-9
                            or abs(float(pipeline.body.data.time) - expected_time) > 1e-9):
                        raise ArticulatedOptomotorError(
                            "pipeline/native MuJoCo clock mismatch at neural tick "
                            f"{tick}: pipeline={physical['time_s']}, mujoco={pipeline.body.data.time}, "
                            f"expected={expected_time}"
                        )
                    if physical["body"]["root_applied_wrench"]["max_abs_generalized_force"] != 0.0:
                        raise ArticulatedOptomotorError("root received an applied wrench")
                    frame_muscle_samples.append(physical["muscles"])
                    for muscle_name, muscle in physical["muscles"].items():
                        stats = per_muscle_max.setdefault(muscle_name, {
                            "active_force_N": 0.0, "passive_force_N": 0.0,
                            "total_force_N": 0.0, "excitation": 0.0, "activation": 0.0,
                        })
                        for field in stats:
                            stats[field] = max(stats[field], float(muscle[field]))
                    for joint_name, joint in physical["body"]["joints"].items():
                        per_joint_peak_passive_spring_torque[joint_name] = max(
                            per_joint_peak_passive_spring_torque.get(joint_name, 0.0),
                            abs(float(joint["passive_spring_torque_Nm"])),
                        )
                    current_body = physical["body"]
                    nonfoot_contact_ticks += int(bool(current_body["body_ground_contact"]))

                current_body = dict(current_body)
                current_pose, current_rpy = _pose_from_telemetry(current_body)
                muscle_summary = _muscle_frame_summary(frame_muscle_samples)
                joint_delta = {
                    name: float(state["q_rad"]) - initial_joints[name]
                    for name, state in current_body["joints"].items()
                }
                record = {
                    "frame_index": frame_index,
                    "frame_time_s": frame_time_s,
                    "timestamp": _timestamp(
                        frame_time_s, clock="control", rate_hz=1.0 / FRAME_DT_S,
                    ).to_dict(),
                    "stimulus_condition": f"stripe_{condition}",
                    "stimulus_name": "stripe_panel",
                    "stimulus_state": _json_safe(stimulus_state),
                    "visual": {"drive_summary": drive_summary, "sampling": dict(sampling)},
                    "neural": {
                        "tick_count": CNS_TICKS_PER_FRAME,
                        "total_spikes": frame_spikes,
                        "annotated_leg_spikes": frame_annotated,
                        "annotated_leg_spikes_by_leg": per_leg_annotated_frame,
                        "mapped_motor_spikes": frame_mapped,
                        "mapped_motor_spikes_by_leg": per_leg_mapped_frame,
                        "articulated_leg_sensory_spikes": frame_afferent,
                        "articulated_leg_sensory_spikes_by_leg": per_leg_afferent_frame,
                    },
                    "sensory_feedback": feedback_record,
                    "body": {
                        **current_body,
                        "root": {
                            **current_body["root"],
                            "orientation_rpy_rad": list(current_rpy),
                            "yaw_rad": float(current_rpy[2]),
                        },
                    },
                    "joint_delta_rad": joint_delta,
                    "muscles": muscle_summary,
                    "nonfoot_ground_contact": bool(current_body["body_ground_contact"]),
                    "nonfoot_ground_contact_ticks_total": nonfoot_contact_ticks,
                    "frame_wall_s": time.perf_counter() - frame_started,
                }
                write_started = time.perf_counter()
                _json_line(telemetry_fh, record)
                frame_dt_values["telemetry_write_s"] += time.perf_counter() - write_started
                frame_stage_rows.append({"frame_index": frame_index, "wall_s": record["frame_wall_s"]})

    except BaseException:
        # Keep partial files as diagnostics. The absence of manifest.json means
        # the directory is not a completed accepted trial.
        raise

    synchronization_started = time.perf_counter()
    runtime = runner.flyvis_runtime_adapter
    runtime.torch.cuda.synchronize(runtime.device)
    neural.engine._torch.cuda.synchronize()
    renderer = runner.world.renderer
    try:
        import warp as wp  # type: ignore[import-not-found]
        wp.synchronize_device(renderer.device)
    except Exception as exc:
        raise ArticulatedOptomotorError(f"Warp CUDA synchronization failed: {exc}") from exc
    sync_s = time.perf_counter() - synchronization_started
    loop_wall_s = time.perf_counter() - loop_started

    if totals["ticks"] != frame_count * CNS_TICKS_PER_FRAME:
        raise ArticulatedOptomotorError("the completed run has an unexpected neural tick count")
    if neural.calls != totals["ticks"]:
        raise ArticulatedOptomotorError("MaleCNS adapter did not execute exactly one scalar step per tick")
    if runtime._frame != frame_count:
        raise ArticulatedOptomotorError("FlyVis runtime frame count differs from the requested frame count")

    runtime_before["renderer"]["actual_cuda_tensor_seen"] = renderer_cuda_tensor_seen
    runtime_before["vision"]["frames_stepped"] = frame_count
    runtime_before["neural"]["ticks_stepped"] = totals["ticks"]
    runtime_before["verified"] = bool(
        renderer_cuda_tensor_seen
        and runtime_before["vision"]["device"].startswith("cuda:")
        and runtime_before["neural"]["device"] == "cuda"
        and not runtime_before["fallback_used"]
    )
    if not runtime_before["verified"]:
        raise ArticulatedOptomotorError("actual CUDA execution markers did not all pass")

    final_body = current_body
    final_pose = _initial_root_pose(final_body)
    root_delta = np.asarray(final_pose["position_m"], dtype=np.float64) - np.asarray(
        initial_pose["position_m"], dtype=np.float64,
    )
    summary = {
        "schema_version": "articulated-optomotor-summary-1",
        "trial_id": f"seed-{seed}-{condition}-{sensory_feedback}",
        "seed": int(seed),
        "condition": f"stripe_{condition}",
        "sensory_feedback": mode_contract(sensory_feedback),
        "evidence_class": "digital_simulation",
        "biological_validation": "NOT_VALIDATED",
        "initial_root_pose": initial_pose,
        "final_root_pose": final_pose,
        "final_joint_delta_rad": {
            name: float(state["q_rad"]) - initial_joints[name]
            for name, state in final_body["joints"].items()
        },
        "root_displacement_m": root_delta.tolist(),
        "root_displacement_norm_m": float(np.linalg.norm(root_delta)),
        "root_yaw_change_rad": final_pose["yaw_rad"] - initial_pose["yaw_rad"],
        "initial_height_difference_from_original_camera": _root_camera_difference(initial_pose),
        "counts": {
            **totals,
            "annotated_leg_spikes_by_leg": annotated_by_leg,
            "mapped_motor_spikes_by_leg": mapped_by_leg,
            "articulated_leg_sensory_spikes_by_leg": afferent_by_leg,
            "neural_spike_sha256": neural_spike_digest.hexdigest(),
            "articulated_sensory_target_count_note": (
                "spikes counted in target pools; off/legacy values can be spontaneous and do not establish input-evoked activity"
            ),
        },
        "population_mapping": mapping_audit,
        "muscle_profile_options": {
            "muscle_config_path": str(selected_muscle_path),
            "muscle_force_scale": selected_force_scale,
        },
        "per_muscle_peak": per_muscle_max,
        "per_joint_peak_passive_spring_torque_Nm": per_joint_peak_passive_spring_torque,
        "foot_contacts_final": {
            leg: bool(final_body["feet"][leg]["contact"]) for leg in LEG_NAMES
        },
        "foot_normal_force_final_N": {
            leg: float(final_body["feet"][leg]["normal_force_N"]) for leg in LEG_NAMES
        },
        "nonfoot_ground_contact_ticks": nonfoot_contact_ticks,
        "peak_nonfoot_contact": any(
            bool(row) for row in (int(nonfoot_contact_ticks),)
        ),
        "max_passive_muscle_force_N": max(
            (record["passive_force_N"] for record in per_muscle_max.values()), default=0.0,
        ),
        "max_active_muscle_force_N": max(
            (record["active_force_N"] for record in per_muscle_max.values()), default=0.0,
        ),
        "initial_scene_adaptation": adaptation_report,
        "limitations": [
            "MaleCNS motor spikes are individually observed at 0.5 ms and passed to the established muscle decoder.",
            "Muscle, tendon, contact and passive support parameters are engineering assumptions.",
            "The model's authored 1.275 mm initial root is 75 micrometres above the old 1.2 mm camera reference.",
            "In the preserved project convention the articulated anatomical left legs are on world +Y, while the existing camera/FlyVis left-eye sampling axis is world -Y; stripe labels retain world-Y meaning.",
            "Unmapped annotated leg motor neurons remain unmapped; no gait or body traction is supplied.",
            "No autonomous optomotor turning claim follows from this engineering trial.",
        ],
    }
    final_snapshot = pipeline.body.snapshot()
    final_snapshot_path = out / "final_body_snapshot.json"
    final_snapshot_path.write_text(
        json.dumps(final_snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    _verify_source_hashes(source_hashes, source_paths, phase="during the simulation")
    if (record_motor_units and motor_unit_mapping_path is not None
            and _sha256_file(motor_unit_mapping_path) != motor_unit_mapping_sha256_before_loop):
        raise ArticulatedOptomotorError("motor-unit mapping snapshot changed during the simulation")
    csr_hashes = _csr_hashes(Path(neural.engine.csr_path))
    if csr_hashes_before != _csr_hashes(Path(neural.engine.csr_path)):
        raise ArticulatedOptomotorError("MaleCNS CSR sources changed during the simulation")
    csr_hashes = csr_hashes_before
    artifact_paths = {
        "summary.json": summary_path,
        "performance.json": out / "performance.json",
        "telemetry.jsonl": out / "telemetry.jsonl",
        "neural_spike_counts.jsonl": out / "neural_spike_counts.jsonl",
        "final_body_snapshot.json": final_snapshot_path,
    }
    if record_motor_units:
        if motor_unit_mapping_path is None:
            raise ArticulatedOptomotorError("motor-unit mapping snapshot was not written")
        artifact_paths["motor_unit_mapping.json"] = motor_unit_mapping_path
        artifact_paths["motor_unit_spikes.jsonl"] = out / "motor_unit_spikes.jsonl"
    stage_seconds = {
        **frame_dt_values,
        "cuda_synchronization_s": sync_s,
        "other_loop_s": max(0.0, loop_wall_s - sum(frame_dt_values.values()) - sync_s),
    }
    stage_semantics = {
        "render_s": "two exact Warp eye-column renders per 5 ms frame",
        "flyvis_s": "fitted FlyVis CUDA frame, including device output mapping",
        "feedback_s": "legacy aggregate and per-leg afferent body-state encoding",
        "neural_s": "one Torch CUDA MaleCNS scalar integration per 0.5 ms tick",
        "muscle_physics_s": "per-neuron decode, Hill muscles and MuJoCo contact integration",
        "telemetry_write_s": "full per-frame body trajectory serialization and write",
        "spike_count_write_s": "per-tick spike counts and population count serialization",
        "cuda_synchronization_s": "final FlyVis, MaleCNS Torch and Warp device synchronization",
    }
    if record_motor_units:
        stage_semantics["motor_unit_spike_write_s"] = (
            "optional per-tick sparse leg motor-unit body-ID events"
        )
    performance = {
        "schema_version": "articulated-optomotor-performance-1",
        "initialization_s": setup_s,
        "simulation_loop_s": loop_wall_s,
        "simulated_duration_s": frame_count * FRAME_DT_S,
        "flyvis_frames": frame_count,
        "cns_ticks": totals["ticks"],
        "frames_per_wall_second": frame_count / loop_wall_s if loop_wall_s > 0 else None,
        "simulated_seconds_per_wall_second": frame_count * FRAME_DT_S / loop_wall_s if loop_wall_s > 0 else None,
        "mean_frame_wall_ms": loop_wall_s * 1000.0 / frame_count,
        "mean_neural_tick_wall_ms": stage_seconds["neural_s"] * 1000.0 / totals["ticks"],
        "stage_wall_s": stage_seconds,
        "stage_semantics": stage_semantics,
        "memory": {
            **resource_report,
            "torch_cuda_peak_allocated_bytes": int(runtime.torch.cuda.max_memory_allocated(runtime.device)),
            "torch_cuda_peak_reserved_bytes": int(runtime.torch.cuda.max_memory_reserved(runtime.device)),
        },
        "scope": "host loop includes neural hashing, per-tick counts, physics, full trajectory and synchronous final device drain",
    }
    performance_path = artifact_paths["performance.json"]
    performance_path.write_text(json.dumps(performance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifact_hashes = {name: _sha256_file(path) for name, path in artifact_paths.items()}
    config_hashes = {
        str(path): _sha256_file(path)
        for path in (CONFIG_PATH, STIMULUS_PATH, SCENE_PATH)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "trial_id": summary["trial_id"],
        "seed": int(seed),
        "protocol": {
            "condition": f"stripe_{condition}",
            "stimulus_name": "stripe_panel",
            "stimulus_condition": CONDITION_STIMULI[condition],
            "duration_s": frame_count * FRAME_DT_S,
            "flyvis_frames": frame_count,
            "cns_ticks": totals["ticks"],
            "flyvis_dt_s": FRAME_DT_S,
            "cns_dt_s": CNS_DT_S,
            "cns_ticks_per_frame": CNS_TICKS_PER_FRAME,
            "neural_gain": float(neural.engine.gain),
            "stimulus_anchor": "world",
            "camera_initial_pose": initial_pose,
            "initial_root_vs_original_camera": _root_camera_difference(initial_pose),
            "coordinate_semantics": {
                "articulated_anatomical_left_leg_world_side": "+Y",
                "camera_and_flyvis_left_eye_world_sampling_side": "-Y",
                "stripe_left_right_labels": "world Y velocity sign only (+0.004 / -0.004 m/s); no biological turning interpretation",
                "coordinate_transform_changed_for_this_assay": False,
            },
            "synthetic_gait": False,
            "body_traction_command": False,
            "direct_motor_input": False,
        },
        "sensory_feedback": mode_contract(sensory_feedback),
        "requested_backends": {
            "renderer": "warp_cuda", "vision": "flyvis_cuda",
            "neural": "torch_cuda", "muscle": "hill_cpu", "physics": "mujoco_cpu",
        },
        "effective_backends": {
            "renderer": runtime_before["renderer"]["effective"],
            "vision": runtime_before["vision"]["effective"],
            "vision_device": runtime_before["vision"]["device"],
            "neural": runtime_before["neural"]["effective"],
            "neural_device": runtime_before["neural"]["device"],
            "muscle": "hill_cpu", "physics": "mujoco_cpu",
        },
        "cuda_execution": runtime_before,
        "resources": resource_report,
        "initial_state": {
            "root_pose": initial_pose,
            "all_six_initial_foot_contacts": {
                leg: bool(initial_body["feet"][leg]["contact"]) for leg in LEG_NAMES
            },
            "all_six_initial_foot_centers_m": {
                leg: list(initial_body["feet"][leg]["position_m"]) for leg in LEG_NAMES
            },
            "initial_mujoco_time_s": initial_mujoco_time_s,
            "initial_visual_adaptation": adaptation_report,
        },
        "mapping": mapping_audit,
        "runtime_manifests": _world_runtime_manifests(runner),
        "options": {
            "muscle_config_path": str(selected_muscle_path),
            "muscle_force_scale": selected_force_scale,
            "record_motor_units": record_motor_units,
        },
        "muscle_config_archive": {
            "path": f"sources/{_muscle_config_source_key(selected_muscle_path)}",
            "sha256": source_hashes[_muscle_config_source_key(selected_muscle_path)],
        },
        "muscle_pipeline_provenance": pipeline.provenance(),
        "motor_unit_recording": {
            "enabled": record_motor_units,
            "events_file": "motor_unit_spikes.jsonl" if record_motor_units else None,
            "mapping_file": "motor_unit_mapping.json" if record_motor_units else None,
            "events_sha256": (
                artifact_hashes["motor_unit_spikes.jsonl"] if record_motor_units else None
            ),
            "mapping_sha256": (
                artifact_hashes["motor_unit_mapping.json"] if record_motor_units else None
            ),
            "mapping_counts": motor_unit_mapping["counts"] if motor_unit_mapping else None,
            "classification": (
                "unknown; no fast/slow labels were supplied" if record_motor_units else None
            ),
            "event_semantics": (
                {
                    "population": "all annotated leg motor-neuron pools, mapped and unmapped",
                    "encoding": "one JSONL row per 0.5 ms tick with sparse active body IDs",
                    "neural_t_s": "neural tick start time",
                    "physical_t_s": "body state time at the start of the driven physical interval",
                    "physical_interval_end_t_s": "end of the physical interval driven by this event row",
                    "mapped_rates_reconstructable_from_events": True,
                    "rate_filter_tau_s": float(pipeline.rate_filter_tau_s),
                }
                if record_motor_units else None
            ),
        },
        "source_sha256": source_hashes,
        "source_snapshot_directory": "sources",
        "config_sha256": config_hashes,
        "scene_asset_sha256": _asset_hashes(),
        "csr_sha256": csr_hashes,
        "counts": summary["counts"],
        "artifacts": {
            name: {"path": name, "sha256": digest}
            for name, digest in artifact_hashes.items()
        },
        "evidence_class": "digital_simulation",
        "biological_validation": "NOT_VALIDATED",
        "autonomous_optomotor_behavior": "NOT_ESTABLISHED",
        "fallback_permitted": False,
        "fallback_used": False,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "out": str(out),
        "trial_id": summary["trial_id"],
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "performance": str(performance_path),
        "telemetry_rows": frame_count,
        "neural_ticks": totals["ticks"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--condition", choices=tuple(CONDITION_STIMULI), required=True)
    parser.add_argument("--sensory-feedback", choices=("legacy", "off", "on"), default="legacy")
    parser.add_argument("--muscle-config", type=Path, default=None,
                        help="optional strict leg muscle parameter JSON profile")
    parser.add_argument("--muscle-force-scale", type=_parse_force_scale, default=1.0,
                        help="multiply each configured muscle F_iso once; range (0, 10]")
    parser.add_argument("--record-motor-units", action="store_true",
                        help="write sparse per-tick leg motor-neuron body-ID events and mapping")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_trial(
        args.out,
        seed=args.seed,
        duration_s=args.duration,
        condition=args.condition,
        sensory_feedback=args.sensory_feedback,
        muscle_config_path=args.muscle_config,
        muscle_force_scale=args.muscle_force_scale,
        record_motor_units=args.record_motor_units,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
