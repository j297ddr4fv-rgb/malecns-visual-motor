#!/usr/bin/env python3
"""Causal, per-leg sensory currents for the articulated six-leg body.

The encoder translates recorded joint kinematics and foot contact load into
annotation-group current drives. These are bounded engineering proxies: the
MaleCNS annotation groups do not provide per-cell polarity, individual
receptor mapping, or a measured body-to-afferent transfer function.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "leg_sensory_feedback.json"
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
JOINTS = (
    "coxa_yaw", "coxa_roll", "coxa_pitch", "femur_pitch", "femur_roll",
    "tibia_pitch", "tarsus_pitch",
)
LEG_ANNOTATIONS = {
    "LF": ("ProLN", "L", 23),
    "RF": ("ProLN", "R", 13),
    "LM": ("MesoLN", "L", 80),
    "RM": ("MesoLN", "R", 83),
    "LH": ("MetaLN", "L", 93),
    "RH": ("MetaLN", "R", 100),
}
ANNOTATION_CLASS = "mechanosensory_proprioceptive"
CHORDOTONAL_SUBCLASS = "chordotonal organ"
CAMPANIFORM_SUBCLASS = "campaniform sensilla"
CONFIG_KEYS = {
    "schema_version", "dataset", "gain_mV_per_ms", "gain_scale",
    "joint_displacement_scale_rad", "joint_speed_scale_rad_s",
    "per_leg_weight_reference_N", "legs", "provenance",
}
PROVENANCE_KEYS = {
    "annotation_source", "counts_verified_against", "selector_basis",
    "mapping_uncertainty", "neuron_ids_included", "model_rest_rad",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _selector(subclass: str, nerve: str, side: str) -> str:
    return f"class:{ANNOTATION_CLASS}&subclass:{subclass}&nerve:{nerve}/{side}"


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {where} config key(s): {', '.join(sorted(map(str, unknown)))}")


def _index_values(index: Any, selector: str) -> list[int]:
    """Resolve a selector to unique dense indices, failing closed on bad APIs."""

    try:
        result = index.resolve(selector)
    except Exception as exc:
        raise ValueError(f"annotation index failed to resolve selector {selector!r}") from exc
    if result is None:
        raise ValueError(f"annotation index returned no result for selector {selector!r}")
    if hasattr(result, "tolist"):
        result = result.tolist()
    try:
        values = list(result)
    except TypeError as exc:
        raise ValueError(f"annotation index returned an invalid result for {selector!r}") from exc
    dense: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"annotation index returned a non-integer index for {selector!r}")
        try:
            dense.append(operator.index(value))
        except TypeError as exc:
            raise ValueError(f"annotation index returned a non-integer index for {selector!r}") from exc
    if len(dense) != len(set(dense)):
        raise ValueError(f"annotation index returned duplicate indices for {selector!r}")
    return dense


def _body_ids(index: Any, dense_indices: list[int], selector: str) -> list[int]:
    ids = getattr(index, "ids", None)
    if ids is None:
        raise ValueError("annotation index must expose ids to verify target identity")
    try:
        size = len(ids)
    except TypeError as exc:
        raise ValueError("annotation index ids must be an indexable sequence") from exc
    result = []
    for dense in dense_indices:
        if dense < 0 or dense >= size:
            raise ValueError(f"annotation index returned out-of-range index for {selector!r}")
        value = ids[dense]
        if isinstance(value, bool):
            raise ValueError(f"annotation index contains a non-integer body ID for {selector!r}")
        try:
            result.append(operator.index(value))
        except TypeError as exc:
            raise ValueError(f"annotation index contains a non-integer body ID for {selector!r}") from exc
    if len(result) != len(set(result)):
        raise ValueError(f"annotation index body IDs are not unique for {selector!r}")
    return result


def _id_sha256(body_ids: list[int]) -> str:
    """Hash sorted source IDs without putting the full ID inventory in output."""

    canonical = json.dumps(sorted(body_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


class LegSensoryFeedback:
    """Encode one physical body record as bounded per-selector currents.

    ``body_record`` may be the articulated body's telemetry dictionary or a
    complete ``snapshot()`` containing it under ``telemetry``. The record's
    ``time_s`` is preserved as ``source_time_s``; ``timestamp_s`` is the
    neural frame boundary at which the currents are applied.
    """

    def __init__(self, index: Any = None, config_path: str | Path | None = None):
        if index is None:
            # Keep this import lazy so callers that supply the already-loaded
            # annotation index do not import or load any unrelated runtime.
            from neuromuscular_adapter import AnnotationIndex, DEFAULT_CSR
            try:
                index = AnnotationIndex(DEFAULT_CSR)
            except Exception as exc:
                raise ValueError(f"could not load the annotation index at {DEFAULT_CSR}") from exc
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
        try:
            raw_bytes = self.config_path.read_bytes()
            config = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load leg sensory feedback config: {self.config_path}") from exc
        if not isinstance(config, dict):
            raise ValueError("leg sensory feedback config must be a JSON object")
        _reject_unknown(config, CONFIG_KEYS, "top-level")
        if set(config) != CONFIG_KEYS:
            missing = CONFIG_KEYS - set(config)
            raise ValueError(f"missing leg sensory feedback config key(s): {', '.join(sorted(missing))}")
        if config["schema_version"] != 1 or isinstance(config["schema_version"], bool):
            raise ValueError("schema_version must be 1")
        if config["dataset"] != "MaleCNS v1.0":
            raise ValueError("dataset must be MaleCNS v1.0")

        self.gain_mV_per_ms = _finite(config["gain_mV_per_ms"], "gain_mV_per_ms")
        self.gain_scale = _finite(config["gain_scale"], "gain_scale")
        if not 0.0 <= self.gain_mV_per_ms <= 2.0:
            raise ValueError("gain_mV_per_ms must be in [0, 2]")
        if not 0.0 <= self.gain_scale <= 4.0:
            raise ValueError("gain_scale must be in [0, 4]")
        if self.gain_mV_per_ms * self.gain_scale > 2.0:
            raise ValueError("effective gain must not exceed 2 mV/ms")
        self.effective_gain_mV_per_ms = self.gain_mV_per_ms * self.gain_scale
        self.joint_displacement_scale_rad = _finite(
            config["joint_displacement_scale_rad"], "joint_displacement_scale_rad"
        )
        self.joint_speed_scale_rad_s = _finite(
            config["joint_speed_scale_rad_s"], "joint_speed_scale_rad_s"
        )
        self.per_leg_weight_reference_N = _finite(
            config["per_leg_weight_reference_N"], "per_leg_weight_reference_N"
        )
        if self.joint_displacement_scale_rad <= 0:
            raise ValueError("joint_displacement_scale_rad must be positive")
        if self.joint_speed_scale_rad_s <= 0:
            raise ValueError("joint_speed_scale_rad_s must be positive")
        if self.per_leg_weight_reference_N <= 0:
            raise ValueError("per_leg_weight_reference_N must be positive")

        if config["provenance"] is None or not isinstance(config["provenance"], dict):
            raise ValueError("provenance must be an object")
        provenance = config["provenance"]
        _reject_unknown(provenance, PROVENANCE_KEYS, "provenance")
        if set(provenance) != PROVENANCE_KEYS:
            raise ValueError("provenance is missing required fields")
        if provenance["neuron_ids_included"] is not False:
            raise ValueError("neuron_ids_included must be false")
        if provenance["model_rest_rad"] != 0.0:
            raise ValueError("model_rest_rad is immutable at 0 radians")
        for key in ("annotation_source", "counts_verified_against", "selector_basis", "mapping_uncertainty"):
            if not isinstance(provenance[key], str) or not provenance[key].strip():
                raise ValueError(f"provenance.{key} must be a non-empty string")
        self._provenance_config = dict(provenance)

        raw_legs = config["legs"]
        if not isinstance(raw_legs, dict) or set(raw_legs) != set(LEGS):
            raise ValueError("legs must contain exactly LF, RF, LM, RM, LH and RH")
        self._leg_selectors: dict[str, dict[str, str]] = {}
        expected_counts: dict[str, int] = {}
        for leg in LEGS:
            row = raw_legs[leg]
            if not isinstance(row, dict):
                raise ValueError(f"legs.{leg} must be an object")
            allowed = {"nerve", "side", "chordotonal_expected_count", "campaniform_expected_count"}
            _reject_unknown(row, allowed, f"legs.{leg}")
            if set(row) != allowed:
                raise ValueError(f"legs.{leg} is missing required fields")
            nerve, side, chordotonal_count = LEG_ANNOTATIONS[leg]
            if row["nerve"] != nerve or row["side"] != side:
                raise ValueError(f"legs.{leg} must preserve its audited nerve and side")
            campaniform_count = row["campaniform_expected_count"]
            if (isinstance(row["chordotonal_expected_count"], bool)
                    or row["chordotonal_expected_count"] != chordotonal_count):
                raise ValueError(f"legs.{leg} chordotonal count differs from the audited CSR count")
            if isinstance(campaniform_count, bool) or campaniform_count != 2:
                raise ValueError(f"legs.{leg} campaniform count must be the audited count 2")
            chord = _selector(CHORDOTONAL_SUBCLASS, nerve, side)
            camp = _selector(CAMPANIFORM_SUBCLASS, nerve, side)
            # Fail closed if a future edit accidentally redirects drives to a
            # motor target or leaves the annotation-selector vocabulary.
            for target in (chord, camp):
                terms = target.split("&")
                if ("vnc_motor" in target or len(terms) != 3
                        or not terms[0].startswith("class:")
                        or not terms[1].startswith("subclass:")
                        or not terms[2].startswith("nerve:")):
                    raise ValueError(f"invalid annotation-only sensory selector: {target}")
            self._leg_selectors[leg] = {"chordotonal": chord, "campaniform": camp}
            expected_counts[chord] = chordotonal_count
            expected_counts[camp] = 2

        # Resolve every exact selector against the active annotation source.
        # Config counts are expectations only and cannot manufacture a target.
        self._annotation_source = index
        self._selector_counts: dict[str, int] = {}
        self._selector_id_sha256: dict[str, str] = {}
        self._selector_dense: dict[str, set[int]] = {}
        self._selector_body_ids: dict[str, set[int]] = {}
        for selector, expected_count in expected_counts.items():
            dense = _index_values(index, selector)
            if not dense:
                raise ValueError(f"sensory selector resolves to no neurons: {selector}")
            if len(dense) != expected_count:
                raise ValueError(
                    f"sensory selector {selector!r} resolved {len(dense)} neurons; "
                    f"expected {expected_count}"
                )
            body_ids = _body_ids(index, dense, selector)
            self._selector_counts[selector] = len(dense)
            self._selector_id_sha256[selector] = _id_sha256(body_ids)
            self._selector_dense[selector] = set(dense)
            self._selector_body_ids[selector] = set(body_ids)
        self._selector_expected_counts = dict(expected_counts)

        selectors = list(expected_counts)
        for offset, selector in enumerate(selectors):
            for other in selectors[offset + 1:]:
                if self._selector_dense[selector] & self._selector_dense[other]:
                    raise ValueError(f"sensory annotation targets overlap: {selector!r} and {other!r}")
                if self._selector_body_ids[selector] & self._selector_body_ids[other]:
                    raise ValueError(f"sensory body-ID targets overlap: {selector!r} and {other!r}")

        motor_selector = "superclass:vnc_motor"
        motor_dense = _index_values(index, motor_selector)
        if not motor_dense:
            raise ValueError("superclass:vnc_motor resolves to no neurons; cannot verify sensory target separation")
        motor_body_ids = _body_ids(index, motor_dense, motor_selector)
        sensory_dense_union = set().union(*self._selector_dense.values())
        sensory_body_id_union = set().union(*self._selector_body_ids.values())
        if sensory_dense_union & set(motor_dense) or sensory_body_id_union & set(motor_body_ids):
            raise ValueError("sensory annotation targets overlap superclass:vnc_motor")
        self._motor_count = len(motor_dense)
        self._motor_id_sha256 = _id_sha256(motor_body_ids)
        self._resolved_count_total = sum(self._selector_counts.values())

        self.config_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self._config_values = config

    def provenance(self) -> dict[str, Any]:
        """Return the validated mapping and the config content hash."""

        return {
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "dataset": "MaleCNS v1.0",
            "annotation_source": self._provenance_config["annotation_source"],
            "counts_verified_against": self._provenance_config["counts_verified_against"],
            "selector_basis": self._provenance_config["selector_basis"],
            "mapping_uncertainty": self._provenance_config["mapping_uncertainty"],
            "model_rest_rad": 0.0,
            "selectors": dict(self._selector_counts),
            "expected_selectors": dict(self._selector_expected_counts),
            "selector_body_id_sha256": dict(self._selector_id_sha256),
            "identity_source": "index.ids[dense_indices] from the active AnnotationIndex; IDs hashed in sorted order",
            "resolved_count_total": self._resolved_count_total,
            "neuron_ids_included": False,
            "vnc_motor_resolved_count": self._motor_count,
            "vnc_motor_body_id_sha256": self._motor_id_sha256,
            "vnc_motor_target_overlap": False,
        }

    def encode(
        self,
        body_record: Any,
        timestamp_s: Any,
        *,
        enabled: bool = True,
        silenced: bool = False,
    ) -> dict[str, Any]:
        """Encode the previous physical tick at a neural frame boundary.

        ``silenced`` masks this sensory adapter's currents only; it does not
        silence afferent neurons or alter the MaleCNS network.
        """

        if not isinstance(enabled, bool) or not isinstance(silenced, bool):
            raise ValueError("enabled and silenced must be bool values")
        applied_at_s = _finite(timestamp_s, "timestamp_s")
        record = body_record
        if isinstance(record, Mapping) and isinstance(record.get("telemetry"), Mapping):
            record = record["telemetry"]
        if not isinstance(record, Mapping):
            raise ValueError("body_record must be telemetry or a snapshot containing telemetry")
        if "time_s" not in record:
            raise ValueError("body_record requires time_s")
        source_time_s = _finite(record["time_s"], "body_record.time_s")
        if source_time_s > applied_at_s + 1e-10:
            raise ValueError("body_record is from the future relative to timestamp_s")

        joints = record.get("joints")
        feet = record.get("feet")
        if not isinstance(joints, Mapping) or not isinstance(feet, Mapping):
            raise ValueError("body_record requires joints and feet mappings")

        currents: dict[str, float] = {}
        raw_readouts: dict[str, Any] = {
            "joint_angles_rad": {},
            "joint_speeds_rad_s": {},
            "foot_normal_force_N": {},
        }
        derived_by_leg: dict[str, Any] = {}
        active = enabled and not silenced
        for leg in LEGS:
            angle_values: dict[str, float] = {}
            speed_values: dict[str, float] = {}
            for joint in JOINTS:
                name = f"{leg}_{joint}"
                state = joints.get(name)
                if not isinstance(state, Mapping):
                    raise ValueError(f"body_record is missing joint {name}")
                if "q_rad" not in state or "qd_rad_s" not in state:
                    raise ValueError(f"joint {name} requires q_rad and qd_rad_s")
                angle_values[joint] = _finite(state["q_rad"], f"joints.{name}.q_rad")
                speed_values[joint] = _finite(state["qd_rad_s"], f"joints.{name}.qd_rad_s")
            foot = feet.get(leg)
            if not isinstance(foot, Mapping) or "normal_force_N" not in foot:
                raise ValueError(f"body_record is missing normal force for leg {leg}")
            normal_force_N = _finite(foot["normal_force_N"], f"feet.{leg}.normal_force_N")
            if normal_force_N < 0:
                raise ValueError(f"feet.{leg}.normal_force_N must be nonnegative")

            # RMS across the seven model joints is rectified magnitude only;
            # there is no left/right or flexion/extension directional tuning.
            displacement_rms_rad = math.sqrt(sum(value * value for value in angle_values.values()) / len(JOINTS))
            speed_rms_rad_s = math.sqrt(sum(value * value for value in speed_values.values()) / len(JOINTS))
            displacement_norm = min(1.0, displacement_rms_rad / self.joint_displacement_scale_rad)
            speed_norm = min(1.0, speed_rms_rad_s / self.joint_speed_scale_rad_s)
            chordotonal_signal = 0.5 * (displacement_norm + speed_norm)
            load_proxy = min(1.0, normal_force_N / self.per_leg_weight_reference_N)
            raw_readouts["joint_angles_rad"][leg] = angle_values
            raw_readouts["joint_speeds_rad_s"][leg] = speed_values
            raw_readouts["foot_normal_force_N"][leg] = normal_force_N
            derived_by_leg[leg] = {
                "joint_displacement_rms_rad": displacement_rms_rad,
                "joint_displacement_norm": displacement_norm,
                "joint_speed_rms_rad_s": speed_rms_rad_s,
                "joint_speed_norm": speed_norm,
                "chordotonal_proxy_norm": chordotonal_signal,
                "foot_load_proxy_norm": load_proxy,
            }
            chord_selector = self._leg_selectors[leg]["chordotonal"]
            camp_selector = self._leg_selectors[leg]["campaniform"]
            currents[chord_selector] = (
                self.effective_gain_mV_per_ms * chordotonal_signal if active else 0.0
            )
            currents[camp_selector] = self.effective_gain_mV_per_ms * load_proxy if active else 0.0

        return {
            "source_time_s": source_time_s,
            "applied_at_s": applied_at_s,
            "currents_mV_per_ms": currents,
            "raw_readouts": raw_readouts,
            "derived_readouts": {"per_leg": derived_by_leg},
            "enabled": enabled,
            "input_muted": silenced,
            "input_applied": active,
            "current_units": "mV/ms",
            "provenance": self.provenance(),
        }
