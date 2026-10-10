#!/usr/bin/env python3
"""Scope-limited DNp15 baseline and direction-readout diagnostics.

This module contains no FlyVis/MaleCNS runner, stimulus selector, body model,
or actuation code. It makes the historical fixed-head c=1 stripe reference
explicit, exposes a separate zero-gray candidate, and recomputes rate/EMA/
opponent/gate diagnostics from recorded rows. A readout sign is an
opponent-rate sign in this engineered coordinate; it is not a natural
DNp15 direction, yaw, or behavior label.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

try:
    from flygym_rest_contract import FlyGymRestContract
except ModuleNotFoundError:
    from scripts.flygym_rest_contract import FlyGymRestContract


SCHEMA_VERSION = "flygym-direction-calibration-v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "flygym_direction_calibration_v1.json"
)


class DirectionCalibrationError(ValueError):
    """Raised when a calibration request violates its declared contract."""


class UnsupportedBaselineError(DirectionCalibrationError):
    """Raised when requested baseline evidence is absent or out of scope."""


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise DirectionCalibrationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectionCalibrationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise DirectionCalibrationError(f"{name} must be finite{suffix}")
    return result


@dataclass(frozen=True)
class BaselineSpec:
    """Resolved bilateral rate reference, with its evidence boundary."""

    mode: str
    left_hz: float
    right_hz: float
    normalization_hz: float
    scope_id: str
    evidence_status: str
    diagnostic_only: bool


def load_calibration_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the versioned calibration JSON."""
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionCalibrationError(f"cannot load calibration config: {source}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise DirectionCalibrationError(f"unsupported calibration config: {source}")
    return config


def resolve_baseline(
    config: Mapping[str, Any], mode: str | None = None
) -> BaselineSpec:
    """Resolve one fixed baseline policy; stimulus labels never select it.

    `calibrated_static` reproduces the legacy static-stripe control reference.
    `zero_gray` is the explicit candidate using the independently observed gray
    prelude. The zero rate is normalized by the separately frozen 142.5 Hz
    scale so that a zero denominator cannot manufacture a saturated signal.
    A broad/background-neutral baseline fails closed until independently
    calibrated.
    """
    policy = config.get("baseline_policy")
    if not isinstance(policy, Mapping):
        raise DirectionCalibrationError("baseline_policy must be a mapping")
    selected = mode or policy.get("default_mode")
    if not isinstance(selected, str):
        raise DirectionCalibrationError("baseline mode is required")

    if selected == "independent_neutral":
        neutral = policy.get("independent_neutral")
        status = neutral.get("status") if isinstance(neutral, Mapping) else None
        reason = neutral.get("unsupported_reason") if isinstance(neutral, Mapping) else None
        raise UnsupportedBaselineError(
            f"independent_neutral baseline is {status or 'unsupported'}: "
            f"{reason or 'no qualifying calibration data'}"
        )

    definitions = policy.get("modes")
    entry = definitions.get(selected) if isinstance(definitions, Mapping) else None
    if not isinstance(entry, Mapping):
        raise DirectionCalibrationError(f"unknown baseline mode: {selected}")
    if entry.get("status") not in {"calibrated_scope_limited", "candidate_reference"}:
        raise UnsupportedBaselineError(
            f"baseline mode {selected} is not available: {entry.get('status')}"
        )
    rates = entry.get("rates_hz")
    if not isinstance(rates, Mapping):
        raise DirectionCalibrationError(f"baseline mode {selected} has no bilateral rates")
    normalization = _finite_number(
        entry.get("normalization_hz"), f"{selected}.normalization_hz", minimum=0.0
    )
    if normalization <= 0.0:
        raise DirectionCalibrationError(f"baseline mode {selected} needs positive normalization")
    return BaselineSpec(
        mode=selected,
        left_hz=_finite_number(rates.get("left"), f"{selected}.left_hz", minimum=0.0),
        right_hz=_finite_number(rates.get("right"), f"{selected}.right_hz", minimum=0.0),
        normalization_hz=normalization,
        scope_id=str(entry.get("scope_id", "unspecified")),
        evidence_status=str(entry["status"]),
        diagnostic_only=bool(entry.get("diagnostic_only", True)),
    )


def validate_static_stripe_scope(
    requested_scope: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Check a caller-supplied scope against the static-reference family.

    This is an explicit validation call. There is deliberately no automatic
    selection from a case name, moving direction, desired response, or yaw.
    The reference may be used only for matched fixed-head c=1, 60-degree
    angular stripe comparisons; this check does not validate their direction
    signs or biological interpretation. The scope request must also bind to
    the exact hash-pinned source cohort that supplied this one baseline.
    """
    baseline = resolve_baseline(config, "calibrated_static")
    family = config["baseline_policy"]["modes"]["calibrated_static"]["application_scope"]
    calibration_source = config["baseline_policy"]["modes"][
        "calibrated_static"
    ]["calibration_source"]
    expected = {
        "visual_family": family["visual_family"],
        "pose_mode": family["pose_mode"],
        "display_geometry": family["display_geometry"],
        "contrast": family["contrast"],
        "wavelength_deg": family["wavelength_deg"],
        "calibration_source_cohort_id": calibration_source["source_cohort_id"],
        "runtime_source_identity": calibration_source["source_cohort_identity"],
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, value in expected.items():
        actual = requested_scope.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                matched = math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matched = False
        else:
            matched = actual == value
        if not matched:
            mismatches[key] = (actual, value)
    if mismatches:
        fields = ", ".join(
            f"{key}={actual!r} (requires {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise UnsupportedBaselineError(
            f"static baseline {baseline.scope_id} is out of scope: {fields}"
        )
    return {
        "scope_id": baseline.scope_id,
        "validated": True,
        "matched_fields": list(expected),
        "direction_sign_validated": False,
        "natural_behavior_validated": False,
    }


def _ema_alpha(frame_dt_s: float, tau_s: float) -> float:
    return 1.0 - math.exp(-frame_dt_s / tau_s)


def recompute_direction_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    baseline_mode: str | None = None,
    power_threshold: float | None = None,
    readout_mode: str = "control",
    requested_scope: Mapping[str, Any] | None = None,
    allow_out_of_scope_static_shadow: bool = False,
) -> list[dict[str, Any]]:
    """Recompute raw rates, EMA, opponent deltas and power-gate states.

    Input rows need only `P`, `dn_p15_left_rate_hz`, and
    `dn_p15_right_rate_hz`; optional `time_s` is copied to output. No stimulus,
    case label, desired direction, body pose, or yaw is consulted. Both raw
    opponent difference and baseline-corrected control values are emitted on
    every row so a reader can audit either branch. Gate threshold can be
    overridden for offline re-analysis; the configured 1e-6 value reproduces
    the historical diagnostic split and is explicitly not an independently
    calibrated behavior threshold.
    """
    baseline = resolve_baseline(config, baseline_mode)
    if readout_mode not in {"control", "raw_difference"}:
        raise DirectionCalibrationError("readout_mode must be control or raw_difference")
    scope_validation: dict[str, Any] = {
        "validated": None,
        "status": "not_applicable",
    }
    if baseline.mode == "calibrated_static":
        if requested_scope is None:
            if not allow_out_of_scope_static_shadow:
                raise UnsupportedBaselineError(
                    "calibrated_static requires explicit matched-stripe scope metadata; "
                    "use zero_gray for the fixed cross-case candidate policy"
                )
            scope_validation = {
                "validated": False,
                "status": "out_of_scope_diagnostic_only",
            }
        else:
            try:
                validated = validate_static_stripe_scope(requested_scope, config)
                scope_validation = {**validated, "status": "matched_scope"}
            except UnsupportedBaselineError:
                if not allow_out_of_scope_static_shadow:
                    raise
                scope_validation = {
                    "validated": False,
                    "status": "out_of_scope_diagnostic_only",
                }

    timing = config.get("readout", {})
    dt = _finite_number(timing.get("frame_dt_s"), "frame_dt_s", minimum=0.0)
    tau = _finite_number(timing.get("ema_tau_s"), "ema_tau_s", minimum=0.0)
    if dt <= 0.0 or tau <= 0.0:
        raise DirectionCalibrationError("frame_dt_s and ema_tau_s must be positive")

    gate_config = config.get("power_gate", {})
    threshold_value = (
        gate_config.get("legacy_diagnostic_threshold_p")
        if power_threshold is None
        else power_threshold
    )
    threshold = _finite_number(threshold_value, "power_threshold", minimum=0.0)
    alpha = _ema_alpha(dt, tau)

    raw_ema_left: float | None = None
    raw_ema_right: float | None = None
    command_ema_left = baseline.left_hz
    command_ema_right = baseline.right_hz
    rest_contract = FlyGymRestContract(threshold)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DirectionCalibrationError(f"row {index} must be a mapping")
        power = _finite_number(row.get("P"), f"row[{index}].P", minimum=0.0)
        left = _finite_number(
            row.get("dn_p15_left_rate_hz"),
            f"row[{index}].dn_p15_left_rate_hz",
            minimum=0.0,
        )
        right = _finite_number(
            row.get("dn_p15_right_rate_hz"),
            f"row[{index}].dn_p15_right_rate_hz",
            minimum=0.0,
        )

        raw_ema_initialized = raw_ema_left is None
        if raw_ema_initialized:
            raw_ema_left, raw_ema_right = left, right
        else:
            raw_ema_left += alpha * (left - raw_ema_left)
            raw_ema_right += alpha * (right - raw_ema_right)

        gate = rest_contract.step(power)
        active = gate.active
        transition = gate.transition

        # The command EMA uses the selected reference only while active.
        # Independently, raw_ema_* continues through static/neutral frames so
        # a later gate threshold can be recomputed without losing rate history.
        if active:
            command_ema_left += alpha * (left - command_ema_left)
            command_ema_right += alpha * (right - command_ema_right)
        else:
            command_ema_left = baseline.left_hz
            command_ema_right = baseline.right_hz

        command_ema_difference_hz = command_ema_left - command_ema_right
        raw_difference_hz = raw_ema_left - raw_ema_right
        corrected_difference_hz = (
            (raw_ema_left - baseline.left_hz)
            - (raw_ema_right - baseline.right_hz)
        )
        command_corrected_difference_hz = (
            (command_ema_left - baseline.left_hz)
            - (command_ema_right - baseline.right_hz)
        )
        normalization = baseline.normalization_hz
        control_unclipped = corrected_difference_hz / normalization
        command_control_unclipped = command_corrected_difference_hz / normalization
        raw_unclipped = raw_difference_hz / normalization
        selected_unclipped = (
            control_unclipped if readout_mode == "control" else raw_unclipped
        )
        ungated_clipped_control = max(-1.0, min(1.0, control_unclipped))
        clipped_raw = max(-1.0, min(1.0, raw_unclipped))
        gated_control = command_control_unclipped if active else 0.0
        gated_raw = raw_unclipped if active else 0.0
        control_authorized = (
            baseline.mode == "zero_gray"
            and readout_mode == "control"
            and not baseline.diagnostic_only
        )
        legacy_shadow_gated = gated_control if baseline.mode == "calibrated_static" else None
        legacy_shadow_clipped = (
            max(-1.0, min(1.0, legacy_shadow_gated))
            if legacy_shadow_gated is not None
            else None
        )
        emitted_gated_control = gated_control if control_authorized else None
        clipped_control = (
            max(-1.0, min(1.0, emitted_gated_control))
            if emitted_gated_control is not None
            else None
        )
        gated_clipped_raw = max(-1.0, min(1.0, gated_raw))
        selected_clipped = (
            clipped_control if readout_mode == "control" and control_authorized
            else gated_clipped_raw if readout_mode == "raw_difference"
            else None
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "frame_index": index,
            "baseline_mode": baseline.mode,
            "baseline_scope_id": baseline.scope_id,
            "baseline_scope_validated": scope_validation["validated"],
            "baseline_scope_status": scope_validation["status"],
            "baseline_diagnostic_only": baseline.diagnostic_only,
            "control_authorized": control_authorized,
            "readout_mode": readout_mode,
            "raw_left_hz": left,
            "raw_right_hz": right,
            "raw_ema_left_hz": raw_ema_left,
            "raw_ema_right_hz": raw_ema_right,
            "raw_ema_initialized": raw_ema_initialized,
            "ema_left_hz": command_ema_left,
            "ema_right_hz": command_ema_right,
            "baseline_left_hz": baseline.left_hz,
            "baseline_right_hz": baseline.right_hz,
            "normalization_hz": normalization,
            "raw_opponent_difference_hz": raw_difference_hz,
            "continuous_raw_ema_opponent_difference_hz": raw_difference_hz,
            "command_ema_opponent_difference_hz": command_ema_difference_hz,
            "baseline_corrected_opponent_difference_hz": corrected_difference_hz,
            "unclipped_control_signal": control_unclipped,
            "ungated_opponent_signal": control_unclipped,
            "ungated_clipped_control_signal": ungated_clipped_control,
            "command_ema_baseline_corrected_opponent_difference_hz": command_corrected_difference_hz,
            "command_unclipped_control_signal": command_control_unclipped,
            "unclipped_raw_difference_signal": raw_unclipped,
            "selected_unclipped_signal": selected_unclipped,
            "clipped_control_signal": clipped_control,
            "clipped_raw_difference_signal": clipped_raw,
            "gated_clipped_raw_difference_signal": gated_clipped_raw,
            "selected_clipped_signal": selected_clipped,
            "power_raw": power,
            "power_threshold": threshold,
            "power_active": active,
            "power_static": gate.static,
            "power_state": gate.state,
            "power_control": gate.control_power,
            "power_previous_active": gate.previous_active,
            "power_gate_transition": transition,
            "gated_unclipped_control_signal": emitted_gated_control,
            "legacy_shadow_gated_unclipped_signal": legacy_shadow_gated,
            "legacy_shadow_clipped_signal": legacy_shadow_clipped,
            "gated_unclipped_raw_difference_signal": gated_raw,
            "gate_threshold_status": gate_config.get("status", "unspecified"),
            "direction_semantics": "opponent-rate coordinate only; natural DNp15 direction and yaw unvalidated",
        }
        if "time_s" in row:
            record["time_s"] = _finite_number(row["time_s"], f"row[{index}].time_s")
        output.append(record)
    return output


def unsupported_general_neutral_status(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the declared evidence status without silently fitting a baseline."""
    policy = config.get("baseline_policy", {})
    neutral = policy.get("independent_neutral", {}) if isinstance(policy, Mapping) else {}
    return {
        "status": neutral.get("status", "unsupported"),
        "supported": False,
        "unsupported_reason": neutral.get("unsupported_reason", "no calibration evidence"),
        "required_policy": neutral.get("future_policy"),
    }


def derive_independent_neutral_candidate(
    captures: Iterable[Mapping[str, Any]],
    *,
    preregistered_context_ids: Iterable[str],
) -> dict[str, Any]:
    """Pool a separate no-motion calibration panel without target labels.

    Each row must provide `capture_id`, `context_id`, `role` (exactly
    `neutral_calibration`), `motion_present=False`, `target_present=False`,
    `direction_label=None`, `yaw_used=False`, `behavior_outcome_used=False`,
    `duration_s`, and non-negative left/right spike counts. The context list
    must be frozen before capture. The function returns a candidate estimate
    only; it cannot upgrade `independent_neutral` from unsupported. A separate
    preregistered equivalence assessment and held-out validation are still
    required before installation.

    Capture rates are averaged within context, then contexts receive equal
    weight. This prevents longer or more numerous strata from defining the
    shared scalar baseline by accident. No moving case, target side, desired
    direction, downstream output, or yaw can be passed as a selector.
    """
    context_ids = tuple(preregistered_context_ids)
    if not context_ids or any(not isinstance(value, str) or not value for value in context_ids):
        raise DirectionCalibrationError(
            "preregistered_context_ids must be a non-empty frozen list of names"
        )
    if len(set(context_ids)) != len(context_ids):
        raise DirectionCalibrationError("preregistered_context_ids must be unique")

    seen_capture_ids: set[str] = set()
    rates_by_context: dict[str, list[tuple[float, float]]] = {
        context_id: [] for context_id in context_ids
    }
    for index, capture in enumerate(captures):
        if not isinstance(capture, Mapping):
            raise DirectionCalibrationError(f"capture {index} must be a mapping")
        capture_id = capture.get("capture_id")
        context_id = capture.get("context_id")
        if not isinstance(capture_id, str) or not capture_id:
            raise DirectionCalibrationError(f"capture {index} needs capture_id")
        if capture_id in seen_capture_ids:
            raise DirectionCalibrationError(f"duplicate capture_id: {capture_id}")
        seen_capture_ids.add(capture_id)
        if not isinstance(context_id, str) or context_id not in rates_by_context:
            raise DirectionCalibrationError(
                f"capture {capture_id} uses undeclared context_id {context_id!r}"
            )
        required_flags = {
            "role": "neutral_calibration",
            "motion_present": False,
            "target_present": False,
            "direction_label": None,
            "yaw_used": False,
            "behavior_outcome_used": False,
        }
        for key, expected in required_flags.items():
            if capture.get(key) != expected:
                raise DirectionCalibrationError(
                    f"capture {capture_id} violates independent-neutral field {key}"
                )
        duration = _finite_number(
            capture.get("duration_s"), f"capture[{capture_id}].duration_s", minimum=0.0
        )
        if duration <= 0.0:
            raise DirectionCalibrationError(f"capture {capture_id} duration must be positive")
        left_events = capture.get("left_spike_count")
        right_events = capture.get("right_spike_count")
        if (
            isinstance(left_events, bool)
            or not isinstance(left_events, int)
            or left_events < 0
        ):
            raise DirectionCalibrationError(
                f"capture {capture_id} left_spike_count must be a non-negative integer"
            )
        if (
            isinstance(right_events, bool)
            or not isinstance(right_events, int)
            or right_events < 0
        ):
            raise DirectionCalibrationError(
                f"capture {capture_id} right_spike_count must be a non-negative integer"
            )
        rates_by_context[str(context_id)].append(
            (left_events / duration, right_events / duration)
        )

    missing = [context for context, values in rates_by_context.items() if not values]
    if missing:
        raise DirectionCalibrationError(
            f"independent-neutral panel is missing preregistered contexts: {missing}"
        )
    context_means = {
        context_id: {
            "captures": len(values),
            "left_hz": sum(rate[0] for rate in values) / len(values),
            "right_hz": sum(rate[1] for rate in values) / len(values),
        }
        for context_id, values in rates_by_context.items()
    }
    baseline_left = sum(value["left_hz"] for value in context_means.values()) / len(context_means)
    baseline_right = sum(value["right_hz"] for value in context_means.values()) / len(context_means)
    for value in context_means.values():
        value["left_delta_from_equal_context_mean_hz"] = value["left_hz"] - baseline_left
        value["right_delta_from_equal_context_mean_hz"] = value["right_hz"] - baseline_right
    return {
        "status": "candidate_pending_independent_equivalence_and_heldout_validation",
        "supported_for_general_use": False,
        "left_hz": baseline_left,
        "right_hz": baseline_right,
        "context_ids": list(context_ids),
        "captures": len(seen_capture_ids),
        "weighting": "equal context weight; equal capture weight within each context",
        "by_context": context_means,
        "direction_labels_used": False,
        "yaw_used": False,
        "behavior_outcomes_used": False,
        "installable": False,
    }
