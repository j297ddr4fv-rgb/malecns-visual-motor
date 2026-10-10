#!/usr/bin/env python3
"""Stream and assess prescribed single-leg lift/release engineering traces.

This validator deliberately treats a useful swing as a sustained clearance
event with continuous support, rather than accepting a peak clearance. It
checks recorded traces and their declared input schedule; it does not run the
musculoskeletal model or establish biological validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
DEFAULT_DT_S = 0.0005
DEFAULT_FOOT_RADIUS_M = 75e-6
DEFAULT_GAP_M = 25e-6
DEFAULT_GAP_DURATION_S = 0.020
DEFAULT_RELEASE_STABLE_S = 0.020
DEFAULT_MIN_BODY_HEIGHT_M = 0.0008


class TraceError(ValueError):
    """Raised for malformed traces that cannot be assessed safely."""


def _finite(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceError(f"{where} is not numeric") from exc
    if not math.isfinite(result):
        raise TraceError(f"{where} is not finite")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_name(row: Mapping[str, Any]) -> str:
    value = row.get("phase", row.get("phase_name"))
    if not isinstance(value, str) or not value:
        raise TraceError("row has no phase name")
    aliases = {
        "settle": "pre_settle", "settling": "pre_settle",
        "drive": "imposed_drive", "active_drive": "imposed_drive",
        "release_after_drive": "release", "post_drive_release": "release",
    }
    return aliases.get(value, value)


def _case_name(row: Mapping[str, Any]) -> str:
    value = row.get("case", row.get("case_name", row.get("case_id", row.get("trial"))))
    if not isinstance(value, str) or not value:
        raise TraceError("row has no case name")
    return value


def _feet(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize both the established LF trace and six-foot trace formats."""
    raw = row.get("feet")
    if isinstance(raw, Mapping):
        feet: dict[str, dict[str, Any]] = {}
        for leg in LEGS:
            record = raw.get(leg)
            if not isinstance(record, Mapping):
                continue
            position = record.get("position_m", record.get("center_position_m"))
            center_z = record.get("center_z_m")
            if center_z is None and isinstance(position, (list, tuple)) and len(position) >= 3:
                center_z = position[2]
            feet[leg] = {
                "contact": record.get("contact"),
                "normal_force_N": record.get("normal_force_N", record.get("normal_load_N")),
                "center_z_m": center_z,
                "clearance_m": record.get("clearance_m", record.get(
                    "clearance_if_airborne_m", record.get("gap_m"))),
            }
        return feet

    # Compatibility with calibrate_leg_lift.py's established per-tick schema.
    lf = row.get("LF_foot")
    other = row.get("other_five")
    feet = {}
    if isinstance(lf, Mapping):
        feet["LF"] = {
            "contact": lf.get("contact"),
            "normal_force_N": lf.get("normal_force_N"),
            "center_z_m": lf.get("center_z_m"),
            "clearance_m": lf.get("clearance_if_airborne_m"),
        }
    if isinstance(other, Mapping) and isinstance(other.get("legs"), Mapping):
        for leg, record in other["legs"].items():
            if leg in LEGS and isinstance(record, Mapping):
                feet[leg] = {
                    "contact": record.get("contact"),
                    "normal_force_N": record.get("normal_force_N"),
                    "center_z_m": record.get("center_z_m"),
                    "clearance_m": record.get("clearance_m"),
                }
    return feet


def _root_position(row: Mapping[str, Any]) -> list[float] | None:
    value = row.get("root_position_m")
    root = row.get("root")
    if value is None and isinstance(root, Mapping):
        value = root.get("position_m")
    if value is None:
        value = row.get("body_position_m")
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [_finite(x, "root_position_m") for x in value[:3]]
    return None


def _root_height(row: Mapping[str, Any], root_position: list[float] | None) -> float | None:
    for key in ("body_height_m", "root_height_m"):
        if row.get(key) is not None:
            return _finite(row[key], key)
    return root_position[2] if root_position is not None else None


def _root_wrench_peak(row: Mapping[str, Any]) -> float | None:
    value = row.get("root_applied_wrench_peak")
    if value is not None:
        return _finite(value, "root_applied_wrench_peak")
    wrench = row.get("root_applied_wrench")
    if isinstance(wrench, Mapping):
        if wrench.get("available") is False:
            return None
        value = wrench.get("max_abs_generalized_force")
        if value is not None:
            return _finite(value, "root_applied_wrench.max_abs_generalized_force")
        components: list[float] = []
        for key in ("force_N", "torque_Nm", "force", "torque"):
            part = wrench.get(key)
            if isinstance(part, (list, tuple)):
                components.extend(_finite(x, f"root_applied_wrench.{key}") for x in part)
        if components:
            return max(abs(x) for x in components)
    return None


def _rates(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("input_rates_hz", "prescribed_rates_hz", "prescribed_rates_hz_by_channel",
                "rates_hz_by_channel", "rate_hz_by_channel", "applied_rates_hz", "case_rates_hz"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    input_record = row.get("input")
    if isinstance(input_record, Mapping):
        for key in ("rates_hz", "rates_hz_by_channel", "prescribed_rates_hz"):
            value = input_record.get(key)
            if isinstance(value, Mapping):
                return value
    return None


def _schedule_phase(value: Any) -> str:
    name = str(value)
    return {
        "settle": "pre_settle", "settling": "pre_settle",
        "drive": "imposed_drive", "active_drive": "imposed_drive",
        "post_drive_release": "release", "release_after_drive": "release",
    }.get(name, name)


def _rate_for_phase_schedule(spec: Any, phase: str, row: Mapping[str, Any], where: str) -> float:
    """Resolve one channel's phase schedule at a recorded native sample."""
    if spec is None:
        return 0.0
    if not isinstance(spec, Mapping):
        return _finite(spec, where)
    shape = spec.get("shape")
    if shape == "constant":
        return _finite(spec.get("rate_hz"), f"{where}.rate_hz")
    if shape != "linear_rise_then_hold":
        raise TraceError(f"{where}: unsupported schedule shape {shape!r}")
    start = _finite(spec.get("start_rate_hz", 0.0), f"{where}.start_rate_hz")
    target = _finite(spec.get("target_rate_hz"), f"{where}.target_rate_hz")
    rise_s = _finite(spec.get("rise_duration_s"), f"{where}.rise_duration_s")
    time_basis = str(spec.get("time_basis", "phase_time_s"))
    if rise_s <= 0.0:
        raise TraceError(f"{where}.rise_duration_s must be positive")
    if time_basis != "phase_time_s" or row.get("phase_time_s") is None:
        raise TraceError(f"{where} requires command-time phase_time_s")
    elapsed = _finite(row["phase_time_s"], "phase_time_s")
    return start + (target - start) * min(1.0, max(0.0, elapsed / rise_s))


def _declared_case_data(report: Mapping[str, Any] | None,
                        profile: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if report is None:
        return result
    cases = report.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            name = case.get("name", case.get("case", case.get("case_name", case.get("case_id"))))
            if isinstance(name, str):
                result[name] = dict(case)
    single_case = report.get("case")
    if isinstance(single_case, Mapping):
        name = single_case.get("name", single_case.get("case_id", single_case.get("case_name")))
        if isinstance(name, str):
            result[name] = dict(single_case)
    if profile is not None:
        leg = report.get("leg_key", profile.get("selected_leg_at_calibration", "LF"))
        groups = profile.get("drive_groups", ())
        if isinstance(leg, str) and isinstance(groups, (list, tuple)):
            rate = _finite(profile.get("rates_hz_per_selected_exact_channel", 0.0),
                           "profile.rates_hz_per_selected_exact_channel")
            pool_validation = report.get("pool_validation", {})
            pool_counts = pool_validation.get("mapped_motor_count_by_channel", {}) \
                if isinstance(pool_validation, Mapping) else {}
            zero_channels = set(pool_validation.get("zero_channels", ())) \
                if isinstance(pool_validation, Mapping) else set()
            expected_rates = {}
            for group in groups:
                channel = str(group) if str(group).startswith(f"{leg}_") else f"{leg}_{group}"
                count = pool_counts.get(channel)
                available = count is not None and int(count) > 0
                if channel in zero_channels or count == 0:
                    available = False
                elif count is None and not zero_channels:
                    # In old profiles, the calibration pool is exact but the replay
                    # report did not repeat counts; defer to the row's declared map.
                    continue
                expected_rates[channel] = rate if available else 0.0
            for case in result.values():
                if expected_rates:
                    case.setdefault("rate_hz_by_channel", expected_rates)
                case.setdefault("leg_key", leg)
                case.setdefault("active_drive", True)
                if isinstance(pool_validation, Mapping):
                    case.setdefault("pool_validation", dict(pool_validation))
    if isinstance(report.get("leg_key"), str):
        for case in result.values():
            case.setdefault("leg_key", report["leg_key"])
    protocol = report.get("protocol")
    pool_validation = report.get("pool_validation")
    if isinstance(protocol, Mapping) and isinstance(pool_validation, Mapping):
        leg = str(report.get("leg_key", pool_validation.get("leg_key", "LF")))
        rate = protocol.get("exact_rate_hz_per_selected_channel",
                            protocol.get("rates_hz_per_selected_exact_channel"))
        groups = protocol.get("groups", ())
        if rate is not None and isinstance(groups, (list, tuple)):
            rate = _finite(rate, "protocol.exact_rate_hz_per_selected_channel")
            pool_counts = pool_validation.get("mapped_motor_count_by_channel", {})
            zero_channels = set(pool_validation.get("zero_channels", ()))
            available_channels = set(pool_validation.get("available_channels", ()))
            for case_id, case in result.items():
                case.setdefault("leg_key", leg)
                active = case.get("active_drive")
                if active is None:
                    active = "zero_rate" not in case_id.lower()
                    case["active_drive"] = active
                expected: dict[str, float] = {}
                for group_template in groups:
                    channel = str(group_template).replace("<leg>", leg)
                    if not channel.startswith(f"{leg}_"):
                        channel = f"{leg}_{channel}"
                    count = pool_counts.get(channel) if isinstance(pool_counts, Mapping) else None
                    is_available = (count is not None and int(count) > 0) or channel in available_channels
                    if channel in zero_channels or count == 0:
                        is_available = False
                    expected[channel] = rate if active and is_available else 0.0
                case.setdefault("rate_hz_by_channel", expected)
    schedule = protocol.get("drive_rate_schedule") if isinstance(protocol, Mapping) else None
    if not isinstance(schedule, Mapping) and isinstance(profile, Mapping):
        schedule = profile.get("drive_rate_schedule")
    if isinstance(schedule, Mapping):
        for case in result.values():
            case.setdefault("drive_rate_schedule", dict(schedule))
    schedule_by_channel = (
        protocol.get("rate_schedule_by_channel") if isinstance(protocol, Mapping) else None
    )
    if isinstance(schedule_by_channel, Mapping):
        for case in result.values():
            case.setdefault("rate_schedule_by_channel", dict(schedule_by_channel))
    return result


def _phase_durations(report: Mapping[str, Any] | None) -> dict[str, float]:
    if report is None:
        return {}
    candidates: list[Any] = [report.get("phase_seconds")]
    case = report.get("case")
    if isinstance(case, Mapping):
        candidates.append(case.get("phase_seconds"))
    for key in ("protocol", "profile", "timing", "parameters"):
        value = report.get(key)
        if isinstance(value, Mapping):
            candidates.append(value.get("phase_seconds"))
    for value in candidates:
        if isinstance(value, Mapping):
            result = {}
            for name, duration in value.items():
                result[str(name)] = _finite(duration, f"phase_seconds.{name}")
            return result
    return {}


class CaseAccumulator:
    def __init__(self, name: str, expected_case: Mapping[str, Any] | None,
                 *, dt_s: float, gap_m: float, gap_duration_s: float,
                 release_stable_s: float, body_height_m: float,
                 contact_floor_n: float, foot_radius_m: float,
                 phase_durations: Mapping[str, float]) -> None:
        self.name = name
        self.expected_case = dict(expected_case or {})
        self.selected_leg = str(self.expected_case.get("leg_key", "LF"))
        if self.selected_leg not in LEGS:
            self.errors = [f"invalid selected leg {self.selected_leg!r}"]
        else:
            self.errors: list[str] = []
        self.dt_s = dt_s
        self.gap_m = gap_m
        self.gap_ticks_required = max(1, math.ceil(gap_duration_s / dt_s - 1e-10))
        self.release_ticks_required = max(1, math.ceil(release_stable_s / dt_s - 1e-10))
        self.body_height_threshold_m = body_height_m
        self.contact_floor_n = contact_floor_n
        self.foot_radius_m = foot_radius_m
        self.phase_durations = phase_durations
        self.phase_counts: dict[str, int] = {}
        self.phase_tick_last: dict[str, int] = {}
        self.phase_order: list[str] = []
        self.rows = 0
        self.previous_time: float | None = None
        self.previous_phase_time: float | None = None
        self.previous_state_phase_time: float | None = None
        self.previous_phase: str | None = None
        self.previous_tick: int | None = None
        self.time_step_samples = 0
        self.time_step_max_error_s = 0.0
        self.min_body_height_m = math.inf
        self.nonfoot_contact_samples = 0
        self.root_wrench_nonzero_samples = 0
        self.root_wrench_unavailable_samples = 0
        self.missing_feet: set[str] = set()
        self.support_failure_samples = 0
        self.support_min_normal_force_n = {leg: math.inf for leg in LEGS if leg != self.selected_leg}
        self.support_contact_ticks = {leg: 0 for leg in LEGS if leg != self.selected_leg}
        self.support_load_ticks = {leg: 0 for leg in LEGS if leg != self.selected_leg}
        self.support_failure_ticks = {leg: 0 for leg in LEGS if leg != self.selected_leg}
        self.lf_min_normal_force_drive_n = math.inf
        self.drive_rows = 0
        self.drive_support_ok_rows = 0
        self.drive_lf_support_ok_rows = 0
        self.drive_max_gap_m = 0.0
        self.drive_gap_ticks = 0
        self.drive_gap_best_ticks = 0
        self.drive_gap_best_min_m = math.inf
        self.drive_gap_active_start_s: float | None = None
        self.drive_gap_best_start_s: float | None = None
        self.drive_gap_best_end_s: float | None = None
        self.drive_gap_best_support_ok = True
        self.drive_gap_best_min_support_n = {leg: math.inf for leg in LEGS if leg != self.selected_leg}
        self.release_tail_ticks = 0
        self.release_tail_min_load_n = {leg: math.inf for leg in LEGS}
        self.release_rows = 0
        self.rate_rows = 0
        self.rate_missing_rows = 0
        self.rate_mismatch_rows = 0
        self.rate_max_abs_error_hz = 0.0
        self.rate_first_mismatch: str | None = None
        self.observed_rate_channels: set[str] = set()

    def _check_rate_schedule(self, row: Mapping[str, Any], phase: str) -> None:
        actual = _rates(row)
        if actual is None:
            self.rate_missing_rows += 1
            return
        self.rate_rows += 1
        normalized: dict[str, float] = {}
        for channel, raw_value in actual.items():
            normalized[str(channel)] = _finite(raw_value, f"input_rates_hz.{channel}")
            self.observed_rate_channels.add(str(channel))

        channel_schedule = self.expected_case.get("rate_schedule_by_channel")
        if isinstance(channel_schedule, Mapping):
            expected_now: dict[str, float] = {}
            try:
                for channel, phase_map in channel_schedule.items():
                    if not isinstance(phase_map, Mapping):
                        self.rate_mismatch_rows += 1
                        self.rate_first_mismatch = self.rate_first_mismatch or (
                            f"rate_schedule_by_channel.{channel} must map phases to rates"
                        )
                        return
                    phase_specs = {
                        _schedule_phase(name): value for name, value in phase_map.items()
                    }
                    rate = _rate_for_phase_schedule(
                        phase_specs.get(phase), phase, row,
                        f"rate_schedule_by_channel.{channel}.{phase}",
                    )
                    if rate != 0.0:
                        expected_now[str(channel)] = rate
            except TraceError as exc:
                self.rate_mismatch_rows += 1
                self.rate_first_mismatch = self.rate_first_mismatch or str(exc)
                return
            actual_nonzero = {key: value for key, value in normalized.items() if value != 0.0}
            keys_match = set(actual_nonzero) == set(expected_now)
            values_match = keys_match and all(
                math.isclose(actual_nonzero[key], expected_now[key], rel_tol=0.0, abs_tol=1e-8)
                for key in expected_now
            )
            if keys_match:
                self.rate_max_abs_error_hz = max(
                    self.rate_max_abs_error_hz,
                    max((abs(actual_nonzero[key] - expected_now[key]) for key in expected_now), default=0.0),
                )
            if not values_match:
                self.rate_mismatch_rows += 1
                self.rate_first_mismatch = self.rate_first_mismatch or (
                    f"{phase}: observed nonzero rates {actual_nonzero!r} "
                    f"do not match declared {expected_now!r}"
                )
            return

        declared = self.expected_case.get("rate_hz_by_channel",
                                         self.expected_case.get("rates_hz_by_channel"))
        if declared is None:
            declared = self.expected_case.get("prescribed_rate_hz_by_channel")
        if declared is None and isinstance(row.get("case_rate_hz_by_channel"), Mapping):
            declared = row["case_rate_hz_by_channel"]
        if not isinstance(declared, Mapping):
            self.rate_mismatch_rows += 1
            if self.rate_first_mismatch is None:
                self.rate_first_mismatch = "case input schedule is not declared in report or trace"
            return
        expected_now = {
            str(key): _finite(value, f"declared_rates.{key}")
            for key, value in declared.items()
        }
        schedule = self.expected_case.get("drive_rate_schedule")
        if phase != "imposed_drive" or self.expected_case.get("active_drive") is False:
            expected_now = {}
        elif isinstance(schedule, Mapping):
            schedule_phase = str(schedule.get("phase", "imposed_drive"))
            if schedule_phase in {"drive", "active_drive"}:
                schedule_phase = "imposed_drive"
            if phase != schedule_phase:
                expected_now = {}
                schedule = None
        if isinstance(schedule, Mapping) and phase == "imposed_drive":
            shape = schedule.get("shape")
            if shape != "linear_rise_then_hold":
                self.rate_mismatch_rows += 1
                if self.rate_first_mismatch is None:
                    self.rate_first_mismatch = f"unsupported declared drive-rate schedule shape: {shape!r}"
                return
            start_rate = _finite(schedule.get("start_rate_hz", 0.0), "drive_rate_schedule.start_rate_hz")
            target_rate = _finite(schedule.get("target_rate_hz", 0.0), "drive_rate_schedule.target_rate_hz")
            rise_s = _finite(schedule.get("rise_duration_s"), "drive_rate_schedule.rise_duration_s")
            if rise_s <= 0.0 or target_rate <= 0.0:
                self.rate_mismatch_rows += 1
                if self.rate_first_mismatch is None:
                    self.rate_first_mismatch = "drive-rate schedule needs positive rise duration and target"
                return
            declared_targets = [value for value in expected_now.values() if value > 0.0]
            if any(not math.isclose(value, target_rate, rel_tol=0.0, abs_tol=1e-9)
                   for value in declared_targets):
                self.rate_mismatch_rows += 1
                if self.rate_first_mismatch is None:
                    self.rate_first_mismatch = (
                        "drive-rate schedule target does not match declared channel rate map"
                    )
                return
            time_basis = str(schedule.get("time_basis", "phase_time_s"))
            if time_basis != "phase_time_s" or row.get("phase_time_s") is None:
                self.rate_mismatch_rows += 1
                if self.rate_first_mismatch is None:
                    self.rate_first_mismatch = "ramped schedule requires phase_time_s"
                return
            elapsed_s = _finite(row["phase_time_s"], "phase_time_s")
            rate_now = start_rate + (target_rate - start_rate) * min(1.0, max(0.0, elapsed_s / rise_s))
            expected_now = {
                key: value * (rate_now / target_rate)
                for key, value in expected_now.items()
            }
        expected_nonzero = {key: value for key, value in expected_now.items() if value != 0.0}
        actual_nonzero = {key: value for key, value in normalized.items() if value != 0.0}
        keys_match = set(actual_nonzero) == set(expected_nonzero)
        values_match = keys_match and all(
            math.isclose(actual_nonzero[key], expected_nonzero[key], rel_tol=0.0, abs_tol=1e-8)
            for key in expected_nonzero
        )
        if keys_match:
            self.rate_max_abs_error_hz = max(
                self.rate_max_abs_error_hz,
                max((abs(actual_nonzero[key] - expected_nonzero[key]) for key in expected_nonzero), default=0.0),
            )
        if not values_match:
            self.rate_mismatch_rows += 1
            if self.rate_first_mismatch is None:
                self.rate_first_mismatch = (
                    f"{phase}: observed nonzero rates {actual_nonzero!r} "
                    f"do not match declared {expected_now!r}"
                )

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows += 1
        row_leg = row.get("leg_key")
        if row_leg is not None:
            row_leg = str(row_leg)
            if row_leg not in LEGS:
                self.errors.append(f"invalid selected leg {row_leg!r} at row {self.rows}")
            elif self.rows == 1:
                if "leg_key" not in self.expected_case:
                    self.selected_leg = row_leg
                    self.support_min_normal_force_n = {
                        leg: math.inf for leg in LEGS if leg != self.selected_leg
                    }
                    self.support_contact_ticks = {leg: 0 for leg in LEGS if leg != self.selected_leg}
                    self.support_load_ticks = {leg: 0 for leg in LEGS if leg != self.selected_leg}
                    self.support_failure_ticks = {leg: 0 for leg in LEGS if leg != self.selected_leg}
                    self.drive_gap_best_min_support_n = {
                        leg: math.inf for leg in LEGS if leg != self.selected_leg
                    }
            elif row_leg != self.selected_leg:
                self.errors.append(f"selected leg changes from {self.selected_leg} to {row_leg} at row {self.rows}")
        phase = _phase_name(row)
        time_s = _finite(row.get("time_s", row.get("native_time_s")), "time_s")
        phase_time_raw = row.get("phase_time_s")
        phase_time_s = _finite(phase_time_raw, "phase_time_s") if phase_time_raw is not None else None
        if phase not in self.phase_counts:
            self.phase_counts[phase] = 0
            self.phase_order.append(phase)
            self.drive_gap_ticks = 0
        self.phase_counts[phase] += 1

        tick_raw = row.get("tick", row.get("phase_tick", row.get("tick_in_phase")))
        tick = int(tick_raw) if tick_raw is not None else None
        if self.previous_time is not None:
            delta = time_s - self.previous_time
            error = abs(delta - self.dt_s)
            self.time_step_samples += 1
            self.time_step_max_error_s = max(self.time_step_max_error_s, error)
            if error > max(1e-10, self.dt_s * 1e-5):
                self.errors.append(f"native clock discontinuity before row {self.rows}: dt={delta:.12g}s")
        elif abs(time_s - self.dt_s) > max(1e-10, self.dt_s * 1e-5):
            self.errors.append(f"native clock starts at {time_s:.12g}s rather than one native step")
        if phase_time_s is not None:
            if phase != self.previous_phase:
                if min(abs(phase_time_s), abs(phase_time_s - self.dt_s)) > max(1e-10, self.dt_s * 1e-5):
                    self.errors.append(f"phase {phase!r} clock does not start at zero or one native step")
            elif self.previous_phase_time is not None:
                phase_delta = phase_time_s - self.previous_phase_time
                if abs(phase_delta - self.dt_s) > max(1e-10, self.dt_s * 1e-5):
                    self.errors.append(f"phase {phase!r} clock discontinuity before row {self.rows}")
        elif self.rows == 1 and row.get("schema") == "leg-capacity-trace-sample-1":
            self.errors.append("phase_time_s missing from native-capacity trace")
        elif self.previous_phase_time is not None and phase == self.previous_phase:
            self.errors.append(f"phase_time_s missing during phase {phase!r}")
        state_phase_raw = row.get("state_phase_time_s")
        state_phase_time_s = _finite(state_phase_raw, "state_phase_time_s") if state_phase_raw is not None else None
        if state_phase_time_s is not None:
            if phase != self.previous_phase:
                if abs(state_phase_time_s - self.dt_s) > max(1e-10, self.dt_s * 1e-5):
                    self.errors.append(f"phase {phase!r} state clock does not start at one native step")
            elif self.previous_state_phase_time is not None:
                state_delta = state_phase_time_s - self.previous_state_phase_time
                if abs(state_delta - self.dt_s) > max(1e-10, self.dt_s * 1e-5):
                    self.errors.append(f"phase {phase!r} state clock discontinuity before row {self.rows}")
        if phase == self.previous_phase and tick is not None and self.previous_tick is not None:
            if tick != self.previous_tick + 1:
                self.errors.append(f"phase {phase!r} tick discontinuity at row {self.rows}")
        elif phase != self.previous_phase and tick is not None and tick not in (0, 1):
            self.errors.append(f"phase {phase!r} starts at unexpected tick {tick}")
        self.previous_time, self.previous_phase, self.previous_tick = time_s, phase, tick
        self.previous_phase_time = phase_time_s
        self.previous_state_phase_time = state_phase_time_s
        if phase in self.phase_tick_last and tick is not None and self.phase_tick_last[phase] + 1 != tick:
            self.errors.append(f"phase {phase!r} has nonconsecutive native ticks")
        if tick is not None:
            self.phase_tick_last[phase] = tick

        self._check_rate_schedule(row, phase)
        feet = _feet(row)
        missing = set(LEGS) - set(feet)
        if missing:
            self.missing_feet.update(missing)

        root_position = _root_position(row)
        body_height = _root_height(row, root_position)
        if body_height is None:
            self.errors.append(f"root position missing at row {self.rows}")
        else:
            self.min_body_height_m = min(self.min_body_height_m, body_height)

        nonfoot = row.get("nonfoot_ground_contact", row.get("body_ground_contact"))
        if nonfoot is None:
            self.errors.append(f"nonfoot contact state missing at row {self.rows}")
        elif bool(nonfoot):
            self.nonfoot_contact_samples += 1

        wrench_peak = _root_wrench_peak(row)
        if wrench_peak is None:
            self.root_wrench_unavailable_samples += 1
        elif wrench_peak != 0.0:
            self.root_wrench_nonzero_samples += 1

        drive = phase == "imposed_drive"
        if drive:
            self.drive_rows += 1
            gap = 0.0
            selected_foot = feet.get(self.selected_leg)
            if selected_foot is not None:
                contact = selected_foot.get("contact")
                normal = selected_foot.get("normal_force_N")
                if normal is not None:
                    normal_n = _finite(normal, f"feet.{self.selected_leg}.normal_force_N")
                    self.lf_min_normal_force_drive_n = min(self.lf_min_normal_force_drive_n, normal_n)
                    if contact is True and normal_n > self.contact_floor_n:
                        self.drive_lf_support_ok_rows += 1
                if contact is False:
                    clearance = row.get("selected_leg_clearance_m", selected_foot.get("clearance_m"))
                    if clearance is not None:
                        gap = max(0.0, _finite(clearance, f"feet.{self.selected_leg}.clearance_m"))
                    elif selected_foot.get("center_z_m") is not None:
                        gap = max(0.0, _finite(selected_foot["center_z_m"],
                                               f"feet.{self.selected_leg}.center_z_m") - self.foot_radius_m)
                    elif root_position is not None:
                        # Legacy LF-only telemetry stores selected-foot geometry separately.
                        lf_detail = row.get("LF_foot") if self.selected_leg == "LF" else None
                        if isinstance(lf_detail, Mapping) and lf_detail.get("center_z_m") is not None:
                            gap = max(0.0, _finite(lf_detail["center_z_m"], "LF_foot.center_z_m") - self.foot_radius_m)
            self.drive_max_gap_m = max(self.drive_max_gap_m, gap)
            support_ok = True
            for leg in LEGS:
                if leg == self.selected_leg:
                    continue
                foot = feet.get(leg)
                if foot is None or foot.get("contact") is not True or foot.get("normal_force_N") is None:
                    support_ok = False
                    self.support_failure_samples += 1
                    self.support_failure_ticks[leg] += 1
                    continue
                self.support_contact_ticks[leg] += 1
                force = _finite(foot["normal_force_N"], f"feet.{leg}.normal_force_N")
                self.support_min_normal_force_n[leg] = min(self.support_min_normal_force_n[leg], force)
                if force <= self.contact_floor_n:
                    support_ok = False
                    self.support_failure_samples += 1
                    self.support_failure_ticks[leg] += 1
                else:
                    self.support_load_ticks[leg] += 1
            self.drive_support_ok_rows += int(support_ok)
            if gap >= self.gap_m:
                if self.drive_gap_ticks == 0:
                    self.drive_gap_active_start_s = time_s
                    self.drive_gap_best_min_m = math.inf
                    self.drive_gap_best_support_ok = True
                    self.drive_gap_best_min_support_n = {
                        leg: math.inf for leg in LEGS if leg != self.selected_leg
                    }
                self.drive_gap_ticks += 1
                self.drive_gap_best_min_m = min(self.drive_gap_best_min_m, gap)
                self.drive_gap_best_support_ok &= support_ok
                for leg in LEGS:
                    if leg != self.selected_leg and feet.get(leg, {}).get("normal_force_N") is not None:
                        force = _finite(feet[leg]["normal_force_N"], f"feet.{leg}.normal_force_N")
                        self.drive_gap_best_min_support_n[leg] = min(self.drive_gap_best_min_support_n[leg], force)
                if self.drive_gap_ticks > self.drive_gap_best_ticks:
                    self.drive_gap_best_ticks = self.drive_gap_ticks
                    self.drive_gap_best_start_s = self.drive_gap_active_start_s
                    self.drive_gap_best_end_s = time_s
            else:
                self.drive_gap_ticks = 0
                self.drive_gap_active_start_s = None

        if phase == "release":
            self.release_rows += 1
            stable = True
            for leg in LEGS:
                foot = feet.get(leg)
                if foot is None or foot.get("contact") is not True or foot.get("normal_force_N") is None:
                    stable = False
                    continue
                force = _finite(foot["normal_force_N"], f"feet.{leg}.normal_force_N")
                self.release_tail_min_load_n[leg] = min(self.release_tail_min_load_n[leg], force)
                stable &= force > self.contact_floor_n
            if stable:
                self.release_tail_ticks += 1
            else:
                self.release_tail_ticks = 0
                self.release_tail_min_load_n = {leg: math.inf for leg in LEGS}

    def finish(self) -> dict[str, Any]:
        expected_phases = ["pre_settle", "imposed_drive", "release"]
        phase_sequence_ok = self.phase_order == expected_phases
        for phase, expected_duration in self.phase_durations.items():
            normalized = {"settle": "pre_settle", "drive": "imposed_drive"}.get(phase, phase)
            count = self.phase_counts.get(normalized)
            if count is not None and not math.isclose(count * self.dt_s, expected_duration,
                                                      rel_tol=0.0, abs_tol=self.dt_s * 1.01):
                self.errors.append(
                    f"phase {normalized!r} duration {count * self.dt_s:.9g}s "
                    f"differs from declared {expected_duration:.9g}s"
                )
        clock_ok = (self.rows > 0 and self.time_step_samples == self.rows - 1
                    and self.time_step_max_error_s <= max(1e-10, self.dt_s * 1e-5)
                    and phase_sequence_ok and not self.errors)
        heights_ok = (self.min_body_height_m != math.inf
                      and self.min_body_height_m > self.body_height_threshold_m)
        root_wrench_ok = (self.root_wrench_unavailable_samples == 0
                          and self.root_wrench_nonzero_samples == 0)
        nonfoot_ok = self.nonfoot_contact_samples == 0
        feet_complete = not self.missing_feet
        support_ok = (feet_complete and self.drive_rows > 0
                      and self.drive_support_ok_rows == self.drive_rows)
        gap_duration_s = self.drive_gap_best_ticks * self.dt_s
        lift_ok = (self.drive_gap_best_ticks >= self.gap_ticks_required
                   and self.drive_gap_best_min_m >= self.gap_m
                   and self.drive_gap_best_support_ok)
        declared_role = self.expected_case.get("role", self.expected_case.get("classification"))
        role = str(declared_role).lower() if declared_role is not None else ""
        if (role in {"sham", "matched_sham", "support_sham", "control_sham"}
                or self.expected_case.get("active_drive") is False):
            is_sham = True
        elif role in {"candidate", "active", "heldout", "held_out", "lift_candidate"}:
            is_sham = False
        else:
            declared_rates = self.expected_case.get(
                "rate_hz_by_channel", self.expected_case.get("rates_hz_by_channel")
            )
            is_sham = isinstance(declared_rates, Mapping) and bool(declared_rates) and not any(
                _finite(value, f"declared_rates.{key}") != 0.0
                for key, value in declared_rates.items()
            )
        release_ok = (self.release_rows > 0
                      and self.release_tail_ticks >= self.release_ticks_required
                      and all(self.release_tail_min_load_n[leg] > self.contact_floor_n
                              for leg in LEGS))
        rates_ok = (self.rate_rows == self.rows and self.rate_mismatch_rows == 0)
        checks = {
            "native_clock_continuous": clock_ok,
            "prescribed_rates_recorded_and_match": rates_ok,
            "all_six_foot_channels_present": feet_complete,
            "other_five_load_bearing_through_drive": support_ok,
            "no_nonfoot_ground_contact": nonfoot_ok,
            "root_applied_wrench_exactly_zero": root_wrench_ok,
            "body_height_above_threshold": heights_ok,
            "all_feet_stably_loaded_at_release_end": release_ok,
        }
        if is_sham:
            checks[f"{self.selected_leg}_remains_contacting_and_loaded_through_sham_drive"] = (
                self.drive_rows > 0 and self.drive_lf_support_ok_rows == self.drive_rows
            )
            checks[f"{self.selected_leg}_has_no_sustained_clearance_event"] = not lift_ok
        else:
            checks[f"{self.selected_leg}_clearance_at_least_threshold_for_contiguous_duration"] = lift_ok
        required = bool(self.expected_case.get("required_for_release", True))
        case_role = "sham" if is_sham else (
            "screening" if not required else (declared_role or "candidate")
        )
        return {
            "case": self.name,
            "role": case_role,
            "required_for_release": required,
            "accepted": all(checks.values()),
            "checks": checks,
            "metrics": {
                "rows": self.rows,
                "phase_rows": self.phase_counts,
                "phase_order": self.phase_order,
                "drive_rows": self.drive_rows,
                "drive_support_ok_rows": self.drive_support_ok_rows,
                "drive_support_failure_samples": self.support_failure_samples,
                "drive_LF_contact_and_loaded_rows": self.drive_lf_support_ok_rows,
                "other_five_min_normal_force_N": {
                    leg: None if value == math.inf else value
                    for leg, value in self.support_min_normal_force_n.items()
                },
                "other_five_drive_contact_fraction": {
                    leg: ticks / self.drive_rows if self.drive_rows else None
                    for leg, ticks in self.support_contact_ticks.items()
                },
                "other_five_drive_loadbearing_fraction": {
                    leg: ticks / self.drive_rows if self.drive_rows else None
                    for leg, ticks in self.support_load_ticks.items()
                },
                "other_five_drive_support_failure_samples_by_leg": self.support_failure_ticks,
                "selected_leg": self.selected_leg,
                "selected_leg_drive_min_normal_force_N": None if self.lf_min_normal_force_drive_n == math.inf
                else self.lf_min_normal_force_drive_n,
                "selected_leg_drive_max_gap_m": self.drive_max_gap_m,
                "selected_leg_longest_contiguous_gap_s": gap_duration_s,
                "selected_leg_longest_gap_interval_start_time_s": self.drive_gap_best_start_s,
                "selected_leg_longest_gap_interval_end_time_s": self.drive_gap_best_end_s,
                "selected_leg_min_gap_during_longest_interval_m": None
                if self.drive_gap_best_min_m == math.inf else self.drive_gap_best_min_m,
                "support_min_normal_force_during_longest_gap_N": {
                    leg: None if value == math.inf else value
                    for leg, value in self.drive_gap_best_min_support_n.items()
                },
                "release_stable_contact_duration_s": self.release_tail_ticks * self.dt_s,
                "release_tail_min_normal_force_N": {
                    leg: None if value == math.inf else value
                    for leg, value in self.release_tail_min_load_n.items()
                },
                "minimum_root_height_m": None if self.min_body_height_m == math.inf
                else self.min_body_height_m,
                "nonfoot_contact_samples": self.nonfoot_contact_samples,
                "root_wrench_nonzero_samples": self.root_wrench_nonzero_samples,
                "root_wrench_unavailable_samples": self.root_wrench_unavailable_samples,
                "clock_max_step_error_s": self.time_step_max_error_s,
                "missing_foot_keys": sorted(self.missing_feet),
                "rate_rows": self.rate_rows,
                "rate_rows_missing": self.rate_missing_rows,
                "rate_mismatch_rows": self.rate_mismatch_rows,
                "rate_max_abs_error_hz": self.rate_max_abs_error_hz,
                "rate_mismatch_example": self.rate_first_mismatch,
                "observed_rate_channels": sorted(self.observed_rate_channels),
                "declared_drive_rate_schedule": self.expected_case.get("drive_rate_schedule"),
                "declared_rate_schedule_by_channel": self.expected_case.get("rate_schedule_by_channel"),
                "declared_motor_pool": self.expected_case.get("pool_validation"),
            },
            "errors": self.errors,
        }


def _read_report(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot read report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceError("report root must be a JSON object")
    return value, _sha256(path)


def _read_linked_profile(report: Mapping[str, Any] | None,
                         report_path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if report is None or report_path is None:
        return None, None
    linked = report.get("profile_path")
    expected_hash = report.get("profile_sha256")
    if not isinstance(linked, str):
        candidate = report.get("candidate_profile")
        if isinstance(candidate, Mapping):
            linked = candidate.get("path")
            expected_hash = candidate.get("sha256")
    if not isinstance(linked, str):
        return None, None
    linked_path = Path(linked)
    if not linked_path.is_absolute():
        linked_path = report_path.parent / linked_path
    if not linked_path.is_file():
        return None, None
    try:
        profile = json.loads(linked_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot read linked candidate profile {linked_path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise TraceError("linked candidate profile root must be a JSON object")
    actual_hash = _sha256(linked_path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise TraceError(f"linked candidate profile hash mismatch: {linked_path}")
    profile["_resolved_path"] = str(linked_path.resolve())
    return profile, actual_hash


CORE_SOURCE_PATHS = (
    "scripts/articulated_leg_body.py",
    "scripts/leg_muscle_dynamics.py",
    "scripts/leg_muscle_parameters.py",
    "scripts/leg_muscle_pipeline.py",
    "scripts/leg_muscle_decoder.py",
    "scripts/neuromuscular_adapter.py",
    "config/articulated_leg_body.json",
    "config/leg_muscle_parameters.json",
    "config/leg_muscle_map.json",
)
CORE_CONFIG_FIELDS = (
    "body_config", "muscle_parameter_config", "motor_map",
)


def _hash_value_map(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    inputs = report.get("input_hashes", {})
    inputs = inputs if isinstance(inputs, Mapping) else {}
    sources = inputs.get("project_sources", report.get("source_hashes", {}))
    sources = {str(k): str(v) for k, v in sources.items()} if isinstance(sources, Mapping) else {}
    configs = {}
    for key in CORE_CONFIG_FIELDS:
        value = inputs.get(key)
        if value is not None:
            configs[key] = str(value)
    extra = report.get("original_config_hashes")
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            configs[str(key)] = str(value)
    # Some probe reports place the unchanged source configurations in the
    # source digest map while input_hashes contains only the generated,
    # per-case configuration. Compare the former against the reference run.
    for field, source_path in {
        "body_config": "config/articulated_leg_body.json",
        "muscle_parameter_config": "config/leg_muscle_parameters.json",
        "motor_map": "config/leg_muscle_map.json",
    }.items():
        if field not in configs and source_path in sources:
            configs[field] = sources[source_path]
    return sources, configs


def _compare_reference_hashes(report: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    current_sources, current_configs = _hash_value_map(report)
    reference_sources, reference_configs = _hash_value_map(reference)
    comparisons: dict[str, dict[str, Any]] = {}
    for name in CORE_SOURCE_PATHS:
        current = current_sources.get(name)
        baseline = reference_sources.get(name)
        comparisons[name] = {
            "current_sha256": current, "reference_sha256": baseline,
            "match": current is not None and baseline is not None and current == baseline,
        }
    for name in CORE_CONFIG_FIELDS:
        current = current_configs.get(name)
        baseline = reference_configs.get(name)
        comparisons[name] = {
            "current_sha256": current, "reference_sha256": baseline,
            "match": current is not None and baseline is not None and current == baseline,
        }
    return {
        "verified": bool(comparisons) and all(item["match"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def _verify_report_file_hashes(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    expected_files: dict[str, str] = {}
    case_config_paths: dict[str, str] = {}
    inputs = report.get("input_hashes", {})
    inputs = inputs if isinstance(inputs, Mapping) else {}

    for source_key in ("project_sources", "source_files", "files"):
        values = inputs.get(source_key)
        if isinstance(values, Mapping):
            expected_files.update({str(k): str(v) for k, v in values.items()})
    for source_key in ("source_hashes", "source_sha256"):
        values = report.get(source_key)
        if isinstance(values, Mapping):
            expected_files.update({str(k): str(v) for k, v in values.items()})
    config_source_paths = {
        "body_config": "config/articulated_leg_body.json",
        "muscle_parameter_config": "config/leg_muscle_parameters.json",
        "motor_map": "config/leg_muscle_map.json",
    }
    for key, source_path in config_source_paths.items():
        if inputs.get(key) is not None:
            expected_files[source_path] = str(inputs[key])
    original_configs = report.get("original_config_hashes")
    if isinstance(original_configs, Mapping):
        expected_files.update({str(k): str(v) for k, v in original_configs.items()})

    snapshot = report.get("source_snapshot")
    if isinstance(snapshot, Mapping):
        for name, entry in snapshot.items():
            if isinstance(entry, Mapping) and entry.get("sha256") is not None:
                copy_path = entry.get("copy", str(name).replace("/", "__"))
                expected_files[str(copy_path)] = str(entry["sha256"])

    source_copies = report.get("source_copy_paths")
    source_hashes = report.get("source_hashes", {})
    if isinstance(source_copies, Mapping) and isinstance(source_hashes, Mapping):
        for name, copy_path in source_copies.items():
            digest = source_hashes.get(name)
            if isinstance(copy_path, str) and isinstance(digest, str):
                expected_files[copy_path] = digest

    config_paths = report.get("config_paths")
    if isinstance(config_paths, Mapping):
        for name, entry in config_paths.items():
            if isinstance(entry, Mapping):
                path = entry.get("path")
                digest = entry.get("sha256")
                if isinstance(path, str) and isinstance(digest, str):
                    expected_files[path] = digest
                    case_config_paths[path] = digest

    generated = report.get("generated_config_sha256")
    if isinstance(generated, Mapping):
        expected_files.update({str(k): str(v) for k, v in generated.items()})
        case_config_paths.update({str(k): str(v) for k, v in generated.items()})

    case_list = report.get("cases", [])
    if isinstance(report.get("case"), Mapping):
        case_list = [*case_list, report["case"]] if isinstance(case_list, list) else [report["case"]]
    if isinstance(case_list, list):
        for case in case_list:
            if not isinstance(case, Mapping):
                continue
            for stem in ("muscle_config", "body_config"):
                path = case.get(f"{stem}_relative_path", case.get(f"{stem}_path"))
                digest = case.get(f"{stem}_sha256")
                if isinstance(path, str) and isinstance(digest, str):
                    expected_files[path] = digest
                    case_config_paths[path] = digest
            configs = case.get("configs")
            if isinstance(configs, Mapping):
                for entry in configs.values():
                    if isinstance(entry, Mapping):
                        path, digest = entry.get("path"), entry.get("sha256")
                        if isinstance(path, str) and isinstance(digest, str):
                            expected_files[path] = digest
                            case_config_paths[path] = digest
            paths = case.get("config_paths")
            if isinstance(paths, Mapping):
                for name, path in paths.items():
                    if not isinstance(path, str):
                        continue
                    entry = configs.get(name) if isinstance(configs, Mapping) else None
                    digest = entry.get("sha256") if isinstance(entry, Mapping) else None
                    if isinstance(digest, str):
                        expected_files[path] = digest
                        case_config_paths[path] = digest

    checked: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for declared_path, expected_digest in expected_files.items():
        raw_path = Path(declared_path)
        candidates = [raw_path] if raw_path.is_absolute() else [
            report_path.parent / raw_path,
            report_path.parent / "sources" / declared_path.replace("/", "__"),
        ]
        found = next((path for path in candidates if path.is_file()), None)
        if found is None:
            unresolved.append(declared_path)
            checked.append({"path": declared_path, "expected_sha256": expected_digest,
                            "actual_sha256": None, "match": False})
            continue
        actual = _sha256(found)
        checked.append({"path": str(found.resolve()), "expected_sha256": expected_digest,
                        "actual_sha256": actual, "match": actual == expected_digest})

    checked_by_declared = {
        declared_path: row for declared_path, row in zip(expected_files, checked)
    }
    case_config_rows = [checked_by_declared.get(path) for path in case_config_paths]
    case_configs_verified = bool(case_config_paths) and all(
        row is not None and row["match"]
        and row["expected_sha256"] == case_config_paths[path]
        for path, row in zip(case_config_paths, case_config_rows)
    )
    return {
        "verified": bool(checked) and not unresolved and all(row["match"] for row in checked),
        "case_configs_verified": case_configs_verified,
        "case_config_hashes_declared": len(case_config_paths),
        "files_checked": len(checked),
        "unresolved_paths": unresolved,
        "mismatches": [row for row in checked if not row["match"]],
        "files": checked,
    }


def validate_trace(
    trace_path: Path,
    *,
    report_path: Path | None = None,
    reference_report_path: Path | None = None,
    dt_s: float = DEFAULT_DT_S,
    gap_m: float = DEFAULT_GAP_M,
    gap_duration_s: float = DEFAULT_GAP_DURATION_S,
    release_stable_s: float = DEFAULT_RELEASE_STABLE_S,
    min_body_height_m: float = DEFAULT_MIN_BODY_HEIGHT_M,
    contact_floor_n: float = 0.0,
    foot_radius_m: float = DEFAULT_FOOT_RADIUS_M,
) -> dict[str, Any]:
    report, report_hash = _read_report(report_path)
    reference_report, reference_report_hash = _read_report(reference_report_path)
    provenance: dict[str, Any] | None = None
    if reference_report_path is not None:
        if report is None or report_path is None or reference_report is None:
            raise TraceError("--reference-report requires both readable report files")
        current_files = _verify_report_file_hashes(report, report_path)
        source_comparison = _compare_reference_hashes(report, reference_report)
        embedded_reference = report.get("reference_temporal_ramp")
        embedded_hash = (embedded_reference.get("report_sha256")
                         if isinstance(embedded_reference, Mapping) else None)
        reference_link_ok = embedded_hash is None or embedded_hash == reference_report_hash
        provenance = {
            "verified": bool(current_files["verified"]
                              and source_comparison["verified"]
                              and reference_link_ok),
            "current_report_file_hashes": current_files,
            "unchanged_core_hashes_match_reference": source_comparison,
            "embedded_reference_report_sha256": embedded_hash,
            "reference_report_sha256": reference_report_hash,
            "reference_report_hash_matches_embedded_link": reference_link_ok,
        }
    profile, profile_hash = _read_linked_profile(report, report_path)
    case_data = _declared_case_data(report, profile)
    phase_durations = _phase_durations(report) or _phase_durations(profile)
    candidate_case_id = report.get("candidate_case_id") if report else None
    replay_case_id = None
    if report and report.get("schema") == "leg-capacity-profile-replay-1":
        single_case = report.get("case")
        if isinstance(single_case, Mapping):
            replay_case_id = single_case.get("case_id")
    for case_id, details in case_data.items():
        role = str(details.get("role", details.get("classification", ""))).lower()
        matched_sham = (details.get("active_drive") is False or role in {
            "sham", "matched_sham", "support_sham", "control_sham"
        } or "zero_rate" in str(details.get("profile_kind", "")))
        declared_candidate = role in {
            "candidate", "active", "heldout", "held_out", "lift_candidate"
        }
        details["required_for_release"] = bool(
            matched_sham or declared_candidate
            or case_id == candidate_case_id or case_id == replay_case_id
        )
    try:
        trace_hash = _sha256(trace_path)
    except OSError as exc:
        raise TraceError(f"cannot read trace {trace_path}: {exc}") from exc
    trajectory_meta = report.get("trajectory") if report else None
    reported_trace_hash = trajectory_meta.get("sha256") if isinstance(trajectory_meta, Mapping) else None
    trace_hash_matches_report = (
        isinstance(reported_trace_hash, str) and reported_trace_hash == trace_hash
    )
    accumulators: dict[str, CaseAccumulator] = {}
    try:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TraceError(f"{trace_path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(row, Mapping):
                    raise TraceError(f"{trace_path}:{line_no}: trace row is not an object")
                name = _case_name(row)
                accumulator = accumulators.get(name)
                if accumulator is None:
                    accumulator = CaseAccumulator(
                        name, case_data.get(name), dt_s=dt_s, gap_m=gap_m,
                        gap_duration_s=gap_duration_s, release_stable_s=release_stable_s,
                        body_height_m=min_body_height_m, contact_floor_n=contact_floor_n,
                        foot_radius_m=foot_radius_m, phase_durations=phase_durations,
                    )
                    accumulators[name] = accumulator
                try:
                    accumulator.add(row)
                except TraceError as exc:
                    accumulator.errors.append(f"{trace_path}:{line_no}: {exc}")
    except OSError as exc:
        raise TraceError(f"cannot read trace {trace_path}: {exc}") from exc
    case_results = [acc.finish() for acc in accumulators.values()]
    if provenance is not None:
        for case in case_results:
            case["checks"]["source_and_config_hashes_verified"] = provenance["verified"]
            case["accepted"] = all(case["checks"].values())
    required_cases = [case for case in case_results if case["required_for_release"]]
    candidate_required = any(
        case["required_for_release"] and case["role"] != "sham" for case in case_results
    )
    overall_clock = all(c["checks"]["native_clock_continuous"] for c in case_results)
    accepted = trace_hash_matches_report and bool(required_cases) and candidate_required and all(
        case["accepted"] for case in required_cases
    )
    no_candidate_profile = bool(
        report and str(report.get("schema", "")).startswith("leg-capacity-")
        and report.get("schema") != "leg-capacity-profile-replay-1"
        and not isinstance(report.get("candidate_profile"), Mapping)
        and not report.get("candidate_case_id")
    )
    return {
        "schema_version": "leg-capacity-validation-1",
        "status": "ACCEPTED_ENGINEERING_LIFT_RELEASE" if accepted else
        ("NO_ACCEPTED_CANDIDATE_PROFILE" if no_candidate_profile else "FAILED_ENGINEERING_LIFT_RELEASE_GATE"),
        "accepted": accepted,
        "candidate_available": candidate_required,
        "evidence_class": "prescribed_drive_mechanical_acceptance",
        "biological_validation": "NOT_VALIDATED",
        "autonomous_walking_claim": False,
        "input": {
            "trace_path": str(trace_path.resolve()), "trace_sha256": trace_hash,
            "report_path": str(report_path.resolve()) if report_path else None,
            "report_sha256": report_hash,
            "reported_trace_sha256": reported_trace_hash,
            "trace_sha256_matches_report": trace_hash_matches_report,
            "candidate_profile_path": profile.get("_resolved_path") if profile else None,
            "candidate_profile_sha256": profile_hash,
            "reference_report_path": str(reference_report_path.resolve())
            if reference_report_path else None,
            "reference_report_sha256": reference_report_hash,
        },
        "thresholds": {
            "native_dt_s": dt_s,
            "selected_leg_minimum_gap_m": gap_m,
            "selected_leg_minimum_contiguous_gap_s": gap_duration_s,
            "release_stable_contact_load_s": release_stable_s,
            "minimum_root_height_m": min_body_height_m,
            "per_foot_contact_noise_floor_N": contact_floor_n,
            "foot_radius_m_for_center_height_conversion": foot_radius_m,
            "root_wrench_tolerance": 0.0,
        },
        "decision_note": (
            "A peak clearance alone cannot pass. A prescribed-drive candidate needs one contiguous "
            "interval meeting both the gap and duration thresholds while its other five feet remain "
            "individually contacting and load-bearing. A matched zero-rate sham must stay supported "
            "without such an interval. These are engineering gates, not biological thresholds."
        ),
        "native_clock_continuous": overall_clock,
        "trace_sha256_matches_report": trace_hash_matches_report,
        "source_and_config_hash_provenance": provenance,
        "candidate_case_id": candidate_case_id or replay_case_id,
        "required_cases_pass": bool(required_cases) and all(
            case["accepted"] for case in required_cases
        ),
        "cases": case_results,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Full per-tick trajectory JSONL")
    parser.add_argument("--report", type=Path, help="Protocol/case report declaring prescribed rates and timings")
    parser.add_argument("--reference-report", type=Path,
                        help="Baseline report whose original core source/config digests must match")
    parser.add_argument("--out", type=Path, help="Write the acceptance JSON here; default is stdout")
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--minimum-gap-um", type=float, default=DEFAULT_GAP_M * 1e6)
    parser.add_argument("--minimum-gap-ms", type=float, default=DEFAULT_GAP_DURATION_S * 1e3)
    parser.add_argument("--release-stable-ms", type=float, default=DEFAULT_RELEASE_STABLE_S * 1e3)
    parser.add_argument("--minimum-root-height-mm", type=float, default=DEFAULT_MIN_BODY_HEIGHT_M * 1e3)
    parser.add_argument("--contact-noise-floor-n", type=float, default=0.0,
                        help="Declared per-foot force floor; zero requires strictly positive load")
    parser.add_argument("--foot-radius-um", type=float, default=DEFAULT_FOOT_RADIUS_M * 1e6)
    args = parser.parse_args(argv)
    values = (args.dt_s, args.minimum_gap_um, args.minimum_gap_ms,
              args.release_stable_ms, args.minimum_root_height_mm,
              args.contact_noise_floor_n, args.foot_radius_um)
    if not all(math.isfinite(value) and value >= 0.0 for value in values) or args.dt_s == 0.0:
        parser.error("thresholds must be finite and nonnegative; --dt-s must be positive")
    try:
        result = validate_trace(
            args.trace, report_path=args.report, reference_report_path=args.reference_report,
            dt_s=args.dt_s,
            gap_m=args.minimum_gap_um * 1e-6,
            gap_duration_s=args.minimum_gap_ms * 1e-3,
            release_stable_s=args.release_stable_ms * 1e-3,
            min_body_height_m=args.minimum_root_height_mm * 1e-3,
            contact_floor_n=args.contact_noise_floor_n,
            foot_radius_m=args.foot_radius_um * 1e-6,
        )
    except TraceError as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
