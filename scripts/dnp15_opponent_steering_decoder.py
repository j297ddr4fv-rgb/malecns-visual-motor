#!/usr/bin/env python3
"""Pure, engineered DNp15-to-template steering readout candidate.

This module accepts only neural leg power and the two DNp15 population rates.
It does not access visual stimulus direction, timestamps, body state, or yaw,
and it does not run a worker or body simulation. The emitted DNa-like and
DNg-like labels name an engineered template interface, not identified
DNp15-to-muscle pathways or actual DNa02/DNg13 activity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

BASELINE_SCHEMA_VERSION = "dnp15-fixed-head-static-baseline-1"
READOUT_SCHEMA_VERSION = "dnp15-opponent-template-readout-1"
BASELINE_WINDOW_S = (6.0, 10.0)
BASELINE_FRAME_DT_S = 0.005
BASELINE_FRAME_COUNT = 800
BASELINE_FRAME_RANGE = (1200, 2000)
DNp15_MEMBERSHIP = {
    "DNp15/L": {"csr_index": 1935, "neuron_id": 12069, "side": "L"},
    "DNp15/R": {"csr_index": 1124, "neuron_id": 11215, "side": "R"},
}
_REQUIRED_SOURCE_HASHES = (
    "case_manifest_sha256",
    "completion_sha256",
    "run_integrity_sha256",
    "terminal_audit_sha256",
    "ready_validation_sha256",
    "frames_jsonl_sha256",
    "neural_state_diagnostic_sha256",
    "worker_provenance_sha256",
    "worker_integrity_sha256",
    "stimulus_spec_sha256",
    "fixed_pose_source_sha256",
    "source_freeze_sha256",
    "protocol_sha256",
    "config_sha256",
    "worker_source_sha256",
    "adapter_source_sha256",
)


class BaselinePayloadError(ValueError):
    """Raised when the static reference is not the frozen calibration source."""


def _finite_number(value: Any, field: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return number


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class StaticRateBaseline:
    """Bilateral DNp15 rates from the fixed-head, stationary c=1 capture."""

    left_hz: float
    right_hz: float
    source_case_id: str
    source_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StaticRateBaseline":
        if not isinstance(payload, Mapping):
            raise BaselinePayloadError("baseline payload must be a mapping")
        if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
            raise BaselinePayloadError("unsupported baseline payload schema")
        if payload.get("status") != "complete":
            raise BaselinePayloadError("baseline payload must be complete")

        source = payload.get("source")
        if not isinstance(source, Mapping):
            raise BaselinePayloadError("source metadata is required")
        required_source = {
            "condition": "stationary",
            "pose_mode": "fixed_head",
            "body_physics_instantiated": False,
            "brain_output_fed_back_to_pose": False,
            "stimulus_name": "stationary-c1",
        }
        for key, expected in required_source.items():
            if source.get(key) != expected:
                raise BaselinePayloadError(f"baseline source {key} must be {expected!r}")
        stimulus = source.get("stimulus")
        if not isinstance(stimulus, Mapping):
            raise BaselinePayloadError("stationary stimulus provenance is required")
        if _finite_number(stimulus.get("contrast"), "stimulus.contrast") != 1.0:
            raise BaselinePayloadError("this candidate is calibrated only for c=1")
        if _finite_number(stimulus.get("wavelength_deg"), "stimulus.wavelength_deg") != 60.0:
            raise BaselinePayloadError("baseline wavelength must be 60 degrees")
        if _finite_number(stimulus.get("temporal_frequency_hz"), "stimulus.temporal_frequency_hz") != 0.0:
            raise BaselinePayloadError("baseline must be stationary")
        if stimulus.get("display_geometry") != "fixed_arena_infinite_distance_angular_field":
            raise BaselinePayloadError("unexpected static-scene geometry")

        rate_measurement = payload.get("rate_measurement")
        if not isinstance(rate_measurement, Mapping):
            raise BaselinePayloadError("rate-measurement provenance is required")
        if rate_measurement.get("source_field") != "population_spike_counts":
            raise BaselinePayloadError("baseline rates must come from saved per-frame spike counts")
        if rate_measurement.get("cell_count_per_selector") != 1:
            raise BaselinePayloadError("each side must be one exact DNp15 selector")

        window = payload.get("window")
        if not isinstance(window, Mapping):
            raise BaselinePayloadError("baseline window is required")
        exact_window = {
            "start_s": 6.0,
            "stop_s_exclusive": 10.0,
            "frame_dt_s": BASELINE_FRAME_DT_S,
            "frame_count": BASELINE_FRAME_COUNT,
            "seq_start_inclusive": BASELINE_FRAME_RANGE[0],
            "seq_stop_exclusive": BASELINE_FRAME_RANGE[1],
        }
        for key, expected in exact_window.items():
            if window.get(key) != expected:
                raise BaselinePayloadError(f"baseline window {key} must be {expected!r}")

        if payload.get("moving_case_data_used") is not False:
            raise BaselinePayloadError("moving-case data cannot set the baseline")
        if payload.get("observed_yaw_used") is not False:
            raise BaselinePayloadError("observed yaw cannot set the baseline")

        audit = payload.get("audit")
        if not isinstance(audit, Mapping):
            raise BaselinePayloadError("capture audit metadata is required")
        if audit.get("completion_status") != "complete":
            raise BaselinePayloadError("baseline capture must be complete")
        if audit.get("terminal_status") != "pass":
            raise BaselinePayloadError("baseline terminal audit must pass")
        if audit.get("run_integrity_status") != "complete":
            raise BaselinePayloadError("baseline integrity audit must be complete")
        if audit.get("selector_count") != 18:
            raise BaselinePayloadError("baseline must use the 18-selector diagnostic capture")
        if audit.get("shared_arrays_exact_to_reference") is not True:
            raise BaselinePayloadError("shared capture consistency must pass")
        if audit.get("shared_14_population_rates_and_counts_exact_to_reference") is not True:
            raise BaselinePayloadError("shared population telemetry consistency must pass")

        selectors = payload.get("selectors")
        if not isinstance(selectors, Mapping):
            raise BaselinePayloadError("selector membership is required")
        for name, expected in DNp15_MEMBERSHIP.items():
            found = selectors.get(name)
            if not isinstance(found, Mapping):
                raise BaselinePayloadError(f"missing exact selector {name}")
            for key, value in expected.items():
                if found.get(key) != value:
                    raise BaselinePayloadError(f"selector {name} has unexpected {key}")

        cells = payload.get("cells")
        if not isinstance(cells, Mapping):
            raise BaselinePayloadError("baseline cell rates are required")
        rates: dict[str, float] = {}
        for name in DNp15_MEMBERSHIP:
            cell = cells.get(name)
            if not isinstance(cell, Mapping):
                raise BaselinePayloadError(f"missing baseline rate for {name}")
            spikes = cell.get("spikes")
            if isinstance(spikes, bool) or not isinstance(spikes, int) or spikes < 0:
                raise BaselinePayloadError(f"{name} spikes must be a non-negative integer")
            rate = _finite_number(cell.get("rate_hz"), f"{name}.rate_hz", minimum=0.0)
            expected_rate = spikes / 4.0
            if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
                raise BaselinePayloadError(f"{name} rate does not match its 4 s spike count")
            rates[name] = rate

        source_hashes = payload.get("source_hashes")
        if not isinstance(source_hashes, Mapping):
            raise BaselinePayloadError("raw source hashes are required")
        hashes: list[tuple[str, str]] = []
        for key in _REQUIRED_SOURCE_HASHES:
            value = source_hashes.get(key)
            if not _is_sha256(value):
                raise BaselinePayloadError(f"missing valid SHA-256 for {key}")
            hashes.append((key, value))

        source_case_id = source.get("case_id")
        if not isinstance(source_case_id, str) or not source_case_id:
            raise BaselinePayloadError("source case_id is required")
        return cls(
            left_hz=rates["DNp15/L"],
            right_hz=rates["DNp15/R"],
            source_case_id=source_case_id,
            source_hashes=tuple(hashes),
        )


@dataclass(frozen=True)
class DecoderConfig:
    """Fixed candidate settings; there are no fitted gains."""

    frame_dt_s: float = 0.005
    ema_tau_s: float = 0.150
    denominator_floor_hz: float = 1.0
    command_clip: float = 1.0
    full_scale_endpoint_mm: float = 0.05

    def __post_init__(self) -> None:
        fixed = {
            "frame_dt_s": 0.005,
            "ema_tau_s": 0.150,
            "denominator_floor_hz": 1.0,
            "command_clip": 1.0,
            "full_scale_endpoint_mm": 0.05,
        }
        for key, expected in fixed.items():
            value = _finite_number(getattr(self, key), key, minimum=0.0)
            if value != expected:
                raise ValueError(f"{key} is frozen at {expected}; this prototype has no fit parameter")


@dataclass(frozen=True)
class TemplateAction:
    action: str
    endpoint_pattern: str
    command_magnitude: float
    soma_side: str | None
    target_side: str | None
    full_scale_endpoint_mm: float
    requested_endpoint_magnitude_mm: float


@dataclass(frozen=True)
class DecoderFrame:
    tick_index: int
    leg_power_left: float
    leg_power_right: float
    dn_p15_left_rate_hz: float
    dn_p15_right_rate_hz: float
    ema_left_rate_hz: float
    ema_right_rate_hz: float
    baseline_left_rate_hz: float
    baseline_right_rate_hz: float
    denominator_hz: float
    raw_opponent_signal: float
    clipped_opponent_signal: float
    command_amplitude: float
    soma_side: str | None
    zero_power_baseline_hold: bool
    dna_like_ipsilateral_aep_short: TemplateAction
    dng_like_contralateral_extension: TemplateAction

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": READOUT_SCHEMA_VERSION,
            **asdict(self),
            "classification": "engineered DNp15-to-template interface; not a known DNp15-to-muscle map",
            "power_mapping": "neural leg_power P is preserved as the symmetric CPG pair (P,P), replacing the old pooled steering source",
            "action_boundary": "DNa-like/DNg-like are abstract template intents, not actual DNa02/DNg13 spike commands or forces",
        }


class Dnp15OpponentSteeringDecoder:
    """Stateful pure 200 Hz rate-difference decoder for a gait-template adapter.

    Call once per 5 ms neural frame with exactly P, DNp15/L rate and DNp15/R
    rate. A zero-power frame restores the calibrated zero-motion EMA baseline,
    holds it, and emits zero power and zero template commands.
    """

    def __init__(self, baseline: StaticRateBaseline, config: DecoderConfig | None = None):
        if not isinstance(baseline, StaticRateBaseline):
            raise TypeError("a validated fixed-head stationary baseline payload is required")
        self.config = config or DecoderConfig()
        self.baseline = baseline
        self._alpha = 1.0 - math.exp(-self.config.frame_dt_s / self.config.ema_tau_s)
        self._tick_index = 0
        self._ema_left = baseline.left_hz
        self._ema_right = baseline.right_hz

    @property
    def tick_index(self) -> int:
        return self._tick_index

    def reset(self) -> None:
        self._tick_index = 0
        self._ema_left = self.baseline.left_hz
        self._ema_right = self.baseline.right_hz

    @staticmethod
    def _opposite(side: str | None) -> str | None:
        if side == "left":
            return "right"
        if side == "right":
            return "left"
        return None

    def step(self, leg_power: float, dn_p15_left_rate_hz: float,
             dn_p15_right_rate_hz: float) -> DecoderFrame:
        """Advance the decoder by one fixed 5 ms frame and return a pure intent."""
        power = _finite_number(leg_power, "leg_power", minimum=0.0, maximum=1.0)
        left_rate = _finite_number(dn_p15_left_rate_hz, "dn_p15_left_rate_hz", minimum=0.0)
        right_rate = _finite_number(dn_p15_right_rate_hz, "dn_p15_right_rate_hz", minimum=0.0)
        tick = self._tick_index
        self._tick_index += 1

        zero_hold = power == 0.0
        if zero_hold:
            self._ema_left = self.baseline.left_hz
            self._ema_right = self.baseline.right_hz
            raw = 0.0
            signal = 0.0
        else:
            self._ema_left += self._alpha * (left_rate - self._ema_left)
            self._ema_right += self._alpha * (right_rate - self._ema_right)
            denominator = max(
                self.baseline.left_hz + self.baseline.right_hz,
                self.config.denominator_floor_hz,
            )
            raw = (
                (self._ema_left - self.baseline.left_hz)
                - (self._ema_right - self.baseline.right_hz)
            ) / denominator
            signal = max(-self.config.command_clip, min(self.config.command_clip, raw))

        denominator = max(
            self.baseline.left_hz + self.baseline.right_hz,
            self.config.denominator_floor_hz,
        )
        amplitude = abs(signal)
        side = "left" if signal > 0.0 else "right" if signal < 0.0 else None
        contra_side = self._opposite(side)
        full_scale = self.config.full_scale_endpoint_mm
        scaled_mm = full_scale * amplitude
        dna = TemplateAction(
            action="ipsilateral_aep_shortening",
            endpoint_pattern="AEP posterior; PEP unchanged",
            command_magnitude=amplitude,
            soma_side=side,
            target_side=side,
            full_scale_endpoint_mm=full_scale,
            requested_endpoint_magnitude_mm=scaled_mm,
        )
        dng = TemplateAction(
            action="contralateral_endpoint_extension",
            endpoint_pattern="AEP forward and PEP backward",
            command_magnitude=amplitude,
            soma_side=side,
            target_side=contra_side,
            full_scale_endpoint_mm=full_scale,
            requested_endpoint_magnitude_mm=scaled_mm,
        )
        return DecoderFrame(
            tick_index=tick,
            leg_power_left=power,
            leg_power_right=power,
            dn_p15_left_rate_hz=left_rate,
            dn_p15_right_rate_hz=right_rate,
            ema_left_rate_hz=self._ema_left,
            ema_right_rate_hz=self._ema_right,
            baseline_left_rate_hz=self.baseline.left_hz,
            baseline_right_rate_hz=self.baseline.right_hz,
            denominator_hz=denominator,
            raw_opponent_signal=raw,
            clipped_opponent_signal=signal,
            command_amplitude=amplitude,
            soma_side=side,
            zero_power_baseline_hold=zero_hold,
            dna_like_ipsilateral_aep_short=dna,
            dng_like_contralateral_extension=dng,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": READOUT_SCHEMA_VERSION,
            "classification": "engineered readout candidate; not a biological DNp15-to-muscle pathway",
            "baseline_case_id": self.baseline.source_case_id,
            "baseline_rates_hz": {"DNp15/L": self.baseline.left_hz, "DNp15/R": self.baseline.right_hz},
            "baseline_source_hashes": dict(self.baseline.source_hashes),
            "formula": "clip((((EMA_L-bL)-(EMA_R-bR))/max(bL+bR,1Hz)),-1,+1)",
            "ema": {"sample_dt_s": self.config.frame_dt_s, "time_constant_s": self.config.ema_tau_s, "alpha": self._alpha},
            "zero_power_policy": "reset and hold both EMA states at static baseline; emit (0,0) power and zero commands",
            "power_policy": "output left/right CPG power as (P,P); pooled leg_steering is not an input",
            "command_policy": "both abstract template actions receive abs(d); d>0 selects left soma, d<0 right soma",
            "full_scale_endpoint_mm": self.config.full_scale_endpoint_mm,
            "uses_stimulus_direction": False,
            "uses_external_time_or_yaw": False,
            "runs_neural_worker_or_body": False,
            "actual_dna02_or_dng13_spikes": False,
            "known_dn_to_muscle_mapping": False,
            "calibration_scope": "fixed-head stationary c=1, 60-degree-period angular scene; no contrast generalization claimed",
        }
