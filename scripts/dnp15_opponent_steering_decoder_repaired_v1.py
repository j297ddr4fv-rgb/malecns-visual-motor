#!/usr/bin/env python3
"""Deadband-consistent DNp15 decoder with a backwards-compatible API.

This remains a pure rate readout. It accepts neural leg power P and bilateral
DNp15 event rates; it never reads stimulus labels, desired direction, pose, yaw,
or body state. The decoder's control EMA, signed d, and template actions share
one strict activity rule: P is active iff P > the configured deadband. The
default deadband is the existing phase controller's 1e-6 engineering value.

The old module remains frozen. Historical names and the positional
``step(P, L, R)`` call are retained here. The extended schema distinguishes raw
diagnostics from the zeroed control signal sent to a consumer.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math
from typing import Any

try:  # Support both ``scripts.module`` and a scripts-on-PYTHONPATH import.
    from .dnp15_opponent_steering_decoder import (
        BASELINE_SCHEMA_VERSION,
        BaselinePayloadError,
        DecoderFrame as _LegacyDecoderFrame,
        StaticRateBaseline,
        TemplateAction,
    )
    from .flygym_direction_calibration import (
        DirectionCalibrationError,
        load_calibration_config,
        resolve_baseline,
    )
    from .flygym_rest_contract import (
        DEFAULT_P_DEADBAND,
        FlyGymRestContract,
        validate_leg_power,
        validate_p_deadband,
    )
except ImportError:  # pragma: no cover - exercised by direct scripts imports
    from dnp15_opponent_steering_decoder import (
        BASELINE_SCHEMA_VERSION,
        BaselinePayloadError,
        DecoderFrame as _LegacyDecoderFrame,
        StaticRateBaseline,
        TemplateAction,
    )
    from flygym_direction_calibration import (
        DirectionCalibrationError,
        load_calibration_config,
        resolve_baseline,
    )
    from flygym_rest_contract import (
        DEFAULT_P_DEADBAND,
        FlyGymRestContract,
        validate_leg_power,
        validate_p_deadband,
    )


READOUT_SCHEMA_VERSION = "dnp15-opponent-template-readout-repaired-v1"
SUPPORTED_BASELINE_MODES = ("calibrated_static", "zero_gray")
_FIXED_LEGACY_CONFIG = {
    "frame_dt_s": 0.005,
    "ema_tau_s": 0.150,
    "denominator_floor_hz": 1.0,
    "command_clip": 1.0,
    "full_scale_endpoint_mm": 0.05,
}


def _finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
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


@dataclass(frozen=True)
class DecoderConfig:
    """Fixed decoder constants plus explicit rest and baseline policies.

    The legacy positional fields keep their old names and values. The
    ``calibrated_static`` default preserves the old constructor behavior and
    requires a validated frozen baseline payload. The ``zero_gray`` mode is
    an explicit candidate reference and requires its fixed normalization to
    be stated by the caller.
    """

    frame_dt_s: float = 0.005
    ema_tau_s: float = 0.150
    denominator_floor_hz: float = 1.0
    command_clip: float = 1.0
    full_scale_endpoint_mm: float = 0.05
    p_deadband: float = DEFAULT_P_DEADBAND
    baseline_mode: str = "calibrated_static"
    normalization_hz: float | None = None

    def __post_init__(self) -> None:
        for key, expected in _FIXED_LEGACY_CONFIG.items():
            value = _finite_number(getattr(self, key), key, minimum=0.0)
            if value != expected:
                raise ValueError(f"{key} is frozen at {expected}; this prototype has no fit parameter")
        validate_p_deadband(self.p_deadband)
        if self.baseline_mode not in SUPPORTED_BASELINE_MODES:
            raise ValueError(
                "baseline_mode must be calibrated_static or zero_gray"
            )
        if self.normalization_hz is not None:
            _finite_number(self.normalization_hz, "normalization_hz", minimum=0.0)
            if float(self.normalization_hz) <= 0.0:
                raise ValueError("normalization_hz must be positive")


@dataclass(frozen=True)
class DecoderFrame(_LegacyDecoderFrame):
    """One decoder output, retaining old keys and adding explicit diagnostics.

    ``leg_power_left/right`` keep their legacy meaning as the raw neural P
    input. Consumers that actuate a controller must use ``control_power`` or
    ``control_power_left/right``; these are exactly zero in the shared static
    deadband. ``ema_*`` denotes the command EMA; ``raw_ema_*`` is its
    continuously updated diagnostic counterpart. ``raw_opponent_signal``
    keeps the legacy command-EMA preclip formula. The added
    ``ungated_opponent_signal`` and ``unclipped_control_signal`` are computed
    from the continuous diagnostic EMA, independently of the gate.
    """

    # Explicit aliases and diagnostics for the repaired contract. All legacy
    # dataclass fields are inherited unchanged from the old module.
    raw_leg_power: float
    power_raw: float
    control_power: float
    control_power_left: float
    control_power_right: float
    raw_left_hz: float
    raw_right_hz: float
    ema_left_hz: float
    ema_right_hz: float
    raw_ema_left_hz: float
    raw_ema_right_hz: float
    raw_ema_initialized: bool
    raw_opponent_difference_hz: float
    continuous_raw_ema_opponent_difference_hz: float
    command_ema_opponent_difference_hz: float
    baseline_corrected_opponent_difference_hz: float
    command_ema_baseline_corrected_opponent_difference_hz: float
    ungated_opponent_signal: float
    unclipped_raw_difference_signal: float
    ungated_clipped_control_signal: float
    gated_unclipped_raw_difference_signal: float
    clipped_raw_difference_signal: float
    gated_clipped_raw_difference_signal: float
    unclipped_control_signal: float
    gated_unclipped_control_signal: float
    command_unclipped_control_signal: float
    clipped_control_signal: float
    power_active: bool
    static_gate_active: bool
    p_deadband: float
    power_threshold: float
    power_gate_transition: str
    power_gate_state: str
    baseline_mode: str
    normalization_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": READOUT_SCHEMA_VERSION,
            **asdict(self),
            "classification": (
                "engineered DNp15-to-template interface; not a known "
                "DNp15-to-muscle map"
            ),
            "power_mapping": (
                "raw neural P is preserved in legacy leg_power fields; use "
                "control_power for actuation; effective active power is "
                "symmetric (P_control,P_control)"
            ),
            "legacy_field_semantics": {
                "raw_opponent_signal": "command EMA relative to selected baseline, before clipping; zero in the static gate",
                "clipped_opponent_signal": "gated command signal; zero for P <= p_deadband",
                "zero_power_baseline_hold": "true only when raw P == 0",
                "ema_left_rate_hz/ema_right_rate_hz": "command EMA, reset to selected baseline while static",
            },
            "action_boundary": (
                "DNa-like/DNg-like are abstract template intents, not actual "
                "DNa02/DNg13 spike commands or forces"
            ),
        }


class Dnp15OpponentSteeringDecoder:
    """Stateful pure 200 Hz rate-difference decoder.

    The old constructor remains valid::

        decoder = Dnp15OpponentSteeringDecoder(validated_baseline, DecoderConfig())

    A candidate zero-gray decoder is explicit and has no stimulus-label
    argument::

        decoder = Dnp15OpponentSteeringDecoder(
            baseline_mode="zero_gray", normalization_hz=142.5
        )

    ``step(P, left_hz, right_hz)`` is unchanged. Only P controls the shared
    gate; raw rate diagnostics continue to update while control is static.
    """

    def __init__(
        self,
        baseline: StaticRateBaseline | None = None,
        config: DecoderConfig | Any | None = None,
        *,
        baseline_mode: str | None = None,
        normalization_hz: float | None = None,
        p_deadband: float | None = None,
    ) -> None:
        if config is None:
            config = DecoderConfig()
        elif not isinstance(config, DecoderConfig):
            # Accept the old module's DecoderConfig without requiring callers
            # to change an already constructed config instance.
            legacy = config
            required = tuple(_FIXED_LEGACY_CONFIG)
            if not all(hasattr(legacy, key) for key in required):
                raise TypeError("config must be a repaired or legacy DecoderConfig")
            config = DecoderConfig(
                **{key: getattr(legacy, key) for key in required},
                p_deadband=DEFAULT_P_DEADBAND,
                baseline_mode="calibrated_static",
            )

        updates: dict[str, Any] = {}
        if baseline_mode is not None:
            updates["baseline_mode"] = baseline_mode
        if normalization_hz is not None:
            updates["normalization_hz"] = normalization_hz
        if p_deadband is not None:
            updates["p_deadband"] = p_deadband
        if updates:
            config = replace(config, **updates)
        self.config = config

        try:
            calibration_config = load_calibration_config()
            spec = resolve_baseline(calibration_config, self.config.baseline_mode)
        except DirectionCalibrationError as exc:
            raise ValueError(f"baseline mode is not available: {exc}") from exc

        if self.config.baseline_mode == "calibrated_static":
            if not isinstance(baseline, StaticRateBaseline):
                raise TypeError(
                    "calibrated_static requires a validated StaticRateBaseline payload"
                )
            if (
                baseline.left_hz != spec.left_hz
                or baseline.right_hz != spec.right_hz
            ):
                raise BaselinePayloadError(
                    "validated baseline rates do not match the declared calibrated_static policy"
                )
            self.baseline = baseline
            self._baseline_left_hz = baseline.left_hz
            self._baseline_right_hz = baseline.right_hz
        else:
            if baseline is not None:
                raise ValueError(
                    "zero_gray uses its declared zero-rate reference; pass baseline=None"
                )
            self.baseline = None
            self._baseline_left_hz = spec.left_hz
            self._baseline_right_hz = spec.right_hz

        requested_norm = self.config.normalization_hz
        if self.config.baseline_mode == "zero_gray" and requested_norm is None:
            raise ValueError(
                "zero_gray requires explicit normalization_hz (the frozen candidate uses 142.5 Hz)"
            )
        self.normalization_hz = (
            float(requested_norm) if requested_norm is not None
            else max(
                self._baseline_left_hz + self._baseline_right_hz,
                self.config.denominator_floor_hz,
            )
        )
        if not math.isclose(
            self.normalization_hz,
            spec.normalization_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"normalization_hz is frozen at {spec.normalization_hz:g} for "
                f"baseline_mode={self.config.baseline_mode!r}"
            )

        self._alpha = 1.0 - math.exp(
            -self.config.frame_dt_s / self.config.ema_tau_s
        )
        self._rest = FlyGymRestContract(self.config.p_deadband)
        self._tick_index = 0
        self._raw_ema_left = 0.0
        self._raw_ema_right = 0.0
        self._raw_ema_initialized = False
        self._control_ema_left = self._baseline_left_hz
        self._control_ema_right = self._baseline_right_hz

    @property
    def tick_index(self) -> int:
        return self._tick_index

    def reset(self) -> None:
        """Reset command EMA to baseline and raw EMA to first-sample state."""
        self._tick_index = 0
        self._rest.reset()
        self._control_ema_left = self._baseline_left_hz
        self._control_ema_right = self._baseline_right_hz
        self._raw_ema_left = 0.0
        self._raw_ema_right = 0.0
        self._raw_ema_initialized = False

    @staticmethod
    def _opposite(side: str | None) -> str | None:
        if side == "left":
            return "right"
        if side == "right":
            return "left"
        return None

    def step(
        self,
        leg_power: float,
        dn_p15_left_rate_hz: float,
        dn_p15_right_rate_hz: float,
    ) -> DecoderFrame:
        """Advance one fixed 5 ms frame and return raw plus gated readouts."""
        power = validate_leg_power(leg_power)
        left_rate = _finite_number(
            dn_p15_left_rate_hz, "dn_p15_left_rate_hz", minimum=0.0
        )
        right_rate = _finite_number(
            dn_p15_right_rate_hz, "dn_p15_right_rate_hz", minimum=0.0
        )
        gate = self._rest.step(power)
        tick = self._tick_index
        self._tick_index += 1

        raw_ema_initialized_this_frame = not self._raw_ema_initialized
        if raw_ema_initialized_this_frame:
            self._raw_ema_left = left_rate
            self._raw_ema_right = right_rate
            self._raw_ema_initialized = True
        else:
            self._raw_ema_left += self._alpha * (left_rate - self._raw_ema_left)
            self._raw_ema_right += self._alpha * (right_rate - self._raw_ema_right)

        if gate.active:
            self._control_ema_left += self._alpha * (
                left_rate - self._control_ema_left
            )
            self._control_ema_right += self._alpha * (
                right_rate - self._control_ema_right
            )
        else:
            self._control_ema_left = self._baseline_left_hz
            self._control_ema_right = self._baseline_right_hz

        command_ema_difference_hz = (
            self._control_ema_left - self._control_ema_right
        )
        continuous_raw_difference_hz = self._raw_ema_left - self._raw_ema_right
        corrected_raw_difference_hz = (
            (self._raw_ema_left - self._baseline_left_hz)
            - (self._raw_ema_right - self._baseline_right_hz)
        )
        corrected_difference_hz = (
            (self._control_ema_left - self._baseline_left_hz)
            - (self._control_ema_right - self._baseline_right_hz)
        )
        command_unclipped = corrected_difference_hz / self.normalization_hz
        raw_unclipped = corrected_raw_difference_hz / self.normalization_hz
        raw_difference_unclipped = (
            continuous_raw_difference_hz / self.normalization_hz
        )
        gated_unclipped = command_unclipped if gate.active else 0.0
        gated_raw_difference_unclipped = (
            raw_difference_unclipped if gate.active else 0.0
        )
        raw_control_clipped = max(
            -self.config.command_clip,
            min(self.config.command_clip, raw_unclipped),
        )
        raw_difference_clipped = max(
            -self.config.command_clip,
            min(self.config.command_clip, raw_difference_unclipped),
        )
        gated_raw_difference_clipped = max(
            -self.config.command_clip,
            min(self.config.command_clip, gated_raw_difference_unclipped),
        )
        clipped = max(
            -self.config.command_clip,
            min(self.config.command_clip, gated_unclipped),
        )

        # The legacy raw field remains the command-EMA preclip formula. The
        # continuous, ungated raw-rate diagnostic has its own new field.
        raw_signal = command_unclipped
        amplitude = abs(clipped)
        side = "left" if clipped > 0.0 else "right" if clipped < 0.0 else None
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
            ema_left_rate_hz=self._control_ema_left,
            ema_right_rate_hz=self._control_ema_right,
            baseline_left_rate_hz=self._baseline_left_hz,
            baseline_right_rate_hz=self._baseline_right_hz,
            denominator_hz=self.normalization_hz,
            raw_opponent_signal=raw_signal,
            clipped_opponent_signal=clipped,
            command_amplitude=amplitude,
            soma_side=side,
            zero_power_baseline_hold=(power == 0.0),
            dna_like_ipsilateral_aep_short=dna,
            dng_like_contralateral_extension=dng,
            raw_leg_power=power,
            power_raw=power,
            control_power=gate.control_power,
            control_power_left=gate.control_power,
            control_power_right=gate.control_power,
            raw_left_hz=left_rate,
            raw_right_hz=right_rate,
            ema_left_hz=self._control_ema_left,
            ema_right_hz=self._control_ema_right,
            raw_ema_left_hz=self._raw_ema_left,
            raw_ema_right_hz=self._raw_ema_right,
            raw_ema_initialized=raw_ema_initialized_this_frame,
            raw_opponent_difference_hz=continuous_raw_difference_hz,
            continuous_raw_ema_opponent_difference_hz=continuous_raw_difference_hz,
            command_ema_opponent_difference_hz=command_ema_difference_hz,
            baseline_corrected_opponent_difference_hz=corrected_raw_difference_hz,
            command_ema_baseline_corrected_opponent_difference_hz=corrected_difference_hz,
            ungated_opponent_signal=raw_unclipped,
            unclipped_raw_difference_signal=raw_difference_unclipped,
            ungated_clipped_control_signal=raw_control_clipped,
            gated_unclipped_raw_difference_signal=gated_raw_difference_unclipped,
            clipped_raw_difference_signal=raw_difference_clipped,
            gated_clipped_raw_difference_signal=gated_raw_difference_clipped,
            unclipped_control_signal=raw_unclipped,
            gated_unclipped_control_signal=gated_unclipped,
            command_unclipped_control_signal=command_unclipped,
            clipped_control_signal=clipped,
            power_active=gate.active,
            static_gate_active=gate.static,
            p_deadband=gate.p_deadband,
            power_threshold=gate.p_deadband,
            power_gate_transition=gate.transition,
            power_gate_state=gate.state,
            baseline_mode=self.config.baseline_mode,
            normalization_hz=self.normalization_hz,
        )

    def provenance(self) -> dict[str, Any]:
        if self.config.baseline_mode == "calibrated_static":
            scope_id = "fixed_head_c1_60deg_angular_stripe_reference"
            baseline_source_hashes = dict(self.baseline.source_hashes)
            evidence_status = "calibrated_scope_limited"
            calibration_scope = (
                "fixed-head stationary C=1, 60-degree-period angular scene; "
                "not a walking/full-pose or general-background baseline"
            )
        else:
            scope_id = "frozen_gray_prelude_zero_rate_reference"
            baseline_source_hashes = {}
            evidence_status = "candidate_reference"
            calibration_scope = (
                "uniform-gray prelude only; no general background/contrast "
                "neutrality or biological calibration claimed"
            )
        return {
            "schema_version": READOUT_SCHEMA_VERSION,
            "classification": "engineered readout candidate; not a biological DNp15-to-muscle pathway",
            "baseline_mode": self.config.baseline_mode,
            "baseline_scope_id": scope_id,
            "baseline_evidence_status": evidence_status,
            "baseline_scope_validated": False,
            "baseline_scope_validation": "not performed; decoder accepts no stimulus or case label",
            "calibrated_static_out_of_scope_policy": "diagnostic-only outside matched fixed-head C=1, 60-degree stationary stripe scope",
            "baseline_rates_hz": {
                "DNp15/L": self._baseline_left_hz,
                "DNp15/R": self._baseline_right_hz,
            },
            "baseline_source_hashes": baseline_source_hashes,
            "normalization_hz": self.normalization_hz,
            "formula": "clip(gate(P) * (((EMA_L-bL)-(EMA_R-bR))/normalization_hz), -1, +1)",
            "ema": {
                "sample_dt_s": self.config.frame_dt_s,
                "time_constant_s": self.config.ema_tau_s,
                "alpha": self._alpha,
                "control_initialization": "selected fixed baseline",
                "raw_diagnostic_initialization": "first observed raw DNp15 rates",
            },
            "power_gate": {
                "active_if": "P > p_deadband",
                "static_if": "P <= p_deadband",
                "p_deadband": self.config.p_deadband,
                "threshold_status": "phase-controller-compatible engineering value; pending independent noise/sensitivity calibration",
                "raw_P_preserved": True,
            },
            "static_policy": "reset command EMA to selected baseline; emit zero control P, d, and template commands; continue raw rate EMA diagnostics",
            "raw_policy": "continuous raw EMA and EMA_L-EMA_R are diagnostics only and do not bypass the power gate",
            "power_policy": "effective control power is symmetric (P_control,P_control); use control_power fields for actuation",
            "command_policy": "both abstract template actions receive abs(control d); sign selects a template soma side",
            "full_scale_endpoint_mm": self.config.full_scale_endpoint_mm,
            "uses_stimulus_direction": False,
            "uses_case_label": False,
            "uses_external_time_or_yaw": False,
            "runs_neural_worker_or_body": False,
            "actual_dna02_or_dng13_spikes": False,
            "known_dn_to_muscle_mapping": False,
            "calibration_scope": calibration_scope,
        }


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "BaselinePayloadError",
    "DecoderConfig",
    "DecoderFrame",
    "Dnp15OpponentSteeringDecoder",
    "READOUT_SCHEMA_VERSION",
    "StaticRateBaseline",
    "SUPPORTED_BASELINE_MODES",
    "TemplateAction",
]
