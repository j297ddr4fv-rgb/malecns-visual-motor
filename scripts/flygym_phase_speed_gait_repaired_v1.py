#!/usr/bin/env python3
"""Version-pinned entrypoint for the current V4.1/AP FlyGym mechanics.

The existing V4.1 controller already produces measured lift/touchdown events
on all six legs in its saved flat-ground captures. This module therefore does
not change gait equations, clearance, actuator targets, CPG timing, or the AP
sign. It adds a stable, hash-guarded API for mechanics captures while the AP
to free-body-yaw convention is calibrated separately from saved thorax yaw.

This remains an engineered position-servo/CPG controller, not a muscle or
connectome reconstruction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from flygym_ap_endpoint_gesture_prototype import (
    APEndpointGestureConfig,
    APEndpointGestureController as _APEndpointGestureController,
)
from flygym_phase_speed_gait_v4_1 import (
    PhaseSpeedGaitConfig as _V41Config,
)
from flygym_rest_contract import DEFAULT_P_DEADBAND, validate_p_deadband


_EXPECTED_V41_SHA256 = "2006a11f78a19158812d22d07876f5e9c4d6da83eea7f06ed441b6f44982f20d"
_EXPECTED_AP_SHA256 = "bcee1d927a2b731bddcf6935ebbeec70d353bf3a6eabcf4ecc7ccc2fd46dc4a0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_parent_sources() -> dict[str, str]:
    """Fail closed unless this entrypoint resolves the reviewed V4.1/AP code."""
    here = Path(__file__).resolve().parent
    expected = {
        "flygym_phase_speed_gait_v4_1.py": _EXPECTED_V41_SHA256,
        "flygym_ap_endpoint_gesture_prototype.py": _EXPECTED_AP_SHA256,
    }
    observed: dict[str, str] = {}
    for name, sha in expected.items():
        path = here / name
        actual = _sha256(path)
        if actual != sha:
            raise RuntimeError(f"reviewed controller source changed: {name}: {actual} != {sha}")
        observed[name] = actual
    return observed


class PhaseSpeedGaitController(_APEndpointGestureController):
    """Current V4.1 gait with the already-existing opt-in AP endpoint adapter.

    The class preserves the current AP sign convention exactly: for ``d >= 0``
    both gesture soma sides are left; for ``d < 0`` they are right. It does not
    inspect visual or stimulus labels and does not feed body yaw back to the
    controller. A calibration result may describe this sign, but never changes
    it automatically.
    """

    def __init__(self, session, config: APEndpointGestureConfig | None = None):
        parent_sources = verify_parent_sources()
        selected = config or APEndpointGestureConfig()
        shared_deadband = validate_p_deadband(DEFAULT_P_DEADBAND)
        if float(selected.active_threshold) != shared_deadband:
            raise ValueError(
                "active_threshold must match flygym_rest_contract.DEFAULT_P_DEADBAND"
            )
        super().__init__(session, selected)
        self.provenance.update({
            "controller": "PhaseSpeedGaitControllerRepairedV1",
            "controller_version": "repaired-v1-source-pinned-current-mechanics",
            "controller_parent": "APEndpointGestureController over PhaseSpeedGaitControllerV4_1",
            "controller_parent_sources": parent_sources,
            "shared_rest_contract": {
                "module": "flygym_rest_contract.py",
                "p_deadband": shared_deadband,
                "active_rule": "P > p_deadband",
                "zero_gate_behavior": "inherited V4.1 neutral hold; static inputs do not advance CPG phase",
            },
            "mechanics_delta_from_parent": "none; existing measured six-leg V4.1 gait and AP command are preserved",
            "repair_status": "no gait-trajectory bug localized in current V4.1 captures; AP-to-free-body-yaw sign awaits paired calibration",
            "stimulus_or_desired_yaw_feedback": False,
            "body_yaw_feedback": False,
            "neural_or_muscle_force_actuation": False,
        })


PhaseSpeedGaitConfig = APEndpointGestureConfig


def controller_config_for_differential_ap(
    differential_ap_d: float,
    *,
    base_config: APEndpointGestureConfig | None = None,
) -> APEndpointGestureConfig:
    """Return the current mirror-coded AP config for an externally decoded d.

    This is a mechanics adapter only. It never derives ``d`` from a stimulus
    label; callers provide the signed scalar. Positive ``d`` uses the current
    left-soma convention and negative ``d`` its mirror, with no extra gain.
    """
    value = float(differential_ap_d)
    if not (-1.0 <= value <= 1.0):
        raise ValueError("differential_ap_d must be finite and in [-1, 1]")
    if value != value or abs(value) == float("inf"):
        raise ValueError("differential_ap_d must be finite")
    base = base_config or APEndpointGestureConfig(active_threshold=DEFAULT_P_DEADBAND)
    if float(base.active_threshold) != validate_p_deadband(DEFAULT_P_DEADBAND):
        raise ValueError("base_config.active_threshold must match the shared rest contract")
    soma_side = "left" if value >= 0.0 else "right"
    return APEndpointGestureConfig(
        **{
            **vars(base),
            "active_threshold": DEFAULT_P_DEADBAND,
            "dna02_command": abs(value),
            "dng13_command": abs(value),
            "dna02_soma_side": soma_side,
            "dng13_soma_side": soma_side,
        }
    )


__all__ = [
    "APEndpointGestureConfig",
    "PhaseSpeedGaitConfig",
    "PhaseSpeedGaitController",
    "controller_config_for_differential_ap",
    "verify_parent_sources",
]
