#!/usr/bin/env python3
"""Phase-speed mapping over FlyGym's native joint-template gait.

This is an engineered position-servo gait controller.  It keeps the pinned
hybrid CPG, adhesion events, and sensory corrections, while mapping bilateral
drive to phase rate instead of joint-template amplitude.  A small optional
hind-foot position-only correction adds swing clearance from the same-tick
native template target; it never writes the live or root state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from importlib.resources import files

import numpy as np

from flygym_demo.complex_terrain.common import (
    LocomotionAction,
    get_default_locomotion_dof_order,
)
from flygym_demo.complex_terrain.hybrid_controller import HybridController


LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
_SIDE_CHANNELS = {"left": (0, 1, 2), "right": (3, 4, 5)}
_SOURCE_MODULES = (
    "turning_controller.py",
    "hybrid_controller.py",
    "cpg_controller.py",
    "preprogrammed.py",
)


@dataclass(frozen=True)
class PhaseSpeedGaitConfig:
    """Small controller settings in the pinned model's native units.

    ``cycle_frequency_hz`` defaults to the loaded single-step asset's source
    cycle rate.  Joint targets remain full-template targets at every positive
    drive; drive controls oscillator frequency and side-to-side rate only.
    Hind command bounds are engineering limits around each tick's native
    target, not model joint limits or biological angle ranges.
    """

    cycle_frequency_hz: float | None = None
    active_threshold: float = 1e-6
    enable_hind_lift: bool = True
    hind_lift_mm: float = 0.05
    hind_command_offset_limit_rad: float = 0.35
    hind_ik_delta_limit_rad: float = 0.04
    hind_ik_damping_mm: float = 0.025
    hind_ik_iterations: int = 6


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class PhaseSpeedGaitController:
    """Wrap the pinned hybrid controller with drive-to-cadence mapping.

    Attach before the first control tick, then use the session's normal
    ``step_control_tick([left, right])`` interface.  Signed input controls
    forward/reverse phase direction; absolute input controls cadence.  An
    inactive side is held at neutral with adhesion on.  All-zero input holds
    the whole fly at neutral and freezes CPG phase.
    """

    supports_compact_result = True

    def __init__(self, session, config: PhaseSpeedGaitConfig | None = None):
        self.session = session
        self.base = session.controller
        if hasattr(self.base, "base"):
            raise ValueError("wrap the original session controller, not another adapter")
        if int(getattr(session, "_control_samples", 0)) != 0:
            raise ValueError("attach PhaseSpeedGaitController before the first control tick")
        if tuple(self.base.legs) != LEGS:
            raise ValueError(f"expected native leg order {LEGS}, got {self.base.legs}")

        self.config = config or PhaseSpeedGaitConfig()
        self._validate_config()
        self.mj = session.mj
        self.model = session.model
        self.steps = self.base.preprogrammed_steps
        self.dof_order = self.base.output_dof_order or get_default_locomotion_dof_order()
        self.position_actuator_ids = np.asarray(session.position_actuator_ids, dtype=int)
        self.joint_ids = self.model.actuator_trnid[self.position_actuator_ids, 0].astype(int)
        if len(self.joint_ids) != 42 or np.any(self.joint_ids < 0):
            raise ValueError("expected 42 native position actuators with joint transmissions")
        self.qadr = np.asarray(self.model.jnt_qposadr[self.joint_ids], dtype=int)
        self.vadr = np.asarray(self.model.jnt_dofadr[self.joint_ids], dtype=int)
        self._validate_actuator_order()
        self.leg_channels = {
            leg: np.asarray(
                [i for i, dof in enumerate(self.dof_order)
                 if dof.child.name.startswith(leg + "_")],
                dtype=int,
            )
            for leg in LEGS
        }
        if any(len(self.leg_channels[leg]) != 7 for leg in LEGS):
            raise ValueError("expected seven ordered active position DOFs per leg")

        self.neutral_angles = np.asarray(
            self.steps.default_pose_by_dof_order(self.dof_order), dtype=float
        )
        if self.neutral_angles.shape != (42,):
            raise ValueError(f"unexpected neutral template shape {self.neutral_angles.shape}")
        source_frequency = float(self.steps.step_cycle_frequency_hz)
        self.cycle_frequency_hz = (
            source_frequency if self.config.cycle_frequency_hz is None
            else float(self.config.cycle_frequency_hz)
        )
        if not math.isfinite(self.cycle_frequency_hz) or self.cycle_frequency_hz <= 0:
            raise ValueError("cycle_frequency_hz must be finite and positive")

        self.foot_ids = np.asarray(session.foot_body_ids, dtype=int)
        self.thorax_id = int(session.thorax_body_id)
        self._ik_data = self.mj.MjData(self.model)
        self._tick_index = 0
        self._last_signal = np.zeros(2, dtype=float)
        self._last_telemetry: dict[str, object] = {}
        self.provenance = self._build_provenance(source_frequency)

    def _validate_config(self) -> None:
        c = self.config
        values = (
            c.active_threshold,
            c.hind_lift_mm,
            c.hind_command_offset_limit_rad,
            c.hind_ik_delta_limit_rad,
            c.hind_ik_damping_mm,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("phase-speed config values must be finite")
        if c.active_threshold < 0 or c.hind_lift_mm < 0:
            raise ValueError("active_threshold and hind_lift_mm must be nonnegative")
        if c.hind_command_offset_limit_rad <= 0 or c.hind_ik_delta_limit_rad <= 0:
            raise ValueError("hind joint correction limits must be positive")
        if c.hind_ik_damping_mm <= 0 or int(c.hind_ik_iterations) < 1:
            raise ValueError("hind IK damping and iteration count must be positive")

    def _validate_actuator_order(self) -> None:
        if len(self.dof_order) != len(self.position_actuator_ids):
            raise ValueError("base controller DOF order and position actuator count differ")
        for channel, (dof, actuator_id) in enumerate(
            zip(self.dof_order, self.position_actuator_ids)
        ):
            name = self.mj.mj_id2name(
                self.model, self.mj.mjtObj.mjOBJ_ACTUATOR, int(actuator_id)
            ) or ""
            expected = f"{dof.parent.name}-{dof.child.name}-{dof.axis.value}-position"
            if not name.endswith(expected):
                raise ValueError(
                    f"position channel {channel} maps to {name!r}; expected DOF suffix {expected!r}"
                )

    def _build_provenance(self, source_frequency: float) -> dict[str, object]:
        package = files("flygym_demo.complex_terrain")
        native_sources: dict[str, dict[str, str]] = {}
        for filename in _SOURCE_MODULES:
            resource = package / filename
            try:
                payload = resource.read_bytes()
                native_sources[filename] = {
                    "resource": str(resource),
                    "sha256": _sha256(payload),
                }
            except (FileNotFoundError, OSError):
                native_sources[filename] = {"resource": str(resource), "sha256": "unavailable"}
        asset = package / "assets/single_steps_untethered.pkl"
        try:
            asset_sha = _sha256(asset.read_bytes())
        except (FileNotFoundError, OSError):
            asset_sha = "unavailable"
        return {
            "controller": "PhaseSpeedGaitController",
            "controller_type": "engineered native-template position-servo gait",
            "native_controller": "HybridTurningController / HybridController",
            "native_source_sha256": native_sources,
            "source_step_asset": str(asset),
            "source_step_asset_sha256": asset_sha,
            "source_cycle_frequency_hz": float(source_frequency),
            "effective_cycle_frequency_hz_at_unit_drive": float(self.cycle_frequency_hz),
            "drive_to_frequency": "signed_intrinsic_frequency_hz = source_cycle_frequency_hz * side_drive",
            "template_amplitude": "intrinsic amplitude target 1.0 for any active side; drive magnitude does not scale joint targets",
            "zero_drive": "neutral position targets, adhesion on when enabled, CPG magnitudes zeroed, phase frozen",
            "inactive_side": "neutral position targets and adhesion on; active side continues its neural frequency command",
            "hind_lift": {
                "enabled": bool(self.config.enable_hind_lift),
                "additional_swing_lift_mm": float(self.config.hind_lift_mm),
                "task": "tarsus5 position only; nominal XY held, target offset from same-tick native foot position along current thorax-local +Z",
                "phase_window": "native swing plus the native adhesion swing extension; sin-squared endpoints",
                "solve_seed": "same-tick native hybrid action; recomputed independently each tick",
                "engineering_offset_limit_rad": float(self.config.hind_command_offset_limit_rad),
                "offset_limit_is_physiological": False,
                "model_joint_limits_modified": False,
                "foot_orientation_constrained": False,
            },
            "external_force_or_root_state_write": False,
            "stimulus_or_desired_yaw_feedback": False,
            "config": asdict(self.config),
        }

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def controller_metadata(self) -> dict[str, object]:
        return deepcopy(self.provenance)

    def telemetry_snapshot(self) -> dict[str, object]:
        return deepcopy(self._last_telemetry)

    def reset(
        self,
        *,
        seed: int | None = None,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
    ) -> None:
        self.base.reset(seed=seed, init_phases=init_phases, init_magnitudes=init_magnitudes)
        self._tick_index = 0
        self._last_signal[:] = 0.0
        self._last_telemetry = {}

    def step(self, descending_signal, obs, *, compact_result: bool = False) -> LocomotionAction:
        signal = np.asarray(descending_signal, dtype=float)
        if signal.shape != (2,) or not np.all(np.isfinite(signal)) or np.any(np.abs(signal) > 1):
            raise ValueError("descending_signal must be two finite values in [-1, 1]")

        network = self.base.cpg_network
        side_active = np.abs(signal) > self.config.active_threshold
        node_signal = np.repeat(signal, 3)
        network.intrinsic_freqs = self.cycle_frequency_hz * node_signal
        network.intrinsic_amps = np.repeat(side_active.astype(float), 3)
        neutral_action = self._neutral_action()

        if not np.any(side_active):
            network.intrinsic_freqs[:] = 0.0
            network.intrinsic_amps[:] = 0.0
            network.curr_magnitudes[:] = 0.0
            self.base.retraction_correction[:] = 0.0
            self.base.stumbling_correction[:] = 0.0
            self.base.retraction_persistence_counter[:] = 0
            self._last_signal = signal.copy()
            if compact_result:
                self._last_telemetry = {}
            else:
                self._last_telemetry = self._telemetry(
                    signal,
                    network.curr_phases.copy(),
                    np.full(6, bool(self.session.config.adhesion_enabled), dtype=bool),
                    {},
                    True,
                    {},
                )
            self._tick_index += 1
            return neutral_action

        # Call the parent implementation directly to retain its CPG,
        # adhesion timing, and sensory corrections without letting the native
        # turning override replace our frequency/amplitude mapping.
        native_action = HybridController.step(self.base, obs)
        targets = np.asarray(native_action.joint_angles, dtype=float).copy()
        adhesion = (
            np.asarray(native_action.adhesion_onoff, dtype=bool).copy()
            if native_action.adhesion_onoff is not None
            else np.ones(6, dtype=bool)
        )
        for side, leg_indices in _SIDE_CHANNELS.items():
            side_index = 0 if side == "left" else 1
            if side_active[side_index]:
                continue
            channels = np.concatenate([self.leg_channels[LEGS[i]] for i in leg_indices])
            targets[channels] = self.neutral_angles[channels]
            adhesion[list(leg_indices)] = bool(self.session.config.adhesion_enabled)

        hind_info: dict[str, object] = {}
        if self.config.enable_hind_lift and self.config.hind_lift_mm > 0:
            compact_hind_lift = getattr(self, "_apply_hind_lift_compact", None)
            if compact_result and callable(compact_hind_lift):
                targets = compact_hind_lift(targets, side_active)
            else:
                targets, hind_info = self._apply_hind_lift(targets, side_active)

        action = LocomotionAction(
            joint_angles=targets,
            adhesion_onoff=(adhesion if self.session.config.adhesion_enabled else None),
        )
        self._last_signal = signal.copy()
        if compact_result:
            self._last_telemetry = {}
        else:
            self._last_telemetry = self._telemetry(
                signal,
                network.curr_phases.copy(),
                adhesion,
                hind_info,
                False,
                self.base.last_info,
            )
        self._tick_index += 1
        return action

    def _neutral_action(self) -> LocomotionAction:
        adhesion = (
            np.ones(6, dtype=bool)
            if self.session.config.adhesion_enabled
            else None
        )
        return LocomotionAction(
            joint_angles=self.neutral_angles.copy(), adhesion_onoff=adhesion
        )

    def _lift_profile(self, leg: str, phase: float) -> float:
        swing_start, swing_end = self.steps.swing_period[leg]
        swing_end = float(swing_end + self.base.swing_extension)
        span = swing_end - float(swing_start)
        if span <= 0:
            return 0.0
        progress = (float(phase) % (2.0 * np.pi) - float(swing_start)) % (2.0 * np.pi)
        if progress >= span:
            return 0.0
        return float(self.config.hind_lift_mm * np.sin(np.pi * progress / span) ** 2)

    def _apply_hind_lift(
        self, targets: np.ndarray, side_active: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        network = self.base.cpg_network
        phases = np.asarray(network.curr_phases, dtype=float)
        phase_template = np.asarray(
            self.steps.get_joint_angles_by_dof_order(
                phases, np.ones(6, dtype=float), self.dof_order
            ),
            dtype=float,
        )
        info: dict[str, object] = {}
        for leg_index in (2, 5):
            side_index = 0 if leg_index < 3 else 1
            if not side_active[side_index]:
                continue
            leg = LEGS[leg_index]
            requested_lift = self._lift_profile(leg, phases[leg_index])
            if requested_lift <= 1e-10:
                info[leg] = {
                    "phase_rad": float(phases[leg_index]),
                    "additional_lift_target_mm": 0.0,
                    "applied": False,
                }
                continue
            channels = self.leg_channels[leg]
            nominal = phase_template[channels]
            starting = targets[channels].copy()
            lower = nominal - self.config.hind_command_offset_limit_rad
            upper = nominal + self.config.hind_command_offset_limit_rad
            start_bounded = np.clip(starting, lower, upper)
            targets[channels] = start_bounded

            self.mj.mj_copyData(self._ik_data, self.model, self.session.data)
            self._ik_data.qpos[self.qadr] = targets
            self.mj.mj_forward(self.model, self._ik_data)
            foot_id = int(self.foot_ids[leg_index])
            thorax_rot = self._ik_data.xmat[self.thorax_id].reshape(3, 3).copy()
            nominal_foot = self._ik_data.xpos[foot_id].copy()
            target_foot = nominal_foot + thorax_rot[:, 2] * requested_lift

            leg_qadr = self.qadr[channels]
            leg_vadr = self.vadr[channels]
            current = self._ik_data.qpos[leg_qadr].copy()
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            delta_clip_count = 0
            box_clip_count = int(np.count_nonzero(np.abs(start_bounded - starting) > 1e-12))
            for _ in range(int(self.config.hind_ik_iterations)):
                self._ik_data.qpos[leg_qadr] = current
                self.mj.mj_fwdPosition(self.model, self._ik_data)
                error = target_foot - self._ik_data.xpos[foot_id]
                if float(np.linalg.norm(error)) <= 1e-4:
                    break
                jacp.fill(0.0)
                jacr.fill(0.0)
                self.mj.mj_jacBody(
                    self.model, self._ik_data, jacp, jacr, foot_id
                )
                jac = jacp[:, leg_vadr]
                regularized = (
                    jac @ jac.T
                    + float(self.config.hind_ik_damping_mm) ** 2 * np.eye(3)
                )
                delta = jac.T @ np.linalg.solve(regularized, error)
                delta_limited = np.clip(
                    delta,
                    -float(self.config.hind_ik_delta_limit_rad),
                    float(self.config.hind_ik_delta_limit_rad),
                )
                delta_clip_count += int(
                    np.count_nonzero(np.abs(delta - delta_limited) > 1e-12)
                )
                candidate = current + delta_limited
                bounded = np.clip(candidate, lower, upper)
                box_clip_count += int(np.count_nonzero(np.abs(candidate - bounded) > 1e-12))
                current = bounded

            self._ik_data.qpos[leg_qadr] = current
            self.mj.mj_fwdPosition(self.model, self._ik_data)
            solved_foot = self._ik_data.xpos[foot_id].copy()
            targets[channels] = current
            info[leg] = {
                "phase_rad": float(phases[leg_index]),
                "additional_lift_target_mm": float(requested_lift),
                "nominal_foot_world_mm": nominal_foot.tolist(),
                "target_foot_world_mm": target_foot.tolist(),
                "solved_foot_world_mm": solved_foot.tolist(),
                "achieved_displacement_mm": float(np.linalg.norm(solved_foot - nominal_foot)),
                "position_residual_mm": float(np.linalg.norm(target_foot - solved_foot)),
                "joint_delta_from_same_phase_template_rad": (current - nominal).tolist(),
                "max_abs_template_offset_rad": float(np.max(np.abs(current - nominal))),
                "engineering_box_clip_count": int(box_clip_count),
                "ik_delta_clip_count": int(delta_clip_count),
                "orientation_constrained": False,
                "applied": True,
            }
        return targets, info

    def _telemetry(
        self,
        signal: np.ndarray,
        phases: np.ndarray,
        adhesion: np.ndarray,
        hind_info: dict[str, object],
        neutral_hold: bool,
        hybrid_info: dict,
    ) -> dict[str, object]:
        side_active = np.abs(signal) > self.config.active_threshold
        frequencies = self.cycle_frequency_hz * np.repeat(signal, 3)
        return {
            "tick_index": int(self._tick_index),
            "descending_signal_lr": signal.tolist(),
            "side_active_lr": side_active.tolist(),
            "commanded_intrinsic_frequency_hz_by_leg": frequencies.tolist(),
            "intrinsic_amplitude_target_by_leg": np.repeat(side_active.astype(float), 3).tolist(),
            "cpg_phase_rad_by_leg": np.asarray(phases, dtype=float).tolist(),
            "cpg_magnitude_by_leg": self.base.cpg_network.curr_magnitudes.astype(float).tolist(),
            "adhesion_on_by_leg": np.asarray(adhesion, dtype=bool).tolist(),
            "neutral_hold": bool(neutral_hold),
            "hind_lift": _jsonable(hind_info),
            "hybrid_controller": _jsonable(hybrid_info),
            "interpretation": "position-servo controller with native joint templates; not a muscle or connectome actuator model",
        }


__all__ = ["PhaseSpeedGaitConfig", "PhaseSpeedGaitController"]
