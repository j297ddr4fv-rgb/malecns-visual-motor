#!/usr/bin/env python3
"""Phase-speed gait with hind swing clearance anchored to loaded stance Z.

V2 keeps the frozen phase-speed mapping and changes only the hind correction:
the tarsus5 world-Z target is referenced to the measured flat-floor stance
marker.  It remains a position-servo engineering controller, not a muscle or
connectome actuator model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from flygym_demo.complex_terrain.common import LocomotionAction
from flygym_phase_speed_gait import (
    PhaseSpeedGaitConfig as _PhaseSpeedGaitConfigV1,
    PhaseSpeedGaitController as _PhaseSpeedGaitControllerV1,
    _jsonable,
)


_STANCE_SOURCE = {
    "condition": "legacy_rest, constant [1,1], seed 1729, 7.407407 Hz",
    "analysis_window_s": [0.5, 1.5],
    "sample_basis": "loaded tarsus5 body-origin world Z, per-leg median, Fz/bodyweight > 0.01 independent of adhesion command",
    "lh_stance_marker_world_z_mm": 0.09638690948486328,
    "rh_stance_marker_world_z_mm": 0.09632554650306702,
    "trajectory_npz": "/data/artifacts/by-project/malecns-gpu/engineering/flygym-cartesian-repair-20261006-v1/legacy-rest-probe/attempt-2/legacy_rest/trajectory.npz",
    "trajectory_npz_sha256": "f980e5c95f51e41ad7b74d1288da3076f7624472a5cf535bc30f3992865234cd",
    "loaded_marker_anchor_json": "/data/artifacts/by-project/malecns-gpu/engineering/flygym-phase-speed-repair-20261006-v1/diagnosis/settled_loaded_marker_anchors.json",
    "loaded_marker_anchor_json_sha256": "f7331370fd1f5fe27ab7bda3111faadc0b6b828e8a318e0e416487f6686fa27f",
    "compiled_model_xml_sha256": "c4db94617aae9f138af2526fb9fa751f99ed230602e4e7e5c9670c1bf8421631",
    "stance_workspace_diagnosis_json": "/data/artifacts/by-project/malecns-gpu/engineering/flygym-phase-speed-repair-20261006-v1/diagnosis/hind_swing_workspace.json",
    "stance_workspace_diagnosis_sha256": "8d9e4746aff256835fcbb9def3a886d55b3e32bca927c7caec30b3f619514d03",
}


@dataclass(frozen=True)
class PhaseSpeedGaitConfig:
    """V2 parameters; clearance is peak height above measured stance marker."""

    cycle_frequency_hz: float | None = None
    active_threshold: float = 1e-6
    enable_hind_clearance: bool = True
    clearance_peak_mm: float = 0.08
    lh_stance_marker_world_z_mm: float = _STANCE_SOURCE["lh_stance_marker_world_z_mm"]
    rh_stance_marker_world_z_mm: float = _STANCE_SOURCE["rh_stance_marker_world_z_mm"]
    command_offset_limit_rad: float = 0.35
    ik_delta_limit_rad: float = 0.04
    ik_damping_mm: float = 0.025
    ik_iterations: int = 6


class PhaseSpeedGaitController(_PhaseSpeedGaitControllerV1):
    """Frozen phase-speed driver with stance-anchored hind swing clearance."""

    def __init__(self, session, config: PhaseSpeedGaitConfig | None = None):
        self.phase_speed_config = config or PhaseSpeedGaitConfig()
        self._validate_v2_config(self.phase_speed_config)
        c = self.phase_speed_config
        # The parent uses this internal compatibility config to keep its
        # phase-speed, zero-drive, and telemetry path unchanged.  Its
        # hind_lift_mm field is only a positive switch; V2's actual target is
        # the separately named absolute stance-relative clearance_peak_mm.
        parent_config = _PhaseSpeedGaitConfigV1(
            cycle_frequency_hz=c.cycle_frequency_hz,
            active_threshold=c.active_threshold,
            enable_hind_lift=c.enable_hind_clearance,
            hind_lift_mm=(c.clearance_peak_mm if c.enable_hind_clearance else 0.0),
            hind_command_offset_limit_rad=c.command_offset_limit_rad,
            hind_ik_delta_limit_rad=c.ik_delta_limit_rad,
            hind_ik_damping_mm=c.ik_damping_mm,
            hind_ik_iterations=c.ik_iterations,
        )
        super().__init__(session, parent_config)
        self.provenance.pop("hind_lift", None)
        self.provenance.pop("config", None)
        local_path = Path(__file__).with_name("flygym_phase_speed_gait.py")
        try:
            phase_speed_v1_sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
        except OSError:
            phase_speed_v1_sha = "unavailable"
        self.provenance.update({
            "controller": "PhaseSpeedGaitControllerV2",
            "controller_version": "v2",
            "phase_speed_parent_module": "flygym_phase_speed_gait.py",
            "phase_speed_parent_sha256": phase_speed_v1_sha,
            "config": asdict(c),
            "hind_clearance": {
                "enabled": bool(c.enable_hind_clearance),
                "clearance_peak_mm": float(c.clearance_peak_mm),
                "clearance_peak_is_additive": False,
                "stance_marker_world_z_mm": {
                    "lh": float(c.lh_stance_marker_world_z_mm),
                    "rh": float(c.rh_stance_marker_world_z_mm),
                },
                "stance_marker_source": deepcopy(_STANCE_SOURCE),
                "clearance_cycle_gate_mm": 0.05,
                "engineering_margin_above_gate_mm": float(c.clearance_peak_mm - 0.05),
                "clearance_peak_rationale": "0.08 mm peak = 0.05 mm measured-cycle gate + 0.03 mm margin so the swing envelope can exceed the gate for measurable duration",
                "clearance_reference": "absolute world-Z target above the per-leg loaded stance marker median",
                "target_rule": "deltaZ = sin(pi*u)^2 * max(0, stance_marker_world_z + clearance_peak_mm - nominal_foot_world_z)",
                "phase_window": "hind native swing plus native adhesion-off extension; exact zero during adhesion-on and smooth zero at both edges",
                "controlled_marker": "hind tarsus5 body-origin world Z",
                "target_xy": "same-tick native target FK world XY held fixed",
                "foot_orientation_constrained": False,
                "joint_solve_seed": "same-tick native hybrid position target; solved from scratch each tick",
                "joint_offset_box_rad": float(c.command_offset_limit_rad),
                "offset_box_is_physiological": False,
                "model_joint_limits_modified": False,
                "private_fk_ik_data_only": True,
            },
        })

    @staticmethod
    def _validate_v2_config(c: PhaseSpeedGaitConfig) -> None:
        finite = (
            c.active_threshold,
            c.clearance_peak_mm,
            c.lh_stance_marker_world_z_mm,
            c.rh_stance_marker_world_z_mm,
            c.command_offset_limit_rad,
            c.ik_delta_limit_rad,
            c.ik_damping_mm,
        )
        if not all(np.isfinite(float(v)) for v in finite):
            raise ValueError("phase-speed V2 config values must be finite")
        if c.active_threshold < 0 or c.clearance_peak_mm < 0:
            raise ValueError("active_threshold and clearance_peak_mm must be nonnegative")
        if c.command_offset_limit_rad <= 0 or c.ik_delta_limit_rad <= 0:
            raise ValueError("IK joint correction limits must be positive")
        if c.ik_damping_mm <= 0 or int(c.ik_iterations) < 1:
            raise ValueError("IK damping and iteration count must be positive")

    def controller_metadata(self) -> dict[str, object]:
        return deepcopy(self.provenance)

    def _telemetry(
        self,
        signal: np.ndarray,
        phases: np.ndarray,
        adhesion: np.ndarray,
        hind_info: dict[str, object],
        neutral_hold: bool,
        hybrid_info: dict,
    ) -> dict[str, object]:
        snapshot = super()._telemetry(
            signal, phases, adhesion, {}, neutral_hold, hybrid_info
        )
        snapshot.pop("hind_lift", None)
        snapshot["hind_clearance"] = _jsonable(hind_info)
        snapshot["clearance_peak_mm"] = float(self.phase_speed_config.clearance_peak_mm)
        snapshot["clearance_cycle_gate_mm"] = 0.05
        snapshot["engineering_margin_above_gate_mm"] = float(
            self.phase_speed_config.clearance_peak_mm - 0.05
        )
        snapshot["stance_marker_world_z_mm"] = {
            "lh": float(self.phase_speed_config.lh_stance_marker_world_z_mm),
            "rh": float(self.phase_speed_config.rh_stance_marker_world_z_mm),
        }
        return snapshot

    def _apply_hind_lift(
        self, targets: np.ndarray, side_active: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        c = self.phase_speed_config
        phases = np.asarray(self.base.cpg_network.curr_phases, dtype=float)
        markers = {
            "lh": float(c.lh_stance_marker_world_z_mm),
            "rh": float(c.rh_stance_marker_world_z_mm),
        }
        info: dict[str, object] = {}
        for leg_index in (2, 5):
            leg = ("lf", "lm", "lh", "rf", "rm", "rh")[leg_index]
            side_index = 0 if leg_index < 3 else 1
            if not side_active[side_index]:
                continue

            phase = float(phases[leg_index])
            if self.base._get_adhesion_onoff(leg, phase):
                info[leg] = {
                    "phase_rad": phase,
                    "adhesion_on": True,
                    "applied": False,
                    "reason": "stance_path_unchanged",
                }
                continue

            swing_start, swing_end = self.steps.swing_period[leg]
            start = float(swing_start)
            end = float(swing_end + self.base.swing_extension)
            span = end - start
            progress = (phase % (2.0 * np.pi) - start) % (2.0 * np.pi)
            if span <= 0 or progress >= span:
                info[leg] = {
                    "phase_rad": phase,
                    "adhesion_on": False,
                    "applied": False,
                    "reason": "outside_smooth_lift_window",
                }
                continue
            envelope = float(np.sin(np.pi * progress / span) ** 2)

            channels = self.leg_channels[leg]
            nominal_q = np.asarray(targets[channels], dtype=float).copy()
            self.mj.mj_copyData(self._ik_data, self.model, self.session.data)
            self._ik_data.qpos[self.qadr] = targets
            self.mj.mj_forward(self.model, self._ik_data)
            foot_id = int(self.foot_ids[leg_index])
            nominal_foot = self._ik_data.xpos[foot_id].copy()
            marker_world_z = markers[leg]
            clearance_goal_z = marker_world_z + float(c.clearance_peak_mm)
            required_peak_delta_z = max(0.0, clearance_goal_z - float(nominal_foot[2]))
            delta_z = envelope * required_peak_delta_z
            if delta_z <= 1e-10:
                info[leg] = {
                    "phase_rad": phase,
                    "adhesion_on": False,
                    "swing_envelope": envelope,
                    "stance_marker_world_z_mm": marker_world_z,
                    "clearance_peak_mm": float(c.clearance_peak_mm),
                    "nominal_foot_world_z_mm": float(nominal_foot[2]),
                    "target_world_z_mm": float(nominal_foot[2]),
                    "target_delta_z_mm": 0.0,
                    "applied": False,
                    "reason": "nominal_target_already_above_stance_clearance_target",
                }
                continue

            target_foot = nominal_foot.copy()
            target_foot[2] += delta_z
            lower = nominal_q - float(c.command_offset_limit_rad)
            upper = nominal_q + float(c.command_offset_limit_rad)
            leg_qadr = self.qadr[channels]
            leg_vadr = self.vadr[channels]
            current = self._ik_data.qpos[leg_qadr].copy()
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            ik_delta_clips = 0
            box_clips = 0
            for _ in range(int(c.ik_iterations)):
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
                    jac @ jac.T + float(c.ik_damping_mm) ** 2 * np.eye(3)
                )
                delta = jac.T @ np.linalg.solve(regularized, error)
                clipped_delta = np.clip(
                    delta, -float(c.ik_delta_limit_rad), float(c.ik_delta_limit_rad)
                )
                ik_delta_clips += int(
                    np.count_nonzero(np.abs(delta - clipped_delta) > 1e-12)
                )
                candidate = current + clipped_delta
                bounded = np.clip(candidate, lower, upper)
                box_clips += int(np.count_nonzero(np.abs(candidate - bounded) > 1e-12))
                current = bounded

            self._ik_data.qpos[leg_qadr] = current
            self.mj.mj_fwdPosition(self.model, self._ik_data)
            solved_foot = self._ik_data.xpos[foot_id].copy()
            targets[channels] = current
            info[leg] = {
                "phase_rad": phase,
                "adhesion_on": False,
                "swing_envelope": envelope,
                "stance_marker_world_z_mm": marker_world_z,
                "clearance_peak_mm": float(c.clearance_peak_mm),
                "clearance_goal_world_z_mm": clearance_goal_z,
                "nominal_foot_world_mm": nominal_foot.tolist(),
                "target_foot_world_mm": target_foot.tolist(),
                "solved_foot_world_mm": solved_foot.tolist(),
                "nominal_deficit_to_peak_clearance_mm": float(required_peak_delta_z),
                "target_delta_z_mm": float(delta_z),
                "achieved_clearance_above_stance_marker_mm": float(
                    solved_foot[2] - marker_world_z
                ),
                "position_residual_mm": float(np.linalg.norm(target_foot - solved_foot)),
                "joint_delta_from_tick_native_target_rad": (current - nominal_q).tolist(),
                "max_abs_tick_target_offset_rad": float(np.max(np.abs(current - nominal_q))),
                "engineering_box_clip_count": int(box_clips),
                "ik_delta_clip_count": int(ik_delta_clips),
                "xy_target_shift_mm": (solved_foot[:2] - nominal_foot[:2]).tolist(),
                "foot_orientation_constrained": False,
                "applied": True,
            }
        return targets, info


PhaseSpeedGaitV2Config = PhaseSpeedGaitConfig
PhaseSpeedGaitV2Controller = PhaseSpeedGaitController

__all__ = [
    "PhaseSpeedGaitConfig",
    "PhaseSpeedGaitController",
    "PhaseSpeedGaitV2Config",
    "PhaseSpeedGaitV2Controller",
]
