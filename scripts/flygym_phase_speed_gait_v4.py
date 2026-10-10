#!/usr/bin/env python3
"""Phase-speed gait with position-priority hind-foot orientation refinement.

V4 inherits the frozen V3 phase-speed behavior and clearance target. For a
hind swing target it first computes the exact V3 six-iteration position-only
solution, then uses redundant leg DOFs to move the tarsus5 orientation toward
the same-tick raw named native-template orientation. The refinement is bounded
and falls back to the exact V3 solution if it cannot preserve the Cartesian
position tolerance while improving orientation.

This is an engineered position-servo controller, not a muscle or connectome
actuator reconstruction. The ±joint command box is an engineering bound, not
a biological or measured physiological joint range.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from flygym_phase_speed_gait_v3 import (
    PhaseSpeedGaitConfig,
    PhaseSpeedGaitController as _PhaseSpeedGaitControllerV3,
)
from flygym_pose_priority_ik import (
    DEFAULT_CONFIG as _POSE_PRIORITY_CONFIG,
    refine_orientation_at_position,
)


_EXPECTED_V3_SHA256 = "a2eadd1d85f76f74cd0fa7510b5614b9357f610027689c47e236b809070dd3dd"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhaseSpeedGaitController(_PhaseSpeedGaitControllerV3):
    """Frozen V3 behavior plus bounded raw-foot-R nullspace refinement."""

    def __init__(self, session, config: PhaseSpeedGaitConfig | None = None):
        v3_path = Path(__file__).with_name("flygym_phase_speed_gait_v3.py")
        v3_sha = _sha256(v3_path)
        if v3_sha != _EXPECTED_V3_SHA256:
            raise RuntimeError("V3 parent source differs from its frozen candidate")
        super().__init__(session, config)
        helper_path = Path(__file__).with_name("flygym_pose_priority_ik.py")
        helper_sha = _sha256(helper_path)
        self.provenance.update({
            "controller": "PhaseSpeedGaitControllerV4",
            "controller_version": "v4-static-candidate",
            "phase_speed_parent_module": "flygym_phase_speed_gait_v3.py",
            "phase_speed_parent_sha256": v3_sha,
            "pose_priority_helper_module": helper_path.name,
            "pose_priority_helper_sha256": helper_sha,
            "position_priority_refinement": {
                "scope": "hind tarsus5 swing only; all V3 phase, XY, world-Z, stance-anchor, clearance, CPG, zero-drive, and control-gate behavior is inherited unchanged",
                "raw_orientation_target": "same private FK pose with same-tick native named full-magnitude template joint targets and unchanged root/passive qpos",
                "position_target": "exact unchanged V3 world XYZ target",
                "solver": "SVD position-Jacobian nullspace orientation descent, bounded line search, and position retraction",
                "fallback": "return the exact V3 six-iteration position-only solution if the position slack is exceeded or orientation does not improve",
                "joint_box": "same V3 per-tick nominal-target-centered engineering command box; not a biological or measured physiological range",
                "no_physics_step": True,
                "no_root_state_write": True,
                "no_external_force": True,
                "stimulus_or_desired_yaw_feedback": False,
                "helper_config": {
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in vars(_POSE_PRIORITY_CONFIG).items()
                },
            },
        })

    def _apply_hind_lift(
        self, targets: np.ndarray, side_active: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply the unchanged V3 XYZ lift, then redundant-DOF R refinement."""
        c = self.phase_speed_config
        phases = np.asarray(self.base.cpg_network.curr_phases, dtype=float)
        markers = {
            "lh": float(c.lh_stance_marker_world_z_mm),
            "rh": float(c.rh_stance_marker_world_z_mm),
        }
        info: dict[str, object] = {
            "envelope_profile": "piecewise_sin2_rise_cos2_fall",
            "peak_phase_rad": float(c.peak_phase_rad),
            "profile_by_leg": deepcopy(self._clearance_profile),
            "pose_priority": {
                "enabled": True,
                "algorithm": "bounded SVD position-nullspace orientation descent with line search and XYZ retraction",
                "raw_orientation_target": "same-tick named native template FK at saved root/passive pose",
                "fallback": "exact V3 position-only solution on position-slack violation or no orientation gain",
            },
        }

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

            profile = self._clearance_profile[leg]
            start = float(profile["start_phase_rad"])
            end = float(profile["end_phase_rad"])
            span = float(profile["swing_span_rad"])
            peak_progress = float(profile["peak_progress_rad"])
            progress = (phase % (2.0 * np.pi) - start) % (2.0 * np.pi)
            if span <= 0.0 or progress >= span:
                info[leg] = {
                    "phase_rad": phase,
                    "adhesion_on": False,
                    "applied": False,
                    "reason": "outside_smooth_lift_window",
                }
                continue
            envelope = self._profile_value(progress, peak_progress, span)

            channels = self.leg_channels[leg]
            nominal_q = np.asarray(targets[channels], dtype=float).copy()
            self.mj.mj_copyData(self._ik_data, self.model, self.session.data)
            self._ik_data.qpos[self.qadr] = targets
            self.mj.mj_forward(self.model, self._ik_data)
            foot_id = int(self.foot_ids[leg_index])
            nominal_foot = self._ik_data.xpos[foot_id].copy()
            raw_template_rotation = self._ik_data.xmat[foot_id].reshape(3, 3).copy()
            marker_world_z = markers[leg]
            clearance_goal_z = marker_world_z + float(c.clearance_peak_mm)
            required_peak_delta_z = max(0.0, clearance_goal_z - float(nominal_foot[2]))
            delta_z = envelope * required_peak_delta_z
            if delta_z <= 1e-10:
                info[leg] = {
                    "phase_rad": phase,
                    "adhesion_on": False,
                    "swing_envelope": float(envelope),
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
            current = nominal_q.copy()
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            ik_delta_clips = 0
            box_clips = 0

            # Frozen V3 baseline: exactly its native-target seed, six iterations,
            # DLS damping, per-step clip, and per-tick engineering joint box.
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
                regularized = jac @ jac.T + float(c.ik_damping_mm) ** 2 * np.eye(3)
                delta = jac.T @ np.linalg.solve(regularized, error)
                clipped_delta = np.clip(
                    delta,
                    -float(c.ik_delta_limit_rad),
                    float(c.ik_delta_limit_rad),
                )
                ik_delta_clips += int(
                    np.count_nonzero(np.abs(delta - clipped_delta) > 1e-12)
                )
                proposed = current + clipped_delta
                bounded = np.clip(proposed, lower, upper)
                box_clips += int(np.count_nonzero(np.abs(proposed - bounded) > 1e-12))
                current = bounded

            baseline_q = current.copy()
            self._ik_data.qpos[leg_qadr] = baseline_q
            self.mj.mj_fwdPosition(self.model, self._ik_data)
            baseline_foot = self._ik_data.xpos[foot_id].copy()
            baseline_rotation = self._ik_data.xmat[foot_id].reshape(3, 3).copy()
            baseline_position_residual = float(np.linalg.norm(target_foot - baseline_foot))
            baseline_rotation_error_deg = float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(
                    raw_template_rotation @ baseline_rotation.T
                ).as_rotvec()
            )))

            solved_q, priority_info = refine_orientation_at_position(
                self.mj,
                self.model,
                self._ik_data,
                foot_id,
                leg_qadr,
                leg_vadr,
                nominal_q,
                baseline_q,
                target_foot,
                raw_template_rotation,
                box_rad=float(c.command_offset_limit_rad),
            )
            self._ik_data.qpos[leg_qadr] = solved_q
            self.mj.mj_fwdPosition(self.model, self._ik_data)
            solved_foot = self._ik_data.xpos[foot_id].copy()
            final_rotation = self._ik_data.xmat[foot_id].reshape(3, 3).copy()
            final_delta = solved_q - nominal_q
            targets[channels] = solved_q
            info[leg] = {
                "phase_rad": phase,
                "adhesion_on": False,
                "swing_envelope": float(envelope),
                "peak_phase_rad": float(profile["peak_phase_rad"]),
                "profile_progress_rad": float(progress),
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
                "v3_position_only_position_residual_mm": baseline_position_residual,
                "v3_position_only_orientation_error_deg": baseline_rotation_error_deg,
                "joint_delta_from_tick_native_target_rad": final_delta.tolist(),
                "max_abs_tick_target_offset_rad": float(np.max(np.abs(final_delta))),
                "engineering_box_clip_count": int(box_clips),
                "ik_delta_clip_count": int(ik_delta_clips),
                "xy_target_shift_mm": (solved_foot[:2] - nominal_foot[:2]).tolist(),
                "raw_template_orientation_constrained_by_priority_solver": True,
                "pose_priority": priority_info,
                "raw_template_orientation_matrix_world": raw_template_rotation.tolist(),
                "solved_orientation_matrix_world": final_rotation.tolist(),
                "applied": True,
            }
        return targets, info


PhaseSpeedGaitV4Config = PhaseSpeedGaitConfig
PhaseSpeedGaitV4Controller = PhaseSpeedGaitController

__all__ = [
    "PhaseSpeedGaitConfig",
    "PhaseSpeedGaitController",
    "PhaseSpeedGaitV4Config",
    "PhaseSpeedGaitV4Controller",
]
