#!/usr/bin/env python3
"""Phase-speed gait with the hind clearance peak aligned to native FK.

V3 inherits the frozen V2 phase-speed controller and changes only the
time-profile of the hind world-Z clearance correction.  It remains an
engineered position-servo controller, not a muscle or connectome model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from flygym_phase_speed_gait_v2 import (
    PhaseSpeedGaitConfig as _PhaseSpeedGaitConfigV2,
    PhaseSpeedGaitController as _PhaseSpeedGaitControllerV2,
    _STANCE_SOURCE,
    _jsonable,
)


_PHASE_PEAK_SOURCE = {
    "diagnosis_json": "/data/artifacts/by-project/malecns-gpu/engineering/flygym-phase-speed-repair-20261006-v1/diagnosis/hind_swing_workspace.json",
    "diagnosis_json_sha256": "8d9e4746aff256835fcbb9def3a886d55b3e32bca927c7caec30b3f619514d03",
    "raw_template_source": "pinned PreprogrammedSteps full-magnitude FK over saved legacy-rest model and initial pose; swing marker peak selected relative to template stance median",
    "raw_peak_phase_rad_by_leg": {
        "lh": 0.29670597283903605,
        "rh": 0.29670597283903605,
    },
    "raw_peak_marker_rise_mm_by_leg": {
        "lh": 0.09818957642766402,
        "rh": 0.09818957642766402,
    },
    "compiled_model_xml_sha256": "c4db94617aae9f138af2526fb9fa751f99ed230602e4e7e5c9670c1bf8421631",
    "full_drive_trajectory_npz_sha256": "f980e5c95f51e41ad7b74d1288da3076f7624472a5cf535bc30f3992865234cd",
    "preprogrammed_source_sha256": "ad945cee5af607e24f1a17ada0b8becd622c1fabebbb9f4f35e49dc7fd69e97f",
    "preprogrammed_steps_asset_sha256": "1e5b28bb6b3f50ac95a04773ea37af28d05e26d03bd90fdba57fa1b3eacfaf8c",
    "scope": "phase selection from raw hind FK only; no stimulus, yaw, stripe, or physical-outcome fit",
}


@dataclass(frozen=True)
class PhaseSpeedGaitConfig(_PhaseSpeedGaitConfigV2):
    """V3 config; all V2 fields retain their defaults and meanings."""

    peak_phase_rad: float = _PHASE_PEAK_SOURCE["raw_peak_phase_rad_by_leg"]["lh"]


class PhaseSpeedGaitController(_PhaseSpeedGaitControllerV2):
    """Frozen V2 controller with an asymmetric, native-peak clearance profile."""

    def __init__(self, session, config: PhaseSpeedGaitConfig | None = None):
        selected = config or PhaseSpeedGaitConfig()
        if not np.isfinite(float(selected.peak_phase_rad)):
            raise ValueError("peak_phase_rad must be finite")
        if not 0.0 <= float(selected.peak_phase_rad) < 2.0 * np.pi:
            raise ValueError("peak_phase_rad must be normalized to [0, 2*pi)")
        super().__init__(session, selected)

        peak = float(selected.peak_phase_rad)
        self._clearance_profile: dict[str, dict[str, float]] = {}
        for leg in ("lh", "rh"):
            start_raw, swing_end_raw = self.steps.swing_period[leg]
            start = float(start_raw)
            end = float(swing_end_raw + self.base.swing_extension)
            span = end - start
            peak_progress = (peak - start) % (2.0 * np.pi)
            if not 0.0 < peak_progress < span:
                raise ValueError(
                    f"peak_phase_rad={peak:.9g} must be strictly inside the "
                    f"{leg} clearance interval ({start:.9g}, {end:.9g}) rad"
                )
            self._clearance_profile[leg] = {
                "start_phase_rad": start,
                "peak_phase_rad": start + peak_progress,
                "end_phase_rad": end,
                "swing_span_rad": span,
                "peak_progress_rad": peak_progress,
            }

        parent_path = Path(__file__).with_name("flygym_phase_speed_gait_v2.py")
        try:
            parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        except OSError:
            parent_sha = "unavailable"
        prior_v1_sha = str(self.provenance.get("phase_speed_parent_sha256", "unavailable"))
        self.provenance.update({
            "controller": "PhaseSpeedGaitControllerV3",
            "controller_version": "v3",
            "phase_speed_parent_module": "flygym_phase_speed_gait_v2.py",
            "phase_speed_parent_sha256": parent_sha,
            "phase_speed_v1_parent_sha256": prior_v1_sha,
            "config": asdict(selected),
            "clearance_envelope": {
                "change_from_v2": "only the hind world-Z correction phase envelope peak",
                "profile": "sin(pi/2*u)^2 from native swing start to peak; cos(pi/2*u)^2 from peak to native swing-plus-adhesion-off-extension end",
                "boundary_continuity": "envelope and first derivative are zero at both event boundaries; value is one and first derivative zero at the peak",
                "peak_phase_rad": peak,
                "peak_phase_by_leg_rad": {
                    leg: float(values["peak_phase_rad"])
                    for leg, values in self._clearance_profile.items()
                },
                "interval_by_leg": deepcopy(self._clearance_profile),
                "raw_fk_peak_source": deepcopy(_PHASE_PEAK_SOURCE),
                "position_target_rule": "inherited unchanged from V2: deltaZ=envelope*max(0, loaded_stance_marker_world_z+clearance_peak_mm-nominal_foot_world_z)",
                "stance_marker_source": deepcopy(_STANCE_SOURCE),
                "clearance_peak_mm": float(selected.clearance_peak_mm),
                "position_only_ik": True,
                "engineering_joint_offset_box_rad": float(selected.command_offset_limit_rad),
                "orientation_constraint_added": False,
                "phase_speed_zero_drive_and_sensory_corrections": "inherited unchanged from V2",
                "external_force_or_root_state_write": False,
                "stimulus_or_desired_yaw_feedback": False,
            },
        })

    @staticmethod
    def _profile_value(progress: float, peak_progress: float, span: float) -> float:
        """C1 piecewise half-wave envelope, with the peak before swing-window mid."""
        if progress <= peak_progress:
            u = progress / peak_progress
            return float(np.sin(0.5 * np.pi * u) ** 2)
        tail = span - peak_progress
        u = (progress - peak_progress) / tail
        return float(np.cos(0.5 * np.pi * u) ** 2)

    def _apply_hind_lift(
        self, targets: np.ndarray, side_active: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
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
            marker_world_z = markers[leg]
            clearance_goal_z = marker_world_z + float(c.clearance_peak_mm)
            required_peak_delta_z = max(0.0, clearance_goal_z - float(nominal_foot[2]))
            delta_z = envelope * required_peak_delta_z
            if delta_z <= 1e-10:
                info[leg] = {
                    "phase_rad": phase,
                    "adhesion_on": False,
                    "swing_envelope": envelope,
                    "peak_phase_rad": float(profile["peak_phase_rad"]),
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
                "joint_delta_from_tick_native_target_rad": (current - nominal_q).tolist(),
                "max_abs_tick_target_offset_rad": float(np.max(np.abs(current - nominal_q))),
                "engineering_box_clip_count": int(box_clips),
                "ik_delta_clip_count": int(ik_delta_clips),
                "xy_target_shift_mm": (solved_foot[:2] - nominal_foot[:2]).tolist(),
                "foot_orientation_constrained": False,
                "applied": True,
            }
        return targets, info


PhaseSpeedGaitV3Config = PhaseSpeedGaitConfig
PhaseSpeedGaitV3Controller = PhaseSpeedGaitController

__all__ = [
    "PhaseSpeedGaitConfig",
    "PhaseSpeedGaitController",
    "PhaseSpeedGaitV3Config",
    "PhaseSpeedGaitV3Controller",
]
