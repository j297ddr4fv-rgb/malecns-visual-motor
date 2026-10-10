#!/usr/bin/env python3
"""Independent AP endpoint-gesture prototype built on FlyGym V4.1.

This is an engineered kinematic position-target controller. It applies
body-local anterior-posterior endpoint displacements to the named
PreprogrammedSteps foot path, keeps native adhesion timing, retains the V4.1
hind swing-clearance profile, and uses the existing pose-priority IK for hind
tarsus orientation. It does not represent neural forces or establish
biological equivalence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from flygym_phase_speed_gait_v4_1 import (
    PhaseSpeedGaitConfig as _V41Config,
    PhaseSpeedGaitController as _V41Controller,
)
from flygym_pose_priority_ik_v2 import (
    _qpos_quaternions_are_normalization_stable,
    refine_orientation_at_position,
)


LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
SIDES = ("left", "right")
_TAU = 2.0 * np.pi


@dataclass(frozen=True)
class APEndpointGestureConfig(_V41Config):
    """Opt-in endpoint targets; distances are millimetres, commands in [-1,1].

    The default commands are zero so constructing the prototype preserves the
    inherited V4.1 behavior. The predeclared independent perturbation sizes
    are both 0.05 mm. A positive DNa02-like command moves only its side's AEP
    posteriorly; a positive DNg13-like command moves the contralateral AEP
    anteriorly and PEP posteriorly by equal amounts.
    """

    dna02_command: float = 0.0
    dng13_command: float = 0.0
    dna02_soma_side: str = "left"
    dng13_soma_side: str = "left"
    dna02_aep_posterior_shift_mm: float = 0.05
    dng13_endpoint_shift_mm: float = 0.05
    endpoint_phase_samples: int = 512


def _smooth5(u: float) -> float:
    x = float(np.clip(u, 0.0, 1.0))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _phase_gap(a: float, b: float) -> float:
    return float((float(a) - float(b) + np.pi) % _TAU - np.pi)


class APEndpointGestureController(_V41Controller):
    """V4.1 with an opt-in six-leg AP endpoint deformation.

    ``descending_signal`` remains the inherited left/right CPG drive. The
    fixed DNa02-like and DNg13-like command fields are deliberately separate:
    they are hypothetical mechanical conditions, not neural readout values.
    """

    def __init__(self, session, config: APEndpointGestureConfig | None = None):
        selected = config or APEndpointGestureConfig()
        self._validate_gesture_config(selected)
        super().__init__(session, selected)
        self.gesture_config = selected
        self.reuse_prepared_fk = False
        self._leg_qadr = {
            leg: self.qadr[self.leg_channels[leg]] for leg in LEGS
        }
        self._leg_vadr = {
            leg: self.vadr[self.leg_channels[leg]] for leg in LEGS
        }
        self.endpoint_catalogue = self._build_endpoint_catalogue(
            int(selected.endpoint_phase_samples)
        )
        self.provenance.update({
            "controller": "APEndpointGestureController",
            "controller_version": "prototype-v1-static-candidate",
            "classification": "engineered six-leg AP Cartesian endpoint gesture over native joint template",
            "base_controller": "PhaseSpeedGaitControllerV4_1",
            "gesture_commands": {
                "dna02_command": float(selected.dna02_command),
                "dng13_command": float(selected.dng13_command),
                "dna02_soma_side": selected.dna02_soma_side,
                "dng13_soma_side": selected.dng13_soma_side,
                "dna02_target_side": selected.dna02_soma_side,
                "dng13_target_side": self._opposite(selected.dng13_soma_side),
                "dna02_aep_posterior_shift_mm": float(selected.dna02_aep_posterior_shift_mm),
                "dng13_aep_forward_and_pep_backward_mm": float(selected.dng13_endpoint_shift_mm),
                "command_is_neural_readout": False,
                "fit_to_visual_trial_or_yaw": False,
            },
            "endpoint_definition": "AEP=maximum and PEP=minimum of tarsus5 body-origin position projected onto thorax-local +X over a full native template cycle; measured phase extrema, not assumed adhesion boundaries",
            "endpoint_profile": "periodic quintic smoothstep from PEP position to AEP position and back; endpoint shifts are retained through native adhesion transitions and are not zeroed at contact events",
            "vertical_clearance": "inherited V4.1 hind swing world-Z correction and stance markers; unchanged formula/phase window",
            "hind_orientation": "existing bounded pose-priority IK targeting same-tick pre-gesture tarsus5 rotation while retaining position tolerance",
            "front_middle_orientation": "not constrained; position-only 3x7 IK",
            "native_adhesion_schedule": "unchanged; adhesion flags are recorded separately from measured ground-contact events",
            "actual_contact_timing": "not available in controller input; must be joined from runner's saved contact trace",
            "static_endpoint_catalogue": deepcopy(self.endpoint_catalogue),
            "physics_step_calls": 0,
            "live_state_write": False,
            "external_wrench": False,
            "config": asdict(selected),
        })

    @staticmethod
    def _opposite(side: str) -> str:
        return "right" if side == "left" else "left"

    @property
    def reuse_prepared_fk(self) -> bool:
        """Whether to enable newly added full-qpos-verified FK cache hits.

        The default remains off so callers can compare exact real-MjData
        traces before enabling this independent performance switch.
        """
        return self._reuse_prepared_fk

    @reuse_prepared_fk.setter
    def reuse_prepared_fk(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("reuse_prepared_fk must be a bool")
        self._reuse_prepared_fk = enabled

    @classmethod
    def _validate_gesture_config(cls, c: APEndpointGestureConfig) -> None:
        for name in ("dna02_command", "dng13_command"):
            value = float(getattr(c, name))
            if not math.isfinite(value) or abs(value) > 1.0:
                raise ValueError(f"{name} must be finite and in [-1,1]")
        if c.dna02_soma_side not in SIDES or c.dng13_soma_side not in SIDES:
            raise ValueError("neuron soma sides must be 'left' or 'right'")
        for name in ("dna02_aep_posterior_shift_mm", "dng13_endpoint_shift_mm"):
            value = float(getattr(c, name))
            if not math.isfinite(value) or not 0.0 < value <= 0.20:
                raise ValueError(f"{name} must be in (0,0.20] mm")
        if int(c.endpoint_phase_samples) < 128 or int(c.endpoint_phase_samples) > 4096:
            raise ValueError("endpoint_phase_samples must be between 128 and 4096")
        if not c.enable_hind_clearance or c.clearance_peak_mm <= 0:
            raise ValueError("prototype preserves V4.1 clearance; enable_hind_clearance and a positive peak are required")
        dna_target = c.dna02_soma_side
        dng_target = cls._opposite(c.dng13_soma_side)
        if (
            abs(float(c.dna02_command)) > 1e-12
            and abs(float(c.dng13_command)) > 1e-12
            and dna_target == dng_target
        ):
            raise ValueError("simultaneous commands must target opposite sides to keep the two gestures separately identifiable")

    def _build_endpoint_catalogue(self, sample_count: int) -> dict[str, dict[str, object]]:
        """Find kinematic AP extrema using fixed-root private-data template FK."""
        phases_1d = np.arange(sample_count, dtype=float) * (_TAU / sample_count)
        phases = np.broadcast_to(phases_1d[None, :], (6, sample_count)).copy()
        magnitudes = np.ones_like(phases)
        target_grid = np.asarray(
            self.steps.get_joint_angles_by_dof_order(phases, magnitudes, self.dof_order),
            dtype=float,
        )
        if target_grid.shape != (42, sample_count):
            raise ValueError(f"unexpected full-cycle target array {target_grid.shape}")

        saved_qpos = np.asarray(self.session.data.qpos, dtype=float).copy()
        saved_qvel = np.asarray(self.session.data.qvel, dtype=float).copy()
        self._ik_data.qpos[:] = saved_qpos
        self._ik_data.qvel[:] = saved_qvel
        ap_by_leg = np.zeros((sample_count, 6), dtype=float)
        for k in range(sample_count):
            self._ik_data.qpos[self.qadr] = target_grid[:, k]
            self.mj.mj_fwdPosition(self.model, self._ik_data)
            thorax_position = np.asarray(self._ik_data.xpos[self.thorax_id], dtype=float)
            thorax_rotation = np.asarray(self._ik_data.xmat[self.thorax_id], dtype=float).reshape(3, 3)
            for i, foot_id in enumerate(self.foot_ids):
                relative_local = thorax_rotation.T @ (
                    np.asarray(self._ik_data.xpos[int(foot_id)], dtype=float) - thorax_position
                )
                ap_by_leg[k, i] = relative_local[0]

        out: dict[str, dict[str, object]] = {}
        phase_step = _TAU / sample_count
        for i, leg in enumerate(LEGS):
            curve = ap_by_leg[:, i]
            max_i = int(np.argmax(curve))
            min_i = int(np.argmin(curve))
            aep_phase = self._refine_extremum_phase(curve, max_i, phase_step, maximum=True)
            pep_phase = self._refine_extremum_phase(curve, min_i, phase_step, maximum=False)
            aep_ap = float(np.interp(aep_phase, np.r_[phases_1d, _TAU], np.r_[curve, curve[0]]))
            pep_ap = float(np.interp(pep_phase, np.r_[phases_1d, _TAU], np.r_[curve, curve[0]]))

            pep_to_aep_span = (aep_phase - pep_phase) % _TAU
            aep_to_pep_span = _TAU - pep_to_aep_span
            off_pep_aep = self._adhesion_off_fraction(leg, pep_phase, pep_to_aep_span)
            off_aep_pep = self._adhesion_off_fraction(leg, aep_phase, aep_to_pep_span)
            transitions = self._adhesion_transitions(leg, phases_1d)
            off_starts = [t["phase_rad"] for t in transitions if t["transition"] == "on_to_off"]
            off_ends = [t["phase_rad"] for t in transitions if t["transition"] == "off_to_on"]
            gap_pep_off_start = min((abs(_phase_gap(pep_phase, p)) for p in off_starts), default=None)
            gap_aep_off_end = min((abs(_phase_gap(aep_phase, p)) for p in off_ends), default=None)
            out[leg] = {
                "ap_axis": "thorax-local +X; tarsus5 body-origin projected into thorax frame",
                "aep_phase_rad": float(aep_phase),
                "pep_phase_rad": float(pep_phase),
                "aep_ap_mm": aep_ap,
                "pep_ap_mm": pep_ap,
                "endpoint_span_mm": float(aep_ap - pep_ap),
                "aep_native_adhesion_on": bool(self.base._get_adhesion_onoff(leg, aep_phase)),
                "pep_native_adhesion_on": bool(self.base._get_adhesion_onoff(leg, pep_phase)),
                "native_adhesion_transition_phases": transitions,
                "pep_to_aep_arc_adhesion_off_fraction": float(off_pep_aep),
                "aep_to_pep_arc_adhesion_off_fraction": float(off_aep_pep),
                "pep_to_aep_phase_arc_matches_return_flag_heuristic": bool(off_pep_aep > off_aep_pep),
                "pep_to_nearest_adhesion_off_start_rad": gap_pep_off_start,
                "aep_to_nearest_adhesion_on_start_rad": gap_aep_off_end,
                "template_ap_range_mm": float(np.ptp(curve)),
            }
        return out

    @staticmethod
    def _refine_extremum_phase(curve: np.ndarray, index: int, step: float, *, maximum: bool) -> float:
        y0 = float(curve[index])
        ym = float(curve[(index - 1) % len(curve)])
        yp = float(curve[(index + 1) % len(curve)])
        denominator = ym - 2.0 * y0 + yp
        offset = 0.0 if abs(denominator) < 1e-15 else 0.5 * (ym - yp) / denominator
        if maximum and denominator > 0.0:
            offset = 0.0
        if not maximum and denominator < 0.0:
            offset = 0.0
        return float(((index + float(np.clip(offset, -0.5, 0.5))) * step) % _TAU)

    def _adhesion_off_fraction(self, leg: str, start: float, span: float) -> float:
        if span <= 0.0:
            return 0.0
        phases = (float(start) + (np.arange(65, dtype=float) + 0.5) * span / 65.0) % _TAU
        off = [not bool(self.base._get_adhesion_onoff(leg, float(p))) for p in phases]
        return float(np.mean(off))

    def _adhesion_transitions(self, leg: str, phases: np.ndarray) -> list[dict[str, object]]:
        values = [bool(self.base._get_adhesion_onoff(leg, float(p))) for p in phases]
        transitions: list[dict[str, object]] = []
        for i, current in enumerate(values):
            previous = values[(i - 1) % len(values)]
            if previous != current:
                transitions.append({
                    "phase_rad": float(phases[i]),
                    "transition": "off_to_on" if current else "on_to_off",
                })
        return transitions

    def _endpoint_offsets_for_leg(self, leg_index: int) -> tuple[float, float, dict[str, float]]:
        aep_delta, pep_delta, dna_aep, dng_aep, dng_pep = self._endpoint_deltas_for_leg(leg_index)
        return aep_delta, pep_delta, {
            "dna02_aep_delta_mm": dna_aep,
            "dng13_aep_delta_mm": dng_aep,
            "dng13_pep_delta_mm": dng_pep,
        }

    def _endpoint_deltas_for_leg(
        self, leg_index: int
    ) -> tuple[float, float, float, float, float]:
        """Return control offsets without allocating their telemetry mapping."""
        c = self.gesture_config
        side = "left" if leg_index < 3 else "right"
        dna_aep = -float(c.dna02_command) * float(c.dna02_aep_posterior_shift_mm) if side == c.dna02_soma_side else 0.0
        dng_aep = float(c.dng13_command) * float(c.dng13_endpoint_shift_mm) if side == self._opposite(c.dng13_soma_side) else 0.0
        dng_pep = -float(c.dng13_command) * float(c.dng13_endpoint_shift_mm) if side == self._opposite(c.dng13_soma_side) else 0.0
        return dna_aep + dng_aep, dng_pep, dna_aep, dng_aep, dng_pep

    def endpoint_ap_delta(self, leg_index: int, phase_rad: float) -> float:
        """Continuous periodic body-local AP offset through native events."""
        leg = LEGS[int(leg_index)]
        item = self.endpoint_catalogue[leg]
        aep_delta, pep_delta, _, _, _ = self._endpoint_deltas_for_leg(int(leg_index))
        pep = float(item["pep_phase_rad"])
        aep = float(item["aep_phase_rad"])
        span = (aep - pep) % _TAU
        progress = (float(phase_rad) - pep) % _TAU
        if span <= 1e-10 or (_TAU - span) <= 1e-10:
            raise RuntimeError(f"{leg}: degenerate AEP/PEP phase separation")
        if progress <= span:
            u = progress / span
            return float(pep_delta + _smooth5(u) * (aep_delta - pep_delta))
        u = (progress - span) / (_TAU - span)
        return float(aep_delta + _smooth5(u) * (pep_delta - aep_delta))

    def prepare_targets_at_phase(
        self,
        phases: np.ndarray,
        side_active: np.ndarray | tuple[bool, bool] = (True, True),
        targets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Run private FK/IK at explicit phases without advancing physics.

        This is the public static-preparation hook used by the companion
        prepare script. ``targets`` may be omitted to use full-magnitude raw
        template targets at ``phases``.
        """
        phases = np.asarray(phases, dtype=float)
        side_active = np.asarray(side_active, dtype=bool)
        if phases.shape != (6,) or side_active.shape != (2,) or not np.isfinite(phases).all():
            raise ValueError("expected six finite phases and two side-active flags")
        if targets is None:
            targets = np.asarray(self.steps.get_joint_angles_by_dof_order(
                phases, np.ones(6, dtype=float), self.dof_order
            ), dtype=float)
        else:
            targets = np.asarray(targets, dtype=float).copy()
        if targets.shape != (42,):
            raise ValueError(f"expected 42 joint targets, got {targets.shape}")
        if not (abs(self.gesture_config.dna02_command) > 1e-12 or abs(self.gesture_config.dng13_command) > 1e-12):
            old_phases = np.asarray(self.base.cpg_network.curr_phases, dtype=float).copy()
            self.base.cpg_network.curr_phases[:] = phases
            try:
                out, inherited = super()._apply_hind_lift(targets, side_active)
            finally:
                self.base.cpg_network.curr_phases[:] = old_phases
            return out, {"mode": "exact_inherited_v4_1_zero_gesture", "per_leg": inherited}
        return self._solve_endpoint_targets(targets, side_active, phases)

    def _apply_hind_lift(self, targets: np.ndarray, side_active: np.ndarray):
        phases = np.asarray(self.base.cpg_network.curr_phases, dtype=float).copy()
        if not (abs(self.gesture_config.dna02_command) > 1e-12 or abs(self.gesture_config.dng13_command) > 1e-12):
            return super()._apply_hind_lift(targets, side_active)
        return self._solve_endpoint_targets(np.asarray(targets, dtype=float), np.asarray(side_active, dtype=bool), phases)

    def _apply_hind_lift_compact(self, targets: np.ndarray, side_active: np.ndarray) -> np.ndarray:
        """Apply the same task targets while omitting AP/helper telemetry work."""
        phases = np.asarray(self.base.cpg_network.curr_phases, dtype=float).copy()
        if not (
            abs(self.gesture_config.dna02_command) > 1e-12
            or abs(self.gesture_config.dng13_command) > 1e-12
        ):
            return super()._apply_hind_lift_compact(targets, side_active)
        solved, _ = self._solve_endpoint_targets(
            np.asarray(targets, dtype=float),
            np.asarray(side_active, dtype=bool),
            phases,
            collect_diagnostics=False,
        )
        return solved

    def _solve_endpoint_targets(
        self,
        targets: np.ndarray,
        side_active: np.ndarray,
        phases: np.ndarray,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[np.ndarray, dict[str, object]]:
        c = self.phase_speed_config
        result = np.asarray(targets, dtype=float).copy()
        per_leg: dict[str, dict[str, object]] | None = {} if collect_diagnostics else None
        total_delta_clips = 0
        total_box_clips = 0
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)

        for i, leg in enumerate(LEGS):
            side_index = 0 if i < 3 else 1
            phase = float(phases[i] % _TAU)
            adhesion_on = bool(self.base._get_adhesion_onoff(leg, phase))
            ap_delta = self.endpoint_ap_delta(i, phase)
            ap_terms = (
                self._endpoint_offsets_for_leg(i)[2]
                if collect_diagnostics
                else None
            )
            should_lift = False
            delta_z = 0.0
            nominal_q = result[self.leg_channels[leg]].copy()
            if side_active[side_index] and i in (2, 5) and not adhesion_on:
                profile = self._clearance_profile[leg]
                start = float(profile["start_phase_rad"])
                span = float(profile["swing_span_rad"])
                peak_progress = float(profile["peak_progress_rad"])
                progress = (phase - start) % _TAU
                if 0.0 <= progress < span and span > 0:
                    envelope = self._profile_value(progress, peak_progress, span)
                    should_lift = True
                else:
                    envelope = 0.0
            else:
                envelope = 0.0

            if not side_active[side_index] or (abs(ap_delta) <= 1e-12 and not should_lift):
                if collect_diagnostics:
                    assert per_leg is not None
                    per_leg[leg] = {
                        "phase_rad": phase,
                        "adhesion_on_native_schedule": adhesion_on,
                        "applied": False,
                        "reason": "inactive_side_or_zero_task_offset",
                        "target_endpoint_ap_delta_mm": float(ap_delta),
                        "aep_pep_command_terms_mm": ap_terms,
                    }
                continue

            self.mj.mj_copyData(self._ik_data, self.model, self.session.data)
            self._ik_data.qpos[self.qadr] = result
            self.mj.mj_fwdPosition(self.model, self._ik_data)
            nominal_fk_qpos_bytes = self._ik_data.qpos.tobytes(order="C")
            foot_id = int(self.foot_ids[i])
            nominal_foot = np.asarray(self._ik_data.xpos[foot_id], dtype=float).copy()
            raw_rotation = np.asarray(self._ik_data.xmat[foot_id], dtype=float).reshape(3, 3).copy()
            thorax_position = np.asarray(self._ik_data.xpos[self.thorax_id], dtype=float).copy()
            thorax_rotation = np.asarray(self._ik_data.xmat[self.thorax_id], dtype=float).reshape(3, 3).copy()
            nominal_local = (
                thorax_rotation.T @ (nominal_foot - thorax_position)
                if collect_diagnostics
                else None
            )

            if should_lift:
                marker = float(c.lh_stance_marker_world_z_mm if leg == "lh" else c.rh_stance_marker_world_z_mm)
                clearance_goal = marker + float(c.clearance_peak_mm)
                required_z = max(0.0, clearance_goal - float(nominal_foot[2]))
                delta_z = float(envelope * required_z)
            else:
                marker = None
                clearance_goal = None
                required_z = 0.0

            target_foot = nominal_foot + thorax_rotation[:, 0] * float(ap_delta)
            target_foot[2] += float(delta_z)
            channels = self.leg_channels[leg]
            qadr = self._leg_qadr[leg]
            vadr = self._leg_vadr[leg]
            lower = nominal_q - float(c.command_offset_limit_rad)
            upper = nominal_q + float(c.command_offset_limit_rad)
            current = nominal_q.copy()
            ik_delta_clips = 0
            box_clips = 0
            prepared_fk_qpos_bytes = nominal_fk_qpos_bytes
            jac_singular_values: list[float] | None = [] if collect_diagnostics else None
            jac_rank = 0

            for iteration_index in range(int(c.ik_iterations)):
                self._ik_data.qpos[qadr] = current
                # Compare all qpos against the last prepared MjData state.
                # mj_fwdPosition may normalize the root quaternion, so matching
                # only the active leg coordinates would be unsafe.
                current_qpos_bytes = self._ik_data.qpos.tobytes(order="C")
                if iteration_index == 0:
                    refresh_fk = current_qpos_bytes != nominal_fk_qpos_bytes
                else:
                    refresh_fk = not (
                        self.reuse_prepared_fk
                        and current_qpos_bytes == prepared_fk_qpos_bytes
                        and _qpos_quaternions_are_normalization_stable(
                            self.mj,
                            self.model,
                            self._ik_data.qpos,
                        )
                    )
                if refresh_fk:
                    self.mj.mj_fwdPosition(self.model, self._ik_data)
                    prepared_fk_qpos_bytes = self._ik_data.qpos.tobytes(order="C")
                error = target_foot - np.asarray(self._ik_data.xpos[foot_id], dtype=float)
                if float(np.linalg.norm(error)) <= 1e-4:
                    break
                jacp.fill(0.0)
                jacr.fill(0.0)
                self.mj.mj_jacBody(self.model, self._ik_data, jacp, jacr, foot_id)
                jp = jacp[:, vadr]
                regularized = jp @ jp.T + float(c.ik_damping_mm) ** 2 * np.eye(3)
                delta = jp.T @ np.linalg.solve(regularized, error)
                clipped = np.clip(delta, -float(c.ik_delta_limit_rad), float(c.ik_delta_limit_rad))
                if collect_diagnostics:
                    ik_delta_clips += int(np.count_nonzero(np.abs(delta - clipped) > 1e-12))
                proposed = current + clipped
                bounded = np.clip(proposed, lower, upper)
                if collect_diagnostics:
                    box_clips += int(np.count_nonzero(np.abs(proposed - bounded) > 1e-12))
                current = bounded

            self._ik_data.qpos[qadr] = current
            if not (
                self.reuse_prepared_fk
                and self._ik_data.qpos.tobytes(order="C") == prepared_fk_qpos_bytes
                and _qpos_quaternions_are_normalization_stable(
                    self.mj,
                    self.model,
                    self._ik_data.qpos,
                )
            ):
                self.mj.mj_fwdPosition(self.model, self._ik_data)
                prepared_fk_qpos_bytes = self._ik_data.qpos.tobytes(order="C")
            if collect_diagnostics:
                if i not in (2, 5):
                    before_pose_position = np.asarray(self._ik_data.xpos[foot_id], dtype=float).copy()
                    before_pose_rotation = np.asarray(self._ik_data.xmat[foot_id], dtype=float).reshape(3, 3).copy()
                    position_residual_position_only = float(np.linalg.norm(target_foot - before_pose_position))

            pose_priority_final_fk_ready = False
            if i in (2, 5):
                solved, pose_info = refine_orientation_at_position(
                    self.mj,
                    self.model,
                    self._ik_data,
                    foot_id,
                    qadr,
                    vadr,
                    nominal_q,
                    current,
                    target_foot,
                    raw_rotation,
                    box_rad=float(c.command_offset_limit_rad),
                    baseline_fk_prepared=True,
                    collect_diagnostics=collect_diagnostics,
                    reuse_prepared_fk=bool(self.reuse_prepared_fk),
                )
                pose_priority_final_fk_ready = True
            else:
                solved = current
                pose_info = (
                    {"applied": False, "reason": "front_middle_position_only"}
                    if collect_diagnostics
                    else None
                )

            self._ik_data.qpos[qadr] = solved
            if not pose_priority_final_fk_ready and collect_diagnostics:
                if not (
                    self.reuse_prepared_fk
                    and self._ik_data.qpos.tobytes(order="C") == prepared_fk_qpos_bytes
                    and _qpos_quaternions_are_normalization_stable(
                        self.mj,
                        self.model,
                        self._ik_data.qpos,
                    )
                ):
                    self.mj.mj_fwdPosition(self.model, self._ik_data)
            if collect_diagnostics:
                solved_foot = np.asarray(self._ik_data.xpos[foot_id], dtype=float).copy()
                solved_rotation = np.asarray(self._ik_data.xmat[foot_id], dtype=float).reshape(3, 3).copy()
                solved_local = thorax_rotation.T @ (solved_foot - thorax_position)
                target_local = thorax_rotation.T @ (target_foot - thorax_position)
                if i in (2, 5):
                    position_residual_position_only = float(
                        pose_info["position_error_before_mm"]
                    )
                    orientation_before_rad = float(pose_info["orientation_error_before_rad"])
                    orientation_after_rad = float(pose_info["orientation_error_after_rad"])
                else:
                    orientation_before_rad = float(np.linalg.norm(
                        Rotation.from_matrix(raw_rotation @ before_pose_rotation.T).as_rotvec()
                    ))
                    orientation_after_rad = float(np.linalg.norm(
                        Rotation.from_matrix(raw_rotation @ solved_rotation.T).as_rotvec()
                    ))
                jacp.fill(0.0)
                jacr.fill(0.0)
                self.mj.mj_jacBody(self.model, self._ik_data, jacp, jacr, foot_id)
                solved_jacobian = jacp[:, vadr]
                jac_singular_values = [float(v) for v in np.linalg.svd(solved_jacobian, compute_uv=False)]
                largest = jac_singular_values[0] if jac_singular_values else 0.0
                jac_rank = int(sum(v > max(largest * 1e-8, 1e-10) for v in jac_singular_values))
            result[channels] = solved
            if collect_diagnostics:
                total_delta_clips += ik_delta_clips
                total_box_clips += box_clips
                assert per_leg is not None and nominal_local is not None
                assert jac_singular_values is not None
                per_leg[leg] = {
                    "phase_rad": phase,
                    "adhesion_on_native_schedule": adhesion_on,
                    "actual_ground_contact_observed_by_controller": False,
                    "native_aep_phase_rad": float(self.endpoint_catalogue[leg]["aep_phase_rad"]),
                    "native_pep_phase_rad": float(self.endpoint_catalogue[leg]["pep_phase_rad"]),
                    "native_endpoint_ap_delta_mm": float(ap_delta),
                    "aep_pep_command_terms_mm": ap_terms,
                    "nominal_foot_body_ap_mm": float(nominal_local[0]),
                    "target_foot_body_ap_mm": float(target_local[0]),
                    "prescribed_ap_task_delta_mm": float(ap_delta),
                    "world_z_clearance_projection_into_body_ap_mm": float(target_local[0] - nominal_local[0] - ap_delta),
                    "solved_foot_body_ap_mm": float(solved_local[0]),
                    "nominal_foot_world_xyz_mm": nominal_foot.tolist(),
                    "target_foot_world_xyz_mm": target_foot.tolist(),
                    "solved_foot_world_xyz_mm": solved_foot.tolist(),
                    "target_ap_delta_mm": float(ap_delta),
                    "target_world_z_delta_mm": float(delta_z),
                    "hind_clearance_marker_world_z_mm": marker,
                    "hind_clearance_goal_world_z_mm": clearance_goal,
                    "hind_clearance_envelope": float(envelope),
                    "position_residual_position_only_mm": position_residual_position_only,
                    "position_residual_after_pose_priority_mm": float(np.linalg.norm(target_foot - solved_foot)),
                    "hind_orientation_error_before_rad": orientation_before_rad,
                    "hind_orientation_error_after_rad": orientation_after_rad,
                    "hind_pose_priority": pose_info,
                    "position_jacobian_rank_at_solved_pose": int(jac_rank),
                    "position_jacobian_singular_values_mm_per_rad_at_solved_pose": jac_singular_values,
                    "ik_delta_clip_count": int(ik_delta_clips),
                    "joint_box_clip_count": int(box_clips),
                    "joint_delta_from_same_tick_native_target_rad": (solved - nominal_q).tolist(),
                    "applied": True,
                }

        if not collect_diagnostics:
            return result, {}
        assert per_leg is not None
        return result, {
            "mode": "phase_locked_cartesian_endpoint_gesture",
            "profile": "C2 periodic quintic; values specified at FK-derived AEP/PEP phase extrema",
            "native_adhesion_schedule_modified": False,
            "actual_contact_timing": "separately measured by runner contact trace; not inferred from adhesion flag",
            "vertical_clearance_retained": True,
            "hind_orientation_constraint_retained": True,
            "total_ik_delta_clip_count": int(total_delta_clips),
            "total_joint_box_clip_count": int(total_box_clips),
            "per_leg": per_leg,
        }


__all__ = ["APEndpointGestureConfig", "APEndpointGestureController"]
