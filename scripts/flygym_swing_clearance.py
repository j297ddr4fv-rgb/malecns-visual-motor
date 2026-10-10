"""Opt-in flat-floor swing clearance for an engineered FlyGym controller.

This changes position-actuator targets, using FK/IK on a separate MjData.
It never writes the live body state, adds a body wrench, or reads a stimulus.
It is a mechanical gait aid, not a reconstructed motor-neuron/muscle model.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np


class SwingClearanceController:
    """Wrap the existing controller and raise nominally grounded swing feet.

    ``stance_marker_z_mm`` is measured from a mechanical reference on the same
    flat floor. It describes tarsus5 origins rather than toe material points.
    Neural side inputs keep their original amplitude and phase authority.
    """

    def __init__(self, session, stance_marker_z_mm, clearance_mm=0.15,
                 legs_to_lift=None, damping=0.03, iterations=6):
        self.base = session.controller
        self.session = session
        self.mj = session.mj
        self.model = session.model
        self.work = self.mj.MjData(self.model)
        self.legs = tuple(self.base.legs)
        self.stance_z = np.asarray(stance_marker_z_mm, dtype=float)
        if self.stance_z.shape != (6,) or not np.isfinite(self.stance_z).all():
            raise ValueError("six finite stance marker heights are required")
        if not 0 < clearance_mm <= 0.5 or iterations < 1:
            raise ValueError("clearance must be in (0, 0.5] mm and iterations positive")
        self.clearance_mm = float(clearance_mm)
        self.damping = float(damping)
        self.iterations = int(iterations)
        self.lift_legs = set(self.legs if legs_to_lift is None else legs_to_lift)
        if not self.lift_legs.issubset(self.legs):
            raise ValueError("unknown lift leg")
        joint_ids = self.model.actuator_trnid[session.position_actuator_ids, 0]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.joint_ids = joint_ids
        self.indices = {}
        for leg in self.legs:
            self.indices[leg] = np.array([
                i for i, joint_id in enumerate(joint_ids)
                if self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                     int(self.model.jnt_bodyid[joint_id]))
                .split("/")[-1].startswith(leg + "_")
            ], dtype=int)
            if len(self.indices[leg]) != 7:
                raise ValueError(f"{leg}: expected seven position-actuated DOFs")
        self.jac = np.zeros((3, self.model.nv))
        self.last_clearance_info = {}

    def __getattr__(self, name):
        # The session observes the original CPG and records its original states.
        return getattr(self.base, name)

    def step(self, descending_signal, obs):
        signal = np.asarray(descending_signal, dtype=float)
        action = self.base.step(signal, obs)
        nominal = np.asarray(action.joint_angles, dtype=float)
        if nominal.shape != self.qadr.shape:
            raise ValueError("position actuator order/command length mismatch")
        target_angles = nominal.copy()
        self.work.qpos[:] = self.session.data.qpos
        self.work.qpos[self.qadr] = target_angles
        self.mj.mj_fwdPosition(self.model, self.work)
        errors = {}
        for i, leg in enumerate(self.legs):
            if (leg not in self.lift_legs or action.adhesion_onoff[i]
                    or abs(signal[0 if leg.startswith('l') else 1]) < 1e-8):
                continue
            start, end = self.base.preprogrammed_steps.swing_period[leg]
            end += self.base.swing_extension
            phase = self.base.cpg_network.curr_phases[i] % (2 * np.pi)
            fraction = (phase - start) / (end - start)
            if not 0 < fraction < 1:
                continue
            lift = self.clearance_mm * np.sin(np.pi * fraction) ** 2
            body_id = int(self.session.foot_body_ids[i])
            goal = self.work.xpos[body_id].copy()
            goal[2] = max(goal[2], self.stance_z[i] + lift)
            idx = self.indices[leg]
            for _ in range(self.iterations):
                error = goal - self.work.xpos[body_id]
                if np.linalg.norm(error) <= 0.002:
                    break
                self.jac.fill(0)
                self.mj.mj_jacBody(self.model, self.work, self.jac, None, body_id)
                jac = self.jac[:, self.vadr[idx]]
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + self.damping ** 2 * np.eye(3), error)
                candidate = target_angles[idx] + np.clip(delta, -0.15, 0.15)
                candidate = np.clip(candidate, nominal[idx] - 0.7, nominal[idx] + 0.7)
                for j, angle in enumerate(candidate):
                    joint_id = self.joint_ids[idx[j]]
                    if self.model.jnt_limited[joint_id]:
                        candidate[j] = np.clip(angle, *self.model.jnt_range[joint_id])
                target_angles[idx] = candidate
                self.work.qpos[self.qadr[idx]] = candidate
                self.mj.mj_fwdPosition(self.model, self.work)
            errors[leg] = float(np.linalg.norm(goal - self.work.xpos[body_id]))
        self.last_clearance_info = {
            "foot_target_residual_mm": errors,
            "max_joint_offset_rad": float(np.max(np.abs(target_angles - nominal))),
        }
        return replace(action, joint_angles=target_angles)
