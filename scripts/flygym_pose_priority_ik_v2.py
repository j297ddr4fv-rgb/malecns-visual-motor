#!/usr/bin/env python3
"""V2 position-priority refinement with safe early-stop diagnostics.

The routine starts at the frozen V3 position-only solution. It takes short
world-frame orientation steps in the local nullspace of the foot-position
Jacobian, retracts toward the same Cartesian position, and accepts a proposal
only when it preserves the baseline position tolerance and reduces raw-foot
orientation error. The joint box is an engineering command bound, not an
anatomical or measured physiological range.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation


PARENT_HELPER_SHA256 = "38ebe35e422c0fb03d06cb915d9f151b2252bd12f51b25eca74bc2d44c235749"


@dataclass(frozen=True)
class PosePriorityIKConfig:
    max_orientation_iterations: int = 6
    orientation_step_limit_rad: float = 0.02
    position_retraction_iterations: int = 2
    position_retraction_step_limit_rad: float = 0.01
    orientation_damping_rad: float = 0.01
    position_damping_mm: float = 0.025
    svd_relative_tolerance: float = 1e-8
    position_acceptance_slack_mm: float = 1e-5
    minimum_orientation_gain_rad: float = 1e-8
    line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)


DEFAULT_CONFIG = PosePriorityIKConfig()


def _qpos_quaternions_are_normalization_stable(mj, model, qpos: np.ndarray) -> bool:
    """Return whether FK quaternion normalization leaves every joint quat exact.

    This guard is used only for newly added opt-in cache hits. A missing model
    joint layout is not enough evidence to skip FK, so unknown interfaces fail
    closed. Models with no free or ball joints are safe by construction.
    """
    if not hasattr(model, "jnt_type") or not hasattr(model, "jnt_qposadr"):
        return False
    try:
        joint_types = np.asarray(model.jnt_type)
        qpos_addresses = np.asarray(model.jnt_qposadr, dtype=int)
    except (TypeError, ValueError):
        return False
    if joint_types.ndim != 1 or qpos_addresses.ndim != 1 or len(joint_types) != len(qpos_addresses):
        return False
    if len(joint_types) == 0:
        return True

    joint_enum = getattr(mj, "mjtJoint", None)
    free_type = getattr(joint_enum, "mjJNT_FREE", None)
    ball_type = getattr(joint_enum, "mjJNT_BALL", None)
    if free_type is None or ball_type is None:
        return False

    quaternion_ranges: list[tuple[int, int]] = []
    for joint_type, qpos_address in zip(joint_types, qpos_addresses):
        if int(joint_type) == int(free_type):
            quaternion_ranges.append((int(qpos_address) + 3, int(qpos_address) + 7))
        elif int(joint_type) == int(ball_type):
            quaternion_ranges.append((int(qpos_address), int(qpos_address) + 4))
    if not quaternion_ranges:
        return True

    normalize4 = getattr(mj, "mju_normalize4", None)
    if not callable(normalize4):
        return False
    qpos_array = np.asarray(qpos)
    for start, end in quaternion_ranges:
        if start < 0 or end > qpos_array.size:
            return False
        quaternion = np.ascontiguousarray(qpos_array[start:end], dtype=np.float64)
        if quaternion.shape != (4,):
            return False
        normalized = quaternion.copy()
        normalize4(normalized)
        if normalized.tobytes(order="C") != quaternion.tobytes(order="C"):
            return False
    return True


def _rot_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def _body_state(
    mj,
    model,
    data,
    body_id: int,
    *,
    fk_prepared: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not fk_prepared:
        mj.mj_fwdPosition(model, data)
    return (
        np.asarray(data.xpos[body_id], dtype=float).copy(),
        np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy(),
    )


def _jacobians(
    mj,
    model,
    data,
    body_id: int,
    dof_addresses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mj.mj_jacBody(model, data, jacp, jacr, int(body_id))
    return jacp[:, dof_addresses].copy(), jacr[:, dof_addresses].copy()


def _nullspace_orientation_step(
    position_jacobian: np.ndarray,
    rotation_jacobian: np.ndarray,
    rotation_error: np.ndarray,
    free: np.ndarray,
    *,
    relative_tolerance: float,
    damping_rad: float,
    collect_diagnostics: bool = True,
) -> tuple[np.ndarray, int, list[float]]:
    """Return a minimum-norm angular correction in the position nullspace."""
    count = position_jacobian.shape[1]
    output = np.zeros(count, dtype=float)
    free_indices = np.flatnonzero(free)
    if not len(free_indices):
        return output, 0, []

    jp = position_jacobian[:, free_indices]
    jr = rotation_jacobian[:, free_indices]
    _, singular_values, vt = np.linalg.svd(jp, full_matrices=True)
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = largest * float(relative_tolerance)
    rank = int(np.count_nonzero(singular_values > tolerance))
    row_space = vt[:rank, :]
    null_projector = np.eye(len(free_indices)) - row_space.T @ row_space

    # Both the error vector and Jr are world-frame radians. Damping is therefore
    # expressed in radians and regularizes the angular least-squares system.
    projected_angular_jacobian = jr @ null_projector
    gram = (
        projected_angular_jacobian @ projected_angular_jacobian.T
        + float(damping_rad) ** 2 * np.eye(3)
    )
    step = null_projector @ projected_angular_jacobian.T @ np.linalg.solve(
        gram, rotation_error
    )
    output[free_indices] = step
    diagnostic_singular_values = (
        [float(value) for value in singular_values]
        if collect_diagnostics
        else []
    )
    return output, rank, diagnostic_singular_values


def refine_orientation_at_position(
    mj,
    model,
    data,
    body_id: int,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    nominal_q: np.ndarray,
    baseline_q: np.ndarray,
    target_xyz: np.ndarray,
    target_rotation: np.ndarray,
    *,
    box_rad: float = 0.35,
    config: PosePriorityIKConfig = DEFAULT_CONFIG,
    baseline_fk_prepared: bool = False,
    collect_diagnostics: bool = True,
    reuse_prepared_fk: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Refine foot rotation while preserving the baseline Cartesian target.

    ``data`` must be a private ``MjData`` whose nonactive qpos values already
    describe the desired fixed body/passive pose. This function updates only
    the supplied active joint qpos addresses. It never advances physics.
    When ``baseline_fk_prepared`` is true, the caller certifies that the
    position-stage caches in ``data`` already correspond to the current full
    qpos. The helper still checks the complete qpos after installing
    ``baseline_q`` before reusing those caches.
    Set ``collect_diagnostics=False`` to retain the solver's control math while
    skipping the returned diagnostic dictionary and its scalar/list assembly.
    ``reuse_prepared_fk`` gates only additional FK cache hits inside the
    orientation loop and its final pose readback; the prepared baseline
    shortcut remains governed by ``baseline_fk_prepared``.
    """
    start_time = time.perf_counter() if collect_diagnostics else 0.0
    qpos_addresses = np.asarray(qpos_addresses, dtype=int)
    dof_addresses = np.asarray(dof_addresses, dtype=int)
    nominal_q = np.asarray(nominal_q, dtype=float)
    baseline_q = np.asarray(baseline_q, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    if not (
        qpos_addresses.shape == dof_addresses.shape == nominal_q.shape == baseline_q.shape
    ):
        raise ValueError("joint address and vector dimensions must match")
    if target_xyz.shape != (3,) or len(nominal_q) != 7:
        raise ValueError("expected one XYZ target and seven active leg joints")
    if box_rad <= 0 or not np.isfinite(box_rad):
        raise ValueError("box_rad must be finite and positive")

    lower = nominal_q - float(box_rad)
    upper = nominal_q + float(box_rad)
    box_violation = float(max(
        0.0,
        np.max(lower - baseline_q),
        np.max(baseline_q - upper),
    ))
    if box_violation > 1e-10:
        raise ValueError(f"baseline V3 solution is outside its engineering box by {box_violation} rad")

    prepared_fk = bool(baseline_fk_prepared)
    prepared_qpos = np.asarray(data.qpos).copy()

    def evaluate(
        q: np.ndarray,
        *,
        allow_cached: bool,
        verify_quaternions: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal prepared_fk, prepared_qpos
        data.qpos[qpos_addresses] = q
        # Cache validity is tied to this exact MjData and its complete qpos.
        # In particular, MuJoCo may normalize a free-root quaternion during
        # FK, so active-joint equality alone is not a sufficient cache key.
        cache_matches = prepared_fk and np.asarray(data.qpos).tobytes(order="C") == prepared_qpos.tobytes(order="C")
        if cache_matches and verify_quaternions:
            cache_matches = _qpos_quaternions_are_normalization_stable(
                mj,
                model,
                data.qpos,
            )
        if not allow_cached or not cache_matches:
            mj.mj_fwdPosition(model, data)
            prepared_qpos = np.asarray(data.qpos).copy()
            prepared_fk = True
        return _body_state(
            mj,
            model,
            data,
            body_id,
            fk_prepared=True,
        )

    base_xyz, base_rotation = evaluate(
        baseline_q,
        allow_cached=baseline_fk_prepared,
    )
    baseline_position_error = float(np.linalg.norm(target_xyz - base_xyz))
    baseline_orientation_error = _rot_error(target_rotation, base_rotation)
    baseline_orientation_norm = float(np.linalg.norm(baseline_orientation_error))

    q = baseline_q.copy()
    accepted_steps = 0
    line_search_trials = 0
    retraction_steps = 0
    orientation_step_clip_count = 0
    bound_clip_count = 0
    retraction_clip_count = 0
    ranks: list[int] = []
    singular_values_by_iter: list[list[float]] = []
    active_bound_indices: list[int] = []

    for _ in range(int(config.max_orientation_iterations)):
        # The first pass often examines the same pose just evaluated for the
        # baseline. ``evaluate`` reuses that FK only while full qpos still
        # matches the saved MjData state.
        point, current_rotation = evaluate(
            q,
            allow_cached=reuse_prepared_fk,
            verify_quaternions=True,
        )
        error_xyz = target_xyz - point
        error_rotation = _rot_error(target_rotation, current_rotation)
        if float(np.linalg.norm(error_rotation)) <= config.minimum_orientation_gain_rad:
            break
        jp, jr = _jacobians(mj, model, data, body_id, dof_addresses)

        # Boundary joints are fixed only when the local angular descent points
        # out of the engineering box. Recompute the SVD after activating bounds.
        free = np.ones(7, dtype=bool)
        step = np.zeros(7, dtype=float)
        singular_values: list[float] = []
        rank = 0
        for _active_pass in range(8):
            step, rank, singular_values = _nullspace_orientation_step(
                jp,
                jr,
                error_rotation,
                free,
                relative_tolerance=config.svd_relative_tolerance,
                damping_rad=config.orientation_damping_rad,
                collect_diagnostics=collect_diagnostics,
            )
            at_lower = q <= lower + 1e-9
            at_upper = q >= upper - 1e-9
            outward = ((at_lower & (step < 0.0)) | (at_upper & (step > 0.0))) & free
            if not np.any(outward):
                break
            newly_active = np.flatnonzero(outward)
            if collect_diagnostics:
                active_bound_indices.extend(int(i) for i in newly_active)
            free[newly_active] = False
        if collect_diagnostics:
            ranks.append(int(rank))
            singular_values_by_iter.append(singular_values)

        clipped_step = np.clip(
            step,
            -float(config.orientation_step_limit_rad),
            float(config.orientation_step_limit_rad),
        )
        if collect_diagnostics:
            orientation_step_clip_count += int(
                np.count_nonzero(np.abs(step - clipped_step) > 1e-12)
            )
        if float(np.linalg.norm(clipped_step)) <= config.minimum_orientation_gain_rad:
            break

        accepted = False
        for scale in config.line_search_scales:
            if collect_diagnostics:
                line_search_trials += 1
            proposed = q + float(scale) * clipped_step
            bounded = np.clip(proposed, lower, upper)
            if collect_diagnostics:
                bound_clip_count += int(np.count_nonzero(np.abs(proposed - bounded) > 1e-12))
            candidate = bounded

            # Retract each proposal to the same position target. The clip and
            # bound are repeated at every retraction so no correction escapes.
            for _ in range(int(config.position_retraction_iterations)):
                candidate_xyz, _ = evaluate(
                    candidate,
                    allow_cached=reuse_prepared_fk,
                    verify_quaternions=True,
                )
                position_error = target_xyz - candidate_xyz
                if float(np.linalg.norm(position_error)) <= 1e-7:
                    break
                retract_jp, _ = _jacobians(mj, model, data, body_id, dof_addresses)
                gram = (
                    retract_jp @ retract_jp.T
                    + float(config.position_damping_mm) ** 2 * np.eye(3)
                )
                retract = retract_jp.T @ np.linalg.solve(gram, position_error)
                clipped_retract = np.clip(
                    retract,
                    -float(config.position_retraction_step_limit_rad),
                    float(config.position_retraction_step_limit_rad),
                )
                if collect_diagnostics:
                    retraction_clip_count += int(
                        np.count_nonzero(np.abs(retract - clipped_retract) > 1e-12)
                    )
                before_bound = candidate + clipped_retract
                candidate = np.clip(before_bound, lower, upper)
                if collect_diagnostics:
                    bound_clip_count += int(
                        np.count_nonzero(np.abs(before_bound - candidate) > 1e-12)
                    )
                    retraction_steps += 1

            candidate_xyz, candidate_rotation = evaluate(
                candidate,
                allow_cached=reuse_prepared_fk,
                verify_quaternions=True,
            )
            candidate_position_error = float(np.linalg.norm(target_xyz - candidate_xyz))
            candidate_orientation_error = float(
                np.linalg.norm(_rot_error(target_rotation, candidate_rotation))
            )
            position_ok = (
                candidate_position_error
                <= baseline_position_error + float(config.position_acceptance_slack_mm)
            )
            orientation_ok = (
                candidate_orientation_error
                < float(np.linalg.norm(error_rotation)) - config.minimum_orientation_gain_rad
            )
            if position_ok and orientation_ok:
                q = candidate.copy()
                if collect_diagnostics:
                    accepted_steps += 1
                accepted = True
                break
        if not accepted:
            break

    final_xyz, final_rotation = evaluate(
        q,
        allow_cached=reuse_prepared_fk,
        verify_quaternions=True,
    )
    final_position_error = float(np.linalg.norm(target_xyz - final_xyz))
    final_orientation_error = float(
        np.linalg.norm(_rot_error(target_rotation, final_rotation))
    )
    final_gain = baseline_orientation_norm - final_orientation_error
    fallback = (
        final_position_error > baseline_position_error + config.position_acceptance_slack_mm
        or final_gain <= config.minimum_orientation_gain_rad
    )
    if fallback:
        q = baseline_q.copy()
        final_xyz, final_rotation = evaluate(
            q,
            allow_cached=reuse_prepared_fk,
            verify_quaternions=True,
        )
        final_position_error = float(np.linalg.norm(target_xyz - final_xyz))
        final_orientation_error = float(
            np.linalg.norm(_rot_error(target_rotation, final_rotation))
        )
        final_gain = baseline_orientation_norm - final_orientation_error

    if not collect_diagnostics:
        return q, {}

    metrics: dict[str, object] = {
        "algorithm": "SVD position-nullspace orientation descent with bounded line search and XYZ retraction",
        "helper_version": "v2",
        "supersedes_helper_sha256": PARENT_HELPER_SHA256,
        "target_rotation_source": "same-pose raw named full-magnitude native template FK",
        "position_error_before_mm": baseline_position_error,
        "position_error_after_mm": final_position_error,
        "position_acceptance_slack_mm": float(config.position_acceptance_slack_mm),
        "orientation_error_before_rad": baseline_orientation_norm,
        "orientation_error_after_rad": final_orientation_error,
        "orientation_error_before_deg": float(np.degrees(baseline_orientation_norm)),
        "orientation_error_after_deg": float(np.degrees(final_orientation_error)),
        "orientation_gain_deg": float(np.degrees(final_gain)),
        "orientation_gain_accepted": bool(final_gain > config.minimum_orientation_gain_rad),
        "fallback_to_exact_v3_solution": bool(fallback),
        "accepted_orientation_steps": int(accepted_steps),
        "orientation_iterations_max": int(config.max_orientation_iterations),
        "line_search_trials": int(line_search_trials),
        "position_retraction_steps": int(retraction_steps),
        "orientation_step_clip_count": int(orientation_step_clip_count),
        "position_retraction_clip_count": int(retraction_clip_count),
        "engineering_box_clip_count": int(bound_clip_count),
        "active_box_dof_count": int(np.count_nonzero(
            (q <= lower + 1e-8) | (q >= upper - 1e-8)
        )),
        "active_outward_bound_dof_indices_seen": sorted(set(active_bound_indices)),
        "maximum_absolute_offset_rad": float(np.max(np.abs(q - nominal_q))),
        "box_rad": float(box_rad),
        "position_svd_rank_by_iteration": ranks,
        "position_svd_singular_values_by_iteration": singular_values_by_iter,
        "svd_relative_tolerance": float(config.svd_relative_tolerance),
        "orientation_damping_rad": float(config.orientation_damping_rad),
        "position_retraction_damping_mm": float(config.position_damping_mm),
        "parameters": asdict(config),
        "cpu_s": float(time.perf_counter() - start_time),
    }
    return q, metrics


__all__ = [
    "PosePriorityIKConfig",
    "DEFAULT_CONFIG",
    "PARENT_HELPER_SHA256",
    "_qpos_quaternions_are_normalization_stable",
    "refine_orientation_at_position",
]
