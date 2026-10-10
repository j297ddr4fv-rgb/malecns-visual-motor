#!/usr/bin/env python3
"""CPU-only articulated grooming micro-body with particle-level contact.

This is a deliberately small engineered mechanism, separate from MaleCNS and
the flight body.  It models two planar, two-link forelegs and deterministic
point dust on fixed body-coordinate strips.  A particle is removed only when a
moving spherical toe comes within ``FOOT_CONTACT_RADIUS_MM`` of that specific
particle along its articulated forward-kinematic path.

The model is useful for measuring reachability, joint motion, contact, and
remaining-particle tactile proxies.  It is not a reconstructed fly skeleton:
there are no hindlegs, wings, abdomen motion, muscles, contact forces,
deformation, grooming reflexes, or neural/CNS control.  The posterior patches
are intentional out-of-reach specificity controls for this reduced foreleg
mechanism.  Distances are in millimeters; angles are in radians.

Typical use::

    body = GroomingBody(dust_seed=1729, dust_by_region={"antenna": 8})
    patch = REGION_PATCHES["antenna"]
    target = tuple(patch.center_mm[i] + 0.1 * patch.sweep_axis[i]
                   for i in range(3))
    frame = body.step(0.005, {"fore_left": target})

``targets`` is a mapping from leg name to a body-coordinate foot target in
millimeters.  ``GroomingFrame.joint_angles`` stores ``(hip_rad, knee_rad)``;
``foot_positions_mm`` stores current toe positions; ``cleaned`` contains
``(region, particle_id)`` pairs removed during that step; ``remaining`` and
``tactile_loads`` are per-region snapshots.  Tactile load is the fraction of
the region's initially seeded particles that remain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import random
from types import MappingProxyType
from typing import TypeAlias


Vector3: TypeAlias = tuple[float, float, float]
JointPair: TypeAlias = tuple[float, float]

LEG_NAMES = ("fore_left", "fore_right")
REGION_NAMES = ("antenna", "head", "wing", "abdomen")


@dataclass(frozen=True)
class GroomingPatch:
    """A fixed one-dimensional dust strip in body coordinates.

    Dust particles are placed on the line from ``center_mm - half_length_mm *
    sweep_axis`` to ``center_mm + half_length_mm * sweep_axis``.  The axis is
    unit length.  A small line strip keeps the contact target explicit and
    makes the runner's sweep repeatable.
    """

    center_mm: Vector3
    sweep_axis: Vector3
    half_length_mm: float


# Body axes: +X anterior, +Y left, +Z dorsal.  The two anterior strips lie in
# the left foreleg's planar workspace.  The posterior strips are intentionally
# separated from either foreleg's workspace by more than link reach.
REGION_PATCHES: Mapping[str, GroomingPatch] = MappingProxyType({
    "antenna": GroomingPatch(
        center_mm=(1.38, 0.24, 0.58),
        sweep_axis=(0.0, 0.0, 1.0),
        half_length_mm=0.10,
    ),
    "head": GroomingPatch(
        center_mm=(0.90, 0.24, 0.15),
        sweep_axis=(1.0, 0.0, 0.0),
        half_length_mm=0.10,
    ),
    "wing": GroomingPatch(
        center_mm=(0.0, 2.30, -0.40),
        sweep_axis=(1.0, 0.0, 0.0),
        half_length_mm=0.10,
    ),
    "abdomen": GroomingPatch(
        center_mm=(-1.75, 0.24, -0.61),
        sweep_axis=(-1.0, 0.0, 0.0),
        half_length_mm=0.10,
    ),
})

LEG_ROOTS_MM: Mapping[str, Vector3] = MappingProxyType({
    "fore_left": (0.34, 0.24, -0.61),
    "fore_right": (0.34, -0.24, -0.61),
})
LEG_LENGTHS_MM: Mapping[str, tuple[float, float]] = MappingProxyType({
    "fore_left": (0.95, 0.95),
    "fore_right": (0.95, 0.95),
})
JOINT_SPEED_LIMITS_RAD_S: Mapping[str, float] = MappingProxyType({
    "hip": 6.0,
    "knee": 8.0,
})
FOOT_CONTACT_RADIUS_MM = 0.05
_HIP_LIMITS_RAD = (-math.pi, math.pi)
_KNEE_LIMITS_RAD = (0.05, math.pi - 0.05)
_HOME_ANGLES_RAD: JointPair = (-1.40, 2.20)
_TRACE_ANGLE_STEP_RAD = math.radians(0.75)
_PLANE_EPSILON_MM = 1.0e-9


@dataclass(frozen=True)
class GroomingParticle:
    """Immutable identity and body-coordinate position for one dust particle."""

    particle_id: int
    position_mm: Vector3


@dataclass(frozen=True)
class GroomingFrame:
    """Snapshot returned by :meth:`GroomingBody.step`.

    Joint pair order is ``(hip_rad, knee_rad)``.  Position vectors are in
    millimeters.  Mapping values are immutable snapshots and ``cleaned`` is a
    tuple of ``(region_name, particle_id)`` pairs removed during this step.
    """

    time_s: float
    joint_angles: Mapping[str, JointPair]
    foot_positions_mm: Mapping[str, Vector3]
    cleaned: tuple[tuple[str, int], ...]
    remaining: Mapping[str, int]
    tactile_loads: Mapping[str, float]


class GroomingBody:
    """A minimal physically measurable two-foreleg grooming body.

    Args:
        dust_seed: Integer seed for deterministic individual particle offsets.
        dust_by_region: Nonnegative particle counts keyed by known region.
            Omitted regions are initialized empty.

    ``step`` validates all inputs before changing state.  Unreachable or
    out-of-plane foot targets raise ``ValueError`` rather than silently
    teleporting a leg or projecting it onto a different contact patch.
    """

    def __init__(self, dust_seed: int, dust_by_region: Mapping[str, int]) -> None:
        if isinstance(dust_seed, bool) or not isinstance(dust_seed, int):
            raise TypeError("dust_seed must be an integer")
        if not isinstance(dust_by_region, Mapping):
            raise TypeError("dust_by_region must be a mapping")
        unknown_regions = set(dust_by_region) - set(REGION_NAMES)
        if unknown_regions:
            raise ValueError(f"unknown dust region(s): {sorted(unknown_regions)!r}")

        counts: dict[str, int] = {}
        for region in REGION_NAMES:
            count = dust_by_region.get(region, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"dust count for {region!r} must be a nonnegative integer")
            counts[region] = count

        self._initial_counts = counts
        self._particles: dict[str, dict[int, GroomingParticle]] = {}
        for region in REGION_NAMES:
            patch = REGION_PATCHES[region]
            # Derive an independent stable seed per region.  Adding dust in a
            # different region must not shift this region's particle layout.
            region_key = f"{dust_seed}:{region}".encode("utf-8")
            region_seed = int.from_bytes(hashlib.sha256(region_key).digest(), "big")
            generator = random.Random(region_seed)
            by_id: dict[int, GroomingParticle] = {}
            for particle_id in range(counts[region]):
                offset = generator.uniform(-patch.half_length_mm, patch.half_length_mm)
                position = tuple(
                    patch.center_mm[index] + offset * patch.sweep_axis[index]
                    for index in range(3)
                )
                by_id[particle_id] = GroomingParticle(particle_id, position)  # type: ignore[arg-type]
            self._particles[region] = by_id

        self._joint_angles: dict[str, JointPair] = {
            leg: _HOME_ANGLES_RAD for leg in LEG_NAMES
        }
        self._time_s = 0.0

    def step(
        self,
        dt_s: float,
        targets: Mapping[str, tuple[float, float, float]] | None = None,
        disabled_legs: frozenset[str] = frozenset(),
    ) -> GroomingFrame:
        """Advance bounded joint motion and remove only particles contacted.

        ``targets`` maps leg names to body-coordinate toe targets in
        millimeters.  Every enabled leg follows its two-link inverse-kinematic
        solution at bounded angular velocity.  Collision detection traces the
        articulated forward-kinematic toe path with at most 0.75 degrees per
        trace segment and tests the segment against each individual particle.
        A stationary foot does not clean dust.
        """

        step = _finite_scalar(dt_s, "dt_s")
        if step <= 0.0:
            raise ValueError("dt_s must be positive")
        if targets is None:
            target_map: dict[str, tuple[float, float, float]] = {}
        elif isinstance(targets, Mapping):
            target_map = {}
            for leg, raw_target in targets.items():
                if leg not in LEG_NAMES:
                    raise ValueError(f"unknown leg target: {leg!r}")
                target_map[leg] = _vector3(raw_target, f"targets[{leg!r}]")
        else:
            raise TypeError("targets must be a mapping or None")

        if isinstance(disabled_legs, (str, bytes)) or not isinstance(disabled_legs, (set, frozenset)):
            raise TypeError("disabled_legs must be a set or frozenset of leg names")
        disabled = frozenset(disabled_legs)
        if disabled - set(LEG_NAMES):
            raise ValueError(f"unknown disabled leg(s): {sorted(disabled - set(LEG_NAMES))!r}")

        # Resolve every requested target first, so malformed or unreachable
        # inputs fail closed without partially advancing any leg.
        desired: dict[str, JointPair] = {}
        for leg, target in target_map.items():
            desired[leg] = _inverse_kinematics(leg, target)

        previous = dict(self._joint_angles)
        next_angles = dict(previous)
        for leg in LEG_NAMES:
            if leg in disabled or leg not in desired:
                continue
            current = previous[leg]
            command = desired[leg]
            next_angles[leg] = (
                _slew_angle(current[0], command[0], JOINT_SPEED_LIMITS_RAD_S["hip"] * step),
                _slew_angle(current[1], command[1], JOINT_SPEED_LIMITS_RAD_S["knee"] * step),
            )

        # Test the motion before committing state.  Particle order and leg
        # order are fixed, so multiple simultaneous contacts are deterministic.
        hit_particles: set[tuple[str, int]] = set()
        for leg in LEG_NAMES:
            if leg in disabled:
                continue
            start_angles = previous[leg]
            end_angles = next_angles[leg]
            if start_angles == end_angles:
                continue
            for region in REGION_NAMES:
                for particle_id, particle in self._particles[region].items():
                    key = (region, particle_id)
                    if key in hit_particles:
                        continue
                    if _toe_path_contacts(
                        leg, start_angles, end_angles, particle.position_mm,
                    ):
                        hit_particles.add(key)

        self._joint_angles = next_angles
        self._time_s += step
        cleaned: list[tuple[str, int]] = []
        for region in REGION_NAMES:
            for particle_id in sorted(particle_id for r, particle_id in hit_particles if r == region):
                self._particles[region].pop(particle_id, None)
                cleaned.append((region, particle_id))

        return GroomingFrame(
            time_s=self._time_s,
            joint_angles=MappingProxyType(dict(self._joint_angles)),
            foot_positions_mm=MappingProxyType({
                leg: _forward_kinematics(leg, self._joint_angles[leg])
                for leg in LEG_NAMES
            }),
            cleaned=tuple(cleaned),
            remaining=MappingProxyType(self.remaining()),
            tactile_loads=MappingProxyType(self.tactile_loads()),
        )

    def tactile_loads(self) -> dict[str, float]:
        """Return per-region normalized tactile load from particles remaining."""

        return {
            region: (len(self._particles[region]) / count if count else 0.0)
            for region, count in self._initial_counts.items()
        }

    def remaining(self) -> dict[str, int]:
        """Return active individual-particle counts for every region."""

        return {region: len(self._particles[region]) for region in REGION_NAMES}

    def particle_positions(self) -> Mapping[str, tuple[GroomingParticle, ...]]:
        """Return immutable snapshots of active particles grouped by region."""

        return MappingProxyType({
            region: tuple(self._particles[region][particle_id]
                          for particle_id in sorted(self._particles[region]))
            for region in REGION_NAMES
        })


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector3(value: object, name: str) -> Vector3:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise TypeError(f"{name} must contain exactly three finite coordinates")
    return tuple(_finite_scalar(component, f"{name}[{index}]")
                 for index, component in enumerate(value))  # type: ignore[return-value]


def _forward_kinematics(leg: str, angles: JointPair) -> Vector3:
    root_x, root_y, root_z = LEG_ROOTS_MM[leg]
    upper, lower = LEG_LENGTHS_MM[leg]
    hip, knee = angles
    second = hip + knee
    return (
        root_x + upper * math.cos(hip) + lower * math.cos(second),
        root_y,
        root_z + upper * math.sin(hip) + lower * math.sin(second),
    )


def _inverse_kinematics(leg: str, target_mm: Vector3) -> JointPair:
    root_x, root_y, root_z = LEG_ROOTS_MM[leg]
    dx = target_mm[0] - root_x
    dy = target_mm[1] - root_y
    dz = target_mm[2] - root_z
    if abs(dy) > _PLANE_EPSILON_MM:
        raise ValueError(f"target for {leg} is outside its planar XZ workspace")

    upper, lower = LEG_LENGTHS_MM[leg]
    radius_sq = dx * dx + dz * dz
    radius = math.sqrt(radius_sq)
    min_reach = abs(upper - lower)
    max_reach = upper + lower
    if radius <= min_reach + 1.0e-12 or radius >= max_reach - 1.0e-12:
        raise ValueError(f"target for {leg} is outside its reachable annulus")

    cosine = (radius_sq - upper * upper - lower * lower) / (2.0 * upper * lower)
    knee = math.acos(max(-1.0, min(1.0, cosine)))
    hip = math.atan2(dz, dx) - math.atan2(
        lower * math.sin(knee), upper + lower * math.cos(knee),
    )
    if not _HIP_LIMITS_RAD[0] <= hip <= _HIP_LIMITS_RAD[1]:
        raise ValueError(f"target for {leg} exceeds hip joint limits")
    if not _KNEE_LIMITS_RAD[0] <= knee <= _KNEE_LIMITS_RAD[1]:
        raise ValueError(f"target for {leg} exceeds knee joint limits")
    return (hip, knee)


def _slew_angle(current: float, desired: float, max_delta: float) -> float:
    delta = desired - current
    if abs(delta) <= max_delta:
        return desired
    return current + math.copysign(max_delta, delta)


def _point_segment_distance_sq(point: Vector3, start: Vector3, end: Vector3) -> float:
    direction = tuple(end[i] - start[i] for i in range(3))
    relative = tuple(point[i] - start[i] for i in range(3))
    length_sq = sum(component * component for component in direction)
    if length_sq <= 1.0e-24:
        return sum(component * component for component in relative)
    parameter = sum(relative[i] * direction[i] for i in range(3)) / length_sq
    parameter = max(0.0, min(1.0, parameter))
    closest = tuple(start[i] + parameter * direction[i] for i in range(3))
    return sum((point[i] - closest[i]) ** 2 for i in range(3))


def _toe_path_contacts(
    leg: str,
    start_angles: JointPair,
    end_angles: JointPair,
    particle_position_mm: Vector3,
) -> bool:
    """Check the articulated toe's sampled continuous path against a particle."""

    max_delta = max(abs(end_angles[i] - start_angles[i]) for i in range(2))
    if max_delta <= 1.0e-14:
        return False
    sample_count = max(1, math.ceil(max_delta / _TRACE_ANGLE_STEP_RAD))
    previous_position = _forward_kinematics(leg, start_angles)
    radius_sq = FOOT_CONTACT_RADIUS_MM * FOOT_CONTACT_RADIUS_MM
    for sample in range(1, sample_count + 1):
        fraction = sample / sample_count
        angles = (
            start_angles[0] + fraction * (end_angles[0] - start_angles[0]),
            start_angles[1] + fraction * (end_angles[1] - start_angles[1]),
        )
        position = _forward_kinematics(leg, angles)
        if _point_segment_distance_sq(particle_position_mm, previous_position, position) <= radius_sq:
            return True
        previous_position = position
    return False


__all__ = [
    "FOOT_CONTACT_RADIUS_MM",
    "GroomingBody",
    "GroomingFrame",
    "GroomingParticle",
    "GroomingPatch",
    "JOINT_SPEED_LIMITS_RAD_S",
    "LEG_LENGTHS_MM",
    "LEG_NAMES",
    "LEG_ROOTS_MM",
    "REGION_NAMES",
    "REGION_PATCHES",
]
