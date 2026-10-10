"""Contract tests for the independent physical-contact grooming micro-body."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grooming_body import (  # noqa: E402
    FOOT_CONTACT_RADIUS_MM,
    GroomingBody,
    JOINT_SPEED_LIMITS_RAD_S,
    LEG_NAMES,
    REGION_NAMES,
    REGION_PATCHES,
)


class GroomingBodyTest(unittest.TestCase):
    def test_fixed_patches_and_seeded_individual_particles_are_reproducible(self) -> None:
        counts = {"antenna": 5, "head": 3, "wing": 2, "abdomen": 1}
        first = GroomingBody(dust_seed=19, dust_by_region=counts)
        second = GroomingBody(dust_seed=19, dust_by_region=counts)

        self.assertEqual(first.particle_positions(), second.particle_positions())
        self.assertEqual(set(first.particle_positions()), set(REGION_NAMES))
        self.assertEqual(first.remaining(), {**counts})
        for region, particles in first.particle_positions().items():
            patch = REGION_PATCHES[region]
            self.assertEqual([item.particle_id for item in particles], list(range(counts[region])))
            for particle in particles:
                offset = sum((particle.position_mm[i] - patch.center_mm[i]) * patch.sweep_axis[i]
                             for i in range(3))
                self.assertLessEqual(abs(offset), patch.half_length_mm)

        # Patch and particle snapshots cannot mutate the body geometry.
        with self.assertRaises(TypeError):
            REGION_PATCHES["antenna"] = REGION_PATCHES["head"]  # type: ignore[index]
        with self.assertRaises((AttributeError, TypeError)):
            first.particle_positions()["antenna"][0].particle_id = 9  # type: ignore[misc]

    def test_region_particles_do_not_depend_on_other_regions_being_populated(self) -> None:
        seed = 31
        head_only = GroomingBody(seed, {"head": 5})
        antenna_only = GroomingBody(seed, {"antenna": 4})
        dual = GroomingBody(seed, {"head": 5, "antenna": 4})

        self.assertEqual(
            head_only.particle_positions()["head"],
            dual.particle_positions()["head"],
        )
        self.assertEqual(
            antenna_only.particle_positions()["antenna"],
            dual.particle_positions()["antenna"],
        )

    def test_neutral_steps_hold_pose_and_do_not_clean(self) -> None:
        body = GroomingBody(dust_seed=1, dust_by_region={"antenna": 4, "head": 4})
        initial_positions = body.particle_positions()
        initial_loads = body.tactile_loads()
        initial_counts = body.remaining()
        initial = body.step(0.005)

        self.assertEqual(initial.cleaned, ())
        self.assertEqual(initial.remaining, initial_counts)
        self.assertEqual(initial.tactile_loads, initial_loads)
        self.assertEqual(body.particle_positions(), initial_positions)
        self.assertEqual(initial.joint_angles["fore_left"], initial.joint_angles["fore_right"])
        self.assertEqual(initial.time_s, 0.005)

    def test_anterior_patches_are_reachable_and_particle_contact_cleans(self) -> None:
        for region in ("antenna", "head"):
            with self.subTest(region=region):
                body = GroomingBody(dust_seed=47, dust_by_region={region: 1})
                particle = body.particle_positions()[region][0]
                neutral = body.step(0.005)
                self.assertEqual(neutral.cleaned, ())
                start_angles = neutral.joint_angles["fore_left"]

                frame = neutral
                for _ in range(200):
                    frame = body.step(0.005, {"fore_left": particle.position_mm})
                    if frame.cleaned:
                        break

                self.assertEqual(frame.cleaned, ((region, particle.particle_id),))
                self.assertEqual(body.remaining()[region], 0)
                self.assertEqual(body.tactile_loads()[region], 0.0)
                end_angles = frame.joint_angles["fore_left"]
                self.assertGreater(max(abs(end_angles[i] - start_angles[i]) for i in range(2)), 0.1)
                toe = frame.foot_positions_mm["fore_left"]
                distance = math.dist(toe, particle.position_mm)
                self.assertLessEqual(distance, FOOT_CONTACT_RADIUS_MM + 1.0e-9)

    def test_joint_speed_is_bounded_and_bad_targets_fail_without_advancing(self) -> None:
        body = GroomingBody(dust_seed=7, dust_by_region={"head": 2})
        before = body.step(0.005)
        target = REGION_PATCHES["head"].center_mm
        moved = body.step(0.005, {"fore_left": target})
        for joint_index, joint_name in enumerate(("hip", "knee")):
            change = abs(moved.joint_angles["fore_left"][joint_index]
                         - before.joint_angles["fore_left"][joint_index])
            self.assertLessEqual(change, JOINT_SPEED_LIMITS_RAD_S[joint_name] * 0.005 + 1.0e-12)

        before_bad_call = body.remaining()
        with self.assertRaises(ValueError):
            body.step(0.005, {"fore_left": (float("nan"), 0.24, 0.15)})
        with self.assertRaises(ValueError):
            body.step(0.005, {"fore_left": REGION_PATCHES["wing"].center_mm})
        after = body.step(0.005)
        self.assertEqual(after.time_s, 0.015)
        self.assertEqual(body.remaining(), before_bad_call)

    def test_disabled_leg_neither_moves_nor_cleans_then_enabled_leg_contacts(self) -> None:
        body = GroomingBody(dust_seed=8, dust_by_region={"antenna": 1})
        particle = body.particle_positions()["antenna"][0]
        neutral = body.step(0.005)
        start_angles = neutral.joint_angles["fore_left"]

        for _ in range(80):
            frame = body.step(
                0.005,
                {"fore_left": particle.position_mm},
                disabled_legs=frozenset(("fore_left",)),
            )
        self.assertEqual(frame.joint_angles["fore_left"], start_angles)
        self.assertEqual(body.remaining()["antenna"], 1)
        self.assertEqual(frame.cleaned, ())

        for _ in range(200):
            frame = body.step(0.005, {"fore_left": particle.position_mm})
            if frame.cleaned:
                break
        self.assertEqual(frame.cleaned, (("antenna", particle.particle_id),))

    def test_posterior_patches_are_outside_foreleg_reach(self) -> None:
        for region in ("wing", "abdomen"):
            with self.subTest(region=region):
                body = GroomingBody(dust_seed=2, dust_by_region={region: 1})
                with self.assertRaises(ValueError):
                    body.step(0.005, {"fore_left": REGION_PATCHES[region].center_mm})
                self.assertEqual(body.remaining()[region], 1)

    def test_invalid_constructor_and_step_inputs_fail_closed(self) -> None:
        with self.assertRaises(TypeError):
            GroomingBody(dust_seed=True, dust_by_region={})
        with self.assertRaises(ValueError):
            GroomingBody(dust_seed=1, dust_by_region={"antenna": -1})
        with self.assertRaises(ValueError):
            GroomingBody(dust_seed=1, dust_by_region={"unknown": 1})

        body = GroomingBody(dust_seed=3, dust_by_region={"head": 1})
        before = body.step(0.005)
        with self.assertRaises(ValueError):
            body.step(float("inf"))
        with self.assertRaises(ValueError):
            body.step(0.005, {"fore_left": (0.9, 0.25, 0.15)})
        after = body.step(0.005)
        self.assertEqual(after.time_s, before.time_s + 0.005)
        self.assertEqual(after.joint_angles, before.joint_angles)

        with self.assertRaises(ValueError):
            body.step(0.005, disabled_legs=frozenset(("hind_left",)))
        self.assertEqual(body.remaining()["head"], 1)


if __name__ == "__main__":
    unittest.main()
