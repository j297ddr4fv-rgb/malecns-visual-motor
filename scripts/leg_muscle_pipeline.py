#!/usr/bin/env python3
"""CPU neural-output → muscle tension → articulated contact physics.

This independent experimental path does not change the accepted reduced-body
launcher. It accepts dense real per-neuron observations; it never generates a
gait, desired joint pose, body traction or desired body displacement.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from articulated_leg_body import ArticulatedLegBody
from leg_muscle_decoder import LegMuscleDecoder
from leg_muscle_dynamics import HillMuscle
from leg_muscle_parameters import LegMuscleParameters


class LegMusclePipeline:
    def __init__(self, decoder: LegMuscleDecoder | None = None,
                 body: ArticulatedLegBody | None = None,
                 rate_filter_tau_s: float = 0.020,
                 muscle_config_path: str | Path | None = None,
                 muscle_force_scale: float = 1.0):
        if not math.isfinite(rate_filter_tau_s) or rate_filter_tau_s <= 0:
            raise ValueError("rate_filter_tau_s must be finite and positive")
        self.decoder = decoder if decoder is not None else LegMuscleDecoder()
        self.body = body if body is not None else ArticulatedLegBody()
        self.rate_filter_tau_s = rate_filter_tau_s
        self._filtered_rates: np.ndarray | None = None
        self.muscle_parameter_loader = LegMuscleParameters(
            config_path=muscle_config_path,
            force_scale=muscle_force_scale,
            known_groups={descriptor.group for descriptor in self.body.muscles.values()},
        )
        self.muscles: dict[str, HillMuscle] = {}
        for name, descriptor in self.body.muscles.items():
            self.muscles[name] = HillMuscle(
                self.muscle_parameter_loader.parameters_for(descriptor)
            )
        decoder_names = {c["name"] for c in self.decoder.channels}
        if decoder_names != set(self.muscles):
            raise ValueError(
                f"decoder/body muscle mismatch: decoder_only={sorted(decoder_names-set(self.muscles))}, "
                f"body_only={sorted(set(self.muscles)-decoder_names)}"
            )
        self.time_s = 0.0

    def step(self, *, dt_s: float, rates_hz: Any = None,
             spikes: Any = None) -> dict[str, Any]:
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if (rates_hz is None) == (spikes is None):
            raise ValueError("supply exactly one of rates_hz and spikes")
        if spikes is not None:
            observed = np.asarray(spikes, dtype=float) / dt_s
            if observed.ndim != 1 or not np.isfinite(observed).all() or (observed < 0).any():
                raise ValueError("spikes must be a finite nonnegative dense vector")
            if self._filtered_rates is None:
                self._filtered_rates = np.zeros_like(observed)
            if self._filtered_rates.shape != observed.shape:
                raise ValueError("neural vector shape changed during a run")
            alpha = -math.expm1(-dt_s / self.rate_filter_tau_s)
            self._filtered_rates += alpha * (observed - self._filtered_rates)
            rates_hz = self._filtered_rates
        excitation = self.decoder.decode(rates_hz=rates_hz)
        # Muscle activation and kinematics are refreshed at the physical
        # integration interval, rather than holding stale forces across a frame.
        substep_s = min(dt_s, 0.0001)
        remaining = dt_s
        muscle_record: dict[str, Any] = {}
        body_record: dict[str, Any] = {}
        while remaining > 1e-12:
            h = min(substep_s, remaining)
            kinematics = self.body.muscle_kinematics()
            force: dict[str, float] = {}
            for name, muscle in self.muscles.items():
                state = muscle.step(
                    excitation=float(excitation[name]),
                    length_m=float(kinematics[name]["length_m"]),
                    velocity_mps=float(kinematics[name]["velocity_mps"]),
                    dt_s=h,
                )
                force[name] = float(state.total_force_N)
                muscle_record[name] = {"excitation": float(excitation[name]), **asdict(state)}
            body_record = self.body.step(force, dt_s=h)
            remaining -= h
        self.time_s += dt_s
        return {"time_s": self.time_s, "muscles": muscle_record, "body": body_record}

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "articulated-neural-muscle-pipeline-1",
            "evidence_class": "digital_simulation",
            "biological_validation": "NOT_VALIDATED",
            "control_path": ["per-neuron rates/spikes", "exact muscle-type decoder",
                             "Hill-type muscle tension", "fixed tendon moment arms",
                             "MuJoCo articulated rigid bodies", "foot-ground contact/friction"],
            "synthetic_gait": False,
            "body_traction_command": False,
            "rate_filter_tau_s": self.rate_filter_tau_s,
            "decoder": self.decoder.provenance(),
            "body": self.body.provenance(),
            "muscle_parameter_config": self.muscle_parameter_loader.provenance(),
            "muscle_parameters": {n: asdict(m.params) for n, m in self.muscles.items()},
            "unactuated_muscle_channels": [n for n, descriptor in self.body.muscles.items()
                                           if not descriptor.joint_moment_arms_m],
            "limitations": [
                "muscle force and activation parameters are engineering hypotheses",
                "fixed moment arms approximate anatomical tendon geometry",
                "sternotrochanter extension is projected onto a reduced proximal axis; biarticular routing is omitted",
                "unknown MN muscle targets remain unmapped",
                "joint/foot feedback is recorded; afferent tuning needs independent calibration",
                "no claim of spontaneous walking or successful visual turning",
            ],
        }
