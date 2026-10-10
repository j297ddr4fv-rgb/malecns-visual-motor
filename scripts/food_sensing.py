#!/usr/bin/env python3
"""Pure environmental food-cue adapter for a digital fly.

The adapter reads an explicit physical-food mapping (or JSON file); it never
looks at visual meshes or semantic roles. The original odor field is a
hand-built isotropic Gaussian with SI spatial coordinates. An optional
advected intermittent puff train is available for experiments that pass the
world's local wind vector. Its parameters are uncalibrated engineering proxies.
The caller owns contact sensing: contact sugar is emitted only when
``physical_contact`` is literally ``True`` and the body-transformed mouth
proxy lies inside the configured edible sphere.

Schema version 1 uses ``concentration_unit`` for the field amplitude and
``sigma_m`` for the static field's spatial scale or the puff train's initial
radius. Odor selector weights have units of Hz per configured concentration
unit; sugar selector values are contact rates in Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple


Vector3 = Tuple[float, float, float]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vec3(value: Any, name: str) -> Vector3:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a 3-vector in metres")
    try:
        items = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be a 3-vector in metres") from exc
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    return items  # type: ignore[return-value]


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _distance(a: Vector3, b: Vector3) -> float:
    distance = math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
    if not math.isfinite(distance):
        raise ValueError("computed physical distance must be finite")
    return distance


def _rotate_offset(offset: Vector3, yaw_rad: float) -> Vector3:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * offset[0] - sine * offset[1],
        sine * offset[0] + cosine * offset[1],
        offset[2],
    )


def _translate(origin: Vector3, offset: Vector3) -> Vector3:
    point = (origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2])
    if not all(math.isfinite(value) for value in point):
        raise ValueError("transformed sensor position must be finite")
    return point


def _selector_map(value: Any, name: str) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a selector-to-value mapping")
    result: Dict[str, float] = {}
    for selector, raw_weight in value.items():
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError(f"{name} selector names must be nonempty strings")
        rate_or_weight = _nonnegative(raw_weight, f"{name}[{selector!r}]")
        result[selector] = rate_or_weight
    return result


def _check_keys(
    value: Mapping[str, Any], name: str, *, required: Sequence[str], optional: Sequence[str] = ()
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    missing = sorted(set(required).difference(value))
    if missing:
        raise ValueError(f"{name} is missing required key(s): {', '.join(missing)}")
    unknown = sorted(set(value).difference(required).difference(optional))
    if unknown:
        raise ValueError(f"{name} has unknown key(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class FoodCueSample:
    """One deterministic cue sample, ready for a caller's neural adapter."""

    odor_poisson_hz: Mapping[str, float]
    sugar_poisson_hz: Mapping[str, float]
    poisson_hz: Mapping[str, float]
    telemetry: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly copy of rates and compact telemetry."""

        return {
            "odor_poisson_hz": dict(self.odor_poisson_hz),
            "sugar_poisson_hz": dict(self.sugar_poisson_hz),
            "poisson_hz": dict(self.poisson_hz),
            "telemetry": dict(self.telemetry),
        }


@dataclass(frozen=True)
class FoodCueModel:
    """Physical food geometry and a deterministic odor concentration field."""

    concentration_unit: str
    provenance: str
    contact_surface_id: str
    center_m: Vector3
    amplitude: float
    background: float
    sigma_m: float
    field_kind: str
    plume_diffusivity_m2_per_s: float
    plume_burst_period_s: float
    plume_burst_duration_s: float
    plume_puff_interval_s: float
    plume_history_s: float
    plume_seed: int
    plume_parameter_status: str
    antenna_offsets_m: Mapping[str, Vector3]
    mouth_offset_m: Vector3
    edible_center_m: Vector3
    edible_radius_m: float
    odor_selector_weights_hz_per_unit: Mapping[str, Mapping[str, float]]
    sugar_selector_rates_hz: Mapping[str, float]
    selector_expected_counts: Mapping[str, int]

    @property
    def odor_selector_names(self) -> Tuple[str, ...]:
        """Configured odor selector names in stable sorted order."""

        return tuple(sorted(selector for side in ("left", "right")
                            for selector in self.odor_selector_weights_hz_per_unit[side]))

    @property
    def sugar_selector_names(self) -> Tuple[str, ...]:
        """Configured contact-sugar selector names in stable sorted order."""

        return tuple(sorted(self.sugar_selector_rates_hz))

    @property
    def selector_names(self) -> Tuple[str, ...]:
        """All configured receptor selectors in stable sorted order."""

        return tuple(sorted(self.odor_selector_names + self.sugar_selector_names))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FoodCueModel":
        """Parse schema version 1 of the separate physical-food configuration.

        Required fields are ``field``, antenna and mouth offsets, edible
        geometry, and explicit selector maps. The schema intentionally has no
        glTF or ``semantic_role`` input.
        """

        if not isinstance(mapping, Mapping):
            raise ValueError("food config must be a mapping")
        _check_keys(
            mapping,
            "food config",
            required=("schema_version", "concentration_unit", "provenance", "contact_surface_id",
                      "field", "antenna_offsets_m", "mouth_offset_m", "edible_region",
                      "odor_selector_weights_hz_per_unit", "sugar_selector_rates_hz"),
            optional=("selector_expected_counts",),
        )
        version = mapping.get("schema_version")
        if isinstance(version, bool) or version != 1:
            raise ValueError("food config schema_version must be 1")

        concentration_unit = mapping.get("concentration_unit")
        if not isinstance(concentration_unit, str) or not concentration_unit.strip():
            raise ValueError("concentration_unit must be a nonempty declared unit")
        provenance = mapping.get("provenance")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("provenance must identify the physical food cue source")
        contact_surface_id = mapping.get("contact_surface_id")
        if not isinstance(contact_surface_id, str) or not contact_surface_id.strip():
            raise ValueError("contact_surface_id must be a nonempty physical surface id")

        field = mapping.get("field")
        if not isinstance(field, Mapping):
            raise ValueError("field must be an explicit physical odor-field mapping")
        field_kind = field.get("kind")
        if field_kind == "isotropic_gaussian":
            _check_keys(field, "field", required=("kind", "center_m", "amplitude", "sigma_m"), optional=("background",))
        elif field_kind == "advected_intermittent_puff_train":
            _check_keys(
                field,
                "field",
                required=("kind", "center_m", "amplitude", "sigma_m",
                          "diffusivity_m2_per_s", "burst_period_s", "burst_duration_s",
                          "puff_interval_s", "history_s", "seed", "parameter_status"),
                optional=("background",),
            )
        else:
            raise ValueError(
                "field.kind must be 'isotropic_gaussian' or "
                "'advected_intermittent_puff_train'"
            )
        center_m = _vec3(field.get("center_m"), "field.center_m")
        amplitude = _nonnegative(field.get("amplitude"), "field.amplitude")
        background = _nonnegative(field.get("background", 0.0), "field.background")
        sigma_m = _finite(field.get("sigma_m"), "field.sigma_m")
        if sigma_m <= 0.0:
            raise ValueError("field.sigma_m must be greater than zero")
        maximum_concentration = amplitude + background
        if not math.isfinite(maximum_concentration):
            raise ValueError("field peak concentration must be finite")

        plume_diffusivity = 0.0
        plume_burst_period = 1.0
        plume_burst_duration = 1.0
        plume_puff_interval = 1.0
        plume_history = 0.0
        plume_seed = 0
        plume_parameter_status = "not_applicable"
        if field_kind == "advected_intermittent_puff_train":
            plume_diffusivity = _nonnegative(
                field.get("diffusivity_m2_per_s"), "field.diffusivity_m2_per_s"
            )
            plume_burst_period = _finite(field.get("burst_period_s"), "field.burst_period_s")
            plume_burst_duration = _finite(field.get("burst_duration_s"), "field.burst_duration_s")
            plume_puff_interval = _finite(field.get("puff_interval_s"), "field.puff_interval_s")
            plume_history = _finite(field.get("history_s"), "field.history_s")
            raw_seed = field.get("seed")
            if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
                raise ValueError("field.seed must be an integer")
            plume_seed = raw_seed
            plume_parameter_status = field.get("parameter_status")
            if not isinstance(plume_parameter_status, str) or not plume_parameter_status.strip():
                raise ValueError("field.parameter_status must identify calibration status")
            if plume_parameter_status != "uncalibrated_engineering_proxy":
                raise ValueError(
                    "advected puff-train parameters must be labeled "
                    "'uncalibrated_engineering_proxy'"
                )
            if plume_burst_period <= 0.0:
                raise ValueError("field.burst_period_s must be greater than zero")
            if plume_burst_duration <= 0.0 or plume_burst_duration > plume_burst_period:
                raise ValueError("field.burst_duration_s must be in (0, burst_period_s]")
            if plume_puff_interval <= 0.0:
                raise ValueError("field.puff_interval_s must be greater than zero")
            if plume_history <= 0.0:
                raise ValueError("field.history_s must be greater than zero")
            puffs_per_burst = math.ceil(plume_burst_duration / plume_puff_interval)
            bursts_in_history = math.ceil(plume_history / plume_burst_period) + 2
            if puffs_per_burst * bursts_in_history > 4096:
                raise ValueError("field puff-train settings request more than 4096 puff samples")

        antenna_config = mapping.get("antenna_offsets_m")
        if not isinstance(antenna_config, Mapping):
            raise ValueError("antenna_offsets_m must define left and right offsets")
        _check_keys(antenna_config, "antenna_offsets_m", required=("left", "right"))
        antenna_offsets = {
            side: _vec3(antenna_config.get(side), f"antenna_offsets_m.{side}")
            for side in ("left", "right")
        }

        edible = mapping.get("edible_region")
        if not isinstance(edible, Mapping):
            raise ValueError("edible_region must define its physical geometry")
        _check_keys(edible, "edible_region", required=("center_m", "radius_m"))
        edible_center_m = _vec3(edible.get("center_m"), "edible_region.center_m")
        edible_radius_m = _finite(edible.get("radius_m"), "edible_region.radius_m")
        if edible_radius_m <= 0.0:
            raise ValueError("edible_region.radius_m must be greater than zero")

        odor_config = mapping.get("odor_selector_weights_hz_per_unit")
        if not isinstance(odor_config, Mapping):
            raise ValueError("odor_selector_weights_hz_per_unit must define left and right maps")
        _check_keys(odor_config, "odor_selector_weights_hz_per_unit", required=("left", "right"))
        odor_weights = {
            side: _selector_map(odor_config.get(side), f"odor_selector_weights_hz_per_unit.{side}")
            for side in ("left", "right")
        }
        # Every selector must have one source so odor/sugar ablations remain
        # auditable and exact.
        duplicated_odor = set(odor_weights["left"]).intersection(odor_weights["right"])
        if duplicated_odor:
            raise ValueError(f"odor selectors must belong to one antenna side: {sorted(duplicated_odor)!r}")
        for side, selector_weights in odor_weights.items():
            for selector, weight in selector_weights.items():
                if not math.isfinite(maximum_concentration * weight):
                    raise ValueError(f"maximum odor rate for {side} selector {selector!r} must be finite")

        sugar_rates = _selector_map(mapping.get("sugar_selector_rates_hz"), "sugar_selector_rates_hz")
        odor_names = set(odor_weights["left"]).union(odor_weights["right"])
        overlap = odor_names.intersection(sugar_rates)
        if overlap:
            raise ValueError(f"odor and sugar selector maps must not overlap: {sorted(overlap)!r}")

        expected_counts_raw = mapping.get("selector_expected_counts", {})
        if not isinstance(expected_counts_raw, Mapping):
            raise ValueError("selector_expected_counts must map selectors to positive neuron counts")
        expected_counts: Dict[str, int] = {}
        configured_names = odor_names.union(sugar_rates)
        for selector, count in expected_counts_raw.items():
            if selector not in configured_names:
                raise ValueError(f"selector_expected_counts names undeclared selector {selector!r}")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"selector_expected_counts[{selector!r}] must be a positive integer")
            expected_counts[selector] = count

        return cls(
            concentration_unit=concentration_unit.strip(),
            provenance=provenance.strip(),
            contact_surface_id=contact_surface_id.strip(),
            center_m=center_m,
            amplitude=amplitude,
            background=background,
            sigma_m=sigma_m,
            field_kind=field_kind,
            plume_diffusivity_m2_per_s=plume_diffusivity,
            plume_burst_period_s=plume_burst_period,
            plume_burst_duration_s=plume_burst_duration,
            plume_puff_interval_s=plume_puff_interval,
            plume_history_s=plume_history,
            plume_seed=plume_seed,
            plume_parameter_status=plume_parameter_status,
            antenna_offsets_m=MappingProxyType(antenna_offsets),
            mouth_offset_m=_vec3(mapping.get("mouth_offset_m"), "mouth_offset_m"),
            edible_center_m=edible_center_m,
            edible_radius_m=edible_radius_m,
            odor_selector_weights_hz_per_unit=MappingProxyType({
                side: MappingProxyType(weights) for side, weights in odor_weights.items()
            }),
            sugar_selector_rates_hz=MappingProxyType(sugar_rates),
            selector_expected_counts=MappingProxyType(expected_counts),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FoodCueModel":
        """Load and validate a standalone JSON physical-food configuration."""

        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                config = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load food config {path!s}: {exc}") from exc
        return cls.from_mapping(config)

    @staticmethod
    def _seed_fraction(seed: int) -> float:
        """Stable, process-independent pseudo-random fraction in [0, 1)."""

        # SplitMix64 finalizer. Python's built-in hash is salted between
        # processes, so it must not be used for replayable plume phases.
        mask = (1 << 64) - 1
        value = (seed + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return value / float(1 << 64)

    def _concentration(
        self,
        point_m: Vector3,
        *,
        time_s: float = 0.0,
        wind_mps: Vector3 = (0.0, 0.0, 0.0),
        seed: int | None = None,
    ) -> float:
        if self.field_kind == "isotropic_gaussian":
            distance_m = _distance(point_m, self.center_m)
            scaled_distance = distance_m / self.sigma_m
            # Beyond 40 sigma the contribution is below double precision's useful
            # range. The branch also avoids overflowing an unnecessary square.
            spatial_weight = 0.0 if scaled_distance >= 40.0 else math.exp(-0.5 * scaled_distance * scaled_distance)
            concentration = self.background + self.amplitude * spatial_weight
        else:
            concentration = self._puff_train_concentration(
                point_m,
                time_s=time_s,
                wind_mps=wind_mps,
                seed=self.plume_seed if seed is None else seed,
            )
        if not math.isfinite(concentration):
            raise ValueError("computed odor concentration is not finite")
        return concentration

    def _puff_train_concentration(
        self,
        point_m: Vector3,
        *,
        time_s: float,
        wind_mps: Vector3,
        seed: int,
    ) -> float:
        """Sample diffusively spreading Gaussian puffs released in bursts.

        Puff centers advect with the wind passed for this sample. Their spread
        follows ``sigma^2 = sigma0^2 + 2 D age`` and their concentration is
        scaled by inverse puff volume. The sum is capped at the configured
        amplitude. This replayable model is an engineering proxy rather than
        a calibrated atmospheric transport solution.
        """

        phase_s = self._seed_fraction(seed) * self.plume_burst_period_s
        earliest = time_s - self.plume_history_s
        first_burst = math.floor(
            (earliest - phase_s - self.plume_burst_duration_s)
            / self.plume_burst_period_s
        )
        last_burst = math.floor((time_s - phase_s) / self.plume_burst_period_s)
        total = 0.0
        burst_count = math.ceil(self.plume_burst_duration_s / self.plume_puff_interval_s)

        for burst_index in range(first_burst, last_burst + 1):
            burst_start = phase_s + burst_index * self.plume_burst_period_s
            for puff_index in range(burst_count):
                emission_time = burst_start + puff_index * self.plume_puff_interval_s
                age_s = time_s - emission_time
                if age_s < 0.0 or age_s > self.plume_history_s:
                    continue
                sigma_m = math.sqrt(
                    self.sigma_m * self.sigma_m
                    + 2.0 * self.plume_diffusivity_m2_per_s * age_s
                )
                puff_center = tuple(
                    source + velocity * age_s
                    for source, velocity in zip(self.center_m, wind_mps)
                )
                scaled_distance = _distance(point_m, puff_center) / sigma_m
                if scaled_distance >= 40.0:
                    continue
                volume_factor = (self.sigma_m / sigma_m) ** 3
                total += volume_factor * math.exp(-0.5 * scaled_distance * scaled_distance)
                if total >= 1.0:
                    total = 1.0
                    break
            if total >= 1.0:
                break

        concentration = self.background + self.amplitude * total
        if not math.isfinite(concentration):
            raise ValueError("computed advected puff concentration is not finite")
        return concentration

    def sample(
        self,
        position_m: Sequence[float],
        yaw_rad: Any,
        time_s: Any,
        physical_contact: bool,
        *,
        wind_mps: Sequence[float] = (0.0, 0.0, 0.0),
        seed: Any = None,
    ) -> FoodCueSample:
        """Sample body-center and bilateral odor plus explicitly gated sugar.

        ``position_m`` is the body origin, and ``yaw_rad`` rotates the local
        sensor offsets about world +Z. ``wind_mps`` is the current world-space
        wind vector (zero by default). The puff-train variant is evaluated
        statelessly from position, time, wind and seed, so replaying those
        inputs reproduces the same cue without a mutable plume cache.
        """

        position = _vec3(position_m, "position_m")
        yaw = _finite(yaw_rad, "yaw_rad")
        time = _finite(time_s, "time_s")
        if time < 0.0:
            raise ValueError("time_s must be nonnegative")
        wind = _vec3(wind_mps, "wind_mps")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer when supplied")
        if type(physical_contact) is not bool:
            raise ValueError("physical_contact must be a caller-supplied bool")

        body_concentration = self._concentration(
            position, time_s=time, wind_mps=wind, seed=seed
        )
        antenna_points = {
            side: _translate(position, _rotate_offset(offset, yaw))
            for side, offset in self.antenna_offsets_m.items()
        }
        antenna_concentration = {
            side: self._concentration(
                antenna_points[side], time_s=time, wind_mps=wind, seed=seed
            )
            for side in ("left", "right")
        }
        mouth_point = _translate(position, _rotate_offset(self.mouth_offset_m, yaw))
        mouth_distance_m = _distance(mouth_point, self.edible_center_m)
        inside_edible_region = mouth_distance_m <= self.edible_radius_m
        edible_contact = physical_contact and inside_edible_region

        odor_poisson_hz: Dict[str, float] = {}
        for side in ("left", "right"):
            concentration = antenna_concentration[side]
            for selector, weight in self.odor_selector_weights_hz_per_unit[side].items():
                rate = concentration * weight
                if not math.isfinite(rate):
                    raise ValueError(f"computed odor rate for {selector!r} is not finite")
                odor_poisson_hz[selector] = rate

        sugar_poisson_hz = {selector: 0.0 for selector in self.sugar_selector_rates_hz}
        if edible_contact:
            sugar_poisson_hz.update(self.sugar_selector_rates_hz)

        combined: Dict[str, float] = dict(odor_poisson_hz)
        combined.update(sugar_poisson_hz)
        selectors = set(combined)
        for side in ("left", "right"):
            selectors.update(self.odor_selector_weights_hz_per_unit[side])
        poisson_hz = {selector: combined.get(selector, 0.0) for selector in sorted(selectors)}

        telemetry: Dict[str, Any] = {
            "time_s": time,
            "concentration": body_concentration,
            "left_sample": antenna_concentration["left"],
            "right_sample": antenna_concentration["right"],
            "concentration_unit": self.concentration_unit,
            "mouth_distance_m": mouth_distance_m,
            "physical_contact": physical_contact,
            "edible_region_contact": inside_edible_region,
            "contact": edible_contact,
            "contact_surface_id": self.contact_surface_id,
            "provenance": self.provenance,
        }
        if self.field_kind == "advected_intermittent_puff_train":
            telemetry.update({
                "odor_field_kind": self.field_kind,
                "odor_field_parameter_status": self.plume_parameter_status,
                "wind_mps": list(wind),
                "plume_seed": self.plume_seed if seed is None else seed,
            })
        return FoodCueSample(
            odor_poisson_hz=MappingProxyType(odor_poisson_hz),
            sugar_poisson_hz=MappingProxyType(sugar_poisson_hz),
            poisson_hz=MappingProxyType(poisson_hz),
            telemetry=telemetry,
        )


__all__ = ["FoodCueModel", "FoodCueSample"]
