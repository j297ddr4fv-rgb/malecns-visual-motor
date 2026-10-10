#!/usr/bin/env python3
"""Strict shared and muscle-group Hill parameter configuration.

The body descriptor's immutable ``optimal_length_m`` is the muscle's l0.
Configuration supplies normalized shortening speed (optimal lengths/second),
so vmax scales with l0 while remaining 10 l0/s by default.  The Hill class's
standalone defaults remain independent of this pipeline policy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from .leg_muscle_dynamics import MuscleParameters
except ImportError:  # direct script/test import with scripts/ on sys.path
    from leg_muscle_dynamics import MuscleParameters


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "leg_muscle_parameters.json"
DEFAULT_BODY_CONFIG_PATH = ROOT / "config" / "articulated_leg_body.json"

_CONFIG_FIELDS = {
    "max_isometric_force_N",
    "max_shortening_velocity_optimal_lengths_per_s",
    "activation_time_constant_s",
    "deactivation_time_constant_s",
    "length_tension_width",
    "force_velocity_shape",
    "eccentric_force_multiplier",
    "eccentric_velocity_scale",
    "passive_slack_length_m",
    "passive_strain_scale",
    "passive_force_multiplier",
    "passive_force_cap_multiplier",
}
_REQUIRED_TOP_LEVEL = {
    "schema_version", "units", "description", "evidence", "shared", "groups",
}


def _positive_scale(value: Any, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return result


def _body_group_names(path: Path) -> set[str]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        groups = body["muscles"]["groups"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read muscle groups from body config: {path}") from exc
    if not isinstance(groups, dict) or not groups or any(
        not isinstance(name, str) or not name for name in groups
    ):
        raise ValueError("body config muscles.groups must be a nonempty object")
    return set(groups)


class LegMuscleParameters:
    """Resolve immutable descriptor-based Hill parameters from JSON.

    ``force_scale`` multiplies each configured isometric force for sensitivity
    analysis.  It is deliberately bounded to (0, 10] so it cannot silently
    become an unbounded force knob.
    """

    def __init__(self, config_path: str | Path | None = None,
                 force_scale: float = 1.0, *,
                 known_groups: set[str] | None = None):
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.force_scale = _positive_scale(force_scale, "force_scale", maximum=10.0)
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read leg muscle parameters: {self.config_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("muscle parameter config must be a JSON object")
        unknown_top = set(raw) - _REQUIRED_TOP_LEVEL
        missing_top = _REQUIRED_TOP_LEVEL - set(raw)
        if unknown_top or missing_top:
            raise ValueError(
                f"muscle parameter config keys invalid: unknown={sorted(unknown_top)}, "
                f"missing={sorted(missing_top)}"
            )
        if (isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1
                or raw["units"] != "SI"):
            raise ValueError("muscle parameter config must use schema_version 1 and SI units")
        if not isinstance(raw["description"], str) or not raw["description"].strip():
            raise ValueError("description must be a nonempty string")
        if not isinstance(raw["evidence"], dict):
            raise ValueError("evidence must be an object")

        self._known_groups = set(known_groups) if known_groups is not None else _body_group_names(
            DEFAULT_BODY_CONFIG_PATH
        )
        if not self._known_groups or any(not isinstance(g, str) or not g for g in self._known_groups):
            raise ValueError("known_groups must contain nonempty group names")
        shared = raw["shared"]
        groups = raw["groups"]
        if not isinstance(shared, dict) or not isinstance(groups, dict):
            raise ValueError("shared and groups must be JSON objects")
        missing_shared = _CONFIG_FIELDS - set(shared)
        if missing_shared:
            raise ValueError(f"shared is missing Hill parameters: {sorted(missing_shared)}")
        self._validate_override(shared, "shared")
        unknown_groups = set(groups) - self._known_groups
        if unknown_groups:
            raise ValueError(f"unknown muscle groups in parameter config: {sorted(unknown_groups)}")
        for group, override in groups.items():
            if not isinstance(group, str) or not isinstance(override, dict):
                raise ValueError("groups must map known group names to parameter objects")
            self._validate_override(override, f"groups.{group}")
        self._raw = raw
        self._shared = dict(shared)
        self._group_overrides = {name: dict(values) for name, values in groups.items()}

    @staticmethod
    def _validate_override(values: Mapping[str, Any], where: str) -> None:
        unknown = set(values) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"unknown Hill parameter(s) in {where}: {sorted(unknown)}")
        translated: dict[str, Any] = {}
        for key, value in values.items():
            if key == "max_shortening_velocity_optimal_lengths_per_s":
                translated["max_shortening_velocity_mps"] = value
            elif key == "passive_slack_length_m" and value is None:
                translated[key] = None
            else:
                translated[key] = value
        # Constructing the actual dataclass centralizes its finite/range checks.
        # Use a one-metre l0; normalized speed remains numerically unchanged.
        translated.setdefault("max_isometric_force_N", 1.0e-5)
        translated.setdefault("max_shortening_velocity_mps", 10.0)
        try:
            MuscleParameters(optimal_length_m=1.0, **translated)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Hill parameters in {where}: {exc}") from exc

    def parameters_for(self, descriptor: Any) -> MuscleParameters:
        """Return parameters using the descriptor's immutable optimal length.

        ``descriptor`` may be the body's frozen dataclass or a mapping with
        ``group`` and ``optimal_length_m`` keys.  Current pose/kinematic lengths
        are intentionally never consulted.
        """
        if isinstance(descriptor, Mapping):
            group = descriptor.get("group")
            optimal_length = descriptor.get("optimal_length_m")
        else:
            group = getattr(descriptor, "group", None)
            optimal_length = getattr(descriptor, "optimal_length_m", None)
        if not isinstance(group, str) or group not in self._known_groups:
            raise ValueError(f"unknown muscle descriptor group: {group!r}")
        l0 = _positive_scale(optimal_length, "descriptor.optimal_length_m")
        resolved = dict(self._shared)
        resolved.update(self._group_overrides.get(group, {}))
        velocity_per_l0_s = _positive_scale(
            resolved.pop("max_shortening_velocity_optimal_lengths_per_s"),
            f"{group}.max_shortening_velocity_optimal_lengths_per_s",
        )
        force = _positive_scale(resolved["max_isometric_force_N"],
                                f"{group}.max_isometric_force_N")
        resolved["max_isometric_force_N"] = force * self.force_scale
        if not math.isfinite(resolved["max_isometric_force_N"]):
            raise ValueError("scaled max_isometric_force_N overflowed")
        resolved["max_shortening_velocity_mps"] = velocity_per_l0_s * l0
        return MuscleParameters(optimal_length_m=l0, **resolved)

    def provenance(self) -> dict[str, Any]:
        """Return effective configuration and the evidence caveats verbatim."""
        return {
            "schema": "leg-muscle-parameters-1",
            "config_path": str(self.config_path),
            "config": json.loads(json.dumps(self._raw)),
            "force_scale": self.force_scale,
            "force_scale_limit": [0.0, 10.0],
            "known_groups": sorted(self._known_groups),
            "force_scale_effect": "multiplies max_isometric_force_N for every group",
            "optimal_length_source": "immutable body muscle descriptor optimal_length_m",
            "notes": [
                "F_iso values are engineering hypotheses; no muscle-specific measured strength is encoded",
                "normalized vmax is scaled from each descriptor l0, independent of current pose",
            ],
        }


__all__ = ["LegMuscleParameters"]
