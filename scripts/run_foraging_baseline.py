#!/usr/bin/env python3
"""Run an approximate engineered, non-CNS elevated-food controller on CPU.

This is a separate diagnostic experiment, not a mode of ``run_digital_fly``.
The controller receives bilateral odor rates, configured ambient wind rotated
into body axes, a physical support-contact bit, and a bounded hunger scalar.
The runner owns the sensor geometry and collision world; it never passes food
coordinates or body pose to the policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flight_dynamics import (  # noqa: E402
    ContactSurface,
    FlightDynamics,
    FlightDynamicsConfig,
    FlightState,
)
from food_sensing import FoodCueModel  # noqa: E402
from foraging_policy import (  # noqa: E402
    ForagingPolicy, ForagingPolicyConfig, POLICY_NAME, POLICY_SCOPE,
)
from foraging_state import ForagingState  # noqa: E402
from fitted_foraging_controller import (  # noqa: E402
    FeedbackForagingConfig, FeedbackForagingController, FEEDBACK_SCOPE,
)


DEFAULT_WORLD_CONFIG = ROOT / "config" / "foraging_elevated_experiment.json"
DEFAULT_FOOD_CONFIG = ROOT / "config" / "foraging_food_cue.json"
DEFAULT_OUTPUT_ROOT = Path("/data/artifacts/by-project/malecns-gpu/experiments")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load JSON config {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_config(path: Path | None) -> ForagingPolicyConfig | FeedbackForagingConfig:
    """Load a portable parameter artifact; use historical defaults if omitted."""
    if path is None:
        return ForagingPolicyConfig()
    artifact = _read_json(path)
    parameters = artifact.get("policy_config")
    if not isinstance(parameters, dict):
        raise ValueError("policy artifact must contain a policy_config object")
    try:
        kind = artifact.get("controller_type", "legacy")
        if kind not in ("legacy", "feedback"):
            raise ValueError(f"unknown controller_type: {kind}")
        cls = FeedbackForagingConfig if kind == "feedback" else ForagingPolicyConfig
        return cls(**parameters)
    except TypeError as exc:
        raise ValueError(f"invalid policy_config keys: {exc}") from exc


def _yaw_from_quat(quaternion: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wind_body_components(
    wind_world_mps: tuple[float, float, float], yaw_rad: float
) -> tuple[float, float]:
    """Return configured ambient wind in body axes (simplified direction sensor)."""

    wx, wy = wind_world_mps[:2]
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return wx * cosine + wy * sine, -wx * sine + wy * cosine


def _touching_surface(
    state: FlightState,
    *,
    plane_z_m: float,
    contains_xy: Any,
    tolerance_m: float = 2.0e-8,
    vertical_speed_tolerance_mps: float = 2.0e-3,
) -> bool:
    return (
        contains_xy(state.position_m)
        and abs(state.position_m[2] - plane_z_m) <= tolerance_m
        and abs(state.velocity_mps[2]) <= vertical_speed_tolerance_mps
    )


def _contact_flags(
    state: FlightState,
    *,
    ground_plane_z_m: float,
    target_surface: ContactSurface,
) -> tuple[bool, bool]:
    on_pad = _touching_surface(
        state,
        plane_z_m=target_surface.top_z_m,
        contains_xy=target_surface.contains_xy,
    )
    on_ground = _touching_surface(
        state,
        plane_z_m=ground_plane_z_m,
        contains_xy=lambda _position: True,
    )
    return on_ground or on_pad, on_pad


def _initial_state(world_config: Mapping[str, Any], yaw_offset_rad: float) -> tuple[FlightState, float, float]:
    experiment = world_config.get("experiment", {})
    world = world_config.get("world", {})
    pose = experiment.get("initial_pose") or world_config.get("scenarios", {}).get("escape", {}).get("initial_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("world config is missing an experiment initial_pose")
    position = tuple(float(value) for value in pose.get("position_m", (0.0, 0.0, 0.0012)))
    velocity = tuple(float(value) for value in pose.get("velocity_mps", (0.0, 0.0, 0.0)))
    yaw = float(pose.get("yaw_rad", 0.0)) + yaw_offset_rad
    roll = float(pose.get("roll_rad", 0.0))
    pitch = float(pose.get("pitch_rad", 0.0))
    state = FlightState.from_any({
        "position_m": position,
        "velocity_mps": velocity,
        "orientation_rad": (roll, pitch, yaw),
    })
    return state, float(world.get("ground_z_m", 0.0)), float(world.get("body_radius_m", 0.0012))


def _world_surface(
    world_config: Mapping[str, Any], body_radius_m: float
) -> ContactSurface:
    target = world_config.get("world", {}).get("target")
    if not isinstance(target, Mapping):
        raise ValueError("world config must define world.target as the raised-food support")
    target_id = str(target.get("id", "")).strip()
    position = tuple(float(value) for value in target.get("position_m", ()))
    size = tuple(float(value) for value in target.get("size_m", ()))
    if len(position) != 3 or len(size) != 3 or any(value <= 0.0 for value in size):
        raise ValueError("world.target position_m and size_m must be finite 3-vectors")
    return ContactSurface(
        surface_id=target_id,
        min_x_m=position[0] - 0.5 * size[0],
        max_x_m=position[0] + 0.5 * size[0],
        min_y_m=position[1] - 0.5 * size[1],
        max_y_m=position[1] + 0.5 * size[1],
        top_z_m=position[2] + 0.5 * size[2] + body_radius_m,
    )


def _side_rate(sample: Any, selectors: Mapping[str, float]) -> float:
    values = [float(sample.odor_poisson_hz[selector]) for selector in selectors]
    if not values:
        raise ValueError("food config must provide at least one odor selector per antenna")
    return sum(values) / len(values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    world_path = args.world_config.resolve()
    food_path = args.food_config.resolve()
    world_config = _read_json(world_path)
    food_model = FoodCueModel.from_json(food_path)
    initial_state, ground_z_m, body_radius_m = _initial_state(world_config, args.yaw_offset_rad)
    world = world_config["world"]
    target_config = world["target"]
    if str(target_config.get("id", "")) != food_model.contact_surface_id:
        raise ValueError("food contact_surface_id must match world.target.id")
    surface = _world_surface(world_config, body_radius_m)
    ground_plane_center_z_m = ground_z_m + body_radius_m
    dynamics_config = FlightDynamicsConfig(
        ground_plane_z_m=ground_plane_center_z_m,
        contact_surfaces=(surface,),
        wind_mps=tuple(args.wind_mps),
        fit_status="order_of_magnitude_defaults__engineered_foraging_baseline",
    )
    dt_s = 0.005
    duration_s = args.duration_s
    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise ValueError("duration_s must be finite and positive")
    frames = int(round(duration_s / dt_s))
    if abs(frames * dt_s - duration_s) > 1.0e-9:
        raise ValueError("duration_s must align to the 5 ms step")

    dynamics = FlightDynamics(config=dynamics_config, state=initial_state)
    policy_path = getattr(args, "policy_config", None)
    policy_path = policy_path.resolve() if policy_path is not None else None
    policy_config = load_policy_config(policy_path)
    feedback_policy = isinstance(policy_config, FeedbackForagingConfig)
    policy = (FeedbackForagingController(policy_config) if feedback_policy
              else ForagingPolicy(policy_config))
    policy_name = POLICY_NAME if policy_path is None else "parameterized_" + POLICY_NAME
    policy_scope = POLICY_SCOPE if policy_path is None else (
        "hand-built policy structure with externally supplied parameters; "
        "fitting provenance is recorded in the parameter artifact; "
        "not MaleCNS and not biologically validated"
    )
    if feedback_policy:
        policy_name, policy_scope = "feedback_foraging_corridor", FEEDBACK_SCOPE
    internal_state = ForagingState(initial_hunger=args.initial_hunger)
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    telemetry_path = output_dir / "telemetry.jsonl"
    odor_selector_weights = food_model.odor_selector_weights_hz_per_unit
    left_selectors = odor_selector_weights["left"]
    right_selectors = odor_selector_weights["right"]

    initial_energy = dynamics.state.cumulative_energy_J
    peak_altitude_m = dynamics.state.position_m[2]
    pad_contact_frames = 0
    edible_contact_frames = 0
    contact_sugar_frames = 0
    peak_contact_sugar_rate_hz = 0.0
    pad_landing_events = 0
    previous_pad_contact = False
    peak_food_rate_hz = 0.0
    last_surface_id: str | None = None
    edible_contact_seconds = 0.0
    time_above_target_plane_s = 0.0
    phase_counts: Counter[str] = Counter()
    first_pad_landing_time_s: float | None = None
    first_food_contact_time_s: float | None = None
    first_touchdown_horizontal_speed_mps: float | None = None
    peak_speed_mps = 0.0

    with telemetry_path.open("w", encoding="utf-8") as stream:
        for frame in range(frames):
            state = dynamics.state
            support_contact, pad_contact = _contact_flags(
                state,
                ground_plane_z_m=ground_plane_center_z_m,
                target_surface=surface,
            )
            raw_sample = food_model.sample(
                state.position_m,
                _yaw_from_quat(state.orientation_quat),
                state.time_s,
                physical_contact=pad_contact,
                wind_mps=args.wind_mps,
                seed=args.seed,
            )
            raw_left = _side_rate(raw_sample, left_selectors)
            raw_right = _side_rate(raw_sample, right_selectors)
            left = 0.0 if args.odor_off else raw_left
            right = 0.0 if args.odor_off else raw_right
            yaw_rad = _yaw_from_quat(state.orientation_quat)
            wind_body = _wind_body_components(tuple(args.wind_mps), yaw_rad)
            hunger_before = internal_state.hunger
            feedback_observations = {}
            if feedback_policy:
                vx, vy, vz = state.velocity_mps
                cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
                feedback_observations = {
                    "height_m": state.position_m[2] - ground_z_m,
                    "velocity_body_mps": (vx * cosine + vy * sine,
                                          -vx * sine + vy * cosine, vz),
                    "yaw_rate_rad_s": state.angular_velocity_rad_s[2],
                }
            decision = policy.step(
                left, right, support_contact,
                wind_body_mps=wind_body,
                hunger=hunger_before,
                **feedback_observations,
            )
            output = dynamics.step(decision.command, dt_s=dt_s)
            next_state = output.state
            _, next_pad_contact = _contact_flags(
                next_state,
                ground_plane_z_m=ground_plane_center_z_m,
                target_surface=surface,
            )
            peak_altitude_m = max(peak_altitude_m, next_state.position_m[2])
            speed_mps = math.sqrt(sum(value * value for value in next_state.velocity_mps))
            peak_speed_mps = max(peak_speed_mps, speed_mps)
            if next_state.position_m[2] > surface.top_z_m:
                time_above_target_plane_s += dt_s
            phase_counts[decision.phase] += 1
            peak_food_rate_hz = max(peak_food_rate_hz, 0.5 * (raw_left + raw_right))
            pad_contact_frames += int(next_pad_contact)
            food_contact = bool(raw_sample.telemetry["contact"])
            hunger_after = internal_state.advance(dt_s, food_contact)
            edible_contact_seconds += dt_s if food_contact else 0.0
            edible_contact_frames += int(food_contact)
            if food_contact and first_food_contact_time_s is None:
                first_food_contact_time_s = state.time_s
            sugar_max = max(raw_sample.sugar_poisson_hz.values(), default=0.0)
            contact_sugar_frames += int(sugar_max > 0.0)
            peak_contact_sugar_rate_hz = max(peak_contact_sugar_rate_hz, sugar_max)
            if next_pad_contact and not previous_pad_contact:
                pad_landing_events += 1
                if first_pad_landing_time_s is None:
                    first_pad_landing_time_s = output.time_s
                    first_touchdown_horizontal_speed_mps = math.hypot(
                        next_state.velocity_mps[0], next_state.velocity_mps[1]
                    )
            previous_pad_contact = next_pad_contact

            policy_telemetry = decision.telemetry()
            policy_telemetry.update(policy=policy_name, policy_scope=policy_scope)
            row = {
                "frame": frame,
                "time_s": output.time_s,
                "policy": policy_telemetry,
                "sensor": {
                    "odor_off": args.odor_off,
                    "raw_left_mean_rate_hz": raw_left,
                    "raw_right_mean_rate_hz": raw_right,
                    "raw_odor_poisson_hz": dict(raw_sample.odor_poisson_hz),
                    "raw_contact_sugar_poisson_hz": dict(raw_sample.sugar_poisson_hz),
                    "food_cue": raw_sample.telemetry,
                    "ambient_wind_body_forward_left_mps": list(wind_body),
                    "hunger_before_command": hunger_before,
                    "hunger_after_step": hunger_after,
                    "edible_contact_for_state_update": food_contact,
                },
                "world": {
                    "position_m": list(next_state.position_m),
                    "velocity_mps": list(next_state.velocity_mps),
                    "orientation_quat": list(next_state.orientation_quat),
                    "angular_velocity_rad_s": list(next_state.angular_velocity_rad_s),
                    "support_contact_observed": support_contact,
                    "raised_food_support_contact_observed": pad_contact,
                "pad_contact_after_step": next_pad_contact,
                "food_contact_at_sampled_state": food_contact,
                    "contact_surface_event": output.contact_surface_id,
                },
                "dynamics": {
                    "lift_force_body_N": list(output.lift_force_body_N),
                    "total_force_world_N": list(output.total_force_world_N),
                    "power_W": output.power_W,
                    "cumulative_energy_J": output.cumulative_energy_J,
                    "fit_status": output.fit_status,
                    "model": output.model,
                    "model_scope": output.model_scope,
                },
            }
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            last_surface_id = output.contact_surface_id or last_surface_id

    summary = {
        "experiment": "engineered_elevated_food_baseline",
        "policy": policy_name,
        "policy_scope": policy_scope,
        "policy_parameter_artifact": str(policy_path) if policy_path else None,
        "evidence_class": "engineered_non_CNS_behavior_policy_approximation",
        "behavior_classification": "approximate_engineered_foraging_policy_not_MaleCNS_autonomy",
        "odor_condition": "odor_off_matched_control" if args.odor_off else "odor_on",
        "frames": frames,
        "dt_s": dt_s,
        "duration_s": frames * dt_s,
        "initial_pose_m": list(initial_state.position_m),
        "yaw_offset_rad": args.yaw_offset_rad,
        "wind_mps": list(args.wind_mps),
        "odor_field_kind": food_model.field_kind,
        "peak_altitude_m": peak_altitude_m,
        "time_above_target_plane_s": time_above_target_plane_s,
        "peak_speed_mps": peak_speed_mps,
        "final_position_m": list(dynamics.state.position_m),
        "final_velocity_mps": list(dynamics.state.velocity_mps),
        "pad_contact_frames": pad_contact_frames,
        "pad_landing_events": pad_landing_events,
        "first_pad_landing_time_s": first_pad_landing_time_s,
        "first_touchdown_horizontal_speed_mps": first_touchdown_horizontal_speed_mps,
        "edible_contact_frames": edible_contact_frames,
        "first_food_contact_time_s": first_food_contact_time_s,
        "contact_sugar_frames": contact_sugar_frames,
        "peak_contact_sugar_rate_hz": peak_contact_sugar_rate_hz,
        "food_contact_observed": edible_contact_frames > 0,
        "peak_raw_mean_odor_rate_hz": peak_food_rate_hz,
        "initial_hunger": args.initial_hunger,
        "final_hunger": internal_state.hunger,
        "edible_contact_seconds": edible_contact_seconds,
        "policy_phase_frames": dict(sorted(phase_counts.items())),
        "internal_state": internal_state.telemetry(),
        "last_contact_surface_event": last_surface_id,
        "telemetry": telemetry_path.name,
        "telemetry_sha256": _sha256(telemetry_path),
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": summary["experiment"],
        "evidence_class": summary["evidence_class"],
        "model_type": "non_malecns_engineered_baseline",
        "controller": {
            "name": policy_name,
            "scope": policy_scope,
            "source": ("scripts/fitted_foraging_controller.py" if feedback_policy
                       else "scripts/foraging_policy.py"),
            "sha256": _sha256(Path(__file__).with_name(
                "fitted_foraging_controller.py" if feedback_policy else "foraging_policy.py")),
            "parameter_artifact": (
                {"path": str(policy_path), "sha256": _sha256(policy_path),
                 "content": _read_json(policy_path)}
                if policy_path is not None else None
            ),
            "observation_contract": [
                "mean left food-odor receptor rate in Hz",
                "mean right food-odor receptor rate in Hz",
                "boolean physical support contact",
                "body-frame configured ambient wind (forward/left), a simplified engineering wind-direction proxy that ignores self-motion",
                "internal hunger scalar in [0, 1], not a connectome state",
            ] + (["idealized height above ground, body velocity, yaw rate"]
                 if feedback_policy else []),
            "forbidden_policy_inputs": [
                "body XY world position", "body orientation",
                "food-field center", "world target coordinates", "glTF semantics",
                "MaleCNS output",
            ] + ([] if feedback_policy else ["body height", "body velocity", "yaw rate"]),
            "autonomous_malecns_claim": False,
        },
        "internal_state_model": {
            "name": "bounded_contact_gated_hunger_satiety_proxy",
            "scope": "scalar engineering approximation; not a fitted physiological model",
            "source": "scripts/foraging_state.py",
            "sha256": _sha256(ROOT / "scripts" / "foraging_state.py"),
        },
        "runner": {
            "source": "scripts/run_foraging_baseline.py",
            "sha256": _sha256(Path(__file__).resolve()),
            "scope": "CPU orchestration and sensor/world adapter; no neural endpoint",
        },
        "cue_model": {
            "path": str(food_path),
            "sha256": _sha256(food_path),
            "provenance": food_model.provenance,
            "physical_contact_surface_id": food_model.contact_surface_id,
        },
        "world": {
            "path": str(world_path),
            "sha256": _sha256(world_path),
            "support_surface": {
                "id": surface.surface_id,
                "body_center_plane_z_m": surface.top_z_m,
                "footprint_xy_m": [surface.min_x_m, surface.max_x_m,
                                    surface.min_y_m, surface.max_y_m],
            },
            "ground_body_center_plane_z_m": ground_plane_center_z_m,
            "ambient_wind_mps": list(args.wind_mps),
        },
        "dynamics": {
            "name": dynamics.model_name,
            "scope": dynamics.model_scope,
            "fit_status": dynamics.config.fit_status,
            "parameters": dynamics.parameters,
            "source": "scripts/flight_dynamics.py",
            "sha256": _sha256(ROOT / "scripts" / "flight_dynamics.py"),
        },
        "condition": {
            "odor_off": args.odor_off,
            "dt_s": dt_s,
            "duration_s": frames * dt_s,
            "seed": args.seed,
            "stochastic_sampling": False,
        },
        "artifacts": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-config", type=Path, default=DEFAULT_WORLD_CONFIG)
    parser.add_argument("--food-config", type=Path, default=DEFAULT_FOOD_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=None,
                        help="optional JSON parameter artifact with a policy_config object")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--yaw-offset-rad", type=float, default=0.0,
                        help="environment-only initial heading perturbation; never passed to policy")
    parser.add_argument("--seed", type=int, default=1729,
                        help="replayable odor plume seed for matched-run provenance")
    parser.add_argument("--wind-mps", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("WX", "WY", "WZ"),
                        help="ambient world-frame airflow; used by both plume advection and reduced dynamics")
    parser.add_argument("--initial-hunger", type=float, default=1.0,
                        help="initial internal hunger proxy in [0, 1]; default is food-seeking state")
    parser.add_argument("--odor-off", action="store_true",
                        help="mask odor rates at the policy input while retaining raw cue telemetry")
    parser.add_argument("--out", type=Path, default=None,
                        help="new, non-existing run directory; default is under /data/artifacts/by-project/malecns-gpu/experiments")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.yaw_offset_rad):
        parser.error("--yaw-offset-rad must be finite")
    if any(not math.isfinite(value) for value in args.wind_mps):
        parser.error("--wind-mps values must be finite")
    if not math.isfinite(args.initial_hunger) or not 0.0 <= args.initial_hunger <= 1.0:
        parser.error("--initial-hunger must be finite and in [0, 1]")
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = "odor-off" if args.odor_off else "odor-on"
        args.out = DEFAULT_OUTPUT_ROOT / f"engineered-foraging-{stamp}-{tag}"
    try:
        summary = run(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
