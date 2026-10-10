#!/usr/bin/env python3
"""Fit bounded parameters of the hand-built foraging controller on CPU.

This is a deterministic, derivative-free parameter search over the existing
hand-built controller. It does not train a neural controller or make
biological claims. The environment may measure distance to the edible region
for scoring; the controller receives only its declared runtime observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flight_dynamics import FlightDynamics, FlightDynamicsConfig  # noqa: E402
from food_sensing import FoodCueModel  # noqa: E402
from foraging_policy import ForagingPolicy, ForagingPolicyConfig, POLICY_NAME, POLICY_SCOPE  # noqa: E402
from foraging_state import ForagingState  # noqa: E402
from run_foraging_baseline import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    _contact_flags,
    _initial_state,
    _read_json,
    _side_rate,
    _wind_body_components,
    _world_surface,
    _yaw_from_quat,
)


DEFAULT_WORLD_CONFIG = ROOT / "config" / "foraging_elevated_experiment.json"
DEFAULT_FOOD_CONFIG = ROOT / "config" / "foraging_food_cue_advected_plume.json"
DEFAULT_TRAIN_YAWS_DEG = (0.0, 90.0, -90.0, 180.0)
DEFAULT_HOLDOUT_YAWS_DEG = (45.0, -45.0, 135.0, -135.0)
DEFAULT_FEEDBACK_HOLDOUT_YAWS_DEG = (30.0, -60.0, 120.0, -150.0)
DEFAULT_WIND_MPS = (-0.08, 0.0, 0.0)
DT_S = 0.005

# Small bounded search over legacy motor-policy parameters. The bounds are
# narrower than the ForagingPolicyConfig validator limits where useful.
LEGACY_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "takeoff_s": (0.04, 0.20),
    "takeoff_power": (0.65, 1.0),
    "hover_power": (0.45, 0.90),
    "wind_hover_power": (0.40, 0.90),
    "low_odor_power": (0.40, 0.90),
    "approach_bias_rad": (-0.20, 0.20),
    "brake_bias_rad": (-0.20, 0.20),
    "max_yaw_torque_Nm": (5.0e-11, 2.0e-9),
    "wind_landing_delay_s": (0.20, 2.50),
    "wind_landing_brake_lead_s": (0.0, 0.80),
}
PARAMETER_BOUNDS = LEGACY_PARAMETER_BOUNDS

LEGACY_CONTROLLER_TYPE = "legacy"
FEEDBACK_CONTROLLER_TYPE = "feedback"


def _controller_spec(controller_type: str) -> tuple[type[Any], type[Any], dict[str, tuple[float, float]]]:
    if controller_type == LEGACY_CONTROLLER_TYPE:
        return ForagingPolicyConfig, ForagingPolicy, LEGACY_PARAMETER_BOUNDS
    if controller_type != FEEDBACK_CONTROLLER_TYPE:
        raise ValueError(f"unsupported controller_type {controller_type!r}")
    try:
        from fitted_foraging_controller import (  # type: ignore[import-not-found]
            FeedbackForagingConfig,
            FeedbackForagingController,
        )
        import fitted_foraging_controller as feedback_module  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(f"feedback controller is not available: {exc}") from exc
    bounds = getattr(feedback_module, "FEEDBACK_PARAMETER_BOUNDS", None)
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("fitted_foraging_controller must define FEEDBACK_PARAMETER_BOUNDS")
    return FeedbackForagingConfig, FeedbackForagingController, bounds


def _controller_observations(controller_type: str) -> list[str]:
    common = [
        "mean left food-odor receptor rate in Hz",
        "mean right food-odor receptor rate in Hz",
        "body-frame ambient wind (forward/left)",
        "physical support contact boolean",
        "bounded hunger scalar",
    ]
    if controller_type == FEEDBACK_CONTROLLER_TYPE:
        common.extend([
            "body-center height above ground_z in metres",
            "body-frame velocity (forward, left, vertical) in m/s",
            "world-z yaw rate in rad/s",
        ])
    return common


def _body_velocity_mps(
    velocity_world_mps: Sequence[float], yaw_rad: float
) -> tuple[float, float, float]:
    vx, vy, vz = _finite_vector(velocity_world_mps, "velocity_world_mps", 3)
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    result = (vx * cosine + vy * sine, -vx * sine + vy * cosine, vz)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("body-frame velocity must remain finite")
    return result


@dataclass(frozen=True)
class Scenario:
    name: str
    yaw_offset_rad: float


@dataclass(frozen=True)
class Environment:
    world_path: Path
    food_path: Path
    world_config: Mapping[str, Any]
    food_model: FoodCueModel
    ground_z_m: float
    body_radius_m: float
    ground_body_center_z_m: float
    target_surface: Any
    wind_mps: tuple[float, float, float]
    seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(values: Sequence[float], name: str, size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _scenario_list(yaws_deg: Sequence[float], prefix: str) -> tuple[Scenario, ...]:
    result = []
    for index, degrees in enumerate(yaws_deg):
        value = float(degrees)
        if not math.isfinite(value):
            raise ValueError(f"{prefix} yaw headings must be finite")
        result.append(Scenario(f"{prefix}_{index + 1}_{value:g}deg", math.radians(value)))
    if not result:
        raise ValueError(f"{prefix} requires at least one yaw heading")
    return tuple(result)


def _yaw_key(yaw_rad: float) -> int:
    # Shift/modulo gives both representations of the half-turn (-pi and +pi)
    # the same canonical value, including headings outside one revolution.
    wrapped = (yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    return round(wrapped * 1.0e8)


def _validate_scenario_split(
    training: Sequence[Scenario], holdout: Sequence[Scenario]
) -> None:
    holdout_keys = {_yaw_key(scenario.yaw_offset_rad) for scenario in holdout}
    overlap = [scenario.name for scenario in training if _yaw_key(scenario.yaw_offset_rad) in holdout_keys]
    if overlap:
        raise ValueError("training headings overlap held-out evaluation headings: " + ", ".join(overlap))


def _load_environment(
    world_path: Path,
    food_path: Path,
    wind_mps: Sequence[float],
    seed: int,
) -> Environment:
    world_path = world_path.resolve()
    food_path = food_path.resolve()
    world_config = _read_json(world_path)
    food_model = FoodCueModel.from_json(food_path)
    wind = _finite_vector(wind_mps, "wind_mps", 3)
    _, ground_z_m, body_radius_m = _initial_state(world_config, 0.0)
    target_surface = _world_surface(world_config, body_radius_m)
    if str(world_config["world"]["target"].get("id", "")) != food_model.contact_surface_id:
        raise ValueError("food contact_surface_id must match world.target.id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return Environment(
        world_path=world_path,
        food_path=food_path,
        world_config=world_config,
        food_model=food_model,
        ground_z_m=ground_z_m,
        body_radius_m=body_radius_m,
        ground_body_center_z_m=ground_z_m + body_radius_m,
        target_surface=target_surface,
        wind_mps=wind,  # type: ignore[arg-type]
        seed=seed,
    )


def _mouth_distance_mouth_region_gap(mouth_center_distance_m: float, radius_m: float) -> float:
    result = max(0.0, mouth_center_distance_m - radius_m)
    if not math.isfinite(result):
        raise ValueError("mouth-to-edible distance must remain finite")
    return result


def _simulate_case(
    env: Environment,
    policy_config: Any,
    scenario: Scenario,
    *,
    duration_s: float,
    initial_hunger: float,
    odor_off: bool,
    controller_type: str = LEGACY_CONTROLLER_TYPE,
) -> dict[str, Any]:
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    frames = round(duration_s / DT_S)
    if abs(frames * DT_S - duration_s) > 1.0e-9:
        raise ValueError("duration_s must align to the 5 ms dynamics step")

    state, _, _ = _initial_state(env.world_config, scenario.yaw_offset_rad)
    dynamics_config = FlightDynamicsConfig(
        ground_plane_z_m=env.ground_body_center_z_m,
        contact_surfaces=(env.target_surface,),
        wind_mps=env.wind_mps,
        fit_status="order_of_magnitude_defaults__engineered_foraging_parameter_fit",
    )
    dynamics = FlightDynamics(config=dynamics_config, state=state)
    _config_type, controller_impl, _parameter_bounds = _controller_spec(controller_type)
    policy = controller_impl(policy_config)
    internal_state = ForagingState(initial_hunger=initial_hunger)
    selectors = env.food_model.odor_selector_weights_hz_per_unit
    left_selectors = selectors["left"]
    right_selectors = selectors["right"]
    edible_radius_m = env.food_model.edible_radius_m

    minimum_gap_m = math.inf
    initial_gap_m: float | None = None
    minimum_center_distance_m = math.inf
    edible_contact_frames = 0
    support_contact_frames = 0
    pad_contact_frames = 0
    first_edible_contact_s: float | None = None
    peak_left_odor_hz = 0.0
    peak_right_odor_hz = 0.0
    phase_counts: Counter[str] = Counter()
    peak_altitude_m = state.position_m[2]

    for _frame in range(frames):
        state = dynamics.state
        support_contact, pad_contact = _contact_flags(
            state,
            ground_plane_z_m=env.ground_body_center_z_m,
            target_surface=env.target_surface,
        )
        yaw_rad = _yaw_from_quat(state.orientation_quat)
        raw_sample = env.food_model.sample(
            state.position_m,
            yaw_rad,
            state.time_s,
            physical_contact=pad_contact,
            wind_mps=env.wind_mps,
            seed=env.seed,
        )
        mouth_distance_m = float(raw_sample.telemetry["mouth_distance_m"])
        edible_gap_m = _mouth_distance_mouth_region_gap(mouth_distance_m, edible_radius_m)
        if initial_gap_m is None:
            initial_gap_m = edible_gap_m
        minimum_gap_m = min(minimum_gap_m, edible_gap_m)
        minimum_center_distance_m = min(minimum_center_distance_m, mouth_distance_m)

        raw_left = _side_rate(raw_sample, left_selectors)
        raw_right = _side_rate(raw_sample, right_selectors)
        left = 0.0 if odor_off else raw_left
        right = 0.0 if odor_off else raw_right
        wind_body = _wind_body_components(env.wind_mps, yaw_rad)
        # These are the complete runtime policy inputs. The target coordinate,
        # mouth distance and cue telemetry stay on this environment side.
        if controller_type == FEEDBACK_CONTROLLER_TYPE:
            decision = policy.step(
                left,
                right,
                support_contact,
                wind_body_mps=wind_body,
                hunger=internal_state.hunger,
                height_m=state.position_m[2] - env.ground_z_m,
                velocity_body_mps=_body_velocity_mps(state.velocity_mps, yaw_rad),
                yaw_rate_rad_s=state.angular_velocity_rad_s[2],
            )
        else:
            decision = policy.step(
                left,
                right,
                support_contact,
                wind_body_mps=wind_body,
                hunger=internal_state.hunger,
            )
        output = dynamics.step(decision.command, dt_s=DT_S)
        food_contact = bool(raw_sample.telemetry["contact"])
        internal_state.advance(DT_S, food_contact)
        edible_contact_frames += int(food_contact)
        support_contact_frames += int(support_contact)
        pad_contact_frames += int(pad_contact)
        if food_contact and first_edible_contact_s is None:
            first_edible_contact_s = state.time_s
        peak_left_odor_hz = max(peak_left_odor_hz, raw_left)
        peak_right_odor_hz = max(peak_right_odor_hz, raw_right)
        peak_altitude_m = max(peak_altitude_m, output.state.position_m[2])
        phase_counts[decision.phase] += 1

    assert initial_gap_m is not None
    approach_progress = min(1.0, max(0.0, (initial_gap_m - minimum_gap_m) / max(initial_gap_m, 1.0e-12)))
    contact_seconds = edible_contact_frames * DT_S
    contact_fraction = min(1.0, contact_seconds / 0.40)
    case_score = 0.60 * approach_progress + 0.40 * contact_fraction
    result = {
        "scenario": scenario.name,
        "yaw_offset_deg": math.degrees(scenario.yaw_offset_rad),
        "odor_condition": "odor_off_matched_control" if odor_off else "odor_on",
        "duration_s": frames * DT_S,
        "initial_edible_region_gap_m": initial_gap_m,
        "minimum_edible_region_gap_m": minimum_gap_m,
        "minimum_mouth_to_edible_center_distance_m": minimum_center_distance_m,
        "approach_progress_fraction": approach_progress,
        "physical_support_contact_frames": support_contact_frames,
        "raised_food_pad_contact_frames": pad_contact_frames,
        "edible_contact_frames": edible_contact_frames,
        "edible_contact_seconds": contact_seconds,
        "first_edible_contact_time_s": first_edible_contact_s,
        "contact_fraction_score_term": contact_fraction,
        "case_score": case_score,
        "peak_left_odor_rate_hz": peak_left_odor_hz,
        "peak_right_odor_rate_hz": peak_right_odor_hz,
        "peak_altitude_m": peak_altitude_m,
        "final_position_m": list(dynamics.state.position_m),
        "policy_phase_frames": dict(sorted(phase_counts.items())),
    }
    if not _all_finite(result):
        raise ValueError(f"non-finite rollout metric in {scenario.name}")
    return result


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(key) and _all_finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _aggregate_score(metrics: Sequence[Mapping[str, Any]]) -> float:
    if not metrics:
        raise ValueError("cannot score an empty scenario set")
    scores = [float(item["case_score"]) for item in metrics]
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("scenario scores must be finite values in [0, 1]")
    # Half mean and half worst-case keeps the optimizer from winning by
    # sacrificing every heading except its easiest case.
    return 0.50 * (sum(scores) / len(scores)) + 0.50 * min(scores)


def _evaluate_scenarios(
    env: Environment,
    config: Any,
    scenarios: Sequence[Scenario],
    *,
    duration_s: float,
    initial_hunger: float,
    odor_off: bool = False,
    controller_type: str = LEGACY_CONTROLLER_TYPE,
) -> dict[str, Any]:
    metrics = [
        _simulate_case(
            env,
            config,
            scenario,
            duration_s=duration_s,
            initial_hunger=initial_hunger,
            odor_off=odor_off,
            controller_type=controller_type,
        )
        for scenario in scenarios
    ]
    return {"aggregate_score": _aggregate_score(metrics), "scenarios": metrics}


def _config_with_parameters(
    base: Any,
    parameters: Mapping[str, float],
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> Any:
    bounds = parameter_bounds or PARAMETER_BOUNDS
    values = asdict(base)
    for name, raw_value in parameters.items():
        if name not in bounds:
            raise ValueError(f"unknown fitted parameter {name!r}")
        low, high = bounds[name]
        value = float(raw_value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        values[name] = value
    return type(base)(**values)


def _parameter_values(
    config: Any,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    bounds = parameter_bounds or PARAMETER_BOUNDS
    return {name: float(getattr(config, name)) for name in bounds}


def _cross_entropy_search(
    initial_config: Any,
    evaluator: Callable[[Any], Mapping[str, Any]],
    *,
    generations: int,
    population: int,
    seed: int,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[Any, float, list[dict[str, Any]]]:
    """Bounded deterministic cross-entropy search, maximizing evaluator score."""
    bounds = parameter_bounds or PARAMETER_BOUNDS
    rng = random.Random(seed)
    mean = _parameter_values(initial_config, bounds)
    spread = {name: 0.32 * (high - low) for name, (low, high) in bounds.items()}
    incumbent = initial_config
    incumbent_score = -math.inf
    history: list[dict[str, Any]] = []
    elite_count = max(2, math.ceil(population * 0.25))

    for generation in range(generations):
        candidates = [incumbent]
        for _ in range(population - 1):
            values = {}
            for name, (low, high) in bounds.items():
                values[name] = min(high, max(low, rng.gauss(mean[name], spread[name])))
            candidates.append(_config_with_parameters(initial_config, values, bounds))

        evaluated: list[tuple[Any, float, Mapping[str, Any]]] = []
        for config in candidates:
            details = evaluator(config)
            score = float(details["aggregate_score"])
            if not math.isfinite(score):
                raise ValueError("optimizer objective returned a non-finite score")
            evaluated.append((config, score, details))
            if score > incumbent_score:
                incumbent, incumbent_score = config, score

        evaluated.sort(key=lambda row: row[1], reverse=True)
        elites = evaluated[:elite_count]
        old_mean = mean
        for name, (low, high) in bounds.items():
            elite_values = [float(getattr(row[0], name)) for row in elites]
            elite_mean = sum(elite_values) / len(elite_values)
            mean[name] = min(high, max(low, 0.65 * elite_mean + 0.35 * old_mean[name]))
            variance = sum((value - elite_mean) ** 2 for value in elite_values) / len(elite_values)
            minimum_spread = 0.035 * (high - low)
            spread[name] = max(minimum_spread, min(spread[name], 0.80 * math.sqrt(variance)))

        history.append({
            "generation": generation + 1,
            "elite_count": elite_count,
            "best_score_this_generation": evaluated[0][1],
            "best_score_so_far": incumbent_score,
            "best_parameters_so_far": _parameter_values(incumbent, bounds),
            "population": [
                {"aggregate_score": score, "parameters": _parameter_values(config, bounds)}
                for config, score, _details in evaluated
            ],
        })
    return incumbent, incumbent_score, history


def load_policy_config(path: Path) -> tuple[ForagingPolicyConfig, dict[str, Any]]:
    """Load and validate a legacy or feedback policy artifact."""
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load fitted policy artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict) or not isinstance(artifact.get("policy_config"), dict):
        raise ValueError("fitted policy artifact must contain a policy_config object")
    if artifact.get("artifact_type") != "fitted_parameters_of_handbuilt_policy":
        raise ValueError("artifact_type must be fitted_parameters_of_handbuilt_policy")
    controller_type = str(artifact.get("controller_type", LEGACY_CONTROLLER_TYPE))
    config_type, _controller_impl, _bounds = _controller_spec(controller_type)
    try:
        config = config_type(**artifact["policy_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid policy_config in fitted artifact: {exc}") from exc
    return config, artifact


def _artifact(
    config: Any,
    *,
    world_path: Path,
    food_path: Path,
    controller_type: str,
    parameter_bounds: Mapping[str, tuple[float, float]],
    training_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _config_type, controller_impl, _bounds = _controller_spec(controller_type)
    source_paths = {
        "fit_script": Path(__file__).resolve(),
        "foraging_policy": ROOT / "scripts" / "foraging_policy.py",
        "foraging_state": ROOT / "scripts" / "foraging_state.py",
        "food_sensing": ROOT / "scripts" / "food_sensing.py",
        "flight_dynamics": ROOT / "scripts" / "flight_dynamics.py",
        "baseline_runner": ROOT / "scripts" / "run_foraging_baseline.py",
    }
    if controller_type == FEEDBACK_CONTROLLER_TYPE:
        source_paths["feedback_controller"] = Path(
            sys.modules[controller_impl.__module__].__file__
        ).resolve()
    if controller_type == LEGACY_CONTROLLER_TYPE:
        scope = POLICY_SCOPE
    else:
        scope = getattr(sys.modules[controller_impl.__module__], "FEEDBACK_SCOPE", (
            "hand-built altitude, velocity, odor and wind feedback prototype; "
            "limited to this engineered task, not MaleCNS and not biologically validated"
        ))
    return {
        "artifact_type": "fitted_parameters_of_handbuilt_policy",
        "controller_type": controller_type,
        "policy_config": asdict(config),
        "method": {
            "name": "bounded_derivative_free_cross_entropy_search",
            "optimizer": "stdlib random.Random, clipped Gaussian population, elite mean/spread update",
            "selected_parameters": list(parameter_bounds),
            "score": "per-case 0.60*approach_progress_fraction + 0.40*min(edible_contact_seconds/0.40,1); aggregate 0.50*mean + 0.50*worst-case",
        },
        "claim_boundary": {
            "controller": "hand-built controller with fitted parameter values",
            "malecns_controller": False,
            "neural_training": False,
            "biologically_validated": False,
            "policy_name": POLICY_NAME if controller_type == LEGACY_CONTROLLER_TYPE else "feedback_foraging_corridor",
            "policy_scope": scope,
        },
        "training_provenance": dict(training_provenance),
        "source_hashes": {name: _sha256(path) for name, path in source_paths.items()},
        "input_hashes": {
            "world_config": _sha256(world_path),
            "food_config": _sha256(food_path),
        },
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _new_output_dir(path: Path | None, mode: str) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = DEFAULT_OUTPUT_ROOT / f"foraging-policy-fit-{stamp}-{mode}"
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.duration_s) or not 0.5 <= args.duration_s <= 10.0:
        parser.error("--duration-s must be finite and between 0.5 and 10 seconds")
    if abs(round(args.duration_s / DT_S) * DT_S - args.duration_s) > 1.0e-9:
        parser.error("--duration-s must align to the 5 ms dynamics step")
    if not math.isfinite(args.initial_hunger) or not 0.0 <= args.initial_hunger <= 1.0:
        parser.error("--initial-hunger must be finite and in [0, 1]")
    if args.generations < 1 or args.generations > 12:
        parser.error("--generations must be in [1, 12]")
    if args.population < 4 or args.population > 32:
        parser.error("--population must be in [4, 32]")
    try:
        _finite_vector(args.wind_mps, "wind_mps", 3)
        _scenario_list(args.train_yaws_deg, "train")
        if args.holdout_yaws_deg is not None:
            _scenario_list(args.holdout_yaws_deg, "holdout")
            if args.controller is not None:
                _validate_scenario_split(
                    _scenario_list(args.train_yaws_deg, "train"),
                    _scenario_list(args.holdout_yaws_deg, "holdout"),
                )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "evaluate" and args.policy_config is None:
        parser.error("--policy-config is required with --mode evaluate")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _new_output_dir(args.out, args.mode)
    try:
        return _run_reserved(args, output_dir)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "experiment": "handbuilt_foraging_policy_parameter_fit_pilot",
            "mode": args.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": "No fit or evaluation result is asserted by this failure report.",
        }
        try:
            _write_json(output_dir / "failure.json", failure)
        except OSError:
            pass
        raise


def _run_reserved(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.mode == "train":
        controller_type = args.controller or LEGACY_CONTROLLER_TYPE
        config_type, _controller_impl, parameter_bounds = _controller_spec(controller_type)
        original_artifact: dict[str, Any] | None = None
    else:
        chosen_config, original_artifact = load_policy_config(args.policy_config)
        controller_type = str(original_artifact.get("controller_type", LEGACY_CONTROLLER_TYPE))
        if args.controller is not None and args.controller != controller_type:
            raise ValueError("--controller must match the fitted artifact controller_type")
        config_type, _controller_impl, parameter_bounds = _controller_spec(controller_type)

    train_scenarios = _scenario_list(args.train_yaws_deg, "train")
    holdout_yaws = args.holdout_yaws_deg
    if holdout_yaws is None:
        holdout_yaws = (
            DEFAULT_FEEDBACK_HOLDOUT_YAWS_DEG
            if controller_type == FEEDBACK_CONTROLLER_TYPE
            else DEFAULT_HOLDOUT_YAWS_DEG
        )
    holdout_scenarios = _scenario_list(holdout_yaws, "holdout")
    _validate_scenario_split(train_scenarios, holdout_scenarios)
    env = _load_environment(args.world_config, args.food_config, args.wind_mps, args.seed)
    baseline_config = config_type()

    if args.mode == "train":
        train_cache: dict[tuple[float, ...], dict[str, Any]] = {}

        def train_objective(config: Any) -> Mapping[str, Any]:
            key = tuple(asdict(config).values())
            if key not in train_cache:
                train_cache[key] = _evaluate_scenarios(
                    env, config, train_scenarios,
                    duration_s=args.duration_s,
                    initial_hunger=args.initial_hunger,
                    controller_type=controller_type,
                )
            return train_cache[key]

        baseline_train = train_objective(baseline_config)
        chosen_config, chosen_score, history = _cross_entropy_search(
            baseline_config,
            train_objective,
            generations=args.generations,
            population=args.population,
            seed=args.seed,
            parameter_bounds=parameter_bounds,
        )
        chosen_train = train_objective(chosen_config)
        optimization = {
            "selection_data": "training headings only; held-out results are evaluated after selection",
            "generations": args.generations,
            "population": args.population,
            "candidate_budget_upper_bound": args.generations * args.population,
            "unique_training_evaluations": len(train_cache),
            "seed": args.seed,
            "controller_type": controller_type,
            "objective_score": chosen_score,
            "training_scenarios": list(train_scenarios[i].name for i in range(len(train_scenarios))),
            "baseline": baseline_train,
            "chosen": chosen_train,
        }
    else:
        history = []
        optimization = {
            "selection_data": "none; frozen artifact only",
            "training_scenarios": [],
            "evaluated_artifact_type": original_artifact["artifact_type"],
            "controller_type": controller_type,
            "evaluated_policy_parameters": _parameter_values(chosen_config, parameter_bounds),
        }

    # These headings are never passed to the optimizer callback. Both the
    # frozen fit and default baseline are evaluated with odor on/off.
    holdout_results: dict[str, Any] = {}
    for label, config in (("baseline", baseline_config), ("chosen", chosen_config)):
        holdout_results[label] = {
            condition: _evaluate_scenarios(
                env,
                config,
                holdout_scenarios,
                duration_s=args.duration_s,
                initial_hunger=args.initial_hunger,
                odor_off=(condition == "odor_off"),
                controller_type=controller_type,
            )
            for condition in ("odor_on", "odor_off")
        }

    current_provenance = _artifact(
        chosen_config,
        world_path=env.world_path,
        food_path=env.food_path,
        controller_type=controller_type,
        parameter_bounds=parameter_bounds,
        training_provenance={"status": "not_applicable"},
    )
    if args.mode == "train":
        training_provenance = {
            "status": "completed_parameter_search",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "duration_s": args.duration_s,
            "wind_mps": list(env.wind_mps),
            "initial_hunger": args.initial_hunger,
            "controller_type": controller_type,
            "training_headings_deg": [math.degrees(s.yaw_offset_rad) for s in train_scenarios],
            "holdout_headings_deg": [math.degrees(s.yaw_offset_rad) for s in holdout_scenarios],
            "optimizer_budget": {
                "generations": args.generations,
                "population": args.population,
                "candidate_budget_upper_bound": args.generations * args.population,
                "unique_training_evaluations": optimization["unique_training_evaluations"],
            },
            "baseline_training_score": baseline_train["aggregate_score"],
            "selected_training_score": chosen_score,
            "world_config_path": str(env.world_path),
            "food_config_path": str(env.food_path),
        }
        fitted_artifact = _artifact(
            chosen_config,
            world_path=env.world_path,
            food_path=env.food_path,
            controller_type=controller_type,
            parameter_bounds=parameter_bounds,
            training_provenance=training_provenance,
        )
        fitted_artifact_hash = None
    else:
        # Keep the frozen artifact and its original fitting provenance intact.
        fitted_artifact = original_artifact
        fitted_artifact_hash = _sha256(args.policy_config.resolve())
    _write_json(output_dir / "fitted_policy.json", fitted_artifact)
    if history:
        _write_json(output_dir / "history.json", {
            "selection_data": "training headings only",
            "generations": history,
        })
    else:
        _write_json(output_dir / "history.json", {"selection_data": "none; frozen evaluation", "generations": []})

    report = {
        "schema_version": 1,
        "experiment": "handbuilt_foraging_policy_parameter_fit_pilot",
        "evidence_class": "deterministic_engineered_policy_parameter_search",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "controller_type": controller_type,
        "controller_scope": "fitted_parameters_of_handbuilt_policy",
        "policy_name": POLICY_NAME if controller_type == LEGACY_CONTROLLER_TYPE else "feedback_foraging_corridor",
        "policy_scope": current_provenance["claim_boundary"]["policy_scope"],
        "claims": {
            "malecns_autonomy": False,
            "neural_controller_fitted": False,
            "biological_validation": False,
        },
        "mode": args.mode,
        "conditions": {
            "dt_s": DT_S,
            "duration_s": args.duration_s,
            "wind_mps": list(env.wind_mps),
            "seed": args.seed,
            "initial_hunger": args.initial_hunger,
            "odor_field_kind": env.food_model.field_kind,
            "food_provenance": env.food_model.provenance,
            "controller_type": controller_type,
            "training_yaw_offsets_deg": [math.degrees(s.yaw_offset_rad) for s in train_scenarios],
            "heldout_yaw_offsets_deg": [math.degrees(s.yaw_offset_rad) for s in holdout_scenarios],
            "policy_observations": _controller_observations(controller_type),
            "environment_only_scoring_observations": [
                "3D mouth-to-edible-region distance",
                "physical edible contact duration",
            ],
        },
        "objective": {
            "per_case": "0.60*clamp((initial_region_gap-minimum_region_gap)/initial_region_gap,0,1) + 0.40*clamp(edible_contact_seconds/0.40,0,1)",
            "aggregate": "0.50*mean(per_case_scores) + 0.50*min(per_case_scores)",
            "selection_scenarios": list(scenario.name for scenario in train_scenarios),
            "heldout_used_for_selection": False,
        },
        "optimization": optimization,
        "heldout_evaluation": holdout_results,
        "fitted_parameters": _parameter_values(chosen_config, parameter_bounds),
        "artifacts": {
            "directory": str(output_dir),
            "policy_config": "fitted_policy.json",
            "history": "history.json",
            "report": "report.json",
        },
        "source_hashes": current_provenance["source_hashes"],
        "input_hashes": current_provenance["input_hashes"],
        "fitted_artifact_provenance": {
            "sha256_of_input_artifact": fitted_artifact_hash,
            "training_provenance": fitted_artifact.get("training_provenance"),
            "source_hashes": fitted_artifact.get("source_hashes"),
            "input_hashes": fitted_artifact.get("input_hashes"),
        },
    }
    _write_json(output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--controller", choices=(LEGACY_CONTROLLER_TYPE, FEEDBACK_CONTROLLER_TYPE), default=None,
                        help="controller to fit; omitted means legacy for training and artifact type for evaluation")
    parser.add_argument("--world-config", type=Path, default=DEFAULT_WORLD_CONFIG)
    parser.add_argument("--food-config", type=Path, default=DEFAULT_FOOD_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=None,
                        help="fitted_policy.json input for frozen --mode evaluate")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--wind-mps", type=float, nargs=3, default=DEFAULT_WIND_MPS,
                        metavar=("WX", "WY", "WZ"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--initial-hunger", type=float, default=1.0)
    parser.add_argument("--train-yaws-deg", type=float, nargs="+", default=DEFAULT_TRAIN_YAWS_DEG,
                        help="selection headings; must not overlap evaluation headings")
    parser.add_argument("--holdout-yaws-deg", type=float, nargs="+", default=None,
                        help="evaluation-only headings; feedback defaults to fresh 30,-60,120,-150 degree headings")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None,
                        help="new output directory under /data/artifacts/by-project/malecns-gpu/experiments by default")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        report = run(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
