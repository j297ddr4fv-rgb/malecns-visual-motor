#!/usr/bin/env python3
"""Benchmark one resident DigitalFlyRunner.

The normal launcher constructs a fitted FlyVis/MaleCNS runner and then exits.
This command constructs it once, performs an optional short warm-up, and
measures a resident run separately from output serialization.  It is intended
for performance evidence, not for changing the production simulation path.

The module-level helpers are dependency-light and accept injected runners and
clocks, which lets the unit tests exercise the timing/record contract without
importing CUDA, FlyVis, or running a GPU workload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence


DEFAULT_OUT = Path("/data/artifacts/by-project/malecns-gpu/resident-flight-benchmark")
BENCHMARK_SCHEMA = "resident-flight-benchmark-1"
ACCEPTANCE_SCHEMA = "performance-acceptance-record-1"
FRAME_DT_S = 0.005


class ResidentBenchmarkError(ValueError):
    """Raised when a resident benchmark request is malformed."""


def aligned_duration(value: float, *, name: str, allow_zero: bool = False) -> float:
    """Validate and normalize a duration to the 5 ms simulation grid."""

    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidentBenchmarkError(f"{name} must be finite") from exc
    if duration != duration or duration in (float("inf"), float("-inf")):
        raise ResidentBenchmarkError(f"{name} must be finite")
    if duration < 0.0 or (duration == 0.0 and not allow_zero):
        raise ResidentBenchmarkError(f"{name} must be positive")
    frames = round(duration / FRAME_DT_S)
    if duration == 0.0 and allow_zero:
        return 0.0
    if frames <= 0 or abs(duration - frames * FRAME_DT_S) > 1.0e-8:
        raise ResidentBenchmarkError(
            f"{name} must be a positive multiple of the {FRAME_DT_S:g}s frame period"
        )
    return float(frames * FRAME_DT_S)


def _json_safe(value: Any) -> Any:
    """Convert the small benchmark record to JSON-safe builtin values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _payload_duration(payload: Any, fallback: float) -> float:
    """Extract simulated seconds from the runner payload."""

    if isinstance(payload, Mapping):
        direct = payload.get("simulated_duration_s")
        if direct is not None:
            return float(direct)
        manifest = payload.get("manifest")
        if isinstance(manifest, Mapping):
            results = manifest.get("scenario_results")
            if isinstance(results, Mapping):
                durations = [
                    float(item.get("duration_s", 0.0))
                    for item in results.values()
                    if isinstance(item, Mapping)
                ]
                if durations:
                    return float(sum(durations))
        trajectory = payload.get("trajectory")
        if isinstance(trajectory, Mapping):
            direct = trajectory.get("duration_s")
            if direct is not None:
                return float(direct)
            rows = trajectory.get("trajectory")
            if isinstance(rows, Sequence) and rows:
                last = rows[-1]
                if isinstance(last, Mapping):
                    for key in ("time_s", "frame_time_s", "timestamp_s"):
                        if key in last:
                            return float(last[key]) + FRAME_DT_S
    return float(fallback)


def _payload_frame_count(payload: Any) -> Optional[int]:
    if not isinstance(payload, Mapping):
        return None
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return None
    rows = trajectory.get("trajectory")
    return len(rows) if isinstance(rows, Sequence) else None


def _scene_identity(path: Optional[str | Path]) -> Optional[dict[str, str]]:
    """Return stable identity metadata for an explicitly selected scene."""

    if path is None:
        return None
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ResidentBenchmarkError(
            f"cannot read scene manifest {manifest_path}: {exc}"
        ) from exc
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidentBenchmarkError(
            f"scene manifest is not valid UTF-8 JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ResidentBenchmarkError("scene manifest root must be a JSON object")
    scene_id = manifest.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ResidentBenchmarkError("scene manifest must have a non-empty scene_id")
    return {
        "manifest_path": str(manifest_path),
        "scene_id": scene_id,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _set_duration(runner: Any, duration: float) -> None:
    options = getattr(runner, "options", None)
    if options is None or not hasattr(options, "duration_override_s"):
        raise ResidentBenchmarkError(
            "resident runner must expose options.duration_override_s"
        )
    options.duration_override_s = duration
    # A frame/step cap would silently shorten the requested measurement.
    if hasattr(options, "frame_count"):
        options.frame_count = None
    if hasattr(options, "max_steps"):
        options.max_steps = None


def run_resident_benchmark(
    runner_factory: Callable[[], Any],
    *,
    warmup_duration_s: float = 0.5,
    measurement_duration_s: float = 10.0,
    keep_warmup_output: bool = False,
    out: str | Path = DEFAULT_OUT,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    """Construct one runner and benchmark its warm and measurement runs.

    ``runner_factory`` is called exactly once.  ``runner.run()`` assembles the
    simulation payload; ``runner.write_outputs(payload)`` is timed separately.
    The combined timer encloses both operations, so its RTF is not inferred by
    adding independently rounded measurements.
    """

    warmup = aligned_duration(warmup_duration_s, name="warmup_duration_s", allow_zero=True)
    measurement = aligned_duration(measurement_duration_s, name="measurement_duration_s")
    output_root = Path(out)
    output_root.mkdir(parents=True, exist_ok=True)

    def sync() -> None:
        if synchronize is not None:
            synchronize()

    setup_started = clock()
    runner = runner_factory()
    sync()
    setup_wall = clock() - setup_started
    options = getattr(runner, "options", None)
    if options is not None and hasattr(options, "out"):
        options.out = output_root

    warmup_record: dict[str, Any] = {
        "requested_duration_s": warmup,
        "wall_s": 0.0,
        "output_retained": bool(keep_warmup_output),
    }
    if warmup > 0.0:
        _set_duration(runner, warmup)
        sync()
        warmup_started = clock()
        warmup_payload = runner.run()
        sync()
        warmup_record["wall_s"] = float(clock() - warmup_started)
        warmup_record["simulated_duration_s"] = _payload_duration(warmup_payload, warmup)
        if keep_warmup_output:
            warmup_root = output_root / "warmup"
            if options is not None and hasattr(options, "out"):
                options.out = warmup_root
            warmup_record["output_paths"] = _json_safe(runner.write_outputs(warmup_payload))
            if options is not None and hasattr(options, "out"):
                options.out = output_root
    else:
        warmup_record["simulated_duration_s"] = 0.0

    _set_duration(runner, measurement)
    sync()
    combined_started = clock()
    simulation_started = clock()
    payload = runner.run()
    sync()
    simulation_wall = clock() - simulation_started
    write_started = clock()
    output_paths = runner.write_outputs(payload)
    sync()
    write_wall = clock() - write_started
    combined_wall = clock() - combined_started
    simulated = _payload_duration(payload, measurement)
    expected_frames = int(round(measurement / FRAME_DT_S))
    observed_frames = _payload_frame_count(payload)
    if abs(simulated - measurement) > 1.0e-8:
        raise ResidentBenchmarkError(
            f"runner simulated {simulated:g}s, expected {measurement:g}s"
        )
    if observed_frames is not None and observed_frames != expected_frames:
        raise ResidentBenchmarkError(
            f"runner emitted {observed_frames} frames, expected {expected_frames}"
        )
    record = {
        "schema_version": BENCHMARK_SCHEMA,
        "status": "OK",
        "scope": "resident-runner simulation+output",
        "runner_constructed_once": True,
        "setup_wall_s": float(setup_wall),
        "warmup": warmup_record,
        "measurement": {
            "requested_duration_s": measurement,
            "simulated_duration_s": simulated,
            "frame_count": observed_frames,
            "simulation_wall_s": float(simulation_wall),
            "write_wall_s": float(write_wall),
            "simulation_plus_write_wall_s": float(combined_wall),
            "rtf_simulation": float(simulated / simulation_wall) if simulation_wall > 0 else None,
            "rtf_simulation_plus_write": float(simulated / combined_wall) if combined_wall > 0 else None,
            "synchronization": {
                "before": synchronize is not None,
                "after": synchronize is not None,
                "scope": "CUDA work drained around the measured simulation and output window",
            },
            "output_paths": _json_safe(output_paths),
        },
        "output_root": str(output_root),
    }
    return _json_safe(record)


def write_json(path: str | Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(destination)


def _execution_fields(
    execution: Mapping[str, Any], backends: Mapping[str, Any], *, effective: bool = False,
) -> dict[str, Any]:
    graph = execution.get("neural_cuda_graph")
    if effective and isinstance(graph, Mapping):
        graph = graph.get("effective")
    return {
        "renderer_backend": execution.get("renderer_backend"),
        "neural_backend": execution.get("neural_backend"),
        "neural_device": execution.get("neural_device"),
        "neural_cuda_graph": graph,
        "vision_backend": execution.get("vision_backend", backends.get("vision")),
        "vision_device": execution.get("vision_device"),
    }


def _execution_evidence(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the small no-fallback subset required by the performance gate."""

    requested, effective, requested_backends, effective_backends, components = (
        value if isinstance(value, Mapping) else {}
        for value in (
            run_manifest.get("requested_execution"), run_manifest.get("effective_execution"),
            run_manifest.get("requested_backends"), run_manifest.get("effective_backends"),
            run_manifest.get("components"),
        )
    )

    component_evidence: dict[str, Any] = {}
    for name in ("world", "vision", "neural", "body"):
        value = components.get(name)
        component_evidence[name] = {
            "is_stub": value.get("is_stub") if isinstance(value, Mapping) else None
        }
    return {
        "requested_execution": _execution_fields(requested, requested_backends),
        "effective_execution": _execution_fields(effective, effective_backends, effective=True),
        "components": component_evidence,
    }


def acceptance_record(
    benchmark: Mapping[str, Any], *, run_manifest: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Create a gate-compatible record with an explicit measurement scope.

    The returned object can be installed as ``performance_acceptance`` in an
    ``engineering-manifest-1`` wrapper.  Requested/effective execution and
    component evidence remain properties of that wrapper or of the measured
    run manifest; this record only attests to the timed resident window.
    """

    measurement = benchmark.get("measurement", {})
    record = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "type": "full_run_performance_acceptance",
        "decision": "UNREVIEWED",
        "production": True,
        "evidence_class": "digital_simulation_performance",
        "runner_constructed_once": bool(benchmark.get("runner_constructed_once")),
        "simulated_duration_s": measurement.get("simulated_duration_s"),
        "wall_time_s": measurement.get("simulation_plus_write_wall_s"),
        "real_time_factor": measurement.get("rtf_simulation_plus_write"),
        "wall_time_scope": "resident-runner simulation+output",
        "setup_wall_excluded": True,
        "setup_wall_time_s": benchmark.get("setup_wall_s"),
        "scene_identity": benchmark.get("scene_identity"),
        "simulation_wall_time_s": measurement.get("simulation_wall_s"),
        "output_wall_time_s": measurement.get("write_wall_s"),
        "synchronization": measurement.get("synchronization"),
        "source_benchmark": benchmark.get("benchmark_path", benchmark.get("output_root")),
    }
    if run_manifest is not None:
        record.update(_execution_evidence(run_manifest))
    return record


def build_production_options(run_options_type: Any, *, out: Path, config: Path,
                             seed: int, max_vram_gib: float,
                             parallel_frame_stages: bool = False,
                             scene_manifest: Optional[Path] = None) -> Any:
    """Build the locked strict production option set without importing runtime modules."""

    return run_options_type(
        config=config,
        out=out,
        scenarios=("escape",),
        mode="escape-experiment",
        strict=True,
        escape_experiment=True,
        neural="malecns",
        vision="flyvis",
        body="reduced",
        seed=int(seed),
        neural_device="cuda",
        neural_backend="triton",
        neural_cuda_graph=True,
        vision_device="cuda:0",
        max_vram_gib=float(max_vram_gib),
        telemetry_mode="minimal",
        renderer_backend="warp_cuda",
        # Keep the experimental overlap out of the locked reference run until
        # a same-process GPU A/B establishes that it is a net win.  The flag
        # remains available for an explicit candidate measurement.
        parallel_frame_stages=bool(parallel_frame_stages),
        scene_manifest=scene_manifest,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=None,
                        help="strict escape config; defaults to run_digital_fly's free escape config")
    parser.add_argument("--scene-manifest", type=Path, default=None,
                        help="optional external visual scene manifest, such as config/realistic_scene.json")
    parser.add_argument("--warmup-duration", type=float, default=0.5)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--measurement-duration", type=float, default=10.0)
    parser.add_argument("--keep-warmup-output", action="store_true")
    parser.add_argument("--acceptance-record", type=Path, default=None,
                        help="optional JSON path for a performance acceptance record")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-vram-gib", type=float, default=15.5)
    parser.add_argument(
        "--parallel-frame-stages",
        action="store_true",
        help=(
            "measure the experimental sensory-prepare/vision overlap; "
            "off by default until a GPU A/B confirms a benefit"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scene_manifest = (
        None if args.scene_manifest is None else args.scene_manifest.expanduser().resolve()
    )
    scene_identity = _scene_identity(scene_manifest)
    # Keep all optional runtime imports inside main: importing this module for
    # pure timing tests must not import Torch, FlyVis, Warp, or CUDA.
    from run_digital_fly import DEFAULT_FREE_ESCAPE_CONFIG, DigitalFlyRunner, RunOptions

    config = DEFAULT_FREE_ESCAPE_CONFIG if args.config is None else args.config
    out = args.out.expanduser()
    options = build_production_options(
        RunOptions, out=out, config=config, seed=args.seed, max_vram_gib=args.max_vram_gib,
        parallel_frame_stages=args.parallel_frame_stages,
        scene_manifest=scene_manifest,
    )
    import torch

    def synchronize_cuda() -> None:
        torch.cuda.synchronize(torch.device(options.vision_device))

    started = time.perf_counter()
    record = run_resident_benchmark(
        lambda: DigitalFlyRunner(options),
        warmup_duration_s=0.0 if args.no_warmup else args.warmup_duration,
        measurement_duration_s=args.measurement_duration,
        keep_warmup_output=args.keep_warmup_output,
        out=out,
        synchronize=synchronize_cuda,
    )
    record["cli_wall_s"] = float(time.perf_counter() - started)
    record["scene_identity"] = scene_identity
    benchmark_path = out / "resident_benchmark.json"
    record["benchmark_path"] = str(benchmark_path)
    write_json(benchmark_path, record)
    if args.acceptance_record is not None:
        acceptance_path = args.acceptance_record
        if not acceptance_path.is_absolute():
            acceptance_path = out / acceptance_path
        output_paths = record.get("measurement", {}).get("output_paths", {})
        manifest_path = output_paths.get("manifest") if isinstance(output_paths, Mapping) else None
        if not isinstance(manifest_path, str) or not Path(manifest_path).is_file():
            raise ResidentBenchmarkError(
                "cannot create an acceptance record without the measured run manifest"
            )
        loaded_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, Mapping):
            raise ResidentBenchmarkError("measured run manifest must be a JSON object")
        record["acceptance_record_path"] = write_json(
            acceptance_path, acceptance_record(record, run_manifest=loaded_manifest)
        )
        # Update the benchmark with the pointer after the acceptance record is
        # written; both artifacts remain independently readable.
        write_json(benchmark_path, record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
