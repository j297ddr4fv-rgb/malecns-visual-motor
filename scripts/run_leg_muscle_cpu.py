#!/usr/bin/env python3
"""Run explicit motor-neuron interventions through CPU articulated physics.

Default source is the actual full MaleCNS NumPy LIF engine. The optional
prescribed-rates source is an explicitly labelled motor-interface control,
not a connectome result. No movement sequence or body trajectory is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sys

import numpy as np

from articulated_leg_body import ArticulatedLegBody
from leg_muscle_decoder import LegMuscleDecoder
from leg_muscle_pipeline import LegMusclePipeline
from neuromuscular_adapter import AnnotationIndex, DEFAULT_CSR
from leg_muscle_neural_backend import (build_leg_neural_engine, neural_backend_preflight,
                                      NeuralBackendGateError, DEFAULT_MAX_VRAM_GIB)

DEFAULT_OUT = Path("/data/artifacts/by-project/malecns-gpu/experiments/articulated-neuromuscular-cpu-20261002")
CASES = ("sham", "flexor", "extensor", "extensor-no-friction")
LEG_SELECTORS = {"LF": "fl/L", "RF": "fl/R", "LM": "ml/L",
                 "RM": "ml/R", "LH": "hl/L", "RH": "hl/R"}
CSR_FILES = ("ids.npy", "offsets.npy", "target.npy", "weight.npy", "sign.npy",
             "annotations.npz", "tables.json", "meta.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_spikes(spikes: object, n: int) -> np.ndarray:
    """Preserve per-neuron identity; refuse malformed or nonbinary outputs."""
    value = np.asarray(spikes)
    if value.shape != (n,) or value.dtype.kind not in "biuf":
        raise ValueError("neural spikes must be a numeric dense vector in CSR order")
    if not np.isfinite(value).all() or ((value != 0) & (value != 1)).any():
        raise ValueError("per-tick neural spikes must be finite and binary")
    return np.ascontiguousarray(value, dtype=np.uint8)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_case(case: str, out: Path, duration_s: float = 0.3,
             source: str = "malecns", fixed_root: bool = False,
             current_mV_per_ms: float = 2.0, leg: str = "LF",
             muscle_group: str | None = None, neural_backend: str = "cpu",
             seed: int = 1729, max_vram_gib: float = DEFAULT_MAX_VRAM_GIB,
             csr_path: Path = DEFAULT_CSR, passive_support: bool | None = None,
             muscle_force_scale: float = 1.0, sensory_feedback: str = "off",
             progress: bool = False) -> dict:
    if case not in CASES:
        raise ValueError("unknown case")
    if leg not in LEG_SELECTORS:
        raise ValueError("unknown leg")
    if muscle_group not in (None, "sternotrochanter", "tergotrochanter"):
        raise ValueError("unknown optional muscle probe group")
    if muscle_group is not None and case == "flexor":
        raise ValueError("optional extensor groups cannot be used for the flexor case")
    if not math.isfinite(duration_s) or duration_s < 0.025 or duration_s > 10.0:
        raise ValueError("duration_s must be in [0.025, 10] for this bounded probe")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean")
    if source not in ("malecns", "prescribed-rates"):
        raise ValueError("unknown neural source")
    if neural_backend not in ("cpu", "torch_cuda", "triton"):
        raise ValueError("neural_backend must be cpu, torch_cuda or triton")
    if source != "malecns" and neural_backend != "cpu":
        raise ValueError("CUDA backend requires the actual malecns neural source")
    if passive_support is not None and not isinstance(passive_support, bool):
        raise ValueError("passive_support must be a boolean or None")
    if not math.isfinite(muscle_force_scale) or not 0 < muscle_force_scale <= 10:
        raise ValueError("muscle_force_scale must be finite and in (0,10]")
    if sensory_feedback not in ("off", "on", "muted"):
        raise ValueError("sensory_feedback must be off, on or muted")
    if source != "malecns" and sensory_feedback != "off":
        raise ValueError("sensory feedback requires the actual malecns neural source")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    if not math.isfinite(max_vram_gib) or not 0 < max_vram_gib <= DEFAULT_MAX_VRAM_GIB:
        raise ValueError("max_vram_gib must be finite and in (0,15.5]")
    if not math.isfinite(current_mV_per_ms) or not 0 <= current_mV_per_ms <= 5.0:
        raise ValueError("current_mV_per_ms must be in [0,5]")
    dt_s = 0.0005
    count = round(duration_s / dt_s)
    if abs(count * dt_s - duration_s) > 1e-10:
        raise ValueError("duration_s must be a multiple of 0.0005")
    # Gate before loading CSR/model data or creating a run directory. The
    # engine rechecks immediately before GPU imports to catch state changes.
    preflight = neural_backend_preflight(neural_backend)
    if not preflight["gate_passed"]:
        raise NeuralBackendGateError(preflight)
    csr_path = Path(csr_path)
    # Never mix artifacts from different commands or overwrite prior evidence.
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_paths = (Path(__file__), Path(__file__).with_name("leg_muscle_pipeline.py"),
                    Path(__file__).with_name("leg_muscle_decoder.py"),
                    Path(__file__).with_name("leg_muscle_dynamics.py"),
                    Path(__file__).with_name("leg_muscle_parameters.py"),
                    Path(__file__).with_name("leg_sensory_feedback.py"),
                    Path(__file__).with_name("articulated_leg_body.py"),
                    Path(__file__).with_name("malecns_engine.py"),
                    Path(__file__).with_name("neuromuscular_adapter.py"),
                    Path(__file__).with_name("leg_muscle_neural_backend.py"),
                    Path(__file__).with_name("run_leg_muscle_gpu.py"),
                    Path(__file__).with_name("malecns_gate.py"),
                    Path(__file__).with_name("twin_contracts.py"),
                    Path(__file__).with_name("population_selectors.py"),
                    Path(__file__).resolve().parents[1] / "config" / "sensory_map.json",
                    Path(__file__).resolve().parents[1] / "config" / "leg_muscle_map.json",
                    Path(__file__).resolve().parents[1] / "config" / "leg_muscle_parameters.json",
                    Path(__file__).resolve().parents[1] / "config" / "leg_sensory_feedback.json",
                    Path(__file__).resolve().parents[1] / "config" / "articulated_leg_body.json")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    snapshots = out / "sources"
    snapshots.mkdir()
    for path in source_paths:
        (snapshots / path.name).write_bytes(path.read_bytes())
    csr_hashes = {name: file_sha256(csr_path / name) for name in CSR_FILES}
    index = AnnotationIndex(csr_path)
    decoder = LegMuscleDecoder(csr_path=csr_path, population_index=index)
    mapped_motor_indices = np.unique(np.concatenate([
        index.resolve(c["selector"]) for c in decoder.provenance()["channels"].values()]))
    body = ArticulatedLegBody(fixed_root=fixed_root, passive_support=passive_support,
                              **({"friction": 0.0} if case.endswith("no-friction") else {}))
    pipe = LegMusclePipeline(decoder=decoder, body=body, muscle_force_scale=muscle_force_scale)
    feedback_adapter = None
    if sensory_feedback != "off":
        from leg_sensory_feedback import LegSensoryFeedback
        feedback_adapter = LegSensoryFeedback(index)
    selected_group = muscle_group or ("tibia_flexor" if case == "flexor" else "tibia_extensor")
    selected_channel = f"{leg}_{selected_group}"
    type_name = next(c["type_name"] for c in decoder.channels if c["name"] == selected_channel)
    selector = f"type:{type_name}&superclass:vnc_motor&subclass:{LEG_SELECTORS[leg]}"
    selected = index.resolve(selector)
    if not selected.size:
        raise ValueError("requested intervention resolves to no neurons")
    arms = body.muscles[selected_channel].joint_moment_arms_m
    if not arms:
        raise ValueError("selected muscle has no declared mechanical path")
    joint_name = max(arms, key=lambda name: abs(arms[name]))
    neural = None
    if source == "malecns":
        from twin_contracts import SensoryFrame, Timestamp
        neural = build_leg_neural_engine(csr_path, backend=neural_backend, seed=seed,
                                        max_vram_gib=max_vram_gib)
    actual_backend = neural.backend if neural is not None else "none"
    actual_device = neural.device if neural is not None else "none"
    gpu_used = actual_device == "cuda"
    spike_digest = hashlib.sha256()
    initial = body.snapshot()
    previous_body = initial["telemetry"]
    afferent_indices = None
    total_afferent_spikes = 0
    total_mapped_motor_spikes = 0
    maximum_sensory_current = 0.0
    intervention_start_s = duration_s * 0.25
    intervention_end_s = duration_s * 0.65
    total_spikes = target_spikes = 0
    intervention_ticks = []
    maximum_active_force_N = 0.0
    peak_tangential_force_N = 0.0
    tangential_impulse_Ns = np.zeros(3)
    normal_impulse_Ns = 0.0
    all_ground_impulse_Ns = np.zeros(3)
    nonfoot_ground_contact_ticks = 0
    final = None
    records = []
    record_stride = max(1, round(0.005 / dt_s))
    initialization_s = time.monotonic() - started
    stage_totals = dict.fromkeys(("neural_s", "muscle_physics_s", "sensory_encode_s", "other_s"), 0.)
    window_totals = dict(stage_totals)
    windows = []
    window_start_tick = 0
    window_elapsed = 0.
    window_ticks = round(1.0 / dt_s)
    loop_started = time.perf_counter()
    with (out / "telemetry.jsonl").open("w", encoding="utf-8") as fh, \
         (out / "neural_spike_counts.jsonl").open("w", encoding="utf-8") as counts_fh, \
         (out / "sensory_feedback.jsonl").open("w", encoding="utf-8") as feedback_fh:
        for tick in range(count):
            tick_started = time.perf_counter()
            neural_elapsed = sensory_elapsed = physics_elapsed = 0.
            t_s = tick * dt_s
            on = case != "sham" and intervention_start_s <= t_s < intervention_end_s
            if on:
                intervention_ticks.append(tick)
            spike_count = 0
            selected_spike_count = 0
            afferent_spike_count = 0
            mapped_motor_spike_count = 0
            feedback_record = None
            sensory_currents = {}
            if feedback_adapter is not None:
                sense_started = time.perf_counter()
                feedback_record = feedback_adapter.encode(
                    previous_body, t_s, enabled=True, silenced=sensory_feedback == "muted")
                if abs(float(feedback_record["source_time_s"]) - t_s) > 1e-10:
                    raise RuntimeError("feedback must use the state at this tick boundary")
                sensory_currents = feedback_record["currents_mV_per_ms"]
                maximum_sensory_current = max(maximum_sensory_current,
                    max((abs(float(v)) for v in sensory_currents.values()), default=0.0))
                if afferent_indices is None:
                    afferent_indices = np.unique(np.concatenate(
                        [index.resolve(selector) for selector in sensory_currents]))
                sensory_elapsed = time.perf_counter() - sense_started
            if neural is not None:
                timestamp = Timestamp(t_s=t_s, tick=tick, rate_hz=1.0 / dt_s)
                currents = dict(sensory_currents)
                if on:
                    if selector in currents:
                        raise RuntimeError("sensory targets must not replace motor interventions")
                    currents[selector] = current_mV_per_ms
                channels = {"current_mV_per_ms": currents} if currents else {}
                neural_started = time.perf_counter()
                observation = neural.step(SensoryFrame(timestamp, channels=channels), timestamp)
                neural_elapsed = time.perf_counter() - neural_started
                spikes = canonical_spikes(observation.spikes, index.n)
                spike_digest.update(spikes.tobytes(order="C"))
                spike_count = int(spikes.sum())
                selected_spike_count = int(spikes[selected].sum())
                mapped_motor_spike_count = int(spikes[mapped_motor_indices].sum())
                total_mapped_motor_spikes += mapped_motor_spike_count
                if afferent_indices is not None:
                    afferent_spike_count = int(spikes[afferent_indices].sum())
                total_afferent_spikes += afferent_spike_count
                target_spikes += selected_spike_count
                counts_fh.write(json.dumps({"tick": tick, "t_s": t_s,
                    "total_spikes": spike_count, "target_spikes": selected_spike_count,
                    "afferent_spikes": afferent_spike_count,
                    "mapped_motor_spikes": mapped_motor_spike_count}, allow_nan=False) + "\n")
                physics_started = time.perf_counter()
                final = pipe.step(dt_s=dt_s, spikes=spikes)
                physics_elapsed = time.perf_counter() - physics_started
            else:
                prescribed = np.zeros(index.n)
                if on:
                    prescribed[selected] = 30.0
                physics_started = time.perf_counter()
                final = pipe.step(dt_s=dt_s, rates_hz=prescribed)
                physics_elapsed = time.perf_counter() - physics_started
            total_spikes += spike_count
            if feedback_record is not None:
                feedback_record["tick"] = tick
                feedback_record["afferent_spikes"] = afferent_spike_count
                feedback_fh.write(json.dumps(feedback_record, allow_nan=False) + "\n")
                final["sensory_feedback"] = feedback_record
            previous_body = final["body"]
            final["neural"] = {"source": source, "intervention_active": on,
                               "total_spikes": spike_count, "target_spikes": selected_spike_count,
                               "mapped_motor_spikes": mapped_motor_spike_count,
                               "afferent_spikes": afferent_spike_count,
                               "tick": tick}
            selected_muscle = final["muscles"][selected_channel]
            maximum_active_force_N = max(maximum_active_force_N,
                                         selected_muscle["active_force_N"])
            for foot in final["body"]["feet"].values():
                tangent = np.asarray(foot["tangential_force_N"], dtype=float)
                tangential_impulse_Ns += tangent * dt_s
                peak_tangential_force_N = max(peak_tangential_force_N, float(np.linalg.norm(tangent)))
                normal_impulse_Ns += float(foot["normal_force_N"]) * dt_s
            nonfoot_ground_contact_ticks += int(final["body"]["body_ground_contact"])
            for contact in final["body"]["body_contacts"]:
                if contact["ground_contact"]:
                    sign = 1.0 if contact["geom1"] == "ground" else -1.0
                    all_ground_impulse_Ns += sign * np.asarray(contact["force_on_geom2_N"]) * dt_s
            if tick % record_stride == 0 or tick == count - 1:
                fh.write(json.dumps(final, allow_nan=False) + "\n")
                records.append(final)
            tick_elapsed = time.perf_counter() - tick_started
            parts = {"neural_s": neural_elapsed, "muscle_physics_s": physics_elapsed,
                     "sensory_encode_s": sensory_elapsed,
                     "other_s": max(0., tick_elapsed-neural_elapsed-physics_elapsed-sensory_elapsed)}
            for key, value in parts.items():
                stage_totals[key] += value
                window_totals[key] += value
            window_elapsed += tick_elapsed
            if (tick+1) % window_ticks == 0 or tick == count-1:
                windows.append({"start_tick": window_start_tick, "end_tick": tick,
                    "simulated_end_s": (tick+1)*dt_s, "elapsed_s": window_elapsed,
                    "stages_s": dict(window_totals)})
                if progress:
                    print(json.dumps({"event": "progress", "case": case,
                        "backend": actual_backend, "simulated_s": (tick+1)*dt_s,
                        "loop_wall_s": time.perf_counter()-loop_started,
                        "height_m": final["body"]["root"]["position_m"][2],
                        "spikes": total_spikes, "mapped_motor_spikes": total_mapped_motor_spikes}), flush=True)
                window_start_tick = tick+1
                window_elapsed = 0.
                window_totals = dict.fromkeys(stage_totals, 0.)
    loop_wall_s = time.perf_counter() - loop_started
    performance = {"schema_version": "leg-neuromuscular-performance-1",
        "initialization_s": initialization_s, "simulation_loop_s": loop_wall_s,
        "stages_s": stage_totals, "windows": windows,
        "neural_ticks": count, "simulated_duration_s": duration_s,
        "stage_semantics": {"neural_s": "neural.step including host observation transfer and synchronization",
            "muscle_physics_s": "decoder, rate filter, Hill muscle substeps and MuJoCo physics/telemetry",
            "sensory_encode_s": "physical-state to afferent current mapping",
            "other_s": "dense spike hashing/counts, input assembly and telemetry serialization/I/O"},
        "scope": "host wall-clock stages; not standalone GPU kernel timing or biological validation"}
    write_json(out / "performance.json", performance)
    assert final is not None
    final_state = body.snapshot()
    initial_body = initial["telemetry"]
    final_body = final_state["telemetry"]
    delta = np.asarray(final_body["root"]["position_m"]) - np.asarray(initial_body["root"]["position_m"])
    joint_delta = final_body["joints"][joint_name]["q_rad"] - initial_body["joints"][joint_name]["q_rad"]
    authority = ("sham_no_external_neural_drive" if case == "sham" else
                 "explicit_motor_neuron_current_intervention" if neural is not None else
                 "prescribed_per_neuron_rate_control")
    if sensory_feedback == "on":
        authority = ("body_derived_sensory_current_proxy" if case == "sham" else
                     authority + "+body_derived_sensory_current_proxy")
    summary = {
        "case": case, "leg": leg, "source": source, "evidence_class": "digital_simulation",
        "biological_validation": "NOT_VALIDATED",
        "autonomous_behavior": False,
        "control_authority": authority,
        "backend": f"{actual_backend}+mujoco_cpu" if neural is not None else "mujoco_cpu",
        "neural_backend_requested": neural_backend, "neural_backend_actual": actual_backend,
        "neural_device_actual": actual_device, "physics_backend": "mujoco_cpu",
        "seed": seed, "max_vram_gib": max_vram_gib,
        "neural_observation_dtype": "uint8" if neural is not None else None,
        "neural_spike_encoding": "tick-major dense uint8 C-order" if neural is not None else None,
        "neural_spike_shape": [count, index.n] if neural is not None else None,
        "full_neural_trace_ticks": count if neural is not None else 0,
        "neural_spike_sha256": spike_digest.hexdigest() if neural is not None else None,
        "gpu_used": gpu_used, "fixed_root": fixed_root,
        "sensory_feedback_mode": sensory_feedback,
        "muscle_force_scale": muscle_force_scale,
        "passive_support_enabled": body.passive_support,
        "afferent_spikes": total_afferent_spikes,
        "mapped_motor_spikes": total_mapped_motor_spikes,
        "maximum_sensory_current_mV_per_ms": maximum_sensory_current,
        "duration_s": duration_s, "neural_dt_s": dt_s,
        "wall_s": time.monotonic() - started,
        "simulation_loop_wall_s": loop_wall_s, "initialization_wall_s": initialization_s,
        "selector": selector, "selected_neuron_count": int(selected.size),
        "muscle_group": selected_group, "selected_channel": selected_channel,
        "selected_body_ids": [int(x) for x in index.ids[selected]],
        "intervention_window_s": [intervention_start_s, intervention_end_s],
        "effective_intervention_window_s": [intervention_ticks[0]*dt_s,
                                             (intervention_ticks[-1]+1)*dt_s]
                                            if intervention_ticks else None,
        "current_mV_per_ms": current_mV_per_ms if neural is not None else None,
        "prescribed_rate_hz": 30.0 if neural is None else None,
        "total_spikes": total_spikes, "target_spikes": target_spikes,
        "maximum_selected_active_muscle_force_N": maximum_active_force_N,
        "selected_joint": joint_name, "selected_joint_delta_rad": float(joint_delta),
        "root_displacement_m": delta.tolist(),
        "center_of_mass_displacement_m": (np.asarray(final_body["center_of_mass_position_m"]) -
                                          np.asarray(initial_body["center_of_mass_position_m"])).tolist(),
        "all_ground_contact_impulse_Ns": all_ground_impulse_Ns.tolist(),
        "contact_impulse_sampling_dt_s": dt_s,
        "nonfoot_ground_contact_fraction": nonfoot_ground_contact_ticks / count,
        "peak_foot_tangential_force_N": peak_tangential_force_N,
        "summed_foot_tangential_impulse_Ns": tangential_impulse_Ns.tolist(),
        "summed_foot_normal_impulse_Ns": normal_impulse_Ns,
        "initial": initial, "final": final_state,
    }
    manifest = {"pipeline": pipe.provenance(),
                "neural": neural.manifest() if neural is not None else {"source": source},
                "summary": {k: v for k, v in summary.items() if k not in ("initial", "final")}}
    manifest["source_sha256"] = source_hashes
    manifest["source_snapshot_directory"] = "sources"
    manifest["csr_sha256"] = csr_hashes
    manifest["neural_backend_preflight"] = preflight
    manifest["performance"] = {"path": "performance.json", "sha256": file_sha256(out / "performance.json")}
    manifest["sensory_feedback"] = {
        "mode": sensory_feedback,
        "input_muting_only": sensory_feedback == "muted",
        "causality": "previous physics result at current tick boundary -> neural step -> next physics result",
        "adapter": feedback_adapter.provenance() if feedback_adapter is not None else None,
    }
    manifest["sensory_feedback_sha256"] = file_sha256(out / "sensory_feedback.jsonl")
    if source_hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}:
        raise RuntimeError("source files changed during simulation; refuse to accept this run")
    manifest["telemetry_sha256"] = hashlib.sha256((out / "telemetry.jsonl").read_bytes()).hexdigest()
    manifest["neural_counts_sha256"] = file_sha256(out / "neural_spike_counts.jsonl")
    if csr_hashes != {name: file_sha256(csr_path / name) for name in CSR_FILES}:
        raise RuntimeError("CSR data changed during simulation; refuse to accept this run")
    for path in source_paths:
        if file_sha256(snapshots / path.name) != source_hashes[path.name]:
            raise RuntimeError("source snapshot does not match the executed-source digest")
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", summary)
    try:
        plot_case(records, out / "trace.png", case, leg, selected_group, joint_name, actual_backend)
    except ImportError:
        pass
    return summary


def plot_case(records: list[dict], path: Path, title: str, leg: str = "LF",
              group: str = "tibia_extensor", joint: str | None = None,
              neural_backend: str = "numpy_cpu") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["time_s"] for r in records]
    names = (f"{leg}_tibia_flexor", f"{leg}_tibia_extensor") if group.startswith("tibia_") else (
        f"{leg}_trochanter_flexor", f"{leg}_{group}")
    joint = joint or f"{leg}_tibia_pitch"
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    for name in names:
        axes[0].plot(t, [r["muscles"][name]["activation"] for r in records], label=name)
        axes[1].plot(t, [r["muscles"][name]["active_force_N"] * 1e6 for r in records], label=name)
    axes[0].set_ylabel("Activation")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Active force (uN)")
    axes[2].plot(t, [r["body"]["joints"][joint]["q_rad"] for r in records])
    axes[2].set_ylabel(f"{joint} (rad)")
    initial = np.array(records[0]["body"]["root"]["position_m"])
    for k, axis in enumerate(("x", "y", "z")):
        axes[3].plot(t, [(r["body"]["root"]["position_m"][k] - initial[k]) * 1e3 for r in records], label=axis)
    axes[3].set_ylabel("Body change (mm)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=8)
    axes[0].set_title(f"{neural_backend} neuron-muscle-contact probe: {leg} {group} {title}")
    for a in axes:
        a.grid(alpha=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None, *, default_neural_backend: str = "cpu",
         default_out: Path = DEFAULT_OUT, default_duration: float = 0.3) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="extensor")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--leg", choices=tuple(LEG_SELECTORS), default="LF")
    parser.add_argument("--muscle-group", choices=("sternotrochanter", "tergotrochanter"))
    parser.add_argument("--source", choices=("malecns", "prescribed-rates"), default="malecns")
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument("--neural-backend", choices=("cpu", "torch_cuda", "triton"), default=default_neural_backend)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-vram-gib", type=float, default=DEFAULT_MAX_VRAM_GIB)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--fixed-root", action="store_true")
    parser.add_argument("--disable-passive-support", action="store_true")
    parser.add_argument("--muscle-force-scale", type=float, default=1.0)
    parser.add_argument("--sensory-feedback", choices=("off", "on", "muted"), default="off")
    parser.add_argument("--current-mV-per-ms", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args(argv)
    if args.suite and args.muscle_group is not None:
        parser.error("--muscle-group requires a single --case; the standard suite includes a tibia flexor probe")
    if args.source != "malecns" and args.neural_backend != "cpu":
        parser.error("--neural-backend CUDA modes require --source malecns")
    if args.preflight_only:
        report = neural_backend_preflight(args.neural_backend)
        report["gpu_runtime_imported"] = [name for name in ("torch", "triton", "warp") if name in sys.modules]
        print(json.dumps(report, indent=2))
        return 0 if report["gate_passed"] else 2
    cases = CASES if args.suite else (args.case,)
    for case in cases:
        try:
            result = run_case(case, args.out / case, duration_s=args.duration,
                              source=args.source, fixed_root=args.fixed_root,
                              current_mV_per_ms=args.current_mV_per_ms, leg=args.leg,
                              muscle_group=args.muscle_group, neural_backend=args.neural_backend,
                              seed=args.seed, max_vram_gib=args.max_vram_gib,
                              passive_support=False if args.disable_passive_support else None,
                              muscle_force_scale=args.muscle_force_scale,
                              sensory_feedback=args.sensory_feedback, progress=args.progress)
        except Exception as exc:
            failure_dir = args.out / case
            if not isinstance(exc, FileExistsError) and failure_dir.exists() and not (failure_dir / "manifest.json").exists():
                write_json(failure_dir / "failure.json", {"status": "FAILED",
                           "case": case, "error_type": type(exc).__name__, "error": str(exc)})
            raise
        print(json.dumps({k: v for k, v in result.items() if k not in ("initial", "final")}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
