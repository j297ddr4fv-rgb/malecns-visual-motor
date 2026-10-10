#!/usr/bin/env python3
"""Replay fixed retinal movies through intact and rewired MaleCNS graphs."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
BUNDLE = PROJECT.parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--common-inputs", type=Path, default=BUNDLE / "data/common-inputs")
    p.add_argument("--worker-config", type=Path, default=PROJECT / "config/flygym_online_neural_worker.json")
    p.add_argument("--csr-path", type=Path, default=BUNDLE / "data/csr-flyb")
    p.add_argument("--flyvis-env", type=Path, default=Path("/data/environments/projects/flyvis-opticlobe"))
    p.add_argument("--model-root", type=Path, default=Path(os.environ.get("FLYVIS_ROOT_DIR", "/data/models/flyvis")))
    p.add_argument("--target-path", type=Path, default=PROJECT / "artifacts/flyvis-eye/t4t5_dirs.npz")
    p.add_argument("--asset-root", type=Path, default=BUNDLE / "models/optomotor-panel")
    p.add_argument("--graph-helper", type=Path, default=PROJECT / "helpers/bootstrap_retina_extension_bc.py")
    p.add_argument("--pixel-adapter", type=Path, default=PROJECT / "helpers/pixel_illusion_adapter.py")
    p.add_argument("--variant", choices=("original", "shuffle_20261010", "shuffle_20261011", "shuffle_20261012"))
    return p.parse_args()


def cases(a):
    doc = read(a.common_inputs / "conditions.json")
    if doc.get("schema") != "common-input-replay-v1":
        raise ValueError("unsupported common-input manifest")
    rows = doc["conditions"]
    if len(rows) != 9:
        raise ValueError("expected nine conditions")
    for row in rows:
        for field in ("input_file", "schedule_file", "spec_file"):
            if not (a.common_inputs / row[field]).is_file():
                raise FileNotFoundError(a.common_inputs / row[field])
        for field, expected in (
            ("schedule_file", row["schedule_sha256"]),
            ("spec_file", row["spec_sha256"]),
            ("input_file", row["input_sha256"]),
        ):
            if sha256(a.common_inputs / row[field]) != expected:
                raise ValueError(f"{field} hash mismatch: {row['case_id']}")
    return doc, rows


def pose_request(worker, seq, quaternion):
    q = quaternion.astype(float) * [1, -1, 1, -1]
    roll, pitch, yaw = worker._quat_to_rpy_wxyz(q)
    return {
        "protocol": worker.PROTOCOL, "type": "frame", "seq": seq,
        "time_s": seq * .005, "dt_s": .005,
        "pose": {
            "position_m": [0, 0, .002], "quaternion_wxyz": q.tolist(),
            "yaw_rad": yaw, "pitch_rad": pitch, "roll_rad": roll,
            "velocity_mps": [0, 0, 0], "angular_velocity_radps": [0, 0, 0],
        },
    }


def child(a):
    import numpy as np
    from functools import partial

    a.project, a.out = a.project.resolve(), a.out.resolve()
    a.common_inputs, a.csr_path = a.common_inputs.resolve(), a.csr_path.resolve()
    os.environ.update({
        "MALECNS_CSR": str(a.csr_path),
        "FLYVIS_ENV_ROOT": str(a.flyvis_env.resolve()),
        "FLYVIS_ROOT_DIR": str(a.model_root.resolve()),
        "FLYVIS_T4T5_TARGETS": str(a.target_path.resolve()),
        "MALECNS_ASSET_ROOT": str(a.asset_root.resolve()),
    })
    sys.path.insert(0, str(a.project / "scripts"))
    sys.path.insert(0, str(a.pixel_adapter.parent.resolve()))
    import malecns_gate
    import pixel_illusion_adapter

    malecns_gate.cuda_blockers = partial(malecns_gate.cuda_blockers, min_available_gib=0.0)
    pixel_illusion_adapter.install_worker_renderer()
    helper = module("publication_graph_helper", a.graph_helper)
    helper.SCRIPTS, helper.PROJECT = a.project / "scripts", a.project
    os.environ["BC_GRAPH_SHUFFLE_SEED"] = "20261010" if a.variant == "original" else a.variant.rsplit("_", 1)[-1]
    helper._patch_graph_loader("original" if a.variant == "original" else "degree_preserving_target_shuffle")
    worker_module = module(
        "publication_neural_worker",
        a.project / "scripts/flygym_online_neural_worker_eye_parity_state_v2.py",
    )
    manifest, selected = cases(a)
    previous = None
    frame_count = int(manifest["frames"])
    for case in selected:
        case_id = case["case_id"]
        run = a.out / "runs" / a.variant / case_id
        run.mkdir(parents=True, exist_ok=True)
        if (run / "summary.json").exists():
            continue
        os.environ.update({
            "FLYGYM_EXTENSION_SCHEDULE_PATH": str(a.common_inputs / case["schedule_file"]),
            "FLYGYM_EXTENSION_SCHEDULE_SHA256": case["schedule_sha256"],
            "FLYGYM_EXTENSION_CASE_ID": case_id,
        })
        argv = [
            "--config", str(a.worker_config.resolve()), "--csr-path", str(a.csr_path),
            "--stimulus-condition", "angular", "--angular-stimulus-spec",
            str(a.common_inputs / case["spec_file"]), "--seed", str(case["seed"]),
            "--max-frames", str(frame_count), "--static-warmup-s", "1.0",
            "--capture-mode", "light", "--deterministic-vision",
            "--deterministic-neural-delivery", "--out", str(run / "worker"),
        ]
        worker = worker_module.OnlineNeuralWorker(worker_module._parse_cli(argv), reuse_from=previous)
        with np.load(a.common_inputs / case["input_file"], allow_pickle=False) as data:
            pixels, quaternions = data["pixels"].copy(), data["thorax_quat_wxyz"].copy()
            saved_targets = data["target_indices"].copy()
        if pixels.shape != (frame_count, 2, 721) or quaternions.shape != (frame_count, 4):
            raise ValueError(f"unexpected common input shape: {case_id}")
        if not np.array_equal(saved_targets, worker.target_indices):
            raise ValueError(f"FlyVis target order differs from common input: {case_id}")

        retinal_hash = hashlib.sha256()
        drive_hash = hashlib.sha256(np.ascontiguousarray(worker.target_indices).tobytes())
        original_render, original_step = worker._render_columns, worker.engine.step_window

        def render_shared(pose, time_s):
            seq = round(time_s / .005)
            return pixels[seq, 0], pixels[seq, 1], float(time_s)

        def record_drive(sensory, ts, **kwargs):
            values = np.ascontiguousarray(sensory.channels["flyvis_t4t5"]["values"], dtype=np.float32)
            drive_hash.update(values.tobytes())
            return original_step(sensory, ts, **kwargs)

        worker._render_columns = render_shared
        worker.engine.step_window = record_drive
        records, start = [], time.monotonic()
        state = read(a.out / "state.json")
        state["current_run"] = a.variant + "/" + case_id
        write(a.out / "state.json", state)
        for seq in range(frame_count):
            reply = worker.process_frame(pose_request(worker_module, seq, quaternions[seq]))
            retinal_hash.update(pixels[seq].tobytes())
            rates, named = reply["population_rates_hz"], worker.engine.last_named_outputs
            records.append({
                "seq": seq, "frame_time_s": seq * .005,
                "DNp15_L_hz": rates["DNp15/L"], "DNp15_R_hz": rates["DNp15/R"],
                "raw_P": reply["leg_power"], "network_spikes": named["spike_count"],
            })
        worker._render_columns, worker.engine.step_window = original_render, original_step
        retinal_digest, drive_digest = retinal_hash.hexdigest(), drive_hash.hexdigest()
        if a.variant != "original":
            baseline = read(a.out / "runs/original" / case_id / "summary.json")
            if retinal_digest != baseline["retinal_movie_sha256"] or drive_digest != baseline["fitted_drive_stream_sha256"]:
                raise RuntimeError("shared retinal or fitted-current stream differs from intact reference")
        stopped = worker.stop()
        with (run / "neural_trace.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        active = records[200:]
        summary = {
            "variant": a.variant, "case_id": case_id, "condition": case["condition"],
            "initialization_seed": case["seed"],
            "rewiring_seed": None if a.variant == "original" else int(a.variant.rsplit("_", 1)[-1]),
            "frames": frame_count, "retinal_movie_sha256": retinal_digest,
            "fitted_drive_stream_sha256": drive_digest, "matched_retina_and_fitted_drive": True,
            "DNp15_L_hz": float(np.mean([r["DNp15_L_hz"] for r in active])),
            "DNp15_R_hz": float(np.mean([r["DNp15_R_hz"] for r in active])),
            "DNp15_opponent_hz": float(np.mean([r["DNp15_L_hz"] - r["DNp15_R_hz"] for r in active])),
            "mean_raw_P": float(np.mean([r["raw_P"] for r in active])),
            "mean_network_spikes_per_5ms": float(np.mean([r["network_spikes"] for r in active])),
            "wall_s": time.monotonic() - start,
            "neural_worker_stopped": stopped.get("type") == "stopped",
            "no_body_simulation_or_feedback": True,
        }
        write(run / "network_intervention.json", helper.GRAPH_RECORD)
        write(run / "summary.json", summary)
        state = read(a.out / "state.json")
        state["completed_runs"] += 1
        state["runs"].append(summary)
        state["current_run"] = None
        write(a.out / "state.json", state)
        print(json.dumps({"completed": state["completed_runs"], "summary": summary}), flush=True)
        previous = worker


def parent(a):
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "state.json").exists():
        raise FileExistsError("output already contains a queue state; inspect before resuming")
    manifest, _ = cases(a)
    write(a.out / "protocol.json", {
        "design": "common-input replay of recorded intact-body retinal movies",
        "conditions_manifest": str((a.common_inputs / "conditions.json").resolve()),
        "stimuli": ["static", "positive60", "negative60"],
        "initialization_seeds": [1729, 1730, 1731],
        "initial_variants": ["original", "shuffle_20261010"],
        "additional_rewiring_seeds": [20261011, 20261012],
        "frames_per_run": manifest["frames"], "dt_s": manifest["dt_s"],
        "pose_reflection": "quaternion (w,x,y,z) -> (w,-x,y,-z)",
        "shared_retinal_and_current_hashes_required": True,
        "no_body_actuation": True,
    })
    write(a.out / "state.json", {
        "status": "running", "completed_runs": 0, "total_runs": 18,
        "current_run": None, "runs": [],
    })
    variants = ["original", "shuffle_20261010"]
    index = 0
    try:
        while index < len(variants):
            variant = variants[index]
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--project", str(a.project), "--out", str(a.out),
                "--common-inputs", str(a.common_inputs),
                "--worker-config", str(a.worker_config),
                "--csr-path", str(a.csr_path), "--flyvis-env", str(a.flyvis_env),
                "--model-root", str(a.model_root), "--target-path", str(a.target_path),
                "--asset-root", str(a.asset_root), "--graph-helper", str(a.graph_helper),
                "--pixel-adapter", str(a.pixel_adapter), "--variant", variant,
            ]
            subprocess.run(command, check=True)
            if index == 1:
                state = read(a.out / "state.json")
                lookup = {(r["variant"], r["initialization_seed"], r["condition"]): r for r in state["runs"]}
                effects = {
                    arm: {
                        condition: sum(
                            lookup[arm, seed, condition]["DNp15_opponent_hz"]
                            - lookup[arm, seed, "static"]["DNp15_opponent_hz"]
                            for seed in (1729, 1730, 1731)
                        ) / 3
                        for condition in ("positive60", "negative60")
                    }
                    for arm in variants
                }
                intact, null = effects["original"], effects["shuffle_20261010"]
                extend = (
                    intact["positive60"] > 0 and intact["negative60"] < 0
                    and sum(abs(v) for v in null.values()) < .5 * sum(abs(v) for v in intact.values())
                )
                state["first_graph_effects_hz"] = effects
                state["additional_graphs_scheduled"] = extend
                if extend:
                    variants.extend(["shuffle_20261011", "shuffle_20261012"])
                    state["total_runs"] = 36
                write(a.out / "state.json", state)
            index += 1
        state = read(a.out / "state.json")
        state.update(status="complete", current_run=None)
        write(a.out / "state.json", state)
    except BaseException as error:
        state = read(a.out / "state.json")
        state.update(status="failed", error=str(error))
        write(a.out / "state.json", state)
        raise


if __name__ == "__main__":
    args = arguments()
    child(args) if args.variant else parent(args)
