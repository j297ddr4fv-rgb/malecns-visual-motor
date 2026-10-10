#!/usr/bin/env python3
"""Prepare, launch, resume, or summarize a single resident visual-motor queue."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

from artifact_common import read_json, sha256_file, utc_now_iso, write_json

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
DATA = Path("/data/artifacts/by-project/malecns-gpu")
DEFAULT_ROOT = DATA / "experiments/visual-motor-large-queue-20261009"
REFERENCE = DATA / "engineering/feedback-diagnosis-20261007-v1/canonical-fullfield-optomotor-v1/revision-3/protocol.json"
ADAPTER = Path("/home/xvhang/Projects/scratch/malecns-visual-illusions/pixel_illusion_adapter.py")
BOOTSTRAP = SCRIPTS / "bootstrap_retina_extension_queue.py"
BODY_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
CHECK_SECONDS = 2400
MAX_SECONDS = 48 * 3600
CASE_SECONDS = 1200
WINDOWS = {"pre": [0.0, 1.0], "early": [1.0, 2.0],
           "late": [7.0, 13.0], "stimulus": [1.0, 13.0]}


def checkpoint(root, plan, state):
    from research_queue_summary import write_summary
    with (root / "summary.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return write_summary(root, plan, state)


def sample_resources(root):
    usage = {"updated_at": utc_now_iso(), "data_free_bytes": shutil.disk_usage(root).free}
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    for name in ("memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / group.lstrip("/") / name
        if path.is_file():
            usage[name] = int(path.read_text().strip())
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        usage["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        usage["gpu"] = str(error)
    write_json(root / "runtime.json", usage)


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def make_cases():
    cases = []

    def add(family, condition, seed, *, speed=0.0, contrast=1.0,
            wavelength=60.0, phase=0.0, visual=None, intervention="none",
            reference=None, reverse=False):
        ident = f"{family}_{condition}_seed{seed}"
        stimulus = visual or {"kind": "continuous_grating", "direction_deg_s": speed,
                              "temporal_frequency_hz": abs(speed) / wavelength}
        stimulus = {"mean_luminance": 0.5, "contrast": contrast,
                    "wavelength_deg": wavelength, "phase_at_onset_rad": phase, **stimulus}
        windows = copy.deepcopy(WINDOWS)
        if reverse:
            windows.update(pre_switch=[1.0, 7.0], post_switch=[7.0, 13.0])
        cases.append({"id": ident, "family": family, "condition": condition, "seed": seed,
                      "duration_s": 13.0, "onset_s": 1.0, "switch_time_s": 7.0 if reverse else None,
                      "contrast": contrast, "wavelength_deg": wavelength, "phase_at_onset_rad": phase,
                      "visual": stimulus, "intervention": intervention,
                      "reference_id": reference or f"speed_static_seed{seed}",
                      "analysis_windows": windows})

    for seed in range(1729, 1739):
        for condition, speed in [("static", 0), ("positive60", 60), ("negative60", -60),
                                 ("gray", 0), ("positive30", 30), ("negative30", -30),
                                 ("positive120", 120), ("negative120", -120)]:
            add("speed", condition, seed, speed=speed, contrast=0.0 if condition == "gray" else 1.0,
                visual={"kind": "uniform_gray"} if condition == "gray" else None)
    for seed in range(1729, 1739):
        for name, speed in [("positive_to_negative", 60), ("negative_to_positive", -60)]:
            add("reversal", name, seed, speed=speed, reverse=True)
    for seed in range(1729, 1734):
        for label, contrast, wavelength in [("c025", .25, 60), ("c050", .5, 60),
                                           ("c075", .75, 60), ("period30", 1, 30), ("period120", 1, 120)]:
            for direction, speed in [("static", 0), ("positive60", 60), ("negative60", -60)]:
                add("parameter", f"{label}_{direction}", seed, speed=speed,
                    contrast=contrast, wavelength=wavelength,
                    reference=f"parameter_{label}_static_seed{seed}")
    illusions = {
        "stepped40": {"kind": "stepped_grating", "step_deg": 5., "step_interval_s": .04},
        "reversephi40": {"kind": "reversephi_grating", "step_deg": 5., "step_interval_s": .04},
        "flicker40": {"kind": "flicker_grating", "step_deg": 0., "step_interval_s": .04},
        "counterphase": {"kind": "counterphase_grating", "temporal_frequency_hz": 1., "component_amplitude": .25},
        "stepped200": {"kind": "stepped_grating", "step_deg": 15., "step_interval_s": .2},
        "interleave_short": {"kind": "interleaved_stepped_grating", "step_deg": 15., "step_interval_s": .2, "gray_gap_s": .015},
        "interleave_long": {"kind": "interleaved_stepped_grating", "step_deg": 15., "step_interval_s": .2, "gray_gap_s": .135},
    }
    for seed in range(1729, 1739):
        for name, visual in illusions.items():
            add("illusion", name, seed, visual=visual)
    for seed in range(1729, 1734):
        for intervention in ("frozen_pose", "zero_steering"):
            for direction, speed in [("static", 0), ("positive60", 60), ("negative60", -60)]:
                add("ablation", f"{intervention}_{direction}", seed, speed=speed,
                    intervention=intervention, reference=f"ablation_{intervention}_static_seed{seed}")
    for seed in range(1729, 1732):
        for label, phase in [("quarter", .5), ("half", 1.), ("three_quarter", 1.5)]:
            for direction, speed in [("static", 0), ("positive60", 60), ("negative60", -60)]:
                add("phase", f"{label}_{direction}", seed, speed=speed,
                    phase=phase * 3.141592653589793, reference=f"phase_{label}_static_seed{seed}")
    # The first twelve are four matched triplets; subsequent blocks retain all
    # observations with a fixed randomized condition order.
    first = [x for seed in range(1729, 1733) for name in ("static", "positive60", "negative60")
             for x in cases if x["id"] == f"speed_{name}_seed{seed}"]
    rng = random.Random(90210)
    remaining = [x for x in cases if x not in first]
    ordered = list(first)
    for family in ("speed", "reversal", "parameter", "illusion", "ablation", "phase"):
        for seed in sorted({x["seed"] for x in remaining if x["family"] == family}):
            block = [x for x in remaining if x["family"] == family and x["seed"] == seed]
            rng.shuffle(block)
            ordered.extend(block)
    return ordered


def prepare(root):
    if root.exists():
        raise FileExistsError(f"prepared queue already exists: {root}")
    template = read_json(REFERENCE)
    old_files = read_json(Path(template["source_freeze"]["path"]))["files"]
    # Retain original identities of model/config assets; do not copy datasets.
    files = {p: r for p, r in old_files.items() if Path(p).suffix not in (".py", ".md")}
    for raw in files:
        Path(raw).resolve(strict=True)
    for p in list(SCRIPTS.glob("*.py")) + [ADAPTER]:
        files[str(p.resolve())] = record(p)
    root.mkdir(parents=True)
    inputs = root / "inputs"
    inputs.mkdir()
    from standard_optomotor_stimulus import AngularOptomotorStimulus
    sys.path.insert(0, str(ADAPTER.parent))
    import pixel_illusion_adapter as pixel
    cases, runtime_cases = make_cases(), []
    prepared = {}
    worker_config = Path(template["runtime"]["worker_config"]).resolve(strict=True)
    prepared[str(worker_config)] = record(worker_config)
    for case in cases:
        visual = case["visual"]
        schedule = {"schema_version": pixel.SCHEMA, "adapter_id": pixel.ADAPTER_ID,
                    "case_id": case["id"], "motor_authority": pixel.MOTOR_AUTHORITY,
                    "stimulus_onset_s": 1.0,
                    "clock": {"frame_dt_s": .005, "duration_s": 13.0, "frame_count": 2600,
                              "interval_convention": "half-open [start_s,end_s)"},
                    "segments": [{"segment_id": "gray", "start_s": 0., "end_s": 1.,
                                  "visual": {"kind": "uniform_gray", "mean_luminance": .5}},
                                 {"segment_id": "stimulus", "start_s": 1., "end_s": 13., "visual": visual}]}
        if case["switch_time_s"]:
            schedule["segments"][1]["end_s"] = 7.0
            after = dict(visual, direction_deg_s=-visual["direction_deg_s"],
                         phase_at_onset_rad=visual["phase_at_onset_rad"] +
                         2 * 3.141592653589793 * visual["direction_deg_s"] * 6 / case["wavelength_deg"])
            schedule["segments"].append({"segment_id": "reversed", "start_s": 7., "end_s": 13., "visual": after})
        pixel.validate_schedule(schedule, case["id"])
        nominal = pixel.phase_record(schedule, 1.0025)
        speed = float(nominal.get("direction_deg_s_fly_left_positive", 0.0))
        frequency = float(nominal.get("temporal_frequency_hz", 0.0))
        stimulus = AngularOptomotorStimulus(name=case["id"], contrast=case["contrast"],
            head_direction_sign=1 if speed >= 0 else -1, wavelength_deg=case["wavelength_deg"],
            temporal_frequency_hz=frequency, onset_s=1., duration_s=12., phase_at_onset_rad=case["phase_at_onset_rad"])
        spec = {"schema_version": "angular-optomotor-stimulus-spec-1",
                "display_geometry": "fixed_arena_infinite_distance_angular_field",
                "stimulus": {key: getattr(stimulus, key) for key in stimulus.__dataclass_fields__}}
        schedule_path, spec_path = inputs / f"{case['id']}.schedule.json", inputs / f"{case['id']}.spec.json"
        write_json(schedule_path, schedule)
        write_json(spec_path, spec)
        for path in (schedule_path, spec_path):
            prepared[str(path)] = record(path)
        runtime = copy.deepcopy(template["stimuli"]["conditions"][0])
        runtime.update(case_id=case["id"], cohort_id=f"research_seed{case['seed']}",
                       body_seed=case["seed"], neural_seed=case["seed"], seed=case["seed"],
                       direction=case["condition"], contrast=case["contrast"],
                       wavelength_deg=case["wavelength_deg"], phase_at_onset_rad=case["phase_at_onset_rad"],
                       head_direction_sign=stimulus.head_direction_sign,
                       head_angular_speed_deg_s=speed, direction_deg_s=speed,
                       arena_angular_speed_deg_s=-speed, speed_magnitude_deg_s=abs(speed),
                       temporal_frequency_hz=frequency, switch_time_s=case["switch_time_s"],
                       spec_path=str(spec_path), spec_sha256=prepared[str(spec_path)]["sha256"],
                       schedule_path=str(schedule_path), schedule_sha256=prepared[str(schedule_path)]["sha256"],
                       prefix_reference_case_id=case["reference_id"],
                       intervention=case["intervention"], interpretation="fixed-model engineering observation")
        runtime["effective_walk_config"]["seed"] = case["seed"]
        runtime_cases.append(runtime)
    protocol = {key: copy.deepcopy(template[key]) for key in
                ("body", "decoder", "runtime", "prepared_inputs", "evidence_boundary")}
    protocol.update(schema_version="visual-motor-research-queue-1", classification="fixed-model engineering observations",
                    stimuli={"conditions": runtime_cases}, prepared_files=prepared,
                    analysis_windows_s_half_open=WINDOWS)
    protocol["runtime"].update(capture_mode="light", worker_max_frames=2600,
        one_worker_one_body_session_per_case=False, resident_worker=True,
        selected_state_capture="compact population readout; no full neural state arrays",
        stimulus_source="shared pixel schedule on canonical live world-ray geometry",
        reviewed_binding="resident worker with fresh body and decoder for every case",
        retina_adapter_path=str(ADAPTER), worker_bootstrap_path=str(BOOTSTRAP))
    protocol["prepared_inputs"].update(retina_adapter=str(ADAPTER), worker_bootstrap=str(BOOTSTRAP),
                                     case_angular_specs_dir=str(inputs), case_retina_schedules_dir=str(inputs))
    protocol["body"]["seed_triplets"] = list(range(1729, 1739))
    write_json(root / "source-identity.json", {"files": files, "reference_protocol": record(REFERENCE)})
    protocol["source_freeze"] = record(root / "source-identity.json")
    write_json(root / "protocol.json", protocol)
    plan = {"schema_version": "research-queue-plan-1", "status": "prepared_not_started",
            "created_at": utc_now_iso(), "cases": cases, "check_interval_s": CHECK_SECONDS,
            "max_runtime_s": MAX_SECONDS, "case_timeout_s": CASE_SECONDS,
            "memory_max": "5G", "cpu_quota": "1200%", "capture_stride": 50,
            "fail_policy": "stop on first execution failure; retain logs; resume only explicitly",
            "scientific_scope": "seed repeats are model replicates, not independent biological animals",
            "protocol_sha256": sha256_file(root / "protocol.json")}
    write_json(root / "plan.json", plan)
    state = {"status": "prepared", "started_at": None,
             "cases": {c["id"]: {"status": "pending", "run_dir": str(root / "runs" / c["id"])} for c in cases}}
    write_json(root / "state.json", state)
    checkpoint(root, plan, state)
    write_readme(root, len(cases))
    return plan


def write_readme(root, count):
    (root / "README.md").write_text(
        "# Visual-motor research queue\n\nExecution status: `state.json`; preparation does not start computation.\n\n"
        f"{count} cases, 13 seconds each. One resident neural worker, fresh body/decoder per case.\n"
        "Core 80; reversal 20; parameter 75; illusion 70; intervention 30; phase 27.\n\n"
        "Model weights, decoder and gait settings are fixed. Frozen-pose and zero-steering arms are explicit interventions.\n"
        "All cases share a 1-second gray prelude; reversal occurs at 7 seconds.\n"
        "Low-contrast and period scans include their own matched static controls.\n\n"
        "Progress is summarized every 40 minutes, after the first 12 cases and on exit.\n"
        "Execution errors stop the queue without automatic retries. Completed cases are skipped on explicit resume.\n"
        "Create STOP in this directory to stop at the next case boundary.\n\n"
        f"Launch: `/usr/bin/python3.12 {Path(__file__).resolve()} launch --root {root}`\n"
        f"Status: `/usr/bin/python3.12 {Path(__file__).resolve()} status --root {root}`\n"
        "stdout.log/stderr.log retain supervisor and worker errors. Bulk source inputs are referenced in place.\n",
        encoding="utf-8")


def bind_runtime(root, protocol):
    import run_flygym_free_body_online_lowcontrast_v2 as base
    import run_flygym_free_body_online_highcontrast_dynamics_v1 as high
    sys.path.insert(0, str(ADAPTER.parent))
    import pixel_illusion_adapter as pixel
    original_ready, writer = base._validate_worker_ready, base._write_jsonl
    high.BASE = high.PREPARATION = root
    high.RUNS = root / "runs"
    high.FRAME_COUNT, high.DURATION_S = 2600, 13.0
    high.CONTRAST = 1.0
    high.WINDOWS = {key: tuple(value) for key, value in WINDOWS.items()}
    base = high._configure_base(protocol, tuple(protocol["stimuli"]["conditions"]))
    base._write_jsonl = writer
    base.STIMULUS_DURATION_S = 12.0
    # Validate identities once before the queue; ready checks retain actual
    # loaded model, config, renderer and backend identities for each reset.
    base.verify_frozen_inputs = lambda _: None

    def command(case, output, active):
        argv = high._worker_command(case, output, active)
        argv[1] = str(BOOTSTRAP)
        return argv

    def ready(value, case, active):
        original_ready(value, case, active)
        angular = value["angular_stimulus"]["stimulus_description"]
        expected = {"runtime_schedule_adapter_id": pixel.ADAPTER_ID,
                    "runtime_schedule_adapter_path": str(ADAPTER),
                    "runtime_schedule_adapter_sha256": sha256_file(ADAPTER),
                    "runtime_schedule_path": case["schedule_path"],
                    "runtime_schedule_sha256": case["schedule_sha256"]}
        if any(angular.get(key) != val for key, val in expected.items()):
            raise ValueError("loaded pixel schedule identity differs from prepared case")
        if angular["runtime_schedule"]["case_id"] != case["case_id"]:
            raise ValueError("resident renderer retained a different case")

    metadata = base._controller_metadata

    def controller_metadata(controller, case, active):
        value = metadata(controller, case, active)
        value.update(control_intervention=case["intervention"],
                     closed_loop_visual_pose=case["intervention"] != "frozen_pose",
                     case_label_selects_pixels_only=case["intervention"] == "none")
        value.pop("highcontrast_schedule_path", None)
        value.pop("highcontrast_schedule_sha256", None)
        return value

    base._worker_command, base._validate_worker_ready = command, ready
    base._controller_metadata = controller_metadata
    return base, high


def worker(root):
    plan, protocol = read_json(root / "plan.json"), read_json(root / "protocol.json")
    state = read_json(root / "state.json")
    base, high = bind_runtime(root, protocol)
    lock_path = base.GLOBAL_NN_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_cases = {c["case_id"]: c for c in protocol["stimuli"]["conditions"]}
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = base.ResidentNeuralWorker(runtime_max_s=MAX_SECONDS)
        try:
            for case in plan["cases"]:
                item = state["cases"][case["id"]]
                if item["status"] == "complete":
                    continue
                if (root / "STOP").exists():
                    state["status"] = "stopped"
                    break
                if item["status"] != "pending":
                    raise RuntimeError(f"unresolved case {case['id']}; use explicit resume")
                current = runtime_cases[case["id"]]
                base.STUDY_CONTRAST = current["contrast"]
                high.SEED = current["body_seed"]
                item.update(status="running", started_epoch_s=time.time(), started_at=utc_now_iso())
                state.update(status="running", current_case=case["id"])
                write_json(root / "state.json", state)
                import run_flygym_online_optomotor_eye_parity_v4_1_attempt5 as bridge
                original_pose = bridge._build_virtual_pose
                initial_pose = None

                def frozen_pose(*args, **kwargs):
                    nonlocal initial_pose
                    value = original_pose(*args, **kwargs)
                    if initial_pose is None:
                        initial_pose = copy.deepcopy(value["pose"])
                    value["pose"] = copy.deepcopy(initial_pose)
                    return value

                if case["intervention"] == "frozen_pose":
                    bridge._build_virtual_pose = frozen_pose
                try:
                    result = base._run_case(current, protocol, capture_mode="light", capture_stride=50,
                        resident_worker=pool, control_intervention="zero_steering" if case["intervention"] == "zero_steering" else "none")
                    item.update(status="complete", finished_at=utc_now_iso(),
                                wall_s=result["timing_s"]["case_wall_s"], worker_pid=result["worker_pid"])
                except BaseException:
                    item.update(status="failed", error=traceback.format_exc(), finished_at=utc_now_iso())
                    state["status"] = "failed"
                    write_json(root / "state.json", state)
                    raise
                finally:
                    bridge._build_virtual_pose = original_pose
                state["current_case"] = None
                write_json(root / "state.json", state)
                print(f"complete {case['id']} {item['wall_s']:.1f}s", flush=True)
                if sum(x["status"] == "complete" for x in state["cases"].values()) == 12:
                    checkpoint(root, plan, state)
            else:
                state["status"] = "complete"
            write_json(root / "state.json", state)
        finally:
            pool.close()


def supervise(root):
    plan = read_json(root / "plan.json")
    sample_resources(root)
    process = subprocess.Popen([str(BODY_PYTHON), str(Path(__file__).resolve()), "--worker", "--root", str(root)],
                               start_new_session=True)
    next_check = time.monotonic() + CHECK_SECONDS
    try:
        while process.poll() is None:
            state = read_json(root / "state.json")
            current = state["cases"].get(state.get("current_case"), {})
            if current.get("status") == "running" and time.time() - current["started_epoch_s"] > CASE_SECONDS:
                raise TimeoutError(f"case exceeded {CASE_SECONDS}s: {state['current_case']}")
            if time.monotonic() >= next_check:
                sample_resources(root)
                checkpoint(root, plan, state)
                if shutil.disk_usage(root).free < 100 * 1024**3:
                    raise RuntimeError("data disk free space fell below 100 GiB")
                print(f"40-minute check: {sum(x['status'] == 'complete' for x in state['cases'].values())}/{len(plan['cases'])}", flush=True)
                next_check = time.monotonic() + CHECK_SECONDS
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise RuntimeError(f"queue worker exited with {process.returncode}")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state = read_json(root / "state.json")
        state.update(status="failed", supervisor_error=traceback.format_exc())
        for item in state["cases"].values():
            if item["status"] == "running":
                item.update(status="interrupted", finished_at=utc_now_iso())
        write_json(root / "state.json", state)
        raise
    finally:
        checkpoint(root, plan, read_json(root / "state.json"))


def launch(root, resume):
    plan, state = read_json(root / "plan.json"), read_json(root / "state.json")
    if resume and state["status"] in ("running", "starting"):
        unit = state.get("unit")
        if unit and subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0:
            raise RuntimeError("existing queue service is still active")
        for item in state["cases"].values():
            if item["status"] == "running":
                run = Path(item["run_dir"]) / "run-record.json"
                completed = read_json(run) if run.is_file() else {}
                item["status"] = "complete" if completed.get("status") == "complete" else "interrupted"
        state["status"] = "stopped"
    if state["status"] in ("running", "starting", "complete"):
        raise RuntimeError(f"queue is {state['status']}; inspect status before launching")
    if (root / "STOP").exists():
        raise RuntimeError("STOP exists; remove it explicitly before launch")
    if shutil.disk_usage(root).free < 100 * 1024**3:
        raise RuntimeError("data disk requires 100 GiB free before this queue")
    protocol = read_json(root / "protocol.json")
    if sha256_file(root / "protocol.json") != plan["protocol_sha256"]:
        raise RuntimeError("queue protocol changed after preparation")
    identities = read_json(root / "source-identity.json")["files"]
    for raw, entry in {**identities, **protocol["prepared_files"]}.items():
        if sha256_file(raw) != entry["sha256"]:
            raise RuntimeError(f"prepared source/input changed: {raw}")
    if resume:
        for ident, item in state["cases"].items():
            if item["status"] in ("failed", "interrupted"):
                old = Path(item["run_dir"])
                if old.exists():
                    failed = root / "failed"
                    failed.mkdir(exist_ok=True)
                    old.rename(failed / f"{ident}-{time.time_ns()}")
                item.update(status="pending")
                item.pop("error", None)
    elif any(item["status"] != "pending" for item in state["cases"].values()):
        raise RuntimeError("nonempty queue requires explicit --resume")
    unit = f"malecns-research-{time.time_ns()}"
    state.update(status="starting", unit=unit, started_at=state.get("started_at") or utc_now_iso())
    write_json(root / "state.json", state)
    argv = ["systemd-run", "--user", "--quiet", f"--unit={unit}",
            "--property=MemoryMax=5G", "--property=MemorySwapMax=0", "--property=CPUQuota=1200%",
            f"--property=RuntimeMaxSec={MAX_SECONDS}", "--property=TimeoutStopSec=30", "--property=KillMode=control-group",
            f"--property=WorkingDirectory={PROJECT}", f"--property=StandardOutput=append:{root / 'stdout.log'}",
            f"--property=StandardError=append:{root / 'stderr.log'}",
            "--setenv=ONLINE_LIGHT_CHILD=1", f"--setenv=ONLINE_FREEBODY_CASE_UNIT={unit}",
            "--setenv=CHECK_MIN_AVAILABLE_GIB=0", "--setenv=PYTHONDONTWRITEBYTECODE=1",
            "--setenv=PYTHONUNBUFFERED=1", "--setenv=MPLBACKEND=Agg",
            "/usr/bin/systemd-inhibit", "--what=sleep:idle", "--mode=block",
            "--who=MaleCNS research queue", "--why=Continuous simulation queue",
            str(BODY_PYTHON), str(Path(__file__).resolve()), "--supervise", "--root", str(root)]
    try:
        subprocess.run(argv, check=True)
    except BaseException:
        state.update(status="launch_failed", supervisor_error=traceback.format_exc())
        write_json(root / "state.json", state)
        raise
    print(f"started {unit}; progress: {root / 'progress.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("prepare", "launch", "status"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker or args.supervise:
        if not os.environ.get("ONLINE_FREEBODY_CASE_UNIT"):
            parser.error("internal execution requires the capped queue service")
        (worker if args.worker else supervise)(root)
    elif args.action == "prepare":
        print(f"prepared {len(prepare(root)['cases'])} cases at {root}; not started")
    elif args.action == "launch":
        launch(root, args.resume)
    elif args.action == "status":
        checkpoint(root, read_json(root / "plan.json"), read_json(root / "state.json"))
        print((root / "progress.md").read_text())
    else:
        parser.error("choose prepare, launch, or status")


if __name__ == "__main__":
    main()
