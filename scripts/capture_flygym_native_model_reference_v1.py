#!/usr/bin/env python3
"""Capture the constructor's native FlyGym/MuJoCo model without stepping.

This is a one-shot forensic capture.  ``--preflight`` does not import FlyGym or
construct a model.  ``--capture`` must run in the named systemd cgroup and can
construct at most one FlyGymReferenceSession.  It never invokes a control tick,
settle, warmup, or physics step.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback
import traceback

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path("/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1")
OUT = BASE / "mechanical-native-model-reference-v1"
REV4 = BASE / "mechanical-steering-dose-calibration-v1/revision-4"
REV2 = BASE / "mechanical-steering-dose-calibration-v1/revision-2"
V41_SOURCE_FREEZE = Path("/data/artifacts/by-project/malecns-gpu/engineering/flygym-phase-speed-repair-20261006-v1/development/v4-1/source_freeze.json")
ORIGINAL_AUDIT = BASE / "mechanical-model-initialization-audit-v1"
PAIR_CASE = Path("/data/artifacts/by-project/malecns-gpu/engineering/standardized-repair-20261007-v1/mechanics/mechanics_cases/static_d0")
EXPECTED_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
EXPECTED_UNIT = "flygym-native-model-reference-20261008-v1.service"
EXPECTED_BODY_CONFIG_SHA256 = "2972591d4d0aa2e4f647d26ba23504d3bad137b6367f1836683003e0247f24fa"
EXPECTED_R4_NATIVE_MJB_SHA256 = "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"
EXPECTED_DIRECT_XML_MJB_SHA256 = "c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f"
EXPECTED_PAIRED_SCREEN_XML_SHA256 = "bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57"
EXPECTED_INITIAL_QPOS_SHA256 = "d292b37df0f9a83265423da2ce53b0b140d8ef6a49d6909e464b805453e75b99"
EXPECTED_INITIAL_QVEL_SHA256 = "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
EXPECTED_V41_SOURCE_FREEZE_SHA256 = "f7dc3d758857ddeda4923194027311bd90f53aeff9d57515f6e4987f786f8bd0"
EXPECTED_REV2_DOSE_SOURCE_DRIFT = {
    str(PROJECT / "scripts/prepare_flygym_steering_dose_calibration_v1.py"): {
        "frozen_sha256": "6796c08c41bdaafd8878d69c758cde3c2ca6a9e3942fe40f4be53d7129d64735",
        "current_revision_4_sha256": "2ff7c6183fa10496d23640aa6ca223272c1d155373640258470099ed1661734a",
    },
    str(PROJECT / "scripts/run_flygym_steering_dose_calibration_v1.py"): {
        "frozen_sha256": "210b4c28317705060c2a89f7c8a62a19fa44ebdb134b7c9035d1ea73925d6113",
        "current_revision_4_sha256": "bacc001e7afb302763d30d9ab84715bb6973baa387b2c1e46c452c9c4e86de56",
    },
}
PREFLIGHT_ATTEMPT_HISTORY = [
    {
        "unit": "flygym-native-model-reference-preflight-20261008-v1.service",
        "invocation_id": "ca7829fcb62e449eacdcd08f30a42503",
        "script_sha256": "cacae216f71f8df7a89b4a7d899f4a50b2ca4037a64208296440379166b1d6e2",
        "exit_code": 1,
        "elapsed_wall_s": 0.113,
        "failure": "initial static pin check looked for flygym_legacy_rest_compat.py in the mechanical revision freeze instead of its authoritative v4.1 source freeze",
        "constructor_calls": 0,
        "output_directory_created": False,
    },
    {
        "unit": "flygym-native-model-reference-preflight-retry1-20261008-v1.service",
        "invocation_id": "dee7cad95f1e404eb0eed5e74ddcb8a1",
        "script_sha256": "502a43e6c0ea7c159365c3c4937cc4aa731a8a59c613d5e3f9a6e49621e50f67",
        "exit_code": 1,
        "systemd_runtime_display": "166ms",
        "failure": "revision-2 freeze contains two later-revised dose orchestration sources; no model constructor was called",
        "constructor_calls": 0,
        "output_directory_created": False,
        "systemd_display_memory_summary": "not used as an authoritative cgroup measurement",
    },
    {
        "unit": "flygym-native-model-reference-preflight-retry2-20261008-v1.service",
        "invocation_id": "ef0df8a3d1c34429907ef1d1b3473a6f",
        "script_sha256": "bb304901882b408b9f23de41375712d21272d36e04dbf9ded559e8f32cbf3c6b",
        "exit_code": 1,
        "failure": "cgroup preflight attempted to read cpu.max at the hierarchy mount root where that controller file is absent; process cgroup caps and memory were not evaluated to constructor and no model was created",
        "constructor_calls": 0,
        "output_directory_created": False,
        "systemd_display_memory_summary": "not used as an authoritative cgroup measurement",
    },
]
LIMIT_CPU_CORES = 2.0
LIMIT_MEMORY_BYTES = 2 * 1024**3
LIMIT_RUNTIME_SECONDS = 120
RUN_ID = "steering-dose-seed1729_dzero_0p00-T7s"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_config_sha(value: object) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_config() -> tuple[dict, dict]:
    config_path = REV2 / "frozen_inputs/v4_1_effective_config.json"
    source = read_json(config_path)
    walk = dict(source["walk_config"])
    walk.update({"duration_s": 7.0, "seed": 1729})
    for key in ("spawn_position_mm", "spawn_quaternion_wxyz", "position_actuator_forcerange_native", "camera_resolution_hw"):
        if key in walk:
            walk[key] = tuple(walk[key])
    return walk, source


def source_freezes() -> list[tuple[str, Path, dict]]:
    return [
        ("revision_2_original_capture", REV2 / "source_freeze.json", read_json(REV2 / "source_freeze.json")),
        ("revision_4_runtime", REV4 / "source_freeze.json", read_json(REV4 / "source_freeze.json")),
    ]


def verify_freeze(freeze: dict, label: str) -> dict:
    checked_sources = 0
    checked_assets = 0
    mismatches: list[dict] = []
    expected_revision_drift: list[dict] = []
    for raw, record in freeze.get("source_files", {}).items():
        path = Path(raw)
        checked_sources += 1
        actual = sha_file(path) if path.is_file() else None
        if actual != record.get("sha256"):
            drift = EXPECTED_REV2_DOSE_SOURCE_DRIFT.get(raw) if label.startswith("revision_2") else None
            if drift and record.get("sha256") == drift["frozen_sha256"] and actual == drift["current_revision_4_sha256"]:
                expected_revision_drift.append({
                    "path": raw,
                    "frozen_sha256": record.get("sha256"),
                    "current_revision_4_sha256": actual,
                    "scope_note": "dose orchestration source is not imported by this constructor-only capture; current revision-4 source pin is checked separately",
                })
            else:
                mismatches.append({"kind": "source", "path": raw, "expected": record.get("sha256"), "actual": actual})
    files = freeze.get("asset_manifest", {}).get("files", {})
    for logical, record in files.items():
        path = Path(record["physical_path"])
        checked_assets += 1
        actual = sha_file(path) if path.is_file() else None
        if actual != record.get("sha256"):
            mismatches.append({"kind": "asset", "logical": logical, "path": str(path), "expected": record.get("sha256"), "actual": actual})
    return {
        "label": label,
        "freeze_sha256": sha_file((REV2 if label.startswith("revision_2") else REV4) / "source_freeze.json"),
        "source_count": checked_sources,
        "asset_count": checked_assets,
        "mismatches": mismatches,
        "expected_later_source_drift": expected_revision_drift,
        "all_applicable_pins_match": not mismatches,
        "all_match": not mismatches,
    }


def method_has_forbidden_integration(source_path: Path) -> dict:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FlyGymReferenceSession":
            target = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
            break
    if target is None:
        raise ValueError("AST did not find FlyGymReferenceSession.__init__")
    forbidden: list[str] = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if called in {"step", "step_with_profile", "mj_step", "settle", "warmup", "run"}:
            forbidden.append(called)
    return {"target": "FlyGymReferenceSession.__init__", "forbidden_calls": forbidden, "integration_call_absent": not forbidden}


def preflight_record() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to reuse output directory: {OUT}")
    walk, config_source = load_config()
    r2_freeze = read_json(REV2 / "source_freeze.json")
    r4_freeze = read_json(REV4 / "source_freeze.json")
    expected_sources = r4_freeze.get("source_files", {})
    walk_path = str(PROJECT / "scripts/run_flygym_reference_walk.py")
    legacy_path = str(PROJECT / "scripts/flygym_legacy_rest_compat.py")
    current_source_pins = {
        walk_path: sha_file(Path(walk_path)),
        legacy_path: sha_file(Path(legacy_path)),
    }
    if walk_path not in expected_sources or current_source_pins[walk_path] != expected_sources[walk_path]["sha256"]:
        raise ValueError("session builder does not match revision-4 source freeze")
    v41_freeze_hash = sha_file(V41_SOURCE_FREEZE)
    v41_legacy_pin = read_json(V41_SOURCE_FREEZE).get("sources", {}).get("legacy_rest_factory", {})
    if v41_freeze_hash != EXPECTED_V41_SOURCE_FREEZE_SHA256:
        raise ValueError("v4.1 source-freeze pin changed")
    if v41_legacy_pin.get("path") != legacy_path or current_source_pins[legacy_path] != v41_legacy_pin.get("sha256"):
        raise ValueError("legacy-rest adapter does not match its authoritative v4.1 source freeze")
    if canonical_config_sha(walk) != EXPECTED_BODY_CONFIG_SHA256:
        raise ValueError("WalkConfig does not match the frozen duration-7 seed-1729 body config")
    original_preflight = read_json(ORIGINAL_AUDIT / "preflight.json")
    if original_preflight.get("walk_config_sha256") != EXPECTED_BODY_CONFIG_SHA256:
        raise ValueError("original initialization audit config hash changed")
    if original_preflight.get("run_flygym_reference_walk_sha256") != current_source_pins[walk_path]:
        raise ValueError("session builder differs from original initialization capture")
    if original_preflight.get("legacy_rest_adapter_sha256") != current_source_pins[legacy_path]:
        raise ValueError("legacy-rest adapter differs from original initialization capture")

    run_record_path = REV4 / "runs/seed1729_dzero_0p00/run-record.json"
    r4_run = read_json(run_record_path)
    if canonical_config_sha(r4_run["body_config"]) != EXPECTED_BODY_CONFIG_SHA256:
        raise ValueError("revision-4 runtime body config hash differs")
    normalized_walk = json.loads(json.dumps(walk, allow_nan=False))
    if normalized_walk != r4_run["body_config"]:
        raise ValueError("revision-4 runtime WalkConfig fields differ")

    revision2_hash = sha_file(REV2 / "source_freeze.json")
    revision4_hash = sha_file(REV4 / "source_freeze.json")
    if revision2_hash != "33f92817712cfe87bf89067706a3f0297fb5e891ba569352d812d3b9e56275be":
        raise ValueError("revision-2 original source-freeze pin changed")
    if revision4_hash != "71bf1d89bb8a3468386b097a0b24de8540c8510bab79efbeb738c0de52da8e83":
        raise ValueError("revision-4 source-freeze pin changed")
    freeze_checks = [verify_freeze(item, label) for label, _path, item in source_freezes()]
    if not all(item["all_match"] for item in freeze_checks):
        raise ValueError("source/asset freeze mismatch: " + json.dumps(freeze_checks, sort_keys=True))

    prior_order_path = REV4 / "frozen_inputs/prior_array_order.json"
    prior_options_path = REV4 / "frozen_inputs/prior_model_options.json"
    initial_state_path = PAIR_CASE / "initial_state.json"
    prior_order = read_json(prior_order_path)
    prior_options = read_json(prior_options_path)
    initial = read_json(initial_state_path)
    qpos_hash = sha_bytes(struct.pack("<%sd" % len(initial["qpos"]), *initial["qpos"]))
    qvel_hash = sha_bytes(struct.pack("<%sd" % len(initial["qvel"]), *initial["qvel"]))
    if qpos_hash != EXPECTED_INITIAL_QPOS_SHA256 or qvel_hash != EXPECTED_INITIAL_QVEL_SHA256:
        raise ValueError("paired-screen initial state raw float64 hashes differ from the frozen summary")
    if initial.get("time_s") != 0.0 or initial.get("source") != "FlyGym keyframe reset followed by mj_forward; no qpos/qvel assignment by runner":
        raise ValueError("paired-screen initial-state provenance differs")

    r4_terminal = read_json(REV4 / "root-terminal-observation.json")
    actual_mjb = REV4 / "runs/seed1729_dzero_0p00/compiled_model.mjb"
    direct_xml_mjb = REV4 / "frozen_inputs/compiled_model_reference.mjb"
    paired_xml = REV4 / "frozen_inputs/paired_screen_compiled_model.xml"
    if sha_file(actual_mjb) != EXPECTED_R4_NATIVE_MJB_SHA256:
        raise ValueError("revision-4 actual runtime MJB pin changed")
    if sha_file(direct_xml_mjb) != EXPECTED_DIRECT_XML_MJB_SHA256:
        raise ValueError("direct-XML MJB reference pin changed")
    if sha_file(paired_xml) != EXPECTED_PAIRED_SCREEN_XML_SHA256:
        raise ValueError("paired-screen XML pin changed")
    if r4_terminal.get("status") != "failed_stopped" or r4_terminal.get("actual_control_ticks") != 0 or r4_terminal.get("actual_native_samples") != 0:
        raise ValueError("revision-4 failed zero-control terminal record changed")
    if r4_terminal.get("actual_runtime_mjb_sha256") != EXPECTED_R4_NATIVE_MJB_SHA256:
        raise ValueError("revision-4 terminal record no longer names the saved native MJB")
    if (PAIR_CASE / "compiled_model.mjb").exists():
        paired_has_mjb = True
    else:
        paired_has_mjb = False

    ast_check = method_has_forbidden_integration(Path(walk_path))
    if not ast_check["integration_call_absent"]:
        raise ValueError("session constructor AST contains an integration/settle call")
    source_summary = {
        "revision_2_source_freeze": {"path": str(REV2 / "source_freeze.json"), "sha256": revision2_hash},
        "revision_4_source_freeze": {"path": str(REV4 / "source_freeze.json"), "sha256": revision4_hash},
        "original_initialization_preflight": {"path": str(ORIGINAL_AUDIT / "preflight.json"), "sha256": sha_file(ORIGINAL_AUDIT / "preflight.json")},
        "session_builder": {"path": walk_path, "sha256": current_source_pins[walk_path]},
        "legacy_rest_adapter": {"path": legacy_path, "sha256": current_source_pins[legacy_path]},
        "v4_1_source_freeze": {"path": str(V41_SOURCE_FREEZE), "sha256": v41_freeze_hash},
        "r4_runtime_mjb": {"path": str(actual_mjb), "sha256": sha_file(actual_mjb)},
        "direct_xml_mjb": {"path": str(direct_xml_mjb), "sha256": sha_file(direct_xml_mjb)},
        "paired_screen_xml": {"path": str(paired_xml), "sha256": sha_file(paired_xml)},
        "paired_screen_initial_state": {"path": str(initial_state_path), "sha256": sha_file(initial_state_path), "qpos_sha256": qpos_hash, "qvel_sha256": qvel_hash, "has_saved_mjb": paired_has_mjb},
        "prior_array_order": {"path": str(prior_order_path), "sha256": sha_file(prior_order_path), "fields": len(prior_order)},
        "prior_model_options": {"path": str(prior_options_path), "sha256": sha_file(prior_options_path), "values": prior_options},
        "r4_run_record": {"path": str(run_record_path), "sha256": sha_file(run_record_path), "status": r4_run.get("status"), "control_ticks": r4_run.get("actual_control_tick_count", r4_run.get("actual_control_ticks", 0)), "runtime_mjb_sha256": r4_run.get("model_initialization_provenance", {}).get("compiled_model_mjb_sha256")},
    }
    preflight_cgroup = cgroup_attestation(require_unit=False)
    return {
        "schema_version": "flygym-native-model-reference-preflight-v1",
        "status": "preflight_complete_constructor_not_yet_called",
        "created_utc": utc_now(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha_file(Path(__file__)),
        "project": str(PROJECT),
        "walk_config": json.loads(json.dumps(walk, allow_nan=False)),
        "walk_config_source": str(REV2 / "frozen_inputs/v4_1_effective_config.json"),
        "walk_config_source_sha256": sha_file(REV2 / "frozen_inputs/v4_1_effective_config.json"),
        "walk_config_sha256": canonical_config_sha(walk),
        "seed": 1729,
        "duration_s": 7.0,
        "run_id": RUN_ID,
        "asset_and_source_checks": freeze_checks,
        "constructor_ast_check": ast_check,
        "provenance": source_summary,
        "constructor_policy": "one FlyGymReferenceSession; no control tick, Sim.step, mj_step, settle, warmup, NN, GPU, or tests",
        "prior_static_preflight_attempts": PREFLIGHT_ATTEMPT_HISTORY,
        "runtime_limits": {"cpu_cores_max": LIMIT_CPU_CORES, "memory_bytes_max": LIMIT_MEMORY_BYTES, "runtime_seconds_max": LIMIT_RUNTIME_SECONDS},
        "preflight_cgroup": preflight_cgroup,
        "native_baseline_policy": "only mj_saveModel(session.model, ...); XML remains constructor-emitted provenance and is not recompiled as baseline",
        "gate_policy": "capture and compare only; do not promote an identity gate or rewrite any failed revision-3/revision-4 records",
        "no_execution_approval_created": True,
        "revision_4_failed_record_preserved": True,
    }


def cgroup_attestation(*, require_unit: bool) -> dict:
    membership = next((line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")), None)
    if membership is None:
        raise RuntimeError("process is not in a unified cgroup-v2 hierarchy")
    current = Path("/sys/fs/cgroup") / membership.lstrip("/")
    cpu_raw = (current / "cpu.max").read_text().strip().split()
    mem_raw = (current / "memory.max").read_text().strip()
    if len(cpu_raw) != 2:
        raise RuntimeError(f"unexpected cpu.max: {cpu_raw}")
    own_cpu = None if cpu_raw[0] == "max" else float(cpu_raw[0]) / float(cpu_raw[1])
    own_mem = None if mem_raw == "max" else int(mem_raw)
    cpu_caps: list[float] = []
    mem_caps: list[int] = []
    hierarchy: list[dict] = []
    node = current
    root = Path("/sys/fs/cgroup")
    while True:
        cpu_path = node / "cpu.max"
        mem_path = node / "memory.max"
        cpu_text = cpu_path.read_text().strip().split() if cpu_path.exists() else None
        mem_text = mem_path.read_text().strip() if mem_path.exists() else None
        cpu_cap = None if not cpu_text or cpu_text[0] == "max" else float(cpu_text[0]) / float(cpu_text[1])
        mem_cap = None if mem_text is None or mem_text == "max" else int(mem_text)
        if cpu_cap is not None:
            cpu_caps.append(cpu_cap)
        if mem_cap is not None:
            mem_caps.append(mem_cap)
        hierarchy.append({
            "directory": str(node),
            "cpu.max": " ".join(cpu_text) if cpu_text else None,
            "memory.max": mem_text,
            "cpu_controller_file_present": cpu_text is not None,
            "memory_controller_file_present": mem_text is not None,
        })
        if node == root:
            break
        node = node.parent
    effective_cpu = min(cpu_caps) if cpu_caps else None
    effective_mem = min(mem_caps) if mem_caps else None
    unit_name = os.environ.get("FLYG_NATIVE_UNIT_NAME")
    if require_unit and (unit_name != EXPECTED_UNIT or not membership.endswith("/" + EXPECTED_UNIT)):
        raise RuntimeError(f"process is not in expected transient systemd unit cgroup: {membership}; expected {EXPECTED_UNIT}")
    if own_cpu is None or own_cpu > LIMIT_CPU_CORES or effective_cpu is None or effective_cpu > LIMIT_CPU_CORES:
        raise RuntimeError(f"cgroup CPU cap missing or exceeds {LIMIT_CPU_CORES}: {hierarchy}")
    if own_mem is None or own_mem > LIMIT_MEMORY_BYTES or effective_mem is None or effective_mem > LIMIT_MEMORY_BYTES:
        raise RuntimeError(f"cgroup memory cap missing or exceeds {LIMIT_MEMORY_BYTES}: {hierarchy}")
    if os.environ.get("FLYG_NATIVE_RUNTIME_MAX_S") != str(LIMIT_RUNTIME_SECONDS):
        raise RuntimeError("systemd RuntimeMaxSec attestation variable missing or wrong")
    peak_path = current / "memory.peak"
    return {
        "cgroup_version": 2,
        "membership_path": membership,
        "directory": str(current),
        "unit_name": unit_name,
        "cpu.max": " ".join(cpu_raw),
        "own_cpu_quota_cores": own_cpu,
        "effective_cpu_cap_cores": effective_cpu,
        "memory.max": mem_raw,
        "own_memory_cap_bytes": own_mem,
        "effective_memory_cap_bytes": effective_mem,
        "memory_peak_bytes": int(peak_path.read_text().strip()) if peak_path.exists() else None,
        "runtime_limit_s": LIMIT_RUNTIME_SECONDS,
        "hierarchy": hierarchy,
    }


def upstream_source_preflight() -> dict:
    reference_root = Path("/home/xvhang/Projects/external-references/flygym-v2.1.0")
    expected_commit = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
    relpaths = [
        "src/flygym/__init__.py", "src/flygym/anatomy.py", "src/flygym/compose/physics.py",
        "src/flygym/compose/fly/base_fly.py", "src/flygym/compose/fly/neuromechfly.py",
        "src/flygym/compose/world/base_world.py", "src/flygym/compose/world/flat_ground.py",
        "src/flygym/simulation.py", "src/flygym/rendering.py", "src/flygym/utils/mjcf.py",
        "src/flygym_demo/complex_terrain/common.py", "src/flygym_demo/complex_terrain/preprogrammed.py",
        "src/flygym_demo/complex_terrain/cpg_controller.py", "src/flygym_demo/complex_terrain/hybrid_controller.py",
        "src/flygym_demo/complex_terrain/turning_controller.py",
    ]
    commit = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    installed_root = Path(metadata.distribution("flygym").locate_file(""))
    records = {}
    mismatches = []
    for rel in relpaths:
        package_rel = rel.removeprefix("src/")
        installed = installed_root / package_rel
        checkout = reference_root / rel
        installed_hash = sha_file(installed) if installed.is_file() else None
        checkout_hash = sha_file(checkout) if checkout.is_file() else None
        equal = installed_hash is not None and installed_hash == checkout_hash
        records[rel] = {
            "installed_path": str(installed), "installed_sha256": installed_hash,
            "checkout_path": str(checkout), "checkout_sha256": checkout_hash,
            "equal": equal,
        }
        if not equal:
            mismatches.append(rel)
    result = {
        "reference_repository": str(reference_root),
        "expected_commit": expected_commit,
        "actual_commit": commit,
        "commit_match": commit == expected_commit,
        "installed_package_version": metadata.version("flygym"),
        "installed_vs_pinned_checkout_files": records,
        "mismatched_files": mismatches,
    }
    result["all_match"] = result["commit_match"] and not mismatches
    if not result["all_match"]:
        raise RuntimeError("upstream FlyGym source preflight failed: " + json.dumps(result, sort_keys=True))
    return result


def capture() -> int:
    pre_path = OUT / "preflight.json"
    if not pre_path.is_file():
        raise FileNotFoundError("preflight.json missing; run --preflight before the one-shot capture")
    preflight = read_json(pre_path)
    if preflight.get("status") != "preflight_complete_constructor_not_yet_called":
        raise RuntimeError("preflight record is not in the expected state")
    script_hash = sha_file(Path(__file__))
    if preflight.get("script_sha256") != script_hash:
        raise RuntimeError("capture script changed after its preflight")
    marker = OUT / "capture_started.json"
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError("one-shot capture marker already exists; refusing a second constructor") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"status": "one_shot_capture_started", "started_utc": utc_now(), "script_sha256": script_hash}, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    started = time.monotonic()
    result = {
        "schema_version": "flygym-native-model-reference-capture-v1",
        "status": "started",
        "started_utc": utc_now(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": script_hash,
        "preflight_sha256": sha_file(pre_path),
        "sys_executable": sys.executable,
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "expected_python": str(EXPECTED_PYTHON),
        "expected_python_realpath": str(EXPECTED_PYTHON.resolve()),
        "cgroup_before_constructor": cgroup_attestation(require_unit=True),
        "constructor_calls": 0,
        "integration_calls": 0,
        "gpu_used": False,
        "neural_forward_called": False,
        "gate_modified": False,
        "failed_r3_r4_records_modified": False,
    }
    session = None
    try:
        if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
            raise RuntimeError("capture is not using the pinned FlyGym environment interpreter")
        os.environ.update({
            "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1",
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "FLYGYM_ASSET_CACHE_DIR": "/data/models/flygym/assets",
            "MPLCONFIGDIR": "/data/cache/flygym/matplotlib",
        })
        result["environment"] = {key: os.environ.get(key) for key in (
            "MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBGL_ALWAYS_SOFTWARE", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "FLYGYM_ASSET_CACHE_DIR",
        )}
        sys.path.insert(0, str(PROJECT / "scripts"))
        import numpy as np
        import mujoco
        import flygym
        from run_flygym_reference_walk import FlyGymReferenceSession, WalkConfig
        from flygym.simulation import Simulation

        result["package_versions"] = {
            "flygym": metadata.version("flygym"),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        }
        result["runtime_module_origins"] = {
            name: getattr(sys.modules.get(name), "__file__", None)
            for name in ("flygym", "flygym.simulation", "mujoco", "numpy", "run_flygym_reference_walk", "flygym_legacy_rest_compat")
        }
        if result["package_versions"] != {"flygym": "2.1.0", "mujoco": "3.9.0", "numpy": "2.5.3"}:
            raise RuntimeError(f"runtime package versions differ from pinned initialization: {result['package_versions']}")
        if Path(result["runtime_module_origins"]["run_flygym_reference_walk"]).resolve() != Path(PROJECT / "scripts/run_flygym_reference_walk.py").resolve():
            raise RuntimeError("runtime session builder imported from an unexpected path")
        result["upstream_source_preconstruction"] = upstream_source_preflight()
        original_step = Simulation.step
        integration_guard = {"calls": 0}

        def forbid_sim_step(_self, *_args, **_kwargs):
            integration_guard["calls"] += 1
            raise RuntimeError("Simulation.step is forbidden during constructor-only capture")

        Simulation.step = forbid_sim_step
        walk = dict(preflight["walk_config"])
        for key in ("spawn_position_mm", "spawn_quaternion_wxyz", "position_actuator_forcerange_native", "camera_resolution_hw"):
            if key in walk:
                walk[key] = tuple(walk[key])
        result["constructor_calls"] = 1
        try:
            session = FlyGymReferenceSession(
                duration_s=7.0,
                seed=1729,
                render_video=False,
                config=WalkConfig(**walk),
                run_id=RUN_ID,
            )
        finally:
            Simulation.step = original_step
        result["integration_calls"] = integration_guard["calls"]

        if session.model is not session.sim.mj_model or session.data is not session.sim.mj_data:
            raise RuntimeError("session model/data do not directly reference FlyGym Simulation model/data")

        # Save the actual compiled object and its constructor state before any
        # analysis calls.  The XML is saved as provenance only.
        mjb_path = OUT / "actual_session_model.mjb"
        session.mj.mj_saveModel(session.model, str(mjb_path), None)
        xml_bytes = session.compiled_mjcf_xml.encode("utf-8")
        xml_path = OUT / "constructor_emitted_mjcf.xml"
        xml_path.write_bytes(xml_bytes)
        qpos = np.ascontiguousarray(session.data.qpos, dtype=np.float64).copy()
        qvel = np.ascontiguousarray(session.data.qvel, dtype=np.float64).copy()
        data_path = OUT / "constructor_state.npz"
        np.savez(data_path, qpos=qpos, qvel=qvel, data_time_s=np.asarray(float(session.data.time), dtype=np.float64))
        effective_options = dict(session.effective_model_options)
        options_path = OUT / "effective_model_options.json"
        write_json(options_path, effective_options)
        time_s = float(session.time_s)
        data_time_s = float(session.data.time)
        counters = {
            "session_time_s": time_s,
            "data_time_s": data_time_s,
            "control_samples": int(session._control_samples),
            "constructor_native_samples": int(session._native_samples),
            "simulation_curr_step": int(session.sim._curr_step),
            "root_qpos_write_count": int(session._root_qpos_write_count),
            "root_qvel_write_count": int(session._root_qvel_write_count),
            "constructor_step_guard_calls": int(integration_guard["calls"]),
            "mj_forward_called_by_session_constructor": True,
        }

        order = session._array_order_names()
        order_path = OUT / "runtime_array_order.json"
        write_json(order_path, order)
        prior_order_path = REV4 / "frozen_inputs/prior_array_order.json"
        prior_order = read_json(prior_order_path)
        prior_options = read_json(REV4 / "frozen_inputs/prior_model_options.json")
        initial = read_json(PAIR_CASE / "initial_state.json")
        qpos_sha = sha_bytes(qpos.tobytes(order="C"))
        qvel_sha = sha_bytes(qvel.tobytes(order="C"))
        prior_qpos_sha = sha_bytes(struct.pack("<%sd" % len(initial["qpos"]), *initial["qpos"]))
        prior_qvel_sha = sha_bytes(struct.pack("<%sd" % len(initial["qvel"]), *initial["qvel"]))
        model_sha = sha_file(mjb_path)
        r4_actual_mjb = REV4 / "runs/seed1729_dzero_0p00/compiled_model.mjb"
        direct_xml_mjb = REV4 / "frozen_inputs/compiled_model_reference.mjb"
        model_identity = {
            "actual_session_mjb_path": str(mjb_path),
            "actual_session_mjb_sha256": model_sha,
            "actual_session_mjb_size_bytes": mjb_path.stat().st_size,
            "matches_r4_saved_session_mjb_sha256": model_sha == sha_file(r4_actual_mjb),
            "r4_saved_session_mjb_sha256": sha_file(r4_actual_mjb),
            "matches_direct_xml_mjb_sha256": model_sha == sha_file(direct_xml_mjb),
            "direct_xml_mjb_sha256": sha_file(direct_xml_mjb),
            "native_source": "mujoco.mj_saveModel(session.model, ...), where session.model is session.sim.mj_model",
            "direct_xml_recompiled_as_baseline": False,
        }
        state_bridge = {
            "qpos_shape": list(qpos.shape),
            "qvel_shape": list(qvel.shape),
            "captured_qpos_sha256": qpos_sha,
            "captured_qvel_sha256": qvel_sha,
            "paired_screen_qpos_sha256": prior_qpos_sha,
            "paired_screen_qvel_sha256": prior_qvel_sha,
            "qpos_bytes_equal_to_paired_screen_initial": qpos_sha == prior_qpos_sha,
            "qvel_bytes_equal_to_paired_screen_initial": qvel_sha == prior_qvel_sha,
            "paired_screen_initial_time_s": initial.get("time_s"),
            "paired_screen_initial_source": initial.get("source"),
            "array_order_equal_to_frozen_paired_screen": order == prior_order,
            "array_order_sha256": sha_file(order_path),
            "frozen_paired_screen_array_order_sha256": sha_file(prior_order_path),
        }
        xml_info = {
            "path": str(xml_path),
            "sha256": sha_bytes(xml_bytes),
            "bytes": len(xml_bytes),
            "matches_original_initialization_xml": sha_bytes(xml_bytes) == sha_file(ORIGINAL_AUDIT / "compiled_model.xml"),
            "matches_r4_runtime_xml": sha_bytes(xml_bytes) == sha_file(REV4 / "runs/seed1729_dzero_0p00/compiled_model.xml"),
            "semantic_role": "actual session's MjSpec.to_xml serialization, retained as provenance; not a binary baseline and not recompiled",
        }
        result.update({
            "status": "complete",
            "finished_constructor_utc": utc_now(),
            "model": model_identity,
            "xml": xml_info,
            "effective_model_options": effective_options,
            "effective_model_options_match_paired_screen": effective_options == prior_options,
            "effective_model_options_sha256": canonical_config_sha(effective_options),
            "constructor_counters": counters,
            "constructor_state": {
                "npz_path": str(data_path),
                "npz_sha256": sha_file(data_path),
                "qpos_sha256": qpos_sha,
                "qvel_sha256": qvel_sha,
            },
            "historical_bridge": state_bridge,
            "runtime_assets_root": str(flygym.assets_dir),
            "runtime_upstream_source_hashes": session.upstream_source_hashes,
            "no_integration_performed": integration_guard["calls"] == 0 and counters["simulation_curr_step"] == 0 and counters["constructor_native_samples"] == 0,
            "baseline_promotion": False,
            "identity_gate_modified": False,
            "revision_3_revision_4_failed_records_preserved": True,
            "interpretation_limits": [
                "Exact equality to the retained revision-4 session MJB establishes those saved compiled bytes for this pinned source/config/asset/environment capture; it does not create a new acceptance gate.",
                "The older paired-screen static case retained XML and initial state but no MJB; historical exact model bytes before revision 4 remain unproved.",
                "The constructor-emitted XML is an MJSpec serialization and is not treated as the compiled native model's exact numeric serialization.",
                "No dynamics, body motion, NN, GPU, steering dose, or biological outcome was run.",
            ],
        })
        if time_s != 0.0 or data_time_s != 0.0 or any(counters[key] != 0 for key in (
            "control_samples", "constructor_native_samples", "simulation_curr_step", "root_qpos_write_count", "root_qvel_write_count", "constructor_step_guard_calls"
        )):
            raise RuntimeError("constructor-only invariant failed; time/counters/state writes advanced")
        if not state_bridge["qpos_bytes_equal_to_paired_screen_initial"] or not state_bridge["qvel_bytes_equal_to_paired_screen_initial"]:
            result["historical_bridge_status"] = "not_exact_initial_state_bridge"
        else:
            result["historical_bridge_status"] = "exact_qpos_qvel_array_order_bridge"
        if session is not None:
            before_close = (float(session.time_s), int(session._native_samples), int(session._control_samples))
            session.close()
            result["after_close"] = {
                "session_time_s": float(session.time_s),
                "native_samples": int(session._native_samples),
                "control_samples": int(session._control_samples),
                "counters_unchanged_by_close": before_close == (float(session.time_s), int(session._native_samples), int(session._control_samples)),
            }
        comparison = {
            "schema_version": "flygym-native-model-reference-comparison-v1",
            "classification": "read-only constructor-only native compiled-model capture",
            "captured_model": result.get("model"),
            "r4_saved_session_model": {
                "path": str(REV4 / "runs/seed1729_dzero_0p00/compiled_model.mjb"),
                "sha256": sha_file(REV4 / "runs/seed1729_dzero_0p00/compiled_model.mjb"),
            },
            "direct_xml_compiled_reference": {
                "path": str(REV4 / "frozen_inputs/compiled_model_reference.mjb"),
                "sha256": sha_file(REV4 / "frozen_inputs/compiled_model_reference.mjb"),
            },
            "constructor_emitted_xml": result.get("xml"),
            "effective_model_options_match_paired_screen": result.get("effective_model_options_match_paired_screen"),
            "historical_bridge": result.get("historical_bridge"),
            "constructor_counters": result.get("constructor_counters"),
            "baseline_promotion": False,
            "identity_gate_modified": False,
            "existing_r3_r4_failures_modified": False,
            "interpretation": (
                "The actual session.model MJB is the native baseline artifact. The emitted MJSpec XML and an MJB recompiled from it are comparison/provenance inputs only."
            ),
            "limits": [
                "A byte match to the r4 saved session MJB proves reproducibility of that retained zero-step runtime model for the pinned inputs; it does not retroactively prove older paired-screen MJB bytes.",
                "The paired-screen static case has a full saved initial qpos/qvel and array order but no retained compiled_model.mjb.",
                "No physics integration or behavior was run.",
            ],
        }
        result["comparison"] = comparison
        write_json(OUT / "comparison.json", comparison, exclusive=True)
        markdown = f"""# Native FlyGym model reference capture

Status: **{result.get('status')}**. This is a zero-step constructor capture for the frozen `seed=1729`, `duration=7 s`, run ID `{RUN_ID}` configuration.

## Native compiled model

- Captured from `session.model` using `mujoco.mj_saveModel`; it is the same object as `session.sim.mj_model`.
- Captured SHA-256: `{result.get('model', {}).get('actual_session_mjb_sha256')}`.
- Revision-4 saved runtime MJB SHA-256: `{result.get('model', {}).get('r4_saved_session_mjb_sha256')}`.
- Exact byte-hash match to revision 4: `{result.get('model', {}).get('matches_r4_saved_session_mjb_sha256')}`.
- Direct-XML compiled reference SHA-256: `{result.get('model', {}).get('direct_xml_mjb_sha256')}`; exact match: `{result.get('model', {}).get('matches_direct_xml_mjb_sha256')}`.
- Baseline promotion and identity-gate changes: **false**.

The saved `constructor_emitted_mjcf.xml` is the session's `MjSpec.to_xml()` serialization retained as provenance. It is not recompiled to produce the native baseline and is not treated as an exact serialization of compiled model numeric fields.

## Historical state bridge

- Constructor qpos SHA-256: `{result.get('historical_bridge', {}).get('captured_qpos_sha256')}`; paired-screen qpos SHA-256: `{result.get('historical_bridge', {}).get('paired_screen_qpos_sha256')}`.
- Constructor qvel SHA-256: `{result.get('historical_bridge', {}).get('captured_qvel_sha256')}`; paired-screen qvel SHA-256: `{result.get('historical_bridge', {}).get('paired_screen_qvel_sha256')}`.
- Exact qpos/qvel equality: `{result.get('historical_bridge', {}).get('qpos_bytes_equal_to_paired_screen_initial')}` / `{result.get('historical_bridge', {}).get('qvel_bytes_equal_to_paired_screen_initial')}`.
- Array-order equality to frozen paired-screen order: `{result.get('historical_bridge', {}).get('array_order_equal_to_frozen_paired_screen')}`.
- The paired-screen state record says FlyGym keyframe reset followed by `mj_forward`; it contains no saved MJB. The newer revision-4 run does retain its actual session MJB, and its failed control capture remains unchanged.

## Zero-step and resource record

`session.time_s={result.get('constructor_counters', {}).get('session_time_s')}`, `data.time={result.get('constructor_counters', {}).get('data_time_s')}`, control/native samples `{result.get('constructor_counters', {}).get('control_samples')}/{result.get('constructor_counters', {}).get('constructor_native_samples')}`, and guarded `Simulation.step` calls `{result.get('constructor_counters', {}).get('constructor_step_guard_calls')}`. Constructor `mj_forward` is expected and is not an integration step.

The capture ran under the attested systemd cgroup with CPU cap `{result.get('cgroup_before_constructor', {}).get('effective_cpu_cap_cores')}` cores, memory cap `{result.get('cgroup_before_constructor', {}).get('effective_memory_cap_bytes')}` bytes, and runtime cap `{LIMIT_RUNTIME_SECONDS}` seconds. See `capture.json` and `terminal.json` for cgroup and process exit evidence.

No NN, GPU, dose, body stepping, tests, or biological outcome was run. Revision-3/revision-4 failed records were preserved.
"""
        (OUT / "README.md").write_text(markdown, encoding="utf-8")
    except BaseException as exc:
        result.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        if session is not None:
            try:
                session.close()
            except BaseException as close_exc:
                result["close_error"] = f"{type(close_exc).__name__}: {close_exc}"
    finally:
        try:
            result["cgroup_after_capture"] = cgroup_attestation(require_unit=True)
        except BaseException as cg_exc:
            result["cgroup_after_error"] = f"{type(cg_exc).__name__}: {cg_exc}"
        result["elapsed_wall_s"] = time.monotonic() - started
        result["finished_utc"] = utc_now()
        try:
            write_json(OUT / "capture.json", result, exclusive=True)
        except FileExistsError:
            result["record_write_error"] = "capture.json already exists"
            raise
    return 0 if result.get("status") == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="verify frozen inputs/source/assets without constructing FlyGym")
    mode.add_argument("--capture", action="store_true", help="one-shot zero-step capture of FlyGymReferenceSession.model")
    args = parser.parse_args(argv)
    if args.preflight:
        record = preflight_record()
        OUT.mkdir(parents=True, exist_ok=False)
        write_json(OUT / "preflight.json", record, exclusive=True)
        print(OUT / "preflight.json")
        return 0
    return capture()


if __name__ == "__main__":
    raise SystemExit(main())
