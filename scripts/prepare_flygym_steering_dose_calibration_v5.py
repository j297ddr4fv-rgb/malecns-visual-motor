#!/usr/bin/env python3
"""Prepare the frozen revision-5 FlyGym mechanical-dose bundle.

This standard-library-only preparer inherits the revision-4 source/assets and
case design, then adds the independently captured native session.model MJB as
the strict exact-byte reference. It verifies and snapshots pinned evidence; it
does not import FlyGym or MuJoCo, compile a model, run a simulation, or create
an execution-approval artifact.

Prepare once with:

    python scripts/prepare_flygym_steering_dose_calibration_v5.py --prepare
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
BASE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1"
)
OUTPUT_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-5"
REVISION_4_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-4"
NATIVE_REFERENCE_ROOT = BASE / "mechanical-native-model-reference-v1"
NATIVE_CAPTURE_SCRIPT = PROJECT / "scripts/capture_flygym_native_model_reference_v1.py"

EXPECTED_NATIVE_MJB_SHA256 = "6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3"
EXPECTED_NATIVE_MJB_SIZE_BYTES = 23230821
EXPECTED_DIRECT_XML_MJB_SHA256 = "c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f"
EXPECTED_NATIVE_CAPTURE_SHA256 = "4e7b0ea4bd7050ecf00a55fb6f1cea2bca3481a2484182db02d748c56f772639"
EXPECTED_NATIVE_COMPARISON_SHA256 = "60dfdb19741554c47b9656f75ef477844e4816d2b72d6647f323aba4708a1e07"
EXPECTED_NATIVE_TERMINAL_SHA256 = "82d1c9437fd9129b3a639de2c96f9b3e0625570cf1c3dc71533f1a855f4ada86"
EXPECTED_NATIVE_MANIFEST_SHA256 = "2152e2c224e277ee94d9b537f1f351686dddd2da3acffb0e87bf6317f49fd4ff"
EXPECTED_QPOS_SHA256 = "d292b37df0f9a83265423da2ce53b0b140d8ef6a49d6909e464b805453e75b99"
EXPECTED_QVEL_SHA256 = "1a0295f4bf5986c5f74eca9153a6a4cb10b073a01a76ba4a457fd862c78966a4"
EXPECTED_ARRAY_ORDER_SHA256 = "0c8cdbb7006f774f42a19f068312605879a4ea7ca73e3db77e21e4903db14b0b"
EXPECTED_MODEL_OPTIONS_SHA256 = "be19cd0cc6f915baeabee9187f4ff9e92c1b8665aa929abbcf00a0024eee2ae4"
EXPECTED_PACKAGE_VERSIONS = {"flygym": "2.1.0", "mujoco": "3.9.0", "numpy": "2.5.3"}
EXPECTED_FLYGYM_COMMIT = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"


class PreparationError(RuntimeError):
    """Raised when a frozen source, reference, or parent bundle has changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"expected a JSON object: {path}")
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pin_file(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationError(f"required pinned file is missing: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size, "role": role}


def copy_verified(src: Path, dst: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if not src.is_file():
        raise PreparationError(f"required input is missing: {src}")
    observed = sha256_file(src)
    if expected_sha is not None and observed != expected_sha:
        raise PreparationError(f"input hash differs from its frozen pin: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    copied = sha256_file(dst)
    if copied != observed:
        raise PreparationError(f"copy verification failed: {dst}")
    return {"sha256": copied, "size_bytes": dst.stat().st_size}


def verify_path_pin(path: Path, pinned: dict[str, Any], description: str) -> None:
    if not path.is_file():
        raise PreparationError(f"missing {description}: {path}")
    if path.stat().st_size != pinned.get("size_bytes") or sha256_file(path) != pinned.get("sha256"):
        raise PreparationError(f"changed {description}: {path}")


def validate_native_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = NATIVE_REFERENCE_ROOT / "manifest.json"
    capture_path = NATIVE_REFERENCE_ROOT / "capture.json"
    comparison_path = NATIVE_REFERENCE_ROOT / "comparison.json"
    terminal_path = NATIVE_REFERENCE_ROOT / "terminal.json"
    manifest = read_json(manifest_path)
    capture = read_json(capture_path)
    comparison = read_json(comparison_path)
    terminal = read_json(terminal_path)
    if manifest.get("schema_version") != "flygym-native-model-reference-bundle-manifest-v1":
        raise PreparationError("native reference manifest schema changed")
    if manifest.get("status") != "complete_constructor_only_capture":
        raise PreparationError("native reference manifest is not a completed constructor-only capture")
    if manifest.get("native_mjb_sha256") != EXPECTED_NATIVE_MJB_SHA256:
        raise PreparationError("native reference manifest MJB hash differs from the frozen pin")
    if manifest.get("native_mjb_size_bytes") != EXPECTED_NATIVE_MJB_SIZE_BYTES:
        raise PreparationError("native reference manifest MJB size differs from the frozen pin")
    if capture.get("status") != "complete" or terminal.get("native_capture_status") != "complete":
        raise PreparationError("native model capture is not complete")
    if terminal.get("actual_exec_exit_code") != 0 or terminal.get("systemd_result") != "success":
        raise PreparationError("native model capture did not exit successfully")
    if capture.get("gpu_used") is not False or capture.get("no_integration_performed") is not True:
        raise PreparationError("native reference was not a CPU constructor-only capture")
    if capture.get("integration_calls") != 0 or capture.get("neural_forward_called") is not False:
        raise PreparationError("native reference performed integration or neural work")
    if capture.get("package_versions") != EXPECTED_PACKAGE_VERSIONS:
        raise PreparationError("native reference runtime package versions differ from the reviewed pins")
    upstream_preconstruction = capture.get("upstream_source_preconstruction", {})
    if (upstream_preconstruction.get("all_match") is not True
            or upstream_preconstruction.get("commit_match") is not True
            or upstream_preconstruction.get("actual_commit") != EXPECTED_FLYGYM_COMMIT
            or upstream_preconstruction.get("expected_commit") != EXPECTED_FLYGYM_COMMIT):
        raise PreparationError("native capture FlyGym upstream source/commit verification failed")
    cgroup = capture.get("cgroup_before_constructor", {})
    if (cgroup.get("effective_cpu_cap_cores") != 2.0
            or cgroup.get("effective_memory_cap_bytes") != 2 * 1024 * 1024 * 1024):
        raise PreparationError("native reference was not captured under the pinned CPU2/Mem2G caps")
    counters = capture.get("comparison", {}).get("constructor_counters", {})
    expected_zero_counters = {
        "constructor_native_samples": 0,
        "constructor_step_guard_calls": 0,
        "control_samples": 0,
        "data_time_s": 0.0,
        "session_time_s": 0.0,
        "simulation_curr_step": 0,
        "root_qpos_write_count": 0,
        "root_qvel_write_count": 0,
    }
    if any(counters.get(key) != value for key, value in expected_zero_counters.items()):
        raise PreparationError("native capture constructor time/counters are not all zero")
    if counters.get("mj_forward_called_by_session_constructor") is not True:
        raise PreparationError("native capture does not record the expected constructor mj_forward")

    native = capture.get("comparison", {}).get("captured_model", {})
    if (native.get("actual_session_mjb_sha256") != EXPECTED_NATIVE_MJB_SHA256
            or native.get("actual_session_mjb_size_bytes") != EXPECTED_NATIVE_MJB_SIZE_BYTES
            or native.get("native_source") != "mujoco.mj_saveModel(session.model, ...), where session.model is session.sim.mj_model"):
        raise PreparationError("captured MJB is not pinned to the actual session.model object")
    if (native.get("matches_r4_saved_session_mjb_sha256") is not True
            or native.get("r4_saved_session_mjb_sha256") != EXPECTED_NATIVE_MJB_SHA256
            or native.get("matches_direct_xml_mjb_sha256") is not False
            or native.get("direct_xml_mjb_sha256") != EXPECTED_DIRECT_XML_MJB_SHA256
            or native.get("direct_xml_recompiled_as_baseline") is not False):
        raise PreparationError("native capture does not distinguish the MjSpec model from XML round-trip bytes")
    bridge = capture.get("comparison", {}).get("historical_bridge", {})
    if (bridge.get("qpos_bytes_equal_to_paired_screen_initial") is not True
            or bridge.get("qvel_bytes_equal_to_paired_screen_initial") is not True
            or bridge.get("paired_screen_qpos_sha256") != EXPECTED_QPOS_SHA256
            or bridge.get("paired_screen_qvel_sha256") != EXPECTED_QVEL_SHA256
            or bridge.get("captured_qpos_sha256") != EXPECTED_QPOS_SHA256
            or bridge.get("captured_qvel_sha256") != EXPECTED_QVEL_SHA256
            or bridge.get("array_order_equal_to_frozen_paired_screen") is not True
            or bridge.get("array_order_sha256") != EXPECTED_ARRAY_ORDER_SHA256
            or bridge.get("frozen_paired_screen_array_order_sha256") != EXPECTED_ARRAY_ORDER_SHA256
            or capture.get("comparison", {}).get("effective_model_options_match_paired_screen") is not True
            or capture.get("effective_model_options_sha256") != EXPECTED_MODEL_OPTIONS_SHA256):
        raise PreparationError("native capture fails the exact historical qpos/qvel/options/order bridge")
    if (capture.get("baseline_promotion") is not False
            or capture.get("comparison", {}).get("baseline_promotion") is not False
            or capture.get("identity_gate_modified") is not False
            or capture.get("gate_modified") is not False
            or capture.get("failed_r3_r4_records_modified") is not False):
        raise PreparationError("native capture may not promote the reference or modify previous gates/records")
    if capture.get("revision_3_revision_4_failed_records_preserved") is not True:
        raise PreparationError("native capture does not confirm preservation of prior failure records")

    files = manifest.get("files", {})
    for name, pin in files.items():
        verify_path_pin(NATIVE_REFERENCE_ROOT / name, pin, f"native evidence file {name}")
    actual_pin = files.get("actual_session_model.mjb", {})
    if actual_pin.get("sha256") != EXPECTED_NATIVE_MJB_SHA256 or actual_pin.get("size_bytes") != EXPECTED_NATIVE_MJB_SIZE_BYTES:
        raise PreparationError("native reference manifest file pin differs from the strict runtime gate")
    if sha256_file(capture_path) != EXPECTED_NATIVE_CAPTURE_SHA256:
        raise PreparationError("native capture record hash differs from the reviewed pin")
    if sha256_file(comparison_path) != EXPECTED_NATIVE_COMPARISON_SHA256:
        raise PreparationError("native comparison record hash differs from the reviewed pin")
    if sha256_file(terminal_path) != EXPECTED_NATIVE_TERMINAL_SHA256:
        raise PreparationError("native terminal record hash differs from the reviewed pin")
    if sha256_file(manifest_path) != EXPECTED_NATIVE_MANIFEST_SHA256:
        raise PreparationError("native manifest hash differs from the reviewed pin")
    return manifest, capture, comparison, terminal


def protocol_for_revision5(r4_protocol: dict[str, Any], runner_sha: str,
                           r4_freeze_sha: str, native_pins: dict[str, Any],
                           historical_records: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads(json.dumps(r4_protocol))
    protocol["schema_version"] = "flygym-steering-dose-calibration-protocol-v5"
    protocol["revision"] = 5
    protocol["entrypoint_source_sha256"] = runner_sha
    protocol["preparation_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    protocol["source_parent"] = {
        "revision_4_bundle": str(REVISION_4_ROOT),
        "revision_4_source_freeze_sha256": r4_freeze_sha,
        "revision_4_protocol_sha256": sha256_file(REVISION_4_ROOT / "protocol.json"),
        "native_model_reference_bundle": str(NATIVE_REFERENCE_ROOT),
        "native_capture_sha256": native_pins["capture_sha256"],
        "native_comparison_sha256": native_pins["comparison_sha256"],
        "native_terminal_sha256": native_pins["terminal_sha256"],
        "native_manifest_sha256": native_pins["manifest_sha256"],
        "historical_failure_records": historical_records,
    }
    protocol["execution"]["entrypoint"] = str(
        PROJECT / "scripts/run_flygym_steering_dose_calibration_v5.py"
    )
    protocol["execution"]["invocation"] = (
        "/data/environments/projects/flygym-reference/bin/python "
        "scripts/run_flygym_steering_dose_calibration_v5.py --execute"
    )
    protocol["execution"]["model_identity_gate"] = {
        "method": "save the complete compiled model from the actual FlyGym session.model with mujoco.mj_saveModel, then SHA-256 the saved MJB bytes",
        "expected_sha256": EXPECTED_NATIVE_MJB_SHA256,
        "expected_size_bytes": EXPECTED_NATIVE_MJB_SIZE_BYTES,
        "comparison": "exact full-file SHA-256 and size equality to the independent native constructor capture; no numerical tolerance, subset comparison, or hash allowlist",
        "raw_xml": "always preserve raw constructor XML and hash; record whether it matches the prior paired-screen XML and the known revision-3 serialization; raw XML equality is provenance only and is not a gate",
        "constructor_invariants": [
            "session.model is session.sim.mj_model",
            "session and MuJoCo data time are zero and constructor control/native sample counters are zero",
            "save and gate the actual session model before dose-controller construction or dynamics",
            "effective model options exactly equal the frozen paired-screen options",
            "runtime model array order exactly equals the frozen paired-screen array order",
            "constructor qpos/qvel SHA-256 exactly equals the historical paired-screen initial state and the same-seed reference case",
        ],
        "native_reference": {
            "capture_sha256": native_pins["capture_sha256"],
            "comparison_sha256": native_pins["comparison_sha256"],
            "terminal_sha256": native_pins["terminal_sha256"],
            "manifest_sha256": native_pins["manifest_sha256"],
            "actual_session_mjb_sha256": EXPECTED_NATIVE_MJB_SHA256,
            "actual_session_mjb_size_bytes": EXPECTED_NATIVE_MJB_SIZE_BYTES,
            "capture_status": "complete_constructor_only_capture",
            "constructor_integration_steps": 0,
            "baseline_promotion": False,
        },
        "historical_bridge": {
            "paired_screen_qpos_sha256": EXPECTED_QPOS_SHA256,
            "paired_screen_qvel_sha256": EXPECTED_QVEL_SHA256,
            "paired_screen_array_order_sha256": EXPECTED_ARRAY_ORDER_SHA256,
            "paired_screen_model_options_sha256": EXPECTED_MODEL_OPTIONS_SHA256,
            "native_capture_qpos_qvel_array_order_and_options_all_match": True,
            "historical_paired_screen_compiled_mjb_retained": False,
        },
        "revision_4_outcome": {
            "record_path": historical_records["revision_4"]["path"],
            "record_sha256": historical_records["revision_4"]["sha256"],
            "completed_cases": 0,
            "control_ticks": 0,
            "interpretation": "failed before any dose-control tick; contains no control outcome to promote",
        },
        "native_model_difference_from_xml_roundtrip": {
            "xml_roundtrip_mjb_sha256": EXPECTED_DIRECT_XML_MJB_SHA256,
            "native_mjb_equals_xml_roundtrip": False,
            "evidence": "the actual session.model native MJB capture and r4 saved session MJB agree exactly; direct XML recompilation is not the native model",
        },
    }
    protocol["instrumentation"] = {
        "preconstruction": protocol["instrumentation"].get("preconstruction", []),
        "immediately_after_constructor_before_controller_or_dynamics": [
            "preserve constructor raw MJCF bytes and SHA-256; record match status against historical XML without requiring equality",
            "save the complete actual session.model to compiled_model.mjb and require the strict native reference SHA-256 and size",
            "record zero-time/sample counters, effective model options, exact runtime array order, exact historical qpos/qvel hashes, source hashes, module origins, and versions",
        ],
        "model_gate": (
            "require exact SHA-256 6d7b1f8b9a7558cd5367be9e12e1a866adca3d98e6312d431b62d27ad747b2f3 and size 23230821 for the complete MJB saved from session.model, plus exact old options/array-order/qpos/qvel checks, before controller construction or dynamics; raw XML hash is provenance only"
        ),
        "model_or_controller_changes": False,
    }
    protocol["revision_5_changes"] = [
        "Use the independent zero-step native session.model capture as the single strict full-MJB byte reference; the XML-roundtrip MJB is not promoted.",
        "Retain raw XML bytes/hash and old paired-screen raw hash as serialization provenance only.",
        "Require exact effective-model-options, runtime-array-order, and historical initial qpos/qvel matches before controller construction or dynamics.",
        "Keep the revision-4 15-case seeds/P/d schedule/controller/source/assets/resource caps/timeouts/queue serialization/capture design unchanged.",
        "Preserve all revision-2/revision-3/revision-4 failure records; revision 4 has no completed dose case or control outcome to promote.",
    ]
    return protocol


def prepare() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise PreparationError(f"refusing to overwrite existing revision-5 bundle: {OUTPUT_ROOT}")
    r4_freeze_path = REVISION_4_ROOT / "source_freeze.json"
    r4_protocol_path = REVISION_4_ROOT / "protocol.json"
    r4_review_path = REVISION_4_ROOT / "preparation-review.json"
    r4_freeze = read_json(r4_freeze_path)
    r4_protocol = read_json(r4_protocol_path)
    r4_review = read_json(r4_review_path)
    if r4_freeze.get("revision") != 4 or r4_freeze.get("status") != "preparation_frozen_execution_entrypoint_ready_not_run":
        raise PreparationError("revision-4 source freeze is not the frozen parent bundle")
    if r4_freeze.get("protocol_sha256") != sha256_file(r4_protocol_path) or r4_protocol.get("revision") != 4:
        raise PreparationError("revision-4 protocol no longer matches its source freeze")
    if r4_review.get("status") != "prepared_not_executed" or r4_review.get("runner_sha256") != r4_freeze.get("runner_sha256"):
        raise PreparationError("revision-4 review does not identify its frozen parent runner")

    # Reverify the complete inherited source/asset/input set before freezing r5.
    for raw_path, pinned in r4_freeze.get("source_files", {}).items():
        verify_path_pin(Path(raw_path), pinned, "revision-4 inherited source")
    assets = r4_freeze.get("asset_manifest", {})
    for logical_name, pinned in assets.get("files", {}).items():
        verify_path_pin(Path(pinned["physical_path"]), pinned, f"upstream asset {logical_name}")
    if assets.get("asset_count") != 79 or assets.get("verified_asset_count") != 79:
        raise PreparationError("revision-4 inherited asset count is not exactly 79")
    for filename, pinned in r4_freeze.get("copied_inputs", {}).items():
        verify_path_pin(Path(pinned["snapshot_path"]), pinned, f"revision-4 copied input {filename}")
    for filename, pinned in r4_freeze.get("bundle_files", {}).items():
        verify_path_pin(REVISION_4_ROOT / filename, pinned, f"revision-4 bundle file {filename}")

    native_manifest, native_capture, native_comparison, native_terminal = validate_native_evidence()
    native_source_records: dict[str, dict[str, Any]] = {}
    for name, role in (
        ("manifest.json", "independent native-reference bundle manifest"),
        ("capture.json", "zero-step actual-session-model capture and historical bridge"),
        ("comparison.json", "native MJB and historical state/options/order comparison"),
        ("terminal.json", "successful terminal attestation for zero-step native capture"),
        ("actual_session_model.mjb", "strict full native session.model reference MJB"),
        ("effective_model_options.json", "native capture effective options"),
        ("runtime_array_order.json", "native capture runtime array-order map"),
        ("constructor_emitted_mjcf.xml", "native capture raw MJSpec serialization provenance"),
        ("constructor_state.npz", "native capture constructor qpos/qvel bridge bytes"),
    ):
        path = NATIVE_REFERENCE_ROOT / name
        native_source_records[str(path.resolve())] = pin_file(path, role=role)
    if NATIVE_CAPTURE_SCRIPT.is_file():
        native_source_records[str(NATIVE_CAPTURE_SCRIPT.resolve())] = pin_file(
            NATIVE_CAPTURE_SCRIPT, role="script that produced the independent native capture"
        )
        if sha256_file(NATIVE_CAPTURE_SCRIPT) != native_capture.get("script_sha256"):
            raise PreparationError("native capture script hash differs from its capture record")
    else:
        raise PreparationError(f"native capture script is missing: {NATIVE_CAPTURE_SCRIPT}")
    # Freeze the installed FlyGym and pinned checkout source bytes that the native
    # capture checked before construction, plus the upstream license record.
    upstream = native_capture["runtime_upstream_source_hashes"]
    upstream_preconstruction = native_capture["upstream_source_preconstruction"]
    installed_sources = upstream.get("installed_source_sha256", {})
    for raw_path, expected_sha in installed_sources.items():
        path = Path(raw_path)
        record = pin_file(path, role="installed FlyGym source verified by native capture")
        if record["sha256"] != expected_sha:
            raise PreparationError(f"installed upstream FlyGym source changed: {path}")
        native_source_records[str(path.resolve())] = record
    upstream_pairs = upstream_preconstruction["installed_vs_pinned_checkout_files"]
    for relative, pair in upstream_pairs.items():
        for path_key, hash_key, role in (
            ("checkout_path", "checkout_sha256", "pinned FlyGym checkout source verified by native capture"),
            ("installed_path", "installed_sha256", "installed FlyGym source verified against pinned checkout"),
        ):
            path = Path(pair[path_key])
            record = pin_file(path, role=role)
            if record["sha256"] != pair[hash_key]:
                raise PreparationError(f"upstream FlyGym source differs from its capture pin: {relative} ({path})")
            native_source_records[str(path.resolve())] = record
    license_path = Path(upstream["license_path"])
    license_record = pin_file(license_path, role="upstream FlyGym license verified by native capture")
    if license_record["sha256"] != upstream.get("license_sha256"):
        raise PreparationError("pinned FlyGym checkout license changed")
    native_source_records[str(license_path.resolve())] = license_record

    historical_paths = {
        "revision_2": BASE / "mechanical-steering-dose-calibration-v1/revision-2/root-terminal-observation.json",
        "revision_3": BASE / "mechanical-steering-dose-calibration-v1/revision-3/root-terminal-observation.json",
        "revision_4": REVISION_4_ROOT / "root-terminal-observation.json",
    }
    historical_records: dict[str, Any] = {}
    for revision, path in historical_paths.items():
        record = read_json(path)
        pin = pin_file(path, role=f"immutable revision-{revision[-1]} historical failed-run terminal record")
        historical_records[revision] = {"path": str(path), **pin}
        native_source_records[str(path.resolve())] = pin
        if revision == "revision_4" and (
            record.get("actual_exec_exit_code") != 2
            or record.get("completed_cases") != 0
            or record.get("actual_control_ticks") != 0
            or record.get("actual_native_samples") != 0
            or record.get("actual_runtime_mjb_sha256") != EXPECTED_NATIVE_MJB_SHA256
        ):
            raise PreparationError("revision-4 terminal record is not the known pre-tick native-MJB mismatch")
    difference_path = REVISION_4_ROOT / "session-model-difference-v1.json"
    native_source_records[str(difference_path.resolve())] = pin_file(
        difference_path, role="zero-step exposed-field difference report for XML-roundtrip versus native session model"
    )

    runner_path = PROJECT / "scripts/run_flygym_steering_dose_calibration_v5.py"
    prep_path = Path(__file__).resolve(strict=True)
    runner_text = runner_path.read_text(encoding="utf-8")
    if not (runner_path.is_file()
            and "EXPECTED_COMPILED_MJB_SHA256 = \"" + EXPECTED_NATIVE_MJB_SHA256 + "\"" in runner_text
            and runner_text.index("session.mj.mj_saveModel(session.model") < runner_text.index("if not model_options_match_paired_screen")
            and runner_text.index("if not model_options_match_paired_screen") < runner_text.index("controller = PhaseSpeedGaitController(session, zero_config)")
            and runner_text.index("if not array_order_match_paired_screen") < runner_text.index("controller = PhaseSpeedGaitController(session, zero_config)")
            and runner_text.index("if not historical_initial_state_match") < runner_text.index("controller = PhaseSpeedGaitController(session, zero_config)")):
        raise PreparationError("revision-5 runner does not place strict source-state/model gates before controller creation")
    r4_mjb_path = REVISION_4_ROOT / "runs/seed1729_dzero_0p00/compiled_model.mjb"
    r4_mjb_pin = pin_file(r4_mjb_path, role="revision-4 actual zero-step session MJB, preserved as failed-run evidence")
    if r4_mjb_pin["sha256"] != EXPECTED_NATIVE_MJB_SHA256 or r4_mjb_pin["size_bytes"] != EXPECTED_NATIVE_MJB_SIZE_BYTES:
        raise PreparationError("revision-4 retained actual-session MJB differs from native reference")
    native_source_records[str(r4_mjb_path.resolve())] = r4_mjb_pin

    output_parent = OUTPUT_ROOT.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mechanical-dose-rev5-", dir=output_parent))
    try:
        copied_inputs: dict[str, Any] = {}
        for filename, pinned in r4_freeze.get("copied_inputs", {}).items():
            src = Path(pinned["snapshot_path"])
            dst = stage / "frozen_inputs" / filename
            record = copy_verified(src, dst, pinned["sha256"])
            copied_inputs[filename] = {
                **record,
                "source_path": str(src),
                "snapshot_path": str(OUTPUT_ROOT / "frozen_inputs" / filename),
                "role": pinned.get("role", "inherited frozen input"),
                "verified_unchanged_from_revision_4": True,
            }

        # Snapshot every file in the independent capture folder, including the 23 MB native MJB.
        native_copied_inputs: dict[str, Any] = {}
        for name, pinned in native_manifest["files"].items():
            src = NATIVE_REFERENCE_ROOT / name
            dst_relative = Path("frozen_inputs/native_model_reference") / name
            record = copy_verified(src, stage / dst_relative, pinned["sha256"])
            key = f"native_model_reference/{name}"
            native_copied_inputs[key] = {
                **record,
                "source_path": str(src),
                "snapshot_path": str(OUTPUT_ROOT / dst_relative),
                "role": pinned.get("role", "native reference evidence snapshot"),
            }
        manifest_record = pin_file(NATIVE_REFERENCE_ROOT / "manifest.json", role="native-reference manifest snapshot")
        manifest_rel = Path("frozen_inputs/native_model_reference/manifest.json")
        manifest_copy = copy_verified(NATIVE_REFERENCE_ROOT / "manifest.json", stage / manifest_rel, manifest_record["sha256"])
        native_copied_inputs["native_model_reference/manifest.json"] = {
            **manifest_copy,
            "source_path": str(NATIVE_REFERENCE_ROOT / "manifest.json"),
            "snapshot_path": str(OUTPUT_ROOT / manifest_rel),
            "role": "native-reference bundle manifest",
        }
        copied_inputs.update(native_copied_inputs)

        historical_snapshot_names = {
            "revision_2": "revision_2_root_terminal.json",
            "revision_3": "revision_3_root_terminal.json",
            "revision_4": "revision_4_root_terminal.json",
        }
        for revision, filename in historical_snapshot_names.items():
            src = Path(historical_records[revision]["path"])
            dst_relative = Path("frozen_inputs/historical_records") / filename
            record = copy_verified(src, stage / dst_relative, historical_records[revision]["sha256"])
            copied_inputs[f"historical_records/{filename}"] = {
                **record,
                "source_path": str(src),
                "snapshot_path": str(OUTPUT_ROOT / dst_relative),
                "role": f"immutable {revision} failure record",
            }
        diff_snapshot = Path("frozen_inputs/historical_records/revision_4_session_model_difference.json")
        diff_pin = native_source_records[str(difference_path.resolve())]
        diff_copy = copy_verified(difference_path, stage / diff_snapshot, diff_pin["sha256"])
        copied_inputs["historical_records/revision_4_session_model_difference.json"] = {
            **diff_copy,
            "source_path": str(difference_path),
            "snapshot_path": str(OUTPUT_ROOT / diff_snapshot),
            "role": "zero-step native versus XML-roundtrip exposed-field comparison",
        }

        bundle_files: dict[str, Any] = {}

        def snapshot_bundle_file(src: Path, relative: str, expected_sha: str | None = None,
                                 role: str = "immutable parent bundle snapshot") -> None:
            record = copy_verified(src, stage / relative, expected_sha)
            bundle_files[relative] = {**record, "role": role}

        for filename, pinned in r4_freeze.get("bundle_files", {}).items():
            snapshot_bundle_file(REVISION_4_ROOT / filename, filename, pinned["sha256"],
                                 "inherited revision-4 immutable bundle snapshot")
        snapshot_bundle_file(prep_path, f"source_snapshots/revision-5/{prep_path.name}", role="revision-5 standard-library preparation script")
        snapshot_bundle_file(runner_path, f"source_snapshots/revision-5/{runner_path.name}", role="revision-5 frozen execution runner")

        source_files: dict[str, Any] = {
            path: {key: value for key, value in pinned.items() if key in ("sha256", "size_bytes", "role")}
            for path, pinned in r4_freeze.get("source_files", {}).items()
        }
        for path, pinned in native_source_records.items():
            source_files[path] = pinned
        source_files[str(runner_path.resolve())] = pin_file(runner_path, role="revision-5 execution entrypoint")
        source_files[str(prep_path.resolve())] = pin_file(prep_path, role="revision-5 standard-library preparation script")

        native_pins = {
            "capture_sha256": sha256_file(NATIVE_REFERENCE_ROOT / "capture.json"),
            "comparison_sha256": sha256_file(NATIVE_REFERENCE_ROOT / "comparison.json"),
            "terminal_sha256": sha256_file(NATIVE_REFERENCE_ROOT / "terminal.json"),
            "manifest_sha256": sha256_file(NATIVE_REFERENCE_ROOT / "manifest.json"),
        }
        evidence = {
            "capture_snapshot": "frozen_inputs/native_model_reference/capture.json",
            "comparison_snapshot": "frozen_inputs/native_model_reference/comparison.json",
            "terminal_snapshot": "frozen_inputs/native_model_reference/terminal.json",
            "manifest_snapshot": "frozen_inputs/native_model_reference/manifest.json",
            "mjb_snapshot": "frozen_inputs/native_model_reference/actual_session_model.mjb",
        }
        protocol = protocol_for_revision5(
            r4_protocol,
            source_files[str(runner_path.resolve())]["sha256"],
            sha256_file(r4_freeze_path),
            native_pins,
            historical_records,
        )
        if protocol.get("design") != r4_protocol.get("design"):
            raise PreparationError("revision-5 dose design differs from revision 4")
        dump_json(stage / "protocol.json", protocol)
        inherited_asset_manifest = json.loads(json.dumps(assets))
        inherited_asset_manifest["inherited_from_revision_4_source_freeze_sha256"] = sha256_file(r4_freeze_path)
        native_summary = {
            "capture_path": str(NATIVE_REFERENCE_ROOT / "capture.json"),
            "comparison_path": str(NATIVE_REFERENCE_ROOT / "comparison.json"),
            "terminal_path": str(NATIVE_REFERENCE_ROOT / "terminal.json"),
            "manifest_path": str(NATIVE_REFERENCE_ROOT / "manifest.json"),
            "actual_session_mjb_path": str(NATIVE_REFERENCE_ROOT / "actual_session_model.mjb"),
            "actual_session_mjb_snapshot_path": str(OUTPUT_ROOT / evidence["mjb_snapshot"]),
            "expected_mjb_sha256": EXPECTED_NATIVE_MJB_SHA256,
            "expected_mjb_size_bytes": EXPECTED_NATIVE_MJB_SIZE_BYTES,
            "capture_status": "complete_constructor_only_capture",
            "capture_sha256": native_pins["capture_sha256"],
            "comparison_sha256": native_pins["comparison_sha256"],
            "terminal_sha256": native_pins["terminal_sha256"],
            "manifest_sha256": native_pins["manifest_sha256"],
            "evidence": evidence,
            "mjb_snapshot_role": "strict native actual-session-model reference; not an XML recompile",
            "constructor_integration_calls": 0,
            "gpu_used": False,
            "baseline_promotion": False,
            "historical_bridge": {
                "paired_screen_qpos_sha256": EXPECTED_QPOS_SHA256,
                "paired_screen_qvel_sha256": EXPECTED_QVEL_SHA256,
                "paired_screen_array_order_sha256": EXPECTED_ARRAY_ORDER_SHA256,
                "paired_screen_model_options_sha256": EXPECTED_MODEL_OPTIONS_SHA256,
                "native_capture_qpos_qvel_array_order_options_match": True,
                "old_paired_screen_compiled_mjb_retained": False,
            },
            "direct_xml_roundtrip_mjb_sha256": EXPECTED_DIRECT_XML_MJB_SHA256,
            "native_mjb_equals_direct_xml_roundtrip": False,
            "r4_saved_session_mjb_equals_native_reference": True,
            "historical_records": historical_records,
            "native_capture_runtime": {
                "sys_executable": native_capture.get("sys_executable"),
                "python_realpath": native_capture.get("sys_executable_realpath"),
                "package_versions": native_capture.get("package_versions"),
                "runtime_assets_root": native_capture.get("runtime_assets_root"),
                "runtime_upstream_source_hashes": native_capture.get("runtime_upstream_source_hashes"),
                "upstream_source_preconstruction": native_capture.get("upstream_source_preconstruction"),
                "cgroup_before_constructor": native_capture.get("cgroup_before_constructor"),
                "cgroup_after_capture": native_capture.get("cgroup_after_capture"),
            },
        }
        source_freeze = {
            "schema_version": "flygym-steering-dose-calibration-source-freeze-v5",
            "revision": 5,
            "status": "preparation_frozen_execution_entrypoint_ready_not_run",
            "prior_revision_4_source_freeze": {
                "path": str(r4_freeze_path),
                "sha256": sha256_file(r4_freeze_path),
                "all_source_files_reverified": True,
                "all_assets_reverified": True,
                "source_file_count_inherited": len(r4_freeze["source_files"]),
                "asset_count_inherited": assets["asset_count"],
                "verified_asset_count_inherited": assets["verified_asset_count"],
            },
            "source_files": source_files,
            "asset_manifest": inherited_asset_manifest,
            "copied_inputs": copied_inputs,
            "bundle_files": bundle_files,
            "native_model_reference": native_summary,
            "historical_failure_records": historical_records,
            "runner_sha256": source_files[str(runner_path.resolve())]["sha256"],
            "preparation_script_sha256": source_files[str(prep_path.resolve())]["sha256"],
            "protocol_sha256": sha256_file(stage / "protocol.json"),
            "raw_asset_binaries_copied": False,
            "future_execution_must_reverify_all_source_asset_input_and_reference_hashes": True,
            "no_execution_approval_created": True,
            "revision_4_failed_run_records_modified": False,
            "historical_paired_screen_compiled_mjb_retained": False,
            "native_reference_is_independent_capture_not_automatic_baseline_promotion": True,
        }
        dump_json(stage / "source_freeze.json", source_freeze)
        source_freeze_sha = sha256_file(stage / "source_freeze.json")
        review = {
            "schema_version": "flygym-steering-dose-calibration-preparation-review-v5",
            "status": "prepared_not_executed",
            "conclusion": "Revision 5 is prepared for independent review. It gates the complete actual session.model MJB on exact bytes from an independent successful zero-step native capture, and checks exact historical options, array order, and initial qpos/qvel before controller creation. No simulation, model compilation, test, NN, GPU, or execution approval was performed.",
            "entrypoint_path": str(runner_path.resolve()),
            "runner_sha256": source_freeze["runner_sha256"],
            "preparation_script_sha256": source_freeze["preparation_script_sha256"],
            "source_freeze_sha256": source_freeze_sha,
            "protocol_sha256": source_freeze["protocol_sha256"],
            "case_count": len(protocol["execution"]["ordered_cases"]),
            "dose_design_unchanged_from_revision_4": protocol["design"] == r4_protocol["design"],
            "source_and_asset_inheritance": {
                "revision_4_source_file_count": len(r4_freeze["source_files"]),
                "revision_4_asset_count": assets["asset_count"],
                "verified_asset_count": assets["verified_asset_count"],
                "verified_total_asset_bytes": assets["verified_total_bytes"],
                "all_revision_4_sources_reverified": True,
                "all_79_assets_reverified": True,
            },
            "native_model_identity": {
                "strict_expected_mjb_sha256": EXPECTED_NATIVE_MJB_SHA256,
                "strict_expected_mjb_size_bytes": EXPECTED_NATIVE_MJB_SIZE_BYTES,
                "native_capture_sha256": native_pins["capture_sha256"],
                "native_comparison_sha256": native_pins["comparison_sha256"],
                "native_terminal_sha256": native_pins["terminal_sha256"],
                "native_manifest_sha256": native_pins["manifest_sha256"],
                "native_capture_status": "complete_constructor_only_capture",
                "constructor_zero_time_and_samples": True,
                "native_reference_mjb_equals_r4_saved_session_mjb": True,
                "native_reference_mjb_equals_xml_roundtrip_mjb": False,
                "strict_exact_byte_identity": True,
                "numeric_tolerance": None,
                "hash_allowlist": None,
                "raw_xml_gate": False,
            },
            "historical_bridge": {
                "qpos_sha256": EXPECTED_QPOS_SHA256,
                "qvel_sha256": EXPECTED_QVEL_SHA256,
                "array_order_sha256": EXPECTED_ARRAY_ORDER_SHA256,
                "effective_model_options_sha256": EXPECTED_MODEL_OPTIONS_SHA256,
                "all_match_old_paired_screen": True,
                "old_paired_screen_compiled_mjb_available": False,
            },
            "revision_4_failure": {
                "record": historical_records["revision_4"],
                "completed_cases": 0,
                "control_ticks": 0,
                "promotion": False,
            },
            "scope_boundary": "Preparation only; no separate root execution approval or run output was created.",
        }
        dump_json(stage / "preparation-review.json", review)
        review_md = [
            "# Mechanical steering dose calibration — revision 5",
            "",
            "**Status: prepared, not executed.** This bundle has no execution approval and contains no new run output. Revision-2, revision-3, and revision-4 failure records remain unchanged.",
            "",
            "## Strict model identity gate",
            "",
            f"The gate requires the exact complete `session.model` MJB SHA-256 `{EXPECTED_NATIVE_MJB_SHA256}` and size `{EXPECTED_NATIVE_MJB_SIZE_BYTES}` before controller creation or dynamics. This pin comes from the independent constructor-only native capture at `{NATIVE_REFERENCE_ROOT}`; that capture completed successfully at zero time with zero control/native samples and zero integration calls.",
            "",
            f"Revision 4's actual constructor MJB has the same complete bytes. Its expected `{EXPECTED_DIRECT_XML_MJB_SHA256}` reference was created by recompiling serialized MJCF through the six-significant-digit XML representation. The native capture demonstrates these are distinct MJB byte streams. Revision 5 therefore uses the full MJB saved directly from the actual `session.model`; it has no numeric tolerance, subset comparison, or alternate-hash allowlist. The raw XML is kept and hashed as provenance only.",
            "",
            "## Historical bridge and source proof",
            "",
            f"The native capture reports exact qpos `{EXPECTED_QPOS_SHA256}`, qvel `{EXPECTED_QVEL_SHA256}`, array-order `{EXPECTED_ARRAY_ORDER_SHA256}`, and effective-options `{EXPECTED_MODEL_OPTIONS_SHA256}` matches to the paired-screen record. The runner repeats those exact options, order, and qpos/qvel checks immediately after construction, before creating the dose controller. The old paired-screen run did not retain a compiled MJB, so this does not claim retroactive byte identity to that older runtime model.",
            "",
            "The runner saves `session.model` directly, checks it is `session.sim.mj_model`, checks zero session/data time and sample counters, then requires the single exact native MJB hash and size. Existing same-seed pairing, controller, input schedules, and capture gates remain in place.",
            "",
            "## Revision-4 outcome",
            "",
            "Revision 4 exited before any case or control tick because its XML-roundtrip MJB gate did not match the constructor's native MJB. The pinned terminal record reports zero completed cases and zero samples. It has no dose-control outcome to promote; that record remains unchanged.",
            "",
            "## Unchanged cohort and limits",
            "",
            "The 15-case seeds and P/d schedule, 7-second duration, ramp, source/assets, controller, CPU2/Mem2G caps, 600-second case timeout, 9000-second batch timeout, and queue serialization are inherited unchanged from revision 4. No body simulation, NN, GPU, model compilation, tests, or 15-case run was performed.",
            "",
            "The bundle snapshots the complete native reference evidence set and inherits the full revision-4 source/asset/config pins. It does not contain a root execution approval.",
            "",
        ]
        (stage / "preparation-review.md").write_text("\n".join(review_md), encoding="utf-8")
        stage.rename(OUTPUT_ROOT)
        return review
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True,
                        help="verify all pins and prepare the revision-5 bundle")
    parser.parse_args()
    try:
        review = prepare()
    except PreparationError as exc:
        print(f"preparation failed: {exc}")
        return 2
    print(json.dumps({
        "status": review["status"],
        "bundle_root": str(OUTPUT_ROOT),
        "runner_sha256": review["runner_sha256"],
        "source_freeze_sha256": review["source_freeze_sha256"],
        "protocol_sha256": review["protocol_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
