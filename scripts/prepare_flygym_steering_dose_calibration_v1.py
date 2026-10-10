#!/usr/bin/env python3
"""Prepare a frozen, non-executing FlyGym AP dose-calibration bundle.

This script uses only the Python standard library. It verifies the existing
mechanics source and asset hashes, snapshots small configuration/provenance
files, and writes a proposed protocol. It never imports FlyGym, MuJoCo, NumPy,
or a neural-network module, and it cannot execute a body simulation.

Prepare the original bundle or the revision-3 runtime-instrumented bundle with:

    python scripts/prepare_flygym_steering_dose_calibration_v1.py --prepare
    python scripts/prepare_flygym_steering_dose_calibration_v1.py --prepare --revision 3
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
MECHANICS_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "standardized-repair-20261007-v1/mechanics"
)
OLD_V41_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "flygym-phase-speed-repair-20261006-v1/development/v4-1/weak_0p30"
)
OUTPUT_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1/mechanical-steering-dose-calibration-v1"
)
REVISION_2_ROOT = OUTPUT_ROOT / "revision-2"
REVISION_3_ROOT = OUTPUT_ROOT / "revision-3"
REVISION_3_SOURCE_SNAPSHOT_STAGE = OUTPUT_ROOT / ".revision-3-source-snapshot-staging"

DOSE_VALUES = (0.0, 0.2, -0.2, 0.4, -0.4)
SEEDS = (1729, 1730, 1731)
POWER = 0.258
DURATION_S = 7.0
COMMON_PREFIX_S = 1.0
RAMP_S = 0.25
ANALYSIS_WINDOW_S = (1.25, 7.0)
CASE_TIMEOUT_S = 600
BATCH_TIMEOUT_S = 9000


class PreparationError(RuntimeError):
    """Raised when a pinned input is absent, changed, or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


def copy_pinned_input(src: Path, dst: Path) -> dict[str, Any]:
    if not src.is_file():
        raise PreparationError(f"required input is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {
        "source_path": str(src),
        "snapshot_path": str(dst),
        "sha256": sha256_file(dst),
        "size_bytes": dst.stat().st_size,
    }


def case_id(seed: int, d: float) -> str:
    dose = "0p00" if d == 0 else f"{abs(d):.2f}".replace(".", "p")
    sign = "zero" if d == 0 else ("plus" if d > 0 else "minus")
    return f"seed{seed}_d{sign}_{dose}"


def protocol_payload() -> dict[str, Any]:
    cases = [
        {"case_id": case_id(seed, d), "seed": seed, "d": d}
        for seed in SEEDS
        for d in DOSE_VALUES
    ]
    return {
        "schema_version": "flygym-steering-dose-calibration-preparation-v1",
        "status": "prepared_not_executed",
        "classification": (
            "CPU-only engineered position-servo/CPG AP-dose calibration; "
            "P and d are externally specified; no neural forward or visual input"
        ),
        "design": {
            "power_P_left_right": [POWER, POWER],
            "signed_d_values": list(DOSE_VALUES),
            "seeds": list(SEEDS),
            "case_count": len(cases),
            "duration_s": DURATION_S,
            "common_prefix_s": COMMON_PREFIX_S,
            "ramp_duration_s": RAMP_S,
            "analysis_window_s": list(ANALYSIS_WINDOW_S),
            "dose_schedule": {
                "zero_dose": "d=0 for the full 7.0 s",
                "nonzero_dose": (
                    "d=0 through t=1.0 s; quintic smoothstep ramp to target "
                    "over [1.0,1.25] s; hold target through t=7.0 s"
                ),
                "ramp_function": "S(u)=6u^5-15u^4+10u^3; u=clip((t-1.0)/0.25,0,1)",
                "P_schedule": "constant symmetric P=(0.258,0.258) for the full 7.0 s",
            },
            "same_seed_pairing": (
                "For each seed, instantiate a fresh session for every d condition. "
                "Require equality of initial qpos/qvel hashes and equal CPG phase "
                "prefix through t=1.0 s. Initial states may differ between seeds."
            ),
            "template": (
                "Reuse the hash-frozen FlyGym V4.1/AP phase-speed controller and "
                "the existing flat-ground free-body model/configuration; no gait, "
                "actuator, clearance, adhesion, or endpoint-scale edits."
            ),
            "dose_mapping": {
                "both_AP_channels": "dna02_command=dng13_command=abs(d)",
                "signed_side_rule": "d>=0 selects left soma side; d<0 selects right soma side",
                "predeclared_endpoint_scale_mm": 0.05,
                "nominal_command_mm_by_abs_d": {
                    "0.00": 0.0,
                    "0.20": 0.01,
                    "0.40": 0.02,
                },
                "meaning": "nominal target offset, not realized foot displacement, force, torque, or yaw",
            },
            "cases": cases,
        },
        "execution_scope": {
            "body_simulation_run_during_preparation": False,
            "gpu": False,
            "neural_forward": False,
            "visual_input": False,
            "fixed_P_d_experiment_is_NN_driven": False,
            "root_qpos_qvel_wrench_intervention": False,
            "qfrc_or_xfrc_application": False,
            "post_initialization_root_state_write": False,
            "biological_claim": False,
            "biological_pass_threshold": None,
        },
        "planned_capture": {
            "per_control_tick": [
                "P_left_right and scheduled signed d",
                "six-leg CPG phases",
                "per-leg requested and solved AP endpoint target terms",
            ],
            "native_trajectory": [
                "thorax heading yaw and angular velocity",
                "thorax/world position, COM position, and velocity",
                "initial-heading-frame forward and lateral displacement (derived offline)",
                "per-leg tarsus positions, contact force/normal, contact point, and contact events",
                "thorax and non-leg ground-contact indicators",
                "qpos/qvel and applied-force audit arrays",
            ],
            "required_pairing_checks": [
                "same-seed initial qpos and qvel hashes equal across five doses",
                "same-seed CPG phase prefixes equal through the shared 1.0 s prefix",
                "P trace identical across all 15 cases",
                "d command trace equals the registered schedule",
                "root qpos/qvel post-initialization write counts remain zero",
                "root/body wrench and generalized applied-force traces remain zero",
                "all pinned source, configuration, model, and asset hashes match",
            ],
        },
        "resource_plan": {
            "cpu_quota": "200% maximum (2 CPU cores)",
            "thread_env": "OMP/BLAS thread counts pinned to 1",
            "memory_max_bytes": 2147483648,
            "per_case_timeout_s": CASE_TIMEOUT_S,
            "whole_batch_timeout_s": BATCH_TIMEOUT_S,
            "status": "proposed limits for a future execution, not attested or run",
        },
        "interpretation_limits": [
            "This calibrates an engineered endpoint-position command in a FlyGym mechanics model.",
            "P is a fixed direct CPG drive; the experiment is not NN-driven and does not validate neural decoding.",
            "A requested endpoint offset is not a measured contact displacement or force.",
            "Three seeds are a small engineering cohort, not a biological acceptance criterion.",
            "Preserve raw yaw oscillations and report seed-wise dose contrasts; do not fit a turn gain from this preparation alone.",
        ],
        "missing_execution_functionality": [
            "The existing CLI only permits duration=3.0 s, seed=1729, P=0.30 and d=0/+0.20/-0.20.",
            "Its _run_condition helper accepts one constant d per run; it has no registered per-tick d schedule or shared-prefix/ramp execution path.",
            "Its summary window is fixed to [0.5,3.0] s and does not produce the requested 7 s dose-window summary or initial-heading-frame forward displacement.",
            "A separate execution runner or reviewed new entrypoint must construct the same existing controller/session, apply only P and AP command inputs, save native/control telemetry, and enforce cgroup limits. The old runner is retained unchanged.",
        ],
    }


def prepare() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        if any(OUTPUT_ROOT.iterdir()):
            raise PreparationError(f"refusing to overwrite non-empty output: {OUTPUT_ROOT}")
        OUTPUT_ROOT.rmdir()
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mechanical-dose-prep-", dir=OUTPUT_ROOT.parent))
    try:
        prior_freeze_path = MECHANICS_ROOT / "source_freeze.json"
        prior_freeze = read_json(prior_freeze_path)
        if prior_freeze.get("status") != "frozen_before_capture":
            raise PreparationError("prior mechanics source freeze is not in frozen_before_capture status")
        prior_freeze_hash = sha256_file(prior_freeze_path)

        source_records: dict[str, dict[str, Any]] = {}
        mismatches: list[str] = []
        for raw_path, pinned in prior_freeze["files"].items():
            path = Path(raw_path)
            if not path.is_file():
                mismatches.append(f"missing source: {path}")
                continue
            observed_hash = sha256_file(path)
            observed_size = path.stat().st_size
            if observed_hash != pinned["sha256"] or observed_size != pinned["size_bytes"]:
                mismatches.append(f"source differs from prior freeze: {path}")
            source_records[str(path)] = {
                "sha256": observed_hash,
                "size_bytes": observed_size,
                "prior_freeze_sha256": pinned["sha256"],
                "verified_against_prior_freeze": observed_hash == pinned["sha256"],
            }
        if mismatches:
            raise PreparationError("; ".join(mismatches))

        asset_manifest_path = MECHANICS_ROOT / "mechanics_cases/ap_dplus_0p20/upstream_asset_hashes.json"
        asset_manifest = read_json(asset_manifest_path)
        asset_records: dict[str, dict[str, Any]] = {}
        asset_bytes = 0
        for logical_name, record in asset_manifest.items():
            asset_path = Path(record.get("physical_path", record["path"]))
            if not asset_path.is_file():
                raise PreparationError(f"pinned FlyGym asset is missing: {asset_path}")
            observed = sha256_file(asset_path)
            if observed != record["sha256"]:
                raise PreparationError(f"pinned FlyGym asset hash changed: {asset_path}")
            size = asset_path.stat().st_size
            asset_bytes += size
            asset_records[logical_name] = {
                "physical_path": str(asset_path),
                "sha256": observed,
                "size_bytes": size,
                "verified_against_prior_capture_manifest": True,
            }

        screen_summary_path = MECHANICS_ROOT / "analysis/body-yaw-sign-calibration.json"
        screen_summary = read_json(screen_summary_path)
        if screen_summary.get("acceptance_status") != "pending_three_seed_pairing":
            raise PreparationError("prior paired screen status changed; review before preparing this dose protocol")
        if screen_summary.get("cases") != ["static_d0", "ap_dplus_0p20", "ap_dminus_0p20"]:
            raise PreparationError("prior paired screen case list differs from the registered three-case screen")
        case_references: dict[str, Any] = {}
        for case_name in screen_summary["cases"]:
            case_dir = MECHANICS_ROOT / "mechanics_cases" / case_name
            paths = {
                name: case_dir / name
                for name in (
                    "run-record.json", "effective_config.json", "manifest.json",
                    "trajectory.npz", "contacts.npz", "controller_telemetry.jsonl",
                )
            }
            entries = {}
            for logical_name, path in paths.items():
                if not path.is_file():
                    raise PreparationError(f"paired-screen evidence is missing: {path}")
                entries[logical_name] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            run_record = read_json(paths["run-record.json"])
            if run_record.get("status") != "complete":
                raise PreparationError(f"paired-screen case is not complete: {case_name}")
            case_references[case_name] = entries

        input_specs = {
            "prior_source_freeze.json": prior_freeze_path,
            "prior_mechanics_protocol.json": MECHANICS_ROOT / "protocol.json",
            "prior_paired_screen_summary.json": screen_summary_path,
            "upstream_asset_hashes.json": asset_manifest_path,
            "prior_model_options.json": MECHANICS_ROOT / "mechanics_cases/ap_dplus_0p20/model_options.json",
            "prior_array_order.json": MECHANICS_ROOT / "mechanics_cases/ap_dplus_0p20/array_order.json",
            "prior_case_effective_config.json": MECHANICS_ROOT / "mechanics_cases/ap_dplus_0p20/effective_config.json",
            "v4_1_controller_config.json": OLD_V41_ROOT / "controller_config.json",
            "v4_1_effective_config.json": OLD_V41_ROOT / "effective_config.json",
            "cartesian_repair_protocol.json": PROJECT / "config/flygym_cartesian_repair_protocol.json",
        }
        copied_inputs: dict[str, Any] = {}
        for filename, source in input_specs.items():
            source_path = source.resolve(strict=True)
            destination = stage / "frozen_inputs" / filename
            copied_inputs[filename] = copy_pinned_input(source_path, destination)
            copied_inputs[filename]["snapshot_path"] = str(
                OUTPUT_ROOT / "frozen_inputs" / filename
            )

        protocol = protocol_payload()
        protocol["preparation_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        protocol["source_parent"] = {
            "mechanics_root": str(MECHANICS_ROOT),
            "source_freeze_path": str(prior_freeze_path),
            "source_freeze_sha256": prior_freeze_hash,
            "prior_mechanics_protocol_sha256": prior_freeze["protocol_sha256"],
        }

        prep_script = Path(__file__).resolve(strict=True)
        source_records[str(prep_script)] = {
            "sha256": sha256_file(prep_script),
            "size_bytes": prep_script.stat().st_size,
            "role": "non-executing preparation and provenance script",
        }
        source_freeze = {
            "schema_version": "flygym-steering-dose-calibration-source-freeze-v1",
            "status": "preparation_frozen_not_execution_authorized",
            "prior_source_freeze": {
                "path": str(prior_freeze_path),
                "sha256": prior_freeze_hash,
                "all_files_reverified": True,
            },
            "source_files": source_records,
            "asset_manifest": {
                "path": str(asset_manifest_path),
                "sha256": sha256_file(asset_manifest_path),
                "asset_count": len(asset_records),
                "verified_asset_count": len(asset_records),
                "verified_total_bytes": asset_bytes,
                "files": asset_records,
            },
            "copied_inputs": copied_inputs,
            "raw_asset_binaries_copied": False,
            "future_execution_must_reverify_all_hashes": True,
        }

        paired_reference = {
            "schema_version": "flygym-prior-paired-screen-reference-v1",
            "source_summary_path": str(screen_summary_path),
            "source_summary_sha256": sha256_file(screen_summary_path),
            "summary": screen_summary,
            "raw_case_files": case_references,
            "readout": {
                "static_d0_yaw_change_deg_0p5_3s": screen_summary["native_thorax_yaw_deg_by_case"]["static_d0"],
                "d_plus_0p2_yaw_change_deg_0p5_3s": screen_summary["native_thorax_yaw_deg_by_case"]["ap_dplus_0p20"],
                "d_minus_0p2_yaw_change_deg_0p5_3s": screen_summary["native_thorax_yaw_deg_by_case"]["ap_dminus_0p20"],
                "d_plus_0p2_minus_static_deg": screen_summary["positive_d_minus_static_yaw_deg"],
                "d_minus_0p2_minus_static_deg": screen_summary["negative_d_minus_static_yaw_deg"],
                "paired_odd_deg": screen_summary["paired_odd_yaw_deg"],
                "acceptance_status": screen_summary["acceptance_status"],
            },
            "limits": [
                "One seed (1729), P=0.30, 3 s, and only d=0/+/-0.20; this is context, not a completed three-seed dose curve.",
                "The requested new cohort uses P=0.258 and 7 s, so its raw yaw is not directly interchangeable with the old summary window.",
                "The old case records and raw captures remain at their original paths and are not modified.",
            ],
        }

        dump_json(stage / "protocol.json", protocol)
        source_freeze["protocol_sha256"] = sha256_file(stage / "protocol.json")
        dump_json(stage / "source_freeze.json", source_freeze)
        dump_json(stage / "paired-screen-reference.json", paired_reference)

        report = {
            "schema_version": "flygym-steering-dose-calibration-preparation-review-v1",
            "status": "prepared_not_executed",
            "conclusion": "Preparation is complete; no simulation or test was run.",
            "direct_reuse": [
                "The frozen V4.1/AP controller/session produces free-body mechanics without neural forward or visual input.",
                "Existing captures include native yaw/body position/velocity, tarsus and contact traces, applied-force audits, CPG phase, and AP target telemetry.",
                "The old paired screen verifies matched initial qpos/qvel, CPG phase trace, symmetric drive, model signature, and mirrored AP targets for seed 1729.",
            ],
            "blocking_execution_gaps": protocol_payload()["missing_execution_functionality"],
            "scope_boundary": "This bundle does not authorize a run and contains no root approval artifact.",
            "source_freeze_sha256": sha256_file(stage / "source_freeze.json"),
            "protocol_sha256": sha256_file(stage / "protocol.json"),
            "paired_screen_reference_sha256": sha256_file(stage / "paired-screen-reference.json"),
            "asset_count_verified": len(asset_records),
            "copied_input_count": len(copied_inputs),
        }
        dump_json(stage / "preparation-review.json", report)
        md = [
            "# Mechanical steering dose calibration: preparation review",
            "",
            "**Status: prepared, not executed.** This bundle contains no root approval artifact. No body, NN, GPU, or tests were run.",
            "",
            "## Protocol prepared",
            "",
            "The proposed cohort is 15 fresh-session cases: seeds 1729, 1730, and 1731 crossed with d=0, +0.20, -0.20, +0.40, and -0.40. All runs use constant symmetric P=(0.258, 0.258) for 7 seconds. Nonzero d conditions share a 1-second d=0 prefix, a 250 ms quintic ramp, then hold their dose; the d=0 condition remains at zero. Pairing checks compare initial qpos/qvel and CPG phase prefixes within each seed.",
            "",
            "The existing 0.05 mm AP endpoint scale is retained. Thus |d|=0.20 requests 0.010 mm and |d|=0.40 requests 0.020 mm per configured command mapping. These are target offsets, not realized foot motion or force. P and d are direct engineering inputs, so this is not NN-driven and has no biological pass threshold.",
            "",
            "## Reuse and execution gap",
            "",
            "The existing runner uses the intended V4.1/AP mechanics, saves phase and contact telemetry, and audited zero post-initialization root-state writes and zero applied wrench/force. Its CLI remains pinned to 3 seconds, seed 1729, P=0.30, and d=0/±0.20. Its per-case helper holds one constant d and summarizes only [0.5, 3.0] seconds. It cannot directly execute this protocol.",
            "",
            "A future execution entrypoint must add the 7-second/three-seed/five-dose schedule and save forward displacement alongside yaw, CPG phase, solved/requested AP targets, contacts, and authority arrays. It must preserve the old runner and verify the source/asset freeze before starting. Planned limits are 2 CPU cores, 2 GiB memory, 600 seconds per case, and a 9000-second batch cap; none is attested in this preparation.",
            "",
            "## Previous paired screen",
            "",
            f"The existing one-seed P=0.30, 3-second screen reported d=0 yaw {screen_summary['native_thorax_yaw_deg_by_case']['static_d0']:.4f}°, d=+0.20 yaw {screen_summary['native_thorax_yaw_deg_by_case']['ap_dplus_0p20']:.4f}°, and d=-0.20 yaw {screen_summary['native_thorax_yaw_deg_by_case']['ap_dminus_0p20']:.4f}° over [0.5, 3.0] s. The mirrored contrasts were +{screen_summary['positive_d_minus_static_yaw_deg']:.4f}° and {screen_summary['negative_d_minus_static_yaw_deg']:.4f}°, with status `{screen_summary['acceptance_status']}`. The new bundle keeps these as a reference; it does not upgrade the prior one-seed screen.",
            "",
            "## Bundle files",
            "",
            "- `protocol.json`: proposed dose schedule and capture requirements.",
            "- `source_freeze.json`: verified source, configuration, model, and asset hashes.",
            "- `paired-screen-reference.json`: prior paired result and raw-file hashes.",
            "- `frozen_inputs/`: snapshots of small protocol/config/provenance JSON files.",
            "- `preparation-review.json`: machine-readable review status.",
            "",
        ]
        (stage / "preparation-review.md").write_text("\n".join(md), encoding="utf-8")
        stage.rename(OUTPUT_ROOT)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare_revision3() -> dict[str, Any]:
    """Freeze the instrumented runtime entrypoint without authorizing execution."""
    if REVISION_3_ROOT.exists():
        raise PreparationError(f"refusing to overwrite revision 3: {REVISION_3_ROOT}")
    if not REVISION_3_SOURCE_SNAPSHOT_STAGE.is_dir():
        raise PreparationError("revision-2 runner/preparer source snapshots must be staged before edits")
    r2_freeze_path = REVISION_2_ROOT / "source_freeze.json"
    r2_protocol_path = REVISION_2_ROOT / "protocol.json"
    r2_review_path = REVISION_2_ROOT / "preparation-review.json"
    r2_freeze = read_json(r2_freeze_path)
    r2_protocol = read_json(r2_protocol_path)
    r2_review = read_json(r2_review_path)
    if r2_freeze.get("status") != "preparation_frozen_execution_entrypoint_ready_not_run":
        raise PreparationError("revision-2 source freeze status is unexpected")
    if r2_review.get("status") != "prepared_not_executed":
        raise PreparationError("revision-2 preparation review status is unexpected")
    if r2_freeze.get("protocol_sha256") != sha256_file(r2_protocol_path):
        raise PreparationError("revision-2 protocol no longer matches its source freeze")

    runner_path = (PROJECT / "scripts/run_flygym_steering_dose_calibration_v1.py").resolve(strict=True)
    prep_path = Path(__file__).resolve(strict=True)
    source_records: dict[str, dict[str, Any]] = {}
    for raw_path, prior in r2_freeze["source_files"].items():
        path = Path(raw_path)
        if not path.is_file():
            raise PreparationError(f"revision-2 source disappeared: {path}")
        observed = sha256_file(path)
        if path.resolve() in (runner_path, prep_path):
            staged = REVISION_3_SOURCE_SNAPSHOT_STAGE / path.name
            if not staged.is_file() or sha256_file(staged) != prior["sha256"]:
                raise PreparationError(f"revision-2 source snapshot is missing or changed: {staged}")
            source_records[str(path)] = {
                "sha256": observed,
                "size_bytes": path.stat().st_size,
                "prior_revision_2_sha256": prior["sha256"],
                "revision_3_instrumentation_change": True,
            }
        else:
            if observed != prior["sha256"] or path.stat().st_size != prior["size_bytes"]:
                raise PreparationError(f"unintended source drift from revision 2: {path}")
            source_records[str(path)] = {
                "sha256": observed,
                "size_bytes": path.stat().st_size,
                "verified_unchanged_from_revision_2": True,
            }
    runner_sha = source_records[str(runner_path)]["sha256"]

    stage = Path(tempfile.mkdtemp(prefix=".mechanical-dose-rev3-", dir=OUTPUT_ROOT.parent))
    try:
        copied_inputs: dict[str, Any] = {}
        for filename, prior in r2_freeze["copied_inputs"].items():
            source = Path(prior["snapshot_path"])
            if not source.is_file() or sha256_file(source) != prior["sha256"]:
                raise PreparationError(f"revision-2 frozen input changed: {source}")
            destination = stage / "frozen_inputs" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied_inputs[filename] = {
                "source_path": str(source),
                "snapshot_path": str(REVISION_3_ROOT / "frozen_inputs" / filename),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "verified_unchanged_from_revision_2": True,
            }

        asset_manifest = r2_freeze["asset_manifest"]
        for logical_name, asset in asset_manifest["files"].items():
            asset_path = Path(asset["physical_path"])
            if (not asset_path.is_file() or sha256_file(asset_path) != asset["sha256"]
                    or asset_path.stat().st_size != asset["size_bytes"]):
                raise PreparationError(f"revision-2 model asset changed: {logical_name} ({asset_path})")

        bundle_files: dict[str, Any] = {}
        paired_source = REVISION_2_ROOT / "paired-screen-reference.json"
        paired_prior = r2_freeze["bundle_files"]["paired-screen-reference.json"]
        if sha256_file(paired_source) != paired_prior["sha256"]:
            raise PreparationError("revision-2 paired-screen reference changed")
        shutil.copyfile(paired_source, stage / "paired-screen-reference.json")
        bundle_files["paired-screen-reference.json"] = {
            "sha256": sha256_file(stage / "paired-screen-reference.json"),
            "size_bytes": (stage / "paired-screen-reference.json").stat().st_size,
        }
        for filename in (runner_path.name, prep_path.name):
            source = REVISION_3_SOURCE_SNAPSHOT_STAGE / filename
            relative = Path("source_snapshots/revision-2") / filename
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            bundle_files[relative.as_posix()] = {
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "role": "immutable revision-2 source snapshot; current source has a separately pinned revision-3 hash",
            }

        protocol = dict(r2_protocol)
        protocol["execution"] = dict(r2_protocol["execution"])
        protocol.update({
            "schema_version": "flygym-steering-dose-calibration-protocol-v3",
            "revision": 3,
            "entrypoint_source_sha256": runner_sha,
            "preparation_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_parent": {
                "revision_2_root": str(REVISION_2_ROOT),
                "revision_2_protocol_sha256": sha256_file(r2_protocol_path),
                "revision_2_source_freeze_sha256": sha256_file(r2_freeze_path),
                "revision_2_preparation_review_sha256": sha256_file(r2_review_path),
            },
            "instrumentation": {
                "preconstruction": [
                    "sys.executable, sys.version, sys.path, PYTHONPATH, and PYTHONHASHSEED",
                    "loaded key module origins and package versions",
                    "exact WalkConfig, configuration hash, OSMesa/cache environment and loaded FlyGym assets root",
                ],
                "immediately_after_constructor_before_controller_or_hash_gate": [
                    "raw compiled_model.xml bytes and SHA-256",
                    "effective model options and hash",
                    "FlyGym upstream source hashes and post-constructor loaded module origins",
                    "constructor time/control/native-sample counters",
                ],
                "model_gate": "preserve and enforce the original paired-screen compiled model hash bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57; mismatch fails before stepping",
                "model_or_controller_changes": False,
            },
        })
        protocol["execution"]["artifact_policy"] = (
            "save preconstruction runtime/config provenance; immediately after constructor save raw compiled MJCF, hash, model options, and upstream source hashes before the unchanged model-hash gate; then preserve native trajectory, contacts, per-tick phase/AP telemetry, cgroup memory.peak, and artifact SHA-256"
        )
        dump_json(stage / "protocol.json", protocol)

        source_freeze = {
            "schema_version": "flygym-steering-dose-calibration-source-freeze-v3",
            "status": "preparation_frozen_execution_entrypoint_ready_not_run",
            "revision": 3,
            "prior_preparation_bundle": {
                "path": str(REVISION_2_ROOT),
                "protocol_sha256": sha256_file(r2_protocol_path),
                "source_freeze_sha256": sha256_file(r2_freeze_path),
                "preparation_review_sha256": sha256_file(r2_review_path),
                "prior_root_execution_review_sha256": sha256_file(REVISION_2_ROOT / "root-execution-review.json"),
                "prior_failed_run_record_sha256": sha256_file(REVISION_2_ROOT / "runs/seed1729_dzero_0p00/run-record.json"),
                "prior_cohort_terminal_sha256": sha256_file(REVISION_2_ROOT / "runs/cohort-manifest.json"),
            },
            "runner_sha256": runner_sha,
            "source_files": source_records,
            "asset_manifest": asset_manifest,
            "copied_inputs": copied_inputs,
            "bundle_files": bundle_files,
            "raw_asset_binaries_copied": False,
            "future_execution_must_reverify_all_source_asset_and_input_hashes": True,
        }
        source_freeze["protocol_sha256"] = sha256_file(stage / "protocol.json")
        dump_json(stage / "source_freeze.json", source_freeze)

        paired = read_json(stage / "paired-screen-reference.json")
        execution_without_artifact_policy = {
            key: value for key, value in protocol["execution"].items() if key != "artifact_policy"
        }
        prior_execution_without_artifact_policy = {
            key: value for key, value in r2_protocol["execution"].items() if key != "artifact_policy"
        }
        review = {
            "schema_version": "flygym-steering-dose-calibration-preparation-review-v3",
            "status": "prepared_not_executed",
            "conclusion": "Revision-3 runtime-provenance instrumentation is frozen; no body, NN, GPU, or tests were run and this bundle contains no execution approval.",
            "entrypoint_path": str(runner_path),
            "runner_sha256": runner_sha,
            "protocol_sha256": sha256_file(stage / "protocol.json"),
            "source_freeze_sha256": sha256_file(stage / "source_freeze.json"),
            "paired_screen_reference_sha256": sha256_file(stage / "paired-screen-reference.json"),
            "revision_2_bundle_files_snapshotted_or_referenced": list(bundle_files),
            "ordered_case_count": len(protocol["execution"]["ordered_cases"]),
            "frozen_asset_count": asset_manifest["asset_count"],
            "dose_design_unchanged_from_revision_2": execution_without_artifact_policy == prior_execution_without_artifact_policy,
            "strict_model_gate_sha256": "bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57",
            "instrumentation": protocol["instrumentation"],
            "runtime_gates": [
                "Existing fixed-pose terminal-state, shared-lock, cgroup CPU/memory, 15-case order, timeout, and pairing gates are unchanged.",
                "The raw constructor MJCF and runtime provenance are written before the unchanged compiled-model hash gate.",
                "A model-hash mismatch remains a failure and stops the cohort before any control tick.",
                "Revision 2 approval and failed run remain at their original paths; revision 3 requires a separate review/decision.",
            ],
            "not_verified_until_execution": [
                "Runtime module paths/versions on the next actual launch.",
                "Captured raw XML hash on the next actual constructor.",
                "All 15 cases, same-seed pairing, authority, contacts, cgroup peak, and time/resource limits.",
            ],
            "scope_boundary": "Preparation only; no root approval artifact or run output was created.",
            "paired_screen_status": paired.get("summary", {}).get("acceptance_status"),
        }
        dump_json(stage / "preparation-review.json", review)
        md = [
            "# Mechanical steering dose calibration: revision 3 preparation",
            "",
            "**Status: prepared, not executed.** This revision adds runtime/module provenance and immediate constructor MJCF capture. It does not change the 15-case design, mechanics, controller, dose mapping, seed order, P/d schedule, resource caps, or the strict model-hash gate.",
            "",
            "## Initialization evidence",
            "",
            "Before construction, the runner saves the actual interpreter, Python path/hash seed, loaded module origins and versions, OSMesa/cache environment, exact WalkConfig, and configuration hash. Immediately after construction, it saves the exact raw MJCF and SHA-256, effective model options, FlyGym upstream source hashes, and constructor counters before the existing baseline-hash comparison.",
            "",
            "The gate remains pinned to `bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57`. Any mismatch remains a failure before stepping. No baseline replacement or model/controller change is included.",
            "",
            "## Preserved revision 2",
            "",
            "Revision 2 approval, failed capture, and outputs remain untouched. Revision-2 runner and preparation-script bytes are snapshotted under `source_snapshots/revision-2/` and their hashes are included in this bundle's source freeze.",
            "",
            "## Status",
            "",
            "The protocol retains 15 cases, 7 s, seeds 1729/1730/1731, P=(0.258,0.258), d=0/±0.2/±0.4, the 1 s common prefix, 250 ms ramp, 2-core/2-GiB cgroup caps, and 600/9000 s timeouts. No body, NN, GPU, tests, or approval artifacts were created.",
            "",
        ]
        (stage / "preparation-review.md").write_text("\n".join(md), encoding="utf-8")
        stage.rename(REVISION_3_ROOT)
        shutil.rmtree(REVISION_3_SOURCE_SNAPSHOT_STAGE)
        return review
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True,
                        help="verify and write the non-executing preparation bundle")
    parser.add_argument("--revision", type=int, choices=(1, 3), default=1,
                        help="prepare the original bundle (1) or the separate instrumented runtime bundle (3)")
    args = parser.parse_args(argv)
    if not args.prepare:
        raise SystemExit("only --prepare is supported; this script cannot run a simulation")
    try:
        report = prepare_revision3() if args.revision == 3 else prepare()
    except (OSError, KeyError, ValueError, PreparationError) as exc:
        raise SystemExit(f"preparation failed: {exc}") from exc
    print(json.dumps({"output_root": str(OUTPUT_ROOT), **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
