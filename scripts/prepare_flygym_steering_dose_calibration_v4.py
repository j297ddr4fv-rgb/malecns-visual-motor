#!/usr/bin/env python3
"""Prepare the frozen revision-4 FlyGym steering-dose bundle.

This standard-library-only preparer snapshots the revision-3 inputs and the
already completed, zero-step compiled-model identity forensics. It verifies the
15-case design and every source/asset pin, but does not import FlyGym or MuJoCo,
compile a model, run a simulation, or create an execution-approval artifact.

Prepare once with:

    python scripts/prepare_flygym_steering_dose_calibration_v4.py --prepare
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
BASE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/"
    "feedback-diagnosis-20261007-v1"
)
OUTPUT_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-4"
REVISION_3_ROOT = BASE / "mechanical-steering-dose-calibration-v1/revision-3"
FORENSICS_ROOT = REVISION_3_ROOT / "compiled-identity-forensics-v1"
REFERENCE_XML_PATH = BASE / "mechanical-model-initialization-audit-v1/compiled_model.xml"
REVISION_3_XML_PATH = REVISION_3_ROOT / "runs/seed1729_dzero_0p00/compiled_model.xml"
ANALYSIS_PATH = FORENSICS_ROOT / "analysis.json"
REFERENCE_MJB_PATH = FORENSICS_ROOT / "reference.mjb"
ACTUAL_MJB_PATH = FORENSICS_ROOT / "actual.mjb"
ANALYSIS_TOOL_PATH = PROJECT / "scripts/analyze_flygym_compiled_model_identity_v1.py"

EXPECTED_REFERENCE_XML_SHA256 = "bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57"
EXPECTED_REVISION_3_XML_SHA256 = "976e72a119b25b591e8b88aac63c2a5f06913013ec787330ae727f0f8be432fd"
EXPECTED_MJB_SHA256 = "c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f"
EXPECTED_MJB_SIZE_BYTES = 23230821
EXPECTED_MODEL_ARRAY_COUNT = 468
FIXED_POSE_QUEUE_BUNDLE = BASE / "spatial-fixed-pose-calibration-v1-revision-2"
FIXED_POSE_QUEUE_STATUS_PATH = FIXED_POSE_QUEUE_BUNDLE / "queue-status.json"
FIXED_POSE_QUEUE_LOCK_PATH = FIXED_POSE_QUEUE_BUNDLE / "queue.lock"


class PreparationError(RuntimeError):
    """Raised when a pinned input is absent, changed, or inconsistent."""


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
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }


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
    return {
        "source_path": str(src),
        "snapshot_path": str(OUTPUT_ROOT / dst.relative_to(_STAGE_ROOT)),
        "sha256": copied,
        "size_bytes": dst.stat().st_size,
    }


_STAGE_ROOT = Path(".")


def _mesh_decl(line: str) -> tuple[str, str, bool] | None:
    match = re.search(r'<mesh\s+name="([^"]+)"([^>]*)/>', line)
    if match is None:
        return None
    name, attrs = match.groups()
    file_match = re.search(r'\bfile="([^"]+)"', attrs)
    if file_match is None:
        return None
    return name, file_match.group(1), bool(re.search(r'\bcontent_type="model/stl"', attrs))


def inspect_xml_serialization_difference(reference: Path, actual: Path) -> dict[str, Any]:
    """Prove the pinned XML pair differs only in two mirrored content-type flags."""
    reference_text = reference.read_text(encoding="utf-8")
    actual_text = actual.read_text(encoding="utf-8")
    reference_lines = reference_text.splitlines()
    actual_lines = actual_text.splitlines()
    if len(reference_lines) != len(actual_lines):
        raise PreparationError("reference and revision-3 XML line counts differ")
    remove_flag = re.compile(r'\s+content_type="model/stl"')
    if [remove_flag.sub("", line) for line in reference_lines] != [
        remove_flag.sub("", line) for line in actual_lines
    ]:
        raise PreparationError(
            "the retained XML pair has differences beyond content_type='model/stl' placement"
        )

    changed_rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(reference_lines, actual_lines), start=1):
        if left == right:
            continue
        old_decl = _mesh_decl(left)
        new_decl = _mesh_decl(right)
        if old_decl is None or new_decl is None or old_decl[:2] != new_decl[:2]:
            raise PreparationError(f"non-mesh or different-mesh XML change at line {index}")
        changed_rows.append({
            "line": index,
            "mesh_name": old_decl[0],
            "file": old_decl[1],
            "reference_has_content_type_model_stl": old_decl[2],
            "revision_3_has_content_type_model_stl": new_decl[2],
            "normalized_declaration_identical": remove_flag.sub("", left)
            == remove_flag.sub("", right),
        })
    expected_names = {
        "nmf/l_eye", "nmf/r_eye", "nmf/l_haltere", "nmf/r_haltere",
    }
    if {row["mesh_name"] for row in changed_rows} != expected_names or len(changed_rows) != 4:
        raise PreparationError("the XML diff is not the expected two left/right mesh-attribute swaps")
    for left_name, right_name in (("nmf/l_eye", "nmf/r_eye"), ("nmf/l_haltere", "nmf/r_haltere")):
        left = next(row for row in changed_rows if row["mesh_name"] == left_name)
        right = next(row for row in changed_rows if row["mesh_name"] == right_name)
        if left["file"] != right["file"]:
            raise PreparationError(f"mirrored meshes do not share the same source file: {left_name}/{right_name}")
        if left["reference_has_content_type_model_stl"] == left["revision_3_has_content_type_model_stl"]:
            raise PreparationError(f"left mesh content_type did not move: {left_name}")
        if right["reference_has_content_type_model_stl"] == right["revision_3_has_content_type_model_stl"]:
            raise PreparationError(f"right mesh content_type did not move: {right_name}")
        if left["reference_has_content_type_model_stl"] == right["reference_has_content_type_model_stl"]:
            raise PreparationError(f"reference pair does not have one explicit content_type: {left_name}/{right_name}")
        if left["revision_3_has_content_type_model_stl"] == right["revision_3_has_content_type_model_stl"]:
            raise PreparationError(f"revision-3 pair does not have one explicit content_type: {left_name}/{right_name}")

    return {
        "schema_version": "compiled-xml-serialization-difference-review-v1",
        "status": "verified",
        "reference_xml_path": str(reference),
        "reference_xml_sha256": sha256_file(reference),
        "revision_3_xml_path": str(actual),
        "revision_3_xml_sha256": sha256_file(actual),
        "line_count_reference": len(reference_lines),
        "line_count_revision_3": len(actual_lines),
        "changed_mesh_declaration_count": len(changed_rows),
        "all_xml_text_equal_after_removing_only_explicit_model_stl_attribute": True,
        "changed_mesh_declarations": changed_rows,
        "interpretation": (
            "The raw MJCF serialization differs only because the explicit content_type='model/stl' "
            "attribute is assigned to the opposite member of each same-file left/right mesh pair. "
            "This is a serialization difference for this retained XML pair; runtime admission still "
            "requires exact complete saved-MJB SHA-256 identity."
        ),
    }


def _validate_forensics(analysis: dict[str, Any]) -> None:
    if analysis.get("schema_version") != "compiled-model-byte-identity-forensics-v1":
        raise PreparationError("compiled-model forensic schema is unexpected")
    if analysis.get("status") != "complete":
        raise PreparationError("compiled-model forensic analysis is not complete")
    expected_inputs = {
        str(REFERENCE_XML_PATH): EXPECTED_REFERENCE_XML_SHA256,
        str(REVISION_3_XML_PATH): EXPECTED_REVISION_3_XML_SHA256,
    }
    if analysis.get("inputs") != expected_inputs:
        raise PreparationError("forensic analysis does not identify the frozen XML pair")
    if analysis.get("integration_steps") != 0 or analysis.get("body_control_ticks") != 0:
        raise PreparationError("forensic analysis was not a zero-step inspection")
    if analysis.get("gpu_used") is not False or analysis.get("existing_gate_modified") is not False:
        raise PreparationError("forensic analysis used GPU or modified the old gate")
    if analysis.get("mujoco_version") != "3.9.0":
        raise PreparationError("forensic MuJoCo version differs from the reviewed pin")
    if analysis.get("complete_saved_mjb_bytes_identical") is not True:
        raise PreparationError("forensic complete saved-MJB bytes are not identical")
    if analysis.get("model_array_bytes_identical") is not True:
        raise PreparationError("forensic model-array bytes are not identical")
    if analysis.get("model_array_fields_different") != []:
        raise PreparationError("forensic report lists different model arrays")
    if len(analysis.get("model_array_fields_compared", [])) != EXPECTED_MODEL_ARRAY_COUNT:
        raise PreparationError("forensic report does not compare all 468 expected model arrays")
    for key in ("reference", "actual"):
        record = analysis.get("binary_models", {}).get(key, {})
        if record.get("sha256") != EXPECTED_MJB_SHA256 or record.get("size_bytes") != EXPECTED_MJB_SIZE_BYTES:
            raise PreparationError(f"forensic {key} MJB differs from the frozen reference")


def protocol_for_revision4(r3_protocol: dict[str, Any], runner_sha: str,
                           raw_xml_review: dict[str, Any], forensic_sha: str) -> dict[str, Any]:
    protocol = json.loads(json.dumps(r3_protocol))
    protocol["schema_version"] = "flygym-steering-dose-calibration-protocol-v4"
    protocol["revision"] = 4
    protocol["entrypoint_source_sha256"] = runner_sha
    protocol["preparation_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    protocol["source_parent"] = {
        "revision_3_bundle": str(REVISION_3_ROOT),
        "revision_3_source_freeze_sha256": sha256_file(REVISION_3_ROOT / "source_freeze.json"),
        "revision_3_protocol_sha256": sha256_file(REVISION_3_ROOT / "protocol.json"),
        "compiled_identity_forensics_path": str(ANALYSIS_PATH),
        "compiled_identity_forensics_sha256": forensic_sha,
    }
    protocol["execution"]["entrypoint"] = str(
        PROJECT / "scripts/run_flygym_steering_dose_calibration_v4.py"
    )
    protocol["execution"]["invocation"] = (
        "python scripts/run_flygym_steering_dose_calibration_v4.py --execute"
    )
    protocol["execution"]["model_identity_gate"] = {
        "method": "save the complete compiled model from the actual FlyGym session.model with mujoco.mj_saveModel, then SHA-256 the saved MJB bytes",
        "expected_sha256": EXPECTED_MJB_SHA256,
        "expected_size_bytes": EXPECTED_MJB_SIZE_BYTES,
        "comparison": "exact SHA-256 equality required before controller construction; no numerical tolerance, subset comparison, or hash allowlist",
        "raw_xml": "always preserve raw constructor XML and hash; record whether it matches the prior paired-screen XML; raw XML equality is provenance only and is not a gate",
        "constructor_invariants": [
            "session.model is session.sim.mj_model",
            "session time is exactly zero within 1e-12 s",
            "constructor control/native sample counters are both zero",
            "save and gate the actual session model before creating the dose controller or advancing dynamics",
        ],
        "frozen_forensic_reference": {
            "analysis_sha256": forensic_sha,
            "reference_and_actual_saved_mjb_sha256": EXPECTED_MJB_SHA256,
            "reference_and_actual_saved_mjb_size_bytes": EXPECTED_MJB_SIZE_BYTES,
            "complete_saved_mjb_bytes_identical": True,
            "model_array_count_compared": EXPECTED_MODEL_ARRAY_COUNT,
            "model_array_bytes_identical": True,
            "forensic_execution_steps": 0,
            "forensic_mujoco_version": "3.9.0",
        },
        "raw_xml_difference_review": raw_xml_review,
    }
    protocol["instrumentation"] = {
        "preconstruction": protocol["instrumentation"].get("preconstruction", []),
        "immediately_after_constructor_before_controller_or_dynamics": [
            "preserve raw constructor MJCF bytes and SHA-256; record match status against prior paired-screen XML and known revision-3 XML",
            "save the complete actual session.model to compiled_model.mjb and record size/SHA-256",
            "record session.model/simulation model object identity, effective model options and hash, upstream source hashes, runtime module origins, session time and zero sample counters",
        ],
        "model_gate": (
            "require exact SHA-256 c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f "
            "for the complete MJB saved from session.model before controller construction; raw XML hash is not a model-identity gate"
        ),
        "model_or_controller_changes": False,
    }
    protocol["revision_4_changes"] = [
        "Replace the raw serialized MJCF SHA gate with exact SHA-256 identity of the complete saved MJB from the actual session.model.",
        "Retain raw XML and old-reference hash-match status as provenance; do not require the old raw XML hash.",
        "Keep the 15-case design, seeds, P/d schedules, controller, source/assets, CPU/memory limits, timeouts, queue serialization, and capture requirements unchanged.",
    ]
    return protocol


def prepare() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise PreparationError(f"refusing to overwrite existing revision-4 bundle: {OUTPUT_ROOT}")
    if not REVISION_3_ROOT.is_dir():
        raise PreparationError(f"revision-3 bundle is missing: {REVISION_3_ROOT}")

    r3_freeze_path = REVISION_3_ROOT / "source_freeze.json"
    r3_protocol_path = REVISION_3_ROOT / "protocol.json"
    r3_review_path = REVISION_3_ROOT / "preparation-review.json"
    r3_freeze = read_json(r3_freeze_path)
    r3_protocol = read_json(r3_protocol_path)
    r3_review = read_json(r3_review_path)
    if r3_freeze.get("revision") != 3 or r3_freeze.get("status") != "preparation_frozen_execution_entrypoint_ready_not_run":
        raise PreparationError("revision-3 source freeze is not a frozen, unexecuted entrypoint bundle")
    if r3_protocol.get("revision") != 3 or r3_freeze.get("protocol_sha256") != sha256_file(r3_protocol_path):
        raise PreparationError("revision-3 protocol no longer matches its source freeze")
    if r3_review.get("status") != "prepared_not_executed":
        raise PreparationError("revision-3 preparation review has an unexpected status")
    if r3_review.get("runner_sha256") != r3_freeze.get("runner_sha256"):
        raise PreparationError("revision-3 review does not identify the frozen runner")

    analysis = read_json(ANALYSIS_PATH)
    _validate_forensics(analysis)
    if sha256_file(ANALYSIS_TOOL_PATH) != analysis.get("source_sha256"):
        raise PreparationError("compiled-model forensics source tool changed")
    if sha256_file(REFERENCE_XML_PATH) != EXPECTED_REFERENCE_XML_SHA256:
        raise PreparationError("paired-screen reference XML hash changed")
    if sha256_file(REVISION_3_XML_PATH) != EXPECTED_REVISION_3_XML_SHA256:
        raise PreparationError("revision-3 constructor XML hash changed")
    if sha256_file(REFERENCE_MJB_PATH) != EXPECTED_MJB_SHA256 or REFERENCE_MJB_PATH.stat().st_size != EXPECTED_MJB_SIZE_BYTES:
        raise PreparationError("frozen MJB reference bytes changed")
    if sha256_file(ACTUAL_MJB_PATH) != EXPECTED_MJB_SHA256 or ACTUAL_MJB_PATH.stat().st_size != EXPECTED_MJB_SIZE_BYTES:
        raise PreparationError("forensic actual MJB bytes changed")
    raw_xml_review = inspect_xml_serialization_difference(REFERENCE_XML_PATH, REVISION_3_XML_PATH)

    for raw_path, prior in r3_freeze.get("source_files", {}).items():
        source = Path(raw_path)
        if not source.is_file():
            raise PreparationError(f"revision-3 frozen source disappeared: {source}")
        if sha256_file(source) != prior.get("sha256") or source.stat().st_size != prior.get("size_bytes"):
            raise PreparationError(f"revision-3 frozen source changed: {source}")
    for logical_name, prior in r3_freeze.get("asset_manifest", {}).get("files", {}).items():
        asset = Path(prior["physical_path"])
        if not asset.is_file() or sha256_file(asset) != prior.get("sha256") or asset.stat().st_size != prior.get("size_bytes"):
            raise PreparationError(f"revision-3 source asset changed: {logical_name} ({asset})")
    if r3_freeze.get("asset_manifest", {}).get("asset_count") != 79 or r3_freeze.get("asset_manifest", {}).get("verified_asset_count") != 79:
        raise PreparationError("revision-3 frozen asset count is not the expected 79")

    output_parent = OUTPUT_ROOT.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mechanical-dose-rev4-", dir=output_parent))
    global _STAGE_ROOT
    _STAGE_ROOT = stage
    try:
        copied_inputs: dict[str, Any] = {}
        for filename, prior in r3_freeze["copied_inputs"].items():
            source = Path(prior["snapshot_path"])
            if not source.is_file() or sha256_file(source) != prior["sha256"]:
                raise PreparationError(f"revision-3 frozen input changed: {source}")
            dst = stage / "frozen_inputs" / filename
            copied_inputs[filename] = copy_verified(source, dst, prior["sha256"])
            copied_inputs[filename]["verified_unchanged_from_revision_3"] = True

        forensic_inputs = {
            "compiled_identity_analysis.json": (ANALYSIS_PATH, pin_file(ANALYSIS_PATH, role="zero-step complete MJB and model-array identity report")),
            "paired_screen_compiled_model.xml": (REFERENCE_XML_PATH, pin_file(REFERENCE_XML_PATH, role="prior paired-screen raw compiled MJCF")),
            "revision_3_compiled_model.xml": (REVISION_3_XML_PATH, pin_file(REVISION_3_XML_PATH, role="revision-3 constructor raw compiled MJCF")),
            "compiled_model_reference.mjb": (REFERENCE_MJB_PATH, pin_file(REFERENCE_MJB_PATH, role="frozen exact complete saved-MJB identity reference")),
        }
        for filename, (source, prior) in forensic_inputs.items():
            copied_inputs[filename] = copy_verified(source, stage / "frozen_inputs" / filename, prior["sha256"])
            copied_inputs[filename]["role"] = prior["role"]

        bundle_files: dict[str, Any] = {}

        def snapshot_bundle_file(source: Path, relative: str, expected_sha: str | None = None,
                                 role: str | None = None) -> None:
            destination = stage / relative
            record = copy_verified(source, destination, expected_sha)
            bundle_files[relative] = {
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "role": role or "immutable source snapshot",
            }

        for filename, prior in r3_freeze.get("bundle_files", {}).items():
            source = REVISION_3_ROOT / filename
            snapshot_bundle_file(source, filename, prior["sha256"], prior.get("role"))
        for name in ("prepare_flygym_steering_dose_calibration_v1.py", "run_flygym_steering_dose_calibration_v1.py"):
            source = PROJECT / "scripts" / name
            snapshot_bundle_file(source, f"source_snapshots/revision-3/{name}",
                                 r3_freeze["source_files"][str(source.resolve())]["sha256"],
                                 "revision-3 source entrypoint snapshot")
        prep_path = Path(__file__).resolve(strict=True)
        runner_path = (PROJECT / "scripts/run_flygym_steering_dose_calibration_v4.py").resolve(strict=True)
        snapshot_bundle_file(prep_path, f"source_snapshots/revision-4/{prep_path.name}", role="revision-4 standard-library preparation script")
        snapshot_bundle_file(runner_path, f"source_snapshots/revision-4/{runner_path.name}", role="revision-4 frozen execution entrypoint")
        dump_json(stage / "frozen_inputs" / "raw-xml-difference-review.json", raw_xml_review)
        raw_review_record = pin_file(stage / "frozen_inputs" / "raw-xml-difference-review.json", role="standard-library verification of the retained XML attribute-placement diff")
        raw_review_record["snapshot_path"] = str(OUTPUT_ROOT / "frozen_inputs/raw-xml-difference-review.json")
        raw_review_record["source_path"] = str(REVISION_3_XML_PATH)
        copied_inputs["raw-xml-difference-review.json"] = raw_review_record

        source_records: dict[str, dict[str, Any]] = {}
        for raw_path, prior in r3_freeze["source_files"].items():
            source_records[raw_path] = {
                "sha256": prior["sha256"],
                "size_bytes": prior["size_bytes"],
                "verified_unchanged_from_revision_3": True,
            }
        for artifact, role in (
            (r3_freeze_path, "revision-3 source freeze parent record"),
            (r3_protocol_path, "revision-3 frozen protocol parent record"),
            (r3_review_path, "revision-3 preparation review parent record"),
            (ANALYSIS_PATH, "completed zero-step compiled-model identity forensics"),
            (REFERENCE_XML_PATH, "paired-screen reference raw MJCF"),
            (REVISION_3_XML_PATH, "revision-3 constructor raw MJCF"),
            (REFERENCE_MJB_PATH, "forensic reference complete saved MJB"),
            (ACTUAL_MJB_PATH, "forensic actual complete saved MJB"),
            (ANALYSIS_TOOL_PATH, "source code used for the compiled-model forensics"),
        ):
            source_records[str(artifact.resolve())] = pin_file(artifact, role=role)
        source_records[str(prep_path)] = pin_file(prep_path, role="revision-4 standard-library preparation script")
        source_records[str(runner_path)] = pin_file(runner_path, role="revision-4 execution entrypoint")

        runner_sha = source_records[str(runner_path)]["sha256"]
        prep_sha = source_records[str(prep_path)]["sha256"]
        analysis_sha = sha256_file(ANALYSIS_PATH)
        protocol = protocol_for_revision4(r3_protocol, runner_sha, raw_xml_review, analysis_sha)
        dump_json(stage / "protocol.json", protocol)

        asset_manifest = json.loads(json.dumps(r3_freeze["asset_manifest"]))
        asset_manifest["inherited_from_revision_3_source_freeze_sha256"] = sha256_file(r3_freeze_path)
        source_freeze = {
            "schema_version": "flygym-steering-dose-calibration-source-freeze-v4",
            "revision": 4,
            "status": "preparation_frozen_execution_entrypoint_ready_not_run",
            "prior_revision_3_source_freeze": {
                "path": str(r3_freeze_path),
                "sha256": sha256_file(r3_freeze_path),
                "all_source_files_reverified": True,
                "all_assets_reverified": True,
                "asset_count_inherited": asset_manifest["asset_count"],
                "verified_asset_count_inherited": asset_manifest["verified_asset_count"],
            },
            "source_files": source_records,
            "asset_manifest": asset_manifest,
            "copied_inputs": copied_inputs,
            "bundle_files": bundle_files,
            "compiled_identity_forensics": {
                "analysis_path": str(ANALYSIS_PATH),
                "analysis_snapshot_path": str(OUTPUT_ROOT / "frozen_inputs/compiled_identity_analysis.json"),
                "analysis_sha256": analysis_sha,
                "analysis_tool_path": str(ANALYSIS_TOOL_PATH),
                "analysis_tool_sha256": source_records[str(ANALYSIS_TOOL_PATH.resolve())]["sha256"],
                "reference_xml_sha256": EXPECTED_REFERENCE_XML_SHA256,
                "revision_3_xml_sha256": EXPECTED_REVISION_3_XML_SHA256,
                "reference_mjb_source_path": str(REFERENCE_MJB_PATH),
                "actual_mjb_source_path": str(ACTUAL_MJB_PATH),
                "reference_mjb_snapshot_path": str(OUTPUT_ROOT / "frozen_inputs/compiled_model_reference.mjb"),
                "expected_complete_saved_mjb_sha256": EXPECTED_MJB_SHA256,
                "expected_complete_saved_mjb_size_bytes": EXPECTED_MJB_SIZE_BYTES,
                "complete_saved_mjb_bytes_identical": analysis["complete_saved_mjb_bytes_identical"],
                "model_array_count_compared": len(analysis["model_array_fields_compared"]),
                "model_array_fields_different": analysis["model_array_fields_different"],
                "model_array_bytes_identical": analysis["model_array_bytes_identical"],
                "forensic_integration_steps": analysis["integration_steps"],
                "forensic_body_control_ticks": analysis["body_control_ticks"],
                "forensic_gpu_used": analysis["gpu_used"],
                "forensic_mujoco_version": analysis["mujoco_version"],
                "forensic_numpy_version": analysis["numpy_version"],
                "raw_xml_difference_review_snapshot_path": str(OUTPUT_ROOT / "frozen_inputs/raw-xml-difference-review.json"),
                "raw_xml_difference_review_sha256": raw_review_record["sha256"],
            },
            "runner_sha256": runner_sha,
            "preparation_script_sha256": prep_sha,
            "protocol_sha256": sha256_file(stage / "protocol.json"),
            "raw_asset_binaries_copied": False,
            "future_execution_must_reverify_all_source_asset_input_and_reference_hashes": True,
            "no_execution_approval_created": True,
            "older_revision_2_and_revision_3_run_records_modified": False,
        }
        dump_json(stage / "source_freeze.json", source_freeze)
        freeze_sha = sha256_file(stage / "source_freeze.json")
        review = {
            "schema_version": "flygym-steering-dose-calibration-preparation-review-v4",
            "status": "prepared_not_executed",
            "conclusion": (
                "Revision 4 is prepared for independent review. Runtime model identity uses exact SHA-256 of the complete MJB saved from the actual session.model, before controller construction; the old raw XML hash is provenance only. No body, NN, GPU, model compilation, test, or 15-case run was performed."
            ),
            "entrypoint_path": str(runner_path),
            "runner_sha256": runner_sha,
            "preparation_script_sha256": prep_sha,
            "source_freeze_sha256": freeze_sha,
            "protocol_sha256": source_freeze["protocol_sha256"],
            "case_count": len(protocol["execution"]["ordered_cases"]),
            "dose_design_unchanged_from_revision_3": protocol["design"] == r3_protocol["design"],
            "source_and_asset_inheritance": {
                "revision_3_source_file_count": len(r3_freeze["source_files"]),
                "revision_3_asset_count": asset_manifest["asset_count"],
                "verified_asset_count": asset_manifest["verified_asset_count"],
                "verified_total_asset_bytes": asset_manifest["verified_total_bytes"],
                "all_revision_3_sources_reverified": True,
                "all_79_assets_reverified": True,
                "forensic_input_files_snapshotted": sorted(forensic_inputs),
            },
            "compiled_model_identity_evidence": source_freeze["compiled_identity_forensics"],
            "raw_xml_difference_review": raw_xml_review,
            "model_gate": {
                "expected_complete_saved_mjb_sha256": EXPECTED_MJB_SHA256,
                "expected_complete_saved_mjb_size_bytes": EXPECTED_MJB_SIZE_BYTES,
                "strict_exact_byte_identity": True,
                "numerical_tolerance": None,
                "hash_allowlist": None,
                "raw_xml_gate": False,
                "controller_created_before_gate": False,
                "simulation_stepped_before_gate": False,
            },
            "static_session_model_audit": {
                "source": str(PROJECT / "scripts/run_flygym_reference_walk.py"),
                "pinned_sha256": r3_freeze["source_files"][str(PROJECT / "scripts/run_flygym_reference_walk.py")]["sha256"],
                "session_model_binding": "self.model = self.sim.mj_model",
                "constructor_call_before_capture": "mj.mj_forward(self.model, self.data)",
                "mj_forward_changes_data_not_model": True,
                "runner_saves_direct_session_model": True,
                "runner_compiles_from_xml_at_runtime": False,
                "controller_constructor_runs_after_exact_mjb_gate": True,
                "postconstruction_session_model_writes_found_in_pinned_session_source": False,
                "session_zero_time_and_sample_counters_are_checked_before_gate": True,
            },
            "runtime_gates": [
                "Reverify all frozen project source hashes, upstream controller sources, all 79 source assets, copied inputs, forensic report, reference binary, and bundle snapshots before loading physics dependencies.",
                "Preserve raw constructor XML and match status against the prior paired-screen XML without requiring equality.",
                "Save the complete actual session.model as compiled_model.mjb and require exact SHA-256 c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f before creating the dose controller or stepping dynamics.",
                "The MJB gate remains strict: no numeric tolerance, partial-array comparison, or alternate accepted hash.",
                "Keep the 15-case design, seeds, P/d schedule, controller, two-core/2-GiB caps, 600-second case timeout, and 9000-second batch timeout unchanged.",
            ],
            "not_verified_until_execution": [
                "Runtime MJB bytes from each future session constructor.",
                "All 15 cases, same-seed pairing, authority/contact captures, cgroup peaks, and time/resource limits.",
            ],
            "scope_boundary": "Preparation only; no independent root approval artifact or execution output was created.",
        }
        dump_json(stage / "preparation-review.json", review)
        md = [
            "# Mechanical steering dose calibration — revision 4",
            "",
            "**Status: prepared, not executed.** No body simulation, NN, GPU, new model compilation, tests, or 15-case run was performed. No independent root approval artifact was created. Revision-2/revision-3 failures and records remain at their original paths.",
            "",
            "## Model identity gate",
            "",
            "Revision 3 stopped before the first control step because the constructor's raw XML SHA-256 was `976e72a119b25b591e8b88aac63c2a5f06913013ec787330ae727f0f8be432fd`, while the prior paired-screen XML SHA-256 was `bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57`. The completed zero-step forensics saved both XMLs to MJB and found exact equality of the full 23,230,821-byte MJB files (`c21516b1229e4da7ac44a513b8ec1315d8a7cce6546668983473fc37b7c7e69f`) and all 468 model arrays.",
            "",
            "A standard-library diff was also checked and frozen: the retained XML pair differs only by which member of the `l_eye/r_eye` and `l_haltere/r_haltere` same-file mesh pairs explicitly carries `content_type=\"model/stl\``. This is evidence for the observed raw-serialization discrepancy; it does not relax the runtime gate.",
            "",
            "At runtime the new entrypoint preserves and hashes the constructor's raw XML, records whether it matches the old XML, then saves the complete model object from the actual FlyGym session via `mujoco.mj_saveModel(session.model, ..., None)`. It requires the exact expected MJB SHA before constructing the dose controller or advancing dynamics. There is no tolerance, subset-array gate, or hash allowlist. The old raw XML hash is provenance only.",
            "",
            "## Static constructor audit",
            "",
            "The pinned session source binds `session.model` directly to `session.sim.mj_model`; the constructor calls `mj_forward(model, data)` to populate derived state. The v4 runner saves that same model object directly rather than recompiling the XML. It checks zero time plus zero native/control sample counters before the identity gate. Controller construction is after the gate. The pinned session implementation has no subsequent assignment to model fields; `mj_forward` updates data, while later dynamics are gated behind the passing MJB check.",
            "",
            "## Unchanged cohort",
            "",
            "The 15 seeds/dose combinations, P=(0.258,0.258), 7-second duration, 1-second shared prefix, 250-ms ramp, controller/source/assets, CPU limit of 2 cores, memory limit of 2 GiB, 600-second per-case timeout, and 9000-second batch timeout are inherited unchanged from revision 3.",
            "",
            "## Frozen evidence",
            "",
            "`source_freeze.json` retains the 15 revision-3 source pins, all 79 verified upstream assets, the prior frozen configuration inputs, forensic source tool, both XMLs, forensic report, and the saved MJB reference. Raw asset binaries are not recopied. The runner rechecks the source/asset/reference pins before loading physics dependencies.",
            "",
            "The bundle contains `protocol.json`, `source_freeze.json`, `preparation-review.json`, this review, `frozen_inputs/`, and source snapshots. It does not contain an execution approval.",
            "",
        ]
        (stage / "preparation-review.md").write_text("\n".join(md), encoding="utf-8")
        stage.rename(OUTPUT_ROOT)
        return review
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True,
                        help="verify frozen pins and prepare the revision-4 bundle")
    args = parser.parse_args()
    del args
    report = prepare()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
