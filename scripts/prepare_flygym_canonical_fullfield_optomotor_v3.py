#!/usr/bin/env python3
"""Prepare R3 metadata-only correction for the canonical full-field cohort.

This adapter binds the unchanged R2 scientific runner and preparer by SHA-256,
adds the one descriptive case key consumed by the frozen summary writer, and
creates an isolated revision-3 bundle.  It never reads numeric R2 arrays or
starts a body, neural worker, GPU, or simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

sys.dont_write_bytecode = True

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
PARENT = Path("/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1")
R2_BASE = PARENT / "canonical-fullfield-optomotor-v1/revision-2"
BASE = PARENT / "canonical-fullfield-optomotor-v1/revision-3"
RUNNER = SCRIPTS / "run_flygym_canonical_fullfield_optomotor_v3.py"
PREPARER = Path(__file__).resolve()
R2_RUNNER = SCRIPTS / "run_flygym_canonical_fullfield_optomotor_v2.py"
R2_PREPARER = SCRIPTS / "prepare_flygym_canonical_fullfield_optomotor_v2.py"
R2_RUNNER_SHA256 = "69cfea56b30214f9777e795a42870988aaff957b726c4ef310b5ff27ae9ef2b4"
R2_PREPARER_SHA256 = "ad929cad9306ebf4b0f916a4834faa4b3bc25e0d580bb4e2ab2a4fe55a189924"
R2_PROTOCOL_SHA256 = "130ade32fdd2e865ae5cd3f65407a72d8376778393416443b706bbf9ab965657"
R2_FREEZE_SHA256 = "6f52660e4c47a3acf31e3285a373cbac664bb0709caeffea2d6864921fce9435"
R2_MANIFEST_SHA256 = "e28f9ba24bb5c4a37766d5c2404e192e75425fe2d0ee80652977b69088d8a4a0"
R2_CASE_ID = "canonical_fullfield_seed1729_static_c1"
AUDIT_MD = R2_BASE / "analysis/legacy-case-contract-audit-v1.md"
AUDIT_JSON = R2_BASE / "analysis/legacy-case-contract-audit-v1.json"
EXPECTED_R2_EXECUTION = {
    "unit": "canonical-fullfield-optomotor-v1-r2.service",
    "exec_session_id": 20739,
    "actual_exit_code": 1,
    "actual_runtime_s": 547.449,
    "root_reported_status": "failed_stopped",
}


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"expected JSON object at {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def load_parent_preparer():
    expected = {
        R2_RUNNER: R2_RUNNER_SHA256,
        R2_PREPARER: R2_PREPARER_SHA256,
        R2_BASE / "protocol.json": R2_PROTOCOL_SHA256,
        R2_BASE / "preparation/source-freeze.json": R2_FREEZE_SHA256,
        R2_BASE / "prepare-manifest.json": R2_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise PreparationError(f"immutable revision-2 pin changed: {path}")
    spec = importlib.util.spec_from_file_location("canonical_fullfield_parent_r2_preparer", R2_PREPARER)
    if spec is None or spec.loader is None:
        raise PreparationError("could not load the pinned revision-2 preparer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BASE = BASE
    module.PREP = BASE / "preparation"
    module.FREEZE_PATH = module.PREP / "source-freeze.json"
    module.RUNS = BASE / "runs"
    module.ANALYSIS = BASE / "analysis"
    module.RUNNER = RUNNER
    module.PREPARER = PREPARER
    return module


def preserve_revision2_failure() -> dict[str, Any]:
    case_dir = R2_BASE / "runs" / R2_CASE_ID
    queue = read_json(R2_BASE / "queue-status.json")
    case_record = read_json(case_dir / "run-record.json")
    summary = read_json(case_dir / "summary.json")
    if (queue.get("status") != "failed_stopped" or queue.get("current") != R2_CASE_ID
            or queue.get("completed") != [] or case_record.get("status") != "failed_stopped"
            or case_record.get("completed_neural_frames") != 2600
            or case_record.get("completed_body_ticks") != 13000
            or "KeyError: 'interpretation'" not in str(case_record.get("error", ""))
            or summary.get("status") != "complete"):
        raise PreparationError("revision-2 retained failure does not match the root-reported completed-capture summary-writer error")
    if not AUDIT_MD.is_file() or not AUDIT_JSON.is_file():
        raise PreparationError("independent source-only case-key audit is missing")
    audit = read_json(AUDIT_JSON)
    missing = audit.get("missing_key", {})
    if (missing.get("case_key") != "interpretation"
            or audit.get("other_missing_required_case_keys") != []
            or audit.get("other_missing_required_protocol_paths") != []):
        raise PreparationError("independent case-key audit does not support the single-key metadata correction")
    outputs = []
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            outputs.append(record(path))
    evidence = {
        "schema_version": "canonical-fullfield-r2-failed-capture-provenance-v1",
        "status": "failed_capture_preserved_not_reused_as_accepted_case",
        "r2_bundle_path": str(R2_BASE.resolve()),
        "r2_protocol": record(R2_BASE / "protocol.json"),
        "r2_prepare_manifest": record(R2_BASE / "prepare-manifest.json"),
        "r2_source_freeze": record(R2_BASE / "preparation/source-freeze.json"),
        "r2_runner": record(R2_RUNNER),
        "r2_preparer": record(R2_PREPARER),
        "r2_root_execution_review": record(R2_BASE / "root-execution-review.json"),
        "r2_queue_status": record(R2_BASE / "queue-status.json"),
        "r2_failed_case": {
            "case_id": R2_CASE_ID,
            "run_record": record(case_dir / "run-record.json"),
            "summary": record(case_dir / "summary.json"),
            "output_file_records": outputs,
            "status": "failed_stopped",
            "completed_neural_frames": 2600,
            "completed_body_ticks": 13000,
            "finalization_error": "KeyError: 'interpretation' in the existing summary writer",
            "not_reused_for_r3": True,
        },
        "root_reported_execution_terminal": EXPECTED_R2_EXECUTION,
        "independent_source_only_case_key_audit": {
            "markdown": record(AUDIT_MD),
            "json": record(AUDIT_JSON),
            "audit_status": audit.get("status"),
            "only_missing_required_case_key": "interpretation",
        },
        "numeric_case_arrays_loaded": False,
        "raw_r2_files_moved_or_modified": False,
    }
    provenance_dir = BASE / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    path = provenance_dir / "revision2-failed-capture-preserved.json"
    write_json(path, evidence)
    return record(path)


def patch_parent(module, provenance_record: dict[str, Any]) -> None:
    base_case_defs = module.case_defs
    interpretation = {
        "static_c1": "Static full-field C=1 stripes after the common 1 s gray prelude; descriptive engineering control.",
        "positive60": "Constant +60 deg/s full-field C=1 stripe motion from 1 to 13 s; stimulus metadata only.",
        "negative60": "Constant -60 deg/s full-field C=1 stripe motion from 1 to 13 s; stimulus metadata only.",
    }

    def case_defs_with_interpretation():
        rows = base_case_defs()
        for row in rows:
            row["interpretation"] = interpretation[row["direction"]]
        return rows

    module.case_defs = case_defs_with_interpretation
    base_freeze = module._source_freeze

    def source_freeze_with_parent_provenance(high_freeze, high_protocol, native, prepared_files, cases):
        freeze = base_freeze(high_freeze, high_protocol, native, prepared_files, cases)
        historical = {
            R2_RUNNER: R2_RUNNER_SHA256,
            R2_PREPARER: R2_PREPARER_SHA256,
            R2_BASE / "protocol.json": R2_PROTOCOL_SHA256,
            R2_BASE / "preparation/source-freeze.json": R2_FREEZE_SHA256,
            R2_BASE / "prepare-manifest.json": R2_MANIFEST_SHA256,
        }
        for path, digest in historical.items():
            item = module.record(path)
            if item["sha256"] != digest:
                raise PreparationError(f"revision-2 parent changed while preparing R3: {path}")
            freeze["files"][item["path"]] = item
        freeze["files"][provenance_record["path"]] = provenance_record
        audit_records = [record(AUDIT_MD), record(AUDIT_JSON)]
        for item in audit_records:
            freeze["files"][item["path"]] = item
        freeze["source_bindings"]["revision2_parent_runner"] = record(R2_RUNNER)
        freeze["source_bindings"]["revision2_parent_preparer"] = record(R2_PREPARER)
        freeze["source_bindings"]["revision2_failed_capture_provenance"] = provenance_record
        freeze["source_bindings"]["revision2_case_key_audit"] = {
            "markdown": audit_records[0], "json": audit_records[1],
        }
        freeze["runtime_file_paths"] = sorted(
            raw for raw in freeze["files"]
            if ("/scripts/" in raw or raw.startswith("/data/environments/projects/flygym-reference/")
                or raw.startswith("/data/models/flyvis/") or raw.startswith("/data/cache/malecns-gpu/")
                or ("/runs/" not in raw and "/analysis/" not in raw
                    and raw not in {str(module.HIGH_PROTOCOL), str(module.HIGH_MANIFEST)}))
        )
        return freeze

    module._source_freeze = source_freeze_with_parent_provenance
    base_make_protocol = module._make_protocol

    def make_protocol_r3(*args, **kwargs):
        protocol = base_make_protocol(*args, **kwargs)
        protocol["suite_id"] = "canonical-fullfield-optomotor-v1-revision-3"
        protocol["prepare_revision"] = 3
        protocol["revision3_metadata_correction"] = {
            "scope": "case descriptive metadata only",
            "r2_failure": "The frozen summary writer required `interpretation`, omitted from every R2 case row; R2 capture and failure remain immutable.",
            "only_case_field_added": "interpretation",
            "scientific_conditions_or_parameters_changed": False,
            "r2_capture_reused_as_r3_case": False,
            "r2_failed_capture_provenance_path": provenance_record["path"],
            "r2_failed_capture_provenance_sha256": provenance_record["sha256"],
        }
        command = (
            "systemd-run --user --wait --pipe --collect "
            "--unit=canonical-fullfield-optomotor-v1-r3.service "
            "--property=CPUQuota=200% --property=RuntimeMaxSec=18000 "
            "--setenv=CANONICAL_COORDINATOR_UNIT=canonical-fullfield-optomotor-v1-r3.service "
            f"{module.BODY_PYTHON} {RUNNER} --execute"
        )
        protocol["commands"]["execute_after_root_review"] = command
        return protocol

    module._make_protocol = make_protocol_r3


def _refresh_preparation_integrity(module) -> None:
    integrity_path = BASE / "preparation/preparation-integrity.json"
    files = {}
    for path in sorted(BASE.rglob("*")):
        if path.is_file() and path != integrity_path:
            item = module.record(path)
            files[item["path"]] = item
    write_json(integrity_path, {
        "schema_version": "canonical-fullfield-preparation-integrity-v1",
        "status": "prepared_not_executed",
        "files": files,
    })


def prepare() -> dict[str, Any]:
    if BASE.exists() and any(BASE.iterdir()):
        raise FileExistsError(f"refusing to overwrite a revision-3 bundle: {BASE}")
    provenance_record = preserve_revision2_failure()
    module = load_parent_preparer()
    patch_parent(module, provenance_record)
    result = module.prepare()

    manifest_path = BASE / "prepare-manifest.json"
    manifest = read_json(manifest_path)
    manifest["schema_version"] = "canonical-fullfield-optomotor-prepare-manifest-v3"
    manifest["revision3_metadata_correction"] = {
        "only_case_metadata_added": "interpretation",
        "r2_failed_capture_preserved_not_reused": True,
        "scientific_sources_and_parameters_changed": False,
        "failed_capture_provenance": provenance_record,
    }
    write_json(manifest_path, manifest)

    template_path = BASE / "root-execution-review.template.json"
    template = read_json(template_path)
    template["schema_version"] = "canonical-fullfield-root-review-template-v3"
    template["prepare_manifest_sha256"] = sha256(manifest_path)
    template["review_scope"] = (
        "revision 3 metadata-only correction after the immutable R2 finalization failure; nine sequential cases, "
        "same stimulus/body/neural sources and parameters, new R3 output paths, exact precontroller MJB, "
        "case CPU10/Memory5GiB/1800s and coordinator CPU2/outer18000s"
    )
    write_json(template_path, template)

    plan_path = BASE / "cohort-plan.json"
    plan = read_json(plan_path)
    plan["schema_version"] = "canonical-fullfield-cohort-plan-v3"
    plan["revision2_failed_capture_not_reused"] = {
        "path": provenance_record["path"], "sha256": provenance_record["sha256"],
    }
    write_json(plan_path, plan)

    protocol_path = BASE / "protocol.json"
    protocol = read_json(protocol_path)
    readme_path = BASE / "README.md"
    readme_path.write_text(
        "# Canonical full-field optomotor revision 3\n\n"
        "Status: prepared, not executed. Revision 3 preserves the same nine sequential cases, visual stimuli, body, "
        "neural worker, controller, decoder, and model as R2. It adds the missing descriptive `interpretation` "
        "case field needed by the existing summary writer. No scientific setting changed.\n\n"
        "Revision 2 captured the first static case through 2,600 neural frames and 13,000 control ticks, then "
        "failed during summary finalization because that case field was absent. The complete R2 output paths and "
        "hashes remain under revision 2 and are referenced in the R3 provenance record; that case is not reused "
        "as an accepted R3 capture. R3 uses separate output paths.\n\n"
        "The pre-onset same-seed triplet comparison remains the primary integrity check. Historical V2 comparison "
        "remains secondary. Descriptive captures do not imply biological validation.\n\n"
        f"Prepare: `{protocol['commands']['prepare']}`\n\n"
        f"Verify pins only: `{protocol['commands']['verify']}`\n\n"
        f"Execute after separate root review: `{protocol['commands']['execute_after_root_review']}`\n",
        encoding="utf-8",
    )
    _refresh_preparation_integrity(module)
    return {
        **result,
        "status": "prepared_not_executed",
        "bundle": str(BASE),
        "protocol_sha256": sha256(protocol_path),
        "runner_sha256": sha256(RUNNER),
        "preparer_sha256": sha256(PREPARER),
        "prepare_manifest_sha256": sha256(manifest_path),
        "source_freeze_sha256": sha256(BASE / "preparation/source-freeze.json"),
        "revision2_failed_capture_provenance": provenance_record,
        "body_physics_steps": 0,
        "nn_forward_started": False,
        "gpu_used": False,
        "root_execution_review_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    args = parser.parse_args()
    if not args.prepare:
        parser.error("only --prepare is supported")
    print(json.dumps(prepare(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
