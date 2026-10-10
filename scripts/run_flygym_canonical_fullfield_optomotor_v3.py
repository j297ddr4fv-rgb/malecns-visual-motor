#!/usr/bin/env python3
"""Revision 3 path adapter for the frozen canonical full-field runner.

Revision 3 changes only the per-case descriptive ``interpretation`` field
required by the existing final summary writer.  All scientific work remains
in the hash-pinned revision-2 implementation and its already frozen sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
PARENT = Path("/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1")
BASE = PARENT / "canonical-fullfield-optomotor-v1/revision-3"
PREP = BASE / "preparation"
RUNS = BASE / "runs"
ANALYSIS = BASE / "analysis"
PROTOCOL_PATH = BASE / "protocol.json"
FREEZE_PATH = PREP / "source-freeze.json"
RUNNER = Path(__file__).resolve()
PREPARER = SCRIPTS / "prepare_flygym_canonical_fullfield_optomotor_v3.py"
PARENT_RUNNER = SCRIPTS / "run_flygym_canonical_fullfield_optomotor_v2.py"
PARENT_RUNNER_SHA256 = "69cfea56b30214f9777e795a42870988aaff957b726c4ef310b5ff27ae9ef2b4"
PARENT_PREPARER = SCRIPTS / "prepare_flygym_canonical_fullfield_optomotor_v2.py"
PARENT_PREPARER_SHA256 = "ad929cad9306ebf4b0f916a4834faa4b3bc25e0d580bb4e2ab2a4fe55a189924"


class AdapterError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdapterError(f"expected JSON object at {path}")
    return value


def load_parent_runner():
    if sha256(PARENT_RUNNER) != PARENT_RUNNER_SHA256:
        raise AdapterError("frozen revision-2 runner changed; refusing to import it")
    if sha256(PARENT_PREPARER) != PARENT_PREPARER_SHA256:
        raise AdapterError("frozen revision-2 preparer changed; refusing to bind its protocol")
    spec = importlib.util.spec_from_file_location("canonical_fullfield_parent_r2", PARENT_RUNNER)
    if spec is None or spec.loader is None:
        raise AdapterError("could not load pinned revision-2 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BASE = BASE
    module.PREP = PREP
    module.RUNS = RUNS
    module.ANALYSIS = ANALYSIS
    module.PROTOCOL_PATH = PROTOCOL_PATH
    module.FREEZE_PATH = FREEZE_PATH
    module.RUNNER = RUNNER
    module.PREPARER = PREPARER
    return module


def _metadata_preflight(module) -> dict[str, Any]:
    unit = os.environ.get("CANONICAL_METADATA_PREFLIGHT_UNIT", "")
    if not unit.endswith(".service"):
        raise AdapterError("metadata preflight must run in its bounded systemd service")
    protocol = read_json(PROTOCOL_PATH)
    module.verify_frozen_inputs(protocol, full=True)
    if protocol.get("prepare_revision") != 3 or protocol.get("suite_id") != "canonical-fullfield-optomotor-v1-revision-3":
        raise AdapterError("R3 protocol identity mismatch")
    conditions = protocol.get("stimuli", {}).get("conditions", [])
    if len(conditions) != 9 or any(not isinstance(row.get("interpretation"), str) or not row["interpretation"]
                                   for row in conditions):
        raise AdapterError("R3 case descriptive metadata is incomplete")
    props = module._systemd_properties(unit)
    cpu_usec = module._systemd_timespan_usec(props.get("CPUQuotaPerSecUSec", ""))
    runtime_usec = module._systemd_timespan_usec(props.get("RuntimeMaxUSec", ""))
    memory_bytes = module._memory_property_bytes(props.get("MemoryMax", ""))
    cgroup = module._cgroup_path()
    effective = module._effective_cgroup_caps()
    if (cpu_usec != 2_000_000 or runtime_usec != 120_000_000
            or memory_bytes != 1024**3 or props.get("ControlGroup") != cgroup
            or effective.get("control_group") != cgroup
            or effective.get("effective_memory_cap_bytes") != 1024**3
            or str(effective.get("effective_cpu_cap_cores")) != "2"):
        raise AdapterError("bounded metadata-preflight systemd/cgroup readbacks do not match 2 CPU, 1 GiB, 120 s")
    return {
        "schema_version": "canonical-fullfield-r3-metadata-preflight-v1",
        "status": "metadata_binding_verified",
        "unit": unit,
        "systemd_properties": props,
        "effective_cgroup": effective,
        "parent_revision2_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_revision2_preparer_sha256": PARENT_PREPARER_SHA256,
        "r3_runner_sha256": sha256(RUNNER),
        "r3_preparer_sha256": sha256(PREPARER),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "prepare_manifest_sha256": sha256(BASE / "prepare-manifest.json"),
        "simulation_constructed": False,
        "body_physics_steps": 0,
        "nn_started": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-preflight", action="store_true",
                        help="verify R3 metadata/path bindings and bounded systemd caps only")
    args, _ = parser.parse_known_args()
    module = load_parent_runner()
    if args.metadata_preflight:
        print(json.dumps(_metadata_preflight(module), indent=2, sort_keys=True))
        return 0
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
