#!/usr/bin/env python3
"""Read-only reproducibility and performance gate for benchmark artifacts.

This gate is intentionally narrower than the flight and experiment validators
in this repository.  It does not inspect a trajectory for takeoff, landing,
biological evidence, or controller behaviour.  It checks whether an artifact
has enough *engineering evidence* to be reproduced and compared safely:

* the complete unit-test command and a passing result are recorded;
* GPU/cgroup preflight facts are explicit (and CUDA never gets initialised);
* a versioned manifest has a deterministic seed and an exact config hash;
* required table columns have exact, canonical SHA-256 hashes;
* no controller is declared as the authority for the benchmark;
* measured resources and compact artifact sizes stay below policy ceilings;
* numerical-equivalence observations are inside declared tolerances; and
* benchmark metadata is sufficient to interpret timing measurements; and
* a declared ``renderer_backend=warp_cuda`` carries a strict Warp/GPU/scene
  contract, separated cold-JIT and steady-state timings, exact-column hashes,
  and an analytic-reference comparison.

The command only reads the artifact directory and its declared config/data
files.  It never writes a report, launches a workload, imports CUDA/Torch, or
changes a cgroup.  A non-zero exit status is returned for any missing or
inconsistent evidence.

Typical use::

    python3 scripts/engineering_gate.py artifacts/my-benchmark
    python3 scripts/engineering_gate.py artifacts/my-benchmark --json

The manifest contract and limits live in ``config/engineering_gates.json``.
The public helpers are dependency-free and useful to focused unit tests:
``audit_artifact``, ``column_hash``, ``canonical_column_bytes`` and
``load_policy``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "config" / "engineering_gates.json"
REPORT_SCHEMA_VERSION = "engineering-gate-report-1"
EXPECTED_POLICY_SCHEMA = "engineering-gates-1"
MISSING = object()


class EngineeringGateError(RuntimeError):
    """Raised only for invalid gate-policy input, not for artifact failures."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise EngineeringGateError(f"cannot read {description} {path}: {exc}") from exc


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load and minimally validate the policy JSON.

    A malformed policy is an operator/configuration error and raises
    :class:`EngineeringGateError`; malformed artifacts are instead represented
    as a failed audit report so CI can print every useful reason.
    """

    policy_path = Path(path).expanduser()
    value = _load_json(policy_path, "engineering-gate policy")
    if not isinstance(value, dict):
        raise EngineeringGateError("engineering-gate policy must be a JSON object")
    if value.get("schema_version") != EXPECTED_POLICY_SCHEMA:
        raise EngineeringGateError(
            "unsupported engineering-gate policy schema: "
            f"{value.get('schema_version')!r}"
        )
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EngineeringGateError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_column_bytes(values: Iterable[Any]) -> bytes:
    """Return the exact bytes hashed for a column.

    Each row value is encoded as one UTF-8 canonical JSON record followed by a
    newline.  This keeps hashes independent of JSON object key order and
    whitespace while preserving row order and value types.  It also makes the
    algorithm straightforward to reproduce outside Python.
    """

    chunks: list[bytes] = []
    for value in values:
        chunks.append(_canonical_json(value))
        chunks.append(b"\n")
    return b"".join(chunks)


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    """Read row-oriented JSON, JSONL, or CSV data without normalising values."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise EngineeringGateError(
                            f"{path.name}: blank JSONL record at line {line_number}"
                        )
                    value = json.loads(line, parse_constant=_reject_json_constant)
                    if not isinstance(value, Mapping):
                        raise EngineeringGateError(
                            f"{path.name}: JSONL row {line_number} is not an object"
                        )
                    rows.append(value)
        except (OSError, UnicodeError, ValueError) as exc:
            if isinstance(exc, EngineeringGateError):
                raise
            raise EngineeringGateError(f"cannot read rows from {path}: {exc}") from exc
        return rows

    if suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise EngineeringGateError(f"cannot read rows from {path}: {exc}") from exc

    value = _load_json(path, "row data")
    if isinstance(value, list):
        rows_value = value
    elif isinstance(value, Mapping):
        rows_value = None
        for key in ("rows", "records", "trajectory", "samples", "telemetry", "data"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                rows_value = candidate
                break
        if rows_value is None:
            # A single row object is useful for compact benchmark probes.
            rows_value = [value]
    else:
        raise EngineeringGateError(f"{path.name}: expected row-oriented JSON")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows_value):
        if not isinstance(row, Mapping):
            raise EngineeringGateError(f"{path.name}: row {index} is not an object")
        rows.append(row)
    return rows


def _nested_value(row: Mapping[str, Any], column: str) -> Any:
    value: Any = row
    for part in column.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return MISSING
        value = value[part]
    return value


def column_hash(path: str | Path, column: str) -> str:
    """Compute the canonical SHA-256 hash for ``column`` in a row artifact."""

    data_path = Path(path)
    rows = _read_rows(data_path)
    if not rows:
        raise EngineeringGateError(f"{data_path.name}: cannot hash empty row data")
    values: list[Any] = []
    for index, row in enumerate(rows):
        value = _nested_value(row, column)
        if value is MISSING:
            raise EngineeringGateError(
                f"{data_path.name}: required column {column!r} is missing in row {index}"
            )
        values.append(value)
    return hashlib.sha256(canonical_column_bytes(values)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EngineeringGateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _finite_number(value: Any, label: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if integer:
        if number != math.trunc(number):
            raise ValueError(f"{label} must be an integer")
        return int(number)
    return number


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_or_empty(value: Any, label: str, failures: list[str]) -> Mapping[str, Any]:
    try:
        return _mapping(value, label)
    except ValueError as exc:
        failures.append(str(exc))
        return {}


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def _artifact_files(artifact: Path) -> list[Path]:
    files: list[Path] = []
    try:
        for path in artifact.rglob("*"):
            if path.is_file():
                files.append(path)
    except OSError as exc:
        raise EngineeringGateError(f"cannot enumerate artifact files: {exc}") from exc
    return files


def _resolve_declared_path(artifact: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return None
    candidate = Path(str(raw_path)).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [artifact / candidate, ROOT / candidate]
    for item in candidates:
        try:
            if item.is_file():
                return item
        except OSError:
            continue
    return None


def _status(failures: Sequence[str]) -> str:
    return "PASS" if not failures else "FAIL"


def _check_manifest_schema(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("manifest"), "policy.manifest")
    expected_schema = spec.get("schema_version")
    if manifest.get("schema_version") != expected_schema:
        failures.append(
            f"schema_version must be {expected_schema!r}, got {manifest.get('schema_version')!r}"
        )
    for field in spec.get("required_fields", []):
        if field not in manifest:
            failures.append(f"missing required manifest field {field!r}")
    run_id = manifest.get("run_id")
    pattern = spec.get("run_id_pattern")
    if not isinstance(run_id, str) or not isinstance(pattern, str) or not re.fullmatch(pattern, run_id):
        failures.append("run_id must be a portable non-empty identifier")
    return failures


def _check_unit_tests(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("unit_tests"), "policy.unit_tests")
    try:
        tests = _mapping(manifest.get("unit_tests"), "manifest.unit_tests")
    except ValueError as exc:
        return [str(exc)]
    for field in spec.get("required_fields", []):
        if field not in tests:
            failures.append(f"unit_tests missing {field!r}")
    command = tests.get("command")
    expected = spec.get("required_command")
    if command != expected:
        failures.append(f"unit-test command must be exactly {expected!r}, got {command!r}")
    if tests.get("status") != spec.get("required_status"):
        failures.append(f"unit-test status must be {spec.get('required_status')!r}")
    try:
        passed = _finite_number(tests.get("passed"), "unit_tests.passed", integer=True)
        failed = _finite_number(tests.get("failed", 0), "unit_tests.failed", integer=True)
        errors = _finite_number(tests.get("errors", 0), "unit_tests.errors", integer=True)
        skipped = _finite_number(tests.get("skipped", 0), "unit_tests.skipped", integer=True)
        minimum = int(spec.get("minimum_passed", 1))
        if passed < minimum:
            failures.append(f"unit-test passed count {passed} is below minimum {minimum}")
        if failed != 0 or errors != 0:
            failures.append(f"unit-test failures/errors must be zero (failed={failed}, errors={errors})")
        if skipped < 0:
            failures.append("unit_tests.skipped must be non-negative")
    except ValueError as exc:
        failures.append(str(exc))
    return failures


def _check_preflight(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("preflight"), "policy.preflight")
    try:
        preflight = _mapping(manifest.get("preflight"), "manifest.preflight")
    except ValueError as exc:
        return [str(exc)]
    for field in spec.get("required_fields", []):
        if field not in preflight:
            failures.append(f"preflight missing {field!r}")
    gpu_requested = preflight.get("gpu_requested")
    if not isinstance(gpu_requested, bool):
        failures.append("preflight.gpu_requested must be boolean")
        gpu_requested = False
    gate = str(preflight.get("cuda_gate", "")).strip().lower()
    if gpu_requested and gate != str(spec.get("gpu_gate_open_value", "open")):
        failures.append(f"GPU run requires cuda_gate={spec.get('gpu_gate_open_value')!r}, got {gate!r}")
    if not gpu_requested and gate != str(spec.get("cpu_gate_value", "not_requested")):
        failures.append(f"CPU/non-GPU run requires cuda_gate={spec.get('cpu_gate_value')!r}, got {gate!r}")

    uvm = str(preflight.get("uvm_disable_hmm", "")).strip().upper()
    if gpu_requested and uvm not in {"Y", "YES", "1", "TRUE"}:
        failures.append("GPU run requires uvm_disable_hmm=Y")
    if not gpu_requested and uvm not in {"", "N/A", "NA", "NOT_APPLICABLE", "NOT-APPLICABLE"}:
        failures.append("non-GPU run must mark uvm_disable_hmm as not applicable")

    nvidia = preflight.get("nvidia_smi")
    if not isinstance(nvidia, Mapping):
        failures.append("preflight.nvidia_smi must be an object with explicit presence")
    elif gpu_requested:
        if nvidia.get("present") is not True:
            failures.append("GPU run requires nvidia_smi.present=true")
        if not str(nvidia.get("query", nvidia.get("name", ""))).strip():
            failures.append("GPU run requires a non-empty nvidia-smi identity/query")

    cgroup = preflight.get("cgroup")
    try:
        cgroup_map = _mapping(cgroup, "preflight.cgroup")
    except ValueError as exc:
        return failures + [str(exc)]
    for field in spec.get("required_cgroup_fields", []):
        if field not in cgroup_map:
            failures.append(f"preflight.cgroup missing {field!r}")
    if spec.get("require_cgroup_memory_ceiling", True):
        try:
            memory_max = _finite_number(
                cgroup_map.get("memory_max_bytes"),
                "preflight.cgroup.memory_max_bytes",
                integer=True,
            )
            if memory_max <= 0:
                failures.append("preflight.cgroup.memory_max_bytes must be positive")
        except ValueError as exc:
            failures.append(str(exc))
    cpu_max = cgroup_map.get("cpu_max")
    if isinstance(cpu_max, str):
        if not cpu_max.strip():
            failures.append("preflight.cgroup.cpu_max must not be empty")
    elif isinstance(cpu_max, Mapping):
        if "quota_us" not in cpu_max or "period_us" not in cpu_max:
            failures.append("preflight.cgroup.cpu_max object needs quota_us and period_us")
    elif cpu_max is not None:
        failures.append("preflight.cgroup.cpu_max must be text or an object")
    return failures


def _check_authority(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("authority"), "policy.authority")
    try:
        authority = _mapping(manifest.get("authority"), "manifest.authority")
    except ValueError as exc:
        return [str(exc)]
    for field in spec.get("required_fields", []):
        if field not in authority:
            failures.append(f"authority missing {field!r}")
    expected = str(spec.get("controller_authority", "none"))
    if authority.get("controller_authority") != expected:
        failures.append(
            f"controller authority must be {expected!r}, got {authority.get('controller_authority')!r}"
        )
    if authority.get("writes_body_state") is not False:
        failures.append("authority.writes_body_state must be false")
    if authority.get("writes_motor_command") is not False:
        failures.append("authority.writes_motor_command must be false")
    return failures


def _check_determinism(
    artifact: Path, manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("determinism"), "policy.determinism")
    seed = manifest.get("seed")
    if spec.get("require_integer_seed", True):
        try:
            integer_seed = _finite_number(seed, "manifest.seed", integer=True)
            if integer_seed < 0:
                failures.append("manifest.seed must be non-negative")
        except ValueError as exc:
            failures.append(str(exc))

    deterministic = manifest.get("deterministic")
    if spec.get("require_explicit_deterministic", True) and deterministic is not True:
        failures.append("manifest.deterministic must be true")

    config = manifest.get("config")
    try:
        config_map = _mapping(config, "manifest.config")
    except ValueError as exc:
        return failures + [str(exc)]
    config_path = _resolve_declared_path(artifact, config_map.get("path"))
    if config_path is None:
        failures.append(f"config path does not resolve to a file: {config_map.get('path')!r}")
        return failures
    declared_hash = str(config_map.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
        failures.append("config.sha256 must be a 64-character lowercase SHA-256 digest")
    else:
        actual_hash = _sha256_file(config_path)
        if actual_hash != declared_hash:
            failures.append(f"config SHA-256 mismatch: declared {declared_hash}, actual {actual_hash}")
    if spec.get("require_config_seed_match", True):
        try:
            config_value = _load_json(config_path, "declared config")
            if not isinstance(config_value, Mapping) or "seed" not in config_value:
                failures.append("declared config must contain an explicit seed")
            elif config_value.get("seed") != seed:
                failures.append(
                    f"seed mismatch: manifest.seed={seed!r}, config.seed={config_value.get('seed')!r}"
                )
        except EngineeringGateError as exc:
            failures.append(str(exc))
    return failures


def _column_specs(manifest: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any]]:
    value = manifest.get("column_hashes")
    if not isinstance(value, Mapping):
        return None, {}
    algorithm = value.get("algorithm")
    files_value = value.get("files", value.get("artifacts", {}))
    if not isinstance(files_value, Mapping):
        return algorithm if isinstance(algorithm, str) else None, {}
    return algorithm if isinstance(algorithm, str) else None, files_value


def _declared_columns(files: Mapping[str, Any], filename: str) -> Mapping[str, Any] | None:
    raw = files.get(filename)
    if not isinstance(raw, Mapping):
        return None
    columns = raw.get("columns", raw)
    if not isinstance(columns, Mapping):
        return None
    return columns


def _check_column_hashes(artifact: Path, manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("column_hashes"), "policy.column_hashes")
    algorithm, files = _column_specs(manifest)
    expected_algorithm = spec.get("algorithm")
    if algorithm != expected_algorithm:
        failures.append(f"column hash algorithm must be {expected_algorithm!r}, got {algorithm!r}")
    if not files:
        return failures + ["manifest.column_hashes.files must be a non-empty object"]
    required_specs = spec.get("required", [])
    if not isinstance(required_specs, Sequence):
        return failures + ["policy.column_hashes.required must be an array"]
    for required in required_specs:
        if not isinstance(required, Mapping):
            failures.append("policy column hash requirement is not an object")
            continue
        filename = str(required.get("file", ""))
        data_path = artifact / filename
        if not data_path.is_file():
            failures.append(f"required column artifact is missing: {filename}")
            continue
        columns = _declared_columns(files, filename)
        if columns is None:
            failures.append(f"manifest.column_hashes.files missing {filename!r}")
            continue
        for column in required.get("columns", []):
            column_name = str(column)
            declared = columns.get(column_name)
            if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared):
                failures.append(f"missing/invalid hash for {filename}.{column_name}")
                continue
            try:
                actual = column_hash(data_path, column_name)
            except EngineeringGateError as exc:
                failures.append(str(exc))
                continue
            if actual.lower() != declared.lower():
                failures.append(
                    f"column hash mismatch for {filename}.{column_name}: declared {declared}, actual {actual}"
                )
    return failures


def _check_resources(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("resources"), "policy.resources")
    try:
        resources = _mapping(manifest.get("resources"), "manifest.resources")
        limits = _mapping(resources.get("limits"), "manifest.resources.limits")
        usage = _mapping(resources.get("usage"), "manifest.resources.usage")
    except ValueError as exc:
        return [str(exc)]
    preflight = manifest.get("preflight", {})
    gpu_requested = isinstance(preflight, Mapping) and preflight.get("gpu_requested") is True
    required_limits = list(spec.get("required_limit_fields", []))
    required_usage = list(spec.get("required_usage_fields", []))
    if gpu_requested:
        required_limits.extend(spec.get("gpu_limit_fields", []))
        required_usage.extend(spec.get("gpu_usage_fields", []))
    ceilings = _mapping(spec.get("ceilings"), "policy.resources.ceilings")
    parsed_limits: dict[str, float | int] = {}
    parsed_usage: dict[str, float | int] = {}
    for field in required_limits:
        if field not in limits:
            failures.append(f"resources.limits missing {field!r}")
            continue
        try:
            value = _finite_number(limits[field], f"resources.limits.{field}", integer=True)
            if value <= 0:
                failures.append(f"resources.limits.{field} must be positive")
            parsed_limits[field] = value
            ceiling = ceilings.get(field)
            if ceiling is not None and value > int(ceiling):
                failures.append(f"resources.limits.{field} exceeds policy ceiling {ceiling}")
        except ValueError as exc:
            failures.append(str(exc))
    for field in required_usage:
        if field not in usage:
            failures.append(f"resources.usage missing {field!r}")
            continue
        try:
            value = _finite_number(usage[field], f"resources.usage.{field}", integer=True)
            if value < 0:
                failures.append(f"resources.usage.{field} must be non-negative")
            parsed_usage[field] = value
        except ValueError as exc:
            failures.append(str(exc))
    comparisons = {
        "peak_host_rss_bytes": "host_rss_max_bytes",
        "peak_cuda_allocated_bytes": "cuda_allocated_max_bytes",
        "peak_cuda_reserved_bytes": "cuda_reserved_max_bytes",
    }
    for usage_field, limit_field in comparisons.items():
        if usage_field in parsed_usage and limit_field in parsed_limits:
            if parsed_usage[usage_field] > parsed_limits[limit_field]:
                failures.append(
                    f"{usage_field}={parsed_usage[usage_field]} exceeds {limit_field}={parsed_limits[limit_field]}"
                )
    if isinstance(preflight, Mapping):
        cgroup = preflight.get("cgroup")
        if isinstance(cgroup, Mapping) and "memory_max_bytes" in cgroup and "cgroup_memory_max_bytes" in parsed_limits:
            try:
                preflight_max = _finite_number(
                    cgroup["memory_max_bytes"],
                    "preflight.cgroup.memory_max_bytes",
                    integer=True,
                )
                if preflight_max != parsed_limits["cgroup_memory_max_bytes"]:
                    failures.append("preflight cgroup memory ceiling disagrees with resources.limits")
            except ValueError as exc:
                failures.append(str(exc))
    return failures


def _check_artifact_budget(artifact: Path, manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("artifact_budget"), "policy.artifact_budget")
    try:
        budget = _mapping(manifest.get("artifact_budget"), "manifest.artifact_budget")
        declared_total = _finite_number(budget.get("max_total_bytes"), "artifact_budget.max_total_bytes", integer=True)
        if declared_total <= 0:
            failures.append("artifact_budget.max_total_bytes must be positive")
        policy_total = int(spec.get("max_total_bytes", 0))
        if policy_total and declared_total > policy_total:
            failures.append(f"artifact total budget {declared_total} exceeds policy {policy_total}")
    except ValueError as exc:
        return [str(exc)]
    per_file = budget.get("max_file_bytes")
    if not isinstance(per_file, Mapping):
        failures.append("artifact_budget.max_file_bytes must be an object")
        per_file = {}
    policy_per_file = spec.get("max_file_bytes", {})
    if not isinstance(policy_per_file, Mapping):
        policy_per_file = {}
    policy_file_limits = [
        int(value)
        for value in policy_per_file.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    policy_default_file_limit = max(
        policy_file_limits or [int(spec.get("max_total_bytes", 0))]
    )
    try:
        files = _artifact_files(artifact)
        total = sum(path.stat().st_size for path in files)
        if total > int(declared_total):
            failures.append(f"artifact total size {total} exceeds declared budget {declared_total}")
        for path in files:
            rel = path.relative_to(artifact).as_posix()
            limit_value = per_file.get(rel)
            if limit_value is None:
                # Unlisted files are bounded by the policy's largest explicit
                # file limit, so adding a huge untracked payload cannot evade
                # the compact-artifact gate.
                limits = [
                    int(value)
                    for value in per_file.values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                limit_value = max(limits) if limits else policy_default_file_limit
            limit = int(_finite_number(limit_value, f"artifact_budget.max_file_bytes.{rel}", integer=True))
            policy_limit = policy_per_file.get(rel)
            if policy_limit is not None and limit > int(policy_limit):
                failures.append(f"file budget for {rel} exceeds policy ceiling {policy_limit}")
            elif policy_limit is None and limit > policy_default_file_limit:
                failures.append(
                    f"file budget for unlisted {rel} exceeds policy ceiling {policy_default_file_limit}"
                )
            if path.stat().st_size > limit:
                failures.append(f"artifact file {rel} size {path.stat().st_size} exceeds budget {limit}")
    except (OSError, ValueError) as exc:
        failures.append(str(exc))
    return failures


def _check_equivalence(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("equivalence"), "policy.equivalence")
    raw = manifest.get("equivalence", manifest.get("numerical_equivalence"))
    try:
        equivalence = _mapping(raw, "manifest.equivalence")
    except ValueError as exc:
        return [str(exc)]
    if equivalence.get("status") != "PASS":
        failures.append("equivalence.status must be 'PASS'")
    if spec.get("required_reference", True) and not str(
        equivalence.get("reference_id", equivalence.get("reference_artifact", ""))
    ).strip():
        failures.append("equivalence must identify a reference artifact/run")
    tolerances = equivalence.get("tolerances", {})
    if not isinstance(tolerances, Mapping):
        failures.append("equivalence.tolerances must be an object")
        tolerances = {}
    observed = equivalence.get("metrics", equivalence.get("observed", {}))
    if not isinstance(observed, Mapping):
        failures.append("equivalence.metrics must be an object")
        observed = {}
    policy_tolerances = _mapping(spec.get("tolerances"), "policy.equivalence.tolerances")
    for metric in spec.get("metrics", []):
        entry = observed.get(metric)
        if isinstance(entry, Mapping):
            value = entry.get("observed", entry.get("value"))
            tolerance = entry.get("tolerance", tolerances.get(metric, policy_tolerances.get(metric)))
        else:
            value = entry
            tolerance = tolerances.get(metric, policy_tolerances.get(metric))
        if value is None:
            failures.append(f"equivalence metric {metric!r} is missing")
            continue
        try:
            observed_value = _finite_number(value, f"equivalence.metrics.{metric}")
            tolerance_value = _finite_number(tolerance, f"equivalence.tolerances.{metric}")
            if observed_value < 0 or tolerance_value < 0:
                failures.append(f"equivalence {metric} values must be non-negative")
            elif observed_value > tolerance_value:
                failures.append(
                    f"equivalence {metric}={observed_value} exceeds tolerance {tolerance_value}"
                )
        except ValueError as exc:
            failures.append(str(exc))
    return failures


def _performance_acceptance_record(
    manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str] | None:
    """Find an explicitly declared full-run acceptance record, if present."""

    raw_spec = policy.get("performance_acceptance", {})
    if not isinstance(raw_spec, Mapping):
        raise ValueError("policy.performance_acceptance must be an object")
    spec = raw_spec
    keys = spec.get("record_keys", ("performance_acceptance", "acceptance"))
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
        raise ValueError("policy.performance_acceptance.record_keys must be a list")
    for key_value in keys:
        key = str(key_value)
        candidate = manifest.get(key)
        if isinstance(candidate, Mapping):
            return candidate, key

    benchmark = manifest.get("benchmark")
    if isinstance(benchmark, Mapping):
        nested = benchmark.get("performance_acceptance", benchmark.get("acceptance"))
        if isinstance(nested, Mapping):
            return nested, "benchmark.acceptance"
        record_type = benchmark.get("type", benchmark.get("record_type"))
        record_types = spec.get("record_types", ())
        if record_type is not None and record_type in record_types:
            return benchmark, "benchmark"
    return None


def _acceptance_value(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    location: str,
    field: str,
) -> Any:
    """Read an execution field from the record first, then the manifest."""

    roots: list[Any] = []
    candidate = record.get(location)
    if isinstance(candidate, Mapping):
        roots.append(candidate)
    candidate = manifest.get(location)
    if isinstance(candidate, Mapping):
        roots.append(candidate)
    aliases = {
        "renderer_backend": ("renderer_backend", "renderer", "render_backend"),
        "neural_backend": ("neural_backend",),
        "neural_device": ("neural_device",),
        "neural_cuda_graph": ("neural_cuda_graph",),
        "vision_backend": ("vision_backend", "vision"),
        "vision_device": ("vision_device",),
    }
    direct_aliases = {
        ("requested_execution", "renderer_backend"): (
            "requested_renderer_backend",
            "renderer_backend_requested",
        ),
        ("effective_execution", "renderer_backend"): (
            "effective_renderer_backend",
            "renderer_backend_effective",
        ),
    }
    for source in (record, manifest):
        for key in direct_aliases.get((location, field), ()):
            if key in source:
                return source[key]
    for root in roots:
        for key in aliases.get(field, (field,)):
            if key in root:
                return root[key]
    return MISSING


def _check_performance_acceptance(
    manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[str]:
    """Check an opt-in 10-second cold or declared resident-runner contract."""

    found = _performance_acceptance_record(manifest, policy)
    if found is None:
        return []
    record, label = found
    spec = _mapping(policy.get("performance_acceptance"), "policy.performance_acceptance")
    failures: list[str] = []
    for field in spec.get("required_fields", []):
        if field not in record:
            failures.append(f"{label} missing {field!r}")

    record_types = spec.get("record_types", ())
    record_type = record.get("type", record.get("record_type"))
    if record_type is not None and record_type not in record_types:
        failures.append(f"{label}.type is not a supported performance acceptance record")

    simulated: float | None = None
    wall: float | None = None
    try:
        simulated = _finite_number(
            record.get("simulated_duration_s"),
            f"{label}.simulated_duration_s",
        )
        if simulated < float(spec.get("minimum_simulated_duration_s", 10.0)):
            failures.append(
                f"{label}.simulated_duration_s={simulated} is below minimum "
                f"{spec.get('minimum_simulated_duration_s', 10.0)}"
            )
    except ValueError as exc:
        failures.append(str(exc))
    try:
        wall = _finite_number(record.get("wall_time_s"), f"{label}.wall_time_s")
        if wall <= 0:
            failures.append(f"{label}.wall_time_s must be positive")
    except ValueError as exc:
        failures.append(str(exc))

    scope = record.get("wall_time_scope")
    configured_scopes = spec.get("allowed_wall_time_scopes")
    if isinstance(configured_scopes, Sequence) and not isinstance(
        configured_scopes, (str, bytes)
    ):
        allowed_scopes = tuple(str(item) for item in configured_scopes)
    else:
        allowed_scopes = (
            str(spec.get("wall_time_scope", "startup+simulation+output")),
        )
    if scope not in allowed_scopes:
        failures.append(
            f"{label}.wall_time_scope must be one of {allowed_scopes!r}"
        )

    supplied_rtf: float | None = None
    try:
        supplied_rtf = _finite_number(record.get("real_time_factor"), f"{label}.real_time_factor")
        if supplied_rtf < 0:
            failures.append(f"{label}.real_time_factor must be non-negative")
    except ValueError as exc:
        failures.append(str(exc))
    if simulated is not None and wall is not None and wall > 0:
        # The supplied field is evidence for inspection only.  The gate's
        # decision always uses this recomputation from measured durations.
        measured_rtf = simulated / wall
        if supplied_rtf is not None and not math.isclose(
            supplied_rtf, measured_rtf, rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            failures.append(
                f"{label}.real_time_factor={supplied_rtf} disagrees with "
                f"simulated_duration_s/wall_time_s={measured_rtf}"
            )
        minimum_rtf = float(spec.get("minimum_real_time_factor", 0.5))
        if measured_rtf < minimum_rtf:
            failures.append(
                f"{label} measured real-time factor {measured_rtf} is below minimum {minimum_rtf}"
            )

    production = spec.get("production", {})
    if not isinstance(production, Mapping):
        failures.append("policy.performance_acceptance.production must be an object")
        return failures
    production_types = production.get("record_types", record_types)
    production_declared = (
        label == "performance_acceptance"
        or record.get("production") is True
        or record_type in production_types
    )
    if not production_declared:
        return failures

    required_execution = production.get("required_execution", {})
    if not isinstance(required_execution, Mapping):
        failures.append("policy.performance_acceptance.production.required_execution must be an object")
        return failures
    locations = production.get("execution_locations", {})
    if not isinstance(locations, Mapping):
        failures.append("policy.performance_acceptance.production.execution_locations must be an object")
        locations = {}
    requested_locations = locations.get("requested", ["requested_execution", "requested_backends"])
    effective_locations = locations.get("effective", ["effective_execution", "effective_backends"])
    if not isinstance(requested_locations, Sequence) or isinstance(requested_locations, (str, bytes)):
        requested_locations = ["requested_execution", "requested_backends"]
    if not isinstance(effective_locations, Sequence) or isinstance(effective_locations, (str, bytes)):
        effective_locations = ["effective_execution", "effective_backends"]
    for field, expected in required_execution.items():
        requested = MISSING
        effective = MISSING
        for location in requested_locations:
            value = _acceptance_value(record, manifest, str(location), str(field))
            if value is not MISSING:
                requested = value
                break
        for location in effective_locations:
            value = _acceptance_value(record, manifest, str(location), str(field))
            if value is not MISSING:
                effective = value
                break
        if requested is MISSING:
            failures.append(f"{label} missing requested execution field {field!r}")
        if effective is MISSING:
            failures.append(f"{label} missing effective execution field {field!r}")
        if requested is not MISSING and requested != expected:
            failures.append(
                f"{label} requested {field}={requested!r}, expected {expected!r}"
            )
        if effective is not MISSING and effective != expected:
            failures.append(
                f"{label} effective {field}={effective!r}, expected {expected!r}"
            )

    components = record.get("components", manifest.get("components"))
    if not isinstance(components, Mapping):
        failures.append(f"{label} requires components for production no-fallback acceptance")
    else:
        stub_field = str(production.get("stub_field", "is_stub"))
        for component in production.get("required_components", []):
            value = components.get(str(component))
            if not isinstance(value, Mapping):
                failures.append(f"{label} missing production component {component!r}")
            elif value.get(stub_field) is not False:
                failures.append(
                    f"{label} component {component!r}.{stub_field} must be false"
                )
    return failures


def _check_benchmark(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _mapping(policy.get("benchmark"), "policy.benchmark")
    try:
        benchmark = _mapping(manifest.get("benchmark"), "manifest.benchmark")
    except ValueError as exc:
        return [str(exc)]
    raw_acceptance_spec = policy.get("performance_acceptance", {})
    acceptance_spec = (
        raw_acceptance_spec if isinstance(raw_acceptance_spec, Mapping) else {}
    )
    benchmark_type = benchmark.get("type", benchmark.get("record_type"))
    if benchmark_type in acceptance_spec.get("record_types", ()):
        # A typed full-run acceptance record has its own compact contract;
        # do not force it to duplicate component-benchmark fields such as
        # warmup/sample counts and throughput.
        return []
    # Warp manifests may use the required ``benchmark`` member as a
    # namespaced container for ``cold_jit`` and ``steady`` records.  Those
    # records are validated by the conditional Warp check below; do not force
    # them through the single-run benchmark schema as well.
    if (
        _renderer_backend(manifest) == "warp_cuda"
        and "cold_jit" in benchmark
        and "steady" in benchmark
        and "name" not in benchmark
    ):
        return []
    for field in spec.get("required_fields", []):
        if field not in benchmark:
            failures.append(f"benchmark missing {field!r}")
    for field in ("name", "version", "backend", "device", "dtype", "timestamp_utc", "command"):
        if field in benchmark and (
            not isinstance(benchmark[field], str) or not benchmark[field].strip()
        ):
            failures.append(f"benchmark.{field} must be non-empty text")
    for field, minimum in (
        ("warmup_iterations", int(spec.get("minimum_warmup_iterations", 1))),
        ("measurement_iterations", int(spec.get("minimum_measurement_iterations", 3))),
        ("sample_count", int(spec.get("minimum_measurement_iterations", 3))),
    ):
        try:
            value = _finite_number(benchmark.get(field), f"benchmark.{field}", integer=True)
            if value < minimum:
                failures.append(f"benchmark.{field}={value} is below minimum {minimum}")
        except ValueError as exc:
            failures.append(str(exc))
    try:
        sample_count = int(_finite_number(benchmark.get("sample_count"), "benchmark.sample_count", integer=True))
        measurements = int(
            _finite_number(
                benchmark.get("measurement_iterations"),
                "benchmark.measurement_iterations",
                integer=True,
            )
        )
        if sample_count < measurements:
            failures.append("benchmark.sample_count must be at least measurement_iterations")
    except ValueError:
        pass
    for field in ("simulated_duration_s", "wall_time_s"):
        try:
            value = _finite_number(benchmark.get(field), f"benchmark.{field}")
            if value <= 0:
                failures.append(f"benchmark.{field} must be positive")
        except ValueError as exc:
            failures.append(str(exc))
    throughput = benchmark.get("throughput")
    if not isinstance(throughput, Mapping):
        failures.append("benchmark.throughput must be an object")
    else:
        try:
            rate = _finite_number(throughput.get("value"), "benchmark.throughput.value")
            if rate <= 0:
                failures.append("benchmark.throughput.value must be positive")
        except ValueError as exc:
            failures.append(str(exc))
        if not str(throughput.get("unit", "")).strip():
            failures.append("benchmark.throughput.unit must be non-empty")
    sync = benchmark.get("synchronization")
    if spec.get("require_synchronization_metadata", True) and not isinstance(sync, (Mapping, bool)):
        failures.append("benchmark.synchronization must be a boolean or object")
        sync = None
    device = str(benchmark.get("device", "")).lower()
    gpu_device = device.startswith("cuda") or "gpu" in device
    preflight = manifest.get("preflight")
    if gpu_device and not (isinstance(preflight, Mapping) and preflight.get("gpu_requested") is True):
        failures.append("GPU benchmark device requires preflight.gpu_requested=true")
    if gpu_device and spec.get("require_gpu_synchronization", True):
        synchronized = False
        if sync is True:
            synchronized = True
        elif isinstance(sync, Mapping):
            synchronized = sync.get("before") is True and sync.get("after") is True
        if not synchronized:
            failures.append("GPU benchmark must record synchronization before and after measurement")
    return failures


def _backend_label(value: Any) -> str | None:
    """Return a normalized backend label when ``value`` is explicit text."""

    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _check_backend_consistency(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    """Reject mismatched explicit requested/effective backend pairs."""

    failures: list[str] = []
    spec = _mapping(policy.get("backend_consistency"), "policy.backend_consistency")

    def compare(container: Mapping[str, Any], pair: Sequence[Any], label: str) -> None:
        if len(pair) != 2:
            return
        requested_key, effective_key = str(pair[0]), str(pair[1])
        requested = _backend_label(container.get(requested_key))
        effective = _backend_label(container.get(effective_key))
        if requested is not None and effective is not None and requested != effective:
            failures.append(
                f"{label}: requested {requested_key}={container.get(requested_key)!r} "
                f"differs from effective {effective_key}={container.get(effective_key)!r}"
            )

    for pair in spec.get("direct_pairs", []):
        compare(manifest, pair, "backend consistency")

    for key in ("renderer", "renderer_backend"):
        nested = manifest.get(key)
        if isinstance(nested, Mapping):
            for pair in spec.get("nested_renderer_pairs", []):
                compare(nested, pair, f"{key} backend consistency")

    for requested_key, effective_key in (
        ("requested_execution", "effective_execution"),
        ("requested_backends", "effective_backends"),
    ):
        requested = manifest.get(requested_key)
        effective = manifest.get(effective_key)
        if not isinstance(requested, Mapping) or not isinstance(effective, Mapping):
            continue
        for field in ("renderer_backend", "renderer", "render_backend"):
            compare(
                {"requested": requested.get(field), "effective": effective.get(field)},
                (("requested", "effective")),
                f"{requested_key}/{effective_key}.{field}",
            )
    return failures


def _renderer_backend(manifest: Mapping[str, Any]) -> str | None:
    """Find the explicit renderer backend without inferring one from timing."""

    for key in ("renderer_backend", "effective_renderer_backend", "renderer_backend_effective"):
        value = _backend_label(manifest.get(key))
        if value is not None:
            return value
    renderer = manifest.get("renderer")
    if isinstance(renderer, Mapping):
        for key in ("backend", "effective_backend", "renderer_backend"):
            value = _backend_label(renderer.get(key))
            if value is not None:
                return value
    return None


def _verify_declared_file_hash(
    artifact: Path,
    declaration: Mapping[str, Any],
    required_fields: Sequence[Any],
    label: str,
) -> list[str]:
    failures: list[str] = []
    for field in required_fields:
        if field not in declaration:
            failures.append(f"{label} missing {field!r}")
    path = _resolve_declared_path(artifact, declaration.get("path"))
    if path is None:
        failures.append(f"{label}.path does not resolve to a file: {declaration.get('path')!r}")
        return failures
    declared = str(declaration.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared):
        failures.append(f"{label}.sha256 must be a 64-character lowercase SHA-256 digest")
        return failures
    actual = _sha256_file(path)
    if declared != actual:
        failures.append(f"{label} SHA-256 mismatch: declared {declared}, actual {actual}")
    return failures


def _benchmark_record_failures(
    record: Any,
    policy_spec: Mapping[str, Any],
    label: str,
    *,
    minimum_measurements: int | None = None,
    require_gpu_sync: bool = True,
) -> list[str]:
    """Validate one benchmark record, including timing metadata and sync."""

    failures: list[str] = []
    try:
        benchmark = _mapping(record, label)
    except ValueError as exc:
        return [str(exc)]
    required_fields = policy_spec.get("required_fields", [])
    for field in required_fields:
        if field not in benchmark:
            failures.append(f"{label} missing {field!r}")
    for field in ("name", "version", "backend", "device", "dtype", "timestamp_utc", "command"):
        if field in benchmark and (
            not isinstance(benchmark[field], str) or not benchmark[field].strip()
        ):
            failures.append(f"{label}.{field} must be non-empty text")
    minimum = int(policy_spec.get("minimum_measurement_iterations", 3))
    if minimum_measurements is not None:
        minimum = max(minimum, int(minimum_measurements))
    for field, field_minimum in (
        ("warmup_iterations", int(policy_spec.get("minimum_warmup_iterations", 1))),
        ("measurement_iterations", minimum),
        ("sample_count", minimum),
    ):
        try:
            value = _finite_number(benchmark.get(field), f"{label}.{field}", integer=True)
            if value < field_minimum:
                failures.append(f"{label}.{field}={value} is below minimum {field_minimum}")
        except ValueError as exc:
            failures.append(str(exc))
    try:
        sample_count = int(
            _finite_number(benchmark.get("sample_count"), f"{label}.sample_count", integer=True)
        )
        measurements = int(
            _finite_number(
                benchmark.get("measurement_iterations"),
                f"{label}.measurement_iterations",
                integer=True,
            )
        )
        if sample_count < measurements:
            failures.append(f"{label}.sample_count must be at least measurement_iterations")
    except ValueError:
        pass
    for field in ("simulated_duration_s", "wall_time_s"):
        try:
            value = _finite_number(benchmark.get(field), f"{label}.{field}")
            if value <= 0:
                failures.append(f"{label}.{field} must be positive")
        except ValueError as exc:
            failures.append(str(exc))
    throughput = benchmark.get("throughput")
    if not isinstance(throughput, Mapping):
        failures.append(f"{label}.throughput must be an object")
    else:
        try:
            rate = _finite_number(throughput.get("value"), f"{label}.throughput.value")
            if rate <= 0:
                failures.append(f"{label}.throughput.value must be positive")
        except ValueError as exc:
            failures.append(str(exc))
        if not str(throughput.get("unit", "")).strip():
            failures.append(f"{label}.throughput.unit must be non-empty")
    sync = benchmark.get("synchronization")
    if policy_spec.get("require_synchronization_metadata", True) and not isinstance(sync, (Mapping, bool)):
        failures.append(f"{label}.synchronization must be a boolean or object")
        sync = None
    device = str(benchmark.get("device", "")).lower()
    gpu_device = device.startswith("cuda") or "gpu" in device
    if gpu_device and require_gpu_sync:
        synchronized = sync is True or (
            isinstance(sync, Mapping) and sync.get("before") is True and sync.get("after") is True
        )
        if not synchronized:
            failures.append(f"{label} GPU benchmark must record synchronization before and after measurement")
    return failures


def _check_warp_cuda(artifact: Path, manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    """Audit the conditional NVIDIA Warp CUDA renderer contract."""

    spec = _mapping(policy.get("warp_cuda"), "policy.warp_cuda")
    expected_backend = _backend_label(spec.get("renderer_backend"))
    if _renderer_backend(manifest) != expected_backend:
        return []

    failures: list[str] = []
    for field in spec.get("required_manifest_fields", []):
        if field not in manifest:
            failures.append(f"Warp manifest missing {field!r}")
    requested = _backend_label(manifest.get("requested_renderer_backend"))
    effective = _backend_label(manifest.get("effective_renderer_backend"))
    if requested != expected_backend:
        failures.append(f"Warp requested renderer must be {expected_backend!r}, got {requested!r}")
    if effective != expected_backend:
        failures.append(f"Warp effective renderer must be {expected_backend!r}, got {effective!r}")
    raw_warp = manifest.get("warp")
    if isinstance(raw_warp, Mapping):
        warp = dict(raw_warp)
    else:
        # Keep the schema decoupled from a renderer implementation.  A
        # backend may publish the same contract at the manifest top level
        # while it is still settling its runtime-specific nesting.
        warp = {}
    aliases: dict[str, tuple[str, ...]] = {
        "version": ("warp_version",),
        "gpu": ("gpu",),
        "scene": ("scene",),
        "config": ("config",),
        "exact_columns": ("exact_columns",),
        "benchmarks": ("benchmarks",),
        "analytic_reference": ("analytic_reference",),
        "resources": ("warp_resources",),
    }
    for field, candidates in aliases.items():
        if field in warp:
            continue
        for candidate in candidates:
            if candidate in manifest:
                warp[field] = manifest[candidate]
                break
    if "gpu" not in warp and any(key in manifest for key in ("gpu_name", "gpu_arch", "compute_capability")):
        warp["gpu"] = {
            "name": manifest.get("gpu_name"),
            "arch": manifest.get("gpu_arch"),
            "compute_capability": manifest.get("compute_capability"),
        }
    if "scene" not in warp and ("scene_path" in manifest or "scene_hash" in manifest):
        warp["scene"] = {"path": manifest.get("scene_path"), "sha256": manifest.get("scene_hash")}
    if "config" not in warp and isinstance(manifest.get("config"), Mapping):
        warp["config"] = manifest["config"]
    if "exact_columns" not in warp:
        exact_hashes = manifest.get("exact_column_hashes")
        exact_fields = manifest
        exact_file = manifest.get("exact_columns_file")
        if not isinstance(exact_hashes, Mapping):
            column_hashes = manifest.get("column_hashes")
            exact_hashes = column_hashes.get("exact") if isinstance(column_hashes, Mapping) else None
            exact_fields = exact_hashes if isinstance(exact_hashes, Mapping) else {}
            if isinstance(exact_hashes, Mapping):
                exact_file = exact_hashes.get("file", exact_file)
        if isinstance(exact_hashes, Mapping):
            warp["exact_columns"] = {
                "file": exact_file,
                "shape_column": exact_fields.get("shape_column", "shape"),
                "direction_column": exact_fields.get("direction_column", "direction"),
                "shape_hash": exact_hashes.get("shape_hash", exact_hashes.get("shape")),
                "direction_hash": exact_hashes.get("direction_hash", exact_hashes.get("direction")),
                "algorithm": exact_hashes.get("algorithm", spec.get("required_hash_algorithm")),
            }
    if "benchmarks" not in warp:
        if isinstance(manifest.get("benchmarks"), Mapping):
            warp["benchmarks"] = manifest["benchmarks"]
        elif isinstance(manifest.get("benchmark"), Mapping):
            benchmark_container = manifest["benchmark"]
            if "cold_jit" in benchmark_container or "steady" in benchmark_container:
                warp["benchmarks"] = benchmark_container
        elif "cold_jit_benchmark" in manifest or "steady_benchmark" in manifest:
            warp["benchmarks"] = {
                "cold_jit": manifest.get("cold_jit_benchmark"),
                "steady": manifest.get("steady_benchmark"),
            }
    if "analytic_reference" not in warp and isinstance(manifest.get("analytic_reference"), Mapping):
        warp["analytic_reference"] = manifest["analytic_reference"]
    if "analytic_reference" not in warp and isinstance(manifest.get("equivalence"), Mapping):
        equivalence = manifest["equivalence"]
        candidate = equivalence.get("analytic_reference")
        if isinstance(candidate, Mapping):
            warp["analytic_reference"] = candidate
    if "resources" not in warp and isinstance(manifest.get("resources"), Mapping):
        warp["resources"] = manifest["resources"]
    for field in spec.get("required_warp_fields", []):
        if field not in warp:
            failures.append(f"manifest.warp missing {field!r}")

    version = warp.get("version")
    if not isinstance(version, str) or not version.strip():
        failures.append("manifest.warp.version must be non-empty text")
    gpu = _mapping_or_empty(warp.get("gpu"), "manifest.warp.gpu", failures)
    for field in spec.get("required_gpu_fields", []):
        if field not in gpu or not str(gpu.get(field)).strip():
            failures.append(f"manifest.warp.gpu.{field} must be non-empty")

    scene = _mapping_or_empty(warp.get("scene"), "manifest.warp.scene", failures)
    failures.extend(
        _verify_declared_file_hash(
            artifact,
            scene,
            spec.get("required_scene_fields", []),
            "manifest.warp.scene",
        )
    )
    warp_config = _mapping_or_empty(warp.get("config"), "manifest.warp.config", failures)
    failures.extend(
        _verify_declared_file_hash(
            artifact,
            warp_config,
            spec.get("required_config_fields", []),
            "manifest.warp.config",
        )
    )

    exact = _mapping_or_empty(warp.get("exact_columns"), "manifest.warp.exact_columns", failures)
    for field in spec.get("required_exact_column_fields", []):
        if field not in exact:
            failures.append(f"manifest.warp.exact_columns missing {field!r}")
    if exact.get("algorithm") != spec.get("required_hash_algorithm"):
        failures.append(
            "manifest.warp.exact_columns.algorithm must be "
            f"{spec.get('required_hash_algorithm')!r}"
        )
    exact_path = _resolve_declared_path(artifact, exact.get("file"))
    if exact_path is None:
        failures.append(f"Warp exact-column file does not resolve: {exact.get('file')!r}")
    else:
        for column_field, hash_field in (
            ("shape_column", "shape_hash"),
            ("direction_column", "direction_hash"),
        ):
            column = exact.get(column_field)
            declared = str(exact.get(hash_field, "")).lower()
            if not isinstance(column, str) or not column.strip():
                failures.append(f"manifest.warp.exact_columns.{column_field} must be non-empty text")
                continue
            if not re.fullmatch(r"[0-9a-f]{64}", declared):
                failures.append(f"manifest.warp.exact_columns.{hash_field} must be a SHA-256 digest")
                continue
            try:
                actual = column_hash(exact_path, column)
            except EngineeringGateError as exc:
                failures.append(str(exc))
                continue
            if actual != declared:
                failures.append(
                    f"Warp exact-column {column} hash mismatch: declared {declared}, actual {actual}"
                )

    benchmark_policy = _mapping(policy.get("benchmark"), "policy.benchmark")
    benchmarks = _mapping_or_empty(warp.get("benchmarks"), "manifest.warp.benchmarks", failures)
    records: dict[str, Mapping[str, Any]] = {}
    for field in spec.get("benchmark_fields", []):
        record = benchmarks.get(field)
        if not isinstance(record, Mapping):
            failures.append(f"manifest.warp.benchmarks.{field} must be an object")
            continue
        records[field] = record
        failures.extend(_benchmark_record_failures(record, benchmark_policy, f"Warp benchmark {field}"))
    for field, marker_spec in (
        ("cold_jit", spec.get("cold_jit_marker", {})),
        ("steady", spec.get("steady_marker", {})),
    ):
        record = records.get(field)
        if record is None or not isinstance(marker_spec, Mapping):
            continue
        marker_field = marker_spec.get("field")
        marker_value = marker_spec.get("value")
        if record.get(marker_field) is not marker_value:
            failures.append(
                f"Warp benchmark {field}.{marker_field} must be {marker_value!r}"
            )
    steady = records.get("steady")
    if steady is not None:
        try:
            measured = _finite_number(
                steady.get("measurement_iterations"),
                "Warp benchmark steady.measurement_iterations",
                integer=True,
            )
            if measured < int(spec.get("minimum_steady_measurements", 3)):
                failures.append(
                    "Warp steady benchmark measurement_iterations is below configured minimum "
                    f"{spec.get('minimum_steady_measurements')}"
                )
        except ValueError as exc:
            failures.append(str(exc))

    reference = _mapping_or_empty(
        warp.get("analytic_reference"), "manifest.warp.analytic_reference", failures
    )
    if reference.get("status") != spec.get("analytic_reference_status", "PASS"):
        failures.append(
            "Warp analytic_reference.status must be "
            f"{spec.get('analytic_reference_status', 'PASS')!r}"
        )
    for field in spec.get("analytic_reference_fields", []):
        if field not in reference:
            failures.append(f"manifest.warp.analytic_reference missing {field!r}")
    if not str(reference.get("reference_id", "")).strip():
        failures.append("Warp analytic_reference.reference_id must be non-empty")
    tolerance = reference.get("tolerance")
    if not isinstance(tolerance, Mapping):
        failures.append("Warp analytic_reference.tolerance must be an object")
        tolerance = {}
    configured_tolerances = _mapping(
        spec.get("analytic_reference_tolerances"),
        "policy.warp_cuda.analytic_reference_tolerances",
    )
    for metric in spec.get("analytic_reference_metrics", []):
        try:
            integer = metric == "mismatch_count"
            observed = _finite_number(
                reference.get(metric), f"Warp analytic_reference.{metric}", integer=integer
            )
            allowed = _finite_number(
                tolerance.get(metric), f"Warp analytic_reference.tolerance.{metric}", integer=integer
            )
            configured = _finite_number(
                configured_tolerances.get(metric),
                f"policy.warp_cuda.analytic_reference_tolerances.{metric}",
            )
            if observed < 0 or allowed < 0:
                failures.append(f"Warp analytic_reference {metric} values must be non-negative")
            elif allowed > configured:
                failures.append(
                    f"Warp analytic_reference tolerance {metric}={allowed} exceeds policy {configured}"
                )
            elif observed > allowed:
                failures.append(
                    f"Warp analytic_reference {metric}={observed} exceeds tolerance {allowed}"
                )
        except ValueError as exc:
            failures.append(str(exc))

    warp_resources = _mapping_or_empty(
        warp.get("resources"), "manifest.warp.resources", failures
    )
    usage = warp_resources.get("usage", warp_resources)
    if not isinstance(usage, Mapping):
        failures.append("manifest.warp.resources.usage must be an object")
        usage = {}
    for field in spec.get("required_resource_usage_fields", []):
        try:
            value = _finite_number(usage.get(field), f"manifest.warp.resources.{field}", integer=True)
            if value < 0:
                failures.append(f"manifest.warp.resources.{field} must be non-negative")
        except ValueError as exc:
            failures.append(str(exc))
    resources = manifest.get("resources")
    if isinstance(resources, Mapping) and isinstance(resources.get("limits"), Mapping):
        limits = resources["limits"]
        for usage_field, limit_field in (
            ("peak_host_rss_bytes", "host_rss_max_bytes"),
            ("peak_cuda_allocated_bytes", "cuda_allocated_max_bytes"),
            ("peak_cuda_reserved_bytes", "cuda_reserved_max_bytes"),
        ):
            if usage_field not in usage or limit_field not in limits:
                continue
            try:
                usage_value = _finite_number(
                    usage[usage_field],
                    f"manifest.warp.resources.{usage_field}",
                    integer=True,
                )
                limit_value = _finite_number(
                    limits[limit_field],
                    f"resources.limits.{limit_field}",
                    integer=True,
                )
                if usage_value > limit_value:
                    failures.append(
                        f"Warp resource {usage_field} exceeds resources.limits.{limit_field}"
                    )
            except ValueError as exc:
                failures.append(str(exc))
    return failures


def audit_artifact(
    artifact: str | Path,
    *,
    policy: str | Path | Mapping[str, Any] = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    """Audit one artifact directory without modifying it.

    The returned report always contains a top-level ``status``.  ``PASS``
    means every engineering check passed; ``FAIL`` includes one or more
    human-readable ``failures``.  Missing or malformed manifests are handled
    as failed reports rather than uncaught tracebacks.
    """

    artifact_path = Path(artifact).expanduser()

    def report(checks: dict[str, dict[str, Any]], failures: list[str]) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact": str(artifact_path),
            "read_only": True,
            "status": "FAIL" if failures else "PASS",
            "checks": checks,
            "failures": failures,
        }

    try:
        policy_value = load_policy(policy) if not isinstance(policy, Mapping) else dict(policy)
    except EngineeringGateError as exc:
        return report(
            {"policy": {"status": "FAIL", "failures": [str(exc)]}},
            [f"policy: {exc}"],
        )

    checks: dict[str, dict[str, Any]] = {}
    all_failures: list[str] = []

    def run_check(name: str, function: Any) -> None:
        try:
            failures = list(function())
        except (EngineeringGateError, OSError, ValueError, TypeError) as exc:
            failures = [str(exc)]
        status = _status(failures)
        checks[name] = {"status": status, "failures": failures}
        all_failures.extend(f"{name}: {failure}" for failure in failures)

    if not artifact_path.is_dir():
        all_failures.append(f"artifact: directory does not exist: {artifact_path}")
        checks["artifact"] = {"status": "FAIL", "failures": [all_failures[-1]]}
        return report(checks, all_failures)

    manifest_path = artifact_path / "manifest.json"
    manifest: Mapping[str, Any] | None = None
    if not manifest_path.is_file():
        failure = f"manifest.json is missing from {artifact_path}"
        checks["manifest_schema"] = {"status": "FAIL", "failures": [failure]}
        all_failures.append(f"manifest_schema: {failure}")
    else:
        try:
            loaded = _load_json(manifest_path, "manifest")
            manifest = _mapping(loaded, "manifest")
            run_check("manifest_schema", lambda: _check_manifest_schema(manifest, policy_value))
        except (EngineeringGateError, ValueError) as exc:
            failure = str(exc)
            checks["manifest_schema"] = {"status": "FAIL", "failures": [failure]}
            all_failures.append(f"manifest_schema: {failure}")

    if manifest is not None:
        run_check("unit_tests", lambda: _check_unit_tests(manifest, policy_value))
        run_check("preflight", lambda: _check_preflight(manifest, policy_value))
        run_check("authority", lambda: _check_authority(manifest, policy_value))
        run_check("determinism", lambda: _check_determinism(artifact_path, manifest, policy_value))
        run_check("column_hashes", lambda: _check_column_hashes(artifact_path, manifest, policy_value))
        run_check("resources", lambda: _check_resources(manifest, policy_value))
        run_check("artifact_budget", lambda: _check_artifact_budget(artifact_path, manifest, policy_value))
        run_check("equivalence", lambda: _check_equivalence(manifest, policy_value))
        if _performance_acceptance_record(manifest, policy_value) is not None:
            run_check(
                "performance_acceptance",
                lambda: _check_performance_acceptance(manifest, policy_value),
            )
        run_check("benchmark", lambda: _check_benchmark(manifest, policy_value))
        run_check("backend_consistency", lambda: _check_backend_consistency(manifest, policy_value))
        run_check("warp_cuda", lambda: _check_warp_cuda(artifact_path, manifest, policy_value))

    return report(checks, all_failures)


# Short aliases are kept for callers that prefer a verb over the noun used in
# the command-line documentation.
audit = audit_artifact
compute_column_hash = column_hash


def _print_human(report: Mapping[str, Any]) -> None:
    print(f"Engineering gate: {report.get('status', 'FAIL')}")
    print(f"Artifact: {report.get('artifact', '')}")
    for name, check in report.get("checks", {}).items():
        status = check.get("status", "FAIL") if isinstance(check, Mapping) else "FAIL"
        print(f"  {status:<4} {name}")
        if isinstance(check, Mapping):
            for failure in check.get("failures", []):
                print(f"       - {failure}")
    if report.get("status") == "FAIL" and not report.get("checks"):
        for failure in report.get("failures", []):
            print(f"  - {failure}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, help="artifact directory containing manifest.json")
    parser.add_argument("--artifact", dest="artifact_option", type=Path, help="artifact directory (option form)")
    parser.add_argument(
        "--config", "--policy", dest="config", type=Path, default=DEFAULT_POLICY_PATH,
        help="engineering gate policy JSON",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="print the machine-readable report")
    args = parser.parse_args(argv)
    artifact = args.artifact_option or args.artifact
    if artifact is None:
        parser.error("an artifact directory is required (positional or --artifact)")
    report = audit_artifact(artifact, policy=args.config)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
