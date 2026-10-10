#!/usr/bin/env python3
"""Plan and audit repeatable digital-fly experiments.

This module is intentionally a small, dependency-free orchestration layer.  It
does not implement a new fly model or a new validator.  Instead it combines
the project's existing read-only preflight gate, declarative protocol, and
``run_digital_fly.py`` launcher into a safe command-line toolbox.

The important boundary is that a plan is data, not shell code.  Launch
commands are built as argv arrays and are executed with ``shell=False``.  A
launch is always wrapped in a 5 GiB cgroup memory limit and carries the
launcher-controlled 15.5 GiB VRAM ceiling.  Artifact-producing paths are
restricted to this project's ``artifacts/`` tree or ``/data``.

Typical use::

    python3 scripts/experiment_toolbox.py preflight --json
    python3 scripts/experiment_toolbox.py validate --json
    python3 scripts/experiment_toolbox.py plan --replicates 1 --json
    python3 scripts/experiment_toolbox.py matrix --replicates 2 --json
    python3 scripts/experiment_toolbox.py launch --trial-id trial-01-001-01
    python3 scripts/experiment_toolbox.py inventory --json
    python3 scripts/experiment_toolbox.py status --json

The default protocol is the locked escape protocol, but all paths, tiers, and
conditions remain declarative.  There are no behavior presets in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_TOOLBOX_CONFIG = ROOT / "config" / "experiment_toolbox.json"
DEFAULT_PROTOCOL_CONFIG = ROOT / "config" / "escape_experiment_protocol.json"
DEFAULT_RUNTIME_CONFIG = ROOT / "config" / "free_escape_experiment.json"
# Inventory/status cover the whole project artifact tree by default.  Generated
# toolbox launches still land in its dedicated ``experiment-toolbox`` child.
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts"
DATA_ARTIFACT_ROOT = Path("/data/artifacts/by-project/malecns-gpu")

SCHEMA_VERSION = "experiment-toolbox-1"
PLAN_SCHEMA_VERSION = "experiment-plan-1"
MATRIX_SCHEMA_VERSION = "experiment-seed-matrix-1"
INVENTORY_SCHEMA_VERSION = "experiment-artifact-inventory-1"
STATUS_SCHEMA_VERSION = "experiment-status-1"

GIB = 1024**3
HOST_RAM_LIMIT_GIB = 5.0
MAX_VRAM_GIB = 15.5
HOST_RAM_LIMIT_BYTES = int(HOST_RAM_LIMIT_GIB * GIB)
RESOURCE_WRAPPER = (
    "systemd-run",
    "--user",
    "--scope",
    "--property=MemoryMax=5G",
    "--",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
class ToolboxError(ValueError):
    """Base class for malformed configuration or unsafe launch requests."""


class ConfigurationError(ToolboxError):
    """Raised when protocol/toolbox configuration fails closed."""


class ResourcePolicyError(ToolboxError):
    """Raised when a launch cannot carry the required resource limits."""


def _json_safe(value: Any) -> Any:
    """Convert common path/tuple/set values to strict JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


def _read_json(path: str | Path) -> Mapping[str, Any]:
    candidate = Path(path).expanduser()
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"JSON file does not exist: {candidate}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {candidate}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read {candidate}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"top level of {candidate} must be a JSON object")
    return value


def load_toolbox_config(path: str | Path = DEFAULT_TOOLBOX_CONFIG) -> dict[str, Any]:
    """Load the toolbox config without mutating it."""

    return dict(_read_json(path))


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_CONFIG) -> dict[str, Any]:
    """Load the locked protocol JSON."""

    return dict(_read_json(path))


def load_runtime_config(path: str | Path = DEFAULT_RUNTIME_CONFIG) -> dict[str, Any]:
    """Load the selected digital-fly runtime configuration."""

    return dict(_read_json(path))


def _resolved(path: str | Path, *, root: Path = ROOT, base: Path | None = None) -> Path:
    """Resolve a config reference while keeping relative paths predictable.

    Project-root references (``config/foo.json``) take precedence.  A local
    path beside a temporary config remains useful for tests and for copied
    protocol bundles when the project-root reference does not exist.
    """

    raw = str(path)
    if "\x00" in raw:
        raise ToolboxError("path contains NUL")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    root_candidate = (root / candidate).resolve(strict=False)
    if root_candidate.exists() or base is None:
        return root_candidate
    return (base / candidate).resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def allowed_artifact_path(
    path: str | Path,
    *,
    root: Path = ROOT,
    extra_roots: Iterable[str | Path] = (),
) -> Path:
    """Resolve a write target and fail unless it is an approved artifact.

    Reading arbitrary protocol/config files is allowed; this restriction is
    only for paths that the toolbox itself or a generated workload may write.
    """

    candidate = Path(path).expanduser()
    if "\x00" in str(path):
        raise ToolboxError("artifact path contains NUL")
    resolved = candidate.resolve(strict=False)
    roots = [(root / "artifacts").resolve(strict=False), Path("/data").resolve()]
    for item in extra_roots:
        extra = Path(item).expanduser()
        if not extra.is_absolute():
            extra = root / extra
        extra = extra.resolve(strict=False)
        if not any(_within(extra, approved) for approved in roots):
            raise ToolboxError(f"extra artifact root is outside approved roots: {extra}")
        roots.append(extra)
    if not any(_within(resolved, allowed) for allowed in roots):
        allowed = ", ".join(str(item) for item in roots)
        raise ToolboxError(f"write path is outside approved artifact roots: {resolved} (allowed: {allowed})")
    return resolved


def _safe_identifier(value: Any, name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ToolboxError(f"{name} must be a non-empty identifier")
    text = str(value).strip()
    if not text or not _SAFE_ID.fullmatch(text):
        raise ToolboxError(f"{name} must contain only letters, numbers, '.', '_' or '-': {value!r}")
    return text


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ToolboxError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolboxError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ToolboxError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ToolboxError(f"{name} must be >= {minimum}")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ToolboxError(f"{name} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ToolboxError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolboxError(f"{name} must be an integer") from exc
    if number < minimum:
        raise ToolboxError(f"{name} must be >= {minimum}")
    return number


def _reference(config: Mapping[str, Any], name: str, default: Path, *, root: Path = ROOT) -> Path:
    paths = config.get("paths", {})
    if isinstance(paths, Mapping) and name in paths:
        return _resolved(paths[name], root=root)
    # Accept the short top-level spelling for small copied configs.
    if name in config:
        return _resolved(config[name], root=root)
    return default.resolve(strict=False)


def _configuration_shape_errors(config: Mapping[str, Any], *, root: Path = ROOT) -> tuple[list[str], list[str], dict[str, Path]]:
    errors: list[str] = []
    warnings: list[str] = []
    if config.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    paths: dict[str, Path] = {
        "protocol": _reference(config, "protocol", DEFAULT_PROTOCOL_CONFIG, root=root),
        "runtime": _reference(config, "runtime", DEFAULT_RUNTIME_CONFIG, root=root),
        "mission": _reference(config, "mission", ROOT / "config" / "scenarios.json", root=root),
        "launcher": _reference(config, "launcher", ROOT / "scripts" / "run_digital_fly.py", root=root),
        "validator": _reference(config, "validator", ROOT / "scripts" / "validate_escape_experiment.py", root=root),
    }
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"{name} path does not exist: {path}")

    policy = config.get("resource_policy")
    if not isinstance(policy, Mapping):
        errors.append("resource_policy must be a JSON object")
    else:
        try:
            ram = _finite_number(policy.get("host_ram_gib"), "resource_policy.host_ram_gib", minimum=0.01)
            if not math.isclose(ram, HOST_RAM_LIMIT_GIB, rel_tol=0.0, abs_tol=1.0e-12):
                errors.append(f"host RAM limit must be exactly {HOST_RAM_LIMIT_GIB:g} GiB (got {ram:g})")
        except ToolboxError as exc:
            errors.append(str(exc))
        try:
            vram = _finite_number(policy.get("max_vram_gib"), "resource_policy.max_vram_gib", minimum=0.01)
            if not math.isclose(vram, MAX_VRAM_GIB, rel_tol=0.0, abs_tol=1.0e-12):
                errors.append(f"VRAM limit must be exactly {MAX_VRAM_GIB:g} GiB (got {vram:g})")
        except ToolboxError as exc:
            errors.append(str(exc))
        wrapper = policy.get("host_wrapper")
        if not isinstance(wrapper, list) or not all(isinstance(token, str) for token in wrapper):
            errors.append("resource_policy.host_wrapper must be a list of strings")
        elif tuple(wrapper) != RESOURCE_WRAPPER:
            errors.append("resource_policy.host_wrapper must use systemd-run MemoryMax=5G")

    launch = config.get("launch", {})
    if not isinstance(launch, Mapping):
        errors.append("launch must be a JSON object")
    else:
        base_args = launch.get("base_args", [])
        if not isinstance(base_args, list) or not all(isinstance(token, str) for token in base_args):
            errors.append("launch.base_args must be a list of strings")
        else:
            for token in base_args:
                if "\x00" in token:
                    errors.append("launch.base_args contains a NUL byte")
                # Presets are deliberately not a supported toolbox feature.
                if token == "--preset" or token.startswith("--preset="):
                    errors.append("behavior presets are not supported; declare explicit launcher arguments")
        output_subdir = launch.get("output_subdir", "experiment-toolbox")
        try:
            _safe_identifier(output_subdir, "launch.output_subdir")
        except ToolboxError as exc:
            errors.append(str(exc))

    roots = config.get("artifact_roots", [])
    if roots is not None and (not isinstance(roots, list) or not all(isinstance(item, str) for item in roots)):
        errors.append("artifact_roots must be a list of strings")
    elif isinstance(roots, list):
        approved = [(root / "artifacts").resolve(strict=False), Path("/data").resolve()]
        for item in roots:
            try:
                candidate = _resolved(item, root=root)
            except ToolboxError as exc:
                errors.append(f"artifact_roots: {exc}")
                continue
            if not any(_within(candidate, allowed) for allowed in approved):
                errors.append(f"artifact root is outside project artifacts or /data: {candidate}")
    return errors, warnings, paths


def validate_protocol(protocol: Mapping[str, Any] | str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Validate protocol invariants without running a workload.

    The checks cover the parts needed to construct an auditable matrix.  The
    scientific endpoint checks remain in ``validate_escape_experiment.py``.
    """

    value: Mapping[str, Any] = _read_json(protocol) if isinstance(protocol, (str, Path)) else protocol
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["protocol must be a JSON object"], "warnings": []}
    if not str(value.get("schema_version", "")).strip():
        errors.append("protocol.schema_version is required")
    if not str(value.get("protocol_id", "")).strip():
        errors.append("protocol.protocol_id is required")

    conditions = value.get("conditions")
    condition_ids: list[str] = []
    if not isinstance(conditions, list) or not conditions:
        errors.append("protocol.conditions must be a non-empty list")
    else:
        for index, item in enumerate(conditions):
            if not isinstance(item, Mapping):
                errors.append(f"conditions[{index}] must be an object")
                continue
            try:
                condition_ids.append(_safe_identifier(item.get("id"), f"conditions[{index}].id"))
            except ToolboxError as exc:
                errors.append(str(exc))
            if not str(item.get("family", "")).strip():
                errors.append(f"conditions[{index}].family is required")
    if len(condition_ids) != len(set(condition_ids)):
        errors.append("condition ids must be unique")

    tiers = value.get("speed_tiers")
    tier_ids: list[str] = []
    if not isinstance(tiers, list) or not tiers:
        errors.append("protocol.speed_tiers must be a non-empty list")
    else:
        primary_count = 0
        for index, item in enumerate(tiers):
            if not isinstance(item, Mapping):
                errors.append(f"speed_tiers[{index}] must be an object")
                continue
            try:
                tier_ids.append(_safe_identifier(item.get("id"), f"speed_tiers[{index}].id"))
            except ToolboxError as exc:
                errors.append(str(exc))
            for key in ("speed_mps", "duration_ms", "impact_ms"):
                try:
                    _finite_number(item.get(key), f"speed_tiers[{index}].{key}", minimum=0.0)
                except ToolboxError as exc:
                    errors.append(str(exc))
            if bool(item.get("primary")):
                primary_count += 1
        if primary_count != 1:
            errors.append("speed_tiers must contain exactly one primary tier")
    if len(tier_ids) != len(set(tier_ids)):
        errors.append("speed tier ids must be unique")

    randomization = value.get("randomization")
    if not isinstance(randomization, Mapping):
        errors.append("protocol.randomization must be an object")
    else:
        try:
            _integer(randomization.get("master_seed"), "randomization.master_seed", minimum=0)
        except ToolboxError as exc:
            errors.append(str(exc))
        for key in ("digital_replicates_per_intact_condition_per_speed_tier", "exact_e14_baseline_replays"):
            try:
                _integer(randomization.get(key), f"randomization.{key}", minimum=1)
            except ToolboxError as exc:
                errors.append(str(exc))
        derivation = str(randomization.get("seed_derivation", ""))
        if "SHA256" not in derivation or "opaque_trial_id" not in derivation:
            errors.append("randomization.seed_derivation must document SHA256 and opaque_trial_id")

    lesions = value.get("lesions", [])
    if lesions is not None:
        if not isinstance(lesions, list):
            errors.append("protocol.lesions must be a list")
        else:
            lesion_ids: list[str] = []
            for index, item in enumerate(lesions):
                if not isinstance(item, Mapping):
                    errors.append(f"lesions[{index}] must be an object")
                    continue
                try:
                    lesion_ids.append(_safe_identifier(item.get("id"), f"lesions[{index}].id"))
                except ToolboxError as exc:
                    errors.append(str(exc))
            if len(lesion_ids) != len(set(lesion_ids)):
                errors.append("lesion ids must be unique")

    if not isinstance(value.get("forbidden_signals"), list) or not value.get("forbidden_signals"):
        warnings.append("protocol.forbidden_signals is empty or absent")
    tracks = value.get("tracks")
    if not isinstance(tracks, Mapping) or not isinstance(tracks.get("pure_autonomous"), Mapping):
        warnings.append("pure_autonomous track metadata is absent")
    report = {
        "schema_version": str(value.get("schema_version", "")),
        "protocol_id": str(value.get("protocol_id", "")),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "condition_ids": condition_ids,
        "tier_ids": tier_ids,
    }
    return report


def validate_configuration(
    config: Mapping[str, Any] | str | Path = DEFAULT_TOOLBOX_CONFIG,
    protocol: Mapping[str, Any] | str | Path | None = None,
    runtime: Mapping[str, Any] | str | Path | None = None,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate toolbox, protocol, and runtime configuration together."""

    config_path: Path | None = None
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve(strict=False)
        try:
            config_value = load_toolbox_config(config_path)
        except ToolboxError as exc:
            return {"schema_version": SCHEMA_VERSION, "valid": False, "errors": [str(exc)], "warnings": []}
    else:
        config_value = dict(config)
    errors, warnings, paths = _configuration_shape_errors(config_value, root=root)
    protocol_path = paths["protocol"] if protocol is None else None
    runtime_path = paths["runtime"] if runtime is None else None
    try:
        protocol_value = load_protocol(protocol_path) if protocol is None else (
            _read_json(protocol) if isinstance(protocol, (str, Path)) else dict(protocol)
        )
        protocol_report = validate_protocol(protocol_value, root=root)
        errors.extend(f"protocol: {item}" for item in protocol_report["errors"])
        warnings.extend(f"protocol: {item}" for item in protocol_report["warnings"])
    except ToolboxError as exc:
        protocol_value = {}
        protocol_report = {"valid": False, "errors": [str(exc)], "warnings": []}
        errors.extend(f"protocol: {item}" for item in protocol_report["errors"])
    try:
        runtime_value = load_runtime_config(runtime_path) if runtime is None else (
            _read_json(runtime) if isinstance(runtime, (str, Path)) else dict(runtime)
        )
        if not str(runtime_value.get("schema_version", "")).strip():
            errors.append("runtime: schema_version is required")
        if not isinstance(runtime_value.get("scenarios"), Mapping):
            errors.append("runtime: scenarios must be an object")
    except ToolboxError as exc:
        runtime_value = {}
        errors.append(f"runtime: {exc}")

    launch = config_value.get("launch", {})
    if isinstance(launch, Mapping):
        base_args = launch.get("base_args", [])
        if isinstance(base_args, list):
            # All commands are argv arrays, but reject an explicit shell
            # launcher so configuration cannot opt into shell interpretation.
            if any(token in {"sh", "bash", "zsh", "fish", "cmd.exe", "powershell"} for token in base_args):
                errors.append("launch.base_args may not invoke a shell")

    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "paths": {name: str(path) for name, path in paths.items()},
        "protocol": protocol_report,
        "resource_policy": _json_safe(config_value.get("resource_policy", {})),
        "protocol_id": str(protocol_value.get("protocol_id", "")),
        "runtime_schema_version": str(runtime_value.get("schema_version", "")),
        "config_path": None if config_path is None else str(config_path),
    }


def assert_valid_configuration(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = validate_configuration(*args, **kwargs)
    if not report["valid"]:
        raise ConfigurationError("configuration validation failed: " + "; ".join(report["errors"]))
    return report


def derive_seed(protocol_id: str, master_seed: int, opaque_trial_id: str) -> int:
    """Derive the protocol's uint32 SHA-256 seed deterministically.

    The concatenation follows the locked protocol string exactly: protocol id,
    decimal master seed, and opaque id with no shell-visible delimiters.
    """

    protocol_text = str(protocol_id)
    master = _integer(master_seed, "master_seed", minimum=0)
    opaque = _safe_identifier(opaque_trial_id, "opaque_trial_id")
    digest = hashlib.sha256(f"{protocol_text}{master}{opaque}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


seed_for_trial = derive_seed


def _opaque_condition(protocol_id: str, condition_id: str) -> str:
    digest = hashlib.sha256(f"{protocol_id}:{condition_id}".encode("utf-8")).hexdigest()[:12]
    return f"condition-{digest}"


def _pair_seed(protocol_id: str, master_seed: int, pair_id: str) -> tuple[str, int]:
    opaque_id = hashlib.sha256(
        f"{protocol_id}:{master_seed}:{pair_id}".encode("utf-8")
    ).hexdigest()[:16]
    return opaque_id, derive_seed(protocol_id, master_seed, opaque_id)


_MATRIX_FIELDS = """trial_id opaque_trial_id opaque_condition pair_id replicate tier_index
condition_index condition_id condition_family tier_id speed_mps impact_ms duration_ms
primary_tier seed replicate_kind lesion_id launchable blocked_reason""".split()


def _matrix_row(
    trial_id: str, opaque_id: str, protocol_id: str, pair_id: str, replicate: int,
    tier_index: int, condition_index: int, condition_id: str, family: str,
    tier_id: str, tier: Mapping[str, Any], seed: int, kind: str,
    lesion_id: str | None = None, launchable: bool = True, blocked_reason: str | None = None,
) -> dict[str, Any]:
    values = (
        trial_id, opaque_id, _opaque_condition(protocol_id, condition_id), pair_id,
        replicate, tier_index, condition_index, condition_id, family, tier_id,
        _finite_number(tier["speed_mps"], "speed_mps", minimum=0.0),
        _finite_number(tier["impact_ms"], "impact_ms", minimum=0.0),
        _finite_number(tier["duration_ms"], "duration_ms", minimum=0.0),
        bool(tier.get("primary", False)), seed, kind, lesion_id, launchable,
    )
    fields = _MATRIX_FIELDS if blocked_reason is not None else _MATRIX_FIELDS[:-1]
    return dict(zip(fields, values + ((blocked_reason,) if blocked_reason is not None else ())))


def _select_entries(
    entries: Sequence[Mapping[str, Any]],
    selected: Sequence[str] | None,
    name: str,
) -> list[Mapping[str, Any]]:
    if not selected:
        return list(entries)
    wanted = {str(item) for item in selected}
    result = [item for item in entries if str(item.get("id")) in wanted]
    missing = sorted(wanted - {str(item.get("id")) for item in entries})
    if missing:
        raise ToolboxError(f"unknown {name}: {', '.join(missing)}")
    return result


def build_seed_matrix(
    protocol: Mapping[str, Any] | str | Path = DEFAULT_PROTOCOL_CONFIG,
    *,
    conditions: Sequence[str] | None = None,
    tiers: Sequence[str] | None = None,
    replicates: int | None = None,
    n_replicates: int | None = None,
    include_lesions: bool = False,
    include_baseline_replays: bool = False,
) -> list[dict[str, Any]]:
    """Build a deterministic paired condition/tier/seed matrix.

    All conditions in one tier/replicate block share ``seed`` and
    ``pair_id``.  This is what makes sham, mirror, and matched controls
    directly comparable.  Lesions, when requested, are separate rows carrying
    the same matched seed and are marked non-launchable when the existing
    launcher has no generic lesion operator.
    """

    value: Mapping[str, Any] = _read_json(protocol) if isinstance(protocol, (str, Path)) else protocol
    report = validate_protocol(value)
    if not report["valid"]:
        raise ConfigurationError("protocol validation failed: " + "; ".join(report["errors"]))
    protocol_id = str(value["protocol_id"])
    randomization = value["randomization"]
    assert isinstance(randomization, Mapping)
    master_seed = _integer(randomization["master_seed"], "master_seed", minimum=0)
    default_replicates = _integer(
        randomization["digital_replicates_per_intact_condition_per_speed_tier"],
        "digital_replicates",
        minimum=1,
    )
    if replicates is not None and n_replicates is not None and int(replicates) != int(n_replicates):
        raise ToolboxError("replicates and n_replicates disagree")
    requested_replicates = replicates if replicates is not None else n_replicates
    count = default_replicates if requested_replicates is None else _integer(requested_replicates, "replicates", minimum=1)
    raw_conditions = [item for item in value["conditions"] if isinstance(item, Mapping)]
    raw_tiers = [item for item in value["speed_tiers"] if isinstance(item, Mapping)]
    selected_conditions = _select_entries(raw_conditions, conditions, "condition")
    selected_tiers = _select_entries(raw_tiers, tiers, "tier")
    lesion_entries = [item for item in value.get("lesions", []) if isinstance(item, Mapping)]

    matrix: list[dict[str, Any]] = []
    raw_tier_positions = {str(item.get("id")): index for index, item in enumerate(raw_tiers, start=1)}
    raw_condition_positions = {str(item.get("id")): index for index, item in enumerate(raw_conditions, start=1)}
    for tier in selected_tiers:
        tier_id = _safe_identifier(tier["id"], "tier id")
        tier_index = raw_tier_positions[tier_id]
        for replicate in range(1, count + 1):
            # The opaque block token is the preregistered seed material.  It
            # contains no condition label, so paired rows cannot leak the
            # condition to a blinded analyst.
            pair_id = f"pair-{tier_index:02d}-{replicate:03d}"
            opaque_block, seed = _pair_seed(protocol_id, master_seed, pair_id)
            for condition in selected_conditions:
                condition_id = _safe_identifier(condition["id"], "condition id")
                condition_index = raw_condition_positions[condition_id]
                trial_id = f"trial-{tier_index:02d}-{replicate:03d}-{condition_index:02d}"
                matrix.append(_matrix_row(
                    trial_id, opaque_block, protocol_id, pair_id, replicate, tier_index,
                    condition_index, condition_id, str(condition.get("family", "")),
                    tier_id, tier, seed, "intact",
                ))

            if include_lesions:
                # Lesion blocks are kept separate in metadata while retaining
                # the matched seed for causal comparisons.  The current
                # launcher explicitly has no generic lesion flag.
                for lesion_index, lesion in enumerate(lesion_entries, start=1):
                    lesion_id = _safe_identifier(lesion.get("id"), "lesion id")
                    matrix.append(_matrix_row(
                        f"lesion-{tier_index:02d}-{replicate:03d}-{lesion_index:02d}",
                        opaque_block, protocol_id, pair_id, replicate, tier_index, 1,
                        "true_right", "lesion", tier_id, tier, seed, "lesion",
                        lesion_id, False, "existing launcher has no generic lesion operator",
                    ))

    if include_baseline_replays:
        pressure = next((tier for tier in raw_tiers if str(tier.get("id")) == "pressure_10mps"), None)
        replay_count = _integer(randomization["exact_e14_baseline_replays"], "baseline_replays", minimum=1)
        if pressure is not None:
            for replay in range(1, replay_count + 1):
                pair_id = f"pressure-baseline-{replay:03d}"
                opaque_block, seed = _pair_seed(protocol_id, master_seed, pair_id)
                matrix.append(_matrix_row(
                    f"baseline-pressure-{replay:03d}", opaque_block, protocol_id, pair_id,
                    replay, raw_tiers.index(pressure) + 1, 1, "true_right", "baseline_replay",
                    "pressure_10mps", pressure, seed, "baseline_replay",
                ))
    return matrix


def _arg_set(args: list[str], option: str, value: str) -> None:
    """Set one option in an argv list without shell interpretation."""

    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    result.extend((option, value))
    args[:] = result


def _resource_policy(config: Mapping[str, Any]) -> tuple[float, float, tuple[str, ...]]:
    policy = config.get("resource_policy")
    if not isinstance(policy, Mapping):
        raise ResourcePolicyError("resource_policy is missing")
    ram = _finite_number(policy.get("host_ram_gib"), "host_ram_gib", minimum=0.01)
    vram = _finite_number(policy.get("max_vram_gib"), "max_vram_gib", minimum=0.01)
    if ram > HOST_RAM_LIMIT_GIB or vram > MAX_VRAM_GIB:
        raise ResourcePolicyError("configured resource limit exceeds the 5 GiB/15.5 GiB policy")
    wrapper = policy.get("host_wrapper")
    if not isinstance(wrapper, list) or tuple(wrapper) != RESOURCE_WRAPPER:
        raise ResourcePolicyError("resource wrapper must be systemd-run --user --scope MemoryMax=5G")
    return ram, vram, tuple(str(item) for item in wrapper)


def shell_command(argv: Sequence[str]) -> str:
    """Return display-only shell quoting; this string is never executed."""

    return shlex.join([str(item) for item in argv])


@dataclass(frozen=True)
class LaunchSpec:
    """A generated, resource-bounded argv command."""

    row: Mapping[str, Any]
    argv: tuple[str, ...]
    output_dir: Path
    command: str | None
    launchable: bool
    blocked_reason: str | None = None
    ram_limit_gib: float = HOST_RAM_LIMIT_GIB
    vram_limit_gib: float = MAX_VRAM_GIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.row.get("trial_id"),
            "row": _json_safe(dict(self.row)),
            "argv": list(self.argv),
            "command": self.command,
            "safe_command": self.command,
            "output_dir": str(self.output_dir),
            "launchable": self.launchable,
            "blocked_reason": self.blocked_reason,
            "resources": {
                "host_ram_gib": self.ram_limit_gib,
                "max_vram_gib": self.vram_limit_gib,
                "host_wrapper": list(RESOURCE_WRAPPER),
            },
        }


def build_launch_spec(
    row: Mapping[str, Any],
    config: Mapping[str, Any] | str | Path = DEFAULT_TOOLBOX_CONFIG,
    *,
    output_root: str | Path | None = None,
    root: Path = ROOT,
) -> LaunchSpec:
    """Generate one safe launcher command for a matrix row."""

    toolbox = load_toolbox_config(config) if isinstance(config, (str, Path)) else dict(config)
    assert_valid_configuration(toolbox, root=root)
    ram, vram, wrapper = _resource_policy(toolbox)
    trial_id = _safe_identifier(row.get("trial_id"), "trial_id")
    configured_roots = toolbox.get("artifact_roots", [])
    extra_roots = [Path(item) for item in configured_roots] if isinstance(configured_roots, list) else []
    launch = toolbox.get("launch", {})
    assert isinstance(launch, Mapping)
    output_base = (
        _resolved(output_root, root=root)
        if output_root is not None
        else _resolved(launch.get("output_root", DEFAULT_ARTIFACT_ROOT), root=root)
    )
    output_subdir = _safe_identifier(launch.get("output_subdir", "experiment-toolbox"), "output_subdir")
    output_dir = allowed_artifact_path(output_base / output_subdir / trial_id, root=root, extra_roots=extra_roots)
    if not bool(row.get("launchable", True)):
        reason = str(row.get("blocked_reason") or "matrix row is not launchable")
        return LaunchSpec(row=row, argv=(), output_dir=output_dir, command=None, launchable=False, blocked_reason=reason, ram_limit_gib=ram, vram_limit_gib=vram)

    paths = toolbox.get("paths", {})
    if isinstance(paths, Mapping) and "launcher" in paths:
        launcher = _resolved(paths["launcher"], root=root)
    else:
        launcher = _resolved("scripts/run_digital_fly.py", root=root)
    if not launcher.is_file():
        raise ConfigurationError(f"launcher does not exist: {launcher}")
    base_args = launch.get("base_args", [])
    if not isinstance(base_args, list) or not all(isinstance(token, str) for token in base_args):
        raise ConfigurationError("launch.base_args must be a list of strings")
    args = list(base_args)
    runtime_ref = paths.get("runtime", str(DEFAULT_RUNTIME_CONFIG)) if isinstance(paths, Mapping) else str(DEFAULT_RUNTIME_CONFIG)
    runtime_path = _resolved(runtime_ref, root=root)
    _arg_set(args, "--config", str(runtime_path))
    _arg_set(args, "--out", str(output_dir))
    _arg_set(args, "--seed", str(_integer(row.get("seed"), "seed", minimum=0)))
    _arg_set(args, "--max-vram-gib", f"{vram:g}")
    if row.get("condition_id") is not None:
        _arg_set(args, "--stimulus-condition", _safe_identifier(row["condition_id"], "condition_id"))
    if row.get("speed_mps") is not None:
        _arg_set(args, "--palm-speed-mps", f"{_finite_number(row['speed_mps'], 'speed_mps', minimum=0.0):g}")
    if row.get("impact_ms") is not None:
        impact = _finite_number(row["impact_ms"], "impact_ms", minimum=0.0) / 1000.0
        _arg_set(args, "--impact-time-s", f"{impact:g}")

    argv = (str(item) for item in (sys.executable, launcher, *args))
    command_argv = tuple(wrapper) + tuple(argv)
    # The wrapper itself is part of the command, so a caller can copy this
    # exact line into a terminal.  It remains safe when an id contains spaces
    # because execution uses the tuple, not this display string.
    return LaunchSpec(
        row=row,
        argv=command_argv,
        output_dir=output_dir,
        command=shell_command(command_argv),
        launchable=True,
        ram_limit_gib=ram,
        vram_limit_gib=vram,
    )


def build_launch_command(*args: Any, **kwargs: Any) -> list[str]:
    """Compatibility helper returning only the safe argv list."""

    spec = build_launch_spec(*args, **kwargs)
    if not spec.launchable:
        raise ResourcePolicyError(spec.blocked_reason or "matrix row is not launchable")
    return list(spec.argv)


def build_plan(
    config: Mapping[str, Any] | str | Path = DEFAULT_TOOLBOX_CONFIG,
    *,
    protocol: Mapping[str, Any] | str | Path | None = None,
    replicates: int | None = None,
    n_replicates: int | None = None,
    conditions: Sequence[str] | None = None,
    tiers: Sequence[str] | None = None,
    include_lesions: bool = False,
    include_baseline_replays: bool = False,
    output_root: str | Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build a dry-run plan, including every safe command or block reason."""

    toolbox = load_toolbox_config(config) if isinstance(config, (str, Path)) else dict(config)
    validation = assert_valid_configuration(toolbox, protocol=protocol, root=root)
    protocol_path = _reference(toolbox, "protocol", DEFAULT_PROTOCOL_CONFIG, root=root) if protocol is None else protocol
    protocol_value = load_protocol(protocol_path) if isinstance(protocol_path, (str, Path)) else dict(protocol_path)
    matrix = build_seed_matrix(
        protocol_value,
        conditions=conditions,
        tiers=tiers,
        replicates=replicates,
        n_replicates=n_replicates,
        include_lesions=include_lesions,
        include_baseline_replays=include_baseline_replays,
    )
    launches = [build_launch_spec(row, toolbox, output_root=output_root, root=root).to_dict() for row in matrix]
    ram, vram, wrapper = _resource_policy(toolbox)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "valid": True,
        "protocol_id": protocol_value.get("protocol_id"),
        "validation": validation,
        "resources": {"host_ram_gib": ram, "max_vram_gib": vram, "host_wrapper": list(wrapper)},
        "matrix_size": len(matrix),
        "launches": launches,
    }


def _cgroup_memory_limit_bytes() -> int | None:
    path = Path("/sys/fs/cgroup/memory.max")
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].lstrip("/")
                path = Path("/sys/fs/cgroup") / relative / "memory.max"
                break
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value == "max" or not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def preflight_report(
    config: Mapping[str, Any] | str | Path = DEFAULT_TOOLBOX_CONFIG,
    *,
    require_cuda: bool | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Return read-only project, gate, and resource facts.

    ``malecns_gate`` and ``twin_runtime.preflight_report`` are imported only
    here; neither import opens Torch or a CUDA context.  Failures are retained
    in the report instead of silently selecting a fallback backend.
    """

    toolbox = load_toolbox_config(config) if isinstance(config, (str, Path)) else dict(config)
    validation = validate_configuration(toolbox, root=root)
    launch = toolbox.get("launch", {})
    base_args = launch.get("base_args", []) if isinstance(launch, Mapping) else []
    requested_cuda = any(
        str(token).lower() in {"cuda", "cuda:0", "cuda:1", "torch", "triton", "flyvis"}
        for token in base_args
    )
    if require_cuda is None:
        require_cuda = requested_cuda

    gate_report: dict[str, Any] = {"available": False, "facts": {}, "blockers": []}
    twin_report: dict[str, Any] | None = None
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import malecns_gate as gate  # type: ignore

        gate_report = {
            "available": True,
            "facts": _json_safe(gate.facts()),
            "blockers": list(gate.cuda_blockers()),
        }
    except Exception as exc:  # pragma: no cover - defensive host variation
        gate_report["error"] = f"{type(exc).__name__}: {exc}"
        gate_report["blockers"] = ["malecns_gate could not be inspected"]
    try:
        from twin_runtime import preflight_report as twin_preflight  # type: ignore

        device = "cuda" if require_cuda else "auto"
        backend = "torch" if require_cuda else "auto"
        twin_report = _json_safe(twin_preflight(device=device, backend=backend))
    except Exception as exc:  # pragma: no cover - optional metadata only
        twin_report = {"error": f"{type(exc).__name__}: {exc}"}

    cgroup_bytes = _cgroup_memory_limit_bytes()
    wrapper_available = shutil.which(RESOURCE_WRAPPER[0]) is not None
    blockers: list[str] = []
    if not validation["valid"]:
        blockers.append("toolbox/protocol/runtime configuration is invalid")
    if require_cuda and gate_report.get("blockers"):
        blockers.extend(str(item) for item in gate_report["blockers"])
    if require_cuda and twin_report and twin_report.get("gate", {}).get("blockers"):
        blockers.extend(str(item) for item in twin_report["gate"]["blockers"])
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": not blockers,
        "require_cuda": bool(require_cuda),
        "ram_limit_gib": HOST_RAM_LIMIT_GIB,
        "vram_limit_gib": MAX_VRAM_GIB,
        "safe_command_prefix": list(RESOURCE_WRAPPER),
        "configuration": validation,
        "cuda_gate": gate_report,
        "twin_preflight": twin_report,
        "resource_policy": {
            "host_ram_limit_gib": HOST_RAM_LIMIT_GIB,
            "host_ram_limit_bytes": HOST_RAM_LIMIT_BYTES,
            "max_vram_gib": MAX_VRAM_GIB,
            "wrapper": list(RESOURCE_WRAPPER),
            "wrapper_available": wrapper_available,
            "current_cgroup_memory_limit_gib": None if cgroup_bytes is None else cgroup_bytes / GIB,
            "note": "generated workloads are always launched with the wrapper; a missing wrapper blocks --execute",
        },
        "blockers": list(dict.fromkeys(blockers)),
    }


def inventory_artifacts(
    root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    hash_files: bool = False,
    max_hash_bytes: int | None = None,
) -> dict[str, Any]:
    """Inventory existing artifacts without writing an index file."""

    base = Path(root).expanduser().resolve(strict=False)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    skipped: list[str] = []
    if not base.exists():
        return {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "root": str(base),
            "exists": False,
            "files": [],
            "total_bytes": 0,
            "skipped": [],
        }
    if not base.is_dir():
        raise ToolboxError(f"artifact inventory root is not a directory: {base}")
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            skipped.append(str(path.relative_to(base)))
            continue
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append(f"{path.relative_to(base)}: {exc}")
            continue
        item: dict[str, Any] = {
            "path": str(path.relative_to(base)),
            "bytes": size,
            "suffix": path.suffix.lower(),
        }
        total_bytes += size
        if hash_files and (max_hash_bytes is None or size <= max_hash_bytes):
            digest = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                item["sha256"] = digest.hexdigest()
            except OSError as exc:
                skipped.append(f"{path.relative_to(base)}: {exc}")
        files.append(item)
    suffix_counts = Counter(item["suffix"] or "<none>" for item in files)
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "root": str(base),
        "exists": True,
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "skipped": skipped,
    }


artifact_inventory = inventory_artifacts


def status_report(root: str | Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    """Summarize run manifests and summaries already present on disk."""

    base = Path(root).expanduser().resolve(strict=False)
    runs: list[dict[str, Any]] = []
    if base.exists() and base.is_dir():
        candidates = set(path.parent for path in base.rglob("manifest.json"))
        candidates.update(path.parent for path in base.rglob("summary.json") if path.parent not in candidates)
        for directory in sorted(candidates):
            manifest: Mapping[str, Any] = {}
            summary: Mapping[str, Any] = {}
            for name, target in (("manifest.json", manifest), ("summary.json", summary)):
                path = directory / name
                if not path.is_file():
                    continue
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    parsed = {}
                if isinstance(parsed, Mapping):
                    if name.startswith("manifest"):
                        manifest = parsed
                    else:
                        summary = parsed
            status = manifest.get("status", summary.get("status", "UNKNOWN"))
            failure = manifest.get("failure_class", summary.get("failure_class"))
            runs.append({
                "path": str(directory),
                "trial_id": manifest.get("trial_id", summary.get("trial_id")),
                "status": str(status),
                "failure_class": failure,
                "seed": manifest.get("seed", summary.get("seed")),
                "condition": manifest.get("condition", manifest.get("stimulus_condition", summary.get("condition"))),
                "has_manifest": bool(manifest),
                "has_summary": bool(summary),
            })
    counts = Counter(item["status"] for item in runs)
    failure_counts = Counter(str(item["failure_class"]) for item in runs if item["failure_class"] is not None)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "root": str(base),
        "run_count": len(runs),
        "status_counts": dict(sorted(counts.items())),
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "runs": runs,
    }


summarize_status = status_report

# Small descriptive aliases keep the module convenient for callers that use
# the CLI nouns as Python API names.
validate_config = validate_configuration
preflight = preflight_report
seed_matrix = build_seed_matrix
dry_run_plan = build_plan
generate_launch_command = build_launch_spec


def _write_json_artifact(path: str | Path, payload: Mapping[str, Any], *, root: Path = ROOT) -> Path:
    target = allowed_artifact_path(path, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _matrix_row_for_args(
    protocol: Mapping[str, Any],
    *,
    trial_id: str | None,
    condition: str | None,
    tier: str | None,
    replicate: int,
) -> dict[str, Any]:
    matrix = build_seed_matrix(protocol, conditions=[condition] if condition else None, tiers=[tier] if tier else None, replicates=max(1, replicate))
    if trial_id:
        matches = [row for row in build_seed_matrix(protocol) if row["trial_id"] == trial_id]
        if not matches:
            raise ToolboxError(f"unknown trial id: {trial_id}")
        return matches[0]
    if not condition or not tier:
        raise ToolboxError("launch requires --trial-id or both --condition and --tier")
    matches = [row for row in matrix if row["condition_id"] == condition and row["tier_id"] == tier and row["replicate"] == replicate]
    if not matches:
        raise ToolboxError("requested condition/tier/replicate is not in the seed matrix")
    return matches[0]


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_TOOLBOX_CONFIG, help="toolbox config JSON")
    parser.add_argument("--json", action="store_true", help="emit JSON")


def _emit(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", aliases=["check"], help="read-only gate and resource preflight")
    _add_common_config_args(preflight)
    preflight.add_argument("--require-cuda", action="store_true", help="treat CUDA gate blockers as launch blockers")

    validate = sub.add_parser("validate", aliases=["check-config"], help="validate protocol/toolbox/runtime JSON")
    _add_common_config_args(validate)
    validate.add_argument("--protocol", type=Path, default=None)
    validate.add_argument("--runtime", type=Path, default=None)

    matrix = sub.add_parser("matrix", aliases=["seed-matrix"], help="build the paired control/seed matrix")
    _add_common_config_args(matrix)
    matrix.add_argument("--protocol", type=Path, default=None)
    matrix.add_argument("--replicates", type=int, default=None)
    matrix.add_argument("--condition", action="append", dest="conditions")
    matrix.add_argument("--tier", action="append", dest="tiers")
    matrix.add_argument("--include-lesions", action="store_true")
    matrix.add_argument("--include-baseline-replays", action="store_true")
    matrix.add_argument("--output", type=Path, default=None)

    plan = sub.add_parser("plan", aliases=["dry-run"], help="validate and print a complete dry-run launch plan")
    _add_common_config_args(plan)
    plan.add_argument("--protocol", type=Path, default=None)
    plan.add_argument("--replicates", type=int, default=None)
    plan.add_argument("--condition", action="append", dest="conditions")
    plan.add_argument("--tier", action="append", dest="tiers")
    plan.add_argument("--include-lesions", action="store_true")
    plan.add_argument("--include-baseline-replays", action="store_true")
    plan.add_argument("--output-root", type=Path, default=None)
    plan.add_argument("--output", type=Path, default=None)

    launch = sub.add_parser("launch", aliases=["run"], help="print or execute one resource-bounded launch")
    _add_common_config_args(launch)
    launch.add_argument("--protocol", type=Path, default=None)
    launch.add_argument("--trial-id", default=None)
    launch.add_argument("--condition", default=None)
    launch.add_argument("--tier", default=None)
    launch.add_argument("--replicate", type=int, default=1)
    launch.add_argument("--output-root", type=Path, default=None)
    launch.add_argument("--execute", action="store_true", help="execute only after preflight and wrapper checks")

    inventory = sub.add_parser("inventory", aliases=["artifacts"], help="inventory existing artifact files")
    inventory.add_argument("root", nargs="?", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    inventory.add_argument("--hash", action="store_true", dest="hash_files")
    inventory.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="summarize existing run manifests")
    status.add_argument("root", nargs="?", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    status.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"preflight", "check"}:
            payload = preflight_report(args.config, require_cuda=True if args.require_cuda else None)
            _emit(payload, as_json=args.json)
            return 0 if payload["ready"] else 2
        if args.command in {"validate", "check-config"}:
            payload = validate_configuration(args.config, protocol=args.protocol, runtime=args.runtime)
            _emit(payload, as_json=args.json)
            return 0 if payload["valid"] else 2
        if args.command in {"matrix", "seed-matrix"}:
            toolbox = load_toolbox_config(args.config)
            protocol_path = args.protocol or _reference(toolbox, "protocol", DEFAULT_PROTOCOL_CONFIG)
            protocol = load_protocol(protocol_path)
            rows = build_seed_matrix(
                protocol,
                conditions=args.conditions,
                tiers=args.tiers,
                replicates=args.replicates,
                include_lesions=args.include_lesions,
                include_baseline_replays=args.include_baseline_replays,
            )
            payload = {"schema_version": MATRIX_SCHEMA_VERSION, "protocol_id": protocol.get("protocol_id"), "count": len(rows), "rows": rows}
            if args.output is not None:
                payload["output"] = str(_write_json_artifact(args.output, payload))
            _emit(payload, as_json=args.json)
            return 0
        if args.command in {"plan", "dry-run"}:
            payload = build_plan(
                args.config,
                protocol=args.protocol,
                replicates=args.replicates,
                conditions=args.conditions,
                tiers=args.tiers,
                include_lesions=args.include_lesions,
                include_baseline_replays=args.include_baseline_replays,
                output_root=args.output_root,
            )
            if args.output is not None:
                payload["output"] = str(_write_json_artifact(args.output, payload))
            _emit(payload, as_json=args.json)
            return 0
        if args.command in {"launch", "run"}:
            toolbox = load_toolbox_config(args.config)
            protocol_path = args.protocol or _reference(toolbox, "protocol", DEFAULT_PROTOCOL_CONFIG)
            protocol = load_protocol(protocol_path)
            row = _matrix_row_for_args(protocol, trial_id=args.trial_id, condition=args.condition, tier=args.tier, replicate=args.replicate)
            spec = build_launch_spec(row, toolbox, output_root=args.output_root)
            payload = spec.to_dict()
            if not args.execute:
                payload["mode"] = "dry-run"
                _emit(payload, as_json=args.json)
                return 0 if spec.launchable else 2
            if not spec.launchable:
                _emit(payload, as_json=args.json)
                return 2
            preflight = preflight_report(toolbox, require_cuda=True)
            if not preflight["ready"]:
                payload["mode"] = "blocked"
                payload["preflight"] = preflight
                _emit(payload, as_json=args.json)
                return 2
            if not preflight["resource_policy"]["wrapper_available"]:
                payload["mode"] = "blocked"
                payload["preflight"] = preflight
                _emit(payload, as_json=args.json)
                return 2
            # No shell is involved.  cwd is fixed to the project and the only
            # write path supplied to the child was checked by build_launch_spec.
            completed = subprocess.run(list(spec.argv), cwd=str(ROOT), check=False, shell=False)
            payload["mode"] = "executed"
            payload["returncode"] = completed.returncode
            _emit(payload, as_json=args.json)
            return int(completed.returncode)
        if args.command in {"inventory", "artifacts"}:
            _emit(inventory_artifacts(args.root, hash_files=args.hash_files), as_json=args.json)
            return 0
        if args.command == "status":
            _emit(status_report(args.root), as_json=args.json)
            return 0
    except (ToolboxError, OSError, TypeError, ValueError) as exc:
        print(f"experiment_toolbox: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI smoke tests exercise this
    raise SystemExit(main())


__all__ = [
    "ConfigurationError",
    "DATA_ARTIFACT_ROOT",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_PROTOCOL_CONFIG",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_TOOLBOX_CONFIG",
    "HOST_RAM_LIMIT_BYTES",
    "HOST_RAM_LIMIT_GIB",
    "LaunchSpec",
    "MAX_VRAM_GIB",
    "MATRIX_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "ResourcePolicyError",
    "SCHEMA_VERSION",
    "ToolboxError",
    "allowed_artifact_path",
    "artifact_inventory",
    "build_launch_command",
    "build_launch_spec",
    "build_plan",
    "build_seed_matrix",
    "derive_seed",
    "dry_run_plan",
    "generate_launch_command",
    "inventory_artifacts",
    "load_protocol",
    "load_runtime_config",
    "load_toolbox_config",
    "main",
    "preflight_report",
    "preflight",
    "seed_matrix",
    "seed_for_trial",
    "shell_command",
    "status_report",
    "summarize_status",
    "validate_configuration",
    "validate_config",
    "validate_protocol",
]
