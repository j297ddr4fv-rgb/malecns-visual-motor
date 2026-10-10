#!/usr/bin/env python3
"""Run a resumable, fail-closed suite of digital-fly conditions.

The suite runner is intentionally an orchestration layer.  It does not encode
motor policy, synthesize a trajectory, or modify ``run_digital_fly.py``.  A
condition supplies the documented launcher arguments (for example a grounded
loom, matched static scene, sham, or an explicit ablation argument), and this
module gives every condition a deterministic seed and an isolated output
directory.

The default executor starts one launcher subprocess at a time.  This is the
safe implementation for the current runner: immutable assets are fingerprinted
once and their fingerprints are reused in the plan, but model/GPU state is not
pretended to survive a subprocess.  A future resident-model hook should expose
an immutable session from ``DigitalFlyRunner`` and let the suite call
``run_condition(condition, resident_session=session)``.  The hook must return
the same model-initialisation key recorded here and must not allow conditions
to mutate the model or bypass the seed/resource checks.

Example JSON configuration::

    {
      "schema_version": "digital-fly-suite-1",
      "protocol_id": "escape-e14-v1",
      "master_seed": 20260921,
      "renderer_backend": "analytic_reference",
      "runner": {
        "script": "scripts/run_digital_fly.py",
        "base_args": ["--mode", "mission", "--duration", "0.01"],
        "assets": ["config/scenarios.json"]
      },
      "resources": {"max_vram_gib": 15.5, "max_wall_time_s": 3600},
      "conditions": [
        {"id": "loom", "family": "loom",
         "stimulus_name": "looming_grounded"},
        {"id": "static", "family": "static",
         "stimulus_name": "static_grounded"},
        {"id": "sham", "family": "sham",
         "stimulus_name": "sham_grounded"},
        {"id": "loom-input-ablation", "family": "ablation",
         "runner_args": ["--flight-state-drive", "config/ablation.json"]}
      ]
    }

The plan is written before the first process starts.  ``suite-manifest.json``
is immutable for a suite, while ``manifest-index.json`` is atomically updated
after every condition.  Completed attempts are reused only when their plan
fingerprint, seed, child manifest, and immutable asset hashes still match.
Incomplete attempts are retained and a later invocation gets a new numbered
attempt directory.  Renderer selection is suite-global: the default is
``analytic_reference``; a requested ``warp_cuda`` run is rejected unless the
launcher advertises ``--renderer-backend``, and an observed analytic result is
never accepted as a Warp run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_RUNNER = SCRIPT_DIR / "run_digital_fly.py"
SCHEMA_VERSION = "digital-fly-suite-1"
INDEX_SCHEMA_VERSION = "digital-fly-suite-index-1"
ATTEMPT_SCHEMA_VERSION = "digital-fly-suite-attempt-1"
INDEX_FILENAME = "manifest-index.json"
PLAN_FILENAME = "suite-manifest.json"
LOCK_FILENAME = ".suite.lock"

# This is deliberately a contract description, not a callable fallback.  The
# current runner has no resident-session argument and therefore receives one
# subprocess per condition.
FUTURE_RESIDENT_MODEL_HOOK = (
    "DigitalFlyRunner.create_resident_session() + "
    "DigitalFlyRunner.run_condition(condition, resident_session=session)"
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_FAMILIES = frozenset({
    "loom", "static", "sham", "ablation", "ablations", "control", "matched",
    "mirror", "lateral", "receding", "intact", "pressure", "baseline",
})
_FAMILY_ALIASES = {
    "ablations": "ablation", "looming": "loom", "static_control": "static",
    "sham_control": "sham", "lesion": "ablation",
}

# These options are owned by the suite.  Letting a condition replace any of
# them would make resume/index identity ambiguous or could write outside the
# condition directory.
_CONTROLLED_FLAGS = frozenset({
    "--out", "-o", "--seed", "--config", "--max-vram-gib", "--renderer-backend",
    "--condition-id", "--suite-id",
})
_RENDERER_BACKENDS = frozenset({"analytic_reference", "warp_cuda"})
_PATH_FLAGS = frozenset({
    "--config", "--visual-calibration", "--calibration",
    "--visual-calibration-config", "--calibration-config",
    "--neural-csr-path", "--flight-state-drive", "--flight-state-drive-config",
    "--state-drive", "--flight-body-stabilizer", "--body-stabilizer",
    "--stimulus-config",
})
_MODEL_INIT_FLAGS = frozenset({
    "--mode", "--execution-mode", "--strict", "--free-behavior", "--escape-experiment",
    "--neural", "--vision", "--body", "--width", "--height", "--columns", "--t4t5",
    "--neural-device", "--neural-backend", "--neural-gain", "--vision-device",
    "--visual-calibration", "--visual-calibration-config", "--neural-csr-path",
    "--flight-state-drive", "--flight-body-stabilizer", "--max-vram-gib",
})
_OPTION_KEYS = frozenset({
    "mode", "execution_mode", "strict", "free_behavior", "escape_experiment",
    "neural", "vision", "body", "width", "height", "columns", "t4t5",
    "neural_device", "neural_backend", "neural_gain", "vision_device",
    "duration_override_s", "max_steps", "frame_count", "full_neural_telemetry",
    "visual_calibration", "visual_calibration_config", "visual_tonic_state_inputs",
    "neural_csr_path", "flight_state_drive", "flight_body_stabilizer",
    "dynamic_stimulus", "jump_channel", "posture_channel", "stimulus_config",
    "stimulus_name", "stimulus_condition", "palm_speed_mps", "impact_time_s",
})
_FORBIDDEN_BEHAVIOUR_KEYS = frozenset({
    "behavior", "behaviour", "control_script", "motor_policy", "posture_policy",
    "trajectory", "target_trajectory", "controller", "autopilot_script",
})


class SuiteError(RuntimeError):
    """Base class for suite setup, execution, and validation failures."""


class SuiteConfigError(SuiteError):
    """Raised when a suite plan is ambiguous or unsafe."""


class SuiteExecutionError(SuiteError):
    """Raised for a child process or output contract failure."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> Any:
    """Convert JSON-ish values to deterministic, JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SuiteConfigError("non-finite values are not allowed in a suite plan")
        return value
    raise SuiteConfigError(f"unsupported non-JSON value in suite plan: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SuiteConfigError(f"cannot hash immutable asset {path}: {exc}") from exc
    return digest.hexdigest()


def _hash_path(path: Path) -> dict[str, Any]:
    """Return a content fingerprint for a file or a deterministic directory."""

    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SuiteConfigError(f"immutable asset does not exist: {path}") from exc
    if resolved.is_file():
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise SuiteConfigError(f"cannot stat immutable asset {resolved}: {exc}") from exc
        return {
            "path": str(path),
            "resolved_path": str(resolved),
            "kind": "file",
            "size_bytes": int(size),
            "sha256": _sha256_file(resolved),
        }
    if resolved.is_dir():
        entries: list[dict[str, Any]] = []
        try:
            children = sorted(resolved.rglob("*"), key=lambda item: str(item.relative_to(resolved)))
        except OSError as exc:
            raise SuiteConfigError(f"cannot enumerate immutable asset directory {resolved}: {exc}") from exc
        for child in children:
            if child.is_symlink():
                raise SuiteConfigError(f"immutable asset directory contains symlink: {child}")
            if child.is_file():
                rel = str(child.relative_to(resolved))
                entries.append({"path": rel, "size_bytes": child.stat().st_size, "sha256": _sha256_file(child)})
        tree_hash = _sha256_bytes(entries)
        return {
            "path": str(path),
            "resolved_path": str(resolved),
            "kind": "directory",
            "file_count": len(entries),
            "size_bytes": sum(int(item["size_bytes"]) for item in entries),
            "sha256": tree_hash,
            "entries": entries,
        }
    raise SuiteConfigError(f"immutable asset is not a regular file or directory: {resolved}")


def derive_seed(master_seed: int, trial_id: str, *, protocol_id: str = SCHEMA_VERSION) -> int:
    """Derive a stable uint32 seed from a suite seed and opaque trial id.

    A NUL-delimited encoding avoids accidental collisions between e.g.
    ``(1, 23)`` and ``(12, 3)``.  The result intentionally uses the first four
    SHA-256 bytes, matching the protocol's uint32 derivation convention.
    """

    try:
        if isinstance(master_seed, bool):
            raise ValueError
        master = int(master_seed)
        if isinstance(master_seed, float) and master_seed != master:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise SuiteConfigError("master_seed must be an integer") from exc
    if not 0 <= master <= (1 << 64) - 1:
        raise SuiteConfigError("master_seed must be an unsigned 64-bit integer")
    if not isinstance(trial_id, str) or not trial_id:
        raise SuiteConfigError("trial_id must be a non-empty string")
    payload = f"{protocol_id}\0{master}\0{trial_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _resolve_path(value: str | Path, *, bases: Sequence[Path], must_exist: bool = False) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [base / raw for base in bases]
    for candidate in candidates:
        if candidate.exists() or not must_exist:
            return candidate.resolve()
    raise SuiteConfigError(f"path does not exist: {value}")


def _flag_name(value: str) -> str:
    name = str(value).strip()
    if not name:
        raise SuiteConfigError("empty command-line option")
    return name if name.startswith("-") else "--" + name.replace("_", "-")


def _split_flag(token: str) -> tuple[str, Optional[str]]:
    text = str(token)
    if not text.startswith("-"):
        return text, None
    if "=" in text:
        name, value = text.split("=", 1)
        return name, value
    return text, None


def _check_args(
    args: Iterable[str], *, where: str, allow_config: bool = True,
    allow_vram: bool = False,
) -> tuple[str, ...]:
    values = tuple(str(item) for item in args)
    if any("\x00" in item for item in values):
        raise SuiteConfigError(f"NUL byte in {where}")
    for item in values:
        name, _ = _split_flag(item)
        if name in _CONTROLLED_FLAGS and not (
            (allow_config and name == "--config") or (allow_vram and name == "--max-vram-gib")
        ):
            raise SuiteConfigError(f"{where} may not override suite-owned option {name}")
    return values


def _options_to_args(options: Mapping[str, Any], *, where: str) -> tuple[str, ...]:
    if not isinstance(options, Mapping):
        raise SuiteConfigError(f"{where} must be an object")
    unknown = set(str(key) for key in options) - _OPTION_KEYS
    if unknown:
        raise SuiteConfigError(f"{where} has unsupported options: {sorted(unknown)}")
    result: list[str] = []
    for key in sorted(options, key=str):
        name = _flag_name(str(key))
        if name in _CONTROLLED_FLAGS:
            raise SuiteConfigError(f"{where} may not override suite-owned option {name}")
        value = options[key]
        if value is None or value is False:
            # False means use the documented launcher default.  A true inverse
            # flag can always be supplied explicitly through runner_args.
            continue
        if value is True:
            result.append(name)
            continue
        if isinstance(value, (list, tuple)):
            for element in value:
                result.extend((name, str(element)))
            continue
        result.extend((name, str(value)))
    return tuple(result)


def _validate_metadata(value: Any, *, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SuiteConfigError(f"{where} must be an object")
    return _canonical(dict(value))


@dataclass(frozen=True)
class ResourceCaps:
    """Caps applied to every child process.

    ``max_vram_gib`` is also passed to the documented launcher.  Host memory
    and CPU caps use POSIX rlimits where available; wall time is enforced by
    the parent process.  The suite remains sequential, so one cap applies to
    one live child at a time.
    """

    max_vram_gib: Optional[float] = 15.5
    max_host_memory_bytes: Optional[int] = None
    max_cpu_seconds: Optional[int] = None
    max_wall_time_s: Optional[float] = None
    max_output_bytes: Optional[int] = 64 * 1024 * 1024
    max_parallel: int = 1

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "ResourceCaps":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise SuiteConfigError("resources must be an object")
        allowed = {
            "max_vram_gib", "max_host_memory_gib", "max_host_memory_bytes",
            "max_cpu_seconds", "max_wall_time_s", "max_output_bytes", "max_parallel",
        }
        unknown = set(str(key) for key in raw) - allowed
        if unknown:
            raise SuiteConfigError(f"resources has unsupported fields: {sorted(unknown)}")

        def positive_number(key: str, value: Any, *, integer: bool = False) -> Any:
            if value is None:
                return None
            try:
                if isinstance(value, bool):
                    raise ValueError
                number = int(value) if integer else float(value)
                if integer and isinstance(value, float) and value != number:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise SuiteConfigError(f"resources.{key} must be numeric") from exc
            if number <= 0 or (isinstance(number, float) and not math.isfinite(number)):
                raise SuiteConfigError(f"resources.{key} must be positive and finite")
            return number

        vram = positive_number("max_vram_gib", raw.get("max_vram_gib"))
        memory = raw.get("max_host_memory_bytes")
        if memory is None and raw.get("max_host_memory_gib") is not None:
            gib = positive_number("max_host_memory_gib", raw.get("max_host_memory_gib"))
            memory = int(gib * (1024 ** 3))
        elif memory is not None:
            memory = positive_number("max_host_memory_bytes", memory, integer=True)
        cpu = positive_number("max_cpu_seconds", raw.get("max_cpu_seconds"), integer=True)
        wall = positive_number("max_wall_time_s", raw.get("max_wall_time_s"))
        output = positive_number("max_output_bytes", raw.get("max_output_bytes"), integer=True)
        raw_parallel = raw.get("max_parallel", 1)
        if isinstance(raw_parallel, bool):
            raise SuiteConfigError("resources.max_parallel must be an integer")
        parallel = int(raw_parallel)
        if isinstance(raw_parallel, float) and raw_parallel != parallel:
            raise SuiteConfigError("resources.max_parallel must be an integer")
        if parallel != 1:
            raise SuiteConfigError("only max_parallel=1 is supported by the fail-closed executor")
        return cls(
            max_vram_gib=None if vram is None else float(vram),
            max_host_memory_bytes=None if memory is None else int(memory),
            max_cpu_seconds=None if cpu is None else int(cpu),
            max_wall_time_s=None if wall is None else float(wall),
            max_output_bytes=None if output is None else int(output),
            max_parallel=parallel,
        )

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class RunnerSpec:
    script: Path
    base_args: tuple[str, ...] = ()
    config: Optional[Path] = None
    assets: tuple[Path, ...] = ()
    renderer_backend_interface: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "script": str(self.script),
            "base_args": list(self.base_args),
            "config": None if self.config is None else str(self.config),
            "assets": [str(item) for item in self.assets],
            "renderer_backend_interface": self.renderer_backend_interface,
        }


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    family: str
    seed: int
    seed_source: str
    runner_args: tuple[str, ...] = ()
    assets: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    stimulus_name: Optional[str] = None
    stimulus_condition: Optional[str] = None
    model_initialization_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.condition_id,
            "family": self.family,
            "seed": self.seed,
            "seed_source": self.seed_source,
            "runner_args": list(self.runner_args),
            "assets": [str(item) for item in self.assets],
            "metadata": _canonical(self.metadata),
            "stimulus_name": self.stimulus_name,
            "stimulus_condition": self.stimulus_condition,
            "model_initialization_key": self.model_initialization_key,
        }


# Readable compatibility aliases for callers that prefer shorter names.
ExperimentCondition = ConditionSpec


def _renderer_manifest_fields(
    requested: str, effective: str, provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "requested_renderer_backend": requested,
        "effective_renderer_backend": effective,
        "renderer_provenance": _canonical(provenance),
        "renderer_backend": {"requested": requested, "effective": effective, "provenance": _canonical(provenance)},
    }


@dataclass(frozen=True)
class SuitePlan:
    name: str
    protocol_id: str
    master_seed: int
    runner: RunnerSpec
    conditions: tuple[ConditionSpec, ...]
    resources: ResourceCaps
    assets: tuple[dict[str, Any], ...]
    model_initialization_key: str
    requested_renderer_backend: str
    effective_renderer_backend: str
    renderer_provenance: Mapping[str, Any]
    plan_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "protocol_id": self.protocol_id,
            "master_seed": self.master_seed,
            "runner": self.runner.to_dict(),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "resources": self.resources.to_dict(),
            "assets": [_canonical(asset) for asset in self.assets],
            "fail_closed": True,
            **_renderer_manifest_fields(
                self.requested_renderer_backend, self.effective_renderer_backend, self.renderer_provenance,
            ),
            "reuse": {
                "immutable_assets": {
                    "possible": True,
                    "mechanism": "content-addressed fingerprints shared by all attempts",
                },
                "model_initialization": {
                    "possible": False,
                    "mechanism": "one subprocess per condition",
                    "key": self.model_initialization_key,
                    "condition_keys": {
                        condition.condition_id: condition.model_initialization_key
                        for condition in self.conditions
                    },
                    "future_hook": FUTURE_RESIDENT_MODEL_HOOK,
                },
            },
            "metadata": _canonical(self.metadata),
        }
        if include_hash:
            value["plan_hash"] = self.plan_hash
        return value


def _parse_condition(
    raw: Mapping[str, Any], *, master_seed: int, protocol_id: str, root: Path,
    extra_bases: Sequence[Path] = (),
) -> ConditionSpec:
    if not isinstance(raw, Mapping):
        raise SuiteConfigError("each condition must be an object")
    allowed = {
        "id", "name", "condition_id", "trial_id", "replicate", "family", "seed", "runner_args", "args",
        "cli_args", "options", "overrides", "stimulus", "stimulus_name",
        "stimulus_condition", "assets", "metadata", "ablation", "intervention",
    }
    unknown = set(str(key) for key in raw) - allowed
    if unknown:
        raise SuiteConfigError(f"condition has unsupported fields: {sorted(unknown)}")
    for forbidden in _FORBIDDEN_BEHAVIOUR_KEYS:
        if forbidden in raw:
            raise SuiteConfigError(f"condition field {forbidden!r} is prohibited; supply documented runner args")
    condition_id = str(raw.get("id", raw.get("condition_id", raw.get("name", "")))).strip()
    if not _SAFE_ID.fullmatch(condition_id):
        raise SuiteConfigError(f"invalid condition id {condition_id!r}")
    family = str(raw.get("family", condition_id.split("-", 1)[0])).strip().lower()
    family = _FAMILY_ALIASES.get(family, family)
    if family not in _ALLOWED_FAMILIES:
        raise SuiteConfigError(f"unsupported condition family {family!r}; allowed={sorted(_ALLOWED_FAMILIES)}")
    trial_id = str(raw.get("trial_id", condition_id))
    if raw.get("replicate") is not None:
        try:
            if isinstance(raw["replicate"], bool):
                raise ValueError
            replicate = int(raw["replicate"])
            if isinstance(raw["replicate"], float) and raw["replicate"] != replicate:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise SuiteConfigError(f"condition {condition_id!r} replicate must be an integer") from exc
        if replicate < 0:
            raise SuiteConfigError(f"condition {condition_id!r} replicate must be non-negative")
        trial_id = f"{trial_id}\0replicate={replicate}"
    explicit_seed = raw.get("seed")
    if explicit_seed is None:
        seed = derive_seed(master_seed, trial_id, protocol_id=protocol_id)
        seed_source = "sha256"
    else:
        try:
            if isinstance(explicit_seed, bool):
                raise ValueError
            seed = int(explicit_seed)
            if isinstance(explicit_seed, float) and explicit_seed != seed:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise SuiteConfigError(f"condition {condition_id!r} seed must be an integer") from exc
        if not 0 <= seed <= 0xFFFFFFFF:
            raise SuiteConfigError(f"condition {condition_id!r} seed must be uint32")
        seed_source = "explicit"

    args: list[str] = []
    for key in ("runner_args", "args", "cli_args"):
        if key in raw:
            value = raw[key]
            if not isinstance(value, (list, tuple)):
                raise SuiteConfigError(f"condition {condition_id!r} {key} must be a list")
            args.extend(str(item) for item in value)
    for key in ("options", "overrides"):
        if key in raw:
            args.extend(_options_to_args(raw[key], where=f"condition {condition_id!r}.{key}"))
    stimulus_name = raw.get("stimulus_name")
    stimulus_condition = raw.get("stimulus_condition")
    stimulus = raw.get("stimulus")
    if stimulus is not None:
        if not isinstance(stimulus, str) or not stimulus.strip():
            raise SuiteConfigError(f"condition {condition_id!r}.stimulus must be a non-empty string")
        if stimulus_name is None:
            stimulus_name = stimulus
    for key, value in (("stimulus_name", stimulus_name), ("stimulus_condition", stimulus_condition)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SuiteConfigError(f"condition {condition_id!r}.{key} must be a non-empty string")
    if stimulus_name is not None and not any(_split_flag(arg)[0] == "--stimulus-name" for arg in args):
        args.extend(("--stimulus-name", str(stimulus_name)))
    if stimulus_condition is not None and not any(_split_flag(arg)[0] == "--stimulus-condition" for arg in args):
        args.extend(("--stimulus-condition", str(stimulus_condition)))
    args_tuple = _check_args(args, where=f"condition {condition_id!r}", allow_config=False)

    asset_values = raw.get("assets", ())
    if not isinstance(asset_values, (list, tuple)):
        raise SuiteConfigError(f"condition {condition_id!r}.assets must be a list")
    bases = (*extra_bases, root, Path.cwd())
    assets = tuple(_resolve_path(str(item), bases=bases, must_exist=True) for item in asset_values)
    metadata: dict[str, Any] = _validate_metadata(raw.get("metadata"), where=f"condition {condition_id!r}.metadata")
    if raw.get("replicate") is not None:
        metadata["replicate"] = int(raw["replicate"])
    for key in ("ablation", "intervention"):
        if key in raw:
            metadata[key] = _canonical(raw[key])
    return ConditionSpec(
        condition_id=condition_id,
        family=family,
        seed=seed,
        seed_source=seed_source,
        runner_args=args_tuple,
        assets=assets,
        metadata=metadata,
        stimulus_name=None if stimulus_name is None else str(stimulus_name),
        stimulus_condition=None if stimulus_condition is None else str(stimulus_condition),
    )


def _parse_runner(
    raw: Any, *, root: Path, override_script: Optional[Path], extra_bases: Sequence[Path] = (),
) -> RunnerSpec:
    if raw is None:
        raw = {}
    if isinstance(raw, (str, Path)):
        raw = {"script": raw}
    if not isinstance(raw, Mapping):
        raise SuiteConfigError("runner must be an object or script path")
    allowed = {"script", "base_args", "args", "config", "assets"}
    unknown = set(str(key) for key in raw) - allowed
    if unknown:
        raise SuiteConfigError(f"runner has unsupported fields: {sorted(unknown)}")
    script_value = override_script if override_script is not None else raw.get("script", DEFAULT_RUNNER)
    bases = (*extra_bases, root, SCRIPT_DIR, Path.cwd())
    script = _resolve_path(script_value, bases=bases, must_exist=True)
    if not script.is_file():
        raise SuiteConfigError(f"runner script is not a regular file: {script}")
    values = raw.get("base_args", raw.get("args", ()))
    if not isinstance(values, (list, tuple)):
        raise SuiteConfigError("runner.base_args must be a list")
    base_args = _check_args(values, where="runner.base_args", allow_config=True, allow_vram=True)
    config = None
    if raw.get("config") is not None:
        config = _resolve_path(raw["config"], bases=(*extra_bases, root, Path.cwd()), must_exist=True)
        if not config.is_file():
            raise SuiteConfigError(f"runner config is not a regular file: {config}")
    asset_values = raw.get("assets", ())
    if not isinstance(asset_values, (list, tuple)):
        raise SuiteConfigError("runner.assets must be a list")
    assets = tuple(_resolve_path(str(item), bases=(*extra_bases, root, Path.cwd()), must_exist=True) for item in asset_values)
    return RunnerSpec(
        script=script,
        base_args=base_args,
        config=config,
        assets=assets,
        renderer_backend_interface=_detect_renderer_interface(script),
    )


def _detect_renderer_interface(script: Path) -> bool:
    """Detect the narrow documented renderer flag without executing a runner.

    Static probing keeps planning side-effect free.  A Warp request is allowed
    only when both backend names and the flag are visible in the launcher
    source; an opaque executable must opt in through a future explicit runner
    capability record rather than receiving an unverified argument.
    """

    try:
        source = script.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return all(token in source for token in ("--renderer-backend", "analytic_reference", "warp_cuda"))


def _collect_flag_paths(
    args: Sequence[str], *, root: Path, extra_bases: Sequence[Path] = (),
) -> tuple[Path, ...]:
    result: list[Path] = []
    index = 0
    while index < len(args):
        name, inline = _split_flag(args[index])
        if name in _PATH_FLAGS:
            value = inline
            if value is None:
                if index + 1 >= len(args):
                    raise SuiteConfigError(f"path option {name} has no value")
                index += 1
                value = args[index]
            try:
                path = _resolve_path(value, bases=(*extra_bases, root, Path.cwd()), must_exist=True)
            except SuiteConfigError as exc:
                raise SuiteConfigError(f"path option {name}: {exc}") from exc
            result.append(path)
        index += 1
    return tuple(result)


def _model_initialization_args(args: Sequence[str]) -> tuple[str, ...]:
    """Keep only arguments that can change model/kernel initialization."""

    selected: list[str] = []
    index = 0
    while index < len(args):
        name, inline = _split_flag(args[index])
        if name in _MODEL_INIT_FLAGS:
            selected.append(name if inline is None else f"{name}={inline}")
            if inline is None and index + 1 < len(args) and not args[index + 1].startswith("-"):
                index += 1
                selected.append(args[index])
        index += 1
    return tuple(selected)


def _resolve_renderer_selection(
    raw: Mapping[str, Any], *, requested_override: Optional[str], runner: RunnerSpec,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve one suite-global renderer without permitting fallback."""

    configured: Any = raw.get("renderer_backend")
    renderer_block = raw.get("renderer")
    if renderer_block is not None:
        if isinstance(renderer_block, str):
            if configured is not None and str(configured) != renderer_block:
                raise SuiteConfigError("renderer_backend and renderer.backend disagree")
            configured = renderer_block
        elif isinstance(renderer_block, Mapping):
            unknown = set(str(key) for key in renderer_block) - {"backend", "renderer_backend", "requested", "provenance"}
            if unknown:
                raise SuiteConfigError(f"renderer has unsupported fields: {sorted(unknown)}")
            nested = renderer_block.get(
                "backend", renderer_block.get("renderer_backend", renderer_block.get("requested"))
            )
            if nested is not None:
                if configured is not None and str(configured) != str(nested):
                    raise SuiteConfigError("renderer_backend and renderer.backend disagree")
                configured = nested
        else:
            raise SuiteConfigError("renderer must be a backend string or object")
    requested = str(requested_override if requested_override is not None else (configured or "analytic_reference"))
    if requested not in _RENDERER_BACKENDS:
        raise SuiteConfigError(
            f"unsupported renderer backend {requested!r}; allowed={sorted(_RENDERER_BACKENDS)}"
        )
    interface = bool(runner.renderer_backend_interface)
    if requested == "warp_cuda" and not interface:
        raise SuiteConfigError(
            "requested renderer backend warp_cuda, but the launcher does not advertise "
            "the --renderer-backend interface; refusing analytic fallback"
        )
    effective = requested
    provenance = {
        "requested": requested,
        "effective": effective,
        "launcher_interface_supported": interface,
        "selection": "launcher_flag" if interface else "suite_default",
        "flag_injected": interface,
        "fallback": False,
        "fail_closed": True,
    }
    if renderer_block is not None and isinstance(renderer_block, Mapping) and renderer_block.get("provenance") is not None:
        supplied = renderer_block["provenance"]
        if not isinstance(supplied, Mapping):
            raise SuiteConfigError("renderer.provenance must be an object")
        # Provenance is descriptive metadata only; selection/effective fields
        # stay owned by this resolver and cannot be used to claim a fallback.
        provenance["declared"] = _canonical(dict(supplied))
    return requested, effective, provenance


def _read_suite_config(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteConfigError(f"cannot read suite config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SuiteConfigError(f"invalid JSON in suite config {path}: {exc}") from exc


def build_plan(
    config: Mapping[str, Any] | str | Path,
    *,
    out: str | Path,
    runner_script: Optional[str | Path] = None,
    master_seed: Optional[int] = None,
    resources: Optional[Mapping[str, Any]] = None,
    renderer_backend: Optional[str] = None,
) -> SuitePlan:
    """Validate a suite config and produce a deterministic execution plan."""

    config_path: Optional[Path] = None
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        raw = _read_suite_config(config_path)
    else:
        raw = dict(config)
    if not isinstance(raw, Mapping):
        raise SuiteConfigError("suite config must be a JSON object")
    allowed = {
        "schema_version", "name", "suite_id", "protocol_id", "master_seed", "seed",
        "runner", "conditions", "ablations", "resources", "assets", "metadata", "randomization",
        "renderer_backend", "renderer",
    }
    unknown = set(str(key) for key in raw) - allowed
    if unknown:
        raise SuiteConfigError(f"suite config has unsupported fields: {sorted(unknown)}")
    schema = str(raw.get("schema_version", SCHEMA_VERSION))
    if schema != SCHEMA_VERSION:
        raise SuiteConfigError(f"unsupported suite schema_version {schema!r}")
    root = ROOT
    config_bases = () if config_path is None else (config_path.parent,)
    protocol_id = str(raw.get("protocol_id", raw.get("suite_id", "digital-fly-suite")))
    if not protocol_id or "\x00" in protocol_id:
        raise SuiteConfigError("protocol_id must be a non-empty string without NUL")
    original_master = master_seed if master_seed is not None else raw.get("master_seed", raw.get("seed", 1729))
    selected_master = original_master
    try:
        if isinstance(selected_master, bool):
            raise ValueError
        selected_master = int(selected_master)
        if isinstance(original_master, float):
            if original_master != selected_master:
                raise ValueError
    except (TypeError, ValueError) as exc:
        raise SuiteConfigError("master_seed must be an integer") from exc
    if not 0 <= selected_master <= (1 << 64) - 1:
        raise SuiteConfigError("master_seed must be an unsigned 64-bit integer")
    runner = _parse_runner(
        raw.get("runner"), root=root,
        override_script=None if runner_script is None else Path(runner_script),
        extra_bases=config_bases,
    )
    requested_renderer_backend, effective_renderer_backend, renderer_provenance = _resolve_renderer_selection(
        raw, requested_override=renderer_backend, runner=runner,
    )
    base_paths = list(_collect_flag_paths(runner.base_args, root=root, extra_bases=config_bases))
    if runner.config is not None:
        base_paths.append(runner.config)
    base_paths.extend(runner.assets)
    top_asset_values = raw.get("assets", ())
    if not isinstance(top_asset_values, (list, tuple)):
        raise SuiteConfigError("assets must be a list")
    top_assets = tuple(_resolve_path(str(item), bases=(*config_bases, root, Path.cwd()), must_exist=True) for item in top_asset_values)
    condition_values = raw.get("conditions", [])
    if condition_values is None:
        condition_values = []
    ablation_values = raw.get("ablations", [])
    if ablation_values is None:
        ablation_values = []
    if not isinstance(condition_values, (list, tuple)) or not isinstance(ablation_values, (list, tuple)):
        raise SuiteConfigError("conditions and ablations must be lists")
    normalized_ablations: list[Any] = []
    for ablation in ablation_values:
        if isinstance(ablation, Mapping) and "family" not in ablation:
            normalized = dict(ablation)
            normalized["family"] = "ablation"
            normalized_ablations.append(normalized)
        else:
            normalized_ablations.append(ablation)
    condition_values = [*condition_values, *normalized_ablations]
    if not isinstance(condition_values, (list, tuple)) or not condition_values:
        raise SuiteConfigError("conditions must be a non-empty list")
    conditions: list[ConditionSpec] = []
    seen: set[str] = set()
    for value in condition_values:
        condition = _parse_condition(
            value, master_seed=selected_master, protocol_id=protocol_id,
            root=root, extra_bases=config_bases,
        )
        if condition.condition_id in seen:
            raise SuiteConfigError(f"duplicate condition id {condition.condition_id!r}")
        seen.add(condition.condition_id)
        conditions.append(condition)
        base_paths.extend(condition.assets)
        base_paths.extend(_collect_flag_paths(condition.runner_args, root=root, extra_bases=config_bases))
    # Keep only unique paths while retaining deterministic order.
    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in [runner.script, *top_assets, *base_paths]:
        resolved = path.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_paths.append(resolved)
    assets = tuple(_hash_path(path) for path in unique_paths)
    runner_asset = next((asset for asset in assets if asset.get("resolved_path") == str(runner.script.resolve())), None)
    renderer_provenance = {
        **renderer_provenance,
        "launcher_script": str(runner.script),
        "launcher_interface_probe": "static_source_tokens",
        "launcher_script_sha256": None if runner_asset is None else runner_asset.get("sha256"),
    }
    resource_values: Optional[Mapping[str, Any]]
    if resources is None:
        resource_values = raw.get("resources")
    else:
        # CLI caps are overlays; an explicit --max-vram-gib must not erase a
        # host-memory or wall-time cap declared in the locked suite config.
        configured_resources = raw.get("resources", {}) or {}
        if not isinstance(configured_resources, Mapping):
            raise SuiteConfigError("resources must be an object")
        merged_resources = dict(configured_resources)
        merged_resources.update(dict(resources))
        resource_values = merged_resources
    resources_obj = ResourceCaps.from_mapping(resource_values)
    # A condition may request an explicit max-vram value through runner args,
    # but it must not exceed the suite cap or quietly defeat the cap.
    for args in (runner.base_args, *(condition.runner_args for condition in conditions)):
        for index, token in enumerate(args):
            name, inline = _split_flag(token)
            if name != "--max-vram-gib":
                continue
            value = inline
            if value is None and index + 1 < len(args):
                value = args[index + 1]
            if value is None:
                raise SuiteConfigError("--max-vram-gib has no value")
            try:
                requested = float(value)
            except ValueError as exc:
                raise SuiteConfigError("--max-vram-gib must be numeric") from exc
            if resources_obj.max_vram_gib is not None and requested > resources_obj.max_vram_gib + 1e-12:
                raise SuiteConfigError("runner --max-vram-gib exceeds suite resource cap")
    model_key_base = {
        "runner": runner.to_dict(),
        "assets": [asset for asset in assets],
        "resources": resources_obj.to_dict(),
        "renderer_backend": {
            "requested": requested_renderer_backend,
            "effective": effective_renderer_backend,
            "provenance": renderer_provenance,
        },
    }
    condition_model_keys: dict[str, str] = {}
    keyed_conditions: list[ConditionSpec] = []
    for condition in conditions:
        model_key = _sha256_bytes({
            **model_key_base,
            "model_initialization_args": _model_initialization_args(
                (*runner.base_args, *condition.runner_args)
            ),
        })
        condition_model_keys[condition.condition_id] = model_key
        keyed_conditions.append(replace(condition, model_initialization_key=model_key))
    conditions = keyed_conditions
    model_key = _sha256_bytes({"condition_model_initialization_keys": condition_model_keys})
    metadata = _validate_metadata(raw.get("metadata"), where="metadata")
    if "randomization" in raw:
        metadata["randomization"] = _canonical(raw["randomization"])
    name = str(raw.get("name", raw.get("suite_id", "digital-fly-experiment-suite"))).strip()
    if not name:
        raise SuiteConfigError("suite name must be non-empty")
    preliminary = SuitePlan(
        name=name,
        protocol_id=protocol_id,
        master_seed=selected_master,
        runner=runner,
        conditions=tuple(conditions),
        resources=resources_obj,
        assets=assets,
        model_initialization_key=model_key,
        requested_renderer_backend=requested_renderer_backend,
        effective_renderer_backend=effective_renderer_backend,
        renderer_provenance=renderer_provenance,
        plan_hash="",
        metadata=metadata,
    )
    plan_hash = _sha256_bytes(preliminary.to_dict(include_hash=False))
    return replace(preliminary, plan_hash=plan_hash)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteError(f"cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SuiteError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SuiteError(f"{label} {path} must contain an object")
    return value


def _iso_elapsed(start: float) -> float:
    return round(max(0.0, time.monotonic() - start), 6)


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    wall_time_s: float
    timed_out: bool = False
    output_bytes: int = 0
    output_limit_exceeded: bool = False


def _make_preexec(caps: ResourceCaps) -> Callable[[], None]:
    def apply() -> None:
        if caps.max_cpu_seconds is not None:
            seconds = int(caps.max_cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
        if caps.max_host_memory_bytes is not None and hasattr(resource, "RLIMIT_AS"):
            memory = int(caps.max_host_memory_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    return apply


def execute_subprocess(
    command: Sequence[str], cwd: Path, caps: ResourceCaps,
    stdout_path: Path, stderr_path: Path,
) -> ExecutionResult:
    """Run one child in a new process group with the suite resource caps."""

    started = time.monotonic()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                list(command), cwd=str(cwd), stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True,
                preexec_fn=_make_preexec(caps) if os.name == "posix" else None,
            )
        except OSError as exc:
            raise SuiteExecutionError(f"cannot start runner: {exc}") from exc
        timed_out = False
        output_limit_exceeded = False

        def current_output_bytes() -> int:
            total = 0
            for path in (stdout_path, stderr_path):
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
            return total

        deadline = None if caps.max_wall_time_s is None else started + caps.max_wall_time_s
        while process.poll() is None:
            if caps.max_output_bytes is not None and current_output_bytes() > caps.max_output_bytes:
                output_limit_exceeded = True
                break
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0.0:
                timed_out = True
                break
            try:
                process.wait(timeout=0.1 if remaining is None else min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
        if timed_out or output_limit_exceeded:
            try:
                if process.pid and os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if process.pid and os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except OSError:
                    pass
                process.wait()
        output_bytes = 0
        for path in (stdout_path, stderr_path):
            try:
                output_bytes += path.stat().st_size
            except OSError:
                pass
        return ExecutionResult(
            returncode=-signal.SIGTERM if timed_out else int(process.returncode),
            wall_time_s=_iso_elapsed(started),
            timed_out=timed_out,
            output_bytes=output_bytes,
            output_limit_exceeded=output_limit_exceeded,
        )


Executor = Callable[[Sequence[str], Path, ResourceCaps, Path, Path], ExecutionResult]


class _SuiteLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: Optional[int] = None

    def __enter__(self) -> "_SuiteLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(self.fd, f"pid={os.getpid()}\nstarted={_now()}\n".encode("utf-8"))
        except FileExistsError as exc:
            # A killed parent can leave a lock behind and would otherwise make
            # the resumable suite permanently unusable.  Remove only a lock
            # whose recorded PID is definitely no longer alive; an unreadable
            # or live lock remains a hard concurrency failure.
            stale = False
            try:
                text = self.path.read_text(encoding="utf-8")
                pid_line = next(line for line in text.splitlines() if line.startswith("pid="))
                pid = int(pid_line.split("=", 1)[1])
                if pid > 0:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        stale = False
            except (OSError, StopIteration, ValueError):
                stale = False
            if stale:
                try:
                    self.path.unlink()
                except OSError as unlink_exc:
                    raise SuiteError(f"suite is already locked: {self.path}") from unlink_exc
                return self.__enter__()
            raise SuiteError(f"suite is already locked: {self.path}") from exc
        except OSError as exc:
            raise SuiteError(f"cannot acquire suite lock {self.path}: {exc}") from exc
        return self

    def __exit__(self, *_: Any) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


class ExperimentSuite:
    """Execute and resume a :class:`SuitePlan` without mutating the launcher."""

    def __init__(
        self,
        plan: SuitePlan,
        out: str | Path,
        *,
        resume: bool = True,
        fail_fast: bool = False,
        executor: Optional[Executor] = None,
        root: Path = ROOT,
    ) -> None:
        self.plan = plan
        self.out = Path(out).expanduser().resolve()
        if self.out in {Path("/"), Path.home().resolve(), Path(root).resolve(), SCRIPT_DIR}:
            raise SuiteError(f"refusing unsafe broad suite output directory: {self.out}")
        self.resume = bool(resume)
        self.fail_fast = bool(fail_fast)
        self.executor = executor or execute_subprocess
        self.root = Path(root).resolve()

    def _plan_manifest(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "created_at": _now(),
            "fail_closed": True,
            "resume_contract": {
                "completed_attempt_requires": [
                    "matching plan_hash", "matching condition seed", "valid child manifest",
                    "unchanged immutable asset hashes",
                ],
                "incomplete_attempt_policy": "retain and allocate a new numbered attempt",
            },
        }

    def _verify_assets(self) -> None:
        """Re-hash locked assets before any child process starts."""

        for expected in self.plan.assets:
            resolved = expected.get("resolved_path")
            if not isinstance(resolved, str):
                raise SuiteError("suite plan contains an invalid immutable asset record")
            current = _hash_path(Path(resolved))
            if current.get("sha256") != expected.get("sha256") or current.get("kind") != expected.get("kind"):
                raise SuiteError(f"immutable asset changed since plan creation: {resolved}")
            if current.get("size_bytes") != expected.get("size_bytes"):
                raise SuiteError(f"immutable asset size changed since plan creation: {resolved}")

    def _initial_index(self) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "suite_manifest": PLAN_FILENAME,
            "suite_name": self.plan.name,
            "protocol_id": self.plan.protocol_id,
            "master_seed": self.plan.master_seed,
            "plan_hash": self.plan.plan_hash,
            **_renderer_manifest_fields(
                self.plan.requested_renderer_backend, self.plan.effective_renderer_backend,
                self.plan.renderer_provenance,
            ),
            "status": "planned",
            "fail_closed": True,
            "created_at": _now(),
            "updated_at": _now(),
            "reuse": self.plan.to_dict()["reuse"],
            "conditions": [
                {
                    "id": condition.condition_id,
                    "family": condition.family,
                    "seed": condition.seed,
                    "status": "pending",
                    "attempts": [],
                }
                for condition in self.plan.conditions
            ],
        }

    def _validate_renderer_manifest(self, value: Mapping[str, Any], label: str) -> None:
        if (
            value.get("requested_renderer_backend") != self.plan.requested_renderer_backend
            or value.get("effective_renderer_backend") != self.plan.effective_renderer_backend
            or value.get("renderer_provenance") != _canonical(self.plan.renderer_provenance)
        ):
            raise SuiteError(f"existing {label} renderer backend differs from the plan")
        renderer = value.get("renderer_backend")
        if not isinstance(renderer, Mapping) or renderer.get("provenance") != _canonical(self.plan.renderer_provenance):
            raise SuiteError(f"existing {label} renderer provenance differs from the plan")

    def _prepare_files(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        plan_path = self.out / PLAN_FILENAME
        index_path = self.out / INDEX_FILENAME
        if plan_path.exists():
            existing = _read_json(plan_path, label="suite manifest")
            if existing.get("plan_hash") != self.plan.plan_hash:
                raise SuiteError("existing suite manifest has a different plan_hash; refusing to mix plans")
            self._validate_renderer_manifest(existing, "suite manifest")
        else:
            _atomic_write_json(plan_path, self._plan_manifest())
        if index_path.exists():
            index = dict(_read_json(index_path, label="manifest index"))
            if index.get("plan_hash") != self.plan.plan_hash:
                raise SuiteError("existing manifest index has a different plan_hash")
            if str(index.get("schema_version")) != INDEX_SCHEMA_VERSION:
                raise SuiteError("existing manifest index has an unsupported schema_version")
            self._validate_renderer_manifest(index, "manifest index")
            ids = [str(item.get("id")) for item in index.get("conditions", []) if isinstance(item, Mapping)]
            if len(ids) != len(set(ids)):
                raise SuiteError("existing manifest index contains duplicate condition ids")
            if set(ids) != {condition.condition_id for condition in self.plan.conditions}:
                raise SuiteError("existing manifest index condition table differs from the plan")
            return index
        index = self._initial_index()
        _atomic_write_json(index_path, index)
        return index

    def _condition_record(self, index: MutableMapping[str, Any], condition: ConditionSpec) -> MutableMapping[str, Any]:
        for item in index.get("conditions", []):
            if isinstance(item, MutableMapping) and item.get("id") == condition.condition_id:
                return item
        raise SuiteError(f"condition {condition.condition_id!r} missing from manifest index")

    def _attempt_dir(self, condition: ConditionSpec) -> Path:
        root = self.out / "conditions" / condition.condition_id
        root.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            candidate = root / f"attempt-{number:04d}"
            if not candidate.exists():
                candidate.mkdir(parents=True)
                return candidate
            number += 1

    def _condition_args(self, condition: ConditionSpec, attempt_dir: Path) -> list[str]:
        args: list[str] = [sys.executable, str(self.plan.runner.script)]
        args.extend(self.plan.runner.base_args)
        base_flags = {_split_flag(value)[0] for value in args}
        if self.plan.runner.config is not None and "--config" not in base_flags:
            args.extend(("--config", str(self.plan.runner.config)))
        args.extend(condition.runner_args)
        flags = {_split_flag(value)[0] for value in args}
        if self.plan.runner.renderer_backend_interface:
            args.extend(("--renderer-backend", self.plan.effective_renderer_backend))
        if self.plan.resources.max_vram_gib is not None and "--max-vram-gib" not in flags:
            args.extend(("--max-vram-gib", str(self.plan.resources.max_vram_gib)))
        args.extend(("--seed", str(condition.seed), "--out", str(attempt_dir)))
        return args

    def _child_manifest(self, attempt_dir: Path, condition: ConditionSpec) -> Mapping[str, Any]:
        path = attempt_dir / "manifest.json"
        if path.is_symlink() or not path.is_file():
            raise SuiteExecutionError(f"condition {condition.condition_id!r} did not produce {path}")
        manifest = _read_json(path, label="child run manifest")
        schema = str(manifest.get("schema_version", ""))
        if not schema.startswith("digital-fly-run-"):
            raise SuiteExecutionError(f"condition {condition.condition_id!r} child manifest has unsupported schema")
        if manifest.get("seed") != condition.seed:
            raise SuiteExecutionError(
                f"condition {condition.condition_id!r} child manifest seed mismatch "
                f"(expected {condition.seed}, got {manifest.get('seed')!r})"
            )
        requested = manifest.get("requested_renderer_backend")
        requested_execution = manifest.get("requested_execution")
        if requested is None and isinstance(requested_execution, Mapping):
            requested = requested_execution.get("renderer_backend")
        effective = manifest.get("effective_renderer_backend")
        effective_execution = manifest.get("effective_execution")
        if effective is None and isinstance(effective_execution, Mapping):
            effective = effective_execution.get("renderer_backend")
        effective_backends = manifest.get("effective_backends")
        if effective is None and isinstance(effective_backends, Mapping):
            effective = effective_backends.get("renderer")
        if self.plan.runner.renderer_backend_interface or self.plan.requested_renderer_backend == "warp_cuda":
            if requested is None or effective is None:
                raise SuiteExecutionError(
                    f"condition {condition.condition_id!r} child manifest omitted renderer provenance"
                )
        if requested is not None and str(requested) != self.plan.requested_renderer_backend:
            raise SuiteExecutionError(
                f"condition {condition.condition_id!r} child requested renderer mismatch "
                f"(expected {self.plan.requested_renderer_backend}, got {requested!r})"
            )
        if effective is not None and str(effective) != self.plan.effective_renderer_backend:
            raise SuiteExecutionError(
                f"condition {condition.condition_id!r} child effective renderer mismatch "
                f"(expected {self.plan.effective_renderer_backend}, got {effective!r})"
            )
        if self.plan.requested_renderer_backend == "warp_cuda" and str(effective) != "warp_cuda":
            raise SuiteExecutionError(
                f"condition {condition.condition_id!r} requested warp_cuda but did not produce warp_cuda"
            )
        return manifest

    def _attempt_is_resumable(self, condition: ConditionSpec, attempt: Mapping[str, Any]) -> bool:
        if str(attempt.get("status")) != "complete":
            return False
        if attempt.get("plan_hash") != self.plan.plan_hash or attempt.get("seed") != condition.seed:
            return False
        if attempt.get("model_initialization_key") != condition.model_initialization_key:
            return False
        if (
            attempt.get("requested_renderer_backend") != self.plan.requested_renderer_backend
            or attempt.get("effective_renderer_backend") != self.plan.effective_renderer_backend
            or attempt.get("renderer_provenance") != _canonical(self.plan.renderer_provenance)
        ):
            return False
        expected_assets = [asset["sha256"] for asset in self.plan.assets]
        if attempt.get("immutable_asset_hashes") != expected_assets:
            return False
        path_value = attempt.get("attempt_dir")
        if not isinstance(path_value, str):
            return False
        path = Path(path_value)
        if not path.is_absolute():
            path = self.out / path
        try:
            path = path.resolve()
            path.relative_to(self.out.resolve())
        except (OSError, ValueError):
            return False
        try:
            self._child_manifest(path, condition)
        except SuiteError:
            return False
        expected_manifest_hash = attempt.get("child_manifest_sha256")
        if not isinstance(expected_manifest_hash, str):
            return False
        try:
            if _sha256_file(path / "manifest.json") != expected_manifest_hash:
                return False
        except SuiteError:
            return False
        return True

    def _run_condition(self, condition: ConditionSpec, record: MutableMapping[str, Any], index: MutableMapping[str, Any]) -> bool:
        attempt_dir = self._attempt_dir(condition)
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        command = self._condition_args(condition, attempt_dir)
        started_at = _now()
        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "plan_hash": self.plan.plan_hash,
            "condition_id": condition.condition_id,
            "family": condition.family,
            "seed": condition.seed,
            **_renderer_manifest_fields(
                self.plan.requested_renderer_backend, self.plan.effective_renderer_backend,
                self.plan.renderer_provenance,
            ),
            "command": command,
            "attempt_dir": str(attempt_dir),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "started_at": started_at,
            "status": "running",
            "immutable_asset_hashes": [asset["sha256"] for asset in self.plan.assets],
            "model_initialization_key": condition.model_initialization_key,
            "fail_closed": True,
        }
        attempts = record.setdefault("attempts", [])
        if not isinstance(attempts, list):
            raise SuiteError(f"condition {condition.condition_id!r} attempts field is not a list")
        attempts.append(attempt)
        record["status"] = "running"
        index["status"] = "running"
        index["updated_at"] = _now()
        _atomic_write_json(self.out / INDEX_FILENAME, index)
        try:
            result = self.executor(
                command, self.root, self.plan.resources, stdout_path, stderr_path,
            )
            attempt["returncode"] = result.returncode
            attempt["wall_time_s"] = result.wall_time_s
            attempt["timed_out"] = result.timed_out
            attempt["output_bytes"] = result.output_bytes
            attempt["output_limit_exceeded"] = result.output_limit_exceeded
            if self.plan.resources.max_output_bytes is not None and result.output_bytes > self.plan.resources.max_output_bytes:
                raise SuiteExecutionError("child stdout/stderr exceeded max_output_bytes")
            if result.output_limit_exceeded:
                raise SuiteExecutionError("child stdout/stderr exceeded max_output_bytes")
            if result.timed_out:
                raise SuiteExecutionError("child process exceeded max_wall_time_s")
            if result.returncode != 0:
                raise SuiteExecutionError(f"child process exited with status {result.returncode}")
            child_manifest = self._child_manifest(attempt_dir, condition)
            attempt["child_manifest"] = str(attempt_dir / "manifest.json")
            attempt["child_manifest_sha256"] = _sha256_file(attempt_dir / "manifest.json")
            attempt["child_schema_version"] = child_manifest.get("schema_version")
            attempt["finished_at"] = _now()
            attempt["status"] = "complete"
            record["status"] = "complete"
            record["latest_attempt"] = str(attempt_dir)
            return True
        except (SuiteError, OSError, ValueError) as exc:
            attempt["status"] = "error"
            attempt["error"] = str(exc)
            attempt["finished_at"] = _now()
            record["status"] = "error"
            record["error"] = str(exc)
            record["latest_attempt"] = str(attempt_dir)
            return False
        finally:
            index["updated_at"] = _now()
            _atomic_write_json(self.out / INDEX_FILENAME, index)
            _atomic_write_json(attempt_dir / "suite-attempt.json", attempt)

    def run(self) -> dict[str, Any]:
        """Run pending conditions, retaining all attempts and returning index data."""

        with _SuiteLock(self.out / LOCK_FILENAME):
            self._verify_assets()
            index = self._prepare_files()
            if not self.resume and any(
                isinstance(record, Mapping) and record.get("attempts") for record in index.get("conditions", [])
            ):
                raise SuiteError("suite already contains attempts; use resume=True or a new output directory")
            all_ok = True
            for condition in self.plan.conditions:
                record = self._condition_record(index, condition)
                prior = record.get("attempts", [])
                if self.resume and isinstance(prior, list) and prior:
                    latest = prior[-1]
                    if isinstance(latest, Mapping) and self._attempt_is_resumable(condition, latest):
                        record["status"] = "skipped"
                        record["resumed_from"] = latest.get("attempt_dir")
                        index["updated_at"] = _now()
                        _atomic_write_json(self.out / INDEX_FILENAME, index)
                        continue
                ok = self._run_condition(condition, record, index)
                all_ok = all_ok and ok
                if not ok and self.fail_fast:
                    break
            # Conditions not reached under fail_fast remain pending and make
            # the suite explicitly incomplete/error rather than appearing done.
            all_ok = all_ok and all(
                isinstance(record, Mapping) and record.get("status") in {"complete", "skipped"}
                for record in index.get("conditions", [])
            )
            index["status"] = "complete" if all_ok else "error"
            index["updated_at"] = _now()
            _atomic_write_json(self.out / INDEX_FILENAME, index)
            return index


def load_suite_config(path: str | Path) -> Mapping[str, Any]:
    """Load and parse a JSON suite configuration without building a plan."""

    source = Path(path).expanduser().resolve()
    value = _read_suite_config(source)
    if not isinstance(value, Mapping):
        raise SuiteConfigError("suite config must be an object")
    return value


def run_suite(
    config_or_plan: SuitePlan | Mapping[str, Any] | str | Path,
    out: str | Path,
    *,
    resume: bool = True,
    fail_fast: bool = False,
    runner_script: Optional[str | Path] = None,
    master_seed: Optional[int] = None,
    resources: Optional[Mapping[str, Any]] = None,
    executor: Optional[Executor] = None,
) -> dict[str, Any]:
    """Build (when necessary) and execute one suite.

    This small functional entry point is useful to callers that do not need
    to retain the plan object.  It intentionally shares all validation with
    :func:`build_plan` and :class:`ExperimentSuite`.
    """

    if isinstance(config_or_plan, SuitePlan):
        plan = config_or_plan
    else:
        plan = build_plan(
            config_or_plan, out=out, runner_script=runner_script,
            master_seed=master_seed, resources=resources,
        )
    return ExperimentSuite(
        plan, out, resume=resume, fail_fast=fail_fast, executor=executor,
    ).run()


def _cli_config(args: argparse.Namespace) -> Mapping[str, Any] | Path:
    if args.config is not None:
        return args.config
    if not args.condition:
        raise SuiteConfigError("--config is required unless at least one --condition is supplied")
    conditions: list[dict[str, Any]] = []
    for name in args.condition:
        condition_id = str(name)
        family = condition_id.split("-", 1)[0].lower()
        if family not in _ALLOWED_FAMILIES:
            raise SuiteConfigError(f"--condition {condition_id!r} requires a known family or --config")
        conditions.append({"id": condition_id, "family": family})
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "cli-experiment-suite",
        "protocol_id": "cli-experiment-suite",
        "master_seed": args.master_seed,
        "runner": {"script": str(args.runner) if args.runner is not None else str(DEFAULT_RUNNER)},
        "conditions": conditions,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--suite-config", type=Path, default=None)
    parser.add_argument("--condition", "--conditions", action="append", default=[], help="CLI-only condition id (prefer --config for explicit args)")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "experiment-suite")
    parser.add_argument("--runner", type=Path, default=None)
    parser.add_argument("--master-seed", type=int, default=1729)
    parser.add_argument(
        "--renderer-backend", choices=tuple(sorted(_RENDERER_BACKENDS)), default=None,
        help="suite-global renderer; defaults to analytic_reference",
    )
    parser.add_argument("--max-vram-gib", type=float, default=None)
    parser.add_argument("--max-host-memory-gib", type=float, default=None)
    parser.add_argument("--max-cpu-seconds", type=int, default=None)
    parser.add_argument("--max-wall-time-s", type=float, default=None)
    parser.add_argument("--max-output-bytes", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = _cli_config(args)
        resource_values: dict[str, Any] = {}
        for key, value in {
            "max_vram_gib": args.max_vram_gib,
            "max_host_memory_gib": args.max_host_memory_gib,
            "max_cpu_seconds": args.max_cpu_seconds,
            "max_wall_time_s": args.max_wall_time_s,
            "max_output_bytes": args.max_output_bytes,
        }.items():
            if value is not None:
                resource_values[key] = value
        plan = build_plan(
            config, out=args.out, runner_script=args.runner,
            master_seed=args.master_seed if args.config is None else None,
            resources=resource_values or None,
            renderer_backend=args.renderer_backend,
        )
        if args.dry_run:
            print(json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0
        result = ExperimentSuite(
            plan, args.out, resume=args.resume, fail_fast=args.fail_fast,
        ).run()
        print(json.dumps({
            "status": result["status"],
            "index": str(Path(args.out).resolve() / INDEX_FILENAME),
            "plan_hash": plan.plan_hash,
            "requested_renderer_backend": plan.requested_renderer_backend,
            "effective_renderer_backend": plan.effective_renderer_backend,
            "conditions": [
                {"id": item.get("id"), "status": item.get("status"), "seed": item.get("seed")}
                for item in result.get("conditions", []) if isinstance(item, Mapping)
            ],
        }, ensure_ascii=False, sort_keys=True))
        return 0 if result["status"] == "complete" else 2
    except (SuiteError, OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc), "fail_closed": True}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


__all__ = [
    "ATTEMPT_SCHEMA_VERSION", "ConditionSpec", "ExecutionResult", "ExperimentCondition", "ExperimentSuite",
    "FUTURE_RESIDENT_MODEL_HOOK", "INDEX_FILENAME", "ResourceCaps", "RunnerSpec",
    "SCHEMA_VERSION", "SuiteConfigError", "SuiteError", "SuiteExecutionError",
    "SuitePlan", "build_plan", "derive_seed", "execute_subprocess", "load_suite_config",
    "main", "parse_args", "run_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
