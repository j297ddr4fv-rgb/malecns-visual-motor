#!/usr/bin/env python3
"""Read-only latency profiler and duration estimator for the flight pipeline.

This module intentionally has no NumPy, Torch, FlyVis, CUDA, or simulator
dependency.  It has two related jobs:

* :class:`StageProfiler`, :class:`BatchProbe`, and :class:`CacheProbe` are
  small hooks that a caller can put around an already-selected backend.  They
  measure the work the caller asks them to measure; they never construct a
  backend or start a flight run.
* The command line reads existing JSON/JSONL/CSV artifacts and estimates the
  wall time for a requested amount of simulated flight.  Existing open-loop
  MaleCNS measurements are reported as a neural-only lower bound.  The older
  FlyVis flight artifact is reported as a proxy, not as timing for the new
  ``run_digital_fly.py`` path.

The public API is deliberately usable from a small launcher without changing
the simulation modules.  A CUDA caller may pass an explicit ``synchronize``
callback to :class:`StageProfiler`; the profiler does not import Torch merely
to make a measurement synchronous.  ``--gpu-profile`` lazily imports the
existing launcher for a bounded real-GPU run, reports exact 721-column eyes
separately from the rectangular stereo frame, and records warmup/steady-state
timings plus the 5 GiB host / 15.5 GiB VRAM envelope.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts"
SCHEMA_VERSION = "flight-gpu-profile-2"
# Keep the names stable: these are the boundary labels consumed by the
# engineering reports and by downstream notebooks.  The two visual render
# paths are intentionally separate.  A rectangular 32x24 stereo frame is a
# telemetry/video product; the fitted FlyVis model consumes its exact 721
# columns per eye.
PIPELINE_STAGES = (
    "world_raycast",
    "rectangular_stereo_render",
    "exact_column_render",
    "flyvis",
    "malecns_10tick_window",
    "neuromuscular",
    "body",
    "serialization",
)
PROFILE_PHASES = ("warmup", "steady")
# The explicit raycast pass and the rectangular frame are diagnostic
# alternatives.  The former is a representative public ``trace_ray`` pass
# (the renderers perform their own raycasts); the latter is telemetry/video
# only while the fitted FlyVis path uses exact columns.  Neither is added to
# the exact-column end-to-end sum below, preventing an intentional comparison
# pass from being mistaken for extra work in one frame.
NON_ADDITIVE_PIPELINE_STAGES = frozenset({
    "world_raycast", "rectangular_stereo_render",
})
FLYVIS_HZ = 200.0
CNS_HZ = 2000.0
CNS_STEPS_PER_FLYVIS = int(round(CNS_HZ / FLYVIS_HZ))
FRAME_DT_S = 1.0 / FLYVIS_HZ
GIB = 1024.0 ** 3

T = TypeVar("T")
R = TypeVar("R")


class ProfileError(ValueError):
    """Raised for malformed profiler input or impossible estimates."""


def _finite(value: Any, name: str, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ProfileError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ProfileError(f"{name} must be finite")
    return number


def _int_value(value: Any, name: str, *, default: int = 1, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileError(f"{name} must be an integer") from exc
    if number < minimum:
        raise ProfileError(f"{name} must be >= {minimum}")
    return number


def _json_safe(value: Any) -> Any:
    """Convert common values to JSON without importing array packages."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return _json_safe(value.value)
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return repr(value)


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Linear-interpolated percentile with no dependency on NumPy."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(1.0, max(0.0, float(quantile)))
    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def _sample_metadata(value: Mapping[str, Any], *, limit: int = 32) -> dict[str, Any]:
    """Keep metadata compact; artifact rows can contain giant image arrays."""

    allowed = {
        "backend", "device", "kind", "evidence", "graph", "stim", "stimulus",
        "legacy", "real", "source_label", "units", "note", "scenario",
        "cache", "cache_key", "operation",
        "renderer_backend", "cold_jit_candidate", "diagnostic_only", "path",
        "ray_count", "resolution", "columns_per_eye", "eyes", "input_path",
        "steps", "dt_ms", "integrator_calls", "window_steps", "format", "compact",
    }
    out: dict[str, Any] = {}
    for key in allowed:
        if key in value:
            item = value[key]
            if isinstance(item, (str, int, float, bool)) or item is None:
                out[key] = _json_safe(item)
            elif isinstance(item, (list, tuple)) and len(item) <= limit:
                out[key] = _json_safe(item)
            elif isinstance(item, Mapping):
                out[key] = _json_safe({str(k): v for k, v in list(item.items())[:limit]})
    return out


@dataclass(frozen=True, slots=True)
class StageSample:
    """One measured observation.

    ``count`` is the number of logical items represented by the observation
    (CNS ticks, frames, or rows); ``batch_size`` describes how many items were
    submitted to one call when that distinction is known.  ``sim_s`` is the
    simulated-time interval represented by the observation, not wall time.
    """

    stage: str
    wall_s: float
    sim_s: Optional[float] = None
    count: int = 1
    batch_size: Optional[int] = None
    cache_hit: Optional[bool] = None
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # ``phase`` is explicit rather than inferred from sample order.  This is
    # important for GPU kernels whose first invocation includes context/JIT
    # startup and whose later invocations are the throughput measurement.
    phase: str = "steady"

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        if not stage:
            raise ProfileError("stage must not be empty")
        wall = _finite(self.wall_s, "wall_s")
        assert wall is not None
        if wall < 0:
            raise ProfileError("wall_s must be >= 0")
        sim = _finite(self.sim_s, "sim_s", allow_none=True)
        if sim is not None and sim < 0:
            raise ProfileError("sim_s must be >= 0")
        count = _int_value(self.count, "count", default=1, minimum=1)
        batch = None if self.batch_size is None else _int_value(self.batch_size, "batch_size", minimum=1)
        if self.cache_hit is not None and not isinstance(self.cache_hit, bool):
            raise ProfileError("cache_hit must be bool or None")
        phase = str(self.phase).strip().lower() or "steady"
        if phase not in PROFILE_PHASES:
            raise ProfileError(f"phase must be one of {PROFILE_PHASES!r}")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "wall_s", wall)
        object.__setattr__(self, "sim_s", sim)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "batch_size", batch)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "phase", phase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "wall_s": self.wall_s,
            "sim_s": self.sim_s,
            "count": self.count,
            "batch_size": self.batch_size,
            "cache_hit": self.cache_hit,
            "source": self.source,
            "metadata": _json_safe(self.metadata),
            "phase": self.phase,
        }


@dataclass(frozen=True, slots=True)
class StageStats:
    """Aggregated observations for one stage."""

    stage: str
    samples: tuple[StageSample, ...]

    @property
    def wall_values(self) -> tuple[float, ...]:
        return tuple(sample.wall_s for sample in self.samples)

    @property
    def sim_values(self) -> tuple[float, ...]:
        return tuple(sample.sim_s for sample in self.samples if sample.sim_s is not None)

    @property
    def total_wall_s(self) -> float:
        return sum(self.wall_values)

    @property
    def total_sim_s(self) -> Optional[float]:
        values = self.sim_values
        return sum(values) if values else None

    @property
    def total_count(self) -> int:
        return sum(sample.count for sample in self.samples)

    @property
    def cache_hits(self) -> int:
        return sum(sample.cache_hit is True for sample in self.samples)

    @property
    def cache_misses(self) -> int:
        return sum(sample.cache_hit is False for sample in self.samples)

    def phase(self, name: str) -> "StageStats":
        """Return only observations from ``warmup`` or ``steady`` phase."""

        phase = str(name).strip().lower()
        if phase not in PROFILE_PHASES:
            raise ProfileError(f"phase must be one of {PROFILE_PHASES!r}")
        return StageStats(self.stage, tuple(item for item in self.samples if item.phase == phase))

    @property
    def warmup(self) -> "StageStats":
        return self.phase("warmup")

    @property
    def steady(self) -> "StageStats":
        return self.phase("steady")

    @property
    def wall_throughput_per_s(self) -> Optional[float]:
        """Logical items per wall second (not a simulated-time RTF)."""

        return self.total_count / self.total_wall_s if self.total_wall_s > 0 else None

    @property
    def real_time_factor(self) -> Optional[float]:
        """Simulated seconds processed per wall second."""

        if self.total_sim_s is None or self.total_wall_s <= 0:
            return None
        return self.total_sim_s / self.total_wall_s

    def to_dict(self) -> dict[str, Any]:
        summary = _stage_summary(self)
        batches = [sample.batch_size for sample in self.samples if sample.batch_size is not None]
        return {
            "stage": self.stage,
            **summary,
            "batch_size_min": min(batches) if batches else None,
            "batch_size_max": max(batches) if batches else None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if self.cache_hits + self.cache_misses else None
            ),
            "sources": sorted({sample.source for sample in self.samples if sample.source}),
            "phases": {name: _stage_summary(self.phase(name)) for name in PROFILE_PHASES},
        }


def _stage_summary(rows: StageStats) -> dict[str, Any]:
    values = rows.wall_values
    sim_values = rows.sim_values
    return {
        "samples": len(rows.samples),
        "total_wall_s": rows.total_wall_s,
        "total_sim_s": rows.total_sim_s,
        "total_count": rows.total_count,
        "wall_s_per_count": rows.total_wall_s / rows.total_count if rows.total_count else None,
        "throughput_per_s": rows.wall_throughput_per_s,
        "real_time_factor": rows.real_time_factor,
        "wall_p50_s": _percentile(values, 0.50),
        "wall_p95_s": _percentile(values, 0.95),
        "wall_max_s": max(values) if values else None,
        "sim_s_per_sample": statistics.fmean(sim_values) if sim_values else None,
    }


def _group_stage_stats(samples: Iterable[StageSample]) -> dict[str, StageStats]:
    groups: dict[str, list[StageSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.stage].append(sample)
    return {name: StageStats(name, tuple(rows)) for name, rows in sorted(groups.items())}


class StageProfiler:
    """Dependency-light wall-clock hook for caller-owned pipeline stages.

    ``clock`` is injectable for deterministic tests.  ``synchronize`` is an
    optional callback supplied by the caller when an asynchronous accelerator
    must be synchronized around the interval.  It is deliberately not
    imported or guessed here.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Optional[Callable[[], Any]] = None,
        max_samples: int = 100_000,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable or None")
        if int(max_samples) < 1:
            raise ValueError("max_samples must be >= 1")
        self.clock = clock
        self.synchronize = synchronize
        self.max_samples = int(max_samples)
        self._samples: list[StageSample] = []
        self.dropped_samples = 0
        # Boundary synchronizations are deliberately counted separately from
        # stage wall time.  The callback is invoked only immediately before a
        # timer starts and immediately after the measured operation returns;
        # no synchronization is inserted into a stage's inner loop.
        self.synchronize_calls = 0

    def record(
        self,
        stage: str,
        wall_s: float,
        *,
        sim_s: Optional[float] = None,
        count: int = 1,
        batch_size: Optional[int] = None,
        cache_hit: Optional[bool] = None,
        source: str = "hook",
        metadata: Optional[Mapping[str, Any]] = None,
        phase: str = "steady",
    ) -> StageSample:
        sample = StageSample(
            stage, wall_s, sim_s, count, batch_size, cache_hit, source,
            metadata or {}, phase,
        )
        if len(self._samples) < self.max_samples:
            self._samples.append(sample)
        else:
            self.dropped_samples += 1
        return sample

    @contextmanager
    def measure(
        self,
        stage: str,
        *,
        sim_s: Optional[float] = None,
        count: int = 1,
        batch_size: Optional[int] = None,
        cache_hit: Optional[bool] = None,
        source: str = "hook",
        metadata: Optional[Mapping[str, Any]] = None,
        phase: str = "steady",
    ) -> Iterator[None]:
        if self.synchronize is not None:
            self.synchronize_calls += 1
            self.synchronize()
        started = float(self.clock())
        try:
            yield
        finally:
            if self.synchronize is not None:
                self.synchronize_calls += 1
                self.synchronize()
            elapsed = float(self.clock()) - started
            # A monotonic clock should not go backwards, but a test clock or a
            # platform clock can.  Keeping a zero sample is more useful than
            # hiding the exception that the wrapped operation raised.
            self.record(
                stage,
                max(0.0, elapsed),
                sim_s=sim_s,
                count=count,
                batch_size=batch_size,
                cache_hit=cache_hit,
                source=source,
                metadata=metadata,
                phase=phase,
            )

    timed = measure

    def wrap(
        self,
        stage: str,
        fn: Optional[Callable[..., R]] = None,
        *,
        count: int = 1,
        batch_size: Optional[int] = None,
        sim_s: Optional[float] = None,
        source: str = "hook",
        metadata: Optional[Mapping[str, Any]] = None,
        phase: str = "steady",
    ) -> Callable[..., R] | Callable[[Callable[..., R]], Callable[..., R]]:
        """Wrap a callable while preserving its return value and exceptions."""

        def decorate(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapped(*args: Any, **kwargs: Any) -> R:
                with self.measure(
                    stage,
                    sim_s=sim_s,
                    count=count,
                    batch_size=batch_size,
                    source=source,
                    metadata=metadata,
                    phase=phase,
                ):
                    return func(*args, **kwargs)

            return wrapped

        return decorate(fn) if fn is not None else decorate

    def samples(self) -> tuple[StageSample, ...]:
        return tuple(self._samples)

    def stage_stats(self) -> dict[str, StageStats]:
        return _group_stage_stats(self._samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [sample.to_dict() for sample in self._samples],
            "dropped_samples": self.dropped_samples,
            "synchronize_calls": self.synchronize_calls,
            "stages": {name: stats.to_dict() for name, stats in self.stage_stats().items()},
        }

    def clear(self) -> None:
        self._samples.clear()
        self.dropped_samples = 0
        self.synchronize_calls = 0


class BatchProbe:
    """Measure a stream of batches without changing the batch payload."""

    def __init__(self, profiler: StageProfiler, stage: str) -> None:
        self.profiler = profiler
        self.stage = str(stage)

    @contextmanager
    def measure(
        self,
        batch_size: int,
        *,
        items: Optional[int] = None,
        sim_s: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        source: str = "batch-hook",
        phase: str = "steady",
    ) -> Iterator[None]:
        size = _int_value(batch_size, "batch_size", minimum=1)
        count = size if items is None else _int_value(items, "items", minimum=1)
        with self.profiler.measure(
            self.stage,
            sim_s=sim_s,
            count=count,
            batch_size=size,
            cache_hit=cache_hit,
            source=source,
            phase=phase,
        ):
            yield

    def run(
        self,
        batches: Iterable[T],
        fn: Callable[[T], R],
        *,
        batch_size: Optional[int] = None,
        item_count: Optional[Callable[[T], int]] = None,
        source: str = "batch-hook",
        phase: str = "steady",
    ) -> Iterator[R]:
        """Yield ``fn(batch)`` results while timing every batch.

        This is intentionally a generator: it does not collect the complete
        flight trace or duplicate large frames just to measure a stage.
        """

        for batch in batches:
            size = batch_size
            if size is None:
                try:
                    size = len(batch)  # type: ignore[arg-type]
                except TypeError:
                    size = 1
            items = item_count(batch) if item_count is not None else size
            # Compute before yielding.  Keeping ``yield`` inside the context
            # would include the consumer's pause between generator pulls in
            # the stage timing.
            with self.measure(size, items=items, source=source, phase=phase):
                result = fn(batch)
            yield result


def iter_batches(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield bounded lists from any iterable; useful for measured warmups."""

    size = _int_value(batch_size, "batch_size", minimum=1)
    iterator = iter(items)
    while True:
        batch: list[T] = []
        try:
            for _ in range(size):
                batch.append(next(iterator))
        except StopIteration:
            pass
        if not batch:
            return
        yield batch


@dataclass(slots=True)
class CacheStats:
    """Counters emitted by :class:`CacheProbe`."""

    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_bytes: int = 0
    miss_bytes: int = 0

    @property
    def hit_rate(self) -> Optional[float]:
        return self.hits / self.requests if self.requests else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_bytes": self.hit_bytes,
            "miss_bytes": self.miss_bytes,
            "hit_rate": self.hit_rate,
        }


class CacheProbe:
    """A bounded LRU cache with timing/caching counters.

    It is intended for metadata, sampled frames, or mapping lookups.  The
    default capacity is small and the cache stores references (it never copies
    a tensor/array), so a caller can safely use it as an optional measurement
    hook.  ``max_entries=0`` disables retention while preserving miss timing.
    """

    def __init__(
        self,
        *,
        max_entries: int = 64,
        profiler: Optional[StageProfiler] = None,
        stage: str = "cache_lookup",
    ) -> None:
        if int(max_entries) < 0:
            raise ValueError("max_entries must be >= 0")
        self.max_entries = int(max_entries)
        self.profiler = profiler
        self.stage = str(stage)
        self._cache: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self.stats = CacheStats()

    def observe(self, hit: bool, *, bytes_: int = 0) -> None:
        """Record a result when an external cache owns the value storage."""

        amount = _int_value(bytes_, "bytes_", minimum=0)
        self.stats.requests += 1
        if hit:
            self.stats.hits += 1
            self.stats.hit_bytes += amount
        else:
            self.stats.misses += 1
            self.stats.miss_bytes += amount

    def get_or_compute(
        self,
        key: Any,
        factory: Callable[[], T],
        *,
        size_bytes: int = 0,
        batch_size: Optional[int] = None,
    ) -> T:
        amount = _int_value(size_bytes, "size_bytes", minimum=0)
        hit = key in self._cache
        if self.profiler is None:
            if hit:
                value, _ = self._cache.pop(key)
                self._cache[key] = (value, amount)
                self.observe(True, bytes_=amount)
                return value
            self.observe(False, bytes_=amount)
            value = factory()
            self._store(key, value, amount)
            return value
        with self.profiler.measure(
            self.stage,
            count=1,
            batch_size=batch_size,
            cache_hit=hit,
            source="cache-hook",
            metadata={"cache_key": repr(key)[:120]},
        ):
            if hit:
                value, stored_bytes = self._cache.pop(key)
                self._cache[key] = (value, stored_bytes)
                self.observe(True, bytes_=stored_bytes)
                return value
            self.observe(False, bytes_=amount)
            value = factory()
            self._store(key, value, amount)
            return value

    def _store(self, key: Any, value: Any, size_bytes: int) -> None:
        if self.max_entries <= 0:
            return
        self._cache[key] = (value, size_bytes)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
            self.stats.evictions += 1

    def clear(self, *, reset_stats: bool = False) -> None:
        self._cache.clear()
        if reset_stats:
            self.stats = CacheStats()

    def to_dict(self) -> dict[str, Any]:
        out = self.stats.to_dict()
        out.update({"max_entries": self.max_entries, "resident_entries": len(self._cache), "stage": self.stage})
        return out


def _timing_value(row: Mapping[str, Any]) -> tuple[Optional[float], Optional[str]]:
    """Return a wall duration in seconds from common artifact aliases."""

    for key in ("wall_s", "elapsed_s", "latency_s", "duration_s", "seconds", "time_s"):
        if key in row and row[key] is not None:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                return value, key
    for key in ("wall_ms", "elapsed_ms", "latency_ms", "duration_ms", "milliseconds", "time_ms"):
        if key in row and row[key] is not None:
            try:
                value = float(row[key]) / 1000.0
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                return value, key
    return None, None


def _sim_value(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("sim_s", "simulated_s", "simulation_s", "neural_s"):
        if key in row:
            value = _finite(row[key], key, allow_none=True)
            if value is not None and value >= 0:
                return value
    for key in ("sim_ms", "simulated_ms", "simulation_ms", "neural_ms"):
        if key in row:
            value = _finite(row[key], key, allow_none=True)
            if value is not None and value >= 0:
                return value / 1000.0
    return None


def _count_value(row: Mapping[str, Any], *, default: int = 1) -> int:
    for key in ("count", "items", "events", "steps", "frames", "batches"):
        if key in row and row[key] is not None:
            try:
                return _int_value(row[key], key, default=default, minimum=1)
            except ProfileError:
                return default
    return default


def _batch_value(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("batch_size", "batch", "batch_items"):
        if key in row and row[key] is not None:
            try:
                return _int_value(row[key], key, minimum=1)
            except ProfileError:
                return None
    return None


def _cache_value(row: Mapping[str, Any]) -> Optional[bool]:
    for key in ("cache_hit", "hit", "cached"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in {"1", "true", "yes", "hit"}:
                return True
            if value.lower() in {"0", "false", "no", "miss"}:
                return False
    return None


def _phase_value(row: Mapping[str, Any]) -> str:
    """Read the explicit warmup/steady marker used by profile artifacts."""

    value = row.get("phase", row.get("timing_phase", row.get("state", "steady")))
    # Do not guess from arbitrary state labels.  Unknown labels are kept as
    # steady for backwards compatibility with pre-phase artifacts.
    phase = str(value).strip().lower()
    return phase if phase in PROFILE_PHASES else "steady"


def _explicit_sample(row: Mapping[str, Any], source: str) -> Optional[StageSample]:
    stage_value = row.get("stage") or row.get("stage_name") or row.get("operation")
    if stage_value is None:
        return None
    wall, _ = _timing_value(row)
    if wall is None:
        return None
    try:
        metadata = _sample_metadata(row)
        nested_metadata = row.get("metadata")
        if isinstance(nested_metadata, Mapping):
            metadata.update(_sample_metadata(nested_metadata))
        return StageSample(
            str(stage_value), wall, _sim_value(row), _count_value(row),
            _batch_value(row), _cache_value(row), source,
            metadata, _phase_value(row),
        )
    except ProfileError:
        return None


def _compact_manifest_facts(value: Mapping[str, Any], source: str) -> dict[str, Any]:
    limits = value.get("resource_limits")
    usage = value.get("resource_usage")
    requested = value.get("requested_execution")
    effective = value.get("effective_backends")
    components = value.get("components") if isinstance(value.get("components"), Mapping) else {}
    vision = components.get("vision", {}) if isinstance(components, Mapping) else {}
    neural = components.get("neural", {}) if isinstance(components, Mapping) else {}
    vision_runtime = vision.get("runtime", {}) if isinstance(vision, Mapping) else {}
    neural_runtime = neural.get("runtime", {}) if isinstance(neural, Mapping) else {}
    selected = {
        "source": source,
        "schema_version": value.get("schema_version"),
        "scenario_names": value.get("scenario_names"),
        "requested_execution": requested,
        "effective_backends": effective,
        "resource_limits": limits,
        "resource_usage": usage,
        "vision_runtime": {
            key: vision_runtime.get(key)
            for key in ("backend", "device", "n_columns_per_eye", "n_targets", "units", "view")
            if key in vision_runtime
        },
        "neural_runtime": {
            key: neural_runtime.get(key)
            for key in ("backend", "backend_requested", "device", "n", "edges", "dt_ms", "max_vram_gib")
            if key in neural_runtime
        },
    }
    return _json_safe(selected)


def _resource_observation(value: Mapping[str, Any], source: str) -> Optional[dict[str, Any]]:
    limits = value.get("resource_limits") if isinstance(value.get("resource_limits"), Mapping) else {}
    usage = value.get("resource_usage") if isinstance(value.get("resource_usage"), Mapping) else {}
    if not limits and not usage:
        return None
    cuda = usage.get("cuda") if isinstance(usage.get("cuda"), Mapping) else {}
    observation = {
        "source": source,
        "ram_limit_bytes": limits.get("cgroup_memory_max_bytes"),
        "vram_limit_gib": limits.get("torch_allocator_max_vram_gib"),
        "peak_host_rss_bytes": usage.get("peak_host_rss_bytes"),
        "current_host_rss_bytes": usage.get("current_host_rss_bytes"),
        "peak_vram_allocated_bytes": cuda.get("peak_allocated_bytes"),
        "peak_vram_reserved_bytes": cuda.get("peak_reserved_bytes"),
        "vram_total_bytes": cuda.get("total_bytes"),
    }
    return _json_safe(observation)


def _parse_manifest(value: Mapping[str, Any], source: str) -> tuple[list[StageSample], dict[str, Any]]:
    samples: list[StageSample] = []
    # Only traverse explicit timing containers.  A telemetry row may contain
    # nested ``duration_s`` fields unrelated to a stage, and a manifest can
    # contain megabytes of visual arrays.
    for key in ("stage_timings", "timings", "profile", "profiling"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            samples.extend(_collect_explicit(nested, source))
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, Mapping):
                    sample = _explicit_sample(item, source)
                    if sample is not None:
                        samples.append(sample)
    top_sample = _explicit_sample(value, source)
    if top_sample is not None:
        samples.append(top_sample)
    facts: dict[str, Any] = {"manifests": [_compact_manifest_facts(value, source)]}
    resource = _resource_observation(value, source)
    if resource is not None:
        facts["resource_observations"] = [resource]
    scenario_results = value.get("scenario_results")
    if isinstance(scenario_results, Mapping):
        facts["scenario_durations_s"] = {
            str(name): item.get("duration_s")
            for name, item in scenario_results.items()
            if isinstance(item, Mapping) and item.get("duration_s") is not None
        }
    return samples, facts


def _collect_explicit(value: Any, source: str, default_stage: Optional[str] = None) -> list[StageSample]:
    samples: list[StageSample] = []
    if isinstance(value, Mapping):
        direct = _explicit_sample(value, source)
        if direct is None and default_stage is not None and _timing_value(value)[0] is not None:
            row = dict(value)
            row["stage"] = default_stage
            direct = _explicit_sample(row, source)
        if direct is not None:
            samples.append(direct)
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                child_stage = None if key in {
                    "samples", "rows", "stages", "stage_timings", "timings", "profile", "profiling"
                } else str(key)
                samples.extend(_collect_explicit(item, source, child_stage))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (Mapping, list, tuple)):
                samples.extend(_collect_explicit(item, source, default_stage))
    return samples


def _parse_summary(value: Mapping[str, Any], source: str) -> tuple[list[StageSample], dict[str, Any]]:
    samples: list[StageSample] = []
    wall, _ = _timing_value(value)
    sim_s = _sim_value(value)
    if wall is not None and sim_s is not None:
        backend = str(value.get("backend", "unknown"))
        dt_ms = _finite(value.get("dt_ms"), "dt_ms", allow_none=True)
        count = max(1, int(round(sim_s / (dt_ms / 1000.0)))) if dt_ms and dt_ms > 0 else _count_value(value)
        samples.append(StageSample(
            "malecns_open_loop", wall, sim_s, count, None, None, source,
            {
                "backend": backend,
                "kind": "open_loop",
                "evidence": "artifact_summary",
                "graph": value.get("graph"),
                "real": backend in {"torch", "torch_cuda", "triton"},
                "stimulus": value.get("stim", []),
            },
        ))
    explicit = _explicit_sample(value, source)
    if explicit is not None:
        samples.append(explicit)
    facts: dict[str, Any] = {"summaries": [{
        "source": source,
        "backend": value.get("backend"),
        "graph": value.get("graph"),
        "neurons": value.get("neurons"),
        "edges": value.get("edges"),
        "dt_ms": value.get("dt_ms"),
        "simulated_ms": value.get("simulated_ms"),
        "wall_s": value.get("wall_s"),
        "real_time_factor": value.get("real_time_factor"),
        "total_spikes": value.get("total_spikes"),
        "total_deliveries": value.get("total_deliveries"),
    }]}
    resource = {
        "source": source,
        "ram_limit_bytes": None,
        # ``gate_shmem_limit_gib`` is a host Shmem safety ceiling, not a VRAM
        # allocator limit.  Keep it out of the VRAM budget calculation.
        "vram_limit_gib": None,
        "shmem_limit_gib": value.get("gate_shmem_limit_gib"),
        "peak_host_rss_bytes": value.get("peak_host_rss_bytes"),
        "current_host_rss_bytes": None,
        "peak_vram_allocated_bytes": (value.get("gpu_memory") or {}).get("peak_allocated_bytes")
        if isinstance(value.get("gpu_memory"), Mapping) else None,
        "peak_vram_reserved_bytes": (value.get("gpu_memory") or {}).get("peak_reserved_bytes")
        if isinstance(value.get("gpu_memory"), Mapping) else None,
        "vram_total_bytes": (value.get("gpu_memory") or {}).get("total_bytes")
        if isinstance(value.get("gpu_memory"), Mapping) else None,
    }
    if any(item is not None for key, item in resource.items() if key != "source"):
        facts["resource_observations"] = [resource]
    return samples, facts


def _parse_flight(value: Mapping[str, Any], source: str) -> tuple[list[StageSample], dict[str, Any]]:
    samples: list[StageSample] = []
    wall, _ = _timing_value(value)
    frames = _int_value(value.get("frames"), "frames", default=0, minimum=0)
    frame_ms = _finite(value.get("frame_ms"), "frame_ms", allow_none=True)
    if wall is not None and frames > 0:
        sim_s = frames * (frame_ms / 1000.0) if frame_ms and frame_ms > 0 else None
        stage = "legacy_flyvis_closed_loop" if "flyvis" in source.lower() else "legacy_closed_loop"
        samples.append(StageSample(
            stage, wall, sim_s, frames, 1, None, source,
            {"kind": "legacy_proxy", "legacy": True, "evidence": "flight_json", "real": True},
        ))
    explicit = _explicit_sample(value, source)
    if explicit is not None:
        samples.append(explicit)
    return samples, {"flight_summaries": [{
        "source": source,
        "frames": value.get("frames"),
        "frame_ms": value.get("frame_ms"),
        "wall_s": value.get("wall_s"),
        "vision": "flyvis" if "flyvis" in source.lower() else None,
        "legacy": True,
    }]}


def _parse_json_file(path: Path) -> tuple[list[StageSample], dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"could not read JSON artifact {path}: {exc}") from exc
    source = str(path)
    if not isinstance(value, (Mapping, list)):
        return [], {"warnings": [f"ignored non-object JSON artifact: {source}"]}
    # A profiler report already contains canonical ``sample_rows``.  Do not
    # recursively inspect its facts/projection maps: fields such as
    # ``profile_setup.wall_s`` are metadata, not a measured pipeline stage.
    if (
        isinstance(value, Mapping)
        and str(value.get("schema_version", "")).startswith("flight-gpu-profile-")
    ):
        rows: list[StageSample] = []
        for item in value.get("sample_rows", ()):
            if isinstance(item, Mapping):
                sample = _explicit_sample(item, source)
                if sample is not None:
                    rows.append(sample)
        facts: dict[str, Any] = {
            "profile_reports": [{
                "source": source,
                "schema_version": value.get("schema_version"),
                "pipeline_stage_order": value.get("pipeline_stage_order"),
                "sample_count": value.get("sample_count"),
            }],
        }
        raw_facts = value.get("facts")
        if isinstance(raw_facts, Mapping):
            for key in ("profile_config", "profile_setup", "resource_envelope",
                        "pipeline_stage_order", "stage_contract",
                        "cuda_synchronization_calls", "cuda_synchronization_expected",
                        "stage_profiler", "sources", "renderer", "renderer_timing"):
                if key in raw_facts:
                    facts[key] = raw_facts[key]
            if isinstance(value.get("renderer"), Mapping):
                facts.setdefault("renderer", value["renderer"])
            if isinstance(value.get("renderer_backend"), Mapping):
                facts.setdefault("renderer_backend", value["renderer_backend"])
            observations = raw_facts.get("resource_observations")
            if isinstance(observations, list):
                facts["resource_observations"] = observations
        return rows, facts
    name = path.name.lower()
    if isinstance(value, Mapping) and name == "manifest.json":
        return _parse_manifest(value, source)
    if isinstance(value, Mapping) and name == "summary.json":
        return _parse_summary(value, source)
    if isinstance(value, Mapping) and name == "flight.json":
        return _parse_flight(value, source)
    return _collect_explicit(value, source), {}


def _parse_jsonl_file(path: Path) -> tuple[list[StageSample], dict[str, Any]]:
    samples: list[StageSample] = []
    warnings: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ProfileError(f"could not read JSONL artifact {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
                continue
            if isinstance(row, Mapping):
                sample = _explicit_sample(row, f"{path}:{line_number}")
                if sample is not None:
                    samples.append(sample)
                for key in ("stage_timing", "stage_timings", "timings", "profile", "profiling"):
                    nested = row.get(key)
                    if isinstance(nested, (Mapping, list, tuple)):
                        samples.extend(_collect_explicit(nested, f"{path}:{line_number}"))
    return samples, {"warnings": warnings} if warnings else {}


def _parse_csv_file(path: Path) -> tuple[list[StageSample], dict[str, Any]]:
    samples: list[StageSample] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample = _explicit_sample(row, str(path))
                if sample is not None:
                    samples.append(sample)
    except OSError as exc:
        raise ProfileError(f"could not read CSV artifact {path}: {exc}") from exc
    return samples, {}


def _artifact_files(path: Path, *, include_telemetry: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ProfileError(f"artifact path does not exist: {path}")
    names = {"manifest.json", "summary.json", "flight.json"}
    if include_telemetry:
        names.add("telemetry.jsonl")
    files: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in names:
            files.append(candidate)
        elif candidate.name.startswith("profile") and candidate.suffix.lower() in {".json", ".jsonl", ".csv"}:
            files.append(candidate)
    return files


def profile_artifacts(
    paths: Sequence[str | Path],
    *,
    include_telemetry: bool = False,
) -> "ProfileReport":
    """Read artifacts without importing or launching any runtime."""

    if not paths:
        paths = (DEFAULT_ARTIFACT_ROOT,)
    files: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        for path in _artifact_files(Path(raw), include_telemetry=include_telemetry):
            resolved = path.resolve()
            if resolved not in seen:
                files.append(path)
                seen.add(resolved)
    samples: list[StageSample] = []
    facts: dict[str, Any] = {"sources": [str(path) for path in files]}
    warnings: list[str] = []
    for path in files:
        try:
            if path.suffix.lower() == ".jsonl":
                rows, extra = _parse_jsonl_file(path)
            elif path.suffix.lower() == ".csv":
                rows, extra = _parse_csv_file(path)
            elif path.suffix.lower() == ".json":
                rows, extra = _parse_json_file(path)
            else:
                continue
        except ProfileError as exc:
            warnings.append(str(exc))
            continue
        samples.extend(rows)
        for key, value in extra.items():
            if key == "warnings":
                warnings.extend(str(item) for item in value)
            elif isinstance(value, list):
                facts.setdefault(key, []).extend(value)
            elif isinstance(value, Mapping) and isinstance(facts.get(key), Mapping):
                facts[key] = {**facts[key], **value}
            else:
                facts[key] = value
    return ProfileReport(samples=samples, facts=facts, warnings=warnings)


def _theil_sen(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    slopes: list[float] = []
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            dx = xs[right] - xs[left]
            if abs(dx) > 1.0e-12:
                slopes.append((ys[right] - ys[left]) / dx)
    slope = statistics.median(slopes) if slopes else statistics.median(
        (y / x for x, y in zip(xs, ys) if x > 1.0e-12),
    )
    slope = max(0.0, float(slope))
    intercept = statistics.median(y - slope * x for x, y in zip(xs, ys))
    return max(0.0, float(slope)), max(0.0, float(intercept))


@dataclass(frozen=True, slots=True)
class StageEstimate:
    """Robust affine fit and projected wall time for one stage."""

    stage: str
    samples: int
    axis: str
    slope_s_per_unit: Optional[float]
    intercept_s: Optional[float]
    p50_s: Optional[float]
    p95_s: Optional[float]
    projected_s: Optional[float]
    target_units: Optional[float]
    confidence: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "samples": self.samples,
            "axis": self.axis,
            "slope_s_per_unit": self.slope_s_per_unit,
            "intercept_s": self.intercept_s,
            "wall_p50_s": self.p50_s,
            "wall_p95_s": self.p95_s,
            "projected_s": self.projected_s,
            "target_units": self.target_units,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def fit_stage(stats: StageStats, *, target_sim_s: Optional[float] = None, target_count: Optional[int] = None) -> StageEstimate:
    """Fit a robust startup-plus-throughput model to one stage."""

    values = stats.wall_values
    sim_rows = [(sample.sim_s, sample.wall_s) for sample in stats.samples if sample.sim_s is not None and sample.sim_s > 0]
    # A frame-oriented caller (the historical FlyVis flight proxy) should be
    # projected by frame count even though the artifact also records its
    # simulated seconds.  All other callers prefer the simulation-time axis.
    use_event_axis = target_count is not None and target_sim_s is None
    if len(sim_rows) >= 1 and not use_event_axis:
        xs = [float(item[0]) for item in sim_rows]
        ys = [float(item[1]) for item in sim_rows]
        if len(sim_rows) >= 2 and max(xs) - min(xs) > 1.0e-12:
            slope, intercept = _theil_sen(xs, ys)
            confidence = "measured-fit" if len(sim_rows) >= 3 else "two-point-fit"
        else:
            slope = statistics.median(y / x for x, y in sim_rows if x > 0)
            intercept = 0.0
            confidence = "single-scale"
        target = target_sim_s
        projected = intercept + slope * target if target is not None else None
        return StageEstimate(
            stats.stage, len(stats.samples), "sim_s", slope, intercept,
            _percentile(values, 0.50), _percentile(values, 0.95), projected,
            target, confidence,
            tuple(sorted({sample.metadata.get("evidence", "artifact") for sample in stats.samples})),
        )
    event_rows = [(sample.count, sample.wall_s) for sample in stats.samples if sample.count > 0]
    if not event_rows:
        return StageEstimate(stats.stage, 0, "none", None, None, None, None, None, None, "no-data", ())
    event_x = [float(count) for count, _ in event_rows]
    event_y = [float(wall) for _, wall in event_rows]
    if len(event_rows) >= 2 and max(event_x) - min(event_x) > 1.0e-12:
        slope, intercept = _theil_sen(event_x, event_y)
        confidence = "measured-fit"
    else:
        slope = statistics.median(wall / count for count, wall in event_rows)
        intercept = 0.0
        confidence = "event-rate" if len(event_rows) >= 2 else "single-event"
    target = float(target_count) if target_count is not None else None
    projected = intercept + slope * target if target is not None else None
    return StageEstimate(
        stats.stage, len(stats.samples), "count", slope, intercept,
        _percentile(values, 0.50), _percentile(values, 0.95), projected,
        target, confidence,
        tuple(sorted({sample.metadata.get("evidence", "artifact") for sample in stats.samples})),
    )


def recommended_smoke_ladder() -> tuple[dict[str, Any], ...]:
    """Return the safe, timestamp-aligned ladder documented for this project."""

    rows = (
        ("preflight", 0.0, 0, 0, 10.0, "read-only gate/assets/resource checks"),
        ("one-frame", 0.005, 1, 10, 300.0, "current real FlyVis + MaleCNS closed-loop baseline"),
        ("ten-frame", 0.050, 10, 100, 300.0, "hold/caching/timestamp continuity"),
        ("warm-100-frame", 0.500, 100, 1000, 600.0, "steady stage timing and first neural activity"),
        ("e20-length", 2.000, 400, 4000, 900.0, "compare with the historical 400-frame FlyVis flight"),
        ("sustained-10s", 10.000, 2000, 20000, 1800.0, "first sustained flight candidate under the budget"),
    )
    return tuple({
        "name": name,
        "simulated_s": sim_s,
        "flyvis_frames": frames,
        "malecns_steps": steps,
        "wall_budget_s": budget,
        "purpose": purpose,
        "frame_dt_s": FRAME_DT_S,
        "cns_dt_s": 1.0 / CNS_HZ,
    } for name, sim_s, frames, steps, budget, purpose in rows)


@dataclass(frozen=True, slots=True)
class GPUProfileConfig:
    """Bounded configuration for :func:`run_gpu_profile`.

    The defaults are deliberately tiny: one warmup frame followed by two
    recorded 5 ms windows.  A caller can request more frames, but the
    function rejects an unbounded run and records the resource envelope in
    the resulting report.  Runtime imports are kept inside the runner so
    importing this module remains dependency-light.
    """

    frames: int = 2
    warmup_frames: int = 1
    width: int = 32
    height: int = 24
    vision_device: str = "cuda:0"
    neural_backend: str = "torch"
    max_vram_gib: float = 15.5
    ram_limit_gib: float = 5.0
    seed: int = 1729
    scenario: str = "hover"
    config_path: Optional[str | Path] = None
    csr_path: Optional[str | Path] = None
    source: str = "real-gpu-profile"
    # ``analytic_reference`` is the deterministic scientific baseline.  Warp
    # is opt-in and the launcher fails closed when its external adapter is not
    # installed; this profiler never substitutes the analytic renderer.  Keep
    # this field at the end so older positional construction remains valid.
    renderer_backend: str = "analytic_reference"

    def __post_init__(self) -> None:
        frames = _int_value(self.frames, "frames", minimum=1)
        warmup = _int_value(self.warmup_frames, "warmup_frames", minimum=0)
        if warmup > frames:
            raise ProfileError("warmup_frames must be <= frames")
        width = _int_value(self.width, "width", minimum=1)
        height = _int_value(self.height, "height", minimum=1)
        vram = _finite(self.max_vram_gib, "max_vram_gib")
        ram = _finite(self.ram_limit_gib, "ram_limit_gib")
        assert vram is not None and ram is not None
        if vram <= 0 or ram <= 0:
            raise ProfileError("max_vram_gib and ram_limit_gib must be positive")
        device = str(self.vision_device).strip()
        if device != "cpu" and not device.startswith("cuda:"):
            raise ProfileError("vision_device must be cpu or cuda:N")
        backend = str(self.neural_backend).strip().lower()
        if backend not in {"torch", "torch_cpu", "triton", "cpu", "auto"}:
            raise ProfileError("neural_backend must be torch, torch_cpu, triton, cpu, or auto")
        renderer = str(self.renderer_backend).strip().lower()
        if renderer not in {"analytic_reference", "warp_cuda"}:
            raise ProfileError(
                "renderer_backend must be analytic_reference or warp_cuda"
            )
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "warmup_frames", warmup)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "renderer_backend", renderer)
        object.__setattr__(self, "max_vram_gib", vram)
        object.__setattr__(self, "ram_limit_gib", ram)
        object.__setattr__(self, "vision_device", device)
        object.__setattr__(self, "neural_backend", backend)
        object.__setattr__(self, "scenario", str(self.scenario).strip() or "hover")

    @property
    def simulated_s(self) -> float:
        return float(self.frames) * FRAME_DT_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "steady_frames": self.frames - self.warmup_frames,
            "width": self.width,
            "height": self.height,
            "renderer_backend": self.renderer_backend,
            "rectangular_rays_per_frame": 2 * self.width * self.height,
            "exact_columns_per_eye": 721,
            "exact_rays_per_frame": 2 * 721,
            "combined_diagnostic_rays_per_frame": 2 * self.width * self.height + 2 * 721,
            "vision_device": self.vision_device,
            "neural_backend": self.neural_backend,
            "max_vram_gib": self.max_vram_gib,
            "ram_limit_gib": self.ram_limit_gib,
            "seed": self.seed,
            "scenario": self.scenario,
            "config_path": None if self.config_path is None else str(self.config_path),
            "csr_path": None if self.csr_path is None else str(self.csr_path),
            "source": self.source,
        }


def _cgroup_memory_limit_bytes_for_profile() -> Optional[int]:
    """Read the current cgroup-v2 memory ceiling without shelling out."""

    candidates = [Path("/sys/fs/cgroup/memory.max")]
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].lstrip("/")
                candidates.insert(0, Path("/sys/fs/cgroup") / relative / "memory.max")
                break
    except OSError:
        pass
    for path in candidates:
        try:
            text = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if text and text != "max":
            try:
                return int(text)
            except ValueError:
                continue
    return None


def _runtime_memory_snapshot(*, cuda_device: str = "cuda:0",
                             ram_limit_gib: Optional[float] = None,
                             vram_limit_gib: Optional[float] = None) -> dict[str, Any]:
    """Capture RAM/VRAM counters at an already-synchronized boundary.

    This function intentionally does *not* call ``torch.cuda.synchronize``.
    GPU synchronization belongs to :class:`StageProfiler`'s measurement
    boundaries; inserting a hidden sync here would make a memory probe change
    the timing it is supposed to describe.
    """

    import resource as _resource

    peak_rss = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss) * 1024
    current_rss: Optional[int] = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                current_rss = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    ram_limit_bytes = _cgroup_memory_limit_bytes_for_profile()
    ram_limit_source = "cgroup" if ram_limit_bytes is not None else None
    if ram_limit_bytes is None and ram_limit_gib is not None:
        ram_limit_bytes = int(float(ram_limit_gib) * GIB)
        ram_limit_source = "profile_config"
    result: dict[str, Any] = {
        "ram_limit_bytes": ram_limit_bytes,
        "ram_limit_gib": None if ram_limit_bytes is None else ram_limit_bytes / GIB,
        "peak_host_rss_bytes": peak_rss,
        "current_host_rss_bytes": current_rss,
        "peak_host_rss_gib": peak_rss / GIB,
        "ram_limit_source": ram_limit_source,
        "ram_limit_enforced": ram_limit_source == "cgroup",
    }
    # The module is looked up rather than imported.  This keeps CPU/artifact
    # reads from touching CUDA, while still collecting allocator counters in a
    # real run after the runner has explicitly selected its CUDA backend.
    torch = sys.modules.get("torch")
    if torch is not None and str(cuda_device).strip().lower() != "cpu":
        try:
            if bool(torch.cuda.is_available()):
                device = torch.device(cuda_device)
                allocated = int(torch.cuda.memory_allocated(device))
                reserved = int(torch.cuda.memory_reserved(device))
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                total = int(torch.cuda.get_device_properties(device).total_memory)
                result["cuda"] = {
                    "device": str(device),
                    "total_bytes": total,
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "total_gib": total / GIB,
                    "allocated_gib": allocated / GIB,
                    "reserved_gib": reserved / GIB,
                    "peak_allocated_gib": peak_allocated / GIB,
                    "peak_reserved_gib": peak_reserved / GIB,
                    "vram_limit_gib": vram_limit_gib,
                }
                result["peak_vram_allocated_bytes"] = peak_allocated
                result["peak_vram_reserved_bytes"] = peak_reserved
                result["peak_vram_allocated_gib"] = peak_allocated / GIB
                result["peak_vram_reserved_gib"] = peak_reserved / GIB
                result["vram_limit_gib"] = vram_limit_gib
        except Exception as exc:
            result["cuda_measurement_error"] = f"{type(exc).__name__}: {exc}"
    return _json_safe(result)


def _profile_gpu_runner_imports() -> tuple[Any, Any]:
    """Import the existing launcher lazily for the real-GPU profile only."""

    script_dir = ROOT / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        module = __import__("run_digital_fly")
    except Exception as exc:
        raise ProfileError(f"could not import the digital-fly launcher: {exc}") from exc
    return module, getattr(module, "DigitalFlyRunner", None)


def _profile_raycast_batch(world: Any, pose: Any, *, width: int, height: int,
                           timestamp_s: float) -> int:
    """Run the public raycast for every rectangular stereo ray.

    ``render_stereo`` calls the same private trace implementation internally,
    but this explicit pass makes the geometry/raycast cost visible as its own
    stage.  The count is exact (2*width*height), not a sample extrapolation.
    """

    ray = getattr(world, "_ray", None)
    if not callable(ray):
        raise ProfileError("VirtualWorld does not expose its rectangular ray generator")
    camera = getattr(world, "camera", None)
    if camera is None:
        raise ProfileError("VirtualWorld does not expose camera geometry")
    traced = 0
    for eye in ("left", "right"):
        for py in range(int(height)):
            for px in range(int(width)):
                origin, direction = ray(pose, eye, px, py, camera)
                world.trace_ray(origin, direction, timestamp_s=timestamp_s)
                traced += 1
    return traced


def _profile_body_command(nm_output: Any) -> dict[str, float]:
    """Convert one real NM output to the reduced body's public command map."""

    return {
        "left_wing": float(getattr(nm_output, "left_wing_power", 0.0)),
        "right_wing": float(getattr(nm_output, "right_wing_power", 0.0)),
        "thrust": float(getattr(nm_output, "wing_power", 0.0)),
        "yaw": float(getattr(nm_output, "wing_steering", 0.0)),
        "pitch": 0.0,
        "roll": 0.0,
        "brake": 0.0,
    }


def _profile_exact_flyvis_step(runner: Any, left_columns: Any,
                               right_columns: Any, timestamp_s: float) -> Any:
    """Run the fitted runner's exact-column FlyVis seam.

    ``StereoFlyVis.process_frame`` is the raster/frontend path used by the
    small deterministic backend.  A fitted runtime must instead use the
    launcher's ``_process_exact_flyvis_columns`` seam so that the profile has
    the same device-resident input contract as production.  Keep this helper
    narrow so it can be mocked without importing FlyVis or CUDA.
    """

    method = getattr(runner, "_process_exact_flyvis_columns", None)
    if not callable(method):
        raise ProfileError(
            "real GPU profile requires the runner._process_exact_flyvis_columns seam"
        )
    try:
        return method(left_columns, right_columns, timestamp_s)
    except Exception as exc:
        raise ProfileError(
            f"exact-column FlyVis processing failed at t={float(timestamp_s):g}s: {exc}"
        ) from exc


def _profile_hash_only_record(value: Any) -> dict[str, Any]:
    """Describe an array-like value without serializing its elements.

    The real compact telemetry contract keeps large visual arrays as a
    lossless record.  A bounded profiler should measure the hot JSON path,
    not allocate/compress another copy of every 721-column or T4/T5 vector.
    Hashing an already-host-resident value is representative of the minimal
    audit record and deliberately has no NumPy/Torch dependency.
    """

    candidate = value
    shape = getattr(candidate, "shape", None)
    if shape is None and isinstance(candidate, (list, tuple)):
        shape = (len(candidate),)
    try:
        shape_value = [int(size) for size in shape] if shape is not None else None
    except (TypeError, ValueError):
        shape_value = None
    dtype = str(getattr(candidate, "dtype", "float32"))
    raw: Optional[bytes] = None
    tobytes = getattr(candidate, "tobytes", None)
    if callable(tobytes):
        try:
            raw = bytes(tobytes(order="C"))
        except (TypeError, ValueError):
            try:
                raw = bytes(tobytes())
            except (TypeError, ValueError):
                raw = None
    if raw is None and isinstance(candidate, (bytes, bytearray, memoryview)):
        raw = bytes(candidate)
    if raw is None:
        # Do not turn a large list into a second JSON representation merely to
        # hash it.  This fallback is still an explicit audit marker.
        return {
            "encoding": "hash-unavailable",
            "dtype": dtype,
            "shape": shape_value,
        }
    return {
        "encoding": "hash-only",
        "lossless": True,
        "dtype": dtype,
        "shape": shape_value,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _profile_compact_record(
    frame_index: int, timestamp: Any, flyvis_output: Any,
    neural_output: Any, body_command: Mapping[str, Any], dynamics_output: Any,
    dynamics: Any,
) -> dict[str, Any]:
    """Build the bounded/hash-only record used by the profiler's JSON timer."""

    left = getattr(flyvis_output, "left_columns", getattr(flyvis_output, "left", None))
    right = getattr(flyvis_output, "right_columns", getattr(flyvis_output, "right", None))
    drive = getattr(flyvis_output, "t4t5_drive", getattr(flyvis_output, "drive", None))
    named = getattr(neural_output, "named_outputs", {})
    if not isinstance(named, Mapping):
        named = {}
    state = getattr(dynamics_output, "state", getattr(dynamics, "state", None))
    state_record = state.to_dict() if callable(getattr(state, "to_dict", None)) else state
    return _json_safe({
        "schema_version": "digital-fly-telemetry-compact-1",
        "frame_index": int(frame_index),
        "timestamp": timestamp.to_dict() if callable(getattr(timestamp, "to_dict", None)) else timestamp,
        "visual_input": {
            "left_columns": _profile_hash_only_record(left),
            "right_columns": _profile_hash_only_record(right),
            "t4t5_drive": _profile_hash_only_record(drive),
        },
        "neural_window": {
            "steps": 10,
            "spike_count": int(named.get("spike_count", 0)),
            "backend": named.get("backend"),
        },
        "motor": dict(body_command),
        "body": state_record,
    })


def run_gpu_profile(config: GPUProfileConfig | None = None, **kwargs: Any) -> "ProfileReport":
    """Measure a bounded real FlyVis -> MaleCNS -> NM/body pipeline.

    The existing launcher remains the source of truth for component
    construction and contracts.  This helper only wraps calls at boundaries;
    it never patches or changes ``engine``, ``flyvis_runtime_adapter``, or the
    launcher loop.  ``warmup`` and ``steady`` samples are explicit and all
    CUDA synchronization is performed by the stage hook immediately around a
    measured operation.

    The returned :class:`ProfileReport` contains stage samples and compact
    memory observations.  Renderer provenance is explicit: the requested and
    effective backend are recorded, and a Warp run keeps first-call cold-JIT
    cost separate from steady-state rectangular/exact-column timings.  Warp
    setup failures propagate as ``ProfileError``; there is no analytic
    fallback.  Callers can write it with ``to_dict`` or pass an output path
    through ``kwargs['out']``.
    """

    if config is None:
        config = GPUProfileConfig(**kwargs)
    elif kwargs:
        raise ProfileError("pass either GPUProfileConfig or keyword options, not both")
    if not isinstance(config, GPUProfileConfig):
        raise ProfileError("config must be GPUProfileConfig or None")
    launcher_module, runner_class = _profile_gpu_runner_imports()
    if runner_class is None:
        raise ProfileError("run_digital_fly does not expose DigitalFlyRunner")
    RunOptions = getattr(launcher_module, "RunOptions", None)
    if RunOptions is None:
        raise ProfileError("run_digital_fly does not expose RunOptions")

    config_path = Path(config.config_path) if config.config_path is not None else ROOT / "config" / "scenarios.json"
    setup_started = time.perf_counter()
    try:
        options = RunOptions(
            config=config_path,
            out=ROOT / "artifacts" / "profile-gpu-transient",
            scenarios=(config.scenario,),
            neural="malecns",
            vision="flyvis",
            body="reduced",
            renderer_backend=config.renderer_backend,
            seed=int(config.seed),
            width=int(config.width),
            height=int(config.height),
            columns=721,
            t4t5=13582,
            neural_device="cuda" if config.neural_backend not in {"cpu", "torch_cpu"} else "cpu",
            neural_backend=config.neural_backend,
            vision_device=config.vision_device,
            max_vram_gib=float(config.max_vram_gib),
            duration_override_s=config.simulated_s,
            max_steps=int(config.frames),
            frame_count=int(config.frames),
            neural_csr_path=None if config.csr_path is None else Path(config.csr_path),
        )
        runner = runner_class(options)
    except Exception as exc:
        raise ProfileError(f"real GPU runner setup failed: {type(exc).__name__}: {exc}") from exc
    setup_wall_s = time.perf_counter() - setup_started

    effective_renderer = str(getattr(
        runner.world,
        "renderer_backend",
        getattr(runner.world, "backend_role", config.renderer_backend),
    ))
    renderer_object = getattr(runner.world, "renderer", None)
    renderer_manifest: Any = None
    renderer_manifest_error: Optional[str] = None
    manifest_method = getattr(renderer_object, "manifest", None)
    if callable(manifest_method):
        try:
            renderer_manifest = _json_safe(manifest_method())
        except Exception as exc:
            # The timing run is still useful, but a broken optional manifest
            # must remain visible rather than being silently discarded.
            renderer_manifest_error = f"{type(exc).__name__}: {exc}"

    torch = sys.modules.get("torch")
    synchronize: Optional[Callable[[], Any]] = None
    if torch is not None and config.vision_device != "cpu":
        try:
            device = torch.device(config.vision_device)
            if bool(torch.cuda.is_available()):
                synchronize = lambda: torch.cuda.synchronize(device)
        except Exception:
            synchronize = None
    profiler = StageProfiler(synchronize=synchronize)
    facts: dict[str, Any] = {
        "sources": [config.source],
        "profile_config": config.to_dict(),
        "renderer": {
            "requested_backend": config.renderer_backend,
            "effective_backend": effective_renderer,
            "backend_match": effective_renderer == config.renderer_backend,
            "optional_renderer_manifest": renderer_manifest,
            "optional_renderer_manifest_error": renderer_manifest_error,
            "cold_jit_policy": (
                "first rectangular_stereo_render and exact_column_render invocation"
                if config.renderer_backend == "warp_cuda" else "not_applicable"
            ),
        },
        "profile_setup": {
            "wall_s": setup_wall_s,
            "phase": "warmup",
            "construction_is_unmeasured": True,
            "cuda_synchronization_boundaries": bool(synchronize is not None),
            "synchronization_policy": "only before/after StageProfiler.measure; no inner-loop synchronization",
        },
        "resource_observations": [],
        "pipeline_stage_order": list(PIPELINE_STAGES),
        "stage_contract": {
            "world_raycast": "2*width*height public trace_ray calls",
            "rectangular_stereo_render": "VirtualWorld.render_stereo, telemetry/video frame only",
            "exact_column_render": "VirtualWorld.render_eye_directions, 721 columns per eye",
            "flyvis": "DigitalFlyRunner._process_exact_flyvis_columns on exact columns",
            "malecns_10tick_window": "one held-input engine.step_window(..., steps=10)",
            "neuromuscular": "one NeuromuscularAdapter.step per 5 ms window",
            "body": "one FlightDynamics.step per 5 ms window",
            "serialization": "one compact JSON record per frame",
        },
    }
    # Keep explicit progress separate from the sample count.  A single frame
    # contributes several samples, and an exception can occur before any
    # sample is recorded; neither is a reliable frame index.
    stage_samples: list[StageSample] = []
    completed_frames = 0
    active_frame_index: Optional[int] = None

    try:
        scenario = runner.world.scenarios.get(config.scenario)
        if scenario is None:
            raise ProfileError(f"profile scenario {config.scenario!r} is not configured")
        dynamics = runner._new_body(scenario)
        neural = runner._new_neural(config.scenario)
        runner.flyvis.reset()
        reset_nm = getattr(runner.neuromuscular, "reset", None)
        if callable(reset_nm):
            reset_nm(clear_baseline=False)
        for frame_index in range(config.frames):
            active_frame_index = frame_index
            phase = "warmup" if frame_index < config.warmup_frames else "steady"
            t_s = frame_index * FRAME_DT_S
            timestamp = launcher_module._timestamp(t_s)
            state_before = dynamics.state.copy()
            pose = launcher_module._pose_from_state(state_before)
            memory_before = _runtime_memory_snapshot(
                cuda_device=config.vision_device,
                ram_limit_gib=config.ram_limit_gib,
                vram_limit_gib=config.max_vram_gib,
            )

            rays = 2 * config.width * config.height
            with profiler.measure(
                "world_raycast", sim_s=FRAME_DT_S, count=rays, batch_size=rays,
                source=config.source, phase=phase,
                metadata={"units": "rays", "ray_count": rays, "path": "public_trace_ray",
                          "renderer_backend": "analytic_reference",
                          "diagnostic_only": config.renderer_backend == "warp_cuda"},
            ):
                _profile_raycast_batch(
                    runner.world, pose, width=config.width, height=config.height,
                    timestamp_s=t_s,
                )
            with profiler.measure(
                "rectangular_stereo_render", sim_s=FRAME_DT_S, count=1,
                batch_size=rays, source=config.source, phase=phase,
                metadata={"units": "frames", "ray_count": rays, "resolution": [config.width, config.height],
                          "eyes": 2, "path": "render_stereo",
                          "renderer_backend": effective_renderer,
                          "cold_jit_candidate": bool(
                              config.renderer_backend == "warp_cuda" and frame_index == 0
                          )},
            ):
                stereo = runner.world.render_stereo(
                    pose, (config.width, config.height), timestamp_s=t_s,
                )
            # ``visual_observation`` is contract glue, not a raycast path.  It
            # is kept out of the render timer so the two render paths compare
            # only their actual world sampling work.
            visual_world = runner.world.visual_observation(stereo, t_s, dt_s=FRAME_DT_S)

            exact_rays = 2 * 721
            with profiler.measure(
                "exact_column_render", sim_s=FRAME_DT_S, count=1,
                batch_size=exact_rays, source=config.source, phase=phase,
                metadata={"units": "frames", "ray_count": exact_rays,
                          "columns_per_eye": 721, "eyes": 2, "path": "render_eye_directions",
                          "renderer_backend": effective_renderer,
                          "cold_jit_candidate": bool(
                              config.renderer_backend == "warp_cuda" and frame_index == 0
                          )},
            ):
                left_columns, right_columns, exact_meta = runner._free_exact_flyvis_columns(
                    pose, t_s,
                )

            with profiler.measure(
                "flyvis", sim_s=FRAME_DT_S, count=1, batch_size=exact_rays,
                source=config.source, phase=phase,
                metadata={"units": "frames", "columns_per_eye": 721,
                          "backend": getattr(runner.flyvis_runtime_adapter, "backend_name", "flyvis"),
                          "input_path": "exact_column_render"},
            ):
                flyvis_out = _profile_exact_flyvis_step(
                    runner, left_columns, right_columns, t_s,
                )

            body_contract = launcher_module._make_contract_body(state_before, timestamp)
            channels = launcher_module._flyvis_drive_contract(
                runner.flyvis, flyvis_out.t4t5_drive,
            )
            sensory = launcher_module.SensoryFrame(
                timestamp=timestamp, body=body_contract, channels=channels,
                source=launcher_module.SourceLabel.HANDBUILT,
            )
            with profiler.measure(
                "malecns_10tick_window", sim_s=FRAME_DT_S, count=10,
                batch_size=10, source=config.source, phase=phase,
                metadata={"units": "cns_ticks", "steps": 10, "dt_ms": 0.5,
                          "input_hold": "one_flyvis_frame"},
            ):
                neural_output = neural.step_window(
                    sensory, timestamp, steps=CNS_STEPS_PER_FLYVIS,
                )
            if neural_output is None:
                raise ProfileError("real MaleCNS adapter did not provide step_window")

            with profiler.measure(
                "neuromuscular", sim_s=FRAME_DT_S, count=1, batch_size=1,
                source=config.source, phase=phase,
                metadata={"units": "control_frames", "window_steps": 10},
            ):
                nm_output = runner.neuromuscular.step(
                    neural_output, dt_ms=FRAME_DT_S * 1000.0, timestamp=timestamp,
                )
            body_command = _profile_body_command(nm_output)
            with profiler.measure(
                "body", sim_s=FRAME_DT_S, count=1, batch_size=1,
                source=config.source, phase=phase,
                metadata={"units": "body_steps", "integrator_calls": 1},
            ):
                dynamics_output = dynamics.step(body_command, dt_s=FRAME_DT_S)

            # Measure a bounded/hash-only version of the compact production
            # record.  Lossless per-frame array compression belongs to the
            # output-writer benchmark, not this short stage profile.
            compact = _profile_compact_record(
                frame_index, timestamp, flyvis_out, neural_output,
                body_command, dynamics_output, dynamics,
            )
            with profiler.measure(
                "serialization", sim_s=FRAME_DT_S, count=1, batch_size=1,
                source=config.source, phase=phase,
                metadata={"units": "records", "format": "json", "compact": True,
                          "array_encoding": "hash-only"},
            ):
                json.dumps(compact, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

            memory_after = _runtime_memory_snapshot(
                cuda_device=config.vision_device,
                ram_limit_gib=config.ram_limit_gib,
                vram_limit_gib=config.max_vram_gib,
            )
            facts["resource_observations"].append({
                "source": f"{config.source}:frame:{frame_index}",
                "phase": phase,
                "before": memory_before,
                "after": memory_after,
                # Flatten the after snapshot for ProfileReport.memory_summary.
                "ram_limit_bytes": memory_after.get("ram_limit_bytes"),
                "vram_limit_gib": memory_after.get("vram_limit_gib"),
                "peak_host_rss_bytes": memory_after.get("peak_host_rss_bytes"),
                "peak_vram_allocated_bytes": memory_after.get("peak_vram_allocated_bytes"),
                "peak_vram_reserved_bytes": memory_after.get("peak_vram_reserved_bytes"),
                "vram_total_bytes": (memory_after.get("cuda") or {}).get("total_bytes")
                if isinstance(memory_after.get("cuda"), Mapping) else None,
            })
            completed_frames += 1
            # Preserve the local name as an explicit audit point.  It also
            # makes progress accounting available while debugging a bounded
            # profile without changing StageProfiler's storage policy.
            stage_samples = list(profiler.samples())
    except ProfileError as exc:
        failed_frame = active_frame_index if active_frame_index is not None else completed_frames
        raise ProfileError(
            f"real GPU profile failed at frame {failed_frame} "
            f"(completed {completed_frames}/{config.frames}): {exc}"
        ) from exc
    except Exception as exc:
        failed_frame = active_frame_index if active_frame_index is not None else completed_frames
        raise ProfileError(
            f"real GPU profile failed at frame {failed_frame} "
            f"(completed {completed_frames}/{config.frames}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    stage_samples = list(profiler.samples())
    report = ProfileReport(
        samples=stage_samples,
        facts=facts,
        warnings=[],
    )
    if facts.get("resource_observations"):
        first_after = facts["resource_observations"][0].get("after", {})
        if isinstance(first_after, Mapping):
            facts["resource_envelope"] = {
                "ram_limit_gib": first_after.get("ram_limit_gib"),
                "ram_limit_source": first_after.get("ram_limit_source"),
                "ram_limit_enforced": first_after.get("ram_limit_enforced"),
                "vram_limit_gib": first_after.get("vram_limit_gib"),
                "vram_limit_source": "profile_config",
                "policy": "5 GiB host cgroup / 15.5 GiB CUDA allocator",
            }
    # Add this after constructing the report so callers can audit the exact
    # boundary synchronization count without parsing StageProfiler internals.
    report.facts["cuda_synchronization_calls"] = profiler.synchronize_calls
    report.facts["cuda_synchronization_expected"] = (
        2 * len(report.samples) if synchronize is not None else 0
    )
    report.facts["stage_profiler"] = {
        "dropped_samples": profiler.dropped_samples,
        "synchronize_calls": profiler.synchronize_calls,
        "synchronization_scope": "measurement_boundaries_only",
    }
    renderer_stage_names = ("rectangular_stereo_render", "exact_column_render")
    cold_samples = [
        sample for sample in report.samples
        if sample.stage in renderer_stage_names
        and bool(sample.metadata.get("cold_jit_candidate", False))
    ]
    renderer_stage_stats = report.stage_stats()
    steady_renderer: dict[str, Any] = {}
    warmup_renderer: dict[str, Any] = {}
    for stage_name in renderer_stage_names:
        stats = renderer_stage_stats.get(stage_name)
        if stats is None:
            continue
        steady_renderer[stage_name] = stats.steady.to_dict()
        warmup_renderer[stage_name] = stats.warmup.to_dict()
    cold_total = sum(sample.wall_s for sample in cold_samples)
    renderer_timing = {
        "requested_backend": config.renderer_backend,
        "effective_backend": effective_renderer,
        "cold_jit": {
            "included": config.renderer_backend == "warp_cuda",
            "phase": "warmup",
            "sample_count": len(cold_samples),
            "wall_s": cold_total if config.renderer_backend == "warp_cuda" else None,
            "synchronize_calls": (
                2 * len(cold_samples)
                if synchronize is not None and config.renderer_backend == "warp_cuda" else 0
            ),
            "stages": {
                sample.stage: sample.to_dict()
                for sample in cold_samples
            },
            "note": (
                "first rectangular and exact-column renderer calls; JIT cost is "
                "not merged into steady-state timings"
                if config.renderer_backend == "warp_cuda" else "not_applicable"
            ),
        },
        "warmup_renderer_stages": warmup_renderer,
        "steady_state_renderer_stages": steady_renderer,
        "steady_state_synchronization_calls": (
            2 * sum(len(renderer_stage_stats[name].steady.samples)
                    for name in renderer_stage_names if name in renderer_stage_stats)
            if synchronize is not None else 0
        ),
    }
    report.facts["renderer_timing"] = renderer_timing
    report.facts["renderer"]["timing"] = renderer_timing
    return report


# Readable aliases for callers and notebooks using either terminology.
profile_gpu_pipeline = run_gpu_profile
profile_pipeline = run_gpu_profile


_LEGACY_STAGES = {"legacy_flyvis_closed_loop", "legacy_closed_loop"}
_NEURAL_STAGES = {"malecns_open_loop", "malecns", "neural"}
_DIRECT_STAGES = {"end_to_end", "pipeline", "closed_loop"}
_EXCLUDED_SUM_STAGES = (
    _LEGACY_STAGES | _DIRECT_STAGES | NON_ADDITIVE_PIPELINE_STAGES | {"malecns_open_loop"}
)


def _projection_estimates(
    stats: Mapping[str, StageStats], target: float, frames: int,
    *, phase: Optional[str] = None,
) -> dict[str, StageEstimate]:
    estimates: dict[str, StageEstimate] = {}
    for name, rows in stats.items():
        if phase is not None:
            rows = rows.phase(phase)
            if not rows.samples:
                continue
        use_sim_time = name in _NEURAL_STAGES or (
            name not in _LEGACY_STAGES and any(s.sim_s is not None for s in rows.samples)
        )
        estimates[name] = fit_stage(
            rows,
            target_sim_s=target if use_sim_time else None,
            target_count=None if use_sim_time else frames,
        )
    return estimates


def _projection_choice(
    estimates: Mapping[str, StageEstimate], *, steady: bool = False,
) -> tuple[
    Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str, Optional[str]
]:
    direct = next((e.projected_s for n, e in estimates.items()
                   if n in _DIRECT_STAGES and e.projected_s is not None), None)
    neural_estimate = next((estimates[n] for n in ("malecns_open_loop", "malecns", "neural")
                            if n in estimates), None)
    legacy_estimate = next((estimates[n] for n in ("legacy_flyvis_closed_loop", "legacy_closed_loop")
                            if n in estimates), None)
    neural = neural_estimate.projected_s if neural_estimate is not None else None
    legacy = legacy_estimate.projected_s if legacy_estimate is not None else None
    parts = [e.projected_s for n, e in estimates.items()
             if n not in _EXCLUDED_SUM_STAGES and e.projected_s is not None]
    stage_sum = sum(parts) if parts else None
    if direct is not None:
        recommended, status = direct, "measured-e2e-fit"
        caveat = "explicit end_to_end/pipeline hook samples"
    elif stage_sum is not None:
        recommended, status = stage_sum, "stage-sum-fit"
        caveat = (
            "sum of exact-column pipeline stages; rectangular stereo and the "
            "diagnostic raycast pass are reported separately and excluded"
        )
    elif legacy is not None or (steady and legacy_estimate is not None):
        recommended = legacy
        status = "legacy-proxy" if steady else "legacy-proxy-plus-neural-lower-bound"
        caveat = "legacy FlyVis flight proxy; current e2e wall time is not directly measured"
    elif neural is not None or (steady and neural_estimate is not None):
        recommended = neural
        status = "neural-lower-bound" if steady else "neural-lower-bound-only"
        caveat = "open-loop MaleCNS only; world, FlyVis, readout, and startup are missing"
    else:
        recommended, status = None, "unmeasured"
        caveat = "add StageProfiler hooks to the real closed-loop process"
    return direct, stage_sum, neural, legacy, recommended, status, caveat if not steady else None


def _throughput_rates(
    projected: Optional[float], target: float, frames: int, cns_steps: int, fly_hz: float,
) -> dict[str, Any]:
    valid = projected is not None and projected > 0
    return {
        "real_time_factor": target / projected if valid else None,
        "flyvis_frames_per_s": frames / projected if valid else None,
        "malecns_steps_per_s": cns_steps / projected if valid else None,
        "frame_budget_s": 1.0 / fly_hz,
        "meets_realtime": bool(projected is not None and projected <= target),
    }


@dataclass(slots=True)
class ProfileReport:
    samples: list[StageSample] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def stage_stats(self) -> dict[str, StageStats]:
        return _group_stage_stats(self.samples)

    def memory_summary(self) -> dict[str, Any]:
        observations = self.facts.get("resource_observations", [])
        if not isinstance(observations, list):
            observations = []
        def maximum(key: str) -> Optional[int | float]:
            values: list[int | float] = []
            for item in observations:
                if isinstance(item, Mapping) and item.get(key) is not None:
                    try:
                        if key.endswith("_bytes"):
                            values.append(int(item[key]))
                        else:
                            values.append(float(item[key]))
                    except (TypeError, ValueError):
                        pass
            return max(values) if values else None
        ram_limit = maximum("ram_limit_bytes")
        vram_limit_gib = maximum("vram_limit_gib")
        peak_rss = maximum("peak_host_rss_bytes")
        peak_reserved = maximum("peak_vram_reserved_bytes")
        peak_allocated = maximum("peak_vram_allocated_bytes")
        budget_checks: list[bool] = []
        if ram_limit is not None and peak_rss is not None:
            budget_checks.append(peak_rss <= ram_limit)
        if vram_limit_gib is not None and peak_reserved is not None:
            budget_checks.append(peak_reserved / GIB <= vram_limit_gib)
        return {
            "ram_limit_bytes": ram_limit,
            "ram_limit_gib": ram_limit / GIB if ram_limit is not None else None,
            "vram_limit_gib": vram_limit_gib,
            "peak_host_rss_bytes": peak_rss,
            "peak_host_rss_gib": peak_rss / GIB if peak_rss is not None else None,
            "peak_vram_reserved_bytes": peak_reserved,
            "peak_vram_reserved_gib": peak_reserved / GIB if peak_reserved is not None else None,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_allocated_gib": peak_allocated / GIB if peak_allocated is not None else None,
            "ram_headroom_bytes": ram_limit - peak_rss if ram_limit is not None and peak_rss is not None else None,
            "vram_headroom_gib": (
                vram_limit_gib - peak_reserved / GIB
                if vram_limit_gib is not None and peak_reserved is not None else None
            ),
            "within_budget": all(budget_checks) if budget_checks else None,
            "observations": observations,
        }

    def projection(
        self,
        duration_s: float,
        *,
        flyvis_hz: float = FLYVIS_HZ,
        cns_hz: float = CNS_HZ,
    ) -> dict[str, Any]:
        target = _finite(duration_s, "duration_s")
        assert target is not None
        if target < 0:
            raise ProfileError("duration_s must be >= 0")
        fly_hz = _finite(flyvis_hz, "flyvis_hz")
        cns_hz_value = _finite(cns_hz, "cns_hz")
        assert fly_hz is not None and cns_hz_value is not None
        if fly_hz <= 0 or cns_hz_value <= 0:
            raise ProfileError("flyvis_hz and cns_hz must be positive")
        frames = int(math.ceil(target * fly_hz - 1.0e-12)) if target > 0 else 0
        steps_per_frame = int(round(cns_hz_value / fly_hz))
        if steps_per_frame < 1 or abs(steps_per_frame * fly_hz - cns_hz_value) > 1.0e-6:
            raise ProfileError("cns_hz/flyvis_hz must resolve to an integer CNS-step hold")
        cns_steps = frames * steps_per_frame
        stats = self.stage_stats()
        estimates = _projection_estimates(stats, target, frames)
        direct_s, explicit_sum, neural_s, legacy_s, recommended, status, caveat = (
            _projection_choice(estimates)
        )

        # A profile that contains a cold invocation must expose the
        # steady-state projection separately.  The historical recommendation
        # above intentionally includes startup/intercept cost; this second
        # view is what an operator should use when estimating a long run.
        steady_estimates = _projection_estimates(stats, target, frames, phase="steady")
        _, steady_sum, _, _, steady_recommended, steady_status, _ = _projection_choice(
            steady_estimates, steady=True,
        )
        stage_payload: dict[str, Any] = {}
        for name, estimate in estimates.items():
            payload = estimate.to_dict()
            if estimate.projected_s is not None and estimate.projected_s > 0:
                payload["throughput_units_per_s"] = (
                    estimate.target_units / estimate.projected_s
                    if estimate.target_units is not None else None
                )
                payload["real_time_factor"] = target / estimate.projected_s
            else:
                payload["throughput_units_per_s"] = None
                payload["real_time_factor"] = None
            rows = stats.get(name)
            if rows is not None:
                payload["phase_samples"] = {
                    phase: rows.phase(phase).to_dict()
                    for phase in PROFILE_PHASES
                    if rows.phase(phase).samples
                }
                # Steady-state throughput is the number used for capacity
                # planning.  Keep warmup visible but do not blend startup/JIT
                # cost into the steady rate.
                steady = rows.steady
                payload["steady_throughput_per_s"] = steady.wall_throughput_per_s
                payload["steady_real_time_factor"] = steady.real_time_factor
            stage_payload[name] = payload
        overall_payload = {
            "simulated_s": target,
            "projected_wall_s": recommended,
            **_throughput_rates(recommended, target, frames, cns_steps, fly_hz),
        }
        steady_payload = {
            "stage_estimates": {
                name: estimate.to_dict() for name, estimate in steady_estimates.items()
            },
            "projected_s": steady_recommended,
            "explicit_stage_sum_projected_s": steady_sum,
            "status": steady_status,
            **_throughput_rates(steady_recommended, target, frames, cns_steps, fly_hz),
            "warmup_wall_s": sum(item.warmup.total_wall_s for item in stats.values()),
            "steady_samples": sum(len(item.steady.samples) for item in stats.values()),
            "meets_realtime": bool(
                steady_recommended is not None and steady_recommended <= target
            ),
        }
        return {
            "target_simulated_s": target,
            "flyvis_hz": fly_hz,
            "cns_hz": cns_hz_value,
            "flyvis_frames": frames,
            "cns_steps_per_frame": steps_per_frame,
            "malecns_steps": cns_steps,
            "frame_budget_s": 1.0 / fly_hz,
            "stage_estimates": stage_payload,
            "direct_e2e_projected_s": direct_s,
            "explicit_stage_sum_projected_s": explicit_sum,
            "neural_lower_bound_projected_s": neural_s,
            "legacy_proxy_projected_s": legacy_s,
            "recommended_projected_s": recommended,
            "status": status,
            "caveat": caveat,
            "throughput": overall_payload,
            "steady_state": steady_payload,
            "non_additive_stage_names": sorted(NON_ADDITIVE_PIPELINE_STAGES),
        }

    def to_dict(self, *, duration_s: float = 1.0) -> dict[str, Any]:
        renderer_facts = self.facts.get("renderer", {})
        renderer_timing = self.facts.get("renderer_timing")
        if not isinstance(renderer_timing, Mapping) and isinstance(renderer_facts, Mapping):
            renderer_timing = renderer_facts.get("timing")
        return {
            "schema_version": SCHEMA_VERSION,
            "module": "scripts/profile_flight_pipeline.py",
            "runtime_imports": (
                "lazy runtime imports; bounded GPU workload launched"
                if "profile_config" in self.facts
                else "stdlib-only; no GPU workload launched"
            ),
            "source_count": len(self.facts.get("sources", [])),
            "sample_count": len(self.samples),
            "sample_rows": [sample.to_dict() for sample in self.samples],
            "warnings": list(self.warnings),
            "facts": _json_safe(self.facts),
            "renderer": _json_safe(self.facts.get("renderer", {})),
            "renderer_backend": {
                "requested": self.facts.get("renderer", {}).get("requested_backend")
                if isinstance(self.facts.get("renderer"), Mapping) else None,
                "effective": self.facts.get("renderer", {}).get("effective_backend")
                if isinstance(self.facts.get("renderer"), Mapping) else None,
            },
            "warp_cold_jit": (
                renderer_timing.get("cold_jit")
                if isinstance(renderer_timing, Mapping) else None
            ),
            "renderer_steady_state": (
                renderer_timing.get("steady_state_renderer_stages")
                if isinstance(renderer_timing, Mapping) else None
            ),
            "pipeline_stage_order": list(self.facts.get("pipeline_stage_order", ())),
            "memory": self.memory_summary(),
            "stages": {name: stats.to_dict() for name, stats in self.stage_stats().items()},
            "projection": self.projection(duration_s),
            "recommended_smoke_ladder": list(recommended_smoke_ladder()),
        }


def _human_summary(report: ProfileReport, duration_s: float) -> str:
    projection = report.projection(duration_s)
    lines = [
        f"flight GPU profile: {len(report.samples)} samples, {len(report.stage_stats())} stages",
        f"target: {projection['target_simulated_s']:.3g}s sim = {projection['flyvis_frames']} FlyVis frames = {projection['malecns_steps']} CNS steps",
    ]
    for name, estimate in projection["stage_estimates"].items():
        projected = estimate.get("projected_s")
        text = "n/a" if projected is None else f"{projected:.3f}s"
        lines.append(f"  {name}: {text} ({estimate['confidence']}, {estimate['samples']} samples)")
    renderer = report.facts.get("renderer", {})
    if isinstance(renderer, Mapping):
        requested_renderer = renderer.get("requested_backend")
        effective_renderer = renderer.get("effective_backend")
        if requested_renderer is not None or effective_renderer is not None:
            lines.append(
                f"renderer: requested={requested_renderer}, effective={effective_renderer}"
            )
        timing = report.facts.get("renderer_timing", renderer.get("timing", {}))
        if isinstance(timing, Mapping):
            cold = timing.get("cold_jit", {})
            if isinstance(cold, Mapping) and cold.get("included"):
                cold_wall = cold.get("wall_s")
                lines.append(
                    "renderer cold-JIT: "
                    + ("n/a" if cold_wall is None else f"{float(cold_wall):.3f}s")
                )
    rec = projection.get("recommended_projected_s")
    lines.append(f"projection: {'n/a' if rec is None else f'{rec:.3f}s'} [{projection['status']}]")
    steady = projection.get("steady_state", {})
    steady_rec = steady.get("projected_s") if isinstance(steady, Mapping) else None
    if steady_rec is not None:
        steady_rtf = steady.get("real_time_factor") if isinstance(steady, Mapping) else None
        lines.append(
            f"steady-state projection: {steady_rec:.3f}s "
            f"(RTF {'n/a' if steady_rtf is None else f'{float(steady_rtf):.4g}'}x)"
        )
    throughput = projection.get("throughput", {})
    if isinstance(throughput, Mapping) and throughput.get("real_time_factor") is not None:
        lines.append(
            f"throughput: {float(throughput['flyvis_frames_per_s']):.3g} FlyVis frames/s, "
            f"{float(throughput['malecns_steps_per_s']):.3g} CNS steps/s, "
            f"RTF {float(throughput['real_time_factor']):.4g}x"
        )
    lines.append(f"caveat: {projection['caveat']}")
    memory = report.memory_summary()
    if memory.get("peak_host_rss_gib") is not None or memory.get("peak_vram_reserved_gib") is not None:
        lines.append(
            "memory: RSS " + (f"{memory['peak_host_rss_gib']:.2f} GiB" if memory.get("peak_host_rss_gib") is not None else "n/a")
            + ", VRAM reserved " + (f"{memory['peak_vram_reserved_gib']:.2f} GiB" if memory.get("peak_vram_reserved_gib") is not None else "n/a")
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-profile", action="store_true",
        help="run the bounded real FlyVis -> MaleCNS -> NM/body GPU profile",
    )
    parser.add_argument("--profile-frames", type=int, default=2,
                        help="recorded+warmup 5 ms frames for --gpu-profile (default: 2)")
    parser.add_argument("--profile-warmup-frames", type=int, default=1,
                        help="leading frames labelled warmup (default: 1)")
    parser.add_argument("--profile-width", type=int, default=32,
                        help="rectangular stereo width for --gpu-profile")
    parser.add_argument("--profile-height", type=int, default=24,
                        help="rectangular stereo height for --gpu-profile")
    parser.add_argument(
        "--profile-renderer-backend", default="analytic_reference",
        choices=("analytic_reference", "warp_cuda"),
        help="world renderer backend (default: analytic_reference; warp never falls back)",
    )
    parser.add_argument("--profile-vision-device", default="cuda:0",
                        help="FlyVis CUDA device for --gpu-profile")
    parser.add_argument("--profile-neural-backend", default="torch",
                        choices=("torch", "triton", "cpu", "torch_cpu", "auto"),
                        help="MaleCNS backend for --gpu-profile (default: torch)")
    parser.add_argument("--profile-max-vram-gib", type=float, default=15.5,
                        help="MaleCNS VRAM allocator ceiling")
    parser.add_argument("--profile-ram-limit-gib", type=float, default=5.0,
                        help="declared host-memory envelope")
    parser.add_argument("--profile-scenario", default="hover",
                        help="configured scenario label used for body initialization")
    parser.add_argument("--profile-csr-path", type=Path, default=None,
                        help="optional CSR cache path")
    parser.add_argument(
        "--artifact", action="append", dest="artifacts", metavar="PATH",
        help="artifact file/directory; repeatable (default: project artifacts/)",
    )
    parser.add_argument("--duration-s", type=float, default=1.0, help="simulated flight duration to project")
    parser.add_argument("--include-telemetry", action="store_true", help="scan telemetry.jsonl for explicit stage rows")
    parser.add_argument("--human", action="store_true", help="print a short human summary instead of JSON")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument("--out", type=Path, default=None, help="optional report path; no file is written by default")
    args = parser.parse_args(argv)
    try:
        if args.gpu_profile:
            config = GPUProfileConfig(
                frames=args.profile_frames,
                warmup_frames=args.profile_warmup_frames,
                width=args.profile_width,
                height=args.profile_height,
                renderer_backend=args.profile_renderer_backend,
                vision_device=args.profile_vision_device,
                neural_backend=args.profile_neural_backend,
                max_vram_gib=args.profile_max_vram_gib,
                ram_limit_gib=args.profile_ram_limit_gib,
                scenario=args.profile_scenario,
                csr_path=args.profile_csr_path,
            )
            report = run_gpu_profile(config)
            duration_s = config.simulated_s
        else:
            report = profile_artifacts(
                args.artifacts or [DEFAULT_ARTIFACT_ROOT],
                include_telemetry=args.include_telemetry,
            )
            duration_s = args.duration_s
        payload = (
            _human_summary(report, duration_s) if args.human else json.dumps(
                report.to_dict(duration_s=duration_s),
                ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True,
            )
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0
    except (ProfileError, OSError, ValueError) as exc:
        print(f"profile_flight_pipeline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "BatchProbe", "CacheProbe", "CacheStats", "CNS_HZ", "CNS_STEPS_PER_FLYVIS",
    "FLYVIS_HZ", "GPUProfileConfig", "NON_ADDITIVE_PIPELINE_STAGES",
    "PIPELINE_STAGES", "PROFILE_PHASES",
    "ProfileError", "ProfileReport", "SCHEMA_VERSION", "StageEstimate",
    "StageProfiler", "StageSample", "StageStats", "fit_stage", "iter_batches", "main",
    "profile_artifacts", "profile_gpu_pipeline", "profile_pipeline", "run_gpu_profile",
    "recommended_smoke_ladder",
]
