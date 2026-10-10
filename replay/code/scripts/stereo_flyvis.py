#!/usr/bin/env python3
"""Stereo, time-aligned FlyVis front end for the MaleCNS interface.

This module is intentionally separate from ``flyvis_eye.py``.  The latter is
the E20 single-network experiment; this file is the safer interface used by
new code and tests.  In particular, it keeps left and right eyes in separate
state, applies a spatial anti-aliasing filter before column sampling, removes
a static luminance baseline, and never clips signed T4/T5 responses.

The public path is::

    frame -> left/right column luminance -> signed T4/T5 drive

FlyVis is sampled at 200 Hz (5 ms).  ``expand_drive_to_cns`` repeats one
FlyVis sample over exactly ten 0.5 ms CNS ticks; it does not silently round a
timestamp or interpolate a half-frame.

When the optional FlyVis environment is absent, ``backend="auto"`` selects
``DeterministicStubBackend``.  Every result carries ``backend`` and
``is_stub`` metadata so that a stub run cannot be mistaken for a fitted
FlyVis run.  A small ``RecordedResponseBackend`` is also provided for replay
of a response trace without installing FlyVis.

No dependency beyond NumPy is required for the stub, sampler, diagnostics, or
recorded backend.  The optional real adapter is loaded lazily and is never
imported during normal testing.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import numpy as np


FLYVIS_HZ = 200.0
FLYVIS_DT = 1.0 / FLYVIS_HZ
CNS_HZ = 2000.0
CNS_DT = 1.0 / CNS_HZ
CNS_TICKS_PER_FLYVIS = 10
# Readable aliases used by adapters that call a FlyVis sample a frame.
FLYVIS_FRAME_HZ = FLYVIS_HZ
FLYVIS_FRAME_DT = FLYVIS_DT
CNS_TICK_HZ = CNS_HZ
CNS_TICK_DT = CNS_DT
CNS_TICKS_PER_FRAME = CNS_TICKS_PER_FLYVIS
DEFAULT_COLUMNS = 721


class ShapeError(ValueError):
    """Raised when a frame, column vector, or backend response has bad shape."""


def _finite_float_array(value: Any, name: str) -> np.ndarray:
    try:
        a = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ShapeError(f"{name} must be a rectangular numeric array") from exc
    if a.size == 0:
        raise ShapeError(f"{name} must not be empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def _check_columns(value: Any, n: int, name: str) -> np.ndarray:
    a = _finite_float_array(value, name)
    if a.shape != (n,):
        raise ShapeError(f"{name} must have shape ({n},), got {a.shape}")
    return a.copy()


def _gaussian_kernel(sigma: float) -> np.ndarray:
    if not np.isfinite(sigma):
        raise ValueError("blur sigma must be finite")
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()
    return k.astype(np.float32)


def _blur1d(x: np.ndarray, sigma: float, wrap: bool = True) -> np.ndarray:
    """Small NumPy-only Gaussian blur used for already-sampled columns."""
    k = _gaussian_kernel(sigma)
    if len(k) == 1:
        return x.astype(np.float32, copy=True)
    radius = len(k) // 2
    if wrap:
        # Indexing modulo the input length also handles a one-column eye and
        # kernels wider than the sampled array without special cases.
        padded = x[np.arange(-radius, x.size + radius) % x.size]
    else:
        padded = _reflect_pad_1d(x, radius)
    return np.convolve(padded, k, mode="valid").astype(np.float32)


def _reflect_indices(n: int, radius: int) -> np.ndarray:
    """Indices for reflect padding, including n=1 and wide kernels."""
    if n <= 0:
        raise ShapeError("cannot pad an empty spatial axis")
    if radius <= 0:
        return np.arange(n, dtype=np.int64)
    if n == 1:
        return np.zeros(n + 2 * radius, dtype=np.int64)
    period = 2 * n - 2
    idx = np.arange(-radius, n + radius, dtype=np.int64) % period
    return np.where(idx >= n, period - idx, idx).astype(np.int64)


def _reflect_pad_1d(x: np.ndarray, radius: int) -> np.ndarray:
    return x[_reflect_indices(x.size, radius)]


def _blur2d(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32, copy=True)
    k = _gaussian_kernel(sigma)
    if len(k) == 1:
        return x.astype(np.float32, copy=True)
    r = len(k) // 2
    # np.pad(mode="reflect") rejects a pad larger than a one/two-pixel
    # dimension.  Explicit indices retain the same boundary convention for
    # normal frames and make tiny deterministic test frames valid too.
    p = x[np.ix_(_reflect_indices(x.shape[0], r),
                 _reflect_indices(x.shape[1], r))]
    # Separable convolution.  Frames are small compared with a CNS run and
    # this avoids making SciPy a requirement for the deterministic backend.
    y = np.apply_along_axis(lambda row: np.convolve(row, k, mode="valid"), 1, p)
    y = np.apply_along_axis(lambda col: np.convolve(col, k, mode="valid"), 0, y)
    return y.astype(np.float32)


def _normalise_luminance(frame: Any, name: str) -> np.ndarray:
    try:
        source = np.asarray(frame)
        source_is_integer = np.issubdtype(source.dtype, np.integer)
    except (TypeError, ValueError):
        source_is_integer = False
    a = _finite_float_array(frame, name)
    if a.ndim not in (1, 2):
        raise ShapeError(f"{name} must be 1-D columns or a 2-D grayscale frame, got {a.shape}")
    # Explicitly accept [0, 255] camera frames while keeping already
    # normalised [0, 1] values exact.  Silently clipping a malformed frame
    # would turn a shape/range bug into a plausible but wrong optic stimulus.
    amin, amax = float(a.min()), float(a.max())
    if amin < 0.0:
        raise ValueError(f"{name} luminance must be non-negative")
    if amax <= 1.0:
        return a.astype(np.float32, copy=False)
    # Integer camera buffers are unambiguous.  For floating point input only
    # treat a clearly 8-bit-scale value as [0,255]; a small overshoot such as
    # 1.2 is rejected instead of being silently divided by 255.
    if amax <= 255.0 and (source_is_integer or amax >= 2.0):
        return (a / 255.0).astype(np.float32, copy=False)
    raise ValueError(f"{name} luminance must be in [0, 1] or [0, 255], got max {amax:g}")


def _bilinear(frame: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample ``frame`` at normalized ``(x, y)`` positions."""
    if frame.ndim != 2:
        raise ShapeError("bilinear sampling requires a 2-D frame")
    h, w = frame.shape
    x = np.clip(xy[:, 0], 0.0, 1.0) * max(w - 1, 0)
    y = np.clip(xy[:, 1], 0.0, 1.0) * max(h - 1, 0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    return ((1 - wx) * (1 - wy) * frame[y0, x0]
            + wx * (1 - wy) * frame[y0, x1]
            + (1 - wx) * wy * frame[y1, x0]
            + wx * wy * frame[y1, x1]).astype(np.float32)


def _default_positions(n: int, eye: str) -> np.ndarray:
    """Stable pseudo-hexagonal positions for a dependency-free default layout."""
    if n <= 0:
        raise ValueError("n_columns must be positive")
    # A rectangular sampling layout is deliberately conservative: the
    # anti-aliasing filter, rather than a guessed hexagonal geometry, owns the
    # spatial bandwidth.  Mirroring x gives the two eyes distinct geometry.
    side = int(np.ceil(np.sqrt(n)))
    ix = np.arange(n, dtype=np.int64) % side
    iy = np.arange(n, dtype=np.int64) // side
    x = (ix + 0.5) / side
    y = (iy + 0.5) / side
    if eye.upper() == "R":
        x = 1.0 - x
    return np.stack([x, y], axis=1).astype(np.float32)


def _validate_positions(pos: Any, n: int, name: str) -> np.ndarray:
    a = _finite_float_array(pos, name)
    if a.shape != (n, 2):
        raise ShapeError(f"{name} must have shape ({n}, 2), got {a.shape}")
    if np.any((a < 0.0) | (a > 1.0)):
        raise ValueError(f"{name} coordinates must be normalized to [0, 1]")
    return a.copy()


class ResponseBackend(Protocol):
    """Backend contract: both eye vectors are separate and response is signed."""

    name: str
    is_stub: bool
    n_t4t5: int

    def reset(self) -> None: ...

    def step(self, left_columns: np.ndarray, right_columns: np.ndarray) -> np.ndarray: ...


class DeterministicStubBackend:
    """Deterministic signed push-pull motion backend for tests and no-FlyVis hosts.

    It is not a biological model.  It approximates a signed Reichardt-like
    local detector using delayed neighbour products, then projects the
    two independent eyes to reproducible T4/T5 channels.  Crucially, it
    returns signed values; positive and negative preferred directions survive
    all the way to the CNS interface.
    """

    name = "deterministic_stub"
    is_stub = True
    is_recorded = False

    def __init__(self, n_columns: int, n_t4t5: int = 8, seed: int = 1729):
        if isinstance(n_t4t5, bool) or int(n_t4t5) != n_t4t5 or int(n_t4t5) <= 0:
            raise ValueError("n_t4t5 must be positive")
        self.n_columns = int(n_columns)
        self.n_t4t5 = int(n_t4t5)
        self.seed = int(seed)
        # Fixed, local phase functions make the result deterministic across
        # processes without relying on a global RNG state.
        phase = (np.arange(self.n_t4t5, dtype=np.float64) + 0.5) / self.n_t4t5
        self._phase = (2.0 * np.pi * phase).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        self._prev_left: Optional[np.ndarray] = None
        self._prev_right: Optional[np.ndarray] = None

    def _motion(self, cur: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        if prev is None:
            return np.zeros_like(cur, dtype=np.float32)
        # Two neighbouring delayed products form a signed Reichardt/EMD-like
        # detector.  The subtraction is essential: a plain temporal gradient
        # product has a second-spatial-harmonic cancellation for a travelling
        # sinusoid and can look spuriously silent.
        forward = cur * np.roll(prev, -1) - prev * np.roll(cur, -1)
        backward = cur * np.roll(prev, 1) - prev * np.roll(cur, 1)
        return (0.5 * (forward - backward)).astype(np.float32)

    def step(self, left_columns: np.ndarray, right_columns: np.ndarray) -> np.ndarray:
        left = _check_columns(left_columns, self.n_columns, "left_columns")
        right = _check_columns(right_columns, self.n_columns, "right_columns")
        ml = self._motion(left, self._prev_left)
        mr = self._motion(right, self._prev_right)
        self._prev_left, self._prev_right = left, right
        # Eye-wise detector summary, retained with its sign.  Opposite phase
        # channels form the push/pull pair instead of max(response, 0).
        spatial_phase = np.arange(self.n_columns, dtype=np.float32) * (2 * np.pi / self.n_columns)
        sl = float(np.mean(ml * np.sin(spatial_phase)))
        cl = float(np.mean(ml * np.cos(spatial_phase)))
        sr = float(np.mean(mr * np.sin(spatial_phase)))
        cr = float(np.mean(mr * np.cos(spatial_phase)))
        # Keep the DC directional component too.  It is the robust signal for
        # a coherent travelling pattern; the harmonic terms retain spatial
        # structure and make binocular asymmetry observable.
        dl, dr = float(np.mean(ml)), float(np.mean(mr))
        common_direction, differential_direction = dl + dr, dl - dr
        vals = np.empty(self.n_t4t5, dtype=np.float32)
        for i, p in enumerate(self._phase):
            # L/R contributions are intentionally not shared state.  The
            # alternating pair gives a useful signed direction axis; the eye
            # difference also keeps stereo asymmetry observable.
            vals[i] = (common_direction * np.cos(p) + differential_direction * np.sin(p)
                       + (sl + sr) * np.cos(p) + (cl + cr) * np.sin(p)
                       + 0.5 * (sl - sr) * np.sin(2 * p)
                       + 0.5 * (cl - cr) * np.cos(2 * p))
        return vals


class RecordedResponseBackend:
    """Replay signed T4/T5 responses from an ``npz`` file.

    Required field: ``responses`` with shape ``(frames, n_t4t5)``.  ``drive``
    and ``t4t5_drive`` are accepted as compatible field names.  Optional
    fields ``left_columns`` and ``right_columns`` are used only as metadata;
    frontend sampling still occurs and shape checks remain active.
    """

    is_stub = False
    is_recorded = True

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            field_name = next((name for name in ("responses", "t4t5_drive", "drive")
                               if name in data), None)
            if field_name is None:
                raise ValueError(f"{self.path} must contain a 'responses' array")
            self.responses = _finite_float_array(data[field_name], field_name)
        if self.responses.ndim != 2:
            raise ShapeError(f"responses must have shape (frames, n_t4t5), got {self.responses.shape}")
        if self.responses.shape[0] == 0 or self.responses.shape[1] == 0:
            raise ShapeError("recorded responses must contain at least one frame and one channel")
        self.name = f"recorded:{self.path.name}"
        self.n_t4t5 = int(self.responses.shape[1])
        self.n_columns: Optional[int] = None
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def step(self, left_columns: np.ndarray, right_columns: np.ndarray) -> np.ndarray:
        if self._index >= len(self.responses):
            raise IndexError(f"recorded response exhausted at frame {self._index}")
        out = self.responses[self._index].copy()
        self._index += 1
        return out


class OptionalFlyVisBackend:
    """Lazy real-FlyVis adapter hook.

    The project-specific FlyVis mapping is intentionally injected as a
    callable.  This prevents importing the optional environment during unit
    tests and prevents accidentally reusing the old shared stereo state.  A
    callable receives ``(left_columns, right_columns)`` and must return a
    signed ``(n_t4t5,)`` vector.  ``is_stub`` is always false.
    """

    name = "flyvis"
    is_stub = False
    is_recorded = False

    def __init__(self, response_fn: Any, n_columns: int, n_t4t5: int):
        if not callable(response_fn):
            raise TypeError("response_fn must be callable")
        if isinstance(n_t4t5, bool) or int(n_t4t5) != n_t4t5 or int(n_t4t5) <= 0:
            raise ValueError("n_t4t5 must be positive")
        self.response_fn = response_fn
        self.n_columns = int(n_columns)
        self.n_t4t5 = int(n_t4t5)

    def reset(self) -> None:
        reset = getattr(self.response_fn, "reset", None)
        if callable(reset):
            reset()

    def step(self, left_columns: np.ndarray, right_columns: np.ndarray) -> np.ndarray:
        out = _finite_float_array(self.response_fn(left_columns.copy(), right_columns.copy()), "FlyVis response")
        if out.shape != (self.n_t4t5,):
            raise ShapeError(f"FlyVis response must have shape ({self.n_t4t5},), got {out.shape}")
        return out.astype(np.float32, copy=True)


def _make_backend(backend: Any, n_columns: int, n_t4t5: int) -> tuple[ResponseBackend, str]:
    if backend is None or (isinstance(backend, str) and backend.lower() == "auto"):
        # No import attempt is needed: the real model must be explicitly
        # supplied because it has project-specific node mapping and weights.
        return DeterministicStubBackend(n_columns, n_t4t5), "FlyVis unavailable; deterministic stub selected"
    if isinstance(backend, str) and backend.lower() == "stub":
        return DeterministicStubBackend(n_columns, n_t4t5), "explicit deterministic stub"
    if isinstance(backend, (str, Path)):
        return RecordedResponseBackend(backend), "recorded-response replay"
    if not hasattr(backend, "step") or not hasattr(backend, "n_t4t5"):
        raise TypeError("backend must be 'auto', 'stub', an npz path, or a ResponseBackend")
    if (isinstance(backend.n_t4t5, bool) or int(backend.n_t4t5) != backend.n_t4t5
            or int(backend.n_t4t5) <= 0):
        raise ValueError("backend.n_t4t5 must be positive")
    return backend, "custom backend"


@dataclass(frozen=True)
class StereoDiagnostics:
    """Machine-readable diagnostics for one output or a stimulus comparison."""

    direction_correlation: float = float("nan")
    direction_common_mode_ratio: float = float("nan")
    direction_selective: bool = False
    left_right_correlation: float = float("nan")
    left_right_common_mode_ratio: float = float("nan")
    aliasing_score_left: float = float("nan")
    aliasing_score_right: float = float("nan")
    aliasing_warning: bool = False
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction_correlation": self.direction_correlation,
            "direction_common_mode_ratio": self.direction_common_mode_ratio,
            "direction_selective": self.direction_selective,
            "left_right_correlation": self.left_right_correlation,
            "left_right_common_mode_ratio": self.left_right_common_mode_ratio,
            "aliasing_score_left": self.aliasing_score_left,
            "aliasing_score_right": self.aliasing_score_right,
            "aliasing_warning": self.aliasing_warning,
            "warnings": list(self.warnings),
        }


@dataclass
class StereoOutput:
    """One 5 ms sample and its exact ten-tick CNS representation."""

    left_columns: np.ndarray
    right_columns: np.ndarray
    t4t5_drive: np.ndarray
    frame_index: int
    timestamp_s: float
    cns_tick_start: int
    cns_ticks: int
    backend: str
    is_stub: bool
    aliasing_warning: bool = False
    aliasing_score_left: float = float("nan")
    aliasing_score_right: float = float("nan")

    @property
    def drive(self) -> np.ndarray:
        """Alias used by CNS callers."""
        return self.t4t5_drive

    @property
    def left(self) -> np.ndarray:
        return self.left_columns

    @property
    def right(self) -> np.ndarray:
        return self.right_columns

    @property
    def t4t5(self) -> np.ndarray:
        return self.t4t5_drive

    @property
    def timestamp(self) -> float:
        return self.timestamp_s

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "is_stub": self.is_stub,
            "frame_index": self.frame_index, "timestamp_s": self.timestamp_s,
            "cns_tick_start": self.cns_tick_start, "cns_ticks": self.cns_ticks,
            "aliasing_warning": self.aliasing_warning,
            "aliasing_score_left": self.aliasing_score_left,
            "aliasing_score_right": self.aliasing_score_right,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form compatible with the shared contract records."""
        out = self.as_dict()
        out["left_columns"] = self.left_columns.tolist()
        out["right_columns"] = self.right_columns.tolist()
        out["t4t5_drive"] = self.t4t5_drive.tolist()
        return out

    @property
    def cns_tick_end(self) -> int:
        """Exclusive end tick for this 5 ms sample."""
        return self.cns_tick_start + self.cns_ticks

    @property
    def cns_tick_indices(self) -> np.ndarray:
        return np.arange(self.cns_tick_start, self.cns_tick_end, dtype=np.int64)

    @property
    def cns_timestamps_s(self) -> np.ndarray:
        return self.cns_tick_indices.astype(np.float64) * CNS_DT

    @property
    def is_recorded(self) -> bool:
        return self.backend.startswith("recorded:")

    def cns_timestamp(self) -> Any:
        """Return a ``twin_contracts.Timestamp`` without making it required.

        The import is lazy so the stereo front end remains usable as a
        NumPy-only component.  The timestamp is in the neural/CNS domain: a
        FlyVis frame start at 200 Hz is represented by its exact 2 kHz tick.
        """
        from twin_contracts import Timestamp
        return Timestamp(self.timestamp_s, self.cns_tick_start, "neural", CNS_HZ)

    @property
    def timestamp_contract(self) -> Any:
        return self.cns_timestamp()

    def to_visual_observation(self, *, pixels: Any = None, sensor_id: str = "compound_eye") -> Any:
        """Adapt this output to the repository's ``VisualObservation`` contract."""
        from twin_contracts import VisualObservation
        if pixels is None:
            pixels = np.stack([self.left_columns, self.right_columns], axis=0)
        return VisualObservation(
            timestamp=self.cns_timestamp(), pixels=pixels, sensor_id=sensor_id,
            metadata={"backend": self.backend, "is_stub": self.is_stub,
                      "frame_index": self.frame_index, "cns_ticks": self.cns_ticks,
                      "aliasing_warning": self.aliasing_warning,
                      "aliasing_score_left": self.aliasing_score_left,
                      "aliasing_score_right": self.aliasing_score_right})

    def to_sensory_frame(self, *, pixels: Any = None, sensor_id: str = "compound_eye") -> Any:
        """Adapt this output to a ``SensoryFrame`` with explicit CNS channels."""
        from twin_contracts import SensoryFrame
        return SensoryFrame(
            timestamp=self.cns_timestamp(),
            visual=self.to_visual_observation(pixels=pixels, sensor_id=sensor_id),
            channels={"t4t5_drive": self.t4t5_drive,
                      "left_columns": self.left_columns,
                      "right_columns": self.right_columns,
                      "backend": self.backend, "is_stub": self.is_stub})

    def __getitem__(self, key: str) -> Any:
        if key in ("left", "left_columns"):
            return self.left_columns
        if key in ("right", "right_columns"):
            return self.right_columns
        if key in ("drive", "t4t5_drive"):
            return self.t4t5_drive
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_columns": self.left_columns,
            "right_columns": self.right_columns,
            "t4t5_drive": self.t4t5_drive,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "cns_tick_start": self.cns_tick_start,
            "cns_ticks": self.cns_ticks,
            "backend": self.backend,
            "is_stub": self.is_stub,
            "aliasing_warning": self.aliasing_warning,
            "aliasing_score_left": self.aliasing_score_left,
            "aliasing_score_right": self.aliasing_score_right,
        }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


def direction_selectivity(positive: Any, negative: Any, eps: float = 1e-12) -> dict[str, Any]:
    """Compare + and - motion drives without rectifying either direction.

    ``d+`` and ``d-`` are expected to be stimulus-minus-static-baseline
    T4/T5 vectors.  Negative correlation and a small common/differential ratio
    are evidence for push-pull direction selectivity.
    """
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    p = _finite_float_array(positive, "positive")
    m = _finite_float_array(negative, "negative")
    if p.shape != m.shape:
        raise ShapeError(f"positive and negative must have the same shape, got {p.shape} and {m.shape}")
    p, m = p.reshape(-1), m.reshape(-1)
    common = 0.5 * (p + m)
    differential = 0.5 * (p - m)
    corr = _corr(p, m)
    ratio = _rms(common) / max(_rms(differential), eps)
    return {
        "direction_correlation": corr,
        "direction_common_mode_ratio": ratio,
        "direction_selective": bool(corr < -0.1 and ratio < 1.0),
    }


def left_right_common_mode(left: Any, right: Any, eps: float = 1e-12) -> dict[str, float]:
    """Quantify binocular common-mode versus inter-eye difference."""
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    l = _finite_float_array(left, "left")
    r = _finite_float_array(right, "right")
    if l.shape != r.shape:
        raise ShapeError(f"left and right must have the same shape, got {l.shape} and {r.shape}")
    l, r = l.reshape(-1), r.reshape(-1)
    common, differential = 0.5 * (l + r), 0.5 * (l - r)
    return {
        "left_right_correlation": _corr(l, r),
        "left_right_common_mode_ratio": _rms(common) / max(_rms(differential), eps),
    }


def aliasing_score(frame: Any, filtered: Optional[Any] = None) -> float:
    """Return high-frequency residual / contrast; larger means aliasing risk.

    ``filtered`` is the actual pre-sampling low-pass image when available.  A
    score therefore measures the detail removed before a column is sampled,
    rather than applying a second blur to the already-filtered image.
    """
    a = _normalise_luminance(frame, "frame")
    if filtered is not None:
        f = _normalise_luminance(filtered, "filtered")
        if f.shape != a.shape:
            raise ShapeError(f"filtered must have shape {a.shape}, got {f.shape}")
        residual = a - f
    else:
        smooth = _blur1d(a, 1.0) if a.ndim == 1 else _blur2d(a, 1.0)
        residual = a - smooth
    contrast = float(np.std(a))
    # A uniform frame should score exactly (or numerically) as no aliasing;
    # dividing by a 1e-6 floor would amplify float32 convolution residue into
    # a false warning.  1e-3 is still far below normal camera contrast.
    return _rms(residual) / max(contrast, 1e-3)


def aliasing_warning(score: Any, threshold: float = 0.35) -> bool:
    """Return whether an aliasing score exceeds a finite threshold."""
    value = float(score)
    limit = float(threshold)
    if not np.isfinite(value) or not np.isfinite(limit):
        raise ValueError("aliasing score and threshold must be finite")
    if limit < 0:
        raise ValueError("aliasing threshold must be non-negative")
    return bool(value > limit)


def _unwrap_frame(value: Any) -> tuple[Any, Optional[float]]:
    """Accept raw pixels or the shared VisualObservation/SensoryFrame shape."""
    visual = getattr(value, "visual", None)
    if visual is not None and hasattr(visual, "pixels"):
        value = visual
    if hasattr(value, "pixels") and hasattr(value, "timestamp"):
        pixels = getattr(value, "pixels")
        if pixels is None:
            raise ShapeError("visual observation has no pixels")
        ts = getattr(value.timestamp, "t_s", getattr(value.timestamp, "time_s", None))
        if ts is None:
            raise ValueError("visual observation timestamp has no t_s/time_s")
        return pixels, float(ts)
    return value, None


class StereoFlyVis:
    """Frame-to-stereo-columns-to-signed-T4/T5 interface."""

    _TIME_TOL_S = 1e-9

    def __init__(self, n_columns: int = DEFAULT_COLUMNS, n_t4t5: int = 8,
                 *, left_positions: Any = None, right_positions: Any = None,
                 blur_sigma_px: float = 1.0, baseline: Any = None,
                 baseline_alpha: float = 0.0, aliasing_threshold: float = 0.35,
                 backend: Any = "auto", start_time_s: float = 0.0):
        if isinstance(n_columns, bool) or int(n_columns) != n_columns or int(n_columns) <= 0:
            raise ValueError("n_columns must be positive")
        if not np.isfinite(blur_sigma_px) or blur_sigma_px < 0:
            raise ValueError("blur_sigma_px must be non-negative")
        if not np.isfinite(baseline_alpha) or not 0.0 <= baseline_alpha <= 1.0:
            raise ValueError("baseline_alpha must be in [0, 1]")
        start_time_s = float(start_time_s)
        if not np.isfinite(start_time_s) or start_time_s < 0:
            raise ValueError("start_time_s must be finite and non-negative")
        start_tick = int(round(start_time_s / CNS_DT))
        if abs(start_time_s - start_tick * CNS_DT) > self._TIME_TOL_S:
            raise ValueError(f"start_time_s must align to the CNS {CNS_DT:g}s tick")
        aliasing_threshold = float(aliasing_threshold)
        if not np.isfinite(aliasing_threshold) or aliasing_threshold < 0:
            raise ValueError("aliasing_threshold must be finite and non-negative")
        self.n_columns = int(n_columns)
        self.blur_sigma_px = float(blur_sigma_px)
        self.baseline_alpha = float(baseline_alpha)
        self.aliasing_threshold = aliasing_threshold
        self.left_positions = (_default_positions(self.n_columns, "L") if left_positions is None
                               else _validate_positions(left_positions, self.n_columns, "left_positions"))
        self.right_positions = (_default_positions(self.n_columns, "R") if right_positions is None
                                else _validate_positions(right_positions, self.n_columns, "right_positions"))
        self.backend, self.backend_note = _make_backend(backend, self.n_columns, n_t4t5)
        # A recorded/custom backend can determine the response width.
        self.n_t4t5 = int(self.backend.n_t4t5)
        self._baseline_left: Optional[np.ndarray] = None
        self._baseline_right: Optional[np.ndarray] = None
        self._configured_baseline_left: Optional[np.ndarray] = None
        self._configured_baseline_right: Optional[np.ndarray] = None
        if baseline is not None:
            bl, br = self._split_baseline(baseline)
            self._configured_baseline_left = self._coerce_baseline(bl, self.left_positions, "baseline.left")
            self._configured_baseline_right = self._coerce_baseline(br, self.right_positions, "baseline.right")
            self._baseline_left = self._configured_baseline_left.copy()
            self._baseline_right = self._configured_baseline_right.copy()
        self._frame_index = 0
        self._start_time_s = start_time_s
        self._start_cns_tick = start_tick
        self._last_timestamp_s: Optional[float] = None

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend, "name", self.backend.__class__.__name__))

    @property
    def backend_is_stub(self) -> bool:
        return bool(getattr(self.backend, "is_stub", False))

    @property
    def is_stub(self) -> bool:
        """Component-level provenance flag used by the shared runtime."""
        return self.backend_is_stub

    @property
    def backend_is_recorded(self) -> bool:
        return bool(getattr(self.backend, "is_recorded", False))

    @property
    def is_recorded(self) -> bool:
        return self.backend_is_recorded

    @property
    def n_t4_t5(self) -> int:
        """Spelling alias used by callers that separate the cell families."""
        return self.n_t4t5

    @staticmethod
    def _split_baseline(baseline: Any) -> tuple[Any, Any]:
        """Return independent eye baselines without mistaking a 2-column vector for a pair."""
        if isinstance(baseline, Mapping):
            if "left" not in baseline or "right" not in baseline:
                raise ShapeError("baseline mapping must contain both 'left' and 'right'")
            return baseline["left"], baseline["right"]
        try:
            a = np.asarray(baseline)
        except (TypeError, ValueError):
            a = None
        if a is not None and a.ndim == 2 and a.shape[0] == 2:
            return a[0], a[1]
        if isinstance(baseline, (tuple, list)) and len(baseline) == 2:
            # A pair of one-dimensional arrays is an explicit binocular
            # baseline.  A plain [v0, v1] remains one shared 2-column vector.
            try:
                first, second = np.asarray(baseline[0]), np.asarray(baseline[1])
            except (TypeError, ValueError):
                first = second = np.asarray(())
            if first.ndim == 1 and second.ndim == 1:
                return baseline[0], baseline[1]
        return baseline, baseline

    def _coerce_baseline(self, value: Any, positions: np.ndarray, name: str) -> np.ndarray:
        """Accept a sampled column baseline or a static grayscale baseline frame."""
        a = _normalise_luminance(value, name)
        if a.ndim == 1:
            return _check_columns(a, self.n_columns, name)
        if a.ndim == 2:
            columns, _ = self._sample_eye(a, positions, name)
            return columns
        raise ShapeError(f"{name} must have shape ({self.n_columns},) or a 2-D frame, got {a.shape}")

    @property
    def backend_metadata(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "is_stub": self.backend_is_stub,
                "is_recorded": self.backend_is_recorded,
                "backend_note": self.backend_note, "flyvis_hz": FLYVIS_HZ,
                "flyvis_dt_s": FLYVIS_DT, "cns_dt_s": CNS_DT,
                "cns_ticks_per_flyvis": CNS_TICKS_PER_FLYVIS,
                "start_time_s": self._start_time_s,
                "start_cns_tick": self._start_cns_tick}

    def reset(self) -> None:
        self._baseline_left = (None if self._configured_baseline_left is None
                               else self._configured_baseline_left.copy())
        self._baseline_right = (None if self._configured_baseline_right is None
                                else self._configured_baseline_right.copy())
        self._frame_index = 0
        self._last_timestamp_s = None
        reset = getattr(self.backend, "reset", None)
        if callable(reset):
            reset()

    def set_baseline(self, left: Any, right: Any = None) -> None:
        """Set independent static luminance baselines.

        Each value may be a sampled ``(n_columns,)`` vector or a grayscale
        frame.  Omitting ``right`` deliberately shares the left baseline.
        """
        if right is None:
            right = left
        self._configured_baseline_left = self._coerce_baseline(left, self.left_positions, "baseline.left")
        self._configured_baseline_right = self._coerce_baseline(right, self.right_positions, "baseline.right")
        self._baseline_left = self._configured_baseline_left.copy()
        self._baseline_right = self._configured_baseline_right.copy()

    def _sample_eye(self, frame: Any, positions: np.ndarray, name: str) -> tuple[np.ndarray, float]:
        a = _normalise_luminance(frame, name)
        if a.ndim == 1:
            if a.shape != (self.n_columns,):
                raise ShapeError(f"{name} 1-D input must have shape ({self.n_columns},), got {a.shape}")
            filtered = _blur1d(a, self.blur_sigma_px)
            smooth = filtered
            score_filtered = filtered if self.blur_sigma_px > 0 else _blur1d(a, 1.0)
        else:
            filtered = _blur2d(a, self.blur_sigma_px)
            smooth = _bilinear(filtered, positions)
            score_filtered = filtered if self.blur_sigma_px > 0 else _blur2d(a, 1.0)
        score = aliasing_score(a, filtered=score_filtered)
        return smooth.astype(np.float32), float(score)

    def frame_to_columns(self, frame: Any, right_frame: Any = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Render one frame into independent left/right column luminance."""
        if right_frame is None:
            right_frame = frame
        frame, _ = _unwrap_frame(frame)
        right_frame, _ = _unwrap_frame(right_frame)
        left, al = self._sample_eye(frame, self.left_positions, "left_frame")
        right, ar = self._sample_eye(right_frame, self.right_positions, "right_frame")
        warned = aliasing_warning(max(al, ar), self.aliasing_threshold)
        return left, right, {"aliasing_score_left": al, "aliasing_score_right": ar,
                             "aliasing_warning": warned}

    # Naming aliases keep the public path explicit at call sites.
    frame_to_stereo_columns = frame_to_columns
    sample_stereo = frame_to_columns

    def process_frame(self, frame: Any, right_frame: Any = None,
                      *, timestamp_s: Optional[float] = None) -> StereoOutput:
        """Process one 5 ms FlyVis sample and return a signed T4/T5 drive."""
        frame, frame_timestamp = _unwrap_frame(frame)
        right_timestamp: Optional[float] = None
        if right_frame is not None:
            right_frame, right_timestamp = _unwrap_frame(right_frame)
        if (timestamp_s is not None and frame_timestamp is not None
                and abs(float(timestamp_s) - frame_timestamp) > self._TIME_TOL_S):
            raise ValueError("timestamp_s does not match the visual observation timestamp")
        if timestamp_s is None and frame_timestamp is not None:
            timestamp_s = frame_timestamp
        if (frame_timestamp is not None and right_timestamp is not None
                and abs(frame_timestamp - right_timestamp) > self._TIME_TOL_S):
            raise ValueError("left and right visual timestamps must match")
        frame_index = self._frame_index
        expected_timestamp = self._start_time_s + frame_index * FLYVIS_DT
        if timestamp_s is None:
            timestamp_s = expected_timestamp
        else:
            timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if abs(timestamp_s - expected_timestamp) > self._TIME_TOL_S:
            raise ValueError(
                f"timestamp must equal frame {frame_index} at {expected_timestamp:g}s; got {timestamp_s:g}s"
            )
        tick_float = timestamp_s / CNS_DT
        tick = int(round(tick_float))
        if abs(tick_float - tick) > self._TIME_TOL_S / CNS_DT:
            raise ValueError(f"timestamp {timestamp_s:g}s is not aligned to the CNS {CNS_DT:g}s tick")
        if self._last_timestamp_s is not None:
            delta = timestamp_s - self._last_timestamp_s
            if abs(delta - FLYVIS_DT) > self._TIME_TOL_S:
                raise ValueError(f"FlyVis timestamps must advance by {FLYVIS_DT:g}s; got {delta:g}s")

        left, right, alias = self.frame_to_columns(frame, right_frame)
        baseline_was_none = self._baseline_left is None
        if baseline_was_none:
            # First sample establishes the static baseline.  This makes a
            # static grey screen exactly silent rather than a startup burst.
            baseline_left, baseline_right = left, right
        else:
            baseline_left, baseline_right = self._baseline_left, self._baseline_right
        dleft = left - baseline_left
        dright = right - baseline_right
        response = _finite_float_array(self.backend.step(dleft.copy(), dright.copy()), "t4t5_drive")
        if response.shape != (self.n_t4t5,):
            raise ShapeError(f"backend drive must have shape ({self.n_t4t5},), got {response.shape}")
        if baseline_was_none:
            self._baseline_left, self._baseline_right = left.copy(), right.copy()
        # Optional slow adaptation is explicit; default 0.0 means a genuinely
        # static baseline, not a moving high-pass that silently erases a scene.
        if self.baseline_alpha:
            a = self.baseline_alpha
            self._baseline_left = (1 - a) * self._baseline_left + a * left
            self._baseline_right = (1 - a) * self._baseline_right + a * right
        out = StereoOutput(left_columns=left, right_columns=right,
                           t4t5_drive=response.copy(), frame_index=frame_index,
                           timestamp_s=timestamp_s, cns_tick_start=tick,
                           cns_ticks=CNS_TICKS_PER_FLYVIS, backend=self.backend_name,
                           is_stub=self.backend_is_stub, **alias)
        self._frame_index += 1
        self._last_timestamp_s = timestamp_s
        return out

    # Explicitly named alias for callers documenting the full data path.
    frame_to_t4t5 = process_frame
    process = process_frame
    step = process_frame

    @staticmethod
    def expand_drive_to_cns(drive: Any) -> np.ndarray:
        """Repeat one signed FlyVis drive over ten 0.5 ms CNS ticks."""
        if isinstance(drive, StereoOutput):
            a = drive.t4t5_drive
        else:
            a = _finite_float_array(drive, "drive")
            if a.ndim != 1:
                raise ShapeError(f"drive must have shape (n_t4t5,), got {a.shape}")
        return np.repeat(a[None, :], CNS_TICKS_PER_FLYVIS, axis=0).astype(np.float32)

    to_cns = expand_drive_to_cns

    def process_frames(self, frames: Any, right_frames: Any = None,
                       *, start_time_s: Optional[float] = None) -> list[StereoOutput]:
        """Process a sequence while preserving one backend state per eye."""
        a = _finite_float_array(frames, "frames")
        if a.ndim not in (2, 3):
            raise ShapeError("frames must be (T, H, W) or (T, n_columns)")
        if right_frames is None:
            b = None
        else:
            b = _finite_float_array(right_frames, "right_frames")
            if b.shape[0] != a.shape[0]:
                raise ShapeError("frames and right_frames must have the same number of frames")
            if b.ndim not in (2, 3):
                raise ShapeError("right_frames must be (T, H, W) or (T, n_columns)")
            if b.ndim != a.ndim:
                raise ShapeError("frames and right_frames must have matching rank")
        if start_time_s is not None and self._frame_index:
            raise ValueError("start_time_s may only be supplied before processing starts")
        if start_time_s is not None:
            start_time_s = float(start_time_s)
            if not np.isfinite(start_time_s) or start_time_s < 0:
                raise ValueError("start_time_s must be finite and non-negative")
            start_tick = int(round(start_time_s / CNS_DT))
            if abs(start_time_s - start_tick * CNS_DT) > self._TIME_TOL_S:
                raise ValueError(f"start_time_s must align to the CNS {CNS_DT:g}s tick")
            self._start_time_s = start_time_s
            self._start_cns_tick = start_tick
        result = []
        for i in range(a.shape[0]):
            result.append(self.process_frame(a[i], None if b is None else b[i]))
        return result

    def diagnostics(self, *, positive: Any = None, negative: Any = None,
                    left_drive: Any = None, right_drive: Any = None,
                    aliasing_scores: Optional[tuple[float, float]] = None) -> StereoDiagnostics:
        """Compute direction, binocular common-mode, and aliasing diagnostics."""
        d: dict[str, Any] = {}
        if positive is not None or negative is not None:
            if positive is None or negative is None:
                raise ValueError("positive and negative must be supplied together")
            d.update(direction_selectivity(positive, negative))
        if left_drive is not None or right_drive is not None:
            if left_drive is None or right_drive is None:
                raise ValueError("left_drive and right_drive must be supplied together")
            d.update(left_right_common_mode(left_drive, right_drive))
        if aliasing_scores is not None:
            if len(aliasing_scores) != 2:
                raise ShapeError("aliasing_scores must be (left, right)")
            al, ar = map(float, aliasing_scores)
            if not np.isfinite(al) or not np.isfinite(ar):
                raise ValueError("aliasing scores must be finite")
            d["aliasing_score_left"], d["aliasing_score_right"] = al, ar
            d["aliasing_warning"] = aliasing_warning(max(al, ar), self.aliasing_threshold)
        notes = [self.backend_note] if self.backend_is_stub else []
        if d.get("aliasing_warning", False):
            notes.append("spatial aliasing risk exceeds threshold")
        return StereoDiagnostics(**d, warnings=tuple(notes))


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="auto", help="auto, stub, or path to recorded-response .npz")
    ap.add_argument("--columns", type=int, default=16)
    ap.add_argument("--t4t5", type=int, default=8)
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(1234)
    engine = StereoFlyVis(args.columns, args.t4t5, backend=args.backend)
    movie = np.full((args.frames, args.columns), 0.5, dtype=np.float32)
    if args.frames > 1:
        movie[1:] += 0.2 * rng.standard_normal(movie[1:].shape).astype(np.float32)
        movie = np.clip(movie, 0, 1)
    out = engine.process_frames(movie)
    print(json.dumps({**engine.backend_metadata,
                      "frames": len(out), "drive_shape": list(out[-1].drive.shape),
                      "drive_min": float(out[-1].drive.min()),
                      "drive_max": float(out[-1].drive.max()),
                      "cns_shape_one_frame": list(engine.expand_drive_to_cns(out[-1]).shape)},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
