#!/usr/bin/env python3
"""Preprocess an open-loop movie into exact 721-column FlyVis luminance.

The fitted FlyVis adapter owns the hexagonal column order. Pass its exact
left/right direction arrays here; this module deliberately has no guessed
retinal grid and never imports the optional FlyVis or Torch packages.

Movies are sampled at 200 Hz (one frame every 5 ms) by FFmpeg's ``fps`` filter.
Each sampled frame is mapped to float32 grayscale luminance in [0, 1]. The
``monocular_replicated`` mode explicitly copies one sampled 721-vector into
both eyes; it is not described as binocular POV. ``binocular_pov`` requires a
side-by-side, left-then-right source video and samples each half against its
matching fitted eye geometry.

The default projection is rectilinear with edge clamping for rays outside the
declared camera field of view. The number of clamped columns is included in the
manifest. Panoramic sources can select ``equirectangular`` instead.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np


N_COLUMNS = 721
SAMPLE_HZ = 200
SAMPLE_DT_S = 1.0 / SAMPLE_HZ
ARTIFACT_SCHEMA = "malecns.courtship_movie.columns.v1"
EYE_MODES = {"binocular_pov", "monocular_replicated"}
PROJECTIONS = {"rectilinear", "equirectangular"}
STEREO_LAYOUTS = {"side_by_side_lr"}


class CourtshipMovieError(ValueError):
    """Raised when the movie, geometry, or sampling request is invalid."""


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_directions(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise CourtshipMovieError(f"{name} must be a numeric direction array") from exc
    if array.shape != (N_COLUMNS, 3):
        raise CourtshipMovieError(
            f"{name} must have exact FlyVis shape ({N_COLUMNS}, 3), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise CourtshipMovieError(f"{name} contains NaN or infinite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1.0e-12):
        raise CourtshipMovieError(f"{name} contains a zero direction")
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-4):
        raise CourtshipMovieError(f"{name} directions must be unit vectors")
    return np.ascontiguousarray(array)


def _normalise_frames(frames: Any) -> np.ndarray:
    """Validate ``(T,H,W)`` gray or ``(T,H,W,3)`` RGB movie pixels."""
    try:
        source = np.asarray(frames)
        data = np.asarray(frames, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise CourtshipMovieError("frames must be a rectangular numeric array") from exc
    if data.ndim not in (3, 4) or data.shape[0] <= 0:
        raise CourtshipMovieError("frames must have non-empty shape (T,H,W) or (T,H,W,3)")
    if data.shape[1] < 2 or data.shape[2] < 2:
        raise CourtshipMovieError("movie frames must be at least 2x2 pixels")
    if data.ndim == 4 and data.shape[3] != 3:
        raise CourtshipMovieError("RGB movie frames must have exactly 3 channels")
    if not np.all(np.isfinite(data)):
        raise CourtshipMovieError("frames contain NaN or infinite values")
    if np.issubdtype(source.dtype, np.integer):
        if np.any(data < 0.0) or np.any(data > 255.0):
            raise CourtshipMovieError("integer movie pixels must be in [0, 255]")
        data = data / 255.0
    else:
        if np.any(data < 0.0) or np.any(data > 1.0):
            raise CourtshipMovieError("floating movie pixels must be in [0, 1]")
    if data.ndim == 4:
        # ITU-R BT.601 luma on the decoded RGB values. FFmpeg's gray decoder
        # already supplies luma and takes the 2-D path below.
        data = (0.299 * data[..., 0] + 0.587 * data[..., 1]
                + 0.114 * data[..., 2])
    return np.ascontiguousarray(data, dtype=np.float32)


def _bilinear(frame: np.ndarray, x: np.ndarray, y: np.ndarray,
              *, wrap_x: bool = False) -> np.ndarray:
    height, width = frame.shape
    if wrap_x:
        x = np.mod(x, width)
        x0 = np.floor(x).astype(np.int64)
        x1 = (x0 + 1) % width
    else:
        x = np.clip(x, 0.0, width - 1.0)
        x0 = np.floor(x).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
    y = np.clip(y, 0.0, height - 1.0)
    y0 = np.floor(y).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return ((1.0 - wy) * ((1.0 - wx) * frame[y0, x0] + wx * frame[y0, x1])
            + wy * ((1.0 - wx) * frame[y1, x0] + wx * frame[y1, x1])).astype(np.float32)


def _column_pixels(
    directions: np.ndarray,
    shape: tuple[int, int],
    *,
    projection: str,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return bilinear pixel coordinates and count of out-of-view rays."""
    height, width = shape
    forward, left, up = directions.T.astype(np.float64)
    if projection == "equirectangular":
        azimuth = np.arctan2(left, forward)
        elevation = np.arcsin(np.clip(up, -1.0, 1.0))
        x = (0.5 - azimuth / (2.0 * np.pi)) * width
        y = (0.5 - elevation / np.pi) * (height - 1)
        return x, y, 0

    tan_h = math.tan(math.radians(horizontal_fov_deg) / 2.0)
    tan_v = math.tan(math.radians(vertical_fov_deg) / 2.0)
    in_front = forward > 1.0e-12
    safe_forward = np.where(in_front, forward, 1.0e-12)
    x_angular = left / safe_forward
    y_angular = up / safe_forward
    in_view = (in_front & (np.abs(x_angular) <= tan_h)
               & (np.abs(y_angular) <= tan_v))
    x = (0.5 - 0.5 * x_angular / tan_h) * (width - 1)
    y = (0.5 - 0.5 * y_angular / tan_v) * (height - 1)
    # For rays behind the camera, choose the horizontal border from the ray's
    # lateral sign; a directly rearward ray is assigned to the left border.
    x = np.where(in_front, x, np.where(left > 0.0, 0.0, width - 1.0))
    return x, y, int((~in_view).sum())


def _sample_eye(frame: np.ndarray, directions: np.ndarray, *, projection: str,
                horizontal_fov_deg: float, vertical_fov_deg: float) -> tuple[np.ndarray, int]:
    x, y, outside = _column_pixels(
        directions, frame.shape, projection=projection,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
    )
    values = _bilinear(frame, x, y, wrap_x=(projection == "equirectangular"))
    return values, outside


def _ffprobe_video_info(path: Path) -> tuple[int, int, float]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise CourtshipMovieError("ffprobe is required to read video dimensions")
    result = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:format=duration",
         "-of", "json", str(path)],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        raise CourtshipMovieError(f"ffprobe failed for {path}: {detail}")
    try:
        report = json.loads(result.stdout)
        streams = report["streams"]
        width, height = int(streams[0]["width"]), int(streams[0]["height"])
        duration_raw = streams[0].get("duration") or report.get("format", {}).get("duration")
        duration_s = float(duration_raw)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CourtshipMovieError(f"ffprobe found no usable video stream in {path}") from exc
    if width < 2 or height < 2:
        raise CourtshipMovieError(f"video dimensions must be at least 2x2, got {width}x{height}")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise CourtshipMovieError(f"video duration must be positive and finite, got {duration_s!r}")
    return width, height, duration_s


def _decode_gray_200hz(
    path: Path,
    width: int,
    height: int,
    *,
    expected_frames: int,
    transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Decode frames one at a time and retain only transformed narrow data."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise CourtshipMovieError("ffmpeg is required to decode video")
    command = [
        executable, "-v", "error", "-noautorotate", "-i", str(path), "-map", "0:v:0",
        "-vf", "fps=fps=200:start_time=0:round=down", "-frames:v", str(expected_frames),
        "-an", "-sn", "-dn",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise CourtshipMovieError(f"could not start ffmpeg: {exc}") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    frame_bytes = width * height
    decoded: list[np.ndarray] = []
    while True:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            process.kill()
            process.wait()
            raise CourtshipMovieError("ffmpeg returned a truncated grayscale frame")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        try:
            decoded.append(np.asarray(transform(frame), dtype=np.float32))
        except Exception:
            process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()
            raise
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    code = process.wait()
    process.stdout.close()
    process.stderr.close()
    if code != 0:
        raise CourtshipMovieError(f"ffmpeg failed for {path}: {stderr or f'exit {code}'}")
    if not decoded:
        raise CourtshipMovieError(f"ffmpeg decoded no 200 Hz frames from {path}")
    if len(decoded) != expected_frames:
        raise CourtshipMovieError(
            f"ffmpeg decoded {len(decoded)} frames, expected {expected_frames} from stream duration"
        )
    try:
        return np.stack(decoded, axis=0)
    except ValueError as exc:
        raise CourtshipMovieError(f"frame preprocessing returned inconsistent shapes: {exc}") from exc


def _validate_options(
    left_directions: Any,
    right_directions: Any,
    eye_mode: str,
    stereo_layout: Optional[str],
    projection: str,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    left_dirs = _validate_directions(left_directions, "left_directions")
    right_dirs = _validate_directions(right_directions, "right_directions")
    if eye_mode not in EYE_MODES:
        raise CourtshipMovieError(f"eye_mode must be one of {sorted(EYE_MODES)}")
    if eye_mode == "binocular_pov":
        if stereo_layout not in STEREO_LAYOUTS:
            raise CourtshipMovieError("binocular_pov requires stereo_layout='side_by_side_lr'")
    elif stereo_layout is not None:
        raise CourtshipMovieError("monocular_replicated requires stereo_layout=None")
    if projection not in PROJECTIONS:
        raise CourtshipMovieError(f"projection must be one of {sorted(PROJECTIONS)}")
    for name, value in (("horizontal_fov_deg", horizontal_fov_deg),
                        ("vertical_fov_deg", vertical_fov_deg)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise CourtshipMovieError(f"{name} must be finite")
        if projection == "rectilinear" and not 0.0 < float(value) < 180.0:
            raise CourtshipMovieError(f"{name} must be strictly between 0 and 180 degrees")
    return left_dirs, right_dirs, float(horizontal_fov_deg), float(vertical_fov_deg)


def _map_frames(
    gray: np.ndarray,
    *,
    left_dirs: np.ndarray,
    right_dirs: np.ndarray,
    eye_mode: str,
    projection: str,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if eye_mode == "binocular_pov":
        if gray.shape[2] % 2:
            raise CourtshipMovieError("side-by-side video width must be even")
        midpoint = gray.shape[2] // 2
        left_frames, right_frames = gray[:, :, :midpoint], gray[:, :, midpoint:]
    else:
        left_frames, right_frames = gray, None
    sampled = np.empty((gray.shape[0], 2, N_COLUMNS), dtype=np.float32)
    outside_left = outside_right = 0
    for frame_index in range(gray.shape[0]):
        left, outside_l = _sample_eye(
            left_frames[frame_index], left_dirs, projection=projection,
            horizontal_fov_deg=horizontal_fov_deg, vertical_fov_deg=vertical_fov_deg,
        )
        if right_frames is None:
            right, outside_r = left.copy(), outside_l
        else:
            right, outside_r = _sample_eye(
                right_frames[frame_index], right_dirs, projection=projection,
                horizontal_fov_deg=horizontal_fov_deg, vertical_fov_deg=vertical_fov_deg,
            )
        sampled[frame_index, 0] = left
        sampled[frame_index, 1] = right
        outside_left += outside_l
        outside_right += outside_r
    return sampled, {"left": outside_left, "right": outside_right}


def _manifest(
    *,
    eye_mode: str,
    stereo_layout: Optional[str],
    projection: str,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    out_of_fov_sample_count: Mapping[str, int],
    left_dirs: np.ndarray,
    right_dirs: np.ndarray,
    source_frame_count: int,
    source_frame_shape: tuple[int, int],
    source_duration_s: float,
    source_path: Optional[str],
    source_sha256: Optional[str],
    decoder: str,
) -> dict[str, Any]:
    return {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_frame_count": int(source_frame_count),
        "source_frame_shape": [int(source_frame_shape[0]), int(source_frame_shape[1])],
        "source_duration_s": float(source_duration_s),
        "decoder": decoder,
        "decoder_sample_rate_hz": SAMPLE_HZ,
        "temporal_sampling": "ffmpeg fps filter at 200 Hz; zero-order hold; origin 0",
        "eye_mode": eye_mode,
        "stereo_layout": stereo_layout,
        "projection": projection,
        "horizontal_fov_deg": horizontal_fov_deg,
        "vertical_fov_deg": vertical_fov_deg,
        "out_of_fov_policy": "rectilinear edge clamp" if projection == "rectilinear" else "none",
        "out_of_fov_sample_count": dict(out_of_fov_sample_count),
        "luminance": "FFmpeg grayscale or BT.601 RGB luma; divide 8-bit values by 255",
        "orientation": "encoded stream orientation; automatic rotation disabled",
        "camera_coordinate_assumption": (
            "optical axis is head +forward; image-left maps to head +left; image-up maps to head +up"
        ),
        "left_directions_sha256": _array_sha256(left_dirs),
        "right_directions_sha256": _array_sha256(right_dirs),
        "right_eye_sampling": (
            "copy of left vector; not binocular POV"
            if eye_mode == "monocular_replicated"
            else "independent right image half and exact right directions"
        ),
    }


class CourtshipMovie:
    """Precomputed 200 Hz movie drive for the strict fitted FlyVis boundary."""

    def __init__(self, columns: Any, manifest: Mapping[str, Any]) -> None:
        try:
            array = np.asarray(columns, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise CourtshipMovieError("columns must be a numeric array") from exc
        if array.ndim != 3 or array.shape[0] <= 0 or array.shape[1:] != (2, N_COLUMNS):
            raise CourtshipMovieError(
                f"columns must have shape (T, 2, {N_COLUMNS}) with T > 0, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise CourtshipMovieError("columns contain NaN or infinite values")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise CourtshipMovieError("columns must be luminance values in [0, 1]")
        self._columns = np.ascontiguousarray(array)
        self._manifest = dict(manifest)
        self._manifest["artifact_schema"] = ARTIFACT_SCHEMA
        self._manifest["sample_hz"] = SAMPLE_HZ
        self._manifest["sample_dt_s"] = SAMPLE_DT_S
        self._manifest["sample_count"] = int(array.shape[0])
        self._manifest["columns_shape"] = list(array.shape)
        self._manifest["columns_dtype"] = "float32"
        self._manifest["columns_sha256"] = _array_sha256(self._columns)

    @classmethod
    def from_frames(
        cls,
        frames: Any,
        *,
        left_directions: Any,
        right_directions: Any,
        eye_mode: str,
        stereo_layout: Optional[str] = None,
        projection: str = "rectilinear",
        horizontal_fov_deg: float = 120.0,
        vertical_fov_deg: float = 90.0,
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
        decoder: str = "in_memory_frames",
    ) -> "CourtshipMovie":
        """Map frames already sampled at 200 Hz to exact FlyVis columns.

        The ``frames`` argument is ``(T,H,W)`` grayscale or ``(T,H,W,3)`` RGB.
        A side-by-side video is represented by concatenated eye images in each
        frame and must declare ``stereo_layout='side_by_side_lr'``.
        """
        left_dirs, right_dirs, horizontal_fov_deg, vertical_fov_deg = _validate_options(
            left_directions, right_directions, eye_mode, stereo_layout, projection,
            horizontal_fov_deg, vertical_fov_deg,
        )
        gray = _normalise_frames(frames)
        sampled, outside = _map_frames(
            gray, left_dirs=left_dirs, right_dirs=right_dirs, eye_mode=eye_mode,
            projection=projection, horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
        )
        metadata = _manifest(
            eye_mode=eye_mode, stereo_layout=stereo_layout, projection=projection,
            horizontal_fov_deg=horizontal_fov_deg, vertical_fov_deg=vertical_fov_deg,
            out_of_fov_sample_count=outside, left_dirs=left_dirs, right_dirs=right_dirs,
            source_frame_count=gray.shape[0], source_frame_shape=gray.shape[1:],
            source_duration_s=gray.shape[0] * SAMPLE_DT_S,
            source_path=source_path, source_sha256=source_sha256, decoder=decoder,
        )
        return cls(sampled, metadata)

    @classmethod
    def from_exact_columns(
        cls,
        columns: Any,
        *,
        left_directions: Any,
        right_directions: Any,
        source_metadata: Mapping[str, Any],
    ) -> "CourtshipMovie":
        """Wrap an already sampled binocular exact-column replay sequence.

        Use this for rendered scenes whose renderer already sampled each
        frame from the fitted FlyVis direction arrays. The payload must be
        ``(T, 2, 721)`` float luminance in [0, 1], ordered left then right,
        with one sample every 5 ms from t=0. ``source_metadata`` is preserved
        under the ``source_metadata`` manifest key and should identify the
        scene, render path, pose, source/config hashes, and sampling settings.
        """
        if not isinstance(source_metadata, Mapping) or not source_metadata:
            raise CourtshipMovieError("source_metadata must be a non-empty mapping")
        try:
            metadata = json.loads(json.dumps(
                dict(source_metadata), sort_keys=True, allow_nan=False,
            ))
        except (TypeError, ValueError) as exc:
            raise CourtshipMovieError("source_metadata must contain only finite JSON values") from exc
        if "sample_hz" in metadata and metadata["sample_hz"] != SAMPLE_HZ:
            raise CourtshipMovieError("exact-column source_metadata sample_hz must equal 200")
        if "sample_dt_s" in metadata and metadata["sample_dt_s"] != SAMPLE_DT_S:
            raise CourtshipMovieError("exact-column source_metadata sample_dt_s must equal 0.005")
        left_dirs = _validate_directions(left_directions, "left_directions")
        right_dirs = _validate_directions(right_directions, "right_directions")
        manifest = {
            "source_path": metadata.get("source_path"),
            "source_sha256": metadata.get("source_sha256"),
            "source_duration_s": metadata.get("source_duration_s", columns.shape[0] * SAMPLE_DT_S),
            "source_kind": "world_rendered_exact_flyvis_columns",
            "source_metadata": metadata,
            "eye_mode": "binocular_world_render",
            "stereo_layout": "direct_left_right_column_vectors",
            "projection": "direct_exact_columns",
            "horizontal_fov_deg": None,
            "vertical_fov_deg": None,
            "out_of_fov_policy": "none",
            "out_of_fov_sample_count": {"left": 0, "right": 0},
            "decoder": "not_applicable_exact_columns",
            "decoder_sample_rate_hz": SAMPLE_HZ,
            "temporal_sampling": "provided exact left/right columns; 200 Hz grid; origin 0",
            "luminance": "caller-provided finite float32 luminance in [0,1]",
            "left_directions_sha256": _array_sha256(left_dirs),
            "right_directions_sha256": _array_sha256(right_dirs),
            "right_eye_sampling": "independent renderer-provided right columns",
        }
        return cls(columns, manifest)

    @classmethod
    def from_video(
        cls,
        path: str | Path,
        *,
        left_directions: Any,
        right_directions: Any,
        eye_mode: str,
        stereo_layout: Optional[str] = None,
        projection: str = "rectilinear",
        horizontal_fov_deg: float = 120.0,
        vertical_fov_deg: float = 90.0,
    ) -> "CourtshipMovie":
        """Decode and preprocess a movie with local FFmpeg/ffprobe tools."""
        movie_path = Path(path).expanduser()
        if not movie_path.is_file():
            raise CourtshipMovieError(f"movie does not exist or is not a file: {movie_path}")
        left_dirs, right_dirs, horizontal_fov_deg, vertical_fov_deg = _validate_options(
            left_directions, right_directions, eye_mode, stereo_layout, projection,
            horizontal_fov_deg, vertical_fov_deg,
        )
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise CourtshipMovieError("both ffmpeg and ffprobe are required")
        width, height, duration_s = _ffprobe_video_info(movie_path)
        if eye_mode == "binocular_pov":
            if stereo_layout != "side_by_side_lr":
                raise CourtshipMovieError(
                    "binocular_pov requires stereo_layout='side_by_side_lr'"
                )
            if width % 2:
                raise CourtshipMovieError("side-by-side video width must be even")
        elif eye_mode == "monocular_replicated":
            if stereo_layout is not None:
                raise CourtshipMovieError("monocular_replicated requires stereo_layout=None")
        else:
            raise CourtshipMovieError(f"eye_mode must be one of {sorted(EYE_MODES)}")
        outside = {"left": 0, "right": 0}

        def transform(frame_u8: np.ndarray) -> np.ndarray:
            gray = (frame_u8.astype(np.float32) / 255.0)[None, ...]
            columns, counts = _map_frames(
                gray, left_dirs=left_dirs, right_dirs=right_dirs, eye_mode=eye_mode,
                projection=projection, horizontal_fov_deg=horizontal_fov_deg,
                vertical_fov_deg=vertical_fov_deg,
            )
            outside["left"] += counts["left"]
            outside["right"] += counts["right"]
            return columns[0]

        expected_frames = int(math.ceil(duration_s * SAMPLE_HZ - 1.0e-9))
        columns = _decode_gray_200hz(
            movie_path, width, height, expected_frames=expected_frames, transform=transform,
        )
        return cls(
            columns,
            _manifest(
                eye_mode=eye_mode, stereo_layout=stereo_layout, projection=projection,
                horizontal_fov_deg=horizontal_fov_deg, vertical_fov_deg=vertical_fov_deg,
                out_of_fov_sample_count=outside, left_dirs=left_dirs, right_dirs=right_dirs,
                source_frame_count=columns.shape[0], source_frame_shape=(height, width),
                source_duration_s=duration_s,
                source_path=str(movie_path.resolve()), source_sha256=_file_sha256(movie_path),
                decoder="ffmpeg/ffprobe",
            ),
        )

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        left_directions: Any,
        right_directions: Any,
    ) -> "CourtshipMovie":
        """Load a narrow ``.npz`` artifact and verify its exact eye geometry."""
        left_dirs = _validate_directions(left_directions, "left_directions")
        right_dirs = _validate_directions(right_directions, "right_directions")
        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                columns = np.asarray(archive["columns"], dtype=np.float32)
                raw_manifest = archive["manifest_json"].item()
            manifest = json.loads(str(raw_manifest))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CourtshipMovieError(f"could not read movie artifact {path}: {exc}") from exc
        if manifest.get("artifact_schema") != ARTIFACT_SCHEMA:
            raise CourtshipMovieError("unsupported movie artifact schema")
        if manifest.get("sample_hz") != SAMPLE_HZ or manifest.get("sample_dt_s") != SAMPLE_DT_S:
            raise CourtshipMovieError("movie artifact does not use the strict 200 Hz sampling grid")
        if manifest.get("left_directions_sha256") != _array_sha256(left_dirs):
            raise CourtshipMovieError("movie artifact left direction hash does not match runtime")
        if manifest.get("right_directions_sha256") != _array_sha256(right_dirs):
            raise CourtshipMovieError("movie artifact right direction hash does not match runtime")
        result = cls(columns, manifest)
        if manifest.get("columns_sha256") != result._manifest["columns_sha256"]:
            raise CourtshipMovieError("movie artifact column hash does not match its payload")
        return result

    @property
    def n_frames(self) -> int:
        return int(self._columns.shape[0])

    @property
    def duration_s(self) -> float:
        """Duration covered by the 5 ms sample grid."""
        return self.n_frames * SAMPLE_DT_S

    def sample(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return separate left/right float32 ``(721,)`` luminance vectors.

        Time must lie on the fixed 5 ms grid. Sampling outside the decoded
        movie fails rather than silently holding the last frame.
        """
        if isinstance(t_s, bool):
            raise CourtshipMovieError("sample time must be a finite number of seconds")
        try:
            time_s = float(t_s)
        except (TypeError, ValueError) as exc:
            raise CourtshipMovieError("sample time must be a finite number of seconds") from exc
        if not math.isfinite(time_s) or time_s < 0.0:
            raise CourtshipMovieError("sample time must be finite and non-negative")
        sample_index = int(round(time_s * SAMPLE_HZ))
        snapped = sample_index * SAMPLE_DT_S
        if not math.isclose(time_s, snapped, rel_tol=0.0, abs_tol=1.0e-7):
            raise CourtshipMovieError("sample time must align to the 5 ms FlyVis grid")
        if sample_index >= self.n_frames:
            raise IndexError(
                f"sample time {time_s:g}s is beyond movie coverage {self.duration_s:g}s"
            )
        return self._columns[sample_index, 0].copy(), self._columns[sample_index, 1].copy()

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe copy of preprocessing provenance."""
        return json.loads(json.dumps(self._manifest, sort_keys=True, allow_nan=False))

    def save(self, path: str | Path) -> Path:
        """Save only the narrow sampled columns and JSON provenance."""
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            columns=self._columns,
            manifest_json=np.asarray(json.dumps(
                self._manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
            )),
        )
        return output
