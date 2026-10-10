#!/usr/bin/env python3
"""Run the project-facing engineering six-leg walking baseline.

This is a thin launcher for ``run_flygym_reference_walk.py``. The pinned
FlyGym runner owns the articulated body, controller, contact physics, trace,
and rendering. This front end only selects explicit duration/seed/left-right
controller inputs and the output directory; it does not read stripe labels or
choose a turn from visual input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUNNER = PROJECT_ROOT / "scripts" / "run_flygym_reference_walk.py"
DEFAULT_RUNTIME_PYTHON = Path("/data/environments/projects/flygym-reference/bin/python")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/artifacts/by-project/malecns-gpu/experiments/termproject-walk-demo"
)


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("walk-%Y%m%dT%H%M%S-%fZ")


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned FlyGym six-leg CPG/hybrid position-servo baseline. "
            "Left/right commands are explicit engineering inputs; no stripe "
            "condition is interpreted as a turn command."
        ),
        epilog=(
            "Each signed descending-drive channel is in [-1, 1]. Positive "
            "drives that side's nominal forward CPG phase; negative reverses "
            "its phase direction; absolute magnitude scales stride amplitude. "
            "The base CPG rhythm remains fixed at 12 Hz, so this interface has "
            "no calibrated walking-speed unit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration-s", type=float, default=2.0,
        help="simulation duration in seconds (must align to the native/control steps)",
    )
    parser.add_argument("--seed", type=int, default=0, help="upstream CPG seed")
    parser.add_argument(
        "--left-descending-drive", type=float, default=None,
        help="constant signed left-side CPG drive in [-1, 1] (default: +1)",
    )
    parser.add_argument(
        "--right-descending-drive", type=float, default=None,
        help="constant signed right-side CPG drive in [-1, 1] (default: +1)",
    )
    parser.add_argument(
        "--drive-csv", type=Path,
        help=(
            "optional time series with columns time_s,left_descending_drive," 
            "right_descending_drive; values are linearly interpolated at 1 ms"
        ),
    )
    parser.add_argument(
        "--out", type=Path,
        help="new output directory (relative paths are resolved from the project root)",
    )
    parser.add_argument("--run-id", help="run identifier stored in the output manifest")
    parser.add_argument(
        "--runtime-python", type=Path, default=Path(
            os.environ.get("FLYGYM_REFERENCE_PYTHON", str(DEFAULT_RUNTIME_PYTHON))
        ),
        help="Python interpreter for the isolated FlyGym/MuJoCo runtime",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="skip optional MuJoCo video capture; numerical traces are still recorded",
    )
    parser.add_argument("--quiet-progress", action="store_true", help="hide periodic progress lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    if args.drive_csv is not None:
        if args.left_descending_drive is not None or args.right_descending_drive is not None:
            parser.error("use either --drive-csv or constant left/right drives, not both")
        drive_csv = _project_path(args.drive_csv).resolve()
        if not drive_csv.is_file():
            parser.error(f"drive CSV does not exist: {drive_csv}")
    else:
        left = 1.0 if args.left_descending_drive is None else args.left_descending_drive
        right = 1.0 if args.right_descending_drive is None else args.right_descending_drive
        for name, value in (("left", left), ("right", right)):
            if not math.isfinite(value) or abs(value) > 1:
                parser.error(f"--{name}-descending-drive must be finite and in [-1, 1]")
        drive_csv = None

    # Keep the venv's `bin/python` symlink intact. Resolving it follows the
    # link out to the base interpreter and loses the venv's site-packages.
    runtime_python = args.runtime_python.expanduser().absolute()
    if not runtime_python.is_file():
        parser.error(
            "FlyGym reference interpreter is unavailable: "
            f"{runtime_python}. Use --runtime-python or set FLYGYM_REFERENCE_PYTHON."
        )
    if not REFERENCE_RUNNER.is_file():
        parser.error(f"pinned FlyGym runner is missing: {REFERENCE_RUNNER}")

    run_id = args.run_id or _utc_run_id()
    if not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("--run-id must be a nonempty single path component")
    output_dir = (
        _project_path(args.out).resolve()
        if args.out is not None
        else DEFAULT_OUTPUT_ROOT / run_id
    )

    command = [
        str(runtime_python), str(REFERENCE_RUNNER),
        "--duration-s", format(args.duration_s, ".17g"),
        "--seed", str(args.seed),
        "--out", str(output_dir),
        "--run-id", run_id,
    ]
    if drive_csv is not None:
        command.extend(("--drive-csv", str(drive_csv)))
    else:
        command.extend((
            "--left-descending-drive", format(left, ".17g"),
            "--right-descending-drive", format(right, ".17g"),
        ))
    if args.no_video:
        command.append("--no-video")
    if args.quiet_progress:
        command.append("--quiet-progress")

    env = os.environ.copy()
    # Hard CPU-only rendering policy for this demo, even if a parent shell was
    # configured for EGL/GPU rendering. Hide CUDA devices as an additional
    # guard; MuJoCo rendering uses Mesa OSMesa and imageio's libx264 software
    # codec in the pinned upstream renderer.
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("FLYGYM_ASSET_CACHE_DIR", "/data/models/flygym/assets")
    env.setdefault("MPLCONFIGDIR", "/data/cache/flygym/matplotlib")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
