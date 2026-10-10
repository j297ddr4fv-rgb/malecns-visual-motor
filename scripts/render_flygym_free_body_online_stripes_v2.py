#!/usr/bin/env python3
"""Render short videos from completed online free-body stripe captures.

This is a presentation replay of saved qpos/qvel only. It calls MuJoCo
``mj_forward`` to rebuild render transforms and never advances physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


BUNDLE = Path(
    "/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1/"
    "online-free-body-stripes-v2"
)
PRESENTATION = BUNDLE / "presentation"
CASE_DIRS = {
    "stripes_static": BUNDLE / "runs/stripes_static",
    "stripes_positive60": BUNDLE / "runs/stripes_positive60",
    "stripes_negative60": BUNDLE / "runs/stripes_negative60",
}
FPS = 30
PLAYBACK_SPEED = 0.25
NATIVE_DT_S = 1e-4
SIM_DURATION_S = 4.0
FRAME_COUNT = int(round(SIM_DURATION_S / PLAYBACK_SPEED * FPS))
WIDTH, HEIGHT = 960, 720
VIEW_HEIGHT = 620
EXPECTED_DIMS = {"nq": 73, "nv": 72, "njnt": 67, "nu": 48, "nbody": 69, "ngeom": 70}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def source_paths(case_dir: Path) -> dict[str, Path]:
    return {
        "manifest": case_dir / "manifest.json",
        "run_record": case_dir / "run-record.json",
        "summary": case_dir / "summary.json",
        "compiled_model": case_dir / "compiled_model.xml",
        "array_order": case_dir / "array_order.json",
        "trajectory": case_dir / "trajectory.npz",
        "integrity": case_dir / "integrity.json",
        "online_integrity": case_dir / "online-integrity.json",
        "stimulus_spec": case_dir / "worker/angular_stimulus_spec.json",
    }


def hash_sources(paths: dict[str, Path]) -> dict[str, str]:
    return {key: sha256(path) for key, path in paths.items()}


def validate_case(case_key: str, case_dir: Path, mj) -> tuple[dict, dict, dict, dict]:
    paths = source_paths(case_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"capture is incomplete; missing {path}")
    manifest = read_json(paths["manifest"])
    record = read_json(paths["run_record"])
    summary = read_json(paths["summary"])
    integrity = read_json(paths["integrity"])
    online_integrity = read_json(paths["online_integrity"])
    if (manifest.get("status") != "complete" or record.get("status") != "complete"
            or summary.get("status") != "complete" or online_integrity.get("status") != "complete"):
        raise ValueError(f"{case_key}: capture/worker is not complete")
    if (record.get("case_id") != case_key
            or summary.get("run_id") != manifest.get("run_id")):
        raise ValueError(f"{case_key}: case/run identity mismatch")
    if (int(record.get("completed_neural_frames", -1)) != 800
            or int(record.get("completed_body_ticks", -1)) != 4000
            or int(record.get("physics_step_count", -1)) != 40000
            or int(record.get("native_sample_count", -1)) != 40001
            or int(summary.get("native_sample_count", -1)) != 40001
            or not math.isclose(float(summary.get("duration_s", -1)), SIM_DURATION_S, abs_tol=1e-9)):
        raise ValueError(f"{case_key}: expected a complete 4 s / 800 frame capture")
    if (not isinstance(integrity.get("files"), dict)
            or not isinstance(online_integrity.get("files"), dict)):
        raise ValueError(f"{case_key}: integrity sidecars are malformed")

    actual_model_sha = sha256(paths["compiled_model"])
    if actual_model_sha != manifest.get("compiled_model_sha256"):
        raise ValueError(f"{case_key}: compiled model hash differs from manifest")
    order = read_json(paths["array_order"])
    model = mj.MjModel.from_xml_path(str(paths["compiled_model"]))
    for field, expected in EXPECTED_DIMS.items():
        actual = int(getattr(model, field))
        if actual != expected:
            raise ValueError(f"{case_key}: model.{field}={actual}, expected {expected}")

    def names(obj_type, count: int) -> list[str]:
        return [mj.mj_id2name(model, obj_type, i) or f"unnamed-{i}" for i in range(count)]

    comparisons = {
        "native_joint_names": names(mj.mjtObj.mjOBJ_JOINT, model.njnt),
        "native_joint_qposadr": model.jnt_qposadr.astype(int).tolist(),
        "native_joint_dofadr": model.jnt_dofadr.astype(int).tolist(),
        "native_joint_type": model.jnt_type.astype(int).tolist(),
        "native_actuator_names": names(mj.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "native_actuator_trnid": model.actuator_trnid.astype(int).tolist(),
        "geom_names_by_id": names(mj.mjtObj.mjOBJ_GEOM, model.ngeom),
    }
    for field, actual in comparisons.items():
        if order.get(field) != actual:
            raise ValueError(f"{case_key}: {field} differs from the saved native array order")
    if len(order.get("body_ids", [])) != model.nbody - 1 or len(order.get("body_names", [])) != model.nbody - 1:
        raise ValueError(f"{case_key}: body-name map length differs from the model")
    for body_id, body_name in zip(order["body_ids"], order["body_names"]):
        actual = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(body_id))
        if int(body_id) <= 0 or int(body_id) >= model.nbody or actual != f"nmf/{body_name}":
            raise ValueError(f"{case_key}: body ID/name map mismatch at {body_id}: {actual!r}")
    if not math.isclose(float(model.opt.timestep), NATIVE_DT_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_key}: native model timestep differs from 0.1 ms")
    if int(record.get("seed", -1)) != 1729 and int(record.get("body_seed", -1)) != 1729:
        raise ValueError(f"{case_key}: unexpected body seed")

    expected_artifacts = manifest.get("artifact_sha256", {})
    for key in ("compiled_model", "array_order", "trajectory"):
        path = paths[key].resolve()
        expected = expected_artifacts.get(str(path))
        if not isinstance(expected, str) or expected != sha256(path):
            raise ValueError(f"{case_key}: {key} missing/mismatched in manifest artifact hashes")
    return paths, manifest, record, summary


def font(size: int):
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose_frame(body: np.ndarray, case_key: str, sim_time: float, yaw_deg: float) -> np.ndarray:
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (15, 20, 29))
    image = Image.fromarray(body, mode="RGB")
    if image.size != (WIDTH, VIEW_HEIGHT):
        image = image.resize((WIDTH, VIEW_HEIGHT), Image.Resampling.LANCZOS)
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    title = font(24)
    body_font = font(17)
    small = font(14)
    draw.rectangle((0, 0, WIDTH, 86), fill=(15, 20, 29))
    draw.text((24, 10), f"ONLINE VISION • {case_key}", font=title, fill=(238, 244, 250))
    draw.text((25, 46), "Fresh NN outputs → engineered gait controller • captured body pose", font=body_font,
              fill=(178, 193, 210))
    draw.rectangle((0, 660, WIDTH, HEIGHT), fill=(15, 20, 29))
    draw.text((24, 670), f"sim {sim_time:0.3f} s", font=body_font, fill=(240, 244, 249))
    draw.text((195, 672), "playback 0.25×", font=body_font, fill=(99, 205, 183))
    draw.text((400, 672), f"actual thorax yaw {yaw_deg:+.2f}°", font=body_font, fill=(245, 195, 95))
    draw.text((24, 700), "Saved qpos/qvel replay • mj_forward only • no physics step • no synthetic stripe overlay",
              font=small, fill=(151, 165, 181))
    progress = min(1.0, max(0.0, sim_time / SIM_DURATION_S))
    x0, x1, y = 24, WIDTH - 24, 654
    draw.line((x0, y, x1, y), fill=(79, 92, 108), width=3)
    px = x0 + round((x1 - x0) * progress)
    draw.line((px, y - 4, px, y + 4), fill=(239, 198, 91), width=3)
    return np.asarray(canvas, dtype=np.uint8)


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                          check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def render_case(case_key: str, case_dir: Path, output_dir: Path) -> dict:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import mujoco as mj
    import imageio.v2 as imageio

    paths, manifest, record, summary = validate_case(case_key, case_dir, mj)
    source_before = hash_sources(paths)
    with np.load(paths["trajectory"], allow_pickle=False) as z:
        required = ("time_s", "qpos", "qvel", "thorax_heading_yaw_rad")
        if any(key not in z.files for key in required):
            raise ValueError(f"{case_key}: trajectory is missing qpos/qvel/time/yaw arrays")
        time_s, qpos, qvel, yaw = (z[key].copy() for key in required)
    model = mj.MjModel.from_xml_path(str(paths["compiled_model"]))
    if qpos.shape != (40001, model.nq) or qvel.shape != (40001, model.nv):
        raise ValueError(f"{case_key}: saved state array shape is incompatible with model dimensions")
    if yaw.shape != (40001,) or time_s.shape != (40001,):
        raise ValueError(f"{case_key}: saved time/yaw array shape mismatch")
    if not (np.isfinite(time_s).all() and np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(yaw).all()):
        raise ValueError(f"{case_key}: nonfinite saved pose data")
    if (abs(float(time_s[0])) > 1e-12 or abs(float(time_s[-1]) - SIM_DURATION_S) > 1e-9
            or not np.allclose(np.diff(time_s), NATIVE_DT_S, rtol=0.0, atol=1e-10)):
        raise ValueError(f"{case_key}: saved native timebase differs from the 4 s capture")

    out = output_dir / case_key
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"{case_key}-saved-pose-0p25x.mp4"
    still = out / f"{case_key}-t2s.png"
    metadata_path = out / f"{case_key}-render-metadata.json"
    if any(p.exists() for p in (video, still, metadata_path)):
        raise FileExistsError(f"refusing to overwrite renderer output for {case_key}")
    temp_video = out / f".{video.stem}.rendering-{os.getpid()}.mp4"

    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.time = float(time_s[0])
    mj.mj_forward(model, data)
    thorax_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "nmf/c_thorax")
    if thorax_id < 1:
        raise ValueError(f"{case_key}: expected thorax body missing from model")

    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = data.xpos[thorax_id]
    camera.lookat[2] -= 0.55
    camera.distance = 8.5 * float(model.stat.extent)
    camera.azimuth = 135.0
    camera.elevation = -20.0
    renderer = mj.Renderer(model, height=VIEW_HEIGHT, width=WIDTH)
    frame_count = FRAME_COUNT
    still_frame = int(round(2.0 * FPS / PLAYBACK_SPEED))
    last_time = None
    try:
        with imageio.get_writer(temp_video, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_params=["-threads", "1"]) as writer:
            for frame_i in range(frame_count):
                requested_s = frame_i * PLAYBACK_SPEED / FPS
                index = int(np.clip(round(requested_s / NATIVE_DT_S), 0, len(time_s) - 1))
                sim_time = float(time_s[index])
                data.qpos[:] = qpos[index]
                data.qvel[:] = qvel[index]
                data.time = sim_time
                mj.mj_forward(model, data)
                if (not np.array_equal(data.qpos, qpos[index]) or not np.array_equal(data.qvel, qvel[index])
                        or not math.isclose(float(data.time), sim_time, abs_tol=1e-12, rel_tol=0.0)):
                    raise RuntimeError(f"{case_key}: mj_forward altered saved state/time")
                renderer.update_scene(data, camera)
                body = renderer.render()
                frame = compose_frame(body, case_key, sim_time, math.degrees(float(yaw[index])))
                writer.append_data(frame)
                if frame_i == still_frame:
                    from PIL import Image
                    Image.fromarray(frame, mode="RGB").save(still, format="PNG")
                last_time = sim_time
                if frame_i == 0 or (frame_i + 1) % 120 == 0 or frame_i + 1 == frame_count:
                    print(f"render_progress case={case_key} frame={frame_i + 1}/{frame_count} sim_time_s={sim_time:.3f} mj_step_called=false",
                          flush=True)
        temp_video.replace(video)
    except BaseException:
        if temp_video.exists():
            temp_video.unlink()
        if still.exists() and not video.exists():
            still.unlink()
        raise
    finally:
        renderer.close()

    source_after = hash_sources(paths)
    if source_before != source_after:
        raise RuntimeError(f"{case_key}: raw capture source hashes changed during rendering")
    probe = ffprobe(video)
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if not stream or stream.get("codec_name") != "h264" or (int(stream.get("width", 0)), int(stream.get("height", 0))) != (WIDTH, HEIGHT):
        raise RuntimeError(f"{case_key}: encoded video stream metadata is invalid")
    if int(stream.get("nb_frames", -1)) != frame_count or not math.isclose(float(probe["format"]["duration"]), 16.0, abs_tol=0.1):
        raise RuntimeError(f"{case_key}: expected 480 frames and 16 s playback")
    model_root = ET.fromstring(paths["compiled_model"].read_bytes())
    mesh_paths = sorted({node.attrib["file"] for node in model_root.iter("mesh") if "file" in node.attrib})
    metadata = {
        "schema_version": "flygym-online-free-body-stripes-v2-saved-pose-video-1",
        "status": "render_complete",
        "case": case_key,
        "classification": "fresh neural outputs via engineered gait controller, online vision; recorded-qpos presentation replay",
        "claim_boundary": "not direct-muscle neural actuation; video does not prove biological stripe behavior",
        "stimulus_spec_sha256": source_before["stimulus_spec"],
        "simulated_duration_s": SIM_DURATION_S,
        "playback_speed": PLAYBACK_SPEED,
        "playback_duration_s": 16.0,
        "fps": FPS,
        "frame_count": frame_count,
        "video_size_px": [WIDTH, HEIGHT],
        "last_frame_saved_simulation_time_s": last_time,
        "actual_thorax_yaw_deg_first_last": [math.degrees(float(yaw[0])), math.degrees(float(yaw[-1]))],
        "replay_method": "saved qpos/qvel/time_s copied into private MjData then mj_forward; mj_step was never called",
        "ground_view": "fixed free camera in world coordinates; no moving visual stimulus or synthetic stripe overlay",
        "saved_array_shapes": {"qpos": list(qpos.shape), "qvel": list(qvel.shape), "time_s": list(time_s.shape), "thorax_heading_yaw_rad": list(yaw.shape)},
        "compiled_model_dimensions": {k: int(getattr(model, k)) for k in EXPECTED_DIMS},
        "array_order_verified": True,
        "capture_status": {"manifest": manifest.get("status"), "run_record": record.get("status"), "summary": summary.get("status")},
        "raw_source_hashes_before": source_before,
        "raw_source_hashes_after": source_after,
        "raw_source_hashes_unchanged": True,
        "external_mesh_files_referenced_by_xml": mesh_paths,
        "output": {
            "video": str(video), "video_sha256": sha256(video), "video_bytes": video.stat().st_size,
            "screenshot": str(still), "screenshot_sha256": sha256(still), "ffprobe": probe,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_DIRS), required=True)
    parser.add_argument("--case-dir", type=Path, help="override case directory for an independently completed capture")
    parser.add_argument("--output-dir", type=Path, default=PRESENTATION)
    args = parser.parse_args()
    case_dir = (args.case_dir or CASE_DIRS[args.case]).expanduser().resolve()
    result = render_case(args.case, case_dir, args.output_dir.expanduser().resolve())
    print(json.dumps({"status": result["status"], "case": args.case, "video": result["output"]["video"],
                      "video_sha256": result["output"]["video_sha256"], "metadata": str(args.output_dir.resolve() / args.case / f"{args.case}-render-metadata.json")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
