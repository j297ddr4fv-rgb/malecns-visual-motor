#!/usr/bin/env python3
"""Compact offline progress and descriptive metrics for the digital-fly queue."""
from __future__ import annotations
import csv, hashlib, io, json, math, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "digital-fly-research-queue-progress-v1"
COMPLETE = {"complete", "completed", "success", "succeeded"}
FAILED = {"failed", "error", "cancelled", "canceled", "aborted", "interrupted"}
CACHE_VERSION = 2
def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)): return None
    x = float(x); return x if math.isfinite(x) else None
def _json(path: Path) -> dict[str, Any] | None:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None
    return value if isinstance(value, dict) else None
def _stat(path: Path) -> dict[str, int] | None:
    try: s = path.stat()
    except OSError: return None
    return {"dev": s.st_dev, "ino": s.st_ino, "size": s.st_size, "mtime_ns": s.st_mtime_ns}
def _first(objects: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for obj in objects:
        for key in keys:
            if (value := _num(obj.get(key))) is not None: return value
    return None
def _windows(case: dict[str, Any]) -> dict[str, tuple[float, float]]:
    result = {}
    configured = case.get("analysis_windows")
    if isinstance(configured, dict):
        for name, pair in configured.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = _num(pair[0]), _num(pair[1])
                if a is not None and b is not None and b > a: result[str(name)] = (a, b)
    duration = _num(case.get("duration_s")) or 0.0
    if result:
        end = duration or max(b for _, b in result.values())
        if end > 0: result.setdefault("overall", (0.0, end))
    elif duration > 0:
        result = {"early": (0.0, duration / 2), "late": (duration / 2, duration),
                  "overall": (0.0, duration)}
    return result
def _frame_time(row: dict[str, Any]) -> float | None:
    value = _first([row], ("retinal_pattern_time_s", "time_s", "t_s"))
    return value if value is not None else (seq * .005 if (seq := _num(row.get("seq"))) is not None else None)
def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _scrub(v) for k, v in sorted(value.items())
                if not any(t in str(k).lower() for t in
                           ("time", "seq", "tick_index", "condition", "stimulus", "case_id"))}
    if isinstance(value, list): return [_scrub(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value
def _valid_hashes(value: Any) -> bool: return isinstance(value, dict) and all(isinstance((h := value.get(k)), str) and len(h) == 64 and all(c in "0123456789abcdefABCDEF" for c in h) for k in ("left", "right"))
def _empty(windows: dict[str, tuple[float, float]]) -> dict[str, Any]:
    return {"windows": {n: {"frame_rows": 0, "p_sum": 0., "p_n": 0,
                            "d_applied_sum": 0., "d_applied_n": 0,
                            "d_queued_sum": 0., "d_queued_n": 0} for n in windows},
            "malformed_rows": 0, "prefix_sha256": "0" * 64,
            "prefix_rows": 0, "prefix_retinal_rows": 0}
def _add(acc: dict[str, Any], name: str, value: float | None) -> None:
    if value is not None:
        acc[f"{name}_sum"] += value; acc[f"{name}_n"] += 1

def _frames(path: Path, old: dict[str, Any], wins: dict[str, tuple[float, float]],
            prefix_end: float) -> dict[str, Any]:
    st = _stat(path)
    if st is None: return {"stat": None, "offset": 0, "acc": _empty(wins)}
    prev, offset, acc = old.get("frames_stat"), int(old.get("frames_offset", 0)), old.get("frames_acc")
    reset = (not isinstance(prev, dict) or st["dev"] != prev.get("dev") or st["ino"] != prev.get("ino")
             or st["size"] < offset or (st["size"] == prev.get("size") and st["mtime_ns"] != prev.get("mtime_ns")))
    if reset or not isinstance(acc, dict) or set(acc.get("windows", {})) != set(wins):
        acc, offset = _empty(wins), 0
    try:
        with path.open("rb") as f:
            f.seek(offset)
            while True:
                start, raw = f.tell(), f.readline()
                if not raw: offset = f.tell(); break
                if not raw.endswith(b"\n"): offset = start; break
                offset = f.tell()
                if not raw.strip(): continue
                try: row = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError: acc["malformed_rows"] += 1; continue
                if not isinstance(row, dict) or (t := _frame_time(row)) is None: continue
                dec = row.get("decoder_output") if isinstance(row.get("decoder_output"), dict) else {}
                applied = row.get("body_command_applied_this_frame") or {}
                queued = row.get("decoder_output_queued_for_next_frame") or {}
                if not isinstance(applied, dict): applied = {}
                if not isinstance(queued, dict): queued = {}
                p = _first([row, dec], ("raw_worker_leg_power_P", "power_raw", "raw_leg_power_P", "raw_leg_power"))
                da, dq = _num(applied.get("signed_d")), _num(queued.get("signed_d"))
                for name, (a, b) in wins.items():
                    if a <= t < b:
                        w = acc["windows"][name]; w["frame_rows"] += 1
                        _add(w, "p", p); _add(w, "d_applied", da); _add(w, "d_queued", dq)
                if 0 <= t < prefix_end:
                    payload = {"retinal_column_hashes": _scrub(row.get("retinal_column_hashes")), "decoder_output": _scrub(dec),
                               "worker_raw": [row.get("raw_worker_leg_power_P"), row.get("raw_worker_DNp15_L_rate_hz"), row.get("raw_worker_DNp15_R_rate_hz")]}
                    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                    acc["prefix_sha256"] = hashlib.sha256((acc["prefix_sha256"] + digest).encode()).hexdigest()
                    acc["prefix_rows"] += 1
                    acc["prefix_retinal_rows"] += _valid_hashes(row.get("retinal_column_hashes"))
    except OSError: pass
    return {"stat": st, "offset": offset, "acc": acc}
def _mean(total: Any, count: Any) -> float | None:
    n = int(count or 0); return float(total) / n if n else None
def _frame_summary(acc: dict[str, Any], wins: dict[str, tuple[float, float]]) -> dict[str, Any]:
    out = {}
    for name in wins:
        w = acc.get("windows", {}).get(name, {})
        out[name] = {"frame_rows": int(w.get("frame_rows", 0)),
                     "P_raw_mean": _mean(w.get("p_sum", 0), w.get("p_n", 0)),
                     "d_applied_mean": _mean(w.get("d_applied_sum", 0), w.get("d_applied_n", 0)),
                     "d_queued_mean": _mean(w.get("d_queued_sum", 0), w.get("d_queued_n", 0))}
    return out

def _trajectory(path: Path, wins: dict[str, tuple[float, float]]) -> dict[str, Any] | None:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            def get(*keys: str) -> Any:
                return next((np.asarray(z[k]) for k in keys if k in z.files), None)
            t, pos = get("time_s", "trajectory_time_s"), get("thorax_position_mm", "position_mm")
            yaw, yaw_deg = get("thorax_yaw_rad", "thorax_heading_yaw_rad", "yaw_rad"), get("thorax_yaw_deg", "yaw_deg")
            heading = get("thorax_heading_xy", "heading_xy")
            quat, com = get("thorax_quat_wxyz"), get("com_position_mm")
            contact, ct = get("control_foot_contact_fz_up_native"), get("control_time_s")
    except (ImportError, OSError, ValueError, KeyError): return None
    if t is None or pos is None or (yaw is None and yaw_deg is None): return None
    t, pos = np.asarray(t, float).reshape(-1), np.asarray(pos, float)
    if pos.ndim != 2 or len(pos) != len(t) or pos.shape[1] < 2: return None
    if pos.shape[1] == 2: pos = np.column_stack((pos, np.zeros(len(t))))
    pos = pos[:, :3]
    y = np.radians(np.asarray(yaw_deg, float).reshape(-1)) if yaw is None else np.asarray(yaw, float).reshape(-1)
    if len(y) != len(t): return None
    valid = np.isfinite(t) & np.isfinite(y) & np.all(np.isfinite(pos), axis=1)
    if valid.sum() < 2: return None
    t, pos, y = t[valid], pos[valid], y[valid]
    order = np.argsort(t, kind="stable"); t, pos, y = t[order], pos[order], y[order]
    unique = np.r_[True, np.diff(t) > 0]; t, pos, y = t[unique], pos[unique], np.unwrap(y[unique])
    q = np.asarray(quat, float)[valid][order][unique] if quat is not None and len(quat) == len(valid) else None
    center = np.asarray(com, float)[valid][order][unique] if com is not None and len(com) == len(valid) else None
    if len(t) < 2: return None
    if heading is not None and np.asarray(heading).ndim == 2 and len(heading) == len(valid):
        h = np.asarray(heading, float)[valid][order][unique, :2]
    else: h = None
    out = {"source": path.name, "sample_count": int(len(t)), "windows": {}}
    for name, (a, b) in wins.items():
        lo, hi = max(a, float(t[0])), min(b, float(t[-1]))
        if abs(lo - a) <= 1e-8: lo = a
        if abs(hi - b) <= 1e-8: hi = b
        if hi <= lo:
            out["windows"][name] = {"status": "unavailable", "observed_interval_s": None, "sample_count": 0}; continue
        tt = np.r_[lo, t[(t > lo) & (t < hi)], hi]
        p = np.column_stack([np.interp(tt, t, pos[:, j]) for j in range(3)])
        yy = np.interp(tt, t, y); dp = np.diff(p, axis=0); dt = hi - lo
        if h is not None and np.any(np.all(np.isfinite(h), axis=1)):
            good = np.all(np.isfinite(h), axis=1)
            hx, hy = np.interp(tt, t[good], h[good, 0]), np.interp(tt, t[good], h[good, 1])
            norm = np.hypot(hx, hy); ok = norm > 1e-12
            hx[ok] /= norm[ok]; hy[ok] /= norm[ok]
            hx[~ok], hy[~ok] = np.cos(yy[~ok]), np.sin(yy[~ok])
        else: hx, hy = np.cos(yy), np.sin(yy)
        path_mm = float(np.linalg.norm(dp, axis=1).sum())
        forward_mm = float((dp[:, 0] * (hx[:-1] + hx[1:]) / 2 + dp[:, 1] * (hy[:-1] + hy[1:]) / 2).sum())
        row = {"status": "complete" if abs(lo - a) <= 1e-8 and abs(hi - b) <= 1e-8 else "partial_window",
               "requested_interval_s": [a, b], "observed_interval_s": [lo, hi], "sample_count": len(tt),
               "native_yaw_change_deg": float(np.degrees(yy[-1] - yy[0])),
               "native_yaw_rate_deg_s": float(np.degrees(yy[-1] - yy[0]) / dt),
               "forward_speed_mm_s": forward_mm / dt, "path_speed_mm_s": path_mm / dt,
               "path_length_mm": path_mm}
        selected_pose = (t >= lo) & (t < hi)
        if q is not None and q.ndim == 2 and q.shape[1] == 4 and selected_pose.any():
            norm2 = np.sum(q * q, axis=1)
            up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) / np.maximum(norm2, 1e-12)
            row["upright_fraction_tilt_under_45deg"] = float(np.mean(up_z[selected_pose] > math.sqrt(.5)))
        if center is not None and center.ndim == 2 and center.shape[1] >= 3 and selected_pose.any():
            row["com_z_min_mm"] = float(np.min(center[selected_pose, 2]))
        if contact is not None and ct is not None:
            c, ctime = np.asarray(contact, float), np.asarray(ct, float).reshape(-1)
            selected = (ctime >= lo) & (ctime < hi) & np.isfinite(ctime)
            if c.ndim >= 2 and len(c) == len(ctime):
                row["positive_contact_fraction"] = float(np.mean(c[selected] > 0)) if selected.any() else None
                row["at_least_two_positive_contacts_fraction"] = float(np.mean(np.sum(c[selected] > 0, axis=1) >= 2)) if selected.any() else None
        out["windows"][name] = row
    return out
def _full_stimulus(metrics: dict[str, Any] | None) -> bool:
    w = (metrics or {}).get("windows", {}).get("stimulus", {}); req, obs = w.get("requested_interval_s"), w.get("observed_interval_s")
    return bool(isinstance(req, list) and isinstance(obs, list) and len(req) == len(obs) == 2
                and abs(req[0] - 1.0) <= 1e-8 and abs(req[1] - 13.0) <= 1e-8
                and abs(obs[0] - 1.0) <= 1e-8 and abs(obs[1] - 13.0) <= 1e-8)
def _wall(summary: dict[str, Any], record: dict[str, Any], state_case: dict[str, Any]) -> float | None:
    src = [record.get("timing_s", {}), record, summary.get("timing_s", {}), summary, state_case]; return _first([x for x in src if isinstance(x, dict)], ("case_wall_s", "walltime_s", "wall_s", "elapsed_s", "elapsed_seconds"))
def _run_dir(root: Path, case: dict[str, Any], sc: dict[str, Any]) -> Path:
    raw = sc.get("run_dir", sc.get("case_dir", case.get("run_dir", case.get("id", "")))); p = Path(str(raw)); return p if p.is_absolute() else root / p
def _epoch(value: Any) -> float | None:
    if (number := _num(value)) is not None: return number
    try: return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError): return None
def _tail_neural(path: Path, limit: int = 262144) -> tuple[int | None, int | None]:
    st = _stat(path)
    if st is None: return None, None
    if not st["size"]: return 0, None
    try:
        with path.open("rb") as f: f.seek(max(0, st["size"] - limit)); lines = f.read().splitlines()
    except OSError: return None, None
    for line in reversed(lines):
        try: row = json.loads(line.decode("utf-8", errors="replace"))
        except json.JSONDecodeError: continue
        if isinstance(row, dict) and (seq := _num(row.get("seq"))) is not None: return max(0, int(seq) + 1), int(seq)
    return None, None
def _queue_info(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    cases = state.get("cases", {}); cases = cases if isinstance(cases, dict) else {}
    current = state.get("current_case"); item = cases.get(current, {}) if current else {}
    item = item if isinstance(item, dict) else {}; active = str(item.get("status", "")).lower() == "running"
    now, age, frames, last_seq = datetime.now(timezone.utc), None, None, None
    if active:
        start = _epoch(item.get("started_epoch_s", item.get("started_at")))
        age = max(0.0, now.timestamp() - start) if start is not None else None
        path = Path(str(item.get("run_dir", root / "runs" / str(current))))
        if not path.is_absolute(): path = root / path
        frames, last_seq = _tail_neural(path / "online-frames.jsonl")
    pending = sum(str(v.get("status", "")).lower() == "pending" for v in cases.values() if isinstance(v, dict))
    running = sum(str(v.get("status", "")).lower() == "running" for v in cases.values() if isinstance(v, dict))
    walls = [_num(v.get("wall_s")) for v in cases.values() if isinstance(v, dict) and str(v.get("status", "")).lower() in COMPLETE]
    walls = [x for x in walls if x is not None]; mean_wall = sum(walls) / len(walls) if walls else None
    remaining = (max(0.0, mean_wall * (pending + running) - (min(age, mean_wall) if active and age is not None else 0.0))
                 if mean_wall is not None else (0.0 if state.get("status") == "complete" else None))
    eta = (datetime.fromtimestamp(now.timestamp() + remaining, timezone.utc).isoformat() if remaining is not None else None)
    return {"queue_status": state.get("status"), "current_case": current, "started_at": state.get("started_at"),
            "current_case_age_s": age, "recorded_neural_frames": frames, "last_recorded_neural_seq": last_seq,
            "pending_count": pending, "completed_mean_wall_s": mean_wall, "estimated_remaining_s": remaining,
            "estimated_completion_utc": eta,
            "eta_method": "estimate from mean completed wall_s; counts pending and running cases, subtracts active-case age capped at the mean"}

def _case(case: dict[str, Any], sc: dict[str, Any], run: Path,
          old: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cid, wins = str(case.get("id", "")), _windows(case); old = old if isinstance(old, dict) else {}
    wkey = {k: list(v) for k, v in wins.items()}; old = old if old.get("windows_key") == wkey and old.get("cache_version") == CACHE_VERSION else {}
    sp = run / "summary.json"
    if not sp.is_file(): sp = run / "window-summary.json"
    record = _json(run / "run-record.json") or {}
    ss, tp = _stat(sp), next((run / n for n in ("trajectory.npz", "trajectory_light.npz", "light-trajectory.npz") if (run / n).is_file()), None)
    ts = _stat(tp) if tp else None
    summary = (old.get("summary_data") if ss == old.get("summary_sig") else _json(sp)) or {}
    trajectory = old.get("trajectory_metrics") if ts == old.get("trajectory_sig") else (_trajectory(tp, wins) if tp else None)
    onset = _num(case.get("onset_s"))
    if onset is None: onset = wins.get("pre", (0., 0.))[1]
    frame = _frames(run / "online-frames.jsonl", old, wins, onset)
    acc, state_status = frame["acc"], str(sc.get("status", "pending")).lower()
    summary_status = str(summary.get("status", "")).lower()
    if state_status in COMPLETE:
        eligible = summary_status in COMPLETE and _full_stimulus(trajectory)
        ast, reason = (("complete", None) if eligible else
                       ("complete_output_missing", "complete summary and full stimulus trajectory coverage from 1 to 13 s required"))
    elif state_status == "interrupted": eligible, ast, reason = False, "interrupted", "run interrupted; excluded"
    elif state_status in FAILED or "fail" in state_status: eligible, ast, reason = False, "failed", "failed or aborted run"
    elif state_status in {"running", "started", "in_progress"}: eligible, ast, reason = False, "running", "not complete"
    elif state_status == "pending": eligible, ast, reason = False, "pending", "not started"
    else: eligible, ast, reason = False, "missing", "no run state or artifacts"
    signature = {"sha256": acc["prefix_sha256"] if acc["prefix_rows"] else None,
                 "rows": acc["prefix_rows"], "retinal_hash_rows": acc["prefix_retinal_rows"],
                 "available": bool(acc["prefix_rows"] and acc["prefix_rows"] == round(onset / .005) and acc["prefix_rows"] == acc["prefix_retinal_rows"])}
    row = {"id": cid, "family": case.get("family"), "condition": case.get("condition"),
           "seed": case.get("seed"), "duration_s": case.get("duration_s"),
           "reference_id": case.get("reference_id"), "state_status": state_status,
           "analysis_status": ast, "completed_statistic_eligible": eligible, "excluded_reason": reason,
           "run_dir": str(run), "summary_status": summary.get("status"),
           "walltime_s": _wall(summary, record, sc), "trajectory": trajectory,
           "controller": _frame_summary(acc, wins), "prefix_signature": signature,
           "malformed_jsonl_rows_skipped": acc["malformed_rows"],
           "analysis_windows": wkey}
    cache = {"cache_version": CACHE_VERSION, "windows_key": wkey, "summary_sig": ss, "summary_data": summary,
             "trajectory_sig": ts, "trajectory_metrics": trajectory,
             "frames_stat": frame["stat"], "frames_offset": frame["offset"], "frames_acc": acc}
    return row, cache
def _references(rows: list[dict[str, Any]]) -> None:
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        ref = by_id.get(str(r.get("reference_id"))) if r.get("reference_id") else None
        c = {"reference_id": r.get("reference_id"), "status": "unavailable",
             "yaw_change_residual_deg": None, "forward_speed_residual_mm_s": None,
             "path_speed_residual_mm_s": None, "prefix_match": None, "scientific_issue": None}
        if ref and r.get("seed") != ref.get("seed"):
            c["status"], c["scientific_issue"] = "reference_seed_mismatch", "reference_id points to a different seed"
        elif ref and ref["completed_statistic_eligible"] and r["completed_statistic_eligible"]:
            c["status"] = "matched"
            w = "stimulus" if "stimulus" in r["trajectory"]["windows"] else "overall"
            c["analysis_window"] = w
            a, b = r["trajectory"]["windows"].get(w, {}), ref["trajectory"]["windows"].get(w, {})
            for key, dest in (("native_yaw_change_deg", "yaw_change_residual_deg"),
                              ("forward_speed_mm_s", "forward_speed_residual_mm_s"),
                              ("path_speed_mm_s", "path_speed_residual_mm_s")):
                x, y = _num(a.get(key)), _num(b.get(key)); c[dest] = x - y if x is not None and y is not None else None
            x, y = r["prefix_signature"], ref["prefix_signature"]
            if x["available"] and y["available"]:
                c["prefix_match"] = x["sha256"] == y["sha256"]
                if not c["prefix_match"]: c["scientific_issue"] = "pre-onset retinal/decoder prefix differs from matched reference"
        elif ref: c["status"] = "reference_or_case_not_complete"
        r["matched_static"] = c
def _groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("family") or ""), json.dumps(r.get("condition"), sort_keys=True, ensure_ascii=False))
        groups.setdefault(key, []).append(r)
    output = []
    keys = ("native_yaw_change_deg", "forward_speed_mm_s", "path_speed_mm_s")
    for (family, condition), members in sorted(groups.items()):
        good = [r for r in members if r["completed_statistic_eligible"]]
        excluded: dict[str, int] = {}
        for r in members:
            if not r["completed_statistic_eligible"]:
                s = r["analysis_status"]; excluded[s] = excluded.get(s, 0) + 1
        means = {}
        for key in keys:
            w = "stimulus" if good and "stimulus" in good[0].get("trajectory", {}).get("windows", {}) else "overall"
            vals = [_num(r.get("trajectory", {}).get("windows", {}).get(w, {}).get(key)) for r in good]
            vals = [v for v in vals if v is not None]
            means[key] = sum(vals) / len(vals) if vals else None
        output.append({"family": family, "condition": json.loads(condition), "planned_count": len(members),
                       "completed_count": len(good), "excluded_by_status": excluded,
                       "analysis_window": "stimulus" if good and "stimulus" in good[0].get("trajectory", {}).get("windows", {}) else "overall",
                       "completed_mean": means})
    return output

def summarize_queue(root: Path, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Summarize completed and in-progress cases, resuming append-only JSONL scans."""
    root = Path(root); cases = plan.get("cases", []) if isinstance(plan, dict) else []
    state_cases = state.get("cases", {}) if isinstance(state, dict) else {}
    prior = _json(root / "progress.json") or {}; old_cache = prior.get("_analysis_cache", {}); old_cache = old_cache if isinstance(old_cache, dict) else {}
    if not isinstance(cases, list): cases = []
    if not isinstance(state_cases, dict): state_cases = {}
    results, cache = [], {}
    for case in cases:
        if not isinstance(case, dict): continue
        cid = str(case.get("id", "")); sc = state_cases.get(cid, {})
        if not isinstance(sc, dict): sc = {"status": str(sc)}
        row, cached = _case(case, sc, _run_dir(root, case, sc), old_cache.get(cid, {}))
        results.append(row); cache[cid] = cached
    _references(results)
    counts: dict[str, int] = {}
    for r in results: counts[r["analysis_status"]] = counts.get(r["analysis_status"], 0) + 1
    return {"schema_version": SCHEMA, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "planned_count": len(results), "status_counts": counts,
            "runtime_resources": _json(root / "runtime.json") or {},
            **_queue_info(root, state),
            "completed_statistic_count": sum(r["completed_statistic_eligible"] for r in results),
            "completed_statistics": _groups(results), "cases": results,
            "interpretation": "Descriptive digital-simulation summaries only; biological acceptance and validation are not assessed.",
            "_analysis_cache": cache}
def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def _markdown(report: dict[str, Any]) -> str:
    issues = sum(bool(r.get("matched_static", {}).get("scientific_issue")) for r in report["cases"])
    incomplete = report["status_counts"].get("complete_output_missing", 0)
    age = report.get("current_case_age_s")
    age_text = "—" if age is None else (f"{age / 3600:.1f} h" if age >= 3600 else f"{age / 60:.1f} min")
    frames = report.get("recorded_neural_frames"); frames_text = "unavailable" if frames is None else str(frames)
    remaining = report.get("estimated_remaining_s")
    eta_text = ("unavailable until a completed walltime is recorded" if remaining is None else
                f"~{remaining / 3600:.1f} h, around {report.get('estimated_completion_utc')}")
    lines = ["# Digital fly research queue progress", "", f"Updated: `{report['updated_utc']}`", "",
             f"**Prefix scientific issues: {issues}; incomplete output cases: {incomplete}.**", "",
             f"Queue: `{report.get('queue_status')}`; started: `{report.get('started_at') or '—'}`.",
             f"Current case: `{report.get('current_case') or '—'}`; age: {age_text}; recorded neural frames: {frames_text}.",
             f"ETA estimate: {eta_text}; {report.get('pending_count', 0)} pending cases, based on completed walltimes.", "",
             f"Cases: {report['planned_count']} planned; {report['completed_statistic_count']} eligible for completed descriptive statistics.",
             "", "State counts: " + (", ".join(f"{k}={v}" for k, v in sorted(report["status_counts"].items())) or "none"), "",
             "Failed, interrupted, running, pending, missing and incomplete-output cases are excluded from completed means.", "",
             "| Family | Condition | Complete / planned | Mean yaw change (°) | Forward speed (mm/s) | Path speed (mm/s) |",
             "|---|---|---:|---:|---:|---:|"]
    for g in report["completed_statistics"]:
        m = g["completed_mean"]; show = lambda x: "—" if x is None else f"{x:.4g}"
        lines.append(f"| {g['family']} | {json.dumps(g['condition'], ensure_ascii=False, sort_keys=True)} | "
                     f"{g['completed_count']} / {g['planned_count']} | {show(m['native_yaw_change_deg'])} | "
                     f"{show(m['forward_speed_mm_s'])} | {show(m['path_speed_mm_s'])} |")
    if not report["completed_statistics"]: lines.append("| — | — | 0 / 0 | — | — | — |")
    resources = report.get("runtime_resources", {})
    if resources:
        current, peak = resources.get("memory.current", 0), resources.get("memory.peak", 0)
        lines += ["", f"Host cgroup memory: {current / 1024**3:.2f} GiB current / {peak / 1024**3:.2f} GiB peak.",
                  f"Data disk free: {resources.get('data_free_bytes', 0) / 1024**3:.1f} GiB; GPU: {resources.get('gpu', 'unavailable')}."]
    lines += ["", "Windowed P/d, matched-reference residuals and prefix comparisons are detailed in `results.csv` and `progress.json`.",
              "", report["interpretation"], ""]
    return "\n".join(lines)
def _csv(report: dict[str, Any]) -> str:
    base = ["id", "family", "condition", "seed", "duration_s", "reference_id", "state_status", "analysis_status",
            "completed_statistic_eligible", "excluded_reason", "run_dir", "walltime_s", "matched_yaw_residual_deg",
            "matched_forward_speed_residual_mm_s", "matched_path_speed_residual_mm_s", "prefix_sha256",
            "prefix_match", "prefix_scientific_issue", "malformed_jsonl_rows_skipped"]
    rows, cols = [], list(base)
    for r in report["cases"]:
        flat = {k: r.get(k) for k in base}; m = r["matched_static"]; sig = r["prefix_signature"]
        flat.update({"matched_yaw_residual_deg": m["yaw_change_residual_deg"],
                     "matched_forward_speed_residual_mm_s": m["forward_speed_residual_mm_s"],
                     "matched_path_speed_residual_mm_s": m["path_speed_residual_mm_s"],
                     "prefix_sha256": sig["sha256"], "prefix_match": m["prefix_match"],
                     "prefix_scientific_issue": m["scientific_issue"]})
        for name, v in r["controller"].items():
            for k, value in v.items(): flat[f"{name}_{k}"] = value
        for name, v in (r.get("trajectory") or {}).get("windows", {}).items():
            for k, value in v.items():
                if isinstance(value, (str, int, float)) or value is None: flat[f"{name}_{k}"] = value
        for k in flat:
            if k not in cols: cols.append(k)
        rows.append(flat)
    stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()

def write_summary(root: Path, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Atomically write progress.json, progress.md and results.csv below root."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True); report = summarize_queue(root, plan, state)
    _atomic(root / "progress.json", json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    _atomic(root / "progress.md", _markdown(report)); _atomic(root / "results.csv", _csv(report))
    return report
