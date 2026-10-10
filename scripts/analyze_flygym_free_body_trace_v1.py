#!/usr/bin/env python3
"""Read-only measured body diagnostics; never runs physics or neural inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open runs of True, including runs touching an array boundary."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def analyze(run: Path, start: float, end: float) -> dict:
    trace = run / 'trajectory.npz'
    order_path = run / 'array_order.json'
    order = json.loads(order_path.read_text())
    thorax = order['body_names'].index('c_thorax')
    with np.load(trace, allow_pickle=False) as a:
        t = a['time_s'].astype(np.float64)
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Body time must be finite and strictly increasing')
        dt = float(np.median(np.diff(t)))
        if not np.allclose(np.diff(t), dt, rtol=0, atol=1e-10):
            raise ValueError('Nonuniform body samples')
        if not (t[0] <= start < end <= t[-1] + 1e-8):
            raise ValueError('Requested window is outside captured time')
        mask = (t >= start - 1e-8) & (t <= end + 1e-8)
        yaw = np.unwrap(a['thorax_heading_yaw_rad'].astype(np.float64))
        if not np.isfinite(yaw).all():
            raise ValueError('Nonfinite body yaw')
        yaw_delta = float(np.interp(end, t, yaw) - np.interp(start, t, yaw))
        body_weight = float(a['body_mass_native'].sum()) * 9810.0
        threshold = .01 * body_weight
        loaded = a['foot_contact_fz_up_native'][mask] >= threshold
        support = loaded.sum(axis=1)
        z = a['tarsus5_z_mm'][mask]
        event_rows = []
        for leg, name in enumerate(LEGS):
            stance = loaded[:, leg]
            if not stance.any():
                event_rows.append(dict(leg=name, complete_events=0, stance_height_mm=None, events=[]))
                continue
            baseline = float(np.median(z[stance, leg]))
            events = []
            for lo, hi in segments(~stance):
                # Require continuous loaded samples on both sides of the swing.
                before = lo
                while before > 0 and stance[before - 1]:
                    before -= 1
                after = hi
                while after < len(stance) and stance[after]:
                    after += 1
                clearance = float(z[lo:hi, leg].max() - baseline)
                if (lo - before) * dt + 1e-10 >= .002 and (after - hi) * dt + 1e-10 >= .002 and (hi - lo) * dt + 1e-10 >= .002 and clearance >= .05:
                    events.append(dict(unload_start_s=float(t[mask][lo]), touchdown_s=float(t[mask][hi]), unloaded_s=(hi-lo)*dt, peak_rise_mm=clearance))
            event_rows.append(dict(leg=name, complete_events=len(events), stance_height_mm=baseline, events=events))
        q = a['body_xquat_wxyz'][mask, thorax].astype(np.float64)
        q /= np.linalg.norm(q, axis=1)[:, None]
        w, x, y, zz = q.T
        qyaw = np.arctan2(2*(w*zz+x*y), 1-2*(y*y+zz*zz))
        pitch = np.arcsin(np.clip(2*(w*y-zz*x), -1, 1))
        roll = np.arctan2(2*(w*x+y*zz), 1-2*(x*x+y*y))
        yaw_error = np.angle(np.exp(1j*(qyaw - yaw[mask])))
        authority = {}
        for name in ('root_qpos_write_count', 'root_qvel_write_count', 'root_external_wrench_native', 'body_xfrc_applied_native', 'qfrc_applied_native'):
            values = a[name]
            authority[name] = float(np.max(np.abs(values)))
        return dict(
            evidence_class='digital_simulation', biological_validation='NOT_VALIDATED',
            source_files={str(trace): digest(trace), str(order_path): digest(order_path)},
            analyzer_sha256=digest(Path(__file__).resolve()), run=str(run),
            window_s=[start, end], native_samples=len(t), native_dt_s=dt,
            yaw_delta_deg=math.degrees(yaw_delta), mean_yaw_rate_deg_s=math.degrees(yaw_delta)/(end-start),
            support=dict(body_weight_native=body_weight, per_leg_threshold_fraction=.01, per_leg_threshold_native=threshold, minimum_loaded_legs=int(support.min()), fraction_at_least_two=float((support >= 2).mean())),
            event_definition=dict(stance_reference='per-leg median height in loaded samples within selected window', rise_mm=.05, unloaded_min_s=.002, loaded_before_and_after_min_s=.002, complete_events_only=True),
            per_leg_events=event_rows,
            posture=dict(max_abs_roll_deg=float(np.degrees(np.abs(roll)).max()), max_abs_pitch_deg=float(np.degrees(np.abs(pitch)).max()), heading_xy_norm_min=float(np.linalg.norm(a['thorax_heading_xy'][mask], axis=1).min()), quaternion_heading_max_error_deg=float(np.degrees(np.abs(yaw_error)).max()), thorax_z_min_mm=float(a['body_xpos_mm'][mask, thorax, 2].min()), collision_clearance_min_mm=float(a['thorax_collision_clearance_min_mm'][mask].min()), nonleg_contact_samples=int(np.count_nonzero(a['nonleg_ground_contact'][mask])), thorax_contact_samples=int(np.count_nonzero(a['thorax_ground_contact'][mask]))),
            authority_max_abs_over_entire_capture=authority,
            interpretation='Measured foot/body diagnostics do not alone establish biological gait or visual following; no stimulus label, neural inference, or direction acceptance is used.',
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--start-s', type=float, default=1.)
    p.add_argument('--end-s', type=float, default=4.)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = dict(schema='flygym-free-body-trace-analysis-v1', cases=[analyze(r.resolve(), args.start_s, args.end_s) for r in args.run])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(cases=len(result['cases']), output=str(args.out))))


if __name__ == '__main__':
    main()
