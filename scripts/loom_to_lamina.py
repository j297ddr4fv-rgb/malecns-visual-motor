#!/usr/bin/env python3
"""Looming stimulus -> L2/L3 OFF drive, per the gap-2 recipe, for the MaleCNS LIF.

Why this exists: the naive route (drive photoreceptors R1-R6) cannot reach the
medulla -- photoreceptors are histaminergic/inhibitory onto a lamina that sits at
0 Hz, and T4/T5 additionally need per-synapse delays the connectome does not
carry.  The reference project's own gap-2 report scored three workarounds and
recommended this one, the only recipe with correct specificity:

    L   = column luminance 0..1
    Lf  = one-pole low-pass of L, tau_adapt = 50 ms for motion stimuli
    dec = clamp(Lf - L, 0, 1)                    # luminance decrement = OFF
    rate = 200 Hz * (dec + 0.30 * (1 - L))
    drive L2[c] and L3[c] (cholinergic) with Poisson events at that rate
    NEVER drive L1 (glutamatergic -> only suppresses Mi1/Tm3)

Result claimed by that report: DNp01 fires 28-53 Hz with first spike 72-146 ms,
silent for static and drifting stimuli; LC4 velocity-tuned (compressed ~5x);
LPLC2 silent (it needs T4/T5, which the medulla cannot deliver).

This script synthesises the loom analytically on the *measured* eye map (each
column's unit viewing direction from column_directions.csv), converts it to
per-column L2/L3 firing rates, and writes a stimulus table for
`lif_flyb.py --stim-table`.

Usage:
    python3 scripts/loom_to_lamina.py --out artifacts/loom_lamina.npz --preview artifacts/loom_preview.png
    python3 scripts/lif_flyb.py --ms 1000 --stim-table artifacts/loom_lamina.npz \
        --report "LC4;LPLC2;DNp01;DNp04;DNp02;DNp11;TTMn" --trace-ms 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CSR = Path("/data/cache/malecns-gpu/csr-flyb")
EYE_MAP = Path("/home/xvhang/Projects/fly-brain-minecraft/src/main/resources/"
               "assets/fruitfly/brain/column_directions.csv")
RETINA_L1, RETINA_L2, RETINA_L3 = 3, 4, 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preview", type=Path, default=None)
    ap.add_argument(
        "--ms", type=float, default=None,
        help="stimulus duration (default: 130 ms for --swat, otherwise 1000 ms)",
    )
    ap.add_argument(
        "--frame-ms", type=float, default=None,
        help="stimulus frame period (default: 1 ms for --swat, otherwise 5 ms)",
    )
    ap.add_argument("--loom-start", type=float, default=200.0)
    ap.add_argument("--loom-end", type=float, default=700.0)
    ap.add_argument("--radius0", type=float, default=5.0, help="deg at loom-start")
    ap.add_argument("--radius1", type=float, default=90.0, help="deg at loom-end")
    ap.add_argument("--az", type=float, default=60.0, help="disc centre azimuth, deg right of forward")
    ap.add_argument("--el", type=float, default=0.0)
    ap.add_argument("--r-max", type=float, default=200.0)
    ap.add_argument("--tau-adapt", type=float, default=50.0)
    ap.add_argument("--sustained", type=float, default=0.30)
    ap.add_argument("--swat", action="store_true",
                    help="physical approaching hand instead of a linear radius ramp: "
                         "theta(t) = atan((hand/2) / (distance - speed*t))")
    ap.add_argument(
        "--control",
        choices=("expansion", "sham", "static", "contraction"),
        default="expansion",
        help="visual counterfactual; it changes only the sensory movie and never emits a behavior label",
    )
    ap.add_argument("--hand-cm", type=float, default=9.0, help="palm width (cm)")
    ap.add_argument("--speed", type=float, default=10.0, help="approach speed (m/s)")
    ap.add_argument("--distance", type=float, default=1.0, help="initial distance (m)")
    args = ap.parse_args()

    # The ecological swat protocol is a millisecond-scale experiment.  Older
    # invocations documented ``--swat`` as a 130 x 1 ms table, but argparse's
    # generic loom defaults silently produced 200 x 5 ms instead.  Make the
    # protocol defaults explicit while preserving user overrides and the
    # historical non-swat loom defaults.
    if args.ms is None:
        args.ms = 130.0 if args.swat else 1000.0
    if args.frame_ms is None:
        args.frame_ms = 1.0 if args.swat else 5.0
    for name in ("ms", "frame_ms", "tau_adapt", "r_max"):
        if float(getattr(args, name)) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.swat and (args.hand_cm <= 0.0 or args.speed <= 0.0 or args.distance <= 0.0):
        raise SystemExit("--hand-cm, --speed and --distance must be positive for --swat")

    # ---- measured eye map: one unit viewing direction per hex column
    rows = list(csv.DictReader(EYE_MAP.open()))
    key = {(r["side"], int(r["hex1"]), int(r["hex2"])): np.array(
        [float(r["dir_front"]), float(r["dir_left"]), float(r["dir_up"])]) for r in rows}

    # ---- which neurons sit in which column, per lamina cell type
    ej = np.load(CSR / "retina.npy")
    sides = json.loads((CSR / "tables.json").read_text())["sides"]
    lam = {}
    for kind, name in ((RETINA_L2, "L2"), (RETINA_L3, "L3")):
        idx, dirs = [], []
        for s, h1, h2, neu, k in zip(ej["side"], ej["hex1"], ej["hex2"],
                                     ej["neuron"], ej["kind"]):
            if int(k) != kind or int(s) not in (1, 3):
                continue
            d = key.get((sides[int(s)], int(h1), int(h2)))
            if d is None:
                continue
            idx.append(int(neu))
            dirs.append(d)
        lam[name] = (np.array(idx, dtype=np.int32), np.array(dirs, dtype=np.float32))
    print(f"lamina columns: L2 {len(lam['L2'][0]):,}  L3 {len(lam['L3'][0]):,}  (L1 deliberately not driven)")

    # ---- analytic loom on the sphere
    c = np.array([np.cos(np.radians(args.az)), -np.sin(np.radians(args.az)), 0.0])  # right of forward
    c = c / np.linalg.norm(c)
    frames = int(round(args.ms / args.frame_ms))
    times = np.arange(frames) * args.frame_ms
    if args.swat:
        # physical approach: the angular radius is determined by palm width, speed
        # and distance, so the decision latency in ms is comparable to a real fly's
        R = (args.hand_cm / 2.0) / 100.0                       # metres
        dist = args.distance - args.speed * times / 1000.0
        dist = np.maximum(dist, 1e-3)
        radius = np.degrees(np.arctan(R / dist))
        impact_ms = args.distance / args.speed * 1000.0
        print(f"swat: {args.hand_cm:g} cm palm at {args.speed:g} m/s from {args.distance:.2f} m "
              f"-> impact at t = {impact_ms:.0f} ms; angular radius "
              f"{radius[0]:.2f} deg -> {radius[-1]:.1f} deg")
    else:
        frac = np.clip((times - args.loom_start) / max(1e-6, args.loom_end - args.loom_start), 0, 1)
        radius = args.radius0 + (args.radius1 - args.radius0) * frac
    if args.control == "sham":
        radius = np.zeros_like(radius)
    elif args.control == "static":
        radius = np.full_like(radius, radius[0])
    elif args.control == "contraction":
        radius = radius[::-1].copy()

    rates = {}
    lum_trace = {}
    for name, (idx, dirs) in lam.items():
        cosang = dirs @ c
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))      # angle to the disc centre
        L = np.ones((frames, idx.size), dtype=np.float32)         # bright background
        for f in range(frames):
            L[f, ang < radius[f]] = 0.0                           # dark disc
        Lf = np.empty_like(L); acc = L[0].copy()
        a = args.frame_ms / args.tau_adapt
        for f in range(frames):
            acc = acc + a * (L[f] - acc)
            Lf[f] = acc
        dec = np.clip(Lf - L, 0.0, 1.0)
        rates[name] = (args.r_max * (dec + args.sustained * (1.0 - L))).astype(np.float32)
        lum_trace[name] = L

    neurons = np.concatenate([lam["L2"][0], lam["L3"][0]])
    table = np.concatenate([rates["L2"], rates["L3"]], axis=1)
    print(f"stimulus: {frames} frames x {table.shape[1]:,} lamina cells, "
          f"rate range [{table.min():.1f}, {table.max():.1f}] Hz, "
          f"disc radius {radius[0]:.2f}->{radius[-1]:.2f} deg at az {args.az:.0f} deg")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stimulus_meta = {
        "kind": "physical_swat" if args.swat else "linear_loom",
        "visual_control": args.control,
        "recipe": "gap-2 recipe (i): L2/L3 OFF drive, L1 not driven",
        "duration_ms": float(args.ms),
        "frame_ms": float(args.frame_ms),
        "frames": int(frames),
        "r_max": args.r_max,
        "tau_adapt": args.tau_adapt,
        "sustained": args.sustained,
        "az_deg": args.az,
        "el_deg": args.el,
        "radius_deg": [float(radius[0]), float(radius[-1])],
    }
    if args.swat:
        stimulus_meta["physical_swat"] = {
            "hand_width_cm": float(args.hand_cm),
            "speed_mps": float(args.speed),
            "initial_distance_m": float(args.distance),
            "impact_ms": float(impact_ms),
        }
    else:
        stimulus_meta["loom_ms"] = [float(args.loom_start), float(args.loom_end)]
    np.savez_compressed(
        args.out,
        rates=table,
        neurons=neurons,
        frame_ms=np.float64(args.frame_ms),
        meta=json.dumps(stimulus_meta, sort_keys=True),
    )
    print(f"-> {args.out}")

    if args.preview:
        from PIL import Image, ImageDraw
        W, H = 900, 300
        img = Image.new("RGB", (W, H), (12, 12, 16))
        d = ImageDraw.Draw(img)
        for j, name in enumerate(("L2", "L3")):
            v = rates[name].max(axis=1)
            pts = [(20 + int(k / frames * (W - 40)), H - 40 - int(x / max(1.0, v.max()) * (H - 80)))
                   for k, x in enumerate(v)]
            d.line(pts, fill=(90, 220, 255) if name == "L2" else (255, 190, 90), width=2)
            d.text((20, 12 + 16 * j), f"{name} OFF drive (max over columns), peak {v.max():.0f} Hz",
                   fill=(90, 220, 255) if name == "L2" else (255, 190, 90))
        stimulus_label = (
            f"swat impact {impact_ms:.0f} ms"
            if args.swat else f"loom {args.loom_start:.0f}-{args.loom_end:.0f} ms"
        )
        d.text((20, H - 24), f"{stimulus_label}, "
                             f"radius {radius[0]:.1f}->{radius[-1]:.1f} deg, az {args.az:.0f} deg",
               fill=(170, 170, 180))
        img.save(args.preview)
        print(f"preview -> {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
