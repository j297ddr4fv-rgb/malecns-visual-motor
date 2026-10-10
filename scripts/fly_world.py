#!/usr/bin/env python3
"""Fly the MaleCNS connectome in an infinitely extending cubic-lattice world.

WHAT IS REAL HERE AND WHAT IS GLUE -- read this before quoting any result.

  real   the world        an infinite 3-D cubic wireframe lattice, rendered by
                          ray-splatting its edges onto the fly's celestial
                          sphere.  It never ends: the segment set is regenerated
                          around the fly every frame, so it can fly forever.
  real   the brain        the full MaleCNS v1.0 LIF, 176,422 neurons, stepped
                          inside the control loop at dt = 0.5 ms.  The path
                          lobula plate -> descending neuron -> wing motor neuron
                          is entirely the connectome's own wiring.
  GLUE   the sensor       the model's medulla is silent (T4/T5/Mi1 = 0 Hz, see
                          EXPERIMENT_LOG.md E12), so image motion cannot reach
                          the lobula plate on its own.  This script computes the
                          motion explicitly with a Reichardt elementary-motion
                          detector and injects the result into HS/VS -- exactly
                          the substitution skulitom/haltere makes when it wires
                          a drone's gyro into the fly's haltere afferents.
  GLUE   the body         a 3-DOF point-mass/attitude integrator, not
                          aerodynamics.

So this is not "the fly flies".  It is "the fly's wing-control circuit flies a
body in a lattice", with the sensor and the body supplied from outside.  That
is the honest version of the claim, and it is the same claim haltere makes.

Modes
  --mode reflex   open loop: sweep a known optic-flow value, measure the wing
                  motor neurons' differential response.  Answers "does this
                  connectome contain an optomotor reflex to the wings?"
  --mode fly      closed loop: render, detect motion, inject, step the brain,
                  read the wings, integrate the body, repeat.

Usage
  python3 scripts/fly_world.py --mode reflex --allow-shmem-gib 4
  python3 scripts/fly_world.py --mode fly --frames 240 --allow-shmem-gib 4 \
      --out artifacts/lattice-flight
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np

CSR = Path("/data/cache/malecns-gpu/csr-flyb")
EYE_MAP = Path("/home/xvhang/Projects/fly-brain-minecraft/src/main/resources/"
               "assets/fruitfly/brain/column_directions.csv")

# ---------------------------------------------------------------- eye geometry


def load_photoreceptors(want_kinds=(0,)):
    """R1-R6 (kind 0) with their measured head-frame viewing directions."""
    ej = np.load(CSR / "retina.npy")
    sides = json.loads((CSR / "tables.json").read_text())["sides"]
    rows = list(__import__("csv").DictReader(EYE_MAP.open()))
    eye = {(r["side"], int(r["hex1"]), int(r["hex2"])):
           (float(r["dir_front"]), float(r["dir_left"]), float(r["dir_up"])) for r in rows}
    photo = np.isin(ej["kind"], list(want_kinds))
    rec = ej[photo]
    keys = [(sides[int(s)], int(h1), int(h2)) for s, h1, h2 in zip(rec["side"], rec["hex1"], rec["hex2"])]
    d = np.array([eye.get(k, (1.0, 0.0, 0.0)) for k in keys], dtype=np.float32)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))          # + = toward the fly's left
    el = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))    # + = up
    side = np.array([k[0] for k in keys])
    return rec["neuron"].astype(np.int64), az.astype(np.float32), el.astype(np.float32), side


# ------------------------------------------------------------------- the world


def cell_samples(center, radius, spacing, jitter=0.7, per_face=4):
    """Sample points over the FACES of every cell within `radius`.

    A bare wireframe leaves most of the visual field black, so at the fly's
    ~4.5 deg ommatidial resolution each column integrates almost nothing and
    flyvis loses direction selectivity (E20).  Filling the cells with a fixed
    per-cell brightness gives a dense, aperiodic, band-limited world -- which is
    also what a real lattice of frame members looks like from the inside.
    """
    lo = np.floor((np.asarray(center) - radius) / spacing).astype(int)
    hi = np.ceil((np.asarray(center) + radius) / spacing).astype(int)
    g = [np.arange(lo[k], hi[k]) * spacing for k in range(3)]
    I, J, K = np.meshgrid(g[0], g[1], g[2], indexing="ij")
    cen = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    keep = np.linalg.norm(cen + spacing / 2.0 - np.asarray(center), axis=1) <= radius
    cen = cen[keep] + spacing / 2.0
    u = (np.arange(per_face) + 0.5) / per_face - 0.5
    pts = []
    for ax in range(3):                       # the two faces normal to each axis
        for sgn in (-0.5, 0.5):
            o = [np.meshgrid(*(u, u), indexing="ij") for _ in range(1)][0]
            A, B = o[0].ravel(), o[1].ravel()
            q = np.tile(cen, (per_face * per_face, 1))
            others = [k for k in range(3) if k != ax]
            q[:, others[0]] += np.tile(A, len(cen))
            q[:, others[1]] += np.tile(B, len(cen))
            q[:, ax] += sgn * spacing
            pts.append(q)
    P = np.concatenate(pts)
    cid = np.repeat(np.arange(len(cen)), 6 * per_face * per_face)
    h = (((cen[:, 0] / spacing).astype(np.int64) * 73856093)
         ^ ((cen[:, 1] / spacing).astype(np.int64) * 19349663)
         ^ ((cen[:, 2] / spacing).astype(np.int64) * 83492791)) % 1000003
    hf = h.astype(np.float64) / 1000003.0
    inten = (1.0 + jitter * (2.0 * hf - 1.0)).astype(np.float32)[cid]
    return P - center, inten


def _edge_key(p0, p1, spacing):
    """Deterministic per-edge index.  Used to give every edge its own fixed
    brightness so the world is aperiodic -- a *regular* lattice is an aliased
    stimulus for any correlation-type motion detector (E20)."""
    m = 0.5 * (p0 + p1) / spacing
    ax = np.argmax(np.abs(p1 - p0), axis=1)
    k = np.rint(m).astype(np.int64)
    return (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791) ^ (ax * 2654435761)


def lattice_segments(center, radius, spacing, jitter=0.0):
    """All unit-cell edges of the infinite cubic lattice within `radius` of
    `center`.  Regenerating this every frame is what makes the world endless.

    Returns (p0, p1, inten) with p0/p1 in head-frame-ready world coordinates
    relative to `center`, and `inten` a per-edge brightness in [1-jitter, 1+jitter].
    """
    lo = np.floor((np.asarray(center) - radius) / spacing).astype(int)
    hi = np.ceil((np.asarray(center) + radius) / spacing).astype(int)
    grids = [np.arange(lo[k], hi[k] + 1) * spacing for k in range(3)]
    segs = []
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        A, B = np.meshgrid(grids[others[0]], grids[others[1]], indexing="ij")
        A, B = A.ravel(), B.ravel()
        C0, C1 = grids[axis][:-1], grids[axis][1:]
        nA, nB, nC = A.size, 1, C0.size
        # build (nA*nC) segments along `axis`
        A_ = np.repeat(A, nC)
        B_ = np.repeat(B, nC)
        C0_ = np.tile(C0, nA)
        C1_ = np.tile(C1, nA)
        p0 = np.empty((A_.size, 3), np.float64)
        p1 = np.empty((A_.size, 3), np.float64)
        p0[:, others[0]] = A_
        p1[:, others[0]] = A_
        p0[:, others[1]] = B_
        p1[:, others[1]] = B_
        p0[:, axis] = C0_
        p1[:, axis] = C1_
        segs.append((p0, p1))
    p0 = np.concatenate([s[0] for s in segs])
    p1 = np.concatenate([s[1] for s in segs])
    mid = 0.5 * (p0 + p1)
    keep = np.linalg.norm(mid - np.asarray(center), axis=1) <= radius
    p0, p1 = p0[keep], p1[keep]
    if jitter > 0:
        h = (_edge_key(p0, p1, spacing) % 1000003).astype(np.float64) / 1000003.0
        inten = (1.0 + jitter * (2.0 * h - 1.0)).astype(np.float32)
    else:
        inten = np.ones(len(p0), np.float32)
    return p0 - center, p1 - center, inten


def render_equirect(p0, p1, az_bins, el_bins, near=0.25, fog=2.6,
                    do_blur=True, density=1.6, max_samples=192, inten=None,
                    pts=None, pts_inten=None):
    """Splat lattice edges onto an equirectangular luminance map in head frame.

    Head frame: +x front, +y left, +z up.  az = atan2(y, x), el = asin(z/|r|).
    Each segment is resampled *in angle*, with a per-segment sample count set by
    its angular length, so a near edge (which spans many bins) stays a solid
    line instead of a dotted one.

    Intensity falls off as exp(-d/fog).  That is what makes a finite segment set
    read as an unbounded lattice: near edges are bright, far ones fade out, and
    the gradient itself says "it keeps going".  Normalisation is FIXED so that
    frame-to-frame brightness differences stay physically meaningful.
    """
    def to_ae(p):
        r = np.maximum(np.linalg.norm(p, axis=1), 1e-6)
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / r, -1, 1)))
        return az, el, r

    az0, el0, r0 = to_ae(p0)
    az1, el1, r1 = to_ae(p1)
    daz = az1 - az0
    daz = np.where(daz > 180.0, daz - 360.0, np.where(daz < -180.0, daz + 360.0, daz))
    del_ = el1 - el0

    deg_per_x = 360.0 / az_bins
    deg_per_y = 180.0 / el_bins
    span = np.maximum(np.abs(daz) / deg_per_x, np.abs(del_) / deg_per_y)
    need = np.clip(np.ceil(span * density).astype(np.int64), 2, max_samples)

    idx = np.repeat(np.arange(need.size), need)
    starts = np.concatenate([[0], np.cumsum(need)[:-1]])
    k = np.arange(idx.size) - starts[idx]
    t = (k / need[idx]).astype(np.float32)

    AZ = (az0[idx] + daz[idx] * t)
    EL = (el0[idx] + del_[idx] * t)
    R = (r0[idx] + (r1 - r0)[idx] * t)

    good = R > near
    AZ, EL, R, seg = AZ[good], EL[good], R[good], idx[good]
    xb = np.floor((AZ + 180.0) / 360.0 * az_bins).astype(np.int64)
    yb = np.floor((90.0 - EL) / 180.0 * el_bins).astype(np.int64)
    ok = (xb >= 0) & (xb < az_bins) & (yb >= 0) & (yb < el_bins)
    xb, yb, R = xb[ok], yb[ok], R[ok]
    w = np.exp(-R / fog).astype(np.float32)
    if inten is not None:
        w = w * inten[seg[ok]]
    acc = np.zeros(el_bins * az_bins, dtype=np.float32)
    np.add.at(acc, yb * az_bins + xb, w)
    if pts is not None:
        r = np.maximum(np.linalg.norm(pts, axis=1), 1e-6)
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        el = np.degrees(np.arcsin(np.clip(pts[:, 2] / r, -1, 1)))
        qx = np.floor((az + 180.0) / 360.0 * az_bins).astype(np.int64)
        qy = np.floor((90.0 - el) / 180.0 * el_bins).astype(np.int64)
        qk = (qx >= 0) & (qx < az_bins) & (qy >= 0) & (qy < el_bins) & (r > near)
        np.add.at(acc, qy[qk] * az_bins + qx[qk],
                  (np.exp(-r[qk] / fog) * pts_inten[qk]).astype(np.float32))
    acc = acc.reshape(el_bins, az_bins)
    if do_blur:                                # a splat should read as a line
        s = acc.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            s += np.roll(np.roll(acc, dy, 0), dx, 1)
        acc = s
    return acc


def equirect_to_persp(lum, fov_h_deg=110.0, fov_v_deg=80.0, W=640, H=420):
    """Resample the equirect map onto a forward-looking pinhole camera, so the
    video panel shows what a wide-angle camera on the fly's head would record."""
    el_bins, az_bins = lum.shape
    xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    ys = (np.arange(H) + 0.5) / H * 2.0 - 1.0
    X, Y = np.meshgrid(xs, ys)
    th = np.tan(np.radians(fov_h_deg) / 2.0)
    tv = np.tan(np.radians(fov_v_deg) / 2.0)
    az = np.degrees(np.arctan(X * th))
    el = np.degrees(np.arctan(-Y * tv / np.sqrt(1.0 + (X * th) ** 2)))
    xb = (np.floor((az + 180.0) / 360.0 * az_bins).astype(np.int64)) % az_bins
    yb = np.clip(np.floor((90.0 - el) / 180.0 * el_bins).astype(np.int64), 0, el_bins - 1)
    return lum[yb, xb]


# ------------------------------------------------------- motion (the glue part)


def emd(prev, cur, axis, shift=2):
    """Reichardt elementary-motion detector.

    Positive = features moved toward increasing azimuth (axis=1) / increasing
    elevation (axis=0).  The leading minus sign is NOT cosmetic: without it the
    detector reports the opposite direction, which silently turns the flight
    loop into positive feedback.  Verified against a synthetically shifted line
    (see docs/EXPERIMENT_LOG.md, E18).
    """
    a = np.roll(cur, -shift, axis=axis)
    b = np.roll(prev, -shift, axis=axis)
    return -(cur * b - prev * a)


# ------------------------------------------------------------------- the brain

def build_brain(gain, kc_gain, dev, torch):
    meta = json.loads((CSR / "meta.json").read_text())
    tables = json.loads((CSR / "tables.json").read_text())
    ann_npz = np.load(CSR / "annotations.npz")
    ann = {k: ann_npz[k] for k in ann_npz.files}
    ids = np.load(CSR / "ids.npy")
    offsets_np = np.load(CSR / "offsets.npy")
    target_np = np.load(CSR / "target.npy")
    weight_np = np.load(CSR / "weight.npy")
    sign_np = np.load(CSR / "sign.npy")
    n = int(ids.size)

    import importlib.util
    spec = importlib.util.spec_from_file_location("lif_flyb", Path(__file__).with_name("lif_flyb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                      # safe: main() is guarded
    pops = mod.Populations(tables, ann)
    pops.ids = ids

    post_gain = np.ones(n, dtype=np.float32)
    post_gain[pops.resolve("class:Kenyon_Cell")] = kc_gain
    src_of_edge = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets_np))
    w_final = (sign_np[src_of_edge].astype(np.float32) * (0.275 * gain)
               * weight_np.astype(np.float32) * post_gain[target_np])
    del src_of_edge

    st = {
        "torch": torch, "n": n, "dev": dev,
        "offsets": torch.from_numpy(offsets_np).to(dev),
        "target": torch.from_numpy(target_np.astype(np.int64)).to(dev),
        "wf": torch.from_numpy(w_final).to(dev),
        "pops": pops,
        "dt": 0.5, "v_rest": -52.0, "v_thr": -45.0, "v_reset": -52.0,
        "tau_m": 20.0, "tau_s": 5.0,
    }
    del w_final
    return st


def init_state(st, torch):
    n, dev = st["n"], st["dev"]
    st["decay"] = float(np.exp(-st["dt"] / st["tau_m"]))
    st["syn_decay"] = float(np.exp(-st["dt"] / st["tau_s"]))
    st["syn_coupling"] = ((st["tau_s"] / (st["tau_m"] - st["tau_s"]))
                          * (np.exp(-st["dt"] / st["tau_m"]) - np.exp(-st["dt"] / st["tau_s"])))
    st["delay_steps"] = max(1, int(round(1.8 / st["dt"])))
    st["ref_steps"] = max(1, int(round(2.2 / st["dt"])))
    st["slots"] = st["delay_steps"] + 1
    st["v"] = torch.full((n,), st["v_rest"], device=dev)
    st["g"] = torch.zeros(n, device=dev)
    st["ref_until"] = torch.zeros(n, dtype=torch.int32, device=dev)
    st["ring"] = torch.zeros((st["slots"], n), device=dev)
    st["inject"] = torch.zeros(n, device=dev)
    st["tick"] = torch.zeros(n, dtype=torch.int32, device=dev)
    st["step"] = 0
    st["gen"] = torch.Generator(device=dev)
    st["gen"].manual_seed(20260903)


def brain_step(st, torch):
    """One dt step; identical update order to the validated lif_flyb.py port."""
    dev = st["dev"]
    read = st["step"] % st["slots"]
    write = (st["step"] + st["delay_steps"]) % st["slots"]
    v, g, ref = st["v"], st["g"], st["ref_until"]
    inp = st["ring"][read]
    g.add_(inp)
    inp.zero_()
    refr = ref > st["step"]
    v_new = (st["v_rest"] + (v - st["v_rest"]) * st["decay"]
             + g * st["syn_coupling"] + st["inject"])
    v.copy_(torch.where(refr, torch.full_like(v, st["v_reset"]), v_new))
    g.copy_(torch.where(refr, g, g * st["syn_decay"]))
    spiked = (~refr) & (v >= st["v_thr"])
    v.masked_fill_(spiked, st["v_reset"])
    g.masked_fill_(spiked, 0.0)
    ref.masked_fill_(spiked, st["step"] + st["ref_steps"])
    idx = spiked.nonzero().flatten()
    if idx.numel():
        starts = st["offsets"].index_select(0, idx)
        lens = st["offsets"].index_select(0, idx + 1) - starts
        cum = torch.cumsum(lens, 0)
        total = int(cum[-1].item())
        if total:
            pos = torch.arange(total, device=dev)
            grp = torch.searchsorted(cum, pos, right=True)
            syn = starts.index_select(0, grp) + pos - (cum.index_select(0, grp)
                                                       - lens.index_select(0, grp))
            st["ring"][write].index_add_(0, st["target"].index_select(0, syn),
                                         st["wf"].index_select(0, syn))
    st["tick"].add_(spiked.to(torch.int32))
    st["step"] += 1
    return int(idx.numel())


def rates_over(st, pop_idx, steps):
    """Spikes emitted by `pop_idx` during the last `steps` felt like a rate."""
    return float(st["tick"][pop_idx].sum().item()) / (pop_idx.size * steps * st["dt"] / 1000.0)


# ------------------------------------------------------------------- the body


class Body:
    """3-DOF body: yaw + speed, with fly-scale inertia.

    `damp` is the yaw damping rate 1/tau.  It is THE parameter that decides what
    the trajectory looks like: a Drosophila is damped wingbeat-synchronously
    (tau ~= 10-20 ms), so a loop that closes through a 30-50 ms neural delay is
    running *faster than the body can follow*, which is the standard hypothesis
    for why fly flight comes out as a saccade train rather than a smooth arc.

    A room-scale body (damp = 2.2 /s, tau = 455 ms) is 30x too sluggish and can
    only ever produce the smooth circle of EXPERIMENT_LOG.md E17.
    """

    def __init__(self, speed=0.3, omega_max=40.0):
        self.p = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0          # rad, 0 = +x
        self.pitch = 0.0
        self.omega = 0.0        # yaw rate
        self.q = 0.0
        self.speed = speed
        self.omega_max = omega_max
        self.traj = []

    def step(self, dt_s, u_yaw, u_pitch, k_yaw, k_pitch, damp=2.2, bias=0.0,
             u_thrust=0.0, k_thrust=0.0, tau_v=0.05, v0=1.2, pitch_hold=0.0):
        u = float(np.clip(u_yaw, -1.0, 1.0))
        # k_yaw/damp is the commanded yaw RATE at full deflection: interpretable,
        # and it makes the loop gain independent of the damping parameter.
        target = (k_yaw / damp) * u + bias / damp
        self.omega += (target - self.omega) * damp * dt_s
        self.omega = float(np.clip(self.omega, -self.omega_max, self.omega_max))
        self.q += (k_pitch * u_pitch - damp * self.q) * dt_s
        vt = v0 * (1.0 + k_thrust * float(np.clip(u_thrust, -1.0, 1.0)))
        self.speed += (max(0.0, vt) - self.speed) / max(tau_v, 1e-4) * dt_s
        self.yaw += self.omega * dt_s
        self.pitch = float(np.clip(self.pitch + self.q * dt_s, -0.6, 0.6))
        v = self.speed * np.array([np.cos(self.pitch) * np.cos(self.yaw),
                                   np.cos(self.pitch) * np.sin(self.yaw),
                                   np.sin(self.pitch)])
        self.p += v * dt_s
        self.traj.append(self.p.copy())


# ------------------------------------------------------------------ rendering


class Video:
    def __init__(self, path: Path, w, h, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = w, h
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()



def label(canvas, pw, vh, fi, args):
    """Burn the panel captions in.  One PIL pass per frame, ~1 ms."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((8, 6), "what the fly sees  (pinhole, 110 deg H)", fill=(235, 235, 245))
    cm = args.spacing * 100.0
    cell = f"{cm:.0f} cm" if cm >= 1 else f"{cm*10:.0f} mm"
    d.text((pw + 8, 6), f"top-down: infinite cubic lattice, {cell} cells", fill=(235, 235, 245))
    d.text((8, vh - 108),
           "white = heading (deg)   red = yaw rate   green/blue = wm-L / wm-R   yellow = optic flow",
           fill=(200, 200, 215))
    mode = "OPEN LOOP (gain 0)" if args.readout_gain == 0 else f"DECODER IN THE LOOP (gain {args.readout_gain:g})"
    d.text((8, 24), mode, fill=(255, 210, 120))
    d.text((pw + 8, 24), f"frame {fi}  ({fi * args.frame_ms / 1000.0:.2f} s neural)  "
                       f"cell {args.spacing*100:.0f} cm  {args.speed:.2f} m/s", fill=(200, 200, 215))
    return np.asarray(im)


def upscale(img, w, h):
    """Nearest-neighbour upscale of a float [0,1] or uint8 2-D image."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    yi = (np.arange(h) * a.shape[0] // h).clip(0, a.shape[0] - 1)
    xi = (np.arange(w) * a.shape[1] // w).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]


def topdown(center_xy, traj, half, size, spacing):
    """Top-down (X-Y) view of the lattice around the fly, plus its path."""
    img = np.full((size, size), 18, np.uint8)
    lo = np.floor((center_xy[0] - half) / spacing).astype(int)
    hi = np.ceil((center_xy[0] + half) / spacing).astype(int)
    px = np.arange(size)
    world = center_xy[0] - half + px / (size - 1) * 2 * half
    for i in range(lo, hi + 1):
        x = i * spacing
        c = int((x - (center_xy[0] - half)) / (2 * half) * (size - 1))
        if 0 <= c < size:
            img[:, c] = 112                     # vertical lattice lines (Y = const)
    lo = np.floor((center_xy[1] - half) / spacing).astype(int)
    hi = np.ceil((center_xy[1] + half) / spacing).astype(int)
    world_y = center_xy[1] - half + px / (size - 1) * 2 * half
    for j in range(lo, hi + 1):
        y = j * spacing
        r = int((y - (center_xy[1] - half)) / (2 * half) * (size - 1))
        if 0 <= r < size:
            img[r, :] = 112
    for p in traj[-4000:]:
        c = int((p[0] - (center_xy[0] - half)) / (2 * half) * (size - 1))
        r = int((p[1] - (center_xy[1] - half)) / (2 * half) * (size - 1))
        if 0 <= c < size and 0 <= r < size:
            img[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = 235
    c = int((center_xy[0] - (center_xy[0] - half)) / (2 * half) * (size - 1))
    r = int((center_xy[1] - (center_xy[1] - half)) / (2 * half) * (size - 1))
    img[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = 255
    return np.stack([img] * 3, -1)


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["reflex", "probe", "fly"], default="fly")
    ap.add_argument("--frames", type=int, default=240, help="control frames")
    ap.add_argument("--vision", choices=["emd", "flyvis"], default="emd",
                    help="emd = hand-written Reichardt detector into HS/VS (E17/E18);\n                          flyvis = fitted optic-lobe model driving our own T4/T5 (E20)")
    ap.add_argument("--inject-target", choices=["hsvs", "t4t5"], default=None,
                    help="which population the drive is written into")
    ap.add_argument("--flyvis-view", default="flow/0000/000")
    ap.add_argument("--fv-base", type=float, default=0.40, help="mV/ms at zero flyvis drive")
    ap.add_argument("--fv-gain", type=float, default=0.40, help="mV/ms per unit flyvis drive")
    ap.add_argument("--fv-grey", type=float, default=0.16,
                    help="grey level the optic lobe is settled on before take-off")
    ap.add_argument("--preset", choices=["room", "fly"], default="room",
                    help="room = E17 geometry; fly = 4 cm cells, 0.3 m/s, 17 ms yaw damping")
    ap.add_argument("--frame-ms", type=float, default=None, help="neural ms per control frame")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=0.65, help="synaptic gain (E15)")
    ap.add_argument("--kc-gain", type=float, default=0.25)
    ap.add_argument("--spacing", type=float, default=None, help="lattice cell size, m")
    ap.add_argument("--radius", type=float, default=None, help="render radius, m")
    ap.add_argument("--az-bins", type=int, default=720)
    ap.add_argument("--el-bins", type=int, default=360)
    ap.add_argument("--base-inject", type=float, default=0.45, help="mV/ms tonic on HS/VS")
    ap.add_argument("--flow-gain", type=float, default=0.9, help="mV/ms per unit of EMD signal")
    ap.add_argument("--flow-scale", type=float, default=None, help="normalised EMD -> injection units")
    ap.add_argument("--inject-clip", type=float, default=1.2,
                    help="hard limit on the commanded push-pull, like HS saturation")
    ap.add_argument("--readout-gain", type=float, default=1.0,
                    help="amplifier on the wing-motor readout; THIS is the fitted/trained term")
    ap.add_argument("--power-ref", type=float, default=25.0, help="wing rate treated as level flight")
    ap.add_argument("--k-yaw", type=float, default=None, help="body yaw gain, rad/s^2 per unit")
    ap.add_argument("--k-thrust", type=float, default=None, help="speed modulation depth")
    ap.add_argument("--damp", type=float, default=None, help="yaw damping rate 1/tau, per s")
    ap.add_argument("--tau-v", type=float, default=0.05, help="speed time constant, s")
    ap.add_argument("--tau-flow", type=float, default=0.020, help="optic-flow lowpass, s")
    ap.add_argument("--tau-motor", type=float, default=0.015, help="wing-readout lowpass, s")
    ap.add_argument("--omega-per-u", type=float, default=None,
                    help="commanded yaw rate at full readout deflection, rad/s")
    ap.add_argument("--emd-shift", type=int, default=None, help="EMD correlation offset, bins")
    ap.add_argument("--body-mm", type=float, default=2.5, help="body length, mm (for reporting)")
    ap.add_argument("--saccade-dps", type=float, default=300.0,
                    help="yaw rate above which a sample counts as a saccade")
    ap.add_argument("--k-pitch", type=float, default=1.0)
    ap.add_argument("--yaw-sign", type=float, default=None, help="sign of the yaw readout")
    ap.add_argument("--pitch-sign", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=None, help="forward speed, m/s")
    ap.add_argument("--gust-at", type=float, default=150.0, help="frame of the yaw impulse")
    ap.add_argument("--gust", type=float, default=2.5, help="impulse, rad/s")
    ap.add_argument("--bias", type=float, default=0.0,
                    help="constant yaw torque, rad/s^2; open loop settles at bias/damp")
    ap.add_argument("--fog", type=float, default=None, help="lattice depth cue, m")
    ap.add_argument("--solid", action="store_true", default=None,
                    help="fill each cell with a fixed brightness (needed for flyvis)")
    ap.add_argument("--edge-jitter", type=float, default=0.7,
                    help="per-edge brightness spread; breaks the lattice periodicity")
    ap.add_argument("--optics-deg", type=float, default=2.5,
                    help="ommatidial acceptance angle used to pre-blur the flyvis input")
    ap.add_argument("--exposure", type=float, default=1.3, help="display gain for the video panel")
    ap.add_argument("--vid-w", type=int, default=1280)
    ap.add_argument("--vid-h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=25, help="output frame rate (slow-mo if > 1/frame_ms)")
    ap.add_argument("--allow-shmem-gib", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ridge", type=float, default=1.0, help="ridge lambda for the readout fit")
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--decoder", type=Path, default=None,
                    help="decoder.npz from --mode probe; without it the naive\n                          population differential is used")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    # ---- scale presets.  "room" reproduces E17; "fly" is the fly-scale rebuild.
    PRESETS = {
        # cell, radius, speed, frame_ms, damp, k_yaw, k_thrust, emd_shift, fog
        "room": dict(spacing=1.0, radius=6.5, speed=1.2, frame_ms=40.0,
                     damp=2.2, k_yaw=2.6, k_thrust=0.0, emd_shift=2, fog=1.8,
                     flow_scale=7.0, yaw_sign=1.0, inject_target="hsvs"),
        "fly":  dict(spacing=0.04, radius=0.30, speed=0.30, frame_ms=5.0,
                     damp=60.0, k_yaw=1800.0, k_thrust=0.35, emd_shift=1, fog=0.09,
                     flow_scale=1.0, yaw_sign=-1.0, inject_target="t4t5", solid=True),
    }
    pre = PRESETS[args.preset]
    for k, v in pre.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.vision == "flyvis":
        args.inject_target = "t4t5"      # flyvis drives the motion detectors themselves

    import importlib.util
    spec = importlib.util.spec_from_file_location("malecns_gate", Path(__file__).with_name("malecns_gate.py"))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    if args.allow_shmem_gib:
        gate.require_cuda(max_shmem_gib=args.allow_shmem_gib)
    else:
        gate.require_cuda()
    import torch
    dev = torch.device("cuda:0")

    outdir = args.out or Path(f"artifacts/fly-world-{time.strftime('%Y%m%dT%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MaleCNS connectome flying in an infinite cubic lattice")
    print("=" * 78)
    st = build_brain(args.gain, args.kc_gain, dev, torch)
    init_state(st, torch)
    st_all_types = np.asarray(json.loads((CSR / "tables.json").read_text())["types"], dtype=object)
    st_type_idx = np.load(CSR / "annotations.npz")["typeIdx"]
    pops = st["pops"]
    n = st["n"]

    P = lambda s: pops.resolve(s)
    hsL, hsR = P("prefix:HS/L,prefix:VS/L"), P("prefix:HS/R,prefix:VS/R")
    wmL, wmR = P("subclass:wm/L"), P("subclass:wm/R")
    _fam = np.array([str(x).startswith(("T4", "T5")) for x in st_all_types])
    t4t5 = np.flatnonzero(_fam[st_type_idx])          # every T4/T5 neuron in our graph
    for nm, idx in [("HS/VS left", hsL), ("HS/VS right", hsR), ("T4/T5", t4t5),
                    ("wing MN left", wmL), ("wing MN right", wmR)]:
        print(f"  {nm:<16} {idx.size:>5} neurons")
    assert hsL.size and hsR.size and wmL.size and wmR.size and t4t5.size, "empty population"
    print(f"  vision   : {args.vision}   drive into: {args.inject_target}")

    inj_np = np.zeros(n, dtype=np.float32)
    def set_inject(flow_yaw, flow_pitch):
        """Push-pull drive on HS/VS (emd) or a per-neuron drive on T4/T5 (flyvis)."""
        inj_np[:] = 0.0
        if args.inject_target == "hsvs":
            fy = float(np.clip(flow_yaw, -args.inject_clip, args.inject_clip))
            inj_np[hsL] = max(0.0, args.base_inject + args.flow_gain * fy)
            inj_np[hsR] = max(0.0, args.base_inject - args.flow_gain * fy)
        else:
            d = np.asarray(flow_yaw, dtype=np.float32)
            inj_np[t4t5] = args.fv_base + args.fv_gain * np.clip(d, 0.0, None)
        st["inject"] = torch.from_numpy(inj_np).to(dev)

    _, paz, pel, _ = load_photoreceptors()
    az_bins, el_bins = args.az_bins, args.el_bins
    paz_b = np.floor((paz + 180.0) / 360.0 * az_bins).astype(np.int64) % az_bins
    pel_b = np.clip(np.floor((90.0 - pel) / 180.0 * el_bins).astype(np.int64), 0, el_bins - 1)

    def render(center, yaw, pitch):
        p0, p1, ein = lattice_segments(center, args.radius, args.spacing, args.edge_jitter)
        fp, fin = (cell_samples(center, args.radius, args.spacing, args.edge_jitter)
                   if args.solid else (None, None))
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        cp, sp = np.cos(-pitch), np.sin(-pitch)
        for arr in (p0, p1):                      # world -> head frame
            x, y, z = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
            arr[:, 0] = cy * x + sy * y
            arr[:, 1] = -sy * x + cy * y
            x, z = arr[:, 0].copy(), arr[:, 2].copy()
            arr[:, 0] = cp * x + sp * z
            arr[:, 2] = -sp * x + cp * z
        return render_equirect(p0, p1, az_bins, el_bins, fog=args.fog, inten=ein,
                               pts=fp, pts_inten=fin)

    dt_s = args.frame_ms / 1000.0
    steps_per_frame = max(1, int(round(args.frame_ms / args.dt)))

    # ------------------------------------------------------------ reflex mode
    if args.mode == "reflex":
        print("\n  OPEN LOOP: a fixed optic-flow value on HS/VS, read the wings")
        print(f"  {'flow':>8} {'HS-L':>8} {'HS-R':>8} {'wm-L':>8} {'wm-R':>8} {'wmR-wmL':>9}")
        rows = []
        for flow in (-1.0, -0.5, 0.0, 0.5, 1.0):
            init_state(st, torch)
            set_inject(flow, 0.0)
            steps = int(600 / args.dt)
            for _ in range(steps):
                brain_step(st, torch)
            r = dict(flow=flow,
                     hsL=rates_over(st, hsL, steps), hsR=rates_over(st, hsR, steps),
                     wmL=rates_over(st, wmL, steps), wmR=rates_over(st, wmR, steps))
            r["diff"] = r["wmR"] - r["wmL"]
            rows.append(r)
            print(f"  {flow:>8.2f} {r['hsL']:>8.1f} {r['hsR']:>8.1f} "
                  f"{r['wmL']:>8.1f} {r['wmR']:>8.1f} {r['diff']:>9.2f}")
        (outdir / "reflex.json").write_text(json.dumps(rows, indent=2))
        print(f"\n  -> {outdir/'reflex.json'}")
        return 0

    # ------------------------------------------------------------ probe mode
    if args.mode == "probe":
        print("\n  PROBE: random-walk optic flow on HS/VS, record per-neuron wing")
        print("         rates, then fit a linear decoder wing-rate -> yaw flow.")
        rng = np.random.default_rng(args.seed)
        flow = 0.0
        R, Y = [], []
        for fi in range(args.frames):
            flow = float(np.clip(0.85 * flow + rng.normal(0, 0.32), -1.0, 1.0))
            set_inject(flow, 0.0)
            for _ in range(steps_per_frame):
                brain_step(st, torch)
            wl = st["tick"][wmL].float().cpu().numpy()
            wr = st["tick"][wmR].float().cpu().numpy()
            R.append(np.concatenate([wl, wr]) / (steps_per_frame * args.dt / 1000.0))
            Y.append(flow)
            st["tick"].zero_()
            if fi % 40 == 0:
                print(f"    f{fi:>4}  flow={flow:>7.3f}  wm total={R[-1].sum():>7.1f} Hz")
        R = np.asarray(R, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        def fit_eval(tr, te):
            mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-6
            Rn = (R - mu) / sd
            A = Rn[tr].T @ Rn[tr] + args.ridge * np.eye(Rn.shape[1])
            ww = np.linalg.solve(A, Rn[tr].T @ Y[tr])
            p = Rn @ ww
            sse = lambda a, b: float(((a - b) ** 2).sum())
            r2 = 1.0 - sse(Y[te], p[te]) / sse(Y[te], np.full(len(te), Y[tr].mean()))
            return ww, mu, sd, r2

        ntr = int(len(Y) * 0.7)
        # (a) temporal split -- does one fixed decoder stay valid as state drifts?
        w, mu, sd, r2_time = fit_eval(np.arange(ntr), np.arange(ntr, len(Y)))
        r2_fit = fit_eval(np.arange(ntr), np.arange(ntr))[3]
        # (b) shuffled split -- is the wing->flow map learnable at all?
        perm = np.random.default_rng(args.seed + 1).permutation(len(Y))
        r2_rand = fit_eval(perm[:ntr], perm[ntr:])[3]

        print("\n  decoder: 67 wing motor neurons -> yaw flow")
        print(f"    flow sd {Y.std():.3f}   |flow|>0.8 in {int((np.abs(Y) > 0.8).sum())}/{len(Y)} frames")
        print(f"    R^2 in-sample (temporal train)  {r2_fit:+.3f}")
        print(f"    R^2 held-out TEMPORAL split     {r2_time:+.3f}   <- stationarity test")
        print(f"    R^2 held-out SHUFFLED split     {r2_rand:+.3f}   <- learnability test")
        np.savez(outdir / "decoder.npz", w=w, mu=mu, sd=sd, r2_train=r2_fit,
                 r2_test=r2_time, r2_random=r2_rand,
                 wmL=wmL, wmR=wmR, base_inject=args.base_inject, flow_gain=args.flow_gain,
                 frame_ms=args.frame_ms, dt=args.dt, gain=args.gain)
        names = np.concatenate([np.full(wmL.size, "wm-L"), np.full(wmR.size, "wm-R")])
        ids = np.concatenate([wmL, wmR])
        print("    strongest weights (temporal fit):")
        for k in np.argsort(-np.abs(w))[:8]:
            print(f"      {names[k]} neuron {ids[k]:>7}  w={w[k]:+.3f}")
        print(f"\n  -> {outdir/'decoder.npz'}")
        return 0

    # -------------------------------------------------------------- fly mode
    dec = None
    if args.decoder:
        dz = np.load(args.decoder)
        dec = {k: dz[k] for k in ("w", "mu", "sd")}
        print(f"\n  readout: linear decoder from {args.decoder}")
        print(f"           fitted at base_inject {float(dz['base_inject']):.2f}, "
              f"R^2 shuffled {float(dz['r2_random']):+.3f} / temporal {float(dz['r2_test']):+.3f}")
    else:
        print("\n  readout: naive population differential (wm-R minus wm-L)")

    fv = None
    if args.vision == "flyvis":
        import importlib.util as _ilu
        _s = _ilu.spec_from_file_location("flyvis_eye", Path(__file__).with_name("flyvis_eye.py"))
        fe = _ilu.module_from_spec(_s); _s.loader.exec_module(fe)
        fv = fe.FlyvisEye(args.flyvis_view)
        _idx, _dirs, _typ = fe.load_t4t5_targets()
        fv_nodes = fv.map_to_connectome(_idx, _dirs, _typ)
        order = np.argsort(_idx)
        fv_nodes = [fv_nodes[k] for k in order]
        t4t5 = _idx[order]                    # the driveable subset (13,582 of 13,585)
        assert np.array_equal(t4t5, _idx[order])
        print(f"  flyvis   : {fv.n_cols} eye columns, {len(fv_nodes):,} T4/T5 neurons driven "
              f"by {len(np.unique(np.concatenate(fv_nodes))):,} optic-lobe nodes")
        fv.reset(args.fv_grey)

    body = Body(speed=args.speed)
    vid = None
    if not args.no_video:
        vid = Video(outdir / "lattice_flight.mp4", args.vid_w, args.vid_h, args.fps)

    prev_lum = None
    fv_hist = deque(maxlen=8)
    prev_flow = prev_flow_p = 0.0
    rate_lp = None
    rec = []
    t0 = time.time()
    pos = np.zeros(3)
    yaw = pitch = 0.0
    for fi in range(args.frames):
        lum = render(pos, yaw, pitch)
        if prev_lum is None:
            prev_lum = lum.copy()
        if fv is not None:
            # ---- flyvis: fitted optic lobe -> our own T4/T5
            resp = fv.step(lum, lum)      # stateful: T4/T5 see a continuous movie
            drive = fv.to_inject(fv_nodes, resp)
            drive = drive.reshape(t4t5.shape) if drive.shape != t4t5.shape else drive
            set_inject(drive, None)
            flow_yaw_n = float(drive.mean())
            flow_pitch_n = 0.0
            prev_lum = lum.copy()
        else:
            mh = emd(prev_lum, lum, axis=1, shift=args.emd_shift)    # horizontal
            mv = emd(prev_lum, lum, axis=0, shift=args.emd_shift)    # vertical
            band = slice(int(el_bins * 0.22), int(el_bins * 0.78))   # |el| < ~50 deg
            # normalise by the total luminance so the loop gain does not drift with
            # how many lattice edges happen to be in view
            norm = float(lum.sum()) + 1e-6
            # EMD output is proportional to the per-frame image displacement, so it
            # MUST be divided by the frame interval to become an angular rate --
            # otherwise the loop gain silently changes when --frame-ms changes.
            flow_yaw_raw = float(mh[band, :].sum()) / norm / dt_s    # + = world moves left
            flow_pitch_raw = float(mv[band, :].sum()) / norm / dt_s
            prev_lum = lum.copy()
            # the fly's motion pathway is low-passed (~20 ms); without this the EMD
            # is dominated by per-frame noise from the moving lattice
            a_f = float(np.exp(-dt_s / args.tau_flow))
            flow_yaw_n = a_f * prev_flow + (1 - a_f) * flow_yaw_raw
            flow_pitch_n = a_f * prev_flow_p + (1 - a_f) * flow_pitch_raw
            prev_flow, prev_flow_p = flow_yaw_n, flow_pitch_n

            set_inject(flow_yaw_n * args.flow_scale, flow_pitch_n * args.flow_scale)
        for _ in range(steps_per_frame):
            brain_step(st, torch)
        tick = st["tick"]
        scale_r = 1.0 / (steps_per_frame * args.dt / 1000.0)
        raw_wm = np.concatenate([tick[wmL].cpu().numpy(), tick[wmR].cpu().numpy()]).astype(np.float64) * scale_r
        if rate_lp is None:
            rate_lp = raw_wm.copy()
        else:
            a_r = float(np.exp(-dt_s / args.tau_motor))
            rate_lp = a_r * rate_lp + (1 - a_r) * raw_wm
        wl = float(rate_lp[:wmL.size].mean())
        wr = float(rate_lp[wmL.size:].mean())
        if dec is not None:
            flow_hat = float(((rate_lp - dec["mu"]) / dec["sd"]) @ dec["w"])
        else:
            flow_hat = (wr - wl) / 40.0
        st["tick"].zero_()

        # ---- the readout: THIS is the trained/fitted term (see the module doc)
        u_yaw = args.readout_gain * args.yaw_sign * flow_hat
        common = (wr + wl) / 2.0
        u_pitch = args.readout_gain * args.pitch_sign * (common - args.power_ref) / 40.0
        u_thrust = (common - args.power_ref) / args.power_ref      # brain-driven airspeed
        body.step(dt_s, u_yaw, u_pitch, args.k_yaw, args.k_pitch, damp=args.damp,
                  bias=args.bias, u_thrust=u_thrust, k_thrust=args.k_thrust,
                  tau_v=args.tau_v, v0=args.speed)
        if abs(fi - args.gust_at) < 0.5:
            body.omega += args.gust
        pos, yaw, pitch = body.p, body.yaw, body.pitch
        rec.append(dict(frame=fi, t=fi * dt_s, yaw=yaw, pitch=pitch, omega=body.omega,
                        wmL=wl, wmR=wr, flow_yaw=flow_yaw_n, flow_pitch=flow_pitch_n,
                        speed=body.speed, px=pos[0], py=pos[1], pz=pos[2]))

        if vid is not None:
            pw = args.vid_w - args.vid_h                     # perspective panel width
            fov_v = 2.0 * np.degrees(np.arctan(np.tan(np.radians(110.0) / 2.0)
                                               * args.vid_h / pw))
            persp = equirect_to_persp(lum, 110.0, fov_v, pw, args.vid_h)
            canvas = np.zeros((args.vid_h, args.vid_w, 3), np.uint8)
            canvas[:, :pw] = np.stack(
                [(np.clip(persp * args.exposure, 0, 1) * 255).astype(np.uint8)] * 3, -1)
            canvas[:, pw:] = topdown(pos[:2], body.traj, args.radius, args.vid_h, args.spacing)
            canvas[:, pw - 1] = 90
            yawdeg = np.degrees(np.unwrap([r["yaw"] for r in rec]))
            arr = np.array([yawdeg, [r["omega"] for r in rec],
                            [r["wmL"] / 60.0 for r in rec], [r["wmR"] / 60.0 for r in rec],
                            [np.clip(r["flow_yaw"] * 8.0, -1, 1) for r in rec]])
            H, m = 108, args.vid_h - 116                       # trace strip
            canvas[m:] = 12
            canvas[m - 1, :] = 70
            for k, col in enumerate([(255, 255, 255), (255, 90, 90), (90, 255, 120),
                                     (90, 200, 255), (255, 220, 90)]):
                v = arr[k] / 90.0 + 0.5 if k == 0 else (arr[k] + 2.0) / 4.0
                c = (np.arange(len(v)) / max(1, args.frames - 1) * (args.vid_w - 1)).astype(int)
                r_ = np.clip((1.0 - v) * (H - 1), 0, H - 1).astype(int)
                canvas[m + r_, c] = col
            vid.write(label(canvas, pw, args.vid_h, fi, args))
        if fi % 20 == 0:
            print(f"  f{fi:>4}  yaw={np.degrees(yaw):>7.1f}deg  omega={body.omega:>7.3f}  "
                  f"wm L/R={wl:>5.1f}/{wr:>5.1f}  flow={flow_yaw_n:>9.5f}")
    if vid is not None:
        vid.close()

    traj = np.array(body.traj)
    om = np.array([r["omega"] for r in rec])
    spd = np.array([r["speed"] for r in rec])
    BL = args.body_mm / 1000.0
    # a "saccade" = a contiguous run of samples above the rate threshold
    hi = np.abs(np.degrees(om)) > args.saccade_dps
    sac, k = 0, 0
    while k < len(hi):
        if hi[k]:
            sac += 1
            while k < len(hi) and hi[k]:
                k += 1
        else:
            k += 1
    quiet = ~hi
    radius_sac = float((spd[hi].mean() / np.abs(om[hi]).mean()) / BL) if hi.any() and np.abs(om[hi]).mean() > 1e-9 else float("nan")
    radius_all = float((spd.mean() / np.abs(om).mean()) / BL) if np.abs(om).mean() > 1e-9 else float("nan")
    summary = dict(
        mode="fly", frames=args.frames, frame_ms=args.frame_ms, dt=args.dt,
        gain=args.gain, spacing=args.spacing, radius=args.radius,
        base_inject=args.base_inject, flow_gain=args.flow_gain,
        k_yaw=args.k_yaw, yaw_sign=args.yaw_sign, speed=args.speed,
        gust_at=args.gust_at, gust=args.gust,
        wall_s=time.time() - t0,
        preset=args.preset, damp=args.damp, tau_yaw_ms=1000.0 / args.damp,
        body_lengths_per_s=float(args.speed / BL),
        speed_mean=float(spd.mean()), speed_min=float(spd.min()), speed_max=float(spd.max()),
        saccade_dps=args.saccade_dps, n_saccades=sac,
        saccade_frac=float(hi.mean()),
        turn_radius_BL_saccade=radius_sac, turn_radius_BL_mean=radius_all,
        straightness=float(np.linalg.norm(traj[-1] - traj[0])
                           / max(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum(), 1e-9)),
        yaw_rms_deg=float(np.degrees(np.std(np.unwrap([r["yaw"] for r in rec])))),
        omega_absmean_after_gust=float(np.abs(om[int(args.gust_at) + 1:]).mean()),
        omega_absmean_before_gust=float(np.abs(om[:int(args.gust_at)]).mean()),
        omega_steady=float(np.abs(om[len(om) // 2:]).mean()),
        heading_drift_deg=float(abs(np.degrees(np.unwrap([r["yaw"] for r in rec])[-1]
                                               - np.unwrap([r["yaw"] for r in rec])[0]))),
        readout_gain=args.readout_gain, bias=args.bias,
        decoder=str(args.decoder) if args.decoder else None,
        path_length_m=float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()),
        net_displacement_m=float(np.linalg.norm(traj[-1] - traj[0])),
        records=rec)
    (outdir / "flight.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(outdir / "flight.npz", traj=traj,
                        yaw=np.array([r["yaw"] for r in rec]),
                        omega=om, wmL=np.array([r["wmL"] for r in rec]),
                        wmR=np.array([r["wmR"] for r in rec]),
                        flow=np.array([r["flow_yaw"] for r in rec]))
    print(f"\n  wall {summary['wall_s']:.1f}s   path {summary['path_length_m']:.2f} m   "
          f"net {summary['net_displacement_m']:.2f} m")
    print(f"  yaw sd {summary['yaw_rms_deg']:.1f} deg   "
          f"|omega| before gust {summary['omega_absmean_before_gust']:.3f} "
          f"after {summary['omega_absmean_after_gust']:.3f} rad/s")
    print(f"  -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
