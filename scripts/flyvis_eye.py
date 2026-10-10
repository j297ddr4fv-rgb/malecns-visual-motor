#!/usr/bin/env python3
"""flyvis as the optic-lobe front end for the MaleCNS flight loop (E20).

What this replaces
------------------
EXPERIMENT_LOG.md E19 established that the MaleCNS model's own retina -> lamina ->
medulla chain is impassable at the calibrated synaptic gain: T4 and T5 (13,585
neurons in our graph) sit at 0 Hz no matter how hard the lamina is driven, and
opening the medulla needs a 25-100x local gain.  E17/E18 therefore fed the flight
loop with a hand-written numpy Reichardt detector injected into 42 lobula-plate
cells -- 0.08 % of the 54,720-neuron visual system.

This module swaps that out for flyvis: a *connectome-constrained* optic-lobe
network (Lappalainen et al., Nature 2024) whose weights were fitted so that its
T4/T5 responses match recorded fly neurons.  It renders the same lattice world
onto one compound eye, runs flyvis, and returns a firing rate for every one of
our 13,585 T4/T5 neurons.

Where the boundary now sits
---------------------------
  retina + lamina + medulla   ->  flyvis (a real fitted optic-lobe model)
  T4 / T5                     ->  OUR connectome's 13,585 neurons, driven by flyvis
  lobula plate -> DN -> wing  ->  OUR connectome (T4/T5 -> VS is a direct 71k-synapse projection)

So this is the first configuration in which the motion detectors themselves are
in the loop.

Eye alignment
-------------
flyvis models ONE eye as a 721-element hexagonal array (u, v in [-15, 15]).
We match it to the MaleCNS eye map (column_directions.csv, 892 columns/eye with
measured viewing directions) by fitting an equidistant azimuthal projection of
the hex lattice against the measured directions, maximising mutual nearest
neighbours:

    az_head = az0 + mirror * rr * Theta * cos(phi)
    el_head = el0 +           rr * Theta * sin(phi)
    rr, phi from the BoxEye pixel layout: x = 13v, y = 13(u + v/2)

Fitted on both eyes: Theta = 85 deg, mirror = +1, el0 = +10 deg, az0 = -70 (R)
/ +70 (L), giving 612/721 (R) and 610/721 (L) mutual nearest-neighbour pairs at
a 1.9 deg median distance.  See scripts/flyvis_align.py for the fit itself.

Requires the flyvis virtualenv:
    FLYVIS_ROOT_DIR=/data/models/flyvis \
    /data/environments/projects/flyvis-opticlobe/bin/python ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CSR = Path("/data/cache/malecns-gpu/csr-flyb")
EYE_MAP = Path("/home/xvhang/Projects/fly-brain-minecraft/src/main/resources/"
               "assets/fruitfly/brain/column_directions.csv")

# fitted by scripts/flyvis_align.py (612/721 and 610/721 mutual NN at 1.9 deg)
EYE_FIT = {
    "R": dict(theta=85.0, mirror=+1.0, az0=-70.0, el0=10.0),
    "L": dict(theta=85.0, mirror=+1.0, az0=+70.0, el0=10.0),
}
# flyvis runs at dt = 1/200 s natively
FLYVIS_DT = 1.0 / 200.0
T4T5_PREFIX = ("T4", "T5")


def hex_to_dirs(uv, side):
    """flyvis hex (u, v) -> head-frame unit vectors for one eye."""
    f = EYE_FIT[side]
    u = uv[:, 0].astype(float)
    v = uv[:, 1].astype(float)
    x = v / 15.0                       # BoxEye: x = 13 v
    y = (u + v / 2.0) / 15.0           # BoxEye: y = 13 (u + v/2)
    rr = np.sqrt(x ** 2 + y ** 2)
    rr = rr / rr.max()
    phi = np.arctan2(y, x)
    az = np.radians(f["az0"] + f["mirror"] * rr * f["theta"] * np.cos(phi))
    el = np.radians(f["el0"] + rr * f["theta"] * np.sin(phi))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)


def optics_blur(lum, sigma_deg, az_bins=None):
    """Low-pass the world at the ommatidial acceptance angle.

    A fly ommatidium integrates ~4.5 deg, so a render sharp enough to alias the
    721-column sampling grid is *unphysical* and destroys direction selectivity
    (E20: corr(d+, d-) goes from +0.86 sharp to -0.55 at a 4 deg blur).
    """
    if sigma_deg <= 0:
        return lum
    import scipy.ndimage as ndi
    az_bins = az_bins or lum.shape[1]
    s = sigma_deg / (360.0 / az_bins)
    return ndi.gaussian_filter(lum, sigma=(s, s), mode=("nearest", "wrap"))


def normalise_contrast(lum, blur_deg, az_bins, mean=0.5, sd=0.18, floor=0.02):
    """Blur, then map to the luminance range flyvis was trained on ([0,1], ~0.5 grey)."""
    L = optics_blur(lum, blur_deg, az_bins)
    m, s = float(L.mean()), float(L.std()) + 1e-9
    return np.clip(mean + (L - m) * (sd / s), 0.0, 1.0)


def sample_equirect(lum, dirs):
    """Sample an equirect luminance map (el_bins, az_bins) at head-frame dirs."""
    el_bins, az_bins = lum.shape
    az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    el = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    xb = np.floor((az + 180.0) / 360.0 * az_bins).astype(np.int64) % az_bins
    yb = np.clip(np.floor((90.0 - el) / 180.0 * el_bins).astype(np.int64), 0, el_bins - 1)
    return lum[yb, xb]


class FlyvisEye:
    """One fitted optic lobe per eye, mapped onto our connectome's T4/T5."""

    def __init__(self, view: str = "flow/0000/000", device: str = "cuda:0"):
        os.environ.setdefault("FLYVIS_ROOT_DIR", "/data/models/flyvis")
        import torch
        from flyvis import NetworkView

        self.torch = torch
        self.nv = NetworkView(view)
        self.net = self.nv.init_network()
        self.net.eval()
        c = self.nv.connectome
        nodes_uv = np.stack([np.asarray(c.nodes.u), np.asarray(c.nodes.v)], 1)
        self.types = np.asarray(c.nodes.type).astype(str)
        self.columns, first = np.unique(nodes_uv, axis=0, return_index=True)
        self.col_of_node = {tuple(k): i for i, k in enumerate(map(tuple, self.columns))}
        self.node_col = np.array([self.col_of_node[tuple(x)] for x in nodes_uv])
        self.n_cols = len(self.columns)
        self.col_dir = {s: hex_to_dirs(self.columns, s) for s in ("L", "R")}

        # per (subtype, column) index into the flyvis response vector
        self.sub_col = {}
        for sub in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
            m = self.types == sub
            lut = np.full(self.n_cols, -1, np.int64)
            lut[self.node_col[m]] = np.flatnonzero(m)
            self.sub_col[sub] = lut
        self.input_index = np.asarray(self.net.stimulus.input_index)   # (8, 721)
        self._state = None
        self._frame = 0

    # ------------------------------------------------------------------ mapping
    BASE = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

    def _base(self, sub):
        """'T4_unclear' -> the four T4 subtypes; 'T5a_unclear' -> T5a."""
        if sub in self.BASE:
            return [sub]
        fam = "T4" if sub.startswith("T4") else "T5"
        if "_unclear" in sub and len(sub) > len(fam) + len("_unclear"):
            cand = sub.split("_")[0]
            if cand in self.BASE:
                return [cand]
        return [s for s in self.BASE if s.startswith(fam)]

    def map_to_connectome(self, t4t5_idx, t4t5_dirs, t4t5_type):
        """For each of our T4/T5 neurons pick the flyvis node(s) to read.

        `t4t5_dirs` are head-frame preferred directions obtained from the
        neuron's presynaptic columns (synapse-weighted), see the E20 section.
        Returns a list of node-index arrays, one per neuron (usually length 1;
        the few `*_unclear` cells average over their family).
        """
        out = [None] * len(t4t5_idx)
        side = np.where(t4t5_dirs[:, 1] < 0, "R", "L")     # +y is the fly's left
        for sub in sorted(set(t4t5_type)):
            for s in ("L", "R"):
                m = (t4t5_type == sub) & (side == s)
                if not m.any():
                    continue
                cd = self.col_dir[s]
                best = (t4t5_dirs[m] @ cd.T).argmax(1)
                nodes = [self.sub_col[b][best] for b in self._base(sub)]
                stack = np.stack(nodes, 0)                  # (n_base, n_matched)
                if (stack < 0).any():
                    raise RuntimeError(f"no flyvis {sub} on side {s} for some column")
                for k, j in enumerate(np.flatnonzero(m)):
                    out[j] = stack[:, k]
        if any(o is None for o in out):
            miss = sorted({t for t, o in zip(t4t5_type, out) if o is None})
            raise RuntimeError(f"unmapped T4/T5 types: {miss}")
        return out

    # ------------------------------------------------------------------ forward
    def reset(self, value=0.16):
        """Settle the network on a grey screen so the startup transient is gone.

        Without this every call restarts from the initial state and the
        relaxation transient (order 1.0) swamps the stimulus-driven part (order
        1e-3) -- which is exactly why an earlier version of this file produced
        identical responses for static, forward and yawing motion.
        """
        torch = self.torch
        with torch.no_grad():
            self._state = self.net.steady_state(
                t_pre=1.0, dt=FLYVIS_DT, batch_size=2, value=float(value))
        self._frame = 0

    def step(self, lum_left, lum_right):
        """One 5 ms frame for both eyes; returns (n_nodes,) responses.

        The network state is carried between calls, so T4/T5 see a continuous
        movie rather than a sequence of unrelated snapshots.
        """
        torch = self.torch
        if self._state is None:
            self.reset()
        vals = np.zeros((2, 1, 1, self.n_cols), np.float32)
        vals[0, 0, 0] = sample_equirect(lum_left, self.col_dir["L"])
        vals[1, 0, 0] = sample_equirect(lum_right, self.col_dir["R"])
        x = torch.tensor(vals, dtype=torch.float32)
        self.net.stimulus.zero(2, 1)
        self.net.stimulus.add_input(x)                       # -> (2, 1, 45669)
        with torch.no_grad():
            states = self.net(self.net.stimulus(), dt=FLYVIS_DT,
                              state=self._state, as_states=True)
        self._state = states[-1]
        self._frame += 1
        return states[-1].nodes.activity.detach().cpu().numpy().mean(0)

    def rates(self, resp_t4t5):
        """flyvis response units -> a drive value per T4/T5 node (>= 0)."""
        return np.clip(resp_t4t5, 0.0, None)

    def to_inject(self, nodes_per_neuron, resp):
        """Per-neuron flyvis response -> per-neuron drive, ready for the LIF.

        `nodes_per_neuron` is the list returned by map_to_connectome; the few
        `*_unclear` cells average over their family.
        """
        return np.array([resp[o].mean() for o in nodes_per_neuron], dtype=np.float32)


def load_t4t5_targets(csr: Path = CSR):
    """Our T4/T5 neurons with a preferred direction, from artifacts/flyvis-eye."""
    d = np.load(Path("artifacts/flyvis-eye/t4t5_dirs.npz"), allow_pickle=True)
    return d["idx"].astype(np.int64), d["dirs"].astype(np.float64), d["type"].astype(str)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", default="flow/0000/000")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()
    eye = FlyvisEye(a.view)
    idx, dirs, typ = load_t4t5_targets()
    nodes = eye.map_to_connectome(idx, dirs, typ)
    used = np.unique(np.concatenate(nodes))
    print(f"flyvis columns {eye.n_cols}, T4/T5 mapped {len(idx):,}")
    from collections import Counter
    print("  subtypes:", dict(Counter(typ)))
    print(f"  distinct flyvis nodes read: {len(used):,} / {eye.net.n_nodes:,}")
    lum = np.zeros((360, 720), np.float32); lum[120:240, :] = 1.0
    resp = eye.step(lum, lum)
    drive = eye.to_inject(nodes, resp)
    print(f"  test stimulus: response mean {resp.mean():+.3f} max {resp.max():+.3f}")
    print(f"  drive over T4/T5: mean {drive.mean():.3f} max {drive.max():.3f} "
          f"nonzero {np.mean(drive>0)*100:.0f}%")
