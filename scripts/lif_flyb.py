#!/usr/bin/env python3
"""Calibrated MaleCNS LIF on the FLYB graph, ported from fly-brain-minecraft.

Runs the *same* model as the validated Java implementation
(blendi-remade/fly-brain-minecraft, MIT) on the *same* graph, so spike output is
directly comparable:

    tau_m 20 ms, tau_s 5 ms, v_rest = v_reset = -52 mV, threshold -45 mV,
    delay 1.8 ms, refractory 2.2 ms, wSyn 0.275 mV, gain 0.65,
    w[e] = ntSign[src] * wSyn * gain * synapseCount[e] * postsynapticGain[target]

Graph: /data/cache/malecns-gpu/csr-flyb (176,422 neurons, 6,287,749 edges >= 5
synapses, 90,296,905 synapses) produced by flyb_to_csr.py from their FLYB file.

Backends
    torch  - straightforward tensor ops, one host sync per step (reference)
    triton - FastFly-style fused pipeline:
             1. neuron_step: consume the delay slot, integrate, threshold,
                Poisson stimulus, emit spike flags (one launch)
             2. compact: block-level scan + one atomic per block -> dense active list
             3. deliver: grid-stride, one active neuron at a time, vectorised
                chunks of its CSR row, atomic add into the future delay slot
             No host synchronisation inside the loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import malecns_gate as gate  # noqa: E402
from population_selectors import comma_terms, intersect_terms  # noqa: E402

CSR = Path("/data/cache/malecns-gpu/csr-flyb")
ROOT = Path(__file__).resolve().parent.parent


def _aligned_steps(duration_ms: float, step_ms: float, label: str) -> int:
    """Return an exact positive step count or fail before any backend starts.

    Escape latency is a primary result, so silently rounding a 130 ms protocol
    to three 50 ms reporting windows is not acceptable.  Callers must choose a
    reporting/trace grid that exactly represents the declared neural time.
    """

    duration = float(duration_ms)
    step = float(step_ms)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{label} duration must be a positive finite value")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"{label} step must be a positive finite value")
    count = int(round(duration / step))
    if count <= 0 or not math.isclose(count * step, duration, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"{label} duration {duration:g} ms is not aligned to step {step:g} ms"
        )
    return count


# --------------------------------------------------------------------- populations
class Populations:
    """Python port of PopulationIndex (same spec grammar)."""

    def __init__(self, tables: dict, ann: dict):
        self.t = {k: np.array(v) for k, v in tables.items()}
        self.a = ann
        self.n = len(self.a["typeIdx"])
        self._cache: dict[str, np.ndarray] = {}

    def _eq(self, codes, table, needle):
        lut = np.array([x == needle for x in self.t[table]])
        return np.flatnonzero(lut[codes])

    def resolve(self, spec: str) -> np.ndarray:
        if spec in self._cache:
            return self._cache[spec]
        parts = []
        for _raw, term, side in comma_terms(spec):
            ids = self._term(term)
            if side is not None:
                ids = ids[self.t["sides"][self.a["sideIdx"][ids]] == side]
            parts.append(ids)
        out = np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
        self._cache[spec] = out
        return out

    def _term(self, term: str) -> np.ndarray:
        if "&" in term:
            return intersect_terms(term, self._simple)
        return self._simple(term)

    def _simple(self, term: str) -> np.ndarray:
        if ":" not in term:
            return np.arange(self.n) if term.lower() == "all" else self._eq(self.a["typeIdx"], "types", term)
        kind, arg = term.split(":", 1)
        kind, arg = kind.strip().lower(), arg.strip()
        ty = self.t["types"][self.a["typeIdx"]]
        if kind in {"prefix", "contains"}:
            names = ty.astype(str)
            matched = np.char.startswith(names, arg) if kind == "prefix" else np.char.find(names, arg) >= 0
            return np.flatnonzero(matched)
        fields = {
            "type": ("typeIdx", "types"), "class": ("classIdx", "classes"),
            "superclass": ("superclassIdx", "superclasses"),
            "subclass": ("subclassIdx", "subclasses"), "nt": ("ntIdx", "nts"),
            "nerve": ("nerveIdx", "nerves"), "neuromere": ("neuromereIdx", "neuromeres"),
        }
        if kind in fields:
            field, table = fields[kind]
            return self._eq(self.a[field], table, arg)
        if kind == "body":
            return np.flatnonzero(self.ids == int(arg))
        raise SystemExit(f"unsupported population term: {term}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", type=float, default=500.0, help="simulated milliseconds")
    ap.add_argument("--tick-ms", type=float, default=50.0, help="reporting tick")
    ap.add_argument("--dt", type=float, default=0.5, help="integration step (Java bench default 0.1)")
    ap.add_argument("--gain", type=float, default=0.65)
    ap.add_argument("--kc-gain", type=float, default=0.25, help="postsynaptic gain on class:Kenyon_Cell")
    ap.add_argument("--stim", default="", help='"spec:hz;spec:hz"')
    ap.add_argument("--report", default="superclass:descending_neuron;superclass:vnc_motor;MN9;DNp01")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--backend", choices=["torch", "triton"], default="torch")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cpu runs skip the CUDA gate: the gate protects against the "
                         "nvidia_uvm ZONE_DEVICE bug, which a CPU run cannot trigger")
    ap.add_argument("--allow-shmem-gib", type=float, default=None,
                    help="explicitly raise the gate's Shmem ceiling for this run only; "
                         "the override is printed and recorded in summary.json")
    ap.add_argument("--inject", default="",
                    help='"spec:mV_per_ms;..." tonic current injection, as the mod\'s sensory '
                         'encoder does for the lamina (photoreceptors are inhibitory, so L1/L2/L3 '
                         'need a tonic depolarising drive that light suppresses)')
    ap.add_argument("--stim-table", type=Path, default=None,
                    help="time-varying stimulus from video_to_eye.py: an .npz with "
                         "rates (T,K) in Hz, neurons (K,) and frame_ms. Overrides --stim.")
    ap.add_argument("--trace-ms", type=float, default=0.0,
                    help="record which neurons spiked every N ms of neural time into "
                         "spikes.npz, for offline rendering (0 = off)")
    ap.add_argument("--profile", type=int, default=0,
                    help="profile N steps with torch.profiler and print per-kernel GPU times")
    ap.add_argument("--tick-print-every", type=int, default=1,
                    help="print every Nth tick row (long runs produce thousands)")
    ap.add_argument("--out", type=Path, default=None)
    # ---- short-term plasticity (Tsodyks-Markram), E21
    ap.add_argument("--stp", action="store_true",
                    help="enable Tsodyks-Markram short-term plasticity")
    ap.add_argument("--stp-u", type=float, default=0.45, help="baseline utilization U")
    ap.add_argument("--stp-tau-rec", type=float, default=500.0, help="recovery tau, ms")
    ap.add_argument("--stp-tau-facil", type=float, default=200.0, help="facilitation tau, ms")
    ap.add_argument("--stp-ref-hz", type=float, default=20.0,
                    help="presynaptic rate at which the STP gain is normalised to 1;\n                          without this every synapse is rescaled by its release\n                          fraction (~0.02 at steady state) and the network goes silent")
    args = ap.parse_args()

    try:
        integration_steps = _aligned_steps(args.ms, args.dt, "simulation")
        ticks = _aligned_steps(args.ms, args.tick_ms, "reporting")
        steps_per_tick = _aligned_steps(args.tick_ms, args.dt, "reporting/integration")
        if args.trace_ms > 0.0:
            _aligned_steps(args.ms, args.trace_ms, "trace")
            steps_per_frame = _aligned_steps(args.trace_ms, args.dt, "trace/integration")
        else:
            steps_per_frame = 0
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if ticks * steps_per_tick != integration_steps:
        raise SystemExit("internal time-grid mismatch")

    if args.device == "cuda":
        if args.allow_shmem_gib is not None:
            print(f"!! CUDA gate override: Shmem ceiling raised to {args.allow_shmem_gib} GiB "
                  f"for this run (default {gate.DEFAULT_MAX_SHMEM_GIB}). Host MemAvailable is "
                  f"{int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) // 1024} MiB. "
                  f"uvm_disable_hmm=1 is applied, so the ZONE_DEVICE path that caused the "
                  f"2026-09-15 freeze is skipped; this is a host-RAM guard, not a VRAM one.")
            gate.require_cuda(max_shmem_gib=args.allow_shmem_gib)
        else:
            gate.require_cuda()  # fail closed, before torch is imported
    else:
        print("device cpu: CUDA gate not applicable (no CUDA context is created)")
    import torch

    dev = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    def sync():
        """No-op on the CPU path -- sync() raises when no GPU
        is present, which broke every CPU run once the eGPU was unplugged."""
        if dev.type == "cuda":
            torch.cuda.synchronize()
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
    e = int(target_np.size)

    pops = Populations(tables, ann)
    pops.ids = ids

    # ---- per-edge effective weight: sign(src) * wSyn * gain * synapses * postGain(target)
    post_gain = np.ones(n, dtype=np.float32)
    kc = pops.resolve("class:Kenyon_Cell")
    post_gain[kc] = args.kc_gain
    src_of_edge = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets_np))
    w_final = (sign_np[src_of_edge].astype(np.float32)
               * (0.275 * args.gain)
               * weight_np.astype(np.float32)
               * post_gain[target_np])

    offsets = torch.from_numpy(offsets_np).to(dev)
    target = torch.from_numpy(target_np.astype(np.int64)).to(dev)
    wf = torch.from_numpy(w_final).to(dev)
    del src_of_edge, w_final

    # ---- model constants (LifConfig defaults)
    dt = args.dt
    tau_m, tau_s = 20.0, 5.0
    v_rest, v_thr, v_reset = -52.0, -45.0, -52.0
    decay = float(np.exp(-dt / tau_m))
    syn_decay = float(np.exp(-dt / tau_s))
    if abs(tau_m - tau_s) < 1e-9:
        syn_coupling = (dt / tau_m) * np.exp(-dt / tau_m)
    else:
        syn_coupling = (tau_s / (tau_m - tau_s)) * (np.exp(-dt / tau_m) - np.exp(-dt / tau_s))
    delay_steps = max(1, int(round(1.8 / dt)))
    ref_steps = max(1, int(round(2.2 / dt)))
    slots = delay_steps + 1

    # ---- stimulus
    stim_prob = np.zeros(n, dtype=np.float32)
    stim_desc = []
    table_probs_t = None          # (T, K+1) per-step probabilities, last column = silent
    table_slot_t = None           # (n,) which table column each neuron reads
    table_bins = 0
    table_spb = 0
    stimulus_table_meta = None
    stimulus_table_sha256 = None
    if args.stim_table is not None:
        td = np.load(args.stim_table)
        rates, tneur, frame_ms = td["rates"], td["neurons"], float(td["frame_ms"])
        try:
            _aligned_steps(frame_ms, args.dt, "stimulus/integration")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if "meta" in td.files:
            try:
                stimulus_table_meta = json.loads(str(td["meta"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit("stimulus table meta must be valid JSON") from exc
        stimulus_table_sha256 = hashlib.sha256(args.stim_table.read_bytes()).hexdigest()
        k = len(tneur)
        probs = np.zeros((rates.shape[0], k + 1), dtype=np.float32)
        probs[:, :k] = np.clip(rates * dt / 1000.0, 0.0, 1.0)
        tslots = np.full(n, k, dtype=np.int32)      # default: the silent column
        tslots[tneur] = np.arange(k, dtype=np.int32)
        table_probs_t = torch.from_numpy(probs).to(dev)
        table_slot_t = torch.from_numpy(tslots).to(dev)
        table_bins, table_spb = rates.shape[0], max(1, int(round(frame_ms / dt)))
        stim_desc.append(f"video table {args.stim_table.name}: {rates.shape[0]} frames x {k:,} "
                         f"target neurons, {frame_ms:.1f} ms/frame "
                         f"(mean {rates.mean():.1f} Hz, max {rates.max():.1f} Hz)")
        args.ms = min(args.ms, rates.shape[0] * frame_ms)
    for entry in [x for x in args.stim.split(";") if x.strip()]:
        k = entry.rfind(":")
        spec, hz = entry[:k].strip(), float(entry[k + 1:])
        pop = pops.resolve(spec)
        stim_prob[pop] = min(1.0, hz * dt / 1000.0)
        stim_desc.append(f"{spec} ({pop.size:,} neurons @ {hz} Hz)")
    stim_prob_t = torch.from_numpy(stim_prob).to(dev)

    inject_step = np.zeros(n, dtype=np.float32)
    inject_desc = []
    for entry in [x for x in args.inject.split(";") if x.strip()]:
        k = entry.rfind(":")
        spec, mv = entry[:k].strip(), float(entry[k + 1:])
        pop = pops.resolve(spec)
        inject_step[pop] = mv * dt
        inject_desc.append(f"{spec} ({pop.size:,} neurons @ {mv} mV/ms)")
    inject_t = torch.from_numpy(inject_step).to(dev)
    has_inject = bool(inject_step.any())

    reports, report_pops = [], []
    for spec in [s.strip() for s in args.report.split(";") if s.strip()]:
        p = pops.resolve(spec)
        reports.append(spec)
        report_pops.append(torch.from_numpy(p.astype(np.int64)).to(dev))

    print(f"device   : {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'}")
    print(f"graph    : {n:,} neurons, {e:,} edges, {int(weight_np.sum()):,} synapses "
          f"(sign +{int((sign_np > 0).sum()):,} / -{int((sign_np < 0).sum()):,})")
    print(f"model    : dt={dt}ms tau_m={tau_m} tau_s={tau_s} vRest={v_rest} vThr={v_thr} "
          f"delay={delay_steps}*{dt}ms refr={ref_steps}*{dt}ms gain={args.gain} kcGain={args.kc_gain}")
    print(f"stimuli  : {stim_desc if stim_desc else '(none - silent brain)'}")
    for spec, p in zip(reports, report_pops):
        print(f"report {spec:<40} {p.numel():>7,} neurons")
    print(f"backend  : {args.backend}")
    print()

    # ---- state
    v = torch.full((n,), v_rest, device=dev)
    g = torch.zeros(n, device=dev)
    ref_until = torch.zeros(n, dtype=torch.int32, device=dev)
    ring = torch.zeros((slots, n), device=dev)
    spk = torch.zeros(n, dtype=torch.uint8, device=dev)
    spike_tick = torch.zeros(n, dtype=torch.int32, device=dev)

    # ---- short-term plasticity state, one entry per EDGE (6.29 M)
    stp_x = stp_u = stp_last = None
    if args.stp:
        stp_x = torch.ones(e, device=dev)                       # available resources
        stp_u = torch.full((e,), args.stp_u, device=dev)        # utilization
        stp_last = torch.full((e,), -10_000_000, dtype=torch.int32, device=dev)
        # steady-state release at a reference rate, used to keep the calibrated
        # operating point: the connectome's 0.275 mV/synapse assumes release = 1
        r_ref = args.stp_ref_hz / 1000.0                      # per ms
        u_ss = args.stp_u * (1.0 + args.stp_tau_facil * r_ref) / (
            1.0 + args.stp_u * args.stp_tau_facil * r_ref)
        x_ss = 1.0 / (1.0 + u_ss * args.stp_tau_rec * r_ref)
        stp_norm = 1.0 / max(u_ss * x_ss, 1e-9)
        print(f"stp      : Tsodyks-Markram  U={args.stp_u} tauRec={args.stp_tau_rec}ms "
              f"tauFacil={args.stp_tau_facil}ms   state {e*3*4/1e6:.0f} MB on {e:,} edges")
        print(f"           normalised at {args.stp_ref_hz:g} Hz: u_ss={u_ss:.3f} x_ss={x_ss:.4f} "
              f"release={u_ss*x_ss:.4f} -> weight x{stp_norm:.1f}")

    has_stim = bool(stim_prob.any()) or table_probs_t is not None

    def torch_step(step: int, gen: torch.Generator):
        """Exact port of LifNetwork.step()/integrateSegment()/emitSpike()."""
        read = step % slots
        write = (step + delay_steps) % slots
        # 1. Poisson spike generators: emitted *before* integration; the neuron is
        #    reset and made refractory, and g is left untouched.
        if has_stim:
            if table_probs_t is not None:
                # time-varying drive: one row of the video table per stimulus bin
                b = min(step // table_spb, table_bins - 1)
                p = table_probs_t[b].index_select(0, table_slot_t)
            else:
                p = stim_prob_t
            stim = torch.rand(n, device=dev, generator=gen) < p
            v.masked_fill_(stim, v_reset)
            ref_until.masked_fill_(stim, step + ref_steps)
        else:
            stim = None
        # 2. integrate: delayed input joins g, then the exact two-state update.
        #    While refractory v is clamped to vReset and g does not decay.
        inp = ring[read]
        g.add_(inp)
        inp.zero_()
        refr = ref_until > step
        v_new = v_rest + (v - v_rest) * decay + g * syn_coupling + inject_t
        v.copy_(torch.where(refr, torch.full_like(v, v_reset), v_new))
        g.copy_(torch.where(refr, g, g * syn_decay))
        spiked = (~refr) & (v >= v_thr)
        v.masked_fill_(spiked, v_reset)
        g.masked_fill_(spiked, 0.0)
        ref_until.masked_fill_(spiked, step + ref_steps)
        allspk = spiked if stim is None else (spiked | stim)
        # 3. deliver every spike of this step into the future delay slot
        idx = allspk.nonzero().flatten()
        deliveries = 0
        if idx.numel():
            starts = offsets.index_select(0, idx)
            lens = offsets.index_select(0, idx + 1) - starts
            cum = torch.cumsum(lens, 0)
            total = int(cum[-1].item())
            if total:
                pos = torch.arange(total, device=dev)
                grp = torch.searchsorted(cum, pos, right=True)
                syn = starts.index_select(0, grp) + pos - (cum.index_select(0, grp) - lens.index_select(0, grp))
                if stp_x is None:
                    w_eff = wf.index_select(0, syn)
                else:
                    # --- Tsodyks-Markram: only touched edges are updated, and the
                    #     decay is applied lazily from each edge's own last-spike
                    #     time (decaying all 6.29 M edges every step would cost
                    #     ~100 MB/step of traffic, ~13x the whole delivery budget).
                    xe = stp_x.index_select(0, syn)
                    ue = stp_u.index_select(0, syn)
                    te = stp_last.index_select(0, syn)
                    dts = (step - te).to(torch.float32) * dt
                    xf = 1.0 - (1.0 - xe) * torch.exp(-dts / args.stp_tau_rec)
                    uf = args.stp_u + (ue - args.stp_u) * torch.exp(-dts / args.stp_tau_facil)
                    uf = uf + args.stp_u * (1.0 - uf)           # facilitation on this spike
                    rel = uf * xf                               # released fraction
                    stp_x.index_copy_(0, syn, xf - rel)
                    stp_u.index_copy_(0, syn, uf)
                    stp_last.index_copy_(0, syn, torch.full((syn.numel(),), step, dtype=torch.int32, device=dev))
                    w_eff = wf.index_select(0, syn) * (rel * stp_norm)
                ring[write].index_add_(0, target.index_select(0, syn), w_eff)
                deliveries = total
        spike_tick.add_(allspk.to(torch.int32))
        return int(idx.numel()), deliveries

    triton_step = None
    if args.backend == "triton":
        import triton
        import triton.language as tl

        offsets_i32 = torch.from_numpy(offsets_np.astype(np.int32)).to(dev)
        target_i32 = torch.from_numpy(target_np).to(dev)
        list_buf = torch.zeros(n, dtype=torch.int32, device=dev)
        spk_ctr = torch.zeros(1, dtype=torch.int32, device=dev)       # spikes this step
        compact_ctr = torch.zeros(1, dtype=torch.int32, device=dev)   # compaction base/count
        deliv_ctr = torch.zeros(1, dtype=torch.int64, device=dev)

        @triton.jit
        def _neuron_step(v_ptr, g_ptr, ref_ptr, spk_ptr, ring_base, slot, n,
                         step, decay, syn_coupling, syn_decay, v_rest, v_thr, v_reset,
                         ref_steps, stim_ptr, seed, cnt_ptr,
                         probs_ptr, sslot_ptr, nbins, spb, pw, inj_ptr,
                         USE_TABLE: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            v = tl.load(v_ptr + offs, mask=m, other=v_rest)
            g = tl.load(g_ptr + offs, mask=m, other=0.0)
            ref = tl.load(ref_ptr + offs, mask=m, other=0)
            if USE_TABLE:
                b = tl.minimum(step // spb, nbins - 1)
                p = tl.load(probs_ptr + b * pw + tl.load(sslot_ptr + offs, mask=m, other=0),
                            mask=m, other=0.0)
            else:
                p = tl.load(stim_ptr + offs, mask=m, other=0.0)
            # 1. Poisson generators fire BEFORE integration: reset v and go refractory,
            #    leaving g untouched (mirrors LifNetwork.step's stim loop).
            drive = tl.rand(seed, offs) < p
            v = tl.where(drive, v_reset, v)
            ref = tl.where(drive, step + ref_steps, ref)
            # 2. delayed input joins g, then the exact two-state update; while refractory
            #    v is clamped to vReset and g is frozen (no decay).
            ring = tl.load(ring_base + slot * n + offs, mask=m, other=0.0)
            tl.store(ring_base + slot * n + offs, tl.zeros_like(ring), mask=m)
            g = g + ring
            inj = tl.load(inj_ptr + offs, mask=m, other=0.0)
            v_new = v_rest + (v - v_rest) * decay + g * syn_coupling + inj
            frozen = ref > step
            v = tl.where(frozen, v_reset, v_new)
            g = tl.where(frozen, g, g * syn_decay)
            spiked = (~frozen) & (v >= v_thr)
            v = tl.where(spiked, v_reset, v)
            g = tl.where(spiked, 0.0, g)
            ref = tl.where(spiked, step + ref_steps, ref)
            tl.store(v_ptr + offs, v, mask=m)
            tl.store(g_ptr + offs, g, mask=m)
            tl.store(ref_ptr + offs, ref, mask=m)
            tl.store(spk_ptr + offs, (spiked | drive).to(tl.uint8), mask=m)
            tl.atomic_add(cnt_ptr, tl.sum((spiked | drive).to(tl.int32), axis=0))

        @triton.jit
        def _compact(spk_ptr, list_ptr, cnt_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            s = tl.load(spk_ptr + offs, mask=m, other=0).to(tl.int32)
            excl = tl.cumsum(s, axis=0) - s
            base = tl.atomic_add(cnt_ptr, tl.sum(s, axis=0))
            tl.store(list_ptr + base + excl, offs, mask=m & (s > 0))

        @triton.jit
        def _deliver(list_ptr, cnt_ptr, off_ptr, tgt_ptr, w_ptr, ring_base, slot,
                     deliv_ptr, nprog, BLOCK_S: tl.constexpr):
            pid = tl.program_id(0)
            cnt = tl.load(cnt_ptr)
            while pid < cnt:
                nn = tl.load(list_ptr + pid).to(tl.int64)
                start = tl.load(off_ptr + nn)
                end = tl.load(off_ptr + nn + 1)
                base = start
                while base < end:
                    idx = base + tl.arange(0, BLOCK_S).to(tl.int64)
                    mm = idx < end
                    t = tl.load(tgt_ptr + idx, mask=mm, other=0)
                    w = tl.load(w_ptr + idx, mask=mm, other=0.0)
                    tl.atomic_add(ring_base + slot * nprog + t, w, mask=mm)
                    base += BLOCK_S
                tl.atomic_add(deliv_ptr, end - start)
                pid += tl.num_programs(0)

        BLOCK_N, BLOCK_C, BLOCK_S = 1024, 1024, 64
        grid_n = (triton.cdiv(n, BLOCK_N),)
        grid_c = (triton.cdiv(n, BLOCK_C),)
        grid_d = (2048,)

        step_spk = torch.zeros(1, dtype=torch.int64, device=dev)
        step_deliv = torch.zeros(1, dtype=torch.int64, device=dev)

        use_table = table_probs_t is not None
        probs_arg = table_probs_t if use_table else stim_prob_t
        sslot_arg = table_slot_t if use_table else stim_prob_t
        pw_arg = (table_probs_t.shape[1] if use_table else 0)
        nbins_arg = table_bins if use_table else 0
        spb_arg = table_spb if use_table else 1

        def triton_step_fn(step: int, seed: int):
            slot = step % slots
            write = (step + delay_steps) % slots
            # three separate counters: the neuron kernel's spike count, the
            # compaction base (which becomes the active-list length), and the
            # delivery counter. Sharing one tensor double-counts and makes the
            # delivery kernel walk uninitialised list entries.
            spk_ctr.zero_()
            compact_ctr.zero_()
            deliv_ctr.zero_()
            _neuron_step[grid_n](v, g, ref_until, spk, ring, slot, n, step, decay, syn_coupling,
                                 syn_decay, v_rest, v_thr, v_reset, ref_steps, stim_prob_t,
                                 seed, spk_ctr,
                                 probs_arg, sslot_arg, nbins_arg, spb_arg, pw_arg, inject_t,
                                 USE_TABLE=use_table, BLOCK=BLOCK_N)
            _compact[grid_c](spk, list_buf, compact_ctr, n, BLOCK=BLOCK_C)
            _deliver[grid_d](list_buf, compact_ctr, offsets_i32, target_i32, wf, ring, write,
                             deliv_ctr, n, BLOCK_S=BLOCK_S)
            spike_tick.add_(spk)
            # accumulate on device: reading these per step would force two host syncs
            step_spk.add_(spk_ctr)
            step_deliv.add_(deliv_ctr)
            return 0, 0

        # correctness: 20 deterministic steps (no stimulus) from an excited state
        # through both implementations; a one-step v/g comparison misses semantic
        # drift in refractory handling, so compare the whole trajectory.
        saved_prob = stim_prob_t.clone()
        saved_table = table_probs_t.clone() if table_probs_t is not None else None
        stim_prob_t.zero_()
        if table_probs_t is not None:
            table_probs_t.zero_()
        chk_gen = torch.Generator(device=dev).manual_seed(4242)
        v0 = torch.full((n,), v_rest, device=dev)
        v0[:500] = v_thr + 2.0

        # 1. single step must match exactly: this validates the whole per-step
        #    arithmetic (delay ring, refractory freeze, threshold, delivery).
        v.copy_(v0); g.zero_(); ref_until.zero_(); ring.zero_(); spike_tick.zero_()
        s_eager, _ = torch_step(0, chk_gen)
        ev1, eg1, er1 = v.clone(), g.clone(), ref_until.clone()
        v.copy_(v0); g.zero_(); ref_until.zero_(); ring.zero_(); spike_tick.zero_()
        step_spk.zero_(); step_deliv.zero_()
        triton_step_fn(0, 7)
        s_fused = int(step_spk.item())
        dv1, dg1 = float((ev1 - v).abs().max()), float((eg1 - g).abs().max())
        print(f"triton single-step check: spikes {s_eager} vs {s_fused}, "
              f"max|dv|={dv1:.2e} mV max|dg|={dg1:.2e}")
        if s_eager != s_fused or dv1 > 1e-3 or dg1 > 1e-3:
            raise SystemExit("fused kernel diverges from the eager reference on a single step")

        # 2. multi-step drift is reported, not enforced: the eager path sums a
        #    neuron's input with index_add_ and the fused path with atomic_add, so
        #    the float32 accumulation order differs. Near threshold that flips a
        #    few spikes and the network is chaotic, so exact equality over many
        #    steps is not attainable (and not what the cross-check needs).
        v.copy_(v0); g.zero_(); ref_until.zero_(); ring.zero_(); spike_tick.zero_()
        eager_seq = []
        for k in range(20):
            s, _ = torch_step(k, chk_gen)
            eager_seq.append(s)
        v.copy_(v0); g.zero_(); ref_until.zero_(); ring.zero_(); spike_tick.zero_()
        step_spk.zero_(); step_deliv.zero_()
        fused_seq = []
        for k in range(20):
            triton_step_fn(k, 7 + k)
            fused_seq.append(int(step_spk.item()))
            step_spk.zero_()
            step_deliv.zero_()
        stim_prob_t.copy_(saved_prob)
        if saved_table is not None:
            table_probs_t.copy_(saved_table)
        drift = abs(sum(fused_seq) - sum(eager_seq)) / max(1, sum(eager_seq))
        print(f"triton 20-step drift (informational): {sum(eager_seq):,} vs {sum(fused_seq):,} spikes "
              f"({drift:.1%} apart, chaotic amplification of float ordering)")
        if drift > 0.5:
            raise SystemExit("fused kernel is off by more than 50% over 20 steps")
        print()
        v.zero_().add_(v_rest); g.zero_(); ref_until.zero_(); ring.zero_(); spk.zero_()
        # warm up so Triton's JIT compilation is not charged to the timed region
        for w in range(30):
            triton_step_fn(w, 999 + w)
        sync()
        v.fill_(v_rest); g.zero_(); ref_until.zero_(); ring.zero_(); spk.zero_()
        spike_tick.zero_(); step_spk.zero_(); step_deliv.zero_()
        triton_step = triton_step_fn

    step_fn = triton_step or (lambda step, seed: torch_step(step, gen))
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    total_spikes = 0
    total_deliv = 0
    sim_nanos = 0
    rows = []
    hdr = f"{'t(ms)':>8} {'active':>8} {'spikes':>8} {'wall(ms)':>9}"
    for spec in reports:
        hdr += f" {spec[:13]:>14}"
    print(hdr)
    step_index = 0
    # Untimed warm-up tick. First-touch costs (Triton module load, allocator and
    # handle growth) otherwise land inside the first measured tick and dominate it.
    for _ in range(steps_per_tick):
        step_fn(step_index, args.seed + step_index)
        step_index += 1
    sync()
    v.fill_(v_rest); g.zero_(); ref_until.zero_(); ring.zero_(); spk.zero_()
    spike_tick.zero_()
    step_index = 0
    gen.manual_seed(args.seed)
    if triton_step is not None:
        step_spk.zero_(); step_deliv.zero_()
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile):
                step_fn(step_index, args.seed + step_index)
                step_index += 1
            sync()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
        v.fill_(v_rest); g.zero_(); ref_until.zero_(); ring.zero_(); spk.zero_()
        spike_tick.zero_()
        step_index = 0
        gen.manual_seed(args.seed)
        if triton_step is not None:
            step_spk.zero_(); step_deliv.zero_()
    tracing = args.trace_ms > 0
    trace_frames: list[np.ndarray] = []
    trace_times: list[float] = []
    frame_spk = torch.zeros(n, dtype=torch.uint8, device=dev) if tracing else None
    for t in range(ticks):
        t0 = time.perf_counter()
        tick_spikes = 0
        for _ in range(steps_per_tick):
            s, d = step_fn(step_index, args.seed + step_index)
            tick_spikes += s
            total_deliv += d
            step_index += 1
            if tracing:
                torch.maximum(frame_spk, spk, out=frame_spk)
                if step_index % steps_per_frame == 0:
                    trace_frames.append(frame_spk.nonzero().flatten().cpu().numpy().astype(np.int32))
                    trace_times.append(step_index * dt)
                    frame_spk.zero_()
        sync()
        wall = time.perf_counter() - t0
        if triton_step is not None:
            tick_spikes = int(step_spk.item())
            total_deliv += int(step_deliv.item())
            step_spk.zero_()
            step_deliv.zero_()
        sim_nanos += wall
        total_spikes += tick_spikes
        counts = spike_tick.cpu().numpy().copy()  # copy: zero_() would alias the view
        spike_tick.zero_()
        active = int((counts > 0).sum())
        line = (f"{step_index * dt:8.0f} {active:8d} {tick_spikes:8d} {wall * 1e3:9.1f}")
        rates = {}
        for spec, p in zip(reports, report_pops):
            pn = p.cpu().numpy()
            hz = counts[pn].sum() / (pn.size * args.tick_ms / 1000.0) if pn.size else 0.0
            rates[spec] = hz
            line += f" {hz:11.1f}Hz"
        if t % max(1, args.tick_print_every) == 0 or t == ticks - 1:
            print(line, flush=True)
        rows.append({"t_ms": step_index * dt, "active": active, "spikes": tick_spikes,
                     "wall_ms": wall * 1e3, "rates": rates})

    rtf = args.ms / (sim_nanos * 1e3)
    print(f"simulated {args.ms:.0f} ms in {sim_nanos:.2f} s wall ({rtf:.2f}x real time), "
          f"{total_spikes} spikes total, {total_deliv:,} deliveries")
    peak_host_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    gpu_memory = None
    if dev.type == "cuda":
        sync()
        gpu_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
            "total_bytes": int(torch.cuda.get_device_properties(dev).total_memory),
        }
        print(
            "memory: host peak RSS "
            f"{peak_host_rss_bytes / gate.GiB:.2f} GiB, GPU peak allocated "
            f"{gpu_memory['peak_allocated_bytes'] / gate.GiB:.2f} GiB / "
            f"{gpu_memory['total_bytes'] / gate.GiB:.2f} GiB"
        )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "graph": str(CSR), "neurons": n, "edges": e,
        "dt_ms": dt, "gain": args.gain, "kc_gain": args.kc_gain,
        "gate_shmem_limit_gib": (args.allow_shmem_gib if args.allow_shmem_gib is not None
                                  else gate.DEFAULT_MAX_SHMEM_GIB),
        "gate_shmem_override_gib": args.allow_shmem_gib,
        "peak_host_rss_bytes": peak_host_rss_bytes,
        "gpu_memory": gpu_memory,
        "stim": stim_desc,
        "stimulus_table": (
            None if args.stim_table is None else {
                "path": str(args.stim_table.resolve()),
                "sha256": stimulus_table_sha256,
                "meta": stimulus_table_meta,
            }
        ),
        "timing_contract": {
            "simulated_ms": float(args.ms),
            "integration_dt_ms": float(args.dt),
            "integration_steps": int(integration_steps),
            "report_tick_ms": float(args.tick_ms),
            "report_ticks": int(ticks),
            "trace_frame_ms": float(args.trace_ms),
            "exact_grid": True,
        },
        "reports": reports,
        "simulated_ms": args.ms, "wall_s": sim_nanos, "real_time_factor": rtf,
        "total_spikes": total_spikes, "total_deliveries": total_deliv, "ticks": rows,
    }
    out_dir = args.out or (ROOT / "artifacts" / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-flyb-{args.backend}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if tracing and trace_frames:
        offsets_np = np.cumsum([0] + [len(f) for f in trace_frames]).astype(np.int64)
        np.savez_compressed(out_dir / "spikes.npz",
                            times=np.asarray(trace_times, dtype=np.float64),
                            offsets=offsets_np,
                            neurons=np.concatenate(trace_frames))
        print(f"spike trace: {len(trace_frames)} frames, {int(offsets_np[-1]):,} spikes -> {out_dir / 'spikes.npz'}")
        print(f"  render with: python3 scripts/viz_cns.py --trace {out_dir / 'spikes.npz'}")
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
