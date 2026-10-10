#!/usr/bin/env python3
"""Bounded recorder for annotation-backed leg motor-neuron spike windows."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MAX_CAPTURE_FRAMES = 4000
EXPECTED_CELLS = 381
EXPECTED_MAPPED = 328
EXPECTED_UNMAPPED = 53
WINDOW_DT_S = 0.005
WINDOW_STEPS = 10
MAX_WINDOW_SPIKES = 10


class MotorChannelCapture:
    """Record actual raw neural counts for all six legs, without driving them.

    ``dense_counts`` is the host-side, per-neuron window count vector returned
    by ``MaleCNSEngine.step_window``. The recorder only reads it and the
    engine's graph annotations; it never advances or changes the engine.
    """

    def __init__(self, engine: Any, max_frames: int, dt_s: float,
                 out_dir: str | Path, mapping_path: str | Path):
        if isinstance(max_frames, bool) or int(max_frames) != max_frames:
            raise ValueError("max_frames must be an integer")
        self.max_frames = int(max_frames)
        if not 1 <= self.max_frames <= MAX_CAPTURE_FRAMES:
            raise ValueError(f"max_frames must be in [1, {MAX_CAPTURE_FRAMES}]")
        self.dt_s = float(dt_s)
        if not math.isfinite(self.dt_s) or not math.isclose(
                self.dt_s, WINDOW_DT_S, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("motor capture requires 5 ms windows (dt_s=0.005)")
        engine_dt_ms = float(getattr(engine, "dt_ms", float("nan")))
        if not math.isclose(engine_dt_ms, self.dt_s * 1000 / WINDOW_STEPS,
                            rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("engine dt_ms must be 0.5 for ten-tick 5 ms windows")
        self.engine = engine
        self.out_dir = Path(out_dir)
        if self.out_dir.exists():
            if not self.out_dir.is_dir():
                raise FileExistsError(f"capture output path is not a directory: {self.out_dir}")
        else:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.out_dir / "muscle_channels.npz"
        self.membership_path = self.out_dir / "muscle_channels_membership.json"
        for path in (self.output_path, self.membership_path):
            if path.exists():
                raise FileExistsError(f"capture output already exists: {path}")

        self.mapping_path = Path(mapping_path)
        mapping_bytes = self.mapping_path.read_bytes()
        self.mapping_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
        self.mapping = json.loads(mapping_bytes.decode("utf-8"))
        if not isinstance(self.mapping, Mapping):
            raise ValueError("leg muscle mapping must be a JSON object")
        self._legs = tuple(dict(item) for item in self.mapping.get("legs", ()))
        self._muscles = tuple(dict(item) for item in self.mapping.get("muscle_types", ()))
        if len(self._legs) != 6 or len({str(x.get("name", "")) for x in self._legs}) != 6:
            raise ValueError("mapping must define six uniquely named legs")
        type_names = [str(item.get("type_name", "")) for item in self._muscles]
        if not type_names or any(not name for name in type_names) or len(set(type_names)) != len(type_names):
            raise ValueError("mapping muscle type names must be non-empty and unique")
        if any(not str(item.get("group", "")) for item in self._muscles):
            raise ValueError("every mapped muscle type must define a group")
        rate_anchor = self.mapping.get("rate_anchor", {}).get("maximum_rate_hz")
        self.rate_anchor_hz = float(rate_anchor)
        if not math.isfinite(self.rate_anchor_hz) or self.rate_anchor_hz <= 0:
            raise ValueError("mapping rate_anchor.maximum_rate_hz must be positive")

        self.leg_names = np.asarray([str(item["name"]) for item in self._legs], dtype=str)
        self.muscle_types = np.asarray(type_names, dtype=str)
        self.muscle_groups = np.asarray(
            [str(item["group"]) for item in self._muscles], dtype=str,
        )
        (self.dense_indices, self.body_ids, self.names, self.sides, self.legs,
         self.types, self.mapped_groups, self.mapped, self._muscle_columns,
         self.membership) = self._resolve_membership(engine)
        self.n_cells = int(self.dense_indices.size)
        if self.n_cells != EXPECTED_CELLS:
            raise ValueError(f"expected {EXPECTED_CELLS} leg motor cells, found {self.n_cells}")
        if int(np.count_nonzero(self.mapped)) != EXPECTED_MAPPED:
            raise ValueError(f"expected {EXPECTED_MAPPED} exactly mapped cells")
        if int(np.count_nonzero(~self.mapped)) != EXPECTED_UNMAPPED:
            raise ValueError(f"expected {EXPECTED_UNMAPPED} explicitly unmapped cells")

        self.counts = np.zeros((self.max_frames, self.n_cells), dtype=np.uint16)
        self.seq = np.empty(self.max_frames, dtype=np.int64)
        self.time_s = np.empty(self.max_frames, dtype=np.float64)
        shape = (self.max_frames, len(self._legs), len(self._muscles))
        self.muscle_spike_counts = np.zeros(shape, dtype=np.uint32)
        self.muscle_rates_hz = np.zeros(shape, dtype=np.float32)
        self.muscle_normalized_rates = np.zeros(shape, dtype=np.float32)
        self._next_seq: int | None = None
        self._frames = 0
        self._total_spikes = 0
        self._max_spikes_per_cell_window = 0
        self._finalized = False

    def _resolve_membership(self, engine: Any) -> tuple[Any, ...]:
        graph = getattr(engine, "graph", None)
        popindex = getattr(engine, "populations", None)
        if graph is None or popindex is None:
            raise TypeError("engine must expose graph and populations")
        annotations = graph.annotations
        tables = graph.tables
        ids = np.asarray(graph.ids, dtype=np.int64)
        type_codes = np.asarray(annotations["typeIdx"], dtype=np.int64)
        type_table = np.asarray(tables["types"]).astype(str)
        side_codes = np.asarray(annotations["sideIdx"], dtype=np.int64)
        side_table = np.asarray(tables["sides"]).astype(str)
        if ids.size != int(engine.n) or type_codes.size != ids.size or side_codes.size != ids.size:
            raise ValueError("engine graph annotations and dense population size disagree")
        type_mapping = {
            str(item["type_name"]): (str(item["group"]), str(item.get("slug", "")))
            for item in self._muscles
        }

        all_indices: list[np.ndarray] = []
        all_ids: list[int] = []
        all_names: list[str] = []
        all_sides: list[str] = []
        all_legs: list[str] = []
        all_types: list[str] = []
        all_groups: list[str] = []
        all_mapped: list[bool] = []
        muscle_columns: list[list[np.ndarray]] = []
        membership: list[dict[str, Any]] = []
        for leg in self._legs:
            name = str(leg["name"])
            side = str(leg["side"])
            selector = f"superclass:vnc_motor&subclass:{leg['subclass']}/{side}"
            indices = np.asarray(popindex.resolve(selector), dtype=np.int64).reshape(-1)
            if indices.size == 0 or np.any(indices < 0) or np.any(indices >= ids.size):
                raise ValueError(f"invalid or empty engine population for {name}: {selector}")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"population selector returned duplicate dense indices for {name}")
            all_indices.append(indices)
            leg_columns: list[np.ndarray] = []
            for dense_index in indices:
                type_code = int(type_codes[dense_index])
                side_code = int(side_codes[dense_index])
                if not 0 <= type_code < type_table.size or not 0 <= side_code < side_table.size:
                    raise ValueError(f"invalid graph annotation at dense index {int(dense_index)}")
                type_name = str(type_table[type_code])
                annotated_side = str(side_table[side_code])
                if annotated_side != side:
                    raise ValueError(f"population selector side mismatch at dense index {int(dense_index)}")
                assignment = type_mapping.get(type_name)
                group = assignment[0] if assignment else "unmapped"
                is_mapped = assignment is not None
                all_ids.append(int(ids[dense_index]))
                all_names.append(f"{name}:body:{int(ids[dense_index])}")
                all_sides.append(annotated_side)
                all_legs.append(name)
                all_types.append(type_name)
                all_groups.append(group)
                all_mapped.append(is_mapped)
                membership.append({
                    "dense_index": int(dense_index), "body_id": int(ids[dense_index]),
                    "name": f"{name}:body:{int(ids[dense_index])}", "side": annotated_side,
                    "leg": name, "type_name": type_name, "mapped_group": group,
                    "mapped": is_mapped,
                    "muscle_slug": assignment[1] if assignment else "",
                    "mapping_reason": "exact_type_name" if is_mapped else "no_exact_muscle_type_mapping",
                })
            for muscle in self._muscles:
                leg_columns.append(np.flatnonzero(
                    np.asarray([entry["leg"] == name and
                                entry["type_name"] == str(muscle["type_name"])
                                for entry in membership], dtype=bool)
                ).astype(np.int64))
            muscle_columns.append(leg_columns)

        dense_indices = np.concatenate(all_indices).astype(np.int64, copy=False)
        if np.unique(dense_indices).size != dense_indices.size:
            raise ValueError("leg motor populations overlap; dense membership must be unique")
        # The membership loop assigned columns in the same deterministic order.
        if len(membership) != dense_indices.size:
            raise AssertionError("internal membership ordering mismatch")
        for entry, dense_index in zip(membership, dense_indices):
            if entry["dense_index"] != int(dense_index):
                raise AssertionError("internal membership index mismatch")
        return (
            dense_indices, np.asarray(all_ids, dtype=np.int64), np.asarray(all_names, dtype=str),
            np.asarray(all_sides, dtype=str), np.asarray(all_legs, dtype=str),
            np.asarray(all_types, dtype=str), np.asarray(all_groups, dtype=str),
            np.asarray(all_mapped, dtype=bool), muscle_columns, membership,
        )

    def metadata(self) -> dict[str, Any]:
        """Return compact capture and exact-membership counts for a run manifest."""
        per_leg = {}
        for leg in self.leg_names.tolist():
            mask = self.legs == leg
            per_leg[leg] = {
                "total": int(np.count_nonzero(mask)),
                "mapped": int(np.count_nonzero(mask & self.mapped)),
                "unmapped": int(np.count_nonzero(mask & ~self.mapped)),
            }
        return {
            "schema_version": 1, "output": str(self.output_path),
            "max_frames": self.max_frames, "frames_captured": self._frames,
            "counts_shape": [self._frames, self.n_cells],
            "counts_dtype": "uint16",
            "window_dt_s": self.dt_s, "window_steps": WINDOW_STEPS,
            "total_leg_motor_cells": self.n_cells,
            "mapped_cells": int(np.count_nonzero(self.mapped)),
            "unmapped_cells": int(np.count_nonzero(~self.mapped)),
            "expected_membership_counts": {
                "total": EXPECTED_CELLS, "mapped": EXPECTED_MAPPED,
                "unmapped": EXPECTED_UNMAPPED,
            },
            "actual_counts": {
                "shape": [self._frames, self.n_cells], "dtype": "uint16",
                "total_spikes": self._total_spikes,
                "maximum_cell_window_count": self._max_spikes_per_cell_window,
            },
            "per_leg": per_leg, "mapping_path": str(self.mapping_path),
            "mapping_config_sha256": self.mapping_sha256,
            "membership_path": str(self.membership_path),
            "cell_membership": self.membership,
            "rate_semantics": "mean cell rate_hz; normalized ratio is unclipped and is not a drive",
        }

    def capture(self, dense_counts: Any, seq: int, time_s: float) -> None:
        if self._finalized:
            raise RuntimeError("capture is already finalized")
        if self._frames >= self.max_frames:
            raise RuntimeError(f"capture frame limit {self.max_frames} reached")
        if isinstance(seq, (bool, np.bool_)) or int(seq) != seq or int(seq) < 0:
            raise ValueError("seq must be a non-negative integer")
        seq = int(seq)
        if self._next_seq is not None and seq != self._next_seq:
            raise ValueError(f"non-sequential capture seq: expected {self._next_seq}, got {seq}")
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("time_s must be finite and non-negative")
        if self._frames and not math.isclose(
                time_s - self.time_s[self._frames - 1], self.dt_s,
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("capture times must advance by exactly one 5 ms window")
        vector = np.asarray(dense_counts)
        if vector.ndim != 1 or vector.size != int(self.engine.n):
            raise ValueError(f"dense_counts must be a vector of length {self.engine.n}")
        if not np.issubdtype(vector.dtype, np.number):
            raise ValueError("dense_counts must be numeric")
        selected = vector[self.dense_indices]
        if (not np.isfinite(selected).all() or np.any(selected < 0)
                or np.any(selected > MAX_WINDOW_SPIKES)
                or np.any(selected != np.floor(selected))):
            raise ValueError("leg motor counts must be integer values in [0, 10]")
        frame = self._frames
        row = selected.astype(np.uint16, copy=False)
        self.counts[frame] = row
        self._total_spikes += int(np.sum(row, dtype=np.uint64))
        self._max_spikes_per_cell_window = max(
            self._max_spikes_per_cell_window, int(row.max(initial=0)),
        )
        self.seq[frame] = seq
        self.time_s[frame] = time_s
        for leg_index, muscle_columns in enumerate(self._muscle_columns):
            for muscle_index, columns in enumerate(muscle_columns):
                if columns.size:
                    total = int(np.sum(row[columns], dtype=np.uint64))
                    rate = total / (columns.size * self.dt_s)
                else:
                    total = 0
                    rate = 0.0
                self.muscle_spike_counts[frame, leg_index, muscle_index] = total
                self.muscle_rates_hz[frame, leg_index, muscle_index] = rate
                self.muscle_normalized_rates[frame, leg_index, muscle_index] = (
                    rate / self.rate_anchor_hz
                )
        self._frames += 1
        self._next_seq = seq + 1

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("capture is already finalized")
        if self._frames == 0:
            raise RuntimeError("cannot finalize an empty motor capture")
        if self.output_path.exists():
            raise FileExistsError(f"capture output already exists: {self.output_path}")
        n = self._frames
        provenance = {
            **self.metadata(), "mapping_config": self.mapping,
            "mapping_rate_anchor_hz": self.rate_anchor_hz,
            "engine": {
                "backend": str(getattr(self.engine, "backend", "unknown")),
                "device": str(getattr(self.engine, "device", "unknown")),
                "dt_ms": float(self.engine.dt_ms), "seed": int(getattr(self.engine, "seed", -1)),
                "csr_path": str(getattr(self.engine, "csr_path", "")),
                "graph_n": int(self.engine.graph.n), "graph_edges": int(self.engine.graph.e),
                "graph_meta": getattr(self.engine.graph, "meta", {}),
            },
            "count_semantics": "actual raw_neural.spikes window counts; no clipping or drive conversion",
            "unmapped_cells": [entry for entry in self.membership if not entry["mapped"]],
            "normalization_semantics": "mean_rate_hz / configured 30 Hz anchor; unclipped; analysis only",
            "vm_recorded": False,
        }
        # Refuse races with another writer immediately before creating either
        # of the two files owned by this recorder.
        for path in (self.output_path, self.membership_path):
            if path.exists():
                raise FileExistsError(f"capture output already exists: {path}")
        self.membership_path.write_text(json.dumps({
            "schema_version": 1,
            "mapping_config_sha256": self.mapping_sha256,
            "dense_indices": self.dense_indices.tolist(),
            "cells": self.membership,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.output_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                counts=self.counts[:n], dense_indices=self.dense_indices,
                body_ids=self.body_ids, names=self.names, sides=self.sides, legs=self.legs,
                types=self.types, mapped_groups=self.mapped_groups, mapped=self.mapped,
                seq=self.seq[:n], time_s=self.time_s[:n],
                leg_names=self.leg_names, muscle_types=self.muscle_types,
                muscle_groups=self.muscle_groups,
                muscle_spike_counts=self.muscle_spike_counts[:n],
                muscle_rates_hz=self.muscle_rates_hz[:n],
                muscle_normalized_rates=self.muscle_normalized_rates[:n],
                membership_json=np.asarray(json.dumps(
                    self.membership, ensure_ascii=False, sort_keys=True,
                ), dtype=np.str_),
                provenance_json=np.asarray(json.dumps(
                    provenance, ensure_ascii=False, sort_keys=True,
                ), dtype=np.str_),
            )
        self._finalized = True
        digest = hashlib.sha256(self.output_path.read_bytes()).hexdigest()
        return {
            "path": str(self.output_path),
            "size_bytes": self.output_path.stat().st_size,
            "sha256": digest,
        }
