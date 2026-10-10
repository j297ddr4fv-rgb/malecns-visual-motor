#!/usr/bin/env python3
"""Scoped bootstrap for the Biological Cybernetics mechanism queue."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / 'scripts'
ADAPTER_ROOT = Path(__file__).resolve().parent
SUMMARY_FIELDS = tuple(f'{side}_{subtype}' for side in ('L', 'R')
                       for subtype in ('T4a', 'T4b', 'T4c', 'T4d', 'T4_unclear',
                                       'T5a', 'T5a_unclear', 'T5b', 'T5c', 'T5d'))
GRAPH_RECORD = None


def _measure_shuffle_graph(graph, seed):
    global GRAPH_RECORD
    target_before = graph.target
    indegree_before = np.bincount(target_before, minlength=graph.n)
    target_shuffled = np.array(target_before, copy=True)
    rng = np.random.default_rng(seed)
    source_ids = np.repeat(np.arange(graph.n, dtype=np.int32), np.diff(graph.offsets))
    signed_weights = graph.weight.astype(np.float64) * graph.sign[source_ids]
    # Target-specific Kenyon gains are applied when the engine builds weights.
    kc_classes = np.flatnonzero(np.asarray(graph.tables['classes']) == 'Kenyon_Cell')
    kc_nodes = np.isin(graph.annotations['classIdx'], kc_classes)
    target_kc_before = kc_nodes[target_before]
    order = np.lexsort((target_kc_before, signed_weights))
    ordered_weight = signed_weights[order]
    ordered_kc = target_kc_before[order]
    boundaries = np.r_[0, 1 + np.flatnonzero((ordered_weight[1:] != ordered_weight[:-1]) |
                                            (ordered_kc[1:] != ordered_kc[:-1])), graph.e]
    del ordered_weight, ordered_kc, source_ids
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        ids = order[int(begin):int(end)]
        destinations = target_shuffled[ids].copy()
        rng.shuffle(destinations)
        target_shuffled[ids] = destinations
    del order, destinations
    indegree_after = np.bincount(target_shuffled, minlength=graph.n)
    if not np.array_equal(indegree_before, indegree_after):
        raise RuntimeError('target-stub permutation changed the in-degree sequence')
    if not np.array_equal(target_kc_before, kc_nodes[target_shuffled]):
        raise RuntimeError('target-dependent gain would change an effective edge weight')
    signed_strength_max_error = {}
    for name, sign_mask in [('excitatory', signed_weights > 0), ('inhibitory', signed_weights < 0)]:
        values = np.abs(signed_weights[sign_mask])
        before = np.bincount(target_before[sign_mask], weights=values, minlength=graph.n)
        after = np.bincount(target_shuffled[sign_mask], weights=values, minlength=graph.n)
        error = float(np.max(np.abs(before-after)))
        signed_strength_max_error[name] = error
        if error > 1e-9:
            raise RuntimeError(f'{name} incoming weighted strength changed by {error}')
    changed_mask = target_shuffled != target_before
    changed_edges = int(np.count_nonzero(changed_mask))
    dn_types = np.flatnonzero(np.asarray(graph.tables['types']) == 'DNp15')
    dn_nodes = np.isin(graph.annotations['typeIdx'], dn_types)
    dn_incoming = dn_nodes[target_before]
    dn_afferent_edges = int(np.count_nonzero(dn_incoming))
    dn_afferent_changed = int(np.count_nonzero(changed_mask & dn_incoming))
    del changed_mask, dn_incoming, dn_nodes
    del signed_weights, target_kc_before, kc_nodes, values, before, after, sign_mask
    graph.target = target_shuffled
    self_loops = 0
    duplicate_target_entries = 0
    for source, (start, end) in enumerate(zip(graph.offsets[:-1], graph.offsets[1:])):
        row = graph.target[int(start):int(end)]
        self_loops += int(np.count_nonzero(row == source))
        if row.size > 1:
            ordered = np.sort(row)
            duplicate_target_entries += int(np.count_nonzero(ordered[1:] == ordered[:-1]))
    GRAPH_RECORD = {
        'variant': 'degree_preserving_target_stub_shuffle',
        'method': 'CSR target-stub permutation within equal signed-weight and target-Kenyon-gain strata; source rows remain fixed',
        'rng': 'numpy.random.default_rng',
        'rng_seed': int(seed),
        'neurons': int(graph.n),
        'edges': int(graph.e),
        'target_array_dtype': str(graph.target.dtype),
        'exact_in_degree_preserved': True,
        'exact_out_degree_preserved': True,
        'edge_count_preserved': True,
        'edge_weight_array_unchanged': True,
        'presynaptic_sign_array_unchanged': True,
        'effective_edge_weights_preserved': True,
        'neuron_excitatory_incoming_strength_preserved': True,
        'neuron_inhibitory_incoming_strength_preserved': True,
        'incoming_strength_max_abs_error': signed_strength_max_error,
        'weight_gain_strata_count': int(len(boundaries)-1),
        'changed_target_edges': changed_edges,
        'changed_target_fraction': changed_edges / graph.e,
        'original_DNp15_afferent_edges': dn_afferent_edges,
        'original_DNp15_afferent_edges_reassigned': dn_afferent_changed,
        'self_loop_edges_after_shuffle': self_loops,
        'duplicate_target_entries_within_source_rows': duplicate_target_entries,
        'self_loops_and_parallel_entries_are_retained_as_generated_by_stub_matching': True,
        'persistent_csr_files_modified': False,
        'extra_target_copy_bytes_during_load': int(target_shuffled.nbytes),
    }
    del target_before, indegree_before, indegree_after
    print(json.dumps({'network_intervention': GRAPH_RECORD}, sort_keys=True), file=sys.stderr, flush=True)


def _patch_graph_loader(variant):
    global GRAPH_RECORD
    sys.path.insert(0, str(SCRIPTS))
    import malecns_engine
    original = malecns_engine._load_graph
    seed = int(os.environ.get('BC_GRAPH_SHUFFLE_SEED', '20261010'))

    def load_graph(path):
        global GRAPH_RECORD
        graph = original(path)
        if variant == 'degree_preserving_target_shuffle':
            _measure_shuffle_graph(graph, seed)
        elif variant == 'original':
            GRAPH_RECORD = {
                'variant': 'original',
                'method': 'registered CSR target, weight, sign and offsets loaded unchanged',
                'neurons': int(graph.n), 'edges': int(graph.e),
                'persistent_csr_files_modified': False,
                'no_network_intervention': True,
            }
        else:
            raise ValueError(f'unknown BC_NETWORK_VARIANT={variant!r}')
        return graph

    malecns_engine._load_graph = load_graph


def _install_worker_observers(module):
    cls = module.OnlineNeuralWorker
    original_init = cls.__init__
    original_process_frame = cls.process_frame
    original_stop = cls.stop
    original_fail = cls.fail

    def init(self, args, reuse_from=None):
        original_init(self, args, reuse_from=reuse_from)
        self._bc_network_record = dict(GRAPH_RECORD or {'variant': os.environ.get('BC_NETWORK_VARIANT', 'unknown')})
        self._bc_network_record['worker_case_out_dir'] = str(self.out_dir)
        (self.out_dir / 'network_intervention.json').write_text(
            json.dumps(self._bc_network_record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        mapping = self.flyvis.mapping
        self._bc_subtype_side_masks = {
            key: ((np.asarray(mapping.side) == key.split('_', 1)[0]) &
                  (np.asarray(mapping.subtype) == key.split('_', 1)[1]))
            for key in SUMMARY_FIELDS
        }
        self._bc_subtype_side_counts = {
            key: int(np.count_nonzero(mask))
            for key, mask in self._bc_subtype_side_masks.items()
        }
        self._bc_side_masks = {side: np.asarray(mapping.side) == side for side in ('L', 'R')}
        self._bc_directions = np.asarray(mapping.directions, dtype=np.float32)
        self._bc_rows_path = self.out_dir / 't4t5_direction_summary.jsonl'
        self._bc_rows_handle = None

    def process_frame(self, request):
        reply = original_process_frame(self, request)
        output = self.flyvis.last_output
        mapping = self.flyvis.mapping
        values = np.asarray(output.drive, dtype=np.float32)
        if int(output.frame_index) != int(reply['seq']) or values.shape != self._bc_directions.shape[:1]:
            raise RuntimeError('FlyVis summary row is not aligned with the worker frame')
        subtype_means = {
            key: (float(np.mean(values[mask], dtype=np.float64)) if np.any(mask) else None)
            for key, mask in self._bc_subtype_side_masks.items()
        }
        axis_moments = {}
        for side, mask in self._bc_side_masks.items():
            axis_moments[side] = [
                float(np.mean(values[mask].astype(np.float64) * self._bc_directions[mask, axis], dtype=np.float64))
                for axis in range(3)
            ]
        row = {
            'seq': int(reply['seq']),
            'time_s': float(reply['time_s']),
            'units': str(output.units),
            'target_count': int(values.size),
            'subtype_eye_mean': subtype_means,
            'subtype_eye_target_counts': self._bc_subtype_side_counts,
            'eye_direction_vector_weighted_mean': axis_moments,
            'all_target_mean': float(np.mean(values, dtype=np.float64)),
            'all_target_abs_mean': float(np.mean(np.abs(values), dtype=np.float64)),
        }
        named = self.engine.last_named_outputs
        row['network_activity'] = {
            key: int(named[key]) for key in ('spike_count', 'active_neurons', 'deliveries')
        }
        row['population_rates_hz'] = dict(named.get('population_rates_hz', {}))
        row['population_spike_counts'] = dict(named.get('population_spike_counts', {}))
        if self._bc_rows_handle is None:
            self._bc_rows_handle = self._bc_rows_path.open('x', encoding='utf-8', buffering=1)
        self._bc_rows_handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        return reply

    def _close_summary(self):
        handle = getattr(self, '_bc_rows_handle', None)
        if handle is not None and not handle.closed:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def stop(self):
        _close_summary(self)
        return original_stop(self)

    def fail(self, error):
        _close_summary(self)
        return original_fail(self, error)

    cls.__init__ = init
    cls.process_frame = process_frame
    cls.stop = stop
    cls.fail = fail


def main():
    if len(sys.argv) < 5:
        raise SystemExit('usage: bootstrap WORKER SCHEDULE SHA CASE_ID [WORKER_ARGS...]')
    worker_path, schedule, digest, case_id, *worker_args = sys.argv[1:]
    expected = SCRIPTS / 'flygym_online_neural_worker_eye_parity_state_v2.py'
    if Path(worker_path).resolve() != expected.resolve():
        raise ValueError('unexpected neural worker')
    sys.path.insert(0, str(ADAPTER_ROOT))
    os.environ.update({
        'FLYGYM_EXTENSION_SCHEDULE_PATH': schedule,
        'FLYGYM_EXTENSION_SCHEDULE_SHA256': digest,
        'FLYGYM_EXTENSION_CASE_ID': case_id,
        'RESEARCH_QUEUE_PREFIX_RETINA': '1',
    })
    from functools import partial
    import malecns_gate
    import pixel_illusion_adapter
    malecns_gate.cuda_blockers = partial(
        malecns_gate.cuda_blockers, min_available_gib=0.0,
    )
    pixel_illusion_adapter.install_worker_renderer()
    _patch_graph_loader(os.environ.get('BC_NETWORK_VARIANT', 'original'))
    spec = importlib.util.spec_from_file_location('bc_online_neural_worker', worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('could not load the pinned worker')
    worker_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker_module
    spec.loader.exec_module(worker_module)
    _install_worker_observers(worker_module)
    sys.argv = [worker_path, *worker_args]
    worker_module.main()


if __name__ == '__main__':
    main()
