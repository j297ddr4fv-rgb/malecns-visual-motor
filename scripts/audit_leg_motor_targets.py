#!/usr/bin/env python3
"""Trace unresolved MaleCNS MNs to raw fields and published MANC matches.

MaleCNS IDs and MANC IDs refer to different specimens. Only an explicit
mancBodyid annotation is a predicted body-ID bridge; numeric identity is never a match.
The audit reports candidates and never assigns an unidentified MN to a muscle.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from leg_muscle_decoder import LegMuscleDecoder
from neuromuscular_adapter import DEFAULT_CSR

DATA = Path('/data/datasets/raw/malecns-muscle-target-audit/20261002')
RAW = Path('/data/datasets/raw/malecns/v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather')
SUPPLEMENT = DATA / 'elife-96084-supp3-v1.csv'
OUT = Path('/data/artifacts/by-project/malecns-gpu/engineering/motor-target-audit-20261002')
COARSE_TARGETS = {'', 'front leg', 'middle leg', 'hind leg', 'leg', 'TBD', 'unknown'}


def integral_id(value):
    if value is None or value == '':
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError(f'invalid cross-connectome ID: {value!r}')
    return int(number)


def crosswalk(row: dict, manc_rows: list[dict]) -> dict:
    """Keep curated group matches separate from error-prone predicted matches."""
    curated_group = integral_id(row.get('mancGroup'))
    curated_rows = [r for r in manc_rows if integral_id(r['group']) == curated_group] if curated_group else []
    curated_targets = sorted({r['target'].strip() for r in curated_rows if r['target'].strip()})
    curated_specific = sorted(set(curated_targets) - COARSE_TARGETS)
    bridge = integral_id(row.get('mancBodyid'))
    if bridge is not None:
        matched = [r for r in manc_rows if integral_id(r['bodyid']) == bridge]
        method = 'predicted_raw_mancBodyid' if matched else 'predicted_mancBodyid_not_in_supplement'
    else:
        name = row.get('mancType') or row.get('type')
        matched = [r for r in manc_rows if name and r['type'] == name]
        method = 'exact_type_context_only' if matched else 'no_published_bridge'
    targets = sorted({r['target'].strip() for r in matched if r['target'].strip()})
    specific = sorted(set(targets) - COARSE_TARGETS)
    known_type = row.get('mancType')
    disagreement = bool(known_type and any(known_type not in (r['type'], r['systematic_type']) for r in matched))
    candidate = len(specific) == 1 and len(targets) == 1 and method == 'predicted_raw_mancBodyid' and not disagreement
    return {
        'curated_manc_group_id': curated_group,
        'curated_group_match_count': len(curated_rows),
        'curated_group_targets': curated_targets,
        'curated_specific_target': curated_specific[0] if len(curated_targets) == 1 and len(curated_specific) == 1 else None,
        'match_method': method, 'exported_manc_body_id': bridge,
        'published_manc_rows': [{k: r[k] for k in (
            'bodyid', 'type', 'systematic_type', 'subclass', 'exit_nerve', 'target',
            'synonyms', 'match_certainty(1-5)', 'publication_match', 'line_match', 'match_notes')} for r in matched],
        'published_targets': targets, 'specific_target_candidate': specific[0] if candidate else None,
        'type_disagreement': disagreement,
        'status': 'predicted_candidate_requires_manual_review' if candidate else 'muscle_target_unresolved',
    }


def source(path: Path, url: str | None = None) -> dict:
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'url': url}


def audit(raw_path: Path = RAW, supplement_path: Path = SUPPLEMENT) -> dict:
    import pyarrow.feather as feather
    decoder = LegMuscleDecoder()
    unresolved = decoder.provenance()['unmapped_mn']
    ids = {x['body_id'] for x in unresolved}
    fields = ['bodyId', 'type', 'instance', 'mancType', 'mancBodyid', 'mancGroup', 'mancSerial',
              'group', 'superclass', 'subclass', 'somaSide', 'rootSide', 'somaNeuromere',
              'entryNerve', 'exitNerve', 'synonyms', 'matchingNotes', 'vfbId', 'flywireType',
              'hemibrainType', 'status', 'statusLabel']
    raw = {r['bodyId']: r for r in feather.read_table(raw_path, columns=fields).to_pylist() if r['bodyId'] in ids}
    if set(raw) != ids:
        raise ValueError(f'raw annotation lacks IDs: {sorted(ids-set(raw))}')
    with supplement_path.open(newline='', encoding='utf-8-sig') as fh:
        manc = list(csv.DictReader(fh))
    if not manc or 'systematic_type' not in manc[0]:
        raise ValueError('unexpected MANC supplement format')
    records = []
    for item in unresolved:
        row = raw[item['body_id']]
        if (row['type'] or '') != (item['type_name'] or ''):
            raise ValueError(f'CSR/raw type disagreement: {item["body_id"]}')
        records.append({'leg': item['leg'], 'male_cns_body_id': item['body_id'],
                        'raw_annotations': row, 'literature_crosswalk': crosswalk(row, manc)})
    counts = Counter(r['leg'] for r in records)
    return {
        'schema': 'male-cns-unresolved-muscle-target-audit-1',
        'dataset': 'male-cns:v1.0', 'audit_date': '2026-10-02', 'gpu_used': False,
        'sources': {
            'raw_annotations': source(raw_path, 'https://male-cns.janelia.org/download/'),
            'manc_supplement_3': source(supplement_path, 'https://cdn.elifesciences.org/articles/96084/elife-96084-supp3-v1.csv'),
            'csr_annotations': source(DEFAULT_CSR / 'annotations.npz'),
            'csr_types': source(DEFAULT_CSR / 'tables.json'), 'csr_ids': source(DEFAULT_CSR / 'ids.npy'),
            'decoder_map': source(decoder.map_path),
        },
        'unresolved_count': len(records), 'counts_by_leg': dict(counts),
        'blank_type_count': sum(not r['raw_annotations']['type'] for r in records),
        'numbered_type_count': sum(bool(r['raw_annotations']['type']) for r in records),
        'explicit_manc_bridge_count': sum(r['raw_annotations']['mancBodyid'] is not None for r in records),
        'curated_group_bridge_count': sum(r['raw_annotations']['mancGroup'] is not None for r in records),
        'curated_specific_target_count': sum(r['literature_crosswalk']['curated_specific_target'] is not None for r in records),
        'specific_target_candidate_count': sum(r['literature_crosswalk']['specific_target_candidate'] is not None for r in records),
        'type_disagreement_count': sum(r['literature_crosswalk']['type_disagreement'] for r in records),
        'records': records,
        'policy': ['No connectivity-based muscle target assignment.',
                   'mancGroup is curated; mancBodyid is a predicted match and cannot confirm a muscle target.',
                   'Field semantics: https://natverse.org/malecns/reference/mcns_predict_group.html',
                   'A whole-leg target is not an individual muscle target.',
                   'The entryNerve field in CSR is not the motor axon exitNerve field.',
                   'Candidates require review and are not applied to the decoder by this audit.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--manc-supplement', type=Path, default=SUPPLEMENT)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    report = audit(args.raw, args.manc_supplement)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'unmapped_mn_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('unresolved_count', 'counts_by_leg', 'blank_type_count',
         'numbered_type_count', 'explicit_manc_bridge_count', 'curated_group_bridge_count',
         'curated_specific_target_count', 'specific_target_candidate_count', 'type_disagreement_count')}, indent=2))


if __name__ == '__main__':
    main()
