#!/usr/bin/env python3
"""Compile retained MJCFs without stepping; compare their complete MJB bytes.

This is an initialization forensic experiment, not a steering-dose run.
It does not modify an existing model identity gate or its failed records.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/data/artifacts/by-project/malecns-gpu/engineering/feedback-diagnosis-20261007-v1')
REFERENCE = BASE / 'mechanical-model-initialization-audit-v1/compiled_model.xml'
ACTUAL = BASE / 'mechanical-steering-dose-calibration-v1/revision-3/runs/seed1729_dzero_0p00/compiled_model.xml'
OUT = BASE / 'mechanical-steering-dose-calibration-v1/revision-3/compiled-identity-forensics-v1'
PINS = {str(REFERENCE): 'bc7a58abf929f7260eb3afd67d88c5184932711d80188abe3866c53848a16d57',
        str(ACTUAL): '976e72a119b25b591e8b88aac63c2a5f06913013ec787330ae727f0f8be432fd'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from run_flygym_steering_dose_calibration_v1 import cgroup_attestation
    started = time.monotonic()
    before = cgroup_attestation()
    for path, expected in PINS.items():
        if sha(Path(path)) != expected:
            raise ValueError(f'input changed: {path}')
    OUT.mkdir(exist_ok=False)
    record = {'schema_version': 'compiled-model-byte-identity-forensics-v1',
              'status': 'started', 'source_sha256': sha(Path(__file__)),
              'inputs': PINS, 'sys_executable': sys.executable,
              'command': sys.argv, 'cgroup_before': before,
              'integration_steps': 0, 'body_control_ticks': 0,
              'gpu_used': False, 'existing_gate_modified': False}
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        import mujoco
        import numpy as np
        record['mujoco_version'] = mujoco.__version__
        record['numpy_version'] = np.__version__
        models = {}
        binaries = {}
        for name, path in [('reference', REFERENCE), ('actual', ACTUAL)]:
            model = mujoco.MjModel.from_xml_path(str(path))
            models[name] = model
            binary = OUT / (name + '.mjb')
            mujoco.mj_saveModel(model, str(binary), None)
            binaries[name] = {'path': str(binary), 'sha256': sha(binary),
                              'size_bytes': binary.stat().st_size}
        a, b = models['reference'], models['actual']
        compared, different = [], []
        for key in dir(a):
            if key.startswith('_'):
                continue
            x, y = getattr(a, key), getattr(b, key)
            if not isinstance(x, np.ndarray):
                continue
            equal = (isinstance(y, np.ndarray) and x.dtype == y.dtype
                     and x.shape == y.shape and x.tobytes() == y.tobytes())
            compared.append(key)
            if not equal:
                different.append(key)
        same = binaries['reference']['sha256'] == binaries['actual']['sha256']
        record.update(status='complete', binary_models=binaries,
                      complete_saved_mjb_bytes_identical=same,
                      model_array_fields_compared=compared,
                      model_array_fields_different=different,
                      model_array_bytes_identical=not different,
                      interpretation=('Retained different XMLs compile to identical complete saved MJB bytes.'
                                      if same else 'Complete compiled model byte identity is not established.'),
                      limits=['No dynamics or dose calibration was run.',
                              'This comparison covers these two retained XMLs and this MuJoCo version.',
                              'Earlier revision-2 differing XML was not retained.',
                              'A revised runtime identity gate requires separate review; failed cohorts remain failed.'])
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(cgroup_after=cgroup_attestation(),
                      elapsed_wall_s=time.monotonic() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        with (OUT / 'analysis.json').open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(OUT / 'analysis.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
