#!/usr/bin/env python3
"""Discover the full suite while preventing optional GPU runtime imports."""
from __future__ import annotations

import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GPU_MODULES = frozenset(('torch', 'triton', 'warp'))


class NoGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in GPU_MODULES:
            raise ModuleNotFoundError(
                'Optional GPU-library imports blocked during CPU-only verification', name=fullname)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    if args.report is not None:
        if args.report.exists():
            parser.error('--report exists; refusing to overwrite')
        if not args.report.parent.is_dir():
            parser.error('--report parent directory must already exist')
    if GPU_MODULES.intersection(sys.modules):
        raise RuntimeError('Start CPU verification in a fresh interpreter without GPU libraries')
    sys.meta_path.insert(0, NoGpuImports())
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=0 if args.quiet else 1).run(suite)
    imported = sorted(GPU_MODULES.intersection(sys.modules))
    success = result.wasSuccessful() and not imported
    report = {
        'status': 'PASSED' if success else 'FAILED',
        'test_policy': 'full_discovery_with_optional_gpu_library_imports_blocked',
        'tests_run': result.testsRun, 'skipped_count': len(result.skipped),
        'skipped': [{'test': test.id(), 'reason': reason} for test, reason in result.skipped],
        'failure_count': len(result.failures), 'error_count': len(result.errors),
        'gpu_runtime_imported': imported, 'gpu_numerical_validation': 'PENDING',
    }
    if args.report is not None:
        with args.report.open('x') as fh:
            json.dump(report, fh, indent=2, allow_nan=False)
            fh.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
