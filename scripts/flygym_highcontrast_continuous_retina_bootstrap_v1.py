#!/usr/bin/env python3
"""Install the pinned C=1 retina schedule adapter, then start the worker."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
WORKER = SCRIPTS / "flygym_online_neural_worker_eye_parity_state_v2.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def main() -> int:
    if len(sys.argv) < 5:
        raise SystemExit("usage: bootstrap.py WORKER SCHEDULE_PATH SCHEDULE_SHA256 CASE_ID [worker arguments ...]")
    worker = Path(sys.argv[1]).expanduser().resolve(strict=True)
    if worker != WORKER.resolve(strict=True):
        raise SystemExit(f"worker path differs from the pinned state-v2 worker: {worker}")
    schedule_path, schedule_sha256, case_id = sys.argv[2:5]
    worker_args = sys.argv[5:]
    if not worker_args or worker_args[0] != "--config":
        raise SystemExit("worker arguments must start with --config")
    os.environ["FLYGYM_HIGHCONTRAST_SCHEDULE_PATH"] = schedule_path
    os.environ["FLYGYM_HIGHCONTRAST_SCHEDULE_SHA256"] = schedule_sha256
    os.environ["FLYGYM_HIGHCONTRAST_CASE_ID"] = case_id
    import flygym_highcontrast_continuous_retina_v1 as retina_adapter

    retina_adapter.install_worker_renderer()
    sys.argv = [str(worker), *worker_args]
    runpy.run_path(str(worker), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
