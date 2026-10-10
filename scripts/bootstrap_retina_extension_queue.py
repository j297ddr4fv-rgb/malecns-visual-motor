#!/usr/bin/env python3
"""Bind the shared pixel schedule and start the resident neural worker."""
import os
from pathlib import Path
import runpy
import sys
from functools import partial


def main():
    worker, schedule, digest, case_id, *argv = sys.argv[1:]
    expected = Path(__file__).with_name("flygym_online_neural_worker_eye_parity_state_v2.py")
    if Path(worker).resolve() != expected.resolve():
        raise ValueError("unexpected neural worker")
    os.environ.update(FLYGYM_EXTENSION_SCHEDULE_PATH=schedule,
                      FLYGYM_EXTENSION_SCHEDULE_SHA256=digest,
                      FLYGYM_EXTENSION_CASE_ID=case_id,
                      RESEARCH_QUEUE_PREFIX_RETINA="1")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, "/home/xvhang/Projects/scratch/malecns-visual-illusions")
    import malecns_gate
    import pixel_illusion_adapter
    malecns_gate.cuda_blockers = partial(malecns_gate.cuda_blockers, min_available_gib=0.0)
    pixel_illusion_adapter.install_worker_renderer()
    sys.argv = [worker, *argv]
    runpy.run_path(worker, run_name="__main__")


if __name__ == "__main__":
    main()
