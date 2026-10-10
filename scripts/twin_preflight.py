#!/usr/bin/env python3
"""Read-only preflight for :mod:`twin_runtime`.

The command deliberately performs capability discovery with import metadata
only.  It never imports Torch, CUDA, Triton or FlyVis.  A CUDA request is
reported as blocked when ``malecns_gate.py`` is closed; no model component is
constructed by this command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from twin_runtime import preflight_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "gpu"), default="auto")
    parser.add_argument("--backend", choices=("auto", "cpu", "stub", "cpu_stub", "torch", "triton", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    args = parser.parse_args()
    report = preflight_report(device=args.device, backend=args.backend)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("=== MaleCNS twin pre-flight ===")
    gate = report["gate"]
    print(f"CUDA gate     : {'OPEN' if gate['open'] else 'CLOSED'}")
    if gate["blockers"]:
        for blocker in gate["blockers"]:
            print(f"  - {blocker}")
    print(f"safe default  : {report['safe_default']}")
    print("backends      :")
    for name, value in report["capabilities"].items():
        if not isinstance(value, dict):
            continue
        state = "available" if value.get("usable") else "absent"
        version = f" ({value['version']})" if value.get("version") else ""
        print(f"  {name:<12} {state}{version}")
    print(f"torch imported: {report['torch_imported']}")
    print(f"cuda imported : {report['cuda_imported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
