#!/usr/bin/env python3
"""Read-only pre-flight report. Never imports torch, never touches CUDA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import malecns_gate as gate  # noqa: E402


def main() -> int:
    f = gate.facts()
    blockers = gate.cuda_blockers()
    print("=== MaleCNS pre-flight ===")
    for key, value in f.items():
        print(f"{key:>18}: {value}")
    print()
    if blockers:
        print("CUDA gate: CLOSED")
        for r in blockers:
            print(f"  - {r}")
        print("\nCPU streaming path is available and safe right now:")
        print("  scripts/load_edges.py --device cpu")
    else:
        print("CUDA gate: OPEN (uvm_disable_hmm=Y, memory headroom ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
