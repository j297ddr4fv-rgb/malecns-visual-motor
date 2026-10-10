"""Fail-closed safety gate for MaleCNS GPU work on the XPS 13 eGPU.

The 2026-09-15 host freeze happened when a CUDA initialisation made
``nvidia_uvm`` register the 16 GB eGPU as ZONE_DEVICE memory; the kernel could
not allocate contiguous vmemmap blocks (``vmemmap alloc failure: order:9``)
and the machine hard-locked.

Module option ``nvidia_uvm uvm_disable_hmm=1`` makes ``uvm_pmm_devmem_init()``
return early and skips that path.  It is a read-only module parameter, so it
only takes effect after a reboot (or a full nvidia module reload).

Nothing in this project may open a CUDA context while the gate is closed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

UVM_PARAM = Path("/sys/module/nvidia_uvm/parameters/uvm_disable_hmm")
DATA_DIR = Path("/data/datasets/raw/malecns/v1.0")
WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BODY_STATS = DATA_DIR / "body-stats-male-cns-v1.0-minconf-0.5.feather"

GiB = 1024**3

# User-approved operating envelope, 2026-09-21.  The HMM mitigation remains
# mandatory; this only raises the desktop-shared-memory ceiling from the
# original conservative 2 GiB to 5 GiB.  GPU jobs are still launched under a
# separate 5 GiB host-memory cgroup limit.
DEFAULT_MIN_AVAILABLE_GIB = 6.0
DEFAULT_MAX_SHMEM_GIB = 5.0


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        out[key.strip()] = int(rest.split()[0]) * 1024
    return out


def uvm_hmm_disabled() -> bool:
    """True only when the driver has HMM/ZONE_DEVICE registration switched off."""
    try:
        return UVM_PARAM.read_text().strip().upper().startswith("Y")
    except OSError:
        return False


def gpu_query() -> dict[str, str]:
    if not shutil.which("nvidia-smi"):
        return {"nvidia-smi": "missing"}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"nvidia-smi": f"error: {exc}"}
    if proc.returncode != 0:
        return {"nvidia-smi": f"exit {proc.returncode}: {proc.stderr.strip()[:200]}"}
    return {"nvidia-smi": proc.stdout.strip()}


def facts() -> dict[str, object]:
    m = meminfo()
    return {
        "uvm_disable_hmm": "Y" if uvm_hmm_disabled() else "N",
        "uvm_param_path": str(UVM_PARAM),
        "mem_total_gib": round(m.get("MemTotal", 0) / GiB, 2),
        "mem_available_gib": round(m.get("MemAvailable", 0) / GiB, 2),
        "swaptotal_gib": round(m.get("SwapTotal", 0) / GiB, 2),
        "shmem_gib": round(m.get("Shmem", 0) / GiB, 2),
        "unevictable_gib": round(m.get("Unevictable", 0) / GiB, 2),
        "data_weights": WEIGHTS.exists(),
        "data_bytes": WEIGHTS.stat().st_size if WEIGHTS.exists() else 0,
        "gpu": gpu_query(),
    }


def cuda_blockers(
    min_available_gib: float = DEFAULT_MIN_AVAILABLE_GIB,
    max_shmem_gib: float = DEFAULT_MAX_SHMEM_GIB,
) -> list[str]:
    """Reasons CUDA must NOT be initialised right now. Empty list == cleared."""
    m = meminfo()
    reasons: list[str] = []
    if not uvm_hmm_disabled():
        reasons.append(
            "nvidia_uvm uvm_disable_hmm is not Y: the vmemmap/ZONE_DEVICE "
            "registration path that froze the host on 2026-09-15 is still active. "
            "Apply `options nvidia_uvm uvm_disable_hmm=1` and reboot first."
        )
    avail = m.get("MemAvailable", 0) / GiB
    if avail < min_available_gib:
        reasons.append(
            f"MemAvailable {avail:.2f} GiB < required {min_available_gib:.2f} GiB"
        )
    shmem = m.get("Shmem", 0) / GiB
    if shmem > max_shmem_gib:
        reasons.append(f"Shmem {shmem:.2f} GiB > allowed {max_shmem_gib:.2f} GiB")
    if not WEIGHTS.exists():
        reasons.append(f"weights table missing: {WEIGHTS}")
    # A hot-plugged eGPU can leave NVIDIA modules and /dev nodes behind after
    # the PCIe link drops (Xid 79).  The resource checks above then pass even
    # though opening a CUDA context cannot succeed.  Require a live NVML GPU
    # query as part of the fail-closed gate.
    gpu_status = gpu_query().get("nvidia-smi", "")
    if not gpu_status or gpu_status.startswith(("missing", "error:", "exit ")):
        reasons.append(f"nvidia-smi cannot see a usable GPU: {gpu_status or 'empty result'}")
    return reasons


def require_cuda(**kwargs) -> None:
    reasons = cuda_blockers(**kwargs)
    if reasons:
        raise SystemExit(
            "CUDA GATE CLOSED - refusing to open a CUDA context:\n  - "
            + "\n  - ".join(reasons)
        )
