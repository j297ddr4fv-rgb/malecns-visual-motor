#!/usr/bin/env bash
# Apply the one-time mitigation for the 2026-09-15 eGPU host freeze.
#
#   nvidia_uvm uvm_disable_hmm=1  ->  uvm_pmm_devmem_init() returns early and
#   skips memremap_pages()/ZONE_DEVICE registration, which is the allocation
#   that failed (`vmemmap alloc failure: order:9`) and hard-locked the host.
#
# Requires: sudo, then a manual reboot.  The script never reboots for you.
set -euo pipefail

CONF=/etc/modprobe.d/nvidia-uvm-hmm.conf
LINE='options nvidia_uvm uvm_disable_hmm=1'

if [[ ${EUID} -ne 0 ]]; then
  echo "run with sudo: sudo $0" >&2
  exit 1
fi

echo "==> writing $CONF"
printf '# Disable UVM HMM/ZONE_DEVICE registration on the RTX 5060 Ti eGPU.\n' > "$CONF"
printf '# Reason: vmemmap alloc failure (order:9) in uvm_pmm_devmem_init froze\n' >> "$CONF"
printf '# the host on 2026-09-15 on this 15 GiB machine. See env registry\n' >> "$CONF"
printf '# xps13-rtx5060ti-egpu-cuda.md and $HOME/Projects/malecns-gpu/.\n' >> "$CONF"
printf '%s\n' "$LINE" >> "$CONF"
cat "$CONF"

echo "==> rebuilding initramfs"
update-initramfs -u

echo
echo "==> current live value (will only change after reboot)"
cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm

echo
echo "Next: reboot yourself, then verify with"
echo "  cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm   # expect Y"
echo "  python3 ~/Projects/malecns-gpu/scripts/preflight.py     # expect gate OPEN"
echo "  python3 ~/Projects/malecns-gpu/scripts/load_edges.py --device cuda --max-batches 1"
