#!/usr/bin/env bash
# tbv-roce-boot.sh — boot wrapper for tbv-reload-roce.sh (runs as root from
# tbv-roce.service). Waits for the thunderbolt0 link to negotiate and get its
# 192.168.100.x address (only happens once the USB4 peer box is up), then runs
# the RoCE bring-up + NHI interrupt-throttle. Retries a few times.

# RDMA latency-tail mitigation: irqbalance
# migrates the NHI MSI-X IRQs between cores, adding tail — keep it off.
# (C3-disable was tested too but with the poll-mode RX driver fix it only
# buys ~1us at the 99.9th percentile for ~10C of package heat — not worth it.)
echo "[tbv-roce-boot] stopping irqbalance for stable NHI IRQ affinity"
systemctl stop irqbalance 2>/dev/null || true

echo "[tbv-roce-boot] waiting for thunderbolt0 with a 192.168.100.x address..."
for i in $(seq 1 120); do
  if ip -4 addr show thunderbolt0 2>/dev/null | grep -q 'inet 192\.168\.100\.'; then
    echo "[tbv-roce-boot] thunderbolt0 up: $(ip -4 -br addr show thunderbolt0)"
    break
  fi
  sleep 2
done

for attempt in 1 2 3; do
  echo "[tbv-roce-boot] bring-up attempt $attempt"
  if /var/lib/tbv/tbv-reload-roce.sh; then
    # confirm the ibverbs device actually appeared
    if ls /sys/class/infiniband/ 2>/dev/null | grep -q usb4_rdma; then
      echo "[tbv-roce-boot] success"
      exit 0
    fi
  fi
  echo "[tbv-roce-boot] attempt $attempt did not yield usb4_rdma; retry in 5s"
  sleep 5
done
echo "[tbv-roce-boot] FAILED to bring up usb4_rdma after 3 attempts"
exit 1
