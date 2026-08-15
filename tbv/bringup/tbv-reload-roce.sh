#!/usr/bin/env bash
# tbv-reload-roce.sh — load thunderbolt_ibverbs WITH roce_netdev=thunderbolt0 so the
# RoCE GID table gets populated (needed for perftest/RCCL QP connect).
# Run on BOTH boxes:  sudo /var/lib/tbv/tbv-reload-roce.sh
set +e
HOME_DIR=$(eval echo "~${SUDO_USER:-$USER}")
KO_SRC="$HOME_DIR/tbv/thunderbolt-ibverbs/kernel/thunderbolt_ibverbs.ko"
THR_SRC="$HOME_DIR/tbv/nhi-throttle-mod/nhi_throttle.ko"
# SELinux (enforcing): a systemd service context may not load modules labeled
# user_home_t. Stage the .kos into /var/lib/tbv with modules_object_t; re-copy
# every run so freshly rebuilt modules in $HOME are picked up automatically.
mkdir -p /var/lib/tbv
# A source-tree build overrides the staged copy; without one (built-stack
# install), the modules already in /var/lib/tbv are used as-is.
[ -f "$KO_SRC" ] && cp -f "$KO_SRC" /var/lib/tbv/thunderbolt_ibverbs.ko
[ -f "$THR_SRC" ] && cp -f "$THR_SRC" /var/lib/tbv/nhi_throttle.ko
[ -f /var/lib/tbv/thunderbolt_ibverbs.ko ] || { echo "no thunderbolt_ibverbs.ko staged in /var/lib/tbv (run install-modules.sh first)"; exit 1; }
chcon -t modules_object_t /var/lib/tbv/*.ko 2>/dev/null || true
KO=/var/lib/tbv/thunderbolt_ibverbs.ko
echo "### reload with roce_netdev=thunderbolt0  ($(uname -n) $(uname -r)) ###"
ip -br addr show thunderbolt0 >/dev/null 2>&1 || { echo "!! thunderbolt0 missing — need thunderbolt_net loaded"; exit 1; }

# remove stale instance if present
rmmod thunderbolt_ibverbs 2>/dev/null
lsmod | grep -q '^thunderbolt_ibverbs' && { echo "!! old module stuck"; exit 1; }

# load dependencies. NOTE: need `-a` to load MULTIPLE modules; without it modprobe
# treats extra names as params to the first module (the old bug: ib_uverbs never loaded,
# insmod then failed on Unknown symbol ib_umem_get/ib_umem_release/ib_umem_copy_from).
modprobe -a configfs ib_core ib_uverbs 2>&1
for m in ib_core ib_uverbs; do
  lsmod | grep -qw "$m" || { echo "!! dependency $m failed to load — try: sudo modprobe $m"; exit 1; }
done
echo "deps ok: configfs ib_core ib_uverbs loaded"

insmod "$KO" \
  profile=linux_perf bind_services=1 allocate_rings=1 start_rings=1 \
  negotiate_native=1 enable_tunnels=1 register_verbs=1 \
  native_tx_max_inflight=128 \
  native_rc_split_zcopy=1 \
  roce_netdev=thunderbolt0 2>&1 || { echo "insmod failed — check: sudo dmesg | grep thunderbolt_ibverbs | tail"; exit 1; }
  # native_tx_max_inflight=128 (default 32): avoids the >=1MB WRITE-BW
  # collapse from TX-ring in-flight starvation.
  # native_rc_split_zcopy=1: >=8K host-MR RC WRITEs go header-framed
  # multi-frame-chunk zero-copy (chunk_frames=16, min_bytes=8192 defaults).
  # BOTH boxes must run matching builds (proto F_SPLIT_PAYLOAD cap); remove
  # this line to fall back to the copied fastpath, which needs no cross-box
  # coordination.

# Single-cable topology: the one USB4 cable's NHI holds control + tbnet-IP
# (thunderbolt0) + the tbv main RDMA path (both DMA ring pairs used). There is
# no free ring for the zero-copy payload path, so the driver auto-falls-back:
# the rail doesn't advertise ZC_PATH, no zc negotiation happens, and RDMA
# WRITE payload uses the classic single-path raw stream with the framed RX
# copy. TX stays zero-copy (GPU dma-buf MRs). RX zero-copy is off by design
# here (it needs a 2nd cable's NHI); measured CPU-neutral, so no loss.
sleep 3

D=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
[ -z "$D" ] && { echo "!! no infiniband device appeared after insmod — check: sudo dmesg | grep thunderbolt_ibverbs | tail"; exit 1; }
if [ "$D" != usb4_rdma0 ] && [ "$D" != usb4_rdma5 ]; then
  rdma dev set "$D" name usb4_rdma0 2>&1 && D=usb4_rdma0
fi
echo "device: $D"
echo "=== rdma link ==="; rdma link 2>/dev/null
echo "=== GID table (should be NON-zero for RoCE to work) ==="
for i in 0 1 2 3; do echo "gid[$i]=$(cat /sys/class/infiniband/$D/ports/1/gids/$i 2>/dev/null)"; done
echo "### done — verify with: ibv_devices ###"

# NHI interrupt throttle fix: mainline hardcodes 128us IRQ moderation -> 65us
# RDMA latency floor. Set 8us (measured 8.5us typical after fix).
# ALWAYS re-apply: the module only patches NHIs that are awake at insmod time,
# and a previously-suspended NHI (e.g. the second cable's) would otherwise
# keep the stock 128us throttle across tbv reloads.
rmmod nhi_throttle 2>/dev/null
insmod /var/lib/tbv/nhi_throttle.ko ns=8000 2>/dev/null \
  && echo "nhi_throttle: 8us set on all active NHIs" \
  || echo "!! nhi_throttle load failed"

# TX raw zero-copy: engage for non-retryable (UC) writes >= 4096B. dma-buf
# (GPU) MRs stream pre-mapped pages DMA-direct (no CPU read of WC memory);
# host MRs dma_map their pages. Runtime-revertible: echo 0 > the param.
echo 4096 > /sys/module/thunderbolt_ibverbs/parameters/zcopy_min_bytes 2>/dev/null \
  && echo "zcopy_min_bytes=4096" || echo "!! zcopy_min_bytes set failed"
