#!/usr/bin/env bash
# install-modules.sh -- stage a build-modules.sh output set into /var/lib/tbv
# and install the boot plumbing: stock-driver blacklist, the patched-core boot
# unit, tbv-roce, and the ~/tbv boot-chain scripts. Idempotent; run with sudo
# on the box being set up.
#
#   sudo ./install-modules.sh [KVER]    # default: the running kernel
#
# Left as documented manual steps (site-dependent): fix-memlock.sh (RDMA
# memlock limits) and bringup/60-rdma-persistent-naming.rules (disable udev
# RDMA renaming when the tbv rails are the only RDMA devices).
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

REPO=$(cd "$(dirname "$0")" && pwd)
HOME_DIR=$(eval echo "~${SUDO_USER:-$USER}")   # build cache lives in the invoking user's home
KVER=${1:-$(uname -r)}
# cables: from the deployed site config, else the repo default (1 = single cable)
CFG=$HOME_DIR/ds4-config.yaml; [ -f "$CFG" ] || CFG=$REPO/../host/ds4-config.yaml
CABLES=$(sed -n 's/^cables:[[:space:]]*\([0-9]\).*/\1/p' "$CFG" 2>/dev/null | head -1)
CABLES=${CABLES:-1}
OUT=${TBV_BUILD_CACHE:-$HOME_DIR/.cache/tbv-build}/$KVER/out

MODS="thunderbolt-patched.ko thunderbolt_net.ko thunderbolt_ibverbs.ko nhi_throttle.ko"
if [ -d "$OUT" ]; then
  echo "== stage modules ($OUT -> /var/lib/tbv) =="
  for m in $MODS; do
    [ -f "$OUT/$m" ] || { echo "!! $OUT/$m missing -- run build-modules.sh $KVER first"; exit 1; }
  done
  mkdir -p /var/lib/tbv
  cp -f "$OUT"/*.ko /var/lib/tbv/
else
  # No build output on this box: fine as long as a full set is already staged
  # (scripts/units still get [re]installed below).
  echo "== no build output at $OUT -- keeping the already-staged modules =="
  for m in $MODS; do
    [ -f "/var/lib/tbv/$m" ] || { echo "!! /var/lib/tbv/$m missing too -- run build-modules.sh $KVER first"; exit 1; }
  done
fi
chcon -t modules_object_t /var/lib/tbv/*.ko 2>/dev/null || true
for m in /var/lib/tbv/*.ko; do
  printf "   %-28s %s\n" "$(basename "$m")" "$(modinfo -F vermagic "$m")"
done

echo "== blacklist stock thunderbolt (patched pair loads via the boot unit) =="
cat > /etc/modprobe.d/00-tbv-thunderbolt.conf <<'EOF'
# tbv: stock thunderbolt is replaced at boot by the callback_xd-patched build
# loaded via tbv-thunderbolt-patched.service. Remove this file + the service to
# revert. The install lines stop udev from autoloading the stock modules (stock
# net over the patched core has mismatched ABI and panics on cable connect);
# the boot unit uses modprobe -i to bypass them when it deliberately falls back
# to the stock pair.
blacklist thunderbolt
install thunderbolt /bin/false
install thunderbolt_net /bin/false
EOF
if command -v grubby >/dev/null; then
  grubby --update-kernel=ALL --args="rd.driver.blacklist=thunderbolt modprobe.blacklist=thunderbolt"
else
  echo "   (no grubby -- add 'rd.driver.blacklist=thunderbolt modprobe.blacklist=thunderbolt' to the kernel args manually)"
fi

echo "== boot-chain scripts -> /var/lib/tbv =="
install -m 755 "$REPO/bringup/tbv-roce-boot.sh" "$REPO/bringup/tbv-reload-roce.sh" /var/lib/tbv/

if [ "$CABLES" = 2 ]; then
  echo "== second cable: keep tb1+ NM-unmanaged (frees its NHI rings for the RX zero-copy rail) =="
  cp -f "$REPO/bringup/99-tbv-zc-second-link.conf" /etc/NetworkManager/conf.d/
  nmcli con reload >/dev/null 2>&1 || true
  echo "   one-time HopID/persist prep for the second cable: $REPO/bringup/tbv-second-cable-prep.sh"
fi

echo "== systemd units =="
cp -f "$REPO/systemd/tbv-roce.service" /etc/systemd/system/
cp -f "$REPO/systemd/tbv-thunderbolt-patched.service" /etc/systemd/system/
if [ -d "$REPO/systemd/tbv-thunderbolt-patched.service.d" ]; then
  mkdir -p /etc/systemd/system/tbv-thunderbolt-patched.service.d
  cp -f "$REPO/systemd/tbv-thunderbolt-patched.service.d/"* /etc/systemd/system/tbv-thunderbolt-patched.service.d/
fi
systemctl daemon-reload
systemctl enable tbv-thunderbolt-patched.service tbv-roce.service

echo
echo "installed. One-time site setup still needed if not done:"
echo "  # autoconnect IP on thunderbolt0 BEFORE ibverbs claims the DMA rings:"
echo "  sudo nmcli con add type ethernet ifname thunderbolt0 con-name tbv-tb0 \\"
echo "    ipv4.method manual ipv4.addresses <192.168.100.1 or .2>/24 ipv4.may-fail no \\"
echo "    ipv6.method disabled connection.autoconnect yes connection.autoconnect-priority 100"
echo "  sudo $REPO/bringup/fix-memlock.sh                    # RDMA memlock limits"
echo "  sudo cp $REPO/bringup/60-rdma-persistent-naming.rules /etc/udev/rules.d/"
echo "Then reboot BOTH boxes together (clean load order) and verify:"
echo "  ls /sys/class/infiniband/   # expect usb4_rdma devices"
