#!/usr/bin/env bash
# build-modules.sh -- build the full tbv module set for a target kernel: the
# callback_xd-patched thunderbolt core, thunderbolt_net, thunderbolt_ibverbs,
# and nhi_throttle. Third-party sources are FETCHED at pinned commits (westeri
# thunderbolt.git for core+net; hellas-ai/thunderbolt-ibverbs for the RDMA
# driver and the kernel patch series), with this repo's local diffs applied on
# top -- nothing third-party is vendored here.
#
#   ./build-modules.sh [KVER]     # default: the running kernel
#
# Needs kernel-devel for KVER (/usr/src/kernels/<KVER> or
# /lib/modules/<KVER>/build), plus git + network on the first run (clones the
# pinned westeri tree into the build cache). No sudo. Outputs land in
# ~/.cache/tbv-build/<KVER>/out under their /var/lib/tbv staging names;
# install-modules.sh stages them.
#
# The patched core+net source is westeri/thunderbolt.git @ BASE (the flake's
# linux-src pin; the 01xx upstream series in kernel-workflow/patches is
# already merged there) plus the local patch series below -- the same
# composition kernel-workflow/patches/*.nix declare, applied with git apply
# -C1 (the vendored patches carry drifted context, like nixpkgs' fuzzy patch).
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BASE=503c5ae1e72aa9ed91925dafa3d82ee2e992747f
REMOTE=https://git.kernel.org/pub/scm/linux/kernel/git/westeri/thunderbolt.git
IBV_BASE=76ba39b630a70accb72f19388eefe48844b50eb8
IBV_REMOTE=https://github.com/hellas-ai/thunderbolt-ibverbs
LOCAL_SERIES="
0002-thunderbolt-tunnel-add-dma-priority-weight-params.patch
0003-thunderbolt-nhi-add-ring-debugfs-instrumentation.patch
0006-thunderbolt-xdomain-bound-response-copy.patch
0004-thunderbolt-nhi-clear-pending-before-unmask.patch
0005-thunderbolt-xdomain-log-unmatched-protocol-uuids.patch
0007-thunderbolt-xdomain-pass-source-to-protocol-handlers.patch
0008-thunderbolt-xdomain-pin-protocol-handler-owner.patch
0009-thunderbolt-xdomain-match-properties-by-identity.patch
0010-thunderbolt-xdomain-drain-protocol-callbacks-on-unr.patch
0009-thunderbolt-xdomain-lane-bonding-module-param.patch
"
KVER=${1:-$(uname -r)}
CACHE=${TBV_BUILD_CACHE:-$HOME/.cache/tbv-build}
WORK=$CACHE/$KVER
OUT=$WORK/out
WESTERI=$CACHE/westeri-thunderbolt
IBVERBS=$CACHE/ibverbs
SERIES_DIR=$IBVERBS/kernel-workflow/patches
CAP="nice -n 19 ionice -c3"

KDEV=""
for d in "/usr/src/kernels/$KVER" "/lib/modules/$KVER/build"; do
  [ -f "$d/Makefile" ] && KDEV=$(readlink -f "$d") && break
done
[ -n "$KDEV" ] || { echo "!! no kernel-devel for $KVER -- install kernel-devel, or extract the rpm and symlink it under /usr/src/kernels/"; exit 1; }
echo "== target kernel: $KVER  (devel: $KDEV) =="

echo "== thunderbolt-ibverbs @ ${IBV_BASE:0:12} + local patch =="
if [ ! -d "$IBVERBS/.git" ]; then
  git clone "$IBV_REMOTE" "$IBVERBS"
fi
git -C "$IBVERBS" cat-file -e "$IBV_BASE^{commit}" 2>/dev/null || git -C "$IBVERBS" fetch origin "$IBV_BASE"
git -C "$IBVERBS" checkout -qf "$IBV_BASE"
git -C "$IBVERBS" clean -qfdx
git -C "$IBVERBS" apply "$REPO/ibverbs-local.patch" || { echo "!! ibverbs-local.patch did not apply"; exit 1; }

echo "== westeri tree @ ${BASE:0:12} + local series (from the ibverbs clone) =="
if [ ! -d "$WESTERI/.git" ]; then
  git clone "$REMOTE" "$WESTERI"
fi
git -C "$WESTERI" cat-file -e "$BASE^{commit}" 2>/dev/null || git -C "$WESTERI" fetch origin "$BASE"
git -C "$WESTERI" checkout -qf "$BASE"
git -C "$WESTERI" clean -qfd
for p in $LOCAL_SERIES; do
  git -C "$WESTERI" apply -C1 "$SERIES_DIR/$p" || { echo "!! $p did not apply"; exit 1; }
done
grep -q callback_xd "$WESTERI/include/linux/thunderbolt.h" || { echo "!! series did not take (no callback_xd in thunderbolt.h)"; exit 1; }

echo "== KDIR (symlink-farm kernel-devel + westeri header overlay + USB4_CONFIGFS=y) =="
rm -rf "$WORK"; mkdir -p "$OUT"
cp -as "$KDEV" "$WORK/kdir"
rm -f "$WORK/kdir/include/linux/thunderbolt.h"
cp "$WESTERI/include/linux/thunderbolt.h" "$WORK/kdir/include/linux/thunderbolt.h"
real=$(readlink -f "$WORK/kdir/include/config/auto.conf")
rm -f "$WORK/kdir/include/config/auto.conf"
cp "$real" "$WORK/kdir/include/config/auto.conf"
grep -q '^CONFIG_USB4_CONFIGFS=y' "$WORK/kdir/include/config/auto.conf" \
  || echo 'CONFIG_USB4_CONFIGFS=y' >> "$WORK/kdir/include/config/auto.conf"

echo "== 1/4 thunderbolt core =="
$CAP make -j1 -C "$WORK/kdir" M="$WESTERI/drivers/thunderbolt" clean >/dev/null 2>&1 || true
$CAP make -j"$(nproc)" -C "$WORK/kdir" M="$WESTERI/drivers/thunderbolt" 2>&1 | tail -3
cp -f "$WESTERI/drivers/thunderbolt/thunderbolt.ko" "$OUT/thunderbolt-patched.ko"

echo "== 2/4 thunderbolt_net (against core symvers) =="
$CAP make -j1 -C "$WORK/kdir" M="$WESTERI/drivers/net/thunderbolt" clean >/dev/null 2>&1 || true
$CAP make -j"$(nproc)" -C "$WORK/kdir" M="$WESTERI/drivers/net/thunderbolt" \
  KBUILD_EXTRA_SYMBOLS="$WESTERI/drivers/thunderbolt/Module.symvers" 2>&1 | tail -3
cp -f "$WESTERI/drivers/net/thunderbolt/thunderbolt_net.ko" "$OUT/thunderbolt_net.ko"

echo "== 3/4 thunderbolt_ibverbs =="
# built in the (cleaned, patched) clone; kernel/ resolves proto/*.h via its parent
$CAP make -C "$IBVERBS/kernel" KDIR="$WORK/kdir" modules 2>&1 | tail -3
cp -f "$IBVERBS/kernel/thunderbolt_ibverbs.ko" "$OUT/thunderbolt_ibverbs.ko"
aware=$(strings "$OUT/thunderbolt_ibverbs.ko" | grep -c 'source-aware XDomain handler' || true)
[ "$aware" -ge 1 ] || { echo "!! ibverbs built legacy-only (patched header did not take)"; exit 1; }

echo "== 4/4 nhi_throttle =="
rm -rf "$WORK/nhi"; cp -r "$REPO/nhi-throttle-mod" "$WORK/nhi"
# The Makefile passes M=$(PWD), so it must be built from inside its dir.
(cd "$WORK/nhi" && $CAP make KDIR="$WORK/kdir" 2>&1 | tail -3)
cp -f "$WORK/nhi/nhi_throttle.ko" "$OUT/nhi_throttle.ko"

echo "== module set for $KVER ==" | tee "$OUT/MANIFEST"
for m in "$OUT"/*.ko; do
  printf "  %-28s %s\n" "$(basename "$m")" "$(modinfo -F vermagic "$m")" | tee -a "$OUT/MANIFEST"
done
echo "install: sudo $REPO/install-modules.sh $KVER"
