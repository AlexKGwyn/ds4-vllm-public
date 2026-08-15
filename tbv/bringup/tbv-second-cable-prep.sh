#!/usr/bin/env bash
# tbv-second-cable-prep.sh — one-time prep for the two-cable RDMA topology.
# Run on BOTH boxes:  sudo ~/tbv-second-cable-prep.sh
# Then (after this ran on BOTH boxes):  sudo systemctl restart tbv-roce
#
# Step 1: keep NetworkManager off the second cable's netdev (thunderbolt1+).
#         If NM brings it up, thunderbolt_net grabs the new NHI's hop 1 and
#         the zero-copy payload path can't allocate its ring pair.
# Step 2: unbind tbverbs from the OLD cable's service on the LIVE module.
#         While a box has two native rails, the legacy source-blind control
#         handler cross-matches their HELLOs (both peers are route 0x2 in
#         their own domains) and poisons hopid state — each box must be down
#         to one RDMA rail BEFORE either box reloads tbv-roce.
set -e
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

echo "### step 1: NetworkManager ignore for thunderbolt1+ ###"
tee /etc/NetworkManager/conf.d/99-tbv-zc-second-link.conf >/dev/null <<'EOF'
[keyfile]
unmanaged-devices=interface-name:thunderbolt1;interface-name:thunderbolt2;interface-name:thunderbolt3
EOF
systemctl reload NetworkManager
echo "ok: thunderbolt1+ unmanaged"

echo "### step 2: unbind tbverbs from the IP-link (old) cable ###"
DRV=/sys/bus/thunderbolt/drivers/thunderbolt_ibverbs
IP_SVC=$(readlink -f /sys/class/net/thunderbolt0/device 2>/dev/null | xargs -r basename)
if [ -z "$IP_SVC" ]; then
  echo "!! could not resolve thunderbolt0's service — is thunderbolt0 up?"
  exit 1
fi
IP_XD_PREFIX="${IP_SVC%.*}"   # e.g. 1-2.0 -> 1-2
unbound=0
for svc in "$DRV"/"$IP_XD_PREFIX".*; do
  [ -e "$svc" ] || continue
  name=$(basename "$svc")
  [ "$name" = "$IP_SVC" ] && continue
  if echo "$name" > "$DRV/unbind" 2>/dev/null; then
    echo "ok: unbound tbverbs from $name"
    unbound=$((unbound + 1))
  fi
done
[ "$unbound" = 0 ] && echo "note: no tbverbs services on the IP link (already unbound?)"

echo
echo "remaining tbverbs services (should be the second cable only):"
ls "$DRV" | grep -E '^[0-9]' || echo "(none)"
echo
echo "### done — after this ran on BOTH boxes: sudo systemctl restart tbv-roce ###"
