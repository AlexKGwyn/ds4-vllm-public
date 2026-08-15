#!/usr/bin/env bash
# container-heal.sh <container> — verify a podman container is actually
# exec-able, reconciling the wedged state where podman's DB says "Up" but the
# crun container is dead (conmon SIGKILLed).
# The cluster bringup runs this on both boxes so a wedged container heals
# automatically.
# Exits 0 always: the main unit command does its own retrying.
set -u
c="${1:?usage: container-heal.sh <container>}"

if podman exec "$c" true 2>/dev/null; then
    exit 0   # healthy
fi

if podman ps --format '{{.Names}}' | grep -qx "$c"; then
    echo "[container-heal] $c wedged (podman says Up, exec dead) — reconciling"
    podman stop -t 0 "$c" >/dev/null 2>&1   # expect conmon-exited noise; fine
    sleep 1
fi

# stopped (or just reconciled): start it so toolbox/distrobox exec succeeds
podman start "$c" >/dev/null 2>&1
sleep 2
if podman exec "$c" true 2>/dev/null; then
    echo "[container-heal] $c healed"
else
    echo "[container-heal] $c still not exec-able after reconcile (will retry on next unit restart)"
fi
exit 0
