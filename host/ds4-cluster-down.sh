#!/usr/bin/env bash
# Full DS4 cluster teardown: serve unit -> stranded vllm serve procs -> ray on
# both boxes. The stop half of ds4-cluster-restart.sh, for callers that need the
# stack DOWN rather than restarted (mode flips, `systemctl --user stop
# ds4-vllm`). Idempotent: every step is a no-op when its target is already gone.
#
# The explicit process reap exists because ds4-vllm-manual is a transient unit
# supervising a `podman exec` wrapper, not the vllm serve inside the container:
# stopping the unit alone strands the server, still holding the API port.
set -uo pipefail

# Teardown must work even with a broken/missing config -- fall back to defaults.
eval "$("$HOME/ds4-config" "$HOME/ds4-config.yaml" 2>/dev/null)" 2>/dev/null || true
WORKER_IP=${DS4_WORKER_IP:-192.168.100.2}
CTR=${DS4_CONTAINER:-vllm}
UNIT=ds4-vllm-manual

systemctl --user stop "$UNIT.service" 2>/dev/null
systemctl --user reset-failed "$UNIT.service" 2>/dev/null
sleep 2

# The bracket keeps this grep from matching its own command line.
for p in $(ps -eo pid,cmd --no-headers | grep "bin/[v]llm serve deepseek" | awk '{print $1}'); do
  echo "[cluster-down] reaping vllm serve pid=$p"
  kill "$p" 2>/dev/null; sleep 3
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
done

timeout 60 podman exec -u 1000:1000 -w "$HOME" "$CTR" \
  bash -lc 'ray stop --force >/dev/null 2>&1' 2>/dev/null
timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=6 "$WORKER_IP" \
  "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc 'ray stop --force >/dev/null 2>&1'" 2>/dev/null

echo "[cluster-down] done"
exit 0
