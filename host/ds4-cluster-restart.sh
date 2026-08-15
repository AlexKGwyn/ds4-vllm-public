#!/usr/bin/env bash
# Full DS4 cluster restart: teardown -> Ray on both boxes -> vllm serve -> verify.
#
# Run from box1 (the Ray head); box2 is driven over ssh. Each box's vllm
# container is verified exec-able (container-heal.sh starts/reconciles it)
# before Ray comes up, so this works from a fresh boot.
#
# Two things this encodes that a bare `systemctl restart` gets wrong:
#
#   1. REAPING. `ds4-vllm-manual` is a transient systemd-run unit supervising a
#      `podman exec` wrapper, NOT the process inside the container. Stopping the
#      unit kills the wrapper and leaves `vllm serve` running, holding its port
#      and ~0.4 GB. Repeated restarts strand one husk each. We stop the unit,
#      then explicitly kill any surviving `vllm serve` and confirm zero remain
#      before bringing anything back up.
#
#   2. RAY WORKER POOL. `ray start` passes --num_prestart_python_workers=<num_cpus>
#      straight through (ray/_private/services.py:1980) -- there is no env
#      override. Unbounded, box1 gets 32 idle Python workers at ~40 MB each,
#      ~1.28 GB that vLLM never touches: its workers are RayWorkerProc *actors*,
#      and `ray status` reports 0.0/64.0 CPU with both GPUs reserved, i.e. the
#      placement groups request no logical CPU at all. RAY_NUM_CPUS below caps
#      the pool. The dashboard is off for the same reason -- nothing reads it.
#
# Context/batch size live in ds4-vllm-manual-serve.sh; the KV pin, GPU
# ceiling, and disk-KV sizing come from ds4-config.yaml.
set -uo pipefail

# Site specifics (IPs, transport, container name, HCA pin) come from
# ~/ds4-config.yaml; everything else is fixed stack layout.
eval "$("$HOME/ds4-config" "$HOME/ds4-config.yaml")"
HEAD_IP=${DS4_HEAD_IP:?ds4-config.yaml: head_ip missing}
WORKER_IP=${DS4_WORKER_IP:?ds4-config.yaml: worker_ip missing}
PORT=${DS4_API_PORT:-1234}
CTR=${DS4_CONTAINER:-vllm}
TRANSPORT=${DS4_TRANSPORT:-rdma}
RAYTMP=$HOME/ray-tmp
RAY_NUM_CPUS=${RAY_NUM_CPUS:-4}
CENV=$HOME/ds4-cluster-env.$TRANSPORT.sh
SERVE=$HOME/ds4-vllm-manual-serve.sh
UNIT=ds4-vllm-manual
# Exports that must reach the env files on BOTH boxes (sourced at ray start).
ENVPASS="export DS4_RDMA_HCA=${DS4_RDMA_HCA:-};"
[ -f "$CENV" ] || { echo "!! $CENV missing (transport=$TRANSPORT)"; exit 1; }
# Memory + disk-KV knobs: yaml -> the DS4_* the serve script reads.
case "${DS4_DISK_KV:-true}" in true|1|on|yes) DISK_KV=1;; *) DISK_KV=0;; esac
DISK_KV_BYTES=$(( ${DS4_DISK_KV_GIB:-30} * 1024*1024*1024 ))
KV_BYTES=$(( ${DS4_KV_PIN_GIB:-6} * 1024*1024*1024 ))
GPU_UTIL=${DS4_GPU_MEM_UTIL:-0.83}
WARMUP_CTX=${DS4_WARMUP_CTX:-2048}
MODEL=${DS4_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}
MAX_CTX=${DS4_MAX_CTX:-524288}

box2() { timeout "${2:-120}" ssh -o BatchMode=yes "$WORKER_IP" "$1"; }
inbox() { timeout "${2:-120}" podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc "$1"; }

echo "== teardown =="
systemctl --user stop "$UNIT.service" 2>/dev/null
systemctl --user reset-failed "$UNIT.service" 2>/dev/null
sleep 4

# The bracket keeps this grep from matching its own command line.
for p in $(ps -eo pid,cmd --no-headers | grep "bin/[v]llm serve deepseek" | awk '{print $1}'); do
  echo "   reaping stranded vllm serve pid=$p"
  kill "$p" 2>/dev/null; sleep 3
  kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
done
residual=$(ps -eo cmd --no-headers | grep -c "bin/[v]llm serve deepseek")
[ "$residual" -eq 0 ] || { echo "!! $residual vllm serve process(es) still alive -- aborting"; exit 1; }

inbox 'ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
box2 "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc 'ray stop --force >/dev/null 2>&1'" >/dev/null 2>&1

for _ in $(seq 1 40); do
  u1=$(free -g | awk '/^Mem:/{print $3}')
  u2=$(box2 "free -g | awk '/^Mem:/{print \$3}'" 20 2>/dev/null || echo 99)
  [ "${u1:-99}" -lt 45 ] && [ "${u2:-99}" -lt 45 ] && break
  sleep 5
done
echo "   drained: box1=${u1}G box2=${u2}G swap=$(free -m | awk '/^Swap:/{print $3}')MB"

echo "== containers =="
# Nothing starts the containers at boot on its own, so heal both here.
"$HOME/container-heal.sh" "$CTR" 2>&1 | sed 's/^/   /'
box2 "\$HOME/container-heal.sh $CTR" 60 2>/dev/null | sed 's/^/   /'
inbox true 20 >/dev/null 2>&1 || { echo "!! box1 $CTR container not exec-able"; exit 1; }
box2 "podman exec $CTR true" 20 >/dev/null 2>&1 || { echo "!! box2 $CTR container not exec-able"; exit 1; }
echo "   $CTR container exec-able on both boxes"

echo "== ray =="
# --include-dashboard is head-only; ray PANICs if it is passed to a worker.
RAYFLAGS="--num-gpus=1 --num-cpus=$RAY_NUM_CPUS --temp-dir=$RAYTMP"
HEADFLAGS="$RAYFLAGS --include-dashboard=false"
if ! out=$(inbox "$ENVPASS source $CENV; ray start --head --node-ip-address=$HEAD_IP --port=6379 $HEADFLAGS" 180 2>&1); then
  echo "!! box1 ray start failed:"; echo "$out" | tail -5 | sed 's/^/     /'; exit 1
fi
echo "   box1 head up"
if ! out=$(box2 "podman exec -u 1000:1000 -w \$HOME $CTR bash -lc '$ENVPASS source $CENV; ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP $RAYFLAGS'" 180 2>&1); then
  echo "!! box2 ray start failed:"; echo "$out" | tail -5 | sed 's/^/     /'; exit 1
fi
echo "   box2 worker up"

for _ in $(seq 1 8); do
  g=$(inbox 'ray status 2>/dev/null' 60 | grep -oE "[0-9.]+/[0-9.]+ GPU")
  [ "${g#*/}" = "2.0 GPU" ] && break
  sleep 4
done
[ "${g#*/}" = "2.0 GPU" ] || { echo "!! ray never reached 2 GPUs (saw '${g:-none}') -- aborting"; exit 1; }
echo "   ray: $g  (prestart python workers capped at $RAY_NUM_CPUS/node, dashboard off)"

echo "== serve =="
systemd-run --user --unit="$UNIT" --description="DS4 vLLM TP=2 (disk KV)" \
  --working-directory="$HOME" \
  /usr/bin/podman exec -u 1000:1000 -w "$HOME" "$CTR" bash -lc \
  "$ENVPASS export DS4_TRANSPORT=$TRANSPORT DS4_DISK_KV=$DISK_KV DS4_DISK_KV_BYTES=$DISK_KV_BYTES DS4_KV_BYTES=$KV_BYTES DS4_GPU_UTIL=$GPU_UTIL DS4_MODEL=$MODEL DS4_API_PORT=$PORT DS4_MAX_CTX=$MAX_CTX; exec bash $SERVE" >/dev/null 2>&1

# Warm bringup answers in ~4 min; a cold kernel-cache bringup (first after a
# cache wipe) spends ~25 min more in Triton/LLVM compiles before the API is up.
for _ in $(seq 1 105); do
  code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && break
  systemctl --user is-active --quiet "$UNIT.service" || { echo "!! unit died during boot"; exit 1; }
  sleep 20
done
[ "$code" = "200" ] || { echo "!! API never came up"; exit 1; }

# Warm the JIT kernels, the bf16 weight cache, and the prefill-indexer buckets
# so the first real request runs at full prefill speed. Backgrounded transient
# unit: bringup finishes now, warmup follows in ~a minute. Best-effort.
systemctl --user reset-failed ds4-vllm-warmup.service 2>/dev/null
systemd-run --user --collect --unit=ds4-vllm-warmup \
  --setenv=DS4_VLLM_PORT=$PORT --setenv=DS4_WARMUP_CTX=$WARMUP_CTX \
  /usr/bin/python3 "$HOME/ds4-vllm-warmup.py" >/dev/null 2>&1 \
  && echo "   warmup dispatched (ctx=$WARMUP_CTX; journalctl --user -u ds4-vllm-warmup)" \
  || echo "   warmup dispatch failed (non-fatal)"

echo "== verify =="
journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-20min" 2>/dev/null \
  | grep -aE "GPU KV cache size|Maximum concurrency" | tail -2 | sed 's/^/   /'
rdma=$(journalctl --user -u "$UNIT.service" --no-pager -o cat --since "-20min" 2>/dev/null \
  | grep -aoE "tbv_ar2: rank[0-9] ready \(qpn=[0-9]+ peer_qpn=[0-9]+\)" | head -1)
echo "   RDMA: ${rdma:-!! tbv_ar2 NOT ready -- decode all-reduce is not on RDMA}"
echo "   vllm serve procs: $(ps -eo cmd --no-headers | grep -c 'bin/[v]llm serve deepseek') (want 1)"
echo "   ray idle workers: $(ps -eo cmd --no-headers | grep -c '[r]ay::IDLE')"
echo "   MemAvailable: $(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)MB"
