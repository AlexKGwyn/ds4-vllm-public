#!/usr/bin/env bash
# Production vLLM launcher (512K ctx + MTP, RDMA). Run INSIDE the vllm container;
# started by the ds4-vllm-manual systemd user unit. Sources the canonical RDMA
# cluster-env, then execs vllm serve on api_port from ds4-config.yaml.
#
# THIS FILE IS THE SOURCE OF TRUTH -- copy it to the serve user's home before
# serving. The running copy lives outside the repo; edit here, then redeploy.
#
# Memory flags, since they are the ones that bite:
#   --kv-cache-memory-bytes  Pin KV. Inferring it oversubscribes this box: the
#     bf16 weight cache and the RDMA buffers are allocated AFTER the profiling
#     pass that would do the inferring. The pool is a fixed-size LRU: it does not
#     grow with --max-model-len, and its occupancy is NOT what
#     vllm:kv_cache_usage_perc reports (that metric counts only blocks referenced
#     by in-flight requests; cached-but-idle blocks sit in the free queue and
#     read as free). Capacity = (pin / bytes-per-block) blocks x 256 tokens.
#   --gpu-memory-utilization  INERT while the pin above is set -- vLLM skips
#     memory profiling and ignores it entirely (gpu_worker.py:384).
#   --max-num-batched-tokens 512  NOT 2048; the indexer/top-k workspace scales
#     with batch x context and 2048 costs ~10 GiB more at 256K.
#
# Do not add comments inside the backslash-continued `vllm serve` command below:
# a '#' there silently comments out every remaining argument, and `bash -n`
# still reports the file as valid.
set -u
source "$HOME/ds4-cluster-env.${DS4_TRANSPORT:-rdma}.sh"

# NVMe KV cache (fs_lru tier), ON by default. Prefix blocks evicted from GPU
# are kept on node-local disk and reloaded instead of re-prefilled, and the
# cache survives a restart -- the case the in-GPU prefix cache cannot cover.
#
# The tier is DISTRIBUTED: the scheduler decides what to store/evict/promote,
# but every rank does its own file I/O against its own slice of its own
# node-local mmap, under <dir>/<model>_r<rank>/ (r0 on this box, r1 on box2).
# That is a correctness requirement, not an optimization: a tier that moves
# bytes only on the scheduler's node restores garbage on the remote rank,
# and bad KV degrades recall silently instead of raising. If recall ever
# drops on restored prefixes, FIRST check both boxes' cache dirs are being
# written.
#
#   DS4_DISK_KV_BYTES      per-NODE disk cap, default 30 GiB. Not optional --
#                          the stock "fs" tier has no capacity argument and
#                          never evicts. Check df on BOTH boxes before
#                          raising it, and expect it to sit full under real
#                          traffic: LRU churn at the cap is normal. Each box
#                          stores only its own rank's slice (~1 MiB per
#                          block file, ~100 tokens/file across the groups).
#   DS4_DISK_KV_CPU_BYTES  CPU staging tier, default 4 GiB. Disk is a secondary
#                          tier behind a CPU primary and ROCm has no GPU->disk
#                          path, so blocks land in host memory first.
#
#                          4 GiB does NOT cost 4 GiB of RAM. With staging on
#                          disk (below) the region is a file-backed mmap, so it
#                          is reclaimable page cache -- resident only while hot,
#                          dropped under pressure. That is the entire reason for
#                          moving it off tmpfs, and it is what lets this be
#                          sized for correctness rather than for RAM.
#
#                          A read hit requires the matched prefix to be
#                          simultaneously resident in this tier, so keep it
#                          comfortably above the block count of the longest
#                          prefix you expect to reuse.
#
# The KV pin below is 6 GiB (~1.44M tokens). The pool size decides how often
# a session resume pays a multi-second disk restore instead of hitting the
# GPU prefix cache, so size it as large as MemAvailable tolerates; staging
# is file-backed page cache, not a tmpfs reservation, so the disk cache adds
# no reserved RAM on top of this pin.
#
# Staging also defaults onto the NVMe (DS4_DISK_KV_STAGE_ON_DISK=0 to keep it in
# /dev/shm). tmpfs pages are charged to RAM and can only be pushed to swap; a
# real file makes them ordinary page cache, which the kernel writes back and
# evicts cleanly, so staging stops being a reservation. Costs unpinned DMA for
# the GPU->host copy (cudaHostRegister is skipped deliberately -- page-locking
# would undo the whole point). If decode or TTFT regresses, set
# DS4_DISK_KV_STAGE_ON_DISK=0 and compare.
DS4_DISK_KV=${DS4_DISK_KV:-1}

# The tier ships in the image, not in this script. Against an image built
# before it, vllm serve dies at startup with "Unknown secondary tier type:
# 'fs_lru'" -- so a restart that happens before container/build.sh has run (a
# crash, the watchdog, a reboot) would take the engine down rather than merely
# skip the cache. Degrade to no cache instead, loudly. distributed.py is the
# newer half (per-rank data plane); an image with only the old lru_manager
# would run the scheduler-side tier that corrupts rank1's restores.
KV_TIER_DIR=/opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs
if [ "$DS4_DISK_KV" = "1" ] && { [ ! -f "$KV_TIER_DIR/lru_manager.py" ] \
    || [ ! -f "$KV_TIER_DIR/distributed.py" ]; }; then
  echo "[manual-serve] WARNING: this image has no distributed fs_lru tier -- rebuild"
  echo "[manual-serve]          with container/build.sh. Serving WITHOUT the disk KV cache."
  DS4_DISK_KV=0
fi

OFFLOAD=()
if [ "$DS4_DISK_KV" = "1" ]; then
  KVDIR=${DS4_DISK_KV_DIR:-$HOME/ds4-kvcache}
  mkdir -p "$KVDIR"
  if [ "${DS4_DISK_KV_STAGE_ON_DISK:-1}" = "1" ]; then
    # A SIBLING of the cache dir, never inside it. fs_lru owns root_dir
    # exclusively: it walks the tree on startup, adopts every file it finds as
    # a cached block, counts it against the cap, and will os.remove() it when
    # it becomes the LRU victim. Nesting staging under root_dir therefore hands
    # the evictor the live mmap.
    export DS4_OFFLOAD_MMAP_DIR=${DS4_OFFLOAD_MMAP_DIR:-${KVDIR%/}-stage}
    echo "[manual-serve] staging region on disk: $DS4_OFFLOAD_MMAP_DIR"
  fi
  OFFLOAD=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${DS4_DISK_KV_CPU_BYTES:-4294967296},\"secondary_tiers\":[{\"type\":\"fs_lru\",\"root_dir\":\"$KVDIR\",\"max_bytes\":${DS4_DISK_KV_BYTES:-32212254720}}]}}")
  echo "[manual-serve] disk KV ON: dir=$KVDIR cap=${DS4_DISK_KV_BYTES:-32212254720} cpu=${DS4_DISK_KV_CPU_BYTES:-4294967296}"
fi

# Decode runs with breakable (PIECEWISE) cudagraphs. Three things make this
# correct on gfx1151; all three live in the patched engine
# (container/patches/vllm-upstream.patch):
#   1. VLLM_USE_BREAKABLE_CUDAGRAPH=1 exported in ds4-cluster-env.sh so the
#      prestarted Ray workers see it at import time. Without it the
#      eager-break decorator strips itself in the workers and the capture
#      silently degrades to one monolithic graph with host-side attention
#      metadata baked in (deterministic garbage on replay). The patched
#      decorator also checks at call time as a backstop.
#   2. Padding-row zeroing after each replayed eager segment: eager ops
#      write only the valid rows of their in-place outputs, and NaN pool
#      garbage in the padding rows otherwise poisons the whole batch
#      through row-mixing fp8-quant reductions in the next graph segment.
#   3. The TP all-reduce is a functional eager break: tbv_ar2 refuses to
#      run during stream capture, so capturing it would bake the RCCL
#      fallback into every replay.
# The capture sizes cover the MTP-5 decode shapes (6*num_seqs) exactly;
# padded replays are correct but slower. The speculative config's
# "enforce_eager":true is LOAD-BEARING and must stay: the DSpark drafter
# cannot be captured by the runner's wrapper (its replay-marker test is a
# blocking D2H, illegal during capture) and manages its own step-0 graphs
# instead (DS4_MTP_CUDAGRAPH, default on).
exec vllm serve "${DS4_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}" \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization ${DS4_GPU_UTIL:-0.83} \
  --kv-cache-memory-bytes ${DS4_KV_BYTES:-6442450944} \
  --max-model-len "${DS4_MAX_CTX:-524288}" \
  --max-num-batched-tokens 512 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  -cc.cudagraph_mode=PIECEWISE \
  -cc.cudagraph_capture_sizes='[1,2,4,6,8,12,16,18,24,32,48,64]' \
  --override-generation-config '{"temperature":1.0,"top_p":1.0}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":5,"disable_padded_drafter_batch":true,"enforce_eager":true}' \
  "${OFFLOAD[@]}" \
  --host 127.0.0.1 --port "${DS4_API_PORT:-1234}"
