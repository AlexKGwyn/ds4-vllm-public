# ds4-cluster-env.sh — canonical env for the DeepSeek-V4 vLLM TP=2 cluster.
# Sourced by the ray head, the box2 ray worker, and vllm serve (both boxes keep
# an identical copy in each box's home). NCCL/RDMA transport knobs plus the
# winning DS4_* tuning knobs.
# Silence torch's per-step all_gather_into_tensor deprecation FutureWarning
# (fires on vLLM's logits all-gather every decode step -> journal spam). Only
# FutureWarnings are hidden; real errors/UserWarnings still log. Inherited by
# the ray workers spawned after this is sourced.
# Stable block-content hashes for the disk KV tier. Python seeds hash() randomly
# per process, and vLLM derives NONE_HASH (the prefix-cache chain seed) from it,
# so without a fixed value identical token content produces different block
# filenames every run and the disk cache can never hit across a restart.
export PYTHONHASHSEED=0
# Bound allocator growth. The caching allocator never returns freed blocks, and
# its garbage collector is off by default (threshold 0), so the pool only
# grows without bound on a UMA box.
# The threshold is a fraction of total visible memory (~124 GiB here), so 0.85
# starts reclaiming unused cached blocks at ~105 GiB. It cannot starve a live
# allocation -- only cached blocks are reclaimed -- so the cost is re-mapping,
# never an OOM.
# expandable_segments lets the allocator grow/shrink segments in virtual address
# space instead of stranding fixed-size blocks. That is the fragmentation half:
# the GC threshold alone fires at ~105 GiB but does not hold the line, because
# GC can only reclaim blocks
# that are unused, and fragmented blocks that cannot satisfy incoming requests
# make the allocator grow anyway. Safe with --enforce-eager (no graph capture).
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::FutureWarning"
export NCCL_SOCKET_IFNAME=thunderbolt0
export GLOO_SOCKET_IFNAME=thunderbolt0
export NCCL_IB_HCA=usb4_rdma
export NCCL_IB_GID_INDEX=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_TIMEOUT=23
export NCCL_PROTO=LL
export NCCL_ALGO=Ring
export NCCL_IB_RETRY_CNT=7
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT_MS=2400000
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
# Inductor re-points the Triton JIT cache at its own dir at runtime, so pin
# BOTH to persistent paths: the defaults land under /tmp (tmpfs), and the
# first bringup after a boot then recompiles every kernel -- ~25 min CPU-bound
# in LLVM before the API can answer.
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export HIP_VISIBLE_DEVICES=0
export VLLM_ROCM_USE_AITER=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# winning DS4 tuning knobs
export DS4_W8A8_BF16=1
# DS4_W8A8_BF16_DIRECT=1: hand the caller's bf16 activation straight to the GEMM
# instead of quantising it to fp8 for w8a8_block_fp8_bf16 to immediately
# dequantise back. Removes 4 kernel launches and 2 dtype conversions on each of
# ~255 block-scaled linear calls per decode step, and is strictly MORE accurate
# (the fp8 round-trip is gone). Because it is more accurate it does change token
# output, so gate changes here on accuracy + MTP acceptance, not byte-identity.
# Set 0 to fall through to the stock path, which is byte-for-byte the old
# behaviour.
export DS4_W8A8_BF16_DIRECT=${DS4_W8A8_BF16_DIRECT:-1}
export DS4_TBV_AR=1
# DS4_TBV_AR2=1 uses the v2 GPU-poll+progress-thread all-reduce (~105us vs v1
# 228us). Takes precedence over DS4_TBV_AR; auto-falls-back to v1 if it fails to
# init. Set 0 to use v1.
export DS4_TBV_AR2=${DS4_TBV_AR2:-1}
export DS4_MTP_CAPTURE=1
# DS4_MTP_MAXSEQS: above this many concurrent sequences the DSpark drafter takes
# its `n_seg > max_seqs` bail-out and stops speculating ENTIRELY -- acceptance
# silently drops to 1.00 and aggregate throughput collapses. The limit only
# bounds win_kv, [n_stages, max_seqs, window, head_dim] bf16 = 3.1 MB at 8,
# 25 MB at 64. Keep this at or above the highest concurrency you serve.
export DS4_MTP_MAXSEQS=${DS4_MTP_MAXSEQS:-64}
# propagate DS4_* to box2 ray workers (not in ray's default copy prefixes)
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=DS4_
# Blocking (interrupt-based) GPU waits instead of busy-poll.
# TheRock ROCm busy-polls (rocr InterruptSignal::WaitRelaxed)
# -> ~2-3 cores spinning during inference -> CPU thermal throttle. Set 0 to revert to
# spin (lower latency, higher CPU).
export HSA_ENABLE_INTERRUPT=${DS4_HSA_INTERRUPT:-1}
# Load the patched bundled ROCr first. Login-shell setup adds /opt/rocm/llvm/lib
# to LD_LIBRARY_PATH; its runtime libraries change ROCr's initialization order
# and reproduce the AsyncEventsLoop spin even with the patched DSO preloaded.
# Clang uses its own RUNPATH, so omit only that entry from serving processes
# while preserving every other configured library path.
_ds4_ld_library_path=""
IFS=: read -r -a _ds4_ld_entries <<< "${LD_LIBRARY_PATH:-}"
for _ds4_ld_entry in "${_ds4_ld_entries[@]}"; do
  [[ -z "$_ds4_ld_entry" || "$_ds4_ld_entry" == /opt/rocm/llvm/lib ]] && continue
  _ds4_ld_library_path+="${_ds4_ld_library_path:+:}$_ds4_ld_entry"
done
export LD_LIBRARY_PATH="$_ds4_ld_library_path"
unset _ds4_ld_library_path _ds4_ld_entries _ds4_ld_entry
# The image also contains /opt/rocm's base runtime; preloading the patched
# bundled DSO (built by container/Dockerfile with AMD's event-age series plus
# rocr-force-block-indefinite-active-wait.patch) makes every HIP/HSA consumer
# resolve it first. This path exists only inside the serving container, so
# host-side setup is unaffected. Set DS4_HSA_PRELOAD=0 to disable only the
# preload; use the preserved image tag for a complete runtime rollback.
DS4_HSA_RUNTIME=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
if [[ -f "$DS4_HSA_RUNTIME" ]]; then
  _ds4_ld_preload=""
  IFS=: read -r -a _ds4_preload_entries <<< "${LD_PRELOAD:-}"
  for _ds4_preload_entry in "${_ds4_preload_entries[@]}"; do
    [[ -z "$_ds4_preload_entry" || "$_ds4_preload_entry" == "$DS4_HSA_RUNTIME" ]] && continue
    _ds4_ld_preload+="${_ds4_ld_preload:+:}$_ds4_preload_entry"
  done
  if [[ "${DS4_HSA_PRELOAD:-1}" != 0 ]]; then
    _ds4_ld_preload="$DS4_HSA_RUNTIME${_ds4_ld_preload:+:$_ds4_ld_preload}"
  fi
  export LD_PRELOAD="$_ds4_ld_preload"
  unset _ds4_ld_preload _ds4_preload_entries _ds4_preload_entry
fi
# HSA_ENABLE_MWAITX is retained as an optional fallback for finite active waits.
# The patched runtime converts only infinite active waits to blocked KFD
# interrupts, so latency-sensitive finite waits preserve upstream behavior.
# Set DS4_HSA_MWAITX=1 to use MONITORX/MWAITX during those finite waits.
export HSA_ENABLE_MWAITX=${DS4_HSA_MWAITX:-0}
# GPU-direct all-reduce: tbv_ar v1 data slots live in DEVICE memory as
# dma-buf MRs (RDMA lands straight in GPU pages; flags stay host-side for the
# CPU spin). Bit-exact and faster than host staging. Set 0 to fall back to
# pinned-host staging.
export DS4_TBV_AR_GPU=${DS4_TBV_AR_GPU:-1}
# MXFP4 matmul_ogs DECODE kernel config (tuned on this hardware; block_k 256 +
# num_stages 2 is the bandwidth lever the stock heuristic never picks).
# Decode-scoped in opt_flags (only under block_m<128) so prefill is untouched.
# MUST live here so BOTH TP ranks match. Unset the 5 vars to revert.
# DS4_IDX_OFFICIAL=1: indexer scoring uses the official QAT graph
# (hadamard128 + fp4-sim on indexer Q and compressor K rows, see
# ds4_tl_indexer.py). Production setting -- must match on BOTH boxes (the
# indexer is replicated).
export DS4_IDX_OFFICIAL=${DS4_IDX_OFFICIAL:-1}
# DS4_TOPK=1: the sparse indexer selects its top-512 with ds4_topk.select_topk
# (radix-select + ordered compaction) instead of a full row sort plus a second
# canonicalising sort. Deterministic by construction -- no float accumulation and
# no atomics assigning output slots, which is what made upstream's
# top_k_per_row_* launch-dependent at exact-cutoff ties. Biggest win at long
# contexts; short contexts gain little because cost scales with row width. Set 0 to fall
# back to the two-sort path in-process (no rebuild). Must match on BOTH boxes.
export DS4_TOPK=${DS4_TOPK:-1}
export DS4_MOE_BN=${DS4_MOE_BN:-32}
export DS4_MOE_NW=${DS4_MOE_NW:-2}
export DS4_MOE_NS=${DS4_MOE_NS:-2}
export DS4_MOE_BK=${DS4_MOE_BK:-256}
export DS4_MOE_WPE=${DS4_MOE_WPE:-1}
