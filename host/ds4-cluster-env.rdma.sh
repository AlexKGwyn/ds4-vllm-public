#!/usr/bin/env bash
# RDMA-transport env for the DS4 vLLM cluster.
# The base env sets NCCL_IB_HCA=usb4_rdma, a PREFIX that can match more than
# one device (e.g. after a link reset leaves two ACTIVE rails on one netdev),
# which makes RCCL's ncclCommInitRank fail with "internal error". Pin one
# unambiguous HCA here (rdma_hca in ds4-config.yaml). GPU-direct all-reduce
# still runs via tbv_ar (DS4_TBV_AR2/DS4_TBV_AR_GPU from the base env).
source "$HOME/ds4-cluster-env.sh"
export NCCL_IB_HCA=${DS4_RDMA_HCA:-usb4_rdma5}
# RCCL logging is off. Re-enable to debug an init failure (the ncclCommInitRank
# trap above is the one that matters); expect ~50 lines per boot, mostly the
# harmless "GPU Direct RDMA not available for device 0" -- gfx1151 has no
# GPUDirect, so RCCL host-stages the >1 MiB prefill all-reduce. That is
# expected, not a fault.
#   export NCCL_DEBUG=WARN
#   export NCCL_DEBUG_SUBSYS=INIT,NET,ENV

# This file adds only what is specific to the RDMA transport. Everything else --
# DS4_W8A8_BF16_DIRECT, DS4_MTP_MAXSEQS and the rest -- lives in the base env
# above, in `${VAR:-default}` form so a caller can override it. Do not restate
# those knobs here: a bare `export VAR=...` after the source runs LAST and
# clobbers anything passed in via `podman exec --env`, which is exactly how the
# A/B harness sets them (bench_cycle.sh -> ds4-bench-serve.sh:12 sources this
# file). A sweep of a clobbered knob silently measures nothing and reports "no
# difference".
