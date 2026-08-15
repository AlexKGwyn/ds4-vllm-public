# Patch manifest — DeepSeek-V4-Flash vLLM (gfx1151)

All changes are a single overlay on the **`kyuz0/vllm-therock-gfx1151`** base
(pinned digest `sha256:25fd294f…`, which ships vLLM commit **`470229c`**).

- **31 modified** files — shipped as `vllm-upstream.patch` in this folder, applied to the base's own sources at image build
  (`*.patch`, base → patched). The Dockerfile does **not** apply these; it
  `COPY`s the final files from `../rootfs`. The diffs are here for review so a
  reader can see exactly what changed versus upstream.
- **12 new** files — added by the patch set; no diff (whole file is new). Their
  final form is in `../rootfs`.
- 1 file (`aiter_meta/csrc/cpp_itfs/utils.py`) was flagged changed by the image
  layer but is **byte-identical** to the base (metadata-only touch) and is
  deliberately **excluded**.

Paths below are relative to the container root
(`/opt/venv/lib/python3.12/site-packages/…`, except the venv top-level
modules and the aiter config).

## DeepSeek-V4 model (AMD / gfx1151 path)
| file | Δ | purpose |
|---|---|---|
| `vllm/models/deepseek_v4/__init__.py` | +1/-1 | model registration |
| `vllm/models/deepseek_v4/amd/model.py` | +67/-1 | AMD DSpark model wiring (custom all-reduce hook, layer glue) |
| `vllm/models/deepseek_v4/amd/rocm.py` | +43 | ROCm-specific op paths |
| `vllm/models/deepseek_v4/amd/dspark_mtp.py` | **new (885)** | DSpark Multi-Token-Prediction (MTP) drafter |
| `vllm/models/deepseek_v4/attention.py` | +32/-4 | MLA / sparse-attention wiring |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | +43 | KV-cache helpers (fp8_ds_mla latents) |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | +101/-40 | fused compress+quant of the MLA KV latent (UE8M0 fp8) |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | +82 | indexer-Q quantization (Hadamard128 + FP4 QAT) |

## Sparse indexer / mid-context retrieval
| file | Δ | purpose |
|---|---|---|
| `vllm/model_executor/layers/sparse_attn_indexer.py` | +1/-1 | route to the official indexer path |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | +261/-146 | ROCm sparse-MLA top-512 attention, writer-aware indexer-cache layout, deterministic selection/order, and bounded prefill JIT specialization (largest rewrite) |
| `ds4_topk.py` *(venv top-level)* | **new (457)** | deterministic radix-select top-k for the sparse indexer: ascending output removes the second sort, no full-row sort, no transient scratch. 2.7x at decode 128K, 10.2x at prefill 228K. `DS4_TOPK=0` restores the two-sort path |
| `ds4_tl_indexer.py` *(venv top-level)* | **new (476)** | TileLang indexer + official QAT scoring (`DS4_IDX_OFFICIAL`); also the `w8a8_block_fp8_bf16` fast GEMM helper, which serves decode and (reusing the cache decode populates) prefill |
| `ds4_synctrace.py` *(venv top-level)* | **new (93)** | attributes blocking device->host syncs to python call sites by wrapping only the Tensor methods that can force a D2H (~25 events/step, versus ~3,200 for `with_stack=True`, which pegs the worker). Driven from `ds4_tl_indexer._maybe_profile`, so it shares the `DS4_PROFILE` window and needs no new env var |
| `ds4_expert_union.py` *(venv top-level)* | **new (128)** | `DS4_EXPERT_UNION=1` probe: counts DISTINCT experts per MoE routing call, to measure the expert union a decode step actually touches (the free term in the decode roof). Wraps `make_routing_data`; vLLM's own `--enable-return-routed-experts` cannot see this model, whose routing never reaches `BaseRouter.route()`. Device-only accumulation, no per-step sync. Inert unless enabled |

## MoE / GEMM kernel tuning
| file | Δ | purpose |
|---|---|---|
| `vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags.py` | +72/-3 | decode-scoped MXFP4 matmul_ogs knobs (`DS4_MOE_BN/NW/NS/BK/WPE`), plus latching the per-call invariants: `get_cdna_version()`, the backend query, CU count and the seven DS4_MOE_* env reads were re-done on every call (~92 calls/step) for values fixed for the process lifetime |
| `vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_amd.py` | **new (57)** | vendored stock + the same latching for `compute_block_nk`'s 2x `get_cdna_version()` and CU-count query |
| `vllm/third_party/triton_kernels/routing_details/_routing_compute.py` | +4/-2 | MoE routing compute tweak |
| `vllm/third_party/triton_kernels/target_info.py` | +39/-4 | memoise the `is_hip*` / `cuda_capability_geq` target queries, which are re-answered per call for values fixed for the process lifetime — same latching idea as `opt_flags.py` |
| `vllm/model_executor/kernels/linear/scaled_mm/triton.py` | +83 | `DS4_W8A8_BF16` fast bf16 GEMM path (via ds4_tl_indexer), plus `DS4_W8A8_BF16_DIRECT` which skips the caller-side fp8 quantisation the bf16 path immediately undoes; DS4 flags latched at import instead of per call |
| `aiter/ops/triton/configs/gemm/gfx1151-GEMM-A8W8_BLOCKSCALE.json` | **new (15)** | tuned A8W8 blockscale GEMM config for gfx1151 |

## Distributed all-reduce over Thunderbolt-4 RDMA
| file | Δ | purpose |
|---|---|---|
| `vllm/distributed/device_communicators/cuda_communicator.py` | +51 | hook `DS4_TBV_AR` / `DS4_TBV_AR2` custom all-reduce |
| `tbv_ar.py` *(venv top-level)* | **new (255)** | v1 TB4-RDMA all-reduce (GPU dma-buf MRs) |
| `tbv_ar2.py` *(venv top-level)* | **new (69)** | v2 GPU-poll + progress-thread all-reduce (~105 µs) |

## Scheduler / KV / cudagraph / MTP
| file | Δ | purpose |
|---|---|---|
| `vllm/v1/core/kv_cache_utils.py` | +62/-1 | KV-cache accounting (fp8_ds_mla, hybrid manager) |
| `vllm/v1/core/sched/scheduler.py` | +42 | scheduler tweak |
| `vllm/v1/worker/gpu_model_runner.py` | +8 | model-runner hook |
| `vllm/compilation/breakable_cudagraph.py` | +18/-2 | piecewise cudagraph, keeps attention + custom all-reduce eager |
| `vllm/v1/spec_decode/llm_base_proposer.py` | +36/-1 | MTP proposer adjustment |

## Disk KV cache (`fs_lru`, distributed)
| file | Δ | purpose |
|---|---|---|
| `vllm/v1/kv_offload/tiering/fs/lru_manager.py` | **new (485)** | `fs_lru` scheduler-side control plane: byte cap + LRU eviction (which the stock `fs` tier has no mechanism for), block state tracking, and per-step disk-job directives. Does no file I/O itself — the stock scheduler-side design silently assumes every rank shares the scheduler's mmap, which is false on multi-node TP and restored functionally wrong KV on the remote rank (0/8 needle recall vs 8/8 fresh at 100% reported hits) |
| `vllm/v1/kv_offload/tiering/fs/distributed.py` | **new (345)** | the per-rank data plane: DiskJobDirective types plus the executor every worker runs against its own node-local `<base>_r<rank>` directory and its own slice of the offload mmap. Torch-free so the tier logic is unit-testable outside the image (tests/test_fs_lru_tier.py) |
| `vllm/v1/kv_offload/tiering/factory.py` | +8 | register the `fs_lru` tier type |
| `vllm/v1/kv_offload/tiering/spec.py` | +6 | expose the worker-side SharedOffloadRegion on the spec so the connector worker can execute disk jobs against this rank's slice |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | +156/-6 | bound the per-call store batch (stock asks for every un-offloaded block at once and does not advance the cursor when the tier refuses, so one refusal ratchets the ask past the tier and stores stop for the rest of the request), bound promotions per group, log per-group lookup outcomes, and route disk-tier directives/completions between the tier manager and the workers |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py` | +110/-3 | offload the whole KV page (page_size_bytes, not values-only real_page_size_bytes — the region past the values holds the UE8M0 scales for fp8_ds_mla), and execute disk-tier jobs: lazily build the per-rank DiskJobExecutor and report per-rank completions |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py` | +40/-4 | carry disk-tier directives scheduler→workers and per-rank completion reports workers→scheduler in the existing per-step connector metadata |
| `ds4_offload_batch.py` *(venv top-level)* | **new (102)** | the store-batch budget itself, kept out of `vllm` so it is unit-testable without torch. `DS4_OFFLOAD_STORE_BATCH_FRAC=0` restores stock behaviour |
| `vllm/v1/kv_offload/cpu/shared_offload_region.py` | +43/-3 | `DS4_OFFLOAD_MMAP_DIR` moves the staging region off `/dev/shm`. tmpfs pages are charged to RAM and can only go to swap; on a real filesystem they are page cache the kernel writes back and evicts, so staging costs no reserved RAM. Also skips `MADV_POPULATE_WRITE` when file-backed, and exposes the rank slice geometry the disk-tier executor addresses |
| `vllm/v1/kv_offload/cpu/gpu_worker.py` | +12/-1 | skip `cudaHostRegister` for a file-backed region — page-locking it would make the pages unreclaimable, which is the entire reason for moving it off tmpfs. Falls back to unpinned DMA, which the stock code already treats as a supported (slower) path |

## OpenAI API: reasoning + tools
| file | Δ | purpose |
|---|---|---|
| `vllm/tokenizers/deepseek_v4_encoding.py` | +18/-3 | retain prior assistant reasoning (`reasoning_content`) on tool conversations — the "vLLM drops reasoning" fix |
| `vllm/v1/structured_output/__init__.py` | +47/-20 | fix `</think>` boundary so the grammar engages after reasoning |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | +10 | reasoning/tool protocol fields |
| `vllm/entrypoints/openai/engine/protocol.py` | +9 | engine protocol fields |

## Kernel config lookup

| file | Δ | purpose |
|---|---|---|
| `vllm/platforms/rocm.py` | +31/-7 | `get_device_name()` falls back to torch when amdsmi is mocked, so tuned-kernel config lookups (fp8/int8/fused_moe/mamba) can match a file instead of always using defaults |

## ROCr / HIP idle CPU fixes

| file | Δ | purpose |
|---|---|---|
| `_rocm_sdk_core/lib/libhsa-runtime64.so.1` | build-stage replacement | exact bundled ROCr source plus AMD's complete event-age series (`c06ea68`, `933596e`, `78b874d`) and the reviewed `rocr-force-block-indefinite-active-wait.patch`; fixes both idle CPU cores while retaining finite active waits |
