# DeepSeek-V4-Flash on vLLM — gfx1151 2-box rebuild kit

A reproducible rebuild of the hand-patched vLLM engine that serves
**DeepSeek-V4-Flash** across **two AMD Strix Halo (gfx1151) boxes**, tensor-parallel
(TP=2), with the inter-GPU all-reduce carried over a **Thunderbolt-4 / USB4 RDMA**
link. It contains everything needed to reconstruct the *software* from a public
base image plus the host-side scripts that launch and drive it.

This code and stack was pretty much entirely put together by AI, I probably can not help too much outside of prompting my agent.
PRs are welcome for performance improvements.

## Performance

Single-stream, TP=2 across the two boxes over Thunderbolt RDMA, with the
**DFlash parallel drafter** (DSpark MTP speculative decoding) enabled. Decode
speed depends on how often the drafter's parallel tokens are accepted, so
prose and code generate at different rates. Measured on the reference rig
(2× Ryzen AI Max+ 395 / Radeon 8060S, 128 GB UMA each): fresh uncached
prompts, temperature 0, thinking disabled, 300-token generations.

| context | prefill tok/s | decode — prose | decode — code |
|---|---|---|---|
| 512 | ~300 | 23 | 32 |
| 10k | ~270 | 23 | 27 |
| 50k | ~260 | 19 | 27 |
| 100k | ~235 | 19 | 22 |

Prefill is content-agnostic (prose and code measured within a few percent).
A fresh bringup warms its own kernels/caches automatically
(`warmup_ctx`), so these rates hold from the first real request.

---

## TL;DR — minimal bring-up

Hardware: **2× AMD Strix Halo (gfx1151)** boxes (~128 GB unified memory each), a
**Thunderbolt-4/USB4 cable** between them, **Secure Boot disabled**. Then, in
order:

```bash
# 0. model weights, ~150 GB, on BOTH boxes (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
hf download deepseek-ai/DeepSeek-V4-Flash-0731

# 1. RDMA kernel modules, on BOTH boxes, then reboot both together
tbv/build-modules.sh && sudo tbv/install-modules.sh

# 2. the patched vLLM image (box1; copy to box2 with podman save | podman load)
container/build.sh                                  # then create the distrobox — §2 below

# 3. site config + host scripts
#    edit host/ds4-config.yaml (IPs, transport, cables, memory) and deploy per §3

# 4. launch (box1) — full 2-box bringup, OpenAI API on :1234 when done
systemctl --user start ds4-vllm
```

Full ordered runbook with the gates and gotchas: [`AGENTS.md`](AGENTS.md).

---

> **Read this first — what "rebuildable" means here.** The **container rebuilds
> deterministically on any machine** with podman (`container/` below). *Serving*
> the model, however, needs the matching rig: 2× gfx1151 boxes, a working
> Thunderbolt-4 RDMA fabric, ROCm 7, and the model weights. This is a
> hardware-specific research build, not a general-purpose vLLM package. See
> **Prerequisites**.
>
> **Setting it up? Follow [`AGENTS.md`](AGENTS.md)** — the ordered end-to-end
> runbook (RDMA → container → serve) written for a person or agent doing the
> bring-up on a fresh pair of boxes.

---

## License & attribution

Original work here is **Apache-2.0** ([LICENSE](LICENSE)). This project
builds on **vLLM** (Apache-2.0), the **Linux kernel thunderbolt drivers** and
**hellas-ai/thunderbolt-ibverbs** (GPL-2.0), and **rdma-core** — their
sources are fetched at pinned revisions at build time rather than
redistributed; the patches shipped here are derivative works licensed like
the code they modify. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the full component/license table and upstream references.

---

## Layout

```
ds4-vllm-share/
├── README.md                     ← this file
├── AGENTS.md                     ← ordered bring-up runbook (RDMA → container → serve)
├── container/                    ← rebuild the patched vLLM engine (route 2)
│   ├── Dockerfile                ← FROM kyuz0 gfx1151 base + COPY the patch-set
│   ├── build.sh                  ← podman build helper (runs the packaging tests first)
│   ├── verify-patches.sh         ← prove patches/ really is base → rootfs
│   ├── rootfs/                   ← the 12 NEW files at their real paths (modified files ship as the patch)
│   └── patches/                  ← vllm-upstream.patch (base → patched) + MANIFEST.md
├── tbv/                          ← Thunderbolt-4 / USB4 soft-RDMA stack (the interconnect)
│   ├── ibverbs-local.patch       ← our diff on the pinned upstream thunderbolt-ibverbs
│   ├── nhi-throttle-mod/         ← NHI IRQ-throttle module source
│   ├── build-scripts/, bringup/  ← per-kernel build recipes + coordinated bring-up
│   ├── systemd/                  ← boot units (matched core+net; RoCE bring-up)
├── host/                         ← host-side orchestration (run outside the container)
│   ├── ds4-config.yaml, ds4-config ← site config (IPs, transport, disk KV) + loader
│   ├── ds4-cluster-restart.sh    ← full validated bringup (ExecStart of ds4-vllm.service)
│   ├── ds4-cluster-down.sh       ← full teardown (ExecStop/StopPost)
│   ├── ds4-vllm-manual-serve.sh  ← the vllm serve invocation + all serving flags
│   ├── ds4-vllm-warmup.py        ← post-start JIT/prefill-cache warmer (warmup_ctx)
│   ├── ds4-cluster-env*.sh       ← canonical env + DS4_* tuning knobs (rdma/tcp variants)
│   ├── container-heal.sh         ← reconcile/start a wedged podman container
│   └── systemd/                  ← ds4-vllm.service
```

---

## Prerequisites (to actually serve)

- **2× AMD Strix Halo / gfx1151** boxes, ~128 GB unified memory each. box1 is the
  ray head; box2 joins as a worker.
- **ROCm 7** (provided inside the container via the kyuz0 base — you do not
  install it on the host).
- **podman** + **distrobox** on both hosts (rootless is fine; the live setup uses it).
- **Thunderbolt-4 / USB4 RDMA** between the boxes. The scripts expect an RDMA
  device named `usb4_rdma` and a `thunderbolt0` IP interface (head IP
  `192.168.100.1`). This depends on the custom `tbv` kernel modules — **included**
  in [`tbv/`](tbv/) (source + build scripts + reference binaries; see
  [`tbv/README.md`](tbv/README.md) and [`AGENTS.md`](AGENTS.md) §1). They are
  kernel-version-specific and must be rebuilt per kernel. **Secure Boot must be
  disabled** (or sign the modules with your own MOK) — they are unsigned and a
  Secure Boot kernel refuses to load them. Without RDMA, the stack still runs
  on `transport: tcp` (much slower decode).
- **Model weights**: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  (~150 GB checkpoint). Not included — `hf download deepseek-ai/DeepSeek-V4-Flash-0731`
  on both boxes (the `model:` key in `ds4-config.yaml` takes an HF id or a local
  path). Served as `deepseek-v4-flash`.

---

## 1. Rebuild the container

```bash
cd container
./build.sh                       # -> ds4-vllm-patched:local
```

This does `FROM docker.io/kyuz0/vllm-therock-gfx1151@<pinned-digest>`, applies
`container/patches/vllm-upstream.patch` to the base's own vLLM sources (31
files), overlays the 12 new files from `container/rootfs/`
(see [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md)), builds
the `usb4_rdma` libibverbs provider from source (rdma-core `v57.0` + the
upstream provider patches, both fetched at pinned revisions), compiles the
TB4-RDMA all-reduce natives, and rebuilds ROCr with the idle-wait fix. The
base is ~35 GB and is pulled on first build; network is needed on the first
build for the pinned source fetches.

**Base image drift.** The Dockerfile pins the base by **digest** so the rebuild
matches the engine the patches were developed against (vLLM commit `470229c`). If
that digest is ever unpullable, replace it with `:latest` — but be aware kyuz0's
`latest` moves, and a newer base could carry a different vLLM whose files the
patches assume. Prefer the pinned digest.

**Keeping the patch honest.** Two checks:

```bash
python3 -m unittest discover -s tests -v   # manifest vs rootfs vs patch vs Dockerfile
container/verify-patches.sh                # patch applies cleanly to the pinned base
```

`build.sh` runs the first automatically; the second needs the base image
locally. To regenerate the patch after editing engine files, point
`DS4_PATCH_SRC` at a tree holding the desired files and run
`container/verify-patches.sh --write`.

## 2. Create the serving distrobox

The cluster scripts `podman exec` into a container named per
`ds4-config.yaml` (default **`vllm`**), so create one from the image you just
built (on **both** boxes):

```bash
distrobox create --name vllm --image ds4-vllm-patched:local --additional-flags \
  '--privileged --ipc host --pid host \
   --device /dev/kfd --device /dev/dri --device /dev/infiniband \
   --group-add video --group-add render --security-opt seccomp=unconfined'
distrobox enter vllm -- vllm --version
```

**Do not pass `--network host` in `--additional-flags`.** distrobox already
selects host networking, and passing it again fails with
`cannot set multiple networks without bridge network mode`. The resulting
container still gets `net=host`; verify with
`podman inspect vllm --format '{{.HostConfig.NetworkMode}}'`.

## 3. Run

Site specifics live in **`host/ds4-config.yaml`** — deploy it (edited for your
site) as `~/ds4-config.yaml` on box1 next to the scripts:

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-0731   # HF id or local path; weights on BOTH boxes
transport: rdma        # rdma | tcp — which ds4-cluster-env.<transport>.sh the cluster sources
head_ip: 192.168.100.1
worker_ip: 192.168.100.2
container: vllm        # podman container name (same on both boxes)
rdma_hca: usb4_rdma0   # NCCL_IB_HCA pin, rdma transport only
api_port: 1234
disk_kv: true          # NVMe prefix-KV tier (fs_lru); prefixes survive restarts
disk_kv_gib: 30        # per-NODE disk cap; check df on BOTH boxes before raising
max_ctx: 524288        # --max-model-len (512K, the validated profile)
kv_pin_gib: 6          # pinned GPU KV pool; sized against MemAvailable, not grown by max_ctx
gpu_mem_util: 0.83     # vLLM --gpu-memory-utilization; IGNORED while the KV pin above is set
warmup_ctx: 2048       # post-start warmup prefill size; 0 disables
cables: 1              # Thunderbolt cables between the boxes: 1 works; 2 adds the RX zero-copy rail
```

`host/ds4-config` (stdlib Python, no pyyaml) turns it into `DS4_*` exports for
the scripts. `transport: tcp` runs the same cluster without RDMA (correctness /
fallback profile; the tbv_ar decode all-reduce is disabled).

Deploy (paths are `$HOME`-relative, same layout on both boxes):

- **box1**: `host/ds4-config{,.yaml}`, `ds4-cluster-restart.sh`,
  `ds4-cluster-down.sh`, `ds4-vllm-manual-serve.sh`, `ds4-vllm-warmup.py`,
  `container-heal.sh`, all three `ds4-cluster-env*.sh`, and
  `host/systemd/ds4-vllm.service` into `~/.config/systemd/user/`.
- **box2**: `ds4-cluster-env*.sh` and `container-heal.sh` only — box2 is driven
  over ssh (key auth box1→box2 required).

Then `systemctl --user start ds4-vllm` brings up the whole 2-box cluster
(teardown → container heal → ray on both boxes → `vllm serve` → API/RDMA
verify); `stop` tears it down. The env files must stay **identical on both
boxes** — the two TP ranks silently diverge otherwise. 

---

## What was patched, and why

Full table in [`container/patches/MANIFEST.md`](container/patches/MANIFEST.md).
The themes:

- **DeepSeek-V4 model on gfx1151** — the AMD/ROCm DSpark model path, MLA
  attention, fp8 (UE8M0) KV-latent compress/quant, and the MTP drafter.
- **Mid-context retrieval** — the sparse indexer runs the *official* QAT graph
  (Hadamard128 + FP4 sim) before top-512 scoring (`DS4_IDX_OFFICIAL`), which the
  stock FP8 indexer skipped; plus a ROCm sparse-MLA attention rewrite.
- **MoE / GEMM tuning** — decode-scoped MXFP4 `matmul_ogs` knobs
  (`DS4_MOE_BN/NW/NS/BK/WPE`, the `block_k` bandwidth lever), a tuned gfx1151
  A8W8 GEMM config, and a `DS4_W8A8_BF16` fast bf16 path.
- **Thunderbolt-4 RDMA all-reduce** — custom `tbv_ar` (v1) / `tbv_ar2` (v2)
  all-reduce hooked into vLLM's communicator, replacing NCCL for the TP
  all-reduce on the TB4 link.
- **Disk KV cache** an `fs_lru` secondary
  tier gives the KV offloader a byte cap with LRU eviction, which the stock `fs`
  tier has no mechanism for, so it can point at a filesystem shared with
  everything else. Alongside it, the offloading scheduler now bounds each store
  batch: stock asks for every un-offloaded block at once and does not advance its
  cursor when the tier refuses, so a single refusal ratchets the ask past the
  tier and stores stop for the rest of the request. Bounding it lets a few
  hundred MiB of staging carry a full-length prefill.

---

## Provenance & notes

- Base: `docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294f…`, vLLM commit `470229c`.
- The original container also had `py-spy` pip-installed (a profiler) and some
  incidental OS packages under `/usr/lib/python3.14`; neither affects serving and
  both are intentionally omitted. Re-add `py-spy` with `pip install py-spy` inside
  the container if you want it.
