# AGENTS.md — bring up DeepSeek-V4-Flash on the 2-box gfx1151 vLLM cluster

You are an agent setting this up on a fresh pair of machines. This file is the
runbook: build the pieces, wire them together, get the model serving, and verify
it. Read it top to bottom **before** running anything — several steps are
hard-to-reverse and order matters.

## 0. What you are building

DeepSeek-V4-Flash served by a **patched vLLM**, tensor-parallel across **two AMD
Strix Halo (gfx1151) boxes**, with the inter-GPU all-reduce carried over a
**Thunderbolt-4 / USB4 RoCE-RDMA** link.

```
        ┌────────────── box1 (ray HEAD, gfx1151) ──────────────┐
        │  distrobox "vllm"  ──►  vllm serve  TP rank 0         │
        │  ds4-vllm.service → ds4-cluster-restart.sh            │
        └───────────────┬───────────────────────────────────────┘
                        │  Thunderbolt-4 cable
                        │  RoCE-RDMA dev = usb4_rdma*   (tbv stack)
                        │  IP link thunderbolt0 = 192.168.100.1/.2
        ┌───────────────┴───────────────────────────────────────┐
        │  distrobox "vllm"  ──►  ray worker  TP rank 1          │
        └────────────── box2 (ray WORKER, gfx1151) ─────────────┘
```

Three independent layers, build/verify them in this order:

1. **tbv RDMA** (`tbv/`) — the Thunderbolt interconnect. Foundational and the
   riskiest; do it first. *(The cluster will also run without it on a slow TCP
   fallback — see §1.5 to de-risk by validating vLLM first, then adding RDMA.)*
2. **vLLM engine** (`container/`) — rebuild the patched image, one distrobox per box.
3. **Host orchestration** (`host/`) — the launch scripts, env, model weights.

## 0.1 Prerequisites (verify these exist; do NOT try to synthesize them)

- **2× AMD Strix Halo / gfx1151**, ~128 GB unified memory each, on the same LAN.
- A **Thunderbolt-4 / USB4 cable** physically connecting the two boxes.
- Linux with **kernel headers/devel** for the running kernel on each box, `podman`,
  `distrobox`, `rdma-core`/`libibverbs`, `git`, build toolchain.
- The model weights **`deepseek-ai/DeepSeek-V4-Flash-0731`** (~150 GB) downloaded
  on **both** boxes (`hf download deepseek-ai/DeepSeek-V4-Flash-0731`).
- Root/sudo on both boxes (kernel modules, systemd units).
- **Secure Boot disabled on both boxes** — the tbv modules are unsigned;
  a Secure Boot kernel will refuse every `insmod` in §1.

Pick roles now and keep them consistent everywhere: **box1 = ray head**, IP on
Thunderbolt `192.168.100.1`; **box2 = worker**, `192.168.100.2`. Site values
(IPs, container name, transport, HCA pin, disk KV) live in
`host/ds4-config.yaml`, deployed as `~/ds4-config.yaml` on box1 (see §3);
paths in the scripts are `$HOME`-relative.

---

## 0.2 Recall / context integrity — fixed, gated

Long-context recall on this stack is correct for the deployed profile. What
keeps it correct:

- `DS4_IDX_OFFICIAL=1` in `host/ds4-cluster-env.sh` -- the sparse indexer's
  official Hadamard128 + FP4 QAT scoring graph. Must be engaged on BOTH TP
  ranks (it is exported from the shared env, so keep the env identical).
- The `deepseek_v4_encoding.py` patch -- the chat encoder no longer strips
  prior assistant reasoning on tool conversations.

Any change to the indexer, MTP, kernels, or tuning knobs must re-pass
needle/recall probes at your target context depth before it ships (see §5).

Kernel/fusion changes must also pass `host/ds4-kernel-harness.py` (bit-exact
fused-vs-stock comparison with crash-shaped + stress inputs, run inside the
container with `HIP_LAUNCH_BLOCKING=1`). Trap: Triton 3.7.0 / ROCm 7.14
miscompiles runtime-trip-count `scf.for` loops (garbage loads/stores, sporadic
illegal memory accesses — the 2026-08-19 `_topk_ragged_decode_kernel` fault);
use `tl.static_range` + masked inactive iterations in single-program kernels.
See the fixed kernel in `ds4_fused_glue.py`.

---

## 1. tbv — Thunderbolt RDMA (do this first)

Full detail in [`tbv/README.md`](tbv/README.md); this is the ordered action list.
**The kernel modules are vermagic-locked to a specific kernel**,
so plan to build for your exact kernel. Run every module step on **both**
boxes.

### 1.1 Build the matched core+net (per box, per kernel)

The `thunderbolt` core, `thunderbolt_net`, and `thunderbolt_ibverbs` must be one
matched set or the box **panics on cable connect**.

```bash
tbv/build-modules.sh [KVER]          # all four modules, no sudo
sudo tbv/install-modules.sh [KVER]   # stage /var/lib/tbv + blacklist + boot units
```

`build-modules.sh` fetches the pinned upstream trees (westeri @`503c5ae`;
hellas-ai/thunderbolt-ibverbs @`76ba39b` + `tbv/ibverbs-local.patch`), applies
the kernel patch series the ibverbs repo carries, and builds against the
target kernel-devel. Gotchas it handles:
- Force **`CONFIG_USB4_CONFIGFS=y`** in the KDIR `auto.conf` (else
  `tb_configfs_init/exit` are undefined at link).
- Build `thunderbolt_net` with **`KBUILD_EXTRA_SYMBOLS=<core>/Module.symvers`**
  (it needs `tb_ring_throttling` from the patched core).
- MODVERSIONS is off; only **vermagic** must match `uname -r`.

### 1.2 Build the out-of-tree modules

`build-modules.sh` above already builds ibverbs + nhi_throttle (steps 3-4)
against the same patched KDIR; nothing separate to run.

### 1.3 Install the userspace provider (host AND container)

```bash
# The serving container image builds and ships the provider itself
# (container/Dockerfile provider-build stage) — nothing to install for serving.
# For host-side diagnostics (ibv_devices on the host), build the same way:
# rdma-core v57.0 + the provider patches from the upstream ibverbs repo.
```
The provider matches devices **by name**, which is why the device must be
renamed to `usb4_rdma*`.

### 1.4 Stage, load, and bring up — COORDINATED

- `sudo tbv/install-modules.sh` (on each box) does the whole install: stages
  the built modules into `/var/lib/tbv`, blacklists the stock thunderbolt
  driver (modprobe.d + kernel args, which also keeps the initramfs copy from
  shadowing it), installs and enables `tbv-thunderbolt-patched.service`
  (loads matched core+net at boot) and `tbv-roce.service` (the RoCE bring-up:
  loads ibverbs with `roce_netdev=thunderbolt0`, renames the rail to
  `usb4_rdma0`, populates the GID table, sets the 8 µs NHI throttle).
- Ensure `thunderbolt0` autoconnects to its `192.168.100.x` IP at boot **before**
  ibverbs claims the DMA rings — `install-modules.sh` prints the `nmcli`
  command that writes the autoconnect NetworkManager profile.
- **Then reboot BOTH boxes ~together.** The handshake only converges when both
  come up in the right order simultaneously.

> 🛑 **Do NOT** live-reload the core, and **do NOT** stagger per-box ibverbs
> reloads. Both wedge the Thunderbolt HopID/tunnel allocator and require a
> **coordinated reboot of both boxes** to recover. Live-swapping only
> `thunderbolt_net` is the one safe hot operation.

### 1.5 Verify RDMA (gate)

```bash
ls /sys/class/infiniband/                 # -> usb4_rdma0 (or usb4_rdma5)
rdma link                                 # port state ACTIVE / PHYS_STATE LinkUp
cat /sys/class/infiniband/usb4_rdma0/ports/1/gids/1   # NON-zero (RoCEv2 IPv4 GID = index 1)
ibv_devices                               # lists usb4_rdma0
```
If `gids/1` is all-zero, `thunderbolt0` has no `192.168.100.x` IP yet — fix the IP
first. If `usb4_rdma*` never appears after a kernel change, the `.ko` vermagic no
longer matches: rebuild (§1.1–1.2) and `sudo systemctl restart tbv-roce.service`
on both boxes.

**De-risk option:** RDMA is a performance layer, not a correctness gate — set
`transport: tcp` in `~/ds4-config.yaml` to run the same cluster over sockets
(much slower decode). If you want to validate the model path first, skip to
§2–§4 on TCP now and return to finish RDMA once tokens are flowing.

---

## 2. Build the vLLM engine

See [`container/`](container/). On **each** box:

```bash
cd container && ./build.sh                # -> ds4-vllm-patched:local  (base ~35 GB pulled once)
```
This is `FROM kyuz0/vllm-therock-gfx1151@<pinned digest>` + the DS4 patch-set
(31 modified files as `patches/vllm-upstream.patch`, 12 new — see
`container/patches/MANIFEST.md`). Then create the serving container, named per
`container:` in `ds4-config.yaml` (default **`vllm`**):

```bash
distrobox create --name vllm --image ds4-vllm-patched:local --additional-flags \
  '--privileged --ipc host --pid host \
   --device /dev/kfd --device /dev/dri --device /dev/infiniband \
   --group-add video --group-add render --security-opt seccomp=unconfined'
distrobox enter vllm -- vllm --version          # gate: prints a version
distrobox enter vllm -- ibv_devices             # gate: lists usb4_rdma0 (if §1 done)
```

## 3. Host orchestration + config

Deploy the `host/` files per README §3 and set the site values in
`~/ds4-config.yaml` on box1 (head/worker IPs, container name, `transport:
rdma|tcp`, RDMA HCA pin, disk KV). Two rules that bite:

- `ds4-cluster-env*.sh` **must be byte-identical on both boxes** — the two TP
  ranks silently diverge otherwise. Copy the same files to both.
- Box1 needs passwordless ssh to the worker IP: the cluster scripts drive
  box2's container over ssh.

## 4. Start serving

```bash
systemctl --user start ds4-vllm      # box1; ~5 min warm
```

`ds4-cluster-restart.sh` (the unit's ExecStart) does the whole sequence:
teardown + stranded-process reap, container heal on both boxes, ray head +
box2 worker (2 GPUs gate), then `vllm serve` via `ds4-vllm-manual-serve.sh`
as the transient `ds4-vllm-manual` unit (MTP speculative decode,
`deepseek_v4` tokenizer/reasoning/tool parsers, fp8 KV, eager, disk KV
tier), verifies the API and the RDMA all-reduce, and dispatches the
warmup (`ds4-vllm-warmup.py`, `warmup_ctx` in the yaml) before reporting
success. `systemctl --user stop ds4-vllm` tears everything down.
