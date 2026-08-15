# tbv — Thunderbolt-4 / USB4 soft-RDMA stack

This is the interconnect the DeepSeek-V4 TP=2 cluster runs its GPU all-reduce
over: a **RoCE-over-Thunderbolt** RDMA link between the two boxes, carried on the
USB4 cable(s). It replaces "real" InfiniBand/Ethernet NICs. vLLM's `tbv_ar` /
`tbv_ar2` all-reduce (baked into the container) talks to the RDMA device this
stack exposes as `usb4_rdma*`.

> ⚠️ **This is the deepest, most fragile part of the whole setup and it is
> hardware- and kernel-version-specific.** It is not "install and go". Budget
> real time, and read [`../AGENTS.md`](../AGENTS.md) §RDMA before touching a
> serving box. Several failure modes here require a **coordinated reboot of both
> boxes** to recover.

## The three kernel modules (must be a MATCHED SET)

All three must be built from the **same** source tree, or the box **panics on
cable connect** (the core↔net ring ABI diverged in the westeri tree):

1. **`thunderbolt` (core)** — patched USB4/Thunderbolt core.
2. **`thunderbolt_net`** — patched tbnet (gives the `thunderbolt0` IP link).
3. **`thunderbolt_ibverbs`** — the out-of-tree RDMA provider (exposes
   `usb4_rdma*` to libibverbs). Source: `thunderbolt-ibverbs/`.

Source of truth for core+net: **westeri/thunderbolt.git @ `503c5ae`** plus
the local patch series carried by the upstream ibverbs repo's
`kernel-workflow/patches/` — `build-modules.sh` fetches both at pinned
commits and encodes exactly that composition. The ibverbs module is
**hellas-ai/thunderbolt-ibverbs @ `76ba39b`** plus this repo's
`ibverbs-local.patch` (the RC-write zero-copy fastpath). Nothing third-party
is vendored: sources are fetched at build time. Built this way, all four
modules reproduce the production set bit-identically in `.text`.

Plus a 4th, standalone helper:

4. **`nhi_throttle`** (`nhi-throttle-mod/`) — drops the NHI MSI-X IRQ moderation
   from the hardcoded 128 µs to 8 µs (RDMA latency floor). Trivial to rebuild.

## Userspace side

The libibverbs provider (`libusb4_rdma-rdmav5*.so` + `usb4_rdma.driver`) is
built from source inside the container image (rdma-core + the upstream
provider patches — see `container/Dockerfile`). **libibverbs matches the
provider to the device *by name*, so the RDMA device MUST be named
`usb4_rdma*`** or `ibv_devices` is empty. The serving container is where it
matters; for host-side diagnostics build the same way or use the upstream
repo's nix packaging.

## What's in this folder

| path | what |
|---|---|
| `thunderbolt-ibverbs/` | ibverbs module source + `kernel-workflow/patches/` (the core+net patch series) + flake pinning westeri@503c5ae |
| `nhi-throttle-mod/` | NHI IRQ-throttle module source |
| `bringup/` | `tbv-reload-roce.sh` (loads ibverbs w/ `roce_netdev`, renames to `usb4_rdma0`, sets GID, NHI throttle), `tbv-roce-boot.sh`, the 2-cable prep + one-time site files |
| `systemd/` | `tbv-thunderbolt-patched.service` (loads the matched core+net at boot) and `tbv-roce.service` (runs the RoCE bring-up) |
| `ibverbs-local.patch` | this repo's diff on the pinned upstream thunderbolt-ibverbs (RC-write zero-copy fastpath) |

## Build recipe (per kernel, both boxes)

The modules are **unsigned**: disable Secure Boot (or sign them with your own
MOK) or the kernel refuses to load them.

**One cable or two?** A single cable carries everything (IP + RDMA); the
driver then auto-disables the RX zero-copy payload rail (no free NHI rings)
and TX stays zero-copy — fully functional. With `cables: 2` in
`ds4-config.yaml`, `install-modules.sh` keeps `thunderbolt1+` NM-unmanaged so
the second cable's NHI is dedicated to the RX zero-copy rail;
`bringup/tbv-second-cable-prep.sh` does the one-time prep.

Kernel modules are vermagic-locked, so **rebuild after every kernel update** on
each box:

```bash
./build-modules.sh                # all four modules, running kernel, no sudo
sudo ./install-modules.sh         # stage /var/lib/tbv + blacklist + boot units
```

`build-modules.sh` needs only kernel-devel and (first run) network: it clones
the pinned westeri tree, applies the local patch series, and builds core, net,
ibverbs, and nhi_throttle against the target kernel-devel. It
encodes the gotchas (KDIR symlink farm with `CONFIG_USB4_CONFIGFS=y` forced in
`auto.conf`, patched `thunderbolt.h` overlay, net built with
`KBUILD_EXTRA_SYMBOLS=<core>/Module.symvers`). Pass a kernel version to
cross-build for the other box.

`kernel-workflow/patches/` is the upstream-reviewable series regenerated from
the same tree. On Fedora/mutable installs (box2), modules can alternatively go
to `/lib/modules/$(uname -r)/updates/` + `depmod -a && dracut -f` so the
initramfs stops shadowing the stock core -- see `build-scripts/box2-*.sh`.

## Non-negotiable operational rules (learned the hard way)

- **Bring the 2-box link up COORDINATED** — reboot both boxes ~together with the
  correct boot order (`thunderbolt0` up **before** ibverbs claims the DMA rings).
- **Never live-reload the core, and never stagger per-box ibverbs reloads.** Both
  wedge the Thunderbolt HopID/tunnel allocator (`failed to allocate Rx HopID`,
  `native tunnel enable failed … EBUSY`) — recoverable **only by rebooting both
  boxes** (or unplug/replug the cable). Live-swapping only `thunderbolt_net`
  (leaf) is safe.
- **Never `git checkout` a kernel rev on a serving box** — it hard-hung box1.
- After a kernel update, RDMA silently falls back to slow TCP (`tbv_ar init
  failed`); rebuild + `sudo systemctl restart tbv-roce.service` on **both** boxes.
