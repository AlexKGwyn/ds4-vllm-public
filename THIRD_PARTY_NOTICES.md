# Third-party code and licenses

This project builds on the projects below. Their sources are **fetched at
pinned revisions at build time, not redistributed here** — what this
repository itself ships is at most a derivative patch against them (each
patch is licensed like the code it modifies). Original code in this
repository (the `host/` orchestration, `container/` build tooling, the new
engine files under `container/rootfs/`, `tbv/` build/bringup scripts, and docs) is licensed under
[Apache-2.0](LICENSE). The local `tbv/nhi-throttle-mod/` kernel module is
GPL-2.0 (`MODULE_LICENSE("GPL")`), as kernel modules must be.

| Component | Upstream | License | What this repo ships |
|---|---|---|---|
| **vLLM** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) @ `470229c` | Apache-2.0 | A derivative patch ([`container/patches/vllm-upstream.patch`](container/patches/vllm-upstream.patch), 31 files) applied to the base image's own vLLM sources at build time, plus 12 new files under `container/rootfs/`. Change inventory: [`MANIFEST.md`](container/patches/MANIFEST.md). |
| **Linux kernel — thunderbolt drivers** | [westeri/thunderbolt.git](https://git.kernel.org/pub/scm/linux/kernel/git/westeri/thunderbolt.git) (maintainer tree) | GPL-2.0 | Nothing redistributed — `tbv/build-modules.sh` fetches the pinned upstream tree and applies the patch series carried by the (also fetched) ibverbs repo. GPL-2.0 text: [`LICENSES/GPL-2.0.txt`](LICENSES/GPL-2.0.txt). |
| **thunderbolt-ibverbs** | [github.com/hellas-ai/thunderbolt-ibverbs](https://github.com/hellas-ai/thunderbolt-ibverbs) @ `76ba39b` | GPL-2.0 | Nothing redistributed — fetched at the pin by `tbv/build-modules.sh`, with this repo's [`tbv/ibverbs-local.patch`](tbv/ibverbs-local.patch) (GPL-2.0, derivative) applied. The local `tbv/nhi-throttle-mod/` companion module is this repo's own code (`MODULE_LICENSE("GPL")`). |
| **rdma-core** | [github.com/linux-rdma/rdma-core](https://github.com/linux-rdma/rdma-core) | GPL-2.0 OR Linux-OpenIB | Nothing redistributed — the `usb4_rdma` provider is built inside the image from rdma-core `v57.0` plus the upstream ibverbs repo's provider patches, both fetched at pinned revisions (`container/Dockerfile`). |
| **amd-strix-halo-vllm-toolboxes** | [github.com/kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) | MIT | Nothing redistributed — the container build pulls the base image by pinned digest (`container/Dockerfile`). |
| **ROCR-Runtime** | [github.com/ROCm/ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) | MIT (NCSA-style) | One patch (`container/patches/rocr-force-block-indefinite-active-wait.patch`) applied to the base image's bundled runtime at image build time. |

**Model weights** (`deepseek-ai/DeepSeek-V4-Flash-0731`) are not included and
are governed by their own license on Hugging Face.
