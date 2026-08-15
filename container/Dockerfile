# DeepSeek-V4-Flash — patched vLLM engine for AMD gfx1151 (Strix Halo), TP=2.
#
# This rebuilds the hand-patched serving engine as a THIN layer over kyuz0's
# TheRock/ROCm gfx1151 base image — the exact base the patches were developed
# against. The base already contains the full ROCm 7 + PyTorch + vLLM build
# (vLLM commit 470229c); this file applies the DeepSeek-V4 / DSpark patch
# (patches/vllm-upstream.patch, 31 files), overlays the 12 new files from
# ./rootfs, builds the usb4_rdma provider from source (rdma-core + the
# upstream thunderbolt-ibverbs patches), compiles the TB4-RDMA all-reduce
# natives, and rebuilds ROCr. Third-party sources are FETCHED at pinned
# commits, not vendored (network needed on first build).
#
# Build (context is the REPO ROOT):
#   container/build.sh               # -> ds4-vllm-patched:local
#   podman build -t ds4-vllm-patched:local -f container/Dockerfile .
#
# The FROM line is pinned to the exact base digest the patches were built
# against. If that digest is no longer pullable, swap it for :latest and see
# "Base image drift" in ../README.md before trusting the result.

ARG DS4_BASE="docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294fde9f729d1e75f109022ab4496c78190c0a6dc0142440529f7af20e4d"

# Rebuild the base image's exact ROCr source with AMD's complete event-age fix
# series plus the DS4 infinite-wait policy. The bundled June runtime can pin
# one CPU core in AsyncEventsLoop and another in an indefinite active wait.
FROM ${DS4_BASE} AS rocr-idle-fix
USER root
RUN dnf install -y git elfutils-libelf-devel numactl-devel && dnf clean all
COPY container/patches/rocr-force-block-indefinite-active-wait.patch /tmp/
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/ROCm/rocm-systems.git /tmp/rocm-systems && \
    git -C /tmp/rocm-systems sparse-checkout init --cone && \
    git -C /tmp/rocm-systems sparse-checkout set projects/rocr-runtime && \
    git -C /tmp/rocm-systems checkout \
        d34cbb640961dcd0c0aa6c1c3c1cfa5d4becfa31 && \
    for commit in \
        c06ea68a592f6fcf08ada1e10e7b5b54b21aeae7 \
        933596e921d72e050fc982faebec0fea4cb2d64e \
        78b874d46014128675bae911934323361eaee03d; do \
      GIT_COMMITTER_DATE="$(git -C /tmp/rocm-systems show -s --format=%cI "$commit")" \
        git -C /tmp/rocm-systems \
          -c user.name="DS4 reproducible build" \
          -c user.email="ds4-build@invalid" \
          cherry-pick "$commit"; \
    done && \
    git -C /tmp/rocm-systems apply \
      /tmp/rocr-force-block-indefinite-active-wait.patch
RUN cmake \
        -S /tmp/rocm-systems/projects/rocr-runtime \
        -B /tmp/rocm-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON && \
    cmake --build /tmp/rocm-build --parallel 16

# Build the usb4_rdma libibverbs provider from source: rdma-core at the ABI
# the base ships (rdmav57) plus the provider patches from the upstream
# thunderbolt-ibverbs repo (pinned; same pin as tbv/build-modules.sh).
FROM ${DS4_BASE} AS provider-build
USER root
RUN dnf install -y git cmake gcc libnl3-devel pkgconf-pkg-config && dnf clean all
RUN git clone --depth 1 -b v57.0 https://github.com/linux-rdma/rdma-core /tmp/rdma-core && \
    git clone https://github.com/hellas-ai/thunderbolt-ibverbs /tmp/tbv-ibverbs && \
    git -C /tmp/tbv-ibverbs checkout 76ba39b630a70accb72f19388eefe48844b50eb8 && \
    cd /tmp/rdma-core && \
    for p in /tmp/tbv-ibverbs/packaging/rdma-core-patches/0*.patch; do git apply -C1 "$p"; done && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 && \
    cmake --build build --parallel 16 && \
    ls build/lib/libusb4_rdma-rdmav57.so

FROM ${DS4_BASE}

# Apply the vLLM patch-set to the base's own sources (31 files), then overlay
# the 12 new files from rootfs/ (which mirrors "/"). The patch is the same
# reviewable diff in patches/; container/verify-patches.sh proves it applies.
COPY container/patches/vllm-upstream.patch /tmp/
RUN cd / && git apply -p1 --whitespace=nowarn /tmp/vllm-upstream.patch && rm /tmp/vllm-upstream.patch
COPY --chown=root:root container/rootfs/ /

# Replace PyTorch's bundled ROCr SONAME with the ABI-compatible build above.
# The immutable base image remains the rollback path.
COPY --from=rocr-idle-fix --chown=root:root \
  /tmp/rocm-build/rocr/lib/libhsa-runtime64.so.1.21.0 \
  /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1

# usb4_rdma userspace provider. libibverbs matches a provider to a device BY
# NAME, so without these two files `ibv_devices` is EMPTY inside the container:
# tbv_ar init fails and NCCL_IB cannot see the rail, and the cluster silently
# falls back to TCP.
COPY --from=provider-build --chown=root:root \
  /tmp/rdma-core/build/lib/libusb4_rdma-rdmav57.so /usr/lib64/libibverbs/
RUN printf 'driver usb4_rdma\n' > /etc/libibverbs.d/usb4_rdma.driver

# Build the TB4-RDMA all-reduce natives that tbv_ar.py / tbv_ar2.py ctypes-load.
# rootfs/ carries only the python wrappers, so without this step DS4_TBV_AR2
# init fails at "libtbv_ar2.so not found" and the all-reduce falls back.
# rdma-core-devel is only needed for <infiniband/verbs.h> at build time; the
# base ships the runtime but not the headers.
COPY --chown=root:root container/native/ /tmp/ds4-native/
RUN dnf install -y rdma-core-devel && \
    /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1151 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libtbv_ar2.so \
      /tmp/ds4-native/tbv_ar2.hip -libverbs -lpthread && \
    gcc -O2 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libtbv_ar.so \
      /tmp/ds4-native/tbv_ar.c -libverbs && \
    dnf clean all && rm -rf /tmp/ds4-native

# Drop stale bytecode for the packages we touched so the first import recompiles
# from the patched sources. (COPY already writes fresh mtimes, which invalidates
# the .pyc; this is belt-and-suspenders.)
RUN find /opt/venv/lib/python3.12/site-packages/vllm \
         /opt/venv/lib/python3.12/site-packages/aiter \
         -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# com.github.containers.toolbox is what `toolbox create` checks for; the pinned
# base digest does not carry it, and box2 uses toolbox rather than distrobox.
LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.title="ds4-vllm-patched" \
      org.opencontainers.image.description="DeepSeek-V4-Flash vLLM patched engine (AMD gfx1151, TP=2, TB4-RDMA)" \
      ds4.base="docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294f" \
      ds4.vllm_commit="470229c" \
      ds4.snapshot_equiv="localhost/ds4-vllm-patched:20260728"
