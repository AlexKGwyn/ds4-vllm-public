#!/usr/bin/env bash
# Rebuild the DeepSeek-V4-Flash patched vLLM image from the kyuz0 gfx1151 base.
# The base (~35 GB) is pulled on first build if not already present locally.
set -euo pipefail
cd "$(dirname "$0")/.."   # build context = repo root (Dockerfile needs tbv/tbv-provider)

TAG="${1:-ds4-vllm-patched:local}"

# Cheap guards first: the overlay agreeing with MANIFEST.md and the Dockerfile's
# own file count, and no bytecode in container/rootfs (COPY would bake a stale
# .pyc into the image, shadowing the source it was compiled from). No torch, no
# image, no network. Deeper base-vs-patches check: container/verify-patches.sh.
echo ">> Checking the patch-set is self-consistent"
python3 -m unittest discover -s tests -p 'test_patchset_packaging.py' -q || {
  echo "!! patch-set is inconsistent -- fix before building (see the failures above)" >&2
  exit 1
}

echo ">> Building $TAG"
echo "   base: docker.io/kyuz0/vllm-therock-gfx1151 (pinned digest, see Dockerfile)"
podman build -t "$TAG" -f container/Dockerfile .

echo
echo ">> Built $TAG"
echo
echo "Sanity-check the patched files landed:"
echo "  podman run --rm $TAG ls -la /opt/venv/lib/python3.12/site-packages/tbv_ar2.py \\"
echo "                              /opt/venv/lib/python3.12/site-packages/ds4_tl_indexer.py"
echo
echo "Then create the serving distrobox (named per ds4-config.yaml, default 'vllm'):"
echo "  distrobox create --name vllm --image $TAG --additional-flags \\"
echo "    '--privileged --ipc host --pid host \\"
echo "     --device /dev/kfd --device /dev/dri --device /dev/infiniband \\"
echo "     --group-add video --group-add render --security-opt seccomp=unconfined'"
echo "  distrobox enter vllm -- vllm --version"
