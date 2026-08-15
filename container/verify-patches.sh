#!/usr/bin/env bash
# Prove container/patches/vllm-upstream.patch applies cleanly to the pinned
# base image -- the patch IS the build path for the modified files, so a
# failed apply means a failed image build.
#
# --write regenerates the patch: point DS4_PATCH_SRC at a tree holding the
# desired final files at SITE-relative paths (e.g. `podman cp` them out of a
# container you edited, or a checkout that still carries them).
#
#   container/verify-patches.sh                                # verify
#   DS4_PATCH_SRC=<tree> container/verify-patches.sh --write   # regenerate
#
# Needs the pinned base image locally (~35 GB). It is the same digest the
# Dockerfile builds FROM, read from the Dockerfile so the two cannot diverge.
# The offline half of these checks -- manifest vs overlay vs Dockerfile counts --
# lives in tests/test_patchset_packaging.py and needs no image.
set -euo pipefail
cd "$(dirname "$0")/.."

SITE="opt/venv/lib/python3.12/site-packages"
# Absolute: the patches are applied with the work tree as git's CWD, so any
# relative path here would resolve against the wrong root.
REPO="$(pwd)"
PATCHDIR="$REPO/container/patches"
OVERLAY="$REPO/container/rootfs/$SITE"
WRITE=0
[ "${1:-}" = "--write" ] && WRITE=1

BASE=$(sed -n 's/^ARG DS4_BASE="\(.*\)"$/\1/p' container/Dockerfile)
[ -n "$BASE" ] || { echo "could not read ARG DS4_BASE from container/Dockerfile" >&2; exit 1; }
echo ">> base: $BASE"

if ! podman image exists "$BASE"; then
  echo "!! base image not present locally. Pull it first (~35 GB):" >&2
  echo "     podman pull $BASE" >&2
  exit 2
fi

# All modified files live in ONE combined diff; each file section starts with
# "--- a/<SITE>/<path>". On --write the file list comes from the manifest
# instead (the patch may not exist yet / may be stale).
COMBINED="$PATCHDIR/vllm-upstream.patch"
if [ "$WRITE" = 1 ]; then
  # modified (non-"new") rows of the manifest are the files with an upstream diff
  mapfile -t PATHS < <(python3 - "$PATCHDIR/MANIFEST.md" <<'PYEOF'
import re, sys
for line in open(sys.argv[1]):
    m = re.match(r"\| `([^`]+)` \| (?!\*\*new)", line)
    if m and "/" in m.group(1):
        print(m.group(1))
PYEOF
  )
else
  [ -f "$COMBINED" ] || { echo "no $COMBINED found" >&2; exit 1; }
  mapfile -t PATHS < <(sed -n "s|^--- a/$SITE/||p" "$COMBINED" | sort -u)
fi
[ "${#PATHS[@]}" -gt 0 ] || { echo "no patched files found" >&2; exit 1; }
echo ">> ${#PATHS[@]} patched files to verify"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/$SITE"

# One container invocation for all of them; per-file `podman run` is ~1s each.
podman run --rm --entrypoint sh "$BASE" \
  -c "cd /$SITE && tar cf - $(printf '%q ' "${PATHS[@]}")" \
  | tar xf - -C "$WORK/$SITE"

ok=0; bad=0
if [ "$WRITE" = 1 ]; then
  : > "$COMBINED"
  for rel in "${PATHS[@]}"; do
    base="$WORK/$SITE/$rel"; ship="${DS4_PATCH_SRC:?--write needs DS4_PATCH_SRC=<tree of desired files>}/$rel"
    [ -f "$base" ] || { echo "  MISSING FROM BASE  $rel"; bad=$((bad+1)); continue; }
    # manifest rows built by the Dockerfile itself (e.g. the rocr DSO) have no diff
    [ -f "$ship" ] || { echo "  (skipping $rel -- not in DS4_PATCH_SRC)"; continue; }
    diff -u --label "a/$SITE/$rel" --label "b/$SITE/$rel" "$base" "$ship" >> "$COMBINED" || true
    ok=$((ok + 1))
  done
  echo
  echo ">> wrote $ok file sections into $(basename "$COMBINED"). Re-run without"
  echo "   --write to verify, and update the Δ column in $PATCHDIR/MANIFEST.md"
  echo "   (tests/test_patchset_packaging.py checks it)."
  exit "$([ $bad -eq 0 ]; echo $?)"
fi

# Apply the whole combined diff onto the extracted base tree exactly as the
# Dockerfile will. -C so the stripped paths resolve inside the work tree
# rather than this repo (itself a git checkout).
if ! git -C "$WORK" apply -p1 "$COMBINED" 2>&1 | sed 's/^/  /'; then
  echo ">> combined patch failed to apply to the base tree"
  exit 1
fi
echo
echo ">> vllm-upstream.patch applies cleanly to the base (${#PATHS[@]} files)"
