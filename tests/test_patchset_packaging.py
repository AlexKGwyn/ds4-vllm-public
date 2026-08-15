"""Keep the patch, the overlay, the manifest and the Dockerfile in agreement.

The Dockerfile applies patches/vllm-upstream.patch to the base's own sources
(the modified files) and `COPY container/rootfs/ /` (the new files). A file in
the overlay or patch but not in the manifest is an undocumented change someone
else cannot review; a manifest row backed by neither is a change that silently
will not be applied. Neither shows up as a build failure -- the image just
quietly behaves differently from what the docs claim.

Pure filesystem checks: no torch, no vllm, no image, no network. Run with

    python3 -m unittest discover -s tests -v

Checks that need the base image (does every "modified" path exist upstream?)
belong in container/build.sh, not here.
"""

import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTFS = os.path.join(REPO, "container/rootfs")
SITE = "opt/venv/lib/python3.12/site-packages"
OVERLAY = os.path.join(ROOTFS, SITE)
MANIFEST = os.path.join(REPO, "container/patches/MANIFEST.md")
PATCHES = os.path.join(REPO, "container/patches")
DOCKERFILE = os.path.join(REPO, "container/Dockerfile")

# Rows that legitimately have no file under container/rootfs.
NOT_IN_OVERLAY = {
    # Built in a Dockerfile stage and COPY'd from there, not from rootfs.
    "_rocm_sdk_core/lib/libhsa-runtime64.so.1",
}


def overlay_files():
    """Container-relative paths of every file in the overlay."""
    out = set()
    for dirpath, _, names in os.walk(ROOTFS):
        for name in names:
            full = os.path.join(dirpath, name)
            out.add(os.path.relpath(full, OVERLAY))
    return out


def manifest_rows():
    """{path: delta} for every table row in the manifest."""
    rows = {}
    with open(MANIFEST) as fh:
        for line in fh:
            m = re.match(r"^\|\s*`([^`]+)`\s*(?:\*\([^)]*\)\*)?\s*\|\s*([^|]+)\|", line)
            if m:
                rows[m.group(1)] = m.group(2).strip()
    return rows


class TestPatchsetPackaging(unittest.TestCase):
    def setUp(self):
        self.overlay = overlay_files()
        self.rows = manifest_rows()

    def test_overlay_is_not_empty(self):
        # Guards against the walk silently finding nothing and every other
        # assertion below passing vacuously.
        self.assertGreater(len(self.overlay), 5)
        self.assertGreater(len(self.rows), 20)

    def test_no_bytecode_in_the_overlay(self):
        # `python3 -m py_compile` on an overlay file drops __pycache__ right
        # next to it, and COPY would bake stale .pyc into the image.
        junk = [
            p
            for p in self.overlay
            if p.endswith(".pyc") or "__pycache__" in p.split(os.sep)
        ]
        self.assertEqual(junk, [], f"bytecode in the overlay: {junk}")

    def test_every_overlay_file_is_documented(self):
        undocumented = sorted(self.overlay - set(self.rows))
        self.assertEqual(
            undocumented,
            [],
            "these ship in the image but are absent from MANIFEST.md, so a "
            f"reader cannot tell what they change: {undocumented}",
        )

    def test_every_documented_file_ships(self):
        # new rows ship from rootfs; modified rows ship via the combined patch
        shipped = self.overlay | set(self._combined_sections()) | NOT_IN_OVERLAY
        missing = sorted(set(self.rows) - shipped)
        self.assertEqual(
            missing,
            [],
            "MANIFEST.md documents these but they are in neither rootfs nor "
            f"vllm-upstream.patch, so the build would not apply them: {missing}",
        )

    def test_modified_files_are_not_also_in_rootfs(self):
        # A modified file present in rootfs would COPY over the patched one and
        # silently pin an older content -- modified files ship ONLY via patch.
        dupes = sorted(set(self._combined_sections()) & self.overlay)
        self.assertEqual(dupes, [], f"in both patch and rootfs: {dupes}")

    def test_manifest_header_counts_match_the_tables(self):
        with open(MANIFEST) as fh:
            text = fh.read()
        claimed_mod = int(re.search(r"\*\*(\d+) modified\*\* files", text).group(1))
        claimed_new = int(re.search(r"\*\*(\d+) new\*\* files", text).group(1))

        # A row is "new" if its delta cell says so; everything else is a diff
        # against an upstream file. The build-stage replacement is neither.
        actual_new = sum(
            1
            for path, delta in self.rows.items()
            if "new" in delta and path not in NOT_IN_OVERLAY
        )
        actual_mod = sum(
            1
            for path, delta in self.rows.items()
            if "new" not in delta and path not in NOT_IN_OVERLAY
        )
        self.assertEqual(claimed_new, actual_new, "'N new' header is stale")
        self.assertEqual(claimed_mod, actual_mod, "'N modified' header is stale")

    def test_dockerfile_file_count_matches_the_overlay(self):
        with open(DOCKERFILE) as fh:
            text = fh.read()
        claimed = int(re.search(r"overlays the (\d+) new files", text).group(1))
        claimed_patched = int(re.search(r"vllm-upstream\.patch, (\d+) files", text).group(1))
        self.assertEqual(claimed_patched, len(self._combined_sections()),
                         "the Dockerfile's patched-file count is stale")
        self.assertEqual(
            claimed,
            len(self.overlay),
            "the Dockerfile's rootfs file count is stale",
        )

    @staticmethod
    def _combined_sections():
        """Split the combined vllm-upstream.patch into {path: [diff lines]}."""
        combined = os.path.join(PATCHES, "vllm-upstream.patch")
        sections, current = {}, None
        with open(combined) as fh:
            for ln in fh.read().splitlines():
                if ln.startswith(f"--- a/{SITE}/"):
                    current = ln[len(f"--- a/{SITE}/"):]
                    sections[current] = []
                if current is not None:
                    sections[current].append(ln)
        return sections

    def test_every_modified_file_has_a_reviewable_diff(self):
        # The manifest promises a reviewable diff for each modified file. A
        # missing one means a change nobody can review against upstream.
        sections = self._combined_sections()
        missing = []
        for path, delta in self.rows.items():
            if "new" in delta or path in NOT_IN_OVERLAY:
                continue
            if path not in sections:
                missing.append(path)
        self.assertEqual(sorted(missing), [],
                         f"no section in vllm-upstream.patch for: {sorted(missing)}")

    def test_manifest_deltas_match_their_patches(self):
        # The Δ column is the only summary most readers will look at. It drifts
        # silently when a patched file is edited again without regenerating the
        # diff, which is how 11 of these came to disagree at once.
        wrong = []
        sections = self._combined_sections()
        for path, delta in sorted(self.rows.items()):
            if "new" in delta or path in NOT_IN_OVERLAY:
                continue
            lines = sections.get(path)
            if lines is None:
                continue  # reported by test_every_modified_file_has_a_reviewable_diff
            add = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
            rem = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
            actual = f"+{add}" if rem == 0 else f"+{add}/-{rem}"
            if actual != delta:
                wrong.append(f"{path}: manifest {delta}, patch {actual}")
        self.assertEqual(wrong, [], "stale Δ cells: " + "; ".join(wrong))

    def test_patches_use_container_absolute_paths(self):
        # `a/opt/venv/.../vllm/x.py`, not `a/vllm/x.py`. The prefix is what
        # tells a reader where the file lives in the image, and mixing the two
        # conventions means no single -p level applies the whole set.
        bad = []
        with open(os.path.join(PATCHES, "vllm-upstream.patch")) as fh:
            for ln in fh.read().splitlines():
                if ln.startswith("--- ") and not ln.startswith(f"--- a/{SITE}/"):
                    bad.append(ln)
        self.assertEqual(bad, [], "wrong path prefix: " + "; ".join(bad))

    def test_every_patch_file_names_a_shipped_file(self):
        # Map forwards, never backwards: "__" is both the path separator in a
        # patch filename and a literal in `__init__.py`, so decoding a name is
        # ambiguous while encoding a path is not.
        expected = {p.replace("/", "__") + ".patch" for p in self.overlay}
        orphans = [
            name
            for name in os.listdir(PATCHES)
            if name.startswith("vllm__")
            and name.endswith(".patch")
            and name not in expected
        ]
        self.assertEqual(
            sorted(orphans),
            [],
            f"patches with no corresponding overlay file: {sorted(orphans)}",
        )

    def test_overlay_python_compiles(self):
        import py_compile
        import tempfile

        # cfile into a temp dir so this never writes __pycache__ into rootfs --
        # which would break test_no_bytecode_in_the_overlay on the next run.
        with tempfile.TemporaryDirectory() as tmp:
            for rel in sorted(self.overlay):
                if not rel.endswith(".py"):
                    continue
                src = os.path.join(OVERLAY, rel)
                out = os.path.join(tmp, rel.replace("/", "_") + "c")
                try:
                    py_compile.compile(src, cfile=out, doraise=True)
                except py_compile.PyCompileError as exc:
                    self.fail(f"{rel} does not compile: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
