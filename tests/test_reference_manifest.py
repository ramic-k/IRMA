"""Content-address the committed reference artifacts (SHA256SUMS.txt).

The pinned tests (byte-exact tapes, noncubic fast-CI pins, the Euphonic
cross-code harness, the PyChop reference gate) are only as trustworthy as their
references. Git already makes any change to a reference blob a reviewable
diff; this test adds the active assertion git tracking lacks, so a silently
re-gzipped, truncated, or swapped reference fails LOUDLY here with a named
hash mismatch instead of a mysterious downstream byte diff or pin mismatch.
Pin updates after an intentional reference change by regenerating
SHA256SUMS.txt (see its header).
"""
import hashlib
import os

import pytest

TESTS_DIR = os.path.dirname(__file__)
MANIFEST = os.path.join(TESTS_DIR, "SHA256SUMS.txt")


def _parse_manifest(path):
    """Yield (expected_sha256, relpath) for each non-comment manifest line."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            sha, _, rel = line.partition("  ")   # two-space shasum format
            assert rel, f"malformed manifest line: {line!r}"
            entries.append((sha.strip(), rel.strip()))
    return entries


_ENTRIES = _parse_manifest(MANIFEST)


def test_manifest_is_nonempty_and_well_formed():
    assert _ENTRIES, "SHA256SUMS.txt has no entries"
    for sha, rel in _ENTRIES:
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), sha


@pytest.mark.parametrize("expected_sha, relpath",
                         _ENTRIES,
                         ids=[rel for _, rel in _ENTRIES])
def test_reference_artifact_matches_manifest(expected_sha, relpath):
    path = os.path.join(TESTS_DIR, relpath)
    assert os.path.exists(path), f"manifest references a missing file: {relpath}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == expected_sha, (
        f"{relpath} sha256 mismatch: the committed reference artifact changed. "
        "If intentional, regenerate tests/SHA256SUMS.txt (see its header)."
    )
