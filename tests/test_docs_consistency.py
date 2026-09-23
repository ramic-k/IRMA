"""Documentation-consistency tests: the NJOY cliq patch line in njoy.md,
the public CHANGELOG history, and the version string kept in sync across
irma/__init__.py, CHANGELOG.md and CITATION.cff. Pure text assertions."""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def test_njoy_doc_gives_two_axis_guard_fix():
    text = (DOCS / "njoy.md").read_text(encoding="utf-8")
    # the validated one-line two-axis guard, both cliq sites
    assert "sab(1,1).gt.sab(2,1).and.sab(1,1).gt.sab(1,2)" in text, (
        "njoy.md must give the two-axis cliq guard patch"
    )


# ---------------------------------------------------------------------------
# CHANGELOG — fresh public history: the record starts at 1.0.0
# ---------------------------------------------------------------------------

def _changelog():
    return CHANGELOG.read_text(encoding="utf-8")


def test_changelog_is_the_public_history():
    """The public CHANGELOG starts at 1.0.0 (decision at the GitHub cut:
    the pre-1.0 development history stays in the internal archive). At
    most one [Unreleased] section may sit above the released sections,
    and releases must be in descending order."""
    text = _changelog()
    releases = re.findall(r"^##\s*\[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    assert releases, "CHANGELOG must carry at least one release heading"
    assert releases[-1] == "1.0.0", (
        "the oldest public release must be 1.0.0; the pre-1.0 internal "
        "history is not part of the public changelog")
    parsed = [tuple(int(x) for x in r.split(".")) for r in releases]
    assert parsed == sorted(parsed, reverse=True), (
        "CHANGELOG releases must be newest-first")
    assert text.count("[Unreleased]") <= 1


def test_version_matches_release():
    """__version__ must match the top RELEASED CHANGELOG section and CITATION.cff.

    Derived (not hardcoded) so it can't rot on a version bump: it instead pins
    that irma/__init__.py, the changelog, and the citation file stay in sync,
    which is what the CI release tag-guard also checks. CITATION.cff is in the
    net because it rotted silently through the 0.17.0 bump (it carried 0.16.0).
    """
    root = Path(__file__).resolve().parent.parent
    init = (root / "irma" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init)
    assert m is not None, "could not find __version__ in irma/__init__.py"
    version = m.group(1)
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    # first numeric "## [x.y.z]" heading -- skips the non-numeric "## [Unreleased]"
    rel = re.search(r'^##\s*\[(\d+\.\d+\.\d+)\]', changelog, re.MULTILINE)
    assert rel is not None, "no released version heading found in CHANGELOG.md"
    assert version == rel.group(1), (
        f"irma.__version__ {version!r} must match the top CHANGELOG release "
        f"{rel.group(1)!r}")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    cit = re.search(r'^version:\s*"([^"]+)"', citation, re.MULTILINE)
    assert cit is not None, "could not find version: in CITATION.cff"
    assert cit.group(1) == version, (
        f"CITATION.cff version {cit.group(1)!r} must match irma.__version__ "
        f"{version!r} -- update CITATION.cff on every release bump")
