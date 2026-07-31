"""Documentation-consistency tests.

These tests pin the manual's NJOY guidance and the changelog so the
two stale/misleading claims fixed in QA3 cannot silently reappear:

- F5: the stock-NJOY THERMR coherent-mode-2 failure is the `cliq`
  liquid-extrapolation guard bug (decay tested along alpha but the
  formula's sign needs decay along beta too), producing ~1e91-barn
  garbage. The docs must name it, give the one-line two-axis guard fix,
  note mode-1/most materials are unaffected, and must NOT claim a
  uniform beta grid is required for NJOY (it does not prevent the bug).
- CHANGELOG contract since the 1.0.0 public cut: the public history
  starts at 1.0.0 (the pre-1.0 development history is internal), and
  the version string stays pinned three ways (__init__, CHANGELOG,
  CITATION.cff).

Pure text assertions over the repository docs — no imports, no display.
"""
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def _read(rel):
    return (DOCS / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F5 — NJOY guidance names the cliq bug and gives the two-axis guard fix
# ---------------------------------------------------------------------------

def test_njoy_doc_names_the_cliq_bug():
    text = _read("njoy.md").lower()
    assert "cliq" in text, "njoy.md must name the THERMR cliq bug"
    assert "1e91" in text, "njoy.md must state the ~1e91-barn garbage symptom"
    assert "thermr.f90" in text, "njoy.md must point at the thermr.f90 source"


def test_njoy_doc_gives_two_axis_guard_fix():
    text = _read("njoy.md")
    # The validated one-line two-axis guard, both cliq sites.
    assert "sab(1,1).gt.sab(2,1).and.sab(1,1).gt.sab(1,2)" in text, (
        "njoy.md must give the two-axis cliq guard patch"
    )


def test_njoy_doc_says_mode1_and_most_materials_unaffected():
    text = _read("njoy.md").lower()
    assert "0/1" in text or "mode-1" in text or "mode 1" in text
    assert "unaffected" in text, (
        "njoy.md must state that mode-1/0 tapes and most materials are unaffected"
    )


def test_njoy_doc_has_a_symptom_table():
    text = _read("njoy.md")
    # A markdown table whose header row mentions the patch.
    assert "After the two-axis patch" in text, (
        "njoy.md must carry the before/after THERMR symptom table"
    )


def test_njoy_doc_points_upstream():
    text = _read("njoy.md").lower()
    assert "njoy/njoy2016" in text, "njoy.md must point the patch upstream"


# ---------------------------------------------------------------------------
# F5 — the stale "uniform beta grid is required for NJOY" claim is gone
# ---------------------------------------------------------------------------

# Pages that previously prescribed a uniform beta grid for NJOY-bound tapes.
# grids.md and input-reference.md joined the list in QA4: their copies of the
# stale advice survived the QA3 sweep because they were not parametrized here.
_PAGES_PREVIOUSLY_CLAIMING_UNIFORM_GRID = [
    "njoy.md",
    "troubleshooting.md",
    "quickstart.md",
    "gui.md",
    "index.md",
    "grids.md",
    "input-reference.md",
]

# Phrasings that assert a uniform grid is needed / recommended for NJOY.
_STALE_PATTERNS = [
    r"uniform energy grid for tapes bound for njoy",
    r"use a uniform energy \(beta\) grid for tapes bound for njoy",
    r"njoy-bound tapes prefer a uniform energy grid",
    r"njoy-bound tapes\b.*\buniform",
    r"generate (?:the law |with )?(?:on )?a\s+\*?\*?uniform",
    r"build the beta grid uniform",
    r"uniform-energy-grid guidance",
]


@pytest.mark.parametrize("page", _PAGES_PREVIOUSLY_CLAIMING_UNIFORM_GRID)
def test_no_stale_uniform_grid_requirement(page):
    text = _read(page).lower()
    for pat in _STALE_PATTERNS:
        m = re.search(pat, text)
        assert m is None, (
            f"{page} still asserts a uniform beta grid for NJOY "
            f"(matched {pat!r}: {m.group(0)!r})"
        )


@pytest.mark.parametrize("page", _PAGES_PREVIOUSLY_CLAIMING_UNIFORM_GRID)
def test_coherent_mode2_pages_route_to_cliq_guidance(page):
    """Every page that used to push the uniform-grid remedy must now point
    at the cliq bug instead (directly or via njoy.md)."""
    text = _read(page).lower()
    assert ("cliq" in text) or ("njoy.md" in text), (
        f"{page} must now reference the cliq bug or link to njoy.md"
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
