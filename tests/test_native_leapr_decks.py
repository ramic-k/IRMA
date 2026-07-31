"""Format checks for the committed IRMA decks under
tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/.

These decks are derived from the reference ``tsl-*.leapr`` evaluations by
``leapr_to_irma_input.py`` and are used for deck-level validation against the
native LEAPR/NJOY reference ENDF (see that directory's README.md). The full
physics reproduction is a slow, reference-data-dependent harness
(``validate_native_leapr_endf.py``); here we only pin, cheaply, that each
derived deck still PARSES into the expected LEAPR control cards. This catches
converter/format regressions (Card 1 reformatting, title quoting, the
iel/grid/temperature header) in milliseconds without running the
scattering-law calculation.
"""
import os
import re

import pytest

from irma.core.engine import parse_leapr_input, TokenReader

DECK_DIR = os.path.join(os.path.dirname(__file__),
                        "native_LEAPR_NJOY_ENDF_validation", "leapr_decks")

# (nout, ntempr, mat, za, iel, nalpha, nbeta, lat, nss) expected from each
# derived deck. All are pure converter output reproducing NJOY-LEAPR
# references: graphite (iel=1), iron (iel=6) and aluminum (iel=4) built-in
# coherent elastic, polyethylene (iel=0) with a free-gas secondary scatterer,
# liquid methane (trans diffusion + 4 discrete oscillators + free-gas
# secondary), liquid ortho-/para-hydrogen (coldh + Skold + trans + discre,
# with a fresh detail block at every temperature) and BeO (mixed-moderator
# two-scatterer). nss is the Card 6 secondary-scatterer count (1 for the
# bound/free secondary decks, 0 otherwise).
EXPECTED = {
    "tsl-crystalline-graphite.input": (24, 10, 30, 130.0, 1, 150, 400, 1, 0),
    "tsl-026_Fe_056.input":           (25,  6, 101, 26056.0, 6, 82, 151, 1, 0),
    "tsl-013_Al_027.input":           (25,  6, 101, 13027.0, 4, 149, 144, 1, 0),
    "tsl-HinCH2.input":               (24, 15, 37, 137.0, 0, 200, 500, 1, 1),
    "tsl-l-CH4.input":                (20,  1, 33, 1001.0, 0, 89, 81, 0, 1),
    "tsl-ortho-H.input":              (60,  7, 3, 1001.0, 0, 193, 298, 0, 0),
    "tsl-para-H.input":               (60,  7, 2, 1001.0, 0, 193, 298, 0, 0),
    "tsl-BeO.input":                  (20,  8, 27, 127.0, 3, 90, 117, 1, 1),
}

# Expected count of bare negative-temperature cards ("reuse the previous
# spectrum" entries — fragile converter territory per the directory README)
# in each derived deck. A converter regression that drops or duplicates one
# leaves ntempr (Card 3) unchanged, so the header test alone cannot catch it;
# this count pins the actual temperature-block structure. graphite has 9
# negatives (296 K + 9 reuse), Fe/Al 5 each, CH2 14, BeO 14 (7 per scatterer
# block), and the cold-hydrogen / liquid-methane decks carry a full spectrum
# at every temperature so have none.
EXPECTED_NEG_TEMP_CARDS = {
    "tsl-crystalline-graphite.input": 9,
    "tsl-026_Fe_056.input":           5,
    "tsl-013_Al_027.input":           5,
    "tsl-HinCH2.input":               14,
    "tsl-l-CH4.input":                0,
    "tsl-ortho-H.input":              0,
    "tsl-para-H.input":               0,
    "tsl-BeO.input":                  14,
}

# A bare negative-temperature card: optional leading whitespace, a negative
# number (int or float, incl. the trailing-dot form ``-400.`` the Fe deck
# uses), optional trailing whitespace and card terminator. The committed
# decks contain no inline negative-number data cards, so this matches only
# the reuse-spectrum temperature lines.
_NEG_TEMP_RE = re.compile(r"^\s*-\d+(?:\.\d*)?\s*/?\s*$")


def _read_header(path):
    """Drive a TokenReader through Cards 1-7 exactly as run_leapr does."""
    tokens = parse_leapr_input(path)[0]
    r = TokenReader(tokens)
    nout = r.read_ints(1)[0]                                   # Card 1
    title = r.read_string()                                    # Card 2
    ntempr, iprint, nphon = r.read_ints(3, defaults=[1, 1, 100])  # Card 3
    mat, za, isabt, ilog, smin = r.read_floats(5, defaults=[0, 0, 0, 0, 1e-75])  # Card 4
    awr, spr, npr, iel, ncold, nsk = r.read_floats(6, defaults=[0] * 6)          # Card 5
    nss = r.read_floats(5, defaults=[0] * 5)[0]                # Card 6
    # All committed decks use iel<10, so Card 7 follows Card 6 directly.
    nalpha, nbeta, lat = r.read_ints(3, defaults=[0, 0, 0])    # Card 7
    return dict(nout=nout, title=title, ntempr=ntempr, mat=int(mat), za=za,
                iel=int(iel), nalpha=nalpha, nbeta=nbeta, lat=lat, nss=int(nss))


@pytest.mark.parametrize("fname", sorted(EXPECTED))
def test_derived_deck_header(fname):
    path = os.path.join(DECK_DIR, fname)
    assert os.path.exists(path), f"missing derived deck {fname}"
    h = _read_header(path)
    nout, ntempr, mat, za, iel, nalpha, nbeta, lat, nss = EXPECTED[fname]
    assert h["nout"] == nout
    assert h["ntempr"] == ntempr
    assert h["mat"] == mat
    assert h["za"] == za
    assert h["iel"] == iel
    assert (h["nalpha"], h["nbeta"]) == (nalpha, nbeta)
    assert h["lat"] == lat
    assert h["nss"] == nss


@pytest.mark.parametrize("fname", sorted(EXPECTED))
def test_derived_deck_title_is_quoted_nonempty(fname):
    """The reconstructed title must survive as a full (non-truncated) string."""
    h = _read_header(os.path.join(DECK_DIR, fname))
    assert h["title"].strip()


@pytest.mark.parametrize("fname", sorted(EXPECTED_NEG_TEMP_CARDS))
def test_derived_deck_negative_temperature_card_count(fname):
    """Pin the count of reuse-spectrum (negative-temperature) cards.

    ntempr (Card 3) is asserted by the header test, but a converter regression
    that drops or duplicates a negative-temperature card does not change
    ntempr and would otherwise only surface in the slow non-CI physics
    harness. Counting the bare negative cards catches that cheaply.
    """
    path = os.path.join(DECK_DIR, fname)
    with open(path) as f:
        n_neg = sum(1 for ln in f if _NEG_TEMP_RE.match(ln))
    assert n_neg == EXPECTED_NEG_TEMP_CARDS[fname]


def test_no_njoy_wrapper_lines():
    """Derived decks must not carry NJOY 'leapr'/'stop' wrapper lines."""
    for fname in EXPECTED:
        with open(os.path.join(DECK_DIR, fname)) as f:
            lowers = [ln.strip().lower() for ln in f]
        assert "leapr" not in lowers, f"{fname} still has a 'leapr' module line"
        assert "stop" not in lowers, f"{fname} still has a 'stop' line"


def _load_converter():
    """Load the standalone converter (not an importable package)."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__),
                        "native_LEAPR_NJOY_ENDF_validation",
                        "leapr_to_irma_input.py")
    spec = importlib.util.spec_from_file_location("leapr_to_irma_input", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_title_preserves_embedded_apostrophe():
    """An unquoted title with an apostrophe must survive the round-trip.

    Fortran-doubling the embedded quote keeps IRMA's quote scanner from
    treating the apostrophe as the closing quote and truncating the title.
    """
    conv = _load_converter()
    emitted = conv._normalize_title("Stedman's Al data / source note\n")
    assert emitted == "'Stedman''s Al data'/\n"

    # Round-trip the emitted line through IRMA's title tokenizer.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".input", delete=False) as f:
        f.write("25 /\n")
        f.write(emitted)
        f.write("1 1 100 /\n")
        path = f.name
    try:
        tokens = parse_leapr_input(path)[0]
        reader = TokenReader(tokens)
        reader.read_ints(1)               # nout
        title = reader.read_string()      # the reconstructed title
    finally:
        os.unlink(path)
    assert title == "Stedman's Al data"


def test_normalize_title_passthrough_cases():
    """Quoted titles are untouched; apostrophe-free unquoted titles still wrap."""
    conv = _load_converter()
    assert conv._normalize_title("'Already quoted'/\n") == "'Already quoted'/\n"
    assert conv._normalize_title("graphite\n") == "'graphite'/\n"
