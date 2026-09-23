"""Headless coverage for the Tk-free GUI deck-text helpers.

Exercises irma/gui/deck_text.py — the pure emit helpers (_quote, fmt_array,
emit_comment_lines) and the transactional import parser (parse_deck_to_staging)
— WITHOUT importing tkinter, so the QA2-001 HIGH quote-doubling fix and the
import staging/validation logic get CI coverage on a display-less runner.

Findings: F26 (quote helper moved Tk-free + round-trip), F20+C10 (staging
parse is transactional and complete), C11 (title/iprint preserved), C12
(noncubic controls validated with engine ranges), F23 (comment whitespace
round-trips). F22 (_parse_atoms token-count guard) lives on a Tk widget and is
covered by the display-gated GUI suite; its pure validation rule is mirrored
here against the staging parser's atom path.
"""
import pytest

from irma.core.deck import parse_leapr_input, TokenReader
from irma.gui.deck_text import (
    _quote, fmt_array, emit_comment_lines, parse_deck_to_staging,
)


def _reader(deck_text, tmp_path):
    p = tmp_path / "deck.input"
    p.write_text(deck_text)
    tokens, _raw, _start, token_lines = parse_leapr_input(str(p))
    return TokenReader(tokens, token_lines=token_lines, filename=str(p)), str(p)


def _stage(deck_text, tmp_path):
    reader, path = _reader(deck_text, tmp_path)
    return parse_deck_to_staging(reader, path)


CLASSIC_DECK = (
    "20 /\n'import test'/\n1 0 4/\n7 125./\n"
    "0.99917 20.449 2 0 0 0/\n0/\n3 4 1/\n"
    "0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n0.005 6/\n"
    "0.0 0.20 0.45 0.55 0.30 0.0/\n0. 0. 1./\n0/\n/\n"
)


# ---------------- F26: quote-doubling, Tk-free ----------------

def test_quote_doubles_embedded_single_quotes():
    assert _quote("O'Brien's / x") == "'O''Brien''s / x'"


def test_quote_roundtrip_through_tokenizer(tmp_path):
    original = "6-C-12 O'Brien's deck / rev 2"
    p = tmp_path / "c.input"
    p.write_text(f"{_quote(original)} /\n")
    tokens, _raw, _start, token_lines = parse_leapr_input(str(p))
    reader = TokenReader(tokens, token_lines=token_lines, filename=str(p))
    assert reader.read_string() == original


def test_quote_helper_import_is_tk_free():
    # Importing the module must not require tkinter (headless coverage).
    import importlib
    mod = importlib.import_module("irma.gui.deck_text")
    assert hasattr(mod, "_quote")


def test_fmt_array_wraps_five_per_line():
    out = fmt_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert out.count("\n") == 1
    assert out.splitlines()[1].strip() == f"{6.0:.6e}"


# ---------------- F20 + C10: transactional, complete staging ----------------

def test_staging_parses_classic_deck_completely(tmp_path):
    st = _stage(CLASSIC_DECK, tmp_path)
    assert st['iel'] == 0
    assert st['npr'] == 2
    assert st['nphon'] == 4
    assert st['temperatures'] == [300.0]
    assert st['delta1'] == pytest.approx(0.005)
    assert st['rho'][:2] == pytest.approx([0.0, 0.20])
    # Fields absent from a classic deck are explicit defaults, not missing,
    # so the apply step can deterministically reset them.
    assert st['noncubic'] is None
    assert st['partial_spectra'] == []
    assert st['inelastic_mode'] == 0


def test_staging_is_transactional_on_malformed_deck(tmp_path):
    # A deck that parses Card 1-5 then dies on Card 6 (nss=3) must raise from
    # the pure parser before any apply — nothing is half-applied.
    bad = (
        "20 /\n'x'/\n1 0 4/\n7 125./\n"
        "0.99917 20.449 1 0 0 0/\n3 1 1 1 1/\n3 4 1/\n"
        "0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n0.005 6/\n"
        "0.0 0.20 0.45 0.55 0.30 0.0/\n0. 0. 1./\n0/\n/\n"
    )
    with pytest.raises(ValueError, match="nss must be 0 or 1"):
        _stage(bad, tmp_path)


# ---------------- C11: title + iprint preserved ----------------

def test_staging_preserves_title_and_iprint(tmp_path):
    deck = CLASSIC_DECK.replace("1 0 4/", "1 5 4/")
    st = _stage(deck, tmp_path)
    assert st['title'] == "import test"
    assert st['iprint'] == 5


# ---------------- C12: noncubic controls validated with engine ranges ----

NONCUBIC_HEAD = (
    "20 /\n'nc'/\n1 0 4/\n1 6012./\n"
    "11.907856 4.724629 1 10 0 0/\n0 0 0 0 0/\n"
    "1 2 0 {imode}/\n"          # Card 6b: elastic_mode nat nspec inelastic_mode
    "2.46 2.46 6.7 90 90 120/\n"  # Card 6c lattice
    "6 12 11.9 0.66 0.0 1/\n0 0 0/\n"  # Card 6d atom + one position
    "6 12 11.9 0.66 0.0 1/\n0 0 0.5/\n"  # second atom
    "'/tmp/phonopy.yaml'/\n"
    "{mesh}/\n"
    "{cutoff}\n"
    "{ctrl}/\n"
    "3 4 1/\n0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n/\n"
)


def _nc_deck(mesh="20 20 20 1 0", ctrl="10000 1000 1", imode=2,
             cutoff=""):
    return NONCUBIC_HEAD.format(
        mesh=mesh, ctrl=ctrl, imode=imode, cutoff=cutoff)


def test_noncubic_valid_controls_parse(tmp_path):
    st = _stage(_nc_deck(), tmp_path)
    nc = st['noncubic']
    assert nc['ndir'] == 10000 and nc['mpdir'] == 1000
    assert nc['auto_order'] == 1
    assert nc['use_born'] == 0
    assert nc['min_phonon_energy_mev'] == 0.0


def test_min_phonon_energy_card_parse_and_legacy_compatibility(tmp_path):
    legacy = _stage(_nc_deck(), tmp_path)['noncubic']
    selected = _stage(_nc_deck(cutoff="0.5/"), tmp_path)['noncubic']
    assert legacy['min_phonon_energy_mev'] == 0.0
    assert selected['min_phonon_energy_mev'] == 0.5


@pytest.mark.parametrize("cutoff", ["nan/", "inf/"])
def test_min_phonon_energy_card_rejects_non_numbers(tmp_path, cutoff):
    with pytest.raises(ValueError, match="expected a number"):
        _stage(_nc_deck(cutoff=cutoff), tmp_path)


def test_noncubic_four_field_card6g_rejected(tmp_path):
    with pytest.raises(ValueError, match="Card 6g"):
        _stage(_nc_deck(ctrl="10000 1000 0 1"), tmp_path)


# ---------------- F23: comment whitespace round-trips ----------------

def test_comment_whitespace_preserved_through_staging(tmp_path):
    deck = (
        "20 /\n'import test'/\n1 0 4/\n7 125./\n"
        "0.99917 20.449 2 0 0 0/\n0/\n3 4 1/\n"
        "0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n0.005 6/\n"
        "0.0 0.20 0.45 0.55 0.30 0.0/\n0. 0. 1./\n0/\n"
        "'  6-C-12   LANL  '/\n/\n")
    st = _stage(deck, tmp_path)
    assert st['comments'] == ["  6-C-12   LANL  "]


def test_emit_comment_lines_roundtrip(tmp_path):
    raw = "  6-C-12   LANL  \nrev 2\n"   # Tk widget always trails a newline
    emitted = emit_comment_lines(raw)
    assert emitted == ["'  6-C-12   LANL  ' /", "'rev 2' /"]
    # And the emitted cards re-import with whitespace intact.
    p = tmp_path / "cm.input"
    p.write_text("\n".join(emitted) + "\n/\n")
    tokens, _raw, _start, token_lines = parse_leapr_input(str(p))
    reader = TokenReader(tokens, token_lines=token_lines, filename=str(p))
    assert reader.read_comment_strings() == ["  6-C-12   LANL  ", "rev 2"]


def test_emit_comment_lines_empty_block_emits_nothing():
    assert emit_comment_lines("\n") == []
    assert emit_comment_lines("") == []
    assert emit_comment_lines("   \n") == []


# ---------------- F22: _parse_atoms token-count guard ----------------
#
# the pure atom-line parser lives in the Tk-free deck_text module
# (F22/F26); importing irma.gui.app here would pull in tkinter and break
# tkinter-less CI runners.

def _parse_atoms(text):
    from irma.gui.deck_text import parse_atoms_text
    return parse_atoms_text(text)


def test_parse_atoms_underspecified_line_names_the_line():
    # npos=2 but only one (x,y,z) triplet supplied.
    with pytest.raises(ValueError, match=r"declares npos=2 but provides 3"):
        _parse_atoms("6 12 11.9 0.66 0.0 2 0 0 0\n")


def test_parse_atoms_too_few_fields_named():
    with pytest.raises(ValueError, match="must contain at least 6 fields"):
        _parse_atoms("6 12 11.9\n")


def test_parse_atoms_valid_line_parses():
    atoms = _parse_atoms("6 12 11.9 0.66 0.0 1 0 0 0.5\n")
    assert atoms[0]['npos'] == 1
    assert atoms[0]['positions'] == [(0.0, 0.0, 0.5)]


# ---------------- SPG-6 + CDX-2 + SPG-7: engine guards mirrored -------------
#
# The staging parser promises to check "engine-enforced ranges" on import.
# These tests pin the mirror of the Card 4 and Card 6b-6e guards from
# irma/core/driver.py and irma/core/crystal_cards.py: a deck the engine
# rejects must be refused on import, and a near-miss deck the engine
# accepts must still import (the mirror must not be over-broad).

def _iel10_deck(card4="31 6012. 0 0 1e-75 0",
                card5="11.898 4.739 1 10 0 0",
                card6="0",
                card6b="2 1 1 0",
                card6c="2.46 2.46 6.7 90 90 120",
                card6d="6 12 11.9 6.646 0.001 1/\n0 0 0",
                card6e="6 12 0.005 6/\n0.0 0.2 0.45 0.55 0.3 0.0/\n"):
    """A valid iel=10 inelastic_mode=0 deck (one atom type, one Card 6e
    partial spectrum) with per-card override points for the guard tests."""
    return (
        "20 /\n'iel10 import'/\n1 0 100/\n"
        f"{card4}/\n"
        f"{card5}/\n"
        f"{card6}/\n"
        f"{card6b}/\n"
        f"{card6c}/\n"
        f"{card6d}/\n"
        f"{card6e}"
        "3 4 1/\n0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n"
        "0.005 6/\n0.0 0.2 0.45 0.55 0.3 0.0/\n0. 0. 1./\n0/\n/\n"
    )


def test_iel10_near_miss_deck_still_imports(tmp_path):
    """The near-miss control: nspec=1 WITH inelastic_mode=0 is legal, as are
    sigma_inc at 0.001, a single position, and the 4-field Card 6b."""
    st = _stage(_iel10_deck(), tmp_path)
    assert st['elastic_mode'] == 2
    assert st['inelastic_mode'] == 0
    assert st['edge_group_bpd'] == 0
    assert st['atoms'][0]['Z'] == 6 and st['atoms'][0]['npos'] == 1
    assert len(st['partial_spectra']) == 1
    assert st['partial_spectra'][0]['ni'] == 6


@pytest.mark.parametrize("card6b,msg", [
    ("2 1 1 5", "inelastic_mode must be 0, 1, or 2"),
    ("2 1 1 0 25.5", "bins_per_decade must be an integer"),
])
def test_iel10_card6b_structure_checks(tmp_path, card6b, msg):
    with pytest.raises(ValueError, match=msg):
        _stage(_iel10_deck(card6b=card6b), tmp_path)


def test_iel10_card6e_non_integral_z_rejected(tmp_path):
    with pytest.raises(ValueError, match="Z must be an integer"):
        _stage(_iel10_deck(
            card6e="6.5 12 0.005 6/\n0.0 0.2 0.45 0.55 0.3 0.0/\n"), tmp_path)


def test_iel10_phonopy_mode_ncold_rejected(tmp_path):
    with pytest.raises(ValueError, match="ncold/nsk"):
        _stage(_iel10_deck(card5="11.898 4.739 1 10 4 0",
                           card6b="2 1 0 2", card6e=""), tmp_path)


# ---------------- SPG-7: Card 4 checked conversion --------------------------

@pytest.mark.parametrize("card4,msg", [
    ("31 6012. 0 0 1e-75 1.9", "iint must be an integer"),
    ("31 6012. 0 1.5 1e-75 0", "ilog must be an integer"),
])
def test_card4_flags_use_checked_conversion(tmp_path, card4, msg):
    """A non-integral iint=1.9 used to import as the valid lin-lin flag 1
    (silent int() truncation) although the engine rejects the deck."""
    with pytest.raises(ValueError, match=msg):
        _stage(_iel10_deck(card4=card4), tmp_path)


def test_card4_exactly_integral_flags_still_import(tmp_path):
    """NJOY list-directed compatibility: exactly-integral floats (1.0) stay
    accepted, mirroring reader.to_int."""
    st = _stage(_iel10_deck(card4="31 6012. 1.0 1.0 1e-75 1.0"), tmp_path)
    assert st['isabt'] == 1 and st['ilog'] == 1 and st['iint'] == 1


def test_natural_element_A0_is_accepted(tmp_path):
    """A = 0 is ENDF's natural-element code, not an error.

    Card 6d's A is an identity key (the constants on the row carry the
    physics), matched against Card 4's za = 1000*Z + A. Naming the
    natural element on both cards lets an evaluator write natural
    carbon honestly instead of borrowing C-12's mass number for
    natural-abundance constants (which is what the reference decks do).
    """
    staged = _stage(_iel10_deck(card4="31 6000. 0 0 1e-75 0",
                                card6d="6 0 11.9 6.646 0.001 1/\n0 0 0",
                                card6e="6 0 0.005 6/\n"
                                       "0.0 0.2 0.45 0.55 0.3 0.0/\n"),
                    tmp_path)
    assert staged["atoms"][0]["A"] == 0
    assert int(float(staged["za"])) == 6000
