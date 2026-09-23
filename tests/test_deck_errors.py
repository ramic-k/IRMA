"""Friendly deck-validation errors (DeckError with card/line context).

Malformed input decks must fail with a message naming the offending card,
what was expected, what was found, and the input line — never with cryptic
downstream numerics (ZeroDivisionError, NaN conversions) and never by running
to completion on garbage (a non-monotonic grid used to produce silent wrong
output). Each case here was a real observed failure mode.
"""
import os
import tempfile

import pytest

from irma.core.engine import run_leapr, DeckError

# A minimal well-formed classic deck the cases below mutate.
GOOD = """20 /
'deck-error test'/
1 1 4/
1 1./
1.0 20.0 1 0 0/
0/
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
300/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
/
"""


def _run(deck_text):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, os.path.join(d, "out.endf"))


def _expect(deck_text, *fragments):
    with pytest.raises(DeckError) as exc:
        _run(deck_text)
    msg = str(exc.value)
    for frag in fragments:
        assert frag in msg, f"expected {frag!r} in error message:\n  {msg}"
    return msg


def test_good_deck_runs():
    _run(GOOD)   # sanity: the template itself is valid


def test_temperatures_must_increase():
    # the edge thinning uses the first temperature; THERMR reads them in order
    deck = GOOD.replace("1 1 4/", "2 1 4/").replace("0/\n/\n", "0/\n-200/\n/\n")
    _expect(deck, "temperatures must increase")


def test_non_ascii_comment_rejected():
    # a wider character shifts the fixed MAT/MF/MT columns of the MF1 record
    _expect(GOOD.replace("0/\n/\n", "0/\n'M\u00e1rquez'/\n/\n"), "non-ASCII")


def test_cutoff_that_removes_every_mode_is_refused():
    # a one-value card before Card 6g is the cutoff; '500 /' (e.g. a Card 6g
    # missing mpdir) would remove every graphite mode
    pytest.importorskip("phonopy")
    from test_noncubic_fast_ci import _DECK, _YAML
    deck = _DECK.format(mode=2, yaml=_YAML).replace("40 20/\n", "500/\n40 20/\n")
    _expect(deck, "removes every phonon mode")


def test_empty_file():
    _expect("", "no LEAPR cards found")


def test_text_where_number_expected():
    deck = GOOD.replace("1 1 4/", "one 1 4/")
    _expect(deck, "Card 3", "expected a number", "'one'", "line 3")


def test_truncated_input():
    deck = "\n".join(GOOD.splitlines()[:8]) + "\n0.1 0.5\n"
    _expect(deck, "Card 9", "beta", "input ended")


def test_array_terminated_early_by_slash():
    """ni says 6 rho values but the card supplies 4 — caught at Card 12,
    not as a misaligned cascade further down."""
    deck = GOOD.replace("0.0 0.20 0.45 0.55 0.30 0.0/", "0.0 0.20 0.45 0.0/")
    _expect(deck, "Card 12", "4 of 6", "check the count")


def test_zero_nalpha():
    deck = GOOD.replace("3 4 1/", "0 4 1/")
    _expect(deck, "Card 7", "nalpha")


def test_invalid_iel():
    deck = GOOD.replace("1.0 20.0 1 0 0/", "1.0 20.0 1 7 0/")
    _expect(deck, "Card 5", "iel", "got 7")


def test_nonmonotonic_beta_grid():
    """A non-monotonic beta grid must fail at deck read, not write silent garbage."""
    deck = GOOD.replace("0.0 0.6 2.0 6.0/", "0.0 2.0 0.6 6.0/")
    _expect(deck, "Card 9", "strictly increasing", "0.6", "2",
            "input line 9", "deck.input")


def test_nonpositive_alpha():
    deck = GOOD.replace("0.05 1.0 8.0/", "0.0 1.0 8.0/")
    _expect(deck, "Card 8", "alpha")


def test_zero_tbeta():
    """A zero tbeta must fail at deck read, not as a later ZeroDivisionError."""
    deck = GOOD.replace("0. 0. 1./", "0. 0. 0./")
    _expect(deck, "Card 13", "tbeta")


def test_zero_temperature():
    deck = GOOD.replace("300/", "0/", 1)
    _expect(deck, "temperature", "nonzero")


def test_missing_temperature_block():
    """ntempr=2 but only one temperature card supplied: the comment
    terminator gets consumed as an empty temperature card and is rejected."""
    deck = GOOD.replace("1 1 4/", "2 1 4/")
    _expect(deck, "temperature 2 of 2", "nonzero")


def test_negative_first_temperature_still_reads_block():
    """itemp==0 always reads the detail block even for a negative card."""
    deck = GOOD.replace("300/", "-300/", 1)
    _run(deck)   # must not raise


def test_fortran_d_exponents_accepted():
    """NJOY-style D-exponent numbers must parse like their E equivalents."""
    deck = (GOOD.replace("0.005 6/", "5.0d-3 6/")
                .replace("300/", "3.0D2/", 1))
    _run(deck)   # must not raise


def test_bad_elastic_mode_is_deck_error():
    """iel=10 semantic validation must present as a DeckError with card
    context, not a bare ValueError traceback."""
    deck = """20 /
'iel10 semantic'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
3 1 0 0/
"""
    _expect(deck, "Card 6b", "elastic_mode", "got 3")


def test_principal_scatterer_mismatch_is_deck_error():
    deck = """20 /
'iel10 za mismatch'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 0/
2.46 2.46 6.7 90. 90. 120./
4 9 8.93 7.79 0.0018 1/
0.0 0.0 0.0/
"""
    _expect(deck, "Card 4 ZA=6012 (C-12)", "Card 6d", "No row has Z=6")

_SPLIT_PRINCIPAL_DECK = """20 /
'duplicate principal types'/
1 1 5/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 2 0 1/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.5/
'/nonexistent/phonopy.yaml' /
8 8 8 1 0 /
100 100 /
"""


def test_principal_split_across_atom_types_is_merged_in_phonopy_modes():
    """QA4 F3 (full fix): with inelastic_mode=1/2 a principal (Z,A) spread
    over several Card 6d atom types is MERGED into one group (the MT4 law
    accumulates over every represented site), so the deck must get PAST the
    principal bookkeeping and only fail later on the nonexistent phonopy
    model."""
    with pytest.raises(RuntimeError, match="phonopy"):
        _run(_SPLIT_PRINCIPAL_DECK)


def test_principal_split_with_mismatched_data_rejected():
    """Merging is only valid when the duplicate entries carry identical
    nuclear data; a differing b_coh must stay a loud deck error."""
    deck = _SPLIT_PRINCIPAL_DECK.replace(
        "6 12 11.9 6.646 0.001 1/\n0.0 0.0 0.5/",
        "6 12 11.9 7.000 0.001 1/\n0.0 0.0 0.5/")
    _expect(deck, "differ in awr/b_coh/sigma_inc", "merged into one group")


_BE_SEF_DECK = """20 /
'be sef'/
1 1 4/
26 4009./
8.93478 6.153875 1 10 0/
0/
1 {nat} 0 0/
2.2866 2.2866 3.5833 90.0 90.0 120.0/
{rows}
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
296/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
/
"""


def test_mode0_sef_split_principal_writes_the_same_comb():
    """A principal split over two Card 6d rows is merged in mode 0 too, so
    the SEF coherent comb matches the one-row deck (it used to double)."""
    one = _BE_SEF_DECK.format(nat=1, rows="4 9 8.93478 7.79 0.0018 2/\n"
                              "0.33333333 0.66666667 0.75  0.66666667 0.33333333 0.25/")
    split = _BE_SEF_DECK.format(nat=2, rows="4 9 8.93478 7.79 0.0018 1/\n"
                                "0.33333333 0.66666667 0.75/\n"
                                "4 9 8.93478 7.79 0.0018 1/\n"
                                "0.66666667 0.33333333 0.25/")
    mt2 = []
    for deck in (one, split):
        d = tempfile.mkdtemp()
        inp, out = os.path.join(d, "be.input"), os.path.join(d, "be.endf")
        with open(inp, "w") as f:
            f.write(deck)
        run_leapr(inp, out)
        with open(out) as f:
            mt2.append([ln[:66] for ln in f if ln[70:75] == " 7  2"])
    assert len(mt2[0]) > 10 and mt2[1] == mt2[0]


# An iel=10 mode-1 deck that reaches Card 6g without needing phonopy
# (the 6g check fires before the phonopy mesh load is attempted).
_MODE1_HEAD = """20 /
'card 6g format deck'/
1 1 5/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 1/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
'/nonexistent/phonopy.yaml' /
8 8 8 1 0 /
"""


def test_ncold_nsk_rejected_in_phonopy_modes():
    """coldh/skold consume S(kappa) tables the phonopy-backed MT4 path never
    builds; the combination must die as a deck error, not a mid-run crash."""
    deck = _MODE1_HEAD.replace("11.9 4.74 1 10 0 0/",
                               "11.9 4.74 1 10 1 0/") + "100 100 /\n"
    _expect(deck, "ncold/nsk", "inelastic_mode=1")


def test_secondary_scatterer_rejected_in_phonopy_modes():
    deck = _MODE1_HEAD.replace("0/\n1 1 0 1/",
                               "1 1. 15.85 3.88 1/\n1 1 0 1/") + "100 100 /\n"
    _expect(deck, "secondary scatterer", "inelastic_mode=1")


def test_nonzero_nspec_rejected_in_phonopy_modes():
    """Card 6e partial spectra have no role when MT4/DW come from Phonopy:
    nspec != 0 with inelastic_mode=1/2 is a deck error, never silently
    consumed and ignored."""
    deck = _MODE1_HEAD.replace("1 1 0 1/", "1 1 1 1/") + "100 100 /\n"
    _expect(deck, "Card 6b", "nspec must be 0", "inelastic_mode=1")


# ENG-1: the generalized (iel=10) MF7/MT2 builder reads only the last-computed
# Debye-Waller array, which the bound (b7=0) two-pass merge leaves holding the
# SECONDARY scatterer's data. The combination must die at Card 6b, not write a
# tape whose whole elastic section comes from the wrong species. An analytic
# (b7=1 free gas / b7=2 diffusion) secondary is single-pass — the principal's
# Debye-Waller data stays in place — so it must stay legal.
_IEL10_MODE0_BOUND_SECONDARY = """20 /
'iel10 mode0 + bound two-pass secondary (must reject)'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
1 0. 11.9 4.74 1/
1 1 0 0/
"""


def test_iel10_bound_secondary_rejected():
    _expect(_IEL10_MODE0_BOUND_SECONDARY,
            "Card 6 nss=1", "b7=0", "iel=10", "Debye-Waller")


def test_iel10_freegas_secondary_gets_past_the_dw_guard():
    """b7=1 (free gas) is single-pass: dwpix keeps the principal's DW, so the
    guard must NOT fire. The deck head ends after Card 6b, so the parse fails
    at Card 6c — proving it got PAST the guard."""
    deck = _IEL10_MODE0_BOUND_SECONDARY.replace("1 0. 11.9 4.74 1/",
                                                "1 1. 11.9 4.74 1/")
    msg = _expect(deck, "Card 6c")
    assert "Debye-Waller" not in msg and "two-pass" not in msg, (
        f"analytic b7=1 secondary wrongly rejected by the iel=10 DW guard: {msg}")


def test_five_field_card_6g_rejected():
    """A 5-field Card 6g (extra trailing values) must fail loudly with
    the supported layout named, never be silently reinterpreted."""
    deck = _MODE1_HEAD + "100 100 1 0 0 /\n"
    _expect(deck, "Card 6g", "2 values plus an optional 3rd value")


def test_new_card_6g_parses_then_fails_at_phonopy_load():
    """The new 2-field Card 6g passes parsing; with a bogus phonopy path the
    deck then fails at the mesh load, proving 6g itself was accepted."""
    deck = _MODE1_HEAD + "100 100 /\n"
    with pytest.raises(RuntimeError, match="phonopy mesh"):
        _run(deck)


def test_stray_empty_card_before_array_is_deck_error():
    """A '/' on its own line where an array should start used to be silently
    skipped, reading the array from the NEXT card and misaligning every card
    after it."""
    deck = GOOD.replace("0.005 6/", "0.005 6/\n/")
    _expect(deck, "empty '/' card", "stray terminator")


def test_zero_oscillator_energy_is_deck_error():
    """A zero oscillator energy used to flow into discre and divide by zero."""
    deck = GOOD.replace("0/\n/", "2/\n0.0 0.2/\n0.3 0.2/\n/")
    _expect(deck, "oscillator energies", "> 0")


def test_negative_oscillator_weight_is_deck_error():
    deck = GOOD.replace("0/\n/", "1/\n0.2/\n-0.3/\n/")
    _expect(deck, "oscillator weights")


def test_invalid_b7_is_deck_error():
    deck = GOOD.replace("0/\n3 4 1/", "1 5. 16.0 4.0 1/\n3 4 1/")
    _expect(deck, "b7", "got 5")


# ---------- generalized-card validation ----------

_GEN_HEAD = """20 /
'gen validation deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 0/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 4/
0.0 0.0 0.0  0.0 0.0 0.5  0.333333 0.666667 0.0  0.666667 0.333333 0.5/
"""


def test_nbeta_one_rejected_at_card7():
    """Every beta-grid consumer (trans slopes, coldh tables, NJOY sbfill)
    needs a two-point bracket; nbeta=1 crashed mid-kernel."""
    deck = GOOD.replace("3 4 1/", "3 1 1/").replace(
        "0.0 0.6 2.0 6.0/", "0.0/")
    _expect(deck, "Card 7", "nbeta must be >= 2")


def test_integer_field_rejects_non_integral_float():
    """1.9 on an integer-coded field is a malformed deck, not a 1."""
    deck = GOOD.replace("3 4 1/", "3 4.9 1/")
    _expect(deck, "expected an integer", "4.9")


def test_integer_field_accepts_exactly_integral_float():
    deck = GOOD.replace("3 4 1/", "3.0 4.0 1.0/")
    _run(deck)   # NJOY list-directed compatibility: 4.0 is 4


def test_card6c_nonpositive_lattice_length_rejected():
    """CX2-10: a non-positive lattice edge silently collapses the unit-cell
    volume downstream; reject it at parse time with Card 6c context."""
    deck = _GEN_HEAD.replace("2.46 2.46 6.7 90. 90. 120./",
                             "0.0 2.46 6.7 90. 90. 120./")
    _expect(deck, "Card 6c", "lattice a", "> 0")


def test_card6c_out_of_range_angle_rejected():
    deck = _GEN_HEAD.replace("2.46 2.46 6.7 90. 90. 120./",
                             "2.46 2.46 6.7 90. 90. 200./")
    _expect(deck, "Card 6c", "lattice angle gamma", "(0, 180)")


def test_card6d_zero_npos_rejected():
    deck = _GEN_HEAD.replace("6 12 11.9 6.646 0.001 4/",
                             "6 12 11.9 6.646 0.001 0/")
    _expect(deck, "Card 6d", "npos must be >= 1")


def test_card6d_nonpositive_awr_rejected():
    deck = _GEN_HEAD.replace("6 12 11.9 6.646 0.001 4/",
                             "6 12 0.0 6.646 0.001 4/")
    _expect(deck, "Card 6d", "awr must be > 0")


def test_card6d_negative_sigma_inc_rejected():
    deck = _GEN_HEAD.replace("6 12 11.9 6.646 0.001 4/",
                             "6 12 11.9 6.646 -0.5 4/")
    _expect(deck, "Card 6d", "sigma_inc must be >= 0")


def test_card6e_bad_spectrum_rejected():
    """Card 6e partial spectra obey the same validity rules as the classic
    Card 11/12 spectrum they replace."""
    head = _GEN_HEAD.replace("1 1 0 0/", "1 1 1 0/")
    deck_zero_delta = head + "6 12 0.0 4/\n0.0 0.2 0.5 0.3/\n"
    _expect(deck_zero_delta, "Card 6e", "delta", "must be > 0")
    deck_all_zero = head + "6 12 0.005 4/\n0.0 0.0 0.0 0.0/\n"
    _expect(deck_all_zero, "Card 6e", "rho values are all zero")
    deck_neg = head + "6 12 0.005 4/\n0.0 -0.2 0.5 0.3/\n"
    _expect(deck_neg, "Card 6e", "rho values must be >= 0")


_MODE2_6F_HEAD = """20 /
'card 6f validation deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 2/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
'/nonexistent/phonopy.yaml' /
"""


def test_card6f_semantic_checks():
    _expect(_MODE2_6F_HEAD + "0 8 8 1 0 /\n100 100 /\n",
            "Card 6f", "mesh dimensions must all be >= 1")
    _expect(_MODE2_6F_HEAD + "8 8 8 0 0 /\n100 100 /\n",
            "Card 6f", "ncpu must be >= 1")
    _expect(_MODE2_6F_HEAD + "8 8 8 1 2 /\n100 100 /\n",
            "Card 6f", "use_born must be 0 or 1")


def test_missing_string_card_fails_loudly():
    """A numeric token where the phonopy.yaml path card belongs (i.e. the
    path card is missing) must not be silently stringified."""
    deck = _MODE2_6F_HEAD.replace("'/nonexistent/phonopy.yaml' /\n", "") \
        + "8 8 8 1 0 /\n100 100 /\n"
    _expect(deck, "expected a quoted string", "number")


def test_numeric_title_accepted():
    """Card 2 is free text: an unquoted numeric title must not be rejected
    by the strict string reader used for path cards."""
    deck = GOOD.replace("'deck-error test'/", "1234/")
    _run(deck)   # must not raise


# ---------- unquoted paths and the cold-law two-pass conflict ----------
# An unquoted phonopy/BORN path containing '/' would be truncated at the first
# slash (the card terminator); the two-pass merge (nss>0, b7=0) cannot carry
# the cold-H asymmetric law (ssp, ncold>0).

# iel=10 inelastic_mode=2 head reaching the Card 6f-1 path read, with the path
# written UNQUOTED so the '/' truncates `nonexistent/phonopy.yaml` -> `nonexistent`.
_UNQUOTED_PATH_DECK = """20 /
'unquoted path deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 2/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
nonexistent/phonopy.yaml /
"""


def test_unquoted_path_with_slash_rejected():
    with pytest.raises(DeckError, match="truncated at"):
        _run(_UNQUOTED_PATH_DECK)


# Same ortho-H coldh deck as the byte-exact minitape (ncold=1), but Card 6 now
# requests a secondary scatterer (nss=1) -> the dropped-ssp combination.
_COLDH_PLUS_SECONDARY = """20 /
'coldh + secondary scatterer (must reject)'/
1 1 20/
3 1001. 0 0 1e-100/
.99917 20.43634 2 0 1 2/
1 0 2.0 20.0 1/
6 10/
1e-4 5e-4 0.0025 0.01 0.05 0.25/
0.0 0.5 1.0 2.0 4.0 7.0 11.0 16.0 22.0 30.0/
14.0/
0.0005 8/
0.0 0.4 0.9 1.0 0.8 0.5 0.2 0.0/
0.1104682205 1.124899936572044 0.3895317795/
1/
0.546/
0.166666666666/
12 0.05/
0.4 0.7 1.3 1.15 0.95 1.0 1.02 0.99 1.0 1.0 1.0 1.0/
0.02144/
'mini coldh reference'/
/
"""


def test_secondary_scatterer_with_cold_law_rejected():
    with pytest.raises(DeckError, match="two-pass"):
        _run(_COLDH_PLUS_SECONDARY)


_BARE_FILENAME_PATH_DECK = """20 /
'bare filename + glued terminator'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 2/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
phonopy.yaml/
"""


def test_bare_filename_with_glued_terminator_not_rejected():
    # `phonopy.yaml/` is a valid bare filename + the legal glued '/' terminator,
    # NOT a truncated path -- it must reach the mesh loader (and fail there on
    # the missing file), not be rejected by the truncation guard.
    try:
        _run(_BARE_FILENAME_PATH_DECK)
    except Exception as e:  # noqa: BLE001 - any non-truncation failure is fine
        assert "truncated at" not in str(e), f"bare filename wrongly rejected: {e}"
