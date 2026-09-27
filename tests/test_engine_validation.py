"""Parse-time validation in run_leapr: Card 4 za and smin, Card 6b grouping
fields, Card 12e/6d atom matching, the Card 6f ncpu clamp, and the engine
__main__ delegating to irma.cli. The decks mirror tests/test_deck_errors.py.
"""
import os
import subprocess
import sys
import tempfile

import pytest

from irma.core.engine import run_leapr, DeckError


# A minimal well-formed classic deck (Card 4 carries only `mat za`).
GOOD = """20 /
'engine validation deck'/
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

# An iel=10 inelastic_mode=0 head whose Card 6b (`1 1 0 0/`) fields are the
# validation surface below.
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

# A two-species iel=10 mode-0 head (C principal, O) up to Card 9; the
# temperature blocks with their Card 12e spectra are added by _card12e_deck.
_TWO_TYPE_HEAD = """20 /
'card 12e deck'/
{ntempr} 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
2 2 {nspec} 0{grouping}/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 2/
0.0 0.0 0.0  0.0 0.0 0.5/
8 16 15.86 5.803 0.0 2/
0.333333 0.666667 0.0  0.666667 0.333333 0.5/
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
"""
_PRINCIPAL_SPECTRUM = "0.005 6/\n0.0 0.20 0.45 0.55 0.30 0.0/\n"
_OXYGEN = "8 16 0.005 6/\n0.0 0.10 0.30 0.60 0.40 0.0/\n"


def _card12e_deck(spectra, grouping="", later=()):
    """The two-species deck: a 300 K block with the given Card 12e spectra,
    then one block per (temperature, spectra) in ``later`` (a negative
    temperature reuses the previous block and takes no cards)."""
    text = _TWO_TYPE_HEAD.format(ntempr=1 + len(later), nspec=len(spectra),
                                 grouping=grouping)
    for temp, block in [(300, spectra), *later]:
        text += f"{temp}/\n"
        if temp > 0:
            text += _PRINCIPAL_SPECTRUM + "".join(block) + "0. 0. 1./\n0/\n"
    return text + "/\n"


# An iel=10 inelastic_mode=2 head reaching Card 6f (ncpu) — the phonopy path
# is bogus on purpose so the run fails at the mesh load AFTER Card 6f parses.
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


def _run(deck_text):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, os.path.join(d, "out.endf"))


def _tape(tmp_path, name, deck_text):
    """Run a deck and return its tape text."""
    inp, out = tmp_path / f"{name}.input", tmp_path / f"{name}.endf"
    inp.write_text(deck_text)
    run_leapr(str(inp), str(out))
    return out.read_text()


def _mf7(tape, mt):
    return [line[:66] for line in tape.splitlines()
            if line[70:72].strip() == "7" and line[72:75].strip() == str(mt)]


def _expect(deck_text, *fragments):
    with pytest.raises(DeckError) as exc:
        _run(deck_text)
    msg = str(exc.value)
    for frag in fragments:
        assert frag in msg, f"expected {frag!r} in error message:\n  {msg}"
    return msg


# ---------- Card 4 za integer coercion ----------

def test_card4_fractional_za_rejected():
    """A non-integral ENDF ZA must be rejected with Card 4 context, not
    emitted verbatim as a malformed ENDF float."""
    deck = GOOD.replace("1 1./", "1 6012.5/")
    _expect(deck, "Card 4", "za must be an integer", "6012.5")


def test_card4_integral_float_za_accepted():
    """An exactly-integral za (NJOY list-directed style) still parses."""
    deck = GOOD.replace("1 1./", "1 1.0/")
    _run(deck)   # must not raise


# ---------- Card 4 smin finite/nonnegative ----------

def test_card4_negative_smin_rejected():
    deck = GOOD.replace("1 1./", "1 1. 0 0 -1.0/")
    _expect(deck, "Card 4", "smin", ">= 0")


def test_card4_nonfinite_smin_rejected():
    deck = GOOD.replace("1 1./", "1 1. 0 0 1e999/")
    # 1e999 is rejected by the numeric reader before semantic validation,
    # but it must still surface as a Card 4 deck error, never an inf cutoff.
    _expect(deck, "Card 4")


# ---------- Card 6b grouping field signs ----------

def test_card6b_negative_bins_per_decade_rejected():
    """A negative bins_per_decade must not silently disable grouping."""
    deck = _GEN_HEAD.replace("1 1 0 0/", "1 1 0 0 -5 2.0/")
    _expect(deck, "Card 6b", "bins_per_decade", ">= 0")


def test_card6b_negative_threshold_rejected():
    """A negative threshold is rejected, not replaced by the 1 eV default."""
    deck = _GEN_HEAD.replace("1 1 0 0/", "1 1 0 0 10 -2.0/")
    _expect(deck, "Card 6b", "threshold", ">= 0")


# ---------- Card 12e partial spectra ----------

def test_card12e_unmatched_spectrum_rejected():
    """A spectrum whose Z/A matches no Card 6d atom type would never be used
    (the principal's DW would stand in); it must fail loudly."""
    deck = _card12e_deck(["26 56 0.005 4/\n0.0 0.2 0.5 0.3/\n"])
    _expect(deck, "Card 12e", "does not match any Card 6d atom type")


def test_card12e_matched_spectrum_accepted():
    """A spectrum for a non-principal Card 6d atom runs, with positive Card 6b
    grouping fields (bins_per_decade=10, threshold 2 eV)."""
    _run(_card12e_deck([_OXYGEN], grouping=" 10 2.0"))


@pytest.mark.parametrize("spectra, fragments", [
    ([_OXYGEN, _OXYGEN], ("duplicate Card 12e", "Z=8", "A=16")),
    (["6 12 0.005 4/\n0.0 0.2 0.5 0.3/\n"], ("principal scatterer", "Cards 11-12")),
], ids=["duplicate", "principal"])
def test_card12e_duplicate_or_principal_spectrum_rejected(spectra, fragments):
    """Two spectra for one species are ambiguous, and the principal's
    spectrum is Cards 11-12 of the same block."""
    _expect(_card12e_deck(spectra), "Card 12e", *fragments)


def test_card12e_species_change_between_blocks_rejected():
    """Every temperature block gives spectra for the same species."""
    nitrogen = "7 14 0.005 4/\n0.0 0.2 0.5 0.3/\n"
    deck = _card12e_deck([_OXYGEN], later=[(600, [nitrogen])]).replace(
        "2 2 1 0/", "2 3 1 0/").replace(
        "0.333333 0.666667 0.0  0.666667 0.333333 0.5/\n",
        "0.333333 0.666667 0.0  0.666667 0.333333 0.5/\n"
        "7 14 13.88 9.36 0.5 1/\n0.5 0.5 0.25/\n")
    _expect(deck, "Card 12e", "same species as the first block")


def test_old_card6e_position_rejected():
    """Spectra left after Card 6d, where Card 6e used to be, are refused with
    a message that says where they go now."""
    deck = _card12e_deck([_OXYGEN])
    deck = deck.replace(_OXYGEN, "", 1).replace("3 4 1/\n", _OXYGEN + "3 4 1/\n", 1)
    _expect(deck, "Card 7", "partial spectrum after Card 6d", "Card 12e")


def test_card12e_spectra_follow_their_temperature(tmp_path, monkeypatch):
    """Each temperature's Card 12e spectrum sets that temperature's Debye-
    Waller lambda (start() of that spectrum at that T). A negative
    temperature equals repeating the block, and the principal's MT4 does
    not depend on the other species' spectra."""
    import numpy as np
    import irma.core.driver as driver
    from irma.core.constants import BK
    from irma.core.kernels import start

    seen = []
    compute = driver._compute_per_species_msd

    def record(crystal_info, tempr_arr, ntempr, dwpix):
        compute(crystal_info, tempr_arr, ntempr, dwpix)
        seen.append(np.array(crystal_info['atom_types'][1]['dwpix']))
    monkeypatch.setattr(driver, "_compute_per_species_msd", record)

    other = "8 16 0.004 7/\n0.0 0.05 0.20 0.50 0.60 0.30 0.0/\n"
    reuse = _tape(tmp_path, "reuse", _card12e_deck([_OXYGEN], later=[(-600, None)]))
    repeat = _tape(tmp_path, "repeat", _card12e_deck([_OXYGEN], later=[(600, [_OXYGEN])]))
    own = _tape(tmp_path, "own", _card12e_deck([_OXYGEN], later=[(600, [other])]))

    def lam(delta, rho, temp):
        return start(np.array(rho), len(rho), delta, BK * temp, 1.0)[1]
    oxygen = (0.005, [0.0, 0.10, 0.30, 0.60, 0.40, 0.0])
    stiff = (0.004, [0.0, 0.05, 0.20, 0.50, 0.60, 0.30, 0.0])
    np.testing.assert_allclose(seen[0], [lam(*oxygen, 300), lam(*oxygen, 600)], rtol=1e-12)
    np.testing.assert_allclose(seen[2], [lam(*oxygen, 300), lam(*stiff, 600)], rtol=1e-12)
    assert _mf7(repeat, 2) == _mf7(reuse, 2)
    assert _mf7(own, 2) != _mf7(reuse, 2)     # the writer uses the new lambda
    assert _mf7(own, 4) == _mf7(reuse, 4)


def test_card6d_duplicate_za_with_spectrum_rejected():
    """Two Card 6d atom types sharing a non-principal (Z, A) make spectrum
    matching ambiguous ONLY when a Card 12e spectrum targets that (Z, A);
    reject exactly that combination. (Rows of the principal are merged
    into one group before the matching.)"""
    deck = _card12e_deck([_OXYGEN]).replace("2 2 1 0/", "2 3 1 0/").replace(
        "0.333333 0.666667 0.0  0.666667 0.333333 0.5/\n",
        "0.333333 0.666667 0.0  0.666667 0.333333 0.5/\n"
        "8 16 15.86 5.803 0.0 1/\n0.5 0.5 0.25/\n")
    _expect(deck, "Card 12e", "duplicate Card 6d", "Z=8", "A=16")


def test_card6d_duplicate_za_without_spectrum_accepted():
    """Two Card 6d atom types sharing (Z, A) at distinct sites is a valid
    configuration (resolved by position downstream) when no Card 12e
    spectrum needs an unambiguous (Z, A) match — it must NOT be rejected
    at the Card 12e matching stage."""
    head = """20 /
'dup 6d deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 2 0 0/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
6 12 11.9 6.646 0.001 1/
0.5 0.5 0.5/
"""
    try:
        _run(head)
    except DeckError as exc:
        assert "duplicate Card 6d" not in str(exc), (
            "nspec=0 duplicate Card 6d (Z, A) must not be rejected by the "
            "spectrum-matching guard")


# ---------- Card 6f ncpu clamp to cpu_count ----------

def test_card6f_ncpu_over_cpu_count_clamped_with_warning(monkeypatch, capsys):
    """A wildly large ncpu is clamped to os.cpu_count() with a printed
    warning, before the run forks; the deck then fails at the bogus phonopy
    mesh load, proving the clamp fired during parsing."""
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    deck = _MODE2_6F_HEAD + "8 8 8 99999 0 /\n100 100 /\n"
    with pytest.raises(RuntimeError, match="phonopy mesh"):
        _run(deck)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "ncpu=99999" in out
    assert "clamping to 4" in out


# ---------- engine __main__ delegates to cli.main ----------

def test_engine_main_delegates_friendly_deckerror():
    """`python -m irma.core.engine` on a malformed deck must surface the
    friendly cli DeckError handler (exit 2, 'Input deck error'), not a raw
    traceback."""
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    # Card 5 with an invalid iel raises DeckError inside run_leapr.
    with open(inp, "w") as f:
        f.write(GOOD.replace("1.0 20.0 1 0 0/", "1.0 20.0 1 7 0/"))
    env = dict(os.environ, MPLCONFIGDIR="/tmp", PYTHONPYCACHEPREFIX="/tmp/irma_test_pyc")
    proc = subprocess.run(
        [sys.executable, "-m", "irma.core.engine", inp,
         os.path.join(d, "out.endf")],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Input deck error" in proc.stderr
    assert "Traceback" not in proc.stderr
