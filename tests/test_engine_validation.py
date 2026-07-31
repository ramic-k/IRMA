"""Parse-time validation hardening for irma.core.engine (run_leapr).

Each case below was a confirmed silent-misconfiguration or
malformed-output foot-gun in the QA campaign:

  F11 (+ Codex C6) — Card 6e partial spectrum whose Z/A matches no Card 6d
                     atom, and duplicate Card 6e / Card 6d (Z, A), must
                     fail loudly with card context instead of being
                     silently dropped/overwritten.
  F12             — Card 4 za must be coerced to an integer ENDF ZA; a
                     fractional za is a malformed deck.
  F13             — Card 6b Bragg-edge grouping fields (bins_per_decade,
                     threshold) must reject negative values rather than
                     silently disabling/defaulting.
  C5              — Card 4 smin must be finite and >= 0.
  F16 (engine)    — mode-1/2 ncpu>1 on Windows (fork-only parallelism) must
                     raise a DeckError advising ncpu=1, not a raw ValueError.
  F19             — Card 6f ncpu greatly exceeding the core count is clamped
                     to os.cpu_count() with a printed warning.
  F30             — `python -m irma.core.engine` delegates to irma.cli.main,
                     so a bad deck gets the friendly DeckError handler.

The deck construction mirrors tests/test_deck_errors.py.
"""
import os
import subprocess
import sys
import tempfile

import pytest

from irma.core.engine import run_leapr, DeckError


# A minimal well-formed classic deck (Card 4 carries only `mat za`).
GOOD = """20 /
'qa3 engine test'/
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

# An iel=10 inelastic_mode=0 head whose Card 6b (`1 1 0 0/`) and Card 6e
# atom-matching are the validation surfaces for F11/F13/C6.
_GEN_HEAD = """20 /
'qa3 gen validation deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 0/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 4/
0.0 0.0 0.0  0.0 0.0 0.5  0.333333 0.666667 0.0  0.666667 0.333333 0.5/
"""

# Classic Cards 7-13 tail appended after the iel=10 mode-0 crystal cards so a
# positive-path deck runs to completion (mirrors the GOOD tail).
_GEN_TAIL = """3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
300/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
/
"""

# An iel=10 inelastic_mode=2 head reaching Card 6f (ncpu) — the phonopy path
# is bogus on purpose so the run fails at the mesh load AFTER Card 6f parses.
_MODE2_6F_HEAD = """20 /
'qa3 card 6f validation deck'/
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


def _expect(deck_text, *fragments):
    with pytest.raises(DeckError) as exc:
        _run(deck_text)
    msg = str(exc.value)
    for frag in fragments:
        assert frag in msg, f"expected {frag!r} in error message:\n  {msg}"
    return msg


# ---------- F12: Card 4 za integer coercion ----------

def test_card4_fractional_za_rejected():
    """A non-integral ENDF ZA must be rejected with Card 4 context, not
    emitted verbatim as a malformed ENDF float."""
    deck = GOOD.replace("1 1./", "1 6012.5/")
    _expect(deck, "Card 4", "za must be an integer", "6012.5")


def test_card4_integral_float_za_accepted():
    """An exactly-integral za (NJOY list-directed style) still parses."""
    deck = GOOD.replace("1 1./", "1 1.0/")
    _run(deck)   # must not raise


# ---------- C5: Card 4 smin finite/nonnegative ----------

def test_card4_negative_smin_rejected():
    deck = GOOD.replace("1 1./", "1 1. 0 0 -1.0/")
    _expect(deck, "Card 4", "smin", ">= 0")


def test_card4_nonfinite_smin_rejected():
    deck = GOOD.replace("1 1./", "1 1. 0 0 1e999/")
    # 1e999 is rejected by the numeric reader before semantic validation,
    # but it must still surface as a Card 4 deck error, never an inf cutoff.
    _expect(deck, "Card 4")


def test_card4_default_smin_accepted():
    """The implicit default smin (1e-75) remains valid."""
    _run(GOOD)   # sanity: the template has no explicit smin


# ---------- F13: Card 6b grouping field signs ----------

def test_card6b_negative_bins_per_decade_rejected():
    """A negative bins_per_decade must not silently disable grouping."""
    deck = _GEN_HEAD.replace("1 1 0 0/", "1 1 0 0 -5 2.0/")
    _expect(deck, "Card 6b", "bins_per_decade", ">= 0")


def test_card6b_negative_threshold_rejected():
    """A negative threshold used to be silently replaced by the 1 eV default
    (and the on-screen message even reported 'above 1 eV')."""
    deck = _GEN_HEAD.replace("1 1 0 0/", "1 1 0 0 10 -2.0/")
    _expect(deck, "Card 6b", "threshold", ">= 0")


def test_card6b_grouping_fields_off_accepted():
    """Zero bins (grouping off) and a positive threshold remain valid."""
    deck = _GEN_HEAD.replace("1 1 0 0/", "1 1 0 0 0 2.0/") + _GEN_TAIL
    _run(deck)   # must not raise


def test_card6b_positive_grouping_fields_accepted():
    deck = _GEN_HEAD.replace("1 1 0 0/", "1 1 0 0 10 2.0/") + _GEN_TAIL
    _run(deck)   # must not raise


# ---------- F11 + Codex C6: Card 6e atom-matching cardinality ----------

def test_card6e_unmatched_spectrum_rejected():
    """A Card 6e spectrum whose Z/A matches no Card 6d atom type used to be
    parsed, validated, then silently never used (falling back to principal
    DW). It must now fail loudly (F11)."""
    head = _GEN_HEAD.replace(
        "1 1 0 0/", "1 1 1 0/"
    ).replace("6 12 11.9 6.646 0.001 4/", "6 12 11.9 6.646 0.001 1/").replace(
        "0.0 0.0 0.0  0.0 0.0 0.5  0.333333 0.666667 0.0  0.666667 0.333333 0.5/",
        "0.0 0.0 0.0/",
    )
    # Spectrum for Z=8 A=16 (oxygen), but the only Card 6d atom is Z=6 A=12.
    deck = head + "8 16 0.005 4/\n0.0 0.2 0.5 0.3/\n"
    _expect(deck, "Card 6e", "does not match any Card 6d atom type")


def test_card6e_matched_spectrum_accepted():
    """A Card 6e spectrum that DOES match its Card 6d atom still parses."""
    head = _GEN_HEAD.replace(
        "1 1 0 0/", "1 1 1 0/"
    ).replace("6 12 11.9 6.646 0.001 4/", "6 12 11.9 6.646 0.001 1/").replace(
        "0.0 0.0 0.0  0.0 0.0 0.5  0.333333 0.666667 0.0  0.666667 0.333333 0.5/",
        "0.0 0.0 0.0/",
    )
    deck = head + "6 12 0.005 4/\n0.0 0.2 0.5 0.3/\n" + _GEN_TAIL
    _run(deck)   # must not raise


def test_card6e_duplicate_spectrum_za_rejected():
    """Two Card 6e spectra for the same (Z, A) silently overwrote the
    atom's spectrum_idx; reject the ambiguity (Codex C6)."""
    head = _GEN_HEAD.replace(
        "1 1 0 0/", "1 1 2 0/"
    ).replace("6 12 11.9 6.646 0.001 4/", "6 12 11.9 6.646 0.001 1/").replace(
        "0.0 0.0 0.0  0.0 0.0 0.5  0.333333 0.666667 0.0  0.666667 0.333333 0.5/",
        "0.0 0.0 0.0/",
    )
    deck = (head
            + "6 12 0.005 4/\n0.0 0.2 0.5 0.3/\n"
            + "6 12 0.005 4/\n0.0 0.2 0.5 0.3/\n")
    _expect(deck, "Card 6e", "duplicate", "Z=6", "A=12")


def test_card6d_duplicate_za_with_spectrum_rejected():
    """Two Card 6d atom types sharing (Z, A) make spectrum matching
    ambiguous ONLY when a Card 6e spectrum targets that (Z, A); reject
    exactly that combination (Codex C6)."""
    head = """20 /
'qa3 dup 6d deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 2 1 0/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
6 12 11.9 6.646 0.001 1/
0.5 0.5 0.5/
6 12 0.005 4/
0.0 0.2 0.5 0.3/
"""
    _expect(head, "Card 6e", "duplicate Card 6d", "Z=6", "A=12")


def test_card6d_duplicate_za_without_spectrum_accepted():
    """Two Card 6d atom types sharing (Z, A) at distinct sites is a valid
    configuration (resolved by position downstream) when no Card 6e
    spectrum needs an unambiguous (Z, A) match — it must NOT be rejected
    at the Card 6e matching stage (reviewer-flagged backward compat)."""
    head = """20 /
'qa3 dup 6d deck'/
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


# ---------- F16 (engine-side): Windows fork-only parallelism ----------

def test_mode2_ncpu_gt1_on_windows_accepted(monkeypatch):
    """The spawn + shared-memory pool works on every platform, so ncpu>1 on
    Windows must pass the Card 6f validation (the old fork-only rejection is
    gone) and proceed to the (bogus) phonopy mesh load."""
    monkeypatch.setattr(sys, "platform", "win32")
    deck = _MODE2_6F_HEAD + "8 8 8 4 0 /\n100 100 /\n"
    with pytest.raises(RuntimeError, match="phonopy mesh"):
        _run(deck)


def test_mode2_ncpu_eq1_on_windows_allowed(monkeypatch):
    """ncpu=1 is the platform-independent path: it must pass the Windows
    guard and proceed to the (bogus) phonopy mesh load instead."""
    monkeypatch.setattr(sys, "platform", "win32")
    deck = _MODE2_6F_HEAD + "8 8 8 1 0 /\n100 100 /\n"
    with pytest.raises(RuntimeError, match="phonopy mesh"):
        _run(deck)


# ---------- F19: Card 6f ncpu clamp to cpu_count ----------

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


def test_card6f_ncpu_within_cpu_count_not_clamped(monkeypatch, capsys):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    deck = _MODE2_6F_HEAD + "8 8 8 2 0 /\n100 100 /\n"
    with pytest.raises(RuntimeError, match="phonopy mesh"):
        _run(deck)
    out = capsys.readouterr().out
    assert "clamping" not in out


# ---------- F30: engine __main__ delegates to cli.main ----------

def test_engine_main_delegates_friendly_deckerror():
    """`python -m irma.core.engine` on a malformed deck must surface the
    friendly cli DeckError handler (exit 2, 'Input deck error'), not a raw
    traceback."""
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    # Card 5 with an invalid iel raises DeckError inside run_leapr.
    with open(inp, "w") as f:
        f.write(GOOD.replace("1.0 20.0 1 0 0/", "1.0 20.0 1 7 0/"))
    env = dict(os.environ, MPLCONFIGDIR="/tmp", PYTHONPYCACHEPREFIX="/tmp/qa3pyc")
    proc = subprocess.run(
        [sys.executable, "-m", "irma.core.engine", inp,
         os.path.join(d, "out.endf")],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 2, proc.stderr
    assert "Input deck error" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_engine_main_no_args_prints_cli_usage():
    """No-args invocation delegates to cli.main, which prints its usage and
    exits 0 (single-sourced help, not the old bare engine usage string)."""
    env = dict(os.environ, MPLCONFIGDIR="/tmp", PYTHONPYCACHEPREFIX="/tmp/qa3pyc")
    proc = subprocess.run(
        [sys.executable, "-m", "irma.core.engine"],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, proc.stderr
    # single-sourced cli help (with the new subcommands), not a bare engine string
    assert "Usage:" in proc.stdout and "irma spectra" in proc.stdout
