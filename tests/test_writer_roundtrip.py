"""Writer round-trip: run a ~1s deck, re-parse the tape, pin the structure.

The deck-level physics is covered by the native LEAPR/NJOY validation; here we
pin the STRUCTURAL contract of ``write_endf_output`` cheaply — the MF7/MT4
B-array (whose layout follows NJOY's endout: B1=npr*spr, B2=beta_max, B3=awr,
B4=0.0253*beta_max, B6=npr, and the 6 secondary-scatterer entries when nss>0),
the LAT/LASYM/LLN flags, the temperature blocks, the grids, and the MF7/MT2
incoherent-elastic (LTHR=2) section that iel=0 decks fall back to when twt=0.
"""
import os
import tempfile

import numpy as np
import pytest

from irma.core.engine import run_leapr

# Tiny 2-temperature deck: awr=1, spr=20, npr=1, iel=0, twt=0 (-> incoherent
# elastic), alpha=[0.05,1,8], beta=[0,0.6,2,6], 300 K + reused 400 K.
_DECK_PRINCIPAL = """20 /
'roundtrip principal-only deck'/
2 1 4/
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
-400/
' roundtrip test deck '/
/
"""

# Same deck with an analytic (b7=1, free-gas) secondary scatterer:
# nss=1, b7=1, aws=16.0, sps=4.0, mss=1.
_DECK_SECONDARY = _DECK_PRINCIPAL.replace(
    "1.0 20.0 1 0 0/\n0/",
    "1.0 20.0 1 0 0/\n1 1. 16.0 4.0 1/")


def _run_and_parse(deck_text, d):
    """Run the deck in directory d: (tape path, parsed dict, text lines)."""
    from endf_parserpy import EndfParserPy
    inp, out = os.path.join(d, "deck.input"), os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, out)
    p = EndfParserPy(ignore_number_mismatch=True, ignore_zero_mismatch=True,
                     ignore_varspec_mismatch=True)
    with open(out) as f:
        lines = f.read().splitlines()
    return out, p.parsefile(out), lines


@pytest.fixture(scope="module")
def principal_run(tmp_path_factory):
    return _run_and_parse(_DECK_PRINCIPAL, tmp_path_factory.mktemp("principal"))


@pytest.fixture(scope="module")
def principal_tape(principal_run):
    return principal_run[1]


@pytest.fixture(scope="module")
def secondary_tape(tmp_path_factory):
    return _run_and_parse(_DECK_SECONDARY, tmp_path_factory.mktemp("secondary"))[1]


def test_mt4_b_array_principal(principal_tape):
    b = principal_tape[7][4]["B"]
    assert b[1] == pytest.approx(1 * 20.0, rel=5e-7)        # npr*spr
    assert b[2] == pytest.approx(6.0, rel=5e-7)             # beta_max
    assert b[3] == pytest.approx(1.0, rel=5e-7)             # awr
    assert b[4] == pytest.approx(0.0253 * 6.0, rel=5e-7)    # E_max (eV)
    assert b[5] == 0.0
    assert b[6] == pytest.approx(1.0)                       # npr
    assert len(b) == 6                                      # NI=6, no secondary


def test_mt4_flags_and_grids(principal_tape):
    mt4 = principal_tape[7][4]
    assert (mt4["LAT"], mt4["LASYM"], mt4["LLN"], mt4["NS"]) == (1, 0, 0, 0)
    assert mt4["NB"] == 4 and mt4["NP"] == 3
    beta = np.array([mt4["beta"][k] for k in sorted(mt4["beta"])])
    np.testing.assert_allclose(beta, [0.0, 0.6, 2.0, 6.0], rtol=5e-7)
    alpha = np.array(mt4["S_table"][1]["alpha"])
    np.testing.assert_allclose(alpha, [0.05, 1.0, 8.0], rtol=5e-7)


def test_mt4_temperature_blocks(principal_tape):
    mt4 = principal_tape[7][4]
    assert mt4["T0"] == pytest.approx(300.0)
    extra = mt4["T"]
    assert [extra[k] for k in sorted(extra)] == [pytest.approx(400.0)]
    # teff table covers both temperatures
    teff = mt4["teff0_table"]
    assert len(teff["Tint"]) == 2
    assert all(t_eff >= t for t, t_eff in zip(teff["Tint"], teff["Teff0"]))


def test_mt2_incoherent_elastic_fallback(principal_tape):
    """iel=0 with twt=0 must emit incoherent elastic (LTHR=2)."""
    mt2 = principal_tape[7][2]
    assert mt2["LTHR"] == 2
    # SB = bound xs * npr = spr*((1+awr)/awr)^2 = 20*(2/1)^2 = 80 b
    assert mt2["SB"] == pytest.approx(80.0, rel=5e-7)
    wvals = mt2["Wp"]
    assert len(wvals) == 2 and all(v > 0 for v in wvals)


def test_mt4_b_array_secondary(secondary_tape):
    """nss=1, b7=1 (free-gas O-like secondary) fills B7-B12."""
    b = secondary_tape[7][4]["B"]
    assert len(b) == 12
    assert b[7] == pytest.approx(1.0)                       # b7 (free gas)
    assert b[8] == pytest.approx(1 * 4.0, rel=5e-7)         # mss*sps
    assert b[9] == pytest.approx(16.0, rel=5e-7)            # aws
    assert b[10] == 0.0 and b[11] == 0.0
    assert b[12] == pytest.approx(1.0)                      # mss
    assert secondary_tape[7][4]["NS"] == 1


def test_principal_law_unchanged_by_analytic_secondary(principal_tape,
                                                       secondary_tape):
    """b7>0 secondaries are analytic (B-array only): same S(alpha,beta)."""
    s_p = principal_tape[7][4]["S_table"]
    s_s = secondary_tape[7][4]["S_table"]
    for j in sorted(s_p):
        np.testing.assert_allclose(s_p[j]["S"], s_s[j]["S"], rtol=0, atol=0)


def _read_tape_lines(deck_text):
    d = tempfile.mkdtemp()
    inp, out = os.path.join(d, "deck.input"), os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, out)
    with open(out) as f:
        return f.read().splitlines()


def _audit_directory(lines):
    """Return list of (MFx, MTx, NCx, actual) from the MF1/MT451 directory."""
    sections = {}
    for ln in lines:
        if len(ln) < 75:
            continue
        try:
            mf, mt = int(ln[70:72]), int(ln[72:75])
        except ValueError:
            continue
        if mf and mt:
            sections[(mf, mt)] = sections.get((mf, mt), 0) + 1
    out = []
    for ln in lines:
        if (len(ln) >= 75 and ln[:22].strip() == ""
                and ln[70:72] == " 1" and ln[72:75] == "451"):
            try:
                mfx, mtx, ncx = int(ln[22:33]), int(ln[33:44]), int(ln[44:55])
                int(ln[55:66])
            except ValueError:
                continue
            out.append((mfx, mtx, ncx, sections.get((mfx, mtx), 0)))
    return out


def test_mf1_directory_counts_are_exact(principal_run):
    """Every MF1/MT451 directory NCx must equal the section's actual record
    count (the NJOY closed-form estimates were wrong for this writer's line
    wrapping and for the iel=10 LTHR=2/3 and grouped-elastic branches)."""
    lines = principal_run[2]
    entries = _audit_directory(lines)
    assert {(m, t) for m, t, _, _ in entries} == {(1, 451), (7, 2), (7, 4)}
    for mfx, mtx, ncx, actual in entries:
        assert ncx == actual, f"MF{mfx}/MT{mtx}: NCx={ncx} actual={actual}"


def test_mf1_nwd_counts_all_written_text_records(principal_run):
    """The 5 structured header text records are always emitted (blank-padded
    for short decks): NWD must never be below 5, and MF1's record count must
    be 4 CONTs + NWD + NXC."""
    lines = principal_run[2]                    # deck has 1 comment card
    i0 = next(i for i, ln in enumerate(lines)
              if len(ln) >= 75 and ln[70:72] == " 1" and ln[72:75] == "451")
    hdr4 = lines[i0 + 3]
    nwd, nxc = int(hdr4[44:55]), int(hdr4[55:66])
    assert nwd == 5                              # 1 comment -> 5 header records
    entries = _audit_directory(lines)
    nc_451 = next(nc for mf, mt, nc, _ in entries if (mf, mt) == (1, 451))
    assert nc_451 == 4 + nwd + nxc


# _DECK_PRINCIPAL with ENDF-conventional MF1 header comment cards (the shape
# NJOY decks ship): card 1 begins with the column-1 blank that is part of the
# 11-char ZSYMAM field; card 2 carries REF plus the DIST/REV dates and the
# 8-char ENDATE in their fixed columns.
_DECK_MF1_HEADER = _DECK_PRINCIPAL.replace(
    "' roundtrip test deck '/",
    "' Graphite  LEIP LAB   EVAL-SEP17 A.I. Hawari, Y. Zhu, J.L. Wormald'/\n"
    "' NDS 118, 1 (2014)    DIST-FEB18 REV1-DEC17            20170917   '/")


def test_mf1_header_fields_land_in_spec_columns():
    """The header comment cards' leading blank is column 1 of ZSYMAM; the
    writer must preserve it (rstrip, not strip), or ZSYMAM/ALAB/EDATE/AUTH
    and the card-2 REF/DDATE/RDATE/ENDATE all shift one column left and lose
    a character at each field boundary."""
    lines = _read_tape_lines(_DECK_MF1_HEADER)
    i0 = next(i for i, ln in enumerate(lines)
              if len(ln) >= 75 and ln[70:72] == " 1" and ln[72:75] == "451")
    rec1, rec2 = lines[i0 + 4][:66], lines[i0 + 5][:66]
    # Record 1: ZSYMAM(11) + ALAB(11) + EDATE(10) + AUTH(33 from col 34)
    assert rec1[:11] == " Graphite  "                        # ZSYMAM
    assert rec1[11:22] == "LEIP LAB   "                      # ALAB
    assert rec1[22:32] == "EVAL-SEP17"                       # EDATE
    assert rec1[33:66] == "A.I. Hawari, Y. Zhu, J.L. Wormald"  # AUTH
    # Record 2: blank + REF(21) + DDATE(10) + blank + RDATE(10) + pad(12)
    # + ENDATE(8) + pad(3)
    assert rec2[1:22] == "NDS 118, 1 (2014)    "             # REF
    assert rec2[22:32] == "DIST-FEB18"                       # DDATE
    assert rec2[33:43] == "REV1-DEC17"                       # RDATE
    assert rec2[55:63] == "20170917"                         # ENDATE


def test_tape_contains_no_carriage_returns(principal_run):
    """Tapes must be LF-only on every platform: a CRLF tape written on
    Windows would not be byte-identical to the NJOY references."""
    with open(principal_run[0], "rb") as f:
        data = f.read()
    assert data.count(b"\n") > 50               # sanity: a real multi-line tape
    assert b"\r" not in data
