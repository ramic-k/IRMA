"""CEF incoherent-elastic (LTHR=2) writer round-trip: the SUCCESS paths of
_build_cef_incoherent and both dispatch branches that select it.

The Eq-25 single-atom incoherent-dominant branch (sigma_inc >= sigma_coh) and
the Eq-26 polyatomic non-DC redistribution branch (paper: K. Ramic et al.,
NIM-A 1027 (2022) 166227) were only ever exercised through their ValueError
guards (test_cef_guards.py); no test pinned the SB / W'(T) they actually emit.
Here each branch is driven through write_endf_output (iel=10) to a REAL tape
and read back with irma.spectra.elastic.from_endf_mf7mt2 — also the first
writer-produced input that reader's LTHR=2 dispatch has ever seen. Expected
values are computed from the paper formulas, not from the code's output:

  Eq 25: SB = sigma_inc x (sigma_coh + sigma_inc)/sigma_inc
  Eq 26: SB = sigma_inc_p x [1 + f_DC/(1-f_DC) x sigma_inc_DC/sigma_inc_p]
  W'(T) = dwpix(T)/(awr x T x kB)   [1/eV on tape; reader returns 1/meV]
"""
import numpy as np
import pytest

from irma.core.constants import BK
from irma.core.endf_writer import _build_cef_incoherent, write_endf_output
from irma.spectra.elastic import (from_endf_mf7mt2, _Cursor, _mf7mt2_lines,
                                  _mf7mt4_npr)

_TEMPR = [296.0, 500.0]
_DWPIX = [0.8, 1.1]   # raw DW integrals; the writer stores W' = dwpix/(awr*T*kB)

# Eq-25 case: single atom with sigma_inc >= sigma_coh (hydrogen-like).
_H = {'Z': 1, 'A': 1, 'awr': 0.99917, 'b_coh': -3.7406, 'sigma_inc': 80.27,
      'sigma_coh': 1.7568, 'fraction': 1.0, 'dwpix': list(_DWPIX)}

# Eq-26 case: principal (idx 0) is coherent-dominant but NOT the DC atom, so
# the dispatch must still pick the incoherent builder with redistribution.
_O = {'Z': 8, 'A': 16, 'awr': 15.86, 'b_coh': 5.803, 'sigma_inc': 0.35,
      'sigma_coh': 4.232, 'fraction': 0.4, 'dwpix': list(_DWPIX)}
_BE = {'Z': 4, 'A': 9, 'awr': 8.93, 'b_coh': 7.79, 'sigma_inc': 0.22,
       'sigma_coh': 7.63, 'fraction': 0.6, 'dwpix': list(_DWPIX)}


def _ci(atom_types, principal=0, dc=None):
    return {
        'elastic_mode': 1,   # CEF
        'atom_types': atom_types,
        'nat': len(atom_types),
        'principal_atom_idx': principal,
        'dc_atom_idx': dc,
        'species_corr': None,
        'F_species_per_temp': None,
        'bragg_dir_terms': None,
    }


def _ci_mef(atom_types, principal=0):
    """MEF (elastic_mode=2 / LTHR=3) crystal_info, same shape as _ci()."""
    ci = _ci(atom_types, principal=principal)
    ci['elastic_mode'] = 2
    return ci


# A minimal two-edge Bragg set. The CEF/SEF incoherent builder ignores
# bragg/nedge entirely, but the MEF builder writes the coherent edges too and
# rejects an empty list, so the MEF fixtures pass these.
_BRAGG = [(2.5e-3, 1.0), (5.0e-3, 0.5)]


def _write_iel10_tape(path, za, awr, crystal_info, npr=1, bragg=None):
    """Drive write_endf_output with iel=10 and a minimal synthetic MF7/MT4
    payload (2 alpha x 2 beta x 2 T); only the MT2 section is under test.
    The incoherent builder ignores bragg/nedge, so an empty edge list is fine
    there; the MEF builder needs real edges (see _BRAGG)."""
    ssm = np.full((2, 2, 2), 0.1)
    edges = list(bragg or [])
    write_endf_output(
        str(path), 1, za, awr, 5.5, npr, 10,      # mat za awr spr npr iel
        0, 0.0, 0.0, 0.0, 0, 2, 2, 1,             # nss b7 aws sps mss nalpha nbeta lat
        np.array([0.1, 1.0]), np.array([0.0, 1.0]), ssm, None,
        np.array(_TEMPR), 2, np.array(_DWPIX), np.array(_DWPIX),
        np.array([320.0, 520.0]), np.array([320.0, 520.0]),
        edges, len(edges), 0, 0, 1.0e-6,
        comments=None, crystal_info=crystal_info)


@pytest.fixture(scope="module")
def eq25_tape(tmp_path_factory):
    p = tmp_path_factory.mktemp("cef_eq25") / "eq25.endf"
    _write_iel10_tape(p, 1001.0, _H['awr'], _ci([_H]))
    return p


@pytest.fixture(scope="module")
def eq26_tape(tmp_path_factory):
    p = tmp_path_factory.mktemp("cef_eq26") / "eq26.endf"
    _write_iel10_tape(p, 8016.0, _O['awr'], _ci([_O, _BE], dc=1))
    return p


def test_eq25_emits_lthr2(eq25_tape):
    m = from_endf_mf7mt2(str(eq25_tape), T_K=296.0)
    assert "LTHR=2" in m.label
    assert m.has_incoherent and not m.has_coherent


def test_eq26_emits_lthr2(eq26_tape):
    # sigma_coh > sigma_inc for the principal: only the non-DC dispatch may
    # send this to the incoherent builder (single-atom logic would emit LTHR=1).
    m = from_endf_mf7mt2(str(eq26_tape), T_K=296.0)
    assert "LTHR=2" in m.label
    assert m.has_incoherent and not m.has_coherent


def test_eq25_sb_scales_sigma_inc_to_total(eq25_tape):
    expected = _H['sigma_inc'] * (_H['sigma_coh'] + _H['sigma_inc']) / _H['sigma_inc']
    m = from_endf_mf7mt2(str(eq25_tape), T_K=296.0)
    assert m.sigma_b == pytest.approx(expected, rel=1e-6)


def test_eq26_sb_carries_redistribution_factor(eq26_tape):
    f_dc = _BE['fraction']
    redist = 1.0 + (f_dc / (1.0 - f_dc)) * (_BE['sigma_inc'] / _O['sigma_inc'])
    expected = _O['sigma_inc'] * redist
    m = from_endf_mf7mt2(str(eq26_tape), T_K=296.0)
    assert m.sigma_b == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("tape_fixture,awr", [("eq25_tape", _H['awr']),
                                              ("eq26_tape", _O['awr'])])
@pytest.mark.parametrize("temp_idx", [0, 1])
def test_wprime_survives_roundtrip(request, tape_fixture, awr, temp_idx):
    tape = request.getfixturevalue(tape_fixture)
    T = _TEMPR[temp_idx]
    m = from_endf_mf7mt2(str(tape), T_K=T)
    # 7-sigfig ENDF storage bounds the round-trip error
    expected_inv_mev = _DWPIX[temp_idx] / (awr * T * BK) / 1.0e3
    assert m.Wprime_invmeV == pytest.approx(expected_inv_mev, rel=1e-6)


# ---- review E1: generalized CEF stores the molecular SB = per-principal x npr

def _raw_mt2_sb(path):
    """The verbatim MF7/MT2 SB field of an LTHR=2 tape (molecular, x npr),
    bypassing the reader's per-principal division. LTHR=2 layout ONLY: on an
    LTHR=3 tape the coherent block sits between the HEAD and the incoherent
    TAB1, so the second head() would return T0 (a temperature, not SB) --
    fail loudly instead of returning it (use _parsed_mt2 for LTHR=3)."""
    cur = _Cursor(_mf7mt2_lines(str(path)))
    _za, _awr, lthr, _l2, _n1, _n2 = cur.head()  # HEAD: ZA, AWR, LTHR, ...
    assert lthr == 2, f"_raw_mt2_sb assumes the LTHR=2 layout, got LTHR={lthr}"
    return cur.head()[0]     # incoherent TAB1: C1 = SB


@pytest.fixture(scope="module")
def eq25_npr4_tape(tmp_path_factory):
    p = tmp_path_factory.mktemp("cef_eq25_npr4") / "eq25_npr4.endf"
    _write_iel10_tape(p, 1001.0, _H['awr'], _ci([_H]), npr=4)
    return p


@pytest.fixture(scope="module")
def eq26_npr2_tape(tmp_path_factory):
    p = tmp_path_factory.mktemp("cef_eq26_npr2") / "eq26_npr2.endf"
    _write_iel10_tape(p, 8016.0, _O['awr'], _ci([_O, _BE], dc=1), npr=2)
    return p


def test_eq25_npr_stores_molecular_sb(eq25_npr4_tape):
    """The tape SB carries the classic molecular convention (x npr); dividing
    once by MT4's B(6)=npr recovers the Eq-25 per-principal effective value --
    the exact division the ENDFTSL converter applies uniformly (review E1:
    the generalized writer used to store the per-principal value raw, making
    that conversion 1/npr low). from_endf_mf7mt2 now applies the same
    division itself (PHY-3), so the reader returns the per-principal value."""
    per_principal = _H['sigma_inc'] * (
        (_H['sigma_coh'] + _H['sigma_inc']) / _H['sigma_inc'])
    assert _raw_mt2_sb(eq25_npr4_tape) == pytest.approx(4 * per_principal,
                                                        rel=1e-6)
    m = from_endf_mf7mt2(str(eq25_npr4_tape), T_K=296.0)
    assert m.sigma_b == pytest.approx(per_principal, rel=1e-6)


def test_eq26_npr_stores_molecular_sb(eq26_npr2_tape):
    f_dc = _BE['fraction']
    redist = 1.0 + (f_dc / (1.0 - f_dc)) * (_BE['sigma_inc'] / _O['sigma_inc'])
    per_principal = _O['sigma_inc'] * redist
    assert _raw_mt2_sb(eq26_npr2_tape) == pytest.approx(2 * per_principal,
                                                        rel=1e-6)
    m = from_endf_mf7mt2(str(eq26_npr2_tape), T_K=296.0)
    assert m.sigma_b == pytest.approx(per_principal, rel=1e-6)


def test_npr1_unchanged(eq25_tape):
    """npr=1 tapes (every existing generalized fixture) are byte-equivalent
    under the new convention: x1 is the identity."""
    per_principal = _H['sigma_inc'] * (
        (_H['sigma_coh'] + _H['sigma_inc']) / _H['sigma_inc'])
    m = from_endf_mf7mt2(str(eq25_tape), T_K=296.0)
    assert m.sigma_b == pytest.approx(per_principal, rel=1e-6)


# ---- PHY-1: the MEF (LTHR=3) writer must use the SAME molecular convention --
# The suite was entirely npr=1 before this, which is exactly why the missing
# "* npr" in _build_mef_elastic survived: at npr=1 the bug is invisible.

@pytest.fixture(scope="module")
def mef_npr2_tape(tmp_path_factory):
    p = tmp_path_factory.mktemp("mef_npr2") / "mef_npr2.endf"
    _write_iel10_tape(p, 4009.0, _BE['awr'], _ci_mef([_BE, _O]), npr=2,
                      bragg=_BRAGG)
    return p


@pytest.fixture(scope="module")
def mef_npr1_tape(tmp_path_factory):
    p = tmp_path_factory.mktemp("mef_npr1") / "mef_npr1.endf"
    _write_iel10_tape(p, 4009.0, _BE['awr'], _ci_mef([_BE, _O]), npr=1,
                      bragg=_BRAGG)
    return p


@pytest.fixture(scope="module")
def classic_npr2_tape(tmp_path_factory):
    """Classic (iel<0) incoherent-elastic tape for the SAME principal as the
    MEF fixtures: spr is chosen so the classic writer's internal bound
    sb = spr*((1+awr)/awr)^2 equals _BE's sigma_inc, making its SB directly
    comparable to the MEF and SEF values."""
    p = tmp_path_factory.mktemp("classic_npr2") / "classic_npr2.endf"
    spr = _BE['sigma_inc'] * (_BE['awr'] / (1.0 + _BE['awr'])) ** 2
    ssm = np.full((2, 2, 2), 0.1)
    write_endf_output(
        str(p), 1, 4009.0, _BE['awr'], spr, 2, -1,   # mat za awr spr npr iel
        0, 0.0, 0.0, 0.0, 0, 2, 2, 1,                # nss b7 aws sps mss nalpha nbeta lat
        np.array([0.1, 1.0]), np.array([0.0, 1.0]), ssm, None,
        np.array(_TEMPR), 2, np.array(_DWPIX), np.array(_DWPIX),
        np.array([320.0, 520.0]), np.array([320.0, 520.0]),
        [], 0, 0, 0, 1.0e-6,
        comments=None, crystal_info=None)
    return p


def _parsed_mt2(path):
    """The full MF7/MT2 dict via the real ENDF parser: layout-independent, so
    it works for LTHR=2 and LTHR=3 alike (the raw cursor helper above assumes
    the LTHR=2 layout, where the incoherent TAB1 follows the HEAD directly).
    For LTHR=3 the coherent edges land in T0/LT/S_T0_table (+ per-LT 'T'/'S')
    and the incoherent block in SB/NBT/INT/Tint/Wp."""
    from endf_parserpy import EndfParserPy
    parser = EndfParserPy(ignore_number_mismatch=True,
                          ignore_zero_mismatch=True,
                          ignore_varspec_mismatch=True)
    return parser.parsefile(str(path), include=[(7, 2)])[7][2]


def _parsed_mt2_sb(path):
    """MF7/MT2 SB via the real ENDF parser (see _parsed_mt2)."""
    return float(_parsed_mt2(path)['SB'])


def test_mef_stores_molecular_sb(mef_npr2_tape):
    """MF7/MT2 SB on a MEF tape is per-principal sigma_inc x npr, matching the
    classic (iel<0) and SEF/CEF writers, so a consumer's single division by
    MT4's B(6)=npr is correct regardless of which writer produced the tape
    (QA finding PHY-1)."""
    assert _parsed_mt2_sb(mef_npr2_tape) == pytest.approx(
        2 * _BE['sigma_inc'], rel=1e-6)


def test_mef_npr1_unchanged(mef_npr1_tape):
    """The npr=1 case is untouched by the fix: every shipped deck is npr=1, so
    a change here would be a regression, not the fix."""
    assert _parsed_mt2_sb(mef_npr1_tape) == pytest.approx(
        _BE['sigma_inc'], rel=1e-6)


def test_mef_and_cef_scale_by_npr_alike(mef_npr2_tape, mef_npr1_tape,
                                        eq26_tape, eq26_npr2_tape):
    """Both writers apply the SAME x npr molecular convention (PHY-1).

    Their SB VALUES differ by design and must not be compared directly: SEF
    assigns the whole coherent component to the designated-coherent atom and
    redistributes the remainder through the incoherent term, while MEF keeps
    both components separately. What has to match is the npr scaling, so
    compare each writer against ITSELF at npr=2 vs npr=1."""
    assert (_parsed_mt2_sb(mef_npr2_tape) /
            _parsed_mt2_sb(mef_npr1_tape)) == pytest.approx(2.0, rel=1e-6)
    assert (_parsed_mt2_sb(eq26_npr2_tape) /
            _parsed_mt2_sb(eq26_tape)) == pytest.approx(2.0, rel=1e-6)


def test_mef_classic_sef_writers_agree(mef_npr2_tape, classic_npr2_tape):
    """All THREE incoherent-SB writers store the identical molecular value for
    the same material at npr=2: the MEF (LTHR=3) tape, the classic iel<0
    (LTHR=2) tape, and the SEF/CEF builder called directly. This is the
    absolute cross-writer agreement (PHY-1): the relative test above proves
    only that each writer scales by npr, not that the classic path — the
    convention THERMR was written against — stores the same number."""
    expected = 2 * _BE['sigma_inc']
    assert _parsed_mt2_sb(mef_npr2_tape) == pytest.approx(expected, rel=1e-6)
    assert _raw_mt2_sb(classic_npr2_tape) == pytest.approx(expected, rel=1e-6)
    sb_sef = _build_cef_incoherent(1, 4009.0, _BE['awr'], 2, _TEMPR,
                                   list(_DWPIX), _BE['sigma_inc'], 2)['SB']
    assert sb_sef == pytest.approx(expected, rel=1e-6)


def test_mef_tapes_are_lthr3(mef_npr2_tape, mef_npr1_tape):
    """The MEF fixtures really produce LTHR=3 tapes: assert the flag on the
    tape itself rather than trusting the elastic_mode=2 dispatch. A fixture
    that silently fell into a CEF branch (LTHR=1/2) would make every "MEF"
    assertion above vacuous."""
    assert int(_parsed_mt2(mef_npr2_tape)['LTHR']) == 3
    assert int(_parsed_mt2(mef_npr1_tape)['LTHR']) == 3


def test_mef_coherent_block_npr_invariant(mef_npr2_tape, mef_npr1_tape):
    """Only the incoherent SB carries npr. The coherent edges (T0 TAB1 plus
    the higher-temperature LISTs) and the incoherent W'(T) table must be
    IDENTICAL between the npr=1 and npr=2 tapes: a writer that leaked npr
    into the per-atom coherent normalization or into W' would corrupt the
    elastic split while keeping the SB assertions above green. The two tapes
    differ only in npr, so exact equality is the right bar."""
    d1 = _parsed_mt2(mef_npr1_tape)
    d2 = _parsed_mt2(mef_npr2_tape)
    for key in ('T0', 'LT', 'S_T0_table', 'T', 'S'):   # coherent block
        assert d2[key] == d1[key], f"coherent field {key} changed with npr"
    assert d2['Tint'] == d1['Tint']                    # incoherent W'(T) table
    assert d2['Wp'] == d1['Wp']
    assert float(d2['SB']) == pytest.approx(2.0 * float(d1['SB']), rel=1e-6)


@pytest.mark.parametrize("temp_idx", [0, 1])
def test_mef_npr2_roundtrip_per_principal(mef_npr2_tape, temp_idx):
    """PHY-3 consumer side: from_endf_mf7mt2 on a MEF npr=2 tape divides the
    molecular SB by the npr it reads from the same tape's MF7/MT4 B(6) and
    returns the PER-PRINCIPAL sigma_inc, with both channels present and the
    W'(T) interpolation intact (W' is per-principal already and must NOT be
    divided)."""
    T = _TEMPR[temp_idx]
    assert _mf7mt4_npr(str(mef_npr2_tape)) == 2.0      # the divisor's source
    m = from_endf_mf7mt2(str(mef_npr2_tape), T_K=T)
    assert "LTHR=3" in m.label
    assert m.has_coherent and m.has_incoherent
    assert m.sigma_b == pytest.approx(_BE['sigma_inc'], rel=1e-6)
    expected_wp = _DWPIX[temp_idx] / (_BE['awr'] * T * BK) / 1.0e3
    assert m.Wprime_invmeV == pytest.approx(expected_wp, rel=1e-6)


def test_mef_reader_coherent_edges_match_tape(mef_npr2_tape):
    """from_endf_mf7mt2 walks the LTHR=3 coherent block correctly (a
    mis-aligned cursor there would also corrupt the incoherent TAB1 that
    follows): the de-cumulated Bragg edges reproduce the tape's cumulative
    S(E) at T0 and the edge energies survive the eV->meV conversion."""
    d = _parsed_mt2(mef_npr2_tape)
    m = from_endf_mf7mt2(str(mef_npr2_tape), T_K=296.0)
    assert m.E_edge_meV == pytest.approx(
        [1.0e3 * e for e in d['S_T0_table']['Eint']], rel=1e-9)
    assert np.cumsum(m.f_bragg) * 1.0e-3 == pytest.approx(
        d['S_T0_table']['S'], rel=1e-6)
