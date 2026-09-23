"""Mechanical tests for the coherent-extinction integration (irma.core.
elastic_extinction + the endf_writer extinction S-table). Uses a synthetic
single-species directional-DW state; the full physics gate is the CrysXT-plugin
comparison (separate harness)."""
import math
import types

import numpy as np
import pytest

from irma.core.constants import WL2EKIN
from irma.core.elastic_extinction import make_sigma_coh_ext
from irma.core import endf_writer


# ---- a minimal Be-like single-species directional-DW state ----
_V, _N = 16.225, 2          # Be cell
_B = 0.779                  # b_coh(Be) in sqrt(barn)
_AWR = 8.93
_XSF = 0.5 * WL2EKIN / (_V * _N)
# (d [A], fsq [barn]) for a few Be families
_FAMILIES = [(1.980, 0.574), (1.792, 2.270), (1.233, 1.85), (1.020, 1.2)]


def _synthetic_species_dw(W_tensor_scale=0.0, use_dir=True):
    """nsp=1 single-species DW state. Directional (F-matrix) or per-species."""
    F = np.eye(3) * W_tensor_scale
    return types.SimpleNamespace(
        use_dir_dw=use_dir, use_ps=not use_dir, nsp=1, b_sqb=[_B], awr_sp=[_AWR],
        F_species_per_temp=[[F]], W_ps=[[0.0]], bragg_dir_terms=None)


def _synthetic_species_dw_multitemp(W_scales):
    """ntempr>1 directional-DW state: one F-tensor scale per temperature."""
    Fs = [[np.eye(3) * w] for w in W_scales]
    return types.SimpleNamespace(
        use_dir_dw=True, use_ps=False, nsp=1, b_sqb=[_B], awr_sp=[_AWR],
        F_species_per_temp=Fs, W_ps=[[0.0] * len(W_scales)], bragg_dir_terms=None)


def _synthetic_bragg():
    bragg, dir_terms = [], []
    for d, fsq in _FAMILIES:
        e_thr = WL2EKIN / (4.0 * d * d)
        sig = d * fsq * 2.0 * _XSF          # mult=2 kinematic weight
        bragg.append([e_thr, sig])
        # one (h,k,l) dir-term carrying this family's |F|^2 via D_st = d*(fsq/b^2)*mult*xsf
        D_st = np.array([[d * (fsq / _B ** 2) * 2.0 * _XSF]])
        dir_terms.append([(np.array([0.0, 0.0, 1.0]), D_st)])
    bragg.append([WL2EKIN / (4.0 * 0.5 ** 2), 0.0])     # emax flat extension
    dir_terms.append([])
    return np.array(bragg), dir_terms


def _cfg(**kw):
    base = dict(model="BC_mix", l=8550.0, g=170.0, L=75750.0, dist="Gauss",
                recipe="std")
    base.update(kw)
    return base


def test_sigma_coh_ext_runs_and_reduces_below_kinematic():
    sdw = _synthetic_species_dw()
    bragg = _synthetic_bragg()[0]
    sigma_fn, edge_E, _ = make_sigma_coh_ext(bragg, _synthetic_bragg()[1], sdw, _V, _N, 1.0, _cfg(), [296.0])
    # kinematic reference (no extinction): BC_pure with no sizes -> y -> 1
    kin_fn, _, _ = make_sigma_coh_ext(bragg, _synthetic_bragg()[1], sdw, _V, _N, 1.0,
                                      _cfg(model="BC_pure", l=0.0, g=0.0, L=0.0), [296.0])
    for E in (0.006, 0.02, 0.1):
        ext = sigma_fn(E, 0)
        kin = kin_fn(E, 0)
        assert ext > 0.0
        assert ext < kin * 1.0001                 # extinction never increases sigma
        assert ext > 0.5 * kin                     # ... and is a modest reduction here


def test_no_extinction_limit_matches_kinematic():
    # l=g=L=0 -> y == 1 everywhere -> sigma equals the plain kinematic edge sum
    sdw = _synthetic_species_dw()
    bragg = _synthetic_bragg()[0]
    sig_fn, _, _ = make_sigma_coh_ext(bragg, _synthetic_bragg()[1], sdw, _V, _N, 1.0,
                                      _cfg(model="BC_pure", l=0.0, g=0.0, L=0.0), [296.0])
    # plain kinematic: (1/E) sum_{E_j<=E} sigma_j
    for E in (0.006, 0.02, 0.1):
        kin = sum(b[1] for b in bragg if b[0] <= E) / E
        assert sig_fn(E, 0) == pytest.approx(kin, rel=1e-12)


def test_extinction_s_table_is_histogram_and_monotonic():
    sdw = _synthetic_species_dw()
    bragg, dir_terms = _synthetic_bragg()
    sigma_fn, edge_E, E_active = make_sigma_coh_ext(
        bragg, dir_terms, sdw, _V, _N, 1.0, _cfg(), [296.0])
    # minimal kinematic edge table (one node per edge energy, cumulative kinematic S)
    Ek, Scum, acc = [], [], 0.0
    for bj in bragg:
        acc += bj[1]
        Ek.append(endf_writer.sigfig(float(bj[0]), 7, 0))
        Scum.append(endf_writer.sigfig(acc, 7, 0))
    kin_table = {'S_T0_table': {'Eint': Ek, 'S': Scum}}
    tbl = endf_writer._coherent_extinction_s_table(
        kin_table, sigma_fn, edge_E, E_active, 1, [296.0], tol=1e-3)
    st = tbl['S_T0_table']
    assert st['INT'] == [1]                         # histogram (standard MF7/MT2)
    assert tbl['NP'] == len(st['Eint']) == len(st['S'])
    assert all(b >= a for a, b in zip(st['S'], st['S'][1:]))     # monotonic S
    assert all(b > a for a, b in zip(st['Eint'], st['Eint'][1:]))  # ascending E

    # read as a STEP function (what THERMR does), reproduces sigma_fn between edges
    import bisect

    def table_sigma_step(E):
        j = bisect.bisect_right(st['Eint'], E) - 1
        return None if j < 0 else st['S'][j] / E

    for ea, eb in zip(edge_E[:-1], edge_E[1:]):
        e_mid = math.sqrt(ea * eb)               # geometric mid-gap
        got = table_sigma_step(e_mid)
        if got is not None and e_mid > edge_E[0] * 1.001:
            assert got == pytest.approx(sigma_fn(e_mid, 0), rel=2e-2)


def test_per_species_isotropic_dw_path_runs():
    # inelastic_mode=0 path: per-species (non-directional) DW must also work
    sdw = _synthetic_species_dw(use_dir=False)
    bragg, dir_terms = _synthetic_bragg()
    sig_fn, _, _ = make_sigma_coh_ext(bragg, dir_terms, sdw, _V, _N, 1.0,
                                      _cfg(), [296.0])
    assert 0.0 < sig_fn(0.02, 0) < 1e4


def test_debye_waller_attenuation_lowers_sigma():
    """The per-plane structure factor carries exp(-2 (W_i+W_j) E_thr): a LARGER
    Debye-Waller exponent must LOWER the coherent-elastic sigma (more thermal
    disorder weakens the Bragg peaks). This pins the SIGN of the DW term -- a
    flipped sign (e.g. exp(+...) instead of exp(-...)) would make sigma grow
    explosively with W, which this test catches. Uses no extinction (BC_pure with
    no sizes -> y == 1) so the only variable is the DW weight."""
    bragg, dir_terms = _synthetic_bragg()
    cfg = _cfg(model="BC_pure", l=0.0, g=0.0, L=0.0)
    cold, _, _ = make_sigma_coh_ext(
        bragg, dir_terms, _synthetic_species_dw(W_tensor_scale=0.0), _V, _N, 1.0, cfg, [296.0])
    warm, _, _ = make_sigma_coh_ext(
        bragg, dir_terms, _synthetic_species_dw(W_tensor_scale=5.0), _V, _N, 1.0, cfg, [296.0])
    saw_attenuation = False
    for E in (0.02, 0.1, 0.3):
        c, w = cold(E, 0), warm(E, 0)
        assert w > 0.0
        assert w < c * (1.0 + 1e-9)          # DW never raises sigma
        if c > 0.0 and w < c * 0.999:
            saw_attenuation = True
    assert saw_attenuation                    # and it measurably lowers it somewhere


def test_multi_temperature_sigma_path():
    """ntempr>1 must build a per-temperature kinematic prefix + per-temperature
    cutoff, and return a DIFFERENT (more DW-attenuated) sigma at the hotter index.
    Single-temperature tests never exercise the per_temp[itemp] indexing."""
    bragg, dir_terms = _synthetic_bragg()
    sdw = _synthetic_species_dw_multitemp([1.0, 12.0])   # T0 colder, T1 hotter DW
    tempr = [296.0, 1200.0]
    sig_fn, edge_E, _ = make_sigma_coh_ext(
        bragg, dir_terms, sdw, _V, _N, 1.0, _cfg(), tempr)
    for E in (0.02, 0.1):
        s0, s1 = sig_fn(E, 0), sig_fn(E, 1)
        assert s0 > 0.0 and s1 > 0.0
        assert s1 < s0                         # hotter -> stronger DW -> lower sigma


def test_extinction_s_table_multitemp_has_per_temp_block():
    """The writer must emit LI=2 + a per-temperature S block for ntempr>1, with one
    S entry per assembled node, all finite. Covers the multi-temperature splice
    that the single-temperature table test does not."""
    bragg, dir_terms = _synthetic_bragg()
    sdw = _synthetic_species_dw_multitemp([1.0, 12.0])
    tempr = [296.0, 1200.0]
    sig_fn, edge_E, E_active = make_sigma_coh_ext(
        bragg, dir_terms, sdw, _V, _N, 1.0, _cfg(), tempr)
    Ek, S0, acc = [], [], 0.0
    for bj in bragg:
        acc += bj[1]
        Ek.append(endf_writer.sigfig(float(bj[0]), 7, 0))
        S0.append(endf_writer.sigfig(acc, 7, 0))
    kin_table = {'S_T0_table': {'Eint': Ek, 'S': S0},
                 'S': {q + 1: {1: S0[q]} for q in range(len(Ek))}}
    tbl = endf_writer._coherent_extinction_s_table(
        kin_table, sig_fn, edge_E, E_active, 2, tempr, tol=1e-3)
    assert tbl['LI'] == 2 and tbl['T'][1] == 1200.0
    nS = len(tbl['S_T0_table']['Eint'])
    assert set(tbl['S']) == set(range(1, nS + 1))         # one S entry per node
    assert all(1 in tbl['S'][q] for q in range(1, nS + 1))
    s1 = [tbl['S'][q][1] for q in range(1, nS + 1)]
    assert all(np.isfinite(v) for v in s1)
    assert all(b >= a - 1e-9 for a, b in zip(s1, s1[1:]))  # cumulative -> monotonic


def test_extinction_splice_consumes_real_grouped_edges():
    """Feed the extinction splice the edge table produced by the real grouping
    helper. Edges above the 1 eV threshold are merged; the splice reuses those
    grouped nodes verbatim while the sub-cutoff extinction region stays
    identical to the ungrouped splice."""
    sdw = _synthetic_species_dw()
    bragg, dir_terms = _synthetic_bragg()
    sigma_fn, edge_E, _ = make_sigma_coh_ext(
        bragg, dir_terms, sdw, _V, _N, 1.0, _cfg(), [296.0])
    E_active = 0.15
    lo = sorted(edge_E[:-1])                       # real Be-like sub-cutoff edges
    hi = [1.0 + 0.05 * i for i in range(40)]       # 40 dense edges in [1, 3] eV
    edgeE = lo + hi
    cbragg = np.array([[e, 1.0] for e in edgeE])

    def edge_delta(j, itemp, energy=None):
        return 1.0                                 # unit weight per edge

    kin_g = endf_writer._coherent_s_table_or_grouped(
        cbragg, len(edgeE), 1, [296.0], edge_delta,
        {'coh_edge_group_bins_per_decade': 10, 'coh_edge_group_threshold_ev': 1.0})
    kin_p = endf_writer._coherent_s_table_or_grouped(
        cbragg, len(edgeE), 1, [296.0], edge_delta, {})
    assert len(kin_g['S_T0_table']['Eint']) < len(kin_p['S_T0_table']['Eint'])

    tg = endf_writer._coherent_extinction_s_table(
        kin_g, sigma_fn, edge_E, E_active, 1, [296.0], 1e-3)
    tp = endf_writer._coherent_extinction_s_table(
        kin_p, sigma_fn, edge_E, E_active, 1, [296.0], 1e-3)
    assert tg['NP'] < tp['NP']                     # grouped splice is smaller
    # the nodes above the cutoff are reused unchanged
    assert tp['NP'] - tg['NP'] == (len(kin_p['S_T0_table']['Eint'])
                                   - len(kin_g['S_T0_table']['Eint']))
    below_g = [e for e in tg['S_T0_table']['Eint'] if e < E_active]
    below_p = [e for e in tp['S_T0_table']['Eint'] if e < E_active]
    assert below_g == below_p and len(below_g) > 0  # extinction region identical


# ---- the scan stop must be a proven bound, not a heuristic -----------------

def _weak_edges_plus_strong(n_weak=25, planes_per_weak=21, fsq_weak=1e-12,
                           e_strong=0.01, fsq_strong=5.0):
    """A long run of negligible edges (enough to satisfy the consecutive-below
    counter and the plane-count floor) followed by one strong reflection at
    higher energy."""
    bragg, dir_terms = [], []
    for i in range(n_weak):
        e_thr = 0.001 + (0.0034 - 0.001) * i / (n_weak - 1)
        d = math.sqrt(WL2EKIN / (4.0 * e_thr))
        bragg.append([e_thr, d * fsq_weak * 2.0 * _XSF * planes_per_weak])
        D_st = np.array([[d * (fsq_weak / _B ** 2) * 2.0 * _XSF]])
        dir_terms.append([(np.array([0.0, 0.0, 1.0]), D_st)] * planes_per_weak)
    d_s = math.sqrt(WL2EKIN / (4.0 * e_strong))
    bragg.append([e_strong, d_s * fsq_strong * 2.0 * _XSF])
    D_s = np.array([[d_s * (fsq_strong / _B ** 2) * 2.0 * _XSF]])
    dir_terms.append([(np.array([0.0, 0.0, 1.0]), D_s)])
    return np.array(bragg), dir_terms


def test_late_strong_reflection_is_not_spliced_kinematically():
    """A strong reflection after 25 weak edges must stay inside E_active and
    keep its extinction correction (the old consecutive-below stop returned it
    as uncorrected kinematic; the tail bound must prevent that)."""
    sdw = _synthetic_species_dw()
    bragg, dir_terms = _weak_edges_plus_strong()
    cfg = _cfg(model="BC_pure", l=1.0e7, g=0.0, L=0.0)
    sigma_fn, _, E_active = make_sigma_coh_ext(
        bragg, dir_terms, sdw, _V, _N, 1.0, cfg, [296.0])
    assert E_active > 0.01                        # strong edge inside active region
    kin_fn, _, _ = make_sigma_coh_ext(
        bragg, dir_terms, sdw, _V, _N, 1.0,
        _cfg(model="BC_pure", l=0.0, g=0.0, L=0.0), [296.0])
    E = 0.0105
    ext, kin = sigma_fn(E, 0), kin_fn(E, 0)
    assert kin > 0.0
    # l=1e7 A crystallites extinguish the strong plane almost completely; the
    # returned sigma must reflect that, not the kinematic edge sum.
    assert ext < 0.05 * kin
