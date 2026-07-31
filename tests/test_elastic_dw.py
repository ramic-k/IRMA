"""irma.core.elastic_dw (P4) -- byte-pin the shared DW edge kernels.

Guards the extraction that replaced three copy-pasted ``_edge_delta`` closures
in endf_writer. Each kernel is checked against an INLINE reference that
reproduces the original closure arithmetic operation-for-operation (exact ``==``,
not approx), the ``* scale`` with ``scale==1.0`` is pinned as an exact no-op for
both the python-float and numpy-float paths, ``resolve_species_dw`` detection is
pinned, and the CEF/MEF builders are shown to share one arithmetic (their edge
deltas differ by exactly the CEF ``scale``). Fast, data-free, CI-safe.
"""
import numpy as np
import pytest
from math import exp

from irma.core.constants import BK
from irma.core.elastic_dw import (
    resolve_species_dw, isotropic_edge_delta,
    per_species_edge_delta, directional_edge_delta, make_edge_delta,
)
from irma.core.endf_writer import _build_cef_coherent, _build_mef_elastic


# ---- reference reimplementations of the ORIGINAL closure arithmetic ---------
def _ref_iso(w, e, amp, scale):
    return exp(-4.0 * w * e) * amp * scale


def _ref_ps(e, nsp, b_sqb, W_ps, itemp, sc_j, scale):
    delta = 0.0
    for si in range(nsp):
        w_si = W_ps[si][itemp]
        for ti in range(nsp):
            dw = exp(-2.0 * (w_si + W_ps[ti][itemp]) * e)
            delta += b_sqb[si] * b_sqb[ti] * dw * sc_j[si, ti]
    return delta * scale


def _ref_dir(e, nsp, b_sqb, awr_sp, kT, F_itemp, dir_terms_j, scale):
    delta = 0.0
    for G_hat, D_st_plane in dir_terms_j:
        for si in range(nsp):
            W_si = float(G_hat @ F_itemp[si] @ G_hat) / (awr_sp[si] * kT)
            for ti in range(nsp):
                W_ti = float(G_hat @ F_itemp[ti] @ G_hat) / (awr_sp[ti] * kT)
                dw = exp(-2.0 * (W_si + W_ti) * e)
                delta += b_sqb[si] * b_sqb[ti] * dw * D_st_plane[si, ti]
    return delta * scale


def _two_species_sdw(directional, ntempr=3):
    tempr = [296.0, 600.0, 1000.0][:ntempr]
    atom_types = [
        {'Z': 6, 'A': 12, 'awr': 11.898, 'b_coh': 6.646, 'sigma_inc': 0.001,
         'dwpix': [0.5, 0.4, 0.3]},
        {'Z': 8, 'A': 16, 'awr': 15.858, 'b_coh': 5.803, 'sigma_inc': 0.0008,
         'dwpix': [0.4, 0.32, 0.25]},
    ]
    sc = np.array([[[1.0, 0.3], [0.3, 0.8]], [[0.9, 0.2], [0.2, 0.7]]])
    ci = {'atom_types': atom_types, 'principal_atom_idx': 0, 'elastic_mode': 1,
          'species_corr': sc, 'F_species_per_temp': None, 'bragg_dir_terms': None}
    if directional:
        ci['F_species_per_temp'] = [[np.diag([0.5, 0.5, 3.0]),
                                     np.diag([0.4, 0.4, 2.5])]] * ntempr
        ci['bragg_dir_terms'] = [
            [(np.array([0.0, 0.0, 1.0]), sc[0]), (np.array([1.0, 0.0, 0.0]), sc[1])],
            [(np.array([1.0, 1.0, 1.0]) / np.sqrt(3), sc[0])]]
    return ci, tempr


# ---- kernels reproduce the original arithmetic exactly ----------------------
@pytest.mark.parametrize("scale", [1.0, 1.37])
def test_isotropic_kernel_exact(scale):
    for w, e, amp in [(0.5, 0.002, 1.1), (0.3, 0.05, 0.7), (10.0, 1e-4, 2.0)]:
        assert isotropic_edge_delta(w, e, amp, scale) == _ref_iso(w, e, amp, scale)


@pytest.mark.parametrize("scale", [1.0, 1.37])
def test_per_species_kernel_exact(scale):
    ci, tempr = _two_species_sdw(directional=False)
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    for itemp in range(len(tempr)):
        for j, e in enumerate([0.002, 0.01]):
            got = per_species_edge_delta(e, sdw, itemp, sdw.sc[j], scale)
            ref = _ref_ps(e, sdw.nsp, sdw.b_sqb, sdw.W_ps, itemp, sdw.sc[j], scale)
            assert got == ref


@pytest.mark.parametrize("scale", [1.0, 1.37])
def test_directional_kernel_exact(scale):
    ci, tempr = _two_species_sdw(directional=True)
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    for itemp in range(len(tempr)):
        for j, e in enumerate([0.002, 0.01]):
            kT = tempr[itemp] * BK
            got = directional_edge_delta(e, sdw, itemp, sdw.bragg_dir_terms[j], kT, scale)
            ref = _ref_dir(e, sdw.nsp, sdw.b_sqb, sdw.awr_sp, kT,
                           sdw.F_species_per_temp[itemp], sdw.bragg_dir_terms[j], scale)
            assert got == ref


def test_scale_one_is_exact_noop_all_paths():
    """delta * 1.0 == delta bit-for-bit (python float AND numpy float)."""
    assert isotropic_edge_delta(0.5, 0.002, 1.1, 1.0) == isotropic_edge_delta(0.5, 0.002, 1.1)
    ci_ps, tempr = _two_species_sdw(directional=False)
    sps = resolve_species_dw(ci_ps, tempr, len(tempr))
    d_ps = per_species_edge_delta(0.01, sps, 1, sps.sc[0])  # default scale=1.0
    assert d_ps == _ref_ps(0.01, sps.nsp, sps.b_sqb, sps.W_ps, 1, sps.sc[0], 1.0)
    ci_dir, tempr = _two_species_sdw(directional=True)
    sdir = resolve_species_dw(ci_dir, tempr, len(tempr))
    kT = tempr[1] * BK
    d_dir = directional_edge_delta(0.01, sdir, 1, sdir.bragg_dir_terms[0], kT)
    assert d_dir == _ref_dir(0.01, sdir.nsp, sdir.b_sqb, sdir.awr_sp, kT,
                             sdir.F_species_per_temp[1], sdir.bragg_dir_terms[0], 1.0)


# ---- resolver detection -----------------------------------------------------
def test_resolver_picks_directional_over_per_species():
    ci, tempr = _two_species_sdw(directional=True)
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    assert sdw.use_dir_dw and not sdw.use_ps and sdw.W_ps is None


def test_resolver_per_species_when_no_F():
    ci, tempr = _two_species_sdw(directional=False)
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    assert sdw.use_ps and not sdw.use_dir_dw and sdw.W_ps is not None


def test_resolver_none_for_isotropic():
    assert resolve_species_dw(None, [296.0], 1) is None
    assert resolve_species_dw({'species_corr': None}, [296.0], 1) is None


def test_resolver_directional_only_crystal_info_needs_no_species_corr():
    """species_corr is optional for the directional path (F41): a
    crystal_info carrying only F_species_per_temp + bragg_dir_terms must
    resolve to the directional kernel, not KeyError."""
    ci, tempr = _two_species_sdw(directional=True)
    del ci['species_corr']
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    assert sdw is not None
    assert sdw.use_dir_dw and not sdw.use_ps
    assert sdw.sc is None and sdw.W_ps is None
    # the directional kernel itself never touches sc
    kT = tempr[0] * BK
    d = directional_edge_delta(0.002, sdw, 0, sdw.bragg_dir_terms[0], kT)
    assert d == _ref_dir(0.002, sdw.nsp, sdw.b_sqb, sdw.awr_sp, kT,
                         sdw.F_species_per_temp[0], sdw.bragg_dir_terms[0], 1.0)


def test_resolver_W_ps_matches_endf_formula():
    ci, tempr = _two_species_sdw(directional=False)
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    for si, at in enumerate(ci['atom_types']):
        for it in range(len(tempr)):
            assert sdw.W_ps[si][it] == at['dwpix'][it] / (at['awr'] * tempr[it] * BK)


# ---- make_edge_delta dispatch ----------------------------------------------
def test_factory_dispatches_directional():
    ci, tempr = _two_species_sdw(directional=True)
    sdw = resolve_species_dw(ci, tempr, len(tempr))
    bragg = [(0.002, 1.0), (0.01, 1.1)]
    ed = make_edge_delta(bragg, [0.5, 0.4, 0.3], scale=1.37, species_dw=sdw, tempr=tempr)
    kT = tempr[1] * BK
    assert ed(0, 1) == _ref_dir(bragg[0][0], sdw.nsp, sdw.b_sqb, sdw.awr_sp, kT,
                                sdw.F_species_per_temp[1], sdw.bragg_dir_terms[0], 1.37)
    # energy override feeds the exponential, not bragg[j][0]
    assert ed(0, 1, energy=0.05) == _ref_dir(0.05, sdw.nsp, sdw.b_sqb, sdw.awr_sp, kT,
                                             sdw.F_species_per_temp[1], sdw.bragg_dir_terms[0], 1.37)


def test_factory_isotropic_when_no_species_dw():
    bragg = [(0.002, 1.1), (0.01, 0.7)]
    ed = make_edge_delta(bragg, [0.5, 0.4], scale=1.37)
    assert ed(1, 0) == _ref_iso(0.5, 0.01, 0.7, 1.37)


# ---- CEF and MEF share one arithmetic (differ only by the CEF scale) --------
def test_cef_and_mef_edge_arithmetic_differ_only_by_scale():
    """The whole point of the extraction: CEF == scale * MEF per edge."""
    ci_dir, tempr = _two_species_sdw(directional=True, ntempr=1)
    bragg = [(0.002, 1.0), (0.004, 1.0)]
    scale = 1.37
    cef = _build_cef_coherent(1, 6012.0, 11.898, bragg, 2, 1, tempr,
                              [0.5], scale, {**ci_dir})
    mef = _build_mef_elastic(1, 6012.0, 11.898, bragg, 2, 1, tempr,
                             {**ci_dir}, [0.5])
    cef_S = np.array(cef['S_T0_table']['S'][:2])
    mef_S = np.array(mef['S_T0_table']['S'][:2])
    # cumulative S scales linearly with the per-edge delta scale (pre-sigfig the
    # deltas are exactly scale*; compare to several sig-figs post-rounding)
    np.testing.assert_allclose(cef_S, scale * mef_S, rtol=1e-6)
