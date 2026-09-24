"""irma.spectra.elastic.from_engine_elastic_state -- engine-free pins.

Builds an ElasticModel tape-free from a synthetic engine elastic_state (the DW
matrices + primitive geometry the noncubic engine surfaces) and checks the
contract the graphite gate validates against a real tape: the Bragg geometry is
reproduced exactly from the lattice, the coherent Bragg peaks integrate back to
sigma_coh (selftest), and the incoherent path produces the THERMR Debye-Waller
line. No phonopy / engine run needed -- the elastic_state is hand-built.

Bragg edges are enumerated to emax_eV=0.3, where every assertion here is
reach-independent.
"""
import numpy as np
import pytest

from irma.spectra.elastic import (
    from_engine_elastic_state, selftest, ElasticModel)
from irma.core.crystal import CrystalStructure, AtomSite, compute_bragg_edges_general
from irma.core.constants import HBAR2_OVER_2MN_MEV_A2 as C_E


def _state(a=3.567, U_iso=0.004, symbols=("C",), positions=((0.0, 0.0, 0.0),),
           T=296.0, masses=(12.0,)):
    n = len(symbols)
    return {
        "thermal_displacement_matrices_ang2": np.array([U_iso * np.eye(3)] * n),
        "primitive_lattice_ang": np.diag([a, a, a]).astype(float),
        "primitive_scaled_positions": np.asarray(positions, float),
        "primitive_symbols": list(symbols),
        "primitive_masses_amu": np.asarray(masses, float),
        "temperature_k": T,
    }


def test_coherent_builds_model_with_edges():
    m = from_engine_elastic_state(_state(), b_coh_fm=6.646, sigma_inc_b=0.001,
                                  awr=11.898, elastic_kind="coherent",
                                  emax_eV=0.3)
    assert isinstance(m, ElasticModel)
    assert m.has_coherent and not m.has_incoherent
    assert m.Q_bragg.size > 0
    assert np.all(np.diff(m.Q_bragg) >= 0)          # sorted ascending
    assert np.all(m.f_bragg > 0)                    # thinned of non-positive edges


def test_bragg_geometry_matches_compute_bragg_edges():
    """Edge positions come PURELY from the lattice -> reproduce the Bragg routine."""
    m = from_engine_elastic_state(_state(a=3.567), b_coh_fm=6.646,
                                  sigma_inc_b=0.001, awr=11.898, emax_eV=0.3)
    cr = CrystalStructure(3.567, 3.567, 3.567, 90.0, 90.0, 90.0,
                          [AtomSite(6.646, [(0.0, 0.0, 0.0)])])
    bd, *_ = compute_bragg_edges_general(cr, emax=0.3)
    # the builder drops edges with a zero structure factor
    Q_ref = np.sort(2.0 * np.sqrt(bd[bd[:, 1] > 0, 0] * 1000.0 / C_E))
    np.testing.assert_allclose(np.sort(m.Q_bragg), Q_ref, rtol=1e-9)


def test_selftest_holds_for_coherent_peaks():
    m = from_engine_elastic_state(_state(), b_coh_fm=6.646, sigma_inc_b=0.001,
                                  awr=11.898, elastic_kind="coherent",
                                  emax_eV=0.3)
    lhs, rhs = selftest(m, 200.0)
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_incoherent_path_is_debye_waller_line():
    m = from_engine_elastic_state(_state(U_iso=0.02), b_coh_fm=3.0,
                                  sigma_inc_b=80.0, awr=0.9999,
                                  elastic_kind="incoherent")
    assert m.has_incoherent and not m.has_coherent
    assert m.sigma_b == 80.0
    assert len(m.incoherent_channels) == 1
    assert m.Wprime_invmeV > 0
    # dsigma/dOmega(Q=0) = sigma_b/4pi
    assert float(m.incoherent_dsigma_dOmega(0.0)[0]) == pytest.approx(80.0 / (4 * np.pi))
    # attenuates with Q
    assert m.incoherent_dsigma_dOmega(8.0)[0] < m.incoherent_dsigma_dOmega(1.0)[0]


def test_both_carries_coherent_and_incoherent_channels():
    """Default 'both' = pure Bragg peaks + separate incoherent DW (no fold)."""
    both = from_engine_elastic_state(_state(), b_coh_fm=6.646, sigma_inc_b=2.0,
                                     awr=11.898,   # default elastic_kind='both'
                                     emax_eV=0.3)
    assert both.has_coherent and both.has_incoherent
    assert both.sigma_b == 2.0 and both.Wprime_invmeV > 0

    # the coherent Bragg peaks are PURE (no (sigma_coh+sigma_inc)/sigma_coh fold):
    # their f_bragg matches the coherent-only build exactly, regardless of sigma_inc.
    coh = from_engine_elastic_state(_state(), b_coh_fm=6.646, sigma_inc_b=2.0,
                                    awr=11.898, elastic_kind="coherent",
                                    emax_eV=0.3)
    np.testing.assert_allclose(both.f_bragg, coh.f_bragg, rtol=1e-12)
    assert not coh.has_incoherent

    # total elastic = coherent + incoherent (both channels add)
    Q = np.array([1.5, 4.0, 7.0])
    np.testing.assert_allclose(
        both.elastic_dsigma_dOmega(Q, q_res=0.05),
        both.coherent_dsigma_dOmega(Q, q_res=0.05) + both.incoherent_dsigma_dOmega(Q),
        rtol=1e-12)


def test_dw_is_determined_by_U_not_by_T_override():
    """The DW attenuation depends on U (which already carries temperature), so a
    T_K override at fixed U cancels out of F=A.kT.U/(hbar^2/2m_n) -> W=f0/(awr.kT)
    and leaves f_bragg unchanged. Larger U (hotter sample) DOES attenuate more."""
    base = from_engine_elastic_state(_state(U_iso=0.004), b_coh_fm=6.646,
                                     sigma_inc_b=0.001, awr=11.898, T_K=5.0,
                                     emax_eV=0.3)
    override = from_engine_elastic_state(_state(U_iso=0.004), b_coh_fm=6.646,
                                         sigma_inc_b=0.001, awr=11.898,
                                         T_K=1000.0, emax_eV=0.3)
    np.testing.assert_allclose(base.f_bragg, override.f_bragg, rtol=1e-12)
    hotter_U = from_engine_elastic_state(_state(U_iso=0.02), b_coh_fm=6.646,
                                         sigma_inc_b=0.001, awr=11.898, T_K=5.0,
                                         emax_eV=0.3)
    assert hotter_U.f_bragg[-1] < base.f_bragg[-1]   # more displacement -> more DW


def test_two_species_grouping():
    st = _state(a=4.5, symbols=("Na", "Cl"),
                positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)), masses=(22.99, 35.45))
    m = from_engine_elastic_state(st, b_coh_fm=[3.63, 9.58],
                                  sigma_inc_b=[1.62, 5.3], awr=[22.79, 35.15],
                                  elastic_kind="coherent", emax_eV=0.3)
    assert m.has_coherent and m.Q_bragg.size > 0
    lhs, rhs = selftest(m, 150.0)
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_incoherent_keeps_every_species_channel():
    """A hydrogenous-like sample must keep the strong-incoherent minority
    species' Debye-Waller line even when the coherent principal is the other
    species, not only the principal's sigma_inc."""
    # 'Zr'-like principal by |b_coh| with negligible sigma_inc; 'H'-like
    # minority with sigma_inc ~80 b. 1 Zr + 2 H per cell.
    st = _state(a=4.8, symbols=("Zr", "H", "H"),
                positions=((0.0, 0.0, 0.0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)),
                masses=(91.22, 1.008, 1.008))
    m = from_engine_elastic_state(st, b_coh_fm=[7.16, -3.74, -3.74],
                                  sigma_inc_b=[0.02, 80.27, 80.27],
                                  awr=[90.44, 0.9992, 0.9992],
                                  elastic_kind="incoherent")
    assert m.has_incoherent and not m.has_coherent
    # channels are (mult_d/N * sigma_inc_d, W'_d) per species
    assert len(m.incoherent_channels) == 2
    sigmas = sorted(sb for sb, _ in m.incoherent_channels)
    assert sigmas[0] == pytest.approx(0.02 / 3.0, rel=1e-12)        # Zr: 1/3 cell
    assert sigmas[1] == pytest.approx(2.0 * 80.27 / 3.0, rel=1e-12)  # H: 2/3 cell
    # the Q->0 incoherent level is dominated by H, not the principal's 0.02 b
    level = float(m.incoherent_dsigma_dOmega(np.array([1e-4]))[0])
    assert level == pytest.approx((0.02 / 3 + 2 * 80.27 / 3) / (4 * np.pi), rel=1e-6)
