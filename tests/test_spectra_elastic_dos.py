"""DOS-based (mode-0) elastic line: irma.spectra.elastic.from_dos_and_lattice.

Pins the isotropic coherent-Bragg + incoherent-DW builder used by the DOS path.
The key physics check is an EXACT equivalence with the anisotropic engine builder
``from_engine_elastic_state`` when the thermal-displacement matrix is isotropic
(U = u0 I): both reduce the elastic Debye-Waller to 2W = alpha*lambda_s with the
same lambda_s, so the directional and per-species-isotropic kernels must agree
bit-for-bit on the Bragg comb and the incoherent W'.

Combs here are enumerated to emax_eV=0.3 -- every assertion is independent of
the comb's reach (channel bookkeeping, DW monotonicity, builder equivalence at
MATCHED emax), and truncation purity itself is pinned in
test_spectra_elastic_reach.py. The full-5 eV enumeration stays pinned in
test_spectra_elastic_from_engine.py::test_bragg_geometry_matches_compute_bragg_edges.
"""
import numpy as np
import pytest

from irma.core.constants import BK, HBAR2_OVER_2MN_MEV_A2 as C_E
from irma.core.crystal import CrystalStructure, AtomSite
from irma.spectra.elastic import from_dos_and_lattice, from_engine_elastic_state


def _bcc(a=2.866, b_coh_fm=9.45):
    """A BCC-iron-like 2-atom cubic cell, one species."""
    return CrystalStructure(a, a, a, 90.0, 90.0, 90.0,
                            [AtomSite(b_coh_fm=b_coh_fm,
                                      positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])])


# ---- basic behaviour --------------------------------------------------------
def test_builds_finite_model_with_both_channels():
    em = from_dos_and_lattice(_bcc(), awr=[55.0], sigma_inc_b=[0.4],
                              f0_lambda=[3.0], T_K=296.0, elastic_kind="both",
                              emax_eV=0.3)
    assert em.has_coherent and em.has_incoherent
    assert em.Q_bragg.size > 0 and np.all(np.diff(em.Q_bragg) >= -1e-12)   # sorted
    assert np.all(em.f_bragg > 0) and np.all(np.isfinite(em.f_bragg))
    # per-ATOM incoherent: BCC has mult=2, N=2 -> sigma_b = (mult/N)*sigma_inc = 0.4
    assert em.sigma_b == pytest.approx(0.4)
    assert len(em.incoherent_channels) == 1
    kT_meV = BK * 1000.0 * 296.0
    sb, wp = em.incoherent_channels[0]
    assert sb == pytest.approx(0.4)
    assert wp == pytest.approx(3.0 / (55.0 * kT_meV), rel=1e-12)   # W' = lambda/(awr kT)


def test_higher_f0_attenuates_the_comb_monotonically():
    sums = []
    for f0 in (0.0, 3.0, 8.0):
        em = from_dos_and_lattice(_bcc(), awr=[55.0], sigma_inc_b=[0.4],
                                  f0_lambda=[f0], T_K=296.0,
                                  elastic_kind="coherent", emax_eV=0.3)
        sums.append(em.f_bragg.sum())
    assert sums[0] > sums[1] > sums[2] > 0.0      # Debye-Waller suppresses edges


@pytest.mark.parametrize("kind,coh,inc", [
    ("both", True, True), ("coherent", True, False), ("incoherent", False, True)])
def test_elastic_kind_isolates_channels(kind, coh, inc):
    em = from_dos_and_lattice(_bcc(), awr=[55.0], sigma_inc_b=[0.4],
                              f0_lambda=[3.0], T_K=296.0, elastic_kind=kind,
                              emax_eV=0.3)
    assert em.has_coherent is coh and em.has_incoherent is inc
    if not coh:
        assert em.Q_bragg.size == 0


# ---- the physics cross-check: isotropic U == DOS f0 -------------------------
def test_matches_engine_builder_for_isotropic_U():
    """from_dos_and_lattice(f0) == from_engine_elastic_state(U=u0 I) EXACTLY when
    f0 = trace(F)/3 = awr*kT_meV*u0/C_E. Both are per-atom now, so for this
    single-species BCC cell the Bragg comb, the incoherent line and W' all match
    bit-for-bit (the per-atom-average incoherent reduces to the single principal
    channel for one species)."""
    a = 2.866
    awr, b_coh_fm, sig_inc, T_K, u0 = 55.0, 9.45, 0.4, 296.0, 0.0052
    kT_meV = BK * 1000.0 * T_K
    f0 = awr * kT_meV * u0 / C_E                       # trace(F)/3 for U = u0 I

    # engine path: isotropic U for both atoms
    U = np.broadcast_to(u0 * np.eye(3), (2, 3, 3)).copy()
    es = {
        "thermal_displacement_matrices_ang2": U,
        "primitive_symbols": ["Fe", "Fe"],
        "primitive_scaled_positions": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        "primitive_lattice_ang": np.diag([a, a, a]).astype(float),
        "temperature_k": T_K,
    }
    eng = from_engine_elastic_state(es, b_coh_fm=b_coh_fm, sigma_inc_b=sig_inc,
                                    awr=awr, T_K=T_K, elastic_kind="both",
                                    emax_eV=0.3)
    dos = from_dos_and_lattice(_bcc(a, b_coh_fm), awr=[awr], sigma_inc_b=[sig_inc],
                               f0_lambda=[f0], T_K=T_K, elastic_kind="both",
                               emax_eV=0.3)

    assert dos.Q_bragg.size == eng.Q_bragg.size and dos.Q_bragg.size > 0
    assert np.allclose(dos.Q_bragg, eng.Q_bragg, rtol=1e-12, atol=1e-12)
    assert np.allclose(dos.f_bragg, eng.f_bragg, rtol=1e-10, atol=1e-18)   # per-atom comb
    Q = np.linspace(0.5, 12.0, 40)
    assert np.allclose(dos.incoherent_dsigma_dOmega(Q),
                       eng.incoherent_dsigma_dOmega(Q), rtol=1e-12)
    assert dos.sigma_b == pytest.approx(eng.sigma_b)


def test_incoherent_sums_all_species_not_just_principal():
    """The incoherent line keeps EVERY species even when the largest-|b_coh|
    scatterer has zero sigma_inc (the old principal-only bug)."""
    cr = CrystalStructure(4.0, 4.0, 4.0, 90, 90, 90, [
        AtomSite(b_coh_fm=6.0, positions=[(0.0, 0.0, 0.0)]),         # bigger |b_coh|, no inc
        AtomSite(b_coh_fm=-3.0, positions=[(0.5, 0.5, 0.5)]),        # carries the incoherent
    ])
    # 2 sites, 1 atom each -> N=2; per-atom channel sigma_b = (mult/N)*sigma_inc
    em = from_dos_and_lattice(cr, awr=[12.0, 16.0], sigma_inc_b=[0.0, 0.5],
                              f0_lambda=[2.0, 1.5], T_K=296.0, elastic_kind="both",
                              emax_eV=0.3)
    assert em.has_incoherent is True and em.has_coherent is True       # NOT dropped
    assert len(em.incoherent_channels) == 1 and em.sigma_b == pytest.approx(0.5 / 2)
    em2 = from_dos_and_lattice(cr, awr=[12.0, 16.0], sigma_inc_b=[0.3, 0.5],
                               f0_lambda=[2.0, 1.5], T_K=296.0, elastic_kind="both",
                               emax_eV=0.3)
    assert len(em2.incoherent_channels) == 2
    assert em2.sigma_b == pytest.approx((0.3 + 0.5) / 2)


def test_ch2_incoherent_keeps_hydrogen():
    """Headline case: C has the larger |b_coh| but H (sigma_inc~80, mult 2)
    dominates the incoherent elastic and must NOT be dropped."""
    cr = CrystalStructure(5.0, 5.0, 5.0, 90, 90, 90, [
        AtomSite(b_coh_fm=6.6460, positions=[(0.0, 0.0, 0.0)]),                 # C
        AtomSite(b_coh_fm=-3.7406, positions=[(0.25, 0.25, 0.25),
                                              (0.75, 0.75, 0.75)]),             # 2 H
    ])
    em = from_dos_and_lattice(cr, awr=[11.9, 0.9991], sigma_inc_b=[0.001, 80.26],
                              f0_lambda=[1.0, 5.0], T_K=296.0, elastic_kind="incoherent")
    assert em.has_incoherent and len(em.incoherent_channels) == 2
    sbs = sorted(sb for sb, _ in em.incoherent_channels)
    # N=3 (1 C + 2 H); H channel sigma_b = (mult_H/N)*sigma_inc_H = (2/3)*80.26
    assert sbs[-1] == pytest.approx((2 / 3) * 80.26)   # H present, not dropped


# ---- input validation -------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"awr": [55.0, 1.0]},                       # wrong length vs 1 species
    {"sigma_inc_b": [0.4, 0.4]},
    {"f0_lambda": [3.0, 3.0]},
    {"awr": [-1.0]},                            # awr must be > 0
    {"f0_lambda": [-0.1]},                      # f0 must be >= 0
    {"sigma_inc_b": [-0.1]},                    # sigma_inc must be >= 0
])
def test_bad_per_species_inputs_rejected(kw):
    base = dict(awr=[55.0], sigma_inc_b=[0.4], f0_lambda=[3.0])
    base.update(kw)
    with pytest.raises(ValueError):
        from_dos_and_lattice(_bcc(), T_K=296.0, elastic_kind="both", **base)


def test_bad_elastic_kind_rejected():
    with pytest.raises(ValueError):
        from_dos_and_lattice(_bcc(), awr=[55.0], sigma_inc_b=[0.4], f0_lambda=[3.0],
                             T_K=296.0, elastic_kind="bragg")
