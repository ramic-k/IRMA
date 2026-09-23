"""Site-resolved directional coherent-elastic Debye-Waller (review finding P3).

The directional MT2 path used to average the per-site displacement tensors
over each Card 6d species group BEFORE exponentiation. The correct physics
per reflection is the DW-attenuated complex amplitude sum

    F^2(E,T) = |sum_i b_i exp(-W_i(Ghat) E) exp(i phi_i)|^2,
    W_i(Ghat) = (Ghat . F_i . Ghat) / (awr_sp(i) kT),
    phi_i     = 2 pi (h,k,l) . pos_i,

with the code's 2-factor convention exp(-2 W_i e) per amplitude (so the
uniform-tensor limit reproduces the pair form exp(-2 (W_s+W_t) e) exactly).
Verified error of the old averaging: old/exact = 0.4807 for two in-phase
same-species sites with W1=0, W2=100 1/eV at (200), a=3 A; 0.9677 at W2=20.

These tests pin the new site-resolved branch of ``directional_edge_delta``
and ``make_sigma_coh_ext`` against inline exact references, the exact-zero
anti-phase cancellation, and the byte-pinned uniform fast path. Synthetic
states mirror tests/test_elastic_extinction.py's _synthetic_* helpers.
"""
import math
import types

import numpy as np
import pytest

from irma.core.constants import BK, WL2EKIN
from irma.core.elastic_dw import directional_edge_delta
from irma.core.elastic_extinction import make_sigma_coh_ext


# ---- a minimal two-sites-one-species crystal: a=3 A cubic-like cell ---------
_A0 = 3.0                              # lattice constant [A]
_D200 = _A0 / 2.0                      # (200) d-spacing
_E200 = WL2EKIN / (4.0 * _D200 ** 2)   # (200) Bragg-edge energy [eV]
_B = 0.6646                            # b_coh [sqrt(barn)]
_AWR = 11.898
_T = 296.0
_KT = _T * BK
_V, _N = _A0 ** 3, 2                   # cell volume [A^3], atoms/cell
_XSF = 0.5 * WL2EKIN / (_V * _N)
_GHAT = np.array([1.0, 0.0, 0.0])      # (h00) plane normal


def _site_terms(positions, hkl):
    """crystal.py-style per-plane site phases for a single-species cell."""
    h, k, l = hkl
    pos = np.asarray(positions, dtype=float)
    phase = 2.0 * np.pi * (h * pos[:, 0] + k * pos[:, 1] + l * pos[:, 2])
    cos_a, sin_a = np.cos(phase), np.sin(phase)
    d = _A0 / math.sqrt(h * h + k * k + l * l)
    pref = d * 2.0 * _XSF                          # d * mult * xsectfact
    sp_idx = np.zeros(len(pos), dtype=np.intp)
    return cos_a, sin_a, sp_idx, pref


def _plane(positions, hkl=(2, 0, 0), G_hat=_GHAT):
    """One bragg_dir_terms plane entry (G_hat, D_st, site_terms), with the
    species-collapsed D_st built exactly as compute_bragg_edges_general does:
    D_st = pref * (fr fr^T + fi fi^T)."""
    st = _site_terms(positions, hkl)
    cos_a, sin_a, _, pref = st
    fr, fi = float(np.sum(cos_a)), float(np.sum(sin_a))
    D_st = np.array([[fr * fr + fi * fi]]) * pref
    return (np.asarray(G_hat, dtype=float), D_st, st)


def _sdw(F_sites, uniform):
    """nsp=1 directional-DW state with two sites. The species-averaged tensor
    mirrors driver._store_directional_species_dw (np.mean over the group)."""
    F_sites = np.asarray(F_sites, dtype=float)
    return types.SimpleNamespace(
        use_dir_dw=True, use_ps=False, nsp=1, b_sqb=[_B], awr_sp=[_AWR],
        F_species_per_temp=[[F_sites.mean(axis=0)]],
        F_sites_per_temp=[F_sites],
        dir_tensors_uniform=uniform, W_ps=None, bragg_dir_terms=None)


def _F_scalar(W_inv_eV):
    """Isotropic site tensor whose directional DW exponent is W [1/eV]."""
    return W_inv_eV * _AWR * _KT * np.eye(3)


# ---- inline references -------------------------------------------------------
def _exact_site_sum(e, G_hat, st, F_sites, b_sqb=(_B,), awr=(_AWR,), kT=_KT):
    """Exact complex amplitude sum |sum_i b_i e^{-2 W_i e} e^{i phi_i}|^2 pref."""
    cos_a, sin_a, sp_idx, pref = st
    z = 0.0 + 0.0j
    for i in range(len(sp_idx)):
        sp = int(sp_idx[i])
        W = float(G_hat @ np.asarray(F_sites, dtype=float)[i] @ G_hat) / (awr[sp] * kT)
        z += b_sqb[sp] * math.exp(-2.0 * W * e) * complex(cos_a[i], sin_a[i])
    return (z.real * z.real + z.imag * z.imag) * pref


def _old_averaged(e, G_hat, D_st, F_avg_list, b_sqb=(_B,), awr=(_AWR,), kT=_KT):
    """The pre-P3 species double sum over the group-AVERAGED tensors."""
    nsp = len(F_avg_list)
    delta = 0.0
    for si in range(nsp):
        W_si = float(G_hat @ F_avg_list[si] @ G_hat) / (awr[si] * kT)
        for ti in range(nsp):
            W_ti = float(G_hat @ F_avg_list[ti] @ G_hat) / (awr[ti] * kT)
            delta += (b_sqb[si] * b_sqb[ti]
                      * math.exp(-2.0 * (W_si + W_ti) * e) * D_st[si, ti])
    return delta


_IN_PHASE = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]   # phi = 0, 2pi for (200)


# ---- 1+2: scalar-tensor contrast at the (200) edge ---------------------------
@pytest.mark.parametrize("W2, ratio_lo, ratio_hi", [
    # old/exact = 4 x / (1+x)^2 with x = exp(-2 W2 E200):
    #   W2=100 1/eV -> 0.4807 (the review's verified factor-2 error)
    #   W2=20  1/eV -> 0.9677 (the ~3.2% case)
    (100.0, 0.46, 0.50),
    (20.0, 0.95, 0.98),
])
def test_two_inphase_sites_scalar_contrast(W2, ratio_lo, ratio_hi):
    F_sites = np.array([_F_scalar(0.0), _F_scalar(W2)])
    sdw = _sdw(F_sites, uniform=False)
    plane = _plane(_IN_PHASE)
    e = _E200
    got = directional_edge_delta(e, sdw, 0, [plane], _KT)
    exact = _exact_site_sum(e, _GHAT, plane[2], F_sites)
    assert got == pytest.approx(exact, rel=1e-12)
    # the OLD averaged-tensor behavior is wrong by 4x/(1+x)^2
    old = _old_averaged(e, _GHAT, plane[1], sdw.F_species_per_temp[0])
    x = math.exp(-2.0 * W2 * e)
    assert old / exact == pytest.approx(4.0 * x / (1.0 + x) ** 2, rel=1e-9)
    assert ratio_lo < old / exact < ratio_hi


# ---- 3: rotated anisotropic tensors ------------------------------------------
def test_rotated_anisotropic_tensors():
    """Two sites whose tensors are 90-degree rotations of each other:
    diag(a,b,b) and diag(b,a,b). The group average is isotropic in the basal
    plane, so the old path sees W_avg for both sites; the site-resolved
    result must equal the exact complex-sum reference instead."""
    a_w, b_w = 30.0, 5.0
    F1 = np.diag([a_w, b_w, b_w]) * _AWR * _KT
    F2 = np.diag([b_w, a_w, b_w]) * _AWR * _KT
    F_sites = np.array([F1, F2])
    sdw = _sdw(F_sites, uniform=False)
    plane = _plane(_IN_PHASE)
    e = _E200
    got = directional_edge_delta(e, sdw, 0, [plane], _KT)
    exact = _exact_site_sum(e, _GHAT, plane[2], F_sites)
    assert got == pytest.approx(exact, rel=1e-12)
    # and the averaged behavior is measurably different (W1=30 vs W2=5 along x)
    old = _old_averaged(e, _GHAT, plane[1], sdw.F_species_per_temp[0])
    assert abs(old / exact - 1.0) > 0.01


# ---- 4: anti-phase cancellation ----------------------------------------------
def test_antiphase_cancellation_activates_with_tensor_contrast():
    """Two same-species sites in exact anti-phase (equal b): equal tensors
    cancel to delta == 0.0 EXACTLY; unequal tensors break the cancellation
    and must match the exact reference."""
    # hand-built exact anti-phase terms (a float representation of pi gives
    # sin(pi) ~ 1e-16, so build cos/sin directly for the exact-zero check)
    pref = _D200 * 2.0 * _XSF
    st = (np.array([1.0, -1.0]), np.array([0.0, 0.0]),
          np.array([0, 0], dtype=np.intp), pref)
    D_zero = np.array([[0.0]])                        # (1-1)^2 + 0^2
    plane = (_GHAT, D_zero, st)
    e = _E200

    F_eq = np.array([_F_scalar(10.0), _F_scalar(10.0)])
    got_eq = directional_edge_delta(e, _sdw(F_eq, uniform=False), 0, [plane], _KT)
    assert got_eq == 0.0                              # exact, not approx

    F_ne = np.array([_F_scalar(0.0), _F_scalar(100.0)])
    got_ne = directional_edge_delta(e, _sdw(F_ne, uniform=False), 0, [plane], _KT)
    exact = _exact_site_sum(e, _GHAT, st, F_ne)
    assert got_ne > 0.0
    assert got_ne == pytest.approx(exact, rel=1e-12)


# ---- 5: identical-tensor control (byte-pinned uniform fast path) --------------
def test_identical_tensors_keep_old_path_bit_for_bit():
    """With identical site tensors the uniform flag stays True and the kernel
    output is IDENTICAL (==, not approx) to the old species double sum -- the
    graphite/Be/BeO golden tapes ride on this."""
    F = np.diag([0.4, 0.4, 2.5]) * _AWR * _KT
    F_sites = np.array([F, F])
    sdw = _sdw(F_sites, uniform=True)
    assert sdw.dir_tensors_uniform is True
    # the group average of identical tensors is bitwise the tensor itself
    assert np.array_equal(sdw.F_species_per_temp[0][0], F)
    planes = [_plane(_IN_PHASE, hkl=(2, 0, 0)),
              _plane(_IN_PHASE, hkl=(4, 0, 0))]
    for e in (0.002, _E200, 0.05):
        got = directional_edge_delta(e, sdw, 0, planes, _KT)
        ref = sum(_old_averaged(e, p[0], p[1], sdw.F_species_per_temp[0])
                  for p in planes)
        assert got == ref                             # exact bit equality
        # and the site-resolved branch agrees to rounding (same physics here)
        got_sr = directional_edge_delta(
            e, _sdw(F_sites, uniform=False), 0, planes, _KT)
        assert got_sr == pytest.approx(got, rel=1e-12)


# ---- driver flag + site ordering ----------------------------------------------
def _driver_crystal_info(mesh_positions):
    return {
        'inelastic_mode': 2,
        'nc_mesh_data': types.SimpleNamespace(
            atom_positions=np.asarray(mesh_positions, dtype=float),
            atom_symbols=['C', 'C']),
        'nc_atom_type_site_groups': [[0, 1]],
        'atom_types': [{
            'Z': 6, 'A': 12, 'awr': _AWR, 'b_coh': 6.646, 'sigma_inc': 0.0,
            'positions': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        }],
    }


def test_driver_stores_site_tensors_in_card6d_order_and_flags_nonuniform():
    """_store_directional_species_dw must store per-site tensors reordered to
    the Card 6d position order (the site_terms flattening order) and set the
    uniformity flag from exact tensor equality."""
    from irma.core.driver import _store_directional_species_dw
    # phonopy atom 0 sits at (0.5,0,0), atom 1 at (0,0,0): REVERSED Card 6d order
    ci = _driver_crystal_info([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    F_half = np.diag([1.0, 2.0, 3.0])
    F_zero = np.diag([4.0, 5.0, 6.0])
    _store_directional_species_dw(ci, 0, 1,
                                  np.array([F_half, F_zero]))
    assert ci['dir_tensors_uniform'] is False
    F_sites = ci['F_sites_per_temp'][0]
    assert np.array_equal(F_sites[0], F_zero)        # Card 6d site (0,0,0)
    assert np.array_equal(F_sites[1], F_half)        # Card 6d site (0.5,0,0)
    # averaged tensor unchanged (fast path input)
    assert np.array_equal(ci['F_species_per_temp'][0][0],
                          np.mean([F_half, F_zero], axis=0))


def test_driver_flags_uniform_for_identical_tensors():
    from irma.core.driver import _store_directional_species_dw
    ci = _driver_crystal_info([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    F = np.diag([1.0, 2.0, 3.0])
    _store_directional_species_dw(ci, 0, 1, np.array([F, F]))
    assert ci['dir_tensors_uniform'] is True


def test_driver_flags_uniform_through_eigensolver_bit_noise():
    """Symmetry-equivalent sites carry ~1e-16 noise in symmetry-forbidden
    tensor elements; that must NOT kick the material off the byte-pinned
    species-averaged fast path (graphite/Be/BeO goldens ride on this)."""
    from irma.core.driver import _store_directional_species_dw
    ci = _driver_crystal_info([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    F = np.diag([1.0, 2.0, 3.0])
    F_noise = F.copy()
    F_noise[0, 2] = F_noise[2, 0] = 4.0e-16
    _store_directional_species_dw(ci, 0, 1,
                                  np.array([F, F_noise]))
    assert ci['dir_tensors_uniform'] is True
    # a real (graphite-sublattice-scale, ~0.1%) difference DOES flag
    ci2 = _driver_crystal_info([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    _store_directional_species_dw(ci2, 0, 1,
                                  np.array([F, F * 1.001]))
    assert ci2['dir_tensors_uniform'] is False


def test_driver_pairs_sites_modulo_shared_origin_shift():
    """The deck and the phonopy cell may use different origins (the vendored
    graphite model: deck z=1/4,3/4 vs phonopy z=0,1/2). Pairing must succeed
    modulo one shared shift, keeping tensor-to-position assignment intact."""
    from irma.core.driver import _store_directional_species_dw
    # phonopy cell shifted by (0,0,0.75) relative to the Card 6d positions
    ci = _driver_crystal_info([[0.0, 0.0, 0.75], [0.5, 0.0, 0.75]])
    F_a = np.diag([1.0, 2.0, 3.0])
    F_b = np.diag([4.0, 5.0, 6.0])
    _store_directional_species_dw(ci, 0, 1,
                                  np.array([F_a, F_b]))
    assert ci['dir_tensors_uniform'] is False
    F_sites = ci['F_sites_per_temp'][0]
    # deck (0,0,0) <-> phonopy atom 0 at (0,0,0.75); deck (0.5,0,0) <-> atom 1
    assert np.array_equal(F_sites[0], F_a)
    assert np.array_equal(F_sites[1], F_b)


def test_driver_raises_when_nonuniform_sites_cannot_be_paired():
    from irma.core.driver import _store_directional_species_dw
    # mesh positions match NO Card 6d position (under any shared shift) -> no
    # 1:1 pairing; with non-uniform tensors this must be a hard error, not a
    # silent mispairing
    ci = _driver_crystal_info([[0.1234, 0.2, 0.3], [0.7, 0.8, 0.9]])
    with pytest.raises(ValueError, match="cannot pair"):
        _store_directional_species_dw(
            ci, 0, 1,
            np.array([np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])]))




# ---- extinction path -------------------------------------------------------------
def _two_edge_bragg(F_sites, uniform):
    """(200)+(400) edge pair with 3-tuple dir terms and a matching sdw."""
    planes = [_plane(_IN_PHASE, hkl=(2, 0, 0)),
              _plane(_IN_PHASE, hkl=(4, 0, 0))]
    bragg, dir_terms = [], []
    for hkl, plane in zip([(2, 0, 0), (4, 0, 0)], planes):
        h = hkl[0]
        d = _A0 / h
        e_thr = WL2EKIN / (4.0 * d * d)
        bragg.append([e_thr, float(plane[1][0, 0]) ])
        dir_terms.append([plane])
    return np.array(bragg), dir_terms, _sdw(F_sites, uniform)


def test_extinction_site_resolved_delta_matches_reference():
    """make_sigma_coh_ext with a non-uniform state: the per-plane deltas must
    be the site-resolved values (checked via the no-extinction limit, where
    sigma(E) = sum of deltas / E exactly), and both uniform and non-uniform
    states must run without error under a real extinction model."""
    F_ne = np.array([_F_scalar(0.0), _F_scalar(100.0)])
    F_eq = np.array([_F_scalar(10.0), _F_scalar(10.0)])
    no_ext = dict(model="BC_pure", l=0.0, g=0.0, L=0.0, dist="Gauss",
                  recipe="std")
    real_ext = dict(model="BC_mix", l=8550.0, g=170.0, L=75750.0,
                    dist="Gauss", recipe="std")

    bragg, dir_terms, sdw_ne = _two_edge_bragg(F_ne, uniform=False)
    sig_ne, _, _ = make_sigma_coh_ext(bragg, dir_terms, sdw_ne,
                                      _V, _N, 1.0, no_ext, [_T])
    E = 2.0 * float(bragg[1][0])                    # above both edges
    ref = sum(_exact_site_sum(float(bragg[j][0]), dir_terms[j][0][0],
                              dir_terms[j][0][2], F_ne)
              for j in range(2)) / E
    assert sig_ne(E, 0) == pytest.approx(ref, rel=1e-12)

    # uniform state keeps the old (averaged == exact here) arithmetic
    bragg_u, dir_terms_u, sdw_eq = _two_edge_bragg(F_eq, uniform=True)
    sig_eq, _, _ = make_sigma_coh_ext(bragg_u, dir_terms_u, sdw_eq,
                                      _V, _N, 1.0, no_ext, [_T])
    ref_u = sum(_old_averaged(float(bragg_u[j][0]), dir_terms_u[j][0][0],
                              dir_terms_u[j][0][1],
                              sdw_eq.F_species_per_temp[0])
                for j in range(2)) / E
    assert sig_eq(E, 0) == pytest.approx(ref_u, rel=1e-12)

    # both run without error under a real extinction model, y <= 1
    for br, dt, sdw in [(bragg, dir_terms, sdw_ne),
                        (bragg_u, dir_terms_u, sdw_eq)]:
        sfn, _, _ = make_sigma_coh_ext(br, dt, sdw, _V, _N, 1.0,
                                       real_ext, [_T])
        kfn, _, _ = make_sigma_coh_ext(br, dt, sdw, _V, _N, 1.0,
                                       no_ext, [_T])
        s, k = sfn(E, 0), kfn(E, 0)
        assert 0.0 < s <= k * (1.0 + 1e-12)


# ---- review P2: multi-species cancellation must survive the zero-DW prune --

def test_cross_species_cancellation_retained(tmp_path):
    """A B2-like cell whose (100) cancels at zero attenuation (equal b, two
    species, anti-phase) must KEEP the plane under the default fsquarecut so
    unequal per-species Debye-Waller can activate it; single-species
    systematic absences must stay pruned."""
    from irma.core.crystal import (AtomSite, CrystalStructure,
                                   compute_bragg_edges_general)

    e100 = 0.0022723391555555554

    # two species, equal b, anti-phase at (100): exact zero-DW cancellation
    crystal = CrystalStructure(
        a=3.0, b=3.0, c=3.0, alpha=90.0, beta=90.0, gamma=90.0,
        sites=[AtomSite(b_coh_fm=5.0, positions=[(0.0, 0.0, 0.0)]),
               AtomSite(b_coh_fm=5.0, positions=[(0.5, 0.5, 0.5)])])
    bragg, nbe, corr, dir_terms = compute_bragg_edges_general(crystal, 0.05)
    has_100 = any(abs(float(bragg[j][0]) - e100) < 1e-9 for j in range(nbe))
    assert has_100, "(100) cross-species cancellation was pruned (review P2)"

    # single species, anti-phase: a systematic absence must remain pruned
    crystal1 = CrystalStructure(
        a=3.0, b=3.0, c=3.0, alpha=90.0, beta=90.0, gamma=90.0,
        sites=[AtomSite(b_coh_fm=5.0,
                        positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])])
    bragg1, nbe1, _, _ = compute_bragg_edges_general(crystal1, 0.05)
    has_100_single = any(abs(float(bragg1[j][0]) - e100) < 1e-9
                         for j in range(nbe1))
    assert not has_100_single, "systematic absence must stay pruned"
