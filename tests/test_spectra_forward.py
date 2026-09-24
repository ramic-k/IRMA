"""irma.spectra forward-model physics: kinematics, detailed balance, the
elastic Bragg and Debye-Waller forms, instrument projection and the MF7/MT2
reader, on the in-repo ENDF/B graphite tape and synthetic S(Q,E) laws.
"""
import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

from irma.spectra import sqe as si
from irma.spectra import elastic as el
from irma.spectra import instruments as ins

SIGMA_B_C = 5.551
T = 296.0

# A real published ENDF/B coherent-elastic graphite tape vendored in the repo, so
# the elastic parser + peak/edge/bank gates run on CI with no external tree. The
# uncompressed *.endf is gitignored (.gitignore: *.endf), but the .gz IS tracked
# -- the `graphite_endf` fixture below decompresses it to a temp file so these
# tests run on a clean git checkout (and local == CI: both read the same .gz).
REPO_GRAPHITE_ENDF_GZ = (Path(__file__).resolve().parent /
                         "native_LEAPR_NJOY_ENDF_validation" / "leapr_decks" /
                         "tsl-crystalline-graphite.endf.gz")


@pytest.fixture(scope="module")
def graphite_endf(tmp_path_factory):
    """Path to the published ENDF/B graphite tape, decompressed from the tracked
    .endf.gz to a temp file (the plain .endf is gitignored)."""
    if not REPO_GRAPHITE_ENDF_GZ.exists():
        pytest.skip("graphite tape (.endf.gz) not present in checkout")
    out = tmp_path_factory.mktemp("graphite_tape") / "tsl-crystalline-graphite.endf"
    with gzip.open(REPO_GRAPHITE_ENDF_GZ, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return str(out)


def _synthetic_powder_qe(nq=24, nE=60, T_K=T):
    """A smooth, strictly-positive S(Q,E) PowderSQE on a (q, E) grid -- data-free.

    Used by the detailed-balance + geometry-projection gates, which exercise the
    energy-gain reconstruction and the instrument loci, not any material-specific
    number -- so a synthetic powder runs them on CI without the external tree.
    """
    q = np.linspace(0.5, 10.0, nq)
    E = np.linspace(0.0, 200.0, nE)
    gQ = (q ** 2 / (1.0 + (q / 6.0) ** 4))[:, None]
    fE = np.exp(-0.5 * ((E - 40.0) / 18.0) ** 2)[None, :]
    S = gQ * fE + 1.0e-6
    return si.from_noncubic_arrays(q, E, S, T_K=T_K, sigma_b=SIGMA_B_C)


# ---- constants / kinematics -------------------------------------------------
def test_k_of_E_roundtrip():
    E = np.array([3.5, 50.0, 250.0])
    k = si.k_of_E(E)
    assert np.allclose(si.C_E * k**2, E)


def test_indirect_elastic_Q():
    Ef = 3.5
    kf = np.sqrt(Ef / si.C_E)
    for tt in (45.0, 90.0, 135.0):
        Q = si.Q_indirect(np.array([0.0]), Ef, tt)[0]
        assert Q == pytest.approx(2 * kf * np.sin(np.deg2rad(tt) / 2), rel=1e-6)


def test_direct_elastic_Q():
    Ei = 250.0
    ki = np.sqrt(Ei / si.C_E)
    Q = si.Q_direct(np.array([0.0]), Ei, 90.0)[0]
    assert Q == pytest.approx(2 * ki * np.sin(np.deg2rad(90.0) / 2), rel=1e-6)


def test_indirect_Q_grows_with_loss():
    Ef = 3.5
    E = np.array([0.0, 50.0, 200.0])
    Q = si.Q_indirect(E, Ef, 135.0)
    assert np.all(np.diff(Q) > 0)


# ---- detailed balance -------------------------------------------------------
def test_detailed_balance_gain_side():
    """signed_sqe builds the energy-GAIN side by detailed balance:
    S(Q,-E) = exp(-E/kT) S(Q,+E). Data-free (synthetic powder) -- exercises the
    reconstruction every forward-model projection relies on."""
    p = _synthetic_powder_qe()
    q, Es, Ss = si.signed_sqe(p, include_gain=True)
    kT = si.KB * T
    iE = int(np.argmin(np.abs(Es - 60.0)))
    iEn = int(np.argmin(np.abs(Es + Es[iE])))
    ratio = Ss[:, iEn] / np.clip(Ss[:, iE], 1e-30, None)
    good = Ss[:, iE] > 1e-12
    assert np.allclose(ratio[good], np.exp(-Es[iE] / kT), rtol=1e-6)


# ---- elastic: parse + self-test --------------------------------------------
def test_elastic_peaks_integrate_to_sigma_coh(graphite_endf):
    """Bragg-peak normalization self-test on the in-repo published graphite tape:
    the Bragg-summed elastic cross section equals sigma_coherent at every energy.
    The first Bragg edge is near 1.876 1/A."""
    m = el.from_endf_mf7mt2(graphite_endf, T_K=T)
    assert m.has_coherent and m.Q_bragg.size > 100
    assert m.Q_bragg[0] == pytest.approx(1.876, abs=1e-2)
    for E in (20.0, 100.0, 200.0, 400.0):
        lhs, rhs = el.selftest(m, E_meV=E)
        assert lhs == pytest.approx(rhs, rel=1e-6)


def test_edge_to_shell_mapping(graphite_endf):
    m = el.from_endf_mf7mt2(graphite_endf, T_K=T)
    assert np.allclose(m.Q_bragg, 2 * np.sqrt(m.E_edge_meV / el.C_E), rtol=1e-9)


def test_incoherent_dw_form():
    m = el.ElasticModel(T_K=T, Q_bragg=np.array([]), f_bragg=np.array([]),
                        E_edge_meV=np.array([]), sigma_b=4.0,
                        Wprime_invmeV=0.02, has_coherent=False,
                        has_incoherent=True)
    E = 50.0
    k = np.sqrt(E / el.C_E)
    Q = np.linspace(1e-4, 2 * k, 400001)
    integ = m.incoherent_dsigma_dOmega(Q) * (Q * el.C_E / E)
    sig = 2 * np.pi * np.trapezoid(integ, Q)
    assert sig == pytest.approx(float(m.sigma_incoherent(E)[0]), rel=1e-4)


def test_bank_elastic_equals_sigma_coh_when_window_covers_all(graphite_endf):
    m = el.from_endf_mf7mt2(graphite_endf, T_K=T)
    Ef = 3.5
    xs = el.bank_elastic_area(m, Ef, 90.0, dtheta_deg=89.0, per_sr=False)
    assert xs == pytest.approx(float(m.sigma_coherent(Ef)[0]), rel=1e-6)


# ---- end-to-end geometry ----------------------------------------------------
def test_simulate_runs_all_geometries(graphite_endf):
    """End-to-end projection of a synthetic powder + real in-repo elastic model
    through all three instrument geometries. Runs on CI."""
    p = _synthetic_powder_qe(nq=48, nE=120)
    m = el.from_endf_mf7mt2(graphite_endf, T_K=T)
    E = np.linspace(-30, 300, 331)
    for instr in (ins.VISION(),
                  ins.indirect(3.5, [92.0], bank_halfwidth_deg=10.0),
                  ins.direct(250.0, [30, 90])):
        r = ins.simulate(p, instr, E, elastic_model=m)
        assert r["I_total"].shape == E.shape
        assert np.all(np.isfinite(r["I_total"]))
        assert r["I_inelastic"].max() > 0


# ---- constant-Q cut band integration (cut dQ) -- self-contained --------------
def _synthetic_powder(gofQ):
    """A PowderSQE with S(Q,E) = gofQ(Q) * peak(E) on a fine grid."""
    q = np.linspace(1.0, 10.0, 91)              # 0.1/A spacing
    E = np.linspace(0.0, 50.0, 101)
    fE = np.exp(-0.5 * ((E - 20.0) / 3.0) ** 2)  # a peak at 20 meV
    S = np.asarray(gofQ(q))[:, None] * fE[None, :]
    return si.from_noncubic_arrays(q, E, S, T_K=300.0, sigma_b=1.0), E


def test_constant_q_cut_band_averages_over_dq():
    """A finite cut dQ averages S(Q,E) over |Q| in [Q0-dQ, Q0+dQ]. For a convex
    S ~ Q^2 the band cut exceeds the thin slice by <Q^2>/Q0^2 = 1 + dQ^2/(3 Q0^2);
    for a Q-independent S the band and thin cuts are identical."""
    p, E = _synthetic_powder(lambda q: q ** 2)
    thin = ins.simulate_q_cuts(p, [5.0], E, (0.5, 0, 0),
                               include_gain=False)["I_inelastic_per_q"][0]
    band = ins.simulate_q_cuts(p, [5.0], E, (0.5, 0, 0), include_gain=False,
                               q_band=1.0, q_res=0.1)["I_inelastic_per_q"][0]
    assert band.max() / thin.max() == pytest.approx(1.0 + 1.0 / (3 * 25.0), rel=0.03)

    pflat, E2 = _synthetic_powder(lambda q: np.ones_like(q))
    t2 = ins.simulate_q_cuts(pflat, [5.0], E2, (0.5, 0, 0),
                             include_gain=False)["I_inelastic_per_q"][0]
    b2 = ins.simulate_q_cuts(pflat, [5.0], E2, (0.5, 0, 0), include_gain=False,
                             q_band=1.0, q_res=0.1)["I_inelastic_per_q"][0]
    assert np.allclose(t2, b2, rtol=1e-6)        # Q-independent -> band == thin


# ---- masked 2-D map export ---------------------------------------------------
def test_save_sqe_map_full_and_masked(tmp_path):
    from irma.spectra.forward import save_sqe_map, kinematic_mask
    Q = np.linspace(0.5, 10.0, 20)
    E = np.linspace(0.0, 100.0, 11)
    S = np.ones((Q.size, E.size))
    env = (E, np.full_like(E, 2.0), np.full_like(E, 8.0))   # accessible 2..8 /A
    m = kinematic_mask(Q, *env[1:])

    full = np.load(save_sqe_map(str(tmp_path / "m.npz"), Q, E, S, envelope=env))
    assert np.array_equal(full["S"], S) and "envelope_E" in full.files

    masked = np.load(save_sqe_map(str(tmp_path / "mm.npz"), Q, E, S,
                                  envelope=env, masked=True))
    assert np.all(np.isnan(masked["S"][~m])) and np.all(masked["S"][m] == 1.0)

    csv = np.loadtxt(save_sqe_map(str(tmp_path / "mm.csv"), Q, E, S,
                                  envelope=env, masked=True), delimiter=",", skiprows=1)
    assert csv.shape == (int(m.sum()), 3)        # only accessible cells, Q,E,S
    assert np.all((csv[:, 0] >= 2.0 - 1e-9) & (csv[:, 0] <= 8.0 + 1e-9))


def test_locus_support_extends_loss_grid_for_deep_gain():
    """A gain side requested below -e_max must extend the engine loss
    grid to |e_min| so the gain wing has loss data to mirror (else the
    fill_value=0 interpolator silently zeros it). For |e_min| <= e_max the grid
    is unchanged (byte-stable for the common case)."""
    from irma.spectra.forward import build_locus_support
    _, deep = build_locus_support("direct", 500.0, [45.0, 135.0], dE=5.0,
                                  e_max=100.0, dQ=0.1, e_min=-300.0,
                                  include_gain=True)
    assert deep.max() >= 300.0 - 1e-9                 # extended to cover the gain wing
    _, nogain = build_locus_support("direct", 500.0, [45.0, 135.0], dE=5.0,
                                    e_max=100.0, dQ=0.1, e_min=-300.0,
                                    include_gain=False)
    assert nogain.max() <= 100.0 + 5.0               # no gain side -> not extended
    _, shallow = build_locus_support("direct", 500.0, [45.0, 135.0], dE=5.0,
                                     e_max=100.0, dQ=0.1, e_min=-50.0,
                                     include_gain=True)
    assert shallow.max() <= 100.0 + 5.0              # |e_min| <= e_max -> unchanged


# ---- MF7/MT2 reader cursor row-alignment -------------------------------------
def test_mf7mt2_every_tabulated_temperature_reachable(graphite_endf):
    """The published graphite tape tabulates 296..2000 K (LT=9). A misaligned
    record cursor burns LIST iterations per temperature, so a request above
    700 K silently returns the 700 K block. Every tabulated temperature must
    be selected exactly, and the cumulative-S total must decrease
    monotonically with T (Debye-Waller)."""
    tabulated = [296.0, 400.0, 500.0, 600.0, 700.0, 800.0,
                 1000.0, 1200.0, 1600.0, 2000.0]
    sums = []
    for T_tab in tabulated:
        m = el.from_endf_mf7mt2(graphite_endf, T_K=T_tab)
        assert m.T_K == pytest.approx(T_tab), (
            f"requested {T_tab} K, parser selected {m.T_K} K")
        sums.append(float(np.sum(m.f_bragg)))
    assert all(s2 < s1 for s1, s2 in zip(sums, sums[1:])), (
        "sum of Bragg structure factors must fall with temperature")


def test_mf7mt2_top_edges_not_lost(graphite_endf):
    """A mid-row cursor drops the last data pairs of the edge table (186 of
    187 edges; top edge 1279.7 meV instead of the table's 5000 meV)."""
    m = el.from_endf_mf7mt2(graphite_endf, T_K=296.0)
    assert m.E_edge_meV[-1] == pytest.approx(5000.0, rel=1e-6)


def _endf11(x):
    return f"{x:11.4E}"


def _synthetic_lthr2_tape(path, SB=80.0, table=((200.0, 30.0), (300.0, 40.0))):
    """A minimal format-correct MF7/MT2 LTHR=2 section: HEAD + TAB1 whose
    (T, W') data line follows a blank-padded interpolation line."""
    def rec(fields):
        body = "".join(_endf11(f) if f is not None else " " * 11
                       for f in fields).ljust(66)
        return body + "1234" + " 7" + "  2" + "12345"
    NP = len(table)
    lines = [
        rec([1001.0, 0.999, 2, 0, 0, 0]),            # HEAD: ZA, AWR, LTHR=2
        rec([SB, 0.0, 0, 0, 1, NP]),                 # TAB1 head: SB, NR=1, NP
        rec([float(NP), 2.0, None, None, None, None]),  # interp (NBT, INT) + blanks
        rec([v for tw in table for v in tw]
            + [None] * (6 - 2 * NP)),                # (T, W') pairs, blank-padded
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def test_mf7mt2_incoherent_wprime_not_zero(tmp_path):
    """LTHR=2 with NP<=2 must not read only the interpolation line's blank
    padding (W'=0 -> a Debye-Waller-free elastic line at all Q)."""
    tape = _synthetic_lthr2_tape(str(tmp_path / "lthr2.endf"))
    m = el.from_endf_mf7mt2(tape, T_K=250.0)
    assert m.has_incoherent and m.sigma_b == pytest.approx(80.0)
    # np.interp(250, [200, 300], [30, 40]) = 35 eV^-1 = 0.035 meV^-1
    assert m.Wprime_invmeV == pytest.approx(0.035, rel=1e-12)


# ---- constant-Q cut bands restricted to the powder Q-support ------------------
def test_q_cut_band_not_diluted_by_out_of_support_zeros():
    pytest.importorskip("scipy")                # sqe_interpolator needs scipy
    q = np.linspace(1.0, 5.0, 81)
    E = np.linspace(0.0, 50.0, 101)
    p = si.PowderSQE(q=q, E=E, S=np.ones((q.size, E.size)), T_K=300.0,
                     sigma_b=1.0, label="flat")
    E_out = np.linspace(1.0, 40.0, 80)
    out = ins.simulate_q_cuts(p, [1.2, 3.0], E_out, [0.5],
                              include_gain=False, q_band=1.0)
    I_edge = float(np.trapezoid(out["I_inelastic_per_q"][0], E_out))
    I_mid = float(np.trapezoid(out["I_inelastic_per_q"][1], E_out))
    # a band clipped at the support edge must not average in exact zeros
    # from below q_min: the flat law gives the same integral at both cuts
    assert I_edge == pytest.approx(I_mid, rel=0.02)
