"""Elastic line on the dense 2-D S(Q,E) map (compute_sqe_map / run_map).

Before this file, map mode silently IGNORED the elastic switch: ``run_map``
passed nothing elastic to ``compute_sqe_map``, so a config with
``physics.elastic: true`` produced an inelastic-only map while the SAME config
through ``run_spectra`` carried the line -- and there was no warning. These
tests pin the new contract:

  * the map deposits the SAME per-Q elastic area the 1-D path evaluates,
    ``elastic_dsigma_dOmega(Q, q_res=dQ_map)`` [barn/sr], as an E=0 line whose
    energy integral is EXACTLY that area (two-bin linear split around E=0 --
    uniform output grids generally have no exact-zero bin);
  * the deposit happens BEFORE the resolution pass, so broadening gives the
    line sigma(E=0) like every other feature while conserving its integral;
  * all three config routes reach the map: mode-0 DOS-derived (lattice-free
    incoherent + crystal Bragg peaks), an explicit ElasticModel (the ENDF-tape
    route), and metadata flags reporting what was built.

CI-safe: mode-0 DOS path + a synthetic engine elastic_state -- pure numpy, no
phonopy, no engine run.
"""
import numpy as np

from irma.spectra.forward import compute_sqe_map
from irma.spectra.elastic import (
    from_dos_elastic, from_engine_elastic_state,
    instrument_reach_emax_eV)
from irma.core.crystal import CrystalStructure, AtomSite

T_K = 296.0


def _dos(w_max_meV=40.0, opt_meV=90.0, n=400, delta_ev=0.0005):
    """A Debye acoustic band + a Gaussian optical peak on a uniform omega grid (eV)."""
    omega = np.arange(n) * delta_ev
    w = omega * 1000.0                                   # meV
    rho = np.where(w <= w_max_meV, (w / w_max_meV) ** 2, 0.0)
    rho += 0.5 * np.exp(-0.5 * ((w - opt_meV) / 5.0) ** 2)
    return omega, rho


def _carbon(**extra):
    """A graphite-like single species for the DOS path (values not load-bearing)."""
    omega, rho = _dos()
    sp = {"symbol": "C", "omega_ev": omega, "rho": rho, "awr": 11.898,
          "sigma_bound_b": 5.551, "sigma_inc_b": 0.001, "multiplicity": 1}
    sp.update(extra)
    return sp


GRAPHITE_LATTICE = (2.464, 2.464, 6.711, 90.0, 90.0, 120.0)
GRAPHITE_SITES = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5),
                  (1.0 / 3, 2.0 / 3, 0.0), (2.0 / 3, 1.0 / 3, 0.5)]

# mode-0 ignores phonopy_yaml/mesh/sab_*; they are required kwargs, so pass
# placeholders. nphon pinned explicit (auto-order OFF) for determinism/speed.
BASE = dict(phonopy_yaml=None, temperature_k=T_K, mesh=None,
            sab_mass_ratio=11.898, sab_sigma_barn=5.551,
            multiphonon_max_order=40, auto_multiphonon_order=False,
            progress=lambda *a, **k: None)
# an output grid whose E axis crosses 0 BETWEEN bins (-20 + 13*1.5 = -0.5):
# the deposit must land split across the (-0.5, 1.0) pair, nothing elsewhere.
GRID = dict(geometry="direct", e_fixed_meV=250.0, q_min=1.0, q_max=8.0,
            dQ_map=0.25, e_min=-20.0, e_max=60.0, dE=1.5)


def _zero_bins(E):
    """Indices of the two output-energy bins bracketing E=0."""
    j = int(np.searchsorted(E, 0.0, side="right") - 1)
    j = min(max(j, 0), E.size - 2)
    return j, j + 1


def _f0(sqe_map):
    """The DOS-derived isotropic Debye-Waller f0 the mode-0 path computed."""
    return float(sqe_map.metadata["engine_metadata"]["per_species"][0]["dw_lambda"])


def test_map_mode0_incoherent_elastic_line_quantitative():
    """Lattice-free incoherent line: the on/off map difference is confined to
    the two bins bracketing E=0, and its energy integral per Q reproduces the
    reference Debye-Waller area exactly."""
    sp = _carbon(sigma_inc_b=1.2)
    m_off = compute_sqe_map(dos_species=[sp], elastic=False, broaden=False,
                            **GRID, **BASE)
    m_on = compute_sqe_map(dos_species=[sp], elastic=True,
                           elastic_kind="incoherent", broaden=False,
                           **GRID, **BASE)
    assert m_off.metadata["elastic"] is False
    assert m_on.metadata["elastic"] is True
    assert m_on.metadata["elastic_kind"] == "incoherent"
    assert m_on.metadata["n_bragg_edges"] == 0

    diff = m_on.S - m_off.S
    j0, j1 = _zero_bins(m_on.E)
    off_line = np.ones(m_on.E.size, bool)
    off_line[[j0, j1]] = False
    assert np.all(diff[:, off_line] == 0.0)          # deposit localized at E=0
    assert diff[:, [j0, j1]].min() >= 0.0

    ref = from_dos_elastic(
        None, awr=[sp["awr"]], sigma_inc_b=[1.2], f0_lambda=[_f0(m_on)],
        multiplicity=[1], T_K=T_K)
    area = ref.incoherent_dsigma_dOmega(m_on.Q)
    got = diff.sum(axis=1) * GRID["dE"]              # energy integral per Q
    assert np.allclose(got, area, rtol=1e-9)
    assert np.all(np.diff(got) < 0.0)                # pure DW decay with Q


def test_map_mode0_coherent_elastic_bragg_peaks():
    """dos_crystal enables the Bragg peaks: edges appear in the metadata and the
    deposited area matches the q_res=dQ_map-broadened reference peaks + DW line."""
    sp = _carbon(b_coh_fm=6.646, positions=GRAPHITE_SITES)
    m_off = compute_sqe_map(dos_species=[sp], elastic=False, broaden=False,
                            **GRID, **BASE)
    m_on = compute_sqe_map(dos_species=[sp], elastic=True, elastic_kind="both",
                           dos_crystal=GRAPHITE_LATTICE, broaden=False,
                           **GRID, **BASE)
    assert m_on.metadata["elastic"] is True
    assert m_on.metadata["n_bragg_edges"] > 0

    diff = m_on.S - m_off.S
    j0, j1 = _zero_bins(m_on.E)
    off_line = np.ones(m_on.E.size, bool)
    off_line[[j0, j1]] = False
    assert np.all(diff[:, off_line] == 0.0)

    crystal = CrystalStructure(*GRAPHITE_LATTICE,
                               [AtomSite(b_coh_fm=6.646,
                                         positions=GRAPHITE_SITES)])
    ref = from_dos_elastic(
        crystal, awr=[sp["awr"]], sigma_inc_b=[sp["sigma_inc_b"]],
        f0_lambda=[_f0(m_on)], multiplicity=[len(GRAPHITE_SITES)], T_K=T_K,
        elastic_kind="both",
        emax_eV=instrument_reach_emax_eV(q_max_invA=float(m_on.Q.max()),
                                         q_cuts=[float(m_on.Q.max())],
                                         q_res_invA=GRID["dQ_map"]))
    area = ref.elastic_dsigma_dOmega(m_on.Q, q_res=GRID["dQ_map"])
    got = diff.sum(axis=1) * GRID["dE"]
    assert np.allclose(got, area, rtol=1e-9)
    # the Bragg peaks give the line genuine Q structure, unlike the
    # monotone DW-only case
    assert np.any(np.diff(got) > 0.0)


def test_map_elastic_broadening_spreads_the_line_and_keeps_its_integral():
    """broaden=True applies sigma(E=0) to the deposited line: the on/off
    difference spreads beyond the two seed bins while its per-Q energy integral
    is conserved by the (normalized) resolution kernel."""
    sp = _carbon(sigma_inc_b=1.2)
    m_off = compute_sqe_map(dos_species=[sp], elastic=False, broaden=True,
                            **GRID, **BASE)
    m_on = compute_sqe_map(dos_species=[sp], elastic=True,
                           elastic_kind="incoherent", broaden=True,
                           **GRID, **BASE)
    diff = m_on.S - m_off.S
    ref = from_dos_elastic(
        None, awr=[sp["awr"]], sigma_inc_b=[1.2], f0_lambda=[_f0(m_on)],
        multiplicity=[1], T_K=T_K)
    area = ref.incoherent_dsigma_dOmega(m_on.Q)
    got = diff.sum(axis=1) * GRID["dE"]
    assert np.allclose(got, area, rtol=2e-2)         # kernel conserves the area
    # ... but the line is no longer a two-bin spike
    thresh = 1e-6 * float(np.abs(diff).max())
    assert int((np.abs(diff) > thresh).sum(axis=1).min()) > 2


def test_map_elastic_skipped_when_E0_outside_axis():
    """A loss-only window that excludes E=0 cannot show the line: the map must
    equal the elastic-off map and say so, not crash or deposit at the edge."""
    sp = _carbon(sigma_inc_b=1.2)
    grid = dict(GRID, e_min=10.0, e_max=60.0)
    msgs = []
    m_off = compute_sqe_map(dos_species=[sp], elastic=False, broaden=False,
                            **grid, **BASE)
    m_on = compute_sqe_map(dos_species=[sp], elastic=True,
                           elastic_kind="incoherent", broaden=False,
                           **dict(grid, **{k: v for k, v in BASE.items()
                                           if k != "progress"}),
                           progress=msgs.append)
    assert np.array_equal(m_on.S, m_off.S)
    assert any("elastic line" in str(x) for x in msgs)
    # metadata honesty: a model was active but no line landed on this axis
    assert m_on.metadata["elastic"] is True
    assert m_on.metadata["elastic_deposited"] is False


def test_map_elastic_on_bin_interior_zero_exact():
    """When E=0 falls EXACTLY on an interior grid point (-15 + 10*1.5), the
    whole deposit lands in that single bin with an exact integral."""
    sp = _carbon(sigma_inc_b=1.2)
    grid = dict(GRID, e_min=-15.0, e_max=60.0)         # dE=1.5 -> 0.0 on-grid
    m_off = compute_sqe_map(dos_species=[sp], elastic=False, broaden=False,
                            **grid, **BASE)
    m_on = compute_sqe_map(dos_species=[sp], elastic=True,
                           elastic_kind="incoherent", broaden=False,
                           **grid, **BASE)
    assert m_on.metadata["elastic_deposited"] is True
    diff = m_on.S - m_off.S
    jz = int(np.flatnonzero(np.isclose(m_on.E, 0.0))[0])
    off_line = np.ones(m_on.E.size, bool)
    off_line[jz] = False
    assert np.all(diff[:, off_line] == 0.0)            # single-bin deposit
    ref = from_dos_elastic(
        None, awr=[sp["awr"]], sigma_inc_b=[1.2], f0_lambda=[_f0(m_on)],
        multiplicity=[1], T_K=T_K)
    area = ref.incoherent_dsigma_dOmega(m_on.Q)
    assert np.allclose(diff[:, jz] * grid["dE"], area, rtol=1e-9)


def test_map_elastic_axis_endpoint_zero_carries_half_line():
    """The default e_min=0 axis puts the line CENTER on the axis edge: only the
    on-axis half of the line is representable, so the broadened map carries
    half the elastic area under the kernel's trapezoidal quadrature -- and a
    NOTE says so. (Physical truncation, not a numerics loss: extend e_min<0
    for the full line.)"""
    sp = _carbon(sigma_inc_b=1.2)
    grid = dict(GRID, e_min=0.0, e_max=60.0)
    msgs = []
    m_off = compute_sqe_map(dos_species=[sp], elastic=False, broaden=True,
                            **grid, **BASE)
    m_on = compute_sqe_map(dos_species=[sp], elastic=True,
                           elastic_kind="incoherent", broaden=True,
                           **dict(grid, **{k: v for k, v in BASE.items()
                                           if k != "progress"}),
                           progress=msgs.append)
    assert m_on.metadata["elastic_deposited"] is True
    assert any("endpoint" in str(x) for x in msgs)
    diff = m_on.S - m_off.S
    ref = from_dos_elastic(
        None, awr=[sp["awr"]], sigma_inc_b=[1.2], f0_lambda=[_f0(m_on)],
        multiplicity=[1], T_K=T_K)
    area = ref.incoherent_dsigma_dOmega(m_on.Q)
    got = np.trapezoid(diff, m_on.E, axis=1)
    assert np.allclose(got, 0.5 * area, rtol=5e-2)     # the visible half


def test_map_explicit_elastic_model_deposits():
    """An explicit ElasticModel (the ENDF-tape / engine route) deposits on the
    map without any tape-free construction -- pins the elastic_model= pass-through
    with a synthetic engine elastic_state, no phonopy."""
    es = {"thermal_displacement_matrices_ang2": np.array([0.004 * np.eye(3)]),
          "primitive_lattice_ang": np.diag([3.567] * 3).astype(float),
          "primitive_scaled_positions": np.zeros((1, 3)),
          "primitive_symbols": ["C"],
          "primitive_masses_amu": np.array([12.0]),
          "temperature_k": T_K}
    model = from_engine_elastic_state(es, b_coh_fm=6.646, sigma_inc_b=0.001,
                                      awr=11.898, elastic_kind="both",
                                      emax_eV=0.3)
    m_off = compute_sqe_map(dos_species=[_carbon()], elastic=False,
                            broaden=False, **GRID, **BASE)
    m_on = compute_sqe_map(dos_species=[_carbon()], elastic_model=model,
                           broaden=False, **GRID, **BASE)
    diff = m_on.S - m_off.S
    got = diff.sum(axis=1) * GRID["dE"]
    area = model.elastic_dsigma_dOmega(m_on.Q, q_res=GRID["dQ_map"])
    assert np.allclose(got, area, rtol=1e-9)
    assert m_on.metadata["elastic"] is True
    assert m_on.metadata["n_bragg_edges"] == int(model.Q_bragg.size)


# ---- config funnel: run_map carries physics.elastic --------------------------
def _write_dos_file(tmp_path, name="c.dos", w_max_meV=40.0, n=121):
    rows = "".join(f"{w} {((w / w_max_meV) ** 2 if w <= w_max_meV else 0.0)}\n"
                   for w in range(0, n))
    p = tmp_path / name
    p.write_text("# freq_meV  dos\n" + rows)
    return str(p)


def test_run_map_mode0_elastic_from_config(tmp_path):
    """physics.elastic=true reaches the map through run_map (the GUI/CLI
    funnel): the elastic-on map carries the E=0 line the elastic-off map lacks.
    This is the regression pin for the silently-ignored switch."""
    from irma.spectra.config import SpectraConfig, run_map

    d = {"material": {"temperature_K": T_K,
                      "scatterers": [{"symbol": "C",
                                      "dos_file": _write_dos_file(tmp_path),
                                      "dos_unit": "meV", "awr": 11.898,
                                      "sigma_bound_b": 5.551,
                                      "sigma_inc_b": 1.2}]},
         "physics": {"inelastic_mode": 0, "elastic": True,
                     "elastic_kind": "incoherent", "max_phonon_order": 40},
         "grid": {"e_min_meV": -20.0, "e_max_meV": 60.0, "de_meV": 1.5,
                  "dq_max_invA": 0.1},
         "instrument": {"geometry": "direct", "e_fixed_meV": 250.0,
                        "angles_deg": [30.0, 120.0]}}
    cfg_on = SpectraConfig.from_dict(d)
    import copy
    d_off = copy.deepcopy(d)
    d_off["physics"]["elastic"] = False
    cfg_off = SpectraConfig.from_dict(d_off)

    silent = lambda *a, **k: None
    m_on = run_map(cfg_on, q_min=1.0, q_max=6.0, dQ_map=0.5, broaden=False,
                   progress=silent)
    m_off = run_map(cfg_off, q_min=1.0, q_max=6.0, dQ_map=0.5, broaden=False,
                    progress=silent)
    assert m_on.metadata["elastic"] is True
    assert m_off.metadata["elastic"] is False
    diff = m_on.S - m_off.S
    j0, j1 = _zero_bins(m_on.E)
    off_line = np.ones(m_on.E.size, bool)
    off_line[[j0, j1]] = False
    assert np.all(diff[:, off_line] == 0.0)
    assert diff[:, [j0, j1]].sum(axis=1).min() > 0.0   # the line is present at every Q
