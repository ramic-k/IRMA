"""Instrument-reach Bragg cutoff: the reach algebra, truncation as a
bit-identical prefix of the engine builder, and byte-identical elastic
output on the committed graphite mode-0 example against a full 5 eV run.
"""
import numpy as np
import pytest
from pathlib import Path

from irma.core.constants import HBAR2_OVER_2MN_MEV_A2 as C_E
from irma.spectra.elastic import (
    instrument_reach_emax_eV, from_engine_elastic_state)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "spectra"


def _E_eV_of_Q(Q):
    """Bragg-edge energy [eV] at momentum transfer Q [1/A] (Q = 2 sqrt(E/C_E))."""
    return C_E * (0.5 * Q) ** 2 * 1.0e-3


# ---- the reach algebra -------------------------------------------------------
def test_helper_never_exceeds_the_old_5ev_default():
    assert instrument_reach_emax_eV(q_max_invA=500.0) == 5.0
    assert instrument_reach_emax_eV(q_max_invA=1.0, e_fixed_meV=1.0e6) == 5.0


def test_helper_covers_the_simulated_q_support():
    """Every edge on the simulated Q grid must be enumerable: emax >= E(q_max)."""
    for q_max in (2.0, 8.0, 12.6, 30.0):
        emax = instrument_reach_emax_eV(q_max_invA=q_max)
        assert emax >= _E_eV_of_Q(q_max)


def test_helper_covers_the_bank_elastic_ring():
    """bank_elastic_area windows reach Q <= 2 k_el, i.e. E_edge <= E_fixed: the
    cutoff must be at least E_fixed even when the Q-support is smaller."""
    for e_fixed in (3.5, 250.0, 1000.0):
        emax = instrument_reach_emax_eV(q_max_invA=0.1, e_fixed_meV=e_fixed)
        assert emax >= e_fixed * 1.0e-3                  # meV -> eV


def test_helper_extends_for_q_cuts_with_gaussian_tail_headroom():
    """A constant-Q cut at Q0 +/- band evaluates the q_res-broadened Bragg peaks; the
    cutoff must reach beyond Q0 + band by the full exp-underflow tail (40
    sigma), so dropped peaks contribute EXACT zeros."""
    base = instrument_reach_emax_eV(q_max_invA=5.0)
    cut = instrument_reach_emax_eV(q_max_invA=5.0, q_cuts=[9.0], cut_dq=0.5,
                                   q_res_invA=0.1)
    assert cut > base
    assert cut >= _E_eV_of_Q(9.0 + 0.5 + 40.0 * 0.1)


# ---- truncation purity: smaller emax == bit-identical prefix ------------------
def _engine_state(a=3.567):
    return {
        "thermal_displacement_matrices_ang2": np.array([0.004 * np.eye(3)]),
        "primitive_lattice_ang": np.diag([a, a, a]).astype(float),
        "primitive_scaled_positions": np.array([[0.0, 0.0, 0.0]]),
        "primitive_symbols": ["C"],
        "temperature_k": 296.0,
    }


def test_engine_builder_truncation_is_a_bit_identical_prefix():
    kw = dict(b_coh_fm=6.646, sigma_inc_b=0.001, awr=11.898,
              elastic_kind="coherent")
    full = from_engine_elastic_state(_engine_state(), emax_eV=0.5, **kw)
    trunc = from_engine_elastic_state(_engine_state(), emax_eV=0.12, **kw)
    n = trunc.Q_bragg.size
    assert 0 < n < full.Q_bragg.size                     # genuinely truncated
    assert np.array_equal(trunc.Q_bragg, full.Q_bragg[:n])
    assert np.array_equal(trunc.f_bragg, full.f_bragg[:n])         # DW included
    assert np.array_equal(trunc.E_edge_meV, full.E_edge_meV[:n])
    assert float(trunc.E_edge_meV.max()) <= 0.12 / 1.0e-3          # <= emax


# ---- the load-bearing proof: committed example, reach-derived vs full 5 eV ----
def test_committed_example_spectrum_identical_to_full_5ev_enumeration(monkeypatch):
    """End-to-end on examples/spectra/graphite_mode0_dosfile.yaml (plus two
    constant-Q cuts to exercise the q_res-broadened peak path): forcing the old
    full-5 eV enumeration changes NOTHING in the elastic output -- the dropped
    edges are beyond the instrument's reach by construction. Equality is
    asserted bitwise (tolerance 0)."""
    yaml = pytest.importorskip("yaml")  # noqa: F841  (example config is YAML)
    from irma.spectra.config import load, _assemble_dos_species
    import irma.spectra.elastic as el
    from irma.spectra.forward import compute_spectrum

    cfg = load(EXAMPLES / "graphite_mode0_dosfile.yaml", validate_cfg=False)
    mat = cfg.material
    for s in mat.scatterers:                       # dos_file is example-relative
        s.dos_file = str(EXAMPLES / s.dos_file)
    sp = _assemble_dos_species(mat)

    common = dict(
        geometry="vision", phonopy_yaml=None, mesh=None,
        temperature_k=float(mat.temperature_K),
        sab_mass_ratio=float(mat.scatterers[0].awr),
        sab_sigma_barn=float(mat.scatterers[0].sigma_bound_b),
        e_fixed_meV=float(cfg.instrument.e_fixed_meV),
        dE=float(cfg.grid.de_meV), dQ=float(cfg.grid.dq_max_invA),
        e_min=float(cfg.grid.e_min_meV), e_max=float(cfg.grid.e_max_meV),
        dos_species=sp, dos_crystal=tuple(float(x) for x in mat.lattice),
        elastic=True, elastic_kind="both",
        q_cuts=[2.0, 5.0], cut_dq=0.2,
        progress=lambda *a, **k: None)

    r_reach = compute_spectrum(**common)
    monkeypatch.setattr(el, "instrument_reach_emax_eV", lambda **kw: 5.0)
    r_full = compute_spectrum(**common)

    # the reach-derived run really enumerates far fewer edges...
    assert 0 < r_reach.metadata["n_bragg_edges"] < r_full.metadata["n_bragg_edges"]
    # ...and the elastic output is BYTE-IDENTICAL everywhere it is evaluated
    np.testing.assert_array_equal(r_reach.I_elastic, r_full.I_elastic)
    np.testing.assert_array_equal(r_reach.I_elastic_per_angle,
                                  r_full.I_elastic_per_angle)
    np.testing.assert_array_equal(r_reach.I_elastic_per_q, r_full.I_elastic_per_q)
    np.testing.assert_array_equal(r_reach.I_total, r_full.I_total)
