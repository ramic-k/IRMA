"""Direct energy-gain side for the eigenvector engine (modes 1/2).

The engine computes the E<0 side with explicit Bose ANNIHILATION factors
(n(omega) at -omega for the one-phonon coherent + incoherent terms; the
negative half of the signed multiphonon convolution) under the opt-in
``emit_gain_side`` flag -- never a detailed-balance mirror.

Two contracts are pinned here against a real (small) graphite phonopy model:

1. LOSS BYTE-IDENTITY (the ENDF byte-exact guard): every loss
   ``sqe_*_barn_per_meV`` array is BITWISE identical whether ``emit_gain_side``
   is off or on. The gain path is purely additive and never perturbs a loss
   float, so the SAB/tape outputs are unaffected.

2. GAIN == MIRROR TO O(dE/kT), CONVERGING. For the equilibrium harmonic model
   detailed balance is exact in the continuum, so the directly-computed gain
   equals the detailed-balance mirror of the loss side up to the discrete-grid
   difference between depositing a line at its TRUE energy -omega (direct) and
   reflecting the loss BIN whose center is offset by up to dE/2 (mirror). That
   sub-bin difference is ~dE/kT and must SHRINK ~linearly as dE -> 0 -- which
   both proves the gain side is correct and shows the direct method is the more
   accurate one (no bin-center approximation).

Phonopy-gated and kept small (mesh 4^3, ndir 40, mpdir 20) -- a few seconds.
"""
import os

import numpy as np
import pytest

pytest.importorskip("phonopy")

from irma.core.noncubic_inelastic import run_noncubic_sab_inprocess  # noqa: E402
from irma.spectra.forward import compute_sqe_map  # noqa: E402

_YAML = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "mode2_euphonic_n1_validation", "graphite",
    "phonopy.yaml"))
_KT = 0.0861733326 * 296.0


def _engine(mode, emit, e_grid, q_grid=None):
    q_grid = np.arange(1.0, 10.01, 0.5) if q_grid is None else q_grid
    import contextlib
    import io
    # jobs=1 (no worker pool): these are correctness pins, not parallelism, and
    # every pooled run pays the spawn start-up. The pool path is exercised by
    # test_spectra_engine_integration.py.
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_noncubic_sab_inprocess(
            inelastic_mode=mode, emit_gain_side=emit, phonopy_yaml=_YAML,
            temperature_k=296.0, mesh=(4, 4, 4), q_grid_ang_inv=q_grid,
            e_grid_mev=e_grid, output_prefix="/tmp/_gain_test_unused",
            sab_mass_ratio=11.898, sab_sigma_barn=5.551,
            num_directions=40, multiphonon_num_directions=20,
            multiphonon_max_order=10, auto_multiphonon_order=False, jobs=1,
            write_output_files=False)
    return res["output_arrays"]


def _loss_keys(out):
    return [k for k in out if k.startswith("sqe_")
            and k.endswith("_barn_per_meV") and "_gain_" not in k]


@pytest.mark.parametrize("mode", [1, 2])
def test_loss_byte_identical_emit_gain_off_vs_on(mode):
    """The gain side is purely additive: every loss array is BITWISE identical
    with emit_gain_side off vs on. This is the ENDF byte-exact guard."""
    E = np.arange(0.0, 120.01, 2.0)
    off = _engine(mode, False, E)
    on = _engine(mode, True, E)
    keys = _loss_keys(off)
    assert keys, "expected loss sqe_* arrays"
    for k in keys:
        assert k in on
        assert np.array_equal(off[k], on[k]), f"loss array {k} not bit-identical"
    # the gain run additionally surfaces the gain grid + gain arrays
    assert any(k.endswith("_gain_barn_per_meV") for k in on)
    Eg = np.asarray(on["e_gain_mev"], float)
    Epos = on["e_mev"][on["e_mev"] > 0]
    assert Eg.shape == Epos.shape
    assert np.allclose(Eg, -Epos[::-1])              # strictly-negative mirror
    # every surfaced gain array has the gain-grid energy length and is finite/>=0
    for k in on:
        if k.endswith("_gain_barn_per_meV"):
            g = np.asarray(on[k], float)
            assert g.shape[1] == Eg.size
            assert np.all(np.isfinite(g)) and np.all(g >= -1e-12)


def _gain_vs_mirror_max(out, base_key):
    loss = out[base_key + "_barn_per_meV"]
    gain = out[base_key + "_gain_barn_per_meV"]
    Eloss = out["e_mev"]
    boltz = np.exp(-Eloss[Eloss > 0] / _KT)
    mirror = (loss[:, Eloss > 0] * boltz[None, :])[:, ::-1]
    msk = np.abs(mirror) > 1e-10 * np.abs(mirror).max()
    return float(np.max(np.abs(gain[msk] / mirror[msk] - 1.0)))


def test_gain_matches_mirror_and_converges_with_grid():
    """Direct gain == detailed-balance mirror to O(dE/kT), and the agreement
    TIGHTENS ~linearly as dE halves -- the signature of the sub-bin
    (line-energy vs bin-center) difference, proving the direct gain is correct
    and more accurate than the mirror."""
    errs = {}
    for dE in (2.0, 1.0, 0.5):
        E = np.arange(0.0, 120.0 + 0.5 * dE, dE)
        on = _engine(1, True, E)
        errs[dE] = _gain_vs_mirror_max(on, "sqe_incoherent_approx_n1_term")
    # absolute scale: ~dE/kT per meV (dE=2 -> a few percent), never wild
    assert errs[2.0] < 0.08
    # convergence: halving dE roughly halves the max error (allow slack)
    assert errs[1.0] < 0.65 * errs[2.0]
    assert errs[0.5] < 0.65 * errs[1.0]


@pytest.mark.parametrize("mode", [1, 2])
def test_bridge_attaches_engine_gain_direct(mode):
    """Through the forward model: gain_side='direct' on modes 1/2 attaches the
    engine gain side (metadata gain_side_used='direct') and yields a finite,
    Boltzmann-suppressed energy-gain wing -- no silent mirror, no fallback."""
    common = dict(geometry="direct", e_fixed_meV=250.0,
                  phonopy_yaml=_YAML, temperature_k=296.0, mesh=(4, 4, 4),
                  sab_mass_ratio=11.898, sab_sigma_barn=5.551,
                  inelastic_mode=mode, num_directions=40,
                  multiphonon_num_directions=20, multiphonon_max_order=10,
                  auto_multiphonon_order=False, jobs=1, elastic=False,
                  q_min=1.0, q_max=9.0, dQ_map=0.5, e_min=-60.0, e_max=120.0,
                  dE=2.0, broaden=False, progress=lambda *a, **k: None)
    m_dir = compute_sqe_map(gain_side="direct", **common)
    m_db = compute_sqe_map(gain_side="detailed_balance", **common)
    assert m_dir.metadata["gain_side"] == "direct"
    assert m_dir.metadata["gain_side_used"] == "direct"      # engine computed it
    assert m_db.metadata["gain_side_used"] == "detailed_balance"
    gn = m_dir.E < -2.0
    assert gn.any()
    # the gain wing is present, finite and Boltzmann-suppressed vs the loss side
    g_dir = np.nan_to_num(m_dir.S[:, gn])
    assert np.all(np.isfinite(g_dir)) and g_dir.max() > 0.0
    lo = m_dir.E > 2.0
    assert np.nan_to_num(m_dir.S[:, lo]).sum() > g_dir.sum()
    # direct and mirror agree on the gain side to the sub-bin O(dE/kT) level
    db = np.nan_to_num(m_db.S[:, gn])
    msk = db > 1e-3 * db.max()
    assert np.max(np.abs(g_dir[msk] / db[msk] - 1.0)) < 0.15
