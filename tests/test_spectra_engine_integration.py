"""Live engine-backed coverage for the ``compute_spectrum`` mode-1/2 path.

Every other ``compute_spectrum`` test drives the cheap mode-0 DOS branch
(``dos_species=...``), so the PRODUCTION spectra path -- the engine block of
``irma.spectra.forward`` -- ran in CI exactly never: ``_get_engine_context`` +
``run_noncubic_sab_inprocess`` on the locus grid, ``_pick_sqe_key`` against
live engine output, the S-orientation seam, and the tape-free elastic glue
built from the engine's surfaced ``elastic_state``.

The live test here mirrors the fast-CI noncubic gauge (vendored graphite
phonopy model, mesh 4^3, ndir=40, mpdir=20) through the FORWARD-MODEL entry
point instead of ``run_leapr``, with elastic=True so the engine
``elastic_state`` -> ``from_engine_elastic_state`` glue runs too, and pins:

  * the inelastic and elastic spectrum integrals to frozen values
    (characterization pins -- blessed from two byte-identical runs in
    separate processes, same determinism contract as the fast-CI tape pins),
  * the live ``_pick_sqe_key`` selection recorded in metadata,
  * the elastic line being populated AND concentrated at E=0,
  * the I_total = I_inelastic + I_elastic decomposition.

The two ``ValueError`` guards of the elastic glue (elastic=True without
``elastic_scatterers``; ``elastic_scatterers`` missing a primitive species)
are covered by cheap stubbed-engine tests -- no phonopy work needed there.
"""
import os

import numpy as np
import pytest

pytest.importorskip("phonopy")

import irma.core.noncubic_inelastic as nci  # noqa: E402
import irma.spectra.forward as fwd  # noqa: E402
from irma.spectra.forward import compute_spectrum  # noqa: E402

_YAML = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "mode2_euphonic_n1_validation", "graphite",
    "phonopy.yaml"))

# Shared by the live test; values mirror the fast-CI gauge. nphon=10 is the
# smallest order above the engine's own anisotropic-DW convergence floor (~9
# at this Q_max), so the run stays warning-free AND cheap (~5 s).
_LIVE = dict(
    geometry="vision", phonopy_yaml=_YAML, temperature_k=296.0,
    mesh=(4, 4, 4), sab_mass_ratio=11.898, sab_sigma_barn=5.551,
    inelastic_mode=1, num_directions=40, multiphonon_num_directions=20,
    multiphonon_max_order=10, auto_multiphonon_order=False,
    dE=2.0, dQ=0.25, e_max=120.0, jobs=2,
    elastic=True, elastic_kind="both",
    elastic_scatterers={"C": {"b_coh_fm": 6.6460, "sigma_inc_b": 0.001,
                              "awr": 11.898}},
    progress=lambda *a, **k: None)

# Frozen characterization pins (trapezoid integrals over the output E axis).
# Determinism: two runs in SEPARATE processes reproduced every digit below
# (the engine's ordered 2-worker pool + pinned BLAS threads, as in the
# fast-CI tape pins) — but only on ONE platform. Across platforms the spectra
# layer (scipy interpolation/broadening, BLAS) jitters at the ~1e-5 level:
# the linux CI runner reproduced the engine-side law pins (rel 1e-6, see
# test_noncubic_fast_ci) yet differed here by 9.4e-6 relative. The pin
# tolerance is therefore 1e-4 — loose enough for platform jitter, tight
# enough that any real orchestration regression (wrong sqe key, dropped
# component, broken normalization) still fails.
_I_INELASTIC = 2.891399038353e-02
_I_ELASTIC = 7.831436600712e-05


def test_compute_spectrum_live_engine_mode1_with_elastic():
    r = compute_spectrum(**_LIVE)
    assert np.all(np.isfinite(r.I_total))
    assert np.allclose(r.I_total, r.I_inelastic + r.I_elastic,
                       rtol=0, atol=1e-12)
    # live _pick_sqe_key selection: mode 1 with multiphonon order >= 2
    assert r.metadata["sqe_key"] == ("sqe_incoherent_approx_n1_term_plus_"
                                     "incoherent_approx_multiphonon_barn_per_meV")
    # frozen pins on the engine-backed spectrum (see module docstring)
    assert np.trapezoid(r.I_inelastic, r.E) == pytest.approx(
        _I_INELASTIC, rel=1.0e-4)
    assert np.trapezoid(r.I_elastic, r.E) == pytest.approx(
        _I_ELASTIC, rel=1.0e-4)
    # the tape-free elastic line is real AND lives at the elastic position
    assert r.metadata["elastic"] is True
    assert r.metadata["n_bragg_edges"] > 0
    near0 = np.abs(r.E) <= 4.0
    assert r.I_elastic[near0].max() > 0.0
    assert r.I_elastic[~near0].max() < 1e-30


# --- elastic-glue ValueError guards (stubbed engine; no phonopy work) --------

@pytest.fixture
def stubbed_engine(monkeypatch):
    """Replace the context builder + engine driver so compute_spectrum reaches
    the elastic glue instantly. The stub echoes the locus grids back as the
    engine's output axes and surfaces a two-species elastic_state."""
    calls = {"inprocess_kwargs": []}

    def fake_context(**kwargs):
        return {"stub": True}

    def fake_run(**kwargs):
        calls["inprocess_kwargs"].append(kwargs)
        q = np.asarray(kwargs["q_grid_ang_inv"], float)
        E = np.asarray(kwargs["e_grid_mev"], float)
        return {
            "output_arrays": {
                "q_ang_inv": q, "e_mev": E,
                "sqe_incoherent_approx_n1_term_barn_per_meV":
                    np.zeros((q.size, E.size)),
            },
            "metadata": {"multiphonon_max_order": 1},
            "elastic_state": {"primitive_symbols": ["C", "H"]},
        }

    monkeypatch.setattr(fwd, "_get_engine_context", fake_context)
    monkeypatch.setattr(nci, "run_noncubic_sab_inprocess", fake_run)
    return calls


def _stub_call(**extra):
    base = dict(geometry="vision", phonopy_yaml="phonopy.yaml",
                temperature_k=296.0, mesh=(2, 2, 2), sab_mass_ratio=11.898,
                sab_sigma_barn=5.551, dE=10.0, dQ=1.0, e_max=50.0,
                elastic=True, progress=lambda *a, **k: None)
    base.update(extra)
    return compute_spectrum(**base)


def test_elastic_true_requires_scatterers(stubbed_engine):
    with pytest.raises(ValueError, match="requires.*elastic_scatterers"):
        _stub_call(elastic_scatterers=None)
    # the guard fires AFTER the engine ran, on its surfaced elastic_state
    assert len(stubbed_engine["inprocess_kwargs"]) == 1


def test_elastic_scatterers_must_cover_every_species(stubbed_engine):
    with pytest.raises(ValueError, match="missing entries for"):
        _stub_call(elastic_scatterers={"C": {"b_coh_fm": 6.646,
                                             "sigma_inc_b": 0.001,
                                             "awr": 11.898}})
    assert len(stubbed_engine["inprocess_kwargs"]) == 1
