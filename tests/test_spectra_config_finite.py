"""S7 -- finite/range validation for fields the whitelist previously missed.

Before this fix, NaN/Inf (and some negatives) slipped past validate() through
four holes: instrument.q_cuts was only float-cast in from_dict; the scatterer
numerics (awr / sigma_bound_b / b_coh_fm / sigma_inc_b) were only None-checked;
instrument.bank_halfwidth_deg's bare `> 0` passed +Inf; and run_map's own
kwargs (q_min / q_max / dQ_map / angle_range) bypassed the config schema
entirely. Every reject goes through the real from_dict/validate path (or
run_map with the engine stubbed -- no phonons are ever computed), and each
field has an accept-control proving valid values still pass.
"""
import pytest

from irma.spectra.config import (
    SpectraConfig, SpectraConfigError, run_map, validate,
)

NAN = float("nan")
INF = float("inf")


def _cfg(scatterer=None, instrument=None):
    """A valid mode-2 vision config built through the real from_dict path."""
    scat = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
            "b_coh_fm": 6.646, "sigma_inc_b": 0.001}
    scat.update(scatterer or {})
    ins = {"geometry": "vision", "e_fixed_meV": 3.5,
           "angles_deg": [45.0, 135.0]}
    ins.update(instrument or {})
    return SpectraConfig.from_dict({
        "material": {
            "phonopy_yaml": "graphite.yaml", "mesh": [40, 40, 40],
            "temperature_K": 296.0, "scatterers": [scat],
        },
        "physics": {"inelastic_mode": 2},
        "grid": {"e_min_meV": 0.0, "e_max_meV": 250.0, "de_meV": 0.5,
                 "dq_max_invA": 0.05},
        "instrument": ins,
    })


# ---- instrument.q_cuts (was only float-cast, never range-checked) -----------
@pytest.mark.parametrize("bad", [
    [-1.0], [0.0], [NAN], [INF], [-INF], [2.0, NAN], [2.0, -3.0]])
def test_bad_q_cuts_rejected(bad):
    cfg = _cfg(instrument={"q_cuts": bad})
    with pytest.raises(SpectraConfigError, match="q_cuts"):
        validate(cfg)


def test_valid_q_cuts_accepted():
    cfg = _cfg(instrument={"q_cuts": [2.0, 5.0]})
    assert validate(cfg) is cfg


# ---- scatterer numerics (were only None-checked) -----------------------------
@pytest.mark.parametrize("field,bad", [
    ("awr", NAN), ("awr", INF), ("awr", -1.0), ("awr", 0.0),
    ("sigma_bound_b", NAN), ("sigma_bound_b", INF),
    ("sigma_bound_b", -5.0), ("sigma_bound_b", 0.0),
    ("b_coh_fm", NAN), ("b_coh_fm", INF), ("b_coh_fm", -INF),
    ("sigma_inc_b", NAN), ("sigma_inc_b", INF), ("sigma_inc_b", -0.5),
])
def test_bad_scatterer_numeric_rejected(field, bad):
    cfg = _cfg(scatterer={field: bad})
    with pytest.raises(SpectraConfigError, match=field):
        validate(cfg)


def test_scatterer_error_names_species_and_field():
    cfg = _cfg(scatterer={"awr": NAN})
    with pytest.raises(SpectraConfigError, match=r"scatterer 'C'.*awr"):
        validate(cfg)


def test_negative_finite_b_coh_fm_accepted():
    # b_coh is a SIGNED scattering length (H, Ti, Mn are negative)
    cfg = _cfg(scatterer={"b_coh_fm": -3.739})
    assert validate(cfg) is cfg


def test_zero_sigma_inc_b_accepted():
    # 0.0 deliberately declares "no incoherent channel"
    cfg = _cfg(scatterer={"sigma_inc_b": 0.0})
    assert validate(cfg) is cfg


def test_scatterer_checks_apply_in_mode0_too():
    cfg = SpectraConfig.from_dict({
        "material": {
            "temperature_K": 296.0,
            "scatterers": [{"symbol": "C", "sigma_bound_b": INF, "awr": 11.898,
                            "sigma_inc_b": 0.001, "dos_file": "c.dat"}],
        },
        "physics": {"inelastic_mode": 0, "elastic": False},
    })
    with pytest.raises(SpectraConfigError, match="sigma_bound_b"):
        validate(cfg)


# ---- instrument.bank_halfwidth_deg / cut_dq_invA / chopper frequency ---------
@pytest.mark.parametrize("bad", [NAN, INF, -INF, 0.0, -1.0])
def test_bad_bank_halfwidth_rejected(bad):
    cfg = _cfg(instrument={"bank_halfwidth_deg": bad})
    with pytest.raises(SpectraConfigError, match="bank_halfwidth_deg"):
        validate(cfg)


@pytest.mark.parametrize("bad", [NAN, INF])
def test_nonfinite_cut_dq_rejected(bad):
    cfg = _cfg(instrument={"cut_dq_invA": bad})
    with pytest.raises(SpectraConfigError, match="cut_dq_invA"):
        validate(cfg)


def test_baseline_config_accepted():
    cfg = _cfg()
    assert validate(cfg) is cfg


# ---- physics sampling counts (Inf/NaN passed the old `<= 0` guards) ----------
@pytest.mark.parametrize("field,bad", [
    ("n_directions", NAN), ("n_directions", INF),
    ("multiphonon_directions", NAN), ("multiphonon_directions", INF),
    ("jobs", NAN), ("jobs", INF),
])
def test_nonfinite_physics_counts_rejected(field, bad):
    cfg = _cfg()
    setattr(cfg.physics, field, bad)
    with pytest.raises(SpectraConfigError):
        validate(cfg)


# ---- run_map kwargs (bypass the config schema entirely) ----------------------
def _stub_engine(monkeypatch):
    """Replace compute_sqe_map; record whether/with what it was called."""
    import irma.spectra.forward as fwd
    called = {}

    def fake_compute_sqe_map(**kw):
        called["kw"] = kw
        return {"stub": True}

    monkeypatch.setattr(fwd, "compute_sqe_map", fake_compute_sqe_map)
    return called


@pytest.mark.parametrize("kwargs", [
    {"q_min": NAN}, {"q_min": INF}, {"q_min": -0.5},
    {"q_max": NAN}, {"q_max": INF}, {"q_max": 0.0},   # 0.0 <= default q_min 0.0
    {"dQ_map": NAN}, {"dQ_map": INF}, {"dQ_map": 0.0}, {"dQ_map": -0.1},
    {"angle_range": (NAN, 90.0)}, {"angle_range": (30.0, INF)},
    {"angle_range": (120.0, 30.0)}, {"angle_range": (30.0,)},
    # review SP-3: the kwarg route accepted nonphysical two-theta ranges the
    # config route rejects; both now enforce 0 < min < max < 180.
    {"angle_range": (-10.0, 200.0)}, {"angle_range": (0.0, 90.0)},
    {"angle_range": (30.0, 180.0)}, {"angle_range": (30.0, 250.0)},
])
def test_run_map_bad_kwargs_rejected_before_engine(monkeypatch, kwargs):
    called = _stub_engine(monkeypatch)
    cfg = _cfg()
    kwarg_name = next(iter(kwargs))
    with pytest.raises(SpectraConfigError, match=kwarg_name):
        run_map(cfg, **kwargs)
    assert not called, "engine was invoked despite the invalid kwarg"


def test_run_map_config_qmax_below_qmin_rejected(monkeypatch):
    # q_max=None defers to grid.q_max_invA, which is finite/positive per the
    # schema but can still undershoot the kwarg q_min -- reject the RESOLVED pair.
    called = _stub_engine(monkeypatch)
    cfg = _cfg()
    cfg.grid.q_max_invA = 0.3
    with pytest.raises(SpectraConfigError, match="q_max"):
        run_map(cfg, q_min=0.5)
    assert not called


def test_run_map_valid_kwargs_reach_engine(monkeypatch):
    called = _stub_engine(monkeypatch)
    cfg = _cfg()
    out = run_map(cfg, q_min=0.5, q_max=10.0, dQ_map=0.1,
                  angle_range=(30.0, 120.0), progress=lambda *a, **k: None)
    assert out == {"stub": True}
    kw = called["kw"]
    assert kw["q_min"] == 0.5 and kw["q_max"] == 10.0 and kw["dQ_map"] == 0.1
    assert kw["angle_range_deg"] == (30.0, 120.0)


def test_run_map_defaults_reach_engine(monkeypatch):
    # q_min=0 (include the elastic column) and the config-derived q_max are legal
    called = _stub_engine(monkeypatch)
    cfg = _cfg()
    out = run_map(cfg, q_min=0.0, progress=lambda *a, **k: None)
    assert out == {"stub": True}
    # grid.q_max_invA unset -> auto-cover the kinematic envelope, capped 40
    import numpy as np
    from irma.spectra.forward import kinematic_envelope
    g, ins = cfg.grid, cfg.instrument
    Eg = np.arange(g.e_min_meV, g.e_max_meV + 0.5 * g.de_meV, g.de_meV)
    cov = ins.map_coverage_deg or ins.angles_deg or [45.0, 135.0]
    _, q_hi = kinematic_envelope(ins.geometry, ins.e_fixed_meV,
                                 min(cov), max(cov), Eg)
    finite_q = q_hi[np.isfinite(q_hi)]
    expected = min(float(finite_q.max()) + 0.5, 40.0)
    assert called["kw"]["q_max"] == pytest.approx(expected)
    assert called["kw"]["dQ_map"] == g.dq_max_invA   # dQ defers to the config


# ---- review SP-1: count-like fields demand exact integers -------------------

@pytest.mark.parametrize("physics,field", [
    ({"max_phonon_order": True}, "max_phonon_order"),
    ({"n_directions": 3.9}, "n_directions"),
    ({"multiphonon_directions": 2.7}, "multiphonon_directions"),
    ({"jobs": 0.9}, "jobs"),
    ({"jobs": True}, "jobs"),
])
def test_count_fields_reject_bool_and_fractional(physics, field):
    scat = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
            "b_coh_fm": 6.646, "sigma_inc_b": 0.001}
    d = {"material": {"phonopy_yaml": "graphite.yaml", "mesh": [4, 4, 4],
                      "temperature_K": 296.0, "scatterers": [scat]},
         "physics": physics}
    with pytest.raises(SpectraConfigError, match=field):
        validate(SpectraConfig.from_dict(d))


def test_mesh_rejects_fractional_entries():
    d = {"material": {"phonopy_yaml": "graphite.yaml", "mesh": [4.9, 4, 4],
                      "temperature_K": 296.0}}
    with pytest.raises(SpectraConfigError, match="mesh"):
        SpectraConfig.from_dict(d)


def test_count_fields_accept_integral_floats():
    scat = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
            "b_coh_fm": 6.646, "sigma_inc_b": 0.001}
    d = {"material": {"phonopy_yaml": "graphite.yaml", "mesh": [4.0, 4, 4],
                      "temperature_K": 296.0, "scatterers": [scat]},
         "physics": {"n_directions": 4000.0, "jobs": 4.0}}
    cfg = SpectraConfig.from_dict(d)
    validate(cfg)
    assert cfg.physics.n_directions == 4000 and cfg.physics.jobs == 4
    assert cfg.material.mesh == [4, 4, 4]
