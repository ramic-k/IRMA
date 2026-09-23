"""Range validation of the spectra config numerics and the run_map kwargs.

Every reject goes through the real from_dict/validate path (or run_map with
the engine stubbed -- no phonons are computed).
"""
import pytest

from irma.spectra.config import (
    SpectraConfig, SpectraConfigError, run_map, validate,
)


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


@pytest.mark.parametrize("bad", [[-1.0], [0.0], [2.0, -3.0]])
def test_bad_q_cuts_rejected(bad):
    with pytest.raises(SpectraConfigError, match="q_cuts"):
        validate(_cfg(instrument={"q_cuts": bad}))


@pytest.mark.parametrize("field,bad", [
    ("awr", -1.0), ("awr", 0.0),
    ("sigma_bound_b", -5.0), ("sigma_bound_b", 0.0), ("sigma_inc_b", -0.5),
])
def test_bad_scatterer_numeric_rejected(field, bad):
    with pytest.raises(SpectraConfigError, match=rf"scatterer 'C'.*{field}"):
        validate(_cfg(scatterer={field: bad}))


def test_negative_b_coh_and_zero_sigma_inc_accepted():
    # b_coh is a signed scattering length; sigma_inc_b = 0 means no channel
    cfg = _cfg(scatterer={"b_coh_fm": -3.739, "sigma_inc_b": 0.0})
    assert validate(cfg) is cfg


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_bad_bank_halfwidth_rejected(bad):
    with pytest.raises(SpectraConfigError, match="bank_halfwidth_deg"):
        validate(_cfg(instrument={"bank_halfwidth_deg": bad}))


# ---- run_map kwargs ----------------------------------------------------------
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
    {"q_min": -0.5}, {"dQ_map": 0.0}, {"angle_range": (120.0, 30.0)}])
def test_run_map_bad_kwargs_rejected_before_engine(monkeypatch, kwargs):
    called = _stub_engine(monkeypatch)
    with pytest.raises(SpectraConfigError, match=next(iter(kwargs))):
        run_map(_cfg(), **kwargs)
    assert not called, "engine was invoked despite the invalid kwarg"


def test_run_map_config_qmax_below_qmin_rejected(monkeypatch):
    called = _stub_engine(monkeypatch)
    cfg = _cfg()
    cfg.grid.q_max_invA = 0.3
    with pytest.raises(SpectraConfigError, match="q_max"):
        run_map(cfg, q_min=0.5)
    assert not called


def test_run_map_valid_kwargs_reach_engine(monkeypatch):
    called = _stub_engine(monkeypatch)
    out = run_map(_cfg(), q_min=0.5, q_max=10.0, dQ_map=0.1,
                  angle_range=(30.0, 120.0), progress=lambda *a, **k: None)
    assert out == {"stub": True}
    kw = called["kw"]
    assert kw["q_min"] == 0.5 and kw["q_max"] == 10.0 and kw["dQ_map"] == 0.1
    assert kw["angle_range_deg"] == (30.0, 120.0)


# ---- count-like fields ------------------------------------------------------
def test_count_fields_accept_integral_floats():
    d = {"material": {"phonopy_yaml": "graphite.yaml", "mesh": [4.0, 4, 4],
                      "temperature_K": 296.0,
                      "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                                      "awr": 11.898}]},
         "physics": {"n_directions": 4000.0, "jobs": 4.0}}
    cfg = validate(SpectraConfig.from_dict(d))
    assert cfg.physics.n_directions == 4000 and cfg.physics.jobs == 4
    assert cfg.material.mesh == [4, 4, 4]
