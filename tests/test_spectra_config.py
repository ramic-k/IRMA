"""irma.spectra.config -- schema round-trip + validator, no engine/phonopy.

Pins the input contract: (1) load(dump(cfg)) == cfg for every accepted format
(YAML/JSON), proving dump/from_dict are exact inverses with normalized types;
(2) the validator rejects each documented inconsistency (geometry/Ei conflict,
bad spacings, missing required fields, non-positive resolution sigma, unknown
keys, mistyped booleans); (3) run_spectra forwards the config to
compute_spectrum faithfully. Fast, data-free, CI-safe.
"""
import pytest

from irma.spectra.config import (
    SpectraConfig, PhysicsConfig, GridConfig, InstrumentConfig,
    SpectraConfigError, load, dump, validate,
)


def _vision_cfg():
    return SpectraConfig.from_dict({
        "material": {
            "phonopy_yaml": "graphite.yaml", "mesh": [40, 40, 40],
            "temperature_K": 5.0,
            "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                            "b_coh_fm": 6.646, "sigma_inc_b": 0.001}],
        },
        "physics": {"inelastic_mode": 2, "max_phonon_order": "auto"},
        "grid": {"e_min_meV": 0.0, "e_max_meV": 250.0, "de_meV": 0.5,
                 "dq_max_invA": 0.05},
        "instrument": {"geometry": "vision", "e_fixed_meV": 3.5,
                       "angles_deg": [45.0, 135.0],
                       "sigma_coeffs": [0.31, 0.005, 8.07e-7]},
    })


def _direct_cfg():
    return SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "sio2.yaml"},
        "physics": {"inelastic_mode": 1, "max_phonon_order": 50},
        "grid": {"e_max_meV": 200.0, "de_meV": 1.0, "dq_max_invA": 0.05},
        "instrument": {"geometry": "direct", "e_fixed_meV": 250.0,
                       "angles_deg": [10.0, 60.0, 120.0]},
    })


def _direct_map_cfg():
    cfg = _direct_cfg()
    cfg.instrument.output_mode = "map"
    cfg.instrument.cut_by = "q"
    cfg.instrument.cut_dq_invA = 0.1
    cfg.instrument.map_coverage_deg = [2.373, 135.955]
    cfg.instrument.map_mask = False
    return cfg


def _mutated(factory, changes):
    """factory() with {"section.field": value} changes applied."""
    cfg = factory()
    for path, value in changes.items():
        section, field = path.split(".")
        setattr(getattr(cfg, section), field, value)
    return cfg


# ---- round-trip -------------------------------------------------------------
@pytest.mark.parametrize("suffix", [".yaml", ".json"])
@pytest.mark.parametrize("factory", [_vision_cfg, _direct_cfg, _direct_map_cfg])
def test_round_trip_identity(tmp_path, suffix, factory):
    cfg = factory()
    assert validate(cfg) is cfg
    p = dump(cfg, tmp_path / f"cfg{suffix}")
    assert load(p) == cfg


def test_round_trip_via_toml_read(tmp_path):
    # tomllib is read-only; hand-write a TOML and check it parses to the dict cfg
    toml = (
        '[material]\nphonopy_yaml = "be.yaml"\nmesh = [20, 20, 20]\n'
        'temperature_K = 296.0\n\n'
        '[physics]\ninelastic_mode = 2\nmax_phonon_order = "auto"\n\n'
        '[grid]\ne_max_meV = 250.0\nde_meV = 0.5\ndq_max_invA = 0.05\n\n'
        '[instrument]\ngeometry = "indirect"\ne_fixed_meV = 4.0\n'
        'angles_deg = [30.0, 90.0, 150.0]\n'
    )
    p = tmp_path / "cfg.toml"
    p.write_text(toml)
    cfg = load(p)
    assert cfg.material.phonopy_yaml == "be.yaml"
    assert cfg.instrument.geometry == "indirect"
    assert cfg.instrument.angles_deg == [30.0, 90.0, 150.0]


def test_defaults_filled_for_missing_sections():
    cfg = SpectraConfig.from_dict({"material": {"phonopy_yaml": "x.yaml"}})
    assert cfg.physics == PhysicsConfig()
    assert cfg.grid == GridConfig()
    assert cfg.instrument == InstrumentConfig()


# ---- validator: structural --------------------------------------------------
def test_missing_material_raises():
    with pytest.raises(SpectraConfigError):
        SpectraConfig.from_dict({"physics": {"inelastic_mode": 1}})


def test_unknown_field_raises():
    with pytest.raises(SpectraConfigError):
        SpectraConfig.from_dict({"material": {"phonopy_yaml": "x.yaml",
                                              "temperatureK": 5.0}})


@pytest.mark.parametrize("factory,changes", [
    (_vision_cfg, {"material.phonopy_yaml": ""}),
    (_vision_cfg, {"physics.inelastic_mode": 3}),
    (_vision_cfg, {"physics.inelastic_mode": -1}),
    (_vision_cfg, {"physics.max_phonon_order": "all"}),
    (_vision_cfg, {"grid.de_meV": 0.0}),
    (_vision_cfg, {"grid.dq_max_invA": -0.1}),
    (_vision_cfg, {"grid.e_min_meV": 100.0, "grid.e_max_meV": 50.0}),
    (_vision_cfg, {"grid.q_pad_invA": -1.0}),        # narrows the Q support
    (_vision_cfg, {"instrument.geometry": "spallation"}),
    (_direct_cfg, {"grid.e_max_meV": 300.0}),         # >= Ei = 250
    (_direct_cfg, {"instrument.angles_deg": None}),   # direct needs angles
    (_direct_cfg, {"instrument.angles_deg": [10.0, 200.0]}),
    (_vision_cfg, {"instrument.sigma_coeffs": [0.0, -0.01, 0.0]}),
    (_vision_cfg, {"instrument.sigma_coeffs": [0.31, 0.005, 8.1e-7, 1e-9]}),
])
def test_validator_rejects(factory, changes):
    with pytest.raises(SpectraConfigError):
        validate(_mutated(factory, changes))


@pytest.mark.parametrize("changes", [
    {"instrument.angles_deg": None},          # vision fills its own angles
    {"grid.q_pad_invA": 0.0},                 # no padding
    {"instrument.output_mode": "map"},        # `irma spectra map` on vision
])
def test_validator_accepts(changes):
    cfg = _mutated(_vision_cfg, changes)
    assert validate(cfg) is cfg


# ---- validator: geometry / kinematics --------------------------------------
@pytest.mark.parametrize("coeffs", [[0.5], [0.5, 0.01]])
def test_short_sigma_coeffs_validate_without_indexerror(coeffs):
    """A 1- or 2-element sigma poly (the GUI emits these from its trailing-blank
    trim; the CLI/YAML accept any length) must validate cleanly -- zero-padded
    through sigma_of_E -- not raise a raw IndexError."""
    cfg = _vision_cfg()
    cfg.instrument.sigma_coeffs = coeffs
    assert validate(cfg) is cfg


def test_sigma_poly_validated_over_full_gain_side_domain():
    """QA4 (Codex): the runtime evaluates sigma at |E|, so a wide energy-gain
    window (|e_min| > e_max) probes the poly BEYOND e_max; validation must
    cover [0, max(e_max, |e_min|)] or a negative gain-side width would be
    silently clipped to a near-zero linewidth downstream."""
    cfg = _vision_cfg()
    cfg.grid.e_min_meV, cfg.grid.e_max_meV = -300.0, 100.0
    # positive everywhere on [0, 100] but negative at |E| ~ 250
    cfg.instrument.sigma_coeffs = [5.0, -0.03, 0.0]
    with pytest.raises(SpectraConfigError):
        validate(cfg)
    cfg.instrument.sigma_coeffs = [5.0, -0.01, 0.0]     # positive out to 300
    assert validate(cfg) is cfg


def test_list_frequency_accepted():
    """A PyChop [resolution, frame] frequency list is accepted (the resolution
    disk, the first element, sets the burst)."""
    cfg = SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "g.yaml"},
        "grid": {"e_max_meV": 100.0, "de_meV": 1.0, "dq_max_invA": 0.05},
        "instrument": {"geometry": "direct", "e_fixed_meV": 250.0,
                       "angles_deg": [30.0, 60.0], "resolution_model": "chopper",
                       "chopper_spec": {"instrument": "ARCS",
                                        "package": "ARCS-700-1.5-AST",
                                        "frequency": [600, 60]}},
    })
    assert validate(cfg) is cfg


def test_chopper_rejects_lorentzian_shape():
    """chopper resolution is Gaussian by construction; pairing it with a
    Lorentzian kernel would mis-scale the width and shape (audit sweep-4)."""
    cfg = _direct_cfg()
    cfg.instrument.resolution_model = "chopper"
    cfg.instrument.resolution_shape = "lorentzian"
    with pytest.raises(SpectraConfigError, match="chopper.*[Gg]aussian"):
        validate(cfg)


@pytest.mark.parametrize("field,value", [
    ("output_mode", "movie"), ("cut_by", "diagonal"), ("cut_dq_invA", 0.0),
    ("cut_dq_invA", -0.1), ("map_coverage_deg", [10.0]),
    ("map_coverage_deg", [120.0, 30.0]), ("map_coverage_deg", [0.0, 90.0])])
def test_bad_direct_output_fields_rejected(field, value):
    cfg = _direct_cfg()
    setattr(cfg.instrument, field, value)
    with pytest.raises(SpectraConfigError):
        validate(cfg)


# ---- strictly-typed booleans -------------------------------------------------
@pytest.mark.parametrize("val,expected", [(True, True), (0, False)])
def test_config_accepts_bool_and_int01_for_boolean_field(val, expected):
    cfg = SpectraConfig.from_dict(
        {"material": {}, "physics": {"elastic": val}})
    assert cfg.physics.elastic is expected


@pytest.mark.parametrize("val", ["off", 2])
def test_config_rejects_non_boolean_values(val):
    with pytest.raises(SpectraConfigError, match="physics.elastic"):
        SpectraConfig.from_dict(
            {"material": {}, "physics": {"elastic": val}})


# ---- inelastic_mode string aliases -------------------------------------------
def test_spectra_config_inelastic_mode_aliases():
    base = {"material": {"phonopy_yaml": "x/phonopy.yaml",
                         "mesh": [4, 4, 4], "temperature_K": 296.0}}
    for alias, want in (("coherent", 2), ("incoherent", 1)):
        cfg = SpectraConfig.from_dict(
            {**base, "physics": {"inelastic_mode": alias}})
        validate(cfg)
        assert cfg.physics.inelastic_mode == want
    cfg = SpectraConfig.from_dict(
        {**base, "physics": {"inelastic_mode": "bogus"}})
    with pytest.raises(SpectraConfigError, match="inelastic_mode"):
        validate(cfg)


# ---- run_spectra forwarding ---------------------------------------------------
def test_run_spectra_forwards_q_cuts(monkeypatch, tmp_path):
    """The flag-form CLI has no --cut-by, so q_cuts must reach
    compute_spectrum even with the default cut_by='angles' (it previously
    arrived as None: --q-cuts was silently inert)."""
    import irma.spectra.forward as fwd
    import irma.spectra.config as cfgmod
    captured = {}

    def fake_compute_spectrum(**kw):
        captured.update(kw)
        return {"fake": True}

    monkeypatch.setattr(fwd, "compute_spectrum", fake_compute_spectrum)
    cfg = SpectraConfig.from_dict({
        "material": {
            "phonopy_yaml": str(tmp_path / "phonopy.yaml"),
            "temperature_K": 300.0,
            "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                            "awr": 11.898}],
        },
        "physics": {"elastic": False},
        "instrument": {"geometry": "vision", "q_cuts": [2.0, 5.0]},
    })
    out = cfgmod.run_spectra(cfg)
    assert out == {"fake": True}
    assert captured["q_cuts"] == [2.0, 5.0]
    assert captured["produce_angle_spectra"] is True


# ---- modes-1/2 species-symbol check (GUI-autofill review, NEW-1) ------------
def _species_yaml(tmp_path, symbols):
    """Minimal plain-YAML phonopy.yaml with a unit_cell points list."""
    p = tmp_path / "phonopy.yaml"
    pts = "\n".join(f"  - symbol: {s}\n    coordinates: [0.0, 0.0, 0.0]"
                    for s in symbols)
    p.write_text(f"unit_cell:\n  points:\n{pts}\n")
    return str(p)


def test_phonopy_species_parses_plain_yaml(tmp_path):
    from irma.spectra.config import phonopy_species
    assert phonopy_species(_species_yaml(tmp_path, ["C", "C", "O"])) == ["C", "O"]
    assert phonopy_species(str(tmp_path / "missing.yaml")) == []


def test_species_check_refuses_nonmatching_symbol(tmp_path):
    """A C-13 row on a C structure must fail loudly: the engine matches
    per-site constants by exact chemical symbol and would silently fall
    back to built-in natural-C values."""
    from irma.spectra.config import (MaterialConfig, Scatterer,
                                     _check_species_symbols)
    m = MaterialConfig(
        phonopy_yaml=_species_yaml(tmp_path, ["C"]),
        scatterers=[Scatterer(symbol="C-13", sigma_bound_b=5.5, awr=12.9)])
    with pytest.raises(SpectraConfigError, match="C-13"):
        _check_species_symbols(m, PhysicsConfig(inelastic_mode=2))
    with pytest.raises(SpectraConfigError, match="exact chemical symbol"):
        _check_species_symbols(m, PhysicsConfig(inelastic_mode=1))


def test_species_check_passes_matching_and_skips_mode0(tmp_path):
    from irma.spectra.config import (MaterialConfig, Scatterer,
                                     _check_species_symbols)
    yaml_path = _species_yaml(tmp_path, ["C"])
    ok = MaterialConfig(phonopy_yaml=yaml_path,
                        scatterers=[Scatterer(symbol="C", sigma_bound_b=5.55,
                                              awr=11.9)])
    _check_species_symbols(ok, PhysicsConfig(inelastic_mode=2))   # no raise
    iso = MaterialConfig(phonopy_yaml=yaml_path,
                         scatterers=[Scatterer(symbol="C-13", sigma_bound_b=5.5,
                                               awr=12.9)])
    _check_species_symbols(iso, PhysicsConfig(inelastic_mode=0))  # mode 0 free
    # unparseable yaml -> advisory skip, engine reports its own error
    iso.phonopy_yaml = str(tmp_path / "nope.yaml")
    _check_species_symbols(iso, PhysicsConfig(inelastic_mode=2))
