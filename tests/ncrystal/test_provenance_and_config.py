"""Provenance metadata + export-config parsing. Pure/fast."""
from __future__ import annotations

import pytest

from irma.ncrystal.provenance import collect_provenance, file_sha256, irma_version
from irma.ncrystal.config import NCrystalExportConfig
from irma.spectra.config import SpectraConfigError


def test_collect_provenance_keys():
    meta = collect_provenance(
        phonopy_yaml=None, mesh=[40, 40, 40], temperature_K=296.0,
        num_directions=10000, multiphonon_num_directions=1000,
        multiphonon_max_order="auto", inelastic_mode=2)
    for key in ("irma_version", "irma_git_sha", "mesh", "temperature_K",
                "num_directions", "multiphonon_num_directions",
                "multiphonon_max_order", "irma_inelastic_mode"):
        assert key in meta
    assert meta["mesh"] == "40x40x40"
    assert meta["multiphonon_max_order"] == "auto"
    assert meta["irma_inelastic_mode"] == "2"


def test_file_sha256_deterministic(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("phonopy: data\n")
    assert file_sha256(f) == file_sha256(f)
    assert file_sha256(tmp_path / "missing") == "unknown"


def test_irma_version_never_raises():
    assert isinstance(irma_version(), str)


def _cfg_dict(**over):
    d = {
        "material": {
            "phonopy_yaml": "graphite/phonopy.yaml",
            "mesh": [40, 40, 40],
            "temperature_K": 296.0,
            "scatterers": [
                {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                 "b_coh_fm": 6.646, "sigma_inc_b": 0.001},
            ],
        },
        "export": {"material_id": "graphite"},
    }
    d["export"].update(over)
    return d


def test_config_from_dict_defaults():
    cfg = NCrystalExportConfig.from_dict(_cfg_dict())
    assert cfg.material_id == "graphite"
    assert cfg.inelastic_mode == 2
    assert cfg.gain_side == "scaled_sym"
    assert cfg.auto_multiphonon_order is True
    assert cfg.effective_multiphonon_max_order == 100
    assert cfg.material.scatterers[0].symbol == "C"


def test_config_rejects_unknown_section():
    d = _cfg_dict()
    d["instrument"] = {}
    with pytest.raises(SpectraConfigError, match="unknown config section"):
        NCrystalExportConfig.from_dict(d)


def test_config_rejects_unknown_export_key():
    with pytest.raises(SpectraConfigError, match="unknown export key"):
        NCrystalExportConfig.from_dict(_cfg_dict(bogus=1))


def test_config_rejects_unknown_scatterer_key():
    d = _cfg_dict()
    d["material"]["scatterers"][0]["typo"] = 1
    with pytest.raises(SpectraConfigError, match="unknown scatterer key"):
        NCrystalExportConfig.from_dict(d)


def test_config_rejects_bad_gain_side():
    with pytest.raises(SpectraConfigError, match="gain_side"):
        NCrystalExportConfig.from_dict(_cfg_dict(gain_side="mirror"))


def test_config_rejects_bad_mode():
    with pytest.raises(SpectraConfigError, match="inelastic_mode"):
        NCrystalExportConfig.from_dict(_cfg_dict(inelastic_mode=3))


def test_config_rejects_bad_mesh():
    # wrong length -> the three-entries message; nonpositive or fractional
    # entries -> the exact-integer >= 1 message. Both name the mesh.
    for bad in ([40, 40], [40, 0, 40], [40, -1, 40], [40, 4.9, 40]):
        d = _cfg_dict()
        d["material"]["mesh"] = bad
        with pytest.raises(SpectraConfigError, match="mesh"):
            NCrystalExportConfig.from_dict(d)


def test_config_rejects_missing_phonopy_yaml():
    d = _cfg_dict()
    d["material"]["phonopy_yaml"] = ""
    with pytest.raises(SpectraConfigError, match="phonopy_yaml"):
        NCrystalExportConfig.from_dict(d)


def test_config_rejects_negative_site_group_index():
    with pytest.raises(SpectraConfigError, match="site_groups index"):
        NCrystalExportConfig.from_dict(_cfg_dict(site_groups=[[0], [-1]]))


# -- export-config sibling validations ----------------------------------------

def test_config_rejects_nonpositive_temperature():
    for bad in (0.0, -5.0, float("nan")):  # NaN slips past a bare `<= 0` check
        d = _cfg_dict()
        d["material"]["temperature_K"] = bad
        with pytest.raises(SpectraConfigError, match="temperature_K must be"):
            NCrystalExportConfig.from_dict(d)


def test_config_rejects_bad_partition_mode():
    # a plausible typo (underscores) must fail at config load, not deep in the engine
    with pytest.raises(SpectraConfigError, match="coherent_partition_mode"):
        NCrystalExportConfig.from_dict(
            _cfg_dict(coherent_partition_mode="principal_xs_weighted"))


def test_config_accepts_valid_partition_modes():
    for mode in ("auto", "exact-total", "principal-xs-weighted"):
        cfg = NCrystalExportConfig.from_dict(_cfg_dict(coherent_partition_mode=mode))
        assert cfg.coherent_partition_mode == mode


def test_config_rejects_empty_site_groups():
    with pytest.raises(SpectraConfigError, match="site_groups must be non-empty"):
        NCrystalExportConfig.from_dict(_cfg_dict(site_groups=[]))


def test_config_rejects_empty_inner_group():
    with pytest.raises(SpectraConfigError, match="empty group"):
        NCrystalExportConfig.from_dict(_cfg_dict(site_groups=[[0, 1], []]))


def test_auto_grid_is_the_endf_converged_grid_not_uniform():
    """The NCrystal export's automatic grid IS the ENDF evaluator's converged
    grid with the lin-lin treatment (generate_beta_grid_for_iint(iint=1) +
    generate_alpha_grid), NOT a uniform Q/E grid and NOT the bare uncapped
    builder: NCrystal interpolates S(alpha,beta) linearly in beta, so the
    baked tail must carry the DELTA_BETA_MAX_LINLIN recoil-ridge cap like
    every other auto-grid caller (NCB-1). freq_max is pinned here so the
    check needs no phonopy."""
    import numpy as np
    from irma.ncrystal.build import _auto_beta_grid, _group_grids
    from irma.core.grids import (generate_beta_grid, generate_beta_grid_for_iint,
                                 generate_alpha_grid,
                                 grid_reference_temperature_K)

    awr = 11.898
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(freq_max_eV=0.2))
    assert cfg.grid_mode == "auto"
    t_ref = grid_reference_temperature_K(cfg.lat, cfg.material.temperature_K)

    beta, preloaded = _auto_beta_grid(cfg, t_ref, awr)
    assert preloaded is None        # freq_max pinned -> no phonopy mesh loaded
    alpha, beta_out = _group_grids(cfg, awr, beta, t_ref)

    from irma.core.grids import AUTO_GRID_DEFAULTS as _D
    beta_ref = generate_beta_grid_for_iint(
        0.2, t_ref, iint=1, awr=awr, n_lower=_D["n_lower"], n_phonon=300,
        n_upper=80, beta_max_eV=5.0)
    alpha_ref = generate_alpha_grid(beta_ref, awr, t_ref, dq_ang_inv=0.05,
                                    q_cut_ang_inv=12.0, n_log=160)
    assert np.allclose(beta, beta_ref)
    assert np.allclose(alpha, alpha_ref)
    assert np.allclose(beta_out, beta_ref)
    # ...and it is genuinely the CAPPED grid: the bare uncapped builder gives a
    # different (coarser) tail, so this test fails if build.py reverts to it.
    beta_uncapped = generate_beta_grid(0.2, t_ref, n_lower=_D["n_lower"],
                                       n_phonon=300,
                                       n_upper=80, beta_max_eV=5.0)
    assert len(beta) != len(beta_uncapped)


def test_auto_grid_honors_config_overrides():
    """The optional knobs (freq_max_eV, n_phonon, alpha_dq) flow into the grid."""
    import numpy as np
    from irma.ncrystal.build import _auto_beta_grid, _group_grids
    from irma.core.grids import (generate_beta_grid_for_iint, generate_alpha_grid,
                                 grid_reference_temperature_K)

    awr = 8.93478
    cfg = NCrystalExportConfig.from_dict(
        _cfg_dict(freq_max_eV=0.15, n_phonon=120, alpha_dq_invA=0.1))
    t_ref = grid_reference_temperature_K(cfg.lat, cfg.material.temperature_K)
    beta, _preloaded = _auto_beta_grid(cfg, t_ref, awr)
    alpha, _ = _group_grids(cfg, awr, beta, t_ref)
    from irma.core.grids import AUTO_GRID_DEFAULTS as _D
    beta_ref = generate_beta_grid_for_iint(
        0.15, t_ref, iint=1, awr=awr, n_lower=_D["n_lower"], n_phonon=120,
        n_upper=80, beta_max_eV=5.0)
    alpha_ref = generate_alpha_grid(beta_ref, awr, t_ref, dq_ang_inv=0.1,
                                    q_cut_ang_inv=12.0, n_log=160)
    assert np.allclose(beta, beta_ref)
    assert np.allclose(alpha, alpha_ref)


def test_auto_beta_grid_caps_step_below_recoil_ridge():
    """NCB-1 guard: the exporter's baked auto tail respects the lin-lin step
    cap below the recoil ridge — NCrystal's SAB kernel interpolates linearly
    in beta, so the uncapped pure-log tail (which demonstrably violates the
    cap on the same knobs) would overshoot the free-atom cross section."""
    import numpy as np
    from irma.ncrystal.build import _auto_beta_grid
    from irma.core.constants import BK
    from irma.core.grids import (generate_beta_grid, DELTA_BETA_MAX_LINLIN,
                                 grid_reference_temperature_K)

    awr = 11.898
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(freq_max_eV=0.2))
    t_ref = grid_reference_temperature_K(cfg.lat, cfg.material.temperature_K)
    beta, _ = _auto_beta_grid(cfg, t_ref, awr)
    # The cap governs the UPPER (post-phonon-region) tail below the recoil
    # ridge; the linear phonon region keeps its freq_max/n_phonon step.
    lin_end = 0.2 * (1.0 - 1.0 / cfg.n_phonon) / (BK * t_ref)
    ridge = 4.0 * (cfg.beta_max_eV / (BK * t_ref)) / awr
    steps = np.diff(beta)
    in_tail = (beta[1:] > lin_end + 1e-9) & (beta[1:] <= ridge + 1e-9)
    assert in_tail.sum() > 0
    assert steps[in_tail].max() <= DELTA_BETA_MAX_LINLIN * (1.0 + 1e-12)
    # The assertion discriminates: the bare (uncapped) builder violates the
    # cap below the ridge on the SAME knobs, so a revert of build.py to
    # generate_beta_grid fails here.
    uncapped = generate_beta_grid(
        0.2, t_ref, n_lower=cfg.n_lower, n_phonon=cfg.n_phonon,
        n_upper=cfg.n_upper, beta_max_eV=cfg.beta_max_eV)
    usteps = np.diff(uncapped)
    u_tail = (uncapped[1:] > lin_end + 1e-9) & (uncapped[1:] <= ridge + 1e-9)
    assert usteps[u_tail].max() > DELTA_BETA_MAX_LINLIN


def test_endf_core_never_imports_the_exporter():
    """NCB-1 scope guard: the exporter's grid treatment must not leak into the
    ENDF write path. ENDF tapes are built from irma.core's own grid logic, so
    no irma.core module may import from irma.ncrystal (comment/docstring
    mentions are fine; actual imports are not)."""
    import ast
    import pathlib
    import irma.core
    core_dir = pathlib.Path(irma.core.__file__).resolve().parent
    offenders = []
    for p in sorted(core_dir.glob("*.py")):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                if any(a.name.startswith("irma.ncrystal") for a in node.names):
                    offenders.append(p.name)
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("irma.ncrystal"):
                    offenders.append(p.name)
    assert offenders == []


def test_config_normalizes_grid_knob_types():
    """YAML-string / fractional grid knobs are coerced at config load (and written
    back), not left to crash later in generate_*_grid / geomspace."""
    cfg = NCrystalExportConfig.from_dict(
        _cfg_dict(n_phonon="120", n_lower=10.0, freq_max_eV="0.2",
                  alpha_dq_invA="0.05"))
    assert cfg.n_phonon == 120 and isinstance(cfg.n_phonon, int)
    assert cfg.n_lower == 10 and isinstance(cfg.n_lower, int)
    assert cfg.freq_max_eV == 0.2 and isinstance(cfg.freq_max_eV, float)
    assert cfg.alpha_dq_invA == 0.05 and isinstance(cfg.alpha_dq_invA, float)


def test_config_explicit_order_int():
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(multiphonon_max_order=8))
    assert cfg.auto_multiphonon_order is False
    assert cfg.effective_multiphonon_max_order == 8


def test_gain_side_asym_rejected_at_config_load():
    # gain_side='asym' is designed but unimplemented: the config contract must
    # reject it at LOAD (not let it ride to a NotImplementedError at bake time,
    # after any preceding work).
    with pytest.raises(SpectraConfigError, match="asym"):
        NCrystalExportConfig.from_dict(_cfg_dict(gain_side="asym"))


def test_gain_side_asym_build_guard_still_present():
    # Defense in depth: a caller that bypasses validation (mutating the config
    # after construction) must still fail loudly in build_packs BEFORE any
    # phonopy work (so this needs no fixture).
    from irma.ncrystal.build import build_packs
    cfg = NCrystalExportConfig.from_dict(_cfg_dict())
    cfg.gain_side = "asym"
    with pytest.raises(NotImplementedError, match="asym"):
        build_packs(cfg, progress=lambda *a: None)


def test_config_inelastic_mode_string_aliases():
    assert NCrystalExportConfig.from_dict(
        _cfg_dict(inelastic_mode="incoherent")).inelastic_mode == 1
    # the export has no mode 0, so the spectra-only 'dos' alias is rejected
    with pytest.raises(SpectraConfigError, match="inelastic_mode"):
        NCrystalExportConfig.from_dict(_cfg_dict(inelastic_mode="dos"))


def test_config_rejects_nonfinite_and_degenerate_grid_knobs():
    with pytest.raises(SpectraConfigError, match="freq_max_eV"):
        NCrystalExportConfig.from_dict(_cfg_dict(freq_max_eV=float("inf")))
    with pytest.raises(SpectraConfigError, match="beta_max_eV"):
        NCrystalExportConfig.from_dict(_cfg_dict(beta_max_eV=float("nan")))
    with pytest.raises(SpectraConfigError, match="alpha_nlog"):
        NCrystalExportConfig.from_dict(_cfg_dict(alpha_nlog=1))


def test_config_rejects_bad_incoherent_elastic_mode():
    with pytest.raises(SpectraConfigError, match="incoherent_elastic_mode"):
        NCrystalExportConfig.from_dict(
            _cfg_dict(incoherent_elastic_mode="anisotropic"))


def test_config_accepts_incoherent_elastic_modes():
    for mode in ("isotropic", "directional"):
        cfg = NCrystalExportConfig.from_dict(
            _cfg_dict(incoherent_elastic_mode=mode))
        assert cfg.incoherent_elastic_mode == mode
    assert NCrystalExportConfig.from_dict(
        _cfg_dict()).incoherent_elastic_mode == "isotropic"


def test_config_directional_incoherent_requires_elastic():
    with pytest.raises(SpectraConfigError, match="requires .*elastic"):
        NCrystalExportConfig.from_dict(
            _cfg_dict(incoherent_elastic_mode="directional", elastic=False))


# ---- review S1: strict booleans -------------------------------------------

@pytest.mark.parametrize("val", [True, False, 0, 1])
def test_elastic_accepts_real_booleans(val):
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(elastic=val))
    assert cfg.elastic is bool(val)


@pytest.mark.parametrize("val", ["true", "false", "on", "off", "False", "0",
                                 None, 2, "x", []])
def test_elastic_rejects_non_boolean_representations(val):
    with pytest.raises(SpectraConfigError, match="export.elastic"):
        NCrystalExportConfig.from_dict(_cfg_dict(elastic=val))


# ---- review S2: elastic neutron constants are required --------------------

def _cfg_dict_scatterer(scatterer, **over):
    d = _cfg_dict(**over)
    d["material"]["scatterers"] = [scatterer]
    return d


@pytest.mark.parametrize("field", ["b_coh_fm", "sigma_inc_b"])
def test_elastic_requires_neutron_constants(field):
    sc = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
          "b_coh_fm": 6.646, "sigma_inc_b": 0.001}
    del sc[field]
    with pytest.raises(SpectraConfigError, match=f"'C'.*{field}"):
        NCrystalExportConfig.from_dict(_cfg_dict_scatterer(sc))


@pytest.mark.parametrize("field,val", [
    ("b_coh_fm", float("nan")), ("b_coh_fm", float("inf")),
    ("sigma_inc_b", float("nan")), ("sigma_inc_b", float("inf")),
    ("sigma_inc_b", -0.5),
])
def test_elastic_rejects_nonphysical_neutron_constants(field, val):
    sc = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
          "b_coh_fm": 6.646, "sigma_inc_b": 0.001, field: val}
    with pytest.raises(SpectraConfigError, match=f"'C' {field}"):
        NCrystalExportConfig.from_dict(_cfg_dict_scatterer(sc))


def test_negative_b_coh_is_physical():
    sc = {"symbol": "H", "sigma_bound_b": 82.02, "awr": 0.99917,
          "b_coh_fm": -3.7406, "sigma_inc_b": 80.26}
    cfg = NCrystalExportConfig.from_dict(_cfg_dict_scatterer(sc))
    assert cfg.material.scatterers[0].b_coh_fm == pytest.approx(-3.7406)


# ---- review NC-1: constants are required even for inelastic-only exports --

@pytest.mark.parametrize("field", ["b_coh_fm", "sigma_inc_b"])
def test_inelastic_only_export_still_requires_neutron_constants(field):
    """The mode-1/2 engine derives its channel weights from b_coh_fm and
    sigma_inc_b; omitting them with elastic=false used to substitute 0.0 and
    bake an identically zero S(alpha,beta) with exit 0 (review NC-1)."""
    sc = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
          "b_coh_fm": 6.646, "sigma_inc_b": 0.001}
    del sc[field]
    with pytest.raises(SpectraConfigError, match=f"'C'.*{field}"):
        NCrystalExportConfig.from_dict(_cfg_dict_scatterer(sc, elastic=False))


@pytest.mark.parametrize("field,val", [
    ("b_coh_fm", float("nan")),
    ("sigma_inc_b", float("nan")),
    ("sigma_inc_b", -1.0),
])
def test_inelastic_only_export_rejects_nonphysical_constants(field, val):
    """A negative sigma_inc_b with elastic=false used to be accepted and
    silently scale the inelastic kernel (review NC-1 probe: ratio 0.82)."""
    sc = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
          "b_coh_fm": 6.646, "sigma_inc_b": 0.001, field: val}
    with pytest.raises(SpectraConfigError, match=f"'C' {field}"):
        NCrystalExportConfig.from_dict(_cfg_dict_scatterer(sc, elastic=False))


def test_duplicate_scatterer_symbols_rejected():
    d = _cfg_dict()
    d["material"]["scatterers"] = d["material"]["scatterers"] * 2
    with pytest.raises(SpectraConfigError, match="duplicate symbol"):
        NCrystalExportConfig.from_dict(d)


# ---- review S3: integer-coded fields must be exactly integral -------------

@pytest.mark.parametrize("field,val", [
    ("num_directions", 3.9), ("num_directions", True),
    ("num_directions", float("nan")), ("num_directions", float("inf")),
    ("num_directions", 0), ("num_directions", -4),
    ("multiphonon_num_directions", 7.7),
    ("inelastic_mode", 1.9),
    ("jobs", 2.5),
    ("n_phonon", 100.5), ("n_upper", 3.3), ("alpha_nlog", 4.5),
])
def test_integer_fields_reject_inexact_values(field, val):
    with pytest.raises(SpectraConfigError):
        NCrystalExportConfig.from_dict(_cfg_dict(**{field: val}))


@pytest.mark.parametrize("field,val,expect", [
    ("num_directions", "6", 6), ("num_directions", 6.0, 6),
    ("inelastic_mode", 2.0, 2), ("jobs", 4, 4),
])
def test_integer_fields_accept_exact_values(field, val, expect):
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(**{field: val}))
    assert getattr(cfg, field) == expect


def test_mesh_rejects_fractional_entries():
    d = _cfg_dict()
    d["material"]["mesh"] = [4.9, 4, 4]
    with pytest.raises(SpectraConfigError, match="mesh"):
        NCrystalExportConfig.from_dict(d)


def test_site_groups_reject_fractional_and_duplicate_indices():
    with pytest.raises(SpectraConfigError, match="EXACT integer"):
        NCrystalExportConfig.from_dict(_cfg_dict(site_groups=[[0.9, 1]]))
    with pytest.raises(SpectraConfigError, match="more than once"):
        NCrystalExportConfig.from_dict(_cfg_dict(site_groups=[[0, 1], [1]]))
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(site_groups=[[0, 1.0]]))
    assert cfg.site_groups == [[0, 1]]


# ---- review NC-2: export config fails fast on nonphysical scalars/grids ----

@pytest.mark.parametrize("field,val,pat", [
    ("sigma_bound_b", float("nan"), "sigma_bound_b"),
    ("sigma_bound_b", -1.0, "sigma_bound_b"),
    ("awr", float("inf"), "awr"),
    ("awr", -1.0, "awr"),
])
def test_nonphysical_scalars_rejected_at_config(field, val, pat):
    sc = {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
          "b_coh_fm": 6.646, "sigma_inc_b": 0.001, field: val}
    with pytest.raises(SpectraConfigError, match=pat):
        NCrystalExportConfig.from_dict(_cfg_dict_scatterer(sc))


@pytest.mark.parametrize("alpha,beta,pat", [
    ([2.0, 1.0], [0.0, 1.0], "alpha_grid.*strictly increasing"),
    ([0.1, 1.0], [-1.0, 0.0], "beta_grid.*nonnegative"),
    ([0.0, 1.0], [0.0, 1.0], "alpha_grid.*positive"),
    ([float("nan"), 1.0], [0.0, 1.0], "alpha_grid.*finite"),
    ([0.1], [0.0, 1.0], "alpha_grid.*two points"),
])
def test_explicit_grids_validated_at_config(alpha, beta, pat):
    with pytest.raises(SpectraConfigError, match=pat):
        NCrystalExportConfig.from_dict(
            _cfg_dict(alpha_grid=alpha, beta_grid=beta))
