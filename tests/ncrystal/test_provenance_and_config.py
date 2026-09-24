"""Provenance metadata + export-config parsing. Pure/fast."""
from __future__ import annotations

import pytest

from irma.ncrystal.provenance import collect_provenance, file_sha256
from irma.ncrystal.config import NCrystalExportConfig
from irma.spectra.config import SpectraConfigError


def test_collect_provenance_keys(tmp_path):
    yaml_path = tmp_path / "phonopy.yaml"
    yaml_path.write_text("phonopy: data\n")
    meta = collect_provenance(
        phonopy_yaml=yaml_path, mesh=[40, 40, 40], temperature_K=296.0,
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


def _export(**kv):
    return lambda d: d["export"].update(kv)


def _material(**kv):
    return lambda d: d["material"].update(kv)


def _scatterer(**kv):
    return lambda d: d["material"]["scatterers"][0].update(kv)


def _drop_scatterer_field(name):
    return lambda d: d["material"]["scatterers"][0].pop(name)


def _duplicate_scatterer(d):
    d["material"]["scatterers"] *= 2


@pytest.mark.parametrize("edit, match", [
    pytest.param(lambda d: d.update(instrument={}), "unknown config section",
                 id="section"),
    pytest.param(_export(bogus=1), "unknown export key", id="export-key"),
    pytest.param(_scatterer(typo=1), "unknown field", id="scatterer-key"),
    pytest.param(_export(gain_side="mirror"), "gain_side", id="gain-side"),
    pytest.param(_export(inelastic_mode=3), "inelastic_mode", id="mode"),
    pytest.param(_material(mesh=[40, 40]), "mesh", id="mesh-length"),
    pytest.param(_material(mesh=[40, 0, 40]), "mesh", id="mesh-zero"),
    pytest.param(_material(mesh=[40, 4.9, 40]), "mesh", id="mesh-fraction"),
    pytest.param(_material(phonopy_yaml=""), "phonopy_yaml", id="no-yaml"),
    pytest.param(_material(temperature_K=0.0), "temperature_K must be",
                 id="temperature"),
    # a plausible typo (underscores) must fail at config load, not in the engine
    pytest.param(_export(coherent_partition_mode="principal_xs_weighted"),
                 "coherent_partition_mode", id="partition-mode"),
    pytest.param(_export(incoherent_elastic_mode="anisotropic"),
                 "incoherent_elastic_mode", id="incoherent-mode"),
    pytest.param(_export(incoherent_elastic_mode="directional", elastic=False),
                 "requires .*elastic", id="directional-needs-elastic"),
    pytest.param(_drop_scatterer_field("b_coh_fm"), "'C'.*b_coh_fm",
                 id="no-b-coh"),
    pytest.param(_drop_scatterer_field("sigma_inc_b"), "'C'.*sigma_inc_b",
                 id="no-sigma-inc"),
    pytest.param(_scatterer(sigma_inc_b=-0.5), "'C' sigma_inc_b",
                 id="negative-sigma-inc"),
    pytest.param(_scatterer(sigma_bound_b=-1.0), "sigma_bound_b",
                 id="negative-sigma-bound"),
    pytest.param(_scatterer(awr=-1.0), "awr", id="negative-awr"),
    pytest.param(_duplicate_scatterer, "duplicate symbol", id="duplicate"),
    pytest.param(_export(num_directions=0), "num_directions", id="ndir-zero"),
    pytest.param(_export(num_directions=3.9), "num_directions",
                 id="ndir-fraction"),
    pytest.param(_export(alpha_grid=[2.0, 1.0], beta_grid=[0.0, 1.0]),
                 "alpha_grid.*strictly increasing", id="alpha-order"),
    pytest.param(_export(alpha_grid=[0.1, 1.0], beta_grid=[-1.0, 0.0]),
                 "beta_grid.*nonnegative", id="beta-negative"),
    pytest.param(_export(alpha_grid=[0.0, 1.0], beta_grid=[0.0, 1.0]),
                 "alpha_grid.*positive", id="alpha-zero"),
    pytest.param(_export(alpha_grid=[0.1], beta_grid=[0.0, 1.0]),
                 "alpha_grid.*two points", id="alpha-short"),
])
def test_config_rejects(edit, match):
    d = _cfg_dict()
    edit(d)
    with pytest.raises(SpectraConfigError, match=match):
        NCrystalExportConfig.from_dict(d)


def test_config_accepts_valid_modes():
    for mode in ("auto", "exact-total", "principal-xs-weighted"):
        cfg = NCrystalExportConfig.from_dict(_cfg_dict(coherent_partition_mode=mode))
        assert cfg.coherent_partition_mode == mode
    for mode in ("isotropic", "directional"):
        cfg = NCrystalExportConfig.from_dict(
            _cfg_dict(incoherent_elastic_mode=mode))
        assert cfg.incoherent_elastic_mode == mode
    assert NCrystalExportConfig.from_dict(
        _cfg_dict()).incoherent_elastic_mode == "isotropic"


def test_auto_grid_is_the_endf_converged_grid_not_uniform():
    """The NCrystal export's automatic grid IS the ENDF evaluator's converged
    grid with the lin-lin treatment (generate_beta_grid(iint=1) +
    generate_alpha_grid), with the config's knobs: NCrystal interpolates
    S(alpha,beta) linearly in beta, so the tail must carry the lin-lin cap
    (on these knobs the capped grid has 354 points, the uncapped one 215).
    freq_max is pinned here so the check needs no phonopy."""
    import numpy as np
    from irma.ncrystal.build import _auto_beta_grid, _group_grids
    from irma.core.grids import (generate_beta_grid, generate_alpha_grid,
                                 grid_reference_temperature_K)
    from irma.core.grids import AUTO_GRID_DEFAULTS as _D

    awr = 11.898
    cfg = NCrystalExportConfig.from_dict(
        _cfg_dict(freq_max_eV=0.15, n_phonon=120, alpha_dq_invA=0.1))
    assert cfg.grid_mode == "auto"
    t_ref = grid_reference_temperature_K(cfg.lat, cfg.material.temperature_K)

    beta, preloaded = _auto_beta_grid(cfg, t_ref, awr)
    assert preloaded is None        # freq_max pinned -> no phonopy mesh loaded
    alpha, beta_out = _group_grids(cfg, awr, beta, t_ref)

    beta_ref = generate_beta_grid(
        0.15, t_ref, iint=1, awr=awr, n_lower=_D["n_lower"], n_phonon=120,
        n_upper=_D["n_upper"], beta_max_eV=_D["beta_max_eV"],
        evaluation_temperatures_K=[cfg.material.temperature_K])
    alpha_ref = generate_alpha_grid(beta_ref, awr, t_ref, dq_ang_inv=0.1,
                                    q_cut_ang_inv=_D["alpha_qcut_invA"],
                                    n_log=_D["alpha_nlog"])
    assert np.allclose(beta, beta_ref)
    assert np.allclose(alpha, alpha_ref)
    assert np.allclose(beta_out, beta_ref)


def test_endf_core_never_imports_the_exporter():
    """Scope guard: the exporter's grid treatment must not leak into the
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


def test_config_explicit_order_int():
    cfg = NCrystalExportConfig.from_dict(_cfg_dict(multiphonon_max_order=8))
    assert cfg.auto_multiphonon_order is False
    assert cfg.effective_multiphonon_max_order == 8


def test_config_inelastic_mode_string_aliases():
    assert NCrystalExportConfig.from_dict(
        _cfg_dict(inelastic_mode="incoherent")).inelastic_mode == 1
    # the export has no mode 0, so the spectra-only 'dos' alias is rejected
    with pytest.raises(SpectraConfigError, match="inelastic_mode"):
        NCrystalExportConfig.from_dict(_cfg_dict(inelastic_mode="dos"))


def test_negative_b_coh_is_physical():
    sc = {"symbol": "H", "sigma_bound_b": 82.02, "awr": 0.99917,
          "b_coh_fm": -3.7406, "sigma_inc_b": 80.26}
    d = _cfg_dict()
    d["material"]["scatterers"] = [sc]
    cfg = NCrystalExportConfig.from_dict(d)
    assert cfg.material.scatterers[0].b_coh_fm == pytest.approx(-3.7406)
