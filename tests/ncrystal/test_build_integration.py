"""Integration (reference) tests for the IRMA→NCrystal exporter (SP1).

Slow: runs the mode-2 engine on tiny grids. Gated on phonopy + the bundled
graphite/BeO phonopy fixtures. The core assertion is ENGINE-OUTPUT-EQUALITY: the
pack arrays equal ``run_noncubic_standalone_sab``'s output after the documented
conversion — convert is the only transform, IRMA is the reference implementation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("phonopy")

from irma.ncrystal.config import NCrystalExportConfig
from irma.ncrystal.build import build_packs, write_packs
from irma.ncrystal.pack import read_pack

_REPO = Path(__file__).resolve().parents[2]
_GRAPHITE_YAML = _REPO / "tests/mode2_euphonic_n1_validation/graphite/phonopy.yaml"
_BEO_YAML = _REPO / "tests/mode2_euphonic_n1_validation/beo/phonopy.yaml"

# tiny, fast smoke grids (mirror the scaffold's compact example)
_ALPHA = [0.08, 0.25, 0.7, 1.5, 3.0, 6.0, 12.0]
_BETA = [0.0, 0.25, 0.8, 1.6, 3.0, 5.0, 8.0]


def _graphite_cfg(**over):
    d = {
        "material": {
            "phonopy_yaml": str(_GRAPHITE_YAML),
            "mesh": [2, 2, 2],
            "temperature_K": 296.0,
            "scatterers": [
                {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                 "b_coh_fm": 6.646, "sigma_inc_b": 0.001},
            ],
        },
        "export": {
            "material_id": "graphite", "inelastic_mode": 2,
            "num_directions": 64, "multiphonon_num_directions": 16,
            "multiphonon_max_order": 2, "alpha_grid": _ALPHA, "beta_grid": _BETA,
        },
    }
    d["export"].update(over)
    return NCrystalExportConfig.from_dict(d)


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_graphite_single_pack_engine_equality(tmp_path):
    cfg = _graphite_cfg()
    packs, snippet = build_packs(cfg, progress=lambda *a: None)
    assert len(packs) == 1
    pack = packs[0]
    assert pack.material_id == "graphite__C"
    assert pack.sab_representation == "scaled_sym_sab"
    assert pack.temperature_K == pytest.approx(296.0)
    assert pack.beta_grid[0] == 0.0

    # ENGINE-OUTPUT-EQUALITY: re-run the engine directly and confirm the pack is
    # exactly that output, converted. source_sigma == bound_xs -> rescale is unity.
    from irma.core.noncubic_inelastic import NoncubicInelasticControls
    from irma.core.standalone_sab import run_noncubic_standalone_sab
    res = run_noncubic_standalone_sab(
        alpha=np.asarray(_ALPHA, float), beta=np.asarray(_BETA, float), lat=1,
        temperature_k=296.0, awr=11.898,
        phonopy_yaml_path=str(_GRAPHITE_YAML), mesh_dim=(2, 2, 2),
        num_jobs=1, sigma_mev=0.0,
        controls=NoncubicInelasticControls(
            num_directions=64, multiphonon_num_directions=16,
            multiphonon_max_order=2, auto_multiphonon_order=False),
        inelastic_mode=2, represented_principal_site_count=4,
        principal_group_index=0, site_groups=[[0, 1, 2, 3]],
        coherent_partition_mode="principal-xs-weighted",
        sab_sigma_barn=5.551,
        site_scattering_lengths_angstrom=[6.646e-5] * 4,
        site_incoherent_cross_sections_barn=[0.001] * 4)

    alpha_ref = np.asarray(res["alpha_abs"], float)
    beta_ref = np.asarray(res["beta_downscatter_abs"], float)
    sab_ref = np.asarray(res["sab_downscatter_qe"], float)   # (nalpha, nbeta)

    assert pack.alpha_grid == pytest.approx((alpha_ref * 11.898).tolist())
    assert pack.beta_grid == pytest.approx(beta_ref.tolist())

    nalpha, nbeta = sab_ref.shape
    for ib in range(nbeta):
        scale = math.exp(-0.5 * beta_ref[ib])
        for ia in range(nalpha):
            expected = max(sab_ref[ia, ib], 0.0) * scale   # negative fringe -> 0
            assert pack.sab_values[ib * nalpha + ia] == pytest.approx(
                expected, rel=1e-9, abs=1e-12)


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_graphite_elastic_block_and_roundtrip(tmp_path):
    cfg = _graphite_cfg()
    pack_paths, ncmat_path = write_packs(cfg, tmp_path, progress=lambda *a: None)
    assert len(pack_paths) == 1
    got = read_pack(pack_paths[0])
    # anisotropic DW: 4 primitive C sites -> 36 tensor floats, 4 symbols, 12 pos
    assert len(got.elastic_u_tensors_a2) == 36
    assert got.elastic_u_symbols == ["C", "C", "C", "C"]
    assert len(got.elastic_u_frac_positions) == 12
    # per-site neutron data carries the CONFIG b_coh (6.646 fm -> 0.6646 sqrt-barn)
    # + sigma_inc (0.001 b), so the C++ honors IRMA, not NCrystal's atom DB.
    assert got.elastic_u_coherent_scatlen_sqrtbarn == pytest.approx([0.6646] * 4)
    assert got.elastic_u_incoherent_xs_barn == pytest.approx([0.001] * 4)
    assert got.elastic_coherent is True
    assert got.elastic_msd_a2 is None            # structure mode

    # the written .ncmat is a complete, self-consistent material:
    ncmat = ncmat_path.read_text()
    assert ncmat_path.name == "graphite.ncmat"
    assert ncmat.startswith("NCMAT v5")
    for sec in ("@CELL", "@ATOMPOSITIONS", "@DYNINFO", "type vdosdebye",
                "@CUSTOM_IRMA"):
        assert sec in ncmat
    assert "graphite__C.irmapack" in ncmat
    # CRITICAL: the NCMAT atom positions must equal the pack's DW-tensor
    # positions (the plugin matches them by fractional position, tol 1e-6).
    pack_pos = np.asarray(got.elastic_u_frac_positions, float).reshape(4, 3)
    apos_lines = ncmat.split("@ATOMPOSITIONS")[1].split("@DYNINFO")[0].strip().splitlines()
    ncmat_pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in apos_lines])
    assert np.allclose(np.sort(pack_pos, axis=0), np.sort(ncmat_pos, axis=0), atol=1e-9)
    # provenance pin present
    assert got.metadata.get("irma_git_sha")
    assert got.metadata.get("phonopy_yaml_sha256")


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_graphite_auto_grid(tmp_path):
    # auto (converged ENDF-style) grid path: omit alpha/beta. Pin freq_max and
    # use small knob overrides so the end-to-end build stays fast (the real
    # defaults build a 300x400 grid; the layout itself is pinned against the ENDF
    # generators in tests/ncrystal/test_provenance_and_config.py).
    d = {
        "material": {
            "phonopy_yaml": str(_GRAPHITE_YAML), "mesh": [2, 2, 2],
            "temperature_K": 296.0,
            "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                            "b_coh_fm": 6.646, "sigma_inc_b": 0.001}]},
        "export": {"material_id": "graphite", "inelastic_mode": 2,
                   "num_directions": 48, "multiphonon_num_directions": 16,
                   "multiphonon_max_order": 2,
                   "freq_max_eV": 0.2, "n_lower": 2, "n_phonon": 6, "n_upper": 2,
                   "alpha_dq_invA": 1.0, "alpha_qcut_invA": 6.0, "alpha_nlog": 4},
    }
    cfg = NCrystalExportConfig.from_dict(d)
    assert cfg.grid_mode == "auto"
    packs, _ncmat = build_packs(cfg, progress=lambda *a: None)
    assert len(packs) == 1
    pack = packs[0]
    # alpha strictly increasing & scaled by AWR; beta starts at 0
    assert pack.alpha_grid[0] > 0 and all(
        b > a for a, b in zip(pack.alpha_grid, pack.alpha_grid[1:]))
    assert pack.beta_grid[0] == 0.0
    assert pack.metadata["grid_mode"] == "auto"
    assert "auto(ENDF)" in pack.metadata["grid_spec"]


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_auto_grid_pack_beta_carries_linlin_cap(tmp_path):
    """NCB-1 end-to-end: a REAL auto-grid export bakes a beta grid whose
    largest step below the recoil ridge respects DELTA_BETA_MAX_LINLIN
    (NCrystal's SAB kernel interpolates S(alpha,beta) linearly in beta).
    Knobs are chosen small but such that the old uncapped tail demonstrably
    violates the cap, so this fails if build.py reverts to the bare builder."""
    from irma.core.constants import BK
    from irma.core.grids import (
        generate_beta_grid, generate_beta_grid_for_iint, DELTA_BETA_MAX_LINLIN,
        grid_reference_temperature_K)
    awr = 11.898
    d = {
        "material": {
            "phonopy_yaml": str(_GRAPHITE_YAML), "mesh": [2, 2, 2],
            "temperature_K": 296.0,
            "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551, "awr": awr,
                            "b_coh_fm": 6.646, "sigma_inc_b": 0.001}]},
        "export": {"material_id": "graphite", "inelastic_mode": 2,
                   "num_directions": 48, "multiphonon_num_directions": 16,
                   "multiphonon_max_order": 2,
                   "freq_max_eV": 0.2, "n_lower": 2, "n_phonon": 6, "n_upper": 4,
                   "beta_max_eV": 1.0,
                   "alpha_dq_invA": 1.0, "alpha_qcut_invA": 6.0, "alpha_nlog": 4},
    }
    cfg = NCrystalExportConfig.from_dict(d)
    assert cfg.grid_mode == "auto"
    packs, _ncmat = build_packs(cfg, progress=lambda *a: None)
    beta_abs = np.asarray(packs[0].beta_grid, float)

    # the pack grid is stored in ABSOLUTE beta = E/kT(296 K); the auto grid is
    # generated in lat=1 THERM units, so everything scales by t_ref/T.
    t_ref = grid_reference_temperature_K(cfg.lat, 296.0)
    scale = t_ref / 296.0
    expected = generate_beta_grid_for_iint(
        0.2, t_ref, iint=1, awr=awr, n_lower=2, n_phonon=6, n_upper=4,
        beta_max_eV=1.0,
        evaluation_temperatures_K=[296.0])
    assert beta_abs == pytest.approx((expected * scale).tolist())

    # the cap property itself, on the pack's own axis. The cap governs the
    # UPPER (post-phonon-region) tail below the recoil ridge; the linear
    # phonon region legitimately keeps its freq_max/n_phonon step.
    kT = BK * t_ref
    lin_end_abs = 0.2 * (1.0 - 1.0 / 6) / kT * scale
    ridge_abs = 4.0 * (1.0 / kT) / awr * scale
    steps = np.diff(beta_abs)
    in_tail = (beta_abs[1:] > lin_end_abs + 1e-9) & (
        beta_abs[1:] <= ridge_abs + 1e-9)
    assert in_tail.sum() > 0
    assert steps[in_tail].max() <= DELTA_BETA_MAX_LINLIN * scale * (1.0 + 1e-9)

    # discriminator: the bare uncapped builder violates the cap on these knobs
    uncapped = generate_beta_grid(0.2, t_ref, n_lower=2, n_phonon=6, n_upper=4,
                                  beta_max_eV=1.0)
    lin_end = 0.2 * (1.0 - 1.0 / 6) / kT
    ridge = 4.0 * (1.0 / kT) / awr
    usteps = np.diff(uncapped)
    u_tail = (uncapped[1:] > lin_end + 1e-9) & (uncapped[1:] <= ridge + 1e-9)
    assert usteps[u_tail].max() > DELTA_BETA_MAX_LINLIN
    assert len(uncapped) != len(expected)


@pytest.mark.skipif(not _BEO_YAML.exists(), reason="BeO fixture absent")
def test_auto_beta_cap_sized_to_lightest_species(monkeypatch):
    """The shared auto beta grid must size its lin-lin recoil-ridge cap to the
    LIGHTEST principal species (smallest AWR -> highest ridge), which covers
    every heavier species in the material. Spies on _auto_beta_grid and aborts
    before the engine runs, so this only exercises the wiring."""
    import irma.ncrystal.build as B

    class _Abort(Exception):
        pass

    captured = {}

    def spy(cfg, t_ref, recoil_awr, progress=print):
        captured["recoil_awr"] = recoil_awr
        raise _Abort

    monkeypatch.setattr(B, "_auto_beta_grid", spy)
    cfg = NCrystalExportConfig.from_dict({
        "material": {
            "phonopy_yaml": str(_BEO_YAML), "mesh": [2, 2, 2],
            "temperature_K": 296.0,
            "scatterers": [
                {"symbol": "Be", "sigma_bound_b": 7.63, "awr": 8.93478,
                 "b_coh_fm": 7.79, "sigma_inc_b": 0.0018},
                {"symbol": "O", "sigma_bound_b": 4.232, "awr": 15.8575,
                 "b_coh_fm": 5.803, "sigma_inc_b": 0.0},
            ],
        },
        "export": {"material_id": "beo", "inelastic_mode": 2,
                   "freq_max_eV": 0.14},
    })
    assert cfg.grid_mode == "auto"
    with pytest.raises(_Abort):
        build_packs(cfg, progress=lambda *a: None)
    assert captured["recoil_awr"] == pytest.approx(8.93478)   # Be, not O


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_freq_max_auto_estimated_from_phonopy():
    # The auto grid derives freq_max from the phonopy mesh when not pinned;
    # graphite's max phonon energy is ~0.2 eV.
    from irma.ncrystal.build import _estimate_freq_max_eV
    fmax = _estimate_freq_max_eV(_graphite_cfg().material)
    assert 0.1 < fmax < 0.4, f"implausible graphite freq_max: {fmax} eV"


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_explicit_site_groups_not_exhausted():
    # Regression for the convention-review F1 flag: the explicit-site_groups
    # path must survive being iterated more than once (it's a list, not a spent
    # generator). resolve_principal_groups loads only the primitive cell.
    from irma.ncrystal.build import resolve_principal_groups
    cfg = _graphite_cfg(site_groups=[[0, 1, 2, 3]])
    groups, site_groups, b_coh, sinc, symbols, positions, lattice = \
        resolve_principal_groups(cfg)
    assert site_groups == [[0, 1, 2, 3]]
    assert len(groups) == 1 and groups[0].symbol == "C"
    # iterate again to prove it is not exhausted
    assert [list(g) for g in site_groups] == [[0, 1, 2, 3]]


@pytest.mark.skipif(not _BEO_YAML.exists(), reason="BeO fixture absent")
def test_beo_two_packs_summed(tmp_path):
    cfg = NCrystalExportConfig.from_dict({
        "material": {
            "phonopy_yaml": str(_BEO_YAML),
            "mesh": [2, 2, 2], "temperature_K": 296.0,
            "scatterers": [
                {"symbol": "Be", "sigma_bound_b": 7.63, "awr": 8.93478,
                 "b_coh_fm": 7.79, "sigma_inc_b": 0.0018},
                {"symbol": "O", "sigma_bound_b": 4.232, "awr": 15.8575,
                 "b_coh_fm": 5.803, "sigma_inc_b": 0.0},
            ],
        },
        "export": {
            "material_id": "beo", "inelastic_mode": 2,
            "num_directions": 32, "multiphonon_num_directions": 16,
            "multiphonon_max_order": 2, "alpha_grid": _ALPHA, "beta_grid": _BETA,
        },
    })
    packs, snippet = build_packs(cfg, progress=lambda *a: None)
    assert len(packs) == 2
    ids = {p.material_id for p in packs}
    assert ids == {"beo__Be", "beo__O"}
    # exactly one pack carries the coherent Bragg (double-count-free, see D1)
    coherent = [p for p in packs if p.elastic_u_tensors_a2]
    assert len(coherent) == 1
    # both packs are valid precomputed_sab kernels on the same grid
    for p in packs:
        assert p.backend == "precomputed_sab"
        assert p.beta_grid[0] == 0.0
        assert len(p.sab_values) == len(p.alpha_grid) * len(p.beta_grid)
    assert snippet.count("pack ") == 2
    # per-atom normalization: each pack's inelastic bound_xs is its species
    # sigma_bound scaled by the atom fraction (BeO is 1:1 -> 0.5 each), so the
    # plugin's weight-1.0 pack sum is per-atom-average, not per-formula-unit.
    by_id = {p.material_id: p for p in packs}
    assert by_id["beo__Be"].bound_xs_barn == pytest.approx(0.5 * 7.63)
    assert by_id["beo__O"].bound_xs_barn == pytest.approx(0.5 * 4.232)


# -- pack-filename collision + coherent double-count guards -------------------

def _beo_cfg(**over):
    d = {
        "material": {
            "phonopy_yaml": str(_BEO_YAML),
            "mesh": [2, 2, 2], "temperature_K": 296.0,
            "scatterers": [
                {"symbol": "Be", "sigma_bound_b": 7.63, "awr": 8.93478,
                 "b_coh_fm": 7.79, "sigma_inc_b": 0.0018},
                {"symbol": "O", "sigma_bound_b": 4.232, "awr": 15.8575,
                 "b_coh_fm": 5.803, "sigma_inc_b": 0.0},
            ],
        },
        "export": {
            "material_id": "beo", "inelastic_mode": 2,
            "num_directions": 32, "multiphonon_num_directions": 16,
            "multiphonon_max_order": 2, "alpha_grid": _ALPHA, "beta_grid": _BETA,
        },
    }
    d["export"].update(over)
    return NCrystalExportConfig.from_dict(d)


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_explicit_site_groups_splitting_species_rejected():
    # H2: graphite's C sites split into two groups -> both groups carry symbol
    # 'C' -> identical pack filename. Must be rejected, not silently overwritten.
    from irma.ncrystal.build import resolve_principal_groups
    cfg = _graphite_cfg(site_groups=[[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="split species"):
        resolve_principal_groups(cfg)


@pytest.mark.skipif(not _BEO_YAML.exists(), reason="BeO fixture absent")
def test_exact_total_multigroup_rejected():
    # H3: 'exact-total' on a 2-species export would give every pack the whole-
    # crystal coherent total and double-count it on summation. Must be rejected.
    cfg = _beo_cfg(coherent_partition_mode="exact-total")
    with pytest.raises(ValueError, match="exact-total"):
        build_packs(cfg, progress=lambda *a: None)


@pytest.mark.skipif(not _GRAPHITE_YAML.exists(), reason="graphite fixture absent")
def test_inelastic_only_pack_carries_the_same_nonzero_sab(tmp_path):
    """Review NC-1: disabling the elastic OUTPUT block must not change the
    inelastic physics. The old zero-sentinel path baked an identically zero
    S(alpha,beta) when the neutron constants were omitted with
    elastic=false; with the constants present (now mandatory), the
    elastic=false pack must carry byte-for-byte the same kernel as the
    elastic=true pack, just without the elastic block."""
    packs_el, _ = build_packs(_graphite_cfg(), pack_path_prefix="el")
    packs_inel, _ = build_packs(_graphite_cfg(elastic=False),
                                pack_path_prefix="inel")
    p_el, p_inel = packs_el[0], packs_inel[0]
    assert max(p_inel.sab_values) > 0.0
    assert p_inel.sab_values == p_el.sab_values
    assert p_inel.alpha_grid == p_el.alpha_grid
    assert p_inel.beta_grid == p_el.beta_grid
    assert p_inel.bound_xs_barn == p_el.bound_xs_barn
    assert not p_inel.elastic_u_tensors_a2      # elastic block genuinely absent
    assert p_el.elastic_u_tensors_a2            # ...and present when requested


def test_extra_scatterer_row_is_an_error():
    """A scatterer for a species the structure lacks used to be SILENTLY
    ignored (review NC-2)."""
    from irma.ncrystal.build import resolve_principal_groups
    d = {
        "material": {
            "phonopy_yaml": str(_GRAPHITE_YAML), "mesh": [2, 2, 2],
            "temperature_K": 296.0,
            "scatterers": [
                {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                 "b_coh_fm": 6.646, "sigma_inc_b": 0.001},
                {"symbol": "Xe", "sigma_bound_b": 1.0, "awr": 130.0,
                 "b_coh_fm": 4.92, "sigma_inc_b": 0.0},
            ],
        },
        "export": {"material_id": "graphite", "inelastic_mode": 2,
                   "alpha_grid": _ALPHA, "beta_grid": _BETA},
    }
    cfg = NCrystalExportConfig.from_dict(d)
    if not _GRAPHITE_YAML.exists():
        pytest.skip("graphite fixture absent")
    with pytest.raises(ValueError, match="Xe.*does not contain|does not contain.*Xe"):
        resolve_principal_groups(cfg)


def test_attach_elastic_raises_on_missing_state():
    """elastic=true is a hard output contract: warn-and-continue used to
    write an inelastic-only pack that claimed success (review NC-3)."""
    from irma.ncrystal.build import _attach_elastic
    with pytest.raises(RuntimeError, match="no\\s+elastic_state|no elastic_state"):
        _attach_elastic(object(), None, 2, {}, progress=lambda *_: None)


def test_attach_elastic_raises_on_site_mismatch():
    from irma.ncrystal.build import _attach_elastic
    state = {
        "thermal_displacement_matrices_ang2": [[[0.005, 0, 0],
                                                [0, 0.005, 0],
                                                [0, 0, 0.005]]],
        "primitive_symbols": ["C"],
        "primitive_scaled_positions": [[0.0, 0.0, 0.0]],
    }
    with pytest.raises(RuntimeError, match="1 sites.*primitive has 2|mismatched"):
        _attach_elastic(object(), state, 2, {}, progress=lambda *_: None)
