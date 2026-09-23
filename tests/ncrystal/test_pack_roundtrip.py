"""Format round-trip + validation guards for the .irmapack container (SP1).

Pure/fast: no engine, no phonopy.
"""
from __future__ import annotations

import pytest

from irma.ncrystal.pack import (
    IRMAPack,
    write_pack,
    read_pack,
    MAGIC,
    SCHEMA_VERSION,
)


def _minimal_pack(**over) -> IRMAPack:
    kw = dict(
        material_id="graphite__C",
        backend="precomputed_sab",
        sab_representation="scaled_sym_sab",
        temperature_K=296.0,
        bound_xs_barn=5.551,
        element_mass_amu=12.011,
        alpha_grid=[0.1, 0.2, 0.4],
        beta_grid=[0.0, 0.5, 1.0],
        sab_values=[float(i) * 0.01 for i in range(9)],
        metadata={"irma_git_sha": "abc1234", "mesh": "40x40x40"},
    )
    kw.update(over)
    return IRMAPack(**kw)


@pytest.mark.parametrize("over, match", [
    (dict(temperature_K=-1.0), "temperature_K must be finite and positive"),
    (dict(bound_xs_barn=0.0), "bound_xs_barn must be finite and positive"),
    (dict(element_mass_amu=float("nan")), "element_mass_amu must be finite and positive"),
    (dict(alpha_grid=[0.1, float("inf"), 0.4]), "alpha_grid must contain only finite"),
    (dict(sab_values=[float("nan")] + [0.0] * 8), "sab_values must be finite and non-negative"),
    (dict(sab_values=[-0.1] + [0.0] * 8), "sab_values must be finite and non-negative"),
    (dict(sab_values=[0.0] * 9), "identically zero"),
])
def test_precomputed_sab_rejects_nonfinite_or_nonpositive(tmp_path, over, match):
    """Hardening (Wave B): a hand-assembled pack with NaN/Inf grids/SAB, a
    negative SAB, or a non-positive T/sigma/mass is rejected at write/read time."""
    with pytest.raises(ValueError, match=match):
        write_pack(_minimal_pack(**over), tmp_path / "bad.irmapack")


def test_write_read_roundtrip_exact(tmp_path):
    pack = _minimal_pack()
    path = tmp_path / "p.irmapack"
    write_pack(pack, path)
    got = read_pack(path)
    assert got.material_id == pack.material_id
    assert got.backend == pack.backend
    assert got.sab_representation == pack.sab_representation
    assert got.temperature_K == pytest.approx(pack.temperature_K)
    assert got.bound_xs_barn == pytest.approx(pack.bound_xs_barn)
    assert got.element_mass_amu == pytest.approx(pack.element_mass_amu)
    assert got.alpha_grid == pytest.approx(pack.alpha_grid)
    assert got.beta_grid == pytest.approx(pack.beta_grid)
    assert got.sab_values == pytest.approx(pack.sab_values)
    assert got.metadata == pack.metadata


def test_header_is_v2_magic(tmp_path):
    path = tmp_path / "p.irmapack"
    write_pack(_minimal_pack(), path)
    lines = path.read_text().splitlines()
    assert lines[0] == MAGIC
    assert lines[1] == f"schema_version = {SCHEMA_VERSION}"


def test_v1_pack_now_rejected(tmp_path):
    # v1 was a pre-release scaffold layout IRMA never shipped a real pack for;
    # only the current schema (v2) is accepted on read now.
    path = tmp_path / "v1.irmapack"
    body = "\n".join([
        MAGIC,
        "schema_version = 1",
        "material_id = old__C",
        "backend = precomputed_sab",
        "units = angstrom_meV_barn_K",
        "sab_representation = scaled_sym_sab",
        "temperature_K = 296",
        "bound_xs_barn = 5.551",
        "element_mass_amu = 12.011",
        "alpha_grid = 0.1 0.2 0.4",
        "beta_grid = 0 0.5 1.0",
        "sab_values = " + " ".join(str(i * 0.01) for i in range(9)),
    ]) + "\n"
    path.write_text(body)
    with pytest.raises(ValueError, match="schema_version"):
        read_pack(path)


def test_unsupported_schema_rejected(tmp_path):
    path = tmp_path / "v9.irmapack"
    path.write_text(f"{MAGIC}\nschema_version = 9\nmaterial_id = x\n"
                    "backend = precomputed_sab\n")
    with pytest.raises(ValueError, match="schema_version"):
        read_pack(path)


def test_bad_magic_rejected(tmp_path):
    path = tmp_path / "bad.irmapack"
    path.write_text("NOT_A_PACK\nmaterial_id = x\n")
    with pytest.raises(ValueError, match="magic"):
        read_pack(path)


def test_scaled_sym_requires_beta_zero_start():
    with pytest.raises(ValueError, match="beta_grid must start at zero"):
        from irma.ncrystal.pack import _validate
        _validate(_minimal_pack(beta_grid=[0.5, 1.0, 1.5]))


def test_sab_values_length_checked():
    with pytest.raises(ValueError, match="sab_values length"):
        from irma.ncrystal.pack import _validate
        _validate(_minimal_pack(sab_values=[0.0, 1.0]))


def test_per_site_neutron_data_round_trips(tmp_path):
    # the per-tensor-site b_coh (sqrt-barn) + sigma_inc (barn) survive write -> read.
    pack = _minimal_pack(
        elastic_u_tensors_a2=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 2,
        elastic_u_symbols=["C", "O"],
        elastic_u_frac_positions=[0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
        elastic_u_coherent_scatlen_sqrtbarn=[0.6646, 0.5803],
        elastic_u_incoherent_xs_barn=[0.001, 0.0],
    )
    p = tmp_path / "rt.irmapack"
    write_pack(pack, p)
    got = read_pack(p)
    assert got.elastic_u_coherent_scatlen_sqrtbarn == [0.6646, 0.5803]
    assert got.elastic_u_incoherent_xs_barn == [0.001, 0.0]


def test_alpha_grid_must_be_strictly_increasing():
    from irma.ncrystal.pack import _validate
    with pytest.raises(ValueError, match="alpha_grid must be strictly increasing"):
        _validate(_minimal_pack(alpha_grid=[0.2, 0.1, 0.4]))


def test_beta_grid_must_be_strictly_increasing():
    from irma.ncrystal.pack import _validate
    # still starts at 0 (passes the scaled_sym start check) but is out of order
    with pytest.raises(ValueError, match="beta_grid must be strictly increasing"):
        _validate(_minimal_pack(beta_grid=[0.0, 1.0, 0.5]))


def _tensor_pack(**over) -> IRMAPack:
    kw = dict(
        elastic_u_tensors_a2=[0.00226, 0.0, 0.0,
                              0.0, 0.00226, 0.0,
                              0.0, 0.0, 0.0149],
        elastic_u_symbols=["C"],
        elastic_u_frac_positions=[0.0, 0.0, 0.0],
        elastic_u_coherent_scatlen_sqrtbarn=[0.6646],
        elastic_u_incoherent_xs_barn=[0.001],
    )
    kw.update(over)
    return _minimal_pack(**kw)


def test_directional_mode_roundtrips(tmp_path):
    pack = _tensor_pack(incoherent_elastic_mode="directional")
    path = tmp_path / "dir.irmapack"
    write_pack(pack, path)
    lines = path.read_text().splitlines()
    assert lines[1] == f"schema_version = {SCHEMA_VERSION}"
    assert "incoherent_elastic_mode = directional" in lines
    got = read_pack(path)
    assert got.incoherent_elastic_mode == "directional"
    assert got.elastic_u_tensors_a2 == pytest.approx(pack.elastic_u_tensors_a2)


def test_default_tensor_pack_has_no_mode_line(tmp_path):
    """Default (isotropic) packs are byte-identical to pre-feature packs."""
    path = tmp_path / "iso.irmapack"
    write_pack(_tensor_pack(), path)
    text = path.read_text()
    assert "incoherent_elastic_mode" not in text
    assert read_pack(path).incoherent_elastic_mode == "isotropic"
