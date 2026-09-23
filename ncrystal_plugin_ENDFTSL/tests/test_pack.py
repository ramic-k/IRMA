import pytest
from ncrystal_plugin_ENDFTSL.pack import ENDFTSLPack, write_pack, read_pack, MAGIC


def _good_pack(**overrides):
    kw = dict(
        material_id="graphite", temperature_K=296.0, bound_xs_barn=5.551,
        element_mass_amu=12.0107, sab_representation="scaled_sym_sab",
        alpha_grid=[0.1, 0.2], beta_grid=[0.0, 0.5],
        sab_values=[1.0, 2.0, 3.0, 4.0],          # beta-major: len == n_alpha * n_beta
        coh_edges_ev=[1e-3, 2e-3], coh_cumS=[0.0, 0.4],
        elastic_msd_a2=0.005, elastic_incoherent_xs_barn=0.001, elastic_scale=1.0,
        metadata={"source_endf_sha256": "abc"})
    kw.update(overrides)
    return ENDFTSLPack(**kw)


def test_roundtrip(tmp_path):
    p = _good_pack()
    f = tmp_path / "g.endftslpack"
    write_pack(p, f)
    assert f.read_text().splitlines()[0] == MAGIC
    q = read_pack(f)
    assert q.material_id == "graphite" and q.coh_edges_ev == [1e-3, 2e-3]
    assert q.sab_values == [1.0, 2.0, 3.0, 4.0]
    assert q.elastic_msd_a2 == 0.005 and q.metadata["source_endf_sha256"] == "abc"


def test_reader_rejects_bad_magic(tmp_path):
    f = tmp_path / "bad.endftslpack"
    f.write_text("NOPE\n")
    with pytest.raises(ValueError, match="magic"):
        read_pack(f)


# --- write/read validation --------------------------------------------------
# (field-name pattern in the message, {constructor overrides}) per malformed case
_MALFORMED = [
    ("temperature_K", {"temperature_K": -10.0}),
    ("alpha_grid", {"alpha_grid": [0.2, 0.1]}),                 # unsorted
    ("sab_values", {"sab_values": [1.0, 2.0, 3.0]}),            # length mismatch
    ("sab_values", {"sab_values": [1.0, -2.0, 3.0, 4.0]}),      # negative
    ("coh_cumulative_s", {"coh_cumS": [0.4, 0.0]}),             # decreasing
    ("coh_edges_ev", {"coh_edges_ev": [2e-3, 1e-3]}),           # unsorted
]


@pytest.mark.parametrize("fieldname,overrides", _MALFORMED,
                         ids=[f"{f}-{i}" for i, (f, _) in enumerate(_MALFORMED)])
def test_write_pack_rejects_malformed(tmp_path, fieldname, overrides):
    f = tmp_path / "bad.endftslpack"
    with pytest.raises(ValueError, match=fieldname):
        write_pack(_good_pack(**overrides), f)
    assert not f.exists(), "write_pack must not leave a file behind on rejection"


def test_read_pack_validates(tmp_path):
    f = tmp_path / "g.endftslpack"
    write_pack(_good_pack(), f)
    text = f.read_text()
    assert "temperature_K = 296\n" in text
    f.write_text(text.replace("temperature_K = 296\n", "temperature_K = -10\n"))
    with pytest.raises(ValueError, match="temperature_K"):
        read_pack(f)
