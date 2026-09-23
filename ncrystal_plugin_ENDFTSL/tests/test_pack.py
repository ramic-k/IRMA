import math

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


# --- S6: shared write/read validation --------------------------------------
# (field-name pattern in the message, {constructor overrides}) per malformed case
_MALFORMED = [
    ("temperature_K", {"temperature_K": math.nan}),
    ("temperature_K", {"temperature_K": -10.0}),
    ("element_mass_amu", {"element_mass_amu": 0.0}),
    ("bound_xs_barn", {"bound_xs_barn": -5.551}),
    ("alpha_grid", {"alpha_grid": [0.2, 0.1]}),                 # unsorted
    ("alpha_grid", {"alpha_grid": [0.1]}),                      # too short
    ("beta_grid", {"beta_grid": [0.0, math.inf]}),              # non-finite
    ("sab_values", {"sab_values": [1.0, 2.0, 3.0]}),            # length mismatch
    ("sab_values", {"sab_values": [1.0, -2.0, 3.0, 4.0]}),      # negative
    ("sab_values", {"sab_values": [1.0, math.inf, 3.0, 4.0]}),  # Inf
    ("coh_cumulative_s", {"coh_cumS": [0.0]}),                  # length mismatch
    ("coh_cumulative_s", {"coh_cumS": [0.4, 0.0]}),             # decreasing
    ("coh_cumulative_s", {"coh_cumS": [-0.1, 0.4]}),            # negative
    ("coh_edges_ev", {"coh_edges_ev": [2e-3, 1e-3]}),           # unsorted
]


@pytest.mark.parametrize("fieldname,overrides", _MALFORMED,
                         ids=[f"{f}-{i}" for i, (f, _) in enumerate(_MALFORMED)])
def test_write_pack_rejects_malformed(tmp_path, fieldname, overrides):
    f = tmp_path / "bad.endftslpack"
    with pytest.raises(ValueError, match=fieldname.replace(".", r"\.")):
        write_pack(_good_pack(**overrides), f)
    assert not f.exists(), "write_pack must not leave a file behind on rejection"


def _write_tampered(tmp_path, replacements):
    """Serialize a good pack bypassing write_pack validation, then rewrite whole
    key = value lines (craft the file bytes directly for the read side)."""
    good = tmp_path / "good.endftslpack"
    write_pack(_good_pack(), good)
    lines = good.read_text(encoding="utf-8").splitlines()
    keys = dict(replacements)
    out = []
    for ln in lines:
        key = ln.split("=", 1)[0].strip() if "=" in ln else None
        out.append(keys.pop(key) if key in keys else ln)
    out.extend(keys.values())          # replacements for keys absent from the file
    bad = tmp_path / "bad.endftslpack"
    bad.write_text("\n".join(out) + "\n", encoding="utf-8")
    return bad


# read-side: same failure classes, crafted directly in the file bytes
_TAMPERED = [
    ("temperature_K", {"temperature_K": "temperature_K = nan"}),
    ("bound_xs_barn", {"bound_xs_barn": "bound_xs_barn = -5.551"}),
    ("alpha_grid", {"alpha_grid": "alpha_grid = 0.2 0.1"}),
    ("alpha_grid", {"alpha_grid": "alpha_grid = 0.1"}),
    ("beta_grid", {"beta_grid": "beta_grid = 0.0 inf"}),
    ("sab_values", {"sab_values": "sab_values = 1 2 3"}),
    ("sab_values", {"sab_values": "sab_values = 1 -2 3 4"}),
    ("sab_values", {"sab_values": "sab_values = 1 inf 3 4"}),
    ("coh_cumulative_s", {"coh_cumulative_s": "coh_cumulative_s = 0.4"}),
    ("coh_cumulative_s", {"coh_cumulative_s": "coh_cumulative_s = 0.4 0.0"}),
    ("coh_edges_ev", {"coh_edges_ev": "coh_edges_ev = 2e-3 1e-3"}),
]


@pytest.mark.parametrize("fieldname,replacements", _TAMPERED,
                         ids=[f"{f}-{i}" for i, (f, _) in enumerate(_TAMPERED)])
def test_read_pack_rejects_malformed(tmp_path, fieldname, replacements):
    bad = _write_tampered(tmp_path, replacements)
    with pytest.raises(ValueError, match=fieldname.replace(".", r"\.")):
        read_pack(bad)
