"""Force-constants discovery for the phonopy-backed paths (iel=10 modes 1/2).

The MT4 driver used to require a text FORCE_CONSTANTS next to phonopy.yaml
while the MT2 loader accepted force_constants.hdf5 and FORCE_SETS too, so a
valid hdf5-based model worked for MT2 but errored in MT4. Both now share
resolve_force_constants_source: yaml-embedded force constants first (they
outrank every file inside phonopy.load), then force_constants.hdf5,
FORCE_CONSTANTS, FORCE_SETS next to the yaml — and a hard error when nothing
exists, instead of phonopy's fallback of searching the process cwd.
"""
import numpy as np
import pytest

import irma.core.noncubic_inelastic as nci
from irma.core.noncubic_inelastic import NoncubicInelasticControls
from irma.core.phonopy_io import (
    resolve_force_constants_source, phonopy_yaml_embeds_force_constants,
)
from irma.core.standalone_sab import (
    run_noncubic_standalone_sab, _irma_grid_to_physical_qe,
)

_ALPHA = np.array([0.1, 0.5, 1.0, 2.0])
_BETA = np.array([0.0, 0.5, 1.0, 3.0, 6.0])
_LAT, _TEMP, _AWR = 1, 296.0, 11.9

_CONTROLS = NoncubicInelasticControls(
    num_directions=10,
    multiphonon_num_directions=10,
    multiphonon_max_order=1,
)


def _yaml(tmp_path, embed_fc=False):
    body = "phonopy:\n  version: 2.48\nlattice:\n- [1, 0, 0]\n"
    if embed_fc:
        body += "force_constants:\n  format: full\n"
    p = tmp_path / "phonopy.yaml"
    p.write_text(body)
    return p


# ---------- resolver unit behavior ----------

def test_priority_hdf5_over_text_over_sets(tmp_path):
    yaml = _yaml(tmp_path)
    (tmp_path / "FORCE_SETS").write_text("#\n")
    assert resolve_force_constants_source(yaml) == {
        "force_sets_filename": str(tmp_path / "FORCE_SETS")}
    (tmp_path / "FORCE_CONSTANTS").write_text("#\n")
    assert resolve_force_constants_source(yaml) == {
        "force_constants_filename": str(tmp_path / "FORCE_CONSTANTS")}
    (tmp_path / "force_constants.hdf5").write_bytes(b"\x89HDF")
    assert resolve_force_constants_source(yaml) == {
        "force_constants_filename": str(tmp_path / "force_constants.hdf5")}


def test_embedded_outranks_files(tmp_path):
    """phonopy.load uses yaml-embedded FC even over an explicit filename,
    so the resolver must report the embedded source, not the file."""
    yaml = _yaml(tmp_path, embed_fc=True)
    (tmp_path / "FORCE_CONSTANTS").write_text("#\n")
    assert phonopy_yaml_embeds_force_constants(yaml)
    assert resolve_force_constants_source(yaml) == {}


def test_nothing_found_is_loud(tmp_path):
    yaml = _yaml(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_force_constants_source(yaml)
    msg = str(exc.value)
    for frag in ("force_constants.hdf5", "FORCE_CONSTANTS", "FORCE_SETS",
                 "embedded"):
        assert frag in msg


# ---------- through the MT4 standalone driver (stubbed compute) ----------

@pytest.fixture
def stubbed_driver(monkeypatch):
    calls = {"context_args": [], "inprocess_kwargs": []}

    def fake_build_compute_context(args, q_grid_ang_inv, e_grid_mev,
                                   preloaded_full_mesh=None):
        calls["context_args"].append(args)
        return {"stub": True}

    def fake_run_inprocess(**kwargs):
        calls["inprocess_kwargs"].append(kwargs)
        _, _, alpha_abs, beta_abs = _irma_grid_to_physical_qe(
            _ALPHA, _BETA, _LAT, _TEMP, _AWR)
        return {
            "output_arrays": {
                "alpha": alpha_abs,
                "beta_downscatter_abs": beta_abs,
                "sab_asym_downscatter_incoherent_approx_n1_term":
                    np.zeros((len(alpha_abs), len(beta_abs))),
            },
            "metadata": {"multiphonon_max_order": 1},
        }

    monkeypatch.setattr(nci, "build_compute_context", fake_build_compute_context)
    monkeypatch.setattr(nci, "run_noncubic_sab_inprocess", fake_run_inprocess)
    return calls


def _run(yaml):
    return run_noncubic_standalone_sab(
        alpha=_ALPHA, beta=_BETA, lat=_LAT, temperature_k=_TEMP, awr=_AWR,
        phonopy_yaml_path=str(yaml), mesh_dim=[4, 4, 4],
        num_jobs=1, inelastic_mode=1, controls=_CONTROLS, context_cache={},
    )


def test_hdf5_only_model_reaches_driver(tmp_path, stubbed_driver):
    yaml = _yaml(tmp_path)
    (tmp_path / "force_constants.hdf5").write_bytes(b"\x89HDF")
    _run(yaml)
    args = stubbed_driver["context_args"][0]
    assert args.force_constants.endswith("force_constants.hdf5")
    assert args.force_sets is None
    kw = stubbed_driver["inprocess_kwargs"][0]
    assert str(kw["force_constants"]).endswith("force_constants.hdf5")
    assert kw["force_sets"] is None


def test_force_sets_only_model_reaches_driver(tmp_path, stubbed_driver):
    yaml = _yaml(tmp_path)
    (tmp_path / "FORCE_SETS").write_text("#\n")
    _run(yaml)
    args = stubbed_driver["context_args"][0]
    assert args.force_constants is None
    assert args.force_sets.endswith("FORCE_SETS")


def test_embedded_fc_model_reaches_driver(tmp_path, stubbed_driver):
    yaml = _yaml(tmp_path, embed_fc=True)
    _run(yaml)
    args = stubbed_driver["context_args"][0]
    assert args.force_constants is None
    assert args.force_sets is None


def test_no_force_constants_fails_before_compute(tmp_path, stubbed_driver):
    yaml = _yaml(tmp_path)
    with pytest.raises(FileNotFoundError, match="No force constants"):
        _run(yaml)
    assert not stubbed_driver["context_args"]
