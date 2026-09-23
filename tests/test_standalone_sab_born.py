"""BORN/NAC plumbing through the in-process noncubic MT4 driver.

The deck's Card 6f BORN path used to reach only the MT2 directional
Debye-Waller factors (load_phonopy_mesh); run_noncubic_standalone_sab had no
born parameter at all, so the MT4 mode sums were silently built without NAC
and the context cache could not distinguish NAC on/off. These tests pin the
plumbing without needing phonopy: the context builder and compute driver are
stubbed, and only the argument/caching behavior is asserted.
"""
import numpy as np
import pytest

import irma.core.noncubic_inelastic as nci
import irma.core.noncubic_inelastic_context as ncc
from irma.core.noncubic_inelastic import NoncubicInelasticControls
from irma.core.standalone_sab import (
    run_noncubic_standalone_sab, _irma_grid_to_physical_qe, _pick_sab_key,
)

_ALPHA = np.array([0.1, 0.5, 1.0, 2.0])
_BETA = np.array([0.0, 0.5, 1.0, 3.0, 6.0])
_LAT, _TEMP, _AWR = 1, 296.0, 11.9

_CONTROLS = NoncubicInelasticControls(
    num_directions=10,
    multiphonon_num_directions=10,
    multiphonon_max_order=1,
)


@pytest.fixture
def phonopy_files(tmp_path):
    yaml = tmp_path / "phonopy.yaml"
    yaml.write_text("# stub\n")
    (tmp_path / "FORCE_CONSTANTS").write_text("# stub\n")
    born = tmp_path / "BORN"
    born.write_text("# stub\n")
    return yaml, born


@pytest.fixture
def stubbed_driver(monkeypatch):
    """Replace the context builder and compute driver with recorders."""
    calls = {"context_args": [], "inprocess_kwargs": []}

    def fake_build_compute_context(args, q_grid_ang_inv, e_grid_mev,
                                   preloaded_full_mesh=None, model_context=None):
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

    monkeypatch.setattr(ncc, "build_compute_context", fake_build_compute_context)
    monkeypatch.setattr(nci, "run_noncubic_sab_inprocess", fake_run_inprocess)
    return calls


def _run(yaml, born_path, cache):
    return run_noncubic_standalone_sab(
        alpha=_ALPHA, beta=_BETA, lat=_LAT, temperature_k=_TEMP, awr=_AWR,
        phonopy_yaml_path=str(yaml), mesh_dim=[4, 4, 4],
        born_path=born_path, num_jobs=1, inelastic_mode=1,
        controls=_CONTROLS, context_cache=cache,
    )


def test_born_path_reaches_context_and_driver(phonopy_files, stubbed_driver):
    yaml, born = phonopy_files
    _run(yaml, str(born), {})
    args = stubbed_driver["context_args"][0]
    assert args.born == str(born.resolve())
    kwargs = stubbed_driver["inprocess_kwargs"][0]
    assert str(kwargs["born_path"]) == str(born.resolve())
    _run(yaml, None, {})                                   # no BORN stays None
    assert stubbed_driver["context_args"][1].born is None
    assert stubbed_driver["inprocess_kwargs"][1]["born_path"] is None


def test_context_cache_distinguishes_nac(phonopy_files, stubbed_driver):
    """NAC on/off must never share a cached context; identical settings must."""
    yaml, born = phonopy_files
    cache = {}
    _run(yaml, None, cache)
    _run(yaml, str(born), cache)        # different key -> second build
    _run(yaml, str(born), cache)        # same key -> cache hit
    assert len(stubbed_driver["context_args"]) == 2
    # Size-one eviction policy: building the NAC-on
    # context evicted the stale NAC-off entry, so exactly the current key
    # remains (the stubbed builder returns no "_model_context", so no
    # model-layer entry is cached alongside it here).
    assert len(cache) == 1


def test_missing_born_file_fails_loudly(phonopy_files, stubbed_driver):
    yaml, born = phonopy_files
    with pytest.raises(FileNotFoundError, match="BORN"):
        _run(yaml, str(born) + ".missing", {})
    assert not stubbed_driver["context_args"]   # failed before any build


def test_phonopy_yaml_embeds_nac_probe(tmp_path):
    """The cheap top-level 'nac:' scan that decides phonopy.load's is_nac.
    NAC embedded in the deck-named phonopy.yaml is honored; phonopy's
    auto-read of ./BORN from the process cwd must never decide the physics."""
    from irma.core.phonopy_io import phonopy_yaml_embeds_nac
    import gzip

    plain = tmp_path / "phonopy.yaml"
    plain.write_text("phonopy:\n  version: 2.48\nlattice:\n- [1, 0, 0]\n")
    assert not phonopy_yaml_embeds_nac(plain)

    with_nac = tmp_path / "phonopy_nac.yaml"
    with_nac.write_text(
        "phonopy:\n  version: 2.48\nnac:\n  born_effective_charge:\n  - [1]\n")
    assert phonopy_yaml_embeds_nac(with_nac)

    # indented 'nac:' (not top-level) must not count
    nested = tmp_path / "phonopy_nested.yaml"
    nested.write_text("phonopy:\n  nac: something\n")
    assert not phonopy_yaml_embeds_nac(nested)

    # older flat layout (no nac: wrapper) that phonopy still parses
    legacy = tmp_path / "phonopy_legacy.yaml"
    legacy.write_text(
        "phonopy:\n  version: 1.x\nborn_effective_charge:\n- [1]\n"
        "dielectric_constant:\n- [2]\n")
    assert phonopy_yaml_embeds_nac(legacy)

    gz = tmp_path / "phonopy.yaml.gz"
    with gzip.open(gz, "wt") as f:
        f.write("phonopy:\n  version: 2.48\nnac:\n  unit_conversion: 14.4\n")
    assert phonopy_yaml_embeds_nac(gz)


# --- _pick_sab_key mode/order mapping (QA2-028) -----------------------------

_N1_INCOH = "sab_asym_downscatter_incoherent_approx_n1_term"
_N1_INCOH_MP = (
    "sab_asym_downscatter_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon")
_ONE_PHONON = "sab_asym_downscatter_one_phonon_total"
_ONE_PHONON_MP = (
    "sab_asym_downscatter_one_phonon_total_plus_incoherent_approx_multiphonon")


@pytest.mark.parametrize("mode, order, expected", [
    (1, 1, _N1_INCOH),
    (1, 2, _N1_INCOH_MP),
    (2, 1, _ONE_PHONON),
    (2, 2, _ONE_PHONON_MP),
])
def test_pick_sab_key_mapping(mode, order, expected):
    """The key is a pure function of (inelastic_mode, effective order)."""
    assert _pick_sab_key(order, mode) == expected
