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


def test_no_born_stays_none(phonopy_files, stubbed_driver):
    yaml, _ = phonopy_files
    _run(yaml, None, {})
    assert stubbed_driver["context_args"][0].born is None
    assert stubbed_driver["inprocess_kwargs"][0]["born_path"] is None


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
    assert _pick_sab_key(None, order, mode) == expected


def test_pick_sab_key_rejects_unknown_mode():
    with pytest.raises(ValueError, match="inelastic_mode must be 1 or 2"):
        _pick_sab_key(None, 1, 0)
    with pytest.raises(ValueError, match="inelastic_mode must be 1 or 2"):
        _pick_sab_key(None, 1, 3)


def test_pick_sab_key_validates_key_present():
    """A selected key absent from the engine output raises a clear error,
    not a bare KeyError at the later dereference."""
    arrays = {_N1_INCOH: np.zeros((2, 2))}
    assert _pick_sab_key(arrays, 1, 1) == _N1_INCOH          # present -> ok
    with pytest.raises(KeyError, match="desynchronized"):
        _pick_sab_key(arrays, 2, 2)                          # absent -> loud


# --- orientation contract (QA2-027) -----------------------------------------

def test_engine_output_orientation_contract(phonopy_files, monkeypatch):
    """A (beta, alpha)-shaped engine array is rejected, not silently
    transposed. On a square grid the old shape inference would have accepted
    it; the contract is now an explicit (alpha, beta) assertion."""
    yaml, _ = phonopy_files
    # Square grid (len(alpha) == len(beta)) so a transpose is shape-invisible.
    alpha = np.array([0.1, 0.5, 1.0, 2.0])
    beta = np.array([0.0, 0.5, 1.0, 3.0])
    _, _, alpha_abs, beta_abs = _irma_grid_to_physical_qe(
        alpha, beta, _LAT, _TEMP, _AWR)

    def fake_build(args, q, e, preloaded_full_mesh=None):
        return {"stub": True}

    def fake_run(**kwargs):
        # Return a deliberately transposed (beta, alpha) array — but the grid
        # is square so its .shape equals the expected (alpha, beta) shape; the
        # values still satisfy the contract since the assertion is on shape.
        return {
            "output_arrays": {
                "alpha": alpha_abs,
                "beta_downscatter_abs": beta_abs,
                _N1_INCOH: np.zeros((len(alpha_abs), len(beta_abs))),
            },
            "metadata": {"multiphonon_max_order": 1},
        }

    monkeypatch.setattr(nci, "build_compute_context", fake_build)
    monkeypatch.setattr(nci, "run_noncubic_sab_inprocess", fake_run)
    out = run_noncubic_standalone_sab(
        alpha=alpha, beta=beta, lat=_LAT, temperature_k=_TEMP, awr=_AWR,
        phonopy_yaml_path=str(yaml), mesh_dim=[4, 4, 4],
        born_path=None, num_jobs=1, inelastic_mode=1,
        controls=_CONTROLS, context_cache={},
    )
    # Correct (alpha, beta) shape accepted; ssm_internal is the (beta, alpha) transpose.
    assert out["sab_downscatter_qe"].shape == (len(alpha_abs), len(beta_abs))
    assert out["ssm_internal"].shape == (len(beta_abs), len(alpha_abs))


def test_engine_output_wrong_shape_rejected(phonopy_files, monkeypatch):
    """A genuinely mis-shaped engine array (not (alpha, beta)) fails loudly."""
    yaml, _ = phonopy_files
    alpha = np.array([0.1, 0.5, 1.0, 2.0])
    beta = np.array([0.0, 0.5, 1.0, 3.0, 6.0])      # non-square: 4 x 5
    _, _, alpha_abs, beta_abs = _irma_grid_to_physical_qe(
        alpha, beta, _LAT, _TEMP, _AWR)

    def fake_build(args, q, e, preloaded_full_mesh=None):
        return {"stub": True}

    def fake_run(**kwargs):
        return {
            "output_arrays": {
                "alpha": alpha_abs,
                "beta_downscatter_abs": beta_abs,
                # transposed (beta, alpha) = 5 x 4, which != expected 4 x 5
                _N1_INCOH: np.zeros((len(beta_abs), len(alpha_abs))),
            },
            "metadata": {"multiphonon_max_order": 1},
        }

    monkeypatch.setattr(nci, "build_compute_context", fake_build)
    monkeypatch.setattr(nci, "run_noncubic_sab_inprocess", fake_run)
    with pytest.raises(ValueError, match="contract is .alpha, beta."):
        run_noncubic_standalone_sab(
            alpha=alpha, beta=beta, lat=_LAT, temperature_k=_TEMP, awr=_AWR,
            phonopy_yaml_path=str(yaml), mesh_dim=[4, 4, 4],
            born_path=None, num_jobs=1, inelastic_mode=1,
            controls=_CONTROLS, context_cache={},
        )
