"""Fast CI coverage for the lat=0 noncubic engine path (modes 1/2).

Every other mode-1/2 engine test runs a lat=1 deck, so the lat=0 branch —
the 'physical-qe' grid_key in run_noncubic_standalone_sab and the sc=1 path
of _irma_grid_to_physical_qe (irma/core/standalone_sab.py), where the deck
alpha/beta are already in kT(T) units — never executed. lat=0 is a fully
supported Card 7 input (driver.py accepts lat in (0, 1)).

This reuses the fast-CI graphite deck (mesh 4^3, ndir=40, mpdir=20) with
lat=0 and the alpha/beta grids pre-multiplied by sc = THERM/kT(296) using
the engine's own constants. Because Python float repr round-trips exactly,
the engine then sees BIT-IDENTICAL physical Q/E grids in both conventions,
so lat=0 must reproduce the lat=1 law values exactly — the only legitimate
tape differences are the LAT flag and the stored (deck-unit) grids. That
turns the lat=1 pins into the lat=0 pins and directly guards the contract
that lat changes grid UNITS only, never the physics.

The pins are characterization values (current behavior, frozen from two
identical runs), not external ground truth; physics-level validation lives
in the dedicated harnesses.
"""
import os
import tempfile

import pytest

pytest.importorskip("phonopy")

from irma.core.engine import run_leapr  # noqa: E402
from irma.core.constants import BK, THERM  # noqa: E402

# Shared fast-CI machinery: same vendored graphite model, deck template,
# MT4 reducer, and the lat=1 frozen pins this file mirrors.
from test_noncubic_fast_ci import _DECK, _PINS, _TEFF0, _YAML, _mt4_stats  # noqa: E402 isort:skip

_TEMPERATURE_K = 296.0
_SC = THERM / (BK * _TEMPERATURE_K)            # ~0.99187: lat=1 -> kT units
_ALPHA_LAT1 = (0.1, 0.4, 1.0, 2.5, 6.0, 15.0)
_BETA_LAT1 = (0.0, 0.4, 1.0, 2.0, 3.5, 5.5, 8.0, 12.0)


def _lat0_deck(mode):
    """The fast-CI deck with lat=0 and grids converted to kT(296) units.

    repr() round-trips floats exactly, so alpha_deck = alpha_lat1 * sc parsed
    back from the deck text equals the product bit-for-bit and the physical
    grids match the lat=1 run exactly.
    """
    base = _DECK.format(mode=mode, yaml=_YAML)
    deck = base.replace("6 8 1/", "6 8 0/").replace(
        "0.1 0.4 1.0 2.5 6.0 15.0/",
        " ".join(repr(a * _SC) for a in _ALPHA_LAT1) + "/").replace(
        "0.0 0.4 1.0 2.0 3.5 5.5 8.0 12.0/",
        " ".join(repr(b * _SC) for b in _BETA_LAT1) + "/")
    # all three replaces really happened (guards against deck-template drift)
    assert "6 8 0/" in deck and "6 8 1/" not in deck
    assert "15.0/" not in deck and "12.0/" not in deck
    return deck


def _run_lat0(mode, tag):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, f"nc_lat0_{tag}.input")
    out = os.path.join(d, f"nc_lat0_{tag}.endf")
    with open(inp, "w") as f:
        f.write(_lat0_deck(mode))
    run_leapr(inp, out)
    return out


def _lat_flag(path):
    from endf_parserpy import EndfParserPy
    return int(EndfParserPy().parsefile(path)[7][4]["LAT"])


@pytest.fixture(scope="module")
def lat0_tapes():
    return {1: _run_lat0(1, "m1"), 2: _run_lat0(2, "m2")}


@pytest.mark.parametrize("mode", [1, 2])
def test_lat0_law_matches_lat1_pins_and_is_physical(lat0_tapes, mode):
    """With bit-identical physical grids the lat=0 law must hit the SAME
    frozen pins as the lat=1 fast-CI run. Verified by two independent
    processes, both printing 1: 9.098600568286e-01 / 2: 2.633836729729e+00
    — exactly _PINS, so the lat=1 pins double as the lat=0 pins and any
    future re-bless there re-blesses here."""
    s_sum, s_min, teff0 = _mt4_stats(lat0_tapes[mode])
    assert _lat_flag(lat0_tapes[mode]) == 0     # the lat=0 branch wrote it
    assert s_sum == pytest.approx(_PINS[mode], rel=1.0e-6)
    assert s_min >= 0.0
    assert teff0 == pytest.approx(_TEFF0, abs=0.01)


def test_lat0_model_layer_reuse_is_bit_identical(capsys):
    """A lat=0 deck's physical Q/E grids scale with kT, so a second
    temperature misses the full-context cache key but must hit the model
    layer (phonopy load, mesh eigensolves, star average) and give S(a,b)
    bit-identical to a fresh build at that temperature."""
    import numpy as np
    from irma.core.noncubic_inelastic import NoncubicInelasticControls
    from irma.core.standalone_sab import run_noncubic_standalone_sab

    controls = NoncubicInelasticControls(
        num_directions=40, multiphonon_num_directions=20,
        multiphonon_max_order=6)
    kw = dict(
        alpha=np.array(_ALPHA_LAT1), beta=np.array(_BETA_LAT1), lat=0,
        awr=11.898, phonopy_yaml_path=_YAML, mesh_dim=[4, 4, 4], num_jobs=1,
        inelastic_mode=1, controls=controls)

    shared = {}
    run_noncubic_standalone_sab(
        temperature_k=296.0, context_cache=shared, **kw)
    second = run_noncubic_standalone_sab(
        temperature_k=400.0, context_cache=shared, **kw)
    out = capsys.readouterr().out
    assert "Reusing cached noncubic model context" in out

    fresh = run_noncubic_standalone_sab(
        temperature_k=400.0, context_cache={}, **kw)
    assert np.array_equal(second["ssm_internal"], fresh["ssm_internal"])


def test_model_input_identity_changes_on_rewrite(tmp_path):
    """The model cache key must miss when an input file is rewritten at the
    same path (content identity = path + size + mtime_ns)."""
    import os
    from irma.core.standalone_sab import _file_identity

    a = tmp_path / "FORCE_CONSTANTS"
    a.write_text("1 1\n0.0\n")
    id1 = _file_identity(a)
    os.utime(a, ns=(id1[2] + 1_000_000, id1[2] + 1_000_000))
    assert _file_identity(a) != id1
