"""The committed examples/ tree must stay loadable, indexed, and non-alarming.

Pins:
  * every spectra YAML parses AND passes semantic validation (an example
    that fails kinematic sanity only at run time sprays multiphonon
    kernel-truncation WARNINGs at the user);
  * the q-cuts example's grid covers graphite's full one-phonon spectrum
    (~200 meV) so the multiphonon work grid does not truncate kernels;
  * the documented mode-2 deck is production-shaped (Card 6g auto_order=1),
    not the one-phonon-only Euphonic validation template;
  * the README claim that every example carries its run command stays true
    for both families (YAML headers / first-card trailing text).
"""
import os

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
SPECTRA_DIR = os.path.join(EXAMPLES_DIR, "spectra")
TSL_DIR = os.path.join(EXAMPLES_DIR, "tsl")


def _spectra_yamls():
    return sorted(
        f for f in os.listdir(SPECTRA_DIR) if f.endswith((".yaml", ".yml"))
    )


def _tsl_decks():
    return sorted(f for f in os.listdir(TSL_DIR) if f.endswith(".input"))


@pytest.mark.parametrize("name", _spectra_yamls())
def test_spectra_example_loads_and_validates(name):
    pytest.importorskip("yaml")
    from irma.spectra import config as scfg

    cfg = scfg.load(os.path.join(SPECTRA_DIR, name), validate_cfg=True)
    assert cfg.material.scatterers, f"{name}: no scatterers configured"


@pytest.mark.parametrize("name", _spectra_yamls())
def test_spectra_example_header_has_run_command(name):
    with open(os.path.join(SPECTRA_DIR, name)) as f:
        head = f.read(2000)
    assert "python -m irma spectra" in head, (
        f"{name} must carry its exact run command in the header comment "
        f"(examples/README.md promises it)")


@pytest.mark.parametrize("name", _tsl_decks())
def test_tsl_deck_first_card_has_run_command(name):
    with open(os.path.join(TSL_DIR, name)) as f:
        first = f.readline()
    assert "$ run" in first and "python -m irma" in first, (
        f"{name} must carry its run command as trailing text on Card 1 "
        f"(examples/README.md promises it)")


def test_qcuts_grid_covers_graphite_phonon_spectrum():
    """grid.e_max_meV must exceed graphite's ~200 meV one-phonon cutoff:
    the engine derives the multiphonon work grid from the output grid, so a
    narrower window truncates the kernels and fires the kernel-area WARNING
    on every run of the committed example."""
    pytest.importorskip("yaml")
    from irma.spectra import config as scfg

    cfg = scfg.load(
        os.path.join(SPECTRA_DIR, "graphite_direct_qcuts.yaml"),
        validate_cfg=True)
    assert cfg.grid.e_max_meV >= 210.0
    assert cfg.instrument.e_fixed_meV > cfg.grid.e_max_meV  # direct-geometry rule


def test_mode2_deck_is_production_shaped():
    with open(os.path.join(TSL_DIR, "graphite_mode2.input")) as f:
        deck = f.read()
    card_6g = next(
        line for line in deck.splitlines() if line.startswith("10000 1000"))
    fields = card_6g.split("/")[0].split()
    assert fields == ["10000", "1000", "1"], (
        "examples/tsl/graphite_mode2.input Card 6g must keep auto_order=1 "
        "(ndir mpdir auto): without it the deck's nphon=1 yields a "
        "one-phonon-only law that the examples/tsl/README.md §3 walkthrough "
        "would then present as ready-to-run")


def test_examples_index_mentions_every_family_member():
    with open(os.path.join(EXAMPLES_DIR, "README.md")) as f:
        index = f.read()
    for name in _spectra_yamls() + _tsl_decks() + ["lln_low_temperature_demo.py"]:
        assert name in index, f"examples/README.md does not mention {name}"
