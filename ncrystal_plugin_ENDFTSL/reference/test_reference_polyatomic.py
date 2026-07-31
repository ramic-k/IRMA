"""ENDFTSL polyatomic reference gate: PER-ATOM normalization through the compiled .so.

The C++ plugin (NCPhysicsModel.cc createPluginProcess) sums EVERY pack's scatter
components at weight 1.0 -- it does NOT apply the NCMAT @DYNINFO atom fractions. So
per-atom normalization is entirely the converter's job: build_packs scales each pack
by its atom fraction f, and sum_i f_i*sigma_i = the per-atom average. This gate proves
that contract through the real plugin: the graphite tape baked as TWO pseudo-species at
fraction 0.5 each must give the SAME per-atom cross section as the single-species
graphite pack at fraction 1.0 -- not 2x (which is what a per-formula bug would yield).

This is the C++-side counterpart of tests/test_polyatomic.py
(test_fraction_weighting_recovers_per_atom_all_channels), which checks the same identity
at the Python pack level. Skips unless the compiled plugin is discovered by NCrystal.
"""
from pathlib import Path

import pytest

NC = pytest.importorskip("NCrystal", exc_type=ModuleNotFoundError)
pytest.importorskip("endf_parserpy", exc_type=ModuleNotFoundError)

from ncrystal_plugin_ENDFTSL.reader import read_tsl                    # noqa: E402
from ncrystal_plugin_ENDFTSL.convert import build_pack, build_packs, SpeciesSpec  # noqa: E402
from ncrystal_plugin_ENDFTSL.pack import write_pack                    # noqa: E402
from ncrystal_plugin_ENDFTSL.ncmat import multi_pack_ncmat, structure_free_ncmat  # noqa: E402

HERE = Path(__file__).parent
TAPE = HERE.parent / "examples" / "graphite" / "graphite_mef_296K.endf"


def _plugin_available() -> bool:
    try:
        return "ENDFTSL" in [p[0] for p in NC.browsePlugins()]
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _plugin_available(),
    reason="ENDFTSL NCrystal plugin not installed/discovered (pip install the plugin first)")


@pytest.fixture(scope="module")
def cfgs(tmp_path_factory):
    out = tmp_path_factory.mktemp("endftsl_poly")
    ev = read_tsl(str(TAPE))

    # (a) single species, fraction 1.0 -> the per-atom graphite reference
    single = build_pack(ev, 296.0, "g1", "C", 12.0107)
    sp = out / "g1.endftslpack"
    write_pack(single, sp)
    (out / "single.ncmat").write_text(
        structure_free_ncmat("C", 2.26, str(sp.resolve())), encoding="utf-8")

    # (b) two pseudo-species at fraction 0.5 each, BOTH the graphite tape. The second
    # symbol is a stand-in for a distinct principal scatterer; its @DYNINFO placeholder
    # is overridden and the pack carries graphite's own mass, so the physics is pure
    # graphite. build_packs scales each pack by 0.5 (inelastic + incoherent + coherent),
    # so the C++ weight-1.0 sum must recover the single-species per-atom cross section.
    specs = [SpeciesSpec(str(TAPE), "C", 12.0107, 0.5),
             SpeciesSpec(str(TAPE), "Si", 12.0107, 0.5)]
    packs = build_packs(specs, 296.0, "g2")
    pack_paths = []
    for pk in packs:
        p = out / f"{pk.material_id}.endftslpack"
        write_pack(pk, p)
        pack_paths.append(str(p.resolve()))
    (out / "poly.ncmat").write_text(
        multi_pack_ncmat([("C", 0.5), ("Si", 0.5)], 2.26, pack_paths), encoding="utf-8")

    return (f"{out / 'single.ncmat'};temp=296.0K",
            f"{out / 'poly.ncmat'};temp=296.0K")


def test_polyatomic_total_is_per_atom_not_per_formula(cfgs):
    single_cfg, poly_cfg = cfgs
    sc1 = NC.createScatter(single_cfg)
    sc2 = NC.createScatter(poly_cfg)
    for E in (0.001, 0.005, 0.025, 0.1, 0.5, 1.0, 5.0):
        x1 = sc1.crossSectionIsotropic(E)
        x2 = sc2.crossSectionIsotropic(E)
        # per-atom: the two half-weighted pseudo-species sum back to single graphite.
        # a per-formula bug would give x2 ~ 2*x1 (the test would fail loudly).
        assert x2 == pytest.approx(x1, rel=1e-4), f"E={E} eV: poly {x2} vs single {x1}"


def test_polyatomic_per_channel_is_per_atom(cfgs):
    # the same per-atom identity must hold channel-by-channel (coherent + inelastic +
    # incoherent each fraction-weighted), not just by a cancelling total.
    single_cfg, poly_cfg = cfgs
    for comp in ("coh_elas", "inelas", "incoh_elas"):
        sc1 = NC.createScatter(single_cfg + f";comp={comp}")
        sc2 = NC.createScatter(poly_cfg + f";comp={comp}")
        for E in (0.005, 0.025, 0.2, 1.0):
            x1 = sc1.crossSectionIsotropic(E)
            x2 = sc2.crossSectionIsotropic(E)
            assert x2 == pytest.approx(x1, rel=1e-4, abs=1e-9), \
                f"{comp} E={E} eV: poly {x2} vs single {x1}"
