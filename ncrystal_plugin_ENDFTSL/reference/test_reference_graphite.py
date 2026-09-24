"""ENDFTSL plugin reference gate (graphite MEF, LTHR=3).

Requires the compiled plugin installed (NCrystal auto-discovers it via
ncrystal-pypluginmgr) plus NCrystal + endf_parserpy. Converts the vendored tape,
loads it through NCrystal, and checks three reference identities:
  (1) coherent  sigma == the tape's own S(E)/E  (to 1e-6 relative),
  (2) incoherent sigma == ENDF (sb/2)(1-e^-4EW')/(2EW'),
  (3) total sigma == a vendored reference grid.
This test suite must not import irma.
"""
import json
import math
from pathlib import Path

import pytest

NC = pytest.importorskip("NCrystal", exc_type=ModuleNotFoundError)
pytest.importorskip("endf_parserpy", exc_type=ModuleNotFoundError)

from ncrystal_plugin_ENDFTSL.reader import read_tsl          # noqa: E402
from ncrystal_plugin_ENDFTSL import physics                  # noqa: E402
from ncrystal_plugin_ENDFTSL.__main__ import main as convert_main  # noqa: E402

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
def cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("endftsl_out")
    convert_main([str(TAPE), "-o", str(out), "--material-id", "graphite",
                  "--symbol", "C", "--mass", "12.0107", "--density", "2.26",
                  "--temperature", "296"])
    return f"{out / 'graphite.ncmat'};temp=296.0K"


def test_plugin_activates(cfg):
    # the plugin's SABScatter (about 0.5 b) replaces the freegas placeholder (~4.9 b)
    inel = NC.createScatter(cfg + ";comp=inelas").crossSectionIsotropic(0.025)
    assert 0.1 < inel < 1.0


def test_coherent_xs_is_tape_S_over_E(cfg):
    ev = read_tsl(TAPE)
    edges, cumS = physics.coherent_edges(ev, 296.0)
    sc_coh = NC.createScatter(cfg + ";comp=coh_elas")
    for idx in (4, 20, 100):
        # an energy strictly inside bin [edges[idx], edges[idx+1]): S(E)=cumS[idx]
        E = 0.5 * (edges[idx] + edges[idx + 1])
        assert sc_coh.crossSectionIsotropic(E) == pytest.approx(cumS[idx] / E, rel=1e-6)


def test_incoherent_xs_matches_endf_formula(cfg):
    ev = read_tsl(TAPE)
    _msd, sb = physics.incoherent_msd(ev, 296.0)
    Wp = physics._interp_T(ev.incoh_temps, [[w] for w in ev.incoh_Wp], 296.0)[0]
    sc_inc = NC.createScatter(cfg + ";comp=incoh_elas")
    for E in (0.005, 0.025, 0.1):
        endf = (sb / 2.0) * (1.0 - math.exp(-4.0 * E * Wp)) / (2.0 * E * Wp)
        assert sc_inc.crossSectionIsotropic(E) == pytest.approx(endf, rel=1e-3)


def test_total_xs_matches_expected(cfg):
    sc = NC.createScatter(cfg)
    ref = json.loads((HERE / "expected" / "graphite_ref_xs.json").read_text())
    for e_str, xs in ref.items():
        assert sc.crossSectionIsotropic(float(e_str)) == pytest.approx(xs, rel=1e-4)
