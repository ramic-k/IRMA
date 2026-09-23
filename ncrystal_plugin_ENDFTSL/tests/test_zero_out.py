"""Zero-out coverage: a single-component-elastic tape (SEF graphite, LTHR=1)
imports the coherent channel and cleanly OMITS the absent incoherent channel."""
import tempfile
from pathlib import Path

import pytest

from ncrystal_plugin_ENDFTSL.reader import read_tsl
from ncrystal_plugin_ENDFTSL import convert

TAPE = Path(__file__).parent / "data" / "graphite_cef_296K.endf"


def test_coherent_only_tape_has_no_incoherent_block():
    ev = read_tsl(TAPE)
    assert ev.lthr == 1 and ev.incoh_Wp is None
    pk = convert.build_pack(ev, 296.0, "graphite", 12.0107)
    assert pk.coh_edges_ev              # coherent present
    assert pk.elastic_msd_a2 is None    # incoherent zeroed out


def test_zero_out_end_to_end():
    NC = pytest.importorskip("NCrystal", exc_type=ModuleNotFoundError)
    if "ENDFTSL" not in [p[0] for p in NC.browsePlugins()]:
        pytest.skip("ENDFTSL plugin not installed/discovered")
    from ncrystal_plugin_ENDFTSL.__main__ import main as convert_main
    out = Path(tempfile.mkdtemp())
    convert_main([str(TAPE), "-o", str(out), "--material-id", "graphite_cef",
                  "--symbol", "C", "--mass", "12.0107", "--density", "2.26",
                  "--temperature", "296"])
    cfg = f"{out / 'graphite_cef.ncmat'};temp=296.0K"
    coh = NC.createScatter(cfg + ";comp=coh_elas").crossSectionIsotropic(0.01)
    inc = NC.createScatter(cfg + ";comp=incoh_elas").crossSectionIsotropic(0.025)
    assert coh > 0.0          # coherent Bragg present
    assert inc == 0.0         # incoherent absent (freegas base -> 0)
