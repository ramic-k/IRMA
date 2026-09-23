from pathlib import Path
import pytest
from ncrystal_plugin_ENDFTSL.reader import read_tsl
from ncrystal_plugin_ENDFTSL import convert, ncmat, physics

TAPE = Path(__file__).parents[1] / "examples" / "graphite" / "graphite_mef_296K.endf"


def test_build_pack_shapes_and_massscale():
    ev = read_tsl(TAPE)
    pk = convert.build_pack(ev, 296.0, "graphite", 12.0107)
    assert len(pk.sab_values) == len(pk.alpha_grid) * len(pk.beta_grid)
    # α mass-scaled by AWR
    law = physics.physical_inelastic(ev, 296.0)
    assert pk.alpha_grid[1] / law.alpha_phys[1] == pytest.approx(ev.awr, rel=1e-9)
    # MEF tape ⇒ both elastic blocks present
    assert pk.coh_edges_ev and pk.elastic_msd_a2 is not None


def test_ncmat_has_no_cell():
    s = ncmat.multi_pack_ncmat([("C", 1.0)], 2.26, ["graphite.endftslpack"])
    assert s.startswith("NCMAT v5")
    assert "@CELL" not in s and "@ATOMPOSITIONS" not in s
    assert "@CUSTOM_ENDFTSL" in s and "pack graphite.endftslpack" in s
    assert "freegas" in s
