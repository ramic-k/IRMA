from pathlib import Path
from ncrystal_plugin_ENDFTSL.reader import read_tsl, _as_list

TAPE = Path(__file__).parents[1] / "examples" / "graphite" / "graphite_mef_296K.endf"


def test_reads_lthr3_and_mt4_header():
    ev = read_tsl(TAPE)
    assert ev.lthr == 3 and ev.lat == 1 and ev.lasym == 0 and ev.lln == 0
    assert ev.coh_edges_ev is not None and len(ev.coh_edges_ev) > 1
    # coherent cumulative S is monotonically non-decreasing at the principal temperature
    s0 = ev.coh_cumS[0]
    assert all(s0[i] <= s0[i + 1] + 1e-12 for i in range(len(s0) - 1))
    assert ev.incoh_sb_barn is not None and ev.incoh_Wp is not None
    # inelastic grids present and rectangular
    assert len(ev.beta) > 1 and len(ev.alpha[0]) > 1
    assert len(ev.sab) == len(ev.beta)
    assert len(ev.sab[0]) == len(ev.alpha[0])


def test_indexing_normalizer():
    assert _as_list([1.0, 2.0]) == [1.0, 2.0]
    assert _as_list({1: 10.0, 2: 20.0}) == [10.0, 20.0]   # 1-indexed dict -> list
