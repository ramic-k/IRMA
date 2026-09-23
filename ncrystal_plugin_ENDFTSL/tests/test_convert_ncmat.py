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


def test_beta_refinement_follows_the_tapes_log_linear_law():
    # NCrystal interpolates the pack linearly in beta. Over the intervals the
    # tape declares log-linear (INT=4, no zero cell) the refined table must
    # integrate like the tape's own law, integrated exactly, to the 1e-3 tolerance
    import math
    ev = read_tsl(TAPE)
    law = physics.physical_inelastic(ev, 296.0)
    pk = convert.build_pack(ev, 296.0, "graphite", 12.0107)
    tb, pb, na = law.beta_phys, pk.beta_grid, len(pk.alpha_grid)
    assert len(pb) > len(tb) and set(tb) <= set(pb)       # tape points kept
    at = {b: i for i, b in enumerate(pb)}
    S = law.sab_scaled_sym
    refined = [j for j in range(len(tb) - 1) if ev.beta_int[j] == 4
               and all(row[j] > 0.0 and row[j + 1] > 0.0 for row in S)]
    for ai in range(0, na, 25):
        exact = linear = 0.0
        for j in refined:
            a, b, h = S[ai][j], S[ai][j + 1], tb[j + 1] - tb[j]
            exact += h * (b - a) / math.log(b / a) if a != b else h * a
            for k in range(at[tb[j]], at[tb[j + 1]]):
                s0, s1 = pk.sab_values[k * na + ai], pk.sab_values[(k + 1) * na + ai]
                linear += (pb[k + 1] - pb[k]) * (s0 + s1) / 2.0
        assert linear == pytest.approx(exact, rel=1e-3)


def test_ncmat_has_no_cell():
    s = ncmat.multi_pack_ncmat([("C", 1.0)], 2.26, ["graphite.endftslpack"])
    assert s.startswith("NCMAT v5")
    assert "@CELL" not in s and "@ATOMPOSITIONS" not in s
    assert "@CUSTOM_ENDFTSL" in s and "pack graphite.endftslpack" in s
    assert "freegas" in s
