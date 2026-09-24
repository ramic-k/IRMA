"""Polyatomic converter path: fraction-weighted coherent + fraction validation.

Uses the monatomic graphite fixture as two pseudo-species (f=0.5 each) to exercise
build_packs' multi-pack coherent math without a large polyatomic fixture: a per-atom
Bragg-edge structure replicated on both tapes must, after fraction-weighting, sum back
to the single per-atom edge structure (this is exactly the ENDF/B-VIII.1 BeO layout,
where both tapes carry the full per-atom edges).
"""
import math
import os
from pathlib import Path

import pytest

from ncrystal_plugin_ENDFTSL.reader import read_tsl
from ncrystal_plugin_ENDFTSL.convert import build_pack, build_packs, SpeciesSpec

TAPE = Path(__file__).parents[1] / "examples" / "graphite" / "graphite_mef_296K.endf"


def _specs(*fracs):
    return [SpeciesSpec(str(TAPE), sym, 12.0107, f) for sym, f in zip("ABCD", fracs)]


@pytest.mark.parametrize("specs, match", [
    (_specs(0.3, 0.5), "sum to 1"),
    (_specs(1.5, -0.5), r"\(0, 1\]"),
    ([SpeciesSpec(str(TAPE), "A", 12.0107, None)], "need atom fractions"),
], ids=["sum", "range", "missing"])
def test_build_packs_rejects_bad_fractions(specs, match):
    with pytest.raises(ValueError, match=match):
        build_packs(specs, 296.0, "bad")


def test_fraction_weighting_recovers_per_atom_all_channels():
    ev = read_tsl(str(TAPE))
    single = build_pack(ev, 296.0, "g", 12.0107)  # default scales of 1
    assert single.coh_cumS, "fixture tape must carry coherent Bragg edges"
    packs = build_packs(_specs(0.5, 0.5), 296.0, "g2")
    assert len(packs) == 2
    for pk in packs:
        # inelastic (bound_xs) is fraction-weighted -> per-atom after summing
        assert math.isclose(pk.bound_xs_barn, 0.5 * single.bound_xs_barn, rel_tol=1e-12)
        # incoherent elastic is fraction-weighted the same way
        assert math.isclose(pk.elastic_incoherent_xs_barn,
                            0.5 * single.elastic_incoherent_xs_barn, rel_tol=1e-12)
        # coherent Bragg edges are fraction-weighted
        assert len(pk.coh_cumS) == len(single.coh_cumS)
        for scaled, raw in zip(pk.coh_cumS, single.coh_cumS):
            assert math.isclose(scaled, 0.5 * raw, rel_tol=1e-12, abs_tol=1e-15)
    # the C++ sums packs at weight 1.0 -> per-atom values are recovered
    assert math.isclose(packs[0].bound_xs_barn + packs[1].bound_xs_barn,
                        single.bound_xs_barn, rel_tol=1e-12)
    for i, raw in enumerate(single.coh_cumS):
        summed = packs[0].coh_cumS[i] + packs[1].coh_cumS[i]
        assert math.isclose(summed, raw, rel_tol=1e-12, abs_tol=1e-15)


def test_sole_coherent_carrier_is_scaled_by_its_fraction(tmp_path, monkeypatch):
    # species B's tape stands in for an LTHR=2 tape: no coherent elastic, so A
    # is the only coherent carrier; its edges still get A's atom fraction
    import copy
    from ncrystal_plugin_ENDFTSL import convert
    no_coh = tmp_path / "no_coherent.endf"
    no_coh.write_bytes(TAPE.read_bytes())
    real = convert.read_tsl

    def read(path):
        ev = real(path)
        if str(path) == str(no_coh):
            ev = copy.copy(ev)
            ev.coh_temps = ev.coh_edges_ev = ev.coh_cumS = None
        return ev
    monkeypatch.setattr(convert, "read_tsl", read)
    single = build_pack(read_tsl(str(TAPE)), 296.0, "g", 12.0107)
    specs = [SpeciesSpec(str(TAPE), "A", 12.0107, 0.3),
             SpeciesSpec(str(no_coh), "B", 12.0107, 0.7)]
    a, b = build_packs(specs, 296.0, "sole")
    assert b.coh_cumS == []
    for scaled, raw in zip(a.coh_cumS, single.coh_cumS):
        assert math.isclose(scaled, 0.3 * raw, rel_tol=1e-12, abs_tol=1e-15)


# Optional local multi-temperature tape. Point ENDFTSL_UO2_TAPE at a tsl-UinUO2.endf
# to exercise multi-temperature column selection; skipped when unset/absent so the
# suite stays portable (the committed graphite fixture already covers the
# non-stored-temperature rejection in test_build_pack_rejects_non_stored_temperature).
_UO2_TAPE_ENV = os.environ.get("ENDFTSL_UO2_TAPE")
_UO2_TAPE = Path(_UO2_TAPE_ENV) if _UO2_TAPE_ENV else None


@pytest.mark.skipif(_UO2_TAPE is None or not _UO2_TAPE.exists(),
                    reason="multi-temperature UO2 tape absent (set ENDFTSL_UO2_TAPE)")
def test_multi_temperature_column_selection():
    # A multi-temperature tape: build_pack at two different STORED temps gives different
    # S(a,b) columns + LAT-scaled grids (same bound_xs); a non-stored temp raises.
    ev = read_tsl(str(_UO2_TAPE))
    assert len(ev.temps_mt4) > 1
    pa = build_pack(ev, ev.temps_mt4[0], "x", 238.0289)
    pb = build_pack(ev, ev.temps_mt4[1], "x", 238.0289)
    assert pa.bound_xs_barn == pytest.approx(pb.bound_xs_barn)   # T-independent
    assert pa.beta_grid != pb.beta_grid                          # LAT-scaled by T
    assert pa.sab_values != pb.sab_values                        # different T column
    mid = 0.5 * (ev.temps_mt4[0] + ev.temps_mt4[1])
    with pytest.raises(NotImplementedError, match="no interpolation"):
        build_pack(ev, mid, "x", 238.0289)


def test_build_pack_rejects_non_stored_temperature():
    # build_pack must select a column matching one of the tape's stored MT4 temps
    # (no interpolation); a non-stored temperature raises instead of silently using
    # the wrong column on a wrong-T grid.
    ev = read_tsl(str(TAPE))
    t0 = ev.temps_mt4[0]
    build_pack(ev, t0, "g", 12.0107)  # a stored temperature: OK
    with pytest.raises(NotImplementedError, match="no interpolation"):
        build_pack(ev, t0 + 200.0, "g", 12.0107)
