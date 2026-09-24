from pathlib import Path
import pytest
from ncrystal_plugin_ENDFTSL.reader import read_tsl
from ncrystal_plugin_ENDFTSL import physics

TAPE = Path(__file__).parents[1] / "examples" / "graphite" / "graphite_mef_296K.endf"


def test_coherent_edges_verbatim():
    ev = read_tsl(TAPE)
    edges, cumS = physics.coherent_edges(ev, 296.0)
    # at the principal temperature the values equal the stored column exactly
    assert edges == ev.coh_edges_ev
    assert cumS == ev.coh_cumS[0]


def test_incoherent_msd_positive_and_sigma_matches_sb():
    ev = read_tsl(TAPE)
    msd, sinc = physics.incoherent_msd(ev, 296.0)
    assert msd > 0.0
    assert sinc == pytest.approx(ev.incoh_sb_barn, rel=1e-12)
    # MSD = W'·ħ²/2mₙ
    from ncrystal_plugin_ENDFTSL.constants import HBAR2_OVER_2MN_EV_A2
    Wp = physics._interp_T(ev.incoh_temps, [[w] for w in ev.incoh_Wp], 296.0)[0]
    assert msd == pytest.approx(Wp * HBAR2_OVER_2MN_EV_A2, rel=1e-12)


def test_lasym1_tape_is_refused():
    # LEAPR's cold H2/D2 tapes store S for -beta..+beta (LASYM=1)
    ev = read_tsl(TAPE)
    ev.lasym = 1
    with pytest.raises(NotImplementedError, match="LASYM=1"):
        physics.physical_inelastic(ev, 296.0)


def test_inelastic_grids_physical_and_lat_unscaled():
    ev = read_tsl(TAPE)
    law = physics.physical_inelastic(ev, 296.0)
    assert law.beta_phys[0] == pytest.approx(0.0, abs=1e-12)
    assert all(law.beta_phys[i] < law.beta_phys[i + 1] for i in range(len(law.beta_phys) - 1))
    # LAT=1: physical β = stored β · T_LAT/T  (T<T_LAT ⇒ physical β larger)
    from ncrystal_plugin_ENDFTSL.constants import T_LAT_K
    if ev.lat == 1 and len(ev.beta) > 1:
        assert law.beta_phys[1] == pytest.approx(ev.beta[1] * T_LAT_K / 296.0, rel=1e-9)
    # the scaled-symmetric table is rectangular [alpha][beta] and equal to the
    # stored values (LLN=0 here)
    assert len(law.sab_scaled_sym) == len(law.alpha_phys)
    assert law.sab_scaled_sym[3][5] == max(0.0, ev.sab[5][3][0])
