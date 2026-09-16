"""Automatic alpha/beta grid generation (irma.core.grids).

The GUI's auto-grid feature builds its grids here; these tests pin the
structural guarantees the rest of the pipeline relies on (origin point,
monotonicity, point counts, region boundaries, the linear-in-Q alpha
layout).
"""
import numpy as np
import pytest

from irma.core.grids import (generate_beta_grid, generate_alpha_grid,
                               generate_beta_grid_for_iint,
                               estimate_freq_max_from_dos,
                               DELTA_BETA_MAX_LINLIN)

KB = 8.617333262e-5

from irma.core.constants import HBAR2_OVER_2MN_MEV_A2 as HB2  # noqa: E402


def test_beta_grid_structure_defaults():
    freq_max, T = 0.2, 296.0
    beta = generate_beta_grid(freq_max, T)
    kT = KB * T

    assert beta[0] == 0.0
    assert np.all(np.diff(beta) > 0.0), "beta grid must be strictly increasing"
    # 1 (zero) + n_lower (50) + n_phonon-1 (299) + n_upper (20)
    assert len(beta) == 1 + 50 + 299 + 20
    # upper tail ends at beta_max_eV/kT
    assert beta[-1] == pytest.approx(5.0 / kT, rel=1e-12)
    # linear region: constant spacing freq_max/n_phonon (in beta units)
    d = freq_max / 300 / kT
    lin = beta[51:51 + 299]
    np.testing.assert_allclose(np.diff(lin), d, rtol=1e-9)
    # the linear region tops out just below freq_max
    assert lin[-1] == pytest.approx(freq_max * 299 / 300 / kT, rel=1e-9)


def test_beta_grid_no_tails():
    beta = generate_beta_grid(0.2, 296.0, n_lower=0, n_upper=0)
    assert beta[0] == 0.0
    assert len(beta) == 1 + 299
    assert np.all(np.diff(beta) > 0.0)


def test_beta_grid_upper_tail_skipped_if_phonon_region_reaches_max():
    """If the linear region already exceeds beta_max_eV, no upper tail."""
    with pytest.warns(UserWarning, match="upper log tail dropped"):
        beta = generate_beta_grid(6.0, 296.0, beta_max_eV=5.0)
    kT = KB * 296.0
    assert beta[-1] == pytest.approx(6.0 * 299 / 300 / kT, rel=1e-9)
    assert len(beta) == 1 + 50 + 299


def test_beta_grid_dropped_tail_warns_but_grid_is_unchanged():
    """beta_max_eV at or below the linear region's end: the grid stays
    byte-identical to the historical output (the cap is NOT enforced — the
    linear phonon region is never truncated) but the silent drop now warns,
    naming the cap and the linear-region end."""
    import warnings

    kT = KB * 296.0
    with pytest.warns(UserWarning, match="beta_max_eV"):
        beta = generate_beta_grid(0.45, 296.0, beta_max_eV=0.2)
    # identical to the historical silent result: ends at freq_max*(1-1/n_phonon)
    assert beta[-1] == pytest.approx(0.45 * 299 / 300 / kT, rel=1e-9)
    assert len(beta) == 1 + 50 + 299
    # n_upper=0 is an explicit no-tail request: stays silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        beta0 = generate_beta_grid(0.45, 296.0, n_upper=0, beta_max_eV=0.2)
    np.testing.assert_array_equal(beta0, beta[:len(beta0)])
    # a cap above the linear region keeps the tail and stays silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        beta_tail = generate_beta_grid(0.45, 296.0, beta_max_eV=5.0)
    assert beta_tail[-1] == pytest.approx(5.0 / kT, rel=1e-12)


def test_beta_grid_default_cap_none_is_byte_identical():
    """delta_beta_max=None reproduces the historical pure-geometric upper
    tail, so every log-lin (iint=0) auto-grid is unchanged.

    The cap=None alias is byte-identical to the default (same code path,
    asserted exactly). The tail is compared against an independent
    np.geomspace reconstruction, which matches the grid builder only to
    the last ULP: the two constructions round transcendentals through
    different numpy code paths, and numpy builds across Python versions
    and runner CPUs disagree in the final bit (seen on the hosted-runner
    3.13 job at 1e-15 relative). rtol=1e-13 still fails on any genuine
    change to the tail's construction, which moves points at >1e-6."""
    base = generate_beta_grid(0.2, 296.0)
    explicit_none = generate_beta_grid(0.2, 296.0, delta_beta_max=None)
    np.testing.assert_array_equal(base, explicit_none)
    kT = KB * 296.0
    beta_lin_end = 0.2 * 299 / 300 / kT
    old_upper = np.geomspace(beta_lin_end, 5.0 / kT, 21)[1:]
    got_upper = base[base > beta_lin_end + 1e-12]
    np.testing.assert_allclose(got_upper, old_upper, rtol=1e-13)


def test_beta_grid_cap_bounds_upper_step_uniform():
    """With delta_beta_max set and no AWR, every upper-tail interval is capped
    (uniform fine tail), the grid stays monotonic, and beta_max is preserved."""
    kT = KB * 296.0
    beta_lin_end = 0.2 * 299 / 300 / kT
    capped = generate_beta_grid(0.2, 296.0,
                                delta_beta_max=DELTA_BETA_MAX_LINLIN)
    assert np.all(np.diff(capped) > 0.0)
    assert capped[-1] == pytest.approx(5.0 / kT, rel=1e-12)
    upper = capped[capped > beta_lin_end + 1e-12]
    assert np.diff(upper).max() <= DELTA_BETA_MAX_LINLIN + 1e-9
    # the cap densifies the tail relative to the 20-point pure-log tail
    assert len(capped) > len(generate_beta_grid(0.2, 296.0))


def test_beta_grid_recoil_ridge_coarsens_deep_tail():
    """With recoil_awr, the fine cap applies only up to beta_fine =
    4*beta_max/awr (the recoil ridge); above it the original coarse log spacing
    returns, so the grid is SMALLER than the uniform cap while the fine region
    stays bounded by delta_beta_max."""
    kT = KB * 296.0
    awr = 11.898
    beta_max = 5.0 / kT
    beta_fine = 4.0 * beta_max / awr
    ridge = generate_beta_grid(0.2, 296.0, delta_beta_max=DELTA_BETA_MAX_LINLIN,
                               recoil_awr=awr)
    uniform = generate_beta_grid(0.2, 296.0,
                                 delta_beta_max=DELTA_BETA_MAX_LINLIN)
    assert np.all(np.diff(ridge) > 0.0)
    assert ridge[-1] == pytest.approx(beta_max, rel=1e-12)
    # recoil-ridge cutoff yields fewer points than the uniform cap
    assert len(ridge) < len(uniform)
    # fine region (below the ridge) respects the step cap...
    d = np.diff(ridge)
    centers = ridge[:-1]
    fine = d[(centers > 0.2 * 299 / 300 / kT) & (centers < beta_fine)]
    assert fine.max() <= DELTA_BETA_MAX_LINLIN + 1e-9
    # ...while the deep tail above the ridge is allowed to be coarse
    deep = d[centers >= beta_fine]
    assert deep.size == 0 or deep.max() > DELTA_BETA_MAX_LINLIN


def test_beta_grid_light_atom_caps_whole_tail():
    """A light atom's recoil ridge (4*beta_max/awr) reaches beta_max, so the
    whole tail is fine-capped with no coarse region."""
    kT = KB * 296.0
    beta_lin_end = 0.2 * 299 / 300 / kT
    light = generate_beta_grid(0.2, 296.0, delta_beta_max=DELTA_BETA_MAX_LINLIN,
                               recoil_awr=1.0)  # 4*beta_max/1 >> beta_max
    upper = light[light > beta_lin_end + 1e-12]
    assert np.diff(upper).max() <= DELTA_BETA_MAX_LINLIN + 1e-9
    assert light[-1] == pytest.approx(5.0 / kT, rel=1e-12)


def test_beta_grid_heavy_atom_no_duplicate_seam():
    """A heavy atom's back-scatter alpha (4*beta_max/awr) falls below the
    phonon region end, but the law's ridge at that alpha still has a width,
    so the fine step reaches a few widths past it (linlin_fine_beta_limit):
    the grid must stay strictly increasing across the seam and carry the fine
    step up to that limit. For an atom so heavy that even the limit falls
    below the phonon region end, the whole upper tail is off-ridge and the
    grid falls back to the pure-log tail."""
    from irma.core.grids import linlin_fine_beta_limit
    kT = 8.617333262e-5 * 296.0
    heavy = generate_beta_grid(0.2, 296.0, delta_beta_max=DELTA_BETA_MAX_LINLIN,
                               recoil_awr=300.0)  # 4*beta_max/300 < beta_lin_end
    assert np.all(np.diff(heavy) > 0.0), "duplicate/non-monotonic node at seam"
    limit = linlin_fine_beta_limit(5.0 / kT, 300.0, 0.2, 296.0)
    beta_lin_end = (0.2 / kT) * (1.0 - 1.0 / 300.0)
    assert limit > beta_lin_end
    fine = heavy[(heavy > beta_lin_end) & (heavy <= limit)]
    assert np.max(np.diff(fine)) <= DELTA_BETA_MAX_LINLIN * (1 + 1e-12)
    very_heavy = generate_beta_grid(0.2, 296.0, delta_beta_max=DELTA_BETA_MAX_LINLIN,
                                    recoil_awr=3000.0)
    assert linlin_fine_beta_limit(5.0 / kT, 3000.0, 0.2, 296.0) < beta_lin_end
    np.testing.assert_array_equal(very_heavy, generate_beta_grid(0.2, 296.0))


def test_beta_grid_invalid_delta_beta_max_raises():
    """A non-positive delta_beta_max would never advance the tail loop; reject
    it instead of hanging."""
    with pytest.raises(ValueError):
        generate_beta_grid(0.2, 296.0, delta_beta_max=0.0)
    with pytest.raises(ValueError):
        generate_beta_grid(0.2, 296.0, delta_beta_max=-0.5, recoil_awr=11.9)


def test_alpha_grid_linear_q_layout():
    """Linear in Q at dq up to q_cut, log tail to the beta grid's reach.

    The previous recoil layout (alpha = 4*beta/awr) under-integrated the
    thermal inelastic XS by 14-19% on graphite (grid-convergence study
    2026-06); the linear-in-Q layout pinned here lands within 3-5% at
    the same point count.
    """
    awr, T = 11.898, 296.0
    beta = generate_beta_grid(0.2, T)
    alpha = generate_alpha_grid(beta, awr, T)
    kt_mev = KB * T * 1.0e3
    q = np.sqrt(alpha * awr * kt_mev / HB2)

    assert np.all(np.diff(alpha) > 0.0)
    # default layout: 240 linear points (dq=0.05 to q_cut=12) + 159 log
    assert len(alpha) == 240 + 159
    np.testing.assert_allclose(np.diff(q[:240]), 0.05, rtol=1e-6)
    assert q[239] == pytest.approx(12.0, rel=1e-6)
    # log tail: constant ratio up to the beta grid's kinematic reach
    ratios = q[240:] / q[239:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
    assert alpha[-1] == pytest.approx(4.0 * beta[-1] / awr, rel=1e-9)


def test_alpha_grid_short_beta_is_linear_only():
    """A beta grid whose reach stays below q_cut gets a pure linear-in-Q
    grid with the endpoint pinned at the kinematic maximum."""
    awr, T = 11.898, 296.0
    beta = np.array([0.0, 0.5])
    alpha = generate_alpha_grid(beta, awr, T)
    kt_mev = KB * T * 1.0e3
    q = np.sqrt(alpha * awr * kt_mev / HB2)
    q_max = np.sqrt(4.0 * 0.5 / awr * awr * kt_mev / HB2)
    assert q_max < 12.0
    np.testing.assert_allclose(np.diff(q[:-1]), 0.05, rtol=1e-6)
    assert alpha[-1] == pytest.approx(4.0 * 0.5 / awr, rel=1e-9)
    assert np.all(np.diff(alpha) > 0.0)


def test_alpha_grid_extra_low_points():
    beta = generate_beta_grid(0.2, 296.0)
    alpha0 = generate_alpha_grid(beta, 11.898, 296.0)
    alpha = generate_alpha_grid(beta, 11.898, 296.0, n_extra_low=10)
    assert len(alpha) == len(alpha0) + 10
    # the added points sit below the smallest alpha, down to 1% of it
    assert np.all(alpha[:10] < alpha0[0])
    assert alpha[0] == pytest.approx(alpha0[0] * 0.01, rel=1e-9)
    assert np.all(np.diff(alpha) > 0.0)


def test_alpha_grid_rejects_all_zero_beta():
    """An all-zero beta grid must fail loudly, not yield a zero-length alpha."""
    with pytest.raises(ValueError, match="no positive values"):
        generate_alpha_grid(np.array([0.0, 0.0]), 11.898, 296.0)


def test_estimate_freq_max_from_dos():
    e = np.linspace(0.0, 0.4, 401)
    dos = np.where(e <= 0.2, 1.0, 0.0)          # hard cutoff at 0.2 eV
    dos[e > 0.2] = 0.001                         # sub-threshold tail (0.1% of max)
    assert estimate_freq_max_from_dos(e, dos) == pytest.approx(0.2, abs=1e-3)
    # all-above-threshold DOS falls back to the last grid point
    assert estimate_freq_max_from_dos(e, np.ones_like(e)) == e[-1]


def test_estimate_freq_max_clamps_negative_dos():
    """An all-negative (unphysical) DOS gives an empty above-threshold mask via
    the non-negative-clamped threshold, falling back to the last grid point
    rather than inverting the 'above 1% of peak' test."""
    e = np.linspace(0.0, 0.4, 401)
    dos = -np.ones_like(e)
    assert estimate_freq_max_from_dos(e, dos) == e[-1]
    # Clamp is an exact no-op for a non-negative DOS: same result as before.
    pos = np.where(e <= 0.2, 1.0, 0.0)
    assert estimate_freq_max_from_dos(e, pos) == pytest.approx(0.2, abs=1e-3)


def test_generate_beta_grid_validates_inputs():
    from irma.core.grids import generate_beta_grid
    with pytest.raises(ValueError, match="n_phonon"):
        generate_beta_grid(0.2, 296.0, n_phonon=1)
    with pytest.raises(ValueError, match="freq_max"):
        generate_beta_grid(0.0, 296.0)
    with pytest.raises(ValueError, match="temperature"):
        generate_beta_grid(0.2, 0.0)


def test_estimate_freq_max_rejects_empty_dos():
    from irma.core.grids import estimate_freq_max_from_dos
    import numpy as np
    with pytest.raises(ValueError, match="empty DOS"):
        estimate_freq_max_from_dos(np.array([]), np.array([]))


# ---- knob validation ---------------------------------------------------------
def test_alpha_grid_rejects_degenerate_log_tail():
    beta = generate_beta_grid(0.2, 296.0)
    # default reaches the kinematic alpha_max through the log tail
    a = generate_alpha_grid(beta, 11.898, 296.0)
    assert a[-1] == pytest.approx(4.0 * beta[-1] / 11.898, rel=1e-12)
    # n_log 0/1 would end the grid silently at q_cut (factor ~67 less reach)
    for n_log in (0, 1):
        with pytest.raises(ValueError, match="n_log"):
            generate_alpha_grid(beta, 11.898, 296.0, n_log=n_log)


def test_beta_grid_rejects_nonfinite_knobs():
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            generate_beta_grid(bad, 296.0)
        with pytest.raises(ValueError, match="finite"):
            generate_beta_grid(0.2, bad)
        with pytest.raises(ValueError, match="finite"):
            generate_beta_grid(0.2, 296.0, beta_max_eV=bad)


def test_alpha_grid_rejects_nonfinite_knobs():
    beta = generate_beta_grid(0.2, 296.0)
    with pytest.raises(ValueError, match="finite"):
        generate_alpha_grid(beta, float("inf"), 296.0)
    with pytest.raises(ValueError, match="finite"):
        generate_alpha_grid(beta, 11.898, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        generate_alpha_grid(beta, 11.898, 296.0, dq_ang_inv=float("inf"))


# ---- the iint-aware beta-grid helper ----------------------------------------
def test_beta_grid_for_iint_loglin_matches_base_grid():
    # iint=0 with the base n_upper reproduces the historical grid exactly
    base = generate_beta_grid(0.2, 296.0, n_upper=20)
    helper = generate_beta_grid_for_iint(0.2, 296.0, iint=0, awr=11.898,
                                         n_upper=20)
    assert np.array_equal(base, helper)


def test_beta_grid_for_iint_linlin_caps_tail():
    # iint=1 wires the step cap: no upper-tail step below the recoil ridge
    # exceeds DELTA_BETA_MAX_LINLIN
    beta = generate_beta_grid_for_iint(0.2, 296.0, iint=1, awr=11.898)
    kT = 8.617333262e-5 * 296.0
    beta_lin_end = (0.2 / kT) * (1.0 - 1.0 / 300.0)
    beta_ridge = 4.0 * beta[-1] / 11.898
    tail = beta[(beta > beta_lin_end) & (beta <= beta_ridge)]
    if tail.size > 1:
        assert np.max(np.diff(tail)) <= DELTA_BETA_MAX_LINLIN * (1 + 1e-12)
    with pytest.raises(ValueError, match="iint"):
        generate_beta_grid_for_iint(0.2, 296.0, iint=3, awr=11.898)


# ---- lat=1 auto-grids anchored at THERM/BK -----------------------------------
def test_grid_reference_temperature():
    from irma.core.constants import BK, THERM
    from irma.core.grids import grid_reference_temperature_K
    assert grid_reference_temperature_K(1, 77.0) == pytest.approx(THERM / BK)
    assert grid_reference_temperature_K(1, 500.0) == pytest.approx(THERM / BK)
    assert grid_reference_temperature_K(0, 77.0) == 77.0


def test_lat1_auto_grid_independent_of_first_temperature():
    """A lat=1 deck's stored alpha/beta are in fixed 0.0253 eV units, so the
    generated grid must not depend on temps[0]."""
    from irma.core.grids import grid_reference_temperature_K
    b77 = generate_beta_grid(0.2, grid_reference_temperature_K(1, 77.0))
    b500 = generate_beta_grid(0.2, grid_reference_temperature_K(1, 500.0))
    np.testing.assert_array_equal(b77, b500)


# ---- lin-lin fine region: kernel width, temperature, and the user note ------
def test_effective_temperature_bound_limits():
    from irma.core.grids import effective_temperature_bound_ratio
    # classical limit: T_eff -> T when kT is far above the phonon cutoff
    assert effective_temperature_bound_ratio(0.001, 3000.0) == pytest.approx(1.0, rel=1e-3)
    # low-temperature limit of a single mode at the cutoff: T_eff -> E_max / 2k
    x_max = 0.2 / (8.617333262e-5 * 10.0)
    assert effective_temperature_bound_ratio(0.2, 10.0) == pytest.approx(x_max / 2.0, rel=1e-6)
    # it bounds a Debye spectrum with the same cutoff (whose T_eff/T is the
    # integral of 3x^2/x_max^3 (x/2)coth(x/2))
    x = np.linspace(0.0, x_max, 20001)[1:]
    debye = np.trapezoid(3.0 * x**2 / x_max**3 * (x / 2.0) / np.tanh(x / 2.0), x)
    assert effective_temperature_bound_ratio(0.2, 10.0) > debye
    with pytest.raises(ValueError):
        effective_temperature_bound_ratio(0.0, 296.0)


def test_linlin_fine_limit_sits_past_the_back_scatter_alpha():
    from irma.core.grids import linlin_fine_beta_limit, RIDGE_MARGIN_SIGMAS
    kT = 8.617333262e-5 * 296.0
    beta_max = 5.0 / kT
    alpha_max = 4.0 * beta_max / 11.898
    limit = linlin_fine_beta_limit(beta_max, 11.898, 0.2, 296.0)
    assert limit > alpha_max
    # the margin scales with the kernel width, so it grows with beta_max and
    # shrinks for a heavier atom
    assert linlin_fine_beta_limit(2.0 * beta_max, 11.898, 0.2, 296.0) > limit
    assert linlin_fine_beta_limit(beta_max, 50.0, 0.2, 296.0) < limit
    assert linlin_fine_beta_limit(beta_max, 11.898, 0.2, 296.0, 0.0) == pytest.approx(alpha_max)
    assert RIDGE_MARGIN_SIGMAS > 0.0
    # the kernel is wider at a hotter evaluation temperature: the limit is
    # taken at the hottest temperature of the deck, and a colder one adds nothing
    hot = linlin_fine_beta_limit(beta_max, 11.898, 0.2, 296.0,
                                 evaluation_temperatures_K=[296.0, 600.0])
    cold = linlin_fine_beta_limit(beta_max, 11.898, 0.2, 296.0,
                                  evaluation_temperatures_K=[77.0, 296.0])
    assert hot > limit
    assert cold == pytest.approx(limit)
    with pytest.raises(ValueError, match="evaluation_temperatures_K"):
        linlin_fine_beta_limit(beta_max, 11.898, 0.2, 296.0, evaluation_temperatures_K=[-5.0])


def test_linlin_grid_fine_step_reaches_the_limit():
    from irma.core.grids import linlin_fine_beta_limit
    kT = 8.617333262e-5 * 296.0
    beta = generate_beta_grid_for_iint(0.2, 296.0, iint=1, awr=11.898)
    limit = linlin_fine_beta_limit(beta[-1], 11.898, 0.2, 296.0)
    beta_lin_end = (0.2 / kT) * (1.0 - 1.0 / 300.0)
    # every upper-tail interval up to the limit, the entrance interval
    # included, is at most the cap, and the limit is reached
    upper = beta[beta >= beta_lin_end * (1 - 1e-12)]
    fine = upper[upper <= limit]
    assert fine.size > 10
    assert np.max(np.diff(fine)) <= DELTA_BETA_MAX_LINLIN * (1 + 1e-12)
    assert fine[-1] >= limit - DELTA_BETA_MAX_LINLIN
    # above the limit the coarse log tail resumes: at least one wider step
    coarse = beta[beta >= limit]
    assert np.max(np.diff(coarse)) > DELTA_BETA_MAX_LINLIN


def test_linlin_cap_scales_with_the_lowest_temperature():
    from irma.core.grids import grid_reference_temperature_K
    t_ref = grid_reference_temperature_K(1, 77.0)
    warm = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898)
    cold, details = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898,
                                                evaluation_temperatures_K=[77.0],
                                                return_details=True)
    kT = 8.617333262e-5 * t_ref
    beta_lin_end = (0.2 / kT) * (1.0 - 1.0 / 300.0)
    ridge = 4.0 * cold[-1] / 11.898
    cold_fine = cold[(cold > beta_lin_end) & (cold <= ridge)]
    # a 77 K law evaluated on a 293.6 K-unit grid needs a step 77/293.6 as large
    assert np.max(np.diff(cold_fine)) <= DELTA_BETA_MAX_LINLIN * 77.0 / t_ref * (1 + 1e-12)
    assert details["stored_cap"] == pytest.approx(DELTA_BETA_MAX_LINLIN * 77.0 / t_ref)
    assert len(cold) > len(warm)
    # a higher evaluation temperature keeps the 0.5 stored cap but widens the
    # fine region, so it never coarsens the grid
    hot = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898,
                                      evaluation_temperatures_K=[600.0])
    assert len(hot) >= len(warm)
    assert np.max(np.diff(hot[(hot > beta_lin_end) & (hot <= ridge)])) <= DELTA_BETA_MAX_LINLIN * (1 + 1e-12)
    # log-lin grids ignore the temperatures entirely
    np.testing.assert_array_equal(
        generate_beta_grid_for_iint(0.2, t_ref, iint=0, awr=11.898, evaluation_temperatures_K=[77.0, 600.0]),
        generate_beta_grid_for_iint(0.2, t_ref, iint=0, awr=11.898))
    with pytest.raises(ValueError, match="evaluation_temperatures_K"):
        generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898, evaluation_temperatures_K=[-1.0])


def test_describe_beta_grid_reports_what_was_built():
    from irma.core.grids import describe_beta_grid, grid_reference_temperature_K
    t_ref = grid_reference_temperature_K(1, 296.0)
    beta, details = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898,
                                                evaluation_temperatures_K=[296.0],
                                                return_details=True)
    note = describe_beta_grid(details)
    assert f"{len(beta)} points" in note and "log-lin grid would have" in note
    assert f"{details['loglin_n_beta']}" in note and "kernel widths" in note
    assert not details["fine_reaches_end"]
    # a light atom's fine step runs to the end of the grid, and the note says so
    light, ldetails = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=1.0,
                                                  return_details=True)
    assert ldetails["fine_reaches_end"]
    assert "whole upper tail" in describe_beta_grid(ldetails)
    # a cold deck reports its stored cap, not 0.5
    _, cdetails = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898,
                                              evaluation_temperatures_K=[77.0],
                                              return_details=True)
    assert f"{cdetails['stored_cap']:.4g}" in describe_beta_grid(cdetails)
    # log-lin
    _, zdetails = generate_beta_grid_for_iint(0.2, t_ref, iint=0, awr=11.898, return_details=True)
    assert "log-lin tail" in describe_beta_grid(zdetails)
    # no upper tail
    _, ndetails = generate_beta_grid_for_iint(0.2, t_ref, iint=1, awr=11.898, n_upper=0,
                                              beta_max_eV=0.1, return_details=True)
    assert "no upper tail" in describe_beta_grid(ndetails)


def test_linlin_grid_survives_the_deck_precision_round_trip():
    """A seam node within deck precision of the phonon-region end would be
    written as a duplicate value and rejected by the engine; the generator
    drops it. Constructed case from the 2026-09-15 review (AWR 238)."""
    from irma.core.grids import grid_reference_temperature_K, _drop_nodes_indistinct_in_a_deck
    t_ref = grid_reference_temperature_K(1, 296.0)
    beta = generate_beta_grid_for_iint(0.40507000876176497, t_ref, iint=1, awr=238.0,
                                       n_lower=15, n_phonon=300, n_upper=80, beta_max_eV=5.0)
    printed = np.array([float(f"{x:.6e}") for x in beta])
    assert np.all(np.diff(printed) > 0.0)
    assert np.all(np.diff(beta) > 0.0)
    # the helper itself keeps the later node of a pair, so the end node survives
    kept = _drop_nodes_indistinct_in_a_deck(np.array([0.0, 1.0, 1.0 + 1e-9, 2.0]))
    np.testing.assert_allclose(kept, [0.0, 1.0 + 1e-9, 2.0])
