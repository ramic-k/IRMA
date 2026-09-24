"""Regression tests for the coherent-elastic Bragg-edge grouping (ENDF-102 7.2.2).

The grouping must: keep edges <= threshold, collapse the dense edges above the
threshold into ~bins_per_decade*log10(emax/threshold) steps, and preserve the
cumulative S (total bound XS) and the 1/E integral exactly at the reference T.
"""
import numpy as np

from irma.core.endf_writer import _grouped_coherent_s_table
from irma.core.kernels import sigfig


def _make_delta_fn(E, s_raw, W):
    """edge_delta_fn(j, itemp): DW-weighted increment at each edge's own energy."""
    def fn(j, it, energy=None):
        return float(np.exp(-4.0 * W[it] * E[j]) * s_raw[j])
    return fn


def _group(dense_bragg, bins_per_decade=20, threshold=1.0, temps=(296.0,), W=(0.02,)):
    E, s_raw, emax = dense_bragg
    bragg = [(float(E[j]), 0.0) for j in range(len(E))]
    fn = _make_delta_fn(E, s_raw, list(W))
    table = _grouped_coherent_s_table(bragg, len(E), len(temps), list(temps),
                                      fn, threshold, bins_per_decade)
    return E, emax, table, fn


def _integ(Eg, Sg):
    """1/E integral of the histogram-S staircase = INT sigma dE."""
    Eg = np.asarray(Eg, float); Sg = np.asarray(Sg, float)
    return float(np.sum(Sg[:-1] * np.log(Eg[1:] / Eg[:-1])))


def test_total_S_preserved_single_temp(dense_bragg):
    """Cumulative S at the last point equals the (sigfig-rounded) sum of all increments."""
    E, emax, table, fn = _group(dense_bragg)
    total_true = sum(fn(j, 0) for j in range(len(E)))
    assert table["S_T0_table"]["S"][-1] == sigfig(total_true, 7, 0)


def test_subthreshold_edges_kept(dense_bragg):
    """Every edge at or below the threshold is kept individually."""
    E, emax, table, fn = _group(dense_bragg, threshold=1.0)
    En = np.asarray(table["S_T0_table"]["Eint"])
    n_below = int(np.sum(E[E < emax] <= 1.0))
    assert int(np.sum(En <= 1.0)) == n_below


def test_grouped_count_matches_bins_per_decade(dense_bragg):
    """Steps above the threshold ~ bins_per_decade * log10(emax/threshold)."""
    E, emax, table, fn = _group(dense_bragg, bins_per_decade=20, threshold=1.0)
    En = np.asarray(table["S_T0_table"]["Eint"])
    n_above = int(np.sum(En > 1.0))                 # includes the emax endpoint
    expected = 20 * np.log10(emax / 1.0)            # ~14
    assert expected - 1 <= n_above <= expected + 2


def test_energies_ascending_and_endpoint(dense_bragg):
    E, emax, table, fn = _group(dense_bragg)
    En = np.asarray(table["S_T0_table"]["Eint"])
    assert table["NP"] == len(En)
    assert np.all(np.diff(En) > 0)                  # strictly ascending
    assert np.isclose(En[-1], emax)                 # defined up to emax
    assert np.all(np.diff(np.asarray(table["S_T0_table"]["S"])) >= -1e-12)  # S non-decreasing


def test_integral_fidelity_at_T0(dense_bragg):
    """The log-mean E_rep placement preserves the 1/E integral at the reference T."""
    E, emax, table, fn = _group(dense_bragg)
    cum = np.cumsum([fn(j, 0) for j in range(len(E))])
    I_ungrouped = _integ(E, cum)
    I_grouped = _integ(table["S_T0_table"]["Eint"], table["S_T0_table"]["S"])
    assert abs(I_grouped - I_ungrouped) / I_ungrouped < 1e-6


def test_multi_temperature_structure(dense_bragg):
    """Two temperatures share one energy grid; total preserved at every temperature."""
    E, emax, table, fn = _group(dense_bragg, temps=(296.0, 1000.0), W=(0.02, 0.05))
    NP = table["NP"]
    assert table["LI"] == 2
    assert set(table["T"].keys()) == {1}
    assert set(table["S"].keys()) == set(range(1, NP + 1))
    total_T1 = sum(fn(j, 1) for j in range(len(E)))
    assert abs(table["S"][NP][1] - total_T1) / total_T1 < 1e-5


def test_allzero_underflow_bins_merge_not_ungroup():
    """Edges whose increments underflowed to exactly 0.0 at EVERY temperature
    carry no cross section: they must merge into one zero step per log bin,
    not ungroup into one redundant point per raw edge (in the
    extreme-Debye-Waller regime that makes the tape about 100 times larger)."""
    n_zero = 600
    E_sub = np.array([0.5, 0.8])                    # kept individually
    E_zero = np.geomspace(1.5, 4.5, n_zero)        # all-zero above threshold
    E = np.concatenate([E_sub, E_zero, [5.0]])      # flat emax endpoint
    bragg = [(float(e), 0.0) for e in E]

    def fn(j, it):
        return 0.1 if E[j] <= 1.0 else 0.0          # zero at every T above 1 eV

    table = _grouped_coherent_s_table(bragg, len(E), 2, [296.0, 500.0],
                                      fn, 1.0, 20)
    En = np.asarray(table["S_T0_table"]["Eint"], float)
    Sn = np.asarray(table["S_T0_table"]["S"], float)
    # ~bins_per_decade*log10(4.5/1.0) ~ 13 zero bins + 2 sub-threshold + emax:
    # far fewer points than one per raw edge
    assert len(En) < 30, f"zero bins did not merge: {len(En)} points"
    assert np.all(np.diff(En) > 0)
    assert En[-1] == sigfig(5.0, 7, 0)
    # the zero tail adds nothing: cumulative S stays at the sub-threshold sum
    assert Sn[-1] == sigfig(0.2, 7, 0)
    assert np.all(np.diff(Sn) >= 0)                 # monotone
    assert np.all(Sn[1:] == Sn[-1])                 # flat after the sub-threshold edges


def test_cancellation_bin_with_nonzero_increments_still_ungroups():
    """A bin whose increments are nonzero but sum to <= 0 (negative
    interference or roundoff) ungroups, so every emitted step stays
    physical."""
    E_sub = np.array([0.5])
    E_pair = np.array([2.0, 2.05])                  # same log bin at 20/decade
    E = np.concatenate([E_sub, E_pair, [5.0]])
    bragg = [(float(e), 0.0) for e in E]
    d = {0: 0.1, 1: 1.0e-3, 2: -1.0e-3, 3: 0.0}     # pair cancels exactly

    def fn(j, it):
        return d[j]

    table = _grouped_coherent_s_table(bragg, len(E), 1, [296.0], fn, 1.0, 20)
    En = np.asarray(table["S_T0_table"]["Eint"], float)
    # both pair edges survive individually (4 points: sub, 2.0, 2.05, emax)
    assert len(En) == 4
    assert sigfig(2.0, 7, 0) in En and sigfig(2.05, 7, 0) in En
