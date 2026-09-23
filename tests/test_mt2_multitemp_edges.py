"""IRMA's multi-temperature coherent-elastic MT2 table (_coherent_s_table)
matches an independent reimplementation of NJOY-LEAPR endout for a graphite
case whose high-energy tail thins. In both codes the additional-temperature
LIST blocks re-add the frozen last retained edge (leapr.f90, iel.ge.1).
"""
from math import exp

import pytest

from irma.core.engine import coher
from irma.core.kernels import sigfig
from irma.core.endf_writer import _coherent_s_table

TOL = 0.9e-7  # NJOY endout edge-thinning tolerance


def _graphite_edges():
    """Real built-in graphite Bragg edges (345 raw, thins heavily)."""
    bragg, nedge = coher(1, 1, 5.0)
    return bragg, nedge


def _make_edge_delta(bragg, dwpix):
    """Classic isotropic Debye-Waller edge weight (mirrors _build_coherent_elastic).

    ``dwpix`` is the per-temperature Debye-Waller integral (eV^-1). The
    ``energy`` override exists so the LT path can feed the sigfig'd energy into
    the exponential exactly as NJOY's endout does.
    """
    def edge_delta(j, itemp, energy=None):
        e = energy if energy is not None else bragg[j][0]
        return exp(-4.0 * dwpix[itemp] * e) * bragg[j][1]
    return edge_delta


def _njoy_endout_reference(bragg, nedge, ntempr, dwpix):
    """Independent reimplementation of NJOY-LEAPR endout coherent-elastic.

    Returns ``(jmax, E_T0, S_T0, {itemp: S_LT})`` mirroring leapr.f90 exactly.
    """
    # --- thinning at the first temperature ---
    ssum = 0.0
    suml = 0.0
    jmax = 0
    for j in range(1, nedge + 1):
        e = bragg[j - 1][0]
        ssum += exp(-4.0 * dwpix[0] * e) * bragg[j - 1][1]
        if ssum - suml > TOL * ssum:
            jmax = j
            suml = ssum

    # --- T0 (i==1): raw edge-j energy and delta; jj frozen overwrites last ---
    e_t0 = []
    s_t0 = []
    ssum = 0.0
    jj = 0
    for j in range(1, nedge + 1):
        e = bragg[j - 1][0]
        if j <= jmax:
            jj += 1
        val = ssum + exp(-4.0 * dwpix[0] * e) * bragg[j - 1][1]
        ssum = val  # un-sigfig'd cumulative carries forward (NJOY: sum=scr before sigfig)
        if jj > len(s_t0):
            e_t0.append(sigfig(e, 7, 0))
            s_t0.append(sigfig(val, 7, 0))
        else:
            e_t0[jj - 1] = sigfig(e, 7, 0)
            s_t0[jj - 1] = sigfig(val, 7, 0)

    # --- LT (i>1): jj-frozen energy/delta (re-adds the last retained edge) ---
    lt = {}
    for itemp in range(1, ntempr):
        s = [None] * jmax
        ssum = 0.0
        jj = 0
        for j in range(1, nedge + 1):
            if j <= jmax:
                jj += 1
            e = sigfig(bragg[jj - 1][0], 7, 0)
            val = ssum + exp(-4.0 * dwpix[itemp] * e) * bragg[jj - 1][1]
            ssum = val
            s[jj - 1] = sigfig(val, 7, 0)
        lt[itemp] = s
    return jmax, e_t0, s_t0, lt


# Per-temperature Debye-Waller integrals (eV^-1), decreasing with temperature,
# chosen so the high-energy tail thins (jmax << nedge) at every temperature.
DWPIX = [10.0, 7.0, 5.0]
TEMPR = [296.0, 600.0, 1000.0]


def test_t0_block_matches_njoy_endout():
    """First-temperature TAB1 (energies + cumulative S) is byte-faithful to NJOY."""
    bragg, nedge = _graphite_edges()
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, DWPIX))
    jmax, e_ref, s_ref, _ = _njoy_endout_reference(bragg, nedge, len(TEMPR), DWPIX)

    assert out['NP'] == jmax
    assert nedge - jmax > 50            # the folding branch (j > jmax) is exercised
    assert out['S_T0_table']['Eint'] == e_ref
    assert out['S_T0_table']['S'] == s_ref


@pytest.mark.parametrize("itemp", [1, 2])
def test_lt_block_matches_njoy_endout(itemp):
    """Additional-temperature LIST blocks are byte-faithful to NJOY endout."""
    bragg, nedge = _graphite_edges()
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, DWPIX))
    jmax, _, _, lt_ref = _njoy_endout_reference(bragg, nedge, len(TEMPR), DWPIX)

    irma_lt = [out['S'][q][itemp] for q in range(1, jmax + 1)]
    assert irma_lt == lt_ref[itemp]
