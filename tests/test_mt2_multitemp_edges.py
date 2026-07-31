"""Characterization of the multi-temperature coherent-elastic MT2 edge folding.

The subtle spot is the additional-temperature LIST-block folding in
``_coherent_s_table`` (endf_writer.py): a suspicious-looking detail is that
once ``j >= jmax`` the running edge index ``jj`` stays frozen at ``jmax-1`` and
the code re-evaluates ``edge_delta(jj, ...)`` for every folded edge, "duplicating
the last retained edge" instead of accumulating each folded *raw* edge ``j``.
Because the first-temperature (T0) block instead folds every raw ``j`` delta into
the last retained point, the report suspected the LT blocks could disagree.

This file is a REFUTATION pin. ``_coherent_s_table`` is a deliberate, byte-level
reproduction of NJOY-LEAPR's ``endout`` (NJOY2016 src/leapr.f90, the ``iel.ge.1``
coherent-elastic emission). The same T0/LT asymmetry exists *in NJOY itself*:

  * T0 block (i==1)::

        e = bragg(1+2*j-2)                       ! RAW edge-j energy
        scr(...) = sum + exp(-4*w*e)*bragg(1+2*j-1)   ! RAW edge-j delta
        ... (when j>jmax, jj frozen -> overwrite the last retained slot)

  * LT block (i>1)::

        if (j.le.jmax) jj=jj+1
        e = sigfig(bragg(1+2*jj-2),7,0)          ! jj (frozen at jmax for j>jmax)
        scr(...) = sum + exp(-4*w*e)*bragg(1+2*jj-1)  ! jj's delta -> re-adds last edge

So NJOY *also* re-adds the frozen last edge in every additional-temperature
block. The T0/LT difference at the final retained point is intentional: the
thinned high-energy tail is attenuated by a temperature-dependent Debye-Waller
factor, so its folded contribution legitimately differs per temperature. The
IRMA graphite native tape reproduces a real multi-temperature NJOY coherent tape
bit-for-bit (tests/native_LEAPR_NJOY_ENDF_validation/), which would be impossible
if this folding were wrong.

The tests below reimplement ``endout`` independently and assert IRMA matches it
for a thinning multi-temperature case, and additionally pin that IRMA does NOT
match the proposed "accumulate each raw edge into the correct retained slot" fix
(which would silently break the byte-exact NJOY reproduction).
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


def _proposed_fix_lt(bragg, nedge, jmax, itemp, dwpix):
    """The Codex-suggested 'fix': fold each RAW edge-j into the last retained slot.

    Pinned here only to prove IRMA deliberately does NOT do this.
    """
    s = [None] * jmax
    ssum = 0.0
    jj = 0
    for j in range(1, nedge + 1):
        if j <= jmax:
            jj += 1
        e = sigfig(bragg[j - 1][0], 7, 0)  # RAW edge j, not frozen jj
        ssum += exp(-4.0 * dwpix[itemp] * e) * bragg[j - 1][1]
        s[jj - 1] = sigfig(ssum, 7, 0)
    return s


# Per-temperature Debye-Waller integrals (eV^-1), decreasing with temperature,
# chosen so the high-energy tail thins (jmax << nedge) at every temperature.
DWPIX = [10.0, 7.0, 5.0]
TEMPR = [296.0, 600.0, 1000.0]


def test_thinning_actually_occurs():
    """The scenario must exercise the folding branch (j >= jmax) or it proves
    nothing. Graphite at these DW values thins ~345 -> ~146 edges."""
    bragg, nedge = _graphite_edges()
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, DWPIX))
    jmax = out['NP']
    assert jmax < nedge, "no edges thinned — folding branch never exercised"
    assert nedge - jmax > 50, "expected substantial folding for graphite"


def test_t0_block_matches_njoy_endout():
    """First-temperature TAB1 (energies + cumulative S) is byte-faithful to NJOY."""
    bragg, nedge = _graphite_edges()
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, DWPIX))
    jmax, e_ref, s_ref, _ = _njoy_endout_reference(bragg, nedge, len(TEMPR), DWPIX)

    assert out['NP'] == jmax
    assert out['S_T0_table']['Eint'] == e_ref
    assert out['S_T0_table']['S'] == s_ref


@pytest.mark.parametrize("itemp", [1, 2])
def test_lt_block_matches_njoy_endout(itemp):
    """Additional-temperature LIST blocks are byte-faithful to NJOY endout.

    This is the heart of the C4 refutation: the 'duplicate the last retained
    edge' folding IS NJOY's documented behavior, so IRMA reproducing it is
    correct, not a bug.
    """
    bragg, nedge = _graphite_edges()
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, DWPIX))
    jmax, _, _, lt_ref = _njoy_endout_reference(bragg, nedge, len(TEMPR), DWPIX)

    irma_lt = [out['S'][q][itemp] for q in range(1, jmax + 1)]
    assert irma_lt == lt_ref[itemp]


def test_lt_block_is_not_the_proposed_accumulate_fix():
    """Pin that IRMA does NOT silently switch to the 'accumulate each raw edge'
    variant. That variant diverges from NJOY at the last retained point, so
    adopting it would break the byte-exact native graphite tape."""
    bragg, nedge = _graphite_edges()
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, DWPIX))
    jmax = out['NP']

    irma_lt1 = [out['S'][q][1] for q in range(1, jmax + 1)]
    fixed_lt1 = _proposed_fix_lt(bragg, nedge, jmax, 1, DWPIX)

    # They share every fully-retained edge (j <= jmax) and differ only where
    # folding kicks in — at the final retained slot.
    assert irma_lt1[:-1] == fixed_lt1[:-1]
    assert irma_lt1[-1] != fixed_lt1[-1], \
        "IRMA unexpectedly matches the accumulate-raw-edges variant"


def test_no_thinning_t0_and_lt_use_every_edge():
    """Control case: with negligible Debye-Waller attenuation nothing thins, so
    every raw edge is retained and the T0/LT folding asymmetry cannot arise.
    Here all blocks must agree edge-for-edge with the reference."""
    bragg, nedge = _graphite_edges()
    dwpix = [1e-12, 1e-12, 1e-12]  # essentially no attenuation -> no thinning
    out = _coherent_s_table(bragg, nedge, len(TEMPR), TEMPR,
                            _make_edge_delta(bragg, dwpix))
    jmax, e_ref, s_ref, lt_ref = _njoy_endout_reference(
        bragg, nedge, len(TEMPR), dwpix)

    assert out['NP'] == jmax == nedge
    assert out['S_T0_table']['Eint'] == e_ref
    assert out['S_T0_table']['S'] == s_ref
    for itemp in (1, 2):
        irma_lt = [out['S'][q][itemp] for q in range(1, jmax + 1)]
        assert irma_lt == lt_ref[itemp]
