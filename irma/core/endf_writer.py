"""ENDF-6 output for IRMA (MF1/MT451, MF7/MT2, MF7/MT4).

write_endf_output assembles the tape via endf-parserpy. The MF7/MT2 elastic
section comes from one of the builders: the classic LEAPR coherent table
(_build_coherent_elastic, iel=1-6), the generalized CEF/MEF builders
(iel=10), or the incoherent-elastic builder (iel<0). MF7/MT4 stores
the symmetric law S*exp(-beta/2) via _endf_s (isym/ilog variants).

CEF in identifiers is the single-channel elastic format the docs and GUI call SEF.
"""

import sys
import warnings
from collections import Counter

import numpy as np
from math import exp, log, sqrt

from irma.core.constants import BK, THERM
from irma.core.deck import DeckError
from irma.core.kernels import sigfig
from irma.core.elastic_dw import resolve_species_dw, make_edge_delta


# math.exp raises OverflowError ("math range error") for any argument
# strictly above ln(DBL_MAX) = 709.782712893384. The isym=1 asymmetric-law
# overflow handling below and the Card 10 parse-time early warning in
# driver.py both derive their thresholds from this ceiling.
_LN_FLOAT_MAX = log(sys.float_info.max)

# ln of the largest real value that rounds to float64 0.0 (2**-1075): the
# ceiling on what a stored exact-zero S(alpha,beta) could have been before
# it underflowed.
_LN_MAX_STORED_ZERO = -1075.0 * log(2.0)


def _writer():
    """The compiled endf-parserpy writer, or the pure-Python one (same bytes)."""
    try:
        from endf_parserpy import EndfParserCpp
        return EndfParserCpp()
    except ImportError:
        from endf_parserpy import EndfParserPy
        return EndfParserPy()


def _patch_mf1_directory_counts(lines, nwd, sections):
    """Set each MF1/MT451 directory NCx to the section's written record count.

    The directory entries follow the tape header line, the 4 MT451 header
    records and the NWD text records. SEND/FEND/MEND/TEND lines (MF or MT 0)
    are not counted.
    """
    counts = Counter(ln[70:75] for ln in lines
                     if ln[70:72] != ' 0' and ln[72:75] != '  0')
    first = 1 + 4 + nwd
    for k, (mf, mt) in enumerate(sections):
        ln = lines[first + k]
        lines[first + k] = ln[:44] + str(counts[f"{mf:2d}{mt:3d}"]).rjust(11) + ln[55:]
    return lines


def write_endf_output(filename, mat, za, awr, spr, npr, iel, nss,
                      b7, aws, sps, mss, nalpha, nbeta, lat,
                      alpha, beta, ssm, ssp, tempr, ntempr,
                      dwpix, dwp1, tempf, tempf1,
                      bragg, nedge, isym, ilog, smin,
                      iint=0, comments=None, crystal_info=None):
    """Write ENDF-6 output file using endf-parserpy."""
    # MF7/MT4 interpolation for both the alpha and beta tables: iint 0 ->
    # log-lin (INT=4, NJOY), 1 -> lin-lin (INT=2).
    coh_int = 2 if iint == 1 else 4

    # Bound cross section of the principal scatterer.
    sb = spr * ((1.0 + awr) / awr)**2

    # Compute Debye-Waller integral for ENDF output
    dwpix_out = dwpix.copy()
    dwp1_out = dwp1.copy()
    for i in range(ntempr):
        if nss == 0 or b7 > 0.0:
            dwpix_out[i] = dwpix[i] / (awr * tempr[i] * BK)
        else:
            # bound two-pass secondary: W' is the secondary's, as NJOY's endout writes it
            dwpix_out[i] = dwpix[i] / (aws * tempr[i] * BK)
            dwp1_out[i] = dwp1[i] / (awr * tempr[i] * BK)

    # Determine symmetry type
    # isym: 0 = symmetric S, 1 = S for +/- beta (coldh),
    #        2 = ss for -beta, 3 = ss for +/- beta

    parser = _writer()

    # ---- MF1/MT451 ----
    mf1 = {}
    mf1['MAT'] = mat
    mf1['MF'] = 1
    mf1['MT'] = 451
    mf1['ZA'] = za
    mf1['AWR'] = awr
    mf1['LRP'] = -1
    mf1['LFI'] = 0
    mf1['NLIB'] = 0
    mf1['NMOD'] = 0
    mf1['ELIS'] = 0.0
    mf1['STA'] = 0.0
    mf1['LIS'] = 0
    mf1['LISO'] = 0
    mf1['NFOR'] = 6
    mf1['AWI'] = 1.0
    mf1['EMAX'] = 0.0
    mf1['LREL'] = 0
    mf1['NSUB'] = 12  # thermal scattering sub-library
    mf1['NVER'] = 6
    mf1['TEMP'] = 0.0
    mf1['LDRV'] = 0

    # Text records: card 1 is ZSYMAM(11) ALAB(11) EDATE(10) AUTH(33), card 2
    # REF(21) DDATE(10) RDATE(10) ENDATE(8), cards 3-5 HSUB, cards 6+ the
    # description. rstrip only: column 1 of ZSYMAM stays blank (NJOY).
    text = []
    for c in comments or []:
        c = c.rstrip()
        if c[:1] in ("'", '"') and c.endswith(c[:1]):
            c = c[1:-1]
        text.append(c.ljust(66)[:66])
    text += [' ' * 66] * (5 - len(text))
    r1, r2 = text[0], text[1]
    if (r2[43:55] + r2[63:66]).strip():
        warnings.warn(
            "MF1/MT451 comment card 2 has text in columns that map to no ENDF "
            f"field and is dropped ({(r2[43:55] + r2[63:66]).strip()!r}); put "
            "free text in card 6 onward.", stacklevel=2)
    mf1.update(ZSYMAM=r1[:11], ALAB=r1[11:22], EDATE=r1[22:32], AUTH=r1[33:66],
               REF=r2[1:22], DDATE=r2[22:32], RDATE=r2[33:43], ENDATE=r2[55:63],
               HSUB={1: text[2], 2: text[3], 3: text[4]}, NWD=len(text),
               DESCRIPTION={i + 1: t for i, t in enumerate(text[5:])})

    # Directory: MF1/MT451, MF7/MT2 if present, MF7/MT4. The NCx record
    # counts are filled in after the tape is written.
    sections = [(1, 451)] + ([(7, 2)] if iel != 0 else []) + [(7, 4)]
    mf1['NXC'] = len(sections)
    mf1['MFx'] = {k + 1: mf for k, (mf, _) in enumerate(sections)}
    mf1['MTx'] = {k + 1: mt for k, (_, mt) in enumerate(sections)}
    mf1['NCx'] = {k + 1: 0 for k in range(len(sections))}
    mf1['MOD'] = {k + 1: 0 for k in range(len(sections))}

    # ---- MF7/MT4 (inelastic) ----
    mf7mt4 = {}
    mf7mt4['MAT'] = mat
    mf7mt4['MF'] = 7
    mf7mt4['MT'] = 4
    mf7mt4['ZA'] = za
    mf7mt4['AWR'] = awr
    mf7mt4['LAT'] = lat
    mf7mt4['LASYM'] = isym
    mf7mt4['LLN'] = 1 if ilog != 0 else 0
    # NS = number of non-principal scatterers (sizes the B array: NI=6(NS+1))
    ns_val = nss if nss > 0 else 0
    mf7mt4['NS'] = ns_val

    ni = 6
    if ns_val > 0:
        ni = 6 * (ns_val + 1)
    mf7mt4['NI'] = ni

    # B array — layout follows NJOY's endout (leapr.f90):
    #   B[1] = npr*spr (free-atom cross section x principal atom count)
    #   B[2] = beta_max          B[3] = AWR
    #   B[4] = 0.0253*beta_max (E_max, eV)
    #   B[5] = 0                 B[6] = npr
    # and, when a secondary scatterer is present (nss>0):
    #   B[7] = b7 (0=SCT, 1=free gas, 2=diffusion)
    #   B[8] = mss*sps           B[9] = aws
    #   B[10] = B[11] = 0        B[12] = mss
    mf7mt4['B'] = {}
    mf7mt4['B'][1] = npr * spr
    mf7mt4['B'][2] = beta[nbeta - 1]
    mf7mt4['B'][3] = awr
    mf7mt4['B'][4] = sigfig(THERM * beta[nbeta - 1], 7, 0)
    mf7mt4['B'][5] = 0.0
    mf7mt4['B'][6] = float(npr)

    if ns_val > 0:
        mf7mt4['B'][7] = b7
        mf7mt4['B'][8] = mss * sps
        mf7mt4['B'][9] = aws
        mf7mt4['B'][10] = 0.0
        mf7mt4['B'][11] = 0.0
        mf7mt4['B'][12] = float(mss)

    # Beta grid
    nbt = nbeta
    if isym == 1 or isym == 3:
        nbt = 2 * nbeta - 1
    mf7mt4['NB'] = nbt
    mf7mt4['beta_interp'] = {'NBT': [nbt], 'INT': [coh_int]}

    mf7mt4['beta'] = {}
    mf7mt4['LT'] = {}
    mf7mt4['S_table'] = {}
    mf7mt4['T0'] = tempr[0]

    if ntempr > 1:
        mf7mt4['T'] = {j: tempr[j] for j in range(1, ntempr)}
        mf7mt4['LI'] = {j: 4 for j in range(1, ntempr)}
        mf7mt4['NP'] = nalpha
        # S[q][i][j]: q=alpha(1..NP), i=beta(1..NB), j=temp(1..LT)
        mf7mt4['S'] = {q: {} for q in range(1, nalpha + 1)}

    sc_vals = np.ones(ntempr)
    for nt in range(ntempr):
        if lat == 1:
            sc_vals[nt] = THERM / (BK * tempr[nt])

    # Low-temperature underflow detector (isym=0, ilog=0): beta grows like 1/T,
    # so S*exp(-beta/2) at large transfer can fall below what ENDF can hold
    # while the scattering there is significant. Counted per temperature.
    _lln_loss_n = [0] * ntempr
    _lln_loss_emax = [0.0] * ntempr
    _lln_check = (isym == 0 and ilog == 0 and ssm is not None
                  and getattr(ssm, "size", 0) > 0)
    _lln_sig = float(np.abs(ssm).max()) * 1.0e-4 if _lln_check else 0.0
    _ENDF_S_FLOOR = 1.0e-90

    for ii in range(1, nbt + 1):
        # Beta row ii: isym 1/3 store -beta (from ssm) then +beta (from ssp).
        if isym % 2 == 0:
            src, i_beta, beta_val = ssm, ii - 1, beta[ii - 1]
        elif ii < nbeta:
            src, i_beta, beta_val = ssm, nbeta - ii, -beta[nbeta - ii]
        else:
            src, i_beta, beta_val = ssp, ii - nbeta, beta[ii - nbeta]

        mf7mt4['beta'][ii] = beta_val
        mf7mt4['LT'][ii] = ntempr - 1

        for nt in range(ntempr):
            be = beta_val * sc_vals[nt]
            vals = [_endf_s(src[i_beta, j, nt], be, isym, ilog, smin,
                            beta_card=beta_val, temp_k=tempr[nt])
                    for j in range(nalpha)]
            # Points with significant scattering that the linear symmetric
            # storage underflows.
            if _lln_sig > 0.0:
                for j in range(nalpha):
                    if ssm[i_beta, j, nt] > _lln_sig and 0.0 <= vals[j] < _ENDF_S_FLOOR:
                        _lln_loss_n[nt] += 1
                        _lln_loss_emax[nt] = max(_lln_loss_emax[nt],
                                                 abs(be) * BK * tempr[nt] * 1000.0)
            if nt == 0:
                mf7mt4['S_table'][ii] = {
                    'NBT': [nalpha],
                    'INT': [coh_int],
                    'alpha': alpha.tolist(),
                    'S': [float(v) for v in vals],
                }
            else:
                for q in range(nalpha):
                    mf7mt4['S'][q + 1].setdefault(ii, {})[nt] = vals[q]

    for _nt in range(ntempr):
        if _lln_loss_n[_nt] > 0:
            print(
                f"WARNING: at T={tempr[_nt]:g} K, {_lln_loss_n[_nt]} S(alpha,beta) "
                f"points with significant scattering (up to "
                f"{_lln_loss_emax[_nt]:.0f} meV transfer) underflow the linear "
                f"ilog=0 storage and are written as 0; set ilog=1 on Card 4.",
                flush=True,
            )

    # Effective temperature table(s).
    # Teff0 is the PRINCIPAL scatterer's effective temperature. Where it lives
    # depends on how many scatterer passes ran (mirroring NJOY's temp arrays):
    #   * single pass (nss=0, or an analytic b7>0 secondary): the principal's
    #     values are in tempf; tempf1 is never filled.
    #   * two passes (nss>0 with an SCT b7<=0 secondary): the principal's
    #     values were saved into tempf1 before the second pass overwrote
    #     tempf with the secondary's, and the secondary gets its own Teff1
    #     table (ENDF-102: only B(7)=0 secondaries carry one).
    if nss != 0 and b7 <= 0.0:
        teff_principal = tempf1
        mf7mt4['teff1_table'] = {
            'NBT': [ntempr],
            'INT': [2],
            'Tint': [sigfig(tempr[i], 7, 0) for i in range(ntempr)],
            'Teff1': [sigfig(tempf[i], 7, 0) for i in range(ntempr)]
        }
    else:
        teff_principal = tempf

    mf7mt4['teff0_table'] = {
        'NBT': [ntempr],
        'INT': [2],
        'Tint': [sigfig(tempr[i], 7, 0) for i in range(ntempr)],
        'Teff0': [sigfig(teff_principal[i], 7, 0) for i in range(ntempr)]
    }

    # ---- MF7/MT2 (elastic) ----
    mf7mt2 = None
    if iel == 10:
        mf7mt2 = _build_generalized_elastic(
            mat, za, awr, bragg, nedge, ntempr, tempr,
            crystal_info, dwpix_out, sb, npr=npr)
    elif iel < 0:
        mf7mt2 = _build_cef_incoherent(mat, za, awr, ntempr, tempr, dwpix_out,
                                       sb, npr)
    elif iel >= 1:
        # Built-in coherent elastic (iel=1-6)
        mf7mt2 = _build_coherent_elastic(mat, za, awr, bragg, nedge, ntempr,
                                          tempr, dwpix_out, dwp1_out, nss, b7)

    # ---- Assemble and write ----
    endf_dict = {}
    endf_dict[0] = {0: {'MAT': 1, 'MF': 0, 'MT': 0,
                         'TAPEDESCR': ' ' * 66}}
    endf_dict[1] = {451: mf1}
    endf_dict[7] = {}
    if mf7mt2 is not None:
        endf_dict[7][2] = mf7mt2
    endf_dict[7][4] = mf7mt4

    parser.writefile(filename, endf_dict, overwrite=True)

    # endf-parserpy writes zeros in the data fields of SEND/FEND/MEND/TEND
    # records, NJOY leaves them blank.
    with open(filename, 'r') as f:
        lines = [' ' * 66 + ln[66:] if ln[72:75] == '  0' else ln for ln in f]
    lines = _patch_mf1_directory_counts(lines, mf1['NWD'], sections)
    # newline='\n' keeps the tapes LF-only on every platform.
    with open(filename, 'w', newline='\n', encoding='ascii') as f:
        f.writelines(lines)
        # endf-parserpy does not end the file with a newline; NJOY does.
        if lines and not lines[-1].endswith('\n'):
            f.write('\n')
    print(f"ENDF output written to {filename}", flush=True)

def _asym_overflow_s(s_stored, be, beta_card, temp_k, smin):
    """Evaluate the asymmetric-law product S*exp(be/2) when exp overflows.

    Reached only from the isym=1 (cold-hydrogen/deuterium, LASYM=1) ilog=0
    branches of :func:`_compute_endf_s`, and only when ``math.exp(be/2)``
    alone exceeds float64 (be/2 > ln(DBL_MAX) ~ 709.78 -- lat=1 at
    cryogenic temperature scales the stored beta by THERM/(k*T)). The
    PRODUCT is usually still representable: by detailed balance the stored
    value is ~exp(-be/2), so the direct-form crash is an evaluation-order
    artifact. Evaluate exp(log(S) + be/2) instead -- the same quantity the
    isym=1 ilog=1 branch writes in log form without ever overflowing.
    Where no evaluation order can represent the point (the
    stored S underflowed to 0, or the true product genuinely exceeds
    float64), raise a DeckError naming the card, the beta value, and the
    remedy (Card 4 ilog=1, LLN=1 log storage) instead of letting a bare
    OverflowError escape after the expensive kernel run.
    """
    remedy = ("set ilog=1 (LLN=1 log storage) on Card 4; see "
              "examples/lln_low_temperature_demo.py")
    context = (f"cold-hydrogen/deuterium asymmetric law (Card 5 ncold>0, "
               f"LASYM=1) at Card 9 beta={abs(beta_card):g} "
               f"(scaled beta*THERM/(k*T) = {be:g} at T={temp_k:g} K)")
    if s_stored <= 0.0:
        # The internal S(alpha,beta) underflowed to exactly 0 before the
        # writer ever saw it, so S*exp(be/2) is unrecoverable here. The
        # true product is bounded by exp(be/2 + ln(2^-1075)) (the largest
        # value that stores as 0, times exp(be/2)); only when even that
        # ceiling falls below the Card 4 smin cutoff -- which zeroes the
        # written value regardless -- is 0 provably what the tape would
        # hold anyway.
        if smin > 0.0 and be / 2.0 + _LN_MAX_STORED_ZERO <= log(smin):
            return 0.0
        raise DeckError(
            f"{context}: the stored S(alpha,beta) has underflowed to 0, so "
            f"the asymmetric law S*exp(beta/2) cannot be recovered in "
            f"linear (ilog=0) storage -- {remedy}")
    arg = log(s_stored) + be / 2.0
    try:
        return exp(arg)
    except OverflowError:
        raise DeckError(
            f"{context}: the asymmetric law S*exp(beta/2) = exp({arg:.1f}) "
            f"exceeds the largest float64 (~1.8e308) and cannot be written "
            f"in linear (ilog=0) storage -- {remedy}") from None


def _endf_s(s, be, isym, ilog, smin, beta_card=0.0, temp_k=0.0):
    """The MF7/MT4 value written for S(alpha, beta) = ``s`` at scaled beta ``be``.

    isym 0 stores S*exp(-be/2), isym 1 S*exp(be/2) (``be`` is signed), isym 2
    and 3 store S itself; ilog=1 stores the logarithm. ``beta_card`` and
    ``temp_k`` only label the isym=1 overflow error.

    Deliberate NJOY divergence: zero S is written as ln S = -999 at every
    temperature. NJOY writes 0 in its isym=0 additional-temperature records
    (leapr.f90:3482), which THERMR reads as S = exp(0) = 1.
    """
    # A species' share of the coherent law can be negative where the
    # interference is destructive. ENDF cannot hold negative S, so those
    # cells become zero (ln S = -999), and the summed material law is then
    # slightly larger than the exact total.
    shift = (-be / 2.0, be / 2.0, 0.0, 0.0)[isym]
    if ilog:
        return sigfig(log(s) + shift, 7, 0) if s > 0.0 else -999.0
    try:
        v = s * exp(shift)
    except OverflowError:
        # isym=1 only: exp(be/2) overflows, the product usually does not.
        v = _asym_overflow_s(s, be, beta_card, temp_k, smin)
    v = sigfig(v, 7 if v >= 1.0e-9 else 6, 0)
    return 0.0 if v < smin else v


def _coherent_s_table(bragg, nedge, ntempr, tempr, edge_delta):
    """Shared MF7/MT2 coherent-elastic table assembly.

    All coherent-elastic builders produce the same ENDF structure — a thinned
    cumulative-S TAB1 at the first temperature plus per-temperature LIST
    blocks — and differ only in how a single Bragg edge's Debye-Waller-
    weighted contribution is evaluated. That evaluation is supplied as
    ``edge_delta(j, itemp, energy=None)``; everything else lives here.

    Edge thinning matches NJOY's endout: an edge survives while it still
    changes the running total by more than ``tol`` relatively; edges beyond
    ``jmax`` are folded into the last surviving point (Fortran loop
    behavior).
    """
    tol = 0.9e-7
    out = {}

    # Thin negligible edges at the first temperature, caching the deltas.
    deltas_T0 = []
    total_sum = 0.0
    suml = 0.0
    jmax = 0
    for j in range(nedge):
        d = edge_delta(j, 0)
        deltas_T0.append(d)
        total_sum += d
        if total_sum - suml > tol * total_sum:
            jmax = j + 1
            suml = total_sum

    if jmax == 0:
        raise ValueError(
            "coherent elastic: no Bragg edge with a positive contribution "
            f"(nedge={nedge}); check the Card 6d coherent lengths and the lattice")

    # First temperature: TAB1 with energies and cumulative S.
    out['NP'] = jmax
    out['S_T0_table'] = {
        'NBT': [jmax],
        'INT': [1],  # histogram interpolation
    }

    energies = []
    s_cumulative = []
    s = 0.0
    for j in range(nedge):
        s += deltas_T0[j]
        if j < jmax:
            energies.append(sigfig(bragg[j][0], 7, 0))
            s_cumulative.append(sigfig(s, 7, 0))
        else:
            energies[-1] = sigfig(bragg[j][0], 7, 0)
            s_cumulative[-1] = sigfig(s, 7, 0)

    out['S_T0_table']['Eint'] = energies
    out['S_T0_table']['S'] = s_cumulative

    # Additional temperatures: LIST blocks on the thinned grid.
    if ntempr > 1:
        out['T'] = {}
        out['LI'] = 2  # lin-lin interpolation
        out['S'] = {q: {} for q in range(1, jmax + 1)}

        for i in range(1, ntempr):
            out['T'][i] = tempr[i]
            s = 0.0
            jj = 0
            for j in range(nedge):
                if j < jmax:
                    jj = j
                # For j >= jmax, jj stays at jmax-1 (Fortran behavior)
                e_sf = sigfig(bragg[jj][0], 7, 0)
                s += edge_delta(jj, i, energy=e_sf)
                out['S'][jj + 1][i] = sigfig(s, 7, 0)

    return out


def _coherent_s_table_or_grouped(bragg, nedge, ntempr, tempr, edge_delta,
                                 crystal_info):
    """The kinematic coherent Bragg edges, GROUPED above the threshold when Card 6b
    enables it (ENDF-102 7.2.2), else the full set. Shared by the plain coherent
    path AND the extinction splice, so grouping composes with extinction (the
    extinction path feeds the grouped edges as its above-cutoff piece)."""
    group_bpd = int(crystal_info.get('coh_edge_group_bins_per_decade', 0))
    if group_bpd > 0:
        group_thr = float(crystal_info.get('coh_edge_group_threshold_ev', 1.0))
        return _grouped_coherent_s_table(bragg, nedge, ntempr, tempr, edge_delta,
                                         group_thr, group_bpd)
    return _coherent_s_table(bragg, nedge, ntempr, tempr, edge_delta)


def _coherent_extinction_s_table(kin_table, sigma_fn, edge_E, E_active,
                                 ntempr, tempr, tol):
    """Extinction-corrected MF7/MT2 coherent-elastic table (INT=1 histogram).

    Extinction makes sigma_coh depend on E within a Bragg interval, but only
    below ``E_active``. Below it the table has one node per Bragg edge plus
    adaptive nodes wherever S = E*sigma changes by more than ``tol``; above it
    the thinned kinematic edges of ``kin_table`` are reused. The table is a
    histogram because THERMR reads MF7/MT2 as a step function whatever the INT
    flag.
    """
    kst = kin_table['S_T0_table']
    Ek, Sk = kst['Eint'], kst['S']          # thinned kinematic edges (sigfig'd)

    # split the kinematic edges at the cutoff; everything >= E_active is kinematic
    cut = next((i for i, e in enumerate(Ek) if e >= E_active), len(Ek))
    top = Ek[cut] if cut < len(Ek) else (edge_E[-1] if edge_E else E_active)

    # raw-energy cumulative-S nodes below the cutoff. Each Bragg edge is a node with
    # the POST-jump cumulative (sigma_fn(E) includes the edge at E). Between edges the
    # smooth extinction rise is refined to tol using PRE-jump right endpoints, so the
    # refinement never "chases" a jump and every node keeps a well-defined side.
    #
    # The interior is refined at every temperature and the nodes are united, so
    # a colder (sharper) temperature is resolved as well as T0.
    edges_below = sorted(e for e in set(edge_E) if e < top)
    node_E = set(edges_below)                # Bragg edges are nodes at every temperature
    bounds = edges_below + [top]
    for it in range(ntempr):
        for k in range(len(bounds) - 1):
            a = bounds[k]
            b = bounds[k + 1] * (1.0 - 1e-9)  # just below the next edge (pre-jump side)
            if b <= a:
                continue
            stack = [(a, b)]
            while stack:
                lo, hi = stack.pop()
                if hi <= lo * (1.0 + 1e-6):
                    continue
                slo = lo * sigma_fn(lo, it)
                shi = hi * sigma_fn(hi, it)
                if abs(shi - slo) > tol * max(abs(shi), 1e-30):
                    m = sqrt(lo * hi)
                    node_E.add(m)
                    stack.append((lo, m))
                    stack.append((m, hi))
    node_S = {e: e * sigma_fn(e, 0) for e in node_E}    # T0 cumulative-S at every node

    # Assemble: the fine nodes below the cutoff, then the kinematic edges. Dedup
    # to ascending 7-figure energies; on a collision the post-jump value wins.
    # Assumes E_active is below the grouping threshold (true for the 1 eV
    # default), so the edges at the seam are ungrouped.
    energies, s0, src = [], [], []           # src: raw energy (below) | ('kin', qi)
    for e in sorted(node_S):
        es = sigfig(e, 7, 0)
        sv = sigfig(node_S[e], 7, 0)
        if energies and es == energies[-1]:
            s0[-1], src[-1] = sv, e          # collision: keep post-jump value
        elif not energies or es > energies[-1]:
            energies.append(es); s0.append(sv); src.append(e)
    for qi in range(cut, len(Ek)):
        if not energies or Ek[qi] > energies[-1]:
            energies.append(Ek[qi]); s0.append(Sk[qi]); src.append(('kin', qi))

    out = {'NP': len(energies)}
    out['S_T0_table'] = {
        'NBT': [len(energies)], 'INT': [1],   # histogram (standard MF7/MT2)
        'Eint': energies, 'S': s0,
    }
    if ntempr > 1:
        out['T'] = {}
        out['LI'] = 2
        out['S'] = {q: {} for q in range(1, len(energies) + 1)}
        kin_per_temp = kin_table.get('S', {})
        for i in range(1, ntempr):
            out['T'][i] = tempr[i]
            for q, sc in enumerate(src):
                if isinstance(sc, tuple):     # above cutoff: reuse kinematic per-temp
                    out['S'][q + 1][i] = kin_per_temp[sc[1] + 1][i]
                else:                          # below cutoff: raw-energy evaluation
                    out['S'][q + 1][i] = sigfig(sc * sigma_fn(sc, i), 7, 0)
    return out


def _build_coherent_elastic(mat, za, awr, bragg, nedge, ntempr,
                             tempr, dwpix, dwp1, nss, b7):
    """Build MF7/MT2 coherent elastic for the built-in materials (iel=1-6).

    The edge weight is the classic LEAPR isotropic Debye-Waller attenuation,
    averaged with the secondary scatterer's for an SCT (b7=0) mixed
    moderator.
    """
    # Isotropic DW integral per temperature. For an SCT (b7==0) mixed
    # moderator the principal and secondary scatterer integrals are averaged
    # (byte-identical to the former per-edge averaging).
    if nss > 0 and b7 == 0.0:
        w_iso = [(dwpix[i] + dwp1[i]) / 2.0 for i in range(ntempr)]
    else:
        w_iso = [dwpix[i] for i in range(ntempr)]
    _edge_delta = make_edge_delta(bragg, w_iso)

    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 1  # coherent elastic
    mf7mt2['T0'] = tempr[0]
    mf7mt2['LT'] = ntempr - 1
    mf7mt2.update(_coherent_s_table(bragg, nedge, ntempr, tempr, _edge_delta))
    return mf7mt2

def _build_generalized_elastic(mat, za, awr, bragg, nedge, ntempr, tempr,
                                crystal_info, dwpix_out, sb, npr=1):
    """Build MF7/MT2 for generalized elastic (iel=10).

    Implements both CEF (elastic_mode=1) and MEF (elastic_mode=2) following
    the paper: K. Ramic, J. I. Damian Marquez, et al., NIM-A 1027 (2022) 166227.

    For CEF:
      - Single atom (nat=1): Eq 24/25 — store the designated-coherent component, scale by
        (σ_coh + σ_inc) / σ_dominant
      - Polyatomic (nat>1): Eq 26 — DC atom gets coherent elastic scaled
        by 1/f_DC; other atoms get incoherent elastic with redistribution

    For MEF:
      - Eq 23: LTHR=3 with coherent elastic (per atom) + incoherent elastic
    """

    elastic_mode = crystal_info['elastic_mode']
    atom_types = crystal_info['atom_types']
    nat = crystal_info['nat']
    principal_idx = crystal_info['principal_atom_idx']
    dc_idx = crystal_info['dc_atom_idx']  # None if MEF or nat==1
    principal_at = atom_types[principal_idx]

    sigma_coh_p = principal_at['sigma_coh']   # barns
    sigma_inc_p = principal_at['sigma_inc']   # barns

    # ---- MEF (elastic_mode=2): LTHR=3 ----
    if elastic_mode == 2:
        return _build_mef_elastic(mat, za, awr, bragg, nedge, ntempr, tempr,
                                   crystal_info, dwpix_out, npr)

    # ---- CEF (elastic_mode=1) ----

    if nat == 1:
        # Single atom CEF: Eq 24 or 25
        if sigma_coh_p > sigma_inc_p:
            # Coherent approximation (Eq 24)
            # Scale Bragg edges by (σ_coh + σ_inc) / σ_coh
            scale = (sigma_coh_p + sigma_inc_p) / sigma_coh_p
            print(f"  SEF single atom: coherent approx, "
                  f"scale={(sigma_coh_p + sigma_inc_p):.4f}/{sigma_coh_p:.4f} = {scale:.4f}", flush=True)
            return _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr,
                                        tempr, dwpix_out, scale,
                                        crystal_info=crystal_info)
        else:
            # Incoherent approximation (Eq 25)
            # Scale SB by (σ_coh + σ_inc) / σ_inc
            if sigma_inc_p <= 0.0:
                raise ValueError(
                    "SEF single-atom: sigma_coh and sigma_inc are both 0 on "
                    "Card 6d; no elastic law to write")
            scale = (sigma_coh_p + sigma_inc_p) / sigma_inc_p
            print(f"  SEF single atom: incoherent approx, "
                  f"SB scale={scale:.4f}", flush=True)
            return _build_cef_incoherent(mat, za, awr, ntempr, tempr,
                                          dwpix_out, sigma_inc_p * scale, npr)

    # Polyatomic CEF (nat > 1): Eq 26
    if principal_idx == dc_idx:
        # Principal scatterer IS the DC atom → LTHR=1 (coherent elastic)
        # Scale by 1/f_DC to get per-DC-atom cross section
        f_dc = principal_at['fraction']
        scale = 1.0 / f_dc
        print(f"  SEF polyatomic: principal is DC atom, "
              f"scale=1/f_DC=1/{f_dc:.4f}={scale:.4f}", flush=True)
        return _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr,
                                    tempr, dwpix_out, scale,
                                    crystal_info=crystal_info)
    else:
        # Principal scatterer is NOT the DC atom → LTHR=2 (incoherent elastic)
        # with redistribution factor from Eq 26
        dc_at = atom_types[dc_idx]
        f_dc = dc_at['fraction']
        sigma_inc_dc = dc_at['sigma_inc']

        if sigma_inc_p <= 0.0:
            raise ValueError(
                "SEF polyatomic: the principal has sigma_inc=0 but is not the "
                "designated-coherent atom, so the Eq. 26 redistribution is "
                "undefined; use elastic_mode=2")
        # Redistribution factor: [1 + f_DC/(1-f_DC) × σ_inc_DC/σ_inc_i]
        redist = 1.0 + (f_dc / (1.0 - f_dc)) * (sigma_inc_dc / sigma_inc_p)

        # SB for incoherent elastic: σ_inc of this atom × redistribution factor
        sb_redist = sigma_inc_p * redist
        print(f"  SEF polyatomic: principal is NOT DC atom, "
              f"redist factor={redist:.4f}, SB={sb_redist:.4f} b "
              f"(x npr={npr} on tape)", flush=True)
        return _build_cef_incoherent(mat, za, awr, ntempr, tempr,
                                      dwpix_out, sb_redist, npr)

def _report_grouping_fidelity(E, delta, out_E, out_S, ntempr, tempr):
    """Print the relative error in the integral cross section introduced by
    grouping, per temperature, so the user can judge fidelity.

    sigma_coh_el(E,T) = S(E,T)/E, so the area under the curve is
    INT sigma dE = INT S/E dE = sum_segments S_seg * ln(E_{seg+1}/E_seg) for the
    histogram-S staircase. We compare the ungrouped and grouped staircases.
    """
    def _integ(Egrid, Scum):
        """Integral of sigma(E) = S_cum(E)/E over the grid (log-E measure)."""
        Egrid = np.asarray(Egrid, dtype=float)
        Scum = np.asarray(Scum, dtype=float)
        if Egrid.size < 2:
            return 0.0
        return float(np.sum(Scum[:-1] * np.log(Egrid[1:] / Egrid[:-1])))

    cum_ungrouped = np.cumsum(delta, axis=1)          # (ntempr, nedge)
    parts, worst = [], 0.0
    for it in range(ntempr):
        i_u = _integ(E, cum_ungrouped[it])
        i_g = _integ(out_E, [out_S[g][it] for g in range(len(out_S))])
        rel = abs(i_g - i_u) / i_u if i_u > 0 else 0.0
        worst = max(worst, rel)
        parts.append(f"{tempr[it]:.0f}K:{rel:.1e}")
    print(f"    grouped {len(E)} -> {len(out_E)} edges; "
          f"INT(sigma)dE rel.err [{', '.join(parts)}], max {worst:.2e}", flush=True)

def _grouped_coherent_s_table(bragg, nedge, ntempr, tempr, edge_delta_fn,
                              threshold_ev, bins_per_decade):
    """Coherent-elastic cumulative-S table with Bragg edges grouped above
    ``threshold_ev`` (ENDF-102 sec 7.2.2).

    Above the threshold the edges in each log-uniform bin (``bins_per_decade``
    per decade) become one step at the T0-weighted log-mean energy
    ``ln(E_rep) = sum_i d_i ln(E_i) / sum_i d_i``, which keeps the bin's 1/E
    integral exact at T0. The cumulative S is exact at every group boundary
    and every temperature; the shape inside a bin drifts at T != T0 (printed
    by ``_report_grouping_fidelity``; raise ``bins_per_decade`` if needed).
    Edges at or below the threshold are kept.
    """
    E = np.array([bragg[j][0] for j in range(nedge)], dtype=float)
    if nedge == 0:
        raise ValueError("coherent elastic: no Bragg edge to group")
    emax = float(E[-1])
    # Per-edge DW-weighted increments at each edge's OWN energy, all temperatures.
    delta = np.array(
        [[float(edge_delta_fn(j, it)) for j in range(nedge)] for it in range(ntempr)],
        dtype=float,
    )  # shape (ntempr, nedge)

    def _bin(e):
        """Logarithmic grouping-bin index of edge energy ``e``."""
        return int(np.floor(bins_per_decade * np.log10(e / threshold_ev)))

    # --- partition edges into groups ---
    groups = []
    j = 0
    while j < nedge:
        e = E[j]
        if e <= threshold_ev or e >= emax:
            groups.append(np.array([j]))            # keep individually
            j += 1
            continue
        m = _bin(e)
        k = j
        while (k < nedge and threshold_ev < E[k] < emax and _bin(E[k]) == m):
            k += 1
        idx = np.arange(j, k)
        # Do not silently clamp a non-positive bin increment caused by
        # negative interference terms / roundoff: ungroup it so every step
        # stays physical. A bin whose increments are exactly zero at EVERY
        # temperature (Debye-Waller underflow at high E) carries no cross
        # section at all -- merge it into one zero step instead of emitting
        # one redundant point per raw edge.
        if (len(idx) > 1 and delta[0, idx].sum() <= 0.0
                and np.any(delta[:, idx] != 0.0)):
            groups.extend(np.array([t]) for t in idx)
        else:
            groups.append(idx)
        j = k

    # --- emit one (E_rep, cumulative S) point per group ---
    out_E, out_S = [], []
    cum = np.zeros(ntempr)
    for idx in groups:
        cum = cum + delta[:, idx].sum(axis=1)       # exact cumulative for every T
        if idx.size == 1:
            e_rep = float(E[idx[0]])
        else:
            w = delta[0, idx]                        # T0 structure-factor weights
            sw = float(np.sum(w))
            if sw > 0.0:
                e_rep = float(np.exp(np.sum(w * np.log(E[idx])) / sw))
            else:                                    # all-zero (underflow) bin
                e_rep = float(np.exp(np.mean(np.log(E[idx]))))
        out_E.append(e_rep)
        out_S.append(cum.copy())
    np_pts = len(out_E)

    _report_grouping_fidelity(E, delta, out_E, out_S, ntempr, tempr)

    table = {
        'NP': np_pts,
        'S_T0_table': {
            'NBT': [np_pts],
            'INT': [1],                              # histogram interpolation
            'Eint': [sigfig(e, 7, 0) for e in out_E],
            'S': [sigfig(out_S[g][0], 7, 0) for g in range(np_pts)],
        },
    }
    if ntempr > 1:
        table['T'] = {i: tempr[i] for i in range(1, ntempr)}
        table['LI'] = 2
        table['S'] = {
            q: {i: sigfig(out_S[q - 1][i], 7, 0) for i in range(1, ntempr)}
            for q in range(1, np_pts + 1)
        }
    return table

def _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr, tempr,
                         dwpix_out, scale, crystal_info):
    """Build LTHR=1 (coherent elastic) section with a multiplicative scale.

    The scale factor accounts for:
    - Single atom CEF: (σ_coh + σ_inc)/σ_coh  (Eq 24)
    - Polyatomic CEF DC atom: 1/f_DC  (Eq 26)

    If crystal_info with species_corr is provided, per-species Debye-Waller
    factors are applied inside the structure factor (matching NCrystal):
        δ_j = scale × Σ_{s,t} b_s b_t exp(-2(W_s+W_t) E_j) D_{st,j}
    where W_s = dwpix_s / (awr_s T kB) is the per-species ENDF DW parameter.

    When crystal_info additionally contains 'F_species_per_temp' and
    'bragg_dir_terms' (set when inelastic_mode=1/2), the directional Debye-Waller is
    used instead and evaluated plane-by-plane inside each merged Bragg edge:
        W_s(Ĝ) = (Ĝ · F_s · Ĝ) / (awr_s × kT)
    This preserves anisotropic attenuation when an edge contains multiple
    crystallographically distinct reciprocal-vector directions.
    """

    # --- Resolve per-species/directional DW (inelastic_mode=1/2) or fall back
    #     to the scalar isotropic form; arithmetic shared with the MEF builder
    #     via irma.core.elastic_dw. ---
    species_dw = resolve_species_dw(crystal_info, tempr, ntempr)
    _edge_delta = make_edge_delta(bragg, dwpix_out, scale=scale,
                                  species_dw=species_dw, tempr=tempr)

    # --- Build ENDF dict ---
    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 1  # coherent elastic
    mf7mt2['T0'] = tempr[0]
    mf7mt2['LT'] = ntempr - 1

    # Optional crystalline extinction (see _coherent_extinction_s_table).
    ext_cfg = crystal_info.get('coherent_extinction')
    if ext_cfg:
        from irma.core.elastic_extinction import make_sigma_coh_ext
        crystal = crystal_info['crystal']
        sigma_fn, edge_E, E_active = make_sigma_coh_ext(
            bragg, crystal_info['bragg_dir_terms'], species_dw,
            crystal.volume, crystal.n_atoms, scale, ext_cfg, tempr)
        kin_table = _coherent_s_table_or_grouped(
            bragg, nedge, ntempr, tempr, _edge_delta, crystal_info)
        mf7mt2.update(_coherent_extinction_s_table(
            kin_table, sigma_fn, edge_E, E_active, ntempr, tempr,
            float(ext_cfg.get('rmse_tol', 1e-3))))
        return mf7mt2

    mf7mt2.update(_coherent_s_table_or_grouped(
        bragg, nedge, ntempr, tempr, _edge_delta, crystal_info))
    return mf7mt2

def _build_cef_incoherent(mat, za, awr, ntempr, tempr, dwpix_out, sb_value,
                          npr=1):
    """Build LTHR=2 (incoherent elastic) section for CEF non-DC atom.

    sb_value is the per-principal bound cross section (barns), including any
    Eq 26 redistribution. The tape stores SB = sb_value*npr, the molecular
    convention of every IRMA writer (npr is MT4's B(6)).
    """
    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 2  # incoherent elastic

    ndw = max(ntempr, 2)
    # NJOY writes SB unrounded (up to 9 digits); IRMA rounds to 7 significant
    # figures, so compare this record numerically.
    mf7mt2['SB'] = sb_value * npr
    mf7mt2['NBT'] = [ndw]
    mf7mt2['INT'] = [2]  # linear-linear interpolation
    mf7mt2['Tint'] = []
    mf7mt2['Wp'] = []

    for i in range(ndw):
        idx_t = min(i, ntempr - 1)
        mf7mt2['Tint'].append(tempr[idx_t])
        mf7mt2['Wp'].append(sigfig(dwpix_out[idx_t], 7, 0))

    return mf7mt2

def _build_mef_elastic(mat, za, awr, bragg, nedge, ntempr, tempr,
                        crystal_info, dwpix_out, npr=1):
    """Build MF7/MT2 with LTHR=3 (mixed elastic format).

    Stores both coherent elastic (Bragg edges, per atom) and incoherent
    elastic in a single section, following the proposed ENDF extension
    (Eq 23 from the paper).

    The incoherent SB follows the molecular convention ``SB = sigma_inc_p *
    npr`` (npr is recorded in MF7/MT4 B(6)), the same convention the classic
    (iel<0) and SEF/CEF incoherent writers use, so a consumer's single
    division by npr is correct regardless of which writer produced the tape.

    σ^el_i = σ^coh/N + σ^inc_i

    When crystal_info carries 'F_species_per_temp' and 'bragg_dir_terms'
    (inelastic_mode=1/2), the coherent part uses the same plane-by-plane
    directional Debye-Waller attenuation as the CEF builder:
        W_s(Ĝ) = (Ĝ · F_s · Ĝ) / (awr_s × kT)
    The incoherent term's W' stays isotropic: the LTHR=3 tabulation supports
    only a scalar W', so this uses the powder-averaged (isotropic) trace Tr(F)/3
    of the per-species displacement tensor F (the same F as in W_s(Ĝ) above).
    For an anisotropic material that is an APPROXIMATION to the true
    direction-averaged incoherent DW, which the scalar ENDF MF7/MT2 format cannot
    represent exactly; the full anisotropy is carried instead in the coherent
    (per-plane W_s(Ĝ) above) and inelastic channels.
    """
    sigma_inc_p = crystal_info['atom_types'][crystal_info['principal_atom_idx']]['sigma_inc']
    mf7mt2 = _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr, tempr,
                                 dwpix_out, 1.0, crystal_info=crystal_info)
    inc = _build_cef_incoherent(mat, za, awr, ntempr, tempr, dwpix_out,
                                sigma_inc_p, npr)
    mf7mt2['LTHR'] = 3  # mixed elastic
    for key in ('SB', 'NBT', 'INT', 'Tint', 'Wp'):
        mf7mt2[key] = inc[key]
    print(f"  MEF: LTHR=3, {mf7mt2['NP']} Bragg edges, σ_inc={sigma_inc_p:.4f} b "
          f"(x npr={npr} on tape)", flush=True)
    return mf7mt2
