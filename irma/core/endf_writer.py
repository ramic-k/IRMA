"""ENDF-6 output for IRMA (MF1/MT451, MF7/MT2, MF7/MT4).

write_endf_output assembles the tape via endf-parserpy. The MF7/MT2 elastic
section comes from one of the builders: the classic LEAPR coherent table
(_build_coherent_elastic, iel=1-6), the generalized CEF/MEF builders
(iel=10), or the inline incoherent-elastic branch (iel<0). MF7/MT4 stores
the symmetric law S*exp(-beta/2) via _compute_endf_s (isym/ilog variants).

Terminology: "CEF" throughout this module (and the ``_build_cef_*`` helper
names, kept to avoid an API/identifier break) is the single-channel elastic
format now called SEF in the docs and GUI. It was named the current ENDF
format (CEF) in Ramic et al., NIM-A 1027 (2022) 166227, when the mixed
elastic format (MEF) was introduced; only the user-facing label changed.
Printed summaries and error messages use SEF.
"""

import sys
import warnings

import numpy as np
from math import exp, log, sqrt

from irma.core.constants import BK, THERM
from irma.core.deck import DeckError
from irma.core.kernels import sigfig
from irma.core.elastic_dw import resolve_species_dw, make_edge_delta


_CPP_FALLBACK_WARNED = False   # the auto-fallback warns once per process

# math.exp raises OverflowError ("math range error") for any argument
# strictly above ln(DBL_MAX) = 709.782712893384. The isym=1 asymmetric-law
# overflow handling below and the Card 10 parse-time early warning in
# driver.py both derive their thresholds from this ceiling.
_LN_FLOAT_MAX = log(sys.float_info.max)

# ln of the largest real value that rounds to float64 0.0 (2**-1075): the
# ceiling on what a stored exact-zero S(alpha,beta) could have been before
# it underflowed.
_LN_MAX_STORED_ZERO = -1075.0 * log(2.0)


def _select_writer_backend():
    """Instantiate the endf-parserpy backend that serializes the tape.

    The default is the compiled ``EndfParserCpp`` backend: it is
    byte-identical to the pure-Python writer on the full expected evidence
    set (native NJOY expected decks, the NJOY minitape byte pins, the
    noncubic fast-CI/lat0 tapes and the Bragg-grouping path) and several
    times faster, which dominates classic-path wall time.

    The ``IRMA_ENDF_WRITER`` environment variable overrides the choice:

    * unset/empty/``auto`` — use ``EndfParserCpp``; if the compiled module
      cannot be loaded (source-only endf-parserpy builds), fall back to
      the pure-Python ``EndfParserPy`` with a loud warning.
    * ``py``  — force the legacy pure-Python ``EndfParserPy`` writer.
    * ``cpp`` — require the compiled backend; a missing compiled module is
      a hard error instead of a fallback.

    Any other value raises ``ValueError`` (no silent misconfiguration).
    """
    import os
    from endf_parserpy import EndfParserPy

    choice = os.environ.get("IRMA_ENDF_WRITER", "auto").strip().lower()
    if choice == "":
        choice = "auto"
    if choice not in ("auto", "cpp", "py"):
        raise ValueError(
            f"IRMA_ENDF_WRITER={choice!r} is not a valid ENDF writer "
            f"backend: use 'cpp' (compiled, default), 'py' (pure Python), "
            f"or unset/'auto' (compiled with pure-Python fallback).")
    if choice == "py":
        return EndfParserPy()
    try:
        from endf_parserpy import EndfParserCpp
        return EndfParserCpp()
    except ImportError:
        if choice == "cpp":
            raise
        global _CPP_FALLBACK_WARNED
        if not _CPP_FALLBACK_WARNED:
            _CPP_FALLBACK_WARNED = True
            print("WARNING: compiled ENDF writer backend (EndfParserCpp) is "
                  "unavailable in this endf-parserpy install; falling back to "
                  "the pure-Python writer (identical output, slower). Set "
                  "IRMA_ENDF_WRITER=py to silence this warning.", flush=True)
        return EndfParserPy()


def _patch_mf1_directory_counts(lines):
    """Set every MF1/MT451 directory NCx to the tape's ACTUAL record count.

    The closed-form NJOY NCx estimates carried over from leapr.f90 cannot
    track this writer's exact line wrapping — in particular for the iel=10
    incoherent/mixed-elastic and grouped coherent-elastic branches — so the
    records actually emitted are counted instead — exact by construction for
    every section and every future format change.
    Patching values in place never changes a section's line count, so a
    single pass over the written tape suffices.
    """
    # Count data records per (MF, MT); SEND/FEND/MEND/TEND have MT=0 or
    # MF=0 and are excluded automatically.
    counts = {}
    for line in lines:
        if len(line) < 75:
            continue
        try:
            mf = int(line[70:72])
            mt = int(line[72:75])
        except ValueError:
            continue
        if mf == 0 or mt == 0:
            continue
        counts[(mf, mt)] = counts.get((mf, mt), 0) + 1

    # Locate the directory region structurally instead of by column shape
    # alone. MF1/MT451 is: 4 header CONT records, then NWD text records, then
    # the NXC directory entries. NWD lives in the 4th header record's N1 field
    # ([44:55]). Confining the patch to records past 4 + NWD makes it
    # impossible for a free-text DESCRIPTION/HSUB line to be mistaken for a
    # directory entry just because its columns happen to parse as integers.
    sec_idx = -1          # record index within the current MT451 section
    dir_start = None      # first record index that is a directory entry
    patched = []
    for line in lines:
        is_mt451 = (len(line) >= 75 and line[70:72] == ' 1'
                    and line[72:75] == '451')
        if is_mt451:
            sec_idx += 1
            if sec_idx == 3:  # 4th header record (TEMP/.../NWD/NXC)
                try:
                    nwd = int(line[44:55])
                    dir_start = 4 + nwd
                except ValueError:
                    dir_start = None
        else:
            # Any non-MT451 line ends the section (SEND/FEND, MF7 records, …).
            sec_idx = -1
            dir_start = None

        is_dir_entry = False
        if (is_mt451 and dir_start is not None and sec_idx >= dir_start
                and line[:22].strip() == ''):
            try:
                mfx = int(line[22:33])
                mtx = int(line[33:44])
                int(line[44:55])
                int(line[55:66])
                is_dir_entry = (mfx, mtx) in counts
            except ValueError:
                pass
        if is_dir_entry:
            line = (line[:44] + str(counts[(mfx, mtx)]).rjust(11)
                    + line[55:])
        patched.append(line)
    return patched


def _warn_mf1_comment_loss(card_no, lost_text):
    """Warn that an MF1/MT451 header comment card carries text in columns that
    map to no ENDF field, so it is silently dropped from the tape. Comment cards
    1-5 fill the ENDF-102 structured header; free text belongs in card 6 onward.
    """
    warnings.warn(
        f"MF1/MT451 comment card {card_no} has text in reserved/pad columns "
        f"that will not appear on the tape ({lost_text.strip()!r}). Comment "
        "cards 1-5 fill the structured header (ZSYMAM/ALAB/EDATE/AUTH; "
        "REF/DDATE/RDATE/ENDATE; HSUB1-3); put free-form text in card 6 onward.",
        stacklevel=2)


def write_endf_output(filename, mat, za, awr, spr, npr, iel, ncold, nss,
                      b7, aws, sps, mss, nalpha, nbeta, lat,
                      alpha, beta, ssm, ssp, tempr, ntempr,
                      dwpix, dwp1, tempf, tempf1,
                      bragg, nedge, isym, ilog, smin, iprint,
                      iint=0, comments=None, crystal_info=None):
    """Write ENDF-6 output file using endf-parserpy."""
    small = 1.0e-9

    # iint selects the MF7/MT4 S(alpha,beta) interpolation law for BOTH the
    # alpha (per-beta TAB1) and beta (TAB2) tables: 0 -> log-lin (ENDF INT=4,
    # the classic/NJOY-faithful default); 1 -> lin-lin (INT=2). Coherent
    # one-phonon laws have structural near-zeros that log interpolation floors;
    # lin-lin preserves them. INT-aware THERMR honors whichever is written.
    coh_int = 2 if iint == 1 else 4

    # Compute bound scattering cross section (sb). The secondary-scatterer
    # bound XS (sbs) and the mixed-moderator SAB merge live in engine.py
    # (it recomputes sb/sbs/srat before calling this writer), so no sbs is
    # needed here.
    sb = spr * ((1.0 + awr) / awr)**2

    # Compute Debye-Waller integral for ENDF output
    dwpix_out = dwpix.copy()
    dwp1_out = dwp1.copy()
    for i in range(ntempr):
        if nss == 0 or b7 > 0.0:
            dwpix_out[i] = dwpix[i] / (awr * tempr[i] * BK)
        else:
            dwpix_out[i] = dwpix[i] / (aws * tempr[i] * BK)
            dwp1_out[i] = dwp1[i] / (awr * tempr[i] * BK)

    # Determine symmetry type
    # isym: 0 = symmetric S, 1 = S for +/- beta (coldh),
    #        2 = ss for -beta, 3 = ss for +/- beta

    parser = _select_writer_backend()

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

    # Text fields (must be padded to exact ENDF field widths)
    mf1['ZSYMAM'] = ' ' * 11
    mf1['ALAB'] = ' ' * 11
    mf1['EDATE'] = ' ' * 10
    mf1['AUTH'] = ' ' * 33
    mf1['REF'] = ' ' * 21
    mf1['DDATE'] = ' ' * 10
    mf1['RDATE'] = ' ' * 10
    mf1['ENDATE'] = ' ' * 8
    mf1['HSUB'] = {1: ' ' * 66, 2: ' ' * 66, 3: ' ' * 66}
    mf1['NWD'] = 0
    mf1['DESCRIPTION'] = {}

    # Populate text fields from comment cards
    # ENDF MF1/MT451 text records: NWD total records, structured as:
    #   Record 1: ZSYMAM(11) + ALAB(11) + EDATE(10) + AUTH(33) = 65 chars
    #   Record 2: REF(21) + DDATE(10) + RDATE(10) + ENDATE(8) = 49 chars
    #   Records 3-5: HSUB[1-3], 66 chars each
    #   Records 6+: DESCRIPTION[1..NWD-5], 66 chars each
    if comments is None:
        comments = []
    clean_comments = []
    for c in comments:
        # rstrip only: the deck tokenizer already removed the quotes, and the
        # inner leading blank is column 1 of the 11-char ZSYMAM field (NJOY
        # convention). A full strip() would shift every header field one
        # column left.
        c = c.rstrip()
        if (c.startswith("'") and c.endswith("'")) or \
           (c.startswith('"') and c.endswith('"')):
            c = c[1:-1]
        clean_comments.append(c)

    # Card 1 → ZSYMAM + ALAB + EDATE + AUTH
    if len(clean_comments) > 0:
        line0 = clean_comments[0].ljust(66)[:66]
        mf1['ZSYMAM'] = line0[:11]
        mf1['ALAB'] = line0[11:22]
        mf1['EDATE'] = line0[22:32]
        mf1['AUTH'] = line0[33:66]
        # (Card 1's only unmapped column is the single [32] EDATE/AUTH boundary;
        # a 1-char clip on continuous text isn't worth a warning -- see card 2.)
    # Card 2 → REF + DDATE + RDATE + ENDATE
    # endf_parserpy layout: {1}blank + REF{21} + DDATE{10} + {1}blank + RDATE{10} + {12}pad + ENDATE{8} + {3}pad
    # Positions: [0]=blank, [1:22]=REF, [22:32]=DDATE, [32]=blank, [33:43]=RDATE, [43:55]=pad, [55:63]=ENDATE
    if len(clean_comments) > 1:
        line1 = clean_comments[1].ljust(66)[:66]
        mf1['REF'] = line1[1:22]
        mf1['DDATE'] = line1[22:32]
        mf1['RDATE'] = line1[33:43]
        mf1['ENDATE'] = line1[55:63]
        # Warn only on the genuine multi-char pads -- the [43:55] block (12 cols)
        # and the [63:66] tail -- where a real chunk of comment text silently
        # vanishes. The 1-char field-boundary gaps ([0], [32]) are not worth it.
        lost1 = line1[43:55] + line1[63:66]
        if lost1.strip():
            _warn_mf1_comment_loss(2, lost1)
    # Cards 3-5 → HSUB[1-3]
    for i in range(3):
        idx = i + 2
        if idx < len(clean_comments):
            mf1['HSUB'][i + 1] = clean_comments[idx].ljust(66)[:66]
    # Cards 6+ → DESCRIPTION[1..]
    desc_start = 5
    desc_lines = clean_comments[desc_start:] if len(clean_comments) > desc_start else []
    # NWD = total text records actually written: the 5 structured header
    # records (ZSYMAM/ALAB/EDATE/AUTH, REF/DDATE/RDATE/ENDATE, HSUB 1-3)
    # are ALWAYS emitted — blank-padded when the deck has fewer than 5
    # comment cards — plus one record per description line.
    mf1['NWD'] = 5 + len(desc_lines)
    mf1['DESCRIPTION'] = {i + 1: desc_lines[i].ljust(66)[:66]
                          for i in range(len(desc_lines))}

    # Directory - entries for MF1/MT451 + MF7/MT2 (if present) + MF7/MT4.
    # The NCx values below are NJOY-formula ESTIMATES (leapr.f90 3126-3150)
    # used only to size the records; _patch_mf1_directory_counts replaces
    # every NCx with the actual emitted record count after the tape is
    # written.
    nxc_sections = 1  # MF7/MT4 always present
    if iel != 0:
        nxc_sections += 1  # MF7/MT2
    nxc = nxc_sections + 1  # +1 for MF1/MT451 itself
    mf1['NXC'] = nxc
    mf1['MFx'] = {}
    mf1['MTx'] = {}
    mf1['NCx'] = {}
    mf1['MOD'] = {}

    # Compute NCx for MF1/MT451: 4 header CONTs + NWD text + NXC directory
    nc_mt451 = 4 + mf1['NWD'] + nxc

    # Compute NCx for MF7/MT4 (NJOY formula: leapr.f90 lines 3147-3149)
    per_beta = 2 + (2 * nalpha + 4) // 6
    if ntempr > 1:
        per_beta += (ntempr - 1) * (1 + (nalpha + 5) // 6)
    nc_mt4 = 5 + nbeta * per_beta

    # Compute NCx for MF7/MT2 if present (leapr.f90 lines 3135-3138)
    nc_mt2 = 0
    if iel < 0:
        nc_mt2 = 3 + (2 * ntempr + 4) // 6
    elif iel > 0:
        nc_mt2 = 3 + (2 * nedge + 4) // 6
        if ntempr > 1:
            nc_mt2 += (ntempr - 1) * (1 + (nedge + 5) // 6)

    idx = 1
    mf1['MFx'][idx] = 1
    mf1['MTx'][idx] = 451
    mf1['NCx'][idx] = nc_mt451
    mf1['MOD'][idx] = 0
    idx += 1
    if iel != 0:
        mf1['MFx'][idx] = 7
        mf1['MTx'][idx] = 2
        mf1['NCx'][idx] = nc_mt2
        mf1['MOD'][idx] = 0
        idx += 1
    mf1['MFx'][idx] = 7
    mf1['MTx'][idx] = 4
    mf1['NCx'][idx] = nc_mt4
    mf1['MOD'][idx] = 0

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

    # --- low-temperature underflow detector (symmetric storage, ilog=0) ------
    # With ilog=0 the symmetric law is stored linearly as S*exp(-beta/2). At low
    # temperature beta grows like 1/T, so exp(-beta/2) at high energy transfer
    # becomes ~1e-100, and those points are written as 0 even though the PHYSICAL
    # scattering there is significant -- silently losing the high-E (e.g. optic)
    # phonon structure on read-back. We count such losses and warn; ilog=1 (LLN
    # log storage) preserves them. Harmless at room T, corrupts cryogenic tapes.
    # Per-temperature counters: beta ~ 1/T, so a LATER cryogenic temperature on
    # a multi-T tape can underflow even when T0 is warm -- checking only T0
    # (the original behavior) silently zeroed the cold table's high-E structure
    # with no warning (pre-release review P5).
    _lln_loss_n = [0] * ntempr
    _lln_loss_emax = [0.0] * ntempr
    _lln_check = (isym == 0 and ilog == 0 and ssm is not None
                  and getattr(ssm, "size", 0) > 0)
    _lln_sig = float(np.abs(ssm).max()) * 1.0e-4 if _lln_check else 0.0
    _ENDF_S_FLOOR = 1.0e-90

    for ii in range(1, nbt + 1):
        # Determine beta value
        if isym % 2 == 0:
            i_beta = ii - 1  # 0-based index into beta array
            beta_val = beta[i_beta]
        else:
            if ii < nbeta:
                i_beta = nbeta - ii  # reversed
                beta_val = -beta[i_beta]
            else:
                i_beta = ii - nbeta  # 0-based
                beta_val = beta[i_beta]

        mf7mt4['beta'][ii] = beta_val
        mf7mt4['LT'][ii] = ntempr - 1

        # S values for T0 (first temperature)
        nt = 0
        be = beta_val * sc_vals[nt]
        s_values = np.zeros(nalpha)

        for j in range(nalpha):
            s_val = _compute_endf_s(ssm, ssp, isym, ilog, ii, nbeta, j, nt,
                                     be, smin, small,
                                     beta_card=beta_val, temp_k=tempr[nt])
            s_values[j] = s_val
            # flag points whose physical scattering is significant but whose
            # linear symmetric storage underflows the ENDF representation
            if _lln_sig > 0.0:
                phys = ssm[ii - 1, j, nt]
                if phys > _lln_sig and 0.0 <= s_val < _ENDF_S_FLOOR:
                    _lln_loss_n[nt] += 1
                    _lln_loss_emax[nt] = max(_lln_loss_emax[nt],
                                             abs(be) * BK * tempr[nt] * 1000.0)

        mf7mt4['S_table'][ii] = {
            'NBT': [nalpha],
            'INT': [coh_int],
            'alpha': alpha.tolist(),
            'S': s_values.tolist()
        }

        # Additional temperatures: S[q][i][j] = S[alpha][beta][temp]
        if ntempr > 1:
            for j_temp in range(1, ntempr):
                be = beta_val * sc_vals[j_temp]
                for q in range(nalpha):
                    s_val = _compute_endf_s(ssm, ssp, isym, ilog, ii, nbeta,
                                            q, j_temp, be, smin, small,
                                            beta_card=beta_val,
                                            temp_k=tempr[j_temp])
                    if _lln_sig > 0.0:
                        phys = ssm[ii - 1, q, j_temp]
                        if phys > _lln_sig and 0.0 <= s_val < _ENDF_S_FLOOR:
                            _lln_loss_n[j_temp] += 1
                            _lln_loss_emax[j_temp] = max(
                                _lln_loss_emax[j_temp],
                                abs(be) * BK * tempr[j_temp] * 1000.0)
                    if ii not in mf7mt4['S'][q + 1]:
                        mf7mt4['S'][q + 1][ii] = {}
                    mf7mt4['S'][q + 1][ii][j_temp] = s_val

    for _nt in range(ntempr):
        if _lln_loss_n[_nt] > 0:
            print(
                f"WARNING: ilog=0 (linear symmetric-law storage) at "
                f"T={tempr[_nt]:g} K: {_lln_loss_n[_nt]} S(alpha,beta) points with "
                f"significant scattering (up to E~{_lln_loss_emax[_nt]:.0f} meV transfer) "
                f"underflow the ENDF symmetric law [S*exp(-beta/2) < 1e-90] and are "
                f"written as 0 -- the high-energy phonon structure (e.g. optic modes) "
                f"will be LOST on read-back. Set ilog=1 (LLN log storage) on Card 4 "
                f"to preserve it. (Harmless at room T; corrupts cryogenic tapes.)",
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
        # Generalized elastic output (CEF or MEF). crystal_info is mandatory
        # here: without it the elif chain below would fall through to the
        # built-in coherent-elastic builder (10 >= 1) and silently emit a
        # wrong MT2 from the hardcoded-material arrays. The engine always
        # supplies crystal_info for iel==10; guard the module-level API so a
        # direct caller fails fast instead of producing a corrupt section.
        if crystal_info is None:
            raise ValueError(
                "iel=10 (generalized elastic) requires crystal_info; "
                "got None. The built-in coherent-elastic builder cannot "
                "represent a generalized (iel=10) material.")
        mf7mt2 = _build_generalized_elastic(
            mat, za, awr, bragg, nedge, ntempr, tempr,
            crystal_info, dwpix_out, sb, npr=npr)

    elif iel < 0:
        # Incoherent elastic (LTHR=2)
        mf7mt2 = {}
        mf7mt2['MAT'] = mat
        mf7mt2['MF'] = 7
        mf7mt2['MT'] = 2
        mf7mt2['ZA'] = za
        mf7mt2['AWR'] = awr
        mf7mt2['LTHR'] = 2  # incoherent elastic

        ndw = max(ntempr, 2)
        mf7mt2['SB'] = sb * npr
        mf7mt2['NBT'] = [ndw]
        mf7mt2['INT'] = [2]  # linear-linear interpolation
        mf7mt2['Tint'] = []
        mf7mt2['Wp'] = []

        for i in range(ndw):
            idx_t = min(i, ntempr - 1)
            mf7mt2['Tint'].append(tempr[idx_t])
            mf7mt2['Wp'].append(sigfig(dwpix_out[idx_t], 7, 0))

    elif iel >= 1:
        # Coherent elastic (standard hardcoded materials)
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

    # Post-process: fix SEND/FEND/MEND/TEND record formatting
    # endf_parserpy writes "0.000000+0 0.000000+0 ..." but NJOY writes blank fields
    with open(filename, 'r') as f:
        lines = f.readlines()
    fixed = []
    for line in lines:
        if len(line) >= 75:
            # Check for FEND/MEND/TEND (MT=0) and SEND records. SEND carries
            # MT=0 with line# 99999 in cols 76-80, so it is already covered
            # by the mt_field==0 test below; no separate 99999 branch needed.
            try:
                mt_field = int(line[72:75])
            except (ValueError, IndexError):
                mt_field = None
            is_special = False
            if mt_field is not None and mt_field == 0:
                is_special = True  # SEND/FEND/MEND/TEND
            if is_special:
                # Blank out the data fields (first 66 chars), keep MAT/MF/MT/line#
                line = ' ' * 66 + line[66:]
        fixed.append(line)
    fixed = _patch_mf1_directory_counts(fixed)
    # newline='\n' keeps tapes byte-identical across platforms (no CRLF on
    # Windows); the golden comparisons and NJOY parity depend on LF-only.
    with open(filename, 'w', newline='\n') as f:
        f.writelines(fixed)
        # Ensure file ends with a newline (Fortran NJOY always does). Check the
        # list actually written (`fixed`), not the pre-patch `lines`.
        if fixed and not fixed[-1].endswith('\n'):
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


def _compute_endf_s(ssm, ssp, isym, ilog, ii, nbeta, j, nt, be, smin, small,
                    beta_card=0.0, temp_k=0.0):
    """Compute the ENDF S value for a given (beta, alpha, temperature) point.

    ``beta_card``/``temp_k`` are diagnostic context (the Card 9 beta value in
    card units and the temperature in K) for the isym=1 overflow DeckError;
    they do not affect any computed value.
    """
    # DELIBERATE NJOY DIVERGENCE: the ln-S sentinel for zero-S points is
    # -999 at EVERY temperature. NJOY's endout uses -999 in the
    # first-temperature TAB1 and the isym=1/2/3 LIST branches, but its
    # isym=0 additional-temperature LIST branch writes 0 instead
    # (leapr.f90:3482) — and THERMR stores ilog values verbatim as ln(S)
    # (thermr.f90:1754), so NJOY's 0 sentinel resurrects those zero-S
    # points as S = exp(0) = 1 downstream. A demonstrable NJOY bug; the
    # uniform -999 keeps S ~ 0 (below THERMR's sabflg = -225 floor).
    tiny = -999.0

    if isym == 0:
        # Symmetric S(a,b): store S*exp(-b/2)
        i_beta = ii - 1  # 0-based
        if ilog == 0:
            s_val = ssm[i_beta, j, nt] * exp(-be / 2.0)
            if s_val >= small:
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = sigfig(s_val, 6, 0)
        else:
            if ssm[i_beta, j, nt] > 0.0:
                s_val = log(ssm[i_beta, j, nt]) - be / 2.0
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = tiny

    elif isym == 1:
        # Asymmetric S for +/- beta (cold hydrogen)
        if ii < nbeta:
            i_beta = nbeta - ii  # 0-based
            if ilog == 0:
                try:
                    s_val = ssm[i_beta, j, nt] * exp(be / 2.0)
                except OverflowError:
                    # exp(be/2) alone exceeds float64 (be/2 > ~709.78; lat=1
                    # at cryogenic T), but the PRODUCT usually does not:
                    # evaluate it in log space (see _asym_overflow_s). Every
                    # non-overflowing value takes the direct path above,
                    # keeping existing tapes byte-identical.
                    s_val = _asym_overflow_s(ssm[i_beta, j, nt], be,
                                             beta_card, temp_k, smin)
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssm[i_beta, j, nt] > 0.0:
                    s_val = log(ssm[i_beta, j, nt]) + be / 2.0
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny
        else:
            i_beta = ii - nbeta  # 0-based
            if ilog == 0:
                try:
                    s_val = ssp[i_beta, j, nt] * exp(be / 2.0)
                except OverflowError:
                    # Same evaluation-order rescue as the -beta half above.
                    s_val = _asym_overflow_s(ssp[i_beta, j, nt], be,
                                             beta_card, temp_k, smin)
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssp[i_beta, j, nt] > 0.0:
                    s_val = log(ssp[i_beta, j, nt]) + be / 2.0
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny

    elif isym == 2:
        # Asymmetric SS for -beta (isabt=1)
        i_beta = ii - 1
        if ilog == 0:
            s_val = ssm[i_beta, j, nt]
            if s_val >= small:
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = sigfig(s_val, 6, 0)
        else:
            if ssm[i_beta, j, nt] > 0.0:
                s_val = log(ssm[i_beta, j, nt])
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = tiny

    elif isym == 3:
        # Asymmetric SS for +/- beta
        if ii < nbeta:
            i_beta = nbeta - ii
            if ilog == 0:
                s_val = ssm[i_beta, j, nt]
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssm[i_beta, j, nt] > 0.0:
                    s_val = log(ssm[i_beta, j, nt])
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny
        else:
            i_beta = ii - nbeta
            if ilog == 0:
                s_val = ssp[i_beta, j, nt]
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssp[i_beta, j, nt] > 0.0:
                    s_val = log(ssp[i_beta, j, nt])
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny
    else:
        s_val = 0.0

    if ilog == 0 and s_val < smin:
        s_val = 0.0

    return s_val

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

    # No edge carries a positive contribution (nedge==0, or every edge thinned
    # away because total_sum stayed <= 0). There is no Bragg peak to tabulate;
    # building the TAB1 below would index an empty energy list (energies[-1]).
    # A valid coherent-elastic section needs at least one surviving edge, so
    # fail fast rather than emit a degenerate/empty MT2.
    if jmax == 0:
        raise ValueError(
            "coherent elastic: no Bragg edge with a positive contribution "
            f"(nedge={nedge}) — there is no coherent-elastic peak to "
            "tabulate. Check Card 6d coherent lengths / the lattice, or "
            "omit the elastic section.")

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
    """The kinematic coherent comb, GROUPED above the threshold when Card 6b
    enables it (ENDF-102 7.2.2), else the full comb. Shared by the plain coherent
    path AND the extinction splice, so grouping composes with extinction (the
    extinction path feeds the grouped comb as its above-cutoff piece)."""
    group_bpd = int((crystal_info or {}).get('coh_edge_group_bins_per_decade', 0) or 0)
    if group_bpd > 0:
        group_thr = float((crystal_info or {}).get('coh_edge_group_threshold_ev', 1.0) or 1.0)
        return _grouped_coherent_s_table(bragg, nedge, ntempr, tempr, edge_delta,
                                         group_thr, group_bpd)
    return _coherent_s_table(bragg, nedge, ntempr, tempr, edge_delta)


def _coherent_extinction_s_table(kin_table, sigma_fn, edge_E, E_active,
                                 ntempr, tempr, tol):
    """Extinction-corrected MF7/MT2 coherent-elastic table (INT=1 histogram).

    An extinction-corrected sigma_coh(E) is energy-dependent WITHIN a Bragg
    interval (the per-plane factor y depends on the incident wavelength), but only
    BELOW the cutoff ``E_active`` -- above it y->1 and sigma_ext == sigma_kin. We
    therefore keep the standard ideal-crystal histogram structure and only refine
    the low-energy region:

      * ABOVE ``E_active``: reuse the thinned kinematic comb (``kin_table``)
        verbatim -- same energy grid as the no-extinction tape;
      * BELOW ``E_active``: one node per Bragg edge plus tolerance-adaptive
        histogram nodes wherever the STEP value of S=E*sigma jumps by more than
        ``tol`` relatively across a smooth interval.

    The result is a single-region **histogram (INT=1)** cumulative-S table, the
    standard MF7/MT2 form. This is deliberate: a processor reads MF7/MT2 as a step
    function regardless of the INT flag (NJOY THERMR's ``sigcoh`` reads ``np``/``nr``
    but never the INT array, and reconstructs sigma = S(E_i<=E)/E). Tabulating for
    that step -- rather than a lin-lin curve that gets read as a staircase anyway --
    is both smaller and more faithful, and needs no INT-honoring patch.
    """
    kst = kin_table['S_T0_table']
    Ek, Sk = kst['Eint'], kst['S']          # thinned kinematic comb (sigfig'd)

    # split the kinematic comb at the cutoff; everything >= E_active is kinematic
    cut = next((i for i, e in enumerate(Ek) if e >= E_active), len(Ek))
    top = Ek[cut] if cut < len(Ek) else (edge_E[-1] if edge_E else E_active)

    # raw-energy cumulative-S nodes below the cutoff. Each Bragg edge is a node with
    # the POST-jump cumulative (sigma_fn(E) includes the edge at E). Between edges the
    # smooth extinction rise is refined to tol using PRE-jump right endpoints, so the
    # refinement never "chases" a jump and every node keeps a well-defined side.
    #
    # The smooth interior is refined at EVERY temperature and the node energies are
    # UNIONed. Extinction strength falls as temperature rises (the Debye-Waller factor
    # shrinks the effective |F|^2), so a grid resolved only at the reference T0 could
    # under-sample a colder, sharper temperature. E*sigma is monotone within each
    # inter-edge interval, so the union stays tol-bounded for every temperature and,
    # when T0 is the coldest (the usual case), reduces to exactly the T0 grid.
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

    # assemble: below-cutoff fine nodes (raw S, NOT re-evaluated at rounded E), then
    # the kinematic comb verbatim. Dedup to strictly ascending 7-sig-fig energies; on
    # a collision the later (larger, post-jump) cumulative-S wins so jumps survive.
    #
    # Splice continuity: the below-cutoff nodes carry sigma_fn, which returns the FULL
    # kinematic cumulative once E >= E_active, while the appended comb (Ek/Sk) may be
    # edge-GROUPED above the grouping threshold. These never disagree at the seam
    # because the extinction cutoff (x ~ lambda^2 -> 0 above ~0.1 eV) sits well below
    # the grouping threshold (default 1 eV), so the comb is still ungrouped -- i.e.
    # equal to the full cumulative -- at and below `top`. Extinction and grouping act
    # on disjoint energy ranges by construction.
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
      - Single atom (nat=1): Eq 24/25 — store dominant channel, scale by
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
                # sigma_coh <= sigma_inc and sigma_inc == 0 means the
                # principal scatterer has NO elastic channel at all.
                raise ValueError(
                    "SEF single-atom: principal scatterer has zero "
                    "coherent AND incoherent elastic cross sections "
                    "(b_coh=0, sigma_inc=0 on Card 6d) — there is no "
                    "elastic law to represent; remove the elastic "
                    "section or fix the Card 6d cross sections.")
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
            # Eq 26 redistributes the DC atom's incoherent strength onto
            # this atom's incoherent channel — impossible without one.
            raise ValueError(
                "SEF polyatomic: the principal scatterer has "
                "sigma_inc=0 on Card 6d but is not the designated "
                "coherent atom, so the Eq. 26 incoherent redistribution "
                "is undefined. Use elastic_mode=2 (mixed elastic) or "
                "give the principal scatterer its physical incoherent "
                "cross section.")
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
    """Build the coherent-elastic cumulative-S table with high-energy Bragg-edge
    GROUPING, per ENDF-102 sec 7.2.2.

    WHY: above ~1 eV the Bragg edges of some materials/space groups become
    extremely dense and each "stair step" of S(E,T) is tiny. ENDF-102 sec 7.2.2
    permits grouping those edges into fewer steps "while still preserving the
    average value of the cross section". This collapses thousands of negligible
    edges into a handful while keeping the cumulative S -- hence the total bound
    cross section and the high-E 1/E tail -- exact at every group boundary and
    every temperature.

    bins_per_decade: resolution of the grouping above ``threshold_ev``. The
    energy axis above the threshold is split into geometric (log-uniform) bins,
    ``bins_per_decade`` of them per factor-of-10 in energy (bin width ratio
    10**(1/bins_per_decade)). All edges in a bin are merged into ONE step placed
    at the structure-factor-weighted log-mean energy
    ``ln(E_rep) = sum_i d_i ln(E_i) / sum_i d_i`` -- the placement that EXACTLY
    preserves that bin's 1/E cross-section integral at the reference
    temperature T0.

    FIDELITY AT T > T0: the cumulative S at every group boundary stays exact at
    EVERY temperature (the per-T increments are accumulated in full), so the
    total bound cross section and the high-E 1/E tail are exact. Only the
    intra-bin 1/E SHAPE drifts at T != T0, because E_rep is fixed using the T0
    structure-factor weights while each edge's Debye-Waller attenuation is
    T-dependent. That drift is bounded only by the per-temperature relative
    errors that ``_report_grouping_fidelity`` always prints; for materials with
    strongly T-dependent Debye-Waller factors, raise ``bins_per_decade`` (finer
    grouping) if those numbers are too large.
    The number of grouped steps above the threshold is
    ``~ bins_per_decade * log10(emax/threshold)`` regardless of how many raw
    edges the space group produced (e.g. 20/decade over 1->emax=5 eV keeps ~14).
    Larger bins_per_decade -> finer grouping (more steps, closer to ungrouped).

    Edges at or below ``threshold_ev`` are kept individually (unchanged). emax
    is taken from the actual data (the flat endpoint bragg[-1][0]) so the table
    stays defined up to the real upper limit.
    """
    E = np.array([bragg[j][0] for j in range(nedge)], dtype=float)
    # An empty edge set has no upper energy (E[-1]) and no coherent-elastic
    # peak to group; mirror _coherent_s_table and fail fast.
    if nedge == 0 or E.size == 0:
        raise ValueError(
            "coherent elastic: no Bragg edge to group (nedge=0) — there is "
            "no coherent-elastic peak to tabulate. Check Card 6d coherent "
            "lengths / the lattice, or omit the elastic section.")
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
                         dwpix_out, scale, crystal_info=None):
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
    use_dir_dw = species_dw is not None and species_dw.use_dir_dw
    use_ps = species_dw is not None and species_dw.use_ps

    if use_dir_dw:
        atom_types, awr_sp = species_dw.atom_types, species_dw.awr_sp
        print(f"    Per-species directional DW (inelastic_mode=1/2) at T={tempr[0]:.2f}K:", flush=True)
        for si, at in enumerate(atom_types):
            F0 = species_dw.F_species_per_temp[0][si]
            f_ab = 0.5 * (F0[0, 0] + F0[1, 1])
            f_c  = F0[2, 2]
            W_ab = f_ab / (awr_sp[si] * tempr[0] * BK)
            W_c  = f_c  / (awr_sp[si] * tempr[0] * BK)
            print(f"      {at['Z']}-{at['A']}: W_ab={W_ab:.6f}, W_c={W_c:.6f} 1/eV", flush=True)
    elif use_ps:
        print(f"    Per-species DW (ENDF W, 1/eV) at T={tempr[0]:.2f}K:", flush=True)
        for si, at in enumerate(species_dw.atom_types):
            print(f"      {at['Z']}-{at['A']}: W={species_dw.W_ps[si][0]:.6f}", flush=True)

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

    # Optional crystalline EXTINCTION correction (sample-specific). When enabled,
    # the coherent comb is energy-dependent only BELOW a cutoff E_active (above it
    # extinction has died out and sigma_ext == sigma_kin). We keep the standard
    # histogram (INT=1) cumulative-S form: reuse the thinned kinematic comb above
    # E_active and splice in fine adaptive histogram nodes below it.
    ext_cfg = (crystal_info or {}).get('coherent_extinction')
    if ext_cfg:
        from irma.core.elastic_extinction import make_sigma_coh_ext
        crystal = crystal_info['crystal']
        sigma_fn, edge_E, E_active = make_sigma_coh_ext(
            bragg, crystal_info['bragg_dir_terms'], species_dw, dwpix_out,
            crystal.volume, crystal.n_atoms, scale, ext_cfg, tempr)
        # The spliced-in above-cutoff comb composes with edge GROUPING: extinction
        # acts only below ~0.1 eV and the grouping threshold is >= 1 eV, so the two
        # are disjoint — fine extinction nodes below the cutoff, GROUPED edges above
        # the threshold (the extinction tape's high-E region then matches the
        # grouped no-extinction tape).
        kin_table = _coherent_s_table_or_grouped(
            bragg, nedge, ntempr, tempr, _edge_delta, crystal_info)
        mf7mt2.update(_coherent_extinction_s_table(
            kin_table, sigma_fn, edge_E, E_active, ntempr, tempr,
            float(ext_cfg.get('rmse_tol', 1e-3))))
        return mf7mt2

    # No extinction: the plain comb, GROUPED above the threshold if Card 6b asks
    # (ENDF-102 7.2.2). Same helper the extinction splice uses.
    mf7mt2.update(_coherent_s_table_or_grouped(
        bragg, nedge, ntempr, tempr, _edge_delta, crystal_info))
    return mf7mt2

def _build_cef_incoherent(mat, za, awr, ntempr, tempr, dwpix_out, sb_value,
                          npr=1):
    """Build LTHR=2 (incoherent elastic) section for CEF non-DC atom.

    sb_value is the effective PER-PRINCIPAL bound cross section (barns), which
    may include the redistribution factor from Eq 26. The tape stores
    ``SB = sb_value * npr`` -- the same molecular convention the classic
    (iel<0) writer has always used -- so a consumer recovers the
    per-principal value by dividing once by MT4's B(6)=npr regardless of
    which writer produced the tape (pre-release review E1: the generalized
    writer used to store the per-principal value raw, and the ENDFTSL
    converter's uniform division then under-predicted incoherent elastic by
    exactly npr for generalized CEF tapes with npr > 1).
    """
    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 2  # incoherent elastic

    ndw = max(ntempr, 2)
    # FORMAT NOTE (deliberate NJOY divergence): NJOY stores this SB raw
    # (leapr.f90:3169, no sigfig) and its adaptive formatter renders it
    # with up to 9 significant digits in no-exponent form; IRMA renders
    # every value with the uniform 7-significant-figure convention. The
    # values agree to 7 figures — byte comparisons against NJOY tapes
    # must compare this one record numerically.
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
    principal_at = crystal_info['atom_types'][crystal_info['principal_atom_idx']]
    sigma_inc_p = principal_at['sigma_inc']  # barns

    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 3  # mixed elastic

    # --- Resolve directional/per-species DW (same detection as CEF), shared
    #     via irma.core.elastic_dw; MEF carries no structure-factor scale. ---
    species_dw = resolve_species_dw(crystal_info, tempr, ntempr)
    use_dir_dw = species_dw is not None and species_dw.use_dir_dw

    if use_dir_dw:
        atom_types, awr_sp = species_dw.atom_types, species_dw.awr_sp
        print(f"    MEF per-species directional DW (inelastic_mode=1/2) "
              f"at T={tempr[0]:.2f}K:", flush=True)
        for si, at in enumerate(atom_types):
            F0 = species_dw.F_species_per_temp[0][si]
            f_ab = 0.5 * (F0[0, 0] + F0[1, 1])
            f_c = F0[2, 2]
            W_ab = f_ab / (awr_sp[si] * tempr[0] * BK)
            W_c = f_c / (awr_sp[si] * tempr[0] * BK)
            print(f"      {at['Z']}-{at['A']}: W_ab={W_ab:.6f}, W_c={W_c:.6f} 1/eV", flush=True)

    _edge_delta = make_edge_delta(bragg, dwpix_out, scale=1.0,
                                  species_dw=species_dw, tempr=tempr)

    # --- Coherent elastic part (Bragg edges, per-atom average) ---
    mf7mt2['T0'] = tempr[0]
    mf7mt2['LT'] = ntempr - 1

    ext_cfg = (crystal_info or {}).get('coherent_extinction')
    if ext_cfg:
        # Crystalline extinction on the per-atom coherent comb (scale=1.0, the
        # same per-atom normalization as the kinematic MEF comb). Reuses the CEF
        # machinery; composes with grouping (the above-cutoff piece is the grouped
        # comb). The incoherent part below is unaffected.
        from irma.core.elastic_extinction import make_sigma_coh_ext
        crystal = crystal_info['crystal']
        sigma_fn, edge_E, E_active = make_sigma_coh_ext(
            bragg, crystal_info['bragg_dir_terms'], species_dw, dwpix_out,
            crystal.volume, crystal.n_atoms, 1.0, ext_cfg, tempr)
        kin_table = _coherent_s_table_or_grouped(
            bragg, nedge, ntempr, tempr, _edge_delta, crystal_info)
        mf7mt2.update(_coherent_extinction_s_table(
            kin_table, sigma_fn, edge_E, E_active, ntempr, tempr,
            float(ext_cfg.get('rmse_tol', 1e-3))))
    else:
        # Plain coherent comb, GROUPED above the threshold if Card 6b asks.
        mf7mt2.update(_coherent_s_table_or_grouped(
            bragg, nedge, ntempr, tempr, _edge_delta, crystal_info))

    # --- Incoherent elastic part ---
    # Uses same key names as LTHR=2 (SB, NBT, INT, Tint, Wp)
    # endf-parserpy handles LTHR=3 natively with this layout
    ndw = max(ntempr, 2)
    # Molecular convention: per-principal sigma_inc x npr, matching the
    # classic (iel<0) and SEF/CEF incoherent writers (see docstring).
    mf7mt2['SB'] = sigma_inc_p * npr
    mf7mt2['NBT'] = [ndw]
    mf7mt2['INT'] = [2]
    mf7mt2['Tint'] = []
    mf7mt2['Wp'] = []

    for i in range(ndw):
        idx_t = min(i, ntempr - 1)
        mf7mt2['Tint'].append(tempr[idx_t])
        mf7mt2['Wp'].append(sigfig(dwpix_out[idx_t], 7, 0))

    print(f"  MEF: LTHR=3, {mf7mt2['NP']} Bragg edges, σ_inc={sigma_inc_p:.4f} b "
          f"(x npr={npr} on tape)", flush=True)

    return mf7mt2
