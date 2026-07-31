#!/usr/bin/env python3
"""Validate a IRMA-reproduced TSL ENDF tape against its reference.

For a derived IRMA deck (see ``leapr_to_irma_input.py``) this:

  1. runs IRMA on ``leapr_decks/tsl-<name>.input`` (unless an already-produced
     tape is supplied), and
  2. compares the resulting MF7/MT4 (inelastic S(alpha,beta)) and MF7/MT2
     (coherent / incoherent elastic) against the reference ``<name>.endf``.

The inelastic law is the primary check: IRMA recomputes S(alpha,beta) from the
deck's phonon spectrum on the SAME alpha/beta grid as the reference, so the
tables are directly comparable point by point. The elastic channel is reported
for diagnostics — for the built-in crystalline materials (iel=1/4/6) the Bragg
structure comes from IRMA's own ``coher`` lattice.

This is a SLOW validation harness, not a unit test, but it is fully
self-contained: the source LEAPR deck (``tsl-<name>.leapr``), its IRMA
translation (``tsl-<name>.input``) and the reference tape
(``tsl-<name>.endf.gz``) all live together under ``leapr_decks/``. Run it
manually:

    python tests/native_LEAPR_NJOY_ENDF_validation/validate_native_leapr_endf.py \\
        crystalline-graphite

Exit status is non-zero if the inelastic agreement is worse than --rtol.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DECK_DIR = os.path.join(HERE, "leapr_decks")


def _d2a(x):
    """endf_parserpy stores tables as 1-based dicts; normalize to ndarray."""
    if isinstance(x, dict):
        return np.array([x[k] for k in sorted(x.keys())], dtype=float)
    return np.asarray(x, dtype=float)


def _parse(path):
    from endf_parserpy import EndfParserPy
    p = EndfParserPy(ignore_number_mismatch=True,
                     ignore_zero_mismatch=True,
                     ignore_varspec_mismatch=True)
    if path.endswith(".gz"):
        # Reference tapes are vendored gzipped; decompress to a temp file
        # (endf_parserpy parses by path) and parse that.
        import gzip
        import tempfile
        with gzip.open(path, "rt") as fh:
            text = fh.read()
        tmp = tempfile.NamedTemporaryFile("w", suffix=".endf", delete=False)
        try:
            tmp.write(text)
            tmp.close()
            return p.parsefile(tmp.name)
        finally:
            os.remove(tmp.name)
    return p.parsefile(path)


def _mt4_temps(mt4):
    """Return [T0, T1, ...] for the inelastic section."""
    temps = [float(mt4["T0"])]
    extra = mt4.get("T", {})
    temps += [float(extra[k]) for k in sorted(extra.keys())]
    return temps


def _mt4_S_at_temp(mt4, itemp):
    """S(alpha, beta) as a (nbeta, nalpha) array for temperature index itemp.

    itemp=0 is the T0 TAB1 block (S_table); itemp>0 are the LIST blocks (S).
    """
    s_table = mt4["S_table"]                       # {beta_idx: {alpha, S, ...}}
    nbeta = len(s_table)
    nalpha = len(s_table[1]["alpha"])
    out = np.zeros((nbeta, nalpha))
    if itemp == 0:
        for j in range(1, nbeta + 1):
            out[j - 1] = _d2a(s_table[j]["S"])
    else:
        # Extra-temperature LIST blocks are nested S[alpha_idx][beta_idx][temp_idx].
        S = mt4["S"]
        for i in range(1, nalpha + 1):
            col = S[i]
            for j in range(1, nbeta + 1):
                out[j - 1, i - 1] = col[j][itemp]
    return out


def compare_mt4(ref, thw):
    rmt4, tmt4 = ref[7][4], thw[7][4]
    r_temps, t_temps = _mt4_temps(rmt4), _mt4_temps(tmt4)
    r_alpha = _d2a(rmt4["S_table"][1]["alpha"])
    t_alpha = _d2a(tmt4["S_table"][1]["alpha"])
    r_beta = _d2a(rmt4["beta"])
    t_beta = _d2a(tmt4["beta"])

    # Pointwise S comparison is only meaningful on identical grids.
    if len(r_alpha) != len(t_alpha) or len(r_beta) != len(t_beta):
        print(f"GRID MISMATCH: alpha {len(r_alpha)} vs {len(t_alpha)}, "
              f"beta {len(r_beta)} vs {len(t_beta)} — the tapes were made "
              f"on different grids; pointwise comparison aborted.")
        # Same (worst_sig, worst_int) shape as the normal path: both gates
        # exceed any tolerance, so the caller prints a clean FAIL.
        return float("inf"), float("inf")

    print(f"  MT4 LAT={rmt4['LAT']}/{tmt4['LAT']} LASYM={rmt4['LASYM']}/{tmt4['LASYM']} "
          f"LLN={rmt4['LLN']}/{tmt4['LLN']}")
    print(f"  MT4 alpha pts: ref={len(r_alpha)} irma={len(t_alpha)}  "
          f"(max|d alpha|={np.max(np.abs(r_alpha - t_alpha)) if len(r_alpha)==len(t_alpha) else 'n/a'})")
    print(f"  MT4 beta pts:  ref={len(r_beta)} irma={len(t_beta)}   "
          f"(max|d beta|={np.max(np.abs(r_beta - t_beta)) if len(r_beta)==len(t_beta) else 'n/a'})")
    print(f"  MT4 temps: ref={r_temps}")
    print(f"             thw={t_temps}")

    # Principal effective temperatures (Teff0) — used by THERMR's SCT
    # extension, so zeros/garbage here corrupt downstream cross sections even
    # when the S tables agree.
    r_teff = _d2a(rmt4["teff0_table"]["Teff0"])
    t_teff = _d2a(tmt4["teff0_table"]["Teff0"])
    if len(r_teff) == len(t_teff) and np.all(r_teff > 0):
        teff_rel = float(np.max(np.abs(t_teff - r_teff) / r_teff))
        print(f"  MT4 Teff0 max rel diff: {teff_rel:.3e}  "
              f"(ref[0]={r_teff[0]:.2f} K, thw[0]={t_teff[0]:.2f} K)")
    else:
        teff_rel = float("inf")
        print(f"  MT4 Teff0 MISMATCH: ref={r_teff[:3]}... thw={t_teff[:3]}...")

    worst_sig = teff_rel       # Teff0 disagreement fails the same rtol gate
    worst_int = 0.0
    ntemp = min(len(r_temps), len(t_temps))
    # "max rel d" over all non-trivial points is dominated by a handful of
    # near-zero tail points; "max rel d (sig)" restricts to physically
    # significant points (S > 1e-3 * Smax) and is the metric used for PASS,
    # together with the alpha/beta-integrated "sum S ratio".
    print(f"  {'T (K)':>8} {'max rel d':>11} {'max rel d(sig)':>15} "
          f"{'median rel d':>13} {'sum S ratio':>12}")
    for it in range(ntemp):
        Sr = _mt4_S_at_temp(rmt4, it)
        St = _mt4_S_at_temp(tmt4, it)
        smax = Sr.max()
        m = Sr > 1e-6 * smax          # any non-trivial point
        sig = Sr > 1e-3 * smax        # physically significant point
        rel = np.abs(St[m] - Sr[m]) / Sr[m]
        rel_sig = np.abs(St[sig] - Sr[sig]) / Sr[sig]
        maxrel = float(rel.max()) if rel.size else 0.0
        maxrel_sig = float(rel_sig.max()) if rel_sig.size else 0.0
        medrel = float(np.median(rel)) if rel.size else 0.0
        ratio = float(St[m].sum() / Sr[m].sum()) if m.any() else float("nan")
        worst_sig = max(worst_sig, maxrel_sig)
        worst_int = max(worst_int, abs(ratio - 1.0))
        print(f"  {r_temps[it]:8.1f} {maxrel:11.3e} {maxrel_sig:15.3e} "
              f"{medrel:13.3e} {ratio:12.6f}")
    return worst_sig, worst_int


def compare_mt2(ref, thw):
    if 2 not in ref[7] or 2 not in thw[7]:
        print("  MT2: absent in one tape — skipping elastic comparison")
        return
    rmt2, tmt2 = ref[7][2], thw[7][2]
    print(f"  MT2 LTHR={rmt2['LTHR']}/{tmt2['LTHR']}")
    if rmt2["LTHR"] == 1:  # coherent elastic: cumulative S(E) vs E
        rE = _d2a(rmt2["S_T0_table"]["Eint"])
        tE = _d2a(tmt2["S_T0_table"]["Eint"])
        rS = _d2a(rmt2["S_T0_table"]["S"])
        tS = _d2a(tmt2["S_T0_table"]["S"])
        print(f"  MT2 coherent-elastic edges (E grid pts): ref={len(rE)} irma={len(tE)}")
        print(f"  MT2 E range: ref[{rE[0]:.4e},{rE[-1]:.4e}] irma[{tE[0]:.4e},{tE[-1]:.4e}]")
        print(f"  MT2 cumulative S at Emax (~total coh-el strength): "
              f"ref={rS[-1]:.5e} irma={tS[-1]:.5e} ratio={tS[-1]/rS[-1]:.5f}")
    else:
        print(f"  MT2 LTHR={rmt2['LTHR']} (incoherent/mixed) — compare debye-waller/SB")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="material tag, e.g. crystalline-graphite, 026_Fe_056, 013_Al_027")
    ap.add_argument("--ref", help="reference tape (default: the vendored "
                                  "leapr_decks/tsl-<name>.endf[.gz])")
    ap.add_argument("--deck", help="IRMA .input (default: leapr_decks/tsl-<name>.input)")
    ap.add_argument("--tape", help="use a pre-produced IRMA tape instead of running IRMA")
    ap.add_argument("--rtol", type=float, default=0.02,
                    help="max allowed inelastic max-rel-diff at significant S "
                         "(default 0.02; ~1e-4 for the NJOY-LEAPR references here)")
    ap.add_argument("--itol", type=float, default=0.01,
                    help="max allowed |sum-S ratio - 1| per temperature (default 0.01)")
    args = ap.parse_args(argv[1:])

    deck = args.deck or os.path.join(DECK_DIR, f"tsl-{args.name}.input")
    # Default to the reference tape vendored alongside the deck (gzip preferred).
    ref = args.ref
    if ref is None:
        cand_gz = os.path.join(DECK_DIR, f"tsl-{args.name}.endf.gz")
        cand = os.path.join(DECK_DIR, f"tsl-{args.name}.endf")
        ref = cand_gz if os.path.exists(cand_gz) else cand
    if not os.path.exists(ref):
        print(f"reference not found: {ref}\n(pass --ref to point at the reference tape)")
        return 2

    tape = args.tape
    if tape is None:
        tape = f"/tmp/irma_expected_{args.name}.endf"
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), ".."))
        from irma.core.engine import run_leapr
        print(f"running IRMA: {deck} -> {tape}")
        run_leapr(deck, tape)

    print(f"\nparsing reference: {ref}")
    rp = _parse(ref)
    print(f"parsing IRMA:    {tape}")
    tp = _parse(tape)

    print("\n=== MF7/MT4 inelastic S(alpha,beta) ===")
    worst_sig, worst_int = compare_mt4(rp, tp)
    print("\n=== MF7/MT2 elastic ===")
    compare_mt2(rp, tp)

    print(f"\nworst inelastic max-rel-diff (significant S) = {worst_sig:.3e}  (rtol={args.rtol})")
    print(f"worst |sum-S ratio - 1|                      = {worst_int:.3e}  (itol={args.itol})")
    ok = (worst_sig <= args.rtol) and (worst_int <= args.itol)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
