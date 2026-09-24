#!/usr/bin/env python3
"""Mode-2 exact one-phonon cross-validation: IRMA vs Euphonic.

Runs IRMA on the committed mode-2 nphon=1 deck (mesh 40^3, ndir=10000) and
compares the resulting MF7/MT4 one-phonon S(alpha,beta) against the frozen
Euphonic powder average of the SAME vendored phonon model
(``<material>/euphonic_n1_reference.npz``, 10000 golden-spiral directions —
see ``generate_euphonic_n1_reference.py`` for exactly how it was produced).

What is compared
----------------
* RAW differential cuts at the 12 validation Q targets
  (0.5, 1, 1.5, 1.75, 2, 3, 5, 7.5, 10, 15, 20, 30 1/A — nearest grid alpha),
  plotted exactly like the historical validation figures. Raw low-Q cuts are
  intrinsically jagged (sparse reciprocal-space sampling), so they are
  PLOTTED but gated only after matched Gaussian broadening in beta.
* Integrated spectra (the numeric gates):
  - alpha-integrated beta spectrum  I(beta) = integral S dalpha
  - beta-integrated alpha spectrum  J(alpha) = integral S dbeta
    (J is the absolute-normalization check: any scale error in either code
    shifts the whole curve.)

IRMA's MT4 stores the symmetric law S(a,b)*exp(-b_act/2) with
b_act = beta * 0.0253/kT (LAT=1); the comparator inverts exactly that
transform to recover the injected asymmetric downscatter law.

This is a SLOW validation harness (not CI): IRMA ~1-2 min on 14 cores.

Usage:
    python validate_mode2_n1.py graphite [--tape out.endf] [--measure]
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the enclosing checkout's irma over any ambient installation. This
# MUST come before every `irma` import (an insert after the first import
# would silently validate the wrong checkout).
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import numpy as np

import irma
from irma.core.constants import BK

print(f"irma resolved from: {irma.__file__}")

Q_TARGETS = [0.5, 1.0, 1.5, 1.75, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
THERM = 0.0253


def _d2a(x):
    if isinstance(x, dict):
        return np.array([x[k] for k in sorted(x.keys())], dtype=float)
    return np.asarray(x, dtype=float)


def run_irma(material_dir, workdir):
    template = open(os.path.join(material_dir,
                                 "irma_mode2_n1.input.template")).read()
    deck = template.replace("PHONOPY_YAML_PATH",
                            os.path.join(material_dir, "phonopy.yaml"))
    deck_path = os.path.join(workdir, "irma_mode2_n1.input")
    tape = os.path.join(workdir, "irma_mode2_n1.endf")
    with open(deck_path, "w") as f:
        f.write(deck)
    from irma.core.engine import run_leapr
    print(f"running IRMA: {deck_path} -> {tape}")
    run_leapr(deck_path, tape)
    return tape


def load_irma_ss(tape):
    """Recover the asymmetric downscatter law from MF7/MT4 (T0 block)."""
    from endf_parserpy import EndfParserPy
    p = EndfParserPy(ignore_number_mismatch=True, ignore_zero_mismatch=True,
                     ignore_varspec_mismatch=True)
    mt4 = p.parsefile(tape)[7][4]
    st = mt4["S_table"]
    nbeta = len(st)
    alpha = _d2a(st[1]["alpha"])
    beta = _d2a(mt4["beta"])
    T0 = float(mt4["T0"])
    sbar = np.zeros((nbeta, len(alpha)))
    for j in range(1, nbeta + 1):
        sbar[j - 1] = _d2a(st[j]["S"])
    sc = THERM / (BK * T0) if mt4["LAT"] == 1 else 1.0
    ss = sbar * np.exp(0.5 * beta * sc)[:, None]   # invert the writer exactly
    return alpha, beta, ss, T0


def broaden(beta, cut, sigma=0.25, bmax=16.0, db=0.02):
    """Matched Gaussian broadening on a uniform beta grid."""
    bu = np.arange(0.0, bmax, db)
    cu = np.interp(bu, beta, cut, left=0.0, right=0.0)
    n = int(4 * sigma / db)
    kern = np.exp(-0.5 * ((np.arange(-n, n + 1) * db) / sigma) ** 2)
    kern /= kern.sum()
    return bu, np.convolve(cu, kern, mode="same")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("material")
    ap.add_argument("--tape", help="pre-produced IRMA tape (skip the run)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write plots (default: <material>_report/)")
    ap.add_argument("--measure", action="store_true",
                    help="report metrics WITHOUT gating and always exit 0 "
                         "(tolerance survey only — never use as an "
                         "automated pass/fail gate)")
    ap.add_argument("--rtol-integrated", type=float, default=0.05,
                    help="gate on I(beta)/J(alpha) where significant")
    ap.add_argument("--rtol-cuts", type=float, default=0.15,
                    help="gate on broadened cuts at Q>=15, applied as twice "
                         "this value (Q<3 and 3<=Q<15 use 0.10 and 0.20)")
    args = ap.parse_args(argv[1:])

    mat_dir = os.path.join(HERE, args.material)
    out_dir = args.out_dir or os.path.join(HERE, f"{args.material}_report")
    os.makedirs(out_dir, exist_ok=True)

    ref = np.load(os.path.join(mat_dir, "euphonic_n1_reference.npz"))
    meta = json.loads(str(ref["metadata"]))

    # The reference must have been generated from THIS deck template:
    # a mesh or direction-count mismatch silently invalidates the gate.
    from generate_euphonic_n1_reference import parse_deck_template
    (_, _, _, _, _, _, deck_mesh, deck_ndir) = parse_deck_template(mat_dir)
    if (list(meta.get("dw_mesh", [])) != [deck_mesh] * 3
            or int(meta.get("npts", -1)) != deck_ndir):
        raise SystemExit(
            f"euphonic_n1_reference.npz was generated with "
            f"dw_mesh={meta.get('dw_mesh')}, npts={meta.get('npts')} but the "
            f"deck template specifies mesh={deck_mesh}^3, ndir={deck_ndir}; "
            f"regenerate the reference (generate_euphonic_n1_reference.py) "
            f"or fix the template before gating.")
    a_eu, b_eu, ss_eu, q_eu = (ref["alpha"], ref["beta"], ref["ss"],
                               ref["q_of_alpha"])

    tape = args.tape or run_irma(mat_dir, tempfile.mkdtemp())
    a_th, b_th, ss_th, T0 = load_irma_ss(tape)
    assert np.allclose(a_th, a_eu, rtol=5e-7) and np.allclose(b_th, b_eu, rtol=5e-7), \
        "IRMA tape grid differs from the Euphonic reference grid"
    print(f"grids match: {len(a_th)} alpha x {len(b_th)} beta, T0={T0} K, "
          f"euphonic: npts={meta['npts']}, dw_mesh={meta['dw_mesh']}")

    bsel = b_th <= meta["beta_max_tabulated"]
    beta = b_th[bsel]
    Sth, Seu = ss_th[bsel], ss_eu[bsel]

    # ---------------- integrated spectra (the numeric gates) ----------------
    # Both codes bin sharp one-phonon delta contributions with DIFFERENT
    # stochastic samplings (mesh shells vs golden sphere), so raw per-bin
    # ratios carry sampling jitter and the steep support edge at
    # beta = omega_max/kT amplifies tiny edge-shape differences into huge
    # relative errors. The gates therefore use matched-broadened curves inside
    # the one-phonon support window plus integral (L1/total) metrics; the raw
    # overlays are what the plots show.
    I_th = np.trapezoid(Sth, a_th, axis=1)
    I_eu = np.trapezoid(Seu, a_th, axis=1)
    J_th = np.trapezoid(Sth, beta, axis=0)
    J_eu = np.trapezoid(Seu, beta, axis=0)

    ratio = float(np.trapezoid(J_th, a_th) / np.trapezoid(J_eu, a_th))

    # broadened I(beta) inside the one-phonon support window (excludes the
    # quasi-elastic first bins and the cutoff edge at beta = omega_max/kT,
    # which is derived from the reference itself — material dependent:
    # ~7.85 for graphite, ~3.1 for Be at 296 K)
    sigma_b = 0.25
    beta_edge = float(beta[I_eu > 0].max())
    bu, Ib_th = broaden(beta, I_th, sigma=sigma_b)
    _, Ib_eu = broaden(beta, I_eu, sigma=sigma_b)
    win = (bu >= 0.3) & (bu <= beta_edge - 3 * sigma_b) & \
        (Ib_eu > 1e-3 * Ib_eu.max())
    print(f"one-phonon support edge: beta = {beta_edge:.3f} "
          f"(gate window 0.3 .. {beta_edge - 3*sigma_b:.2f})")
    relI = float(np.max(np.abs(Ib_th[win] - Ib_eu[win]) / Ib_eu[win]))
    medI = float(np.median(np.abs(Ib_th[win] - Ib_eu[win]) / Ib_eu[win]))

    # J(alpha): median over the well-sampled region Q<=20; at higher Q the
    # Debye-Waller-suppressed powder average is direction-sampling limited.
    mj = (J_eu > 1e-3 * J_eu.max()) & (q_eu <= 20.0)
    relJ_med = float(np.median(np.abs(J_th[mj] - J_eu[mj]) / J_eu[mj]))
    relJ_max = float(np.max(np.abs(J_th[mj] - J_eu[mj]) / J_eu[mj]))

    print(f"\nglobal integral ratio IRMA/Euphonic = {ratio:.5f}")
    print(f"I(beta) broadened, window 0.3<=b<={beta_edge - 3*sigma_b:.2f}: "
          f"max-rel {relI:.3e}  median {medI:.3e}")
    print(f"J(alpha) Q<=20: max-rel {relJ_max:.3e}  median {relJ_med:.3e}")

    # ---------------- cuts at the validation Q targets ----------------------
    # Gate metric: relative L1 distance of the matched-broadened cuts
    # (integral metric — robust to bin jitter and the support edge).
    cut_metrics = []
    for qt in Q_TARGETS:
        j = int(np.argmin(np.abs(q_eu - qt)))
        bu, cth = broaden(beta, Sth[:, j])
        _, ceu = broaden(beta, Seu[:, j])
        rel_l1 = float(np.trapezoid(np.abs(cth - ceu), bu)
                       / np.trapezoid(ceu, bu)) if ceu.max() > 0 else np.inf
        cut_metrics.append((qt, q_eu[j], j, rel_l1))
        print(f"cut Q={q_eu[j]:7.3f} (target {qt:>5}): broadened rel-L1 "
              f"{rel_l1:.3f}")

    # ---------------- plots -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(20, 13), sharex=True)
    for ax, (qt, q, j, rel_l1) in zip(axes.ravel(), cut_metrics):
        ax.semilogy(beta, Seu[:, j], "r--", lw=1.2, label="Euphonic n=1")
        ax.semilogy(beta, Sth[:, j], "g-.", lw=1.2, label="IRMA mode-2 n=1")
        ax.set_title(f"Q = {q:.3f} A$^{{-1}}$ (target {qt})  "
                     f"[broadened rel-L1 {rel_l1:.1%}]", fontsize=10)
        ax.set_ylim(1e-6, 2.0)
        ax.set_xlim(0, 15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("beta")
    for ax in axes[:, 0]:
        ax.set_ylabel("S(alpha,beta)")
    fig.suptitle(f"{args.material}: IRMA mode-2 exact n=1 vs Euphonic "
                 f"(mesh {meta['dw_mesh'][0]}^3, {meta['npts']} directions, "
                 f"T={T0:.0f} K) — raw cuts", fontsize=13)
    fig.tight_layout()
    cuts_png = os.path.join(out_dir, "sab_cuts_irma_vs_euphonic.png")
    fig.savefig(cuts_png, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].semilogy(beta, I_eu, "r--", label="Euphonic")
    axes[0, 0].semilogy(beta, I_th, "g-.", label="IRMA")
    axes[0, 0].set_xlabel("beta"); axes[0, 0].set_ylabel("I(beta)")
    axes[0, 0].set_title("alpha-integrated beta spectrum"); axes[0, 0].legend()
    axes[0, 1].plot(beta, np.divide(I_th, I_eu, out=np.ones_like(I_th),
                                    where=I_eu > 1e-3 * I_eu.max()))
    axes[0, 1].axhline(1.0, color="k", lw=0.6)
    axes[0, 1].set_ylim(0.8, 1.2)
    axes[0, 1].set_xlabel("beta"); axes[0, 1].set_title("I ratio IRMA/Euphonic")
    axes[1, 0].loglog(a_th, J_eu, "r--", label="Euphonic")
    axes[1, 0].loglog(a_th, J_th, "g-.", label="IRMA")
    axes[1, 0].set_xlabel("alpha"); axes[1, 0].set_ylabel("J(alpha)")
    axes[1, 0].set_title("beta-integrated alpha spectrum"); axes[1, 0].legend()
    axes[1, 1].semilogx(a_th, np.divide(J_th, J_eu, out=np.ones_like(J_th),
                                        where=J_eu > 1e-3 * J_eu.max()))
    axes[1, 1].axhline(1.0, color="k", lw=0.6)
    axes[1, 1].set_ylim(0.8, 1.2)
    axes[1, 1].set_xlabel("alpha"); axes[1, 1].set_title("J ratio IRMA/Euphonic")
    fig.suptitle(f"{args.material}: integrated spectra (gates)", fontsize=13)
    fig.tight_layout()
    integ_png = os.path.join(out_dir, "integrated_irma_vs_euphonic.png")
    fig.savefig(integ_png, dpi=130)
    plt.close(fig)
    print(f"\nplots: {cuts_png}\n       {integ_png}")

    if args.measure:
        print("\n--measure: metrics reported, no gating")
        return 0

    # Gates (defaults sit ~2x above the measured graphite agreement floor;
    # see README for the measured values).
    checks = [
        ("global integral ratio", abs(ratio - 1.0), 0.03),
        ("I(beta) broadened max-rel", relI, args.rtol_integrated * 3),
        ("I(beta) broadened median", medI, args.rtol_integrated),
        ("J(alpha) Q<=20 median", relJ_med, args.rtol_integrated),
    ]
    for qt, q, j, rel_l1 in cut_metrics:
        tol = 0.10 if qt < 3.0 else (0.20 if qt < 15.0 else args.rtol_cuts * 2)
        checks.append((f"cut Q~{qt} rel-L1", rel_l1, tol))

    ok = True
    print()
    for name, val, tol in checks:
        good = val <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {val:.3f} (tol {tol})")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
