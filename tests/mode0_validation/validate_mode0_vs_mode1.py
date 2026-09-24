#!/usr/bin/env python
"""Cross-validate the DOS-based mode-0 forward spectrum against the phonopy
eigenvector engine (mode 1, incoherent approximation) on graphite.

Both are the INCOHERENT APPROXIMATION and both are normalized PER REPRESENTED
ATOM, so they should agree in ABSOLUTE scale -- the integral ratio
mode0/mode1 should be ~ 1. The one understood difference is the Debye-Waller:
mode 0 is ISOTROPIC (one scalar lambda_s per species from the DOS), mode 1 is
ANISOTROPIC (per-atom U tensors from phonopy). Graphite is strongly anisotropic,
so a small residual (in the integral and in the shape) is expected
and physical -- the same isotropic approximation OCLIMAX TASK=0 makes.

Run (needs phonopy):
    OMP_NUM_THREADS=1 python validate_mode0_vs_mode1.py
Writes mode0_vs_mode1_graphite.{json,png} beside this file. Exits nonzero if
the sanity bounds fail (see the gate at the end of main()).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the enclosing checkout's irma over any ambient installation (same
# convention as ../mode2_euphonic_n1_validation/validate_mode2_n1.py); must
# come before every `irma` import.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import numpy as np

import irma

print(f"irma resolved from: {irma.__file__}")
YAML = os.path.join(HERE, "..", "mode2_euphonic_n1_validation", "graphite", "phonopy.yaml")
MESH = [16, 16, 16]
T_K = 300.0
EGRID = dict(e_min_meV=0.0, e_max_meV=220.0, de_meV=1.0, dq_max_invA=0.1)
# natural carbon
C = dict(symbol="C", sigma_bound_b=5.551, awr=11.898, b_coh_fm=6.646, sigma_inc_b=0.001)


def _base(mode, **physics):
    from irma.spectra.config import SpectraConfig
    return SpectraConfig.from_dict({
        "material": {"temperature_K": T_K, "phonopy_yaml": YAML, "mesh": MESH,
                     "scatterers": [C]},
        "physics": {"inelastic_mode": mode, "elastic": False,
                    "max_phonon_order": "auto", "n_directions": 4000,
                    "multiphonon_directions": 400, **physics},
        "grid": EGRID,
        "instrument": {"geometry": "vision", "e_fixed_meV": 3.5,
                       "angles_deg": [45.0, 135.0]},
    })


def main():
    from irma.spectra.config import run_spectra

    print("mode 0 (DOS, isotropic DW, dos_source=phonopy) ...")
    r0 = run_spectra(_base(0, dos_source="phonopy"), progress=lambda *a, **k: None)
    print("mode 1 (engine, anisotropic DW) ...")
    r1 = run_spectra(_base(1), progress=lambda *a, **k: None)

    E = r0.E
    I0, I1 = np.asarray(r0.I_inelastic, float), np.asarray(r1.I_inelastic, float)
    # integral (trapezoid) ratio pins the absolute (per-atom) scale
    A0, A1 = np.trapezoid(I0, E), np.trapezoid(I1, E)
    scale = A0 / A1 if A1 else float("nan")
    # shape agreement after removing the global scale: cosine similarity
    n0 = I0 / (np.linalg.norm(I0) or 1.0)
    n1 = I1 / (np.linalg.norm(I1) or 1.0)
    cos = float(n0 @ n1)
    summary = {
        "mesh": MESH, "T_K": T_K, "n_atoms_cell": 4,
        "integral_mode0": float(A0), "integral_mode1": float(A1),
        "integral_ratio_mode0_over_mode1": float(scale),
        "shape_cosine_similarity": cos,
    }
    with open(os.path.join(HERE, "mode0_vs_mode1_graphite.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(E, I0, label="mode 0 (DOS, isotropic DW)")
        ax1.plot(E, I1, label="mode 1 (engine, anisotropic DW)")
        ax1.set_xlabel("E (meV)"); ax1.set_ylabel("I_inel (per atom)")
        ax1.set_title("graphite VISION -- absolute"); ax1.legend()
        ax2.plot(E, I0 / (A0 or 1.0), label="mode 0")
        ax2.plot(E, I1 / (A1 or 1.0), label="mode 1")
        ax2.set_xlabel("E (meV)"); ax2.set_ylabel("area-normalized I_inel")
        ax2.set_title(f"shape (cos={cos:.4f}, scale m0/m1={scale:.2f})")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "mode0_vs_mode1_graphite.png"), dpi=110)
        print("wrote mode0_vs_mode1_graphite.png")
    except Exception as e:                       # plotting is optional
        print(f"(plot skipped: {e})")

    # Sanity gate. This harness is a consistency check, not a controlled
    # proof (see README.md), so the bounds are deliberately generous: the
    # committed graphite result is ratio ~ 1.016 and cosine ~ 0.92, and the
    # known physical residual (mode 0 isotropic vs mode 1 anisotropic
    # Debye-Waller) is a few percent on this strongly anisotropic crystal.
    # A violation means one of the paths broke, not that the residual grew.
    RATIO_LO, RATIO_HI = 0.9, 1.1
    COS_MIN = 0.85
    checks = [
        (f"integral ratio mode0/mode1 in [{RATIO_LO}, {RATIO_HI}]",
         RATIO_LO <= scale <= RATIO_HI, scale),
        (f"shape cosine similarity >= {COS_MIN}", cos >= COS_MIN, cos),
    ]
    ok = True
    print()
    for name, good, val in checks:
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {val:.4f}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
