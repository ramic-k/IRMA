# Mode-0 (DOS) spectra validation

Self-contained cross-validation of the DOS-based `inelastic_mode=0` forward
spectrum (not part of CI; needs phonopy + the graphite fixture in
`../mode2_euphonic_n1_validation/graphite/`).

```bash
OMP_NUM_THREADS=1 python validate_mode0_vs_mode1.py
```

`validate_mode0_vs_mode1.py` runs mode 0 (DOS, isotropic Debye-Waller, sourced
from the phonopy partial DOS) and mode 1 (the eigenvector engine, anisotropic
Debye-Waller) on graphite at VISION, and writes
`mode0_vs_mode1_graphite.{json,png}`. It exits nonzero if the sanity bounds
fail (integral ratio outside [0.9, 1.1] or shape cosine below 0.85 — generous
by design, see below).

This is a sanity-level consistency check, not a controlled proof. Result
(16³ mesh, 300 K, with the histogram phonon DOS): both paths are normalized
per represented atom, so they agree in absolute scale, with an integral ratio
mode0/mode1 of **0.998**, and the area-normalized shapes track closely (cosine
0.970; both peak at 1 meV). The remaining shape difference is consistent with
mode 0's isotropic Debye-Waller factor against mode 1's anisotropic one on
this strongly anisotropic crystal (the same isotropic approximation OCLIMAX
`TASK=0` makes); it is not separately isolated here. A quantitative
comparison against OCLIMAX `TASK=0` and measured hydrogen-rich spectra has
not been done. The per-atom normalization convention is documented in
`docs/spectra-mode0.md`.
