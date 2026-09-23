# Mode-2 exact one-phonon validation against Euphonic

Cross-code validation of IRMA's `inelastic_mode=2` **exact n=1**
(coherent + incoherent one-phonon) S(α,β): the same vendored phonon model is
fed to IRMA and to **Euphonic** (an independent, widely used phonon
spectroscopy library), and the powder-averaged one-phonon laws are compared
differentially (cuts at fixed Q) and integrally.

This is a SLOW validation harness — it is **not part of CI**. Run it manually
when mode-2 physics changes.

## Why n=1

`nphon=1` isolates the hardest mode-2 physics — the coherent one-phonon term
(eigenvectors, structure factors, anisotropic Debye-Waller, powder average) —
with no multiphonon convolution, no SCT, and no multiphonon-direction
convergence questions. Graphite and beryllium are essentially pure coherent
scatterers (σ_inc/σ_coh < 4·10⁻⁴), so IRMA's n=1 total and Euphonic's
coherent one-phonon are the same physical quantity. Both calculations use a
40×40×40 mesh for the Debye-Waller factor and **10000 powder directions**
(IRMA `ndir`; Euphonic golden-spiral sphere sampling).

## What is here

```
graphite/                       beryllium/
  phonopy.yaml                    (same layout)
  FORCE_CONSTANTS                 hcp Be, a=2.2856, c=3.5842 A
  irma_mode2_n1.input.template  mode-2 deck: nphon=1, mesh 40^3, ndir=10000
  euphonic_n1_reference.npz       frozen Euphonic powder average (committed)
generate_euphonic_n1_reference.py reproducible origin of the .npz
validate_mode2_n1.py              runs IRMA, gates + writes the plots
```

The deck template, the Euphonic generator and the comparator all read the
grid/atom data from the same committed deck, so the two calculations cannot
drift apart silently. The `.npz` carries its full provenance (mesh, npts,
sampling, temperature, conversion convention, euphonic version) as JSON
metadata.

## Running

```bash
python tests/mode2_euphonic_n1_validation/validate_mode2_n1.py graphite
python tests/mode2_euphonic_n1_validation/validate_mode2_n1.py beryllium
# regenerate the frozen reference (only needed if the phonon model changes):
python tests/mode2_euphonic_n1_validation/generate_euphonic_n1_reference.py graphite
```

IRMA runs ~1-2 min on 14 cores; the comparator then writes
`<material>_report/sab_cuts_irma_vs_euphonic.png` (raw cuts at the 12
validation Q targets: 0.5, 1, 1.5, 1.75, 2, 3, 5, 7.5, 10, 15, 20, 30 1/A)
and `integrated_irma_vs_euphonic.png`.

## How the comparison is gated

Both codes bin sharp one-phonon δ-contributions with *different* stochastic
samplings (IRMA mesh shells, Euphonic golden sphere), so raw per-bin ratios
carry sampling jitter, and the steep support edge at β = ω_max/kT (≈7.85 for
graphite at 296 K) turns tiny edge-shape differences into huge pointwise
relative errors. The plots therefore show the RAW overlays, while the gates
use:

* the **global integral ratio** (absolute normalization — no free scale
  factors anywhere in the chain),
* matched-Gaussian-broadened (σ_β=0.25) **α-integrated β spectrum** inside
  the one-phonon support window 0.3 ≤ β ≤ β_edge−3σ, where β_edge =
  ω_max/kT is derived from the reference itself (≈7.9 for graphite, ≈3.3
  for Be at 296 K),
* the **β-integrated α spectrum** median over Q ≤ 20 (beyond that the
  Debye-Waller-suppressed powder average is direction-sampling limited),
* per-cut **relative L1 distance** of the broadened cuts (integral metric),
  with tiered tolerances (Q<3: 10%, 3≤Q<15: 20%, Q≥15: 30%).

## Measured agreement (296 K)

| metric | graphite | beryllium |
|---|---|---|
| global integral ratio IRMA/Euphonic | **1.011** | **1.021** |
| I(β) broadened, support window: median / max | 2.7% / 8.4% | 0.9% / 5.5% |
| J(α) median (Q ≤ 20) | 1.9% | 0.4% |
| cuts rel-L1, Q ≤ 2 | 1.2 – 5.2% | 2.4 – 6.9% |
| cuts rel-L1, 3 ≤ Q ≤ 15 | 3.1 – 11.4% | 2.0 – 4.9% |
| cuts rel-L1, Q = 20 / 30 | 14.8% / 20.4% | 4.5% / 4.9% |

These numbers compare IRMA's total n = 1 law with Euphonic's coherent one on
this harness's grid. They are not the coherent-only shared-domain ratios
(1.00001 graphite, 1.0002 beryllium) quoted on the validation pages, which
come from a separate comparison.

The ~1–2% global ratio is the absolute-scale check: the unit chain
(Euphonic mb/sr → unitless S(Q,E) via σ_coh → SS(α,β) = kT·S) contains no
adjustable factors. The graphite Q=20/30 cuts are sampling-noise limited
(strongly anisotropic Debye-Waller concentrates the powder average in few
directions); isotropic-DW beryllium stays at ~5% there. Expected console
warning from IRMA: the one-phonon law is identically zero above
β = ω_max/kT while the deck grid extends further — that is the physics of
n=1, not an error.
