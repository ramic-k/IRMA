# Validation methodology

IRMA generates thermal scattering laws, and a scattering law is only as
trustworthy as the evidence behind it. The validation record distinguishes
**numerical verification** from **physical validation**. The classic kernels
are verified against published and freshly generated reference tapes; the
phonopy-based calculations are
compared with the independent Euphonic and OCLIMAX implementations; processed
cross sections are compared with the released ENDF/B-VIII.1 evaluations and
with available measurements; and the evaluation is finally used in a Monte
Carlo simulation of a real beamline. This page describes the reference stack,
the rules that keep a comparison honest, and the supporting tools. The
material pages ([graphite](graphite.md), [beryllium](beryllium.md),
[beryllium oxide](beryllium-oxide.md)) walk through the results.

## The reference stack

No single reference covers the whole problem, so IRMA is checked against
several kinds of evidence, each strongest in a different regime.

| Reference or test | Purpose | Material(s) | Principal result |
| --- | --- | --- | --- |
| Published ENDF/B-VIII.1 tapes | kernel verification | C, Fe, Al, H/CH₂ | ≤ 7×10⁻⁵ |
| Fresh NJOY2016.78 tapes | kernel verification | l-CH₄, o/p-H₂, BeO | exact |
| Euphonic, coherent n = 1 | independent-code verification | C, Be, BeO | R = 1.00001, 1.0002, 0.9998 |
| OCLIMAX, coherent n = 1 | Debye-Waller treatment isolation | C, Be, BeO | C: 1.9 released, 1.02 full-tensor; Be, BeO: 0.99 either |
| OCLIMAX, full law | independent-code verification | C, Be, BeO | R = 0.96, 0.98, 0.99 |
| ENDF/B-VIII.1 evaluations | processed-output comparison | C, Be, BeO | cross-section overlays |
| CrysXT extinction | model-port verification | Be | 0.000%; 0.07% median |
| Measurements | physical validation | C (Steyerl, VISION, ARCS), Be (EXFOR 11204003), BeO (BNL-325), Ni (VENUS, EXFOR) | overlays on the material pages |
| ARCS through McStas | transport application | C | S(Q,E) map and cut morphology |

### Verification against the reference set

IRMA is a Python reimplementation and generalization of the LEAPR module of
NJOY2016, so the first and tightest test is to reproduce the reference
tapes: the published ENDF/B-VIII.1 files, and fresh NJOY2016.78 runs of the
same decks. Eight evaluations test the classic components:

| Material | Components | Elastic | N_T | Reference | Agreement |
| --- | --- | --- | --- | --- | --- |
| Graphite | C | `iel=1` | 10 | published tape | 3.4×10⁻⁵ |
| Fe (bcc) | C | `iel=6` | 6 | published tape | 6.9×10⁻⁵ |
| Al (fcc) | C | `iel=4` | 6 | published tape | 6.2×10⁻⁵ |
| H in CH₂ | C + free-gas C | incoh. | 15 | published tape | 2.0×10⁻⁵ |
| liquid CH₄ | C+T+D | - | 1 | NJOY2016.78 | exact |
| ortho-H₂ | C+T+D+Y+S | - | 7 | NJOY2016.78 | exact |
| para-H₂ | C+T+D+Y+S | - | 7 | NJOY2016.78 | exact |
| BeO | C, 2P | `iel=3` | 8 | NJOY2016.78 | exact |

Components: C = continuous phonon expansion, T = translational (free gas or
diffusion), D = discrete oscillators, Y = Young-Koppel cold H₂, S = Sköld,
2P = two-pass secondary scatterer. N_T is the number of temperatures
compared. Against the published ENDF/B-VIII.1 graphite, bcc iron, fcc
aluminum, and H-in-polyethylene files, the maximum relative difference in
physically significant values of the scattering law is below 7×10⁻⁵, and the
αβ-integrated ratios are within 2×10⁻⁵ of unity. The published cold-hydrogen
tapes were produced with the CAB-modified NJOY-H2D2 rather than standard
LEAPR, so reference tapes for the three liquids were generated from the
supplied decks with unmodified NJOY2016.78; IRMA reproduces them exactly, at
the precision of every tabulated S value and effective temperature. The BeO
case, taken from the NJOY test suite, also agrees exactly and verifies the
two-pass mixed-moderator treatment. The published tapes name their
generating codes in their own MF1/MT451 headers: NJOY LEAPR for graphite,
FLASSH for H in CH2, and no code for Al and Fe, so the comparison targets
are the released files rather than any single generating program. The
decks, IRMA inputs, and reference tapes live in
`tests/native_LEAPR_NJOY_ENDF_validation/`.

!!! note "Why bit-exact matters"
    Reproducing a reference tape to floating-point precision proves that the
    phonon expansion, Debye-Waller treatment, and ENDF formatting are not
    merely *close* but *identical* to the reference implementation. That makes every
    later, looser comparison interpretable: a discrepancy must come from the
    new physics under test, not from a hidden change in the classic path.

### NJOY defects found during verification

The comparisons also exposed four defects in NJOY2016, reported upstream:
THERMR's low-α extrapolation guard tests the wrong variable and can inflate
coherent laws by orders of magnitude (issue 399), THERMR does not honor
lin-lin (INT=2) interpolation through its cross-section reconstruction, the
`discre` early-exit logic can omit the final discrete-oscillator convolution
(issue 402), and the `pb4` coherent-elastic path contains a separate error
(issue 403). The two THERMR defects are fixed by local source patches, and
that corrected THERMR is used for all NJOY processing in the validation
record. See [NJOY interoperability](../njoy.md) for the details and the
patches.

## The integral-ratio metric

The cross-code comparisons account for differences in tabulation. IRMA and
Euphonic sample a finite set of powder directions, whereas OCLIMAX bins
modes from an expanded phonon mesh into ΔQ shells, so sharp one-phonon
features can fall at slightly different energies even when the underlying
calculations agree, and pointwise residuals would depend on grid placement.
The normalization metric therefore integrates rather than compares points.
Each downscattering law is written in the symmetric form
$\bar S(\alpha,\beta) = S(\alpha,\beta)\,e^{-\beta/2}$, and the symmetric
laws are integrated on their own (α, β) grids over a shared domain contained
within both positive-α and positive-β ranges:

$$
R_{AB} \;=\; \frac{\int_{\mathcal{D}} \bar S_A\,d\alpha\,d\beta}
                  {\int_{\mathcal{D}} \bar S_B\,d\alpha\,d\beta},
$$

with IRMA as A and the comparison code as B, and no extrapolation. Two rules
keep the metric honest:

- **Same components on both sides.** The Euphonic ratio compares coherent
  n = 1 scattering from both codes. The OCLIMAX coherent n = 1 ratio
  isolates the coherent part on both sides: the incoherent cross sections
  are zeroed in the OCLIMAX material file (`.oclimax`), and the IRMA side
  is the coherent component of the same mode-2 run. A
  total-IRMA/coherent-Euphonic ratio is never reported.
- **Same phonon model on both sides.** The metric is used only between
  calculations that start from the same phonon model, where any deviation
  isolates implementation and convention differences. It is not applied to
  the released ENDF/B-VIII.1 files: they were built from different phonon
  models, and every properly normalized law integrates to nearly the same
  value anyway (the BeO evaluation still integrates to within 0.1% of IRMA
  mode 2). Model differences appear in the pointwise structure and in the
  processed cross sections instead.

## Comparison discipline

### Matched physical cuts

Codes report S on different internal variables (α/β versus Q/E) and on
different grids. Before two laws are compared as cuts, they are put on a
common physical target (the same momentum transfer Q and the same energy
transfer E) so that a cut through one surface lines up with the same
physical slice through the other. Comparing raw rows by index compares
different physics and is never done.

### Broadened metrics for stochastic powder sampling

The phonopy-backed modes compute powder averages by sampling discrete
directions on a sphere: the validation campaign used 10000 coherent and 1000
incoherent/multiphonon directions (Card 6g `10000 1000 1`; production
defaults are lower, the campaign added headroom). Discrete sampling puts
sharp coherent features at slightly different energies in two codes even
when the physics matches, so agreement is measured through the shared-domain
integral above or through a broadened (resolution-convolved) metric inside
the law's support window, never through raw delta-like bins.

!!! warning "Direction count is a convergence parameter"
    Increasing the direction counts converges the powder average; it is not
    free precision. When a stochastic metric looks noisy, confirm the
    direction count before suspecting the physics.

### Tape-byte regression gates

The classic-kernel reproductions are protected by byte-level regression
gates: the generated tape must match a frozen reference exactly (or to the
documented ≤ 7×10⁻⁵ tolerance). These gates are intentionally unforgiving so
that any change to the classic path, even a harmless-looking refactor, is
caught before it ships. The LEAPR and Euphonic harnesses under `tests/` are
the home for these checks.

## The Debye-Waller convention

!!! warning "Label the Debye-Waller convention on every anisotropic-crystal curve"
    The released OCLIMAX attenuates its coherent powder average with a
    first-order, almost-isotropic Debye-Waller approximation inherited from
    aCLIMAX, and convolves the multiphonon orders with a fully isotropic
    factor; IRMA modes 1 and 2 retain the directional tensor, and IRMA
    mode 0 is fully isotropic. For strongly anisotropic graphite
    ($W_c/W_{ab} \approx 6.6$) the difference is dramatic at high Q: for
    the one-phonon row at E ≈ 2 meV, the released-OCLIMAX-to-mode-2 ratio
    is about 0.48 at Q = 20 Å⁻¹ and 2.3×10⁻⁷ at Q = 50 Å⁻¹. An
    unreleased OCLIMAX build provided by its author, identical except for a
    full-tensor treatment of the coherent powder average, moves those
    ratios to 1.19 and 0.34 and the graphite coherent integral ratio from
    1.9 to 1.02, which pins the divergence to the Debye-Waller treatment
    alone. This is not a bug in either code, and a code-to-code
    "disagreement" is a physics observation until proven otherwise: label
    both curves with their Debye-Waller convention before drawing
    conclusions. The full-tensor build enters only the coherent one-phonon
    comparison; every other OCLIMAX result in the validation record uses
    the released code. The [graphite page](graphite.md) isolates the
    effect, and nearly isotropic [beryllium](beryllium.md) is the control
    where the same curves stay together.

## The role of `thermr_mimic`

Comparing laws is only half the story; evaluators ultimately care about the
cross section THERMR reconstructs from a tape. `thermr_mimic` is a local
reimplementation of THERMR's tabulated-S(α,β) cross-section integration that
integrates the bilinear lin-lin law exactly. On the uniform dQ = 0.05 Å⁻¹
graphite table (the finest grid both programs process stably), it agrees
with corrected NJOY THERMR to a median of 0.05% and at worst 0.8% over
10⁻³ to 4.9 eV. In the [grid-convergence study](../grids.md) it scores all
five grids, because the two finest extend below THERMR's stable α range
(α_min = 6.88×10⁻⁷ and 6.88×10⁻⁹ against THERMR's internal 10⁻⁶ cutoff) and
its exact integration adds no numerical noise of its own.

!!! note "`thermr_mimic` is validation tooling, not part of the package"
    It is not shipped in the `irma` package and is not used by the engine.
    It exists so that law-to-cross-section comparisons are fast,
    deterministic, and free of THERMR's stability limits. Laws converted
    from other codes (e.g. OCLIMAX) can also diverge in real THERMR through
    the short-collision-time extrapolation, so those comparisons use the
    mimic deliberately to compare on equal footing.

The grid-convergence result that anchors the numerics: relative to the
converged dQ = 0.01 Å⁻¹ reference, the automatic grid changes the graphite
0.01–25 meV integral by −0.33% and agrees pointwise to 1.2% rms (largest
local difference 4.3%, at 2.12 meV, in the thermal window around the
inelastic minimum, where the cross section is only about 0.25 b); only a
deliberately coarse dQ = 0.25 Å⁻¹ grid
degrades visibly. See [Automatic grids](../grids.md) for the full table and
figure.

## Honesty rules

The single most important rule in the validation record is that **every
curve is labeled with what it includes**. Two curves that look comparable
can encode different physics (a different Debye-Waller convention, a
different set of channels, a different phonon model), and an unlabeled plot
invites exactly the wrong conclusion. In practice:

- **Name the source and the physics on every curve**: "IRMA mode 2",
  "Euphonic n = 1 (coherent)", "OCLIMAX MAXO=1", "ENDF/B-VIII.1
  graphite+Sd", and so on.
- **State the channels.** An inelastic-plus-absorption total is not a
  measured total above a Bragg cutoff; say so on the plot.
- **State the Debye-Waller convention** whenever an anisotropic crystal is
  involved (see the warning above).
- **Report the metric, not just the number.** A ratio is meaningless without
  its domain: matched cut, shared-domain integral, or broadened comparison.
- **Attribute discrepancies with measurement neutrally.** The graphite total
  runs about 7% below the Steyerl measurement below the Bragg cutoff (mean
  ratio 0.930), while the measured numerical effects (a −0.33% grid effect
  and a +0.10% processing effect on the same thermal integral) are far
  smaller. The offset therefore lies in the physical inputs: it could come
  from the phonon model, from the measured sample, or from the transmission
  measurement itself, and this single comparison cannot tell which. State
  that, rather than assigning a culprit the evidence cannot localize.

The end-to-end example that puts all of this together is the
[graphite validation page](graphite.md).
