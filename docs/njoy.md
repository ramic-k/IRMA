# NJOY interoperability

IRMA writes standard ENDF-6 MF7 tapes carrying the thermal scattering
law S(α,β), so the files it produces are meant to flow straight into
NJOY's processing chain. This page shows how to push an IRMA MF7 tape
through RECONR / BROADR / THERMR / ACER, documents the two stock-NJOY
THERMR defects that matter for coherent `inelastic_mode = 2` tapes and
how to patch them, and describes `thermr_mimic`, the offline validation
tool that compares two S(α,β) tables without launching a full NJOY job,
along with the cases where only a real THERMR run will do.

## Processing an IRMA tape with NJOY

IRMA produces the MF7 evaluation; NJOY turns it into broadened,
self-shielded, ACE-format thermal data. The usual module order is:

| Module | Role for a thermal scattering evaluation |
| --- | --- |
| RECONR | Reconstruct the pointwise cross sections for the principal scatterer's neutron sublibrary and write a clean PENDF tape. |
| BROADR | Doppler-broaden those cross sections to the evaluation temperature. |
| THERMR | Read IRMA's MF7 (MT2 coherent/incoherent elastic, MT4 inelastic), reconstruct the secondary energy–angle distributions, and add the thermal cross sections to the PENDF tape. |
| ACER | Format the result as an ACE thermal (`mt = 0`/`itype` thermal) file for transport codes. |

THERMR is the module that actually processes the tabulated S(α,β), so
it is where the two patches below matter.

One unit convention trips people up when reading results back: NJOY's
tape34-style cross-section output reports incident neutron energies in
MeV, not eV, while the deck and IRMA's own grids work in eV or units of
`kT`. Keep the conversion in mind when you compare a processed
inelastic cross section against an IRMA-internal curve.

## Stock-NJOY THERMR mangles coherent mode-2 tapes (`cliq` bug)

A coherent `inelastic_mode = 2` tape (graphite and other strongly
coherent crystals) run through an unpatched NJOY2016 THERMR produces
either ~`1e91`-barn nonsense in the inelastic cross section above
roughly `0.27 eV`, or a multi-hour stall, regardless of how the beta
grid is spaced. The bug is in stock `thermr.f90`, not in the IRMA
tape: THERMR trips on the shape of a coherent S(α,β), so emitting a
uniform energy grid does not cure it. It is important to note that a
correct tape is no protection here; apply the one-line two-axis guard
fix below before trusting a coherent mode-2 tape through NJOY. Most
materials, and all `inelastic_mode = 0/1` tapes, are unaffected.

### Mechanism

THERMR has a small-α liquid extrapolation it uses below the table's first
α point (forward scattering, `μ → 1`):

```fortran
s = sab(1,1) + log(alpha(1)/a)/2 - cliq*b**2/a
cliq = (sab(1,1) - sab(1,2)) * alpha(1) / beta(2)**2
```

The activation guard tests for decay along alpha only:

```fortran
if (sab(1,1).gt.sab(2,1)) then       ! cited at two sites in thermr.f90
```

but the `cliq` formula's sign requires decay along beta as well. A
coherent mode-2 S(α,β) has a `β = 0` row that decays in α (acoustic
intensity concentrated in the first low-`Q` bin) yet *rises* in β at
`α₁`. That drives `cliq` negative, so `-cliq*b**2/a` becomes
`+|cliq|*b²/a` and `exp(+|cliq|*b²/a)` explodes to `~1e150..1e354`:
the `~1e91`-barn tape34 garbage. Liquids satisfy both decays, which is
why upstream never saw it.

### Symptom table

| Tape | Stock NJOY2016 THERMR | After the two-axis patch |
| --- | --- | --- |
| `inelastic_mode = 0` (classic) | clean | clean (byte-identical) |
| `inelastic_mode = 1` (incoherent approx.) | clean | clean (byte-identical) |
| `inelastic_mode = 2`, weakly coherent | usually clean | clean |
| `inelastic_mode = 2`, strongly coherent (e.g. graphite) | ~`1e91` b above ~`0.27 eV`, or a multi-hour stall | clean (mode-1 tape34 byte-identical pre/post patch) |

### The one-line two-axis guard fix

Require decay along both axes at each `cliq` activation site in
`src/thermr.f90`:

```fortran
! before
if (sab(1,1).gt.sab(2,1)) then
! after
if (sab(1,1).gt.sab(2,1).and.sab(1,1).gt.sab(1,2)) then
```

Apply it at both `cliq` sites (the `sig()` extrapolation path and its
duplicate), then rebuild NJOY. The patch is regression-clean: an
`inelastic_mode = 0/1` tape34 is byte-identical before and after, and a
patched THERMR processes a coherent mode-2 graphite tape (auto grid
included) straight through to a sane cross section. The defect is
reported upstream, with a synthetic reproducer, as
[njoy/NJOY2016#399](https://github.com/njoy/NJOY2016/issues/399).

Grid spacing is a non-issue here. Earlier guidance suggested forcing a
uniform ~1 meV beta grid for NJOY, but the cliq blowup fires on the
shape of a coherent S(α,β), and a uniform grid preserves the shape, so
it does not prevent the failure. With the patched THERMR you can hand
NJOY the standard IRMA automatic grid directly. (A uniform grid only
ever sidestepped a separate, older log-tail stall; it was never the fix
for the cliq garbage.)

Transfers that fall beyond the tabulated S(α,β) are handled downstream
by THERMR's short-collision-time (SCT) extension, which is driven by the
tape's effective temperature; you do not need to extend the IRMA grid to
cover them.

## THERMR ignores the MF7/MT4 interpolation flag (lin-lin tapes)

The second defect is quieter. THERMR reads the MF7/MT4 interpolation
flag but does not honor lin-lin (INT=2) interpolation through the path
that reconstructs the inelastic cross section; everything is treated as
log-lin there. Reported upstream as
[njoy/NJOY2016#410](https://github.com/njoy/NJOY2016/issues/410), with
the patch attached. For sharp coherent mode-2 tapes written with IRMA's
`iint` option this matters: log-lin interpolation floors the deep
coherent near-zeros of the table and biases the cross section near the
inelastic minimum low (for graphite, lin-lin processing raises the
minimum by about 9% at 296 K and about 6% at 500 K; the two schemes
converge on a sufficiently dense grid). The validation record therefore
uses a second local THERMR patch that selects the interpolation rule
from the tape; it leaves processing of log-lin tapes byte-identical.
Smooth incoherent mode-0/1 tables have no such near-zeros, and log-lin
remains adequate for them.

Neither the cliq fix nor the interpolation fix is part of the released
NJOY2016 distribution; the corrected THERMR built from both patches is
what the [validation record](validation/methodology.md) means by
"corrected NJOY THERMR".

## Known deliberate divergences from NJOY's LEAPR

IRMA reproduces NJOY2016 LEAPR byte-for-byte on the expected validation
set, but eight NJOY behaviors are deliberately handled differently. Seven
are bugs, placeholders, or numerical hazards in NJOY itself that IRMA does
*not* reproduce; the eighth is an NJOY quirk that IRMA deliberately *does*
reproduce for byte parity. Each carries a `DELIBERATE NJOY DIVERGENCE`
comment at the code site (ten sites in all, because the last entry is
tagged in each of its three implementations).

* **Discrete-oscillator delta lines (`twt = 0` decks).** NJOY's `discre`
  reuses its `idone` flag for both the line loop and the inner
  grid-search (`leapr.f90:1549-1585`), so it adds only the *first*
  in-range negative delta line and then stops. IRMA convolves **every**
  in-range line, the physically complete treatment. Consequence: a
  `twt = 0` deck with discrete oscillators (Einstein-solid hydrides and
  similar) will not match an NJOY tape at the delta-line betas; expect
  large local ratios there. Decks with `twt > 0` (all the expected decks)
  never enter this path and remain byte-faithful.
* **`iel = 5` (lead) coherent elastic.** NJOY ships `pb4 = 1.0` barn (a
  placeholder, not the physical `sigma_coh(Pb) = 11.115` b), so NJOY
  iel=5 tapes carry a coherent-elastic component ~11.1x too small. IRMA
  uses the physical constant (`irma/core/crystal.py`). An IRMA Pb tape
  therefore deliberately disagrees with NJOY's by that factor.
* **SCT effective temperature accumulated across the α grid (`discre`).**
  NJOY initialises the short-collision-time effective-temperature ratio
  once before the α loop (`leapr.f90:1401`) and adds the oscillator
  contributions for every α without ever resetting it
  (`leapr.f90:1494`), so its `T_eff/T` grows linearly with the α
  *index*: in a two-oscillator probe, by the last α of a 40-point grid
  the SCT tails change support entirely (313 of 512 tape lines differ,
  by up to 100% on tail values), and merely adding α points changes the
  answer. That is a latent NJOY bug: `T_eff/T` is a property of the
  phonon spectrum, not of the grid position. IRMA resets the ratio for
  every α. Affects only SCT tails reached through discrete oscillators;
  the expected decks have no SCT-tail exposure and match NJOY to ~1e-4
  either way.
* **ln-S sentinel for zero-S points (ENDF writer).** NJOY's `endout`
  writes the ln-S sentinel −999 in the first-temperature TAB1 and the
  `isym=1/2/3` LIST branches, but writes **0** in the `isym=0`
  additional-temperature LIST branch (`leapr.f90:3482`). THERMR stores
  `ilog` values verbatim as ln S (`thermr.f90:1754`), so NJOY's 0
  sentinel resurrects those zero-S points as S = e⁰ = 1 downstream, a
  demonstrable NJOY bug. IRMA writes −999 at every temperature, which
  keeps S ≈ 0 (safely below THERMR's `sabflg = −225` floor). Affects the
  additional-temperature blocks of `isym=0`, log-stored tapes.
* **Print-flag-dependent SCT start index (phonon expansion).** The
  monotonicity clamp on the SCT start index sits inside
  `if (iprint.ne.0)` in NJOY (`leapr.f90:566-571`), so NJOY's *physics*
  output depends on the print flag. IRMA applies the clamp
  unconditionally. All NJOY reference tapes in the expected set were
  generated with `iprint=1` (clamp active), and IRMA reproduces them to
  ~1e-4; an `iprint=0` NJOY run can differ where the SCT range begins.
* **Translational self-term clamp (`trans`).** NJOY clamps only the
  convolution part of the translational S(α,β) (`leapr.f90:1041`) and
  writes any self-term sum down to `smin = 1e-75` via `endout`. IRMA
  re-clamps the combined convolution-plus-self term below 1e-30, so a
  self term landing in (1e-75, 1e-30), reached only at high α where the
  Debye-Waller weight `α·f0 ≳ 70`, is zeroed where NJOY keeps it.
  All expected decks reproduce their NJOY references to ~1e-4 with the
  clamp active.
* **Overflow guard in the discrete-oscillator Bessel factors
  (`bfact`).** NJOY has no guard on exponential arguments above 709:
  `exp(709+)` overflows to Inf, and Inf times an underflowed Bessel
  coefficient of 0 puts NaN in the table. IRMA zeroes those terms; since
  both codes already zero the coefficients below 1e-30, any term the
  guard suppresses is numerically meaningless for a sum-rule-bounded
  S(α,β). The condition only arises for high-energy oscillators at
  cryogenic temperatures (e.g. a 0.2 eV oscillator below ~30 K at
  expansion order ≳ 18).
* **SCT prefactor missing its square root (`sint`), reproduced rather
  than fixed.** The SCT tail evaluated when a discrete-oscillator or
  rotational shift pushes |β| past the tabulated range divides by
  `4π·wt·α·T̄`, although the Gaussian normalization, and every sibling
  SCT/free-gas expression in the same module, divides by the *square
  root* of that quantity. The missing square root is NJOY's own
  (`leapr.f90:1892`), and IRMA reproduces it deliberately for byte
  parity with the reference tapes. It affects only the SCT tail reached
  from `discre` and `coldh`. Tagged at all three implementations
  (`sint`, `sint_vec`, `_sint_batch_exact`), which are kept in lockstep.

The first two have been reported upstream:
[njoy/NJOY2016#402](https://github.com/njoy/NJOY2016/issues/402) (`discre`
delta lines) and
[njoy/NJOY2016#403](https://github.com/njoy/NJOY2016/issues/403) (lead
`pb4`).

## Comparing S(α,β) tables with `thermr_mimic`

`thermr_mimic` is a validated reimplementation of THERMR's
tabulated-`S(α, β)` inelastic-cross-section kernel. It is validation
tooling, not part of the `irma` package: its purpose is to let you
compare two S(α, β) tables (for example an IRMA mode-2 table against
one derived from OCLIMAX) at the inelastic-cross-section level without
launching a full NJOY job. On the finest grid both programs can process,
it agrees with corrected NJOY THERMR to a median of 0.05% and at worst
0.8% (see the [validation methodology](validation/methodology.md)).

Use `thermr_mimic` when you want a fast comparison of two tabulated
S(α, β) tables through the same cross-section kernel. Reach for real
THERMR when you need the parts of the processing that the mimic
deliberately does not reproduce: the short-collision-time extrapolation
beyond the tabulated grid, the full secondary energy–angle
reconstruction, and the final ACE-format output.

It is important to note that a table converted from an external code
(for example an OCLIMAX `S(Q, E)` map) can agree with the mimic yet
diverge in real THERMR through the short-collision-time extrapolation.
When you process such a table through NJOY, work from `thermr_mimic`
for the comparison on the table's own grid, and from a composite or
uniformly resampled tape for the actual THERMR run.

## THERMR `calcem` cosine-clamping warnings

When THERMR's `calcem` routine reconstructs angular distributions it can
emit warnings about clamping scattering cosines back into the physical
`[-1, 1]` range. In our runs these clamp messages have been harmless:
they appear to be a consequence of the adaptive reconstruction at grid
edges rather than a defect in the IRMA tape. That is an observation
from our runs, not a guarantee: if the messages appear on a tape that
also fails downstream checks, investigate rather than assume they are
benign.

## See also

- [Automatic grids](grids.md) — the structure of IRMA's automatic
  alpha/beta grids and how to override them.
- [Scattering modes](modes.md) — what `inelastic_mode = 0/1/2` produce,
  including the coherent `mode-2` structure referenced here.
- [Input deck reference](input-reference.md) — the full Card 7–9
  grid-card specification.
