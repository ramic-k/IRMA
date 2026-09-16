# Theory background

IRMA evaluates the thermal scattering response of a material as the thermal
scattering law `S(α,β)`, a tabulated function of momentum and energy transfer,
and writes it to ENDF-6 File 7. This page collects the physics behind that
sentence, in just enough depth to read the rest of the manual with confidence.
IRMA is a Python reimplementation and generalization of the LEAPR module of
NJOY2016: the classic phonon-expansion kernels are reproduced faithfully
(validated against published NJOY tapes), and on top of them IRMA adds a
generalized coherent-elastic (Bragg-edge) treatment for any crystal and an
exact one-phonon inelastic treatment for noncubic crystals, driven by phonopy
eigenvectors. If you already know LEAPR, the first half of the page is
familiar ground and the noncubic sections are the new material.

## In plain language

A slow ("thermal") neutron entering a material can scatter from its atoms and,
in doing so, gain or lose energy by creating or absorbing lattice vibrations,
called phonons. `S(α,β)` is the bookkeeping for that exchange: a table that
says how likely a scattering event is for each combination of momentum kick
(`α`) and energy change (`β`). Reactor-physics and neutron-instrument codes
read this table to predict how neutrons slow down in, or scatter from, a
moderator or sample. IRMA's job is to compute that table from a description of
the material's atoms and their vibrations and to write it in the standard
ENDF-6 format other codes expect. The recurring terms, in plain words:

| Term | Plain meaning |
|------|---------------|
| **`S(α,β)`** | the per-material table of scattering probabilities vs. momentum (`α`) and energy (`β`) transfer |
| **`α`** | dimensionless momentum transfer, how hard the neutron is kicked |
| **`β`** | dimensionless energy transfer, how much energy the neutron gains/loses (`β>0` = energy loss) |
| **Phonon** | a quantum of lattice vibration; *one-phonon* = a single vibration created/destroyed, *multiphonon* = several |
| **Elastic** | scattering with no energy change (`β=0`): coherent Bragg edges + the incoherent Debye-Waller line |
| **Inelastic** | scattering that creates/absorbs phonons (`β≠0`) |
| **Debye-Waller factor** | the thermal-vibration damping of scattering, `e^{-2W}` |
| **ENDF-6 / MF7** | the standard nuclear-data file format; File 7 holds thermal scattering data (MT2 elastic, MT4 inelastic) |
| **LEAPR** | the NJOY2016 module IRMA reimplements and generalizes |
| **Tape** | the historical name for an ENDF-format data file, used throughout |
| **THERMR** | the NJOY module that turns the S(α,β) table into cross sections downstream |

## `S(α,β)`: conventions

The double-differential thermal cross section, differential in both scattering
angle and outgoing energy, is carried by `S(α,β)`, written as a function of
the dimensionless momentum and energy transfers

$$
\alpha = \frac{\hbar^2 Q^2}{2 M k_B T}, \qquad
\beta = \frac{E}{k_B T},
$$

where `Q` is the momentum transfer, `E` the energy transfer, `M` the scatterer
mass, `T` the temperature, `ℏ` the reduced Planck constant, and `k_B` the
Boltzmann constant. `α` is non-negative; `β` runs over both signs, with
`β > 0` for energy loss (down-scatter, the neutron loses energy) and `β < 0`
for energy gain (up-scatter).

Two conventions for `S(α,β)` appear in the literature and in ENDF tapes. The
symmetric convention obeys detailed balance explicitly (the thermodynamic
relation that fixes the ratio of up-scatter to down-scatter at a given
temperature) and is even in `β`. The asymmetric (sometimes "script-S")
convention folds the `exp(−β/2)` Boltzmann factor in, so the two sides of `β`
differ. IRMA's noncubic engine builds the physical, energy-loss (down-scatter)
side and stores a downscatter-side asymmetric table,

$$
S_\text{asym}(\alpha, |\beta|) = \frac{4\pi\, k_B T}{\sigma_b}\, S(Q, E),
$$

with `β = E/kT`, `α` as above, and `σ_b` the bound scattering cross section of
the scatterer. The factor `4π` converts the internal per-steradian
differential `S(Q,E)` into the angle-integrated normalization of `S(α,β)`.
The programs that process the tape recover the up-scatter side through
detailed balance; Card 4 `isabt` (cards are the numbered records of the
input file; see the [input file reference](input-reference.md)) selects
whether `S(α,−β)` is also written.

The grids themselves may be supplied either at the actual temperature or at
the LEAPR reference temperature `T_0 = 0.0253 eV` (≈ 293.6 K), chosen by
Card 7 `lat`. With `lat = 1` the α and β grids are interpreted as given at
`T_0`, and the kernels rescale every grid value internally by `T_0/kT` before
evaluation; with `lat = 0` the grids are at the actual temperature. The choice
only changes how the grid numbers map to physical `Q` and `E`. The underlying
physics is identical.

## The classic LEAPR phonon-expansion kernels

For the incoherent inelastic part of `S(α,β)`, and for the incoherent
approximation (every atom treated as if it scattered incoherently, with no
interference between atoms, while carrying the full cross-section strength),
IRMA ports the LEAPR machinery from NJOY2016's `leapr.f90`. The scatterer's
frequency spectrum is reduced to a first-order phonon kernel `T_1(β)` and a
Debye-Waller `λ` (`start`/`contin`), and the full `S(α,β)` is built up by the
harmonic phonon expansion: successive orders are generated by
self-convolution (`convol`), weighted by Poisson factors in `α λ`, and summed.
Beyond the tabulated support, a short-collision-time (SCT) approximation, an
analytic free-gas-like limit that becomes accurate at large energy and
momentum transfer, closes the calculation. The pieces below are each one
optional contribution to the same expansion.

| Kernel | Routine | Role |
|--------|---------|------|
| Continuous spectrum | `contin` | Phonon expansion from the tabulated DOS |
| Translational | `trans` | Free-gas or diffusive center-of-mass motion |
| Discrete oscillators | `discre` | Sharp internal vibrational lines |
| Cold H/D | `coldh` | Rotational structure of ortho/para H₂ and D₂ |
| Sköld | `skold_approx` | Static intermolecular coherence correction |

**Continuous spectrum.** The phonon density of states (DOS) `ρ(ε)`
(Cards 11–12) is the backbone of the calculation. `start` transforms it into
the normalized first-order kernel `T_1(β)` and computes the Debye-Waller `λ`
and effective temperature. `contin` then runs the phonon expansion to order
`nphon` (Card 3): each order is the convolution of `T_1` with the previous
order, scaled by the Poisson weight `e^{−αλ}(αλ)^n/n!`. This is the standard
incoherent harmonic expansion; it captures the smooth multiphonon continuum
of a moderator.

**Translational.** Materials such as liquids have a diffusive or free-gas
center-of-mass mode that is not in the bound phonon spectrum. `trans` builds a
diffusion (`twt c tbeta`, Card 13) or free-gas table (the `stable`
routine) and convolves
it with the bound `S(α,β)` (the part built from the phonon spectrum). `c = 0`
selects the free-gas limit; a nonzero diffusion constant gives the
Egelstaff-Schofield diffusion model.

**Discrete oscillators.** Sharp molecular vibrations (Cards 14–16) are handled
analytically as Einstein oscillators. `discre` builds each line's
Bessel-function weight ladder (`bfact`) and convolves the resulting discrete
delta functions into the continuous `S(α,β)`, so a moderator can carry both a
broad phonon continuum and crisp internal modes.

**Cold hydrogen / deuterium.** Below room temperature the rotational quantum
structure of H₂ and D₂ matters, and the ortho and para nuclear-spin species
scatter differently. `coldh` (Card 5 `ncold`, options 0–4) convolves the
bound `S(α,β)` with the molecular rotational transitions, using spherical
Bessel functions (`sjbes`) and Clebsch-Gordan coefficients (`cn_cg`), and
reproduces the ortho/para-H₂ and -D₂ scattering kernels. The `S(κ)` form
factor (Cards 17–18) feeds the intramolecular structure.

**Sköld.** Intermolecular coherence in liquids is approximated by the Sköld
prescription (Card 5 `nsk = 2`): `skold_approx` rescales `α` by a static
structure factor `S(κ)` (Cards 17–18) and blends the coherent piece in with
weight `cfrac` (Card 19). Of the pair-correlation options only Sköld modifies
the stored table; Vineyard (an alternative pair-correlation prescription)
is accepted for compatibility but does not alter `S(α,β)`.

These kernels are reproduced, not reinvented. They match published
ENDF/B-VIII.1 reference tapes to 7e-5, and freshly generated NJOY2016
tapes exactly for several material families (liquid methane,
ortho-/para-hydrogen and -deuterium, BeO). A handful of deliberate,
documented divergences from NJOY exist (noted inline in
`irma/core/kernels.py`); none affect the validated comparisons. Treat the
classic kernels as a faithful LEAPR; the
[validation methodology](validation/methodology.md) page has the details.

## Generalized coherent elastic (Bragg edges)

For a crystalline solid, coherent elastic scattering produces the familiar
sawtooth of Bragg edges; below the lowest reflection there is no coherent
elastic scattering at all. The legacy LEAPR path (`iel = 1–6`) carries this
for a fixed list of built-in materials. IRMA generalizes it (`iel = 10`) to
any crystal supplied through Cards 6c–6d: the lattice and atom positions are
enumerated, reciprocal-lattice planes are generated down to a `d`-spacing
(interplanar distance) cutoff derived from the maximum energy, and the
coherent structure factor of each plane family is accumulated into edges at
the corresponding threshold energies. Per-edge plane contributions are kept
separate, so the Debye-Waller suppression is applied with the correct
direction for each `(hkl)` family rather than collapsing equal-energy edges
prematurely.

The reciprocal-lattice and Bragg-edge enumeration in `irma/core/crystal.py`
is adapted from NCrystal (Apache-2.0; see `THIRD_PARTY_NOTICES.md`) and
follows the NCrystal elastic-scattering formalism; the licensing notice is
retained as required. The formalism is that of T. Kittelmann et al., "Elastic
neutron scattering models for NCrystal", Computer Physics Communications
**267** (2021) 108082, and K. Ramic, J. I. Marquez Damian, et al.,
"NJOY+NCrystal", NIM-A **1027** (2022) 166227.

### Elastic format: SEF vs. MEF (Card 6b field 1)

Two ENDF-6 elastic formats are available, chosen by `elastic_mode`:

| `elastic_mode` | Format | ENDF `LTHR` | Behavior |
|----------------|--------|-------------|----------|
| 1 | SEF (single-channel elastic format) | 1 or 2 | The full elastic strength is folded into one elastic component per tape (rules below); the standard one-component layout transport codes expect today |
| 2 | MEF (mixed elastic format) | 3 | Coherent and incoherent elastic coexist on one material; needs downstream code support |

SEF was called the current ENDF format (CEF) when the mixed elastic format was
introduced in Ramić et al., *NIM-A* **1027** (2022) 166227; the name was
updated because "current" stopped discriminating once MEF entered the ENDF-6
standard.

SEF folds the full elastic strength (`σ_coh + σ_inc`) into a single elastic
component, chosen as follows (Eqs. 24–26 of Ramic et al., *NIM-A* **1027**
(2022) 166227). For a single atom type the dominant component carries it: if
`σ_coh > σ_inc`, the Bragg edges are written (`LTHR=1`) scaled by
`(σ_coh + σ_inc)/σ_coh` (Eq. 24); otherwise the incoherent Debye-Waller line
is written (`LTHR=2`) with bound cross section `σ_coh + σ_inc` (Eq. 25), and
no coherent elastic appears on the tape at all. For a polyatomic cell the
designated-coherent (DC) atom, the type with the smallest `f/(1−f)·σ_inc`
(`f` is the type's atom fraction in the cell) and hence the smallest
incoherent contribution, carries the coherent elastic (`LTHR=1`, scaled by
`1/f_DC`). A principal scatterer (the atom species the evaluation is written
for) that is not the DC atom gets incoherent elastic (`LTHR=2`) with the DC
atom's incoherent strength redistributed onto it (Eq. 26).

For materials whose high-energy region carries a very dense forest of Bragg
edges, optional edge grouping (ENDF-102 §7.2.2; Card 6b fields 5–6) merges
the steps above a threshold energy into `bins_per_decade` log-uniform bins
with structure-factor-weighted placement. The merge is mass-conserving:
cumulative `S` and the total cross section are preserved. It is off by
default.

## The noncubic inelastic engine

Cubic crystals have an isotropic Debye-Waller factor, so the legacy
scalar-DOS expansion (`inelastic_mode = 0`) is adequate. Anisotropic crystals
do not: the mean-square displacement is a tensor, and treating it as a scalar
mis-suppresses high-`Q` scattering. IRMA's noncubic engine
(`irma/core/noncubic_engine.py`, `inelastic_mode = 1/2`) builds the inelastic
`S(α,β)` directly from phonopy eigenvectors and frequencies and keeps the
directional information throughout.

### Exact one-phonon scattering

From the phonopy mesh (eigenvectors `e`, frequencies `ω`, occupations) the
engine forms three exact harmonic one-phonon terms at fixed `Q`. The coherent
term `S^{(1)}_coh(Q,E)` is the crystal amplitude sum: atom contributions
`b_coh (Q·e) e^{iQ·r}` (with `b_coh` the atom's coherent scattering length
and `r` its position in the cell) are summed over the cell and only then
squared, so interference between sites is exact; IRMA also records the
diagonal (self) and interference pieces separately. This is Squires' `(UV)` term (§3.7), his label for the coherent
one-phonon contribution. The incoherent term `S^{(1)}_inc(Q,E)` is the exact per-atom
self term, Squires' `(UV0)` label (§3.9) for the incoherent one-phonon
contribution, weighted by `σ_inc`. The
incoherent-approximation term `S^{(1)}_approx(Q,E)` is the same self kernel
scaled with the total cross section `σ_tot` instead of `σ_inc`; it is the
practical `n = 1` partner of the multiphonon background.

The mode definitions follow from which of these are combined:

| `inelastic_mode` | One-phonon term | Multiphonon |
|------------------|-----------------|-------------|
| 0 | (legacy cubic, scalar DOS, isotropic Debye-Waller) | classic LEAPR expansion |
| 1 | incoherent-approximation self term (`σ_tot`) | incoherent-approximation |
| 2 | **exact** coherent + incoherent (`σ_coh` + `σ_inc`) | incoherent-approximation |

Each one-phonon line carries a Bose occupation factor (the thermal phonon
population at temperature `T`) and both an emission branch (energy loss,
phonon creation) and an absorption branch (energy gain, phonon annihilation);
the engine deposits both, so detailed balance is built in before the
conversion to `S(α,β)`.

### Powder averaging over directions

A polycrystalline sample averages over all crystal orientations, so the
fixed-`Q` terms must be powder-averaged over the sphere. IRMA samples
directions with a golden-spiral (Fibonacci) quadrature, a set of points that
covers the sphere with equal solid angle per point, one radius per `Q` bin.
This is the rigorous continuous-direction spherical average, equivalent to
Euphonic's `golden` powder method. The direction count is Card 6g `ndir`
(production default `10000`, the validation-campaign sampling). The coherent
one-phonon term is validated against Euphonic, coherent component against
coherent component: the integrals, compared in the symmetric convention,
agree to ratios of 1.00001 (graphite), 1.0002 (beryllium), and 0.9998
(BeO).

### Multiphonon via self-convolution on a work grid

Exact coherent multiphonon scattering is not attempted. Instead a smooth
higher-order background is built in the incoherent approximation, in the
style of Squires §3.10: a per-atom self kernel is normalized to the
directional mean-square displacement, and higher orders are generated by the
harmonic recursion `T_n = (T_1 * T_{n−1}) / n` (self-convolution), with
Debye-Waller and cross-section factors applied after the convolution. The
orders are weighted by a bounded Poisson factor in `2W = Q²a`, where `a` is
the directional mean-square displacement, rather than by the overflow-prone
`(Q²)^n/n!` split.

Which modes exist at all is decided once, by one mask, for every term.
Imaginary modes (negative frequencies) and numerical noise near zero energy
fall under two fixed floors, 1 µeV in general and 0.1 meV at Γ, so a mesh
with acoustic-sum-rule noise at Γ still evaluates. An optional user minimum
phonon energy raises that floor for every term at once, the coherent
one-phonon term included; the removed modes are not replaced by a Debye or
any other continuation and the remaining spectrum is not renormalised, so a
positive value makes the evaluation a deliberately truncated vibrational
model. Because the mean-square displacement weights modes as 1/E², the
Debye-Waller factors respond to such a cutoff far more strongly than the
mode count suggests (on graphite at 296 K a 5 meV cutoff removes 0.13% of
the modes and 29% of the displacement), which is why the run reports both.

Self-convolution needs a uniform signed-energy work grid. A uniform output
grid is used directly. A non-uniform (for example log-tailed) output grid
would blow the work grid up to billions of bins on its finest spacing, so the
engine works on a uniform grid at the output grid's own phonon-region step
(the spacing that repeats across the linear phonon region, so the deck's
phonon subdivision sets the multiphonon resolution and the number of tail
points cannot change it) and rebins the smooth result back in an
integral-conserving way. The multiphonon direction count is
Card 6g `mpdir` (converges by ~50–100; 1000 recommended, cost linear).
Transfers beyond the tabulated grid are covered downstream by THERMR's
short-collision-time extension.

### Directional anisotropic Debye-Waller: why isotropic fails

For each atom the thermal-displacement tensor `U_d` enters `S(α,β)` through
the Debye-Waller exponent: along a unit direction `\hat u`,

$$
2W = Q^2\, (\hat u \cdot U_d \cdot \hat u),
$$

the directional mean-square displacement, evaluated per powder direction
before averaging, *not* the orientation-averaged scalar `\mathrm{Tr}(U_d)/3`.
This distinction is the whole point of the noncubic modes.

Replacing the tensor with its trace-averaged
scalar applies one suppression to every direction, and for a strongly
anisotropic crystal the exact directional factor and the isotropic one
diverge at high `Q`. In graphite (`W_c/W_ab ≈ 6.6`, the ratio of
out-of-plane to in-plane Debye-Waller exponents) the isotropically averaged
one-phonon `S` is suppressed by a factor of about 2 at `Q = 20 1/Å` and about
4×10⁶ at `Q = 50 1/Å` relative to IRMA's exact directional powder average,
for the constant-energy cut at E ≈ 2 meV (the figure below). This is not a
numerical artifact: IRMA's own `inelastic_mode = 0` (isotropic Debye-Waller)
reproduces the isotropic roll-off; the Debye-Waller treatment is the only
difference between the two runs. Use
`inelastic_mode = 2` for anisotropic crystals.

![Directional vs isotropic Debye-Waller attenuation in the graphite one-phonon
S at E ≈ 2 meV: the computed curves on top, isotropic-to-directional ratios below](assets/validation/graphite/fig_graphite_dw_directional.png)

## References

- **LEAPR / classic kernels**: R. E. MacFarlane, D. W. Muir, R. M. Boicourt,
  A. C. Kahler III, "The NJOY Nuclear Data Processing System, Version 2016",
  LA-UR-17-20093 (2016). See the LEAPR chapter for the phonon-expansion,
  translational, oscillator, cold-hydrogen, and Sköld formalism IRMA ports.
- **Bragg-edge / coherent-elastic format**: ENDF-102, *Data Formats and
  Procedures for the Evaluated Nuclear Data Files*, §7.2.2 (thermal coherent
  elastic, edge grouping).
- **Generalized elastic algorithm**: T. Kittelmann et al., *Comput. Phys.
  Commun.* **267** (2021) 108082; K. Ramic et al., *NIM-A* **1027** (2022)
  166227.
- **Phonopy**: A. Togo, "First-principles Phonon Calculations with
  Phonopy and Phono3py", *J. Phys. Soc. Jpn.* **92** (2023) 012001.
- **One-phonon scattering theory**: G. L. Squires, *Introduction to the Theory
  of Thermal Neutron Scattering* (§3.7, §3.9, §3.10).
