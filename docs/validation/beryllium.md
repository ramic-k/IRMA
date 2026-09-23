# Beryllium validation

Beryllium metal is the natural companion benchmark to graphite: a coherent
crystalline scatterer with a published ENDF/B-VIII.1 evaluation, a
structure-dependent (+Sd) variant, and a measured cold-neutron total cross
section to overlay. Unlike graphite it is only weakly anisotropic, which
makes it the control case for the Debye-Waller argument: where graphite's
$W_c/W_{ab} \approx 6.6$ makes the first-order approximation fail by orders
of magnitude at high momentum transfer, beryllium's near-isotropic
displacement tensor lets the isotropic and directional treatments stay
together. The beryllium phonon calculation is a representative VASP/PAW PBE
calculation (4×4×3 supercell, finite-displacement force constants in
phonopy) from the same workflow as the graphite calculation; it is a test
case for the anisotropic methods, and, like every phonon calculation in
this record, it is parameter-free, fitted to nothing, so the results on
this page are method demonstrations rather than
best-fit benchmarks. All comparisons are at 296 K.

Throughout this page, **IRMA mode 2** is the thermal
scattering law S(α,β) computed from the phonopy calculation with the
exact coherent one-phonon term, **IRMA
mode 1** is its incoherent-approximation counterpart, and **Euphonic
n = 1** is the independent coherent one-phonon reference on the same
phonon calculation. **OCLIMAX MAXO=1/100** are OCLIMAX runs truncated at
multiphonon order 1 or 100; at MAXO=1 both the released code and the
unreleased full-tensor Debye-Waller build appear, as for graphite.
**ENDF/B-VIII.1 beryllium-metal** and **Be+Sd** are the released
evaluations, built from phonon calculations different from the one used
here.

---

## The coherent one-phonon term against Euphonic and OCLIMAX

The classic kernels are verified on the reference set of the
[methodology page](methodology.md) (graphite, iron, aluminum,
polyethylene, and the fresh-tape materials); beryllium enters at the
directional rungs of the ladder. The comparison is set up exactly as for
[graphite](graphite.md#the-coherent-one-phonon-term-against-euphonic-and-oclimax):
the coherent part of the n = 1 term on the same phonon calculation, OCLIMAX
keeping only the coherent part by zeroing the incoherent cross sections in
its material file, in
both the released and full-tensor Debye-Waller variants, and Euphonic
evaluated at the IRMA (α, β) grid, with no regridding or broadening.

![Coherent one-phonon isolation for beryllium at 296 K](../assets/validation/be/fig_be_n1_coh_cuts.png)

*Fixed-energy cuts through the coherent one-phonon (n = 1) S(α,β) at
296 K: IRMA mode 2 (coherent component), Euphonic, and OCLIMAX with the
incoherent cross sections set to zero, in the released and full-tensor
variants; each curve at the nearest energy of its own tabulated grid.*

All four coherent curves stay together at every Q. This is the control side
of the Debye-Waller argument: in nearly isotropic beryllium the first-order
approximation costs nothing, the released and full-tensor variants
coincide, and the graphite divergence therefore comes from the anisotropy
rather than from any code's implementation of the coherent term. The
shared-domain integral ratio against Euphonic is 1.0002, with a median
difference of 0.005% in the energy integral J(Q) = ∫ S(Q,E) dE over
Q ≤ 20 Å⁻¹; against OCLIMAX the coherent ratio is 0.99 with either
variant.

### The full one-phonon term and its components

![Fixed-energy cuts through the beryllium n=1 S(α,β) tables at 296 K](../assets/validation/be/fig_be_n1_cuts.png)

*Raw fixed-energy cuts through the beryllium n = 1 S(α,β) tables at
296 K. Euphonic carries only the coherent one-phonon term; IRMA mode 2 and
OCLIMAX MAXO=1 carry both one-phonon components; IRMA mode 1 is the
incoherent counterpart.*

---

## The full S(α,β) against OCLIMAX and the released evaluations

![Beryllium full mode-2 S(α,β) vs OCLIMAX MAXO=100 and the ENDF/B-VIII.1 evaluations](../assets/validation/be/fig_be_full_law.png)

*The full beryllium mode-2 S(α,β) (symmetric form) at 296 K, compared
with OCLIMAX (MAXO=100) and the ENDF/B-VIII.1 beryllium-metal and Be+Sd
evaluations, both built from phonon calculations different from the one
used here.*

Both codes were run to the same multiphonon order, and the shared-window
integrals of the
symmetric tables agree to about 2% (R = 0.98). The agreement is also
uniform: cut by cut, the two stay within about 2% at every Q, since with no strong anisotropy, no residual
accumulates in the high-Q multiphonon tail as it does for graphite. The
Be+Sd file contains narrow spikes near Q ≈ 0.5, 1.8, and 3 Å⁻¹ that
neither the IRMA nor the OCLIMAX calculation contains; as for the graphite
Sd structure, the origin of this structure in the evaluation is not
established here.

---

## Processed cross sections and the measured cold-region total

![Beryllium cross sections from NJOY processing at 296 K](../assets/validation/be/fig_be_xs.png)

*Beryllium cross sections at 296 K, with the NJOY processing and
interpolation conventions of the graphite page: IRMA mode 2 and mode 1
against the ENDF/B-VIII.1 beryllium-metal and Be+Sd evaluations. Panels:
inelastic, coherent elastic, total scattering, and total-plus-absorption
with the measured cold-region total (EXFOR 11204003).*

Both directional modes follow the evaluations through the inelastic,
coherent-elastic, and total panels, and below the Bragg cutoff the
calculated total-plus-absorption passes through the measured cold-region
points. Lin-lin interpolation shifts the beryllium thermal features by
about 2–5%, smaller than for graphite because its coherent near-zeros are
shallower. EXFOR 11204003 is tabulated without per-point uncertainties, so
the measured total is shown as points without error bars.

---

## Crystalline extinction

Beryllium is also the verification case for the opt-in
[crystalline extinction](../extinction.md) correction. Extinction is
the reduction of Bragg intensity in a real crystallite: once a beam is
strongly Bragg-scattered it is depleted before it can scatter again, so
measured peaks fall below the ideal kinematic values (the
[extinction page](../extinction.md) has the physics). The models are
ported from CrysXT, the NCrystal extinction plugin of Kittelmann et
al. (references on the extinction page). The port was verified in two stages. At the kernel
level, IRMA reproduces CrysXT within rounding (0.000%) for nine reference
cases spanning the five extinction models; the frozen cases are regression
references, so the test requires neither NCrystal nor CrysXT at test time.
At the evaluation level, with the unit cell matched to the reference
structure, a beryllium `iel=10` evaluation written by IRMA and run through
NJOY processing agrees with the CrysXT coherent-elastic cross section to a
median of 0.07%; this second comparison tests the processed MF7/MT2
histogram rather than only the analytic kernels.

![Beryllium coherent-elastic cross section with and without extinction](../assets/validation/be/fig_extinction.png)

*Coherent-elastic cross section per atom for beryllium with and without
crystalline extinction. The kinematic curves are from NCrystal and IRMA
mode 0; the extinction-corrected curves use the Becker-Coppens `BC_mix`
model in CrysXT and IRMA, for a specimen with crystallite size 0.855 μm,
mosaic parameter 170 rad⁻¹ (the Becker-Coppens mosaic-distribution
parameter, an inverse angular width), and grain size 7.58 μm. The
parameters demonstrate the models; they were not fitted to the
transmission data shown above.*

For these specimen parameters, extinction reduces the kinematic Bragg
intensity by up to about 20% at the lowest energies; the correction falls
below the tabulation tolerance above approximately 0.1 eV and does not
alter the inelastic component. In the measured cold-region comparison
above, the ideal-crystal calculation jumps to the full kinematic Bragg
pattern at the cutoff, while measured transmission data from real
specimens can rise less sharply; both extinction and instrument
resolution suppress the sharp edge. Fitting the extinction parameters to
a measured transmission (as in Xu et al. 2025, cited on the
[extinction page](../extinction.md)) is exactly the use case this
correction serves.

---

## What the beryllium suite establishes

The directional one-phonon term agrees with Euphonic to a shared-domain
integral ratio of 1.0002 (median J(Q) difference 0.005%), and the coherent
comparison with OCLIMAX gives 0.99 with either Debye-Waller variant. The
full S(α,β) tracks order-matched OCLIMAX to 2%, uniformly in Q, and the
processed cross sections follow the released evaluations and pass through
the measured cold-region total below the Bragg cutoff. The extinction port
is verified to 0.000% at the kernel level and to a median of 0.07% through
the full evaluation chain. Just as important, beryllium completes the
graphite Debye-Waller argument: a nearly isotropic crystal is where the
first-order approximation is supposed to hold, and here it does.
