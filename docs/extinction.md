# Crystalline extinction

IRMA can apply an **optional crystalline-extinction correction** to the
coherent-elastic Bragg comb (MF7/MT2) of an `iel=10` evaluation. It is **off by
default**. When disabled the tape is the ideal-crystal comb, byte-identical to a
run without the feature.

## What extinction is

In an ideal (kinematic) crystal the coherent-elastic cross section is the Bragg
sum over reciprocal-lattice planes. A **real** crystallite diffracts so
efficiently that, once a beam is strongly Bragg-scattered, it is depleted before
it can scatter again, so the measured Bragg-peak intensity is **lower** than the
kinematic value. This reduction is *extinction*.

IRMA multiplies each plane's kinematic intensity by an **extinction factor**
`y(x, θ) ∈ (0, 1]`:

$$
\sigma_{\rm coh}^{\rm el}(E) \;=\; \frac{1}{E}\sum_{2d \ge \lambda} \delta_{hkl}\, y_{hkl}(\lambda)
$$

There are two channels:

- **Primary extinction**: multiple scattering *within one perfect mosaic block*.
  Driven by the crystallite size `l`.
- **Secondary extinction**: beam depletion *from block to block* across the
  sample. Driven by the mosaic spread `g` and the grain size `L`.

The dimensionless argument grows with wavelength (`x ∝ λ²` and higher), so
extinction is **strongest at long wavelength / low energy and dies out above
~0.1 eV**. Above that cutoff `σ_ext = σ_kin`, so the high-energy comb is
unchanged.

!!! important "Extinction is a SAMPLE property, not a material property"
    `l`, `g`, `L` describe a *particular specimen*. They come from a fit to a
    measured transmission (as in Xu 2025) or from microstructure (EBSD). The same
    material in a different form has different extinction. Do **not** treat an
    extinction-corrected tape as a generic material library.

### Extinction vs. texture

Both reduce some Bragg peaks, but they are different physics. **Extinction**
attenuates reflections in an orientation-independent way and *reduces* the total
coherent-elastic cross section. **Texture** (preferred orientation) *redistributes*
intensity between reflections and conserves the total. IRMA models extinction
only; texture has no place in an orientation-averaged ENDF/TSL evaluation.

## Models

Five models are available (ported from the NCrystal CrysXT plugin):

| Model | Channels | Parameters |
|-------|----------|------------|
| `Sabine_uncorr` | primary + secondary (uncorrelated block) | `l`, (`g`,`L`) |
| `Sabine_corr` | primary + secondary (correlated block) | `l`, (`g`,`L`) |
| `BC_pure` | primary **or** secondary | `l` *or* (`g`,`L`) |
| `BC_mix` | coupled primary **+** secondary | `l`, `g`, `L` |
| `BC_mod` | secondary only (no primary factor) | `l`, `g`, `L` |

- **`BC_pure`** is the one-knob entry point: `BC_pure l=8550` gives pure primary
  extinction from a single crystallite-size parameter.
- **`BC_mix`** and **`BC_mod`** couple the channels and therefore **require `l>0`,
  `g>0` and `L>0`**: the secondary term is parameterised by the block size `l`, so
  it is needed even though `BC_mod` applies no primary factor.
- The Becker-Coppens models take a `recipe`: **`std`** (the default BC2025
  recipe, Kittelmann 2026) or **`cls`** (the original BC1974 closed forms, which
  can be numerically fragile at strong extinction). The Sabine models are analytic
  and ignore `recipe`.
- The tilt/mosaic **distribution** is `Gauss`/`Lorentz`/`Fresnel` for
  Becker-Coppens and `rect`/`tri` for Sabine (default: `Gauss` / `rect`).

## Enabling it: the deck card

Add an optional `extinction` card as the **last card of the `iel=10` elastic
block** (after Cards 6d/6e, or Card 6g for `inelastic_mode=1/2`), before Card 7:

```
extinction <model> l=<Å> g=<rad⁻¹> L=<Å> [dist=<...>] [rec=cls|std] [rmse_tol=<frac>]
```

Example (`examples/tsl/be_iel10_extinction.input`):

```
2.28660 2.28660 3.58330 90.0 90.0 120.0/   $ Card 6c: Be hcp cell
4 9 8.93478 7.79 0.0018 2/                  $ Card 6d
0.33333333 0.66666667 0.75  0.66666667 0.33333333 0.25/
extinction BC_mix l=8550 g=170 L=75750 dist=Gauss rec=std rmse_tol=1e-3 /
150 400 1/                                  $ Card 7
```

Fields:

| Field | Meaning | Default |
|-------|---------|---------|
| `<model>` | one of the five models above (required) | — |
| `l` | crystallite (block) size [Å], primary | `0` (off) |
| `g` | mosaic spread [rad⁻¹], secondary | `0` (off) |
| `L` | grain size [Å], secondary | `0` (off) |
| `dist` | tilt distribution | `Gauss` (BC) / `rect` (Sabine) |
| `rec` | BC recipe `std`/`cls` | `std` |
| `rmse_tol` | tabulation tolerance | `1e-3` |

The card is validated at parse time (model name, `l/g/L ≥ 0`, at least one active
channel, `g>0 & L>0` for `BC_mix`/`BC_mod`, distribution valid for the model). It
works with every `inelastic_mode` (0, 1, 2) and with both elastic formats
(`elastic_mode=1`/`2`), **provided MF7/MT2 actually carries a coherent comb to
correct**. MEF (`elastic_mode=2`) always does: extinction corrects the per-atom
coherent comb and leaves the incoherent-elastic part untouched. SEF
(`elastic_mode=1`) writes only the dominant channel, and two configurations
route MF7/MT2 to the *incoherent*-elastic builder, which never applies
extinction: a single-atom material whose `sigma_coh <= sigma_inc` (Card 6d), or
a polyatomic whose principal scatterer is not the designated-coherent atom. An
`extinction` card on such a deck is **rejected at parse time** ("extinction
would be a silent no-op ...") instead of silently writing an uncorrected tape
whose comments claim an extinction correction — switch to `elastic_mode=2`
(MEF) to keep a coherent channel, or remove the card. The model and parameters
are stamped into the
MF1/MT451 comments so the tape records that it is an extinction-corrected,
sample-specific evaluation.

## In the GUI

The *ENDF Evaluation ▸ Material* tab has a **Crystalline Extinction (Optional)**
section under the `iel=10` coherent-elastic options: an enable toggle (off by
default), the model dropdown, the `l`/`g`/`L`/distribution/recipe/`rmse_tol`
fields, and an **ⓘ help glyph on every control** (hover for a preview, click
for the full text) explaining the parameter and
linking the references. Importing a deck with an `extinction` card populates the
section automatically.

## Tape format (MF7/MT2)

Unlike the ideal step comb, the extinction-corrected `σ(E)` varies *within* a
Bragg interval below the cutoff (the factor `y` depends on wavelength). IRMA
keeps the standard **histogram (INT=1)** cumulative-S form: it reuses the
kinematic comb above the cutoff and splices fine, tolerance-adaptive nodes below
it. This is deliberate: NJOY THERMR (and other processors) read MF7/MT2 as a
**step function regardless of the interpolation flag**, so tabulating for that
step (and labelling it INT=1) is both smaller and more faithful than a lin-lin
table that gets read as a staircase anyway. **No NJOY patch is needed.**

Extinction adds nodes only below ~0.1 eV; a typical run adds a few hundred points
to the comb (e.g. Be: ~2.5k vs ~1.7k). `rmse_tol` trades node count for fidelity
(default `1e-3` ≈ 0.04 % RMSE on the reconstructed cross section).

## Validation

The five models reproduce the CrysXT plugin to **0.000 %** across nine
model/recipe/distribution cases. A frozen CrysXT capture is replayed by the CI
gate `tests/test_extinction_crysxt_expected.py`, so any drift in the port is caught
without needing NCrystal/CrysXT at test time. End-to-end, a real beryllium
evaluation processed by NJOY THERMR reproduces the CrysXT coherent-elastic cross
section to **0.07 % median** (cell matched to the reference structure).

The broader beryllium evaluation (its inelastic and cross-section suites
against OCLIMAX, Euphonic, and NJOY) is documented in
[Validation ▸ Beryllium](validation/beryllium.md), which also discusses the
experimental signature of extinction at the Bragg cutoff.

## Attribution & references

The extinction models and recipes are **ported (not imported)** from the NCrystal
**CrysXT** plugin:

- **ncplugin-CrysXT**: <https://github.com/dddijulio/ncplugin-CrysXT>

Please read these to understand the physics and how to obtain `l/g/L`:

- T. Kittelmann, D. D. DiJulio, S. Xu & J. I. Marquez Damian, *Revisiting
  Becker-Coppens (1974): updated recipes for estimating extinction factors in
  spherical crystallites*, Acta Cryst. (2026) **A82**, 163–178.
  [doi:10.1107/S2053273326001245](https://doi.org/10.1107/S2053273326001245).
  The BC2025 `std` recipes.
- S. Xu et al., *Impact of extinction effects on neutron transmission in solid
  beryllium metal*, J. Appl. Cryst. (2025) **58**, 1957–1966.
  [doi:10.1107/S1600576725007939](https://doi.org/10.1107/S1600576725007939).
  Concept, motivation, and fitting `l/g/L` to a measured transmission.
- P. J. Becker & P. Coppens, Acta Cryst. (1974) **A30**, 129.
- T. M. Sabine, *International Tables for Crystallography* (2006), Vol. C, ch. 6.4.
