# IRMA

IRMA, **(In)elastic Representation of Materials As S(α,β) evaluations**, is a
Python code for generating thermal neutron scattering laws. One phonon model
feeds three outputs: an ENDF-6 File 7 evaluation, instrument-resolved
inelastic-neutron-scattering spectra and S(Q,E) maps through the
`irma.spectra` forward model, and per-temperature scattering kernels for the
companion NCrystal plugin, which carries the same physics into Monte Carlo
transport. The evaluation side is a modern reimplementation and generalization
of the LEAPR module of NJOY2016: it reads the same NJOY free-format,
LEAPR-style card decks and reproduces the classic kernels, then extends them
with anisotropic (directional) Debye-Waller physics and an exact coherent
one-phonon inelastic treatment that works for arbitrary crystals. The phonon
model itself can come from a deck, from phonopy, or from a bare crystal
structure via the `irma mlip` front end, which builds it with a pretrained
machine-learned interatomic potential. You drive it all from either a graphical
interface or the command line. This page orients you; the manual sections
linked at the bottom cover each workflow in depth.

## What IRMA does

IRMA evaluates the (in)elastic response of a material as S(α,β) and
serializes it into MF7 thermal scattering data. Two layers sit on top of the
LEAPR heritage:

- **Classic LEAPR physics, reproduced.** The phonon-expansion kernels (continuous
  spectrum, convolution, discrete oscillators, translational/diffusion modes,
  cold hydrogen/deuterium, Sköld) are reimplemented from NJOY2016's `leapr.f90`
  and validated against reference tapes (see [Capabilities](#headline-capabilities)).
- **Generalized elastic and inelastic for any crystal.** Coherent-elastic Bragg
  edges and phonopy-backed directional inelastic scattering are computed directly
  from a user-supplied crystal structure, lifting LEAPR's built-in-material and
  cubic/isotropic restrictions.
- **From evaluation to experiment.** The `irma.spectra` forward model predicts
  what an instrument actually measures: instrument-resolved INS spectra and 2-D
  S(Q,E) powder maps for VISION, generic indirect, and direct (chopper)
  geometries, driven by a phonopy model (modes 1/2) or directly by a phonon
  DOS ([mode 0](spectra-mode0.md)).
- **A phonon model without a DFT campaign.** The [`irma mlip` front
  end](mlip.md) builds the phonon model itself from a pretrained
  machine-learned interatomic potential (MLIP) for any ASE-readable crystal
  (nine supported potentials, per-potential environments) and emits
  ready-to-run ENDF, spectra, and NCrystal-exporter inputs from the
  resulting bundle.

One principal scatterer is evaluated per run. Polyatomic crystals (for example
BeO) are handled by keeping the full crystal in the phonon model and running each
principal species in a separate deck.

!!! note "Renamed from THAWNE"
    IRMA was previously named **THAWNE**; it was renamed in June 2026. The
    Python package is `irma` and the console entry points are `irma` /
    `irma-gui`. Older imports of `thawne` should be updated.

## Headline capabilities

| Capability | What it gives you |
|------------|-------------------|
| Classic LEAPR kernels | Continuous spectrum, discrete oscillators, free-gas/diffusion translation, cold ortho/para hydrogen and deuterium, Sköld pair correlation, mixed (secondary) moderators, multi-temperature decks with the LEAPR shared-spectrum (negative-temperature) convention |
| Validated against NJOY | Reproduces published ENDF/B-VIII.1 reference tapes to 7e-5 or better (graphite, Fe, Al, polyethylene) and freshly generated NJOY2016 tapes exactly for liquid methane, ortho-/para-hydrogen, and beryllium oxide (covering the translational, discrete-oscillator, cold-hydrogen, and Sköld kernels) |
| Generalized coherent elastic | Bragg edges computed from the user's crystal structure for **any** material via `iel=10`, with optional ENDF-102 §7.2.2 Bragg-edge grouping for dense high-energy edge structure |
| Noncubic inelastic, phonopy-backed | `inelastic_mode=1` (directional incoherent-approximation one-phonon + multiphonon) and `inelastic_mode=2` (**exact** coherent + incoherent one-phonon + incoherent multiphonon) driven by a phonopy model |
| Automatic, physics-aware grids | Beta grid in three regions (log lower tail, linear phonon region, log upper tail); alpha grid linear in momentum transfer Q to resolve the thermal upscatter windows |
| Neutron-scattering forward model | Instrument-resolved INS spectra and 2-D S(Q,E) powder maps (`irma spectra` CLI and the GUI's **Neutron Scattering Experiments** tab): VISION, generic indirect, and direct geometries; an automatic chopper resolution model validated against PyChop for eight direct-geometry instruments; a tape-free elastic line; and a DOS-driven [mode 0](spectra-mode0.md) that needs no eigenvectors |
| MLIP phonon models | [`irma mlip`](mlip.md) builds a phonon model from a pretrained machine-learned interatomic potential for any ASE-readable crystal (relax, displace, force constants, quick-look DOS) and emits ready-to-run inputs for the tape generator, the forward model, and the NCrystal exporter |
| NCrystal export | [`irma ncrystal`](ncrystal-plugin.md) writes the mode-2 law, elastic data, and provenance as a per-temperature material-data file for the companion NCrystal plugin, taking the same physics into Monte Carlo transport (McStas, OpenMC, ...) |
| Crystalline extinction | Opt-in [extinction correction](extinction.md) (Sabine and Becker-Coppens models, ported from CrysXT) for the coherent-elastic Bragg comb of sample-specific `iel=10` evaluations |
| GUI and CLI | A tkinter graphical interface for interactive configuration, plus a command-line tool for scripting and batch processing |
| ENDF-6 MF7 output | MF7/MT2 (elastic) and MF7/MT4 (inelastic) compatible with standard transport processing (NJOY THERMR/ACER) |

The elastic and inelastic treatments are selected by three small integers on the
deck: `iel` (Card 5), and `elastic_mode` / `inelastic_mode` (Card 6b, only when
`iel=10`):

| Selector | Card | Values |
|----------|------|--------|
| `iel` | 5 | `0` none/incoherent · `1`–`6` built-in coherent elastic (graphite, Be, BeO, Al, Pb, Fe) · `10` generalized (any crystal), recommended for new evaluations |
| `elastic_mode` | 6b | `1` SEF (single-channel elastic format, dominant-channel) · `2` MEF (mixed elastic, LTHR=3; needs downstream support) |
| `inelastic_mode` | 6b | `0` legacy cubic (isotropic DW from a deck DOS) · `1` directional incoherent approximation · `2` exact coherent one-phonon (+ incoherent-approximation multiphonon) |

!!! warning "Phonopy is needed only for noncubic inelastic modes"
    `inelastic_mode=1` and `inelastic_mode=2` require the optional `phonopy`
    package plus a phonopy model (`phonopy.yaml` + force constants) supplied on
    Card 6f. The classic paths (`iel=0`–`6`, and `iel=10` with
    `inelastic_mode=0`) do not need phonopy. The noncubic modes also do **not**
    read the legacy continuous-DOS / translational / oscillator detail cards,
    and they reject `ncold`/`nsk` and a secondary scatterer.

## A first look

Install and run a committed, validated classic deck from the command line:

```bash
pip install -e .              # core (classic LEAPR paths)
pip install -e ".[phonopy]"   # + noncubic inelastic paths (inelastic_mode=1/2)
pip install -e ".[spectra]"   # + the neutron-scattering forward model (irma spectra)
pip install -e ".[mlip]"      # + the MLIP phonon front end (irma mlip)

python -m irma input_file output_file
```

Or launch the graphical interface:

```bash
irma-gui            # or: python -m irma --gui
```

A minimal generalized-elastic deck selects `iel=10` and adds Cards 6b–6g; here
`inelastic_mode=2` requests the exact one-phonon law from a phonopy model:

```text
6b   1 1 0 2 /        elastic_mode=1 (SEF), nat=1, nspec=0, inelastic_mode=2
6c   2.464 2.464 6.711 90 90 120 /     lattice constants [Å] and angles [°]
...
6g   10000 1000 1 /    ndir=10000 mpdir=1000 auto_order=1
```

!!! note "Sampling for noncubic modes"
    The recommended production sampling is Card 6g `10000 1000 1`: 10000
    coherent powder directions, 1000 multiphonon directions (the powder
    averages are converged well below these counts, so a faster look at
    `4000 200 1` is fine for exploration), and auto-sized multiphonon
    order. This is the validation-campaign sampling, and the GUI form and
    `irma mlip emit` default. The auto-sizer raises the multiphonon
    order to converge the anisotropic Debye-Waller Poisson sum at the grid's
    Q_max; a too-small fixed `nphon` truncates the high-Q rows, and the engine
    warns when it does.

## How well it agrees

The mode-2 exact coherent one-phonon law has been cross-validated against
Euphonic (an independent code on the same phonon model), coherent component
against coherent component: the symmetric-law integrals agree to ratios of
1.00001 (graphite) and 1.0002 (beryllium), and the classic LEAPR kernels
reproduce published NJOY tapes (see above). In the figure's fixed-energy
cuts, **IRMA mode-2** and **Euphonic n = 1** coincide through the
interference structure; the remaining curves show what the approximations
(incoherent-only mode 1, and the isotropic-Debye-Waller treatments) lose.

![Fixed-energy cuts through the graphite one-phonon scattering laws: IRMA mode-2 and Euphonic overlay](assets/validation/graphite/fig_graphite_n1_cuts.png)

The automatic alpha grid (linear in Q) is what keeps the thermal inelastic cross
section converged at a modest column budget:

![Graphite inelastic cross section on the automatic grid](assets/validation/graphite/fig_grid_autogrid.png)

See [Validation](validation/graphite.md) for the full set of comparisons,
including OCLIMAX and the total-cross-section study against Steyerl.

## License and attribution

IRMA is distributed under **BSD 3-Clause** for the project code with
**Apache-2.0** derived portions (SPDX expression `BSD-3-Clause AND
Apache-2.0`; see `LICENSE` and `LICENSES/Apache-2.0.txt`), copyright Oak
Ridge National Laboratory. Three third-party attributions are retained as
their licenses require (full text in `THIRD_PARTY_NOTICES.md`):

- **NJOY2016 LEAPR** (BSD 3-Clause, Los Alamos National Laboratory): IRMA's
  classic kernels are in part derived from `leapr.f90`. IRMA is a clearly marked
  reimplementation, not the version available from LANL.
- **NCrystal** (Apache-2.0, NCrystal developers): the generalized
  coherent-elastic (Bragg-edge) machinery mirrors NCrystal's reciprocal-lattice
  and edge algorithms, translated to Python.
- **ncplugin-CrysXT** (Apache-2.0): the crystalline-extinction models
  (Sabine, Becker-Coppens) in `irma/core/extinction.py` are ported from the
  CrysXT NCrystal plugin (pinned upstream revision recorded in the notices
  file).

Optional phonon workflows use **phonopy** (BSD 3-Clause) and ENDF-6
serialization uses **endf-parserpy** (MIT); neither package's source is included
in this distribution.

## Where to go next

- **[Installation](installation.md)**: requirements (Python ≥ 3.11, numpy ≥ 2.0,
  endf-parserpy ≥ 0.12, < 0.18; phonopy and scipy/PyYAML optional), the
  `[phonopy]`/`[spectra]` extras, and verifying the install.
- **[Quickstart](quickstart.md)**: from a fresh install to a written ENDF
  tape in about five minutes, with the console output explained.
- **[Input deck reference](input-reference.md)**: the full Card 1–19 reference,
  including the `iel=10` generalized cards (6b–6g) and the temperature detail block.
- **[Scattering modes](modes.md)**: elastic `iel` options, SEF vs MEF
  (`elastic_mode`), Bragg-edge grouping, and `inelastic_mode` 0/1/2 with the
  phonopy model and directional Debye-Waller physics.
- **[Neutron-scattering spectra](spectra.md)**: the `irma.spectra` forward
  model: instrument-resolved INS spectra and S(Q,E) maps from a phonopy model,
  with an end-to-end graphite example and the full CLI/config reference.
- **[DOS-based spectra (mode 0)](spectra-mode0.md)**: the same forward model
  driven by a phonon DOS alone, with no eigenvectors needed; ideal for
  hydrogen-rich and incoherent materials.
- **[Automatic grids](grids.md)**: the beta and alpha (linear-in-Q) grid
  generators and how to tune them.
- **[MLIP phonon models](mlip.md)**: building phonon models from pretrained
  machine-learned interatomic potentials with `irma mlip`, and
  [end-to-end examples](mlip-examples.md) from crystals to polymer glasses.
- **[NCrystal data exporter](ncrystal-plugin.md)**: exporting the mode-2
  physics as material-data files for Monte Carlo transport.
- **[The GUI](gui.md)**: interactive configuration, importing existing LEAPR
  decks, and saving/loading JSON configurations.
- **[NJOY interoperability](njoy.md)**: processing IRMA MF7 tapes through
  THERMR/ACER and the stock-NJOY THERMR `cliq` patch for coherent mode-2 tapes.
- **[Validation](validation/graphite.md)**: graphite and other material
  comparisons against NJOY, Euphonic, OCLIMAX, and experiment.
