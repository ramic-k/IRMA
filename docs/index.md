# IRMA

IRMA, **(In)elastic Representation of Materials As S(α,β) evaluations**, is a
Python code for computing the thermal scattering law S(α,β) of a material,
tabulated in the dimensionless momentum transfer α and energy transfer β
(β = E/kT; the [theory page](theory.md) defines the conventions).
One phonon calculation feeds three outputs that usually require three separate tool
chains: an ENDF-6 File 7 evaluation, instrument-resolved
inelastic neutron scattering spectra and S(Q,E) maps through the
`irma.spectra` forward model, and per-temperature scattering kernels for the
companion NCrystal plugin in this repository; the plugin carries the same
physics into Monte Carlo transport. The evaluation side is a modern reimplementation and generalization
of the LEAPR module of NJOY2016: IRMA reads the same NJOY free-format,
LEAPR-style input files, reimplements the LEAPR algorithms (the classic
kernels), and extends them
with anisotropic (directional) Debye-Waller physics and an exact coherent
one-phonon inelastic treatment that works for arbitrary crystals. The phonon
calculation itself can come from the input file, from phonopy, or from a bare crystal
structure via the `irma mlip` front end, which builds one with a pretrained
machine-learned interatomic potential. A graphical interface and a
command-line tool drive every path. This page orients you; the manual
sections linked at the bottom cover each workflow in depth.

## What IRMA does

IRMA evaluates the (in)elastic response of a material as S(α,β) and
writes it into MF7 thermal scattering data. The code is built in layers
on the LEAPR heritage:

- **Classic LEAPR physics, reproduced.** The phonon-expansion kernels (continuous
  spectrum, convolution, discrete oscillators, translational/diffusion modes,
  cold hydrogen/deuterium, Sköld) are reimplemented from NJOY2016's `leapr.f90`
  and validated against reference tapes (see [Capabilities](#headline-capabilities)).
  The mode-2 coherent one-phonon `S(α,β)` is cross-validated against Euphonic,
  an independent code run on the same phonon calculation, to shared-domain
  integral ratios of 1.00001 (graphite), 1.0002 (beryllium), and 0.9998 (BeO).
- **Generalized elastic and inelastic for any crystal.** Coherent-elastic Bragg
  edges are computed directly from a user-supplied crystal structure, and
  directional inelastic scattering from a phonopy calculation, lifting LEAPR's
  built-in-material and cubic/isotropic restrictions.
- **From evaluation to experiment.** The `irma.spectra` forward model predicts
  what an instrument actually measures: instrument-resolved INS spectra and 2-D
  S(Q,E) powder maps for VISION, generic indirect, and direct (chopper)
  geometries, driven by a phonopy calculation (modes 1/2) or directly by a phonon
  DOS ([mode 0](spectra-mode0.md)).
- **A phonon calculation without a DFT campaign.** The [`irma mlip` front
  end](mlip.md) builds the phonon calculation itself from a pretrained
  machine-learned interatomic potential (MLIP) for any ASE-readable crystal
  (nine supported potentials, per-potential environments) and emits
  ready-to-run ENDF, spectra, and NCrystal-exporter inputs from the
  resulting bundle.

One principal scatterer is evaluated per run. Polyatomic crystals (for example
BeO) are handled by keeping the full crystal in the phonon calculation and
running each principal species in a separate input file.

IRMA was previously named **THAWNE**; the project was renamed in June 2026.
The Python package is `irma` and the console entry points are `irma` and
`irma-gui`, so older imports of `thawne` should be updated.

## Headline capabilities

| Capability | What it gives you |
|------------|-------------------|
| Classic LEAPR kernels | Continuous spectrum, discrete oscillators, free-gas/diffusion translation, cold ortho/para hydrogen and deuterium, Sköld pair correlation, mixed (secondary) moderators, multi-temperature input files with the LEAPR shared-spectrum (negative-temperature) convention |
| Validated against NJOY | Reproduces published ENDF/B-VIII.1 reference tapes to 7e-5 (graphite, Fe, Al, polyethylene) and freshly generated NJOY2016 tapes exactly for liquid methane, ortho-/para-hydrogen, ortho-/para-deuterium, and beryllium oxide (covering the translational, discrete-oscillator, cold-hydrogen/deuterium, and Sköld kernels) |
| Generalized coherent elastic | Bragg edges computed from the user's crystal structure for **any** material via `iel=10`, with optional ENDF-102 §7.2.2 Bragg-edge grouping for dense high-energy edge structure |
| Noncubic inelastic, phonopy-backed | `inelastic_mode=1` (directional incoherent-approximation one-phonon + multiphonon) and `inelastic_mode=2` (**exact** coherent + incoherent one-phonon + incoherent multiphonon) driven by a phonopy calculation |
| Automatic, physics-aware grids | Beta grid in three regions (log lower tail, linear phonon region, log upper tail); alpha grid linear in momentum transfer Q to resolve the thermal upscatter windows |
| Neutron scattering forward model | Instrument-resolved INS spectra and 2-D S(Q,E) powder maps (`irma spectra` CLI and the GUI's **Neutron Scattering Experiments** tab): VISION, generic indirect, and direct geometries; an automatic chopper resolution model validated against PyChop for eight direct-geometry instruments; an elastic line computed directly, with no ENDF file involved; and a DOS-driven [mode 0](spectra-mode0.md) that needs no eigenvectors |
| MLIP phonon calculations | [`irma mlip`](mlip.md) builds a phonon calculation from a pretrained machine-learned interatomic potential for any ASE-readable crystal (relax, displace, force constants, quick-look DOS) and emits ready-to-run inputs for the tape generator, the forward model, and the NCrystal exporter |
| NCrystal export | [`irma ncrystal`](ncrystal-plugin.md) writes the mode-2 S(α,β), the elastic data, and a record of the IRMA version and run settings as a per-temperature material data file for the companion NCrystal plugin, taking the same physics into Monte Carlo transport (McStas, OpenMC, ...) |
| Crystalline extinction | Opt-in [extinction correction](extinction.md) (Sabine and Becker-Coppens models, ported from CrysXT) for the coherent-elastic Bragg edges of specimen-specific `iel=10` evaluations |
| GUI and CLI | A tkinter graphical interface for interactive configuration, plus a command-line tool for scripting and batch processing |
| ENDF-6 MF7 output | MF7/MT2 (elastic) and MF7/MT4 (inelastic) compatible with standard transport processing (NJOY's THERMR and ACER modules) |

The elastic and inelastic treatments are selected by three small integers in the
input file: `iel` (Card 5), and `elastic_mode` / `inelastic_mode` (Card 6b, only
when `iel=10`):

| Selector | Card | Values |
|----------|------|--------|
| `iel` | 5 | `0` none/incoherent · `1`–`6` built-in coherent elastic (graphite, Be, BeO, Al, Pb, Fe) · `10` generalized (any crystal), recommended for new evaluations |
| `elastic_mode` | 6b | `1` SEF (single-channel elastic format; one elastic component per atom) · `2` MEF (mixed elastic, LTHR=3; needs downstream support) |
| `inelastic_mode` | 6b | `0` legacy cubic (isotropic DW from the input file's tabulated DOS) · `1` directional incoherent approximation · `2` exact coherent one-phonon (+ incoherent-approximation multiphonon) |

Phonopy is needed only for the noncubic inelastic modes: `inelastic_mode=1`
and `inelastic_mode=2` require the optional `phonopy` package plus a phonopy
calculation (`phonopy.yaml` and force constants) supplied on Card 6f. Neither
the classic kernels (`iel=0`–`6`) nor `iel=10` with `inelastic_mode=0` needs
phonopy. The noncubic modes also do not read the legacy continuous-DOS,
translational, or oscillator detail cards, and they reject `ncold`, `nsk`,
and a secondary scatterer.

## A first look

Install and run a committed, validated classic-kernel input file from the
command line (the commands assume a cloned repository; see
[Installation](installation.md)):

```bash
pip install -e .              # core (the classic kernels)
pip install -e ".[phonopy]"   # + the noncubic inelastic modes (inelastic_mode=1/2)
pip install -e ".[spectra]"   # + the neutron scattering forward model (irma spectra)
pip install -e ".[mlip]"      # + the MLIP phonon front end (irma mlip)

python -m irma input_file output_file
```

Or launch the graphical interface:

```bash
irma-gui            # or: python -m irma --gui
```

A minimal generalized-elastic input file selects `iel=10` and adds Cards 6b–6g;
here `inelastic_mode=2` requests the exact one-phonon S(α,β) from a phonopy
calculation:

```text
6b   1 1 0 2 /        elastic_mode=1 (SEF), nat=1, nspec=0, inelastic_mode=2
6c   2.464 2.464 6.711 90 90 120 /     lattice constants [Å] and angles [°]
...
6g   10000 1000 1 /    ndir=10000 mpdir=1000 auto_order=1
```

The Card 6g values in the snippet are the recommended production sampling for
the noncubic modes: 10000 one-phonon powder directions, 1000 multiphonon
directions, and auto-sized multiphonon order. This is the sampling the
validation campaign ran, and it is the default in the GUI form and in
`irma mlip emit`. The powder averages are converged well below these counts,
so a faster look at `4000 200 1` is fine for exploration. With `auto_order=1`
the multiphonon order rises until the anisotropic Debye-Waller Poisson sum
converges at the grid's Q_max; a fixed `nphon` that is too small truncates
the table at high Q, and the engine warns when it does.

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

- **[Validation](validation/methodology.md)**: what was compared against what,
  and the material-by-material record for graphite, beryllium, and beryllium
  oxide, including the comparisons with NJOY, Euphonic, OCLIMAX, and measured
  data.
- **[Installation](installation.md)**: requirements (Python ≥ 3.11, numpy ≥ 2.0,
  endf-parserpy ≥ 0.12, < 0.18; phonopy and scipy/PyYAML optional), the
  `[phonopy]`/`[spectra]` extras, and verifying the install.
- **[Quickstart](quickstart.md)**: from a fresh install to a written ENDF
  tape in about five minutes, with the console output explained.
- **[Input file reference](input-reference.md)**: the full Card 1–19 reference,
  including the `iel=10` generalized cards (6b–6g) and the temperature detail block.
- **[Scattering modes](modes.md)**: elastic `iel` options, SEF vs MEF
  (`elastic_mode`), Bragg-edge grouping, and `inelastic_mode` 0/1/2 with the
  phonopy calculation and directional Debye-Waller physics.
- **[Neutron scattering spectra](spectra.md)**: the `irma.spectra` forward
  model: instrument-resolved INS spectra and S(Q,E) maps from a phonopy
  calculation, with an end-to-end graphite example and the full CLI/config
  reference.
- **[DOS-based spectra (mode 0)](spectra-mode0.md)**: the same forward model
  driven by a phonon DOS alone, with no eigenvectors needed; suited to
  hydrogen-rich and incoherent materials.
- **[Automatic grids](grids.md)**: the beta and alpha (linear-in-Q) grid
  generators and how to tune them.
- **[MLIP phonon calculations](mlip.md)**: building phonon calculations from
  pretrained machine-learned interatomic potentials with `irma mlip`, and
  [end-to-end examples](mlip-examples.md) from crystals to polymers.
- **[NCrystal data exporter](ncrystal-plugin.md)**: exporting the mode-2
  physics as material data files for Monte Carlo transport.
- **[The GUI](gui.md)**: interactive configuration, importing existing LEAPR
  input files, and saving/loading JSON configurations.
- **[NJOY interoperability](njoy.md)**: processing IRMA MF7 tapes through
  THERMR/ACER and the stock-NJOY THERMR `cliq` patch for coherent mode-2 tapes.
