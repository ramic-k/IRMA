# IRMA

(**I**n)elastic **R**epresentation of **M**aterials **A**s S(α,β) evaluations

IRMA turns one phonon calculation into three outputs that usually require
three separate tool chains: an evaluated nuclear data file, predicted
neutron scattering spectra, and scattering kernels for direct use in
Monte Carlo neutron transport codes. The three outputs are computed from
one description of the material, so the evaluation, the spectra that can
validate it, and the transport kernels that use it carry the same physics.

**Nuclear data.** IRMA writes ENDF-6 File 7 thermal scattering
evaluations on automatically constructed (α, β) grids (α and β are the
dimensionless momentum and energy transfer of the thermal scattering
law S(α,β), with β = E/kT). IRMA reimplements NJOY's LEAPR, the standard
module for generating thermal scattering law data. Its classic kernels
reproduce freshly generated NJOY2016 tapes digit for digit, and published
reference tapes to a maximum relative difference of 7e-5 in significant
S values. (A tape is an ENDF output file, the historical name used
throughout.) Extension cards in the same input file select generalized
modes that add the exact coherent one-phonon term, anisotropic
Debye-Waller tensors, coherent elastic for arbitrary crystals, and a
per-species partition for polyatomic materials. NJOY, AMPX and FUDGE
process the tapes for transport codes.

**Neutron spectroscopy.** The `irma.spectra` forward model projects
the same physics onto an instrument's kinematics and resolution: INS
spectra for VISION and generic indirect geometries, and 2-D S(Q,E)
powder maps for direct-geometry spectrometers, from a phonopy calculation or
straight from a phonon DOS. It can be used to predict a proposed
measurement before beam time; in analysis, it supplies the calculated
single-scattering counterpart of a measured spectrum, from the same
material description the evaluation was built from.

**Monte Carlo transport.** The `irma.ncrystal` exporter writes
per-temperature scattering kernels for the companion NCrystal plugin
that ships in this repository,
so McStas, OpenMC, and other NCrystal-aware codes sample the same
physics. The exported kernels carry the per-site anisotropic
Debye-Waller tensors, keeping directional coherent-elastic physics
that NCrystal's standard scalar treatment does not represent. With the
same physics inside a transport code, a full instrument can be
simulated: IRMA's end-to-end validation ran a custom McStas
implementation of the ARCS spectrometer, assembled from the existing
McVine and McStas models, against measured data.

**From a bare crystal structure.** The `irma mlip` front end builds
the phonon calculation itself: a structure file and a choice of potential
are enough. Nine pretrained machine-learned interatomic potentials are
supported, on a laptop CPU, with no first-principles calculation; an
approximate phonon calculation for a new material costs minutes, not a DFT
campaign, and the front end emits prefilled inputs for all three outputs.
The result is a starting point rather than a finished evaluation:
approximate physics with every parameter exposed for review. A
converged atomistic calculation enters the same way, as a phonopy
calculation, when higher fidelity is needed.

Both a graphical interface and a command-line tool drive every path.
Each run evaluates one principal scatterer (the atom species the
evaluation is written for); polyatomic crystals keep
the full crystal in the phonon calculation and run each principal species
separately.

## Features

- **Phonon expansion** of incoherent inelastic scattering to arbitrary order
- **Generalized coherent elastic** (Bragg edges) for any crystal structure
- **Phonopy-backed inelastic modes** (`inelastic_mode=1/2`): directional
  Debye-Waller factors and S(α,β) computed from a phonopy calculation;
  mode 2 adds the exact coherent one-phonon term. Polyatomic crystals such
  as BeO get one principal-scatterer evaluation per species
- **Automatic α/β grid generation** with user-controllable density
  in logarithmic and linear regions
- **Graphical interface** (tkinter) for easy configuration and execution
- **Command-line interface** for scripting and batch processing
- **ENDF-6 output** (inelastic MF7/MT4, plus elastic MF7/MT2 when an
  elastic option is active) compatible with standard transport codes
- **Neutron scattering forward model** (`irma.spectra`): instrument-resolved
  INS spectra and 2-D S(Q,E) powder maps for VISION, generic indirect, and
  direct geometries, from a phonopy calculation (modes 1/2) or straight from a
  phonon DOS (mode 0, no eigenvectors needed), with an automatic chopper
  resolution model validated against Mantid PyChop for eight direct-geometry
  instruments. Run it as `irma spectra ...` or from the GUI's
  **Neutron Scattering Experiments** tab
- **MLIP phonon front end** (`irma mlip`), generalizing the
  machine-learned interatomic potential (MLIP) workflow of ORNL's
  INSPIRED (Han et al., Comput. Phys. Commun. 304, 109288 (2024)): build
  a phonon calculation for any crystal from a pretrained MLIP (nine backends:
  NequIP, GRACE, ORB, SevenNet, MatterSim, MACE, MACE-OFF, PET-MAD,
  DPA-3; CPU-only, no DFT), then emit prefilled inputs for the ENDF
  generator, the spectra forward model, and the NCrystal exporter.
  The potentials' package stacks conflict with each other, so each
  potential gets its own Python environment, built once with
  `irma mlip env create` and then used by `irma mlip` automatically
- Cold hydrogen/deuterium (ortho/para) support
- Mixed moderator (secondary scatterer) support
- Free-gas and diffusion translational modes
- Discrete oscillators
- Multi-temperature input files, including the LEAPR negative-temperature
  convention (one phonon spectrum reused across temperatures)
- Optional coherent-elastic **Bragg-edge grouping** (ENDF-102 §7.2.2) for
  materials with very dense high-energy edge structure
- **Validated**, with committed, rerunnable harnesses under `tests/`:
  - the classic kernels reproduce published ENDF/B-VIII.1 reference tapes
    to 7e-5 (graphite, Fe, Al, H in CH2) and freshly generated NJOY2016.78
    tapes exactly (liquid CH4, ortho/para-H2, two-pass BeO), covering the
    translational, discrete-oscillator, cold-hydrogen, and mixed-moderator
    kernels; fast-suite minitapes pin cold deuterium and the Sköld
    correction against NJOY byte for byte
  - the mode-2 coherent one-phonon law agrees with Euphonic to
    shared-domain integral ratios of 1.00001 (graphite), 1.0002
    (beryllium), and 0.9998 (BeO), with median per-Q differences of
    0.18% and 0.005% for graphite and beryllium
  - the full S(α,β) matches OCLIMAX to integral ratios of 0.96-0.99
    (graphite, beryllium, BeO)
- **Production defaults**: the GUI form, `irma mlip emit`, and the spectra
  and NCrystal configurations default to the validation campaign's values
  (sampling, mesh density, grids). In a hand-written input file, Card 6g's
  `auto_order` defaults to 0; set it to 1 for production. The committed
  examples also include clearly labeled quick variants for fast
  exploration
- Input validation: errors name the card, the expected and found values,
  and the input line
- The GUI imports and exports input files, and saves and loads the
  spectra and NCrystal configurations as YAML

## Installation

### Requirements

- Python >= 3.11
- numpy >= 2.0
- endf-parserpy >= 0.12, < 0.18  (capped below the next unvalidated minor)
- threadpoolctl >= 3.0  (pins native BLAS/OpenMP threads to one in the
  mode-1/2 worker processes; installed automatically as a core dependency)
- phonopy is optional: required for `iel=10` with `inelastic_mode=1` or `2`,
  and by `irma.spectra` for inelastic modes 1/2 and `dos_source: phonopy`
- scipy + PyYAML are optional (`pip install -e ".[spectra]"`): required only
  by the `irma.spectra` forward model
- the NCrystal exporter (`irma ncrystal`) needs both `[phonopy]` and
  `[spectra]` (it exports modes 1/2 from a YAML config); the C++ plugins
  are built separately (below)
- ase + phonopy + PyYAML are optional (`pip install -e ".[mlip]"`): required
  only by the `irma mlip` phonon front end; the pretrained potentials are
  never installed with IRMA: `irma mlip env create <potential>` builds each
  one its own Python environment (see below)

### Install (released package)

From PyPI, `pip install irma` is the core and the extras are opt-in:

```bash
pip install "irma[phonopy,spectra,mlip]"   # everything
pip install irma                           # core only
```

On conda-forge the package is named **`irma-sqw`**: bioconda already
ships an unrelated `irma` (the CDC influenza assembler), and the two
channels are used together, so the conda package is named for the S(Q,ω)
the engine computes. The import and the command stay `irma`. It installs
all of IRMA, core plus the dependencies of every extra (phonopy, scipy,
PyYAML, ase), so the ENDF generator, the spectra forward model, the
NCrystal exporter, and the MLIP phonon front end all work from one
install:

```bash
conda create -n irma -c conda-forge irma-sqw
conda activate irma
```

With either package manager, the pretrained potentials still get their
own environments (`irma mlip env create <potential>`), and the C++
NCrystal plugins are still built separately (below).

### Install from source

For development, or to run the committed examples, the test suites, and
the plugin builds, clone the repository and install it in editable mode
(`-e`: Python imports the code straight from the clone, so edits take
effect without reinstalling):

```bash
git clone https://github.com/ramic-k/IRMA.git
cd IRMA
pip install -e .              # core (the classic kernels, iel=0-6 and iel=10 mode 0)
pip install -e ".[phonopy]"   # + the phonopy-backed modes (inelastic_mode=1/2)
pip install -e ".[spectra]"   # + the neutron scattering forward model (irma spectra)
pip install -e ".[mlip]"      # + the MLIP phonon front end (irma mlip)
```

The `[mlip]` extra installs only `ase`, `phonopy`, and `pyyaml`. The
pretrained potentials are never installed with IRMA, because their
dependency stacks conflict with each other. Instead,
`irma mlip env create <potential>` builds a separate Python environment
for one potential: it creates a venv, installs that potential's packages
into it, and registers it, after which every `irma mlip` command detects
the registered environment and runs that potential in it automatically.
The pretrained weights themselves are downloaded by the potential's own
package on first use.

### NCrystal plugins (built separately)

`pip install` of IRMA does not build the two C++ NCrystal plugins. Each
is its own scikit-build-core package inside this repository and compiles
against an existing NCrystal installation (NCrystal >= 4.3, with cmake,
ninja, and scikit-build-core available in the same environment, for
example from conda-forge):

```bash
pip install --no-build-isolation ./ncrystal_plugin_IRMA      # samples IRMA's exported kernels
pip install --no-build-isolation ./ncrystal_plugin_ENDFTSL   # reads ENDF thermal scattering files directly
ncrystal-pluginmanager --test IRMA                           # self-test of the installed plugins
ncrystal-pluginmanager --test ENDFTSL
```

NCrystal's plugin manager discovers both automatically once they are
installed; no further configuration is needed. Each plugin's README
(`ncrystal_plugin_IRMA/README.md`, `ncrystal_plugin_ENDFTSL/README.md`)
documents its build environment and reference tests.

### Testing and validation

```bash
python -m pytest              # fast regression suite (no phonopy needed)
```

Three validation harnesses live under `tests/` and are run manually
(not in CI); the manual's
[installation page](https://ramic-k.github.io/IRMA/installation/#developer-install-and-running-the-tests)
describes each:

- `tests/native_LEAPR_NJOY_ENDF_validation/`: the classic kernels against
  reference ENDF tapes.
- `tests/mode2_euphonic_n1_validation/`: the mode-2 exact one-phonon law
  against Euphonic (graphite, Be, BeO).
- `tests/mode0_validation/`: the mode-0 (DOS) spectrum against mode 1 for
  graphite at VISION.

## Documentation

The manual lives at https://ramic-k.github.io/IRMA/: installation, a
quickstart, three end-to-end tutorials (one per output), the GUI tour, the full input-card reference, the theory
behind each mode, the validation record, and the MLIP, spectra, and
NCrystal-plugin guides.

If a passage in the manual reads oddly, or you find an error, please open
an issue; documentation reports are as welcome as code bugs.

## Usage

### Graphical interface

```bash
python -m irma --gui
```

This opens the IRMA GUI where you can:

1. Define your crystal structure (lattice parameters, atom types, positions)
2. Set scattering parameters (principal scatterer, cross sections)
3. Configure alpha/beta grids (automatic or manual)
4. Specify phonon data (from phonopy or manual entry)
5. Run the calculation and monitor progress

File > Export Input File writes the form as an input file, and
File > Import Input File reads an existing IRMA/LEAPR input file
(`.input` or `.leapr`) back into the form.

### Command line

```bash
python -m irma input_file output_file
```

The IRMA input file uses a card-based format derived from the LEAPR module
of NJOY; see [Input file reference](#input-file-reference) below.

### MLIP phonon calculations

Build a phonon calculation from a structure file and a pretrained potential,
then generate ready-to-edit inputs for any of the three output paths
(`pip install -e ".[mlip]"`, then build the potential's environment once
with `irma mlip env create <potential>`):

```bash
irma mlip build MgO.cif -o mgo_bundle --potential nequip
irma mlip emit mgo_bundle --to endf,spectra --mat 'Mg=44' --mat 'O=48'
irma mlip validate mgo_bundle
# each --mat assigns the ENDF material number, the integer that labels
# that species' evaluation in a library
```

See the manual's *MLIP phonon calculations* page for the potential support
matrix, model-selection syntax, and the per-potential environment
mechanism.

### Neutron-scattering spectra

The forward model has its own CLI (`pip install -e ".[spectra]"` first):

```bash
# VISION spectrum from a phonopy calculation (inelastic mode 1 or 2).
# --scatterer decodes as: symbol, bound cross section [b], atomic weight
# ratio, b_coh [fm], sigma_inc [b]. The CLI does not look these up, so
# pass all five; the GUI fills them from IRMA's nuclear-data table:
python -m irma spectra vision --phonopy-yaml phonopy.yaml --inelastic-mode 2 \
    --scatterer "C,5.551,11.898,6.646,0.001" --temperature 300 -o vision.csv

# direct-geometry spectrum (e.g. ARCS) with the automatic chopper resolution
# (keep --e-max below Ei -- it is the downscatter window):
python -m irma spectra direct --ei 250 --e-max 240 --angles 5:60:5 \
    --resolution-model chopper --chopper-instrument ARCS \
    --chopper-package ARCS-700-1.5-AST --chopper-frequency 600 \
    --phonopy-yaml phonopy.yaml --scatterer "C,5.551,11.898,6.646,0.001" -o arcs.csv

# any configuration, including the DOS-driven mode 0, from a config file:
python -m irma spectra run config.yaml -o spectrum.csv
```

See the manual's [DOS-based spectra (mode 0)](https://ramic-k.github.io/IRMA/spectra-mode0/) page and
`python -m irma spectra --help` for the full surface, and `examples/` for
ready-to-run configurations.

### NCrystal pack export

The exporter that writes per-temperature `.irmapack` files (plus a loadable
`.ncmat`) for the in-repo C++ NCrystal plugin is available both ways:

```bash
irma ncrystal -o outdir config.yaml          # subcommand form
python -m irma.ncrystal -o outdir config.yaml   # module form
```

See the manual's [NCrystal plugin](https://ramic-k.github.io/IRMA/ncrystal-plugin/) page for the
config shape and the plugin workflow.

### Example input files (committed and validated)

Real, validated input files ship with the repository:

- **Classic kernels** (`iel=1/4/6` built-in coherent elastic, `iel=0` with a
  free-gas secondary scatterer, multi-temperature with negative-temperature
  reuse): `tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/*.input`.
  The input files ship in the source distribution; in the repository each
  sits next to the NJOY LEAPR input file it was derived from and the
  reference tape it reproduces (the multi-MB companions are
  repository-only).
- **Mode 2** (`iel=10`, phonopy-backed exact coherent one-phonon +
  incoherent-approximation multiphonon):
  `examples/tsl/graphite_mode2.input`, a production-style input file with
  auto-sized multiphonon order and the documented sampling standard; see
  `examples/tsl/README.md` for the run command. (The
  `tests/mode2_euphonic_n1_validation/*/irma_mode2_n1.input.template` files
  are the cross-code validation harness: they deliberately compute only
  the one-phonon term and are not production tapes.)

```bash
python -m irma tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/tsl-crystalline-graphite.input graphite.endf
```

NJOY job streams are tolerated on input (a `leapr ... stop` block is
located automatically), and
`tests/native_LEAPR_NJOY_ENDF_validation/leapr_to_irma_input.py` converts
an NJOY LEAPR input file into a standalone IRMA input file.

## Automatic grid generation

IRMA can automatically generate alpha and beta grids optimized for the
phonon spectrum of your material, in the GUI (the Grids part of the ENDF form) or via the
`irma.core.grids` functions from Python; the GUI writes the resulting
explicit Cards 7-9 into the input file (there is no auto-grid flag in the card format). The beta grid consists of three regions:

1. **Lower logarithmic**: Captures the thermal (small energy transfer) region
   with logarithmic spacing. Control with `N lower`.
2. **Linear phonon**: Linearly spaced points covering the phonon spectrum
   energy range [0, freq_max]. Control with `N phonon`.
3. **Upper logarithmic**: Extends to high energy transfers with logarithmic
   spacing. Control with `N upper` and `Beta max`.

The alpha grid is linear in momentum transfer Q (alpha quadratic): a
dQ = 0.05 1/Angstrom segment up to Q = 12 covers the thermal scattering
window, and a logarithmic tail extends to the beta grid's kinematic
reach (alpha_max = 4*beta_max/A). The thermal-energy cross section is
controlled by S(α,β) at small alpha, where this layout resolves
the upscatter windows that an alpha grid built by mirroring the beta grid
through the recoil relation under-samples.

## Input file reference

The IRMA input file format is NJOY free-format (values separated by spaces,
each card terminated by `/`) and follows the LEAPR card structure. Malformed
input files are rejected with the offending card, the expected and found
values, and the input line number. The card-by-card reference, including
the `iel=10` cards (Cards 6b-6g), is the manual's
[Input file reference](https://ramic-k.github.io/IRMA/input-reference/)
page; the [Scattering modes](https://ramic-k.github.io/IRMA/modes/) page
explains `elastic_mode` and `inelastic_mode`.

## AI-assistant skill

`skills/irma/` ships an opt-in skill for AI coding agents (Claude
Code, Codex, Gemini CLI, and compatible tools) that teaches an agent
to drive IRMA calculations and answer physics questions from the code.
See `skills/README.md` for one-line installation.

## Feedback and contributions

Bug reports, questions, and improvement ideas are all welcome in the
[issue tracker](https://github.com/ramic-k/IRMA/issues), and so are
pull requests. IRMA is research software: the physics is validated
against NJOY, Euphonic, OCLIMAX, and measured data (see the validation
record in the manual), and if a result still looks wrong, please report
it. For a bug, the most useful report is the input file or configuration that
reproduces it plus the IRMA version (`irma --version`).
Development setup, the test policy, and the refactoring conventions are
in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing IRMA

IRMA is registered in DOE CODE under DOI
[10.11578/dc.20260803.1](https://doi.org/10.11578/dc.20260803.1), record
[186978](https://www.osti.gov/doecode/biblio/186978). Please cite that DOI
in published work:

> K. Ramic, *IRMA — (In)elastic Representation of Materials As S(α,β)
> evaluations*, Oak Ridge National Laboratory (2026).
> DOI: 10.11578/dc.20260803.1

`CITATION.cff` carries the same metadata in machine-readable form, which is
what GitHub's "Cite this repository" button reads.

## License

IRMA is distributed under BSD 3-Clause (project code) with Apache-2.0
derived portions (SPDX `BSD-3-Clause AND Apache-2.0`; see `LICENSE` and
`LICENSES/Apache-2.0.txt`). The classic kernels are in part derived from
NJOY2016's LEAPR module; the LANL BSD 3-Clause notice is retained in
`THIRD_PARTY_NOTICES.md` as required, alongside the NCrystal (Apache-2.0)
notice for the generalized coherent-elastic algorithms, the
**ncplugin-CrysXT (Apache-2.0)** notice for the crystalline-extinction
models ported to `irma/core/extinction.py`, and the PyChop attribution for
the chopper resolution model (an independent BSD reimplementation; no
PyChop source code is included; see `THIRD_PARTY_NOTICES.md` for the full
provenance statements).

## Acknowledgments

IRMA builds upon algorithms and methods from the following projects:

- **NJOY2016** (BSD-3-Clause, Los Alamos National Laboratory): the phonon
  expansion algorithms (contin, convol, discre, trans, coldh, coher) are
  based on the LEAPR module of NJOY2016.
  *R.E. MacFarlane, D.W. Muir, R.M. Boicourt, A.C. Kahler III, "The NJOY
  Nuclear Data Processing System, Version 2016", LA-UR-17-20093 (2016).*

- **phonopy** (BSD-3-Clause, A. Togo): provides force constants and phonon
  density of states used as input.
  *A. Togo, "First-principles Phonon Calculations with Phonopy and Phono3py",
  J. Phys. Soc. Jpn. 92 (2023) 012001.*

- **INSPIRED** (MIT license, Oak Ridge National Laboratory): the MLIP phonon
  front end generalizes the pretrained machine-learned-potential workflow
  that INSPIRED pioneered for inelastic neutron scattering, and inherits
  several of its conventions (the default mesh-density rule, per-potential
  dtype usage).
  *Han, Savici, Li, and Cheng, Comput. Phys. Commun. 304 (2024) 109288.*

- The generalized elastic scattering implementation follows the formalism of:
  *K. Ramic, J. I. Marquez Damian, et al., "NJOY+NCrystal: An open-source
  tool for creating thermal neutron scattering libraries with mixed elastic
  support", NIM-A 1027 (2022) 166227.*

- **NCrystal** (Apache-2.0, T. Kittelmann and X.-X. Cai;
  https://github.com/mctools/ncrystal): the generalized
  coherent-elastic (Bragg edge) algorithms are based on NCrystal's elastic
  models (the Apache-2.0-derived portions are noticed in
  `THIRD_PARTY_NOTICES.md`), and NCrystal is the transport host of both
  in-repo plugins: `ncrystal_plugin_IRMA` samples the exported scattering
  kernels through NCrystal's standard machinery, and
  `ncrystal_plugin_ENDFTSL` brings ENDF thermal scattering files into any
  NCrystal-aware code.
  *X.-X. Cai and T. Kittelmann, "NCrystal: A library for thermal neutron
  transport", Comput. Phys. Commun. 246 (2020) 106851.*
  *T. Kittelmann and X.-X. Cai, "Elastic neutron scattering models for
  NCrystal", Comput. Phys. Commun. 267 (2021) 108082.*
  *X.-X. Cai, T. Kittelmann, E. Klinkby, and J. I. Marquez Damian,
  "Rejection-based sampling of inelastic neutron scattering",
  J. Comput. Phys. 380 (2019) 400-407.*

- **Mantid PyChop** (GPL-3.0+, the Mantid project): the direct-geometry
  chopper resolution model is an independent BSD reimplementation written from
  the published literature (Ikeda & Carpenter, NIM A 239 (1985) 536;
  Marseguerra & Pauli, NIM 4 (1959) 140; Violini et al., NIM A 736 (2014)
  31; Windsor, *Pulsed Neutron Scattering*, 1981); no PyChop source is
  included or ported. PyChop is gratefully acknowledged as the black-box
  validation reference (IRMA reproduces it to ~1% on all eight supported
  spectrometers) and as the collected source of the factual instrument
  parameters (flight paths, chopper geometry, moderator pulse data).
  See `THIRD_PARTY_NOTICES.md`.

IRMA's development also relied on two AI coding assistants, Anthropic's
Claude Code and OpenAI's Codex, which were used throughout for
implementation, testing, code review, and documentation (the manual and
this README were drafted with Claude Code). The physics decisions, the
validation record, the documentation, and the released code were directed
and reviewed by the author.

IRMA development was supported by the DOE/NRC Collaboration for
Criticality Safety Support for Commercial-Scale HALEU Fuel Cycles and
Transportation. This work was supported by the Nuclear Criticality Safety
Program, funded and managed by the National Nuclear Security
Administration for the Department of Energy. Oak Ridge National
Laboratory is managed by UT-Battelle, LLC, for the U.S. Department of
Energy under contract DE-AC05-00OR22725.
