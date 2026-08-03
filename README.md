# IRMA

(**I**n)elastic **R**epresentation of **M**aterials **A**s S(α,β) evaluations

IRMA turns one phonon model into three outputs that usually require
three separate tool chains: an evaluated nuclear-data file, predicted
neutron-scattering spectra, and scattering kernels for Monte Carlo
transport. The three outputs draw on a single, consistent description
of the material, so the evaluation, the spectroscopy that can validate
it, and the transport that uses it always agree about the physics.

**Nuclear data.** IRMA writes ENDF-6 File 7 thermal scattering
evaluations on automatically constructed (α, β) grids. This part
reimplements and generalizes NJOY's LEAPR: the classic kernels
reproduce freshly generated NJOY2016 tapes digit for digit and
published reference tapes to about 1e-4, and the generalized paths add
the exact coherent one-phonon term, anisotropic Debye-Waller tensors,
coherent elastic for arbitrary crystals, and a per-species partition
for polyatomic materials. The tapes feed NJOY, AMPX, FUDGE, and every
transport code downstream of them.

**Neutron spectroscopy.** The `irma.spectra` forward model projects
the same physics onto an instrument's kinematics and resolution: INS
spectra for VISION and generic indirect geometries, and 2-D S(Q,E)
powder maps for direct-geometry spectrometers, from a phonopy model or
straight from a phonon DOS. It can be used to predict a proposed
measurement before beam time; in analysis, it supplies the calculated
single-scattering counterpart of a measured spectrum, from the same
material description the evaluation was built from.

**Monte Carlo transport.** The `irma.ncrystal` exporter writes
per-temperature scattering kernels for the companion NCrystal plugin,
so McStas, OpenMC, and other NCrystal-aware codes sample the same
physics. The exported kernels carry the per-site anisotropic
Debye-Waller tensors, keeping directional coherent-elastic physics
that NCrystal's standard scalar treatment does not represent. With the
same physics inside a transport code, an entire beamline becomes a
virtual experiment: IRMA's end-to-end validation ran a custom McStas
implementation of the ARCS spectrometer, assembled from the existing
McVine and McStas models, against measured data.

**From a bare crystal structure.** The `irma mlip` front end builds
the phonon model itself: a structure file and a choice of potential
are enough. Nine pretrained machine-learned interatomic potentials are
supported, on a laptop CPU, with no first-principles calculation; an
approximate phonon model for a new material costs minutes, not a DFT
campaign, and the build emits prefilled inputs for all three outputs.
The result is a good starting point rather than a finished evaluation:
survey-quality physics with every parameter exposed for review. A
converged atomistic calculation enters the same way, as a phonopy
model, when higher fidelity is needed.

Both a graphical interface and a command-line tool drive every path.
Each run evaluates one principal scatterer; polyatomic crystals keep
the full crystal in the phonon model and run each principal species
separately.

## Features

- **Phonon expansion** of incoherent inelastic scattering to arbitrary order
- **Generalized coherent elastic** (Bragg edges) for any crystal structure
- **`inelastic_mode` hybrid MT4 workflows** for noncubic crystals, including
  per-species principal-scatterer evaluations of polyatomic crystals such as
  BeO, with directional Debye-Waller treatment and phonopy-backed
  S(alpha,beta)
- **Automatic alpha/beta grid generation** with user-controllable density
  in logarithmic and linear regions
- **Graphical interface** (tkinter) for easy configuration and execution
- **Command-line interface** for scripting and batch processing
- **ENDF-6 output** (inelastic MF7/MT4, plus elastic MF7/MT2 when an
  elastic option is active) compatible with standard transport codes
- **Neutron-scattering forward model** (`irma.spectra`): instrument-resolved
  INS spectra and 2-D S(Q,E) powder maps for VISION, generic indirect, and
  direct geometries, from a phonopy model (modes 1/2) or straight from a
  phonon DOS (mode 0, no eigenvectors needed), with an automatic chopper
  resolution model validated against Mantid PyChop for eight direct-geometry
  instruments. Run it as `irma spectra ...` or from the GUI's
  **Neutron Scattering Experiments** tab
- **MLIP phonon front end** (`irma mlip`), generalizing the
  machine-learned interatomic potential (MLIP) workflow of ORNL's
  INSPIRED (Han et al., Comput. Phys. Commun. 304, 109288 (2024)): build
  a phonon model for any crystal from a pretrained MLIP (nine backends:
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
- Multi-temperature decks, including the LEAPR negative-temperature
  convention (one phonon spectrum reused across temperatures)
- Optional coherent-elastic **Bragg-edge grouping** (ENDF-102 §7.2.2) for
  materials with very dense high-energy edge structure
- **Validated**: reproduces published ENDF/B-VIII.1 reference tapes to
  7e-5 or better (graphite/Fe/Al/CH2) and freshly generated NJOY2016
  tapes exactly (liquid CH4, ortho/para-H2, ortho/para-D2, and two-pass
  BeO, covering the translational, discrete-oscillator,
  cold-hydrogen/deuterium, Sköld, and mixed-moderator kernels), and
  agrees with Euphonic on the coherent one-phonon law to shared-domain
  integral ratios of 1.00001 (graphite), 1.0002 (beryllium), and 0.9998
  (BeO), with median per-Q differences of 0.18% and 0.005% for graphite
  and beryllium (committed, rerunnable harnesses under `tests/`), and
  matches OCLIMAX on the full S(α,β) to integral ratios of 0.96-0.99
  (graphite, beryllium, BeO)
- **Production defaults**: a setting you omit gets the validation
  campaign's value (sampling, mesh density, grids) on every surface;
  the committed examples carry the labeled quick variants
- Friendly deck validation: errors name the card, the expected/found
  values, and the input line
- Save/load configurations as JSON

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

A conda-forge package is on the way, under the name **`irma-sqw`**:
bioconda already ships an unrelated `irma` (the CDC influenza assembler),
and the two channels are used together, so the conda package is named for
the S(Q,ω) the engine computes. The import and the command stay `irma`.
It installs all of IRMA, core plus the dependencies of every extra
(phonopy, scipy, PyYAML, ase), so the ENDF generator, the spectra forward
model, the NCrystal exporter, and the MLIP phonon front end all work from
one install:

```bash
conda create -n irma -c conda-forge irma-sqw    # pending review, see #34407
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
pip install -e .              # core (classic LEAPR paths, iel=0-6 and iel=10 mode 0)
pip install -e ".[phonopy]"   # + the noncubic inelastic paths (inelastic_mode=1/2)
pip install -e ".[spectra]"   # + the neutron-scattering forward model (irma spectra)
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

Deck-level validation harnesses live under `tests/` and are run manually
(not in CI):

- `tests/native_LEAPR_NJOY_ENDF_validation/` — reproduces NJOY-LEAPR
  reference ENDF tapes (graphite, Fe, Al, polyethylene to ~1e-4; liquid
  methane and ortho-/para-hydrogen exactly, covering trans/discre/coldh/
  skold); fully self-contained.
- `tests/mode2_euphonic_n1_validation/` — cross-validates the mode-2 exact
  one-phonon law against Euphonic (graphite, Be). The frozen Euphonic
  reference is committed; regenerating it (only needed if the phonon model
  changes) requires the `euphonic` package.

## Documentation

The manual lives at https://ramic-k.github.io/IRMA/: installation, a
quickstart, three end-to-end tutorials (one per output), the GUI tour, the full input-card reference, the theory
behind each mode, the validation record, and the MLIP, spectra, and
NCrystal-plugin guides.

The manual and this README were drafted with Anthropic's Claude Code
(the Claude Fable 5 model), working under the author's direction and
review, and they are revised continuously as the project evolves. If a
passage reads oddly, or you find an error, please open an issue;
documentation reports are as welcome as code bugs.

## Usage

### Graphical Interface

```bash
python -m irma --gui
```

This opens the IRMA GUI where you can:

1. Define your crystal structure (lattice parameters, atom types, positions)
2. Set scattering parameters (principal scatterer, cross sections)
3. Configure alpha/beta grids (automatic or manual)
4. Specify phonon data (from phonopy or manual entry)
5. Run the calculation and monitor progress

Configurations can be saved/loaded as JSON files via the File menu.
Existing IRMA/LEAPR input files (`.input` or `.leapr`) can be imported
via File > Import Input File to populate all GUI fields.

### Command Line

```bash
python -m irma input_file output_file
```

The IRMA input file uses a card-based format derived from the LEAPR module
of NJOY. See the input card documentation below for details.

### MLIP phonon models

Build a phonon model from a structure file and a pretrained potential,
then generate ready-to-edit inputs for any of the three output paths
(`pip install -e ".[mlip]"`, then build the potential's environment once
with `irma mlip env create <potential>`):

```bash
irma mlip build MgO.cif -o mgo_bundle --potential nequip
irma mlip emit mgo_bundle --to endf,spectra --mat 'Mg=44' --mat 'O=48'
irma mlip validate mgo_bundle
```

See the manual's *MLIP phonon models* page for the potential support
matrix, model-selection syntax, and the per-potential environment
mechanism.

### Neutron-scattering spectra

The forward model has its own CLI (`pip install -e ".[spectra]"` first):

```bash
# VISION spectrum from a phonopy model (inelastic mode 1 or 2):
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

The exporter that bakes per-temperature `.irmapack` files (plus a loadable
`.ncmat`) for the in-repo C++ NCrystal plugin is available both ways:

```bash
irma ncrystal -o outdir config.yaml          # subcommand form
python -m irma.ncrystal -o outdir config.yaml   # module form (unchanged)
```

See the manual's [NCrystal plugin](https://ramic-k.github.io/IRMA/ncrystal-plugin/) page for the
config shape and the plugin workflow.

### Example decks (committed and validated)

Real, validated decks ship with the repository:

- **Classic paths** (`iel=1/4/6` built-in coherent elastic, `iel=0` with a
  free-gas secondary scatterer, multi-temperature with negative-temperature
  reuse): `tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/*.input`.
  The decks ship in the source distribution; in the repository each sits
  next to the NJOY LEAPR deck it was derived from and the reference tape
  it reproduces (the multi-MB companions are repository-only).
- **Noncubic mode 2** (`iel=10`, phonopy-backed exact coherent one-phonon +
  incoherent-approximation multiphonon):
  `examples/tsl/graphite_mode2.input`, a production-style deck with
  auto-sized multiphonon order and the documented sampling standard; see
  `examples/tsl/README.md` for the run command. (The
  `tests/mode2_euphonic_n1_validation/*/irma_mode2_n1.input.template` decks
  are the cross-code validation harness: they deliberately compute the
  one-phonon term ONLY and are not production tapes.)

```bash
python -m irma tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/tsl-crystalline-graphite.input graphite.endf
```

NJOY job streams are tolerated on input (a `leapr ... stop` block is
located automatically), and
`tests/native_LEAPR_NJOY_ENDF_validation/leapr_to_irma_input.py` converts
an NJOY LEAPR deck into a standalone IRMA deck.

## Automatic Grid Generation

IRMA can automatically generate alpha and beta grids optimized for the
phonon spectrum of your material, in the GUI (the Grids part of the ENDF form) or via the
`irma.core.grids` functions from Python; the GUI writes the resulting
explicit Cards 7-9 into the deck (there is no deck-level auto-grid flag). The beta grid consists of three regions:

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
controlled by S(alpha, beta) at small alpha, where this layout resolves
the upscatter windows that a beta-mirrored (recoil-relation) alpha grid
under-samples.

## Input Cards Reference

The IRMA deck format is NJOY free-format (values separated by spaces, each
card terminated by `/`) and follows the LEAPR card structure. Malformed
decks are rejected with the offending card, the expected/found values, and
the input line number.

### Control cards (every deck)

| Card | Fields | Notes |
|------|--------|-------|
| 1 | `nout` | Output unit, kept for LEAPR compatibility (IRMA writes the file named on the command line) |
| 2 | `'title'` | Quoted title string |
| 3 | `ntempr iprint nphon` | Temperature count, print level, phonon-expansion order (`nphon` is also the multiphonon maximum order for `inelastic_mode=1/2`) |
| 4 | `mat za isabt ilog smin` | ENDF MAT/ZA, S(α,−β) output flag, log-storage flag, minimum stored S. For `iel=10` the ZA must encode the physical nuclide (1000·Z+A) so the principal scatterer can be matched to a Card 6d atom |
| 5 | `awr spr npr iel ncold nsk` | Principal scatterer: mass ratio, free-atom cross section, atom count (≥ 1), elastic option, cold-hydrogen option (0–4), pair-correlation option (0 = none, 1 = Vineyard, 2 = Sköld — only Sköld modifies the law). `iel`: 0 = none/incoherent, 1–6 = built-in coherent elastic (graphite, Be, BeO, Al, Pb, Fe), 10 = generalized (any crystal, Cards 6b–6g follow). `ncold`/`nsk` are not available with `inelastic_mode=1/2` |
| 6 | `nss b7 aws sps mss` | Secondary scatterer: count (0 or 1), type (`b7`: 0 = SCT — a second full temperature block follows the principal's, 1 = free gas, 2 = diffusion), mass ratio (> 0), cross section (> 0), atom count (≥ 1). Not available with `inelastic_mode=1/2`. The GUI supports all three `b7` models; for the two-pass case it authors the secondary's phonon model with the shared-spectrum convention (first temperature positive, the rest negative), and `ncold`/`nsk` combined with two-pass remains deck-file-only |
| 7 | `nalpha nbeta lat` | Grid sizes; `lat=1` = grids given at 0.0253 eV reference temperature |
| 8 | α grid | `nalpha` values, strictly increasing, > 0 (may span lines; one `/` ends the card) |
| 9 | β grid | `nbeta` values, strictly increasing, from ≥ 0 |

### Generalized-elastic cards (only when `iel=10`, between Cards 6 and 7)

| Card | Fields | Notes |
|------|--------|-------|
| 6b | `elastic_mode nat nspec inelastic_mode [bins_per_decade] [threshold_eV]` | `elastic_mode`: 1 = SEF (single-channel elastic format, designated-coherent atom), 2 = MEF (LTHR=3 mixed). `inelastic_mode`: see below. Optional fields 5–6 enable ENDF-102 §7.2.2 Bragg-edge grouping: above `threshold_eV` (default 1 eV) the dense edge steps are merged into `bins_per_decade` log-uniform bins per decade with structure-factor-weighted placement; cumulative S and the total cross section are preserved. 0/absent = off |
| 6c | `a b c alpha beta gamma` | Lattice constants [Å] and angles [°] |
| 6d | `Z A awr b_coh sigma_inc npos` + `npos` fractional positions | Repeated `nat` times. `b_coh` in fm, `sigma_inc` in barns. For `inelastic_mode=1/2` the positions must match the phonopy primitive cell |
| 6e | `Z A delta ni` + `ni` rho values | Repeated `nspec` times (mode 0 only: per-species Debye-Waller spectra). For modes 1/2 `nspec` must be 0 and Card 6e omitted (deck error otherwise) |
| 6f | `'phonopy.yaml path'` then `mesh_nx mesh_ny mesh_nz ncpu use_born` (then BORN path if `use_born=1`) | Modes 1/2 only. Force constants are read from the yaml itself if embedded, otherwise discovered next to it (`force_constants.hdf5`, `FORCE_CONSTANTS`, `FORCE_SETS`, in that order; nothing found = error — the working directory is never consulted). `ncpu` = worker-process count for the in-process SAB calculation; `use_born=1` applies the non-analytical-term correction (LO-TO splitting) to both the MT2 directional Debye-Waller factors and every MT4 mode sum (unreadable BORN file = error). With `use_born=0`, NAC embedded in the named phonopy.yaml is still honored, but a `BORN` file in the working directory is never auto-read |
| 6g | `ndir mpdir [auto_order]` | Modes 1/2 only. `ndir` = coherent powder-average directions (golden-spiral quadrature). `mpdir` = multiphonon powder-average directions (converged by ~50–100; production default 1000, the validation-campaign sampling — cost is linear). `auto_order`: 0 = honor Card 3 `nphon` verbatim (default), 1 = auto-size the multiphonon order. The incoherent powder average is always the exact numerical orientational average. Transfers beyond the tabulated law are covered downstream by THERMR's short-collision-time extension (driven by the tape's effective temperature) |

### Temperature cards (Card 10 onward)

For each of the `ntempr` temperatures, one temperature card, then (for the
classic paths and `inelastic_mode=0`) the scattering-law detail block:

| Card | Fields | Notes |
|------|--------|-------|
| 10 | `T` | Temperature [K]. **A negative value reuses the previous temperature's entire detail block** (the LEAPR shared-DOS convention: one spectrum, many temperatures) — no detail cards follow it |
| 11 | `delta ni` | Continuous-spectrum energy spacing [eV] and point count |
| 12 | `rho(1..ni)` | Phonon density of states on the equidistant grid |
| 13 | `twt c tbeta` | Translational weight, diffusion constant (0 = free gas), continuous weight |
| 14 | `nd` | Number of discrete oscillators |
| 15/16 | oscillator energies [eV] / weights | Only if `nd > 0` |
| 17/18 | `nka dka` / `S(κ)` values | Only if `nsk > 0` or `ncold > 0` |
| 19 | `cfrac` | Only if `nsk > 0` |

`inelastic_mode=1/2` decks supply **only** the temperature cards: the
inelastic law comes from the phonopy model, and the detail block is not read.

After the last temperature block: optional quoted comment cards (one per
line) become the ENDF MF1/MT451 description; a bare `/` ends the section.

### The `inelastic_mode` workflows (Card 6b, `iel=10`)

- `inelastic_mode=0` — isotropic Debye-Waller + cubic inelastic from the
  deck's tabulated DOS (no phonopy needed)
- `inelastic_mode=1` — directional DW for the coherent elastic + in-process
  noncubic S(α,β): incoherent-approximation one-phonon plus
  incoherent-approximation multiphonons
- `inelastic_mode=2` — directional DW + in-process noncubic S(α,β): **exact**
  one-phonon (coherent + incoherent) plus incoherent-approximation
  multiphonons

For the noncubic modes: put the full primitive cell in Card 6d (all atom
types of a mixed material); keep one principal scatterer per deck (Card 4
ZA / Card 5 select it — for BeO run two decks); the exported MT4 is the
principal-scatterer law. Mode 1 keeps principal self-terms only; mode 2
assigns cross-group coherent interference to the principal with
coherent-strength weighting. Parallelism is controlled by Card 6f `ncpu`
(native BLAS/OpenMP threads are intentionally pinned to 1 — measured to be
the optimum; see CONTRIBUTING.md).

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
it. For a bug, the most useful report is the deck or configuration that
reproduces it plus the IRMA version (`irma --version`).
Development setup, the test policy, and the refactoring conventions are
in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing IRMA

IRMA is registered in DOE CODE under DOI
[10.11578/dc.20260803.1](https://doi.org/10.11578/dc.20260803.1), record
[186978](https://www.osti.gov/doecode/biblio/186978). Please cite that DOI
in published work:

> K. Ramic, *IRMA — (In)elastic Representation of Materials As S(α,β)
> evaluations*, version 1.0.0, Oak Ridge National Laboratory (2026).
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
the chopper resolution model (an independent BSD reimplementation — no
PyChop source code is included; see `THIRD_PARTY_NOTICES.md` for the full
provenance statements).

## Acknowledgments

IRMA builds upon algorithms and methods from the following projects:

- **NJOY2016** (BSD-3-Clause, Los Alamos National Laboratory) — The phonon
  expansion algorithms (contin, convol, discre, trans, coldh, coher) are
  based on the LEAPR module of NJOY2016.
  *R.E. MacFarlane, D.W. Muir, R.M. Boicourt, A.C. Kahler III, "The NJOY
  Nuclear Data Processing System, Version 2016", LA-UR-17-20093 (2016).*

- **phonopy** (BSD-3-Clause, A. Togo) — Provides force constants and phonon
  density of states used as input.
  *A. Togo, "First-principles Phonon Calculations with Phonopy and Phono3py",
  J. Phys. Soc. Jpn. 92 (2023) 012001.*

- **INSPIRED** (MIT license, Oak Ridge National Laboratory) — The MLIP phonon
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
  https://github.com/mctools/ncrystal) — The generalized
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

- **Mantid PyChop** (GPL-3.0+, the Mantid project) — The direct-geometry
  chopper resolution model is an independent BSD reimplementation written from
  the published literature (Ikeda & Carpenter, NIM A 239 (1985) 536;
  Marseguerra & Pauli, NIM 4 (1959) 140; Violini et al., NIM A 736 (2014)
  31; Windsor, *Pulsed Neutron Scattering*, 1981) — no PyChop source is
  included or ported. PyChop is gratefully acknowledged as the black-box
  validation reference (IRMA reproduces it to ~1% on all eight supported
  spectrometers) and as the collected source of the factual instrument
  parameters (flight paths, chopper geometry, moderator pulse data).
  See `THIRD_PARTY_NOTICES.md`.

IRMA's development also relied on two AI coding assistants, Anthropic's
Claude Code and OpenAI's Codex, which were used throughout for
implementation, testing, code review, and documentation. The physics
decisions, the validation record, and the released code were directed
and reviewed by the author.

IRMA development was supported by the DOE/NRC Collaboration for
Criticality Safety Support for Commercial-Scale HALEU Fuel Cycles and
Transportation. This work was supported by the Nuclear Criticality Safety
Program, funded and managed by the National Nuclear Security
Administration for the Department of Energy. Oak Ridge National
Laboratory is managed by UT-Battelle, LLC, for the U.S. Department of
Energy under contract DE-AC05-00OR22725.
