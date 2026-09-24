# Installation

This page covers installing IRMA from PyPI, from conda-forge, or from source; the optional extras; the separately built C++ NCrystal plugins; and checking the install.

IRMA is a pure-Python package. The core needs NumPy, `endf-parserpy` and `threadpoolctl`, and the GUI uses `tkinter` from the standard library. Three extras add optional workflows: `[phonopy]` (the phonopy-backed inelastic modes 1/2), `[spectra]` (the neutron scattering forward model) and `[mlip]` (the MLIP phonon front end). The NCrystal exporter (`irma ncrystal`) needs `[phonopy,spectra]`.

## Requirements

| Component | Requirement | When you need it |
|-----------|-------------|------------------|
| Python | >= 3.11 | Always |
| NumPy | >= 2.0 | Always (installed automatically) |
| `endf-parserpy` | >= 0.12, < 0.18 | Always (installed automatically; version-capped below 0.18, the next untested minor release) |
| `threadpoolctl` | >= 3.0 | Always (installed automatically; pins native BLAS/OpenMP threads for the mode-1/2 workers) |
| `tkinter` | standard library | Only for the graphical interface |
| `phonopy` | >= 4.2 | For `iel=10` with `inelastic_mode=1` or `2`, and for `irma.spectra` inelastic modes 1/2 or `dos_source: phonopy`, the spectra-config option that takes the DOS from the phonopy calculation (4.x is the only tested major) |
| `scipy` + `PyYAML` | >= 1.10 / >= 6 | Only for the `irma.spectra` forward model (the `[spectra]` extra) |

Most of this table is installed for you. The optional dependency that decides which workflows you can run is `phonopy`. It powers the eigenvector-based workflows: on the ENDF side, the phonopy-backed inelastic modes (`iel=10` combined with `inelastic_mode=1`, the directional incoherent approximation, or `inelastic_mode=2`, the exact coherent one-phonon), and on the spectra side, the inelastic modes 1/2 and the `dos_source: phonopy` option of mode 0. The NCrystal exporter uses it too, because it exports modes 1/2. Everything else runs without it: all the classic LEAPR kernels, `iel=0`–`6`, `iel=10` with `inelastic_mode=0`, and spectra mode 0 from DOS files. The `euphonic` package is used only by offline validation harnesses and is never imported by the engine, so you do not need it for normal use.

## Install (released package)

From PyPI the extras are opt-in: `pip install irma` is the core, and `pip install "irma[phonopy]"`, `"irma[spectra]"`, and `"irma[mlip]"` add the pieces described below. One command gets everything:

```bash
pip install "irma[phonopy,spectra,mlip]"
```

On conda-forge the package is named `irma-sqw`. bioconda already ships an unrelated `irma`, the CDC influenza assembler, and the two channels are routinely enabled together, so the conda package is named for the S(Q,ω) that underpins every IRMA calculation. Only the conda package name differs: the import and the command stay `irma`. That package is all of IRMA in one install, the core plus the dependencies of every extra (phonopy, scipy, PyYAML, ase), so the ENDF generator, the spectra forward model, the NCrystal exporter, and the MLIP phonon front end all work from it:

```bash
conda create -n irma -c conda-forge irma-sqw
conda activate irma
```

With either package manager, the pretrained potentials still get their own environments (`irma mlip env create <potential>`, described under the `[mlip]` extra), and the C++ NCrystal plugins are still built separately (see [NCrystal plugins](#ncrystal-plugins-built-separately)).

## Install from source

For development, or to run the committed examples and the test suites, clone the repository (`git clone https://github.com/ramic-k/IRMA.git`) and, from the source directory, install the core package in editable (development) mode:

```bash
pip install -e .
```

This gives you the classic kernels (`iel=0`–`6`) and `iel=10` with `inelastic_mode=0`. NumPy and `endf-parserpy` are pulled in automatically.

For a regular (non-editable) install:

```bash
pip install .
```

### The ENDF writer backend

IRMA writes the output tape through `endf-parserpy`, which ships two interchangeable serializers. By default IRMA uses the **compiled backend** (`EndfParserCpp`): it produces byte-identical tapes and is several times faster than the pure-Python writer; serialization dominates the wall time of classic runs. The compiled backend is included in the official `endf-parserpy` binary wheels, so a normal `pip install` has it. On a source-only `endf-parserpy` build where the compiled module is unavailable, IRMA uses the pure-Python writer; the output is identical, just slower.

### Adding the phonopy extra

If you plan to run the phonopy-backed inelastic modes (`iel=10` with `inelastic_mode=1` or `2`), install the `phonopy` extra:

```bash
pip install -e ".[phonopy]"
```

This adds `phonopy>=4.2` on top of the core dependencies and enables the mode-1/2 S(α,β) calculation. Phonopy 4 is the only major version the suite is tested against.

Installing the extra makes the code available but does not supply the physics input: a mode 1/2 input file also needs a phonopy calculation on disk, a `phonopy.yaml` plus force constants (named on `Card 6f`) and the `Card 6g` controls. The force constants are read from the YAML if embedded, otherwise discovered next to it (`force_constants.hdf5`, `FORCE_CONSTANTS`, `FORCE_SETS`, in that order); the working directory is never consulted. See the [input file reference](input-reference.md) for the full `Card 6f`/`Card 6g` layout.

### Adding the spectra extra

If you plan to use the neutron scattering forward model (the `irma spectra` CLI or the GUI's **Neutron Scattering Experiments** tab), install the `spectra` extra:

```bash
pip install -e ".[spectra]"
```

This adds `scipy` (S(Q,E) interpolation) and `PyYAML` (the primary config format). The extras combine: `pip install -e ".[phonopy,spectra,mlip]"` covers every workflow. The spectra inelastic modes 1/2 (and mode 0 with `dos_source: phonopy`) and the NCrystal exporter (`irma ncrystal`, which reads a YAML config) need both `[phonopy]` and `[spectra]`. See [DOS-based spectra (mode 0)](spectra-mode0.md) for what each spectra mode requires.

### Adding the mlip extra

The MLIP front end (`irma mlip`) builds phonon calculations directly from
pretrained machine-learned interatomic potentials (MLIPs):

```bash
pip install -e ".[mlip]"
```

This adds `ase`, `phonopy`, and `PyYAML`, deliberately **not** any
potential package. Each potential brings a heavy, mutually conflicting
dependency stack (different `e3nn` pins, an ABI-locked torch, one
TensorFlow backend), so you install only the ones you use. You can
install a potential's package directly into this environment
(`pip install nequip`, `pip install mattersim`, ...), or keep each
potential in its own environment: `irma mlip env create <potential>`
builds a separate Python environment for that one potential, installs
that potential's packages into it, and registers it, after which
`irma mlip` uses the registered environment automatically:

```bash
irma mlip env create mace
```

The full support matrix (which potential needs which package, Python
floor, license, and known conflicts) lives in
[MLIP phonon calculations](mlip.md).

## NCrystal plugins (built separately)

Installing the `irma` package does not build the two C++ NCrystal plugins. Each is its own scikit-build-core package inside the repository and compiles against an existing NCrystal installation (NCrystal >= 4.3, with cmake, ninja, and scikit-build-core available in the same environment, for example from conda-forge; NCrystal itself installs from conda-forge or per its instructions at <https://github.com/mctools/ncrystal>):

```bash
pip install --no-build-isolation ./ncrystal_plugin_IRMA      # samples IRMA's exported kernels
pip install --no-build-isolation ./ncrystal_plugin_ENDFTSL   # reads ENDF thermal scattering files directly
ncrystal-pluginmanager --test IRMA                           # self-test of the installed plugins
ncrystal-pluginmanager --test ENDFTSL
```

NCrystal's plugin manager discovers both automatically once they are installed; no further configuration is needed. Each plugin's README (`ncrystal_plugin_IRMA/README.md`, `ncrystal_plugin_ENDFTSL/README.md`) documents its build environment and reference tests.

## The GUI and tkinter

IRMA ships a graphical interface built on `tkinter`, which is part of the Python standard library, so no extra GUI framework is required. Launch it either way:

```bash
python -m irma --gui   # module flag
irma-gui               # installed entry point
```

Both open the same window.

`tkinter` ships with most Python distributions, but some minimal builds omit it. If you see `ModuleNotFoundError: No module named 'tkinter'`, install the platform package:

| Platform | Command |
|----------|---------|
| macOS (Homebrew Python) | `brew install python-tk` |
| Ubuntu / Debian | `sudo apt install python3-tk` |
| Fedora | `sudo dnf install python3-tkinter` |
| Arch Linux | `sudo pacman -S tk` |
| Conda | `conda install -c conda-forge tk` |

On macOS, installing the official build from [python.org](https://www.python.org/downloads/) includes `tkinter`. On Windows it is included by default with the python.org installer (keep the "tcl/tk and IDLE" option checked). A virtual environment uses its base interpreter's `tkinter`: if `python -c 'import tkinter'` fails inside the venv, install Tk for the base Python (the python.org build, or `brew install python-tk`) and recreate the venv.

## Verify the install

Confirm the package imports and reports its version:

```bash
python -m irma --version
```

This prints the IRMA version number. You can also invoke the command-line tool directly:

```bash
python -m irma input_file output_file   # module form
irma input_file output_file             # installed console script
```

If the import fails with `ModuleNotFoundError: No module named 'irma'`, the install almost always ran in a different environment than the one you are using now. Re-run `pip install -e .` from the IRMA source directory in the active environment.

## Developer install and running the tests

The editable install above (`pip install -e .`) is already a developer install: you can edit the source and the changes take effect without reinstalling. To run the fast regression suite, which does **not** require `phonopy`:

```bash
python -m pytest
```

Three validation harnesses live under `tests/` and are run manually (they are not part of CI):

| Harness | What it checks | Extra needs |
|---------|----------------|-------------|
| `tests/native_LEAPR_NJOY_ENDF_validation/` | The classic kernels reproduce published ENDF/B-VIII.1 tapes to 7e-5 (graphite, Fe, Al, H in CH2) and freshly generated NJOY2016.78 tapes exactly (liquid CH4, ortho/para-H2, two-pass BeO). Fully self-contained. | None |
| `tests/mode2_euphonic_n1_validation/` | Cross-validates the mode-2 exact one-phonon S(α,β) against Euphonic (graphite, Be, BeO). | The committed Euphonic reference is frozen; regenerating it (only if the phonon calculation changes) requires the `euphonic` package |
| `tests/mode0_validation/` | Compares the mode-0 (DOS) VISION spectrum with mode 1 for graphite. | `phonopy` |

Cold deuterium and the Sköld correction are not in these harnesses: the fast-suite minitapes (`tests/test_coldd_minitape.py` and `tests/test_coldh_skold_minitape.py`) pin them against NJOY byte for byte on small decks.

One last note for long-time users: IRMA was renamed from THAWNE in June 2026, and the package is now `irma`. If you have older scripts, update imports and the CLI/GUI entry points (`python -m irma`, `irma`, `irma-gui`) accordingly.
